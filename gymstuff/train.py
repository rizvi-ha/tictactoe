import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from tqdm.auto import trange

from vanishing_tictactoe import VanishingTicTacToeEnv
from ddqn_agent import DDQNAgent

from random_agent import RandomAgent
from simplerulebased_agent import SimpleRuleBasedAgent
from moderaterulebased_agent import ModerateRuleBasedAgent
from complexrulebased_agent import ComplexRuleBasedAgent

# ----------------------------------------------------------------------------- #

def flatten_observation(obs: dict) -> np.ndarray:
    """Flat 1‑D float32 vector: board + X history + O history."""
    return np.concatenate([
        obs["board"].astype(np.float32),
        obs["history_x"].astype(np.float32),
        obs["history_o"].astype(np.float32),
        obs["current_player"].astype(np.float32),
    ])


def init_logger(log_path: Path):
    handlers = [
        logging.FileHandler(log_path, mode="w"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
        datefmt="%H:%M:%S",
    )


# ----------------------------------------------------------------------------- #
# Evaluation helpers


def _winner_from_board(board: np.ndarray, n: int) -> int:
    """Return +1, ‑1 if there is a winner, else 0."""
    lines = (
        [tuple(r * n + c for c in range(n)) for r in range(n)]
        + [tuple(r * n + c for r in range(n)) for c in range(n)]
        + [tuple(i * n + i for i in range(n))]
        + [tuple(i * n + (n - 1 - i) for i in range(n))]
    )
    for line in lines:
        vals = board[list(line)]
        if abs(vals.sum()) == n:
            assert vals[0] == -1 or vals[0] == 1
            return int(vals[0])
    return 0  # draw or unfinished


@torch.no_grad()
def evaluate(agent: DDQNAgent, env: VanishingTicTacToeEnv, episodes: int = 100, max_ep_steps: int = 2000):
    opponents = [
        RandomAgent(env.action_space),
        SimpleRuleBasedAgent(env.action_space),
        ModerateRuleBasedAgent(env.action_space),
        ComplexRuleBasedAgent(env.action_space),
    ]
    opponents.reverse()

    x_results = []
    x_draw_counts = []
    o_results = []
    o_draw_counts = []

    for rule_agent in opponents:
        x_wins = 0
        o_wins = 0
        x_draws = 0
        o_draws = 0
        x_episodes = 0
        o_episodes = 0
        n = env.n
        for ep in range(episodes):
            obs = env.reset()

            # Alternate markers: even episodes agent=X(+1), odd = O(‑1)
            agent_marker = 1 if ep % 2 == 0 else -1
            if ep % 2 == 0:
                x_episodes += 1
            else:
                o_episodes += 1
                action = rule_agent.act(obs)
                obs, _ , _ , _ = env.step(action)
                
            done = False
            steps = 0

            while (not done) and steps < max_ep_steps:

                if random.random() < 0.03:
                    action = RandomAgent(env.action_space).act(obs) 
                else:
                    action = agent.act(obs, greedy=True)
                obs, _, done, _ = env.step(action)

                if not done:
                    action = rule_agent.act(obs)
                    obs, _, done, _ = env.step(action)

                steps += 1

            winner = _winner_from_board(obs["board"], n)
            if winner == agent_marker:
                if ep % 2 == 0:
                    x_wins += 1
                else:
                    o_wins += 1
            elif winner == 0:
                if ep % 2 == 0:
                    x_draws += 1
                else:
                    o_draws += 1


        if x_episodes != x_draws:
            x_results.append(100.0 * x_wins / (x_episodes - x_draws))  # win rate in percent
        else:
            x_results.append(0)
        x_draw_counts.append(x_draws)

        if o_episodes != o_draws:
            o_results.append(100.0 * o_wins / (o_episodes - o_draws))  # win rate in percent
        else:
            o_results.append(0)
        o_draw_counts.append(o_draws)



    return x_results, x_draw_counts, o_results, o_draw_counts


# ----------------------------------------------------------------------------- #


def train(args):
    # Seed all randomness
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Logger
    log_path = Path(args.log_path)
    init_logger(log_path)

    env = VanishingTicTacToeEnv()
    state_dim = env.num_cells + 2 * env.disappear_turn + 1
    action_dim = env.action_space.n

    device = "cuda" if torch.cuda.is_available() else exit("No GPU available, exiting...")
    agent = DDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        target_update_freq=args.target_update_freq,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        n_step= args.n_step_returns,
    )

    if args.preload:
        logging.info("Loading pre-trained model...")
        agent.load("models/wr_75.pth")

    model_path = Path(args.save_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    opponent = ComplexRuleBasedAgent(env.action_space)
    best_winrate_sum = -1

    pbar = trange(1, args.episodes + 1, desc="Training", dynamic_ncols=True)
    wins = 0
    for episode in pbar:
        obs = env.reset()
        state = flatten_observation(obs)
        ep_reward = 0.0
        done = False
        agent_marker = 1 if episode % 2 == 0 else -1

        steps = 0
        
        opp_choice = random.random()
        if opp_choice < 0.05:
            opponent = RandomAgent(env.action_space)
        elif opp_choice < 0.15:
            opponent = SimpleRuleBasedAgent(env.action_space)
        elif opp_choice < 0.30:
            opponent = ModerateRuleBasedAgent(env.action_space)
        else:
            opponent = ComplexRuleBasedAgent(env.action_space)

        # If we are -1 need one opponent move first
        if agent_marker == -1:
            action = opponent.act(obs)
            obs, _, _, _ = env.step(action)
            state = flatten_observation(obs)

        while not done and steps < args.max_ep_steps:

            # ────────────────────────────────
            # 1.  DDQN AGENT MAKES A MOVE
            # ────────────────────────────────
            action = agent.act(obs)
            mid_obs, r_self, done, _ = env.step(action)      # environment after OUR move
            r_self *= agent_marker                           # +1 if we just won, else 0

            # If the game ended on our move, store and quit
            if done:
                next_state = flatten_observation(mid_obs)
                agent.store(state, action, r_self, next_state, done)
                agent.update()

                ep_reward += r_self
                assert ep_reward > 0
                wins += 1
                break

            # ────────────────────────────────
            # 2.  OPPONENT MAKES A MOVE
            #    (part of env dynamics)
            # ────────────────────────────────
            
            opp_action                = opponent.act(mid_obs)

            next_obs, r_opp, done, _  = env.step(opp_action)
            r_opp *= agent_marker      

            r_final = r_self + r_opp

            next_state = flatten_observation(next_obs)

            # ────────────────────────────────
            # 3.  STORE ONE TRANSITION
            #    (our action → env after opp reply)
            # ────────────────────────────────
            agent.store(state, action, r_final, next_state, done)
            agent.update()

            # bookkeeping
            state = next_state
            obs   = next_obs
            ep_reward += r_final
            assert ep_reward <= 0, f"r_opp {r_opp} r_self {r_self} r_final {r_final}"
            steps += 1

        if episode % args.log_every == 0:
            eps = agent._epsilon()
            logging.info("Ep %6d | Reward %.1f | Epsilon %.3f | Total Wins %.1f | Steps %.1f", episode, ep_reward, eps, wins, steps)
            wins = 0

        if episode % args.eval_every == 0:
            x_win_rates, x_draw_counts, o_win_rates, o_draw_counts = evaluate(agent, env, args.eval_episodes, args.max_ep_steps)
            winrate_sum = sum(x_win_rates) + sum(o_win_rates)
            logging.info(
                    """Evaluation after %d eps 

                    → win‑rate %.1f%% vs ComplexRuleBasedAgent with %d draws as X
                      win-rate %.1f%% vs ComplexRuleBasedAgent with %d draws as O 

                    → win‑rate %.1f%% vs ModerateRuleBasedAgent with %d draws as X
                      win-rate %.1f%% vs ModerateRuleBasedAgent with %d draws as O 

                    → win‑rate %.1f%% vs SimpleRuleBasedAgent with %d draws as X
                      win-rate %.1f%% vs SimpleRuleBasedAgent with %d draws as O 

                    → win‑rate %.1f%% vs RandomAgent with %d draws as X
                      win-rate %.1f%% vs RandomAgent with %d draws as O 

                    Total winrate of %.1f%%

                    """,
                episode,
                x_win_rates[0],
                x_draw_counts[0],
                o_win_rates[0],
                o_draw_counts[0],
                x_win_rates[1],
                x_draw_counts[1],
                o_win_rates[1],
                o_draw_counts[1],
                x_win_rates[2],
                x_draw_counts[2],
                o_win_rates[2],
                o_draw_counts[2],
                x_win_rates[3],
                x_draw_counts[3],
                o_win_rates[3],
                o_draw_counts[3],
                winrate_sum/8.0,

            )

            if winrate_sum > best_winrate_sum:
                best_winrate_sum = winrate_sum
                logging.info("*** Saving new best model... ***")
                agent.save(model_path)
            else:
                logging.info("*** Not improved ***")


        # Update progress‑bar postfix
        pbar.set_postfix({"last_R": ep_reward, "ε": agent._epsilon()})

    env.close()


# ----------------------------------------------------------------------------- #


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Double‑DQN on Vanishing Tic Tac Toe")
    parser.add_argument("--episodes", type=int, default=120_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=13e-5)
    parser.add_argument("--target-update-freq", type=int, default=10_000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=int, default=800_000)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--log-every", type=int, default=2_500)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--save-path", type=str, default="models/ddqn_vttt.pth")
    parser.add_argument("--log-path", type=str, default="training.log")
    parser.add_argument("--max-ep-steps", type=int, default=20)
    parser.add_argument("--n-step-returns", type=int, default=1)
    parser.add_argument("--preload", type=bool, default=True)

    train(parser.parse_args())
