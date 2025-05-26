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
            return int(np.sign(vals[0]))
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

    results = []
    draw_counts = []
    for rule_agent in opponents:
        wins = 0
        draws = 0
        n = env.n
        for ep in range(episodes):
            obs = env.reset()
            # Alternate markers: even episodes agent=X(+1), odd = O(‑1)
            agent_marker = 1 if ep % 2 == 0 else -1
            done = False
            steps = 0

            while (not done) and steps < max_ep_steps:
                if env.current_player == agent_marker:
                    action = agent.act(obs, greedy=True)
                else:
                    action = rule_agent.act(obs)
                obs, _, done, _ = env.step(action)
                steps += 1

            winner = _winner_from_board(obs["board"], n)
            if winner == agent_marker:
                wins += 1
            elif winner == 0:
                draws += 1

        results.append(100.0 * wins / (episodes - draws))  # win rate in percent
        draw_counts.append(draws)

    return results, draw_counts


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
    state_dim = env.num_cells + 2 * env.disappear_turn
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
    )

    model_path = Path(args.save_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    opponent = ComplexRuleBasedAgent(env.action_space)

    pbar = trange(1, args.episodes + 1, desc="Training", dynamic_ncols=True)
    for episode in pbar:
        obs = env.reset()
        state = flatten_observation(obs)
        ep_reward = 0.0
        done = False
        agent_marker = 1 if episode % 2 == 0 else -1

        steps = 0
        while not done and steps < args.max_ep_steps:
            if env.current_player == agent_marker:
                action = agent.act(obs)
                try:
                    next_obs, reward, done, _ = env.step(action)
                except ValueError:
                    print("DDQN is agent" + str(agent_marker))
                    exit(1)
                reward *= agent_marker  # If our player is -1, flip the reward sign
                assert reward >= 0 # reward is always non‑negative
                next_state = flatten_observation(next_obs)
                agent.store(state, action, reward, next_state, done)
                agent.update()
                state = next_state
                obs = next_obs
                ep_reward += reward
            else:
                action = opponent.act(obs)
                obs, reward, done, _ = env.step(action)
                state = flatten_observation(obs)
            steps += 1

        if episode % args.log_every == 0:
            eps = agent._epsilon()
            logging.info("Ep %6d | Reward %.1f | Epsilon %.3f", episode, ep_reward, eps)

        if episode % args.eval_every == 0:
            win_rates, draw_counts = evaluate(agent, env, args.eval_episodes, args.max_ep_steps)
            logging.info(
                    """Evaluation after %d eps 
                    → win‑rate %.1f%% vs ComplexRuleBasedAgent with %d draws
                    → win‑rate %.1f%% vs ModerateRuleBasedAgent with %d draws
                    → win‑rate %.1f%% vs SimpleRuleBasedAgent with %d draws
                    → win‑rate %.1f%% vs RandomAgent with %d draws
                    """,
                episode,
                win_rates[0],
                draw_counts[0],
                win_rates[1],
                draw_counts[1],
                win_rates[2],
                draw_counts[2],
                win_rates[3],
                draw_counts[3],
            )

        if episode % args.save_every == 0:
            agent.save(model_path)

        # Update progress‑bar postfix
        pbar.set_postfix({"last_R": ep_reward, "ε": agent._epsilon()})

    # Final save
    agent.save(model_path)
    env.close()


# ----------------------------------------------------------------------------- #


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Double‑DQN on Vanishing Tic Tac Toe")
    parser.add_argument("--episodes", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--target-update-freq", type=int, default=5_000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--log-every", type=int, default=1_000)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=250)
    parser.add_argument("--save-path", type=str, default="models/ddqn_vttt.pth")
    parser.add_argument("--log-path", type=str, default="training.log")
    parser.add_argument("--max-ep-steps", type=int, default=2000)

    train(parser.parse_args())
