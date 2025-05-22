import numpy as np
from vanishing_tictactoe import VanishingTicTacToeEnv
from random_agent import RandomAgent
from tqdm import tqdm
from simplerulebased_agent import SimpleRuleBasedAgent
from moderaterulebased_agent import ModerateRuleBasedAgent
from complexrulebased_agent import ComplexRuleBasedAgent

def simulate_game(env, agent_x, agent_o, verbose=False):
    obs = env.reset()
    done = False
    steps = 0

    while not done:
        current_player = env.current_player
        agent = agent_x if current_player == 1 else agent_o
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        steps += 1

        if verbose:
            print(f"{'X' if current_player == 1 else 'O'} plays: {action}")
            env.render()

    if info.get("invalid", False):
        return -current_player, steps
    elif reward == 1.0:
        return current_player, steps
    elif reward == 0.5:
        return 0, steps
    else:
        return -current_player, steps

if __name__ == "__main__":
    num_games = 500
    board_sizes = [3, 4, 5, 6]

    for board_size in board_sizes:
        env = VanishingTicTacToeEnv(board_size=board_size)
        agent_x = ModerateRuleBasedAgent(env.action_space)
        agent_o = SimpleRuleBasedAgent(env.action_space)

        results = {1: 0, -1: 0}
        total_steps = 0

        for _ in tqdm(range(num_games), desc=f'Running games in board_size {board_size}'):
            outcome, steps = simulate_game(env, agent_x, agent_o)
            total_steps += steps
            if outcome == 0:
                raise ValueError("Draw! (impossible)")
            results[outcome] += 1

        average_steps = total_steps / num_games

        print(f"\nBoard size: {board_size}x{board_size}")
        print(f"  X wins: {results[1]}  O wins: {results[-1]}")
        print(f"  Average steps per game: {average_steps:.2f}")

    num_games = 500
    board_size = 3

    env = VanishingTicTacToeEnv(board_size=board_size)

    # these agents can be changed to different agents in the future!
    agent_x = ComplexRuleBasedAgent(env.action_space)
    agent_o = ModerateRuleBasedAgent(env.action_space)


    results = {1: 0, -1: 0}  # X win, O win
    total_steps = 0

    for _ in tqdm(range(num_games), desc=f'Running games in board_size {board_size}'):
        outcome, steps = simulate_game(env, agent_x, agent_o)
        total_steps += steps
        if outcome == 0:
            raise ValueError("Draw! (impossible)")
        results[outcome] += 1

    average_steps = total_steps / num_games

    print(f"\nResults over {num_games} games (Board size: {board_size}x{board_size}):")
    print(f"Player X Agent: {agent_x.__class__.__name__}")
    print(f"Player O Agent: {agent_o.__class__.__name__}")

    print(f"  X wins:  {results[1]}")
    print(f"  O wins:  {results[-1]}")
    print(f"  Average steps per game: {average_steps:.2f}")