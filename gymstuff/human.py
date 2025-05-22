import numpy as np
from vanishing_tictactoe import VanishingTicTacToeEnv
from moderaterulebased_agent import ModerateRuleBasedAgent
from simplerulebased_agent import SimpleRuleBasedAgent
from random_agent import RandomAgent

DELAY = 0.5
BOARD_SIZE = 3

if __name__ == "__main__":
    side = None
    while side not in ("X", "O"):
        side = input("Choose your side (X/O): ").strip().upper()
    human_player = 1 if side == "X" else -1

    env = VanishingTicTacToeEnv(board_size=BOARD_SIZE)
    agent = ModerateRuleBasedAgent(env.action_space)

    obs = env.reset()
    board = obs["board"]
    done = False

    while not done:
        env.render(delay=DELAY)
        current = env.current_player

        if current == human_player:
            move = None
            prompt = f"Your move (1–{BOARD_SIZE*BOARD_SIZE} or row,col 1–{BOARD_SIZE}): "
            while move is None:
                entry = input(prompt).strip()
                idx = None
                if entry.isdigit():
                    idx = int(entry) - 1
                elif "," in entry:
                    try:
                        r, c = map(int, entry.split(","))
                        idx = (r - 1) * BOARD_SIZE + (c - 1)
                    except:
                        idx = None
                if idx is None or not (0 <= idx < env.num_cells) or board[idx] != 0:
                    print("Invalid choice.")
                    idx = None
                else:
                    move = idx
        else:
            move = agent.act(obs)
            print(f"Agent plays: row {move//BOARD_SIZE+1}, col {move%BOARD_SIZE+1}")

        obs, reward, done, info = env.step(move)
        board = obs["board"]
        if info.get("invalid", False):
            print("Invalid move! Game over.")
            break

    env.render()
    if not info.get("invalid", False):
        if reward == 1.0:
            print("X wins!" if current == 1 else "O wins!")
        else:
            print("Draw!")