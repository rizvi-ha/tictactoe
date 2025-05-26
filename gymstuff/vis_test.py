import numpy as np
from vanishing_tictactoe import VanishingTicTacToeEnv
from random_agent import RandomAgent
from simplerulebased_agent import SimpleRuleBasedAgent
from moderaterulebased_agent import ModerateRuleBasedAgent
from complexrulebased_agent import ComplexRuleBasedAgent

# Render
# This is just a visualization tool, you can

DELAY = 0.001 # if this is larger it will render slower (to make it visible by humans, otherwise use 0)
BOARD_SIZE = 3 # nxn board

if __name__ == "__main__":
    env = VanishingTicTacToeEnv(board_size=BOARD_SIZE)

    # these agents can be changed to different agents in the future!
    # agent_x = RandomAgent(env.action_space)  # player X (1) 
    # agent_o = RandomAgent(env.action_space)  # player O (-1) 
    agent_x = ModerateRuleBasedAgent(env.action_space)  # player X (1) 
    agent_o = ComplexRuleBasedAgent(env.action_space)  # player O (-1)

    obs = env.reset()
    done = False
    env.render(delay=DELAY)

    while not done:
        current_player = env.current_player
        agent = agent_x if current_player == 1 else agent_o
        action = agent.act(obs)
        print(f"{'X' if current_player == 1 else 'O'} plays: {action}")
        obs, reward, done, info = env.step(action)
        env.render(delay=DELAY)

    if info.get("invalid", False):
        print(f"Player {'X' if current_player == 1 else 'O'} made an invalid move and lost.")
    elif reward == 1.0:
        print(f"Player {'X' if current_player == 1 else 'O'} wins!")
    elif reward == 0.5:
        print("It's a draw!")
    else:
        print(f"Player {'X' if current_player == 1 else 'O'} loses.")
