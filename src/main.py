from game import Game
from chessboard import Chessboard
from agent import Agent
import numpy as np
import util
from reinforcementLearningModel import ReinforcementLearningModel as RLM
import parameters
import tensorflow as tf


if __name__ == '__main__':
    rlm = RLM(parameters.neural_network_input, parameters.neural_network_output)
    rlm.build()

    white = Agent(rlm)
    black = Agent(rlm)

    board = Chessboard()
    game = Game(board, white, black)

    # Print settings
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    tf.keras.utils.disable_interactive_logging()

    for i in range(100):
        game.play_move()
        print("---------------------------------------\n")