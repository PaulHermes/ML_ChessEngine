from game import Game
from chessboard import Chessboard
from agent import Agent
import numpy as np
import util
from reinforcementLearningModel import ReinforcementLearningModel as RLM
import parameters

if __name__ == '__main__':
    white = Agent()
    black = Agent()
    board = Chessboard()
    game = Game(board, white, black)

    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    game.chessBoard.board_to_nn_input(game.chessBoard.board)
    #for testing purposes just do first legal move 25 times now
    for i in range(0,25):
        game.play_move()
        print("--------------------------------------- \n")
        game.chessBoard.board_to_nn_input(game.chessBoard.board)

    rlm = RLM(parameters.neural_network_input, parameters.neural_network_output)
    rlm.build(True)