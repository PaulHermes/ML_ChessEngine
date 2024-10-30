from agent import Agent
from chessboard import Chessboard
import random

#A Singular Chess Game. 2 Agents play 1 Game. Moves are made on 1 Chessboard
class Game:
    def __init__(self, chess_board: Chessboard, white: Agent, black: Agent):
        self.chessBoard = chess_board
        self.white = white
        self.black = black
        self.current_agent = white

    def reset(self):
        self.chessBoard.reset()

    def play_move(self):
        best_move = self.current_agent.get_best_move(self.chessBoard.board)
        print("\n" + str(best_move))

        self.chessBoard.move_piece(best_move)
        self.current_agent = self.black if self.current_agent == self.white else self.white

