from agent import Agent
from chessboard import Chessboard
import random
import chess

#A Singular Chess Game. 2 Agents play 1 Game. Moves are made on 1 Chessboard
class Game:
    def __init__(self, chess_board: Chessboard, white: Agent, black: Agent):
        self.chessBoard = chess_board
        self.white = white
        self.black = black
        self.current_agent = white

    def reset(self):
        self.chessBoard.reset()

    def play_move(self, move: chess.Move):
        print("\n" + str(move))

        self.chessBoard.move_piece(move)
        self.current_agent = self.black if self.current_agent == self.white else self.white

