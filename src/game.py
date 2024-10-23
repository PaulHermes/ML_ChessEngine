from agent import Agent
from chessboard import Chessboard
import random

#A Singular Chess Game. 2 Agents play 1 Game. Moves are made on 1 Chessboard
class Game:
    def __init__(self, chess_board: Chessboard, white: Agent, black: Agent):
        self.chessBoard = chess_board
        self.white = white
        self.black = black

    def reset(self):
        self.chessBoard.reset()

    def play_move(self):
        legal_moves = list(self.chessBoard.board.legal_moves)
        if legal_moves:
            self.chessBoard.move_piece(legal_moves[random.randint(0, len(legal_moves)-1)])
