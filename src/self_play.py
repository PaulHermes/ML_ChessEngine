from game import Game
from chessboard import Chessboard
from agent import Agent
from reinforcementLearningModel import ReinforcementLearningModel
import parameters
import numpy as np
import tensorflow as tf
import random
import json
import os
import chess


class SelfPlay:
    def __init__(self, model, num_games=100, random_start_probability=0.5):
        self.model = model
        self.num_games = num_games
        self.random_start_probability = random_start_probability
        self.data = []  # Stores game states, moves, and outcomes
        self.save_folder = "../self_play_records"

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def play(self):
        for game_index in range(self.num_games):
            print(f"\n--- Starting game {game_index + 1} ---")

            if random.random() < self.random_start_probability:
                random_fen = self.get_random_position()
                chess_board = Chessboard(starting_position_FEN=random_fen)
                print(f"Game {game_index + 1} starting from random position: {random_fen}")
            else:
                chess_board = Chessboard()

            white_agent = Agent(self.model)
            black_agent = Agent(self.model)
            game = Game(chess_board, white_agent, black_agent)
            game_data = []

            while not chess_board.board.is_game_over():
                nn_input = Chessboard.board_to_nn_input(chess_board.board)

                best_move = game.current_agent.get_best_move(chess_board.board)

                move_data = {
                    "state": nn_input.tolist(),
                    "move": best_move.uci(),
                    "player": "white" if game.current_agent == white_agent else "black"
                }
                game_data.append(move_data)

                game.play_move()

            outcome = self.get_game_outcome(chess_board)
            for move in game_data:
                move["outcome"] = outcome
            self.data.extend(game_data)

            # Save data after each game
            self.save_data(filename=f"{self.save_folder}/self_play_data_game_{game_index + 1}.json")
            print(f"Game {game_index + 1} finished with outcome: {outcome}")

            game.reset()

        # Saving data of all games as well for now. Will see if this is better for training
        self.save_data(filename=f"{self.save_folder}/self_play_data_all_games.json")

    def get_random_position(self):
        board = chess.Board()
        # 2 to 15ish moves for midgame
        for _ in range(random.randint(2, 15)):
            if board.is_game_over():
                board = chess.Board()
            else:
                move = random.choice(list(board.legal_moves))
                board.push(move)

        return board.fen()

    def get_game_outcome(self, chess_board):
        result = chess_board.board.result()
        if result == "1-0":
            return 1
        elif result == "0-1":
            return 0
        else:
            return 0.5

    def save_data(self, filename="self_play_data.json"):
        with open(filename, 'w') as f:
            json.dump(self.data, f)
        print(f"Self-play data saved to {filename}")


if __name__ == '__main__':
    model = ReinforcementLearningModel(parameters.neural_network_input, parameters.neural_network_output)
    model.build()

    # Keras log settings
    tf.keras.utils.disable_interactive_logging()

    # Run self-play games
    num_games = 10
    self_play = SelfPlay(model, num_games, 1.0)
    self_play.play()