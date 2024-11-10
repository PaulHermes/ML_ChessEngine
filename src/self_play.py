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
import datetime
import util
import glob
from concurrent.futures import ThreadPoolExecutor
import time

class SelfPlay:
    def __init__(self, model, num_games=100, random_start_probability=0.5):
        self.model = model
        self.num_games = num_games
        self.random_start_probability = random_start_probability
        self.save_folder = "../self_play_records"

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        util.load_latest_weights(self.model, "../checkpoints")

    def play_game(self, game_index):
        if random.random() < self.random_start_probability:
            random_fen = self.get_random_position()
            chess_board = Chessboard(starting_position_FEN=random_fen)
        else:
            chess_board = Chessboard()

        white_agent = Agent(self.model)
        black_agent = Agent(self.model)
        game = Game(chess_board, white_agent, black_agent)
        game_data = []

        while not chess_board.board.is_game_over():
            nn_input = Chessboard.board_to_nn_input(chess_board.board)
            best_move = game.current_agent.get_best_move(chess_board.board, greedy=False)
            move_data = {
                "state": nn_input.tolist(),
                "move": best_move.uci(),
                "player": "white" if game.current_agent == white_agent else "black"
            }
            game_data.append(move_data)
            print(f"------------ Game {game_index} ------------- \n")
            game.play_move(best_move)

        outcome = self.get_game_outcome(chess_board)
        for move in game_data:
            move["outcome"] = outcome

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{self.save_folder}/self_play_data_game_{game_index}_{timestamp}.json"
        self.save_data(game_data, filename)

    def play(self):
        start_time = time.time()

        # Run games in batches
        for batch_start in range(0, self.num_games, parameters.self_play_batch_size):
            batch_end = min(batch_start + parameters.self_play_batch_size, self.num_games)
            print(f"Running batch {batch_start // parameters.self_play_batch_size + 1}: Games {batch_start} to {batch_end - 1}")

            with ThreadPoolExecutor(max_workers=parameters.self_play_batch_size) as executor:
                futures = [executor.submit(self.play_game, i) for i in range(batch_start, batch_end)]
                for future in futures:
                    future.result()  # Wait for all games in the batch to complete

        end_time = time.time()
        delta_time = end_time - start_time
        print(f"Total execution time for play function: {delta_time:.2f} seconds")

    def get_random_position(self):
        board = chess.Board()
        # 2 to 15ish moves for midgame
        for _ in range(random.randint(2, 15)):
            if board.is_game_over():
                board = chess.Board()
            else:
                move = random.choice(list(board.legal_moves))
                board.push(move)
        return board.fen() #"1k6/8/8/8/3QK3/8/8/8 w - - 0 1" "4k2K/8/8/8/8/8/3P1P2/8 b - - 0 1" "8/8/8/8/8/3k4/5R2/3K4 w - - 0 1" "K7/7k/R7/7r/8/8/8/8 w - - 0 1"

    def get_game_outcome(self, chess_board):
        result = chess_board.board.result()
        if result == "1-0":
            return 1
        elif result == "0-1":
            return 0
        else:
            return 0.5

    def save_data(self, game_data, filename="self_play_data.json"):
        with open(filename, 'w') as f:
            json.dump(game_data, f)
        print(f"Self-play data saved to {filename}")

    def evaluate_against_previous_versions(self, num_games=100):
        checkpoint_folder = "../checkpoints"
        checkpoints = sorted(glob.glob(f"{checkpoint_folder}/model_weights_cycle_*.h5"), key=os.path.getctime)

        results = {}

        # For each previous checkpoint, evaluate the latest model against it
        for i, checkpoint in enumerate(checkpoints[:-1]):
            print(f"\n--- Evaluating against checkpoint {checkpoint} ---")
            opponent_model = ReinforcementLearningModel(parameters.neural_network_input,
                                                        parameters.neural_network_output)
            opponent_model.build(compile_model=False)
            opponent_model.model.load_weights(checkpoint)

            wins, losses, draws = 0, 0, 0

            for _ in range(num_games):
                chess_board = Chessboard()
                white_agent = Agent(self.model)
                black_agent = Agent(opponent_model)

                game = Game(chess_board, white_agent, black_agent)

                while not chess_board.board.is_game_over():
                    best_move = game.current_agent.get_best_move(chess_board.board, greedy=True)
                    game.play_move(best_move)

                outcome = self.get_game_outcome(chess_board)
                if outcome == 1:
                    wins += 1
                elif outcome == 0:
                    losses += 1
                else:
                    draws += 1

            results[checkpoint] = {"wins": wins, "losses": losses, "draws": draws}
            print(f"Results against {checkpoint}: {wins} wins, {losses} losses, {draws} draws")

        return results

if __name__ == '__main__':
    model = ReinforcementLearningModel(parameters.neural_network_input, parameters.neural_network_output)
    model.build()

    # Keras log settings
    tf.keras.utils.disable_interactive_logging()

    # Run self-play games
    num_games = parameters.self_play_per_cycle
    self_play = SelfPlay(model, num_games, 1)
    self_play.play()

    # Evaluate against previous checkpoints
    #eval_num_games = parameters.eval_games
    #results = self_play.evaluate_against_previous_versions(eval_num_games)