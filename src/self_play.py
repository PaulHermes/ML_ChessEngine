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
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tensorflow.keras import mixed_precision


class SelfPlay:
    def __init__(self, model, num_games=100, random_start_probability=0.5):
        self.model = model
        self.num_games = num_games
        self.random_start_probability = random_start_probability
        self.save_folder = "../self_play_records"
        self.total_moves_played = 0
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
            self.total_moves_played += 1
            print(f"Total moves played: {self.total_moves_played}")
            if self.total_moves_played % 25 == 0:
                print("clearing session")
                tf.keras.backend.clear_session()

        outcome = self.get_game_outcome(chess_board)
        for move in game_data:
            move["outcome"] = outcome

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{self.save_folder}/self_play_data_game_{game_index}_{timestamp}.json"
        self.save_data(game_data, filename)

    def play(self):
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=parameters.self_play_batch_size) as executor:
            # Submit initial batch of games
            futures = {executor.submit(self.play_game, i): i for i in range(parameters.self_play_batch_size)}
            # Keep submitting new games as others finish
            for i in range(parameters.self_play_batch_size, self.num_games):
                done_future = next(as_completed(futures))
                try:
                    done_future.result()  # Ensure the finished game's result is processed
                except Exception as e:
                    print(f"Game {futures[done_future]} failed with exception: {e}")
                # Remove the finished future and submit a new game
                futures.pop(done_future)
                futures[executor.submit(self.play_game, i)] = i
            # Wait for the remaining futures to finish
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Game {futures[future]} failed with exception: {e}")

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
        checkpoints = sorted(glob.glob(f"{checkpoint_folder}/*weights.h5"), key=os.path.getctime)

        results = {}

        def evaluate_checkpoint(checkpoint):
            print(f"\n--- Evaluating against checkpoint {checkpoint} ---")
            start_time = time.time()
            opponent_model = ReinforcementLearningModel(parameters.neural_network_input,
                                                        parameters.neural_network_output)
            opponent_model.build(compile_model=False)
            opponent_model.model.load_weights(checkpoint)

            wins, losses, draws = 0, 0, 0

            def play_single_game():
                chess_board = Chessboard()
                white_agent = Agent(self.model)
                black_agent = Agent(opponent_model)
                game = Game(chess_board, white_agent, black_agent)

                while not chess_board.board.is_game_over():
                    best_move = game.current_agent.get_best_move(chess_board.board, greedy=True)
                    print(checkpoint, best_move)
                    game.play_move(best_move)
                    self.total_moves_played += 1
                    if self.total_moves_played % 25 == 0:
                        tf.keras.backend.clear_session()

                return self.get_game_outcome(chess_board)

            with ThreadPoolExecutor() as executor:
                outcomes = list(executor.map(lambda _: play_single_game(), range(num_games)))

            for outcome in outcomes:
                if outcome == 1:
                    wins += 1
                elif outcome == 0:
                    losses += 1
                else:
                    draws += 1

            end_time = time.time()
            delta_time = end_time - start_time
            print(f"Execution time for checkpoint {checkpoint}: {delta_time:.2f} seconds")
            print(f"Results against {checkpoint}: {wins} wins, {losses} losses, {draws} draws")
            return checkpoint, {"wins": wins, "losses": losses, "draws": draws}

        with ThreadPoolExecutor(max_workers=parameters.self_play_batch_size) as executor:
            future_results = [executor.submit(evaluate_checkpoint, checkpoint) for checkpoint in checkpoints[:-1]]
            for future in future_results:
                checkpoint, result = future.result()
                results[checkpoint] = result

        return results

if __name__ == '__main__':
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    mixed_precision.set_global_policy('float32')

    model = ReinforcementLearningModel(parameters.neural_network_input, parameters.neural_network_output)
    model.build()

    # Keras log settings
    tf.keras.utils.disable_interactive_logging()

    # Run self-play games
    num_games = parameters.self_play_per_cycle
    self_play = SelfPlay(model, num_games, 0.5)
    self_play.play()

    # Evaluate against previous checkpoints
    #eval_num_games = parameters.eval_games
    #results = self_play.evaluate_against_previous_versions(eval_num_games)