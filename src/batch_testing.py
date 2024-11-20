import random
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import tensorflow as tf
from reinforcementLearningModel import ReinforcementLearningModel
import parameters
from tensorflow.keras import mixed_precision
from agent import Agent
from game import Game
from chessboard import Chessboard


class BatchTesting:
    def __init__(self, model: ReinforcementLearningModel, num_models=1):
        # Initialize multiple models
        self.models = [model] + [
            ReinforcementLearningModel(parameters.neural_network_input, parameters.neural_network_output)
            for _ in range(num_models - 1)
        ]
        # Build each model
        for m in self.models:
            m.build()

    def run_test(self, type="threading"):
        chess_board = Chessboard()
        white_agent = Agent(self.models[0])  # Use the first model for agents
        black_agent = Agent(self.models[0])
        game = Game(chess_board, white_agent, black_agent)

        start_time = time.time()

        if type == "threading":  # 93.53 sek
            for i in range(10):
                move = random.choice(list(chess_board.board.legal_moves))
                game.play_move(move)
                board_input = np.expand_dims(Chessboard.board_to_nn_input(game.chessBoard.board), axis=0)
                with ThreadPoolExecutor(max_workers=parameters.self_play_batch_size) as executor:
                    futures = [executor.submit(self.models[0].model.predict, board_input, verbose=0) for j in range(500)]
                    for future in futures:
                        future.result()
        elif type == "sequential":  # 188.42 seconds
            for i in range(10):
                move = random.choice(list(chess_board.board.legal_moves))
                game.play_move(move)
                board_input = np.expand_dims(Chessboard.board_to_nn_input(game.chessBoard.board), axis=0)
                for j in range(500):
                    self.models[0].model.predict(board_input, verbose=0)
        elif type == "batched":  # 4.83 seconds
            for i in range(10):
                move = random.choice(list(chess_board.board.legal_moves))
                game.play_move(move)
                batch = []
                for j in range(500):
                    temp_board = chess_board.board.copy()
                    temp_legal_moves = list(temp_board.legal_moves)
                    if temp_legal_moves:
                        temp_move = random.choice(temp_legal_moves)
                        temp_board.push(temp_move)
                        temp_board.turn = not temp_board.turn
                    board_input = np.expand_dims(Chessboard.board_to_nn_input(temp_board), axis=0)
                    batch.append(board_input)
                batch = np.vstack(batch)
                self.models[0].model.predict(batch, verbose=1)
        elif type == "multi": # 92.41 seconds
            for i in range(10):
                move = random.choice(list(chess_board.board.legal_moves))
                game.play_move(move)
                results = []
                with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
                    futures = []
                    for j in range(500):
                        temp_board = chess_board.board.copy()
                        temp_legal_moves = list(temp_board.legal_moves)
                        if temp_legal_moves:
                            temp_move = random.choice(temp_legal_moves)
                            temp_board.push(temp_move)
                            temp_board.turn = not temp_board.turn
                        board_input = np.expand_dims(Chessboard.board_to_nn_input(temp_board), axis=0)
                        model_idx = j % len(self.models)
                        futures.append(
                            executor.submit(self.models[model_idx].model.predict, board_input, verbose=0)
                        )
                    for future in futures:
                        results.append(future.result())

        end_time = time.time()
        delta_time = end_time - start_time
        print(f"Total execution time for {type} mode: {delta_time:.2f} seconds")


if __name__ == '__main__':
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    mixed_precision.set_global_policy('mixed_float16')

    model = ReinforcementLearningModel(parameters.neural_network_input, parameters.neural_network_output)
    tester = BatchTesting(model, num_models=1)

    #tester.run_test("threading")
    #tester.run_test("sequential")
    tester.run_test("batched")
    #tester.run_test("multi")
