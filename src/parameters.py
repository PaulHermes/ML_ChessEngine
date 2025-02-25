import os

# ------------------- Neural Network Input Parameters ---------------------------
# https://arxiv.org/pdf/1712.01815 page 13
# N × N × (MT + L) where N is 8 M is Feature Planes T is history and L is constant valued input

# P1 Pieces + P2 Pieces + Repetition
amount_feature_planes_m = (6 + 6 + 2)
# Repetition could be compacted to 1 plane if its not binary

# Colour + Total Move count + P1 Castling + P2 Castling + No Progress Count
constant_valued_input_l = (1 + 1 + 2 + 2 + 1)
# No Progress Count could be left out for faster training since this is rather niche in low elo.


history_t = 8 # Is implementing this too much? big ones use history, smaller ones dont.
# Made it optional incase training is too slow
use_history = True

neural_network_input = (8, 8, amount_feature_planes_m * history_t + constant_valued_input_l) if use_history else (8, 8, amount_feature_planes_m + constant_valued_input_l)

# -------------------------------------------------------------------------------


# --------------------------- Neural Network Output Parameters  -----------------
# https://arxiv.org/pdf/1712.01815 Page 14

# 8 * 8 of possible squares to choose the piece from
# Multiplied with 56 possible Queen-Moves (Up to 7 in all 8 directions) + 8 possible Knight-Moves + 9 possible Underpromotion moves (2 Captures + 1 Forward) * 3(Rook, Knight, Bishop)
possible_moves = 56 + 8 + 9
neural_network_output = (8 * 8 * possible_moves,)

# -------------------------------------------------------------------------------

# --------------------------- Neural Network Training Parameters  -----------------
# https://arxiv.org/pdf/1712.01815 Page 14-15
# https://www.chessprogramming.org/Leela_Chess_Zero#Network
# https://www.chessprogramming.org/AlphaZero#Network_Architecture
# the smaller the faster it trains and shows progress, but this comes with a downside of a lower skill cap

convolution_filters = 256

residual_block_count = 19

kernel_size = 3

stride = 1

batch_size = 256

adam_beta_1 = 0.9
adam_beta_2 = 0.999

weight_decay = 1e-5

self_play_per_cycle = 500
self_play_batch_size = 250#os.cpu_count()

eval_games = 1

epochs = 20
# -------------------------------------------------------------------------------

# --------------------------- MCTS Parameters  -----------------

# https://arxiv.org/pdf/1712.01815 Page 14

number_of_simulations = 400

prediction_batch_size = 24

use_prediction_manager = True

enforce_batch_size = False

process_interval = 10
# -------------------------------------------------------------------------------


