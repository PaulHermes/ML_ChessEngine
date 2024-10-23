

# ------------------- Neural Network Input Parameters ---------------------------
# https://arxiv.org/pdf/1712.01815 page 13
# N × N × (MT + L) where N is 8 M is Feature Planes T is history and L is constant valued input

# P1 Pieces + P2 Pieces + Repetition
amount_feature_planes_m = (6 + 6 + 2)
# Repetition could be compacted to 1 plane if its not binary

# Colour + Total Move count + P1 Castling + P2 Castling + No Progress Count
constant_valued_input_l = (1 + 1 + 2 + 2 + 1)
# No Progress Count could be left out for faster training since this is rather niche in low elo. therefore:
# ShouldUseNoProgressCount = True

history_t = 8 #is implementing this too much? big ones use history, smaller ones dont.
# most likely will not implement in beginning and add as way to improve. therefore:
#NeuralNetworkInput = (AmountFeaturePlanesM * HistoryT + ConstantValuedInputL, 8, 8)
# En passant needs history tho so extra layer for en passant?

neural_network_input = (amount_feature_planes_m + constant_valued_input_l, 8, 8)

# -------------------------------------------------------------------------------


# --------------------------- Neural Network Output Parameters  -----------------
# https://arxiv.org/pdf/1712.01815 Page 14

# 8 * 8 of possible squares to choose the piece from
# Multiplied with 56 possible Queen-Moves (Up to 7 in all 8 directions) + 8 possible Knight-Moves + 9 possible Underpromotion moves (2 Captures + 1 Forward) * 3(Rook, Knight, Bishop)
neural_network_output = (8 * 8 * (56 + 8 + 9), 1)

# -------------------------------------------------------------------------------

