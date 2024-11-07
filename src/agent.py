from monteCarloTreeSearch import MonteCarloTree
import numpy as np
import parameters
import chess.syzygy

#AI Agents play chess
class Agent:
    def __init__(self, model):
        self.model = model

    def get_best_move(self, board, greedy=True):
        if len(board.piece_map()) <= 5:
            return self.get_endgame_move(board)

        mcts = MonteCarloTree(board, self.model)
        mcts.run(num_simulations=parameters.number_of_simulations)

        if greedy:
            # Greedy choice for evaluation/real play/endgame
            best_move = mcts.root.children[np.argmax([child.visits for child in mcts.root.children])].move
        else:
            # Stochastic choice for training/self-play
            visits = np.array([child.visits for child in mcts.root.children])
            probabilities = visits / visits.sum()
            selected_child = np.random.choice(mcts.root.children, p=probabilities)
            best_move = selected_child.move

        # Print for debugging
        #print("\n")
        #for child in mcts.root.children:
            #print(str(child.move) + " | " + str(child.visits) + " | " + str(child.wins))
        return best_move

    def get_endgame_move(self, board):
        # Check if the board is in an endgame position and use the tablebase
        with chess.syzygy.open_tablebase('../syzygy') as tablebase:
            try:
                legal_moves = list(board.legal_moves)
                best_move = None
                initial_dtz = tablebase.probe_dtz(board)  # Get the DTZ of the current position
                best_dtz = initial_dtz  # Start with the current DTZ as the baseline

                # Check if initial_dtz is exactly 1, then look for any move that results in a checkmate
                if initial_dtz == 1:
                    for move in legal_moves:
                        board.push(move)

                        # Check if the move results in checkmate
                        if board.is_checkmate():
                            board.pop()
                            return move  # Return immediately since this is the best possible move

                        board.pop()

                # If no mate-in-1 is found, proceed with the regular DTZ improvement check
                for move in legal_moves:
                    board.push(move)
                    dtz = tablebase.probe_dtz(board)
                    # Invert the DTZ comparison due to turn change after board.push()
                    comparison_dtz = -initial_dtz

                    # Choose move that improves DTZ relative to the inverted initial DTZ after the move but avoid draw(cause issues with inversion)
                    if dtz > comparison_dtz and dtz != 0:
                        best_dtz = dtz
                        best_move = move

                    board.pop()

                # If no "optimal" move found based on DTZ, fallback to the first legal move
                if best_move is None and legal_moves:
                    best_move = legal_moves[0]
                    print(f"No optimal move found; using fallback first legal move: {best_move}")

                print(f"Final Selected Best Move: {best_move}")
                return best_move
            except Exception as e:
                print(f"Error probing tablebase: {e}")
                return None

