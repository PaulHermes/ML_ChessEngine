from monteCarloTreeSearch import MonteCarloTree
import numpy as np
import parameters
import chess.gaviota

#AI Agents play chess
class Agent:
    def __init__(self, model):
        self.model = model
        self.gaviota_path = "../gaviota/gaviota_tablebases"

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
        print("\n")
        for child in mcts.root.children:
            print(str(child.move) + " | " + str(child.visits) + " | " + str(child.wins))
        return best_move

    def get_endgame_move(self, board):
        with chess.gaviota.open_tablebase(self.gaviota_path) as tablebase:
            try:
                initial_dtm = tablebase.probe_dtm(board)
                best_move = None
                best_dtm = -initial_dtm


                for move in board.legal_moves:
                    board.push(move)
                    try:
                        dtm = tablebase.probe_dtm(board)  # Probe DTM (Distance-to-Mate)

                        if initial_dtm > 0: # you are winning
                            if dtm >= best_dtm and dtm != 0:
                                best_dtm = dtm
                                best_move = move
                            if dtm == 0 and board.is_checkmate():
                                best_dtm = dtm
                                best_move = move
                        elif initial_dtm < 0: # you are losing
                            if dtm < best_dtm:
                                best_dtm = dtm
                                best_move = move
                        else: # draw
                            pass
                    except chess.gaviota.MissingTableError:
                        # Skip if the position is not found in the tablebase
                        pass
                    finally:
                        board.pop()

                # If no optimal move was found in the tablebase, return a fallback move
                return best_move if best_move is not None else list(board.legal_moves)[0]

            except chess.gaviota.MissingTableError:
                print("No tablebase data available for this position.")
                return None