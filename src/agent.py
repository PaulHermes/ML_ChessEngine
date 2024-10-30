from monteCarloTreeSearch import MonteCarloTree
import numpy as np
import parameters

#AI Agents play chess
class Agent:
    def __init__(self, model):
        self.model = model

    def get_best_move(self, board):
        mcts = MonteCarloTree(board, self.model)
        mcts.run(num_simulations=parameters.number_of_simulations)

        # Select the move based on visit count
        best_move = mcts.root.children[np.argmax([child.visits for child in mcts.root.children])].move

        # Print for debugging
        print("\n")
        for child in mcts.root.children:
            print(str(child.move) + " | " + str(child.visits) + " | " + str(child.wins))
        return best_move