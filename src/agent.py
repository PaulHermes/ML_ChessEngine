from monteCarloTreeSearch import MonteCarloTree
import numpy as np
import parameters

#AI Agents play chess
class Agent:
    def __init__(self, model):
        self.model = model

    def get_best_move(self, board, greedy=True):
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