import numpy as np
import chess
from reinforcementLearningModel import ReinforcementLearningModel
from chessboard import Chessboard


class MonteCarloTreeNode:
    def __init__(self, board: chess.Board, parent=None, prior_prob=0.0):
        self.board = board
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.move = None
        self.prior_prob = prior_prob
        self.value = None

    def is_fully_expanded(self):
        return len(self.children) == len(list(self.board.legal_moves))

    def expand(self, move, prior_prob):
        next_board = self.board.copy()
        next_board.push(move)
        child_node = MonteCarloTreeNode(next_board, parent=self, prior_prob=prior_prob)
        child_node.move = move
        self.children.append(child_node)
        return child_node


class MonteCarloTree:
    def __init__(self, board: chess.Board, neural_network: ReinforcementLearningModel):
        self.root = MonteCarloTreeNode(board)
        self.neural_network = neural_network

    def selection(self):
        node = self.root
        while node.is_fully_expanded() and node.children:
            node = self.best_uct(node)
        return node

    def expansion(self, node):
        if node.is_fully_expanded():
            return

        # Get outputs from neural network
        board_input = np.expand_dims(Chessboard.board_to_nn_input(node.board), axis=0)
        policy_output, value_output = self.neural_network.model.predict(board_input)

        # Calculate legal moves and their probabilities
        legal_moves, legal_probs = self.probabilities_to_actions(policy_output[0], node.board)

        # Expand with children for each legal move
        for move, prob in zip(legal_moves, legal_probs):
            node.expand(move, prob)

        node.value = value_output[0][0]

    def simulation(self, node):
        # Return value or predict new value if not already evaluated
        if node.value is None:
            board_input = np.expand_dims(Chessboard.board_to_nn_input(node.board), axis=0)
            _, value_output = self.neural_network.model.predict(board_input)
            node.value = value_output[0][0]
        return node.value

    def backpropagation(self, node, result):
        while node:
            node.visits += 1
            node.wins += result if node.board.turn == chess.WHITE else (1 - result)
            node = node.parent

    def best_uct(self, node):
        best_value = -float('inf')
        best_node = None
        sqrt_visits = np.sqrt(node.visits)
        c = 1.4

        for child in node.children:
            if child.visits == 0:
                return child  # Prioritize unexplored nodes

            uct_value = (child.wins / child.visits) + c * child.prior_prob * (sqrt_visits / (1 + child.visits))
            if uct_value > best_value:
                best_value = uct_value
                best_node = child

        return best_node

    def run(self, num_simulations):
        for _ in range(num_simulations):
            leaf = self.selection()
            self.expansion(leaf)
            result = self.simulation(leaf)
            self.backpropagation(leaf, result)

    def probabilities_to_actions(self, policy_output, board):
        legal_moves = list(board.legal_moves)
        legal_probs = np.array([policy_output[self.move_to_index(move)] for move in legal_moves])
        total_prob = legal_probs.sum()
        if total_prob > 0:
            legal_probs /= total_prob
        else:
            legal_probs = np.ones(len(legal_probs)) / len(legal_probs)  # Uniform distribution if sum is zero
        return legal_moves, legal_probs

    def move_to_index(self, move):
        return move.from_square * 64 + move.to_square
