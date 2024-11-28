import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np
import chess
from reinforcementLearningModel import ReinforcementLearningModel
from chessboard import Chessboard
from predictionManager import PredictionManager


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
        self.lock = Lock()

    def is_fully_expanded(self):
        return len(self.children) == len(list(self.board.legal_moves))

    def expand(self, move, prior_prob):
        with self.lock:
            for child in self.children:
                if child.move == move:
                    return child

            next_board = self.board.copy()
            next_board.push(move)
            child_node = MonteCarloTreeNode(next_board, parent=self, prior_prob=prior_prob)
            child_node.move = move
            self.children.append(child_node)
            return child_node


class MonteCarloTree:
    def __init__(self, board: chess.Board, neural_network):
        self.root = MonteCarloTreeNode(board)
        self.prediction_manager = PredictionManager()
        self.prediction_manager.set_neural_network(neural_network)
        self.lock = Lock()

    def enqueue_prediction(self, node):
        board_input = Chessboard.board_to_nn_input(node.board)
        self.prediction_manager.enqueue_prediction(node, board_input)

    def get_predictions_for_node(self, node):
        return self.prediction_manager.get_predictions_for_node(node)

    def selection(self):
        with self.lock:
            node = self.root
            while node.is_fully_expanded() and node.children:
                node = self.best_uct(node)
            return node

    def expansion(self, node):
        with self.lock:
            if node.is_fully_expanded():
                return

            # Enqueue prediction for the current node
            self.enqueue_prediction(node)

            # Block until predictions are ready for the node
            predictions = self.get_predictions_for_node(node)
            if predictions is not None:
                policy_output, value_output = predictions
                legal_moves, legal_probs = self.probabilities_to_actions(policy_output, node.board)

                # Expand children with calculated probabilities
                for move, prob in zip(legal_moves, legal_probs):
                    node.expand(move, prob)

                node.value = value_output

    def simulation(self, node):
        # Ensure predictions for the node are available
        while node.value is None:
            predictions = self.get_predictions_for_node(node)
            if predictions is not None:
                _, value_output = predictions
                node.value = value_output

        return node.value

    def backpropagation(self, node, result):
        while node:
            with node.lock:
                node.visits += 1
                node.wins += result if node.board.turn == chess.WHITE else (1 - result)
                node = node.parent

    def best_uct(self, node):
        with node.lock:
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

    def run_simulation(self):
        leaf = self.selection()
        self.expansion(leaf)
        result = self.simulation(leaf)
        self.backpropagation(leaf, result)

    def run(self, num_simulations):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.run_simulation) for _ in range(num_simulations)]
            for future in futures:
                future.result()

    def visualize_mcts_tree(self, max_depth=3, filename="mcts_tree"):
        import graphviz
        from queue import Queue

        dot = graphviz.Digraph(format='pdf')
        dot.attr(rankdir='TB')  # Set higher DPI and larger size

        # Add nodes and edges to the graph
        queue = Queue()
        queue.put((self.root, 0))  # Start with (node, depth) as (root_node, 0)

        while not queue.empty():
            node, depth = queue.get()

            if depth > max_depth:
                continue

            # Create a label for the current node
            label = f"Visits: {node.visits}\nWins: {node.wins:.2f}\nMove: {node.move}"
            dot.node(
                str(id(node)),
                label,
                shape='box',
                fontsize='12',  # Larger font size
                margin='0.2',
            )

            if node.parent:
                dot.edge(
                    str(id(node.parent)),
                    str(id(node)),
                    penwidth='2',  # Thicker edge lines
                )

            # Add children to the queue with incremented depth
            for child in node.children:
                queue.put((child, depth + 1))

        # Render the tree to a file
        dot.render(filename, cleanup=True)
        print(f"MCTS tree visualization saved to {filename}.pdf")

