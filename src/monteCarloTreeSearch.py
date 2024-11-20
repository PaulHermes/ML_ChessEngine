import os
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
    def __init__(self, board: chess.Board, neural_network: ReinforcementLearningModel, batch_size=16):
        self.root = MonteCarloTreeNode(board)
        self.neural_network = neural_network
        self.batch_size = batch_size
        self.prediction_queue = []
        self.prediction_results = {}

    def enqueue_prediction(self, node):
        board_input = Chessboard.board_to_nn_input(node.board)
        self.prediction_queue.append((node, board_input))

        # Trigger batch prediction if queue size reaches the batch size
        if len(self.prediction_queue) >= self.batch_size:
            self.process_prediction_queue()

    def process_prediction_queue(self):
        if not self.prediction_queue:
            return

        # Prepare batch input
        nodes, board_inputs = zip(*self.prediction_queue)
        board_inputs = np.array(board_inputs)


        print(board_inputs.shape)
        # Perform batch prediction
        policy_outputs, value_outputs = self.neural_network.model.predict(board_inputs, verbose=0)

        # Store predictions in results and assign them to nodes
        for i, node in enumerate(nodes):
            policy_output = policy_outputs[i]
            value_output = value_outputs[i][0]
            self.prediction_results[node] = (policy_output, value_output)

        # Clear the queue
        self.prediction_queue = []

    def get_predictions_for_node(self, node):
        if node in self.prediction_results:
            return self.prediction_results.pop(node)
        return None

    def selection(self):
        node = self.root
        while node.is_fully_expanded() and node.children:
            node = self.best_uct(node)
        return node

    def expansion(self, node):
        if node.is_fully_expanded():
            return

        # Enqueue prediction for the current node
        self.enqueue_prediction(node)

        # Check if predictions are ready for the node
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
            self.process_prediction_queue()  # Process any queued predictions
            predictions = self.get_predictions_for_node(node)
            if predictions is not None:
                _, value_output = predictions
                node.value = value_output

        # Ensure node.value is not None
        if node.value is None:
            raise ValueError("Simulation failed: node.value is still None after prediction.")

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
        for _ in range(num_simulations):
            leaf = self.selection()
            self.expansion(leaf)
            result = self.simulation(leaf)
            self.backpropagation(leaf, result)
        # Ensure all queued predictions are processed at the end
        self.process_prediction_queue()


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
        print(f"MCTS tree visualization saved to {filename}.png")
