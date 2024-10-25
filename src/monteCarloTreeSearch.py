import chess
import random
import math
import numpy as np

class MonteCarloTreeNode:
    def __init__(self, board: chess.Board, parent=None):
        self.board = board
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.move = None

    def is_fully_expanded(self):
        return len(self.children) == len(list(self.board.legal_moves))

    def expand(self):
        legal_moves = list(self.board.legal_moves)
        # Expand a random unexplored move
        for move in legal_moves:
            if not any(child.move == move for child in self.children):
                next_board = self.board.copy()
                next_board.push(move)
                child_node = MonteCarloTreeNode(next_board, parent=self)
                child_node.move = move
                self.children.append(child_node)
                return child_node
        return None

class MonteCarloTree:
    def __init__(self, board: chess.Board):
        self.root = MonteCarloTreeNode(board)

    def selection(self):
        node = self.root
        while node.is_fully_expanded() and len(node.children) > 0:
            node = self.best_uct(node)
        return node

    def expansion(self, node):
        if not node.is_fully_expanded():
            return node.expand()
        return node

    def simulation(self, node):
        current_board = node.board.copy()
        while not current_board.is_game_over():
            legal_moves = list(current_board.legal_moves)
            move = random.choice(legal_moves)
            current_board.push(move)

        # Return the result of the simulation (1 for win, 0 for loss, 0.5 for draw)
        if current_board.is_checkmate():
            return 1 if current_board.turn == chess.BLACK else 0  # Black wins if it's White's turn at checkmate
        else:
            return 0.5

    def backpropagation(self, node, result):
        while node is not None:
            node.visits += 1
            node.wins += result if node.board.turn == chess.WHITE else (1 - result)
            node = node.parent

    def best_uct(self, node):
        # UCB1 formula to balance exploration and exploitation
        best_value = -float('inf')
        best_node = None
        c = 1.4  # Exploration constant, typically sqrt(2)

        for child in node.children:
            if child.visits == 0:
                return child  # Prioritize unexplored nodes

            uct_value = (child.wins / child.visits) + c * math.sqrt(math.log(node.visits) / child.visits)
            if uct_value > best_value:
                best_value = uct_value
                best_node = child

        return best_node

    def run(self, num_simulations):
        for _ in range(num_simulations):
            leaf = self.selection()
            expanded_node = self.expansion(leaf)
            if expanded_node is not None:
                leaf = expanded_node
            result = self.simulation(leaf)
            self.backpropagation(leaf, result)

    def best_move_greedy(self):
        return max(self.root.children, key=lambda child: child.visits).move

    def best_move_probability(self, temperature=1.0):
        visit_counts = np.array([child.visits for child in self.root.children])
        if temperature == 0: # 0 Means no randomness so we choose greedily
            return self.best_move_greedy()

        # Apply temperature scaling for exploration (higher temperature means more exploration)
        visit_counts = visit_counts ** (1.0 / temperature)

        probabilities = visit_counts / np.sum(visit_counts)
        chosen_index = np.random.choice(len(self.root.children), p=probabilities)

        return self.root.children[chosen_index].move

# Example usage:
# Start a chess game and initialize MCTS with the current board state
board = chess.Board()

mcts = MonteCarloTree(board)
mcts.run(num_simulations=20000)

best_move = mcts.best_move_greedy()
print(f"Best move: {best_move}")

best_move = mcts.best_move_probability(0.01)
print(f"Best move: {best_move}")

best_move = mcts.best_move_greedy()
print(f"Best move: {best_move}")

best_move = mcts.best_move_probability(1)
print(f"Best move: {best_move}")
