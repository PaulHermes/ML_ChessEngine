import chess
import parameters
import numpy as np
import util


# The actual chessboard where moves are made
class Chessboard:
    # giving StartPosition as FEN. by default its the normal starting position. For training purposes overwriting this and giving puzzles would be best
    def __init__(self, starting_position_FEN: str = chess.STARTING_FEN):
        self.starting_position_FEN = starting_position_FEN
        self.reset()

    # reset board
    def reset(self):
        self.board = chess.Board(self.starting_position_FEN)

    # board to string
    def __str__(self):
        return str(chess.Board(self.board.fen())) + "\n"

    def move_piece(self, move: chess.Move) -> chess.Board:
        self.board.push(move)
        print(self.__str__())
        return self.board

    @staticmethod
    def board_to_nn_input(board_input: chess.Board) -> np.ndarray:
        # Create a copy of the board to safely modify for history states
        board = board_input.copy()

        # --- Colour Plane (1 plane) ---
        colour_plane = np.ones((8, 8)) if board.turn else np.zeros((8, 8))

        # --- Total Move Count Plane (1 plane) ---
        total_move_counter = board.fullmove_number
        total_move_count_plane = util.int_to_binary8By8(total_move_counter)

        # --- Castling Planes (4 planes) ---
        q1_castling_plane = np.ones((8, 8)) if board.has_queenside_castling_rights(chess.WHITE) else np.zeros((8, 8))
        k1_castling_plane = np.ones((8, 8)) if board.has_kingside_castling_rights(chess.WHITE) else np.zeros((8, 8))
        q2_castling_plane = np.ones((8, 8)) if board.has_queenside_castling_rights(chess.BLACK) else np.zeros((8, 8))
        k2_castling_plane = np.ones((8, 8)) if board.has_kingside_castling_rights(chess.BLACK) else np.zeros((8, 8))

        # --- No Progress Count Plane (1 plane) ---
        no_progress_counter = board.halfmove_clock
        no_progress_count_plane = util.int_to_binary8By8(no_progress_counter)

        # Collect the non-history planes
        non_history_planes = [
            colour_plane,
            total_move_count_plane,
            q1_castling_plane,
            k1_castling_plane,
            q2_castling_plane,
            k2_castling_plane,
            no_progress_count_plane
        ]

        # Initialize list to store all planes, starting with non-history planes
        all_planes = []
        all_planes.extend(non_history_planes)

        # Generate historical planes based on `move_stack`
        move_stack = list(board_input.move_stack)
        history_limit = min(len(move_stack), parameters.history_t) if parameters.use_history else 1

        # Step backwards through the history, applying moves in reverse to simulate past states
        for _ in range(history_limit):
            # --- P1 Piece Planes (6 planes) ---
            p1_piece_planes = []
            for piece_type in chess.PIECE_TYPES:
                piece_plane = np.zeros((8, 8))
                for i in list(board.pieces(piece_type, chess.WHITE)):
                    piece_plane[7 - int(i / 8)][i % 8] = 1  # Flip row index to start from bottom-left
                p1_piece_planes.append(piece_plane)

            # --- P2 Piece Planes (6 planes) ---
            p2_piece_planes = []
            for piece_type in chess.PIECE_TYPES:
                piece_plane = np.zeros((8, 8))
                for i in list(board.pieces(piece_type, chess.BLACK)):
                    piece_plane[7 - int(i / 8)][i % 8] = 1
                p2_piece_planes.append(piece_plane)

            # --- Repetition Planes (2 planes) ---
            repetition_planes = [
                np.ones((8, 8)) if board.is_repetition(2) else np.zeros((8, 8)),
                np.ones((8, 8)) if board.is_repetition(3) else np.zeros((8, 8))
            ]

            # Combine the 16 history planes for this board state
            board_history_planes = p1_piece_planes + p2_piece_planes + repetition_planes
            all_planes.extend(board_history_planes)

            # Undo the last move to simulate the previous board state
            board.pop() if board.move_stack else None  # Only pop if there are moves left

        # If not enough history, pad with zero-filled planes to reach the required history_t
        zero_plane = np.zeros((8, 8))
        planes_per_history_state = 6 + 6 + 2  # 6 for P1 pieces, 6 for P2 pieces, and 2 for repetition
        if(parameters.use_history):
            for _ in range(parameters.history_t - history_limit):
                # Add 16 zero planes for each missing history state
                all_planes.extend([zero_plane] * planes_per_history_state)

        # Stack all planes along the first axis
        ret = np.stack(all_planes, axis=-1)

        return ret.astype(bool)

    def print_nn_input(self, nn_input: np.ndarray):
        plane_types = [
            "Colour",
            "Total Move Count",
            "Q1 Castling",
            "K1 Castling",
            "Q2 Castling",
            "K2 Castling",
            "No Progress Count",
            "P1 Pawn",
            "P1 Knight",
            "P1 Bishop",
            "P1 Rook",
            "P1 Queen",
            "P1 King",
            "P2 Pawn",
            "P2 Knight",
            "P2 Bishop",
            "P2 Rook",
            "P2 Queen",
            "P2 King",
            "Repetition 2",
            "Repetition 3",
        ]
        for i in range(nn_input.shape[2]):
            print(f"Plane {i+1}:")
            if i < 7:
                print(plane_types[i])
            else:
                piece_index = (i - 7) % (len(plane_types) - 7) + 7
                print(plane_types[piece_index])
            print(nn_input[:, :, i])
            print()

