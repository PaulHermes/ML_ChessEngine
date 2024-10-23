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

    def board_to_nn_input(self, board_input: chess.Board) -> np.ndarray:
        board = board_input


        # ---------------- Colour Plane -----------------
        colour_plane = np.ones((8, 8)) if board.turn else np.zeros((8, 8))

        # ---------------- Total Move Count Plane -----------------
        total_move_counter = board.fullmove_number
        total_move_count_plane = util.int_to_binary8By8(total_move_counter)

        # ---------------- Castling Planes -----------------
        p1_castling_plane = np.stack([
            np.ones((8, 8)) if board.has_queenside_castling_rights(chess.WHITE) else np.zeros((8, 8)),
            np.ones((8, 8)) if board.has_kingside_castling_rights(chess.WHITE) else np.zeros((8, 8))
        ])

        p2_castling_plane = np.stack([
            np.ones((8, 8)) if board.has_queenside_castling_rights(chess.BLACK) else np.zeros((8, 8)),
            np.ones((8, 8)) if board.has_kingside_castling_rights(chess.BLACK) else np.zeros((8, 8))
        ])

        # ---------------- No Progress Count Plane -----------------
        no_progress_counter = board.halfmove_clock
        no_progress_count_plane = util.int_to_binary8By8(no_progress_counter)

        # ---------------- P1 Piece Plane -----------------
        p1_piece_planes = []
        for pieceType in chess.PIECE_TYPES:
            piece_plane = np.zeros((8, 8))
            for i in list(board.pieces(pieceType, chess.WHITE)):
                piece_plane[7 - int(i / 8)][i % 8] = 1 # cause bottom left is origin
            p1_piece_planes.append(piece_plane)
        p1_piece_planes = np.array(p1_piece_planes)

        # ---------------- P2 Piece Plane -----------------
        p2_piece_planes = []
        for pieceType in chess.PIECE_TYPES:
            piece_plane = np.zeros((8, 8))
            for i in list(board.pieces(pieceType, chess.BLACK)):
                piece_plane[7 - int(i / 8)][i % 8] = 1
            p2_piece_planes.append(piece_plane)
        p2_piece_planes = np.array(p2_piece_planes)

        # ---------------- Repetition Planes -----------------
        repetition_planes = np.stack([
            np.ones((8, 8)) if board.is_repetition(2) else np.zeros((8, 8)),
            np.ones((8, 8)) if board.is_repetition(3) else np.zeros((8, 8))
        ])

        # Concatenate all planes along the first axis for better readability
        ret = np.concatenate([
            colour_plane[np.newaxis],
            total_move_count_plane[np.newaxis],
            p1_castling_plane,
            p2_castling_plane,
            no_progress_count_plane[np.newaxis],
            p1_piece_planes,
            p2_piece_planes,
            repetition_planes
        ], axis=0)

        return ret.astype(bool)

