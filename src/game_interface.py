import chess
import tkinter as tk
from PIL import Image, ImageTk
import cairosvg
import chess.svg
import io
import threading
from chessboard import Chessboard
from agent import Agent
from reinforcementLearningModel import ReinforcementLearningModel
import parameters

model = ReinforcementLearningModel(parameters.neural_network_input, parameters.neural_network_output)
model.build()

white_agent = Agent(model)
black_agent = Agent(model)

board = Chessboard()

# Tkinter setup
root = tk.Tk()
root.title("Chess AI Game")
canvas = tk.Canvas(root, width=600, height=600)
canvas.pack()

SQUARE_SIZE = 75
WHITE = "#f0d9b5"
BLACK = "#b58863"

# Global variables for game settings
HUMAN_COLOR = chess.WHITE
AI_AGENT = black_agent

# Function to generate piece images from SVGs
piece_images = {}
for piece in ['P', 'R', 'N', 'B', 'Q', 'K', 'p', 'r', 'n', 'b', 'q', 'k']:
    svg_code = chess.svg.piece(chess.Piece.from_symbol(piece))
    png_image = cairosvg.svg2png(bytestring=svg_code)
    pil_image = Image.open(io.BytesIO(png_image)).resize((SQUARE_SIZE, SQUARE_SIZE), Image.LANCZOS)
    piece_images[piece] = ImageTk.PhotoImage(pil_image)

def draw_board():
    canvas.delete("all")
    for row in range(8):
        for col in range(8):
            color = WHITE if (row + col) % 2 == 0 else BLACK
            x1, y1 = col * SQUARE_SIZE, row * SQUARE_SIZE
            x2, y2 = x1 + SQUARE_SIZE, y1 + SQUARE_SIZE
            canvas.create_rectangle(x1, y1, x2, y2, fill=color)

def draw_pieces():
    for square in chess.SQUARES:
        piece = board.board.piece_at(square)
        if piece:
            col = chess.square_file(square)
            row = 7 - chess.square_rank(square)
            x, y = col * SQUARE_SIZE, row * SQUARE_SIZE
            piece_symbol = piece.symbol()
            canvas.create_image(x, y, anchor=tk.NW, image=piece_images[piece_symbol])

# Handle user click
selected_square = None

def rebind_click():
    root.bind("<Button-1>", on_square_click)

def apply_ai_move(ai_move):
    if ai_move in board.board.legal_moves:
        board.move_piece(ai_move)
        draw_board()
        draw_pieces()
        root.update()
    rebind_click()
    if board.board.is_game_over():
        print("Game over! Result: ", board.board.result())
        root.quit()

def ai_move_thread():
    try:
        print("AI is thinking...")
        ai_move = AI_AGENT.get_best_move(board.board, greedy=True)
        print(f"AI chose move: {ai_move}")
        root.after(0, apply_ai_move, ai_move)
    except Exception as e:
        print(f"Error during AI move: {e}")
        root.after(0, rebind_click)

def on_square_click(event):
    global selected_square
    col = event.x // SQUARE_SIZE
    row = 7 - (event.y // SQUARE_SIZE)
    square = chess.square(col, row)

    if board.board.turn != HUMAN_COLOR:
        return  # Not the human's turn, ignore click

    if selected_square is None:
        piece = board.board.piece_at(square)
        if piece and piece.color == HUMAN_COLOR:
            selected_square = square
    else:
        move = chess.Move(selected_square, square)
        if move in board.board.legal_moves:
            board.move_piece(move)
            selected_square = None
            draw_board()
            draw_pieces()
            root.update()

            if not board.board.is_game_over():
                root.unbind("<Button-1>")
                threading.Thread(target=ai_move_thread).start()
        else:
            selected_square = None

def play_human_as_white():
    board.reset()
    draw_board()
    draw_pieces()
    root.bind("<Button-1>", on_square_click)
    root.mainloop()

def play_human_as_black():
    board.reset()
    draw_board()
    draw_pieces()
    root.bind("<Button-1>", on_square_click)
    # AI makes first move
    root.unbind("<Button-1>")
    threading.Thread(target=ai_move_thread).start()
    root.mainloop()

def play_human_vs_ai():
    global HUMAN_COLOR, AI_AGENT
    color_choice = input("\nChoose your color:\n1. White\n2. Black\nEnter choice: ")
    if color_choice == '1':
        HUMAN_COLOR = chess.WHITE
        AI_AGENT = black_agent
        play_human_as_white()
    elif color_choice == '2':
        HUMAN_COLOR = chess.BLACK
        AI_AGENT = white_agent
        play_human_as_black()
    else:
        print("Invalid choice, defaulting to White")
        HUMAN_COLOR = chess.WHITE
        AI_AGENT = black_agent
        play_human_as_white()

def play_ai_vs_ai():
    board.reset()
    print("AI vs AI game started!")
    draw_board()
    draw_pieces()
    root.update()

    def ai_vs_ai_loop():
        if board.board.is_game_over():
            print("Game over! Result: " + board.board.result())
            root.quit()
            return
        current_agent = white_agent if board.board.turn == chess.WHITE else black_agent
        try:
            print(f"{('White' if board.board.turn == chess.WHITE else 'Black')} AI is thinking...")
            best_move = current_agent.get_best_move(board.board, greedy=True)
            print(f"{('White' if board.board.turn == chess.WHITE else 'Black')} AI chose move: {best_move}")
        except Exception as e:
            print(f"Error during AI move: {e}")
            root.quit()
            return
        if best_move in board.board.legal_moves:
            board.move_piece(best_move)
            draw_board()
            draw_pieces()
            root.update()
            root.after(500, ai_vs_ai_loop)
        else:
            print("Invalid move chosen by AI.")
            root.quit()

    root.after(500, ai_vs_ai_loop)
    root.mainloop()

def main():
    while True:
        print("\nChoose an option:")
        print("1. Play against AI")
        print("2. Watch AI vs AI")
        print("3. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            play_human_vs_ai()
        elif choice == '2':
            play_ai_vs_ai()
        elif choice == '3':
            print("Goodbye!")
            root.quit()
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    main()