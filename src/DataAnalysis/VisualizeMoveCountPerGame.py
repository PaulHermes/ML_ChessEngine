import os
import json
import matplotlib.pyplot as plt

#Probably dont include this sieht nur komishc aus aufgund von batching
def count_moves_in_file(filepath):
    move_count = 0
    try:
        with open(filepath, 'r') as file:
            game_data = json.load(file)
            move_count = len([entry for entry in game_data if "move" in entry])
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
    return move_count


def get_files_sorted_by_date(folder_path):
    files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(".json")]
    files.sort(key=lambda file: os.path.getmtime(file))  # Sort by creation time
    return files


def get_folders_sorted_by_date(base_folder):
    folders = [os.path.join(base_folder, subfolder) for subfolder in os.listdir(base_folder) if
               os.path.isdir(os.path.join(base_folder, subfolder))]
    folders.sort(key=lambda folder: os.path.getmtime(folder))  # Sort by modification time
    return folders


def count_moves_in_all_games(base_folder):
    game_moves = []
    sorted_folders = get_folders_sorted_by_date(base_folder)
    for folder_path in sorted_folders:
        sorted_files = get_files_sorted_by_date(folder_path)
        for file_path in sorted_files:
            moves = count_moves_in_file(file_path)
            game_moves.append((os.path.basename(folder_path), os.path.basename(file_path), moves))
    return game_moves


def plot_game_moves(game_moves):
    game_labels = [f"{folder}/{game}" for folder, game, _ in game_moves]
    move_counts = [moves for _, _, moves in game_moves]

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(move_counts) + 1), move_counts, marker="o", label="Move Count")

    plt.xlabel("Spiel (sortiert nach Erstellungszeit)", fontsize=12)
    plt.ylabel("Anzahl Halb-Züge", fontsize=12)
    plt.title("Anzahl Halb-Züge pro Spiel in Erstellungsreihenfolge", fontsize=14)

    plt.xticks(range(1, len(move_counts) + 1, max(1, len(move_counts) // 20)), rotation=45, ha="right", fontsize=8)

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.legend(fontsize=10)
    plt.show()


base_folder_path = "C:\\Users\\Infin\\Desktop\\Chess Data backup\\records"

game_moves = count_moves_in_all_games(base_folder_path)

plot_game_moves(game_moves)
