import os
import json
import matplotlib.pyplot as plt

CASTLING_MOVES_WHITE = {"e1g1", "e1c1"}
CASTLING_MOVES_BLACK = {"e8g8", "e8c8"}

def count_castling_moves_in_file(filepath):
    castling_count = 0
    white_castled = False
    black_castled = False
    try:
        with open(filepath, 'r') as file:
            game_data = json.load(file)
            for move_entry in game_data:
                move = move_entry.get("move", "")
                if move in CASTLING_MOVES_WHITE and not white_castled:
                    castling_count += 1
                    white_castled = True
                elif move in CASTLING_MOVES_BLACK and not black_castled:
                    castling_count += 1
                    black_castled = True
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
    return castling_count

def count_castling_moves_and_games_in_folder(folder_path):
    total_castling_count = 0
    game_count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            game_count += 1
            file_path = os.path.join(folder_path, filename)
            total_castling_count += count_castling_moves_in_file(file_path)
    return total_castling_count, game_count

def get_folders_sorted_by_date(base_folder):
    folders = [os.path.join(base_folder, subfolder)
               for subfolder in os.listdir(base_folder)
               if os.path.isdir(os.path.join(base_folder, subfolder))]
    folders.sort(key=lambda folder: os.path.getmtime(folder))
    return folders

def count_castling_moves_in_all_subfolders(base_folder):
    results = {}
    sorted_folders = get_folders_sorted_by_date(base_folder)
    for folder_path in sorted_folders:
        folder_name = os.path.basename(folder_path)
        castling_count, game_count = count_castling_moves_and_games_in_folder(folder_path)
        if game_count > 0:
            average_castling = castling_count / game_count
        else:
            average_castling = 0
        results[folder_name] = average_castling
    return results

def plot_castling_moves(castling_counts):
    folders = list(castling_counts.keys())
    averages = list(castling_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(folders, averages, color='blue')

    plt.axvspan(-0.5, 4.5, color='lightblue', alpha=0.3)
    plt.axvspan(4.5, 14.5, color='lightgreen', alpha=0.3)
    plt.axvspan(14.5, 17, color='lightcoral', alpha=0.3)

    plt.axvline(x=-0.5, color='gray', linestyle='--', linewidth=1)
    plt.axvline(x=4.5, color='gray', linestyle='--', linewidth=1)
    plt.axvline(x=14.5, color='gray', linestyle='--', linewidth=1)

    y_mid = (min(averages) + max(averages)) / 2 if averages else 0
    plt.text(-0.5, y_mid, "Warm-Up", color="gray", fontsize=18, ha="right", va="center", rotation=90)
    plt.text(4.5, y_mid, "Main", color="gray", fontsize=18, ha="left", va="center", rotation=90)
    plt.text(14.5, y_mid, "Fine Tuning", color="gray", fontsize=18, ha="left", va="center", rotation=90)

    plt.xlabel('Self-Play Schritt', fontsize=24)
    plt.ylabel('Durchschnittliche Rochaden pro Spiel', fontsize=24)
    plt.title('Rochaden pro Spiel pro Self-Play Schritt', fontsize=26)
    plt.xticks(rotation=45, ha='right', fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    plt.show()

base_folder_path = "C:\\Users\\Infin\\Desktop\\Chess Data backup\\records" #Replace with path to your folder

castling_counts = count_castling_moves_in_all_subfolders(base_folder_path)

print("Durchschnittliche Rochaden pro Spiel pro Ordner:")
for folder, average in castling_counts.items():
    print(f"{folder}: {average:.2f}")

plot_castling_moves(castling_counts)
