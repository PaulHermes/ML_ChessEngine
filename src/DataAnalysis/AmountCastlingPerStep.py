import os
import json
import matplotlib.pyplot as plt

CASTLING_MOVES = {"e1g1", "e1c1", "e8g8", "e8c8"}

def count_castling_moves_in_file(filepath):
    castling_count = 0
    try:
        with open(filepath, 'r') as file:
            game_data = json.load(file)
            for move_entry in game_data:
                move = move_entry.get("move", "")
                if move in CASTLING_MOVES:
                    castling_count += 1
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
    return castling_count

def count_castling_moves_in_folder(folder_path):
    total_castling_count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            total_castling_count += count_castling_moves_in_file(file_path)
    return total_castling_count

def get_folders_sorted_by_date(base_folder):
    folders = [os.path.join(base_folder, subfolder) for subfolder in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, subfolder))]
    folders.sort(key=lambda folder: os.path.getmtime(folder))  # Sort by modification time
    return folders

def count_castling_moves_in_all_subfolders(base_folder):
    results = {}
    sorted_folders = get_folders_sorted_by_date(base_folder)
    for folder_path in sorted_folders:
        folder_name = os.path.basename(folder_path)
        castling_count = count_castling_moves_in_folder(folder_path)
        results[folder_name] = castling_count
    return results

def plot_castling_moves(castling_counts):
    folders = list(castling_counts.keys())
    counts = list(castling_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(folders, counts, color='blue')

    plt.axvspan(-0.5, 4.5, color='lightblue', alpha=0.3)
    plt.axvspan(4.5, 14.5, color='lightgreen', alpha=0.3)
    plt.axvspan(14.5, 17, color='lightcoral', alpha=0.3)

    plt.axvline(x=-0.5, color='gray', linestyle='--', linewidth=1)
    plt.axvline(x=4.5, color='gray', linestyle='--', linewidth=1)
    plt.axvline(x=14.5, color='gray', linestyle='--', linewidth=1)

    plt.text(-0.5, (min(counts) + max(counts)) / 2, "Warm-Up", color="gray", fontsize=10, ha="right",va="center", rotation=90)
    plt.text(4.5, (min(counts) + max(counts))/2, "Main", color="gray", fontsize=10, ha="left", va="center", rotation=90)
    plt.text(14.5, (min(counts) + max(counts))/2, "Fine Tuning", color="gray", fontsize=10, ha="left", va="center", rotation=90)

    plt.xlabel('Self-Play Schritt', fontsize=12)
    plt.ylabel('Anzahl Rochaden', fontsize=12)
    plt.title('Anzahl Rochaden pro Self-Play Schritt', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.show()

base_folder_path = "C:\\Users\\Infin\\Desktop\\Chess Data backup\\records"

castling_counts = count_castling_moves_in_all_subfolders(base_folder_path)

print("Castling Moves per Folder:")
for folder, count in castling_counts.items():
    print(f"{folder}: {count}")

plot_castling_moves(castling_counts)
