import os
import json
import matplotlib.pyplot as plt

def count_moves_in_file(filepath):
    move_count = 0
    try:
        with open(filepath, 'r') as file:
            game_data = json.load(file)
            move_count = len([entry for entry in game_data if "move" in entry])
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
    return move_count

def average_moves_in_folder(folder_path):
    total_moves = 0
    file_count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            total_moves += count_moves_in_file(file_path)
            file_count += 1
    return total_moves / file_count if file_count > 0 else 0

def get_folders_sorted_by_date(base_folder):
    folders = [os.path.join(base_folder, subfolder) for subfolder in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, subfolder))]
    folders.sort(key=lambda folder: os.path.getmtime(folder))  # Sort by modification time
    return folders

def average_moves_in_all_subfolders(base_folder):
    results = {}
    sorted_folders = get_folders_sorted_by_date(base_folder)
    for folder_path in sorted_folders:
        folder_name = os.path.basename(folder_path)
        avg_moves = average_moves_in_folder(folder_path)
        print(f"Folder: {folder_name}, Average Moves: {avg_moves}")
        results[folder_name] = avg_moves
    return results

def plot_average_moves(averages):
    folders = list(averages.keys())
    avg_moves = list(averages.values())

    plt.figure(figsize=(10, 6))
    plt.bar(folders, avg_moves, color='blue')

    plt.axvspan(-0.5, 4.5, color='lightblue', alpha=0.3)
    plt.axvspan(4.5, 14.5, color='lightgreen', alpha=0.3)
    plt.axvspan(14.5, 17, color='lightcoral', alpha=0.3)

    plt.axvline(x=-0.5, color='gray', linestyle='--', linewidth=1)
    plt.axvline(x=4.5, color='gray', linestyle='--', linewidth=1)
    plt.axvline(x=14.5, color='gray', linestyle='--', linewidth=1)

    plt.text(-0.5, (min(avg_moves) + max(avg_moves)) / 2, "Warm-Up", color="gray", fontsize=18, ha="right",va="center", rotation=90)
    plt.text(4.5, (min(avg_moves) + max(avg_moves))/2, "Main", color="gray", fontsize=18, ha="left", va="center", rotation=90)
    plt.text(14.5, (min(avg_moves) + max(avg_moves))/2, "Fine Tuning", color="gray", fontsize=18, ha="left", va="center", rotation=90)

    plt.xlabel('Self-Play Schritt', fontsize=24)
    plt.ylabel('Durchschnittliche Anzahl Halb-Züge', fontsize=24)
    plt.title('Durchschnittliche Anzahl Halb-Züge pro Self-Play Schritt', fontsize=26)
    plt.xticks(rotation=45, ha='right', fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    plt.show()

base_folder_path = "C:\\Users\\Infin\\Desktop\\Chess Data backup\\records"

average_moves = average_moves_in_all_subfolders(base_folder_path)

plot_average_moves(average_moves)
