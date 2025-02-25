import os
import json
import matplotlib.pyplot as plt

def count_game_results_in_file(filepath):
    try:
        with open(filepath, 'r') as file:
            game_data = json.load(file)
            if game_data and isinstance(game_data, list):
                if "outcome" not in game_data[0]:
                    print(f"Missing 'outcome' key in file: {filepath}")
                    return ""
                outcome = game_data[0].get("outcome", None)  # Use the "outcome" key
                if outcome == 1:
                    return "win"
                elif outcome == 0.5:
                    return "draw"
                elif outcome == 0:
                    return "loss"
                else:
                    print(f"Invalid outcome value ({outcome}) in file: {filepath}")
                    return ""
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
    return ""

def aggregate_game_results_in_folder(folder_path):
    total_results = {"win": 0, "loss": 0, "draw": 0}
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            result = count_game_results_in_file(file_path)
            if result in total_results:
                total_results[result] += 1
    return total_results

def get_folders_sorted_by_date(base_folder):
    folders = [os.path.join(base_folder, subfolder) for subfolder in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, subfolder))]
    folders.sort(key=lambda folder: os.path.getmtime(folder))
    return folders

def aggregate_game_results_in_all_subfolders(base_folder):
    results = {}
    sorted_folders = get_folders_sorted_by_date(base_folder)
    for folder_path in sorted_folders:
        folder_name = os.path.basename(folder_path)
        folder_results = aggregate_game_results_in_folder(folder_path)
        results[folder_name] = folder_results
    return results

def plot_win_loss_draw_ratios(game_results):
    folders = list(game_results.keys())
    win_ratios = []
    loss_ratios = []
    draw_ratios = []

    for folder in folders:
        total = sum(game_results[folder].values())
        if total > 0:
            win_ratios.append(game_results[folder]["win"] / total)
            loss_ratios.append(game_results[folder]["loss"] / total)
            draw_ratios.append(game_results[folder]["draw"] / total)
        else:
            win_ratios.append(0)
            loss_ratios.append(0)
            draw_ratios.append(0)

    x = range(len(folders))
    draw_bottom = draw_ratios
    loss_bottom = [d + l for d, l in zip(draw_ratios, loss_ratios)]

    plt.figure(figsize=(12, 6))
    plt.bar(x, draw_ratios, width=0.6, label='Draw Ratio', color='blue')
    plt.bar(x, loss_ratios, width=0.6, label='Loss Ratio', color='red', bottom=draw_bottom)
    plt.bar(x, win_ratios, width=0.6, label='Win Ratio', color='green', bottom=loss_bottom)

    plt.xlabel('Self-Play Schritt', fontsize=24)
    plt.ylabel('Verhältnis', fontsize=24)
    plt.title('Win/Loss/Draw Verhältnis pro Self-Play Schritt', fontsize=26)
    plt.xticks(x, folders, rotation=45, ha='right', fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=24)
    plt.tight_layout()
    plt.show()

base_folder_path = "C:\\Users\\Infin\\Desktop\\Chess Data backup\\records" #Replace with path to your folder

game_results = aggregate_game_results_in_all_subfolders(base_folder_path)

print("Win/Loss/Draw Ratios per Folder:")
for folder, results in game_results.items():
    total = sum(results.values())
    if total > 0:
        print(f"{folder}: Win: {results['win'] / total:.2f}, Loss: {results['loss'] / total:.2f}, Draw: {results['draw'] / total:.2f}")
    else:
        print(f"{folder}: No games found.")

plot_win_loss_draw_ratios(game_results)
