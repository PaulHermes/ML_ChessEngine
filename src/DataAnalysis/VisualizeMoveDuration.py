import matplotlib.pyplot as plt

#Normalize by num simulations?
#make sure to mention that batching start with main
move_durations = [
    2.18/75, 2.38/75, 2.53/75, 2.41/75, 2.42/75, 2.39/200, 2.2/200, 2.32/200, 2.2/200, 2.3/200, 2.4/200, 2.6/200, 2.48/200, 2.53/200, 3.35/200, 5.52/400, 5.71/400
]

def plot_move_durations(durations):
    steps = range(1, len(durations) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, durations, marker='o', label="Move Duration")

    plt.axvspan(1, 5.5, color='lightblue', alpha=0.3)
    plt.axvspan(5.5, 15.5, color='lightgreen', alpha=0.3)
    plt.axvspan(15.5, 17, color='lightcoral', alpha=0.3)

    plt.axvline(x=1, color='gray', linestyle='--', linewidth=1)
    plt.axvline(x=5.5, color='gray', linestyle='--', linewidth=1)
    plt.axvline(x=15.5, color='gray', linestyle='--', linewidth=1)

    plt.text(1, (min(move_durations) + max(move_durations)) / 2, "Warm-Up", color="gray", fontsize=10, ha="left",va="center", rotation=90)
    plt.text(5.5, (min(move_durations) + max(move_durations))/2, "Main", color="gray", fontsize=10, ha="left", va="center", rotation=90)
    plt.text(15.5, (min(move_durations) + max(move_durations))/2, "Fine Tuning", color="gray", fontsize=10, ha="left", va="center", rotation=90)

    plt.xlabel("Self-Play Schritt", fontsize=12)
    plt.ylabel("Durchschnittliche Simulationsberechnungsdauer", fontsize=12)
    plt.title("Durchschnittliche Simulationsberechnungsdauer pro Self-Play Schritt", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.legend(fontsize=10)

    plt.show()

plot_move_durations(move_durations)