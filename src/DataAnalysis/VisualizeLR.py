import matplotlib.pyplot as plt

def manual_exponential_decay(initial_lr, final_lr, total_epochs, cumulative_epoch):
    decay_rate = (final_lr / initial_lr) ** (1 / total_epochs)
    lr = initial_lr * (decay_rate ** cumulative_epoch)
    return lr

def polynomial_decay(epoch, initial_lr, final_lr, decay_steps, power):
    lr = initial_lr * (1 - epoch / decay_steps) ** power + final_lr * (epoch / decay_steps)
    return lr

manual_lrs = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1625, 0.125, 0.0875, 0.05, 0.05, 0.045, 0.04, 0.035, 0.03, 0.03, 0.025, 0.02, 0.015, 0.01, 0.01, 0.00875, 0.0075, 0.00625, 0.005 ]

initial_lr_expo = 0.0075
final_lr_expo = 0.001
expo_epochs = 100

initial_lr_poly = 0.001
final_lr_poly = 0.0001
poly_epochs = 200
poly_power = 2

decay_steps = poly_epochs

expo_lrs = [manual_exponential_decay(initial_lr_expo, final_lr_expo, expo_epochs, epoch) for epoch in range(1, expo_epochs + 1)]
poly_lrs = [polynomial_decay(epoch, initial_lr_poly, final_lr_poly, decay_steps, poly_power) for epoch in range(1, poly_epochs + 1)]

epochs = list(range(1, len(manual_lrs) + len(expo_lrs) + len(poly_lrs) + 1))
all_lrs = manual_lrs + expo_lrs + poly_lrs

plt.figure(figsize=(10, 6))
plt.plot(epochs, all_lrs, marker="o", label="Learning Rate")

plt.axvspan(1, 25.5, color='lightblue', alpha=0.3)
plt.axvspan(25.5, 125.5, color='lightgreen', alpha=0.3)
plt.axvspan(125.5, 325.5, color='lightcoral', alpha=0.3)

plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
plt.axvline(x=25.5, color='gray', linestyle='--', linewidth=1)
plt.axvline(x=125.5, color='gray', linestyle='--', linewidth=1)
plt.text(0, (min(all_lrs) + max(all_lrs)) / 2, "Warm-Up", color="gray", fontsize=18, ha="left",va="center", rotation=90)
plt.text(25.5, (min(all_lrs) + max(all_lrs))/2, "Main", color="gray", fontsize=18, ha="left", va="center", rotation=90)
plt.text(125.5, (min(all_lrs) + max(all_lrs))/2, "Fine Tuning", color="gray", fontsize=18, ha="left", va="center", rotation=90)

plt.xlabel("Epoche", fontsize=24)
plt.ylabel("Learning Rate (Log Scale)", fontsize=24)
plt.title("Learning Rate über die Epochen", fontsize=26)
plt.yscale("log")
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.legend(fontsize=24)
plt.show()
