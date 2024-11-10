import numpy as np
import os
from glob import glob
from tensorflow.keras.optimizers.schedules import ExponentialDecay, PolynomialDecay
from tensorflow.keras.callbacks import LearningRateScheduler


def int_to_binary8By8(input: int) -> np.ndarray:
    array = np.zeros((8, 8), dtype=int)

    binary_representation = format(input, '064b')

    for row in range(8):
        for col in range(8):
            bit_index = row * 8 + col
            array[row, col] = int(binary_representation[bit_index])

    return array

def load_latest_weights(model, checkpoint_folder="../checkpoints"):
    if not os.path.exists(checkpoint_folder):
        print("No checkpoints folder found. Starting with untrained weights.")
        return

    checkpoints = glob(f"{checkpoint_folder}/*.weights.h5")
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        model.model.load_weights(latest_checkpoint)
        model.compile_model()
        print(f"Loaded weights from {latest_checkpoint}")
    else:
        print("No checkpoint weights found. Starting with untrained weights.")


def get_learning_rate_schedule(stage): # warmup, main, finetune
    if stage == "warmup":
        # Warm-Up Stage: Linear Increase from 0.02 to 0.2 over 25 epochs
        initial_lr = 0.02
        target_lr = 0.2
        epochs_to_increase = 25  # Total epochs for warm-up stage

        def warmup_schedule(epoch, lr):
            if epoch < epochs_to_increase:
                return initial_lr + (target_lr - initial_lr) * (epoch / epochs_to_increase)
            else:
                return target_lr  # Keep learning rate constant after warm-up

        return LearningRateScheduler(warmup_schedule)

    elif stage == "main":
        # Main Training Stage: Exponential Decay from 0.2 to 0.02 over 100 epochs
        initial_lr = 0.2
        final_lr = 0.02
        decay_steps = 100  # Total epochs for main training stage

        return ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=final_lr / initial_lr
        )

    elif stage == "finetune":
        # Fine-Tuning Stage: Polynomial Decay from 0.02 to 0.002 over 200 epochs
        initial_lr = 0.02
        final_lr = 0.002
        decay_steps = 200  # Total epochs for fine-tuning stage

        return PolynomialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            end_learning_rate=final_lr,
            power=2.0  # Smoother decay
        )

    else:
        raise ValueError("Invalid stage. Choose from 'warmup', 'main', or 'finetune'.")
