import numpy as np
import os
import json

import tensorflow
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers.schedules import ExponentialDecay, PolynomialDecay

# Define a file to save the current epoch count
EPOCH_TRACK_FILE = "../epoch_tracking/epoch_tracker.json"

class LearningRateScheduler(Callback):
    def __init__(self, stage, checkpoint_file=EPOCH_TRACK_FILE):
        super().__init__()
        self.stage = stage
        self.checkpoint_file = checkpoint_file
        self.start_epoch = self.load_epoch_count()  # Load the saved epoch count at the start
        self.lr_schedule = self.get_learning_rate_schedule(stage)  # Schedule with respect to start epoch

    def load_epoch_count(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "r") as f:
                data = json.load(f)
            return data.get("epoch", 0)
        return 0

    def save_epoch_count(self, epoch):
        with open(self.checkpoint_file, "w") as f:
            json.dump({"epoch": epoch}, f)

    def manual_exponential_decay(self, initial_lr, final_lr, total_epochs, cumulative_epoch):
        decay_rate = (final_lr / initial_lr) ** (1 / total_epochs)
        lr = initial_lr * (decay_rate ** cumulative_epoch)
        return lr

    def get_learning_rate_schedule(self, stage):
        if stage == "warmup":
            initial_lr = 0.02
            target_lr = 0.005
            epochs_to_increase = 25  # Total epochs for warm-up stage

            def warmup_schedule(epoch):
                effective_epoch = self.start_epoch + epoch
                if effective_epoch < epochs_to_increase:
                    return initial_lr + (target_lr - initial_lr) * (effective_epoch / epochs_to_increase)
                else:
                    return target_lr  # Keep learning rate constant after warm-up

            return warmup_schedule

        elif stage == "main":
            # Parameters for the main phase
            initial_lr = 0.1
            final_lr = 0.02
            decay_steps = 100


            def cumulative_decay_schedule(epoch):
                cumulative_epoch = self.start_epoch + epoch
                lr = self.manual_exponential_decay(initial_lr, final_lr, decay_steps, cumulative_epoch)
                print(f"Cumulative Epoch: {cumulative_epoch}, Learning Rate: {lr:.6f}")
                return lr

            return cumulative_decay_schedule  # Return the adjusted callable
        elif stage == "finetune":
            initial_lr = 0.02
            final_lr = 0.002
            decay_steps = 200  # Total epochs for fine-tuning stage
            power = 2.0

            def polynomial_decay(epoch):
                # Polynomial decay implementation
                lr = initial_lr * (1 - epoch / decay_steps) ** power + final_lr * (epoch / decay_steps)
                print(f"Epoch: {epoch}, Learning Rate: {lr:.6f}")
                return lr
            return polynomial_decay

        else:
            raise ValueError("Invalid stage. Choose from 'warmup', 'main', or 'finetune'.")

    def on_epoch_begin(self, epoch, logs=None):
        # Calculate the new learning rate
        current_lr = self.lr_schedule(epoch)
        tensorflow.keras.backend.set_value(self.model.optimizer.learning_rate, current_lr)
        # Update the model's learning rate
        #self.model.optimizer.learning_rate = current_lr

    def on_epoch_end(self, epoch, logs=None):
        effective_epoch = self.start_epoch + epoch + 1  # Increment for next epoch start
        self.save_epoch_count(effective_epoch)


