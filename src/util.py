import numpy as np
import os
from glob import glob


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

    checkpoints = glob(f"{checkpoint_folder}/model_weights_cycle_*.h5")
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        model.model.load_weights(latest_checkpoint)
        model.compile_model()
        print(f"Loaded weights from {latest_checkpoint}")
    else:
        print("No checkpoint weights found. Starting with untrained weights.")


