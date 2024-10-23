import numpy as np


def int_to_binary8By8(input: int) -> np.ndarray:
    array = np.zeros((8, 8), dtype=int)

    binary_representation = format(input, '064b')

    for row in range(8):
        for col in range(8):
            bit_index = row * 8 + col
            array[row, col] = int(binary_representation[bit_index])

    return array




