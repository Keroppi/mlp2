import numpy as np

def get_targets():
    target_data = ""
    with open('./data/targets.csv', 'r') as fo:
        target_data = fo.read()

    numbers = target_data.split('\n')
    numbers.pop(len(numbers) - 1)

    vectorized_int = np.vectorize(int)
    targets = vectorized_int(numbers)

    return targets