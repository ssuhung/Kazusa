import os

import numpy as np

"""Process the raw output of prime and probe attack according to parameters to compact, clean format ready for training"""

# Processing Parameters
miss_threshold = 120             # Threshold of an access to be cache miss, unit in cycles
retain_timestamp = False          # Whether or not to retain the sample timestamp
remove_empty_row = True          # Whether or not to remove the row with no cache miss
padding_length = 1            # Whether or not to do padding
concatenate_trials = False
hit_miss_as_miss = False         # Whether or not to use the wrong method as in Manifold paper to define victom memory access

# Job Parameters
target_directory = "../data/CelebA_jpg/"

def extract_timestamp(arr):
    return arr[:, 0:1]

def remove_timestamp(arr):
    return arr[:, 1:65]

def attach_timestamp(arr, timestamp):
    return np.concatenate((arr, timestamp), axis=1)

def cycles_to_hit_or_miss(arr, threshold):
    # Including invert the result, since cache miss for the attacker means the access of the victim
    return (arr > threshold).astype(int)

def delete_empty_row(arr, timestamp):
    mask = np.any(arr != 0, axis=1)
    return arr[mask], timestamp[mask]

def padding(arr, length):
    padding_len = length - arr.ndim
    a = np.zeros((length, 64))
    print(a)
    return arr

if __name__ == '__main__':
    files = os.listdir(target_directory)
    max_len = 0

    for file_name in files:
        array = np.array([
            [0.01, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 100, 100, 100, 100, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 200, 200, 210],
            [0.02, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0],
            [0.03, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 210, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0]
            ])
        # array = np.load(file_name)['arr_0']
        
        if hit_miss_as_miss:
            raise NotImplementedError
        
        timestamp = extract_timestamp(array)
        array = remove_timestamp(array)

        array = cycles_to_hit_or_miss(array, miss_threshold)

        if remove_empty_row:
            array, timestamp = delete_empty_row(array, timestamp)

        # if retain_timestamp is False:
        #     array = attach_timestamp(array, timestamp)
        if padding_length is not None:
            array = padding(array, padding_length)

        # print(f"{array.shape=}")
        # print(array)
        # print(f"{timestamp.shape=}")
        # print(timestamp)
        max_len = max(max_len, array.shape[0])

        # np.savez_compressed(array)

    print(f"{max_len=}")
