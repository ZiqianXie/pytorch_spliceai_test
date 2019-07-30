import numpy as np


def one_hot_encode(seq):
    d = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 0}
    data = np.fromiter(map(lambda x: d[x], seq.upper()), dtype='uint8')
    segment = np.zeros((4, len(data)), dtype='f')
    idx = data != 0
    segment[data[idx] - 1, idx] = 1
    return segment