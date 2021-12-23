import numpy as np

"""Calculate the moving average over w indexes for each period"""
def moving_avg(x: np.ndarray, w: int) -> np.ndarray:
    tmp = np.convolve(x, np.ones(w), 'valid') / w
    padding = np.full_like(np.empty(w-1), np.nan)
    return np.insert(tmp, 0, padding)


"""Return the percent change of two moving averages"""
def moving_avg_diff(short, long) -> np.ndarray:
    return (short - long) / long


"""perc_change(x)[i] = x[i] - x[i-shift] / 2"""
def perc_change(x: np.ndarray, shift=1) -> np.ndarray:
    tmp = (x - np.roll(x, shift)) / np.roll(x, shift)
    if tmp.size > 0:
        tmp[0] = np.nan
    return tmp


"""1 if perc_change(x, shift) > 0"""
def binary_labels(x: np.ndarray, shift=1) -> np.ndarray:
    tmp = perc_change(x, shift)
    labels = np.zeros(tmp.shape)
    labels[tmp > 0] = 1
    labels[:shift] = np.nan
    return labels


""" 0 if perc_change(x, shift) <= threshold, 2 if perc_change(x, shift) >= threshold, else 1"""
def trinary_labels(x: np.ndarray, threshold, shift=1):
    tmp = perc_change(x, shift)
    labels = np.ones(tmp.shape)
    labels[tmp >= threshold] = 2
    labels[tmp <= -threshold] = 0
    labels[:shift] = np.nan
    return labels