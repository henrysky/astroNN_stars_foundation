import copy
import numpy as np
from numpy.typing import NDArray
from typing import Optional, List

rng = np.random.default_rng()


def shuffle_row(arrays_ls: List[NDArray], first_n: Optional[int] = None) -> None:
    """
    Function to shuffle row-wise multiple 2D array of the same shape in the same way

    This function would change the original array in place!!!

    Parameters
    ----------
    arrays_ls : list of array
        The array(s) to be shuffled
    first_n : int, optional
        If not None, the first ``first_n`` elements will be shuffled, by default all elements will be shuffled

    Returns
    -------
    None
    """
    array_shape = tuple(arrays_ls[0].shape)
    num_array = len(arrays_ls)
    # deep copy rng except one so they will behave the same in order to shuffle multiple arrays in the same way
    rng_ls = [rng] + [copy.deepcopy(rng) for _ in range(num_array - 1)]

    if first_n is None:  # if yes then use a slightly faster implementation
        # shuffling row by row for all arrays in the same way
        [rng_ls[idx].shuffle(a, axis=1) for idx, a in enumerate(arrays_ls)]
    else:
        if isinstance(first_n, int):
            first_n = [first_n] * array_shape[0]
        # shuffling row by row for all arrays in the same way
        [
            r.shuffle(a[i, :n])
            for a, r in zip(arrays_ls, rng_ls)
            for i, n in zip(np.arange(array_shape[0]), first_n)
        ]
    return None


def random_choice(items: NDArray, prob_matrix: Optional[NDArray] = None) -> NDArray:
    """
    See https://stackoverflow.com/questions/34187130/fast-random-weighted-selection-across-all-rows-of-a-stochastic-matrix
    """
    if prob_matrix is None:  # if not provided, assume uniform distribution
        prob_matrix = np.tile(
            np.ones_like(items.shape[1], dtype=float), (items.shape[0], 1)
        ).T
    # making sure prob_matrix is normalized to 1
    prob_matrix /= prob_matrix.sum(axis=1, keepdims=True)

    s = prob_matrix.cumsum(axis=1)
    r = np.random.rand(prob_matrix.shape[0], 1)
    k = (s < r).sum(axis=1)
    return np.take_along_axis(items, k[:, None], axis=1)
