import os.path
from pathlib import Path
import typing as t

import numpy as np


def load_raw_data(
    file: t.Union[str, Path],
    save_npy: bool = True,
    load_npy: bool = True,
) -> np.ndarray:
    """Load given file and generate a Ndarray of the data.

    Args:
        file: Path to a file with experimental data.
        save_npy: Whether the function creates new npy files from given data.

    Returns:
        Ndarray of the data.
    """
    if isinstance(file, str):
        file = Path(file)

    if file.suffix == ".npy":
        return np.load(file)

    file_npy = file.with_suffix(".npy")
    if file.suffix == ".txt":
        if load_npy and file_npy.exists():
            return np.load(file_npy)

        arr = np.loadtxt(file)
        if save_npy:
            npy_file = change_suffix(file, "npy")
            np.save(npy_file, arr)
        return arr

    raise ValueError(
        f"Extension '{file.suffix[1:]}' cannot be used for SP data. "
        "Use a txt or npy file."
    )


def _check_min_max(min_: float, max_: float) -> None:
    if min_ > max_:
        raise ValueError(f"The argument 'min_' must bigger than 'max_'.")


def _get_matching_range(
    arr: np.ndarray,
    min_: float,
    max_: float,
) -> np.ndarray:
    return np.where((min_ <= arr) & (arr <= max_))[0]


def _remove_offset_1axis(
    y: np.ndarray,
    x: np.ndarray,
    point: float = 0.,
    tol: float = 1e-6,
) -> np.ndarray:
    tol_min = point - tol
    tol_max = point + tol
    offset_indices = _get_matching_range(x, tol_min, tol_max)

    len_indices = len(offset_indices)
    if len_indices == 0:
        raise ValueError(
            "Any offsets are not found. Please check the arguments 'point' and 'tol'."
        )
    if len_indices >= 2:
        raise ValueError(
            f"Found {len_indices} candidates for the offset point. "
            "Please check the arguments 'point' and 'tol' and limit them "
            "so that only one candidate exists."
        )

    offset = y[offset_indices[0]]
    return y - offset


def change_suffix(file: t.Union[str, Path], new_suffix: str) -> str:
    if isinstance(file, str):
        file = Path(file)

    filename, _ = file.name.rsplit(".", 2)
    new_filename = ".".join((filename, new_suffix))
    return os.path.join(file.parent, new_filename)


def tie_neighbors(arr: t.Iterable[float], closeness: float) -> t.List[t.List[float]]:
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    if not len(arr):
        return []

    # Tie neighbors and make blocks of the neighbors.
    is_close = (arr[1:] - arr[:-1]) < closeness
    blocks = []
    block_neighbors = [arr[0]]
    for n, is_neighbor in zip(arr[1:], is_close):
        if is_neighbor:
            block_neighbors.append(n)
        else:
            blocks.append(block_neighbors)
            block_neighbors = [n]
    else:
        if len(block_neighbors):
            blocks.append(block_neighbors)

    return blocks


def smooth_outliers(
    arr: np.ndarray,
    threshold: float,
    closeness: float,
) -> np.ndarray:
    # Extract indices whose values are too big or too small
    # compared with the neighbors.
    indices, = np.where(np.abs(np.gradient(arr)) >= threshold)

    # Remove outliers and update the values as average values of
    # normal values in the neighborhoods.
    blocks = tie_neighbors(indices, closeness)
    for neighbors in blocks:
        index_min = min(neighbors)
        index_max = max(neighbors)

        if index_min == 0 and index_max == (len(arr) - 1):
            continue
        elif index_min == 0:
            avg_normal = arr[index_max + 1]
        elif index_max == (len(arr) - 1):
            avg_normal = arr[index_min - 1]
        else:
            avg_normal = (arr[index_min - 1] + arr[index_max + 1]) / 2

        arr[np.arange(index_min, index_max + 1)] = avg_normal

    return arr
