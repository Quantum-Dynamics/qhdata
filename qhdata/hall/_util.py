from pathlib import Path
import typing as t

import numpy as np

from ..util import load_raw_data


def _load_Isd(
    data_observed: np.ndarray,
    data_Isd: t.Union[int, float, str, Path],
    save_npy: bool = True,
) -> np.ndarray:
    """Load Isd.

    Args:
        data_observed: Array of observed data, e.g., Vxx or Vxy.
        data_Isd: Path to a file of SD current or a constant value. If a
            constant value is given, an array of SD current having same value
            of elements as the given value will be created.
        save_npy: Whether the function creates new npy files from given data.

    Returns:
        Arrays of magnetic field and SD current.
    """
    if isinstance(data_Isd, (int, float)):
        Isd = np.full_like(data_observed, data_Isd)
    else:
        Isd = load_raw_data(data_Isd, save_npy=save_npy)

    return Isd


def _load_Isd_with_Vref(
    data_observed: np.ndarray,
    data_Vref: t.Union[int, float, str, Path],
    resistance_ref: t.Union[int, float],
    save_npy: bool = True,
) -> np.ndarray:
    """Load Isd with given reference voltage and resistance.

    Args:
        data_observed: Array of observed data, e.g., Vxx or Vxy.
        data_Vref: Path to a file of reference voltages or a contant value.
            If a constant value is given, an array of SD current having same
            value of elements as `data_Vref` / `resistance_ref` will be
            created.
        resistance_ref: Value of a reference resistance.
        save_npy: Whether the function creates new npy files from given data.

    Returns:
        Arrays of magnetic field and SD current.
    """
    if resistance_ref <= 0:
        raise ValueError("'resistance_ref' must be a positive number.")

    if isinstance(data_Vref, (int, float)):
        Vref = np.full_like(data_observed, data_Vref)
    else:
        Vref = load_raw_data(data_Vref, save_npy=save_npy)

    return Vref / resistance_ref
