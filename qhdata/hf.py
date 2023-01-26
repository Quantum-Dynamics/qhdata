from __future__ import annotations
from functools import cached_property
from pathlib import Path
import typing as t

import numpy as np

from .util import (
    load_raw_data,
    _check_min_max,
    _get_matching_range,
)


class RSAData:

    @staticmethod
    def load(
        file_freq: t.Union[str, Path],
        file_rsa_dbm: t.Union[str, Path],
        file_rsa_freq: t.Union[str, Path],
        save_npy: bool = True,
    ) -> RSAData:
        return RSAData(
            load_raw_data(file_freq, save_npy=save_npy),
            load_raw_data(file_rsa_dbm, save_npy=save_npy),
            load_raw_data(file_rsa_freq, save_npy=save_npy),
        )

    def __init__(self, freq: np.ndarray, rsa_dbm: np.ndarray, rsa_freq: np.ndarray) -> None:
        self._freq = freq
        self._rsa_dbm = rsa_dbm
        self._rsa_freq = rsa_freq

    @property
    def freq(self) -> np.ndarray:
        return self._freq

    @property
    def rsa_dbm(self) -> np.ndarray:
        return self._rsa_dbm

    @property
    def rsa_freq(self) -> np.ndarray:
        return self._rsa_freq

    @cached_property
    def max_dbm(self) -> np.ndarray:
        return np.max(self.rsa_dbm, axis=1)

    @cached_property
    def max_freq(self) -> np.ndarray:
        return np.array([
            self._rsa_freq[i, np.where(self._rsa_dbm[i, :] == self.max_dbm[i])][0]
            for i in range(len(self._rsa_freq))
        ])

    @property
    def num_freq(self) -> int:
        return len(self._freq)

    @property
    def num_span(self) -> int:
        return self._rsa_dbm.shape[1]

    def crop_freq(self, min_: float, max_: float) -> RSAData:
        _check_min_max(min_, max_)
        indices = _get_matching_range(self._freq, min_, max_)
        return RSAData(self.freq[indices], self.rsa_dbm[indices, :], self.rsa_freq[indices, :])
