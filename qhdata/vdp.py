from __future__ import annotations
from functools import cached_property
from pathlib import Path
import typing as t

import numpy as np
from scipy.optimize import fsolve

from .util import (
    load_raw_data,
    _check_min_max,
    _get_matching_range,
)


def _vdP_conductivity_eq(
    conductivity: float,
    resistance_ab_cd: float,
    resistance_bc_da: float,
) -> float:
    return (
        np.exp(- np.pi * resistance_ab_cd * conductivity) +
        np.exp(- np.pi * resistance_bc_da * conductivity) - 1
    )


class Vdp1Data:

    def load_with_current(
        file_Vcd: t.Union[str, Path],
        file_Vda: t.Union[str, Path],
        file_Iab: t.Union[str, Path],
        file_Ibc: t.Union[str, Path],
        file_axis0: t.Union[str, Path],
        initial_conductivity: float = 0.,
        save_npy: bool = True,
    ) -> Vdp1Data:
        return Vdp1Data(
            load_raw_data(file_Vcd, save_npy=save_npy),
            load_raw_data(file_Vda, save_npy=save_npy),
            load_raw_data(file_Iab, save_npy=save_npy),
            load_raw_data(file_Ibc, save_npy=save_npy),
            load_raw_data(file_axis0, save_npy=save_npy),
            initial_conductivity=initial_conductivity,
        )

    def load_with_Vref(
        file_Vcd: t.Union[str, Path],
        file_Vda: t.Union[str, Path],
        file_Vref_ab: t.Union[str, Path],
        file_Vref_bc: t.Union[str, Path],
        resistance_ref_ab: t.Union[int, float],
        resistance_ref_bc: t.Union[int, float],
        file_axis0: t.Union[str, Path],
        initial_conductivity: float = 0.,
        save_npy: bool = True,
    ) -> Vdp1Data:
        if resistance_ref_ab <= 0:
            raise ValueError("'resistance_ref_ab' must be a positive number.")
        if resistance_ref_bc <= 0:
            raise ValueError("'resistance_ref_bc' must be a positive number.")

        Vref_ab = load_raw_data(file_Vref_ab, save_npy=save_npy)
        Vref_bc = load_raw_data(file_Vref_bc, save_npy=save_npy)
        return Vdp1Data(
            load_raw_data(file_Vcd, save_npy=save_npy),
            load_raw_data(file_Vda, save_npy=save_npy),
            Vref_ab / resistance_ref_ab,
            Vref_bc / resistance_ref_bc,
            load_raw_data(file_axis0, save_npy=save_npy),
            initial_conductivity=initial_conductivity,
        )

    def __init__(
        self,
        Vcd: np.ndarray,
        Vda: np.ndarray,
        Iab: np.ndarray,
        Ibc: np.ndarray,
        axis0: np.ndarray,
        initial_conductivity: float = 0.,
    ) -> None:
        self._Vcd = Vcd
        self._Vda = Vda
        self._Iab = Iab
        self._Ibc = Ibc
        self._axis0 = axis0
        self._initial_conductivity = initial_conductivity

    @property
    def Vcd(self) -> np.ndarray:
        return self._Vcd

    @property
    def Vda(self) -> np.ndarray:
        return self._Vda

    @property
    def Iab(self) -> np.ndarray:
        return self._Iab

    @property
    def Ibc(self) -> np.ndarray:
        return self._Ibc

    @cached_property
    def resistance_ab_cd(self) -> np.ndarray:
        min_current = np.min(self.Iab)
        return self.Vcd / (self.Iab + min_current * 1e-12)

    @cached_property
    def resistance_bc_da(self) -> np.ndarray:
        min_current = np.min(self.Ibc)
        return self.Vda / (self.Ibc + min_current * 1e-12)

    @property
    def axis0(self) -> np.ndarray:
        return self._axis0

    @property
    def initial_conductivity(self) -> float:
        return self._initial_conductivity

    def set_initial_conductivity(
        self,
        initial_conductivity: float,
    ) -> Vdp1Data:
        return Vdp1Data(
            self._Vcd,
            self._Vda,
            self._Iab,
            self._Ibc,
            self._axis0,
            initial_conductivity=initial_conductivity,
        )

    @cached_property
    def conductivity(self) -> np.ndarray:
        return np.array([
            fsolve(
                _vdP_conductivity_eq,
                self._initial_conductivity,
                args=res,
            )[0]
            for res in zip(self.resistance_ab_cd, self.resistance_bc_da)
        ])

    def crop_with(
        self,
        arr: np.ndarray,
        min_: float,
        max_: float,
    ) -> Vdp1Data:
        _check_min_max(min_, max_)
        indices = _get_matching_range(arr, min_, max_)
        return Vdp1Data(
            self.Vcd[indices],
            self.Vda[indices],
            self.Iab[indices],
            self.Ibc[indices],
            self.axis0[indices],
        )

    def crop_Vcd(self, min_: float, max_: float) -> Vdp1Data:
        return self.crop_with(self._Vcd, min_, max_)

    def crop_Vda(self, min_: float, max_: float) -> Vdp1Data:
        return self.crop_with(self._Vda, min_, max_)

    def crop_Iab(self, min_: float, max_: float) -> Vdp1Data:
        return self.crop_with(self._Iab, min_, max_)

    def crop_Ibc(self, min_: float, max_: float) -> Vdp1Data:
        return self.crop_with(self._Ibc, min_, max_)

    def crop_axis0(self, min_: float, max_: float) -> Vdp1Data:
        return self.crop_with(self._axis0, min_, max_)
