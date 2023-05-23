from __future__ import annotations
from functools import cached_property
from pathlib import Path
import typing as t

import numpy as np

from .const import (
    e,
    klitzing_constant,
)
from .util import (
    load_raw_data,
    _check_min_max,
    _get_matching_range,
    _remove_offset_1axis,
)


class _HallData:

    def __init__(
        self,
        Vxx: np.ndarray,
        Vxy: np.ndarray,
        magneticfield: t.Union[np.ndarray, int, float],
        Isd: np.ndarray,
        hallbar_ratio: float = 0.25,
    ) -> None:
        self._Vxx = Vxx
        self._Vxy = Vxy
        self._Isd = Isd
        self._hallbar_ratio = hallbar_ratio

        if isinstance(magneticfield, float):
            self._magneticfield = np.ones_like(self._Vxx) * magneticfield
        else:
            self._magneticfield = magneticfield

    @property
    def Vxx(self) -> np.ndarray:
        return self._Vxx

    @property
    def Vxy(self) -> np.ndarray:
        return self._Vxy

    @property
    def magneticfield(self) -> t.Union[np.ndarray, int, float]:
        return self._magneticfield

    @property
    def hallbar_ratio(self) -> float:
        return self._hallbar_ratio

    @property
    def Isd(self) -> np.ndarray:
        return self._Isd

    @cached_property
    def Rxx(self) -> np.ndarray:
        return self.Vxx / (self.Isd + 1e-15)

    @cached_property
    def rho_xx(self) -> np.ndarray:
        return self.Rxx * self.hallbar_ratio

    @cached_property
    def Rxy(self) -> np.ndarray:
        return self.Vxy / (self.Isd + 1e-15)

    @cached_property
    def rho_xy(self) -> np.ndarray:
        return self.Rxy

    @cached_property
    def Rxy_by_klitzing(self) -> np.ndarray:
        return self.Rxy / klitzing_constant

    @cached_property
    def density(self) -> np.ndarray:
        return self.magneticfield / (e * self.rho_xy)

    @cached_property
    def mobility(self) -> np.ndarray:
        return self.rho_xy / (self.magneticfield * self.rho_xx)

    def crop_with(self, arr: np.ndarray, min_: float, max_: float) -> _HallData:
        _check_min_max(min_, max_)
        indices = _get_matching_range(arr, min_, max_)

        if isinstance(self.magneticfield, (int, float)):
            magneticfield = self.magneticfield
        else:
            magneticfield = magneticfield[indices]

        return _HallData(
            self.Vxx[indices],
            self.Vxy[indices],
            magneticfield,
            self.Isd[indices],
            hallbar_ratio=self.hallbar_ratio,
        )

    def crop_Vxx(self, min_: float, max_: float) -> _HallData:
        return self.crop_with(self.Vxx, min_, max_)

    def crop_Vxy(self, min_: float, max_: float) -> _HallData:
        return self.crop_with(self.Vxy, min_, max_)

    def crop_Isd(self, min_: float, max_: float) -> _HallData:
        return self.crop_with(self.Isd, min_, max_)

    def crop_Rxx(self, min_: float, max_: float) -> _HallData:
        return self.crop_with(self.Rxx, min_, max_)

    def crop_Rxy(self, min_: float, max_: float) -> _HallData:
        return self.crop_with(self.Rxy, min_, max_)


class HallMagneticData(_HallData):

    @staticmethod
    def load_with_Vref(
        file_Vxx: t.Union[str, Path],
        file_Vxy: t.Union[str, Path],
        file_magneticfield: t.Union[str, Path],
        file_Vref: t.Union[str, Path],
        resistance_ref: t.Union[int, float],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallMagneticData:
        if resistance_ref <= 0:
            raise ValueError("'resistance_ref' must be a positive number.")

        Vref = load_raw_data(file_Vref, save_npy=save_npy)
        return HallMagneticData(
            load_raw_data(file_Vxx, save_npy=save_npy),
            load_raw_data(file_Vxy, save_npy=save_npy),
            load_raw_data(file_magneticfield, save_npy=save_npy),
            Vref / resistance_ref,
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_xx_with_Vref(
        file_Vxx: t.Union[str, Path],
        file_magneticfield: t.Union[str, Path],
        file_Vref: t.Union[str, Path],
        resistance_ref: float,
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallMagneticData:
        if resistance_ref <= 0:
            raise ValueError("'resistance_ref' must be a positive number.")

        Vref = load_raw_data(file_Vref, save_npy=save_npy)
        return HallMagneticData(
            load_raw_data(file_Vxx, save_npy=save_npy),
            np.empty_like(Vref),
            load_raw_data(file_magneticfield, save_npy=save_npy),
            Vref / resistance_ref,
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_xy_with_Vref(
        file_Vxy: t.Union[str, Path],
        file_magneticfield: t.Union[str, Path],
        file_Vref: t.Union[str, Path],
        resistance_ref: float,
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallMagneticData:
        if resistance_ref <= 0:
            raise ValueError("'resistance_ref' must be a positive number.")

        Vref = load_raw_data(file_Vref, save_npy=save_npy)
        return HallMagneticData(
            np.empty_like(Vref),
            load_raw_data(file_Vxy, save_npy=save_npy),
            load_raw_data(file_magneticfield, save_npy=save_npy),
            Vref / resistance_ref,
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_with_Isd(
        file_Vxx: t.Union[str, Path],
        file_Vxy: t.Union[str, Path],
        file_magneticfield: t.Union[str, Path],
        file_Isd: t.Union[str, Path],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallMagneticData:
        return HallMagneticData(
            load_raw_data(file_Vxx, save_npy=save_npy),
            load_raw_data(file_Vxy, save_npy=save_npy),
            load_raw_data(file_magneticfield, save_npy=save_npy),
            load_raw_data(file_Isd, save_npy=save_npy),
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_xx_with_Isd(
        file_Vxx: t.Union[str, Path],
        file_magneticfield: t.Union[str, Path],
        file_Isd: t.Union[str, Path],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallMagneticData:
        Isd = load_raw_data(file_Isd, save_npy=save_npy)
        return HallMagneticData(
            load_raw_data(file_Vxx, save_npy=save_npy),
            np.empty_like(Isd),
            load_raw_data(file_magneticfield, save_npy=save_npy),
            Isd,
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_xy_with_Isd(
        file_Vxy: t.Union[str, Path],
        file_magneticfield: t.Union[str, Path],
        file_Isd: t.Union[str, Path],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallMagneticData:
        Isd = load_raw_data(file_Isd, save_npy=save_npy)
        return HallMagneticData(
            np.empty_like(Isd),
            load_raw_data(file_Vxy, save_npy=save_npy),
            load_raw_data(file_magneticfield, save_npy=save_npy),
            Isd,
            hallbar_ratio=hallbar_ratio,
        )

    def __init__(
        self,
        Vxx: np.ndarray,
        Vxy: np.ndarray,
        magneticfield: np.ndarray,
        Isd: np.ndarray,
        hallbar_ratio: float = 0.25,
    ) -> None:
        """
        Args:
            Vxx: Ndarray for Vxx.
            Vxy: Ndarray for Vxy.
            magneticfield: Ndarray for magnetic field.
            Isd: Ndarray for source-drain current.
            hallbar_ratio: Ratio (width / length) of a Hallbar.

        Notes:
            All ndarray object must have same dimension and size.
            `hallbar_ratio` needn't be substituted if your device
            is not a Hallbar, but in this case, values of mobility
            doesn't have meaning.
        """
        super().__init__(
            Vxx,
            Vxy,
            magneticfield,
            Isd,
            hallbar_ratio=hallbar_ratio,
        )

    @property
    def magneticfield(self) -> np.ndarray:
        return self._magneticfield

    def remove_offset_xy(
        self,
        point: float = 0.,
        tol: float = 1e-6,
    ) -> HallMagneticData:
        Vxy_offset_removed = _remove_offset_1axis(
            self.Vxy,
            self.magneticfield,
            point=point,
            tol=tol,
        )
        return HallMagneticData(
            self.Vxx,
            Vxy_offset_removed,
            self.magneticfield,
            self.Isd,
        )

    def crop_with(self, arr: np.ndarray, min_: float, max_: float) -> HallMagneticData:
        _check_min_max(min_, max_)
        indices = _get_matching_range(arr, min_, max_)
        return _HallData(
            self.Vxx[indices],
            self.Vxy[indices],
            self.magneticfield[indices],
            self.Isd[indices],
            hallbar_ratio=self.hallbar_ratio,
        )

    def crop_Vxx(self, min_: float, max_: float) -> HallMagneticData:
        return self.crop_with(self.Vxx, min_, max_)

    def crop_Vxy(self, min_: float, max_: float) -> HallMagneticData:
        return self.crop_with(self.Vxy, min_, max_)

    def crop_Isd(self, min_: float, max_: float) -> HallMagneticData:
        return self.crop_with(self.Isd, min_, max_)

    def crop_Rxx(self, min_: float, max_: float) -> HallMagneticData:
        return self.crop_with(self.Rxx, min_, max_)

    def crop_Rxy(self, min_: float, max_: float) -> HallMagneticData:
        return self.crop_with(self.Rxy, min_, max_)

    def crop_magneticfield(self, min_: float, max_: float) -> HallMagneticData:
        return self.crop_with(self.magneticfield, min_, max_)


class HallGateData(_HallData):

    @staticmethod
    def load_with_Vref(
        file_Vxx: t.Union[str, Path],
        file_Vxy: t.Union[str, Path],
        magneticfield: t.Union[int, float],
        file_gate_voltage: t.Union[str, Path],
        file_Vref: t.Union[str, Path],
        resistance_ref: t.Union[int, float],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallGateData:
        if resistance_ref <= 0:
            raise ValueError("'resistance_ref' must be a positive number.")

        Vref = load_raw_data(file_Vref, save_npy=save_npy)
        return HallGateData(
            load_raw_data(file_Vxx, save_npy=save_npy),
            load_raw_data(file_Vxy, save_npy=save_npy),
            magneticfield,
            load_raw_data(file_gate_voltage, save_npy=save_npy),
            Vref / resistance_ref,
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_xx_with_Vref(
        file_Vxx: t.Union[str, Path],
        magneticfield: t.Union[int, float],
        file_gate_voltage: t.Union[str, Path],
        file_Vref: t.Union[str, Path],
        resistance_ref: float,
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallGateData:
        if resistance_ref <= 0:
            raise ValueError("'resistance_ref' must be a positive number.")

        Vref = load_raw_data(file_Vref, save_npy=save_npy)
        return HallGateData(
            load_raw_data(file_Vxx, save_npy=save_npy),
            np.empty_like(Vref),
            magneticfield,
            load_raw_data(file_gate_voltage, save_npy=save_npy),
            Vref / resistance_ref,
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_xy_with_Vref(
        file_Vxy: t.Union[str, Path],
        magneticfield: t.Union[int, float],
        file_gate_voltage: t.Union[str, Path],
        file_Vref: t.Union[str, Path],
        resistance_ref: float,
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallGateData:
        if resistance_ref <= 0:
            raise ValueError("'resistance_ref' must be a positive number.")

        Vref = load_raw_data(file_Vref, save_npy=save_npy)
        return HallGateData(
            np.empty_like(Vref),
            load_raw_data(file_Vxy, save_npy=save_npy),
            magneticfield,
            load_raw_data(file_gate_voltage, save_npy=save_npy),
            Vref / resistance_ref,
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_with_Isd(
        file_Vxx: t.Union[str, Path],
        file_Vxy: t.Union[str, Path],
        magneticfield: t.Union[int, float],
        file_gate_voltage: t.Union[str, Path],
        file_Isd: t.Union[str, Path],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallGateData:
        return HallGateData(
            load_raw_data(file_Vxx, save_npy=save_npy),
            load_raw_data(file_Vxy, save_npy=save_npy),
            magneticfield,
            load_raw_data(file_gate_voltage, save_npy=save_npy),
            load_raw_data(file_Isd, save_npy=save_npy),
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_xx_with_Isd(
        file_Vxx: t.Union[str, Path],
        magneticfield: t.Union[int, float],
        file_gate_voltage: t.Union[str, Path],
        file_Isd: t.Union[str, Path],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallGateData:
        Isd = load_raw_data(file_Isd, save_npy=save_npy)
        return HallGateData(
            load_raw_data(file_Vxx, save_npy=save_npy),
            np.empty_like(Isd),
            magneticfield,
            load_raw_data(file_gate_voltage, save_npy=save_npy),
            Isd,
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_xy_with_Isd(
        file_Vxy: t.Union[str, Path],
        magneticfield: t.Union[int, float],
        file_gate_voltage: t.Union[str, Path],
        file_Isd: t.Union[str, Path],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallGateData:
        Isd = load_raw_data(file_Isd, save_npy=save_npy)
        return HallGateData(
            np.empty_like(Isd),
            load_raw_data(file_Vxy, save_npy=save_npy),
            magneticfield,
            load_raw_data(file_gate_voltage, save_npy=save_npy),
            Isd,
            hallbar_ratio=hallbar_ratio,
        )

    def __init__(
        self,
        Vxx: np.ndarray,
        Vxy: np.ndarray,
        magneticfield: t.Union[int, float],
        gate_voltage: np.ndarray,
        Isd: np.ndarray,
        hallbar_ratio: float = 0.25,
    ) -> None:
        """
        Args:
            Vxx: Ndarray for Vxx.
            Vxy: Ndarray for Vxy.
            magneticfield: Ndarray for magnetic field.
            Isd: Ndarray for source-drain current.
            hallbar_ratio: Ratio (width / length) of a Hallbar.

        Notes:
            All ndarray object must have same dimension and size.
            `hallbar_ratio` needn't be substituted if your device
            is not a Hallbar, but in this case, values of mobility
            doesn't have meaning.
        """
        super().__init__(
            Vxx,
            Vxy,
            magneticfield,
            Isd,
            hallbar_ratio=hallbar_ratio,
        )
        self._gate_voltage = gate_voltage

    @property
    def magneticfield(self) -> float:
        return self._magneticfield

    @property
    def gate_voltage(self) -> np.ndarray:
        return self._gate_voltage

    def crop_with(self, arr: np.ndarray, min_: float, max_: float) -> HallGateData:
        _check_min_max(min_, max_)
        indices = _get_matching_range(arr, min_, max_)
        return HallGateData(
            self.Vxx[indices],
            self.Vxy[indices],
            self.magneticfield,
            self.gate_voltage[indices],
            self.Isd[indices],
            self.hallbar_ratio,
        )

    def crop_Vxx(self, min_: float, max_: float) -> HallGateData:
        return self.crop_with(self.Vxx, min_, max_)

    def crop_Vxy(self, min_: float, max_: float) -> HallGateData:
        return self.crop_with(self.Vxy, min_, max_)

    def crop_Isd(self, min_: float, max_: float) -> HallGateData:
        return self.crop_with(self.Isd, min_, max_)

    def crop_Rxx(self, min_: float, max_: float) -> HallGateData:
        return self.crop_with(self.Rxx, min_, max_)

    def crop_Rxy(self, min_: float, max_: float) -> HallGateData:
        return self.crop_with(self.Rxy, min_, max_)

    def crop_gate_voltage(self, min_: float, max_: float) -> HallGateData:
        return self.crop_with(self.gate_voltage, min_, max_)

    def calc_corrected_density(self, offset: HallGateData) -> np.ndarray:
        return self.magneticfield / (e * (self.rho_xy - offset.rho_xy))

    def calc_corrected_mobility(self, offset: HallGateData) -> np.ndarray:
        return (
            (self.rho_xy - offset.rho_xy) /
            (self.magneticfield * self.rho_xx)
        )


class _HallStatsData:

    def __init__(
        self,
        Vxx: np.ndarray,
        Vxy: np.ndarray,
        magneticfield: t.Union[np.ndarray, int, float],
        Isd: np.ndarray,
        hallbar_ratio: float = 0.25,
    ) -> None:
        self._Vxx = Vxx
        self._Vxy = Vxy
        self._Isd = Isd
        self._hallbar_ratio = hallbar_ratio

        if isinstance(magneticfield, (int, float)):
            self._magneticfield = np.ones_like(self._Vxx) * magneticfield
        else:
            self._magneticfield = magneticfield

    @property
    def Vxx(self) -> np.ndarray:
        return self._Vxx

    @cached_property
    def Vxx_mean(self) -> np.ndarray:
        return np.mean(self.Vxx, axis=1)

    @cached_property
    def Vxx_std(self) -> np.ndarray:
        return np.std(self.Vxx, axis=1)

    @property
    def Vxy(self) -> np.ndarray:
        return self._Vxy

    @cached_property
    def Vxy_mean(self) -> np.ndarray:
        return np.mean(self.Vxy, axis=1)

    @cached_property
    def Vxy_std(self) -> np.ndarray:
        return np.std(self.Vxy, axis=1)

    @property
    def magneticfield(self) -> t.Union[np.ndarray, int, float]:
        return self._magneticfield

    @property
    def hallbar_ratio(self) -> float:
        return self._hallbar_ratio

    @property
    def Isd(self) -> np.ndarray:
        return self._Isd

    @cached_property
    def Isd_mean(self) -> np.ndarray:
        return np.mean(self.Isd, axis=1)

    @cached_property
    def Isd_std(self) -> np.ndarray:
        return np.std(self.Isd, axis=1)

    @cached_property
    def Rxx_mean(self) -> np.ndarray:
        return self.Vxx_mean / self.Isd_mean

    @cached_property
    def Rxx_std(self) -> np.ndarray:
        return np.sqrt(
            (self.Vxx_std / self.Isd_mean)**2 +
            ((self.Vxx_mean / self.Isd_mean**2) * self.Isd_std)**2
        )

    @cached_property
    def rho_xx_mean(self) -> np.ndarray:
        return self.Rxx_mean * self.hallbar_ratio

    @cached_property
    def rho_xx_std(self) -> np.ndarray:
        return self.Rxx_std * self.hallbar_ratio

    @cached_property
    def Rxy_mean(self) -> np.ndarray:
        return self.Vxy_mean / self.Isd_mean

    @cached_property
    def Rxy_std(self) -> np.ndarray:
        return np.sqrt(
            (self.Vxy_std / self.Isd_mean)**2 +
            ((self.Vxy_mean / self.Isd_mean**2) * self.Isd_std)**2
        )

    @cached_property
    def rho_xy_mean(self) -> np.ndarray:
        return self.Rxy_mean

    @cached_property
    def rho_xy_std(self) -> np.ndarray:
        return self.Rxx_std

    @cached_property
    def Rxy_mean_by_klizing(self) -> np.ndarray:
        return self.Rxy_mean / klitzing_constant

    @cached_property
    def Rxy_std_by_klizing(self) -> np.ndarray:
        return self.Rxy_std / klitzing_constant

    @staticmethod
    def calc_density_mean(
        magneticfield: t.Union[np.ndarray, int, float],
        rho_xy_mean: np.ndarray,
    ) -> np.ndarray:
        return magneticfield / (e * rho_xy_mean)

    @cached_property
    def density_mean(self) -> np.ndarray:
        return self.calc_density_mean(self.magneticfield, self.rho_xy_mean)

    @staticmethod
    def calc_density_std(
        magneticfield: t.Union[np.ndarray, int, float],
        rho_xy_mean: np.ndarray,
        rho_xy_std: np.ndarray,
    ) -> np.ndarray:
        return magneticfield * rho_xy_std / (e * rho_xy_mean**2)

    @cached_property
    def density_std(self) -> np.ndarray:
        return self.calc_density_std(
            self.magneticfield,
            self.rho_xy_mean,
            self.rho_xy_std,
        )

    @staticmethod
    def calc_mobility_mean(
        magneticfield: t.Union[np.ndarray, int, float],
        rho_xx_mean: np.ndarray,
        rho_xy_mean: np.ndarray,
    ) -> np.ndarray:
        return rho_xy_mean / (magneticfield * rho_xx_mean)

    @cached_property
    def mobility_mean(self) -> np.ndarray:
        return self.calc_mobility_mean(
            self.magneticfield,
            self.rho_xx_mean,
            self.rho_xy_mean,
        )

    @staticmethod
    def calc_mobility_std(
        magneticfield: t.Union[np.ndarray, int, float],
        rho_xx_mean: np.ndarray,
        rho_xx_std: np.ndarray,
        rho_xy_mean: np.ndarray,
        rho_xy_std: np.ndarray,
    ) -> np.ndarray:
        return np.sqrt(
            (rho_xy_std / (magneticfield * rho_xx_mean))**2 +
            (rho_xy_mean * rho_xx_std / (magneticfield * rho_xx_mean**2))**2
        )

    @cached_property
    def mobility_std(self) -> np.ndarray:
        return self.calc_mobility_std(
            self.magneticfield,
            self.rho_xx_mean,
            self.rho_xx_std,
            self.rho_xy_mean,
            self.rho_xy_std,
        )

    def crop_with(self, arr: np.ndarray, min_: float, max_: float) -> _HallStatsData:
        _check_min_max(min_, max_)
        indices = _get_matching_range(arr, min_, max_)

        if isinstance(self.magneticfield, (int, float)):
            magneticfield = self.magneticfield
        else:
            magneticfield = self.magneticfield[indices]

        return _HallStatsData(
            self.Vxx[indices, :],
            self.Vxy[indices, :],
            magneticfield,
            self.Isd[indices, :],
            hallbar_ratio=self.hallbar_ratio,
        )

    def crop_Vxx_mean(self, min_: float, max_: float) -> _HallStatsData:
        return self.crop_with(self.Vxx_mean, min_, max_)

    def crop_Vxy_mean(self, min_: float, max_: float) -> _HallStatsData:
        return self.crop_with(self.Vxy_mean, min_, max_)

    def crop_Isd_mean(self, min_: float, max_: float) -> _HallStatsData:
        return self.crop_with(self.Isd_mean, min_, max_)

    def crop_Rxx_mean(self, min_: float, max_: float) -> _HallStatsData:
        return self.crop_with(self.Rxx_mean, min_, max_)

    def crop_Rxy_mean(self, min_: float, max_: float) -> _HallStatsData:
        return self.crop_with(self.Rxy_mean, min_, max_)


class HallMagneticStatsData(_HallStatsData):

    @staticmethod
    def load_with_Vref(
        file_Vxx: t.Union[str, Path],
        file_Vxy: t.Union[str, Path],
        file_magneticfield: t.Union[str, Path],
        file_Vref: t.Union[str, Path],
        resistance_ref: t.Union[int, float],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallMagneticData:
        if resistance_ref <= 0:
            raise ValueError("'resistance_ref' must be a positive number.")

        Vref = load_raw_data(file_Vref, save_npy=save_npy)
        return HallMagneticData(
            load_raw_data(file_Vxx, save_npy=save_npy),
            load_raw_data(file_Vxy, save_npy=save_npy),
            load_raw_data(file_magneticfield, save_npy=save_npy),
            Vref / resistance_ref,
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_xx_with_Vref(
        file_Vxx: t.Union[str, Path],
        file_magneticfield: t.Union[str, Path],
        file_Vref: t.Union[str, Path],
        resistance_ref: float,
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallMagneticData:
        if resistance_ref <= 0:
            raise ValueError("'resistance_ref' must be a positive number.")

        Vref = load_raw_data(file_Vref, save_npy=save_npy)
        return HallMagneticData(
            load_raw_data(file_Vxx, save_npy=save_npy),
            np.empty_like(Vref),
            load_raw_data(file_magneticfield, save_npy=save_npy),
            Vref / resistance_ref,
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_xy_with_Vref(
        file_Vxy: t.Union[str, Path],
        file_magneticfield: t.Union[str, Path],
        file_Vref: t.Union[str, Path],
        resistance_ref: float,
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallMagneticData:
        if resistance_ref <= 0:
            raise ValueError("'resistance_ref' must be a positive number.")

        Vref = load_raw_data(file_Vref, save_npy=save_npy)
        return HallMagneticData(
            np.empty_like(Vref),
            load_raw_data(file_Vxy, save_npy=save_npy),
            load_raw_data(file_magneticfield, save_npy=save_npy),
            Vref / resistance_ref,
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_with_Isd(
        file_Vxx: t.Union[str, Path],
        file_Vxy: t.Union[str, Path],
        file_magneticfield: t.Union[str, Path],
        file_Isd: t.Union[str, Path],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallMagneticData:
        return HallMagneticData(
            load_raw_data(file_Vxx, save_npy=save_npy),
            load_raw_data(file_Vxy, save_npy=save_npy),
            load_raw_data(file_magneticfield, save_npy=save_npy),
            load_raw_data(file_Isd, save_npy=save_npy),
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_xx_with_Isd(
        file_Vxx: t.Union[str, Path],
        file_magneticfield: t.Union[str, Path],
        file_Isd: t.Union[str, Path],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallMagneticData:
        Isd = load_raw_data(file_Isd, save_npy=save_npy)
        return HallMagneticData(
            load_raw_data(file_Vxx, save_npy=save_npy),
            np.empty_like(Isd),
            load_raw_data(file_magneticfield, save_npy=save_npy),
            Isd,
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_xy_with_Isd(
        file_Vxy: t.Union[str, Path],
        file_magneticfield: t.Union[str, Path],
        file_Isd: t.Union[str, Path],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallMagneticData:
        Isd = load_raw_data(file_Isd, save_npy=save_npy)
        return HallMagneticData(
            np.empty_like(Isd),
            load_raw_data(file_Vxy, save_npy=save_npy),
            load_raw_data(file_magneticfield, save_npy=save_npy),
            Isd,
            hallbar_ratio=hallbar_ratio,
        )

    def __init__(
        self,
        Vxx: np.ndarray,
        Vxy: np.ndarray,
        magneticfield: np.ndarray,
        Isd: np.ndarray,
        hallbar_ratio: float = 0.25,
    ) -> None:
        super().__init__(
            Vxx,
            Vxy,
            magneticfield,
            Isd,
            hallbar_ratio=hallbar_ratio,
        )

    @property
    def magneticfield(self) -> np.ndarray:
        return self._magneticfield

    def crop_with(self, arr: np.ndarray, min_: float, max_: float) -> HallMagneticStatsData:
        _check_min_max(min_, max_)
        indices = _get_matching_range(arr, min_, max_)
        return HallMagneticStatsData(
            self.Vxx[indices, :],
            self.Vxy[indices, :],
            self.magneticfield[indices],
            self.Isd[indices, :],
            hallbar_ratio=self.hallbar_ratio,
        )

    def crop_Vxx_mean(self, min_: float, max_: float) -> HallMagneticStatsData:
        return self.crop_with(self.Vxx_mean, min_, max_)

    def crop_Vxy_mean(self, min_: float, max_: float) -> HallMagneticStatsData:
        return self.crop_with(self.Vxy_mean, min_, max_)

    def crop_Isd_mean(self, min_: float, max_: float) -> HallMagneticStatsData:
        return self.crop_with(self.Isd_mean, min_, max_)

    def crop_Rxx_mean(self, min_: float, max_: float) -> HallMagneticStatsData:
        return self.crop_with(self.Rxx_mean, min_, max_)

    def crop_Rxy_mean(self, min_: float, max_: float) -> HallMagneticStatsData:
        return self.crop_with(self.Rxy_mean, min_, max_)


class HallGateStatsData(_HallStatsData):

    @staticmethod
    def load_with_Vref(
        file_Vxx: t.Union[str, Path],
        file_Vxy: t.Union[str, Path],
        magneticfield: t.Union[int, float],
        file_gate_voltage: t.Union[str, Path],
        file_Vref: t.Union[str, Path],
        resistance_ref: t.Union[int, float],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallGateData:
        if resistance_ref <= 0:
            raise ValueError("'resistance_ref' must be a positive number.")

        Vref = load_raw_data(file_Vref, save_npy=save_npy)
        return HallGateData(
            load_raw_data(file_Vxx, save_npy=save_npy),
            load_raw_data(file_Vxy, save_npy=save_npy),
            magneticfield,
            load_raw_data(file_gate_voltage, save_npy=save_npy),
            Vref / resistance_ref,
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_xx_with_Vref(
        file_Vxx: t.Union[str, Path],
        magneticfield: t.Union[int, float],
        file_gate_voltage: t.Union[str, Path],
        file_Vref: t.Union[str, Path],
        resistance_ref: float,
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallGateData:
        if resistance_ref <= 0:
            raise ValueError("'resistance_ref' must be a positive number.")

        Vref = load_raw_data(file_Vref, save_npy=save_npy)
        return HallGateData(
            load_raw_data(file_Vxx, save_npy=save_npy),
            np.empty_like(Vref),
            magneticfield,
            load_raw_data(file_gate_voltage, save_npy=save_npy),
            Vref / resistance_ref,
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_xy_with_Vref(
        file_Vxy: t.Union[str, Path],
        magneticfield: t.Union[int, float],
        file_gate_voltage: t.Union[str, Path],
        file_Vref: t.Union[str, Path],
        resistance_ref: float,
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallGateData:
        if resistance_ref <= 0:
            raise ValueError("'resistance_ref' must be a positive number.")

        Vref = load_raw_data(file_Vref, save_npy=save_npy)
        return HallGateData(
            np.empty_like(Vref),
            load_raw_data(file_Vxy, save_npy=save_npy),
            magneticfield,
            load_raw_data(file_gate_voltage, save_npy=save_npy),
            Vref / resistance_ref,
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_with_Isd(
        file_Vxx: t.Union[str, Path],
        file_Vxy: t.Union[str, Path],
        magneticfield: t.Union[int, float],
        file_gate_voltage: t.Union[str, Path],
        file_Isd: t.Union[str, Path],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallGateData:
        return HallGateData(
            load_raw_data(file_Vxx, save_npy=save_npy),
            load_raw_data(file_Vxy, save_npy=save_npy),
            magneticfield,
            load_raw_data(file_gate_voltage, save_npy=save_npy),
            load_raw_data(file_Isd, save_npy=save_npy),
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_xx_with_Isd(
        file_Vxx: t.Union[str, Path],
        magneticfield: t.Union[int, float],
        file_gate_voltage: t.Union[str, Path],
        file_Isd: t.Union[str, Path],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallGateData:
        Isd = load_raw_data(file_Isd, save_npy=save_npy)
        return HallGateData(
            load_raw_data(file_Vxx, save_npy=save_npy),
            np.empty_like(Isd),
            magneticfield,
            load_raw_data(file_gate_voltage, save_npy=save_npy),
            Isd,
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_xy_with_Isd(
        file_Vxy: t.Union[str, Path],
        magneticfield: t.Union[int, float],
        file_gate_voltage: t.Union[str, Path],
        file_Isd: t.Union[str, Path],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallGateData:
        Isd = load_raw_data(file_Isd, save_npy=save_npy)
        return HallGateData(
            np.empty_like(Isd),
            load_raw_data(file_Vxy, save_npy=save_npy),
            magneticfield,
            load_raw_data(file_gate_voltage, save_npy=save_npy),
            Isd,
            hallbar_ratio=hallbar_ratio,
        )

    def __init__(
        self,
        Vxx: np.ndarray,
        Vxy: np.ndarray,
        magneticfield: t.Union[int, float],
        gate_voltage: np.ndarray,
        Isd: np.ndarray,
        hallbar_ratio: float = 0.25,
    ) -> None:
        """
        Args:
            Vxx: Ndarray for Vxx.
            Vxy: Ndarray for Vxy.
            magneticfield: Ndarray for magnetic field.
            Isd: Ndarray for source-drain current.
            hallbar_ratio: Ratio (width / length) of a Hallbar.

        Notes:
            All ndarray object must have same dimension and size.
            `hallbar_ratio` needn't be substituted if your device
            is not a Hallbar, but in this case, values of mobility
            doesn't have meaning.
        """
        super().__init__(
            Vxx,
            Vxy,
            float(magneticfield),
            Isd,
            hallbar_ratio=hallbar_ratio,
        )
        self._gate_voltage = gate_voltage

    @property
    def magneticfield(self) -> float:
        return self._magneticfield

    @property
    def gate_voltage(self) -> np.ndarray:
        return self._gate_voltage

    def crop_with(self, arr: np.ndarray, min_: float, max_: float) -> HallGateStatsData:
        _check_min_max(min_, max_)
        indices = _get_matching_range(arr, min_, max_)
        return HallGateStatsData(
            self.Vxx[indices, :],
            self.Vxy[indices, :],
            self.magneticfield,
            self.gate_voltage[indices],
            self.Isd[indices, :],
            self.hallbar_ratio,
        )

    def crop_Vxx_mean(self, min_: float, max_: float) -> HallGateStatsData:
        return self.crop_with(self.Vxx_mean, min_, max_)

    def crop_Vxy_mean(self, min_: float, max_: float) -> HallGateStatsData:
        return self.crop_with(self.Vxy_mean, min_, max_)

    def crop_Isd_mean(self, min_: float, max_: float) -> HallGateStatsData:
        return self.crop_with(self.Isd_mean, min_, max_)

    def crop_Rxx_mean(self, min_: float, max_: float) -> HallGateStatsData:
        return self.crop_with(self.Rxx_mean, min_, max_)

    def crop_Rxy_mean(self, min_: float, max_: float) -> HallGateStatsData:
        return self.crop_with(self.Rxy_mean, min_, max_)

    def crop_gate_voltage(self, min_: float, max_: float) -> HallGateStatsData:
        return self.crop_with(self.gate_voltage, min_, max_)

    def calc_corrected_density_mean(self, offset: HallGateStatsData) -> np.ndarray:
        return self.calc_density_mean(
            self.magneticfield,
            self.rho_xy_mean - offset.rho_xy_mean,
        )

    def calc_corrected_density_std(self, offset: HallGateStatsData) -> np.ndarray:
        return self.calc_density_std(
            self.magneticfield,
            self.rho_xy_mean - offset.rho_xy_mean,
            np.sqrt(self.rho_xy_std**2 + offset.rho_xy_std**2),
        )

    def calc_corrected_mobility_mean(self, offset: HallGateStatsData) -> np.ndarray:
        return self.calc_mobility_mean(
            self.magneticfield,
            self.rho_xx_mean,
            self.rho_xy_mean - offset.rho_xy_mean,
        )

    def calc_corrected_mobility_std(self, offset: HallGateStatsData) -> np.ndarray:
        return self.calc_mobility_std(
            self.magneticfield,
            self.rho_xx_mean,
            self.rho_xx_std,
            self.rho_xy_mean - offset.rho_xy_mean,
            np.sqrt(self.rho_xy_std**2 + offset.rho_xy_std**2),
        )
