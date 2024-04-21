from __future__ import annotations
from functools import cached_property
from pathlib import Path
import typing as t

import numpy as np

from ._util import (
    _load_Isd,
    _load_Isd_with_Vref,
)
from ..const import (
    e,
    klitzing_constant,
)
from ..util import (
    load_raw_data,
    _check_min_max,
    _get_matching_range,
)


def _calc_density_mean(
    magneticfield: t.Union[np.ndarray, int, float],
    rho_xy_mean: np.ndarray,
) -> np.ndarray:
    return magneticfield / (e * rho_xy_mean)


def _calc_density_std(
    magneticfield: t.Union[np.ndarray, int, float],
    rho_xy_mean: np.ndarray,
    rho_xy_std: np.ndarray,
) -> np.ndarray:
    return magneticfield * rho_xy_std / (e * rho_xy_mean**2)


def _calc_mobility_mean(
    magneticfield: t.Union[np.ndarray, int, float],
    rho_xx_mean: np.ndarray,
    rho_xy_mean: np.ndarray,
) -> np.ndarray:
    return rho_xy_mean / (magneticfield * rho_xx_mean)


def _calc_mobility_std(
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


class _HallStatsData:

    def __init__(
        self,
        Vxx: np.ndarray,
        Vxy: np.ndarray,
        magneticfield: t.Union[int, float],
        Isd: np.ndarray,
        hallbar_ratio: float = 0.25,
    ) -> None:
        self._Vxx = Vxx
        self._Vxy = Vxy
        self._magneticfield = magneticfield
        self._Isd = Isd
        self._hallbar_ratio = hallbar_ratio

    @property
    def Vxx(self) -> np.ndarray:
        return self._Vxx

    @cached_property
    def Vxx_mean(self) -> np.ndarray:
        return np.mean(self.Vxx, axis=-1)

    @cached_property
    def Vxx_std(self) -> np.ndarray:
        return np.std(self.Vxx, axis=-1)

    @property
    def Vxy(self) -> np.ndarray:
        return self._Vxy

    @cached_property
    def Vxy_mean(self) -> np.ndarray:
        return np.mean(self.Vxy, axis=-1)

    @cached_property
    def Vxy_std(self) -> np.ndarray:
        return np.std(self.Vxy, axis=-1)

    @cached_property
    def magneticfield(self) -> float:
        return float(self._magneticfield)

    @property
    def hallbar_ratio(self) -> float:
        return self._hallbar_ratio

    @property
    def Isd(self) -> np.ndarray:
        return self._Isd

    @cached_property
    def Isd_mean(self) -> np.ndarray:
        return np.mean(self.Isd, axis=-1)

    @cached_property
    def Isd_std(self) -> np.ndarray:
        return np.std(self.Isd, axis=-1)

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

    @cached_property
    def density_mean(self) -> np.ndarray:
        return _calc_density_mean(self.magneticfield, self.rho_xy_mean)

    @cached_property
    def density_std(self) -> np.ndarray:
        return _calc_density_std(
            self.magneticfield,
            self.rho_xy_mean,
            self.rho_xy_std,
        )

    @cached_property
    def mobility_mean(self) -> np.ndarray:
        return _calc_mobility_mean(
            self.magneticfield,
            self.rho_xx_mean,
            self.rho_xy_mean,
        )

    @cached_property
    def mobility_std(self) -> np.ndarray:
        return _calc_mobility_std(
            self.magneticfield,
            self.rho_xx_mean,
            self.rho_xx_std,
            self.rho_xy_mean,
            self.rho_xy_std,
        )

    def crop_with(
        self,
        arr: np.ndarray,
        min_: float,
        max_: float,
    ) -> _HallStatsData:
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


class HallStats1Data(_HallStatsData):

    @staticmethod
    def load_with_Vref(
        file_Vxx: t.Union[str, Path],
        file_Vxy: t.Union[str, Path],
        magneticfield: t.Union[int, float],
        file_axis0: t.Union[str, Path],
        data_Vref: t.Union[int, float, str, Path],
        resistance_ref: t.Union[int, float],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallStats1Data:
        Vxx = load_raw_data(file_Vxx, save_npy=save_npy)
        return HallStats1Data(
            Vxx,
            load_raw_data(file_Vxy, save_npy=save_npy),
            magneticfield,
            load_raw_data(file_axis0, save_npy=save_npy),
            _load_Isd_with_Vref(
                Vxx,
                data_Vref,
                resistance_ref,
                save_npy=save_npy,
            ),
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_xx_with_Vref(
        file_Vxx: t.Union[str, Path],
        magneticfield: t.Union[int, float],
        file_axis0: t.Union[str, Path],
        data_Vref: t.Union[int, float, str, Path],
        resistance_ref: float,
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallStats1Data:
        Vxx = load_raw_data(file_Vxx, save_npy=save_npy)
        return HallStats1Data(
            Vxx,
            np.empty_like(Vxx),
            magneticfield,
            load_raw_data(file_axis0, save_npy=save_npy),
            _load_Isd_with_Vref(
                Vxx,
                data_Vref,
                resistance_ref,
                save_npy=save_npy,
            ),
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_xy_with_Vref(
        file_Vxy: t.Union[str, Path],
        magneticfield: t.Union[int, float],
        file_axis0: t.Union[str, Path],
        data_Vref: t.Union[int, float, str, Path],
        resistance_ref: float,
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallStats1Data:
        Vxy = load_raw_data(file_Vxy, save_npy=save_npy)
        return HallStats1Data(
            np.empty_like(Vxy),
            Vxy,
            magneticfield,
            load_raw_data(file_axis0, save_npy=save_npy),
            _load_Isd_with_Vref(
                Vxy,
                data_Vref,
                resistance_ref,
                save_npy=save_npy,
            ),
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_with_Isd(
        file_Vxx: t.Union[str, Path],
        file_Vxy: t.Union[str, Path],
        magneticfield: t.Union[int, float],
        file_axis0: t.Union[str, Path],
        data_Isd: t.Union[int, float, str, Path],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallStats1Data:
        Vxx = load_raw_data(file_Vxx, save_npy=save_npy)
        return HallStats1Data(
            Vxx,
            load_raw_data(file_Vxy, save_npy=save_npy),
            magneticfield,
            load_raw_data(file_axis0, save_npy=save_npy),
            _load_Isd(Vxx, data_Isd, save_npy=save_npy),
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_xx_with_Isd(
        file_Vxx: t.Union[str, Path],
        magneticfield: t.Union[int, float],
        file_axis0: t.Union[str, Path],
        data_Isd: t.Union[int, float, str, Path],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallStats1Data:
        Vxx = load_raw_data(file_Vxx, save_npy=save_npy)
        return HallStats1Data(
            Vxx,
            np.empty_like(Vxx),
            magneticfield,
            load_raw_data(file_axis0, save_npy=save_npy),
            _load_Isd(Vxx, data_Isd, save_npy=save_npy),
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_xy_with_Isd(
        file_Vxy: t.Union[str, Path],
        magneticfield: t.Union[int, float],
        file_axis0: t.Union[str, Path],
        data_Isd: t.Union[str, Path],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallStats1Data:
        Vxy = load_raw_data(file_Vxy, save_npy=save_npy)
        return HallStats1Data(
            np.empty_like(Vxy),
            Vxy,
            magneticfield,
            load_raw_data(file_axis0, save_npy=save_npy),
            _load_Isd(Vxy, data_Isd, save_npy=save_npy),
            hallbar_ratio=hallbar_ratio,
        )

    def __init__(
        self,
        Vxx: np.ndarray,
        Vxy: np.ndarray,
        magneticfield: t.Union[int, float],
        axis0: np.ndarray,
        Isd: np.ndarray,
        hallbar_ratio: float = 0.25,
    ) -> None:
        """
        Args:
            Vxx: 2-D array of Vxx.
            Vxy: 2-D array of Vxy.
            magneticfield: a constant value of magnetic field.
            axis0: 1-D array of the axis 0.
            Isd: 2-D array of source-drain current.
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
        self._axis0 = axis0

    @property
    def magneticfield(self) -> float:
        return self._magneticfield

    @property
    def axis0(self) -> np.ndarray:
        return self._axis0

    @property
    def sample_size(self) -> int:
        return np.size(self.Vxx, axis=-1)

    def crop_with(
        self,
        arr: np.ndarray,
        min_: float,
        max_: float,
    ) -> HallStats1Data:
        _check_min_max(min_, max_)
        indices = _get_matching_range(arr, min_, max_)
        return HallStats1Data(
            self.Vxx[indices, :],
            self.Vxy[indices, :],
            self.magneticfield,
            self.axis0[indices],
            self.Isd[indices, :],
            self.hallbar_ratio,
        )

    def crop_Vxx_mean(self, min_: float, max_: float) -> HallStats1Data:
        return self.crop_with(self.Vxx_mean, min_, max_)

    def crop_Vxy_mean(self, min_: float, max_: float) -> HallStats1Data:
        return self.crop_with(self.Vxy_mean, min_, max_)

    def crop_Isd_mean(self, min_: float, max_: float) -> HallStats1Data:
        return self.crop_with(self.Isd_mean, min_, max_)

    def crop_Rxx_mean(self, min_: float, max_: float) -> HallStats1Data:
        return self.crop_with(self.Rxx_mean, min_, max_)

    def crop_Rxy_mean(self, min_: float, max_: float) -> HallStats1Data:
        return self.crop_with(self.Rxy_mean, min_, max_)

    def crop_axis0(self, min_: float, max_: float) -> HallStats1Data:
        return self.crop_with(self.axis0, min_, max_)

    def calc_corrected_density_mean(
        self,
        offset: HallStats1Data,
    ) -> np.ndarray:
        return _calc_density_mean(
            self.magneticfield,
            self.rho_xy_mean - offset.rho_xy_mean,
        )

    def calc_corrected_density_std(
        self,
        offset: HallStats1Data,
    ) -> np.ndarray:
        return _calc_density_std(
            self.magneticfield,
            self.rho_xy_mean - offset.rho_xy_mean,
            np.sqrt(self.rho_xy_std**2 + offset.rho_xy_std**2),
        )

    def calc_corrected_mobility_mean(
        self,
        offset: HallStats1Data,
    ) -> np.ndarray:
        return _calc_mobility_mean(
            self.magneticfield,
            self.rho_xx_mean,
            self.rho_xy_mean - offset.rho_xy_mean,
        )

    def calc_corrected_mobility_std(
        self,
        offset: HallStats1Data,
    ) -> np.ndarray:
        return _calc_mobility_std(
            self.magneticfield,
            self.rho_xx_mean,
            self.rho_xx_std,
            self.rho_xy_mean - offset.rho_xy_mean,
            np.sqrt(self.rho_xy_std**2 + offset.rho_xy_std**2),
        )
