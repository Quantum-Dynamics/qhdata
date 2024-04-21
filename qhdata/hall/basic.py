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
        self._magneticfield = magneticfield
        self._Isd = Isd
        self._hallbar_ratio = hallbar_ratio

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


class HallMagneticData(_HallData):
    """Data of Hall measurements sweeping magnetic field."""

    @staticmethod
    def load_with_Vref(
        file_Vxx: t.Union[str, Path],
        file_Vxy: t.Union[str, Path],
        file_magneticfield: t.Union[str, Path],
        data_Vref: t.Union[int, float, str, Path],
        resistance_ref: t.Union[int, float],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallMagneticData:
        """Load data with reference voltages that is to be converted to SD
        currents.

        Args:
            file_Vxx: Path to a file of Vxx.
            file_Vxy: Path to a file of Vxy.
            file_magneticfield: Path to a file of magnetic field.
            data_Vref: Path to a file of reference voltages or a contant
                value. If a constant value is given, an array of SD current
                having same value of elements as `data_Vref` /
                `resistance_ref` will be created.
            resistance_ref: Value of a reference resistance.
            hallbar_ratio: Ratio (width / length) of a Hallbar. This parameter
                is valid only when the device is a Hall bar.
            save_npy: Whether the function creates new npy files from given
                data.

        Returns:
            `HallMagneticData` with loaded data.
        """
        Vxx = load_raw_data(file_Vxx, save_npy=save_npy)
        return HallMagneticData(
            Vxx,
            load_raw_data(file_Vxy, save_npy=save_npy),
            load_raw_data(file_magneticfield, save_npy=save_npy),
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
        file_magneticfield: t.Union[str, Path],
        data_Vref: t.Union[int, float, str, Path],
        resistance_ref: float,
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallMagneticData:
        """Load only Vxx data with reference voltages that is to be converted
        to SD currents.

        Args:
            file_Vxx: Path to a file of Vxx.
            file_magneticfield: Path to a file of magnetic field.
            data_Vref: Path to a file of reference voltages or a contant
                value. If a constant value is given, an array of SD current
                having same value of elements as `data_Vref` /
                `resistance_ref` will be created.
            resistance_ref: Value of a reference resistance.
            hallbar_ratio: Ratio (width / length) of a Hallbar. This parameter
                is valid only when the device is a Hall bar.
            save_npy: Whether the function creates new npy files from given
                data.

        Returns:
            `HallMagneticData` with loaded data.

        Notes:
            Returned `HallMagneticData` has Vxy data which can be referenced
            at `HallMagneticData.Vxy`, but this arrray has no meaning data.
            Thus Vxy data should not be referenced for analysis.
        """
        Vxx = load_raw_data(file_Vxx, save_npy=save_npy)
        return HallMagneticData(
            Vxx,
            np.empty_like(Vxx),
            load_raw_data(file_magneticfield, save_npy=save_npy),
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
        file_magneticfield: t.Union[str, Path],
        data_Vref: t.Union[int, float, str, Path],
        resistance_ref: float,
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallMagneticData:
        """Load only Vxy data with reference voltages that is to be converted
        to SD currents.

        Args:
            file_Vxy: Path to a file of Vxx.
            file_magneticfield: Path to a file of magnetic field.
            data_Vref: Path to a file of reference voltages or a contant
                value. If a constant value is given, an array of SD current
                having same value of elements as `data_Vref` /
                `resistance_ref` will be created.
            resistance_ref: Value of a reference resistance.
            hallbar_ratio: Ratio (width / length) of a Hallbar. This parameter
                is valid only when the device is a Hall bar.
            save_npy: Whether the function creates new npy files from given
                data.

        Returns:
            `HallMagneticData` with loaded data.

        Notes:
            Returned `HallMagneticData` has Vxx data which can be referenced
            at `HallMagneticData.Vxx`, but this arrray has no meaning data.
            Thus Vxx data should not be referenced for analysis.
        """
        Vxy = load_raw_data(file_Vxy, save_npy=save_npy)
        return HallMagneticData(
            np.empty_like(Vxy),
            Vxy,
            load_raw_data(file_magneticfield, save_npy=save_npy),
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
        file_magneticfield: t.Union[str, Path],
        data_Isd: t.Union[int, float, str, Path],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallMagneticData:
        """Load data with SD currents.

        Args:
            file_Vxx: Path to a file of Vxx.
            file_Vxy: Path to a file of Vxx.
            file_magneticfield: Path to a file of magnetic field.
            data_Isd: Path to a file of SD current or a constant value.
                If a constant value is given, an array of SD current having
                same value of elements as the given value will be created.
            hallbar_ratio: Ratio (width / length) of a Hallbar. This parameter
                is valid only when the device is a Hall bar.
            save_npy: Whether the function creates new npy files from given
                data.

        Returns:
            `HallMagneticData` with loaded data.
        """
        Vxx = load_raw_data(file_Vxx, save_npy=save_npy)
        return HallMagneticData(
            Vxx,
            load_raw_data(file_Vxy, save_npy=save_npy),
            load_raw_data(file_magneticfield, save_npy=save_npy),
            _load_Isd(Vxx, data_Isd, save_npy=save_npy),
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_xx_with_Isd(
        file_Vxx: t.Union[str, Path],
        file_magneticfield: t.Union[str, Path],
        data_Isd: t.Union[int, float, str, Path],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallMagneticData:
        """Load only Vxx data with SD currents.

        Args:
            file_Vxx: Path to a file of Vxx.
            file_magneticfield: Path to a file of magnetic field.
            data_Isd: Path to a file of SD current or a constant value.
                If a constant value is given, an array of SD current having
                same value of elements as the given value will be created.
            hallbar_ratio: Ratio (width / length) of a Hallbar. This parameter
                is valid only when the device is a Hall bar.
            save_npy: Whether the function creates new npy files from given
                data.

        Returns:
            `HallMagneticData` with loaded data.

        Notes:
            Returned `HallMagneticData` has Vxy data which can be referenced
            at `HallMagneticData.Vxy`, but this arrray has no meaning data.
            Thus Vxy data should not be referenced for analysis.
        """
        Vxx = load_raw_data(file_Vxx, save_npy=save_npy)
        return HallMagneticData(
            Vxx,
            np.empty_like(Vxx),
            load_raw_data(file_magneticfield, save_npy=save_npy),
            _load_Isd(Vxx, data_Isd, save_npy=save_npy),
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_xy_with_Isd(
        file_Vxy: t.Union[str, Path],
        file_magneticfield: t.Union[str, Path],
        data_Isd: t.Union[int, float, str, Path],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> HallMagneticData:
        """Load only Vxy data with SD currents.

        Args:
            file_Vxy: Path to a file of Vxy.
            file_magneticfield: Path to a file of magnetic field.
            data_Isd: Path to a file of SD current or a constant value.
                If a constant value is given, an array of SD current having
                same value of elements as the given value will be created.
            hallbar_ratio: Ratio (width / length) of a Hallbar. This parameter
                is valid only when the device is a Hall bar.
            save_npy: Whether the function creates new npy files from given
                data.

        Returns:
            `HallMagneticData` with loaded data.

        Notes:
            Returned `HallMagneticData` has Vxx data which can be referenced
            at `HallMagneticData.Vxx`, but this arrray has no meaning data.
            Thus Vxx data should not be referenced for analysis.
        """
        Vxy = load_raw_data(file_Vxy, save_npy=save_npy)
        return HallMagneticData(
            np.empty_like(Vxy),
            Vxy,
            load_raw_data(file_magneticfield, save_npy=save_npy),
            _load_Isd(Vxy, data_Isd, save_npy=save_npy),
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
        """Magnetic field in the measurement."""
        return self._magneticfield

    def remove_offset_xy(
        self,
        point: float = 0.,
        tol: float = 1e-6,
    ) -> HallMagneticData:
        """Remove offset of Vxy.

        This method removes offset of Vxy at B = 0(T). This method can be used
        when you want to eliminate the effect of residual magnetic field.

        Args:
            point: Magnetic field where offset exists. This value is zero in
                general.
            tol: Absolute tolerance of `point`. This method find the actual
                offset point in the region [`point` - `tol`, `point` + `tol`].

        Returns:
            New `HallMagneticData` with Vxy whose offset is removed.
        """
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

    def crop_with(
        self,
        arr: np.ndarray,
        min_: float,
        max_: float,
    ) -> HallMagneticData:
        _check_min_max(min_, max_)
        indices = _get_matching_range(arr, min_, max_)
        return HallMagneticData(
            self.Vxx[indices],
            self.Vxy[indices],
            self.magneticfield[indices],
            self.Isd[indices],
            hallbar_ratio=self.hallbar_ratio,
        )

    def crop_Vxx(self, min_: float, max_: float) -> HallMagneticData:
        """Crop data with respect to Vxx.

        This method crops Vxx data included in the region [`min_`, `max_`].
        Other data like Vxy also is to be cropped so that they have same shape
        of Vxx.

        Args:
            min_: minuite value of the crop window.
            max_: max value of the crop window.

        Returns:
            New `HallMagneticData` with
        """
        return self.crop_with(self.Vxx, min_, max_)

    def crop_Vxy(self, min_: float, max_: float) -> HallMagneticData:
        """Crop data with respect to Vxy.

        This method crops Vxy data included in the region [`min_`, `max_`].
        Other data like Vxx also is to be cropped so that they have same shape
        of Vxy.

        Args:
            min_: minuite value of the crop window.
            max_: max value of the crop window.

        Returns:
            New `HallMagneticData` with
        """
        return self.crop_with(self.Vxy, min_, max_)

    def crop_Isd(self, min_: float, max_: float) -> HallMagneticData:
        """Crop data with respect to Isd.

        This method crops Isd data included in the region [`min_`, `max_`].
        Other data like Vxy also is to be cropped so that they have same shape
        of Isd.

        Args:
            min_: minuite value of the crop window.
            max_: max value of the crop window.

        Returns:
            New `HallMagneticData` with
        """
        return self.crop_with(self.Isd, min_, max_)

    def crop_Rxx(self, min_: float, max_: float) -> HallMagneticData:
        """Crop data with respect to Rxx.

        This method crops Rxx data included in the region [`min_`, `max_`].
        Other data like Rxy also is to be cropped so that they have same shape
        of Rxx.

        Args:
            min_: minuite value of the crop window.
            max_: max value of the crop window.

        Returns:
            New `HallMagneticData` with
        """
        return self.crop_with(self.Rxx, min_, max_)

    def crop_Rxy(self, min_: float, max_: float) -> HallMagneticData:
        """Crop data with respect to Rxy.

        This method crops Rxy data included in the region [`min_`, `max_`].
        Other data like Rxx also is to be cropped so that they have same shape
        of Rxy.

        Args:
            min_: minuite value of the crop window.
            max_: max value of the crop window.

        Returns:
            New `HallMagneticData` with
        """
        return self.crop_with(self.Rxy, min_, max_)

    def crop_magneticfield(self, min_: float, max_: float) -> HallMagneticData:
        """Crop data with respect to Vxx.

        This method crops Vxx data included in the region [`min_`, `max_`].
        Other data like Vxy also is to be cropped so that they have same shape
        of Vxx.

        Args:
            min_: minuite value of the crop window.
            max_: max value of the crop window.

        Returns:
            New `HallMagneticData` with
        """
        return self.crop_with(self.magneticfield, min_, max_)


class Hall1Data(_HallData):

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
    ) -> Hall1Data:
        Vxx = load_raw_data(file_Vxx, save_npy=save_npy)
        return Hall1Data(
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
    ) -> Hall1Data:
        Vxx = load_raw_data(file_Vxx, save_npy=save_npy)
        return Hall1Data(
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
    ) -> Hall1Data:
        Vxy = load_raw_data(file_Vxy, save_npy=save_npy)
        return Hall1Data(
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
    ) -> Hall1Data:
        Vxx = load_raw_data(file_Vxx, save_npy=save_npy)
        return Hall1Data(
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
    ) -> Hall1Data:
        Vxx = load_raw_data(file_Vxx, save_npy=save_npy)
        return Hall1Data(
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
        data_Isd: t.Union[int, float, str, Path],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> Hall1Data:
        Vxy = load_raw_data(file_Vxy, save_npy=save_npy)
        return Hall1Data(
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
            Vxx: 1-D array of Vxx.
            Vxy: 1-D array of Vxy.
            magneticfield: a constant value of magnetic field.
            axis0: 1-D array of the axis 0.
            Isd: 1-D array of source-drain current.
            hallbar_ratio: Ratio (width / length) of a Hallbar.

        Notes:
            `hallbar_ratio` is not needed to be substituted if your device
            is not a Hallbar, but in that case, values of mobility doesn't
            have meaning.
        """
        super().__init__(
            Vxx,
            Vxy,
            magneticfield,
            Isd,
            hallbar_ratio=hallbar_ratio,
        )
        self._axis0 = axis0

    @cached_property
    def magneticfield(self) -> float:
        return float(self._magneticfield)

    @property
    def axis0(self) -> np.ndarray:
        return self._axis0

    def crop_with(
        self,
        arr: np.ndarray,
        min_: float,
        max_: float,
    ) -> Hall1Data:
        _check_min_max(min_, max_)
        indices = _get_matching_range(arr, min_, max_)
        return Hall1Data(
            self.Vxx[indices],
            self.Vxy[indices],
            self.magneticfield,
            self.axis0[indices],
            self.Isd[indices],
            self.hallbar_ratio,
        )

    def crop_Vxx(self, min_: float, max_: float) -> Hall1Data:
        return self.crop_with(self.Vxx, min_, max_)

    def crop_Vxy(self, min_: float, max_: float) -> Hall1Data:
        return self.crop_with(self.Vxy, min_, max_)

    def crop_Isd(self, min_: float, max_: float) -> Hall1Data:
        return self.crop_with(self.Isd, min_, max_)

    def crop_Rxx(self, min_: float, max_: float) -> Hall1Data:
        return self.crop_with(self.Rxx, min_, max_)

    def crop_Rxy(self, min_: float, max_: float) -> Hall1Data:
        return self.crop_with(self.Rxy, min_, max_)

    def crop_axis0(self, min_: float, max_: float) -> Hall1Data:
        return self.crop_with(self.axis0, min_, max_)

    def calc_corrected_density(self, offset: Hall1Data) -> np.ndarray:
        return self.magneticfield / (e * (self.rho_xy - offset.rho_xy))

    def calc_corrected_mobility(self, offset: Hall1Data) -> np.ndarray:
        return (
            (self.rho_xy - offset.rho_xy) / (self.magneticfield * self.rho_xx)
        )


class Hall2Data(_HallData):

    @staticmethod
    def load_with_Vref(
        file_Vxx: t.Union[str, Path],
        file_Vxy: t.Union[str, Path],
        magneticfield: t.Union[int, float],
        file_axis0: t.Union[str, Path],
        file_axis1: t.Union[str, Path],
        data_Vref: t.Union[int, float, str, Path],
        resistance_ref: t.Union[int, float],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> Hall2Data:
        Vxx = load_raw_data(file_Vxx, save_npy=save_npy)
        return Hall2Data(
            Vxx,
            load_raw_data(file_Vxy, save_npy=save_npy),
            magneticfield,
            load_raw_data(file_axis0, save_npy=save_npy),
            load_raw_data(file_axis1, save_npy=save_npy),
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
        file_axis1: t.Union[str, Path],
        data_Vref: t.Union[int, float, str, Path],
        resistance_ref: float,
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> Hall2Data:
        Vxx = load_raw_data(file_Vxx, save_npy=save_npy)
        return Hall2Data(
            Vxx,
            np.empty_like(Vxx),
            magneticfield,
            load_raw_data(file_axis0, save_npy=save_npy),
            load_raw_data(file_axis1, save_npy=save_npy),
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
        file_axis1: t.Union[str, Path],
        data_Vref: t.Union[int, float, str, Path],
        resistance_ref: float,
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> Hall2Data:
        Vxy = load_raw_data(file_Vxy, save_npy=save_npy)
        return Hall2Data(
            np.empty_like(Vxy),
            Vxy,
            magneticfield,
            load_raw_data(file_axis0, save_npy=save_npy),
            load_raw_data(file_axis1, save_npy=save_npy),
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
        file_axis1: t.Union[str, Path],
        data_Isd: t.Union[int, float, str, Path],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> Hall2Data:
        Vxx = load_raw_data(file_Vxx, save_npy=save_npy)
        return Hall2Data(
            Vxx,
            load_raw_data(file_Vxy, save_npy=save_npy),
            magneticfield,
            load_raw_data(file_axis0, save_npy=save_npy),
            load_raw_data(file_axis1, save_npy=save_npy),
            _load_Isd(Vxx, data_Isd, save_npy=save_npy),
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_xx_with_Isd(
        file_Vxx: t.Union[str, Path],
        magneticfield: t.Union[int, float],
        file_axis0: t.Union[str, Path],
        file_axis1: t.Union[str, Path],
        data_Isd: t.Union[int, float, str, Path],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> Hall2Data:
        Vxx = load_raw_data(file_Vxx, save_npy=save_npy)
        return Hall2Data(
            Vxx,
            np.empty_like(Vxx),
            magneticfield,
            load_raw_data(file_axis0, save_npy=save_npy),
            load_raw_data(file_axis1, save_npy=save_npy),
            _load_Isd(Vxx, data_Isd, save_npy=save_npy),
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_xy_with_Isd(
        file_Vxy: t.Union[str, Path],
        magneticfield: t.Union[int, float],
        file_axis0: t.Union[str, Path],
        file_axis1: t.Union[str, Path],
        data_Isd: t.Union[int, float, str, Path],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> Hall2Data:
        Vxy = load_raw_data(file_Vxy, save_npy=save_npy)
        return Hall2Data(
            np.empty_like(Vxy),
            Vxy,
            magneticfield,
            load_raw_data(file_axis0, save_npy=save_npy),
            load_raw_data(file_axis1, save_npy=save_npy),
            _load_Isd(Vxy, data_Isd, save_npy=save_npy),
            hallbar_ratio=hallbar_ratio,
        )

    def __init__(
        self,
        Vxx: np.ndarray,
        Vxy: np.ndarray,
        magneticfield: t.Union[int, float],
        axis0: np.ndarray,
        axis1: np.ndarray,
        Isd: np.ndarray,
        hallbar_ratio: float = 0.25,
    ) -> None:
        """
        Args:
            Vxx: 2-D array of Vxx.
            Vxy: 2-D array of Vxy.
            magneticfield: a constant value of magnetic field.
            axis0: 1-D array of the axis 0.
            axis1: 1-D array of the axis 1.
            Isd: 2-D array of source-drain current.
            hallbar_ratio: Ratio (width / length) of a Hallbar.

        Notes:
            `hallbar_ratio` is not needed to be substituted if your device
            is not a Hallbar, but in that case, values of mobility doesn't
            have meaning.
        """
        super().__init__(
            Vxx,
            Vxy,
            magneticfield,
            Isd,
            hallbar_ratio=hallbar_ratio,
        )
        self._axis0 = axis0
        self._axis1 = axis1

    @cached_property
    def magneticfield(self) -> float:
        return float(self._magneticfield)

    @property
    def axis0(self) -> np.ndarray:
        return self._axis0

    @property
    def axis1(self) -> np.ndarray:
        return self._axis1

    def crop_with(
        self,
        arr: np.ndarray,
        min_: float,
        max_: float,
        axis: int,
    ) -> Hall2Data:
        _check_min_max(min_, max_)

        if axis not in {0, 1}:
            raise ValueError("'axis' must be 0 or 1.")
        elif axis == 0 and np.size(arr) != np.size(self.axis0):
            raise ValueError("'arr' must be same size as axis0.")
        elif axis == 1 and np.size(arr) != np.size(self.axis1):
            raise ValueError("'arr' must be same size as axis1.")

        indices = _get_matching_range(arr, min_, max_)
        if axis == 0:
            return Hall2Data(
                self.Vxx[indices, :],
                self.Vxy[indices, :],
                self.magneticfield,
                self.axis0[indices],
                self.axis1,
                self.Isd[indices, :],
                self.hallbar_ratio,
            )
        else:
            return Hall2Data(
                self.Vxx[:, indices],
                self.Vxy[:, indices],
                self.magneticfield,
                self.axis0,
                self.axis1[indices],
                self.Isd[:, indices],
                self.hallbar_ratio,
            )

    def crop_axis0(self, min_: float, max_: float) -> Hall2Data:
        return self.crop_with(self.axis0, min_, max_, 0)

    def crop_axis1(self, min_: float, max_: float) -> Hall2Data:
        return self.crop_with(self.axis1, min_, max_, 1)

    def calc_corrected_density(self, offset: Hall2Data) -> np.ndarray:
        return self.magneticfield / (e * (self.rho_xy - offset.rho_xy))

    def calc_corrected_mobility(self, offset: Hall2Data) -> np.ndarray:
        return (
            (self.rho_xy - offset.rho_xy) / (self.magneticfield * self.rho_xx)
        )


class Hall3Data(_HallData):

    @staticmethod
    def load_with_Vref(
        file_Vxx: t.Union[str, Path],
        file_Vxy: t.Union[str, Path],
        magneticfield: t.Union[int, float],
        file_axis0: t.Union[str, Path],
        file_axis1: t.Union[str, Path],
        file_axis2: t.Union[str, Path],
        data_Vref: t.Union[int, float, str, Path],
        resistance_ref: t.Union[int, float],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> Hall3Data:
        Vxx = load_raw_data(file_Vxx, save_npy=save_npy)
        return Hall3Data(
            Vxx,
            load_raw_data(file_Vxy, save_npy=save_npy),
            magneticfield,
            load_raw_data(file_axis0, save_npy=save_npy),
            load_raw_data(file_axis1, save_npy=save_npy),
            load_raw_data(file_axis2, save_npy=save_npy),
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
        file_axis1: t.Union[str, Path],
        file_axis2: t.Union[str, Path],
        data_Vref: t.Union[int, float, str, Path],
        resistance_ref: float,
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> Hall3Data:
        Vxx = load_raw_data(file_Vxx, save_npy=save_npy)
        return Hall3Data(
            Vxx,
            np.empty_like(Vxx),
            magneticfield,
            load_raw_data(file_axis0, save_npy=save_npy),
            load_raw_data(file_axis1, save_npy=save_npy),
            load_raw_data(file_axis2, save_npy=save_npy),
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
        file_axis1: t.Union[str, Path],
        file_axis2: t.Union[str, Path],
        data_Vref: t.Union[int, float, str, Path],
        resistance_ref: float,
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> Hall3Data:
        Vxy = load_raw_data(file_Vxy, save_npy=save_npy)
        return Hall3Data(
            np.empty_like(Vxy),
            Vxy,
            magneticfield,
            load_raw_data(file_axis0, save_npy=save_npy),
            load_raw_data(file_axis1, save_npy=save_npy),
            load_raw_data(file_axis2, save_npy=save_npy),
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
        file_axis1: t.Union[str, Path],
        file_axis2: t.Union[str, Path],
        data_Isd: t.Union[int, float, str, Path],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> Hall3Data:
        Vxx = load_raw_data(file_Vxx, save_npy=save_npy)
        return Hall3Data(
            Vxx,
            load_raw_data(file_Vxy, save_npy=save_npy),
            magneticfield,
            load_raw_data(file_axis0, save_npy=save_npy),
            load_raw_data(file_axis1, save_npy=save_npy),
            load_raw_data(file_axis2, save_npy=save_npy),
            _load_Isd(Vxx, data_Isd, save_npy=save_npy),
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_xx_with_Isd(
        file_Vxx: t.Union[str, Path],
        magneticfield: t.Union[int, float],
        file_axis0: t.Union[str, Path],
        file_axis1: t.Union[str, Path],
        file_axis2: t.Union[str, Path],
        data_Isd: t.Union[int, float, str, Path],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> Hall3Data:
        Vxx = load_raw_data(file_Vxx, save_npy=save_npy)
        return Hall3Data(
            Vxx,
            np.empty_like(Vxx),
            magneticfield,
            load_raw_data(file_axis0, save_npy=save_npy),
            load_raw_data(file_axis1, save_npy=save_npy),
            load_raw_data(file_axis2, save_npy=save_npy),
            _load_Isd(Vxx, data_Isd, save_npy=save_npy),
            hallbar_ratio=hallbar_ratio,
        )

    @staticmethod
    def load_xy_with_Isd(
        file_Vxy: t.Union[str, Path],
        magneticfield: t.Union[int, float],
        file_axis0: t.Union[str, Path],
        file_axis1: t.Union[str, Path],
        file_axis2: t.Union[str, Path],
        data_Isd: t.Union[int, float, str, Path],
        hallbar_ratio: float = 0.25,
        save_npy: bool = True,
    ) -> Hall3Data:
        Vxy = load_raw_data(file_Vxy, save_npy=save_npy)
        return Hall3Data(
            np.empty_like(Vxy),
            Vxy,
            magneticfield,
            load_raw_data(file_axis0, save_npy=save_npy),
            load_raw_data(file_axis1, save_npy=save_npy),
            load_raw_data(file_axis2, save_npy=save_npy),
            _load_Isd(Vxy, data_Isd, save_npy=save_npy),
            hallbar_ratio=hallbar_ratio,
        )

    def __init__(
        self,
        Vxx: np.ndarray,
        Vxy: np.ndarray,
        magneticfield: t.Union[int, float],
        axis0: np.ndarray,
        axis1: np.ndarray,
        axis2: np.ndarray,
        Isd: np.ndarray,
        hallbar_ratio: float = 0.25,
    ) -> None:
        """
        Args:
            Vxx: 3-D array of Vxx.
            Vxy: 3-D array of Vxy.
            magneticfield: a constant value of magnetic field.
            axis0: 1-D array of the axis 0.
            axis1: 1-D array of the axis 1.
            axis2: 1-D array of the axis 2.
            Isd: 3-D array of source-drain current.
            hallbar_ratio: Ratio (width / length) of a Hallbar.

        Notes:
            `hallbar_ratio` is not needed to be substituted if your device
            is not a Hallbar, but in that case, values of mobility doesn't
            have meaning.
        """
        super().__init__(
            Vxx,
            Vxy,
            magneticfield,
            Isd,
            hallbar_ratio=hallbar_ratio,
        )
        self._axis0 = axis0
        self._axis1 = axis1
        self._axis2 = axis2

    @cached_property
    def magneticfield(self) -> float:
        return float(self._magneticfield)

    @property
    def axis0(self) -> np.ndarray:
        return self._axis0

    @property
    def axis1(self) -> np.ndarray:
        return self._axis1

    @property
    def axis2(self) -> np.ndarray:
        return self._axis2

    def crop_with(
        self,
        arr: np.ndarray,
        min_: float,
        max_: float,
        axis: int,
    ) -> Hall3Data:
        _check_min_max(min_, max_)

        if axis not in {0, 1, 2}:
            raise ValueError("'axis' must be 0, 1 or 2.")
        elif axis == 0 and np.size(arr) != np.size(self.axis0):
            raise ValueError("'arr' must be same size as axis0.")
        elif axis == 1 and np.size(arr) != np.size(self.axis1):
            raise ValueError("'arr' must be same size as axis1.")
        elif axis == 2 and np.size(arr) != np.size(self.axis2):
            raise ValueError("'arr' must be same size as axis2.")

        indices = _get_matching_range(arr, min_, max_)
        if axis == 0:
            return Hall3Data(
                self.Vxx[indices, :, :],
                self.Vxy[indices, :, :],
                self.magneticfield,
                self.axis0[indices],
                self.axis1,
                self.axis2,
                self.Isd[indices, :, :],
                self.hallbar_ratio,
            )
        elif axis == 1:
            return Hall3Data(
                self.Vxx[:, indices, :],
                self.Vxy[:, indices, :],
                self.magneticfield,
                self.axis0,
                self.axis1[indices],
                self.axis2,
                self.Isd[:, indices, :],
                self.hallbar_ratio,
            )
        else:
            Hall3Data(
                self.Vxx[:, :, indices],
                self.Vxy[:, :, indices],
                self.magneticfield,
                self.axis0,
                self.axis1,
                self.axis2[indices],
                self.Isd[:, :, indices],
                self.hallbar_ratio,
            )

    def crop_axis0(self, min_: float, max_: float) -> Hall3Data:
        return self.crop_with(self.axis0, min_, max_, 0)

    def crop_axis1(self, min_: float, max_: float) -> Hall3Data:
        return self.crop_with(self.axis1, min_, max_, 1)

    def crop_axis2(self, min_: float, max_: float) -> Hall3Data:
        return self.crop_with(self.axis2, min_, max_, 2)

    def calc_corrected_density(self, offset: Hall3Data) -> np.ndarray:
        return self.magneticfield / (e * (self.rho_xy - offset.rho_xy))

    def calc_corrected_mobility(self, offset: Hall3Data) -> np.ndarray:
        return (
            (self.rho_xy - offset.rho_xy) / (self.magneticfield * self.rho_xx)
        )
