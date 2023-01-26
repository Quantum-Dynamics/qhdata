from __future__ import annotations
import abc
from functools import cached_property
from pathlib import Path
import typing as t

import numpy as np

from .const import hc
from .function import gaussian_dist
from .util import (
    load_raw_data,
    smooth_outliers,
    _check_min_max,
    _get_matching_range,
)


def convert_wavelength2energy(wavelength: np.ndarray) -> np.ndarray:
    return hc / (wavelength + 1e-12)


class SPDataBase(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def load_with_energy(
        self,
        file_energy: t.Union[str, Path],
        *args: t.Any,
        **kwargs: t.Any,
    ) -> SPDataBase:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def load_with_wavelength(
        self,
        file_wavelength: t.Union[str, Path],
        *args: t.Any,
        **kwargs: t.Any,
    ) -> SPDataBase:
        raise NotImplementedError

    @abc.abstractmethod
    def __init__(
        self,
        energy: np.ndarray,
        counts: np.ndarray,
        *args: t.Any,
        **kwargs: t.Any,
    ) -> None:
        self._energy = energy
        self._counts = counts

    @cached_property
    def energy(self) -> np.ndarray:
        """Energy profile of the data."""
        return self._energy

    @property
    def wavelength(self) -> np.ndarray:
        """Wavelength profile of the data."""
        return hc / (self._energy + 1e-12)

    @cached_property
    def pixel(self) -> np.ndarray:
        return np.arange(len(self.wavelength))

    @property
    def counts(self) -> np.ndarray:
        """CCD counts of the data."""
        return self._counts

    @cached_property
    def counts_log(self) -> np.ndarray:
        return np.log(self._counts)

    @cached_property
    def counts_log_normalized(self) -> np.ndarray:
        diff_counts_log = self.counts_log - np.min(self.counts_log)
        return diff_counts_log / np.max(diff_counts_log)

    @property
    def num_pixels(self) -> int:
        """Number of CCD pixels."""
        return len(self.wavelength)

    @property
    def num_measure(self) -> int:
        """Number of exposure times."""
        return len(self.counts)

    @abc.abstractmethod
    def remove_darkcounts(self, darkcounts: int) -> SPDataBase:
        """Remove dark counts from the data.

        Args:
            darkcounts: counts to be subtracted.

        Returns:
            New SPData object with same wavelength data and processed
            counts data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def remove_cosmic_noise(
        self,
        threshold: int,
        closeness: int = 3,
    ) -> SPDataBase:
        """Remove effects of cosmic rays.

        Args:
            threshld: Threshold to judge each count are caused by cosmic rays.
            closeness: Width between indices that represents how close each
                index are.

        Returns:
            New SPData object with same wavelength data and processed
            counts data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def filter_gaussian_along_pixel(
        self,
        mean: float,
        std: float,
    ) -> SPDataBase:
        """Apply Gaussian filter to each spectrum along pixel for smoothing.

        Args:
            mean: Mean of the Gaussian.
            std: Standard deviation of the Gaussian.

        Returns:
            New SPData object with same wavelength data and processed
            counts data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def filter_gaussian_along_wavelength(
        self,
        mean: float,
        std: float,
    ) -> SPDataBase:
        """Apply Gaussian filter to each spectrum along wavelength for
        smoothing.

        Args:
            mean: Mean of the Gaussian.
            std: Standard deviation of the Gaussian.

        Returns:
            New SPData object with same wavelength data and processed
            counts data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def filter_gaussian_along_energy(
        self,
        mean: float,
        std: float,
    ) -> SPDataBase:
        """Apply Gaussian filter to each spectrum along energy for
        smoothing.

        Args:
            mean: Mean of the Gaussian.
            std: Standard deviation of the Gaussian.

        Returns:
            New SPData object with same wavelength data and processed
            counts data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def crop_pixel(self, min_: int, max_: int) -> SPData:
        """Crop a range of pixels.

        Args:
            min_: Minimal pixel value of the range.
            max_: Maximal pixel value of the range.

        Returns:
            New SPData object with the cropped data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def crop_wavelength(self, min_: float, max_: float) -> SPData:
        """Crop a range of wavelength.

        Args:
            min_: Minimal wavelength of the range.
            max_: Maximal wavelength of the range.

        Returns:
            New SPData object with the cropped data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def crop_energy(self, min_: float, max_: float) -> SPData:
        """Crop a range of energy.

        Args:
            min_: Minimal energy of the range.
            max_: Maximal energy of the range.

        Returns:
            New SPData object with the cropped data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sum_counts_pixel(
        self,
        min_: int,
        max_: int,
        normalize: bool = False,
    ) -> np.ndarray:
        """Sum all counts up every axes in given range of pixels.

        Args:
            min_: Minimal pixel of the range.
            max_: Maximal pixel of the range.
            normalize: `True` if the output is to be normalized.

        Returns:
            Ndarray of the sums.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sum_counts_wavelength(
        self,
        min_: float,
        max_: float,
        normalize: bool = False,
    ) -> np.ndarray:
        """Sum all counts up every axes in given range of wavelength.

        Args:
            min_: Minimal wavelength of the range.
            max_: Maximal wavelength of the range.
            normalize: `True` if the output is to be normalized.

        Returns:
            Ndarray of the sums.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sum_counts_energy(
        self,
        min_: float,
        max_: float,
        normalize: bool = False,
    ) -> np.ndarray:
        """Sum all counts up every axes in given range of energy.

        Args:
            min_: Minimal energy of the range.
            max_: Maximal energy of the range.
            normalize: `True` if the output is to be normalized.

        Returns:
            Ndarray of the sums.
        """
        raise NotImplementedError


class SPData(SPDataBase):

    @staticmethod
    def load_with_energy(
        file_energy: t.Union[str, Path],
        file_counts: t.Union[str, Path],
        save_npy: bool = True,
        load_npy: bool = True,
    ) -> SPData:
        """Load given files and generate new SPData object.

        Args:
            file_energy: Path to a file with energy profile.
            file_counts: Path to a file with CCD data.
            save_npy: Whether the function creates new npy files from given
                data.

        Returns:
            New SPData object with given data.
        """
        return SPData(
            load_raw_data(file_energy, save_npy=save_npy, load_npy=load_npy),
            load_raw_data(file_counts, save_npy=save_npy, load_npy=load_npy),
        )

    @staticmethod
    def load_with_wavelength(
        file_wavelength: t.Union[str, Path],
        file_counts: t.Union[str, Path],
        save_npy: bool = True,
        load_npy: bool = True,
    ) -> SP1Data:
        """Load given files and generate new SPData object.

        Args:
            file_wavelength: Path to a file with wavelength profile.
            file_counts: Path to a file with CCD data.
            save_npy: Whether the function creates new npy files from given
                data.

        Returns:
            New SPData object with given data.
        """
        wavelength = load_raw_data(
            file_wavelength,
            save_npy=save_npy,
            load_npy=load_npy,
        )
        return SPData(
            convert_wavelength2energy(wavelength),
            load_raw_data(file_counts, save_npy=save_npy, load_npy=load_npy),
        )

    def __init__(self, energy: np.ndarray, counts: np.ndarray) -> None:
        """
        Args:
            energy: Energy profile of a measurement.
            counts: 1-D array of CCD data.
        """
        super().__init__(energy, counts)

    @cached_property
    def peak_intensity(self) -> float:
        return np.max(self.counts)

    def update_counts(self, new_counts: np.ndarray) -> SPData:
        return SPData(self.energy, new_counts)

    def update_counts_normalized_log(self) -> SPData:
        return self.update_counts(self.counts_log_normalized)

    def remove_darkcounts(self, darkcounts: int) -> SPData:
        return SPData(self.energy, self.counts - darkcounts)

    def remove_cosmic_noise(
        self,
        threshold: int,
        closeness: int = 3,
    ) -> SPData:
        counts_cosmic_removed = smooth_outliers(
            self.counts,
            threshold,
            closeness,
        )
        return SPData(self.energy, counts_cosmic_removed)

    def filter_gaussian_along_pixel(self, mean: float, std: float) -> SPData:
        gaussian = gaussian_dist(self.pixel, mean, std)
        counts = np.convolve(self.counts, gaussian, mode="same")
        return SPData(self.energy, counts)

    def fileter_gaussian_along_wavelength(
        self,
        mean: float,
        std: float,
    ) -> SPData:
        gaussian = gaussian_dist(self.wavelength, mean, std)
        counts = np.convolve(self.wavelength, gaussian, mode="same")
        return SPData(self.energy, counts)

    def crop_pixel(self, min_: int, max_: int) -> SPData:
        _check_min_max(min_, max_)
        return SPData(
            self.energy[min_:max_ + 1],
            self.counts[min_:max_ + 1],
        )

    def crop_wavelength(self, min_: float, max_: float) -> SPData:
        _check_min_max(min_, max_)
        indices = _get_matching_range(self.wavelength, min_, max_)
        return SPData(self.energy[indices], self.counts[indices])

    def crop_energy(self, min_: float, max_: float) -> SPData:
        _check_min_max(min_, max_)
        indices = _get_matching_range(self.energy, min_, max_)
        return SPData(self.energy[indices], self.counts[indices])

    def sum_counts_pixel(
        self,
        min_: int,
        max_: int,
        normalize: bool = False,
    ) -> float:
        _check_min_max(min_, max_)
        intensity = np.sum(self.counts[min_:max_ + 1])
        if normalize:
            return intensity / np.max(intensity)
        return intensity

    def sum_counts_wavelength(
        self,
        min_: float,
        max_: float,
        normalize: bool = False,
    ) -> float:
        intensity = np.sum(self.crop_wavelength(min_, max_).counts)
        if normalize:
            return intensity / np.max(intensity)
        return intensity

    def sum_counts_energy(
        self,
        min_: float,
        max_: float,
        normalize: bool = False,
    ) -> float:
        intensity = np.sum(self.crop_energy(min_, max_).counts)
        if normalize:
            return intensity / np.max(intensity)
        return intensity


class SP1Data(SPDataBase):

    @staticmethod
    def load_with_energy(
        file_energy: t.Union[str, Path],
        file_counts: t.Union[str, Path],
        file_axis1: t.Union[str, Path],
        save_npy: bool = True,
        load_npy: bool = True,
    ) -> SP1Data:
        """Load given files and generate new SPData object.

        Args:
            file_energy: Path to a file with energy profile.
            file_counts: Path to a file with CCD data.
            file_axis1: Path to a file with data of the axis 1.
            save_npy: Whether the function creates new npy files from given
                data.

        Returns:
            New SPData object with given data.
        """
        return SP1Data(
            load_raw_data(
                file_energy,
                save_npy=save_npy,
                load_npy=load_npy,
            ),
            load_raw_data(file_counts, save_npy=save_npy, load_npy=load_npy),
            load_raw_data(file_axis1, save_npy=save_npy, load_npy=load_npy),
        )

    @staticmethod
    def load_with_wavelength(
        file_wavelength: t.Union[str, Path],
        file_counts: t.Union[str, Path],
        file_axis1: t.Union[str, Path],
        save_npy: bool = True,
        load_npy: bool = True,
    ) -> SP1Data:
        """Load given files and generate new SPData object.

        Args:
            file_wavelength: Path to a file with wavelength profile.
            file_counts: Path to a file with CCD data.
            file_axis1: Path to a file with data of the axis 1.
            save_npy: Whether the function creates new npy files from given
                data.

        Returns:
            New SPData object with given data.
        """
        wavelength = load_raw_data(
            file_wavelength,
            save_npy=save_npy,
            load_npy=load_npy,
        )
        return SP1Data(
            convert_wavelength2energy(wavelength),
            load_raw_data(file_counts, save_npy=save_npy, load_npy=load_npy),
            load_raw_data(file_axis1, save_npy=save_npy, load_npy=load_npy),
        )

    def __init__(
        self,
        energy: np.ndarray,
        counts: np.ndarray,
        axis1: np.ndarray,
    ) -> None:
        """
        Args:
            energy: Energy profile of a measurement as a 1-D array.
            counts: 2-D array of CCD data.
            axis1: 1-D array of the axis 1.
        """
        super().__init__(energy, counts)

        self._axis1 = axis1

    @property
    def axis1(self) -> np.ndarray:
        return self._axis1

    @cached_property
    def peak_intensity(self) -> np.ndarray:
        return np.max(self.counts, axis=1)

    def update_counts(self, new_counts: np.ndarray) -> SP1Data:
        return SP1Data(
            self.energy,
            new_counts,
            self.axis1,
        )

    def update_counts_normalized_log(self) -> SP1Data:
        return self.update_counts(self.counts_log_normalized)

    def remove_darkcounts(self, darkcounts: int) -> SP1Data:
        return SP1Data(self.energy, self.counts - darkcounts, self.axis1)

    def remove_cosmic_noise(self, threshold: int, closeness: int = 3) -> SP1Data:
        counts_cosmic_removed = np.apply_along_axis(
            smooth_outliers,
            1,
            self.counts,
            threshold,
            closeness,
        )
        return SP1Data(self.energy, counts_cosmic_removed, self.axis1)

    def filter_gaussian_along_pixel(self, mean: float, std: float) -> SP1Data:
        gaussian = gaussian_dist(self.pixel, mean, std)
        counts = np.apply_along_axis(
            np.convolve,
            1,
            self.counts,
            gaussian,
            mode="same",
        )
        return SP1Data(self.energy, counts, self.axis1)

    def filter_gaussian_along_wavelength(
        self,
        mean: float,
        std: float,
    ) -> SP1Data:
        gaussian = gaussian_dist(self.wavelength, mean, std)
        counts = np.apply_along_axis(
            np.convolve,
            1,
            self.counts,
            gaussian,
            mode="same",
        )
        return SP1Data(self.energy, counts, self.axis1)

    def filter_gaussian_along_energy(
        self,
        mean: float,
        std: float,
    ) -> SP1Data:
        gaussian = gaussian_dist(self.energy, mean, std)
        counts = np.apply_along_axis(
            np.convolve,
            1,
            self.counts,
            gaussian,
            mode="same",
        )
        return SP1Data(self.energy, counts, self.axis1)

    def crop_pixel(self, min_: int, max_: int) -> SP1Data:
        _check_min_max(min_, max_)
        return SP1Data(
            self.energy[min_:max_ + 1],
            self.counts[:, min_:max_ + 1],
            self.axis1,
        )

    def crop_wavelength(self, min_: float, max_: float) -> SP1Data:
        _check_min_max(min_, max_)
        indices = _get_matching_range(self.wavelength, min_, max_)
        return SP1Data(
            self.energy[indices],
            self.counts[:, indices],
            self.axis1,
        )

    def crop_energy(self, min_: float, max_: float) -> SP1Data:
        _check_min_max(min_, max_)
        indices = _get_matching_range(self.energy, min_, max_)
        return SP1Data(
            self.energy[indices],
            self.counts[:, indices],
            self.axis1,
        )

    def crop_axis1(self, min_: float, max_: float) -> SP1Data:
        _check_min_max(min_, max_)
        indices = _get_matching_range(self.axis1, min_, max_)
        return SP1Data(
            self.energy,
            self.counts[indices, :],
            self.axis1[indices],
        )

    def sum_counts_pixel(
        self,
        min_: int,
        max_: int,
        normalize: bool = False,
    ) -> np.ndarray:
        _check_min_max(min_, max_)
        intensity = np.sum(self.counts[:, min_:max_ + 1], axis=1)
        if normalize:
            return intensity / np.max(intensity)
        return intensity

    def sum_counts_wavelength(
        self,
        min_: float,
        max_: float,
        normalize: bool = False,
    ) -> np.ndarray:
        intensity = np.sum(self.crop_wavelength(min_, max_).counts, axis=1)
        if normalize:
            return intensity / np.max(intensity)
        return intensity

    def sum_counts_energy(
        self,
        min_: float,
        max_: float,
        normalize: bool = False,
    ) -> np.ndarray:
        intensity = np.sum(self.crop_energy(min_, max_).counts, axis=1)
        if normalize:
            return intensity / np.max(intensity)
        return intensity


class SP2Data(SPDataBase):

    @staticmethod
    def load_with_energy(
        file_energy: t.Union[str, Path],
        file_counts: t.Union[str, Path],
        file_axis1: t.Union[str, Path],
        file_axis2: t.Union[str, Path],
        save_npy: bool = True,
        load_npy: bool = True,
    ) -> SP2Data:
        energy = load_raw_data(file_energy, save_npy=save_npy, load_npy=load_npy)
        counts = load_raw_data(file_counts, save_npy=save_npy, load_npy=load_npy)
        axis1 = load_raw_data(file_axis1, save_npy=save_npy, load_npy=load_npy)
        axis2 = load_raw_data(file_axis2, save_npy=save_npy, load_npy=load_npy)
        return SP2Data(
            energy,
            counts.reshape(len(axis2), len(axis1), len(energy)),
            axis1,
            axis2,
        )

    @staticmethod
    def load_with_wavelength(
        file_wavelength: t.Union[str, Path],
        file_counts: t.Union[str, Path],
        file_axis1: t.Union[str, Path],
        file_axis2: t.Union[str, Path],
        save_npy: bool = True,
        load_npy: bool = True,
    ) -> SP2Data:
        wavelength = load_raw_data(
            file_wavelength,
            save_npy=save_npy,
            load_npy=load_npy,
        )
        counts = load_raw_data(file_counts, save_npy=save_npy, load_npy=load_npy)
        axis1 = load_raw_data(file_axis1, save_npy=save_npy, load_npy=load_npy)
        axis2 = load_raw_data(file_axis2, save_npy=save_npy, load_npy=load_npy)
        return SP2Data(
            convert_wavelength2energy(wavelength),
            counts.reshape(len(axis2), len(axis1), len(wavelength)),
            axis1,
            axis2,
        )

    def __init__(
        self,
        energy: np.ndarray,
        counts: np.ndarray,
        axis1: np.ndarray,
        axis2: np.ndarray,
    ) -> None:
        super().__init__(energy, counts)

        self._axis1 = axis1
        self._axis2 = axis2

    @property
    def axis1(self) -> np.ndarray:
        return self._axis1

    @property
    def axis2(self) -> np.ndarray:
        return self._axis2

    @cached_property
    def peak_intensity(self) -> np.ndarray:
        return np.max(self.counts, axis=2)

    def update_counts(self, new_counts: np.ndarray) -> SP2Data:
        return SP2Data(
            self.energy,
            new_counts,
            self.axis1,
            self.axis2,
        )

    def update_counts_normalized_log(self) -> SP2Data:
        return self.update_counts(self.counts_log_normalized)

    def remove_darkcounts(self, darkcounts: int) -> SP2Data:
        return SP2Data(
            self.energy,
            self.counts - darkcounts,
            self.axis1,
            self.axis2,
        )

    def remove_cosmic_noise(
        self,
        threshold: int,
        closeness: int = 3,
    ) -> SP2Data:
        counts_cosmic_removed = np.apply_along_axis(
            smooth_outliers,
            2,
            self.counts,
            threshold,
            closeness,
        )
        return SP2Data(
            self.energy,
            counts_cosmic_removed,
            self.axis1,
            self.axis2,
        )

    def filter_gaussian_along_pixel(self, mean: float, std: float) -> SP2Data:
        gaussian = gaussian_dist(self.pixel, mean, std)
        counts = np.apply_along_axis(
            np.convolve,
            2,
            self.counts,
            gaussian,
            mode="same",
        )
        return SP2Data(self.energy, counts, self.axis1, self.axis2)

    def filter_gaussian_along_wavelength(
        self,
        mean: float,
        std: float,
    ) -> SP2Data:
        gaussian = gaussian_dist(self.wavelength, mean, std)
        counts = np.apply_along_axis(
            np.convolve,
            2,
            self.counts,
            gaussian,
            mode="same",
        )
        return SP2Data(self.energy, counts, self.axis1, self.axis2)

    def filter_gaussian_along_energy(
        self,
        mean: float,
        std: float,
    ) -> SP2Data:
        gaussian = gaussian_dist(self.energy, mean, std)
        counts = np.apply_along_axis(
            np.convolve,
            2,
            self.counts,
            gaussian,
            mode="same",
        )
        return SP2Data(self.energy, counts, self.axis1, self.axis2)

    def crop_pixel(self, min_: int, max_: int) -> SP2Data:
        _check_min_max(min_, max_)
        return SP2Data(
            self.energy[min_:max_ + 1],
            self.counts[:, :, min_:max_ + 1],
            self.axis1,
            self.axis2,
        )

    def crop_wavelength(self, min_: float, max_: float) -> SP2Data:
        _check_min_max(min_, max_)
        indices = _get_matching_range(self.wavelength, min_, max_)
        return SP2Data(
            self.energy[indices],
            self.counts[:, :, indices],
            self.axis1,
            self.axis2,
        )

    def crop_energy(self, min_: float, max_: float) -> SP2Data:
        _check_min_max(min_, max_)
        indices = _get_matching_range(self.energy, min_, max_)
        return SP2Data(
            self.energy[indices],
            self.counts[:, :, indices],
            self.axis1,
            self.axis2,
        )

    def crop_axis1(self, min_: float, max_: float) -> SP2Data:
        _check_min_max(min_, max_)
        indices = _get_matching_range(self.axis1, min_, max_)
        return SP2Data(
            self.energy,
            self.counts[:, indices, :],
            self.axis1[indices],
            self.axis2,
        )

    def crop_axis2(self, min_: float, max_: float) -> SP2Data:
        _check_min_max(min_, max_)
        indices = _get_matching_range(self.axis2, min_, max_)
        return SP2Data(
            self.energy,
            self.counts[indices, :, :],
            self.axis1,
            self.axis2[indices],
        )

    def sum_counts_pixel(
        self,
        min_: int,
        max_: int,
        normalize: bool = False,
    ) -> np.ndarray:
        intensity = np.sum(self.crop_pixel(min_, max_).counts, axis=2)
        if normalize:
            return intensity / np.max(intensity)
        return intensity

    def sum_counts_wavelength(
        self,
        min_: int,
        max_: int,
        normalize: bool = False,
    ) -> np.ndarray:
        intensity = np.sum(self.crop_wavelength(min_, max_).counts, axis=2)
        if normalize:
            return intensity / np.max(intensity)
        return intensity

    def sum_counts_energy(
        self,
        min_: int,
        max_: int,
        normalize: bool = False,
    ) -> np.ndarray:
        intensity = np.sum(self.crop_energy(min_, max_).counts, axis=2)
        if normalize:
            return intensity / np.max(intensity)
        return intensity
