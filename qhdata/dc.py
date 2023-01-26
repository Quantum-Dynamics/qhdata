from __future__ import annotations
from functools import cached_property
from pathlib import Path
import typing as t

import numpy as np

from .util import (
    load_raw_data,
    _check_min_max,
    _get_matching_range,
    _remove_offset_1axis,
)


class VIData:

    @staticmethod
    def load(
        file_voltage: t.Union[str, Path],
        file_current: t.Union[str, Path],
        save_npy: bool = True,
    ) -> VIData:
        return VIData(
            load_raw_data(file_voltage, save_npy=save_npy),
            load_raw_data(file_current, save_npy=save_npy),
        )

    def __init__(self, voltage: np.ndarray, current: np.ndarray) -> None:
        self._voltage = voltage
        self._current = current

    @property
    def voltage(self) -> np.ndarray:
        return self._voltage

    @property
    def current(self) -> np.ndarray:
        return self._current

    @cached_property
    def gradient_resistance(self) -> np.ndarray:
        return np.gradient(self.voltage, self.current)

    def remove_offset(self, point: float = 0., tol: float = 1e-6) -> VIData:
        current_offset_removed = _remove_offset_1axis(
            self.current,
            self.voltage,
            point=point,
            tol=tol,
        )
        return VIData(self.voltage, current_offset_removed)

    def crop_voltage(self, min_: float, max_: float) -> VIData:
        _check_min_max(min_, max_)
        indices = _get_matching_range(self.voltage, min_, max_)
        return VIData(self.voltage[indices], self.current[indices])

    def crop_current(self, min_: float, max_: float) -> VIData:
        _check_min_max(min_, max_)
        indices = _get_matching_range(self.current, min_, max_)
        return VIData(self.voltage[indices], self.current[indices])


class FETData:

    @staticmethod
    def load(
        file_voltage: t.Union[str, Path],
        file_current: t.Union[str, Path],
        file_gate_voltage: t.Union[str, Path],
        save_npy: bool = True,
    ) -> None:
        return FETData(
            load_raw_data(file_voltage, save_npy=save_npy),
            load_raw_data(file_current, save_npy=save_npy),
            load_raw_data(file_gate_voltage, save_npy=save_npy),
        )

    @staticmethod
    def load_and_concat(
        files_voltage: t.Tuple[t.Union[str, Path], ...],
        files_current: t.Tuple[t.Union[str, Path], ...],
        gate_voltage: t.Union[str, Path, np.ndarray],
        save_npy: bool = True,
    ) -> VIData:
        voltages = np.array(
            [load_raw_data(f, save_npy=save_npy) for f in files_voltage]
        )
        currents = np.array(
            [load_raw_data(f, save_npy=save_npy) for f in files_current]
        )
        if not isinstance(gate_voltage, np.ndarray):
            gate_voltage: np.ndarray = load_raw_data(
                gate_voltage,
                save_npy=save_npy,
            )

        return FETData(voltages, currents, gate_voltage)

    def __init__(
        self,
        voltage: np.ndarray,
        current: np.ndarray,
        gate_voltage: np.ndarray,
    ) -> None:
        if voltage.ndim != 2:
            raise ValueError("'voltage' must be a 2d ndarray.")
        if current.ndim != 2:
            raise ValueError("'current' must be a 2d ndarray.")
        if gate_voltage.ndim != 1:
            raise ValueError("'gate_voltage' must be a 1d ndarray.")

        self._voltage = voltage
        self._current = current
        self._gate_voltage = gate_voltage

    @property
    def voltage(self) -> np.ndarray:
        return self._voltage

    @property
    def current(self) -> np.ndarray:
        return self._current

    @property
    def gate_voltage(self) -> np.ndarray:
        return self._gate_voltage

    @cached_property
    def gradient_resistance(self) -> np.ndarray:
        return np.gradient(self.voltage, self.current, axis=1)

    def remove_offset(self, point: float = 0., tol: float = 1e-6) -> FETData:
        current_offset_removed = np.apply_along_axis(
            _remove_offset_1axis,
            1,
            self.current,
            self.voltage[0, :],
            point=point,
            tol=tol,
        )
        return FETData(
            self.voltage,
            current_offset_removed,
            self.gate_voltage,
        )

    def crop_voltage(self, min_: float, max_: float) -> FETData:
        _check_min_max(min_, max_)
        indices = np.apply_along_axis(
            _get_matching_range,
            1,
            self.voltage,
            min_,
            max_,
        )
        return FETData(
            self.voltage[indices],
            self.current[indices],
            self.gate_voltage,
        )

    def crop_current(self, min_: float, max_: float) -> FETData:
        _check_min_max(min_, max_)
        indices = np.apply_along_axis(
            _get_matching_range,
            1,
            self._current,
            min_,
            max_,
        )
        return FETData(
            self.voltage[indices],
            self.current[indices],
            self.gate_voltage,
        )

    def crop_gate_voltage(self, min_: float, max_: float) -> FETData:
        _check_min_max(min_, max_)
        indices = _get_matching_range(self.gate_voltage, min_, max_)
        return FETData(
            self.voltage[indices, :],
            self.current[indices, :],
            self.gate_voltage[indices],
        )

    def save(
        self,
        file_voltage: t.Union[str, Path, None] = None,
        file_current: t.Union[str, Path, None] = None,
        file_gate_voltage: t.Union[str, Path, None] = None,
    ) -> None:
        if file_voltage is not None:
            np.save(file_voltage, self.voltage)
        if file_current is not None:
            np.save(file_current, self.current)
        if file_gate_voltage is not None:
            np.save(file_gate_voltage, self.gate_voltage)
