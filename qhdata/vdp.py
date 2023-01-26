from functools import cached_property

import numpy as np


class VdpData:

    def __init__(
        self,
        voltage_cd: np.ndarray,
        voltage_da: np.ndarray,
        current_ab: np.ndarray,
        current_bc: np.ndarray,
    ) -> None:
        self._voltage_cd = voltage_cd
        self._voltage_da = voltage_da
        self._current_ab = current_ab
        self._current_bc = current_bc

    @property
    def voltage_cd(self) -> None:
        return self._voltage_cd

    @property
    def voltage_da(self) -> None:
        return self._voltage_da

    @property
    def current_ab(self) -> None:
        return self._current_ab

    @property
    def current_bc(self) -> None:
        return self._current_bc

    @cached_property
    def resistance_ab_cd(self) -> None:
        min_current = np.min(self.current_ab)
        return self.voltage_cd / (self.current_ab + min_current * 1e-12)

    @cached_property
    def resistance_bc_da(self) -> None:
        min_current = np.min(self.current_bc)
        return self.voltage_da / (self.current_bc + min_current * 1e-12)
