"""
Single-harmonic RF bucket geometry below transition.

Phase convention (same as BLonD): a particle receives the kick
``q V sin(phi)`` with ``phi = omega_rf * dt + phi_rf``.  Below transition
the stable fixed point is ``phi_s = arcsin(dE_turn / (q V))`` (0 for a
stationary bucket) and the unstable fixed point is ``pi - phi_s``.

Per turn, with ``a = pi h |eta| / (beta^2 E)``, the motion follows from

    H(phi, dE) = q V G(phi) - a dE^2,
    G(phi) = cos(phi) - cos(phi_s) + (phi - phi_s) sin(phi_s),

so ``H = 0`` at the bucket centre and ``H = q V G(pi - phi_s)`` on the
separatrix.  Contours are labelled by the normalised level
``lam = H / (q V G_u)``: 0 at the centre, 1 on the separatrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

TWO_PI = 2 * np.pi
_N_QUAD = 2001  # points per contour integral (cosine-clustered at the ends)


def g_func(phi, phi_s: float):
    """Potential term G(phi) of the Hamiltonian."""
    return np.cos(phi) - np.cos(phi_s) + (phi - phi_s) * np.sin(phi_s)


def g_unstable(phi_s: float) -> float:
    """G at the unstable fixed point, i.e. on the separatrix (< 0)."""
    return -2 * np.cos(phi_s) + (np.pi - 2 * phi_s) * np.sin(phi_s)


def phi_s_below_transition(
    energy_gain: float, charge: float, voltage: float
) -> float:
    """Synchronous phase in [rad]; NaN if the RF cannot supply the gain."""
    if voltage <= 0:
        return 0.0 if energy_gain == 0 else np.nan
    ratio = energy_gain / (charge * voltage)
    if abs(ratio) >= 1:
        return np.nan
    return float(np.arcsin(ratio))


def _contour_phases(phi_s: float, lam: float) -> tuple[float, float]:
    """Phase extent (left, right) of the contour at level ``lam``."""
    phi_u = np.pi - phi_s
    target = lam * g_unstable(phi_s)
    if lam >= 1:
        if phi_s < 1e-12:
            return -np.pi, np.pi
        left = brentq(lambda p: g_func(p, phi_s) - target, -np.pi, phi_s)
        return left, phi_u
    if lam <= 0:
        return phi_s, phi_s
    left_edge = _contour_phases(phi_s, 1.0)[0]
    left = brentq(lambda p: g_func(p, phi_s) - target, left_edge, phi_s)
    right = brentq(lambda p: g_func(p, phi_s) - target, phi_s, phi_u)
    return left, right


def _contour_integral(phi_s: float, lam: float) -> float:
    """Integral of sqrt(G - lam G_u) over the contour's phase extent."""
    left, right = _contour_phases(phi_s, lam)
    if right <= left:
        return 0.0
    u = np.linspace(0.0, 1.0, _N_QUAD)
    phi = left + (right - left) * 0.5 * (1 - np.cos(np.pi * u))
    integrand = np.sqrt(
        np.clip(g_func(phi, phi_s) - lam * g_unstable(phi_s), 0, None)
    )
    return float(np.trapezoid(integrand, phi))


@lru_cache(maxsize=1)
def _area_factor_table() -> tuple[np.ndarray, np.ndarray]:
    """Tabulated full-bucket integral I(phi_s) for fast interpolation."""
    phi_s = np.linspace(0, np.pi / 2 * (1 - 1e-6), 801)
    integral = np.array([_contour_integral(p, 1.0) for p in phi_s])
    return phi_s, integral


def bucket_integral(phi_s):
    """I(phi_s); I(0) = 4 sqrt(2) for a stationary bucket."""
    table_phi, table_val = _area_factor_table()
    return np.interp(phi_s, table_phi, table_val, right=0.0)


@dataclass
class Bucket:
    """
    Bucket of a single-harmonic RF system below transition.

    Parameters
    ----------
    charge
        Charge state of the particle.
    voltage
        RF voltage, in [V].
    energy_gain
        Reference energy gain per turn, in [eV].
    phi_rf
        RF phase offset as used by BLonD, in [rad].
    beta, total_energy, eta
        Reference beta, total energy [eV] and slippage factor (< 0).
    harmonic
        RF harmonic number.
    omega_rf
        RF angular frequency, in [rad/s].
    """

    charge: float
    voltage: float
    energy_gain: float
    phi_rf: float
    beta: float
    total_energy: float
    eta: float
    harmonic: float
    omega_rf: float

    def __post_init__(self):
        if self.eta >= 0:
            raise ValueError(
                "bucket.py assumes operation below transition (eta < 0)."
            )
        self.phi_s = phi_s_below_transition(
            self.energy_gain, self.charge, self.voltage
        )
        self.valid = self.voltage > 0 and np.isfinite(self.phi_s)
        self.a = np.pi * self.harmonic * abs(self.eta) / (
            self.beta**2 * self.total_energy
        )

    @property
    def phi_u(self) -> float:
        """Unstable fixed point, in [rad]."""
        return np.pi - self.phi_s

    def _scale(self) -> float:
        """sqrt(q V / a): converts sqrt(G) into energy, in [eV]."""
        return np.sqrt(self.charge * self.voltage / self.a)

    def area(self, lam: float = 1.0) -> float:
        """Area enclosed by the contour at level ``lam``, in [eVs]."""
        if not self.valid:
            return 0.0
        integral = (
            bucket_integral(self.phi_s)
            if lam >= 1
            else _contour_integral(self.phi_s, lam)
        )
        return 2 * self._scale() * integral / self.omega_rf

    def half_height(self, lam: float = 1.0) -> float:
        """Energy half-height of the contour at level ``lam``, in [eV]."""
        if not self.valid:
            return 0.0
        return self._scale() * np.sqrt(-lam * g_unstable(self.phi_s))

    def phase_extent(self, lam: float = 1.0) -> tuple[float, float]:
        """RF phase extent of the contour at level ``lam``, in [rad]."""
        return _contour_phases(self.phi_s, lam)

    def level_for_area(self, area: float) -> float:
        """Contour level enclosing ``area`` [eVs]; NaN if > bucket area."""
        if not self.valid or area >= self.area():
            return np.nan
        return brentq(lambda lam: self.area(lam) - area, 1e-9, 1.0)

    def wrap_phase(self, phi):
        """Map RF phases into the bucket window [phi_u - 2 pi, phi_u)."""
        return np.mod(phi - self.phi_u, TWO_PI) + self.phi_u - TWO_PI

    def particle_levels(self, dt, dE):
        """Contour level of every particle; inf outside the bucket."""
        if not self.valid:
            return np.full(np.shape(dt), np.inf)
        phi = self.wrap_phase(self.omega_rf * np.asarray(dt) + self.phi_rf)
        g_u = g_unstable(self.phi_s)
        hamiltonian = g_func(phi, self.phi_s) - (
            self.a / (self.charge * self.voltage)
        ) * np.asarray(dE) ** 2
        lam = hamiltonian / g_u
        left = self.phase_extent(1.0)[0]
        lam[(phi < left) | (lam > 1)] = np.inf
        return lam

    def contour(self, lam: float = 1.0, n_points: int = 400):
        """Closed contour (phi, dE) at level ``lam`` for plotting."""
        if not self.valid:
            return np.array([]), np.array([])
        left, right = self.phase_extent(lam)
        u = np.linspace(0.0, 1.0, n_points)
        phi = left + (right - left) * 0.5 * (1 - np.cos(np.pi * u))
        g_u = g_unstable(self.phi_s)
        dE = self._scale() * np.sqrt(
            np.clip(g_func(phi, self.phi_s) - lam * g_u, 0, None)
        )
        return np.concatenate([phi, phi[::-1]]), np.concatenate([dE, -dE[::-1]])

    def synchrotron_frequency(self, f_rev: float) -> float:
        """Small-amplitude synchrotron frequency, in [Hz]."""
        if not self.valid:
            return 0.0
        return f_rev * np.sqrt(
            self.harmonic
            * self.charge
            * self.voltage
            * abs(self.eta)
            * np.cos(self.phi_s)
            / (TWO_PI * self.beta**2 * self.total_energy)
        )


def voltage_for_area(
    target_area: float,
    v_max: float,
    **bucket_kwargs,
) -> tuple[float, bool]:
    """
    RF voltage giving a (moving) bucket of ``target_area`` [eVs].

    Returns
    -------
    voltage, capped
        The voltage in [V] and whether it had to be limited to ``v_max``.
    """

    def area_minus_target(voltage):
        return Bucket(voltage=voltage, **bucket_kwargs).area() - target_area

    energy_gain = bucket_kwargs["energy_gain"]
    charge = bucket_kwargs["charge"]
    v_low = abs(energy_gain) / charge * (1 + 1e-9) + 1e-9
    if area_minus_target(v_max) < 0:
        return v_max, True
    return brentq(area_minus_target, v_low, v_max, xtol=1e-6), False
