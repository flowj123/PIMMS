"""
Longitudinal periodicity: particles that leave the bucket on one side
re-enter it from the other.

BLonD 3 (``blonder`` branch, 2026-10-13) has no equivalent of BLonD 2's
``RingAndRFTracker(periodicity=True)``: neither the drift nor the RF kick
wraps ``dt``, which grows without bound for a particle that slips away
from the bunch.  The single-harmonic kick ``q V sin(omega_rf dt + phi_rf)``
is itself periodic in ``dt``, so such a particle already feels the RF of
the neighbouring bucket, but

* the bucket copies sit at ``dt_s + m T_rf(t)``.  For h = 1 they are all
  the same bucket, one turn apart, yet they move apart as T_rf shrinks
  along the ramp.  A particle in copy m must carry a spurious extra
  momentum offset dp/p of about ``m |dT_rf/dt| / |eta|`` to keep up: at
  most 0.8e-5 (p, 250 MeV) to 2.4e-5 (C, 400 MeV/u) per copy, early in
  the ramp, against a bucket half-height of ~3.5e-3;
* a binned profile (collective effects, diagnostics) only sees particles
  inside its time window.

``Periodicity`` shifts ``dt`` by whole RF periods ``T_rf = 2 pi / omega_rf``
so that every particle lies in one RF period.  The window is the one the
diagnostics use: RF phase in [-pi, pi) in the beam frame, where the bunch
centre is at phi_s and the cut lies in the gap between buckets, opposite
the bunch.  A shift by one RF period leaves the kick unchanged.  For h = 1
the RF period is the revolution period (plus the beam-control frequency
offset), so the particle comes round the ring and re-enters the bucket
from the other side, as in BLonD 2.

A particle moves by ``h |eta| dp/p`` of an RF period per turn: at most
~2.5e-3 inside the bucket and ~7e-3 at the momentum aperture.  Wrapping
every ``each_turn_i`` = 10 turns therefore lets it overshoot the cut by
a few per cent of a period before it is moved back, which is harmless
because the kick is periodic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond.core.base import BeamPhysicsRelevant

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.beam.base import BeamBaseClass
    from blond.physics.rf_station import SingleHarmonicRFStation

    from .machine import CyclePrograms

TWO_PI = 2 * np.pi


class Periodicity(BeamPhysicsRelevant):
    """
    Wrap ``dt`` into one RF period around the bunch.

    Parameters
    ----------
    programs
        Cycle programs; give the programmed RF phase, so the window stays
        on the beam during the RF phase jump.
    rf
        The RF station; gives omega_rf and the accumulated phase slip.
    each_turn_i
        Wrap every n-th turn.
    name
        Element name.

    Attributes
    ----------
    n_wrapped
        Number of times a macroparticle was moved by one or more RF
        periods (the initial placement of the beam is not counted).
    """

    def __init__(
        self,
        programs: CyclePrograms,
        rf: SingleHarmonicRFStation,
        each_turn_i: int = 10,
        name: str = "Periodicity",
    ):
        super().__init__(name=name)
        self._programs = programs  # private: kept out of BLonD's printout
        self._rf = rf
        self.each_turn_i = int(each_turn_i)
        self.n_wrapped = 0
        self._placed = False

    def window(self, t: float) -> tuple[float, float, float]:
        """
        RF-period window at time ``t``.

        Returns
        -------
        dt_low, dt_high
            Window limits, in [s].
        period
            RF period, in [s].
        """
        programs, rf = self._programs, self._rf
        omega = float(rf.get_main_harmonic_omega_rf())
        # RF phase in the beam frame, as in Recorder._rf_frame:
        # psi = omega dt + offset, bunch centre at phi_s
        offset = (
            float(rf.delta_phi_rf)
            + programs.phi_rf_applied(t)
            - programs.phi_rf(t)
            + np.pi
        )
        return (-np.pi - offset) / omega, (np.pi - offset) / omega, (
            TWO_PI / omega
        )

    def _track(self, beam: BeamBaseClass) -> None:
        """Move particles outside the window by whole RF periods."""
        if beam.common_array_size == 0:
            return
        dt_low, dt_high, period = self.window(float(beam.reference.time))
        dt = beam.write_partial_dt()
        if dt.min() >= dt_low and dt.max() < dt_high:
            self._placed = True
            return
        outside = (dt < dt_low) | (dt >= dt_high)
        dt[outside] -= period * np.floor((dt[outside] - dt_low) / period)
        if self._placed:
            self.n_wrapped += int(np.count_nonzero(outside))
        self._placed = True
