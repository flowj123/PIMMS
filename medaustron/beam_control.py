"""
Phase and radial loops for the MedAustron cycle.

Built on BLonD 3's beam-feedback framework: ``MedAustronBeamControl`` is a
``BeamFeedbackBase``, and BLonD applies its output ``delta_omega_rf`` to
the RF station, which accumulates the resulting RF phase slip turn by turn.

BLonD 3 ships beam controls for the PSB, PS, SPS and LHC
(``blond.physics.feedbacks.accelerators``).  They are not used as-is
because, on the ``blonder`` branch (2026-10-13):

* ``BeamFeedbackBase.update_phase_error`` sets ``dphi = phi_beam`` without
  subtracting the synchronous phase (BLonD 2 did); on an accelerating ramp
  a phase loop would then drag the bunch away from phi_s;
* ``PSBBeamControl`` pre-computes the turns on which it acts from the
  revolution period at injection, which changes 2-4x along this ramp.

Loop models (sampled every ``period`` seconds, k = sample index):

* Phase loop: phase error e = wrap(phi_beam - phi_s), where phi_beam is the
  phase of the beam's RF component, atan2(<sin phi>, <cos phi>) with
  phi = omega_rf dt + phi_rf (the quantity BLonD's ``beam_phase`` returns,
  computed here from macroparticles instead of a binned profile).  A
  first-order high-pass (corner ``pl_highpass``) removes static offsets,
  e.g. the centroid shift of a large bunch in an accelerating bucket, so
  the loop only damps dipole oscillations:
      y_k = a (y_{k-1} + e_k - e_{k-1}),  a = tau / (tau + dt_k)
      domega_PL = pl_gain * y_k                         [rad/s]
  With a proportional loop the dipole motion obeys
  e'' + pl_gain e' + omega_s^2 e = 0, i.e. damping time 2 / pl_gain.
* Radial loop: relative radial offset from the beam momentum offset,
  x = alpha_c <dE> / (beta^2 E) (= dR/R), and a PI controller acting on
  the relative RF frequency:
      u_k = u_{k-1} + rl_gain_p (x_k - x_{k-1}) + rl_gain_i x_k dt_k
      domega_RL = u_k * omega_rf_design
  A frequency change df/f moves the beam by dR/R = alpha_c/|eta| df/f, so
  the integral time constant is |eta| / (alpha_c rl_gain_i).

Total correction (below transition; both terms are negative feedback):
    delta_omega_rf = domega_error - domega_PL - domega_RL
where ``domega_error`` is an optional RF frequency error used to test the
radial loop.  Both loops act only inside ``window``.  When the window
closes, the phase-loop output returns to zero and the radial-loop output
is held, so the RF frequency does not jump.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from blond.physics.feedbacks.beam_feedback import BeamFeedbackBase

if TYPE_CHECKING:  # pragma: no cover
    from blond.core.beam.base import BeamBaseClass

    from .machine import CyclePrograms

TWO_PI = 2 * np.pi
# Macroparticles used per measurement.  The beam-phase noise of the
# measurement (~ sigma_phi / sqrt(N)) is fed back by the phase loop, so use
# as many as affordable: 5000 gave a visible +-1-2 deg jitter at 3000/s.
_N_MEASURE = 50_000


def _wrap(phase: float) -> float:
    return float(np.mod(phase + np.pi, TWO_PI) - np.pi)


class MedAustronBeamControl(BeamFeedbackBase):
    """
    Beam-phase loop and radial loop with adjustable gains.

    Parameters
    ----------
    programs
        Cycle programs; provide the synchronous phase phi_s(t).
    phase_loop, radial_loop
        Switch the loops on or off.
    pl_gain
        Phase-loop gain, in [1/s] (damping time of dipole oscillations
        ~ 2 / pl_gain).  Schedulable: ``schedule("pl_gain", ...)``.
    pl_highpass
        Corner frequency of the phase-loop high-pass, in [Hz]; 0 = none.
    rl_gain_p
        Radial-loop proportional gain, in [1] (df/f per dR/R).
        Schedulable.
    rl_gain_i
        Radial-loop integral gain, in [1/s].  Schedulable.
    period
        Time between loop updates, in [s].
    window
        (t_on, t_off): loops act for t_on <= t < t_off, in [s].
    frequency_error, frequency_error_time, frequency_error_rise
        Optional RF frequency error df/f switched on at the given time
        with a smooth (cosine) rise, to exercise the radial loop.
    """

    def __init__(
        self,
        programs: CyclePrograms,
        phase_loop: bool = True,
        radial_loop: bool = True,
        pl_gain: float = 3000.0,
        pl_highpass: float = 20.0,
        rl_gain_p: float = 0.5,
        rl_gain_i: float = 300.0,
        period: float = 10e-6,
        window: tuple[float, float] = (0.0, np.inf),
        frequency_error: float = 0.0,
        frequency_error_time: float = 0.0,
        frequency_error_rise: float = 0.0,
        name: str = "Beam control",
    ):
        super().__init__(profile=None, name=name)
        self._programs = programs  # private: kept out of BLonD's printout
        self.phase_loop = phase_loop
        self.radial_loop = radial_loop
        self.pl_gain = pl_gain
        self.pl_highpass = pl_highpass
        self.rl_gain_p = rl_gain_p
        self.rl_gain_i = rl_gain_i
        self.period = period
        self.window = (float(window[0]), float(window[1]))
        self.frequency_error = frequency_error
        self.frequency_error_time = frequency_error_time
        self.frequency_error_rise = frequency_error_rise

        self.phi_s = 0.0
        self.dr_over_r = 0.0
        self.domega_pl = 0.0
        self.domega_rl = 0.0
        self.domega_error = 0.0

        self._act = False
        self._elapsed = 0.0
        self._t_last = -np.inf
        self._pl_active = False
        self._rl_active = False
        self._e_prev = 0.0
        self._y_prev = 0.0
        self._x_prev = 0.0
        self._u = 0.0

        self._register_schedulable_variables(
            "pl_gain", "rl_gain_p", "rl_gain_i"
        )

    def check_main_rf_stations_with_cavity_feedback(self):
        """
        Warn only if *some* main-harmonic stations have a cavity feedback.

        BLonD's version tests ``not all_none or not all_not_none``, which is
        also true when no station has one, so it always warns here.
        """
        feedbacks = [
            rf.get_main_harmonic_cavity_feedback()
            for rf in self._main_cavities
        ]
        n_none = sum(fb is None for fb in feedbacks)
        if 0 < n_none < len(feedbacks):
            super().check_main_rf_stations_with_cavity_feedback()

    def frequency_error_fraction(self, t: float) -> float:
        """Fraction (0..1) of the test frequency error applied at ``t``."""
        x = t - self.frequency_error_time
        if x <= 0:
            return 0.0
        if x >= self.frequency_error_rise:
            return 1.0
        return 0.5 * (1 - np.cos(np.pi * x / self.frequency_error_rise))

    def _needed(self, t: float) -> bool:
        loops = (self.phase_loop or self.radial_loop) and (
            self.window[0] <= t < self.window[1]
        )
        return loops or self.frequency_error != 0

    def update_beam_attributes(self, beam: BeamBaseClass):
        """Measure beam phase and radial offset on loop-update turns."""
        t = float(beam.reference.time)
        self._act = self._needed(t) and t - self._t_last >= self.period
        if not self._act:
            return
        gap = t - self._t_last
        # first update, or first after a pause: use the nominal period
        self._elapsed = gap if gap <= 10 * self.period else self.period
        self._t_last = t

        n = beam.common_array_size
        step = max(1, n // _N_MEASURE)
        dt = np.asarray(beam.read_partial_dt())[::step]
        dE = np.asarray(beam.read_partial_dE())[::step]
        cavity = self._main_cavities[0]
        phase = (
            cavity.get_main_harmonic_omega_rf() * dt
            + cavity.get_main_harmonic_phi_rf()
        )
        self.phi_beam = float(
            np.arctan2(np.sin(phase).sum(), np.cos(phase).sum())
        )
        self.phi_s = self._programs.bucket(t).phi_s
        self.dphi = _wrap(self.phi_beam - self.phi_s)

        self.average_de = float(dE.mean()) if len(dE) else 0.0
        reference = beam.reference
        self.dr_over_r = (
            self._simulation.ring.momentum_compaction_factor
            * self.average_de
            / (reference.beta**2 * reference.total_energy)
        )
        self.drho = self.dr_over_r * self._simulation.ring.circumference / (
            TWO_PI
        )

    def update_frequency_correction(self, beam: BeamBaseClass):
        """Phase loop, radial loop and frequency error -> delta_omega_rf."""
        if self._act:
            t = self._t_last
            dt_k = self._elapsed
            omega_design = self._main_cavities[
                0
            ].get_main_harmonic_omega_rf_design()
            in_window = self.window[0] <= t < self.window[1]

            # --- phase loop: high-passed proportional ----------------------
            if self.phase_loop and in_window:
                e = self.dphi
                if not self._pl_active:  # bumpless switch-on
                    self._e_prev, self._y_prev = e, 0.0
                    self._pl_active = True
                if self.pl_highpass > 0:
                    tau = 1 / (TWO_PI * self.pl_highpass)
                    y = tau / (tau + dt_k) * (self._y_prev + e - self._e_prev)
                else:
                    y = e
                self._e_prev, self._y_prev = e, y
                self.domega_pl = self.pl_gain * y
            else:
                self._pl_active = False
                self.domega_pl = 0.0

            # --- radial loop: PI on dR/R, output held when switched off ---
            if self.radial_loop and in_window:
                x = self.dr_over_r
                if not self._rl_active:  # bumpless switch-on
                    self._x_prev = x
                    self._rl_active = True
                self._u += (
                    self.rl_gain_p * (x - self._x_prev)
                    + self.rl_gain_i * x * dt_k
                )
                self._x_prev = x
            else:
                self._rl_active = False
            self.domega_rl = self._u * omega_design

            self.domega_error = (
                self.frequency_error_fraction(t)
                * self.frequency_error
                * omega_design
            )
        self.delta_omega_rf = (
            self.domega_error - self.domega_pl - self.domega_rl
        )
