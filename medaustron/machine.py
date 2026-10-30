"""
MedAustron synchrotron model and cycle programs.

Beam parameters come from the PIMMS study (CERN-PS-99-010-DI):

* Table 1.1 -- extraction energy ranges and particles per spill.
* Table 3.8 -- adiabatic capture of the injected coasting beam.

Table 3.8 is self-consistent with the PIMMS ring (C = 75.24 m), h = 1,
injection at 20 MeV (p) / 7 MeV/u (C6+) and gamma_tr ~ 1.99: the
"initial beam length" equals the revolution period, the injected emittance
is 2 dE T_rev, and the capture voltage gives exactly K_AREA.  The default
circumference here is the MedAustron value (77.648 m).  Because both the
emittance and the bucket area scale with T_rev, the capture voltages and
K_AREA carry over unchanged.

The cycle is:

    injection -> adiabatic capture -> flat-bottom hold -> ramp
    -> flat-top hold -> RF phase jump to the unstable fixed point
    -> bunch rotation -> adiabatic debunching -> coasting beam
"""

from __future__ import annotations

import bisect
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.constants import c, physical_constants
from scipy.optimize import brentq
from scipy.special import ellipk

from .bucket import Bucket, voltage_for_area

PIMMS_CIRCUMFERENCE = 75.24  # [m], implied by Table 3.8 beam lengths
MEDAUSTRON_CIRCUMFERENCE = 77.648  # [m]

_MEV = 1e6
_M_P = physical_constants["proton mass energy equivalent in MeV"][0] * _MEV
_AMU = (
    physical_constants["atomic mass constant energy equivalent in MeV"][0]
    * _MEV
)
_M_E = physical_constants["electron mass energy equivalent in MeV"][0] * _MEV


@dataclass(frozen=True)
class Species:
    """
    Particle species and its PIMMS beam parameters.

    Energies are kinetic energies per nucleon, in [eV/u].
    """

    key: str
    label: str
    mass: float  # rest energy [eV]
    charge: int  # charge state
    nucleons: int
    injection_energy: float
    extraction_range: tuple[float, float]
    default_extraction_energy: float
    intensity: float  # particles per spill, Table 1.1 (active scanning)
    dp_p_injected: float  # half-width of the injected coasting beam
    capture_voltage: float  # [V], Table 3.8
    capture_time: float  # [s], Table 3.8
    k_area: float  # bucket area / beam emittance, Table 3.8
    pimms_capture: dict = field(default_factory=dict)  # Table 3.8 results


PROTON = Species(
    key="proton",
    label="p",
    mass=_M_P,
    charge=1,
    nucleons=1,
    injection_energy=20 * _MEV,
    extraction_range=(60 * _MEV, 250 * _MEV),
    default_extraction_energy=250 * _MEV,
    intensity=1e10,
    dp_p_injected=1.2e-3,
    capture_voltage=350.0,
    capture_time=26.8e-3,
    k_area=1.5,
    pimms_capture=dict(
        dp_p=2.441e-3, dE=96.6e3, bunch_length=819e-9, emittance=0.117
    ),
)

CARBON = Species(
    key="carbon",
    label="C6+",
    mass=12 * _AMU - 6 * _M_E,  # bare 12C nucleus
    charge=6,
    nucleons=12,
    injection_energy=7 * _MEV,
    extraction_range=(120 * _MEV, 400 * _MEV),
    default_extraction_energy=400 * _MEV,
    intensity=4e8,
    dp_p_injected=1.2e-3,
    capture_voltage=291.0,
    capture_time=40.4e-3,
    k_area=1.6,
    pimms_capture=dict(
        dp_p=2.534e-3, dE=424e3, bunch_length=1307e-9, emittance=0.827
    ),
)

SPECIES = {"proton": PROTON, "carbon": CARBON}


@dataclass
class CycleConfig:
    """
    All settings of one simulated cycle.

    ``None`` means "derive automatically" (see :func:`build_cycle`).
    Times in [s], energies in [eV/u], voltages in [V].
    """

    species: str = "proton"
    extraction_energy: float | None = None  # kinetic [eV/u]

    # machine
    circumference: float = MEDAUSTRON_CIRCUMFERENCE
    gamma_transition: float = 1.99  # derived from PIMMS Table 3.8
    harmonic: int = 1
    bending_radius: float = 4.231  # [m] PIMMS dipoles: 1.5 T at 6.35 Tm
    max_rigidity: float = 6.35  # [Tm]
    max_rf_voltage: float = 5e3
    # Optional floor for the ramp / flat-top voltage.  Near transition the
    # constant-area program asks for very little voltage, which makes the
    # synchrotron frequency (and so the adiabaticity) collapse.
    min_rf_voltage: float | None = None

    # flat bottom: adiabatic capture of the coasting beam
    capture_voltage: float | None = None  # Table 3.8
    capture_time: float | None = None  # Table 3.8
    capture_start_fraction: float = 0.01  # V_start / V_capture
    flat_bottom_hold: float = 5e-3

    # ramp: parabolic-linear-parabolic B-rho, constant bucket area
    ramp_time: float = 0.5
    ramp_rounding_time: float = 0.05
    ramp_area_factor: float | None = None  # None: keep flat-bottom bucket

    # flat top RF gymnastics
    flat_top_hold: float = 5e-3
    flat_top_voltage: float | None = None  # None: same bucket area
    unstable_phase_time: float | None = None  # None: from separatrix flow
    rotation_time: float | None = None  # None: 3/8 T_s at the bunch edge
    debunch_time: float | None = None  # None: from adiabaticity
    debunch_adiabaticity: float = 0.1
    debunch_end_fraction: float = 0.01  # V_end / V_flat_top, then RF off
    coasting_hold: float = 5e-3

    # beam control (see beam_control.py); both loops off by default
    phase_loop: bool = False
    radial_loop: bool = False
    pl_gain: float = 3000.0  # [1/s]; dipole damping time ~ 2 / pl_gain
    pl_highpass: float = 20.0  # [Hz] phase-loop high-pass corner; 0 = none
    rl_gain_p: float = 0.5  # [1] df/f per dR/R
    rl_gain_i: float = 300.0  # [1/s]
    loop_period: float = 10e-6  # [s] between loop updates
    loop_window: tuple[float, float] | None = None  # None: capture end
    # -> phase jump (no bunch to act on before; gymnastics after)

    # test perturbations for the loops (off by default)
    rf_phase_error: float = 0.0  # [deg] step in the RF phase ...
    rf_phase_error_time: float = 0.15  # [s] ... at this time
    frequency_error: float = 0.0  # [1] RF frequency error df/f ...
    frequency_error_time: float = 0.30  # [s] ... switched on at this time
    frequency_error_rise: float = 20e-3  # [s] smooth (cosine) rise time

    # beam
    n_macroparticles: int = 50_000
    injection_distribution: str = "uniform"  # uniform | parabolic | gaussian
    momentum_aperture: float = 1e-2  # |dp/p| beyond which particles are lost
    seed: int = 1
    # particles leaving the bucket re-enter from the other side (dt wrapped
    # into one RF period every `periodicity_every` turns; periodicity.py)
    periodicity: bool = True
    periodicity_every: int = 10

    # numerics / output
    backend: str = "cpp_single_core"
    n_stat_samples: int = 4000
    max_snapshot_particles: int = 20_000
    output_dir: str | None = None

    def __post_init__(self):
        if self.species not in SPECIES:
            raise ValueError(
                f"Unknown species {self.species!r}; use {list(SPECIES)}."
            )
        sp = SPECIES[self.species]
        if self.extraction_energy is None:
            self.extraction_energy = sp.default_extraction_energy
        if self.capture_voltage is None:
            self.capture_voltage = sp.capture_voltage
        if self.capture_time is None:
            self.capture_time = sp.capture_time
        if self.ramp_time < 2 * self.ramp_rounding_time:
            raise ValueError("ramp_time must be >= 2 * ramp_rounding_time.")

    @property
    def particle(self) -> Species:
        """Species preset."""
        return SPECIES[self.species]

    def to_dict(self) -> dict:
        """JSON-serialisable copy."""
        return asdict(self)


class HermiteProgram:
    """
    Fast scalar cubic Hermite interpolation with given slopes.

    Exact for the piecewise-quadratic B-rho ramp when the knots are placed
    at the segment boundaries.  The constructor signature matches what
    ``MagneticCycleByTime`` expects from an ``interpolator``.  Times
    outside the knots are clamped.
    """

    def __init__(self, times, values, derivatives):
        self.t = [float(x) for x in times]
        self.y = [float(x) for x in values]
        self.m = [float(x) for x in derivatives]

    def _segment(self, t):
        i = bisect.bisect_right(self.t, t) - 1
        return min(max(i, 0), len(self.t) - 2)

    def __call__(self, t):
        t = float(t)
        if t <= self.t[0]:
            return self.y[0]
        if t >= self.t[-1]:
            return self.y[-1]
        i = self._segment(t)
        h = self.t[i + 1] - self.t[i]
        s = (t - self.t[i]) / h
        s2, s3 = s * s, s * s * s
        return (
            (2 * s3 - 3 * s2 + 1) * self.y[i]
            + (s3 - 2 * s2 + s) * h * self.m[i]
            + (-2 * s3 + 3 * s2) * self.y[i + 1]
            + (s3 - s2) * h * self.m[i + 1]
        )

    def derivative(self, t):
        """First derivative of the interpolant."""
        t = float(t)
        if t <= self.t[0] or t >= self.t[-1]:
            return 0.0
        i = self._segment(t)
        h = self.t[i + 1] - self.t[i]
        s = (t - self.t[i]) / h
        s2 = s * s
        return (
            (6 * s2 - 6 * s) * self.y[i] / h
            + (3 * s2 - 4 * s + 1) * self.m[i]
            + (-6 * s2 + 6 * s) * self.y[i + 1] / h
            + (3 * s2 - 2 * s) * self.m[i + 1]
        )


class LinearProgram:
    """Fast scalar linear interpolation, clamped outside the knots."""

    def __init__(self, times, values):
        self.t = [float(x) for x in times]
        self.y = [float(x) for x in values]

    def __call__(self, t):
        t = float(t)
        if t <= self.t[0]:
            return self.y[0]
        if t >= self.t[-1]:
            return self.y[-1]
        i = bisect.bisect_right(self.t, t) - 1
        t0, t1 = self.t[i], self.t[i + 1]
        return self.y[i] + (self.y[i + 1] - self.y[i]) * (t - t0) / (t1 - t0)


def kinetic_to_rigidity(kinetic_per_u: float, sp: Species) -> float:
    """Magnetic rigidity [Tm] for a kinetic energy per nucleon [eV/u]."""
    total = sp.mass + sp.nucleons * kinetic_per_u
    return np.sqrt(total**2 - sp.mass**2) / (sp.charge * c)


@dataclass
class CyclePrograms:
    """Machine programs of one cycle and the derived timeline."""

    config: CycleConfig
    timeline: dict
    brho: HermiteProgram
    voltage: LinearProgram
    emittance_injected: float  # [eVs]
    ramp_bucket_area: float  # [eVs]
    flat_top_voltage: float  # [V]
    flat_top_synchrotron_frequency: float  # [Hz]
    voltage_capped: bool

    @property
    def particle(self) -> Species:
        """Species preset."""
        return self.config.particle

    @property
    def t_end(self) -> float:
        """End of the simulated cycle, in [s]."""
        return self.timeline["end"]

    def phases(self) -> list[tuple[str, float, float]]:
        """Named cycle phases (label, start, stop)."""
        tl = self.timeline
        return [
            ("capture", 0.0, tl["capture_end"]),
            ("FB hold", tl["capture_end"], tl["ramp_start"]),
            ("ramp", tl["ramp_start"], tl["ramp_end"]),
            ("FT hold", tl["ramp_end"], tl["phase_jump"]),
            ("unstable phase", tl["phase_jump"], tl["phase_jump_back"]),
            ("rotation", tl["phase_jump_back"], tl["debunch_start"]),
            ("debunching", tl["debunch_start"], tl["debunch_end"]),
            ("coasting", tl["debunch_end"], tl["end"]),
        ]

    # --- scalar programs (called once per turn by BLonD) ---------------

    def phi_rf(self, t: float) -> float:
        """RF phase: pi (stable below transition), 0 while jumped."""
        tl = self.timeline
        if tl["phase_jump"] <= t < tl["phase_jump_back"]:
            return 0.0
        return np.pi

    def phi_rf_applied(self, t: float) -> float:
        """RF phase sent to BLonD: program plus the optional test step."""
        cfg = self.config
        if cfg.rf_phase_error and t >= cfg.rf_phase_error_time:
            return self.phi_rf(t) + np.radians(cfg.rf_phase_error)
        return self.phi_rf(t)

    @property
    def loop_window(self) -> tuple[float, float]:
        """Time window in which the phase / radial loops act, in [s]."""
        if self.config.loop_window is not None:
            return tuple(self.config.loop_window)
        return self.timeline["capture_end"], self.timeline["phase_jump"]

    @property
    def beam_control_needed(self) -> bool:
        """Whether a beam-control element has to be tracked at all."""
        cfg = self.config
        return cfg.phase_loop or cfg.radial_loop or cfg.frequency_error != 0

    def momentum(self, t: float) -> float:
        """Reference momentum, in [eV/c]."""
        return self.brho(t) * self.particle.charge * c

    def total_energy(self, t: float) -> float:
        """Reference total energy, in [eV]."""
        return float(np.hypot(self.momentum(t), self.particle.mass))

    def energy_gain_per_turn(self, t: float) -> float:
        """Reference energy gain per turn q C dB-rho/dt, in [eV]."""
        return (
            self.particle.charge
            * self.config.circumference
            * self.brho.derivative(t)
        )

    def momentum_aperture_energy(self, t: float) -> float:
        """Energy offset matching the momentum aperture, in [eV]."""
        energy = self.total_energy(t)
        beta2 = 1 - (self.particle.mass / energy) ** 2
        return self.config.momentum_aperture * beta2 * energy

    def bucket(self, t: float, voltage: float | None = None) -> Bucket:
        """RF bucket at time ``t`` (optionally for another voltage)."""
        cfg = self.config
        energy = self.total_energy(t)
        gamma = energy / self.particle.mass
        beta = np.sqrt(1 - 1 / gamma**2)
        return Bucket(
            charge=self.particle.charge,
            voltage=self.voltage(t) if voltage is None else voltage,
            energy_gain=self.energy_gain_per_turn(t),
            phi_rf=self.phi_rf(t),
            beta=beta,
            total_energy=energy,
            eta=1 / cfg.gamma_transition**2 - 1 / gamma**2,
            harmonic=cfg.harmonic,
            omega_rf=cfg.harmonic * 2 * np.pi * beta * c / cfg.circumference,
        )

    # --- dense tables for plotting --------------------------------------

    def tabulate(self, n_per_phase: int = 400) -> dict:
        """Programs sampled on a time grid that resolves every phase."""
        t = np.unique(
            np.concatenate(
                [
                    np.linspace(t0, t1, n_per_phase)
                    for _, t0, t1 in self.phases()
                ]
            )
        )
        sp, cfg = self.particle, self.config
        brho = np.array([self.brho(x) for x in t])
        energy = np.array([self.total_energy(x) for x in t])
        gamma = energy / sp.mass
        beta = np.sqrt(1 - 1 / gamma**2)
        f_rev = beta * c / cfg.circumference
        voltage = np.array([self.voltage(x) for x in t])
        gain = np.array([self.energy_gain_per_turn(x) for x in t])
        buckets = [self.bucket(x) for x in t]
        return dict(
            t=t,
            brho=brho,
            b_field=brho / cfg.bending_radius,
            total_energy=energy,
            kinetic_per_u=(energy - sp.mass) / sp.nucleons,
            beta=beta,
            gamma=gamma,
            eta=1 / cfg.gamma_transition**2 - 1 / gamma**2,
            f_rev=f_rev,
            f_rf=cfg.harmonic * f_rev,
            voltage=voltage,
            energy_gain=gain,
            phi_rf=np.array([self.phi_rf(x) for x in t]),
            phi_s=np.array([b.phi_s for b in buckets]),
            bucket_area=np.array([b.area() for b in buckets]),
            bucket_half_height=np.array([b.half_height() for b in buckets]),
            f_s=np.array(
                [b.synchrotron_frequency(f) for b, f in zip(buckets, f_rev)]
            ),
        )


def _brho_knots(cfg: CycleConfig, t_start: float):
    """Knots of the parabolic-linear-parabolic B-rho ramp."""
    sp = cfg.particle
    b0 = kinetic_to_rigidity(sp.injection_energy, sp)
    b1 = kinetic_to_rigidity(cfg.extraction_energy, sp)
    t_r, t_ramp = cfg.ramp_rounding_time, cfg.ramp_time
    bdot = (b1 - b0) / (t_ramp - t_r)  # peak dB-rho/dt
    times = [
        0.0,
        t_start,
        t_start + t_r,
        t_start + t_ramp - t_r,
        t_start + t_ramp,
    ]
    values = [b0, b0, b0 + bdot * t_r / 2, b1 - bdot * t_r / 2, b1]
    slopes = [0.0, 0.0, bdot, bdot, 0.0]
    return times, values, slopes, bdot


def _check_energy(cfg: CycleConfig) -> None:
    sp = cfg.particle
    lo, hi = sp.extraction_range
    e = cfg.extraction_energy
    if not lo <= e <= hi:
        warnings.warn(
            f"{e / _MEV:.0f} MeV/u is outside the PIMMS range "
            f"{lo / _MEV:.0f}-{hi / _MEV:.0f} MeV/u for {sp.key}.",
            stacklevel=3,
        )
    gamma = 1 + sp.nucleons * e / sp.mass
    if gamma >= 0.97 * cfg.gamma_transition:
        raise ValueError(
            f"gamma = {gamma:.3f} at extraction is at/above transition "
            f"(gamma_tr = {cfg.gamma_transition}); this model keeps the "
            "RF phase fixed at the below-transition value."
        )
    if e <= sp.injection_energy:
        raise ValueError("Extraction energy must exceed injection energy.")
    brho = kinetic_to_rigidity(e, sp)
    if brho > cfg.max_rigidity * 1.001:
        warnings.warn(
            f"B-rho = {brho:.3f} Tm exceeds the {cfg.max_rigidity} Tm limit.",
            stacklevel=3,
        )


def build_cycle(cfg: CycleConfig) -> CyclePrograms:
    """
    Derive the timeline and all machine programs of a cycle.

    * Capture: ``V(t) = V0 / (1 - (1 - sqrt(V0/V1)) t/T)^2`` -- constant
      adiabaticity (rate of change of f_s relative to f_s^2).
    * Ramp: the voltage keeps the moving-bucket area equal to the
      flat-bottom bucket area (or ``ramp_area_factor`` x emittance),
      limited to ``max_rf_voltage``.
    * Flat top: phase jump by pi for ``unstable_phase_time``; jump back
      and let the stretched bunch rotate; then debunch with the inverse of
      the capture law and switch the RF off.
    """
    _check_energy(cfg)
    sp = cfg.particle

    t_cap_end = cfg.capture_time
    t_ramp_start = t_cap_end + cfg.flat_bottom_hold
    t_ramp_end = t_ramp_start + cfg.ramp_time
    knots_t, knots_b, knots_m, _ = _brho_knots(cfg, t_ramp_start)
    # Flat top extends far enough for any gymnastics; trimmed below.
    knots_t.append(t_ramp_end + 10.0)
    knots_b.append(knots_b[-1])
    knots_m.append(0.0)
    brho = HermiteProgram(knots_t, knots_b, knots_m)

    # A provisional programs object to evaluate buckets during setup.
    programs = CyclePrograms(
        config=cfg,
        timeline={},
        brho=brho,
        voltage=LinearProgram([0, 1], [cfg.capture_voltage] * 2),
        emittance_injected=np.nan,
        ramp_bucket_area=np.nan,
        flat_top_voltage=np.nan,
        flat_top_synchrotron_frequency=np.nan,
        voltage_capped=False,
    )
    programs.timeline = dict(
        phase_jump=np.inf, phase_jump_back=np.inf, end=np.inf
    )

    # Injected coasting beam: emittance = 2 dE_max T_rev.
    fb = programs.bucket(0.0, voltage=cfg.capture_voltage)
    t_rev0 = cfg.circumference / (fb.beta * c)
    de_max = sp.dp_p_injected * fb.beta**2 * fb.total_energy
    emittance = 2 * de_max * t_rev0

    # --- capture --------------------------------------------------------
    v1 = cfg.capture_voltage
    v0 = cfg.capture_start_fraction * v1
    tau = np.linspace(0, cfg.capture_time, 300)
    k_cap = (1 - np.sqrt(v0 / v1)) / cfg.capture_time
    v_times = list(tau)
    v_values = list(v0 / (1 - k_cap * tau) ** 2)

    # --- ramp -----------------------------------------------------------
    if cfg.ramp_area_factor is None:
        target_area = fb.area()
    else:
        target_area = cfg.ramp_area_factor * emittance
    ramp_t = np.linspace(t_ramp_start, t_ramp_end, 1500)
    capped = False
    v_ramp = []
    for t in ramp_t:
        b = programs.bucket(t, voltage=1.0)  # geometry only
        v, cap = voltage_for_area(
            target_area,
            cfg.max_rf_voltage,
            charge=b.charge,
            energy_gain=b.energy_gain,
            phi_rf=b.phi_rf,
            beta=b.beta,
            total_energy=b.total_energy,
            eta=b.eta,
            harmonic=b.harmonic,
            omega_rf=b.omega_rf,
        )
        capped |= cap
        v_ramp.append(v)
    if cfg.min_rf_voltage is not None:
        # Once the voltage has reached the floor it never drops below it
        # again; this never introduces a step.
        reached = np.minimum(np.maximum.accumulate(v_ramp), cfg.min_rf_voltage)
        v_ramp = np.maximum(v_ramp, reached)
    v_times += list(ramp_t)
    v_values += list(v_ramp)
    if capped:
        warnings.warn(
            "RF voltage limited to max_rf_voltage during the ramp: the "
            "bucket shrinks below the requested area.",
            stacklevel=2,
        )

    # --- flat top ---------------------------------------------------------
    v_ft = v_values[-1] if cfg.flat_top_voltage is None else (
        cfg.flat_top_voltage
    )
    ft_bucket = programs.bucket(t_ramp_end, voltage=v_ft)
    f_rev_ft = ft_bucket.omega_rf / (2 * np.pi * cfg.harmonic)
    f_s = ft_bucket.synchrotron_frequency(f_rev_ft)
    omega_s = 2 * np.pi * f_s

    # Half-length of the bunch contour (enclosing the injected emittance)
    lam = ft_bucket.level_for_area(emittance)
    left, right = ft_bucket.phase_extent(lam if np.isfinite(lam) else 1)
    half_length = (right - left) / 2

    if cfg.unstable_phase_time is None:
        # On the separatrix a particle moves from phase offset p0 to p1
        # (measured from the unstable point) in ln(tan(p1/4)/tan(p0/4))/w_s.
        # Stretch until the rms core reaches pi/2, where the new and old
        # separatrices cross; p0 ~ 0.45 x half-length (rms of a filled
        # contour).
        p0 = 0.45 * half_length
        t_unstable = np.log(np.tan(np.pi / 8) / np.tan(p0 / 4)) / omega_s
        t_unstable = max(float(t_unstable), 0.0)
    else:
        t_unstable = cfg.unstable_phase_time

    # Near the unstable point the bunch is stretched along the diagonal
    # x = p of normalised phase space; the stable rotation (clockwise)
    # brings that direction onto the energy axis after 135 deg = 3/8 T_s.
    # The stretched bunch is large, so use the pendulum synchrotron
    # frequency at the bunch edge, f_s0 * pi / (2 K(sin^2(phi/2))).
    f_s_edge = f_s * np.pi / (2 * ellipk(np.sin(half_length / 2) ** 2))
    t_rotation = (
        3 / 8 / f_s_edge if cfg.rotation_time is None else cfg.rotation_time
    )
    end_fraction = cfg.debunch_end_fraction
    if cfg.debunch_time is None:
        t_debunch = (np.sqrt(1 / end_fraction) - 1) / (
            cfg.debunch_adiabaticity * omega_s
        )
    else:
        t_debunch = cfg.debunch_time

    tl = dict(
        capture_end=t_cap_end,
        ramp_start=t_ramp_start,
        ramp_end=t_ramp_end,
        phase_jump=t_ramp_end + cfg.flat_top_hold,
    )
    tl["phase_jump_back"] = tl["phase_jump"] + t_unstable
    tl["debunch_start"] = tl["phase_jump_back"] + t_rotation
    tl["debunch_end"] = tl["debunch_start"] + t_debunch
    tl["end"] = tl["debunch_end"] + cfg.coasting_hold

    # Flat-top voltage: (optional) linear change during the hold.
    v_times += [tl["phase_jump"], tl["debunch_start"]]
    v_values += [v_ft, v_ft]
    tau = np.linspace(0, t_debunch, 300)[1:]
    k_deb = (np.sqrt(1 / end_fraction) - 1) / t_debunch
    v_times += list(tl["debunch_start"] + tau)
    v_values += list(v_ft / (1 + k_deb * tau) ** 2)
    # RF off for the coasting beam
    v_times += [tl["debunch_end"] + 1e-9, tl["end"] + 1.0]
    v_values += [0.0, 0.0]

    programs.timeline = tl
    programs.voltage = LinearProgram(v_times, v_values)
    programs.emittance_injected = emittance
    programs.ramp_bucket_area = target_area
    programs.flat_top_voltage = v_ft
    programs.flat_top_synchrotron_frequency = f_s
    programs.voltage_capped = capped

    # Keep the magnetic cycle exactly as long as the simulated cycle.
    brho_t = [x for x in knots_t[:-1]] + [tl["end"]]
    programs.brho = HermiteProgram(brho_t, knots_b, knots_m)
    return programs


def estimate_turns(programs: CyclePrograms) -> int:
    """Approximate number of turns in the cycle."""
    tab = programs.tabulate(200)
    return int(np.trapezoid(tab["f_rev"], tab["t"]))


def solve_time_of_energy(programs: CyclePrograms, kinetic_per_u: float):
    """Time on the ramp at which a kinetic energy [eV/u] is reached."""
    sp = programs.particle
    tl = programs.timeline
    target = kinetic_to_rigidity(kinetic_per_u, sp)
    return brentq(
        lambda t: programs.brho(t) - target, tl["ramp_start"], tl["ramp_end"]
    )
