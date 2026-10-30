"""
BLonD 3 simulation of a MedAustron cycle, flat bottom to flat top.

Usage (from the BLonD workspace, in the ``blond3`` conda environment)::

    python -m medaustron.run_cycle                         # protons, 250 MeV
    python -m medaustron.run_cycle --species carbon --energy 400
    python -m medaustron.run_cycle --species proton --energy 60 --show
    python -m medaustron.run_cycle --help                  # all options

or from Python / Spyder::

    from medaustron.run_cycle import run
    from medaustron.machine import CycleConfig
    results = run(CycleConfig(species="carbon", extraction_energy=120e6))

Figures, data (``results.npz``), the configuration and a text summary are
written to ``results/<species>_<energy>MeVu/``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.constants import c

from blond import (
    Beam,
    BoxLosses,
    Coasting,
    DriftSimple,
    MagneticCycleByTime,
    ParticleType,
    Ring,
    ScheduledFunctional,
    Simulation,
    SingleHarmonicRFStation,
    momentum_compaction_factor,
    setup_backend,
)

# from .beam_control import MedAustronBeamControl
from .bucket import Bucket
from .machine import (
    CycleConfig,
    CyclePrograms,
    HermiteProgram,
    build_cycle,
    estimate_turns,
)
from .periodicity import Periodicity

TWO_PI = 2 * np.pi
WORKSPACE = Path(__file__).resolve().parent.parent


def wrap_to_pi(phase):
    """Wrap phases to [-pi, pi)."""
    return np.mod(np.asarray(phase) + np.pi, TWO_PI) - np.pi


def contour_metrics(bucket: Bucket, levels, fraction: float) -> dict:
    """
    Bunch contour (Hamiltonian level) containing ``fraction`` of particles.

    These are the "full" quantities of PIMMS Table 3.8: half-height,
    momentum spread, total length and enclosed area of the contour.
    """
    nan = dict(level=np.nan, area=np.nan, dE=np.nan, dp_p=np.nan,
               length=np.nan)
    if not bucket.valid or len(levels) == 0:
        return nan
    # "higher": no interpolation, which would give inf - inf for particles
    # outside the bucket (level = inf)
    lam = float(np.quantile(levels, fraction, method="higher"))
    if not np.isfinite(lam) or lam > 1:
        return nan
    left, right = bucket.phase_extent(lam)
    half_height = bucket.half_height(lam)
    return dict(
        level=lam,
        area=bucket.area(lam),
        dE=half_height,
        dp_p=half_height / (bucket.beta**2 * bucket.total_energy),
        length=(right - left) / bucket.omega_rf,
    )


@dataclass
class Snapshot:
    """Phase-space snapshot (subsampled) at one moment of the cycle."""

    label: str
    t: float
    turn: int
    psi: np.ndarray  # RF phase in the phi_rf = pi frame, [rad]
    dE: np.ndarray  # [eV]
    n_alive: int
    metrics: dict = field(default_factory=dict)
    dE_centre: float = 0.0  # [eV] synchronous energy of the actual RF


class Recorder:
    """
    BLonD callback recording beam diagnostics during the cycle.

    Every ``stride`` turns it stores beam statistics; at given times it
    stores phase-space snapshots and line-density profiles.
    """

    def __init__(self, programs: CyclePrograms, rf, n_turns_est: int,
                 beam_control=None):
        self.each_turn_i = 1  # BLonD calls us every turn
        self.programs = programs
        self.rf = rf
        self.beam_control = beam_control
        cfg = programs.config
        self.stride = max(1, n_turns_est // cfg.n_stat_samples)
        self.max_snapshot = cfg.max_snapshot_particles
        self.rng = np.random.default_rng(cfg.seed)
        self.n_initial = None

        tl = programs.timeline
        # The phase jump and rotation last only ~1 ms: sample them finely.
        self.fine_window = (
            tl["phase_jump"] - 0.5e-3,
            tl["debunch_start"] + 0.2 * (tl["debunch_end"] - tl["debunch_start"]),
        )
        f_rev_ft = programs.bucket(tl["ramp_end"]).omega_rf / (
            2 * np.pi * cfg.harmonic
        )
        n_fine = (self.fine_window[1] - self.fine_window[0]) * f_rev_ft
        self.fine_stride = max(1, min(self.stride, int(n_fine // 600)))
        self.snapshot_plan = [
            ("End of capture", tl["capture_end"]),
            ("Mid ramp", 0.5 * (tl["ramp_start"] + tl["ramp_end"])),
            ("Flat top", tl["ramp_end"]),
            ("Phase jump", tl["phase_jump"]),
            ("Jump back", tl["phase_jump_back"]),
            ("End of rotation", tl["debunch_start"]),
            ("Mid debunching", 0.5 * (tl["debunch_start"] + tl["debunch_end"])),
            ("End of debunching", tl["debunch_end"]),
        ]
        self.profile_times = np.concatenate(
            [
                np.linspace(0, tl["ramp_start"], 80, endpoint=False),
                np.linspace(tl["ramp_start"], tl["ramp_end"], 150,
                            endpoint=False),
                np.linspace(tl["ramp_end"], tl["end"], 150),
            ]
        )
        self.profile_edges = np.linspace(-np.pi, np.pi, 129)

        self.stats: dict[str, list] = {
            k: []
            for k in (
                "t", "turn", "reference_energy", "voltage_applied",
                "phi_rf_applied", "n_alive", "centroid", "sigma_phase",
                "bunch_length_4sigma", "mean_dE", "sigma_dE", "dp_p_rms",
                "emittance_rms", "emittance_95", "in_bucket",
                "phase_error", "radial_offset", "rf_frequency_offset",
                "pl_correction", "rl_correction", "frequency_error",
            )
        }
        self.snapshots: list[Snapshot] = []
        self.profiles: list[np.ndarray] = []
        self.profile_t: list[float] = []
        self._next_snapshot = 0
        self._next_profile = 0

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _coordinates(beam):
        return (
            np.array(beam.read_partial_dt(), copy=True),
            np.array(beam.read_partial_dE(), copy=True),
        )

    def _rf_frame(self, t: float):
        """
        Bucket of the RF as actually applied (loops, test errors).

        Returns
        -------
        bucket
            Bucket with the actual RF phase and frequency.
        dE_centre
            Synchronous energy of that RF relative to the reference [eV]:
            an RF frequency offset moves the beam off the reference energy.
        phase_offset
            ``psi = wrap(bucket.omega_rf * dt + phase_offset)`` is the RF
            phase in the frame where the programmed phase is pi, so the
            bucket centre is at phi_s (as for an unperturbed cycle).
        """
        p, rf = self.programs, self.rf
        b = p.bucket(t)
        d_omega = float(rf.delta_omega_rf)
        d_phi = float(rf.delta_phi_rf) + p.phi_rf_applied(t) - p.phi_rf(t)
        bucket = replace(b, phi_rf=b.phi_rf + d_phi,
                         omega_rf=b.omega_rf + d_omega)
        dE_centre = (b.beta**2 * b.total_energy * d_omega / b.omega_rf
                     / abs(b.eta))
        return bucket, dE_centre, d_phi + np.pi

    def snapshot(self, simulation, beam, label: str) -> None:
        """Store a (subsampled) phase-space snapshot now."""
        t = float(beam.reference.time)
        dt, dE = self._coordinates(beam)
        bucket, dE_centre, offset = self._rf_frame(t)
        levels = bucket.particle_levels(dt, dE - dE_centre)
        metrics = {
            f"contour_{int(f * 100)}": contour_metrics(bucket, levels, f)
            for f in (0.95, 0.99)
        }
        metrics["in_bucket"] = float(np.mean(levels <= 1)) if len(dt) else 0
        n_alive = len(dt)
        if n_alive > self.max_snapshot:
            pick = self.rng.choice(n_alive, self.max_snapshot, replace=False)
            dt, dE = dt[pick], dE[pick]
        self.snapshots.append(
            Snapshot(
                label=label,
                t=t,
                turn=int(simulation.turn_counter.value),
                psi=wrap_to_pi(bucket.omega_rf * dt + offset),
                dE=dE,
                n_alive=n_alive,
                metrics=metrics,
                dE_centre=dE_centre,
            )
        )

    def _record_stats(self, simulation, beam, t: float) -> None:
        dt, dE = self._coordinates(beam)
        bucket, dE_centre, offset = self._rf_frame(t)
        s = self.stats
        s["t"].append(t)
        s["turn"].append(int(simulation.turn_counter.value))
        s["reference_energy"].append(float(beam.reference.total_energy))
        s["voltage_applied"].append(float(self.rf.voltage))
        s["phi_rf_applied"].append(float(self.rf.phi_rf))
        s["n_alive"].append(len(dt))
        bc = self.beam_control
        s["rf_frequency_offset"].append(
            float(self.rf.delta_omega_rf) / TWO_PI
        )
        s["pl_correction"].append(-bc.domega_pl / TWO_PI if bc else 0.0)
        s["rl_correction"].append(-bc.domega_rl / TWO_PI if bc else 0.0)
        s["frequency_error"].append(bc.domega_error / TWO_PI if bc else 0.0)
        if len(dt) < 2:
            for k in ("centroid", "sigma_phase", "bunch_length_4sigma",
                      "mean_dE", "sigma_dE", "dp_p_rms", "emittance_rms",
                      "emittance_95", "in_bucket", "phase_error",
                      "radial_offset"):
                s[k].append(np.nan)
            return
        psi = wrap_to_pi(bucket.omega_rf * dt + offset)
        centroid = float(np.angle(np.mean(np.exp(1j * psi))))
        rel = wrap_to_pi(psi - centroid)
        t_rel = rel / bucket.omega_rf
        cov = np.cov(t_rel, dE)
        levels = bucket.particle_levels(dt, dE - dE_centre)
        s["centroid"].append(centroid)
        s["sigma_phase"].append(float(np.std(rel)))
        s["bunch_length_4sigma"].append(4 * float(np.std(t_rel)))
        s["mean_dE"].append(float(np.mean(dE)))
        s["sigma_dE"].append(float(np.std(dE)))
        s["dp_p_rms"].append(
            float(np.std(dE)) / (bucket.beta**2 * bucket.total_energy)
        )
        s["emittance_rms"].append(
            4 * np.pi * float(np.sqrt(max(np.linalg.det(cov), 0)))
        )
        s["emittance_95"].append(contour_metrics(bucket, levels, 0.95)["area"])
        s["in_bucket"].append(float(np.mean(levels <= 1)))
        s["phase_error"].append(
            float(wrap_to_pi(centroid - self.programs.bucket(t).phi_s))
        )
        # dR = R alpha_c dp/p, from the mean momentum offset of the beam
        cfg = self.programs.config
        s["radial_offset"].append(
            cfg.circumference / TWO_PI / cfg.gamma_transition**2
            * float(np.mean(dE)) / (bucket.beta**2 * bucket.total_energy)
        )

    def _record_profile(self, beam, t: float) -> None:
        dt = np.asarray(beam.read_partial_dt())
        bucket, _, offset = self._rf_frame(t)
        psi = wrap_to_pi(bucket.omega_rf * dt + offset)
        hist, _ = np.histogram(psi, self.profile_edges)
        self.profiles.append(hist / max(self.n_initial, 1))
        self.profile_t.append(t)

    # -- BLonD callback ------------------------------------------------------

    def __call__(self, simulation, beam) -> None:
        turn = simulation.turn_counter.value
        t = beam.reference.time
        fine = self.fine_window[0] <= t <= self.fine_window[1]
        if turn % (self.fine_stride if fine else self.stride) == 0:
            self._record_stats(simulation, beam, float(t))
        if (
            self._next_snapshot < len(self.snapshot_plan)
            and t >= self.snapshot_plan[self._next_snapshot][1]
        ):
            self.snapshot(
                simulation, beam, self.snapshot_plan[self._next_snapshot][0]
            )
            self._next_snapshot += 1
        if (
            self._next_profile < len(self.profile_times)
            and t >= self.profile_times[self._next_profile]
        ):
            self._record_profile(beam, float(t))
            while (
                self._next_profile < len(self.profile_times)
                and t >= self.profile_times[self._next_profile]
            ):
                self._next_profile += 1

    def arrays(self) -> dict:
        """Recorded statistics as numpy arrays."""
        return {k: np.asarray(v) for k, v in self.stats.items()}


@dataclass
class CycleResults:
    """Everything produced by :func:`run`."""

    config: CycleConfig
    programs: CyclePrograms
    table: dict  # programs on a dense time grid
    stats: dict  # recorded beam statistics
    snapshots: list[Snapshot]
    profiles: np.ndarray
    profile_t: np.ndarray
    profile_edges: np.ndarray
    final_dE: np.ndarray
    final_dp_p: np.ndarray
    injected_dp_p: np.ndarray
    n_initial: int
    n_turns: int
    runtime: float
    output_dir: Path
    figures: dict = field(default_factory=dict)
    n_wrapped: int | None = None  # periodicity wraps (None: not recorded)


def build_simulation(programs: CyclePrograms):
    """Assemble the BLonD 3 ring, magnetic cycle, beam and simulation."""
    cfg = programs.config
    sp = programs.particle
    particle = ParticleType(mass=sp.mass, charge=sp.charge)

    ring = Ring(circumference=cfg.circumference)
    drift = DriftSimple(
        orbit_length=cfg.circumference,
        momentum_compaction_factor=momentum_compaction_factor(
            transition_gamma=cfg.gamma_transition
        ),
    )
    # Below transition the stable phase of BLonD's sine kick is phi_rf = pi.
    rf = SingleHarmonicRFStation(
        harmonic=cfg.harmonic,
        voltage=programs.voltage(0.0),
        phi_rf=np.pi,
        name="RF cavity",
    )
    rf.schedule(
        "voltage",
        ScheduledFunctional(
            lambda turn_i, reference_time: programs.voltage(reference_time)
        ),
    )
    rf.schedule(
        "phi_rf_design",
        ScheduledFunctional(
            lambda turn_i, reference_time: programs.phi_rf_applied(
                reference_time
            )
        ),
    )
    # Momentum aperture: particles beyond |dp/p| are removed (e.g. beam
    # left behind by the ramp).
    de_ap = programs.momentum_aperture_energy(0.0)
    aperture = BoxLosses(
        purge_flagged_macroparticles=True, e_min=-de_ap, e_max=de_ap
    )
    # Particles left behind drift out by only ~dE_turn per turn; checking
    # every 100 turns saves the per-turn purge overhead.
    aperture.each_turn_i = 100
    aperture.schedule(
        "e_min",
        ScheduledFunctional(
            lambda turn_i, reference_time: -programs.momentum_aperture_energy(
                reference_time
            )
        ),
    )
    aperture.schedule(
        "e_max",
        ScheduledFunctional(
            lambda turn_i, reference_time: programs.momentum_aperture_energy(
                reference_time
            )
        ),
    )
    elements = [drift, rf, aperture]
    beam_control = None
    if programs.beam_control_needed:
        beam_control = MedAustronBeamControl(
            programs,
            phase_loop=cfg.phase_loop,
            radial_loop=cfg.radial_loop,
            pl_gain=cfg.pl_gain,
            pl_highpass=cfg.pl_highpass,
            rl_gain_p=cfg.rl_gain_p,
            rl_gain_i=cfg.rl_gain_i,
            period=cfg.loop_period,
            window=programs.loop_window,
            frequency_error=cfg.frequency_error,
            frequency_error_time=cfg.frequency_error_time,
            frequency_error_rise=cfg.frequency_error_rise,
        )
        rf.attach_beam_feedback(beam_control)
        elements.append(beam_control)
    ring.add_elements(elements, reorder=True)
    if cfg.periodicity:
        # after the drift: BLonD cannot auto-order a custom element
        ring.add_element(
            Periodicity(programs, rf, each_turn_i=cfg.periodicity_every)
        )

    # The exact parabolic-linear-parabolic B-rho ramp: cubic Hermite through
    # the segment boundaries with their analytic slopes.
    brho = programs.brho
    magnetic_cycle = MagneticCycleByTime(
        reference_particle=particle,
        reference_time=np.array(brho.t),
        reference_values=np.array(brho.y) * sp.charge * c,
        in_unit="momentum",
        interpolator=HermiteProgram,
        derivatives=np.array(brho.m),
    )

    simulation = Simulation(ring=ring, magnetic_cycle=magnetic_cycle)
    beam = Beam(intensity=sp.intensity, particle_type=particle)
    return simulation, beam, rf, beam_control


def injected_energy_distribution(programs: CyclePrograms):
    """Energy bins and weights of the injected coasting beam."""
    cfg = programs.config
    b = programs.bucket(0.0)
    de_max = programs.particle.dp_p_injected * b.beta**2 * b.total_energy
    kind = cfg.injection_distribution
    if kind == "uniform":
        bins = np.linspace(-de_max, de_max, 401)
        weights = np.ones_like(bins)
    elif kind == "parabolic":
        bins = np.linspace(-de_max, de_max, 401)
        weights = 1 - (bins / de_max) ** 2
    elif kind == "gaussian":  # "full" spread taken as +-2 sigma
        bins = np.linspace(-2 * de_max, 2 * de_max, 801)
        weights = np.exp(-0.5 * (bins / (de_max / 2)) ** 2)
    else:
        raise ValueError(f"Unknown injection_distribution {kind!r}")
    return bins, weights / weights.sum()


def _output_dir(cfg: CycleConfig) -> Path:
    if cfg.output_dir is not None:
        return Path(cfg.output_dir)
    energy = cfg.extraction_energy / 1e6
    name = f"{cfg.species}_{energy:g}MeVu"
    # loops / test perturbations get their own folder (keep references)
    name += "_PL" if cfg.phase_loop else ""
    name += "_RL" if cfg.radial_loop else ""
    if cfg.rf_phase_error or cfg.frequency_error:
        name += "_test"
    name += "" if cfg.periodicity else "_noperiodicity"
    return WORKSPACE / "results" / name


def run(cfg: CycleConfig | None = None, plot: bool = True,
        show: bool = False) -> CycleResults:
    """Build, run and post-process one cycle."""
    cfg = CycleConfig() if cfg is None else cfg
    sp = cfg.particle
    out = _output_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)

    setup_backend(cfg.backend)
    programs = build_cycle(cfg)
    table = programs.tabulate()
    n_est = estimate_turns(programs)
    print(
        f"MedAustron cycle: {sp.label} to "
        f"{cfg.extraction_energy / 1e6:g} MeV/u, "
        f"{cfg.n_macroparticles:,} macroparticles, ~{n_est:,} turns"
    )
    _print_timeline(programs)

    simulation, beam, rf, beam_control = build_simulation(programs)
    simulation.print_one_turn_execution_order()

    bins, weights = injected_energy_distribution(programs)
    simulation.prepare_beam(
        beam=beam,
        preparation_routine=Coasting(
            n_macroparticles=cfg.n_macroparticles,
            energy_bins=bins,
            energy_profile=weights,
            seed=cfg.seed,
        ),
    )
    b0 = programs.bucket(0.0)
    injected_dp_p = np.array(beam.read_partial_dE()) / (
        b0.beta**2 * b0.total_energy
    )

    recorder = Recorder(programs, rf, n_est, beam_control)
    recorder.n_initial = len(injected_dp_p)
    recorder.snapshot(simulation, beam, "Injection")
    recorder._record_profile(beam, 0.0)

    t0 = time.time()
    simulation.run_simulation(beams=(beam,), callbacks=[recorder])
    runtime = time.time() - t0
    recorder.snapshot(simulation, beam, "End of cycle")

    t_end = float(beam.reference.time)
    b_end = programs.bucket(t_end)
    periodicity = [
        e for e in simulation.ring.elements.elements
        if isinstance(e, Periodicity)
    ]
    final_dE = np.array(beam.read_partial_dE())
    results = CycleResults(
        config=cfg,
        programs=programs,
        table=table,
        stats=recorder.arrays(),
        snapshots=recorder.snapshots,
        profiles=np.array(recorder.profiles),
        profile_t=np.array(recorder.profile_t),
        profile_edges=recorder.profile_edges,
        final_dE=final_dE,
        final_dp_p=final_dE / (b_end.beta**2 * b_end.total_energy),
        injected_dp_p=injected_dp_p,
        n_initial=recorder.n_initial,
        n_turns=int(simulation.turn_counter.value) + 1,
        runtime=runtime,
        output_dir=out,
        n_wrapped=periodicity[0].n_wrapped if periodicity else None,
    )
    summary = summarise(results)
    print(summary)
    (out / "summary.txt").write_text(summary)
    save(results)
    if plot:
        from . import diagnostics

        results.figures = diagnostics.plot_all(results, show=show)
    print(f"Results written to {out}")
    return results


def _print_timeline(programs: CyclePrograms) -> None:
    for label, t0, t1 in programs.phases():
        print(f"  {label:15s} {t0 * 1e3:9.2f} -> {t1 * 1e3:9.2f} ms")


def capture_benchmark(results: CycleResults) -> list[tuple]:
    """Simulated end-of-capture bunch vs PIMMS Table 3.8."""
    sp = results.programs.particle
    cfg = results.config
    snap = next(s for s in results.snapshots if s.label == "End of capture")
    m99 = snap.metrics["contour_99"]
    scale = cfg.circumference / 75.24  # lengths/emittance scale with T_rev
    pimms = sp.pimms_capture
    eff = snap.metrics["in_bucket"] * snap.n_alive / results.n_initial
    return [
        ("full dp/p (99% contour)", f"+/-{m99['dp_p']:.3e}",
         f"+/-{pimms['dp_p']:.3e}"),
        ("full dE [keV]", f"+/-{m99['dE'] / 1e3:.1f}",
         f"+/-{pimms['dE'] / 1e3:.1f}"),
        ("bunch length [ns]", f"{m99['length'] * 1e9:.0f}",
         f"{pimms['bunch_length'] * scale * 1e9:.0f} (C-scaled)"),
        ("emittance 99% [eVs]", f"{m99['area']:.3f}",
         f"{pimms['emittance'] * scale:.3f} (C-scaled)"),
        ("captured fraction", f"{eff * 100:.1f} %", "-"),
    ]


def beam_control_summary(results: CycleResults) -> str:
    """Loop settings and the beam's phase / radial excursions."""
    cfg, p, st = results.config, results.programs, results.stats
    t_on, t_off = p.loop_window
    pl = (f"ON (gain {cfg.pl_gain:g} /s, high-pass {cfg.pl_highpass:g} Hz)"
          if cfg.phase_loop else "off")
    rl = (f"ON (P {cfg.rl_gain_p:g}, I {cfg.rl_gain_i:g} /s)"
          if cfg.radial_loop else "off")
    lines = [
        f"phase loop           : {pl}",
        f"radial loop          : {rl}",
        f"loop window          : {t_on * 1e3:.1f} -> {t_off * 1e3:.1f} ms",
    ]
    if cfg.rf_phase_error:
        lines.append(f"test RF phase step   : {cfg.rf_phase_error:g} deg at "
                     f"{cfg.rf_phase_error_time * 1e3:g} ms")
    if cfg.frequency_error:
        lines.append(f"test RF freq. error  : df/f = {cfg.frequency_error:g}"
                     f" from {cfg.frequency_error_time * 1e3:g} ms "
                     f"({cfg.frequency_error_rise * 1e3:g} ms rise)")
    if "phase_error" in st:
        sel = (st["t"] >= p.timeline["ramp_start"]) & (st["t"] < t_off)
        if sel.any():
            phase = np.degrees(st["phase_error"][sel])
            radial = st["radial_offset"][sel] * 1e3
            lines += [
                f"phase error (ramp)   : mean {np.nanmean(phase):+.1f} deg,"
                f" peak-to-peak {np.nanmax(phase) - np.nanmin(phase):.1f} deg",
                f"radial offset (ramp) : max |dR| "
                f"{np.nanmax(np.abs(radial)):.2f} mm, at end of window "
                f"{radial[-1]:+.2f} mm",
            ]
    return "\n".join(lines)


def summarise(results: CycleResults) -> str:
    """Human-readable summary of the run."""
    p, cfg, st = results.programs, results.config, results.stats
    sp = p.particle
    tab = results.table
    ramp = (tab["t"] >= p.timeline["ramp_start"]) & (
        tab["t"] <= p.timeline["ramp_end"]
    )
    lines = [
        "=" * 64,
        f"MedAustron cycle -- {sp.label}, "
        f"{sp.injection_energy / 1e6:g} -> "
        f"{cfg.extraction_energy / 1e6:g} MeV/u",
        "=" * 64,
        f"turns tracked        : {results.n_turns:,} "
        f"({results.runtime:.0f} s)",
        f"circumference        : {cfg.circumference} m, "
        f"gamma_tr = {cfg.gamma_transition}, h = {cfg.harmonic}",
        f"injected emittance   : {p.emittance_injected:.4f} eVs "
        f"(coasting, dp/p = +/-{sp.dp_p_injected:.1e})",
        f"ramp bucket area     : {p.ramp_bucket_area:.4f} eVs "
        f"(K = {p.ramp_bucket_area / p.emittance_injected:.2f})",
        f"peak RF voltage      : {tab['voltage'][ramp].max():.0f} V "
        f"(limit {cfg.max_rf_voltage:.0f} V"
        f"{', REACHED' if p.voltage_capped else ''})",
        f"max synchronous phase: {np.degrees(np.nanmax(tab['phi_s'])):.1f} deg",
        f"flat-top voltage     : {p.flat_top_voltage:.1f} V, "
        f"f_s = {p.flat_top_synchrotron_frequency:.0f} Hz",
        f"unstable phase       : "
        f"{(p.timeline['phase_jump_back'] - p.timeline['phase_jump']) * 1e3:.3f}"
        f" ms, rotation "
        f"{(p.timeline['debunch_start'] - p.timeline['phase_jump_back']) * 1e3:.3f}"
        f" ms, debunching "
        f"{(p.timeline['debunch_end'] - p.timeline['debunch_start']) * 1e3:.1f}"
        " ms",
        beam_control_summary(results),
        periodicity_summary(results),
        "",
        "End of capture vs PIMMS Table 3.8:",
        f"  {'quantity':26s} {'BLonD':>16s}   {'PIMMS':>20s}",
    ]
    for name, sim, ref in capture_benchmark(results):
        lines.append(f"  {name:26s} {sim:>16s}   {ref:>20s}")
    n_end = len(results.final_dE)
    final = results.final_dp_p
    lines += [
        "",
        f"surviving particles  : {n_end / results.n_initial * 100:.1f} %",
        f"final energy         : "
        f"{(st['reference_energy'][-1] - sp.mass) / sp.nucleons / 1e6:.2f}"
        " MeV/u",
        f"final dp/p (coasting): rms {np.std(final):.3e}, "
        f"99% range [{np.quantile(final, 0.005):.3e}, "
        f"{np.quantile(final, 0.995):.3e}]",
        f"final dp/p (injected): rms {np.std(results.injected_dp_p):.3e}",
    ]
    return "\n".join(lines) + "\n"


def periodicity_summary(results: CycleResults) -> str:
    """One line on the longitudinal periodicity (dt wrapping)."""
    cfg = results.config
    if not cfg.periodicity:
        return "periodicity          : off (dt not wrapped)"
    line = (
        f"periodicity          : on, dt wrapped into one RF period "
        f"every {cfg.periodicity_every} turns"
    )
    if results.n_wrapped is not None:
        line += f"; {results.n_wrapped:,} macroparticle wraps"
    return line


def save(results: CycleResults) -> None:
    """Store configuration, programs and diagnostics to the output folder."""
    out = results.output_dir
    cfg_dict = results.config.to_dict()
    cfg_dict["timeline"] = {
        k: float(v) for k, v in results.programs.timeline.items()
    }
    (out / "config.json").write_text(json.dumps(cfg_dict, indent=2))
    data = {f"program_{k}": v for k, v in results.table.items()}
    data.update({f"stats_{k}": v for k, v in results.stats.items()})
    data.update(
        profiles=results.profiles,
        profile_t=results.profile_t,
        profile_edges=results.profile_edges,
        final_dE=results.final_dE,
        final_dp_p=results.final_dp_p,
        injected_dp_p=results.injected_dp_p,
    )
    for i, s in enumerate(results.snapshots):
        data[f"snapshot{i}_psi"] = s.psi
        data[f"snapshot{i}_dE"] = s.dE
        data[f"snapshot{i}_t"] = s.t
    data["snapshot_labels"] = np.array([s.label for s in results.snapshots])
    np.savez_compressed(out / "results.npz", **data)
    info = dict(
        n_initial=results.n_initial,
        n_turns=results.n_turns,
        runtime=results.runtime,
        n_wrapped=results.n_wrapped,
        snapshots=[
            dict(label=s.label, t=s.t, turn=s.turn, n_alive=s.n_alive,
                 metrics=s.metrics, dE_centre=s.dE_centre)
            for s in results.snapshots
        ],
    )
    (out / "run_info.json").write_text(json.dumps(info, indent=2))


def load(output_dir) -> CycleResults:
    """Reload a finished run (e.g. to re-plot without re-tracking)."""
    out = Path(output_dir)
    cfg_dict = json.loads((out / "config.json").read_text())
    cfg_dict.pop("timeline", None)
    cfg_dict.setdefault("periodicity", False)  # runs before it existed
    cfg = CycleConfig(**cfg_dict)
    programs = build_cycle(cfg)
    data = np.load(out / "results.npz")
    info_file = out / "run_info.json"
    info = json.loads(info_file.read_text()) if info_file.exists() else {}
    snap_info = {s["label"]: s for s in info.get("snapshots", [])}

    snapshots = []
    for i, label in enumerate(data["snapshot_labels"]):
        label = str(label)
        psi, dE = data[f"snapshot{i}_psi"], data[f"snapshot{i}_dE"]
        t = float(data[f"snapshot{i}_t"])
        meta = snap_info.get(label)
        if meta is None:  # older runs: recompute from the stored particles
            bucket = programs.bucket(t)
            levels = bucket.particle_levels((psi - np.pi) / bucket.omega_rf,
                                            dE)
            meta = dict(
                turn=-1, n_alive=len(psi),
                metrics={
                    "contour_95": contour_metrics(bucket, levels, 0.95),
                    "contour_99": contour_metrics(bucket, levels, 0.99),
                    "in_bucket": float(np.mean(levels <= 1)),
                },
            )
        snapshots.append(Snapshot(label=label, t=t, turn=meta["turn"],
                                  psi=psi, dE=dE, n_alive=meta["n_alive"],
                                  metrics=meta["metrics"],
                                  dE_centre=meta.get("dE_centre", 0.0)))
    stats = {k[6:]: data[k] for k in data.files if k.startswith("stats_")}
    return CycleResults(
        config=cfg,
        programs=programs,
        table=programs.tabulate(),
        stats=stats,
        snapshots=snapshots,
        profiles=data["profiles"],
        profile_t=data["profile_t"],
        profile_edges=data["profile_edges"],
        final_dE=data["final_dE"],
        final_dp_p=data["final_dp_p"],
        injected_dp_p=data["injected_dp_p"],
        n_initial=info.get("n_initial", len(data["injected_dp_p"])),
        n_turns=info.get("n_turns", int(stats["turn"][-1]) + 1),
        runtime=info.get("runtime", np.nan),
        output_dir=out,
        n_wrapped=info.get("n_wrapped"),
    )


def parse_args(argv=None) -> tuple[CycleConfig, argparse.Namespace]:
    """Command-line interface -> :class:`CycleConfig`."""
    ap = argparse.ArgumentParser(
        description="BLonD 3 MedAustron cycle (flat bottom to flat top).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--species", choices=["proton", "carbon"],
                    default="proton")
    ap.add_argument("--energy", type=float, default=None,
                    help="extraction kinetic energy [MeV/u]; default 250 "
                    "(p) / 400 (C)")
    ap.add_argument("--n-macroparticles", type=int, default=None)
    ap.add_argument("--circumference", type=float, default=None,
                    help="[m]")
    ap.add_argument("--gamma-tr", type=float, default=None)
    ap.add_argument("--ramp-time", type=float, default=None, help="[s]")
    ap.add_argument("--ramp-rounding", type=float, default=None, help="[s]")
    ap.add_argument("--v-max", type=float, default=None,
                    help="max RF voltage [V]")
    ap.add_argument("--v-min", type=float, default=None,
                    help="RF voltage floor on ramp/flat top [V] (once "
                    "reached); useful close to transition")
    ap.add_argument("--flat-top-voltage", type=float, default=None,
                    help="[V]; default keeps the ramp bucket area")
    ap.add_argument("--unstable-time", type=float, default=None,
                    help="time on the unstable phase [s]; default auto")
    ap.add_argument("--rotation-time", type=float, default=None,
                    help="bunch rotation time [s]; default 3/8 T_s (bunch edge)")
    ap.add_argument("--debunch-time", type=float, default=None,
                    help="adiabatic debunching time [s]; default auto")
    ap.add_argument("--phase-loop", action="store_true",
                    help="switch the beam-phase loop on")
    ap.add_argument("--radial-loop", action="store_true",
                    help="switch the radial loop on")
    ap.add_argument("--pl-gain", type=float, default=None,
                    help="phase-loop gain [1/s] (damping time ~2/gain); "
                    "default 3000")
    ap.add_argument("--pl-highpass", type=float, default=None,
                    help="phase-loop high-pass corner [Hz]; default 20")
    ap.add_argument("--rl-gain-p", type=float, default=None,
                    help="radial-loop proportional gain [1]; default 0.5")
    ap.add_argument("--rl-gain-i", type=float, default=None,
                    help="radial-loop integral gain [1/s]; default 300")
    ap.add_argument("--loop-period", type=float, default=None,
                    help="time between loop updates [s]; default 10e-6")
    ap.add_argument("--loop-window", type=float, nargs=2, default=None,
                    metavar=("T_ON", "T_OFF"),
                    help="loops act between these times [s]; default "
                    "end of capture -> phase jump")
    ap.add_argument("--rf-phase-error", type=float, nargs=2, default=None,
                    metavar=("DEG", "T"),
                    help="test: RF phase step of DEG degrees at time T [s]")
    ap.add_argument("--frequency-error", type=float, nargs=2, default=None,
                    metavar=("DF_F", "T"),
                    help="test: RF frequency error df/f from time T [s]")
    ap.add_argument("--no-periodicity", action="store_true",
                    help="do not wrap dt into one RF period (BLonD 3 "
                         "default)")
    ap.add_argument("--distribution",
                    choices=["uniform", "parabolic", "gaussian"],
                    default=None, help="injected energy distribution")
    ap.add_argument("--backend", default=None,
                    help="BLonD backend: cpp_single_core, numba, cpp, auto")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="5,000 macroparticles for a fast check")
    ap.add_argument("--show", action="store_true",
                    help="open the figures after the run")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--replot", metavar="DIR", default=None,
                    help="re-plot a finished run instead of tracking")
    args = ap.parse_args(argv)

    kwargs = dict(species=args.species)
    mapping = {
        "n_macroparticles": args.n_macroparticles,
        "circumference": args.circumference,
        "gamma_transition": args.gamma_tr,
        "ramp_time": args.ramp_time,
        "ramp_rounding_time": args.ramp_rounding,
        "max_rf_voltage": args.v_max,
        "min_rf_voltage": args.v_min,
        "pl_gain": args.pl_gain,
        "pl_highpass": args.pl_highpass,
        "rl_gain_p": args.rl_gain_p,
        "rl_gain_i": args.rl_gain_i,
        "loop_period": args.loop_period,
        "loop_window": (tuple(args.loop_window)
                        if args.loop_window else None),
        "flat_top_voltage": args.flat_top_voltage,
        "unstable_phase_time": args.unstable_time,
        "rotation_time": args.rotation_time,
        "debunch_time": args.debunch_time,
        "injection_distribution": args.distribution,
        "backend": args.backend,
        "output_dir": args.output_dir,
    }
    kwargs.update({k: v for k, v in mapping.items() if v is not None})
    if args.energy is not None:
        kwargs["extraction_energy"] = args.energy * 1e6
    kwargs["phase_loop"] = args.phase_loop
    kwargs["radial_loop"] = args.radial_loop
    kwargs["periodicity"] = not args.no_periodicity
    if args.rf_phase_error is not None:
        kwargs["rf_phase_error"], kwargs["rf_phase_error_time"] = (
            args.rf_phase_error
        )
    if args.frequency_error is not None:
        kwargs["frequency_error"], kwargs["frequency_error_time"] = (
            args.frequency_error
        )
    if args.quick and args.n_macroparticles is None:
        kwargs["n_macroparticles"] = 5_000
    return CycleConfig(**kwargs), args


def main(argv=None) -> None:
    """Command-line entry point."""
    cfg, args = parse_args(argv)
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    if args.replot is not None:
        from . import diagnostics

        results = load(args.replot)
        print(summarise(results))
        diagnostics.plot_all(results, show=args.show)
        print(f"Figures written to {results.output_dir}")
    else:
        run(cfg, plot=not args.no_plots, show=args.show)


if __name__ == "__main__":
    main()
