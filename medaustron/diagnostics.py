"""Diagnostic plots of a simulated MedAustron cycle."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from scipy.constants import c, e

from blond import AllowPlotting

if TYPE_CHECKING:  # pragma: no cover
    from .run_cycle import CycleResults

MS = 1e3
PHASE_COLOURS = {
    "capture": "#dbe9f6",
    "ramp": "#eeeeee",
    "unstable phase": "#f9d5d3",
    "rotation": "#fbe7c6",
    "debunching": "#e3f1dc",
}


def _shade_phases(ax, programs, x_offset=0.0):
    """Shade the cycle phases on a time axis in [ms]."""
    for name, t0, t1 in programs.phases():
        colour = PHASE_COLOURS.get(name)
        if colour is None:
            continue
        ax.axvspan((t0 - x_offset) * MS, (t1 - x_offset) * MS,
                   color=colour, lw=0, zorder=0)


def _phase_legend(fig):
    """Legend explaining the phase shading."""
    handles = [Patch(color=col, label=name)
               for name, col in PHASE_COLOURS.items()]
    fig.legend(handles=handles, loc="outside lower center",
               ncol=len(handles), fontsize=8, frameon=False)


def _title(results):
    cfg, sp = results.config, results.programs.particle
    return (
        f"MedAustron {sp.label}: {sp.injection_energy / 1e6:g} -> "
        f"{cfg.extraction_energy / 1e6:g} MeV/u "
        f"(C = {cfg.circumference} m, h = {cfg.harmonic}, "
        f"$\\gamma_{{tr}}$ = {cfg.gamma_transition})"
    )


def _tracked_kinetic_per_u(results):
    sp = results.programs.particle
    return (results.stats["reference_energy"] - sp.mass) / sp.nucleons


def _tracked_f_rev(results):
    sp = results.programs.particle
    gamma = results.stats["reference_energy"] / sp.mass
    beta = np.sqrt(1 - 1 / gamma**2)
    return beta * c / results.config.circumference


def plot_cycle_programs(results):
    """B field, energy, f_rev, RF voltage, phi_s and bucket area."""
    p, tab, st = results.programs, results.table, results.stats
    cfg = results.config
    t = tab["t"] * MS
    ts = st["t"] * MS
    every = max(1, len(ts) // 60)  # markers for the tracked values

    fig, axes = plt.subplots(3, 2, figsize=(13, 10.5), sharex=True,
                             layout="constrained")
    ax = axes.ravel()

    ax[0].plot(t, tab["b_field"], "C0")
    ax[0].set_ylabel("Dipole field B [T]")
    sec = ax[0].secondary_yaxis(
        "right",
        functions=(lambda b: b * cfg.bending_radius,
                   lambda br: br / cfg.bending_radius),
    )
    sec.set_ylabel(r"B$\rho$ [Tm]")
    ax[0].set_title(f"B field program ($\\rho$ = {cfg.bending_radius} m)")

    ax[1].plot(t, tab["kinetic_per_u"] / 1e6, "C1", label="program")
    ax[1].plot(ts[::every], _tracked_kinetic_per_u(results)[::every] / 1e6,
               "k.", ms=3, label="BLonD reference")
    ax[1].set_ylabel("Kinetic energy [MeV/u]")
    ax[1].set_title("Energy program")
    ax[1].legend(loc="center right", fontsize=8)

    ax[2].plot(t, tab["f_rev"] / 1e6, "C2", label="program")
    ax[2].plot(ts[::every], _tracked_f_rev(results)[::every] / 1e6, "k.",
               ms=3, label="BLonD reference")
    ax[2].set_ylabel(r"$f_{rev}$ [MHz]")
    ax[2].set_title(f"Revolution frequency ($f_{{RF}}$ = {cfg.harmonic} "
                    r"$f_{rev}$)")
    ax[2].legend(loc="center right", fontsize=8)

    ax[3].plot(t, tab["voltage"], "C3", label=r"$V_{RF}$ program")
    ax[3].plot(t, tab["energy_gain"] / p.particle.charge, "C3--", lw=1,
               label=r"$V \sin\phi_s$ = $\Delta E_{turn}/q$")
    ax[3].plot(ts[::every], st["voltage_applied"][::every], "k.", ms=3,
               label="applied in BLonD")
    if tab["voltage"].max() > 0.5 * cfg.max_rf_voltage:
        ax[3].axhline(cfg.max_rf_voltage, color="k", ls=":", lw=1,
                      label="V max")
    ax[3].set_ylabel("RF voltage [V]")
    ax[3].set_title(
        f"RF voltage program (limit {cfg.max_rf_voltage / 1e3:g} kV)"
    )
    ax[3].legend(loc="upper right", fontsize=8)

    ax[4].plot(t, np.degrees(tab["phi_s"]), "C4")
    tl = p.timeline
    ax[4].axvspan(tl["phase_jump"] * MS, tl["phase_jump_back"] * MS,
                  color="C3", alpha=0.6, lw=0)
    ax[4].annotate(
        "RF phase jump +180$\\degree$\n(unstable fixed point)",
        xy=(tl["phase_jump"] * MS, 0), xytext=(-150, 25),
        textcoords="offset points", fontsize=8,
        arrowprops=dict(arrowstyle="->", color="C3"),
    )
    ax[4].set_ylabel(r"$\phi_s$ [deg]")
    ax[4].set_title(r"Synchronous phase ($\phi_s = \arcsin(\Delta E/qV)$, "
                    "below transition)")

    ax[5].plot(t, tab["bucket_area"], "C5", label="bucket area")
    ax[5].axhline(p.emittance_injected, color="k", ls="--", lw=1,
                  label="injected emittance")
    ax[5].plot(ts, st["emittance_95"], ".", color="C0", ms=2,
               label="beam, 95% contour")
    ax[5].plot(ts, st["emittance_rms"], ".", color="C2", ms=2,
               label=r"beam, 4$\pi\sigma_t\sigma_E$")
    ax[5].set_ylabel("Area [eVs]")
    ax[5].set_title("Bucket area")
    ax[5].set_ylim(0, 1.3 * np.nanmax(tab["bucket_area"]))
    ax[5].legend(loc="upper right", fontsize=8)

    for a in ax:
        _shade_phases(a, p)
        a.grid(alpha=0.3)
    for a in axes[-1]:
        a.set_xlabel("Time in cycle [ms]")
    fig.suptitle(_title(results))
    _phase_legend(fig)
    return fig


def plot_beam_evolution(results):
    """Bunch length, momentum spread, emittance, capture and phase."""
    p, tab, st = results.programs, results.table, results.stats
    t, ts = tab["t"] * MS, st["t"] * MS
    n0 = results.n_initial

    fig, axes = plt.subplots(3, 2, figsize=(13, 10.5), sharex=True,
                             layout="constrained")
    ax = axes.ravel()

    ax[0].plot(ts, st["bunch_length_4sigma"] * 1e9, "C0.", ms=2,
               label=r"beam 4$\sigma$")
    ax[0].plot(t, 1e9 / tab["f_rf"], "k:", lw=1, label=r"$T_{RF}$")
    ax[0].set_ylabel("Bunch length [ns]")
    ax[0].set_title(r"Bunch length (4$\sigma$)")
    ax[0].legend(fontsize=8)

    ax[1].plot(ts, st["dp_p_rms"] * 1e3, "C1.", ms=2, label="beam rms")
    dp_bucket = tab["bucket_half_height"] / (
        tab["beta"] ** 2 * tab["total_energy"]
    )
    ax[1].plot(t, dp_bucket * 1e3, "k:", lw=1, label="bucket half-height")
    ax[1].set_ylabel(r"$\Delta p/p$ [$10^{-3}$]")
    ax[1].set_title("Momentum spread")
    ax[1].legend(fontsize=8)

    ax[2].plot(t, tab["bucket_area"], "C5", label="bucket area")
    ax[2].plot(ts, st["emittance_95"], "C0.", ms=2, label="95% contour")
    ax[2].plot(ts, st["emittance_rms"], "C2.", ms=2,
               label=r"4$\pi\sqrt{\det\Sigma}$ (rms)")
    ax[2].axhline(p.emittance_injected, color="k", ls="--", lw=1,
                  label="injected")
    ax[2].set_ylabel("Emittance [eVs]")
    ax[2].set_title("Longitudinal emittance")
    ax[2].set_ylim(0, 1.3 * np.nanmax(tab["bucket_area"]))
    ax[2].legend(fontsize=8)

    ax[3].plot(ts, 100 * st["in_bucket"] * st["n_alive"] / n0, "C3",
               label="inside separatrix")
    ax[3].plot(ts, 100 * st["n_alive"] / n0, "k", lw=1,
               label="surviving (momentum aperture)")
    ax[3].set_ylabel("Fraction of injected [%]")
    ax[3].set_ylim(-2, 105)
    ax[3].set_title("Capture and transmission")
    ax[3].legend(fontsize=8, loc="lower left")

    ax[4].plot(ts, np.degrees(st["centroid"]), "C4.", ms=2,
               label="bunch centroid")
    ax[4].plot(t, np.degrees(tab["phi_s"]), "k", lw=1, label=r"$\phi_s$")
    ax[4].set_ylabel("RF phase [deg]")
    ax[4].set_title("Bunch phase vs synchronous phase")
    ax[4].legend(fontsize=8)

    f_s = tab["f_s"]
    ax[5].plot(t, f_s, "C2")
    ax[5].set_ylabel(r"$f_s$ [Hz]", color="C2")
    ax[5].set_title("Synchrotron frequency and adiabaticity")
    adiab = np.full_like(f_s, np.nan)
    ok = f_s > 0
    omega_s = 2 * np.pi * f_s[ok]
    adiab[ok] = np.abs(np.gradient(omega_s, tab["t"][ok])) / omega_s**2
    twin = ax[5].twinx()
    twin.plot(t, adiab, "C3", lw=1)
    twin.set_ylabel(r"$|\dot\omega_s|/\omega_s^2$", color="C3")
    twin.set_yscale("log")
    twin.set_ylim(1e-4, 10)

    for a in ax:
        _shade_phases(a, p)
        a.grid(alpha=0.3)
    for a in axes[-1]:
        a.set_xlabel("Time in cycle [ms]")
    fig.suptitle(_title(results))
    _phase_legend(fig)
    return fig


def _separatrix_psi(bucket):
    """Separatrix in the phi_rf = pi frame, [deg] and [keV]."""
    phi, dE = bucket.contour(1.0)
    if len(phi) == 0:
        return []
    psi = np.degrees(phi - (bucket.phi_rf - np.pi))
    # A bucket centred near +-180 deg (after the phase jump) is drawn twice
    return [(psi + shift, dE / 1e3) for shift in (-360, 0, 360)]


def plot_phase_space(results):
    """Phase-space snapshots with the separatrix of the moment."""
    p = results.programs
    snaps = results.snapshots
    n = len(snaps)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.4 * nrow),
                             layout="constrained")
    axes = np.atleast_1d(axes).ravel()
    for ax, snap in zip(axes, snaps):
        bucket = p.bucket(snap.t)
        h_max = max(bucket.half_height(), 1.0)
        data_max = np.quantile(np.abs(snap.dE), 0.995) if len(snap.dE) else 1
        y_lim = 1.25 * max(h_max, data_max) / 1e3
        n_bins = int(np.clip(np.sqrt(len(snap.psi)) / 1.5, 40, 110))
        ax.hist2d(
            np.degrees(snap.psi), snap.dE / 1e3, bins=(n_bins, n_bins),
            range=((-180, 180), (-y_lim, y_lim)), cmap="viridis",
            norm=LogNorm(), cmin=1,
        )
        for x, y in _separatrix_psi(bucket):
            ax.plot(x, y, "r", lw=1)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-y_lim, y_lim)
        e_kin = (bucket.total_energy - p.particle.mass) / p.particle.nucleons
        ax.set_title(
            f"{snap.label}\nt = {snap.t * MS:.2f} ms, "
            f"{e_kin / 1e6:.1f} MeV/u",
            fontsize=9,
        )
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.tick_params(labelsize=8)
        ax.text(
            0.02, 0.02,
            f"V = {bucket.voltage:.0f} V\n"
            f"$\\phi_{{rf}}$ = {np.degrees(bucket.phi_rf):.0f}$\\degree$\n"
            f"in bucket {100 * snap.metrics['in_bucket']:.0f}%",
            transform=ax.transAxes, fontsize=7, color="w", va="bottom",
            bbox=dict(boxstyle="round", fc="k", alpha=0.55, lw=0),
        )
    for ax in axes[n:]:
        ax.axis("off")
    for ax in axes[::ncol]:
        ax.set_ylabel(r"$\Delta E$ [keV]")
    for ax in axes[(nrow - 1) * ncol:]:
        ax.set_xlabel(r"RF phase $\omega_{rf}\,dt + \pi$ [deg]")
    fig.suptitle(_title(results) + " -- red: separatrix")
    return fig


def plot_mountain_range(results):
    """Line density vs RF phase over the cycle (three time windows)."""
    p = results.programs
    tl = p.timeline
    edges = results.profile_edges
    psi = np.degrees(0.5 * (edges[1:] + edges[:-1]))
    prof, tp = results.profiles, results.profile_t
    tab = p.tabulate(200)
    windows = [
        ("Flat bottom: capture", 0.0, tl["ramp_start"]),
        ("Ramp", tl["ramp_start"], tl["ramp_end"]),
        ("Flat top: gymnastics", tl["ramp_end"], tl["end"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), layout="constrained")
    vmax = np.quantile(prof, 0.999)
    for ax, (name, t0, t1) in zip(axes, windows):
        sel = (tp >= t0) & (tp <= t1)
        if sel.sum() < 2:
            ax.set_title(name)
            continue
        mesh = ax.pcolormesh(psi, tp[sel] * MS, prof[sel], shading="nearest",
                             cmap="magma", vmin=0, vmax=vmax)
        in_win = (tab["t"] >= t0) & (tab["t"] <= t1)
        ax.plot(np.degrees(tab["phi_s"][in_win]), tab["t"][in_win] * MS,
                "c--", lw=1, label=r"$\phi_s$")
        if name.startswith("Flat top"):
            # (x, ha, va) chosen so that closely spaced labels do not overlap
            marks = {
                "phase_jump": ("phase jump", -178, "left", "top"),
                "phase_jump_back": ("jump back", 178, "right", "bottom"),
                "debunch_start": ("debunching start", -178, "left",
                                  "bottom"),
                "debunch_end": ("debunching end", 178, "right", "bottom"),
            }
            for key, (text, x, ha, va) in marks.items():
                ax.axhline(tl[key] * MS, color="w", lw=0.6, ls=":")
                ax.text(x, tl[key] * MS, text, color="w", fontsize=7,
                        ha=ha, va=va)
        ax.set_ylim(tp[sel][0] * MS, tp[sel][-1] * MS)
        ax.set_title(name)
        ax.set_xlabel(r"RF phase $\omega_{rf}\,dt + \pi$ [deg]")
        ax.set_xlim(-180, 180)
        ax.set_xticks([-180, -90, 0, 90, 180])
    axes[0].set_ylabel("Time in cycle [ms]")
    fig.colorbar(mesh, ax=axes, label="particles per bin / injected",
                 shrink=0.8)
    fig.suptitle("Line density (mountain range) -- " + _title(results))
    return fig


def plot_flat_top(results):
    """Flat-top RF gymnastics: zoom on jump/rotation and full flat top."""
    p, st = results.programs, results.stats
    tl = p.timeline
    t_ref = tl["ramp_end"]
    n0 = results.n_initial
    tab = p.tabulate(2000)
    rotation = tl["debunch_start"] - tl["phase_jump"]
    windows = [
        (tl["phase_jump"] - 0.3 * rotation, tl["debunch_start"] + rotation),
        (tl["ramp_end"], tl["end"]),
    ]
    coasting_length = 4 / np.sqrt(3) * np.pi / p.bucket(t_ref).omega_rf

    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex="col",
                             layout="constrained")
    for col, (t0, t1) in enumerate(windows):
        sel = (st["t"] >= t0) & (st["t"] <= t1)
        ts = (st["t"][sel] - t_ref) * MS
        tsel = (tab["t"] >= t0) & (tab["t"] <= t1)
        tt = (tab["t"][tsel] - t_ref) * MS
        ax = axes[:, col]

        ax[0].plot(tt, tab["voltage"][tsel], "C3", label=r"$V_{RF}$ program")
        ax[0].plot(ts, st["voltage_applied"][sel], "k.", ms=2,
                   label="applied in BLonD")
        ax[0].set_ylabel("RF voltage [V]")
        twin = ax[0].twinx()
        twin.plot(tt, np.degrees(tab["phi_rf"][tsel]), "C0", lw=1)
        twin.plot(ts, np.degrees(st["phi_rf_applied"][sel]), "C0.", ms=2)
        twin.set_ylabel(r"$\phi_{rf}$ [deg]", color="C0")
        twin.set_ylim(-20, 200)
        ax[0].legend(fontsize=8, loc="center right")

        ax[1].plot(ts, st["dp_p_rms"][sel] * 1e3, "C1.", ms=3)
        ax[1].set_ylabel(r"rms $\Delta p/p$ [$10^{-3}$]")

        ax[2].plot(ts, st["bunch_length_4sigma"][sel] * 1e9, "C0.", ms=3)
        ax[2].axhline(coasting_length * 1e9, color="k", ls=":", lw=1,
                      label="uniform (coasting)")
        ax[2].set_ylabel(r"4$\sigma$ bunch length [ns]")
        ax[2].legend(fontsize=8, loc="lower right")

        ax[3].plot(ts, 100 * st["in_bucket"][sel] * st["n_alive"][sel] / n0,
                   "C3.", ms=3)
        ax[3].set_ylabel("Inside separatrix [% of injected]")
        ax[3].set_ylim(-2, 105)
        ax[3].set_xlabel("Time after end of ramp [ms]")

        for a in ax:
            _shade_phases(a, p, x_offset=t_ref)
            a.grid(alpha=0.3)
            a.set_xlim((t0 - t_ref) * MS, (t1 - t_ref) * MS)
    axes[0, 0].set_title("Phase jump and bunch rotation (zoom)")
    axes[0, 1].set_title("Whole flat top")
    fig.suptitle("Flat-top RF gymnastics -- " + _title(results))
    _phase_legend(fig)
    return fig


def plot_beam_current(results):
    """Beam current and number of particles over the cycle."""
    p, st, cfg = results.programs, results.stats, results.config
    sp = p.particle
    ts = st["t"] * MS
    scale = sp.intensity / results.n_initial  # real particles per macro
    n_alive = st["n_alive"] * scale
    n_bucket = st["in_bucket"] * n_alive
    f_rev = _tracked_f_rev(results)
    current_dc = n_alive * sp.charge * e * f_rev

    # Peak current from the line density: particles per phase bin / bin time.
    # A 3-bin (circular) smoothing reduces the upward bias of taking the
    # maximum of noisy histogram bins.
    n_bins = len(results.profile_edges) - 1
    profiles = (
        np.roll(results.profiles, 1, axis=1)
        + results.profiles
        + np.roll(results.profiles, -1, axis=1)
    ) / 3
    f_rev_prof = np.array([
        p.bucket(t).omega_rf / (2 * np.pi * cfg.harmonic)
        for t in results.profile_t
    ])
    current_peak = (
        sp.intensity * sp.charge * e * profiles.max(axis=1)
        * n_bins * cfg.harmonic * f_rev_prof
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             layout="constrained")
    ax = axes[0]
    ax.plot(ts, current_dc * 1e3, "C0", label=r"DC current $N q e f_{rev}$")
    ax.plot(results.profile_t * MS, current_peak * 1e3, "C3.-", ms=3, lw=0.8,
            label="peak current (line-density maximum)")
    ax.set_ylabel("Beam current [mA]")
    ax.set_ylim(bottom=0)
    ax.set_title("Beam current")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    ax.plot(ts, n_alive, "k", label="surviving (momentum aperture)")
    ax.plot(ts, n_bucket, "C3", label="inside separatrix")
    ax.set_ylabel(f"Number of {sp.label} [particles]")
    ax.set_ylim(0, 1.1 * sp.intensity)
    sec = ax.secondary_yaxis(
        "right", functions=(lambda n: n / scale, lambda m: m * scale)
    )
    sec.set_ylabel("Macroparticles")
    ax.set_title(
        f"Number of particles (intensity {sp.intensity:.1e} per spill, "
        f"{results.n_initial:,} macroparticles)"
    )
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlabel("Time in cycle [ms]")

    for a in axes:
        _shade_phases(a, p)
        a.grid(alpha=0.3)
    fig.suptitle(_title(results))
    _phase_legend(fig)
    return fig


def plot_beam_control(results):
    """Beam phase error, radial offset and RF frequency corrections."""
    p, st, cfg = results.programs, results.stats, results.config
    if "phase_error" not in st:  # runs made before the beam control
        return None
    tl = p.timeline
    t_on, t_off = p.loop_window
    t0 = tl["capture_end"] - 2e-3
    t1 = tl["phase_jump"] + 2e-3
    sel = (st["t"] >= t0) & (st["t"] <= t1)
    ts = st["t"][sel] * MS

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                             layout="constrained")
    ax = axes[0]
    ax.plot(ts, np.degrees(st["phase_error"][sel]), "C4", lw=0.8)
    ax.set_ylabel(r"$\phi_{beam} - \phi_s$ [deg]")
    ax.set_title("Beam phase error (bunch centroid relative to the "
                 "synchronous phase; a small static offset is expected "
                 "from the bucket asymmetry)", fontsize=10)

    ax = axes[1]
    ax.plot(ts, st["radial_offset"][sel] * 1e3, "C0", lw=0.8)
    ax.set_ylabel(r"$\Delta R$ [mm]")
    ax.set_title(r"Radial offset $\Delta R = R\,\alpha_c\,\langle\Delta p/p"
                 r"\rangle$", fontsize=10)

    ax = axes[2]
    ax.plot(ts, st["pl_correction"][sel], "C3", lw=0.8, label="phase loop")
    ax.plot(ts, st["rl_correction"][sel], "C2", lw=1.2, label="radial loop")
    ax.plot(ts, st["frequency_error"][sel], "k--", lw=1,
            label="test frequency error")
    ax.plot(ts, st["rf_frequency_offset"][sel], "C1", lw=1,
            label=r"total $\Delta f_{RF}$")
    ax.set_ylabel(r"$\Delta f_{RF}$ [Hz]")
    ax.set_title("RF frequency corrections", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("Time in cycle [ms]")

    events = []
    if cfg.rf_phase_error:
        events.append((cfg.rf_phase_error_time,
                       f"RF phase step {cfg.rf_phase_error:g}$\\degree$"))
    if cfg.frequency_error:
        events.append((cfg.frequency_error_time,
                       f"freq. error {cfg.frequency_error:g}"))
    for a in axes:
        _shade_phases(a, p)
        if cfg.phase_loop or cfg.radial_loop:
            a.axvspan(t_on * MS, t_off * MS, fill=False, hatch="//",
                      ec="C1", alpha=0.25, lw=0)
        for t_ev, _ in events:
            a.axvline(t_ev * MS, color="k", ls=":", lw=1)
        a.grid(alpha=0.3)
        a.set_xlim(t0 * MS, t1 * MS)
    for t_ev, label in events:
        axes[0].text(t_ev * MS, 1.0, " " + label, fontsize=8, va="top",
                     transform=axes[0].get_xaxis_transform())

    pl = (f"phase loop ON (g = {cfg.pl_gain:g}/s, HP {cfg.pl_highpass:g} Hz)"
          if cfg.phase_loop else "phase loop off")
    rl = (f"radial loop ON (P = {cfg.rl_gain_p:g}, I = {cfg.rl_gain_i:g}/s)"
          if cfg.radial_loop else "radial loop off")
    fig.suptitle(f"Beam control: {pl}, {rl} -- " + _title(results),
                 fontsize=10)
    _phase_legend(fig)
    return fig


def plot_momentum_distribution(results):
    """Injected vs final (debunched) momentum distribution."""
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    final, inj = results.final_dp_p, results.injected_dp_p
    lim = 1.1 * max(np.max(np.abs(final)), np.max(np.abs(inj)))
    n_bins = int(np.clip(np.sqrt(len(final)), 40, 200))
    bins = np.linspace(-lim, lim, n_bins + 1) * 1e3
    ax.hist(inj * 1e3, bins, density=True, histtype="step", color="k",
            label=f"injected (rms {np.std(inj) * 1e3:.2f}e-3)")
    ax.hist(final * 1e3, bins, density=True, histtype="stepfilled",
            alpha=0.5, color="C1",
            label=f"after debunching (rms {np.std(final) * 1e3:.2f}e-3)")
    ax.set_xlabel(r"$\Delta p/p$ [$10^{-3}$]")
    ax.set_ylabel("Density")
    ax.set_title("Momentum distribution of the coasting beam -- "
                 + _title(results), fontsize=9)
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_all(results: CycleResults, show: bool = False) -> dict:
    """Create, save (PNG) and optionally show all diagnostic figures."""
    makers = {
        "cycle_programs": plot_cycle_programs,
        "beam_evolution": plot_beam_evolution,
        "phase_space_snapshots": plot_phase_space,
        "mountain_range": plot_mountain_range,
        "flat_top_gymnastics": plot_flat_top,
        "beam_current": plot_beam_current,
        "beam_control": plot_beam_control,
        "momentum_distribution": plot_momentum_distribution,
    }
    figures = {}
    with AllowPlotting():
        for name, maker in makers.items():
            fig = maker(results)
            if fig is None:
                continue
            fig.savefig(results.output_dir / f"{name}.png", dpi=130)
            figures[name] = fig
        if show:
            plt.show()
        else:
            for fig in figures.values():
                plt.close(fig)
    return figures
