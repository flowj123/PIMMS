# MedAustron cycle in BLonD 3

`medaustron` is a BLonD 3 simulation of the MedAustron synchrotron cycle, from flat bottom to flat top. Its parameters come from the PIMMS study.

The simulated cycle runs as follows:

```
coasting beam injection -> adiabatic capture -> flat-bottom hold -> ramp
-> flat-top hold -> RF phase jump to the unstable fixed point
-> bunch rotation -> adiabatic debunching -> RF off (coasting beam)
```

## Running it

Run all commands from this workspace (`.../MedAustron/BLonD`) in the `blond3` conda environment (Python 3.12). BLonD 3 is installed there in editable mode from `./blond_repo`, which is a clone of the `blonder` branch.

```bash
conda activate blond3
python -m medaustron.run_cycle                               # protons, 250 MeV (default)
python -m medaustron.run_cycle --species carbon --energy 400 # MeV/u
python -m medaustron.run_cycle --species proton --energy 60 --show
python -m medaustron.run_cycle --energy 800 --v-min 400      # 800 MeV reference (near transition)
python -m medaustron.run_cycle --quick                       # 5k macroparticles
python -m medaustron.run_cycle --no-periodicity              # do not wrap dt (BLonD 3 default)
python -m medaustron.run_cycle --replot results/proton_250MeVu  # re-plot, no tracking
python -m medaustron.run_cycle --help                        # every option
```

You can also run it from Python or Spyder. Set the working directory to this folder first.

```python
from medaustron.machine import CycleConfig
from medaustron.run_cycle import run

res = run(CycleConfig(species="carbon", extraction_energy=250e6,  # eV/u
                      n_macroparticles=100_000, ramp_time=0.4))
res.stats["dp_p_rms"], res.table["voltage"], res.snapshots[...]
```

On this laptop a run with 50k macroparticles takes 5–10 minutes. These times were measured with two runs in parallel:

| Run | Time |
|---|---|
| p 60 MeV | 273 s |
| C 120 MeV/u | 308 s |
| p 250 MeV | 473 s |
| C 400 MeV/u | 596 s |
| p 800 MeV (`--v-min 400`) | 677 s |

Each run uses one core (the `cpp_single_core` backend). The CPU has only 2 physical cores, so run at most two at a time. Run on mains power: on battery with Low Power Mode, macOS capped the CPU at 22% (`pmset -g therm` shows the limit), and runs were about 10× slower.

## Output

Each run writes its output to `results/<species>_<energy>MeVu/`.

| File | Contents |
|---|---|
| `cycle_programs.png` | B field (and Bρ), energy, revolution frequency, RF voltage, synchronous phase and bucket area. The programmed curves are overlaid with the values BLonD actually tracked. |
| `beam_evolution.png` | Bunch length, Δp/p, emittance vs bucket area, capture and transmission, bunch centroid vs φs, synchrotron frequency and adiabaticity. |
| `phase_space_snapshots.png` | Phase space with the separatrix at each key moment of the cycle. |
| `mountain_range.png` | Line density against RF phase over capture, ramp and flat top. |
| `flat_top_gymnastics.png` | Zoom on the phase jump and rotation, plus the whole flat top. |
| `beam_control.png` | Beam phase error, radial offset and RF frequency corrections from the phase loop, the radial loop and the test error. |
| `beam_current.png` | Two panels. Top: the DC beam current N·q·e·f_rev and the peak (bunched) current from the line density. Bottom: the number of particles (surviving and inside the separatrix), with the macroparticle count on the right axis. |
| `momentum_distribution.png` | Injected and final (debunched) Δp/p. |
| `summary.txt` | Key numbers, including a capture benchmark against PIMMS Table 3.8. |
| `results.npz`, `config.json`, `run_info.json` | All recorded data, enough to re-plot with `--replot`. |

## Where the parameters come from

| Parameter | Value | Source |
|---|---|---|
| Circumference | 77.648 m | MedAustron (set by the user). PIMMS used 75.24 m. |
| Injection energy | 20 MeV (p), 7 MeV/u (C6+) | PIMMS Table 3.8: the initial beam length equals T_rev. |
| Harmonic number | 1 | PIMMS (one bunch per T_rev). |
| γ_tr | 1.99 | Derived from Table 3.8 (capture voltage together with K_AREA). |
| Injected Δp/p, emittance | ±1.2e-3; 2·ΔE·T_rev | Table 3.8. |
| Capture | 350 V / 26.8 ms (p); 291 V / 40.4 ms (C) | Table 3.8. |
| Extraction energy range | 60–250 MeV (p); 120–400 MeV/u (C) | Table 1.1. |
| Intensity | 1e10 (p), 4e8 (C) | Table 1.1. It has no effect, because collective effects are off. |
| Max RF voltage | 5 kV | Set by the user. |
| Ramp | 0.5 s with 50 ms parabolic rounding | Assumed. Table 1.1 gives about 1 s for ramp up plus ramp down. |
| Bending radius | 4.231 m | Assumed (PIMMS: 1.5 T at 6.35 Tm). It only sets the B axis of the plots. |

Bucket areas and emittances both scale with T_rev, so the PIMMS capture voltages and K_AREA carry over to the MedAustron circumference unchanged. Table 1.2 (transverse emittances) is not used, because BLonD only models longitudinal motion.

## Programs and defaults

These programs are built in `machine.build_cycle`. Every automatic value can be overridden in `CycleConfig` or on the command line.

- **Capture.** V(t) = V0 / (1 − (1 − √(V0/V1)) t/T)², which keeps the adiabaticity constant. V0 is 1% of V1.
- **Ramp.** Bρ follows a parabolic–linear–parabolic shape. The RF voltage keeps the moving bucket area equal to the flat-bottom bucket area (K = 1.5 for p, 1.6 for C), capped at `max_rf_voltage`. BLonD keeps the RF frequency at h·f_rev automatically.
- **Voltage floor (`min_rf_voltage`, `--v-min`, off by default).** Once the ramp voltage has risen above the floor it never drops below it again. Close to transition (γ_tr ≈ 1.99) the constant-area program asks for very little voltage. For 800 MeV protons that would be 10.6 V at flat top, giving f_s ≈ 24 Hz and a strongly non-adiabatic end of the ramp. The 800 MeV reference therefore uses `--v-min 400`, which keeps the adiabaticity at the 250 MeV level.
- **RF phase.** Below transition BLonD's stable phase is `phi_rf = π`. For the phase jump the program sets `phi_rf = 0` for `unstable_phase_time`.
- **Unstable-phase time (automatic).** The time the bunch's rms core takes to flow along the separatrix to ±90°, where the old and new separatrices cross.
- **Rotation time (automatic).** 3/8 of the synchrotron period at the bunch edge. This uses the pendulum f_s(φ̂), which brings the stretched bunch upright, so Δp/p is at its maximum.
- **Debunching (automatic).** The inverse of the capture law down to 1% of the voltage, with adiabaticity 0.1. The RF is then switched off.
- **Momentum aperture.** Particles beyond |Δp/p| > 1% are removed. This is checked every 100 turns. There is no cut in time, so particles are never lost just for leaving the bucket.
- **Periodicity (on by default; `--no-periodicity` turns it off).** Particles that leave the bucket on one side re-enter it from the other. See the next section.

## Periodicity (`periodicity.py`)

BLonD 3 has no equivalent of BLonD 2's `RingAndRFTracker(periodicity=True)`. That option survives only in `blond/legacy`. The BLonD 3 drift and RF kick never wrap the arrival time dt, so a particle that slips away from the bunch simply accumulates dt.

The single-harmonic kick `sin(ω_rf·dt + φ_rf)` is periodic in dt, so such a particle already feels the RF of the neighbouring bucket. Leaving dt unwrapped still has two drawbacks:

- **Spurious momentum offset on a ramp.** The copies of the bucket sit at dt_s + m·T_rf. For h = 1 they are all the same bucket, one turn apart, but they move apart as T_rf shrinks during the ramp. A particle in copy m must carry a spurious offset of about m·|dT_rf/dt|/|η| to keep up. That is at most 0.8e-5 (p, 250 MeV) to 2.4e-5 (C, 400 MeV/u) per copy, against a bucket half-height of about 3.5e-3.
- **Binned profiles.** A binned profile only sees particles inside its time window. This matters as soon as collective effects are added.

The `Periodicity` element therefore shifts dt by whole RF periods, T_rf = 2π/ω_rf, every 10 turns (`periodicity_every`):

- **Window.** Every particle is moved into one RF period around the bunch. This is the window the diagnostics already use: RF phase in [−π, π) in the beam frame, with the cut in the gap between buckets, opposite the bunch.
- **Effect on the kick.** None: shifting dt by one RF period leaves the kick unchanged, to 1e-13.
- **Effect for h = 1.** T_rf is the revolution period, plus any beam-control frequency offset. A particle leaving one side of the bucket therefore comes round the ring and re-enters from the other side, as with BLonD 2's periodicity.
- **Wrap count.** The number of wraps is reported in `summary.txt`.

Comparison for protons to 250 MeV, with 5,000 macroparticles:

| Cycle phase | No periodicity: furthest particle from the bunch | With periodicity: wraps |
|---|---|---|
| Capture of the coasting beam | 18 RF periods | 35,158 |
| Flat bottom, ramp, flat-top hold | 18 (no further slipping) | 0 (the whole beam stays in the bucket) |
| Phase jump and rotation | 19 | 802 (particles flowing along the separatrix) |
| Debunching and coasting | 54 | 47,670 |

- **Why particles stray during capture.** The capture voltage starts at only 1% of its final value, so edge particles of the coasting beam drift freely for thousands of turns before the bucket catches them. Without periodicity they end up captured into bucket copies up to 18 periods away.
- **Effect on the results.** Small. The final Δp/p agreed within about 1% (rms 3.24e-4 with periodicity, 3.22e-4 without), as did the 99% range. The emittance and survival were the same. The diagnostics already wrapped the RF phase, so the plots look the same either way.
- **Existing results.** All eight folders in `results/` were rerun with periodicity on (2026-10-14). Each run wraps 0.8–0.9 million macroparticles, about 17 wraps per particle, almost all of them during capture and after debunching. Compared with the earlier runs without periodicity, survival is unchanged. The final Δp/p rms agrees within 0.7% and the end-of-ramp emittance within 1.3%. The loop demonstration gives the same results as before (with loops off: +4.22 mm radial offset, 30.8° ramp phase peak-to-peak).

## Phase and radial loops (`beam_control.py`)

### What BLonD 3 already provides

BLonD 3 ships a beam-feedback framework (`BeamFeedbackBase`) and beam controls for the PSB, PS, SPS and LHC. They are in `blond.physics.feedbacks.accelerators`. `PSBBeamControl` contains a phase loop and a radial loop.

On the `blonder` branch they are only tested with a constant (non-accelerating) cycle, and two issues rule them out for this ramp:

- **Missing synchronous phase.** `BeamFeedbackBase.update_phase_error` uses `dphi = phi_beam` and does not subtract φs, although BLonD 2 did. During acceleration the phase loop would therefore pull the bunch away from φs, which reaches up to 49° here.
- **Fixed update interval.** `PSBBeamControl` pre-computes the turns on which it acts from the revolution period at injection. That period changes by a factor of 2–4 along this ramp.

`MedAustronBeamControl` is therefore its own `BeamFeedbackBase` subclass. BLonD applies its `delta_omega_rf` to the RF station, which accumulates the resulting RF phase slip.

### The two loops

Both loops update every `loop_period` (10 µs).

- **Phase loop.** The error is the beam phase minus φs(t). The beam phase is the phase of the beam's RF component, the same quantity BLonD's `beam_phase` computes. A first-order high-pass (20 Hz) removes static offsets, such as the centroid shift of a large bunch in an accelerating bucket, so the loop only damps dipole oscillations. The proportional gain `pl_gain` is in 1/s, and the damping time is about 2/`pl_gain`.
- **Radial loop.** ΔR/R = α_c·⟨Δp/p⟩, measured from the beam's mean energy offset. A PI controller (`rl_gain_p` in 1, `rl_gain_i` in 1/s) corrects the RF frequency. The integral time constant is |η|/(α_c·`rl_gain_i`).

### Operation

- **Switching.** Both loops are off by default; switch them on with `phase_loop=True` / `radial_loop=True`, or `--phase-loop` / `--radial-loop`.
- **Window.** They act between `loop_window` = (t_on, t_off). The default is from the end of capture to the phase jump. Before that there is no bunch to act on, and afterwards they must not fight the flat-top gymnastics.
- **When the window closes.** The phase-loop correction returns to zero and the radial-loop correction is held. Switch-on is bumpless.
- **Default gains.** These were chosen from a gain scan:

  | Gain | Default |
  |---|---|
  | `pl_gain` | 3000 /s |
  | `rl_gain_p` | 0.5 |
  | `rl_gain_i` | 300 /s |

  `pl_gain` up to 10,000 /s and `rl_gain_i` from 100–300 /s were tested stable.
- **Changing gains during the cycle.** The gains can be scheduled like any BLonD parameter, e.g. `beam_control.schedule("pl_gain", (times, values))`. Use `build_simulation()` from `run_cycle.py` to get at the `beam_control` object.

### Test perturbations

Two test perturbations exercise the loops. Both are off by default.

| Option | Effect |
|---|---|
| `--rf-phase-error DEG T` | Step in the RF phase at time T, which excites a dipole oscillation. |
| `--frequency-error DF_F T` | RF frequency error df/f from time T, with a 20 ms smooth rise. The beam moves radially by ΔR/R = α_c/\|η\|·df/f. |

```bash
python -m medaustron.run_cycle --phase-loop --radial-loop \
    --rf-phase-error 20 0.15 --frequency-error 5e-4 0.30
```

Runs with loops or test perturbations are written to their own folders, for example `results/proton_250MeVu_PL_RL_test/`. `beam_control.png` shows the beam phase error, the radial offset, and the RF frequency corrections of each loop.

## Files

- `bucket.py`: single-harmonic bucket geometry below transition (separatrix, area, Hamiltonian levels).
- `machine.py`: species presets, `CycleConfig`, and the B/energy/voltage/phase programs.
- `beam_control.py`: the phase and radial loops (a BLonD `BeamFeedbackBase`).
- `periodicity.py`: wraps dt into one RF period, so particles leaving the bucket re-enter from the other side.
- `run_cycle.py`: builds the BLonD objects, records the diagnostics, provides the command-line interface, and handles save/load.
- `diagnostics.py`: the plots.

Collective effects (space charge, impedance) are not modelled yet. To add them, put a `StaticProfile` and a `WakeField` in `build_simulation`.
