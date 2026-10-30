# PIMMS

BLonD 3 simulations of the MedAustron synchrotron, with parameters taken from the PIMMS study.

This file explains how to **install** the project. For what the simulation does, how to run it, and what it writes out, see [`medaustron/README.md`](medaustron/README.md).

## Repository layout

```
PIMMS/
├── medaustron/        # the simulation package (see medaustron/README.md)
├── blond_repo/        # BLonD 3 source, pinned as a git submodule-style link (empty after a plain clone)
├── requirements.txt   # pinned Python dependencies
└── results/           # created on first run; output of each simulation (not tracked by git)
```

## Prerequisites

- **Git**
- **Conda** (Miniconda, Anaconda or Miniforge). The project was developed in a conda environment named `blond3`.
- **Python 3.12**
- **A C/C++ compiler** (`gcc`/`g++` on Linux, Xcode command line tools on macOS, Microsoft C++ Build Tools on Windows). BLonD's default `cpp_single_core` backend, which this project uses, needs it. If a compiler is not available, see [Choosing a backend](#choosing-a-backend).
- **Network access to `gitlab.cern.ch`**, where BLonD is hosted.

## Installation (Linux and macOS)

Windows users: follow [Installation on Windows](#installation-on-windows) below, which has the same steps with Windows commands.

### 1. Clone the repository

```bash
git clone https://github.com/flowj123/PIMMS.git
cd PIMMS
```

`blond_repo/` is recorded in the repository as a link to a specific BLonD commit, but there is no `.gitmodules` file, so `git clone --recursive` and `git submodule update` will **not** fill it. It will be an empty folder. Step 3 fills it by hand.

### 2. Create the conda environment

```bash
conda create -n blond3 python=3.12
conda activate blond3
```

### 3. Get BLonD 3 at the pinned commit

```bash
rmdir blond_repo            # remove the empty placeholder folder
git clone https://gitlab.cern.ch/blond/BLonD.git blond_repo
git -C blond_repo checkout 5822f9678365d87eae1bd93f22a32ebbe3f6aa1d
```

This is the commit that `requirements.txt` pins. The simulation was developed against BLonD 3 from the `blonder` branch, so if the checkout fails, fetch that branch first (`git -C blond_repo fetch origin blonder`).

### 4. Install the Python dependencies

`requirements.txt` was exported from the original development machine, so two of its lines should not be passed to `pip` as they are:

- The first line (`-e git+https://gitlab.cern.ch/...`) installs BLonD from GitLab. Step 5 installs it from `blond_repo/` instead.
- The `packaging @ file:///home/conda/...` line points to a path that only exists on the original machine.

Filter both out and install the rest:

```bash
grep -v -e '^-e ' -e '^packaging @' requirements.txt > /tmp/requirements-clean.txt
pip install -r /tmp/requirements-clean.txt
```

### 5. Install BLonD in editable mode

```bash
pip install -e ./blond_repo
```

Building BLonD's C++ library happens here or on first import, depending on the BLonD version. Watch the output for compiler errors.

### 6. Check the installation

```bash
python -c "import blond; print('BLonD imported from', blond.__file__)"
python -m medaustron.run_cycle --help
python -m medaustron.run_cycle --quick
```

Run these from the repository root (the folder that contains `medaustron/`). The `--quick` run uses 5,000 macroparticles and writes its output to `results/proton_250MeVu/`. If it finishes and produces `summary.txt` and the PNG files, the installation works.

## Installation on Windows

The steps are the same as above. Only the commands differ, because Windows has no `grep` and no `/tmp`. The Windows instructions have not been tested. There are two routes.

### Option A: WSL2 (recommended)

[WSL2](https://learn.microsoft.com/windows/wsl/install) runs a Linux system inside Windows. Install it (`wsl --install` in an administrator PowerShell, then restart), open the Ubuntu terminal, install Miniconda and a compiler (`sudo apt install build-essential git`), then follow the [Linux and macOS instructions](#installation-linux-and-macos) unchanged. This is the route closest to the one the project was developed on, and it avoids building BLonD's C++ code with a Windows compiler.

Keep the repository inside the Linux file system (for example `~/PIMMS`), not under `/mnt/c/`. Builds and runs are much slower on the Windows drive.

### Option B: native Windows

Use the **Anaconda PowerShell Prompt** (or Miniforge Prompt), not the old Command Prompt, because the filtering command in step 4 is PowerShell.

**Extra prerequisites:** install [Git for Windows](https://git-scm.com/download/win) and, if you want the compiled backend, the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the "Desktop development with C++" workload. Without the Build Tools, skip them and use `--backend numba` (see [Choosing a backend](#choosing-a-backend)).

**1. Clone the repository**

```powershell
git clone https://github.com/flowj123/PIMMS.git
cd PIMMS
```

**2. Create the conda environment**

```powershell
conda create -n blond3 python=3.12
conda activate blond3
```

**3. Get BLonD 3 at the pinned commit**

```powershell
rmdir blond_repo
git clone https://gitlab.cern.ch/blond/BLonD.git blond_repo
git -C blond_repo checkout 5822f9678365d87eae1bd93f22a32ebbe3f6aa1d
```

**4. Install the Python dependencies**

This does the same as the `grep` command above: it drops the `-e git+...` and `packaging @ file:///...` lines.

```powershell
Get-Content requirements.txt | Where-Object { $_ -notmatch '^(-e |packaging @)' } | Set-Content "$env:TEMP\requirements-clean.txt"
pip install -r "$env:TEMP\requirements-clean.txt"
```

**5. Install BLonD in editable mode**

```powershell
pip install -e .\blond_repo
```

If the build cannot find the compiler, try again from the "x64 Native Tools Command Prompt for VS" (installed with the Build Tools), after running `conda activate blond3` there.

**6. Check the installation**

```powershell
python -c "import blond; print('BLonD imported from', blond.__file__)"
python -m medaustron.run_cycle --help
python -m medaustron.run_cycle --quick --backend numba
```

Drop `--backend numba` from the last command if you built the C++ backend. Run all of these from the repository root.

I could not check whether BLonD's C++ library builds with the Microsoft compiler. If it does not, use `--backend numba` on native Windows or switch to WSL2.

## Choosing a backend

The simulation uses BLonD's `cpp_single_core` backend by default. You can switch it on the command line:

```bash
python -m medaustron.run_cycle --backend numba
```

The options are `cpp_single_core`, `numba`, `cpp` and `auto`. `numba` is listed in `requirements.txt` and needs no separate C++ build, so it is the natural fallback if the compiled backend will not build. Timings in `medaustron/README.md` were measured with `cpp_single_core`.

## Using VS Code or Spyder

- **VS Code:** the repository's `.vscode/settings.json` selects conda as the environment manager. Open the folder and choose the `blond3` interpreter.
- **Spyder or any other IDE:** set the working directory to the repository root before running `from medaustron.run_cycle import run`, so that `medaustron` can be imported and `results/` is written in the right place.

## Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| `blond_repo/` is empty | Expected after a plain `git clone`. Follow step 3. |
| `ModuleNotFoundError: No module named 'blond'` | The `blond3` environment is not active, or step 5 was skipped. |
| `ModuleNotFoundError: No module named 'medaustron'` | Run from the repository root, or use `python -m medaustron.run_cycle` instead of calling the file directly. |
| `pip` fails on `packaging @ file:///...` | Use the filtered requirements file from step 4. |
| `git clone` from `gitlab.cern.ch` fails or times out | Check your network or proxy. Some institutional networks block it. |
| Compiler errors while installing BLonD | Install a C/C++ compiler, or use `--backend numba`. |
| Dependency conflicts with `numpy`, `scipy` or `numba` | Use a fresh environment with Python 3.12. The pinned versions need a recent Python. |
| `grep` or `/tmp` not found on Windows | Use the PowerShell commands in [Installation on Windows](#installation-on-windows), or WSL2. |
| `Microsoft Visual C++ 14.0 or greater is required` (Windows) | Install the Microsoft C++ Build Tools, or use `--backend numba`. |
| `conda` not recognised in PowerShell | Open the "Anaconda PowerShell Prompt" from the Start menu, or run `conda init powershell` once and reopen the terminal. |
| Runs are very slow on a laptop | Run on mains power and use `--quick` for tests. See the timing notes in `medaustron/README.md`. |

## Next steps

Once the installation check passes, continue with [`medaustron/README.md`](medaustron/README.md) for the run options, the output files, and the physics behind the cycle.
