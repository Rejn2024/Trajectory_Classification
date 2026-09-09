# F-16 scripted-manoeuvre video guide

This guide accompanies [`notebooks/f16_scripted_manoeuvre_video.ipynb`](../notebooks/f16_scripted_manoeuvre_video.ipynb). The notebook uses JSBSim directly for deterministic flight dynamics and makes a portable Matplotlib video. BVR Sim also uses JSBSim, but its environment action is a higher-level four-branch discrete command; the two APIs must not be mixed.

## 1. Reproducible installation

Use Python 3.10 or newer and start at this repository's root:

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e '.[video]'
python -m ipykernel install --user --name bvr-f16-video
jupyter lab notebooks/f16_scripted_manoeuvre_video.ipynb
```

Install FFmpeg with the operating-system package manager (`apt install ffmpeg`, `brew install ffmpeg`, or a trusted Windows distribution) and confirm `ffmpeg -version`. Without it, the notebook deliberately falls back to a larger GIF.

BVR Sim is an external dependency. Clone it separately and check out the project-pinned revision before following its own installation instructions:

```bash
git clone https://github.com/lizi-Margin/bvr_sim.git ../bvr_sim
git -C ../bvr_sim checkout db4f95657c081d120442022a339bed62d340c93b
```

Do not silently use the tip of its default branch: configuration names and controller behavior can change. The notebook itself needs the `jsbsim` Python package, not an import from BVR Sim.

## 2. Selecting aircraft data

The simplest path is to leave `JSBSIM_ROOT` unset. `jsbsim.FGFDMExec(None)` then searches the data bundled in the installed wheel. If BVR Sim supplies a modified F-16 definition, locate the directory whose immediate children include `aircraft/`, `engine/`, and `systems/`, then launch Jupyter with:

```bash
export JSBSIM_ROOT=/absolute/path/to/jsbsim-data
export JSBSIM_MODEL=f16
jupyter lab notebooks/f16_scripted_manoeuvre_video.ipynb
```

Aircraft names are case-sensitive on some systems. Inspect `$JSBSIM_ROOT/aircraft/` and set `JSBSIM_MODEL` to the directory name. Do not combine an aircraft directory from one revision with engine or systems directories from another.

## 3. What the notebook does

1. Loads the F-16 model and sets altitude, heading, velocity, and geodetic location through JSBSim initial-condition properties.
2. Applies a time-based schedule of normalized throttle and control-surface commands at 60 Hz.
3. Records position, attitude, and true airspeed, with the exported CSV using SI units.
4. Plots flight-state checks before rendering.
5. Converts latitude/longitude to a short-range local tangent-plane approximation and renders a 3-D diagnostic MP4 or GIF.

Edit `INITIAL` and `MANOEUVRE` in the configuration cell. Each schedule entry is:

```text
(end time in seconds, label, aileron, elevator, rudder, throttle)
```

Control values are normalized and should remain in `[-1, 1]`; throttle should remain in `[0, 1]`. Start with small changes. An open-loop control schedule is sensitive to aircraft definition and trim, so always inspect the sanity-check plots.

## 4. Connecting the manoeuvre to BVR Sim

BVR Sim's native environment action is `MultiDiscrete([15, 15, 9, 2])`, corresponding to heading, altitude, speed, and fire branches. Those are high-level tactical commands. The notebook's normalized aileron/elevator/rudder/throttle values are low-level JSBSim FCS inputs and **must not** be passed to `env.step()`.

For an environment rollout:

1. Build the pinned BVR Sim F-16-vs-F-16 configuration with JSBSim FDM, entity observations, weapons disabled, and a fixed seed.
2. Express the phases as a BVR scripted policy returning four integer action branches. This repository's `ScheduledPolicy` and `NativeAction` show the intended scheduling and validation pattern.
3. Reset with the seed, step until the configured horizon, and retain observations plus the exact action tuple at every step.
4. Enable the BVR Sim ACMI/Tacview logger using the option documented by the pinned checkout. Open the resulting `.acmi` in Tacview for a combat-replay visualization, or use Tacview's recording workflow to create a presentation video.
5. Record the simulator commit, JSBSim version, backend, aircraft/controller versions, configuration hash, time step, and seed with the artifact.

The exact BVR environment factory and ACMI configuration key are intentionally not guessed here: import paths and configuration schemas are external and revision-specific. Consult the checked-out revision's README/examples and search it locally with:

```bash
rg -n "acmi|tacview|scripted|unit_spec|fdm_type|obs_type" ../bvr_sim
```

## 5. Validation checklist

Before sharing or using the result as training data:

- Confirm the notebook prints the expected JSBSim version and successfully loads `f16`.
- Check altitude and speed for discontinuities, non-finite values, or ground impact.
- Check roll/pitch traces and watch the full video; an open-loop schedule is not guaranteed to suit a modified controller.
- Confirm output duration, playback speed, frame rate, and output file size.
- Replay the same seed twice and compare the CSV files.
- When using BVR Sim, verify the recorded four-branch actions match the script and archive the `.acmi` plus resolved configuration.
- Label the Matplotlib result as a diagnostic visualization, not a cockpit-quality or native BVR render.

## 6. Common failures

| Symptom | Resolution |
|---|---|
| `ModuleNotFoundError: jsbsim` | Activate the notebook kernel's environment and install `.[video]`. |
| `Could not load 'f16'` | Unset a stale `JSBSIM_ROOT`, or point it to the parent of `aircraft/`, `engine/`, and `systems/`; verify model-name case. |
| Required property is absent | The selected aircraft/controller differs from the tested JSBSim property catalog; inspect `fdm.get_property_catalog()` and map the equivalent property explicitly. |
| MP4 writer is unavailable | Install FFmpeg and restart Jupyter, or accept the automatic GIF fallback. |
| Aircraft diverges or tumbles | Reduce inputs, shorten the phase, verify trim/initial speed, and use the controller expected by that aircraft definition. |
| BVR `env.step` rejects controls | Translate phases to its four integer action branches; do not pass low-level FCS values. |

## Outputs

The notebook writes under `artifacts/f16_video/`:

- `f16_scripted_manoeuvre.csv` — time-series state and phase labels;
- `f16_scripted_manoeuvre.mp4` — preferred H.264-compatible container when FFmpeg is available; or
- `f16_scripted_manoeuvre.gif` — fallback animation.

The repository ignores `artifacts/`, so generated binaries are not accidentally committed.
