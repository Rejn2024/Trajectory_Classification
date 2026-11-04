# Tactical-adapted pilot demonstrations

Three native Tacview 3D recordings show trained F-16 pilots engaging a scripted
tactical F-16. They replay encounters from the completed tactical population's
validation-selected checkpoints, with the policy weights frozen during the flights.
They are separate from the 3,000-episode forecasting corpus and from training-epoch
rollouts.

The learned aircraft is **A01, red**. Its scripted tactical opponent is **B01, blue**.
These colours are Tacview identities; they do not indicate the observer/target
roles in the forecasting dataset. The elevated external camera keeps both living
aircraft in view. Aircraft models are enlarged for visibility at BVR distances;
their recorded positions and attitudes are unchanged.

## Videos and observed behaviour

| Video | Duration | Scenario seed | Observed behaviour | Recorded outcome |
| --- | ---: | ---: | --- | --- |
| [Pilot 005](pilot_005_clean_win_seed_1120142.mp4) | 118.3 s | 1120142 | Mixed pursuit and defence, including extension, lag pursuit, beaming and cranking | Opponent eliminated; learned pilot survived |
| [Pilot 010](pilot_010_clean_win_seed_1120142.mp4) | 113.2 s | 1120142 | Frequent turn-cold and extend commands | Opponent eliminated; learned pilot survived |
| [Pilot 003](pilot_003_contested_seed_1120286.mp4) | 120.0 s | 1120286 | Recommitment, lead pursuit and cranking | Both aircraft survived |

Pilots **005 and 010 use the same scenario definition and identical initial aircraft
states**. Their different manoeuvre choices can therefore be viewed side by side
under matched starting conditions. Pilot 003 uses a different scenario and is a
separate illustration, not a matched comparison with the other two.

These encounters were selected for visual interest. They do not establish fixed
pilot personalities, population performance or statistical superiority. Use the
[presentation results](../../../docs/presentations/README.md) for aggregate evidence.

## Playback on a presentation PC

Each MP4 is silent H.264, 1280 x 720, 30 frames per second, at real-time replay speed.
Files are approximately 8-10 MiB each, 26.8 MiB in total. No Tacview installation,
Python environment, notebooks or model checkpoints are required to play them.

On GitHub, open a video's file page and use **Download raw file**. The videos use
Git LFS; downloading them consumes the repository's LFS transfer allowance. They
are already compressed, so an additional ZIP archive is not published.

For a checkout with Git LFS installed, download only these videos with:

```powershell
git lfs pull --include="shared/demonstrators/tactical-checkpoints-v1/*.mp4" --exclude=""
```

Download [index.html](index.html) into the same folder as all three MP4s and open
it in a browser for the offline comparison player. It includes synchronized start,
pause and seeking controls for pilots 005 and 010. GitHub displays HTML source;
download the file before opening it. The page and videos work without a network
connection after download. The three videos can also be inserted directly into
presentation software or opened individually in a video player.

## Source recordings and provenance

The sources are three of the twenty published [ACMI checkpoint replays](../../replays/tactical-checkpoints-v1/README.md).
Their paths inside that archive are:

```text
replays/pilot_005/pilot_005_clean_win_seed_1120142.txt.acmi
replays/pilot_010/pilot_010_clean_win_seed_1120142.txt.acmi
replays/pilot_003/pilot_003_contested_seed_1120286.txt.acmi
```

[manifest.json](manifest.json) records each MP4's size, SHA-256, source ACMI and
checkpoint hashes, frame count and camera settings. Source paths beginning with
`artifacts/` identify the original local rendering inputs; the equivalent ACMIs
are available in the linked archive. All 10,545 output frames were decoded without
errors, and published file sizes and checksums were verified against the completed
local recordings.

The recordings use Tacview 1.9.5 and Windows Graphics Capture. Sensor graphics were
hidden in temporary display copies; the source ACMIs were preserved. The small
overhead map uses the recorded aircraft tracks. The "Last logged command" caption
shows the latest skill-log entry at or before the displayed time, rather than a
continuous measurement of actuator commands.

The rendering implementation is [render_tacview_replays.py](../../../scripts/render_tacview_replays.py)
with its [Tacview Lua add-on](../../../scripts/tacview_capture/main.lua).
Re-rendering requires Windows, Tacview with Lua add-ons, the extracted source ACMIs
and their summaries, and the Python dependencies listed in the renderer. Playback
of the published MP4s does not require those tools.
