# Visualising an ACMI flight in Tacview and recording a video

Tacview replays telemetry; it does not run the flight model. The workflow therefore has
two distinct outputs:

1. [`02_generate_jsbsim_skill_dataset.ipynb`](../notebooks/02_generate_jsbsim_skill_dataset.ipynb)
   writes a Tacview-compatible **ACMI replay** (`.txt.acmi`);
2. Tacview opens that replay, and a screen recorder captures the arranged view as a
   normal video file.

Keep the ACMI file as the authoritative, inspectable result. The video is only a
presentation of one camera angle, playback speed, and set of overlays.

## 1. Generate the ACMI file with the dataset notebook

The JSBSim dataset notebook writes one replay per flight to
`artifacts/datasets/bvr_f16_1v1_jsbsim_skills_v001/acmi/`. Run a small job rather than
the production default of 100,000 flights:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[video]'
export BVR_DATASET_FLIGHTS=1          # PowerShell: $env:BVR_DATASET_FLIGHTS = "1"
export BVR_DATASET_WORKERS=1          # PowerShell: $env:BVR_DATASET_WORKERS = "1"
jupyter lab notebooks/02_generate_jsbsim_skill_dataset.ipynb
```

The environment variables must be set before Jupyter starts so its kernel inherits
them. Run all notebook cells in order. The notebook will simulate one 45-second flight,
validate its outputs, and write this replay:

```text
artifacts/datasets/bvr_f16_1v1_jsbsim_skills_v001/acmi/jsbsim-000000.txt.acmi
```

Use that file throughout the rest of this guide. Wait for the notebook to finish before
opening or copying it so the last samples are flushed. As a quick check, its header
should identify a Tacview text file using ACMI 2.1:

```bash
head -n 2 \
  artifacts/datasets/bvr_f16_1v1_jsbsim_skills_v001/acmi/jsbsim-000000.txt.acmi
```

Expected output:

```text
FileType=text/acmi/tacview
FileVersion=2.1
```

Do not convert the notebook's trajectory CSV directly unless necessary: the notebook
already emits geodetic position and attitude in ACMI 2.1 format, which avoids guessing
coordinate conventions.

## 2. Install Tacview and transfer the replay

1. Download and install Tacview from the official site, then start it once. Tacview is
   normally used on Windows. If simulation runs on Linux, a remote host, WSL, or a
   container, copy only the completed ACMI file to the Windows machine; Tacview need not
   be installed beside the simulator.
2. Preserve the `.acmi` or `.txt.acmi` suffix. Compression for archival is fine, but
   extract the file before troubleshooting it.
3. Keep the notebook-generated `manifest.yaml` and `episodes.parquet` beside the replay.
   They identify the dataset configuration and episode; a video alone cannot recover
   that context.

## 3. Open and verify the flight

1. In Tacview, choose **File > Open**, select the ACMI file, and allow the initial map
   load to complete. Dragging the file into the Tacview window is also convenient.
2. Press play, pause, and drag the bottom time slider through the entire flight. Check
   the start and end rather than assuming that a successful open means the replay is
   complete.
3. Confirm that both expected F-16 aircraft—the observer and target—appear and that their
   altitude, heading, speed, and motion are plausible.
4. If an aircraft is difficult to find, select it in the object list or click its symbol,
   then use Tacview's command for centring/following the selected object. Exact shortcuts
   can differ by Tacview release, so use the labelled menus or the shortcut displayed by
   the installed version.
5. If objects jump, appear underground, or point in the wrong direction, inspect the
   ACMI data before recording. Typical causes are longitude/latitude reversal, metres
   written as feet, radians written as degrees, a wrong altitude datum, or an incomplete
   file.

## 4. Compose a clear Tacview view

Do a rehearsal pass before recording.

1. **Choose the subject.** Select the aircraft whose manoeuvre is being explained.
2. **Choose the camera.** A chase/external view shows attitude well; a free or orbiting
   camera shows geometry between aircraft; a top-down view is best for tactical paths.
   Use one stable view for most of the clip instead of making constant camera changes.
3. **Set useful overlays.** Enable labels, trails, and the telemetry panels needed for
   the story (for example altitude and speed), and hide panels or object categories that
   add clutter. Avoid displaying private file paths or unrelated desktop windows.
4. **Frame the action.** Scrub to the most widely separated part of the manoeuvre and
   choose zoom there. This prevents an aircraft or trail leaving the frame later.
5. **Set trail length and time scale.** Long trails explain overall geometry; short
   trails make close combat easier to read. Use real-time (`1x`) for natural motion or a
   clearly disclosed slower/faster rate. Pause briefly before and after the action.
6. **Set a clean start.** Seek a few seconds before the manoeuvre, pause, place the mouse
   outside the capture region, and note the desired stop time. If commentary will be
   added later, leave visual breathing room at both ends.

For analysis, make multiple clips rather than trying to show everything simultaneously:
one external aircraft view, one top-down geometry view, and—if useful—one pass with
graphs or event panels visible.

## 5. Record with OBS Studio (recommended)

Screen capture is the most version-independent method and records exactly what the
operator sees. OBS Studio is a common choice, but the operating system's screen recorder
can follow the same sequence.

### One-time OBS setup

1. Create a scene such as `Tacview replay`.
2. Add a **Window Capture** source and select the Tacview window. Prefer window capture
   over display capture so notifications and other applications cannot enter the video.
   If window capture is black on a particular GPU/Windows combination, use display
   capture after closing private windows and enabling Do Not Disturb.
3. In **Settings > Video**, set the base and output canvas to the intended delivery size,
   normally 1920×1080, and choose 30 fps for ordinary analysis or 60 fps for rapid camera
   motion.
4. In **Settings > Output > Recording**, choose a hardware encoder when available. Record
   to MKV so an interruption is less likely to corrupt the whole recording; after capture,
   use **File > Remux Recordings** to make an MP4. If recording directly to MP4, understand
   that a crash or power loss can leave it unusable.
5. Capture desktop audio only if Tacview audio is required. Select a microphone only for
   live narration; otherwise disable both to avoid fan noise and notifications.
6. Make a ten-second test. Confirm resolution, text legibility, smooth motion, crop,
   audio, and the folder containing the recording.

### Recording pass

1. In Tacview, pause at the chosen start time and restore the rehearsed camera and
   overlays.
2. Start recording in OBS, wait one or two seconds, then start Tacview playback.
3. Avoid moving the pointer over controls. Make only rehearsed camera changes. For a
   deterministic result, let the replay play continuously rather than repeatedly
   scrubbing the timeline.
4. At the chosen end time, pause Tacview, wait one or two seconds, then stop OBS.
5. Watch the resulting file from beginning to end. Verify that no frames were dropped,
   the full manoeuvre is present, text is readable, and audio (if any) is synchronized.
6. Remux MKV to MP4 in OBS. Remuxing changes the container without re-encoding, so it is
   fast and does not reduce image quality.

If the installed Tacview edition exposes its own video-export command, it can be used
instead; follow the labels and codec options shown by that installed release. The OBS
workflow remains useful because Tacview capabilities and menu placement vary by version
and licence.

## 6. Optional post-processing with FFmpeg

Trim a capture without re-encoding when the requested cut falls on suitable keyframes:

```bash
ffmpeg -ss 00:00:02 -to 00:00:32 -i tacview-capture.mp4 -c copy flight.mp4
```

For frame-accurate trimming and a broadly compatible H.264 output, re-encode:

```bash
ffmpeg -ss 00:00:02 -to 00:00:32 -i tacview-capture.mp4 \
  -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
  -movflags +faststart -c:a aac -b:a 160k flight.mp4
```

If the capture has no audio, add `-an`. `-crf 18` is high quality but relatively large;
values around 20–23 produce smaller files. Do not repeatedly re-encode the same clip.

Inspect the deliverable:

```bash
ffprobe -v error -show_entries format=duration,size \
  -show_entries stream=codec_name,width,height,r_frame_rate \
  -of default=noprint_wrappers=1 flight.mp4
```

## 7. Delivery checklist

- Keep the original ACMI, the notebook-generated `manifest.yaml` and `episodes.parquet`,
  and the raw capture; distribute a derived MP4.
- Name the video with the scenario and run identifier rather than `capture.mp4`.
- State the playback speed if it is not `1x`, and distinguish simulation time from video
  duration.
- Do not interpret a smooth Tacview replay as proof of valid dynamics. Compare key values
  against the trajectory data and check for crashes, discontinuities, and missing frames.
- Confirm that labels, callsigns, paths, and narration contain nothing sensitive before
  publishing.

## Troubleshooting

| Symptom | What to check |
|---|---|
| Tacview reports an unsupported or empty file | Confirm the file is complete and begins with `FileType=text/acmi/tacview`; do not open a temporary file still being written. |
| No aircraft is visible | Select an object and centre/follow it; reset filters; verify that ACMI frames contain longitude, latitude, and altitude. |
| Aircraft is underground or far away | Check coordinate order, altitude datum, and unit conversion in the producer. |
| Motion is jerky | Capture at a stable frame rate, use `1x` or slower playback, reduce overlays, and close GPU-heavy applications. Low-rate telemetry can still look stepped. |
| OBS window capture is black | Update the graphics driver, make OBS and Tacview use the same GPU, or use a tightly cropped display capture. |
| Text is blurry | Capture at the final output resolution, avoid rescaling the source, and use a higher-quality encoder setting. |
| Final MP4 is corrupt | Record to MKV and remux after stopping normally; retain the MKV until the MP4 has been verified. |
