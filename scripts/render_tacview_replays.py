"""Render three existing ACMI recordings with Tacview's native 3D renderer.

Run from the repository: python scripts/render_tacview_replays.py --render
Use --clip 1 --seconds 5 for a short preview. Requires installed Tacview 1.9.5
with Lua add-ons available, plus Pillow, numpy, imageio-ffmpeg and windows-capture.
Outputs are local under artifacts/tacview_videos; the original ACMI is unchanged.
The capture uses only the dedicated Tacview window, never the entire desktop.
"""

import argparse
import atexit
import ctypes
import hashlib
import json
import math
import re
import shutil
import subprocess
import threading
import time
import winreg
from bisect import bisect_right
from ctypes import wintypes
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from windows_capture import WindowsCapture

ROOT = Path(__file__).resolve().parents[1]
TACVIEW = Path(r"C:\Program Files (x86)\Tacview\Tacview64.exe")
WORK = ROOT / "artifacts/tacview_mp4_work"
REPLAYS = ROOT / "artifacts/tactical_checkpoint_replays/replays"
CLIPS = [
    ("pilot_005", "clean_win", 1120142),
    ("pilot_010", "clean_win", 1120142),
    ("pilot_003", "contested", 1120286),
]
TITLES = [
    "Mixed pursuit and defensive manoeuvres",
    "Frequent turns away from the opponent",
    "Repeated recommitment and pursuit",
]
OUTPUT = ROOT / "artifacts/tacview_videos"
WIDTH, HEIGHT = 1280, 720
ACTIVE_WINDOW = None


def replay_paths(index):
    pilot, category, seed = CLIPS[index - 1]
    stem = REPLAYS / pilot / f"{pilot}_{category}_seed_{seed}"
    return Path(str(stem) + ".txt.acmi"), Path(str(stem) + ".summary.json")


def windows_for_pid(pid):
    user = ctypes.windll.user32
    result = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def visit(hwnd, _):
        process_id = wintypes.DWORD()
        user.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        if process_id.value == pid and user.IsWindowVisible(hwnd):
            rect = wintypes.RECT()
            user.GetWindowRect(hwnd, ctypes.byref(rect))
            title = ctypes.create_unicode_buffer(1024)
            user.GetWindowTextW(hwnd, title, len(title))
            result.append(
                {
                    "hwnd": hwnd,
                    "title": title.value,
                    "rect": [rect.left, rect.top, rect.right, rect.bottom],
                }
            )
        return True

    user.EnumWindows(callback_type(visit), 0)
    return result


def prepare(width, height):
    addon = WORK / "addons/ReplayCapture"
    addon.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "scripts/tacview_capture/main.lua", addon / "main.lua")
    paths = []
    for index in range(1, len(CLIPS) + 1):
        source, _ = replay_paths(index)
        # Display-only copies omit sensor cones/lock lines. All positions,
        # attitudes, times, missiles and destruction events remain unchanged.
        display = WORK / source.name
        text = source.read_text(encoding="utf-8")
        text = re.sub(r",(?:Radar\w*|LockedTarget\w*)=[^,\r\n]*", "", text)
        display.write_text(text, encoding="utf-8")
        paths.append(display)
    window_x = 30
    content = (
        f'return {{work = "{WORK.as_posix()}", width = {width}, height = {height}, window_x = {window_x}, window_y = 30, files = {{\n'
        + ",\n".join(f'"{p.as_posix()}"' for p in paths)
        + "\n}}\n"
    )
    (addon / "config.lua").write_text(content, encoding="utf-8")


def request(sequence, clip, seconds, timeout=40):
    content = f"{sequence} {clip} {seconds:.6f}"
    temporary = WORK / "request.tmp"
    temporary.write_text(content, encoding="ascii")
    # Windows denies replacement while Lua briefly holds the request open.
    replace_deadline = time.monotonic() + 5
    while True:
        try:
            temporary.replace(WORK / "request.txt")
            break
        except PermissionError:
            if time.monotonic() >= replace_deadline:
                raise
            time.sleep(0.002)
    deadline = time.monotonic() + timeout
    last_wake = 0.0
    while time.monotonic() < deadline:
        if ACTIVE_WINDOW and time.monotonic() - last_wake > 0.1:
            # Queue an ordinary paint without erasing the background; unlike
            # PrintWindow this does not redraw into a separate capture DC.
            ctypes.windll.user32.InvalidateRect(ACTIVE_WINDOW, None, False)
            last_wake = time.monotonic()
        error = WORK / "error.txt"
        if error.exists():
            raise RuntimeError(error.read_text())
        try:
            fields = (WORK / "ready.txt").read_text().split()
            if fields and int(fields[0]) == sequence:
                return fields
        except (OSError, ValueError):
            pass
        time.sleep(0.005)
    raise TimeoutError(
        f"Tacview did not render frame {sequence}; add-on loaded: {(WORK / 'loaded.txt').exists()}"
    )


def main_window(pid):
    return max(
        windows_for_pid(pid),
        key=lambda x: (x["rect"][2] - x["rect"][0]) * (x["rect"][3] - x["rect"][1]),
    )


class WindowFrames:
    """Capture only Tacview via Windows Graphics Capture; never PrintWindow.

    A rendered 24-bit marker confirms that the compositor delivered exactly the
    requested simulation time. The header covers this marker in the final MP4.
    """

    def __init__(self, hwnd):
        self.condition = threading.Condition()
        self.code, self.buffer, self.closed = None, None, False
        self.capture = WindowsCapture(cursor_capture=False, draw_border=False, window_hwnd=hwnd)

        @self.capture.event
        def on_frame_arrived(frame, capture_control):
            if frame.width != WIDTH or frame.height != HEIGHT:
                return
            pixels = frame.frame_buffer
            code = sum((int(pixels[15, 13 + bit * 6, 0]) > 127) << bit for bit in range(24))
            with self.condition:
                self.code = code
                self.buffer = pixels.copy()
                self.condition.notify_all()

        @self.capture.event
        def on_closed():
            with self.condition:
                self.closed = True
                self.condition.notify_all()

        self.control = self.capture.start_free_threaded()

    def get(self, sequence, timeout=10):
        with self.condition:
            available = self.condition.wait_for(
                lambda: self.code == sequence % 16777216 or self.closed, timeout
            )
            if not available or self.closed:
                raise TimeoutError(
                    f"Window capture did not deliver frame {sequence}; last marker {self.code}"
                )
            return Image.fromarray(self.buffer[:, :, 2::-1].copy())

    def close(self):
        self.control.stop()


def read_tracks(path):
    tracks = {"A01": [], "B01": []}
    seconds = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            seconds = float(line[1:])
        elif line[:3] in tracks and line.startswith(line[:3] + ",T="):
            values = [float(x) for x in line.split(",T=", 1)[1].split(",", 1)[0].split("|")]
            tracks[line[:3]].append([seconds, *values[:6]])
    return {key: np.asarray(values) for key, values in tracks.items()}


class Captions:
    def __init__(self, index, summary, tracks):
        self.index, self.summary, self.tracks = index, summary, tracks
        self.decision_times = [row["time_s"] for row in summary["decisions"]]
        self.fonts = {
            size: ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", size)
            for size in (18, 22, 26, 34)
        }
        all_points = np.concatenate(list(tracks.values()))
        self.lon, self.lat = all_points[:, 1].mean(), all_points[:, 2].mean()
        self.points = {
            key: self.project(values[:, 1], values[:, 2]) for key, values in tracks.items()
        }
        points = np.concatenate(list(self.points.values()))
        self.center = (points.min(axis=0) + points.max(axis=0)) / 2
        self.span = max(np.ptp(points, axis=0).max() * 1.15, 1000)

    def project(self, lon, lat):
        return np.column_stack(
            (
                (lon - self.lon) * math.cos(math.radians(self.lat)) * 111320,
                (lat - self.lat) * 111320,
            )
        )

    def draw(self, frame, seconds, projected=None):
        frame = frame.convert("RGB").resize((1920, 1080))
        if projected:
            projected = [x * 1920 / WIDTH for x in projected]
        draw = ImageDraw.Draw(frame, "RGBA")
        width, height = frame.size
        draw.rectangle((0, 0, width, 111), fill=(4, 15, 29, 255))
        pilot, _category, seed = CLIPS[self.index - 1]
        draw.text(
            (30, 15),
            f"{pilot.replace('_', ' ').title()}  |  {TITLES[self.index - 1]}",
            font=self.fonts[34],
            fill="white",
        )
        draw.text(
            (32, 65),
            f"Selected tactical checkpoint  |  Scripted tactical F-16 opponent  |  Scenario {seed}",
            font=self.fonts[22],
            fill=(185, 206, 222),
        )
        draw.text(
            (width - 210, 22),
            f"{seconds:05.1f} / {self.summary['duration_seconds']:.1f} s",
            font=self.fonts[26],
            fill="white",
        )
        draw.rectangle((0, height - 103, width, height), fill=(4, 15, 29, 225))
        decision = bisect_right(self.decision_times, seconds + 1e-6) - 1
        command = (
            self.summary["decisions"][decision]["skill"].replace("_", " ")
            if decision >= 0
            else "initial guidance"
        )
        own = self.tracks["A01"]
        altitude = np.interp(seconds, own[:, 0], own[:, 3])
        bank = np.interp(seconds, own[:, 0], own[:, 4])
        draw.text(
            (30, height - 88), f"Last logged command: {command}", font=self.fonts[26], fill="white"
        )
        draw.text(
            (32, height - 46),
            f"Altitude {altitude:,.0f} m  |  Bank {bank:+.0f} deg  |  1x replay  |  Aircraft enlarged for visibility",
            font=self.fonts[22],
            fill=(185, 206, 222),
        )
        if projected:
            for offset, label, color in (
                (0, "Trained pilot", (255, 110, 120)),
                (2, "Tactical opponent", (110, 190, 255)),
            ):
                if offset == 2 and seconds > self.tracks["B01"][-1, 0]:
                    continue
                x, y = projected[offset : offset + 2]
                if 0 <= x <= width and 0 <= y <= height:
                    # Tacview's UI coordinates use an origin at bottom left.
                    y = height - y
                    draw.rounded_rectangle(
                        (x - 8, y + 35, x + 205, y + 70), 5, fill=(4, 15, 29, 210)
                    )
                    draw.text((x, y + 39), label, font=self.fonts[22], fill=color)
        if seconds > self.tracks["B01"][-1, 0] and self.summary["eliminated"]:
            draw.text(
                (width - 640, 74),
                "Tactical opponent destroyed",
                font=self.fonts[22],
                fill=(110, 190, 255),
            )
        if seconds >= self.summary["duration_seconds"] - 5:
            outcome = (
                "Outcome: opponent eliminated; trained pilot survived"
                if self.summary["eliminated"]
                else "Outcome: both aircraft survived the recording"
            )
            draw.text((width - 760, height - 82), outcome, font=self.fonts[22], fill="white")
        # Supplementary top-down telemetry view supplies engagement context when
        # the other aircraft is tens of kilometres outside the external camera.
        left, top, size = width - 324, height - 427, 288
        draw.rounded_rectangle(
            (left - 12, top - 38, left + size + 12, top + size + 16), 12, fill=(4, 15, 29, 205)
        )
        draw.text((left, top - 31), "Top-down track  |  N up", font=self.fonts[18], fill="white")

        def screen(points):
            mapped = (points - self.center) / self.span * (size - 20)
            return np.column_stack((left + size / 2 + mapped[:, 0], top + size / 2 - mapped[:, 1]))

        for key, color in (("A01", (255, 95, 108, 255)), ("B01", (96, 181, 255, 255))):
            values = self.tracks[key]
            count = np.searchsorted(values[:, 0], seconds, side="right")
            xy = screen(self.points[key][: max(count, 1)])
            if len(xy) > 1:
                draw.line([tuple(p) for p in xy[::3]] + [tuple(xy[-1])], fill=color, width=2)
            x, y = xy[-1]
            if seconds <= values[-1, 0] + 0.15:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
                label = "Pilot" if key == "A01" else "Tactical"
                label_width = draw.textlength(label, font=self.fonts[18])
                label_x = min(x + 9, left + size - label_width)
                draw.text((label_x, y - 25), label, font=self.fonts[18], fill=color)
            else:
                draw.line((x - 5, y - 5, x + 5, y + 5), fill=color, width=2)
                draw.line((x - 5, y + 5, x + 5, y - 5), fill=color, width=2)
        draw.text(
            (left + 6, top + size - 18),
            f"View spans {self.span / 1000:.0f} km",
            font=self.fonts[18],
            fill=(185, 206, 222),
        )
        return frame.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def render_video(pid, index, seconds_limit=None):
    source, summary_path = replay_paths(index)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    caption = Captions(index, summary, read_tracks(source))
    duration = (
        min(summary["duration_seconds"], seconds_limit)
        if seconds_limit
        else summary["duration_seconds"]
    )
    fps, width, height = 30, WIDTH, HEIGHT
    count = math.ceil(round(duration, 6) * fps)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checks = OUTPUT / "checks"
    checks.mkdir(exist_ok=True)
    destination = OUTPUT / (
        source.name.replace(".txt.acmi", "") + ("_preview" if seconds_limit else "") + ".mp4"
    )
    partial = destination.with_suffix(".partial.mp4")
    log_path = checks / destination.with_suffix(".ffmpeg.log").name
    start = time.monotonic()
    frame_id = int(time.time() * 1000)
    request(frame_id, index, 0)
    hwnd = main_window(pid)["hwnd"]
    frames = WindowFrames(hwnd)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-threads",
        "4",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(partial),
    ]
    with log_path.open("wb") as log:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for frame_number in range(count):
                seconds = frame_number / fps
                actual = request(frame_id + frame_number + 1, index, seconds)
                if abs(float(actual[1]) - seconds) > 0.002 or tuple(map(int, actual[2:4])) != (
                    width,
                    height,
                ):
                    raise RuntimeError(f"Unexpected Tacview frame: {actual}")
                positions = list(map(float, actual[4:8]))
                for offset in (0, 2):
                    if offset == 2 and seconds > caption.tracks["B01"][-1, 0]:
                        continue
                    x, y = positions[offset : offset + 2]
                    if not (50 < x < width - 50 and 75 < y < height - 90):
                        raise RuntimeError(
                            f"Aircraft outside the engagement frame at {seconds}s: {positions}"
                        )
                frame = frames.get(frame_id + frame_number + 1)
                if frame.size != (width, height):
                    raise RuntimeError(f"Unexpected capture dimensions {frame.size}")
                frame = caption.draw(frame, seconds, list(map(float, actual[4:8])))
                if frame_number in (0, count // 4, count // 2, 3 * count // 4, count - 1):
                    frame.save(
                        checks / f"{destination.stem}_frame_{frame_number:05d}.jpg", quality=90
                    )
                encoder.stdin.write(frame.tobytes())
                if frame_number % (fps * 10) == 0:
                    print(
                        f"{CLIPS[index - 1][0]}: {seconds:.0f}/{duration:.1f}s captured; {time.monotonic() - start:.0f}s elapsed",
                        flush=True,
                    )
        finally:
            frames.close()
            encoder.stdin.close()
            return_code = encoder.wait(timeout=60)
        if return_code:
            raise RuntimeError(f"FFmpeg failed: {log_path.read_text()}")
    partial.replace(destination)
    metadata = {
        "video": destination.name,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "bytes": destination.stat().st_size,
        "source_acmi": str(source.relative_to(ROOT)),
        "source_acmi_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "pilot": summary["pilot_id"],
        "scenario_seed": summary["seed"],
        "outcome": summary["outcome"],
        "frames": count,
        "fps": fps,
        "duration_seconds": count / fps,
        "resolution": [width, height],
        "renderer": "Tacview 1.9.5 native 3D window",
        "capture": "Windows Graphics Capture API with per-frame rendered marker verification",
        "camera": {
            "mode": "free_oblique_engagement",
            "yaw_degrees": 35,
            "pitch_degrees": -40,
            "distance": "1.45 times aircraft separation, minimum 16 km; midpoint of aircraft positions",
            "field_of_view_degrees": 50,
            "aircraft_display_scale": "camera distance / 400; positions and attitudes unchanged",
        },
        "display_copy_changes": "Radar and LockedTarget properties omitted to hide sensor graphics; source ACMI unchanged.",
        "captions": "Most recent logged skill at or before replay time; supplementary top-down plot from source ACMI aircraft tracks.",
        "audio": False,
        "playback_speed": 1,
    }
    destination.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Complete: {destination} ({destination.stat().st_size / 1048576:.1f} MiB)", flush=True)
    return metadata


def main():
    global ACTIVE_WINDOW
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--close", action="store_true")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--clip", type=int, nargs="+", choices=(1, 2, 3))
    parser.add_argument(
        "--seconds", type=float, help="Short preview duration; full recording by default"
    )
    args = parser.parse_args()
    ctypes.windll.user32.SetProcessDPIAware()
    if args.close:
        for window in windows_for_pid(args.pid):
            ctypes.windll.user32.PostMessageW(window["hwnd"], 0x0010, 0, 0)
        return
    prepare(WIDTH, HEIGHT)
    if args.pid:
        pid = args.pid
    else:
        for name in ("error.txt", "loaded.txt", "request.txt", "ready.txt"):
            (WORK / name).unlink(missing_ok=True)
        command = (
            f'"{TACVIEW}" /AddOnsFolder:"{(WORK / "addons").as_posix()}/" '
            "/SaveConfig:false /MaxFrameRate:120"
        )
        registry_path = r"Software\Raia Software Inc.\Tacview\Local\AddOns\ReplayCapture"
        old_enabled = None
        existed = False
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, registry_path) as key:
                existed = True
                try:
                    old_enabled = winreg.QueryValueEx(key, "Enabled")
                except FileNotFoundError:
                    pass
        except FileNotFoundError:
            pass

        def restore_addon_setting():
            if existed:
                with winreg.OpenKey(
                    winreg.HKEY_CURRENT_USER, registry_path, 0, winreg.KEY_SET_VALUE
                ) as key:
                    if old_enabled is None:
                        winreg.DeleteValue(key, "Enabled")
                    else:
                        winreg.SetValueEx(key, "Enabled", 0, old_enabled[1], old_enabled[0])
            else:
                winreg.DeleteKey(winreg.HKEY_CURRENT_USER, registry_path)

        atexit.register(restore_addon_setting)
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, registry_path) as key:
            winreg.SetValueEx(key, "Enabled", 0, winreg.REG_DWORD, 1)
        process = subprocess.Popen(command)
        time.sleep(5)
        pid = process.pid
        (WORK / "process.json").write_text(json.dumps({"pid": pid}), encoding="utf-8")
        print(f"Launched dedicated Tacview capture instance {pid}", flush=True)
        time.sleep(5)
    print("Windows", json.dumps(windows_for_pid(pid)), flush=True)
    for window in windows_for_pid(pid):
        if window["title"] == "Controls List":
            ctypes.windll.user32.PostMessageW(window["hwnd"], 0x0010, 0, 0)
    time.sleep(0.5)
    ACTIVE_WINDOW = main_window(pid)["hwnd"]
    ctypes.windll.user32.SetWindowPos(ACTIVE_WINDOW, -1, 30, 30, WIDTH, HEIGHT, 0x0010 | 0x0040)
    if args.render:
        try:
            rows = [
                render_video(pid, index, args.seconds)
                for index in (args.clip if args.clip else range(1, 4))
            ]
            if not args.seconds:
                rows = [
                    json.loads(path.read_text(encoding="utf-8"))
                    for path in sorted(OUTPUT.glob("pilot_*.json"))
                    if not path.name.endswith((".validation.json", "_preview.json"))
                    and path.with_suffix(".mp4").exists()
                ]
            (OUTPUT / "manifest.json").write_text(
                json.dumps(rows, indent=2) + "\n", encoding="utf-8"
            )
        finally:
            for window in windows_for_pid(pid):
                ctypes.windll.user32.PostMessageW(window["hwnd"], 0x0010, 0, 0)
        return
    frame_id = int(time.time() * 1000)
    rendered = request(frame_id, 1, 20)
    print("Render", rendered, flush=True)
    time.sleep(0.3)
    windows = windows_for_pid(pid)
    print("Windows after setup", json.dumps(windows), flush=True)
    window = main_window(pid)
    capture = WindowFrames(window["hwnd"])
    try:
        frame = capture.get(frame_id)
        source, summary_path = replay_paths(1)
        caption = Captions(1, json.loads(summary_path.read_text()), read_tracks(source))
        caption.draw(frame, 20, list(map(float, rendered[4:8]))).save(WORK / "probe_window.png")
    finally:
        capture.close()
    print("Saved scoped Tacview capture probes", flush=True)


if __name__ == "__main__":
    main()
