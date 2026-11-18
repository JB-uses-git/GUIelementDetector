"""Tap a UI element on a live Android device/emulator found purely from a screenshot.

Usage:
    python src/click_demo.py                      # tap the first EditText (fallback: TextButton)
    python src/click_demo.py --cls TextButton --index 1
    python src/click_demo.py --text "Login"       # needs easyocr: find a button by its label
    python src/click_demo.py --dry-run            # detect and show, but do not tap

Steps: adb screencap -> YOLO detection -> choose element -> tap its centre with adb.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from infer import DEFAULT_WEIGHTS, detect, draw, find_by_text, load_model

ROOT = Path(__file__).resolve().parents[1]


def adb(*args, binary=False):
    out = subprocess.run(["adb", *args], capture_output=True, timeout=30)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode(errors="ignore").strip())
    return out.stdout if binary else out.stdout.decode(errors="ignore")


def check_device():
    """Exit with a helpful message if adb or a device is missing."""
    if shutil.which("adb") is None:
        sys.exit("adb not found. Install Android SDK Platform-Tools and add it to PATH "
                 "(see README, 'Click demo'). Skipping click demo.")
    lines = [l for l in adb("devices").splitlines()[1:] if l.strip().endswith("device")]
    if not lines:
        sys.exit("No Android device or emulator connected. Start an emulator in Android Studio "
                 "(Device Manager > Play) or plug in a phone with USB debugging, then retry.")
    print("Using device:", lines[0].split()[0])


def screenshot():
    png = adb("exec-out", "screencap", "-p", binary=True)
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        sys.exit("Could not decode the device screenshot.")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", default="EditText", help="class to tap, e.g. EditText, TextButton, Icon")
    ap.add_argument("--index", type=int, default=0, help="which match (0 = top-most)")
    ap.add_argument("--text", default=None, help="tap the element whose OCR text contains this")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    check_device()
    img = screenshot()
    dets = detect(load_model(args.weights), img, conf=args.conf, ocr=args.text is not None)
    print(f"Detected {len(dets)} elements")

    if args.text:
        target = find_by_text(dets, args.text)
        if target is None:
            sys.exit(f'No element with text "{args.text}" found.')
    else:
        # Sort matches top-to-bottom, left-to-right; fall back to TextButton.
        matches = sorted([d for d in dets if d["class"] == args.cls], key=lambda d: (d["box"]["y"], d["box"]["x"]))
        if not matches and args.cls != "TextButton":
            print(f"No {args.cls} found, falling back to TextButton")
            matches = sorted([d for d in dets if d["class"] == "TextButton"], key=lambda d: (d["box"]["y"], d["box"]["x"]))
        if len(matches) <= args.index:
            sys.exit(f"Not enough matching elements (found {len(matches)}).")
        target = matches[args.index]

    x, y = target["center"]
    out = draw(img, dets)
    cv2.circle(out, (x, y), 18, (0, 0, 255), 4)
    (ROOT / "results").mkdir(exist_ok=True)
    cv2.imwrite(str(ROOT / "results" / "click_demo.jpg"), out)
    print(f'Target: {target["class"]} conf={target["confidence"]} '
          f'{("text=" + repr(target.get("text"))) if args.text else ""} centre=({x}, {y})')
    if args.dry_run:
        print("Dry run: not tapping. Annotated screen saved to results/click_demo.jpg")
        return
    adb("shell", "input", "tap", str(x), str(y))
    print("Tapped. Annotated screen saved to results/click_demo.jpg")


if __name__ == "__main__":
    main()
