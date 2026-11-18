"""Detect UI elements in one screenshot.

Usage:
    python src/infer.py path/to/screenshot.png
    python src/infer.py shot.png --conf 0.3 --out results/infer --ocr

Writes <out>/<name>_pred.jpg (annotated image) and <out>/<name>_pred.json:
    [{"class": "TextButton", "confidence": 0.91,
      "box": {"x": 40, "y": 500, "w": 420, "h": 50}, "center": [250, 525]}, ...]
(x, y) is the top-left corner in pixels; "center" is where you would tap.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "results" / "best.pt"
OCR_CLASSES = {"Text", "TextButton", "EditText"}

# One BGR colour per class id.
COLORS = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
          (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
          (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100)]

_ocr_reader = None


def load_model(weights=DEFAULT_WEIGHTS):
    return YOLO(str(weights))


def detect(model, image_bgr, conf=0.25, imgsz=640, ocr=False):
    """Run the model on a BGR numpy image and return a list of detection dicts."""
    res = model.predict(image_bgr, conf=conf, imgsz=imgsz, verbose=False)[0]
    dets = []
    for (x1, y1, x2, y2), c, cls in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(),
                                       res.boxes.cls.tolist()):
        x, y, w, h = round(x1), round(y1), round(x2 - x1), round(y2 - y1)
        dets.append({
            "class": res.names[int(cls)],
            "class_id": int(cls),
            "confidence": round(c, 3),
            "box": {"x": x, "y": y, "w": w, "h": h},
            "center": [x + w // 2, y + h // 2],
        })
    if ocr:
        add_ocr_text(image_bgr, dets)
    return dets


def add_ocr_text(image_bgr, dets):
    """Read the text inside Text/TextButton/EditText boxes with EasyOCR (optional stretch)."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr  # imported lazily: OCR is optional
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    for d in dets:
        if d["class"] not in OCR_CLASSES:
            continue
        b = d["box"]
        crop = image_bgr[max(0, b["y"]):b["y"] + b["h"], max(0, b["x"]):b["x"] + b["w"]]
        if crop.size == 0:
            continue
        d["text"] = " ".join(_ocr_reader.readtext(crop, detail=0))


def find_by_text(dets, query):
    """Return the detection whose OCR text best matches `query` (case-insensitive), or None."""
    q = query.lower().strip()
    hits = [d for d in dets if q in d.get("text", "").lower()]
    # Prefer buttons, then shortest text (closest match).
    hits.sort(key=lambda d: (d["class"] != "TextButton", len(d.get("text", ""))))
    return hits[0] if hits else None


def draw(image_bgr, dets):
    """Return a copy of the image with labelled boxes drawn on it."""
    img = image_bgr.copy()
    thick = max(1, round(max(img.shape[:2]) / 600))
    for d in dets:
        b, color = d["box"], COLORS[d["class_id"] % len(COLORS)]
        cv2.rectangle(img, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), color, thick)
        label = f'{d["class"]} {d["confidence"]:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4 * thick, 1)
        y0 = max(th + 4, b["y"])
        cv2.rectangle(img, (b["x"], y0 - th - 4), (b["x"] + tw + 2, y0), color, -1)
        cv2.putText(img, label, (b["x"] + 1, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * thick,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return img


def read_image(path):
    """cv2.imread that also works with non-ASCII Windows paths."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {path}")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=str(ROOT / "results" / "infer"))
    ap.add_argument("--ocr", action="store_true", help="read text in Text/TextButton boxes")
    args = ap.parse_args()

    img = read_image(args.image)
    dets = detect(load_model(args.weights), img, args.conf, args.imgsz, args.ocr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    cv2.imwrite(str(out / f"{stem}_pred.jpg"), draw(img, dets))
    (out / f"{stem}_pred.json").write_text(json.dumps(dets, indent=2))
    print(f"{len(dets)} elements found -> {out / (stem + '_pred.jpg')}, {out / (stem + '_pred.json')}")


if __name__ == "__main__":
    main()
