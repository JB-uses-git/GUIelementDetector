"""Classical computer-vision baseline (no learning): Canny edges + contours -> boxes.

It cannot name classes, so it is compared with the model class-agnostically:
a box counts as correct if it overlaps any ground-truth element with IoU >= 0.5.

Usage:
    python src/baseline_cv.py                      # compares with results/best.pt
Writes results/baseline_comparison.csv and results/baseline_example.jpg
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from evaluate import load_gt, match
from infer import detect, draw, read_image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "yolo"


def cv_boxes(img):
    """Edges -> merge nearby edges -> outer contours -> bounding boxes (xyxy)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # Join letters of a word / parts of an icon into one blob.
    k = max(3, round(max(img.shape[:2]) / 200))
    blobs = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (k * 2, k)), iterations=1)
    contours, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img.shape[:2]
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh < 0.0002 * w * h or bw * bh > 0.9 * w * h:  # drop specks and full-screen blobs
            continue
        boxes.append([x, y, x + bw, y + bh])
    return np.array(boxes, float).reshape(-1, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "results" / "best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    model = YOLO(args.weights)

    totals = {"cv": [0, 0], "yolo": [0, 0]}  # [true positives, predictions]
    n_gt = 0
    imgs = sorted((DATA / "images" / "test").glob("*.jpg"))
    for i, p in enumerate(imgs):
        img = read_image(p)
        h, w = img.shape[:2]
        gt_cls, gt_box = load_gt(DATA / "labels" / "test" / f"{p.stem}.txt", w, h)
        n_gt += len(gt_box)

        cvb = cv_boxes(img)
        _, tp = match(gt_cls, gt_box, np.zeros(len(cvb), int), cvb, class_aware=False)
        totals["cv"][0] += tp
        totals["cv"][1] += len(cvb)

        dets = detect(model, img, conf=args.conf)
        yb = np.array([[d["box"]["x"], d["box"]["y"], d["box"]["x"] + d["box"]["w"], d["box"]["y"] + d["box"]["h"]]
                       for d in dets]).reshape(-1, 4)
        _, tp = match(gt_cls, gt_box, np.zeros(len(yb), int), yb, class_aware=False)
        totals["yolo"][0] += tp
        totals["yolo"][1] += len(yb)

        if i == 0:  # save one visual example: CV baseline | YOLO
            a = img.copy()
            for x1, y1, x2, y2 in cvb.astype(int):
                cv2.rectangle(a, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite(str(ROOT / "results" / "baseline_example.jpg"), cv2.hconcat([a, draw(img, dets)]))

    rows = []
    for name, (tp, npred) in totals.items():
        p, r = tp / max(npred, 1), tp / max(n_gt, 1)
        rows.append({"method": "Canny + contours" if name == "cv" else "YOLO (class ignored)",
                     "precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(p + r, 1e-9), 3),
                     "names_classes": name == "yolo"})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "baseline_comparison.csv", index=False)
    print(f"Test images: {len(imgs)}, ground-truth boxes: {n_gt} (IoU >= 0.5, class-agnostic)")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
