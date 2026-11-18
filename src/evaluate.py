"""Evaluate a trained model on the held-out test split.

Usage:
    python src/evaluate.py --weights results/yolov8s_640_best.pt            # metrics only
    python src/evaluate.py --weights results/best.pt --main                  # + samples, custom_test

Always writes:
    results/eval/<name>/metrics.csv        overall + per-class P, R, AP50, AP50-95
    results/eval/<name>/size_recall.csv    recall for small / medium / large boxes
    results/comparison.csv                 one row per evaluated model
With --main also writes results/metrics.csv, results/confusion_matrix.png,
results/samples/ (3 good + 3 failure cases) and results/custom_test/ predictions.
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from infer import COLORS, detect, draw, read_image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "yolo"
RESULTS = ROOT / "results"


def load_gt(label_path, w, h):
    """YOLO label file -> (classes, xyxy pixel boxes)."""
    rows = [l.split() for l in label_path.read_text().splitlines() if l.strip()]
    if not rows:
        return np.zeros(0, int), np.zeros((0, 4))
    a = np.array(rows, float)
    xc, yc, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    return a[:, 0].astype(int), np.stack([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], 1)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(gt_cls, gt_box, pr_cls, pr_box, thr=0.5, class_aware=True):
    """Greedy matching (predictions are sorted by confidence). Returns matched-GT mask and TP count."""
    ious = iou_matrix(pr_box, gt_box)
    used = np.zeros(len(gt_box), bool)
    tp = 0
    for i in range(len(pr_box)):
        cand = ious[i].copy()
        cand[used] = 0
        if class_aware:
            cand[gt_cls != pr_cls[i]] = 0
        j = int(cand.argmax()) if len(cand) else -1
        if j >= 0 and cand[j] >= thr:
            used[j] = True
            tp += 1
    return used, tp


def per_image_analysis(model, imgsz, conf):
    """Per-image F1 and recall by box size (COCO small/medium/large at model input scale)."""
    rows, size_hits = [], {"small": [0, 0], "medium": [0, 0], "large": [0, 0]}
    for img_path in sorted((DATA / "images" / "test").glob("*.jpg")):
        img = read_image(img_path)
        h, w = img.shape[:2]
        gt_cls, gt_box = load_gt(DATA / "labels" / "test" / f"{img_path.stem}.txt", w, h)
        dets = detect(model, img, conf=conf, imgsz=imgsz)
        pr_cls = np.array([d["class_id"] for d in dets], int)
        pr_box = np.array([[d["box"]["x"], d["box"]["y"], d["box"]["x"] + d["box"]["w"],
                            d["box"]["y"] + d["box"]["h"]] for d in dets]).reshape(-1, 4)
        used, tp = match(gt_cls, gt_box, pr_cls, pr_box)
        p = tp / max(len(dets), 1)
        r = tp / max(len(gt_box), 1)
        rows.append({"image": img_path.name, "n_gt": len(gt_box), "n_pred": len(dets), "tp": tp,
                     "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9)})
        # Box area measured after resizing the image so its long side = imgsz.
        scale = imgsz / max(h, w)
        areas = (gt_box[:, 2] - gt_box[:, 0]) * (gt_box[:, 3] - gt_box[:, 1]) * scale ** 2
        for a, hit in zip(areas, used):
            k = "small" if a < 32 ** 2 else "medium" if a < 96 ** 2 else "large"
            size_hits[k][0] += int(hit)
            size_hits[k][1] += 1
    size = pd.DataFrame([{"size": k, "gt_boxes": n, "recall": hit / max(n, 1)}
                         for k, (hit, n) in size_hits.items()])
    return pd.DataFrame(rows), size


def save_samples(model, imgsz, conf, per_img, names):
    """Save 3 best and 3 worst test images as GT | prediction side by side."""
    out = RESULTS / "samples"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    pool = per_img[per_img.n_gt >= 5].sort_values("f1")
    picks = [("good", r) for r in pool.tail(3).iloc[::-1].itertuples()] + \
            [("fail", r) for r in pool.head(3).itertuples()]
    notes = []
    for i, (kind, r) in enumerate(picks, 1):
        img = read_image(DATA / "images" / "test" / r.image)
        h, w = img.shape[:2]
        gt_cls, gt_box = load_gt(DATA / "labels" / "test" / f"{Path(r.image).stem}.txt", w, h)
        gt_img = img.copy()
        for c, (x1, y1, x2, y2) in zip(gt_cls, gt_box.astype(int)):
            cv2.rectangle(gt_img, (x1, y1), (x2, y2), COLORS[c % len(COLORS)], max(1, w // 400))
        dets = detect(model, img, conf=conf, imgsz=imgsz)
        pred_img = draw(img, dets)
        for im, title in ((gt_img, "ground truth"), (pred_img, "prediction")):
            cv2.putText(im, title, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imwrite(str(out / f"{i}_{kind}_{Path(r.image).stem}.jpg"), cv2.hconcat([gt_img, pred_img]))
        gt_counts = pd.Series([names[c] for c in gt_cls]).value_counts().to_dict()
        notes.append(f"- {i}_{kind}: {r.image} | GT {r.n_gt}, pred {r.n_pred}, "
                     f"P={r.precision:.2f} R={r.recall:.2f} F1={r.f1:.2f} | GT classes {gt_counts}")
    (out / "notes.md").write_text("\n".join(notes) + "\n")
    print("\n".join(notes))


def run_custom_test(model, imgsz, conf):
    """Annotate any screenshots the user dropped into data/custom_test (kept separate)."""
    src = ROOT / "data" / "custom_test"
    imgs = [p for p in src.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if not imgs:
        print("data/custom_test is empty - skipping custom screenshots")
        return
    out = RESULTS / "custom_test"
    out.mkdir(parents=True, exist_ok=True)
    for p in imgs:
        img = read_image(p)
        cv2.imwrite(str(out / f"{p.stem}_pred.jpg"), draw(img, detect(model, img, conf=conf, imgsz=imgsz)))
    print(f"Annotated {len(imgs)} custom screenshots -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(RESULTS / "best.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25, help="threshold for samples / per-image stats")
    ap.add_argument("--name", default=None)
    ap.add_argument("--main", action="store_true", help="also save samples, confusion matrix, custom test")
    args = ap.parse_args()

    name = args.name or Path(args.weights).stem.replace("_best", "")
    out = RESULTS / "eval" / name
    model = YOLO(args.weights)

    # Standard detection metrics on the test split.
    m = model.val(data=str(DATA / "data.yaml"), split="test", imgsz=args.imgsz, batch=16,
                  project=str(RESULTS / "eval"), name=name, exist_ok=True, plots=True, verbose=False)
    b = m.box
    rows = [{"class": "all", "instances": int(m.nt_per_class.sum()), "precision": b.mp, "recall": b.mr,
             "mAP50": b.map50, "mAP50-95": b.map}]
    nt = m.nt_per_class
    for i, c in enumerate(b.ap_class_index):
        p, r, ap50, ap = b.class_result(i)
        rows.append({"class": m.names[c], "instances": int(nt[c]),
                     "precision": p, "recall": r, "mAP50": ap50, "mAP50-95": ap})
    df = pd.DataFrame(rows).round(3)
    df.to_csv(out / "metrics.csv", index=False)
    print(df.to_string(index=False))

    # Recall by object size + per-image scores.
    per_img, size = per_image_analysis(model, args.imgsz, args.conf)
    size = size.round(3)
    size.to_csv(out / "size_recall.csv", index=False)
    per_img.to_csv(out / "per_image.csv", index=False)
    print("\nRecall by box size (conf >= %.2f, IoU 0.5):\n" % args.conf, size.to_string(index=False))

    # Summary row in the shared comparison table.
    comp_path = RESULTS / "comparison.csv"
    comp = pd.read_csv(comp_path) if comp_path.exists() else pd.DataFrame()
    row = {"model": name, "imgsz": args.imgsz, "params_M": round(sum(p.numel() for p in model.model.parameters()) / 1e6, 1),
           "precision": round(b.mp, 3), "recall": round(b.mr, 3), "mAP50": round(b.map50, 3),
           "mAP50-95": round(b.map, 3), "ms_per_img": round(m.speed["inference"], 1),
           **{f"recall_{s}": v for s, v in zip(size["size"], size["recall"])}}
    if "model" in comp:
        comp = comp[comp["model"] != name]  # replace an older row for the same model
    comp = pd.concat([comp, pd.DataFrame([row])], ignore_index=True)
    comp.to_csv(comp_path, index=False)

    if args.main:
        df.to_csv(RESULTS / "metrics.csv", index=False)
        for f in ("confusion_matrix.png", "confusion_matrix_normalized.png", "BoxPR_curve.png"):
            if (out / f).exists():
                shutil.copy(out / f, RESULTS / f)
        save_samples(model, args.imgsz, args.conf, per_img, m.names)
        run_custom_test(model, args.imgsz, args.conf)


if __name__ == "__main__":
    main()
