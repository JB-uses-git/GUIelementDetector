"""Convert the VINS dataset (Pascal VOC XML) into YOLO format.

Usage:
    python src/prepare_data.py

Reads   data/raw/All Dataset/<subset>/{Annotations,JPEGImages}
Writes  data/yolo/{images,labels}/{train,val,test} and data/yolo/data.yaml
Saves   results/class_counts.csv and results/label_check/*.jpg
"""
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "All Dataset"
OUT = ROOT / "data" / "yolo"
RESULTS = ROOT / "results"
SEED = 42

# Real screenshots only. "Wireframes" are hand-drawn style mockups, not screenshots.
SUBSETS = ["Android", "Rico", "iphone", "uplabs"]

# Final class list (order = YOLO class id).
CLASSES = [
    "Text", "Icon", "Image", "TextButton", "UpperTaskBar", "EditText",
    "PageIndicator", "CheckedTextView", "BackgroundImage", "Modal",
    "Toolbar", "Drawer", "Switch",
]

# Spelling variants merged into an existing class.
MERGE = {"Checkbox": "CheckedTextView", "CheckBox": "CheckedTextView"}
# Classes with < 100 boxes in the whole dataset are dropped (too rare to learn).
DROP = {"Card", "Multi_Tab", "Bottom_Navigation", "Spinner", "Map", "Remember"}


def parse_xml(xml_path):
    """Return (width, height, [(class_name, x1, y1, x2, y2), ...])."""
    root = ET.parse(xml_path).getroot()
    w = int(float(root.findtext("size/width")))
    h = int(float(root.findtext("size/height")))
    objs = []
    for o in root.iter("object"):
        name = o.findtext("name").strip()
        b = o.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        objs.append((name, x1, y1, x2, y2))
    return w, h, objs


def to_yolo_lines(img_w, img_h, objs, stats):
    """Convert boxes to 'cls xc yc w h' lines (normalised 0-1)."""
    lines = []
    for name, x1, y1, x2, y2 in objs:
        name = MERGE.get(name, name)
        if name in DROP or name not in CLASSES:
            stats["dropped"][name] += 1
            continue
        # Clip to image and skip degenerate boxes.
        x1, x2 = max(0, min(x1, x2)), min(img_w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(img_h, max(y1, y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            stats["degenerate"] += 1
            continue
        xc, yc = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h
        bw, bh = (x2 - x1) / img_w, (y2 - y1) / img_h
        lines.append(f"{CLASSES.index(name)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        stats["kept"][name] += 1
    return lines


def collect_pairs():
    """Find (image, xml) pairs; skip images without annotations or unreadable files."""
    pairs = []
    for sub in SUBSETS:
        imgs = {p.stem: p for p in (RAW / sub / "JPEGImages").glob("*.jpg")}
        for xml in sorted((RAW / sub / "Annotations").glob("*.xml")):
            if xml.stem in imgs:
                pairs.append((sub, imgs[xml.stem], xml))
    return pairs


def draw_check(n=10):
    """Draw converted YOLO labels back onto random images to verify the conversion."""
    out = RESULTS / "label_check"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    imgs = sorted((OUT / "images" / "train").glob("*.jpg"))
    rng = random.Random(SEED)
    for p in rng.sample(imgs, n):
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        for line in (OUT / "labels" / "train" / f"{p.stem}.txt").read_text().splitlines():
            c, xc, yc, bw, bh = line.split()
            xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
            x1, y1, x2, y2 = int(xc - bw / 2), int(yc - bh / 2), int(xc + bw / 2), int(yc + bh / 2)
            color = COLORS[int(c) % len(COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, CLASSES[int(c)], (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imwrite(str(out / p.name), img)
    print(f"Saved {n} label check images to {out}")


COLORS = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
          (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
          (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100)]


def main():
    pairs = collect_pairs()
    print(f"Found {len(pairs)} annotated screenshots in {SUBSETS}")

    # 70 / 15 / 15 split with a fixed seed.
    random.Random(SEED).shuffle(pairs)
    n = len(pairs)
    n_train, n_val = int(0.70 * n), int(0.15 * n)
    splits = {"train": pairs[:n_train], "val": pairs[n_train:n_train + n_val], "test": pairs[n_train + n_val:]}

    shutil.rmtree(OUT, ignore_errors=True)
    stats = {"kept": Counter(), "dropped": Counter(), "degenerate": 0}
    split_counts = {s: Counter() for s in splits}
    skipped = 0
    for split, items in splits.items():
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)
        for sub, img_path, xml_path in items:
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue
            h, w = img.shape[:2]
            xml_w, xml_h, objs = parse_xml(xml_path)
            # Some XML sizes disagree with the real image; rescale boxes if so.
            if (xml_w, xml_h) != (w, h) and xml_w > 0 and xml_h > 0:
                sx, sy = w / xml_w, h / xml_h
                objs = [(c, x1 * sx, y1 * sy, x2 * sx, y2 * sy) for c, x1, y1, x2, y2 in objs]
            before = stats["kept"].copy()
            lines = to_yolo_lines(w, h, objs, stats)
            split_counts[split] += stats["kept"] - before
            name = f"{sub}_{img_path.stem}"  # prefix avoids name clashes between subsets
            shutil.copy(img_path, OUT / "images" / split / f"{name}.jpg")
            (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines))

    # data.yaml without "path": Ultralytics then resolves splits relative to this file,
    # so the folder can be moved or zipped for Colab.
    yaml = ["train: images/train", "val: images/val", "test: images/test",
            f"nc: {len(CLASSES)}", "names:"] + [f"  {i}: {c}" for i, c in enumerate(CLASSES)]
    (OUT / "data.yaml").write_text("\n".join(yaml) + "\n")

    # Class counts table.
    RESULTS.mkdir(exist_ok=True)
    df = pd.DataFrame({s: [split_counts[s][c] for c in CLASSES] for s in splits}, index=CLASSES)
    df["total"] = df.sum(axis=1)
    df.index.name = "class"
    df.to_csv(RESULTS / "class_counts.csv")
    print("\nImages per split:", {s: len(v) for s, v in splits.items()}, "unreadable:", skipped)
    print("\nBoxes per class:\n", df.to_string())
    print("\nDropped (rare/unknown):", dict(stats["dropped"]), "| degenerate boxes:", stats["degenerate"])

    draw_check()


if __name__ == "__main__":
    main()
