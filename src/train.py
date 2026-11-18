"""Fine-tune a YOLOv8 model on the converted VINS dataset.

Usage:
    python src/train.py                         # yolov8s, 640px, up to 100 epochs
    python src/train.py --model yolov8n.pt      # comparison run
    python src/train.py --imgsz 960 --name yolov8s_960
    python src/train.py --epochs 1 --fraction 0.05   # quick smoke test
"""
import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "yolo" / "data.yaml"
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights"  # pretrained COCO weights are cached here


def pick_device():
    """CUDA GPU if present, else Apple Silicon GPU (MPS), else CPU."""
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt", help="pretrained COCO weights")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = auto (CUDA only, else 16)")
    ap.add_argument("--fraction", type=float, default=1.0, help="use part of the train set")
    ap.add_argument("--name", default=None, help="run name (default: model name + imgsz)")
    args = ap.parse_args()

    device = pick_device()
    if device == "cpu":
        print("No GPU found: training on CPU will be slow. Consider notebooks/train_colab.ipynb")
    name = args.name or f"{Path(args.model).stem}_{args.imgsz}"
    weights = WEIGHTS / args.model
    model = YOLO(str(weights) if weights.exists() else args.model)

    model.train(
        data=str(DATA),
        epochs=args.epochs,
        patience=15,  # early stopping
        imgsz=args.imgsz,
        batch=args.batch if device == "0" else (16 if args.batch == -1 else args.batch),
        fraction=args.fraction,
        device=device,
        workers=4,
        seed=42,
        deterministic=True,
        project=str(RESULTS / "runs"),
        name=name,
        exist_ok=True,
        # Augmentation: colour/brightness + light scale only.
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        scale=0.2,
        fliplr=0.0,  # UI layouts are not mirror-symmetric
        flipud=0.0, degrees=0.0, translate=0.0, shear=0.0, perspective=0.0,
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
        plots=True,
    )

    best = RESULTS / "runs" / name / "weights" / "best.pt"
    out = RESULTS / f"{name}_best.pt"
    shutil.copy(best, out)
    print(f"Best weights copied to {out}")


if __name__ == "__main__":
    main()
