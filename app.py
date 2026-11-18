"""Gradio demo: upload a screenshot, get labelled UI elements + Markov Chain layout sequence analysis.

Usage:
    python app.py            then open http://127.0.0.1:7860
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import cv2
import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from infer import DEFAULT_WEIGHTS, detect, draw, load_model  # noqa: E402
from markov import UIMarkovModel  # noqa: E402

model = load_model(DEFAULT_WEIGHTS)
global_markov = UIMarkovModel()

# Pick examples from data/ if available, or fall back to results/label_check
example_paths = sorted((ROOT / "data" / "yolo" / "images" / "test").glob("*.jpg"))
if not example_paths:
    example_paths = sorted((ROOT / "results" / "label_check").glob("*.jpg"))
examples = [[str(p)] for p in example_paths[:6]]


def run(image_rgb, conf, ocr, accumulate_markov):
    if image_rgb is None:
        return None, None, "### No image provided\nPlease upload or select a sample screenshot.", []

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    dets = detect(model, bgr, conf=conf, ocr=ocr)
    annotated = cv2.cvtColor(draw(bgr, dets), cv2.COLOR_BGR2RGB)
    clean_dets = [{k: v for k, v in d.items() if k != "class_id"} for d in dets]

    # Markov Chain layout modeling
    seq = global_markov.extract_sequence(dets)
    if accumulate_markov:
        global_markov.add_sequence(seq)
        df_probs = global_markov.get_transition_matrix()
        title = "Markov Transition Matrix: P(Next | Current) [Accumulated]"
    else:
        df_probs = global_markov.get_transition_matrix(sequence_only=seq)
        title = "Markov Transition Matrix: P(Next | Current) [Current Screen Only]"

    heatmap_img = global_markov.plot_heatmap(df_probs, title=title)
    report_md = global_markov.generate_report(seq, df_probs)

    return annotated, heatmap_img, report_md, clean_dets


def reset_markov():
    global global_markov
    global_markov = UIMarkovModel()
    return None, "### Markov Memory Reset\nAccumulated transitions cleared. Click **Detect UI Elements** to start fresh."


with gr.Blocks(title="GUI Element Detector + Markov Sequence Model") as demo:
    gr.Markdown(
        "# 📱 GUI Element Detector + Markov Chain Layout Model\n"
        "1. **Object Detection**: YOLOv8 predicts bounding boxes and classes for mobile UI elements.\n"
        "2. **Markov Chain Modeling**: Models the sequential reading order (Top ➔ Bottom) and transition probabilities between UI elements."
    )

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="numpy", label="1. Upload Screenshot", height=480)
            with gr.Row():
                conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence Threshold")
                ocr = gr.Checkbox(False, label="OCR Text")
                accumulate = gr.Checkbox(True, label="Accumulate Markov Transitions")
            with gr.Row():
                btn = gr.Button("🔍 Detect UI Elements", variant="primary", scale=2)
                reset_btn = gr.Button("🔄 Reset Markov", variant="secondary", scale=1)

        with gr.Column(scale=1):
            out_img = gr.Image(label="2. YOLOv8 Detections (Annotated Image)", height=480)

    gr.Markdown("---")
    gr.Markdown("## 🧠 Markov Chain: Layout Sequence & Transition Probabilities")

    with gr.Row():
        with gr.Column(scale=1):
            markov_img = gr.Image(label="Markov Transition Heatmap P(Next | Current)", height=420)
        with gr.Column(scale=1):
            markov_md = gr.Markdown(
                "### Layout Sequence\n"
                "*Click **Detect UI Elements** or pick an example below to view the Markov sequence breakdown.*"
            )

    with gr.Accordion("📋 View Raw Detections (JSON)", open=False):
        out_json = gr.JSON(label="Bounding Boxes & Confidence Scores")

    if examples:
        gr.Markdown("### 📂 Sample Screenshots (Click any to test instantly):")
        gr.Examples(
            examples,
            inputs=inp,
            label="Sample Screenshots"
        )

    btn.click(run, [inp, conf, ocr, accumulate], [out_img, markov_img, markov_md, out_json])
    conf.release(run, [inp, conf, ocr, accumulate], [out_img, markov_img, markov_md, out_json])
    reset_btn.click(reset_markov, inputs=[], outputs=[markov_img, markov_md])

if __name__ == "__main__":
    demo.launch()
