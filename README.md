# Road Scene Object Detection with YOLOv2

A computer-vision project that builds an end-to-end **road-scene object-detection inference pipeline** around a pretrained YOLOv2 network. The project implements the key post-processing stages needed to turn dense neural-network outputs into usable object detections: confidence filtering, bounding-box conversion and scaling, Intersection over Union (IoU), non-max suppression (NMS), and annotated-image inference.

![Example road-scene detection](assets/example_detection.png)

## Project Overview

Object detection is a foundational perception task for autonomous-driving systems. Unlike image classification, which assigns a label to an entire image, object detection must identify **what** objects are present and **where** they are located. A practical road-scene detector therefore needs to produce class labels, confidence scores, and bounding boxes for multiple objects in a single frame.

This project uses a **pretrained YOLOv2** model as the feature extractor/detector and focuses on the inference engineering around it. The model is not trained from scratch here. Instead, the project implements and integrates the logic that converts YOLO's dense output tensor into a compact set of final detections.

The repository therefore distinguishes the custom inference/post-processing logic from the external pretrained model assets.

## What the Project Implements

The project implements the following core functionality:

- class-score computation from object confidence and class probabilities;
- confidence-threshold filtering of low-quality candidate boxes;
- Intersection over Union (IoU) for bounding-box overlap measurement;
- TensorFlow-based non-max suppression to remove duplicate detections;
- YOLO center/size to corner-coordinate conversion;
- box scaling from model coordinates back to the source-image resolution;
- an end-to-end image inference function that preprocesses, predicts, filters, draws and saves detections;
- lightweight deterministic checks for the bounding-box geometry implementation.

## Model and Detection Configuration

| Property | Value |
|---|---:|
| Detector | YOLOv2 |
| Model type | Pretrained object detector |
| Input size | 608 × 608 × 3 |
| Output grid | 19 × 19 |
| Anchor boxes per grid cell | 5 |
| Classes | 80 COCO classes |
| Values per anchor | 85 |
| Raw output shape | 19 × 19 × 5 × 85 |
| Candidate boxes before filtering | 1,805 |
| Total parameters | 50,983,561 |
| Trainable parameters in saved architecture | 50,962,889 |
| Non-trainable parameters | 20,672 |
| Example score threshold | 0.30 |
| Example NMS IoU threshold | 0.50 |
| Example max detections | 10 |

The 85 values associated with each anchor correspond to 4 box parameters, 1 object-confidence value and 80 class probabilities.

## End-to-End Pipeline

```text
Input road image
      │
      ▼
Resize / normalize to 608 × 608
      │
      ▼
Pretrained YOLOv2 forward pass
      │
      ▼
19 × 19 × 5 × 85 prediction tensor
      │
      ▼
YOLO decoding
      │
      ▼
Class-score computation
      │
      ▼
Confidence thresholding
      │
      ▼
Convert and rescale bounding boxes
      │
      ▼
Non-max suppression
      │
      ▼
Final scores + boxes + class IDs
      │
      ▼
Draw and save annotated image
```

## How the Post-Processing Works

### 1. Class-score filtering

YOLO predicts an object-confidence value and a probability distribution over the supported classes for every candidate anchor box. The project computes a score for each class using:

\[
\text{class score} = \text{object confidence} \times \text{class probability}
\]

For each candidate box, the class with the largest score is selected. Boxes whose best score is below the configured threshold are removed. In the notebook this logic is implemented in `yolo_filter_boxes` using TensorFlow operations such as `tf.math.argmax`, `tf.math.reduce_max` and `tf.boolean_mask`.

### 2. Intersection over Union

Intersection over Union measures the overlap between two bounding boxes:

\[
IoU(A,B)=\frac{|A \cap B|}{|A \cup B|}
\]

An IoU of 0 means the boxes do not overlap, while a value close to 1 means they cover almost the same area. The project includes a standalone `iou` implementation to make the geometry explicit and testable.

### 3. Non-max suppression

Dense detectors frequently predict several boxes around the same physical object. Non-max suppression keeps the strongest detection and removes sufficiently overlapping lower-confidence boxes. The project uses `tf.image.non_max_suppression` and then gathers the corresponding class IDs and box coordinates.

### 4. Coordinate conversion and scaling

YOLO predicts boxes using a center coordinate plus width and height. `yolo_boxes_to_corners` converts those values into corner coordinates. After confidence filtering, the boxes are scaled from the model coordinate system back to the dimensions of the original image so they can be drawn in the correct locations.

### 5. Final evaluation stage

`yolo_eval` combines the complete post-processing pipeline: decode-ready boxes are converted to corners, filtered by score, scaled to the original image size and passed through NMS. The resulting tensors contain only the final scores, boxes and class IDs.

## Example Inference Result

The preserved sample run uses `test.jpg` with a score threshold of `0.30`, NMS IoU threshold of `0.50`, and a maximum of 10 output boxes. The pipeline returns **10 final detections**.

| Detected class | Confidence | Bounding box |
|---|---:|---|
| car | 0.89 | (367, 300) → (745, 648) |
| car | 0.80 | (761, 282) → (942, 412) |
| car | 0.74 | (159, 303) → (346, 440) |
| car | 0.70 | (947, 324) → (1280, 705) |
| bus | 0.67 | (5, 266) → (220, 407) |
| car | 0.66 | (706, 279) → (786, 350) |
| car | 0.60 | (925, 285) → (1045, 374) |
| car | 0.44 | (336, 296) → (378, 335) |
| car | 0.37 | (965, 273) → (1022, 292) |
| traffic light | 0.36 | (681, 195) → (692, 214) |

The strongest detection in the recorded frame is a car at **0.89 confidence**. The result also demonstrates multi-class detection by identifying a bus and a traffic light alongside the cars.

## Important Result Interpretation

This repository does **not** report an accuracy or mAP score because the notebook does not evaluate the detector against a labeled validation/test set. The recorded image is an inference demonstration, not a quantitative benchmark.

Similarly, the YOLOv2 weights are pretrained. No model training or fine-tuning is performed in this project. This keeps the project description technically accurate and makes it clear which parts of the system are original implementation work and which parts are external model assets.

## Repository Structure

```text
.
├── YOLOv2_Autonomous_Driving_Object_Detection.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── assets/
│   └── example_detection.png
├── images/
│   └── test.jpg                 # add locally
├── model_data/
│   ├── coco_classes.txt         # required locally
│   ├── yolo_anchors.txt         # required locally
│   └── ... pretrained model files
├── yad2k/                       # YAD2K source/package
└── out/                         # generated annotated images
```

The `model_data`, `images`, and `yad2k` resources are external runtime dependencies and are not embedded into the cleaned portfolio bundle.

## Setup

### 1. Create an environment

The original notebook was produced in a legacy Python/TensorFlow environment. For easiest compatibility, use a Python version supported by the TensorFlow release you install.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\\Scripts\\activate       # Windows
pip install -r requirements.txt
```

### 2. Add YAD2K

The notebook imports `yolo_head` and utility functions from YAD2K. Place the YAD2K package in the repository root so the following path exists:

```text
yad2k/models/keras_yolo.py
```

YAD2K repository: https://github.com/allanzelener/YAD2K

### 3. Add the pretrained model assets

Create a `model_data/` directory containing:

```text
model_data/
├── coco_classes.txt
├── yolo_anchors.txt
└── pretrained YOLOv2 SavedModel / model files
```

The notebook expects `load_model("model_data/", compile=False)` to load the saved model. If your model is stored as a single `.h5`/`.keras` file instead, update `MODEL_DIR`/the model-loading cell accordingly.

### 4. Add an input image

Place the road image inside `images/` and call:

```python
out_scores, out_boxes, out_classes = predict("test.jpg")
```

The annotated result is written to `out/` by default.

## Using Different Thresholds

The `predict` function exposes the main post-processing controls:

```python
predict(
    "test.jpg",
    score_threshold=0.40,
    iou_threshold=0.50,
    max_boxes=15,
)
```

A higher score threshold generally reduces false positives but can also remove valid lower-confidence detections. A lower NMS IoU threshold suppresses overlapping boxes more aggressively.

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Pillow
- YOLOv2
- YAD2K
- Jupyter Notebook

## Engineering Design

The notebook is organized as reusable project code rather than a sequence of isolated demonstrations. Functions have concise docstrings, inference thresholds are configurable, path handling uses `pathlib`, missing assets produce actionable errors, output directories are created automatically, and a recorded inference result is included as execution evidence.

## Limitations

This project is an inference prototype rather than a production autonomous-driving perception system. Important limitations include:

- the model is YOLOv2, which is substantially older than modern object detectors;
- the detector is not fine-tuned specifically for the road-scene dataset;
- no labeled validation/test benchmark is included, so mAP/precision/recall are not reported;
- inference latency and real-time throughput are not benchmarked;
- the post-processing NMS is global rather than explicitly class-aware;
- YAD2K is a legacy dependency and may require version compatibility work on recent TensorFlow/Keras releases;
- the notebook processes individual images rather than a real-time camera/video stream.

## Future Improvements

A natural next version of the project would add a labeled evaluation set and compute mAP, precision, recall and per-class performance. Additional improvements could include video-stream inference, latency profiling, class-aware batched NMS, fine-tuning on driving-specific data, automated unit tests, experiment configuration files, and migration to a modern detector/export stack such as a current YOLO implementation with ONNX or TensorRT deployment.

## Skills Demonstrated

This project demonstrates practical understanding of convolutional object detectors, bounding-box representations, tensor operations, IoU, confidence filtering, NMS, pretrained-model integration, image preprocessing, inference visualization, reproducible project structure, and technically accurate reporting of model scope and results.

## Attribution and References

The pretrained detector and decoding approach build on the following work:

- Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi, **You Only Look Once: Unified, Real-Time Object Detection** (2015): https://arxiv.org/abs/1506.02640
- Joseph Redmon, Ali Farhadi, **YOLO9000: Better, Faster, Stronger** (2016): https://arxiv.org/abs/1612.08242
- Allan Zelener, **YAD2K: Yet Another Darknet 2 Keras**: https://github.com/allanzelener/YAD2K
- Original YOLO/Darknet project: https://pjreddie.com/darknet/yolo/

The road-scene sample is from the Drive.ai sample dataset, which was distributed under the **Creative Commons Attribution 4.0 International License**. Pretrained YOLOv2 weights and YAD2K utilities are external components; the project implementation focuses on the post-processing and inference pipeline built around them.
