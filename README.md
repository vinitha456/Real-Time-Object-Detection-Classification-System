# 🎯 Real-Time Object Detection & Classification System

> A multi-feature computer vision pipeline built with YOLOv8 and TensorFlow — supporting object detection, instance segmentation, and transfer learning classification on real-world images.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [Steps Involved](#steps-involved)
- [Modules Breakdown](#modules-breakdown)
- [Results](#results)
- [Limitations](#limitations)


---

## 🔍 Overview

This capstone project demonstrates a complete computer vision system capable of:

- **Detecting** and localizing objects in images using YOLOv8
- **Segmenting** objects at the pixel level using instance segmentation
- **Classifying** images using a custom transfer learning model built on MobileNetV2
- **Analyzing** and visualizing detection results with class distribution charts

The entire pipeline runs on Google Colab with GPU acceleration (T4).

---

## 🛠 Tech Stack

| Category | Tools / Libraries |
|---|---|
| Object Detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (`yolov8n.pt`, `yolov8n-seg.pt`) |
| Deep Learning | TensorFlow 2.x, Keras |
| Transfer Learning | MobileNetV2 (ImageNet weights) |
| Image Processing | OpenCV (`cv2`) |
| Visualization | Matplotlib |
| Utilities | NumPy, Collections, Time, OS |
| Environment | Google Colab (GPU: T4) |
| Language | Python 3.x |

---

## 🏗 Project Architecture

```
Real-Time CV System
│
├── Setup Layer
│   ├── Dependency Installation (ultralytics, opencv, tensorflow)
│   └── Sample Image Download (bus.jpg, zidane.jpg)
│
├── Preprocessing Layer
│   └── ImagePreprocessor
│       └── resize_if_needed() — caps images at 1280px
│
├── Detection Layer
│   └── ObjectDetector (YOLOv8n)
│       ├── detect()              — runs inference + timing
│       ├── analyze_results()     — counts objects per class
│       ├── visualize_results()   — 3-panel plot (original, annotated, bar chart)
│       └── detect_and_report()   — full pipeline wrapper
│
├── Segmentation Layer
│   └── SegmentationModule (YOLOv8n-seg)
│       └── segment()             — pixel-level mask overlay
│
└── Classification Layer
    └── build_custom_classifier()
        ├── Base: MobileNetV2 (frozen, ImageNet)
        ├── GlobalAveragePooling2D
        ├── Dense(256, ReLU) + Dropout(0.5)
        └── Dense(num_classes, Softmax)
```

---

## 🪜 Steps Involved

### Step 0 — Environment Setup
- Install required libraries: `ultralytics`, `opencv-python`, `matplotlib`, `tensorflow`
- Create a `sample_images/` directory
- Download two open-source test images from Ultralytics

### Step 1 — Import Libraries & Load Models
- Import all necessary packages
- Load YOLOv8 detection model (`yolov8n.pt`) and segmentation model (`yolov8n-seg.pt`)
- Model weights are auto-downloaded on first run

### Step 2 — Image Preprocessing
- Build an `ImagePreprocessor` utility class
- Resize oversized images to a max dimension of 1280px to optimize memory usage and inference speed

### Step 3 — Object Detection Module
- Define the `ObjectDetector` class with methods for:
  - Running inference with a configurable confidence threshold
  - Extracting bounding boxes, class names, and confidence scores
  - Visualizing results in a 3-panel figure
  - Printing a formatted detection report to the console

### Step 4 — Run Detection on Sample Images
- Loop through all images in `sample_images/`
- Run the full detect → analyze → visualize pipeline on each

### Step 5 — Instance Segmentation
- Define the `SegmentationModule` class
- Run `yolov8n-seg` on `bus.jpg`
- Display side-by-side: original vs. pixel-masked segmentation output

### Step 6 — Transfer Learning Classifier
- Load `MobileNetV2` as a frozen feature extractor
- Attach a custom classification head for 5 output classes
- Compile with Adam optimizer and sparse categorical cross-entropy loss
- Print model summary

---

## 📦 Modules Breakdown

### `ImagePreprocessor`
Handles safe image resizing before model inference.

```python
preprocessor.resize_if_needed(img, max_size=1280)
```

### `ObjectDetector`
Full object detection pipeline with reporting.

```python
detector = ObjectDetector(det_model)
detector.detect_and_report("sample_images/bus.jpg", conf_threshold=0.5)
```

### `SegmentationModule`
Pixel-level instance segmentation display.

```python
segmenter = SegmentationModule(seg_model)
segmenter.segment("sample_images/bus.jpg")
```

### `build_custom_classifier()`
Builds a MobileNetV2-based classification model.

```python
model = build_custom_classifier(num_classes=5)
model.summary()
```

---


---

## ⚠️ Limitations

| Limitation | Description |
|---|---|
| **No classifier training** | The MobileNetV2 model is built but never trained — no dataset is loaded and `.fit()` is never called |
| **No pose estimation** | Mentioned in the project requirements but not implemented |
| **No video/webcam support** | Only static images are processed; real-time video inference is not included |
| **Small test dataset** | Only 2 sample images are used — no comprehensive evaluation metrics (mAP, precision, recall) |



---


## 📁 Project Structure

```
├── Capstone_Project_computer_vision.ipynb   # Main notebook
├── sample_images/
│   ├── bus.jpg
│   └── zidane.jpg
└── README.md
```

---

*Built as part of a Computer Vision Capstone Project | YOLOv8 + TensorFlow | Google Colab*
