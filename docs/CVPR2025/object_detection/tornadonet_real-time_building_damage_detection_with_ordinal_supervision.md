---
title: >-
  [Paper Note] TornadoNet: Real-Time Building Damage Detection with Ordinal Supervision
description: >-
  [CVPR 2025][Object Detection][Building damage detection] TornadoNet builds the first systematic benchmark for post-tornado street-view building damage assessment. By comparing the performance of the YOLO series (CNNs) and RT-DETR (Transformers) on a 5-level damage detection task and proposing an ordinal-aware supervision strategy, it improves the mAP@0.5 of RT-DETR by 4.8 percentage points. This demonstrates the effectiveness of incorporating the ordinal nature of damage seve…
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "Building damage detection"
  - "ordinal classification"
  - "tornado disaster"
  - "real-time inference"
date: 2026-05-08
content_hash: b8122e9dff8cf06c
---

# TornadoNet: Real-Time Building Damage Detection with Ordinal Supervision

**Conference**: CVPR 2025  
**arXiv**: [2603.11557](https://arxiv.org/abs/2603.11557)  
**Code**: [https://github.com/crumeike/TornadoNet](https://github.com/crumeike/TornadoNet)  
**Area**: Object Detection / Disaster Assessment  
**Keywords**: Building damage detection, ordinal classification, object detection, tornado disaster, real-time inference

## TL;DR
TornadoNet builds the first systematic benchmark for post-tornado street-view building damage assessment. By comparing the performance of the YOLO series (CNNs) and RT-DETR (Transformers) on a 5-level damage detection task and proposing an ordinal-aware supervision strategy, it improves the mAP@0.5 of RT-DETR by 4.8 percentage points. This demonstrates the effectiveness of incorporating the ordinal nature of damage severity into the loss function design.

## Background & Motivation

**Background**: Post-disaster building damage assessment is a critical component of emergency response. Currently, it mostly relies on aerial imagery or manual on-site inspections, with a relative lack of automated street-view level damage detection methods. Deep learning object detection models (such as YOLO and DETR) have achieved outstanding performance in general scenarios, but systematic evaluation and adaptation studies in disaster scenarios remain insufficient.

**Limitations of Prior Work**: (1) Lack of standardized datasets: most existing disaster datasets are from aerial perspectives, lacking high-resolution street-view images and multi-level damage annotations; (2) Treatment of damage levels as unordered categories: standard object detection treats the five damage levels (DS0-DS4) as independent categories, ignoring their ordinal relationship (e.g., misclassifying DS3 as DS4 is much more "acceptable" than misclassifying DS0 as DS4); (3) Lack of controlled comparisons regarding the pros and cons of different detection architectures in disaster scenarios.

**Key Challenge**: Damage assessment is inherently an ordinal classification problem, yet existing object detection frameworks employ standard cross-entropy loss for multi-class classification, completely ignoring the ordinal relationships between categories. This leads to models performing well on the mAP metric but making severe severity misjudgments in practical applications (such as misclassifying complete collapse as no damage).

**Goal**: (1) Establish a standardized street-view building damage detection benchmark; (2) Systematically compare the architectural advantages and disadvantages of CNN and Transformer detectors; (3) Design and validate ordinal-aware supervision strategies to improve the consistency of damage level predictions.

**Key Insight**: The authors utilize 3,333 high-resolution, geotagged street-view images (containing 8,890 annotated building instances) from the 2021 Midwest US tornado disaster. Based on the IN-CORE damage state framework, they perform 5-level annotation and introduce ordinal metrics (Ordinal Top-1 Accuracy, MAOE) to measure the scale consistency of the models.

**Core Idea**: Integrate soft ordinal classification targets and explicit ordinal distance penalties into the classification head of object detection, enabling the model to "know" during training that mispredicting by one level incurs a smaller penalty than mispredicting by two levels.

## Method

### Overall Architecture
TornadoNet does not propose an entirely new model architecture, but rather a systematic benchmark framework. The input consists of street-view images, and the outputs are building bounding boxes with 5-level damage categories (DS0-DS4: No Damage, Minor, Moderate, Severe, Complete Collapse). The framework comprises three components: (1) Dataset construction and annotation pipeline; (2) Standardized training and evaluation of 8 baseline models; (3) Design and integration of the ordinal-aware supervision strategy.

### Key Designs

1. **Five-Level Damage Classification Scheme and Dataset**:

    - **Function**: Provide standardized multi-level damage annotations and high-quality street-view datasets.
    - **Mechanism**: Define five levels from DS0 (No Damage) to DS4 (Complete Collapse) based on the IN-CORE damage state framework. The 3,333 images were collected in the field following the 2021 Midwest US tornadoes and validated through expert cross-annotation. The data is partitioned into 6,184 training, 1,342 validation, and 1,364 test instances (75%/15%/15% split). Each image features a high-resolution street-view perspective with geotagged coordinates.
    - **Design Motivation**: Existing disaster datasets are either from aerial perspectives (such as xBD) or lack fine-grained annotations (often only binary "damaged/undamaged"). Street-view multi-level annotations are crucial for rapid post-disaster assessments in practice (such as vehicle-mounted sweeps).

2. **Soft Ordinal Classification Targets**:

    - **Function**: Encode the ordinal information of damage levels into the target labels of the classification head.
    - **Mechanism**: Unlike traditional one-hot labels that represent DS2 as $[0,0,1,0,0]$, soft ordinal labels adopt a Gaussian distribution centered on the ground-truth level $y_k = \frac{1}{Z}\exp(-\frac{(k-c)^2}{2\psi^2})$, where $c$ is the ground-truth level and $\psi$ controls the distribution width. Consequently, adjacent levels receive non-zero probabilities, allowing the model to learn that "DS1 and DS3 are closer to DS2 than DS0 and DS4". Parameters $\psi=0.5$ and $K=1$ (truncation distance) yield the best results.
    - **Design Motivation**: Standard cross-entropy loss penalizes all misclassifications equally. Soft ordinal labels allow the model to implicitly learn the ordinal structure of levels during training without altering the model architecture—only requiring a replacement of training labels.

3. **Explicit Ordinal Distance Penalty**:

    - **Function**: Explicitly incorporate an ordinal distance penalty term into the loss function.
    - **Mechanism**: An additional term $L_{ord} = \lambda \sum_k |k - c| \cdot p_k$ is added to the standard classification loss, where $p_k$ is the model's predicted probability for level $k$, and $c$ is the true level. When the model predicts a high probability for a level far from the ground truth, this loss imposes a heavier penalty. $\lambda=0.05$ is the optimal weight.
    - **Design Motivation**: While soft labels encode ordinal information from the label side, the ordinal penalty explicitly enforces constraints from the loss side. They can be used complementarily, though experiments indicate that the soft label strategy performs better.

### Loss & Training
All models are trained under a unified protocol: image size 896x896, 250 epochs, with standard data augmentation. The detection loss relies on the respective standard training losses of YOLO/RT-DETR (box regression + classification), with ordinal supervision only modifying the target labels or loss function of the classification branch. Each configuration is run with 3 random seeds, reporting the mean $\pm$ standard deviation.

## Key Experimental Results

### Main Results: Baseline Model Comparison

| Model | Architecture | mAP@0.5 | Ordinal Top-1 ↑ | MAOE ↓ | FPS (A100) | Params |
|------|------|---------|-----------------|--------|------------|--------|
| YOLOv8-n | CNN | 40.98% | 84.01% | 0.78 | 276 | 3.0M |
| YOLOv8-l | CNN | 42.09% | 84.19% | 0.78 | 91 | 43.6M |
| YOLO11-x | CNN | **46.05%** | 85.20% | 0.76 | 66 | 56.8M |
| RT-DETR-L | Transformer | 39.87% | **88.13%** | **0.65** | 78 | 32.0M |
| RT-DETR-X | Transformer | 35.75% | 87.74% | 0.67 | 79 | 65.5M |

### Ablation Study on Ordinal Supervision

| Model | Supervision Strategy | mAP@0.5 | Δ mAP | Ordinal Top-1 ↑ | MAOE ↓ |
|------|----------|---------|-------|-----------------|--------|
| RT-DETR-L | Baseline (Standard Cross-Entropy) | 39.87% | - | 88.13% | 0.65 |
| RT-DETR-L | Soft Ordinal (ψ=0.5, K=1) | **44.70%** | **+4.8pp** | **91.15%** | **0.56** |
| RT-DETR-L | Ordinal Penalty (λ=0.05) | 43.36% | +3.5pp | 89.54% | 0.61 |

### Key Findings
- **Clear Architectural Complementarity**: The YOLO series leads in detection accuracy (mAP) and throughput, but RT-DETR is significantly superior in terms of ordinal consistency metrics (Ordinal Top-1, MAOE). This suggests that the global attention of Transformer architectures possesses an inherent advantage in capturing the ordinal relationships between damage levels.
- **Alignment Needed Between Ordinal Supervision and Architectures**: Soft ordinal labels bring a substantial +4.8pp mAP improvement to RT-DETR, whereas they yield simpler gains for YOLO, indicating that ordinal supervision is more effective under global attention architectures.
- **MAOE is More Practical Than mAP**: In actual disaster response, misclassifying "Severe Damage" as "Moderate Damage" (a difference of 1 level) is far less harmful than misclassifying it as "No Damage" (a difference of 3 levels). MAOE better measures this practical utility.
- RT-DETR-X performs worse than RT-DETR-L, likely due to overfitting on the limited dataset size.

## Highlights & Insights
- The **ordinal-aware supervision strategy** is highly concise and efficient: it merely requires modifying the training labels (soft ordinals) or adding a loss term (ordinal penalty) without changing the model structure, making it plug-and-play. This trick can be extended to all detection/classification tasks with ordered classes (such as medical image grading, product quality grading).
- **Rigorous Benchmark Design**: Unified training protocols, multi-seed repetitions, and simultaneous reporting of detection and ordinal metrics establish a standard of experimental design worth emulating.
- **Discovery of Complementarity Between CNNs and Transformers in Ordinal Classification Tasks**: This serves as a valuable empirical finding.

## Limitations & Future Work
- Limited dataset size (only 3,333 images) may be insufficient for fully training large-scale models.
- Evaluations are restricted to tornado disasters; generalization to other disaster types (earthquakes, floods) has not been validated.
- Street-view perspectives are prone to occlusion (trees, other buildings), which is not fully discussed in the paper.
- Ordinal supervision only modifies the classification head; exploring the projection of ordinal information into feature extraction layers remains unaddressed.

## Related Work & Insights
- **vs xBD Dataset**: xBD evaluates a 4-level damage scheme using pre- vs. post-disaster satellite imagery comparisons, whereas TornadoNet focuses on street-view perspectives with 5-level annotations, making it more practical for real-world drive-by surveys.
- **vs Standard Object Detection Benchmarks (COCO, etc.)**: TornadoNet emphasizes ordinal consistency beyond detection accuracy, introducing new evaluation dimensions such as Ordinal Top-1 and MAOE.
- **vs Ordinal Regression Literature**: Traditional ordinal regression methods (such as CORAL) mainly target full-image classification, whereas TornadoNet is the first to introduce ordinal concepts into the classification head of an object detection framework.

## Rating
- **Novelty**: ⭐⭐⭐ Technical innovation is somewhat limited (soft labels + ordinal penalties are relatively simple), but the benchmark construction is highly valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid experimental design with an 8-model comparison, multi-seed validation, and multi-dimensional metrics.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, detailed dataset descriptions, and transparent experimental setups.
- **Value**: ⭐⭐⭐⭐ Direct practical value for disaster AI applications; the ordinal supervision strategy is generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] YOLOE: Real-Time Seeing Anything](../../ICCV2025/object_detection/yoloe_realtime_seeing_anything.md)
- [\[AAAI 2026\] YOLO-IOD: Towards Real Time Incremental Object Detection](../../AAAI2026/object_detection/yolo-iod_towards_real_time_incremental_object_detection.md)
- [\[CVPR 2025\] Test-Time Backdoor Detection for Object Detection Models](test-time_backdoor_detection_for_object_detection_models.md)
- [\[ICML 2025\] When Every Millisecond Counts: Real-Time Anomaly Detection via the Multimodal Asynchronous Hybrid Network](../../ICML2025/object_detection/when_every_millisecond_counts_real-time_anomaly_detection_via_the_multimodal_asy.md)
- [\[CVPR 2025\] MI-DETR: An Object Detection Model with Multi-time Inquiries Mechanism](mi-detr_an_object_detection_model_with_multi-time_inquiries_mechanism.md)

</div>

<!-- RELATED:END -->
