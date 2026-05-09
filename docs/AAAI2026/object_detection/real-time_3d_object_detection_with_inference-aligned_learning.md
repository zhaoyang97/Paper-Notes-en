---
title: >-
  [Paper Note] Real-Time 3D Object Detection with Inference-Aligned Learning
description: >-
  [AAAI 2026][Object Detection][3D object detection] This paper proposes SR3D, a framework that bridges the training-inference gap in indoor dense 3D object detection via two training-phase components: Spatial-Priority Optimal Transport Assignment (SPOTA) and Ranking-Aware adaptive Self-distillation (RAS). SR3D achieves state-of-the-art performance among dense detectors on ScanNet V2 and SUN RGB-D at a real-time speed of 42ms.
tags:
  - AAAI 2026
  - Object Detection
  - 3D object detection
  - point cloud
  - indoor scene
  - optimal transport
  - label assignment
  - self-distillation
  - real-time
date: 2026-05-08
content_hash: 691ab205fd0c8c4e
---

# Real-Time 3D Object Detection with Inference-Aligned Learning

**Conference**: AAAI 2026
**arXiv**: [2511.16140](https://arxiv.org/abs/2511.16140)
**Code**: [GitHub](https://github.com/zhaocy-ai/sr3d)
**Area**: 3D Vision
**Keywords**: 3D object detection, point cloud, indoor scene, optimal transport, label assignment, self-distillation, real-time

## TL;DR

This paper proposes SR3D, a framework that bridges the training-inference gap in indoor dense 3D object detection via two training-phase components: Spatial-Priority Optimal Transport Assignment (SPOTA) and Ranking-Aware adaptive Self-distillation (RAS). SR3D achieves state-of-the-art performance among dense detectors on ScanNet V2 and SUN RGB-D at a real-time speed of 42ms.

## Background & Motivation

Indoor point cloud 3D object detection is critical for real-time applications such as augmented reality, robotics, and navigation. Existing detectors follow two paradigms:

- **Sparse detectors** (VoteNet, 3DETR, V-DETR, etc.): achieve high localization accuracy by refining a small set of high-quality proposals, but incur large memory overhead and high latency (typically >130ms), making them unsuitable for real-time use.
- **Dense detectors** (GSDN, FCAF3D, TR3D, etc.): perform single-pass prediction by densely tiling anchors in the spatial domain, offering fast inference (~42ms) but substantially lower accuracy than sparse methods.

The authors identify the root cause of accuracy limitations in dense detectors as the **training-inference gap**, manifested in two missing properties:

1. **Lack of spatial reliability**: label assignment during training relies on fixed heuristics (e.g., center prior, IoU threshold), ignoring the actual spatial quality of anchors, which leads to misidentification of high-quality anchors in cluttered indoor scenes.
2. **Lack of ranking awareness**: training applies uniform supervision to all positive samples regardless of their relative localization quality, whereas the AP metric used at inference is inherently ranking-sensitive, causing inconsistency between classification confidence and localization accuracy.

### Case Study Validating the Bottleneck

The authors conduct an elegant oracle experiment: replacing predicted classification scores with ground-truth IoU scores boosts AP25 from 70.8 to 91.8 and AP50 from 55.6 to 87.7. This directly demonstrates that **the lack of ranking awareness is the primary bottleneck**, as the severe misalignment between classification confidence and localization quality substantially limits detection performance.

## Method

### Overall Architecture

SR3D adopts a classic dense detection architecture: sparse convolutional backbone (MinkResNet34) + FPN multi-scale feature fusion + dual-branch task head (classification + regression). The two core components, SPOTA and RAS, are used **only during training**, introducing zero additional overhead at inference and preserving real-time speed.

### 1. Spatial-Priority Optimal Transport Assignment (SPOTA)

Standard OTA formulates label assignment as an optimal transport problem, but direct application to 3D detection is problematic: (1) 3D detection relies more on geometric cues than semantic ones; (2) jointly optimizing classification and regression costs leads to multi-objective conflicts.

**Three key designs in SPOTA**:

**Normalized Vertex Distance**: IoU provides insufficient discriminability for predictions with similar overlap ratios but different geometric structures. SPOTA introduces the normalized vertex distance $\mathcal{R}_{VD}$ to capture fine-grained alignment differences at bounding box vertices:

$$\mathcal{R}_{VD} = \frac{d(\hat{\mathbf{v}}_1, \mathbf{v}_1) + d(\hat{\mathbf{v}}_2, \mathbf{v}_2)}{2\rho(\hat{\mathbf{b}}, \mathbf{b})}$$

where $d(\cdot)$ denotes Euclidean distance and $\rho(\hat{\mathbf{b}}, \mathbf{b})$ is the diagonal length of the minimum enclosing box. Unlike DIoU, which only considers center distance, vertex distance simultaneously captures scale and shape variation.

**Spatial-Priority Strategy**: The classification cost term is entirely removed; label assignment is driven solely by geometric cues. The rationale is that semantic information in 3D point clouds is inherently encoded in geometric structure (object shape, edges, layout), and retaining an explicit classification term introduces redundancy and biases the model toward semantic patterns rather than robust geometric alignment.

**Center Prior Constraint**: A Gaussian center prior $\gamma_c = 1 - \exp(-\mu d^2(\mathbf{c}, \mathbf{c}^{gt}))$ is introduced to stabilize optimization in the early stages of training.

The final cost matrix is:

$$C = \gamma_c \cdot (\mathcal{C}_{reg} + \mathcal{R}_{VD})$$

For each ground truth, the top-$k$ anchors with the lowest cost are selected as positives (default $k=6$, corresponding to the six principal directions in 3D Euclidean space).

### 2. Ranking-Aware Adaptive Self-distillation (RAS)

RAS injects localization quality and ranking information into the classification branch via a self-distillation mechanism, comprising two sub-components:

**Ranking-aware Distillation Loss (RDL)**: Soft targets are constructed from the localization quality (IoU) and soft-rank signals produced by the model's own regression branch, guiding the classification branch:

$$\mathbf{RDL}(\sigma) = (1 - r^{reg})^{\beta} q \log(\sigma) + q(1-q)\log(1-\sigma)$$

where $\sigma$ is the classification confidence, $q$ is the IoU, and $r^{reg}$ is the IoU-based soft rank (higher values indicate better localization). This formulation imposes heavier penalties on poorly localized samples, suppressing inconsistent predictions with high confidence but low localization quality.

**Soft Ranking Algorithm**: A differentiable soft-rank function $R_i = \frac{1}{N}\sum_{j \neq i}\sigma(\frac{s_j - s_i}{\tau})$ is used to compute continuous ranks, preserving pairwise distance information from the original scores and providing richer structural signals than hard ranking.

**Adaptive Weighting Strategy**: The contributions of standard classification loss (Focal Loss) and self-distillation loss are dynamically balanced according to the relative rank of classification scores:

$$\mathcal{L}_{cls} = \sum_{i \in \mathcal{P}} ((1 - r_i^{cls})\mathbf{FL}_i + r_i^{cls}\mathbf{RDL}_i) + \sum_{j \in \mathcal{N}} \mathbf{FL}_j$$

High-ranked (high-confidence) positive samples receive more self-distillation supervision to correct overconfidence, while low-ranked positives are primarily supervised by standard Focal Loss to preserve basic learning capacity.

### 3. Loss & Training

- **Backbone**: MinkResNet34 sparse convolution + generative sparse transposed convolution FPN
- **Optimizer**: AdamW, initial learning rate 1e-3, warmup 300 steps from 1e-5, weight decay 1e-4
- **Training schedule**: 13 epochs; learning rate decayed by 10× at epochs 8, 11, and 12
- **Voxel size**: 0.01m
- **Data augmentation**: random sampling of 66% points, horizontal flipping, rotation ±5°, scaling [0.6, 1.4]
- **Inference**: NMS (IoU threshold 0.5, confidence threshold 0.01)
- **Hardware**: single RTX 4090
- **Default hyperparameters**: $k=6$, $\mu=1$, $\beta=1$, $\tau=0.1$

## Key Experimental Results

### Main Results

Comparison on ScanNet V2 and SUN RGB-D indoor 3D detection benchmarks (parentheses denote averages over 25 evaluations):

| Method | Type | ScanNet AP25 | ScanNet AP50 | Latency | SUN AP25 | SUN AP50 | Latency |
|--------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| VoteNet | Sparse | 58.6 | 33.5 | 71ms | 57.7 | - | 41ms |
| 3DETR | Sparse | 65.0 | 47.0 | 170ms | 59.1 | 32.7 | - |
| CAGroup3D | Sparse | 75.1 (74.5) | 61.3 (60.3) | 472ms | 66.8 (66.4) | 50.2 (49.5) | - |
| V-DETR | Sparse | 77.4 (76.8) | 65.0 (64.5) | 240ms | 67.5 (66.8) | 50.4 (49.7) | - |
| DEST | Sparse | **78.5** (78.3) | **66.6** (66.2) | 263ms | **68.4** (67.4) | **51.8** (50.9) | - |
| GSDN | Dense | 62.8 | 34.8 | 49ms | - | - | - |
| FCAF3D | Dense | 71.5 (70.7) | 57.3 (56.0) | 64ms | 64.2 (63.8) | 48.9 (48.2) | 56ms |
| TR3D | Dense | 72.9 (72.0) | 59.3 (57.4) | 42ms | 67.1 (66.3) | 50.4 (49.6) | 36ms |
| TR3D+DLLA | Dense | 73.8 (72.8) | 60.2 (58.9) | - | 67.3 (67.0) | 50.6 (50.5) | - |
| **SR3D (Ours)** | **Dense** | **74.0** (73.2) | 59.7 (58.5) | **42ms** | **68.1** (67.2) | 50.9 (50.5) | **36ms** |

SR3D surpasses all prior dense detector state-of-the-art methods on all metrics. Compared to the TR3D baseline, it improves AP25 by 1.1/1.0 (ScanNet/SUN) with no increase in latency. SR3D achieves accuracy comparable to DLLA, while DLLA incurs higher computational cost due to its additional auxiliary branches.

### Ablation Study

| SPOTA | RAS | AP25 | AP50 | Latency |
|:---:|:---:|:---:|:---:|:---:|
| ✗ | ✗ | 70.8 | 55.6 | 42ms |
| ✓ | ✗ | 72.3 | 57.4 | 42ms |
| ✗ | ✓ | 72.5 | 57.7 | 42ms |
| **✓** | **✓** | **73.2** | **58.5** | **42ms** |

Both components are individually effective and complementary. The full model achieves +2.4 AP25 / +2.9 AP50 over the baseline with no change in latency.

### SPOTA Design Ablation

| Setting | AP25 | AP50 |
|---------|:---:|:---:|
| **SPOTA (full)** | **73.2** | **58.5** |
| + classification cost | 72.5 (−0.7) | 56.9 (−1.6) |
| − vertex distance | 72.7 (−0.5) | 57.8 (−0.7) |

Adding the classification cost leads to a significant performance drop, validating the spatial-priority strategy. Removing the vertex distance also causes notable degradation, confirming the importance of fine-grained geometric cues.

### RAS vs. Other Quality-Aware Losses

| Method | AP25 | AP50 |
|--------|:---:|:---:|
| QFL (Quality Focal Loss) | 71.9 | 57.7 |
| VFL (Varifocal Loss) | 71.7 | 58.3 |
| **RAS (Ours)** | **73.2** | **58.5** |

RAS substantially outperforms QFL (+1.3 AP25) and VFL (+1.5 AP25). The authors attribute this to the generally low IoU values in 3D detection: directly using IoU as classification supervision targets creates optimization conflicts, whereas RAS distills ranking signals rather than using raw IoU as labels, yielding more stable training.

### Training Cost Comparison

| Method | Training Time/Epoch | Parameters | AP25 | AP50 |
|--------|:---:|:---:|:---:|:---:|
| TR3D | 12.3 min | 14.7M | 72.0 | 57.4 |
| SR3D | 12.6 min | 14.7M | 73.2 | 58.5 |

Parameter counts are identical, training time increases by less than 3%, and inference overhead is zero — a highly cost-effective improvement.

## Highlights & Insights

1. **Precise problem formulation**: The oracle experiment (replacing classification scores with GT IoU boosts AP by 20+) clearly quantifies the severity of the training-inference gap, providing strong motivation for the proposed methods.
2. **Spatial priority over semantic priority**: Counterintuitively removing the classification cost entirely and relying solely on geometric cues for label assignment is logically well-grounded — in 3D point clouds, semantics are already encoded in geometry, making the classification term a redundant source of interference.
3. **Self-distillation with adaptive weighting**: Significant gains are achieved purely through training strategy improvements without introducing any additional modules or parameters, embodying the design philosophy of inference-aligned learning.
4. **Experimental rigor**: Each model is trained 5 times and evaluated 5 times (25 total evaluations), with comprehensive ablation studies covering all design choices, hyperparameters, and qualitative visualizations.

## Limitations & Future Work

1. **Limited to indoor scenes**: SPOTA and RAS are validated on indoor benchmarks (ScanNet V2, SUN RGB-D); transferability to large-scale outdoor scenes (e.g., nuScenes LiDAR data with extreme sparsity and diverse scale distributions) remains to be verified.
2. **No inference acceleration**: SR3D's contributions are concentrated in training strategy; model quantization, knowledge distillation, and lightweight design for inference acceleration are not explored.
3. **Ceiling of the dense paradigm**: Although SR3D substantially narrows the gap with sparse detectors (e.g., DEST), an absolute accuracy gap remains (AP25: 74.0 vs. 78.5), and it is unclear how close the inherent representational capacity of the dense paradigm is to its upper bound.
4. **Single-modal input**: Only point cloud coordinates and colors are used; the potential of multi-modal fusion (e.g., incorporating RGB images or text) is not explored.

## Related Work & Insights

- **Indoor 3D detection**: Sparse methods (VoteNet → CAGroup3D → V-DETR → DEST) lead in accuracy but are slow; dense methods (GSDN → FCAF3D → TR3D) are fast but accuracy is constrained by fixed label assignment strategies.
- **Dynamic label assignment**: FreeAnchor, OTA, SimOTA, AlignOTA, DLLA, and others improve label quality through dynamic matching but do not address the lack of ranking awareness.
- **Self-knowledge distillation**: Born-Again Networks, CS-KD, and related works leverage a model's own knowledge to guide training; SR3D's novelty lies in embedding ranking awareness into the distillation process.

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|:---:|-------|
| Novelty | 4 | The training-inference alignment perspective is novel; spatial-priority OTA and ranking-aware self-distillation are distinctive designs. |
| Technical Depth | 4 | Solid theoretical grounding (OT framework, soft ranking); the oracle experiment is elegant. |
| Experimental Thoroughness | 5 | Two datasets, 25 repeated evaluations, comprehensive ablation, hyperparameter analysis, and qualitative visualization. |
| Value | 4 | Zero-inference-overhead training strategy; plug-and-play, generalizable to other dense detectors. |
| Writing Quality | 4 | Problem motivation is clear and figures are intuitive, though the dense notation requires frequent cross-referencing. |
| **Overall** | **4.2** | A high-quality contribution with precise problem definition, elegant method design, and rigorous experimentation. |

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] YOLO-IOD: Towards Real Time Incremental Object Detection](yolo-iod_towards_real_time_incremental_object_detection.md)
- [\[AAAI 2026\] SAGA: Learning Signal-Aligned Distributions for Improved Text-to-Image Generation](saga_learning_signal-aligned_distributions_for_improved_text-to-image_generation.md)
- [\[AAAI 2026\] An Overall Real-Time Mechanism for Classification and Quality Evaluation of Rice](an_overall_real-time_mechanism_for_classification_and_quality_evaluation_of_rice.md)
- [\[ICCV 2025\] YOLOE: Real-Time Seeing Anything](../../ICCV2025/object_detection/yoloe_realtime_seeing_anything.md)
- [\[AAAI 2026\] REXO: Indoor Multi-View Radar Object Detection via 3D Bounding Box Diffusion](rexo_indoor_multi-view_radar_object_detection_via_3d_bounding_box_diffusion.md)

<!-- RELATED:END -->
