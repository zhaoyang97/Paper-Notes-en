---
title: >-
  [Paper Note] Real-Time 3D Object Detection with Inference-Aligned Learning
description: >-
  [AAAI2026][3D Vision][3D Object Detection] This paper proposes the SR3D framework, which bridges the gap between training and inference behaviors in indoor dense 3D object detection through two training-phase components: Spatial-Prioritized Optimal Transport Assignment (SPOTA) and Rank-Aware Adaptive Self-Distillation (RAS). It refreshes the state-of-the-art (SOTA) for dense detectors on ScanNet V2 and SUN RGB-D while maintaining a real-time inference speed of 42ms.
tags:
  - "AAAI2026"
  - "3D Vision"
  - "3D Object Detection"
  - "Point Cloud"
  - "indoor scene"
  - "Optimal Transport"
  - "label assignment"
  - "self-distillation"
  - "real-time"
date: 2026-05-08
content_hash: b772c9f014c84e3b
---

# Real-Time 3D Object Detection with Inference-Aligned Learning

**Conference**: AAAI2026  
**arXiv**: [2511.16140](https://arxiv.org/abs/2511.16140)  
**Code**: [GitHub](https://github.com/zhaocy-ai/sr3d)  
**Area**: 3D Vision  
**Keywords**: 3D Object Detection, Point Cloud, indoor scene, Optimal Transport, label assignment, self-distillation, real-time

## TL;DR

This paper proposes the SR3D framework, which bridges the gap between training and inference behaviors in indoor dense 3D object detection through two training-phase components: Spatial-Prioritized Optimal Transport Assignment (SPOTA) and Rank-Aware Adaptive Self-Distillation (RAS). It refreshes the state-of-the-art (SOTA) for dense detectors on ScanNet V2 and SUN RGB-D while maintaining a real-time inference speed of 42ms.

## Background & Motivation

Indoor point cloud 3D object detection is crucial for real-time applications such as augmented reality, robotics, and navigation. Existing detectors are divided into two main paradigms:

- **Sparse Detectors** (VoteNet, 3DETR, V-DETR, etc.): Localize objects by refining a small number of high-quality candidate proposals. They achieve high accuracy but suffer from large memory overhead and high latency (typically >130ms), making them unsuitable for real-time scenarios.
- **Dense Detectors** (GSDN, FCAF3D, TR3D, etc.): Densely place anchors in the spatial domain to perform single-pass predictions. They are fast (~42ms) but achieve significantly lower accuracy than sparse methods.

The authors find that the fundamental bottleneck limiting the accuracy of dense detectors is the **training-inference gap**, which manifests as two major deficiencies:

1. **Lack of Spatial Reliability**: Label assignment during training relies on fixed heuristics (e.g., center prior, IoU thresholds) and ignores the actual spatial quality of the anchors, easily misjudging high-quality anchors in cluttered indoor scenes.
2. **Lack of Rank Awareness**: Training imposes uniform supervision on all positive samples, disregarding the relative ranking of their localization accuracy; however, the AP evaluation metric used during inference is inherently rank-sensitive. This leads to inconsistency between classification confidence and localization accuracy.

### Case Study Verifying Bottleneck

The authors conduct an elegant oracle experiment: after replacing the classification scores predicted by the baseline model with ground-truth (GT) IoU scores, AP25 surges from 70.8 to 91.8, and AP50 surges from 55.6 to 87.7. This directly demonstrates that **the lack of rank awareness is the primary bottleneck limiting model performance**. The severe misalignment between classification confidence and localization quality tremendously constrains detection performance.

## Method

### Overall Architecture

SR3D adopts a classic dense detection architecture: a sparse convolutional backbone (MinkResNet34) + FPN multi-scale feature fusion + dual-branch task head (classification + regression). The two core innovative components, SPOTA and RAS, **are only utilized during the training phase**, incurring zero extra overhead at inference time to maintain real-time speed.

### 1. Spatial-Prioritized Optimal Transport Assignment (SPOTA)

Standard OTA models label assignment as an optimal transport problem, but its direct application to 3D detection is problematic: (1) 3D detection relies more heavily on geometric information rather than semantic cues; (2) simultaneously optimizing classification and regression costs leads to multi-objective conflict.

**Key Designs of SPOTA**:

**Normalized Vertex Distance (NVD)**: IoU exhibits poor discriminative power for bounding boxes with large geometric structural differences but similar overlap rates. SPOTA introduces the Normalized Vertex Distance $\mathcal{R}_{VD}$ to capture fine-grained alignment differences of bounding box vertices:

$$\mathcal{R}_{VD} = \frac{d(\hat{\mathbf{v}}_1, \mathbf{v}_1) + d(\hat{\mathbf{v}}_2, \mathbf{v}_2)}{2\rho(\hat{\mathbf{b}}, \mathbf{b})}$$

where $d(\cdot)$ is the Euclidean distance, and $\rho(\hat{\mathbf{b}}, \mathbf{b})$ is the diagonal length of the minimum enclosing box. Unlike DIoU, which only considers center distance, the vertex distance simultaneously perceives scales and shape variations.

**Spatial-Prioritized Strategy**: The classification cost term is completely removed, driving label assignment solely by geometric cues. The rationale is that in 3D point clouds, semantic cues are inherently encoded in the geometric structures (object shapes, edges, layout). Explicitly retaining the classification term introduces redundancy and biases the assignment toward semantic patterns rather than robust geometric alignment.

**Center Prior Constraint**: A Gaussian center prior $\gamma_c = 1 - \exp(-\mu d^2(\mathbf{c}, \mathbf{c}^{gt}))$ is introduced to help stabilize optimization in the early stages of training.

The final cost matrix is:

$$C = \gamma_c \cdot (\mathcal{C}_{reg} + \mathcal{R}_{VD})$$

For each ground truth, the top-k anchors with the lowest cost are selected as positive samples (default $k=6$, corresponding to the six principal directions of 3D Euclidean space).

### 2. Rank-Aware Adaptive Self-Distillation (RAS)

RAS injects localization quality and ranking information into the training of the classification branch through a self-distillation mechanism, comprising two sub-components:

**Rank-aware Distillation Loss (RDL)**: Utilizing the localization accuracy (IoU) and soft ranking information generated by the model's own regression branch, RDL constructs soft targets to guide the classification branch:

$$\mathbf{RDL}(\sigma) = (1 - r^{reg})^{\beta} q \log(\sigma) + q(1-q)\log(1-\sigma)$$

where $\sigma$ is the classification confidence, $q$ is the IoU, and $r^{reg}$ is the soft rank based on IoU (higher values signify better localization). This formula imposes heavier penalties on poorly localized samples, suppressing inconsistent predictions characterized by "high confidence but low localization accuracy".

**Soft Ranking Algorithm**: A differentiable soft ranking function $R_i = \frac{1}{N}\sum_{j \neq i}\sigma(\frac{s_j - s_i}{\tau})$ is employed to compute continuous rankings. This preserves the pairwise distance information of the raw scores, offering richer structural cues than hard ranking.

**Adaptive Weighting Strategy**: Adjusts the contribution ratio between the standard classification loss (Focal Loss) and the self-distillation loss dynamically, based on the relative ranking of classification scores:

$$\mathcal{L}_{cls} = \sum_{i \in \mathcal{P}} ((1 - r_i^{cls})\mathbf{FL}_i + r_i^{cls}\mathbf{RDL}_i) + \sum_{j \in \mathcal{N}} \mathbf{FL}_j$$

Positive samples with high ranks (high confidence) receive stronger self-distillation supervision (to correct overconfidence), whereas positive samples with low ranks are mainly supervised by the standard Focal Loss (to maintain basic learning ability).

### Loss & Training

- **Backbone**: MinkResNet34 sparse convolution + generative sparse transposed convolution FPN
- **Optimizer**: AdamW, initial learning rate of 1e-3, warming up from 1e-5 over 300 steps, weight decay of 1e-4
- **Training Budget**: 13 epochs, with learning rate decayed by 10x at the 8th, 11th, and 12th epochs
- **Voxel Size**: 0.01m
- **Data Augmentation**: Randomly sampling 66% of points, horizontal flipping, rotation within $\pm 5^\circ$, scaling in $[0.6, 1.4]$
- **Inference**: NMS (IoU threshold of 0.5, confidence threshold of 0.01)
- **Hardware**: Single RTX 4090 GPU
- **Default Hyperparameters**: $k=6$, $\mu=1$, $\beta=1$, $\tau=0.1$

## Key Experimental Results

### Main Results

Comparison on ScanNet V2 and SUN RGB-D, two indoor 3D detection benchmarks (numbers in parentheses indicate the average of 25 evaluations):

| Method | Type | ScanNet AP25 | ScanNet AP50 | Latency | SUN AP25 | SUN AP50 | Latency |
|------|------|:---:|:---:|:---:|:---:|:---:|:---:|
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

SR3D outperforms previous dense SOTA detectors across all metrics. Compared to the TR3D baseline, it achieves AP25 improvements of 1.1/1.0 on ScanNet and SUN respectively, without introducing any extra latency. It shows comparable accuracy to DLLA, while DLLA incurs higher computational overhead due to its auxiliary branches.

### Ablation Study

| SPOTA | RAS | AP25 | AP50 | Latency |
|:---:|:---:|:---:|:---:|:---:|
| ✗ | ✗ | 70.8 | 55.6 | 42ms |
| ✓ | ✗ | 72.3 | 57.4 | 42ms |
| ✗ | ✓ | 72.5 | 57.7 | 42ms |
| **✓** | **✓** | **73.2** | **58.5** | **42ms** |

Both components are independently effective and complementary. The full model improves upon the baseline by +2.4 AP25 / +2.9 AP50, while maintaining the same latency.

### SPOTA Design Ablation

| Setting | AP25 | AP50 |
|------|:---:|:---:|
| **SPOTA (Full)** | **73.2** | **58.5** |
| Add Classification Cost Term | 72.5 (-0.7) | 56.9 (-1.6) |
| Remove Vertex Distance | 72.7 (-0.5) | 57.8 (-0.7) |

Adding the classification cost leads to a significant performance drop, justifying the spatial-prioritized strategy. Removing the vertex distance also results in a notable decline, demonstrating the importance of fine-grained geometric cues.

### Comparison of RAS with Other Quality-Aware Losses

| Method | AP25 | AP50 |
|------|:---:|:---:|
| QFL (Quality Focal Loss) | 71.9 | 57.7 |
| VFL (Varifocal Loss) | 71.7 | 58.3 |
| **RAS (Ours)** | **73.2** | **58.5** |

RAS significantly outperforms QFL (+1.3 AP25) and VFL (+1.5 AP25). The authors suggest that this is because IoU values are generally low in 3D detection, and directly using IoU to supervise classification causes optimization conflicts. RAS stabilizes training by distilling ranking signals rather than using IoU directly as labels.

### Comparison of Training Overhead

| Method | Training Time/epoch | Parameters | AP25 | AP50 |
|------|:---:|:---:|:---:|:---:|
| TR3D | 12.3 min | 14.7M | 72.0 | 57.4 |
| SR3D | 12.6 min | 14.7M | 73.2 | 58.5 |

The parameters are identical, and training time increases by less than 3% with zero inference overhead—demonstrating exceptional cost-effectiveness.

## Highlights & Insights

1. **Precise Problem Definition**: Through the oracle experiment (where AP surged by 20+ points when classification scores were replaced with GT IoU), the severity of the training-inference gap is cleanly quantified, providing strong motivation for the design.
2. **Spatial-Prioritized Over Semantic-Prioritized**: Counter-intuitively, the classification cost term is completely removed from label assignment in favor of geometric cues. The underlying logic is robust: in 3D point clouds, semantic information is already encoded in the geometry, rendering the classification cost term redundant or even distracting.
3. **Self-Distillation + Adaptive Weighting**: Significant gains are obtained solely by improving training strategies without introducing any extra modules or parameters, embodying the philosophy of "inference-aligned learning".
4. **Experimental Rigor**: Each model is trained 5 times and tested 5 times (25 evaluations in total) to report the best and average performances. The ablation studies thoroughly cover various design choices and hyperparameters.

## Limitations & Future Work

1. **Indoor-Centric**: The effectiveness of SPOTA and RAS has been verified on indoor benchmarks (ScanNet V2, SUN RGB-D). Whether they generalize to large-scale outdoor scenarios (such as nuScenes LiDAR data, which features extreme sparsity and diverse scale distributions) remains to be validated.
2. **No Focus on Inference Acceleration**: The innovations of SR3D are concentrated on training strategies. Inference acceleration techniques like model quantization, distillation, or lightweight architectures are not explored.
3. **Ceiling of the Dense Detector Paradigm**: Although the gap with sparse detectors (e.g., DEST) is narrowed significantly, there remains an absolute accuracy gap (AP25: 74.0 vs 78.5). Whether the inherent expressive capacity limit of the dense paradigm is being approached warrants further exploration.
4. **Single Modality Input**: The model relies solely on point cloud coordinates and colors, leaving the potential of multi-modal fusion (e.g., integrating RGB images and text) unexplored.

## Related Work & Insights

- **Indoor 3D Detection**: Sparse methods (VoteNet -> CAGroup3D -> V-DETR -> DEST) lead in accuracy but are slow. Dense methods (GSDN -> FCAF3D -> TR3D) are fast but accuracy-limited by fixed label assignment strategies.
- **Dynamic Label Assignment**: Methods like FreeAnchor, OTA, SimOTA, AlignOTA, and DLLA improve label quality through dynamic matching, but do not address the lack of rank-awareness.
- **Self-Knowledge Distillation**: Approaches like Born-Again Networks and CS-KD guide training using a model's intrinsic knowledge. SR3D's novelty lies in embedding rank-awareness directly into the distillation process.

## Rating

| Dimension | Score (1-5) | Explanation |
|------|:---:|------|
| Novelty | 4 | The training-inference alignment perspective is novel; the spatial-prioritized OTA and rank-aware self-distillation are uniquely designed. |
| Technical Depth | 4 | Solid theoretical analysis (OT framework, soft ranking), and an elegant oracle experiment. |
| Experimental Thoroughness | 5 | Evaluated across two datasets with 25 repeated evaluations, comprehensive ablations, hyperparameter analyses, and qualitative visualizations. |
| Value | 4 | A plug-and-play training strategy with zero inference overhead, easily generalizable to other dense detectors. |
| Writing Quality | 4 | Well-motivated with clean charts and diagrams, though some equations require careful cross-checking of notation. |
| **Overall Score** | **4.2** | High-quality work with a precise problem definition, elegant method design, and rigorous experimentation. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Learning Class Prototypes for Unified Sparse-Supervised 3D Object Detection](../../CVPR2025/3d_vision/learning_class_prototypes_for_unified_sparse-supervised_3d_object_detection.md)
- [\[AAAI 2026\] Multi-Modal Assistance for Unsupervised Domain Adaptation on Point Cloud 3D Object Detection](multi-modal_assistance_for_unsupervised_domain_adaptation_on_point_cloud_3d_obje.md)
- [\[CVPR 2026\] Changes in Real Time: Online Scene Change Detection with Multi-View Fusion](../../CVPR2026/3d_vision/changes_in_real_time_online_scene_change_detection_with_multi-view_fusion.md)
- [\[AAAI 2026\] MonoCLUE: Object-Aware Clustering Enhances Monocular 3D Object Detection](monoclue_object-aware_clustering_enhances_monocular_3d_object_detection.md)
- [\[CVPR 2026\] Geometry-Aligned and Anomaly-Aware Reconstruction for 3D Anomaly Detection](../../CVPR2026/3d_vision/geometry-aligned_and_anomaly-aware_reconstruction_for_3d_anomaly_detection.md)

</div>

<!-- RELATED:END -->
