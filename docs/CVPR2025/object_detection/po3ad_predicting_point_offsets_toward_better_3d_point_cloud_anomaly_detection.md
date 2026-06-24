---
title: >-
  [Paper Note] PO3AD: Predicting Point Offsets toward Better 3D Point Cloud Anomaly Detection
description: >-
  [CVPR 2025][Object Detection][Point Cloud Anomaly Detection] PO3AD proposes to learn normal point cloud representations by predicting the offset vectors of pseudo-anomaly points (rather than reconstructing the entire point cloud), thereby focusing the model's attention on anomalous regions. Combined with a normal-guided pseudo-anomaly generation method (Norm-AS), it improves the detection AUC-ROC by 9.0% and 1.4% on Anomaly-ShapeNet and Real3D-AD, respectively…
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "Point Cloud Anomaly Detection"
  - "Offset Prediction"
  - "Normal-Guided"
  - "Pseudo-Anomaly Generation"
  - "Anomaly-Free Training"
date: 2026-05-08
content_hash: c6e1a2ac38371fde
---

# PO3AD: Predicting Point Offsets toward Better 3D Point Cloud Anomaly Detection

**Conference**: CVPR 2025  
**arXiv**: [2412.12617](https://arxiv.org/abs/2412.12617)  
**Code**: None  
**Area**: 3D Vision / Anomaly Detection  
**Keywords**: Point Cloud Anomaly Detection, Offset Prediction, Normal-Guided, Pseudo-Anomaly Generation, Anomaly-Free Training

## TL;DR

PO3AD proposes to learn normal point cloud representations by predicting the offset vectors of pseudo-anomaly points (rather than reconstructing the entire point cloud), thereby focusing the model's attention on anomalous regions. Combined with a normal-guided pseudo-anomaly generation method (Norm-AS), it improves the detection AUC-ROC by 9.0% and 1.4% on Anomaly-ShapeNet and Real3D-AD, respectively, compared to existing methods.

## Background & Motivation

**Background**: 3D point cloud anomaly detection under the anomaly-free training setting requires learning representations that are sufficient to identify deviations using only normal samples. Mainstream approaches include memory-bank-based (PatchCore) and reconstruction-based (IMRNet, R3D-AD) methods.

**Limitations of Prior Work**: Reconstruction methods restore normal samples from their pseudo-anomalous versions, but they suffer from a fundamental issue—the reconstruction loss assigns equal weights to both normal and pseudo-anomalous points, which prevents the model from focusing its attention on the anomalous deviation regions that truly need to be learned. In addition, the disorder and sparsity of 3D point cloud data exacerbate the difficulty of feature learning.

**Key Challenge**: The reconstruction task requires the model to accurately recover the coordinates of each point, whereas the core of anomaly detection lies in identifying deviations—these two goals are misaligned. The equally distributed reconstruction loss dilutes the model's focus on anomalous regions.

**Goal**: To design a learning paradigm that naturally focuses the model's attention on anomalous regions, replacing the traditional reconstruction task.

**Key Insight**: Predicting the offset vectors of points (the displacement from pseudo-anomalous points to their corresponding normal points) is more targeted than reconstructing full coordinates. The offset of normal points is zero (only the magnitude needs to be predicted), while the offset of anomalous points requires predicting both magnitude and direction, naturally guiding the model to focus on anomalous regions.

**Core Idea**: Transform anomaly detection from "reconstructing normal point clouds" into "predicting the offset vector of each point from its pseudo-anomalous position to its normal position." This inherently focuses the model's attention on pseudo-anomalous points, while using normal-guided pseudo-anomaly generation to produce more realistic anomaly samples.

## Method

### Overall Architecture

Given a normal point cloud $P$, pseudo-anomalies $\hat{P}$ are generated via Norm-AS, with the ground-truth offset label defined as $O^{gt} = \hat{P} - P$. Then, $\hat{P}$ is fed into MinkUNet to extract features, which are then passed through an MLP offset predictor to output the predicted offset $O^{pre}$ for each point. The system is trained under the constraint of an offset loss. During inference, the magnitude of the predicted offset is directly used as the anomaly score.

### Key Designs

1. **Point Offset Prediction (Point Offset Prediction)**:

    - **Function**: Focuses model attention on pseudo-anomalous regions to efficiently learn normal data representations.
    - **Mechanism**: The learning goal is shifted from "reconstructing the coordinates of $P$" to "predicting $O = \hat{P} - P$." The offset of normal points is a zero vector (the direction is meaningless and the magnitude is zero), so the model only needs to learn to output zero. Conversely, the offset of anomalous points possesses both magnitude and direction, requiring the model to learn both simultaneously. This asymmetry naturally causes the model's gradients to originate primarily from anomalous points. The offset loss is defined as $\mathcal{L}_{off} = \mathcal{L}_{dist} + \mathcal{L}_{dir}$, where $\mathcal{L}_{dist}$ is the L1 distance loss and $\mathcal{L}_{dir}$ is the negative cosine similarity loss for direction.
    - **Design Motivation**: Experiments demonstrate that reducing the reconstruction loss weight of normal points significantly improves detection performance (see Figure 1), while offset prediction pushes this to the extreme—the effective loss for normal points is zero.

2. **Normal-Guided Pseudo-Anomaly Generation (Norm-AS)**:

    - **Function**: Generates pseudo-anomaly samples that are closer to real anomalies.
    - **Mechanism**: The point cloud is divided into $J$ patches, and one patch is randomly selected as the anomalous region. Points are shifted along their normal vectors to simulate protrusion/intrusion defects: $\hat{ph}_b = ph_b + \alpha \cdot nv_b \cdot (1-w) \cdot \beta$. Here, $\alpha \in \{-1, 1\}$ controls the direction of displacement, $w$ is the normalized distance weight (displacement is maximized at the center and decays toward the boundaries), and $\beta \sim U[0.06, 0.12]$ is the displacement distance. The normal vector $nv_b$ ensures that the points shift along the surface normal.
    - **Design Motivation**: Without normal vector guidance, points may shift in arbitrary directions, causing anomalous regions to overlap with normal regions and confusing the model's learning. Normal-guided displacement produces more realistic protrusion/intrusion effects.

3. **MinkUNet-Based Offset Prediction Network**:

    - **Function**: Regresses point-wise offset vectors from point cloud features.
    - **Mechanism**: MinkUNet (sparse convolutional U-Net) is adopted as the backbone, which is adept at capturing fine-grained local features. After voxelizing the point cloud, voxel-level features $G^V$ are extracted and then converted to point-level features $G^P \in \mathbb{R}^{N \times C}$ using voxel-to-point indices. An MLP offset predictor regresses $O^{pre} = f_O(G^P) \in \mathbb{R}^{N \times 3}$. During inference, the L2 norm of the predicted offset serves as the anomaly score.
    - **Design Motivation**: The local sparse convolution characteristics of MinkUNet make it highly suitable for capturing fine local deviations in point clouds.

### Loss & Training

The offset loss is defined as $\mathcal{L}_{off} = \mathcal{L}_{dist} + \mathcal{L}_{dir}$, where the distance loss employs the L1 norm and the direction loss employs the negative cosine similarity. Note that the direction loss is only meaningful for pseudo-anomaly points with non-zero offsets. Training is conducted using only normal samples combined with the pseudo-anomalies generated by Norm-AS. During inference, the original test point cloud is input, and the magnitude of the predicted offset is directly utilized as the anomaly indicator.

## Key Experimental Results

### Main Results — Anomaly-ShapeNet

| Method | Detection AUC-ROC ↑ | Localization AUC-ROC ↑ |
|------|-------------|-------------|
| BTF (CVPR23) | 56.8 | 64.2 |
| IMRNet | 62.3 | 68.7 |
| R3D-AD | 67.5 | 73.1 |
| PO3AD | **76.5** | **79.8** |

### Real3D-AD

| Method | Detection AUC-ROC ↑ | Localization AUC-ROC ↑ |
|------|-------------|-------------|
| Reg3D-AD | 72.4 | 68.9 |
| R3D-AD | 74.1 | 71.5 |
| PO3AD | **75.5** | **73.2** |

### Ablation Study

| Configuration | Anomaly-ShapeNet AUC ↑ |
|------|---------|
| Only $\mathcal{L}_{dist}$ | 73.2 |
| Only $\mathcal{L}_{dir}$ | 68.5 |
| $\mathcal{L}_{dist} + \mathcal{L}_{dir}$ | **76.5** |
| Without normal vector guidance (Random direction) | 71.8 |
| With normal vector guidance (Norm-AS) | **76.5** |

### Key Findings

- Achieves a **9.0%** AUC-ROC gain on Anomaly-ShapeNet (67.5% $\rightarrow$ 76.5%), representing a significant performance leap.
- Attention visualization clearly confirms that the offset prediction model concentrates its attention on pseudo-anomalous regions, whereas the reconstruction model's attention is scattered across the entire point cloud.
- Normal-guided pseudo-anomaly generation contributes a 4.7% performance improvement (71.8% $\rightarrow$ 76.5%), demonstrating that the quality of pseudo-anomalies is crucial for training.
- Both distance and direction losses are indispensable, yielding the best performance when utilized jointly.

## Highlights & Insights

- **The paradigm shift from "reconstruction" to "offset prediction"** is simple yet highly effective—by altering the learning objective, it naturally resolves the issue of uneven attention distribution.
- **Normal-guided pseudo-anomaly generation** is closer to real defects (protrusions/intrusions) than random-direction shifts, representing a rational utilization of the physical characteristics of 3D point clouds.
- **Directly using the offset magnitude as the anomaly score during inference** is more natural than reconstruction methods, which require manually designed comparison metrics.

## Limitations & Future Work

- Only two anomaly types (protrusion and intrusion) are validated, and generalization to other anomaly types such as missing parts and deformation remains to be verified.
- The number of patches $J$ and the displacement range $\beta$ in Norm-AS need to be manually tuned.
- The voxelization process of MinkUNet may lose some fine geometric information.
- It has not been fully validated on larger-scale industrial inspection datasets.

## Related Work & Insights

- **vs R3D-AD**: R3D-AD utilizes a reconstruction task where the reconstruction loss is evenly distributed; PO3AD's offset prediction naturally focuses on anomalous areas.
- **vs IMRNet**: IMRNet detects anomalies via masked reconstruction, which may miss anomalies in unmasked regions; PO3AD is free from this limitation.
- **Insights for 2D Anomaly Detection**: The concept of offset prediction could potentially be transferred to 2D image anomaly detection.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of replacing reconstruction with offset prediction is simple and effective, and the normal-guided pseudo-anomalies possess practical value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two benchmark datasets, detailed ablation experiments, and attention visualization.
- Writing Quality: ⭐⭐⭐⭐ Clear problem motivation; the attention comparison in Figure 1 is highly convincing.
- Value: ⭐⭐⭐⭐ Provides a new learning paradigm for 3D anomaly detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Back to Point: Exploring Point-Language Models for Zero-Shot 3D Anomaly Detection](../../CVPR2026/object_detection/back_to_point_exploring_point-language_models_for_zero-shot_3d_anomaly_detection.md)
- [\[ECCV 2024\] TAPTR: Tracking Any Point with Transformers as Detection](../../ECCV2024/object_detection/taptr_tracking_any_point_with_transformers_as_detection.md)
- [\[CVPR 2026\] Detect Anything via Next Point Prediction](../../CVPR2026/object_detection/detect_anything_via_next_point_prediction.md)
- [\[CVPR 2025\] AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP](aa-clip_enhancing_zero-shot_anomaly_detection_via_anomaly-aware_clip.md)
- [\[CVPR 2025\] Odd-One-Out: Anomaly Detection by Comparing with Neighbors](odd-one-out_anomaly_detection_by_comparing_with_neighbors.md)

</div>

<!-- RELATED:END -->
