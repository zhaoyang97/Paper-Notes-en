---
title: >-
  [Paper Note] CASL: Curvature-Augmented Self-supervised Learning for 3D Anomaly Detection
description: >-
  [AAAI2026][3D anomaly detection] This work identifies point cloud curvature as a powerful cue for anomaly detection and proposes CASL…
tags:
  - "AAAI2026"
  - "3D anomaly detection"
  - "self-supervised learning"
  - "curvature"
  - "point cloud"
  - "U-Net"
date: 2026-05-08
content_hash: 5750ba6f413f039f
---

# CASL: Curvature-Augmented Self-supervised Learning for 3D Anomaly Detection

**Conference**: AAAI2026
**arXiv**: [2511.12909](https://arxiv.org/abs/2511.12909)  
**Code**: [GitHub](https://github.com/zyh16143998882/CASL)  
**Area**: Others
**Keywords**: 3D anomaly detection, self-supervised learning, curvature, point cloud, U-Net

## TL;DR

This work identifies point cloud curvature as a powerful cue for anomaly detection and proposes CASL, a curvature-augmented self-supervised learning framework. By guiding coordinate reconstruction with multi-scale curvature prompts, CASL learns generalizable 3D representations without any anomaly-detection-specific mechanisms, achieving a 5.6% O-AUROC improvement over the previous state of the art on Real3D-AD.

## Background & Motivation

Deep learning-based 3D anomaly detection holds significant value for industrial quality control. Existing methods fall broadly into two categories:

- **Feature matching methods** (PatchCore, Reg3D-AD, Group3AD): extract features from pretrained models to build memory banks and detect anomalies via feature distance at inference time.
- **Reconstruction methods** (IMRNet, R3D-AD): learn to reconstruct normal samples and detect anomalies via reconstruction error.

Both categories are **specifically designed** for anomaly detection, limiting their generalizability. In contrast, self-supervised point cloud models (Point-MAE, PointGPT, etc.) pursue general representation learning under a unified pretrain–finetune paradigm, yet the authors find empirically that these models perform poorly on anomaly detection tasks.

The authors attribute this to a **geometric shortcut** problem: existing self-supervised methods perform reconstruction directly in coordinate space, where the semantic and positional domains completely overlap, causing learned representations to over-rely on low-level spatial features and leading to representation collapse in fine-grained anomaly detection.

## Core Problem

**How to build a 3D representation learning framework that performs well on anomaly detection while remaining generalizable?**

A key finding motivates the work: a non-learned method that uses per-point curvature directly as an anomaly score already outperforms several classical self-supervised models and dedicated anomaly detection methods. This reveals the central role of curvature in 3D anomaly detection—curvature at the boundaries of anomalous regions is significantly higher than in normal regions. As an intrinsic geometric property orthogonal to coordinate space, curvature can effectively alleviate the geometric shortcut problem.

## Method

### 1. Curvature Computation

For each point $x_i$ in the point cloud, a local covariance matrix is constructed via k-nearest neighbors and decomposed into eigenvalues. Curvature is defined as:

$$\text{Curv}(x_i) = \frac{\lambda_1^i + \lambda_2^i + \lambda_3^i}{\lambda_1^i}$$

where $\lambda_1^i \leq \lambda_2^i \leq \lambda_3^i$ are the sorted eigenvalues of the covariance matrix. This metric measures the degree of local surface geometric variation.

### 2. Curvature-Augmented Self-supervised Learning Framework

The overall architecture is based on U-Net and consists of three components:

**Curvature Encoder**: An MLP embedding layer extracts per-point curvature features, followed by three encoding blocks that capture multi-scale curvature representations. Each encoding block consists of a stride-2 Minkowski convolution for downsampling and a 4-layer residual convolution block.

**Coordinate Encoder**: Takes randomly initialized features for $N$ masked points as input and progressively maps them to high-dimensional features ($N_4 \times 256$) through four encoding blocks.

**Fusion Decoder**: Upsamples features from the previous resolution via transposed convolution, concatenates them with curvature prompts at the current resolution, processes them through convolution blocks, and progressively restores the original resolution. The final output is an $N \times 96$ tensor, which is dimensionality-reduced by an MLP, concatenated with the original curvature features, and mapped back to 3D coordinate space.

### 3. Key Design: Full Coordinate Masking + Multi-scale Curvature Prompts

Unlike conventional methods that mask a subset of coordinates and reconstruct from the remaining ones, CASL masks **all** point coordinates and relies entirely on curvature prompts for reconstruction. This forces the network to learn rich geometric representations solely from curvature information, fundamentally eliminating the geometric shortcut.

Curvature provides complementary information at different scales: fine scales are sensitive to local surface variations (edges, protrusions), while coarse scales capture overall shape contours.

### 4. Loss Function

Given the large scale of point clouds (typically exceeding 100,000 points), Chamfer Distance and EMD are computationally infeasible. An $\ell_1 + \ell_2$ loss is adopted:

$$\mathcal{L}_{recon} = \mathcal{L}_1(p, p_{rec}) + \mathcal{L}_2(p, p_{rec})$$

### 5. Pseudo-Anomaly Classification Fine-tuning

Pseudo-anomalies are synthesized on normal samples by randomly selecting patches and displacing them along the normal direction to simulate bumps or depressions. Only a binary classification head is appended after the pretrained backbone. At inference, the log-ratio of normal/anomaly softmax probabilities serves as the per-point anomaly score, which is aggregated via top-k pooling to produce a sample-level anomaly score.

## Key Experimental Results

### Real3D-AD Dataset

| Method | O-AUROC | P-AUROC |
|--------|---------|---------|
| PatchCore | 0.682 | 0.692 |
| Reg3D-AD | 0.704 | 0.700 |
| Group3AD | 0.751 | 0.735 |
| PO3AD | 0.765 | - |
| ISMP | 0.767 | 0.836 |
| Curvature (non-learned) | 0.723 | 0.729 |
| **CASL** | **0.823** | **0.882** |

CASL surpasses the second-best method by 5.6% in O-AUROC and 4.6% in P-AUROC.

### Anomaly-ShapeNet Dataset

| Method | O-AUROC | P-AUROC |
|--------|---------|---------|
| PO3AD | 0.839 | 0.898 |
| **CASL** | **0.887** | **0.899** |

CASL outperforms PO3AD by 4.8% in mean O-AUROC across 40 categories.

### ScanObjectNN Classification

Pretrained on only 832 samples, CASL achieves leading performance on OBJ-BG (92.08%) and OBJ-ONLY (91.05%).

## Highlights & Insights

- **Curvature as anomaly score**: The finding that a non-learned curvature method outperforms multiple dedicated detection models provides compelling motivation.
- **Eliminating the geometric shortcut**: Full coordinate masking combined with curvature prompts fundamentally decouples the semantic and positional domains.
- **Strong generalizability**: The unified pretrain–finetune paradigm enables the same model to perform effectively on anomaly detection, classification, and segmentation.
- **Data efficiency**: Competitive representations are obtained with pretraining on only 832 samples.

## Limitations & Future Work

- Pretraining data is drawn solely from normal samples of two anomaly detection datasets, limiting data diversity; scaling up pretraining data may further improve performance.
- Curvature is sensitive to noise and may produce noisy estimates on low-quality point clouds, degrading detection performance.
- The pseudo-anomaly generation strategy is relatively simple (only normal-direction displacement), potentially failing to cover all real anomaly types.
- The method does not achieve optimal results on the ScanObjectNN PB-T50-RS variant, indicating room for improvement under extreme occlusion and transformation settings.
- The approach lacks the ability to handle texture/color anomalies; purely geometric methods may be limited in scenarios requiring appearance information.

## Related Work & Insights

- **vs. Point-MAE/PointGPT and similar self-supervised methods**: These methods perform poorly when fine-tuned for anomaly detection due to representation collapse caused by geometric shortcuts; CASL addresses this via curvature prompts.
- **vs. PatchCore/Reg3D-AD and similar feature matching methods**: Although they also rely on pretrained models, they depend on anomaly-detection-specific feature matching mechanisms, inconsistent with the unified fine-tuning paradigm.
- **vs. IMRNet/R3D-AD and similar reconstruction methods**: Designed specifically for anomaly detection with weak generalizability; CASL transfers the same model to classification and segmentation tasks.
- **vs. PO3AD**: PO3AD also employs pseudo-anomaly strategies but does not address the geometric shortcut problem; CASL substantially outperforms it on both benchmarks.

The idea of using curvature as a geometric prior to guide reconstruction can be generalized to other 3D tasks (e.g., 3D completion, deformation detection). The full-masking + cross-domain prompt reconstruction strategy may also extend to other modalities (e.g., reconstructing spatial-domain signals from spectral prompts). The pattern of a non-learned baseline outperforming complex models is worth examining in other tasks, as it may reveal overlooked but critical priors.

## Rating
- Novelty: 8/10 — The discovery that curvature alone serves as an anomaly score and the full-masking reconstruction design are both novel.
- Experimental Thoroughness: 8/10 — Multi-task validation across anomaly detection, classification, and segmentation, with complete ablation studies.
- Writing Quality: 8/10 — Motivation is clearly articulated; the logical chain from observation to method is concise.
- Value: 8/10 — Offers a new perspective on general 3D representation learning with significant anomaly detection improvements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Semi-supervised Graph Anomaly Detection via Robust Homophily Learning](../../NeurIPS2025/others/semi-supervised_graph_anomaly_detection_via_robust_homophily_learning.md)
- [\[AAAI 2026\] RcAE: Recursive Reconstruction Framework for Unsupervised Industrial Anomaly Detection](rcae_recursive_reconstruction_framework_for_unsupervised_industrial_anomaly_dete.md)
- [\[ICML 2026\] Towards Optimal Robustness in Learning-Augmented Paging](../../ICML2026/others/towards_optimal_robustness_in_learning-augmented_paging.md)
- [\[NeurIPS 2025\] ADPretrain: Advancing Industrial Anomaly Detection via Anomaly Representation Pretraining](../../NeurIPS2025/others/adpretrain_advancing_industrial_anomaly_detection_via_anomaly_representation_pre.md)
- [\[ICML 2026\] Parsimonious Learning-Augmented Online Metric Matching](../../ICML2026/others/parsimonious_learning-augmented_online_metric_matching.md)

</div>

<!-- RELATED:END -->
