---
title: >-
  [Paper Note] LVFace: Progressive Cluster Optimization for Large Vision Models in Face Recognition
description: >-
  [ICCV 2025][Human Understanding][face recognition] This paper proposes LVFace, which addresses training instability of ViT in large-scale face recognition via a Progressive Cluster Optimization (PCO) strategy. The traini…
tags:
  - "ICCV 2025"
  - "Human Understanding"
  - "face recognition"
  - "Vision Transformer"
  - "progressive optimization"
  - "large vision model"
  - "margin-based loss"
date: 2026-05-08
content_hash: 0fafb13b405cac96
---

# LVFace: Progressive Cluster Optimization for Large Vision Models in Face Recognition

**Conference**: ICCV 2025
**arXiv**: [2501.13420](https://arxiv.org/abs/2501.13420)  
**Code**: [https://github.com/bytedance/LVFace](https://github.com/bytedance/LVFace)  
**Area**: Human Understanding
**Keywords**: face recognition, Vision Transformer, progressive optimization, large vision model, margin-based loss

## TL;DR

This paper proposes LVFace, which addresses training instability of ViT in large-scale face recognition via a Progressive Cluster Optimization (PCO) strategy. The training process is decomposed into three stages — feature alignment, centroid stabilization, and boundary refinement — achieving state-of-the-art results on multiple benchmarks.

## Background & Motivation

**State of the Field & Limitations of Prior Work**:
- ViT has replaced CNN as the dominant backbone in vision, yet face recognition remains largely CNN-centric.
- Directly applying CNN training paradigms (e.g., single-stage optimization with ArcFace/CosFace) to ViT leads to convergence instability and suboptimal performance.
- The root cause lies in ViT's lack of local inductive bias; the interaction between high-dimensional feature distributions and margin losses tends to destabilize cluster formation.

**Core Idea**:
- Inspired by the multi-stage training paradigm of LLMs/VLMs (pre-training → SFT → continual pre-training), this work decomposes the face recognition optimization into multiple stages, each with a clearly defined objective.
- The goal is to progressively achieve a compact and discriminative feature distribution.

## Method

### Overall Architecture

LVFace adopts a standard ViT as the backbone, paired with an MLP head (two 512-d FC layers + BN) for feature embedding extraction. The core innovation lies in the Progressive Cluster Optimization (PCO) training strategy and the Cosine Stage Scheduler (CSS).

### Key Designs

1. **Stage 1 — Feature Alignment**:

    - In large-scale datasets (millions of identities), positive samples are far outnumbered by negatives, making direct ViT training difficult to converge.
    - Negative Class Subsampling (NCS) is applied with ratio $r=0.1$: $S = \text{NCS}(C, r) = C \times r$
    - CosFace loss is used: $\mathcal{L}_a = \log(1 + \frac{\sum_{j\neq i}^S e^{s\cos\theta_j}}{e^{s(\cos\theta_i - m)}})$
    - **Design Motivation**: Reducing interference from a large number of hard negatives in early training accelerates basic feature alignment.

2. **Stage 2 — Centroid Stabilization**:

    - After feature alignment, hard positive samples may exhibit higher similarity to negative class centroids, misleading classifier updates.
    - A feature expectation $\boldsymbol{e}_i = \mathbb{E}(\boldsymbol{x}_i)$ is introduced as a statistical prototype for each class.
    - Adaptive update rule: $\boldsymbol{e}_i^{new} = \alpha_i \boldsymbol{e}_i^{old} + (1-\alpha_i)\boldsymbol{x}_i$, where $\alpha_i = \sigma(\cos\theta_i^e)$
    - The loss incorporates a feature expectation regularization term:

    $$\mathcal{L}_s = \log\left(1 + \frac{\sum_{j\neq i}^S e^{s\cos\theta_j}}{e^{s(\cos\theta_i - m_1)}} + \frac{\sum_{j\neq i}^S e^{s\cos\theta_j^e}}{e^{s(\cos\theta_i^e - m_2)}}\right)$$

    - **Design Motivation**: Feature expectations serve as anchors to stabilize cluster centers and prevent centroid drift caused by hard samples.

3. **Stage 3 — Boundary Refinement**:

    - NCS is disabled; all negative classes participate in training.
    - The increased number of unseen negative samples penalizes cluster boundaries, achieving intra-class compactness.
    - Since Stage 2 has already stabilized centroids, adding more negatives does not cause convergence issues.
    - The loss retains the same structure as Stage 2, but the summation range is expanded from $S$ to all $C$ classes.
    - **Design Motivation**: Stable centroids serve as anchors, and the full negative set is leveraged to tighten decision boundaries.

4. **Cosine Stage Scheduler (CSS)**:

    - Stage transitions are controlled by monitoring the mean squared cosine similarity within each batch: $s^{(t)} = \frac{1}{|\mathcal{B}^{(t)}|}\sum \|\frac{f_\theta(\mathcal{I}_i) \cdot \boldsymbol{w}_{y_i}^{(t)}}{\|f_\theta(\mathcal{I}_i)\|_2 \|\boldsymbol{w}_{y_i}^{(t)}\|_2}\|^2$
    - Thresholds: $\delta_1 = 0.2$ (→ Stage 2), $\delta_2 = 0.35$ (→ Stage 3)

### Loss & Training

- Optimizer: AdamW (lr=1e-3, $\beta_1=0.9$, $\beta_2=0.999$, weight decay=0.1) with polynomial decay.
- Progressive batch size: 384 for the first 60 epochs, 128 for the remaining 60 epochs.
- Feature scale $s=64$, angular margin $m=0.4$.
- Distributed training on 64 GPUs with AMP mixed precision.

## Key Experimental Results

### Main Results

**MFR-Ongoing Benchmark (trained on WebFace42M)**:

| Method | Backbone | Mask | Children | African | Caucasian | S-Asian | E-Asian | MR-All |
|--------|----------|------|----------|---------|-----------|---------|---------|--------|
| UniFace | R200 | 92.43 | 93.11 | 98.14 | 98.98 | 98.84 | 90.01 | 97.92 |
| TopoFR | R200 | 93.96 | 93.57 | 97.97 | 98.71 | 98.98 | 92.85 | 98.13 |
| Partial FC | ViT-L | 90.88 | - | 98.07 | 98.81 | 98.66 | 89.97 | 97.85 |
| **LVFace** | **ViT-L** | 93.56 | **94.31** | **98.79** | **99.26** | **99.26** | 91.02 | **98.49** |

**IJB-C and IJB-B Benchmarks (trained on Glint360K)**:

| Method | Backbone | IJB-C (1e-5) | IJB-C (1e-4) | IJB-B (1e-4) |
|--------|----------|-------------|-------------|-------------|
| ArcFace | R100 | 95.38 | 96.89 | 95.69 |
| TransFace-B | ViT-B | 96.18 | 97.45 | - |
| **LVFace-B** | **ViT-B** | **97.00** | **97.70** | **96.51** |
| TransFace-L | ViT-L | 96.29 | 97.61 | - |
| **LVFace-L** | **ViT-L** | **97.02** | **97.66** | **96.51** |

### Ablation Study

**Stepwise Gains from PCO Stages (MFR-Ongoing, LVFace-L)**:

| Method | Mask | Child | African | Caucasian | S-Asian | E-Asian | MR-All |
|--------|------|-------|---------|-----------|---------|---------|--------|
| ViT-L Baseline | 89.50 | 91.53 | 97.36 | 98.43 | 98.04 | 87.78 | 97.27 |
| +Stage 1 | 89.99 | 91.79 | 97.73 | 98.65 | 98.37 | 87.97 | 97.52 |
| +Stage 2 | 91.72 | 92.99 | 98.53 | 99.10 | 98.77 | 89.13 | 98.22 |
| +Stage 3 | **93.56** | **94.31** | **98.79** | **99.26** | **99.26** | **91.02** | **98.49** |

**Loss Function Compatibility (Glint360K, ViT-B)**:

| Method | IJB-C (1e-5) | IJB-C (1e-4) | IJB-B (1e-4) |
|--------|-------------|-------------|-------------|
| ArcFace | 96.11 | 97.12 | 96.01 |
| ArcFace+PCO | 96.68 | 97.44 | 96.40 |
| CosFace | 96.15 | 97.28 | 95.99 |
| **CosFace+PCO** | **97.00** | **97.70** | **96.51** |

**Scalability Across Model Sizes and Datasets**:

| Model | Training Set | IJB-C (1e-5) | IJB-C (1e-4) | IJB-B (1e-4) |
|-------|-------------|-------------|-------------|-------------|
| LVFace-T | G360K | 95.63 | 96.67 | 95.41 |
| LVFace-S | G360K | 96.52 | 97.31 | 96.14 |
| LVFace-B | G360K | 97.00 | 97.70 | 96.51 |
| LVFace-L | G360K | 97.02 | 97.66 | 96.51 |
| LVFace-L | W42M | **97.25** | **98.06** | **96.74** |

### Key Findings

- Each of the three PCO stages yields significant improvements, with MR-All rising from 97.27% to 98.49%.
- PCO is effective for both ArcFace and CosFace, with CosFace+PCO achieving the best results.
- Scaling from Tiny to Base yields consistent gains, but Base→Large improvements saturate on Glint360K.
- Training LVFace-L on the larger WebFace42M dataset yields substantial further gains, demonstrating data scalability.
- As of March 2025, LVFace ranks first on the MFR-Ongoing academic leaderboard.

## Highlights & Insights

1. **Borrowing from LLM multi-stage training**: The face recognition training process is decomposed into three progressive stages, each with a well-defined optimization objective.
2. **Effective use of NCS**: Negative subsampling in the first two stages accelerates convergence; full negatives are restored in Stage 3 to refine decision boundaries.
3. **Feature expectation anchoring**: EMA-updated feature expectations stabilize cluster centers, preventing hard samples from distorting centroids.
4. **Automated stage scheduling via CSS**: Stage transitions are determined automatically based on cosine similarity statistics, eliminating manual tuning.
5. **Native ViT compatibility**: The backbone is left unmodified, preserving compatibility with VLM/LLM architectures.

## Limitations & Future Work

- The CSS thresholds $\delta_1, \delta_2$ remain empirically set and may require adjustment for different datasets.
- Validation is limited to the standard closed-set training / open-set testing paradigm; broader application scenarios remain unexplored.
- The three-stage loss function design is relatively hand-crafted; more automated curriculum learning approaches could be considered.
- Large-scale training requires 64 GPUs, imposing considerable computational resource demands.

## Related Work & Insights

- Multi-stage training strategies are particularly important for ViT-based models that lack local inductive bias.
- The feature expectation regularization idea is generalizable to other tasks requiring stable cluster centers.
- This work demonstrates that, given sufficiently large datasets, ViT can comprehensively surpass CNN for face recognition.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The three-stage PCO strategy is well-motivated and draws insightful inspiration from LLM training paradigms.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive ablations across multiple benchmarks, backbones, and datasets, with MFR-Ongoing competition validation.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated, mathematical derivations are complete, and visualizations are informative.
- **Value**: ⭐⭐⭐⭐ Provides an effective training recipe for deploying ViT in face recognition, with open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Bi-Level Optimization for Self-Supervised AI-Generated Face Detection](bi-level_optimization_for_self-supervised_ai-generated_face_detection.md)
- [\[ICML 2026\] Efficient, Validation-Free Intrinsic Quality Estimation for Large-Scale Face Recognition Datasets](../../ICML2026/human_understanding/efficient_validation-free_intrinsic_quality_estimation_for_large-scale_face_reco.md)
- [\[ICCV 2025\] ImHead: A Large-scale Implicit Morphable Model for Localized Head Modeling](imhead_a_large-scale_implicit_morphable_model_for_localized_head_modeling.md)
- [\[ICCV 2025\] Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars](avat3r_large_animatable_gaussian_reconstruction_model_for_hi.md)
- [\[ICCV 2025\] High-Resolution Spatiotemporal Modeling with Global-Local State Space Models for Video-Based Human Pose Estimation](high-resolution_spatiotemporal_modeling_with_global-local_state_space_models_for.md)

</div>

<!-- RELATED:END -->
