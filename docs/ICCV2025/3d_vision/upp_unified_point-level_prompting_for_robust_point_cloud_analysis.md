---
title: >-
  [Paper Note] UPP: Unified Point-Level Prompting for Robust Point Cloud Analysis
description: >-
  [ICCV 2025][3D Vision][point cloud analysis] This paper proposes UPP, a unified point-level prompting framework that reformulates point cloud denoising and completion as prompting mechanisms for downstream tasks. It introduces a Rectification Prompter to filter noise, a Completion Prompter to recover missing regions, and a Shape-Aware Unit to capture geometry-sensitive features. With only 6.3% of the parameters, UPP surpasses full fine-tuning on noisy and incomplete point clouds.
tags:
  - ICCV 2025
  - 3D Vision
  - point cloud analysis
  - parameter-efficient fine-tuning
  - denoising
  - completion
  - prompt learning
date: 2026-05-08
content_hash: fbe797bbe14b84d3
---

# UPP: Unified Point-Level Prompting for Robust Point Cloud Analysis

**Conference**: ICCV 2025
**arXiv**: [2507.18997](https://arxiv.org/abs/2507.18997)
**Code**: [GitHub](https://github.com/zhoujiahuan1991/ICCV2025-UPP)
**Area**: 3D Vision
**Keywords**: point cloud analysis, parameter-efficient fine-tuning, denoising, completion, prompt learning

## TL;DR

This paper proposes UPP, a unified point-level prompting framework that reformulates point cloud denoising and completion as prompting mechanisms for downstream tasks. It introduces a Rectification Prompter to filter noise, a Completion Prompter to recover missing regions, and a Shape-Aware Unit to capture geometry-sensitive features. With only 6.3% of the parameters, UPP surpasses full fine-tuning on noisy and incomplete point clouds.

## Background & Motivation

Pre-trained point cloud models (Point-MAE, ReCon, etc.) have achieved remarkable progress on various downstream tasks. However, **real-world point clouds frequently suffer from noise and incompleteness** due to object occlusion, reflective surfaces, and sensor resolution limitations, which severely degrades model performance.

**Limitations of existing approaches**:

**Dedicated denoising/completion models + downstream tasks** (pipeline paradigm):
   - **Conflicting objectives** between denoising and completion: denoising removes excess points while completion adds missing ones, and naive integration leads to mutual interference
   - **Domain gap** between the enhancement tasks and the downstream task, resulting in suboptimal performance
   - Complex training pipelines with high computational and memory overhead

**Parameter-efficient fine-tuning (PEFT) methods** (IDPT, Point-PEFT, DAPT):
   - Improve representational capacity only in the latent feature space
   - **Neglect explicit suppression of noise and defects in the input point cloud**
   - Features become indistinguishable on low-quality data, leading to severe performance degradation

**UPP's innovation**: Denoising and completion are reformulated as downstream task-oriented prompting mechanisms, intervening in the **input data space** rather than only the feature space, with unified end-to-end training.

## Method

### Overall Architecture

The pre-trained backbone is frozen; three trainable components are inserted:
1. **Rectification Prompter**: Predicts corrective vector prompts after shallow blocks to filter noise
2. **Completion Prompter**: Generates completion point prompts after deep blocks to recover missing regions
3. **Shape-Aware Unit**: Inserted within each block to capture geometry-sensitive features

### Rectification Prompter

Given a noisy incomplete point cloud $\boldsymbol{x} \in \mathbb{R}^{S \times 3}$, encoded into $L$ tokens and processed through $d_r$ transformer blocks. Features are propagated from sparse center tokens to dense points via spatial interpolation:

$$\boldsymbol{f}_r = \mathcal{F}(\boldsymbol{h}_{d_r}, \boldsymbol{c}, \boldsymbol{x}) \in \mathbb{R}^{S \times D_r}$$

An MLP predicts a correction vector $\boldsymbol{v}_r \in \mathbb{R}^{S \times 3}$ for each point. Large-magnitude vectors correspond to low-confidence noisy points, which are filtered by threshold $\tau$:

$$\boldsymbol{x}_r = \{\boldsymbol{x} + \boldsymbol{v}_r \cdot \alpha \mid \tau > \|\boldsymbol{v}_r\|\}$$

**Training objective**: Noisy points are supervised with displacement vectors toward the clean surface; clean points are supervised with zero displacement:

$$\mathcal{L}_{\text{rect}} = \frac{1}{S_n}\sum_{i \in \boldsymbol{n}} \|\boldsymbol{v}_r^i - \boldsymbol{v}_{gt}^i\|^2 + \frac{1}{S}\sum_{i \in \boldsymbol{x}} \|\boldsymbol{v}_r^i\|^2$$

### Completion Prompter

The rectified point cloud $\boldsymbol{x}_r$ is re-sampled and re-encoded; after $d_c$ blocks, tokens are down-projected and concatenated into a global feature $\boldsymbol{f}_c$ to predict coarse centers $\boldsymbol{c}_m$ for missing regions.

Key design: **reuse of the MAE pre-trained decoder** for local patch reconstruction:

$$\boldsymbol{x}_m = \mathcal{D}([\boldsymbol{h}_m + \text{Embed}(\boldsymbol{c}_m), \boldsymbol{h}_{d_c}])$$

The final output merges rectified and completed points via FPS re-sampling: $\boldsymbol{x}_c = \text{FPS}([\boldsymbol{x}_m, \boldsymbol{x}_r])$

**Loss function** (L1 Chamfer Distance):

$$\mathcal{L}_{\text{comp}} = \mathcal{C}_1(\boldsymbol{c}_m, \mathcal{P}_m) + \mathcal{C}_1(\boldsymbol{x}_m, \mathcal{P}_m) + \mathcal{C}_1(\boldsymbol{x}_c, \mathcal{P}_{gt})$$

### Shape-Aware Unit

Inserted within each transformer block, comprising two innovations:

1. **Shape-Aware Attention**: Establishes connections based on **spatial distance** rather than feature similarity; noisy outliers are unlikely to alter spatial neighborhood relations, yielding greater robustness
2. **Low-rank adapter**: $\boldsymbol{h}_{i+1} = W_2 \cdot \sigma(W_1(\hat{\boldsymbol{h}}_i)) + \hat{\boldsymbol{h}}_i$, preventing feature over-smoothing

### Total Loss

$$\mathcal{L} = \mathcal{L}_{\text{rect}} + \mathcal{L}_{\text{comp}} + \mathcal{L}_{\text{task}}$$

A staged optimization strategy is adopted to improve training stability.

## Key Experimental Results

### Noisy Point Cloud Classification (Main Results)

| Method | Reference | Params (M)↓ | Noisy ModelNet40↑ | Noisy ShapeNet55↑ |
|:---|:---:|:---:|:---:|:---:|
| Point-MAE (FFT) | ECCV22 | 22.1 (100%) | 89.42 | 88.13 |
| +Point-PEFT | AAAI24 | 0.7 (3.2%) | 87.52 (−1.90) | 86.01 (−2.12) |
| +DAPT | CVPR24 | 1.1 (5.0%) | 86.43 (−2.99) | 86.33 (−1.80) |
| **+UPP (Ours)** | — | **1.4 (6.3%)** | **92.95 (+3.53)** | **90.40 (+2.27)** |
| ReCon (FFT) | ICML23 | 43.6 (100%) | 89.67 | 89.01 |
| **+UPP (Ours)** | — | **1.4 (3.2%)** | **91.69 (+2.02)** | **89.68 (+0.67)** |
| Point-FEMAE (FFT) | AAAI24 | 27.4 (100%) | 89.59 | 88.63 |
| **+UPP (Ours)** | — | **1.4 (5.1%)** | **91.94 (+2.35)** | **90.08 (+1.45)** |

UPP surpasses full fine-tuning across all three backbones while using only 3.2%–6.3% of the parameters. Existing PEFT methods consistently degrade performance.

### Real-World Data (ScanObjectNN)

| Method | Params (M) | Acc. (%) |
|:---|:---:|:---:|
| Point-FEMAE (baseline) | 27.4 | 90.71 |
| +Point-PEFT | 0.7 | 89.16 |
| +DAPT | 1.1 | 89.67 |
| **+UPP (Ours)** | **1.4** | **91.39** |

### Ablation Study

| Base | Rect. Prompter | Compl. Prompter | SA-Unit | Acc. (%) |
|:---:|:---:|:---:|:---:|:---:|
| ✓ | ✗ | ✗ | ✗ | 89.42 |
| ✓ | ✓ | ✗ | ✗ | 90.90 |
| ✓ | ✗ | ✓ | ✗ | 91.36 |
| ✓ | ✗ | ✗ | ✓ | 91.28 |
| ✓ | ✓ | ✓ | ✓ | **92.95** |

Each component individually contributes 1.5–2 percentage points; their combination achieves the best performance.

### Key Findings

1. **Adverse effect of PEFT methods**: Existing 3D PEFT methods (Point-PEFT, DAPT) actually hurt performance on noisy data, as they ignore explicit handling of input noise
2. **Importance of input-space intervention**: UPP performs rectification and completion in the data space rather than solely in the feature space, yielding more direct and effective improvements
3. **Robustness of Shape-Aware Attention**: Spatial-distance-based attention is more resistant to noise interference than feature-similarity-based attention
4. **Backbone agnosticism**: UPP generalizes effectively across Point-MAE, ReCon, and Point-FEMAE

## Highlights & Insights

1. **Paradigm shift**: Denoising and completion are transformed from standalone pre-processing steps into unified prompts for downstream tasks, eliminating domain gaps and objective conflicts
2. **Data-space prompting**: Unlike VPT and similar methods that add prompt tokens solely in the feature space, UPP operates directly in the point coordinate space (displacing or adding discrete points)
3. **Reuse of pre-trained decoder**: The MAE decoder weights, which are typically discarded after pre-training, are cleverly repurposed for point cloud completion

## Limitations & Future Work

1. The staged optimization strategy increases training complexity
2. The number of completion points $M$ is a fixed hyperparameter, limiting adaptability to varying degrees of incompleteness
3. Validation is limited to classification; effectiveness on segmentation and detection tasks remains to be confirmed

## Related Work & Insights

- **Point cloud pre-training**: Point-MAE, ReCon, Point-FEMAE, PointGPT
- **Point cloud enhancement**: ScoreDenoise, PoinTr, T-CorresNet
- **3D PEFT**: IDPT, Point-PEFT, DAPT, GAPrompt

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Paradigm innovation unifying denoising and completion as prompting mechanisms
- Technical Depth: ⭐⭐⭐⭐ — Elegant three-component design; Shape-Aware Attention is supported by theoretical analysis
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple backbones, multiple datasets, comprehensive ablations
- Value: ⭐⭐⭐⭐ — Parameter-efficient, open-source, directly improves robustness of existing models

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](efficient_spiking_point_mamba_for_point_cloud_analysis.md)
- [\[ICCV 2025\] TurboReg: TurboClique for Robust and Efficient Point Cloud Registration](turboreg_turboclique_for_robust_and_efficient_point_cloud_registration.md)
- [\[ICCV 2025\] UST-SSM: Unified Spatio-Temporal State Space Models for Point Cloud Video Modeling](ust-ssm_unified_spatio-temporal_state_space_models_for_point_cloud_video_modelin.md)
- [\[ICCV 2025\] Noise2Score3D: Tweedie's Approach for Unsupervised Point Cloud Denoising](noise2score3d_tweedies_approach_for_unsupervised_point_cloud_denoising.md)
- [\[ICCV 2025\] Revisiting Point Cloud Completion: Are We Ready For The Real-World?](revisiting_point_cloud_completion_are_we_ready_for_the_real-world.md)

</div>

<!-- RELATED:END -->
