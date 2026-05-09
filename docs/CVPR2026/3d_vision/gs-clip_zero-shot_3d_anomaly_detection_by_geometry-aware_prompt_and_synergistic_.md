---
title: >-
  [Paper Note] GS-CLIP: Zero-shot 3D Anomaly Detection by Geometry-Aware Prompt and Synergistic View Representation Learning
description: >-
  [CVPR 2026][3D Vision][Zero-shot 3D anomaly detection] This paper proposes GS-CLIP, a two-stage framework that injects global shape context and local defect information from 3D point clouds into text prompts via a Geometry Defect Distillation Module (GDDM), and employs a dual-stream LoRA architecture to synergistically fuse rendered images and depth maps, achieving state-of-the-art zero-shot 3D anomaly detection on four large-scale benchmarks.
tags:
  - CVPR 2026
  - 3D Vision
  - Zero-shot 3D anomaly detection
  - CLIP
  - geometry-aware prompt
  - multi-view fusion
  - point cloud
date: 2026-05-08
content_hash: 5bda863634628529
---

# GS-CLIP: Zero-shot 3D Anomaly Detection by Geometry-Aware Prompt and Synergistic View Representation Learning

**Conference**: CVPR 2026
**arXiv**: [2602.19206](https://arxiv.org/abs/2602.19206)
**Code**: [GitHub](https://github.com/zhushengxinyue/GS-CLIP)
**Area**: 3D Vision
**Keywords**: Zero-shot 3D anomaly detection, CLIP, geometry-aware prompt, multi-view fusion, point cloud

## TL;DR

This paper proposes GS-CLIP, a two-stage framework that injects global shape context and local defect information from 3D point clouds into text prompts via a Geometry Defect Distillation Module (GDDM), and employs a dual-stream LoRA architecture to synergistically fuse rendered images and depth maps, achieving state-of-the-art zero-shot 3D anomaly detection on four large-scale benchmarks.

## Background & Motivation

**State of the Field**: 3D anomaly detection is critical in industrial manufacturing. Traditional unsupervised methods (3D-ST, Reg3D-AD) require extensive normal samples from the target category for training, whereas zero-shot 3D anomaly detection (ZS3DAD) aims to train a generalizable model on auxiliary data and directly transfer it to unseen categories—addressing data privacy and sample scarcity concerns.

**Limitations of Prior Work**:
   - **Loss of 3D geometric information**: Existing methods (PointAD, MVP-PCLIP) project 3D point clouds into 2D images for CLIP processing, compressing volumetric structures into planar pixels. The model thus learns a "2D visual proxy" of geometric anomalies rather than their true physical form, causing detection failures when geometric anomalies are visually inconspicuous from certain viewpoints.
   - **Insufficient exploitation of visual information**: Existing methods rely on a single 2D representation. Rendered images are texture-rich but susceptible to lighting/rendering artifacts; depth maps capture overall geometric structure but fail to resolve subtle depth variations (e.g., slight protrusions). Single-modality representations limit detection coverage and generalization.

**Root Cause**: While CLIP's powerful zero-shot generalization has been validated in 2D anomaly detection, extending it to the 3D domain faces two critical gaps: projection-induced information loss and insufficient single-modality visual representation.

**Starting Point**: Rather than adapting only the visual branch, the paper addresses both the text branch and the visual branch simultaneously—injecting 3D geometric priors as anomaly cues on the text side, and fusing complementary information from rendered images and depth maps on the visual side.

## Method

### Overall Architecture

A two-stage learning strategy:
- **Stage 1 (Text Branch)**: Visual components are frozen; a geometry-aware text prompt generator is trained to extract global shape context and local defect information from 3D point clouds and dynamically generate text prompts embedded with geometric priors.
- **Stage 2 (Visual Branch)**: The prompt generator trained in Stage 1 is frozen; a dual-stream visual architecture is trained—rendered images pass through a frozen ViT, depth maps pass through a LoRA-finetuned ViT, and features from both streams are deeply fused via a Synergistic Refinement Module (SRM).

### Key Designs

1. **Geometry Defect Distillation Module (GDDM)**

   Core Idea: Anomalies are fundamentally deviations from normal patterns. A learnable normal prototype memory bank $\mathcal{P} = \{p_1, ..., p_l\} \in \mathbb{R}^{l \times d_{pn}}$ is designed to implicitly fit the distribution of normal local geometric features during training.

   For each point's local feature $f_i$, a geometric outlier score is computed:
    $s_i = 1 - \max_{p_j \in \mathcal{P}} \frac{f_i \cdot p_j}{\|f_i\| \|p_j\|}$

   The top-$k$ point features with the highest scores are selected and aggregated via self-attention, then projected into defect prompts $t_d \in \mathbb{R}^{k \times d}$.

   **Design Motivation**: By distilling directly from 3D geometric features which local structures are most likely anomalous, the model acquires genuine 3D geometric defect perception rather than relying solely on 2D visual cues.

2. **Shape Prompt + Asymmetric Prompt Concatenation**

   A pretrained PointNet++ extracts global point cloud features $F_e$, projected into shape prompts $t_s = \text{Proj}(F_e)$. Normal and abnormal prompts adopt **asymmetric concatenation**:

    $t_N = \text{Concat}(t_s, t_l), \quad t_A = \text{Concat}(t_s, t_l, t_d)$

   The abnormal prompt carries an additional defect descriptor $t_d$ beyond the normal prompt, establishing a clear semantic distinction. These prompts are encoded by the frozen text encoder to produce $T_N, T_A$, which are compared against visual features for classification and segmentation.

3. **Synergistic View Representation Learning (Depth-LoRA + SRM)**

    - **Depth-LoRA**: CLIP naturally adapts to real images, so rendered images pass through a frozen ViT; depth maps exhibit a domain gap relative to natural images, so LoRA is applied only to the MLP layers of the ViT (preserving pretrained spatial relationship modeling):
    $x' = \text{GELU}(W_1 x + \gamma B_1 A_1 x)$

    - **Synergistic Refinement Module (SRM)**: Receives global and local features from both streams and generates a shared affinity matrix $S = f_1(K_i^R) \times f_2(K_i^D)^T$ via bidirectional multiplicative attention, then separately aggregates value vectors from each stream before concatenation and fusion:
    $G_i = \text{MLP}(\text{Concat}(E_i^R, E_i^D))$

   **Design Motivation**: Rendered images excel at capturing appearance anomalies such as scratches, while depth maps excel at detecting geometric anomalies such as pits and protrusions. Dual-stream fusion of complementary information yields more comprehensive coverage than any single stream.

### Loss & Training

- **Stage 1**: $L_{stage1} = L_{cla} + L_{seg}$ (binary cross-entropy + Dice/Focal segmentation loss)
- **Stage 2**: $L_{stage2} = L_{cla} + L_{seg} + \alpha L_{con}$, with an additional **cross-view consistency loss**:
  $$L_{con} = 1 - \frac{1}{v}\sum_{i=1}^v \langle G_i, \bar{G} \rangle$$
  This encourages the model to learn viewpoint-invariant global representations, enhancing generalization.
- Stage 1: 15 epochs, lr=0.002; Stage 2: 10 epochs, lr=0.0005
- 3D-to-2D projection uses 9 viewpoints; CLIP backbone is ViT-L/14@336px.

## Key Experimental Results

### Main Results

| Dataset | Metric | GS-CLIP | PointAD (Prev. SOTA) | Gain |
|--------|------|------|----------|------|
| MVTec3D-AD | O-AUROC / P-PRO | 83.6 / 86.4 | 82.0 / 84.4 | +1.6 / +2.0 |
| Eyecandies | O-AUROC / P-PRO | 71.5 / 73.8 | 69.1 / 71.3 | +2.4 / +2.5 |
| Real3D-AD | O-AUROC | 76.4 | 74.8 | +1.6 |
| Anomaly-ShapeNet | O-AUROC / P-AUROC | 84.1 / 75.2 | 82.6 / 74.1 | +1.5 / +1.1 |
| Cross-dataset (Eyecandies) | O-AUROC / P-AUROC | 70.3 / 92.9 | 69.5 / 91.8 | +0.8 / +1.1 |

### Ablation Study

| Configuration | O-AUROC, O-AP | P-AUROC, P-PRO | Notes |
|------|---------|------|------|
| Rendered image only + learnable prompt | 80.9, 91.7 | 93.5, 83.1 | Baseline |
| + SRM dual-stream fusion | 82.3, 93.9 | 94.6, 84.8 | Large gain from dual-stream fusion |
| + Shape Prompt | 82.5, 94.8 | 95.2, 85.1 | Macro geometric context aids classification |
| + Defect Prompt | 82.9, 94.4 | 95.6, 85.6 | Defect prompt significantly improves localization |
| + Both prompts | 83.1, 96.2 | 96.0, 86.2 | Complementary effect is evident |
| + $L_{con}$ | **83.6, 96.5** | **96.3, 86.4** | Cross-view consistency further improves performance |

### Key Findings

- In GDDM, $k=12$ outlier points is optimal; larger $k$ introduces noise from normal points; prototype count $l=32$ reaches saturation.
- Performance plateaus at 9 viewpoints; additional viewpoints yield diminishing returns.
- Incorporating RGB images (multimodal fusion) raises O-AUROC on MVTec3D-AD to 88.2%, further validating the framework's extensibility.
- Inference overhead: 0.51s/image, 1.96 FPS, 5872 MB memory—slightly higher than the baseline but with significantly superior accuracy.
- Performance degradation under cross-dataset settings is minimal, demonstrating strong generalization.

## Highlights & Insights

- **An information bridge from 3D to text**: Rather than simply projecting 3D data to 2D for CLIP to process, the paper injects 3D geometric information into the text branch as priors, enabling the text prompts to "know what kind of anomaly to look for."
- **Asymmetric prompt design**: Normal and abnormal prompts share shape context, but the abnormal prompt additionally carries defect descriptors, yielding clear semantic differentiation.
- **Two-stage decoupling**: The text branch is optimized first to learn geometric anomaly descriptions, followed by visual branch optimization for alignment, avoiding the instability of joint training.
- **Plug-and-play multimodal extensibility**: The framework naturally accommodates additional modalities such as RGB images.

## Limitations & Future Work

- PointNet++ as the 3D feature extractor may limit expressiveness for complex geometry; stronger 3D backbones (e.g., Point Transformer v3) warrant exploration.
- The current fixed 9-viewpoint multi-view projection strategy could be replaced by adaptive viewpoint selection.
- Inference speed of approximately 2 FPS may be insufficient for real-time industrial inspection; feature caching or model compression could be explored.
- More direct 3D-native representations (e.g., anomaly detection directly on point clouds without 2D projection) remain unexplored.

## Related Work & Insights

- **PointAD (NeurIPS'24)**: Constructs 3D representations from rendered images and interprets anomalies from both point and pixel perspectives; the most direct predecessor to this work.
- **MVP-PCLIP**: Fine-tunes CLIP with depth maps and visual/text prompts, but relies on a single visual modality.
- **AnomalyCLIP / AA-CLIP**: Prompt learning methods for 2D zero-shot anomaly detection; this paper extends their paradigm to the 3D domain with injected geometric priors.
- **Insight**: Injecting domain-specific prior knowledge into the text branch of foundation models—rather than merely adapting the visual branch—is an effective and generalizable paradigm.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of geometry defect distillation, asymmetric prompts, and dual-stream fusion is novel; injecting 3D priors via the text branch is a distinctive perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets, two evaluation settings, detailed ablations, multimodal extension, and parameter sensitivity analysis.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated; the complementarity of rendered images vs. depth maps is illustrated with intuitive figures.
- Value: ⭐⭐⭐⭐ ZS3DAD is an emerging and practically relevant direction; the gains are substantial and cross-dataset generalization is strong.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Back to Point: Exploring Point-Language Models for Zero-Shot 3D Anomaly Detection](back_to_point_exploring_point-language_models_for_zero-shot_3d_anomaly_detection.md)
- [\[CVPR 2026\] VirPro: Visual-referred Probabilistic Prompt Learning for Weakly-Supervised Monocular 3D Detection](virpro_visual-referred_probabilistic_prompt_learning_for_weakly-supervised_monoc.md)
- [\[ICCV 2025\] CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting](../../ICCV2025/3d_vision/clip-gs_unifying_vision-language_representation_with_3d_gaussian_splatting.md)
- [\[CVPR 2026\] A Semantically Disentangled Unified Model for Multi-category 3D Anomaly Detection](a_semantically_disentangled_unified_model_for_multi-category_3d_anomaly_detectio.md)
- [\[CVPR 2026\] SwiftTailor: Efficient 3D Garment Generation with Geometry Image Representation](swifttailor_efficient_3d_garment_generation_with_geometry_image_representation.md)

<!-- RELATED:END -->
