---
title: >-
  [Paper Note] Improving 2D Feature Representations by 3D-Aware Fine-Tuning
description: >-
  [ECCV 2024][3D Vision][Representation learning] Through lifting 2D foundation model features into 3D Gaussian representations for multi-view fusion, followed by backward fine-tuning of the 2D model using rendered 3D-aware features, semantic segmentation and depth estimation performance are improved merely via linear probing.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Representation learning"
  - "3D Gaussian Splatting"
  - "Foundation model fine-tuning"
  - "Feature distillation"
  - "Multi-view consistency"
date: 2026-05-08
content_hash: 2ecdbde550c6cfc8
---

# Improving 2D Feature Representations by 3D-Aware Fine-Tuning

**Conference**: ECCV 2024  
**arXiv**: [2407.20229](https://arxiv.org/abs/2407.20229)  
**Code**: [https://ywyue.github.io/FiT3D](https://ywyue.github.io/FiT3D) (has project page)  
**Area**: 3D Vision  
**Keywords**: Representation learning, 3D Gaussian Splatting, Foundation model fine-tuning, Feature distillation, Multi-view consistency

## TL;DR

Through lifting 2D foundation model features into 3D Gaussian representations for multi-view fusion, followed by backward fine-tuning of the 2D model using rendered 3D-aware features, semantic segmentation and depth estimation performance are improved merely via linear probing.

## Background & Motivation

**Background**: Vision foundation models (DINOv2, CLIP, MAE, etc.) trained purely on unstructured 2D images have demonstrated powerful generic feature extraction capabilities and are widely used in downstream tasks such as segmentation, depth estimation, and correspondence matching.

**Limitations of Prior Work**: The training data for these models consists of scattered 2D images without multi-view or video corresponding relations, leading to a lack of 3D understanding in the models—features generated for the same object from different views are inconsistent, and they perform poorly in texture-less regions or on fine-grained structures.

**Key Challenge**: Images, as simple projections of the 3D world, discard explicit 3D geometric information. Purely 2D-trained models cannot leverage beneficial properties of the 3D world such as multi-view consistency and multi-view complementarity to resolve single-view ambiguity.

**Goal**: How to inject 3D understanding capabilities into existing 2D foundation models in a lightweight manner without pre-training from scratch.

**Key Insight**: Utilize 3D Gaussian Splatting to fuse multi-view 2D features into 3D representations to obtain multi-view consistent "3D-aware features", then use these features as supervision signals to fine-tune the original 2D model.

**Core Idea**: Use 3D Gaussian representation as an intermediate bridge to distill multi-view fused 3D-aware features back to the 2D foundation model, significantly improving downstream task performance with just one epoch of fine-tuning.

## Method

### Overall Architecture

Two-stage pipeline: **Stage One**—train 3D Gaussian representations with extra feature vectors on $K$ different scenes to lift 2D foundation model features into 3D; **Stage Two**—use rendered 3D-aware features as ground truth (GT) to fine-tune the original 2D foundation model. During downstream evaluation, the original features and fine-tuned features are concatenated, and semantic segmentation or depth estimation is performed via simple linear probing.

### Key Designs

1. **3D Feature Gaussians**: On top of standard 3D Gaussian Splatting, an additional low-dimensional feature vector $\mathbf{f} \in \mathbb{R}^D$ ($D=64$, much smaller than DINOv2's 384 dimensions) is appended to each Gaussian. Features are rasterized into 2D feature maps through $\alpha$-blending:

$$\mathbf{F}^{\text{low}} = \sum_{i \in \mathcal{N}} \mathbf{f}_i \alpha_i \prod_{j=1}^{i-1}(1-\alpha_i)$$

Then, a scene-specific CNN decoder (a single $3 \times 3$ convolution layer) up-projects the low-dimensional features to the high-dimensional space $d: \mathbf{F}^{\text{low}} \mapsto \mathbf{F}^{\text{high}}$.

Design Motivation: Directly storing 384-dimensional features in millions of Gaussians leads to prohibitive memory consumption. Low-dimensional features + CNN up-projection achieve a balance between efficiency and quality.

**Optimization**: Jointly optimize Gaussian parameters and features:

$$\hat{\mathcal{G}} = \arg\min \sum_{i=1}^{N} \mathcal{L}^c(r^{\text{rgb}}(\mathcal{G}, \mathbf{P}_i), \mathbf{I}_i) + \mathcal{L}^f(d(r^{\text{feat}}(\mathcal{G}, \mathbf{P}_i)), \mathbf{F}_i)$$

Key design: Feature vectors $\mathbf{f}$ only receive gradients from the feature loss $\mathcal{L}^f$, while other parameters (position, covariance, opacity) only receive gradients from the RGB loss $\mathcal{L}^c$. This ensures that the 3D geometry is supervised by RGB (multi-view consistent), while features are learned on the correct geometry—it is this separation that allows the inconsistencies of 2D features to be corrected by forcing a 3D consistent representation.

2. **3D-Aware Fine-Tuning**: After pre-training Feature Gaussians for $K$ scenes, all Gaussian representations are preloaded into CPU memory. In each training step: randomly sample a viewpoint $\rightarrow$ retrieve the corresponding Feature Gaussian and CNN decoder $\rightarrow$ render 3D-aware features $\mathbf{F}^{\text{high}}$ $\rightarrow$ fine-tune the 2D feature extractor using $l_1$ loss:

$$\mathcal{L} = \|\varepsilon_\theta^{2D}(\mathbf{I}_i) - \mathbf{F}^{\text{high}}\|_1$$

Design Motivation: Training pairs are generated online during runtime (rendering), avoiding the need to save a large number of feature maps. Only 1 epoch is required to transfer 3D-aware capabilities, with a small learning rate ($1\text{e-}5$), without introducing extra network components. Training on ScanNet++ with 230 scenes and 140k viewpoints takes only 8.5 hours on a single A100.

3. **Feature Assembly Strategy**: During downstream evaluation, the original DINOv2 features and the fine-tuned features are **concatenated** (instead of added or linearly fused). This is key to maintaining the generalization ability of the original model while introducing 3D awareness.

Design Motivation: Fine-tuned features contain 3D awareness, but may lose some of the original generalization capability; the concatenation strategy allows the linear probe head to dynamically decide the weight distribution.

### Loss & Training

- Stage one: 30k iterations of training for each scene; $\mathcal{L}^c = l_1 + $ D-SSIM, $\mathcal{L}^f = l_1$
- Stage two: batch size=2, lr=1e-5, AdamW(weight_decay=1e-4), horizontal flip augmentation, 1 epoch
- Linear probing: 40k iterations (8 GPUs) for semantic segmentation, 38,400 iterations (8 GPUs) for depth estimation

## Key Experimental Results

### Main Results

**Indoor Semantic Segmentation** (mIoU↑):

| Method | ScanNet++ | NYUv2 | ScanNet |
|------|-----------|-------|---------|
| DINOv2 | 30.19 | 65.55 | 43.60 |
| **+ Ours** | **32.76 (+2.6)** | **67.50 (+2.0)** | **44.84 (+1.2)** |

**Indoor Depth Estimation** (RMSE↓):

| Method | ScanNet++ | NYUv2 | ScanNet |
|------|-----------|-------|---------|
| DINOv2 | 0.374 | 0.442 | 0.309 |
| **+ Ours** | **0.336** | **0.420** | **0.292** |

**Cross-Domain Generalization** (fine-tuned only on ScanNet++):

| Dataset | Task | DINOv2 | + Ours | Gain |
|--------|------|--------|--------|------|
| ADE20k | mIoU↑ | 44.28 | **45.93** | +1.6 |
| Pascal VOC | mIoU↑ | 81.14 | **82.35** | +1.2 |
| KITTI (outdoor) | RMSE↓ | 3.03 | **2.91** | -0.12 |

### Ablation Study

**Generalization across different vision models** (ScanNet++):

| Model | Orig. mIoU | + Ours mIoU | Orig. RMSE | + Ours RMSE |
|------|-----------|-------------|-----------|-------------|
| DINOv2-reg | 30.92 | **33.39** | 0.419 | **0.382** |
| CLIP | 25.61 | **28.82** | 0.432 | **0.396** |
| MAE | 17.19 | **20.27** | 0.486 | **0.480** |
| DeiT-III | 18.62 | **22.98** | 0.435 | **0.382** |

**Feature Assembly Strategy** (NYUv2):

| Strategy | mAcc | mIoU | aAcc |
|------|------|------|------|
| Addition | 77.97 | 66.00 | 82.85 |
| Linear Fusion | 78.22 | 66.39 | 82.89 |
| **Concatenation** | **80.52** | **67.50** | **83.37**|

### Key Findings

- Fine-tuning on only **one** indoor dataset (ScanNet++) generalizes gains to outdoor scenes (KITTI) and generic scenes (ADE20k).
- **1 epoch is sufficient**; more epochs instead harm generalization (2 epochs mIoU 67.25 vs 1 epoch 67.50).
- Concatenating original + fine-tuned features is a key strategy, outperforming addition by 1.5 mIoU.
- Improvements mainly manifest in: (1) cleaner segmentation/depth estimation in texture-less regions, and (2) more accurate predictions on fine-grained structures (such as chair/table legs).
- RGB supervision ensures correct 3D geometry, while the feature loss independently learns semantics—separating gradients is the key to achieving multi-view consistent 3D features.

## Highlights & Insights

- **"Lift-distill" paradigm**: An information loop of 2D $\rightarrow$ 3D $\rightarrow$ 2D makes study of 3D representation an intermediate bridge rather than the final goal, which is a novel idea.
- **High training efficiency**: Only 1 epoch + 8.5 hours of fine-tuning are needed, which is extremely lightweight compared to methods like Pri3D that require pre-training from scratch.
- **Strong versatility**: Applicable to various vision foundation models like DINOv2, CLIP, MAE, and DeiT-III, with consistent improvements across all.
- The physical intuition behind the gradient separation design (RGB $\rightarrow$ geometry, feature $\rightarrow$ semantics) is solid: 2D features themselves lack 3D consistency and must be "corrected" via geometric anchoring provided by RGB.

## Limitations & Future Work

- The original features must still be preserved (concatenated) to maintain generalization capability, suggesting that fine-tuning might introduce bias from the training data.
- Only trained on a single indoor dataset; performance gains could be expected if scaled to larger and more diverse 3D data.
- The memory consumption of 3D Gaussian representations limits the feature dimension (maximum 64 dimensions, 128 dimensions is already infeasible).
- Larger models have not been tested (only DINOv2-small with 384 dimensions was used); validation on large/giant models is pending.
- Currently handles static scenes only; extending to dynamic scenes (video) might bring additional temporal consistency benefits.

## Related Work & Insights

- Unlike Pri3D (which uses multi-view consistency + 2D-3D correspondence for contrastive learning pre-training), this paper requires no training from scratch, only lightweight fine-tuning.
- Compared to concurrent works like LangSplat and LEGaussians, which distill semantic features into 3D Gaussians for 3D perception tasks, this work is the first to **distill backward into 2D models**.
- Insight: 3D Gaussian Splatting is not just a rendering tool, but also a powerful intermediate representation for feature fusion and enhancement.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The "2D $\rightarrow$ 3D $\rightarrow$ 2D" paradigm is proposed for the first time, simple and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 6 datasets (3 indoor + 3 cross-domain), 5 vision models, multiple ablation experiments.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear structure, rich visualizations, Algorithm 1 is concise and clear.
- **Value**: ⭐⭐⭐⭐ — High practicality through lightweight fine-tuning to enhance existing foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Learning 3D-Aware GANs from Unposed Images with Template Feature Field](learning_3d-aware_gans_from_unposed_images_with_template_feature_field.md)
- [\[NeurIPS 2025\] Mesh-RFT: Enhancing Mesh Generation via Fine-Grained Reinforcement Fine-Tuning](../../NeurIPS2025/3d_vision/mesh-rft_enhancing_mesh_generation_via_fine-grained_reinforcement_fine-tuning.md)
- [\[ECCV 2024\] FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally](flashsplat_2d_to_3d_gaussian_splatting_segmentation_solved_optimally.md)
- [\[ECCV 2024\] Learning 3D Geometry and Feature Consistent Gaussian Splatting for Object Removal](learning_3d_geometry_and_feature_consistent_gaussian_splatting_for_object_remova.md)
- [\[ECCV 2024\] VCD-Texture: Variance Alignment based 3D-2D Co-Denoising for Text-Guided Texturing](vcd-texture_variance_alignment_based_3d-2d_co-denoising_for_text-guided_texturin.md)

</div>

<!-- RELATED:END -->
