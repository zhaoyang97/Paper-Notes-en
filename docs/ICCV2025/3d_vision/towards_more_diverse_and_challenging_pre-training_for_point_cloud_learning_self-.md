---
title: >-
  [Paper Note] Towards More Diverse and Challenging Pre-training for Point Cloud Learning: Self-Supervised Cross Reconstruction with Decoupled Views
description: >-
  [ICCV 2025][3D Vision][Point cloud self-supervised learning] This paper proposes Point-PQAE, the first framework to introduce cross-view reconstruction into 3D generative self-supervised learning. By designing a point cl…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Point cloud self-supervised learning"
  - "cross-view reconstruction"
  - "decoupled views"
  - "positional query"
  - "pre-training"
date: 2026-05-08
content_hash: 5c8fb9d73ace32cc
---

# Towards More Diverse and Challenging Pre-training for Point Cloud Learning: Self-Supervised Cross Reconstruction with Decoupled Views

**Conference**: ICCV 2025
**arXiv**: [2509.01250](https://arxiv.org/abs/2509.01250)  
**Code**: [GitHub](https://github.com/aHapBean/Point-PQAE)  
**Area**: 3D Vision
**Keywords**: Point cloud self-supervised learning, cross-view reconstruction, decoupled views, positional query, pre-training

## TL;DR

This paper proposes Point-PQAE, the first framework to introduce cross-view reconstruction into 3D generative self-supervised learning. By designing a point cloud cropping mechanism to generate decoupled views, a View-Relative Positional Embedding (VRPE), and a Positional Query module, the pre-training task becomes more challenging and informative. Point-PQAE surpasses Point-MAE by an average of 6.7% on ScanObjectNN under the Mlp-Linear protocol.

## Background & Motivation

Self-supervised learning for point clouds has attracted broad interest due to its independence from manual annotations. Existing generative methods (exemplified by Point-MAE) primarily follow a **self-reconstruction** paradigm: a portion of points in a single view is masked, and the model reconstructs the masked regions from the visible ones.

**Core Observations and Motivation**:

**Two-view learning is superior**: In 2D self-supervised learning, two-view paradigms (e.g., SimCLR, MoCo) have been shown to outperform single-view approaches by introducing greater variance and richer information.

**Self-reconstruction is too easy**: The mask-and-reconstruct task within a single view has limited difficulty, potentially allowing the model to exploit shortcuts rather than learning deep semantic representations.

**Cross-reconstruction is more challenging**: Reconstructing one view from another (Cross Reconstruction) is substantially harder than self-reconstruction, as it requires simultaneously learning the relative spatial relationship between the two views and the internal spatial structure of each view.

**Key Challenges**:
- How to construct two views from point cloud data? (Unlike images, there is no well-established Random Crop equivalent.)
- How to enable cross-reconstruction between two decoupled views? (Requires establishing cross-view spatial correspondences.)

## Method

### Overall Architecture

Point-PQAE consists of three core modules:
1. **Decoupled view generation**: Point cloud cropping → independent normalization → random rotation → two decoupled views.
2. **VRPE generation**: Record the geometric center of each crop → compute relative positions → apply fixed sinusoidal positional encoding.
3. **Positional Query module**: Use VRPE as Query and the view's latent representation as Key/Value in cross-attention → decoder reconstructs the other view.

### Key Designs

1. **Point Cloud Crop Mechanism**:

   This work is the first to design a cropping mechanism for 3D self-supervised learning. Unlike 2D images, the number of points within a fixed-size 3D cube can vary drastically. The design proceeds as follows:

   - Randomly sample crop ratios $r_1, r_2 \in [r_m, 1]$ (with $r_m = 0.6$).
   - Randomly select two center points $\mathbf{C_1}, \mathbf{C_2}$.
   - For each center, select the nearest $r_i \times p$ points to form views $\mathbf{X}_1, \mathbf{X}_2$.
   - Record the geometric center $\mathbf{L}_1, \mathbf{L}_2$ of each view.
   - **Independent normalization**: Each view is normalized with its own geometric center as the origin, using min-max normalization.
   - **Random rotation augmentation**: Each view is independently rotated to further decouple the coordinate systems.

   Independent normalization and random rotation completely decouple the coordinate frames of the two views, eliminating any fixed spatial relationship.

2. **View-Relative Positional Embedding (VRPE)**:

   The key to cross-view reconstruction lies in informing the model of the relative spatial relationship between the two views.

   - Define the inter-view relative position: $\mathbf{RL}_{1 \to 2} = \mathbf{L}_1 - \mathbf{L}_2$
   - Combine with patch-level relative positions: $\mathbf{RP}_{1 \to 2} = \text{Concat}(\mathbf{G}_2, \mathbf{RL}_{1 \to 2}) \in \mathbb{R}^{n \times 6}$
   - Map the 6-dimensional relative position to $D$ dimensions using **fixed sinusoidal encoding** (rather than learnable encoding):

   $\mathbf{VRPE}_{1 \to 2}^i = [\sin(\frac{\mathbf{RP}^i}{e^{2/D_{12}}}), \cos(\frac{\mathbf{RP}^i}{e^{2/D_{12}}}), \ldots]$

   Fixed encoding prevents learnable embeddings from degrading the precise expression of relative positions, as confirmed by ablation studies.

3. **Positional Query Block**:

   The core idea is to use VRPE as Query to attend over the latent representation of the other view, enabling cross-view information extraction.

   Given that view 1's latent representation $\mathbf{H}_1$ already encodes its intrinsic features and global context, combining it with $\mathbf{VRPE}_{1 \to 2}$ enables reconstruction of view 2:

   $\mathbf{T}_2 = \text{Softmax}(\frac{\mathbf{Q}_{\mathbf{VRPE}_{1 \to 2}} \mathbf{K}_{\mathbf{H}_1}}{\sqrt{D}}) \mathbf{V}_{\mathbf{H}_1}$

   Reconstruction is symmetric (bidirectional cross-reconstruction), and the final loss uses $\ell_2$ Chamfer Distance.

### Loss & Training

$$\mathcal{L}_{cross} = \mathcal{L}_{2 \to 1} + \mathcal{L}_{1 \to 2}$$

Each term is the standard Chamfer Distance:

$$\mathcal{L}_{2 \to 1} = \frac{1}{|\mathbf{P}_{pred}^1|} \sum_{a \in \mathbf{P}_{pred}^1} \min_{b \in \mathbf{P}_1} \|a - b\|_2^2 + \frac{1}{|\mathbf{P}_1|} \sum_{b \in \mathbf{P}_1} \min_{a \in \mathbf{P}_{pred}^1} \|a - b\|_2^2$$

- Pre-training is conducted on ShapeNet (~51K models) for 300 epochs.
- Encoder: 12 Transformer blocks; Decoder: 4 layers; hidden dimension 384; 6 attention heads.
- AdamW optimizer, lr=0.0005, weight decay=0.05, cosine decay schedule.
- Each point cloud is sampled to 1024 points and cropped into 64 patches × 32 points.

## Key Experimental Results

### Main Results (ScanObjectNN Classification, Real-World 3D Objects)

| Method | Params | OBJ-BG | OBJ-ONLY | PB-T50-RS | Protocol |
|--------|--------|--------|----------|-----------|----------|
| Point-MAE† | 22.1M | 92.6 | 91.9 | 88.4 | Full |
| **Point-PQAE** | 22.1M | **95.0** | **93.6** | **89.6** | Full |
| Point-MAE† | 22.1M | 82.8 | 83.2 | 74.1 | Mlp-Linear |
| **Point-PQAE** | 22.1M | **89.3** | **90.2** | **80.8** | Mlp-Linear |
| Point-MAE† | 22.1M | 85.8 | 85.5 | 80.4 | Mlp-3 |
| **Point-PQAE** | 22.1M | **90.7** | **90.9** | **83.3** | Mlp-3 |

Under the most stringent frozen evaluation protocol (Mlp-Linear), Point-PQAE outperforms Point-MAE by an average of **6.7%** across three variants (82.8→89.3, 83.2→90.2, 74.1→80.8), demonstrating that cross-reconstruction learns substantially higher-quality representations.

### Ablation Study (ScanObjectNN-PB-T50-RS, Full Protocol)

| Configuration | OBJ-BG | OBJ-ONLY | PB-T50-RS | Note |
|---------------|--------|----------|-----------|------|
| Self-recon (Point-MAE) | 92.6 | 91.9 | 88.4 | Self-reconstruction baseline |
| Cross-recon (w/o VRPE) | - | - | ~85 | Cross-recon fails without positional info |
| Cross-recon + learnable PE | - | - | ~88 | Learnable encoding is insufficiently precise |
| Cross-recon + VRPE (Ours) | **95.0** | **93.6** | **89.6** | Fixed sinusoidal encoding is optimal |

### Few-Shot Learning (ModelNet40)

| Method | 5-way 10-shot | 5-way 20-shot | 10-way 10-shot | 10-way 20-shot |
|--------|:---:|:---:|:---:|:---:|
| Point-MAE† | 96.4±2.8 | 97.8±2.0 | 92.5±4.4 | 95.2±3.9 |
| **Point-PQAE** | **96.9±3.2** | **98.9±1.0** | **94.1±4.2** | **96.3±2.7** |
| ReCon (cross-modal) | 97.3±1.9 | 98.9±1.2 | 93.3±3.9 | 95.8±3.0 |

Without relying on cross-modal teachers, Point-PQAE matches or surpasses ReCon, which leverages strong pre-trained 2D teachers.

### Key Findings

1. **Cross-reconstruction significantly outperforms self-reconstruction**: Under the same parameter budget, Point-PQAE achieves an average improvement of 6.7% under the Mlp-Linear protocol, confirming that a more challenging pre-training task leads to better representations.
2. **VRPE is critical to the success of cross-reconstruction**: Without positional information, cross-reconstruction performance drops sharply; fixed sinusoidal encoding outperforms learnable encoding.
3. **Generalization across crop ratios**: Although pre-training uses $r_m = 0.6$, the model generalizes well to other crop ratios.
4. **Single-modal method first approaches cross-modal counterparts**: Under Mlp-Linear/Mlp-3 protocols, Point-PQAE approaches ACT and ReCon, which rely on 2D teachers.

## Highlights & Insights

- **Paradigm innovation**: The shift from self-reconstruction to cross-reconstruction represents a significant paradigm advance in 3D generative self-supervised learning.
- **Novel point cloud cropping mechanism**: The nearest-neighbor-based cropping strategy is simple and effective, avoiding the density imbalance inherent in cubic cropping in 3D space.
- **Elegant VRPE design**: Encoding inter-view and patch-level positional relationships as fixed sinusoidal signals achieves both precision and efficiency.
- **Plug-and-play PQ module**: The Positional Query module can be conveniently applied to other tasks such as knowledge distillation.

## Limitations & Future Work

- Pre-training relies solely on synthetic data (ShapeNet); effectiveness on real-world point clouds (e.g., outdoor LiDAR) remains to be verified.
- Cross-reconstruction introduces additional computational overhead (encoding two views + cross-attention).
- Hyperparameters such as crop ratio and number of views may require tuning for different datasets.
- A comprehensive comparison with more recent 3D self-supervised methods (e.g., additional MAE-based variants) is lacking.

## Related Work & Insights

- Point-MAE's self-reconstruction paradigm serves as the direct baseline; this work improves upon it from the perspective of increasing task difficulty.
- The two-view learning philosophy of 2D methods such as SimCLR and MoCo is successfully transferred to 3D generative learning.
- The VRPE design may inspire other tasks that require modeling spatial relationships between two sets of 3D points.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The cross-reconstruction paradigm is proposed for the first time in 3D self-supervised learning; both the point cloud cropping mechanism and VRPE are original contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated across classification, few-shot learning, and segmentation with three evaluation protocols and thorough ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear, the comparison with self-reconstruction is intuitive, and mathematical derivations are complete.
- **Value**: ⭐⭐⭐⭐ Introduces a new paradigm for 3D self-supervised learning with significant gains under frozen evaluation protocols and strong practical guidance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] StruMamba3D: Exploring Structural Mamba for Self-supervised Point Cloud Representation Learning](strumamba3d_exploring_structural_mamba_for_self-supervised_point_cloud_represent.md)
- [\[ICCV 2025\] 4D Visual Pre-training for Robot Learning](4d_visual_pretraining_for_robot_learning.md)
- [\[CVPR 2026\] E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training](../../CVPR2026/3d_vision/e-rayzer_self-supervised_3d_reconstruction_as_spatial_visual_pre-training.md)
- [\[ICCV 2025\] No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views](no_pose_at_all_self-supervised_pose-free_3d_gaussian_splatting_from_sparse_views.md)
- [\[ICCV 2025\] DAP-MAE: Domain-Adaptive Point Cloud Masked Autoencoder for Effective Cross-Domain Learning](dap-mae_domain-adaptive_point_cloud_masked_autoencoder_for_effective_cross-domai.md)

</div>

<!-- RELATED:END -->
