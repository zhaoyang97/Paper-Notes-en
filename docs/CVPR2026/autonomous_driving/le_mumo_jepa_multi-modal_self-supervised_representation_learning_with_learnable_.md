---
title: >-
  [Paper Note] Le MuMo JEPA: Multi-Modal Self-Supervised Representation Learning with Learnable Fusion Tokens
description: >-
  [CVPR 2026][Autonomous Driving][Multi-modal self-supervision] This work extends the LeJEPA self-supervised framework to a multi-modal setting by introducing learnable fusion tokens as a Perceiver-style latent bottleneck within a shared Transformer, enabling efficient fusion of RGB with companion modalities (LiDAR depth / thermal infrared). A pruning strategy reduces attention overhead by approximately 9×. On Waymo, CenterNet 3D detection mAP XY reaches 23.6 (+4.3 over RGB-only LeJEPA) and Depth MAE improves from 4.704 to 2.860.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Multi-modal self-supervision
  - JEPA
  - fusion tokens
  - latent bottleneck
  - RGB-LiDAR fusion
date: 2026-05-08
content_hash: 35950d824289b706
---

# Le MuMo JEPA: Multi-Modal Self-Supervised Representation Learning with Learnable Fusion Tokens

**Conference**: CVPR 2026
**arXiv**: [2603.24327](https://arxiv.org/abs/2603.24327)
**Code**: N/A
**Area**: Autonomous Driving / Multi-Modal Self-Supervised Learning
**Keywords**: Multi-modal self-supervision, JEPA, fusion tokens, latent bottleneck, RGB-LiDAR fusion

## TL;DR

This work extends the LeJEPA self-supervised framework to a multi-modal setting by introducing learnable fusion tokens as a Perceiver-style latent bottleneck within a shared Transformer, enabling efficient fusion of RGB with companion modalities (LiDAR depth / thermal infrared). A pruning strategy reduces attention overhead by approximately 9×. On Waymo, CenterNet 3D detection mAP XY reaches 23.6 (+4.3 over RGB-only LeJEPA) and Depth MAE improves from 4.704 to 2.860.

## Background & Motivation

**Background**: Autonomous driving perception systems rely on multiple sensors (cameras, LiDAR, etc.), yet dominant multi-modal perception models (BEVFusion, TransFusion, etc.) remain fully supervised and require large quantities of 3D annotations. Self-supervised learning (BYOL, DINO, MAE, I-JEPA, etc.) has achieved strong results in single-modal settings but almost exclusively operates on a single modality.

**Limitations of Prior Work**: (1) Single-modal self-supervised learning discards complementary signals from multiple sensors—RGB captures texture and color while LiDAR provides geometric depth, and learning each independently fails to exploit their synergy. (2) Existing multi-modal self-supervised methods (ImageBind via contrastive learning, MultiMAE via masked reconstruction) do not clearly outperform single-modal baselines under strict from-scratch training. (3) Weak late fusion lacks expressiveness, while full token all-to-all attention incurs quadratic complexity.

**Key Challenge**: Multi-modal fusion requires dense cross-modal interaction to capture complementary information, yet full cross-attention between both modalities is computationally prohibitive (doubling the token count raises attention cost by roughly 4×).

**Key Insight**: The SIGReg regularization in the JEPA framework provides a modality-agnostic shared objective—pulling embeddings from both modalities toward an isotropic Gaussian $\mathcal{N}(0, \mathbf{I})$—without requiring hard negative mining as in pairwise contrastive learning.

**Core Idea**: Learnable fusion tokens are introduced as a spatial memory buffer. After the first attention layer, modality-specific tokens are pruned. This information bottleneck forces the model to compress cross-modal evidence into the fusion token grid at an early stage, while substantially reducing computation in subsequent layers.

## Method

### Overall Architecture

A shared ViT-Small/16 encoder processes three groups of token sequences: $[\text{CLS}(1), \mathbf{F}(N), \mathbf{C}(N), \mathbf{M}(N)]$, where $\mathbf{F}$ denotes fusion tokens, $\mathbf{C}$ RGB tokens, and $\mathbf{M}$ companion modality tokens, totaling $1 + 3N = 589$ tokens. LiDAR depth is projected into camera coordinates to produce an aligned 2D depth map; each modality is tokenized via an independent patch stem. The training objective combines LeJEPA's invariance loss with SIGReg regularization, applied to the joint multi-modal CLS embedding.

### Key Designs

1. **Learnable Fusion Tokens + Pruning Strategy**:
    - **Function**: Performs cross-modal information fusion within the shared Transformer while controlling computational cost.
    - **Mechanism**: $N$ learnable fusion tokens (matching the patch count) are created. In the first layer, each fusion token $\mathbf{f}_i$ attends to the spatially corresponding RGB patch $\mathbf{c}_i$ and companion modality patch $\mathbf{m}_i$. After the first layer, all $2N$ modality tokens are pruned; subsequent layers process only $1 + N$ tokens.
    - **Design Motivation**: Pruning reduces attention overhead from $\mathcal{O}((1+3N)^2)$ to $\mathcal{O}((1+N)^2)$, approximately a 9× reduction. It also forces early cross-modal compression into the fusion tokens, forming an explicit information bottleneck. Gradients still propagate back through the first-layer cross-attention path to update both modality patch stems.

2. **SIGReg Joint Multi-Modal Regularization**:
    - **Function**: Prevents representational collapse and provides a modality-agnostic shared learning objective.
    - **Mechanism**: The joint multi-modal CLS embedding is projected and its empirical distribution is matched to $\mathcal{N}(0, \mathbf{I})$ via SIGReg, implemented through characteristic function matching with random projections at complexity $\mathcal{O}(BK(T+d))$.
    - **Design Motivation**: Compared to VICReg, which matches only variance and covariance, SIGReg more directly suppresses modality-specific anisotropy. No stop-gradient or teacher–student network is required, simplifying the multi-modal training framework.

3. **Unified 2D Spatial Multi-Modal Tokenization**:
    - **Function**: Unifies heterogeneous sensor data into a shared 2D token space.
    - **Mechanism**: LiDAR point clouds are projected into camera coordinates and rendered as an aligned depth map (depth-sorted with closer points occluding farther ones, normalized to a maximum range of 80 m). Thermal infrared images are resized to the same spatial grid. Each modality passes through an independent patch stem with modality embeddings $\mathbf{e}_{cam}$ and $\mathbf{e}_{mod}$.
    - **Design Motivation**: This avoids introducing a separate 3D sparse backbone, maintains a unified dense ViT architecture, and allows the same framework to accommodate both RGB-LiDAR and RGB-Thermal settings by simply swapping the patch stem.

### Loss & Training

$$\mathcal{L}_{\text{MM}} = \lambda \cdot \mathcal{L}_{\text{SIGReg}}(\mathbf{Z}^{(\text{joint})}) + (1 - \lambda) \cdot \mathcal{L}_{\text{inv}}^{(\text{joint})}$$

where $\mathcal{L}_{\text{inv}}^{(\text{joint})}$ is a mean-squared invariance loss that pulls global and local fusion crop embeddings together. Training uses multi-crop augmentation: global crops at $224 \times 224$ (scale $[0.4, 1.0]$) and local crops at $96 \times 96$ (scale $[0.05, 0.4]$). For both Waymo and nuScenes, training consists of 5 epochs of SSL followed by 5 epochs of linear probing.

## Key Experimental Results

### Main Results (Waymo, from scratch)

| Method | Training Data | mAP XY ↑ | Depth MAE ↓ | Seg. mIoU ↑ |
|--------|--------------|----------|-------------|-------------|
| LeJEPA | RGB | 19.3 | 4.704 | 0.261 |
| DINOv3 | RGB | 15.2 | 5.314 | 0.239 |
| LiDAR-only | Depth | 15.4 | 2.982 | 0.151 |
| MultiMAE-SS | RGB+Depth | 13.5 | 4.441 | 0.221 |
| ImageBind | RGB+Depth | 13.4 | 4.309 | 0.243 |
| **Le MuMo JEPA** | **RGB+Depth** | **23.6** | **2.860** | **0.275** |

### Ablation Study (Waymo Fusion Strategy Comparison)

| Configuration | mAP XY ↑ | Depth MAE ↓ | Seg. mIoU ↑ |
|--------------|----------|-------------|-------------|
| Early Fusion RGBD | 18.1 | 4.767 | 0.248 |
| Late Fusion | 18.7 | 4.802 | 0.251 |
| FT-Pruned + VICReg | 22.8 | 2.911 | 0.248 |
| FT-Persistent + SIGReg | 23.1 | 2.846 | 0.271 |
| Le MuMo JEPA (default) | 23.6 | 2.860 | 0.275 |

### Key Findings
- Le MuMo JEPA improves mAP XY by 4.3 over the strongest single-modal baseline (LeJEPA at 19.3) and reduces Depth MAE from 4.704 to 2.860.
- From-scratch ImageBind and MultiMAE on Waymo perform even worse than single-modal LeJEPA, indicating that contrastive and reconstruction objectives are more data-hungry in small-data from-scratch settings.
- Pruned fusion outperforms persistent routing in the efficiency–accuracy trade-off, as the information bottleneck enforces early cross-modal compression.
- SIGReg outperforms VICReg on the joint multi-modal CLS embedding, as the isotropic Gaussian target more directly suppresses modality-specific anisotropy.
- Le MuMo JEPA achieves state-of-the-art results on nuScenes (mAP XY 9.52 vs. second-best 6.95) and on cross-domain transfer to FLIR RGB-Thermal (Waymo→FLIR mAP50 1.56 vs. ImageBind 0.72).

## Highlights & Insights
- **Elegant information bottleneck design**: Fusion tokens absorb cross-modal information only in the first layer, after which modality tokens are pruned. Computational constraints are traded for better representational compression—similar in spirit to Perceiver but more aggressive (pruning after a single layer).
- **SIGReg as a modality adhesive**: Pulling both modalities toward the same data-agnostic target distribution is a more natural alternative to pairwise contrastive learning—no negative samples, no teacher network, conceptually clean and efficient.
- **Unified 2D avoids a 3D backbone**: Projecting LiDAR into 2D rather than retaining the native 3D sparse format sacrifices some 3D structural information but yields architectural uniformity and flexibility (switching between RGB-LiDAR and RGB-Thermal requires only swapping the patch stem).
- **Fair from-scratch comparison**: All methods are trained from scratch under identical data and compute budgets, eliminating the confounding effect of pre-trained weights.

## Limitations & Future Work
- Projecting LiDAR to 2D discards native 3D structure (e.g., occlusion relationships, point cloud density variation), potentially limiting performance in complex 3D reasoning scenarios.
- Evaluation is restricted to ViT-Small/16; larger models (ViT-Base/Large) may exhibit different fusion dynamics.
- Training epochs are very short (5 epochs) compared to standard SSL schedules (300+ epochs), and models may not have fully converged.
- After pruning, all modality token information depends on a single attention pass in the first layer, which may fail to capture complex cross-modal relationships that require multi-layer interaction.
- Downstream evaluation uses frozen patch probing only; comprehensive end-to-end fine-tuning results are not reported.

## Related Work & Insights
- **vs. ImageBind**: ImageBind aligns multi-modal embeddings via contrastive learning; trained from scratch on Waymo, it achieves only mAP XY 13.4, below even LeJEPA's 19.3. Le MuMo JEPA reaches 23.6, demonstrating that contrastive objectives are inferior to SIGReg + fusion tokens in low-data regimes.
- **vs. MultiMAE**: MultiMAE learns multi-modal representations through masked reconstruction and similarly underperforms from scratch (13.5–13.7). Even with multitask supervision (MultiMAE-MT), it falls well short of Le MuMo JEPA.
- **vs. BEVFusion**: BEVFusion is fully supervised and requires extensive 3D annotations; Le MuMo JEPA is entirely self-supervised. Direct numerical comparison is not provided as the experimental settings are incompatible.
- **Insight**: The key to self-supervised multi-modal fusion lies not in aligning two modalities to each other, but in compressing information within a shared representation space—bottleneck design matters more than fusion granularity.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The multi-modal extension of the JEPA framework via fusion tokens and SIGReg is novel; the pruning strategy is cleverly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three datasets, multiple baselines, detailed ablations, and computational efficiency analysis; however, training epochs are short and model scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear, experimental setup is transparent (from-scratch), and ablations are convincing.
- **Value**: ⭐⭐⭐⭐ Provides an efficient fusion paradigm for multi-modal self-supervised learning; practical deployment value awaits validation at larger scale.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] CCF: Complementary Collaborative Fusion for Domain Generalized Multi-Modal 3D Object Detection](ccf_complementary_collaborative_fusion_for_domain_generalized_multi-modal_3d_obj.md)
- [\[CVPR 2026\] Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation](towards_balanced_multi_modal_learning_in_3d_human_pose_estimation.md)
- [\[AAAI 2026\] Dual-branch Spatial-Temporal Self-supervised Representation for Enhanced Road Network Learning](../../AAAI2026/autonomous_driving/dual-branch_spatial-temporal_self-supervised_representation_for_enhanced_road_ne.md)
- [\[CVPR 2026\] TerraSeg: Self-Supervised Ground Segmentation for Any LiDAR](terraseg_self-supervised_ground_segmentation_for_any_lidar.md)
- [\[ICCV 2025\] Self-Supervised Sparse Sensor Fusion for Long Range Perception](../../ICCV2025/autonomous_driving/self-supervised_sparse_sensor_fusion_for_long_range_perception.md)

<!-- RELATED:END -->
