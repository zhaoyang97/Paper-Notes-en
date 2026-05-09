---
title: >-
  [Paper Note] TrackAny3D: Transferring Pretrained 3D Models for Category-unified 3D Point Cloud Tracking
description: >-
  [ICCV 2025][Autonomous Driving][3D Single Object Tracking] TrackAny3D is the first work to transfer large-scale pretrained 3D models to category-agnostic 3D single object tracking. By introducing a dual-path adapter, a Mixture of Geometry Experts (MoGE) module, and a temporal context optimization strategy, it achieves state-of-the-art performance on cross-category unified tracking within a single model.
tags:
  - ICCV 2025
  - Autonomous Driving
  - 3D Single Object Tracking
  - Pretrained Model Transfer
  - Mixture of Geometry Experts
  - Parameter-Efficient Fine-Tuning
  - Point Cloud
date: 2026-05-08
content_hash: ff17a9608238029c
---

# TrackAny3D: Transferring Pretrained 3D Models for Category-unified 3D Point Cloud Tracking

**Conference**: ICCV 2025
**arXiv**: [2507.19908](https://arxiv.org/abs/2507.19908)
**Code**: N/A
**Area**: Autonomous Driving
**Keywords**: 3D Single Object Tracking, Pretrained Model Transfer, Mixture of Geometry Experts, Parameter-Efficient Fine-Tuning, Point Cloud

## TL;DR

TrackAny3D is the first work to transfer large-scale pretrained 3D models to category-agnostic 3D single object tracking. By introducing a dual-path adapter, a Mixture of Geometry Experts (MoGE) module, and a temporal context optimization strategy, it achieves state-of-the-art performance on cross-category unified tracking within a single model.

## Background & Motivation

**Background**: Mainstream 3D LiDAR single object tracking (SOT) methods follow a category-specific learning paradigm, training and evaluating dedicated models for each object category (car, pedestrian, cyclist, etc.), which yields strong per-category accuracy.

**Limitations of Prior Work**: The category-specific paradigm is impractical for real-world deployment — it requires training and storing numerous category-dedicated networks with substantial computational overhead, and critically cannot generalize to unseen categories, which is a fundamental limitation for open-world applications. Experiments show that naively training existing methods (STNet, CXTrack, MBPTrack) in a unified multi-category setting leads to severe performance drops of 3.8%, 10.3%, and 14.2%, respectively.

**Key Challenge**: Objects across categories differ drastically in scale, motion patterns, and structural complexity (e.g., vehicles are large rigid objects while pedestrians are small non-rigid ones). This geometric heterogeneity makes it difficult for a single model to handle all categories simultaneously.

**Goal**: (1) How to learn category-agnostic yet geometry-aware representations without introducing manual biases? (2) How to transfer pretrained 3D models to the tracking task? (3) How to effectively model temporal information?

**Key Insight**: In 2D vision and NLP, pretrained models combined with PEFT have demonstrated strong generalization to downstream tasks. Pretrained 3D point cloud models (e.g., RECON, Point-MAE) have also emerged, yet remain largely unexplored in 3D SOT. The authors hypothesize that geometric priors from pretrained models can alleviate inter-category geometric discrepancies.

**Core Idea**: Freeze the parameters of a pretrained 3D model and adapt it via lightweight dual-path adapters, MoGE for adaptive expert activation, and learnable temporal tokens for propagating historical context — forming the first category-unified pretrained-transfer 3D tracking framework.

## Method

### Overall Architecture

Given template point cloud $\mathcal{P}^t$ and search region point cloud $\mathcal{P}^s$, local features are first extracted via a patch embedding layer and encoded as tokens. Masks $\mathcal{M}^t$, $\mathcal{M}^s$ (processed by a dynamic mask weighting module) and a learnable temporal token $\mathcal{T}_0$ are concatenated and fed into a frozen pretrained encoder (RECON). Each Transformer layer in the encoder is augmented with a dual-path adapter and a MoGE module. The resulting search tokens are passed to a localization head to predict the bounding box $(x, y, z, \theta)$.

### Key Designs

1. **Two-Path Adapter**:

    - **Function**: Efficiently aligns features to the 3D SOT task while keeping the pretrained model frozen.
    - **Mechanism**: Consists of an adaptation path and a gating scoring path. The adaptation path comprises a down-projection $\mathbf{W}_{dn} \in \mathbb{R}^{d \times r}$, GeLU activation, and an up-projection $\mathbf{W}_{up} \in \mathbb{R}^{r \times d}$; the gating path uses a scoring matrix $\mathbf{W}_s \in \mathbb{R}^{d \times 1}$ and ReLU to compute a dynamic scaling factor per token. The two paths are combined via element-wise multiplication: $\text{AD}(\mathbf{F}_i) = \text{ReLU}(\mathbf{F}_i \mathbf{W}_s) \odot \text{GeLU}(\mathbf{F}_i \mathbf{W}_{dn}) \mathbf{W}_{up}$
    - **Design Motivation**: Full fine-tuning overwrites pretrained knowledge and degrades performance. The gating mechanism enables data-driven control of adaptation intensity, allowing different tokens to receive different levels of adaptation.

2. **Mixture of Geometry Experts (MoGE)**:

    - **Function**: Adaptively activates specialized sub-networks to handle objects with different geometric characteristics.
    - **Mechanism**: Contains $M=8$ geometry experts (FFN-structured) with Top-K ($K=4$) gating for expert selection. The router computes scores via a learnable gating network with expert embeddings $\mathbf{W}_j^R \in \mathbb{R}^{d \times M}$, followed by Softmax to select the Top-K experts. The output is: $\mathbf{E}_j(\textbf{Z}_j) = \sum_{m=1}^{M} \mathbf{R}_j(\textbf{Z}_j, K) \mathbf{E}_j^m(\textbf{Z}_j)$
    - **Design Motivation**: A significant distribution gap exists between the pretraining data (e.g., indoor ShapeNet objects) and real 3D SOT scenes. Adapters alone only partially close this gap. MoGE enables different experts to learn distinct geometric patterns — analysis reveals that Experts 3/7 preferentially activate for non-rigid objects (pedestrians, cyclists), while Experts 0/6 are more active for rigid structures (vehicles), enabling adaptive geometry-aware processing.

3. **Temporal Context Optimization**:

    - **Function**: Models temporal information to compensate for the static-task nature of pretrained models.
    - **Mechanism**: (1) Learnable temporal token $\mathcal{T}_0$: propagated and updated across time steps as $\mathcal{T}_0^t = \mathcal{T}_0 + \mathcal{T}_{out}^{t-1}$, interacting fully with template/search tokens at each frame before being passed to the next. (2) Dynamic Mask Weighting (DMW): introduces learnable weights $\beta^t, \beta^s$ to adaptively rescale template masks (foreground 0.8, background 0.2) and search masks (uniform 0.5) as $\tilde{\mathcal{F}}^* = \mathcal{F}^* + \mathcal{M}^* \odot \beta^*$.
    - **Design Motivation**: Pretrained tasks (static shape reconstruction/recognition) lack temporal modeling, whereas tracking requires temporal consistency. The temporal token propagates historical context across frames, while DMW adaptively adjusts mask weights according to category-specific spatiotemporal variations.

### Loss & Training

RECON is used as the pretrained backbone with its original parameters frozen. Each training sample consists of 3 frames. Template and search regions each sample 128 points. The adapter bottleneck dimension is $r=72$. MoGE is inserted after every other Transformer block. Inference runs on a single RTX 3090.

## Key Experimental Results

### Main Results — KITTI Dataset (Category-unified Setting)

| Method | Setting | Car | Pedestrian | Van | Cyclist | Mean (Succ/Prec) |
|--------|---------|-----|------------|-----|---------|-----------------|
| MBPTrack | Specific | 73.4/84.8 | 68.6/93.9 | 61.3/72.7 | 76.7/94.3 | 70.3/87.9 |
| MBPTrack | Unified | 62.3/72.1 | 50.2/80.9 | 66.6/78.2 | 71.8/92.2 | 56.1/74.9 |
| MoCUT | Unified | 67.6/80.5 | 63.3/90.0 | 64.5/78.8 | 76.7/94.2 | 65.8/85.0 |
| **TrackAny3D** | **Unified** | **73.4/85.2** | **59.6/85.6** | **70.0/82.8** | **74.7/94.0** | **67.1/85.4** |

### Ablation Study — KITTI

| Configuration | Car | Ped | Van | Cyclist | Mean (Succ/Prec) |
|--------------|-----|-----|-----|---------|-----------------|
| Full Fine-tune | 69.8/83.6 | 53.6/81.6 | 62.2/73.6 | 65.8/90.8 | 62.0/82.0 |
| FF + MoGE + TT + DMW | 72.5/84.0 | 57.9/82.0 | 70.8/82.9 | 72.8/93.7 | 66.0/83.3 |
| AD only | -- | -- | -- | -- | Baseline |
| AD + MoGE + TT + DMW (**Full**) | 73.4/85.2 | 59.6/85.6 | 70.0/82.8 | 74.7/94.0 | 67.1/85.4 |

### Key Findings

- **PEFT outperforms full fine-tuning**: Frozen pretraining + adapter (67.1%) significantly outperforms full fine-tuning (62.0%), confirming that preserving pretrained geometric priors is critical.
- **MoGE geometric adaptivity is effective**: Expert activation patterns are strongly correlated with object geometry — non-rigid objects (pedestrians, cyclists) preferentially activate Experts 3/7, while rigid objects (vehicles) activate Experts 0/6.
- **Cross-dataset generalization**: A model trained on KITTI directly evaluated on Waymo achieves 64.0% success rate (SOTA), outperforming MoCUT by 2.1%.
- **Narrowing the gap between unified and category-specific models**: On NuScenes, the Bus category even surpasses all category-specific methods, demonstrating that with sufficient data, a unified model can exceed category-specific counterparts.
- **Only 5.30M trainable parameters**, maintaining high efficiency.

## Highlights & Insights

- **First work to transfer pretrained 3D models to 3D SOT**: This establishes a new paradigm of "category-unified + pretrained transfer," which is more suitable for real-world deployment than "category-specific + training from scratch." It demonstrates that geometric priors from 3D pretrained models are genuinely valuable for downstream tasks.
- **The geometric sensitivity analysis of MoGE** is particularly compelling: different experts naturally learn to handle distinct geometric patterns rather than simply partitioning by category label, suggesting potential generalization to unseen categories at test time.
- **The dual-path adapter design** is simple yet effective: the scoring path controls adaptation intensity, overcoming the limitations of a uniform scaling factor.

## Limitations & Future Work

- Inference speed is relatively slow (28 FPS), below methods such as M2Track (57 FPS), primarily due to the multi-expert computation in MoGE.
- Only RECON is explored as the pretrained backbone; the impact of stronger 3D pretrained models (e.g., Point-MAE, CLIP2Point) remains uninvestigated.
- Performance on small non-rigid objects such as pedestrians still has room for improvement; KITTI Pedestrian lags behind the category-specific StreamTrack (59.6 vs. 70.5).
- The learnable weight dimensions in DMW are fixed and do not adapt dynamically.

## Related Work & Insights

- **vs. MoCUT**: The only prior work addressing category-unified 3D SOT, but relies on manually designed non-learnable constraints and hand-tuned hyperparameters. TrackAny3D surpasses MoCUT through data-driven MoGE (+1.3% on KITTI Mean).
- **vs. MBPTrack**: A strong category-specific baseline that drops 14.2% when switched to unified training. TrackAny3D in the unified setting (67.1%) approaches or matches MBPTrack in the category-specific setting (70.3%), with a gap of only 3.2%.
- **vs. MemDisst**: The only related work incorporating 3D pretraining, but requires full network learning and relies on 2D distillation without achieving category unification.
- The adapter + MoGE paradigm is transferable to other 3D point cloud downstream tasks such as semantic segmentation and scene flow estimation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to propose a pretrained-transfer + category-unified paradigm for 3D SOT
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets + cross-dataset generalization + comprehensive ablations
- Writing Quality: ⭐⭐⭐⭐ Clear structure and well-defined problem formulation
- Value: ⭐⭐⭐⭐ Informative for downstream applications of 3D point cloud pretrained models

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Mixed Signals: A Diverse Point Cloud Dataset for Heterogeneous LiDAR V2X Collaboration](mixed_signals_a_diverse_point_cloud_dataset_for_heterogeneous_lidar_v2x_collabor.md)
- [\[CVPR 2026\] Points-to-3D: Structure-Aware 3D Generation with Point Cloud Priors](../../CVPR2026/autonomous_driving/points-to-3d_structure-aware_3d_generation_with_point_cloud_priors.md)
- [\[ICCV 2025\] Hermes: A Unified Self-Driving World Model for Simultaneous 3D Scene Understanding and Generation](hermes_a_unified_self-driving_world_model_for_simultaneous_3d_scene_understandin.md)
- [\[AAAI 2026\] CompTrack: Information Bottleneck-Guided Low-Rank Dynamic Token Compression for Point Cloud Tracking](../../AAAI2026/autonomous_driving/comptrack_information_bottleneckguided_lowrank_dynamic_token_compres.md)
- [\[ICCV 2025\] Distilling Diffusion Models to Efficient 3D LiDAR Scene Completion](distilling_diffusion_models_to_efficient_3d_lidar_scene_completion.md)

<!-- RELATED:END -->
