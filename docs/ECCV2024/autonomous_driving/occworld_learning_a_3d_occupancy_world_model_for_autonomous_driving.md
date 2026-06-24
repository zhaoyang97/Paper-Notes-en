---
title: >-
  [Paper Note] OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving
description: >-
  [ECCV 2024][Autonomous Driving][3D Occupancy] OccWorld proposes learning a world model in 3D occupancy space. It tokenizes 3D occupancy via VQ-VAE and predicts future scene evolution and ego-vehicle trajectories autoregressively using a GPT-style spatial-temporal generative Transformer, achieving competitive planning performance on nuScenes without requiring instance or HD map annotations.
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "3D Occupancy"
  - "World Model"
  - "4D Forecasting"
  - "Trajectory Planning"
date: 2026-05-08
content_hash: 22da02029f287db6
---

# OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving

**Conference**: ECCV 2024  
**arXiv**: [2311.16038](https://arxiv.org/abs/2311.16038)  
**Code**: [https://github.com/wzzheng/OccWorld](https://github.com/wzzheng/OccWorld)  
**Area**: Autonomous Driving  
**Keywords**: 3D Occupancy, World Model, Autonomous Driving, 4D Forecasting, Trajectory Planning

## TL;DR
OccWorld proposes learning a world model in 3D occupancy space. It tokenizes 3D occupancy via VQ-VAE and predicts future scene evolution and ego-vehicle trajectories autoregressively using a GPT-style spatial-temporal generative Transformer, achieving competitive planning performance on nuScenes without requiring instance or HD map annotations.

## Background & Motivation

**Background**: The mainstream autonomous driving solutions follow a cascaded pipeline of "perception $\to$ prediction $\to$ planning" (e.g., UniAD, VAD). They rely on intermediate representations such as 3D detection bounding boxes, semantic maps, and motion prediction, requiring a vast amount of manual annotation at each stage.

**Limitations of Prior Work**: Traditional methods only predict the bounding box movement of target objects, failing to capture finer-grained 3D scene information (e.g., changes in road surface shapes and the evolution of drivable areas). Moreover, the cascaded design makes the training of each stage dependent on independent annotation data, leading to extremely high labeling costs.

**Key Challenge**: Autonomous driving requires understanding the temporal evolution of the entire 3D scene to make decisions. However, modeling at the instance level in existing frameworks falls short of covering all scene elements (e.g., static structures and drivable area changes).

**Goal**  
   - How to simultaneously predict both the evolution of the surrounding scene and the ego-vehicle motion using a unified model?  
   - How to perform planning without using instance bounding boxes and HD map annotations?  
   - How to efficiently model high-dimensional 3D occupancy sequences into predictable token sequences?

**Key Insight**: The authors observe that 3D occupancy possesses three major advantages: strong expressiveness (containing full 3D structure and semantics), high acquisition efficiency (can be generated from sparse LiDAR points), and cross-modal universality (applicable to both vision and LiDAR). Therefore, occupancy is utilized as the scene representation for the world model to replace bounding boxes.

**Core Idea**: Train a GPT-style spatial-temporal generative Transformer within the 3D occupancy space to autoregressively predict future scenes and ego-vehicle trajectories, eliminating the need for instance-level supervision.

## Method

### Overall Architecture
The input consists of past frames of 3D semantic occupancy ($\mathbf{y} \in \mathbb{R}^{H \times W \times D}$) and ego-vehicle poses, while the output yields future multi-frame 3D occupancy and ego-vehicle trajectories. The overall pipeline is split into two stages:
1. **Stage 1**: Train a 3D occupancy scene tokenizer (VQ-VAE) to compress high-dimensional occupancy into discrete tokens;
2. **Stage 2**: Freeze the tokenizer and train a spatial-temporal generative transformer to autoregressively predict future scene tokens and the ego token, which are then reconstructed into occupancy and trajectories via decoders.

### Key Designs

1. **3D Occupancy Scene Tokenizer (VQ-VAE)**:

    - **Function**: Compresses 3D occupancy into discrete token sequences, encoding high-level semantic concepts.
    - **Mechanism**: First, 3D occupancy is transformed into a BEV representation $\hat{\mathbf{y}} \in \mathbb{R}^{H \times W \times DC'}$ (concatenated along the height dimension) via class embedding, and then a 2D convolutional encoder downsamples it by a factor of $d$ to obtain $\hat{\mathbf{z}} \in \mathbb{R}^{H/d \times W/d \times C}$. Concurrently, a codebook $\mathbf{C} \in \mathbb{R}^{N \times C}$ is learned, performing nearest-neighbor quantization on each spatial feature: $\mathbf{z}_{ij} = \arg\min_{\mathbf{c} \in \mathbf{C}} \|\hat{\mathbf{z}}_{ij} - \mathbf{c}\|_2$. The decoder uses 2D deconvolution to restore the original resolution, splits along the channel dimension to recover the height dimension, and performs softmax classification to reconstruct the occupancy.
    - **Design Motivation**: The dimensionality of 3D occupancy is extremely high (200×200×16), making direct modeling difficult. VQ-VAE compresses it into 50×50 discrete tokens, which greatly reduces the complexity of subsequent Transformer modeling, while the codebook encodes reusable high-level scene concepts.

2. **Spatial-Temporal Generative Transformer**:

    - **Function**: Models the spatial-temporal evolutionary relationships of token sequences and autoregressively predicts the next-frame tokens.
    - **Mechanism**: It first performs spatial aggregation (self-attention) on each frame's tokens, and then merges them within a 2x2 window to form multi-scale tokens $\{\mathbf{T}_0, \ldots, \mathbf{T}_K\}$ ($K=3$ levels). For each scale, spatial-wise temporal causal attention is applied independently: $\hat{\mathbf{z}}_{j,i}^{T+1} = \text{TA}(\mathbf{z}_{j,i}^T, \ldots, \mathbf{z}_{j,i}^{T-t})$, which represents masked attention across time for tokens at the same spatial position to predict the future. Finally, a U-Net structure aggregates the multi-scale predictions to ensure spatial consistency.
    - **Design Motivation**: Directly predicting token-by-token like NLP-GPT is too slow due to the large number of tokens. Separating spatial-temporal modeling and incorporating a multi-scale design not only ensures global perception (spatial mixing) but also achieves efficient temporal prediction (position-wise temporal attention).

3. **Ego Token and Trajectory Decoding**:

    - **Function**: Unifies ego-vehicle motion within the token sequence, jointly modeling it with scene tokens.
    - **Mechanism**: An ego token $\mathbf{z}_0 \in \mathbb{R}^C$ is introduced to encode the position of the ego-vehicle, participating in both spatial aggregation and temporal attention alongside the scene tokens. The predicted ego token is decoded into displacement using an MLP ego decoder: $\hat{p}^{T+1} = d_{ego}(\hat{z}_0^{T+1})$.
    - **Design Motivation**: Traditional methods handle scene prediction and ego-vehicle planning separately, neglecting high-order interactions between the two. Integrating the ego token into the world model successfully captures the coupled relationship where "ego-vehicle movement leads to scene changes".

### Loss & Training
- **Stage 1 (Tokenizer)**: $J_{e,d} = L_{soft}(d(e(\mathbf{y})), \mathbf{y}) + \lambda_1 L_{lovasz}(d(e(\mathbf{y})), \mathbf{y})$, which consists of softmax CE + Lovász-softmax loss.
- **Stage 2 (World Model)**: $J_{w} = \sum_t \sum_j L_{soft}(\hat{\mathbf{z}}_{j,0}^t, \mathbf{C}(\mathbf{z}_{j,0}^t)) + \lambda_2 L_{L2}(d_{ego}(\hat{\mathbf{z}}_0^t), \mathbf{p}^t)$, which consists of token classification loss + trajectory L2 loss.
- During training, ground-truth tokens are used as input (teacher forcing), while during inference, autoregressively predicted tokens are advanced frame-by-frame.
- Uses 2s of history to predict 3s of future, batch=1/GPU, 8× RTX 4090.

## Key Experimental Results

### Main Results: 4D Occupancy Forecasting

| Method | Input | Extra Supervision | mIoU(1s) | mIoU(2s) | mIoU(3s) | mIoU(Avg) | IoU(Avg) |
|------|------|----------|----------|----------|----------|-----------|----------|
| Copy&Paste | 3D-Occ | None | 14.91 | 10.54 | 8.52 | 11.33 | 20.52 |
| **OccWorld-O** | 3D-Occ | None | **25.78** | **15.14** | **10.51** | **17.14** | **26.63** |
| OccWorld-D | Camera | 3D-Occ | 11.55 | 8.10 | 6.22 | 8.62 | 16.53 |
| OccWorld-T | Camera | Sem-LiDAR | 4.68 | 3.36 | 2.63 | 3.56 | 8.34 |

### Main Results: Motion Planning (L2 ↓ / Collision Rate ↓)

| Method | Input | Extra Supervision | L2 Avg(m) | Col. Avg(%) |
|------|------|----------|-----------|-------------|
| UniAD | Camera | Map+Box+Motion+Tracklets+Occ | 1.03 | 0.31 |
| VAD-Base | Camera | Map+Box+Motion | 1.22 | 0.53 |
| **OccWorld-O** | 3D-Occ | **None** | **1.17** | 0.60 |
| OccWorld-D | Camera | 3D-Occ | 1.40 | 0.87 |

### Ablation Study: Scene Tokenizer Hyperparameters

| Setting (Num Tokens², Feat Dim, Codebook) | Reconstruction mIoU | Prediction mIoU(Avg) | Planning L2(Avg) | FPS |
|----------------------------------|-----------|---------------|-------------|-----|
| (50², 128, 512) | 66.38 | 17.14 | 1.17 | 18.0 |
| (50², 128, 256) | 63.40 | 16.24 | 1.15 | 17.8 |
| (25², 256, 512) | 36.28 | 8.81 | 6.53 | 28.1 |
| (100², 128, 512) | 78.12 | 12.38 | 1.36 | 6.7 |

### Key Findings
- A token resolution of 50×50 is the optimal trade-off: setting it too small (25²) results in poor reconstruction quality, while setting it too large (100²) yields better reconstruction but degrades both prediction and planning, indicating that excessive tokens make Transformer modeling more difficult.
- OccWorld-O, under **zero extra supervision** (no bounding box or map annotations), surpasses VAD-Base (which requires Map+Box+Motion supervision) in planning performance.
- A codebook size of 512 is optimal; larger sizes (1024) lead to lower utilization and consequently reduce performance.
- The occupancy-based world model successfully predicts the movement of dynamic objects and changes in drivable areas, and even generates more reasonable drivable regions than the ground truth.

## Highlights & Insights
- **Occupancy as a Unified World Model Representation**: Replacing bounding boxes with occupancy as the operating target of the world model naturally incorporates both dynamic and static scene elements without separate modeling. This vision can be easily extended to fields requiring fine-grained 3D scene understanding, such as indoor robot navigation.
- **Scene Tokenization + GPT-Style Autoregression**: Quantizing the continuous 3D space into discrete tokens and predicting via the language modeling paradigm elegantly transforms "scene prediction" into a "sequence prediction" task, leveraging the strong capability of Transformers in sequence modeling.
- **Decomposed Spatial-Temporal Multi-Scale Design**: Utilizing spatial mixing + per-position temporal attention + U-Net fusion guarantees computational efficiency while maintaining global consistency, proving to be significantly more efficient than full spatial-temporal attention.

## Limitations & Future Work
- Out-of-view newly appearing vehicles cannot be predicted (as the input lacks information about these objects); combining with generative models or probabilistic forecasting might address this.
- Self-supervised occupancy (OccWorld-S) quality remains low, with a mIoU of only 0.26, indicating that current self-supervised occupancy methods are not yet mature.
- Validation is limited to the nuScenes dataset, presenting constrained scene diversity; scaling capabilities need to be verified on larger-scale datasets.
- The VQ-VAE reconstruction exhibits significant information loss (mIoU 66.38 vs. theoretical upper bound), capping the upper limit of prediction accuracy.
- Substituting the GPT autoregressive model with a diffusion model can be considered to avoid error accumulation in multi-step generation.

## Related Work & Insights
- **vs. UniAD**: UniAD adopts a cascaded "perception-prediction-planning" pipeline, requiring extensive intermediate annotations (boxes, maps, tracklets, motion, occ), while OccWorld merely utilizes occupancy without any instance-level annotation. Though UniAD shows slightly superior planning results (L2 1.03 vs. 1.17), it utilizes significantly more supervisory signals.
- **vs. MILE/GAIA**: These methods construct world models in 2D image space, lacking 3D comprehension. In contrast, OccWorld models in the 3D occupancy space, yielding outputs directly applicable to 3D planning.
- **vs. PointCloud Forecasting**: Methods like 4D-OCC and NTP forecast future point cloud frames, yet neglect semantic information and do not support vision-only inputs. OccWorld includes semantic occupancy, being fully compatible with both vision and LiDAR modalities.

## Rating
- Novelty: ⭐⭐⭐⭐ Proposes constructing a world model in 3D occupancy space for the first time; the combination of VQ-VAE + GPT is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed experiments and ablations are provided for both 4D forecasting and planning tasks.
- Writing Quality: ⭐⭐⭐⭐ The logical flow of the paper is clear, progressing systematically from motivation to methodology and experiments.
- Value: ⭐⭐⭐⭐⭐ Pioneered the direction of occupancy-based world models, with extensive follow-up works demonstrating its high impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Epona: Autoregressive Diffusion World Model for Autonomous Driving](../../ICCV2025/autonomous_driving/epona_autoregressive_diffusion_world_model_for_autonomous_driving.md)
- [\[ECCV 2024\] OccGen: Generative Multi-modal 3D Occupancy Prediction for Autonomous Driving](occgen_generative_multi-modal_3d_occupancy_prediction_for_autonomous_driving.md)
- [\[ECCV 2024\] Fully Sparse 3D Occupancy Prediction](fully_sparse_3d_occupancy_prediction.md)
- [\[ECCV 2024\] Neural Volumetric World Models for Autonomous Driving](neural_volumetric_world_models_for_autonomous_driving.md)
- [\[CVPR 2025\] GaussianWorld: Gaussian World Model for Streaming 3D Occupancy Prediction](../../CVPR2025/autonomous_driving/gaussianworld_gaussian_world_model_for_streaming_3d_occupancy_prediction.md)

</div>

<!-- RELATED:END -->
