---
title: >-
  [Paper Note] OccGen: Generative Multi-modal 3D Occupancy Prediction for Autonomous Driving
description: >-
  [ECCV 2024][Autonomous Driving][3D Occupancy Prediction] OccGen reformulates 3D semantic occupancy prediction into a generative "noise-to-occupancy" paradigm. It extracts multi-modal features via a conditional encoder and performs diffusion denoising using a progressive refinement decoder to step-by-step generate occupancy maps in a coarse-to-fine manner. It relatively improves mIoU by 9.5%, 6.3%, and 13.3% under multi-modal, LiDAR-only, and camera-only settings on nuScenes-O…
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "3D Occupancy Prediction"
  - "diffusion model"
  - "Multi-modal Fusion"
  - "Generative Perception"
  - "Coarse-to-Fine Refinement"
date: 2026-05-08
content_hash: 1429d2e48eaa2c2d
---

# OccGen: Generative Multi-modal 3D Occupancy Prediction for Autonomous Driving

**Conference**: ECCV 2024  
**arXiv**: [2404.15014](https://arxiv.org/abs/2404.15014)  
**Code**: [https://occgen-ad.github.io/](https://occgen-ad.github.io/)  
**Area**: Autonomous Driving  
**Keywords**: 3D Occupancy Prediction, diffusion model, Multi-modal Fusion, Generative Perception, Coarse-to-Fine Refinement

## TL;DR

OccGen reformulates 3D semantic occupancy prediction into a generative "noise-to-occupancy" paradigm. It extracts multi-modal features via a conditional encoder and performs diffusion denoising using a progressive refinement decoder to step-by-step generate occupancy maps in a coarse-to-fine manner. It relatively improves mIoU by 9.5%, 6.3%, and 13.3% under multi-modal, LiDAR-only, and camera-only settings on nuScenes-Occupancy, respectively.

## Background & Motivation

**Background**: 3D semantic occupancy prediction is a core perception task in autonomous driving, aiming to assign a semantic label to each voxel within the perception range, which retains vertical dimension details better than BEV representations.

**Limitations of Prior Work**: Existing methods (LiDAR-based, vision-based, multi-modal) formulate occupancy prediction as a one-shot voxel segmentation problem, completing prediction with a single forward pass. However, these discriminative methods suffer from two limitations: (1) they only learn the mapping from input to output, neglecting the distribution modeling of occupancy maps; (2) a single forward pass does not suffice to complete fine-grained structures.

**Key Challenge**: Discriminative methods lack step-by-step refinement and scene imagination, making them unable to refine their perception of the entire scene through persistent observation like humans do.

**Goal**: How to introduce a coarse-to-fine step-by-step refinement paradigm into occupancy prediction while supporting uncertainty estimation.

**Key Insight**: The denoising process of diffusion models naturally models the coarse-to-fine refinement of occupancy maps, progressively generating accurate occupancy predictions from random Gaussian noise.

**Core Idea**: OccGen is proposed, adopting a generative "noise-to-occupancy" paradigm where the conditional encoder runs only once, and the decoder progressively refines the prediction over multiple iterations, achieving inference latency comparable to one-shot methods.

## Method

### Overall Architecture

OccGen is a generative perception model based on an encoder-decoder structure. The conditional encoder extracts conditional features by processing multi-modal inputs (running only once), while the progressive refinement decoder uses these features as conditions to step-by-step refine and generate occupancy predictions from a 3D Gaussian noise map through diffusion denoising. During training, Gaussian noise is added to the ground-truth (GT) occupancy map to learn denoising; during inference, the model starts from pure Gaussian noise and undergoes progressive denoising to generate the final occupancy map.

### Key Designs

1. **Noise-to-Occupancy Generative Paradigm**:

    - Function: Formulates occupancy prediction as a progressive generation process from noise to occupancy.
    - Mechanism: Refines step-by-step through a $T$-step diffusion process $Y_T \xrightarrow{f_\theta} Y_{T-1} \xrightarrow{f_\theta} \ldots \rightarrow Y_0$, where each step predicts and removes noise using model $f_\theta$. The forward diffusion process is defined as $z_t = \sqrt{\alpha_t} z_0 + \sqrt{1-\alpha_t} Z$, where $Z \sim \mathcal{N}(0, I)$. The refinement formula for each step is $\Delta Y_t = f_\theta(x, t, Y_{t+1})$ and $Y_t = Y_{t+1} \oplus \Delta Y_t$.
    - Design Motivation: The diffusion denoising process naturally models the coarse-to-fine refinement of 3D occupancy maps instead of one-shot generation. It also naturally supports flexible tradeoffs between accuracy and computation, and enables uncertainty estimation.

2. **Conditional Encoder**:

    - Function: Fuses multi-modal features from LiDAR and camera inputs to generate condition information for the decoder.
    - Mechanism: Employs a dual-stream architecture. The LiDAR stream extracts voxel features $F_p$ using VoxelNet + 3D sparse convolutions. The camera stream extracts multi-view image features using a 2D backbone + FPN, followed by a Hard 2D-to-3D view transformer to obtain camera voxel features $F_c$. The fusion module adopts adaptive weighting: $W = \mathcal{G}_C([\mathcal{G}_C(F_p), \mathcal{G}_C(F_c)])$ and $F_m = \sigma(W) \odot F_p + (1-\sigma(W)) \odot F_c$.
    - Design Motivation: (1) Hard 2D-to-3D view transformation replaces traditional softmax depth estimation with Gumbel-Softmax to guarantee more precise depth predictions (differentiable one-hot encoding); (2) Geometry Mask utilizes LiDAR voxel features to generate a mask that constrains the spatial distribution of camera features, compensating for depth ambiguity in camera features.

3. **Progressive Refinement Decoder**:

    - Function: Performs multiple iterations of denoising refinement on the noisy occupancy map guided by conditional features.
    - Mechanism: The decoder consists of multiple Refinement Layers and an occupancy head. Each layer contains three core operations: (1) 3D Deformable Cross-Attention (DCA) to learn features from conditional inputs: $\text{DCA}_{3D}(Y_t^i, F_m) = \sum_{n \in F_m} \text{DA}_{3D}(q, \text{proj}(q,n), F_m)$; (2) 3D Deformable Self-Attention (DSA) to enhance self-completion capability: $\text{DSA}_{3D}(Y_t^i, Y_t^i)$; (3) Temporal Diffusion Module which performs a scale-and-shift operation on the noisy map utilizing the step index $t$ embedding: $Y_t^{i} := \text{Diff}(Y_t^i, \text{ToEmbed}(t))$. For efficiency, the noisy map is first downsampled to multi-scale dimensions $Y_t^i \in \mathbb{R}^{D/2^i \times H/2^i \times W/2^i \times C_i}$.
    - Design Motivation: Running the encoder only once while iterating through the decoder for multiple steps prevents significant computational overhead, keeping the inference latency comparable to one-shot forward methods.

### Loss & Training

During training, a cosine noise schedule is used to add Gaussian noise to the GT occupancy map (empirically found to outperform the linear schedule). The total loss function is:

$$\mathcal{L}_{total} = \mathcal{L}_{ce} + \mathcal{L}_{ls} + \mathcal{L}_{scal}^{geo} + \mathcal{L}_{scal}^{sem} + \mathcal{L}_d$$

where $\mathcal{L}_{ce}$ is the cross-entropy loss, $\mathcal{L}_{ls}$ is the lovász-softmax loss, $\mathcal{L}_{scal}^{geo}$ and $\mathcal{L}_{scal}^{sem}$ are scene-wise and class-wise affinity losses, and $\mathcal{L}_d$ is the depth loss. During inference, the DDIM strategy and asymmetric time steps ($td=1$) are adopted.

## Key Experimental Results

### Main Results

Results on nuScenes-Occupancy validation set:

| Setting | Method | IoU | mIoU | Gain |
|------|------|-----|------|---------|
| Multi-modal | CONet | 29.5 | 20.1 | - |
| Multi-modal | **OccGen** | **30.3** | **22.0** | +9.5% |
| LiDAR-only | L-CONet | 30.9 | 15.8 | - |
| LiDAR-only | **L-OccGen** | **31.6** | **16.8** | +6.3% |
| Camera-only | C-CONet | 20.1 | 12.8 | - |
| Camera-only | **C-OccGen** | **23.4** | **14.5** | +13.3% |

Results on SemanticKITTI validation set:

| Method | IoU | mIoU |
|------|-----|------|
| OccFormer | 36.50 | 13.46 |
| Symphonize | 41.44 | 13.44 |
| **OccGen** | **36.87** | **13.74** |

### Ablation Study

| Configuration | IoU | mIoU | Description |
|------|-----|------|------|
| Baseline | 28.1 | 20.4 | No encoder improvement + No decoder improvement |
| + Proposed Encoder | 28.6 | 20.7 | Hard LSS + Geo Mask |
| + Proposed Decoder | 30.1 | 21.6 | Progressive Refinement Decoder |
| Complete OccGen | 30.3 | 22.0 | Encoder + Decoder |
| w/o DSA | 30.1 | 21.4 | Remove self-attention |
| w/o Diffusion | 29.3 | 21.7 | Remove diffusion denoising |

### Key Findings

- The decoder (progressive refinement) contributes more than the encoder improvements, indicating that the generative paradigm itself is the key driver of performance gain.
- Cross-attention has a larger impact than self-attention, as learning representations from conditional inputs is more critical.
- Removing the diffusion module drops the mIoU from 22.0% to 21.7%, proving the necessity of the temporal diffusion process.
- Multi-step inference consistently boosts performance: 1 step (21.7%) → 3 steps (22.0%), allowing a trade-off adjustment between computation and accuracy without retraining.
- Multi-modal OccGen outperforms camera-only by 7.5% mIoU, and LiDAR-only by 5.8% mIoU.

## Highlights & Insights

- **First to Introduce Diffusion Models to 3D Occupancy Prediction**: Redefines the traditional discriminative task as a generative task, empowering the model with step-by-step refinement and uncertainty estimation capabilities.
- **Efficient Inference Design**: The encoder only runs once while the decoder iterates through multiple steps, achieving an inference latency comparable to one-shot methods (approx. 294ms vs. 286ms for CONet).
- **Uncertainty Estimation**: The stochastic sampling process naturally allows the computation of voxel-level prediction uncertainty, which is unattainable for discriminative methods.
- **Hard 2D-to-3D View Transformation**: Implements Gumbel-Softmax for differentiable one-hot depth encoding, which is more accurate than traditional softmax.

## Limitations & Future Work

- The accuracy boost brought by multi-step inference is limited (3-step only improves by 0.3% mIoU), and performance saturate with more steps.
- The current generation process operates directly in voxel space; exploring latent space diffusion could be more efficient.
- The comparison with the latest vision-only methods (e.g., FlashOcc, SparseOcc) is insufficient.
- The multi-modal fusion strategy is relatively simple (adaptive weighting); more complex interaction mechanisms could be explored.

## Related Work & Insights

- **vs. CONet**: OccGen replaces the discriminative approach with a generative one, comprehensively outperforming it across all three modalities and demonstrating the advantage of the diffusion paradigm for dense prediction.
- **vs. DiffusionDet / DDP**: Adopts the diffusion "noise-to-X" paradigm but extends it to the sparse voxel space of 3D occupancy, addressing the high-resolution computation challenge in 3D scenes.
- **vs. MonoScene / VoxFormer**: Unlike one-shot forward methods, OccGen can flexibly trade off computation and accuracy while providing uncertainty information.

## Rating

- Novelty: ⭐⭐⭐⭐ The first study to introduce diffusion models to 3D occupancy prediction. While the generative paradigm is novel, applying diffusion to perception tasks is not entirely unprecedented.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluations on two benchmarks with detailed ablation studies; however, it lacks comparisons with a broader range of the latest methods.
- Writing Quality: ⭐⭐⭐⭐ Extremely clear, with detailed method descriptions and complete mathematical derivations.
- Value: ⭐⭐⭐⭐ Provides a fresh generative perspective for occupancy prediction; both uncertainty estimation and flexible inference hold practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Gau-Occ: Geometry-Completed Gaussians for Multi-Modal 3D Occupancy Prediction](../../CVPR2026/autonomous_driving/gau-occ_geometry-completed_gaussians_for_multi-modal_3d_occupancy_prediction.md)
- [\[ECCV 2024\] UniM2AE: Multi-modal Masked Autoencoders with Unified 3D Representation for 3D Perception in Autonomous Driving](unim2ae_multi-modal_masked_autoencoders_with_unified_3d_representation_for_3d_pe.md)
- [\[ECCV 2024\] GraphBEV: Towards Robust BEV Feature Alignment for Multi-Modal 3D Object Detection](graphbev_towards_robust_bev_feature_alignment_for_multi-modal_3d_object_detectio.md)
- [\[ECCV 2024\] OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving](occworld_learning_a_3d_occupancy_world_model_for_autonomous_driving.md)
- [\[ECCV 2024\] Fully Sparse 3D Occupancy Prediction](fully_sparse_3d_occupancy_prediction.md)

</div>

<!-- RELATED:END -->
