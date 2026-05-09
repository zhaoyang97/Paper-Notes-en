---
title: >-
  [Paper Note] Generalizing Visual Geometry Priors to Sparse Gaussian Occupancy Prediction
description: >-
  [CVPR 2026][Autonomous Driving][Occupancy Prediction] GPOcc proposes leveraging generalizable visual geometry priors (e.g., VGGT, DepthAnything) for monocular 3D occupancy prediction. Surface points predicted by these priors are extended inward along camera rays to generate volumetric samples, which serve as centers of sparse Gaussian primitives for probabilistic occupancy inference. A training-free incremental update strategy handles streaming input. On Occ-ScanNet, GPOcc surpasses the previous SOTA by +9.99 mIoU (monocular) and +11.79 mIoU (streaming), while running 2.65× faster under the same depth prior.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Occupancy Prediction
  - Visual Geometry Prior
  - Gaussian Representation
  - Ray Sampling
  - Streaming Update
date: 2026-05-08
content_hash: a3dcd5fd016393a8
---

# Generalizing Visual Geometry Priors to Sparse Gaussian Occupancy Prediction

**Conference**: CVPR 2026
**arXiv**: [2602.21552](https://arxiv.org/abs/2602.21552)
**Code**: [https://github.com/JuIvyy/GPOcc](https://github.com/JuIvyy/GPOcc)
**Area**: Autonomous Driving
**Keywords**: Occupancy Prediction, Visual Geometry Prior, Gaussian Representation, Ray Sampling, Streaming Update

## TL;DR
GPOcc proposes leveraging generalizable visual geometry priors (e.g., VGGT, DepthAnything) for monocular 3D occupancy prediction. Surface points predicted by these priors are extended inward along camera rays to generate volumetric samples, which serve as centers of sparse Gaussian primitives for probabilistic occupancy inference. A training-free incremental update strategy handles streaming input. On Occ-ScanNet, GPOcc surpasses the previous SOTA by +9.99 mIoU (monocular) and +11.79 mIoU (streaming), while running 2.65× faster under the same depth prior.

## Background & Motivation
3D scene understanding is a core capability for embodied intelligence. Occupancy prediction, which provides a unified voxelized representation of foreground objects and background structures, has become a critical foundation for downstream tasks such as navigation, manipulation, and autonomous driving.

Fine-grained occupancy prediction in indoor scenes is more challenging than in outdoor autonomous driving scenarios, due to cluttered spatial layouts and diverse object categories. Existing methods such as ISO lift 2D features to dense 3D volumes via depth distributions and process them with a 3D U-Net, but dense representations incur significant computational waste on empty regions. EmbodiedOcc randomly initializes Gaussian primitives and refines them through iterative cross-attention, yet many Gaussians fall in empty regions, leading to low representational efficiency.

Meanwhile, visual geometry foundation models (VGMs)—including DepthAnything series and VGGT—are rapidly advancing, providing rich 3D priors such as depth, point maps, and camera parameters. **However, the outputs of these models are inherently surface-oriented**: depth maps and point maps are limited to visible surfaces, with each pixel corresponding to a single 3D surface point, leaving the volumetric interior unrepresented. How to transform "surface priors" into "volumetric priors" is the central unsolved problem.

The core idea of GPOcc is to extend predicted surface points inward along camera rays, generating volumetric sample points as Gaussian primitive centers, and to infer occupancy using a sparse Gaussian probabilistic formulation, while maintaining efficiency through opacity pruning.

## Method

### Overall Architecture
Given a single RGB image, a visual geometry prior model (VGGT or DepthAnything) predicts surface points and extracts 3D-aware features. A ray-based volumetric sampling module extends surface points inward along camera rays to generate sample points serving as Gaussian centers. The extracted features are combined with learnable embeddings and passed through an MLP to predict Gaussian attributes (scale, rotation, opacity, semantic features). After opacity pruning, sparse Gaussians are splatted into voxel occupancy via a probabilistic formulation. For streaming scenes, a training-free incremental update strategy integrates per-frame Gaussians into a global memory bank.

### Key Designs
1. **Ray-based Volumetric Sampling**:

    - **Function**: Overcomes the limitation that geometry priors predict only visible surfaces, by generating volumetric samples inside objects.
    - **Mechanism**: Given depth $\mathbf{d}_{(u,v)}$ and normalized ray direction $\mathbf{r}_{(u,v)} = \frac{[x, y, 1]^\top}{\sqrt{x^2+y^2+1}}$ for pixel $(u,v)$, $K$ points are sampled along the ray beyond the surface point: $\mathbf{x}_{(u,v,k)} = (\mathbf{d}_{(u,v)} + \delta_k) \mathbf{r}_{(u,v)}$, where $\{\delta_k\}_{k=1}^K = \text{linspace}(0,1,K) \cdot \text{scale}(\cdot)$, with scale dynamically predicted by the network to accommodate objects of varying sizes.
    - **Feature Extraction**: A learnable embedding $\mathbf{E} \in \mathbb{R}^{K \times C}$ is introduced and broadcast-added to the downsampled feature map: $\hat{\mathbf{F}}^{1/4} = \mathbf{F}^{1/4} \oplus \mathbf{E}$. Gaussian attributes $\{s_i, r_i, a_i, c_i\} = \text{MLP}(\hat{\mathbf{F}}^{1/4})$ are then predicted via MLP.
    - **Design Motivation**: Surface priors cannot capture the true thickness or volume of objects. Extending inward along rays is the most natural way to transition from 2D to 3D, avoiding the overhead of dense 3D anchor grids or full volumetric lifting.

2. **Opacity Pruning + Sparse Gaussian-to-Occupancy**:

    - **Function**: Infers voxel-level occupancy from sparse Gaussian primitives while removing invalid Gaussians to preserve efficiency.
    - **Mechanism**: A probabilistic Gaussian accumulation formulation (from GaussianFormer2) is adopted: $\hat{o}(p; \mathbf{G}) = \sum_{i \in \mathcal{N}(p)} g_i(p; \mu_i, s_i, r_i, a_i, c_i)$, where $o(p; \mathcal{G}_i) = \exp(-\frac{1}{2}(p-\mu_i)^\top \Sigma_i^{-1}(p-\mu_i))$. Gaussians with opacity below threshold $\tau = 0.01$ are pruned.
    - **Design Motivation**: Unlike EmbodiedOcc, which predefined dense 3D Gaussian anchors and then performs per-anchor classification, GPOcc's Gaussians naturally concentrate on object surfaces and interiors. Empty regions, being uncovered by any Gaussian, are automatically classified as empty, yielding far greater efficiency than dense schemes. Regions far from all Gaussians naturally become empty voxels.

3. **Training-Free Incremental Update Strategy**:

    - **Function**: Extends single-frame Gaussian predictions to streaming video input, progressively fusing them into a globally consistent scene representation.
    - **Mechanism**: A global Gaussian memory bank $\mathcal{M}$ is maintained. Per-frame predicted Gaussians are transformed to world coordinates via camera poses, then spatially searched against the memory bank within radius $\epsilon$. If neighbors are found, a weighted-average fusion is applied: $\theta_i \leftarrow \frac{\gamma p_i \theta_i + (1-\gamma) \sum_j p_j \theta_j}{\gamma p_i + (1-\gamma) \sum_j p_j}$, where $\theta \in \{\mu, \Sigma, a, c\}$, and $\gamma < 0.5$ assigns higher weight to the incoming frame. New Gaussians with no neighbors are directly inserted into the memory bank.
    - **Design Motivation**: Streaming input is handled without retraining. The weighted fusion naturally incorporates uncertainty-awareness and temporal smoothing, while the top-1 class confidence $p$ as weight introduces a semantic consistency prior.

### Loss & Training
- Composite loss function: $\mathcal{L} = L_{\text{focal}} + L_{\text{lov}} + L_{\text{scal}}^{\text{geo}} + L_{\text{scal}}^{\text{sem}} + L_{\text{depth}}$
    - $L_{\text{focal}}$: Focal loss to address class imbalance.
    - $L_{\text{lov}}$: Lovász-Softmax loss to optimize IoU.
    - $L_{\text{scal}}^{\text{geo/sem}}$: Scene-class affinity loss (geometric + semantic).
    - $L_{\text{depth}}$: Huber depth loss for end-to-end geometric consistency optimization (unlike EmbodiedOcc, which relies on an external pretrained depth estimator).
- Training: AdamW (weight decay 0.01), 10 epochs, batch size 8, 4× A800 GPUs, cosine learning rate decay to $2 \times 10^{-4}$.
- Input images are resized so that the longer edge is 518 px; gradient clipping at 1.0.

## Key Experimental Results

### Main Results

| Dataset | Metric | GPOcc-VGGT | GPOcc-DPT | EmbodiedOcc++ | Gain (VGGT) |
|--------|------|-----------|-----------|---------------|------------|
| Occ-ScanNet (Monocular) | IoU↑ | **63.14** | 56.96 | 54.90 | +8.24 |
| Occ-ScanNet (Monocular) | mIoU↑ | **56.19** | 51.88 | 46.20 | +9.99 |
| EmbodiedOcc-ScanNet (Streaming) | IoU↑ | **61.41** | 56.39 | 52.20 | +9.21 |
| EmbodiedOcc-ScanNet (Streaming) | mIoU↑ | **55.39** | 51.22 | 43.60 | +11.79 |

### Efficiency Comparison (Occ-ScanNet)

| Model | IoU | mIoU | FPS | Parameters |
|------|-----|------|-----|--------|
| ISO | 42.16 | 28.71 | 3.63 | 303.05M |
| EmbodiedOcc | 53.55 | 45.15 | 10.66 | 231.45M |
| **Ours-DPT** | **56.96** | **51.88** | **28.22** | **97.95M** |
| Ours-VGGT | 63.14 | 56.19 | 5.26 | 942.31M |

### Ablation Study

| Configuration | mIoU | IoU | #Gaussians | Note |
|------|------|-----|------------|------|
| K=1 (surface only) | 47.88 | 53.10 | 3079 | No interior sampling; worst performance |
| K=4 | 55.28 | 60.35 | 2731 | Interior sampling yields large gains |
| K=16 (default) | **56.19** | **63.14** | 5876 | Accuracy saturates; optimal efficiency point |
| K=32 | 56.72 | 63.84 | 20206 | Diminishing returns |
| τ=0.01 (default) | **56.19** | **63.14** | 5876 | Optimal threshold |
| τ=0.05 | 54.16 | 60.84 | 1612 | Excessive pruning |
| τ=0.10 | 52.65 | 58.31 | 930 | Severe accuracy degradation |

### Key Findings
- Under the same depth prior (DepthAnything), GPOcc-DPT runs 2.65× faster than EmbodiedOcc (28.22 vs. 10.66 FPS), achieves +6.73 higher mIoU, and uses less than half the parameters (97.95M vs. 231.45M)—conclusively demonstrating the architectural efficiency of ray sampling combined with sparse Gaussians.
- From K=1 (surface only) to K=16, mIoU improves by +8.31 and IoU by +10.04, confirming the necessity of volumetric interior sampling.
- A stronger geometry prior (VGGT vs. DPT) consistently yields additional gains (+4.31 mIoU), indicating that the framework can fully benefit from more powerful foundation models.
- Opacity pruning at τ=0.01 effectively controls the number of Gaussians with virtually no loss in accuracy.

## Highlights & Insights
- "Extending inward along rays" is the most natural approach to converting surface priors into volumetric priors—elegant and effective.
- Sparse Gaussians naturally concentrate in object regions; areas uncovered by any Gaussian are automatically treated as empty voxels, eliminating the substantial waste of dense schemes.
- The training-free incremental update strategy is elegantly designed: spatial proximity fusion, confidence-weighted aggregation, and higher weighting for incoming frames collectively enable extension to streaming scenes without additional training.
- The framework exhibits strong compatibility with different geometry prior models, enabling "free" performance improvements as foundation models advance.

## Limitations & Future Work
- The VGGT variant has a large parameter count (942.31M) and only achieves 5.26 FPS, falling short of real-time deployment requirements.
- Ray sampling assumes objects have a certain depth behind their surface (predicted via scale), which may be suboptimal for thin structures such as curtains or walls.
- The spatial radius $\epsilon$ and temporal weight $\gamma$ in the incremental update strategy are manually set hyperparameters.
- Validation is limited to the indoor ScanNet dataset; generalization to outdoor or large-scale scenes remains unexplored.

## Related Work & Insights
- The key contrast with EmbodiedOcc lies in predefined dense anchors versus ray-guided sparse Gaussians; the latter outperforms the former comprehensively in both efficiency and accuracy.
- The rapid development of visual geometry foundation models such as VGGT, DUSt3R, and MASt3R provides strong geometry priors for this work; GPOcc demonstrates how to effectively exploit these priors.
- The probabilistic occupancy formulation from GaussianFormer2 is directly adopted, confirming the generality of Gaussian representations for occupancy prediction.
- The paradigm generalizes naturally: any foundation model providing 3D surface information can serve as the front-end for GPOcc.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of ray-based volumetric sampling and sparse Gaussian occupancy is original, though individual components (ray sampling, Gaussian splatting, incremental fusion) are not entirely new in isolation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Two datasets × two priors, plus detailed ablations over K, τ, and efficiency comparisons; the experimental design is rigorous and comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Method motivation is clearly articulated; Figure 1's three-way method comparison is immediately intuitive; mathematical derivations are complete.
- **Value**: ⭐⭐⭐⭐⭐ The DPT variant's 28 FPS throughput and 97.95M parameter count make it practically deployable; compatibility with diverse priors ensures long-term relevance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] O3N: Omnidirectional Open-Vocabulary Occupancy Prediction](o3n_omnidirectional_open-vocabulary_occupancy_prediction.md)
- [\[CVPR 2026\] SparseWorld-TC: Trajectory-Conditioned Sparse Occupancy World Model](sparseworld_tc_trajectory_conditioned_sparse_occupancy_world_model.md)
- [\[AAAI 2026\] Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction](../../AAAI2026/autonomous_driving/visiononly_gaussian_splatting_for_collaborative_semantic_occupancy_p.md)
- [\[CVPR 2026\] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction.md)
- [\[ICCV 2025\] GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow](../../ICCV2025/autonomous_driving/gaussianflowocc_sparse_and_weakly_supervised_occupancy_estimation_using_gaussian.md)

</div>

<!-- RELATED:END -->
