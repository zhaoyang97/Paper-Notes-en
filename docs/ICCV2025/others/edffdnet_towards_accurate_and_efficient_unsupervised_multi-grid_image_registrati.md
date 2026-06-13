---
title: >-
  [Paper Note] EDFFDNet: Towards Accurate and Efficient Unsupervised Multi-Grid Image Registration
description: >-
  [ICCV 2025][image registration] This paper proposes EDFFDNet, which replaces conventional B-spline FFD and TPS with an Exponentially Decaying Free-Form Deformation (EDFFD) model for image registration. Combined with an A…
tags:
  - "ICCV 2025"
  - "image registration"
  - "free-form deformation"
  - "exponentially decaying basis function"
  - "sparse motion aggregation"
  - "unsupervised learning"
date: 2026-05-08
content_hash: c5d6ed8c69893513
---

# EDFFDNet: Towards Accurate and Efficient Unsupervised Multi-Grid Image Registration

**Conference**: ICCV 2025
**arXiv**: [2509.07662](https://arxiv.org/abs/2509.07662)  
**Code**: N/A  
**Area**: Other (Image Registration)
**Keywords**: image registration, free-form deformation, exponentially decaying basis function, sparse motion aggregation, unsupervised learning

## TL;DR

This paper proposes EDFFDNet, which replaces conventional B-spline FFD and TPS with an Exponentially Decaying Free-Form Deformation (EDFFD) model for image registration. Combined with an Adaptive Sparse Motion Aggregator (ASMA) and a progressive correlation strategy, the method achieves a +0.5 dB PSNR improvement on the UDIS-D dataset while reducing parameter count by 70.5% and GPU memory usage by 32.6%.

## Background & Motivation

### Problem Definition

Image registration is a fundamental task in computer vision that requires establishing spatial correspondences between images captured under different conditions. In practice, images often contain depth disparities—scenes with multiple planes or foreground/background objects—making it inherently impossible for a single homography to accurately align all regions.

### Limitations of Prior Work

**Single homography**: Constrained by the planar assumption, it fundamentally cannot achieve accurate alignment in non-planar scenes.

**Multi-grid homography**: Merely an extension of homography that does not resolve its intrinsic representational limitations.

**TPS (Thin Plate Spline)**: Constructs a globally smooth deformation field without local support, performing poorly in scenarios requiring significant local deformation.

**B-spline FFD**: Although offering better locality, the cubic B-spline basis functions incur high computational cost; piecewise computation impedes GPU parallelism, and basis product operations must be computed independently along two dimensions.

**MLP-based motion aggregators**: Parameter-heavy by nature—UDIS++ employs an MLP aggregator with 68.9M parameters, limiting practical deployment.

**Global correlation in local refinement**: Introduces long-range interference during local refinement stages, degrading accuracy.

### Core Motivation

There is a need for a deformation model that retains the locality advantages of B-spline FFD while being computationally efficient and GPU-friendly, complemented by a lightweight motion aggregation scheme and a well-designed correlation strategy.

**Core Idea**: Replace cubic B-splines with exponentially decaying functions as FFD basis functions, adopt grouped linear layers for sparse motion aggregation, and apply a coarse-to-fine progressive correlation strategy.

## Method

### Overall Architecture

EDFFDNet consists of three main modules: (1) a Multi-scale Feature Extractor (MFE) based on ResNet50, which extracts features from the target and reference images at 4×/8×/16× downsampling; (2) a global homography estimation module using global correlation for initial alignment; and (3) one or two local refinement stages employing the EDFFD model for local deformation estimation. The final deformation field is obtained by composing the global homography with residual displacements from each refinement stage.

### Key Designs

#### 1. **Exponentially Decaying Free-Form Deformation (EDFFD)**

- **Function**: Replaces the cubic B-spline basis functions with exponentially decaying functions to compute the influence of control points on spatial locations.
- **Core formula**:

$$\mathbf{x}' = \mathbf{x} + \sum_{m=0}^{M_i}\sum_{n=0}^{N_i} \Delta\mathbf{p}_{m,n} \exp(-r_{m,n}/(\theta\eta))$$

where $r_{m,n} = \|\mathbf{x} - \mathbf{p}_{m,n}\|_2$ is the Euclidean distance, $\theta$ controls the decay rate, and $\eta$ is the grid spacing.

In contrast, the conventional B-spline FFD relies on piecewise cubic polynomials:

$$\beta^3(u) = \begin{cases} \frac{2}{3} - |u|^2 + \frac{|u|^3}{2}, & 0 \leq |u| \leq 1 \\ \frac{(2-|u|)^3}{6}, & 1 \leq |u| < 2 \end{cases}$$

- **Design Motivation**:
    - **Simplified influence metric**: Euclidean distance directly replaces the two-dimensional basis product computation.
    - **Computational efficiency**: The exponential function provides $C^\infty$ smoothness at lower cost than cubic polynomials; hardware-optimized transcendental function units on GPUs provide additional acceleration.
    - **Parallelism-friendly**: The non-piecewise nature enables fully parallel computation across all spatial locations, avoiding the conditional branching that limits GPU utilization in B-splines.
    - **Preserved locality**: The exponential function naturally decays significantly with distance.
    - Experiments identify $\theta = 0.75$ as the optimal trade-off point.

#### 2. **Adaptive Sparse Motion Aggregator (ASMA)**

- **Function**: Replaces the MLP for motion parameter aggregation by converting dense interactions into sparse ones.
- **Core structure**: Two Grouped Linear Layers (GLL) followed by a single linear layer.
    - GLL partitions the input feature $\mathbf{F}_c \in \mathbb{R}^{C_c}$ into $N_g$ groups.
    - Each group undergoes an independent linear transformation: $\mathbf{F}'_{g,k} = \mathbf{W}_k(\mathbf{F}_{g,k}) + \mathbf{b}_k$.
    - The concatenated output passes through ReLU, and a final linear layer adaptively fuses the features and outputs motion parameters.
- **Design Motivation**: Inspired by depthwise separable convolutions, grouped linear layers convert dense connections into sparse ones. At $N_g = 8$, parameters are reduced by 66.6% (23.0M vs. 68.9M) while accuracy improves (PSNR 25.93 vs. 25.87).

#### 3. **Progressive Correlation Strategy**

- **Function**: Global correlation is used in the homography estimation stage; local correlation is applied in the refinement stages.
- **Global correlation**: Patch-to-patch correlation using $K \times K$ dense patches as convolution kernels:

$$\mathbf{C}^g_{(x_r,y_r,x_t,y_t)} = \sum_{i,j} \frac{\langle \mathbf{F}^{(d)}_{r,(x_r+i,y_r+j)}, \mathbf{F}^{(d)}_{t,(x_t+i,y_t+j)} \rangle}{\|\mathbf{F}^{(d)}_{r,(x_r+i,y_r+j)}\| \|\mathbf{F}^{(d)}_{t,(x_t+i,y_t+j)}\|}$$

- **Local correlation**: Computed within a local region of radius $r$ centered at $\mathbf{p}'$.
- **Design Motivation**: Global homography requires a large search range to handle low-overlap scenes, whereas global correlation in the local refinement stage introduces long-range interference. The progressive strategy provides sufficient search range in the global stage and focuses on local context during refinement, yielding a +0.39 dB PSNR gain while reducing inference time by 29.8%.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{content}} + \omega \mathcal{L}_{\text{shape}}$$

- **Content alignment loss**: Bidirectional L1 loss (forward + backward warp), including global and local stage terms with weights $\lambda_0 = 1, \lambda_1 = 1.3, \lambda_2 = 1.7$.
- **Shape preservation loss**: Intra-grid constraint (limiting edge lengths) + inter-grid constraint (encouraging collinearity of adjacent edges in non-overlapping regions); $\omega = 10$.
- **Two-stage training**: The global homography module is trained for 10 epochs, followed by joint training of global and local modules for 100 epochs.
- Adam optimizer, lr=$10^{-4}$, batch size 4.

## Key Experimental Results

### Main Results

Warp accuracy comparison on the UDIS-D dataset:

| Method | PSNR↑ (Easy) | PSNR↑ (Hard) | PSNR↑ (Avg) | SSIM↑ (Avg) |
|--------|-------------|-------------|-------------|-------------|
| SIFT+RANSAC | 27.75 | 18.46 | 22.98 | 0.758 |
| APAP | 27.01 | 19.54 | 23.00 | 0.773 |
| ELA | 29.87 | 19.68 | 24.47 | 0.821 |
| UDIS | 27.84 | 20.70 | 23.80 | 0.793 |
| MGDH | 29.52 | 21.20 | 24.89 | 0.817 |
| UDIS++ | 30.19 | 21.57 | 25.43 | 0.838 |
| **EDFFDNet** | **30.63** | **22.15** | **25.93** | **0.852** |
| **EDFFDNet-2** | **31.09** | **22.79** | **26.49** | **0.868** |

Computational cost comparison:

| Method | PSNR | Params (M) | Memory (GB) | Total Time (ms) |
|--------|------|-----------|------------|----------------|
| UDIS | 23.80 | 188.8 | 7.1 | 66.7 |
| MGDH | 24.89 | 16.4 | 5.3 | 90.3 |
| UDIS++ | 25.43 | 78.0 | 4.6 | 65.8 |
| **EDFFDNet** | **25.93** | **23.0** | **3.1** | **43.6** |
| **EDFFDNet-2** | **26.49** | **34.5** | **4.3** | **55.1** |

### Ablation Study

**Deformation model comparison**:

| Model | PSNR | SSIM | Warp Time (ms) | Memory (GB) |
|-------|------|------|---------------|------------|
| TPS | 25.49 | 0.838 | 30.5 | 3.3 |
| B-spline FFD | 25.95 | 0.850 | 39.1 | 4.7 |
| **EDFFD** | **25.93** | **0.852** | **20.6** | **3.1** |

**Motion aggregator comparison** ($N_g = 8$):

| Method | Params (M) | PSNR | SSIM |
|--------|-----------|------|------|
| MLP | 68.9 | 25.87 | 0.850 |
| MLP (4× compressed) | 27.1 | 25.76 | 0.845 |
| **ASMA** | **23.0** | **25.93** | **0.852** |

### Key Findings

- **EDFFD matches B-spline FFD accuracy while warping 47.3% faster and using 34% less memory**: The exponential decay function substantially reduces computational overhead while preserving locality.
- **ASMA achieves 66.6% fewer parameters yet higher accuracy**: Sparse interaction proves more effective than dense interaction for motion aggregation; even when the MLP is compressed to a comparable parameter count, ASMA remains superior.
- **Balancing effect of locality factor $\theta$**: $\theta = 0.25$ (too small) causes the exponential to decay slowly, resulting in overly large influence regions; $\theta > 1.0$ (too large) restricts influence regions excessively; $\theta = 0.75$ is optimal.
- **Strong zero-shot cross-dataset generalization**: On ScanNet, EDFFDNet-2 achieves PSNR 24.32 (vs. UDIS++ 21.79); on ETH3D, 21.47 (vs. 19.41), demonstrating substantial advantages.
- **Far faster than traditional methods**: Processing a 1500×2000 image takes only 0.078s (vs. APAP 159.6s and LPC 2114.9s).

## Highlights & Insights

1. **Engineering insight in basis function design**: The three computational bottlenecks of B-splines (high-degree polynomials, basis products, piecewise evaluation) are simultaneously resolved by the exponential function, reflecting a deep understanding of GPU computational characteristics.
2. **Counter-intuitive finding that sparse outperforms dense**: ASMA's grouped sparse interactions not only reduce parameters but also improve accuracy, suggesting that excessive global interaction in motion aggregation introduces noise rather than useful signal.
3. **Rationale for progressive correlation**: Using global search for coarse alignment and local search for fine-grained refinement follows the classical coarse-to-fine paradigm in a principled manner.
4. **Efficiency of the additional local refinement stage**: The transition from EDFFDNet to EDFFDNet-2 incurs only 11.5 ms additional warp time while yielding a +0.56 dB PSNR gain.

## Limitations & Future Work

1. **Training and primary evaluation solely on UDIS-D**: Despite zero-shot cross-dataset testing, training data diversity remains limited.
2. **Dynamic scenes not addressed**: The method targets static scene registration and does not account for moving objects.
3. **Single decay function form**: Alternative decay functions (e.g., Gaussian, polynomial decay) are not explored.
4. **Uniform control point grid**: Adaptive control point distributions for handling varying deformation magnitudes across regions are not investigated.
5. **Two-stage training**: The non-end-to-end joint training of global and local modules may limit overall performance.

## Related Work & Insights

- The TPS + MLP scheme of UDIS++ serves as the most direct baseline; EDFFDNet surpasses it across all dimensions.
- FFD models are widely used in medical image registration; this work introduces them to natural image registration with an improved basis function design.
- The grouped linear layer concept is borrowed from depthwise separable convolutions (MobileNet), and its effectiveness is validated in the new context of motion aggregation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The designs of EDFFD and ASMA are concise yet effective, tightly coupling engineering innovation with a thorough understanding of the problem.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Ablations cover all three dimensions (deformation model / aggregator / correlation strategy), supplemented by cross-dataset evaluation, computational efficiency analysis, and speed comparisons with traditional methods.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are clear, design motivations are explicit, and comparative analyses are thorough.
- **Value**: ⭐⭐⭐⭐ — Achieves simultaneous improvements in both accuracy and efficiency for image registration, with high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Revisiting Image Fusion for Multi-Illuminant White-Balance Correction](revisiting_image_fusion_for_multi-illuminant_white-balance_correction.md)
- [\[AAAI 2026\] DcMatch: Unsupervised Multi-Shape Matching with Dual-Level Consistency](../../AAAI2026/others/dcmatch_unsupervised_multi-shape_matching_with_dual-level_consistency.md)
- [\[ICCV 2025\] Is Meta-Learning Out? Rethinking Unsupervised Few-Shot Classification with Limited Entropy](is_meta-learning_out_rethinking_unsupervised_few-shot_classification_with_limite.md)
- [\[ICLR 2026\] Distributionally Robust Classification for Multi-Source Unsupervised Domain Adaptation](../../ICLR2026/others/distributionally_robust_classification_for_multi-source_unsupervised_domain_adap.md)
- [\[AAAI 2026\] DiffMM: Efficient Method for Accurate Noisy and Sparse Trajectory Map Matching via One Step Diffusion](../../AAAI2026/others/diffmm_efficient_method_for_accurate_noisy_and_sparse_trajectory_map_matching_vi.md)

</div>

<!-- RELATED:END -->
