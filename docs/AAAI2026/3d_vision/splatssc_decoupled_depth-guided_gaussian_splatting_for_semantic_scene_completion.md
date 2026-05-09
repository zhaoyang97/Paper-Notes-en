---
title: >-
  [Paper Note] SplatSSC: Decoupled Depth-Guided Gaussian Splatting for Semantic Scene Completion
description: >-
  [AAAI 2026][3D Vision][Semantic Scene Completion] SplatSSC is proposed as a monocular 3D semantic scene completion framework based on depth-guided initialization and a Decoupled Gaussian Aggregator (DGA). Through compact Gaussian primitive initialization and robust geometry-semantics decoupled aggregation, it achieves state-of-the-art performance on Occ-ScanNet with significantly fewer primitives.
tags:
  - AAAI 2026
  - 3D Vision
  - Semantic Scene Completion
  - 3D Gaussian Splatting
  - Depth Guidance
  - Decoupled Aggregation
  - Object-Centric Representation
date: 2026-05-08
content_hash: db5e5c0a8951d62c
---

# SplatSSC: Decoupled Depth-Guided Gaussian Splatting for Semantic Scene Completion

**Conference**: AAAI 2026
**arXiv**: [2508.02261](https://arxiv.org/abs/2508.02261)
**Code**: [GitHub](https://github.com/Made-Gpt/SplatSSC)
**Area**: 3D Vision
**Keywords**: Semantic Scene Completion, 3D Gaussian Splatting, Depth Guidance, Decoupled Aggregation, Object-Centric Representation

## TL;DR

SplatSSC is proposed as a monocular 3D semantic scene completion framework based on depth-guided initialization and a Decoupled Gaussian Aggregator (DGA). Through compact Gaussian primitive initialization and robust geometry-semantics decoupled aggregation, it achieves state-of-the-art performance on Occ-ScanNet with significantly fewer primitives.

## Background & Motivation

### Problem Definition

Monocular 3D semantic scene completion (SSC) aims to infer dense geometric and semantic descriptions of a scene from a single image. This task is critical for embodied intelligence and autonomous driving. Recently, the object-centric paradigm (exemplified by GaussianFormer) has adopted flexible 3D Gaussian primitives to represent scenes, achieving a new balance between performance and efficiency.

### Core Motivation

The object-centric paradigm faces a **fundamental challenge** under a pure vision setting (monocular input): how to efficiently initialize and reliably supervise 3D primitives from monocular cues alone.

The **random initialization strategies** adopted by existing methods lead to two critical problems:

**1. Inefficient primitive initialization**: A large number of primitives are wasted on representing empty or unknown space. For example, GaussianFormer-2 uses 19,200 primitives, many of which correspond to empty regions, resulting in wasted computational resources.

**2. Fragile aggregation against outlier primitives**: Existing Gaussian-to-voxel aggregation strategies (GaussianFormer and GaussianFormer-2's PGS) **lack effective rejection mechanisms**. Outlier primitives spread spurious semantics to distant voxels, producing "floater" artifacts.

### In-Depth Analysis of PGS (Probabilistic Gaussian Superposition) Defects

The authors conduct a rigorous mathematical analysis of GaussianFormer-2's PGS and identify its fundamental flaw:

- PGS uses learned opacity $\mathbf{a}_i$ as the prior probability in a GMM
- For an **isolated outlier primitive** $G_n$, at any nearby point $\mathbf{x}^f$, the likelihood $p(\mathbf{x}^f|G_m)$ of other distant primitives approaches zero
- This causes the posterior probability $p(G_n|\mathbf{x}^f)$ to **degenerate to 1**, regardless of how low the opacity $\mathbf{a}_n$ is:

$$p(G_n|\mathbf{x}^f) \approx \frac{p(\mathbf{x}^f|G_n)\mathbf{a}_n}{p(\mathbf{x}^f|G_n)\mathbf{a}_n + 0} = 1$$

- Result: low-confidence outlier primitives still exert full semantic influence on nearby voxels, producing floaters.

## Method

### Overall Architecture

The SplatSSC pipeline proceeds as follows:
1. An image encoder (EfficientNet-B7 + FPN) extracts multi-scale image features
2. A frozen Depth-Anything-V2 extracts depth features
3. The GMF module fuses image and depth features
4. The depth branch generates a refined depth map → guides initialization of a **compact Gaussian primitive set** (only 1,200 primitives)
5. A multi-stage encoder iteratively refines the primitives
6. DGA converts the refined primitives into a final semantic voxel grid

### Key Designs

#### 1. Group-wise Multi-scale Fusion Module (GMF) and Group Cross-Attention (GCA)

**Function**: Efficiently fuses multi-scale image features and depth features to generate high-quality geometric priors.

**GCA design details**:
1. Features are sampled from depth features $\mathcal{F}_d$ and multi-scale image features $\mathcal{F}_{rgb}$
2. Features are divided into $G$ groups along the channel dimension, each with dimension $D_g = D/G$
3. Queries come from depth features; Keys/Values come from image features at each scale:

$$Q_g = (\mathcal{F}_d^s W_q)^g, \quad K_g^l = (f_{rgb}^{s,l} W_k)^g, \quad V_g^l = (f_{rgb}^{s,l} W_v)^g$$

4. Lightweight linear projections replace standard dot-product attention (inspired by Deformable Attention):

$$A_g^l = \mathbb{S}_l(W_a(Q_g + K_g^l))$$

5. Final fusion: $\mathcal{F}_d' = \mathbb{C}_g(\sum_{l=1}^{L} A_g^l \circ V_g^l) W_o$

**Efficiency analysis**: Standard cross-attention has complexity $\mathcal{O}(LN^2D)$; GCA reduces this to $\mathcal{O}(ND^2(L+2)/G)$, making it particularly suitable for long sequences.

**Design Motivation**:
- $W_a$ is shared across groups and scales, significantly reducing parameters and computation
- The latent features of the depth estimator (rather than raw depth map values) are exploited to obtain richer initialization information
- Gaussian primitives are endowed with initial embeddings carrying both **positional (where)** and **semantic (what)** information

#### 2. Decoupled Gaussian Aggregator (DGA)

**Function**: Decomposes Gaussian-to-voxel aggregation into two independent paths—geometric occupancy prediction and conditional semantic distribution—to robustly handle outlier primitives.

**Geometric Occupancy Prediction**:

$$\alpha'(\mathbf{x}) = 1 - \prod_{i \in \mathcal{N}(\mathbf{x})} (1 - \alpha(\mathbf{x}; G_i) \cdot \mathbf{a}_i)$$

Key distinction: opacity $\mathbf{a}_i$ is directly multiplied into the Gaussian kernel as an **existence confidence**. Low-confidence outlier primitives are directly suppressed in their occupancy contribution.

**Conditional Semantic Distribution** (conditioned on the location being occupied):

$$e^k(\mathbf{x}) = \frac{\sum_{i \in \mathcal{N}(\mathbf{x})} p(\mathbf{x}|G_i) \cdot \tilde{\mathbf{c}}_i^k}{\sum_{j \in \mathcal{N}(\mathbf{x})} p(\mathbf{x}|G_j)}$$

Key distinction: semantic prediction is **entirely independent of opacity**, relying solely on geometric proximity and softmax-normalized semantic attributes.

**Probabilistic Fusion**:

$$\hat{\mathbf{y}}_x^k = \alpha'(\mathbf{x}) \cdot e^k(\mathbf{x}), \quad \hat{\mathbf{y}}_x^{empty} = 1 - \alpha'(\mathbf{x})$$

**Design Motivation**: When a low-opacity outlier primitive is present:
- $\alpha'(\mathbf{x})$ remains low (since $\mathbf{a}_i$ directly participates in occupancy computation)
- Even if $e^k(\mathbf{x})$ may be dominated by the outlier primitive, multiplying by the low $\alpha'(\mathbf{x})$ effectively suppresses its influence
- Floaters are eliminated without any complex heuristics

#### 3. Probability Scale Loss

**Function**: Provides auxiliary geometric supervision for all encoder layers to stabilize end-to-end training.

$$\mathcal{L}_{scal}^{prob} = \frac{1}{2} \sum_{i=1}^{n-1} \frac{i}{n} \cdot \mathcal{L}_{scal}^{geo,i} + \mathcal{L}_{scal}^{geo,n}$$

**Design Motivation**: A linear weighting schedule is introduced—applying weaker constraints to earlier layers and stronger consistency requirements to deeper layers, matching the characteristics of layer-by-layer refinement.

### Loss & Training

**Two-stage training**:

**Stage 1 — Depth Branch Pre-training**:
$$\mathcal{L}_d = \lambda_1 \mathcal{L}_{huber}^{depth} + \lambda_2 \mathcal{L}_{huber}^{pts} + \lambda_3 \mathcal{L}_{grad}$$
- 10 epochs, 2× RTX 3090

**Stage 2 — End-to-End SplatSSC Training**:
$$\mathcal{L}_{ssc} = \mathcal{L}_{sem} + \lambda_4 \mathcal{L}_{scal}^{prob}$$
- $\mathcal{L}_d$ is removed to avoid over-constraining the model
- $\mathcal{L}_{sem} = \lambda_5 \mathcal{L}_{focal} + \lambda_6 \mathcal{L}_{lovasz}$
- 10 epochs (full), 20 epochs (mini), 4× RTX 4090

Loss weights: $\lambda_1=10, \lambda_2=20, \lambda_3=\lambda_4=0.5, \lambda_5=100, \lambda_6=2$

## Key Experimental Results

### Main Results

**Semantic scene completion results on Occ-ScanNet**:

| Method | Input | IoU↑ | mIoU↑ |
|---|---|---|---|
| TPVFormer | RGB | 33.39 | 24.94 |
| MonoScene | RGB | 41.60 | 24.62 |
| GaussianFormer | RGB | 40.91 | 29.93 |
| SurroundOcc | RGB | 42.52 | 30.83 |
| EmbodiedOcc | RGB | 53.95 | 45.48 |
| EmbodiedOcc++ | RGB | 54.90 | 46.20 |
| RoboOcc | RGB | 56.48 | 47.67 |
| **SplatSSC** | **RGB** | **62.83** | **51.83** |

**Improvement**: IoU and mIoU exceed RoboOcc by **6.35% and 4.16%**, respectively.

### Ablation Study

**Network component ablation (Occ-ScanNet-mini)**:

| GMF | Aggregator | IoU↑ | mIoU↑ | Note |
|---|---|---|---|---|
| ✗ | GF.agg | 11.64 | 12.62 | Both poor, nearly non-functional |
| ✗ | GF2.agg | 27.54 | 17.27 | GF2 shows limited improvement |
| ✗ | DGA | 48.85 | 36.91 | DGA performs well even without GMF |
| ✓ | GF.agg | 16.63 | 10.45 | GMF+GF.agg still fails |
| ✓ | GF2.agg | 57.70 | 45.13 | GMF substantially boosts GF2 |
| ✓ | DGA | **60.61** | **48.01** | GMF+DGA optimal |

**Gaussian parameter ablation**:

| # Primitives | Scale Range | IoU↑ | mIoU↑ | Note |
|---|---|---|---|---|
| 19200 | [0.01, 0.08] | 62.77 | 47.69 | More primitives yet lower mIoU |
| 4800 | [0.01, 0.08] | 62.23 | 47.20 | Intermediate count |
| **1200** | **[0.01, 0.16]** | **61.47** | **48.87** | Fewest primitives, highest mIoU |
| 1200 | [0.01, 0.32] | 57.09 | 42.38 | Excessively large scale range degrades performance |

**Depth branch ablation**:

| Configuration | $\delta_1$↑ | RMSE↓ | Note |
|---|---|---|---|
| DAv2 (w/o GMF) | 0.075 | 50.31 | Frozen depth model alone performs poorly |
| DAv2 + GMF | 0.981 | 4.94 | GMF improves $\delta_1$ by 0.906 |
| FT-DAv2 + GMF | **0.993** | **2.98** | Fine-tuning + GMF is optimal |

### Key Findings

1. **GF.agg nearly fails in sparse settings** (10.45% mIoU), demonstrating that floaters are a critical bottleneck in sparse splatting
2. **Only 1,200 primitives are needed to achieve the highest mIoU**, outperforming 19,200 primitives—validating the superiority of depth-guided initialization
3. **GMF has a decisive impact on depth quality**: frozen DAv2 achieves $\delta_1$ of only 0.075, which improves to 0.981 with GMF
4. **Two-stage training outperforms end-to-end training**: end-to-end training yields higher $\mathcal{L}_{ssc}$ and lower mIoU
5. **Efficiency gains are notable**: inference latency reduced by 9.32% and memory reduced by 9.64%

## Highlights & Insights

1. The **rigorous mathematical analysis of PGS defects** is highly convincing—the cancellation of opacity under posterior normalization is demonstrated through limit analysis
2. **The decoupling design is elegant**: geometry is gated by opacity, semantics rely purely on geometric proximity, and the two are fused through probabilistic multiplication
3. **Less is more**: 1,200 primitives outperform 19,200, demonstrating that precise depth-guided initialization is far superior to large-scale random initialization
4. **The dramatic improvement from GMF**—raising $\delta_1$ from 0.075 to 0.981—underscores the importance of leveraging depth estimator features rather than raw depth values
5. **The linear weighting schedule of the Probability Scale Loss** aligns intuitively with the layer-by-layer refinement process

## Limitations & Future Work

1. **Hyperparameter sensitivity**: performance drops sharply when batch size is below 4 (mIoU falls from 48.87 to 36.09)
2. **Single-frame architecture limitation**: the current per-frame prediction design cannot be directly extended to global scene perception (primitive accumulation causes memory growth)
3. Reliance on the frozen pre-trained Depth-Anything model caps the performance ceiling
4. Validation is limited to indoor scenes (ScanNet); applicability to outdoor autonomous driving scenarios remains to be verified
5. The downsampled 30×40 depth grid resolution may limit fine-grained geometry recovery

## Related Work & Insights

- **GaussianFormer/GaussianFormer-2**: Pioneers and refiners of the object-centric 3D occupancy prediction paradigm; the direct comparison and improvement targets of this work
- **EmbodiedOcc/EmbodiedOcc++**: Apply the object-centric paradigm to indoor scenes; this work builds on their framework
- **RoboOcc**: Enhances stability through opacity cues and planar constraints; the primary comparison method on Occ-ScanNet
- **Depth-Anything-V2**: Provides strong depth priors; this work further exploits its latent features
- **VoxFormer**: Proposes a sparse-then-dense two-stage approach, conceptually aligned with this work's sparse initialization philosophy

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Insightful analysis of PGS defects, elegant DGA design, and the impressive result that 1,200 primitives outperform 19,200
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Dual-dataset evaluation, comprehensive ablations (components/parameters/losses/depth/efficiency), comparison against 8+ methods
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Rigorous problem analysis, clear mathematical derivations, and a transparent framework
- **Value**: ⭐⭐⭐⭐⭐ — Absolute improvements of 6.3%/4.1% in IoU/mIoU, deployment-friendly (fewer primitives = higher efficiency)

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Monocular Semantic Scene Completion via Masked Recurrent Networks](../../ICCV2025/3d_vision/monocular_semantic_scene_completion_via_masked_recurrent_networks.md)
- [\[ICCV 2025\] Disentangling Instance and Scene Contexts for 3D Semantic Scene Completion](../../ICCV2025/3d_vision/disentangling_instance_and_scene_contexts_for_3d_semantic_scene_completion.md)
- [\[AAAI 2026\] OceanSplat: Object-aware Gaussian Splatting with Trinocular View Consistency for Underwater Scene Reconstruction](oceansplat_object-aware_gaussian_splatting_with_trinocular_view_consistency_for_.md)
- [\[ICCV 2025\] Global-Aware Monocular Semantic Scene Completion with State Space Models](../../ICCV2025/3d_vision/global-aware_monocular_semantic_scene_completion_with_state_space_models.md)
- [\[AAAI 2026\] Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting](gaussian_blending_rethinking_alpha_blending_in_3d_gaussian_splatting.md)

<!-- RELATED:END -->
