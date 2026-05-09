---
title: >-
  [Paper Note] NTK-Guided Implicit Neural Teaching
description: >-
  [CVPR 2026][3D Vision][Implicit Neural Representations] This paper proposes NINT, which leverages row vectors of the Neural Tangent Kernel (NTK) to measure each coordinate's influence on the global function update, enabling dynamic selection of coordinates with both high fitting error and high global influence for training. This approach reduces INR training time by nearly half without sacrificing reconstruction quality.
tags:
  - CVPR 2026
  - 3D Vision
  - Implicit Neural Representations
  - Neural Tangent Kernel
  - Training Acceleration
  - Coordinate Sampling
  - INR
date: 2026-05-08
content_hash: be6fb72f20a11f83
---

# NTK-Guided Implicit Neural Teaching

**Conference**: CVPR 2026
**arXiv**: [2511.15487](https://arxiv.org/abs/2511.15487)
**Code**: Available (Project page)
**Area**: 3D Vision
**Keywords**: Implicit Neural Representations, Neural Tangent Kernel, Training Acceleration, Coordinate Sampling, INR

## TL;DR
This paper proposes NINT, which leverages row vectors of the Neural Tangent Kernel (NTK) to measure each coordinate's influence on the global function update, enabling dynamic selection of coordinates with both high fitting error and high global influence for training. This approach reduces INR training time by nearly half without sacrificing reconstruction quality.

## Background & Motivation
Implicit Neural Representations (INR) map coordinates to signal values (e.g., pixel colors) via MLPs, enabling resolution-agnostic continuous signal modeling. However, high-resolution signals (e.g., a $1024 \times 1024$ image with one million pixel coordinates) incur prohibitively high training costs.

Existing acceleration methods each have limitations:
- **Partitioning methods** (multiple small MLPs covering different regions): increase architectural complexity and inference overhead
- **Hybrid explicit-implicit methods** (hash grids, tensor decompositions, etc.): increase memory consumption
- **Meta-learning methods** (pre-trained initialization): require large homogeneous datasets and lack flexibility
- **Sampling methods** (training on a subset of coordinates per step): lightweight, but most rely solely on static error heuristics, ignoring the dynamic nature of parameter updates during MLP training

Core insight: Existing error-based sampling methods (e.g., INT, EGRA, EVOS) implicitly assume that the NTK matrix is diagonal and isotropic (i.e., $K_{\theta^t} \approx cI$), implying (1) no cross-coordinate influence, and (2) identical self-leverage for all coordinates. In practice, however, weight sharing in MLPs induces strong off-diagonal coupling, and diagonal values vary by orders of magnitude depending on the region (edges vs. smooth areas). Selecting only high-error points may therefore waste gradient steps on points that are "high-error but low-influence."

## Method

### Overall Architecture
The core mechanism of NINT: rather than performing gradient descent over all $N$ coordinates or naively selecting a subset by error magnitude, NINT evaluates each coordinate's contribution to the global function evolution via the NTK matrix and selects the $B$ coordinates with the largest contribution to form the mini-batch.

Overall pipeline (Algorithm 1):
1. Forward pass to compute predictions $\hat{\mathbf{y}}_i = f_{\theta_t}(\mathbf{x}_i)$ for all coordinates
2. Compute the loss gradient vector for all coordinates: $\mathbf{g}^t = [\nabla_f \mathcal{L}(f_{\theta^t}(\mathbf{x}_i), \mathbf{y}_i)]_{i=1}^N$
3. Compute the NTK row vector $K_{\theta^t}(\mathbf{x}_i, :)$ for each coordinate $\mathbf{x}_i$
4. Select the $B$ coordinates that maximize the NTK-amplified gradient norm: $\mathcal{B}_t = \arg\max_{|\mathcal{B}|=B} \|[K_{\theta^t}(\mathbf{x}_i,:) \cdot \mathbf{g}^t]_{i \in \mathcal{B}}\|_2$
5. Perform gradient updates using only the selected coordinates

### Key Designs

#### 1. NTK-Driven Analysis of Training Dynamics
The function evolution of INR is analyzed from a continuous-time perspective. Applying a first-order Taylor expansion to the parameter update and substituting into the gradient descent parameter evolution equation yields:

$$\frac{\partial f_{\theta^t}(\mathbf{x})}{\partial t} \simeq -\frac{\eta}{N} [K_{\theta^t}(\mathbf{x}_i, \mathbf{x})]_{i=1}^{N \top} \cdot [\nabla_f \mathcal{L}(f_{\theta^t}(\mathbf{x}_i), \mathbf{y}_i)]_{i=1}^N$$

where the NTK is defined as the inner product of parameter gradients: $K_{\theta^t}(\mathbf{x}_i, \mathbf{x}) = \langle \frac{\partial f_{\theta^t}(\mathbf{x}_i)}{\partial \theta^t}, \frac{\partial f_{\theta^t}(\mathbf{x})}{\partial \theta^t} \rangle$.

This reveals two key structural properties:
- **Diagonal elements** (self-leverage): $K_{\theta^t}(\mathbf{x}_i, \mathbf{x}_i) = \|\frac{\partial f_{\theta^t}(\mathbf{x}_i)}{\partial \theta^t}\|_2^2$, measuring the strength of coordinate $\mathbf{x}_i$'s influence on its own output
- **Off-diagonal elements** (cross-coordinate coupling): $K_{\theta^t}(\mathbf{x}_i, \mathbf{x}_j)$, measuring the degree to which a loss-driven parameter update at $\mathbf{x}_i$ incidentally modifies the output at $\mathbf{x}_j$

Design motivation: Ignoring the NTK and relying solely on error is equivalent to approximating the NTK as $cI$, which does not hold for practical MLPs — NTK diagonal values in edge/high-frequency regions are far larger than in smooth regions, and weight sharing induces strong off-diagonal coupling.

#### 2. NINT Sampling Strategy
NINT selects coordinates by maximizing the magnitude of function evolution. Specifically, the score is defined as the norm of the product between the NTK row vector and the global loss gradient vector:

$$\text{score}(\mathbf{x}_i) = \|K_{\theta^t}(\mathbf{x}_i, :) \cdot \mathbf{g}^t\|_2$$

This score simultaneously captures two factors:
- **Fitting error**: reflected through the components of $\mathbf{g}^t$
- **Global influence**: reflected through the NTK row vector $K_{\theta^t}(\mathbf{x}_i, :)$, encompassing both self-leverage and cross-coupling

Comparison with error-only methods: error-only methods select $\arg\max \|\nabla_f \mathcal{L}\|_2$, which is equivalent to assuming $K = cI$; NINT selects $\arg\max \|K_{\theta^t}(\mathbf{x}_i,:) \cdot \mathbf{g}^t\|_2$, explicitly exploiting the full NTK information.

#### 3. Hybrid Sampling and Decay Schedule
The practical implementation adopts a three-part hybrid sampling strategy:
- A fraction $\xi$ (default 0.7) of coordinates sampled randomly
- A fraction $(1-\xi)\exp(-\lambda t / \alpha)$ of coordinates selected via NTK-guided sampling ($\lambda=1.0$, $\alpha=10$)
- The remaining coordinates filled by conventional error-based sampling

The NTK contribution decays exponentially as training progresses, because: (1) in later stages, the error distribution becomes more uniform and the benefit of NTK guidance diminishes; (2) NTK computation incurs overhead, and decay reduces this cost. The parameter $\alpha$ simultaneously controls the NTK recomputation frequency (iterations without recomputation reuse the previous result).

### Loss & Training
- **Loss function**: Standard $\ell_2$ regression loss $\mathcal{L}(f_\theta(\mathbf{x}_i), \mathbf{y}_i) = \|f_\theta(\mathbf{x}_i) - \mathbf{y}_i\|_2^2$
- **Optimizer / learning rate**: $\eta = 1 \times 10^{-4}$
- **Batch size**: 20% of the full coordinate set (Stand. uses 100%)
- **Network architecture**: Default 5-layer × 256-unit SIREN MLP

## Key Experimental Results

### Main Results: Reconstruction Quality at Fixed Iterations

| Method | 250 iter PSNR | 1000 iter PSNR | 5000 iter PSNR | 5000 iter SSIM | 5000 iter LPIPS |
|--------|---------------|----------------|----------------|----------------|-----------------|
| Stand. (full) | 27.90 | 31.67 | 39.76 | 0.962 | 0.022 |
| Uniform | 27.66 | 31.14 | 37.14 | 0.943 | 0.069 |
| EGRA | 27.67 | 31.24 | 37.39 | 0.945 | 0.068 |
| INT | 27.57 | 31.19 | 39.02 | 0.943 | 0.035 |
| EVOS | 28.02 | 31.72 | 37.56 | 0.940 | 0.054 |
| Expan. | 27.99 | 32.15 | 38.22 | 0.947 | 0.056 |
| **NINT** | **28.96** | **32.64** | **39.09** | **0.958** | **0.029** |

### Main Results: Time to Reach Target PSNR

| Method | Time to PSNR=30 (s) | Time to PSNR=35 (s) | Speedup vs. Stand. |
|--------|---------------------|---------------------|--------------------|
| Stand. (full) | 49.11 | 184.78 | — |
| INT | 33.01 | 111.80 | 32.8% / 39.5% |
| EVOS | 31.20 | 143.20 | 36.5% / 22.5% |
| Expan. | 29.16 | 123.60 | 40.6% / 33.1% |
| **NINT** | **25.05** | **102.88** | **49.0% / 44.3%** |

### Ablation Study: Different Network Scales

| Network Scale | 500 iter PSNR | 1000 iter PSNR | 2500 iter PSNR | Time at 3000 iter (s) |
|---------------|---------------|----------------|----------------|-----------------------|
| 3×128 Stand. | 23.17 | 24.17 | 26.14 | 92.16 |
| 3×128 + NINT | 23.20 | 24.51 | 26.52 | 72.14 (21.7%) |
| 5×256 Stand. | 25.61 | 28.69 | 33.69 | 35.42 |
| 5×256 + NINT | 26.85 | 31.27 | 35.10 | 22.16 (37.4%) |

### Ablation Study: Different Network Architectures

| Architecture | 60s PSNR | 120s PSNR | Time to PSNR=25 (s) | Speedup |
|--------------|----------|-----------|----------------------|---------|
| SIREN | 30.51 | 32.44 | 8.25 | — |
| SIREN + NINT | 32.40 | 35.47 | 5.81 | 29.6% |
| FFN | 26.90 | 31.44 | 54.19 | — |
| FFN + NINT | 27.39 | 31.48 | 48.75 | 10.0% |
| WIRE | 23.86 | 27.17 | 83.30 | — |
| WIRE + NINT | 26.62 | 29.13 | 47.23 | 43.3% |

### Key Findings
1. **Training time halved**: Compared to full-batch training, NINT reduces the time to reach target PSNR by up to 49%, with a 27% reduction in iteration count.
2. **Larger networks benefit more**: Time savings increase from approximately 11% to 37.4% as network size grows from 3×64 to 5×256.
3. **Architecture-agnostic**: Effective across seven architectures — MLP, FFN, FINER, GAUSS, PEMLP, SIREN, and WIRE — with a maximum speedup of 43.3% (WIRE).
4. **Robust hyperparameters**: The default setting $(\xi=0.7, \alpha=10, \lambda=1.0)$ is near-optimal; deviations from defaults result in only marginal performance degradation.
5. **Early-stage advantage is most pronounced**: NINT's PSNR lead over baselines is largest during early training (250 iter / 20s).

## Highlights & Insights
1. **Rigorous NTK-based theoretical analysis**: The paper precisely characterizes why error-only sampling is insufficient using NTK theory — it is equivalent to approximating the NTK as $cI$, thereby ignoring self-leverage heterogeneity and cross-coordinate coupling. This is an elegant and persuasive theoretical insight.
2. **Plug-and-play**: NINT is a model-agnostic sampling strategy that does not modify the network architecture and can be directly integrated into any INR training pipeline.
3. **Engineering ingenuity in hybrid sampling**: Given the high computational cost of NTK, the three-part hybrid sampling combined with exponential decay and periodic reuse effectively controls overhead, making the method practically viable.
4. **Visualization enhances understanding**: The visualization of 9×9 NTK matrix blocks in Figure 2 intuitively demonstrates off-diagonal coupling and diagonal heterogeneity, substantially strengthening the motivation for the proposed method.

## Limitations & Future Work
1. **NTK computation overhead**: The full NTK matrix is $N \times N$, which is infeasible for million-scale coordinate sets; although mitigated by decay and periodic reuse, it still constitutes additional overhead.
2. **Primarily evaluated on 2D images**: Main experiments focus on the Kodak and DIV2K image datasets; 1D/3D experiments are relegated to supplementary materials, and validation on large-scale 3D scenes (e.g., NeRF) is insufficient.
3. **Absence of comparison with non-sampling acceleration methods**: No end-to-end comparison against hybrid explicit-implicit methods such as hash grids (Instant-NGP) or TensoRF.
4. **Gap between theory and practice**: The NTK analysis assumes an infinite-width limit or slowly varying parameters; the NTK evolves in finite-width MLPs, and the paper lacks quantitative analysis of this approximation error.
5. **Memory overhead not discussed**: The specific GPU memory requirements for storing and computing NTK row vectors are not explicitly addressed.

## Rating
- **Novelty**: 4/5 — Introducing NTK into INR coordinate sampling is a novel perspective; the theoretical analysis elegantly exposes the fundamental limitation of error-only methods.
- **Experimental Thoroughness**: 4/5 — Experiments adequately cover multiple baselines, network scales, architectures, and hyperparameter sensitivity, though they are primarily limited to 2D images.
- **Writing Quality**: 5/5 — The logical chain from NTK theory to the deficiencies of existing methods to the proposed design is clear and coherent, with well-crafted figures and tables.
- **Value**: 4/5 — The plug-and-play training acceleration approach has substantial practical value, though its applicability to very large-scale scenes remains to be validated given the NTK computation overhead.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] 3DrawAgent: Teaching LLM to Draw in 3D with Early Contrastive Experience](3drawagent_teaching_llm_to_draw_in_3d_with_early_contrastive_experience.md)
- [\[CVPR 2026\] 3D-IDE: 3D Implicit Depth Emergent](3d-ide_3d_implicit_depth_emergent.md)
- [\[ICCV 2025\] SL2A-INR: Single-Layer Learnable Activation for Implicit Neural Representation](../../ICCV2025/3d_vision/sl2a-inr_single-layer_learnable_activation_for_implicit_neural_representation.md)
- [\[AAAI 2026\] Split-Layer: Enhancing Implicit Neural Representation by Maximizing the Dimensionality of Feature Space](../../AAAI2026/3d_vision/split-layer_enhancing_implicit_neural_representation_by_maximizing_the_dimension.md)
- [\[ICCV 2025\] LINR-PCGC: Lossless Implicit Neural Representations for Point Cloud Geometry Compression](../../ICCV2025/3d_vision/linr-pcgc_lossless_implicit_neural_representations_for_point_cloud_geometry_comp.md)

</div>

<!-- RELATED:END -->
