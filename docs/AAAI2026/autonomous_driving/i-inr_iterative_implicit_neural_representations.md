---
title: >-
  [Paper Note] I-INR: Iterative Implicit Neural Representations
description: >-
  [AAAI 2026][Autonomous Driving][Implicit Neural Representations] This paper proposes I-INR (Iterative Implicit Neural Representations), a plug-and-play iterative refinement framework that introduces lightweight FeedbackNet and FuseNet modules (adding only 0.5–2% parameters) to perform progressive multi-step signal reconstruction, effectively alleviating the spectral bias of INRs. I-INR consistently outperforms baselines across image fitting, super-resolution, denoising, and 3D occupancy prediction tasks.
tags:
  - AAAI 2026
  - Autonomous Driving
  - Implicit Neural Representations
  - Iterative Refinement
  - High-Frequency Reconstruction
  - Denoising Robustness
  - Plug-and-Play Framework
date: 2026-05-08
content_hash: 328f20684801ddf6
---

# I-INR: Iterative Implicit Neural Representations

**Conference**: AAAI 2026
**arXiv**: [2504.17364](https://arxiv.org/abs/2504.17364)
**Code**: [https://github.com/optimizer077/I-INR](https://github.com/optimizer077/I-INR)
**Area**: Computer Vision / Signal Representation
**Keywords**: Implicit Neural Representations, Iterative Refinement, High-Frequency Reconstruction, Denoising Robustness, Plug-and-Play Framework

## TL;DR

This paper proposes I-INR (Iterative Implicit Neural Representations), a plug-and-play iterative refinement framework that introduces lightweight FeedbackNet and FuseNet modules (adding only 0.5–2% parameters) to perform progressive multi-step signal reconstruction, effectively alleviating the spectral bias of INRs. I-INR consistently outperforms baselines across image fitting, super-resolution, denoising, and 3D occupancy prediction tasks.

## Background & Motivation

### Core Challenges of INR

Implicit Neural Representations (INRs) employ neural networks (typically MLPs) to directly map spatial/temporal coordinates to signal attributes (e.g., pixel intensity, color, 3D occupancy), offering resolution-agnostic encoding, compact representation, and seamless interpolation. Nevertheless, INRs face three persistent challenges:

**Spectral Bias**: When optimized with L1/L2 losses, INRs inherently favor low-frequency components, causing high-frequency detail loss.

**Poor Noise Robustness**: Existing methods typically assume clean and complete inputs, and performance degrades under noise and occlusion.

**Limited Generalization**: Single-pass fitting results are difficult to generalize to unseen resolutions or degradation conditions.

### Limitations of Prior Work

- **Positional encoding schemes** (e.g., Fourier features): inject high-frequency signals via orthogonal Fourier bases, but are ill-suited for noisy scenarios.
- **Alternative activation functions** (e.g., sinusoidal in SIREN, Gabor wavelets in WIRE, Gaussian in Gauss): better capture high-frequency structures, but struggle to balance high fidelity and noise robustness simultaneously.
- **WIRE**: employs Gabor wavelet activations to improve noise robustness, yet there remains room for improvement.

### Inspiration from Iterative Methods

Iterative approaches such as diffusion models have achieved remarkable success in image restoration and video generation by reversing degradation processes through multiple steps. However, **iterative strategies remain largely unexplored in the INR domain**.

**Core Motivation**: Drawing on the concept of iterative refinement, this work transforms signal reconstruction from a one-shot prediction into a multi-step progressive refinement process, simultaneously improving high-frequency preservation and noise robustness.

## Method

### Overall Architecture

I-INR formulates signal reconstruction as an iterative process from an initial state $\mathcal{Z}$ (t=1) to the final reconstruction $\mathcal{I}(x)$ (t=0), proceeding in steps of size $\delta$. The overall architecture consists of three components:

- **Backbone**: Any existing INR architecture (e.g., SIREN, WIRE, Gauss), executed via a single forward pass.
- **FeedbackNet**: A lightweight MLP (2 layers, width 30) that integrates intermediate states with time conditioning.
- **FuseNet**: A lightweight MLP (2 layers, width 100) that fuses features from the Backbone and FeedbackNet.

### Key Designs

#### 1. **Mathematical Formulation of Iterative Reconstruction**

The forward process linearly interpolates between the target signal and the initial state:

$$g(x)_t = \mathcal{I}(x)(1-t) + \mathcal{Z}t, \quad t \in [0,1]$$

The reconstruction process is iteratively updated via conditional expectation:

$$\hat{g}(x)_{t-\delta} = \frac{\delta}{t}\mathbb{E}[g(x)_0 | \hat{g}(x)_t] + (1 - \frac{\delta}{t})\hat{g}(x)_t$$

The initial state $\mathcal{Z}$ is sampled from a standard normal distribution, which is empirically shown to outperform zero or all-ones initialization.

**Design Motivation**: This formulation is inspired by InDI (Inversion by Direct Iteration)—decomposing a difficult inverse problem into a series of simpler subproblems, where each update only requires estimating a conditional expectation. The approach naturally transitions from a poor initial estimate to an accurate reconstruction, realizing coarse-to-fine multi-scale learning.

#### 2. **Training Objective**

The implicit neural network $f_\theta$ is trained to directly predict the clean target:

$$\min_\theta \mathbb{E}_{x,t,n}\|f_\theta(\tilde{g}(x)_t, x, t) - \mathcal{I}(x)\|_2^2$$

where a small perturbation is added to the intermediate state to ensure regularity:

$$\tilde{g}(x)_t = (1-t)\mathcal{I}(x) + t\mathcal{Z} + \varepsilon t n$$

$\varepsilon$ is empirically set to 0.1 and $n \sim \mathcal{N}(0,1)$.

**Design Motivation**: The small perturbation term $\varepsilon tn$ satisfies regularity requirements and ensures stability of the reconstruction process at inference time. Additionally, sampling random timesteps $t$ enables the network to handle intermediate states ranging from fully blurred to nearly clean.

#### 3. **Network Architecture Design**

The information flow is formalized as:

$$f_\theta(\hat{g}(x)_t, x, t) = \text{FuseNet}(\text{concat}(\mathbf{f}, \mathbf{b})) \odot \mathbf{b}$$

where $\mathbf{b} = \text{Backbone}(x)$ and $\mathbf{f} = \text{FeedbackNet}(\hat{g}(x)_t, x, t)$.

Key efficiency design choices:
- **Backbone executed only once**: Base features are extracted once and reused across all iterative steps.
- **FeedbackNet and FuseNet are extremely lightweight**: Each refinement step adds only 0.43 GFLOPs (compared to 106.8 GFLOPs for the Backbone).
- **Multiplicative fusion**: Element-wise multiplication ($\odot$) is used instead of additive fusion, which is empirically found to be more effective.

**Design Motivation**: Multiplicative fusion allows the FuseNet output to act as a modulation gate on Backbone features—selectively enhancing or suppressing specific feature channels—which is well-suited for fine-grained control of high-frequency details.

### Training and Inference Strategy

**Training** (Algorithm 1):
1. Sample initial state $\mathcal{Z} \sim \mathcal{N}(0,1)$.
2. Randomly sample coordinates $x$, timestep $t \sim \mathcal{U}(0,1)$, and noise $n \sim \mathcal{N}(0,I)$.
3. Construct the intermediate state and compute the reconstruction loss.
4. Update parameters via gradient descent.

**Inference** (Algorithm 2):
1. Start from $\hat{g}(x)_1 = \mathcal{Z}$.
2. Iterate from $t=1$ to $t=0$ with step size $\delta$.
3. At each step: $\hat{g}(x)_{t-\delta} = \frac{\delta}{t}f_\theta(\hat{g}(x)_t, x, t) + (1 - \frac{\delta}{t})\hat{g}(x)_t$

By default, 2-step inference ($\delta = 0.5$) is used to balance quality and efficiency.

## Key Experimental Results

### Main Results

**Image Fitting** (Kodak dataset, 3-layer MLP, 300 neurons/layer):

| Baseline | PSNR | SSIM | I-version PSNR | I-version SSIM | Gain |
|----------|------|------|----------------|----------------|------|
| SIREN | 34.57 | 0.931 | **37.53** | **0.961** | +2.96 |
| WIRE | 32.15 | 0.898 | **33.73** | **0.924** | +1.58 |
| Gauss | 31.33 | 0.880 | **31.93** | **0.884** | +0.60 |

**Super-Resolution** (DIV2K dataset, trained on 2× only):

| Scale | Method | SIREN PSNR/LPIPS | WIRE PSNR/LPIPS | Gauss PSNR/LPIPS |
|-------|--------|-----------------|-----------------|------------------|
| 2× | Baseline | 26.77/0.414 | 26.14/0.457 | 25.19/0.538 |
| 2× | **I-version** | **27.64/0.367** | **27.21/0.388** | **26.82/0.363** |
| 4× | Baseline | 25.03/0.597 | 24.57/0.618 | 23.85/0.673 |
| 4× | **I-version** | **25.53/0.575** | **25.78/0.496** | **25.18/0.620** |

Note: Models are trained on 2× only; 4× results represent zero-shot generalization.

**Image Denoising** (DIV2K, Poisson noise):

| Baseline | PSNR | LPIPS | I-version PSNR | I-version LPIPS | PSNR Gain |
|----------|------|-------|----------------|-----------------|-----------|
| SIREN | 23.86 | 0.604 | **25.59** | **0.540** | +1.73 |
| WIRE | 23.32 | 0.746 | **24.76** | **0.490** | +1.44 |
| Gauss | 23.10 | 0.783 | **24.20** | **0.533** | +1.10 |

I-SIREN achieves up to **+3.25 dB PSNR** improvement in the best-case denoising scenario.

**3D Occupancy Reconstruction** (IoU):

| Method | SIREN | WIRE | Gauss |
|--------|-------|------|-------|
| Baseline | 0.9840 | 0.9917 | 0.9855 |
| **I-version** | **0.9934** | **0.9950** | **0.9967** |

### Ablation Study

**Effect of FeedbackNet and FuseNet** (I-SIREN, Kodak image fitting):

| FeedbackNet | FuseNet | PSNR |
|-------------|---------|------|
| ✗ | ✗ | 20.27 |
| ✓ (1×) | ✗ | 31.88 |
| ✓ (1×) | ✓ (1×) | **37.53** |
| ✓ (2×) | ✓ (1×) | 37.25 |
| ✓ (1×) | ✓ (2×) | 37.77 |
| ✓ (2×) | ✓ (2×) | 37.56 |

**Effect of Number of Refinement Steps**:
- PSNR peaks at steps=4; the largest gain occurs from 1 to 2 steps.
- Beyond 4 steps, PSNR slightly decreases, while perceptual quality (LPIPS) continues to improve—reflecting the Perception-Distortion Tradeoff.
- For denoising, PSNR peaks at steps=2; excessive steps begin to reconstruct noise.

**Inference Computational Cost**:

| Component | GFLOPs |
|-----------|--------|
| Backbone (executed once) | 106.8 |
| Per refinement step (FeedbackNet + FuseNet) | 0.43 |
| Total (steps=2) | 107.6 |
| Additional overhead | ~0.8% |

**Training Time Comparison** (Kodak, 2000 iterations, RTX 4090):

| Metric | SIREN | I-SIREN | Increase |
|--------|-------|---------|----------|
| Training time | baseline | +6% | marginal |
| Inference latency/image | baseline | +3ms | marginal |

### Key Findings

1. Iterative refinement yields consistent improvements across **all three activation functions** (SIREN, WIRE, Gauss), demonstrating the generality of the framework.
2. Gaussian initialization consistently outperforms zero or all-ones initialization—noise provides a better exploratory starting point.
3. I-SIREN with 3 layers surpasses a 5-layer SIREN, demonstrating that iteration is more efficient than increasing depth.
4. Multiplicative fusion consistently outperforms adaptive weighted fusion, suggesting that gating mechanisms are better suited for refinement tasks than linear combinations.
5. Statistical significance tests (5 random seeds, Wilcoxon signed-rank test): all improvements achieve p < 0.00001.

## Highlights & Insights

1. **Plug-and-play is the key selling point**: The original INR architecture remains unchanged; two tiny modules are added to obtain improvements, making the approach highly engineering-friendly.
2. **Negligible computational overhead**: 0.8–2% additional FLOPs yields up to +2.96 dB PSNR, representing an exceptionally high return on investment.
3. **Elegant transfer from diffusion models to INR**: The iterative idea of InDI is adapted to coordinate networks, with a conceptually simple yet empirically effective design.
4. **Cross-task and cross-resolution generalization**: A model trained only on 2× super-resolution can be directly applied to 4×, demonstrating the generalizability of the learned refinement capability.
5. **Natural emergence of the perception-distortion tradeoff**: As the number of steps increases, PSNR decreases while LPIPS improves, indicating that the iterative process naturally refines signals from low to high frequencies.

## Limitations & Future Work

1. Excessive inference steps lead to overfitting (reconstructing noise during denoising), necessitating careful step selection.
2. The architectural choices for FeedbackNet and FuseNet (layer count and width) are relatively simple and may leave room for further optimization.
3. The framework has not been tested in combination with positional encoding-based methods (e.g., DINER, FINER).
4. 3D scene experiments are limited to small-scale models (3 objects) and have not been validated on large-scale NeRF-level scenes.
5. The optimal number of inference steps varies by task (fitting=4, denoising=2), and an adaptive step selection mechanism is lacking.

## Related Work & Insights

- **Relation to diffusion models**: Shares the iterative refinement concept, but I-INR operates in coordinate space rather than pixel space, yielding a more compact formulation.
- **Relation to InDI**: Directly adopts InDI's iterative framework but transfers it from image restoration to INR-based signal representation.
- **Insight**: The design pattern of a single Backbone forward pass followed by lightweight iterative refinement is generalizable to other continuous representation learning scenarios, such as NeRF acceleration.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic application of iterative refinement to INR; transfers InDI to coordinate networks.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four tasks, three baselines, comprehensive ablations, and statistical significance testing.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are clear, visualizations are rich, and experimental setups are transparent.
- **Value**: ⭐⭐⭐⭐⭐ — Plug-and-play, near-zero overhead, open-source code; highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] NeMo-map: Neural Implicit Flow Fields for Spatio-Temporal Motion Mapping](../../ICLR2026/autonomous_driving/nemo-map_neural_implicit_flow_fields_for_spatio-temporal_motion_mapping.md)
- [\[CVPR 2026\] Neural Distribution Prior for LiDAR Out-of-Distribution Detection](../../CVPR2026/autonomous_driving/neural_distribution_prior_for_lidar_ood_detection.md)
- [\[CVPR 2026\] MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction](../../CVPR2026/autonomous_driving/mapgclr_geospatial_contrastive_learning_of_representations_for_online_vectorized.md)
- [\[NeurIPS 2025\] Continuous Simplicial Neural Networks](../../NeurIPS2025/autonomous_driving/continuous_simplicial_neural_networks.md)
- [\[CVPR 2026\] Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](../../CVPR2026/autonomous_driving/sgnlf_spectralgeometric_neural_fields_for_posefre.md)

</div>

<!-- RELATED:END -->
