---
title: >-
  [Paper Note] DiffFNO: Diffusion Fourier Neural Operator
description: >-
  [CVPR 2025][Physics & Scientific Computing][Super-Resolution] This paper proposes DiffFNO, which integrates the Weighted Fourier Neural Operator (WFNO) with a diffusion framework for arbitrary-scale super-resolution. It preserves critical high-frequency components through Mode Rebalancing, fuses frequency-domain and spatial-domain features using a Gated Fusion Mechanism, and accelerates inference with an adaptive-step ODE solver, outperforming existing methods by 2-4 dB in PS…
tags:
  - "CVPR 2025"
  - "Physics & Scientific Computing"
  - "Super-Resolution"
  - "Fourier Neural Operator"
  - "Diffusion Models"
  - "Arbitrary-Scale"
  - "High-Frequency Reconstruction"
date: 2026-05-08
content_hash: 1274a3f1a648e4b7
---

# DiffFNO: Diffusion Fourier Neural Operator

**Conference**: CVPR 2025  
**arXiv**: [2411.09911](https://arxiv.org/abs/2411.09911)  
**Code**: None  
**Area**: Image Restoration  
**Keywords**: Super-Resolution, Fourier Neural Operator, Diffusion Models, Arbitrary-Scale, High-Frequency Reconstruction

## TL;DR
This paper proposes DiffFNO, which integrates the Weighted Fourier Neural Operator (WFNO) with a diffusion framework for arbitrary-scale super-resolution. It preserves critical high-frequency components through Mode Rebalancing, fuses frequency-domain and spatial-domain features using a Gated Fusion Mechanism, and accelerates inference with an adaptive-step ODE solver, outperforming existing methods by 2-4 dB in PSNR across multiple benchmarks.

## Background & Motivation

**Background**: Image super-resolution (SR) has evolved from SRCNN to attention-based deep architectures, and recently to diffusion models for high-fidelity generation. Arbitrary-scale SR (which works at unseen scale factors during training) has emerged as a new trend, where neural operators (e.g., SRNO, HiNOTE) utilize function space mapping to achieve resolution-agnostic upsampling.

**Limitations of Prior Work**: Standard FNO improves efficiency via mode truncation, which discards crucial high-frequency components essential for SR. MLPs suffer from spectral bias (favoring low frequencies). Although diffusion models can generate high-fidelity details, they suffer from slow inference. Operator learning methods like SRNO, originating from physical simulations, suffer from insufficient high-frequency recovery when applied directly to real-world images.

**Key Challenge**: A conflict exists between the efficient global modeling of FNO and the heavy reliance of SR on high-frequency details—mode truncation enhances computational efficiency at the cost of high-frequency information, while diffusion models can recover high frequencies but are computationally expensive.

**Goal**: To design an arbitrary-scale SR method that preserves high-frequency information while maintaining efficient inference.

**Key Insight**: Instead of truncating Fourier modes, all modes are rebalanced using a learnable weight function to reinforce high-frequency components. Concurrently, a diffusion process is utilized for iterative refinement, accelerated by an adaptive-step ODE solver.

**Core Idea**: The WFNO retains all Fourier modes and enhances high frequencies using a frequency-dependent learnable weight $\mathbf{w}(\boldsymbol{\xi}) = 1 + \gamma \cdot \|\boldsymbol{\xi}\|^\alpha$. These features are then fused with an attention operator and iteratively refined within a diffusion framework.

## Method

### Overall Architecture
The input LR image is processed through a CNN encoder (EDSR-baseline/RDN) to extract features. The WFNO captures global dependencies in the frequency domain and enhances high frequencies, while the AttnNO captures local details in the spatial domain using Galerkin attention. A Gated Fusion Mechanism (GFM) adaptively combines features from both pathways. After projecting these features into the RGB space, reverse diffusion is executed efficiently utilizing an Adaptive-Step (ATS) ODE solver to produce the HR output.

### Key Designs

1. **Weighted Fourier Neural Operator (WFNO) and Mode Rebalancing**:

    - **Function**: Adaptively enhances high-frequency components while retaining all Fourier formats.
    - **Mechanism**: Introduces a learnable weight function $\mathbf{w}_l(\boldsymbol{\xi}) = 1 + \gamma_l \cdot \|\boldsymbol{\xi}\|^\alpha$ ($\alpha = 0.7$) into the frequency-domain convolution of standard FNO, formulated as the updated integral operator: $\mathcal{K}_l \mathbf{v}_l = \mathcal{F}^{-1}(\mathbf{w}_l \cdot \mathbf{P}_l \cdot \mathcal{F}[\mathbf{v}_l])$. For $\alpha > 0$, higher frequencies receive larger weights, and $\gamma_l$ is learned independently per layer to control the intensity of the enhancement.
    - **Design Motivation**: Standard FNO truncates high-frequency modes to reduce computation, but the SR task inherently requires high frequencies. Mode rebalancing keeps all modes and allows the network to learn the high-frequency enhancement strategy, offering greater flexibility than manual truncation.

2. **Gated Fusion Mechanism (GFM)**:

    - **Function**: Adaptively fuses the global frequency-domain features of WFNO and the local spatial-domain features of AttnNO.
    - **Mechanism**: Channels of both WFNO and AttnNO feature maps are concatenated and passed through a $1 \times 1$ convolution + sigmoid to generate a spatial gate map $\mathbf{G} \in \mathbb{R}^{B \times H \times W \times 1}$. The fused output is computed as: $\mathbf{v}_{fused} = \mathbf{G} \odot \mathbf{v}_{WFNO} + (1 - \mathbf{G}) \odot \mathbf{v}_{AttnNO}$. AttnNO runs in parallel with WFNO, sharing the same encoder and employing the Galerkin attention mechanism.
    - **Design Motivation**: WFNO excels at capturing global structure and long-range dependencies, while AttnNO excels at local textures and fine-grained details. Gated fusion dynamically balances their contributions at each spatial location.

3. **Adaptive-Step (ATS) ODE Solver**:

    - **Function**: Accelerates the reverse sampling process of the diffusion model.
    - **Mechanism**: Reformulates the stochastic reverse diffusion as a deterministic ODE solved via a numerical solver with adaptive steps. The step size is dynamically adjusted based on the complexity of image regions: larger steps for simpler regions and smaller steps for complex ones, thereby reducing computational overhead while retaining quality.
    - **Design Motivation**: Fixed-step ODE solvers apply the same number of steps to all regions, leading to wasted computational resources. The adaptive-step approach significantly cuts down inference time without sacrificing generation quality.

### Loss & Training
The score matching loss is adopted: $\mathcal{L}(\theta) = \mathbb{E}_{t, \mathbf{x}_0}[\|s_\theta(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t|\mathbf{x}_0)\|_2^2]$. The forward diffusion process utilizes a modified variance-preserving SDE, where the drift term $-\frac{1}{2}\beta(t)(\mathbf{x} - \mathbf{Dx})$ models the high-frequency degradation caused by downsampling. The noise schedule grows linearly with $\beta_{min}=0.1$ and $\beta_{max}=20$.

## Key Experimental Results

### Main Results

| Method | 4x SR PSNR | Inference Time | Arbitrary Scale |
|------|-----------|---------|---------|
| SRNO | Baseline | Slower | Supported |
| HiNOTE | Medium | Slower | Supported |
| DiffFNO | **+2-4 dB** | **Faster** | Supported |
| EDSR | Lower | Fast | Fixed Scale |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| FNO (Standard Truncation) | Lower PSNR | Loss of high frequencies |
| WFNO (Mode Rebalancing) | Significant Improvement | High-frequency preservation is critical |
| Without AttnNO | PSNR Decreases | Loss of local details |
| Naive Concat instead of GFM | PSNR Decreases | Adaptive fusion outperforms simple concatenation |
| Fixed-step ODE | Slower Inference | ATS is more efficient under equal quality |

### Key Findings
- Mode rebalancing is central to performance gains—a simple frequency-dependent weight function substantially improves high-frequency reconstruction.
- The spatially adaptive gating of GFM is more effective than naive concatenation, as different spatial locations have varying demands for global and local features.
- DiffFNO excels not only at scale factors within the training distribution but also remains robust on unseen scale factors.
- The ATS ODE solver significantly reduces inference time while keeping output quality intact.

## Highlights & Insights
- **Parsimonious Form of Frequency Rebalancing**: Working as a weight function, $1 + \gamma \cdot \|\xi\|^\alpha$ is extremely simple with only one additional scalar parameter, yet successfully tackles FNO's high-frequency loss issue. This "minimum intervention, maximum yield" design is highly instructive.
- **Synergy of Diffusion Framework and Neural Operators**: WFNO provides resolution-agnostic upsampling while the diffusion process delivers iterative refinement—the two are naturally complementary. The former ensures "correct structure" whereas the latter ensures "realistic details".
- **Dual-Pathway Design with Shared Encoder**: WFNO and AttnNO share an encoder but run in parallel in different domains (frequency vs. spatial domain), utilizing computational resources in a highly efficient manner.

## Limitations & Future Work
- Even with ATS acceleration, the inference time of the diffusion process is still longer than single forward-pass methods (e.g., EDSR).
- The hyperparameter $\alpha$ in mode rebalancing may require tuning for different datasets or tasks.
- Experiments are primarily validated on standard SR benchmarks; real-world degradation scenarios (blur + noise + compression) have not yet been explored.
- Comparison with recent Transformer-based SR methods (e.g., SwinIR, HAT) is relatively incomplete.

## Related Work & Insights
- **vs SRNO**: SRNO employs standard FNO + Galerkin attention but lacks frequency rebalancing and diffusion refinement. Based on this, DiffFNO introduces WFNO and a diffusion process, achieving a 2-4 dB PSNR improvement.
- **vs HiNOTE**: HiNOTE possesses its own encoder, but its high-frequency reconstruction remains insufficient. DiffFNO directly enhances high frequencies from the frequency domain via mode rebalancing.
- **vs SRDiff/SR3**: Pure diffusion SR methods suffer from extremely slow inference and may not support arbitrary scales. DiffFNO integrates the resolution-independence of operator learning with the high fidelity of diffusion.

## Rating
- Novelty: ⭐⭐⭐⭐ The mode rebalancing of WFNO and its integration with the diffusion framework are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-scale evaluation, comprehensive ablation, and comparison with various baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical derivations, intuitive architecture diagrams, and detailed method descriptions.
- Value: ⭐⭐⭐⭐ Introducing Fourier mode rebalancing to diffusion SR for the first time, establishing a new standard for arbitrary-scale SR.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Tucker-FNO: Tensor Tucker-Fourier Neural Operator and its Universal Approximation Theory](../../ICLR2026/physics/tucker-fno_tensor_tucker-fourier_neural_operator_and_its_universal_approximation.md)
- [\[CVPR 2026\] Spatial-Spectral Residuals Informed Diffusion Neural Operator for Pan-sharpening](../../CVPR2026/physics/spatial-spectral_residuals_informed_diffusion_neural_operator_for_pan-sharpening.md)
- [\[ICLR 2026\] Extending Fourier Neural Operators for Modeling Parameterized and Coupled PDEs](../../ICLR2026/physics/extending_fourier_neural_operators_for_modeling_parameterized_and_coupled_pdes.md)
- [\[ICLR 2026\] Proximal Diffusion Neural Sampler](../../ICLR2026/physics/proximal_diffusion_neural_sampler.md)
- [\[NeurIPS 2025\] Physics-Informed Neural Networks with Fourier Features and Attention-Driven Decoding](../../NeurIPS2025/physics/physics-informed_neural_networks_with_fourier_features_and_attention-driven_deco.md)

</div>

<!-- RELATED:END -->
