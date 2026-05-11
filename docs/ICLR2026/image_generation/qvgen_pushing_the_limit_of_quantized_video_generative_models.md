---
title: >-
  [Paper Note] QVGen: Pushing the Limit of Quantized Video Generative Models
description: >-
  [ICLR 2026][Image Generation][Video Diffusion Models] This paper proposes QVGen, a quantization-aware training (QAT) framework for video diffusion models. It introduces auxiliary modules to reduce gradient norms and impr…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Video Diffusion Models"
  - "Quantization-Aware Training"
  - "Low-Bit Quantization"
  - "Rank Decay Strategy"
  - "Auxiliary Modules"
date: 2026-05-08
content_hash: f14755a4de181634
---

# QVGen: Pushing the Limit of Quantized Video Generative Models

**Conference**: ICLR 2026
**arXiv**: [2505.11497](https://arxiv.org/abs/2505.11497)
**Code**: [https://github.com/ModelTC/QVGen](https://github.com/ModelTC/QVGen)
**Area**: Image Generation
**Keywords**: Video Diffusion Models, Quantization-Aware Training, Low-Bit Quantization, Rank Decay Strategy, Auxiliary Modules

## TL;DR

This paper proposes QVGen, a quantization-aware training (QAT) framework for video diffusion models. It introduces auxiliary modules to reduce gradient norms and improve convergence, and designs a rank decay strategy to progressively eliminate the inference overhead of auxiliary modules during training. QVGen is the first method to achieve near full-precision video generation quality under 4-bit quantization.

## Background & Motivation

- **Background**: Video diffusion models (e.g., CogVideoX, Wan) can generate high-quality videos but demand enormous computation and memory — Wan 14B requires over 30 minutes and 50 GB of VRAM on a single H100 to generate a 10-second 720p video. Model quantization is an effective compression approach: 4-bit quantization can achieve approximately 3× speedup and 4× model size reduction.
- **Limitations of Prior Work**: Directly transferring quantization methods from image diffusion models to video diffusion models yields poor results. Existing QAT methods (e.g., Q-DM, EfficientDM, LSQ) suffer severe quality degradation under 4-bit video quantization.
- **Key Challenge**: Quantized video models exhibit significant **convergence difficulties**.

## Method

### Overall Architecture

QVGen consists of two core components:
1. **Auxiliary Module $\Phi$**: Attached to quantized linear layers to compensate for quantization error and reduce gradient norms to improve convergence.
2. **Rank Decay Strategy**: Progressively eliminates $\Phi$ via SVD decomposition and rank regularization, ensuring no additional inference overhead at deployment.

### Key Design 1: Auxiliary Modules for Improved Convergence

**Theoretical Analysis**: Based on regret analysis, the upper bound on average regret is:

$$\frac{R(T)}{T} \leq \frac{dD_\infty^2}{2T\eta_T^m} + \frac{1}{T}\sum_{t=1}^{T}\frac{\eta_t^M}{2}\|\mathbf{g}_t\|_2^2$$

When the number of training steps $T$ is sufficiently large, the first term becomes negligible. Thus, **minimizing the gradient norm $\|\mathbf{g}_t\|_2$** is key to improving QAT convergence.

With the auxiliary module $\Phi$, the forward computation of the quantized linear layer becomes:

$$\hat{\mathbf{Y}} = \mathcal{Q}_b(\mathbf{W})\mathcal{Q}_b(\mathbf{X}) + \Phi(\mathcal{Q}_b(\mathbf{X}))$$

where $\Phi(\mathcal{Q}_b(\mathbf{X})) = \mathbf{W}_\Phi \mathcal{Q}_b(\mathbf{X})$, and $\mathbf{W}_\Phi$ is initialized as the weight quantization error $\mathbf{W} - \mathcal{Q}_b(\mathbf{W})$.

### Key Design 2: Rank Decay Strategy

$\Phi$ introduces additional full-precision matrix multiplication overhead at inference and must be progressively removed during training.

**Key Observation**: SVD analysis of $\mathbf{W}_\Phi$ reveals that the proportion of small singular values grows from 73% (step 0) to 99% (step 2K) as training progresses, indicating that an increasing fraction of components contribute minimally.

Procedure:
1. Perform SVD decomposition on $\mathbf{W}_\Phi$: $\mathbf{W}_\Phi = \sum_{s=1}^d \sigma_s \mathbf{u}_s \mathbf{v}_s^\top$
2. Rewrite in low-rank form: $\Phi(\mathcal{Q}_b(\mathbf{X})) = \mathbf{L}\mathbf{R}\mathcal{Q}_b(\mathbf{X})$
3. Apply rank regularization $\boldsymbol{\gamma}$:

$$\hat{\mathbf{Y}} = \mathcal{Q}_b(\mathbf{W})\mathcal{Q}_b(\mathbf{X}) + (\boldsymbol{\gamma} \odot \mathbf{L})\mathbf{R}\mathcal{Q}_b(\mathbf{X})$$

where $\boldsymbol{\gamma} = \text{concat}([1]_{n \times (1-\lambda)r}, [u]_{n \times \lambda r})$, and $u$ decays from 1 to 0 via cosine annealing.

4. Once $u$ reaches 0, truncate low-contribution components, reducing the rank from $r$ to $(1-\lambda)r$.
5. Repeat until $r=0$, fully eliminating $\Phi$.

### Loss & Training

A knowledge distillation (KD) objective is adopted with the full-precision model as teacher:

$$\mathcal{L} = \mathbb{E}_{\mathbf{x}_0, \mathcal{C}, \tau}\left[\|\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_\tau, \mathcal{C}, \tau) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_\tau, \mathcal{C}, \tau)\|_F^2\right]$$

## Key Experimental Results

### Main Results

Results on VBench:

| Method | Bits (W/A) | Imaging Quality↑ | Dynamic Degree↑ | Scene Consistency↑ |
|--------|-----------|-----------------|-----------------|-------------------|
| CogVideoX-2B Full Precision | 16/16 | 59.15 | 67.78 | 36.24 |
| SVDQuant (PTQ) | 4/6 | 58.27 | 40.83 | 27.69 |
| Q-DM (QAT) | 4/4 | 54.96 | 48.61 | 28.02 |
| **QVGen (Ours)** | **4/4** | **60.16** | **67.22** | **31.42** |
| **QVGen (Ours)** | **3/3** | **58.36** | **53.89** | **23.85** |

3-bit QVGen surpasses Q-DM by +25.28 on Dynamic Degree and +8.43 on Scene Consistency.

### Ablation Study

| Component | FID↓ |
|-----------|------|
| No auxiliary module (plain QAT) | Poor baseline |
| Auxiliary module + decay all parameters directly | Suboptimal |
| Auxiliary module + rank decay ($\lambda=1/2$) | **Best** |

### Key Findings

- QVGen is the first video QAT method to achieve full-precision-comparable quality under 4-bit quantization.
- The framework generalizes across both CogVideoX and Wan video model families.
- When applied to Wan 14B (one of the largest open-source video models), negligible performance loss is observed on VBench-2.0.
- Gradient norm analysis confirms that $\|\mathbf{g}_t\|_2$ in QVGen consistently remains lower than in Q-DM.

## Highlights & Insights

- First work to theoretically analyze convergence in video QAT, establishing the relationship between gradient norms and convergence behavior.
- The rank decay strategy is elegantly designed, exploiting the natural shrinkage of singular values observed during training.
- Significantly outperforms all baselines at the extreme low-bit regimes of 3-bit and 4-bit quantization.

## Limitations & Future Work

- Training cost is high (Wan 14B requires 32×H100 GPUs for 16 epochs).
- A full-precision teacher model is required for knowledge distillation.
- Current validation is limited to linear layer quantization; other components such as attention mechanisms are not yet addressed.

## Related Work & Insights

- **PTQ Methods**: Post-training quantization approaches such as ViDiT-Q and SVDQuant show limited effectiveness at very low bit-widths.
- **QAT Methods**: Quantization-aware training methods including Q-DM, EfficientDM, and LSQ face convergence difficulties on video models.
- **Model Compression**: Alternative compression techniques such as low-rank decomposition and pruning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of auxiliary modules and rank decay strategy is novel.
- **Theory**: ⭐⭐⭐⭐ — Convergence analysis is rigorous and well-grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 4 state-of-the-art video models ranging from 1.3B to 14B parameters.
- **Value**: ⭐⭐⭐⭐⭐ — Directly addresses a critical bottleneck in deploying video generative models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Purrception: Variational Flow Matching for Vector-Quantized Image Generation](purrception_variational_flow_matching_for_vector-quantized_image_generation.md)
- [\[NeurIPS 2025\] EditInfinity: Image Editing with Binary-Quantized Generative Models](../../NeurIPS2025/image_generation/editinfinity_image_editing_with_binary-quantized_generative_models.md)
- [\[ICLR 2026\] Beyond Confidence: The Rhythms of Reasoning in Generative Models](beyond_confidence_the_rhythms_of_reasoning_in_generative_models.md)
- [\[ICLR 2026\] LVTINO: LAtent Video consisTency INverse sOlver for High Definition Video Restoration](lvtino_latent_video_consistency_inverse_solver_for_high_definition_video_restora.md)
- [\[ICLR 2026\] NeuralOS: Towards Simulating Operating Systems via Neural Generative Models](neuralos_towards_simulating_operating_systems_via_neural_generative_models.md)

</div>

<!-- RELATED:END -->
