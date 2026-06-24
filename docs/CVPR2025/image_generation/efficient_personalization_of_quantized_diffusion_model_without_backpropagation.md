---
title: >-
  [Paper Note] Efficient Personalization of Quantized Diffusion Model without Backpropagation (ZOODiP)
description: >-
  [CVPR 2025][Image Generation][Diffusion Model Personalization] This paper proposes ZOODiP, which achieves personalization (Textual Inversion) on quantized diffusion models via zeroth-order optimization. By utilizing subspace gradient projection for denoising and partial uniform timestep sampling to accelerate training, it achieves personalization quality comparable to gradient-based methods using only 2.37GB of VRAM and forward passes, reducing memory usage by up to 8.2x.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Diffusion Model Personalization"
  - "Quantized Models"
  - "Zeroth-Order Optimization"
  - "Memory-Efficient Fine-Tuning"
  - "Edge Devices"
date: 2026-05-08
content_hash: a3af21f89b2448d9
---

# Efficient Personalization of Quantized Diffusion Model without Backpropagation (ZOODiP)

**Conference**: CVPR 2025  
**arXiv**: [2503.14868](https://arxiv.org/abs/2503.14868)  
**Code**: [https://github.com/ignoww/ZOODiP_project](https://github.com/ignoww/ZOODiP_project)  
**Area**: Diffusion Models  
**Keywords**: Diffusion Model Personalization, Quantized Models, Zeroth-Order Optimization, Memory-Efficient Fine-Tuning, Edge Devices

## TL;DR

This paper proposes ZOODiP, which achieves personalization (Textual Inversion) on quantized diffusion models via zeroth-order optimization. By utilizing subspace gradient projection for denoising and partial uniform timestep sampling to accelerate training, it achieves personalization quality comparable to gradient-based methods using only 2.37GB of VRAM and forward passes, reducing memory usage by up to 8.2x.

## Background & Motivation

**Background**: Diffusion models have shown outstanding performance in image generation. Personalization tasks—customizing the generation of specific concepts using a few user images—represent a crucial application scenario. Major methods include fine-tuning the denoising network (DreamBooth) and optimizing text tokens (Textual Inversion). While quantization has successfully compressed inference memory, training or fine-tuning quantized models still demands substantial memory.

**Limitations of Prior Work**: (1) Most existing methods rely on backpropagation, making them unsuitable for mobile processors that only accelerate inference; (2) storing activations and gradients consumes extensive memory, requiring 6-8GB+ even with parameter-efficient methods like LoRA/QLoRA; (3) evolutionary strategies (like GF-TI), despite being backpropagation-free, require 30 forward passes per iteration, leading to extremely slow and unstable training.

**Key Challenge**: Achieving high-quality personalization on edge devices with severely constrained memory requires simultaneously addressing three challenges: "backpropagation-free training," "non-differentiability of quantized models," and "high noise in zeroth-order gradients."

**Goal**: (1) How to estimate effective gradients on quantized (non-differentiable) models? (2) How to reduce gradient noise in zeroth-order optimization to accelerate convergence? (3) How to select the most effective training timesteps within limited iterations?

**Key Insight**: The authors observe three key phenomena: zeroth-order optimization can handle non-differentiable objectives; token variations during Textual Inversion training concentrate in a low-dimensional subspace; and the influence of text embeddings varies significantly across different diffusion timesteps.

**Core Idea**: Training token embeddings on quantized diffusion models using zeroth-order optimization, combined with subspace gradient projection for denoising and partial uniform timestep sampling, enables ultra-low-memory personalization utilizing only forward passes.

## Method

### Overall Architecture

ZOODiP is based on the Textual Inversion framework: the input consists of a few reference images and a text prompt containing a new token $v^*$, and the output is the optimized token embedding. The core workflow is: (1) apply INT8 quantization to all components of Stable Diffusion (U-Net, VAE, and text encoder); (2) replace backpropagation with Randomized Gradient Estimation (RGE) to estimate gradients; (3) project out noisy dimensions utilizing Subspace Gradient (SG) projection; (4) focus on effective timesteps via Partial Uniform Timestep Sampling (PUTS). The entire training process requires only forward passes, without storing activations or gradients.

### Key Designs

1. **ZO with Quantized Model**:

    - **Function**: Estimate gradients on non-differentiable quantized models, eliminating backpropagation.
    - **Mechanism**: Apply a random perturbation $\mu e_i$ ($e_i \sim \mathcal{N}(0,I)$) to the token embedding $\theta$, and estimate the gradient by computing the difference in loss before and after perturbation: $\hat{g}_\theta = \frac{1}{n}\sum_{i=1}^{n}\frac{\mathcal{L}(\theta+\mu e_i)-\mathcal{L}(\theta)}{\mu}e_i$. Set $n=2$ random directions and a perturbation size $\mu=10^{-3}$. Symmetric N-bit quantization is adopted to map weights to integer ranges, which are then restored using scale and zero-point.
    - **Design Motivation**: The rounding function introduced by quantization renders the model non-differentiable, making traditional backpropagation inapplicable. Zeroth-order optimization estimates gradients with only forward passes, perfectly matching quantized models while avoiding storing intermediate activations, thereby drastically reducing memory overhead.

2. **Subspace Gradient (SG)**:

    - **Function**: Analyze the token trajectory via PCA and project out noisy dimensions in the gradient to accelerate convergence.
    - **Mechanism**: Every $\tau=128$ iterations, the updated tokens are stored in a trajectory buffer $B \in \mathbb{R}^{\tau \times d}$. SVD is performed on the normalized $\bar{B}$ to obtain eigenvalues and eigenvectors. Based on a threshold $\nu=10^{-3}$, the minimal index $i^*$ where the cumulative explanation variance ratio exceeds $1-\nu$ is identified, and the remaining low-variance eigenvectors are used to construct the projection matrix $P_\nu$. At each gradient update, noise is projected out as: $\hat{g}' = \hat{g}(I - P_\nu^\top P_\nu)$. Experiments demonstrate that over 80% of dimensions are projected out.
    - **Design Motivation**: Zeroth-order gradient estimation is inherently noisy, which is exacerbated in personalization tasks with only 1–5 images. The authors find that token variations in Textual Inversion are concentrated in a low-dimensional subspace (retaining only 1/3 of the dimensions is sufficient to maintain the concept), allowing high-noise dimensions to be safely projected out.

3. **Partial Uniform Timestep Sampling (PUTS)**:

    - **Function**: Focus sampling on the interval of timesteps where text embeddings have the most significant impact to enhance training efficiency.
    - **Mechanism**: Uniformly sample timesteps from $U(T_L, T_U)$ ($T_L=500, T_U=900$), skipping low-noise timesteps where the text has a weak influence. Empirical evaluation demonstrates that sampling from $U(0,500)$ fails to learn concept features like color and shape of the reference image, whereas sampling from $U(500,1000)$ succeeds.
    - **Design Motivation**: Diffusion models can be viewed as timestep-based mixture-of-experts systems, where the influence of text conditions varies dramatically across different timesteps. Concentrating the limited training iterations on the timesteps most influenced by the text maximizes the learning effect of each iteration.

### Loss & Training

The standard LDM denoising loss $\mathcal{L}_{LDM} = \mathbb{E}[\|\epsilon - \epsilon_\phi(z_t, t, c(y^*))\|_2^2]$ is used. Only the token embedding $\theta$ is optimized, while the model weights are fully frozen and quantized. The ZOAdam optimizer is employed with a learning rate of $\eta = 5 \times 10^{-3}$, a total of 30,000 iterations, and a batch size of 1.

## Key Experimental Results

### Main Results

Evaluated on 30 subjects in the DreamBooth dataset, using 25 prompts per subject, and generating 5 images per prompt (3750 images in total).

| Method | Quantization | Grad-Free | Memory (GB) | CLIP-T↑ | CLIP-I↑ | DINO↑ |
|------|------|--------|----------|---------|---------|-------|
| DreamBooth | ✗ | ✗ | 19.4 | 0.281 | 0.782 | 0.592 |
| QLoRA | ✓ | ✗ | 7.56 | 0.297 | 0.762 | 0.607 |
| TuneQDM | ✓ | ✗ | 8.96 | 0.289 | 0.788 | 0.555 |
| Textual Inversion | ✗ | ✗ | 6.75 | 0.285 | 0.778 | 0.559 |
| GF-TI | ✓ | ✓ | 2.37 | 0.253 | 0.540 | 0.011 |
| **ZOODiP (Ours)** | ✓ | ✓ | **2.37** | **0.287** | **0.772** | **0.558** |

### Ablation Study

| Configuration | CLIP-T↑ | CLIP-I↑ | DINO↑ |
|------|---------|---------|-------|
| Base ZO (w/o SG w/o PUTS) | 0.273 | 0.736 | 0.505 |
| +SG | 0.265 | 0.747 | 0.527 |
| +PUTS | 0.277 | 0.744 | 0.562 |
| +SG +PUTS (Full Model) | 0.266 | 0.759 | 0.569 |

### Key Findings

- The joint use of SG and PUTS improves the DINO metric from 0.505 to 0.569 (+12.7%), demonstrating their complementarity.
- ZOODiP outscores GF-TI by 43% on CLIP-I, displaying a world of difference in quality under the same 2.37GB memory footprint.
- Training speed: ZOODiP ($n=2$, INT8) reaches 16.1 iter/s, which is 1.7x faster than TI and 22x faster than GF-TI.
- Hyperparameters $\tau=128$ and $\nu=10^{-3}$ are optimal; the timestep range $[500, 900]$ yields the best results across all metrics.
- Despite $\nu$ being small, SG still projects out over 80% of the dimensions, indicating that the effective dimensionality of token optimization is extremely low.

## Highlights & Insights

- **Subspace gradient projection is an elegant denoising strategy**: It leverages the low-rank structure of the optimization trajectory itself to denoise, introducing no extra models or parameters, and incurring minimal computational overhead (PCA dimension is only 128). This concept can be transferred to any zeroth-order optimization scenario.
- **The ultimate combination for memory saving**: Quantization (reducing weight memory) + Zeroth-Order optimization (eliminating activation and gradient memory) + Token-only optimization (minimal parameters). This configuration achieves an 8.2x memory compression, requiring only 3KB of storage for the optimized token.
- **Practical insights on timestep sampling**: It explicitly reveals that the text condition has the most significant impact when $t \in [500, 900]$, providing valuable references for all Textual Inversion-like methods.

## Limitations & Future Work

- Zeroth-order optimization requires 30,000 iterations (approx. 30 minutes); although much faster than GF-TI, it remains relatively slow.
- Validation is only conducted on Stable Diffusion v1.5, without extension to SDXL or larger models.
- The representation capacity of tokens is limited (single token), and the fidelity for complex concepts is lower than that of full-model fine-tuning via DreamBooth.
- The optimal interval $[T_L, T_U]$ for PUTS needs to be determined through additional empirical trials, which may vary across different model versions.

## Related Work & Insights

- **vs Textual Inversion**: TI optimizes tokens using backpropagation, requiring 6.75GB VRAM; ZOODiP replaces it with zeroth-order optimization, reducing the memory to 2.37GB while maintaining close performance.
- **vs GF-TI**: Both are gradient-free methods, but GF-TI relies on evolutionary strategies requiring 30 forward passes per step and yields poor performance (CLIP-I of only 0.540). ZOODiP uses RGE requiring only 2–3 forward passes, yielding significantly better quality.
- **vs TuneQDM**: TuneQDM fine-tunes quantization parameters on the quantized model but still requires backpropagation (8.96GB), whereas ZOODiP completely bypasses backpropagation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of zeroth-order optimization, quantization, and subspace projection is innovative, though individual components are not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Ablations are detailed and hyperparameter analyses are sufficient, but validation is limited to SD v1.5.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic, natural motivational derivation, and good experimental visualization.
- **Value**: ⭐⭐⭐⭐ Highly practical value for on-device personalization; the subspace gradient concept is inspiring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Memory-Efficient Fine-Tuning for Quantized Diffusion Model](../../ECCV2024/image_generation/memory-efficient_fine-tuning_for_quantized_diffusion_model.md)
- [\[CVPR 2025\] CleanDIFT: Diffusion Features without Noise](cleandift_diffusion_features_without_noise.md)
- [\[CVPR 2025\] InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment](inpo_inversion_preference_optimization_diffusion_alignment.md)
- [\[CVPR 2025\] Learning Visual Generative Priors without Text](learning_visual_generative_priors_without_text.md)
- [\[CVPR 2025\] Data-Free Group-Wise Fully Quantized Winograd Convolution via Learnable Scales](data-free_group-wise_fully_quantized_winograd_convolution_via_learnable_scales.md)

</div>

<!-- RELATED:END -->
