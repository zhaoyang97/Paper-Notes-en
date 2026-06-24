---
title: >-
  [Paper Note] EasyInv: Toward Fast and Better DDIM Inversion
description: >-
  [ICML 2025][LLM Efficiency][DDIM Inversion] Proposes EasyInv, which periodically aggregates the current latent state with the previous latent state via a weighted sum (analogous to Kalman filtering) during inversion. This enhances the influence of the initial latent and suppresses noise accumulation errors, achieving comparable or even superior inversion quality to iterative methods without requiring any iterative optimization, while speeding up the inference by approximately…
tags:
  - "ICML 2025"
  - "LLM Efficiency"
  - "DDIM Inversion"
  - "Diffusion Models"
  - "Image Editing"
  - "Kalman Filter"
  - "Inference Acceleration"
date: 2026-05-08
content_hash: 7ef10435f720e024
---

# EasyInv: Toward Fast and Better DDIM Inversion

**Conference**: ICML 2025  
**arXiv**: [2408.05159](https://arxiv.org/abs/2408.05159)  
**Code**: [potato-kitty/EasyInv](https://github.com/potato-kitty/EasyInv)  
**Area**: Diffusion Models / Image Inversion  
**Keywords**: DDIM Inversion, Diffusion Models, Image Editing, Kalman Filter, Inference Acceleration

## TL;DR

Proposes EasyInv, which periodically aggregates the current latent state with the previous latent state via a weighted sum (analogous to Kalman filtering) during inversion. This enhances the influence of the initial latent and suppresses noise accumulation errors, achieving comparable or even superior inversion quality to iterative methods without requiring any iterative optimization, while speeding up the inference by approximately 3x.

## Background & Motivation

DDIM Inversion is a foundational technique for editing real images using diffusion models: given a real image, it maps it back to the noise space by reversing the denoising process, and then performs controllable denoising generation/editing starting from this noise. However, DDIM Inversion is inherently an approximation process. At each inversion step, a deviation exists between the predicted noise $\varepsilon_t^*$ and the expected noise $\varepsilon_t$ during sampling. This error gradually accumulates, leading to a degradation in reconstruction quality.

Existing improvement methods primarily focus on **iteratively optimizing noise estimation**:

- **ReNoise**: Repeatedly adds noise and denoises within each timestep, using the final denoised result as the input for the next iteration. However, it may output completely black images under certain specific inputs.
- **Fixed-Point Iteration (Pan et al.)**: Grounded in fixed-point theory, it blends noise estimations of adjacent iterations at each step. However, it performs poorly on models with limited precision (e.g., SD-V1.4) and requires multiple forward passes, leading to low efficiency.
- **Null-Text Inversion / PTI**: Optimizes condition vectors in the reconstruction branch, leading to high computational overhead.

The common issues of these methods are: **(1) Low computational efficiency** — requiring multiple model forward passes per timestep; **(2) Insufficient robustness** — when model precision (e.g., float16) or capability is limited, iterative noise optimization does not guarantee convergence to a good result and may even exacerbate accumulated errors.

EasyInv fundamentally shifts this perspective: **instead of iteratively optimizing the noise at each step, it directly enhances the influence of the initial latent state throughout the entire inversion process**.

## Method

### Overall Architecture

The core idea of EasyInv is highly straightforward: based on standard DDIM Inversion, at selected timesteps $\bar{t}$, it performs a weighted aggregation of the current inverted latent state $\mathbf{z}_{\bar{t}}^*$ and the previous latent state $\mathbf{z}_{\bar{t}-1}^*$:

$$\mathbf{z}_{\bar{t}}^{*} = \eta \cdot \mathbf{z}_{\bar{t}}^{*} + (1-\eta) \cdot \mathbf{z}_{\bar{t}-1}^{*}$$

where $\eta$ is a trade-off parameter. This operation requires no additional model forward passes and thus incurs almost zero computational overhead.

**Algorithm flow** (Algorithm 1):

1. Input: Inversion algorithm $Inv()$, total inversion steps $T$, initial latent $z_0$, selected step set $\bar{t}$, parameter $\eta$
2. For each timestep $t = 0 \to T-1$:
    - Execute standard inversion: $z_{t+1} = Inv(z_t, t)$
    - If $t \in \bar{t}$: $z_{t+1} = \eta \cdot z_{t+1} + (1-\eta) \cdot z_t$
3. Output: Inverted noise latent $z_T$

### Key Designs

#### 1. Perspective Shift from Noise Optimization to Latent Aggregation

The paper first derives the expansion of the inversion process. Let $\bar{\alpha}_t = \sqrt{\alpha_t / \alpha_{t-1}}$, and $\bar{\beta}_t$ be the corresponding noise coefficient, then:

$$\mathbf{z}_t^* = \bar{\alpha}_t \mathbf{z}_{t-1}^* + \bar{\beta}_t \varepsilon_t^*$$

Recursively expanding this to the initial state:

$$\mathbf{z}_t^* = \left(\prod_{i=1}^{t} \bar{\alpha}_i\right) \mathbf{z}_0^* + \sum_{i=1}^{t} \left(\bar{\beta}_i \prod_{j=i+1}^{t} \bar{\alpha}_j\right) \varepsilon_i^*$$

This indicates that $\mathbf{z}_t^*$ is a weighted sum of the initial latent $\mathbf{z}_0^*$ and a sequence of noise terms. While prior methods focused on optimizing the noise $\varepsilon_i^*$ at each step, EasyInv chooses to **directly boost the weight of $\mathbf{z}_0^*$**, suppressing noise accumulation by blending in the previous latent (which contains more initial information).

#### 2. Kalman Filter Theoretical Support

The paper establishes a theoretical connection between EasyInv and Kalman filtering. In classic Kalman filtering, the system fuses the predicted value $\bar{x}_k$ and the measurement $y_k$ to obtain an optimal estimate:

$$\tilde{x}_k = \bar{x}_k + K(y_k - H\bar{x}_k)$$

Applying this to the inversion problem:

- **Predicted value** $\bar{x}_k \to \mathbf{z}_{\bar{t}}^*$: Output of the DDIM Inversion at the current step
- **Measurement** $y_k \to \mathbf{z}_{\bar{t}-1}^*$: The latent state of the previous step (since the difference $z_t - z_{t-1}$ can be viewed as Gaussian noise)
- **Kalman Gain** $K \to \eta$: Simplified to an empirical constant in this paper

This yields: $\mathbf{z}_{\bar{t}}^* = \eta \cdot \mathbf{z}_{\bar{t}}^* + (1-\eta) \cdot \mathbf{z}_{\bar{t}-1}^*$

This is essentially a **simplified Kalman filter** that bypasses the dynamic calculation of Kalman gain, substituting it with a fixed $\eta$.

#### 3. Noise Distribution Preservation Property

From the perspective of the aggregation operation, the distribution of $\mathbf{z}_{\bar{t}}^* - \mathbf{z}_{\bar{t}-1}^*$ maintains the same variance direction as the transition distribution of the original DDIM Inversion, but its mean is scaled down by a factor of $\eta$. This means EasyInv **retains the noise pattern of the original method, applying only a rescaling factor**, without introducing out-of-distribution modifications.

#### 4. Hyperparameter Selection

- **Aggregation timesteps $\bar{t}$**: Choose to execute the aggregation operation during the steps within $0.05T < \bar{t} < 0.25T$. The early stage of inversion is closest to the initial latent, making this phase the most effective for introducing correction signals.
- **Trade-off parameter $\eta$**: Set to 0.5 by default, meaning equal weights for both the current step and the previous step.
- **Total steps $T$**: Default is 50 steps.

### Loss & Training

EasyInv is a **training-free, test-time method**. It involves no loss functions or gradient updates, only inserting simple linear weighted operations into the standard DDIM Inversion pipeline. This allows it to be **plug-and-play and combinable with any existing inversion method** (such as combining with DirectInv, labeled as "Ours+DirectInv" in the paper).

## Key Experimental Results

### Main Results

Sampling 2,298 images from the COCO 2017 test/validation sets, using SD V1.4, GTX 3090 GPU, float16 precision:

| Method | LPIPS↓ | SSIM↑ | PSNR↑ | Inference Time↓ |
|------|--------|-------|-------|-----------|
| DDIM Inversion | 0.328 | 0.621 | 29.717 | 5s |
| ReNoise | 0.316 | 0.641 | 31.025 | 16s |
| Fixed-Point Iteration | 0.373 | 0.563 | 29.107 | 14s |
| **EasyInv (Ours)** | **0.321** | **0.646** | **30.189** | **5s** |

### Ablation Study

| Configuration | LPIPS↓ | SSIM↑ | PSNR↑ | Time↓ | Description |
|------|--------|-------|-------|-------|------|
| Full Precision (float32) | 0.321 | 0.646 | 30.184 | 9s | Full precision baseline |
| Half Precision (float16) | 0.321 | 0.646 | 30.189 | 5s | Half precision without quality loss, speed improved by 44% |

### Downstream Editing Tasks Comparison

| Inversion Method | Editing Method | Structure Dist.↓ | PSNR↑ | LPIPS↓ | SSIM↑ |
|----------|----------|------------------|-------|--------|-------|
| DDIM | P2P | 69.43 | 17.87 | 208.80 | 71.14 |
| DirectInv | P2P | 11.65 | 27.22 | 54.55 | 84.76 |
| **Ours+DirectInv** | P2P | **11.58** | **27.30** | **53.52** | **84.80** |
| DDIM | PnP | 28.22 | 22.28 | 113.46 | 79.05 |
| DirectInv | PnP | 24.29 | 22.46 | 106.06 | 79.68 |
| **Ours+DirectInv** | PnP | **22.88** | **22.56** | **102.34** | **80.27** |

### Key Findings

1. **Significant Speed Advantage**: EasyInv achieves the same inference time as vanilla DDIM Inversion (5s), which is about 3 times faster than iterative methods.
2. **Robustness at Low Precision**: No degradation of quality is observed under float16, whereas Fixed-Point Iteration collapses when using low precision or weaker models.
3. **Plug-and-Play**: Consistently improves performance across all downstream editing tasks when combined with DirectInv.
4. **Faster Convergence**: Visualization of latent states shows that the intermediate latent of EasyInv is already closer to the original image during the denoising process.

## Highlights & Insights

1. **Minimalist Design with Solid Theory**: The core operation requires only one line of code (linear interpolation), yet provides a clear theoretical exposition through mathematical expansion and the Kalman filtering perspective.
2. **The Power of Perspective Shift**: Moving from "optimizing noise at each step" to "enhancing the influence of the initial latent" is an elegant paradigm shift.
3. **Pareto Improvement in Efficiency and Quality**: Achieves simultaneous advantages in both quality (optimal SSIM) and speed (fastest method).
4. **Plug-and-play Universality**: Overlays on any existing inversion method without modifying the underlying model or the inversion framework.
5. **Half-Precision Friendly**: Since half-precision is standard for practical deployment, the stable performance of EasyInv under float16 is a major practical advantage.

## Limitations & Future Work

1. **Fixed Kalman Gain**: Since $\eta$ is set to a constant 0.5, it is not dynamically adjusted based on the error at each step. Incorporating an adaptive $\eta(t)$ might further elevate performance.
2. **Empirical Setting of Aggregation Timesteps**: The choice of $0.05T < \bar{t} < 0.25T$ lacks a rigorous theoretical proof and might need recalibration across different datasets/models.
3. **Only Evaluated on SD-family Models**: Not tested on newer architectures such as DiT, leaving generalizability unverified.
4. **Editing Quality Ceiling**: The improvement margin on editing fidelity metrics like CLIP Similarity is relatively modest.
5. **Theoretical Analysis Can Be Deepened**: The Kalman filter analogy remains somewhat rough. Producing a closed-form optimal $\eta$ or error bound would strengthen the theoretical contribution.

## Related Work & Insights

- **DDIM Inversion (Couairon et al., 2023)**: The foundational framework for this work.
- **ReNoise (Garibi et al., 2024)**: Inversion optimization method based on iterative noising/denoising.
- **Fixed-Point Iteration (Pan et al., 2023)**: Formulates inversion noise optimization as a fixed-point problem.
- **PNPInversion / DirectInv (Ju et al., 2024)**: Corrects by computing distances between latents in the reconstruction branch.
- **Null-Text Inversion (Mokady et al., 2023)**: Enhances reconstruction by optimizing null-text embeddings.
- **Insights**: Simple interpolation in latent space (similar to the EMA concept) might be effective across various stages of diffusion models; EasyInv successfully generalizes this to the inversion process.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | Novel perspective shift, but the core operation (linear interpolation) is very simple |
| Theoretical Depth | 3.5 | Insightful Kalman filter analogy, but the analysis remains qualitative |
| Experimental Thoroughness | 4 | Main results + precision comparison + downstream tasks, covering comprehensively |
| Practical Value | 4.5 | Plug-and-play, zero extra cost, half-precision friendly, high deployment value |
| Writing Quality | 3.5 | Clean derivations, but some sections are slightly wordy |
| **Overall** | **4** | A simple yet effective method with outstanding engineering value |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference](../../ACL2025/llm_efficiency/smarter_better_faster_longer_a_modern_bidirectional_encoder_for_fast_memory_effi.md)
- [\[NeurIPS 2025\] Advancing Expert Specialization for Better MoE](../../NeurIPS2025/llm_efficiency/advancing_expert_specialization_for_better_moe.md)
- [\[NeurIPS 2025\] SkyLadder: Better and Faster Pretraining via Context Window Scheduling](../../NeurIPS2025/llm_efficiency/skyladder_better_and_faster_pretraining_via_context_window_scheduling.md)
- [\[ICLR 2026\] FutureFill: Fast Generation from Convolutional Sequence Models](../../ICLR2026/llm_efficiency/futurefill_fast_generation_from_convolutional_sequence_models.md)
- [\[ICLR 2026\] Fast-dLLM v2: Efficient Block-Diffusion LLM](../../ICLR2026/llm_efficiency/fast-dllm_v2_efficient_block-diffusion_llm.md)

</div>

<!-- RELATED:END -->
