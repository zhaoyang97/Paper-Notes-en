---
title: >-
  [Paper Note] From Prediction to Perfection: Introducing Refinement to Autoregressive Image Generation
description: >-
  [ICLR 2026][Image Generation][autoregressive image generation] This paper proposes TensorAR, which upgrades standard AR image generation from next-token prediction to next-tensor prediction: each step predicts an overlap…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "autoregressive image generation"
  - "next-tensor prediction"
  - "discrete diffusion noise"
  - "iterative refinement"
  - "plug-and-play"
date: 2026-05-08
content_hash: 092492259160538d
---

# From Prediction to Perfection: Introducing Refinement to Autoregressive Image Generation

**Conference**: ICLR 2026
**arXiv**: [2505.16324](https://arxiv.org/abs/2505.16324)  
**Code**: None  
**Area**: Image Generation / Autoregressive Models
**Keywords**: autoregressive image generation, next-tensor prediction, discrete diffusion noise, iterative refinement, plug-and-play

## TL;DR

This paper proposes TensorAR, which upgrades standard AR image generation from next-token prediction to next-tensor prediction: each step predicts an overlapping tensor (a group of consecutive tokens), and subsequent tensors overlap with preceding ones to enable iterative refinement. A discrete diffusion noise mechanism is introduced to address training information leakage. TensorAR serves as a plug-and-play module compatible with AR models such as LlamaGen, Open-MAGVIT2, and Janus-Pro, consistently improving generation quality on both class-to-image and text-to-image tasks.

## Background & Motivation

**Background**: Autoregressive (AR) models (LlamaGen, VAR, MAR, Open-MAGVIT2) have become one of the dominant paradigms for image generation, offering scalability, controllability, and the potential for unification with multimodal LLMs.

**Limitations of Prior Work**: Standard AR next-token prediction follows a strict left-to-right sequential generation; once a token is predicted, it cannot be corrected, causing early errors to accumulate and degrade final image quality. Existing improvements all require modifying the core paradigm: DART changes the classification objective to regression; MaskGIT/MAR requires bidirectional attention and is incompatible with KV cache; MAR requires additional VQ-VAE training—all of which hinder multimodal unification with standard GPT-style LLMs.

**Key Challenge**: AR models urgently need refinement capability to correct early prediction errors, but refinement mechanisms such as diffusion and masking are fundamentally at odds with the causal structure and classification training paradigm of AR models.

**Goal**: To endow a standard decoder-only AR model with the ability to iteratively refine already-generated tokens, without modifying the underlying Transformer architecture or the classification training objective.

**Key Insight**: If each step predicts not a single token but a group of overlapping consecutive tokens (a tensor), the overlapping region between adjacent tensors naturally provides an opportunity to correct preceding predictions—this "sliding-window refinement" achieves diffusion-like progressive improvement while preserving the causal structure.

**Core Idea**: By extending next-token to next-tensor (overlapping token group) prediction, the approach achieves sliding-window iterative refinement while maintaining causal attention and classification loss.

## Method

### Overall Architecture

TensorAR reorganizes the token sequence $[x_1, x_2, ..., x_T]$ into a sequence of overlapping tensors, where each tensor $\mathbf{x}_{i,k} = [x_i, x_{i+1}, ..., x_{i+k-1}]$ contains $k$ consecutive tokens. At inference step $t$, the model predicts a new tensor $\mathbf{x}_{t,k}$ conditioned on all preceding tensors; since adjacent tensors share $k-1$ overlapping tokens, subsequent predictions naturally refine prior outputs—the first token undergoes $k$ rounds of refinement (finest), while the last token is predicted only once. A lightweight Input Encoder (using a Query Transformer to compress $k$ token embeddings into a single hidden state) and an Output Decoder (reconstructing $k$ tokens from the hidden state) are added on top of the base AR model, both using residual designs to leverage pretrained weights.

### Key Design 1: Sliding Refinement via Overlapping Tensors

- **Function**: Enables standard AR models to iteratively improve already-generated tokens through overlapping regions, without modifying causal attention.
- **Mechanism**: The first token $x_i$ in tensor $\mathbf{x}_{i,k}$ undergoes $k$ rounds of refinement (finest), while the last token $x_{i+k-1}$ is generated only once—forming a coarse-to-fine progressive generation. When $k=1$, the model degenerates to standard AR; when $k=T$, it is equivalent to discrete diffusion (but in left-to-right order); intermediate values achieve a continuous efficiency–quality trade-off.
- **Design Motivation**: Analogous to the global iterative refinement of diffusion models, TensorAR achieves local sliding refinement—the same coarse-to-fine idea but naturally compatible with the causal structure of AR models.

### Key Design 2: Discrete Tensor Noise Mechanism

- **Function**: Resolves the information leakage caused by overlapping tokens during training—in naive training, the model would directly copy overlapping tokens rather than learning meaningful causal dependencies.
- **Mechanism**: Based on discrete diffusion theory, categorical noise is injected into the overlapping tokens of the input tensor: $q(x^*_{t+j}|x_{t+j}, j) = \text{Cat}(x^*_{t+j} | (1-\beta(j))x_{t+j} + \beta(j)/V)$, where the noise intensity $\beta(j)$ increases monotonically from 0 to 1 within the tensor. Four schedule functions are provided (linear/sine/square root/exponential); experiments show robustness to the choice of schedule.
- **Design Motivation**: The noise forces the model to learn denoising reconstruction from corrupted tokens rather than simple copying—acting as a denoiser during training and a refiner during inference.

### Key Design 3: Residual Lightweight Encoder–Decoder Modules

- **Function**: Adapts the model for tensor-level input/output while preserving pretrained information.
- **Mechanism**: The Input Encoder uses a Query Transformer to compress $k$ token embeddings into one hidden state; the Output Decoder reconstructs $k$ tokens from one hidden state. Both wrap the original embedding/linear layers via residual connections.
- **Design Motivation**: The residual design ensures that the information flow of the pretrained model is not disrupted. Additional parameters account for only 1.5%–4.6% of the total, decreasing proportionally as model scale increases.

### Loss & Training

The training objective combines AR cross-entropy and discrete diffusion denoising: $\mathcal{L}(\theta) = \sum_{i=1}^{T}\sum_{j=1}^{k} \mathbb{E}[w_j \log(p_\theta(x_{i+j}|\mathbf{x}_{<i,k}; c))]$, with loss ignored at padding token positions. Default settings: window size $k=4$, single-layer Query Transformer, exponential noise schedule.

## Key Experimental Results

### Main Results: ImageNet 256×256 / 384×384 Class-Conditional Generation

| Model | Params | FID↓ | IS↑ | Precision↑ | Recall↑ |
|-------|--------|------|-----|-----------|---------|
| LlamaGen-B (256) | 111M | 5.46 | 193.6 | 0.83 | 0.45 |
| **+TensorAR** | **116M (+4.6%)** | **4.71** | **225.8** | **0.85** | **0.45** |
| LlamaGen-L (256) | 343M | 3.80 | 248.3 | 0.83 | 0.52 |
| **+TensorAR** | **352M (+2.7%)** | **2.78** | **254.8** | **0.82** | **0.56** |
| LlamaGen-XL (384) | 775M | 2.62 | 244.1 | 0.80 | 0.57 |
| **+TensorAR** | **789M (+1.9%)** | **2.29** | **260.4** | **0.81** | **0.59** |
| LlamaGen-XXL (384) | 1411M | 2.34 | 253.9 | 0.81 | 0.60 |
| **+TensorAR** | **1432M (+1.5%)** | **2.03** | **267.7** | **0.82** | **0.61** |
| Open-MAGVIT2-B (256) | 343M | 3.08 | 258.3 | 0.85 | 0.51 |
| **+TensorAR** | **352M (+2.7%)** | **2.91** | **260.2** | **0.86** | **0.50** |
| Open-MAGVIT2-L (256) | 804M | 2.51 | 271.7 | 0.84 | 0.54 |
| **+TensorAR** | **820M (+2.0%)** | **2.35** | **273.4** | **0.84** | **0.53** |

For reference: MAGVIT-v2 FID=1.78, MaskBit FID=1.52, VAR-2.0B FID=1.73 (all masked AR or specialized architectures). TensorAR-XXL achieves FID=2.03, the best among causal AR models and approaching the level of masked AR.

### Ablation Study: Noise Schedule and Window Size (LlamaGen-B)

| Configuration | FID↓ | IS↑ | Precision↑ | Recall↑ |
|---------------|------|-----|-----------|---------|
| Baseline (no refinement) | 5.46 | 193.6 | 0.83 | 0.45 |
| **Noise Schedule** | | | | |
| Linear | 4.79 | 218.8 | 0.85 | 0.44 |
| Sine | 4.75 | 221.3 | 0.84 | 0.45 |
| Square root | 4.84 | 214.9 | 0.83 | 0.43 |
| **Exponential (default)** | **4.71** | **225.8** | **0.85** | **0.45** |
| **Window Size $k$** | | | | |
| $k=2$ | 4.78 | 221.3 | 0.84 | 0.45 |
| $k=4$ (default) | 4.71 | 225.8 | 0.85 | 0.45 |
| $k=8$ | 4.68 | 226.7 | 0.85 | 0.46 |
| **Query Transformer Depth** | | | | |
| $d=1$ (default) | 4.71 | — | 0.85 | 0.45 |
| $d=2$ | 4.79 | — | 0.85 | 0.46 |
| $d=4$ | 4.90 | — | 0.82 | 0.43 |

### Text-to-Image: GenEval Instruction-Following Evaluation

| Model | Single Obj. | Two Obj. | Counting | Colors | Position | Color Attri. | Overall↑ |
|-------|------------|----------|----------|--------|----------|-------------|---------|
| LlamaGen | 0.71 | 0.34 | 0.21 | 0.58 | 0.07 | 0.04 | 0.32 |
| **+TensorAR** | **0.99** | **0.70** | **0.57** | **0.89** | **0.28** | **0.19** | **0.61** |
| Janus-Pro-7B | 0.99 | 0.89 | 0.59 | 0.90 | 0.79 | 0.66 | 0.80 |
| **+TensorAR** | **0.99** | **0.93** | **0.53** | **0.92** | **0.85** | **0.79** | **0.83** |
| DALL-E 3 | 0.96 | 0.87 | 0.47 | 0.83 | 0.43 | 0.45 | 0.67 |
| SD3-Medium | 0.99 | 0.94 | 0.72 | 0.89 | 0.33 | 0.60 | 0.74 |

### Key Findings

- **Consistent gains across models and scales**: TensorAR consistently reduces FID on both LlamaGen (111M–1.4B) and Open-MAGVIT2. The largest improvement is on LlamaGen-B (5.46→4.71, −13.7%), with a 0.31-point reduction also observed on the 1.4B model (2.34→2.03).
- **Minimal parameter overhead**: Added parameters ≤4.6%, decreasing proportionally as model scale grows (only +1.5% for XXL).
- **Large gains on text-to-image**: GenEval Overall improves from 0.32→0.61 (+91%) on LlamaGen, and from 0.80→0.83 on Janus-Pro.
- **FID decreases monotonically with $k$**: Even $k=2$ substantially outperforms the baseline (5.46→4.78); $k=8$ achieves the lowest FID (4.68), while $k=4$ provides a favorable efficiency–quality balance.
- **All four noise schedules substantially outperform the no-noise baseline**: The exponential schedule performs best (4.71); the model is robust to schedule selection.
- **Query Transformer depth $d=1$ is optimal**: Increasing depth raises FID and latency ($d=4$ yields FID=4.90).
- **Not attributable to simple fine-tuning**: Direct fine-tuning of the base model for the same number of steps yields no FID improvement, confirming that the gains stem from the refinement mechanism.

## Highlights & Insights

- **"Refine, don't regenerate"**: AR models acquire for the first time the ability to correct preceding predictions, analogous to a human "draft → revision" creative process—improving locally rather than regenerating entirely.
- **Discrete diffusion as a training tool, not a generation tool**: The discrete diffusion noise is cleverly applied to resolve training information leakage rather than for image generation itself—transplanting the "denoising" idea from diffusion into the "refinement" requirement of AR.
- **Engineering value of plug-and-play design**: The Transformer architecture remains unchanged (still decoder-only causal attention), the training objective remains unchanged (still classification cross-entropy), and the VQ tokenizer remains unchanged—any GPT-style AR model can benefit simply by adding the lightweight module.
- **Unified perspective**: $k=1$ is standard AR, $k=T$ is discrete diffusion, and TensorAR is the continuous spectrum between the two—providing a theoretical bridge between AR and diffusion.
- **91% GenEval improvement on LlamaGen**: Notably, refinement not only improves image quality but also substantially enhances instruction-following capability.

## Limitations & Future Work

- Increasing window size $k$ linearly increases inference steps and latency; the choice of $k$ requires balancing quality and speed.
- Validation is currently limited to the discrete space of VQ tokenizers; compatibility with continuous-token methods (e.g., MAR's diffusion head) remains unexplored.
- On DPG-Bench, the "Other" sub-metric for Janus-Pro+TensorAR drops from 89.48 to 84.52, suggesting that refinement may occasionally introduce side effects.
- Integration with AR inference acceleration and distillation methods (e.g., speculative decoding) has not been explored—the paper itself identifies this as a promising direction.
- Refinement primarily benefits early tokens; marginal gains may diminish for the latter half of long sequences.

## Related Work & Insights

- **vs. DART**: DART unifies AR and diffusion but changes the training objective to regression and requires a non-Markovian framework; TensorAR preserves the classification objective and standard Markov process.
- **vs. MaskGIT/MAR**: Masked AR requires bidirectional attention and is incompatible with KV cache and standard LLMs; TensorAR preserves causal attention and KV cache.
- **vs. VAR**: VAR uses next-scale prediction (multi-resolution coarse-to-fine); TensorAR uses next-tensor prediction (same-resolution sliding refinement); the two approaches are complementary.
- **Broader inspiration**: The refinement idea may generalize to text AR models—if LLMs could also slidingly correct preceding tokens during generation, long-text coherence might improve.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The combination of next-tensor prediction and discrete noise is elegant and effective, providing a unified AR–diffusion perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two task types, two base models, six scales, and thorough ablations (noise/window/depth); only lacking large-scale text-to-image benchmarks.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The core idea is explained with exceptional clarity; the continuous spectrum from $k=1$ to $k=T$ offers deep insight.
- **Value**: ⭐⭐⭐⭐⭐ Substantively advances the AR image generation paradigm; the plug-and-play design has high practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Condition Errors Refinement in Autoregressive Image Generation with Diffusion Loss](condition_errors_refinement_in_autoregressive_image_generation_with_diffusion_lo.md)
- [\[ICLR 2026\] Autoregressive Image Generation with Randomized Parallel Decoding](autoregressive_image_generation_with_randomized_parallel_decoding.md)
- [\[ICLR 2026\] Locality-aware Parallel Decoding for Efficient Autoregressive Image Generation](locality-aware_parallel_decoding_for_efficient_autoregressive_image_generation.md)
- [\[ICLR 2026\] Visual Autoregressive Modeling for Instruction-Guided Image Editing](visual_autoregressive_modeling_for_instruction-guided_image_editing.md)
- [\[NeurIPS 2025\] Watermarking Autoregressive Image Generation](../../NeurIPS2025/image_generation/watermarking_autoregressive_image_generation.md)

</div>

<!-- RELATED:END -->
