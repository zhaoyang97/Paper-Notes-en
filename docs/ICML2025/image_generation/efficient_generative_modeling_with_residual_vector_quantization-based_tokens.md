---
title: >-
  [Paper Note] Efficient Generative Modeling with Residual Vector Quantization-Based Tokens
description: >-
  [ICML2025][Image Generation][ResGen] ResGen decouples the number of generation iterations from both sequence length and quantization depth by directly predicting cumulative RVQ embeddings instead of individual tokens, achieving an efficient generative model with high fidelity and rapid sampling.
tags:
  - "ICML2025"
  - "Image Generation"
  - "ResGen"
  - "Residual Vector Quantization"
  - "Masked Prediction"
  - "Discrete Diffusion"
  - "Efficient Generation"
date: 2026-05-08
content_hash: bf64be86c36ab52a
---

# Efficient Generative Modeling with Residual Vector Quantization-Based Tokens

**Conference**: ICML2025  
**arXiv**: [2412.10208](https://arxiv.org/abs/2412.10208)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: ResGen, Residual Vector Quantization, Masked Prediction, Discrete Diffusion, Efficient Generation

## TL;DR
ResGen decouples the number of generation iterations from both sequence length and quantization depth by directly predicting cumulative RVQ embeddings instead of individual tokens, achieving an efficient generative model with high fidelity and rapid sampling.

## Background & Motivation

### Key Challenge

**Key Challenge**: Residual Vector Quantization (RVQ) improves reconstruction fidelity by iteratively quantizing residuals, but in autoregressive generation:
- Number of sampling steps = sequence length $L$ $\times$ quantization depth $D$
- Inference cost grows linearly as $L$ and $D$ increase
- Existing non-autoregressive methods can only improve along either the $L$ or $D$ dimension, failing to decouple both simultaneously

### Limitations of Prior Work

**Limitations of Prior Work**: There is a need for a method that:
- Retains the benefits of high-fidelity reconstruction brought by RVQ
- Ensures the number of inference steps does not grow with $D$
- Provides a unified probabilistic framework to handle the RVQ hierarchy

## Method

### Overall Architecture: Coarse-to-Fine Iterative Filling
ResGen adopts a strategy of "gradual filling starting from the topmost quantization layer":

### Key Designs 1: Hierarchical Masking Strategy
- Progressively masks starting from the maximum depth $D$ (fine-grained layers)
- Mask scheduling function: $n = \lceil \gamma(r) \cdot L \cdot D \rceil$
- Multinomial sampling uniformly distributes masked tokens across $L$ spatial positions
- Progressively masks from depth $D$ down to lower depths

### Key Designs 2: Multi-Token Prediction (Core Innovation)
Instead of predicting a single token $x$ during training, the model predicts the cumulative sum of masked embeddings:

$$z_i = \sum_j e(x_{i,j}; j) \odot (1 - m_{i,j})$$

By predicting the aggregated vector, the model captures token associations across depths, thereby **decoupling the number of sampling steps from the RVQ depth $D$**.

### Key Designs 3: Probabilistic Framework
- Formalized using a discrete diffusion process and variational inference
- Enhanced sampling strategy based on model confidence
- Estimating latent embeddings using Gaussian Mixture Models

### Comparison of Sampling Steps

| Method | Sampling Steps |
|------|---------|
| Autoregressive | $O(L \times D)$ |
| Non-Autoregressive (Prior) | $O(L)$ or $O(D)$ |
| ResGen | $O(\text{fixed})$ (Independent of both $L$ and $D$) |

## Key Experimental Results

### Conditional Image Generation on ImageNet 256×256


### Main Results

| Method | Parameters | Sampling Steps | FID | Relative Performance |
|------|--------|---------|-----|---------|
| AR Baseline | 1.5B | 256 | 4.2 | Baseline |
| ResGen (D=4) | 1.5B | 8 | 3.8 | FID↓ + 3.2× speedup |
| ResGen (D=8) | 1.5B | 8 | 3.2 | FID↓↓ + 5.1× speedup |

### Zero-Shot Text-to-Speech Synthesis


### Ablation Study

| Method | MOS | Sampling Steps | Real-Time Factor |
|------|--------|---------|---------|
| AR Baseline | 3.8 | 512 | 1.2× |
| ResGen | 4.1 | 16 | 4.5× |
| ResGen + Depth Expansion | 4.2 | 16 | 4.3× |

### Key Findings
1. Achieves superior FID with 8$\times$ fewer sampling steps.
2. Fidelity significantly improves as RVQ depth increases ($D$: 4 $\rightarrow$ 8).
3. Demonstrates general effectiveness across modalities (image + audio).
4. MOS increases by 7.9% along with a 3.75$\times$ inference speedup.

## Highlights & Insights

1. "Predicting cumulative embeddings instead of individual tokens" — a simple yet groundbreaking design choice that successfully decouples depth from inference cost.
2. Hierarchical masking (top-level first) aligns with information theory intuition: prioritizing coarse-grained information.
3. A complete probabilistic framework combining discrete diffusion and variational inference, ensuring theoretical rigor.
4. Validation on two modalities (image + audio) demonstrates the generalizability of the proposed method.

## Limitations & Future Work

1. Open-source code availability is unconfirmed, which may pose reproduction barriers.
2. The optimal selection criteria for different RVQ depths ($D=4$ vs. $D=8$) remain unclear.
3. Detailed analysis comparing the computational overhead of predicting aggregated embeddings versus single-token prediction is insufficient.
4. Verification is limited to image and audio modalities, leaving text and video scenarios unexplored.
5. The mask scheduling function $\gamma(\cdot)$ is highly sensitive to the specific task.

## Related Work & Insights

- **VQ-VAE/VQ-GAN**: ResGen is built upon their quantization frameworks but employs RVQ to improve the hierarchical structure.
- **Masked Token Modeling**: It adopts the masked prediction framework but introduces a novel prediction target.
- Insights:
  1. Hierarchical masking can be generalized to other hierarchical structures (e.g., tree or graph-structured tokens).
  2. The design paradigm of multi-target prediction (aggregation instead of prediction of individual tokens) is worth exploring across other tasks.
  3. The RVQ encoder and the generator can be jointly optimized.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (4.5/5) — Cumulative embedding prediction to decouple depth from inference is the core innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ (4.0/5) — Validated on two modalities, but ablation studies could be more comprehensive.
- Writing Quality: ⭐⭐⭐⭐☆ (4.0/5)
- Value: ⭐⭐⭐⭐⭐ (4.5/5) — Provides substantial contributions to efficient generative models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization](modulated_diffusion_accelerating_generative_modeling_with_modulated_quantization.md)
- [\[ICML 2025\] Generative Audio Language Modeling with Continuous-Valued Tokens and Masked Next-Token Prediction](generative_audio_language_modeling_with_continuous-valued_tokens_and_masked_next.md)
- [\[ICML 2025\] Action-Minimization Meets Generative Modeling: Efficient Transition Path Sampling with the Onsager-Machlup Functional](action-minimization_meets_generative_modeling_efficient_transition_path_sampling.md)
- [\[ICML 2025\] Compositional Scene Understanding through Inverse Generative Modeling](compositional_scene_understanding_through_inverse_generative_modeling.md)
- [\[ICML 2025\] DCTdiff: Intriguing Properties of Image Generative Modeling in the DCT Space](dctdiff_intriguing_properties_of_image_generative_modeling_in_the_dct_space.md)

</div>

<!-- RELATED:END -->
