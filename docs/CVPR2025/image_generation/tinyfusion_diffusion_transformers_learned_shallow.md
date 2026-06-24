---
title: >-
  [Paper Note] TinyFusion: Diffusion Transformers Learned Shallow
description: >-
  [CVPR 2025][Image Generation][Diffusion Transformers] This paper proposes TinyFusion, a learnable depth pruning method. By utilizing Gumbel-Softmax differentiable sampling for layer masking and co-optimizing weight updates to simulate fine-tuning, it explicitly optimizes the recoverability of the pruned model (rather than minimizing post-pruning loss). On DiT-XL, it constructs shallow Diffusion Transformers at less than 7% of the pre-training cost…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Diffusion Transformers"
  - "Depth Pruning"
  - "Learnable Compression"
  - "Recoverability"
  - "Gumbel-Softmax"
date: 2026-05-08
content_hash: 2d15de1f0b2121fc
---

# TinyFusion: Diffusion Transformers Learned Shallow

**Conference**: CVPR 2025  
**arXiv**: [2412.01199](https://arxiv.org/abs/2412.01199)  
**Code**: [GitHub](https://github.com/VainF/TinyFusion)  
**Area**: Image Generation/Model Compression  
**Keywords**: Diffusion Transformers, Depth Pruning, Learnable Compression, Recoverability, Gumbel-Softmax

## TL;DR

This paper proposes TinyFusion, a learnable depth pruning method. By utilizing Gumbel-Softmax differentiable sampling for layer masking and co-optimizing weight updates to simulate fine-tuning, it explicitly optimizes the recoverability of the pruned model (rather than minimizing post-pruning loss). On DiT-XL, it constructs shallow Diffusion Transformers at less than 7% of the pre-training cost, achieving a 2× speedup with an FID of only 2.86.

## Background & Motivation

- **Inference Burden of Diffusion Transformers**: Models such as DiT, MAR, and SiT have massive parameter sizes, leading to high deployment costs.
- **Advantages of Depth Pruning**: Compared to width pruning (where 50% width reduction only yields a 1.6× speedup), depth pruning achieves a speedup linearly proportional to the compression rate (50% depth reduction = 2× speedup), and remains effective on both parallel and non-parallel devices.
- **Limitations of the Loss Minimization Principle**: Traditional pruning (including feature similarity and sensitivity analysis) aims to minimize post-pruning loss. However, experiments show that low post-pruning loss does not guarantee high post-fine-tuning performance—this principle does not hold for diffusion Transformers.
- **Recoverability is Key**: An analysis of 100,000 randomly sampled models reveals that models with lower post-pruning loss do not necessarily perform better after fine-tuning. The objective should be optimizing "how well the model can recover after fine-tuning."
- **Challenges of Differentiable Search**: Layer selection is a discrete, non-differentiable operation. Evaluating recoverability requires actual fine-tuning, and the search space is massive (e.g., $\binom{28}{14} = 40$ million combinations).

## Method

### Overall Architecture

TinyFusion divides an $L$-layer Transformer into $K$ local blocks, where each block retains $N$ layers using an N:M scheme. The layer mask is modeled as a categorical distribution via Gumbel-Softmax differentiable sampling: $p(\mathfrak{m}) = p(\mathfrak{m}_1) \cdot p(\mathfrak{m}_2) \cdots p(\mathfrak{m}_K)$. Concurrently, a weight update $\Delta\Phi$ (implemented via LoRA) is learned to simulate the fine-tuning effect. The objective is formulated as: $\min_{\mathfrak{m}} \min_{\Delta\Phi} \mathbb{E}_x[\mathcal{L}(x, \Phi + \Delta\Phi, \mathfrak{m})]$.

### Key Designs

**Design 1: Recoverability-Oriented Optimization Objective**
- **Function**: Find the pruning scheme that recovers to the best performance after fine-tuning.
- **Mechanism**: Standing in contrast to the traditional $\min_{\mathfrak{m}} \mathbb{E}[\mathcal{L}(x, \Phi, \mathfrak{m})]$ (minimizing post-pruning loss), TinyFusion introduces $\Delta\Phi$ to simulate fine-tuning: $\min_{\mathfrak{m}} \min_{\Delta\Phi} \mathbb{E}[\mathcal{L}(x, \Phi + \Delta\Phi, \mathfrak{m})]$. LoRA is used as a lightweight proxy for $\Delta\Phi$ and is optimized jointly with mask sampling.
- **Design Motivation**: Empirical findings show no significant correlation between post-pruning loss and post-fine-tuning performance. Schemes with low post-pruning loss selected by efforts such as ShortGPT yield poor FID scores after fine-tuning (22.28 vs. 5.73 for TinyFusion).

**Design 2: Gumbel-Softmax Differentiable Layer Mask Sampling**
- **Function**: Enable discrete layer selection to be optimized via gradient descent.
- **Mechanism**: The model is divided into $K$ blocks, and all $\binom{M}{N}$ possible N:M masks are enumerated in each block. Differentiable sampling is achieved using Gumbel-Softmax + STE: $y = \text{one-hot}(\exp((g_i + \log p_i)/\tau) / \sum_j \exp((g_j + \log p_j)/\tau))$ and $\mathfrak{m} = y^\top \hat{\mathfrak{m}}$. The temperature $\tau$ is annealed from high to low, transitioning from exploration to convergence.
- **Design Motivation**: Direct enumeration yields an excessively large search space. Probabilistic modeling transforms the search into distribution optimization, where sampling patterns with positive feedback gain higher probabilities, progressively converging to the optimal solution.

**Design 3: MaskedKD — Masked Knowledge Distillation**
- **Function**: Enhance performance recovery of the pruned model during fine-tuning.
- **Mechanism**: The original unpruned model acts as the teacher to distill knowledge into the shallow model. A key improvement is masking massive/outlier activations in the hidden layers to prevent them from negatively affecting fine-tuning stability and distillation effectiveness.
- **Design Motivation**: Outlier activations present in diffusion Transformers tend to be amplified during distillation; masking them improves the FID from 5.73 to 3.73.

### Loss & Training

The standard diffusion loss is formulated as $\mathcal{L} = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$. During the search phase, mask distribution parameters and LoRA weights are optimized simultaneously. During the fine-tuning phase, standard retraining or MaskedKD can be optionally employed.

## Key Experimental Results

### DiT-XL/2 Depth Pruning (28→14 Layers, 50% Compression)

| Method | FID ↓ | Sampling Speed (it/s) ↑ | Fine-tuning Cost |
|------|------|-----------------|---------|
| DiT-XL/2 Original (28 layers) | 2.27 | 6.91 | 7000K iters |
| ShortGPT (28→14) | 22.28 | 13.54 | 100K iters |
| Flux-Lite (28→14) | 25.92 | 13.54 | 100K iters |
| Sensitivity (28→14) | 21.15 | 13.54 | 100K iters |
| **TinyFusion (28→14)** | **5.73** | **13.54** | **100K iters** |
| TinyFusion + MaskedKD | **3.73** | 13.54 | 100K iters |
| TinyFusion + MaskedKD (500K) | **2.86** | 13.54 | 500K iters |

### Generalization Across Different Models

| Model | Compression Method | Results |
|------|---------|------|
| DiT-XL/2 | 28→14 layers | FID 2.86 @ 2× Speedup |
| MAR | Depth Pruning | Effective |
| SiT | Depth Pruning | Effective |

### Key Findings

1. After 100K iterations of fine-tuning, TinyFusion achieves an FID of 5.73, which is significantly better than ShortGPT's 22.28 under the exact same pruning ratio and fine-tuning budget.
2. After 500K iterations of fine-tuning (only 7% of the pre-training cost), the FID is reduced to 2.86, which is only 0.59 higher than the original 28-layer model.
3. MaskedKD further reduces the FID from 5.73 to 3.73, demonstrating that handling outlier activations is critical for distillation.
4. 50% width pruning yields only a 1.6× speedup compared to a 2× speedup from 50% depth pruning, verifying that depth pruning is more effective for real-world hardware acceleration.
5. The method generalizes well across three distinct architectures: DiT, MAR, and SiT.

## Highlights & Insights

- **Challenging the Traditional "Post-Pruning Loss Minimization" Paradigm**: Backed by a convincing and large-scale empirical analysis, this work introduces a novel optimization target for the model pruning community.
- **LoRA as a Fine-Tuning Proxy**: Simulates actual fine-tuning effects via low-rank updates, enabling highly efficient evaluation of recoverability during the search phase.
- **Local N:M Scheme**: Decomposes the global search space into independent local searches, preserving valuable local structural patterns.
- The optimal pruning scheme is discovered in just 1 epoch, making the search cost extremely low.

## Limitations & Future Work

- The N:M scheme assumes that each block retains the same proportion of layers, which may not be globally optimal, as certain blocks might benefit more from retaining more layers.
- Currently, the method is evaluated only on ImageNet 256×256 conditional generation; its applicability to higher resolutions and text-to-image generation remains to be explored.
- The masking threshold strategy in MaskedKD is still heuristic (based on standard deviation multiples of activation values), which might require model-specific tuning.
- Although efficient, using LoRA as a proxy for fine-tuning may not fully reflect the exact recovery behavior of full-parameter fine-tuning.

## Related Work & Insights

- **vs. ShortGPT / Flux-Lite**: Although feature-similarity-based heuristic methods can find pruning configurations with low calibration loss, their post-fine-tuning FIDs are significantly worse than TinyFusion's (22+ vs. 5.73), confirming that recoverability $\neq$ low calibration loss.
- **vs. Diff-Pruning (Width Pruning)**: 50% width pruning achieves only a 1.6× speedup with an FID of 3.85, whereas TinyFusion's 50% depth pruning achieves a 2× speedup with an FID of 2.86, proving that depth pruning is more effective on real-world hardware.
- The recoverability-oriented pruning paradigm can be generalized to the depth compression of other large-scale models, such as LLMs.
- The Gumbel-Softmax + co-weight optimization paradigm is applicable to generic discrete structure search problems.

## Rating

⭐⭐⭐⭐ — Profound insights (recoverability vs. post-pruning loss), elegant methodology design (differentiable sampling + LoRA proxy), and strong experimental results (2× speedup with 2.86 FID). It makes a significant contribution to the field of diffusion model compression.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Hiding Images in Diffusion Models by Editing Learned Score Functions](hiding_images_in_diffusion_models_by_editing_learned_score_functions.md)
- [\[ICLR 2026\] DiffSparse: Accelerating Diffusion Transformers with Learned Token Sparsity](../../ICLR2026/image_generation/diffsparse_accelerating_diffusion_transformers_with_learned_token_sparsity.md)
- [\[CVPR 2025\] Dual Prompting Image Restoration with Diffusion Transformers (DPIR)](dual_prompting_image_restoration_with_diffusion_transformers.md)
- [\[CVPR 2025\] Autoregressive Distillation of Diffusion Transformers](autoregressive_distillation_of_diffusion_transformers.md)
- [\[CVPR 2025\] One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers](one_model_many_budgets_elastic_latent_interfaces_for_diffusion_transformers.md)

</div>

<!-- RELATED:END -->
