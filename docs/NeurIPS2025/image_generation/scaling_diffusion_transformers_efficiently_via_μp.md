---
title: >-
  [Paper Note] Scaling Diffusion Transformers Efficiently via μP
description: >-
  [NeurIPS 2025][Image Generation][Diffusion Transformer] This paper extends Maximal Update Parametrization (μP) from standard Transformers to diffusion Transformers (DiT, PixArt-α, MMDiT, etc.)…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Diffusion Transformer"
  - "μP"
  - "hyperparameter transfer"
  - "model scaling"
  - "efficient training"
date: 2026-05-08
content_hash: fcbd841f25555ae1
---

# Scaling Diffusion Transformers Efficiently via μP

**Conference**: NeurIPS 2025
**arXiv**: [2505.15270](https://arxiv.org/abs/2505.15270)
**Code**: [Available](https://github.com/ML-GSAI/Scaling-Diffusion-Transformers-muP)
**Area**: Image Generation
**Keywords**: Diffusion Transformer, μP, hyperparameter transfer, model scaling, efficient training

## TL;DR

This paper extends Maximal Update Parametrization (μP) from standard Transformers to diffusion Transformers (DiT, PixArt-α, MMDiT, etc.), demonstrating that optimal hyperparameters found on small proxy models transfer stably to large models, significantly reducing the hyperparameter tuning cost for large-scale diffusion models.

## Background & Motivation

Diffusion Transformers have become the foundational architecture for visual generation models, with widespread applications in image and video generation. However, as model scale grows to billions of parameters, hyperparameter (HP) tuning becomes prohibitively expensive, often preventing models from reaching their full potential.

μP has previously been proposed for standard Transformers (e.g., LLMs), enabling optimal hyperparameters found on small models to transfer directly to large models, substantially reducing tuning costs. However, diffusion Transformers differ fundamentally from standard Transformers in two respects:

**Architectural differences**: Diffusion Transformers contain additional components (adaLN, cross-attention, etc.) for integrating text and timestep conditioning.

**Training paradigm differences**: The iterative denoising generation framework differs fundamentally from the autoregressive paradigm.

Whether existing μP theory can be directly applied to diffusion Transformers therefore remains an open question, which this paper systematically addresses.

## Method

### Overall Architecture

The methodology proceeds in three steps: (1) theoretically proving that the μP formulation for diffusion Transformers is consistent with that of standard Transformers; (2) validating hyperparameter transferability of DiT-μP; and (3) verifying the efficiency of μTransfer on large-scale text-to-image tasks.

### Key Designs

1. **Theoretical extension of μP (Theorem 3.1)**: Using the Tensor Programs framework, the paper rigorously proves that the forward passes of mainstream diffusion Transformers (U-ViT, DiT, PixArt-α, MMDiT) can be expressed within the Ne⊗or⊤ Program framework. The key contribution lies in showing that diffusion Transformer-specific modules (e.g., adaLN blocks) can also be rewritten using the three operators of this framework. This implies that the abc-parametrization rules of μP apply directly:

    - Input weights: $a_W=0, b_W=0, c_W=0$
    - Hidden weights: $a_W=0, b_W=1/2, c_W=1$ (learning rate requires $\eta \cdot n_{base}/n$ scaling)
    - Output weights: $a_W=1, b_W=0, c_W=0$ (zero initialization)

2. **Width scaling strategy**: Head dimension is fixed while width is scaled by increasing the number of heads. Theoretically (Bordelon et al., 2024), increasing head dimension causes multi-head attention to degenerate toward single-head attention, losing diversity in attention patterns. This approach is also consistent with prevailing practice in LLM scaling.

3. **μTransfer algorithm**: A proxy model (small width, small batch, short training) is used to search for optimal base hyperparameters, which are then transferred directly to the target large model. Both proxy and target models are parametrized using the same $n_{base}$ under μP, thereby sharing the same optimal base hyperparameters.

### Loss & Training

- DiT experiments: fixed head dimension 72, base width 288 (4 heads), AdamW optimizer, no learning rate schedule or weight decay.
- PixArt-α: proxy model 0.04B (4 heads), target model 0.61B (16 heads), proxy trained for 5 epochs.
- MMDiT-18B: proxy model 0.18B (width 512), 80 random searches over 4 base hyperparameters (learning rate, gradient clipping, REPA loss weight, warm-up steps).

## Key Experimental Results

### Main Results

| Model | Method | Key Metric | Tuning Cost |
|-------|--------|------------|-------------|
| DiT-XL-2 | Original | FID convergence @7M steps | Baseline |
| DiT-XL-2-μP | μTransfer | FID convergence @2.4M steps | **2.9× speedup** |
| PixArt-α (30ep) | Original | GenEval 0.15, FID(MJHQ) 42.71 | Baseline |
| PixArt-α-μP (30ep) | μTransfer | GenEval 0.26, FID(MJHQ) 29.96 | **5.5% FLOPs** |
| MMDiT-18B | Manual tuning | GenEval 0.8154, Alignment 0.703 | 5× pretraining cost |
| MMDiT-μP-18B | μTransfer | GenEval 0.8218, Alignment 0.715 | **3% of manual tuning cost** |

### Ablation Study

| Configuration | Key Findings |
|---------------|--------------|
| LR transfer across width | Optimal base LR $2^{-10}$ transfers stably across widths 144–1152 |
| LR transfer across batch size | Optimal LR shared across batch sizes 256–1024 |
| LR transfer across training steps | Optimal LR shared across 150K–400K steps |
| MMDiT HP search | LR has the largest impact; optimal gradient clipping is 1 (vs. conventional 0.1); warm-up effect is negligible |

### Key Findings

- PixArt-α-μP shows less overfitting during extended training, whereas the original PixArt-α degrades after 20 epochs, suggesting μP may enhance generalization.
- μP tends to favor larger learning rates (near the maximum stable value), consistent with the theoretical insight that large learning rates introduce beneficial gradient noise.
- In single-epoch MMDiT training, the optimal learning rate no longer approaches the maximum stable value, exhibiting different behavior from multi-epoch training.

## Highlights & Insights

- **Theoretical rigor**: Rather than empirically applying μP to diffusion Transformers, the paper provides a rigorous proof grounded in the Tensor Programs framework.
- **Substantial practical value**: Hyperparameter tuning for MMDiT-18B costs only 3% of manual tuning, representing enormous savings in compute.
- **Unexpected finding**: The optimal base learning rate for PixArt-α and DiT is identical ($2^{-10}$), hinting that optimal base hyperparameters may transfer across datasets and architectures.

## Limitations & Future Work

- The optimal scale of proxy tasks (proxy model size and required training data volume) remains undetermined.
- Only the fixed-head-dimension, scaling-by-head-count strategy is validated; scaling head dimension is not explored.
- The approach is extensible to more efficient architectures such as linear Transformers and MoE models.
- The framework can be applied to more advanced training strategies, including the Muon optimizer and warmup-stable-decay learning rate schedules.

## Related Work & Insights

This paper establishes the theoretical and practical foundation for μP in visual generation. For large-scale diffusion model training (e.g., Sora-scale video models), μP provides a principled hyperparameter search strategy: search on small proxy models, then transfer zero-shot to large models, avoiding the prohibitively costly trial-and-error process at scale.

## Rating

- Novelty: ⭐⭐⭐⭐ (μP itself is not new; the contribution lies in the theoretical extension and large-scale validation)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (progressively validated from DiT to MMDiT-18B)
- Writing Quality: ⭐⭐⭐⭐⭐ (well-organized, with theory and experiments tightly integrated)
- Value: ⭐⭐⭐⭐⭐ (directly applicable to industrial-scale large model training)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unleashing Diffusion Transformers for Visual Correspondence by Modulating Massive Activations](unleashing_diffusion_transformers_for_visual_correspondence_by_modulating_massiv.md)
- [\[NeurIPS 2025\] Remasking Discrete Diffusion Models with Inference-Time Scaling](remasking_discrete_diffusion_models_with_inference-time_scaling.md)
- [\[NeurIPS 2025\] Scaling Can Lead to Compositional Generalization](scaling_can_lead_to_compositional_generalization.md)
- [\[NeurIPS 2025\] OmniSync: Towards Universal Lip Synchronization via Diffusion Transformers](omnisync_towards_universal_lip_synchronization_via_diffusion.md)
- [\[ICLR 2026\] Routing Matters in MoE: Scaling Diffusion Transformers with Explicit Routing Guidance](../../ICLR2026/image_generation/routing_matters_in_moe_scaling_diffusion_transformers_with_explicit_routing_guid.md)

</div>

<!-- RELATED:END -->
