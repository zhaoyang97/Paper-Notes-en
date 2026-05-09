---
title: >-
  [Paper Note] One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers
description: >-
  [CVPR 2026][Image Generation][Diffusion Transformer] This paper proposes ELIT (Elastic Latent Interface Transformer), which decouples computation from input resolution by inserting variable-length latent token interfaces and lightweight Read/Write cross-attention layers into DiTs. A single model supports multiple inference budgets, achieving 35.3% and 39.6% improvements in FID and FDD respectively on ImageNet-1K 512px.
tags:
  - CVPR 2026
  - Image Generation
  - Diffusion Transformer
  - elastic inference
  - latent interface
  - variable compute budget
  - adaptive computation allocation
date: 2026-05-08
content_hash: 824c7c02e99360b5
---

# One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers

**Conference**: CVPR 2026
**arXiv**: [2603.12245](https://arxiv.org/abs/2603.12245)
**Code**: [https://snap-research.github.io/elit](https://snap-research.github.io/elit)
**Area**: Image Generation / Model Compression / Diffusion Models
**Keywords**: Diffusion Transformer, elastic inference, latent interface, variable compute budget, adaptive computation allocation

## TL;DR
This paper proposes ELIT (Elastic Latent Interface Transformer), which decouples computation from input resolution by inserting variable-length latent token interfaces and lightweight Read/Write cross-attention layers into DiTs. A single model supports multiple inference budgets, achieving 35.3% and 39.6% improvements in FID and FDD respectively on ImageNet-1K 512px.

## Background & Motivation
DiT-based diffusion models have achieved state-of-the-art quality in image/video generation, but suffer from two fundamental computational efficiency problems: (1) the per-step computation is locked to a fixed function of input resolution, precluding flexible latency–quality trade-offs; (2) computation is uniformly distributed across all spatial tokens regardless of whether certain regions are simple or unimportant. The authors confirm the second point through an elegant experiment: training DiT on images padded with zeros to expand the token count doubles FLOPs yet yields no improvement in generation quality—attention maps reveal that zero-valued tokens predominantly attend to each other, demonstrating that DiT cannot reallocate computation from simple regions to difficult ones.

## Core Problem
How can DiT-class models, without altering the training objective or the main architecture, (a) concentrate computation on important/difficult regions rather than distributing it uniformly, and (b) support a wide range of inference compute budgets with a single model? Existing approaches either require architectural changes too large to be practical (e.g., RIN/FIT deviate substantially from the DiT design and are difficult to transfer), accelerate only training while leaving inference unchanged (e.g., masking methods such as MaskDiT), or are training-free but bounded by the quality ceiling of the baseline (e.g., token merging methods).

## Method

### Overall Architecture
ELIT inserts a "latent interface"—a set of learnable, variable-length latent tokens—into the transformer block stack of a DiT. The overall pipeline is divided into three stages:
- **Spatial Head** ($B_{in}$ blocks): performs preliminary processing on patchified inputs, avoiding direct reading from raw noisy patches.
- **Latent Core** ($B_{core}$ blocks): first pulls spatial token information into latent tokens via a Read layer, executes standard transformer blocks in the latent domain, then broadcasts updates back to spatial tokens via a Write layer. This stage constitutes the bulk of computation.
- **Spatial Tail** ($B_{out}$ blocks): completes spatial detail processing to produce the final velocity field prediction.

Inputs and outputs remain unchanged; the training objective is the standard Rectified Flow loss with no auxiliary losses.

### Key Designs
1. **Read/Write Cross-Attention Layers**: The Read layer performs cross-attention with latent tokens as queries and spatial tokens as keys/values, pulling information from the spatial domain into the latent domain and naturally prioritizing difficult spatial regions with higher loss. The Write layer is fully symmetric, broadcasting latent-domain updates back to spatial tokens. Both layers employ adaLN-Zero for timestep conditioning, QK normalization for training stability, and MLPs without hidden-dimension expansion to reduce overhead.

2. **Grouped Cross-Attention**: Spatial tokens are divided into $G$ non-overlapping groups (e.g., a 2D grid), and latent tokens are correspondingly partitioned into $J = K/G$ tokens per group. Cross-attention is computed only within each corresponding group, reducing complexity from $O(NK)$ to $O(NK/G)$. Latent tokens are initialized from a shared set of learnable positional encodings, making the model robust to input resolution changes—increasing resolution only changes $G$ and $N$, not the per-group latent count $J$.

3. **Multi-Budget Training with Random Tail Dropping**: During training, each iteration samples $\tilde{J} \sim \mathrm{Uniform}\{J_{\min}, \ldots, J_{\max}\}$; only the first $\tilde{J}$ latent tokens per group are retained and the rest are discarded. This causes earlier latents to be trained more frequently, forcing them to store the most important global information and forming an importance-ordered hierarchy. At inference time, selecting $\tilde{J}$ directly controls per-step computation, naturally supporting a spectrum of compute budgets.

4. **CCFG (Cheap Classifier-Free Guidance)**: A multi-budget model inherently contains a "weaker version" of itself (the low-budget variant), which can be directly used for autoguidance without additional training. CCFG further drops class conditioning in the guidance term, combining the advantages of autoguidance and CFG to reduce inference FLOPs by approximately 33% while improving quality, at no additional training cost.

### Loss & Training
- Training objective: standard Rectified Flow loss $\mathcal{L}_{RF} = \mathbb{E}\|G(X_t, t) - (X_1 - X_0)\|_2^2$, no auxiliary losses.
- Timestep sampling: logit-normal distribution.
- During multi-budget training, batch size is increased from 256 to 384 to compensate for the reduced computation in low-token iterations and match training FLOPs.
- DiT-XL/2 main experiments: 500k steps, lr=1e-4, 10k warmup, Adam, EMA β=0.9999, gradient clipping=1.0.
- Large-scale experiments (Qwen-Image 20B): RF loss + distillation loss (scaled 20× to match magnitudes); trained at 512px for 60k steps then at 1024px for 60k steps.

## Key Experimental Results

| Dataset | Metric | Ours (ELIT-DiT-XL MB) | Prev. SOTA (DiT-XL) | Gain |
|--------|------|------|----------|------|
| ImageNet-1K 256px | FID↓ (+G) | 3.8 | 5.7 | 33% |
| ImageNet-1K 256px | FDD↓ (+G) | 124.5 | 232.9 | 47% |
| ImageNet-1K 512px | FID↓ (+G) | 4.9 | 9.5 | 48% |
| ImageNet-1K 512px | FDD↓ (+G) | 106.1 | 233.6 | 55% |
| ImageNet-1K 512px (CCFG) | FID↓ | 4.9 | 9.5 (CFG) | 48% + 33% FLOPs saved |
| Kinetics-700 256px | FID↓ (+G) | 10.7 | 11.3 | 5.3% |
| Qwen-Image 1024px | DPG-Bench Avg | 90.45 (100% tok) → 88.02 (25% tok) | 91.27 | up to 63% FLOPs saved |

- Across three architectures—DiT, U-ViT, and HDiT—512px FID is reduced by 53%, 28%, and 23%, respectively.
- Convergence speedup: 3.3× at 256px, 4.0× at 512px.
- ELIT gains increase with model scale while the relative overhead decreases.

### Ablation Study
- **Group size**: 4×4 (16 groups) is optimal at both 256px and 512px; 1×1 degenerates to one-to-one mapping and performs poorly, while 16×16 covering the full image also underperforms. Moderate grouping provides coarse-grained spatial regularization combined with flexible intra-group reallocation.
- **Block allocation**: placing approximately 67–71% of blocks in the latent core is optimal (DiT-B: 3-6-3 or 4-4-4; DiT-XL: 4-20-4).
- **Tail dropping vs. random dropping**: tail dropping significantly outperforms random token dropping, confirming that the importance-ordered hierarchy is critical.
- **Multi-budget joint training vs. single-budget independent training**: joint training outperforms independent training at all budget points, indicating that multi-budget training itself acts as a regularizer.
- **Read/Write design**: a single cross-attention layer outperforms Q-Former-style and full self-attention designs; increasing Write or FFN capacity yields improvements but adds overhead.

## Highlights & Insights
- **Minimalist yet effective**: adding only two lightweight cross-attention layers and a set of learnable latent tokens—without modifying the training objective or introducing auxiliary losses—yields consistently large improvements.
- **The zero-padding experiment** provides an elegant demonstration of DiT's computational waste, using a synthetic ablation to prove that DiT cannot reallocate computation across regions.
- **CCFG mixed guidance** is ingenious: a multi-budget model inherently contains a weaker version of itself, enabling autoguidance + CFG combination and delivering 33% acceleration at no extra cost.
- **Visualization of Read attention** intuitively illustrates importance ordering: earlier latents attend to global structure while later ones focus on fine-grained texture.
- **Strong generality**: effective across four architectures (DiT, U-ViT, HDiT, MM-DiT) and two tasks (image and video generation).

## Limitations & Future Work
- The effectiveness of large-scale training from scratch remains unverified; the Qwen-Image experiment uses distillation fine-tuning rather than training from scratch.
- CCFG is more prone to image oversaturation than CFG, requiring a lower guidance scale.
- Per-step budget scheduling across sampling steps (different noise levels may warrant different token counts) is unexplored; the authors themselves identify this as future work.
- Per-group adaptive token allocation experiments failed—predicting group-level importance via loss maps did not outperform uniform allocation—suggesting that the Read operation already implicitly achieves this effect.

## Related Work & Insights
- **vs. FlexiDiT / multi-patch training**: FlexiDiT achieves variable computation via multiple patchification sizes but still distributes computation uniformly in the spatial domain; ELIT reallocates computation in the latent domain and performs substantially better (ablations show multi-patch training even underperforms standard DiT).
- **vs. RIN/FIT**: RIN and FIT also employ latent tokens with read/write interactions but deviate substantially from the DiT design (requiring specialized optimizers such as LAMB) and have fixed inference budgets. ELIT is a drop-in plug-and-play module that preserves the DiT architecture and RF training.
- **vs. Token Merging (ToMe/SDTM)**: training-free methods are bounded by DiT's quality ceiling; ELIT using only 25% tokens (FID=14.2) still outperforms the DiT baseline (FID=20.9).

The latent interface paradigm is transferable to other transformer architectures (e.g., ViT for visual understanding) for adaptive computation allocation. Importance ordering combined with tail dropping can serve as a general variable-budget inference strategy.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The core idea is not entirely new (latent tokens with read/write interactions appear in RIN/FIT), but streamlining it into a minimalist form and seamlessly integrating it with DiT and multi-budget elastic inference constitutes a significant contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four architectures, two tasks, multiple resolutions, large-scale model experiments, detailed ablations, and failed experiments are all reported.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation experiments are elegant, method descriptions are clear, figures are of high quality, and the appendix is exceptionally detailed.
- **Value**: ⭐⭐⭐⭐⭐ Highly practical; the drop-in design enables direct application to existing DiT systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Circuit Mechanisms for Spatial Relation Generation in Diffusion Transformers](circuit_mechanisms_for_spatial_relation_generation_in_diffusion_transformers.md)
- [\[CVPR 2026\] EdgeDiT: Hardware-Aware Diffusion Transformers for Efficient On-Device Image Generation](edgedit_hardware-aware_diffusion_transformers_for_efficient_on-device_image_gene.md)
- [\[CVPR 2026\] DA-VAE: Plug-in Latent Compression for Diffusion via Detail Alignment](da-vae_plug-in_latent_compression_for_diffusion_via_detail_alignment.md)
- [\[CVPR 2026\] Memory-Efficient Fine-Tuning Diffusion Transformers via Dynamic Patch Sampling and Block Skipping](memory-efficient_fine-tuning_diffusion_transformers_via_dynamic_patch_sampling_a.md)
- [\[CVPR 2026\] Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers](just-in-time_training-free_spatial_acceleration_for_diffusion_transformers.md)

</div>

<!-- RELATED:END -->
