---
title: >-
  [Paper Note] SparseDiT: Token Sparsification for Efficient Diffusion Transformer
description: >-
  [NeurIPS 2025][Image Generation][Diffusion Transformer] This paper proposes SparseDiT, which achieves 55% FLOPs reduction and 175% inference throughput improvement on DiT-XL 512×512 with only 0.09 FID degradation. The me…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Diffusion Transformer"
  - "token sparsification"
  - "inference acceleration"
  - "timestep-adaptive"
  - "architecture design"
date: 2026-05-08
content_hash: e998c92bdb98de6e
---

# SparseDiT: Token Sparsification for Efficient Diffusion Transformer

**Conference**: NeurIPS 2025
**arXiv**: [2412.06028](https://arxiv.org/abs/2412.06028)
**Code**: None
**Area**: Diffusion Models / Model Efficiency
**Keywords**: Diffusion Transformer, token sparsification, inference acceleration, timestep-adaptive, architecture design

## TL;DR

This paper proposes SparseDiT, which achieves 55% FLOPs reduction and 175% inference throughput improvement on DiT-XL 512×512 with only 0.09 FID degradation. The method employs a three-stage spatial architecture (bottom Poolingformer + middle Sparse-Dense Token Module + top full-density processing) combined with a dynamic pruning-rate schedule along the temporal dimension, and successfully extends to video generation and text-to-image generation tasks.

## Background & Motivation

Diffusion Transformers (DiT) have demonstrated strong capabilities in image and video generation by leveraging the scalability of Transformers, serving as the backbone of advanced systems such as Sora. However, DiT suffers from severe computational inefficiency: the quadratic complexity of self-attention grows rapidly with the number of tokens, and the denoising process requires a large number of sampling steps.

Existing acceleration methods primarily focus on reducing the number of sampling steps (e.g., ODE solvers, consistency models, knowledge distillation), while neglecting the efficiency of the **DiT architecture itself**. Unlike U-Net's contracting-expanding structure, which naturally reduces computation, DiT performs full-size token self-attention at every layer. Directly migrating token reduction techniques from classification tasks (e.g., ToMeSD) to DiT yields poor results — a merge ratio of 0.1 on DiT-XL causes FID to spike to 14.74.

Through an in-depth analysis of DiT attention maps, the authors identify three key insights:

**Bottom layers exhibit near-uniform attention distributions**: Self-attention in the bottom layers approximates global average pooling, contributing little beyond what a simple pooling operation would provide.

**Middle layers alternate between global and local attention**: Some layers focus on local details while others focus on global structure, and this pattern is consistent across all sampling steps.

**Later denoising steps require increasingly local information**: As denoising progresses, attention variance increases and the model focuses more on local details.

Based on these insights, SparseDiT dynamically adjusts token density along both spatial and temporal dimensions.

## Method

### Overall Architecture

SparseDiT partitions the Transformer layers of DiT into three segments: Bottom, Middle, and Top. The bottom segment uses Poolingformer to efficiently capture global features; the middle segment employs multiple Sparse-Dense Token Modules (SDTM) to alternately process sparse and dense tokens; the top segment uses the original dense Transformer to refine high-frequency details. The primary computational savings stem from sparse token processing in the middle segment.

### Key Designs

1. **Poolingformer (Bottom)**: Motivated by the observation that bottom-layer attention maps are nearly uniform, this design replaces self-attention in the bottom layers with global average pooling. Specifically, Q and K are removed, and only V is globally average-pooled and added back to the input tokens: $X = X + \bar{V}$. An ablation experiment validates this simplification — replacing the attention maps of the first two layers with all-ones matrices produces nearly identical generation results. Notably, sparse tokens cannot be used in the bottom layers (which would cause training instability), so the full token set must be retained.

2. **Sparse-Dense Token Module (SDTM, Middle)**: The core module, which decouples global structure extraction from local detail extraction. The pipeline is as follows:

   - **Sparse token generation**: Dense tokens $X \in \mathbb{R}^{N \times C}$ are initialized into sparse tokens $X_s \in \mathbb{R}^{M \times C}$ ($M \ll N$) via spatial adaptive pooling, maintaining spatial uniformity. An attention layer then enables sparse tokens to interact with full-size tokens to integrate global information.
   - **Sparse Transformer processing**: Subsequent Transformer layers operate solely on sparse tokens, substantially reducing computation.
   - **Dense token recovery**: Sparse tokens are upsampled and fused with the original dense tokens via two linear layers ($X_{\text{merged}} = \text{UpSample}(X_s) \cdot W_1 + X \cdot W_2$), followed by an attention layer for further integration.
   - **Dense Transformer processing**: A small number of dense Transformer layers enhance local details.
   - Multiple SDTMs are stacked in cascade (4 by default), alternating between sparse and dense representations to effectively preserve both structural and detail information.

3. **Timestep-Adaptive Dynamic Pruning Rate**: Early denoising steps primarily generate low-frequency global structure, while later steps generate high-frequency details. Accordingly, a dynamic pruning rate $r$ is designed: the highest pruning rate $r_{\min}$ is maintained for the first $T/4$ steps, then linearly decayed to $r_{\max}$, gradually increasing the token count. During training, a gating function is introduced to elegantly resolve the conflict between batched training and random timestep sampling.

### Loss & Training

SparseDiT is realized by fine-tuning a pretrained DiT model, with fine-tuning requiring only approximately 6% of the iterations needed for training from scratch (e.g., ~400K iterations for DiT-XL). Initialization strategy: Poolingformer does not load Q/K parameters; fusion weights $W_1$ in SDTM are initialized to all zeros and $W_2$ to the identity matrix, ensuring the dense token path is an identity mapping at initialization. All Transformer layers are initialized from the corresponding pretrained weights. Sinusoidal positional encodings are reintroduced at the boundaries of sparse/dense token transitions.

## Key Experimental Results

### Main Results

| Model/Resolution | Method | FLOPs (G) | Throughput (img/s) | FID↓ | IS↑ |
|---|---|---|---|---|---|
| DiT-XL/256² | Original | 118.64 | 1.58 | 2.27 | 278.24 |
| DiT-XL/256² | SparseDiT ($r \in [0.44, 0.61]$) | 88.91 (−25%) | 2.13 (+35%) | 2.23 | 278.91 |
| DiT-XL/256² | SparseDiT ($r \in [0.61, 0.86]$) | 68.05 (−43%) | 2.95 (+87%) | 2.38 | 276.39 |
| DiT-XL/512² | Original | 525 | 0.249 | 3.04 | 240.82 |
| DiT-XL/512² | SparseDiT ($r \in [0.61, 0.86]$) | 286 (−46%) | 0.609 (+145%) | 2.96 | 242.4 |
| DiT-XL/512² | SparseDiT ($r \in [0.90, 0.96]$) | 235 (−55%) | 0.685 (+175%) | 3.13 | 236.56 |
| PixArt-α | Original | 148.73 | 0.414 | 4.53 | — |
| PixArt-α | SparseDiT | 91.62 (−38%) | 0.701 (+69%) | 4.29 | — |

At 512×512 resolution, pruning over 90% of tokens results in only 0.09 FID increase while achieving 175% throughput improvement.

### Ablation Study

| Configuration | FID↓ | Notes |
|---|---|---|
| 1 SDTM | NaN | Degenerates to U-Net structure; training collapses |
| 2 SDTMs | 3.86 | Insufficient global/local interaction |
| 3 SDTMs | 2.51 | Significant improvement |
| 4 SDTMs (default) | 2.38 | Best |
| 0 Poolingformers | NaN | Training unstable |
| 2 Poolingformers (default) | 2.38 | Stable |
| 3 Poolingformers | 2.56 | Excessive pooling degrades information |
| Fixed token count 8×8 | 2.48 | Static allocation |
| Dynamic 6×6–10×10 | 2.38 | Dynamic pruning superior |

### Key Findings

- DiT exhibits substantial token redundancy: retaining only ~25% of tokens in certain layers is sufficient to preserve performance.
- From 256² to 512², FLOPs increase only 4.4×, yet throughput drops 6.3×, indicating that the bottleneck of DiT lies in the token count rather than model parameters.
- The alternating sparse-dense design of SDTM is critical to success — degenerating to a U-Net structure (a single SDTM) causes training collapse.
- The method is orthogonal and composable with efficient samplers (DDIM, Rectified Flow) — combining with 5-step RFlow yields a 93.4× speedup.
- On video generation (Latte-XL), the method achieves 56% FLOPs reduction, validating cross-modal generalizability.

## Highlights & Insights

- **Attention-map analysis-driven architecture design**: Rather than blindly compressing the model, the architecture is tailored based on a fine-grained analysis of attention behavior at each layer.
- **The "toy experiment" for Poolingformer is highly convincing**: Replacing bottom-layer attention with all-ones matrices leaves generation results nearly unchanged, directly demonstrating the redundancy of complex attention computation in the bottom layers.
- **Spatiotemporal cooperative sparsification**: Token density is allocated per-layer spatially and dynamically adjusted per denoising stage temporally; the two dimensions work in synergy.
- **Fundamental distinction from token merging (ToMeSD)**: SparseDiT does not reduce tokens uniformly at every layer; instead, it designs a sparsification strategy tailored to DiT's alternating global/local attention pattern.

## Limitations & Future Work

- The architecture is manually predefined — the number of layers and sparse token counts per module require human specification, with no automated search.
- Ablation experiments are conducted at 256×256; optimal configurations for higher resolutions and larger models may differ.
- Integration with more advanced acceleration techniques such as attention distillation has not been explored.
- Gains diminish as the number of SDTMs exceeds 4; however, finer-grained configurations within SDTM (e.g., the ratio of sparse to dense Transformer layers) warrant further investigation.

## Related Work & Insights

- ToMeSD pioneered token reduction in diffusion models but performs poorly on DiT, demonstrating the need for DiT-specific strategies.
- DyDiT compresses DiT along multiple dimensions (token/layer/head/channel) but achieves only 29% FLOPs reduction — far below SparseDiT's 46%–55%.
- U-Net's contracting-expanding structure is itself a form of sparse network, inspiring the alternating sparse-dense design of SDTM.
- EDT attempts a U-Net-like architecture but exhibits a substantial FID gap, suggesting that directly transplanting U-Net structure is ill-suited for DiT.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Token Perturbation Guidance for Diffusion Models](token_perturbation_guidance_for_diffusion_models.md)
- [\[NeurIPS 2025\] Rare Text Semantics Were Always There in Your Diffusion Transformer](rare_text_semantics_were_always_there_in_your_diffusion_transformer.md)
- [\[CVPR 2026\] DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](../../CVPR2026/image_generation/ditic_aligned_diffusion_transformer_for_efficient.md)
- [\[NeurIPS 2025\] ICEdit: Enabling Instructional Image Editing with In-Context Generation in Large Scale Diffusion Transformer](in-context_edit_enabling_instructional_image_editing_with_in-context_generation_.md)
- [\[ICCV 2025\] Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation](../../ICCV2025/image_generation/dense2moe_restructuring_diffusion_transformer_to_moe_for_efficient_text-to-image.md)

</div>

<!-- RELATED:END -->
