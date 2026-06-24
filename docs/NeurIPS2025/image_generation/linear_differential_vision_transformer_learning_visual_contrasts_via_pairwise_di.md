---
title: >-
  [Paper Note] Linear Differential Vision Transformer: Learning Visual Contrasts via Pairwise Differentials
description: >-
  [NeurIPS 2025][Image Generation][Vision Transformer] This paper proposes Visual-Contrast Attention (VCA), which generates compact positive/negative visual-contrast tokens via spatial pooling and performs differential interaction, reducing self-attention complexity from $O(N^2C)$ to $O(NnC)$ ($n \ll N$), while achieving consistent improvements on both image classification and generation tasks.
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Vision Transformer"
  - "Linear Attention"
  - "Differential Attention"
  - "Image Classification"
date: 2026-05-08
content_hash: a8251db580e088eb
---

# Linear Differential Vision Transformer: Learning Visual Contrasts via Pairwise Differentials

**Conference**: NeurIPS 2025
**arXiv**: [2511.00833](https://arxiv.org/abs/2511.00833)  
**Code**: [https://github.com/LeapLabTHU/LinearDiff](https://github.com/LeapLabTHU/LinearDiff)  
**Area**: Vision Transformer / Efficient Attention
**Keywords**: Vision Transformer, Linear Attention, Differential Attention, Image Classification, Image Generation

## TL;DR

This paper proposes Visual-Contrast Attention (VCA), which generates compact positive/negative visual-contrast tokens via spatial pooling and performs differential interaction, reducing self-attention complexity from $O(N^2C)$ to $O(NnC)$ ($n \ll N$), while achieving consistent improvements on both image classification and generation tasks.

## Background & Motivation

Multi-head self-attention (MHSA) in Vision Transformers computes quadratic query–key interactions over all token pairs, spending substantial computation on visually weak or redundant correlations. Existing optimization approaches follow two main directions:

**Restricting the receptive field**: sliding-window methods (Swin), dilated attention (DiNAT), etc., at the cost of long-range dependencies.

**Low-rank approximation**: Linformer, Performer, etc., which maintain a global view but treat all correlations as equally important.

Differential Attention in language models subtracts two attention maps to highlight discriminative signals, but remains quadratic in complexity and ignores image-specific redundancy structure.

The core premise of this paper is: **compress the dense query field first, then perform the expensive comparisons**. The spatial smoothness of natural images implies that neighboring patches typically carry nearly identical information, so the query set can be reduced to a small number of prototypes before matching.

## Method

### Overall Architecture

VCA serves as a plug-and-play replacement for MHSA and operates in two stages: Stage I performs global contrast (visual-contrast tokens interact with all keys/values), and Stage II performs per-patch differential attention (original queries interact with the contrast map).

### Key Designs

1. **Visual-Contrast Token Generation**: The query matrix $\mathbf{q}^{(m)} \in \mathbb{R}^{N \times d}$ is reshaped into a 2D spatial layout $H \times W \times d$ and downsampled via average pooling to a coarse $h \times w$ grid, yielding $n = h \cdot w$ visual-contrast tokens (e.g., $8 \times 8 = 64$). Two independent sets of learnable positional embeddings $\mathbf{e}^+$ and $\mathbf{e}^-$ are then added to split the tokens into positive and negative streams. The key elegance of this design is that the two streams share the same pooled content but are decoupled into complementary correlations via distinct positional embeddings.

2. **Stage I — Global Contrast**: The positive and negative visual-contrast tokens each attend to all keys and values, producing $\hat{\mathbf{v}}_+^{(m)}$ and $\hat{\mathbf{v}}_-^{(m)}$ (both of size $n \times d$), followed by a differential operation and RMSNorm: $\hat{\mathbf{v}}^{(m)} = (1-\lambda_{init}^{(1)}) \text{RMSNorm}(\hat{\mathbf{v}}_+^{(m)} - \lambda^{(1)} \hat{\mathbf{v}}_-^{(m)})$. This step compresses the global scene into a contrast map that highlights discriminative differences, with complexity $O(Nnd)$.

3. **Stage II — Per-Patch Differential Attention**: The original $N$ queries each compute attention maps ($N \times n$) against the positive and negative visual-contrast tokens separately; the differential is taken and used to weight values from the contrast map. Since the contrast map contains only $n$ tokens, all three matrix multiplications scale proportionally to $nN$. The final output is RMSNorm-scaled by $(1-\lambda_{init}^{(2)})$ and concatenated across heads.

### Loss & Training

- **Image Classification**: Identical training setup as the baseline — AdamW optimizer for 300 epochs with cosine learning rate decay, combined with RandAugment, Mixup, CutMix, and Random Erasing.
- **Image Generation**: Follows the original DiT/SiT recipe — batch size 256, 400K iterations, constant learning rate $10^{-4}$, EMA decay 0.9999, random horizontal flipping only.
- VCA adds fewer than 0.3M parameters (on DeiT-Tiny) and introduces no additional FLOPs.

## Key Experimental Results

### Image Classification (ImageNet-1K)

| Backbone | Params | FLOPs | Baseline Top-1 | +VCA Top-1 | Gain |
|---------|--------|-------|----------------|------------|------|
| DeiT-Tiny | 5.7M→6.0M | 1.2G | 72.2% | **75.6%** | +3.4 |
| DeiT-Small | 22.1M→22.6M | 4.6G | 79.8% | **80.7%** | +0.9 |
| PVT-Tiny | 13.2M→11.6M | 1.9G→2.0G | 75.1% | **78.2%** | +3.1 |
| Swin-Tiny | 28.9M→28.5M | 4.5G→4.6G | 81.3% | **82.3%** | +1.0 |
| CSwin-Tiny | 20.5M→20.4M | 4.3G | 82.7% | **83.3%** | +0.6 |

### Image Generation (ImageNet-1K 256×256, FID-50K↓)

| Model | Baseline FID | +VCA FID | Improvement |
|------|-------------|----------|------|
| DiT-S/2 | 67.2 | **62.3** | ↓4.9 |
| DiT-S/4 | 97.9 | **92.7** | ↓5.2 |
| DiT-B/2 | 42.9 | **38.9** | ↓4.0 |
| SiT-S/2 | 57.3 | **53.0** | ↓4.3 |
| SiT-B/2 | 35.3 | **32.7** | ↓2.6 |

### Ablation Study

| Configuration | DeiT-Tiny Top-1 | DiT-S/2 FID | Note |
|------|-----------------|-------------|------|
| Stage I only | 75.4 | 64.6 | Global contrast contributes |
| Stage II only | 75.5 | 64.3 | Per-patch differential contributes |
| Vanilla Diff. Attention (two-stage) | 75.1 | 63.9 | Does not exploit visual structure |
| **VCA (two-stage)** | **75.6** | **62.3** | Two-stage synergy is optimal |
| Both streams use Emb. only | 75.1 | 63.7 | Lacks image content |
| Both streams use Pool+Emb. | **75.6** | **62.3** | Pooling + embedding is optimal |

### Key Findings

- The contributions of the two stages are nearly additive, indicating that global contrast and per-patch differential capture complementary information.
- VCA outperforms direct application of the language-domain differential attention (+0.5% Top-1 and −1.8 FID), demonstrating the necessity of vision-specific design.
- Spatial pooling provides low-variance global cues, and dual positional embeddings are essential for decoupling complementary correlations.
- Gains are largest for smaller models (DeiT-T: +3.4%) and remain consistent for Base-scale models (+0.4–1.0%).
- The approach is effective for both diffusion-based and flow-based generative paradigms.

## Highlights & Insights

- **Core Insight**: Attention should be viewed not merely as a similarity metric but as an explicit arena for contrast — focusing on "what makes one region differ from another" is more discriminative than measuring "how similar they are."
- Linear complexity and differential attention are elegantly combined: pooling first compresses the query field, and differential attention then highlights contrasts — achieving two goals at once.
- Architecture-agnostic applicability is a significant advantage; VCA consistently improves plain ViTs, hierarchical ViTs (PVT/Swin/CSwin), and generative models (DiT/SiT).
- The parameter overhead is minimal (< 0.3M) with no additional FLOPs, making it deployment-friendly.
- The complexity reduction factor is $N/n$; for $256^2$ images with patch size 16, this corresponds to approximately a 256× reduction.

## Limitations & Future Work

- Task-agnostic average pooling may discard edge-rich fine-grained details.
- For small images, the overhead of micro-attention may offset speed gains.
- Extensions to video, 3D, or language tasks remain unexplored.
- The pooling strategy is fixed (average pooling); adaptive or learnable pooling may yield further improvements.
- Actual inference latency is not reported (only FLOPs analysis is provided).

## Related Work & Insights

- Differential Transformer (Ye et al., 2024), which introduced differential attention for language modeling, serves as the direct inspiration.
- Linear attention methods based on intermediary tokens, such as Agent Attention (Han et al., 2024), form the technical foundation.
- The work is conceptually aligned with the attention intermediary idea in Efficient DiT (Pu et al., 2024).
- This work provides an important reference for future efficient visual architecture design: "contrast" should be prioritized over "similarity."

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation](../../ICCV2025/image_generation/lit_delving_into_a_simple_linear_diffusion_transformer_for_image_generation.md)
- [\[ICML 2026\] Linearizing Vision Transformer with Test-Time Training](../../ICML2026/image_generation/linearizing_vision_transformer_with_test-time_training.md)
- [\[CVPR 2025\] LaVin-DiT: Large Vision Diffusion Transformer](../../CVPR2025/image_generation/lavin-dit_large_vision_diffusion_transformer.md)
- [\[NeurIPS 2025\] On the Emergence of Linear Analogies in Word Embeddings](on_the_emergence_of_linear_analogies_in_word_embeddings.md)
- [\[NeurIPS 2025\] Pairwise Optimal Transports for Training All-to-All Flow-Based Condition Transfer Model](pairwise_optimal_transports_for_training_all-to-all_flow-based_condition_transfe.md)

</div>

<!-- RELATED:END -->
