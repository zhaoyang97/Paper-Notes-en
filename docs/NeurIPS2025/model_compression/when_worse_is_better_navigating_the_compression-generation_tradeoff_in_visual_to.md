---
title: >-
  [Paper Note] When Worse is Better: Navigating the Compression-Generation Trade-off in Visual Tokenization
description: >-
  [NeurIPS 2025][Model Compression][Visual Tokenizer] This paper systematically investigates the trade-off between visual tokenizer compression rate and generation quality through scaling laws. It finds that more aggressive compression—despite yielding worse reconstruction—benefits generation for smaller models. The paper proposes Causally Regularized Tokenization (CRT), which embeds autoregressive inductive bias into Stage 1 training, achieving 2–3× computational efficiency gains. A 775M-parameter model with 256 tokens/image matches LlamaGen-3B's FID of 2.18.
tags:
  - NeurIPS 2025
  - Model Compression
  - Visual Tokenizer
  - Compression-Generation Trade-off
  - Causal Regularization
  - Scaling Law
  - Autoregressive Generation
date: 2026-05-08
content_hash: 101e056e332bc56a
---

# When Worse is Better: Navigating the Compression-Generation Trade-off in Visual Tokenization

**Conference**: NeurIPS 2025
**arXiv**: [2412.16326](https://arxiv.org/abs/2412.16326)
**Code**: None
**Area**: Model Compression / Image Generation
**Keywords**: Visual Tokenizer, Compression-Generation Trade-off, Causal Regularization, Scaling Law, Autoregressive Generation

## TL;DR

This paper systematically investigates the trade-off between visual tokenizer compression rate and generation quality through scaling laws. It finds that more aggressive compression—despite yielding worse reconstruction—benefits generation for smaller models. The paper proposes Causally Regularized Tokenization (CRT), which embeds autoregressive inductive bias into Stage 1 training, achieving 2–3× computational efficiency gains. A 775M-parameter model with 256 tokens/image matches LlamaGen-3B's FID of 2.18.

## Background & Motivation

**State of the Field**: Modern image generation methods commonly adopt a two-stage training pipeline: Stage 1 trains an auto-encoder (e.g., VQGAN) to compress images into a latent space; Stage 2 trains a generative model (e.g., an autoregressive Transformer) to learn the latent distribution. Consequently, the tokenizer design in Stage 1 profoundly affects the generation performance of Stage 2.

**Limitations of Prior Work**: The optimization objectives of Stage 1 and Stage 2 are fundamentally misaligned—Stage 1 pursues reconstruction quality (low rFID), yet better reconstruction does not necessarily yield better generation (low gFID). Prior works have sporadically observed this phenomenon (e.g., enlarging the codebook unexpectedly degrades generation quality), but a systematic quantitative framework for understanding this trade-off has been lacking.

**Root Cause**: A three-way rate-distortion-generation trade-off exists. At one extreme, no compression yields perfect reconstruction but a hard-to-learn latent distribution; at the other, extreme compression produces a simple distribution but poor reconstruction. The optimal operating point depends on Stage 2 model capacity—an interaction that has not been rigorously studied.

**Paper Goals**: (1) How does compression level affect the generation performance of Stage 2 models at different scales? (2) Given a fixed compute budget, what is the optimal compression-generation configuration? (3) Can Stage 1 tokenizers be designed to directly optimize Stage 2 generation performance?

**Starting Point**: Scaling laws are adopted as an analytical tool to systematically study this trade-off across five orders of magnitude in compute and two orders of magnitude in model scale. CRT is then designed to inject Stage 2's autoregressive inductive bias into Stage 1.

**Core Idea**: By incorporating a lightweight causal Transformer prediction loss into tokenizer training, the resulting tokens are naturally better suited for autoregressive modeling—trading "worse reconstruction" for "better generation."

## Method

### Overall Architecture

The standard two-stage pipeline is retained: Stage 1 uses VQGAN (ResNet encoder + decoder + VQ quantization); Stage 2 uses a Llama-based autoregressive Transformer. CRT introduces a regularization term into Stage 1 training—a 2-layer causal Transformer performs L2 next-token prediction on pre-quantization latents, with gradients backpropagated to the encoder. Stage 2 training remains entirely unchanged.

### Key Designs

1. **Scaling Law Analysis Framework**:

    - Function: Quantitatively reveals the relationship between compression rate and generation performance across different compute budgets.
    - Mechanism: With the Stage 1 tokenizer fixed, Stage 2 model size and training compute are varied, and a scaling law of the form $L(C) = L_{\min} + C^\alpha \cdot e^\lambda$ is fitted. A consistent log-log-linear trend is observed across four orders of magnitude in FLOPs. Key findings: (a) fewer tokens/image are generally more compute-efficient before saturation; (b) a 16k codebook outperforms both 1k and 131k, indicating an optimal compression rate; (c) a 1k codebook outperforms 131k at low compute (gFID gap of 5–10) but the trend reverses at high compute.
    - Design Motivation: Prior work compared tokenizers only at a single scale, neglecting compute budget as a critical dimension.

2. **Causally Regularized Tokenization (CRT)**:

    - Function: Embeds autoregressive inductive bias into Stage 1 training, making tokens easier for Stage 2 to model.
    - Mechanism: On top of the standard VQGAN training loss, a 2-layer causal Transformer performs L2 next-token prediction on pre-quantization latents. Key design choices: (a) L2 loss rather than cross-entropy—because VQ tokens change discretely during training, L2 is naturally similarity-aware; (b) the loss operates on pre-quantization latents rather than quantized tokens—avoiding excessive reconstruction degradation from discretization; (c) loss weight $\lambda = 4$, determined via ablation.
    - Design Motivation: Since Stage 2 is an autoregressive Transformer, making token $i$ as predictable as possible from tokens $0, \ldots, i{-}1$ reduces Stage 2's modeling burden. This effectively lowers the conditional entropy $H(X_i \mid X_{<i})$ in an information-theoretic sense.

3. **Quantitative Analysis of the Reconstruction-Generation Trade-off**:

    - Function: Explains why CRT improves generation despite harming reconstruction.
    - Mechanism: CRT raises rFID from 2.21 to 2.36 (worse reconstruction), yet uniformly reduces Stage 2 validation loss across all sequence positions (especially the tail), confirming a reduction in conditional entropy. Further analysis reveals that changing codebook size has little effect on per-position entropy (codes specialize by position), so codebook scaling has limited impact; whereas changing token count linearly increases inference cost and has a larger scaling impact.
    - Design Motivation: Reconstruction metrics alone are insufficient for tokenizer design—generation performance is the ultimate objective.

### Loss & Training

The CRT regularizer adds only 5% training FLOPs (owing to the extremely lightweight 2-layer Transformer). For a fair comparison, CRT models reduce training iterations by 5% (from 400k to 380k). The CRT loss weight is annealed smoothly from 0 to 4.0 over 1k steps. An independent AdamW optimizer is used.

## Key Experimental Results

### Main Results: ImageNet 256×256 System Comparison

| Model | Params | Tokens/Image | gFID↓ |
|-------|--------|-------------|-------|
| LlamaGen-XL | 775M | 576 | 2.62 |
| LlamaGen-3B | 3.1B | 576 | 2.18 |
| CRT-AR-775M | 775M | 256 | 2.35 |
| CRTopt-AR-775M | 775M | 256 | **2.18** |

### Ablation Study: CRT vs. Baseline (Equal Compute)

| Tokenizer | rFID | 111M gFID | 340M gFID | 775M gFID |
|-----------|------|-----------|-----------|-----------|
| Baseline (2.21 rFID) | 2.21 | 4.90 | 2.89 | 2.55 |
| CRT (2.36 rFID) | 2.36 | 4.34 | 2.75 | 2.35 |

### Key Findings

- CRT yields worse reconstruction (rFID 2.36 vs. 2.21) yet improves generation performance across all model scales and compute budgets—validating the core "worse is better" argument.
- CRT improves the scaling law exponent from $\alpha = -0.65$ to $\alpha = -0.73$, delivering 1.5–3× compute efficiency gains.
- CRTopt (longer training + larger decoder + 131k codebook) matches LlamaGen-3B's FID of 2.18 with 775M parameters and 256 tokens/image—reducing inference compute by 8×.
- L2 loss outperforms cross-entropy: CE loss harms reconstruction too severely without sufficient generation benefit.
- CRT consistently outperforms the baseline on LSUN, confirming the generality of the method.

## Highlights & Insights

- The "worse is better" insight is highly instructive: in two-stage systems, independently optimizing each stage is suboptimal—inter-stage interactions must be considered. CRT trades a marginal reconstruction cost for substantial generation improvement, effectively shifting Stage 2's learning burden into Stage 1. This principle generalizes to any multi-stage ML pipeline.
- Scaling laws prove highly effective as an analytical tool: trends observed across five orders of magnitude in compute budget are robust and avoid spurious conclusions drawn from single-scale comparisons. The log-log-linear relationship makes results predictable and extrapolable.
- CRT's design is remarkably simple: a 2-layer causal Transformer + L2 loss + 5% additional training FLOPs yields 2–3× compute efficiency gains. The paradigm of "embedding inductive bias via a lightweight regularizer" is worth broader adoption.

## Limitations & Future Work

- The largest model evaluated has only 775M parameters (constrained by ImageNet data scale); scaling trends for larger models remain unvalidated.
- Only class-conditional generation is evaluated; extension to text-to-image generation is not explored. The instability of CLIP FID on LSUN also highlights limitations of existing evaluation metrics.
- The CRT method is validated on a fixed VQGAN architecture—its effectiveness on 1D tokenizers or architectures such as VAR may differ.
- Compatibility of CRT with diffusion-based Stage 2 models is not investigated—whether the causal inductive bias benefits non-autoregressive generation remains an open question.

## Related Work & Insights

- **vs. LARP (Wang et al.)**: Also introduces autoregressive inductive bias but uses CE + stochastic VQ + scheduled sampling; this paper achieves better results with a simpler L2 pre-quantization loss—because L2 is more appropriate than CE in VQ-based architectures.
- **vs. VAR (Tian et al.)**: Modifies the autoregressive ordering through multi-scale tokens, requiring architectural changes to the tokenizer; CRT requires no architectural modification, making it more generally applicable.
- **vs. FlexTok / SEED and other 1D tokenizers**: These build entirely new tokenizer architectures to accommodate generation; CRT adds a regularization term to standard VQGAN, incurring significantly lower engineering overhead.

## Rating

- Novelty: ⭐⭐⭐⭐ Analyzing the compression-generation trade-off through a scaling law lens is original; the CRT method is elegant in its simplicity.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers five orders of magnitude in compute, seven model scales, and three datasets; ablations are comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ The experiment-driven narrative is clear and well-structured; figures and tables are well-designed; conclusions are convincing.
- Value: ⭐⭐⭐⭐⭐ Provides significant guidance for visual tokenizer design and multi-stage system optimization.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation](../../ICCV2025/model_compression/bridging_continuous_and_discrete_tokens_for_autoregressive_visual_generation.md)
- [\[NeurIPS 2025\] zip2zip: Inference-Time Adaptive Tokenization via Online Compression](zip2zip_inference-time_adaptive_tokenization_via_online_compression.md)
- [\[NeurIPS 2025\] A Partition Cover Approach for Tokenization](a_partition_cover_approach_to_tokenization.md)
- [\[NeurIPS 2025\] Navigating Simply, Aligning Deeply: Winning Solutions for Mouse vs. AI 2025](navigating_simply_aligning_deeply_winning_solutions_for_mouse_vs_ai_2025.md)
- [\[ICLR 2026\] UniFlow: A Unified Pixel Flow Tokenizer for Visual Understanding and Generation](../../ICLR2026/model_compression/uniflow_a_unified_pixel_flow_tokenizer_for_visual_understanding_and_generation.md)

<!-- RELATED:END -->
