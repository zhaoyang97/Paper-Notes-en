---
title: >-
  [Paper Note] Lightning Unified Video Editing via In-Context Sparse Attention
description: >-
  [ICML 2026][Video Generation][In-Context Learning] To address the secondary attention bottleneck in video editing under the In-Context Learning (ICL) paradigm…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "In-Context Learning"
  - "Sparse Attention"
  - "Video Editing"
  - "Taylor Approximation"
  - "Query Sharpness"
date: 2026-05-08
content_hash: ff6cca1a2ad3550e
---

# Lightning Unified Video Editing via In-Context Sparse Attention

**Conference**: ICML 2026  
**arXiv**: [2605.04569](https://arxiv.org/abs/2605.04569)  
**Code**: Not yet released  
**Area**: Video Generation / Sparse Attention / Video Editing  
**Keywords**: In-Context Learning, Sparse Attention, Video Editing, Taylor Approximation, Query Sharpness

## TL;DR
To address the secondary attention bottleneck in video editing under the In-Context Learning (ICL) paradigm, the authors design In-context Sparse Attention (ISA) based on two insights: "context tokens are significantly less salient than source tokens" and "Query sharpness is proportional to Taylor approximation error." They train LIVEditor, which both accelerates inference by ~60% and surpasses SOTA full-attention models on multiple benchmarks.

## Background & Motivation

**Background**: Video editing is transitioning from task-specific expert models to the In-Context Learning (ICL) paradigm—concatenating context (reference frames/editing instructions) and source (video to be edited) tokens and feeding them into a unified DiT, allowing full attention to interact freely over long sequences. This approach is simple, scalable, and has become mainstream in recent works such as EditVerse and Ditto.

**Limitations of Prior Work**: Videos are inherently long sequences, with 5K–50K tokens making the $\mathcal{O}(N^2)$ attention complexity a major inference bottleneck. ICL further doubles the computation by concatenating source and context tokens of equal length. Existing sparse attention methods (Radial, Sparge, STA, SWA, VSA, etc.) are designed for general video generation and do not distinguish between source and context, thus failing to exploit the unique structure of ICL.

**Key Challenge**: Context tokens are numerous but contribute little to the final output; pruning them indiscriminately risks losing a few truly critical context tokens. Different Queries have vastly different tolerance to approximation error, but existing methods apply the same approximation to all Queries, so "high-error Queries using approximation" becomes the main cause of overall accuracy drop.

**Goal**: (i) Construct an "almost lossless" sparse attention for ICL video editing; (ii) support both end-to-end learning during training and direct replacement of full attention during inference; (iii) train a truly usable unified video editing model on this basis.

**Key Insight**: By visualizing the attention score matrix under ICL, the authors find that source-source block scores are much higher than source-context; theoretically, they prove that the error upper bound of the 0th-order Taylor approximation is determined by the Query "sharpness" $M_i = \mathrm{Var}(\mathrm{softmax}(Q^c_i (K^c)^\top))$, thus turning "which tokens to keep, which Queries to compute precisely" into computable proxies.

**Core Idea**: Use pre-selection to prune redundant context K/V, then use Query sharpness to route Queries to either full attention or 0th-order Taylor block-sparse attention, achieving "precise where needed, approximate where possible."

## Method

### Overall Architecture
The input is a long sequence after ICL concatenation $Q,K,V \in \mathbb{R}^{B\times H\times N\times D}$, with the first $L_{src}$ tokens from the source and the last $L_{ctx}$ from the context. ISA's forward pass consists of four steps: (1) Use pooling attention to obtain coarse-grained score $S_\text{coarse}$ and block mask $M_\text{coarse}$; (2) On the context sub-block of $S_\text{coarse}$, perform Top-k selection to compress context K/V to length $\alpha_s L_{ctx}$; (3) Use coarse-grained variance $M_i$ to evaluate the sharpness of each Query block and split Queries by Flat Ratio $\alpha_f$; (4) High-sharpness Queries use FlashAttention v2/3, low-sharpness Queries use block-wise 0th-order Taylor sparse attention, and finally merge the two online-softmax outputs. The entire forward/backward is implemented as a trainable kernel using Triton/TileLang.

### Key Designs

1. **Context Pre-Selection (Saliency-based K/V Pruning)**:

    - **Function**: Prunes the concatenated K/V length from $N$ to $L_{src} + \alpha_s L_{ctx}$ without losing key context, reducing overall complexity from $\mathcal{O}(NSD)$ to $\mathcal{O}(N(L_{src}+\alpha_s L_{ctx})D)$.
    - **Mechanism**: First, pooling attention yields $S_\text{coarse}\in\mathbb{R}^{B\times H\times N_Q\times N_K}$; extract the source-Query to context-Key sub-block $S^\text{ctx}_\text{coarse}$, average along the Query axis, then use TopK to select $\alpha_s\lceil L_{ctx}/b\rceil$ most salient context blocks, and finally use gather + concat to reconstruct new $K_\text{new}, V_\text{new}$.
    - **Design Motivation**: Visualization (Fig. 4-5) shows $Q^{src}(K^{src})^\top$ is much larger than $Q^{src}(K^{ctx})^\top$, especially in deeper layers, indicating most context tokens are "bystanders." Pruning them saves attention computation and removes noise from synthetic context—explaining why ISA can outperform full attention even in training-free scenarios.

2. **Block-wise 0th-order Taylor Sparse Attention (Switching Precision/Approximation by Block Mask)**:

    - **Function**: For each Query block $Q_i$ and Key/Value block pair $(K_j, V_j)$, $M_{ij}$ determines whether to use the exact or Taylor-approximate path, reducing complexity from $\mathcal{O}(L_Q L_K D)$ to $\mathcal{O}(L_Q D)$.
    - **Mechanism**: When $M_{ij}=1$, use standard online-softmax: $S_{ij}=Q_i K_j^\top/\sqrt{D}$, $P_{ij}=\exp(S_{ij})$, $\ell_i \mathrel{+}= \mathrm{rowsum}(P_{ij})$, $O_i \mathrel{+}= P_{ij} V_j$; when $M_{ij}=0$, use pre-pooled $K_j^c, V_j^c$ as substitutes, $S_{ij}^c=Q_i (K_j^c)^\top/\sqrt{D}$, $P_{ij}^c = \exp(S_{ij}^c)$, treat as representative for the whole block, $\ell_i \mathrel{+}= P_{ij}^c \cdot L_K$, $O_i \mathrel{+}= P_{ij}^c V_j^c \cdot L_K$, finally $O_i = O_i/\ell_i$ for softmax normalization.
    - **Design Motivation**: The authors tried 1st/2nd-order Taylor expansions, but they are hard to kernelize on GPU and incur high extra computation. The 0th-order expansion integrates with FlashAttention's online-softmax and enables contiguous block access on hardware—a practical "sweet spot."

3. **Grouped Computation by Query Sharpness (Routing High/Low Error Queries)**:

    - **Function**: Splits Queries into two groups—high-error Queries use full attention for accuracy, low-error Queries use 0th-order Taylor sparse attention for efficiency, with Flat Ratio $\alpha_f$ controlling the split.
    - **Mechanism**: Theorem 3.1 proves the 0th-order Taylor error upper bound is determined by $M_i = \mathrm{Var}(\mathrm{softmax}(Q^c_i(K^c)^\top))$ (inter-block mean variance) and $\|Q(K-K^c)^\top\|_\infty^2$. The latter is expensive and not a good proxy in practice, so only $M_i$ is used, as it can be read almost for free from pooling scores. Sort $M_i$ descendingly, route the top $\alpha_f$ proportion of Queries to FlashAttention, the rest to Taylor sparse; within the latter, $\alpha_{ns}$ controls the "non-sparse" proportion inside blocks, pushing overall sparsity to 93.75%.
    - **Design Motivation**: Previous sparse methods (STA, SWA, Sparge, etc.) treat all Queries equally, forcing approximation even for a few "very sharp" Queries, which then dominate the error. ISA's "dynamic routing" preserves exact computation for critical Queries, enabling high overall sparsity without sacrificing visual quality.

### Loss & Training
LIVEditor is trained in two stages on the Wan 2.2 high-noise branch: Stage 1 uses 1.7M mix-quality samples, $\eta = 1\mathrm{e}{-5}$, batch size 16 to learn general editing semantics; Stage 2 uses a 0.089M high-quality subset, $\eta = 1\mathrm{e}{-6}$ for aesthetic and instruction alignment refinement, both with DeepSpeed ZeRO-3 Offload. To mitigate positional bias from source/context length mismatch, decoupled RoPE is introduced: source and context are renumbered from 0 independently. Default hyperparameters are $\alpha_s = 0.125, \alpha_{ns} = 0.0625, \alpha_f = 0.5$.

## Key Experimental Results

### Main Results

| Dataset | Method | Quality | Text Align | Temporal Cons. | Editing Quality | Pick(Frame) | Pick(Video) |
|---------|--------|---------|------------|----------------|-----------------|-------------|-------------|
| EditVerseBench | EditVerse (Prev. SOTA) | 7.65 | 20.07 | 26.73 | 23.93 | 98.56 | 98.42 |
| EditVerseBench | LIVEditor (full-attn) | 7.62 | 19.98 | 27.13 | 23.80 | 99.24 | 99.19 |
| EditVerseBench | **LIVEditor (ISA)** | **7.89** | **20.09** | **27.19** | **24.55** | **99.32** | **99.22** |

ISA surpasses the full-attention counterpart on almost all metrics—demonstrating that pre-selection acts as "denoising," directly pruning context tokens that would otherwise degrade performance.

### Ablation Study

| Configuration | Quality | Text Align | Temporal Cons. | Editing Quality | SpeedUp vs FA3 |
|---------------|---------|------------|----------------|-----------------|----------------|
| Radial Attention | 7.09 | 19.68 | 26.86 | 24.13 | 1.28× |
| Sparge Attention | 7.44 | 19.69 | 26.75 | 23.76 | 1.40× |
| STA | 4.45 | 15.76 | 13.02 | 4.82 | 2.09× |
| SWA | 5.95 | 18.48 | 20.06 | 16.74 | 1.37× |
| VSA | 3.60 | 16.85 | 17.30 | 9.88 | 1.38× |
| LIVEditor (full-attn) | 7.62 | 19.98 | 27.13 | 23.80 | 1.00× |
| **LIVEditor (ISA, training-free)** | **7.78** | **20.07** | **27.15** | 24.15 | **1.47×** |

| Stage Ablation | Quality | Text Align | Temporal Cons. | Editing Quality |
|----------------|---------|------------|----------------|-----------------|
| Stage I (1.7M mix) | 6.46 | 19.50 | 25.27 | 22.63 |
| Stage II (+0.089M HQ) | 7.89 | 20.09 | 27.19 | 24.55 |

### Key Findings
- **ISA training-free already surpasses full attention**: Without any fine-tuning, quality reaches 7.78, exceeding the original full-attn's 7.62, indicating that pre-selection acts as implicit "context denoising."
- **Flat Ratio is the most sensitive hyperparameter**: When $\alpha_f$ decreases, all metrics drop significantly, so it is fixed at 0.5 during training; $\alpha_{ns}, \alpha_s$ can be pushed to 0.0625, 0.125 with almost no loss, reflecting the "accuracy-sensitive vs compute-sensitive" asymmetry that enables ISA to achieve ~94% sparsity.
- **Trainable kernel brings extra gains**: Fig. 7 shows that after training, the output difference between ISA and full attention is compressed in almost all blocks—trainability allows ISA to "actively adapt" rather than just "approximate."

## Highlights & Insights
- **Structured sparsity in ICL attention**: The authors explicitly model "source vs context" as first-order structural information, essentially providing a prior that "context is reference, source is the main actor." This task-aware sparsity is more targeted than generic sparsity (local/radial windows).
- **Query sharpness = Taylor error proxy**: Theorem 3.1 turns the engineering intuition of "should Taylor approximation be used" into a metric $M_i$ that can be computed almost for free from pooling scores—an elegant contribution. This "routing expensive computation with cheap statistics" can be transferred to other mixed-precision inference scenarios.
- **Trainable sparsity + data-driven**: Implementing ISA's backward as a Triton kernel enables sparse attention to participate in end-to-end training, which is key to achieving "approximation + further improvement."
- **synthetic-as-context, real-as-source**: The data pipeline design is instructive—placing Gemini-generated images on the context side and real videos on the source side, thus avoiding synthetic artifacts contaminating the main output by design.

## Limitations & Future Work
- The core of the paper is attention acceleration, but latency improvement only addresses one bottleneck in ICL video editing; VAE decoding, text encoding, CFG, and other components are untouched, so end-to-end inference acceleration may fall short of the 60% achieved in the attention module.
- The three hyperparameters $\alpha_f, \alpha_{ns}, \alpha_s$ are manually swept, lacking automatic scheduling; transferring to longer sequences or different resolutions may require re-tuning.
- The accuracy of the 0th-order Taylor approximation under severe OOD (e.g., non-natural synthetic content, extremely long videos) is not fully validated; the theoretical bound only guarantees "bounded error," not "negligible error."
- ISA only distinguishes source/context in ICL scenarios; no extension is provided for truly multi-context inputs.

## Related Work & Insights
- **vs Radial / Sparge / STA / SWA / VSA**: These are task-agnostic sparsity methods (based on distance, momentum, or fixed masks), not distinguishing source and context; ISA's task-aware pre-selection is the key difference from them and LIVEditor.
- **vs EditVerse / Ditto / InsV2V / Lucy Edit**: These use full attention for ICL video editing; this work essentially "replaces the underlying attention + reuses similar training paradigms," proving ISA is "drop-in." The new 1.7M data pipeline and two-stage training further advance overall SOTA.
- **vs FlashAttention v2/v3**: FlashAttention optimizes IO and exact softmax; ISA is complementary—not competitive. High-sharpness Queries are still handled by FA3, while low-sharpness Queries use the custom sparse kernel, forming a "routing, not replacement" relationship.

## Rating
- Novelty: ⭐⭐⭐⭐ Combines ICL structural priors and Taylor approximation theory to deliver trainable sparse attention; clear logic, solid theory and engineering.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple benchmarks (EditVerse/VIE/IVE/FiVE), extensive sparse attention baselines, hyperparameter sensitivity, and both training-free/trainable modes covered.
- Writing Quality: ⭐⭐⭐⭐ Theorems and algorithms are clearly stated; visualizations in Fig. 4-5 intuitively explain "why prune context"; some formula formatting is messy but does not hinder understanding.
- Value: ⭐⭐⭐⭐ Provides the first systematic solution for "ICL video editing-specific sparse attention," achieving lossless ~60% acceleration, with direct engineering value for the long-sequence video generation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VideoCoF: Unified Video Editing with Temporal Reasoner](../../CVPR2026/video_generation/videocof_unified_video_editing_with_temporal_reasoner.md)
- [\[NeurIPS 2025\] Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation](../../NeurIPS2025/video_generation/radial_attention_onlog_n_sparse_attention_with_energy_decay_for_long_video_gener.md)
- [\[NeurIPS 2025\] VORTA: Efficient Video Diffusion via Routing Sparse Attention](../../NeurIPS2025/video_generation/vorta_efficient_video_diffusion_via_routing_sparse_attention.md)
- [\[NeurIPS 2025\] VSA: Faster Video Diffusion with Trainable Sparse Attention](../../NeurIPS2025/video_generation/vsa_faster_video_diffusion_with_trainable_sparse_attention.md)
- [\[CVPR 2026\] NOVA: Sparse Control, Dense Synthesis for Pair-Free Video Editing](../../CVPR2026/video_generation/nova_sparse_control_dense_synthesis_for_pair-free_video_editing.md)

</div>

<!-- RELATED:END -->
