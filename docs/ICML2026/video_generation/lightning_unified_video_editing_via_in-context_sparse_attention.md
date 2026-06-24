---
title: >-
  [Paper Note] Lightning Unified Video Editing via In-Context Sparse Attention
description: >-
  [ICML 2026][Video Generation][In-Context Learning] To address the quadratic attention bottleneck in video editing under the In-Context Learning paradigm, the authors designed In-context Sparse Attention (ISA) based on two insights: "context token saliency is lower than source tokens" and "Query sharpness is proportional to Taylor approximation error." They trained LIVEditor, which achieves a ~60% speedup while surpassing SOTA full-attention models across several benchmarks.
tags:
  - "ICML 2026"
  - "Video Generation"
  - "In-Context Learning"
  - "Sparse Attention"
  - "Video Editing"
  - "Taylor Approximation"
  - "Query Sharpness"
date: 2026-05-08
content_hash: 6efd413973e4e911
---

# Lightning Unified Video Editing via In-Context Sparse Attention

**Conference**: ICML 2026  
**arXiv**: [2605.04569](https://arxiv.org/abs/2605.04569)  
**Code**: Not yet released  
**Area**: Video Generation / Sparse Attention / Video Editing  
**Keywords**: In-Context Learning, Sparse Attention, Video Editing, Taylor Approximation, Query Sharpness

## TL;DR
To address the quadratic attention bottleneck in video editing under the In-Context Learning paradigm, the authors designed In-context Sparse Attention (ISA) based on two insights: "context token saliency is lower than source tokens" and "Query sharpness is proportional to Taylor approximation error." They trained LIVEditor, which achieves a ~60% speedup while surpassing SOTA full-attention models across several benchmarks.

## Background & Motivation

**Background**: Video editing is transitioning from task-specific expert models to the In-Context Learning (ICL) paradigm—concatenating context (reference frames/editing instructions) and source (video to be edited) tokens into a unified DiT, allowing full attention to interact freely over long sequences. This approach is simple and scalable, becoming mainstream in recent works like EditVerse and Ditto.

**Limitations of Prior Work**: Videos are inherently long sequences, where $5\text{K} \sim 50\text{K}$ tokens make the $\mathcal{O}(N^2)$ complexity of attention an inference bottleneck. ICL further quadruples the computation by concatenating source and context of equal length. Existing sparse attention mechanisms (Radial, Sparge, STA, SWA, VSA, etc.) are designed for general video generation and do not distinguish between source and context, thus failing to exploit the specific structure of ICL.

**Key Challenge**: Context tokens are numerous but contribute little to the final output; however, indiscriminate pruning may discard the few truly critical context tokens. Furthermore, the tolerance of different Queries to approximation errors varies significantly, yet existing methods apply the same approximation to all Queries. Using approximation for "high-error Queries" becomes the primary cause of overall accuracy degradation.

**Goal**: (i) Construct an "almost lossless" sparse attention for ICL video editing scenarios; (ii) Support both end-to-end learning during training and direct replacement of full attention during inference; (iii) Train a truly functional unified video editing model based on this.

**Key Insight**: The authors plotted the attention score matrix under ICL and found that scores in the source-source blocks are much larger than those in source-context blocks. They also theoretically proved that the upper bound of the 0-order Taylor approximation error is determined by the "sharpness" of the Query: $M_i = \mathrm{Var}(\mathrm{softmax}(Q^c_i (K^c)^\top))$. Thus, deciding which tokens to keep and which Queries to compute precisely can be transformed into computable proxies.

**Core Idea**: Use pre-selection to prune redundant context K/V, then use Query sharpness to shunt Queries into either full attention or 0-order Taylor block-sparse attention. This ensures "precise calculation where needed, and maximize approximation where possible."

## Method

### Overall Architecture
The input consists of concatenated long sequences $Q,K,V \in \mathbb{R}^{B\times H\times N\times D}$, where the first $L_{src}$ tokens are from the source and the subsequent $L_{ctx}$ tokens are from the context. The forward pass of ISA follows four steps: (1) Use pooling attention to obtain coarse-grained scores $S_\text{coarse}$ and block mask $M_\text{coarse}$, serving as a "cheap probe"; (2) Perform Top-k selection on the context sub-blocks of $S_\text{coarse}$ to compress context K/V to length $\alpha_s L_{ctx}$ (**Context Pre-Selection**); (3) Use coarse-grained variance $M_i$ to evaluate the sharpness of each Query block, shunting Queries based on the Flat Ratio $\alpha_f$ (**Grouped Computation via Query Sharpness**); (4) High-sharpness Queries undergo precise calculation via FlashAttention v2/3, while low-sharpness Queries use **Block-wise 0-order Taylor Sparse Attention**, followed by merging the two-path online-softmax. The entire forward/backward pass is implemented as trainable kernels using Triton/TileLang.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["ICL Concatenated Sequence Q,K,V<br/>source + context"] --> B["Pooling Attention (Coarse Probe)<br/>Obtain S_coarse & block mask M_coarse"]
    B --> C["Context Pre-Selection<br/>Top-k selection of salient K/V in context sub-blocks → K_new, V_new"]
    C --> D["Grouped Computation via Query Sharpness<br/>Sort by M_i, shunt via Flat Ratio α_f"]
    D -->|High Sharpness Query (Precise)| E["FlashAttention v2/3<br/>Precise online-softmax"]
    D -->|Low Sharpness Query (Approx)| F["Block-wise 0-order Taylor Sparse Attention<br/>Representative values Kc, Vc replace blocks"]
    E --> G["Merge Two-path Online-softmax<br/>O_i = O_i / ℓ_i → Output"]
    F --> G
```

### Key Designs

**1. Context Pre-Selection: Pruning Redundant Context K/V by Saliency**

ICL concatenates source and context, quadrupling attention computation. However, observations show $Q^{src}(K^{src})^\top$ scores are significantly higher than $Q^{src}(K^{ctx})^\top$, especially in deeper layers. Most context tokens are redundant. By performing pooling attention to get coarse scores $S_\text{coarse}\in\mathbb{R}^{B\times H\times N_Q\times N_K}$ and slicing the source-Query vs. context-Key sub-block $S^\text{ctx}_\text{coarse}$, the most salient $\alpha_s\lceil L_{ctx}/b\rceil$ context blocks are selected via TopK on the Query-axis mean. This compresses K/V length and reduces complexity to $\mathcal{O}(N(L_{src}+\alpha_s L_{ctx})D)$. Pruning these tokens also removes noise from synthetic context, explaining why ISA can outperform full attention in training-free settings.

**2. Block-wise 0-order Taylor Sparse Attention: Approximating Blocks with Representative Values**

For each Query block $Q_i$ and Key/Value block pair $(K_j, V_j)$, the block mask $M_{ij}$ determines the path. If $M_{ij}=1$, standard online-softmax is used. If $M_{ij}=0$, pre-pooled $K_j^c, V_j^c$ are used as "representatives." The score becomes $S_{ij}^c=Q_i (K_j^c)^\top/\sqrt{D}$, and $P_{ij}^c = \exp(S_{ij}^c)$, with normalization terms scaled by block length: $\ell_i \mathrel{+}= P_{ij}^c \cdot L_K$ and $O_i \mathrel{+}= P_{ij}^c V_j^c \cdot L_K$. While 1st and 2nd-order Taylor expansions were tested, they were too computationally heavy for kernelization; 0-order expansion fits perfectly within the FlashAttention online-softmax framework and allows contiguous block access.

**3. Grouped Computation via Query Sharpness: Precision for "Sharp" Queries, Approximation for Others**

Previous sparse methods treated all Queries equally, leading to failure due to a few extremely sharp Queries. ISA calculates a nearly free proxy for each Query. Theorem 3.1 proves the 0-order Taylor error bound is determined by $M_i = \mathrm{Var}(\mathrm{softmax}(Q^c_i(K^c)^\top))$. By sorting $M_i$, the top $\alpha_f$ (Flat Ratio) high-sharpness Queries use FlashAttention to maintain precision, while the rest use 0-order Taylor sparsity. This "routing expensive computation with cheap statistics" allows for extreme sparsity (up to 93.75%) without sacrificing quality.

### Loss & Training
LIVEditor undergoes two-stage post-training on the Wan 2.2 high-noise branch. Stage I uses 1.7M mix-quality samples ($\eta = 1\mathrm{e}{-5}$, batch 16) to learn generalized editing semantics. Stage II uses a 0.089M high-quality subset ($\eta = 1\mathrm{e}{-6}$) to refine aesthetics and instruction alignment, utilizing DeepSpeed ZeRO-3 Offload. To mitigate position bias from length mismatch, decoupled RoPE is introduced (re-indexing source and context from 0). Default hyperparameters: $\alpha_s = 0.125, \alpha_{ns} = 0.0625, \alpha_f = 0.5$.

## Key Experimental Results

### Main Results

| Dataset | Method | Quality | Text Align | Temporal Cons. | Editing Quality | Pick(Frame) | Pick(Video) |
|--------|------|---------|------------|----------------|-----------------|-------------|-------------|
| EditVerseBench | EditVerse (Prev. SOTA) | 7.65 | 20.07 | 26.73 | 23.93 | 98.56 | 98.42 |
| EditVerseBench | LIVEditor (full-attn) | 7.62 | 19.98 | 27.13 | 23.80 | 99.24 | 99.19 |
| EditVerseBench | **LIVEditor (ISA)** | **7.89** | **20.09** | **27.19** | **24.55** | **99.32** | **99.22** |

ISA outperforms the equivalent full-attention model on nearly all metrics, suggesting that pre-selection acts as "denoising" by removing detrimental context tokens.

### Ablation Study

| Configuration | Quality | Text Align | Temporal Cons. | Editing Quality | SpeedUp vs FA3 |
|------|---------|-----------|-----------------|------------------|----------------|
| Radial Attention | 7.09 | 19.68 | 26.86 | 24.13 | 1.28× |
| Sparge Attention | 7.44 | 19.69 | 26.75 | 23.76 | 1.40× |
| STA | 4.45 | 15.76 | 13.02 | 4.82 | 2.09× |
| SWA | 5.95 | 18.48 | 20.06 | 16.74 | 1.37× |
| VSA | 3.60 | 16.85 | 17.30 | 9.88 | 1.38× |
| LIVEditor (full-attn) | 7.62 | 19.98 | 27.13 | 23.80 | 1.00× |
| **LIVEditor (ISA, training-free)** | **7.78** | **20.07** | **27.15** | 24.15 | **1.47×** |

| Stage Ablation | Quality | Text Align | Temporal Cons. | Editing Quality |
|----------|---------|-----------|-----------------|------------------|
| Stage I (1.7M mix) | 6.46 | 19.50 | 25.27 | 22.63 |
| Stage II (+0.089M HQ) | 7.89 | 20.09 | 27.19 | 24.55 |

### Key Findings
- **ISA surpasses full attention training-free**: Pushing quality to 7.78 without fine-tuning (vs. 7.62 for full-attn) confirms pre-selection as implicit "context denoising."
- **Flat Ratio is the most sensitive hyperparameter**: Performance drops sharply if $\alpha_f$ is too low; $\alpha_{ns}$ and $\alpha_s$ can be reduced significantly with minimal loss, allowing for ~94% sparsity.
- **Trainable kernels provide extra gains**: Training allows ISA to actively adapt rather than just approximate, minimizing the output gap between ISA and full attention.

## Highlights & Insights
- **Structured Sparsity for ICL**: Explicitly modeling "source vs context" as primary structure tells the model that source is the protagonist. This task-aware sparsity is more effective than generic windows.
- **Query Sharpness as Taylor Error Proxy**: Theorem 3.1 elegantly transforms engineering intuition into a cheap metric $M_i$. This concept of "routing via cheap statistics" is applicable to other mixed-precision inference scenarios.
- **Trainable Sparsity + Data-Driven**: Implementing ISA's backward pass as a Triton kernel enables end-to-end training, which is foundational for "approximation with gains."
- **Synthetic-as-context, real-as-source**: The data pipeline uses Gemini-synthesized images as context and real videos as source, preventing synthetic artifacts from polluting the primary output.

## Limitations & Future Work
- While attention latency is reduced, overall end-to-end acceleration might be less than 60% due to non-attention bottlenecks (VAE, text encoding, CFG).
- Hyperparameters $\alpha_f, \alpha_{ns}, \alpha_s$ require manual sweeping; an automatic scheduling mechanism is lacking for different resolutions/lengths.
- Performance on severe OOD content (e.g., non-natural synthesis, extreme long videos) is not fully verified.
- ISA currently targets the source/context distinction of ICL and lacks a generalized scheme for multi-context inputs.

## Related Work & Insights
- **vs Radial / Sparge / STA / SWA / VSA**: These are task-agnostic sparse methods based on distance or fixed masks. ISA's task-aware pre-selection is the key differentiator.
- **vs EditVerse / Ditto / InsV2V / Lucy Edit**: These use full attention for ICL editing. ISA serves as a drop-in replacement, proving that sparsity can coexist with (and even improve) state-of-the-art training paradigms.
- **vs FlashAttention v2/v3**: ISA complements rather than competes with FA; precise paths still utilize FA3, creating a relationship of "routing instead of replacing."

## Rating
- Novelty: ⭐⭐⭐⭐ Combines ICL structural priors with Taylor approximation theory for trainable sparse attention.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple benchmarks, numerous sparse baselines, sensitivity analysis, and both training-free/trainable modes.
- Writing Quality: ⭐⭐⭐⭐ Theorems and algorithms are clear; visualization of context pruning is intuitive.
- Value: ⭐⭐⭐⭐ Provides the first systematic sparse attention for ICL video editing, offering ~60% lossless speedup.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Unified In-Context Video Editing](../../ICLR2026/video_generation/unified_in-context_video_editing.md)
- [\[ICML 2026\] VEDA: Scalable Video Diffusion via Distilled Sparse Attention](veda_scalable_video_diffusion_via_distilled_sparse_attention.md)
- [\[ICML 2026\] DFSAttn: Dynamic Fine-Grained Sparse Attention for Efficient Video Generation](dfsattn_dynamic_fine-grained_sparse_attention_for_efficient_video_generation.md)
- [\[ICML 2026\] Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention](light_forcing_accelerating_autoregressive_video_diffusion_via_sparse_attention.md)
- [\[ICML 2026\] Attention Sparsity is Input-Stable: Training-Free Sparse Attention for Video Generation via Offline Sparsity Profiling and Online QK Co-Clustering](attention_sparsity_is_input-stable_training-free_sparse_attention_for_video_gene.md)

</div>

<!-- RELATED:END -->
