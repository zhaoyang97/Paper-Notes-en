---
title: >-
  [Paper Note] Lightning Unified Video Editing via In-Context Sparse Attention
description: >-
  [ICML 2026][Video Generation][In-Context Learning] Addressing the quadratic attention bottleneck in In-Context Learning (ICL) video editing, the authors design In-context Sparse Attention (ISA) based on two insights: "context token saliency is lower than source tokens" and "Query sharpness is proportional to Taylor approximation error." They train LIVEditor, which achi
tags:
  - ICML 2026
  - Video Generation
  - In-Context Learning
  - Sparse Attention
  - Video Editing
  - Query Sharpness
date: 2026-05-08
content_hash: dbcf2ec55d7d8c49
---
# Lightning Unified Video Editing via In-Context Sparse Attention

**Conference**: ICML 2026  
**arXiv**: [2605.04569](https://arxiv.org/abs/2605.04569)  
**Code**: Not yet public  
**Area**: Video Generation / Sparse Attention / Video Editing  
**Keywords**: In-Context Learning, Sparse Attention, Video Editing, Taylor Approximation, Query Sharpness

## TL;DR
Addressing the quadratic attention bottleneck in In-Context Learning (ICL) video editing, the authors design In-context Sparse Attention (ISA) based on two insights: "context token saliency is lower than source tokens" and "Query sharpness is proportional to Taylor approximation error." They train LIVEditor, which achieves a ~60% speedup while surpassing SOTA full-attention models on multiple benchmarks.

## Background & Motivation

**Background**: Video editing is shifting from task-specific expert models to the In-Context Learning (ICL) paradigm—concatenating context (reference frames/editing instructions) and source (video to be edited) tokens into a unified DiT, allowing full attention to interact freely over long sequences. This approach is simple and scalable, becoming mainstream in recent works like EditVerse and Ditto.

**Limitations of Prior Work**: Videos are inherently long sequences; 5K~50K tokens make the $\mathcal{O}(N^2)$ complexity of attention an inference bottleneck. ICL further quadruples the computation by concatenating source and context of equal lengths. Existing sparse attention methods (Radial, Sparge, STA, SWA, VSA, etc.) are designed for general video generation and do not distinguish between source and context, thus failing to exploit the specific structure of ICL.

**Key Challenge**: Context tokens are numerous but contribute relatively little to the final output, yet crude pruning risks losing the few truly critical context tokens. Furthermore, different Queries have vastly different tolerances for approximation error, but existing methods treat all Queries with the same approximation; "using approximation for high-error Queries" becomes the primary cause of overall accuracy degradation.

**Goal**: (i) Construct a "nearly lossless" sparse attention for ICL video editing; (ii) simultaneously support end-to-end learning during training and direct replacement of full attention during inference; (iii) train a practical unified video editing model based on this.

**Key Insight**: The authors plotted the attention score matrix in ICL and found that source-source block scores are significantly higher than source-context scores. They also theoretically proved that the error upper bound of the 0-th order Taylor approximation is determined by the "sharpness" of the Query $M_i = \mathrm{Var}(\mathrm{softmax}(Q^c_i (K^c)^\top))$. Thus, identifying "which tokens to keep" and "which Queries to compute exactly" is transformed into calculating proxy metrics.

**Core Idea**: Use pre-selection to prune redundant context K/V, then use Query sharpness to shunt Queries into either full attention or 0-th order Taylor block-sparse attention, ensuring "exact computation where necessary, and approximation where possible."

## Method

### Overall Architecture
The input is a concatenated long sequence $Q,K,V \in \mathbb{R}^{B\times H\times N\times D}$, where the first $L_{src}$ tokens are from the source and the subsequent $L_{ctx}$ are from the context. The forward pass of ISA consists of four steps: (1) Use pooling attention to obtain coarse-grained scores $S_\text{coarse}$ and block mask $M_\text{coarse}$, serving as a "cheap probe"; (2) Perform Top-k selection on the context sub-blocks of $S_\text{coarse}$ to compress context K/V to length $\alpha_s L_{ctx}$ (**Context Pre-Selection**); (3) Use coarse-grained variance $M_i$ to evaluate the sharpness of each Query block and shunt Queries based on the Flat Ratio $\alpha_f$ (**Sharpness-based Grouped Computation**); (4) High-sharpness Queries undergo exact FlashAttention v2/3, while low-sharpness Queries use **Block-wise 0-order Taylor Sparse Attention**, followed by merging outputs via online-softmax. The entire forward/backward pass is implemented as a trainable kernel using Triton/TileLang. The data flow is illustrated below:

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["ICL Concatenated Sequence Q,K,V<br/>Source + Context"] --> B["Pooling Attention (Probe)<br/>Get S_coarse and block mask M_coarse"]
    B --> C["Context Pre-Selection<br/>Top-k selection for context K/V → K_new,V_new"]
    C --> D["Sharpness-based Grouped Computation<br/>Sort by M_i, shunt via Flat Ratio α_f"]
    D -->|High Sharpness (Exact)| E["FlashAttention v2/3<br/>Precise online-softmax"]
    D -->|Low Sharpness (Approx)| F["Block-wise 0-order Taylor Sparse Attention<br/>Block representative values Kc,Vc"]
    E --> G["Merge online-softmax<br/>O_i = O_i / ℓ_i → Output"]
    F --> G
```

### Key Designs

**1. Context Pre-Selection: Pruning Redundant context K/V**

ICL doubles the sequence length, quadrupling attention computation; however, the authors observed that $Q^{src}(K^{src})^\top \gg Q^{src}(K^{ctx})^\top$, a phenomenon that becomes more pronounced in deeper layers. Most context tokens are "supporting characters." By slicing the source-Query vs. context-Key sub-block $S^\text{ctx}_\text{coarse}$ from the pooling scores, averaging along the Query axis, and selecting the Top-k most salient context blocks, the K/V length is compressed from $N$ to $L_{src} + \alpha_s L_{ctx}$. This reduces complexity from $\mathcal{O}(NSD)$ to $\mathcal{O}(N(L_{src}+\alpha_s L_{ctx})D)$ and acts as a denoising mechanism for synthetic context.

**2. Block-wise 0-order Taylor Sparse Attention: Approximating Blocks with Representative Values**

For each Query block $Q_i$ and Key/Value block pair $(K_j, V_j)$, the block mask $M_{ij}$ determines the path. If $M_{ij}=1$, standard online-softmax is used. If $M_{ij}=0$, pre-pooled representative values $K_j^c, V_j^c$ are used: $S_{ij}^c=Q_i (K_j^c)^\top/\sqrt{D}$, $P_{ij}^c = \exp(S_{ij}^c)$, with normalization constants adjusted by block length $\ell_i \mathrel{+}= P_{ij}^c \cdot L_K$ and $O_i \mathrel{+}= P_{ij}^c V_j^c \cdot L_K$. While higher-order Taylor expansions were tested, 0-order is an engineering "sweet spot" due to its compatibility with FlashAttention's framework and contiguous memory access.

**3. Sharpness-based Grouped Computation: Precision for "Sharp" Queries**

Previous sparse methods treated all Queries equally. ISA calculates a nearly free proxy: Theorem 3.1 proves the 0-order Taylor error upper bound is determined by $M_i = \mathrm{Var}(\mathrm{softmax}(Q^c_i(K^c)^\top))$. Sorting $M_i$ in descending order, the top $\alpha_f$ (Flat Ratio) proportion of high-sharpness Queries are routed to FlashAttention to preserve accuracy. Others use 0-order Taylor sparsity, with overall sparsity reaching up to 93.75%. This dynamic routing protects critical Queries while allowing aggressive pruning elsewhere.

### Loss & Training
LIVEditor undergoes two-stage post-training on a high-noise branch of Wan 2.2. Stage 1: 1.7M mix-quality samples ($\eta = 1\mathrm{e}{-5}$, batch 16) to learn generalized editing semantics. Stage 2: 0.089M high-quality subset ($\eta = 1\mathrm{e}{-6}$) to refine aesthetics and instruction alignment. Decoupled RoPE is introduced to mitigate position bias, re-indexing source and context from 0. Hyperparameters: $\alpha_s = 0.125, \alpha_{ns} = 0.0625, \alpha_f = 0.5$.

## Key Experimental Results

### Main Results

| Dataset | Method | Quality | Text Align | Temporal Cons. | Editing Quality | Pick(Frame) | Pick(Video) |
|--------|------|---------|------------|----------------|-----------------|-------------|-------------|
| EditVerseBench | EditVerse (Prev. SOTA) | 7.65 | 20.07 | 26.73 | 23.93 | 98.56 | 98.42 |
| EditVerseBench | LIVEditor (full-attn) | 7.62 | 19.98 | 27.13 | 23.80 | 99.24 | 99.19 |
| EditVerseBench | **LIVEditor (ISA)** | **7.89** | **20.09** | **27.19** | **24.55** | **99.32** | **99.22** |

ISA outperforms the full-attention counterpart across almost all metrics, suggesting that pre-selection serves as a "denoiser," removing distracting context tokens.

### Ablation Study

| Config | Quality | Text Align | Temporal Cons. | Editing Quality | SpeedUp vs FA3 |
|------|---------|-----------|-----------------|------------------|----------------|
| Radial Attention | 7.09 | 19.68 | 26.86 | 24.13 | 1.28× |
| Sparge Attention | 7.44 | 19.69 | 26.75 | 23.76 | 1.40× |
| STA | 4.45 | 15.76 | 13.02 | 4.82 | 2.09× |
| SWA | 5.95 | 18.48 | 20.06 | 16.74 | 1.37× |
| VSA | 3.60 | 16.85 | 17.30 | 9.88 | 1.38× |
| LIVEditor (full-attn) | 7.62 | 19.98 | 27.13 | 23.80 | 1.00× |
| **LIVEditor (ISA, training-free)** | **7.78** | **20.07** | **27.15** | 24.15 | **1.47×** |

### Key Findings
- **ISA exceeds full attention training-free**: Pushing quality to 7.78 without fine-tuning (vs. 7.62 for full-attn) confirms pre-selection as an implicit "context denoiser."
- **Flat Ratio is the most sensitive hyperparameter**: If $\alpha_f$ drops, all metrics decline significantly. In contrast, $\alpha_{ns}$ and $\alpha_s$ can be compressed to very low values nearly losslessly.
- **Trainable kernels provide extra gain**: Training reduces the output discrepancy between ISA and full attention across almost all blocks.

## Highlights & Insights
- **Structured Sparsity in ICL**: Explicitly modeling "source vs context" as structural information provides a task-aware prior that is more effective than generic sparse patterns.
- **Sharpness as Taylor Error Proxy**: Theorem 3.1 elegantly transforms an engineering intuition into a computable metric $M_i$. This strategy of "routing expensive computation via cheap statistics" is applicable to other mixed-precision scenarios.
- **Data Pipeline Strategy**: Using Gemini-synthesized images for context and real videos for source prevents synthetic artifacts from contaminating the main output.

## Limitations & Future Work
- Overall latency improvement only addresses the attention bottleneck; VAE decoding and text encoding remain untouched, meaning end-to-end acceleration may be less than the 60% gain in the attention module.
- Hyperparameters ($\alpha_f, \alpha_{ns}, \alpha_s$) require manual sweeping and lack automatic scheduling for varying resolutions or sequence lengths.
- The 0-order Taylor approximation's robustness on extreme OOD data or extremely long videos is not fully explored.

## Related Work & Insights
- **vs Radial / Sparge / STA / SWA / VSA**: These are task-agnostic (distance-based or fixed mask), whereas ISA's task-aware pre-selection is the key differentiator.
- **vs FlashAttention v2/v3**: ISA does not compete with FlashAttention but stacks with it; high-sharpness Queries are still processed by FA3, making it a "routing" rather than a "replacement" relationship.

## Rating
- Novelty: ⭐⭐⭐⭐ Combines ICL structural priors with Taylor approximation theory for a trainable sparse attention.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple benchmarks, numerous baselines, and both training-free/trainable modes.
- Writing Quality: ⭐⭐⭐⭐ Clear presentation of theorems and algorithms; visuals effectively explain context pruning.
- Value: ⭐⭐⭐⭐ Provides the first systematic sparse attention for ICL video editing with lossless ~60% acceleration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Efficient Long-Context Modeling in Diffusion Language Models via Block Approximate Sparse Attention](../../CVPR2026/video_generation/efficient_long-context_modeling_in_diffusion_language_models_via_block_approxima.md)
- [\[ICML 2026\] VEDA: Scalable Video Diffusion via Distilled Sparse Attention](veda_scalable_video_diffusion_via_distilled_sparse_attention.md)
- [\[ICML 2026\] Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention](light_forcing_accelerating_autoregressive_video_diffusion_via_sparse_attention.md)
- [\[ICML 2026\] DFSAttn: Dynamic Fine-Grained Sparse Attention for Efficient Video Generation](dfsattn_dynamic_fine-grained_sparse_attention_for_efficient_video_generation.md)
- [\[ICML 2026\] Attention Sparsity is Input-Stable: Training-Free Sparse Attention for Video Generation via Offline Sparsity Profiling and Online QK Co-Clustering](attention_sparsity_is_input-stable_training-free_sparse_attention_for_video_gene.md)

</div>

<!-- RELATED:END -->
