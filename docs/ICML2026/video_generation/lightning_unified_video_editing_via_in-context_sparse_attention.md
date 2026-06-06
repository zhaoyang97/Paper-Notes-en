---
title: >-
  [Paper Note] Lightning Unified Video Editing via In-Context Sparse Attention
description: >-
  [ICML 2026][Video Generation][In-Context Learning] Addressing the quadratic attention bottleneck in In-Context Learning (ICL) video editing…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "In-Context Learning"
  - "Sparse Attention"
  - "Video Editing"
  - "Taylor Approximation"
  - "Query Sharpness"
date: 2026-05-08
content_hash: c2857d8ea8e5705a
---

# Lightning Unified Video Editing via In-Context Sparse Attention

**Conference**: ICML 2026  
**arXiv**: [2605.04569](https://arxiv.org/abs/2605.04569)  
**Code**: Not yet released  
**Area**: Video Generation / Sparse Attention / Video Editing  
**Keywords**: In-Context Learning, Sparse Attention, Video Editing, Taylor Approximation, Query Sharpness

## TL;DR
Addressing the quadratic attention bottleneck in In-Context Learning (ICL) video editing, the authors design In-context Sparse Attention (ISA) based on two insights: "context tokens are less significant than source tokens" and "Query sharpness is proportional to Taylor approximation error." They train LIVEditor, which achieves a ~60% speedup while outperforming SOTA full-attention models across multiple benchmarks.

## Background & Motivation

**Background**: Video editing is transitioning from single-task expert models to the In-Context Learning (ICL) paradigm—concatenating context (reference frames/instructions) and source (video to edit) tokens into a unified DiT, allowing full attention to interact across long sequences. This approach is simple and scalable, becoming the mainstream for recent works like EditVerse and Ditto.

**Limitations of Prior Work**: Videos are inherently long sequences (5K~50K tokens), making the $\mathcal{O}(N^2)$ attention complexity an inference bottleneck. ICL further quadruples this by concatenating source and context of equal lengths. Existing sparse attention methods (Radial, Sparge, STA, SWA, VSA, etc.) are designed for general video generation and do not distinguish between source and context, failing to exploit the unique structure of ICL.

**Key Challenge**: Context tokens are numerous but contribute little to the final output, yet naive pruning might discard critical information. Furthermore, tolerance for approximation error varies significantly across Queries, but existing methods apply the same approximation to all Queries; "high-error Queries using approximation" becomes the root cause of global accuracy drops.

**Goal**: (i) Construct a "nearly lossless" sparse attention for ICL video editing; (ii) Support both end-to-end training and training-free inference replacement of full attention; (iii) Train a practical unified video editing model based on these designs.

**Key Insight**: Visualizing the attention score matrix in ICL reveals that source-source block scores are significantly higher than source-context scores. Theoretically, the author proves that the upper bound of 0th-order Taylor approximation error is determined by Query "sharpness" $M_i = \mathrm{Var}(\mathrm{softmax}(Q^c_i (K^c)^\top))$. Thus, "which tokens to keep" and "which Queries to compute precisely" are converted into computable proxies.

**Core Idea**: Use pre-selection to prune redundant context K/V, then use Query sharpness to shunt Queries to either full attention or 0th-order Taylor block-sparse attention, ensuring "precision where needed, approximation where possible."

## Method

### Overall Architecture
The input consists of concatenated long sequences $Q,K,V \in \mathbb{R}^{B\times H\times N\times D}$, where the first $L_{src}$ tokens come from the source and the subsequent $L_{ctx}$ from the context. The ISA forward pass consists of four steps: (1) Obtain coarse-grained scores $S_\text{coarse}$ and block masks $M_\text{coarse}$ via pooling attention; (2) Perform Top-k selection on context sub-blocks of $S_\text{coarse}$ to compress context K/V to $\alpha_s L_{ctx}$ length; (3) Evaluate the sharpness of each Query block using coarse-grained variance $M_i$, shunting Queries based on the Flat Ratio $\alpha_f$; (4) High-sharpness Queries use FlashAttention v2/3, while low-sharpness Queries use block-wise 0th-order Taylor sparse attention, merging the two paths via online-softmax. The entire forward/backward pass is implemented as trainable kernels using Triton/TileLang.

### Key Designs

1.  **Context Pre-Selection (Saliency-based K/V Pruning)**:
    - **Function**: Compresses K/V length from $N$ back to $L_{src} + \alpha_s L_{ctx}$ without losing critical context, reducing complexity from $\mathcal{O}(NSD)$ to $\mathcal{O}(N(L_{src}+\alpha_s L_{ctx})D)$.
    - **Mechanism**: Pooling attention yields $S_\text{coarse}\in\mathbb{R}^{B\times H\times N_Q\times N_K}$. Sub-blocks $S^\text{ctx}_\text{coarse}$ (source-Query vs. context-Key) are averaged along the Query axis, and TopK is used to select $\alpha_s\lceil L_{ctx}/b\rceil$ most significant context blocks, followed by gather + concat to reconstruct $K_\text{new}, V_\text{new}$.
    - **Design Motivation**: Attention distribution visualizations (Fig. 4-5) show that $Q^{src}(K^{src})^\top$ is much larger than $Q^{src}(K^{ctx})^\top$, especially in deeper layers. This indicates most context tokens are redundant. Pruning them saves computation and removes noise from synthetic contexts—explaining why ISA can outperform full attention in training-free scenarios.

2.  **Block-wise 0th-order Taylor Sparse Attention (Precision/Approximation Switching)**:
    - **Function**: For each Query block $Q_i$ and Key/Value block pair $(K_j, V_j)$, the mechanism decides between precision and Taylor approximation based on $M_{ij}$, reducing complexity from $\mathcal{O}(L_Q L_K D)$ to $\mathcal{O}(L_Q D)$.
    - **Mechanism**: When $M_{ij}=1$, standard online-softmax is executed: $S_{ij}=Q_i K_j^\top/\sqrt{D}$, $P_{ij}=\exp(S_{ij})$, $\ell_i \mathrel{+}= \mathrm{rowsum}(P_{ij})$, $O_i \mathrel{+}= P_{ij} V_j$. When $M_{ij}=0$, pooled $K_j^c, V_j^c$ are used as substitutes: $S_{ij}^c=Q_i (K_j^c)^\top/\sqrt{D}$, $P_{ij}^c = \exp(S_{ij}^c)$. This is treated as a "representative value" for the block: $\ell_i \mathrel{+}= P_{ij}^c \cdot L_K$, $O_i \mathrel{+}= P_{ij}^c V_j^c \cdot L_K$, with final normalization $O_i = O_i/\ell_i$.
    - **Design Motivation**: 1st/2nd-order Taylor expansions were tested but found difficult to kernelize on GPUs with high overhead. 0th-order expansion integrates with the FlashAttention online-softmax framework and allows contiguous block access, making it an engineering "sweet spot."

3.  **Grouped Computation based on Query Sharpness (High/Low Error Shunting)**:
    - **Function**: Splits Queries into two groups—high-error Queries use full attention for accuracy, while low-error Queries use 0th-order Taylor sparse attention to save compute, controlled by Flat Ratio $\alpha_f$.
    - **Mechanism**: Theorem 3.1 proves the 0th-order Taylor error bound is determined by $M_i = \mathrm{Var}(\mathrm{softmax}(Q^c_i(K^c)^\top))$ (inter-block variance) and $\|Q(K-K^c)^\top\|_\infty^2$. Since the latter is expensive to compute, $M_i$ is used as a nearly-free proxy from pooling scores. Queries are sorted by $M_i$; the top $\alpha_f$ ratio uses FlashAttention, and the rest use Taylor sparsity. The latter further uses $\alpha_{ns}$ to control "non-sparse" ratios within blocks, pushing overall sparsity to 93.75%.
    - **Design Motivation**: Prior sparse methods (STA, SWA, Sparge, etc.) treated all Queries equally, forcing approximation on "sharp" Queries and degrading quality. ISA's "dynamic routing" preserves exact computation for critical Queries, allowing high sparsity without sacrificing visual quality.

### Loss & Training
LIVEditor undergoes two-stage post-training on the Wan 2.2 high-noise branch: Stage I uses 1.7M mix-quality samples ($\eta = 1\mathrm{e}{-5}$, batch 16) to learn generalized editing semantics; Stage II uses a 0.089M high-quality subset ($\eta = 1\mathrm{e}{-6}$) to refine aesthetics and instruction alignment, using DeepSpeed ZeRO-3 Offload. To mitigate position bias from source/context length imbalance, decoupled RoPE is introduced (source and context re-indexed from 0). Default hyperparameters: $\alpha_s = 0.125, \alpha_{ns} = 0.0625, \alpha_f = 0.5$.

## Key Experimental Results

### Main Results

| Dataset | Method | Quality | Text Align | Temporal Cons. | Editing Quality | Pick(Frame) | Pick(Video) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EditVerseBench | EditVerse (Prev. SOTA) | 7.65 | 20.07 | 26.73 | 23.93 | 98.56 | 98.42 |
| EditVerseBench | LIVEditor (full-attn) | 7.62 | 19.98 | 27.13 | 23.80 | 99.24 | 99.19 |
| EditVerseBench | **LIVEditor (ISA)** | **7.89** | **20.09** | **27.19** | **24.55** | **99.32** | **99.22** |

ISA exceeds its full-attention counterpart on nearly all metrics, suggesting that pre-selection acts as "denoising" by pruning counterproductive context tokens.

### Ablation Study

| Configuration | Quality | Text Align | Temporal Cons. | Editing Quality | SpeedUp vs FA3 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Radial Attention | 7.09 | 19.68 | 26.86 | 24.13 | 1.28× |
| Sparge Attention | 7.44 | 19.69 | 26.75 | 23.76 | 1.40× |
| STA | 4.45 | 15.76 | 13.02 | 4.82 | 2.09× |
| SWA | 5.95 | 18.48 | 20.06 | 16.74 | 1.37× |
| VSA | 3.60 | 16.85 | 17.30 | 9.88 | 1.38× |
| LIVEditor (full-attn) | 7.62 | 19.98 | 27.13 | 23.80 | 1.00× |
| **LIVEditor (ISA, training-free)** | **7.78** | **20.07** | **27.15** | 24.15 | **1.47×** |

| Stage Ablation | Quality | Text Align | Temporal Cons. | Editing Quality |
| :--- | :--- | :--- | :--- | :--- |
| Stage I (1.7M mix) | 6.46 | 19.50 | 25.27 | 22.63 |
| Stage II (+0.089M HQ) | 7.89 | 20.09 | 27.19 | 24.55 |

### Key Findings
- **ISA training-free exceeds full attention**: Achieving a quality of 7.78 without fine-tuning (vs. 7.62 for full-attn) confirms that pre-selection is an implicit "context denoiser."
- **Flat Ratio is the most sensitive hyperparameter**: Any decrease in $\alpha_f$ leads to significant drops across all metrics; hence it is fixed at 0.5. Conversely, $\alpha_{ns}, \alpha_s$ can be pushed to 0.0625 and 0.125 with minimal loss, enabling ~94% sparsity via asymmetric sensitivity.
- **Trainable kernels provide extra gains**: Fig. 7 shows the training process compresses the gap between ISA and full attention outputs across almost all blocks, indicating that ISA "actively adapts" rather than just "approximating."

## Highlights & Insights
- **Structured Sparsity in ICL**: Explicitly modeling "source vs context" as first-order structural information tells the model "context is a reference, source is the protagonist." This task-aware sparsity is more effective than task-agnostic windows.
- **Query Sharpness = Taylor Error Proxy**: Theorem 3.1 transforms the engineering intuition of "when to approximate" into a computable metric $M_i$. This strategy of "using cheap statistics to route expensive computation" is elegant and transferable to other mixed-precision scenarios.
- **Trainable Sparsity + Data Driven**: Writing ISA's backward pass as a Triton kernel allows sparse attention to participate in end-to-end training, which is foundational to achieving "approximation + performance improvement."
- **Synthetic-as-context, Real-as-source**: The data pipeline design—placing Gemini-synthesized images on the context side and real videos on the source side—prevents synthetic artifacts from contaminating the main output.

## Limitations & Future Work
- The core focus is attention acceleration, but latency gains only address one ICL bottleneck. VAE decoding, text encoding, and CFG remain untouched; end-to-end acceleration might be lower than the ~60% seen in the attention module.
- Hyperparameters ($\alpha_f, \alpha_{ns}, \alpha_s$) rely on manual sweeps rather than automatic scheduling, which might require re-searching for different resolutions or longer sequences.
- Accuracy of 0th-order Taylor approximation in extreme OOD scenarios (e.g., non-natural synthesis, ultra-long videos) is not fully verified; the theoretical bound only guarantees "limited error" rather than "negligible error."
- ISA only distinguishes source/context in ICL scenarios; general multi-source extensions are not yet proposed.

## Related Work & Insights
- **vs Radial / Sparge / STA / SWA / VSA**: These methods are task-agnostic (based on distance or momentum) and do not distinguish source/context. ISA's task-aware pre-selection is the key differentiator.
- **vs EditVerse / Ditto / InsV2V / Lucy Edit**: These use full attention for ICL editing. This work acts as a "drop-in" replacement for the attention layer while pushing SOTA further through a 1.7M data pipeline and two-stage training.
- **vs FlashAttention v2/v3**: FlashAttention optimizes IO and exact softmax. ISA complements this: sharp Queries are handled by FA3, while low-sharpness Queries use custom sparse kernels—a "shunting rather than replacing" relationship.

## Rating
- Novelty: ⭐⭐⭐⭐ Combines ICL structural priors with Taylor approximation theory to provide a trainable sparse attention with clear theoretical and engineering foundations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple benchmarks, numerous sparse attention baselines, hyperparameter sensitivity, and both training-free/trainable modes.
- Writing Quality: ⭐⭐⭐⭐ Theorems and algorithms are clearly stated; visualizations in Fig. 4-5 are intuitive.
- Value: ⭐⭐⭐⭐ Provides the first systematic sparse attention solution for ICL video editing, offering ~60% lossless acceleration with immediate value for the long-video generation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] VEDA: Scalable Video Diffusion via Distilled Sparse Attention](veda_scalable_video_diffusion_via_distilled_sparse_attention.md)
- [\[ICML 2026\] DFSAttn: Dynamic Fine-Grained Sparse Attention for Efficient Video Generation](dfsattn_dynamic_fine-grained_sparse_attention_for_efficient_video_generation.md)
- [\[ICML 2026\] Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention](light_forcing_accelerating_autoregressive_video_diffusion_via_sparse_attention.md)
- [\[ICML 2026\] Attention Sparsity is Input-Stable: Training-Free Sparse Attention for Video Generation via Offline Sparsity Profiling and Online QK Co-Clustering](attention_sparsity_is_input-stable_training-free_sparse_attention_for_video_gene.md)
- [\[CVPR 2026\] VideoCoF: Unified Video Editing with Temporal Reasoner](../../CVPR2026/video_generation/videocof_unified_video_editing_with_temporal_reasoner.md)

</div>

<!-- RELATED:END -->
