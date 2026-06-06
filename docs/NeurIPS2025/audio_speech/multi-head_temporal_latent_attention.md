---
title: >-
  [Paper Note] Multi-head Temporal Latent Attention
description: >-
  [NeurIPS 2025][Audio & Speech][KV cache compression] MTLA extends MLA's low-rank latent compression along the feature dimension by introducing a hyper-network that dynamically merges temporally adjacent KV vectors…
tags:
  - "NeurIPS 2025"
  - "Audio & Speech"
  - "KV cache compression"
  - "temporal dimension compression"
  - "MLA"
  - "hyper-network"
  - "stride-aware causal mask"
date: 2026-05-08
content_hash: 79e2e1735aa1baaa
---

# Multi-head Temporal Latent Attention

**Conference**: NeurIPS 2025
**arXiv**: [2505.13544](https://arxiv.org/abs/2505.13544)  
**Code**: [https://github.com/D-Keqi/mtla](https://github.com/D-Keqi/mtla)  
**Area**: Efficient Attention / Speech Processing
**Keywords**: KV cache compression, temporal dimension compression, MLA, hyper-network, stride-aware causal mask

## TL;DR
MTLA extends MLA's low-rank latent compression along the feature dimension by introducing a hyper-network that dynamically merges temporally adjacent KV vectors, achieving dual-axis compression of the KV cache across both feature and temporal dimensions. Combined with a stride-aware causal mask to ensure training–inference consistency, MTLA achieves 4.29× speedup and 6.58× memory reduction on speech translation and related tasks, with quality on par with or slightly exceeding standard MHA.

## Background & Motivation

**Background**: During autoregressive inference, the KV cache of Transformer models grows linearly with sequence length, becoming a critical bottleneck for long-sequence tasks—particularly in speech and audio. Existing approaches such as MQA (shared KV heads), GQA (grouped sharing), and MLA (low-rank latent projection) all compress only along the feature dimension, leaving the sequence length dimension untouched.

**Limitations of Prior Work**: MQA and GQA reduce the number of heads at the cost of representational capacity; MLA compresses KV dimensionality effectively via low-rank projection, yet the KV cache still grows linearly with sequence length $T$. Speech tasks involve particularly long sequences (thousands of frames), making temporal compression an unexplored direction.

**Key Challenge**: Adjacent frames in speech and audio signals are highly redundant, yet standard attention stores KV vectors frame-by-frame, leading to significant waste. Existing temporal compression methods (e.g., SnapKV pruning) discard information and degrade quality.

**Goal**: Compress the KV cache along the temporal dimension without quality loss, realizing dual-axis compression, while resolving the technical challenge of behavioral inconsistency between parallel training and incremental inference.

**Key Insight**: Adjacent KV vectors can be merged via weighted combination, but the weights should be dynamically generated based on content and position—rather than fixed pooling—since information density varies across positions.

**Core Idea**: MLA feature-dimension compression + hyper-network dynamic merging for temporal compression + stride-aware causal mask for training–inference consistency = dual-axis KV cache compression.

## Method

### Overall Architecture
Input sequence $\mathbf{X} \in \mathbb{R}^{T \times d}$ → standard projection to Query $\mathbf{Q}$ → low-rank projection to latent vector $\mathbf{C} \in \mathbb{R}^{T \times r}$ (MLA component) → hyper-network generates temporal merging weights → every $s$ adjacent latent vectors are merged into $\hat{\mathbf{C}} \in \mathbb{R}^{\lceil T/s \rceil \times r}$ → $\hat{\mathbf{C}}$ is used directly in attention computation (absorbing $W_K, W_V$ via matrix associativity) → stride-aware causal mask controls the visible range.

### Key Designs

1. **Hyper-network Temporal Merging**:

    - **Function**: Every $s$ adjacent low-rank latent vectors $\mathbf{c}_i$ are merged into a single vector $\hat{\mathbf{c}}_j$ via learnable weights $w_i$.
    - **Mechanism**: $w_i = \text{Sigmoid}(\text{Linear}(\mathbf{c}_i) \cdot \text{Linear}(\mathbf{pe}_j))$, where $\mathbf{c}_i$ carries content features, $\mathbf{pe}_j$ is a positional encoding, and $\cdot$ denotes element-wise multiplication. For $s=2$: $\hat{\mathbf{c}}_1 = w_1 \mathbf{c}_1 + w_2 \mathbf{c}_2$.
    - **Design Motivation**: Since input length is variable, fixed parameters cannot handle all cases; a hyper-network is needed to dynamically generate merging weights from content. Sigmoid ensures non-negative weights (analogous to soft averaging), while positional information allows different merging behavior at different positions. At inference time, the KV cache is reduced from $T$ to $\lceil T/s \rceil$ vectors; at $s=2$ the cache is halved.

2. **Stride-aware Causal Mask**:

    - **Function**: A custom attention mask is designed so that the information visible during parallel training exactly matches that available during incremental inference.
    - **Mechanism**: During training, an augmented sequence $\hat{\mathbf{C}}'$ is constructed in which each group of $s$ positions corresponds to intermediate and final merged states. The standard causal mask is inapplicable—a query at position $m$ may only attend to column $n$ where $n = m$ or ($n < m$ and $n \bmod s = 0$).
    - **Design Motivation**: During inference, step $i$ may attend to a temporarily incomplete vector $\hat{\mathbf{c}}_j'$ (e.g., at $s=2$, steps 1 and 3 have not yet received the next frame). Naïve pre-downsampling at training time would introduce behavioral inconsistency. The stride-aware mask precisely emulates inference behavior, guaranteeing training–inference equivalence.

3. **Decoupled RoPE Temporal Compression**:

    - **Function**: Adapts RoPE positional encoding to the temporally compressed KV cache.
    - **Mechanism**: The RoPE-encoded key $\mathbf{K}^R$ is also compressed along the temporal axis to $\hat{\mathbf{K}}^R$, and the most recent RoPE key cache is updated accordingly at inference time. The final attention is computed as:
      $$\mathbf{Y} = \text{softmax}\left(\frac{\mathbf{X}(W_Q W_K^\top)\hat{\mathbf{C}}^\top + \mathbf{Q}^R(\hat{\mathbf{K}}^R)^\top}{\sqrt{d_h}}\right)\hat{\mathbf{C}}(W_V W_O)$$
    - **Design Motivation**: Decoupled RoPE is a key component of MLA, and MTLA must remain compatible with it. During training, the uncompressed $\mathbf{K}^R$ can substitute for $\hat{\mathbf{K}}^R$ directly, simplifying the implementation.

### Loss & Training
- MTLA replaces the attention module without modifying the task-level loss function.
- Hyperparameters are shared with MLA: $r = 4d_h$, $d_h^R = d_h/2$, default $s=2$.
- Cache size analysis: at $s=2$, the average KV cache elements per token are $9d_h l / (2s) = 2.25 d_h l$, close to MQA's $2d_h l$.

## Key Experimental Results

### Main Results

| Task | Model | Quality | Inference Time (s) | Speedup | GPU Memory (MiB) | Mem. Reduction |
|------|-------|---------|-------------------|---------|-----------------|----------------|
| Speech Translation (En-De) | MHA | 23.18 BLEU | 281.3 | 1.00× | 18646 | 1.00× |
| | MLA | 22.97 BLEU | 97.0 | 2.90× | 5065 | 3.68× |
| | **MTLA** | **23.28 BLEU** | **65.6** | **4.29×** | **2835** | **6.58×** |
| Text Summarization (XSum) | MHA | 23.33 RL | 352.3 | 1.00× | 16141 | 1.00× |
| | **MTLA** | 23.60 RL | 105.2 | 3.35× | 2198 | 7.34× |
| Speech Recognition (AMI) | MHA | 12.98 WER | 269.4 | 1.00× | 17509 | 1.00× |
| | **MTLA** | **12.66 WER** | **71.8** | **3.75×** | **2364** | **7.41×** |
| Spoken Language Understanding (SLURP) | MHA | 86.83 Acc | 133.1 | 1.00× | 14370 | 1.00× |
| | **MTLA** | 86.80 Acc | 52.7 | 2.53× | 2051 | 7.01× |

### Ablation Study

| Method | BLEU | Inference Time (s) | Speedup | GPU Memory (MiB) | Mem. Reduction |
|--------|------|--------------------|---------|-----------------|----------------|
| MHA | 23.18 | 281.3 | 1.00× | 18646 | 1.00× |
| MQA | 22.70 | 168.1 | 1.67× | 3074 | 6.07× |
| GQA (g=2) | 22.75 | 190.6 | 1.48× | 5313 | 3.51× |
| MLA + SnapKV | 21.76 | 80.8 | 3.48× | 4222 | 4.42× |
| Mamba-2 | 18.62 | 157.5 | 1.78× | 5676 | 3.29× |
| MTLA (s=2) | **23.28** | 65.6 | 4.29× | 2835 | 6.58× |
| MTLA (s=3) | 23.25 | 52.7 | 5.34× | 2251 | 8.28× |
| MTLA (s=4) | 23.05 | 48.7 | 5.78× | 1921 | 9.71× |

### Key Findings
- MTLA achieves a slightly higher BLEU than MHA on speech translation (23.28 vs. 23.18), suggesting that compressing redundant temporal information may have a mild regularization effect.
- Compared to MQA: MTLA achieves comparable memory usage but is 2.56× faster, inheriting MLA's matrix absorption advantage that avoids explicit K/V computation.
- MLA + SnapKV pruning incurs a notable quality drop (21.76 BLEU), whereas MTLA's soft merging retains all information and yields superior quality.
- At $s=4$, MTLA still significantly outperforms MQA in translation quality ($p < 0.05$) while being faster and more memory-efficient.
- Results hold with FlashAttention-2: MTLA still achieves 3.99× speedup and 7.34× memory reduction.

## Highlights & Insights
- **Temporal dimension compression is an entirely new direction**: All prior KV compression methods (MQA/GQA/MLA) operate exclusively on the feature dimension. MTLA is the first to demonstrate that the temporal dimension can be effectively compressed, opening an orthogonal axis for efficiency gains.
- **Hyper-network-generated merging weights are an elegant solution**: They address the challenge of dynamic merging over variable-length sequences. Compared to fixed pooling or heuristic pruning (SnapKV), soft merging better preserves information and maintains higher quality.
- **The stride-aware causal mask resolves the core technical challenge**: It aligns the behavior of parallel training and incremental inference under temporal compression—a design principle transferable to other attention variants that compress sequence length.

## Limitations & Future Work
- All experiments involve training medium-scale models from scratch; effectiveness on large models (e.g., 7B+ LLMs) remains unverified.
- Quality degrades as the compression ratio $s$ increases (BLEU drops by 0.23 at $s=4$), limiting applicability in extreme compression scenarios.
- The hyper-network introduces additional parameters and computation, offering limited advantage for short sequences.
- Evaluation is restricted to decoder-only architectures; encoder-decoder architectures have not been tested.

## Related Work & Insights
- **vs. MLA (DeepSeek-V2)**: MLA compresses only the feature dimension ($d \to r$); MTLA further adds temporal compression ($T \to T/s$) and is a direct extension of MLA. The two compress along orthogonal axes, and their combination yields substantial gains.
- **vs. MQA/GQA**: MQA reduces the number of heads without reducing sequence length. At comparable KV cache sizes, MTLA is 2.56× faster, as it reduces the per-step attention computation load.
- **vs. SnapKV pruning**: SnapKV compresses by discarding less important tokens, incurring significant information loss (BLEU drops by 1.42); MTLA's soft merging preserves information and achieves better quality.
- **vs. Mamba-2**: Linear-complexity models have advantages on extremely long sequences, but incur substantial quality degradation (18.62 vs. 23.28 BLEU); MTLA retains the full modeling capacity of quadratic attention.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First proposal of temporal-dimension KV compression; a genuinely new direction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Four tasks with diverse baselines, though model scale is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Method is clearly described; the training–inference consistency analysis is thorough.
- **Value**: ⭐⭐⭐⭐ Introduces a new dimension for KV cache compression with potentially significant impact on LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Latent Space Factorization in LoRA](latent_space_factorization_in_lora.md)
- [\[ICCV 2025\] Latent Swap Joint Diffusion for 2D Long-Form Latent Generation](../../ICCV2025/audio_speech/latent_swap_joint_diffusion_for_2d_long-form_latent_generation.md)
- [\[NeurIPS 2025\] Efficient Speech Language Modeling via Energy Distance in Continuous Latent Space](efficient_speech_language_modeling_via_energy_distance_in_continuous_latent_spac.md)
- [\[AAAI 2026\] Multi-granularity Interactive Attention Framework for Residual Hierarchical Pronunciation Assessment](../../AAAI2026/audio_speech/multi-granularity_interactive_attention_framework_for_residual_hierarchical_pron.md)
- [\[ICLR 2026\] Latent Speech-Text Transformer](../../ICLR2026/audio_speech/latent_speech_text_transformer.md)

</div>

<!-- RELATED:END -->
