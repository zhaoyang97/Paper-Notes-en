---
title: >-
  [Paper Note] TransMamba: A Sequence-Level Hybrid Transformer-Mamba Language Model
description: >-
  [AAAI 2026][LLM/NLP][Transformer] This paper proposes TransMamba, a sequence-level Transformer-Mamba hybrid architecture that dynamically switches between Attention and SSM computation at different token positions via shared QKV/CBx parameters and a Memory Converter, achieving efficiency advantages for both short and long sequences.
tags:
  - AAAI 2026
  - LLM/NLP
  - Transformer
  - Mamba
  - SSM
  - Hybrid Architecture
  - Sequence Modeling
date: 2026-05-08
content_hash: 311df0d7d4f64e0f
---

# TransMamba: A Sequence-Level Hybrid Transformer-Mamba Language Model

**Conference**: AAAI 2026
**arXiv**: [2503.24067](https://arxiv.org/abs/2503.24067)
**Code**: N/A
**Area**: LLM/NLP
**Keywords**: Transformer, Mamba, SSM, Hybrid Architecture, Sequence Modeling

## TL;DR

This paper proposes TransMamba, a sequence-level Transformer-Mamba hybrid architecture that dynamically switches between Attention and SSM computation at different token positions via shared QKV/CBx parameters and a Memory Converter, achieving efficiency advantages for both short and long sequences.

## Background & Motivation

**Background**: Transformer ($O(T^2)$ complexity) remains the dominant architecture for LLMs. Mamba (SSM, $O(T)$ linear complexity) is more efficient on long sequences but exhibits instability in in-context learning and multi-task generalization. Existing hybrid approaches (Jamba, Zamba, etc.) adopt layer-level interleaving (fixed ratios of Transformer and Mamba layers), but suffer from structural rigidity — they must adhere to specific layer ordering and ratio rules.

**Limitations of Prior Work**: (a) Transformer trains faster on short contexts while Mamba is more efficient on long contexts — yet layer-level mixing cannot exploit the respective efficiency advantages of both within the same sequence; (b) layer-level mixing ratios are fixed (e.g., 4:1), and deviating from the prescribed rules degrades performance; (c) Mamba2 reveals a mathematical duality between Attention and SSM, and Wang et al. demonstrate via distillation that QKV and CBx parameters are mutually transferable — suggesting that a more principled unification of the two mechanisms is possible.

**Key Challenge**: A flexible framework is needed that can adaptively apply Attention or SSM at different positions within the same sequence without information loss during transitions.

**Key Insight**: Exploiting the parameter correspondence between Attention and SSM (Q↔C, K↔B, V↔x) to enable a single set of parameters to support both computation modes.

**Core Idea**: Shared QKV/CBx parameters + lossless Memory Converter + TransPoint scheduling = flexible sequence-level switching between Attention and SSM.

## Method

### Overall Architecture

TransMamba is a stacked-layer decoder-only autoregressive model. Each layer contains the full Mamba parameter set (C/B/x/A/Δ), with QKV and CBx parameters shared based on the Attention-SSM parameter correspondence (Q↔C, K↔B, V↔x). Tokens before the TransPoint are processed via Attention; tokens after the TransPoint are processed via SSM.

### Key Designs

1. **Shared Parameter Mapping (QKV↔CBx)**:

    - **Function**: A single parameter set supports both Attention and SSM computation modes.
    - **Mechanism**: Prefix tokens ($h_s = h[:TransPoint]$) compute Q=δ(h_s W_C), K=δ(h_s W_B), V=δ(h_s W_x) via shared parameters and produce output $y_s = \text{softmax}(QK^T) \cdot V$ through Attention. Subsequent tokens ($h_l = h[TransPoint:]$) use the same parameters to compute C/B/x, together with Δ and A parameters, to produce $y_l$ via the SSM mechanism.
    - **Design Motivation**: The dual form $(L \circ QK^T)V$ vs. $(A^{\times} \circ CB^T)X$ revealed by Mamba2 suggests that the core weights of both mechanisms are unifiable.

2. **Memory Converter (Lossless Information Transfer)**:

    - **Function**: Losslessly converts Attention K/V at the TransPoint into the initial SSM hidden state $h_0$.
    - **Mechanism**: $h_0 = \text{MemoryConverter}(K, V)$. By unrolling the SSM hidden state recurrence into matrix form $h = (A^{\times} \circ B^T)X$, it is theoretically shown that the Attention K/V can perfectly preserve the sequence state information required by SSM.
    - **Design Motivation**: Without this conversion, the SSM component would lose all contextual information from prefix tokens upon switching — the Memory Converter is the critical enabler of the entire framework.

3. **TransPoint Scheduling Strategy**:

    - **Function**: Determines at which token position each layer switches from Attention to SSM.
    - **Mechanism**: Each layer is assigned a single TransPoint. The training FLOPs are $O(P^2N + (T-P)N^2)$ (where $P$ is the TransPoint position), forming a quadratic function of $P$ with an optimal solution. Different TransPoint configurations across layers are also explored.
    - **Design Motivation**: A $P$ that is too small results in insufficient Attention coverage, while a $P$ that is too large dilutes the efficiency advantage of SSM. The optimal TransPoint depends on the ratio between sequence length and model dimension.

### Loss & Training

Standard cross-entropy language modeling loss plus a reconstruction loss for the Memory Converter. Training FLOPs are $O(P^2 N + (T-P)N^2)$ ($P$ denotes the Attention prefix length), which is superior to the $O(T^2 N)$ cost of a pure Transformer.

## Key Experimental Results

### Main Results (400M model, general tasks)

| Model | ARC-E | ARC-C | CoQA | OBQA | PIQA | BoolQ |
|-------|-------|-------|------|------|------|-------|
| Transformer-400M | 60.57 | 58.72 | 5.07 | 42.4 | 52.75 | - |
| Mamba-400M | lower | lower | higher | lower | lower | - |
| **TransMamba-400M** | **best** | **best** | **best** | **best** | **best** | **best** |

- Under an equivalent training budget of 83B tokens, TransMamba outperforms both pure Transformer and pure Mamba baselines on most general benchmarks.
- Notable improvements are observed on PhoneBook (long-range dependency test) and LongBench-v2 (long-context understanding).

### Efficiency Results

| Configuration | Training FLOPs/layer | Notes |
|---------------|----------------------|-------|
| Pure Transformer | $O(T^2 N)$ | Fast on short sequences but bottlenecked on long sequences |
| Pure Mamba | $O(TN^2)$ | Efficient on long sequences but inferior to Transformer on short sequences |
| **TransMamba (optimal P)** | $O(P^2 N + (T-P)N^2)$ | Combines advantages of both |

- Under the setting N=1536, T=8192, the optimal TransPoint is P≈2048, yielding peak training efficiency.

### Ablation Study

| Configuration | Performance |
|---------------|-------------|
| Full TransMamba | Best — optimal efficiency and performance |
| w/o Memory Converter | Significant degradation — SSM loses prefix context |
| Fixed TransPoint (uniform across layers) | Suboptimal — abrupt switching incurs performance loss |
| Log-distributed TransPoint (cycled every 8 layers) | Best — progressive switching balances diversity and efficiency |
| Pure Attention (P=T) | Strong on short sequences but inefficient on long sequences |
| Pure SSM (P=0) | Strong on long sequences but insufficient expressiveness on short sequences |

### Key Findings

- The Memory Converter is a necessary component — removing it causes the SSM portion to collapse in performance, confirming the importance of lossless conversion.
- Sequence-level mixing is more flexible and efficient than layer-level mixing — short sequences benefit from Attention's global interaction, while long sequences benefit from SSM's linear complexity.
- The optimal TransPoint position is related to the ratio between model dimension $N$ and sequence length $T$ — SSM efficiency dominates when $T > N$.
- Assigning different TransPoints to different layers outperforms a uniform assignment — the log-distributed scheme (0/128/256/512/1024/2048/4096/8192 cycled every 8 layers) yields the best results.
- At inference time, a TransPoint strategy different from that used in training can be adopted — the most efficient structure is used during training, while a task-appropriate structure can be applied at inference.

## Highlights & Insights

- **Validates the Transformer-Mamba unification at a deeper level**: Beyond the mathematical duality, the paper empirically demonstrates that a single parameter set can operate under both computation modes — a step further than the theoretical correspondence established by Mamba2.
- **Theoretical guarantee of the Memory Converter** constitutes the key technical contribution: it proves that lossless conversion from K/V to SSM hidden states is feasible, enabling computation mode switching within a sequence.
- The shared parameter design allows the model to gracefully degrade to different architectures (pure Transformer, pure Mamba, or hybrid) depending on sequence length, providing exceptional flexibility.

## Limitations & Future Work

- Each layer is assigned only a single TransPoint; more complex multi-TransPoint structures may yield further improvements.
- TransPoint scheduling is currently determined via prior knowledge or search; adaptive learning of TransPoints is worth exploring.
- The Memory Converter introduces additional computation — while theoretically lossless, a quantitative analysis of practical precision impact is still needed.
- Validation is limited to language modeling; applicability to other modalities (vision, multimodal) remains unknown.

## Related Work & Insights

- **vs. Jamba/Zamba and other layer-level hybrids**: Layer-level mixing operates under fixed and inflexible ratios; TransMamba adaptively switches at the sequence level.
- **vs. Mamba2**: Mamba2 reveals the dual form but remains an independent architecture; TransMamba realizes a unified framework with actual parameter sharing.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First framework to unify Transformer and Mamba at the sequence level; the contributions of parameter sharing and lossless conversion are theoretically significant.
- **Experimental Thoroughness**: ⭐⭐⭐ — Language modeling validation is thorough, but downstream task evaluation is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — Architecture design is clearly presented with complete mathematical derivations.
- **Value**: ⭐⭐⭐⭐ — Opens a new paradigm for Transformer-SSM hybrid architectures.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Rectification Reimagined: A Unified Mamba Model for Image Correction and Rectangling with Prompts](rectification_reimagined_a_unified_mamba_model_for_image_cor.md)
- [\[AAAI 2026\] Uncertainty Under the Curve: A Sequence-Level Entropy Area Metric for Reasoning LLMs](uncertainty_under_the_curve_a_sequence-level_entropy_area_metric_for_reasoning_l.md)
- [\[ACL 2026\] FastDiSS: Few-step Match Many-step Diffusion Language Model on Sequence-to-Sequence Generation](../../ACL2026/llm_nlp/fastdiss_few-step_match_many-step_diffusion_language_model_on_sequence-to-sequen.md)
- [\[AAAI 2026\] Smart: A GNN-LLM Hybrid Surrogate Model for Dragonfly System Application Runtime Prediction](smart_a_surrogate_model_for_predicting_application_runtime_in_dragonfly_systems.md)
- [\[AAAI 2026\] An Invariant Latent Space Perspective on Language Model Inversion](an_invariant_latent_space_perspective_on_language_model_inve.md)

<!-- RELATED:END -->
