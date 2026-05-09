---
title: >-
  [Paper Note] Vamba: Understanding Hour-Long Videos with Hybrid Mamba-Transformers
description: >-
  [ICCV 2025][Video Understanding][Long video understanding] This paper proposes Vamba — a hybrid Mamba-Transformer large multimodal model (LMM) that encodes video tokens with linear complexity via Mamba-2 blocks and updates text tokens via cross-attention. Vamba processes up to 1024 frames on a single GPU and outperforms all efficient LMM methods on hour-level video understanding benchmarks.
tags:
  - ICCV 2025
  - Video Understanding
  - Long video understanding
  - Mamba
  - hybrid architecture
  - large multimodal models
  - computational efficiency
date: 2026-05-08
content_hash: 301c46c1e2ce5cd3
---

# Vamba: Understanding Hour-Long Videos with Hybrid Mamba-Transformers

**Conference**: ICCV 2025
**arXiv**: [2503.11579](https://arxiv.org/abs/2503.11579)
**Code**: [GitHub](https://tiger-ai-lab.github.io/Vamba/)
**Area**: Video Understanding
**Keywords**: Long video understanding, Mamba, hybrid architecture, large multimodal models, computational efficiency

## TL;DR

This paper proposes Vamba — a hybrid Mamba-Transformer large multimodal model (LMM) that encodes video tokens with linear complexity via Mamba-2 blocks and updates text tokens via cross-attention. Vamba processes up to 1024 frames on a single GPU and outperforms all efficient LMM methods on hour-level video understanding benchmarks.

## Background & Motivation

Transformer-based LMMs such as Qwen2-VL achieve strong video understanding performance, but face a fundamental efficiency bottleneck:

**Quadratic complexity**: Causal self-attention incurs $O(d(M+N)^2)$ compute and memory cost, where $M$ denotes the number of video tokens and $N$ the number of text tokens. For long videos, $M$ can reach hundreds of thousands or even millions.

**Frame count limitation**: Qwen2-VL-7B can process only 256 frames (360p) on a single GPU, which is far insufficient for hour-level video understanding.

**Limitations of existing compression methods**: Approaches such as Q-Former compression and adaptive token compression reduce token counts but cause information loss and still rely on quadratic-complexity attention.

**Core insight**: In video LMMs, the number of video tokens $M$ vastly exceeds the number of text tokens $N$ ($M \gg N$, typically $M > 100N$). Therefore, the quadratic bottleneck primarily stems from self-attention among video tokens. Replacing this with a linear-complexity module — while preserving text tokens' attention access to video tokens — can substantially reduce computational overhead.

## Method

### Overall Architecture

Vamba is built upon pretrained Qwen2-VL-7B, replacing the self-attention operation in each Transformer decoder layer with two more efficient components:

- **Cross-attention**: Text tokens serve as queries and video tokens as key-values → updates text tokens
- **Mamba-2 block**: Updates video tokens with linear complexity → replaces self-attention among video tokens

The overall prefill complexity is reduced from $O(d(M+N)^2)$ to $O(dMN + d^2M)$.

### Key Designs

1. **Text token update: Self-attention + Cross-attention**

   The original full self-attention is decomposed into two components:

   $o_{t_j} = (1-\alpha)\underbrace{(\sigma(\frac{q_{t_j}\mathbf{K}_v^\top}{\sqrt{d}})\mathbf{V}_v)\mathbf{W}_o^c}_{\text{Cross-Attention}} + \alpha\underbrace{(\sigma(\frac{q_{t_j}\mathbf{K}_{[t_1:t_j]}^\top}{\sqrt{d}})\mathbf{V}_{[t_1:t_j]})\mathbf{W}_o^s}_{\text{Self-Attention}}$

   where $\alpha \in [0,1]$ is a learnable scalar. Crucially, cross-attention ensures that each text token retains access to all video token information.

   **Weight initialization strategy**: The cross-attention projection matrices $\mathbf{W}_q^c, \mathbf{W}_k^c, \mathbf{W}_v^c, \mathbf{W}_o^c$ are initialized by copying weights from the self-attention layer of the same block. Experiments demonstrate that this strategy is critical (LVBench accuracy jumps from 23.7% to 34.2%).

2. **Video token update: Mamba-2 block**

   Self-attention among video tokens is replaced by a Mamba-2 state space model:

   $o_{v_i} = \text{Mamba}(\text{LN}(v_i), \mathbf{h}_{v_{i-1}}, \bar{\mathbf{A}}, \bar{\mathbf{B}}, \mathbf{C})$

   Mamba-2 adopts a scalar-times-identity simplification for the $\mathbf{A}$ matrix, supports multi-head SSM and larger state dimensions (64 vs. 16 in Mamba), and trains faster. Complexity is reduced from $O(dM^2)$ to $O(d^2M)$.

3. **Two-stage training**

   - **Pretraining**: Pretrained weights are frozen; only the newly introduced cross-attention and Mamba layers are trained, using approximately 3 million image caption samples to recover visual understanding capability.
   - **Instruction tuning**: Full fine-tuning on approximately 7 million image and video instruction samples to enhance instruction-following ability.

### Loss & Training

- Pretraining stage: Standard language modeling loss $\mathcal{L}_{\text{LM}} = -\frac{1}{T}\sum_{t=1}^T \log p(x_t|x_{<t})$
- A distillation loss $\mathcal{L}_{\text{Distill}} = D_{KL}(\mathcal{P}_\Theta || \mathcal{P}_{\Theta'})$ (extracting top-100 logits from a teacher model) was explored, but experiments show that all settings with $\lambda > 0$ degrade performance; only the language modeling loss is ultimately used.
- Instruction tuning stage: Language modeling loss only.

## Key Experimental Results

### Main Results

**Hour-level video understanding**

| Model | Scale | LVBench | HourVideo-dev | HourEval |
|-------|-------|---------|---------------|----------|
| Qwen2-VL | 7B | 42.0 | 33.8 | 53.0 |
| LongVU | 7B | 37.8 | 30.8 | 46.8 |
| Video-XL | 7B | 36.8 | 33.0 | 47.1 |
| LongLLaVA | 9B | 31.2 | 27.7 | 39.1 |
| **Vamba** | **10B** | **42.1** | **33.6** | **50.7** |

Vamba outperforms all efficient LMMs on LVBench by **4.3%**, and even surpasses the baseline Qwen2-VL-7B.

**Medium-long and short video benchmarks**

| Model | Video-MME (w/o sub) | MLVU | MVBench | NExT-QA |
|-------|---------------------|------|---------|---------|
| LongVU | 55.3 | 65.4 | 66.9 | 78.0 |
| Video-XL | 55.5 | 64.9 | 55.3 | 77.5 |
| **Vamba** | **57.8** | **65.9** | **60.4** | **78.1** |

### Ablation Study

| Model ID | Cross-attn init from SA? | Mamba block type | LVBench | Video-MME | MVBench |
|----------|--------------------------|------------------|---------|-----------|---------|
| A | ✗ | None | 23.7 | 47.6 | 40.9 |
| B | ✓ | None | 34.2 | 51.7 | 51.8 |
| C | ✓ | Mamba | 34.2 | 53.4 | 53.5 |
| D | ✓ | Mamba-2 | **35.3** | **54.1** | **53.5** |

Distillation loss ablation (effect of $\lambda$ on G-VEval score):

| $\lambda$ | 0 | 0.001 | 0.01 | 0.5 | 1 | 2 |
|-----------|---|-------|------|-----|---|---|
| G-VEval | **82.19** | 81.05 | 80.68 | 73.69 | 63.65 | 47.61 |

### Key Findings

- **Cross-attention weight initialization is the decisive factor**: Copying weights from self-attention raises LVBench from 23.7% to 34.2% (+10.5%), as the initialized cross-attention more closely approximates the original causal self-attention, easing adaptation.
- **Mamba-2 outperforms Mamba**: Despite a more constrained $\mathbf{A}$ matrix structure, its 64-dimensional state (vs. 16) yields superior performance.
- **Distillation loss is ineffective**: Contrary to findings in prior work such as CEPE, adding teacher distillation loss consistently degrades performance.
- **Training efficiency**: Vamba can be trained on 8 × A800 GPUs, compared to 64 GPUs for LongVU and 24 GPUs for LongLLaVA.
- **Memory efficiency**: Training memory is reduced by over 50% when processing 512 frames, training speed per step nearly doubles, and single-GPU inference supports up to 1024 frames — four times the capacity of Qwen2-VL.

## Highlights & Insights

- **Orthogonal to token compression**: Rather than reducing token counts, Vamba modifies the architecture for processing tokens, avoiding information loss caused by compression.
- The paradigm of **adapting a pretrained LMM into a hybrid architecture** warrants attention: freeze original weights → train only new layers (cross-attention + Mamba) → full fine-tuning, with manageable training cost.
- The initialization ablation suggests that "architectural replacement + weight inheritance" is a key technique for integrating efficient modules into pretrained models.
- The success of Mamba-2 validates the potential of linear-complexity models for visual sequence modeling.

## Limitations & Future Work

- Approximately 3B additional parameters (cross-attention + Mamba layers) are introduced, increasing total parameter count from 7B to 10B.
- Hardware-level optimization for Mamba remains less mature than for Transformers; theoretical speedups have not fully translated into practical gains.
- Vamba is not combined with token compression methods — the authors explicitly note in their conclusion that the two directions are orthogonal and could be jointly applied in future work.
- Due to computational resource constraints, the visual encoder is partially frozen during the instruction tuning stage.

## Related Work & Insights

- LongVU (adaptive compression) and Video-XL (Visual Summarization Token) represent the token compression line of research.
- Methods such as Flamingo and mPlug-Owl3 use cross-attention but generally underperform LLaVA-style models — Vamba attributes this to the fact that video tokens are not updated in those designs.
- Mamba/Mamba-2 has proven effective in language modeling; Vamba is the first to successfully apply it within a video LMM.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The hybrid Mamba-Transformer design for video LMMs is well-motivated and clearly conceived; the initialization strategy demonstrates genuine insight.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive ablations covering benchmarks from hour-level to short video, with detailed efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Well-structured with complete mathematical derivations and fair comparisons against baselines.
- **Value**: ⭐⭐⭐⭐ Provides a novel architectural solution to the efficiency problem in long video LMMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] VideoMiner: Iteratively Grounding Key Frames of Hour-Long Videos via Tree-based Group Relative Policy Optimization](videominer_iteratively_grounding_key_frames_of_hour-long_videos_via_tree-based_g.md)
- [\[NeurIPS 2025\] Unleashing Hour-Scale Video Training for Long Video-Language Understanding](../../NeurIPS2025/video_understanding/unleashing_hour-scale_video_training_for_long_video-language_understanding.md)
- [\[ICCV 2025\] VideoLLaMB: Long Streaming Video Understanding with Recurrent Memory Bridges](videollamb_long_streaming_video_understanding_with_recurrent_memory_bridges.md)
- [\[ICCV 2025\] OVG-HQ: Online Video Grounding with Hybrid-modal Queries](ovg-hq_online_video_grounding_with_hybrid-modal_queries.md)
- [\[AAAI 2026\] APVR: Hour-Level Long Video Understanding with Adaptive Pivot Visual Information Retrieval](../../AAAI2026/video_understanding/apvr_hour-level_long_video_understanding_with_adaptive_pivot.md)

</div>

<!-- RELATED:END -->
