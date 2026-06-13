---
title: >-
  [Paper Note] How Should Video LLMs Output Time? An Analysis of Efficient Temporal Grounding Paradigms
description: >-
  [CVPR 2026][Video Understanding][Video Temporal Grounding] This paper compares three mainstream temporal output paradigms for video temporal grounding (VTG) — text-number generation, temporal token generation…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "Video Temporal Grounding"
  - "Multimodal Large Language Models"
  - "Temporal Output Paradigm"
  - "Efficiency Analysis"
  - "Compact Models"
date: 2026-05-08
content_hash: 2d8832543013cc95
---

# How Should Video LLMs Output Time? An Analysis of Efficient Temporal Grounding Paradigms

**Conference**: CVPR 2026
**arXiv**: [2604.08966](https://arxiv.org/abs/2604.08966)  
**Code**: [https://tg-paradigms.github.io/](https://tg-paradigms.github.io/)  
**Area**: Video Understanding
**Keywords**: Video Temporal Grounding, Multimodal Large Language Models, Temporal Output Paradigm, Efficiency Analysis, Compact Models

## TL;DR

This paper compares three mainstream temporal output paradigms for video temporal grounding (VTG) — text-number generation, temporal token generation, and continuous time decoding — within a unified framework, finding that the continuous distribution paradigm consistently achieves the best efficiency–accuracy Pareto frontier.

## Background & Motivation

**Background**: Video temporal grounding (VTG) is a core task bridging language queries and temporal video segments. Multimodal large language models (MLLMs) have become the dominant backbone for this task, yet existing methods diverge substantially in their temporal output design — some directly generate text-form timestamps, some introduce dedicated temporal tokens, and others predict temporal distributions via continuous decoding.

**Limitations of Prior Work**: Each paradigm adopts its own backbone, dataset, and training protocol, making it impossible to attribute performance differences to the output design itself. Furthermore, as VTG systems are increasingly deployed on resource-constrained edge devices, a systematic efficiency–accuracy trade-off analysis is lacking.

**Key Challenge**: The impact of output paradigm selection on grounding accuracy and computational cost remains unclear, particularly for compact models (0.5B–8B).

**Goal**: To fairly compare the accuracy and efficiency of three paradigms under identical backbone, data, and training protocols.

**Key Insight**: SmolVLM2 (0.5B/2.2B), FastVLM (1.5B), and Molmo2 (4B/8B) are selected as compact backbones, ensuring that the output paradigm is the sole varying factor.

**Core Idea**: The continuous distribution paradigm achieves the optimal efficiency–accuracy trade-off on the Pareto frontier, with minimal latency overhead and robust grounding accuracy.

## Method

### Overall Architecture

The paper implements three representative paradigms: TRACE-style temporal token generation (Gen), DisTime-style continuous distribution decoding (Cont), and VTimeLLM-style text-number generation (Text), evaluated under a unified setting with 1.2M training samples across three benchmarks.

### Key Designs

1. **Text-Number Generation Paradigm (Text)**:

    - Function: Generates temporal boundaries as plain text numbers, directly reusing the LLM's native vocabulary.
    - Mechanism: Formats target timestamps into natural language templates (e.g., "from 52.0 to 63.0 seconds") and applies standard next-token prediction loss $\mathcal{L}_{text} = -\sum_j \log P(w_j | w_{<j}, I, F)$.
    - Design Motivation: Requires no architectural modifications, but temporal semantics become entangled with general-purpose number tokens, potentially limiting accuracy.

2. **Temporal Token Generation Paradigm (Gen)**:

    - Function: Introduces dedicated temporal tokens to create an independent temporal representation space.
    - Mechanism: Adopts TRACE's causal event modeling framework, where each event $e_k=(t_k, s_k, c_k)$ comprises a timestamp, saliency score, and description. A separate tokenizer with 13 character-level tokens and task-specific cross-entropy loss are employed.
    - Design Motivation: Explicitly decouples temporal coordinates from natural language, preserving the inherent structure of video events.

3. **Continuous Time Decoding Paradigm (Cont)**:

    - Function: Models temporal grounding as probabilistic distribution estimation.
    - Mechanism: Introduces a learnable ⟨TIME_STAMP⟩ token whose hidden state is decoded into a temporal distribution via a lightweight MLP. The continuous time space is discretized into $reg_{max}+1$ bins, and the final temporal prediction is computed as a weighted expectation: $\hat{t}_s = \sum_i e_{st}^{(i)} \cdot a_i$.
    - Design Motivation: Naturally models prediction uncertainty and mitigates ambiguity in subjective boundary annotations, with minimal parameter overhead.

### Loss & Training

All paradigms follow the same LoRA fine-tuning protocol (r=32) and identical 1.2M training data. The Gen paradigm uses task-specific cross-entropy loss; the Cont paradigm uses 1D-IoU regression loss combined with distribution focal loss; the Text paradigm uses standard language modeling loss.

## Key Experimental Results

### Main Results

| Benchmark / Paradigm | Metric | Text | Gen | Cont |
|---|---|---|---|---|
| Charades-STA (SmolVLM2-2.2B) | R1@0.5 | Moderate | High | Highest |
| QVHighlights (SmolVLM2-2.2B) | R1@0.5 | Moderate | High | Highest |
| YouCook2 (SmolVLM2-2.2B) | CIDEr | Highest | High | — |

### Ablation Study

| Configuration | Inference Latency | Parameter Overhead | Notes |
|---|---|---|---|
| Text paradigm | High (autoregressive) | None | Multi-step generation increases latency |
| Gen paradigm | Moderate (structured decoding) | Medium | Additional encoder-decoder pair |
| Cont paradigm | Lowest | Minimal (3-layer MLP) | Single-step decoding, non-autoregressive |

### Key Findings

- The continuous distribution paradigm achieves the optimal accuracy–efficiency trade-off across most backbone and task combinations.
- The Text paradigm incurs substantially higher inference latency than the other paradigms due to autoregressive generation.
- The Gen paradigm holds a unique advantage on tasks requiring simultaneous prediction of timestamps and saliency scores (e.g., mAP on QVHighlights).
- The influence of output paradigm selection is amplified on compact models, underscoring the importance of paradigm choice in resource-constrained deployment scenarios.

## Highlights & Insights

- **First fair cross-paradigm comparison**: By rigorously controlling all variables except the output paradigm, this work is the first to isolate the causal effect of output design on VTG performance.
- **Amplification effect on compact models**: Architectural choices have a more pronounced impact on smaller models, providing critical design guidance for edge deployment.
- **Efficiency advantage of the continuous distribution paradigm**: Single-step decoding eliminates the cumulative latency of autoregressive generation, parameter overhead is limited to a 3-layer MLP, and boundary ambiguity is handled naturally.

## Limitations & Future Work

- Evaluation is limited to compact models in the 0.5B–8B range; large-scale backbones beyond 7B are not included.
- Training data formatting inevitably involves paradigm-specific adaptations, which may introduce subtle biases.
- The possibility of combining or hybridizing paradigms remains unexplored.

## Related Work & Insights

- **vs. TRACE**: This work reproduces TRACE's causal event modeling within a unified framework, confirming that its structured decoding offers advantages on certain tasks but at the cost of reduced efficiency.
- **vs. DisTime**: The dual advantages of the distribution decoding paradigm in both efficiency and accuracy are validated across multiple backbones.

## Rating

- Novelty: ⭐⭐⭐ Primarily an empirical study; methodological designs build upon existing work.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five backbones × three paradigms × three benchmarks — highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and rigorous experimental design.
- Value: ⭐⭐⭐⭐ Provides objective empirical guidance for VTG system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SlotVTG: Object-Centric Adapter for Generalizable Video Temporal Grounding](slotvtg_object-centric_adapter_for_generalizable_video_temporal_grounding.md)
- [\[CVPR 2026\] HieraMamba: Video Temporal Grounding via Hierarchical Anchor-Mamba Pooling](hieramamba_video_temporal_grounding_via_hierarchical_anchor-mamba_pooling.md)
- [\[CVPR 2026\] CVA: Context-aware Video-text Alignment for Video Temporal Grounding](cva_context-aware_video-text_alignment_for_video_temporal_grounding.md)
- [\[ICCV 2025\] Sparse-Dense Side-Tuner for Efficient Video Temporal Grounding](../../ICCV2025/video_understanding/sparse-dense_side-tuner_for_efficient_video_temporal_grounding.md)
- [\[CVPR 2026\] Cluster-Wise Spatio-Temporal Masking for Efficient Video-Language Pretraining](cluster-wise_spatio-temporal_masking_for_efficient_video-language_pretraining.md)

</div>

<!-- RELATED:END -->
