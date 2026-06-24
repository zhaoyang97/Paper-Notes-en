---
title: >-
  [Paper Note] LongVPO: From Anchored Cues to Self-Reasoning for Long-Form Video Preference Optimization
description: >-
  [NeurIPS 2025][Video Understanding][Long video understanding] LongVPO proposes a two-stage DPO framework. Stage 1 constructs pseudo-long-video preference data by anchoring short clips and introduces an anchor-only reference model approximation to address context-length mismatch. Stage 2 performs self-training on real long videos via recursive captioning and multi-clip reasoning tasks. Using only 16K synthetic samples, the method surpasses long-video models trained with large-…
tags:
  - "NeurIPS 2025"
  - "Video Understanding"
  - "Long video understanding"
  - "DPO"
  - "vision-language models"
  - "preference optimization"
  - "short-to-long transfer"
date: 2026-05-08
content_hash: f4056a219b76a62a
---

# LongVPO: From Anchored Cues to Self-Reasoning for Long-Form Video Preference Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2602.02341](https://arxiv.org/abs/2602.02341)  
**Code**: [GitHub](https://github.com/MCG-NJU/LongVPO)  
**Area**: LLM Alignment
**Keywords**: Long video understanding, DPO, vision-language models, preference optimization, short-to-long transfer

## TL;DR

LongVPO proposes a two-stage DPO framework. Stage 1 constructs pseudo-long-video preference data by anchoring short clips and introduces an anchor-only reference model approximation to address context-length mismatch. Stage 2 performs self-training on real long videos via recursive captioning and multi-clip reasoning tasks. Using only 16K synthetic samples, the method surpasses long-video models trained with large-scale supervised data.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: Two core challenges for short-context VLMs on long-video tasks:

1. **Scarcity of long-video annotations**: High-quality long-video QA annotations are extremely costly and lack comprehensive coverage.
2. **Positional bias (Lost-in-the-middle)**: When short-context VLMs process long sequences via positional encoding extension, content at middle positions tends to be overlooked.

Existing DPO methods (e.g., LongVILA) assume the reference model itself supports long-context reasoning, a condition that short-context VLMs do not satisfy. Furthermore, relying on closed-source models to generate preference data introduces external bias.

## Method

### Overall Architecture

Two-stage progressive DPO training:
- **Stage 1**: Anchored-clip learning — short-video SFT data are concatenated to form pseudo-long videos; an anchor-only approximation is used to address reference model degradation.
- **Stage 2**: Long-video self-training alignment — preference data are constructed from real long videos via recursive captioning and multi-clip reasoning.

### Key Designs

1. **Short-to-Long Learning with Anchored Cues (Stage 1)**:
    - **Function**: Multiple short clips are concatenated into a pseudo-long video; preference triplets $(q_i, y_i^+, y_i^-)$ are generated around the anchor clip.
    - **Mechanism**: One anchor clip is selected to generate QA pairs as the preferred response; responses from non-anchor clips serve as dispreferred responses. Anchor position is randomized to mitigate positional bias.
    - **Design Motivation**: Simulates "needle-in-a-haystack" scenarios in long videos, training the model to locate the correct clip amid extensive distractors.
    - **Quality Filtering**: Scene similarity filtering (DINOv2 embeddings remove distractor clips too similar to the anchor) + question specificity filtering (LLM verifies that the question genuinely depends on unique anchor information).
    - **Anchor-only Reference Model Approximation**: $\pi_{ref}(y|x_i) \approx \pi_{ref}(y|x_{i,anchor})$, evaluating the reference model only on the anchor clip to avoid score degradation caused by context-length mismatch.

2. **Long-Video Preference Alignment via Self-Training (Stage 2)**:
    - **Function**: Preference data for multi-clip reasoning are constructed from real long videos for DPO training.
    - **Mechanism**: Recursive captioning → LLM generates cross-scene QA pairs → scene reference tracking → construction of dispreferred responses using partial or irrelevant context.
    - **Design Motivation**: Stage 1 uses concatenated clips that lack the narrative coherence of real long videos; Stage 2 supplements causal reasoning and event-chain understanding using authentic video content.
    - **Dispreferred Generation Strategies**: (a) Partial evidence reasoning — only a subset of relevant scenes is provided; (b) Irrelevant scene hallucination — only unrelated scenes are provided.

### Loss & Training

- Both stages employ a DPO loss with an auxiliary SFT loss: $\mathcal{L} = \mathcal{L}_{DPO} + \alpha \cdot \frac{-\log \pi_\theta(y^+|x)}{|y^+|}$
- Stage 1 DPO loss uses the anchor-only reference model approximation.
- Stage 2 fixes the reference model as the Stage 1 checkpoint.
- Total training data amounts to only ~16K samples.

## Key Experimental Results

### Main Results

| Model | Params | LVBench | LongVideoBench | MLVU M-Avg | Video-MME (w/o sub) | MVBench |
|-------|--------|---------|----------------|------------|---------------------|---------|
| LLaVA-Video-7B | 7B | - | 58.2 | 70.8 | 63.3 | 58.6 |
| Kangaroo-8B | 8B | - | 54.8 | 61.0 | 56.0 | 61.0 |
| LongVU-7B | 7B | - | - | 65.4 | 60.6 | 66.9 |
| **LongVPO-7B** | **7B** | **-** | **>58** | **>70** | **>63** | **competitive** |

LongVPO outperforms models trained with large-scale supervised data on multiple long-video benchmarks while maintaining competitive performance on the short-video benchmark MVBench.

### Ablation Study

- Stage 1 alone yields substantial improvements; Stage 2 further enhances multi-clip reasoning capability.
- The anchor-only approximation is critical for Stage 1 — without it, DPO training becomes unstable.
- Both scene similarity filtering and question specificity filtering are individually necessary.
- Position randomization effectively mitigates positional bias.

### Key Findings

- Short-context VLMs exhibit a pronounced "lost-in-the-middle" phenomenon, with significantly degraded retrieval of content at middle positions.
- Only 16K synthetic samples are sufficient to match or surpass models trained with large-scale annotated data.
- The anchor-only reference model approximation is a key innovation that resolves the fundamental barrier to short-to-long DPO transfer.

## Highlights & Insights

- **No long-video annotation required**: Long-video capability is achieved entirely from short-video data through careful data synthesis.
- **Exceptional data efficiency**: 16K samples vs. the million-scale data used by other approaches.
- **Principled progressive design**: Stage 1 establishes basic clip retrieval capability; Stage 2 cultivates cross-clip reasoning ability.
- The anchor-only approximation is an elegant engineering solution that simultaneously reduces computational cost and addresses distributional shift.

## Limitations & Future Work

- Stage 2 depends on the quality of recursive captions; captioning errors propagate into the preference data.
- Scenarios involving ultra-long videos exceeding one hour remain unexplored.
- Self-generated preferred responses are not necessarily perfect, potentially capping performance.
- The distributional gap between concatenated pseudo-long videos and real long videos cannot be fully bridged.

## Related Work & Insights

- Continues the trend of extending DPO from NLP to multimodal settings.
- Distinguishes itself from long-video methods such as LongVILA and VideoChat-Flash by requiring no long-video annotations.
- The anchor-only approximation idea is generalizable to other DPO scenarios that require handling context-length mismatch.

## Rating

⭐⭐⭐⭐ — Elegant method design with outstanding data efficiency; the two-stage progressive framework follows clear logic, establishing an effective paradigm for long-video understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Self-alignment of Large Video Language Models with Refined Regularized Preference Optimization](self-alignment_of_large_video_language_models_with_refined_regularized_preferenc.md)
- [\[NeurIPS 2025\] VGEnt: Graph-Based Retrieval-Reasoning-Augmented Generation for Long Video Understanding](vgent_graph-based_retrieval-reasoning-augmented_generation_for_long_video_unders.md)
- [\[NeurIPS 2025\] VideoLucy: Deep Memory Backtracking for Long Video Understanding](videolucy_deep_memory_backtracking_for_long_video_understanding.md)
- [\[NeurIPS 2025\] Unleashing Hour-Scale Video Training for Long Video-Language Understanding](unleashing_hour-scale_video_training_for_long_video-language_understanding.md)
- [\[AAAI 2026\] TSPO: Temporal Sampling Policy Optimization for Long-form Video Language Understanding](../../AAAI2026/video_understanding/tspo_temporal_sampling_policy_optimization_for_long-form_video_language_understa.md)

</div>

<!-- RELATED:END -->
