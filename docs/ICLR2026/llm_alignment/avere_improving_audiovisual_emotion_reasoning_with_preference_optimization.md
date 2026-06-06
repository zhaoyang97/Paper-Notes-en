---
title: >-
  [Paper Note] AVERE: Improving Audiovisual Emotion Reasoning with Preference Optimization
description: >-
  [ICLR 2026][LLM Alignment][Multimodal Emotion Understanding] To address spurious associations and hallucinations in multimodal large language models (MLLMs) for emotion reasoning…
tags:
  - "ICLR 2026"
  - "LLM Alignment"
  - "Multimodal Emotion Understanding"
  - "Preference Optimization"
  - "DPO"
  - "Hallucination Mitigation"
  - "Audiovisual Reasoning"
date: 2026-05-08
content_hash: 3c40c43ded489867
---

# AVERE: Improving Audiovisual Emotion Reasoning with Preference Optimization

**Conference**: ICLR 2026
**arXiv**: [2602.07054](https://arxiv.org/abs/2602.07054)  
**Code**: [https://github.com/ihp-lab/AVERE](https://github.com/ihp-lab/AVERE)  
**Area**: Alignment & RLHF
**Keywords**: Multimodal Emotion Understanding, Preference Optimization, DPO, Hallucination Mitigation, Audiovisual Reasoning

## TL;DR
To address spurious associations and hallucinations in multimodal large language models (MLLMs) for emotion reasoning, this work proposes the EmoReAlM evaluation benchmark and the AVEm-DPO preference optimization method. By constructing targeted preference pairs and incorporating text-prior regularization, the approach achieves 6–19% relative zero-shot performance gains on DFEW, RAVDESS, and EMER.

## Background & Motivation
Emotion understanding is a core capability for building socially intelligent agents. While MLLMs have made notable progress on emotion recognition tasks, two critical challenges remain:

**Challenge 1: Spurious Associations.** Models frequently associate emotions with irrelevant audiovisual cues—for example, linking a yellow turtleneck in the scene to "happiness" rather than attending to facial expressions. This constitutes a reasoning-level failure.

**Challenge 2: Hallucinations.** The textual priors of the language model backbone drive models to fabricate audiovisual evidence, such as claiming a video shows "clenched fists" to support an "anger" prediction when no such action is present. This constitutes a perception-level failure.

Existing multimodal preference optimization methods (e.g., Vista-DPO) target general-purpose video understanding and are not designed for the specific challenges of emotion understanding. Moreover, no dedicated evaluation tool exists to systematically quantify spurious associations and hallucinations in MLLMs under affective scenarios.

**Core Idea**: Jointly develop an evaluation benchmark (EmoReAlM) and an alignment method (AVEm-DPO). The approach introduces preference pair construction strategies targeting spurious associations and hallucinations, and adds text-prior regularization to fundamentally align the model's audiovisual perception with its emotion reasoning capability.

## Method
The work comprises two main contributions: the EmoReAlM evaluation benchmark and the AVEm-DPO preference optimization method.

### Overall Architecture
AVEm-DPO builds upon the Direct Preference Optimization (DPO) framework, with emotion-specific innovations in preference pair construction and loss function design. The overall pipeline proceeds as: (1) analyze MLLM failure modes on emotion tasks → (2) design EmoReAlM to quantify these issues → (3) construct targeted preference data → (4) apply text-prior regularization for alignment training.

### Key Designs
1. **EmoReAlM Benchmark**: A benchmark specifically designed to evaluate MLLM emotion reasoning, comprising four task types: (a) *Reasoning Basic*—assesses whether the model makes emotion judgments based on correct audiovisual cues; (b) *Stress Test*—examines whether the model hallucinates non-existent cues; (c) *Modality Agreement*—determines whether visual and auditory cues are genuinely consistent; (d) *Hallucination-Free Detection*—verifies whether the model correctly identifies cues that actually exist. The design motivation is that existing benchmarks cannot distinguish between "correct answer with incorrect reasoning."

2. **Preference Pair Construction Strategy**: AVEm-DPO constructs two types of preference pairs. The first targets responses: responses exhibiting spurious associations or hallucinations are treated as rejected, while correct responses serve as chosen. The second targets inputs: textually guided construction of audiovisual input pairs with varying cues teaches the model to distinguish which audiovisual signals are genuinely emotion-relevant. This two-level preference construction is the core innovation of the method.

3. **Text-Prior Regularization**: An additional regularization term penalizes the model's over-reliance on textual priors. When the model generates emotion-related descriptions without corresponding audiovisual evidence, the regularization applies a penalty. This directly addresses the root cause of hallucinations—textual biases learned by the language model backbone.

### Loss & Training
The standard DPO loss is augmented with a text-prior penalty term. Training follows a zero-shot evaluation protocol, assessed on emotion datasets including DFEW, RAVDESS, and EMER.

In the preference pair construction process, the baseline model first generates multiple responses; EmoReAlM's evaluation dimensions are then used to automatically identify responses exhibiting spurious associations or hallucinations, which are designated as rejected. For input-level preference pairs, mismatched input pairs are constructed by substituting the audio or visual modality within videos.

The regularization term takes the form of an additional penalty on visual or audio descriptions that can be generated solely from textual priors, encouraging the model to ground its judgments in genuine multimodal inputs.

Two backbone networks are used to validate generalizability: the paper's base model (based on the VITA-1.5 architecture) and EmotionLLaMA (an emotion-specialized fine-tuned model).

## Key Experimental Results

### Main Results

| Dataset | Metric | AVEm-DPO (Ours) | Naive-DPO | Vista-DPO | Base | Gain (vs Base) |
|---------|--------|-----------------|-----------|-----------|------|----------------|
| DFEW | WAR | 58.54 | 55.67 | 56.42 | 56.78 | +3.1% |
| DFEW | UAR | 64.24 | 59.90 | 62.33 | 60.14 | +6.8% |
| RAVDESS | WAR | 58.66 | 53.63 | 56.94 | 53.59 | +9.5% |
| EmoReAlM | Avg | 83.3 | 68.1 | 76.9 | 65.1 | +28.0% |

### Ablation Study

| Configuration | EmoReAlM Avg | Notes |
|---------------|-------------|-------|
| Base | 65.1 | No preference optimization |
| + Naive-DPO | 68.1 | Standard DPO; limited improvement |
| + Vista-DPO | 76.9 | General-purpose video DPO |
| + AVEm-DPO | 83.3 | Emotion-specific design; best performance |

### Key Findings
- AVEm-DPO surpasses closed-source Gemini 2.5 Pro on EmoReAlM (70.3 → 83.3), demonstrating the effectiveness of targeted alignment.
- The method transfers effectively to the EmotionLLaMA backbone, confirming generalizability.
- The largest gains appear on the Stress Test (hallucination detection) subtask (51.4 → 68.9), validating the effect of text-prior regularization.
- Modality Agreement improves from 66.4 to 94.6, indicating that the model learns to genuinely exploit cross-modal information.

## Highlights & Insights
- This is the first preference optimization method specifically designed for multimodal emotion reasoning, with a highly precise problem framing.
- The EmoReAlM benchmark is elegantly designed; its four task types comprehensively dissect the emotion reasoning weaknesses of MLLMs.
- The two-level preference pair construction (response-level + input-level) is a generalizable paradigm applicable to other multimodal tasks requiring fine-grained alignment.
- Text-prior regularization offers a lightweight yet effective solution to mitigating hallucinations in MLLMs.
- Leaderboard results show that AVEm-DPO enables an open-source model to outperform closed-source Gemini 2.5 Pro on emotion understanding.
- Qualitative examples clearly illustrate how AVEm-DPO helps the model focus on genuine facial expressions and vocal tone rather than fabricating non-existent visual cues.

## Limitations & Future Work
- Code and model weights have been promised for public release but are still being prepared (a HuggingFace checkpoint is available: chaubeyG/AVERE-7B).
- The evaluation set scale and emotion category coverage could be further expanded; the current focus is primarily on basic emotions.
- Evaluation is conducted only under zero-shot settings; few-shot and fine-tuning configurations warrant exploration.
- The strength of text-prior regularization requires manual tuning; an adaptive strategy would be desirable.
- Benchmark audio cue evaluation relies on the model's audio understanding capability, which is itself a challenging problem.
- The automation level of preference pair construction could be further improved; the current pipeline still involves some manual design.

## Related Work & Insights
- **vs. Vista-DPO**: Vista-DPO is a general-purpose video DPO method not designed for affective scenarios; AVEm-DPO constructs preference pairs specifically targeting spurious associations and hallucinations.
- **vs. EmotionLLaMA**: EmotionLLaMA is fine-tuned on emotion data but still suffers from hallucinations; AVEm-DPO further aligns it, making the two approaches complementary.
- **vs. Qwen 2.5 Omni**: This closed-source model is powerful for general audiovisual understanding but underperforms AVEm-DPO on emotion-specific tasks.
- **vs. Naive-DPO**: Applying generic DPO directly yields limited gains (+3%), demonstrating that preference pair quality matters more than the algorithm itself.

## Rating
- Novelty: ⭐⭐⭐⭐ Dual contributions of benchmark and alignment method; novel preference pair construction strategy.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset, multi-backbone validation with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear and the logical flow is coherent.
- Value: ⭐⭐⭐⭐ Directly advances the field of multimodal affective AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Mitigating Mismatch within Reference-based Preference Optimization](mitigating_mismatch_within_reference-based_preference_optimization.md)
- [\[NeurIPS 2025\] LongVPO: From Anchored Cues to Self-Reasoning for Long-Form Video Preference Optimization](../../NeurIPS2025/llm_alignment/longvpo_from_anchored_cues_to_selfreasoning_for_longform_vid.md)
- [\[ICLR 2026\] Dual-IPO: Dual-Iterative Preference Optimization for Text-to-Video Generation](dual-ipo_dual-iterative_preference_optimization_for_text-to-video_generation.md)
- [\[ICLR 2026\] Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning](slow-fast_policy_optimization_reposition-before-update_for_llm_reasoning.md)
- [\[ICLR 2026\] Alignment through Meta-Weighted Online Sampling: Bridging the Gap between Data Generation and Preference Optimization](alignment_through_meta-weighted_online_sampling_bridging_the_gap_between_data_ge.md)

</div>

<!-- RELATED:END -->
