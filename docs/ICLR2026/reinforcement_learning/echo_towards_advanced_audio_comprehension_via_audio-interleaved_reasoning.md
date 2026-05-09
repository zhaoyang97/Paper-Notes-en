---
title: >-
  [Paper Note] Echo: Towards Advanced Audio Comprehension via Audio-Interleaved Reasoning
description: >-
  [ICLR 2026][Reinforcement Learning][Audio Understanding] This paper proposes a novel paradigm called audio-interleaved reasoning, which treats audio as an active component during inference rather than a static context, enabling LALMs to dynamically locate and re-listen to audio segments during the reasoning process. Through a two-stage SFT+RL training framework and a structured data generation pipeline, the authors build the Echo model, which surpasses GPT-4o and Gemini-2.0-Flash on both expert-level and general audio understanding benchmarks.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Audio Understanding
  - Large Audio-Language Models
  - Audio-Interleaved Reasoning
  - Chain-of-Thought
date: 2026-05-08
content_hash: afa5ccab6fde1cf5
---

# Echo: Towards Advanced Audio Comprehension via Audio-Interleaved Reasoning

**Conference**: ICLR 2026
**arXiv**: [2602.11909](https://arxiv.org/abs/2602.11909)
**Code**: [GitHub](https://github.com/wdqqdw/Echo)
**Area**: Reinforcement Learning
**Keywords**: Audio Understanding, Large Audio-Language Models, Audio-Interleaved Reasoning, Reinforcement Learning, Chain-of-Thought

## TL;DR

This paper proposes a novel paradigm called audio-interleaved reasoning, which treats audio as an active component during inference rather than a static context, enabling LALMs to dynamically locate and re-listen to audio segments during the reasoning process. Through a two-stage SFT+RL training framework and a structured data generation pipeline, the authors build the Echo model, which surpasses GPT-4o and Gemini-2.0-Flash on both expert-level and general audio understanding benchmarks.

## Background & Motivation

Large audio-language models (LALMs) perform well on basic audio tasks (speech recognition, sound classification, music analysis), but exhibit a significant performance gap on complex audio tasks requiring fine-grained interpretation and reasoning.

The existing reasoning paradigm — **audio-conditioned text reasoning** — suffers from a fundamental information bottleneck:

1. Audio is encoded once into contextual embeddings, after which reasoning proceeds entirely in the text modality.
2. Audio is a continuous signal carrying richer and more fine-grained information than text; a single encoding cannot retain all subtle details.
3. Empirical evidence: during inference, LALM attention to audio tokens drops rapidly to <5% after the first 25 steps.

**Human cognitive inspiration**: Human auditory processing involves cyclically re-listening to critical acoustic segments, driven by auditory working memory and top-down attentional control. This paper emulates this mechanism by enabling LALMs to actively re-listen to audio during reasoning.

Core contrast: shifting from "thinking *about* audio" to "**thinking *with* audio**" — analogous to the paradigm shift in visual reasoning from "thinking about images" to "thinking with images."

## Method

### Overall Architecture

A two-stage training framework:

1. **Stage 1 (SFT)**: Teaches the model to locate critical audio segments and generate audio-anchored reasoning.
2. **Stage 2 (RL)**: Activates audio-interleaved reasoning capability through reasoning format adaptation and reinforcement learning.

### Key Design 1: Audio-Anchored Reasoning (SFT Stage)

Initialized from Qwen2.5-Omni (7B), the base model tends toward pure text reasoning without actively referencing audio segments.

**SFT data format**: Each sample contains multimodal input $(A, q)$ (audio + question) and ground-truth $(c, a)$ (CoT + answer), where the CoT densely embeds `<seg>start, end</seg>` tag pairs referencing audio segments. Each reference is preceded by a calling rationale and followed by fine-grained analysis grounded in the segment.

**Training objective**: Standard cross-entropy $\mathcal{L}_\text{SFT}(\theta) = -\frac{1}{n}\sum_{t=1}^n \log \pi_\theta(y_{i,t}^*|x_i, y_{i,<t}^*)$

This yields a "cold-start model" capable of referencing specific time intervals during reasoning, though still limited to the text modality.

### Key Design 2: Audio-Interleaved Reasoning (RL Stage)

**Reasoning format adaptation**: The cold-start model's reasoning is extended from text-only to a truly multimodal process — whenever the model generates a `<seg>` tag pair, generation is paused, the corresponding segment $A_{s:e}$ is cropped from the original audio, its tokens are inserted into the reasoning sequence, and generation resumes from the augmented input. This loop continues until `<eos>` is generated.

**RL reward design**:

$$\mathcal{R}(\tau) = \mathcal{R}_\text{format}(\tau) + \mathcal{R}_\text{consist}(\tau) + \mathcal{R}_\text{acc}(\tau) + \mathcal{R}_\text{seg}(\tau)$$

| Reward Component | Score | Description |
|-----------------|-------|-------------|
| $\mathcal{R}_\text{format}$ | 0.5 | Correct use of enclosing tags |
| $\mathcal{R}_\text{consist}$ | −0.1/instance, max −0.5 | Penalizes semantic discontinuity after `</seg>` (e.g., capitalized letter or `<`) |
| $\mathcal{R}_\text{acc}$ | 0.5 | Answer matches ground truth |
| $\mathcal{R}_\text{seg}$ | 0.5 | Correct answer with at least one segment reference; 0 otherwise |

**Optimization algorithm**: GRPO (Group Relative Policy Optimization), sampling $G=8$ candidate responses, normalizing rewards to compute advantages, with PPO-style clipping and KL divergence constraint:

$$\mathcal{L}_\text{RL}(\theta) = -\frac{1}{G}\sum_{g=1}^G \frac{1}{|\tau_g|}\sum_{t=1}^{|\tau_g|} [\min(\rho_{g,t} A_g, \text{clip}(\rho_{g,t}, 1\pm\epsilon) A_g) - \beta D_\text{KL}(\pi_\theta||\pi_\text{ref})]$$

All inserted audio tokens are excluded from loss computation.

### Key Design 3: Data Generation Pipeline

Built upon audio datasets with temporal metadata, including AudioSet-Strong and MusicBench:

1. Qwen2.5-Omni converts audio into three types of textual descriptions (comprehensive description, speech transcription, musical elements).
2. Combined with temporal metadata, DeepSeek-R1 synthesizes QA-CoT triplets.
3. Two-stage quality filtering: high-quality QA+CoT → SFT dataset; high-quality QA only → RL dataset.

Output: EAQA-SFT (75.9k samples with CoT) + EAQA-RL (21.9k samples without CoT).

## Key Experimental Results

### Main Results: MMAR Expert-Level Audio Reasoning

| Model | Size | Sound | Music | Speech | Mixed Modal Avg | Overall Avg |
|-------|------|-------|-------|--------|----------------|-------------|
| Qwen2.5-Omni | 7B | 58.79 | 40.78 | 59.86 | ~58 | 57.33 |
| GPT-4o-Audio | — | 53.94 | 50.97 | 70.41 | ~65 | 64.09 |
| Gemini-2.0-Flash | — | 61.21 | 50.97 | 72.11 | ~70 | 67.90 |
| Audio-Thinker | 7B | 68.48 | 53.88 | 64.29 | ~70 | 67.25 |
| **Echo** | **7B** | 67.27 | **60.68** | **69.39** | ~71 | **69.99** |

Echo, as a 7B open-source model, surpasses GPT-4o-Audio (+5.9%) and Gemini-2.0-Flash (+2.1%).

### Main Results: MMAU General Audio Understanding

| Model | MMAU-mini Avg | MMAU Avg |
|-------|--------------|----------|
| Qwen2.5-Omni (7B) | 71.53 | 71.00 |
| Audio-Thinker (7B) | 78.00 | 75.39 |
| Gemini-2.5-Pro | 71.60 | 69.36 |
| **Echo (7B)** | **80.41** | **76.61** |

Echo exceeds Audio-Thinker by +2.41% on MMAU-mini and +1.22% on MMAU.

### Ablation Study: Training Framework (MMAR Mean Accuracy)

| Model | SFT Data | RL Data | Reasoning Format | Accuracy |
|-------|----------|---------|-----------------|----------|
| Base Model | — | — | Text-conditioned | 51.80% |
| Cold-Start | EAQA-SFT | — | Audio-anchored | 56.77% |
| Cold-Start | EAQA-SFT | — | Audio-interleaved | 52.26% |
| Echo | EAQA-SFT | EAQA-RL | Audio-interleaved | **69.99%** |
| Direct RL | — | EAQA-RL | Text-conditioned | 63.15% |

### Key Findings

- **Reasoning format comparison**: Along the E→B′→D trajectory, greater audio participation correlates with higher performance, while output length and latency remain comparable.
- **Training dynamics**: During RL, the model stabilizes at ~1.9 segment references per response, average segment duration of 3.0s, and segment overlap rate of only ~0.1.
- **Segment coverage**: 99.4% of responses re-listen to at least one segment; 78.0% re-listen to two or more; segments are distributed uniformly across the audio timeline.
- **Skill improvements**: Multi-speaker role mapping +37.0%, event-based sound reasoning +20.8%, emotional state summarization +20.5%.
- **Generalization**: Although SFT data only covers the first 10 seconds, Echo accurately localizes informative segments in longer audio.

## Highlights & Insights

1. **Paradigm innovation**: "Thinking *with* audio" rather than "thinking *about* audio" elevates audio from a static context to an active reasoning component.
2. Attention analysis provides intuitive evidence: audio-interleaved reasoning increases audio token attention from <5% to 10–14% (Δ+140%).
3. The engineering design of reasoning format adaptation is elegant and effective — requiring only pausing generation at `<seg>` tags to insert audio tokens.
4. The consistency reward $\mathcal{R}_\text{consist}$ and segment reward $\mathcal{R}_\text{seg}$ are well-designed, effectively guiding the model to learn meaningful re-listening behavior.

## Limitations & Future Work

1. The current re-listening implementation is relatively simple; more advanced audio operations such as slow playback and frequency band isolation could be explored.
2. CoT annotations in EAQA-SFT are automatically generated from fixed temporal metadata, lacking human-crafted heuristics.
3. Due to DeepSeek-R1's "rumination" tendency, the data may exhibit insufficient diversity in reasoning paths.
4. Computational overhead: each re-listening requires reprocessing audio tokens, resulting in inference latency of ~2.12s (vs. baseline 1.18s).

## Related Work & Insights

- Analogous to the evolution in visual reasoning: from Multimodal CoT → visual grounding reasoning → direct insertion of image patches.
- Complementary to Audio-Reasoner (SFT-only) and Omni-R1 (RL-only); Echo demonstrates the superiority of the two-stage SFT+RL approach.
- The data generation pipeline of "audio → multi-perspective text → LLM-synthesized QA-CoT" is generalizable to other modalities such as video.
- The design pattern of GRPO + multi-component rewards has broad applicability in multimodal RL.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Audio-interleaved reasoning is an entirely new paradigm)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (3 benchmarks, detailed ablations, training dynamics analysis)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, polished figures, in-depth analysis)
- Value: ⭐⭐⭐⭐⭐ (Opens a new direction for audio understanding with strong empirical support)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts](loongrl_rl_for_reasoning_long_contexts.md)
- [\[ICLR 2026\] LadderSym: A Multimodal Interleaved Transformer for Music Practice Error Detection](laddersym_a_multimodal_interleaved_transformer_for_music_practice_error_detectio.md)
- [\[NeurIPS 2025\] Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models](../../NeurIPS2025/reinforcement_learning/incentivizing_reasoning_for_advanced_instruction-following_of_large_language_mod.md)
- [\[ICLR 2026\] Thinking on the Fly: Test-Time Reasoning Enhancement via Latent Thought Policy Optimization](thinking_on_the_fly_test-time_reasoning_enhancement_via_latent_thought_policy_op.md)
- [\[ICLR 2026\] RM-R1: Reward Modeling as Reasoning](rm-r1_reward_modeling_as_reasoning.md)

</div>

<!-- RELATED:END -->
