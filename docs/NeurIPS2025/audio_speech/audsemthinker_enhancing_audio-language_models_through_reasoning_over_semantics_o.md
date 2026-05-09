---
title: >-
  [Paper Note] AudSemThinker: Enhancing Audio-Language Models through Reasoning over Semantics of Sound
description: >-
  [NeurIPS 2025][Audio & Speech][Audio Reasoning] AudSemThinker introduces a structured semantic reasoning framework for audio-language models by defining 9 categories of sound semantic descriptors (who/what/how/when/where, etc.). Built on Qwen2.5-Omni-7B and trained via SFT + GRPO (with verifiable rewards and length constraints), the model produces three-stage outputs in the format \<think\>\<semantic_elements\>\<answer\>, achieving 66.70% on the MMAU benchmark—surpassing Audio-Reasoner (61.71%) and Qwen2.5-Omni (65.60%).
tags:
  - NeurIPS 2025
  - "Audio & Speech"
  - Audio Reasoning
  - Semantic Descriptors
  - GRPO
  - Chain-of-Thought
  - Multimodal
date: 2026-05-08
content_hash: 9e6dfc0712ec5644
---

# AudSemThinker: Enhancing Audio-Language Models through Reasoning over Semantics of Sound

**Conference**: NeurIPS 2025
**arXiv**: [2505.14142](https://arxiv.org/abs/2505.14142)
**Code**: [https://github.com/GLJS/AudSemThinker](https://github.com/GLJS/AudSemThinker)
**Area**: Audio & Speech
**Keywords**: Audio Reasoning, Semantic Descriptors, GRPO, Chain-of-Thought, Multimodal

## TL;DR
AudSemThinker introduces a structured semantic reasoning framework for audio-language models by defining 9 categories of sound semantic descriptors (who/what/how/when/where, etc.). Built on Qwen2.5-Omni-7B and trained via SFT + GRPO (with verifiable rewards and length constraints), the model produces three-stage outputs in the format \<think\>\<semantic_elements\>\<answer\>, achieving 66.70% on the MMAU benchmark—surpassing Audio-Reasoner (61.71%) and Qwen2.5-Omni (65.60%).

## Background & Motivation

**Background**: Text-based LLMs have benefited from mature reasoning enhancement methods (o1, DeepSeek-R1), yet audio-language models still lack structured reasoning capabilities. Benchmarks such as MMAU and AudioBench reveal that audio understanding models perform inadequately on fine-grained semantics.

**Limitations of Prior Work**: (a) Severe benchmark contamination—AudioCaps-WavCaps overlap is 17.6% and Clotho overlap reaches 89%, with most models trained on AudioSet/Freesound leading to homogenization; (b) models lack fine-grained semantic understanding and cannot distinguish dimensions such as "who is producing sound," "how it is produced," and "where it occurs"; (c) Chain-of-Thought methods for audio reasoning remain immature.

**Key Challenge**: Audio contains rich semantic layers (sound-generating agents, physical sources, generation mechanisms, spatiotemporal context, etc.), yet models lack a structured means to reason over these dimensions.

**Goal**: To introduce a structured semantic reasoning framework for audio-language models, enabling the model to "think" through multiple semantic dimensions of sound before generating a response.

**Key Insight**: Nine categories of sound semantic descriptors are defined (sound-generating agents, physical sources, generation mechanisms, temporal/spatial context, acoustic surfaces, signal descriptors, auditory attributes, and non-auditory sensation), and the model is trained to first produce a semantic analysis before delivering an answer.

**Core Idea**: 9 sound semantic descriptor categories + \<think\>\<semantic\>\<answer\> three-stage reasoning + SFT/GRPO training = structured audio semantic reasoning.

## Method

### Overall Architecture
Base model: Qwen2.5-Omni-7B → **SFT** (LoRA fine-tuning of projection layers, AdamW lr=2e-4, 1 epoch ≈12h on a single H100) → Output format: \<think\>[reasoning] \<semantic_elements\>[9 descriptor categories] \<answer\>[response] → **Optional GRPO reinforcement** (accuracy reward + format reward + length constraint reward).

### Key Designs

1. **9 Sound Semantic Descriptor Categories**:

    - Function: Structuralize the semantic dimensions of sound.
    - Mechanism: (1) Sound-Generating Agents (who), (2) Physical Sound Sources (what), (3) Sound Generation Mechanisms (how), (4) Temporal Context (when), (5) Spatial Context (where), (6) Acoustic Surfaces, (7) Signal Descriptors, (8) Auditory Attributes, (9) Non-auditory Sensation.
    - Design Motivation: Sound semantics are inherently multi-dimensional—"a dog barking in a park" involves an agent (dog), spatial context (park), and mechanism (vocal cord vibration). Structured descriptors compel the model to conduct comprehensive analysis.

2. **GRPO Training (Reinforcement Learning with Verifiable Rewards)**:

    - Function: Further optimize reasoning quality via reinforcement learning.
    - Mechanism: Three reward types—(a) accuracy reward (string matching for multiple-choice questions); (b) format compliance reward (XML tag structure validation); (c) length constraint reward (target of 25-word reasoning budget with cosine-shaped penalty). No critic model is required (a property of GRPO).
    - Design Motivation: SFT only teaches imitation, whereas GRPO can explore superior reasoning strategies. Length constraints prevent verbose and uninformative reasoning chains.

3. **LoRA Fine-tuning Strategy**:

    - Function: Apply low-rank adaptation to projection layers.
    - Mechanism: Only the audio-text projection layers of Qwen2.5-Omni are fine-tuned (not the full model), preserving pretrained audio understanding capabilities. Ablations show that omitting LoRA causes catastrophic forgetting (23.6% → 15.5%).
    - Design Motivation: Full fine-tuning disrupts pretrained knowledge; LoRA balances adaptability with knowledge retention.

### Loss & Training
- SFT: Standard cross-entropy loss + LoRA.
- GRPO: Policy gradient with weighted combination of three rewards.
- 12K annotated preference pairs (4K prompts, pairwise comparisons with $k=2$–$9$).

## Key Experimental Results

### Main Results

| Method | MMAU Test-Mini | MMAU Test | MuchoMusic |
|--------|---------------|-----------|------------|
| Audio-CoT | 57.80% | — | — |
| Audio-Reasoner | 61.71% | — | — |
| Qwen2.5-Omni-7B | 65.60% | — | 70.09 |
| **AudSemThinker (SFT)** | 62.90% | 64.41% | — |
| **AudSemThinker-QA GRPO** | **66.70%** | **66.03%** | **76.66** |

### Ablation Study

| Configuration | MMAU Test-Mini |
|---------------|---------------|
| SFT baseline | 61.5% |
| + Semantic descriptors | 63.9% (+2.4%) |
| Without LoRA | 15.5% (catastrophic forgetting) |
| Training from scratch | 2.1% (complete failure) |
| GRPO length constraint 25 words | Best |
| GRPO length constraint 100–150 words | Effective but slightly weaker |
| QA full data vs. QA subset | Full data better (generalization) |

### Key Findings
- Semantic descriptors are effective during the SFT stage (+2.4%) but provide limited benefit during GRPO—likely because the reward signal does not leverage intermediate reasoning steps.
- GRPO substantially improves performance on multiple-choice tasks (66.70% vs. 62.90%) but underperforms SFT on open-ended tasks, since open-ended answers are difficult to verify.
- Music understanding is particularly strong: lyric reasoning achieves 100%, texture 94.12%, and instrument recognition 91.43%—indicating that semantic descriptors are especially valuable for music analysis.
- Speech understanding remains weak (no ASR integration), as the semantic descriptors do not cover speech content dimensions.
- LoRA is critical—it injects reasoning capability while preserving pretrained knowledge.

## Highlights & Insights
- **The 9 sound semantic descriptor categories** constitute a systematic knowledge structure for audio understanding—analogous to "objects, attributes, and relations" in vision, but more fine-grained.
- **The length constraint in GRPO is a notable design choice**: enforcing concise reasoning outperforms allowing verbose reasoning, suggesting that quality matters more than quantity.
- **The exploration of audio Chain-of-Thought is significant**: it demonstrates that structured reasoning is effective in the audio modality, extending the paradigm beyond vision and text.

## Limitations & Future Work
- GRPO fails on open-ended tasks—verifiable rewards for open-ended answers are difficult to define, necessitating improved reward design.
- Speech understanding is weak—semantic descriptors do not cover speech content dimensions; integration of an ASR module is needed.
- Semantic descriptors do not benefit GRPO—outcome-based rewards in GRPO do not exploit intermediate reasoning steps, pointing to the need for process-level rewards.
- Substantial hyperparameter tuning space (e.g., LoRA rank, learning rate, reasoning budget) remains underexplored.
- Benchmark contamination is severe (AudioCaps 17.6%, Clotho 89%), and reported evaluation scores may be inflated.

## Related Work & Insights
- **vs. Audio-Reasoner**: Also targets audio reasoning but without semantic decomposition; AudSemThinker is more structured.
- **vs. Audio-CoT**: Relies on simple CoT prompting without the granularity of semantic descriptors.
- **vs. DeepSeek-R1**: Represents a transfer of the text-based reasoning paradigm to the audio modality.

## Rating
- Novelty: ⭐⭐⭐⭐ First work to introduce structured semantic reasoning into audio-language models.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on MMAU + AudioBench with ablations.
- Writing Quality: ⭐⭐⭐⭐ Semantic descriptor design is clearly presented.
- Value: ⭐⭐⭐⭐ Opens a new direction for reasoning enhancement in audio AI.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Seeing Sound, Hearing Sight: Uncovering Modality Bias and Conflict of AI Models in Sound Localization](seeing_sound_hearing_sight_uncovering_modality_bias_and_conflict_of_ai_models_in.md)
- [\[ICLR 2026\] Stitch: Simultaneous Thinking and Talking with Chunked Reasoning for Spoken Language Models](../../ICLR2026/audio_speech/stitch_simultaneous_thinking_and_talking_with_chunked_reasoning_for_spoken_langu.md)
- [\[NeurIPS 2025\] Target Speaker Extraction Through Comparing Noisy Positive and Negative Audio Enrollments](target_speaker_extraction_through_comparing_noisy_positive_and_negative_audio_en.md)
- [\[NeurIPS 2025\] A Controllable Examination for Long-Context Language Models](a_controllable_examination_for_longcontext_language_models.md)
- [\[NeurIPS 2025\] Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers](physics_of_language_models_part_41_architecture_design_and_the_magic_of_canon_la.md)

<!-- RELATED:END -->
