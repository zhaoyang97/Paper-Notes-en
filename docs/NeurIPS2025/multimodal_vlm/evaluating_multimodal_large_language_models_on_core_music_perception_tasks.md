---
title: >-
  [Paper Note] Evaluating Multimodal Large Language Models on Core Music Perception Tasks
description: >-
  [NeurIPS 2025][Multimodal VLM][Multimodal LLM] This paper systematically evaluates multimodal LLMs on three core music perception tasks—syncopation scoring, transposition detection…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Multimodal LLM"
  - "Music Perception"
  - "Audio Understanding"
  - "Symbolic Reasoning"
  - "LogicLM"
date: 2026-05-08
content_hash: 9efa1b7ba5ed165e
---

# Evaluating Multimodal Large Language Models on Core Music Perception Tasks

**Conference**: NeurIPS 2025
**arXiv**: [2510.22455](https://arxiv.org/abs/2510.22455)  
**Code**: None (stimuli sourced from The MUSE Benchmark)  
**Area**: Multimodal VLM
**Keywords**: Multimodal LLM, Music Perception, Audio Understanding, Symbolic Reasoning, LogicLM

## TL;DR
This paper systematically evaluates multimodal LLMs on three core music perception tasks—syncopation scoring, transposition detection, and chord quality identification—under both audio and MIDI input modalities, revealing that models approach ceiling performance on symbolic reasoning while exhibiting significant deficits in audio perception.

## Background & Motivation
Multimodal foundation models (e.g., Qwen2.5-Omni, Gemini 2.5) claim "music understanding" capabilities, yet existing benchmarks (AIR-Bench, MMAR, MMAU, etc.) focus primarily on classification and captioning tasks, which cannot distinguish whether models genuinely understand musical structure or merely exploit superficial spectral patterns. Audio-language models (SALMONN, Qwen-Audio, Audio Flamingo 2) perform well on speech and sound recognition, but have never been systematically tested on the relational attributes unique to music.

The central motivation is to **decouple perception from reasoning**. Strong performance on MIDI does not imply that a model can "listen" to audio—MIDI strips away micro-timing, dynamics, expressive nuance, and other features that make music musically meaningful.

## Method

### Overall Architecture
The authors design a $3 \times 3 \times 2$ factorial experiment:
- **3 tasks**: Syncopation Scoring, Transposition Detection, Chord Quality Identification
- **3 reasoning strategies**: Standalone (direct answer), CoT (chain-of-thought), LogicLM (symbolic reasoning + deterministic solver)
- **2 input modalities**: Audio (raw waveform) and MIDI (symbolic text representation)
- Crossed with ZS (zero-shot) and FS (few-shot), yielding 12 conditions per task

### Key Designs

**Rationale for the three core tasks:**

1. **Syncopation Scoring**: 20 eight-second rhythmic excerpts at 120 BPM; hi-hat maintains constant eighth notes while kick and snare vary between on-beat and off-beat positions. The task requires counting off-beat kick/snare events and mapping the count to a categorical score (0/2/4/6/8). This **probes sensitivity to metric expectation violation and perception of metrical displacement**.

2. **Transposition Detection**: 20 excerpt pairs (each ≈9 s); the second excerpt is either a transposition of the first or a different melody entirely. The task is binary classification (same/different melody). This **probes melodic identity recognition invariant to absolute pitch**, which is a core perceptual capacity underlying cross-key melody recognition in humans.

3. **Chord Quality Identification**: 44 nine-second excerpts at 120 BPM, each containing block chords followed by ascending arpeggios. Four-way classification: major triad, minor triad, dominant seventh, diminished triad. This **probes interval pattern recognition rather than absolute frequency matching**.

**Adapting LogicLM to the music domain:**
- Adapted from Pan et al.'s LogicLM framework; models serve as "Perceptual Formulators" that generate machine-verifiable symbolic schemas
- A deterministic solver (`solver.py`) executes each schema, preventing "unfaithful reasoning" in which a correct final answer masks erroneous perceptual analysis
- Schema format violations automatically trigger a self-repair loop prompting the model to correct its output

**Stimulus construction:**
- All stimuli are original recordings performed by professional musicians (from the MUSE Benchmark)
- MIDI versions were re-performed on a MIDI keyboard and exported to `.txt` via `mido`
- Audio and MIDI prompts differ only in replacing "you will hear…" with "you will be given MIDI data…"; all schemas remain identical

### Loss & Training
This is an evaluation study; no model training is involved. The three evaluated models are:
- **Gemini 2.5 Pro** and **Gemini 2.5 Flash**: accessed via the `google.genai` SDK
- **Qwen2.5-Omni 7B**: run on NYU HPC with identical prompts, decoding settings, and evaluation pipeline
- All runs are deterministic (temperature = 0); each trial is independent (no cross-trial history)

## Key Experimental Results

### Main Results

| Modality | Shot | Strategy | Flash(Sync.) | Pro(Sync.) | Qwen(Sync.) | Flash(Trans.) | Pro(Trans.) | Qwen(Trans.) | Flash(Chord) | Pro(Chord) | Qwen(Chord) |
|----------|------|----------|--------------|------------|-------------|---------------|-------------|--------------|--------------|------------|-------------|
| Audio | ZS | Stand. | 30.0 | 25.0 | 20.0 | 55.6 | **94.7** | 75.0 | 31.8 | **47.7** | 31.8 |
| Audio | ZS | CoT | 35.0 | 25.0 | 20.0 | 76.9 | **95.0** | 65.0 | 31.8 | 43.2 | 31.8 |
| Audio | ZS | LogicLM | 20.0 | 20.0 | 20.0 | 65.0 | 80.0 | 50.0 | 11.4 | 18.2 | 6.8 |
| MIDI | ZS | Stand. | 84.2 | **95.0** | 25.0 | **100** | **100** | 85.0 | 50.0 | **97.7** | 22.7 |
| MIDI | ZS | CoT | 94.7 | **100** | 35.0 | 95.0 | **100** | 20.0 | **100** | **100** | 25.0 |
| MIDI | ZS | LogicLM | 90.0 | 80.0 | 20.0 | **100** | **100** | 10.0 | 93.2 | **100** | **100** |

**Core finding—large modality gap**: Gemini models substantially outperform on MIDI relative to Audio ($p < 0.001$), with the largest gaps observed on syncopation and chord tasks (MIDI ≈ 84–100% vs. Audio ≈ 6–65%).

### Ablation Study

| Factor | Finding |
|--------|---------|
| ZS vs. FS | No significant main effect (all $p > 0.05$); FS benefits Pro only on audio syncopation (~25% → ~65%) |
| Reasoning strategy | Syncopation: CoT yields marginal gains under audio; LogicLM effective only under MIDI. Transposition: Standalone/CoT optimal; LogicLM degrades accuracy. Chord: LogicLM-Audio collapses due to schema brittleness |
| Model differences | Gemini Pro is overall best; Qwen2.5-Omni lags across the board, with the largest deficits under LogicLM |

### Key Findings
- **Spurious success on transposition**: Gemini Pro frequently preserves the correct sequence length but fails to capture interval structure and contour direction (e.g., ground truth ↓↑↑↓… vs. model output ↑↓↓↑…)
- **LogicLM exposes degenerate strategies**: Standalone/CoT can obscure perceptual errors, whereas LogicLM enforces musical consistency, revealing genuine perceptual deficits
- **Confusion in chord identification**: Under the audio modality, closely related chord qualities (e.g., major triad vs. dominant seventh) are frequently confused

## Highlights & Insights
1. **Elegant experimental design**: The Audio vs. MIDI contrast precisely isolates perceptual from reasoning capabilities—a distinction absent from existing music AI benchmarks
2. **LogicLM adapted to music cognition**: Importing the neuro-symbolic reasoning framework into music perception evaluation exposes deep perceptual deficits concealed beneath surface-level success
3. **Practical implication**: Current systems should not claim "music understanding"—ceiling-level performance on MIDI does not constitute native audio-based capability

## Limitations & Future Work
- Only three models are evaluated; dedicated music foundation models (e.g., MusicGen, Jukebox) are not included
- Stimulus sets are small (20–44 excerpts per task), limiting statistical power
- More complex musical tasks are not addressed (e.g., polyphonic analysis, formal structure, expressive content)
- MIDI as a symbolic proxy oversimplifies musical information; richer formats such as MusicXML warrant consideration
- The specific bottlenecks in audio front-ends (spectral analysis vs. temporal tracking) are not quantified

## Related Work & Insights
- Complements general-purpose audio benchmarks such as AIR-Bench and MMAU; this work fills the gap along the "structural understanding" dimension
- The LogicLM adaptation strategy is generalizable to other multimodal domains requiring separation of perception and reasoning
- Findings have direct implications for music AI applications (playlist recommendation, music education)—systems must be built "audio-first"

## Rating
- Novelty: ⭐⭐⭐⭐ (first systematic decoupling of music perception and reasoning; novel adaptation of LogicLM)
- Experimental Thoroughness: ⭐⭐⭐ (factorial design is well-motivated but stimulus scale is small and model coverage is limited)
- Writing Quality: ⭐⭐⭐⭐ (clear structure, compelling argumentation)
- Value: ⭐⭐⭐⭐ (important cautionary contribution to the music AI field; exposes perceptual bottlenecks)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Evaluating Proactive Risk Awareness of Multimodal Language Models](towards_evaluating_proactive_risk_awareness_of_multimodal_language_models.md)
- [\[NeurIPS 2025\] Adapting Vision-Language Models for Evaluating World Models](adapting_visionlanguage_models_for_evaluating_world_models.md)
- [\[NeurIPS 2025\] Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models](seeing_is_believing_mitigating_ocr_hallucinations_in_multimodal_large_language_m.md)
- [\[NeurIPS 2025\] FlexAC: Towards Flexible Control of Associative Reasoning in Multimodal Large Language Models](flexac_towards_flexible_control_of_associative_reasoning_in_multimodal_large_lan.md)
- [\[NeurIPS 2025\] Efficient Multi-modal Large Language Models via Progressive Consistency Distillation](efficient_multi-modal_large_language_models_via_progressive_consistency_distilla.md)

</div>

<!-- RELATED:END -->
