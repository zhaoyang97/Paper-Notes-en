---
title: >-
  [Paper Note] Beyond Transcription: Unified Audio Schema for Perception-Aware AudioLLMs
description: >-
  [ACL 2026][Audio & Speech][Audio Large Language Models] This paper identifies that the perceptual weakness of current AudioLLMs stems from an ASR-centric training paradigm that systematically suppresses paralinguistic and non-linguistic information. It proposes the Unified Audio Schema (UAS), which structures audio information into a three-dimensional JSON format covering transcription, paralinguistics, and non-linguistic events. The approach achieves a 10.9% improvement in perceptual accuracy on the MMSU benchmark while preserving reasoning capabilities.
tags:
  - ACL 2026
  - "Audio & Speech"
  - Audio Large Language Models
  - Perception Enhancement
  - Unified Audio Schema
  - Paralinguistic Information
  - ASR
date: 2026-05-08
content_hash: 26bc4ec0eb427836
---

# Beyond Transcription: Unified Audio Schema for Perception-Aware AudioLLMs

**Conference**: ACL 2026
**arXiv**: [2604.12506](https://arxiv.org/abs/2604.12506)
**Code**: [GitHub](https://github.com/Tencent/Unified_Audio_Schema)
**Area**: Speech Processing
**Keywords**: Audio Large Language Models, Perception Enhancement, Unified Audio Schema, Paralinguistic Information, ASR

## TL;DR
This paper identifies that the perceptual weakness of current AudioLLMs stems from an ASR-centric training paradigm that systematically suppresses paralinguistic and non-linguistic information. It proposes the Unified Audio Schema (UAS), which structures audio information into a three-dimensional JSON format covering transcription, paralinguistics, and non-linguistic events. The approach achieves a 10.9% improvement in perceptual accuracy on the MMSU benchmark while preserving reasoning capabilities.

## Background & Motivation

**Background**: AudioLLMs exhibit a paradoxical pattern — strong performance on complex reasoning tasks (~70%) contrasts with a sharp decline on basic acoustic perception tasks (~40%). For instance, a model may correctly transcribe "I'm fine" while entirely missing the distress implied by a trembling voice, or fail to register the sound of a slamming door.

**Limitations of Prior Work**: This perceptual deficit persists across model scales and architectures, suggesting the problem lies not in model capacity but in training methodology. The vast majority of AudioLLMs rely on ASR as their core training signal; however, ASR is inherently selective — it deliberately normalizes away prosody, speaker identity, emotion, and acoustic context in order to recover canonical text.

**Key Challenge**: ASR-based training creates a fundamental asymmetry: models are continuously rewarded for reasoning about *what was said*, while being implicitly penalized for attending to *how it was said* and *what other sounds are present*. Perceptual ability is not undertrained — it is systematically de-emphasized.

**Goal**: To design a training supervision format that explicitly preserves acoustic perceptual information without sacrificing semantic alignment.

**Key Insight**: Drawing on Laver's semiotic framework for speech signals, the paper decomposes the audio signal into three information layers: linguistic, paralinguistic, and extralinguistic.

**Core Idea**: A structured JSON schema explicitly encodes these three information layers as training targets, transforming the "implicit discarding" characteristic of ASR into "explicit preservation."

## Method

### Overall Architecture
UAS defines a three-layer JSON schema → an automated pipeline generates UAS annotations from existing ASR corpora → UAS data is incorporated into a standard multi-stage training procedure → the resulting UAS-Audio model achieves both perceptual and reasoning capabilities.

### Key Designs

1. **Three-Layer Structure of the Unified Audio Schema**:

    - **Function**: Explicitly encodes acoustic information discarded during ASR training.
    - **Mechanism**: **Transcription**: verbatim text equivalent to ASR output. **Paralinguistics**: six subfields — age, gender, emotion, accent, prosody, and timbre. **Non-linguistic Events**: environmental descriptions, discrete sound events (e.g., a door slam), and continuous background sounds (e.g., engine rumble). Transcription and paralinguistic fields are set to null for non-speech audio.
    - **Design Motivation**: (1) *Decoupled learning*: decomposing holistic understanding into explicit subtasks prevents feature conflation; (2) *Syntactic invariance*: the JSON format provides a consistent, low-entropy supervision target that is more learnable than unstructured descriptions; (3) *Programmatic accessibility*: downstream applications can reliably extract acoustic attributes.

2. **Scalable UAS Data Generation Pipeline**:

    - **Function**: Automatically generates UAS annotations from existing ASR corpora without manual labeling.
    - **Mechanism**: Three stages — (1) an acoustic description model generates paralinguistic and environmental descriptions from raw audio; (2) an LLM synthesizes these descriptions with the original transcription into a structured UAS JSON; (3) multi-level automated validation (ontology constraints, transcription completeness, logical consistency, duration–content alignment). Manual auditing of 400 samples confirms accuracy exceeding 95% for most attributes.
    - **Design Motivation**: Avoids costly manual annotation by leveraging existing models to convert standard ASR datasets into perceptually aware supervision.

3. **UAS-QA Supplementary Dataset**:

    - **Function**: Trains the model to apply structured acoustic knowledge when answering downstream questions.
    - **Mechanism**: Three types of QA pairs are automatically generated from UAS annotations: direct QA (querying specific fields), multiple-choice questions, and yes/no questions, covering all UAS fields.
    - **Design Motivation**: UAS annotations teach the model *what to perceive*; UAS-QA teaches the model *how to apply that knowledge*.

### Loss & Training
A standard four-stage procedure: (1) discrete token alignment (vocabulary expansion) → (2) Audio-LLM adaptation (LLM and encoder frozen; only the projection layer trained on UAS data) → (3) full-parameter instruction fine-tuning (mixed ASR/TTS + UAS + UAS-QA data) → (4) GRPO reinforcement.

## Key Experimental Results

### Main Results (MMSU / MMAR / MMAU Benchmarks)

| Model | MMSU Perception | MMSU Reasoning | MMSU Overall | MMAR | MMAU | Three-Benchmark Avg. |
|-------|----------------|----------------|--------------|------|------|----------------------|
| Qwen2.5-Omni | 42.0 | 70.0 | ~56 | 55.8 | 64.2 | ~58.7 |
| Kimi-Audio | ~38 | ~68 | ~53 | 56.3 | 65.0 | ~58.1 |
| Step-Audio2-mini | ~40 | ~69 | ~55 | 57.2 | 63.8 | ~58.7 |
| **UAS-Audio** | **52.9** | **70.1** | **~61** | **60.1** | **65.2** | **~62.1** |

### Ablation Study

| Configuration | MMSU Perception | MMSU Reasoning | Notes |
|---------------|----------------|----------------|-------|
| No UAS (ASR only) | ~40 | ~70 | Weak perception, normal reasoning |
| UAS annotations only | ~48 | ~69 | Partial perceptual improvement |
| UAS-QA only | ~45 | ~69 | QA alone is insufficient |
| **UAS + UAS-QA** | **52.9** | **70.1** | Complementary; best overall |

### Key Findings
- **UAS-Audio achieves an absolute improvement of ~11% on MMSU perception** while fully preserving reasoning performance.
- **UAS is effective across both continuous and discrete AudioLLM architectures**, confirming that the root cause lies in supervision rather than architecture.
- **UAS annotations and UAS-QA provide complementary supervision**: annotations teach *what to perceive*, QA teaches *how to use that knowledge*.
- **State-of-the-art results on the MMAR reasoning benchmark (60.1%)** demonstrate that perceptual enhancement does not harm reasoning.
- **Data validation confirms high pipeline quality**: manual auditing of 400 samples yields >95% accuracy for most attributes.

## Highlights & Insights
- **Diagnosing the root cause of perceptual weakness in AudioLLMs** — characterizing it as "systematic de-emphasis" rather than "undertraining" — is arguably more valuable than the method itself, as it provides directional guidance for the entire field.
- **Using a structured JSON schema as a training target** is a generalizable principle applicable to any multi-dimensional perceptual task: it decomposes implicit holistic understanding into explicit, structured subtasks.
- **The annotation-free pipeline** makes the approach highly scalable, enabling any ASR dataset to be converted into perceptually enriched supervision.

## Limitations & Future Work
- The six paralinguistic subfields in UAS are manually defined and may omit certain important dimensions (e.g., breathing patterns, speech rate variation).
- The pipeline depends on the quality of acoustic description models and may degrade for low-resource languages.
- Validation is limited to the 7B scale; effectiveness on larger or smaller models remains to be confirmed.
- Non-linguistic event detection accuracy may decline in acoustically complex scenes.
- Future work could explore allowing the model to autonomously decide when to produce UAS output rather than generating it unconditionally.

## Related Work & Insights
- **vs. Qwen2.5-Omni**: Qwen2.5-Omni is a multimodal model that nonetheless relies primarily on ASR-centric training, resulting in weak perception. UAS addresses this by modifying the supervision paradigm rather than the architecture.
- **vs. Caption-based methods**: Unstructured descriptions exhibit high-entropy variability (the same sound can be described in multiple ways), whereas UAS's JSON format provides a low-entropy, consistent supervision target.
- **vs. Specialized perception models**: Task-specific models for emotion recognition or speaker identification achieve high accuracy but are limited in scope. UAS enables full-dimensional perception within a unified model.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The core insight (ASR-centric training suppresses perception) is more innovative than the method itself.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three major benchmarks, ablation studies, human validation, and cross-architecture verification.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Problem diagnosis is clear; theoretical grounding in the Laver framework is solid.
- **Value**: ⭐⭐⭐⭐⭐ — Identifies a foundational problem in the AudioLLM field and provides a concrete solution pathway.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A TRIANGLE Enables Multimodal Alignment Beyond Cosine Similarity](../../NeurIPS2025/audio_speech/a_triangle_enables_multimodal_alignment_beyond_cosine_simila.md)
- [\[AAAI 2026\] HPSU: A Benchmark for Human-Level Perception in Real-World Spoken Speech Understanding](../../AAAI2026/audio_speech/hpsu_a_benchmark_for_human-level_perception_in_real-world_spoken_speech_understa.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/audio_speech/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [\[CVPR 2026\] UniM: A Unified Any-to-Any Interleaved Multimodal Benchmark](../../CVPR2026/audio_speech/unim_a_unified_any-to-any_interleaved_multimodal_benchmark.md)
- [\[AAAI 2026\] USE: A Unified Model for Universal Sound Separation and Extraction](../../AAAI2026/audio_speech/use_a_unified_model_for_universal_sound_separation_and_extraction.md)

</div>

<!-- RELATED:END -->
