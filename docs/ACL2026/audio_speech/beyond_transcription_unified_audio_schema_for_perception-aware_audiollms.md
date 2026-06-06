---
title: >-
  [Paper Note] Beyond Transcription: Unified Audio Schema for Perception-Aware AudioLLMs
description: >-
  [ACL 2026][Audio & Speech][Audio Large Language Models] Reveals that the perception weakness of current AudioLLMs stems from the ASR-centric training paradigm (systematic suppression of paralinguistic and non-linguistic…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Audio Large Language Models"
  - "Perception Enhancement"
  - "Unified Audio Schema"
  - "Paralinguistic Information"
  - "ASR"
date: 2026-05-08
content_hash: 956d9aae3e7153df
---

# Beyond Transcription: Unified Audio Schema for Perception-Aware AudioLLMs

**Conference**: ACL 2026  
**arXiv**: [2604.12506](https://arxiv.org/abs/2604.12506)  
**Code**: [GitHub](https://github.com/Tencent/Unified_Audio_Schema)  
**Area**: Audio Processing  
**Keywords**: Audio Large Language Models, Perception Enhancement, Unified Audio Schema, Paralinguistic Information, ASR

## TL;DR
Reveals that the perception weakness of current AudioLLMs stems from the ASR-centric training paradigm (systematic suppression of paralinguistic and non-linguistic information). Proposes the Unified Audio Schema (UAS) to structure audio information into a JSON format across three dimensions: transcription, paralinguistic, and non-linguistic events. Achieving a $10.9\%$ perception accuracy improvement on the MMSU benchmark while maintaining reasoning capability.

## Background & Motivation

**Background**: AudioLLMs exhibit a paradox—performing well on complex reasoning tasks (~$70\%$), but dropping sharply on basic acoustic perception tasks (~$40\%$). For instance, a model might correctly transcribe "I'm fine" while completely ignoring the distress implied by a trembling voice, or failing to notice a door slamming.

**Limitations of Prior Work**: This perception defect persists across model scales and architectures, indicating the issue lies in the training methodology rather than model capacity. The vast majority of AudioLLMs use ASR as the core training signal, and ASR is inherently selective—it deliberately normalizes out prosody, speaker identity, emotion, and acoustic context to recover canonical text.

**Key Challenge**: ASR training creates a fundamental asymmetry—the model is continuously rewarded for reasoning about "what was said," while being implicitly punished for focusing on "how it was said" and "what other sounds exist." Perception is not under-trained; it is systematically de-emphasized.

**Goal**: Design a training supervision format that explicitly preserves acoustic perception information without sacrificing semantic alignment.

**Key Insight**: Starting from Laver's semiotic framework of speech signals, the audio signal is decomposed into three information layers: linguistic, paralinguistic, and extralinguistic.

**Core Idea**: Use a structured JSON schema to explicitly encode the three information layers of audio as training targets, turning the "implicit discarding" of ASR into "explicit preservation."

## Method

### Overall Architecture
UAS defines a three-layer JSON schema → an automated pipeline generates UAS annotations from existing ASR corpora → UAS data is inserted into standard multi-stage training workflows → The UAS-Audio model possesses both perception and reasoning capabilities.

### Key Designs

1. **Unified Audio Schema Three-Layer Structure Definition**:

    - **Function**: Explicitly encode acoustic information discarded during ASR training.
    - **Mechanism**: **Transcription**: Verbatim text equivalent to ASR output. **Paralinguistics**: Six sub-fields—age, gender, emotion, accent, prosody, and timbre. **Non-linguistic Events**: Environmental descriptions, discrete sound events (e.g., door slam), and continuous background noise (e.g., engine roar). Transcription and paralinguistic fields are set to null for non-speech audio.
    - **Design Motivation**: (1) Decoupled learning: Decomposes "holistic understanding" into explicit sub-tasks to prevent feature confusion; (2) Syntactic invariance: The JSON format provides a consistent low-entropy supervision target, easier to learn than unstructured descriptions; (3) Programmatic accessibility: Downstream applications can reliably extract acoustic attributes.

2. **Scalable UAS Data Generation Pipeline**:

    - **Function**: Automatically generate UAS labels from existing ASR corpora without manual annotation.
    - **Mechanism**: Three stages—(1) Use acoustic description models to generate paralinguistic and environmental descriptions from raw audio; (2) Use an LLM to synthesize descriptions with original transcriptions into a structured UAS JSON; (3) Multi-level automated verification (ontology constraints, transcription integrity, logical consistency, duration-content alignment). Manual audits of 400 samples show accuracy $>95\%$ for most attributes.
    - **Design Motivation**: Avoid expensive manual labeling, leveraging existing models to transform standard ASR datasets into perception-aware supervision.

3. **UAS-QA Supplementary Dataset**:

    - **Function**: Train the model to utilize structured acoustic knowledge to answer downstream questions.
    - **Mechanism**: Automatically generate three types of QA pairs based on UAS annotations: Direct QA (querying specific fields), Multiple Choice, and Yes/No questions. Covers all UAS fields.
    - **Design Motivation**: UAS annotations teach the model "what to perceive," while UAS-QA teaches "how to apply this knowledge."

### Loss & Training
Standard four stages: (1) Discrete token alignment (vocabulary expansion) → (2) Audio-LLM adaptation (freeze LLM and encoder, train projection layer only using UAS data) → (3) Full-parameter instruction fine-tuning (ASR/TTS + UAS + UAS-QA mixture) → (4) GRPO reinforcement.

## Key Experimental Results

### Main Results (MMSU / MMAR / MMAU Benchmarks)

| Model | MMSU Perception | MMSU Reasoning | MMSU Overall | MMAR | MMAU | Avg. of 3 Benchmarks |
|------|----------|----------|----------|------|------|-----------|
| Qwen2.5-Omni | 42.0 | 70.0 | ~56 | 55.8 | 64.2 | ~58.7 |
| Kimi-Audio | ~38 | ~68 | ~53 | 56.3 | 65.0 | ~58.1 |
| Step-Audio2-mini | ~40 | ~69 | ~55 | 57.2 | 63.8 | ~58.7 |
| **UAS-Audio** | **52.9** | **70.1** | **~61** | **60.1** | **65.2** | **~62.1** |

### Ablation Study

| Configuration | MMSU Perception | MMSU Reasoning | Description |
|------|----------|----------|------|
| W/O UAS (ASR only) | ~40 | ~70 | Weak perception, normal reasoning |
| UAS Labels only | ~48 | ~69 | Partial perception gain |
| UAS-QA only | ~45 | ~69 | QA alone is insufficient |
| **UAS + UAS-QA** | **52.9** | **70.1** | Best performance, complementary |

### Key Findings
- **UAS-Audio achieves an absolute gain of approximately $11\%$ in MMSU Perception**, while fully maintaining reasoning performance.
- **UAS is applicable to both continuous and discrete AudioLLM architectures**, proving the issue lies in supervision rather than architecture.
- **UAS labels and UAS-QA provide complementary supervision**: labels teach "what to perceive," while QA teaches "how to use."
- **Achieved SOTA on the MMAR reasoning benchmark ($60.1\%$)**, indicating perception enhancement does not harm reasoning.
- **Data validation confirms high pipeline quality**: Manual audit of 400 samples shows $>95\%$ accuracy for most attributes.

## Highlights & Insights
- **Diagnosing the root cause of AudioLLM perception weakness** as "systematic de-emphasis" rather than "under-training" in ASR-centric training—this insight is more valuable than the method itself, providing direction for the field.
- **Using a JSON structured schema as a training target** can be generalized to any multi-dimensional perception task—decomposing implicit "holistic understanding" into explicit structured sub-tasks.
- **The pipeline requires no additional manual annotation**, making the method highly scalable and capable of transforming any ASR dataset into perception-enhanced data.

## Limitations & Future Work
- The six paralinguistic sub-fields of UAS are hand-defined and may miss important dimensions (e.g., breathing patterns, speaking rate variation).
- The pipeline depends on the quality of acoustic description models and may degrade in low-resource languages.
- Validated only at the 7B scale; performance on larger or smaller models remains to be confirmed.
- Detection accuracy for non-linguistic events may decrease in complex acoustic scenarios.
- Could explore allowing the model to automatically decide whether to output UAS instead of always generating it.

## Related Work & Insights
- **vs Qwen2.5-Omni**: Qwen2.5-Omni is a multimodal model but still ASR-centric in training, leading to weak perception. UAS solves this by changing the supervision method.
- **vs Caption-based methods**: Unstructured descriptions have high-entropy variability (the same sound can be described in many ways); the JSON format of UAS provides a low-entropy, consistent target.
- **vs Specialized Perception Models**: Specialized models for emotion or speaker recognition are accurate but narrow. UAS achieves all-dimensional perception within a unified model.

## Rating
- Novelty: ⭐⭐⭐⭐ The core insight (ASR-centric training suppresses perception) is more innovative than the method itself.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three major benchmarks + ablations + manual validation, validated across architectures.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem diagnosis, solid theoretical foundation (Laver framework).
- Value: ⭐⭐⭐⭐⭐ Identifies a directional issue and solution path for the AudioLLM field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond Transcripts: A Renewed Perspective on Audio Chaptering](beyond_transcripts_a_renewed_perspective_on_audio_chaptering.md)
- [\[ACL 2026\] Speech-Hands: A Self-Reflection Voice Agentic Approach to Speech Recognition and Audio Reasoning with Omni Perception](speech-hands_a_self-reflection_voice_agentic_approach_to_speech_recognition_and_.md)
- [\[ACL 2026\] UniVocal: Unified Speech-Singing Code-mixed Synthesis](univocal_unified_speech-singing_code-switching_synthesis.md)
- [\[ICML 2026\] Two-Dimensional Quantization for Geometry-Aware Audio Coding](../../ICML2026/audio_speech/two-dimensional_quantization_for_geometry-aware_audio_coding.md)
- [\[ACL 2026\] UniSRM: A Unified Speech Reward Model for Fine-Grained Speech Evaluation](unisrm_a_unified_speech_reward_model_for_reasoning-based_fine-grained_assessment.md)

</div>

<!-- RELATED:END -->
