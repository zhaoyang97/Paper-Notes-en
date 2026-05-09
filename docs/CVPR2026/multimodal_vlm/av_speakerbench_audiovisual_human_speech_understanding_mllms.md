---
title: >-
  [Paper Note] See, Hear, and Understand: Benchmarking Audiovisual Human Speech Understanding in Multimodal Large Language Models
description: >-
  [CVPR 2026][Multimodal VLM][audiovisual reasoning] This paper introduces AV-SpeakerBench, a speaker-centric audiovisual reasoning benchmark comprising 3,212 multiple-choice questions, revealing Gemini 2.5 Pro's superiority in audiovisual fusion while exposing significant deficiencies of open-source models in speaker reasoning.
tags:
  - CVPR 2026
  - Multimodal VLM
  - audiovisual reasoning
  - speaker-centric benchmark
  - multimodal fusion
  - speech understanding
  - temporal localization
date: 2026-05-08
content_hash: bbee6e04d647be31
---

# See, Hear, and Understand: Benchmarking Audiovisual Human Speech Understanding in Multimodal Large Language Models

**Conference**: CVPR 2026
**arXiv**: [2512.02231](https://arxiv.org/abs/2512.02231)
**Code**: [https://plnguyen2908.github.io/AV-SpeakerBench-project-page/](https://plnguyen2908.github.io/AV-SpeakerBench-project-page/)
**Area**: Multimodal VLM / Audiovisual Understanding
**Keywords**: audiovisual reasoning, speaker-centric benchmark, multimodal fusion, speech understanding, temporal localization

## TL;DR

This paper introduces AV-SpeakerBench, a speaker-centric audiovisual reasoning benchmark comprising 3,212 multiple-choice questions, revealing Gemini 2.5 Pro's superiority in audiovisual fusion while exposing significant deficiencies of open-source models in speaker reasoning.

## Background & Motivation

1. **Background**: Multimodal large language models have expanded from image–text to video and audio understanding, with increasing pursuit of unified processing of visual, audio, and linguistic modalities.
2. **Limitations of Prior Work**: Many existing video benchmarks (e.g., Video-MME) can be answered using visual information alone; audiovisual benchmarks either focus on non-speech sound events (AVQA) or coarse-grained classification (VGGSounder), without evaluating fine-grained speaker reasoning.
3. **Key Challenge**: No benchmark systematically evaluates whether models can jointly determine *who is speaking*, *what was said*, and *when it was said*.
4. **Goal**: To construct an audiovisual reasoning benchmark centered on the speaker as the core reasoning unit.
5. **Key Insight**: Fusion-driven question design that embeds audiovisual dependency into the semantics of questions and answer choices.
6. **Core Idea**: Every question requires cross-modal fusion to answer — for example, associating spoken phrases with visible speakers or localizing speech based on visual events.

## Method

### Overall Architecture

An IRB-approved benchmark consisting of 2,051 video clips and 3,212 four-choice multiple-choice questions spanning 12 task types. Videos are sourced from YouTube (movie clips, game shows, street interviews, etc.).

### Key Designs

1. **Speaker-Centric Task Design**:

    - **Function**: Shifts evaluation from scene-level understanding to person-centric audiovisual localization.
    - **Mechanism**: The 12 tasks are grouped into three categories: speaker-centric (detection, identification, counting), vision-centric (attribute, activity, counting recognition), and audio-centric (recognition, duration, pitch, speaking rate, intensity, counting). Each task includes at least 200 validation questions.
    - **Design Motivation**: To cover diverse speaker reasoning patterns, from basic perception to temporal reasoning.

2. **Fusion-Driven Question Design**:

    - **Function**: Ensures that each question requires genuine audiovisual fusion.
    - **Mechanism**: Audiovisual dependencies are embedded in question semantics via: (1) associating spoken phrases with visible identities; (2) localizing speech through visual events; (3) combining audiovisual cues in multi-speaker scenes. Distractors are drawn from entities and events within the same clip.
    - **Design Motivation**: Prevents models from answering correctly using a single modality alone.

3. **Expert-Curated Annotation Pipeline**:

    - **Function**: Ensures annotation quality and cross-modal validity.
    - **Mechanism**: Annotators are experienced researchers rather than crowdworkers. A multi-stage refinement process includes: (1) initial review by independent researchers; (2) language model polishing; (3) final review by at least two additional researchers. Ambiguous or single-modality-solvable samples are filtered out.
    - **Design Motivation**: Ensures all retained questions exhibit temporal sensitivity and speaker localization requirements.

### Loss & Training

This is a pure evaluation benchmark with no training involved. Human baselines are established by graduate researchers.

## Key Experimental Results

### Main Results

| Model | Speaker-Centric | Vision-Centric | Audio-Centric | Overall |
|-------|----------------|---------------|--------------|---------|
| Gemini 2.5 Pro | 76.7 | 71.5 | 72.9 | 73.0 |
| Qwen3-Omni-30B | 54.5 | 51.8 | 53.7 | 54.1 |
| Gemini 2.0 Flash | 57.2 | 54.8 | 51.5 | 53.2 |
| Human | 94.4 | 93.5 | 92.3 | 93.7 |

### Ablation Study

| Configuration | Gemini 2.5 Pro | Qwen3-Omni | Notes |
|--------------|---------------|------------|-------|
| Vision Only | ~55–60% | ~50–55% | Basic visual capability |
| Audio + Vision | ~70–80% | ~50–55% | Gemini gains 10–20 pp |
| Audio Gain | +10–20 pp | 0 or negative | Core fusion capability gap |

### Key Findings

- Adding audio input yields a consistent 10–20 pp improvement for Gemini 2.5 Pro, while gains for Qwen3-Omni are marginal or even negative.
- Error analysis identifies audio perception errors and temporal localization errors as the primary failure modes.
- All models exhibit accuracy degradation as the number of visible speakers increases, with multi-speaker scenes posing the greatest challenge.
- Early open-source audiovisual models (Video-LLaMA, PandaGPT) perform near chance level.

## Highlights & Insights

- **Fusion Capability Diagnosis**: Modality ablation experiments clearly reveal the fusion capability gap across different models.
- **Error Taxonomy**: A systematic categorization of failure types is provided, including visual/audio perception errors, cross-modal attribution errors, and temporal localization errors.
- **Design Rationale**: The authors acknowledge that strong models may partially answer questions via visual cues (e.g., lip motion), framing this as a legitimate capability rather than a benchmark flaw.

## Limitations & Future Work

- In certain tasks, strong vision-only models can answer questions without audio, which, while acknowledged as a valid capability, reduces the necessity of the audio modality.
- All videos are sourced from YouTube, with scenes predominantly from film and television.
- The evaluation currently covers a limited number of open-source audiovisual models.

## Related Work & Insights

- **vs. Video-MME**: Questions in Video-MME can largely be answered using vision alone, whereas AV-SpeakerBench enforces audiovisual fusion.
- **vs. WorldSense**: WorldSense focuses on scene–acoustic matching, while AV-SpeakerBench targets speaker–speech binding.

## Rating

- Novelty: ⭐⭐⭐⭐ Fills a gap in speaker-centric audiovisual reasoning evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers both open- and closed-source models with comprehensive modality ablation and error analysis.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with illuminating case studies.
- Value: ⭐⭐⭐⭐⭐ Provides a much-needed diagnostic tool for the development of audiovisual fusion models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Purify-then-Align: Towards Robust Human Sensing under Modality Missing with Knowledge Distillation from Noisy Multimodal Teacher](purify-then-align_towards_robust_human_sensing_under_modality_missing_with_knowl.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] Venus: Benchmarking and Empowering Multimodal Large Language Models for Aesthetic Guidance and Cropping](venus_benchmarking_and_empowering_multimodal_large_language_models_for_aesthetic.md)
- [\[CVPR 2026\] Towards Multimodal Domain Generalization with Few Labels](towards_multimodal_domain_generalization_with_few_labels.md)
- [\[CVPR 2026\] Tokenization Allows Multimodal Large Language Models to Understand, Generate and Edit Architectural Floor Plans (HouseMind)](tokenization_allows_multimodal_large_language_models_to_understand_generate_and_.md)

</div>

<!-- RELATED:END -->
