---
title: >-
  [Paper Note] T2A-Feedback: Improving Basic Capabilities of Text-to-Audio Generation via Fine-grained AI Feedback
description: >-
  [ACL 2025][Audio & Speech][Text-to-Audio] Proposes three fine-grained AI audio scoring pipelines (event occurrence, event sequence, and acoustic harmony quality) to replace human annotation for constructing a large-scale audio preference dataset, T2A-Feedback (41K prompts, 249K audios). By utilizing preference tuning to enhance the basic capabilities of TTA models, it significantly improves multi-event audio generation quality in both simple (AudioCaps) and complex (T2A-EpicB…
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Text-to-Audio"
  - "AI Feedback"
  - "Preference Tuning"
  - "Multi-event Audio"
  - "Fine-grained Evaluation"
date: 2026-05-08
content_hash: 28e1de11ed58d516
---

# T2A-Feedback: Improving Basic Capabilities of Text-to-Audio Generation via Fine-grained AI Feedback

**Conference**: ACL 2025  
**arXiv**: [2505.10561](https://arxiv.org/abs/2505.10561)  
**Code**: [https://T2Afeedback.github.io](https://T2Afeedback.github.io)  
**Area**: Audio & Speech  
**Keywords**: Text-to-Audio, AI Feedback, Preference Tuning, Multi-event Audio, Fine-grained Evaluation  

## TL;DR
Proposes three fine-grained AI audio scoring pipelines (event occurrence, event sequence, and acoustic harmony quality) to replace human annotation for constructing a large-scale audio preference dataset, T2A-Feedback (41K prompts, 249K audios). By utilizing preference tuning to enhance the basic capabilities of TTA models, it significantly improves multi-event audio generation quality in both simple (AudioCaps) and complex (T2A-EpicBench) scenarios.

## Background & Motivation
**Background**: Text-to-Audio (TTA) generation models can produce diverse audio but perform poorly in complex multi-event scenarios—failing to fully include all described events, follow event order, or organize multiple events harmoniously.

**Limitations of Prior Work**: (a) Existing evaluation metrics like CLAP only assess global audio-text alignment, failing to evaluate event occurrence, sequence, and harmony in a fine-grained manner; (b) Human annotation of audio preference data is extremely costly and unscalable; (c) There is a lack of evaluation benchmarks specifically targeting complex multi-event or narrative scenarios.

**Key Challenge**: Advanced TTA applications (e.g., narrative audio, video dubbing) require precise multi-event control, yet the "basic capabilities" of models (event inclusion, sequence, harmony) remain insufficient, necessitating targeted enhancements for each basic capability.

**Goal**: To replace human annotation with automated AI feedback, constructing fine-grained preference data and evaluation metrics for each of the three basic capabilities of TTA models.

**Key Insight**: Deconstructing TTA preference tuning into three independently evaluable dimensions (event occurrence, event sequence, acoustic harmony) and designing dedicated AI scoring pipelines for each dimension.

**Core Idea**: Three-dimensional fine-grained AI scoring $\rightarrow$ large-scale preference dataset $\rightarrow$ preference tuning to enhance basic capabilities.

## Method

### Overall Architecture
(1) Design three AI scoring pipelines: Event Occurrence Score (EOS), Event Sequence Score (ESS), and Acoustic & Harmonic Quality Score (AHQ); (2) Use these three pipelines to score LLM-generated audio at scale, constructing the T2A-Feedback preference dataset (41K prompts, 249K audios); (3) Enhance existing TTA models using preference tuning (a DPO variant). Additionally, the study constructs T2A-EpicBench to evaluate complex scenarios.

### Key Designs

1. **Event Occurrence Score (EOS)**:

    - **Function**: Verifies whether each event described in the text occurs in the audio.
    - **Mechanism**: Decomposes the prompt into independent event descriptions, calculates the semantic matching score (based on CLAP) between each event and the audio separately, and treats low-scoring events as missing.
    - **Design Motivation**: Global matching via CLAP cannot distinguish between "containing all events" versus "containing only partial events".

2. **Event Sequence Score (ESS)**:

    - **Function**: Verifies whether the sequence of events in the audio aligns with the text description.
    - **Mechanism**: Uses an audio event detection model to estimate the onset and offset times of each event, then compares them with the event sequence in the text.
    - **Design Motivation**: Multi-event audio must not only contain all events but also organize them in the correct order.

3. **Acoustic & Harmonic Quality (AHQ)**:

    - **Function**: Evaluates the overall acoustic quality and the harmony among multiple events in the audio.
    - **Mechanism**: Manually annotates the acoustic and harmonic quality of a subset of audios and trains an automatic predictor.
    - **Design Motivation**: Even if all events exist in the correct order, acoustically unharmonious transitions (e.g., abrupt switching, noise interference) represent low quality.

4. **T2A-EpicBench Evaluation Benchmark**:

    - **Function**: Evaluates advanced capabilities of TTA models under long-format, multi-event, and narrative settings.
    - **Mechanism**: Constructs a test set comprising long fantasy, narrative, and story descriptions, which is significantly more challenging than the simple descriptions in AudioCaps.
    - **Design Motivation**: Existing benchmarks (like AudioCaps) have overly simplistic descriptions, which are insufficient to evaluate complex applications.

### Loss & Training
- Preference tuning based on a DPO variant.
- The three-dimensional scores are used individually or jointly to construct preference pairs.
- Base Model: Make-an-Audio 2 (diffusion-based method).

## Key Experimental Results

### Comparison with Existing Evaluation Metrics (Correlation with Human Preferences)

| Metric | Correlation with Human Preference | Description |
|---|---|---|
| CLAP (Global Matching) | Medium | Unable to evaluate at a fine-grained level |
| FAD/IS (Distributional Metrics) | Low | Does not evaluate individual samples |
| **EOS (Event Occurrence)** | **High** | Fine-grained event verification |
| **ESS (Event Sequence)** | **High** | Sequence verification |
| **AHQ (Acoustic Harmony)** | **High** | Quality prediction |

### Preference Tuning Performance

| Setting | AudioCaps (Simple) | T2A-EpicBench (Complex) |
|---|---|---|
| Make-an-Audio 2 (Baseline) | Baseline | Baseline |
| **+ T2A-Feedback Tuning** | **Significant Gain** | **Significant Gain** |

### Key Findings
- The correlation of the three AI scoring pipelines with human preferences is significantly better than CLAP, validating the necessity of fine-grained evaluation.
- Preference tuning is effective in both simple and complex scenarios—enhancing basic capabilities leads to an "emergent" improvement in advanced performance.
- T2A-EpicBench reveals severe deficiencies of current models in narrative audio, with most models failing almost completely on long-form descriptions.
- The three dimensions can be used independently or jointly for tuning, with each dimension providing complementary benefits.

## Highlights & Insights
- The strategy of **"enhancing basic capabilities $\rightarrow$ emerging advanced performance"** is compelling—it removes the need for specialized training on complex scenarios, as focusing on the three basic dimensions is sufficient.
- The **three-dimensional fine-grained scoring** is far more informative than CLAP's one-dimensional global scoring, establishing a new standard for TTA evaluation.
- T2A-Feedback (249K audios) is the first large-scale TTA preference dataset, filling a critical gap in the field.
- T2A-EpicBench pushes TTA evaluation into the "narrative/multi-event" era.
- The methodology of replacing human annotation with AI feedback is transferrable to other generative domains (e.g., fine-grained evaluation in text-to-video generation).

## Limitations & Future Work
- The accuracy of the AI scoring pipelines themselves limits preference data quality, especially since AHQ training data is limited.
- Preference tuning is only validated on Make-an-Audio 2, leaving its effectiveness on other TTA models unconfirmed.
- The evaluation of T2A-EpicBench still heavily relies on automatic metrics, whereas human evaluation remains more reliable for long audios.
- The Event Sequence Score depends on the accuracy of the audio event detection model, meaning detection errors propagate to the final score.

## Related Work & Insights
- **vs Tango2**: Tango2 uses CLAP for global preference ranking, whereas T2A-Feedback utilizes three-dimensional fine-grained scoring, which offers far richer information.
- **vs FlashAudio**: FlashAudio optimizes inference speed, while T2A-Feedback optimizes generation quality—constituting complementary directions.
- **vs RLHF/DPO in LLM**: Porting mature preference tuning methods from the LLM domain to TTA represents a successful cross-domain methodological transfer.

## Rating
- Novelty: ⭐⭐⭐⭐ The three-dimensional AI scoring pipeline is novel and practical; the large-scale preference dataset is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated via score correlation, preference tuning, and a new benchmark, though it is only evaluated on a single base model.
- Writing Quality: ⭐⭐⭐⭐ The three-dimensional deconstruction is clear and the motivation is well-justified.
- Value: ⭐⭐⭐⭐⭐ The triple contribution of the dataset, evaluation metrics, and the benchmark significantly advances the TTA field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SegTune: Structured and Fine-Grained Control for Song Generation](../../ACL2026/audio_speech/segtune_structured_and_fine-grained_control_for_song_generation.md)
- [\[CVPR 2026\] FoleyDirector: Fine-Grained Temporal Steering for Video-to-Audio Generation via Structured Scripts](../../CVPR2026/audio_speech/foleydirector_fine-grained_temporal_steering_for_video-to-audio_generation_via_s.md)
- [\[ACL 2025\] SpeechWeave: Diverse Multilingual Synthetic Text & Audio Data Generation Pipeline for Training Text to Speech Models](speechweave_diverse_multilingual_synthetic_text_audio_data_generation_pipeline_f.md)
- [\[ACL 2026\] FIGMA: Towards Fine-Grained Music Retrieval](../../ACL2026/audio_speech/figma_towards_fine-grained_music_retrieval.md)
- [\[ACL 2025\] Double Entendre: Robust Audio-Based AI-Generated Lyrics Detection via Multi-View Fusion](double_entendre_robust_audio-based_ai-generated_lyrics_detection_via_multi-view_.md)

</div>

<!-- RELATED:END -->
