---
title: >-
  [Paper Note] egoEMOTION: Egocentric Vision and Physiological Signals for Emotion and Personality Recognition in Real-World Tasks
description: >-
  [NeurIPS 2025][Video Understanding][egocentric vision] This paper introduces egoEMOTION — the first dataset combining egocentric vision (Meta Project Aria glasses) with physiological signals for emotion and personality r…
tags:
  - "NeurIPS 2025"
  - "Video Understanding"
  - "egocentric vision"
  - "emotion recognition"
  - "personality"
  - "physiological signals"
  - "Project Aria"
date: 2026-05-08
content_hash: de8c7453a1a93ac7
---

# egoEMOTION: Egocentric Vision and Physiological Signals for Emotion and Personality Recognition in Real-World Tasks

**Conference**: NeurIPS 2025
**arXiv**: [2510.22129](https://arxiv.org/abs/2510.22129)  
**Code**: Available (open-source dataset + baseline implementation)  
**Area**: Affective Computing / Egocentric Vision / Multimodal Dataset
**Keywords**: egocentric vision, emotion recognition, personality, physiological signals, Project Aria

## TL;DR
This paper introduces egoEMOTION — the first dataset combining egocentric vision (Meta Project Aria glasses) with physiological signals for emotion and personality recognition. It encompasses 43 participants, 50+ hours of recordings, and 16 tasks, and demonstrates that egocentric vision signals (particularly eye-tracking features) outperform conventional physiological signals for emotion prediction in real-world scenarios.

## Background & Motivation

**Background**: Egocentric vision has established large-scale benchmarks (Ego4D, EPIC-KITCHENS), while emotion recognition has relied on physiological signals collected in laboratory settings (DEAP, AMIGOS, etc.).

**Limitations of Prior Work**: (a) Egocentric vision benchmarks ignore participants' emotional states, assuming affective neutrality; (b) existing emotion datasets are confined to laboratory settings with low ecological validity; (c) the only publicly available mobile eye-tracking emotion dataset, eSEE-d, covers only 4 emotion categories, provides no physiological signals, and requires a fixed head position.

**Key Challenge**: Emotion and personality are intrinsic drivers of behavior, yet egocentric vision systems cannot model these internal states.

**Goal**: To construct a high-ecological-validity multimodal affective dataset and demonstrate that egocentric vision signals alone are sufficient for emotion prediction.

**Core Idea**: Egocentric glasses signals — especially eye-tracking — are more effective than conventional physiological signals for emotion prediction in real-world settings.

## Method

### Overall Architecture
43 participants wore Project Aria glasses and physiological sensors while completing 16 tasks (9 emotion-eliciting videos + 7 naturalistic activities). Three benchmarks are defined: (1) binary classification of continuous emotion dimensions V/A/D, (2) 9-class discrete emotion recognition, and (3) binary classification of Big Five personality traits.

### Key Designs

1. **16-Task Protocol**:

    - Session A: 9 video clips (~48s each) corresponding to 8 emotions from Mikels' Wheel plus a neutral condition.
    - Session B: Flappy Bird (frustration), tasting unpleasant gummies (disgust), Jenga (socially tense), drawing with music (relaxation), writing a sad letter (sadness), Slenderman horror game (fear), telling jokes (amusement).

2. **Multi-Level Annotation Scheme**:

    - Emoti-SAM 7-point scale for collecting V/A/D ratings.
    - Weighted Mikels' Wheel: 100% distributed across 9 emotion categories (in 10% increments) to capture the relative intensity of mixed emotions.
    - BFI-2 personality questionnaire.

3. **Sensor Configuration**:

    - Aria glasses: eye-tracking video (640×480@90fps), POV camera (1408×1408@10fps), dual IMU (1000Hz + 800Hz), nose-pad PPG (128Hz).
    - External sensors: ECG (1024Hz), EDA (256Hz), RSP (400Hz).

4. **612-Dimensional Feature Extraction**:

    - ECG/PPG: 77 features; EDA: 31 features; RSP: 14 features.
    - Egocentric-derived features: pupil size, pixel intensity, Fisherface, gaze direction, blink detection, LBP-TOP micro-expressions.
    - 15 statistical descriptors per signal.

### Baseline Methods
- Continuous emotion: SVM-RBF + LOSO cross-validation.
- Discrete emotion: Random Forest + SelectKBest (top-10) + LOSO.
- Personality: Random Forest + SelectKBest + LOSO.
- Deep learning: CNN and WER (Transformer), 5-fold cross-validation.

## Key Experimental Results

### Main Results — Modality Comparison (F1 Score)

| Benchmark | Wearable (ECG/EDA/RSP) | Egocentric Glasses | Full Fusion | Chance Baseline |
|---|---|---|---|---|
| Continuous Emotion (V/A/D avg.) | 0.70 | **0.74** | **0.75** | 0.59 |
| Discrete Emotion (9-class avg.) | 0.24 | **0.46** | **0.46** | 0.11 |
| Personality (Big Five avg.) | 0.50 | **0.57** | **0.59** | 0.53 |

### Classical Methods vs. Deep Learning

| Benchmark | Classical (All) | CNN (All) | Transformer WER (All) |
|---|---|---|---|
| Continuous Emotion | **0.75** | 0.68 | 0.60 |
| Discrete Emotion | **0.46** | 0.22 | 0.21 |
| Personality | **0.59** | — | 0.47 |

### Key Findings
- **Egocentric glasses signals consistently outperform conventional physiological sensors**: the advantage is most pronounced for discrete emotion recognition (0.46 vs. 0.24).
- Gaze features are most informative for continuous emotion; pixel intensity is most effective for discrete emotion; IMU features generalize best across tasks.
- Classical ML substantially outperforms deep learning — deep models severely overfit given the small dataset (43 participants × 16 tasks).
- Personality prediction is the most difficult benchmark, approaching the random baseline.

## Highlights & Insights
- **Weighted Mikels' Wheel Annotation**: enables quantification of the relative intensity of mixed emotions, providing richer labels than simple multi-class selection. This scheme is transferable to large-scale video affective annotation pipelines.
- **Eye-tracking video > conventional physiological signals**: future affective recognition systems may not require contact sensors such as ECG/EDA — an eye-tracking-equipped glasses device may suffice.
- **Fisherface features**: applying PCA+LDA to eye-tracking video frames as a low-cost visual descriptor yields competitive performance.

## Limitations & Future Work
- The sample of 43 participants (predominantly university students) is relatively small.
- Deep learning performance is substantially inferior to classical methods; few-shot or meta-learning strategies for small-sample settings warrant exploration.
- Annotations are collected only after each task concludes rather than as continuous time-series labels.
- Task design minimizes body movement, limiting generalization to unrestricted daily-life scenarios.
- The representational capacity of large visual foundation models (e.g., VideoMAE, InternVideo) remains unexplored.
- The experimenter's presence (seated behind a curtain) may have influenced participants' natural behavior.
- All participants are healthy individuals; generalizability to clinical populations (e.g., anxiety, depression) is unknown.

## Related Work & Insights
- **vs. DEAP/AMIGOS/ASCERTAIN**: These canonical affective datasets are collected in laboratory settings using stationary EEG/ECG equipment, resulting in low ecological validity. egoEMOTION is the first to use a mobile egocentric device in semi-naturalistic settings, more closely approximating real-world use.
- **vs. Ego4D/EPIC-KITCHENS**: Large-scale egocentric vision datasets that lack affective annotations. egoEMOTION addresses this gap but is substantially smaller in scale.
- **vs. eSEE-d**: The only publicly available mobile eye-tracking emotion dataset, yet limited to 4 emotion categories, requiring a chin rest and providing no physiological signals. egoEMOTION significantly extends coverage along every dimension.
- **vs. K-EmoCon/EmoPairCompete**: Emotions are elicited naturalistically in social contexts, but egocentric vision and eye-tracking data are absent. egoEMOTION offers more comprehensive sensor coverage.
- The annotation methodology of egoEMOTION could be applied to extend large-scale datasets such as Ego4D with affective labels.
- The effectiveness of eye-tracking features suggests that built-in eye trackers in mixed reality (MR) devices may already be sufficient to support real-time affective inference.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First multimodal dataset combining egocentric vision and affective signals, filling an important gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 612-dimensional features, cross-modal comparisons, classical vs. deep learning analysis; limited by small participant count.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Exceptionally clear for a dataset paper, with detailed descriptions of experimental design.
- **Value**: ⭐⭐⭐⭐ — The open-source dataset and baselines will advance future research, though the scale constrains direct application.

<!-- NeurIPS 2025 | video_understanding -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] egoPPG: Heart Rate Estimation from Eye-Tracking Cameras in Egocentric Systems to Benefit Downstream Vision Tasks](../../ICCV2025/video_understanding/egoppg_heart_rate_estimation_from_eye-tracking_cameras_in_egocentric_systems_to_.md)
- [\[NeurIPS 2025\] Lattice Boltzmann Model for Learning Real-World Pixel Dynamicity](lattice_boltzmann_model_for_learning_real-world_pixel_dynamicity.md)
- [\[NeurIPS 2025\] LiveStar: Live Streaming Assistant for Real-World Online Video Understanding](livestar_live_streaming_assistant_for_real-world_online_video_understanding.md)
- [\[NeurIPS 2025\] Grounding Foundational Vision Models with 3D Human Poses for Robust Action Recognition](grounding_foundational_vision_models_with_3d_human_poses_for_robust_action_recog.md)
- [\[NeurIPS 2025\] EgoGazeVQA: Egocentric Gaze-Guided Video Question Answering Benchmark](egogazevqa_egocentric_gaze_guided_video_question_answering.md)

</div>

<!-- RELATED:END -->
