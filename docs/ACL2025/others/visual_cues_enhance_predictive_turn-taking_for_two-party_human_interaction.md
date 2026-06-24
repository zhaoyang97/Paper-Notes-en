---
title: >-
  [Paper Note] Visual Cues Enhance Predictive Turn-Taking for Two-Party Human Interaction
description: >-
  [ACL 2025][Turn-Taking] This paper proposes MM-VAP, a multimodal predictive turn-taking model. By incorporating visual cues such as facial expressions, head poses, and gaze direction into a voice predictive model, it improves the hold/shift prediction accuracy from 79% to 84% on a video conferencing corpus.
tags:
  - "ACL 2025"
  - "Turn-Taking"
  - "Multimodal"
  - "Facial Action Units"
  - "Predictive Model"
  - "Video Conferencing"
date: 2026-05-08
content_hash: d14b035e7e14987e
---

# Visual Cues Enhance Predictive Turn-Taking for Two-Party Human Interaction

**Conference**: ACL 2025  
**arXiv**: [2505.21043](https://arxiv.org/abs/2505.21043)  
**Code**: [github.com/russelsa/mm-vap](https://github.com/russelsa/mm-vap)  
**Area**: Other  
**Keywords**: Turn-Taking, Multimodal, Facial Action Units, Predictive Model, Video Conferencing  

## TL;DR

This paper proposes MM-VAP, a multimodal predictive turn-taking model. By incorporating visual cues such as facial expressions, head poses, and gaze direction into a voice predictive model, it improves the hold/shift prediction accuracy from 79% to 84% on a video conferencing corpus.

## Background & Motivation

- **Core Fact**: In two-party conversations, the average silence between turns is only 200 ms, whereas language generation requires at least 600 ms. This indicates that turn-taking is **predictive**—listeners start planning their next turn before the speaker finishes speaking.
- **Limitations of Prior Work**: Almost all predictive turn-taking models (PTTM) rely solely on speech features, neglecting visual cues. While acceptable in telephone scenarios, this is a significant limitation in scenarios where participants can see each other (e.g., video conferencing, face-to-face).
- **Psycholinguistic Evidence**: Research shows that when subjects observe both audio and video, their accuracy in determining turn completion points is higher (Barkhuysen et al., 2008); eyebrow raising can accelerate question recognition (Nota et al., 2023).
- **Research Gap**: It remains unclear whether visual cues can enhance PTTM performance, as related studies are scarce and limited in data scale.

## Method

### Overall Architecture

MM-VAP is extended from the SOTA audio-only model VAP (Ekstedt & Skantze, 2022). VAP uses a Transformer to continuously predict speaking activity in the upcoming 2 seconds (Voice Activity Projection), encoding the activity states of two speakers into 8 binary bins, totaling 256 VAP states. MM-VAP integrates visual features on top of this.

### Key Designs

1. **Visual Feature Extraction**: Uses OpenFace to extract 60-dimensional visual feature vectors frame-by-frame from the video, including:
    - 17 Facial Action Units (FAUs): describing facial muscle movements (e.g., jaw dropping, lip movements) with intensity from 0 to 5.
    - Gaze vectors: a 3D unit vector for each eye.
    - Head position (X, Y, Z) and rotation (roll, pitch, yaw).
    - 15 facial landmarks (eyebrow, jaw, nose, lip regions).

2. **Model Architecture (Late Fusion)**:
    - Audio is processed by a pre-trained feature extractor to obtain 256-dimensional feature vectors (50 Hz).
    - Visual features are projected to 256 dimensions using an MLP, and linearly upsampled from 30 Hz to 50 Hz.
    - First, temporal patterns of audio and video for each speaker are modeled separately using Self-Attention blocks.
    - Then, Cross-Attention blocks are utilized to learn the audio-visual interactions of the same speaker.
    - Subsequently, Cross-Attention blocks are employed to model the cross-modal temporal patterns between the two speakers.
    - A causal mask ensures that the model only predicts from past frames.
    - The total parameters amount to 8.7M (compared to 5.8M for VAP).

3. **ASR Alignment Validation**: This is the first work in PTTM to use automatic speech recognition (ASR) to extract Voice Activity labels instead of manual alignment, representing a scenario closer to real-world deployment. It verifies that the performance degradation caused by ASR on Switchboard is manageable.

### Loss & Training

Cross-entropy loss is used to train the model to output the 256-dimensional softmax distribution matching the target 256 VAP state labels.

## Experiments

### Main Results (Candor Video Conferencing Corpus, 710 hours)

| Model | F₁ (Weighted) | F₁ (Hold) | F₁ (Shift) | Balanced Accuracy |
|------|----------|----------|-----------|-----------|
| Dummy (All Hold) | 0.70 | 0.82 | 0.00 | 50% |
| VAP (Audio-only) | 0.83 | 0.89 | 0.71 | 79% |
| **MM-VAP (Late)** | **0.86** | **0.90** | **0.77** | **83%** |
| **MM-VAP (Early)** | **0.87** | **0.91** | **0.79** | **84%** |

The addition of visual cues improves the Shift F₁ by 6-8 percentage points (0.71 → 0.77/0.79), and the balanced accuracy by 4-5 percentage points.

### Stratified Analysis by Silence Duration (First in PTTM)

| Silence Duration (FTO) | VAP Balanced Accuracy | MM-VAP Balanced Accuracy |
|---------------|--------------|-----------------|
| > 0 ms | 79% | 83% |
| > 250 ms | 79% | 83% |
| > 500 ms | 77% | 81% |
| > 750 ms | 75% | 78% |
| > 1000 ms | 73% | 76% |

MM-VAP outperforms VAP across all silence durations, and the performance of both models declines as the silence duration increases (longer intervals are harder to predict).

### Ablation Study

| Visual Feature Subset | F₁ (Shift) Gain Relative to VAP |
|-------------|---------------------------|
| Full Visual Features | +6-8% |
| Facial Action Units (FAUs) Only | **Largest contribution** |
| Head Pose Only | Improved but small |
| Gaze Direction Only | Improved but small |
| Facial Landmarks Only | Improved but minimal |

Facial expressions (encoded via Facial Action Units) are the most crucial visual cue, which aligns with findings from facial movement analysis—mouth, lip, jaw, and chin movements of the next speaker are significantly enhanced prior to turn transition.

### Key Findings

- **Visual cues are indeed effective**: In mutually visible scenarios, visual features significantly contribute to turn-taking prediction.
- **Facial expressions are the most critical visual cue**: The contribution of FAUs far exceeds that of gaze and head pose.
- **ASR alignment is feasible**: The alignment error caused by automatic speech recognition is about 480 ms, but its impact on PTTM performance is limited.
- **Longer silences are more difficult to predict**: The performance of all models degrades in long-interval scenarios, but visual cues help across all durations.

## Highlights & Insights

- This work systematically validates the value of visual cues for predictive turn-taking on a large-scale corpus (710 hours) for the first time.
- It introduces a novel evaluation method of stratified performance analysis of PTTM by silence duration, which is more informative than a single global metric.
- It demonstrates the feasibility of replacing manual alignment with ASR, significantly lowering the barrier for data annotation.
- The code is open-sourced, facilitating replication and subsequent research.

## Limitations & Future Work

- The performance is only validated in video conferencing scenarios, without testing on face-to-face interactions or more natural settings.
- OpenFace feature extraction failed in some videos (238/1656 sessions), which were excluded.
- Visual feature extraction is performed at the frame level, incurring high computing costs that may not be suitable for real-time deployment.
- Advanced visual feature extractors (such as Transformer-based facial analysis models) were not explored.
- The Candor corpus only consists of casual conversations in American English; thus, the generalizability across cultures and languages remains to be validated.

## Related Work & Insights

- **VAP Model** (Ekstedt & Skantze, 2022): The current SOTA audio-only PTTM, serving as the baseline and foundation for extension in this work.
- **Roddy et al. (2018)**: First to introduce gaze vectors into PTTM, yielding small improvements on the Mahnob corpus (11h).
- **Onishi et al. (2024)**: Extended VAP to support vision, but tested on only 1.5-2 hours of data and required VA labels during inference.
- **Kurata et al. (2023)**: Classified 5-second segments at known turn endings using visual features, which is not a predictive model.
- **Stivers et al. (2009)**: A cross-linguistic study confirming the universality of the average 200 ms silence between conversation turns.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Technical Depth | 4 |
| Experimental Thoroughness | 5 |
| Writing Quality | 5 |
| **Overall Score** | **4.5** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] VAQUUM: Are Vague Quantifiers Grounded in Visual Data?](vaquum_are_vague_quantifiers_grounded_in_visual_data.md)
- [\[ACL 2025\] Unique Hard Attention: A Tale of Two Sides](unique_hard_attention_a_tale_of_two_sides.md)
- [\[CVPR 2026\] Modeling the Visual Ambiguity of Human Sketches](../../CVPR2026/others/modeling_the_visual_ambiguity_of_human_sketches.md)
- [\[ACL 2025\] CONFETTI: Conversational Function-Calling Evaluation Through Turn-Level Interactions](confetti_conversational_function-calling_evaluation_through_turn-level_interacti.md)
- [\[ICCV 2025\] SyncDiff: Synchronized Motion Diffusion for Multi-Body Human-Object Interaction Synthesis](../../ICCV2025/others/syncdiff_synchronized_motion_diffusion_for_multi-body_human-object_interaction_s.md)

</div>

<!-- RELATED:END -->
