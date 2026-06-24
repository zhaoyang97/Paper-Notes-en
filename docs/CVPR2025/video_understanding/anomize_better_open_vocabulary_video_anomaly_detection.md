---
title: >-
  [Paper Note] "Anomize: Better Open Vocabulary Video Anomaly Detection"
description: >-
  [CVPR2025][Video Understanding][LSTM] Academic paper note for "Anomize: Better Open Vocabulary Video Anomaly Detection".
tags:
  - CVPR2025
  - Video Understanding
  - LSTM
  - GPT-4
date: 2026-05-08
content_hash: f7c20fc93d9d6fa3
---
# Anomize: Better Open Vocabulary Video Anomaly Detection

**Conference**: CVPR 2025  
**Institution**: Wuhan University / Fudan University / Peking University  
**Keywords**: Video Anomaly Detection, Open Vocabulary, LSTM, GPT-4, Two-Stream Architecture  

## Background & Motivation

Video Anomaly Detection (VAD) aims to identify events that deviate from normal patterns in videos. Most traditional methods operate under a closed-vocabulary setting, where all possible anomaly categories are known during training. However, in real-world scenarios, the types of anomalous events are infinite—new criminal techniques, rare accident types, and sudden public emergencies cannot be enumerated in advance.

**Open-vocabulary video anomaly detection** is a more practical setting: the model must not only detect anomalies of known categories but also identify and classify novel anomaly types never seen during training. This formulation faces three major challenges:

**Ambiguity of anomaly descriptions**: The same category of anomaly (e.g., "violence") can manifest very differently in different contexts, and simple category names cannot capture this diversity.

**Insufficient temporal modeling**: Vision-language models like CLIP excel at single-frame understanding, but anomalies typically require temporal context (e.g., "accelerating suddenly and then crashing").

**Generalization from known to unknown**: How to train on known anomalies while maintaining detection capabilities for novel anomalies?

Existing methods (such as LAVAD and OVVAD) directly use CLIP features for anomaly detection without fully utilizing the knowledge of large language models to enrich textual descriptions of anomalies, and they lack effective temporal modeling mechanisms.

## Method

### Overall Architecture

Anomize adopts a text-augmented two-stream architecture that decomposes video anomaly detection into two complementary streams: a dynamic stream to capture temporal evolution patterns and a static stream to match concept-level semantics.

### Component 1: LSTM Temporal Encoder

Traditional methods perform classification directly on CLIP frame features, which discards temporal information. Anomize introduces a bidirectional LSTM to encode the frame sequence:

$$h_t = 	ext{BiLSTM}([\overrightarrow{h_t}; \overleftarrow{h_t}]) = 	ext{BiLSTM}(f_{	ext{CLIP}}(I_t), h_{t-1})$$

The hidden states of the LSTM accumulate historical information, enabling the model to comprehend temporal patterns such as "normal walking → sudden running → collision".

### Component 2: GPT-4 Group-Guided Text Encoding

This is the most innovative part of Anomize. Traditional methods use fixed category names (e.g., "fighting") as text queries, which are overly simplistic. Anomize utilizes GPT-4 to generate rich anomaly descriptions through three steps:

**Step 1 - Group**: Group anomaly categories by semantic similarity
- For example: {fighting, robbery, shooting} → "interpersonal violence" group

**Step 2 - Describe**: GPT-4 generates detailed, multi-perspective descriptions for each group
- "interpersonal violence" → describes visual features, temporal patterns, environmental cues, etc.

**Step 3 - Encode**: Convert descriptions into feature vectors using a CLIP text encoder

This group-describe strategy produces richer and more discriminative textual representations.

### Component 3: Text-Augmented Two-Stream Architecture

**Dynamic Stream**: LSTM-encoded temporal features + Text Augmenter module
- The Text Augmenter injects text description information into video features via cross-attention
- Outputs the dynamic anomaly score: $s_{	ext{dyn}} = 	ext{MLP}(	ext{CrossAttn}(h_t, T_{	ext{desc}}))$

**Static Stream**: Concept bank + TopK matching
- Pre-constructs an anomaly concept bank (multiple descriptive features for each anomaly category)
- For each frame feature, computes the cosine similarity with all features in the concept bank
- Takes the average of the TopK highest similarities as the static anomaly score

Final score: $s = lpha \cdot s_{	ext{dyn}} + (1-lpha) \cdot s_{	ext{static}}$

### Two-Stage Training

| Stage | Task | Epoch | Learning Rate | Purpose |
|------|------|-------|--------|------|
| Stage 1 | Anomaly Classification | 16 | 1e-4 | Learn to distinguish different anomaly categories |
| Stage 2 | Anomaly Detection | 64 | 5e-5 | Learn to distinguish normal vs. abnormal |

Stage 1 trains the dynamic stream and the Text Augmenter using classification loss, and Stage 2 fine-tunes the entire architecture using Multiple Instance Learning (MIL) loss.

## Key Experimental Results

### Main Results

| Method | XD-Violence AP | XD-Violence Acc | UCF-Crime AUC |
|------|---------------|-----------------|---------------|
| CLIP baseline | 43.68% | 64.68% | 78.32% |
| LAVAD | 55.40% | 79.15% | 81.20% |
| OVVAD | 61.53% | 83.76% | 82.95% |
| **Anomize** | **69.31%** | **90.29%** | **84.49%** |
| Gain vs. OVVAD | +7.78 | +6.53 | +1.54 |

### Open-Vocabulary Capability

On the novel anomaly category classification task (anomaly types unseen during training):
- Anomize: +56.53% compared to the best baseline
- Demonstrating that the GPT-4 group-description strategy provides excellent textual anchors for novel anomalies

### Ablation Study

| Configuration | XD-Violence AP |
|------|---------------|
| Dynamic stream only | 62.15% |
| Static stream only | 58.43% |
| Two-stream (w/o GPT-4 description) | 64.87% |
| Two-stream (with GPT-4 description) | **69.31%** |

The GPT-4 descriptions yield a 4.44% AP gain, demonstrating the importance of rich textual descriptions.

## Highlights & Insights

1. **Group-guided text encoding**: Introduces LLM-generated structured anomaly descriptions to VAD for the first time.
2. **Complementary dynamic and static streams**: Semantic temporal LSTM captures dynamic patterns, while concept bank matching provides static semantics.
3. **Two-stage training strategy**: Curriculum learning from classification to detection ensures the model first learns effective category representations.

## Limitations & Future Work

- Reliance on GPT-4 for description generation increases deployment costs.
- The long-range dependency modeling capability of LSTM is limited, potentially leading to degraded performance on ultra-long videos.
- The concept bank requires manual or LLM definition, necessitating additional adaptation when extending to entirely new domains.

## Summary

Anomize proposes an end-to-end open-vocabulary video anomaly detection framework. Its core innovation lies in leveraging the knowledge of GPT-4 to enrich the textual representation of anomalies, achieving unified detection of both known and unknown anomalies through LSTM temporal encoding and a two-stream architecture. The 25.61% accuracy improvement on XD-Violence is particularly impressive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Attention to Trajectory: Trajectory-Aware Open-Vocabulary Tracking](../../ICCV2025/video_understanding/attention_to_trajectory_trajectory-aware_open-vocabulary_tracking.md)
- [\[ICCV 2025\] Learning to Generalize Without Bias for Open-Vocabulary Action Recognition](../../ICCV2025/video_understanding/learning_to_generalize_without_bias_for_open-vocabulary_action_recognition.md)
- [\[ICLR 2026\] Language-guided Open-world Video Anomaly Detection under Weak Supervision](../../ICLR2026/video_understanding/language-guided_open-world_video_anomaly_detection_under_weak_supervision.md)
- [\[CVPR 2025\] BiM-VFI: Bidirectional Motion Field-Guided Frame Interpolation for Video with Non-uniform Motions](bim-vfi_bidirectional_motion_field-guided_frame_interpolation_for_video_with_non.md)
- [\[ICLR 2026\] Video-STAR: Reinforcing Open-Vocabulary Action Recognition with Tools](../../ICLR2026/video_understanding/video-star_reinforcing_open-vocabulary_action_recognition_with_tools.md)

</div>

<!-- RELATED:END -->
