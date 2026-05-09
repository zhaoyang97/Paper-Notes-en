---
title: >-
  [Paper Note] Reading Recognition in the Wild
description: >-
  [NeurIPS 2025][Multimodal VLM][Reading recognition] This paper introduces a novel reading recognition task and the first large-scale multimodal "reading-in-the-wild" dataset (100 hours). A lightweight Transformer model fusing three complementary modalities—RGB, gaze, and IMU—enables real-time reading detection on smart glasses.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - Reading recognition
  - wearable devices
  - multimodal fusion
  - eye tracking
  - egocentric vision
date: 2026-05-08
content_hash: e12cfb704d9eac31
---

# Reading Recognition in the Wild

**Conference**: NeurIPS 2025
**arXiv**: [2505.24848](https://arxiv.org/abs/2505.24848)
**Code**: [Project Aria](https://www.projectaria.com/datasets/reading-in-the-wild/)
**Area**: Multimodal VLM
**Keywords**: Reading recognition, wearable devices, multimodal fusion, eye tracking, egocentric vision

## TL;DR

This paper introduces a novel reading recognition task and the first large-scale multimodal "reading-in-the-wild" dataset (100 hours). A lightweight Transformer model fusing three complementary modalities—RGB, gaze, and IMU—enables real-time reading detection on smart glasses.

## Background & Motivation

Smart glasses, as the future platform for AI personal assistants, must understand users' physical context. Reading is one of the most important means of human information acquisition; enabling AI to know when a user is reading is essential for building context-aware personal assistants.

Reading recognition faces two core challenges:

**Ill-posedness of the problem**: The presence of text in the visual field does not imply that the user is reading (e.g., passing a billboard), making visual information alone inherently ambiguous.

**Efficiency constraints**: Wearable devices face hardware limitations in power consumption, bandwidth, and thermal dissipation, precluding continuous execution of heavy models such as OCR or VLMs.

Limitations of prior work:
- Gaze-based methods (e.g., Kelton et al.) rely on hand-crafted features (e.g., fixation and saccade detection) and are evaluated only in controlled environments.
- Egocentric video datasets (Ego4D, Ego-Exo4D) contain very few and non-diverse reading samples.
- Cognitive science datasets (ZuCo, InteRead) are confined to controlled screen-reading scenarios and lack RGB information.

Reading recognition can serve as a lightweight proxy signal, triggering heavy OCR/VLM models only when reading is detected, thereby significantly reducing computational overhead.

## Method

### Overall Architecture

At time $t$, the model predicts a reading confidence score $s_t \in [0,1]$, given inputs from three modalities: gaze trajectory $g$, RGB image crop $I_t$, and head pose (IMU) $z$. The framework is a flexible multimodal Transformer that accepts any subset of modalities at inference time.

### Key Designs

1. **Gaze encoding**: The temporal derivative of 3D gaze points is used as the input representation (rather than 2D projections or retinal images), encoded into feature tokens via three 1D convolutional layers (kernel=9, dim=32). Differential processing encourages the model to focus on gaze movement patterns rather than absolute positions, improving generalization.

2. **RGB cropping strategy**: Based on the fact that the fovea subtends approximately 2° of visual angle, only a 5° FoV region (64×64 pixels, comprising merely 1/484 of the full image) centered on the gaze point is cropped and encoded via three 2D convolutional layers. This design provides sufficient visual context while greatly reducing computation and privacy exposure.

3. **Head pose (IMU/VIO)**: The 6DoF output of visual-inertial odometry (VIO) is used to characterize head motion patterns. Though limited in isolation, it resolves ambiguities (e.g., distinguishing reading from horizontal head turns).

4. **Modality dropout training**: Entire modalities are randomly dropped during training, ensuring: (i) less frequently used modalities are adequately trained; and (ii) the model remains functional at inference even when certain modalities are unavailable.

5. **Cross-lingual generalization**: For languages with non-standard reading directions (e.g., Chinese ↓, Arabic ←), gaze data is rotated 90° or horizontally flipped at inference time, enabling adaptation without retraining.

### Loss & Training

- Binary cross-entropy loss for reading / non-reading classification
- Adam optimizer, learning rate $1 \times 10^{-3}$, trained for 10 epochs
- Modality dropout equalizes the usage probability across one-, two-, and three-modality configurations
- A small proportion of rotation augmentation is applied during training to handle vertical text
- Total parameter count is only 137K; training requires a single GPU

## Key Experimental Results

### Main Results

| Modality Combination | Accuracy (%) | F1 (%) | $P_{R=0.9}$ (%) |
|---|---|---|---|
| Gaze only | 82.3 | 84.5 | 79.8 |
| RGB only | 82.2 | 83.7 | 76.5 |
| IMU only | 74.7 | 80.0 | 71.9 |
| Gaze+RGB | 84.9 | 86.5 | 83.6 |
| Gaze+IMU | 83.5 | 85.2 | 82.3 |
| RGB+IMU | 86.0 | 87.8 | 87.3 |
| **All three** | **86.9** | **88.1** | **88.0** |

The three-modality model improves accuracy by +4.6% over the best single-modality baseline, confirming inter-modal complementarity.

### Ablation Study

| Configuration | Accuracy (%) | F1 (%) | Notes |
|---|---|---|---|
| 3D point (d/dt) | 82.3 | 84.5 | ✓ Optimal gaze representation |
| 2D projection | 79.8 | 81.3 | Loses 3D information |
| 60 Hz sampling | 82.3 | 84.5 | High frequency optimal |
| 10 Hz sampling | 80.4 | 82.9 | Downsampled yet viable |
| FoV 5° (64px) | 82.2 | 83.7 | Optimal efficiency–accuracy trade-off |
| XS model (6K params) | 82.0 | 83.6 | Extremely small model still effective |
| XL model (1M params) | 88.5 | 90.1 | Larger model yields better performance |

### Generalization Results

| Scenario | Accuracy (%) |
|---|---|
| Columbus zero-shot (three modalities) | 82.9 |
| Bengali (left→right) | 93.0 |
| Chinese (↓) + rotation augmentation | 85.1 (+49.6) |
| Arabic (←) + flip augmentation | 51.5 (+30.5) |
| Seattle→EGTEA generalization | 87.7 |
| EGTEA→Seattle generalization | 62.9 |

### Key Findings

- Failure cases of gaze and RGB are complementary: gaze excels in low-light and long-distance scenarios, while RGB is better at detecting reading of short text.
- IMU consistently improves performance when combined with other modalities (+1.3%–+2.6%).
- Accuracy on hard negative samples (text present but not being read) is only 74.7%, constituting the primary challenge.
- Real-time detection latency for the Gaze+RGB+IMU model is approximately 0.72 seconds.
- The S model with only 66K parameters achieves 86.3% accuracy.

## Highlights & Insights

1. **Novel task definition**: Reading recognition is extended from controlled laboratory settings to real-world in-the-wild scenarios, establishing the first large-scale benchmark of its kind.
2. **Extreme efficiency**: Only 0.2% of the image area is cropped; the 137K-parameter model can run for over 4 hours on Aria Gen 2 glasses.
3. **Scalable data collection protocol**: Voice annotations ("Start reading!" / "Stop reading!") combined with WhisperX automatic timestamp extraction eliminate manual labeling.
4. **Privacy-oriented design**: Using gaze alone avoids capturing full RGB images, reducing visual intrusion on bystanders.

## Limitations & Future Work

- Performance degrades on atypical reading scenarios (reading while writing: 55.5%; non-text reading: 65.8%).
- Short text (e.g., road signs) fails to produce distinctive gaze patterns within the temporal window.
- Generalization to right-to-left scripts such as Arabic remains limited (only 51.5% after flipping augmentation).
- The 2-second temporal window may be insufficient to distinguish fine-grained reading modes such as intensive reading, skimming, and scanning.

## Related Work & Insights

- This work bridges computer vision (egocentric activity recognition) and cognitive science (reading comprehension research).
- It provides a practical demonstration of an "on-demand activation" paradigm for future multimodal perception: lightweight models serve as gates, while heavy models are invoked selectively.
- The released dataset is expected to advance research on reading assistance tools for children with dyslexia and individuals with low vision.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First definition of the in-the-wild reading recognition task, with a new dataset and benchmark
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive ablation studies, cross-lingual and cross-dataset generalization, and real-time deployment validation
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-articulated task motivation
- Value: ⭐⭐⭐⭐⭐ Directly applicable to smart glasses products; backed by Meta's internal project infrastructure

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild](../../ICCV2025/multimodal_vlm/molparser_end-to-end_visual_recognition_of_molecule_structures_in_the_wild.md)
- [\[NeurIPS 2025\] Recognition through Reasoning: Reinforcing Image Geo-localization with Large Vision-Language Models](recognition_through_reasoning_reinforcing_image_geo-localization_with_large_visi.md)
- [\[NeurIPS 2025\] Uni-MuMER: Unified Multi-Task Fine-Tuning of Vision-Language Model for Handwritten Mathematical Expression Recognition](uni-mumer_unified_multi-task_fine-tuning_of_vision-language_model_for_handwritte.md)
- [\[ICCV 2025\] A Quality-Guided Mixture of Score-Fusion Experts Framework for Human Recognition](../../ICCV2025/multimodal_vlm/a_qualityguided_mixture_of_scorefusion_experts_framework_for.md)
- [\[ICCV 2025\] ProbRes: Probabilistic Jump Diffusion for Open-World Egocentric Activity Recognition](../../ICCV2025/multimodal_vlm/probres_probabilistic_jump_diffusion_for_open-world_egocentric_activity_recognit.md)

</div>

<!-- RELATED:END -->
