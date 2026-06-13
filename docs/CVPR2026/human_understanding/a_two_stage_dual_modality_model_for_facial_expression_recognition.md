---
title: >-
  [Paper Note] A Two-Stage Dual-Modality Model for Facial Expression Recognition
description: >-
  [CVPR 2026][Human Understanding][Facial Expression Recognition] A two-stage dual-modality framework for facial expression recognition is proposed: Stage I adapts a DINOv2 encoder on external datasets via padding-aware au…
tags:
  - "CVPR 2026"
  - "Human Understanding"
  - "Facial Expression Recognition"
  - "DINOv2"
  - "Audio-Visual Fusion"
  - "Mixture of Experts"
  - "Data Augmentation"
date: 2026-05-08
content_hash: 1c87c893350a27cc
---

# A Two-Stage Dual-Modality Model for Facial Expression Recognition

**Conference**: CVPR 2026
**arXiv**: [2603.12221](https://arxiv.org/abs/2603.12221)  
**Code**: None  
**Area**: Human Understanding / Facial Expression Recognition
**Keywords**: Facial Expression Recognition, DINOv2, Audio-Visual Fusion, Mixture of Experts, Data Augmentation

## TL;DR

A two-stage dual-modality framework for facial expression recognition is proposed: Stage I adapts a DINOv2 encoder on external datasets via padding-aware augmentation and a training-only MoE head; Stage II performs frame-level audio-visual expression classification using multi-scale facial crops, Wav2Vec 2.0 audio features, and a gated fusion module, achieving 0.5368 Macro-F1 in the ABAW 2026 competition.

## Background & Motivation

Frame-level facial expression recognition (EXPR) in the wild faces significant challenges: unstable face localization, large scale variation, and pervasive blur, occlusion, extreme pose, and illumination changes. Raw videos in the Aff-Wild2 dataset contain abundant such disturbances, making single-frame visual features noisy and inconsistent.

Furthermore, affective signals are inherently multimodal—visual information alone may be insufficient for accurate expression recognition in ambiguous scenes, whereas audio cues (e.g., interjections, plosives) can provide critical complementary information. Nevertheless, effectively fusing multimodal data while maintaining temporal consistency remains a challenge.

The proposed strategy addresses this in two stages: first enhancing the visual encoder's expression-awareness on external image datasets, then performing multimodal fusion and temporal smoothing on video data.

## Method

### Overall Architecture

A two-stage pipeline is employed: Stage I fine-tunes a DINOv2 ViT-L/14 encoder on AffectNet + RAF-DB image datasets, incorporating PadAug augmentation and a MoE training head; Stage II extracts multi-scale facial crops from video for visual features, simultaneously extracts frame-aligned audio features via Wav2Vec 2.0, and integrates the dual-modality information through a gated fusion module, with temporal smoothing applied at inference.

### Key Designs

1. **Padding-Aware Augmentation (PadAug)**:

    - Function: Improves model robustness against boundary artifacts introduced by large-scale facial crops.
    - Mechanism: Black padding strips are inserted along the borders of training images with small spatial perturbations, simulating various boundary patterns (left/right/top/bottom/corner). When the crop region extends beyond the image boundary—common in in-the-wild video—padding regions introduce distribution shifts.
    - Design Motivation: In Stage II, large crop boxes frequently exceed the frame boundary, producing padded regions. PadAug proactively exposes these patterns during training to prevent distribution mismatch at inference.

2. **Training-Only Mixture of Experts (MoE) Head**:

    - Function: Provides stronger task-oriented supervision during the visual adaptation stage.
    - Mechanism: A MoE classification head is appended after the DINOv2 encoder, enhancing adaptation via sample-dependent expert routing. Crucially, the MoE head is used only during Stage I training and discarded afterward; only the fine-tuned DINOv2 backbone is retained.
    - Design Motivation: The MoE head supplies richer gradient signals to help DINOv2 learn expression-discriminative features without increasing inference complexity.

3. **Gated Audio-Visual Fusion + Temporal Smoothing**:

    - Function: Integrates dual-modality information and maintains temporal consistency.
    - Mechanism: Visual features are derived from three scales of facial crops (encoded by the Stage I-adapted DINOv2); audio features are extracted from a short window around the target frame via Wav2Vec 2.0. Both feature streams are integrated through a lightweight gated fusion module that learns per-modality weight gates. Sliding-window temporal smoothing is applied at inference to reduce inter-frame prediction jitter.
    - Design Motivation: Audio provides complementary cues in visually ambiguous scenes (e.g., an angry vocal tone helps distinguish anger from disgust), but over-reliance should be avoided when no meaningful audio is present. The gating mechanism adaptively regulates modality weights.

### Loss & Training

Stage I employs cross-entropy loss to fine-tune DINOv2 on AffectNet + RAF-DB for 8-class expression classification. Stage II trains the gated fusion module on Aff-Wild2 video data. Temporal smoothing is applied as post-processing at inference.

## Key Experimental Results

### Main Results

| Evaluation Setting | Macro-F1 | Note |
|-------------------|----------|------|
| Official validation set | **0.5368** | Final submission |
| 5-fold cross-validation | 0.5122 ± 0.0277 | Stability verification |
| Official test set | 0.391 | Challenge server |

### Ablation Study

| Configuration | Macro-F1 Change | Note |
|--------------|----------------|------|
| w/o PadAug | Decrease | Boundary artifacts affect multi-scale crops |
| w/o MoE head | Decrease | Weakened visual adaptation |
| Visual only (w/o audio) | Decrease | Audio provides complementary information |
| w/o temporal smoothing | Decrease | Unstable inter-frame predictions |
| Full model | **Best** | All components are complementary |

### Key Findings

- Domain adaptation of the visual encoder (Stage I) is the primary performance contributor—images generally capture the dominant affective information in video.
- Audio as a complementary modality provides marginal but meaningful gains.
- PadAug is particularly important for multi-scale crop scenarios.
- A large gap exists between validation and test set performance (0.54 vs. 0.39), suggesting potential overfitting or distribution shift.

## Highlights & Insights

- **PadAug targets a practical problem**: Padding artifacts are pervasive yet underappreciated in in-the-wild video; this targeted augmentation outperforms generic alternatives.
- **"Use-and-discard" MoE strategy**: The MoE head leverages rich gradients to aid encoder adaptation without incurring inference overhead—an elegant design pattern.
- **Two-stage decoupling**: Learning visual representations on clean images first, then performing fusion on noisy video, prevents video noise from contaminating visual representation learning during end-to-end training.

## Limitations & Future Work

- The large gap between validation and test set performance (0.54→0.39) raises concerns about generalization.
- The gated fusion module is relatively simple and does not model temporal interactions between audio and visual streams.
- Only DINOv2 is explored as the visual backbone; alternatives such as CLIP remain uninvestigated.
- Temporal smoothing is applied as post-processing rather than through an end-to-end learned temporal model.

## Related Work & Insights

- **vs. MAE-based EXPR**: The reconstruction objective of MAE may be less effective at learning expression-discriminative features compared to DINOv2's self-distillation objective.
- **vs. CLIP-based EXPR**: CLIP's text-image alignment may offer advantages through carefully designed textual prompts for expression classification.
- **vs. Multi-task approaches**: Some ABAW methods jointly train on EXPR + AU + VA, potentially improving performance through shared representations.

## Rating

- Novelty: ⭐⭐⭐ PadAug and the training-only MoE exhibit design ingenuity, though the overall framework is relatively conventional.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablation studies validate each component's contribution, though the large test set performance drop is a concern.
- Writing Quality: ⭐⭐⭐⭐ Method descriptions are clear and figures are intuitive.
- Value: ⭐⭐⭐ A competition solution with practical techniques of reference value, though academic novelty is limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SynFER: Towards Boosting Facial Expression Recognition with Synthetic Data](../../ICCV2025/human_understanding/synfer_towards_boosting_facial_expression_recognition_with_synthetic_data.md)
- [\[CVPR 2026\] FusionAgent: A Multimodal Agent with Dynamic Model Selection for Human Recognition](fusionagent_a_multimodal_agent_with_dynamic_model_selection_for_human_recognitio.md)
- [\[AAAI 2026\] Facial-R1: Aligning Reasoning and Recognition for Facial Emotion Analysis](../../AAAI2026/human_understanding/facial-r1_aligning_reasoning_and_recognition_for_facial_emotion_analysis.md)
- [\[CVPR 2026\] A2P: From 2D Alignment to 3D Plausibility for Occlusion-Robust Two-Hand Reconstruction](from_2d_alignment_to_3d_plausibility_unifying_hete.md)
- [\[ICCV 2025\] DADM: Dual Alignment of Domain and Modality for Face Anti-Spoofing](../../ICCV2025/human_understanding/dadm_dual_alignment_of_domain_and_modality_for_face_anti-spoofing.md)

</div>

<!-- RELATED:END -->
