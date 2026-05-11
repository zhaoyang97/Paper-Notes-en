---
title: >-
  [Paper Note] SynFER: Towards Boosting Facial Expression Recognition with Synthetic Data
description: >-
  [ICCV 2025][Human Understanding][Facial expression recognition] This paper proposes SynFER, a diffusion-model-based facial expression synthesis framework that achieves fine-grained expression generation via dual control…
tags:
  - "ICCV 2025"
  - "Human Understanding"
  - "Facial expression recognition"
  - "synthetic data"
  - "diffusion models"
  - "facial action units"
  - "label calibration"
date: 2026-05-08
content_hash: a9bede8a57029e5f
---

# SynFER: Towards Boosting Facial Expression Recognition with Synthetic Data

**Conference**: ICCV 2025
**arXiv**: [2410.09865](https://arxiv.org/abs/2410.09865)
**Code**: [Available](https://github.com/)
**Area**: Facial Expression Recognition / Synthetic Data
**Keywords**: Facial expression recognition, synthetic data, diffusion models, facial action units, label calibration

## TL;DR

This paper proposes SynFER, a diffusion-model-based facial expression synthesis framework that achieves fine-grained expression generation via dual control signals — text descriptions and Facial Action Units (FAUs) — and introduces a FERAnno label calibrator to ensure annotation reliability. The effectiveness of synthetic data for FER is validated across four learning paradigms: self-supervised, supervised, zero-shot, and few-shot learning.

## Background & Motivation

Facial Expression Recognition (FER) suffers from a severe data scarcity problem. Existing FER datasets are far smaller than general-purpose visual datasets (AffectNet ~280K vs. ImageNet 1.4M), and are plagued by high annotation subjectivity, low-quality images, and elevated label error rates. These deficiencies hinder the development of foundation models for FER.

Directly leveraging diffusion models to generate FER data faces two key challenges: (1) generative model training data lacks diverse expressions, making it difficult to capture subtle expressive semantics; and (2) expressions are abstract and subjective concepts that cannot be directly annotated like depth maps or segmentation masks.

## Method

### Overall Architecture

A three-stage pipeline:
1. **Preparation**: Construct the FEText dataset and generate FAU labels and text descriptions.
2. **Generation**: FAU-controlled, semantically guided diffusion model generates high-fidelity expression images.
3. **Annotation**: FERAnno label calibrator automatically generates reliable labels, validated via ensemble voting.

### Key Designs

1. **FEText Dataset Construction**:

    - The first facial expression image-text pair dataset, integrating FFHQ, CelebA-HQ, AffectNet, and SFEW.
    - Contains 400K curated image-text pairs.
    - A super-resolution model is applied to unify the resolution of low-resolution images (AffectNet, SFEW).
    - ShareGPT-4V, a multimodal large language model, is used to generate detailed expression description texts.
    - Carefully designed prompts guide the model to produce precise, emotionally reflective descriptions.

2. **FAU-Controlled Expression Generation**:

    - Text descriptions provide high-level semantic control but lack fine-grained facial muscle movement information.
    - Facial Action Units (FAUs) are introduced as explicit control signals, with each AU corresponding to a specific facial muscle movement.
    - Following IP-Adapter, a decoupled cross-attention module integrates FAU embeddings.
    - FAU labels are annotated using the OpenGraphAU model.
    - The diffusion model parameters are frozen; only the AU adapter (MLP) is trained.

3. **Semantic Guidance**:

    - Addresses FER label imbalance and inter-expression ambiguity (e.g., disgust vs. anger).
    - Layout initialization: images are randomly selected from FEText and inverted into initial noise to preserve natural facial structure.
    - Semantic guidance: during the late denoising stage, gradients from an external FER classifier are used to update text embeddings.
    - Update rule: $c_{t-1}^{text} = c_t^{text} + \lambda_g \frac{\nabla_{c_t^{text}} \mathcal{L}_g}{\|\nabla_{c_t^{text}} \mathcal{L}_g\|_2}$
    - where $\mathcal{L}_g = -y \log(h(f(\hat{x}_0))_i)$ is the classification loss.

4. **FERAnno Label Calibrator**:

    - A diffusion-model-based pseudo-label generator that leverages U-Net intermediate features and cross-attention maps.
    - Image inversion ($t=1$, preserving maximum facial detail) → feature extraction → dual-branch encoder fusion.
    - Multi-scale feature maps capture global generative information; cross-attention maps provide class-discriminative information.
    - A bidirectional cross-attention block fuses the two feature types → a linear layer outputs expression class probabilities.
    - Ensemble voting with external FER models: when predictions are inconsistent, they are replaced with the ensemble prediction average.

### Loss & Training

- Diffusion model training: standard diffusion loss $\min_\theta \mathbb{E}\|\epsilon - \epsilon_\theta(x_t, c, t)\|_2^2$
- AU adapter training: diffusion model is frozen; only the MLP mapping FAU → embeddings is trained.
- FERAnno training: dual-branch architecture utilizing intermediate diffusion model features and attention maps.

## Key Experimental Results

### Main Results

Self-supervised pre-training + linear probing (ResNet-50):

| SSL Method | Pre-training Data | Scale | RAF-DB | AffectNet | SFEW |
|-----------|------------------|-------|--------|-----------|------|
| MoCo v3 | AffectNet | 0.2M | 79.05 | 51.03 | 49.34 |
| MoCo v3 | SynFER | 1.0M | 81.17(+2.12) | 55.56(+4.53) | 50.78(+1.44) |
| MoCo v3 | Both | 1.2M | 81.68(+2.63) | 57.84(+6.81) | 51.26(+1.92) |

Supervised learning improvements:

| Method | RAF-DB | AffectNet |
|--------|--------|-----------|
| POSTER++ | 91.59 | 67.49 |
| **POSTER++ + SynFER** | **91.95** | **69.04** |
| APViT | 91.78 | 66.94 |
| **APViT + SynFER** | **92.05** | **67.26** |
| **FERAnno (standalone)** | **92.56** | **70.38** |

Training on synthetic data only: 67.23% on AffectNet (equal data volume) → 69.84% (5× data volume).

### Ablation Study

Contribution of each component to generation quality and downstream performance:

| Method | FER Acc. | AU Acc. | RAF-DB | AffectNet |
|--------|----------|---------|--------|-----------|
| SD baseline | 20.06% | 87.72% | 89.42 | 65.36 |
| + FEText | 34.62% | 88.91% | 90.54 | 66.62 |
| + FEText + FAUs | 48.74% | 92.37% | 91.68 | 67.68 |
| + FEText + FAUs + SG | **55.14%** | **93.31%** | **91.95** | **68.13** |

Generation quality comparison (FID↓):

| Method | FID | FER Acc. | User Preference (EA) |
|--------|-----|----------|----------------------|
| Stable Diffusion | 88.40 | 20.06% | 2.86% |
| FineFace | 74.61 | 38.05% | 5.73% |
| **SynFER** | **16.32** | **55.14%** | **59.64%** |

### Key Findings

- **FAU control significantly improves expression accuracy**: FER accuracy increases from 34.62% to 48.74%, rendering ambiguous expressions (e.g., fear vs. surprise) more distinguishable.
- **Semantic guidance provides further improvement**: an additional 6.4% gain in expression accuracy on top of FAU control.
- **Self-supervised learning benefits the most**: synthetic data yields substantially larger gains for SSL than for supervised learning, as the latter demands stricter distributional alignment.
- **FERAnno as a standalone strong classifier**: 92.56% on RAF-DB and 70.38% on AffectNet, surpassing all prior state-of-the-art FER models.
- **Data scaling is effective**: particularly when combined with Real-Fake techniques for distribution alignment, supervised learning also benefits from larger volumes of synthetic data.

## Highlights & Insights

- **End-to-end synthetic data pipeline**: a closed loop spanning dataset construction (FEText) → generation control (FAU + SG) → label calibration (FERAnno) → downstream validation.
- **Dual role of FERAnno**: it functions both as a label calibration tool and as a standalone state-of-the-art FER classifier, demonstrating that diffusion model internal features are highly informative for expression understanding.
- **Validation across four learning paradigms**: comprehensive evaluation over SSL, supervised, zero-shot, and few-shot settings provides strong empirical evidence.
- **Addressing the FER data scale bottleneck**: provides a scalable data generation solution for training FER foundation models.

## Limitations & Future Work

- Gains from synthetic data in supervised learning remain modest (<2%), with distributional alignment still being the primary bottleneck.
- Coverage is limited to basic expression categories (7 classes); compound expressions and micro-expressions require finer-grained control.
- Text descriptions in FEText rely on ShareGPT-4V, which may introduce descriptive bias.
- The accuracy of the FAU detection model (OpenGraphAU) inherently limits the precision of the control signal.

## Related Work & Insights

- The approach of leveraging diffusion model intermediate features for classification in FERAnno is extensible to other visual understanding tasks.
- The FAU control mechanism can be generalized to other generation tasks requiring fine-grained facial control, such as talking head synthesis.
- The finding that SSL benefits more than supervised learning from synthetic data offers a useful reference for synthetic data applications in other domains.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First complete diffusion-based FER synthesis pipeline; FAU and semantic guidance design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four learning paradigms × six datasets × extensive ablations — extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Pipeline described clearly with rich figures and tables.
- **Value**: ⭐⭐⭐⭐ Provides a scalable data generation solution for the FER community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] A Two-Stage Dual-Modality Model for Facial Expression Recognition](../../CVPR2026/human_understanding/a_two_stage_dual_modality_model_for_facial_expression_recognition.md)
- [\[ICCV 2025\] PoseSyn: Synthesizing Diverse 3D Pose Data from In-the-Wild 2D Data](posesyn_synthesizing_diverse_3d_pose_data_from_in-the-wild_2d_data.md)
- [\[ICCV 2025\] Monocular Facial Appearance Capture in the Wild](monocular_facial_appearance_capture_in_the_wild.md)
- [\[AAAI 2026\] New Synthetic Goldmine: Hand Joint Angle-Driven EMG Data Generation Framework for Micro-Gesture Recognition](../../AAAI2026/human_understanding/new_synthetic_goldmine_hand_joint_angle-driven_emg_data_generation_framework_for.md)
- [\[NeurIPS 2025\] BEDLAM2.0: Synthetic Humans and Cameras in Motion](../../NeurIPS2025/human_understanding/bedlam20_synthetic_humans_and_cameras_in_motion.md)

</div>

<!-- RELATED:END -->
