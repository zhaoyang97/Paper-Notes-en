---
title: >-
  [Paper Note] Pose Priors from Language Models
description: >-
  [CVPR 2025][Human Understanding][3D Pose Estimation] Proposes the ProsePose framework, which leverages Large Multimodal Models (LMMs, e.g., GPT-4V) as contact priors to extract body-part contact constraints from images and convert them into optimizable loss functions, improving 3D pose estimation in close-interaction and self-contact scenarios without human contact annotations.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "3D Pose Estimation"
  - "Large Language Models"
  - "Contact Prediction"
  - "Pose Optimization"
  - "Physical Interaction"
date: 2026-05-08
content_hash: 6bffecc367244421
---

# Pose Priors from Language Models

**Conference**: CVPR 2025  
**arXiv**: [2405.03689](https://arxiv.org/abs/2405.03689)  
**Code**: [GitHub](https://prosepose.github.io)  
**Area**: Human Pose Estimation  
**Keywords**: 3D Pose Estimation, Large Language Models, Contact Prediction, Pose Optimization, Physical Interaction

## TL;DR

Proposes the ProsePose framework, which leverages Large Multimodal Models (LMMs, e.g., GPT-4V) as contact priors to extract body-part contact constraints from images and convert them into optimizable loss functions, improving 3D pose estimation in close-interaction and self-contact scenarios without human contact annotations.

## Background & Motivation

Language naturally encodes rich physical and social interaction information, having developed vocabulary over generations to describe nuances such as hugging, shaking hands, and athletic poses. However, most 3D human pose estimation methods overlook this valuable source of information.

Contact scenarios (close interaction or self-contact) are particularly challenging for pose estimation:
1. **Severe Occlusions**: Body parts at contact regions are often occluded, rendering 2D keypoints alone insufficient to infer contact information.
2. **Expensive Data Acquisition**: Existing methods such as BUDDI and REMIPS rely on manually annotated contact maps or motion capture data for training, which are limited to only tens of thousands of images.
3. **Limited Generalization**: Models trained on specific contact datasets struggle to generalize to novel types of interactions.

Key insight: LMMs pre-trained on massive image-text pairs implicitly learn rich knowledge about human poses and interactions. If an LMM can articulate which body parts are in contact, these linguistic descriptions can be converted into constraints for 3D pose optimization. This provides a scalable alternative path **without requiring any contact annotation data**.

## Method

### Overall Architecture

The ProsePose framework consists of three stages:
1. **Initial Pose Estimation**: Obtains a coarse 3D pose using pose regressors (BEV for multi-person, HMR2 for single-person).
2. **LMM Constraint Generation**: Sends images and instructions to the LMM to extract a list of body-part contact pairs.
3. **Constrained Pose Optimization**: Converts contact constraints into loss functions, which are jointly optimized with keypoint loss and prior losses to refine pose parameters.

### Key Designs

1. **LMM Contact Constraint Generation**:
    - **Function**: Automatically infers contact relationships between body parts from images.
    - **Mechanism**: Crops and segments the target person's image, then feeds it into the LMM (GPT-4V). The prompt specifies coarse-grained body regions (arm, shoulder, back, waist, etc.) and requests a list of all contacting region pairs. A chain-of-thought approach is used to first describe the pose before listing the constraints. Left/right hands are not specified (since LMMs are unreliable regarding chirality); instead, all left/right assignments are enumerated to find the minimum loss during optimization. Responses are sampled N=20 times and filtered by frequency.
    - **Design Motivation**: Coarse-grained regions match the linguistic granularity of the LMM. Multi-sampling and frequency filtering alleviate hallucinations.

2. **Conversion from Constraints to Loss Functions**:
    - **Function**: Converts linguistic constraints into differentiable optimization objectives.
    - **Mechanism**: Each constraint $c=(R_a, R_b)$ corresponds to two sets of mesh vertices, where the loss is defined as the minimum distance between the two sets. The average loss over $N$ sampled constraint sets is computed (similar to self-consistency). For two-person scenarios, the minimum loss between the two person-to-mesh mappings is selected. Fallback to the baseline occurs if the empty constraint set exceeds a threshold.
    - **Design Motivation**: Averaging losses from multiple samplings effectively mitigates the impact of individual hallucinations.

3. **Joint Pose Optimization**:
    - **Function**: Optimizes SMPL-X parameters by synthesizing multiple loss constraints.
    - **Mechanism**: Performed in two stages: Stage 1 optimizes $\theta$ (pose) + $\beta$ (shape) + $t$ (translation), and Stage 2 fixes $\beta$ to only optimize $\theta$ and $t$. The total loss is calculated as: $L_{total} = \lambda_{LMM} \cdot L_{LMM} + \lambda_{GMM} \cdot L_{GMM} + \lambda_{\beta} \cdot L_{\beta} + \lambda_{\theta} \cdot L_{\theta} + \lambda_{2D} \cdot L_{2D} + \lambda_{P} \cdot L_{P}$.
    - **Design Motivation**: Multi-source losses balance each other: $L_{LMM}$ guides contact, $L_{2D}$ maintains projection accuracy, $L_{\theta}$ prevents deviation from initialization, and $L_{P}$ prevents interpenetration.

### Loss & Training

- **L_LMM**: LMM contact constraint loss (core contribution)
- **L_GMM**: Gaussian Mixture Model pose prior
- **L_β**: Shape regularization $||\beta||^2_2$
- **L_θ**: Penalty for deviation from initial pose $||\theta' - \theta||^2_2$
- **L_2D**: 2D keypoint reprojection loss (OpenPose + ViTPose)
- **L_P**: Interpenetration penalty loss (based on winding numbers)

## Key Experimental Results

### Main Results

Multi-person interaction (Joint PA-MPJPE↓ / PCC↑):

| Method | Hi4D PM↓ | FlickrCI3D PM↓ | FlickrCI3D PCC↑ | CHI3D PM↓ | CHI3D PCC↑ |
|------|----------|----------------|-----------------|-----------|------------|
| BEV (Initialization) | 144 | 106 | 64.8 | 96 | 71.4 |
| Heuristic | 116 | 67 | 77.8 | 105 | 74.1 |
| **ProsePose** | **93** | **58** | **79.9** | 100 | **75.8** |
| BUDDI (Supervised) | 89 | 66 | 81.9 | 68 | 78.0 |

Single-person yoga (MOYO Dataset):

| Method | PA-MPJPE↓ | PCC↑ | PCC@5mm↑ | PCC@15mm↑ |
|------|-----------|------|----------|-----------|
| HMR2 | 84 | 83.0 | 34.2 | 69.5 |
| HMR2+opt | 81 | 85.2 | 47.7 | 74.6 |
| **ProsePose** | 82 | **87.8** | **54.2** | **81.4** |

### Ablation Study

| Configuration | PA-MPJPE (Hi4D val) |
|------|---------------------|
| Full Model | 81 |
| w/o $L_{LMM}$ | 138 |
| w/o $L_{2D}$ | 130 |
| w/o $L_{GMM}$ | 85 |
| w/o $L_{\beta}$ | 91 |
| w/o $L_{\theta}$ | 84 |
| w/o $L_{P}$ | 78 |

Impact of the number of samples: As $N$ increases from 1 to 20, PA-MPJPE continuously decreases while PCC continuously increases.

### Key Findings

1. **Best Among Unsupervised Methods**: ProsePose performs the best among all methods that do not use contact supervision.
2. **Closing the Gap with Supervised Methods**: On Hi4D, it closes 85% of the PA-MPJPE gap between Heuristic and BUDDI.
3. **L_LMM and L_2D Contribute the Most**: Removing these two losses causes PA-MPJPE to surge to 138 and 130, respectively.
4. **Multiple Sampling is Key**: Single LMM predictions are noisy; averaging over 20 samples significantly boosts performance.
5. **Chirality is the Major Weakness of LMMs**: LMMs struggle to distinguish left and right limbs.

## Highlights & Insights

1. **New Paradigm of LMMs for 3D Understanding**: Extracts the implicit physical knowledge of LMMs into explicit constraints, bridging language and 3D geometry.
2. **Scalable Zero-Annotation Strategy**: Does not require any contact annotation data, directly leveraging pre-trained LMMs.
3. **Elegant Robustness Strategy**: Multi-sampling + frequency filtering + loss averaging = an ensemble approach similar to self-consistency.
4. **Unified Framework**: A single framework handles both multi-person interaction and single-person self-contact scenarios.
5. **Oracle Experiment**: Ground-truth contacts can reduce PA-MPJPE from 93 to 81, suggesting massive room for improvement as LMMs get better.

## Limitations & Future Work

1. **LMM Chirality Issues**: Inability to reliably distinguish left and right limbs limits constraint accuracy.
2. **Coarse-Grained Regions**: The body regions currently used are relatively coarse; finer constraints could yield further improvements.
3. **High Fallback Rate**: On CHI3D, 224 out of 431 test cases fall back to the baseline.
4. **LMM Query Cost**: The API cost for obtaining 20 samples from GPT-4V is substantial.
5. The performance of this framework will naturally scale with future improvements in LMM capabilities.

## Related Work & Insights

- **vs. BUDDI**: BUDDI uses learned diffusion priors and training with contact annotations; ProsePose requires no contact annotations whatsoever.
- **vs. PoseScript/PoseFix**: These methods also utilize language and pose but require large-scale paired text-pose datasets for training; ProsePose directly exploits the zero-shot capabilities of pre-trained LMMs.
- **vs. PoseGPT**: PoseGPT uses language as training data but does not outperform pure regression methods; ProsePose demonstrates the effectiveness of LMM priors in contact scenarios.
- **vs. CloseInt**: CloseInt trains a physics-guided diffusion model using multi-person motion capture data; ProsePose requires no such data.

## Rating

- Novelty: ⭐⭐⭐⭐ Utilizing LMMs as contact priors for 3D pose estimation is a novel and inspiring paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on three multi-person datasets and one single-person dataset, with detailed ablations and in-depth LMM analysis.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured, methodologically rigorous, and thoroughly analyzed.
- Value: ⭐⭐⭐⭐ Opens up a new direction for LMM-to-3D pose optimization, showing huge potential as LMMs evolve.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ChatGarment: Garment Estimation, Generation and Editing via Large Language Models](chatgarment_garment_estimation_generation_and_editing_via_large_language_models.md)
- [\[CVPR 2026\] Bridging Facial Understanding and Animation via Language Models](../../CVPR2026/human_understanding/bridging_facial_understanding_and_animation_via_language_models.md)
- [\[NeurIPS 2025\] VimoRAG: Video-based Retrieval-augmented 3D Motion Generation for Motion Language Models](../../NeurIPS2025/human_understanding/vimorag_video-based_retrieval-augmented_3d_motion_generation_for_motion_language.md)
- [\[CVPR 2025\] HiPART: Hierarchical Pose AutoRegressive Transformer for Occluded 3D Human Pose Estimation](hipart_hierarchical_pose_autoregressive_transformer_for_occluded_3d_human_pose_e.md)
- [\[ICLR 2026\] EmoPrefer: Can Large Language Models Understand Human Emotion Preferences?](../../ICLR2026/human_understanding/emoprefer_can_large_language_models_understand_human_emotion_preferences.md)

</div>

<!-- RELATED:END -->
