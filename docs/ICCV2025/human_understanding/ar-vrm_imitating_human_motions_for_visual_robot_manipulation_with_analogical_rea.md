---
title: >-
  [Paper Note] AR-VRM: Imitating Human Motions for Visual Robot Manipulation with Analogical Reasoning
description: >-
  [ICCV 2025][Human Understanding][visual robot manipulation] This paper proposes AR-VRM, the first method to enhance visual robot manipulation (VRM) through explicit imitation of human hand keypoints. It employs a keypoint vision-language model pretrained on large-scale human activity videos to acquire motion knowledge, and establishes correspondences between human hand keypoints and robot components via analogical reasoning.
tags:
  - ICCV 2025
  - Human Understanding
  - visual robot manipulation
  - analogical reasoning
  - human keypoints
  - vision-language pretraining
  - human-to-robot motion transfer
date: 2026-05-08
content_hash: 437cee008a0d017c
---

# AR-VRM: Imitating Human Motions for Visual Robot Manipulation with Analogical Reasoning

**Conference**: ICCV 2025
**arXiv**: [2508.07626](https://arxiv.org/abs/2508.07626)
**Code**: [https://github.com/idejie/ar](https://github.com/idejie/ar)
**Area**: Human Understanding / Robot Manipulation
**Keywords**: visual robot manipulation, analogical reasoning, human keypoints, vision-language pretraining, human-to-robot motion transfer

## TL;DR

This paper proposes AR-VRM, the first method to enhance visual robot manipulation (VRM) through explicit imitation of human hand keypoints. It employs a keypoint vision-language model pretrained on large-scale human activity videos to acquire motion knowledge, and establishes correspondences between human hand keypoints and robot components via analogical reasoning.

## Background & Motivation

Visual robot manipulation requires paired image, language instruction, and robot action state data, which is extremely costly to collect. Data scarcity severely limits model performance. Although existing methods leverage large-scale pretraining to compensate, two core issues remain:

**Domain gap**: Pretraining on web data such as VQA introduces a large distributional discrepancy with robot manipulation tasks, limiting knowledge transfer.

**Limitations of implicit learning**: Some methods use human activity videos (e.g., Ego4D) but learn implicitly via contrastive learning or pixel-level future frame prediction, inevitably incorporating irrelevant background information and pixel noise.

**Key Insight**: Human hand keypoints and robot actions share underlying structural similarities when manipulating objects. Learning should explicitly focus on motion itself (keypoints) while ignoring irrelevant visual information. The core challenges are: (1) how to extract motion knowledge from large-scale human videos in the form of keypoints, and (2) how to establish correspondences between morphologically disparate human hands and robot arms.

## Method

### Overall Architecture

AR-VRM consists of two stages: a pretraining stage that learns a Keypoint VLM, and a fine-tuning stage that introduces an analogical reasoning module.

### Key Designs

1. **Keypoint VLM Pretraining**

    - Utilizes the large-scale egocentric video dataset Ego4D (3,500 hours, 800K video clips, 8M frames).
    - 3D hand keypoints $k_t \in \mathbb{R}^K$ (3D coordinates in image space) are detected offline using InterHand.
    - Three-modality token alignment:
        - Language: CLIP text encoder + MLP → $z_l \in \mathbb{R}^d$
        - Visual: ViT (MAE pretrained) + Perceiver Resampler → $z_o^{CLS}, z_o^{p_{1:M}} \in \mathbb{R}^d$
        - Keypoint: HandFormer encoder + MLP → $z_k \in \mathbb{R}^d$
    - Concatenated tri-modal tokens are fed into Transformer layers for next-token prediction:
    $\hat{z_{k_t}} = h^{Atten}(z_l, z_{o_1}, z_{k_1}, ..., z_{o_{t-1}}, z_{k_{t-1}})$
    - Pretraining loss is MSE: $\mathcal{L}_{pretrain} = \sum_{t=1}^T \mathcal{L}_{MSE}(\text{KeypointHead}(\hat{z_{k_t}}), k_t)$
    - CLIP text encoder and ViT parameters are frozen; only interaction layers are trained.

2. **Robot Fine-tuning with Analogical Reasoning**

    - **Human motion retrieval**: Relevant human motion videos are retrieved based on language similarity and cosine similarity of visual frame features:
    $sim(\tau^R, \tau^H) = cos(z_l^R, z_l^H) + \sum_t cos(z_{o_t}^R, z_{o_t}^H)$
    - **Analogy mapping matrix**: A learnable matrix $m \in \mathbb{R}^{S \times K}$ is introduced to model the mapping from human hand keypoints to robot components, where $S$ denotes the number of robot components and $K$ the number of hand keypoints.
    - Human keypoint features are transformed into simulated robot state features via the mapping matrix:
    $f_{s_j}^* = (1-\alpha) \cdot m \cdot f_{k_j} + \alpha \cdot f_s$
    - Analogical reasoning loss: $\mathcal{L}_{AR} = \sum_{j=1}^J \mathcal{L}_{MSE}(\text{Linear}(f_{s_{j,T}}^*), s_T)$
    - Total fine-tuning loss: $\mathcal{L}_{finetune} = \mathcal{L}_{state} + \beta \cdot \mathcal{L}_{AR}$

3. **Fine-tuning Strategy**

    - The keypoint encoder and keypoint prediction head are frozen; the VLM Transformer layers are fine-tuned.
    - Freezing the keypoint module preserves the keypoint encoding and decoding capability acquired during pretraining.
    - Fine-tuning the VLM prevents overfitting and catastrophic forgetting on small-scale robot data.
    - Retrieved human video samples serve as a form of data replay during fine-tuning.

### Loss & Training

| Stage | Loss Function | Description |
|-------|--------------|-------------|
| Pretraining | $\mathcal{L}_{pretrain} = \sum \mathcal{L}_{MSE}$ | Predict human hand keypoints |
| Fine-tuning (state) | $\mathcal{L}_{state} = \mathcal{L}_{MSE}$ | Predict robot state |
| Fine-tuning (analogy) | $\mathcal{L}_{AR} = \sum \mathcal{L}_{MSE}$ | Supervision for analogy mapping |

## Key Experimental Results

### Main Results (CALVIN ABCD→D)

| Method | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Avg.Len | Avg.Rate |
|--------|--------|--------|--------|--------|--------|---------|----------|
| MCIL | 0.373 | 0.027 | 0.002 | 0.000 | 0.000 | 0.40 | 8.0% |
| RT-1 | 0.844 | 0.617 | 0.438 | 0.323 | 0.227 | 2.45 | 49.0% |
| GR-1 | 0.949 | 0.896 | 0.844 | 0.789 | 0.731 | 4.21 | 84.2% |
| **AR-VRM** | **0.951** | **0.915** | **0.855** | **0.800** | **0.751** | **4.27** | **85.4%** |

Unseen scene generalization (ABC→D): AR-VRM 65.9% vs. GR-1 61.2% (**+4.7%**)

### Ablation Study

| Pretrain | Retrieval | AR | Avg.Len | Avg.Rate |
|----------|-----------|-----|---------|----------|
| ✗ | ✗ | ✗ | 3.00 | 60.0% |
| ✓ | ✗ | ✗ | 4.06 | 81.3% |
| ✓ | ✓ | ✗ | 4.21 | 84.3% |
| ✓ | ✓ | ✓ | **4.27** | **85.4%** |

Few-shot (10% data): AR-VRM 45.6% vs. GR-1 40.0% (**+5.6%**)

Real-robot experiments:

| Task | RT-1 | GR-1 | AR-VRM |
|------|------|------|--------|
| Seen objects | 0.27 | 0.79 | **0.95** |
| Unseen instances | 0.13 | 0.73 | **0.91** |
| Unseen categories | 0.00 | 0.30 | **0.53** |
| Articulated manipulation | 0.35 | 0.75 | **0.82** |

### Key Findings

- Pretraining yields the largest gain (60% → 81.3%), validating the effectiveness of large-scale human video pretraining.
- Retrieval combined with data replay further improves performance to 84.3%, mitigating forgetting of pretrained knowledge.
- Analogical reasoning provides additional improvement (84.3% → 85.4%), with a more pronounced contribution in low-data regimes (+5.6% with 10% data).
- Freezing keypoint parameters while fine-tuning the VLM is the optimal configuration (85.4%); the reverse leads to significant degradation.
- Visualization of the analogy mapping matrix reveals semantically meaningful human-to-robot correspondences: robot grippers map to human fingertips, and the robot arm base maps to the palm.

## Highlights & Insights

- **The "explicit vs. implicit" insight**: Prior work implicitly learns human motion knowledge via pixel-level prediction or contrastive learning, inevitably mixing in background noise. AR-VRM focuses on keypoints to directly capture the essence of motion.
- **Elegant design of analogical reasoning**: The learnable mapping matrix automatically discovers functional correspondences between human hands and robot components without manual specification.
- **Effective use of data replay**: Incorporating retrieved human videos during fine-tuning not only provides motion guidance but also prevents VLM overfitting.

## Limitations & Future Work

- The analogy mapping matrix is globally shared; different tasks may require distinct mapping relationships.
- The method relies on the quality of InterHand keypoint detection, which may be unstable under occlusion or fast motion.
- Ego4D is biased toward everyday manipulation; its applicability to fine-grained industrial manipulation remains unexplored.
- The 5-task sequence success rate (75.1%) still leaves room for improvement.
- Generalization to unseen categories on real robots (53%) remains a substantial challenge.

## Related Work & Insights

- Unlike ATM, which requires paired human-robot data, AR-VRM only needs human videos and robot demonstrations collected independently.
- The keypoint-prediction pretraining paradigm is extendable to whole-body keypoints, supporting broader human-robot interaction scenarios.
- AR-VRM establishes a new explicit learning paradigm for the research direction of "learning robot skills from human videos."

## Rating
- Novelty: ⭐⭐⭐⭐ First VRM method employing explicit keypoint imitation learning; the analogical reasoning design is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Full CALVIN settings + real-robot experiments + comprehensive ablations + visualization analysis.
- Writing Quality: ⭐⭐⭐⭐ Motivation and methodology are clearly described with thorough comparative analysis.
- Value: ⭐⭐⭐⭐ Provides a practical pretraining solution for robot manipulation under data scarcity.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Generating Attribute-Aware Human Motions from Textual Prompt](../../AAAI2026/human_understanding/generating_attribute-aware_human_motions_from_textual_prompt.md)
- [\[CVPR 2026\] EgoPoseFormer v2: Accurate Egocentric Human Motion Estimation for AR/VR](../../CVPR2026/human_understanding/egoposeformer_v2_accurate_egocentric_human_motion_estimation_for_arvr.md)
- [\[ICLR 2026\] QuaMo: Quaternion Motions for Vision-based 3D Human Kinematics Capture](../../ICLR2026/human_understanding/quamo_quaternion_motion_kinematics.md)
- [\[ICCV 2025\] What's Making That Sound Right Now? Video-centric Audio-Visual Localization](whats_making_that_sound_right_now_video-centric_audio-visual_localization.md)
- [\[CVPR 2026\] UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control from Egocentric Human Videos](../../CVPR2026/human_understanding/unidex_a_robot_foundation_suite_for_universal_dexterous_hand_control_from_egocen.md)

</div>

<!-- RELATED:END -->
