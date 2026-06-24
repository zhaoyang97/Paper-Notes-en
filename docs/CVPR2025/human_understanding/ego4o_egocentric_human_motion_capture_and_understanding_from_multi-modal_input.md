---
title: >-
  [Paper Note] Ego4o: Egocentric Human Motion Capture and Understanding from Multi-Modal Input
description: >-
  [CVPR 2025][Human Understanding][Egocentric Human Motion Capture] A unified framework, Ego4o, is proposed to achieve human motion capture and motion description generation simultaneously from multi-modal inputs of wearable devices (1-3 IMUs + egocentric images + motion descriptions), where the two tasks mutually enhance each other.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Egocentric Human Motion Capture"
  - "Multi-Modal Fusion"
  - "IMU Sensors"
  - "Motion Description Generation"
  - "VQ-VAE"
date: 2026-05-08
content_hash: 529d46e84af439a2
---

# Ego4o: Egocentric Human Motion Capture and Understanding from Multi-Modal Input

**Conference**: CVPR 2025  
**arXiv**: [2504.08449](https://arxiv.org/abs/2504.08449)  
**Code**: [https://jianwang-mpi.github.io/ego4o](https://jianwang-mpi.github.io/ego4o) (project page available)  
**Area**: Video Understanding  
**Keywords**: Egocentric Human Motion Capture, Multi-Modal Fusion, IMU Sensors, Motion Description Generation, VQ-VAE

## TL;DR

A unified framework, Ego4o, is proposed to achieve human motion capture and motion description generation simultaneously from multi-modal inputs of wearable devices (1-3 IMUs + egocentric images + motion descriptions), where the two tasks mutually enhance each other.

## Background & Motivation

With the ubiquity of wearable devices such as VR/AR headsets, smart glasses, smartphones, and smartwatches, utilizing data from these devices for human motion capture and understanding has broad application prospects. However, existing methods face several key challenges:

1. **Unimodal Limitations**: Most existing methods only utilize a single modality—egocentric camera methods (e.g., EgoBody) suffer from severe self-occlusion, IMU-based methods (e.g., IMUPoser) are insufficient for static pose estimation, and text descriptions lack fine-grained motion details.
2. **Unexploited Modality Complementary**: Different modalities possess natural complementarity—images and text provide semantic context (e.g., seeing a table implies sitting), while IMUs provide precise limb motion data (e.g., smartwatch IMUs can distinguish between a smash and a block in table tennis).
3. **Requirements for Input Flexibility**: Users may turn cameras/microphones on or off at any time, or wear varying numbers of sensors; hence, the system needs to adapt to dynamic changes in input modalities.

## Method

### Overall Architecture

Ego4o consists of three core components: (1) a part-aware VQ-VAE that learns part-decomposed discrete motion representations; (2) a multi-modal encoder that projects IMU signals, egocentric images, and motion descriptions into the motion representation space, with a VQ-VAE decoder reconstructing human motion; and (3) a multi-modal LLM that receives motion encodings and egocentric images to generate motion descriptions, which in turn enhance motion capture accuracy, forming a closed loop.

### Key Designs

1. **Part-aware VQ-VAE**:
    - Function: Learning discrete motion representations decomposed by body parts.
    - Mechanism: 22 joints are split into 6 groups (head, left arm, right arm, left leg, right leg, torso), with each group training an independent encoder $\mathcal{E}_i$ and codebook $C_i \in \mathbb{R}^{N_{code} \times d}$. The input motion sequence is converted to HumanML3D representation $J \in \mathbb{R}^{T \times 263}$ and encoded as $Q_i \in \mathbb{R}^{T' \times d}$ ($T'=T/4$), which are concatenated after quantization and fed into a shared decoder $\mathcal{D}$.
    - Design Motivation: Distinct from conventional methods that encode the entire body as a whole, part decomposition allows mapping IMU signals from different locations directly to corresponding part codebooks while inferring the motion of parts without sensor coverage—similar to the text infilling task in NLP.

2. **Multi-Modal Encoder and Random Mask Training**:
    - Function: Fusing multi-modal inputs to output encoded IDs in the motion codebook.
    - Mechanism: Egocentric images and text are encoded using CLIP to obtain $F_I$ and $F_T$, respectively, while IMU accelerations $A \in \mathbb{R}^{T \times 5 \times 3}$ and rotations (converted to 6D representation $R_{6d} \in \mathbb{R}^{T \times 5 \times 6}$) are concatenated as the sequence $F_{imu}$. The three types of features are passed through embedding layers and fed into a Transformer encoder to predict the logits $L_{t,i}$ of the motion code IDs corresponding to each IMU, with Gumbel Softmax used to sample codebook indices. The training loss includes classification loss and reconstruction loss: $\mathcal{L} = \mathbb{E}_{\hat{L}}(-\log P(\hat{L}|A,R,I,T_m)) + \lambda\|\hat{J}-J\|_2$.
    - Design Motivation: **Random masking** is key to supporting flexible inputs in this method—randomly masking images and text, and randomly selecting 1-3 IMUs as active inputs during training, encourages the model to learn to operate under any combination of inputs.

3. **Multi-Modal LLM Motion Understanding and Feedback Enhancement**:
    - Function: Generating motion descriptions, where description feedback enhances motion capture.
    - Mechanism: LLaVA-7B is extended with a new motion modality—motion codes are mapped to the LLM word embedding space through a linear embedding layer $\mathbf{E}_M$. Training consists of two stages: (1) motion pre-training, which trains only the motion embedding layer for feature alignment; and (2) multi-modal fine-tuning, which updates LLM parameters using LoRA while updating both image and motion embedding layers. **Key Insight**: Although generated motion descriptions are not perfect, they serve as valuable inductive biases fed back into the motion capture module, significantly boosting performance in the absence of manual descriptions.
    - Design Motivation: Utilizing the contextual reasoning and image understanding capabilities of LLMs to generate high-quality motion descriptions; the closed-loop design enables mutual enhancement between motion capture and understanding.

### Loss & Training

**Motion Capture**: Classification loss + $\lambda$-weighted reconstruction L2 loss. **Test-time Optimization** (optional): Optimizing the motion feature $Q$ within the VQ-VAE latent space so that the acceleration and orientation of the predicted motion match the IMU observations:

$$Q^* = \arg\min_Q \lambda_a L_a(J, A) + \lambda_r L_r(J, R)$$

**Motion Understanding**: Standard autoregressive negative log-likelihood loss. The training input is selected randomly as $X_{ins} = \text{RandomSelect}\{[I, X_q], [A,R,X_q], [A,R,I,X_q]\}$.

## Key Experimental Results

### Main Results (Motion Capture Accuracy)

| Method | Dataset | MPJPE(mm)↓ | PA-MPJPE(mm)↓ | Jitter(km/s³)↓ |
|------|--------|-----------|--------------|----------------|
| IMUPoser (1-3 IMUs) | DIP-IMU | 97 | — | 0.19 |
| Ego4o-IMU (1-3 IMUs) | DIP-IMU | 84.06 | 63.95 | 0.076 |
| IMUPoser (1-3 IMUs) | Nymeria | 105.7 | 72.94 | 0.054 |
| Ego4o-IMU (1-3 IMUs) | Nymeria | 95.86 | 69.03 | 0.049 |
| **Ego4o (All Modalities)** | **Nymeria** | **84.82** | **62.33** | **0.048** |

### Ablation Study (Motion Capture on Nymeria Dataset)

| Configuration | MPJPE(mm)↓ | PA-MPJPE(mm)↓ | Explanation |
|------|-----------|--------------|------|
| Ego4o-IMU (IMU Only) | 95.86 | 69.03 | Baseline |
| GT text only + IMU | 86.22 | 63.14 | Text brings significant improvement |
| Images only + IMU | 90.81 | 66.04 | Images also contribute |
| Generated text + IMU | 88.65 | 64.79 | Generated descriptions are also effective |
| Images + Generated text + IMU | 87.00 | 63.67 | Close to the full model |
| **Ego4o (Full)** | **84.82** | **62.33** | **Best** |

### Key Findings

- Using IMU input alone, Ego4o-IMU already outperforms IMUPoser (MPJPE: 95.86 vs 105.7), validating the advantage of the part-aware VQ-VAE representation.
- Multi-modal fusion (images + text + IMUs) further reduces MPJPE by about 11mm compared to pure IMUs, validating modality complementarity.
- **AI-generated motion descriptions** can effectively replace manual descriptions to boost performance (88.65 vs 95.86); this is a key finding—the system can generate descriptions to self-enhance.
- In motion description generation, Ego4o significantly outperforms MotionGPT on BERTScore (30.13 vs 14.09) and RougeL (38.95 vs 32.33).
- Removing either image or motion tokens leads to a drop in description generation quality, validating the complementary contributions of both modalities to the understanding task.

## Highlights & Insights

- **Closed-loop design** is the most significant highlight: motion capture $\rightarrow$ generate descriptions $\rightarrow$ descriptions feed back to motion capture. This self-enhancement mechanism is highly valuable in practical deployment (users do not need to manually provide descriptions).
- **Part-aware VQ-VAE** elegantly addresses the issue of flexible IMU configurations, transforming the question of "which sensors are available" into "which codebooks have direct inputs, and which require inference".
- **Random mask training** strategy is simple yet extremely effective, allowing a single model to adapt to all possible input combinations.
- Using the motion codebook as a new "language" for the LLM is an elegant cross-modal bridging solution.

## Limitations & Future Work

- Relying on full motion sequences as input introduces latency, making it unsuitable for real-time online applications.
- Although test-time optimization improves accuracy, it introduces additional inference time costs.
- The Nymeria dataset is limited in scale (~170k sequences); larger scale data could further improve generalization.
- Severe human self-occlusion in egocentric images limits the contribution of the image modality due to perspective constraints.

## Related Work & Insights

- Pure IMU methods like IMUPoser/MobilePoser validated the feasibility of sparse sensors, upon which this work extends with multi-modal fusion.
- Methods like MotionGPT frame motion understanding as a language task; this work further demonstrates that multi-modal inputs can significantly improve description quality.
- EgoLM uses LLMs for motion capture but suffers from high computational costs and low accuracy; this work's encoder-decoder architecture is more practical.
- Part-aware VQ-VAE is derived from TLControl; this work transfers it from motion generation scenarios to motion capture.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ A closed-loop framework for multi-modal motion capture + understanding, where the self-enhancement design is an original contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on DIP-IMU and Nymeria but lacks comparisons with more baselines.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear, and the motivation is fully elaborated.
- Value: ⭐⭐⭐⭐⭐ Targeted at practical application scenarios of consumer-grade wearable devices, the closed-loop enhancement mechanism holds high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SocialGesture: Delving into Multi-Person Gesture Understanding](socialgesture_delving_into_multi-person_gesture_understanding.md)
- [\[CVPR 2025\] HumanMM: Global Human Motion Recovery from Multi-shot Videos](humanmm_global_human_motion_recovery_from_multi-shot_videos.md)
- [\[ECCV 2024\] Large Motion Model for Unified Multi-Modal Motion Generation](../../ECCV2024/human_understanding/large_motion_model_for_unified_multi-modal_motion_generation.md)
- [\[ICCV 2025\] KinMo: Kinematic-Aware Human Motion Understanding and Generation](../../ICCV2025/human_understanding/kinmo_kinematic-aware_human_motion_understanding_and_generation.md)
- [\[CVPR 2026\] MMGait: Towards Multi-Modal Gait Recognition](../../CVPR2026/human_understanding/mmgait_multi_modal_gait_recognition.md)

</div>

<!-- RELATED:END -->
