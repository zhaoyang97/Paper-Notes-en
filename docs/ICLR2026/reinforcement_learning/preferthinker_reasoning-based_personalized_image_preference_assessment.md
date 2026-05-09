---
title: >-
  [Paper Note] PreferThinker: Reasoning-based Personalized Image Preference Assessment
description: >-
  [ICLR2026][Reinforcement Learning][personalized preference assessment] This paper proposes PreferThinker, which introduces a universal visual preference profile to bridge across different users and adopts a predict-then-assess CoT reasoning paradigm for interpretable personalized image preference assessment. Combined with cold-start SFT and GRPO reinforcement learning along with a similarity-aware prediction reward, the 7B model outperforms GPT-4o (+5.2%) and Claude 3.7 (+5.1%).
tags:
  - ICLR2026
  - Reinforcement Learning
  - personalized preference assessment
  - reasoning
  - GRPO
  - predict-then-assess
  - visual preference profile
  - CoT
date: 2026-05-08
content_hash: bab01aa9c3800224
---

# PreferThinker: Reasoning-based Personalized Image Preference Assessment

**Conference**: ICLR2026  
**arXiv**: [2511.00609](https://arxiv.org/abs/2511.00609)  
**Code**: [Project Page](https://preferthinker.github.io/)  
**Area**: Reinforcement Learning  
**Keywords**: personalized preference assessment, reasoning, GRPO, predict-then-assess, visual preference profile, CoT  

## TL;DR
This paper proposes PreferThinker, which introduces a universal visual preference profile to bridge across different users and adopts a predict-then-assess CoT reasoning paradigm for interpretable personalized image preference assessment. Combined with cold-start SFT and GRPO reinforcement learning along with a similarity-aware prediction reward, the 7B model outperforms GPT-4o (+5.2%) and Claude 3.7 (+5.1%).

## Background & Motivation
- **Personalized preference assessment faces two major challenges**:
  1. Per-user personalized data is extremely scarce and not scalable, unlike general preference data with sharable evaluation criteria.
  2. Personalized preferences span multiple dimensions (artistic style, color, medium, etc.), making them complex and diverse.
- **CLIP-based methods** (PickScore, ImageReward, etc.): rely on large-scale general preference data for training, cannot handle personalized scenarios, and only output numerical scores without interpretability.
- **MLLM-based methods** (UnifiedReward, etc.): require large amounts of VQA pairs for fine-tuning, which personalized image collections are insufficient to support.
- **ViPer**: the only existing personalization method, but it only implicitly leverages reference images for score regression without interpretable reasoning steps.
- **Core insight**: Although each user's preference is unique, the fundamental visual elements constituting preferences (art style, color, detail, art medium, saturation) are universal and can serve as a bridge across users.

## Method

### Overall Architecture: Predict-then-Assess Paradigm
Given a user's personalized reference images (liked/disliked) and two candidate images, PreferThinker performs two-stage CoT reasoning:
1. **Profile Prediction**: Predicts the user's visual preference profile and dispreference profile from reference images.
2. **Multi-dimensional Assessment**: Conducts multi-dimensional interpretable scoring of candidate images based on the predicted profile to derive the final result.

### Key Design 1: Visual Preference Profile
- Identifies the 15 most common visual elements from text prompts on the Lexica platform.
- A user study with 100 participants selects the top-5 by vote: **art style, color, detail, art medium, saturation**.
- Collects 288 relevant vocabulary terms to ensure profile diversity.
- Three advantages of the profile: describing complex preferences, enabling cross-user knowledge sharing, and supporting interpretable multi-dimensional assessment.

### Key Design 2: PreferImg-CoT Large-scale Dataset
- **PreferImg Construction**: 80K simulated users (including 20K multi-preference users), 1.36M images.
  - Randomly samples 5 visual preference elements to assign profiles.
  - Uses T2I models to generate reference and candidate images.
  - 190K initial prompts covering Lexica, DiffusionDB, and COCO.
- **CoT Annotation**: Claude 3.7 generates reasoning chains in the predict-then-assess format.
- **Quality Filtering**: Removes samples with logical inconsistencies or mismatched answers.
- Final dataset contains 60K high-quality CoT samples.

### Key Design 3: Two-stage Training + Similarity-aware Prediction Reward
**Stage 1 — Cold-start SFT**:
- Backbone model: Qwen2.5-VL-7B.
- Standard autoregressive cross-entropy loss: $\mathcal{L}_{SFT}(\theta) = -\mathbb{E}_{(x,y)\sim\mathcal{D}_{CoT}}\sum_{t=1}^{T}\log P(y_t|x,y_{<t};\theta)$

**Stage 2 — GRPO Reinforcement Learning**:
- Generates $G$ CoT outputs per input and computes group-normalized advantage $A_i$.
- PPO-clip objective with KL divergence regularization.

**Similarity-aware Prediction Reward**:
- **Text similarity**: SBERT computes semantic similarity $s_{text}$ between predicted and GT profiles.
- **Image similarity**: Images are generated from predicted and GT profiles respectively; DreamSim computes visual similarity $s_{img}$.
- Prediction reward: $r_{predict} = w_{img}s_{img} + w_{text}s_{text}$
- Mixed reward: $r = w_p r_{predict} + w_f r_{format} + w_a r_{accuracy}$ (weights 0.7/0.3/1.0)

## Key Experimental Results

### Main Results (Evaluation Accuracy, %)

| Method | Params | PreferImg Seen-SP | Seen-MP | Unseen-SP | Unseen-MP | PickaPic | Avg. |
|--------|--------|----------|---------|-----------|-----------|---------|------|
| PickScore | 986M | 49.6 | 48.4 | 51.2 | 56.4 | 67.9 | 54.7 |
| ViPer | 8B | 92.4 | 78.0 | 93.4 | 80.0 | 62.2 | 81.2 |
| GPT-4o | - | 94.2 | 80.4 | 92.2 | 85.2 | 65.7 | 83.5 |
| Claude 3.7 | - | 93.8 | 83.2 | 90.2 | 86.0 | 64.9 | 83.6 |
| **PreferThinker** | **7B** | **96.6** | **92.0** | **96.4** | **92.8** | 65.7 | **88.7** |

### Ablation Study

| Configuration | Seen-SP Acc | Seen-SP Pred | Unseen-MP Acc | Unseen-MP Pred |
|---------------|-------------|---------|------------|---------|
| Base (Qwen2.5-VL-7B) | 75.4 | 70.4 | 64.8 | 71.1 |
| + SFT | 92.0 | 84.2 | 81.6 | 74.2 |
| + SFT + RL | 93.8 | 85.0 | 88.4 | 79.5 |
| + SFT + RL + PR (Full) | **96.6** | **87.5** | **92.8** | **83.1** |

### Key Findings
1. **The 7B model surpasses all closed-source models**: PreferThinker comprehensively outperforms GPT-4o and Claude 3.7 on PreferImg.
2. **Most significant improvement in multi-preference (MP) settings**: +8.8% gain over SOTA on Seen-MP, demonstrating that the profile mechanism effectively handles complex preferences.
3. **RL stage substantially enhances generalization**: RL yields a larger improvement on unseen users (+6.8%) than on seen users (+4.6%).
4. **Prediction reward is critical**: More accurate profile prediction leads to more reasonable subsequent assessment (removing PR degrades prediction accuracy and causes assessment errors).
5. **Personalized profiles transfer to image generation**: Predicted preference profiles can guide personalized image generation.

## Highlights & Insights
- Proposes the preference profile concept to bridge across different users, elegantly addressing the scarcity of personalized data.
- The predict-then-assess paradigm enables interpretable multi-dimensional assessment, replacing black-box scoring.
- The similarity-aware prediction reward is cleverly designed, leveraging similarity signals in both text and image spaces.
- A 7B open-source model surpasses commercial models such as GPT-4o and Claude 3.7.

## Limitations & Future Work
- The PreferImg dataset is based on simulated users (T2I-generated images), which may introduce a distribution gap with real user preferences.
- Performance on the PickaPic real-user dataset is moderate (65.7%), as PickaPic annotations reflect general rather than personalized preferences.
- The profile is fixed at 5 visual elements, potentially failing to cover all personalization dimensions (e.g., composition, emotion).
- Training requires a T2I model to generate images for computing image similarity rewards, resulting in relatively high training cost.

## Related Work & Insights
- **Image preference assessment**: CLIP-based (PickScore, ImageReward, HPSv2) → MLLM-based (UnifiedReward, LLaVA-Reward).
- **Personalized preference**: ViPer (ECCV2024) is the first attempt but lacks interpretability.
- **Reasoning MLLMs**: GRPO post-training paradigm inspired by DeepSeek-R1.
- **Preference datasets**: ImageRewardDB, PickaPic, HPD_v2 primarily target general preferences.

## Rating
⭐⭐⭐⭐ (4/5)

The method is well-designed with innovations spanning data construction and training. The preference profile bridging concept is concise and effective. The primary concern is the gap between simulated data and real personalized preferences, which is also evidenced by the performance on PickaPic.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment](reasoning_as_representation_rethinking_visual_reinforcement_learning_in_image_qu.md)
- [\[ICLR 2026\] DiVE-k: Differential Visual Reasoning for Fine-grained Image Recognition](dive-k_differential_visual_reasoning_for_fine-grained_image_recognition.md)
- [\[CVPR 2026\] Reasoning-Driven Anomaly Detection and Localization with Image-Level Supervision](../../CVPR2026/reinforcement_learning/reasoning-driven_anomaly_detection_and_localization_with_image-level_supervision.md)
- [\[ICLR 2026\] FAPO: Flawed-Aware Policy Optimization for Efficient and Reliable Reasoning](fapo_flawed-aware_policy_optimization_for_efficient_and_reliable_reasoning.md)
- [\[ICLR 2026\] P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling](p-genrm_personalized_generative_reward_model_with_test-time_user-based_scaling.md)

<!-- RELATED:END -->
