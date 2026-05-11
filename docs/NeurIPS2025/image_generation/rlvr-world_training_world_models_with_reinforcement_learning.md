---
title: >-
  [Paper Note] RLVR-World: Training World Models with Reinforcement Learning
description: >-
  [NeurIPS 2025][Image Generation][World Models] This paper proposes the RLVR-World framework, extending the Reinforcement Learning with Verifiable Rewards (RLVR) paradigm to world model training. By directly optimizing ta…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "World Models"
  - "RLVR"
  - "GRPO"
  - "Video Prediction"
  - "Autoregressive Generation"
date: 2026-05-08
content_hash: df78a2c7c4d82024
---

# RLVR-World: Training World Models with Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2505.13934](https://arxiv.org/abs/2505.13934)
**Code**: [GitHub](https://thuml.github.io/RLVR-World)
**Area**: World Models / RL Post-Training
**Keywords**: World Models, RLVR, GRPO, Video Prediction, Autoregressive Generation

## TL;DR

This paper proposes the RLVR-World framework, extending the Reinforcement Learning with Verifiable Rewards (RLVR) paradigm to world model training. By directly optimizing target metrics (e.g., prediction accuracy, perceptual quality) as verifiable rewards, the framework achieves significant improvements on both language and video world models.

## Background & Motivation

World models aim to predict environment state transitions under action interventions, serving as a core component in model-based planning and reinforcement learning. Existing world models predominantly adopt maximum likelihood estimation (MLE) as their training objective (e.g., next-token prediction for language models, variational lower-bound optimization for diffusion models). However, such surrogate objectives are fundamentally misaligned with the true purpose of world models — **accurate state transition prediction and perceptual quality**.

Specifically, the limitations of MLE manifest at three levels:

**Objective Misalignment**: Likelihood objectives are not directly aligned with downstream evaluation metrics (e.g., accuracy, LPIPS), leading to degenerate behaviors such as repetitive generation and hallucination.

**Non-End-to-End Optimization**: Autoregressive architectures built on discrete tokenizers cannot directly optimize pixel-level metrics.

**Multi-Step Error Accumulation**: Teacher-forcing training ignores error propagation in multi-step prediction.

Inspired by the success of RLVR in enhancing mathematical and coding capabilities in reasoning models such as DeepSeek-R1, the authors propose extending the RLVR paradigm to world models: replacing learned reward models with rule-based verifiable rewards to directly optimize prediction metrics.

## Method

### Overall Architecture

RLVR-World unifies world models across different modalities under an autoregressive sequence modeling framework. The core mechanism consists of three steps:
1. Encode the current state and action into a "question" token sequence $q(s,a)$, and encode the next state into an "answer" token sequence $o(s')$.
2. Pretrain the world model with MLE.
3. Apply RLVR post-training, using prediction metrics as verifiable rewards for fine-tuning.

### Key Designs

1. **Unified Sequence Modeling**: Regardless of modality — text, video, or sensor data — inputs are converted into token sequences via modality-specific tokenization. Language uses BPE; images/videos use a discrete visual tokenizer (iVideoGPT's compressed tokenizer); low-dimensional continuous values use uniform binning. This unification enables RLVR to generalize across modalities.

2. **Prediction Metrics as Verifiable Rewards**: Given input $q(s,a)$, the model generates a set of samples $\{o_i\}_{i=1}^G$, decodes predicted states $\hat{s}_i'$, and computes rewards by comparing against ground truth $s'$:
   $$R_i = \text{sign}(D) \cdot D(\hat{s}_i', s')$$
   where $\text{sign}(D) = -1$ for lower-is-better metrics (e.g., MSE, LPIPS) and $\text{sign}(D) = 1$ otherwise. The key advantage of this design is that rewards are fully verifiable and require no human annotation.

3. **GRPO Optimization**: Group Relative Policy Optimization (GRPO) is adopted, eliminating the need for a separate value function. Given question $q$, a group of responses $\{o_i\}_{i=1}^G$ is sampled, and within-group normalized advantages are computed:
   $$\hat{A}_{i,t} = \frac{R_i - \text{mean}(\{R_i\}_{i=1}^G)}{\text{std}(\{R_i\}_{i=1}^G)}$$
   Policy updates are performed with a clipped objective and KL divergence penalty.

### Loss & Training

- **Pretraining**: Standard MLE objective $\mathcal{J}_{\text{MLE}}(\theta) = \sum_{t} \log p_\theta(o_t(s') | q(s,a), o_{<t}(s'))$
- **RLVR Post-Training**: GRPO objective incorporating a clipped ratio term and KL regularization.
- **Language World Models**: SFT followed by RLVR, using binary accuracy rewards or task-specific rewards.
- **Video World Models**: Reward defined as the negative sum of L1 loss and LPIPS: $R = -\sum_\tau [L_1(\hat{s}_\tau, s_\tau) + \text{LPIPS}(\hat{s}_\tau, s_\tau)]$

## Key Experimental Results

### Main Results

**Text-Based Game State Prediction (ByteSized32)**

| Model | Unchanged Acc | Changed Acc | Overall Acc |
|------|:---:|:---:|:---:|
| Base (1.5B) | 11.98% | 0.08% | 7.11% |
| SFT | 38.88% | 24.21% | 32.87% |
| **RLVR-World (binary)** | 73.57% | 33.14% | 57.01% |
| **RLVR-World (task-specific)** | **83.66%** | **33.80%** | **63.24%** |
| GPT-4 | 73.90% | 51.60% | 64.76% |

**Video World Model: RT-1 Multi-Step Prediction**

| Model | Repetition Rate↓ | MSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|:---:|:---:|:---:|:---:|:---:|
| Base | 48.6% | 0.659 | 23.1 | 80.9 | 14.8 |
| Base (w/ rep. rejection) | 0.0% | 0.593 | 23.3 | 81.0 | 14.4 |
| **RLVR-World** | **9.9%** | **0.486** | **24.1** | **82.4** | **13.4** |
| Relative Gain Δ | +79.6% | +26.1% | +4.5% | +1.9% | +9.2% |

### Ablation Study

| Configuration | Key Metric | Note |
|------|---------|------|
| Different metrics as rewards | Each metric best optimized by its corresponding reward | Training with LPIPS yields best LPIPS; training with MSE yields best MSE |
| GRPO group size = 2 → 16 | Convergence speed and final performance consistently improve | Larger group size provides better exploration |
| Adding repetition penalty reward | Repetition rate 0%, LPIPS = 13.7 | Eliminates repetition while maintaining prediction quality |
| Test-time scaling | RLVR single-sample > Base best-of-5 | Base catches up to RLVR at N = 100 |

### Key Findings

- RLVR achieves significant improvements within only a few hundred gradient update steps, whereas MLE requires hundreds of thousands of steps.
- RLVR effectively mitigates the repetitive frame problem in video world models, reducing the repetition rate from 48.6% to 9.9%.
- The RLVR-trained world models yield performance gains in downstream model-predictive control (web navigation) and policy evaluation (robotic manipulation).

## Highlights & Insights

- The concept of **RLVR as a general post-training paradigm** is highly forward-looking: it is not limited to reasoning models but can be extended to any generative model with verifiable metrics.
- The paper draws an insightful analogy between world models and reasoning models: both benefit from transitioning away from surrogate objectives toward task-aligned direct optimization.
- The compressed tokenizer from iVideoGPT addresses the sequence length explosion problem in video modality, making GRPO feasible for video world models.
- The Real2Sim policy evaluation experiments demonstrate practical applicability.

## Limitations & Future Work

- RLVR training typically converges within a few hundred steps, with performance ceilings bounded by the capacity of the base model.
- Test-time scaling has an upper bound: the base model can catch up to RLVR as $N$ increases.
- Current video world models are trained on specific datasets; out-of-distribution generalization has not been validated.
- Reward design relies on conventional visual metrics (MSE/LPIPS) and does not incorporate physical constraints or temporal consistency regularization.

## Related Work & Insights

- The work shares a conceptual lineage with DeepSeek-R1's RLVR approach, while extending it to the generative modeling domain.
- It is complementary to works on fine-tuning diffusion models via DPO/RLHF; RLVR's advantage lies in eliminating the need for a learned reward model.
- The proposed framework offers a viable post-training paradigm for future general-purpose world models such as Cosmos.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First to extend RLVR from reasoning models to world models, validated across language and video modalities; conceptually novel and potentially impactful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers diverse scenarios including text games, web navigation, and robotic manipulation with thorough ablations; lacks experiments on larger-scale base models.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Well-structured paper with precise motivation articulation and excellent figure design.
- **Value**: ⭐⭐⭐⭐⭐ The proposed general paradigm can be broadly applied to post-training optimization of various generative models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Long-Context State-Space Video World Models](../../ICCV2025/image_generation/long-context_state-space_video_world_models.md)
- [\[NeurIPS 2025\] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation](coreinforcement_learning_for_unified_multimodal_understandin.md)
- [\[NeurIPS 2025\] Towards Robust Zero-Shot Reinforcement Learning](towards_robust_zero-shot_reinforcement_learning.md)
- [\[ICCV 2025\] Aether: Geometric-Aware Unified World Modeling](../../ICCV2025/image_generation/aether_geometric-aware_unified_world_modeling.md)
- [\[NeurIPS 2025\] DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution](dove_efficient_one-step_diffusion_model_for_real-world_video_super-resolution.md)

</div>

<!-- RELATED:END -->
