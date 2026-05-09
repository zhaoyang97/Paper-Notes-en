---
title: >-
  [Paper Note] Start Small, Think Big: Curriculum-based Relative Policy Optimization for Visual Grounding
description: >-
  [AAAI 2026][Reinforcement Learning][Visual Grounding] This work identifies that CoT reasoning can be counterproductive in visual grounding, and proposes CuRPO (Curriculum-based Relative Policy Optimization), which leverages CoT length and gIoU reward as data complexity proxies for curriculum-based RL training, achieving up to +12.52 mAP improvement over Visual-RFT on RefCOCO.
tags:
  - AAAI 2026
  - Reinforcement Learning
  - Visual Grounding
  - Curriculum Learning
  - GRPO
  - Chain-of-Thought
  - Vision-Language Models
date: 2026-05-08
content_hash: 9299c6db21149095
---

# Start Small, Think Big: Curriculum-based Relative Policy Optimization for Visual Grounding

**Conference**: AAAI 2026
**arXiv**: [2511.13924](https://arxiv.org/abs/2511.13924)
**Code**: [github.com/qyoung-yan/CuRPO](https://github.com/qyoung-yan/CuRPO)
**Area**: Reinforcement Learning
**Keywords**: Visual Grounding, Curriculum Learning, GRPO, Chain-of-Thought, Vision-Language Models

## TL;DR

This work identifies that CoT reasoning can be counterproductive in visual grounding, and proposes CuRPO (Curriculum-based Relative Policy Optimization), which leverages CoT length and gIoU reward as data complexity proxies for curriculum-based RL training, achieving up to +12.52 mAP improvement over Visual-RFT on RefCOCO.

## Background & Motivation

### State of the Field

Visual Grounding requires models to localize target objects in images based on textual descriptions and output bounding box coordinates. Recent combinations of CoT prompting and RL (e.g., Visual-RFT using GRPO) have achieved success across various visual reasoning tasks.

### Three Counter-Intuitive Findings

Through systematic experiments, this paper reveals three important counter-intuitive phenomena:

**Finding 1: CoT reasoning degrades visual grounding performance.** Requiring the model to generate CoT reasoning prior to outputting bounding boxes not only fails to improve but actually reduces localization accuracy. As shown in Figure 1, CoT-guided models produce incorrect boxes due to misinterpretation of textual context, whereas models without CoT localize correctly. With 40 samples, the no-CoT model achieves an mIoU of 35.6, while the CoT model requires 239 samples to reach only 34.4.

**Finding 2: Longer CoT ≈ Harder Task.** Statistical analysis reveals a significant negative correlation between CoT length and gIoU reward:
- Pearson correlation: -0.4395 ($p = 1.04 \times 10^{-12}$)
- Spearman rank correlation: -0.4268 ($p = 5.34 \times 10^{-12}$)
- Kendall's Tau: -0.2981 ($p = 6.81 \times 10^{-12}$)

Theoretical interpretation: modeling the per-step success probability as $p_c < 1$, the overall success probability of a $C$-step reasoning chain $\text{Pr}(\text{success}) = \prod_{c=1}^{C} p_c$ decays exponentially with the number of steps.

**Finding 3: More Data ≠ Better Performance.** Scaling up training set size does not consistently improve performance — CoT-equipped models even exhibit performance oscillation and stagnation, whereas models without CoT improve steadily with more data.

### Core Motivation

Since CoT length serves as a reliable proxy for task difficulty, and the ordering of training data complexity affects learning outcomes, CoT length and reward signals can be leveraged to construct a curriculum training strategy that guides the model to learn easy examples before hard ones.

## Method

### Overall Architecture

CuRPO is a training framework that integrates Curriculum Learning with GRPO-based reinforcement learning. The core pipeline is: compute a complexity metric for each sample → sort and group samples by difficulty → train in stages following an easy-to-hard ordering → apply GRPO optimization within each stage.

### Key Designs

#### 1. **Reward Function Design**: gIoU + Format Reward

Generalized IoU (gIoU) is adopted in place of standard IoU, since standard IoU provides zero gradient when the predicted and ground-truth boxes do not overlap.

$$\text{gIoU}(A, B) = \text{IoU}(A, B) - \frac{C - (A \cup B)}{C}$$

where $C$ is the area of the smallest enclosing rectangle containing both $A$ and $B$. The gIoU range $[-1, 1]$ is linearly scaled to $[0, 2]$ to reduce excessive negative feedback.

Total reward: $R_d = R_{\text{visual}} + R_{\text{format}}$

- $R_{\text{visual}}$: localization accuracy reward based on gIoU
- $R_{\text{format}}$: reward ensuring correct output format

#### 2. **Curriculum Training Strategy**: Difficulty Ordering Based on CoT Length

Detailed procedure:

1. Generate multiple CoT responses (typically 8) per training sample using a pretrained VLM and compute the average length.
2. Sort all samples by CoT length in ascending order.
3. Within each CoT length interval (bin size: 50 tokens), re-sort by reward value.
4. Train in stages:
    - **Initial Stage**: Use only short-CoT samples (easiest) to learn basic visual reasoning patterns.
    - **Intermediate Stage**: Gradually introduce medium-length CoT samples covering moderate complexity scenarios.
    - **Advanced Stage**: Incorporate long-CoT samples to challenge complex reasoning tasks.

Three sorting variants are explored:
- **CuRPO (Length)**: sorted by CoT length
- **CuRPO (Reward)**: sorted by gIoU reward
- **CuRPO (Random)**: random ordering (baseline to verify the effectiveness of curriculum learning itself)

#### 3. **GRPO Training Objective**: Group Relative Policy Optimization

Given a query $q$, policy $\pi_\theta$ samples $G$ candidate outputs $\{o_i\}$, and the group-normalized advantage is computed as:

$$A_i = \frac{r_i' - \mu'}{\sigma'}, \quad \mu' = \frac{1}{G}\sum_{j=1}^{G} r_j', \quad \sigma' = \sqrt{\frac{1}{G}\sum_{j=1}^{G}(r_j' - \mu')^2}$$

The final GRPO objective combines a clipped surrogate loss with KL divergence regularization:

$$L_{\text{GRPO}}(\theta) = -\frac{1}{G}\sum_{i=1}^{G}\min(c_i A_i, \text{clip}(c_i, 1-\epsilon, 1+\epsilon)A_i) - \beta D_{KL}(\pi_\theta \| \pi_{\text{ref}})$$

where $c_i = \frac{\pi_\theta(o_i|q)}{\pi_{\text{old}}(o_i|q)}$ is the probability ratio between the updated and old policies.

### Loss & Training

- Base model: Qwen2-VL-2B
- Baseline: zero-curriculum fine-tuned Qwen2-VL-2B under the "with CoT" setting
- Key design: the model does **not** explicitly output CoT; it directly predicts bounding box coordinates (CoT is used solely as a pre-training difficulty estimator)
- Evaluation metrics: mIoU and mAP

## Key Experimental Results

### Main Results

**LISA Dataset**:

| Method | Model | Training Samples | mIoU |
|------|------|-----------|------|
| SFT | Qwen2-VL-2B | 239 | 29.7 |
| GroundingDINO | X-Decoder | 239 | 28.5 |
| Visual-RFT | Qwen2-VL-2B | 239 | 34.4 |
| **CuRPO (Ours)** | Qwen2-VL-2B | **50** | **37.4 (+3.0)** |
| **CuRPO (Ours)** | Qwen2-VL-2B | 200 | **38.7 (+4.3)** |
| **CuRPO (Ours)** | Qwen2-VL-2B | 239 | **38.4 (+4.0)** |

With only 50 samples, CuRPO surpasses Visual-RFT trained on all 239 samples.

**RefCOCO Series (mAP)**:

| Dataset | Qwen2-VL-2B | Visual-RFT | CuRPO (Random) | CuRPO (Length) | CuRPO (Reward) |
|--------|------------|------------|----------------|----------------|----------------|
| RefCOCO (val) | 11.57 | 21.28 | 33.09 (+11.81) | **33.80 (+12.52)** | 32.64 (+11.36) |
| RefCOCO (test) | 10.70 | 20.38 | 29.92 (+9.54) | **31.42 (+11.04)** | 27.89 (+7.51) |
| RefCOCO+ (val) | 13.72 | 18.41 | 26.82 (+8.41) | 26.18 (+7.77) | **26.85 (+8.44)** |
| RefCOCO+ (test) | 16.11 | 20.90 | 24.34 (+3.44) | **25.10 (+4.20)** | 23.55 (+2.65) |
| RefCOCOg (val) | 14.89 | 23.39 | 27.98 (+4.59) | 29.27 (+5.88) | **32.65 (+9.26)** |

### Ablation Study

| Ordering Strategy | RefCOCO (val) mAP | Notes |
|---------|-------------------|------|
| No Curriculum (Visual-RFT) | 21.28 | Baseline |
| CuRPO (Random) | 33.09 | Even random curriculum yields substantial gains |
| CuRPO (Length) | **33.80** | Best on simpler datasets |
| CuRPO (Reward) | 32.64 | Best on complex-description datasets (RefCOCOg) |

### Key Findings

1. **Curriculum learning is effective on its own**: Even CuRPO (Random) outperforms Visual-RFT by +11.81 mAP on RefCOCO val, demonstrating the intrinsic value of curriculum-based RL training.
2. **Length ordering suits simpler datasets** (RefCOCO/RefCOCO+), while **Reward ordering suits complex-description datasets** (RefCOCOg).
3. **Advantage is amplified in low-data regimes**: With only 50 samples, CuRPO substantially outperforms baselines trained on full data.
4. **Explicit CoT generation consistently underperforms no-CoT**: This finding holds across all data scales and ordering strategies.
5. **Per-category analysis**: The most significant improvements are observed on categories that are difficult to precisely localize, such as chair, bed, and toilet.

## Highlights & Insights

- **Identifying the "overthinking" problem of CoT in visual grounding** is a significant empirical contribution that challenges the prevailing assumption that CoT is universally beneficial.
- **Using CoT length as a difficulty proxy** is an elegant and efficient approach that requires no additional annotation of task difficulty.
- **Surpassing full-data SOTA with only 50 samples** highlights the substantial data efficiency advantages of curriculum learning.
- CuRPO's design is **orthogonal** — it requires no modifications to model architecture and can be directly applied to any VLM + GRPO combination.

## Limitations & Future Work

- Validation is limited to Qwen2-VL-2B; performance on larger models (7B/72B) remains unknown.
- CoT length as a difficulty proxy may not generalize to other tasks such as VQA or image captioning.
- The curriculum stage boundaries (CoT length bin size = 50 tokens) are manually set; adaptive partitioning may yield better results.
- Instance segmentation tasks are not supported; the method is restricted to bounding box detection.
- The choice of ordering strategy depends on dataset characteristics, with no automatic selection mechanism.

## Related Work & Insights

- Shares conceptual similarity with FASTCURL, which uses prompt length as a reasoning complexity proxy for curriculum-based RL training.
- Visual-RFT is the most direct baseline, using GRPO with gIoU reward to train VLMs for visual grounding.
- Complementary to CoT length adaptation methods (e.g., SelfBudgeter): this work finds that certain tasks do not benefit from CoT at all, while SelfBudgeter focuses on regulating CoT length.
- Implication: the importance of data difficulty and ordering in visual reasoning tasks may be substantially underestimated.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The insight of using CoT length as a difficulty indicator is novel, though curriculum learning and GRPO themselves are not new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-dataset validation is thorough, but experiments are limited to the 2B model scale.
- **Writing Quality**: ⭐⭐⭐⭐ — The three findings are presented clearly and persuasively with well-motivated exposition.
- **Value**: ⭐⭐⭐⭐⭐ — Substantial empirical gains (+12.52 mAP) and strong few-shot performance highlight high practical value.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] InfiGUI-G1: Advancing GUI Grounding with Adaptive Exploration Policy Optimization](infigui-g1_advancing_gui_grounding_with_adaptive_exploration_policy_optimization.md)
- [\[AAAI 2026\] Realistic Curriculum Reinforcement Learning for Autonomous and Sustainable Marine Vessel Navigation](realistic_curriculum_reinforcement_learning_for_autonomous_and_sustainable_marin.md)
- [\[AAAI 2026\] Think, Speak, Decide: Language-Augmented Multi-Agent Reinforcement Learning for Economic Decision-Making](think_speak_decide_language-augmented_multi-agent_reinforcement_learning_for_eco.md)
- [\[AAAI 2026\] Behaviour Policy Optimization: Provably Lower Variance Return Estimates for Off-Policy Reinforcement Learning](behaviour_policy_optimization_provably_lower_variance_return_estimates_for_off-p.md)
- [\[AAAI 2026\] Well Begun, Half Done: Reinforcement Learning with Prefix Optimization for LLM Reasoning](well_begun_half_done_reinforcement_learning_with_prefix_optimization_for_llm_rea.md)

<!-- RELATED:END -->
