---
title: >-
  [Paper Note] NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning
description: >-
  [CVPR 2026][Autonomous Driving][VLA model] NoRD demonstrates that autonomous driving VLAs require neither large-scale reasoning annotations nor massive datasets. By identifying the root cause of GRPO failure on weak SFT policies as **difficulty bias** — wherein learning signals from high-variance rollout groups are suppressed — it replaces standard GRPO with Dr. GRPO for RL post-training. Using less than 60% of the data, no reasoning annotations, and 3× fewer tokens, NoRD achieves competitive performance against reasoning-based VLAs on NAVSIM (85.6 PDMS) and WaymoE2E (7.709 RFS).
tags:
  - CVPR 2026
  - Autonomous Driving
  - VLA model
  - reasoning-free driving
  - data efficiency
  - Dr.GRPO
  - reinforcement learning post-training
date: 2026-05-08
content_hash: 507316a270781bb7
---

# NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning

**Conference**: CVPR 2026
**arXiv**: [2602.21172](https://arxiv.org/abs/2602.21172)
**Code**: N/A
**Area**: Autonomous Driving
**Keywords**: VLA model, reasoning-free driving, data efficiency, Dr.GRPO, reinforcement learning post-training

## TL;DR

NoRD demonstrates that autonomous driving VLAs require neither large-scale reasoning annotations nor massive datasets. By identifying the root cause of GRPO failure on weak SFT policies as **difficulty bias** — wherein learning signals from high-variance rollout groups are suppressed — it replaces standard GRPO with Dr. GRPO for RL post-training. Using less than 60% of the data, no reasoning annotations, and 3× fewer tokens, NoRD achieves competitive performance against reasoning-based VLAs on NAVSIM (85.6 PDMS) and WaymoE2E (7.709 RFS).

## Background & Motivation

- **Background**: The standard training pipeline for autonomous driving VLAs consists of large-scale SFT, chain-of-thought reasoning annotations, and GRPO post-training. Models such as AutoVLA achieve strong performance but require 212K+ samples, dense reasoning annotations, and additional inference-time latency from reasoning tokens. These three costs — data, annotation, and computation — are not scalable.

- **Limitations of Prior Work**: Prior theoretical and empirical work has questioned the necessity of reasoning: (a) the "Reasoning-Planning Decoupling Hypothesis" suggests that textual priors alone can match the performance of full multimodal reasoning; (b) RL does not create new reasoning capabilities but only optimizes over distributions already present in the SFT model.

- **Key Challenge**: An initial attempt training NoRD-base (Qwen-2.5VL-3B) on 80K samples without reasoning annotations, followed by GRPO post-training, yielded only a +0.67% improvement (76.66→77.18), compared to AutoVLA's +9% gain. This appeared to confirm the indispensability of reasoning data. However, further analysis reveals that the failure stems not from the weakness of the SFT policy itself, but from a systematic flaw in GRPO's advantage normalization. When group standard deviation $\text{std}$ serves as the denominator, low-variance groups (easy or extremely hard scenarios) receive amplified advantages, while high-variance groups (moderate difficulty, comprising the majority) are suppressed. Weak SFT models tend to produce a large proportion of high-variance rollouts, preventing GRPO from learning from the bulk of training samples.

- **Goal**: To develop a data-efficient, reasoning-free VLA for autonomous driving by diagnosing and correcting the difficulty bias in GRPO, achieving competitive performance with significantly reduced data and annotation cost.

## Method

### Overall Architecture

**Input**: RGB images from three cameras (front, front-left, front-right) + historical ego-trajectory + current speed/acceleration → Qwen-2.5VL-3B-Instruct → direct prediction of future trajectory tokens (no reasoning tokens).

**Training pipeline**: (1) Small-scale SFT (80K samples for NAVSIM / 12K for WaymoE2E) → (2) Dr. GRPO reinforcement learning post-training.

### Key Designs

1. **K-disc Trajectory Tokenization**

   - **Function**: Discretizes continuous trajectories into a codebook of 2,048 tokens.
   - **Mechanism**: All training trajectories are interpolated to 10 Hz → segmented into 0.5s clips → K-means clustering into 2,048 clusters → cluster centers form a discrete trajectory vocabulary.
   - **Design Motivation**: Casting trajectory prediction as next-token prediction aligns naturally with the autoregressive paradigm of VLMs. A codebook size of 2,048 balances precision and generalization.
   - **Token embedding initialization**: New token embeddings are sampled from a multivariate normal distribution parameterized by the mean and covariance of Qwen's existing token embeddings, ensuring compatibility with the pre-trained distribution.

2. **Difficulty Bias Analysis and Dr. GRPO**

   - **GRPO advantage**: $\hat{A}_{i,t}^{\text{GRPO}} = \frac{r(o_i|x) - \frac{1}{G}\sum_{j=1}^G r(o_j|x)}{\text{std}_{j=1,...,G}(r(o_j|x))}$
   - **Key Challenge**: The denominator $\text{std}$ becomes very small ($\ll 1$) for low-variance groups, amplifying their advantages, while large $\text{std}$ in high-variance groups suppresses theirs. Weak SFT models produce a polarized reward distribution: simple scenarios (group mean $\geq 0.8$) and extremely hard scenarios ($\leq 0.15$) exhibit low variance, whereas moderate-difficulty scenarios ($0.2$–$0.65$) — the majority — exhibit high variance.
   - **Dr. GRPO correction**: Standard deviation normalization is removed, yielding $\hat{A}_{i,t}^{\text{DrGRPO}} = r(o_i|x) - \frac{1}{G}\sum_{j=1}^G r(o_i|x)$, ensuring that hard scenarios also contribute sufficient gradient signal.
   - **Auxiliary stabilization**: DAPO-style asymmetric clipping ($1-\epsilon_l, 1+\epsilon_h$) prevents entropy collapse; KL divergence regularization is not used.

3. **Data-Efficient SFT**

   - **Function**: Intentionally limits SFT to small-scale data, shifting the primary learning burden to the RL post-training stage.
   - **NAVSIM**: 80K samples (vs. AutoVLA's 212K+).
   - **WaymoE2E**: 12K samples for SFT + 450 samples for RLFT.
   - **Design Motivation**: To verify that VLAs do not require large-scale data, and that RL post-training can compensate for performance gaps incurred during SFT.

### Loss & Training

- **SFT stage**: Standard next-token prediction cross-entropy loss.
- **RL stage**: Dr. GRPO objective with rewards from PDM score (NAVSIM) or RFS (WaymoE2E), plus auxiliary rewards for trajectory length and output format (weight 0.25), normalized to $[0, 1]$.
- **Training details**: SFT uses 16× A100, lr=5e-5, batch=128; RLFT uses 30× A100 (NAVSIM) or 32 GPUs (WaymoE2E), lr=5e-6/1e-6, group size=8, sampling temperature 1.0.
- **Inference**: Deterministic sampling at temperature 0.01; no reasoning token overhead.

## Key Experimental Results

### Main Results

| Method | Reasoning? | Data | NAVSIM PDMS↑ | WaymoE2E RFS↑ | WaymoE2E ADE↓ |
|---|---|---|---|---|---|
| UniAD | N/A | — | 83.4 | — | — |
| DiffusionDrive | N/A | — | 88.1 | — | — |
| AutoVLA | Yes | 212K+ | 89.1 | 7.556 | 1.3507 |
| RecogDrive | Yes | 2.7M+ | 89.6 | — | — |
| Poutine | Yes | 212K+ | — | **7.986** | 1.2055 |
| **NoRD** | **No** | **<90K** | **85.6** | **7.709** | **1.2504** |

- NoRD-BoN (best-of-6) achieves 92.4 on NAVSIM, surpassing AutoVLA-BoN (92.1).
- On WaymoE2E, NoRD ranks third in RFS and is the only top-performing model without reasoning or ensembling.
- NoRD outperforms all competitors including Poutine on ADE, demonstrating superior trajectory precision.

### Ablation Study

| Configuration | NAVSIM PDMS↑ | Note |
|---|---|---|
| NoRD-base (SFT only) | 76.66 | Weak SFT policy baseline |
| NoRD-base + GRPO | 77.18 (+0.67%) | GRPO nearly ineffective |
| NoRD-base + Dr. GRPO | **85.62 (+11.68%)** | Dr. GRPO successfully optimizes weak policy |

### Key Findings

- **GRPO failure on weak policies is caused by difficulty bias**: During training, GRPO only optimizes a small subset of low-variance samples (already-known easy behaviors), while the group-mean reward distribution of high-variance samples barely shifts.
- **Dr. GRPO unlocks learning from moderate-difficulty samples**: The group-mean reward distribution in the $[0.2, 0.65]$ range shifts significantly rightward during training, indicating that the model learns complex maneuvers such as sharp turns and lane changes.
- **NoRD is the most token-efficient and inference-efficient VLA**: The absence of reasoning tokens results in 3× token reduction and substantially lower inference latency.
- **Exceptional data efficiency**: On WaymoE2E, NoRD uses only 12K samples (vs. Poutine's 200K+), achieves an RFS gap of only 0.277, and attains lower ADE.

## Highlights & Insights

- **First identification of difficulty bias in autonomous driving VLA training**: Dr. GRPO is transferred from the LLM reasoning domain to autonomous driving, demonstrating highly effective cross-domain adaptation.
- **Challenges the assumption of reasoning necessity**: Empirical results refute the prevailing view that VLAs must employ CoT reasoning to achieve high performance, providing a foundation for lightweight VLA design.
- **Diagnostic methodology for training failure**: Analyzing the reward distribution characteristics of rollouts (mean-variance relationship) to diagnose RL optimization failures offers a methodology of broad applicability.
- **Pareto frontier analysis**: On the efficiency-performance plane, NoRD occupies a uniquely favorable position combining high efficiency and high performance.

## Limitations & Future Work

- **Absolute performance gap remains**: NoRD (85.6) vs. AutoVLA (89.1) on NAVSIM, a gap of 3.5 points. Reasoning data may still be beneficial in certain scenarios.
- **Dr. GRPO is not a complete solution**: The paper acknowledges that Dr. GRPO mitigates rather than eliminates difficulty bias, leaving room for further improvement.
- **Model scale fixed at 3B**: Performance at larger or smaller scales is not explored; scaling analysis is absent.
- **Only three front-facing cameras are used**: Rear and rear-side views are not incorporated, which may limit performance in complex traffic scenarios.
- **Future directions**: (a) Design better reward shaping to mitigate polarized reward distributions; (b) Incorporate a small amount of reasoning annotations for hybrid training; (c) Extend evaluation to closed-loop settings.

## Related Work & Insights

- **vs. AutoVLA**: AutoVLA is the representative reasoning-based VLA (212K data + CoT + GRPO). NoRD achieves competitive performance with less than 40% of the data and no reasoning, demonstrating that reasoning is not indispensable.
- **vs. EMMA/SimLingo**: Both are reasoning-free VLAs, but were previously validated only on simpler benchmarks (e.g., nuScenes). NoRD is the first to demonstrate viability on complex benchmarks such as NAVSIM and WaymoE2E.
- **vs. LLM reasoning literature**: Dr. GRPO was originally designed to address difficulty bias in LLM mathematical reasoning. This paper provides the first validation of its effectiveness in autonomous driving with dense reward signals.
- **Insights**: The potential of RL post-training has been severely underestimated due to assumptions about required SFT scale. A correct optimization algorithm may substitute for large volumes of data. The paradigm of a weak policy combined with a strong optimizer may represent a more economical approach.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First identification of GRPO difficulty bias in autonomous driving; effective cross-domain innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual-benchmark evaluation on NAVSIM and WaymoE2E; detailed reward distribution analysis and training dynamics.
- Writing Quality: ⭐⭐⭐⭐⭐ Deep problem analysis and excellent visualizations (reward distribution evolution in Figures 2/3 are highly convincing).
- Value: ⭐⭐⭐⭐⭐ Challenges prevailing paradigm assumptions and opens new pathways toward data-efficient and inference-efficient VLAs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Drive My Way: Preference Alignment of Vision-Language-Action Model for Personalized Driving](drive_my_way_preference_alignment_of_vision-language-action_model_for_personaliz.md)
- [\[CVPR 2026\] Learning Vision-Language-Action World Models for Autonomous Driving](vla_world_learning_vision_language_action_world_models_for_autonomous_driving.md)
- [\[NeurIPS 2025\] AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](../../NeurIPS2025/autonomous_driving/autovla_a_vision-language-action_model_for_end-to-end_autonomous_driving_with_ad.md)
- [\[ICLR 2026\] ST4VLA: Spatially Guided Training for Vision-Language-Action Models](../../ICLR2026/autonomous_driving/st4vla_spatially_guided_training_for_vision-language-action_models.md)
- [\[CVPR 2026\] Traffic Scene Generation from Natural Language Description for Autonomous Vehicles with Large Language Model](ttsg_text_to_traffic_scene_generation_from_natural_language.md)

</div>

<!-- RELATED:END -->
