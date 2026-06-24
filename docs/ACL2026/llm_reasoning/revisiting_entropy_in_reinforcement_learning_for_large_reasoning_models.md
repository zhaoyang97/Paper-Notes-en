---
title: >-
  [Paper Note] Revisiting Entropy in Reinforcement Learning for Large Reasoning Models
description: >-
  [ACL 2026 Findings][Reasoning][Entropy collapse] This work systematically investigates the entropy dynamics of LLMs during RLVR training, revealing that positive-advantage tokens are the primary drivers of entropy collapse. It introduces Positive-Advantage Reweighting to effectively regulate model entropy by dynamically adjusting the loss weights of these tokens.
tags:
  - "ACL 2026 Findings"
  - "Reasoning"
  - "Entropy collapse"
  - "RLVR"
  - "GRPO"
  - "Positive-advantage reweighting"
  - "Reasoning models"
date: 2026-05-08
content_hash: 3daa47b3d3f76afb
---

# Revisiting Entropy in Reinforcement Learning for Large Reasoning Models

**Conference**: ACL 2026 Findings  
**arXiv**: [2511.05993](https://arxiv.org/abs/2511.05993)  
**Code**: [GitHub](https://github.com/cordercorder/EntropyRL)  
**Area**: LLM Reasoning  
**Keywords**: Entropy collapse, RLVR, GRPO, Positive-advantage reweighting, Reasoning models

## TL;DR

This work systematically investigates the entropy dynamics of LLMs during RLVR training, revealing that positive-advantage tokens are the primary drivers of entropy collapse. It introduces Positive-Advantage Reweighting to effectively regulate model entropy by dynamically adjusting the loss weights of these tokens.

## Background & Motivation

- **Background**: Represented by OpenAI o1, DeepSeek-R1, and Kimi k1.5, RLVR (Reinforcement Learning with Verifiable Rewards) has become a mainstream paradigm for enhancing LLM reasoning capabilities, showing significant performance in mathematics and coding tasks.
- **Limitations of Prior Work**: During RLVR training, LLM entropy typically decreases sharply (i.e., "entropy collapse"), causing models to converge prematurely to sub-optimal local optima and lose exploration capabilities as probability mass concentrates on a few tokens.
- **Key Challenge**: While existing methods (e.g., DAPO's Clip-Higher, adaptive entropy regularization, Clip-Cov) attempt to mitigate entropy collapse, a systematic study of entropy dynamics in RLVR is lacking. Three key questions remain unexplored: (1) What is the correlation between entropy and performance? (2) What factors determine entropy dynamics? (3) How can entropy be effectively regulated to improve performance?
- **Goal**: To comprehensively analyze entropy dynamics in RLVR training through extensive experiments, identify the root causes of entropy collapse, and propose a simple yet effective regulation method.
- **Key Insight**: Starting from theoretical gradient analysis, this work distinguishes the different impacts of positive-advantage and negative-advantage tokens on entropy, rather than merely patching the issue with regularization.
- **Core Idea**: Positive-advantage tokens are the primary cause of entropy collapse—they increase the probability of sampled tokens and suppress unsampled ones, leading to excessive distribution concentration. Controlling their loss weight allows for precise entropy regulation.

## Method

### Overall Architecture

Based on the standard GRPO-based RLVR training pipeline, a Positive-Advantage Reweighting mechanism is introduced into the optimization objective. A hyperparameter $\lambda$ controls the loss weight of positive-advantage tokens to dynamically regulate model entropy. The overall workflow remains consistent with standard GRPO, with modifications only in the gradient update stage where different weights are applied based on the sign of the token advantage.

### Key Designs

**1. Theoretically Proving Positive-Advantage Tokens as the Main Cause of Entropy Collapse**

Prior methods to mitigate entropy collapse often relied on blind regularization or clipping because the specific drivers were unclear. This work begins with theory, deriving the gradient of the GRPO objective with respect to logits (Eq. 3/4) and analyzing four scenarios based on whether a token is sampled and its advantage sign. The conclusion is clear: for unsampled tokens, positive advantages decrease their probability; for sampled tokens, positive advantages increase their probability. Since high-probability tokens are more likely to be sampled, positive-advantage updates continuously amplify high-probability tokens and suppress low-probability ones, leading to collapse. Negative-advantage tokens act conversely, helping redistribute probability. Identifying "positive-advantage tokens" as the root cause enables precise regulation.

**2. Positive-Advantage Reweighting: Using $\lambda$ as a Knob for Entropy**

Since positive-advantage tokens drive collapse, their loss is multiplied by a weight $\lambda \in [0,1]$. A smaller $\lambda$ weakens the "tightening" effect, maintaining higher entropy. The authors propose three schedules: Stage-based sets $\lambda=0$ in the first half of training (exploring with non-positive advantage tokens) and linearly increases it to 1; Epoch-wise linearly increases $\lambda$ from 0 to 1 within each epoch, i.e., $\lambda=(e-1)/(E-1)$; Entropy-guided adaptively adjusts $\lambda$ based on current entropy—decreasing $\lambda$ to encourage exploration when entropy is below a threshold $\delta$, and increasing it to promote exploitation when above $\delta$, with a step size $\Delta=0.05$. Unlike implicit methods like Clip-Higher, explicit $\lambda$ control allows quantitative targeting of entropy levels.

**3. Identifying Three External Factors Affecting Entropy Dynamics**

Beyond the root cause, the authors identify three practical "knobs" that influence entropy: First, clipping thresholds—Clip-Higher inhibits collapse while Clip-Lower exacerbates it. Second, the number of off-policy updates $N_{\text{update}}$—more updates amplify entropy trends (but may lead to overfitting). Third, training data diversity—lower diversity leads to lower entropy. Surprisingly, performance using only ~600 selected samples can approach that of ~17k samples. This provides a checklist for practitioners to troubleshoot entropy behavior.

### Loss & Training

- The base objective is the clipped surrogate objective of GRPO.
- Positive-Advantage Reweighting multiplies the loss of positive-advantage tokens by $\lambda \in [0, 1]$.
- Update rule for the Entropy-guided variant: $\lambda_{k+1} = \text{clip}(\lambda_k \pm \Delta, 0, 1)$, determined by comparing current entropy to threshold $\delta$.
- Training utilizes the veRL framework with a Qwen2.5-Math-7B base model and DAPO-Math-17K training data.

## Key Experimental Results

### Main Results

| Model | AIME 2024 (Avg@64/Pass@64) | AIME 2025 | MATH500 | AMC 2023 | Minerva | LiveCodeBench | IF-Eval | Mean (ID) | Entropy |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-Math-7B | 10.00/60.00 | 3.80/33.33 | 43.76/95.60 | 30.04/92.50 | 14.41/60.29 | 3.62/30.15 | 22.67/80.46 | 20.40/68.35 | N/A |
| + GRPO (N=1) | 28.75/63.33 | 14.69/50.00 | 78.14/96.80 | 64.38/97.50 | 34.64/64.34 | 7.85/33.46 | 30.17/72.90 | 44.12/74.39 | 0.118 |
| + Pos-Adv-Reweight (Entropy-guided) | **34.38/73.33** | 15.89/40.00 | 75.93/95.40 | 69.34/92.50 | 32.78/64.71 | 6.89/33.82 | 31.88/66.07 | **45.66/73.19** | 0.187 |
| + Ada-Ent-Reg (δ=0.3657) | 33.96/66.67 | 18.65/50.00 | 73.98/92.80 | 68.52/97.50 | 31.66/61.76 | 6.31/32.35 | 29.66/69.78 | 45.35/73.75 | 0.309 |
| + Clip-Higher | 33.33/60.00 | 15.94/53.33 | 72.35/94.20 | 67.62/97.50 | 30.57/63.97 | 5.88/32.35 | 31.35/66.19 | 43.96/73.80 | 0.539 |

Pos-Adv-Reweight (Entropy-guided) outperforms Clip-Higher on 6 out of 7 benchmarks and achieves the highest ID Avg@64 score (45.66) among all entropy regularization methods.

### Ablation Study

| Setting | Mean (ID) Avg@64 | Entropy | Note |
|---|---|---|---|
| Adv ≥ 0 only | 42.30 | 0.015 | Most severe entropy collapse |
| Adv ≤ 0 only | 42.70 | 0.884 | High entropy but poor OOD performance |
| Rand-Pos-Clip | 44.88 | 0.058 | Randomly clipping positive gradients is also effective |
| Stage-based | 44.85 | 0.330 | Step-wise increase of λ |
| Epoch-wise | 45.05 | 0.052 | Linear increase of λ per epoch |

### Key Findings

- **Non-monotonic Entropy-Performance Relationship**: Higher entropy is not always better; correlation varies significantly across tasks (LiveCodeBench shows a strong negative correlation of -0.89, while others are weak).
- **~600 Samples Comparable to ~17k Samples**: 616 samples selected via K-means clustering achieve performance levels similar to training on the full dataset.
- **Entropy Collapse Degrades Calibration**: More severe entropy collapse is associated with stronger overconfidence and calibration bias.
- **Off-policy Updates Amplify Entropy Shifts**: Increasing $N_{\text{update}}$ accelerates entropy changes but may lead to overfitting (lower Pass@64).

## Highlights & Insights

- **Root Cause Analysis Over Patching**: This work does not just propose another regularization term; it demonstrates that positive-advantage tokens are the fundamental cause of entropy collapse, an insight of universal value.
- **Simplicity is Effective**: Rand-Pos-Clip (randomly zeroing a small fraction of positive-advantage gradients) performs comparably to complex methods like Clip-Cov, suggesting that understanding the mechanism is more important than method complexity.
- **Significant Data Efficiency**: The discovery that 600 samples can match 17k samples has major implications for the practical deployment of RLVR.
- **Entropy-guided Variant is Practical**: Among the three variants, the adaptive regulation version is most versatile as it does not require pre-defined training stages or epoch counts.

## Limitations & Future Work

- Experiments were limited to the mathematics domain and did not cover code generation or agent scenarios, though the authors note that AEPO in QwenLong-L1.5 uses similar logic for long-context reasoning.
- Tested only on 7B models; validation on larger scales is needed.
- The Entropy-guided variant introduces hyperparameters $\delta$ and $\Delta$; automated determination of optimal values remains an open question.
- Theoretical gradient derivations are based on approximations, which may have deviations in actual training.

## Related Work & Insights

- **DAPO (Yu et al., 2025)**: Implicitly mitigates collapse via Clip-Higher but lacks precise entropy control.
- **Clip-Cov / KL-Cov (Cui et al., 2025)**: Analyzes entropy through the covariance of log-probability and advantages.
- **Adaptive Entropy Regularization (He et al., 2025)**: Dynamically adjusts regularization coefficients but is difficult to tune.
- **Entropy-Adv (Cheng et al., 2025)**: Incorporates entropy terms into the advantage function to encourage exploration.
- The positive-advantage reweighting approach is orthogonal to these methods and could be combined in future work.

## Rating

- Novelty: ⭐⭐⭐⭐ Reveals the root cause of entropy collapse from a gradient perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, covering multiple benchmarks, clipping variants, off-policy updates, data diversity, and calibration.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with a logical progression through three core questions.
- Value: ⭐⭐⭐⭐ Simple and practical method with significant insights for the RLVR community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Conditional Advantage Estimation for Reinforcement Learning in Large Reasoning Models](../../ICLR2026/llm_reasoning/conditional_advantage_estimation_for_reinforcement_learning_in_large_reasoning_m.md)
- [\[NeurIPS 2025\] The Impact of Quantization on Large Reasoning Model Reinforcement Learning](../../NeurIPS2025/llm_reasoning/the_impact_of_quantization_on_large_reasoning_model_reinforcement_learning.md)
- [\[ICLR 2026\] A Simple "Motivation" Can Enhance Reinforcement Finetuning of Large Reasoning Models](../../ICLR2026/llm_reasoning/a_simple_motivation_can_enhance_reinforcement_finetuning_of_large_reasoning_mode.md)
- [\[ACL 2026\] Large Reasoning Models Are (Not Yet) Multilingual Latent Reasoners](large_reasoning_models_are_not_yet_multilingual_latent_reasoners.md)
- [\[ACL 2026\] TrigReason: Trigger-Based Collaboration between Small and Large Reasoning Models](trigreason_trigger-based_collaboration_between_small_and_large_reasoning_models.md)

</div>

<!-- RELATED:END -->
