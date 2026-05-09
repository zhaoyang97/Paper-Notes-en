---
title: >-
  [Paper Note] Revisiting Entropy in Reinforcement Learning for Large Reasoning Models
description: >-
  [ACL 2026][LLM Reasoning][entropy collapse] This paper systematically investigates entropy dynamics in RLVR training of LLMs, identifies positive-advantage tokens as the primary driver of entropy collapse, and proposes Positive-Advantage Reweighting, which dynamically adjusts the loss weights of positive-advantage tokens to effectively regulate model entropy.
tags:
  - ACL 2026
  - LLM Reasoning
  - entropy collapse
  - RLVR
  - GRPO
  - positive-advantage reweighting
  - reasoning models
date: 2026-05-08
content_hash: 4e754a6d235f813c
---

# Revisiting Entropy in Reinforcement Learning for Large Reasoning Models

**Conference**: ACL 2026
**arXiv**: [2511.05993](https://arxiv.org/abs/2511.05993)
**Code**: [GitHub](https://github.com/cordercorder/EntropyRL)
**Area**: LLM Reasoning
**Keywords**: entropy collapse, RLVR, GRPO, positive-advantage reweighting, reasoning models

## TL;DR

This paper systematically investigates entropy dynamics in RLVR training of LLMs, identifies positive-advantage tokens as the primary driver of entropy collapse, and proposes Positive-Advantage Reweighting, which dynamically adjusts the loss weights of positive-advantage tokens to effectively regulate model entropy.

## Background & Motivation

- **State of the Field**: Exemplified by OpenAI o1, DeepSeek-R1, and Kimi k1.5, RLVR (Reinforcement Learning with Verifiable Rewards) has become the dominant paradigm for enhancing LLM reasoning, achieving strong results on tasks such as mathematics and code generation.
- **Limitations of Prior Work**: During RLVR training, the entropy of LLMs typically drops sharply—a phenomenon known as "entropy collapse"—causing the model to prematurely converge to suboptimal local optima, concentrate probability mass on a small number of tokens, and lose the ability to explore.
- **Root Cause**: Although several approaches (e.g., DAPO's Clip-Higher, adaptive entropy regularization, Clip-Cov) have been proposed to mitigate entropy collapse, a systematic study of entropy dynamics in RLVR is lacking. Three key questions remain underexplored: (1) How does entropy correlate with performance? (2) What factors govern entropy dynamics? (3) How can entropy be effectively regulated to improve performance?
- **Paper Goals**: Through extensive experiments, this paper comprehensively analyzes entropy dynamics in RLVR training, identifies the root cause of entropy collapse, and proposes a simple yet effective regulation method.
- **Starting Point**: The analysis departs from a theoretical gradient perspective, distinguishing the distinct effects of positive-advantage tokens and negative-advantage tokens on entropy, rather than merely patching the issue from a regularization standpoint.
- **Core Idea**: Positive-advantage tokens are the primary cause of entropy collapse—they increase the probability of sampled tokens while suppressing that of unsampled tokens, causing the probability distribution to over-concentrate. Adjusting their loss weights enables precise entropy control.

## Method

### Overall Architecture

Building upon the standard RLVR training pipeline with GRPO, the proposed method introduces a Positive-Advantage Reweighting mechanism into the optimization objective. A hyperparameter $\lambda$ controls the loss weight of positive-advantage tokens to dynamically regulate model entropy. The overall pipeline remains consistent with standard GRPO, with the sole modification being the application of different weights during gradient updates based on the sign of each token's advantage.

### Key Designs

1. **Theoretical Analysis of Positive-Advantage Tokens Driving Entropy Collapse**:

    - **Function**: Proves from a gradient perspective that positive-advantage tokens are the primary cause of entropy collapse.
    - **Mechanism**: Derives the gradient of the GRPO objective with respect to logits (Eq. 3/4). For unsampled tokens, a positive advantage decreases their probability; for sampled tokens, it increases it. Since high-probability tokens are more likely to be sampled, positive-advantage updates further amplify the probability of already high-probability tokens and suppress low-probability ones, causing distribution concentration. Negative-advantage tokens have the opposite effect and help alleviate entropy collapse.
    - **Design Motivation**: Understanding the root cause of entropy collapse is necessary to devise targeted regulation strategies, rather than blindly adding regularization terms.

2. **Positive-Advantage Reweighting (Three Variants)**:

    - **Function**: Controls entropy by adjusting the loss weight $\lambda$ for positive-advantage tokens.
    - **Mechanism**:
      - **Stage-based**: $\lambda=0$ in the first half of training (using only non-positive-advantage tokens), linearly increasing to 1 in the second half.
      - **Epoch-wise**: $\lambda$ increases linearly from 0 to 1 within each epoch, i.e., $\lambda=(e-1)/(E-1)$.
      - **Entropy-guided**: Adapts $\lambda$ based on current entropy—decreasing $\lambda$ when entropy falls below threshold $\delta$ (encouraging exploration) and increasing it when entropy exceeds $\delta$ (promoting exploitation), with step size $\Delta=0.05$.
    - **Design Motivation**: Compared to implicit methods such as Clip-Higher, explicitly controlling $\lambda$ enables precise regulation of entropy toward a target value.

3. **Identification of Three Key Factors Governing Entropy Dynamics**:

    - **Function**: Provides practical guidance for the community.
    - **Mechanism**: Controlled experiments reveal: (1) clipping threshold—Clip-Higher prevents entropy collapse whereas Clip-Lower exacerbates it; (2) number of off-policy updates—more updates amplify entropy change trends; (3) training data diversity—lower diversity leads to lower entropy, yet as few as ~600 samples can match the performance of ~17k samples.
    - **Design Motivation**: Understanding these factors enables practitioners to set hyperparameters more judiciously.

### Loss & Training

- The base objective is the GRPO clipped surrogate objective.
- Positive-Advantage Reweighting multiplies the loss of positive-advantage tokens by $\lambda \in [0,1]$.
- The update rule for the Entropy-guided variant is: $\lambda_{k+1} = \text{clip}(\lambda_k \pm \Delta,\, 0,\, 1)$, where the direction is determined by comparing the current entropy to threshold $\delta$.
- Training is conducted using the veRL framework, with Qwen2.5-Math-7B as the base model and DAPO-Math-17K as the training dataset.

## Key Experimental Results

### Main Results

| Model | AIME 2024 (Avg@64/Pass@64) | AIME 2025 | MATH500 | AMC 2023 | Minerva | LiveCodeBench | IF-Eval | Avg (ID) | Entropy |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-Math-7B | 10.00/60.00 | 3.80/33.33 | 43.76/95.60 | 30.04/92.50 | 14.41/60.29 | 3.62/30.15 | 22.67/80.46 | 20.40/68.35 | N/A |
| + GRPO (N=1) | 28.75/63.33 | 14.69/50.00 | 78.14/96.80 | 64.38/97.50 | 34.64/64.34 | 7.85/33.46 | 30.17/72.90 | 44.12/74.39 | 0.118 |
| + Pos-Adv-Reweight (Entropy-guided) | **34.38/73.33** | 15.89/40.00 | 75.93/95.40 | 69.34/92.50 | 32.78/64.71 | 6.89/33.82 | 31.88/66.07 | **45.66/73.19** | 0.187 |
| + Ada-Ent-Reg ($\delta$=0.3657) | 33.96/66.67 | 18.65/50.00 | 73.98/92.80 | 68.52/97.50 | 31.66/61.76 | 6.31/32.35 | 29.66/69.78 | 45.35/73.75 | 0.309 |
| + Clip-Higher | 33.33/60.00 | 15.94/53.33 | 72.35/94.20 | 67.62/97.50 | 30.57/63.97 | 5.88/32.35 | 31.35/66.19 | 43.96/73.80 | 0.539 |

Pos-Adv-Reweight (Entropy-guided) outperforms Clip-Higher on 6 out of 7 benchmarks and achieves the best in-domain Avg@64 score (45.66) among all entropy regularization methods.

### Ablation Study

| Setting | Avg (ID) Avg@64 | Entropy | Notes |
|---|---|---|---|
| Adv≥0 only | 42.30 | 0.015 | Most severe entropy collapse |
| Adv≤0 only | 42.70 | 0.884 | High entropy but poor OOD performance |
| Rand-Pos-Clip | 44.88 | 0.058 | Randomly zeroing positive-advantage gradients also effective |
| Stage-based | 44.85 | 0.330 | Gradually increasing $\lambda$ in stages |
| Epoch-wise | 45.05 | 0.052 | Gradually increasing $\lambda$ per epoch |

### Key Findings

- **Non-monotonic relationship between entropy and performance**: Higher entropy is not universally better; the correlation varies substantially across tasks (LiveCodeBench shows a strong negative correlation of −0.89 with entropy, while other benchmarks show weak correlations).
- **~600 samples can match ~17k samples**: A set of 616 training examples selected via K-means clustering achieves performance comparable to training on the full dataset.
- **Entropy collapse degrades calibration**: More severe entropy collapse is associated with stronger overconfidence and greater calibration error.
- **Off-policy updates amplify entropy changes**: Increasing $N_{\text{update}}$ accelerates entropy change trends but may lead to overfitting (Pass@64 decreases).

## Highlights & Insights

- **Root-cause analysis is more effective than patching**: Rather than proposing yet another regularization method, this work proves from a gradient perspective that positive-advantage tokens are the fundamental cause of entropy collapse—an insight of broad applicability.
- **Minimal methods can be highly effective**: Rand-Pos-Clip (randomly zeroing the gradients of a small fraction of positive-advantage tokens) achieves performance comparable to more complex methods such as Clip-Cov, demonstrating that understanding the core mechanism matters more than methodological complexity.
- **The data efficiency finding carries significant implications**: The observation that 600 samples can match 17k samples has major practical implications for deploying RLVR systems.
- **The Entropy-guided variant is the most practical**: Among the three variants, the adaptive version requires no pre-specification of training stages or epoch counts and is therefore the most generalizable.

## Limitations & Future Work

- Experiments are conducted exclusively in the mathematical domain, without covering code generation, agent scenarios, or other settings. The authors note that AEPO in QwenLong-L1.5 adopts a similar idea for long-context reasoning, suggesting the approach may generalize.
- Experiments are limited to a 7B model; validation at larger model scales is absent.
- The Entropy-guided variant introduces two hyperparameters—threshold $\delta$ and step size $\Delta$—and the automatic determination of their optimal values remains an open question.
- The gradient derivations in the theoretical analysis rely on approximations, which may introduce deviations under practical training conditions.

## Related Work & Insights

- **DAPO (Yu et al., 2025)**: Implicitly mitigates entropy collapse via Clip-Higher but does not enable precise entropy control.
- **Clip-Cov / KL-Cov (Cui et al., 2025)**: Restricts updates for tokens with high log-probability–advantage covariance, analyzing entropy dynamics from a covariance perspective.
- **Adaptive Entropy Regularization (He et al., 2025)**: Dynamically adjusts the regularization coefficient but is difficult to tune.
- **Entropy-Adv (Cheng et al., 2025)**: Incorporates an entropy term into the advantage function to encourage exploration.
- The positive-advantage reweighting approach proposed in this paper is orthogonal to the above methods and their joint use warrants future exploration.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Reveals from a gradient-theoretic perspective that positive-advantage tokens drive entropy collapse; the insight is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 7 benchmarks, multiple clipping variants, off-policy updates, data diversity, and calibration analysis; extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with three core questions addressed progressively; rich in figures and tables.
- **Value**: ⭐⭐⭐⭐ — Highly relevant to the RLVR community; the proposed method is simple and practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Reasoning Models Hallucinate More: Factuality-Aware Reinforcement Learning for Large Reasoning Models](../../NeurIPS2025/llm_reasoning/reasoning_models_hallucinate_more_factuality-aware_reinforcement_learning_for_la.md)
- [\[NeurIPS 2025\] The Impact of Quantization on Large Reasoning Model Reinforcement Learning](../../NeurIPS2025/llm_reasoning/the_impact_of_quantization_on_large_reasoning_model_reinforcement_learning.md)
- [\[ACL 2026\] Large Reasoning Models Are (Not Yet) Multilingual Latent Reasoners](large_reasoning_models_are_not_yet_multilingual_latent_reasoners.md)
- [\[ACL 2026\] TrigReason: Trigger-Based Collaboration between Small and Large Reasoning Models](trigreason_trigger-based_collaboration_between_small_and_large_reasoning_models.md)
- [\[AAAI 2026\] LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning](../../AAAI2026/llm_reasoning/llms_for_game_theory_entropy-guided_in-context_learning_and_adaptive_cot_reasoni.md)

</div>

<!-- RELATED:END -->
