---
title: >-
  [Paper Note] Revisiting Entropy in Reinforcement Learning for Large Reasoning Models
description: >-
  [ACL 2026][LLM Reasoning][Entropy collapse] This paper systematically investigates the entropy dynamics of LLMs during RLVR training…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Entropy collapse"
  - "RLVR"
  - "GRPO"
  - "Positive-advantage reweighting"
  - "reasoning models"
date: 2026-05-08
content_hash: 577274eb84ac3c61
---

# Revisiting Entropy in Reinforcement Learning for Large Reasoning Models

**Conference**: ACL 2026  
**arXiv**: [2511.05993](https://arxiv.org/abs/2511.05993)  
**Code**: [GitHub](https://github.com/cordercorder/EntropyRL)  
**Area**: LLM Reasoning  
**Keywords**: Entropy collapse, RLVR, GRPO, Positive-advantage reweighting, reasoning models

## TL;DR

This paper systematically investigates the entropy dynamics of LLMs during RLVR training, revealing that positive-advantage tokens are the primary drivers of entropy collapse. It proposes the Positive-Advantage Reweighting method to effectively regulate model entropy by dynamically adjusting the loss weights of positive-advantage tokens.

## Background & Motivation

- **Background**: Represented by OpenAI o1, DeepSeek-R1, and Kimi k1.5, RLVR (Reinforcement Learning with Verifiable Rewards) has become the mainstream paradigm for enhancing LLM reasoning capabilities, showing outstanding performance in tasks such as mathematics and coding.
- **Limitations of Prior Work**: During RLVR training, the entropy of LLMs typically drops sharply (known as "entropy collapse"), causing the model to converge prematurely to sub-optimal local optima. This leads to probability mass concentrating on a few tokens and a consequent loss of exploration capability.
- **Key Challenge**: Although various methods (such as DAPO's Clip-Higher, adaptive entropy regularization, and Clip-Cov) have attempted to mitigate entropy collapse, a systematic study of entropy dynamics in RLVR is lacking. Three key questions remain insufficiently explored: (1) How does entropy correlate with performance? (2) What factors determine entropy dynamics? (3) How can entropy be effectively regulated to improve performance?
- **Goal**: To comprehensively analyze entropy dynamics in RLVR training through extensive experiments, identify the root cause of entropy collapse, and propose simple yet effective regulation methods.
- **Key Insight**: Starting from theoretical gradient analysis, this work distinguishes the different impacts of positive-advantage and negative-advantage tokens on entropy, rather than merely patching the issue from a regularization perspective.
- **Core Idea**: Positive-advantage tokens are identified as the primary cause of entropy collapse—they increase the probability of sampled tokens while suppressing unsampled tokens, leading to excessive probability concentration. Precisely controlling entropy can be achieved by adjusting their loss weights.

## Method

### Overall Architecture

Based on the standard RLVR training pipeline using GRPO, a Positive-Advantage Reweighting mechanism is introduced into the optimization objective. Model entropy is dynamically regulated via a hyperparameter $\lambda$ that controls the loss weight of positive-advantage tokens. The overall workflow remains consistent with standard GRPO, with the only modification being the application of different weights during the gradient update phase based on the sign of the token advantage.

### Key Designs

1.  **Theoretical Analysis of Entropy Collapse Driven by Positive-Advantage Tokens**:
    - **Function**: To prove from a gradient perspective that positive-advantage tokens are the primary cause of entropy collapse.
    - **Mechanism**: The gradient of the GRPO objective function with respect to the logits is derived (Eq.3/4). When a token is not sampled, a positive advantage causes its probability to decrease; when a token is sampled, a positive advantage causes its probability to increase. Since high-probability tokens are more likely to be sampled, positive-advantage updates further amplify high-probability tokens and suppress low-probability ones, leading to probability concentration. Negative advantage does the opposite, helping to mitigate entropy collapse.
    - **Design Motivation**: Understanding the root cause of entropy collapse is essential for designing precise regulation strategies instead of blindly adding regularization terms.

2.  **Positive-Advantage Reweighting (Three Variants)**:
    - **Function**: To control entropy by adjusting the loss weight $\lambda$ of positive-advantage tokens.
    - **Mechanism**:
        - **Stage-based**: $\lambda = 0$ in the first half of training (using only non-positive-advantage tokens), linearly increasing to 1 in the second half.
        - **Epoch-wise**: $\lambda$ increases linearly from 0 to 1 within each epoch, i.e., $\lambda = (e-1)/(E-1)$.
        - **Entropy-guided**: Adaptive adjustment based on current entropy—decrease $\lambda$ when entropy is below a threshold $\delta$ (encouraging exploration) and increase $\lambda$ when it is above $\delta$ (promoting exploitation), using a step size $\Delta = 0.05$.
    - **Design Motivation**: Compared to implicit methods like Clip-Higher, explicit control of $\lambda$ allows for the precise regulation of entropy toward a predetermined target.

3.  **Identification of Three Factors Influencing Entropy Dynamics**:
    - **Function**: To provide practical guidance for the community.
    - **Mechanism**: Through controlled variable experiments, the study reveals: (1) Clipping threshold: Clip-Higher prevents entropy collapse, while Clip-Lower exacerbates it; (2) Off-policy update frequency: more updates amplify entropy change trends; (3) Training data diversity: lower diversity leads to lower entropy, though approximately 600 samples can achieve performance comparable to approximately 17k samples.
    - **Design Motivation**: Understanding these factors helps in rationally setting hyperparameters in practice.

### Loss & Training

- The base objective function is the clipped surrogate objective of GRPO.
- Positive-Advantage Reweighting multiplies the loss of positive-advantage tokens by a weight $\lambda \in [0, 1]$.
- The update rule for the Entropy-guided variant: $\lambda_{k+1} = \text{clip}(\lambda_k \pm \Delta, 0, 1)$, where the direction is determined by comparing the current entropy with the threshold $\delta$.
- Training is conducted using the veRL framework, with Qwen2.5-Math-7B as the base model and DAPO-Math-17K as the training data.

## Key Experimental Results

### Main Results

| Model | AIME 2024 (Avg@64/Pass@64) | AIME 2025 | MATH500 | AMC 2023 | Minerva | LiveCodeBench | IF-Eval | Avg (ID) | Entropy |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-Math-7B | 10.00/60.00 | 3.80/33.33 | 43.76/95.60 | 30.04/92.50 | 14.41/60.29 | 3.62/30.15 | 22.67/80.46 | 20.40/68.35 | N/A |
| + GRPO (N=1) | 28.75/63.33 | 14.69/50.00 | 78.14/96.80 | 64.38/97.50 | 34.64/64.34 | 7.85/33.46 | 30.17/72.90 | 44.12/74.39 | 0.118 |
| + Pos-Adv-Reweight (Entropy-guided) | **34.38/73.33** | 15.89/40.00 | 75.93/95.40 | 69.34/92.50 | 32.78/64.71 | 6.89/33.82 | 31.88/66.07 | **45.66/73.19** | 0.187 |
| + Ada-Ent-Reg ($\delta=0.3657$) | 33.96/66.67 | 18.65/50.00 | 73.98/92.80 | 68.52/97.50 | 31.66/61.76 | 6.31/32.35 | 29.66/69.78 | 45.35/73.75 | 0.309 |
| + Clip-Higher | 33.33/60.00 | 15.94/53.33 | 72.35/94.20 | 67.62/97.50 | 30.57/63.97 | 5.88/32.35 | 31.35/66.19 | 43.96/73.80 | 0.539 |

Pos-Adv-Reweight (Entropy-guided) outperforms Clip-Higher on 6 out of 7 benchmarks and achieves the highest score (45.66) among all entropy regularization methods on the in-distribution Avg@64.

### Ablation Study

| Setting | Avg (ID) Avg@64 | Entropy | Description |
|---|---|---|---|
| Adv $\ge 0$ only | 42.30 | 0.015 | Most severe entropy collapse |
| Adv $\le 0$ only | 42.70 | 0.884 | High entropy but poor out-of-distribution performance |
| Rand-Pos-Clip | 44.88 | 0.058 | Randomly clipping positive-advantage gradients is also effective |
| Stage-based | 44.85 | 0.330 | Gradually increasing $\lambda$ in stages |
| Epoch-wise | 45.05 | 0.052 | Gradually increasing $\lambda$ by epoch |

### Key Findings

- **Non-monotonic Relationship Between Entropy and Performance**: Higher entropy is not always better; correlation varies significantly across tasks (LiveCodeBench shows a strong negative correlation of -0.89, while other benchmarks show weak correlations).
- **~600 Samples Comparable to ~17k Samples**: Using 616 training samples selected via K-means clustering achieves performance levels equivalent to training on the full dataset.
- **Entropy Collapse Leads to Calibration Degradation**: More severe entropy collapse is accompanied by stronger overconfidence and calibration bias.
- **Off-policy Updates Amplify Entropy Changes**: Increasing $N_{\text{update}}$ accelerates entropy change trends but may lead to overfitting (Pass@64 decreases).

## Highlights & Insights

- **Root Cause Analysis is More Effective Than Patching**: This work does more than propose another regularization method; it provides gradient-level proof that positive-advantage tokens are the root cause of entropy collapse, an insight that holds universal value.
- **Minimalist Methods are Effective**: Rand-Pos-Clip (randomly zeroing out a small portion of positive-advantage token gradients) performs comparably to complex methods like Clip-Cov, demonstrating that understanding the core mechanism is more important than methodological complexity.
- **Data Efficiency Findings are Profound**: The discovery that 600 samples can rival 17k samples is highly significant for the practical deployment of RLVR.
- **Entropy-guided Variant is Most Practical**: Among the three variants, the adaptive regulation version does not require pre-setting training stages or epoch counts, making it the most versatile.

## Limitations & Future Work

- Experiments were limited to the mathematics domain and did not cover code generation or agent scenarios. However, the authors note that AEPO in QwenLong-L1.5 adopted similar ideas for long-context reasoning, suggesting the method is generalizable.
- Experiments were only conducted on 7B models, lacking validation at larger model scales.
- The Entropy-guided variant introduces two hyperparameters, threshold $\delta$ and step size $\Delta$; automatic determination of optimal values remains to be explored.
- The gradient derivation in the theoretical analysis is based on approximations, and its precision in actual training may deviate.

## Related Work & Insights

- **DAPO (Yu et al., 2025)**: Implicitly mitigates entropy collapse via Clip-Higher but lacks precise entropy control.
- **Clip-Cov / KL-Cov (Cui et al., 2025)**: Restricts updates for tokens with high covariance between log-probability and advantage, analyzing entropy dynamics from a covariance perspective.
- **Adaptive Entropy Regularization (He et al., 2025)**: Dynamically adjusts regularization coefficients but is difficult to tune.
- **Entropy-Adv (Cheng et al., 2025)**: Incorporates entropy terms into the advantage function to encourage exploration.
- The positive-advantage reweighting idea proposed in this paper can be orthogonally combined with the aforementioned methods, and joint usage is worth exploring in the future.

## Rating

- Novelty: ⭐⭐⭐⭐ Revealed the root cause of entropy collapse driven by positive-advantage tokens at the gradient theory level; the insight is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, covering 7 benchmarks, various clipping variants, off-policy updates, data diversity, and calibration analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, progressively addressing three core questions with rich charts and tables.
- Value: ⭐⭐⭐⭐ Highly significant reference for the RLVR community; the method is simple and practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning](../../ICML2026/llm_reasoning/break_the_block_dynamic-size_reasoning_blocks_for_diffusion_large_language_model.md)
- [\[ACL 2026\] TemplateRL: Structured Template-Guided Reinforcement Learning for LLM Reasoning](templaterl_structured_template-guided_reinforcement_learning_for_llm_reasoning.md)
- [\[NeurIPS 2025\] Reasoning Models Hallucinate More: Factuality-Aware Reinforcement Learning for Large Reasoning Models](../../NeurIPS2025/llm_reasoning/reasoning_models_hallucinate_more_factuality-aware_reinforcement_learning_for_la.md)
- [\[NeurIPS 2025\] The Impact of Quantization on Large Reasoning Model Reinforcement Learning](../../NeurIPS2025/llm_reasoning/the_impact_of_quantization_on_large_reasoning_model_reinforcement_learning.md)
- [\[ACL 2026\] ETR: Entropy Trend Reward for Efficient Chain-of-Thought Reasoning](etr_entropy_trend_reward_for_efficient_chain-of-thought_reasoning.md)

</div>

<!-- RELATED:END -->
