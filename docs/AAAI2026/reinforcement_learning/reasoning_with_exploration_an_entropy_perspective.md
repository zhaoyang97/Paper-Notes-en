---
title: >-
  [Paper Note] Reasoning with Exploration: An Entropy Perspective
description: >-
  [AAAI 2026][Reinforcement Learning][Entropy] This paper analyzes the positive correlation between exploratory reasoning behaviors in LLMs (pivotal tokens, self-reflection, rare behaviors) and high-entropy regions from an entropy perspective. It proposes a minimalist entropy-based advantage shaping method—requiring only a single line of code modification—that significantly enhances the Pass@K reasoning capability ceiling of LLMs.
tags:
  - AAAI 2026
  - Reinforcement Learning
  - Entropy
  - Exploratory Reasoning
  - Advantage Shaping
  - RLVR
  - LLM Reasoning
date: 2026-05-08
content_hash: fb9f132e1cc9efb7
---

# Reasoning with Exploration: An Entropy Perspective

**Conference**: AAAI 2026
**arXiv**: [2506.14758](https://arxiv.org/abs/2506.14758)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: Entropy, Exploratory Reasoning, Advantage Shaping, RLVR, LLM Reasoning

## TL;DR

This paper analyzes the positive correlation between exploratory reasoning behaviors in LLMs (pivotal tokens, self-reflection, rare behaviors) and high-entropy regions from an entropy perspective. It proposes a minimalist entropy-based advantage shaping method—requiring only a single line of code modification—that significantly enhances the Pass@K reasoning capability ceiling of LLMs.

## Background & Motivation

Current RLVR methods (e.g., GRPO, PPO) primarily rely on accuracy-driven reward signals to train LLMs. Although these methods effectively improve reasoning capabilities in early training stages, models tend to converge toward narrow, over-optimized behavioral patterns as training progresses, gradually losing the motivation to explore alternative strategies. This leads to:

**Performance plateaus**: Reasoning capability stagnates or even degrades in the mid-to-late stages of training.

**Limited multi-step reasoning**: The lack of exploration causes poor performance in complex or under-defined scenarios.

**Contradiction between Pass@K and Pass@1**: Models trained with RL outperform base models on Pass@1, yet when K is sufficiently large, base models achieve higher Pass@K—indicating that RL restricts the model's exploratory capacity.

**Core Observation**: In classical RL, entropy is the central signal for measuring exploration. The authors find that "exploratory behaviors" in LLM reasoning are also highly correlated with high-entropy regions:
- **Pivotal tokens** (e.g., "first," "because," "however") typically exhibit higher entropy.
- **Self-reflective behaviors** (e.g., "Let's verify...") tend to appear under high-entropy conditions.
- **Rare behaviors** (novel reasoning patterns emerging after RL training) are likewise associated with high entropy.

These findings motivate a simple yet profound idea: encouraging exploratory reasoning through entropy.

## Method

### Overall Architecture

An entropy-based auxiliary term is added to the advantage function of standard RL algorithms (PPO or GRPO). This modification is extremely concise—it requires inserting only a single line of code into an existing RLVR training pipeline.

### Key Designs

#### 1. Empirical Analysis of Entropy and Exploratory Reasoning

**Pivotal Tokens**:

By visualizing the token-level entropy distribution of Qwen2.5-Base-7B on mathematical reasoning tasks:
- Causal connectives (because, therefore): entropy significantly higher than ordinary tokens.
- Contrastive markers (however, although): likewise high entropy.
- Sequential words (first, then): high entropy.
- Reasoning verbs (suggest, demonstrate): high entropy.

These tokens act as logical connectors, marking the "decision points" in the model's reasoning flow.

**Reflective Actions**:

Each response is split into sentences; the average entropy of each sentence is computed, and reflective sentences containing keywords such as "verify" and "check" are identified via regular expressions. Results show that reflective sentences exhibit significantly higher average entropy than other sentences. This is the first work to establish a link between entropy and LLM self-reflection.

**Rare Behaviors Emerging During RL Training**:

SBERT is used to embed all response sentences; for each RL-generated sentence, the average distance to its $k=5$ nearest neighbors in the base model's outputs is computed. Sentences in the top 10% by distance are labeled as "rare behaviors" (e.g., transforming a logarithmic system of equations into a linear one—a behavior rarely produced by the base model). These rare behaviors also exhibit higher entropy.

#### 2. Entropy-Based Advantage Shaping

For each token $o_t$, the entropy of the current policy over vocabulary $\mathcal{V}$ is computed:

$$\mathcal{H}_t = -\sum_{v \in \mathcal{V}} \pi_\theta(v|q, o_{<t}) \log \pi_\theta(v|q, o_{<t})$$

The entropy-based advantage term $\psi(\mathcal{H}_t)$ is defined as:

$$\psi(\mathcal{H}_t) = \min\left(\alpha \cdot \mathcal{H}_t^{\text{detach}}, \frac{|A_t|}{\kappa}\right)$$

The shaped advantage becomes:

$$A_t^{\text{shaped}} = A_t + \psi(\mathcal{H}_t)$$

**Three Key Design Choices**:

1. **Clipping ($\kappa > 1$)**: Ensures that the entropy term $\psi(\mathcal{H}_t) \leq \frac{|A_t|}{\kappa}$ does not dominate the advantage. When $A_t < 0$, this constraint guarantees that adding the entropy term does not flip the sign of the advantage—thereby preserving the original optimization direction.
2. **Gradient Detachment**: The entropy term $\mathcal{H}_t^{\text{detach}}$ is detached from the computation graph and does not participate in backpropagation. The policy gradient therefore retains the same form as standard PPO:

$$\nabla_\theta \mathcal{J}^{\text{shaped}} = \mathbb{E}\left[\sum_t (A_t + \psi(\mathcal{H}_t)) \nabla_\theta \log \pi_\theta(o_t|q, o_{<t})\right]$$

3. **Self-Regulation**: Due to the intrinsic tension between entropy and confidence, as the model gains confidence on specific tokens (entropy decreases), the entropy-based advantage naturally diminishes—avoiding over-encouragement.

#### 3. Fundamental Distinction from Entropy Regularization

| Dimension | Entropy Regularization | Entropy-Based Advantage Shaping (Ours) |
|-----------|----------------------|---------------------------------------|
| Training objective | $\mathcal{J} = \mathcal{J}_{\text{PPO}} + \beta\sum_t \mathcal{H}_t$ | $\mathcal{J} = \mathcal{J}_{\text{PPO}}(A_t^{\text{shaped}})$ |
| Policy gradient | $\sum_t A_t \nabla_\theta \log \pi_\theta + \beta\sum_t \nabla_\theta \mathcal{H}_t$ | $\sum_t A_t^{\text{shaped}} \nabla_\theta \log \pi_\theta$ |
| Entropy gradient flow | $\nabla_\theta \mathcal{H}_t \neq 0$ (explicitly encourages high entropy) | $\nabla_\theta \mathcal{H}_t^{\text{detach}} = 0$ (does not alter gradient flow) |

Entropy regularization directly optimizes toward higher entropy (potentially unstable), whereas the proposed method indirectly encourages exploration in uncertain regions solely by adjusting the magnitude of advantage values, preserving the optimization dynamics of the original RL algorithm. The two approaches are orthogonal.

### Loss & Training

- **RL algorithms**: GRPO and PPO
- **Training data**: DAPO
- **Reward**: Outcome reward (correct: +1, incorrect: −1)
- **Key hyperparameters**: $\kappa = 2$ (fixed across all experiments), $\alpha = 0.4$ (GRPO) / $0.1$ (PPO)
- **Baseline enhancement techniques**: Clip-Higher, Token-level Loss, Critic-Pretraining, Group-Sampling
- **Evaluation**: Temperature 0.6, maximum length 8K tokens, top-p = 0.95
- **Framework**: veRL

## Key Experimental Results

### Main Results

Pass@K and Pass@1 performance on Qwen2.5-Base-7B:

| Method | AIME25 Pass@256 | AIME25 Pass@1 | AIME24 Pass@256 | AIME24 Pass@1 | AMC23 Pass@128 | AMC23 Pass@1 | MATH500 Pass@16 | MATH500 Pass@1 |
|--------|----------------|---------------|----------------|---------------|---------------|-------------|----------------|---------------|
| Base | 50.0 | 2.2 | 66.7 | 5.2 | 90.4 | 28.3 | 88.8 | 54.4 |
| +GRPO | 50.0 | 10.7 | 46.7 | 11.9 | 91.6 | 55.6 | 65.4 | 55.3 |
| +GRPO w/ Entropy Adv. | **53.3** | **11.8** | **56.7** | **12.6** | 91.6 | **57.8** | **74.0** | **58.5** |
| $\Delta$ | +3.3 | +1.1 | +10.0 | +0.7 | +0.0 | +2.2 | +8.6 | +3.2 |
| +PPO | 43.3 | 7.9 | 46.7 | 14.2 | 85.5 | 51.8 | 68.4 | 57.9 |
| +PPO w/ Entropy Adv. | **56.7** | **11.7** | **50.0** | **16.8** | **88.0** | **56.1** | **75.2** | **60.9** |
| $\Delta$ | +13.4 | +3.8 | +3.3 | +2.6 | +2.5 | +4.3 | +6.8 | +3.0 |

Qwen2.5-Math-Base-7B + GRPO:

| Method | AIME25 Pass@256 | AIME25 Pass@1 | AIME24 Pass@256 | AIME24 Pass@1 |
|--------|----------------|---------------|----------------|---------------|
| Base | 50.7 | 4.4 | 70.0 | 10.7 |
| +GRPO | 57.4 | 16.3 | 83.3 | 30.9 |
| +GRPO w/ Entropy Adv. | **63.6** | **17.6** | 80.0 | **33.7** |
| $\Delta$ | +6.2 | +1.3 | -3.3 | +2.8 |

### Ablation Study

**Entropy Regularization vs. Entropy-Based Advantage Shaping** (GRPO on Qwen2.5-Base):

| Method | AIME25 Pass@256 | AIME25 Pass@1 | AIME24 Pass@256 | AIME24 Pass@1 | MATH500 Pass@1 |
|--------|----------------|---------------|----------------|---------------|---------------|
| RL w/ Entropy Regularization | 50.0 | 9.3 | 50.0 | 16.0 | 57.4 |
| RL w/ Entropy Advantage | **53.3** | **11.8** | **56.7** | 12.6 | **58.5** |

**Training Dynamics Analysis**:

| Metric | RL Baseline (GRPO) | RL + Entropy Reg. | RL + Entropy Adv. | Note |
|--------|--------------------|-------------------|-------------------|------|
| Training reward | Steadily increasing | Steadily increasing | Higher in later stages | Stronger sustained improvement |
| Response length | Rises then falls | Similar to baseline | Continuously increasing | Encourages deeper reasoning |
| Overall entropy (step 2000) | 0.34 | Sudden spike, unstable | 0.17 | Avoids entropy collapse |
| Pivotal token entropy | Baseline level | — | Significantly lower | More confident at key positions |
| Reflective behavior frequency | Baseline level | — | Significantly increased | More self-verification |
| Repetition rate | Baseline level | — | Comparable to baseline | Longer responses without added redundancy |

### Key Findings

1. **Surpassing the reasoning ceiling of the base model**: On AIME2025 (the most challenging benchmark, released after the training data cutoff), the proposed method not only outperforms the RL baseline but also exceeds the Pass@K upper bound of the base model—demonstrating that the method genuinely expands the reasoning frontier rather than relying on retrieval.
2. **Standard RL restricts exploration**: Across multiple benchmarks, RL-trained models achieve lower Pass@K than the base model at large K; the proposed method effectively mitigates this issue.
3. **Self-regulation of entropy-based advantage**: As training progresses, the proportion of the entropy-based advantage naturally decreases from high to low without manual scheduling.
4. **Instability of entropy regularization**: Sudden entropy spikes appear after step 1500, whereas entropy-based advantage shaping remains consistently stable.
5. **Qualitative improvement in reasoning behaviors**: Case studies show that the model produces more systematic constraint enumeration, case analysis, and dynamic adjustment behaviors.

## Highlights & Insights

- **Extreme simplicity**: The method requires inserting only a single line of code into the `update_policy` function of the veRL framework, offering maximum reproducibility and practical value.
- **Deep insight**: This is the first work to systematically establish quantitative correlations between token-level entropy and exploratory reasoning behaviors (pivotal tokens, reflection, rare behaviors).
- **Elegant self-regulation**: The intrinsic tension between entropy and confidence achieves automatic exploration–exploitation balance without additional hyperparameter scheduling.
- **Orthogonality to entropy regularization**: This is not a "better entropy regularization" but rather a fundamentally different exploration mechanism operating from the perspective of the advantage function.
- **New perspective on Pass@K**: Treating Pass@K as an upper-bound estimator of reasoning capability provides a new dimension for evaluating the "exploratory capacity" of RL methods.

## Limitations & Future Work

1. **Validated only on the Qwen series**: Experiments on Llama were abandoned due to its lack of reasoning behaviors, limiting the generalizability of the conclusions.
2. **$\alpha$ is algorithm-sensitive**: GRPO and PPO require different values of $\alpha$ (0.4 vs. 0.1), and no adaptive selection mechanism is provided.
3. **Pass@256 on AIME24 decreases by 3.3 for GRPO + Entropy**: Indicating the method does not consistently improve performance across all benchmarks.
4. **Validated only on mathematical reasoning**: Other domains such as code reasoning and logical reasoning remain untested.
5. **Lack of comparison with other exploration methods**: Such as curiosity-driven exploration and intrinsic rewards.
6. **Causal direction uncertain**: The causal direction of high entropy → exploratory reasoning may also be reversed (exploratory reasoning naturally produces high entropy).

## Related Work & Insights

- Shares a philosophical connection with the maximum entropy idea in classical RL (Soft Actor-Critic, SAC), but is technically entirely distinct.
- Complementary to the intrinsic motivation approach of a concurrent work (Gao et al., 2025), which designs custom metrics, whereas this paper uses entropy.
- Orthogonal to entropy regularization methods (He et al.; Wang et al.) and can be used in combination.
- Implication: **Exploratory reasoning** may be the next critical direction for improving LLM reasoning capability, rather than solely pursuing single-attempt accuracy.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Analyzing exploratory reasoning in LLMs from an entropy perspective is a genuinely novel angle; the method is minimal yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Analysis is in-depth and benchmarks are diverse, but model coverage is insufficient.
- Writing Quality: ⭐⭐⭐⭐⭐ — Figures are polished, the analysis builds progressively, and the one-line-of-code highlight is compelling.
- Value: ⭐⭐⭐⭐⭐ — The method is extremely easy to apply with notable performance gains, offering high practical guidance value to the RLVR community.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Exploration vs Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward](../../ICLR2026/reinforcement_learning/exploration_vs_exploitation_rethinking_rlvr_through_clipping_entropy_and_spuriou.md)
- [\[ICLR 2026\] Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning](../../ICLR2026/reinforcement_learning/controllable_exploration_in_hybrid-policy_rlvr_for_multi-modal_reasoning.md)
- [\[AAAI 2026\] Reasoning or Memorization? Unreliable Results of Reinforcement Learning Due to Data Contamination](reasoning_or_memorization_unreliable_results_of_reinforcement_learning_due_to_da.md)
- [\[AAAI 2026\] Well Begun, Half Done: Reinforcement Learning with Prefix Optimization for LLM Reasoning](well_begun_half_done_reinforcement_learning_with_prefix_optimization_for_llm_rea.md)
- [\[AAAI 2026\] Language Model Distillation: A Temporal Difference Imitation Learning Perspective](language_model_distillation_a_temporal_difference_imitation_learning_perspective.md)

<!-- RELATED:END -->
