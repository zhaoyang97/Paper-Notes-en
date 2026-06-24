---
title: >-
  [Paper Note] Optimizing Language Models for Inference Time Objectives using Reinforcement Learning
description: >-
  [ICML2025][Reinforcement Learning][inference-time compute] This paper proposes explicitly optimizing inference-time k-sample objectives (pass@k / majority voting) during the RL training phase. By constructing unbiased, low-variance gradient estimators using a leave-one-out control variate, the approach significantly improves inference-time performance on MATH and CodeContests.
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "inference-time compute"
  - "pass@k"
  - "majority voting"
  - "policy gradient"
  - "leave-one-out"
  - "REINFORCE"
  - "inference-time objective optimization"
date: 2026-05-08
content_hash: 63599ede02ab1475
---

# Optimizing Language Models for Inference Time Objectives using Reinforcement Learning

**Conference**: ICML2025  
**arXiv**: [2503.19595](https://arxiv.org/abs/2503.19595)  
**Authors**: Yunhao Tang, Kunhao Zheng, Gabriel Synnaeve, Rémi Munos  
**Institutions**: Meta (FAIR)  
**Code**: Not open-sourced  
**Area**: Reinforcement Learning  
**Keywords**: inference-time compute, pass@k, majority voting, policy gradient, leave-one-out, REINFORCE, inference-time objective optimization

## TL;DR

This paper proposes explicitly optimizing inference-time k-sample objectives (pass@k / majority voting) during the RL training phase. By constructing unbiased, low-variance gradient estimators using a leave-one-out control variate, the approach significantly improves inference-time performance on MATH and CodeContests.

## Background & Motivation

### Core Problem

Traditional language model training optimizes the **expected single-sample reward** $\max_\theta \mathbb{E}_{y\sim\pi_\theta}[r(x,y)]$, whereas deployment often utilizes **multi-sample inference strategies** (e.g., pass@k, majority voting). There exists a **mismatch** between the training objective and the inference objective:

- The model does not know what inference algorithm will be used during training $\rightarrow$ failing to fully exploit inference-time compute.
- Performance gains from multi-sample generation during inference rely on **diversity**, whereas single-sample training tends to suffer from mode collapse.

### Motivation

If it is **known beforehand that pass@k or majority voting will be used at inference time**, can these objectives be explicitly optimized during the training phase? The potential benefits of doing so include:

1. The trained policy is better aligned with the inference algorithm, yielding better k-sample performance.
2. With a fixed inference-compute budget, the utility per unit of inference computation is higher.
3. Providing a theoretical framework for training-inference objective alignment.

### Related Background

- **Inference-time scaling**: Inference-time compute scaling has recently been widely demonstrated to be effective (e.g., AlphaGo, DeepSeek-R1, OpenAI o1).
- **pass@k**: Allows $k$ attempts, succeeding if at least one is correct, which is suitable for scenarios with verifiers (e.g., code generation).
- **majority voting (Self-Consistency)**: Generates $k$ responses and takes the majority vote, widely used in mathematical reasoning.

## Method

### Overall Architecture: Unified Form of k-sample Objectives

This paper unifies inference-time objectives into a **k-sample objective function**:

$$\max_\theta \mathbb{E}_{(y_i)_{i=1}^k \sim \pi_\theta(\cdot|x)} \left[ f(x, y_1, \ldots, y_k) \right]$$

where the aggregation function $f$ can handle an arbitrary number of generated outputs. Special cases include:

| Objective | Aggregation Function $f$ | Explanation |
|------|-------------|------|
| Single-sample (Standard RL) | $f(\mathbf{y}) = r(y_1)$ | Degenerates to traditional RLHF |
| pass@k | $f(\mathbf{y}) = \max(r_1, \ldots, r_k)$ | Best out of $k$ samples |
| majority voting | $f(\mathbf{y}) = r(\text{maj}(a_1, \ldots, a_k))$ | Reward of the majority-vote answer |
| Average reward | $f(\mathbf{y}) = \frac{1}{k}\sum_{i=1}^k r_i$ | Mean of $k$ samples |

### Key Designs: Leave-One-Out Gradient Estimation

#### Problem with Naive REINFORCE Gradients

Applying REINFORCE directly to the k-sample objective yields:

$$f(\mathbf{y}) \sum_{i=1}^{k} \nabla_\theta \log \pi_\theta(y_i|x)$$

The variance of this gradient is $\mathcal{O}(k)$, which **grows linearly** with the number of samples $k$, contrasting sharply with the $\mathcal{O}(k^{-1})$ variance of standard k-sample average gradients. This is because the $k$ samples are **coupled** together through the aggregation function $f$.

#### Leave-One-Out Control Variate

To reduce variance, this paper proposes the **leave-one-out (LOO) control variate**:

$$\sum_{i=1}^{k} \left( f(\mathbf{y}) - f(\mathbf{y}_{-i}) \right) \nabla_\theta \log \pi_\theta(y_i|x)$$

where $\mathbf{y}_{-i}$ denotes the set of remaining samples after removing the $i$-th generation. The effective advantage function is defined as:

$$A_i \coloneqq f(\mathbf{y}) - f(\mathbf{y}_{-i})$$

It measures the **marginal contribution** of the $i$-th generation to the overall objective function.

**Proof of unbiasedness (Lemma 1)**: Since the $k$ samples are independent and identically distributed, $\mathbb{E}[f(\mathbf{y}_{-i}) \nabla_\theta \log \pi_\theta(y_i|x)] = 0$, meaning the LOO control variate does not introduce bias.

#### LOO Gradient Characteristics under Different Objectives

**Sparse Signal Property of pass@k**:

The advantage function degenerates to $A_i = \max(r_{1:k}) - \max(r_{-i})$, which is non-zero only for the **optimal generation** $y_{(k)}$:

$$A_{(k)} = r_{(k)} - r_{(k-1)}$$

Namely, the gradient signal originates solely from the gap between the best and second-best generations. This implies:
- When the problem is too easy (solve rate >> $1/k$) $\rightarrow$ multiple samples are correct, leading to a sparse signal.
- When the problem is too hard (solve rate << $1/k$) $\rightarrow$ practically no correct samples exist, and the signal is likewise sparse.
- **Optimal learning regime**: The gradient signal is most dense when the solve rate is around $\mathcal{O}(k^{-1})$.

**Vote-Flipping Signal of majority voting**:

The advantage function $A_i = r(\text{maj}(\mathbf{a})) - r(\text{maj}(\mathbf{a}_{-i}))$ measures whether the $i$-th answer **flips the majority voting result**. The gradient signal is non-zero only when removing $a_i$ changes the voting outcome.

### Loss & Training

This paper embeds the k-sample objective into an **online RL training framework**:

1. **Sampling**: For each prompt $x$, draw $k$ independent responses from the current policy $\pi_\theta$.
2. **Evaluation**: Compute $r(x, y_i)$ using a verifier/reward model, and calculate the aggregation function $f$ and LOO advantages.
3. **Update**: Perform policy gradient updates using the LOO gradient estimator.
4. **Regularization**: Optionally incorporate KL-divergence gratification to prevent policy degradation.

Key hyperparameters include:
- **Training $k$ value**: The number of samples used during training, which can differ from the inference-time $k$.
- **Temperature scheduling**: Controls sampling diversity.
- **Objective selection**: pass@k vs. majority voting vs. hybrid objectives.

## Key Experimental Results

### MATH Dataset Experiments

The impact of different training objectives on inference-time performance was evaluated on the mathematical reasoning task MATH:

| Training Objective | pass@1 | pass@8 | pass@64 | majority@8 | majority@64 |
|---------|--------|--------|---------|------------|-------------|
| Standard RL (Single-sample) | **Optimal** | Baseline | Baseline | Baseline | Baseline |
| pass@k Optimization | Slight decrease | **Significant Improvement** | **Significant Improvement** | Tied | Tied |
| majority voting Optimization | Slight decrease | Tied | Tied | **Significant Improvement** | **Significant Improvement** |

Key findings:
- Models trained specifically for pass@k significantly outperform standard RL on the pass@k metric.
- Models trained specifically for majority voting perform better on voting metrics.
- However, a **trade-off** exists: optimizing for inference-time objectives slightly sacrifices pass@1 performance.

### CodeContests Code Generation Experiments

Performance on the more challenging CodeContests competitive programming task:

| Method | pass@1 | pass@10 | pass@100 | Relative Gain (pass@100) |
|------|--------|---------|----------|-------------------|
| Standard RL Baseline | Baseline | Baseline | Baseline | — |
| pass@k Optimization (Ours) | Tied/Slight decrease | Significant Improvement | **Large Improvement** | Significant |

Key findings:
- On high-difficulty, low-solve-rate tasks like code generation, the advantages of pass@k optimization are even more pronounced.
- Compared to the baseline, the improvement at pass@100 is far greater than at pass@10, indicating that the method yields larger gains under a **large sampling budget**.
- The ratio of training $k$ to inference $k$ affects the final performance.

## Highlights & Insights

1. **A New Perspective on Training-Inference Alignment**: This study is the first to systematically incorporate inference-time k-sample objectives into RL training, providing a unified mathematical framework that covers both pass@k and majority voting as special cases.

2. **Elegant Design of the LOO Control Variate**: The physical meaning of the leave-one-out advantage function is clear—measuring the marginal contribution of a single generation to the overall objective, with an extremely concise proof of unbiasedness.

3. **Insights into pass@k Gradient Sparsity**: It reveals the relationship between the signal density of pass@k optimization and problem difficulty—showing the optimal learning regime is at a solve rate around $\mathcal{O}(1/k)$, which provides theoretical guidance for curriculum learning.

4. **Practical Trade-off Analysis**: It honestly demonstrates the trade-off between optimizing inference-time objectives and pass@1 performance, helping practitioners choose appropriate training objectives based on deployment scenarios.

5. **High Suitability for Code Generation**: In scenarios like CodeContests where verifiers are naturally available and solve rates are extremely low, the practical value of pass@k optimization is particularly prominent.

## Limitations & Future Work

1. **Limited to Simple Inference Strategies**: Only pass@k and majority voting are considered, without exploring more complex inference-time strategies such as best-of-k (requiring an auxiliary reward model), MCTS, or beam search.

2. **Training Computational Overhead**: Each prompt requires sampling $k$ complete responses, making training costs roughly $k$ times that of standard RL, raising scalability concerns for large-scale training.

3. **Generalization of $k$ Values**: The policy is trained with a fixed $k$. Whether it remains effective for different inference-time $k$ values needs verification, and the optimal ratio between training $k$ and inference $k$ is not yet clear.

4. **Limited Task Coverage**: Only validated on mathematical reasoning (MATH) and code generation (CodeContests); its applicability to broader reasoning tasks (e.g., common sense reasoning, multi-step planning) remains unverified.

5. **Relationship with GRPO/RLOO**: The leave-one-out baseline is closely related to methods like GRPO, but the comparison and distinction in the paper are not sufficiently clear.

6. **Incomplete Experimental Data**: The local cache cuts off after Section 3.1, meaning complete ablation studies and quantitative data could not be acquired.

## Related Work & Insights

### Inference-Time Compute Scaling
- **Self-Consistency (Wang et al., 2022)**: The original proposal of majority voting, whose objective this paper incorporates into training optimization.
- **OpenAI o1 / DeepSeek-R1**: Successful examples of inference-time compute scaling, though their training objectives remain single-sample.
- **AlphaCode (Li et al., 2022)**: Used large-scale sampling + filtering in competitive programming; this paper provides training-side optimization for similar scenarios.

### Strategy Gradient Variance Reduction
- **RLOO (Kool et al., 2019; Ahmadian et al., 2024)**: Application of the leave-one-out baseline under average reward objectives, which this paper generalizes to general k-sample objectives.
- **GRPO (Shao et al., 2024)**: Group Relative Policy Optimization, which also leverages group samples as a baseline, but optimizes a different objective.

### Inspired Future Directions
- The framework could be extended to more complex tree search inference strategies.
- The "marginal contribution" perspective of the LOO advantage function could inspire new credit assignment methods.
- Adaptively selecting the $k$ value during training based on curriculum difficulty might yield further improvements.

## Rating

- Novelty: ⭐⭐⭐⭐ — The perspective of explicitly incorporating inference-time objectives into RL training is novel, and the derivation of the LOO gradient estimator is clean.
- Experimental Thoroughness: ⭐⭐⭐ — Covers two representative scenarios, MATH and CodeContests, though the incomplete cache limits a comprehensive evaluation.
- Writing Quality: ⭐⭐⭐⭐ — Clear mathematical derivations, good intuitive analysis of special cases, progressing step-by-step from the general framework to specific applications.
- Value: ⭐⭐⭐⭐ — Given the increasing importance of inference-time compute, aligning training and inference objectives holds high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] T1: Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling](t1_advancing_language_model_reasoning_through_reinforcement_learning_and_inferen.md)
- [\[ICLR 2026\] Representation-Based Exploration for Language Models: From Test-Time to Post-Training](../../ICLR2026/reinforcement_learning/representation-based_exploration_for_language_models_from_test-time_to_post-trai.md)
- [\[ICLR 2026\] Polychromic Objectives for Reinforcement Learning](../../ICLR2026/reinforcement_learning/polychromic_objectives_for_reinforcement_learning.md)
- [\[ICML 2025\] Enhancing Decision-Making of Large Language Models via Actor-Critic](enhancing_decision-making_of_large_language_models_via_actor-critic.md)
- [\[ICML 2025\] ReVISE: Learning to Refine at Test-Time via Intrinsic Self-Verification](revise_learning_to_refine_at_test-time_via_intrinsic_self-verification.md)

</div>

<!-- RELATED:END -->
