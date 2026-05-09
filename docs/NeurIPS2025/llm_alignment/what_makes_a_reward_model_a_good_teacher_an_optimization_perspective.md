---
title: >-
  [Paper Note] What Makes a Reward Model a Good Teacher? An Optimization Perspective
description: >-
  [NeurIPS 2025][LLM Alignment][Reward Model] From an optimization-theoretic perspective, this paper proves that reward model accuracy alone is insufficient to measure its quality as an RLHF "teacher." Even a perfectly accurate reward model can lead to a flat RLHF objective landscape and extremely slow policy gradient optimization if the induced reward variance is too low. Moreover, different language models require different reward models.
tags:
  - NeurIPS 2025
  - LLM Alignment
  - Reward Model
  - RLHF
  - Reward Variance
  - Optimization Landscape
  - Policy Gradient
date: 2026-05-08
content_hash: dd755fd103fbfb17
---

# What Makes a Reward Model a Good Teacher? An Optimization Perspective

**Conference**: NeurIPS 2025
**arXiv**: [2503.15477](https://arxiv.org/abs/2503.15477)
**Code**: [github](https://github.com/princeton-pli/what-makes-good-rm)
**Area**: LLM Alignment / RLHF
**Keywords**: Reward Model, RLHF, Reward Variance, Optimization Landscape, Policy Gradient

## TL;DR
From an optimization-theoretic perspective, this paper proves that reward model accuracy alone is insufficient to measure its quality as an RLHF "teacher." Even a perfectly accurate reward model can lead to a flat RLHF objective landscape and extremely slow policy gradient optimization if the induced reward variance is too low. Moreover, different language models require different reward models.

## Background & Motivation

**Background**: RLHF is the standard pipeline for aligning LLMs. Its core involves training a proxy reward model $r_{\mathrm{RM}}$ to substitute the inaccessible ground-truth reward $r_{\mathrm{G}}$, followed by maximizing the proxy reward via policy gradient methods such as PPO, RLOO, or GRPO. The dominant metric for evaluating reward models is **accuracy**—the fraction of preference pairs correctly ranked on held-out data.

**Limitations of Prior Work**:
   - Empirically, more accurate reward models do not always yield better alignment, yet a theoretical explanation has been lacking.
   - Mainstream benchmarks (RewardBench, RM-Bench, etc.) evaluate purely based on accuracy and are decoupled from the language model being aligned.

**Key Challenge**: Accuracy only measures whether the ranking direction is correct (sign), while ignoring whether the signal is strong enough—i.e., whether the reward model sufficiently separates the rewards of different outputs under the policy distribution.

**Goal**: To formally characterize what makes a reward model a good RLHF teacher, and to reveal critical factors beyond accuracy.

**Key Insight**: The analysis proceeds from the RLHF optimization landscape—the gradient norm of policy gradient methods is directly related to the reward variance under the policy distribution.

**Core Idea**: Reward variance determines the flatness of the RLHF objective landscape and constitutes a key metric—independent of accuracy—that must be considered when evaluating reward models.

## Method

### Overall Architecture
This paper presents a combined theoretical and empirical study. The theoretical component establishes three core results: (1) low reward variance → flat landscape → slow optimization (Theorem 1); (2) a more accurate reward model is not necessarily a better teacher (Theorem 2); (3) different initial policies require different reward models (Theorem 3). Experiments are conducted on Pythia-2.8B and Llama-3.2.

### Key Designs

1. **Definition and Significance of Reward Variance**:

    - Definition: $\mathrm{Var}_{y \sim \pi_\theta(\cdot|x)}[r_{\mathrm{RM}}(x,y)]$ — the variance of reward values assigned by the reward model to different outputs under the current policy distribution.
    - Equivalent form: $\frac{1}{2} \mathbb{E}_{y,y' \sim \pi_\theta}[(r_{\mathrm{RM}}(x,y) - r_{\mathrm{RM}}(x,y'))^2]$, i.e., the average reward difference between output pairs sampled from the policy.
    - Core insight: Accuracy concerns only the correctness of ranking (sign), while reward variance concerns the degree of separation (magnitude). These two properties are independent.

2. **Theorem 1: Low Variance → Slow Optimization**:

    - For **any** reward function, the time required to achieve an expected reward gain of $\gamma$ is $\Omega(\bar{V}^{-1/3})$.
    - Technical contribution: Beyond proving that gradients are small, the paper further proves that **higher-order derivatives vanish simultaneously**—parameters are "trapped" near their initialization.

3. **Theorem 2: Perfect Accuracy ≠ Good Teacher**:

    - A reward model with acc = 1 but near-zero variance is constructed and compared against one with acc $\leq 2/|\mathcal{Y}|$ but high variance.
    - The latter can improve the true reward arbitrarily faster than the former.

4. **Theorem 3: Different Policies Require Different Reward Models**:

    - Reward variance depends on the policy distribution; the effectiveness of a given reward model differs across initial policies.

### Loss & Training
The paper analyzes the standard RLHF objective:
$$\phi_{\mathrm{RLHF}}(\theta) = \mathbb{E}_x\left[\mathbb{E}_{y \sim \pi_\theta}[r_{\mathrm{RM}}(x,y)] - \lambda \cdot \mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}})\right]$$

## Key Experimental Results

### Main Results

| Reward Model (on-policy %) | Reward Variance | On-Policy Acc | Off-Policy Acc | True Reward Improvement Speed |
|---|---|---|---|---|
| 100% on-policy | 0.630 | 0.660 | 0.596 | Fastest |
| 75% on-policy | 0.616 | 0.659 | 0.610 | Second fastest |
| 50% on-policy | 0.555 | 0.655 | 0.620 | Moderate |
| 25% on-policy | 0.438 | 0.647 | 0.623 | Slower |
| 0% on-policy | 0.314 | 0.626 | **0.651** | Slow |
| Perfect acc, low variance | 0.111 | **1.000** | — | Slowest |
| Ground-truth reward (ArmoRM) | 0.256 | 1.000 | — | Slower than 100% |

### Ablation Study

| Configuration | Key Observation |
|------|---------|
| Highest accuracy (off-policy), 0% on-policy | Lowest reward variance (0.314); slowest true reward improvement after RLHF |
| Perfect accuracy + low variance | Despite acc = 1, optimization is slowest—worse than all imperfect models |
| **Ground-truth vs. proxy reward** | Even with ground-truth reward (acc = 1), proxy reward with higher variance achieves greater improvement within the same number of steps |
| Pythia vs. Llama | The optimal reward model for the same task differs across policy architectures |

### Key Findings
- **Reward variance is the strongest predictor of RLHF performance**: across six training epochs, high-variance reward models consistently outperform low-variance ones.
- **Even ground-truth reward may be insufficient**: ArmoRM (acc = 1) has a variance of only 0.256 and is outperformed by a proxy model with variance 0.630.
- **On-policy training data increases variance**: more on-policy preference pairs yield higher variance, explaining the advantage of on-policy RLHF.
- **Non-transferability across models**: the optimal reward model for Pythia-2.8B is not necessarily optimal for Llama-3.2.

## Highlights & Insights
- The **theoretical proof that accuracy is not sufficient** is the paper's primary contribution. It establishes a complete causal chain: reward variance → landscape flatness → optimization speed.
- The finding that **a perfect reward can be outperformed by a proxy reward** is highly counterintuitive—it suggests that proxy reward models can serve as "signal amplifiers," a insight transferable to any optimization problem involving surrogate objectives.
- **Practical recommendations**: reward model training should incorporate more on-policy data; evaluation should compute reward variance under the target policy rather than relying solely on off-policy benchmarks.
- The **technique of proving simultaneous vanishing of higher-order derivatives** is a reusable analytical tool.

## Limitations & Future Work
- **Tabular policy assumption**: Theorems 2 and 3 are proved only for tabular policies; formal extension to real LLMs remains an open problem.
- **Gradient flow analysis only**: practical training uses stochastic gradient estimates with finite learning rates.
- **Focus on early-stage optimization**: the role of accuracy in preventing reward hacking during prolonged training is not analyzed.
- **Future directions**: (1) incorporating reward variance as a regularization term in reward model training objectives; (2) adaptively adjusting the scale of the reward model.

## Related Work & Insights
- **vs. RewardBench/RM-Bench**: These benchmarks evaluate accuracy exclusively; this paper demonstrates that accuracy is insufficient and argues for including a "variance" dimension.
- **vs. Razin et al. ("Vanishing Gradients in RLHF")**: This paper builds upon that work by further proving lower bounds on optimization speed and establishing stronger results.
- **vs. Best-of-N**: For Best-of-N sampling, accuracy is sufficient (Proposition 1), suggesting that different alignment methods warrant different evaluation criteria.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First rigorous optimization-theoretic proof that accuracy is insufficient for reward model evaluation; the reward variance concept is pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐ Models up to 8B parameters, standard datasets, high agreement between theoretical predictions and empirical results.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical statements are clear and elegant; figures are intuitive.
- Value: ⭐⭐⭐⭐⭐ Shifts the paradigm for reward model evaluation with direct practical implications for RLHF.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization](diffusion_model_as_a_noiseaware_latent_reward_model_for_step.md)
- [\[NeurIPS 2025\] Ask a Strong LLM Judge when Your Reward Model is Uncertain](ask_a_strong_llm_judge_when_your_reward_model_is_uncertain.md)
- [\[NeurIPS 2025\] Limited Preference Data? Learning Better Reward Model with Latent Space Synthesis](limited_preference_data_learning_better_reward_model_with_latent_space_synthesis.md)
- [\[NeurIPS 2025\] Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons](towards_understanding_safety_alignment_a_mechanistic_perspective_from_safety_neu.md)
- [\[NeurIPS 2025\] ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning](responserank_data-efficient_reward_modeling_through_preference_strength_learning.md)

</div>

<!-- RELATED:END -->
