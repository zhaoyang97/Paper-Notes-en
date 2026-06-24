---
title: >-
  [Paper Note] Enhancing Decision-Making of Large Language Models via Actor-Critic
description: >-
  [ICML2025][Reinforcement Learning][Actor-Critic] This work proposes the LAC (LLM-based Actor-Critic) framework, which constructs a Q-function (Critic) using the ratio of positive/negative outcome probabilities from token logits and achieves gradient-free policy optimization (Actor) via a closed-form solution with a KL constraint. It outperforms GPT-4 + ReAct on three benchmarks (ALFWorld, BabyAI-Text, WebShop) using 7B/8B models.
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "Actor-Critic"
  - "LLM Agent"
  - "Gradient-Free Policy Optimization"
  - "Q-Value Estimation"
  - "Sequential Decision Making"
date: 2026-05-08
content_hash: eb8c3a713b467778
---

# Enhancing Decision-Making of Large Language Models via Actor-Critic

**Conference**: ICML2025  
**arXiv**: [2506.06376](https://arxiv.org/abs/2506.06376)  
**Code**: [GitHub](https://github.com/drdh/LAC)  
**Area**: LLM Decision Making / Reinforcement Learning / Agent  
**Keywords**: Actor-Critic, LLM Agent, Gradient-Free Policy Optimization, Q-Value Estimation, Sequential Decision Making

## TL;DR

This work proposes the LAC (LLM-based Actor-Critic) framework, which constructs a Q-function (Critic) using the ratio of positive/negative outcome probabilities from token logits and achieves gradient-free policy optimization (Actor) via a closed-form solution with a KL constraint. It outperforms GPT-4 + ReAct on three benchmarks (ALFWorld, BabyAI-Text, WebShop) using 7B/8B models.

## Background & Motivation

There are two main lines of work when using LLMs for sequential decision-making, each having significant drawbacks:

**Directly utilizing the LLM prior as a policy** (e.g., ReAct): It autoregressively generates actions step by step, lacking long-term planning capabilities and often resulting in local optima but global failure in multi-step tasks.

**Incorporating planning and action evaluation** (e.g., RAP, LATS): It uses the LLM to perform rollouts or MCTS to evaluate candidate actions, but heavily relies on simulation accuracy, leading to drastic performance degradation on lightweight models when rollout deviations are large.

Core Problem: Both directions decouple **LLM prior knowledge** from **action evaluation information**—the former does not plan, while the latter ignores the prior. The goal of LAC is to unify both in a theoretically guaranteed framework.

## Method

### Overall Architecture

At each time step, LAC performs two steps:

1. **Critic Evaluation**: Calculates the Q-value for each of the $n$ candidate actions sampled from the policy $\pi_{\text{LLM}}$;
2. **Actor Optimization**: Updates the prior policy using the Q-values through a closed-form update under a KL constraint, then selects the optimal action.

### 4.1 Critic: Token Logits-Based Q-Value Estimation

**Core Idea**: Instead of having the LLM directly output a rating (which is unstable), this work utilizes the LLM's logits for specific tokens (such as "GOOD"/"BAD" or "SUCCESS"/"FAILURE") to reflect its internal belief about task success or failure.

**Q-Value Formula**:

$$Q_{\text{LLM}}(g, h_t, a_t^i, u_t^i) = \log \frac{P(y_w \mid g, h_t, a_t^i, u_t^i)}{P(y_l \mid g, h_t, a_t^i, u_t^i)}$$

where $y_w$ and $y_l$ correspond to the success and failure signals, respectively, and $u_t^i$ is the future trajectory predicted by the forward world model $f_{\text{LLM}}$. The Q-value is positively correlated with the success probability via the logistic function:

$$P(y_w \mid \cdot) = \frac{1}{1 + \exp(-Q_{\text{LLM}}(\cdot))}$$

**Two Techniques for Improving Evaluation Accuracy**:

- **Trajectory Rollout**: Predicts several steps of future trajectory for each candidate action using the LLM, and calculates the Q-value based on this extended trajectory to capture delayed consequences.
- **Contextual Reflection**: Before sampling and evaluation, the LLM generates a brief reflection (e.g., "I have found object-X, this step is GOOD"), similar to CoT. This helps the policy avoid repetitive errors and improves Critic accuracy.

### 4.2 Actor: KL-Constrained Gradient-Free Policy Optimization

Formulates policy improvement as a KL-constrained optimization problem:

$$\max_{\pi} \; \mathbb{E}_{a_t^i \sim \pi}[Q_{\text{LLM}}(\cdot)] - \frac{1}{\alpha} D_{\text{KL}}[\pi \| \pi_{\text{LLM}}]$$

Closed-form optimal solution:

$$\pi_{\text{new}}(a_t^i \mid g, h_t) \propto \pi_{\text{LLM}}(a_t^i \mid g, h_t) \cdot \exp\!\big(\alpha \, Q_{\text{LLM}}(g, h_t, a_t^i, u_t^i)\big)$$

- $\alpha = 0$ degrades to the pure prior policy (ReAct); $\alpha \to \infty$ degrades to pure Critic action selection.
- **No gradient backpropagation is required**, as the policy update is completed solely by weighting the prior probabilities, leading to minimal computational overhead.
- The KL term ensures that the new policy does not deviate too far from the prior, balancing prior knowledge with Critic evaluation.

### Algorithmic Workflow

1. Sample $n$ candidate actions from $\pi_{\text{LLM}}$;
2. Predict future trajectories via rollouts for each candidate action using $f_{\text{LLM}}$;
3. Calculate Q-values using the positive/negative token logits;
4. Perform weighted policy probability updates according to the closed-form solution;
5. Select the action with the highest probability for execution.

## Key Experimental Results

### Benchmarks and Action Spaces

| Benchmark | Action Type | Reward Type | Scale |
|------|---------|---------|------|
| ALFWorld | High-level (e.g., "go to X take Y") | Binary 0/1 | 134 Tasks |
| BabyAI-Text | Low-level 6 primitive actions | Binary 0/1 | 8x8 grid |
| WebShop | Almost infinite (Search + Click) | Continuous [0,1] | Web shopping |

### Main Results

- **ALFWorld**: The success rate of LAC + Llama-3-8B significantly outperforms ReAct + GPT-4, as well as planning methods like RAP and LATS.
- **BabyAI-Text**: LAC consistently achieves state-of-the-art results across all subtasks, demonstrating a clear advantage particularly in long-horizon tasks.
- **WebShop**: LAC achieves the best performance in both cumulative reward and success rate, proving that the framework is equally effective in continuous reward scenarios.

### Ablation Study

| Variant | Effect |
|------|------|
| LAC w/o critic | Significant performance drop, validating the necessity of the policy optimization step |
| LAC w/o rollout | Performance drop, showing that future trajectory prediction is critical for Q-value accuracy |
| LAC w/o reflection | Performance drop, indicating that the reflection mechanism helps sample better candidates and yields more accurate evaluations |
| critic-only | Performance drop, showing that pure Critic is inferior to the combination with the prior |

### Computational Cost

- Although LAC has a slightly higher step-wise overhead (due to extra Critic + rollout inference), its **total token consumption and execution time are actually lower than baselines like RAP and LATS** because of its high success rate and fewer steps to completion.
- Average steps for successful tasks: LAC 15.32 steps vs ReAct 17.75 steps vs RAP 16.36 steps.

### Statistical Analysis

| Metric | Successful Trajectories | Failed Trajectories |
|------|---------|---------|
| Correlation of log P("GOOD") with time step | +0.35 | -0.37 |
| Correlation of log P("BAD") with time step | -0.32 | +0.38 |
| Correlation of Q-value with time step | +0.34 | -0.41 |

The Q-value increases over time steps in successful trajectories and decreases in failed trajectories, validating that the Q-function indeed tracks task progress.

## Highlights & Insights

1. **Clever Q-Value Estimation**: Instead of letting the LLM directly output scores (which is highly unstable), this work takes the logarithm of the ratio of positive/negative token logits. The formula is concise and carries a clear physical meaning—the log-odds ratio of success to failure.
2. **Closed-Form Policy Optimization**: Derives an analytical solution of exponentially weighted updates under a KL constraint. Since it is entirely gradient-free, it is highly suitable for test-time computation in LLM scenarios. This solution is equivalent to the policy update form in AWR / DPO-style methods, providing a solid theoretical foundation.
3. **Continuous Spectrum Interpretation of $\alpha$**: $\alpha=0$ recovers ReAct, and $\alpha \to \infty$ recovers pure Critic. LAC automatically balances between the two extremes.
4. **Thorough Statistical Validation**: Beyond standard ablations, the authors perform correlation analysis between Q-values and time steps, as well as policy confidence analysis. This demonstrates that the weighted policy indeed adheres to "following whoever is more confident" rather than blindly mixing them.
5. **7B Model Outperforming GPT-4**: Demonstrates that framework design is more critical than model scale, which is of great significance for resource-constrained scenarios.

## Limitations & Future Work

1. **Reflection is only used before action generation**: It can be extended to perform reflection on predicted trajectories after generation to enable resampling.
2. **Single-step rollout expansion**: Currently, only one node is expanded for each candidate action. This can be integrated with tree search (e.g., MCTS) to obtain more precise evaluations.
3. **Coarse handling of continuous rewards**: Currently, "obtaining the maximum reward" is binarized, lacking specialized modeling for continuous rewards.
4. **Larger models not verified**: Tests were only conducted on 7B/8B models; the effectiveness on 70B+ or newer reasoning models (e.g., DeepSeek-R1) remains unknown.
5. **Requires access to token logits**: Relies on the model's output logits, which is not applicable to API-only closed-source models.

## Related Work & Insights

- **ReAct** (Yao et al., 2023): Integrates reasoning and acting but lacks long-term planning $\to$ corresponds to the "w/o Critic" baseline of LAC.
- **RAP** (Hao et al., 2023): Uses LLM as a world model + tree search $\to$ LAC's rollout component is similar but more lightweight.
- **LATS** (Zhou et al., 2024a): MCTS + LLM $\to$ incurs high computational overhead, which LAC replaces with single-step rollouts and closed-form optimization.
- **ICPI** (Brooks et al., 2024): Implements policy iteration on LLMs $\to$ performs poorly under sparse rewards.
- **DPO/AWR Series**: The theoretical basis for KL-constrained policy optimization, which LAC transfers from training-time learning to test-time decision-making.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of extracting Q-values from token logits and closed-form policy optimization during inference is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers three distinct action space benchmarks, four base models, comprehensive ablations, and statistical analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, complete mathematical derivations, and rich tables.
- Value: ⭐⭐⭐⭐ — Provides a concise and efficient paradigm for test-time decision optimization of LLM Agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Risk-Sensitive Exponential Actor Critic](../../AAAI2026/reinforcement_learning/risk-sensitive_exponential_actor_critic.md)
- [\[ICML 2025\] The Sample Complexity of Online Strategic Decision Making with Information Asymmetry and Knowledge Transportability](the_sample_complexity_of_online_strategic_decision_making_with_information_asymm.md)
- [\[ICML 2025\] Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making](counterfactual_effect_decomposition_in_multi-agent_sequential_decision_making.md)
- [\[NeurIPS 2025\] MMaDA: Multimodal Large Diffusion Language Models](../../NeurIPS2025/reinforcement_learning/mmada_multimodal_large_diffusion_language_models.md)
- [\[NeurIPS 2025\] MRO: Enhancing Reasoning in Diffusion Language Models via Multi-Reward Optimization](../../NeurIPS2025/reinforcement_learning/mro_enhancing_reasoning_in_diffusion_language_models_via_multi-reward_optimizati.md)

</div>

<!-- RELATED:END -->
