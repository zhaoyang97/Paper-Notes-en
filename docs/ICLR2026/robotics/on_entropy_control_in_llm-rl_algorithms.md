---
title: >-
  [Paper Note] On Entropy Control in LLM-RL Algorithms
description: >-
  [ICLR 2026][Robotics][Entropy Control] The authors theoretically explain why traditional entropy regularization is nearly ineffective in LLM-RL (due to immense action spaces and sparse optima causing entropy bias to overwhelm optimization gains). They propose the AEnt method, which uses clamped entropy (calculated on a reduced token space) and adaptive coefficients to effectively balance bias and benefits, consistently outperforming baselines in mathematical reasoning tasks.
tags:
  - "ICLR 2026"
  - "Robotics"
  - "Entropy Control"
  - "RLVR"
  - "LLM-RL"
  - "Policy Optimization"
  - "Exploration-Exploitation"
date: 2026-05-08
content_hash: 8adb59d862154983
---

# On Entropy Control in LLM-RL Algorithms

**Conference**: ICLR 2026  
**arXiv**: [2509.03493](https://arxiv.org/abs/2509.03493)  
**Code**: [antgroup/AEnt](https://github.com/antgroup/AEnt)  
**Area**: Robotics  
**Keywords**: Entropy Control, RLVR, LLM-RL, Policy Optimization, Exploration-Exploitation

## TL;DR
The authors theoretically explain why traditional entropy regularization is nearly ineffective in LLM-RL (due to immense action spaces and sparse optima causing entropy bias to overwhelm optimization gains). They propose the AEnt method, which uses clamped entropy (calculated on a reduced token space) and adaptive coefficients to effectively balance bias and benefits, consistently outperforming baselines in mathematical reasoning tasks.

## Background & Motivation

**Background**: Policy gradient methods (PPO/GRPO/DAPO) are dominant in LLM-RL. In traditional RL, entropy regularization (SAC/A3C/PPO) significantly improves performance by maintaining policy randomness to prevent premature convergence.

**Limitations of Prior Work**: Empirical findings indicate that entropy regularization provides almost no gain in LLM-RL. Cui et al. observed that different entropy coefficients have negligible effects on validation accuracy, creating a contradiction with the significant benefits observed in robotics and game RL.

**Key Challenge**: Theoretically, while entropy regularization offers optimization advantages (improving convergence), the introduced bias $O(H\log\frac{|\mathcal{A}|}{|\mathcal{A}_H^*(s_0)|^{1/H}})$ scales sharply with the action space $|\mathcal{A}|$ and optimal sparsity. In LLMs, where vocabularies are ~100k+ and optimal tokens are extremely sparse, the bias far outweighs any optimization gains.

**Key Insight**: Since entropy bias over the entire vocabulary is excessive, it is more effective to calculate clamped entropy over a smaller space of reasonable tokens—encouraging exploration among "likely candidates" rather than the entire vocabulary.

## Method

### Overall Architecture

This paper first addresses a point of confusion before providing a solution. The confusion is: why does entropy regularization, which is highly effective in traditional RL, fail to show improvements in LLM-RL (policy gradient methods like PPO/GRPO/DAPO)? The paper quantifies the cause using two performance bounds and subsequently proposes AEnt, which calculates entropy over a small set of "reasonable candidate tokens" to preserve exploration benefits while suppressing bias.

The entire analysis is built on two propositions. **Without entropy control** (Proposition 1), policy entropy serves as an upper bound for the policy gradient norm: $\|\nabla V^{\pi_\theta}\| \leq 2\mathcal{H}(\pi_\theta)$. Once entropy collapses, the gradient tends toward zero, and learning plateaus; the performance gap is bounded by $V^{\pi^*} - V^{\pi_\theta} \leq \frac{\epsilon}{C^{\pi_\theta}(s_0)}$. **With traditional entropy regularization** (Proposition 2), the performance gap becomes:

$$V^{\pi^*} - V^{\pi_\theta} \leq \frac{\epsilon^2}{2\lambda C_\lambda} + \lambda H\log\frac{|\mathcal{A}|}{|\mathcal{A}_H^*|^{1/H}}$$

The first term $\epsilon^2/2\lambda$ represents the optimization gain from entropy (better convergence), while the second term $\lambda H\log\frac{|\mathcal{A}|}{|\mathcal{A}_H^*|^{1/H}}$ is the introduced bias. The critical observation is that LLM vocabularies $|\mathcal{A}|$ reach the $100,000$ range, and optimal tokens are extremely sparse, causing the $\log|\mathcal{A}|$ term to explode and completely overwhelm the optimization gain. This explains the fundamental difference between LLM-RL and robotics/game RL (where $|\mathcal{A}|$ is only tens or hundreds). Synthetic MDP experiments ($|\mathcal{A}|=10^5$) confirm this: traditional entropy helps when there are 10–15 optimal actions but fails completely when the count drops below 5. All designs in AEnt revolve around narrowing the scope of entropy from the whole vocabulary to a set of reasonable candidates to minimize this bias.

### Key Designs

**1. Clamped Entropy: Calculating entropy only on top-probability tokens to reduce effective $|\mathcal{A}|$ in the bias term**

The bias term explodes because traditional entropy pulls the policy toward a uniform distribution $1/|\mathcal{A}|$, encouraging randomness across all 100k tokens. Since most tokens should not be explored, pulling toward them generates pure bias. Clamped entropy retains only the top $(1-p)$ proportion of tokens (discarding the long tail $p=0.25\sim0.33$), forming a state-dependent subspace $\mathcal{A}(s)$ where the policy is re-normalized:

$$\tilde{\pi}_\theta(a|s) = \frac{\exp(\theta_{s,a})}{\sum_{a' \in \mathcal{A}(s)} \exp(\theta_{s,a'})}, \quad \mathcal{A}(s) = \{\text{top }(1-p)\text{ tokens}\}$$

Entropy is then calculated using $\tilde{\pi}_\theta$. Consequently, exploration only occurs among "reasonable candidates." The bias term is no longer driven by $\log|\mathcal{A}|$ but by a much smaller set of candidates, while the optimization gain is largely preserved. The intuition is that for a pre-trained/fine-tuned model, low-probability tokens are unlikely to be optimal; excluding them reduces bias without harming exploration.

**2. Adaptive Coefficient: Pulling $\lambda$ back to a target range based on current clamped entropy**

A fixed entropy coefficient $\lambda$ suffices for games, but in LLM training, entropy fluctuates violently (e.g., saturating after ~200 steps). AEnt sets a target range $[\tilde{\mathcal{H}}_{\text{low}}, \tilde{\mathcal{H}}_{\text{high}}]$ for clamped entropy and updates the coefficient after each global step using a projection:

$$\lambda' \leftarrow \text{Proj}_{[\lambda_{\text{low}}, \lambda_{\text{high}}]}\big[\lambda - \beta\min(\tilde{\mathcal{H}}(\pi_\theta) - \tilde{\mathcal{H}}_{\text{low}},\, 0) + \beta\min(\tilde{\mathcal{H}}_{\text{high}} - \tilde{\mathcal{H}}(\pi_\theta),\, 0)\big]$$

In simpler terms: if clamped entropy falls below the lower bound (insufficient exploration), $\lambda$ is increased; if it exceeds the upper bound (sufficient stochasticity), $\lambda$ is decreased to prioritize reward maximization and consume excess entropy. $\lambda$ itself is clipped within $[\lambda_{\text{low}}, \lambda_{\text{high}}]$ to prevent overshoot. This adaptive mechanism also suppresses spikes in answer length, making inference more efficient.

### Loss & Training

Combining the two components, AEnt optimizes a target that adds a clamped entropy regularization term to the original policy optimization loss:

$$\mathcal{L}_{\text{AEnt}}(\theta; \lambda) = \mathcal{L}_{\text{PO}}(\theta) + \lambda \tilde{\mathcal{H}}(\pi_\theta)$$

Here, $\mathcal{L}_{\text{PO}}$ is the underlying policy optimization objective (GRPO in the experiments), $\tilde{\mathcal{H}}$ is the clamped entropy, and $\lambda$ is updated via the projection rule. The only modification to the original entropy regularization is replacing global entropy with clamped entropy and the fixed coefficient with an adaptive one.

## Key Experimental Results

### Mathematical Reasoning Benchmarks (Model evaluated at highest training checkpoint; average of 4 samples per problem)

Setup (a) = Qwen2.5-Math-1.5B trained on MATH; Setup (b) = DeepSeek-R1-Distilled-Qwen-1.5B trained on OpenR1-Math subset. EntReg denotes traditional entropy regularization (GRPO + original entropy bonus).

| Method | Setup | MATH-Hard | MATH-500 | AIME24 | Minerva | Olympiad | AMC |
|------|------|-----------|----------|--------|---------|----------|-----|
| Base | (a) | 0.368 | 0.584 | 0.083 | 0.179 | 0.279 | 0.406 |
| GRPO | (a) | 0.524 | **0.756** | 0.192 | 0.311 | 0.364 | 0.550 |
| EntReg | (a) | 0.546 | 0.752 | 0.167 | 0.316 | 0.370 | 0.562 |
| **AEnt** | (a) | **0.552** | 0.750 | **0.217** | **0.330** | **0.377** | **0.581** |
| Base | (b) | 0.661 | 0.792 | 0.225 | 0.311 | 0.432 | 0.594 |
| GRPO | (b) | 0.773 | 0.865 | 0.367 | 0.347 | 0.576 | 0.769 |
| EntReg | (b) | 0.808 | 0.872 | 0.342 | 0.359 | 0.576 | 0.794 |
| **AEnt** | (b) | **0.813** | **0.882** | **0.392** | 0.359 | **0.591** | **0.825** |

### Key Findings
- **Traditional entropy yields almost no gain and sometimes degrades performance**: EntReg shows minimal change compared to GRPO across most benchmarks and even suffers a decline in AIME24 (e.g., 0.192 to 0.167 in setup (a)), confirming Cui et al.'s observations.
- **AEnt consistently leads**: Across both setups, AEnt achieves the best results in 5 out of 6 benchmarks on average (with MATH-500 in setup (a) being the sole exception), showing that clamped entropy effectively addresses the bias issue.
- **Synthesized MDP identifies the root cause**: In a toy MDP where $|\mathcal{A}|=10^5$, traditional entropy helps when the number of optimal actions is 10–15, but fails when it is less than 5; clamped entropy remains effective, directly validating the bias analysis in Proposition 2.
- **Adaptive coefficients are more stable**: Fixed coefficients lose control when entropy fluctuates sharply. Adaptive coefficients keep entropy and answer length within reasonable bounds, preventing explosion.

## Highlights & Insights
- **Theoretic explanation for a long-standing puzzle in LLM-RL**: Why does traditional entropy not work in LLMs? Because the $O(H\log|\mathcal{A}|)$ bias at $|\mathcal{A}|=10^5$ overwhelms everything. This explanation is concise and powerful.
- **Intuition for Clamped Entropy**: One should not encourage the model to explore all 100k tokens; diversity should be maintained only among reasonable candidates. Choosing randomly from top-1000 is far more logical than choosing from the entire vocabulary.
- **Quantifying the Bias-Gain Trade-off**: Propositions 1 and 2 provide actionable theoretical guidance—when $\log|\mathcal{A}|$ is large and the optima are sparse, special handling is required.

## Limitations & Future Work
- The parameter $k$ for top-k currently requires manual setting; an adaptive $k$ might be superior.
- The theoretical analysis assumes a softmax policy, while real LLMs have more complex structures.
- Validation is limited to mathematical reasoning; effects on code or general reasoning are yet to be explored.
- Clamped entropy might overly restrict scenarios requiring extremely broad exploration.

## Related Work & Insights
- **vs DAPO**: DAPO controls entropy indirectly through clipping/constraints, while AEnt applies direct regularization over a truncated space.
- **vs Cui et al.**: They observed the ineffectiveness of entropy bonuses but lacked a theoretical explanation; this paper provides both the explanation and a solution.
- **vs SAC**: SAC's entropy regularization succeeds in robotics because $|\mathcal{A}|$ is small (tens to hundreds), whereas LLM vocabularies are several orders of magnitude larger.

## Rating
- Novelty: ⭐⭐⭐⭐ Both the theoretical explanation and the clamped entropy solution are insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple models, benchmarks, and synthetic MDPs.
- Writing Quality: ⭐⭐⭐⭐ Natural integration of theory and practice.
- Value: ⭐⭐⭐⭐⭐ Solves a significant practical problem in LLM-RL training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Self-Improving Vision-Language-Action Models with Data Generation via Residual RL](self-improving_vision-language-action_models_with_data_generation_via_residual_r.md)
- [\[ICLR 2026\] SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning](simplevla-rl_scaling_vla_training_via_reinforcement_learning.md)
- [\[ICLR 2026\] Masked Generative Policy for Robotic Control](masked_generative_policy_for_robotic_control.md)
- [\[ICLR 2026\] Scaling up Memory for Robotic Control via Experience Retrieval](scaling_up_memory_for_robotic_control_via_experience_retrieval.md)
- [\[ECCV 2024\] LLM as Copilot for Coarse-Grained Vision-and-Language Navigation](../../ECCV2024/robotics/llm_as_copilot_for_coarse-grained_vision-and-language_navigation.md)

</div>

<!-- RELATED:END -->
