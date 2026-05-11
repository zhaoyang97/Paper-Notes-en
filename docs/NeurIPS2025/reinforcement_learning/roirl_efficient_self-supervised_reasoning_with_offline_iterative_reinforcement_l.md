---
title: >-
  [Paper Note] RoiRL: Efficient, Self-Supervised Reasoning with Offline Iterative Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][Offline reinforcement learning] This paper proposes RoiRL, a lightweight self-supervised reasoning framework based on offline iterative reinforcement learning. By replacing online R…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Offline reinforcement learning"
  - "LLM reasoning"
  - "self-supervised learning"
  - "majority voting"
  - "weighted likelihood"
date: 2026-05-08
content_hash: 5b76fb1ca4ace8d4
---

# RoiRL: Efficient, Self-Supervised Reasoning with Offline Iterative Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.02892](https://arxiv.org/abs/2510.02892)
**Code**: Not available
**Area**: Reinforcement Learning
**Keywords**: Offline reinforcement learning, LLM reasoning, self-supervised learning, majority voting, weighted likelihood

## TL;DR

This paper proposes RoiRL, a lightweight self-supervised reasoning framework based on offline iterative reinforcement learning. By replacing online RL (e.g., TTRL) with a weighted log-likelihood objective, RoiRL enables self-improvement of LLM reasoning capabilities without requiring a reference model or ground-truth labels, achieving 2.5× faster training with superior performance.

## Background & Motivation

Reinforcement learning (RL) plays a central role in enhancing the reasoning capabilities of large language models (LLMs). However, conventional RL methods rely on ground-truth labels as reward signals, which presents a significant bottleneck at scale. Test-Time Reinforcement Learning (TTRL) eliminates this dependency by using majority voting as a weak supervision signal, yet it faces two critical challenges:

**High computational cost**: TTRL requires maintaining a reference model during training and computing its logits. Combined with repeated chain-of-thought (CoT) sampling, GPU memory consumption saturates rapidly, making it difficult to scale to larger models.

**Online RL instability**: GRPO-based online training is highly sensitive to hyperparameters, exhibits large performance variance, and is difficult to deploy in practice.

The core problem is: **Can a method as simple and stable as supervised fine-tuning achieve the same optimization objective as TTRL?** RoiRL answers this question affirmatively.

## Method

### Overall Architecture

RoiRL adopts an iterative offline learning paradigm in which each iteration consists of two steps: a **generation phase** and an **offline update phase**. In the generation phase, the current policy $\pi_{m-1}$ samples $k$ candidate solutions for each problem and scores them via majority voting. In the offline update phase, policy parameters are optimized using a weighted log-likelihood objective, requiring neither online interaction nor a reference model.

### Key Designs

1. **Majority voting reward signal**: For each problem $x_i$, the model generates $k$ candidate answers $\{y_i^\ell\}_{\ell \in [k]}$, and a pseudo-label is determined via majority voting: $\tilde{y}_i^k(\theta) = \text{maj}_{\ell \in [k]}(y_i^\ell)$, with the reward function defined as $\tilde{r}_k(y, x_i, \theta) = \mathbb{1}[y = \tilde{y}_i^k(\theta)]$. This approach requires no ground-truth labels and leverages the model's own consistency signal.

2. **Weighted log-likelihood objective**: The core optimization objective is:
$$\theta_m = \arg\max_{\theta} \sum_{i=1}^{n} \mathbb{E}_{(c,y) \sim \pi_{m-1}(\cdot|x_i)} \left[ g_m(\tilde{r}_k(y, x_i, \theta_{m-1})) \log \pi_\theta(c, y | x_i) \right]$$
where $g_m: \mathbb{R} \to \mathbb{R}$ is a monotonically increasing reward transformation function. This is essentially weighted supervised fine-tuning on correct answers, yielding greater stability than online RL.

3. **Two reward transformation instances**:

    - **Identity transformation $g_I(r) = r$**: Reduces to simple SFT on correct answers, retaining only samples consistent with the majority vote for training.
    - **Exponential transformation $g_\beta(r) = \exp(r/\beta)$**: Simulates a KL-regularized objective, targeting the same theoretical optimum as TTRL.

### Loss & Training

**Theoretical guarantee**: The paper proves that the closed-form solution of RoiRL is $\pi_m(c,y|x) \propto \left(\prod_{j=1}^{m} g_j(\tilde{r}_k(y,x,\theta_{j-1}))\right) \pi_0(c,y|x)$. When $g_j(r) = \exp(r/\beta)$ is selected, this solution is consistent in form with the closed-form solution of the KL-regularized RL objective (Proposition 3.1).

**Key advantages**:
- Eliminates the need to maintain a reference model $\pi_0$, substantially reducing memory consumption.
- Offline updates resemble SFT, resulting in stable training.
- The iterative design naturally addresses the non-stationary reward problem.

## Key Experimental Results

### Main Results

Models are trained on MATH500 Train (without labels) and evaluated on reasoning accuracy across three benchmarks:

| Model | Decoding | MATH500 Train | MATH500 Test | AMC | AIME |
|------|------|--------------|--------------|-----|------|
| Qwen2.5-Math-1.5B (Base) | maj₁ | 0.244 | 0.239 | 0.170 | 0.036 |
| TTRL | maj₁ | 0.307 | 0.298 | 0.214 | 0.026 |
| **RoiRL $g_I$** | maj₁ | **0.686** | 0.587 | 0.337 | **0.083** |
| **RoiRL $g_\beta$** | maj₁ | 0.670 | **0.604** | **0.340** | 0.070 |
| Phi4-mini-4B (Base) | maj₁ | 0.210 | 0.160 | 0.071 | 0.000 |
| **RoiRL $g_I$** | maj₁ | **0.660** | **0.511** | **0.246** | **0.016** |
| Llama-3.2-3B (Base) | maj₁₀ | 0.495 | 0.480 | 0.253 | 0.033 |
| TTRL | maj₁₀ | **0.510** | 0.490 | **0.313** | 0.167 |
| **RoiRL $g_I$** | maj₁₀ | 0.508 | 0.520 | **0.313** | **0.200** |

### Ablation Study

| Configuration | Training Speed | Reference Model | Stability | Notes |
|------|---------|---------|--------|------|
| TTRL (GRPO online) | 1× | Required | Poor | Hyperparameter-sensitive, memory saturation |
| RoiRL $g_\beta$ | ~2× | Not required | Good | Simulates KL-regularized objective |
| RoiRL $g_I$ | **~2.5×** | Not required | **Best** | Simple SFT-style update, best performance |

### Key Findings

- RoiRL $g_I$ achieves the best performance in most settings, suggesting that **simple identity transformation may be more effective than KL regularization**.
- RoiRL constitutes genuine self-improvement rather than majority-vote distillation: after training, the model's maj₁ decoding can surpass the base model's maj₁₀, and maj₁₀ can surpass the base model's maj₁₂₈.
- RoiRL consistently outperforms TTRL across three architectures (Qwen, Phi4, Llama), validating the robustness of the approach.

## Highlights & Insights

- Recasting online RL as offline weighted SFT substantially lowers the technical barrier for self-supervised reasoning.
- The paper theoretically demonstrates that RoiRL and TTRL can target the same optimal policy; however, the simpler objective proves more effective in practice.
- The non-stationary reward problem (majority voting shifts as the policy evolves) is naturally resolved within the iterative framework.

## Limitations & Future Work

- Validation is limited to small-scale models (1.5B–4B) under constrained computational budgets; further evaluation on larger LLMs is needed.
- The quality ceiling of majority voting as pseudo-labels is bounded by the model's own capability, which may lead to erroneous reinforcement.
- The choice of reward transformation function $g$ significantly affects performance, yet no adaptive selection mechanism has been established.
- Experiments are confined to mathematical reasoning tasks; generalization to other reasoning domains such as code generation remains to be verified.

## Related Work & Insights

- **TTRL**: Pioneered the use of majority voting in lieu of ground-truth labels for RL training, but incurs high cost due to its online formulation.
- **GRPO/DPO**: Classical methods for KL-regularized RL; RoiRL demonstrates that equivalent effects can be achieved via a simpler offline approach.
- **Offline RL (AWR/REPS)**: The weighted likelihood objective in RoiRL is directly inspired by the offline RL literature.
- **Insight**: In LLM training, simpler methods often outperform more complex ones—consistent with the practical observation that SFT can surpass PPO.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Applying offline RL ideas to self-supervised reasoning is a natural yet effective contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three models, multiple benchmarks, and training efficiency analysis are included, though large-model experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretical derivations are clear, and the logical flow from motivation to method is coherent.
- **Value**: ⭐⭐⭐⭐ Provides a more practical training paradigm for self-supervised LLM reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sample-Efficient Tabular Self-Play for Offline Robust Reinforcement Learning](sample-efficient_tabular_self-play_for_offline_robust_reinforcement_learning.md)
- [\[NeurIPS 2025\] Structural Information-based Hierarchical Diffusion for Offline Reinforcement Learning](structural_information-based_hierarchical_diffusion_for_offline_reinforcement_le.md)
- [\[NeurIPS 2025\] Forecasting in Offline Reinforcement Learning for Non-stationary Environments](forecasting_in_offline_reinforcement_learning_for_non-stationary_environments.md)
- [\[NeurIPS 2025\] Adaptive Neighborhood-Constrained Q Learning for Offline Reinforcement Learning](adaptive_neighborhoodconstrained_q_learning_for_offline_rein.md)
- [\[NeurIPS 2025\] Online Optimization for Offline Safe Reinforcement Learning](online_optimization_for_offline_safe_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
