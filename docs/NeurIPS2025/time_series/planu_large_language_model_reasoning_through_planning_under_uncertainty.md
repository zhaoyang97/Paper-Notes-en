---
title: >-
  [Paper Note] PlanU: Large Language Model Reasoning through Planning under Uncertainty
description: >-
  [NeurIPS 2025][Time Series][LLM decision-making] This paper proposes PlanU—an LLM decision-making method that models node returns via quantile distributions within MCTS and balances exploration and exploitation through a…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "LLM decision-making"
  - "uncertainty"
  - "Monte Carlo tree search"
  - "quantile distribution"
  - "exploration and exploitation"
date: 2026-05-08
content_hash: 6f2483cfdb8dad4c
---

# PlanU: Large Language Model Reasoning through Planning under Uncertainty

**Conference**: NeurIPS 2025
**arXiv**: [2510.18442](https://arxiv.org/abs/2510.18442)  
**Authors**: Ziwei Deng, Mian Deng (Xiamen University), Chenjing Liang, Zeming Gao, Chennan Ma, Chenxing Lin, Haipeng Zhang, Songzhu Mei (National University of Defense Technology), Cheng Wang, Siqi Shen (Xiamen University)
**Code**: [GitHub](https://github.com/Ber666/llm-reasoners)  
**Area**: Time Series
**Keywords**: LLM decision-making, uncertainty, Monte Carlo tree search, quantile distribution, exploration and exploitation

## TL;DR

This paper proposes PlanU—an LLM decision-making method that models node returns via quantile distributions within MCTS and balances exploration and exploitation through an Upper Confidence Bounds with Curiosity (UCC) score. PlanU is the first approach to systematically and simultaneously address both LLM uncertainty and environmental uncertainty, achieving substantial improvements over existing methods across multiple stochastic environment benchmarks.

## Background & Motivation

### State of the Field
LLMs have achieved remarkable success in reasoning and decision-making tasks, yet their performance remains poor in **uncertain environments**. LLM decision-making faces two categories of uncertainty: (1) **LLM uncertainty**—arising from the stochastic sampling process of LLMs, where the same prompt may yield different outputs; and (2) **environmental uncertainty**—arising from stochastic state transitions, where the same action may lead to different next states.

### Limitations of Prior Work
- Methods such as **CoT / Self-Consistency / ToT / RAP** address LLM uncertainty via multiple sampling or tree search, but completely ignore environmental uncertainty.
- **DeLLMa** accounts for environmental uncertainty but is limited to single-step decisions and cannot handle multi-step interactive tasks.
- **Standard MCTS (e.g., as used in RAP)** assumes deterministic transitions; when confronted with stochastic environments, it selects the most frequently observed state as the child node, leading to suboptimal decisions.
- Naive ensemble approaches (e.g., incorporating uncertainty considerations into prompts) perform extremely poorly—a fact the paper validates empirically via a stock investment task.

### Root Cause
In the real world, environmental stochasticity is ubiquitous—even in nominally deterministic environments, state aliasing (e.g., partial observability) introduces effective randomness. A systematic method that simultaneously addresses both LLM uncertainty and environmental uncertainty is therefore needed.

## Method

### Overall Architecture
PlanU is built upon MCTS with two core innovations: (1) replacing the mean used in standard MCTS with a **quantile distribution** to model node returns; and (2) designing a **UCC score** to guide exploration.

### Quantile Distribution Modeling
Standard MCTS represents node value using the expected value $Q(s,a)$. PlanU replaces this with a quantile distribution $Z(s,a)$:

$$Z(s,a) = \sum_{i=1}^{n_q} \delta_{\theta(s,a,\tau_i)} p_i(s,a,\tau_i)$$

where $\theta(s,a,\tau_i)$ is the $i$-th quantile value and $n_q$ is the number of quantiles. A skewed quantile distribution indicates high uncertainty, while a uniform distribution indicates low uncertainty.

### Four-Phase Tree Search
1. **Selection**: Starting from the root node, child action nodes are selected according to the UCC score: $a^* = \arg\max_{a_t} UCC(s_t, a_t)$.
2. **Expansion**: The action nodes of leaf nodes are expanded, and the quantile distribution is initialized using the LLM generation probability $\pi(s_t,a_t) = \prod_{i=1}^n p(t_i|c)$.
3. **Simulation**: Multiple trajectories are simulated from new nodes to terminal states to obtain actual environmental feedback.
4. **Back-propagation**: The distribution is updated along the path via quantile regression (QR), with target distribution $y(s_{t+1},a_{t+1}) = r + \gamma Z(s_{t+1},a)$ and a quantile Huber loss.

### UCC Score Design
The UCC score combines the value distribution with state novelty:

$$UCC(s_t,a_t) = \psi[Z(s_t,a_t)] + c_1 \cdot \frac{r_i(s_t)}{N(s_t,a_t)}$$

- $\psi[Z(s_t,a_t)]$: an operator mapping the quantile distribution to a scalar, defaulting to the expectation $\mathbb{E}[Z(s_t,a_t)]$, with optional consideration of distributional spread.
- $r_i(s_t)$: a novelty reward inspired by Random Network Distillation (RND).

### Novelty Reward and LLM Uncertainty Handling
The novelty reward is defined as $r_i(s_t) = |\hat{f}(e(s_t)) - f(e(s_t))|^2$, where:
- $f$ is a fixed randomly initialized target network and $\hat{f}$ is a trainable predictor network.
- $e(\cdot)$ is a text encoder that maps textual states to feature vectors, addressing the issue of LLMs generating different textual descriptions for semantically identical states (e.g., "the person is to the right of the table" vs. "the table is to the left of the person").
- The predictor network is trained by maintaining a buffer $\mathcal{B}$ of previously visited states.

## Key Experimental Results

### Experiment 1: Stock Investment Task (Validating Intuition)

A simple investment scenario: Stock A yields a fixed return of 0.9; Stock B yields 1 with probability 60% and 0 with probability 40% (expected value 0.6).

| Method | Average Return | Optimal Decision |
|--------|---------------|-----------------|
| CoT | ~0.6 | ✗ (selects B) |
| CoT+U (with uncertainty prompt) | ~0.6 | ✗ |
| DeLLMa | ~0.6 | ✗ |
| RAP | ~0.6 | ✗ (MCTS uses most frequent state, overestimates B) |
| RAP+U | ~0.6 | ✗ |
| Reflexion | ~0.6 | ✗ |
| **PlanU** | **~0.9** | **✓ (correctly learns $\mathbb{E}[Z(s_0,b)]=0.6$, selects A)** |

RAP fails because standard MCTS takes the most frequent next state, always selecting the reward=1 state for B (occurring 60% of the time), resulting in overestimation.

### Experiment 2: Blocksworld Benchmark (Stochastic Environment)

A block stacking task with a 20% action failure rate. Success rates across three ~8B-scale LLMs, categorized by minimum number of steps:

| Model | Method | 2-step | 4-step | 6-step | 8-step |
|-------|--------|--------|--------|--------|--------|
| Mistral-7B | CoT | 0.514 | 0.276 | 0.131 | 0.000 |
| | RAP | 0.892 | 0.514 | 0.166 | 0.000 |
| | RAP-E | 1.000 | 0.592 | 0.338 | 0.084 |
| | **PlanU** | **1.000** | **0.803** | **0.559** | **0.217** |
| LLama3.1-8B | CoT | 0.351 | 0.237 | 0.124 | 0.014 |
| | RAP | 0.946 | 0.553 | 0.255 | 0.175 |
| | RAP-E | 0.946 | 0.763 | 0.414 | 0.140 |
| | **PlanU** | **1.000** | **0.842** | **0.524** | **0.238** |
| DeepSeek-R1-8B | CoT | 0.405 | 0.158 | 0.152 | 0.077 |
| | RAP | 1.000 | 0.724 | 0.200 | 0.196 |
| | RAP-E | 1.000 | 0.697 | 0.448 | 0.175 |
| | **PlanU** | **1.000** | **0.816** | **0.455** | **0.196** |

PlanU achieves the best success rate on nearly all difficulty levels and models, with particularly pronounced advantages on 4-step and 6-step tasks.

### Experiment 3: TravelPlanner & WebShop

| Benchmark | Metric | CoT | RAP | LATS | **PlanU** |
|-----------|--------|-----|-----|------|----------|
| TravelPlanner | Task completion rate | 0.156 | 0.222 | 0.234 | **0.378** |
| TravelPlanner | Constraint satisfaction rate | 0.022 | 0.044 | 0.089 | **0.222** |
| WebShop | Average reward | 0.46 | 0.41 | 0.57 | **0.73** |
| WebShop | Success rate | 0.1 | 0.2 | 0.3 | **0.5** |

PlanU improves task completion rate on TravelPlanner by 61% (vs. LATS) and success rate on WebShop by 67%.

### Ablation Study
- **Removing quantile distribution (PlanU w/o dist)**: Fails to find the optimal path on the Tomato Lettuce Salad task.
- **Removing UCC (PlanU w/o ucc)**: Similarly leads to failure.
- **LLM uncertainty robustness test**: Under Prompt Shuffling and Prompt Injection, PlanU exhibits only a slight decrease in convergence speed, demonstrating strong robustness.

## Highlights & Insights

- **Clear problem formulation**: The paper is the first to systematically distinguish and simultaneously address LLM uncertainty and environmental uncertainty, using a simple stock investment experiment to intuitively expose the fundamental shortcomings of existing methods.
- **Quantile distribution modeling**: Replacing the mean with a quantile distribution to model MCTS node returns captures both the shape of uncertainty (skewed vs. uniform) and enables robust updates via quantile regression.
- **Elegant UCC design**: The integration of RND-inspired novelty rewards with a text encoder to eliminate LLM text uncertainty forms a complete and coherent exploration mechanism.
- **Strong cross-scenario generalization**: PlanU achieves state-of-the-art performance across 5 benchmarks (block stacking, cooking, household, travel planning, online shopping) and 3 LLMs.

## Limitations & Future Work

- **High computational overhead**: Maintaining the quantile distribution, training the RND network, and performing text encoder inference introduce significant overhead compared to standard MCTS; the paper does not report runtime comparisons.
- **Artificially injected environmental uncertainty**: The original Blocksworld environment is deterministic; stochasticity is simulated by adding a fixed failure rate, and the method has not been validated in naturally stochastic environments.
- **Text-only environments**: All environments represent states and actions as text descriptions; visual or continuous state spaces are not addressed.
- **Insufficient hyperparameter sensitivity analysis**: The effects of key hyperparameters such as the number of quantiles $n_q$ and UCC coefficient $c_1$ are not thoroughly ablated.
- **Limited scale**: Experiments are conducted only on 7B–8B scale LLMs; it remains unclear whether larger models still benefit from this framework.
- **No convergence analysis for quantile regression**: Convergence guarantees for the quantile distribution under a limited number of MCTS iterations are not established.

## Related Work & Insights

- **RAP (Hao et al., EMNLP 2023)**: An LLM-MCTS framework that assumes deterministic transitions and handles uncertainty by selecting the most frequent state across multiple queries; this paper demonstrates that such a strategy leads to suboptimal decisions in stochastic environments.
- **LATS (Zhou et al., ICML 2024)**: An LLM-MCTS approach integrating self-reflection and API calls that similarly does not address environmental uncertainty, and underperforms PlanU on both TravelPlanner and WebShop.
- **DeLLMa (Liu et al., ICLR 2025)**: A single-step LLM decision-making method grounded in classical decision theory that is not applicable to multi-step interactive tasks.
- **RAP-D / RAP-E**: Variants of RAP that replace standard MCTS with DMCTS/EMCTS (uncertainty-aware MCTS variants); both outperform RAP but remain inferior to PlanU.
- **QR-DQN (Dabney et al.)**: A quantile distribution method in RL; PlanU draws on its quantile regression ideas and applies them to the LLM-MCTS setting.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of quantile distributions, MCTS, and LLMs is novel, and the UCC design is creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Five benchmarks, three LLMs, and multiple ablations provide broad coverage.
- **Writing Quality**: ⭐⭐⭐⭐ — The stock investment motivating example is intuitive and clear; the overall logic flows smoothly.
- **Value**: ⭐⭐⭐⭐ — Fills a gap in the handling of environmental uncertainty in LLM decision-making, with practical implications for deploying LLM agents in real-world stochastic environments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models](../../ICLR2026/time_series/timeomni-1_incentivizing_complex_reasoning_with_time_series_in_large_language_mo.md)
- [\[NeurIPS 2025\] Diffusion Transformers for Imputation: Statistical Efficiency and Uncertainty Quantification](diffusion_transformers_for_imputation_statistical_efficiency_and_uncertainty_qua.md)
- [\[NeurIPS 2025\] Frequency Matters: When Time Series Foundation Models Fail Under Spectral Shift](frequency_matters_when_time_series_foundation_models_fail_under_spectral_shift.md)
- [\[NeurIPS 2025\] Human-Machine Ritual: Synergic Performance through Real-Time Motion Recognition](human-machine_ritual_synergic_performance_through_real-time_motion_recognition.md)
- [\[NeurIPS 2025\] CausalDynamics: A Large-Scale Benchmark for Structural Discovery of Dynamical Causal Models](causaldynamics_a_large-scale_benchmark_for_structural_discovery_of_dynamical_cau.md)

</div>

<!-- RELATED:END -->
