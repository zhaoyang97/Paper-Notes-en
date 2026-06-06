---
title: >-
  [Paper Note] AutoTool: Automatic Scaling of Tool-Use Capabilities in RL via Decoupled Entropy Constraints
description: >-
  [ICLR 2026][Reinforcement Learning][tool use] This paper proposes a reinforcement learning framework with Decoupled Adaptive Entropy Constraints…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "tool use"
  - "test-time scaling"
  - "entropy constraint"
  - "GRPO"
  - "agentic LLM"
date: 2026-05-08
content_hash: 5fa4322063e88c4f
---

# AutoTool: Automatic Scaling of Tool-Use Capabilities in RL via Decoupled Entropy Constraints

**Conference**: ICLR 2026
**arXiv**: [2603.13348](https://arxiv.org/abs/2603.13348)  
**Code**: None  
**Area**: Reinforcement Learning
**Keywords**: tool use, reinforcement learning, test-time scaling, entropy constraint, GRPO, agentic LLM

## TL;DR

This paper proposes a reinforcement learning framework with Decoupled Adaptive Entropy Constraints, enabling LLMs to automatically switch between long and short reasoning modes based on problem difficulty in tool-calling tasks, achieving a 9.8% accuracy improvement while reducing inference token overhead by approximately 81%.

## Background & Motivation

Integrating large language models with external tools is a critical pathway toward AGI. Test-time Scaling (TTS) via reinforcement learning (RL) has achieved notable success in mathematical reasoning, where RL training enables response length to grow in tandem with accuracy. However, the authors identify two core challenges specific to tool-calling tasks:

1. **Reasoning Collapse**: When tool-calling models are trained directly with RL algorithms such as GRPO, response length decreases rather than increases — accuracy improves as training progresses, but response length drops sharply. Models become "lazy" about generating long chain-of-thought reasoning, leading to degraded performance in complex multi-turn tool-calling scenarios.
2. **Overthinking**: Long-reasoning models obtained via distillation generate verbose reasoning traces for all problems regardless of difficulty, resulting in approximately 10× wasted token overhead.

Further analysis reveals that reasoning collapse is strongly correlated with information entropy — the policy model's entropy drops rapidly during training, causing the model to lose its capacity for exploration and default to short reasoning. Direct length penalties fail to alleviate the low-entropy problem, while static entropy constraints are highly sensitive to the coefficient $\beta$.

## Method

### Overall Architecture

AutoTool adopts a two-stage training pipeline: **warm-up SFT → RL with decoupled adaptive entropy constraints**.

### Stage 1: Data Preparation and Warm-up SFT

- **PubTool Dataset**: Integrates three public sources — ToolACE, xLAM, and Hermes Function-Calling — yielding 8.2k SFT samples and 7k RL samples after downsampling and quality filtering.
- **Mixed Long/Short Reasoning Data**: Training data is annotated via pass@8 inference using both a no-thinking model (Qwen2.5-7B-Instruct) and a thinking model (Qwen3-32B). If the no-thinking model answers correctly, the short reasoning trace is used as the label; otherwise, the thinking model's long reasoning trace is adopted.
- **RL Data Quality Optimization**: Overly easy or difficult samples are removed to balance the distribution; high-quality samples aligned with the model's learning trajectory are selected based on reward variance across multiple GRPO training rounds.
- Warm-up SFT allows the model to develop an initial awareness of data difficulty and learn to distinguish problems requiring long versus short reasoning.

### Stage 2: RL with Decoupled Adaptive Entropy Constraints

The core contribution is incorporating a **decoupled entropy regularization term** into the GRPO policy loss:

$$\beta_i = \beta_s \cdot m_i \cdot \mathbb{I}\{H_i \leq H_s\} + \beta_l \cdot (1-m_i) \cdot \mathbb{I}\{H_i \leq H_l\}$$

where $m_i \in \{0,1\}$ indicates whether the current trajectory is short reasoning ($m_i=1$) or long reasoning ($m_i=0$):

- **Short reasoning $\beta_s$**: A fixed coefficient that prevents excessive exploration and maintains concise responses.
- **Long reasoning $\beta_l$**: An adaptive coefficient dynamically adjusted via an auxiliary loss.

The adaptive entropy coefficient loss is defined as:

$$\mathcal{L}_{\beta}^l = \frac{1}{\sum_j(1-m_j)} \sum_{i=1}^{N} (1-m_i) \cdot \beta_l \cdot (H_i - H_l)$$

When $H_i < H_l$, $\beta_l$ increases to encourage exploration; when $H_i > H_l$, $\beta_l$ decreases to suppress excessive randomness.

### Auto-Thinking Reward Module

An asymmetric reward mechanism is designed to incentivize a balance between efficiency and accuracy:

| Scenario | Reward |
|----------|--------|
| Correct + no-think | +1.0 |
| Correct + think | +0.5 |
| Incorrect + think | -0.5 |
| Incorrect + no-think | -1.0 |

Correctly answering simple problems with short reasoning yields higher rewards; failing on complex problems encourages switching to long reasoning. At inference time, reasoning modes can be controlled via prefix tokens.

## Key Experimental Results

### Benchmarks and Setup

- Base model: Qwen2.5-7B-Instruct
- Evaluation benchmarks: BFCL (Non-Live / Live / Multi-Turn), API-Bank (L-1 / L-2), ACEBench

### Main Results (BFCL)

| Model | Non-Live | Live | Multi-Turn | Overall |
|-------|----------|------|------------|---------|
| Qwen2.5-7B-Instruct | 86.46 | 67.44 | 7.62 | 53.69 |
| PubTool-SFT | 88.98 | 77.28 | 9.68 | 58.17 |
| PubTool-Distilled | 87.73 | 78.64 | 15.65 | 60.30 |
| Qwen3-8B | 88.81 | 78.54 | 33.00 | 66.34 |
| **AutoTool-7B (auto)** | **89.76** | **80.22** | **38.18** | **70.12** |

- Improvement of +11.95% over PubTool-SFT and +16.43% over the base model.
- The most significant gains are observed in the Multi-Turn setting (+28.5% vs. SFT).
- Performance is comparable to frontier models such as GPT-4o (70.42) and o3 (70.32).

### Inference Efficiency Analysis

- AutoTool requires an average of approximately 183 tokens, compared to ~966 tokens for the distilled model, representing an **81% reduction in token cost**.
- The thinking rate is 45% in Multi-Turn scenarios and 0% in simple Non-Live scenarios, demonstrating genuine difficulty-adaptive behavior.
- Under forced no-think mode, Accuracy per Computation Unit (ACU) reaches an optimal value of 0.97.

### Ablation Study

| Variant | Overall Change |
|---------|---------------|
| Full model | 70.12 |
| w/o data refine | −6.43 |
| w/o decouple | −5.89 |
| w/o adapt coeff | −2.34 |

Data quality filtering has the largest impact, followed by the decoupled design. The adaptive coefficient contributes significantly to Multi-Turn stability, with its removal causing a 10.53% drop in Multi-Turn performance.

## Highlights & Insights

1. **In-depth problem analysis**: The paper systematically analyzes the relationship between reasoning collapse and entropy, demonstrating that data distribution is not the root cause — information entropy is the key factor.
2. **Well-motivated method design**: Decoupling entropy constraints for long and short reasoning avoids conflicting optimization objectives, and the adaptive coefficient eliminates the need for sensitive manual tuning.
3. **High practical value**: A 7B model achieves GPT-4o-level tool-calling performance while substantially reducing inference costs.
4. **Controllable reasoning modes**: At inference time, think / no-think / auto modes can be flexibly switched via prefix tokens.
5. **Elegant reward design**: The asymmetric reward naturally guides the model to use short reasoning for simple problems and long reasoning for complex ones.

## Limitations & Future Work

1. Validation is limited to the 7B scale; whether reasoning collapse occurs similarly in larger models and whether the method remains effective has not been investigated.
2. The PubTool dataset is relatively small (7k RL samples); scalability under larger-scale training has not been verified.
3. Warm-up SFT relies on distillation data from Qwen3-32B, requiring a powerful teacher model.
4. The distinction between long and short reasoning depends on the presence of think tokens, offering limited control over finer-grained reasoning depth.
5. Evaluation focuses primarily on function-calling tools; generalization to broader tool-use scenarios such as code execution and web browsing has not been tested.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Entropy-Preserving Reinforcement Learning (REPO / ADAPO)](entropy-preserving_reinforcement_learning.md)
- [\[ICLR 2026\] AutoQD: Automatic Discovery of Diverse Behaviors with Quality-Diversity Optimization](autoqd_automatic_discovery_of_diverse_behaviors_with_quality-diversity_optimizat.md)
- [\[ICLR 2026\] P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling](p-genrm_personalized_generative_reward_model_with_test-time_user-based_scaling.md)
- [\[ICML 2026\] From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents](../../ICML2026/reinforcement_learning/from_self-evolving_synthetic_data_to_verifiable-reward_rl_post-training_multi-tu.md)
- [\[ICLR 2026\] Exploration vs Exploitation: Rethinking RLVR through Clipping, Entropy, and Spurious Reward](exploration_vs_exploitation_rethinking_rlvr_through_clipping_entropy_and_spuriou.md)

</div>

<!-- RELATED:END -->
