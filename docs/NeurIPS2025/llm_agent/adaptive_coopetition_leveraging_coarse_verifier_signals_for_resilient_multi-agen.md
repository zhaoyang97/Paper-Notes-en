---
title: >-
  [Paper Note] Adaptive Coopetition: Leveraging Coarse Verifier Signals for Resilient Multi-Agent LLM Reasoning
description: >-
  [NeurIPS 2025 (Mathematical Reasoning and AI Workshop)][LLM Agent][multi-agent collaboration] This paper proposes the Adaptive Coopetition (AdCo) framework, which employs a UCB multi-armed bandit strategy with coarse-grained verifier signals to enable multiple LLM agents to adaptively switch between cooperative and competitive modes during inference, achieving a 20% relative improvement on mathematical reasoning benchmarks.
tags:
  - NeurIPS 2025 (Mathematical Reasoning and AI Workshop)
  - LLM Agent
  - multi-agent collaboration
  - reasoning enhancement
  - UCB
  - coopetition mechanism
  - inference-time computation
date: 2026-05-08
content_hash: 5b3e474c1763eebf
---

# Adaptive Coopetition: Leveraging Coarse Verifier Signals for Resilient Multi-Agent LLM Reasoning

**Conference**: NeurIPS 2025 (Mathematical Reasoning and AI Workshop)  
**arXiv**: [2510.18179](https://arxiv.org/abs/2510.18179)  
**Code**: [GitHub](https://github.com/AdCo-Research/adaptive-coopetition)  
**Area**: Multi-Agent Systems / LLM Reasoning  
**Keywords**: multi-agent collaboration, reasoning enhancement, UCB, coopetition mechanism, inference-time computation

## TL;DR

This paper proposes the Adaptive Coopetition (AdCo) framework, which employs a UCB multi-armed bandit strategy with coarse-grained verifier signals to enable multiple LLM agents to adaptively switch between cooperative and competitive modes during inference, achieving a 20% relative improvement on mathematical reasoning benchmarks.

## Background & Motivation

Inference-time computation is a key paradigm for enhancing LLM reasoning capabilities; however, existing approaches exhibit significant limitations:

**Limitations of Self-Correction**: Self-correction in LLMs tends to reinforce the model's initial biases and fails to effectively rectify fundamental reasoning errors.

**Failures of Multi-Agent Collaboration (MAC)**: Existing multi-agent methods lack efficient coordination mechanisms, making them prone to collective errors—all agents may converge to the same incorrect answer.

**High Bar for Reliable Verifiers**: Although external verifiers can detect reasoning errors, training reliable verifiers itself demands substantial resources.

A key observation is that **collaboration is not always optimal**. When multiple agents have comparable capabilities, pure collaboration may lead to groupthink, whereas moderate competition can encourage exploration of the solution space. Conversely, when one agent is clearly superior, competition wastes resources.

## Method

### Overall Architecture

AdCo is a multi-round, multi-agent reasoning framework with the following core pipeline:

1. **Initialization**: Multiple LLM agents independently generate initial answers and reasoning chains.
2. **Signal Collection**: A coarse-grained verifier (e.g., a PRM) scores each reasoning chain.
3. **Strategy Selection**: The UCB algorithm determines whether the current round adopts a collaborative or competitive mode.
4. **Reasoning Update**: Each agent updates its reasoning based on the selected mode and peer feedback.
5. **Iteration** until convergence or a maximum number of rounds is reached.

### Key Designs

**UCB-Based Strategy Selection Mechanism**

Drawing on the Upper Confidence Bound (UCB) algorithm from the multi-armed bandit (MAB) literature to balance cooperation and competition:

$$UCB_i = \bar{X}_i + c\sqrt{\frac{\ln N}{n_i}}$$

where $\bar{X}_i$ is the historical average reward of strategy $i$ (collaborative or competitive), $N$ is the total number of rounds, $n_i$ is the number of times strategy $i$ has been selected, and $c$ is the exploration parameter.

**Collaborative Mode**
- Agents share reasoning chains and intermediate results with one another.
- Each agent integrates high-quality reasoning steps from peers.
- Suited for scenarios where one agent is clearly superior.

**Competitive Mode**
- Agents independently refine their reasoning without referencing others' results.
- Comparison and voting occur only at the level of final answers.
- Suited for scenarios where agent capabilities are comparable and diverse solutions need to be explored.

**Coarse-Grained Verifier Signals**
- High-precision step-level verifiers are not required.
- Only coarse-grained "reasoning quality signals" (e.g., overall PRM scores for a reasoning chain) are needed.
- This substantially reduces dependence on verifier quality.

### Loss & Training

AdCo is a **training-free** inference-time framework that requires no additional model training. The core mechanism is online learning of optimal strategy assignments via the UCB algorithm:

- After each round, the reward estimates for each strategy are updated based on verifier signals.
- UCB naturally balances exploitation and exploration.
- As the number of rounds increases, strategy selection converges toward the optimum.

The default agent pool consists of GPT-4o, DeepSeek-R1, and Qwen-QWQ-32B, forming a diverse multi-agent ensemble.

## Key Experimental Results

### Main Results

Accuracy comparison on GSM8K and MATH:

| Method | GSM8K Acc (%) | MATH Acc (%) | Relative Gain (MATH) |
|--------|--------------|-------------|----------------------|
| Single Agent (GPT-4o) | 82.5 | 51.2 | Baseline |
| Self-correction | 83.1 | 52.4 | +2.3% |
| MAC (pure collaboration) | 85.3 | 54.8 | +7.0% |
| MAC (pure competition) | 84.7 | 55.1 | +7.6% |
| Majority voting | 86.2 | 56.3 | +10.0% |
| AdCo (UCB adaptive) | **88.4** | **61.5** | **+20.1%** |

### Ablation Study

Performance comparison under different strategy configurations:

| Strategy | GSM8K Acc (%) | MATH Acc (%) | Strategy Diversity |
|----------|--------------|-------------|-------------------|
| Fixed collaboration | 85.3 | 54.8 | Low |
| Fixed competition | 84.7 | 55.1 | High |
| Random switching | 85.9 | 56.7 | Medium |
| UCB (w/o PRM) | 86.1 | 57.3 | Medium |
| UCB + coarse PRM | **88.4** | **61.5** | Adaptive |

Effect of varying agent count:

| No. of Agents | MATH Acc (%) | Inference Overhead (relative) |
|--------------|-------------|-------------------------------|
| 2 | 57.8 | 1.0x |
| 3 | 61.5 | 1.5x |
| 5 | 62.3 | 2.5x |
| 7 | 62.1 | 3.5x |

### Key Findings

1. **Adaptive strategy significantly outperforms fixed strategies**: The UCB adaptive method surpasses the best fixed strategy by approximately 6 percentage points on MATH.
2. **Coarse-grained signals are sufficient**: Precise step-level verifiers are unnecessary; coarse-grained PRM signals alone can effectively guide strategy selection.
3. **Three agents offer the best trade-off**: Performance saturates at three agents, with diminishing returns beyond this point.
4. **More pronounced gains on harder datasets**: The relative improvement on MATH (~20%) substantially exceeds that on GSM8K (~7%).
5. **Strong robustness**: Performance variance across configurations is small, indicating that the UCB mechanism effectively adapts to diverse scenarios.

## Highlights & Insights

1. **Novel coopetition concept**: Introducing the coopetition strategy from game theory into multi-agent reasoning provides a fresh perspective on inference-time computation.
2. **Elegant application of UCB**: Formulating strategy selection as a multi-armed bandit problem leverages the well-established UCB algorithm to naturally resolve the exploration–exploitation trade-off.
3. **Low verifier requirements**: The method does not rely on high-performance verifiers, lowering the barrier to practical adoption.
4. **Plug-and-play**: As an inference-time framework, it requires no modification to the underlying models, offering broad generalizability.

## Limitations & Future Work

1. **Validation limited to mathematical reasoning**: The framework has not yet been evaluated on code generation, logical reasoning, commonsense reasoning, or other tasks.
2. **High API call costs**: Multi-agent, multi-round inference entails a large volume of API calls, making real-world deployment expensive.
3. **Limited strategy space**: Only two strategies—collaborative and competitive—are considered; finer-grained hybrid strategies could be explored.
4. **Workshop paper limitations**: Some experimental details and depth of analysis are limited, such as the distribution of strategy selections across problems of varying difficulty.
5. **Impact of PRM signal quality**: Although the paper claims independence from high-performance verifiers, the specific effect of PRM signal quality on performance is not fully quantified.

## Related Work & Insights

- **Self-Consistency (Wang et al.)**: Samples multiple reasoning chains and selects the answer via majority voting.
- **Tree of Thoughts (Yao et al.)**: Conducts structured search over the reasoning space.
- **Multi-Agent Debate (Du et al.)**: Employs debate-style reasoning among multiple agents.
- **Process Reward Model**: Verifies the correctness of reasoning step by step.
- **Insights**: The coopetition mechanism can be generalized to other scenarios requiring a balance between exploration and exploitation, such as code debugging and creative writing.

## Rating

- Novelty: ⭐⭐⭐⭐ (novel coopetition concept, elegant UCB application)
- Technical Depth: ⭐⭐⭐ (core technique is relatively straightforward, yet the combination is effective)
- Experimental Thoroughness: ⭐⭐⭐⭐ (multi-dimensional comparisons, including ablation and robustness analysis)
- Writing Quality: ⭐⭐⭐⭐ (clear motivation, rich figures and tables)
- Overall: ⭐⭐⭐⭐ (a valuable directional contribution with a concise and effective methodology)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MAT-Agent: Adaptive Multi-Agent Training Optimization](mat-agent_adaptive_multi-agent_training_optimization.md)
- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](../../ACL2026/llm_agent/atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)
- [\[ICLR 2026\] HAMLET: A Hierarchical and Adaptive Multi-Agent Framework for Live Embodied Theatre](../../ICLR2026/llm_agent/hamlet_a_hierarchical_and_adaptive_multi-agent_framework_for_live_embodied_theat.md)
- [\[NeurIPS 2025\] Lessons Learned: A Multi-Agent Framework for Code LLMs to Learn and Improve](lessons_learned_a_multi-agent_framework_for_code_llms_to_learn_and_improve.md)
- [\[ACL 2026\] EA-Agent: A Structured Multi-Step Reasoning Agent for Entity Alignment](../../ACL2026/llm_agent/ea-agent_a_structured_multi-step_reasoning_agent_for_entity_alignment.md)

</div>

<!-- RELATED:END -->
