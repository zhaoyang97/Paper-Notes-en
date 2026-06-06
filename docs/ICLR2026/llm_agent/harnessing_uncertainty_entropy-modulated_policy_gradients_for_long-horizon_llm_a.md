---
title: >-
  [Paper Note] Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents
description: >-
  [ICLR 2026][LLM Agent][policy gradients] This paper proposes EMPG, a framework that dynamically modulates policy gradient magnitudes using step-level entropy (uncertainty) to address the credit assignment problem under s…
tags:
  - "ICLR 2026"
  - "LLM Agent"
  - "policy gradients"
  - "entropy modulation"
  - "long-horizon agents"
  - "credit assignment"
  - "reinforcement learning"
date: 2026-05-08
content_hash: 65fc7a1a8519f8aa
---

# Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents

**Conference**: ICLR 2026
**arXiv**: [2509.09265](https://arxiv.org/abs/2509.09265)  
**Code**: [Project Page](https://empgseed-seed.github.io/)  
**Area**: LLM Agent
**Keywords**: policy gradients, entropy modulation, long-horizon agents, credit assignment, reinforcement learning

## TL;DR

This paper proposes EMPG, a framework that dynamically modulates policy gradient magnitudes using step-level entropy (uncertainty) to address the credit assignment problem under sparse rewards in long-horizon LLM agent tasks. EMPG achieves significant improvements over GRPO and DAPO on three benchmarks: WebShop, ALFWorld, and Deep Search.

## Background & Motivation

In long-horizon tasks such as web navigation, software engineering, and deep search, LLM agents face a fundamental challenge: **credit assignment under sparse rewards**. Feedback is only available after the entire generation concludes, making it difficult to identify which intermediate steps are critical.

Existing approaches follow two main directions:

**Implicit reward shaping**: reward shaping, intrinsic motivation (curiosity/novelty), inverse reinforcement learning, etc.—difficult to scale to the vast state-action space of LLMs.

**Explicit step-level supervision**: Process Reward Models (PRMs)—costly to annotate, noisy when using synthetic data, poor generalization, and defining "correct steps" in interactive agent tasks is itself highly challenging.

The paper's core observation is that **the gradient magnitude of standard policy gradients is inherently coupled with policy entropy**. Specifically, for a softmax policy, the expected norm of the score function is a monotonic function of the policy's Rényi-2 entropy (Proposition 1). This creates a dual problem:
- **Confident and correct steps** should be strongly reinforced, but their natural gradients are small, limiting learning speed.
- **Uncertain exploratory steps** produce large gradients, introducing noise and destabilizing training.

## Method

### Overall Architecture

EMPG (Entropy-Modulated Policy Gradients) is a framework that performs dual recalibration of policy gradients, comprising two complementary components:

1. **Self-Calibrating Gradient Scaling**: recalibrates gradient magnitudes based on the uncertainty of the current step.
2. **Future Clarity Bonus**: encourages the agent to seek predictable solution paths.

### Key Designs

**Step-level uncertainty quantification**: For $m$ tokens within a "thought-action" step, the average token-level entropy is computed as the step-level entropy $H_t$.

**Modulated advantage function**: For step $t$ in a trajectory, the modulated advantage estimate is defined as:
$$A_{\text{mod}}(i,t) = A^{(i)} \cdot g(H_t^{(i)}) + \zeta \cdot f(H_{t+1}^{(i)})$$
where the first term corresponds to self-calibrating gradient scaling and the second to the future clarity bonus.

**Self-calibrating scaling function** $g(H)$ uses an exponential form and is normalized within the mini-batch (mean constrained to 1). Modulation behavior:
- Confident and correct steps (low $H_t$): $g > 1$, amplifying gradients.
- Uncertain steps (high $H_t$): $g < 1$, attenuating gradients.
- Confident but incorrect steps ($A < 0$, low $H_t$): strong penalty signal.

**Future Clarity Bonus** $f(H)$ encourages actions that lead to low-entropy states at the next step.

### Loss & Training

EMPG functions as an advantage modulation module applied directly on top of baselines such as GRPO or DAPO. The agent follows the ReAct paradigm (generating thoughts before actions), and the framework requires no additional value model, operating in a value-free manner.

Normalization proceeds in two steps:
1. **Batch-level entropy normalization**: min-max scaling of step-level entropy.
2. **Final advantage normalization**: zero-mean normalization of $A_{\text{mod}}$ after computation (reducing variance).

## Key Experimental Results

### Main Results

**ALFWorld and WebShop** (Table 1, average success rate %):

| Method | Base Model | ALFWorld All | WebShop Succ. |
|--------|-----------|-------------|---------------|
| GRPO | Qwen2.5-1.5B | 65.6 | 58.2 |
| + EMPG | Qwen2.5-1.5B | **73.7 (+8.1)** | **60.8 (+2.6)** |
| DAPO | Qwen2.5-1.5B | 80.8 | 73.2 |
| + EMPG | Qwen2.5-1.5B | **88.1 (+7.3)** | **73.8 (+0.6)** |
| GRPO | Qwen2.5-7B | 74.8 | 65.6 |
| + EMPG | Qwen2.5-7B | **78.5 (+3.7)** | **69.3 (+3.7)** |
| DAPO | Qwen2.5-7B | 90.0 | 79.6 |
| + EMPG | Qwen2.5-7B | **91.6 (+1.6)** | **82.7 (+3.1)** |

**Deep Search** (Table 2, Qwen2.5-32B-Instruct):

| Method | ID Avg. | OOD Avg. | Overall |
|--------|---------|----------|---------|
| DAPO | 63.5 | 59.8 | 62.0 |
| + EMPG | **66.6 (+3.1)** | **63.7 (+3.9)** | **65.3 (+3.3)** |

The OOD gain (+3.9) exceeds the ID gain (+3.1), indicating improved generalization.

### Ablation Study

Decomposing the two components on Deep Search (Qwen2.5-32B):

| Variant | ID Avg. | OOD Avg. | Overall |
|---------|---------|----------|---------|
| DAPO baseline | 63.5 | 59.8 | 62.0 |
| + Gradient Scaling only | 63.7 | 63.7 (+3.9) | 63.7 |
| + Future Bonus only | 66.1 (+2.6) | 61.4 | 64.2 |
| + EMPG (full) | **66.6** | **63.7** | **65.3** |

### Key Findings

1. **The two components are complementary**: the Future Clarity Bonus primarily improves in-distribution performance (exploitation), while Gradient Scaling primarily improves out-of-distribution generalization (regularization).
2. **Training stability**: the DAPO baseline exhibits severe KL loss oscillations (policy collapse) after approximately 240 steps, whereas EMPG remains stable throughout training.
3. **Step-level vs. token-level**: unlike token-level analysis, even low-entropy steps exhibit substantial entropy variation, validating the necessity of step-level analysis.
4. **Breaking performance plateaus**: while baselines stagnate on ALFWorld and WebShop after reaching a performance ceiling, EMPG continues to improve beyond these limits.

## Highlights & Insights

1. **Theoretically grounded**: the paper formally proves, for the first time, the inherent coupling between policy gradient magnitude and policy entropy (Proposition 1), revealing the root cause of low learning efficiency in long-horizon RL from a gradient dynamics perspective.
2. **Plug-and-play**: as an advantage modulation module, EMPG can be directly stacked on top of any policy gradient method such as GRPO or DAPO.
3. **No additional models required**: EMPG leverages the agent's own policy entropy as an intrinsic signal, eliminating the need for value models or process reward models.
4. **Elegant dual calibration design**: gradient scaling governs *how much* to learn, while the future clarity bonus governs *where* to explore.
5. **Consistently effective across tasks and scales**: stable improvements are observed from 1.5B to 32B models, and across tasks ranging from web navigation to deep search.

## Limitations & Future Work

1. **Coarse entropy estimation**: using the average token-level entropy as a proxy for step-level uncertainty may overlook importance differences among tokens within a step.
2. **Hyperparameter sensitivity**: the scaling factors $k$, $k'$, and $\zeta$ require tuning, and the paper does not adequately discuss sensitivity analysis.
3. **Limited task coverage**: validation is restricted to web navigation, text-based environment interaction, and search tasks; other long-horizon settings such as code generation and mathematical reasoning remain unexplored.
4. **Combination with PRMs**: EMPG and process reward models are orthogonal; future work could explore integrating the two.
5. **Multi-agent scenarios**: the paper mentions but does not empirically validate the effectiveness of EMPG in multi-agent collaborative settings.

## Related Work & Insights

- **GRPO** [Shao et al.]: estimates advantages via within-group Z-score normalization—EMPG further refines this to the step level.
- **DAPO** [Yu et al.]: adaptive data curation—EMPG provides orthogonal improvements at the gradient level.
- **SEED-GRPO** [Chen et al.]: modulates response-level advantages using semantic uncertainty—limited to single-turn reasoning.
- **EDGE-GRPO** [Wang et al.]: performs entropy modulation in mathematical reasoning—limited to single-turn settings and does not address multi-step credit assignment.
- **ReAct** [Yao et al.]: the thought-action paradigm—EMPG treats each ReAct cycle as a single decision step.

## Rating

| Dimension | Score |
|-----------|-------|
| Theoretical Depth | ⭐⭐⭐⭐⭐ |
| Novelty | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents](solving_the_granularity_mismatch_hierarchical_preference_learning_for_long-horiz.md)
- [\[ICLR 2026\] Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization](exploratory_memory-augmented_llm_agent_via_hybrid_on-_and_off-policy_optimizatio.md)
- [\[ICML 2026\] ACON: Optimizing Context Compression for Long-horizon LLM Agents](../../ICML2026/llm_agent/acon_optimizing_context_compression_for_long-horizon_llm_agents.md)
- [\[ICLR 2026\] The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution](the_tool_decathlon_benchmarking_language_agents_for_diverse_realistic_and_long-h.md)
- [\[ACL 2026\] SOLAR-RL: Semi-Online Long-horizon Assignment Reinforcement Learning](../../ACL2026/llm_agent/solar-rl_semi-online_long-horizon_assignment_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
