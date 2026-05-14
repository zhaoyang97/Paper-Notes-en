---
title: >-
  [Paper Note] Why Keep Your Doubts to Yourself? Trading Visual Uncertainties in Multi-Agent Bandit Systems
description: >-
  [ICLR 2026][Multimodal VLM][Multi-agent systems] This paper proposes Agora, a framework that recasts multi-agent VLM coordination as a decentralized uncertainty trading market. Cognitive uncertainty is minted into quanti…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "Multi-agent systems"
  - "VLM coordination"
  - "uncertainty trading"
  - "market mechanism"
  - "Thompson Sampling"
  - "cost optimization"
date: 2026-05-08
content_hash: 641dae832dbdfe8d
---

# Why Keep Your Doubts to Yourself? Trading Visual Uncertainties in Multi-Agent Bandit Systems

**Conference**: ICLR 2026
**arXiv**: [2601.18735](https://arxiv.org/abs/2601.18735)
**Code**: None
**Area**: Multimodal VLM / Multi-Agent Coordination
**Keywords**: Multi-agent systems, VLM coordination, uncertainty trading, market mechanism, Thompson Sampling, cost optimization

## TL;DR

This paper proposes Agora, a framework that recasts multi-agent VLM coordination as a decentralized uncertainty trading market. Cognitive uncertainty is minted into quantifiable, three-dimensional tradable assets (perceptual / semantic / inferential), and efficient equilibrium allocation is achieved through a profit-driven trading protocol and a market-aware Thompson Sampling Broker. Agora consistently outperforms heuristic baselines across five multimodal benchmarks (e.g., +8.5% accuracy on MMMU with more than 3× cost reduction).

## Background & Motivation

**Background**: VLM-driven multi-agent systems (MAS) exhibit strong collective intelligence for visual understanding tasks, yet face an **economic viability crisis** in practical deployment—rapidly escalating operational costs impede large-scale adoption.

**Key Challenge**: Existing paradigms treat intelligence as a "brute-force commodity" rather than a scarce economic resource. When cognitive uncertainty—the primary cost driver—lacks economic discipline, redundant computation proliferates and decision costs become prohibitive.

**Limitations of Prior Work**:
- **Aggregation-based heuristics (MoA)**: Equate statistical consensus with epistemic truth by assuming i.i.d. errors; however, shared architectural biases across VLMs cause correlated errors to be amplified, producing systematic hallucinations.
- **Routing-based heuristics (KABB)**: Rely on proxy scores combining historical performance and semantic similarity, $S = \alpha \cdot P_{\text{hist}} + \beta \cdot \text{Sim}_{\text{sem}}$, conflating past performance with future cost-effectiveness. Such approaches are both **cost-agnostic** (the cost vector $\mathbf{c}$ is absent) and **uncertainty-structure-agnostic** (the vector $\mathbf{u}(t)$ is collapsed to a scalar).

**Theoretical Justification**: The paper formally proves the *Coordination Inefficiency under Ignorance Theorem* (Theorem 1)—any coordination mechanism that is simultaneously cost-agnostic and uncertainty-structure-agnostic is provably suboptimal with respect to the objective $\min_\pi \mathbb{E}[\mathcal{C}(\pi, \mathbf{u}(t), \mathbf{c}, \Xi)]$.

**Core Motivation**: A shift is needed from heuristic patches to a new paradigm that embraces the decentralized nature of the problem—market-mechanism-based coordination that achieves efficient allocation under information asymmetry through price signals and economic incentives.

## Method

### Overall Architecture

Agora redesigns multi-agent coordination as a **decentralized micro-economy** comprising three core modules:

1. **Uncertainty Asset Minting (§3.1)**: Formalizes cognitive uncertainty as structured tradable assets.
2. **Profit-Driven Trading Protocol (§3.2)**: Executes uncertainty trades according to economically rational rules.
3. **Market-Aware Broker (§3.3)**: Handles intelligent initialization and scheduling based on an extended Thompson Sampling formulation.

**System Setup**: $N$ heterogeneous VLM agents $\mathcal{A} = \{a_1, \dots, a_N\}$, each with unit processing cost $c_i > 0$ and expertise vector $\boldsymbol{\xi}_i = [\xi_{i,\text{perc}}, \xi_{i,\text{sem}}, \xi_{i,\text{inf}}]^T$. The optimization objective is:

$$\min_\pi \mathbb{E}_{t \sim \mathcal{T}}[\mathcal{C}(\pi, \mathbf{u}(t), \mathbf{c}, \Xi)] \quad \text{s.t.} \quad \|\mathbf{u}_{\text{final}}\| \leq \epsilon$$

### Key Design 1: Three-Dimensional Uncertainty Asset Model

**Mechanism**: Total uncertainty is decomposed into tradable epistemic uncertainty and non-tradable aleatoric uncertainty. The tradable component is further factored into a three-dimensional vector:

$$\mathbf{u}_{\text{epis}} = [u_{\text{perc}}, u_{\text{sem}}, u_{\text{inf}}]^T$$

- **Perceptual uncertainty** $u_{\text{perc}}$: Uncertainty at the visual perception level (e.g., OCR, object recognition).
- **Semantic uncertainty** $u_{\text{sem}}$: Uncertainty at the semantic understanding level (e.g., contextual inference).
- **Inferential uncertainty** $u_{\text{inf}}$: Uncertainty at the logical reasoning level (e.g., mathematical reasoning, causal inference).

Each agent maintains an uncertainty portfolio:

$$\mathbf{U}(a_i, t) = \mathbf{U}_{\text{base}}(a_i, t) + \sum_{j \neq i} \mathbf{U}_{\text{transfer}}(a_j \to a_i, t)$$

**Why It Works**: Vectorizing uncertainty enables the system to route each uncertainty type to the expert agent best suited to resolve it, avoiding the redundant computation inherent in one-size-fits-all approaches.

### Key Design 2: Profit-Driven Trading Protocol

**Core Mechanism**: Each trade must satisfy economic rationality—execution occurs only when the trade reduces total system cost. The cost change for transferring uncertainty bundle $T_{ij}(t)$ from agent $a_i$ to $a_j$ is:

$$\Delta \mathcal{C}(T_{ij}(t)) = T_{ij}(t) \cdot [c_j(1 - \xi_j) - c_i]$$

**Trade Admission Rule**:

$$\text{Execute trade}(i \to j, T_{ij}(t)) \iff (\Delta \mathcal{C} < 0) \wedge (U_j(t) + T_{ij}(t) \leq C_j(t))$$

A trade is executed only when it is both **profitable** ($\Delta \mathcal{C} < 0$) and **feasible** (the receiving agent has sufficient cognitive capacity). Each valid trade constitutes a greedy descent step on the global cost function.

### Key Design 3: Market-Aware Broker

The Broker selects the initial agent via extended Thompson Sampling, maximizing a market-aware expected utility:

$$\tilde{\theta}_S^{(t)} = (\mathbb{E}[\text{Reward}_S^{(t)}] - \text{Cost}_S^{(t)}) \cdot \exp(-\lambda \cdot \text{Dist}(S, t)) \cdot U_{\text{strategic}}(S)^\omega \cdot \text{Synergy}(S)^\eta \cdot \gamma^{\Delta t}$$

Factor interpretations:
- **Net return**: Expected reward minus cost, ensuring economic rationality.
- **Task distance** $\text{Dist}(S, t)$: Measures alignment between agent and task.
- **Strategic uncertainty** $U_{\text{strategic}}$: Balances exploration and exploitation.
- **Synergy**: Captures complementarity among agent combinations.
- **Temporal decay** $\gamma^{\Delta t}$: Discounts weights over time.

### Agora Algorithm Pipeline

1. **Phase 1 (Initialization)**: The Broker selects the optimal initial agent $a_{\text{handler}}$ and assigns the initial uncertainty bundle.
2. **Phase 2 (Iterative Trading)**: The system repeatedly identifies the most profitable available trade and executes it until no further beneficial trades exist, at which point the market converges to a local optimal equilibrium.

## Key Experimental Results

### Overall Performance on Five Benchmarks

| Model | MMMU(Val) | MMBench_V11 | MathVision | InfoVQA | CC-OCR |
|-------|-----------|-------------|------------|---------|--------|
| qwen2.5vl-72b | 70.2% | 88.4% | 39.3% | 87.3% | 79.8% |
| gemini-2.0-flash | 70.7% | 83.0% | 41.3% | 83.2% | 73.1% |
| qwen2.5vl-7b | 58.6% | 82.6% | 25.1% | 82.6% | 77.8% |
| gpt-4o-mini | 60.0% | 76.3% | 26.3% | 68.7% | 64.2% |
| gemini-2.5-pro | 81.7% | 88.3% | 63.5% | 81.0% | 73.0% |
| InternVL3-78B | 72.2% | 87.7% | 43.1% | 84.1% | 80.3% |
| **Agora (Ours)** | **79.2% (+8.5%)** | **89.5% (+1.1%)** | **44.3% (+3.0%)** | **88.9% (+1.6%)** | **81.2% (+1.4%)** |

Agora achieves state-of-the-art performance on MMBench, InfoVQA, and CC-OCR, and ranks second on MMMU behind the specialized reasoning model gemini-2.5-pro.

### MAB Strategy Ablation (MMMU Val)

| Method | Acc (%) | $U_{\text{final\_epis}}$ ↓ | COI ↓ | UAPS (%) ↑ |
|--------|---------|--------------------------|-------|------------|
| **Agora (Ours)** | **79.0** | **0.15** | **1.2** | **70.5** |
| Agora (No Trading) | 75.5 | 0.22 | 1.0 | 65.0 |
| KABB Selector + Trading | 76.0 | 0.25 | 1.5 | 65.5 |
| PPO Selector + Trading | 74.0 | 0.28 | 1.6 | 62.0 |
| MCTS Selector + Trading | 74.5 | 0.26 | 1.4 | 63.0 |
| DQN Selector + Trading | 73.0 | 0.30 | 1.7 | 60.0 |

### Module Ablation (MMBench_V11)

| Variant | Acc (%) ↑ | $U_{\text{final}}$ ↓ | COI ↓ | UAPS (%) ↑ | Rel. Cost ↓ |
|---------|-----------|---------------------|-------|------------|-------------|
| **Agora (Full)** | **89.50** | **0.16** | **1.25** | **78.33** | 1.00 |
| w/o $U_{\text{strategic}}$ | 86.42 | 0.23 | 1.45 | 71.58 | 1.06 |
| w/o Synergy | 87.91 | 0.19 | 1.30 | 74.88 | 1.03 |
| w/o Dist | 88.53 | 0.18 | 1.27 | 76.21 | 1.01 |
| w/o $\Delta t$ | 89.05 | 0.17 | 1.26 | 77.14 | 1.00 |
| Only Net Return | 82.15 | 0.31 | 1.08 | 60.72 | 0.92 |

Removing the strategic uncertainty factor $U_{\text{strategic}}$ results in the largest performance drop (Acc −3.08%, UAPS −6.75%), confirming its critical role in guiding agent selection.

## Highlights & Insights

1. **Paradigm Innovation via Economics**: Agora is the first framework to model multi-agent VLM coordination as a decentralized uncertainty trading market, systematically addressing the cost–performance trade-off through the lens of microeconomics (market equilibrium, arbitrage, asset trading) rather than naively stacking models.

2. **Solid Theoretical Foundations**: The suboptimality of existing heuristics (MoA, KABB) is formally proven rather than argued by intuition, establishing the necessity of the new framework through rigorous mathematical argumentation.

3. **Significant Cost Efficiency**: At $N=1$, Agora achieves 87.5% accuracy (cost ratio 0.02057), already surpassing expensive SOTA models; at $N=8$, accuracy reaches 89.6% while the cost ratio remains consistently and substantially below all baselines.

4. **Good Scalability**: Performance increases smoothly as agent pool diversity grows, reaching diminishing marginal returns around $N=8$—consistent with economic theory, eliminating the need to scale an expensive agent pool indefinitely.

5. **Effective Uncertainty Decomposition**: The three-dimensional uncertainty decomposition (perceptual / semantic / inferential) enables precise expert routing according to task characteristics, yielding final residual uncertainty significantly lower than KABB (0.16 vs. 0.21).

## Limitations & Future Work

1. **Agent Pool Dependency**: Agora's performance ceiling is bounded by the quality and diversity of the available agent pool; if all models in the pool are weak on a particular task type, the market trading mechanism cannot compensate.

2. **Uncertainty Quantification Assumptions**: The three-dimensional decomposition assumes that perceptual, semantic, and inferential uncertainty can be cleanly separated, whereas these dimensions may be coupled and mutually interacting in practice.

3. **Benchmark Coverage**: Experiments cover only multiple-choice and short-answer visual understanding tasks; effectiveness in open-ended generation, visual dialogue, and similar settings remains unvalidated.

4. **Simplified Cost Model**: The current cost model assumes a fixed unit processing cost per agent, without accounting for latency, concurrency, network overhead, and other practical deployment factors.

5. **Broker Cold-Start**: The exploration–exploitation trade-off in Thompson Sampling relies on accumulated historical interactions, which may result in reduced efficiency during the cold-start phase.

## Related Work & Insights

- **Multi-agent LLM/VLM systems**: MoA (Guo et al., 2024) aggregates multi-model outputs via hierarchical ensembling; KABB (Zhang et al., 2025) routes queries using a knowledge base; FrugalGPT (Chen et al., 2024c) and RouteLLM (Ong et al., 2024) focus on cost-optimized routing at the expense of accuracy.
- **Uncertainty quantification**: Traditional UQ methods (MC Dropout, Deep Ensemble) focus on single-model uncertainty estimation; Agora extends UQ to inter-agent uncertainty trading and propagation.
- **Multi-armed bandits (MAB)**: Thompson Sampling is a classical exploration–exploitation framework; Agora extends it into a market-aware, multi-factor utility function.

## Rating

| Dimension | Score (1–10) |
|-----------|-------------|
| Novelty | 9 |
| Theoretical Depth | 9 |
| Experimental Thoroughness | 8 |
| Value | 7 |
| Writing Quality | 8 |
| **Overall** | **8.5** |

Novelty and theoretical depth are the paper's greatest strengths; redefining multi-agent coordination from an economics perspective represents a genuine paradigm contribution. The experiments span multiple benchmarks with thorough ablations, though validation on open-ended generation tasks is absent. Practical value is constrained by deployment complexity and agent pool dependency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Concept-RuleNet: Grounded Multi-Agent Neurosymbolic Reasoning in Vision Language Models](../../AAAI2026/multimodal_vlm/concept-rulenet_grounded_multi-agent_neurosymbolic_reasoning.md)
- [\[ICLR 2026\] Multimodal Prompt Optimization: Why Not Leverage Multiple Modalities for MLLMs](multimodal_prompt_optimization_why_not_leverage_multiple_modalities_for_mllms.md)
- [\[CVPR 2026\] VS-Bench: Evaluating VLMs for Strategic Abilities in Multi-Agent Environments](../../CVPR2026/multimodal_vlm/vs_bench_evaluating_vlms_for_strategic_abilities_in_multi_agent_environments.md)
- [\[AAAI 2026\] InEx: Hallucination Mitigation via Introspection and Cross-Modal Multi-Agent Collaboration](../../AAAI2026/multimodal_vlm/inex_hallucination_mitigation_via_introspection_and_cross-mo.md)
- [\[CVPR 2026\] The LLM Bottleneck: Why Open-Source Vision LLMs Struggle with Hierarchical Visual Recognition](../../CVPR2026/multimodal_vlm/the_llm_bottleneck_why_open-source_vision_llms_struggle_with_hierarchical_visual.md)

</div>

<!-- RELATED:END -->
