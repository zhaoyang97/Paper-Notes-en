---
title: >-
  [Paper Note] Why Keep Your Doubts to Yourself? Trading Visual Uncertainties in Multi-Agent Bandit Systems
description: >-
  [ICLR 2026][Multimodal VLM][multi-agent systems] This paper proposes Agora, a framework that recasts multi-agent VLM coordination as a decentralized uncertainty trading market. By decomposing epistemic uncertainty into tradable assets along three dimensions (perceptual / semantic / reasoning) and employing a profitability-driven trading protocol together with Thompson Sampling brokers, Agora achieves cost-aware optimal allocation, yielding up to +8.5% accuracy improvement with over 3× cost reduction across five multimodal benchmarks.
tags:
  - ICLR 2026
  - Multimodal VLM
  - multi-agent systems
  - VLM coordination
  - uncertainty quantification
  - market mechanism
  - Thompson Sampling
date: 2026-05-08
content_hash: cc3ff25b5d9fe4b0
---

# Why Keep Your Doubts to Yourself? Trading Visual Uncertainties in Multi-Agent Bandit Systems

**Conference**: ICLR 2026
**arXiv**: [2601.18735](https://arxiv.org/abs/2601.18735)
**Code**: None
**Area**: Multimodal VLM
**Keywords**: multi-agent systems, VLM coordination, uncertainty quantification, market mechanism, Thompson Sampling

## TL;DR

This paper proposes Agora, a framework that recasts multi-agent VLM coordination as a decentralized uncertainty trading market. By decomposing epistemic uncertainty into tradable assets along three dimensions (perceptual / semantic / reasoning) and employing a profitability-driven trading protocol together with Thompson Sampling brokers, Agora achieves cost-aware optimal allocation, yielding up to +8.5% accuracy improvement with over 3× cost reduction across five multimodal benchmarks.

## Background & Motivation

1. **Runaway costs in VLM multi-agent systems**: As VLMs scale, the operational cost of coordinating heterogeneous agents grows sharply, making economic viability a deployment bottleneck and shifting the need from brute-force compute stacking to fine-grained resource management.
2. **Existing aggregation strategies (MoA) rest on a false independence assumption**: Mixture-of-Agents relies on consensus voting, but shared architectural biases induce highly correlated errors, causing consensus to amplify systematic hallucinations with no convergence guarantee to correct answers.
3. **Existing routing strategies (KABB) ignore cost and uncertainty structure**: Knowledge-aware routers select agents based on historical performance and semantic similarity scores, with neither a cost term nor an uncertainty vector in the scoring function—a dual blind spot of cost-agnosticism and structure-agnosticism.
4. **Provable suboptimality**: The paper formally defines *Agnostic Coordination* and proves that any coordination mechanism satisfying both cost-agnosticism and structure-agnosticism is necessarily suboptimal when the best-performing agent is not the cheapest (Theorem 1).
5. **Economic challenges of information asymmetry and bounded rationality**: Multi-agent systems are inherently decentralized economic problems in which each agent holds private information and heterogeneous capabilities, requiring mechanism design to elicit private information and guide global optimality.
6. **Absence of an uncertainty-trading paradigm**: Prior work treats uncertainty as a scalar or a monolithic burden and never decomposes it into structured, priceable, tradable assets for fine-grained management—this decomposition is the central innovation of Agora.

## Method

### Overall Architecture: Agora Decentralized Uncertainty Trading Market

- **Function**: Reframes multi-agent VLM coordination as a microeconomic market in which uncertainty serves as a tradable asset, agents act as trading parties, and a Broker mediates the market.
- **Design Motivation**: The coordination problem is directly mapped to an economic optimization objective (Eq. 1)—minimizing total cost $\mathcal{C}$ subject to the constraint that residual uncertainty satisfies $\|\mathbf{u}_{\text{final}}\| \leq \epsilon$—eliminating the theoretical blind spots introduced by heuristic proxies.
- **Mechanism**: A three-stage pipeline: (1) *mint* query uncertainty into a three-dimensional tradable asset; (2) the Broker uses market-aware Thompson Sampling to select the initial processing agent; (3) iterate profitable trades until market equilibrium is reached.

### Key Design 1: Three-Dimensional Uncertainty Assetization

- **Function**: Decomposes total uncertainty $\mathbf{u}$ into epistemic uncertainty (tradable) and aleatoric uncertainty (non-tradable); the epistemic component is further split into a three-dimensional vector: perceptual $u_{\text{perc}}$, semantic $u_{\text{sem}}$, and reasoning $u_{\text{inf}}$.
- **Design Motivation**: Vectorization enables independent pricing and trading of each uncertainty type, resolving the structure-agnosticism problem. Each dimension corresponds to a distinct cognitive capability; some agents excel at perception but not reasoning, so fine-grained allocation reduces cost.
- **Mechanism**: Each agent $a_i$ maintains an uncertainty portfolio $\mathbf{U}(a_i, t)$ formed by linearly combining its own baseline uncertainty with uncertainty acquired through trades from other agents (Eq. 3), aggregated via weighted historical transaction records.

### Key Design 2: Profitability-Driven Trading Protocol

- **Function**: Defines a profitability condition for trades; a transfer of uncertainty bundles is executed only when it reduces total system cost.
- **Design Motivation**: Directly eliminates cost-agnosticism—the trade admission rule explicitly includes the cost vector $\mathbf{c}$ and the expertise matrix $\Xi$, thereby violating the suboptimality conditions of Theorem 1.
- **Mechanism**: Computes the cost delta $\Delta\mathcal{C}(T_{ij}) = T_{ij} \cdot [c_j(1 - \xi_j) - c_i]$ (Eq. 4); a trade is executed only when $\Delta\mathcal{C} < 0$ and the receiving agent has residual capacity (Eq. 5). Each trade constitutes a greedy descent step on the global cost function.

### Key Design 3: Market-Aware Broker

- **Function**: A Thompson Sampling–based intelligent broker that selects, for each query, the initial processing agent with the highest economic utility.
- **Design Motivation**: The trading protocol performs local greedy optimization; a well-chosen initial assignment can substantially shorten the convergence path and reduce total cost.
- **Mechanism**: The Broker computes a multi-factor utility function $\tilde{\theta}_S^{(t)}$ (Eq. 6) that jointly considers expected reward minus cost, task distance decay, strategic utility, agent synergy, and temporal decay, and applies Thompson Sampling to balance exploration and exploitation.

## Key Experimental Results

### Main Results: Five-Benchmark Comprehensive Performance (Table 1)

| Model | MMMU | MMBench | MathVision | InfoVQA | CC-OCR |
|---|---|---|---|---|---|
| qwen2.5vl-72b | 70.2% | 88.4% | 39.3% | 87.3% | 79.8% |
| gemini-2.0-flash | 70.7% | 83.0% | 41.3% | 83.2% | 73.1% |
| gemini-2.5-pro | 81.7% | 88.3% | 63.5% | 81.0% | 73.0% |
| InternVL3-78B | 72.2% | 87.7% | 43.1% | 84.1% | 80.3% |
| **Agora** | **79.2%**(+8.5) | **89.5%**(+1.1) | **44.3%**(+2.0) | **88.9%**(+1.6) | **81.2%**(+1.4) |

**Key Findings**: Using a pool of five small-to-medium VLMs, Agora surpasses all individual models—including gemini-2.5-pro—on MMBench, InfoVQA, and CC-OCR. The largest gain, +8.5% on MMMU, is the highest observed. Agora falls short of the specialized reasoning model gemini-2.5-pro only on MathVision (63.5% vs. 44.3%), yet still outperforms every single model in the pool.

### Routing and Multi-Agent Strategy Comparison (Figure 4)

| Method | MMBench Acc. | Relative Cost | Final Epistemic Uncertainty |
|---|---|---|---|
| **Agora** | **89.50%** | 1.00 | **0.16** |
| KABB-VLM | 87.12% | 1.24× | 0.21 |
| MOA | 86.65% | 3.11× | 0.25 |
| FrugalGPT | 81.50% | 0.73× | 0.27 |
| RouteLLM | 80.85% | 0.91× | 0.30 |

**Key Findings**: Agora achieves the highest accuracy at the lowest cost (normalized to 1.0). KABB and MOA incur 24% and 211% higher costs, respectively, at comparable accuracy levels. Low-cost routing methods (FrugalGPT / RouteLLM) are cheaper but drop accuracy by 8–9 points with higher residual uncertainty, confirming Agora's advantage on the Pareto frontier.

### Ablation Study: MAB Strategy Ablation (Table 2)

| Selection Strategy | MMMU Acc. | Final Uncertainty↓ | UAPS↑ |
|---|---|---|---|
| **Agora (MAB)** | **79.0%** | **0.15** | **70.5%** |
| KABB + Trading | 76.0% | 0.25 | 65.5% |
| PPO + Trading | 74.0% | 0.28 | 62.0% |
| DQN + Trading | 73.0% | 0.30 | 60.0% |
| No Trading | 75.5% | 0.22 | 65.0% |

**Key Findings**: The MAB Broker outperforms the best heuristic (KABB) by 3.0% and RL-based methods (PPO / DQN / A2C / MCTS) by 5–6%, demonstrating the superiority of the economic utility function design over both pure reinforcement learning and heuristics. Using only the Broker for initial model selection without trading already achieves 75.5%; the trading mechanism contributes an additional 3.5%.

## Highlights & Insights

- **Theory-driven paradigm innovation**: The paper formalizes multi-agent coordination through an economic lens, proves the theoretical suboptimality of existing methods, and then designs non-agnostic mechanisms to overcome it.
- **Uncertainty assetization**: This is the first work to decompose epistemic uncertainty into three-dimensional tradable assets, transforming the vague notion of "confidence" into a quantifiable, priceable, and tradable economic object.
- **Pareto-optimal cost-efficiency**: Accuracy improvements and cost reductions are achieved simultaneously across five heterogeneous benchmarks, most notably +8.5% on MMMU with over 3× cost savings.
- **Algorithmic simplicity**: The core trading rule (Eq. 5) requires only two conditional checks, yielding an efficient and interpretable coordination mechanism.

## Limitations & Future Work

- The trading protocol performs greedy descent and converges to a local rather than global optimum; when agent pool heterogeneity is insufficient, the tradable space is limited.
- The quantification of the three uncertainty dimensions (perceptual / semantic / reasoning) relies on prompt engineering and heuristics, lacking automated uncertainty estimation.
- All models in the experiments are accessed via the OpenRouter API, so the cost model depends on API pricing; cost structures in self-hosted deployment scenarios may differ substantially.
- Evaluation is restricted to visual understanding tasks in multiple-choice and short-answer formats; applicability to open-ended generation or video understanding has not been verified.

## Related Work & Insights

| Dimension | Agora (Ours) | MoA (Mixture-of-Agents) | KABB-VLM |
|---|---|---|---|
| Coordination mechanism | Decentralized market trading | Centralized aggregation voting | Heuristic routing score |
| Cost modeling | Explicit cost term, profitability-driven trading | No cost awareness | No cost awareness |
| Uncertainty handling | Three-dimensional vector, tradable | Scalar consensus | Scalar semantic similarity |
| MMBench Acc. | 89.50% | 86.65% | 87.12% |
| Relative cost | 1.00× | 3.11× | 1.24× |

## Rating

- ⭐⭐⭐⭐ **Novelty**: Introducing an economic market mechanism into multi-agent VLM coordination yields original contributions in both theory and methodology.
- ⭐⭐⭐⭐ **Experimental Thoroughness**: Five benchmarks, diverse baselines (routing / MAS / RL / ablation), and a comprehensive cost–performance Pareto analysis.
- ⭐⭐⭐ **Writing Quality**: The extensive use of economic concepts and notation, combined with a lengthy paper body, raises the reading barrier.
- ⭐⭐⭐⭐ **Value**: Provides a deployable cost-optimization solution for multi-agent VLM systems, particularly practical in API-based settings.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Reasoning-Driven Multimodal LLM for Domain Generalization](reasoning-driven_multimodal_llm_for_domain_generalization.md)
- [\[ICLR 2026\] Post-hoc Probabilistic Vision-Language Models](post-hoc_probabilistic_vision-language_models.md)
- [\[ICLR 2026\] Multimodal Prompt Optimization: Why Not Leverage Multiple Modalities for MLLMs](multimodal_prompt_optimization_why_not_leverage_multiple_modalities_for_mllms.md)
- [\[ICLR 2026\] Why Reinforcement Fine-Tuning Preserves Prior Knowledge Better: A Data Perspective](why_reinforcement_fine-tuning_enables_mllms_preserve_prior_knowledge_better_a_da.md)
- [\[ICLR 2026\] Enhancing Multi-Image Understanding through Delimiter Token Scaling](enhancing_multi-image_understanding_through_delimiter_token_scaling.md)

<!-- RELATED:END -->
