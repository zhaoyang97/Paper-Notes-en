---
title: >-
  [Paper Note] From Debate to Equilibrium: Belief-Driven Multi-Agent LLM Reasoning via Bayesian Nash Equilibrium
description: >-
  [ICML 2025][Multi-Agent][Multi-Agent LLM] This paper models multi-LLM coordination as an incomplete information game and proposes the ECON framework. It achieves implicit belief-driven multi-agent coordinated reasoning via Bayesian Nash Equilibrium (BNE) without explicit message passing while providing theoretical convergence guarantees, yielding an average improvement of 11.2% across six reasoning benchmarks.
tags:
  - "ICML 2025"
  - "Multi-Agent"
  - "Multi-Agent LLM"
  - "Bayesian Nash Equilibrium"
  - "Reinforcement Learning"
  - "Belief Coordination"
  - "Scalable Reasoning"
date: 2026-05-08
content_hash: 2d8e525c9ff56d42
---

# From Debate to Equilibrium: Belief-Driven Multi-Agent LLM Reasoning via Bayesian Nash Equilibrium

**Conference**: ICML 2025

**arXiv**: [2506.08292](https://arxiv.org/abs/2506.08292)

**Authors**: Xie Yi, Zhanke Zhou, Chentao Cao, Qiyu Niu, Tongliang Liu, Bo Han (TMLR Group)

**Area**: LLM Agent

**Keywords**: Multi-Agent LLM, Bayesian Nash Equilibrium, Reinforcement Learning, Belief Coordination, Scalable Reasoning

**Code**: [GitHub](https://github.com/tmlr-group/ECON)

---

## TL;DR

This paper models multi-LLM coordination as an incomplete information game and proposes the ECON framework. It achieves implicit belief-driven multi-agent coordinated reasoning via Bayesian Nash Equilibrium (BNE) without explicit message passing while providing theoretical convergence guarantees, yielding an average improvement of 11.2% across six reasoning benchmarks.

## Background & Motivation

While multi-agent LLM frameworks (such as Multi-Agent Debate, MAD) have proven effective in enhancing reasoning capabilities, existing methods suffer from three fundamental limitations:

**Excessive communication overhead**: Traditional multi-round debates require agents to explicitly pass messages, causing token consumption and computational overhead to scale linearly with the number of rounds.

**Lack of convergence guarantees**: Prior methods lack theoretical guarantees that the debate will converge to a correct or consistent answer, potentially leading to infinite loops.

**Poor scalability**: Information exchange between agents can easily exceed the context length limits of LLMs, causing performance to degrade as the number of agents increases.

Key Insight: Instead of having agents "converse" directly (explicit communication), it is more effective to let each agent independently make an optimal response based on its **probabilistic beliefs** about other agents' policies. This formulation directly aligns with the concept of Bayesian Nash Equilibrium (BNE) in game theory.

## Method

### Overall Architecture — ECON

ECON (Efficient Coordination via Nash Equilibrium) adopts a hierarchical architecture:

- **Coordinator LLM**: Generates strategic instructions ($\le$ 50 tokens) without directly revealing answers, responsible for final commitment.
- **Executor LLMs**: Multiple agents process problems independently, generating answers based on the Coordinator's strategy and their own beliefs.
- **BeliefNetwork**: Manages the belief state of each agent and computes Q-values.
- **BeliefEncoder**: Aggregates group representations using an attention mechanism.
- **Mixer**: An attention-based agent interaction layer that aggregates local Q-values and incorporates commitment alignment and consistency regularization.

### Game-Theoretic Modeling

Multi-LLM coordination is formulated as an incomplete information game $\Gamma = (N, \{A_i\}, \{\Theta_i\}, \{u_i\}, p)$:

- $N$ LLM agents, where each agent $i$ has an action space $A_i$ (i.e., potential answers).
- Type space $\Theta_i$ represents the private information of an agent (e.g., model capability, context understanding).
- Utility function $u_i$ measures the quality of the answer.
- Prior distribution $p$ describes the belief about other agents' types.

### Bayesian Nash Equilibrium (BNE)

In a BNE, the strategic policy $\sigma_i^*$ of each agent satisfies:

$$\sigma_i^*(\theta_i) = \arg\max_{a_i \in A_i} \mathbb{E}_{\theta_{-i} \sim p(\cdot|\theta_i)} \left[ u_i(a_i, \sigma_{-i}^*(\theta_{-i}), \theta_i) \right]$$

Namely, given its own type and beliefs about other agents' strategies, each agent selects the action that maximizes its expected utility.

### Two-Stage BNE Coordination

**Stage 1 — Individual Belief Formation**:

- Each Executor independently forms a belief state $b_i$ and generates an initial answer.
- Belief states are maintained and updated via the BeliefNetwork.

**Stage 2 — BNE Iterative Coordination**:

- Agents iteratively update beliefs through equilibrium computation until convergence.
- The Coordinator generates a commitment, prompting termination when the commitment remains unchanged across consecutive rounds or parameter variance falls below a threshold.

### Reward System

Three reward components are dynamically combined via learnable weights $\alpha$:

$$R = \alpha_1 \cdot R_{\text{TS}} + \alpha_2 \cdot R_{\text{AL}} + \alpha_3 \cdot R_{\text{CC}}$$

| Reward Component | Definition | Computation Method |
|----------|------|----------|
| $R_{\text{TS}}$ (Task-Specific) | Task correctness | Numerical matching with the ground truth (binary) |
| $R_{\text{AL}}$ (Action Likelihood) | Action-commitment alignment | Cosine similarity between the embeddings of the Executor's output and the Coordinator's commitment |
| $R_{\text{CC}}$ (Collaborative Contribution) | Collaborative contribution | Faithfulness (consistency with the commitment) + novelty (dissimilarity to peers' answers) |

The embedding model used is `BAAI/bge-large-en-v1.5`.

### Loss & Training

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{TD}} + \mathcal{L}_{\text{mixer}} + \mathcal{L}_{\text{BNE}}$$

- $\mathcal{L}_{\text{TD}}$: TD error of local Q-values
- $\mathcal{L}_{\text{mixer}}$: Global TD + consistency loss + commitment alignment loss
- $\mathcal{L}_{\text{BNE}}$: Equilibrium loss + commitment improvement term

The target networks are updated softly: $\phi' \leftarrow \tau \phi + (1-\tau)\phi'$ with $\tau=0.01$.

### Theoretical Analysis — Regret Bound

The paper theoretically proves that ECON's regret bound is significantly tighter than non-equilibrium multi-agent frameworks. Letting $T$ denote the total number of interaction rounds, the regret bound of ECON is:

$$\text{Regret}(T) = \widetilde{O}(\sqrt{T})$$

In contrast, the regret bound of traditional MAD methods is $O(T^{2/3})$ or looser. This theoretically explains the superior sample efficiency of ECON.

## Key Experimental Results

### Main Results — Six Reasoning and Planning Benchmarks

| Method | MATH | GSM8K | SVAMP | StrategyQA | ARC-C | CSQA | Average |
|------|:----:|:-----:|:-----:|:----------:|:-----:|:----:|:----:|
| Single LLM | 51.2 | 74.8 | 76.5 | 71.3 | 78.2 | 68.9 | 70.2 |
| Self-Consistency | 55.8 | 78.3 | 79.4 | 73.1 | 80.5 | 71.2 | 73.1 |
| MAD (3 agents) | 56.4 | 79.1 | 80.2 | 74.5 | 81.3 | 72.8 | 74.1 |
| MAD (5 agents) | 57.1 | 79.8 | 80.9 | 74.2 | 81.0 | 72.5 | 74.3 |
| **ECON (3 agents)** | **63.2** | **85.6** | **86.3** | **80.1** | **87.4** | **78.9** | **80.3** |
| ECON Gain over MAD | +6.8 | +6.5 | +6.1 | +5.6 | +6.1 | +6.1 | +6.2 |

ECON consistently and significantly outperforms Multi-Agent Debate (MAD) across all benchmarks, with an average improvement of **11.2%** over the Single LLM baseline.

### Scalability Study

| No. of Agents | MAD Accuracy | ECON Accuracy | MAD Token Consumption | ECON Token Consumption |
|:----------:|:----------:|:-----------:|:--------------:|:---------------:|
| 3 | 74.1 | 80.3 | 12.5K | 4.2K |
| 5 | 74.3 | 82.1 | 28.7K | 6.8K |
| 8 | 73.8 | 83.5 | 52.1K | 10.3K |

Key Findings:
- MAD exhibits performance degradation when the number of agents increases to 8 (73.8 < 74.3) due to context overflow.
- ECON consistently improves while token consumption scales only linearly (as it bypasses explicit inter-agent communication).

### Ablation Study

| Configuration | MATH | GSM8K | Average |
|------|:----:|:-----:|:----:|
| Full ECON | 63.2 | 85.6 | 80.3 |
| W/o BNE Coordination | 57.8 | 80.1 | 74.9 |
| W/o Coordinator | 59.1 | 81.3 | 76.2 |
| Fixed α Weights (non-learnable) | 61.5 | 83.8 | 78.4 |
| W/o $R_{\text{CC}}$ | 61.8 | 84.0 | 78.6 |

## Highlights & Insights

1. **Deep Integration of Game Theory and LLMs**: This work is the first to rigorously model multi-agent LLM reasoning as an incomplete information game and solve the BNE, establishing a solid theoretical foundation for multi-agent systems.
2. **Implicit Beliefs over Explicit Communication**: Eliminating message passing between agents effectively reduces token consumption from $O(n^2)$ to $O(n)$.
3. **Consistency Between Theory and Practice**: The theoretical improvement in the regret bound is empirically validated — ECON's sample efficiency and final performance are both significantly superior to MAD.
4. **Flexible Integration of New Models**: New agents can be seamlessly integrated by training their belief networks independently, without requiring a complete system retrain.

## Limitations & Future Work

1. The theoretical analysis relies on strong assumptions (such as finite agent type spaces and smooth utility functions) which may not fully hold for practical LLMs.
2. The belief network and mixer introduce additional parameters and training overhead, which is less straightforward than simple majority voting.
3. Current experiments predominantly employ open-source models (such as Llama-3.3-70B), leaving the efficacy on closed-source counterparts (e.g., GPT-4) unexplored.
4. The embedding alignment (AL/CC) in the reward function relies on external embedding models, increasing overall system complexity.

## Related Work & Insights

- **Multi-Agent Debate (MAD)**: Multi-round explicit debates among agents incur high token costs and lack convergence guarantees. ECON presents a qualitative leap forward over this paradigm.
- **Self-Consistency / CoT-SC**: Relies on multi-sample voting within a single model and lacks cross-model coordination.
- **LLM-Blender / FrugalGPT**: Focuses on routing or blending multiple LLMs without integrating game-theoretic modeling.

Insight: The BNE framework can be extended to other scenarios requiring multi-agent coordination (e.g., code generation, multimodal reasoning). The belief-driven implicit coordination paradigm warrants deeper exploration.

## Rating

| Dimension | Score (1-5) | Description |
|------|:----------:|------|
| Novelty | 5 | A unique hybrid of game theory, RL, and multi-agent LLMs with solid theoretical novelty. |
| Experimental Thoroughness | 4 | Solid evaluation across six benchmarks, scalability, and ablation studies, though comparisons with additional baselines are lacking. |
| Value | 3 | Relative complexity in training; practical engineering deployment requires careful trade-offs. |
| Writing Quality | 4 | Clear formulation in the theoretical sections, though dense notation requires careful reading. |
| **Total Score** | **4.0** | An innovative study with notable theoretical contributions. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Belief-Calibrated Multi-Agent Consensus Seeking for Complex NLP Tasks](../../NeurIPS2025/multi_agent/belief-calibrated_multi-agent_consensus_seeking_for_complex_nlp_tasks.md)
- [\[ICLR 2026\] From EduVisBench to EduVisAgent: A Benchmark and Multi-Agent Framework for Reasoning-Driven Pedagogical Visualization](../../ICLR2026/multi_agent/from_eduvisbench_to_eduvisagent_a_benchmark_and_multi-agent_framework_for_reason.md)
- [\[ICLR 2026\] MAD-Logic: Multi-Agent Debate Enhances Symbolic Translation and Reasoning](../../ICLR2026/multi_agent/mad-logic_multi-agent_debate_enhances_symbolic_translation_and_reasoning.md)
- [\[ICLR 2026\] DoVer: Intervention-Driven Auto Debugging for LLM Multi-Agent Systems](../../ICLR2026/multi_agent/dover_intervention-driven_auto_debugging_for_llm_multi-agent_systems.md)
- [\[ACL 2026\] When Identity Skews Debate: Anonymization for Bias-Reduced Multi-Agent Reasoning](../../ACL2026/multi_agent/when_identity_skews_debate_anonymization_for_bias-reduced_multi-agent_reasoning.md)

</div>

<!-- RELATED:END -->
