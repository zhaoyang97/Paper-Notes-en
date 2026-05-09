---
title: >-
  [Paper Note] Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?
description: >-
  [NeurIPS 2025][LLM Agent][Multi-agent debate] This work establishes, both theoretically and empirically, that the performance gains attributed to Multi-Agent Debate (MAD) stem primarily from majority voting (ensembling) rather than the debate process itself. The debate dynamics are shown to constitute a martingale—meaning debate does not systematically improve correctness in expectation—and this theoretical insight motivates a principled improvement to MAD by biasing updates toward correct signals.
tags:
  - NeurIPS 2025
  - LLM Agent
  - Multi-agent debate
  - majority voting
  - martingale
  - Bayesian belief update
  - LLM ensembling
date: 2026-05-08
content_hash: 6853da273fb80521
---

# Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?

**Conference**: NeurIPS 2025
**arXiv**: [2508.17536](https://arxiv.org/abs/2508.17536)
**Code**: [https://github.com/deeplearning-wisc/debate-or-vote](https://github.com/deeplearning-wisc/debate-or-vote)
**Area**: LLM Agent
**Keywords**: Multi-agent debate, majority voting, martingale, Bayesian belief update, LLM ensembling

## TL;DR
This work establishes, both theoretically and empirically, that the performance gains attributed to Multi-Agent Debate (MAD) stem primarily from majority voting (ensembling) rather than the debate process itself. The debate dynamics are shown to constitute a martingale—meaning debate does not systematically improve correctness in expectation—and this theoretical insight motivates a principled improvement to MAD by biasing updates toward correct signals.

## Background & Motivation

**Background**: Multi-Agent Debate (MAD) has emerged as a popular paradigm for enhancing LLM reasoning, in which multiple LLMs collaborate through structured discussion. Numerous variants have been proposed in recent years, covering diverse topological structures (decentralized, sparse, centralized) and role-assignment schemes.

**Limitations of Prior Work**: MAD frameworks have grown increasingly complex, yet the root source of their effectiveness remains unclear—specifically, whether performance gains arise from the *ensemble effect* of multiple agents or from *inter-agent interaction and discussion*.

**Key Challenge**: If most gains derive from simple ensembling (majority voting), then the additional complexity of debate, communication, and architectural design in MAD is unwarranted.

**Goal**: To rigorously disentangle the contributions of "ensembling" and "debate" in MAD, and to provide a theoretical explanation for why debate alone cannot systematically improve performance.

**Key Insight**: MAD is formalized as a Bayesian belief update process, with agent behavior characterized by a Dirichlet-Compound-Multinomial (DCM) model, under which the debate process is proven to form a martingale.

**Core Idea**: Debate does not improve correctness in expectation (martingale property); the majority of performance gains originate from the ensemble effect of majority voting.

## Method

### Overall Architecture
MAD is decomposed into two components: (1) *Multi-Agent* (ensemble of multiple agents → majority voting) and (2) *Debate* (iterative inter-agent communication → belief update). The individual contributions are isolated by comparing "voting without debate" against "voting after debate."

### Key Designs

1. **DCM Generative Model**:

    - *Function*: Formalizes the answer generation process of each agent.
    - *Mechanism*: At round $t$, each agent $i$ holds a belief vector $\boldsymbol{\alpha}_{i,t}$ (Dirichlet parameters), samples $\boldsymbol{\theta}_{i,t} \sim \text{Dirichlet}(\boldsymbol{\alpha}_{i,t})$, and generates an answer from $\text{Categorical}(\boldsymbol{\theta}_{i,t})$. The marginal probability is $P(y_{i,t}=k) = \alpha_{i,t}^{(k)} / \sum_j \alpha_{i,t}^{(j)}$.
    - *Design Motivation*: The DCM naturally captures both the internal uncertainty of LLMs (Dirichlet prior) and output stochasticity (sampling), while its Bayesian conjugacy facilitates theoretical analysis.

2. **Majority Voting Success Probability (Theorem 1)**:

    - *Core Result*: Even when the correct answer holds only a marginal advantage ($\theta_1 \ll 1/2$), the probability of majority voting yielding the correct answer approaches 1 as the number of agents $N$ grows.
    - *Lower Bound*: $\mathbb{P}(y_{mv}=1) \geq 1 - \exp(-N(\Delta/\sqrt{K} - 1/\sqrt{N})^2)$
    - *Key Insight*: Voting alone exhibits an "amplification effect" that does not require debate.

3. **Martingale Theorem (Theorem 2 — Core Contribution)**:

    - *Function*: Proves that each agent's belief in the correct answer constitutes a martingale throughout the debate process.
    - *Core Result*: $\mathbb{E}[p_{i,t} | \boldsymbol{\alpha}_{t-1}] = p_{i,t-1}$, i.e., debate does not alter the expected belief in the correct answer.
    - *Precondition*: The average belief of neighbors equals the agent's own belief, which is naturally satisfied under homogeneous agents with fully connected topology.
    - *Deeper Implication*: Each round of debate constitutes a random walk—agents are sometimes corrected (beneficial) and sometimes misled (harmful), with these effects **canceling out in expectation**. This explains why MAD often fails to outperform majority voting by a significant margin.

4. **MAD-oracle Improvement Strategy**:

    - *Function*: Breaks the zero-drift property of the martingale by biasing updates toward correct signals.
    - *Mechanism*: Once an agent produces the correct answer, its state is "locked" and shielded from further debate influence (oracle variant, requires ground truth).
    - *Practical Variant*: MAD-confidence—high-confidence answers are less susceptible to modification.
    - *Design Motivation*: The martingale result implies that a deliberate "correct-direction drift" must be introduced for debate to be useful.

## Key Experimental Results

### Main Results (Qwen2.5-7B-Instruct, 5 agents)

| Method | Arithmetics | GSM8K | MMLU PM | MMLU FL | HellaSwag | CSQA | HH-RLHF | Avg |
|--------|------------|-------|---------|---------|-----------|------|---------|-----|
| Single-Agent | 0.814 | 0.871 | 0.787 | 0.491 | 0.788 | 0.815 | 0.477 | 0.721 |
| Decentralized MAD (T=2) | 0.760 | 0.887 | 0.805 | 0.556 | 0.803 | 0.857 | 0.497 | 0.738 |
| Decentralized MAD (T=5) | 0.670 | 0.833 | 0.805 | 0.476 | 0.800 | 0.843 | 0.507 | 0.705 |
| **Majority Voting** | **0.990** | **0.940** | 0.794 | **0.540** | 0.803 | 0.830 | 0.487 | **0.769** |

### Ablation Study

| Observation | Finding |
|-------------|---------|
| Agent count 1→5 | Consistent performance improvement, primarily attributable to ensembling |
| Debate rounds T=2→5 | Performance degrades in some settings (notably Arithmetics) |
| Centralized MAD | Substantially underperforms majority voting (single-judge bottleneck) |
| Martingale validation | Average agent accuracy remains approximately constant across debate rounds (empirically confirms theoretical prediction) |

### Key Findings
- **Majority voting matches or surpasses MAD in most settings**—particularly on Arithmetics (0.99 vs. 0.67–0.84).
- Increasing debate rounds is not consistently beneficial; at T=5, performance drops significantly on several benchmarks, as correct agents are misled through excessive debate.
- Centralized MAD performs worst, with the central node acting as a performance bottleneck.
- MAD-oracle yields substantial improvements, validating the upper bound of the "bias toward correct signals" strategy.
- Findings remain consistent when scaled to 32B models and heterogeneous agent configurations.

## Highlights & Insights
- **Martingale Theoretical Framework**: Formalizing the MAD debate process as a martingale constitutes the first rigorous theoretical analysis of the effectiveness mechanism underlying MAD. The framework is both elegant and explanatory—the zero-drift property of debate precisely accounts for why more rounds of debate do not translate to better outcomes.
- **Counterintuitive "Simplicity Wins" Conclusion**: In multi-agent LLM systems, a simple voting strategy captures the majority of available gains; complex debate architectures may represent over-engineering.
- **Theory-Guided Design**: The martingale analysis directly motivates the actionable design principle of "biasing toward correct signals," demonstrating a tight loop from theoretical analysis to practical recommendations.

## Limitations & Future Work
- The DCM model may only approximate LLM behavior imperfectly—real LLM belief updates do not fully conform to Bayesian conjugacy.
- Experiments are conducted primarily with 7B/8B models; debate dynamics in larger models may differ (though 32B experiments suggest consistent conclusions).
- MAD-oracle requires ground truth labels; the practical confidence-based variant yields limited gains.
- The analysis focuses primarily on homogeneous agent settings (single model for all agents), with heterogeneous experiments treated as supplementary.
- The applicability of the martingale framework to open-ended generation tasks requires further investigation.

## Related Work & Insights
- **vs. BCCS**: BCCS attempts to improve consensus through belief calibration, while this work theoretically demonstrates that debate itself does not improve expected outcomes—the two findings are complementary.
- **vs. DyLAN**: DyLAN dynamically selects agents for debate, but the present work suggests that if the underlying dynamics constitute a martingale, the gains from dynamic selection may be limited.
- **vs. Self-Consistency (Wang et al.)**: Self-Consistency is equivalent to majority voting over a single agent's outputs; this work generalizes that paradigm to multi-agent settings and provides its theoretical foundation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The martingale analysis framework is an entirely novel theoretical contribution with high academic value owing to its counterintuitive conclusion.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 7 benchmarks, multiple MAD variants, and multiple model scales with thorough ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are clear, with tight correspondence between theory and experiments.
- Value: ⭐⭐⭐⭐⭐ Provides foundational guidance for the design of multi-agent LLM systems.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Are Large Language Models Sensitive to the Motives Behind Communication?](are_large_language_models_sensitive_to_the_motives_behind_communication.md)
- [\[NeurIPS 2025\] AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks](agenttts_large_language_model_agent_for_testtime_computeopti.md)
- [\[AAAI 2026\] MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models](../../AAAI2026/llm_agent/medla_a_logic-driven_multi-agent_framework_for_complex_medic.md)
- [\[NeurIPS 2025\] Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning](zero-shot_large_language_model_agents_for_fully_automated_radiotherapy_treatment.md)
- [\[ACL 2026\] Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models](../../ACL2026/llm_agent/agent-gwo_collaborative_agents_for_dynamic_prompt_optimization_in_large_language.md)

<!-- RELATED:END -->
