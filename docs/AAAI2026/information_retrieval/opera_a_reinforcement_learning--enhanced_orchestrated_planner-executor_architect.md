---
title: >-
  [Paper Note] OPERA: A Reinforcement Learning--Enhanced Orchestrated Planner-Executor Architecture for Reasoning-Oriented Multi-Hop Retrieval
description: >-
  [AAAI2026][Information Retrieval & RAG][RAG] This paper proposes OPERA, a hierarchical framework comprising a Goal Planning Module and a Reason-Execute Module…
tags:
  - "AAAI2026"
  - "Information Retrieval & RAG"
  - "RAG"
  - "multi-hop retrieval"
  - "reinforcement-learning"
  - "GRPO"
  - "multi-agent"
date: 2026-05-08
content_hash: 71984a33a729a572
---

# OPERA: A Reinforcement Learning--Enhanced Orchestrated Planner-Executor Architecture for Reasoning-Oriented Multi-Hop Retrieval

**Conference**: AAAI2026
**arXiv**: [2508.16438](https://arxiv.org/abs/2508.16438)  
**Code**: [Ameame1/OPERA](https://github.com/Ameame1/OPERA)  
**Area**: Information Retrieval
**Keywords**: RAG, multi-hop retrieval, reinforcement-learning, GRPO, multi-agent

## TL;DR
This paper proposes OPERA, a hierarchical framework comprising a Goal Planning Module and a Reason-Execute Module, combined with MAPGRPO—a training algorithm specifically designed for multi-agent settings—to substantially improve performance on reasoning-oriented multi-hop retrieval tasks.

## Background & Motivation

### State of the Field

**Background**: Existing RAG systems perform poorly on complex multi-hop questions, with the primary bottleneck being the weak coupling between retrieval and reasoning.

### Limitations of Prior Work

**Limitations of Prior Work**: Static planning approaches (e.g., PlanRAG) cannot dynamically adapt to new information encountered during the retrieval process.

### Root Cause

**Key Challenge**: Assigning both planning and execution responsibilities to a single LLM limits overall reliability.

### Solution Direction

**Solution Direction**: Existing RL-based methods (e.g., BGM) only optimize the gap between the retriever and the LLM, without achieving fine-grained credit assignment at the agent level.

### Paper Goals

**Goal**: How can retrieval and reasoning be deeply coupled within a RAG framework, enabling effective plan decomposition, adaptive retrieval, and precise filtering for complex multi-hop questions?

## Method

### Overall Architecture
OPERA is decoupled into two layers:
1. **Goal Planning Module (GPM)**: A Plan Agent decomposes complex questions into sub-goals $\mathcal{P}=\{p_1,\dots,p_m\}$, with dependencies among sub-goals established via placeholders.
2. **Reason-Execute Module (REM)**: An Analysis-Answer Agent performs information sufficiency judgment $\phi \in \{0,1\}$ and answer extraction; a Rewrite Agent rewrites queries when information is insufficient to improve subsequent retrieval.
3. **Trajectory Memory Component (TMC)**: Records all operational trajectories to enhance interpretability.

### Key Designs: MAPGRPO
GRPO is extended into Multi-Agents Progressive Group Relative Policy Optimization:
- Three agents are trained sequentially, each using a heterogeneous reward function.
- Plan Agent reward: $r_{\text{plan}} = \lambda_1 f_{\text{logic}} + \lambda_2 f_{\text{struct}} + \lambda_3 f_{\text{exec}}$
- Analysis-Answer Agent reward: $r_{\text{ana}} = \alpha \cdot \mathbb{I}[\phi=\phi^*] + \beta \cdot \text{EM}(a_i,a_i^*) + \gamma \cdot f_{\text{format}}$
- Rewrite Agent reward: $r_{\text{rew}} = \omega_1 \sqrt{\text{NDCG@}k} + \omega_2 f_{\text{format}}$
- **High-Score Sample Selection**: A pre-scored high-quality sample $c_{\text{best}}$ is injected into each group to mitigate reward sparsity and reduce policy gradient variance.

## Key Experimental Results

### Main Results

| Method | HotpotQA EM | 2WikiMHQA EM | Musique EM |
|--------|------------|-------------|------------|
| Adaptive-RAG (SFT) | 45.7% | 30.1% | 24.3% |
| BGM (RL) | 41.5% | 44.3% | 19.6% |
| **OPERA (MAPGRPO)** | **57.3%** | **60.2%** | **39.7%** |

- Relative improvement over the best baseline on Musique: 63.4%.
- Ablation: removing the Plan Agent causes EM to drop from 39.7% to 17.1%, falling below the training-free CoT baseline of 21.2%.
- Out-of-domain: on NQ (single-hop), MAPGRPO achieves 36.6% EM, whereas SFT degrades to 19.5%.

## Highlights & Insights
- **Architecture design outweighs training methodology**: the collapse in performance upon removing the Plan Agent confirms that the hierarchical architecture is the core contribution.
- The expert injection strategy in MAPGRPO significantly reduces policy gradient variance.
- The framework generalizes to out-of-domain tasks (single-hop QA), demonstrating that RL training does not overfit to fixed reasoning patterns.

## Limitations & Future Work
- Training of the Rewrite Agent is unstable due to reward sparsity caused by conditional activation.
- Theoretical guarantees cover only local convergence; Musique EM remains below 40%.
- Inference latency is relatively high, with notable fluctuations in Analysis-Answer Agent latency.
- Effectiveness on larger LLMs (>7B parameters) has not yet been validated.
- The framework still struggles with ambiguous decompositions and long reasoning chains.

## Related Work & Insights

| Dimension | OPERA | Adaptive-RAG | ReAct | BGM |
|-----------|-------|-------------|-------|-----|
| Planning | Dynamic sub-goals | Complexity-based routing | No explicit plan | None |
| Retrieval | Rewrite Agent adaptive | Fixed strategy | Reasoning-action loop | RL bridging |
| Training | MAPGRPO (role-specific) | SFT | No training | GRPO |
| Interpretability | TMC trajectory logging | None | Partial | None |

## Related Work & Insights
- A multi-agent hierarchical architecture with role-specific rewards represents a promising design paradigm for RAG systems.
- The sequential training and high-score sample injection strategy of MAPGRPO can be generalized to other multi-agent RL scenarios.
- The TMC design philosophy can be leveraged to improve the auditability and debuggability of AI systems.
- The finding that "architectural contributions outweigh training contributions" merits consideration in the design of other complex systems.
- The conditional activation mechanism of the Rewrite Agent embodies the principle of on-demand allocation of computational resources.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The multi-agent architecture and MAPGRPO training method are original contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluated on three mainstream benchmarks with ablation and OOD experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with reasonably complete theoretical analysis.
- **Value**: ⭐⭐⭐⭐ — Provides practical guidance for the design of complex RAG systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PRIME: Planning and Retrieval-Integrated Memory for Enhanced Reasoning](prime_planning_and_retrieval-integrated_memory_for_enhanced_reasoning.md)
- [\[ACL 2026\] Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning](../../ACL2026/information_retrieval/agentic_conversational_search_with_contextualized_reasoning_via_reinforcement_le.md)
- [\[ACL 2026\] ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering](../../ACL2026/information_retrieval/chatr1_reinforcement_learning_for_conversational_reasoning_and_retrieval_augment.md)
- [\[ACL 2026\] Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation](../../ACL2026/information_retrieval/learning_to_extract_rational_evidence_via_reinforcement_learning_for_retrieval-a.md)
- [\[ACL 2026\] Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation](../../ACL2026/information_retrieval/language-coupled_reinforcement_learning_for_multilingual_retrieval-augmented_gen.md)

</div>

<!-- RELATED:END -->
