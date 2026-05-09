---
title: >-
  [Paper Note] HiFo-Prompt: Prompting with Hindsight and Foresight for LLM-based Automatic Heuristic Design
description: >-
  [ICLR 2026][Model Compression][Automatic Heuristic Design] This paper proposes the HiFo-Prompt framework, which enhances LLM-driven Automatic Heuristic Design (AHD) through two collaborative modules—Hindsight (a retrospective insight pool) and Foresight (a prospective evolutionary navigator)—achieving substantial improvements over existing methods on tasks such as TSP and FSSP.
tags:
  - ICLR 2026
  - Model Compression
  - Automatic Heuristic Design
  - LLM + Evolutionary Computation
  - Knowledge Management
  - Exploration-Exploitation Balance
  - Combinatorial Optimization
date: 2026-05-08
content_hash: adf5df35128391d2
---

# HiFo-Prompt: Prompting with Hindsight and Foresight for LLM-based Automatic Heuristic Design

**Conference**: ICLR 2026
**arXiv**: [2508.13333](https://arxiv.org/abs/2508.13333)
**Code**: [GitHub](https://github.com/Challenger-XJTU/HiFo-Prompt)
**Area**: Model Compression
**Keywords**: Automatic Heuristic Design, LLM + Evolutionary Computation, Knowledge Management, Exploration-Exploitation Balance, Combinatorial Optimization

## TL;DR
This paper proposes the HiFo-Prompt framework, which enhances LLM-driven Automatic Heuristic Design (AHD) through two collaborative modules—Hindsight (a retrospective insight pool) and Foresight (a prospective evolutionary navigator)—achieving substantial improvements over existing methods on tasks such as TSP and FSSP.

## Background & Motivation
The paradigm of combining LLMs with Evolutionary Computation (EC)—exemplified by FunSearch and EoH—has demonstrated the potential of using LLMs as high-level semantic variation operators for automatically designing heuristic algorithms. However, existing approaches face two fundamental challenges:

**Lack of global adaptive guidance**: Existing methods largely rely on local or reactive signals—ReEvo reflects on individual candidates in isolation, while MCTS-AHD passively embeds the exploration-exploitation trade-off into its search structure. Such local control mechanisms cannot proactively intervene in systemic issues such as population stagnation or diversity collapse. Alternative approaches such as EvoTune directly fine-tune LLM weights, but at prohibitive computational cost and with limited interpretability of the acquired knowledge.

**Knowledge Decay**: Successful design strategies tend to be entangled with specific code implementations, so when parent individuals are eliminated, the underlying design logic is lost along with them. The system cannot achieve cumulative learning and repeatedly rediscovers similar concepts from scratch.

**Core Idea**: To elevate the LLM from a "code generator" to a "symbolic meta-optimizer" endowed with hierarchical control capabilities—Foresight observes population dynamics to guide macro-level strategy, while Hindsight distills reusable design principles from elite individuals.

## Method

### Overall Architecture
HiFo-Prompt constructs the LLM prompt for each generation via Guided Prompt Synthesis, integrating three components: a base prompt strategy (equivalent to genetic operators), the Hindsight module (historical knowledge injection), and the Foresight module (current strategy instructions).

### Key Designs

1. **Hindsight Module: Self-Evolving Insight Pool**

   - **Function**: Distills abstract design principles (insights) from successful heuristic code and maintains a persistent knowledge repository.
   - **Mechanism**: Comprises three stages—(1) *Insight Extraction and Admission*: at the end of each generation, design principles are extracted from elite individuals and deduplicated using a Jaccard similarity threshold $\theta_{\text{novelty}}$; (2) *Insight Retrieval and Credit Assignment*: the top-$s$ insights with the highest utility are selected for prompt injection, with a utility function that balances effectiveness, usage penalty, and recency bonus: $U(k_i, t) = E_i(t) - w_u \log(N_i(t)+1) + B_r(t, t_i^{\text{last}})$; (3) *Adaptive Pruning*: when pool capacity is exceeded, insights with the lowest eviction scores are discarded.
   - Credit assignment employs a piecewise function that maps relative population performance to a credit signal: $g_{\text{eff}} = 0.8 + 0.2\tilde{\rho}$ when surpassing the best known, $0.2 + 0.6\tilde{\rho}$ when above the mean, and $-0.3 + 0.5\tilde{\rho}$ when below the mean; utility scores are updated via EMA.
   - **Design Motivation**: Converts transient evolutionary successes into reusable knowledge assets, thereby addressing the knowledge decay problem.

2. **Foresight Module: Evolutionary Navigator**

   - **Function**: Monitors population dynamics in real time and switches among three modes—exploration, exploitation, and balance.
   - **Mechanism**: Maintains two mutually exclusive counters, $C_{\text{prog}}$ (progress) and $C_{\text{stag}}$ (stagnation), to track performance trends, while computing phenotypic diversity $\Delta_p(t)$—the proportion of non-duplicate pairs among all algorithm description texts in the population. A threshold-based rule selects the evolutionary regime: $\theta_{\text{explore}}$ (when stagnation is detected or diversity is low), $\theta_{\text{exploit}}$ (when sustained progress is observed), and $\theta_{\text{balance}}$ (otherwise).
   - Diversity measurement uses exact string matching rather than embedding-based similarity, avoiding the risk that subtle logical differences are smoothed away by semantic representations.
   - **Design Motivation**: Serves as a symbolic surrogate for "linguistic gradients," providing an interpretable global control strategy.

3. **Base Prompt Strategies**

   - **Function**: Provides LLM-equivalent genetic operators.
   - Includes an initialization strategy I1, recombination strategies (E1 synthesizes new structures from multiple parents; E2 abstracts commonalities to produce variants), and mutation strategies (M1 for structural modification, M2 for parameter tuning, M3 for simplification to prevent overfitting).

### Loss & Training
Population size is set to 8; combinatorial optimization tasks run for 8 generations and Bayesian optimization tasks for 4 generations. The backbone LLM is Qwen2.5-Max. The Insight Pool has a capacity of 30, Jaccard threshold of 0.7, top-3 retrieval, EMA rate of 0.3, stagnation threshold of 3, progress threshold of 2, and diversity threshold of 0.3.

## Key Experimental Results

### Main Results: TSP Step-by-step Construction

| Method | TSP50 Gap (%) | TSP100 Gap (%) | TSP200 Gap (%) |
|--------|--------------|----------------|----------------|
| LKH3 | 0.000 | 0.000 | 0.000 |
| EoH | 12.820 | 15.361 | 16.658 |
| ReEvo | 10.239 | 12.577 | 14.890 |
| MCTS-AHD | 10.642 | 12.521 | 13.510 |
| **HiFo-Prompt** | **6.625** | **8.582** | **8.877** |

### TSP Guided Local Search

| Method | TSP100 Gap (%) | TSP200 Gap (%) | TSP500 Gap (%) |
|--------|--------------|----------------|----------------|
| EoH | 0.026 | 0.453 | 2.037 |
| ReEvo | 0.049 | 0.424 | 2.090 |
| **HiFo-Prompt** | **—** | **—** | **—** |

### Key Findings
- On TSP step-by-step construction, HiFo-Prompt reduces the optimality gap from ~13% to ~8%, representing a relative improvement of approximately 40%.
- The credit assignment mechanism of the Insight Pool effectively guides knowledge evolution, preventing low-utility insights from persistently occupying resources.
- The adaptive strategy switching in Foresight is critical for avoiding premature convergence.
- The framework demonstrates competitive performance and reliability on Bayesian Optimization tasks.

## Highlights & Insights
- The concept of "decoupling code from thought" is notably novel—updating and evaluating insights independently from code evolution substantially reduces evaluation cost.
- The lifecycle management of the Insight Pool (extraction → retrieval → credit assignment → pruning) is elegantly designed, drawing an analogy to sparse reward handling in reinforcement learning.
- Explicit control over exploration and exploitation is realized through natural-language "design instructions," serving as a symbolic alternative to parameter tuning.

## Limitations & Future Work
- Population dynamics assessment relies on multiple manually specified thresholds (stagnation = 3, progress = 2, diversity = 0.3), which lack adaptive adjustment mechanisms.
- Experiments use only Qwen2.5-Max as the backbone; different LLMs may respond quite differently to the proposed prompt strategies.
- Semantic similarity among insights in the Insight Pool is measured solely via Jaccard similarity, which may fail to detect semantically equivalent but lexically distinct duplicate insights.

## Related Work & Insights
- **vs. EoH**: EoH lacks knowledge persistence and global control; HiFo-Prompt addresses both deficiencies through the Insight Pool and the Evolutionary Navigator.
- **vs. ReEvo**: ReEvo reflects only on individual candidates, whereas HiFo-Prompt performs macro-level monitoring of the entire population.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The ideas of code-thought decoupling and symbolic meta-optimization are highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multiple tasks including TSP, BPP, FSSP, and BO, though larger-scale instances are absent.
- **Writing Quality**: ⭐⭐⭐⭐ Method descriptions are detailed and clear, supported by rich mathematical formulations.
- **Value**: ⭐⭐⭐⭐ Provides a systematic framework for LLM-driven automatic algorithm design.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] LLM Prompt Duel Optimizer: Efficient Label-Free Prompt Optimization](../../ACL2026/model_compression/llm_prompt_duel_optimizer_efficient_label-free_prompt_optimization.md)
- [\[ICLR 2026\] Stress-Testing Alignment Audits with Prompt-Level Strategic Deception](stress-testing_alignment_audits_with_prompt-level_strategic_deception.md)
- [\[ICLR 2026\] A State-Transition Framework for Efficient LLM Reasoning](a_state-transition_framework_for_efficient_llm_reasoning.md)
- [\[ICLR 2026\] LLM DNA: Tracing Model Evolution via Functional Representations](llm_dna_tracing_model_evolution_via_functional_representations.md)
- [\[ICLR 2026\] ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference](paroquant_pairwise_rotation_quantization_for_efficient_reasoning_llm_inference.md)

<!-- RELATED:END -->
