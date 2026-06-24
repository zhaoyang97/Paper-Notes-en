---
title: >-
  [Paper Note] Wider or Deeper: Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search
description: >-
  [NeurIPS 2025 Spotlight][LLM (Other)][MCTS] AB-MCTS proposes an adaptive-branching Monte Carlo Tree Search framework that dynamically decides at each node whether to go "wider" (generate new candidate answers) or "deeper" (refine existing answers using feedback), balancing exploration and exploitation via Bayesian posterior updates, and outperforms repeated sampling and standard MCTS on programming and engineering tasks.
tags:
  - "NeurIPS 2025 Spotlight"
  - "LLM (Other)"
  - "MCTS"
  - "inference-time compute"
  - "adaptive branching"
  - "Thompson sampling"
  - "code generation"
date: 2026-05-08
content_hash: 513f0847b1c0ee48
---

# Wider or Deeper: Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2503.04412](https://arxiv.org/abs/2503.04412)  
**Code**: [GitHub](https://github.com/SakanaAI/treequest)  
**Area**: LLM/NLP
**Keywords**: MCTS, inference-time compute, adaptive branching, Thompson sampling, code generation

## TL;DR

AB-MCTS proposes an adaptive-branching Monte Carlo Tree Search framework that dynamically decides at each node whether to go "wider" (generate new candidate answers) or "deeper" (refine existing answers using feedback), balancing exploration and exploitation via Bayesian posterior updates, and outperforms repeated sampling and standard MCTS on programming and engineering tasks.

## Background & Motivation

Inference-time compute scaling is a critical direction for improving LLM performance on complex tasks. Existing approaches fall into three categories: (1) post-training fine-tuning (e.g., o1), (2) reward-guided chain-of-thought, and (3) multi-answer generation. Among these, multi-answer generation (e.g., Best-of-N repeated sampling) is simple and effective, but lacks a mechanism to leverage external feedback for iterative improvement.

**Key Challenge**:
- Repeated sampling focuses on "exploration"—independently generating many candidates—without exploiting feedback for "utilization"
- Standard MCTS (e.g., LATS) supports multi-round refinement but uses a fixed branching factor as a hyperparameter, limiting exploitation of LLM's diverse output space
- The success of repeated sampling suggests that effective inference-time scaling requires fully utilizing the broad output space of LLMs, which fixed width impedes

**Key Insight**: Introduce unbounded branching into MCTS, allowing the search tree to adaptively decide at each node whether to expand new branches or deepen along existing ones, unifying exploration and exploitation into a single framework.

## Method

### Overall Architecture

AB-MCTS constructs a search tree $T$ where each non-root node corresponds to an LLM-generated answer. Each iteration consists of three steps: (1) select a node to expand; (2) generate new child nodes; (3) backpropagate scores. The core innovation is the introduction of **GEN nodes**—each node has a GEN child node representing the action "generate a new child answer." When a GEN node is selected, a new branch is expanded from its parent.

### Key Designs

1. **GEN Nodes and Adaptive Branching**:

    - The action set for each node $N$ is $A_N = \{a_0, a_1, \ldots, a_{n_{\text{child}}}\}$
    - $a_0$ corresponds to the GEN node ("go wider"); $a_1 \ldots a_{n_{\text{child}}}$ correspond to existing child nodes ("go deeper")
    - Selecting the GEN node = generating a new candidate answer; selecting an existing child node = deepening refinement
    - Key distinction from standard MCTS: already-expanded nodes can be expanded again, making the branching factor theoretically unbounded

2. **AB-MCTS-M (Mixed Model Variant)**:

    - Fits a mixed Bayesian model for each node: $r_{N_{\text{new}}, a_j} = \alpha_j + \sigma_y \epsilon_{N_{\text{new}}}$
    - $\alpha_j = \mu_\alpha + \sigma_\alpha \epsilon_j$ is the group-level intercept capturing the quality of the base answer
    - The GEN node is treated as a newly introduced group whose parameters are inferred via the shared posterior $\mu_\alpha, \sigma_\alpha$ from other groups
    - MCMC is used to sample from the posterior; Thompson sampling determines which action to select
    - Score backpropagation: scores of new nodes are added to the history of all their ancestors

3. **AB-MCTS-A (Node Aggregation Variant)**:

    - Introduces CONT nodes that aggregate all child nodes, representing "continue improving existing answers"
    - Each node has two options: a GEN node and a CONT node
    - Uses exponential family distributions with conjugate priors for efficient posterior updates
    - Two sub-variants: Gaussian model (unbounded scores) and Beta model (scores in $[0,1]$)
    - More lightweight than AB-MCTS-M with no shared parameters

### Thompson Sampling Selection Strategy

UCT scores are not used because GEN nodes make the problem fundamentally different from the standard multi-armed bandit setting—arms in standard MCTS are static, whereas in AB-MCTS the GEN node dynamically generates new arms. Thompson sampling, which makes decisions by sampling from posterior distributions, is naturally suited to this setting and also supports parallel expansion.

## Key Experimental Results

### Main Results (Multiple Benchmarks × Multiple Models)

| Method | LiveCodeBench (GPT-4o) | CodeContest (GPT-4o) | ARC-AGI (GPT-4o) | Avg. Rank |
|--------|----------------------|---------------------|-------------------|-----------|
| Repeated Sampling | 37.8 | 37.9 | **15.0** | 3.5 |
| Sequential Refinement | 37.8 | 30.1 | 8.7 | 5.5 |
| Standard MCTS | 36.7 | 37.5 | 9.0 | 4.2 |
| **AB-MCTS-M** | 38.9 | **40.6** | 12.3 | **2.3** |
| AB-MCTS-A (Gaussian) | **39.1** | 40.2 | 13.0 | 2.7 |
| AB-MCTS-A (Beta) | 38.7 | 40.4 | 14.0 | 2.7 |

### MLE-Bench (Machine Learning Competition Tasks)

| Method | Nomad2018 | Spooky | Pizza | Avg. Rank |
|--------|-----------|--------|-------|-----------|
| Repeated Sampling | 0.065 | 0.47 | 0.72 | 3.0 |
| Standard MCTS | 0.076 | 0.45 | 0.60 | 3.3 |
| **AB-MCTS-M** | **0.060** | **0.38** | **0.72** | **1.3** |

### Key Findings
- AB-MCTS consistently outperforms baselines on LiveCodeBench and CodeContest, achieving the best average rank (2.3)
- Repeated sampling remains strong on ARC-AGI (which requires broad exploration), but AB-MCTS achieves comparable performance
- When the budget is scaled to 512 (ARC-AGI), the improvement curve of AB-MCTS continues to rise while repeated sampling plateaus
- Search tree analysis shows AB-MCTS tends to produce wider trees (via adaptive expansion) while also deepening along promising branches
- Different tasks benefit from different exploration–exploitation trade-offs, and AB-MCTS adapts accordingly

## Highlights & Insights

- **The core contribution is introducing unbounded branching into MCTS**: this appears simple but is technically non-trivial, requiring an entirely new statistical model and selection strategy
- **The GEN node design is elegant**: it reframes "whether to expand a new branch" as a unified decision problem alongside "which child node to select"
- **Use of Bayesian posterior updates**: compared to count-based UCT, Bayesian methods perform better under data sparsity, which is well-suited to the LLM setting
- **The two variants are complementary**: the M variant leverages cross-group information via shared parameters, while the A variant is more lightweight with analytic updates
- **Orthogonal to post-training methods**: can be seamlessly combined with CoT fine-tuning approaches such as o1

## Limitations & Future Work

- Relies on a reliable external scoring function (e.g., test cases), which is not available for all tasks
- MCMC sampling (M variant) may become a computational bottleneck when the number of nodes is large
- No significant advantage over repeated sampling on ARC-AGI, indicating limited benefit for purely exploration-driven tasks
- Validated only on programming and ML engineering tasks; applicability to mathematical reasoning, creative writing, and other domains remains unknown
- Generation budget is measured in API call counts, without accounting for actual latency differences across calls

## Related Work & Insights

- **vs. Repeated Sampling**: Repeated sampling is pure exploration with no feedback utilization; AB-MCTS adaptively balances exploration and exploitation
- **vs. LATS/Standard MCTS**: Fixed branching factors limit exploitation of LLM output diversity; AB-MCTS supports unbounded branching
- **vs. Sequential Refinement**: Pure exploitation with no exploration; AB-MCTS is more flexible
- **vs. Progressive Widening**: PW is a classical MCTS widening technique but based on visit counts; AB-MCTS uses a Bayesian statistical model

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing unbounded branching into MCTS offers both theoretical and practical value, though the core idea (adaptive width vs. depth) is relatively intuitive
- Experimental Thoroughness: ⭐⭐⭐⭐ Four benchmarks × two frontier models × multiple budget settings × search tree analysis
- Writing Quality: ⭐⭐⭐⭐⭐ Problem motivation is clear, derivations of both variants are rigorous, and figures are professional
- Value: ⭐⭐⭐⭐ Directly relevant to inference-time compute scaling for LLMs, particularly well-suited to programming task settings

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute](../../ICML2025/llm_nlp/best-route_adaptive_llm_routing_with_test-time_optimal_compute.md)
- [\[ACL 2025\] Dynamic Parallel Tree Search for Efficient LLM Reasoning](../../ACL2025/llm_nlp/dynamic_parallel_tree_search_for_efficient_llm_reasoning.md)
- [\[ACL 2025\] Boosting LLM's Molecular Structure Elucidation with Knowledge Enhanced Tree Search Reasoning](../../ACL2025/llm_nlp/boosting_llms_molecular_structure_elucidation_with_knowledge_enhanced_tree_searc.md)
- [\[ICML 2026\] Compute as Teacher: Turning Inference Compute Into Reference-Free Supervision](../../ICML2026/llm_nlp/compute_as_teacher_turning_inference_compute_into_reference-free_supervision.md)
- [\[ACL 2025\] BFS-Prover: Scalable Best-First Tree Search for LLM-Based Automatic Theorem Proving](../../ACL2025/llm_nlp/bfs-prover_scalable_best-first_tree_search_for_llm-based_automatic_theorem_provi.md)

</div>

<!-- RELATED:END -->
