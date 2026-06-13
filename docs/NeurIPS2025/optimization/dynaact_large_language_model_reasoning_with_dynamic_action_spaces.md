---
title: >-
  [Paper Note] DynaAct: Large Language Model Reasoning with Dynamic Action Spaces
description: >-
  [NeurIPS 2025][Optimization][Dynamic Action Space] DynaAct frames action space construction in LLM reasoning as a subset selection problem…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Dynamic Action Space"
  - "Submodular Function"
  - "MCTS"
  - "LLM Reasoning"
  - "Subset Selection"
date: 2026-05-08
content_hash: b53dada985f3cbcb
---

# DynaAct: Large Language Model Reasoning with Dynamic Action Spaces

**Conference**: NeurIPS 2025
**arXiv**: [2511.08043](https://arxiv.org/abs/2511.08043)  
**Code**: [https://github.com/zhaoxlpku/DynaAct](https://github.com/zhaoxlpku/DynaAct)  
**Area**: LLM Reasoning / Decision Optimization
**Keywords**: Dynamic Action Space, Submodular Function, MCTS, LLM Reasoning, Subset Selection

## TL;DR
DynaAct frames action space construction in LLM reasoning as a subset selection problem, dynamically assembling a compact action space at each step via a submodular function that balances utility and diversity. It outperforms rStar, RAP, and other baselines on 6 benchmarks, surpassing rStar by 6.8% on MATH-500.

## Background & Motivation

**Background**: Complex LLM reasoning is commonly formulated as an MDP: a state space and action space are defined, and search strategies such as MCTS are used to find optimal reasoning paths. RAP employs automatically generated sub-questions as actions, while rStar manually defines 5 action types.

**Limitations of Prior Work**: (a) Manually defined action spaces (e.g., rStar's 5 actions) are overly domain-specific and lack scalability; (b) Automatically generated action spaces (e.g., RAP's on-the-fly sub-questions) exhibit high redundancy, making exhaustive search computationally expensive.

**Key Challenge**: A high-quality action space must simultaneously satisfy two conflicting properties—**scalability** (automatically learned from data, generalizable across domains) and **compactness** (retaining only a small number of high-value candidate actions at each step).

**Goal**: To automatically construct an action space that is both general and compact for LLM reasoning.

**Key Insight**: Action space construction is recast as a subset selection problem, leveraging the diminishing marginal returns property of submodular functions to ensure that selected subsets balance utility and diversity.

**Core Idea**: A submodular function combined with a greedy algorithm dynamically selects the optimal 5 actions at each step from a large surrogate action space.

## Method

### Overall Architecture
The framework proceeds in three stages: (1) **Surrogate Action Space Estimation**—general reasoning patterns (observation sketches) are extracted from a diverse corpus to form the complete action space $\mathcal{A}$; (2) **Submodular Function Definition**—a scoring function $F$ that balances utility and diversity is defined, with an embedding model trained via Q-learning; (3) **Dynamic Action Space Construction**—at each step, a greedy algorithm selects $m=5$ actions from $\mathcal{A}$ to form $\mathcal{A}_t$, which is then evaluated by MCTS to determine the optimal action.

### Key Designs

1. **Surrogate Action Space Estimation**:

    - Function: Automatically extracts reusable reasoning patterns from multi-domain problem corpora.
    - Mechanism: The Open-Platypus corpus (24,652 problems) is partitioned into $k=2500$ groups; Llama-3.1-70B extracts an observation sketch from each group. After merging and deduplication, 40,822 observations constitute $\mathcal{A}$.
    - Design Motivation: Eliminates manual definition by automatically discovering cross-domain, generalizable reasoning operations from data.

2. **Submodular Function Design**:

    - Function: Defines the scoring function $F(\mathcal{A}_t; s_t) = \alpha f_\text{util}(\mathcal{A}_t; s_t) + \beta f_\text{div}(\mathcal{A}_t)$.
    - Mechanism:
        - Utility term: $f_\text{util} = \log(\sum_{a \in \mathcal{A}_t} \exp(\mathbf{e}(s_t)^T \mathbf{e}(a)))$; LogSumExp guarantees submodularity.
        - Diversity term: $f_\text{div} = \sum_{a_i} \min_{a_j \neq a_i} (1 - \mathbf{e}(a_i)^T \mathbf{e}(a_j))$; encourages maximal dissimilarity in the embedding space.
        - Weights: $\alpha=0.9$, $\beta=0.1$.
    - Design Motivation: The utility term ensures that selected actions are relevant to the current reasoning state; the diversity term prevents redundancy.

3. **Embedding Model Training (Q-learning)**:

    - Function: Trains the embedding function $\mathbf{e}(\cdot)$ such that $\mathbf{e}(s_t)^T \mathbf{e}(a)$ approximates the Q-value.
    - Mechanism: Observation sketches serve as demonstration data. The loss is defined as $\mathcal{L} = (\mathbf{e}(s_t)^T \mathbf{e}(a) - (r + \log\sum_{a'} \exp(\mathbf{e}(s_{t+1})^T \mathbf{e}(a'))))^2$, where $r=1$ for correct actions and $r=0$ otherwise.
    - Implementation: Llama-3.2-1B serves as the embedding backbone, trained on 83,083 state–action pairs.

4. **Greedy Action Selection**:

    - Function: Selects $m=5$ actions from $\mathcal{A}$ at each step.
    - Mechanism: Exploiting submodularity, the greedy algorithm guarantees a $(1-1/e)$ approximation ratio with complexity $O(m^2|\mathcal{A}|)$.
    - Implementation Optimization: $\mathbf{e}(a)$ can be precomputed and cached; only $\mathbf{e}(s_t)$ needs to be encoded online.

### Search Strategy
- MCTS performs 16 rollouts; the world model is Llama-3.1-8B-Instruct.
- Only the embedding model requires training; the base LLM remains frozen throughout.

## Key Experimental Results

### Main Results

| Type | Benchmark | Zero-shot CoT | SC@maj16 | RAP | rStar | **DynaAct** |
|------|-----------|--------------|----------|-----|-------|------------|
| General | MMLU | 68.87 | 69.66 | 69.46 | 68.61 | **70.22** |
| General | MMLU-Pro | 43.45 | 49.36 | 48.70 | 48.81 | **51.40** |
| Reasoning | GPQA | 31.82 | 34.34 | 38.89 | 36.87 | **39.39** |
| Reasoning | ARC-C | 81.06 | 80.63 | 85.41 | 86.43 | **88.31** |
| Math | GSM8K | 76.12 | 86.66 | 87.79 | 87.11 | **89.16** |
| Math | MATH-500 | 45.40 | 52.00 | 51.60 | 54.20 | **61.00** |

DynaAct achieves state-of-the-art results across all six benchmarks, outperforming rStar by 6.8 percentage points on MATH-500.

### Ablation Study

| Configuration | ARC-C | MATH-500 | Note |
|---------------|-------|----------|------|
| DynaAct (full) | 88.31 | 61.00 | Full model |
| - util | 87.63 | 53.40 | Remove utility term; MATH drops 7.6% |
| - div | 86.52 | 53.80 | Remove diversity term; drops 7.2% |
| - q-learning | 87.80 | 55.80 | No embedding training; drops 5.2% |
| - submodular | 85.15 | 52.00 | Remove submodular function; drops 9.0% |

### Efficiency Comparison

| Method | Relative Time ↓ | Accuracy ↑ |
|--------|----------------|-----------|
| **DynaAct** | 1.00 | **61.00** |
| rStar | 0.95 | 54.20 |
| RAP | 1.12 | 51.60 |

DynaAct incurs only 5% more latency than rStar while achieving 6.8% higher accuracy, and is faster than RAP, whose on-the-fly sub-question generation introduces substantial overhead.

### Key Findings
- The submodular function is central to performance: removing it causes a 9% drop on MATH-500, demonstrating that a dynamic, compact action space is far more critical than random or fixed alternatives.
- Both utility and diversity terms are indispensable: removing either reduces MATH-500 accuracy from 61% to approximately 53%.
- Compactness confers clear advantages: even with $m=5$, DynaAct improves consistently as rollout count increases; RAP shows negligible gains with $m=5$ or $m=10$ and requires $m=15$ to exhibit improvement.
- Critical-step coverage: DynaAct achieves 0.63 vs. rStar's 0.47, confirming that its action selection is meaningfully more relevant.

## Highlights & Insights
- **Formalizing action space construction as subset selection**: This is an underexplored research problem, orthogonal to policy learning and reward modeling, and opens a new direction for MDP-based LLM reasoning.
- **Elegant submodular function design**: The LogSumExp utility term naturally satisfies submodularity; using the sum of nearest-neighbor distances in the embedding space as a diversity measure is concise and effective.
- **Precomputed embeddings with greedy linear complexity**: Action embeddings can be cached, requiring only online encoding of the current state, thereby introducing negligible additional latency.

## Limitations & Future Work
- The surrogate action space is extracted from Open-Platypus, introducing domain bias; coverage may be insufficient for specific vertical domains such as programming or scientific experimentation.
- Substantial redundancy remains among the 40,822 observations; $\mathcal{A}$ itself could be further compressed.
- Extracting observation sketches relies on Llama-3.1-70B, which is unfriendly to smaller-model deployments.
- The current design uses a single embedding function $\mathbf{e}(\cdot)$ with limited representational capacity; multi-head or hierarchical embeddings warrant exploration.

## Related Work & Insights
- **vs. rStar**: rStar's manually defined 5-action space lacks scalability and underperforms DynaAct on MATH-500 and general-purpose tasks.
- **vs. RAP**: RAP generates sub-questions on the fly as the action space, resulting in high redundancy and large latency; DynaAct's pre-extraction combined with dynamic selection is more efficient.
- **vs. test-time scaling**: DynaAct focuses on action space quality rather than search strategy or data quality, constituting an orthogonal contribution.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Modeling action space construction as submodular subset selection is a novel and theoretically grounded perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six benchmarks, complete ablations, efficiency comparisons, compactness analysis, and utility analysis are all provided.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear, with strong integration between theory and experiments.
- Value: ⭐⭐⭐⭐ Provides an important infrastructural contribution to MDP-based LLM reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Large Language Bayes](large_language_bayes.md)
- [\[NeurIPS 2025\] The Trilemma of Truth in Large Language Models](the_trilemma_of_truth_in_large_language_models.md)
- [\[NeurIPS 2025\] Doubly Robust Alignment for Large Language Models](doubly_robust_alignment_for_large_language_models.md)
- [\[NeurIPS 2025\] Constrained Network Slice Assignment via Large Language Models](constrained_network_slice_assignment_via_llms.md)
- [\[NeurIPS 2025\] VERA: Variational Inference Framework for Jailbreaking Large Language Models](vera_variational_inference_framework_for_jailbreaking_large_language_models.md)

</div>

<!-- RELATED:END -->
