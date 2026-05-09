---
title: >-
  [Paper Note] A*-Thought: Efficient Reasoning via Bidirectional Compression for Low-Resource Settings
description: >-
  [NeurIPS 2025][Model Compression][reasoning efficiency] This paper proposes A*-Thought, a CoT compression framework based on the A* search algorithm. It introduces Bidirectional Importance Scoring (BIS) to measure each reasoning step's relevance to both the question and the answer, and combines path-level A* search to efficiently identify the most compact valid reasoning path within an exponentially large search space. Under a 512-token budget, A*-Thought improves QwQ-32B accuracy by 2.39×; under a 4096-token budget, it reduces output tokens by approximately 50% with negligible accuracy loss.
tags:
  - NeurIPS 2025
  - Model Compression
  - reasoning efficiency
  - CoT compression
  - "A* search"
  - bidirectional importance
  - long-to-short
date: 2026-05-08
content_hash: 0c30bf6b8938f0d5
---

# A*-Thought: Efficient Reasoning via Bidirectional Compression for Low-Resource Settings

**Conference**: NeurIPS 2025
**arXiv**: [2505.24550](https://arxiv.org/abs/2505.24550)
**Code**: [GitHub](https://github.com/AI9Stars/AStar-Thought)
**Area**: LLM Reasoning / Model Compression
**Keywords**: reasoning efficiency, CoT compression, A* search, bidirectional importance, long-to-short

## TL;DR
This paper proposes A*-Thought, a CoT compression framework based on the A* search algorithm. It introduces Bidirectional Importance Scoring (BIS) to measure each reasoning step's relevance to both the question and the answer, and combines path-level A* search to efficiently identify the most compact valid reasoning path within an exponentially large search space. Under a 512-token budget, A*-Thought improves QwQ-32B accuracy by 2.39×; under a 4096-token budget, it reduces output tokens by approximately 50% with negligible accuracy loss.

## Background & Motivation

**State of the Field**: Large Reasoning Models (LRMs) such as o1, R1, and QwQ achieve strong reasoning capabilities via long chain-of-thought (CoT), but their verbose reasoning traces incur substantial inference overhead, severely limiting deployment in resource-constrained settings.

**Limitations of Prior Work**: (a) Existing long-to-short methods commonly assume "overthinking" and attempt to compress CoT directly, often at the cost of performance degradation; (b) prompt-based methods (e.g., CoD) yield imprecise compression; (c) training-based methods (e.g., TokenSkip) require additional training with limited effectiveness; (d) no unified framework exists for systematically identifying the optimal subset from an exponentially large search space.

**Root Cause**: Not all steps in a CoT are equally important, yet identifying which steps are necessary and which are dispensable is a combinatorial explosion problem—$N$ steps yield $2^N$ possible subsets.

**Starting Point**: CoT compression is formalized as a search problem, with the A* algorithm applied to efficiently search over ranked reasoning steps.

**Core Idea**: Bidirectional Importance Scoring (considering each step's relevance to both the question and the answer) combined with A* search (current path quality + estimated cost to reach the correct answer) to find the shortest valid reasoning path in a vast search space.

## Method

### Overall Architecture
A two-level design: (1) **Step level** — Bidirectional Importance Scoring (BIS) evaluates the criticality of each reasoning step to guide search priority; (2) **Path level** — A* search iteratively expands nodes on a search tree, using a cost function to identify the shortest reasoning path that leads to a correct solution.

### Key Designs

1. **Bidirectional Importance Scoring (BIS)**:

    - **Function**: Quantifies the importance of each reasoning step $\mathbf{t}^{(n)}$
    - **Mechanism**: A good intermediate step should be highly relevant to both the question (forward) and the answer (backward)
    - Attention level: $\text{ATTN}(\mathbf{q}|\mathbf{t}^{(n)})$ and $\text{ATTN}(\mathbf{s}|\mathbf{t}^{(n)})$
    - Model level: $\text{NLL}(\mathbf{q}|\mathbf{t}^{(n)})$ and $\text{NLL}(\mathbf{s}|\mathbf{t}^{(n)})$
    - BIS = weighted sum of attention scores / weighted sum of NLL, with $\alpha$ controlling the balance between question and answer relevance
    - Computed using a lightweight model (GPT-2) for efficiency
    - **Key observation**: BIS scores are highly skewed — only a small fraction of steps simultaneously contribute strongly to both the question and the answer

2. **Path-Level A* Search**:

    - **Function**: Efficiently identifies the shortest valid reasoning path within a $2^N$ search space
    - **Initialization**: All steps are ranked in descending BIS order to form a search queue (best-first strategy)
    - **Node representation**: Each node is not a single step but a triplet $\mathbf{r}^{(n)} = \langle \mathbf{t}^{(n-1)}, \mathbf{t}^{(n)}, \mathbf{t}^{(n+1)} \rangle$ consisting of a step and its neighbors — mitigating fragmentation
    - **Verification**: When path depth $\geq k_{min}$, a verification model checks whether the current path leads to the correct answer
    - **Expansion**: Upon verification failure, $W$ candidate steps are drawn from the queue as child nodes

3. **Cost Function Design**:

    - Total cost: $f = g + h$ (classical A* framework)
    - Current cost $g$: negative log-likelihood of the current path under the verification model (path quality)
    - Future cost $h$: conditional self-information of generating the correct answer given the current path, $\mathcal{I}(\mathbf{s}|\mathbf{q}, \mathbf{t}'_k) = -\frac{1}{|\mathbf{s}|}\log P_\mathcal{V}(\mathbf{s}|\mathbf{q}, \mathbf{t}'_k)$
    - **Intuition**: A larger $h$ indicates that the current path is farther from the correct answer and requires additional steps
    - A depth upper bound $k_{max}$ prevents excessive search depth

### Loss & Training
- A* search is used offline to compress CoT data
- The resulting compact reasoning paths are used for SFT distillation, training LRMs to directly generate efficient reasoning traces
- Verification model: s1.1-32B
- Training data: long CoT from the s1K-1.1 dataset

## Key Experimental Results

### Main Results (QwQ-32B, varying token budgets)

| Method | 512 Avg Acc. | 512 ACU | 1024 Avg Acc. | 2048 Avg Acc. | 4096 Avg Acc. | 4096 Avg Len. |
|--------|-------------|---------|--------------|--------------|--------------|--------------|
| QwQ-32B baseline | 12.3% | 2.41 | 21.8% | 43.8% | 63.2% | 2812 |
| + CoD | 17.3% | 3.34 | 24.3% | 51.3% | 69.3% | 2804 |
| + BtC Effective | 19.5% | 3.80 | 25.0% | 52.3% | 68.6% | 2795 |
| + TokenSkip | 16.6% | 3.20 | 22.8% | 48.8% | 64.9% | 2549 |
| **+ A*-Thought** | **29.4%** | **5.99** | **48.1%** | **58.9%** | **69.3%** | **1877** |

### Out-of-Domain Generalization (LiveCodeBench + MMLU)

| Budget | QwQ-32B Acc. | + A*-Thought Acc. | + A*-Thought ACU |
|--------|-------------|-------------------|-----------------|
| 512 | 18.8% | **30.6%** | 6.74 |
| 1024 | 28.7% | **41.9%** | 5.37 |
| 4096 | 45.8% | **59.7%** | 3.16 |

### Key Findings
- **Substantial gains under low budgets**: At 512 tokens, accuracy improves from 12.3% to 29.4% (2.39×), and ACU from 2.41 to 5.99 (2.49×)
- **Token reduction without accuracy loss at high budgets**: At 4096 tokens, average length decreases from 2826 to 1877 (−33.6%) with negligible accuracy drop (70.6% → 69.3%)
- **Cross-model generality**: Effective across QwQ-32B, R1-Distill-32B, and s1.1-32B
- **Out-of-domain generalization**: Trained solely on math data, yet yields significant gains on LiveCodeBench and MMLU

## Highlights & Insights
- **Formalizing CoT compression as a search problem**: Viewing reasoning compression through the lens of combinatorial optimization is more principled than heuristic deletion
- **Key insight of bidirectional importance**: Relevance to the question alone is insufficient; relevance to the answer must also be considered — intermediate steps that appear superficially unrelated should be retained if they contribute to deriving the answer
- **Triplet node design**: Steps are not selected in isolation; their immediate neighbors are preserved to avoid fragmented reasoning chains
- **Natural fit of A* search**: The cost function is elegantly designed — the current cost measures path quality while the future cost measures the remaining difficulty of reaching the answer

## Limitations & Future Work
- **Dependency on pre-generated CoT**: A* search is a post-processing method that requires a complete long CoT to be generated first; it cannot reduce the cost of initial inference
- **Verification model overhead**: Each expansion and verification step during A* search requires invoking a 32B verification model, making the search process itself costly
- **BIS requires the ground-truth answer**: Computing BIS presupposes knowledge of the correct answer $\mathbf{s}$, limiting direct applicability in unannotated settings
- **Future directions**: (1) Train a forward BIS predictor that does not require the answer; (2) explore online CoT compression (pruning during generation); (3) reduce search cost by using smaller verification models

## Related Work & Insights
- **vs. CoD (Chain-of-Draft)**: CoD uses prompting to elicit concise reasoning, but compression is imprecise; A*-Thought precisely selects the most critical steps from existing CoT
- **vs. TokenSkip**: TokenSkip requires training a token-level skipping policy; A*-Thought operates at the reasoning-step level, which is a more appropriate granularity
- **vs. Budget Forcing (s1)**: Budget Forcing controls length via truncation or extension but does not optimize which content is retained; A*-Thought explicitly optimizes the combination of preserved steps
- **Insight**: The success of A* search in reasoning compression suggests that CoT contains substantial redundancy — truly critical reasoning steps may constitute only 30–50% of the full trace

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of A* search and bidirectional importance scoring is original and insightful
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three models × four budgets × six benchmarks, with comprehensive ablation studies
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear and methodological derivation is rigorous
- **Value**: ⭐⭐⭐⭐ Directly applicable practical value for deploying LRMs in resource-constrained settings

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Linear Attention for Efficient Bidirectional Sequence Modeling](linear_attention_for_efficient_bidirectional_sequence_modeling.md)
- [\[NeurIPS 2025\] REP: Resource-Efficient Prompting for Rehearsal-Free Continual Learning](rep_resource-efficient_prompting_for_rehearsal-free_continual_learning.md)
- [\[NeurIPS 2025\] QSVD: Efficient Low-Rank Approximation for Unified Query-Key-Value Weight Compression](qsvd_efficient_low-rank_approximation_for_unified_query-key-value_weight_compres.md)
- [\[NeurIPS 2025\] TokenSqueeze: Performance-Preserving Compression for Reasoning LLMs](tokensqueeze_performance-preserving_compression_for_reasoning_llms.md)
- [\[NeurIPS 2025\] FirstAidQA: A Synthetic Dataset for First Aid and Emergency Response in Low-Connectivity Settings](firstaidqa_a_synthetic_dataset_for_first_aid_and_emergency_response_in_low-conne.md)

<!-- RELATED:END -->
