---
title: >-
  [Paper Note] Are Optimal Algorithms Still Optimal? Rethinking Sorting in LLM-Based Pairwise Ranking with Batching and Caching
description: >-
  [ACL 2025][LLM (Other)][Pairwise Ranking] This paper re-examines the choice of sorting algorithms in LLM-based Pairwise Ranking Prompting (PRP). It proposes a core cost model based on the number of LLM inference calls rather than the number of comparisons. It is found that the classical optimal algorithm, Heapsort, is no longer optimal when batching and caching optimizations are introduced. Quicksort reduces the number of inference calls by 44% when the batch size $\ge 2$…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Pairwise Ranking"
  - "Sorting Algorithms"
  - "LLM Inference Cost"
  - "Batching"
  - "Caching Optimization"
date: 2026-05-08
content_hash: 076d4623c2d8306a
---

# Are Optimal Algorithms Still Optimal? Rethinking Sorting in LLM-Based Pairwise Ranking with Batching and Caching

**Conference**: ACL 2025  
**arXiv**: [2505.24643](https://arxiv.org/abs/2505.24643)  
**Area**: LLM / Information Retrieval  
**Keywords**: Pairwise Ranking, Sorting Algorithms, LLM Inference Cost, Batching, Caching Optimization  

## TL;DR
This paper re-examines the choice of sorting algorithms in LLM-based Pairwise Ranking Prompting (PRP). It proposes a core cost model based on the number of LLM inference calls rather than the number of comparisons. It is found that the classical optimal algorithm, Heapsort, is no longer optimal when batching and caching optimizations are introduced. Quicksort reduces the number of inference calls by 44% when the batch size $\ge 2$, providing a new optimal choice for PRP sorting.

## Background & Motivation

**Background**: LLM-based Pairwise Ranking Prompting (PRP) is a popular zero-shot reranking method that ranks retrieval results by prompting an LLM to compare the relevance of two documents. Since each comparison requires one LLM inference, naive all-pairs comparisons incur extremely high cost, which has led researchers to introduce sorting algorithms to reduce the number of comparisons.

**Limitations of Prior Work**: Prior work (Qin et al., 2024) selected Heapsort ($O(n\log n)$ comparison complexity) as the standard sorting algorithm for PRP based on classical sorting theory. However, this analysis treats each comparison as an atomic operation with equal cost, ignoring the practical characteristics of LLM inference—namely, that LLM inference supports batching (processing multiple comparisons in a single inference call) and that duplicate comparisons can be avoided through caching.

**Key Challenge**: The cost model of classical sorting theory is the number of comparisons, whereas the actual cost in the LLM scenario is the number of inference calls. These two costs can be entirely different after introducing batching and caching; an algorithm might have more comparisons but fewer inference calls (as comparisons can be batched), and vice versa.

**Goal**: To establish a new cost model centered on the number of LLM inference calls, and to re-evaluate the efficiency of different sorting algorithms in the PRP scenario.

**Key Insight**: The tree structure of Heapsort makes each comparison sequentially dependent and thus unbatchable. In contrast, the partition operations of Quicksort involve independent comparisons between multiple elements and the pivot, naturally supporting batching. This structural difference in algorithms is negligible in traditional theory but crucial in the LLM scenario.

**Core Idea**: By replacing the number of comparisons with the number of LLM inference calls as the cost function, it is demonstrated that Quicksort significantly outperforms Heapsort when batch size $\ge 2$, and Quicksort is introduced into the PRP framework for the first time.

## Method

### Overall Architecture
With the number of LLM inference calls as the objective function, this paper systematically analyzes the actual number of inference calls of three sorting algorithms (Heapsort, Bubblesort, Quicksort) under three optimizations (batching, caching, Top-k extraction). Optimal algorithm recommendations are provided based on theoretical analysis and experimental validation.

### Key Designs

1. **Quicksort + Batching**:

    - **Function**: Leverage the parallelism in Quicksort's partition phase to reduce the number of inference calls.
    - **Mechanism**: During the partition phase of Quicksort, multiple elements need to be compared with the pivot. These comparisons are mutually independent and can be processed simultaneously in a single LLM inference. For example, comparisons of 20 documents against the pivot can be bundled into a single inference call (batch size=20) instead of 20 independent calls. The median-of-three strategy is used to select the pivot to ensure balanced partitioning.
    - **Design Motivation**: The partitioning operation of Quicksort is naturally parallelizable, and the Partial Quicksort variant supports efficient Top-k extraction.

2. **Bubblesort + Caching**:

    - **Function**: Reduce inference calls by caching redundant adjacent comparisons.
    - **Mechanism**: Bubblesort repeatedly compares adjacent element pairs across multiple passes. If (A, B) was already compared in pass $i$ and their positions have not changed, pass $i+1$ can directly read the result from the cache instead of invoking the LLM again. A dictionary is used to store completed comparison results, with negligible memory overhead.
    - **Design Motivation**: Although Bubblesort has a comparison complexity of $O(n^2)$, many comparisons are redundant. Caching can reduce the actual number of inference calls by approximately 46%.

3. **Cost Framework Centered on Inference Calls**:

    - **Function**: Provide a proper algorithmic analysis foundation for the LLM sorting scenario.
    - **Mechanism**: Replace the cost function from the number of comparisons $C(n)$ to the number of inference calls $I(n, b)$ (where $b$ is the batch size). For Quicksort, $I_{QS}(n, b) \approx \frac{2n\ln n}{b}$. For Heapsort, $I_{HS}(n) \approx 2n\log_2 n$ (unbatchable). When $b \geq 2$, the number of inference calls for Quicksort is already fewer than that of Heapsort.
    - **Design Motivation**: An accurate cost model is essential for making the correct algorithm choice. A minimal parallelism of batch size=2 is sufficient to reverse the advantage of Heapsort.

### Loss & Training
Model training is not involved. Sorting experiments are conducted using multiple pre-trained LLMs (Flan-T5-L/XL/XXL, Mistral-7B, Llama-3-8B), and the sorting quality is measured by nDCG@10.

## Key Experimental Results

### Main Results

| Algorithm | Batch Size=1 Inference Calls | Batch Size=2 Inference Calls | Batch Size=8 Inference Calls | nDCG@10 |
|------|--------------------|--------------------|--------------------|---------| 
| Heapsort | 526±42 | 526±42 (Unbatchable) | 526±42 | 0.698 |
| Bubblesort (No Cache) | 4950±0 | 4950±0 | 4950±0 | 0.702 |
| Bubblesort (With Cache) | 2673±312 | 2673±312 | 2673±312 | 0.702 |
| Quicksort (No Batching) | 631±95 | 631±95 | 631±95 | 0.695 |
| **Quicksort (With Batching)** | 631±95 | **354±58** | **127±22** | **0.695** |

### Ablation Study

| Configuration | Inference Calls | Gain over Heapsort |
|------|---------|----------------|
| Heapsort (baseline) | 526 | — |
| Quicksort, batch=2 | 354 | **-33%** |
| Quicksort, batch=4 | 207 | **-61%** |
| Quicksort, batch=8 | 127 | **-76%** |
| Quicksort, batch=128 (A100) | ~50 | **-90%**, 5.52× real speedup |

### Key Findings
- With only batch size=2, Quicksort requires 33% fewer inference calls than Heapsort, overturning the "optimal Heapsort" conclusion.
- At batch size=128 on an A100 GPU, Quicksort's end-to-end sorting speed is 5.52 times faster than Heapsort.
- The sorting quality (nDCG@10) of the three algorithms is nearly identical; the choice of algorithm does not affect the sorting effectiveness, only the speed.
- Although Bubblesort reduces inference calls by 46% via caching, its absolute count remains significantly higher than the other two algorithms.
- GPU architecture impacts batching benefits: A100 reaches close to ideal linear scaling up to batch=128, while 3090 saturates after batch=32.

## Highlights & Insights
- The core insight of this paper, though simple, is far-reaching: **in the LLM era, the cost model of classical algorithm theory needs to be re-examined**. The cost of pushing and popping a heap element is no longer equal to the cost of one LLM call, as batching changes the rules of the game.
- This finding is immediately applicable to all commercial ranking systems utilizing PRP—simply by replacing Heapsort with Quicksort and enabling batching, substantial speedups can be achieved.
- More broadly, any scenario where LLMs are repeatedly called in a loop (such as Tree-of-Thought, MCTS search) should reconsider the parallelization of call patterns.

## Limitations & Future Work
- Pairwise comparisons of LLMs may violate transitivity (A > B, B > C do not necessarily imply A > C), whereas sorting algorithms assume transitivity holds.
- Experiments only utilize medium-sized models (up to 11B); the batching characteristics of ultra-large API-based models might differ.
- Hybrid strategies are not considered, such as first using an efficient algorithm for coarse ranking, followed by a precise algorithm for fine ranking.
- Future work can explore the integration of sorting/ranking algorithms that do not assume transitivity (e.g., noisy sorting) with LLM cost models.

## Related Work & Insights
- **vs Qin et al. (2024)**: Pioneering work on PRP recommended Heapsort, while this paper demonstrates that this recommendation no longer holds under batching settings.
- **vs PRP-Graph (Luo et al., 2024)**: PRP-Graph reduces the number of comparisons through graph aggregation, which is complementary to the optimization perspective of this paper.
- **vs Listwise ranking**: Listwise methods rank multiple documents at once, sharing a similar underlying goal with PRP+Quicksort.

## Rating
- Novelty: ⭐⭐⭐⭐ The redefinition of the cost model, although conceptually simple, is highly impactful; it introduces Quicksort to PRP for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 LLMs, 8 datasets, 3 algorithms, various batch sizes, and latency analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ The exposition is clear and concise, with theoretical analysis and experimental validation complementing each other perfectly.
- Value: ⭐⭐⭐⭐⭐ Immediately applicable to improving all PRP-based ranking systems, possessing extremely high practical engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Q♯: Provably Optimal Distributional RL for LLM Post-Training](../../NeurIPS2025/llm_nlp/qsharp_provably_optimal_distributional_rl_for_llm_post-training.md)
- [\[ICML 2025\] BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute](../../ICML2025/llm_nlp/best-route_adaptive_llm_routing_with_test-time_optimal_compute.md)
- [\[ACL 2025\] Ranking Unraveled: Recipes for LLM Rankings in Head-to-Head AI Combat](ranking_unraveled_recipes_for_llm_rankings_in_head-to-head_ai_combat.md)
- [\[ACL 2025\] SR-LLM: Rethinking the Structured Representation in Large Language Model](sr-llm_rethinking_the_structured_representation_in_large_language_model.md)
- [\[NeurIPS 2025\] Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models](../../NeurIPS2025/llm_nlp/nemotron-flash_towards_latency-optimal_hybrid_small_language_models.md)

</div>

<!-- RELATED:END -->
