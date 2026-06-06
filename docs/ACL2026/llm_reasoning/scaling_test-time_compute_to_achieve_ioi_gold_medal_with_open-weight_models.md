---
title: >-
  [Paper Note] Scaling Test-Time Compute to Achieve IOI Gold Medal with Open-Weight Models
description: >-
  [ACL 2026][LLM Reasoning][Test-time compute] Ours proposes GenCluster, a scalable test-time compute framework. Through massive parallel generation → behavioral clustering → tournament ranking → round-robin submission str…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Test-time compute"
  - "competitive programming"
  - "IOI"
  - "behavioral clustering"
  - "open-weight models"
date: 2026-05-08
content_hash: 8ebe9a04bf423297
---

# Scaling Test-Time Compute to Achieve IOI Gold Medal with Open-Weight Models

**Conference**: ACL 2026  
**arXiv**: [2510.14232](https://arxiv.org/abs/2510.14232)  
**Code**: [NVIDIA-NeMo/Skills](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes)  
**Area**: LLM Inference  
**Keywords**: Test-time compute, competitive programming, IOI, behavioral clustering, open-weight models

## TL;DR

Ours proposes GenCluster, a scalable test-time compute framework. Through massive parallel generation → behavioral clustering → tournament ranking → round-robin submission strategy, it enables the open-weight model gpt-oss-120b to achieve gold medal level (446.75/600 points) on IOI 2025 for the first time.

## Background & Motivation

**Background**: Competitive programming has become a rigorous benchmark for evaluating LLM reasoning and problem-solving capabilities. The International Olympiad in Informatics (IOI) is one of the most prestigious algorithmic programming competitions. Traditional benchmarks like HumanEval and MBPP have approached saturation, shifting research focus toward more challenging benchmarks such as LiveCodeBench and Codeforces.

**Limitations of Prior Work**: OpenAI claims to have achieved gold medals at IOI 2024/2025 using o1-ioi and o3, but neither the methods nor the models have been disclosed. While AlphaCode 2 published some methodology, it also relies on closed-source models. Open-weight models (e.g., DeepSeek-R1-0528, Qwen3-235B) have shown increased competitiveness on LiveCodeBench and Codeforces but still lag significantly at the IOI-level difficulty.

**Key Challenge**: Each IOI problem allows a maximum of 50 submissions—the critical challenge lies in efficiently selecting the optimal solution from thousands of candidates. Existing methods are either proprietary (OpenAI) or have not been validated at the IOI-level difficulty.

**Goal**: Design a scalable, reproducible, and transparent test-time compute method to enable open-weight models to reach gold medal levels under strict IOI submission limits.

**Key Insight**: IOI problems are essentially reasoning tasks with a "limited verification budget." Unlike math problems where majority voting can be used, majority voting is ineffective in competitive programming when the pass rate is extremely low, necessitating a more granular solution selection strategy.

**Core Idea**: Generate a large number of candidate solutions → group them based on behavioral clustering → rank the clusters using an LLM tournament → maximize scores through round-robin submission.

## Method

### Overall Architecture

GenCluster consists of four stages: (1) Parallel generation of K candidate C++ programs for each subtask; (2) Clustering based on behavioral similarity—generating test inputs and grouping by output hashes; (3) Ranking cluster representatives using an LLM-as-a-judge tournament; (4) Round-robin submission of candidates following the rankings, adhering to the IOI rule of 50 submissions per problem.

### Key Designs

1.  **Behavioral Clustering**:
    - **Function**: Groups functionally identical candidates to reduce ranking complexity.
    - **Mechanism**: LLMs generate 100 test input generators and 100 verifiers; test inputs are filtered via verifier voting (retained if $\ge 75\%$ verifiers pass) to collect 100 valid cases. All candidates are executed on these cases, and those with identical outputs are grouped into the same cluster (using hashing for acceleration); clusters with empty outputs are removed entirely.
    - **Design Motivation**: Comparing tens of thousands of solutions directly is infeasible; behavior-based clustering distinguishes functionally equivalent implementations better than text-based clustering.

2.  **Tournament Ranking**:
    - **Function**: Identifies the most likely correct clusters among a large set after clustering.
    - **Mechanism**: The solution with the longest reasoning chain in each cluster is chosen as the representative. Each representative undergoes $G_n=10$ rounds of pairwise comparisons with randomly selected representatives from other clusters, where an LLM judge selects a winner (randomizing order to mitigate recency bias). Clusters are ranked by win count.
    - **Design Motivation**: Majority voting fails at IOI difficulty due to the extremely low ratio of correct solutions; tournament ranking is more stable than direct scoring.

3.  **Round-Robin Submission strategy**:
    - **Function**: Maximizes the total score within the 50-submission limit.
    - **Mechanism**: Starting from the most difficult subtask, submissions are made following the cluster ranking order; within each cluster, solutions are selected based on reasoning length. Once a subtask reaches a full score, it is skipped to focus on the next. Submissions circulate across clusters.
    - **Design Motivation**: IOI scoring takes the highest score across all submissions for each subtask—round-robin ensures coverage of solutions with different behavioral patterns, increasing the probability of hitting a correct solution.

### Loss & Training

This method requires no training and operates purely as test-time compute. All candidates are generated in C++ with a maximum generation length of 120K tokens (gpt-oss-120b).

## Key Experimental Results

### Main Results

| Model | K=50 | K=1000 | K=5000 | Trend |
| :--- | :--- | :--- | :--- | :--- |
| gpt-oss-120b | ~332 | ~400 | **446.75** | Steady Growth |
| gpt-oss-20b | ~250 | ~300 | ~330 | Slower Growth |
| DeepSeek-R1-0528 | ~280 | ~310 | ~340 | Strong early, then saturates |
| Qwen3-235B-A22B | ~290 | ~330 | ~350 | Saturation after 48K tokens |

IOI 2025 Gold Medal Threshold: ~400 points, Total: 600 points. GenCluster + gpt-oss-120b (K=5000) scored 446.75, reaching gold medal level.

### Ablation Study (K=5000, gpt-oss-120b)

| Method | Use Clustering | Score | Description |
| :--- | :--- | :--- | :--- |
| Random | No | 300.10 | Random selection |
| Longest | No | 277.36 | Longest reasoning chain |
| Cluster-Size | Yes | 299.87 | Ranked by cluster size |
| Cluster-Majority | Yes | 314.22 | Majority voting |
| GenCluster (Random-Rep) | Yes | 406.49 | Random Rep + Tournament |
| GenCluster (Score-Based) | Yes | 441.11 | Average score ranking |
| **GenCluster** | **Yes** | **446.75** | **Longest Rep + Win Ranking** |

### Key Findings

- gpt-oss-120b is the only open-weight model capable of reaching gold medal levels within 5000 generations, showing continuous improvement with scaling.
- Simple majority voting (314.22) and cluster-size ranking (299.87) perform similarly to random selection at IOI difficulty because the proportion of correct solutions is extremely low.
- Tournament ranking slightly outperforms direct scoring (446.75 vs 441.11), and selecting the longest reasoning chain as the representative is superior to random selection (446.75 vs 406.49).
- Increasing test cases from 10 to 100 significantly improves clustering purity (F1), though it also increases the number of clusters, making selection more difficult.
- Reasoning chain length correlates positively with accuracy; gpt-oss models generate longer chains on difficult problems.
- Total compute: Generating 5000 candidates takes approximately 7.3 billion tokens, with another 7.3 billion tokens required for tournament ranking.

## Highlights & Insights

- **First Open-Weight IOI Gold**: Provides a fully transparent and reproducible methodology, contrasting with OpenAI's closed methods, and offering a baseline for the open-source community to catch up.
- **Sophisticated Tournament Ranking**: In hard tasks with low success rates where traditional majority voting fails, the pairwise tournament mechanism better leverages the LLM's evaluative capabilities.
- **Behavioral Clustering + Verifier Voting**: The generator-verifier pipeline for test input generation is simple yet effective; the 75% voting threshold balances coverage and quality.
- **Reasoning Length as Accuracy Proxy**: The simple heuristic of selecting the longest reasoning chain within a cluster as the representative significantly outperforms random selection and is transferable to other reasoning tasks.

## Limitations & Future Work

- Massive computational requirements (~14.6 billion tokens per evaluation) make it infeasible in resource-constrained environments.
- Room for improvement in ranking quality—only 35 out of 39 subtasks had their best solution appear in the Top-50 clusters.
- Only validated on IOI 2025; generalization to other competitions (ICPC, Codeforces) remains to be tested.
- Training details for gpt-oss-120b are not disclosed, so reproducibility of the method partially depends on that specific model.

## Related Work & Insights

- **vs AlphaCode/AlphaCode 2**: Also uses large-scale generation and clustering, but AlphaCode did not solve the ranking problem under constrained submission budgets and utilized closed models.
- **vs OpenAI o1-ioi**: OpenAI achieved gold (o3) with 1K solutions and simple selection, but the method is entirely proprietary; GenCluster reaches gold with 5K solutions and a tournament, maintaining full transparency.
- **vs Best-of-N**: Simple Best-of-N requires a reliable verifier, which is not easily available for IOI; GenCluster replaces the verifier with behavioral clustering and tournaments.

## Rating

- Novelty: ⭐⭐⭐⭐ Tournament ranking and round-robin strategies are innovative, though the core framework (generate+cluster+select) follows the AlphaCode lineage.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Systematic scaling analysis, multi-model comparisons, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear descriptions and sufficient experiments, though partially dependent on the undisclosed gpt-oss model.
- Value: ⭐⭐⭐⭐⭐ A milestone work for the first open-weight IOI gold medal with a fully reproducible method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Scaling Evaluation-Time Compute with Reasoning Models as Evaluators](scaling_evaluation-time_compute_with_reasoning_models_as_evaluators.md)
- [\[NeurIPS 2025\] Provable Scaling Laws for the Test-Time Compute of Large Language Models](../../NeurIPS2025/llm_reasoning/provable_scaling_laws_for_the_testtime_compute_of_large_lang.md)
- [\[ICML 2026\] Diversity Matters: Revisiting Test-Time Compute in Vision-Language Models](../../ICML2026/llm_reasoning/diversity_matters_revisiting_test-time_compute_in_vision-language_models.md)
- [\[ACL 2026\] Efficient Test-Time Scaling via Temporal Reasoning Aggregation](efficient_test-time_scaling_via_temporal_reasoning_aggregation.md)
- [\[NeurIPS 2025\] Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](../../NeurIPS2025/llm_reasoning/towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)

</div>

<!-- RELATED:END -->
