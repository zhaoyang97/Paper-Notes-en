---
title: >-
  [Paper Note] Scaling Test-Time Compute to Achieve IOI Gold Medal with Open-Weight Models
description: >-
  [ACL 2026][LLM Reasoning][Test-time compute] This paper proposes GenCluster, a scalable test-time compute framework that achieves gold-medal performance on IOI 2025 (446.75/600) with the open-weight model gpt-oss-120b, via large-scale parallel generation → behavioral clustering → tournament ranking → round-robin submission.
tags:
  - ACL 2026
  - LLM Reasoning
  - Test-time compute
  - competitive programming
  - IOI
  - behavioral clustering
  - open-weight models
date: 2026-05-08
content_hash: 7b840b794a78c06e
---

# Scaling Test-Time Compute to Achieve IOI Gold Medal with Open-Weight Models

**Conference**: ACL 2026
**arXiv**: [2510.14232](https://arxiv.org/abs/2510.14232)
**Code**: [NVIDIA-NeMo/Skills](https://github.com/NVIDIA-NeMo/Skills/tree/main/recipes)
**Area**: LLM Reasoning
**Keywords**: Test-time compute, competitive programming, IOI, behavioral clustering, open-weight models

## TL;DR

This paper proposes GenCluster, a scalable test-time compute framework that achieves gold-medal performance on IOI 2025 (446.75/600) with the open-weight model gpt-oss-120b, via large-scale parallel generation → behavioral clustering → tournament ranking → round-robin submission.

## Background & Motivation

**State of the Field**: Competitive programming has become a rigorous benchmark for evaluating LLM reasoning and problem-solving capabilities. The International Olympiad in Informatics (IOI) is one of the most prestigious algorithmic programming competitions. Traditional benchmarks such as HumanEval and MBPP are nearing saturation, shifting research focus toward more challenging benchmarks like LiveCodeBench and Codeforces.

**Limitations of Prior Work**: OpenAI claims gold-medal performance on IOI 2024/2025 using o1-ioi and o3, but neither the methods nor the models have been publicly released. AlphaCode 2 discloses partial methodology but also relies on closed-source models. Open-weight models (e.g., DeepSeek-R1-0528, Qwen3-235B) have become increasingly competitive on LiveCodeBench and Codeforces, yet remain significantly behind on IOI-level difficulty.

**Root Cause**: IOI allows at most 50 submissions per problem — efficiently selecting the best solution from thousands of candidates is the central challenge. Existing approaches are either undisclosed (OpenAI) or have not been validated at IOI-level difficulty.

**Paper Goals**: Design a scalable, reproducible, and transparent test-time compute method that enables open-weight models to achieve gold-medal performance under IOI's strict submission constraints.

**Starting Point**: IOI problems are fundamentally reasoning tasks with limited verification budgets. Unlike mathematical problems where majority voting is effective, majority voting fails when the proportion of correct solutions is extremely low, necessitating more refined solution selection strategies.

**Core Idea**: Generate a large pool of candidate solutions → group them via behavioral clustering → rank clusters using an LLM-based tournament → iteratively submit to maximize scores.

## Method

### Overall Architecture

GenCluster consists of four stages: (1) parallel generation of $K$ candidate C++ programs for each subtask; (2) behavioral clustering — generating test inputs and grouping candidates by output hash; (3) tournament ranking of cluster representatives using an LLM as judge; (4) round-robin submission in ranked order, subject to the 50-submission-per-problem IOI constraint.

### Key Designs

1. **Behavioral Clustering**:

    - **Function**: Groups candidates with identical behavior to reduce ranking complexity.
    - **Mechanism**: An LLM generates 100 test input generators and 100 validators; validators vote to filter valid test inputs (retained if ≥75% of validators pass), yielding 100 valid test cases. All candidates are executed on these cases, and those producing identical outputs are assigned to the same cluster (hash-based comparison for efficiency); clusters with empty outputs are discarded entirely.
    - **Design Motivation**: Directly comparing tens of thousands of solutions is infeasible; behavior-based clustering better distinguishes functionally equivalent implementations than text-similarity-based clustering.

2. **Tournament Ranking**:

    - **Function**: Identifies the most likely correct cluster among a large set of clusters post-clustering.
    - **Mechanism**: The solution with the longest chain-of-thought is selected as the representative of each cluster. Each representative undergoes $G_n=10$ pairwise comparisons against randomly selected representatives from other clusters; an LLM judge selects the winner of each round (presentation order is randomized to mitigate recency bias). Clusters are ranked by win count.
    - **Design Motivation**: Majority voting is ineffective at IOI-level difficulty where correct solutions are extremely rare; pairwise tournament comparisons leverage LLM judgment more reliably than direct scoring.

3. **Round-Robin Submission**:

    - **Function**: Maximizes total score within the 50-submission limit.
    - **Mechanism**: Submission begins with the hardest subtasks, following cluster ranking order; within each cluster, solutions are ordered by reasoning length. Once a subtask reaches full score, it is skipped and the next subtask is targeted. Submission rotates across clusters in a round-robin fashion.
    - **Design Motivation**: IOI scoring takes the highest score across all submissions per subtask — round-robin submission ensures coverage of solutions with diverse behavioral patterns, increasing the probability of encountering a correct solution.

### Loss & Training

This method requires no training and operates purely at inference time. All candidate solutions are generated in C++ with a maximum generation length of 120K tokens (gpt-oss-120b).

## Key Experimental Results

### Main Results

| Model | K=50 | K=1000 | K=5000 | Trend |
|-------|------|--------|--------|-------|
| gpt-oss-120b | ~332 | ~400 | **446.75** | Steady improvement |
| gpt-oss-20b | ~250 | ~300 | ~330 | Slower growth |
| DeepSeek-R1-0528 | ~280 | ~310 | ~340 | Strong early but saturates |
| Qwen3-235B-A22B | ~290 | ~330 | ~350 | Saturates after 48K tokens |

IOI 2025 gold medal threshold: ~400 points; maximum score: 600. GenCluster + gpt-oss-120b (K=5000) achieves 446.75, reaching gold-medal level.

### Ablation Study (K=5000, gpt-oss-120b)

| Method | Clustering | Score | Notes |
|--------|-----------|-------|-------|
| Random | No | 300.10 | Random solution selection |
| Longest | No | 277.36 | Select longest chain-of-thought |
| Cluster-Size | Yes | 299.87 | Rank by cluster size |
| Cluster-Majority | Yes | 314.22 | Majority voting |
| GenCluster (Random-Rep) | Yes | 406.49 | Random representative + tournament |
| GenCluster (Score-Based) | Yes | 441.11 | Average-score ranking |
| **GenCluster** | **Yes** | **446.75** | **Longest rep + win-count ranking** |

### Key Findings

- gpt-oss-120b is the only open-weight model capable of reaching gold-medal performance within 5,000 generations, with scores continuing to improve as generation count increases.
- Simple majority voting (314.22) and cluster-size ranking (299.87) perform near-randomly at IOI difficulty, as the proportion of correct solutions is extremely low.
- Tournament ranking marginally outperforms score-based ranking (446.75 vs. 441.11); selecting the longest chain-of-thought as the cluster representative outperforms random selection (446.75 vs. 406.49).
- Clustering purity (F1) improves substantially as the number of test cases increases from 10 to 100, though the number of clusters also grows, making selection harder.
- Reasoning chain length is positively correlated with correctness; gpt-oss models generate longer chains on harder problems.
- Total compute: generating 5,000 candidates requires approximately 7.3 billion tokens; tournament ranking requires an additional 7.3 billion tokens.

## Highlights & Insights

- **First open-weight model to achieve IOI gold**: The fully transparent and reproducible methodology stands in contrast to OpenAI's undisclosed approach, providing the open-source community with a concrete and replicable baseline.
- **Elegant tournament ranking design**: On difficult problems where correct solutions are extremely rare, traditional majority voting fails; the pairwise tournament mechanism better exploits LLM judgment capabilities.
- **Behavioral clustering with validator voting**: The generator–validator pipeline for producing test inputs is concise and effective; the 75% voting threshold balances coverage and quality.
- **Reasoning length as a proxy for correctness**: The simple heuristic of selecting the longest chain-of-thought as the cluster representative yields significant gains over random selection and is potentially transferable to other reasoning tasks.

## Limitations & Future Work

- Computational cost is extremely high (~14.6 billion tokens per evaluation), making the approach infeasible in resource-constrained settings.
- Ranking quality still has room for improvement — the best solution appears within the top-50 clusters for only 35 of 39 subtasks.
- Validation is limited to IOI 2025; generalization to other competitions (ICPC, Codeforces) remains to be tested.
- Training details of gpt-oss-120b are not disclosed, making reproducibility partially dependent on access to this specific model.

## Related Work & Insights

- **vs. AlphaCode/AlphaCode 2**: Both adopt large-scale generation + clustering, but AlphaCode does not address ranking under constrained submission budgets and relies on closed-source models.
- **vs. OpenAI o1-ioi**: OpenAI achieves gold medal with 1K solutions and simple selection (o3), but the method is entirely undisclosed; GenCluster achieves gold medal with 5K solutions and tournament ranking in a fully transparent manner.
- **vs. Best-of-N**: Naive Best-of-N requires reliable verifiers, which are unavailable in the IOI setting; GenCluster replaces verifiers with behavioral clustering and tournament ranking.

## Rating

- Novelty: ⭐⭐⭐⭐ Tournament ranking and round-robin submission are innovative contributions, though the overall framework (generate + cluster + select) follows the AlphaCode paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Systematic scaling analysis, multi-model comparisons, and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear and experiments are thorough, though partial reliance on the undisclosed gpt-oss model is a limitation.
- Value: ⭐⭐⭐⭐⭐ A milestone achievement as the first open-weight model to reach IOI gold-medal level, with a fully reproducible methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Efficient Test-Time Scaling via Temporal Reasoning Aggregation](efficient_test-time_scaling_via_temporal_reasoning_aggregation.md)
- [\[NeurIPS 2025\] Provable Scaling Laws for the Test-Time Compute of Large Language Models](../../NeurIPS2025/llm_reasoning/provable_scaling_laws_for_the_testtime_compute_of_large_lang.md)
- [\[NeurIPS 2025\] Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](../../NeurIPS2025/llm_reasoning/towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)
- [\[ACL 2026\] Parallel Test-Time Scaling for Latent Reasoning Models](parallel_test-time_scaling_for_latent_reasoning_models.md)
- [\[ACL 2026\] FS-Researcher: Test-Time Scaling for Long-Horizon Research Tasks with File-System-Based Agents](fs-researcher_test-time_scaling_for_long-horizon_research_tasks_with_file-system.md)

</div>

<!-- RELATED:END -->
