---
title: >-
  [Paper Note] HPSS: Heuristic Prompting Strategy Search for LLM Evaluators
description: >-
  [ACL 2025][LLM Evaluation][Prompting Strategy Search] By integrating 8 key factors affecting LLM evaluation prompts (scoring scale, ICL examples, evaluation criteria, reference answer, CoT, AutoCoT, evaluation metrics, and component order), this work proposes HPSS, a genetic-algorithm-based heuristic prompting strategy search method. HPSS efficiently searches for the optimal prompting strategy within a search space of 12,960 combinations, outperforming G-Eval and CloserLook a…
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Prompting Strategy Search"
  - "Genetic Algorithm"
  - "Heuristics"
  - "Automatic Evaluation Optimization"
date: 2026-05-08
content_hash: 850399c3fd372a05
---

# HPSS: Heuristic Prompting Strategy Search for LLM Evaluators

**Conference**: ACL 2025  
**arXiv**: [2502.13031](https://arxiv.org/abs/2502.13031)  
**Code**: [https://github.com/thu-coai/HPSS](https://github.com/thu-coai/HPSS)  
**Area**: LLM Evaluation  
**Keywords**: LLM Evaluation, Prompting Strategy Search, Genetic Algorithm, Heuristics, Automatic Evaluation Optimization

## TL;DR

By integrating 8 key factors affecting LLM evaluation prompts (scoring scale, ICL examples, evaluation criteria, reference answer, CoT, AutoCoT, evaluation metrics, and component order), this work proposes HPSS, a genetic-algorithm-based heuristic prompting strategy search method. HPSS efficiently searches for the optimal prompting strategy within a search space of 12,960 combinations, outperforming G-Eval and CloserLook at only 5% of the baseline generation cost.

## Background & Motivation

**Rise of LLM Evaluation**: With the improvement of LLM-generated content quality, using LLMs as automatic evaluators (LLM-as-a-Judge) has become a mainstream trend. However, evaluation performance heavily depends on prompt design.

**Fragmented Research on Prompting Factors**: Existing works (e.g., G-Eval, CloserLook) focus individually on subset factors in evaluation prompts (such as CoT or grading criteria), lacking a systematic integration of all key factors.

**Combinatorial Explosion**: Considering 8 prompting factors simultaneously yields a search space of up to 12,960 combinations, making brute-force search infeasible.

**Cost-Efficiency Trade-offs**: Each evaluation requires an LLM inference call, meaning large-scale search incurs extremely high computational costs, which demands efficient search strategies.

**Varying Optimal Strategies Across Tasks**: Optimal prompting strategies can differ significantly across various evaluation scenarios (e.g., summarization, dialogue, translation), necessitating automated strategy adaptation.

**Lack of a Unified Framework**: Currently, there is no unified framework that organically integrates prompting strategy search with LLM evaluation.

## Method

### Overall Architecture

HPSS models the LLM evaluation prompt design as a combinatorial optimization problem: it defines a discrete search space over 8 prompting factors and utilizes a Genetic Algorithm (GA) combined with a heuristic evaluation function to search for the optimal prompting strategy configuration within a limited computational budget.

### Key Designs

#### Key Design 1: 8 Core Prompting Factors System

This work systematically identifies and integrates 8 key factors affecting LLM evaluation quality:

| Factor | Description | Number of Options |
|------|------|--------|
| Scoring Scale | 1-5 / 1-10 / continuous score | 3 |
| In-Context Examples (ICL) | With/Without examples | 2 |
| Criteria | Specific / Abstract / None | 3 |
| Reference | With/Without reference | 2 |
| Chain-of-Thought (CoT) | Enabled/Disabled | 2 |
| AutoCoT | Automatically generate reasoning steps | 2 |
| Metrics | Selection of different metrics | Multiple |
| Component Order | Arrangement of factors in the prompt | Multiple permutations |

The total search space consists of 12,960 combinations.

#### Key Design 2: Genetic Algorithm Search

Strategy search is conducted by mirroring the evolutionary mechanism of genetic algorithms:
- **Initialization**: Randomly sample an initial population (a set of prompting strategies).
- **Selection**: Select elite individuals based on fitness (evaluation correlation).
- **Crossover**: Swap partial factor configurations between two strategies.
- **Mutation**: Randomly alter a specific factor setting inside an individual.
- **Iteration**: Repeat the selection-crossover-mutation process until convergence.

#### Key Design 3: Heuristic Evaluation Function

A lightweight heuristic function is designed to rapidly evaluate the quality of each prompting strategy, avoiding the need for full LLM inference evaluations on every candidate strategy. The heuristic function estimates the potential of a strategy based on the evaluation correlation on a small sample, significantly reducing the search cost.

#### Key Design 4: Cost-Aware Search

Cost constraints are introduced during the search to ensure that the total number of LLM calls remains within budget. Candidate strategies are pre-screened via the heuristic function, and full evaluation is performed only on high-potential strategies.

### Loss & Training

The correlation coefficient (e.g., Spearman/Kendall correlation) between LLM evaluation scores and human ratings is used as the optimization goal. The search aims to find the prompting strategy combination that maximizes this correlation.

## Key Experimental Results

### Main Results: Evaluation Quality on MT-Bench

| Method | Correlation with Human Rating | Relative Gain | Generation Cost |
|------|------------------|----------|----------------|
| MT-Bench Baseline | 1.00x | — | 1.00x |
| G-Eval | Higher | — | ~20x |
| CloserLook | Higher | — | ~20x |
| **HPSS (Ours)** | **Highest** | **+29.4%** | **0.05x (5%)** |

### Ablation Study: Impact of Factors

| Ablation Setting | Impact |
|----------|------|
| Remove scoring scale search | Significant performance drop |
| Remove CoT factor | Obvious drop on some tasks |
| Remove component order search | Minor performance drop |
| Fix all factors (single strategy) | Substantial drop |

### Key Findings

1. HPSS achieves a 29.4% relative performance improvement over the default prompting strategy of MT-Bench, while maintaining a generation cost of only about 5% of G-Eval and CloserLook.
2. The optimal prompting strategies vary significantly across different evaluation tasks: for instance, CoT is beneficial in summarization evaluation but can sometimes be counterproductive in dialogue evaluation.
3. The impact of component order on evaluation quality is heavily underestimated—simply rearranging the order of factors in the prompt can lead to significant differences in performance.
4. The genetic algorithm typically converges to a near-optimal solution within an exploration of only 5-10% of the search space.

## Highlights & Insights

1. **Systematic Perspective**: This is the first work to unify 8 key prompting factors into a single search framework, avoiding the local optimum of tuning parameters individually.
2. **Extreme Cost Efficiency**: Achieving superior performance at just 5% of the generation cost of competitive methods (which cost 20x more) demonstrates the immense advantage of an intelligent search strategy over brute-force methods.
3. **Transferability**: The searched optimal strategies demonstrate a degree of transferability across different LLM evaluators.
4. **Practical Value**: It directly addresses the engineering challenge of "how to design prompts" in LLM evaluation, providing a plug-and-play framework.

## Limitations & Future Work

1. **Constrained Search Space**: The current 8 factors are manually defined, which may omit other crucial factors (such as prompt style, level of difficulty of examples, etc.).
2. **Dependence on Initial Population**: The results of the genetic algorithm can be influenced by the initial random population, requiring multiple runs to ensure optimality.
3. **Task Generalizability**: Validation is primarily performed on standard benchmarks; efficacy in open-ended scenarios (e.g., creative writing, code review) remains to be explored.
4. **Lack of Dynamic Adaption**: Once search is complete, the policy remains static and cannot dynamically adapt to specific evaluation samples.
5. **Heuristic Function Precision**: The lightweight heuristic function may misjudge strategy quality under some edge cases.

## Related Work & Insights

- **G-Eval**: Employs CoT and multi-step evaluation to improve LLM scoring quality, but does not systematically search the prompt factor combinations.
- **CloserLook**: Conducts an in-depth analysis of the impact of grading criteria and reference answers on evaluation, but employs a fixed strategy.
- **Auto-Arena**: Conducts evaluation through LLM pairwise battles, representing a complementary direction to prompting strategy search.
- **Insights**: This work implies that in all scenarios involving LLM prompt design (not limited to evaluation), systematic factor analysis combined with efficient search represents a far more effective paradigm than manual tuning.

## Rating

- Novelty: ⭐⭐⭐⭐ — Introducing a genetic algorithm to search evaluation prompts is both novel and reasonable.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Thorough validation on multiple benchmarks and rich ablation analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear factor framework and complete method description.
- Value: ⭐⭐⭐⭐⭐ — Highly practical with direct value to the LLM evaluation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] OptiTree: Hierarchical Thoughts Generation with Tree Search for LLM Optimization Modeling](../../NeurIPS2025/llm_evaluation/optitree_hierarchical_thoughts_generation_with_tree_search_for_llm_optimization_.md)
- [\[ACL 2025\] Browsing Lost Unformed Recollections: A Benchmark for Tip-of-the-Tongue Search and Reasoning](browsing_lost_unformed_recollections_a_benchmark_for_tip-of-the-tongue_search_an.md)
- [\[ACL 2025\] Something's Fishy In The Data Lake: A Critical Re-evaluation of Table Union Search Benchmarks](somethings_fishy_in_the_data_lake_a_critical_re-evaluation_of_table_union_search.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)
- [\[ICLR 2026\] AutoMetrics: Approximate Human Judgments with Automatically Generated Evaluators](../../ICLR2026/llm_evaluation/autometrics_approximate_human_judgments_with_automatically_generated_evaluators.md)

</div>

<!-- RELATED:END -->
