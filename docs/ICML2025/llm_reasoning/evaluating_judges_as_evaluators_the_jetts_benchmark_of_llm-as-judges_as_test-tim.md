---
title: >-
  [Paper Note] Evaluating Judges as Evaluators: The JETTS Benchmark of LLM-as-Judges as Test-Time Scaling Evaluators
description: >-
  [ICML 2025][Reasoning][LLM-as-Judge] This paper proposes the JETTS benchmark to systematically evaluate the performance of LLM-judges as evaluators in test-time scaling scenarios (response reranking, step-level beam search, and critique-based refinement). The findings show that while judges are competitive with outcome reward models in reranking, they are significantly weaker than process reward models in beam search, and natural language critiques currently fail to effective…
tags:
  - "ICML 2025"
  - "Reasoning"
  - "LLM-as-Judge"
  - "Test-Time Scaling"
  - "Reward Model"
  - "Reranking"
  - "Beam Search"
  - "Critique"
  - "JETTS"
date: 2026-05-08
content_hash: fcd8482c6a1efeb4
---

# Evaluating Judges as Evaluators: The JETTS Benchmark of LLM-as-Judges as Test-Time Scaling Evaluators

**Conference**: ICML 2025  
**arXiv**: [2504.15253](https://arxiv.org/abs/2504.15253)  
**Code**: Yes (see paper)  
**Area**: LLM Inference / Evaluation Systems / Test-Time Scaling  
**Keywords**: LLM-as-Judge, Test-Time Scaling, Reward Model, Reranking, Beam Search, Critique, JETTS  

## TL;DR

This paper proposes the JETTS benchmark to systematically evaluate the performance of LLM-judges as evaluators in test-time scaling scenarios (response reranking, step-level beam search, and critique-based refinement). The findings show that while judges are competitive with outcome reward models in reranking, they are significantly weaker than process reward models in beam search, and natural language critiques currently fail to effectively guide generator improvements.

## Background & Motivation

Test-time scaling (scaling computation at inference time) is an important paradigm for enhancing LLM performance. Its core component is the **evaluator**, which is used to judge the quality of candidate responses. Existing evaluators fall into two categories:

**Reward Models (RM)**: Non-generative, outputting scalar scores
   - Outcome RM (ORM): Scores the complete response
   - Process RM (PRM): Step-by-step scores of the reasoning process

**LLM-Judges**: Generative, outputting natural language critiques and scores

The unique advantage of an LLM-judge is its ability to generate **natural language critiques**, offering high interpretability. However, their effectiveness in test-time scaling scenarios has not been systematically evaluated. The core questions are:

- Can LLM-judges replace reward models as evaluators for test-time scaling?
- Can natural language critiques from judges help generators refine their responses?

## Method

### JETTS Benchmark Design

JETTS covers three domains and three task setups:

**Three Domains**:
- Mathematical Reasoning (GSM8K, MATH500)
- Code Generation (MBPP+, HumanEval+)
- Instruction Following (IFEval)

**Three Task Setups**:

#### 1. Response Reranking

The generator produces $N$ candidate responses, and the evaluator scores each complete response to select the best one:

$$\hat{y} = \arg\max_{y_i \in \{y_1, \ldots, y_N\}} s(y_i | x)$$

where $s(\cdot)$ is the scoring function of the evaluator. The LLM-judge generates critiques via prompting and extracts the score.

#### 2. Step-Level Beam Search

The reasoning process is decomposed into multiple steps, maintaining $k$ optimal paths at each step. The evaluator scores each step to prune low-quality paths:

$$\text{beam}_t = \text{TopK}\left(\bigcup_{b \in \text{beam}_{t-1}} \{b \oplus s_t^{(j)}\}_{j=1}^{m}, k\right)$$

This task requires the evaluator to have **process-level evaluation capability** to judge the quality of intermediate reasoning steps.

#### 3. Critique-Based Response Refinement

The judge generates a natural language critique of the initial response, which is then fed back to the generator to refine the response. This is a unique capability of LLM-judges (RMs cannot provide textual feedback):

$$y_{\text{refined}} = \text{Generator}(x, y_{\text{initial}}, \text{critique})$$

### Evaluation Scale

- **10 Judge Models**: 7B–70B parameters, including general LLMs and task-specific trained judges
- **8 Generator Models**: 6.7B–72B parameters
- Control Group: ORMs and PRMs

## Key Experimental Results

### Main Results: Reranking Performance

| Evaluator Type | Math Reasoning Acc | Code Generation Acc | Instruction Following Acc |
|-----------|------------|------------|------------|
| Random Selection (Baseline) | 45.2% | 62.1% | 55.8% |
| Outcome RM | 68.3% | 74.5% | 71.2% |
| Best LLM-Judge | 66.8% | 73.1% | 70.5% |
| Oracle (Upper Bound) | 82.1% | 86.3% | 84.7% |

**Findings**: LLM-judges achieve comparable performance to ORMs in reranking, with the gap within 2%.

### Main Results: Beam Search Performance

| Evaluator Type | Math Reasoning (beam=4) | Math Reasoning (beam=8) |
|-----------|------------------|------------------|
| Process RM | 72.5% | 76.8% |
| Best LLM-Judge | 58.3% | 61.2% |
| Outcome RM | 55.1% | 57.6% |

**Findings**: LLM-judges are **significantly weaker** than PRMs in beam search, with gaps of up to 15+ percentage points. Judges lack the fine-grained capability to evaluate intermediate reasoning steps.

### Main Results: Critique Refinement Performance

| Setup | Improvement Rate | Regression Rate | Net Effect |
|------|--------|--------|--------|
| No critique (Regeneration) | 35% | 30% | +5% |
| LLM-Judge critique | 38% | 33% | +5% |
| Oracle critique | 62% | 12% | +50% |

**Findings**: The guidance effect of natural language critiques from judges is almost identical to that without critiques. The potential of critiques for refinement remains largely untapped.

### Ablation Study

- Larger judge scale yields better reranking performance, but brings limited improvement to beam search.
- Specially trained judges exhibit superior scoring consistency compared to general LLMs.
- The specificity of critiques is positively correlated with refinement effectiveness, but current judges often generate critiques that are too general.

## Highlights & Insights

- **The first benchmark to systematically evaluate LLM-judges in test-time scaling.**
- Reveals an asymmetry between the judge's scoring ability and critique ability: they can provide reasonable scores but fail to generate effective refinement suggestions.
- The failure of judges in beam search exposes their **lack of process-level reasoning evaluation capability**, which is the core strength of PRMs.
- Suggests future directions: training judges with process-level evaluation capabilities and high-quality critique generation.

## Limitations & Future Work

- The evaluation is primarily based on accuracy metrics, without a deep-dive analysis of the judge's failure modes.
- Critique refinement only tests single-round improvements, leaving multi-round iterations unexplored.
- Performance of powerful closed-source models like GPT-4 or Claude as judges was not tested.
- The prompt design of the judge heavily influences results, and optimal prompts have not been fully explored.

## Related Work & Insights

- **LLM-as-Judge (Zheng et al., 2024)**: The paradigm of using LLMs for automated evaluation.
- **Process Reward Models (Lightman et al., 2023)**: Step-by-step annotation and scoring of reasoning steps.
- **Test-Time Scaling (Snell et al., 2024)**: Theory and practice of scaling compute at inference time.
- This work bridges the two directions of LLM-judges and test-time scaling, revealing the capability boundaries of current judges.

## Rating

⭐⭐⭐⭐ — Comprehensive and systematic experiments, clear and practically instructive conclusions, revealing key limitations of LLM-judges in test-time scaling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Scaling Evaluation-Time Compute with Reasoning Models as Evaluators](../../ACL2026/llm_reasoning/scaling_evaluation-time_compute_with_reasoning_models_as_evaluators.md)
- [\[ACL 2025\] Linguistic Generalizability of Test-Time Scaling in Mathematical Reasoning](../../ACL2025/llm_reasoning/mclm_multilingual_test_time_scaling.md)
- [\[NeurIPS 2025\] Atom of Thoughts for Markov LLM Test-Time Scaling](../../NeurIPS2025/llm_reasoning/atom_of_thoughts_for_markov_llm_testtime_scaling.md)
- [\[NeurIPS 2025\] Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling](../../NeurIPS2025/llm_reasoning/rethinking_optimal_verification_granularity_for_compute-efficient_test-time_scal.md)
- [\[NeurIPS 2025\] SolverLLM: Solving Optimization Problems via Test-Time Scaling with LLM-Guided Search](../../NeurIPS2025/llm_reasoning/solverllm_leveraging_test-time_scaling_for_optimization_problem_via_llm-guided_s.md)

</div>

<!-- RELATED:END -->
