---
title: >-
  [Paper Note] Lessons Learned: A Multi-Agent Framework for Code LLMs to Learn and Improve
description: >-
  [NeurIPS 2025][LLM Agent][multi-agent collaboration] This paper proposes the LessonL framework, enabling multiple small LLM agents to reflect on both successful and failed cases through mutually shared "lessons," collaboratively optimizing code performance. A combination of three 7B–14B models achieves code optimization results on par with GPT-4o and approaching o3.
tags:
  - NeurIPS 2025
  - LLM Agent
  - multi-agent collaboration
  - code optimization
  - lesson mechanism
  - mutual learning
  - performance optimization
date: 2026-05-08
content_hash: c44bbe549bc8d280
---

# Lessons Learned: A Multi-Agent Framework for Code LLMs to Learn and Improve

**Conference**: NeurIPS 2025
**arXiv**: [2505.23946](https://arxiv.org/abs/2505.23946)
**Code**: [https://github.com/MITIBM-FastCoder/LessonL](https://github.com/MITIBM-FastCoder/LessonL)
**Area**: LLM Agent
**Keywords**: multi-agent collaboration, code optimization, lesson mechanism, mutual learning, performance optimization

## TL;DR
This paper proposes the LessonL framework, enabling multiple small LLM agents to reflect on both successful and failed cases through mutually shared "lessons," collaboratively optimizing code performance. A combination of three 7B–14B models achieves code optimization results on par with GPT-4o and approaching o3.

## Background & Motivation

**State of the Field**: Code optimization is a critical component of software development, yet it has been largely overlooked in AI research. Existing work either focuses on code generation or relies on specialized HPC models.

**Limitations of Prior Work**:
- LLMs exhibit complementary strengths on fine-grained tasks (e.g., Qwen7B outperforms GPT-4o by 2.5× on geometry tasks), but these advantages remain unexploited.
- Multi-agent collaboration either adopts role-based division (planner/coder/debugger) or independent proposal aggregation.
- There is a lack of interpretable knowledge-sharing mechanisms.

**Root Cause**: How to leverage the complementary strengths of multiple small LLMs for code optimization while maintaining interpretability and cost efficiency.

**Paper Goals**: Design a multi-agent learning framework that allows agents to learn from each other's successes and failures.

**Starting Point**: An analogy to peer-assisted learning — learning from textbooks plus learning from classmates.

**Core Idea**: A three-phase iterative process — lesson solicitation, lesson banking, and lesson selection — enabling interpretable optimization experience sharing among agents.

## Method

### Overall Architecture
The core loop of LessonL: initial solution generation → lesson extraction → lesson banking → lesson selection → iterative refinement using selected lessons → repeat. Over $T$ iterations, each of $n$ agents generates one lesson per round, and $k$ lessons are selected for the next round.

### Key Designs

1. **Lesson Solicitation**:

    - **Function**: Extract optimization experience in natural language form from code modification outcomes.
    - **Mechanism**: Four code modification scenarios yield distinct lesson types:
        - (a) Speedup: code is faster and correct → positive lesson (e.g., "reordering loops improves cache locality")
        - (b) Slowdown: code is correct but slower → warning lesson
        - (c) Functional error: fails tests → error lesson
        - (d) Syntax error: compilation failure → syntax lesson
    - **Design Motivation**: Learning not only from successes but also from failures to prevent repeated mistakes.

2. **Lesson Banking & Selection**:

    - **Function**: Manage the lesson pool and select the most useful lessons for the next round.
    - **Mechanism**: A hybrid selection strategy — top $\lceil k/2 \rceil$ lessons by speedup (exploitation) + top $\lfloor k/2 \rfloor$ lessons by relevance (CodeBERT cosine similarity).
    - **Design Motivation**: Selecting solely by speedup may cause trajectory fixation; the hybrid strategy introduces diversity.

3. **Dynamic Effectiveness Adjustment**:

    - **Function**: Adjust lesson weights over time.
    - **Mechanism**: An adjustment factor $f$ is defined such that when a lesson is applied in subsequent rounds, its actual speedup is compared to the original speedup, cumulatively updating $f = c/n$.
    - **Design Motivation**: A lesson that is effective at creation may not remain so; adaptive downgrading is needed for lessons that underperform in practice.

## Key Experimental Results

### Main Results

| Method | ParEval Serial Correctness | ParEval >2× Speedup Ratio | ParEval Serial Speedup | ParEval Parallel Speedup |
|--------|---------------------------|--------------------------|----------------------|------------------------|
| Qwen14B Baseline | 0.67 | 0.14 | 1.60× | 2.28× |
| GPT-4o mini | 0.77 | 0.14 | 1.57× | 2.72× |
| GPT-4o | 0.80 | 0.16 | 1.72× | 2.93× |
| OpenAI o3 | 0.77 | **0.23** | **2.21×** | **3.55×** |
| MapCoder (Qwen14B×3) | 0.88 | 0.15 | 1.85× | 3.43× |
| **LessonL (Qwen×3)** | **0.91** | **0.21** | **2.16×** | **3.46×** |

### Ablation Study

| Configuration | ParEval S Correctness | >2× Ratio | Serial Speedup | Parallel Speedup |
|---------------|-----------------------|-----------|---------------|-----------------|
| Full LessonL | 0.91 | 0.21 | **2.16×** | **3.46×** |
| High-speedup lessons only | 0.92 | 0.21 | 2.08× | 3.20× |
| High-relevance lessons only | 0.91 | 0.18 | 1.96× | 3.40× |
| No speedup adjustment | 0.92 | 0.21 | 2.05× | 3.28× |
| Random lesson selection | 0.91 | 0.19 | 2.03× | 3.47× |
| No lessons | 0.89 | 0.20 | 1.95× | 3.01× |

### Key Findings
- **Emergent capability of small models**: LessonL with three 7B–14B models reaches GPT-4o level and approaches o3, with parallel speedup of 3.46× vs. o3's 3.55×.
- The hybrid selection strategy is optimal — either high-speedup-only or high-relevance-only selection underperforms the hybrid.
- Removing the lesson mechanism leads to a significant performance drop (parallel: 3.01× vs. 3.46×), validating its necessity.
- Three agents is optimal; performance saturates or slightly degrades with six agents.
- Clear cost advantage: LessonL lies on the Pareto frontier compared to MapCoder and MoA.

## Highlights & Insights
- **Interpretable knowledge sharing**: Lessons are human-readable optimization strategies (e.g., "parallelize using OpenMP"), offering greater pedagogical value than black-box embedding fusion.
- **Adaptive lesson weighting**: Dynamic adjustment via the $f$ factor prevents blind adoption of lessons with fixed weights.
- **Learning from failure**: Negative lessons (error and slowdown cases) prevent repeated mistakes, a capability absent in MapCoder and MoA.
- **Fine-grained complementarity discovery**: The framework automatically identifies each model's strengths without prior knowledge.

## Limitations & Future Work
- Lesson extraction and iterative rounds introduce latency, limiting applicability to real-time settings.
- Validation is confined to function-level code; scaling to repository-level SWE tasks remains challenging.
- Marginal gains from lessons diminish significantly after multiple rounds.
- Automatically extracted lessons vary in quality and may contain LLM hallucinations.

## Related Work & Insights
- **vs. MapCoder/MoA**: MapCoder aggregates independent proposals; MoA employs hierarchical aggregation. LessonL introduces an explicit lesson mechanism to fill the knowledge-sharing gap.
- **vs. Self-Refine/Reflexion**: These rely on single-agent self-reflection; LessonL generalizes this to mutual learning among multiple agents.
- **vs. PIE/HPC-Coder**: These use specialized fine-tuning or retrieval; LessonL provides lightweight, on-the-fly knowledge injection.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The lesson concept is novel, with an elegant and concise design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multiple benchmarks, six ablation conditions, cost analysis, and case studies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear logical progression with vivid illustrative examples.
- **Value**: ⭐⭐⭐⭐⭐ Offers a new perspective on multi-agent learning with a clear cost-effectiveness advantage.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code](../../CVPR2026/llm_agent/nerfify_multiagent_nerf_paper_to_code.md)
- [\[NeurIPS 2025\] Distilling LLM Agent into Small Models with Retrieval and Code Tools](distilling_llm_agent_into_small_models_with_retrieval_and_co.md)
- [\[NeurIPS 2025\] R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](rd-agent-quant_a_multi-agent_framework_for_data-centric_factors_and_model_joint_.md)
- [\[NeurIPS 2025\] AgentChangeBench: A Multi-Dimensional Evaluation Framework for Goal-Shift Robustness](agentchangebench_a_multi-dimensional_evaluation_framework_for_goal-shift_robustn.md)
- [\[NeurIPS 2025\] MAT-Agent: Adaptive Multi-Agent Training Optimization](mat-agent_adaptive_multi-agent_training_optimization.md)

<!-- RELATED:END -->
