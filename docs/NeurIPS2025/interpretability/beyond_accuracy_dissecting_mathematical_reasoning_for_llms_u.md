---
title: >-
  [Paper Note] Beyond Accuracy: Dissecting Mathematical Reasoning for LLMs Under Reinforcement Learning
description: >-
  [NeurIPS 2025][Interpretability][SPARKLE] This paper proposes SPARKLE, a three-axis analytical framework (plan following, knowledge integration…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "SPARKLE"
  - "GRPO"
  - "plan following"
  - "knowledge integration"
  - "subproblem decomposition"
  - "multi-stage RL"
date: 2026-05-08
content_hash: e07b9f732b77ae89
---

# Beyond Accuracy: Dissecting Mathematical Reasoning for LLMs Under Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2506.04723](https://arxiv.org/abs/2506.04723)
**Code**: [https://sparkle-reasoning.github.io/](https://sparkle-reasoning.github.io/)
**Area**: Interpretability
**Keywords**: SPARKLE, GRPO, plan following, knowledge integration, subproblem decomposition, multi-stage RL

## TL;DR
This paper proposes SPARKLE, a three-axis analytical framework (plan following, knowledge integration, subproblem decomposition) for fine-grained dissection of how RL shapes LLM reasoning behavior. The analysis reveals that RL primarily enhances knowledge integration and planning flexibility rather than plan execution. The paper further introduces SparkleRL-PSS, a multi-stage RL training pipeline that effectively exploits hard problem data via partial step scaffolding.

## Background & Motivation

**Background**: RL (particularly GRPO) has become the dominant paradigm for improving LLM reasoning. Models such as DeepSeek-R1 and OpenAI o1 have achieved substantial gains on benchmarks including AIME and MATH.

**Limitations of Prior Work**: Nearly all existing work tracks only accuracy improvements, lacking fine-grained understanding of what capabilities RL actually enhances. Whether RL improves planning, execution, knowledge retrieval, or problem decomposition remains unclear, making it difficult to target improvements in RL pipelines.

**Key Challenge**: Hard problems typically yield no positive reward signal (the model fails across 20 attempts) and are therefore filtered out. Discarding hard problems, however, wastes valuable training signal. The question of how to effectively exploit such problems remains open.

**Goal**: (1) Establish a fine-grained analytical framework beyond accuracy to reveal the specific impact of RL on each reasoning dimension; (2) Design a multi-stage RL training scheme that leverages hard problem data.

**Key Insight**: Drawing on cognitive-science theories of human problem solving (Newell & Simon, 1972), the paper decomposes reasoning into three core dimensions — planning, knowledge, and decomposition — and designs controlled experiments for each.

**Core Idea**: By providing or withholding auxiliary information such as plans, knowledge annotations, and subproblem decompositions, the framework identifies the specific reasoning dimensions strengthened by RL, and uses these findings to motivate a partial step scaffolding training strategy.

## Method

### Overall Architecture
SPARKLE comprises two components: (1) a three-axis analytical framework that constructs an augmented dataset annotated with planning skeletons, knowledge annotations, and subproblem chains to compare model behavior before and after RL; and (2) SparkleRL-PSS, a two-stage GRPO training pipeline whose second stage reuses hard problems via partial step scaffolding.

### Key Designs

1. **Three-Axis Analytical Framework**:

    - **Axis 1 (Planning & Execution)**: A planning skeleton is generated for each problem (e.g., "Step 1: Analyze properties of modular arithmetic; Step 2: Detect periodic patterns…"). Comparing model performance with and without the plan disentangles planning ability from execution ability.
    - **Axis 2 (Knowledge Integration)**: Facts, theorems, and lemmas required for each problem (e.g., Fermat's Little Theorem, Chinese Remainder Theorem) are extracted. Comparing performance with and without this knowledge disentangles knowledge retrieval from reasoning ability.
    - **Axis 3 (Subproblem Decomposition)**: Each problem is decomposed into a chain of subproblems (Q1→Q2→Q3…). Answers to previously solved subproblems are provided incrementally to identify where the reasoning chain breaks.

2. **SPARKLE Benchmark Construction**:

    - **Function**: Augments 2,564 problems from AIME24, AMC23, MATH500, GSM8K, and OlympiadBench.
    - **Mechanism**: GPT-4.1 with a Web Agent generates planning skeletons, knowledge annotations, and subproblem chains for each problem; a second GPT-4.1 instance validates the outputs; graduate-level mathematics experts then perform manual review.
    - **Annotations**: Each problem is labeled with an AoPS difficulty level (1–10) and a mathematical domain (9 categories).

3. **SparkleRL-PSS Multi-Stage Training**:

    - **Stage 1**: Standard GRPO training of Qwen-2.5-Math-7B on the 40K math problems from DeepScaleR.
    - **Stage 2**: Hard problems on which the Stage 1 model fails across 20 attempts are identified (6.5K problems, of which 5.7K are verified). The reference solution for each problem is divided into 4 semantic blocks, yielding input variants with 0–4 hint levels. The model must continue reasoning from a partially provided solution. The KL coefficient is increased from 0.001 to 0.01 to prevent excessive deviation.
    - **Design Motivation**: Rather than generating new data, partial scaffolding enables the model to obtain positive reward signals even on hard problems.

### Loss & Training
- Standard GRPO objective with rule-based rewards: correct answer + correct format → 2 points; correct answer + incorrect format → 1 point; incorrect answer → −1 point.
- Stage 1: lr = 1e-6, KL = 0.001.
- Stage 2: lr = 1e-6, KL = 0.01, temperature = 0.6, 32 samples per problem.
- Hardware: 8×H200 + 15×A100-40G + 9×A100-SXM4-40G.

## Key Experimental Results

### Main Results

| Model | AIME24 | AMC23 | MATH500 | GSM8K | OlympiadBench | Avg. |
|---|---|---|---|---|---|---|
| Qwen-2.5-Math-7B (Base) | 16.67 | 42.50 | 44.03 | 42.53 | 28.65 | 35.23 |
| SparkleRL-Stage 1 | 46.67 | 67.50 | 80.00 | 91.77 | 39.11 | 65.01 |
| SparkleRL-Stage 2-hard | 41.67 | 65.94 | 80.50 | 92.45 | 37.39 | 63.59 |
| SparkleRL-Stage 2-mix | 40.00 | 63.44 | 80.78 | 92.52 | 38.85 | 63.12 |
| **SparkleRL-Stage 2-pss** | **50.42** | **71.25** | **81.00** | **92.38** | **40.11** | **67.03** |

Stage 2-pss achieves an average gain of 2.02% over Stage 1, reaching 50.42% on AIME24 — comparable to 32B-scale models. SFT on hard problems causes severe degradation (AIME24: 46.67% → 15.00%).

### Ablation Study (Core Findings from the Three-Axis Analysis)

| Analysis Axis | Base Model | RL Model | Key Difference |
|---|---|---|---|
| +Plan | Performance drops on 4/5 benchmarks, avg. −5.7% | Stable or marginal gain (except AIME24 −2.5%) | RL models are more flexible; human-written plans may mislead them |
| +Knowledge | Avg. −5.4% | Avg. +4.3% | Base models cannot integrate external knowledge; RL substantially improves knowledge utilization |
| Subproblem (SSR) | AIME24: 3.3% SSR vs. 16.7% full acc. | AIME24: 17.5% SSR vs. 50.4% full acc. | All models are far weaker at resolving subproblems step-by-step than at solving full problems |

### Key Findings
- **RL does not primarily enhance plan execution**: Providing RL models with human-authored correct plans can actually reduce performance (AIME24: 50.4% → 47.9%). RL models are more adept at generating their own internal strategies; external plans may conflict with learned heuristics.
- **RL significantly enhances knowledge integration**: Base models perform worse when given knowledge (−5.4%), as they fail to integrate it; RL models benefit substantially (+4.3%), with larger gains at higher difficulty levels (+42.5% knowledge gain at level 8).
- **Subproblem decomposition remains a bottleneck**: Even when hard problems are broken into smaller steps with intermediate answers provided, models still fail at certain steps. This suggests that the "shortcut" strategies learned under RL are inconsistent with rigorous step-by-step reasoning.
- **Hard problems can be effectively exploited**: Partial step scaffolding enables reward signals on hard problems and outperforms both hard-only and mixed training.
- **SFT cannot substitute RL in Stage 2**: SFT on noisy traces causes severe degradation, as SFT encourages memorization whereas RL promotes generalization.

## Highlights & Insights
- **Empirical evidence that "knowledge > planning"**: For RL models, providing external knowledge yields substantially larger gains than providing a plan. This suggests that the reasoning bottleneck in RL models lies more in *what they know* than in *how they reason*, offering guidance for combining RAG with reasoning.
- **The counterintuitive finding that human plans can be harmful** is particularly valuable: it indicates that RL models develop their own distinctive internal reasoning strategies. High-level plans are helpful, but step-by-step plans can be detrimental — a finding with direct implications for prompt engineering.
- **Partial step scaffolding requires no additional data generation**: Simply segmenting existing reference solutions into fragments suffices to guide model exploration on hard problems. This represents a low-cost, high-return curriculum learning design.

## Limitations & Future Work
- Validation is limited to mathematical reasoning; adapting the analytical framework to code reasoning, logical reasoning, and other domains requires further work.
- SPARKLE dataset construction relies on GPT-4.1 and manual review, limiting scalability.
- All findings are empirical; theoretical explanations for why RL enhances knowledge integration rather than plan execution are lacking.
- The four-block segmentation in Stage 2-pss is fixed; adaptive segmentation strategies may yield further improvements.

## Related Work & Insights
- **vs. Yue et al. (2025) "RL does not create new capabilities"**: Yue et al. argue that RL primarily reweights existing reasoning paths. SPARKLE's analysis is more fine-grained, finding that RL genuinely enhances knowledge integration mechanisms — a phenomenon not reducible to mere path reweighting.
- **vs. ARM / Controlling Thinking Speed**: These works focus on macro-level control of reasoning efficiency, whereas SPARKLE conducts micro-level dissection of reasoning capabilities. The two approaches are complementary: understanding what RL enhances should precede decisions about how to regulate it.
- **vs. DeepScaleR**: DeepScaleR scales RL training to improve small model performance. SparkleRL-PSS demonstrates that, on the same data, curriculum design (rather than simple data filtering) yields further gains.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — SPARKLE is the first work to systematically dissect the impact of RL on individual reasoning dimensions, yielding multiple counterintuitive findings.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Five benchmarks, two model scales (7B/32B), comparisons across SFT/RL/multi-stage RL, and statistical significance tests.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with insightful findings; the appendix is overly long.
- Value: ⭐⭐⭐⭐⭐ — Makes an important contribution to understanding the fundamental mechanisms of RL + reasoning; partial step scaffolding has strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Learnable Fractional Reaction-Diffusion Dynamics for Under-Display ToF Imaging and Beyond](../../ICCV2025/interpretability/learnable_fractional_reaction-diffusion_dynamics_for_under-display_tof_imaging_a.md)
- [\[ICLR 2026\] GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](../../ICLR2026/interpretability/gepa_reflective_prompt_evolution_can_outperform_reinforcement_learning.md)
- [\[ACL 2026\] Curing "Miracle Steps" in LLM Mathematical Reasoning with Rubric Rewards](../../ACL2026/interpretability/curing_miracle_steps_in_llm_mathematical_reasoning_with_rubric_rewards.md)
- [\[NeurIPS 2025\] Dynamic Algorithm for Explainable k-medians Clustering under lp Norm](dynamic_algorithm_for_explainable_k-medians_clustering_under_lp_norm.md)
- [\[NeurIPS 2025\] Evaluating LLMs in Open-Source Games](evaluating_llms_in_open-source_games.md)

</div>

<!-- RELATED:END -->
