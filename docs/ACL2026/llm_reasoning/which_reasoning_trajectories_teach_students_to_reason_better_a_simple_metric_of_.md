---
title: >-
  [Paper Note] Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment
description: >-
  [ACL 2026][LLM Reasoning][Knowledge Distillation] This paper proposes the Rank-Surprisal Ratio (RSR), a metric that evaluates training data suitability by jointly measuring the "informativeness" and "alignment" of reason…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Knowledge Distillation"
  - "Reasoning Trajectories"
  - "Data Selection"
  - "Chain-of-Thought"
  - "Large Language Models"
date: 2026-05-08
content_hash: d1bb2f80fba61278
---

# Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment

**Conference**: ACL 2026  
**arXiv**: [2601.14249](https://arxiv.org/abs/2601.14249)  
**Code**: [GitHub](https://github.com/UmeanNever/RankSurprisalRatio)  
**Area**: Model Compression  
**Keywords**: Knowledge Distillation, Reasoning Trajectories, Data Selection, Chain-of-Thought, Large Language Models

## TL;DR

This paper proposes the Rank-Surprisal Ratio (RSR), a metric that evaluates training data suitability by jointly measuring the "informativeness" and "alignment" of reasoning trajectories for a student model. RSR achieves an average Spearman correlation of 0.86 with post-training performance across 55 student-teacher combinations (5 students, 11 teachers) and is successfully applied to trajectory and teacher selection.

## Background & Motivation

**Background**: Long Chain-of-Thought (CoT) trajectories have become the primary means of distilling reasoning capabilities from large models to small student models, where student models learn the teacher's reasoning process through SFT.

**Limitations of Prior Work**: Experiments repeatedly show that stronger teacher models (e.g., DeepSeek-R1 671B) do not necessarily yield better students. The "suitability" between data and the student model is the key factor for distillation success. However, existing methods primarily rely on the student model's log-probability, which tends to select high-probability trajectories already familiar to the student, ignoring data with actual learning value.

**Key Challenge**: The trade-off between informativeness and alignment—data that is too familiar lacks learning value, while data that is too unfamiliar cannot be learned. This echoes the "Zone of Proximal Development" in psychology: effective learning materials should be slightly beyond the learner's current capacity but still comprehensible.

**Goal**: Design a simple, interpretable metric to measure the suitability of reasoning trajectories for specific student models, balancing informativeness and alignment.

**Key Insight**: The authors observe that effective trajectories exhibit a specific pattern—their tokens have low absolute probability under the student model (high surprisal, indicating the student would not generate them), yet remain highly ranked in the vocabulary (low rank, indicating they are within the student's comprehension). This "absolutely unfamiliar but relatively familiar" characteristic perfectly balances informativeness and alignment.

**Core Idea**: Use the ratio of token rank to surprisal (RSR) to measure trajectory suitability—the lower the RSR, the more informative and aligned the trajectory is for the student.

## Method

### Overall Architecture

The method consists of three steps: (1) Perform a single forward pass of the student model on a given reasoning trajectory to obtain the probability distribution for each token; (2) Calculate the surprisal (negative log-likelihood) and rank (vocabulary position) for each token; (3) Define the trajectory-level RSR as the ratio of average rank to average surprisal to evaluate suitability. RSR requires only one forward pass and no additional verifiers or test data.

### Key Designs

1.  **Bi-modal Distribution Theoretical Model**:
    *   Function: Provides a theoretical explanation for RSR's effectiveness.
    *   Mechanism: Models the student's token prediction distribution as a bi-modal mixture—mode $Z_A$ represents the student's dominant generation pattern (high probability, low rank), and mode $Z_B$ represents patterns deviating from the dominant mode but still within the student's knowledge (low probability, but still low rank). Effective teacher trajectories should correspond to $Z_B$. Simulations confirm that $Z_B$ trajectories have the lowest RSR (1.30), while misaligned $Z_C$ trajectories have the highest (2.23).
    *   Design Motivation: Explains why surprisal or rank alone is insufficient—the ratio is needed to distinguish "informative and aligned" ($Z_B$) from "informative but misaligned" ($Z_C$) trajectories.

2.  **Surprisal-weighted Aggregation**:
    *   Function: Stably aggregates token-level RSR into a trajectory-level metric.
    *   Mechanism: Directly averaging token-level $\text{RSR}_\text{token} = \text{Rank}(t_k) / \text{Surprisal}(t_k)$ causes numerical instability (ratio explodes as surprisal approaches zero for high-prob tokens). The authors use surprisal-weighted averaging, which simplifies to $\text{RSR}(\mathbf{x}) = \sum_k \text{Rank}(t_k) / \sum_k \text{Surprisal}(t_k)$, equivalent to average rank divided by average surprisal.
    *   Design Motivation: Tokens with higher surprisal have a greater impact on student learning, making it reasonable to emphasize them; removing this weighting drops correlation from 0.856 to 0.391.

3.  **Rank Truncation**:
    *   Function: Handles numerical instability caused by extremely unfamiliar tokens.
    *   Mechanism: Truncates rank values to $r_{max}$ (default 100), i.e., $\min(\text{Rank}(t_k), r_{max})$. Extremely unfamiliar tokens may have ranks reaching the vocabulary size (e.g., 128K); these are essentially indistinguishable noise to the student.
    *   Design Motivation: Removing truncation drops correlation from 0.856 to 0.700; results remain robust for $r_{max}$ in the 100~500 range.

### Loss & Training

RSR is a data selection metric rather than a training method. In practice, RSR is calculated for candidate trajectories, and those with the lowest RSR are selected for SFT. Training uses the standard supervised fine-tuning loss.

## Key Experimental Results

### Main Results (Correlation Analysis)

| Metric | Qwen-3-14B | LLaMA-3.1-8B | Qwen-2.5-7B | Qwen-3-4B | Qwen-2.5-3B | Average |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Teacher Params | 0.04 | 0.34 | 0.20 | 0.02 | 0.26 | 0.01 |
| Avg-Surprisal | 0.24 | 0.42 | 0.55 | 0.55 | 0.70 | 0.49 |
| GRACE | 0.25 | 0.58 | 0.66 | 0.75 | 0.69 | 0.59 |
| **RSR (Ours)** | **0.85** | **0.85** | **0.92** | **0.82** | **0.85** | **0.86** |

### Ablation Study

| Configuration | Average Correlation | Gain |
| :--- | :--- | :--- |
| RSR ($r_{max}=100$) | 0.856 | - |
| Without rank truncation | 0.700 | -0.156 |
| Without weighted average (Avg-RSRtoken) | 0.391 | -0.465 |
| Filtered average (top 30%) | 0.793 | -0.064 |
| $r_{max}=500$ | 0.822 | -0.034 |
| Using only 200 samples | 0.864 | +0.007 |

### Key Findings
- RSR significantly outperforms all baselines across all 5 student models, with an average Spearman correlation of 0.86 (second best is Rule-based Quality at 0.65).
- Surprisal weighting is the most critical design; removing it causes correlation to plummet by 0.465.
- RSR is insensitive to sample size, achieving the same effect with target data of only 200 samples (4% of the original set).
- In trajectory selection tasks, data selected by RSR achieves training results comparable to or better than brute-force searching all teachers for the optimal result.

## Highlights & Insights

- **Refined insight of "absolutely unfamiliar + relatively familiar"**: Transforming the conflict between informativeness and alignment into a comparison between surprisal and rank dimensions. This observation guides the RSR design and provides a new perspective for distillation data selection.
- **Elegant mathematical simplification**: Starting from token-level weighted averages and deriving a simple "average rank / average surprisal" form, which is computationally efficient (single forward pass) and highly interpretable.
- **Transfer potential**: The core idea—measuring the "learnability" of data for a model—can be generalized to data selection for any SFT scenario beyond reasoning tasks.

## Limitations & Future Work

- Currently validated only on mathematical reasoning tasks; systematic testing on code generation and general dialogue is missing.
- RSR requires a forward pass to calculate ranks for every trajectory, which may incur significant computational costs for massive datasets (millions of samples).
- It considers only the suitability of individual trajectories and does not model diversity or complementarity between trajectories (subset selection).
- While the discussion mentions potential use for non-CoT data, experimental support is currently limited.

## Related Work & Insights

- **vs. Avg-Surprisal (Zhang et al.)**: Measuring suitability only via probability tends to select data already familiar to the student (correlation 0.49); RSR solves this "informativeness blind spot" by introducing the rank dimension.
- **vs. GRACE (Li et al.)**: A gradient-based method requiring extra compute for gradients (correlation 0.59); RSR is simpler (forward pass only) and more effective.
- **vs. Influence Score**: Inspired by influence functions but shows unstable performance across different students; RSR demonstrates consistent performance across students.

## Rating

- Novelty: ⭐⭐⭐⭐ Clear and elegant core insight, though essentially a combination of two existing metrics.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale experiments with 5 students × 11 teachers, detailed ablations, and two downstream application validations.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely smooth logical flow from observation to theory to experimentation.
- Value: ⭐⭐⭐⭐ Direct practical value for reasoning distillation data selection, though the scope of application requires further expansion.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals](llm_reasoning_as_trajectories_step-specific_representation_geometry_and_correctn.md)
- [\[ACL 2026\] Do Not Step Into the Same River Twice: Learning to Reason from Trial and Error](do_not_step_into_the_same_river_twice_learning_to_reason_from_trial_and_error.md)
- [\[NeurIPS 2025\] Reasoning Models Better Express Their Confidence](../../NeurIPS2025/llm_reasoning/reasoning_models_better_express_their_confidence.md)
- [\[ACL 2026\] Chain-of-Thought as a Lens: Evaluating Structured Reasoning Alignment between Human Preferences and Large Language Models](chain-of-thought_as_a_lens_evaluating_structured_reasoning_alignment_between_hum.md)
- [\[ACL 2026\] Discovering a Shared Logical Subspace: Steering LLM Logical Reasoning via Alignment of Natural-Language and Symbolic Views](discovering_a_shared_logical_subspace_steering_llm_logical_reasoning_via_alignme.md)

</div>

<!-- RELATED:END -->
