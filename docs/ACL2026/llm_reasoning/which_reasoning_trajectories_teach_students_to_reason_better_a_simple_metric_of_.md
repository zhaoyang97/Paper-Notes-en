---
title: >-
  [Paper Note] Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment
description: >-
  [ACL 2026][LLM Reasoning][Knowledge Distillation] Ours proposes the Rank-Surprisal Ratio (RSR) metric, which evaluates training data suitability by jointly measuring the "informativeness" and "alignment" of reasoning trajectories for a student model. RSR achieves an average Spearman correlation of 0.86 with post-training performance across 5 student and 11 teacher mod
tags:
  - ACL 2026
  - LLM Reasoning
  - Knowledge Distillation
date: 2026-05-08
content_hash: fb3a334eed36b0a0
---
# Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment

**Conference**: ACL 2026  
**arXiv**: [2601.14249](https://arxiv.org/abs/2601.14249)  
**Code**: [GitHub](https://github.com/UmeanNever/RankSurprisalRatio)  
**Area**: Model Compression  
**Keywords**: Knowledge Distillation, Reasoning Trajectories, Data Selection, Chain-of-Thought, Large Language Models

## TL;DR

Ours proposes the Rank-Surprisal Ratio (RSR) metric, which evaluates training data suitability by jointly measuring the "informativeness" and "alignment" of reasoning trajectories for a student model. RSR achieves an average Spearman correlation of 0.86 with post-training performance across 5 student and 11 teacher model combinations and is successfully applied to trajectory and teacher selection.

## Background & Motivation

**Background**: Long Chain-of-Thought (CoT) trajectories have become the primary means of distilling reasoning capabilities from large reasoning models to smaller student models via SFT, where student models learn the reasoning processes of teachers.

**Limitations of Prior Work**: Experiments consistently demonstrate that stronger teacher models (e.g., DeepSeek-R1 671B) do not necessarily yield better students. The "suitability" between data and student models is the key factor for distillation effectiveness. Existing methods primarily rely on the student model's log-probability to measure suitability, which tends to select high-probability trajectories familiar to the student, ignoring data with genuine learning value.

**Key Challenge**: The trade-off between informativeness and alignment—data that is too familiar lacks learning value, while data that is too unfamiliar is unlearnable. This echoes the concept of the "Zone of Proximal Development" in psychology: the most effective learning materials should be slightly beyond the learner's current capacity but not entirely incomprehensible.

**Goal**: Design a simple, interpretable metric to measure the suitability of reasoning trajectories for a specific student model, balancing both informativeness and alignment.

**Key Insight**: Effective trajectories exhibit a specific pattern—their tokens have low absolute probability under the student model (high surprisal, indicating they are not what the student would naturally generate), yet they still rank high in the vocabulary distribution (low rank, indicating they are within the student's understanding). This "absolutely unfamiliar but relatively familiar" characteristic balances informativeness and alignment.

**Core Idea**: Use the ratio of token rank to surprisal (RSR) to measure trajectory suitability—the lower the RSR, the more the trajectory is both informative and aligned with the student.

## Method

### Overall Architecture

The method consists of three steps: (1) Perform a forward pass of the student model on a given reasoning trajectory to obtain the probability distribution for each token; (2) Calculate the surprisal (negative log-likelihood) and rank (position in the vocabulary distribution) for each token; (3) Define the trajectory-level RSR as the ratio of average rank to average surprisal to evaluate data suitability. RSR requires only a single forward pass and no additional verifiers or test data.

### Key Designs

**1. Dual-Modal Distribution Theory: Why "Absolutely Unfamiliar but Relatively Familiar" identifies good material**

Relying solely on surprisal treats all content the student has not seen as valuable; relying solely on rank selects high-frequency content the student already knows. A single dimension cannot distinguish between "informative and aligned" and "informative but unaligned." The authors model the student model's token prediction distribution as a dual-modal mixture: the primary mode $Z_A$ represents the student's dominant generation patterns (high probability, low rank), while the secondary mode $Z_B$ represents patterns that deviate from the dominant mode but remain within the student's knowledge (low probability, but still low rank).

Effective teacher trajectories should correspond to $Z_B$—low absolute probability (informative because the student wouldn't generate it) but relatively high rank (aligned because the student can understand it). Simulations confirm this: $Z_B$ trajectories yield the lowest RSR (1.30), while misaligned $Z_C$ trajectories yield the highest RSR (2.23). This explains why RSR uses the ratio of rank to surprisal—only by considering both can $Z_B$ be isolated from $Z_C$.

**2. Surprisal-Weighted Aggregation: Stabilizing token-level ratios into a trajectory-level metric**

The token-level RSR is defined as $\text{RSR}_\text{token} = \text{Rank}(t_k) / \text{Surprisal}(t_k)$. However, a simple arithmetic average across tokens would lead to numerical instability as surprisal approaches zero for high-probability tokens. Ours uses a surprisal-weighted average, which simplifies to:

$$\text{RSR}(\mathbf{x}) = \frac{\sum_k \text{Rank}(t_k)}{\sum_k \text{Surprisal}(t_k)}$$

This "average rank divided by average surprisal" avoids division-by-zero issues and naturally assigns higher weights to tokens with higher surprisal, which have a greater impact on student learning. Ablation studies show that removing this weighting causes the correlation to plummet from 0.856 to 0.391.

**3. Rank Truncation: Flattening noise from extremely unfamiliar tokens**

For extremely unfamiliar tokens, the rank can reach the vocabulary size (e.g., 128K). For a student model, ranks of 50,000 and 120,000 are functionally identical—both indicate "complete unfamiliarity." Retaining these large raw values introduces noise. Consequently, rank is truncated at $r_{max}$ (defaulting to 100), calculated as $\min(\text{Rank}(t_k), r_{max})$. Ablation confirms that removing truncation reduces correlation from 0.856 to 0.700, while results remain robust for $r_{max}$ between 100 and 500.

### Loss & Training

RSR is a data selection metric rather than a training method. When applied, RSR is calculated for candidate trajectories, and those with the lowest RSR are selected for SFT. The training itself utilizes standard supervised fine-tuning loss.

## Key Experimental Results

### Main Results (Correlation Analysis)

| Metric | Qwen-3-14B | LLaMA-3.1-8B | Qwen-2.5-7B | Qwen-3-4B | Qwen-2.5-3B | Average |
|------|-----------|-------------|------------|----------|-----------|------|
| Teacher Params | 0.04 | 0.34 | 0.20 | 0.02 | 0.26 | 0.01 |
| Avg-Surprisal | 0.24 | 0.42 | 0.55 | 0.55 | 0.70 | 0.49 |
| GRACE | 0.25 | 0.58 | 0.66 | 0.75 | 0.69 | 0.59 |
| **RSR (Ours)** | **0.85** | **0.85** | **0.92** | **0.82** | **0.85** | **0.86** |

### Ablation Study

| Configuration | Average Correlation | Gain |
|------|---------|------|
| RSR ($r_{max}=100$) | 0.856 | - |
| No rank truncation | 0.700 | -0.156 |
| No weighted average (Avg-RSRtoken) | 0.391 | -0.465 |
| Filtered average (top 30%) | 0.793 | -0.064 |
| $r_{max}=500$ | 0.822 | -0.034 |
| Only 200 samples | 0.864 | +0.007 |

### Key Findings
- RSR significantly outperforms all baseline metrics across all 5 student models, with an average Spearman correlation of 0.86, compared to 0.65 for the best baseline (Rule-based Quality).
- Surprisal weighting is the most critical design; removing it causes the correlation to crash by 0.465.
- RSR is insensitive to sample size; using only 200 samples (4% of the original) yields equivalent performance.
- In trajectory selection tasks, data selected by RSR achieves training results comparable to or exceeding the optimal results found by brute-force searching through all teachers.

## Highlights & Insights

- **Refined Insight on "Absolutely Unfamiliar + Relatively Familiar"**: Translating the conflict between informativeness and alignment into a contrast between surprisal and rank is a refined observation that guides the RSR design and provides a new perspective for understanding distillation data selection.
- **Elegant Mathematical Simplification**: Starting from a token-level weighted average and deriving a clean "average rank / average surprisal" form ensures extremely low computational cost (a single forward pass) and high interpretability.
- **Strong Transfer Potential**: The core idea of RSR—measuring the "learnability" of data for a specific model—can likely be extended to data selection in any SFT scenario beyond reasoning tasks.

## Limitations & Future Work

- Currently validated only on mathematical reasoning; not yet systematically tested on code generation or general dialogue scenarios.
- RSR requires a forward pass for every trajectory to calculate rank, which could involve significant computational costs for massive datasets (million-scale).
- Only considers the suitability of individual trajectories without modeling the diversity or complementarity between trajectories (subset selection scenarios).
- While the discussion mentions potential for non-CoT data and subset selection, experimental support is currently limited.

## Related Work & Insights

- **vs Avg-Surprisal (Zhang et al.)**: Measuring suitability only via probability tends to select data the student already knows (correlation 0.49); RSR solves this "informativeness blind spot" by introducing the rank dimension.
- **vs GRACE (Li et al.)**: Gradient-based methods require extra gradient computation and achieve 0.59 correlation; RSR is simpler (forward pass only) and more effective.
- **vs Influence Score**: Inspired by influence functions but performs inconsistently across different student models; RSR shows consistent performance across students.

## Rating

- Novelty: ⭐⭐⭐⭐ Clear and elegant core insight, though essentially a combination of two existing metrics.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale experiments with 5 students × 11 teachers, detailed ablation, and validation via two downstream applications.
- Writing Quality: ⭐⭐⭐⭐⭐ Logical flow from observation to theory and experimental validation is exceptionally smooth.
- Value: ⭐⭐⭐⭐ High practical value for reasoning distillation data selection, though its scope needs further expansion.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals](llm_reasoning_as_trajectories_step-specific_representation_geometry_and_correctn.md)
- [\[ACL 2026\] Do Not Step Into the Same River Twice: Learning to Reason from Trial and Error](do_not_step_into_the_same_river_twice_learning_to_reason_from_trial_and_error.md)
- [\[ACL 2026\] Chain-of-Thought as a Lens: Evaluating Structured Reasoning Alignment between Human Preferences and Large Language Models](chain-of-thought_as_a_lens_evaluating_structured_reasoning_alignment_between_hum.md)
- [\[ACL 2026\] Discovering a Shared Logical Subspace: Steering LLM Logical Reasoning via Alignment of Natural-Language and Symbolic Views](discovering_a_shared_logical_subspace_steering_llm_logical_reasoning_via_alignme.md)
- [\[ACL 2026\] DVMap: Fine-Grained Pluralistic Value Alignment via High-Consensus Demographic-Value Mapping](dvmap_fine-grained_pluralistic_value_alignment_via_high-consensus_demographic-va.md)

</div>

<!-- RELATED:END -->
