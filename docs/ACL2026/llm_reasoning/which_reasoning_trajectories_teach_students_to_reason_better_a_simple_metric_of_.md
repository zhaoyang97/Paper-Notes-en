---
title: >-
  [Paper Note] Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment
description: >-
  [ACL 2026][Reasoning][Knowledge Distillation] The authors propose the Rank-Surprisal Ratio (RSR) metric, which evaluates training data suitability by jointly measuring the "informativeness" and "alignment" of reasoning trajectories for a student model. RSR achieves an average Spearman correlation of 0.86 with post-training performance across 5 student and 11 teacher model combinations, and it is successfully applied to trajectory and teacher selection.
tags:
  - "ACL 2026"
  - "Reasoning"
  - "Knowledge Distillation"
  - "Reasoning Trajectories"
  - "Data Selection"
  - "Chain-of-Thought"
  - "Large Language Models"
date: 2026-05-08
content_hash: 72a5e240f0ee9a3a
---

# Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment

**Conference**: ACL 2026  
**arXiv**: [2601.14249](https://arxiv.org/abs/2601.14249)  
**Code**: [GitHub](https://github.com/UmeanNever/RankSurprisalRatio)  
**Area**: Model Compression  
**Keywords**: Knowledge Distillation, Reasoning Trajectories, Data Selection, Chain-of-Thought, Large Language Models

## TL;DR

The authors propose the Rank-Surprisal Ratio (RSR) metric, which evaluates training data suitability by jointly measuring the "informativeness" and "alignment" of reasoning trajectories for a student model. RSR achieves an average Spearman correlation of 0.86 with post-training performance across 5 student and 11 teacher model combinations, and it is successfully applied to trajectory and teacher selection.

## Background & Motivation

**Background**: Long Chain-of-Thought (CoT) trajectories have become the primary means of distilling reasoning capabilities from large reasoning models to smaller student models via Supervised Fine-Tuning (SFT).

**Limitations of Prior Work**: Experiments repeatedly demonstrate that stronger teacher models (e.g., DeepSeek-R1 with 671B parameters) do not necessarily yield better students. The "suitability" between data and the student model is the key factor determining distillation effectiveness. However, existing methods primarily rely on the student model's log-probability of the data, which tends to select high-probability trajectories already familiar to the student, ignoring data with genuine learning value.

**Key Challenge**: The trade-off between informativeness and alignment—data that is too familiar has no learning value, while data that is too foreign cannot be learned. This echoes the concept of the "Zone of Proximal Development" in psychology: the most effective learning materials should be slightly beyond the learner's current capability but not entirely incomprehensible.

**Goal**: Design a simple, interpretable metric to measure the suitability of reasoning trajectories for a specific student model, balancing both informativeness and alignment.

**Key Insight**: The authors observe that effective trajectories exhibit a specific pattern—their tokens have low absolute probability under the student model (high surprisal, indicating the student would not generate them), yet they still rank high in the vocabulary distribution (low rank, indicating they are within the student's understanding). This "absolutely unfamiliar but relatively familiar" characteristic perfectly balances informativeness and alignment.

**Core Idea**: Use the ratio of token rank to surprisal (RSR) to measure trajectory suitability—the lower the RSR, the more the trajectory is both informative and aligned with the student.

## Method

### Overall Architecture

The method consists of three steps: (1) Perform a single forward pass with the student model on a given reasoning trajectory to obtain the probability distribution for each token; (2) Calculate the surprisal (negative log-likelihood) and rank (position in the vocabulary) for each token; (3) Define the trajectory-level RSR as the ratio of average rank to average surprisal. RSR requires only one forward pass without additional verifiers or test data.

### Key Designs

**1. Bimodal Distribution Theoretical Model: Why "Absolutely Unfamiliar but Relatively Familiar" yields the best material**

If only surprisal is considered, all unseen content is treated as valuable. If only rank is considered, familiar high-frequency content is selected. To distinguish "informative and aligned" from "informative but misaligned," a single dimension is insufficient. The authors model the student's token prediction distribution as a bimodal mixture: the primary mode $Z_A$ represents the student's dominant generation patterns (high probability, low rank), and the secondary mode $Z_B$ represents patterns that deviate from the primary mode but remain within the student's knowledge (low probability, but still low rank).

Effective teacher trajectories should correspond to $Z_B$—low absolute probability (high informativeness as the student won't generate them) but relatively high rank (high alignment as the student can understand them). Simulations confirm this: $Z_B$ trajectories have the lowest RSR (1.30), while $Z_C$ trajectories (misaligned, unlearnable) have the highest (2.23). This explains why RSR uses the ratio of rank to surprisal—only by considering both can $Z_B$ be isolated from $Z_C$.

**2. Surprisal Weighted Aggregation: Stabilizing token-level ratios into a trajectory-level metric**

Token-level RSR is defined as $\text{RSR}_\text{token} = \text{Rank}(t_k) / \text{Surprisal}(t_k)$. However, a simple arithmetic mean makes the ratio explode as surprisal approaches zero for high-probability tokens. The authors employ a surprisal-weighted average, which simplifies to an elegant mathematical form:

$$\text{RSR}(\mathbf{x}) = \frac{\sum_k \text{Rank}(t_k)}{\sum_k \text{Surprisal}(t_k)}$$

This "average rank divided by average surprisal" avoids numerical instability and naturally gives more weight to high-surprisal tokens, which have a greater impact on student learning. Ablations show that reverting to a simple token-level average causes the correlation to drop from 0.856 to 0.391.

**3. Rank Truncation: Smoothing noise from extremely unfamiliar tokens**

For extremely unfamiliar tokens, the rank can reach the vocabulary size (e.g., 128K). For the student, a rank of 50,000 versus 120,000 is practically identical—both are "completely unknown." Retaining these large raw values introduces noise. The authors truncate the rank at $r_{max}$ (default 100), using $\min(\text{Rank}(t_k), r_{max})$. Removing truncation reduces correlation from 0.856 to 0.700, while results remain robust for $r_{max}$ in the 100–500 range.

### Loss & Training

RSR is a data selection metric rather than a training method. During application, RSR is calculated for candidate trajectories, and those with the lowest RSR are selected for SFT. Training follows the standard supervised fine-tuning loss.

## Key Experimental Results

### Main Results (Correlation Analysis)

| Metric | Qwen-3-14B | LLaMA-3.1-8B | Qwen-2.5-7B | Qwen-3-4B | Qwen-2.5-3B | Average |
|--------|------------|--------------|-------------|-----------|-------------|---------|
| Teacher Params | 0.04 | 0.34 | 0.20 | 0.02 | 0.26 | 0.01 |
| Avg-Surprisal | 0.24 | 0.42 | 0.55 | 0.55 | 0.70 | 0.49 |
| GRACE | 0.25 | 0.58 | 0.66 | 0.75 | 0.69 | 0.59 |
| **RSR (Ours)** | **0.85** | **0.85** | **0.92** | **0.82** | **0.85** | **0.86** |

### Ablation Study

| Configuration | Avg. Correlation | Change |
|---------------|------------------|--------|
| RSR ($r_{max}=100$) | 0.856 | - |
| No rank truncation | 0.700 | -0.156 |
| No weighted average (Avg-RSRtoken) | 0.391 | -0.465 |
| Filtered average (top 30%) | 0.793 | -0.064 |
| $r_{max}=500$ | 0.822 | -0.034 |
| Only 200 samples | 0.864 | +0.007 |

### Key Findings
- RSR significantly outperforms all baseline metrics across all 5 student models, with an average Spearman correlation of 0.86 (the next best rule-based method is 0.65).
- Surprisal weighting is the most critical design; removing it causes the correlation to plummet by 0.465.
- RSR is insensitive to sample size; using only 200 samples (4% of the original) achieves comparable results.
- In trajectory selection tasks, data selected by RSR yields training results that match or exceed the optimal outcomes from brute-force searching all teachers.

## Highlights & Insights

- **Refined Insight on Familiarity**: Translating the conflict between informativeness and alignment into a comparison between surprisal and rank is a concise and elegant observation. This not only guides the RSR design but also provides a new perspective for understanding distillation data selection.
- **Elegant Mathematical Simplification**: The derivation from a token-level weighted average to the simple "average rank / average surprisal" form is efficient (one forward pass) and highly interpretable.
- **High Transfer Potential**: The core idea of RSR—measuring the "learnability" of data for a model—can be extended beyond reasoning tasks to data selection in any SFT scenario.

## Limitations & Future Work

- Currently validated only on mathematical reasoning tasks; systematic testing on code generation and general dialogue is needed.
- RSR requires a forward pass for every trajectory to calculate ranks, which can be computationally expensive for million-scale datasets.
- The metric considers only individual trajectory suitability and does not model diversity or complementarity between trajectories (subset selection).
- While the discussion suggests RSR could be used for non-CoT data, experimental support for this is currently limited.

## Related Work & Insights

- **vs Avg-Surprisal (Zhang et al.)**: Measuring suitability only via probability tends to select data the student already knows (correlation 0.49); RSR addresses this "informativeness blind spot" by introducing the rank dimension.
- **vs GRACE (Li et al.)**: Gradient-based methods require extra computation and show lower correlation (0.59); RSR is simpler (forward pass only) and more effective.
- **vs Influence Score**: Influenced by the influence function, but performance is unstable across different student models; RSR shows consistent performance.

## Rating

- Novelty: ⭐⭐⭐⭐ The core insight is elegant, though it combines two existing concepts.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale experiments with 5 students and 11 teachers, detailed ablations, and two downstream application validations.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely logical flow from observation to theory to experiment.
- Value: ⭐⭐⭐⭐ High practical utility for reasoning distillation, though the scope of application remains to be expanded.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals](llm_reasoning_as_trajectories_step-specific_representation_geometry_and_correctn.md)
- [\[ACL 2026\] Do Not Step Into the Same River Twice: Learning to Reason from Trial and Error](do_not_step_into_the_same_river_twice_learning_to_reason_from_trial_and_error.md)
- [\[ICLR 2026\] GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning](../../ICLR2026/llm_reasoning/gpg_a_simple_and_strong_reinforcement_learning_baseline_for_model_reasoning.md)
- [\[ACL 2026\] Chain-of-Thought as a Lens: Evaluating Structured Reasoning Alignment between Human Preferences and Large Language Models](chain-of-thought_as_a_lens_evaluating_structured_reasoning_alignment_between_hum.md)
- [\[ACL 2026\] Discovering a Shared Logical Subspace: Steering LLM Logical Reasoning via Alignment of Natural-Language and Symbolic Views](discovering_a_shared_logical_subspace_steering_llm_logical_reasoning_via_alignme.md)

</div>

<!-- RELATED:END -->
