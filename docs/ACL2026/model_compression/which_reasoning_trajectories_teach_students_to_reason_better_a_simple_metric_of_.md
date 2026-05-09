---
title: >-
  [Paper Note] Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment
description: >-
  [ACL 2026][Model Compression][Knowledge Distillation] This paper proposes the Rank-Surprisal Ratio (RSR), a metric that jointly measures the informativeness and alignment of reasoning trajectories with respect to a student model, achieving an average Spearman correlation of 0.86 with post-training performance across 5 student models and 11 teacher models, and demonstrating utility in both trajectory selection and teacher selection.
tags:
  - ACL 2026
  - Model Compression
  - Knowledge Distillation
  - Reasoning Trajectories
  - Data Selection
  - Chain-of-Thought
  - Large Language Models
date: 2026-05-08
content_hash: de842db87e9526c8
---

# Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment

**Conference**: ACL 2026
**arXiv**: [2601.14249](https://arxiv.org/abs/2601.14249)
**Code**: [GitHub](https://github.com/UmeanNever/RankSurprisalRatio)
**Area**: Model Compression
**Keywords**: Knowledge Distillation, Reasoning Trajectories, Data Selection, Chain-of-Thought, Large Language Models

## TL;DR

This paper proposes the Rank-Surprisal Ratio (RSR), a metric that jointly measures the informativeness and alignment of reasoning trajectories with respect to a student model, achieving an average Spearman correlation of 0.86 with post-training performance across 5 student models and 11 teacher models, and demonstrating utility in both trajectory selection and teacher selection.

## Background & Motivation

**Background**: Long chain-of-thought (Long CoT) trajectories have become the primary means of distilling reasoning capabilities from large reasoning models into smaller student models via SFT, where the student learns the teacher's reasoning process.

**Limitations of Prior Work**: Empirical evidence repeatedly shows that stronger teacher models (e.g., DeepSeek-R1 at 671B) do not necessarily yield better students. The "fitness" between data and the student model is a critical factor in distillation effectiveness. However, existing methods primarily rely on the student model's log-probability over data to measure fitness, which tends to select high-probability trajectories already familiar to the student, neglecting data with genuine learning value.

**Key Challenge**: There is an inherent trade-off between informativeness and alignment — data that is too familiar offers no learning value, while data that is too unfamiliar cannot be absorbed. This mirrors the psychological concept of the Zone of Proximal Development: the most effective learning materials should slightly exceed the learner's current ability without being entirely incomprehensible.

**Goal**: To design a simple, interpretable metric that measures the fitness of reasoning trajectories for a specific student model, balancing both informativeness and alignment.

**Key Insight**: The authors observe that effective trajectories exhibit a distinctive pattern — their tokens have very low absolute probabilities under the student model (high surprisal, indicating content the student would not generate), yet still rank highly in the vocabulary (low rank, indicating they remain within the student's comprehension). This "absolutely unfamiliar but relatively familiar" property strikes the desired balance between informativeness and alignment.

**Core Idea**: Use the ratio of token rank to surprisal (RSR) to measure trajectory fitness — a lower RSR indicates a trajectory that is both informative and aligned with the student.

## Method

### Overall Architecture

The method consists of three steps: (1) perform a single forward pass over a given reasoning trajectory using the student model to obtain the probability distribution over each token; (2) compute each token's surprisal (negative log-likelihood) and rank (position in the vocabulary); (3) define the trajectory-level RSR as the ratio of average rank to average surprisal, serving as the fitness score. RSR requires only one forward pass and needs no additional verifiers or held-out test data.

### Key Designs

1. **Bimodal Distribution Theoretical Model**:

    - Function: Provides a theoretical justification for the effectiveness of RSR.
    - Mechanism: Models the student's token prediction distribution as a bimodal mixture — the primary mode $Z_A$ represents the student's dominant generation pattern (high probability, low rank), and the secondary mode $Z_B$ represents deviations from the primary mode that remain within the student's knowledge (low probability, but still relatively high rank). Effective teacher trajectories correspond to the $Z_B$ type — low in absolute probability but high in relative rank. Simulation experiments confirm that $Z_B$-type trajectories achieve the lowest RSR (1.30), while misaligned trajectories $Z_C$ achieve the highest (2.23).
    - Design Motivation: Explains why using surprisal or rank alone is insufficient — their ratio is necessary to distinguish "informative and aligned" ($Z_B$) from "informative but misaligned" ($Z_C$) trajectories.

2. **Surprisal-Weighted Aggregation**:

    - Function: Stably aggregates token-level RSR into a trajectory-level metric.
    - Mechanism: Naively averaging token-level $\text{RSR}_\text{token} = \text{Rank}(t_k) / \text{Surprisal}(t_k)$ causes numerical instability, as the surprisal of high-probability tokens approaches zero, causing the ratio to explode. The authors adopt a surprisal-weighted average, which after mathematical derivation simplifies to $\text{RSR}(\mathbf{x}) = \sum_k \text{Rank}(t_k) / \sum_k \text{Surprisal}(t_k)$, equivalent to average rank divided by average surprisal — a form that is both concise and numerically stable.
    - Design Motivation: Tokens with higher surprisal have greater impact on student learning; emphasizing them through weighting is principled. Removing the weighting causes correlation to drop dramatically from 0.856 to 0.391.

3. **Rank Truncation**:

    - Function: Handles numerical instability caused by extremely unfamiliar tokens.
    - Mechanism: Truncates rank values to $r_{max}$ (default 100), i.e., $\min(\text{Rank}(t_k), r_{max})$. Extremely unfamiliar tokens may have ranks as large as the vocabulary size (e.g., 128K); such tokens are essentially indistinguishable to the student, and truncation removes this noise.
    - Design Motivation: Removing truncation reduces correlation from 0.856 to 0.700, confirming its necessity. Results are robust for $r_{max}$ in the range of 100–500.

### Loss & Training

RSR is a data selection metric rather than a training objective. In practice, RSR is computed over candidate trajectories, and those with the lowest RSR are selected for SFT training. Training itself uses a standard supervised fine-tuning loss.

## Key Experimental Results

### Main Results (Correlation Analysis)

| Metric | Qwen-3-14B | LLaMA-3.1-8B | Qwen-2.5-7B | Qwen-3-4B | Qwen-2.5-3B | Avg |
|--------|-----------|-------------|------------|----------|-----------|-----|
| Teacher Params | 0.04 | 0.34 | 0.20 | 0.02 | 0.26 | 0.01 |
| Avg-Surprisal | 0.24 | 0.42 | 0.55 | 0.55 | 0.70 | 0.49 |
| GRACE | 0.25 | 0.58 | 0.66 | 0.75 | 0.69 | 0.59 |
| **RSR (Ours)** | **0.85** | **0.85** | **0.92** | **0.82** | **0.85** | **0.86** |

### Ablation Study

| Configuration | Avg Correlation | Change |
|--------------|----------------|--------|
| RSR ($r_{max}=100$) | 0.856 | — |
| Without rank truncation | 0.700 | −0.156 |
| Without weighted avg (Avg-RSRtoken) | 0.391 | −0.465 |
| Filtered average (top 30%) | 0.793 | −0.064 |
| $r_{max}=500$ | 0.822 | −0.034 |
| Using only 200 samples | 0.864 | +0.007 |

### Key Findings
- RSR significantly outperforms all baselines across all 5 student models, achieving an average Spearman correlation of 0.86; the second-best method (Rule-based Quality) reaches only 0.65.
- Surprisal weighting is the most critical design component; removing it causes a 0.465 drop in correlation.
- RSR is insensitive to sample size — using only 200 samples (4% of the full set) achieves comparable performance.
- In trajectory selection, data selected by RSR yields training performance that matches or exceeds the best results obtained by exhaustively searching all teachers.

## Highlights & Insights

- **The "absolutely unfamiliar yet relatively familiar" insight is remarkably concise**: Decomposing the tension between informativeness and alignment into two dimensions — surprisal and rank — not only motivates the RSR design but also provides a new perspective for understanding distillation data selection.
- **Elegant mathematical simplification**: Starting from a token-level weighted average, the derivation yields the remarkably clean form of "average rank / average surprisal," with minimal computational overhead (single forward pass) and strong interpretability.
- **Strong generalization potential**: The core idea of RSR — measuring the "learnability" of data for a given model — can be extended to data selection in any SFT setting, not limited to reasoning tasks.

## Limitations & Future Work

- Validation is currently limited to mathematical reasoning tasks; systematic evaluation on code generation, general dialogue, and other domains remains to be conducted.
- Computing ranks requires a forward pass over each trajectory; the computational cost for very large-scale datasets (millions of examples) remains non-negligible.
- Only the fitness of individual trajectories is considered; diversity and complementarity among trajectories are not modeled (relevant to subset selection scenarios).
- The authors discuss potential applications of RSR to non-CoT data and subset selection, but experimental support for these claims is insufficient.

## Related Work & Insights

- **vs. Avg-Surprisal (Zhang et al.)**: Uses only probability to measure fitness, biasing selection toward already-familiar data; correlation is only 0.49. RSR addresses the "informativeness blind spot" by introducing the rank dimension.
- **vs. GRACE (Li et al.)**: A gradient-based method requiring additional gradient computation with a correlation of 0.59; RSR is simpler (forward pass only) and more effective.
- **vs. Influence Score**: Inspired by influence functions but exhibits unstable performance across some students (high variance in correlation); RSR demonstrates consistent performance across student models.

## Rating

- Novelty: ⭐⭐⭐⭐ — The core insight is clear and elegant, though the metric is essentially a combination of two existing quantities.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Large-scale experiments across 5 students × 11 teachers, detailed ablations, and validation on two downstream applications.
- Writing Quality: ⭐⭐⭐⭐⭐ — The progression from observation to theory to experiment is exceptionally coherent.
- Value: ⭐⭐⭐⭐ — Offers direct practical value for reasoning distillation data selection, though the scope of applicability remains to be extended.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Reason Only When Needed: Efficient Generative Reward Modeling via Model-Internal Uncertainty](reason_only_when_needed_efficient_generative_reward_modeling_via_model-internal_.md)
- [\[ICCV 2025\] Dataset Distillation via the Wasserstein Metric](../../ICCV2025/model_compression/dataset_distillation_via_the_wasserstein_metric.md)
- [\[ACL 2026\] SeLaR: Selective Latent Reasoning in Large Language Models](selar_selective_latent_reasoning_in_large_language_models.md)
- [\[ICLR 2026\] Modality-free Graph In-context Alignment](../../ICLR2026/model_compression/modality-free_graph_in-context_alignment.md)
- [\[ICLR 2026\] ConFu: Contemplate the Future for Better Speculative Sampling](../../ICLR2026/model_compression/confu_contemplate_the_future_for_better_speculative_sampling.md)

</div>

<!-- RELATED:END -->
