---
title: >-
  [Paper Note] ATLAS: Alibaba Dataset and Benchmark for Learning-Augmented Scheduling
description: >-
  [ICLR 2026][learning_theory][Paper Note] This paper cleans and performs feature engineering on the Alibaba PAI-2020 GPU cluster trace to create ATLAS, a "non-clairvoyant scheduling" dataset containing 730,000 jobs with ground-truth job duration labels. It introduces an end-to-end benchmark, LASched, which evaluates both prediction tasks (estimating job durati
tags:
  - ICLR 2026
  - learning_theory
date: 2026-05-08
content_hash: 1f9076b95fb05dfc
---
# ATLAS: Alibaba Dataset and Benchmark for Learning-Augmented Scheduling

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=QBvxXzHdZx](https://openreview.net/forum?id=QBvxXzHdZx)  
**Code**: https://github.com/zhiyunjiang0810/non-clairvoyant-with-predictions  
**Area**: Datasets and Benchmarks / Learning-Augmented Scheduling / Algorithms and Prediction  
**Keywords**: Learning-augmented scheduling, non-clairvoyant scheduling, Alibaba PAI trace, job duration prediction, competitive ratio

## TL;DR
This paper cleans and performs feature engineering on the Alibaba PAI-2020 GPU cluster trace to create ATLAS, a "non-clairvoyant scheduling" dataset containing 730,000 jobs with ground-truth job duration labels. It introduces an end-to-end benchmark, LASched, which evaluates both prediction tasks (estimating job duration using features available at submission) and scheduling tasks (using predictions to optimize total completion time, max stretch, and makespan). This allows "algorithm + prediction" theories, previously confined to synthetic data, to be fairly reproduced and compared on real-world workloads.

## Background & Motivation
**Background**: A classic challenge in scheduling is **non-clairvoyant scheduling**, where the scheduler does not know the duration of a job (job size) at submission time. Consequently, optimal policies like SRPT (Shortest Remaining Processing Time) cannot be implemented, forcing a reliance on "blind" policies like FIFO or Round-Robin, which achieve suboptimal performance. Recently, **learning-augmented algorithms** (also known as algorithms with predictions) have emerged as a solution. These use an ML predictor to estimate job size $\hat p_j$ for an online algorithm, approaching optimality with accurate predictions while maintaining worst-case guarantees when predictions are poor. This framework has fostered an active community developing algorithms with provable competitive ratios for objectives such as total completion time, max stretch, and makespan.

**Limitations of Prior Work**: While theoretical results are promising, it remains **unknown how these algorithms perform on real workloads**. Existing data and benchmarks are insufficient: (1) Production traces often normalize/anonymize job durations (e.g., Google Borg removes context like users and resource requests), focus only on VM allocation (e.g., Azure/Microsoft VM traces lack job structure), or are designed for workload characterization rather than scheduling evaluation (e.g., original Alibaba traces). (2) Most theoretical works rely on synthetic workloads (exponential/Pareto/uniform distributions), losing complex real-world patterns. (3) The field **lacks a standardized benchmark**: scheduling frameworks (online/offline, preemption), predictor training/validation, and result normalization vary across papers, making cross-paper comparisons impossible. Furthermore, many works ignore temporal constraints or skip proper calibration-test separation, leading to information leakage.

**Key Challenge**: Real production clusters violate theoretical assumptions: jobs are multi-step workflows (pre-processing failures terminate the chain), hardware is heterogeneous (V100/P100/T4), and arrivals are stochastic. This invalidates standard analysis. Consequently, a core question remains: **How do learning-augmented schedulers actually perform in real environments?**

**Goal**: To bridge the gap between theory and practice by creating a "research-ready" real-world dataset and a leak-safe, reproducible benchmark covering the entire pipeline (prediction $\rightarrow$ scheduling).

**Key Insight**: The Alibaba PAI-2020 GPU trace provides complete job lifecycle records (submission, queuing, scheduling, execution). The authors observe that by **strictly separating "features visible at submission" from "ground-truth duration known only after execution,"** and adding leak-proof feature engineering and evaluation protocols, this trace can be transformed into a "gold standard" dataset for non-clairvoyant scheduling.

**Core Idea**: Reconstruct the production trace into the ATLAS dataset (non-clairvoyant + real labels + leak-proof) and build the LASched benchmark to integrate prediction and scheduling tasks using normalized competitive ratios as a unified metric.

## Method

### Overall Architecture
The work consists of a two-layer structure. **The bottom layer is the ATLAS dataset**: Starting from two months of Alibaba PAI-2020 GPU traces (6500+ GPUs, ~1800 machines, 730k+ completed jobs), the authors clean, join, and engineer features from job/task/instance/group-tag tables. This produces a job-level table containing only information known at submission, paired with ground-truth processing duration labels. The dataset is intentionally **non-clairvoyant**: models only see features existing at the submission moment (resource requests, user/group labels, historical frequency, etc.), while post-execution data (actual resource utilization, machine placement, sensor metrics) are excluded.

**The top layer is the LASched benchmark**, which links two tasks:

1. **Prediction Task**: Takes submission features $x_j$ to predict job duration $\hat p_j$, measured by indices like RMSLE, Coverage@$\tau$, and Spearman correlation. The authors implement 4 naive baselines and 6 advanced methods within a leak-proof multi-stage framework (Algorithm 1).
2. **Scheduling Task**: Feeds predictions $\hat p_j$ into learning-augmented schedulers (e.g., SPJF, PRR, SPRPT, EDF-P, LPPT/SPPT). Evaluation covers three objectives: **total completion time** $\sum_j C_j$ (online preemptive, single machine), **makespan** $C_{\max}$ (batch non-preemptive, multi-machine), and **max stretch** $\max_j S_j$ (online preemptive, single machine). Each objective is normalized by an optimal solution or tight bound to produce an **empirical competitive ratio**, enabling comparisons across different methods and scales.

### Key Designs

**1. Non-clairvoyant Dataset Delimitation + Fork-join Ground Truth**:
The dataset's value hinges on a strict boundary: features must come only from the submission moment, and labels must be unbiased execution durations. The authors categorized PAI columns: ✓ Prediction features (submission time, user ID, requested CPU/GPU/memory, instance counts, group tags, etc.), • Ground-truth labels, × Excluded (actual allocated GPU type, instance IDs, runtime sensors, networking). The latter are excluded as they are determined post-execution or by the scheduler itself, which would cause leakage.

The label definition is a technical highlight. Since a job consists of multiple tasks and instances, the authors **do not** use average task duration. Instead, they **reconstruct durations from start/end timestamps**: for task $t$, $s_t=\min_{i\in I(t)} s_i$ and $e_t=\max_{i\in I(t)} e_i$; for the job, $S=\min_t s_t$ and $E=\max_t e_t$. The duration is defined as:
$$p^*_j = E - S.$$
This corresponds to the classic **fork-join completion rule**. Compared to using "average instance duration $\bar d_t$", $p^*_j$ is more conservative and safer, as it accounts for gang-scheduled jobs where the last instance determines resource release (85% of tasks in the trace require gang scheduling).

**2. Leak-proof Feature Engineering**:
To avoid temporal leakage, the authors split data by submission time $r_j$ (earliest 70% Train / middle 15% Val / last 15% Test). All encoders and statistics are fitted **only on the training set**. Features include: (1) Resource requests (log-transformed to handle skewness from 11.02 to 0.17); (2) Time features (sin/cos encoding for hour cycles); (3) Recurrence signatures (matching historical executions via user/group/workload tags); (4) Historical statistics (rolling means/counts using `shift(1)` to ensure lag); (5) Categorical labels (Empirical Bayes shrinkage with $\lambda=5$ for low-support groups).

**3. Unified Multi-stage Prediction Framework (Algorithm 1)**:
The authors designed a three-stage framework to evaluate heterogenous methods fairly:
*   **Stage 1 (Training)**: Trains quantile regressors, LightGBM (L2/Huber), two-stage classifiers, and recency models.
*   **Stage 2 (Calibration)**: Uses the validation set for Conformal Quantile Regression (CQR), isotonic calibration, and Meta-stacking (LightGBM).
*   **Stage 3 (Testing)**: Generates final predictions $\hat p^*_j = \exp(\hat y_j) - 1$. Calibration occurs **strictly on the validation set** to prevent leakage.

**4. Three-Objective Scheduling Evaluation**:
(1) **Total Completion Time** $1|r_j,\text{pmtn}|\sum C_j$: Normalized by SRPT to get $\rho_{TC}$. (2) **Makespan** $P||C_{\max}$: Normalized by McNaughton’s preemption lower bound $\text{OPT}_{pre}$. (3) **Max Stretch** $1|r_j,\text{pmtn}|\max_j S_j$: Stretch is defined as $(C_j-r_j)/p^*_j$, normalized by $S^*$ found via binary search and EDF feasibility.

## Key Experimental Results

### Main Results (Prediction on PAI trace)
Coverage@k = % of jobs where prediction error is within k%:

| Method | Type | Cov@25 (%) All/Rec. | Cov@50 (%) All/Rec. | RMSLE | $\rho$ |
|------|------|------|------|------|------|
| Single-Stage | Baseline | 19.5 / 28.5 | 36.2 / 51.6 | 1.568 | 0.690 |
| **LGBM-Meta** | Advanced | **51.2** / 58.3 | **70.6** / 77.5 | **0.871** | **0.912** |

LGBM-Meta (Meta-stacking) performed best overall. The inclusion of task signatures and EB shrinkage significantly improved performance over baselines.

### Main Results (Scheduling Competitive Ratios, Lower is Better)

| Objective | Representative Method | Comp. Ratio | Comparison |
|------|---------|--------|------|
| Total Completion Time | **Meta-SPJF** | **1.106** | SRPT=1.000, FIFO=5.372 |
| Makespan | Iso-LPPT | **1.574** | LPT=1.000, SPT=1.539 |
| Max Stretch $\rho_{max}$ | Meta-SPRPT | **18.13** | LAS/FB=205.28 (~11× improvement) |

### Key Findings
*   **Signatures and EB Shrinkage are critical**: Baselines without signatures only achieve ~20% Cov@25, while advanced methods exceed 50%.
*   **Prediction error propagation varies**: Total completion time is least sensitive to prediction error (Meta-SPJF is only 10.5% worse than clairvoyant SJF). Max stretch is highly fragile; deadline-based EDF-P **amplifies absolute duration errors** (HRAS-EDF $\rho_{max}$ reached 1700), whereas SPRPT (remaining time) is more stable.

## Highlights & Insights
*   Transforming a workload trace into a scheduling dataset relies on a rigorous protocol: non-clairvoyant feature/label separation and fork-join duration definitions.
*   The use of **fork-join labels** prevents the systematic underestimation of gang-scheduled jobs, prioritizing result safety over simplicity.
*   By linking "prediction $\rightarrow$ scheduling," the paper quantifies how much a 1% improvement in prediction translates to scheduling efficiency. Specifically, for stretch objectives, predictors should prioritize "remaining time" over "absolute deadlines."

## Limitations & Future Work
*   **Labels are not fully decoupled**: The ground truth $p^*_j$ still contains biases from the original Fuxi scheduler's placement decisions and CPU contention.
*   **Single Data Source**: Results are based only on Alibaba PAI-2020; cross-cluster or cross-vendor generalization is unverified.
*   **Simulation Simplifications**: Scheduling models assume single-machine or batch arrival, omitting complex real-world constraints like fractional GPU sharing or specific packing rules.

## Related Work & Insights
*   **vs. Google/Azure Traces**: ATLAS retains full submission context and precise duration labels, whereas others often mask job structures or context.
*   **vs. Original PAI Trace**: While the original was for characterization, ATLAS is "research-ready" for scheduling, using a safer fork-join label definition.
*   **Value**: LASched provides a standardized pipeline for training, calibration, and normalized evaluation, solving the issue of experimental inconsistency in the learning-augmented scheduling field.

## Rating
*   Novelty: ⭐⭐⭐⭐ (Assembles existing methods but fills a critical gap in infrastructure).
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Comprehensive evaluation across 10 prediction methods and 3 scheduling objectives).
*   Writing Quality: ⭐⭐⭐⭐ (Rigorous definitions, though symbol-dense).
*   Value: ⭐⭐⭐⭐⭐ (Provides a long-term reusable benchmark for the community).

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On Smoothness Bounds for Non-Clairvoyant Scheduling with Predictions](on_smoothness_bounds_for_non-clairvoyant_scheduling_with_predictions.md)
- [\[ICLR 2026\] Online Rounding and Learning Augmented Algorithms for Facility Location](online_rounding_and_learning_augmented_algorithms_for_facility_location.md)
- [\[ICML 2026\] Parsimonious Learning-Augmented Online Metric Matching](../../ICML2026/learning_theory/parsimonious_learning-augmented_online_metric_matching.md)
- [\[ICLR 2026\] Better Learning-Augmented Spanning Tree Algorithms via Metric Forest Completion](better_learning-augmented_spanning_tree_algorithms_via_metric_forest_completion.md)
- [\[ICLR 2026\] Decision-Theoretic Approaches for Improved Learning-Augmented Algorithms](decision-theoretic_approaches_for_improved_learning-augmented_algorithms.md)

</div>

<!-- RELATED:END -->
