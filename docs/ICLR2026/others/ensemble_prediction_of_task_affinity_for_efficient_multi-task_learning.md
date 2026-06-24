---
title: >-
  [Paper Note] Ensemble Prediction of Task Affinity for Efficient Multi-Task Learning
description: >-
  [ICLR 2026][Multi-task learning] ETAP combines white-box gradient affinity analysis with data-driven ensemble prediction. Using a minimal number of training groups, it accurately predicts performance gains in multi-task learning (MTL), enabling efficient task partitioning into optimal groups.
tags:
  - "ICLR 2026"
  - "Multi-task learning"
  - "task affinity"
  - "task grouping"
  - "ensemble prediction"
  - "gradient affinity"
date: 2026-05-08
content_hash: 8670b00fe2fa40d6
---

# Ensemble Prediction of Task Affinity for Efficient Multi-Task Learning

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=RuVT3PeX1M](https://openreview.net/forum?id=RuVT3PeX1M)  
**Code**: Open-sourced via paper attachment (supplementary material)  
**Area**: Multi-Task Learning / Transfer Learning  
**Keywords**: Multi-task learning, task affinity, task grouping, ensemble prediction, gradient affinity

## TL;DR

ETAP combines white-box gradient affinity analysis with data-driven ensemble prediction. Using a minimal number of training groups, it accurately predicts performance gains in multi-task learning (MTL), enabling efficient task partitioning into optimal groups.

## Background & Motivation

**Background**: Multi-task learning (MTL) simultaneously trains multiple tasks through shared representations to improve generalization and reduce inference overhead. It is widely applied in computer vision, NLP, and medical informatics.  
**Limitations of Prior Work**: Not all task combinations are mutually beneficial; some joint training leads to "negative transfer," resulting in performance lower than single-task baselines. Finding the optimal task grouping requires an exhaustive search of all $2^n-1$ subsets, which is computationally prohibitive. Existing data-driven methods (HOA, MTGNet, Linear Surrogate) require a large number of labeled training groups for stable prediction. White-box methods (TAG), though theoretically grounded, require $n$ additional forward/backward passes per step, making the total overhead comparable to training multiple MTL models, and they only produce pairwise affinity estimates that fail to model high-order interactions.  
**Key Challenge**: White-box methods are efficient but limited in accuracy (linear approximations, ignoring high-order dependencies); data-driven methods are accurate but require heavy labeling. Both have distinct bias-variance shortcomings.  
**Goal**: Use a very small number of actual MTL training groups (as few as $|\mathcal{G}_\text{train}| = 5–10$) as supervision to accurately predict MTL gains for any task combination, and then use branch-and-bound search to find near-optimal task grouping schemes.  
**Core Idea**: Treat white-box gradient affinity as a strong prior feature, then refine it using a two-stage process—B-spline non-linear mapping followed by residual regression. This achieves a complementary ensemble of white-box and data-driven approaches (ETAP, Ensemble Task-Affinity Predictor).

## Method

### Overall Architecture

ETAP consists of three layers: First, gradient affinity scores are computed "for free" during a single MTL training run (White-box layer). Second, using a few real MTL gains as supervision, a non-linear mapping and a residual correction predictor are trained sequentially (Data-driven layer). Finally, predicted gains are used as the objective function for a branch-and-bound algorithm to select $B$ task groups (Search layer).

```mermaid
flowchart LR
    A[Single MTL Training\nGradient/Loss Collection] --> B[White-box Affinity Score\nz_{ti→tj}]
    B --> C[B-spline Non-linear Mapping\nŷ_aff]
    D[Small Set of Real Training Groups\nGtrain] --> C
    C --> E[Residual Regression Correction\nŷ_final]
    D --> E
    E --> F[Branch-and-Bound Search\nOutput Optimal Grouping]
```

### Key Designs

**1. Zero-Overhead Gradient Affinity Score**: To evaluate the impact of task $t_i$ on $t_j$, traditional TAG requires $n$ extra hypothetical forward/backward passes per step. ETAP directly utilizes gradients and parameter updates already available in standard backpropagation. At step $k$, it calculates direction alignment: $z^k_{t_i \to t_j} = \frac{[\nabla_{\theta^k_s} L^k_{t_j}] \cdot [\eta \nabla_{\theta^k_s} L^k_{t_i} - \beta v^{k-1}]}{L^k_{t_j}}$. A stable pairwise affinity $z_{t_i \to t_j}$ is obtained by time-averaging over all $K$ steps. Group-level affinity $z_{G \to t_i}$ is the average of pairwise affinities within the group. This reduces TAG's computational overhead by 46%/71%/63% on CelebA/ETTm1/Ridership respectively, while achieving higher correlation with ground-truth gains (0.32→0.47 vs. TAG's 0.16→0.43).

**2. B-spline Non-linear Mapping (Stage 1)**: There is a systematic gap in scale and linearity between affinity scores and MTL gains. Pure linear regression often suffers from high bias. ETAP performs a B-spline basis expansion on each $z_{G \to t}$: $\phi(z_{G \to t}) = [N_i(z_{G \to t})]_{i=1}^M$ (piecewise polynomials with local support). Regularized linear regression is then trained in this high-dimensional feature space to get the initial prediction $\hat{y}^\text{aff}_G$. Spline order, number of knots, and regularization strength are tuned via cross-validation on the small training set. This step "pulls" the white-box prior to the scale of the gain and implicitly models certain high-order dependencies.

**3. Multi-hot Encoding Residual Regression (Stage 2)**: Stage 1's $\hat{y}^\text{aff}_G$ may lack modeling of specific task combination particularities, leaving systematic biases. ETAP represents task groups using multi-hot vectors $u_G \in \{0,1\}^{|T|}$. It trains a ridge regression $f_\text{residual}$ using the residuals $e^\text{aff}_G = y_G - \hat{y}^\text{aff}_G$ as labels. The final prediction is $\hat{y}^\text{final}_G = \hat{y}^\text{aff}_G + f_\text{residual}(u_G)$. Compared to learning gains from scratch (as in MTGNet), learning only the residuals significantly reduces the required number of training groups, stabilizing at $|\mathcal{G}_\text{train}|=5$.

**4. Branch-and-Bound Task Grouping Search**: Finding the optimal task grouping is an NP-hard set cover problem. ETAP plugs the predicted gains $\hat{y}^\text{final}$ into a branch-and-bound algorithm (following the framework of Standley et al., 2020) to maximize total predicted gain under a budget $B$ (number of groups), finding near-optimal groupings without exhaustive enumeration.

## Key Experimental Results

### Main Results: Task Grouping MTL Performance (Total Loss, lower is better, $|\mathcal{G}_\text{train}|=10$)

| Dataset | # Groups | ETAP | MTGNet | TAG | Optimal (Exhaustive)|
|---------|----------|------|--------|-----|-------------------|
| CelebA  | 2        | **49.92** | 50.62 | 49.67 | 49.27 |
| CelebA  | 3        | **49.61** | 50.31 | 50.22 | 48.63 |
| ETTm1   | 3        | **3.93**  | 3.96  | 3.96  | 3.83  |
| Chemical| 2        | **4.67**  | 4.79  | 4.69  | 4.56  |
| Ridership| 4        | **17.59** | 18.06 | 18.25 | 16.79 |

*Ours (ETAP) is closest to the exhaustive optimal across all datasets and grouping counts, with the lowest variance.*

### Gain Prediction Correlation Coefficients ($|\mathcal{G}_\text{train}|=10$, higher is better)

| Method | CelebA | ETTm1 | Chemical | Ridership |
|--------|--------|-------|----------|-----------|
| TAG (White-box Baseline) | 0.10 | 0.47 | 0.05 | 0.15 |
| MTGNet | 0.22 | 0.54 | 0.34 | 0.61 |
| **ETAP** | **0.45** | **0.84** | **0.50** | **0.74** |

### Ablation Study

| Configuration | CelebA $R^2$ | Description |
|---------------|--------------|-------------|
| Affinity Score (White-box only) | Low/Unstable | Linear approximation, misaligned scale |
| Stage 1 Only (B-spline map) | Medium | High-order residuals uncorrected |
| **ETAP (Stage 1+2)** | **Highest & Stable** | Optimal two-stage ensemble |
| MTGNet (Same $|\mathcal{G}_\text{train}|$) | Unstable Negative | Highly unstable with small data |

### Key Findings

- ETAP at $|\mathcal{G}_\text{train}|=5$ outperforms MTGNet at $|\mathcal{G}_\text{train}|=50$, a 10$\times$ improvement in data efficiency.
- Compared to PCGrad (implicit gradient conflict handling), ETAP's explicit grouping strategy reduces loss by 7.4% on ETTm1 (4.20→3.89) and 6.6% on Ridership (18.84→17.59).
- ETAP affinity scores show extremely low variance (compared to TAG); Table 2 shows a standard deviation of $\pm 0.00$ on CelebA, indicating the time-averaging strategy significantly stabilizes estimates.

## Highlights & Insights

- **"Free" Affinity**: Reducing the cost of TAG's $n$ extra forward/backward passes to zero by utilizing gradients naturally computed during standard MTL training is a concise and elegant approach.
- **Bias-Variance Complementarity**: White-box methods offer low variance but high bias (linear averaging); data-driven methods offer low bias but high variance (unstable with small samples). Cascading them is a concrete realization of ensemble learning principles for MTL task scheduling.
- **Residual Learning Paradigm**: Learning residuals is much easier than learning gains from scratch, which is the core reason for high supervision efficiency, mirroring the residual logic in ResNet and Boosting.
- **Domain Generalization**: Effectiveness across four distinct domains—Vision (CelebA), Time Series (ETTm1), Molecular Classification (Chemical), and Traffic (Ridership)—demonstrates that the method does not rely on specific inductive biases.

## Limitations & Future Work

- Affinity score calculation still requires a full MTL training run; when task count $n$ is very large, the cost remains significant compared to lightweight surrogate models.
- Stage 2 residual regression uses multi-hot encoding, which may struggle with generalization as $n$ grows excessively large; more structured task representations (e.g., GNNs) might be needed.
- Current evaluations are limited to $n \leq 10$; scalability to large-scale MTL scenarios with hundreds of tasks (e.g., in NLP) remains to be verified.
- Task grouping assumes tasks can be grouped independently; scenarios with hard constraints (certain tasks must be grouped together) are not yet modeled.

## Related Work & Insights

- **vs. TAG (Fifty et al., 2021)**: TAG is the direct predecessor for the white-box component. ETAP uses a similar affinity formula but eliminates additional propagation overhead and compensates for TAG's linear estimation limits via the data-driven layer.
- **vs. MTGNet (Song et al., 2022)**: MTGNet uses self-attention Transformers for gain prediction, relying entirely on data. It is unstable with few training groups. ETAP's white-box prior addresses this deficiency.
- **vs. Linear Surrogate (Li et al., 2023)**: Under the same computational budget, ETAP improves F1 from 0.18 to 0.31 on ETTm1 and correlation from 0.49 to 0.57 on CelebA, proving the ensemble strategy is superior to pure linear surrogates.
- **Benefit**: The "White-box Prior + Residual Data-driven" framework can be transferred to other scenarios requiring subset evaluation, such as surrogate predictors in Neural Architecture Search (NAS) or client contribution estimation in Federated Learning.

## Rating

- Novelty: ⭐⭐⭐⭐ Clear ensemble of white-box and data-driven methods; the zero-overhead derivation of affinity is interesting.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage across four domains, multiple baselines, ablations, and data efficiency curves.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition, rigorous formulas, and highly readable charts.
- Value: ⭐⭐⭐⭐ Provides clear improvements to MTL task grouping, offering high engineering utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ALLNet: Multi-task Dense Prediction for Degraded Images](../../CVPR2026/others/allnet_multi-task_dense_prediction_for_degraded_images.md)
- [\[ICLR 2026\] CircuitNet 3.0: A Multi-Modal Dataset with Task-Oriented Augmentation for AI-Driven Circuit Design](circuitnet_30_a_multi-modal_dataset_with_task-oriented_augmentation_for_ai-drive.md)
- [\[ICLR 2026\] The Hot Mess of AI: How Does Misalignment Scale With Model Intelligence and Task Complexity?](the_hot_mess_of_ai_how_does_misalignment_scale_with_model_intelligence_and_task_.md)
- [\[NeurIPS 2025\] Exploiting Task Relationships in Continual Learning via Transferability-Aware Task Embeddings](../../NeurIPS2025/others/exploiting_task_relationships_in_continual_learning_via_transferability-aware_ta.md)
- [\[CVPR 2026\] Learning What Helps: Task-Aligned Context Selection for Vision Tasks](../../CVPR2026/others/learning_what_helps_task-aligned_context_selection_for_vision_tasks.md)

</div>

<!-- RELATED:END -->
