---
title: >-
  [Paper Note] From Parameter to Representation: A Closed-Form Approach for Controllable Model Merging
description: >-
  [AAAI 2026][Recommender Systems][model merging] This paper proposes ReACT, which shifts controllable model merging from parameter-space optimization to representation-space correction. By deriving a closed-form solution, ReACT enables instant generation of Pareto-optimal models under arbitrary user preferences, achieving 36–208× speedup over existing methods while delivering superior performance.
tags:
  - AAAI 2026
  - Recommender Systems
  - model merging
  - controllable merging
  - Pareto front
  - representation correction
  - multi-objective optimization
date: 2026-05-08
content_hash: 2c2a8d23a1ea67d1
---

# From Parameter to Representation: A Closed-Form Approach for Controllable Model Merging

**Conference**: AAAI 2026
**arXiv**: [2511.10943](https://arxiv.org/abs/2511.10943)
**Code**: None
**Area**: Recommender Systems
**Keywords**: model merging, controllable merging, Pareto front, representation correction, multi-objective optimization

## TL;DR
This paper proposes ReACT, which shifts controllable model merging from parameter-space optimization to representation-space correction. By deriving a closed-form solution, ReACT enables instant generation of Pareto-optimal models under arbitrary user preferences, achieving 36–208× speedup over existing methods while delivering superior performance.

## Background & Motivation

**Background**: Model merging combines multiple task-specific models into a single generalist model, avoiding costly multi-task retraining. Existing approaches include weight averaging, Task Arithmetic, TIES-Merging, and AdaMerging, among others.

**Limitations of Prior Work**: Conventional methods produce static "one-size-fits-all" models that offer no user control over performance trade-offs across tasks. Although Pareto Merging (PM) and MAP achieve controllable merging, both rely on a compile-then-query paradigm—PM requires complex iterative training, while MAP employs evolutionary search whose complexity grows exponentially with the number of tasks.

**Key Challenge**: Controllable merging methods perform expensive multi-objective optimization in parameter space; any change to the task set requires restarting the process from scratch, making practical deployment prohibitively costly.

**Goal**: How can preference-aware model generation be achieved with minimal computational overhead?

**Key Insight**: The authors observe that performance degradation after model merging stems fundamentally from a **global linear distortion** (rotation + scaling) in representation space, rather than complex nonlinear deformation.

**Core Idea**: Construct a linear correction mapping in representation space, derive a closed-form Pareto-optimal solution, and bypass all iterative optimization.

## Method

### Overall Architecture

ReACT operates in two stages:
- **Offline stage**: Given a pre-merged backbone $f_{\text{merge}}$ and $T$ task-expert models $\{f_t\}$, collect a small calibration dataset per task, extract merged-model representations $\mathcal{Z}_t^{\text{mtl}}$ and expert representations $\mathcal{Z}_t^{\text{ind}}$, and compute reusable correction matrix components $\{\hat{W}_t, C_t\}$.
- **Online stage**: Given a user preference vector $\mathbf{p}$, instantly assemble the correction matrix $W_\mathbf{p}$ via a closed-form formula and apply a single linear transformation to representations at inference time.

### Key Designs

1. **Linear Representation Correction**

   - **Function**: Maps merged-model representations back to the task-expert representation space via a linear transformation matrix $W_t$.
   - **Mechanism**: For each task $t$, solve $\min_{W_t} \|W_t \mathcal{Z}_t^{\text{mtl}} - \mathcal{Z}_t^{\text{ind}}\|_F^2$, minimizing the Frobenius norm between corrected and expert representations.
   - **Design Motivation**: t-SNE visualizations show that the deviation between merged and expert representations is primarily a global rotation and scaling (linear distortion), making nonlinear methods unnecessary.

2. **Optimal Orthogonal Regularization**

   - **Function**: Uses the orthogonal Procrustes solution as a regularization prior to prevent overfitting of the correction matrix.
   - **Mechanism**: The regularized objective is $\min_{W_t} \|W_t \mathcal{Z}_t^{\text{mtl}} - \mathcal{Z}_t^{\text{ind}}\|_F^2 + \beta\|W_t - W_t^{\text{orth}}\|_F^2$, where $W_t^{\text{orth}}$ is obtained via the SVD-based Procrustes solution. The closed-form solution is $\hat{W}_t = (\mathcal{Z}_t^{\text{ind}} {\mathcal{Z}_t^{\text{mtl}}}^\top + \beta W_t^{\text{orth}})(\mathcal{Z}_t^{\text{mtl}} {\mathcal{Z}_t^{\text{mtl}}}^\top + \beta I)^{-1}$.
   - **Design Motivation**: Calibration data is typically scarce (unlabeled at test time), making pure least-squares prone to overfitting. The orthogonality constraint preserves the geometric structure of the representation space.

3. **Pareto-Optimal Representation Correction**

   - **Function**: Generates a globally optimal correction matrix for a given user preference $\mathbf{p}$.
   - **Mechanism**: The multi-objective problem is reduced to a single-objective via linear scalarization: $\min_W \sum_{t=1}^T p_t L_t(W)$. Since each $L_t(W)$ is a convex quadratic, the closed-form solution is $W_\mathbf{p} = (\sum_t p_t \hat{W}_t C_t)(\sum_t p_t C_t)^{-1}$, where $C_t = \mathcal{Z}_t^{\text{mtl}} {\mathcal{Z}_t^{\text{mtl}}}^\top + \beta I$.
   - **Design Motivation**: Rather than a simple weighted average $\sum p_t \hat{W}_t$, the data-aware weight matrices $C_t$ grant greater influence to tasks with more prominent feature structures.

### Loss & Training

No training is required. The entire pipeline is based on closed-form computation: SVD and matrix inversion. The only hyperparameter is the regularization strength $\beta$ (default 0.1), which is robust across a wide range of values. Complexity is $O(T D_{\text{rep}}^3)$; for ViT-B/32 with $D_{\text{rep}}=512$, computation takes only milliseconds.

## Key Experimental Results

### Main Results (8 ViT-B/32 Models Merged)

| Preference Setting | Method | Avg Acc (%) | Note |
|---|---|---|---|
| Equal | AMPP+PM | 85.2 | Prev. SOTA |
| Equal | AMPP+Ours | **85.4** | +0.2% |
| Priority | AMPP+PM | 85.9 | — |
| Priority | AMPP+Ours | **87.3** | +1.4% |
| One-hot | AMPP+PM | 83.6 | — |
| One-hot | AMPP+Ours | **88.9** | +5.3% |

### Ablation Study

| Configuration | Equal NAcc | HV | Note |
|---|---|---|---|
| Full model (Ours) | 93.5 | 70.2 | Complete model |
| Naive aggregation | 92.2 | 67.6 | Simple weighted average, lacks data-aware weights |
| Polar decomposition | 91.5 | 64.0 | Decomposed rotation and scaling, worse performance |
| $\beta=0$ (no regularization) | ~82 | — | Overfits calibration data |
| Pure orthogonal ($\beta \to \infty$) | ~94 | — | Lacks data-driven fine-grained correction |

### Key Findings
- **One-hot preference yields the largest gain** (+5.3%), demonstrating that the method does not collapse under extreme preferences as PM does.
- Using only **10% unlabeled calibration data** surpasses the full-data PM baseline.
- Data-aware aggregation outperforms naive weighted averaging by 2.6 HV, confirming that the $C_t$ weight matrices are central to the approach.
- Computational efficiency: merging 8 tasks requires only 0.056 GPU hours, compared to 2.0 hours for PM (36×) and 11.6 hours for MAP (208×).

## Highlights & Insights
- **Closed-form solution replacing iterative optimization**: Controllable model merging is recast from a search/training problem into a matrix algebra problem—an elegant paradigm shift. The core insight is that representation distortion is globally linear.
- **Data-aware aggregation**: $W_\mathbf{p}$ is not simply $\sum p_t \hat{W}_t$; the $C_t$ matrices naturally assign higher weights to tasks with larger feature variance.
- **Transferable perspective**: The representation-space correction paradigm can be extended to representation alignment between old and new tasks in continual learning, or to representation fusion across heterogeneous client models in federated learning.

## Limitations & Future Work
- **Limitations of linear scalarization**: Cannot cover all points on a concave Pareto front.
- **Dependence on unlabeled calibration data**: Although only 10% is needed, a small amount of test-time data is still required; fully zero-shot deployment is infeasible.
- **Scope of the linearity assumption**: Validated only on ViT-B/32; applicability to larger models or non-CLIP architectures remains unknown.
- **Classification tasks only**: All 8 datasets involve image classification; generalization to detection, segmentation, and generation tasks has not been verified.

## Related Work & Insights
- **vs. Pareto Merging (PM)**: PM performs iterative training in parameter space to learn a low-rank representation of the Pareto set; this work uses a closed-form solution in representation space, achieving 36× higher efficiency and 5.3% higher performance under one-hot preferences.
- **vs. MAP**: MAP relies on evolutionary search with exponential complexity growth; this work scales linearly with the number of tasks.
- **vs. Representation Surgery**: Representation Surgery uses per-sample MLPs for nonlinear correction; this work demonstrates that global linearity is sufficient and more data-efficient.

## Rating
- Novelty: ⭐⭐⭐⭐ The perspective shift from parameter space to representation space, combined with a closed-form solution, constitutes a distinctive contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Eight datasets, multiple preference settings, and complete ablations, though limited to classification and ViT-B/32.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical derivations are clear, visualizations are rich, and the narrative is coherent.
- Value: ⭐⭐⭐⭐ Provides a lightweight and elegant controllable merging solution; broader applicability remains to be explored.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Interpretable Reward Model via Sparse Autoencoder](interpretable_reward_model_via_sparse_autoencoder.md)
- [\[ICLR 2026\] Token-Efficient Item Representation via Images for LLM Recommender Systems](../../ICLR2026/recommender/token-efficient_item_representation_via_images_for_llm_recommender_systems.md)
- [\[ICLR 2026\] GoalRank: Group-Relative Optimization for a Large Ranking Model](../../ICLR2026/recommender/goalrank_group-relative_optimization_for_a_large_ranking_model.md)
- [\[NeurIPS 2025\] Measuring What Matters: Construct Validity in Large Language Model Benchmarks](../../NeurIPS2025/recommender/measuring_what_matters_construct_validity_in_large_language_model_benchmarks.md)
- [\[AAAI 2026\] Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation](tokenize_once_recommend_anywhere_unified_item_tokenization_for_multi-domain_llm-.md)

</div>

<!-- RELATED:END -->
