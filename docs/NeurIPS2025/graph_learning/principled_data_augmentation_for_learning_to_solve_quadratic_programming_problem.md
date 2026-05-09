---
title: >-
  [Paper Note] Principled Data Augmentation for Learning to Solve Quadratic Programming Problems
description: >-
  [NeurIPS 2025][Graph Learning][Data Augmentation] This paper proposes a principled data augmentation framework based on affine transformations of the KKT system, generating optimality-preserving augmented instances for MPNN-based learning-to-optimize (L2O) on linear programming (LP) and quadratic programming (QP) tasks. Combined with contrastive pretraining, the framework substantially improves performance in data-scarce and out-of-distribution (OOD) generalization settings.
tags:
  - NeurIPS 2025
  - Graph Learning
  - Data Augmentation
  - Quadratic Programming
  - Graph Neural Networks
  - Contrastive Learning
  - Learning to Optimize
  - KKT Conditions
date: 2026-05-08
content_hash: c43d6c9b509afc39
---

# Principled Data Augmentation for Learning to Solve Quadratic Programming Problems

**Conference**: NeurIPS 2025
**arXiv**: [2506.01728](https://arxiv.org/abs/2506.01728)
**Authors**: Chendi Qian, Christopher Morris (RWTH Aachen University)
**Code**: [GitHub](https://github.com/chendiqian/Data-Augmentation-for-Learning-to-Optimize)
**Area**: Graph Learning
**Keywords**: Data Augmentation, Quadratic Programming, Graph Neural Networks, Contrastive Learning, Learning to Optimize, KKT Conditions

## TL;DR

This paper proposes a principled data augmentation framework based on affine transformations of the KKT system, generating optimality-preserving augmented instances for MPNN-based learning-to-optimize (L2O) on linear programming (LP) and quadratic programming (QP) tasks. Combined with contrastive pretraining, the framework substantially improves performance in data-scarce and out-of-distribution (OOD) generalization settings.

## Background & Motivation

### State of the Field
Linear programming and quadratic programming are fundamental optimization problems in machine learning, operations research, and scientific computing. In recent years, message-passing graph neural network (MPNN)-based L2O methods have attracted wide attention. These approaches encode LP/QP instances as constraint-variable bipartite graphs and learn, in a data-driven manner, to solve or assist solvers (e.g., by approximating strong branching scores).

### Limitations of Prior Work
- **Data scarcity**: L2O methods require large amounts of labeled data (optimal solutions), and labeling is costly, particularly for complex QP instances.
- **General graph augmentations are inapplicable**: Operations in methods such as GraphCL — node dropping, edge perturbation, feature masking — destroy the constraint structure and optimality conditions of LP/QP instances, degrading performance.
- **Absence of structure-preserving augmentation strategies**: Existing L2O work has largely not explored how to design augmentations that preserve both structure and optimality for LP/QP.
- **Gap in contrastive pretraining**: Although graph contrastive learning has succeeded on general graph tasks, effective unsupervised pretraining strategies for LP/QP L2O remain absent.

### Root Cause
There is a need for a theoretically grounded data augmentation framework that can generate structurally diverse, optimality-preserving LP/QP instances without re-invoking a solver, while supporting both supervised training and contrastive pretraining paradigms.

## Method

### Core Idea: Affine Transformations of the KKT System
Given a linearly constrained quadratic program (LCQP) instance $I = (\mathbf{Q}, \mathbf{A}, \mathbf{b}, \mathbf{c})$, the optimal primal-dual solution satisfies the KKT conditions, which can be written compactly as:

$$\begin{bmatrix} \mathbf{Q} & \mathbf{A}^\top \\ \mathbf{A} & \mathbf{0} \end{bmatrix} \begin{bmatrix} \mathbf{x}^* \\ \boldsymbol{\lambda}^* \end{bmatrix} = \begin{bmatrix} -\mathbf{c} \\ \mathbf{b} - \mathbf{s}^* \end{bmatrix}$$

The key observation is that applying affine transformations $\mathbf{M}$ (multiplicative) and $\mathbf{B}, \boldsymbol{\beta}$ (additive) to the KKT system yields a new valid KKT system, from which a new QP instance can be constructed and its optimal solution recovered directly via matrix operations — without calling a solver.

### Conditions for Valid Transformations
For the transformed system to constitute a valid KKT form, the following must hold:
1. $\mathbf{T}_{11} = \mathbf{M}_1 \mathbf{Q} \mathbf{M}_1^\top$ is positive definite (convexity of the new objective).
2. $\mathbf{T}_{22} = \mathbf{0}$ (consistency of constraint structure).
3. $\mathbf{T}_{12} = \mathbf{T}_{21}^\top$ (symmetry).
4. $\mathbf{M}_{22}$ satisfies the conditions in Proposition 2.1 (recoverability of the primal solution).

### Six Concrete Transformations

| Transformation | Operation | Efficiently Recoverable | Solution-Agnostic |
|---------------|-----------|:-----------------------:|:-----------------:|
| Remove free variables | Drop variables whose optimal values are zero | ✓ | ✗ |
| Remove inactive constraints | Drop strictly inactive inequality constraints | ✓ | ✗ |
| Scale variable coefficients | Diagonal $\mathbf{M}_1$ scales variable coefficients in objective and constraints | ✓ | ✓ |
| Scale constraints | Diagonal $\mathbf{M}_2$ scales constraint coefficients and right-hand sides | ✓ | ✓ |
| Add variables | Introduce new variables via $\mathbf{M}_1 = [\mathbf{I}; \mathbf{q}^\top]$ | ✓ | ✓ |
| Add constraints | Append new inequality constraints to the problem | ✓ | ✓ |

"Efficiently recoverable" means the new solution can be computed from the original in $O(n)$ time; "solution-agnostic" means the transformation parameters do not depend on the original optimal solution and can thus be used in unsupervised settings.

### Training Paradigms

**Supervised augmentation**: Transformations are applied dynamically during training to generate augmented instances, and an MPNN is trained with a supervised loss to predict optimal objective values.

**Contrastive pretraining**: Self-supervised pretraining is performed using the NT-Xent loss. For each instance, two augmented views are generated via solution-agnostic transformations as positive pairs, with other instances serving as negatives. Pretraining trains only the MPNN backbone, which is subsequently fine-tuned with an MLP head under supervision.

## Key Experimental Results

### Experiment 1: Data Augmentation under Supervised Learning

Augmentation methods are evaluated on synthetic LP/QP datasets (100 variables × 100 constraints, 10,000 instances). The metric is mean relative objective error (%).

| Augmentation | LP 10% | LP 20% | LP 100% | QP 10% | QP 20% | QP 100% |
|-------------|--------|--------|---------|--------|--------|---------|
| None | 8.539 | 6.149 | 2.784 | 5.304 | 3.567 | 1.240 |
| Drop node | 8.618 | 7.131 | 4.654 | 6.355 | 4.867 | 2.594 |
| Mask node | 8.979 | 7.591 | 3.216 | 5.865 | 4.005 | 1.347 |
| Flip edge | 7.794 | 6.503 | 4.358 | 5.485 | 4.361 | 2.561 |
| **Drop vars (Ours)** | 4.996 | 3.484 | 1.484 | 4.232 | 2.374 | 0.835 |
| **Drop cons (Ours)** | 5.563 | 3.691 | 1.656 | **3.407** | **2.104** | 0.821 |
| **Scale vars (Ours)** | **4.682** | **3.208** | **1.241** | 3.790 | 2.468 | **0.649** |
| **Combo (Ours)** | **3.465** | **2.300** | **1.051** | **2.434** | **1.532** | **0.542** |

General graph augmentations (Drop/Mask/Flip) degrade performance. The combined proposed method (Combo) achieves a relative improvement of 62.6% under the LP 20% setting.

### Experiment 2: Contrastive Pretraining and Fine-tuning

The full unlabeled training set is used for contrastive pretraining, followed by fine-tuning on varying fractions of labeled data.

| Pretraining | LP 10% | LP 20% | LP 100% | QP 10% | QP 20% | QP 100% |
|------------|--------|--------|---------|--------|--------|---------|
| None | 8.539 | 6.149 | 2.784 | 5.304 | 3.567 | 1.240 |
| GraphCL | 9.694 | 6.033 | 2.613 | 6.024 | 3.865 | 1.253 |
| GCC | 8.248 | 5.761 | 2.686 | 11.344 | 6.356 | 1.472 |
| IGSD | 18.097 | 7.631 | 3.082 | 12.799 | 6.306 | 1.302 |
| MVGRL | 20.392 | 9.072 | 2.852 | 8.440 | 4.932 | 1.343 |
| **Ours** | **3.472** | **2.794** | **1.588** | **3.791** | **2.427** | **0.926** |

The proposed pretraining reduces error from 8.539 to 3.472 (59.4% improvement) under LP 10%, and from 5.304 to 3.791 (28.5% improvement) under QP 10%. Most existing graph contrastive learning methods (IGSD, MVGRL, etc.) degrade performance.

### Experiment 3: OOD Generalization

Pretrained models are fine-tuned on unseen optimization problem families to assess transfer learning capability.

**LP OOD Generalization (Set Cover / MIS / CA / CFL)**:

| Problem | Pretrained | 10% | 20% | 100% |
|---------|:----------:|-----|-----|------|
| Set Cover | No | 5.297 | 2.531 | 0.752 |
| Set Cover | **Yes** | **1.965** | **1.349** | **0.662** |
| CA | No | 2.381 | 1.803 | 0.906 |
| CA | **Yes** | **2.303** | **1.713** | **0.753** |

**QP OOD Generalization (SVM / Portfolio / LASSO)**:

| Problem | Pretrained | 10% | 20% | 100% |
|---------|:----------:|-----|-----|------|
| SVM | No | 0.191 | 0.109 | 0.027 |
| SVM | **Yes** | **0.141** | **0.077** | 0.024 |
| LASSO | No | 5.405 | 4.083 | 2.159 |
| LASSO | **Yes** | **5.169** | **3.726** | **1.178** |

The pretrained model outperforms training from scratch on nearly all OOD tasks, demonstrating strong transferability.

## Highlights & Insights

- **Theoretical elegance**: The affine transformation framework over the KKT system rests on a unified mathematical foundation; all transformations carry rigorous theoretical guarantees (optimality-preservation, convexity-preservation), in contrast to heuristic graph augmentations.
- **Practical efficiency**: All six transformations admit $O(n)$-time solution recovery without re-invoking a QP solver; solution-agnostic transformations are directly applicable to unsupervised pretraining.
- **Substantial improvements**: The combined augmentation reduces error by over 60% in data-scarce settings; contrastive pretraining consistently outperforms GraphCL and analogous methods on both LP and QP.
- **OOD generalization**: The pretrained model successfully transfers to diverse problem families including Set Cover, SVM, and Portfolio optimization, demonstrating that the model learns general structural representations of optimization problems.
- **Compelling negative evidence**: Experiments clearly show that general graph augmentation methods (node dropping, edge perturbation, etc.) are harmful on LP/QP tasks, underscoring the necessity of domain-specific augmentation design.

## Limitations & Future Work

- **Restricted to LCQP**: The framework requires $\mathbf{Q}$ to be positive definite and does not extend to non-convex QP, semidefinite programming (SDP), or the integer components of mixed-integer linear programming (MILP).
- **Limited transformation diversity**: Only four solution-agnostic transformations are available in practice (scaling variables/constraints, adding variables/constraints); removal operations require access to the optimal solution.
- **Single regression task**: Evaluation is limited to objective value prediction; the framework has not been tested on more practical L2O applications such as strong branching scoring or cutting plane selection.
- **Small problem scale**: Experiments are fixed at 100 variables × 100 constraints, leaving scalability to large-scale instances (thousands of variables) insufficiently validated.
- **Simple pretraining strategy**: Only the NT-Xent contrastive loss is used; more advanced self-supervised learning methods (e.g., BYOL, VICReg) are not explored.
- **Ad hoc combination strategy**: The Combo method applies transformations randomly without an adaptive selection mechanism to choose the optimal transformation subset based on instance characteristics.

## Related Work & Insights

- **GraphCL (You et al., 2020)**: General graph augmentations (node dropping / edge perturbation / feature masking) degrade performance on LP/QP by destroying the constraint structure of optimization instances.
- **GCC (Qiu et al., 2020)**: Random-walk-based subgraph sampling leads to severe degradation on QP (11.344 vs. 5.304).
- **Li et al. (2024c)**: The only contrastive approach targeting LP; it adopts a CLIP-style formulation, whereas the proposed framework is more general (covering both LP and QP) with theoretically grounded transformations.
- **Duan et al. (2022)**: Designs satisfiability-preserving augmentations for SAT problems — a conceptually related idea, though the present work addresses continuous optimization.
- **Gasse et al. (2019)**: Seminal work on MPNN for strong branching in MILP, but does not consider data augmentation.
- **Chen et al. (2024a)**: Extends the bipartite graph representation of LP to QP by adding edges between variables; this graph representation is adopted as the MPNN input in the present work.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic KKT-theoretic data augmentation framework for LP/QP L2O.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers supervised, pretraining, and OOD settings with comprehensive baselines, though problem scales are limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear, transformation taxonomy is systematic, and experimental design is well-motivated.
- **Value**: ⭐⭐⭐⭐ — Fills a gap in LP/QP data augmentation with a concise and practical approach; application scope warrants further expansion.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Learning Repetition-Invariant Representations for Polymer Informatics](learning_repetition-invariant_representations_for_polymer_informatics.md)
- [\[NeurIPS 2025\] MoEMeta: Mixture-of-Experts Meta Learning for Few-Shot Relational Learning](moemeta_mixture-of-experts_meta_learning_for_few-shot_relational_learning.md)
- [\[AAAI 2026\] Self-Correction Distillation for Structured Data Question Answering](../../AAAI2026/graph_learning/self-correction_distillation_for_structured_data_question_answering.md)
- [\[NeurIPS 2025\] Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks](sketch-augmented_features_improve_learning_long-range_dependencies_in_graph_neur.md)
- [\[AAAI 2026\] Format as a Prior: Quantifying and Analyzing Bias in LLMs for Heterogeneous Data](../../AAAI2026/graph_learning/format_as_a_prior_quantifying_and_analyzing_bias_in_llms_for_heterogeneous_data.md)

<!-- RELATED:END -->
