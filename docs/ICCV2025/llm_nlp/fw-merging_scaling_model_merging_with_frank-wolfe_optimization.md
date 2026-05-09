---
title: >-
  [Paper Note] FW-Merging: Scaling Model Merging with Frank-Wolfe Optimization
description: >-
  [ICCV 2025][LLM/NLP][Model merging] This paper formalizes model merging as a constrained optimization problem and introduces FW-Merging, a Frank-Wolfe optimization-inspired method that iteratively selects the most relevant models and performs local merging. The approach achieves scalable and robust merging over large black-box model pools, surpassing the data-aware method AdaMerging by 8.39% when merging 20 ViT models.
tags:
  - ICCV 2025
  - LLM/NLP
  - Model merging
  - Frank-Wolfe optimization
  - multi-task learning
  - scalability
  - large language models
date: 2026-05-08
content_hash: 9bffe286af71f94e
---

# FW-Merging: Scaling Model Merging with Frank-Wolfe Optimization

**Conference**: ICCV 2025
**arXiv**: [2503.12649](https://arxiv.org/abs/2503.12649)
**Code**: Available (open-sourced as mentioned in the paper)
**Area**: Model Merging / Multi-task Learning
**Keywords**: Model merging, Frank-Wolfe optimization, multi-task learning, scalability, large language models

## TL;DR

This paper formalizes model merging as a constrained optimization problem and introduces FW-Merging, a Frank-Wolfe optimization-inspired method that iteratively selects the most relevant models and performs local merging. The approach achieves scalable and robust merging over large black-box model pools, surpassing the data-aware method AdaMerging by 8.39% when merging 20 ViT models.

## Background & Motivation

Model merging has emerged as a data-efficient alternative to multi-task learning. However, with the rapid growth of the open-source AI ecosystem, existing methods face two critical limitations:

**Lack of adaptability to unknown models**: Existing methods adjust merging coefficients based on known model capability information, performing poorly on models from diverse and partially unknown sources, and failing to distinguish between high-quality and low-quality fine-tuned models.

**Inability to scale effectively**: Performance degrades severely when merging a large number of unknown model checkpoints. The authors' analytical experiments show that adding 16 irrelevant models causes a performance drop of 18.9%–64.4%.

An ideal merging method should satisfy two fundamental scaling properties: (1) adding irrelevant models does not degrade performance; (2) adding relevant models leads to steady performance improvement.

## Method

### Overall Architecture

FW-Merging consists of an iterative process with three core stages:
1. **Relevance Assessment**: Constructs a linear approximation of the objective function using gradients from the current model to identify the most beneficial direction for improvement.
2. **Model Selection**: Selects the most relevant checkpoint from the candidate pool by minimizing the linear approximation.
3. **Knowledge Integration**: Integrates the selected checkpoint into the current model using an orthogonal merging method.

### Key Designs

1. **Constrained Optimization Formulation**: Redefines model merging as minimizing an objective function $\ell(\theta)$ over the convex hull $\mathcal{M} = \text{conv}(\{\theta_i^*\}_{i=1}^n)$. Proposition 1 establishes equivalence with the traditional coefficient optimization formulation. The key advantage is that the Linear Minimization Oracle (LMO) reduces to:

$$\text{LMO}(\{\theta_i^*\}, \theta_t) = \arg\min_{s \in \{\theta_1^*,...,\theta_n^*\}} \langle \nabla\ell(\theta_t), s \rangle$$

which requires only inner product minimization over a finite vertex set and is computationally efficient.

2. **Hard FW vs. Soft FW**:

    - **Hard LMO**: Selects the argmin of the linear subproblem as the merging direction; simple and direct.
    - **Soft LMO**: Selects the top-$k$ vertices and performs internal optimization of their merging coefficients via projected gradient descent onto the simplex. The update rule is $\theta_{t+1} = \theta_t + \sum_{j=1}^k \lambda_j^*(\tilde{s}_j - \theta_t)$.
    - Theorem 1 proves that Soft FW achieves a convergence rate of $O(1/T)$, superior to the vanilla $O(1/\sqrt{T})$.

3. **Task-wise vs. Layer-wise LMO**:

    - Task-wise LMO: Solves the LMO by vectorizing the entire model weight.
    - Layer-wise LMO: Defines the constraint set as the Cartesian product of per-layer convex hulls $\mathcal{M} = \mathcal{M}_1 \times \cdots \times \mathcal{M}_L$, selecting the best model independently per layer; can be viewed as a block-coordinate Frank-Wolfe algorithm.
    - $\text{FW}_{hard}$ is better suited to layer-wise LMO (7.2-point gain on NLP discriminative tasks), while $\text{FW}_{soft}$ is better suited to task-wise LMO (since internal coefficient optimization already operates at the layer level).

### Loss & Training

- Objective function: Task-specific cross-entropy loss minimized on training data.
- $\text{FW}_{hard}$: 10 iterations for NLP, 3 for CV, initialized from Task Arithmetic results.
- $\text{FW}_{soft}$: 15 iterations for CV, initialized from the pretrained model.
- Only 100 training samples per task are required (vs. 2.9K samples/task for conventional MTL).
- Constant memory overhead: a fixed number of models are loaded at each step; simultaneous storage of all models is unnecessary.

## Key Experimental Results

### Main Results

**Vision tasks (merging 20 ViT-B/32 models)**:

| Method | SUN397 | Cars | GTSRB | DTD | Avg. |
|--------|--------|------|-------|-----|------|
| Pretrained | 62.3 | 59.7 | 32.6 | 43.8 | 49.6 |
| Task Arithmetic | 20.4 | 12.2 | 29.8 | 22.3 | 21.2 |
| Ties-Merging | 51.0 | 36.2 | 57.7 | 40.6 | 46.4 |
| AdaMerging | 66.4 | 70.1 | 95.1 | 64.0 | 73.9 |
| Surgery | 69.7 | 71.8 | 96.6 | 73.4 | 77.9 |
| **FW_soft (Ours)** | **72.9** | **74.8** | **96.8** | **76.0** | **80.1** |

**Language tasks**:

| Method | 4 Discriminative Tasks (8 models) | 3 Generative Tasks (16 models) | Avg. Normalized Score |
|--------|-----------------------------------|--------------------------------|-----------------------|
| Traditional MTL | 73.1 | 81.2 | 77.2 |
| Task Arithmetic | 80.8 | 75.9 | 78.4 |
| Ties-Merging | 64.3 | 78.5 | 71.4 |
| **FW_hard (Ours)** | **85.4** | **81.1** | **83.1** |

### Ablation Study

**Scaling experiments (effect of adding irrelevant/relevant models)**:

| #Models | Irrelevant Model Scenario | | Relevant Model Scenario | |
|---------|--------------------------|---|-------------------------|---|
| | Task Arith. | FW_soft | Task Arith. | FW_soft |
| 4 | 70.3 | 74.1 | 59.2 | 59.2 |
| 12 | 47.9 | **74.1** | 52.3 | **67.5** |
| 20 | 21.2 | **74.2** | 36.3 | **68.3** |

FW-Merging demonstrates desirable scaling properties: performance does not degrade after adding 16 irrelevant models (74.1→74.2), and improves by 15.3% after adding 16 relevant models.

**Design variant ablation**:

| Coefficients | Method | LMO | CV Score |
|--------------|--------|-----|----------|
| Optimized | FW_soft | Task-wise | 80.1 |
| Optimized | FW_soft | Layer-wise | 79.7 |
| Not optimized | FW_soft | Task-wise | 70.3 |
| — | FW_hard | Layer-wise | 74.0 |

Optimizing merging coefficients yields gains of up to 9.9 points; task-wise LMO slightly outperforms layer-wise LMO for $\text{FW}_{soft}$.

### Key Findings

- FW-Merging requires only 100 samples per task and 2 minutes, yet surpasses conventional MTL which requires 2.9K samples and 4.2 hours.
- The checkpoint with the smallest linear approximation value is precisely the most relevant model to the target task (validated in Figure 3), indicating that the gradient inner product is a reliable relevance indicator.
- The method demonstrates strong robustness to noisy models (models initialized from different pretraining starting points).
- FW-Merging can be combined with other merging functions (e.g., Ties-Merging) for further improvement.

## Highlights & Insights

- **Elegant theoretical framework**: Establishes a correspondence between model merging and the classical Frank-Wolfe optimization algorithm, providing convergence guarantees and clear geometric intuition.
- **Practical scalability**: Constant memory overhead combined with robustness to irrelevant models is critical for large-scale model merging in the HuggingFace ecosystem.
- **Exceptional data efficiency**: Only 100 samples per task, making the method suitable for privacy-sensitive and data-scarce scenarios.
- **Strong orthogonality**: Serves as a general framework that can be combined with existing merging methods.

## Limitations & Future Work

- The internal optimization (coefficient solving in Soft FW) may increase overall computational cost.
- Only Task Arithmetic is used as the MergeFn to ensure convex hull feasibility; more complex merging functions may violate constraints yet yield better empirical results.
- Validation on larger-scale models (e.g., 70B+) has not been conducted.
- The choice between layer-wise and task-wise LMO lacks an adaptive mechanism and must be manually selected based on the scenario.
- The objective function design relies on a small amount of task-aligned data, making the method inapplicable in fully unsupervised settings.

## Related Work & Insights

- Task Arithmetic pioneered the paradigm of task vector operations in weight space.
- The parameter conflict resolution approaches of TIES-Merging and DARE are complementary to FW-Merging's model selection mechanism.
- AdaMerging's test-time entropy optimization contrasts with FW-Merging's training-set cross-entropy optimization.
- Frank-Wolfe algorithms hold broad promise for constrained optimization in deep learning.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Innovatively bridges classical optimization algorithms with model merging, with solid theoretical contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers NLP discriminative/generative and CV tasks, with detailed scalability analysis and complete ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Mathematical derivations are rigorous, intuitive figures are clear, and theory is tightly integrated with experiments.
- **Value**: ⭐⭐⭐⭐⭐ Establishes a new paradigm for scalable model merging with significant practical value for the open-source model ecosystem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Do Not Merge My Model! Safeguarding Open-Source LLMs Against Unauthorized Model Merging](../../AAAI2026/llm_nlp/do_not_merge_my_model_safeguarding_open-source_llms_against_unauthorized_model_m.md)
- [\[NeurIPS 2025\] SolverLLM: Solving Optimization Problems via Test-Time Scaling with LLM-Guided Search](../../NeurIPS2025/llm_nlp/solverllm_leveraging_test-time_scaling_for_optimization_problem_via_llm-guided_s.md)
- [\[ICCV 2025\] VIM: Versatile Interactive Motion-Language Model](vim_versatile_interactive_motion_language_model.md)
- [\[ICCV 2025\] Balancing Task-Invariant Interaction and Task-Specific Adaptation for Unified Image Fusion](balancing_task-invariant_interaction_and_task-specific_adaptation_for_unified_im.md)
- [\[NeurIPS 2025\] Scaling Up Active Testing to Large Language Models](../../NeurIPS2025/llm_nlp/scaling_up_active_testing_to_large_language_models.md)

</div>

<!-- RELATED:END -->
