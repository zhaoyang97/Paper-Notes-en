---
title: >-
  [Paper Note] Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces
description: >-
  [ICLR2026][LLM Evaluation][Quality Diversity] This paper proposes Discount Model Search (DMS), which replaces the histogram-based discrete representation in CMA-MAE with a neural network that fits a continuous, smooth discount function. This addresses the issue of search stagnation caused by distortion in high-dimensional measure spaces, and enables, for the first time, the direct use of image datasets to define measure spaces (the QDDM paradigm).
tags:
  - ICLR2026
  - LLM Evaluation
  - Quality Diversity
  - MAP-Elites
  - CMA-MAE
  - Discount Model
  - High-Dimensional Measure Space
date: 2026-05-08
content_hash: bb35680dbf5c7880
---

# Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces

**Conference**: ICLR2026  
**arXiv**: [2601.01082](https://arxiv.org/abs/2601.01082)  
**Code**: [discount-models.github.io](https://discount-models.github.io)  
**Area**: LLM Evaluation  
**Keywords**: Quality Diversity, MAP-Elites, CMA-MAE, Discount Model, High-Dimensional Measure Space  

## TL;DR

This paper proposes Discount Model Search (DMS), which replaces the histogram-based discrete representation in CMA-MAE with a neural network that fits a continuous, smooth discount function. This addresses the issue of search stagnation caused by distortion in high-dimensional measure spaces, and enables, for the first time, the direct use of image datasets to define measure spaces (the QDDM paradigm).

## Background & Motivation

Quality Diversity (QD) optimization aims to find a set of solutions that are both high-quality and diverse: each solution must not only maximize an objective function $f$, but also achieve broad coverage in the output space of a user-defined measure function $\bm{m}$. Classical applications include robotic controller search, generative modeling, and LLM red-teaming.

The state-of-the-art black-box QD algorithm CMA-MAE uses a histogram to partition the measure space into discrete cells, storing a scalar discount value in each cell to guide the search. However, in high-dimensional measure spaces, **distortion** causes many solutions to map to a narrow region of the measure space, so numerous solutions fall into the same cell and receive identical discount values. This prevents the algorithm from distinguishing improvement directions among these solutions, causing rapid search stagnation.

The authors validate this phenomenon empirically: on the 10D LP (Sphere) benchmark, CMA-MAE samples 540 solutions per iteration, yet the number of solutions falling into distinct cells drops sharply from several hundred to approximately 30 over time, demonstrating that high-dimensional distortion severely degrades the search signal.

## Core Problem

1. **Amplified distortion in high dimensions**: As measure space dimensionality increases, each cell's volume grows exponentially, causing more solutions with similar measures to be grouped into the same cell. CMA-MAE assigns them identical discount values, preventing CMA-ES from identifying the direction of maximum archive improvement.
2. **Increasing archive resolution is infeasible**: Although smaller cells can mitigate distortion, the required memory grows exponentially with dimensionality.
3. **Lack of application paradigms for high-dimensional measures**: Traditional QD considers only hand-crafted measures of fewer than 10 dimensions, making it difficult to extend to scenarios where high-dimensional data such as images serve as measures.

## Method

### Overall Architecture

DMS retains the MAP-Elites-style archive and CMA-ES emitter, but replaces the histogram with a neural network $\hat{f}_A(\cdot; \psi)$ to represent the discount function. The core procedure is a two-phase loop:

**Phase 1: Search**

- Each emitter samples $\lambda$ solutions from a Gaussian distribution $\mathcal{N}(\bm{\theta}^*, \bm{\Sigma})$.
- For each solution $\bm{\theta}_i$, the objective value $f(\bm{\theta}_i)$ and measure $\bm{m}(\bm{\theta}_i)$ are computed.
- The improvement is computed via the discount model: $\Delta_i = f(\bm{\theta}_i) - \hat{f}_A(\bm{m}(\bm{\theta}_i))$.
- CMA-ES distribution parameters are updated by ranking solutions according to $\Delta_i$, guiding the search toward the direction of maximum archive improvement.
- If a solution outperforms the current occupant of its cell, it replaces that occupant.

**Phase 2: Training the Discount Model**

At each iteration, a training set $\mathcal{D}_A$ is constructed from two types of data:

1. **Solution data**: For each solution sampled by an emitter, an entry $(\bm{m}(\bm{\theta}), t_A)$ is generated, where the target $t_A$ follows the threshold update rule from CMA-MAE:

$$t_A = \begin{cases} \hat{f}_A(\bm{s}) & \text{if } f(\bm{\theta}) \leq \hat{f}_A(\bm{s}) \\ (1-\alpha)\hat{f}_A(\bm{s}) + \alpha f(\bm{\theta}) & \text{if } f(\bm{\theta}) > \hat{f}_A(\bm{s}) \end{cases}$$

2. **Empty cell data**: $n_{empty}$ unoccupied cell centers are randomly sampled from the archive, with target set to $f_{min}$, preventing the model from producing inflated discount values in unexplored regions.

### Key Designs

- **Continuity and smoothness**: As a neural network, the model naturally outputs a continuous function, yielding distinct discount values even for solutions with very similar measures, thereby providing accurate gradient directions.
- **Flexible architecture**: MLPs are used for low-dimensional measures; CNNs, Transformers, or other architectures can be adopted for image or text measures.
- **"Empty Points" regularization**: A clamping mechanism for unexplored regions ensures the model outputs reasonably low discount values in unseen parts of the measure space.
- **Archive learning rate $\alpha$**: Controls the exploration–exploitation trade-off; $\alpha=1$ corresponds to pure exploration and $\alpha=0$ to pure objective optimization.

### QDDM: Defining Measure Spaces via Datasets

DMS enables a novel QD paradigm—Quality Diversity with Datasets of Measures (QDDM):

- Instead of hand-crafting low-dimensional measure functions, the desired measure space is defined directly by a dataset (e.g., a collection of images).
- When constructing the CVT archive, dataset samples serve as Voronoi centroids.
- Under the manifold hypothesis, high-dimensional data lie on a low-dimensional manifold, so the CVT need only partition the subspace of interest to the user.
- The distance function can be chosen flexibly (Euclidean distance, CLIP score, etc.).

## Key Experimental Results

### Benchmark Results (LP Series, 20 Trials)

| Benchmark | DMS QD Score | CMA-MAE QD Score | DMS Coverage | CMA-MAE Coverage |
|---|---|---|---|---|
| 2D LP (Sphere) | **6,978** | 6,328 | **95.9%** | 81.0% |
| 10D LP (Sphere) | **6,410** | 609 | **89.2%** | 7.0% |
| 20D LP (Sphere) | **7,406** | 882 | **96.0%** | 9.1% |
| 50D LP (Sphere) | **6,991** | 2,327 | **87.0%** | 24.2% |
| 10D LP (Rastrigin) | **5,139** | 247 | **88.2%** | 3.0% |

The advantage of DMS is particularly pronounced in high-dimensional settings: on 10D LP (Sphere), DMS achieves a QD Score **10.5×** that of CMA-MAE, with coverage increasing from 7% to 89%.

### QDDM Domains (5 Trials)

| Domain | DMS QD Score | CMA-MAE QD Score | DMS Coverage | CMA-MAE Coverage |
|---|---|---|---|---|
| TA (MNIST) | 951.56 | **954.27** | **99.84%** | 99.48% |
| TA (F-MNIST) | **701.14** | 625.65 | **72.28%** | 63.92% |
| LSI (Hiker) | **214.91** | 14.61 | **3.77%** | 1.56% |

- The high coverage on TA (MNIST) indicates that not all QDDM domains exhibit strong distortion.
- On LSI (Hiker), DMS substantially outperforms CMA-MAE (QD Score 215 vs. 15), though the absolute coverage remains low (3.77%), reflecting the challenges of complex QDDM domains.
- DMS even surpasses DDS—a method specifically designed for diversity—on the diversity-only LP (Flat) domain.

### Computational Overhead

DMS is 2–3× slower than CMA-MAE on LP benchmarks due to discount model training. In QDDM domains, however, solution evaluation (e.g., StyleGAN3 rendering) becomes the bottleneck, making the algorithmic overhead difference negligible.

## Highlights & Insights

1. **Clear and compelling core insight**: The idea of replacing a discrete histogram with a continuous model is elegant and highly effective, achieving order-of-magnitude improvements in dimensions above 10.
2. **QDDM paradigm innovation**: This is the first work to propose using image datasets to directly define measure spaces, lowering the barrier to QD adoption—users need only provide a target dataset rather than hand-crafting measure functions.
3. **Compelling LSI (Hiker) demonstration**: Generated hiker images visibly match clothing styles to terrain (heavy jackets for snowy mountains, light attire for beaches), intuitively illustrating the method's practical value.
4. **Comprehensive experiments**: Covering 9 benchmarks and 3 QDDM domains, with rigorous statistical testing (Welch ANOVA + Games-Howell) over 20/5 trials.
5. **Complete ablation study**: Confirms the critical roles of $\alpha$ and $n_{empty}$.

## Limitations & Future Work

1. **Discount model noise**: In domains requiring precise objective optimization (e.g., TA (MNIST)), model error introduces noise into improvement rankings, preventing DMS from surpassing the exact histogram of CMA-MAE.
2. **Extremely low coverage on LSI (Hiker)**: At only 3.77%, exploration remains far from sufficient in highly complex, high-dimensional QDDM domains.
3. **Computational cost**: DMS is approximately 2–3× slower than CMA-MAE on LP benchmarks; the overhead of training the discount model is non-trivial at scale.
4. **Distance function selection for the CVT archive**: Only Euclidean distance and CLIP score are explored; better distance metrics may further improve performance.
5. **DDS cannot be run on QDDM domains**: KDE runtime scales linearly with dimensionality, limiting the completeness of comparisons.
6. **Absence of non-image QDDM experiments**: Although audio and text measures are mentioned, they are not empirically validated.

## Related Work & Insights

| Method | Core Mechanism | High-Dim. Support | Optimization Target |
|---|---|---|---|
| MAP-Elites | Random mutation + grid archive | Poor (exponential memory) | QD |
| CMA-MAE | CMA-ES + histogram discount | Poor (same-cell stagnation) | QD |
| DDS | KDE density estimation | Moderate (slow KDE) | Diversity only |
| **DMS** | CMA-ES + neural discount model | **Strong** | QD |

DMS inherits the archive improvement framework of CMA-MAE but replaces the discrete histogram with a continuous model. It also draws on the intuition from DDS that smooth signals facilitate exploration. Unlike DDS, DMS jointly optimizes for both objective value and diversity.

The approach of **replacing hand-crafted functions with datasets** has broad transferability: in robotic policy search, target behavior demonstrations can replace hand-designed behavior descriptors; in LLM red-teaming, adversarial sample collections can define diversity directions. The idea of **replacing discrete counters with continuous models** parallels the classical progression from count-based exploration to neural density estimation in reinforcement learning (e.g., RND, ICM). The use of CLIP score as a distance function in QDDM further suggests that pretrained model representations can be leveraged to define measure space structure in other high-dimensional settings.

## Rating
- Novelty: ⭐⭐⭐⭐ (Both the continuous discount model and the QDDM paradigm are original contributions)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (12 domains, rigorous statistical testing, comprehensive ablations)
- Writing Quality: ⭐⭐⭐⭐ (Motivation is clearly articulated; Figure 1 provides an intuitive comparison)
- Value: ⭐⭐⭐⭐ (High-dimensional QD and the QDDM paradigm have strong practical application potential)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Soft Quality-Diversity Optimization](soft_quality-diversity_optimization.md)
- [\[ICLR 2026\] Prompt and Parameter Co-Optimization for Large Language Model Task Adaptation](prompt_and_parameter_co-optimization_for_large_language_models.md)
- [\[ICLR 2026\] How Reliable is Language Model Micro-Benchmarking?](how_reliable_is_language_model_micro-benchmarking.md)
- [\[ICLR 2026\] Function Spaces Without Kernels: Learning Compact Hilbert Space Representations](function_spaces_without_kernels_learning_compact_hilbert_space_representations.md)
- [\[AAAI 2026\] GDBA Revisited: Unleashing the Power of Guided Local Search for Distributed Constraint Optimization](../../AAAI2026/llm_evaluation/gdba_revisited_unleashing_the_power_of_guided_local_search_for_distributed_const.md)

</div>

<!-- RELATED:END -->
