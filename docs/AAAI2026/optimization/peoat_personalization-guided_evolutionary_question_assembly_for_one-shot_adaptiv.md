---
title: >-
  [Paper Note] PEOAT: Personalization-Guided Evolutionary Question Assembly for One-Shot Adaptive Testing
description: >-
  [AAAI 2026][Optimization][Adaptive Testing] This paper is the first to propose the One-shot Adaptive Testing (OAT) task, formulating it as a combinatorial optimization problem…
tags:
  - "AAAI 2026"
  - "Optimization"
  - "Adaptive Testing"
  - "Evolutionary Algorithm"
  - "Personalization"
  - "Combinatorial Optimization"
  - "Intelligent Education"
date: 2026-05-08
content_hash: 765b1058055c41d4
---

# PEOAT: Personalization-Guided Evolutionary Question Assembly for One-Shot Adaptive Testing

**Conference**: AAAI 2026
**arXiv**: [2512.00439](https://arxiv.org/abs/2512.00439)  
**Code**: None  
**Area**: Optimization
**Keywords**: Adaptive Testing, Evolutionary Algorithm, Personalization, Combinatorial Optimization, Intelligent Education

## TL;DR

This paper is the first to propose the One-shot Adaptive Testing (OAT) task, formulating it as a combinatorial optimization problem, and introduces the PEOAT framework—combining personalized initialization, cognitively enhanced evolutionary search, and diversity-preserving selection—to select an optimal question set for each examinee in a single shot without interactive feedback, substantially outperforming conventional CAT methods.

## Background & Motivation

### Practical Limitations of Computerized Adaptive Testing (CAT)

CAT efficiently assesses examinee ability through interactive item selection and incremental ability estimation. The typical pipeline is: selection module $\mathcal{M}_\pi$ selects items based on current ability estimate → examinee responds → diagnostic module $\mathcal{M}_d$ updates ability estimate → repeat.

CAT methods fall into two categories:
- **Heuristic methods** (MKLI, BECAT, MAAT): item selection based on interpretable rules (e.g., maximum Fisher information, KL divergence)
- **Data-driven methods** (BOBCAT, NCAT, GMOCAT, UATS): learning personalized item selection policies via reinforcement learning and similar approaches

**Core Problem**: The interactive, real-time item-by-item selection in CAT is impractical in the following scenarios:

**Large-scale examinations**: High interaction cost makes dynamic per-item selection infeasible.

**Psychological assessments**: Noise and distraction must be minimized; item-by-item feedback may compromise assessment quality.

**Remote/offline testing**: Device constraints and response latency limit interactivity.

**Resource-constrained environments**: Multi-round interaction is unsupportable in time-sensitive settings.

### One-Shot Adaptive Testing (OAT) Task

This paper is the first to define OAT: given an examinee's initial ability estimate $\theta_i^0$, select an optimal fixed-length question set $\mathcal{J}_i$ of length $L$ in a single shot; after the examinee completes all items, a single-step ability update is performed to obtain the final ability estimate $\theta_i^{final}$.

Three key challenges of OAT:

**Student adaptivity**: Without intermediate feedback, item–individual ability matching must be ensured during optimization.

**Enormous search space**: The number of combinations for selecting $L$ items from a large candidate pool is exponential.

**Encoding sparsity**: The candidate item bank far exceeds the test length, leading to a dimensionality curse in the encoded representation.

### Modeling Approach

OAT is naturally formulated as a bi-level combinatorial optimization problem:
- Outer level: select item subset $\mathcal{J}_i$
- Inner level: estimate student ability based on simulated response data
- Objective: the final ability estimate should be as close as possible to the student's true ability

$$\mathcal{J}_i^* = \arg\max_{\mathcal{J}_i \subseteq \mathcal{Q}_i^{untested}} \mathcal{F}(\theta_i^{final}(\mathcal{J}_i), \hat{\theta}_i)$$

## Method

### Overall Architecture

PEOAT consists of three core modules:
1. Personalization-aware population initialization → constructs an informative and diverse initial population
2. Cognitively enhanced evolutionary search → leverages cognitive signals for effective exploration
3. Diversity-preserving environmental selection → balances fitness and diversity

### Key Designs

#### 1. **Personalization-Aware Population Initialization**: Multi-strategy Sampling Based on Ability–Difficulty Matching

**Function**: Adaptively constructs an informative and diverse initial population based on the distance between the student ability vector $\boldsymbol{\theta}_i$ and item difficulty vector $\boldsymbol{\alpha}_j$.

**Encoding scheme**: Each individual is an index sequence of length $L$, $\mathcal{X}_i^{(j)} = [x_1, x_2, \ldots, x_L]$, with no repeated indices.

**Personalized distance vector**:
$$\delta_j = \|\boldsymbol{\theta}_i - \boldsymbol{\alpha}_j\|_2, \quad \forall j \in \{1, 2, \ldots, |\mathcal{Q}_i|\}$$

**Three-strategy initialization**:
- **$\mathcal{O}_{match}$ (matching strategy)**: uniformly samples $L$ items from the $2L$ items with the smallest distances → prioritizes difficulty-matched items
- **$\mathcal{O}_{diverse}$ (diversity strategy)**: uniformly samples from the $2L$ items with the largest distances → explores ability boundaries
- **$\mathcal{O}_{rand}$ (random strategy)**: uniformly samples from items with intermediate distances → introduces randomness

Each individual randomly adopts one strategy, ensuring overall population diversity.

**Design Motivation**: Purely random initialization lacks personalized prior knowledge, leading to an oversized search space; purely matching initialization lacks diversity. The three-strategy mechanism balances exploitation and exploration.

#### 2. **Cognitively Enhanced Evolutionary Search**: Fisher-Information-Guided Mutation

**Pattern-preserving uniform crossover**:
- Generate a binary mask $m_k \sim \text{Bernoulli}(0.5)$
- Two offspring exchange genes at corresponding positions according to the mask
- Repair operator $\mathcal{T}(\cdot)$ resolves duplicate items by random replacement from the unselected item pool

**Cognitively guided mutation**:
- Randomly remove one gene $x_{off}$
- Use the Frobenius norm of the Fisher information matrix as a scalar information gain:

$$\mathbf{I}_j(\boldsymbol{\theta}_i) = |\boldsymbol{\alpha}_j|^2 \cdot p_j(1-p_j)$$

where $p_j = \sigma(\boldsymbol{\theta}_i^\top \boldsymbol{\alpha}_j)$ is the IRT-based correct response probability.

- Construct a categorical sampling distribution weighted by normalized information gain:
$$P(x_j \in \mathcal{Z}) = \frac{\mathbf{I}_j}{\sum_{k \in \mathcal{Z}} \mathbf{I}_k}$$

**Mechanism**: Items with high information gain (near the ability threshold, high discrimination) are sampled with higher probability. This is more efficient than random mutation and ensures newly inserted genes are both personalized and informative.

#### 3. **Diversity-Preserving Environmental Selection**: Hamming Distance Filtering + Elitism

**Fitness evaluation**:
- Simulate the OAT process: student completes selected items → diagnostic model performs virtual parameter update → evaluate on held-out test set
- Combined metric: $\mathcal{F} = (\mathcal{F}_{auc} + \mathcal{F}_{acc})/2$

**Selection mechanism**:
1. Rank by fitness and retain top-$k$ elites ($k = \lfloor |\mathcal{P}|/2 \rfloor$)
2. Filter remaining candidates via Hamming distance:
    - Encode each individual as a binary bit string
    - Compute the minimum Hamming distance to the elite pool
    - Retain only candidates with $\text{HamDist} > \tau$

**Design Motivation**: Pure elitist selection causes premature population convergence; Hamming distance filtering preserves genotypic diversity and avoids local optima.

### Loss & Training

- Diagnostic models: MIRT and NCD as two backbones
- Population size 20, 15 evolutionary generations, crossover rate 0.8, mutation rate 0.2
- Test length $L \in \{5, 10, 15, 20\}$
- Distance threshold $\tau$ searched over $\{0.5, 0.75, 1, 1.25, 1.5\}$

## Key Experimental Results

### Main Results

#### JUNYI Dataset (MIRT Backbone)

| Method | Type | length=5 ACC/AUC | length=10 | length=15 | length=20 |
|------|------|-----------------|-----------|-----------|-----------|
| RAND | Heuristic | 67.98/68.24 | 74.48/73.64 | 79.60/77.73 | 82.47/80.48 |
| MKLI | Heuristic | 70.14/70.27 | 78.03/76.64 | 83.26/81.39 | 86.07/84.27 |
| BOBCAT | Data-driven | 69.15/71.86 | 77.05/77.60 | 81.66/81.12 | 84.29/83.43 |
| NCAT | Data-driven | 71.19/73.48 | 80.23/77.37 | 82.69/81.43 | 84.93/84.04 |
| UATS | Data-driven | 70.83/74.45 | 80.33/77.19 | 83.13/81.27 | 84.38/84.65 |
| **PEOAT** | **Ours** | **79.64/83.05** | **85.38/85.85** | **86.39/86.68** | **86.85/87.83** |

At length=5, PEOAT surpasses the second-best method by **10.61%/10.35%** (ACC/AUC), a substantial margin.

#### PTADisc Dataset (NCD Backbone)

| Method | length=5 | length=10 | length=15 | length=20 |
|------|----------|-----------|-----------|-----------|
| NCAT | 66.38/68.09 | 67.64/69.22 | 68.48/69.61 | 70.17/70.40 |
| GMOCAT | 66.47/68.24 | 67.48/68.97 | 69.14/69.49 | 69.73/70.36 |
| **PEOAT** | **69.37/70.84** | **73.65/73.58** | **75.44/75.07** | **75.91/74.93** |

PEOAT achieves comprehensive superiority across all test lengths and both diagnostic model backbones.

### Ablation Study

| Configuration | length=5 | length=10 | length=15 | length=20 |
|------|----------|-----------|-----------|-----------|
| w/o PI (no personalized initialization) | Largest drop | Largest drop | Largest drop | Largest drop |
| w/o CE (no cognitively enhanced evolution) | Moderate drop | Moderate drop | Moderate drop | Moderate drop |
| w/o ES (no diversity-preserving selection) | Slight drop | Slight drop | Slight drop | Slight drop |
| **Full PEOAT** | **Best** | **Best** | **Best** | **Best** |

**Key Findings**: Personalized initialization contributes the most, indicating that embedding personalized priors into the initial population quality is critical for final performance.

#### PEOAT vs. PEOAT-B (Baseline without Specific Designs)

| CDM | Method | length=5 | length=10 | length=15 | length=20 |
|-----|------|----------|-----------|-----------|-----------|
| MIRT | PEOAT-B | 78.35/81.97 | 84.12/84.48 | 84.96/85.21 | 85.73/86.55 |
| MIRT | **PEOAT** | **79.64/83.05** | **85.38/85.85** | **86.39/86.68** | **86.85/87.83** |
| NCD | PEOAT-B | 73.27/81.81 | 80.64/85.19 | 84.73/87.59 | 86.11/88.80 |
| NCD | **PEOAT** | **74.56/83.06** | **81.90/86.47** | **85.85/88.86** | **87.34/89.78** |

Even the baseline version substantially outperforms CAT baselines, confirming that formulating OAT as combinatorial optimization is the right direction in itself.

### Key Findings

1. PEOAT's advantage is most pronounced at shorter test lengths (10%+ gain at length=5), demonstrating its greatest value in rapid assessment scenarios.
2. Formulating OAT as combinatorial optimization and solving it with evolutionary algorithms constitutes an effective paradigm.
3. Personalized priors have a greater impact on population quality than improvements to evolutionary operators.
4. Fisher-information-guided mutation is more effective than random mutation in cognitive diagnostic settings.

## Highlights & Insights

1. **OAT task definition**: The paper is the first to propose and formalize a novel, practically meaningful educational assessment task, bridging the gap between CAT and static testing.
2. **Combinatorial optimization perspective**: Naturally connecting educational testing with evolutionary optimization represents an innovative interdisciplinary approach.
3. **Fisher-information-driven mutation**: Cleverly integrates Item Response Theory (IRT) with evolutionary computation, grounding the mutation operator in educational psychology.
4. **High practical value**: OAT directly serves high-demand scenarios including large-scale examinations, psychological assessments, and offline testing.

## Limitations & Future Work

1. Each student requires an independent evolutionary search (20 individuals × 15 generations), with computational cost scaling linearly with the number of examinees.
2. Performance depends on the quality of the pre-trained cognitive diagnostic model (MIRT/NCD); errors in the diagnostic model propagate to item selection.
3. Only fixed-length testing is considered; adaptive-length item selection is not explored.
4. Future work could explore end-to-end learning of item selection policies via deep reinforcement learning.
5. Item exposure rate control is not considered (different examinees in the same batch may receive identical items).

## Related Work & Insights

- **BECAT**: Uses full-response gradient approximation to guide CAT item selection; theoretically rigorous.
- **NCAT**: Models CAT as a bi-level reinforcement learning problem with an attention-based selection policy.
- **PEGA**: The first work to apply evolutionary algorithms to personalized exercise assembly, but targets practice recommendation rather than examinations.
- **Evolutionary optimization in education**: Genetic algorithms have been applied to cognitive diagnosis (HGA-CDM); OAT further extends the application scope.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — OAT task proposed for the first time; combinatorial optimization modeling approach is novel
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two real educational datasets, multiple baselines, ablation analysis, but lacks large-scale scenario testing
- Writing Quality: ⭐⭐⭐⭐ — Mathematically rigorous and clear; problem definition is well-formalized
- Value: ⭐⭐⭐⭐ — Practical application value for intelligent education, though the scenario is relatively niche

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] pFed1BS: Personalized Federated Learning with Bidirectional Communication Compression via One-Bit Random Sketching](personalized_federated_learning_with_bidirectional_communication_compression_via.md)
- [\[AAAI 2026\] Bridging Synthetic and Real Routing Problems via LLM-Guided Instance Generation and Progressive Adaptation](bridging_synthetic_and_real_routing_problems_via_llm-guided_instance_generation_.md)
- [\[ICCV 2025\] Class-Wise Federated Averaging for Efficient Personalization](../../ICCV2025/optimization/class-wise_federated_averaging_for_efficient_personalization.md)
- [\[AAAI 2026\] Pareto-Grid-Guided Large Language Models for Fast and High-Quality Heuristics Design in Multi-Objective Combinatorial Optimization](pareto-grid-guided_large_language_models_for_fast_and_high-quality_heuristics_de.md)
- [\[NeurIPS 2025\] Verbalized Algorithms: Zero-shot Classical Algorithmic Reasoning for Correctness and Runtime Guarantees](../../NeurIPS2025/optimization/verbalized_algorithms.md)

</div>

<!-- RELATED:END -->
