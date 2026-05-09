---
title: >-
  [Paper Note] Instance Generation for Meta-Black-Box Optimization through Latent Space Reverse Engineering
description: >-
  [AAAI 2026][Optimization][Meta-Black-Box Optimization] This paper proposes the LSRE framework, which constructs a two-dimensional latent instance space for BBO problems via an autoencoder, and employs enhanced genetic programming to reverse-engineer diverse synthetic optimization problem instances from this space, forming the Diverse-BBO benchmark that substantially improves the generalization performance of MetaBBO methods.
tags:
  - AAAI 2026
  - Optimization
  - Meta-Black-Box Optimization
  - Instance Generation
  - Genetic Programming
  - Latent Space
  - Benchmark
date: 2026-05-08
content_hash: 1452b4524d0bc781
---

# Instance Generation for Meta-Black-Box Optimization through Latent Space Reverse Engineering

**Conference**: AAAI 2026
**arXiv**: [2509.15810](https://arxiv.org/abs/2509.15810)
**Code**: [https://github.com/MetaEvo/Diverse-BBO](https://github.com/MetaEvo/Diverse-BBO)
**Area**: Optimization
**Keywords**: Meta-Black-Box Optimization, Instance Generation, Genetic Programming, Latent Space, Benchmark

## TL;DR

This paper proposes the LSRE framework, which constructs a two-dimensional latent instance space for BBO problems via an autoencoder, and employs enhanced genetic programming to reverse-engineer diverse synthetic optimization problem instances from this space, forming the Diverse-BBO benchmark that substantially improves the generalization performance of MetaBBO methods.

## Background & Motivation

### The Training Data Bottleneck in MetaBBO

Meta-Black-Box Optimization (MetaBBO) leverages the meta-learning paradigm to train neural network policies that automate BBO algorithm design (e.g., parameter control, algorithm selection). The core idea is to learn a generalizable design strategy over a large set of training problems, enabling the lower-level optimizer to perform well on unseen instances.

However, existing MetaBBO methods universally rely on classical benchmark function suites such as CoCo-BBOB as training data. These benchmarks exhibit the following limitations:

**Design Bias**: These functions are inherently hand-crafted to distinguish the performance of traditional optimization algorithms, rather than being tailored to train generalizable MetaBBO strategies.

**Insufficient Diversity**: The limited coverage of 24 base functions renders MetaBBO methods prone to overfitting, resulting in poor generalization to unseen problems.

**Inadequacy of Existing Alternatives**:
   - Collecting real-world problems requires deep domain expertise and incurs high simulation costs.
   - Random function generators lack control over the characteristics and diversity of generated instances.
   - MA-BBOB generates new instances via affine combinations of existing functions but is inherently constrained by the constituent functions.
   - GP-based methods such as GP-BBOB yield limited generation quality and are computationally expensive.

### Core Insight

The diversity of training data is critical to the generalization capability of MetaBBO. This paper presents the first systematic study on how to prepare training data with high generalization potential for MetaBBO, and proposes an instance generation method based on latent space reverse engineering.

## Method

### Overall Architecture

The LSRE (Latent Space Reverse Engineering) framework comprises three core steps:
1. Constructing a two-dimensional latent instance space using ELA features combined with an autoencoder.
2. Uniformly sampling target points on a grid within the latent space to obtain target latent representations.
3. Employing enhanced genetic programming to search for function expressions corresponding to each target point.

The pipeline ultimately produces Diverse-BBO, a benchmark comprising 256 diverse synthetic problem instances.

### Key Designs

#### 1. **Latent Instance Space Analysis**: Compressing High-Dimensional Problem Features into a Controllable Two-Dimensional Space

**Function**: Maps the high-dimensional ELA (Exploratory Landscape Analysis) features of BBO problem instances into a two-dimensional latent space, facilitating diversity measurement and uniform sampling.

**Mechanism**:
- The 24 base functions of CoCo-BBOB are extended across 5 dimensionalities $[2,5,10,30,50]$, yielding 120 functions.
- Each function is subjected to random rotation and translation: $f'(x) = f(\mathbf{R}^T(x-s))$, generating 270 instances per function for a total of 32,400 instances.
- The ELA feature vector $E_f$ is computed for each instance.
- An autoencoder (encoder $W_\theta$ + decoder $W_\phi$) is trained to compress ELA features into a two-dimensional latent space $\mathcal{H}$.
- The training objective is the reconstruction loss: $\mathcal{L}(\theta,\phi) = \frac{1}{2}\|E_f - E'_f\|_2^2$

**Design Motivation**:
- Direct sampling from the high-dimensional ELA space is neither efficient nor practical.
- The autoencoder outperforms PCA: different ELA feature groups are computed independently and do not satisfy the linear correlation assumption required by PCA. Experiments show that the first two principal components of PCA fail to exceed the 51% importance threshold for some instances.

#### 2. **Approximate Symbolic Space and Genetic Programming Search**: Reverse-Engineering Function Expressions from the Latent Space via GP

**Function**: Designs an expressive symbol set and employs genetic programming (GP) to search for function expressions that best match target latent points.

**Symbol Set Design**:
- **Operators** (15): add, sub, mul, div, neg, pow, sin, cos, tanh, exp, log, sqrt, abs, sum, mean (sum and mean aggregation operators are newly introduced to simplify multi-addend structures).
- **Operands** (3): $X$ (decision variable vector), $X[i:j]$ (indexed decision variables for fine-grained generation), $C$ (constant).

**Search Objective**: Given a target point $h \in \mathcal{H}$ in the latent space, find the symbolic tree $\tau^*$ such that:

$$\tau^* = \arg\min_\tau \frac{1}{2}\|W_\theta(E_\tau) - h\|_2$$

where $E_\tau$ is the ELA feature of the function corresponding to the symbolic tree.

**Search Strategy** (based on gplearn):
- Initialize a population of symbolic trees and evaluate their objective values.
- Iterative evolution: roulette-wheel reproduction → cross-dimensional local search → elite preservation.
- Return the best solution after $G$ generations.

#### 3. **Cross-Dimensional Local Search**: Addressing the Dimension–Feature Coupling Problem

**Function**: For each offspring symbolic tree, instantiation is performed across three dimensionalities (2D, 5D, 10D), and the dimensionality yielding the best objective value is selected.

**Mechanism**: Even when two functions share the same mathematical expression, different problem dimensionalities lead to different ELA features. Therefore, the search jointly optimizes over both the function expression and the problem dimensionality.

**Design Motivation**: Ablation studies demonstrate that this strategy contributes most significantly to LSRE performance, validating the hypothesis that the ELA feature space is not sufficiently smooth when dimensionality is not accounted for.

### Generation Pipeline and Acceleration

**Diverse-BBO Generation**:
- Uniformly sample $16 \times 16 = 256$ target points in the two-dimensional latent space $[-1,1] \times [-1,1]$.
- Invoke GP search for each point to reverse-engineer the corresponding function.
- Obtain 256 diverse problem instances in total.

**Distributed Acceleration**:
- Ray is used to distribute 256 GP processes across independent CPU cores.
- Secondary parallel evaluation is performed within each GP process.
- Total complexity is reduced from $\mathcal{O}[M \cdot G \cdot (N^2 + N \cdot L^2)]$ to $\mathcal{O}[G \cdot L^2]$.

## Key Experimental Results

### Main Results

#### Generalization Performance on Synthetic Test Set $\mathbb{D}_{synthetic}$

| Training Benchmark | DEDDQN | LDE | SYMBOL | GLEET | Avg. Rank |
|-------------------|--------|-----|--------|-------|-----------|
| **Diverse-BBO** | **0.8154** (Rank 1) | **0.8315** (Rank 1) | **0.7346** (Rank 1) | 0.8106 (Rank 2) | **1.25** |
| CoCo-BBOB | 0.8106 (Rank 2) | 0.8271 (Rank 2) | 0.7319 (Rank 2) | 0.8058 (Rank 4) | 2.5 |
| MA-BBOB | 0.8006 (Rank 4) | 0.8232 (Rank 3) | 0.7021 (Rank 3) | **0.8112** (Rank 1) | 2.75 |
| GP-BBOB | 0.8045 (Rank 3) | 0.8223 (Rank 4) | 0.6907 (Rank 4) | 0.8088 (Rank 3) | 3.5 |

Diverse-BBO achieves the best average rank across 4 MetaBBO methods, demonstrating superior generalization performance.

#### Real-World Problem Testing

Across three real-world problem suites — HPO (hyperparameter optimization), UAV (unmanned aerial vehicle planning), and Protein (protein docking) — MetaBBO methods trained on Diverse-BBO consistently achieve the best performance, further validating the importance of training set diversity for cross-domain generalization.

### Ablation Study

| Configuration | Search Objective Value (lower is better) | Description |
|--------------|------------------------------------------|-------------|
| **LSRE (Full)** | **Best** | All designs work jointly |
| LSRE-PCA | Significant degradation | PCA replaces autoencoder for dimensionality reduction |
| LSRE-Simple | Significant degradation | Simplified symbol set used |
| LSRE-NO_LS | Moderate degradation | Cross-dimensional local search removed |
| GP-BBOB | Worst | All new designs removed (baseline) |

**Key Findings**: The cross-dimensional local search contributes most significantly, followed by the autoencoder and the enhanced symbol set, which contribute comparably.

### Key Findings

1. Existing BBO benchmarks, despite their long-standing reputation, may be ill-suited for training MetaBBO.
2. The diversity of the training problem set significantly affects the generalization potential of MetaBBO.
3. Diverse-BBO achieves controllable, uniform, and fine-grained instance coverage in the latent space, substantially outperforming other benchmarks in coverage on real-problem localization maps.
4. Although MA-BBOB and GP-BBOB have been validated as useful for evaluating traditional BBO, their effectiveness is reversed in the MetaBBO context.

## Highlights & Insights

1. **First systematic study on MetaBBO training data**: Reframes the research question from "designing better meta-learning algorithms" to "preparing better training data" — a novel and compelling perspective.
2. **Latent space visualization**: The two-dimensional latent space visualization of instance distributions across benchmarks provides an intuitive explanation of the relationship between diversity and generalization.
3. **Distributed acceleration makes large-scale GP search tractable**: The two-level parallelism strategy reduces computational complexity by several orders of magnitude.
4. **General-purpose framework**: LSRE is extensible to user-defined target problem spaces beyond CoCo-BBOB.

## Limitations & Future Work

1. The current use of CoCo-BBOB as the target problem scope $\mathcal{D}$ may limit the coverage of generated instances.
2. The computational cost of GP search remains high despite distributed acceleration.
3. The choice of ELA features influences the results; more optimal feature combinations warrant further exploration.
4. Whether 256 instances suffice and whether the grid resolution $K=16$ is optimal require further investigation.
5. Training generative models directly in the latent space as an alternative to GP search is a promising direction for future work.

## Related Work & Insights

- **Instance Space Analysis (ISA)**: The approach of combining ELA features with dimensionality reduction to analyze BBO problem structure is generalizable to other optimization domains.
- **MetaBox**: A unified evaluation platform for MetaBBO; Diverse-BBO can be directly integrated into this framework.
- **GP for Symbolic Regression**: The symbol set design and search strategy proposed in this work offer reference value for the symbolic regression community.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First systematic study of the MetaBBO training data problem.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 4 MetaBBO methods × 4 benchmarks × multiple test sets; highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, though readability is slightly reduced in notation-dense sections.
- Value: ⭐⭐⭐⭐⭐ — Provides a plug-and-play training set and open-source tools, constituting a significant contribution to the MetaBBO community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Bridging Synthetic and Real Routing Problems via LLM-Guided Instance Generation and Progressive Adaptation](bridging_synthetic_and_real_routing_problems_via_llm-guided_instance_generation_.md)
- [\[ICLR 2026\] ∇-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space](../../ICLR2026/optimization/nabla-reasoner_llm_reasoning_via_test-time_gradient_descent_in_latent_space.md)
- [\[ICLR 2026\] Test-Time Meta-Adaptation with Self-Synthesis](../../ICLR2026/optimization/test-time_meta-adaptation_with_self-synthesis.md)
- [\[AAAI 2026\] MOTIF: Multi-strategy Optimization via Turn-based Interactive Framework](motif_multi-strategy_optimization_via_turn-based_interactive_framework.md)
- [\[AAAI 2026\] Co-Layout: LLM-driven Co-optimization for Interior Layout](co-layout_llm-driven_co-optimization_for_interior_layout.md)

</div>

<!-- RELATED:END -->
