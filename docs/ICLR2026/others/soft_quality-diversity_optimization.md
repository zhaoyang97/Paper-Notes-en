---
title: >-
  [Paper Note] Soft Quality-Diversity Optimization
description: >-
  [ICLR 2026][Others][quality diversity] The authors propose Soft QD Score as a new optimization objective for quality-diversity that eliminates the need for behavior space discretization. Based on this, they derive a differentiable algorithm, SQUAD, which exhibits superior scalability in high-dimensional behavior spaces and maintains competitive performance
tags:
  - ICLR 2026
  - Others
  - quality diversity
  - optimization
  - differentiable
  - evolutionary computation
  - soft objectives
date: 2026-05-08
content_hash: 139100fd838bdb19
---
# Soft Quality-Diversity Optimization

## Meta-Information
- **Conference**: ICLR 2026
- **arXiv**: [2512.00810](https://arxiv.org/abs/2512.00810)
- **Code**: [https://github.com/conflictednerd/soft-qd](https://github.com/conflictednerd/soft-qd)
- **Area**: LLM Evaluation
- **Keywords**: quality diversity, optimization, differentiable, evolutionary computation, soft objectives

## TL;DR
The authors propose Soft QD Score as a new optimization objective for quality-diversity that eliminates the need for behavior space discretization. Based on this, they derive a differentiable algorithm, SQUAD, which exhibits superior scalability in high-dimensional behavior spaces and maintains competitive performance with SOTA on standard benchmarks.

## Background & Motivation
- **Quality-Diversity (QD) Optimization**: Aims to find a set of solutions that are both high-quality and behaviorally diverse. Applications include RL policy diversification, red teaming, and content generation.
- **Limitations of Prior Work**:
  1. Discretizing behavior space into grids or CVT cells suffers from the curse of dimensionality—cell counts grow exponentially or cell volumes expand exponentially.
  2. Non-differentiable discretization hinders gradient-based optimization.
- **Key Challenge**: Can a QD objective be designed without discretization to allow direct differentiable optimization on continuous behavior spaces?

## Method

### Overall Architecture
SQUAD reformulates quality-diversity optimization from "filling archives in discrete cells" to "maximizing a soft objective over a continuous behavior space." It uses Gaussian kernels to spread each solution's quality into a "luminance field" that decays with distance. The total luminance of the entire space is defined as the Soft QD Score. A differentiable lower bound of this objective is derived, allowing $N$ solutions to simultaneously optimize quality while repelling each other via gradients, thereby bypassing the curse of dimensionality inherent in grid discretization.

### Key Designs

**1. Behavior Value and Soft QD Score: Spreading Quality into a Continuous "Luminance Field"**

Traditional QD partitions the behavior space into grids and places the optimal solution in each cell. The cell count explodes exponentially with dimensionality, which is the primary pain point SQUAD aims to avoid. SQUAD treats each solution $\theta_n$ as a light source illuminating the behavior space: its brightness is proportional to its quality $f_n = f(\theta_n)$, and its influence decays as a Gaussian function of the distance to its behavior descriptor $\mathbf{b}_n = \text{desc}(\theta_n)$. Thus, the behavior value at any point $\mathbf{b}$ in space is the maximum luminance of all sources:

$$v_{\bm{\theta}}(\mathbf{b}) = \max_{1 \leq n \leq N} f_n \exp\!\left(-\|\mathbf{b} - \mathbf{b}_n\|^2 / 2\sigma^2\right).$$

By integrating this over the entire behavior space, the scalar objective Soft QD Score is obtained: $S(\bm{\theta}) = \int_{\mathcal{B}} v_{\bm{\theta}}(\mathbf{b})\, d\mathbf{b}$. To maximize this integral, high-quality solutions must be spread across the space as much as possible—quality and diversity are naturally encoded into a single continuous, everywhere-defined objective without relying on cell partitions.

**2. Theoretical Properties: Validating Objective Behavior and Compatibility**

Theorem 1 provides three properties that establish the "luminance field" as a credible objective. **Monotonicity**: Adding a solution or improving the quality of an existing one does not decrease $S(\bm{\theta})$. **Submodularity**: The marginal contribution of new solutions decreases, meaning reward is lower when adding solutions to already dense regions, naturally pushing solutions toward empty areas. **Limit Equivalence**: As the kernel width $\sigma \to 0$, the Gaussian kernel shrinks to a spike, and the Soft QD Score converges to the traditional QD Score (up to a constant factor). This shows Soft QD is a continuous smooth generalization of the old objective, with $\sigma$ as the "softness" control.

**3. SQUAD Differentiable Lower Bound: Replacing Integrals with Pairwise Attraction-Repulsion Forces**

Although the Soft QD Score is continuous, the integral has no closed-form solution and cannot be directly optimized via gradients on $\bm{\theta}$. SQUAD (Soft QD Using Approximated Diversity) maximizes a computable and differentiable lower bound instead:

$$\tilde{S}(\bm{\theta}) = \sum_{n=1}^N f_n - \sum_{1 \leq i < j \leq N} \sqrt{f_i f_j}\, \exp\!\left(-\|\mathbf{b}_i - \mathbf{b}_j\|^2 / \gamma^2\right).$$

The first term is the quality term, acting as an attraction force driving each solution to improve its $f_n$. The second term is the diversity term, penalizing behaviorally similar solution pairs like a repulsion force. Optimization becomes a balance between attraction and repulsion. Crucially, repulsion strength is weighted by the geometric mean of qualities $\sqrt{f_i f_j}$: repulsion is only significant when both solutions are already high-quality. This creates a natural curriculum: poor solutions first improve $f_n$, and as quality rises, repulsion kicks in to disperse them. The hyperparameter $\gamma$ controls the repulsion radius.

**4. Efficient Implementation: Reducing $O(N^2)$ Repulsion to Linear and Handling Bounded Spaces**

The diversity term in the lower bound requires $O(N^2)$ complexity. Since the exponential decay makes distant solutions negligible, SQUAD only calculates repulsion for the $k$-nearest neighbors in behavior space, reducing complexity to $O(Nk)$. Mini-batch updates further reduce memory usage. Additionally, while the lower bound assumes an unbounded space ($\mathcal{B}=\mathbb{R}^d$), many problems have descriptors restricted to bounded intervals. SQUAD uses a logit transform to map bounded spaces to unbounded ones before optimization, which is critical for performance in bounded domains.

### A Complete Example
Using the simultaneous optimization of $N$ solutions as an example: after initialization, an outer loop runs for $T$ steps. In each step, mini-batches are processed: for each solution in the batch, $K$-nearest neighbors are identified to compute the lower bound $\tilde{S}$ and its gradient. Parameters are updated via Adam, and qualities $f_n$ and descriptors $\mathbf{b}_n$ are re-evaluated for the next round. Early low-quality solutions focus on quality gradients; as they improve, the strengthened repulsion terms push them apart, leading the population to self-organize into a high-quality, space-filling distribution.

## Key Experimental Results

### Main Results 1: Scalability in High-Dimensional Behavior Spaces (Linear Projection)

The LP benchmark compares scalability across 4, 8, and 16-dimensional behavior spaces:

| Behavior Dim | CMA-MAEGA / CMA-MEGA | Sep-CMA-MAE / GA-ME / DNS | **SQUAD** |
|-----------|-----------|-------------|---------|
| 4D | Slightly dominant | Significantly behind | Highly competitive |
| 8D | Diminishing lead | Degenerates | **Competitive/Surpassing** |
| 16D | Surpassed | Severely degenerates | **Best in both metrics** |

> Methods utilizing descriptor gradients (SQUAD, CMA-M(A)EGA) significantly outperform gradient-free methods. SQUAD's lack of discretization provides a growing advantage as dimensionality increases, showing the lowest variance and highest stability.

### Main Results 2: Image Collection (IC) and Latent Space Illumination (LSI)

| Method | IC QD Score | IC Vendi Score | LSI QD Score | LSI Vendi Score |
|------|-----------|---------------|-------------|----------------|
| CMA-MAEGA | Top tier | High | Top tier | High |
| Sep-CMA-MAE | High | Medium | High | Medium |
| DNS-G | Medium | High | Medium | High |
| **SQUAD** | Competitive | **Highest** | Competitive | **Highest** |

> SQUAD consistently achieves the best diversity metrics (Vendi Score) and competes with SOTA on QD Score.

### Ablation Study: Impact of $\gamma$

| $\gamma$ Value | Average Quality | Vendi Score | Description |
|-----------|---------|-------------|------|
| Small | Highest | Lowest | Weak repulsion $\to$ Bias toward quality |
| Medium | High | High | Balanced |
| Large | Lower | Highest | Strong repulsion $\to$ Bias toward diversity |

> $\gamma$ intuitively controls the trade-off between quality and diversity.

### Key Findings
1. Cell-based methods fail in high-dimensional behavior spaces due to the curse of dimensionality.
2. SQUAD's continuous objective naturally avoids discretization issues.
3. The geometric mean in the repulsion force creates a "quality-first, diversity-second" curriculum.
4. Logit transformation is essential for bounded behavior spaces.
5. $K$-nearest neighbor approximation has minimal impact on the final results due to exponential decay.

## Highlights & Insights
- **Paradigm Shift**: Moving from discrete cells to continuous soft objectives avoids the curse of dimensionality.
- **Elegant Physical Analogy**: Attraction (quality) + Repulsion (diversity) = Self-organized equilibrium.
- **Theoretical Rigor**: Provides a solid foundation via monotonicity, submodularity, and limit equivalence.
- **Clever Geometric Mean**: Automatically implements a latent curriculum without manual scheduling.

## Limitations
- The lower bound approximation ignores higher-order interactions, which may be inaccurate in extremely tight clusters.
- The $\gamma$ hyperparameter requires tuning and depends on problem scale and behavior space structure.
- Unlike archive-based methods, it lacks explicit solution storage for querying specific behaviors.
- Not directly applicable to non-differentiable objectives (e.g., RL problems requiring black-box simulators).

## Related Work
- **QD Algorithms**: MAP-Elites (Cully et al., 2015), CMA-MEGA (Fontaine & Nikolaidis, 2021/2023)
- **Differentiable QD**: DQD (Fontaine & Nikolaidis, 2021), PGA-ME (Nilsson & Cully, 2021)
- **Novelty Search**: Lehman & Stanley (2011), DNS (Bahlous-Boldi et al., 2025)
- **Diversity Metrics**: Vendi Score (Friedman & Dieng, 2023)

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Soft QD redefines the objective of QD optimization, a paradigmatic contribution.
- Theoretical Depth: ⭐⭐⭐⭐ — Includes monotonicity, submodularity, limit equivalence, and lower bound derivation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Evaluated across three domains, high-dimensional scalability, and ablation analysis.
- Value: ⭐⭐⭐⭐ — A viable solution for high-dimensional QD, though limited to differentiable objectives.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces](discount_model_search_for_quality_diversity_optimization_in_high-dimensional_mea.md)
- [\[ICML 2025\] Diversity By Design: Leveraging Distribution Matching for Offline Model-Based Optimization](../../ICML2025/others/diversity_by_design_leveraging_distribution_matching_for_offline_model-based_opt.md)
- [\[ACL 2025\] Evaluating the Evaluation of Diversity in Commonsense Generation](../../ACL2025/others/evaluating_the_evaluation_of_diversity_in_commonsense_generation.md)
- [\[AAAI 2026\] HyperSHAP: Shapley Values and Interactions for Explaining Hyperparameter Optimization](../../AAAI2026/others/hypershap_shapley_values_and_interactions_for_explaining_hyperparameter_optimiza.md)
- [\[ACL 2025\] A Multi-Persona Framework for Argument Quality Assessment](../../ACL2025/others/a_multi-persona_framework_for_argument_quality_assessment.md)

</div>

<!-- RELATED:END -->
