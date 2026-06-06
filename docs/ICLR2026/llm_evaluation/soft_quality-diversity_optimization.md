---
title: >-
  [Paper Note] Soft Quality-Diversity Optimization
description: >-
  [LLM Evaluation] This paper proposes the Soft QD Score as a novel quality-diversity optimization objective that eliminates the need for behavior space discretization, and derives a differentiable algorithm, SQUAD…
tags:
  - "LLM Evaluation"
date: 2026-05-08
content_hash: dbc434b4cee91ef6
---

# Soft Quality-Diversity Optimization

## Metadata
- **Conference**: ICLR 2026
- **arXiv**: [2512.00810](https://arxiv.org/abs/2512.00810)
- **Code**: [https://github.com/conflictednerd/soft-qd](https://github.com/conflictednerd/soft-qd)
- **Area**: LLM Evaluation
- **Keywords**: quality diversity, optimization, differentiable, evolutionary computation, soft objectives

## TL;DR
This paper proposes the Soft QD Score as a novel quality-diversity optimization objective that eliminates the need for behavior space discretization, and derives a differentiable algorithm, SQUAD, which scales more effectively to high-dimensional behavior spaces while achieving competitive performance on standard benchmarks.

## Background & Motivation
- **Quality-Diversity (QD) Optimization**: Seeks a collection of solutions that are both high-quality and behaviorally diverse, with applications in RL policy diversification, red-teaming, and content generation.
- **Limitations of Prior Work**:
  1. Discretizing the behavior space into grid/CVT cells suffers from the curse of dimensionality — the number of cells grows exponentially or cell volumes become exponentially large.
  2. Non-differentiable discretization impedes gradient-based optimization.
- **Core Problem**: Can a QD objective be designed that operates directly over a continuous behavior space via differentiable optimization, without any discretization?

## Method

### Core Concept: Behavior Value
Each solution is treated as a light source illuminating the behavior space, with brightness proportional to quality and intensity decaying with distance:

$$v_{\bm{\theta}}(\mathbf{b}) = \max_{1 \leq n \leq N} f_n \exp\left(-\frac{\|\mathbf{b} - \mathbf{b}_n\|^2}{2\sigma^2}\right)$$

where $f_n = f(\theta_n)$ denotes the quality of solution $n$, and $\mathbf{b}_n = \text{desc}(\theta_n)$ is its behavior descriptor.

### Soft QD Score
The total behavior value integrated over the entire behavior space:

$$S(\bm{\theta}) = \int_{\mathcal{B}} v_{\bm{\theta}}(\mathbf{b}) \, d\mathbf{b}$$

Intuitively, achieving a high Soft QD Score requires high-quality solutions to be spread across the entire behavior space.

### Theoretical Properties (Theorem 1)
1. **Monotonicity**: Adding a new solution or improving an existing solution's quality does not decrease the score.
2. **Submodularity**: Marginal contributions are diminishing.
3. **Limit Equivalence**: As $\sigma \to 0$, the Soft QD Score converges to the traditional QD Score (up to a constant factor).

### SQUAD Algorithm
Directly maximizing $S(\bm{\theta})$ involves an intractable integral. Theorem 2 provides a computable lower bound:

$$\tilde{S}(\bm{\theta}) = \underbrace{\sum_{n=1}^N f_n}_{\text{quality term}} - \underbrace{\sum_{1 \leq i < j \leq N} \sqrt{f_i f_j} \exp\left(-\frac{\|\mathbf{b}_i - \mathbf{b}_j\|^2}{\gamma^2}\right)}_{\text{diversity term (repulsion)}}$$

**Intuition**:
- **Quality term**: Drives all solutions toward higher quality (attraction).
- **Diversity term**: Penalizes pairs of high-quality solutions with similar behaviors (repulsion).
- The geometric mean $\sqrt{f_i f_j}$ causes low-quality solutions to first improve quality before dispersing — quality before diversity.

### Efficient Implementation
- **K-nearest neighbor acceleration**: Repulsion is computed only over $k$ nearest neighbors per solution ($O(Nk)$ vs. $O(N^2)$).
- **Mini-batch updates**: Batched updates reduce memory consumption.
- **Bounded space handling**: A logit transformation maps bounded spaces to unbounded ones.

### Algorithm Overview
```
Initialize N solutions
for t = 1 to T:
    for each mini-batch:
        Find K nearest neighbors
        Compute S̃ and its gradient
        Update parameters with Adam
        Re-evaluate quality and behavior descriptors
```

## Key Experimental Results

### Main Results 1: Scalability to High-Dimensional Behavior Spaces (Linear Projection)

| Behavior Space Dim. | CMA-MAEGA | Sep-CMA-MAE | GA-ME | **SQUAD** |
|---|---|---|---|---|
| 2D | Best | Good | Medium | Competitive |
| 10D | Notable degradation | Degrades | Degrades | **Remains stable** |
| 50D | Severe degradation | Degrades | Degrades | **Still effective** |
| 100D | Nearly fails | Fails | Fails | **Still effective** |

> SQUAD is the only method that does not severely degrade in high-dimensional behavior spaces.

### Main Results 2: Image Synthesis (IC) and Latent Space Illumination (LSI)

| Method | IC QD Score | IC Vendi Score | LSI QD Score | LSI Vendi Score |
|---|---|---|---|---|
| CMA-MAEGA | Top-tier | High | Top-tier | High |
| Sep-CMA-MAE | High | Medium | High | Medium |
| DNS-G | Medium | High | Medium | High |
| **SQUAD** | Competitive | **Highest** | Competitive | **Highest** |

> SQUAD consistently achieves the best diversity scores while remaining competitive on QD Score.

### Ablation Study: Effect of $\gamma$

| $\gamma$ | Mean Quality | Vendi Score | Note |
|---|---|---|---|
| Small | Highest | Lowest | Weak repulsion → quality-biased |
| Medium | High | High | Balanced |
| Large | Lower | Highest | Strong repulsion → diversity-biased |

> $\gamma$ intuitively controls the quality–diversity trade-off.

### Key Findings
1. Cell-based methods fail in high-dimensional behavior spaces due to the curse of dimensionality.
2. SQUAD's continuous objective naturally avoids discretization issues.
3. The geometric mean in the repulsion term causes low-quality solutions to improve quality before dispersing, yielding a natural "quality-first, then diversity" curriculum.
4. The logit transformation is critical for bounded behavior spaces.
5. The K-nearest neighbor approximation has negligible impact on final results, as the exponential decay renders distant solutions' contributions negligible.

## Highlights & Insights
- **Paradigm shift**: From discrete cells to a continuous soft objective, circumventing the curse of dimensionality.
- **Elegant physical analogy**: Attraction (quality improvement) + repulsion (diversity spread) = self-organized equilibrium.
- **Theoretical completeness**: Monotonicity, submodularity, limit equivalence, and lower-bound derivation provide a solid foundation.
- **Subtle effect of the geometric mean**: Automatically implements a curriculum — low-quality solutions first improve quality; high-quality solutions begin to disperse.

## Limitations & Future Work
- The lower-bound approximation omits three-body and higher-order interactions, which may be inaccurate for very densely clustered populations.
- The hyperparameter $\gamma$ requires tuning and is sensitive to problem scale and behavior space structure.
- Unlike archive-based methods, SQUAD lacks explicit solution storage, making direct querying of specific behaviors less convenient.
- The method is not directly applicable to non-differentiable objective functions, such as RL problems requiring simulators.

## Related Work & Insights
- **QD Algorithms**: MAP-Elites (Cully et al., 2015), CMA-MEGA (Fontaine & Nikolaidis, 2021/2023)
- **Differentiable QD**: DQD (Fontaine & Nikolaidis, 2021), PGA-ME (Nilsson & Cully, 2021)
- **Novelty Search**: Lehman & Stanley (2011), DNS (Bahlous-Boldi et al., 2025)
- **Diversity Metrics**: Vendi Score (Friedman & Dieng, 2023)

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The Soft QD Score redefines the objective of QD optimization, representing a paradigm-level contribution.
- Theoretical Depth: ⭐⭐⭐⭐ — Monotonicity, submodularity, limit equivalence, and lower-bound derivation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three benchmark domains, multi-dimensional scalability analysis, and ablation study.
- Value: ⭐⭐⭐⭐ — A viable approach for high-dimensional QD, though limited to differentiable objectives.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Anomaly-Aware Pre-Training and Fine-Tuning for Graph Anomaly Detection](towards_anomaly-aware_pre-training_and_fine-tuning_for_graph_anomaly_detection.md)
- [\[AAAI 2026\] Towards a Rigorous Understanding of the Population Dynamics of the NSGA-III: Tight Runtime Bounds](../../AAAI2026/llm_evaluation/towards_a_rigorous_understanding_of_the_population_dynamics_of_the_nsga-iii_tigh.md)
- [\[AAAI 2026\] Structured Language Generation Model: Loss Calibration and Formatted Decoding for Efficient Text](../../AAAI2026/llm_evaluation/structured_language_generation_model_loss_calibration_and_formatted_decoding_for.md)
- [\[ICLR 2026\] Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces](discount_model_search_for_quality_diversity_optimization_in_high-dimensional_mea.md)
- [\[ICML 2026\] DEI: Diversity in Evolutionary Inference for Quality-Diversity Search](../../ICML2026/llm_evaluation/dei_diversity_in_evolutionary_inference_for_quality-diversity_search.md)

</div>

<!-- RELATED:END -->
