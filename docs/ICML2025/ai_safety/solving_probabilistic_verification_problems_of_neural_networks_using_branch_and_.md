---
title: >-
  [Paper Note] Solving Probabilistic Verification Problems of Neural Networks Using Branch and Bound
description: >-
  [ICML 2025][AI Safety][neural network verification] This paper proposes a neural network probabilistic verification algorithm based on Branch and Bound. By iteratively refining the upper and lower bounds of the output probability, it answers "what is the probability that the network output satisfies a specific condition under a given input distribution," achieving a speedup of one to two orders of magnitude compared to existing methods.
tags:
  - "ICML 2025"
  - "AI Safety"
  - "neural network verification"
  - "probabilistic verification"
  - "branch and bound"
  - "bound propagation"
  - "fairness"
date: 2026-05-08
content_hash: eb5c397dc9722c7e
---

# Solving Probabilistic Verification Problems of Neural Networks Using Branch and Bound

**Conference**: ICML 2025  
**arXiv**: [2405.17556](https://arxiv.org/abs/2405.17556)  
**Code**: Yes  
**Area**: AI Safety  
**Keywords**: neural network verification, probabilistic verification, branch and bound, bound propagation, fairness

## TL;DR
This paper proposes a neural network probabilistic verification algorithm based on Branch and Bound. By iteratively refining the upper and lower bounds of the output probability, it answers "what is the probability that the network output satisfies a specific condition under a given input distribution," achieving a speedup of one to two orders of magnitude compared to existing methods.

## Background & Motivation

**Background**: Formal verification of neural networks is a crucial topic in AI safety. Traditional neural network verification focuses on deterministic problems (e.g., "whether the output satisfies a certain property for all inputs satisfying the conditions"), but many practical safety requirements are inherently probabilistic.

**Limitations of Prior Work**: Probabilistic verification problems include: (i) fairness verification (demographic parity: $P(\hat{y}=1|g=0) \approx P(\hat{y}=1|g=1)$), (ii) safety quantification ($P(\text{unsafe output}) \leq \delta$), and (iii) robustness quantification (the probability of correct classification under an input perturbation distribution). Existing probabilistic verification methods (such as PROVERO, VEGAS) are based on sampling or volume estimation, which suffer from long computation times (tens of minutes) and limited precision.

**Key Challenge**: Probabilistic verification requires integration over the input space, but the piecewise linear (ReLU) structure of neural networks makes exact integration intractable. Sampling methods require a large number of samples to obtain tight probability estimates, while volume calculation methods are highly inefficient in high-dimensional spaces.

**Goal**: Design an efficient and accurate probabilistic verification algorithm for neural networks.

**Key Insight**: Leverage bound propagation and branch-and-bound techniques, which have been highly successful in non-probabilistic neural network verification, and generalize them to the probabilistic setting.

**Core Idea**: Partition the input space into multiple sub-regions, calculate the upper and lower bounds of the output probability on each sub-region using bound propagation, and then employ a branch-and-bound strategy to selectively subdivide (branch) and prune (bound) these regions, iteratively narrowing the probability estimation.

## Method

### Overall Architecture
Input: Neural network $f$, input distribution $\mu$ (such as Gaussian), property $\phi$ (such as "output label is positive")  
Output: Upper and lower bounds of $P_{x \sim \mu}[\phi(f(x))]$ with controllable precision

Algorithm Flow:
1. Initialization: Treat the entire input space as a single region and compute the initial upper and lower bounds.
2. Selection: Select the region with the largest "uncertainty" from the current set of regions.
3. Branching: Bisect the selected region along a certain dimension.
4. Bounding: Calculate the probability upper and lower bounds for the new sub-regions.
5. Repeat steps 2-4 until the global upper and lower bounds are sufficiently tight.

### Key Designs

1. **Region-Level Probability Bounding**:

    - **Function**: Computes the upper and lower bounds of $P_{x \sim \mu, x \in R}[\phi(f(x))]$ for each sub-region $R$ of the input space.
    - **Mechanism**: Use bound propagation methods like CROWN/$\alpha$-CROWN to compute the output range of $f$ on $R$. If the output range completely satisfies/violates $\phi$, the probability contribution of this region can be precisely determined. Otherwise, the contribution is considered "uncertain."
    - **Key Formula**: $P[\phi(f(x))] \in [P_{\text{certain yes}}, P_{\text{certain yes}} + P_{\text{uncertain}}]$
    - **Design Motivation**: Bound propagation is the most efficient technique in non-probabilistic verification. This paper naturally extends it to the probabilistic setting.

2. **Adaptive Branching Strategy**:

    - **Function**: Selects the region and dimension to branch that are most beneficial for narrowing the global probability estimation.
    - **Mechanism**: Prioritizes branching the region with the largest "uncertain probability mass" and splits along the dimension with the "loosest bound."
    - **Design Motivation**: The larger the probability mass of the uncertain region, the greater the information gain brought by branching. Splitting along the dimension with the loosest bound is most likely to tighten the bounds of the sub-regions.

3. **Completeness Guarantee**:

    - **Function**: Proves that under appropriate heuristics, the algorithm can eventually reach arbitrary precision.
    - **Mechanism**: If the branching strategy guarantees that every uncertain region is eventually subdivided to be sufficiently small, the upper and lower bounds of the output probability will eventually converge.
    - **Design Motivation**: Being sound (correct bounds) is a basic requirement, while being complete (can reach arbitrary precision) is an additional guarantee.

### Loss & Training
Not applicable. The core algorithm is a pure verification/inference process.

## Key Experimental Results

### Main Results

| Benchmark | Metric (Solving Time, s) | Ours (B&B) | PROVERO | VEGAS | Speedup |
|----------|-------------------|---------|---------|-------|-------|
| Fairness (COMPAS) | Solving Time | **12s** | 420s | 312s | 26-35x |
| Safety (ACAS Xu) | Solving Time | **8s** | 195s | 287s | 24-36x |
| Robustness (MNIST) | Solving Time | **23s** | 1080s | 756s | 33-47x |
| Fairness (Adult) | Solving Time | **15s** | 890s | 445s | 30-59x |

### Ablation Study

| Configuration | COMPAS Solving Time | Precision (bound gap) | Description |
|------|---------------|----------------|------|
| Full B&B | **12s** | 0.001 | All components |
| Without adaptive branching (uniform branching) | 85s | 0.001 | Selection strategy is important |
| Without bound propagation (naive bounds) | >600s | 0.001 | BP is the core acceleration component |
| $\alpha$-CROWN bounds | **12s** | 0.001 | Best BP method |
| IBP bounds | 45s | 0.001 | Suboptimal BP |

### Key Findings
- Compared to existing probabilistic verification methods, it achieves a speedup of one to two orders of magnitude (from tens of minutes to tens of seconds).
- The proposed method even outperforms specially designed algorithms on certain restricted probabilistic verification problems.
- The quality of bound propagation has a huge impact on speed—$\alpha$-CROWN is 4 times faster than IBP.
- The adaptive branching strategy is 7 times faster than uniform branching, demonstrating the critical importance of heuristic choices.
- The algorithm is theoretically sound and, under appropriate conditions, complete.

## Highlights & Insights
- **Successful Technology Transfer**: Nuanced and elegant generalization of mature techniques from non-probabilistic verification (B&B + bound propagation) to the probabilistic setting.
- **High Versatility**: The same algorithmic framework can handle various probabilistic verification problems, including fairness, safety, and robustness.
- **Practical Efficiency**: A solving time of tens of seconds makes probabilistic verification feasible for real-world deployment.

## Limitations & Future Work
- Currently mainly supports ReLU networks; additional bound propagation support is needed for other activation functions (e.g., GELU, Swish).
- The number of branches may explode under high-dimensional inputs (such as large images).
- The input distribution is assumed to be Gaussian or truncated uniform; more complex distributions require new integration methods.
- Performance on large-scale networks (e.g., ResNet-50+) has not been tested.

## Related Work & Insights
- $\alpha$-CROWN (Wang et al., 2021): SOTA in non-probabilistic verification.
- PROVERO (Balunovic et al., 2019): Pioneering work in probabilistic verification.
- The framework in this paper provides a unified and efficient baseline for probabilistic verification.

## Rating
- Novelty: ⭐⭐⭐⭐ The technology transfer is natural and effective, though the core components (B&B, BP) are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on multiple types of verification problems, compared against several baselines, with detailed ablations and theoretical analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definitions and systematic algorithm descriptions.
- Value: ⭐⭐⭐⭐⭐ Provides a significant push for probabilistic verification in the field of AI safety.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] On Differential Privacy for Adaptively Solving Search Problems via Sketching](on_differential_privacy_for_adaptively_solving_search_problems.md)
- [\[ICLR 2026\] Certifying the Full YOLO Pipeline: A Probabilistic Verification Approach](../../ICLR2026/ai_safety/certifying_the_full_yolo_pipeline_a_probabilistic_verification_approach.md)
- [\[NeurIPS 2025\] Understanding and Improving Adversarial Robustness of Neural Probabilistic Circuits](../../NeurIPS2025/ai_safety/understanding_and_improving_adversarial_robustness_of_neural_probabilistic_circu.md)
- [\[ICML 2025\] Quadratic Upper Bound for Boosting Robustness](quadratic_upper_bound_for_boosting_robustness.md)
- [\[ICCV 2025\] Backdoor Attacks on Neural Networks via One-Bit Flip](../../ICCV2025/ai_safety/backdoor_attacks_on_neural_networks_via_one_bit_flip.md)

</div>

<!-- RELATED:END -->
