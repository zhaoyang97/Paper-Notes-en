---
title: >-
  [Paper Note] A Recovery Guarantee for Sparse Neural Networks
description: >-
  [ICLR 2026][Model Compression][Sparse neural networks] This paper establishes the first sparse recovery guarantee for ReLU neural networks: for two-layer scalar-output networks with Gaussian random training data…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Sparse neural networks"
  - "compressed sensing"
  - "iterative hard thresholding"
  - "convex reformulation"
  - "ReLU networks"
date: 2026-05-08
content_hash: 5bfac60e28f7aa9a
---

# A Recovery Guarantee for Sparse Neural Networks

**Conference**: ICLR 2026
**arXiv**: [2509.20323](https://arxiv.org/abs/2509.20323)  
**Code**: [https://github.com/voilalab/MLP-IHT](https://github.com/voilalab/MLP-IHT)  
**Area**: Model Compression / Theory
**Keywords**: Sparse neural networks, compressed sensing, iterative hard thresholding, convex reformulation, ReLU networks

## TL;DR
This paper establishes the first sparse recovery guarantee for ReLU neural networks: for two-layer scalar-output networks with Gaussian random training data, an iterative hard thresholding (IHT) algorithm based on convex reformulation can exactly recover sparse network weights, with memory requirements scaling only linearly in the number of nonzero weights.

## Background & Motivation

**Background**: Large-scale neural networks achieve strong performance but demand substantial memory and computation. Post-training weights are typically highly compressible. The existence of high-performing sparse subnetworks is well-established (Lottery Ticket Hypothesis), yet efficient optimization of sparse networks remains an open challenge.

**Limitations of Prior Work**: Existing methods are either memory-inefficient (e.g., IMP requires training a dense network first), yield suboptimal quality (e.g., initialization-based pruning), or lack theoretical guarantees (all existing methods are heuristic). Although the compressed sensing literature offers rich sparse recovery theory, it applies only to linear models.

**Key Challenge**: Sparse neural network optimization is a non-convex problem, and classical sparse recovery theory (requiring restricted isometry properties or strong convexity) does not directly extend to nonlinear ReLU networks.

**Goal**: To provide theoretical guarantees for sparse ReLU MLP weight recovery, including unique identifiability and convergence guarantees for an efficient recovery algorithm.

**Key Insight**: The paper exploits the convex reformulation theory for ReLU networks (Pilanci & Ergen, 2020) to transform non-convex sparse network optimization into a highly structured linear sensing problem, then applies sparse recovery guarantees for IHT.

**Core Idea**: By convexifying sparse MLP training via gated ReLU reformulation, the problem is cast as a linear sparse recovery problem. The paper proves that the sensing matrix satisfies restricted strong convexity and smoothness, enabling IHT to exactly recover sparse weights.

## Method

### Overall Architecture
A two-layer ReLU network $\hat{y} = \sum_{j=1}^p (Xu_j)_+ v_j$ is reformulated into the linear form $y = Aw^*$ by fixing activation patterns, where $A$ is a sensing matrix constructed from data and activation patterns, and $w^*$ is a fused sparse weight vector. IHT is then applied to recover $w^*$.

### Key Designs

1. **Convex Reformulation + Gated ReLU Parameterization**:

    - Function: Transforms non-convex MLP optimization into a convex/linear sparse recovery problem.
    - Mechanism: Fixed random generator vectors $h_i$ replace trainable $u_i$ to determine activation patterns $D_i = \text{Diag}(\mathbb{I}[Xh_i \geq 0])$; weights are fused as $w_i = u_i v_i$, yielding the form $\hat{y} = Aw$. Sparsity dramatically reduces the number of possible activation patterns (from exponential to polynomial).
    - Design Motivation: Convexification removes the non-convexity barrier, enabling the application of classical sparse recovery theory.

2. **Restricted Strong Convexity and Smoothness (Lemma 1)**:

    - Function: Proves that the sensing matrix $A$ satisfies the key conditions required for sparse recovery.
    - Mechanism: Under Assumption 2 (each neuron attends to at least an $\varepsilon$ fraction of the data, and any two distinct neurons differ in their activation patterns on at least a $\gamma$ fraction of samples), with high probability $\alpha I_s \preceq A_S^T A_S \preceq \beta I_s$, with the condition number $\sqrt{\beta/\alpha}$ bounded.
    - Design Motivation: The two conditions in Assumption 2 correspond respectively to "no overfitting to a small subset of samples" and "incoherence between neurons," analogous to the RIP in compressed sensing but less restrictive.

3. **IHT Recovery Guarantee (Theorem 1)**:

    - Function: Proves that IHT exactly recovers sparse MLP weights.
    - Mechanism: Building on Jain et al. (2014), IHT guarantees recovery under any finite condition number, at the cost of projecting to a support size $\tilde{s} > s$ larger than the true sparsity (with the inflation factor growing with the condition number). The convergence rate is $\|w^{k+1} - w^*\|_2 \leq \rho^k \|w^0 - w^*\|_2$.
    - Design Motivation: Standard RIP constant requirements are too stringent for MLPs; the result of Jain et al. permits any finite condition number, trading only convergence speed.

### Loss & Training
MSE loss $f(w) = \frac{1}{2}\|Aw - y\|_2^2$, with IHT update $w^{k+1} = H_{\tilde{s}}(w^k - \eta A^T(Aw^k - y))$. Memory efficiency is central: only $O(s)$ nonzero weights need to be stored, rather than the full dense network.

## Key Experimental Results

### Main Results

**Sparse Planted MLP Recovery:**

| Method | Recovery Error | Memory Efficiency | Theoretical Guarantee |
|--------|---------------|-------------------|-----------------------|
| IHT | ~0 (exact recovery) | Linear in $s$ | ✓ |
| IMP | Comparable / slightly worse | Requires dense network | ✗ |

### Ablation Study

| Experiment | IHT | IMP | Notes |
|------------|-----|-----|-------|
| MNIST Classification | Competitive / superior | Higher memory cost | IHT shows significant memory advantage |
| Implicit Neural Representation | Competitive | Baseline | Extended to 3-layer and vector-output settings |

### Key Findings
- IHT achieves exact recovery on the sparse planted MLP recovery task, validating theoretical predictions.
- IHT typically outperforms IMP in practice while using less memory—theoretical guarantees translate into practical advantages.
- Despite theory covering only 2-layer scalar-output networks, IHT performs well on 3-layer and vector-output networks.
- Sample complexity scales with the number of active (nonzero) weights rather than total weights, realizing genuine compressed sensing.

## Highlights & Insights
- **Bridge Between Compressed Sensing and Deep Learning**: Classical sparse recovery theory, developed over three decades, is rigorously applied to ReLU networks for the first time, demonstrating the vitality of classical theory in new domains.
- **The Enabling Role of Convexification**: Pilanci & Ergen's convex reformulation is not merely of theoretical interest—here it serves as the critical bridge connecting sparse recovery theory and MLPs.
- **Memory Efficiency with Theoretical Backing**: IHT requires only $O(s)$ memory, whereas methods such as IMP require $O(dp)$—for large-scale sparse networks, this represents orders-of-magnitude memory savings.

## Limitations & Future Work
- Theory applies only to two-layer scalar-output ReLU networks; extension to deeper networks and multi-output settings remains an open problem.
- Assumption 1 restricts weight values (binary hidden layer or $\pm 1$ output layer), which is more restrictive than practical settings.
- Enumerating all possible activation patterns is computationally infeasible for non-sparse networks.
- The Gaussian random data assumption diverges from practical data distributions.
- Experiments are conducted at small scale; practical utility on large-scale networks remains to be verified.

## Related Work & Insights
- **vs. Lottery Ticket Hypothesis**: LTH empirically demonstrates the existence of high-performing sparse subnetworks; this paper provides theoretical guarantees for exactly recovering those weights under specific conditions.
- **vs. Pilanci & Ergen Convexification**: Convexification provides a convex equivalent formulation of MLP training; this paper combines it with sparse recovery, representing an important application of convexification theory.
- **vs. Jain et al. (2014) IHT**: Jain et al. establish IHT guarantees for general linear sparse recovery; this paper proves that the sensing matrix of an MLP satisfies the required conditions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First sparse recovery guarantee for ReLU networks; theoretical contribution is clear and significant.
- Experimental Thoroughness: ⭐⭐⭐ Experiments are small-scale, limited to MNIST and simple tasks.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are rigorous and assumptions are clearly stated.
- Value: ⭐⭐⭐⭐ Establishes a theoretical foundation for sparse neural network training, though practical utility awaits further validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Adaptive Width Neural Networks](adaptive_width_neural_networks.md)
- [\[ICLR 2026\] Fine-tuning Quantized Neural Networks with Zeroth-order Optimization](fine-tuning_quantized_neural_networks_with_zeroth-order_optimization.md)
- [\[ICML 2026\] SURGE: Surrogate Gradient Adaptation in Binary Neural Networks](../../ICML2026/model_compression/surge_surrogate_gradient_adaptation_in_binary_neural_networks.md)
- [\[ICLR 2026\] Topology and Geometry of the Learning Space of ReLU Networks: Connectivity and Size](topology_and_geometry_of_the_learning_space_of_relu_networks_connectivity_and_si.md)
- [\[ICLR 2026\] PASER: Post-Training Data Selection for Efficient Pruned Large Language Model Recovery](paser_post-training_data_selection_for_efficient_pruned_large_language_model_rec.md)

</div>

<!-- RELATED:END -->
