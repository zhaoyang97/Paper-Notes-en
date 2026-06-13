---
title: >-
  [Paper Note] Dense Associative Memory with Epanechnikov Energy
description: >-
  [NeurIPS 2025][associative memory] This paper proposes a log-sum-ReLU (LSR) energy function based on the Epanechnikov kernel as a replacement for the conventional log-sum-exp (LSE) energy in Dense Associative Memory. For…
tags:
  - "NeurIPS 2025"
  - "associative memory"
  - "Hopfield network"
  - "energy function"
  - "Epanechnikov kernel"
  - "emergent memory"
  - "ReLU"
date: 2026-05-08
content_hash: 41b3c015f3ab471c
---

# Dense Associative Memory with Epanechnikov Energy

**Conference**: NeurIPS 2025
**arXiv**: [2506.10801](https://arxiv.org/abs/2506.10801)  
**Code**: None  
**Area**: Other
**Keywords**: associative memory, Hopfield network, energy function, Epanechnikov kernel, emergent memory, ReLU

## TL;DR
This paper proposes a log-sum-ReLU (LSR) energy function based on the Epanechnikov kernel as a replacement for the conventional log-sum-exp (LSE) energy in Dense Associative Memory. For the first time, it achieves the coexistence of exact retrieval of all stored patterns and the emergence of novel creative local minima, while preserving exponential memory capacity.

## Background & Motivation
**Background**: Dense Associative Memory (DenseAM / Modern Hopfield Network) stores patterns via energy functions. The LSE energy (corresponding to a Gaussian kernel) is the dominant choice, achieving exponential memory capacity $M^* \sim \exp(d)$.

**Limitations of Prior Work**: The LSE energy exhibits a fundamental tension between memorization and generalization—when all original patterns are retrieved exactly ($\beta \to \infty$), no new local minima emerge; conversely, when new patterns emerge (finite $\beta$), retrieval of original patterns is no longer exact. That is, LSE cannot simultaneously achieve exact memorization and creative emergence.

**Key Challenge**: The conventional view holds that perfect memorization (zero training loss / exact retrieval) is incompatible with generalization. Drawing an analogy to the "double descent" phenomenon in deep learning, the question arises: can one find an energy function for associative memory that simultaneously enables exact storage and the generation of meaningful new patterns?

**Goal**: To identify an energy function that endows DenseAM with both exact memorization and emergent creativity.

**Key Insight**: The duality between energy functions and probability density functions—$\exp[-E(\mathbf{x})]$ constitutes a kernel density estimator. In KDE theory, the Epanechnikov kernel achieves superior estimation efficiency compared to the Gaussian kernel, corresponding to $F(x) = \text{ReLU}(1+x)$.

**Core Idea**: Replace the Gaussian kernel (exp) with the optimal KDE kernel (Epanechnikov/ReLU) as the separation function, yielding the LSR energy that supports both exact memorization and emergence.

## Method

### Overall Architecture
Within the general DenseAM energy framework $E_\beta(\mathbf{x}) = -Q[\sum_\mu F(\beta S(g(\mathbf{x}), \boldsymbol{\xi}_\mu))]$, the LSR energy is defined using $F(x) = \text{ReLU}(1+x)$ and $Q(x) = \log x$. The paper analyzes its properties with respect to memory retrieval, capacity, and emergence.

### Key Designs

1. **LSR Energy Function**:

    - Function: Defines a new energy function $E_\beta^{\text{LSR}}(\mathbf{x}) = -\frac{1}{\beta}\log(\epsilon + \sum_\mu \text{ReLU}(1 - \frac{\beta}{2}\|\mathbf{x} - \boldsymbol{\xi}_\mu\|^2))$
    - Mechanism: The compact support of ReLU ensures that each memory influences only the region within radius $\sqrt{2/\beta}$. When a query point $\mathbf{x}$ falls within the support of exactly one memory, the gradient points precisely toward that memory (exact retrieval); when it falls within the overlapping supports of multiple memories, the gradient points toward their centroid (emergent memory).
    - Design Motivation: The Gaussian kernel underlying LSE has infinite support, so every point is influenced by all memories, requiring $\beta \to \infty$ for exact retrieval—but at this limit, emergent patterns vanish. The compact support of ReLU allows exact retrieval and emergence to coexist at the same finite value of $\beta$.

2. **Exact Retrieval and Exponential Capacity (Theorems 1 & 2)**:

    - Function: Proves that under appropriate $\beta$, all stored memories are retrieved exactly and a single gradient descent step suffices.
    - Mechanism: Let $r$ denote the minimum inter-memory distance. Setting $\beta = 2/(r-\Delta)^2$ ensures that any query within radius $\Delta$ of a memory converges exactly to that memory. Capacity scales as $M^* = \Theta(\exp(\alpha d))$.
    - Design Motivation: Unlike LSE's "approximate" retrieval (where the gradient is only approximately zero), the LSR gradient is exactly zero at finite $\beta$—a direct consequence of ReLU's compact support.

3. **Global Emergence**:

    - Function: Defines and proves that the LSR energy exhibits "global emergence"—all original memories are local minima, while additional novel local minima (emergent memories) simultaneously exist.
    - Mechanism: Emergent memories take the form $\mathbf{x}^* = \frac{1}{|B(\mathbf{x}^*)|} \sum_{\mu \in B(\mathbf{x}^*)} \boldsymbol{\xi}_\mu$, i.e., points lying in the intersection of multiple support balls converge to the centroid of the corresponding memories. The number of emergent memories can reach $O(\exp(MVd/V \cdot (2/\beta)^{d/2} \cdot \log(...)))$.
    - Design Motivation: LSE is proven not to satisfy global emergence (Proposition 1)—under all three cases analyzed, it cannot simultaneously guarantee retrieval of all original memories and the existence of new local minima.

### Summary of Theoretical Contributions
- **Theorem 1**: Under LSR energy, each memory has an exact basin of attraction, and single-step gradient descent achieves retrieval.
- **Theorem 2**: LSR simultaneously achieves exponential memory capacity and exponential numbers of emergent memories.
- **Proposition 1**: LSE energy does not satisfy global emergence (three-case analysis).
- **Proposition 3**: Under a grid construction, the exact order of emergent memories is $\Theta((M^{1/d} - \lambda^{1/d} + 1)^d)$.

## Key Experimental Results

### Analysis of Emergent Memory Count

| Dimension | Stored Patterns $M$ | Max Emergent Memories | Growth Ratio |
|-----------|--------------------|-----------------------|--------------|
| $d=2$ | 5–20 | Hundreds | ~10–50× |
| $d=4$ | 5–20 | Thousands | ~100–1000× |
| $d=8$ | 5–20 | Tens of thousands | ~1000×+ |

### Generation Quality Comparison (Gaussian Mixture Density Estimation)

| Metric | LSR | LSE |
|--------|-----|-----|
| Mean log-likelihood | **Comparable or slightly better** | Baseline |
| Number of unique samples | **Significantly higher (orders of magnitude)** | Low (converges to few memories) |
| Original memory retention rate | High | High |

### Image Generation Experiments

| Dataset | Result |
|---------|--------|
| MNIST (10D VAE) | Emergent memories are meaningful blends of original digits (e.g., fusion of 4 and 9) |
| TinyImageNet (256D VAE) | Global emergence is achievable; emergent images are visually plausible but relatively blurry |

### Key Findings
- At the critical $\beta$ value, LSR can generate orders of magnitude more novel memories than the number of stored patterns.
- LSR emergent memories achieve log-likelihood comparable to LSE on known ground-truth distributions, while exhibiting far greater diversity (500 queries under LSE converge to approximately 10 distinct memories).
- The mechanism underlying emergent memories is remarkably simple—they are centroids of subsets of neighboring memories—yet when decoded in a semantic latent space, they manifest as seemingly "creative" novel patterns.

## Highlights & Insights
- **KDE Theory Guiding AM Design**: Selecting the separation function from the perspective of optimal kernel density estimation elegantly bridges the fields of statistics and associative memory.
- **Unexpected Benefits of ReLU's Compact Support**: Beyond reducing variance in KDE, the compact support of ReLU yields a unique combination of exact retrieval (gradient is exactly zero) and emergence in associative memory.
- **Memorization and Generalization Are Not Contradictory**: Analogous to the "overfitting yet generalizing" phenomenon in deep learning, LSR demonstrates that perfect memorization and creative generation can harmoniously coexist in associative memory.
- **Analogy Between Emergence and Hallucination**: The authors draw a philosophical parallel between emergent memories and LLM hallucinations—both involve "creating" content not present in the training data.

## Limitations & Future Work
- **Limited Quality of Emergent Memories**: As centroids of neighboring memories, emergent memories tend to be blurry in pixel space, making them suboptimal for high-fidelity generation tasks.
- **Sensitivity to $\beta$ Selection**: Global emergence occurs only within a specific range of $\beta$, requiring careful tuning in practical applications.
- **Connection to Transformer Attention Remains Underexplored**: The gradient of LSR corresponds to a sparse weighted average rather than softmax, but this has not been validated in practical attention implementations.

## Related Work & Insights
- **vs. LSE / Modern Hopfield Network**: The infinite support of LSE precludes emergence under exact retrieval; the compact support of LSR enables their coexistence.
- **vs. Sparse Softmax Methods (Hu et al. 2023)**: Those approaches sparsify from the perspective of computational efficiency; this work arrives at a natural sparsification (the zero-valued region of ReLU) from the perspective of statistical optimality.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to achieve coexistence of exact memorization and emergence in associative memory; the introduction of the ReLU energy function is highly elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage of theory, numerical experiments, and image experiments, though large-scale and high-dimensional validation is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ The motivational chain is clear (KDE → energy → AM); definitions and theorems are rigorously organized.
- Value: ⭐⭐⭐⭐⭐ Significant implications for both associative memory theory and the understanding of Transformer architectures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Fixed-Point RNNs: Interpolating from Diagonal to Dense](fixed-point_rnns_interpolating_from_diagonal_to_dense.md)
- [\[NeurIPS 2025\] Learning Dense Hand Contact Estimation from Imbalanced Data](learning_dense_hand_contact_estimation_from_imbalanced_data.md)
- [\[NeurIPS 2025\] Recurrent Self-Attention Dynamics: An Energy-Agnostic Perspective from Jacobians](recurrent_self-attention_dynamics_an_energy-agnostic_perspective_from_jacobians.md)
- [\[AAAI 2026\] PIPHEN: Physical Interaction Prediction with Hamiltonian Energy Networks](../../AAAI2026/others/piphen_physical_interaction_prediction_with_hamiltonian_energy_networks.md)
- [\[AAAI 2026\] Spike Imaging Velocimetry: Dense Motion Estimation of Fluids Using Spike Cameras](../../AAAI2026/others/spike_imaging_velocimetry_dense_motion_estimation_of_fluids_using_spike_cameras.md)

</div>

<!-- RELATED:END -->
