---
title: >-
  [Paper Note] TITAN: A Trajectory-Informed Technique for Adaptive Parameter Freezing in Large-Scale VQE
description: >-
  [NeurIPS 2025][Physics][VQE] This paper proposes TITAN, a framework that employs deep learning models to predict "frozen parameters" in VQE—parameters that remain inactive throughout training—enabling 40–60% of parameter…
tags:
  - "NeurIPS 2025"
  - "Physics"
  - "VQE"
  - "parameter freezing"
  - "barren plateau"
  - "quantum chemistry"
  - "deep learning-assisted quantum computing"
date: 2026-05-08
content_hash: a0acf2751853b794
---

# TITAN: A Trajectory-Informed Technique for Adaptive Parameter Freezing in Large-Scale VQE

**Conference**: NeurIPS 2025
**arXiv**: [2509.15193](https://arxiv.org/abs/2509.15193)
**Code**: GitHub (open-source)
**Area**: Physics
**Keywords**: VQE, parameter freezing, barren plateau, quantum chemistry, deep learning-assisted quantum computing

## TL;DR
This paper proposes TITAN, a framework that employs deep learning models to predict "frozen parameters" in VQE—parameters that remain inactive throughout training—enabling 40–60% of parameters to be frozen at initialization, achieving up to 3× convergence speedup and 40–60% reduction in circuit evaluations, while matching or surpassing baseline accuracy on molecular systems of up to 30 qubits.

## Background & Motivation

**Background**: The Variational Quantum Eigensolver (VQE) is the flagship protocol for electronic structure calculations on quantum computers, iteratively optimizing parameterized quantum circuits to minimize the expectation value of a molecular Hamiltonian.

**Limitations of Prior Work**: (a) The no-cloning theorem requires two circuit evaluations per parameter per gradient step (parameter-shift rule), so measurement overhead scales linearly with parameter count; (b) deeper circuits suffer from barren plateaus (BP), where gradients decay exponentially; (c) a single VQE run for benzene (C₆H₆) requires $10^6$–$10^8$ circuit evaluations (~55 hours on trapped-ion quantum computers).

**Key Challenge**: Sufficiently deep circuits are necessary for expressibility, yet increasing depth causes an explosion in parameter count and exacerbates BP.

**Goal**: Reduce VQE measurement overhead from a parameter-space compression perspective by identifying and freezing parameters that remain inactive throughout training.

**Key Insight**: Empirical observation that VQE exhibits a "frozen parameter" phenomenon—initialization determines the long-term activity of parameters, and a subset of parameters remains consistently unimportant throughout the entire training process.

**Core Idea**: A deep learning model (ResNet-18 + self-attention) predicts which parameters can be frozen from the Hamiltonian and ansatz structure, pruning the parameter space before optimization begins.

## Method

### Overall Architecture
Three stages: (1) the APFA algorithm collects parameter freezing patterns from VQE training trajectories as training labels; (2) CFCSA encoding maps arbitrary $(H, A(L,N,D))$ pairs to a unified input tensor, training a ResNet-18+MHSA model to predict freezing intensity; (3) at inference time, frozen parameters are predicted for a new VQE instance, the parameter space is pruned directly, and optimization proceeds on the reduced space.

### Key Designs

1. **APFA (Adaptive Parameter Freezing and Activation) Algorithm**:

    - Function: Records the time-series freeze/activation state of each parameter during VQE training.
    - Mechanism: Maintains a gradient EMA $\hat{g}_t^{(i)}$ for each parameter; adaptive freezing/activation thresholds $\tau_f^{(t)}, \tau_a^{(t)}$ are dynamically adjusted by the gradient decay ratio $r_t$; a parameter is frozen after $N_f$ consecutive steps below the freezing threshold and reactivated after $N_a$ consecutive steps above the activation threshold.
    - Design Motivation: First quantification of the parameter freezing phenomenon in VQE, providing time-series mask trajectories as deep learning training labels.

2. **Enhanced Gaussian Initialization (Theorem 1)**:

    - Function: Ensures that VQE trajectories in the dataset avoid barren plateaus.
    - Mechanism: Proves that when using an HEA ansatz with parameters sampled from $\mathcal{N}(0, c/L)$, the gradient variance lower bound is $\Theta(1/L)$ (only polynomially rather than exponentially decaying with depth).
    - Design Motivation: Under BP, APFA cannot extract meaningful freezing patterns; this theorem guarantees the information quality of training data.

3. **CFCSA Encoding (Coordinate-aware FC Self-Attention)**:

    - Function: Encodes arbitrary-scale VQE instances into a unified input format.
    - Mechanism: Each parameter is associated with a 3D coordinate $[\ell/(L{-}1),\, d/(D{-}1),\, q/(N{-}1)]$ (layer/gate/qubit) plus a $K$-dimensional descriptor (coupling constants, etc.), forming a tensor $\mathbf{X} \in \mathbb{R}^{(3+K) \times L \times (DN)}$; a fully convolutional + MHSA architecture is insensitive to input dimensionality.
    - Design Motivation: A single trained model can handle arbitrary $(L, N, D)$ combinations without retraining.

### Loss & Training
- Frobenius norm minimization: $\min_\omega \frac{1}{N}\sum_i \|\hat{\mathbf{C}}_i - \mathbf{C}_i\|_F^2$
- ResNet-18 + channel-wise MHSA, Adam optimizer.

## Key Experimental Results

### Main Results — Heisenberg Model (HEA, 5–15 qubits, 5–10 layers)

| Metric | Result |
|--------|--------|
| Proportion with $\Delta E \leq 0$ | >90% of (layer, qubit) configurations |
| Frozen parameter ratio ($\tau$=80) | ~40–60% |
| Convergence speedup | Up to 3× |
| Circuit evaluation reduction | 40–60% |

### Ablation Study — Generalization Across Ansätze

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| TITAN on HEA | $\Delta E \leq 0$ | Matches or exceeds baseline in most configurations |
| TITAN on SU2 | $\Delta E \leq 0$ | Generalizes to unseen ansatz |
| TITAN on SEL | $\Delta E \leq 0$ | Generalizes to unseen ansatz |
| Random Freeze | Significant energy loss (0.3–1.2 Ha) | Random freezing is far inferior to TITAN |

### Key Findings
- **Universal frozen parameter phenomenon**: Observed across 5 to 15 qubits, different ansätze, and different Hamiltonians; initialization determines long-term redundancy patterns.
- **Structured freezing patterns**: In small circuits (5×5), nearly all parameters in the first two layers are frozen; in large circuits (5×10), active regions concentrate near the output end.
- **Random freezing is entirely ineffective**: Confirms that TITAN's gains stem from targeted pruning rather than simply reducing parameter count.
- **Cross-ansatz generalization**: TITAN trained on HEA successfully generalizes to SU2 and SEL ansätze.

## Highlights & Insights
- **A new paradigm for deep learning-assisted quantum computing**: Classical neural networks predict redundancy patterns in quantum circuits, recasting VQE parameter-space compression as a supervised learning problem.
- **General-purpose CFCSA encoding design**: Coordinate + descriptor encoding enables a single model to handle VQE instances of arbitrary scale, offering significant engineering value.
- **Strong complementarity**: TITAN can be combined additively with existing techniques such as measurement grouping, architecture search, and optimizer improvements.

## Limitations & Future Work
- The theoretical guarantee (Theorem 1) applies only to HEA-type ansätze and has not been proven for chemically inspired ansätze such as UCC.
- Generating training data itself requires running a large number of VQE trajectories, representing a non-negligible upfront investment.
- 30 qubits remains far below practical chemical applications (100+ qubits required).
- The effect of real quantum hardware noise on freezing patterns has not been validated.

## Related Work & Insights
- **vs. QuACK**: QuACK optimizes VQE at the optimizer level, while TITAN operates at the parameter-space level; the two approaches are complementary.
- **vs. Quantum Architecture Search (QAS)**: QAS searches for optimal circuit structures, while TITAN prunes parameters within a given structure—complementary directions.
- **vs. classical pruning**: Conceptually analogous to neural network pruning but applied to quantum circuits, representing an interesting quantum–classical cross-domain analogy.

## Rating
- Novelty: ⭐⭐⭐⭐ — The discovery of the frozen parameter phenomenon and the DL prediction framework are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple Hamiltonians, multiple ansätze, ablations, and molecular systems provide broad coverage.
- Writing Quality: ⭐⭐⭐⭐ — Theory and experiments are clearly organized.
- Value: ⭐⭐⭐⭐ — Provides a practical parameter compression tool for scaling VQE.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Simulation-Based Inference for Neutrino Interaction Model Parameter Tuning](simulation-based_inference_for_neutrino_interaction_model_parameter_tuning.md)
- [\[NeurIPS 2025\] FEAT: Free Energy Estimators with Adaptive Transport](feat_free_energy_estimators_with_adaptive_transport.md)
- [\[AAAI 2026\] Adaptive Fidelity Estimation for Quantum Programs with Graph-Guided Noise Awareness](../../AAAI2026/physics/adaptive_fidelity_estimation_for_quantum_programs_with_graph.md)
- [\[NeurIPS 2025\] Knowledge is Overrated: A Zero-Knowledge ML and Cryptographic Hashing-Based Framework for Verifiable, Low Latency Inference at the LHC](knowledge_is_overrated_a_zero-knowledge_machine_learning_and_cryptographic_hashi.md)
- [\[NeurIPS 2025\] Vision Transformers for Cosmological Fields: Application to Weak Lensing Mass Maps](vision_transformers_for_cosmological_fields_application_to_weak_lensing_mass_map.md)

</div>

<!-- RELATED:END -->
