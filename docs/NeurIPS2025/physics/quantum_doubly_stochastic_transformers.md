---
title: >-
  [Paper Note] Quantum Doubly Stochastic Transformers
description: >-
  [NeurIPS 2025][Physics][Variational quantum circuits] This paper proposes QDSFormer (Quantum Doubly Stochastic Transformer), replacing softmax with a variational quantum circuit QontOT to generate doubly stochastic atten…
tags:
  - "NeurIPS 2025"
  - "Physics"
  - "Variational quantum circuits"
  - "doubly stochastic matrices"
  - "attention mechanism"
  - "ViT"
  - "Birkhoff polytope"
date: 2026-05-08
content_hash: 9325a668bd5e851a
---

# Quantum Doubly Stochastic Transformers

**Conference**: NeurIPS 2025
**arXiv**: [2504.16275](https://arxiv.org/abs/2504.16275)
**Code**: None
**Area**: Quantum Computing / Transformer
**Keywords**: Variational quantum circuits, doubly stochastic matrices, attention mechanism, ViT, Birkhoff polytope

## TL;DR

This paper proposes QDSFormer (Quantum Doubly Stochastic Transformer), replacing softmax with a variational quantum circuit QontOT to generate doubly stochastic attention matrices. Both theoretical analysis and experiments demonstrate that quantum-circuit-generated DSMs are more diverse and better at preserving information, consistently outperforming standard ViT and Sinkformer on multiple small-scale visual recognition tasks.

## Background & Motivation

- In Transformers, softmax produces row-stochastic attention matrices (row sums = 1), leading to various training instabilities:
    - Entropy collapse: overly sharp attention causing vanishing gradients
    - Rank collapse and token uniformity
    - Eureka moments: sudden learning transitions in combinatorial problems
- Sinkformer found that attention naturally tends toward doubly stochastic matrices (row and column sums = 1), and enforcing double stochasticity improves performance
- Limitations of the Sinkhorn algorithm:
    - Iterative approximation that is difficult to converge to a true DSM in practice (Frobenius distance still 0.23 at $k=21$)
    - Non-parametric, incapable of learning which DSM to return
    - Requires non-negative inputs (via exponentiation), limiting expressivity
    - Backpropagation gradients can be ill-conditioned
- Key insight: The QontOT quantum circuit is proven to naturally produce DSMs ($\mathbf{U} \odot \bar{\mathbf{U}} \in \Omega_n$), and no known classical parametric method achieves the same property

## Method

### Overall Architecture

QDSFormer replaces softmax in ViT with the QontOT quantum circuit, feeding the attention matrix $\mathbf{QK}^\top$ into the circuit to obtain a doubly stochastic attention matrix. The circuit exploits the property that the Hadamard product of a unitary matrix with its conjugate naturally yields a DSM, enabling flexible DSM generation through parameterized quantum gates.

### Key Designs

1. **QontOT Quantum Circuit for DSM Generation**:
    - Function: Maps unnormalized attention matrices to doubly stochastic matrices
    - Mechanism: For any unitary matrix $\mathbf{U}$, $\mathbf{U} \odot \bar{\mathbf{U}} \in \Omega_n$. QontOT injects control circuit angles via the product of parameters $\theta$ and data $\mathbf{M}$, producing parameterized DSMs
    - Design Motivation: The quantum circuit inherently guarantees the DSM property and is parametric (unlike non-parametric Sinkhorn), enabling learning of optimal DSMs

2. **Expressivity Analysis**:
    - Function: Systematic comparison of DSM diversity among QontOT, Sinkhorn, and QR decomposition
    - Mechanism: Enumerate inputs over a discretized hypercube and count the number of unique DSMs produced
    - Key Result: QontOT (8 layers) produces a unique DSM for each input (approximately injective), while Sinkhorn and QR exhibit significant collision rates

3. **QR Decomposition Doubly Stochastic Operator (Quantum-Inspired)**:
    - Function: Serves as a classical quantum-inspired alternative
    - Mechanism: Apply QR decomposition to $\mathbf{M}$ to obtain a unitary matrix $\mathbf{U}$, then compute $\mathbf{U} \odot \bar{\mathbf{U}}$
    - Limitation: $O(n^3)$ complexity and high collision rate, though it performs well in certain single-layer ViT settings

### Loss & Training

- Three circuit training strategies:
    - **Static**: Uses fixed parameters obtained from quantum hardware experiments; no training required
    - **Mixed**: Alternates 200 steps of gradient-free optimization every epoch
    - **Differentiable**: End-to-end joint training (slowest; affected by Barren Plateaus)
- The static strategy performs best or on par with optimized variants (potentially due to Barren Plateaus)
- Uses 8×8 attention matrices, 16-layer circuits, and 4 ancilla qubits (16 qubits total)

## Key Experimental Results

### Main Results (Table)

Validation accuracy of a 2-layer ViT on FashionMNIST and MNIST:

| Method | FashionMNIST | MNIST |
|--------|-------------|-------|
| Softmax | 88.9 ± 0.1 | 98.1 ± 0.3 |
| Softmax_σ² | 84.6 ± 2.1 | 93.0 ± 4.6 |
| QR | 89.3 ± 0.1 | 98.3 ± 0.1 |
| Sinkhorn | 89.1 ± 0.7 | 98.2 ± 0.3 |
| **QontOT** | **90.0 ± 0.2** | **98.4 ± 0.1** |

MedMNIST (7 datasets): QontOT achieves best performance on 5 out of 7 datasets.

### Ablation Study

- **Circuit depth**: Performance begins to surpass ViT at 4–8 layers; additional layers yield logarithmic gains with diminishing returns beyond 16 layers
- **Static vs. optimized**: Static configurations match or outperform end-to-end optimization, likely due to Barren Plateaus
- **Eureka experiment**: QDSFormer exhibits earlier Eureka moments (sudden transitions in combinatorial reasoning) and more stable training
- **Sinkhorn iterations**: Frobenius distance to the Birkhoff polytope is 0.84 at $k=3$ and still 0.23 at $k=21$; QontOT achieves < 5e-6

### Key Findings

- QontOT produces the highest number of unique DSMs over 43M input matrices, behaving near-injectively (near-zero information loss)
- QontOT naturally exhibits higher attention entropy, mitigating entropy collapse during ViT training
- Sinkhorn maps row-constant matrices (e.g., $\mathbf{e}_2\mathbf{1}^\top$ and $\mathbf{e}_4\mathbf{1}^\top$) to the same DSM, resulting in information loss
- Circuit size scales logarithmically as $O(\log_2(T))$, theoretically favorable for long sequences

## Highlights & Insights

- This is the first work to apply the "DSM inductive bias" from quantum computing to Transformers, opening a design space unreachable by classical parametric methods
- The expressivity analysis is rigorous and comprehensive, covering three dimensions: exhaustive enumeration, information preservation, and entropy analysis
- The quantum-inspired QR decomposition method has independent value as a classical alternative
- The finding that static circuits already outperform classical methods simplifies deployment, eliminating the need for quantum-classical hybrid training

## Limitations & Future Work

- Experiments are limited to small-scale datasets (MNIST-level) and small models (1–4 layer ViT); large-scale validation is absent
- Quantum circuit simulation speed is a bottleneck; efficient execution of attention computation on real quantum hardware is currently infeasible
- Attention matrix sizes must be powers of two (qubit constraints), requiring padding
- Only encoder self-attention is validated; applicability to decoder and cross-attention remains unexplored

## Related Work & Insights

- Sinkformer and ESPFormer/LOTFormer are representative classical doubly stochastic Transformers
- The "$\mathbf{U} \odot \bar{\mathbf{U}}$ naturally yields a DSM" property of QontOT constitutes a unique quantum inductive bias with no known classical equivalent
- As quantum hardware matures, QDSFormer may demonstrate advantages at larger scales
- This work inspires a new paradigm for incorporating physical constraints into neural networks

## Rating

- ⭐⭐⭐⭐ — Theoretically novel with pioneering contributions at the quantum–ML intersection, but constrained by small-scale validation and the current maturity of quantum hardware

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Vision Transformers for Cosmological Fields: Application to Weak Lensing Mass Maps](vision_transformers_for_cosmological_fields_application_to_weak_lensing_mass_map.md)
- [\[NeurIPS 2025\] AstroCo: Self-Supervised Conformer-Style Transformers for Light-Curve Embeddings](astroco_self-supervised_conformer-style_transformers_for_light-curve_embeddings.md)
- [\[ICLR 2026\] Sublinear Time Quantum Algorithm for Attention Approximation](../../ICLR2026/physics/sublinear_time_quantum_algorithm_for_attention_approximation.md)
- [\[AAAI 2026\] Data Verification is the Future of Quantum Computing Copilots](../../AAAI2026/physics/data_verification_is_the_future_of_quantum_computing_copilots.md)
- [\[ICLR 2026\] Feedback-driven Recurrent Quantum Neural Network Universality](../../ICLR2026/physics/feedback-driven_recurrent_quantum_neural_network_universality.md)

</div>

<!-- RELATED:END -->
