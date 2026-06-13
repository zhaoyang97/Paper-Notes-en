---
title: >-
  [Paper Note] Quiver: Quantum-Informed Views for Enhanced Representations in Large ML Models
description: >-
  [ICML 2026][Physics & Scientific Computing][Variational Quantum Circuits] Quiver feeds categorical inputs into an additional Variational Quantum Circuit (VQC) to extract the Quantum Fisher Information Matrix (QFIM) as a…
tags:
  - "ICML 2026"
  - "Physics & Scientific Computing"
  - "Variational Quantum Circuits"
  - "Quantum Fisher Information Matrix"
  - "Multi-modal Representations"
  - "Particle Transformer"
  - "DimeNet++"
date: 2026-05-08
content_hash: a367a1a4f5b0267c
---

# Quiver: Quantum-Informed Views for Enhanced Representations in Large ML Models

**Conference**: ICML 2026  
**arXiv**: [2606.02785](https://arxiv.org/abs/2606.02785)  
**Code**: None (Repository not released)  
**Area**: Physics / Hybrid Quantum-Classical Learning / HEP + Molecular Chemistry  
**Keywords**: Variational Quantum Circuits, Quantum Fisher Information Matrix, Multi-modal Representations, Particle Transformer, DimeNet++  

## TL;DR
Quiver feeds categorical inputs into an additional Variational Quantum Circuit (VQC) to extract the Quantum Fisher Information Matrix (QFIM) as a "Quantum Geometric View." This view is then injected into classical backbones via cross-attention (for Transformers) or residual gating (for GNNs), achieving consistent improvements across two distinct physical tasks: JetClass top quark tagging and QM9 HOMO-LUMO gap regression.

## Background & Motivation

**Background**: Jet tagging in high-energy physics and property prediction in molecular chemistry (QM9) both represent high-dimensional structured data problems. Mainstream approaches such as Particle Transformer (~2.14M parameters) and geometric/equivariant GNNs like DimeNet++ have approached SOTA on their respective benchmarks.

**Limitations of Prior Work**: These models are trained entirely in classical feature spaces. For samples requiring higher-order or non-local correlations—such as color-singlet $W$ jets vs. color-connected QCD jets, or electronic structures in QM9 that depend on many-body correlations—models can only learn these associations implicitly through increased capacity rather than having them explicitly exposed.

**Key Challenge**: Classical feature engineering (kinematics, structural descriptors) is inherently poor at expressing many-body coherence correlations. Simply increasing model capacity or data volume does not efficiently bridge this structural blind spot. A fundamentally different geometric perspective is needed that is complementary to, rather than redundant with, classical features.

**Goal**: To decompose the problem into two sub-tasks: (1) How to use quantum circuits to extract "geometric correlation structures" from classical inputs into a compact, system-agnostic tensor; (2) How to integrate this tensor into existing SOTA classical backbones with minimal parameter cost and physical alignment.

**Key Insight**: A Variational Quantum Circuit $|\psi(\boldsymbol{\Theta})\rangle=U(\boldsymbol{\Theta})|0\rangle^{\otimes N}$ encodes inputs into Hilbert space, where the parameter manifold naturally carries the Fubini-Study metric, which is equivalent (up to a factor of 4) to the Quantum Fisher Information Matrix (QFIM). The diagonal terms of the QFIM represent "single-parameter sensitivity," while off-diagonal terms represent "coherence coupling"—a geometric encoding of many-body correlations that can be computed using classical simulators (e.g., PennyLane).

**Core Idea**: Use the "Quantum Fisher View" as a second modality complementary to the classical view. Once fused, the classical backbone can directly consume quantum geometric information rather than learning it implicitly from scratch.

## Method

### Overall Architecture
Quiver = Classical Input → Task-specific VQC → QFIM Measurement → Modality Fusion Layer → Classical SOTA Backbone. Each application uses a dedicated quantum encoding: 1P1Q (one qubit per particle) for jets, and a new 2A2Q (two-qubit blocks per bonded atom pair) for molecules. Fusion mechanisms are differentiated by backbone: Transformers use cross-attention via sequence concatenation, while GNNs use residual gating on edge states modulated by the QFIM. The entire VQC is simulated classically on PennyLane, with QFIMs pre-computed and cached.

### Key Designs

1.  **Quantum Fisher View: Extracting Many-body Coherence with VQC**:
    - **Function**: Maps classical input $x$ to a parameterized quantum state $|\psi(\boldsymbol{\Theta}(x),\boldsymbol{\theta})\rangle$, then calculates the QFIM $F_{ij}(\boldsymbol{\theta};x)=4\,\text{Re}[\langle \partial_i\psi|\partial_j\psi\rangle-\langle\partial_i\psi|\psi\rangle\langle\psi|\partial_j\psi\rangle]$ at a fixed reference point $\boldsymbol{\theta}_0$ to obtain an input-dependent relation tensor.
    - **Mechanism**: Diagonal elements $F_{ii}$ represent the circuit's local sensitivity to $\theta_i$, serving as "per-qubit dynamic importance"; off-diagonal elements $F_{ij}$ are non-zero only when two directions act on overlapping qubit subsystems, thus directly encoding "coherently coupled input dimensions." For 1P1Q encoding (10 particles × 3 rotation parameters per qubit), the QFIM is a 30×30 real symmetric matrix stored as 90 channels × 10 particles. For 2A2Q, a 10-qubit × 2-layer × 3-rotation setup yields a 60×60 matrix organized into 10×10 blocks of 6×6, where sub-block $Q_{ij}$ corresponds to the coupling of atom pair $(i,j)$.
    - **Design Motivation**: The QFIM is an intrinsic geometry of the parameter manifold, independent of the measurement basis. Off-diagonal elements naturally flag "joint behaviors," corresponding to physically meaningful high-order correlations that are difficult for classical features to capture.

2.  **2A2Q Molecular Encoding: Rotation-Invariant Pairwise Quantum Encoding**:
    - **Function**: A novel molecular quantum encoding for the QM9 task that avoids coordinate dependency and integrates chemical bond information into entanglement operations.
    - **Mechanism**: Each heavy atom is assigned one qubit. First, single-atom embeddings $R_Y(w_{\text{atom}}^j)|0\rangle$ are applied. Then, for each pair of bonded atoms where $d_{ij}<d_{\text{CUTOFF}}=1.7\,\text{Å}$, three angles $\omega_1^{(ij)}=e_{d_1}(1-d_{ij}/d_{\text{CUTOFF}})\cos\theta_{ij}$, $\omega_2^{(ij)}=e_{\text{bond}}^{(ij)}\pi$, and $\omega_3^{(ij)}=e_{d_2}(1-d_{ij}/d_{\text{CUTOFF}})\cos\phi_{ij}$ are used for joint encoding and entanglement: $\mathcal{U}_{ij}=(I_{YY}(\omega_3)I_{ZZ}(\omega_2)I_{XX}(\omega_1))(R_Y\otimes R_Y)|00\rangle$. Finally, $R_Z R_Y R_Z$ rotations are applied to each qubit. Predictions are derived from the expectation value of Hamiltonian $\mathcal{H}=\sum_i c_i Z_i$, trained with Huber loss.
    - **Design Motivation**: Directly encoding Cartesian coordinates onto qubits introduces reference frame dependency. By merging "encoding + entanglement" into pairwise operations, the distance $d_{ij}$ is naturally invariant. $e_{\text{bond}}$ allows the entanglement strength to learn chemical bond types, making the QFIM sub-blocks reflect bonding correlations.

3.  **Differentiated Architectural Injection: Cross-attention and Gated Residuals**:
    - **Function**: Fuses the QFIM modality with classical modalities across different backbones without significant parameter increases.
    - **Mechanism**: For Particle Transformer, the 90 QFIM channels for each particle $i$ are embedded into a 128-d token $q_i=\text{MLP}_{\text{QFIM}}(\mathbf{Q}[:,i])$ and appended to the classical sequence, forming an input of length $2P$. For DimeNet++, a residual gate $\tilde{x}_{ij}^{(l)}=(1+\alpha\cdot\Theta(Q_{ij}))x_{ij}^{(l)}$ modulates edge states, where $\alpha$ is a globally learnable scalar initialized to zero, and $\Theta(Q_{ij})\in[-1,1]$ is processed from the 6×6 QFIM sub-block via a small CNN and $\tanh$.
    - **Design Motivation**: Transformers possess a natural cross-attention mechanism, making sequence concatenation intuitive. GNNs lack this; therefore, "zero-initialized residual gates" ensure that when $\alpha=0$, Quiver is equivalent to the baseline, guaranteeing that improvements stem strictly from QFIM information.

### Loss & Training
Standard cross-entropy is used for JetClass binary classification. Huber loss is used for QM9 regression (robust to outliers, combining $\ell_2$ and $\ell_1$). VQCs are simulated in PennyLane, and QFIMs are computed using standard implementations. Both tasks are evaluated over multiple seeds (5 for JetClass, 10 for QM9).

## Key Experimental Results

### Main Results 1: JetClass Top Quark vs. QCD Classification

| Feature Set | Model | Params | AUC ↑ | 1/ε_B @ ε_S=0.5 ↑ |
|------|------|------|------|------|
| Kin | ParT | 5M | 0.97832 ± 0.00004 | 176 ± 1 |
| Kin | **Quiver** | 5M | **0.98070 ± 0.00003** | **240 ± 1** |
| Full | ParT | 5M | 0.99235 ± 0.00003 | 1306 ± 8 |
| Full | **Quiver** | 5M | **0.99244 ± 0.00003** | **1362 ± 28** |
| Full | ParT | 0.1M | 0.98875 ± 0.00008 | 570 ± 13 |
| Full | **Quiver** | 0.1M | **0.98893 ± 0.00005** | **590 ± 7** |

Using only kinematic features, Quiver with 5M parameters increases the QCD rejection rate from 176 to 240 (+36%). With full features, it increases from 1306 to 1362 (+4%), with a parameter cost of only +7% (2.14M → 2.29M).

### Main Results 2: QM9 HOMO-LUMO Gap Regression

| Model | Params | Test MAE (meV) ↓ | Paired Δ MAE (meV) | Rel. Decrease |
|------|------|------|------|------|
| DimeNet++ | 1.886M | 72.42 ± 1.52 | — | — |
| **𝒬DimeNet++ (Quiver)** | 1.891M | **67.92 ± 1.98** | **4.50 ± 2.46** | **6.21%** |

With a parameter increase of only 0.27%, the 10-seed paired $t$-test yielded $t_9=5.78, p<10^{-3}$, indicating statistical significance.

### Key Findings
- Improvements across both tasks are "persistent": JetClass training curves show that the $\Delta$ MAE between 𝒬DimeNet++ and DimeNet++ remains positive across all epochs.
- Gains do not vanish as the classical model scales: Quiver performs better across 0.1M, 0.5M, and 5M parameter ranges, suggesting QFIM provides "information" rather than just "capacity."
- Relative improvements of several percentage points are achieved with minimal parameter costs (+0.27% to +7%), providing evidence for "quantum advantage without quantum speedup."
- Success across both architectures (Transformer and GNN) validates the claim of being "architecture-agnostic."

## Highlights & Insights
- **QFIM as a Modality, Not Auxiliary Loss**: Unlike most hybrid quantum-classical methods that treat VQCs as part of an end-to-end chain, Quiver extracts QFIM as "data" to be consumed by classical SOTA models. This decoupling allows the method to run on classical simulators today without relying on NISQ hardware.
- **Zero-Initialized Gating Design**: Initializing $\alpha$ to 0 ensures baseline equivalence, making the argument that "Quiver's gains originate from QFIM information" rigorous at the design level.
- **2A2Q Encoding**: By encoding chemical bonds into entanglement strength $e_{\text{bond}}$, the quantum circuit acts as a "physics-aware feature extractor."
- **Cross-domain Stability**: Stable improvements in High Energy Physics (Transformer) and Molecular Chemistry (GNN) suggest that Quantum Fisher geometry encodes a "domain-agnostic many-body correlation structure."
- **"Harvesting the Future"**: Demonstrates quantifiable performance gains for large models using classically simulated VQCs, offering an immediately applicable direction for quantum machine learning in the pre-fault-tolerant era.

## Limitations & Future Work
- Classical simulation costs limit the qubit count to $\leq 10$, necessitating the exclusion of many particles in JetClass and hydrogen atoms in QM9.
- QFIMs are calculated at a fixed reference $\boldsymbol{\theta}_0$ rather than being jointly optimized with the downstream model. The challenge for joint optimization lies in backpropagating through QFIM measurements.
- The paper does not extensively discuss the storage or time costs of QFIM pre-computation, leaving its scalability to industrial-sized datasets (e.g., full JetClass) insufficiently proven.
- Comparison with classical baselines is relatively narrow, focusing only on ParT and DimeNet++, lacking horizontal comparisons with other "explicit higher-order" methods like EFN or PointNet++.

## Related Work & Insights
- **vs. Bal et al. 2025 (1P1Q)**: Adopts their 1P1Q encoding but innovates by using QFIM as a fused view rather than using VQC directly for prediction, bypassing the "VQC performs worse than SOTA" bottleneck.
- **vs. Classical Multi-modal Fusion**: Unlike typical image/text fusion, Quiver's second modality is generated from the first via a physically interpretable transform, eliminating cross-modality alignment issues.
- **vs. Increasing Model Capacity**: Comparisons with wider baselines and the marginal 0.27% parameter increase in 𝒬DimeNet++ confirm that gains come from informational content, not just parameter count.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models](softplus_attention_with_re-weighting_boosts_length_extrapolation_in_large_langua.md)
- [\[ICML 2026\] TriForces: Augmenting Atomistic GNNs for Transferable Representations](triforces_augmenting_atomistic_gnns_for_transferable_representations.md)
- [\[ICML 2026\] $\mathbb{R}^{2k}$ is Theoretically Large Enough for Embedding-based Top-$k$ Retrieval](mathbbr2k_is_theoretically_large_enough_for_embedding-based_top-k_retrieval.md)
- [\[ICML 2026\] Rethink the Role of Neural Decoders in Quantum Error Correction](rethink_the_role_of_neural_decoders_in_quantum_error_correction.md)
- [\[ICLR 2026\] Augmenting Representations with Scientific Papers](../../ICLR2026/physics/augmenting_representations_with_scientific_papers.md)

</div>

<!-- RELATED:END -->
