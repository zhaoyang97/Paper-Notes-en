---
title: >-
  [Paper Note] MōLe-Λ: Learning the Coupled-Cluster Response State for Energies, Gradients, and Properties
description: >-
  [ICML 2026][Physics & Scientific Computing][Coupled-Cluster Theory] MōLe-Λ extends molecular orbital learning from predicting only Coupled-Cluster right-state $T$ amplitudes to simultaneously predicting left-state $\Lamb…
tags:
  - "ICML 2026"
  - "Physics & Scientific Computing"
  - "Coupled-Cluster Theory"
  - "CCSD"
  - "$\\Lambda$ amplitudes"
  - "Equivariant Neural Networks"
  - "Molecular Orbitals"
  - "Response Properties"
date: 2026-05-08
content_hash: 682fd5b46badf256
---

# MōLe-Λ: Learning the Coupled-Cluster Response State for Energies, Gradients, and Properties

**Conference**: ICML 2026  
**arXiv**: [2605.29622](https://arxiv.org/abs/2605.29622)  
**Code**: None  
**Area**: Physics / Quantum Chemistry / Equivariant Neural Networks  
**Keywords**: Coupled-Cluster Theory, CCSD, $\Lambda$ amplitudes, Equivariant Neural Networks, Molecular Orbitals, Response Properties  

## TL;DR
MōLe-Λ extends molecular orbital learning from predicting only Coupled-Cluster right-state $T$ amplitudes to simultaneously predicting left-state $\Lambda$ amplitudes. Using a single equivariant network, it reads $(T_1, T_2, \Lambda_1, \Lambda_2)$ directly from localized Hartree–Fock orbitals. On QM7, the energy/force MAEs are only 0.10 mHa / 0.12 mHa/Bohr. Moreover, response properties—including dipole, quadrupole, polarizability, electron density, and pair density—are all resolved from the same learned "response state," achieving a speedup of over two orders of magnitude compared to CCSD+$\Lambda$ solvers.

## Background & Motivation

**Background**: Coupled-Cluster theory (CCSD/CCSD(T)) is considered the "gold standard" of quantum chemistry, but its formal scaling is $\mathcal{O}(N^6)$, making it computationally prohibitive for larger molecules. Machine learning has alleviated this via two paths: directly fitting energy and forces as Machine Learning Interatomic Potentials (MLIPs, e.g., Mace/eSEN), or learning one-particle quantities like density, density matrices, or Fock matrices to accelerate self-consistent fields or reconstruct observables.

**Limitations of Prior Work**: MLIPs only produce energy/forces and cannot provide properties like dipole, quadrupole, or polarizability that depend on correlated electronic states. Learning density or Hamiltonians only recovers information at the one-particle level. Any property relying on the full response state (dipole/quadrupole/polarizability/electron density/electron pair density) must go through the left-state $\Lambda$ amplitudes in the Coupled-Cluster Lagrangian. However, solving the $\Lambda$ equations themselves still scales as $\mathcal{O}(N^6)$ and has lacked acceleration.

**Key Challenge**: CC theory is not variational with respect to the right-state $T$ amplitudes; thus, the total derivative of energy with respect to external parameters $\xi$, $dE/d\xi$, contains an extra $\partial t_\mu/\partial \xi$ term. Consistent derivatives are only obtained by introducing $\Lambda$ as an adjoint variable and redefining the target as the Lagrangian $\mathcal{L}(T,\Lambda)$, yielding $dE/d\xi = \partial \mathcal{L}/\partial \xi$. The previous MōLe (Thiede et al., 2026) only learned $T_1, T_2$, allowing for energy calculation but failing to capture relaxed response observables.

**Goal**: (i) Train a single neural network to simultaneously provide $(T_1, T_2, \Lambda_1, \Lambda_2)$; (ii) Maintain the equivariant, local, and size-extensible priors of the original MōLe; (iii) Avoid training separate readout heads for each property, deriving all downstream quantities from the amplitudes via standard CC post-processing; (iv) Ensure stable extrapolation across molecular sizes and geometric distortions.

**Key Insight**: The tensor structures of $\Lambda_1, \Lambda_2$ are perfectly symmetric to those of $T_1, T_2$. Both are antisymmetric tensors over occupied/virtual orbital indices and satisfy the same sign-equivariance under orbital phase flips and the same locality (vanishing between non-interacting fragments). Consequently, one can reuse the shared equivariant backbone of MōLe and simply mirror two "odd readout heads" without redesigning the architecture.

**Core Idea**: Shift from "learning properties" to "learning the state"—predicting the full CCSD response state $(T, \Lambda)$ as an object, allowing traditional CC post-processing (1-RDM, 2-RDM, CPHF) to analytically derive all observables from it.

## Method

### Overall Architecture
The input is the molecular geometry $\{\mathbf{R}_A\}$. First, a computationally cheap Restricted Hartree–Fock (RHF) calculation is performed to obtain MO coefficients $\mathbf{C}$. Occupied and virtual orbitals are separately localized (e.g., Foster–Boys), transforming non-local canonical MOs into transferable local MOs. Each local MO is treated as a graph, where padded AO coefficient vectors on atoms are embedded into an equivariant latent space. A shared backbone performs message passing within MOs and attention-based interaction between MOs. Finally, four independent "odd readout heads" yield $T_1, T_2, \Lambda_1, \Lambda_2$. The predicted amplitudes are passed to standard CCSD post-processing (Lagrangian, CPHF, 1-/2-RDM reconstruction) to generate energy, forces, dipole, quadrupole, polarizability, density, and pair density.

### Key Designs

1.  **Shared Equivariant Backbone + Four-Head Mirrored Readout**:
    - **Function**: Produces the four amplitude tensors $T$ and $\Lambda$ simultaneously using a single backbone, avoiding the training of four independent models.
    - **Mechanism**: Each local MO is embedded as an equivariant latent representation. Through Odd-MACE message passing and inter-MO attention, dual/quaternary invariant features $\mathbf{y}_{ia}$ and $\mathbf{y}_{ijab}$ are obtained. Singles are given by $t_i^a = \mathrm{OddReadout}_{T_1}(\mathbf{y}_{ia})$ and $\lambda_a^i = \mathrm{OddReadout}_{\Lambda_1}(\mathbf{y}_{ia})$; doubles are given by $t_{ij}^{ab} = \mathrm{OddReadout}_{T_2}(\mathbf{y}_{ijab})$ and $\lambda_{ab}^{ij} = \mathrm{OddReadout}_{\Lambda_2}(\mathbf{y}_{ijab})$. "Odd readout" refers to sign-equivariance under orbital phase flips, ensuring predicted amplitudes behave correctly under MO phase gauge choices.
    - **Design Motivation**: All four tensors satisfy the same symmetries and antisymmetric index structures. Forcing a shared backbone reduces parameters and binds them in the same latent space, preserving the algebraic consistency required by downstream CC post-processing. The mirrored heads align the inductive bias of $\Lambda$ with that of $T$.

2.  **MP2 Residual Targets**:
    - **Function**: Switches from "learning the full tensor" to "learning corrections relative to MP2" in low-data regimes, allowing the network to focus on physically small but chemically critical higher-order correlations.
    - **Mechanism**: For closed-shell real amplitudes, MP2 provides a zeroth-order baseline $t_{ij,\mathrm{MP2}}^{ab} = \langle ij||ab\rangle / (\varepsilon_i+\varepsilon_j-\varepsilon_a-\varepsilon_b)$, with $T_1^{\mathrm{MP2}}=0$, $\Lambda_2^{\mathrm{MP2}} = T_2^{\mathrm{MP2}}$, and $\Lambda_1^{\mathrm{MP2}}=0$. After transforming canonical MP2 amplitudes to the local gauge, the residual mode fits $\Delta t_{ij}^{ab} = t_{ij,\mathrm{CCSD}}^{ab} - t_{ij,\mathrm{MP2}}^{ab}$ and $\Delta \lambda_{ab}^{ij} = \lambda_{ab,\mathrm{CCSD}}^{ij} - t_{ij,\mathrm{MP2}}^{ab}$.
    - **Design Motivation**: CCSD labels are expensive. In low-data scenarios, it is difficult for a network to learn the entire correlation structure. Using MP2 to subtract the known leading-order dynamical correlation allows the NN to only learn the "difference," injecting physical priors into the target and significantly reducing sample complexity.

3.  **Amplitude Reconstruction Loss instead of Property Loss**:
    - **Function**: No properties are included in the loss function; only the four amplitude tensors are supervised, ensuring the model learns the "state" rather than specific "properties."
    - **Mechanism**: The loss for each molecule is $\mathcal{J}_{\mathrm{amp}} = \frac{1}{B}\sum_{b}\sum_{X\in\{T_1,T_2,\Lambda_1,\Lambda_2\}} w_X \sum_{n=1}^{N_X^{(b)}} (\hat X_{b,n} - X_{b,n}^{\mathrm{ref}})^2$, with weights $w_X$ set to 1. Downstream energy is derived from $E_{\mathrm{corr}} = \sum_{ijab}(\frac{1}{4}t_{ij}^{ab}+\frac{1}{2}t_i^a t_j^b)\langle ij||ab\rangle$, forces from $\mathbf{F}_A = -\partial \mathcal{L}(T,\Lambda)/\partial \mathbf{R}_A$ (including CPHF orbital response), and one-/two-particle observables via 1-RDM and 2-RDM reconstruction.
    - **Design Motivation**: Directly supervising a specific property might make the model accurate for that property but distorted for others. Supervising amplitudes forces the network to fit the entire response state, ensuring all properties enjoy algebraic consistency and naturally supporting new observables (e.g., higher-order multipoles) not seen during training.

## Key Experimental Results

### Main Results

Trained on QM7 (5732 molecules) and tested on QM7 (1433 molecules) plus three generalization sets (18 amino acids, 100 PubChem molecules with 14 heavy atoms, and three geometric distortion scans), using CCSD/def2-SVP labels. Energy and Force MAE (units: mHa, mHa/Bohr):

| Method | QM7 E | QM7 F | Amino Acid E | Amino Acid F | PubChem E | PubChem F | Diels-Alder E | Diels-Alder F |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MP2 | 57.32 | 1.50 | 60.49 | 1.33 | 82.55 | 1.32 | 69.33 | 1.18 |
| Mace (Direct CCSD) | 0.79 | 1.20 | 9.03 | 9.99 | 19.45 | 9.44 | 11.25 | 7.99 |
| Mace+MP2 (Δ-learning) | 0.16 | 0.23 | 0.51 | 1.90 | 2.07 | 2.49 | 1.61 | 1.43 |
| eSEN+MP2 | 0.15 | 0.17 | 3.20 | 0.69 | 8.12 | 1.81 | 1.81 | 1.94 |
| **Ours (MōLe-Λ)** | **0.10** | **0.12** | **0.37** | **0.27** | **0.63** | **0.26** | **1.09** | **0.24** |

On QM7, amplitude MAEs for $T_1, \Lambda_1$ are approximately $2.6\text{-}2.7\times 10^{-5}$, and for $T_2, \Lambda_2$ approximately $5.3\times 10^{-7}$. Multipole MAEs for response properties (dipole, quadrupole, polarizability) decreased significantly compared to HF, MP2, and the right-state-only MōLe-XCCSD.

### Ablation Study

| Configuration / Evaluation Dimension | Key Observation | Value |
| :--- | :--- | :--- |
| Direct vs. MP2 Residual Mode | Residual mode has significantly lower MAE in low-data regimes; they converge as data increases | Physics priors are most valuable when samples are scarce |
| Right-state only MōLe (XCCSD) | Errors in dipole, density, and pair density are significantly higher than MōLe-Λ | $\Lambda$ is essential information for relaxed response observables |
| Across Molecule Size (QM7 → Amino Acids/PubChem) | Geometric MLIP errors amplify by 10x+; MōLe-Λ only by 3-6x | Local orbital amplitudes are truly size-transferable representations |
| Out-of-equilibrium Scans | Mace gives "erratic" predictions; MōLe-Λ remains stable with low error | Learning the state is more robust than learning properties for extrapolation |
| Computational Cost (H100 / C17H36) | Standard CCSD runs out of VRAM; MōLe-Λ scales beyond C21; $(T, \Lambda)$ prediction is >100x faster than CCSD solvers | Empirical scaling is much lower than $\mathcal{O}(N^3)$ |

### Key Findings
- **Learning States > Learning Properties**: By supervising only four amplitude tensors, downstream energy, forces, dipole, quadrupole, polarizability, density, and pair density are all superior, avoiding the common trade-off where one property improves at the expense of others.
- **$\Lambda$ is the Key Marginal Gain**: Without $\Lambda$, electron density residuals are diffused across the molecular volume; with $\Lambda$, errors near bonds are suppressed, and the broad MP2 error lobes in pair density difference maps (2-RDM) are nearly eliminated.
- **Physics Priors Reduce Data Requirements**: MP2 residualization delegates leading-order correlation to perturbation theory, letting the network learn only the delta; this remarkably boosts data efficiency in low-data regimes.
- **MLIPs are Fragile under Distortions**: Mace directly fitting CCSD fails significantly on butane dihedral scans compared to MōLe-Λ, indicating that geometric feature spaces cannot carry the burden of predicting electronic rearrangement without orbitals.

## Highlights & Insights
- **"Learning the State" Paradigm Updates Supervision Granularity**: Chemistry ML has long focused on "which property to learn"; this work shifts the target to the electronic structure object itself, making properties byproducts and naturally avoiding multi-task conflicts.
- **Architecture Mirroring over Stacking**: The shared and mirrored backbone for $T$ and $\Lambda$ heads means adding $\Lambda$ introduces almost no parameters or training complexity while expanding the set of recoverable observables by an order of magnitude. This "near-zero cost extension" philosophy is a valuable lesson for other physical ML tasks.
- **Local MOs as True Transferable Representations**: Geometry-based MLIPs degrade quickly during size extrapolation, whereas local orbital amplitudes naturally satisfy size-extensibility (amplitudes between non-interacting fragments vanish), suggesting the "correct inductive bias" for molecular ML may not be purely Euclidean.
- **Transferability to Other Adjoint-Dependent Physics**: The "right state + left state" structure of the CCSD Lagrangian is isomorphic to structures in elasticity, optimal control, and variational inference. The idea of "learning the adjoint" can be transferred to other scientific computing problems requiring response derivatives.

## Limitations & Future Work
- **Basis Set and Element Constraints**: Only trained on the def2-SVP basis set and five elements (C/N/O/S/H) from QM7. Larger basis sets (aug-cc-pVTZ), transition metals, and open-shell systems are not yet covered.
- **Dense $T_2, \Lambda_2$ Output as a Potential Bottleneck**: Currently only an issue for molecules with dozens of heavy atoms; larger systems will require sparse, local, or compressed doubles representations.
- **Unoptimized Pre-processing**: HF, localization, and MP2 run on the CPU, creating a bottleneck relative to the GPU forward pass.
- **Absence of Triple Excitations**: CCSD(T) is the true "gold standard." This work only reaches the CCSD level; how to integrate $T_3$ corrections for triple excitations remains an open question.

## Related Work & Insights
- **vs. MōLe (Thiede et al., 2026)**: MōLe only predicted $T$ amplitudes and used XCCSD for energy and 1-body density. MōLe-Λ closes the loop on the response state, recovering observables (dipole/quadrupole/polarizability/pair density) accessible only via $\Lambda$.
- **vs. Mace / eSEN**: MLIPs only provide energy and forces, and geometric feature spaces are fragile under size extrapolation and distortion scans. MōLe-Λ wins on both performance and transferability by accessing the electronic structure object.
- **vs. Δ-learning (Mace+MP2, eSEN+MP2)**: While Δ-learning uses MP2 as a baseline, it does so at the property level. MōLe-Λ brings residualization to the amplitude level, aligning physical priors with the supervision target.
- **vs. Learning Density/Hamiltonian**: Those paths typically only recover one-particle observables. This work includes two-particle quantities via the 2-RDM, providing broader coverage.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to elevate the ML supervision target from "properties" to the full CCSD response state $(T, \Lambda)$; a paradigm shift rather than an incremental improvement.
- Experimental Thoroughness: ⭐⭐⭐⭐ QM7 training + three generalization sets + out-of-equilibrium scans + multiple observables + scaling comparisons; lacks verification on larger basis sets and heavy elements.
- Writing Quality: ⭐⭐⭐⭐ The causal chain from Lagrangian motivation to the mirrored architecture is clear; some RDM reconstruction details are unfortunately relegated to the appendix.
- Value: ⭐⭐⭐⭐⭐ Transforms CC-level response properties from "O(N^6) unreachable" to "attainable via a single forward pass," which is of immense practical significance for catalysis, materials, and molecular design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning](../../ICLR2026/physics/drift-net_a_spectral--coupled_neural_operator_for_pdes_learning.md)
- [\[ICLR 2026\] HyperKKL: Enabling Non-Autonomous State Estimation through Dynamic Weight Conditioning](../../ICLR2026/physics/hyperkkl_enabling_non-autonomous_state_estimation_through_dynamic_weight_conditi.md)
- [\[ICML 2026\] Topology-Preserving Neural Operator Learning via Hodge Decomposition](topology-preserving_neural_operator_learning_via_hodge_decomposition.md)
- [\[ICML 2026\] Hermite-NGP: Gradient-Augmented Hash Encoding for Learning PDEs](hermite-ngp_gradient-augmented_hash_encoding_for_learning_pdes.md)
- [\[ICML 2026\] A Call to Lagrangian Action: Learning Population Mechanics from Temporal Snapshots](a_call_to_lagrangian_action_learning_population_mechanics_from_temporal_snapshot.md)

</div>

<!-- RELATED:END -->
