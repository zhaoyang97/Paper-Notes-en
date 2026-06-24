---
title: >-
  [Paper Note] Piloting Structure-Based Drug Design via Modality-Specific Optimal Schedule
description: >-
  [ICML 2025][Computational Biology][Structure-Based Drug Design] Proposes the VLB-Optimal Scheduling (VOS) strategy. By theoretically analyzing the path-dependent VLB characteristics of joint noise scheduling of multimodal data (continuous 3D coordinates + discrete 2D topology), it utilizes dynamic programming to search for the optimal noise scheduling path, achieving state-of-the-art performance in SBDD with a 95.9% PoseBusters pass rate on CrossDock.
tags:
  - "ICML 2025"
  - "Computational Biology"
  - "Structure-Based Drug Design"
  - "Bayesian Flow Network"
  - "Multimodal Noise Schedule"
  - "VLB Optimization"
  - "Molecular Geometry"
date: 2026-05-08
content_hash: 33b326a872a240ac
---

# Piloting Structure-Based Drug Design via Modality-Specific Optimal Schedule

**Conference**: ICML 2025  
**arXiv**: [2505.07286](https://arxiv.org/abs/2505.07286)  
**Code**: [https://github.com/AlgoMole/MolCRAFT](https://github.com/AlgoMole/MolCRAFT)  
**Area**: Diffusion Models / Molecule Generation  
**Keywords**: Structure-Based Drug Design, Bayesian Flow Network, Multimodal Noise Schedule, VLB Optimization, Molecular Geometry

## TL;DR
Proposes the VLB-Optimal Scheduling (VOS) strategy. By theoretically analyzing the path-dependent VLB characteristics of joint noise scheduling of multimodal data (continuous 3D coordinates + discrete 2D topology), it utilizes dynamic programming to search for the optimal noise scheduling path, achieving state-of-the-art performance in SBDD with a 95.9% PoseBusters pass rate on CrossDock.

## Background & Motivation
**Background**: Structure-Based Drug Design (SBDD) utilizes deep generative models (such as diffusion models and BFNs) to generate ligand molecules conditioned on protein structures.

**Limitations of Prior Work**: Existing models exhibit a significant drop in intramolecular validity (bond length/angle distribution) under strict PoseBusters testing, indicating inconsistency between the generated 3D geometry and the 2D molecular topology.

**Key Challenge**: Default noise scheduling causes the 3D modality (atom coordinates) to be denoised first, preventing the model from effectively using the 2D topology information to constrain the 3D structure. The "3D-dominant" probability path is suboptimal.

**Goal**: How to design an optimal noise schedule for multiple modalities so that 2D and 3D modalities can guide each other?

**Key Insight**: Proving that the VLB is path-dependent in multimodal scenarios (unlike in single-modality settings), and then searching for the optimal path.

**Core Idea**: Multimodal VLB is no longer invariant to the shape of the schedule. The optimal inter-modality scheduling path can be found through decoupled timestep training followed by dynamic programming search.

## Method

### Overall Architecture
Based on the BFN framework: (1) During the training phase, independent decoupled sampling of $(t_c, t_d)$ is used to train a generalized loss, allowing a single model to cover all possible noise combinations; (2) Estimate the 2D cost matrix $C(t_c, t_d)$; (3) Use dynamic programming to search for the minimum cumulative cost path, yielding the optimal schedule $\boldsymbol{\beta}^*$.

### Key Designs

1. **Path-Dependent VLB (Eq.8-10)**:

    - Function: Proves that multimodal VLB is dependent on the coupled path of the noise schedule.
    - Mechanism: In the single-modality setting, $\mathcal{L}^\infty$ only depends on $\beta(0), \beta(1)$ (endpoint invariance). In the multimodal setting, it is formulated as a path integral $\int_{\beta_c,\beta_d} \mathbb{E}\|\mathbf{x} - \tilde{\mathbf{x}}_\phi\|^2 d\boldsymbol{\beta}$, where different paths result in different VLBs.
    - Design Motivation: This provides the theoretical foundation for designing the optimal schedule.

2. **Generalized Loss + Decoupled Timestep Training (Eq.14)**:

    - Function: Trains a single model to support arbitrary combinations of $(t_c, t_d)$.
    - Mechanism: $\dot{\mathcal{L}}^\infty = \frac{1}{2}\int_0^1\int_0^1 \mathbb{E}\|\mathbf{x} - \tilde{\mathbf{x}}_\phi(\boldsymbol{\theta}, \boldsymbol{t})\|^2 dt_c dt_d$, where $t_c, t_d \sim U(0,1)$ are sampled independently during training.
    - Design Motivation: Enables evaluation of the loss over the entire $[0,1]^2$ plane with a single training session, avoiding the need to retrain for each individual schedule.

3. **Dynamic Programming Search for Optimal Path (Algorithm 1)**:

    - Function: Finds the minimum cumulative cost path on a discretized grid of $(t_c, t_d)$.
    - Mechanism: $J(t_c,t_d) = \min_{(\epsilon_c,\epsilon_d)}(J(t_c-\epsilon_c, t_d-\epsilon_d) + \boldsymbol{\alpha}C(t_c,t_d))$ traversing from $[0,0]$ to $[1,1]$.
    - Design Motivation: Exhaustive search over all paths is intractable; DP guarantees finding the global optimum in polynomial time.

### Intuition of the Optimal Schedule
The discovered optimal path exhibits a two-stage characteristic: (1) Shape-driven drafting phase ($t<0.3$): generates coarse 3D atom positions first; (2) Topology-driven fitting phase ($t>0.8$): refines the 3D conformation conditioned on the 2D molecular topology.

## Key Experimental Results

### Main Results

| Method | PB-Valid ↑ | Vina Dock ↓ | scRMSD<2Å ↑ |
|------|-----------|-------------|-------------|
| TargetDiff | 50.5% | -7.80 | 37.1% |
| DecompDiff | 71.7% | -7.03 | 24.2% |
| MolCRAFT (Default Schedule) | 84.3% | -7.90 | 38.7% |
| **MolPilot (VOS)** | **95.9%** | **-8.08** | **34.0%** |

### Ablation Study

| Schedule Type | PB-Valid ↑ | Description |
|----------|-----------|------|
| Default Schedule | 84.3% | 3D-dominant path |
| Linear Interpolation | ~90% | Uniform coupling |
| **VOS Optimal** | **95.9%** | Two-stage path |

### Key Findings
- VOS improves the PoseBusters pass rate by >10% compared to the default schedule.
- Bond length/angle distribution plots demonstrate that MolPilot yields predictions closest to the ground-truth distribution.
- Performance remains robust in OOD (PoseBusters) evaluation.

## Highlights & Insights
- **Theory-Driven Schedule Design**: Formulated starting from VLB path-dependency instead of heuristic trial-and-error.
- **Single Training, Multi-Schedule Evaluation**: The generalized loss design with decoupled timesteps is highly elegant.
- **Transferable Concept**: The VOS methodology can be generalized to the noise schedule design of any multimodal generative task.

## Related Work & Insights
- **vs TargetDiff/DecompDiff (Diffusion)**: These methods utilize fixed noise schedules and do not account for the schedule coupling between different modalities. MolPilot identifies the optimal coupling path via VOS.
- **vs MolCRAFT (BFN)**: MolPilot is built directly on the MolCRAFT framework; VOS can be considered as a noise schedule optimization for BFNs.
- **vs EquiFM**: EquiFM designs mixing paths using information-theoretic heuristics and lacks theoretical optimality guarantees. VOS is theoretically grounded in VLB optimality.
- The rationale of VOS can be directly transferred to other multimodal generation problems (e.g., devising noise schedules for text and image modalities in joint text-to-image generation).

## Limitations & Future Work
- The complexity of DP search grows exponentially when scaling to more modalities (>2), requiring approximate algorithms.
- The interpretability of the optimal schedule still warrants further theoretical analysis (e.g., why the two-stage path is optimal).
- The success rate of RMSD < 2Å on molecular docking tasks is only 44%, leaving substantial room for improvement.
- Training with the generalized loss (decoupled timesteps) may result in slower convergence compared to standard training.
- The efficacy of VOS remains unverified on other molecular generation tasks such as protein design.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The path-dependency of multimodal VLB is a significant theoretical discovery.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated extensively on CrossDock + PoseBusters OOD.
- Writing Quality: ⭐⭐⭐⭐ Well-integrated theory and experiments.
- Value: ⭐⭐⭐⭐⭐ Broad implications for SBDD and multimodal generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SYNC: Measuring and Advancing Synthesizability in Structure-Based Drug Design](../../ICLR2026/computational_biology/sync_measuring_and_advancing_synthesizability_in_structure-based_drug_design.md)
- [\[ICML 2025\] Flexibility-conditioned Protein Structure Design with Flow Matching](flexibility-conditioned_protein_structure_design_with_flow_matching.md)
- [\[ICML 2026\] EvoEGF-Mol: Evolving Exponential Geodesic Flow for Structure-based Drug Design](../../ICML2026/computational_biology/evoegf-mol_evolving_exponential_geodesic_flow_for_structure-based_drug_design.md)
- [\[ICML 2025\] Leveraging Partial SMILES Validation Scheme for Enhanced Drug Design in Reinforcement Learning Frameworks](leveraging_partial_smiles_validation_scheme_for_enhanced_drug_design_in_reinforc.md)
- [\[NeurIPS 2025\] Pharmacophore-Guided Generative Design of Novel Drug-Like Molecules](../../NeurIPS2025/computational_biology/pharmacophore-guided_generative_design_of_novel_drug-like_molecules.md)

</div>

<!-- RELATED:END -->
