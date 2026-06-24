---
title: >-
  [Paper Note] UniSim: A Unified Simulator for Time-Coarsened Dynamics of Biomolecules
description: >-
  [ICML 2025][Computational Biology][Molecular Dynamics] UniSim is the first deep generative model for cross-domain (small molecules/peptides/proteins) all-atom time-coarsened molecular dynamics. Through a three-stage pipeline—multi-head pre-training for unified atomic representation, a stochastic interpolation vector field model for long-timestep state propagation, and a force-guidance kernel for parameter-efficient adaptation to different chemical environments—it achieves tra…
tags:
  - "ICML 2025"
  - "Computational Biology"
  - "Molecular Dynamics"
  - "Time-Coarsening"
  - "Cross-Domain Pre-training"
  - "Stochastic Interpolation"
  - "Force Guidance"
  - "All-Atom Simulation"
date: 2026-05-08
content_hash: f66eccbc5de782c6
---

# UniSim: A Unified Simulator for Time-Coarsened Dynamics of Biomolecules

**Conference**: ICML 2025  
**arXiv**: [2506.03157](https://arxiv.org/abs/2506.03157)  
**Code**: [https://github.com/yaledeus/UniSim](https://github.com/yaledeus/UniSim)  
**Area**: Molecular Dynamics / Computational Biology  
**Keywords**: Molecular Dynamics, Time-Coarsening, Cross-Domain Pre-training, Stochastic Interpolation, Force Guidance, All-Atom Simulation

## TL;DR

UniSim is the first deep generative model for cross-domain (small molecules/peptides/proteins) all-atom time-coarsened molecular dynamics. Through a three-stage pipeline—multi-head pre-training for unified atomic representation, a stochastic interpolation vector field model for long-timestep state propagation, and a force-guidance kernel for parameter-efficient adaptation to different chemical environments—it achieves transferable dynamics simulation across molecular domains.

## Background & Motivation

**Background**: Classical molecular dynamics (MD) simulations require extremely small femtosecond-scale ($\Delta t \approx 10^{-15}$ s) timesteps to ensure numerical integration stability, limiting their ability to simulate long-timescale processes such as protein folding. Quantum mechanics methods are accurate but computationally expensive, while empirical force field methods are fast but lack sufficient accuracy.

**Limitations of Prior Work**: Recent deep learning methods (e.g., FBM, Timewarp, ITO) substantially accelerate simulation by learning a "time-coarsened" propagation mapping $\mathbf{X}_t \to \mathbf{X}_{t+\tau}$ ($\tau \gg \Delta t$). However, two major bottlenecks remain: (1) almost all methods are confined to a single molecular domain (e.g., only peptides or only proteins), lacking cross-domain transferability; (2) some models rely on hand-crafted domain-specific representations (such as $\gamma$-carbon labeling for leucine), making them incapable of identifying proteins containing non-canonical amino acids.

**Key Challenge**: MD trajectory data are scarce, while molecular systems are highly diverse with variable chemical environments (temperature/pressure/solvent). This necessitates a generalizable model rather than training a separate model for each molecular system.

**Goal**: Build a unified all-atom time-coarsened simulator transferable across small molecules, peptides, and proteins, while enabling parameter-efficient fine-tuning to adapt to different chemical environments.

**Key Insight**: Leverage cross-domain 3D molecular data via multi-head pre-training to acquire unified atomic representations, learn state propagation based on a stochastic interpolation generative framework, and utilize a force-guidance kernel to achieve "train once, adapt to multiple environments."

**Core Idea**: Pre-trained unified representation + Stochastic interpolation vector field + Force-guidance kernel = Cross-domain transferable time-coarsened MD simulator.

## Method

### Overall Architecture

UniSim consists of four stages: (a) multi-head pre-training of the atomic representation model $\varphi$ on multi-domain 3D molecular data; (b) training the vector field model $\phi = \{v, \eta_z\}$ based on a stochastic interpolation framework to learn the mapping $\mathbf{X}_t \to \mathbf{X}_{t+\tau}$; (c) training a force-guidance kernel $\zeta$ to adapt to different chemical environments (with $\varphi, \phi$ parameters frozen); and (d) iteratively solving an SDE during inference to generate trajectories. The underlying network architecture employs the SO(3)-equivariant graph neural network TorchMD-NET.

### Key Designs

1. **Gradient-Environment Subgraph**: Addresses the issue of scale disparities across molecular domains (small molecules with dozens of atoms vs. proteins with thousands of atoms). For macromolecules with more than 1000 atoms, a center atom $c$ is chosen at random to construct a gradient subgraph $\mathcal{G}_g = \{j : \|\mathbf{x}_j - \mathbf{x}_c\|_2 < \delta_{\min}\}$ and an environment subgraph $\mathcal{G}_e = \{j : \|\mathbf{x}_j - \mathbf{x}_c\|_2 < \delta_{\max}\}$ ($\delta_{\min} = 8$Å, $\delta_{\max} = 20$Å). Only $\mathcal{G}_e$ is input to the model, and only atoms in $\mathcal{G}_g$ undergo loss computation. When $\delta_{\max} - \delta_{\min}$ is sufficiently large, the influence of atoms outside $\mathcal{G}_e$ on $\mathcal{G}_g$ is negligible—which is physically reasonable and computationally efficient.

2. **Atomic Embedding Expansion**: Identical elements in proteins exhibit discrete chemical patterns (e.g., CA and CB for carbon), with highly regular bond lengths and angles. Representing elements based solely on the periodic table results in a vocabulary that is too coarse. Method: Define a base vocabulary $\mathbf{A}_b \in \mathbb{R}^{A \times H}$ and an expanded vocabulary $\mathbf{A}_e \in \mathbb{R}^{A \times D \times H}$ ($D$ is the number of patterns per element), compute the pattern probability $\mathbf{w}_i = \text{softmax}(\text{lin}(\mathbf{A}_b[i], \mathbf{n}_i))$ using neighbor information, and obtain the final embedding as $\mathbf{z}_i = \text{lin}(\mathbf{A}_b[i], \mathbf{w}_i^\top \mathbf{A}_e[i], \mathbf{n}_i)$. Ablation studies confirm that removing this expanded embedding degrades PWD from 0.332 to 0.389.

3. **Force Guidance Kernel**: Freezes all parameters of $\varphi, \phi$, and introduces a new TorchMD-NET $\Psi$ and output network $\psi$ to fit the intermediate force field $\nabla \varepsilon_t$. Goal: generate the target distribution $p_t(\cdot) \propto q_t(\cdot) \exp(-\alpha \varepsilon_t(\cdot))$ by modifying the denoiser to $\eta_z'(t, \mathbf{X}) = \eta_z(t, \mathbf{X}) + \alpha \gamma(t) \nabla \varepsilon_t(\mathbf{X})$. To ensure continuity with the MD force field at the endpoints, $\psi$ takes the interpolated form $(1-t)\psi_0 + t\psi_1 + t(1-t)\psi_2$. Changes in the chemical environment are reflected in the generative distribution through the MD potential energy $\varepsilon$—"pre-train once, adapt to multiple environments."

### Loss & Training

- **Pre-training**: Force alignment $\mathcal{L}_o = \|\nabla_{\mathbf{X}}(\sum_i \mathbf{H}_\text{out}[i]) + \mathbf{F}\|_2^2$ (off-equilibrium) + denoising $\mathcal{L}_e$ (equilibrium), utilizing multiple heads to distinguish between different force fields.
- **Vector Field**: $\mathcal{L}_v = \mathbb{E}[\|v(t, \mathbf{X}_t) - (\mathbf{X}_1 - \mathbf{X}_0)\|^2 + \|\eta_z(t, \mathbf{X}_t) - \mathbf{Z}\|^2]$
- **Force Guidance**: Endpoint force fitting + intermediate force field fitting (four joint losses).
- OpenMM energy minimization is applied post-inference for conformation refinement of peptides/proteins (averaging 69.3 steps).
- Training Environment: 8× RTX 3090, completed within one week.

## Key Experimental Results

### Main Results: Peptides (PepMD 14 test peptides, JS distance ↓)

| Model | PWD↓ | RG↓ | TIC↓ | TIC-2D↓ | VAL-CA↑ | CONTACT↓ |
|------|------|-----|------|---------|---------|----------|
| FBM | 0.361 | 0.411 | 0.510 | 0.736 | 0.539 | 0.205 |
| Timewarp | 0.362 | 0.386 | 0.514 | 0.745 | 0.028 | 0.195 |
| ITO | 0.367 | 0.371 | 0.495 | 0.748 | 0.160 | 0.174 |
| SD | 0.727 | 0.776 | 0.541 | 0.782 | 0.268 | 0.466 |
| UniSim/g | 0.332 | 0.332 | 0.510 | 0.738 | 0.505 | 0.162 |
| **UniSim** | **0.328** | **0.330** | **0.510** | **0.731** | **0.575** | **0.157** |

### Proteins (ATLAS 14 test proteins)

| Model | PWD↓ | RG↓ | TIC↓ | VAL-CA↑ | CONTACT↓ |
|------|------|-----|------|---------|----------|
| FBM | 0.519 | 0.597 | 0.621 | 0.012 | 0.252 |
| ITO | 0.588 | 0.775 | 0.624 | 0.052 | 0.428 |
| SD | 0.604 | 0.762 | 0.605 | 0.001 | 0.235 |
| UniSim/g | 0.508 | 0.569 | 0.543 | 0.071 | 0.171 |
| **UniSim** | **0.506** | **0.554** | **0.542** | **0.079** | **0.173** |

### Ablation Study

| Ablation Item | Key Changes |
|-------|---------|
| Removing atomic embedding expansion | PWD: 0.332→0.389, CONTACT: 0.162→0.228 |
| Force guidance (increasing $\alpha$) | VAL-CA improves significantly, but with a declining trend in distribution diversity |
| SDE steps (increasing $T$) | Most metrics degrade—smaller $T$ suffices to balance accuracy and efficiency |
| Small molecule TIC (UniSim/g→UniSim) | 0.408→0.368, force guidance improves cross-domain transfer |

### Key Findings

- Cross-domain pre-training does not compromise single-domain performance: UniSim consistently outperforms from-scratch trained FBM on peptides.
- Crucial role of force-guidance kernel: VAL-CA improves from 0.505 to 0.575 (peptides), significantly boosting conformational validity.
- Breakthrough in the protein domain: CONTACT decreases from 0.252 of FBM to 0.173 (a 31% improvement).
- Inference efficiency: ESS/s is approximately 25 times faster than classical MD.
- Successfully recovers five metastable states ($C5$, $C7eq$, $\alpha_R'$, $\alpha_R$, $\alpha_L$) in the Alanine-Dipeptide case study.

## Highlights & Insights

- The first cross-domain all-atom time-coarsened simulator, successfully introducing the unified pre-training paradigm into the molecular dynamics field.
- The gradient-environment subgraph design is elegant, addressing the computational bottleneck of cross-domain scale disparities under physically reasonable assumptions.
- The "frozen backbone + trained adapter" approach of the force-guidance kernel shares strong conceptual similarities with LoRA in NLP.

## Limitations & Future Work

- Cumulative prediction errors in autoregressive generation can cause instability in long-range simulations of large proteins, necessitating OpenMM post-processing.
- The evaluated trajectory length is relatively short ($10^3$ steps), potentially missing additional metastable states.
- The scale of pre-training data is limited by the scarcity of MD trajectories.
- Integration with structure prediction models such as AlphaFold3 has not yet been established.

## Related Work & Insights

- FBM (Yu et al. 2024): The force guidance module of UniSim is directly inherited from FBM but extended to cross-domain scenarios.
- DPA-2 (Zhang et al. 2024): Shares a similar multi-task pre-training concept but focuses on materials systems.
- Insights: The paradigm of unified pre-training combined with domain adaptation can be generalized to other scientific simulation fields.

## Rating

⭐⭐⭐⭐ — The first cross-domain all-atom time-coarsened simulator, featuring solid technical innovations (gradient subgraph, atomic embedding expansion, force-guidance kernel). Experiments across three molecular domains consistently outperform baselines. Cumulative error in long-range protein simulations and dependence on post-processing remain the primary limitations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Remasking Discrete Diffusion Models with Inference-Time Scaling](../../NeurIPS2025/computational_biology/remasking_discrete_diffusion_models_with_inference-time_scaling.md)
- [\[NeurIPS 2025\] Unified All-Atom Molecule Generation with Neural Fields](../../NeurIPS2025/computational_biology/unified_all-atom_molecule_generation_with_neural_fields.md)
- [\[ICLR 2026\] Unified Biomolecular Trajectory Generation via Pretrained Variational Bridge](../../ICLR2026/computational_biology/unified_biomolecular_trajectory_generation_via_pretrained_variational_bridge.md)
- [\[ICML 2025\] UniMoMo: Unified Generative Modeling of 3D Molecules for De Novo Binder Design](unimomo_unified_generative_modeling_of_3d_molecules_for_de_novo_binder_design.md)
- [\[NeurIPS 2025\] ConfRover: Simultaneous Modeling of Protein Conformation and Dynamics via Autoregression](../../NeurIPS2025/computational_biology/confrover_simultaneous_modeling_of_protein_conformation_and_dynamics_via_autoreg.md)

</div>

<!-- RELATED:END -->
