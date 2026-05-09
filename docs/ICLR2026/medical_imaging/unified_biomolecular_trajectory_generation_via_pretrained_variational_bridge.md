---
title: >-
  [Paper Note] Unified Biomolecular Trajectory Generation via Pretrained Variational Bridge
description: >-
  [ICLR 2026][Medical Imaging][molecular dynamics] PVB (Pretrained Variational Bridge) unifies the training objectives of single-structure pretraining and paired-trajectory fine-tuning via an encoder-decoder architecture combined with augmented bridge matching, enabling cross-domain biomolecular trajectory generation. It further accelerates protein–ligand holo-state exploration through RL fine-tuning based on adjoint matching.
tags:
  - ICLR 2026
  - Medical Imaging
  - molecular dynamics
  - trajectory generation
  - variational bridge matching
  - pretraining
  - reinforcement learning fine-tuning
date: 2026-05-08
content_hash: a8041aa8c9f9e7f8
---

# Unified Biomolecular Trajectory Generation via Pretrained Variational Bridge

**Conference**: ICLR 2026
**arXiv**: [2602.07588](https://arxiv.org/abs/2602.07588)
**Code**: None
**Area**: Medical Imaging
**Keywords**: molecular dynamics, trajectory generation, variational bridge matching, pretraining, reinforcement learning fine-tuning

## TL;DR
PVB (Pretrained Variational Bridge) unifies the training objectives of single-structure pretraining and paired-trajectory fine-tuning via an encoder-decoder architecture combined with augmented bridge matching, enabling cross-domain biomolecular trajectory generation. It further accelerates protein–ligand holo-state exploration through RL fine-tuning based on adjoint matching.

## Background & Motivation

**State of the Field**: Molecular dynamics (MD) simulation is a fundamental tool for characterizing molecular behavior, yet its computational cost is prohibitively high, requiring femtosecond-level time steps. Recent deep generative models have begun learning dynamics at coarsened time scales to generate trajectories efficiently.

**Limitations of Prior Work**: Existing methods suffer from three key issues: (1) insufficient generalization across molecular systems; (2) limited molecular diversity in trajectory data, preventing full exploitation of structural information; and (3) a predominant focus on single-molecule simulation, with little attention to multi-molecule systems such as protein–ligand complexes.

**Root Cause**: The most closely related prior work, UniSim, achieves cross-domain generalization via 3D molecular pretraining; however, a training objective mismatch exists between pretraining (unconditional generation of single structures $x$) and fine-tuning (conditional generation of trajectory pairs $(x_t, x_{t+\tau})$), leading to insufficient transfer of pretrained knowledge.

**Paper Goals**: (1) How to design a unified training framework in which pretraining and fine-tuning share the same generative objective? (2) How to apply generated trajectories to rapid holo-state exploration in protein–ligand docking?

**Starting Point**: A latent variable $\mathbf{Y}_0$ is introduced to model the generative process as a Markov chain $\mathbf{X}_0 \to \mathbf{Y}_0 \to \mathbf{Y}_1$. A variational encoder maps the initial structure to a noisy latent space, and an augmented bridge matching decoder transports it to the target state.

**Core Idea**: A unified encoder-decoder framework with augmented bridge matching eliminates the objective mismatch between pretraining and fine-tuning, while adjoint-matching-based RL fine-tuning accelerates holo-state exploration.

## Method

### Overall Architecture

PVB adopts an encoder-decoder architecture. The input is a molecular conformation $(z, C, x)$ (atomic numbers, covalent bonds, 3D coordinates), and the output is the conformation at the next time step. Training proceeds in three stages:
1. **Pretraining**: Trained on large-scale high-resolution single-structure data, setting $(\mathbf{X}_0, \mathbf{Y}_1) = (x, x)$.
2. **Fine-tuning**: Fine-tuned on paired MD trajectory data, setting $(\mathbf{X}_0, \mathbf{Y}_1) = (x_t, x_{t+\tau})$.
3. **RL Fine-tuning** (optional): Reinforcement learning via adjoint matching to guide trajectories toward the holo state.

### Key Designs

1. **Variational Encoder**:

    - Function: Maps the initial state $\mathbf{X}_0$ to the latent variable $\mathbf{Y}_0$.
    - Mechanism: The prior is set as $q_e(d\mathbf{Y}_0|\mathbf{X}_0) \coloneqq \mathcal{N}(x_0, \sigma_e^2 I)$ with $\sigma_e = \sqrt{0.5}$ Å. A neural network $\varphi_e$ learns the posterior distribution $p_e$ by minimizing the KL divergence: $\mathcal{L}_{KL} = -\frac{1}{2}\mathbb{E}[1 + \log \mathbf{V} - 2\log\sigma_e - \frac{\mathbf{V}}{\sigma_e^2}]$.
    - Design Motivation: The primary purpose of introducing a latent variable is to prevent the conditional distribution from degenerating into a Dirac measure during single-structure pretraining. A sufficiently large $\sigma_e$ ensures that the encoding process retains adequate structural information while avoiding trivial decoder collapse.

2. **Augmented Bridge Matching Decoder**:

    - Function: Generates the target state $\mathbf{Y}_1$ from the latent variable $\mathbf{Y}_0$ while preserving the coupling between $(\mathbf{Y}_0, \mathbf{Y}_1)$.
    - Mechanism: A Brownian bridge path measure is defined, and a vector field $\varphi_d$ is trained to minimize $\mathcal{L}_{ABM} = \mathbb{E}_{t, (\mathbf{Y}_0, \mathbf{Y}_1)}[\|\varphi_d(t, \mathbf{Y}_0, \mathbf{Y}_t) - \frac{\mathbf{Y}_1 - \mathbf{Y}_t}{1-t}\|^2]$. At inference, the target is generated by simulating the non-Markovian SDE $d\mathbf{Y}_t = \varphi_d^*(t, \mathbf{Y}_0, \mathbf{Y}_t)dt + \sigma d\mathbf{B}_t$.
    - Design Motivation: Augmented bridge matching ensures that the endpoint coupling $\Pi_{0,1}$ is exactly preserved throughout the generative process, which is critical for faithfully reproducing MD dynamical properties. Proposition 1 guarantees that the encoder-decoder composition provides an unbiased estimate of the target conditional distribution.

3. **Adjoint Matching-Based RL Fine-tuning**:

    - Function: Introduces a control vector field $u$ to steer the generative distribution so that trajectories rapidly approach the protein–ligand holo state.
    - Mechanism: The KL-regularized objective $\max_u \mathbb{E}[r(\mathbf{Y}_1) - \frac{\beta}{2}\int_0^1 \|u\|^2 dt]$ is optimized. Via Girsanov's theorem and adjoint matching, the stochastic optimal control (SOC) problem is converted to $\mathcal{L}_{adj} = \mathbb{E}[\|u(t, \mathbf{Y}_0, \mathbf{Y}_t) + \sigma\tilde{a}(t)\|^2]$, where the reduced adjoint state $\tilde{a}$ is backpropagated through an ODE.
    - Design Motivation: Directly simulating from the apo to the holo state requires millisecond-level time scales, which is computationally intractable. RL fine-tuning steers the generative distribution via an explicit reward function, bypassing inefficient local exploration. Adjoint matching, rather than direct gradient accumulation, enables memory-efficient training.

### Loss & Training

- **Pretraining + Fine-tuning**: $\mathcal{L} = w_{KL} \cdot \mathcal{L}_{KL} + w_{ABM} \cdot \mathcal{L}_{ABM}$
- **RL Fine-tuning**: $\mathcal{L}_{adj}$, with reward function $r(\mathbf{X}) = -\text{rmsd}(\mathbf{X}, \mathbf{X}_{ref})$
- The control vector field is reparametrized as $u = \frac{1}{\sigma}(\varphi_d^u - \varphi_d^*)$, requiring no additional network.
- Pretraining data: PCQM4Mv2 + ANI-1x (small molecules), PDB (proteins), PDBBind2020 (protein–ligand complexes).

## Key Experimental Results

### Main Results (ATLAS Protein Trajectory Generation)

| Model | JSD-Rg ↓ | JSD-TIC ↓ | JSD-MSM ↓ | VAL-CA ↑ | Decorr-TIC0 ↑ |
|------|----------|-----------|-----------|----------|---------------|
| MD (10ns) | 0.379 | 0.684 | 0.596 | 0.926 | 0.000 |
| ITO | 0.792 | 0.400 | 0.469 | 0.001 | 0.714 |
| MDGEN | 0.493 | 0.400 | 0.463 | 0.098 | 0.857 |
| UniSim | 0.538 | 0.372 | 0.344 | 0.106 | 0.786 |
| **PVB** | **0.457** | **0.371** | **0.333** | **0.975** | **0.929** |

### Ablation Study / Protein–Ligand Complex (MISATO)

| Model | EMD-ligand ↓ | EMD-CoM ↓ | RMSE-CONTACT ↓ |
|------|-------------|-----------|----------------|
| ITO | 0.494 | 0.479 | 0.987 |
| UniSim | 0.196 | 0.128 | **0.049** |
| **PVB** | **0.133** | **0.089** | 0.055 |

### Key Findings
- PVB substantially outperforms all baselines on VAL-CA (conformational validity) with a score of 0.975, compared to 0.106 for UniSim, indicating highly physically plausible generated conformations.
- On slow dynamical modes (TIC, MSM), PVB surpasses 10 ns MD simulation and achieves the highest decorrelation rate (0.929).
- On protein–ligand complexes, PVB reduces ligand RMSD error by 32% relative to UniSim.
- After RL fine-tuning, the median ligand RMSD in protein–ligand docking decreases by approximately 18% compared to Vina+PVB (without RL).

## Highlights & Insights
- **Unified Pretraining–Fine-tuning Framework**: The objective mismatch between single-structure pretraining and paired trajectory fine-tuning is elegantly resolved via latent variables and augmented bridge matching. This idea is transferable to other generative models requiring cross-task transfer.
- **Substantial Advantage in VAL-CA**: Nearly all conformations generated by PVB are free of bond breaks or atomic clashes (97.5% valid), far exceeding ITO's 0.1%, attributable to the encoder-decoder architecture's effective preservation of structural information.
- **Application of Adjoint Matching**: Introducing SOC theory into RL fine-tuning for molecular trajectory generation enables memory-efficient training. This paradigm is generalizable to other diffusion or flow matching models that require guided generation.

## Limitations & Future Work
- Only heavy atoms are considered; hydrogen atoms and solvent effects are not modeled.
- RL fine-tuning requires a known holo state as a reward signal, limiting applicability in truly blind docking scenarios.
- The scale of pretraining data remains limited, particularly for protein–ligand complex structures.
- The temporal resolution of generated trajectories is constrained by the coarsened time step $\tau$.

## Related Work & Insights
- **vs. UniSim**: Both employ pretraining strategies, but UniSim's pretraining only produces an atomic representation model, whereas PVB integrates the generative model into pretraining and achieves objective consistency via the latent space.
- **vs. AlphaFlow**: AlphaFlow generates i.i.d. protein conformation samples and cannot preserve temporal dependencies, thus precluding estimation of dynamical observables.
- **vs. ITO/MDGEN**: Trained from scratch, lacking cross-domain generalization; ITO exhibits extremely low conformational validity.

## Rating
- Novelty: ⭐⭐⭐⭐ The unified encoder-decoder + augmented bridge matching framework is elegantly designed; adjoint matching-based RL fine-tuning is a novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers protein monomers (ATLAS + mdCATH) and protein–ligand complexes (MISATO + PDBBind) with comprehensive evaluation metrics.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are rigorous and the framework is described clearly.
- Value: ⭐⭐⭐⭐ Provides a unified and efficient solution for cross-domain molecular dynamics simulation; the substantial improvement in conformational validity has practical application value.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Discrete Diffusion Trajectory Alignment via Stepwise Decomposition](discrete_diffusion_trajectory_alignment_via_stepwise_decomposition.md)
- [\[NeurIPS 2025\] Unified All-Atom Molecule Generation with Neural Fields](../../NeurIPS2025/medical_imaging/unified_all-atom_molecule_generation_with_neural_fields.md)
- [\[ICLR 2026\] Human Behavior Atlas: Benchmarking Unified Psychological and Social Behavior Understanding](human_behavior_atlas_benchmarking_unified_psychological_and_social_behavior_unde.md)
- [\[NeurIPS 2025\] Fractional Diffusion Bridge Models](../../NeurIPS2025/medical_imaging/fractional_diffusion_bridge_models.md)
- [\[ICLR 2026\] SynCoGen: Synthesizable 3D Molecule Generation via Joint Reaction and Coordinate Modeling](syncogen_synthesizable_3d_molecule_generation_via_joint_reaction_and_coordinate_.md)

<!-- RELATED:END -->
