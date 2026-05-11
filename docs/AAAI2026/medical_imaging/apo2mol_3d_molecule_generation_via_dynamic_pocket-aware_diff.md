---
title: >-
  [Paper Note] Apo2Mol: 3D Molecule Generation via Dynamic Pocket-Aware Diffusion Models
description: >-
  [AAAI 2026][Medical Imaging][Structure-based drug design] This paper proposes Apo2Mol, a diffusion-based all-atom framework that simultaneously generates 3D ligand molecules and corresponding holo (bound-state) pocket co…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Structure-based drug design"
  - "apo-holo conformational change"
  - "protein pocket dynamics"
  - "3D molecule generation"
  - "SE(3) equivariance"
date: 2026-05-08
content_hash: 55c966c93a39e635
---

# Apo2Mol: 3D Molecule Generation via Dynamic Pocket-Aware Diffusion Models

**Conference**: AAAI 2026
**arXiv**: [2511.14559v1](https://arxiv.org/abs/2511.14559v1)
**Code**: [https://github.com/AIDD-LiLab/Apo2Mol](https://github.com/AIDD-LiLab/Apo2Mol)
**Area**: AI for Science / Drug Design / Diffusion Models
**Keywords**: Structure-based drug design, apo-holo conformational change, protein pocket dynamics, 3D molecule generation, SE(3) equivariance

## TL;DR
This paper proposes Apo2Mol, a diffusion-based all-atom framework that simultaneously generates 3D ligand molecules and corresponding holo (bound-state) pocket conformations from protein apo (unbound) conformations. Trained on 24K experimentally resolved apo-holo structure pairs, it achieves state-of-the-art performance in binding affinity (Vina min −7.86) and drug-likeness.

## Background & Motivation

**Background**: Existing deep generative models for structure-based drug design (SBDD), such as TargetDiff and DecompDiff, assume a rigid protein pocket and train/generate directly on holo conformations. However, proteins are inherently dynamic—ligand binding induces conformational rearrangements in the binding pocket. When only apo conformations are available (e.g., novel targets without co-crystal structures), the generation quality of these methods degrades substantially.

**Limitations of Prior Work**: DynamicFlow attempts to model pocket dynamics using molecular dynamics (MD) simulation trajectories, but MD simulations are computationally expensive, constrained by force field parameterization, and may introduce simulation-specific artifacts.

**Key Challenge**: The rigid-pocket assumption in existing SBDD models fails to capture the conformational flexibility that is fundamental to ligand–protein recognition.

**Goal**: To enable simultaneous generation of high-affinity ligands and physically plausible holo pocket conformations from apo structures alone, without relying on MD simulation data.

## Core Problem
**How can one simultaneously generate high-affinity ligands and plausible holo pocket conformations given only an apo protein conformation, without dependence on MD simulation data?**

## Method

### Overall Architecture
Apo2Mol consists of: data preparation (apo-holo alignment + interpolation) + SE(3)-equivariant hierarchical graph diffusion model.
- **Forward diffusion**: Ligand coordinates are corrupted with noise; pocket conformation is linearly interpolated from holo toward apo.
- **Reverse diffusion**: Starting from the apo pocket and a noisy ligand, the model jointly denoises the ligand and transforms the pocket from apo to holo.

Learning objective: $p(\mathcal{P}^H, \mathcal{M} | \mathcal{P}^A)$

### Key Designs

1. **Experimentally resolved apo-holo pairs**: The training data consists of 24,601 experimentally resolved apo-holo-ligand triplets filtered from the PLINDER database, with 100% sequence identity and resolution ≤ 2.5 Å. No MD simulation data is used, avoiding simulation artifacts. Train/validation/test splits are divided by time.

2. **Residue-level conformational interpolation**: Pocket conformational changes are modeled as residue-level translations $\mathbf{tr}$, rotations $\mathbf{q}$ (quaternions), and chi-angle updates $\boldsymbol{\mathcal{X}}$. During the forward process, translations and chi angles undergo linear interpolation with Gaussian noise, while rotations are interpolated using spherical linear interpolation (Slerp). This preserves protein structural integrity.

3. **Hierarchical graph message passing**: A protein–ligand complex graph is constructed with four edge types: intra-ligand, ligand–residue, intra-residue, and inter-residue. SE(3)-equivariant attention layers jointly update atomic coordinates and chemical features. Residue-level predictions are aggregated from atom-level representations via SAGPooling.

### Loss & Training
Five loss terms: ligand position MSE + ligand atom type KL divergence + pocket translation MSE + pocket rotation L1 loss with norm regularization + chi-angle cosine loss. Adam optimizer with learning rate 5e-4 and plateau scheduling. Trained on 4×A100-80G GPUs with batch size 8, converging in approximately 150 epochs.

## Key Experimental Results

**Ligand generation from apo structures (Table 1):**

| Method | Vina min (Avg)↓ | Vina min (Med)↓ | QED (Avg)↑ | High Affinity↑ |
|------|----------------|----------------|-----------|---------------|
| IPDiff | -6.40 | -6.56 | 0.51 | 29.6% |
| DecompDiff | -6.37 | -6.40 | 0.56 | 34.3% |
| **Apo2Mol** | **-6.79** | **-7.09** | **0.59** | **42.7%** |

**Comparison with holo-trained baselines (Table 2; Apo2Mol still conditions on apo):**

| Method | Vina min (Avg) | Vina min (Med) | High Affinity |
|------|---------------|---------------|--------------|
| IPDiff (holo) | -7.09 | -7.08 | 44.9% |
| **Apo2Mol (apo→holo)** | **-7.86** | **-8.03** | **52.9%** |

### Ablation Study
- **Hierarchical graph vs. single edge type**: Removing the hierarchical graph degrades Vina min from −6.79 to −6.18 and QED from 0.587 to 0.524.
- **Quaternion vs. rotation vector**: Replacing quaternions degrades Vina min from −6.79 to −6.51, confirming the numerical stability and smooth interpolation advantages of quaternion representation.
- **Molecular structural validity**: C–C bond distance distribution JSD: Apo2Mol 0.178 vs. IPDiff 0.216 vs. TargetDiff 0.273.
- **Pocket generation**: The RMSD distribution of generated pockets achieves JSD = 0.317 relative to experimental holo distributions—room for improvement remains, but overall trends are physically reasonable.
- **Validity/novelty of generated molecules**: Validity 88.9%, novelty 95.3% (vs. IPDiff 87.6%, 91.1%).

## Highlights & Insights
- **Precise problem formulation**: Incorporating the apo→holo conformational transition into the generative framework represents a fundamental improvement to SBDD, reflecting realistic drug discovery scenarios.
- **Data-driven over simulation-driven**: Replacing MD simulation data with 24K experimentally resolved structures avoids force field bias.
- **Residue-level conformational modeling**: Rather than directly predicting atomic coordinates, the model predicts rigid-body transformations and chi angles per residue, preserving the physical plausibility of protein structure.
- **Quaternion rotation representation**: Avoids singularities inherent to Euler angles and rotation vectors; Slerp provides smooth interpolation on the rotation manifold.

## Limitations & Future Work
- **Distributional gap in pocket generation**: JSD = 0.317 indicates a non-trivial gap between generated and true holo pockets; large-scale protein structure pretraining may help close this gap.
- **Neglect of water molecules and ions**: Water molecules frequently participate in hydrogen-bond networks at binding sites; omitting them may affect binding affinity prediction accuracy.
- **Static evaluation protocol**: Binding affinity is assessed via Vina scoring rather than free energy perturbation (FEP) or experimental validation.
- **Dataset bias**: PDB-derived experimental structures are biased toward crystallizable proteins and known drug targets.
- **Training cost**: Requires 4×A100-80G GPUs for approximately 150 epochs.

## Related Work & Insights

| Method | Pocket Assumption | Data Source | Key Difference from Apo2Mol |
|------|---------|---------|-------------------|
| **TargetDiff/DecompDiff/IPDiff** | Rigid holo | Experimental holo | Ignores pocket dynamics; performance degrades under apo conditions |
| **DynamicFlow** | Dynamic (MD) | MD simulation trajectories | Relies on simulation data, potentially introducing artifacts; Apo2Mol uses experimental data |
| **Pocket2Mol** | Rigid holo | Experimental holo | Autoregressive generation; does not model pocket conformational change |

**Broader Insights**:
- The apo→holo conformational modeling paradigm can be transferred to conformational selection problems in protein–protein docking (PPD).
- The hierarchical graph (atom→residue) message-passing design has broad reference value for protein-related tasks.
- The data curation strategy is instructive: high-quality filtering from large-scale databases such as PLINDER offers superior cost-effectiveness compared to running custom simulations.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Integrating pocket dynamics into the diffusion framework is a substantive innovation; the data strategy is also novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across two settings (apo vs. holo baselines), ablation analysis, and structural analysis of both molecules and pockets.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is clearly articulated; methodological derivations are rigorous.
- **Value**: ⭐⭐⭐⭐ Significant practical value for the drug design community, particularly in novel target drug discovery scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SynCoGen: Synthesizable 3D Molecule Generation via Joint Reaction and Coordinate Modeling](../../ICLR2026/medical_imaging/syncogen_synthesizable_3d_molecule_generation_via_joint_reaction_and_coordinate_.md)
- [\[AAAI 2026\] Distributional Priors Guided Diffusion for Generating 3D Molecules in Low Data Regimes](distributional_priors_guided_diffusion_for_generating_3d_molecules_in_low_data_r.md)
- [\[NeurIPS 2025\] Atomic Diffusion Models for Small Molecule Structure Elucidation from NMR Spectra](../../NeurIPS2025/medical_imaging/atomic_diffusion_models_for_small_molecule_structure_elucidation_from_nmr_spectr.md)
- [\[AAAI 2026\] A Disease-Aware Dual-Stage Framework for Chest X-ray Report Generation](a_disease-aware_dual-stage_framework_for_chest_x-ray_report_.md)
- [\[NeurIPS 2025\] Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number](../../NeurIPS2025/medical_imaging/prior-guided_flow_matching_for_target-aware_molecule_design_with_learnable_atom_.md)

</div>

<!-- RELATED:END -->
