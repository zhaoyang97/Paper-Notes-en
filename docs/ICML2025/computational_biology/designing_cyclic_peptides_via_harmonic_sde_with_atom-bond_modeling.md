---
title: >-
  [Paper Note] Designing Cyclic Peptides via Harmonic SDE with Atom-Bond Modeling
description: >-
  [ICML2025][Computational Biology][Cyclic peptide design] The authors propose the CpSDE framework, which uses alternate sampling between a harmonic SDE generative model (AtomSDE) and a residue-type predictor (ResRouter) to achieve the first all-type cyclic peptide design based on 3D receptor structures, surpassing existing linear peptide design methods in both stability and affinity.
tags:
  - "ICML2025"
  - "Computational Biology"
  - "Cyclic peptide design"
  - "Harmonic SDE"
  - "All-atom modeling"
  - "Chemical bond modeling"
  - "Diffusion models"
  - "Drug discovery"
date: 2026-05-08
content_hash: eba2ae2dcbc7b8d9
---

# Designing Cyclic Peptides via Harmonic SDE with Atom-Bond Modeling

**Conference**: ICML2025  
**arXiv**: [2505.21452](https://arxiv.org/abs/2505.21452)  
**Code**: To be confirmed  
**Area**: Molecular Design  
**Keywords**: Cyclic peptide design, Harmonic SDE, All-atom modeling, Chemical bond modeling, Diffusion models, Drug discovery

## TL;DR

The authors propose the CpSDE framework, which uses alternate sampling between a harmonic SDE generative model (AtomSDE) and a residue-type predictor (ResRouter) to achieve the first all-type cyclic peptide design based on 3D receptor structures, surpassing existing linear peptide design methods in both stability and affinity.

## Background & Motivation

- **Limitations of Linear Peptides**: Traditional linear peptide drugs suffer from short half-lives, poor stability, and susceptibility to enzymatic degradation, limiting their therapeutic potential.
- **Advantages of Cyclic Peptides**: Cyclic peptides form closed loops through head-to-tail or side-chain connections, enhancing enzymatic stability and binding to protein surfaces with high affinity in more stable conformations.
- **Diverse Types of Cyclic Peptides**: Based on cyclization atoms, they are classified into four categories: head-to-tail, side-to-tail, head-to-side, and side-to-side.
- **Limitations of Prior Work**: Existing methods support only a single type of cyclic peptide (such as disulfide-bonded or head-to-tail cyclic peptides) and cannot unifiedly handle different cyclization constraints; furthermore, the 3D structure data of protein-cyclic peptide complexes is extremely scarce.
- **Key Challenge**: Under data scarcity, it is necessary to simultaneously address cyclization geometric constraints, non-standard amino acid modeling, and joint sequence-structure generation.

## Method

### Overall Architecture

CpSDE consists of two core models and a sampling algorithm:

1. **AtomSDE**: A generative structure prediction model based on harmonic SDE, learning the conditional distribution of ligand atom coordinates.
2. **ResRouter**: A residue type predictor that predicts amino acid types based on the denoised structures.
3. **Routed Sampling**: Alternately invokes the above two models to iteratively update sequences and structures.

### AtomSDE—Harmonic SDE Structure Generation

The Variance Preserving (VP) SDE is adopted (rather than VE SDE, which causes noisy ligands to drift away from receptors and lose effective interactions). A harmonic SDE related to the chemical graph is introduced:

$$
\mathrm{d}\mathbf{x}^L = -\frac{1}{2}\beta(t)\tilde{\mathbf{x}}^L \mathrm{d}t + \sqrt{\beta(t)} \mathbf{\Lambda}^{1/2} \mathbf{P}^\top \mathrm{d}\mathbf{w}
$$

where $\mathbf{H} = \mathbf{L} + \sigma_P^{-2}\mathbf{I}$ is a positive-definite matrix combining the chemical graph Laplacian $\mathbf{L}$ with a receptor-related scalar, and $\mathbf{H} = \mathbf{P}\mathbf{\Lambda}\mathbf{P}^\top$ represents its eigendecomposition. This anisotropic perturbation process leverages the connectivity information from the chemical graph to keep the initial positions of bonded atoms close and perturbs them with correlated noise.

The perturbation kernel has an analytical form:

$$
p_{0t}(\mathbf{x}^L | \mathbf{x}_0^L) = \mathcal{N}\left(\mathbf{x}_t^L;\; \mathbf{x}_0^L e^{-\frac{1}{2}\int_0^t \beta(s)\mathrm{d}s},\; \mathbf{H} - \mathbf{H}e^{-\int_0^t \beta(s)\mathrm{d}s}\right)
$$

**Loss & Training**—A simple reconstruction loss (approximately equivalent to score matching):

$$
\mathcal{L}_{\text{AtomSDE}} = \mathbb{E}_{t, p_0(\mathbf{x}_0^L), p_{0t}(\mathbf{x}_t^L|\mathbf{x}_0^L)} \left[\| \mathbf{D}_\theta(\mathbf{x}_t^L, t) - \mathbf{x}_0^L \|^2 \right]
$$

The model is based on an SE(3)-equivariant neural network, simultaneously encoding the protein-ligand KNN graph and the ligand chemical graph.

### ResRouter—Residue Type Prediction

Resolving the "chicken-and-egg" issue of sequence and structure: predicting residue types given a noisy structure. To prevent the model from taking shortcuts, standard amino acid side chains are removed during input, and the hidden states of backbone atoms (N-Cα-C-O) are aggregated to predict amino acid types using an MLP:

$$
\mathcal{L}_{\text{ResRouter}} = \sum_{i}^{N} -\log p_\phi(a_i | \mathbf{D}_\theta(\mathbf{x}_t^L, t), \mathcal{G}_C, \mathcal{T}, t)
$$

The parameters of AtomSDE are frozen after pre-training, and then ResRouter is trained.

### Routed Sampling Strategy

- Atoms are categorized into two types: **cyclization-constrained atoms** (backbone atoms + cyclization-related atoms, with a known chemical graph) and **free residue atoms** (non-cyclization standard residue side chains, with an unknown chemical graph).
- For the free residue part, an Atom73 state ("superposition state") is maintained, where all possible amino acids share the backbone and Cβ atoms, and each has its own independent side-chain atoms.
- In each reverse SDE step: AtomSDE denoises coordinates → ResRouter predicts residue types → fold into concrete atom states according to predicted types → update the chemical graph → next step.
- All atoms are aligned to the same noise level to prevent insufficient updates caused by intermittent sampling of side-chain atoms.

## Key Experimental Results

### Dataset

- Small molecule dataset: From PDBBind, consisting of 14,348 protein-ligand complexes.
- Peptide dataset: From RCSB PDB / Propedia / PepBDB, consisting of 20,033 complexes (ligand < 30 residues).
- Train/validation sets were split by clustering using a receptor sequence homology threshold of 0.3.

### Main Results

| Method | Co-design | Peptide Type | Stability Avg↓ | Stability Med↓ | Affinity Avg↓ | Affinity Med↓ | Diversity↑ |
|------|--------|--------|------------|------------|------------|------------|---------|
| Reference | N/A | Linear | -672.53 | -634.71 | -85.03 | -78.70 | N/A |
| RFDiffusion | ✗ | Linear | **-633.51** | **-607.82** | **-70.30** | **-61.35** | 0.55 |
| ProteinGenerator | ✓ | Linear | -576.39 | -554.70 | -46.98 | -40.39 | 0.58 |
| PepFlow | ✓ | Linear | -576.16 | -498.31 | -47.88 | -42.40 | 0.70 |
| PepGLAD | ✓ | Linear | -359.44 | -310.33 | -45.06 | -38.56 | 0.79 |
| **CpSDE (Mix)** | ✓ | Mixed Cyclic | **-580.67** | **-527.80** | **-55.71** | **-48.42** | **0.79** |

- Among all co-design methods, CpSDE performs best in both stability and affinity, while also achieving the highest diversity.
- Head-to-tail and head-to-side cyclic peptides perform better than other types, potentially because there are more C-N bonds than S-S/C-S bonds in the training data.
- RFDiffusion achieves the optimal energy but exhibits low diversity (tending to generate α-helices).

### Case Study—Molecular Dynamics Validation

**SMYD2 Inhibitors (head-to-tail cyclization)**:

- Eight cyclic peptides were generated, with H2T-6 achieving the best Rosetta affinity (-33.9 kcal/mol).
- 100 ns MD simulation: H2T-6 RMSD = 3.05 Å (compared to 4.59 Å for the PepFlow linear peptide).
- MM-PBSA binding free energy: H2T-6 = **-24.02** kcal/mol, while the ground-truth linear peptide is -19.00 and PepFlow is -7.26.

**SET8 Inhibitors (side-to-side cyclization)**:

- S2S-4 RMSD = 2.54 Å (compared to 4.06 Å for ground truth and 5.23 Å for PepFlow).
- Binding free energy: S2S-4 = **-12.48** kcal/mol, while ground truth is -6.39 and PepFlow is -9.26.

## Highlights & Insights

1. **First All-Type Cyclic Peptide Generation Framework**: Unifiedly handles four cyclization types without requiring model modifications for specific types.
2. **All-Atom and Chemical Bond Modeling**: Avoids the limitations of residue-level representations, naturally supporting non-standard amino acids and cyclization bond constraints.
3. **Clever Utilization of Chemical Graphs via Harmonic SDE**: Anisotropic noise perturbation maintains the spatial correlation of bonded atoms, yielding higher sampling quality.
4. **High Data Efficiency**: Jointly training AtomSDE with small molecule and linear peptide data significantly mitigates the scarcity of cyclic peptide 3D data.
5. **Thorough MD Validation**: In case studies of two actual drug targets, the designed cyclic peptides outperform the ground-truth linear peptides in both stability and affinity.

## Limitations & Future Work

- Currently, evaluations rely solely on Rosetta energy and short-term MD simulations, lacking wet-lab validation.
- There are few samples of certain cyclization types (such as S-S and C-S bonds) in the training data, leading to relatively weaker performance on corresponding cyclic peptide designs.
- It does not support cyclization designs containing completely unnatural building blocks (e.g., D-amino acids, β-amino acids).
- The "superposition state" introduced by the Atom73 representation increases memory and computational overhead.
- Comparisons with recent structure prediction methods, such as AlphaFold3, are not yet provided.

## Related Work & Insights

- **Protpardelle** (Chu et al., 2024): The inspiration source for the Atom73 superposition state representation.
- **RFDiffusion / ProteinGenerator**: Diffusion models for protein backbone design.
- **PepFlow / PepGLAD**: All-atom design benchmarks for linear peptides.
- **EigenFold** (Jing et al., 2023): The application of harmonic priors in protein conformation sampling.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The first unified all-type cyclic peptide generation framework; the combination of harmonic SDE and Routed Sampling is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Compared against multiple benchmarks and validated with MD case studies; however, it lacks wet-lab experiments.
- Writing Quality: ⭐⭐⭐⭐ — Highly structured, with intuitive schematics of cyclic peptide classifications and complete mathematical derivations.
- Value: ⭐⭐⭐⭐⭐ — Offers significant potential to advance peptide drug discovery.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Pallatom-Ligand: an All-Atom Diffusion Model for Designing Ligand-Binding Proteins](../../ICLR2026/computational_biology/pallatom-ligand_an_all-atom_diffusion_model_for_designing_ligand-binding_protein.md)
- [\[ICML 2025\] Geometric Generative Modeling with Noise-Conditioned Graph Networks](geometric_generative_modeling_with_noise-conditioned_graph_networks.md)
- [\[NeurIPS 2025\] Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings](../../NeurIPS2025/computational_biology/generative_modeling_of_full-atom_protein_conformations_using_latent_diffusion_on.md)
- [\[ICML 2025\] PepTune: De Novo Generation of Therapeutic Peptides with Multi-Objective-Guided Discrete Diffusion](peptune_de_novo_generation_of_therapeutic_peptides_with_multi-objective-guided_d.md)
- [\[NeurIPS 2025\] Unified All-Atom Molecule Generation with Neural Fields](../../NeurIPS2025/computational_biology/unified_all-atom_molecule_generation_with_neural_fields.md)

</div>

<!-- RELATED:END -->
