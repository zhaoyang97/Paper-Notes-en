---
title: >-
  [Paper Note] UniMoMo: Unified Generative Modeling of 3D Molecules for De Novo Binder Design
description: >-
  [ICML2025][Computational Biology][Unified Molecule Generation] UniMoMo is proposed as the first unified 3D binder design framework for three types of molecules: small molecules, peptides, and antibodies. It uses "Graph of Blocks" as a unified representation, an iterative all-atom autoencoder for compressing the latent space, and an E(3)-equivariant diffusion model for generation, outperforming domain-specific models across three benchmarks.
tags:
  - "ICML2025"
  - "Computational Biology"
  - "Unified Molecule Generation"
  - "Small Molecules"
  - "Peptides"
  - "Antibodies"
  - "Geometric Latent Space Diffusion"
  - "Autoencoders"
  - "Block-based Representation"
date: 2026-05-08
content_hash: a93745c1bef65baf
---

# UniMoMo: Unified Generative Modeling of 3D Molecules for De Novo Binder Design

**Conference**: ICML2025  
**arXiv**: [2503.19300](https://arxiv.org/abs/2503.19300)  
**Code**: None  
**Area**: Biomolecular Design / Drug Discovery  
**Keywords**: Unified Molecule Generation, Small Molecules, Peptides, Antibodies, Geometric Latent Space Diffusion, Autoencoders, Block-based Representation

## TL;DR

UniMoMo is proposed as the first unified 3D binder design framework for three types of molecules: small molecules, peptides, and antibodies. It uses "Graph of Blocks" as a unified representation, an iterative all-atom autoencoder for compressing the latent space, and an E(3)-equivariant diffusion model for generation, outperforming domain-specific models across three benchmarks.

## Background & Motivation

### Background

**Background**: The three classes of molecules have distinct advantages: small molecules are suitable for oral administration (good absorption), peptides excel at intracellular targeting (penetration capability), and antibodies offer high specificity for treating major diseases.

### Limitations of Prior Work

**Limitations of Prior Work**: Fragmented nature of existing methods: baseline generative models are specialized for each separate domain, preventing data and knowledge sharing across domains.

### Key Challenge

**Key Challenge**: Challenges in unified representation: small molecules are combinations of functional groups, whereas peptides/antibodies are linear arrangements of amino acids. Pure atom-level graphs ignore hierarchical priors and are computationally expensive.

### Key Insight

**Key Insight**: Cross-domain transferability: physicochemical rules of molecule-protein interactions are universal. Unified modeling can exploit larger and more diverse datasets.

## Method

### Unified Representation: Graph of Blocks

- Each block corresponds to a standard amino acid or a molecular fragment (extracted by the Principal Subgraph algorithm).
- It preserves all-atom geometry and hierarchical structures.
- A binary prompt controls the generation type (1 = amino acids only, 0 = molecular fragments allowed).
- The vocabulary $\mathbb{V}$ contains 20 standard amino acids + extracted common molecular fragments (e.g., benzene ring, indole, etc.).
- The molecular graph is defined as $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where $\mathcal{V}$ contains the atom types and coordinates of each block, and $\mathcal{E}$ contains intra-block and inter-block chemical bonds.

### Iterative All-Atom Autoencoder (VAE)

- **Encoder**: Compress each block into a latent representation $(\bm{z}_i \in \mathbb{R}^8, \vec{\bm{z}}_i \in \mathbb{R}^3)$ with KL regularization.
- **Two-Stage Decoder**:
    - Step 1: Predict the block type $s_i \in \mathbb{V}$ and retrieve atom types and intra-block bonds via lookup.
    - Step 2: Iterative structural module, resembling a lightweight Flow Matching process, approximating real atom coordinates in 10 steps initialized by Gaussian sampling.
- **Inter-block Chemical Bond Prediction**: Predicts bond types based on spatially proximal (3.5Å) atom pairs.

### Geometric Latent Space Diffusion Model

- DDPM operates in the latent space $[\bm{z}_i, \vec{\bm{z}}_i]$, avoiding costly iterations on all atoms.
- The denoising network utilizes an Equivariant Transformer (GeoTF) to guarantee E(3) equivariance.
- Training Objective:
  $$\mathcal{L}_{\text{LDM}} = \mathbb{E}_{t}\left[\frac{\sum_i \|\epsilon_i - \epsilon_\theta(\mathcal{Z}_x^t, \mathcal{Z}_y, t)[i]\|^2}{|\mathcal{Z}_x^t|}\right]$$

## Key Experimental Results

### Peptide Design

| Model | AAR↑ | C-RMSD↓ | L-RMSD↓ | ΔG↓ | IMP↑ |
|---|---|---|---|---|---|
| RFDiffusion | 34.68% | 4.69 | 1.88 | -13.47 | 5.38% |
| PepFlow | 35.47% | 2.87 | 1.79 | -21.71 | 15.22% |
| PepGLAD | **38.62%** | 2.74 | 1.60 | -23.12 | 18.28% |
| **UniMoMo (single)** | 37.59% | **2.48** | **1.48** | **-28.72** | - |

### Antibodies and Small Molecules

- Antibody CDR-H3 design: UniMoMo achieves competitive or superior performance compared to MEAN/DiffAb in both AAR and RMSD.
- Small-molecule SBDD: UniMoMo is competitive with TargetDiff and DecompDiff on metrics such as Vina Score, QED, and SA.

### Ablation Study

- **Joint multi-domain training vs. single-domain training**: The unified model outperforms single-domain models on most metrics, validating cross-domain transferability.
- **Hybrid molecular type generation**: Capable of generating different types of binders for the same target.

## Highlights & Insights

1. **First unified molecular generation framework**: Bridges the representation gap between small molecules and biomacromolecules using the Graph of Blocks representation.
2. **Efficiency of latent space diffusion**: Diffusion in block-level latent space results in substantially lower complexity than all-atom diffusion.
3. **Iterative decoder**: A lightweight flow-matching-style decoder maps latent features to concrete atomic coordinates, balancing accuracy and efficiency.
4. **Empirical proof of cross-domain transfer**: Joint training tangibly improves performance across all domains.
5. **Training techniques**: Adding random noise to the encoder's output coordinates enhances decoder robustness; teacher forcing uses ground-truth (GT) inter-block chemical bonds with a 50% probability.
6. **Distance loss design**: Pairwise atomic distance constraints are only applied when $t < 0.25$, avoiding ineffective constraints during the chaotic early diffusion stages.
7. **Two-round refinement during inference**: An additional encode-decode cycle is performed post-generation, leveraging predicted inter-block chemical bonds to further optimize structural details.

### Training Data Scale

- Peptides: ~8K peptide-protein complexes
- Antibodies: ~5K antigen-antibody structures
- Small molecules: CrossDocked2020 dataset
- Unified training utilizes around 13K+ cross-domain samples

## Limitations & Future Work

- Unvalidated by wet-lab experiments, relying solely on computational metrics (e.g., Vina Score).
- The size and quality of the fragment vocabulary restrict generation diversity; suboptimal decomposition may prevent the generation of certain structures.
- Handles only single-chain binders, not yet extended to multi-chain complexes or covalent modifications.
- Potential inconsistencies between inter-block bond prediction and output coordinates in the decoder.
- The choice of latent space dimension $d=8$ is not fully ablated.
- The effectiveness on highly flexible molecules (e.g., long-chain small molecules) remains unknown.

## Related Work & Insights

- **PepGLAD (Kong et al., 2024b)**: Used as the baseline for peptide design in this work, which also utilizes geometric latent diffusion.
- **GET (Kong et al., 2024a)**: A pioneer in unified biomolecular representation, though limited to interaction prediction.
- **AlphaFold 3**: A milestone in structure prediction across diverse molecular types, pioneering the unified modeling paradigm.
- **Inspiration**: The paradigm of unified representation coupled with cross-domain training can be extended to broader molecular types, including nucleic acids and glycans.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](../../NeurIPS2025/computational_biology/uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)
- [\[NeurIPS 2025\] Pharmacophore-Guided Generative Design of Novel Drug-Like Molecules](../../NeurIPS2025/computational_biology/pharmacophore-guided_generative_design_of_novel_drug-like_molecules.md)
- [\[NeurIPS 2025\] Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling](../../NeurIPS2025/computational_biology/towards_unified_and_lossless_latent_space_for_3d_molecular_latent_diffusion_mode.md)
- [\[NeurIPS 2025\] De novo generation of functional terpene synthases using TpsGPT](../../NeurIPS2025/computational_biology/de_novo_generation_of_functional_terpene_synthases_using_tpsgpt.md)
- [\[ICML 2025\] Geometric Generative Modeling with Noise-Conditioned Graph Networks](geometric_generative_modeling_with_noise-conditioned_graph_networks.md)

</div>

<!-- RELATED:END -->
