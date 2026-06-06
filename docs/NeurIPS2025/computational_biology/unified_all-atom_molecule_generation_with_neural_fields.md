---
title: >-
  [Paper Note] Unified All-Atom Molecule Generation with Neural Fields
description: >-
  [NeurIPS 2025][Computational Biology][Neural fields] This paper proposes FuncBind, a framework that represents molecules as continuous atomic density functions via neural fields…
tags:
  - "NeurIPS 2025"
  - "Computational Biology"
  - "Neural fields"
  - "molecule generation"
  - "all-atom representation"
  - "structure-based drug design"
  - "diffusion models"
date: 2026-05-08
content_hash: 2cee232ba156b539
---

# Unified All-Atom Molecule Generation with Neural Fields

**Conference**: NeurIPS 2025
**arXiv**: [2511.15906](https://arxiv.org/abs/2511.15906)  
**Code**: [GitHub](https://github.com/prescient-design/funcbind)  
**Area**: Medical Imaging / Drug Design
**Keywords**: Neural fields, molecule generation, all-atom representation, structure-based drug design, diffusion models

## TL;DR

This paper proposes FuncBind, a framework that represents molecules as continuous atomic density functions via neural fields, constructing a unified conditional generative model capable of target-conditioned generation across three drug modalities: small molecules, macrocyclic peptides, and antibody CDR loops.

## Background & Motivation

- **Background**: Structure-based drug design (SBDD) is a core task in drug discovery, aiming to generate candidate molecules with high affinity given a target protein's 3D structure. Current generative models typically focus on a single molecular modality: small-molecule models use point cloud or voxel representations, while protein models rely on residue-level point clouds and sequence databases.
- **Limitations of Prior Work**: This modality-specific design limits generalization — knowledge cannot be transferred across modalities, yet real-world drug discovery frequently involves multi-modal molecular interface design.
- **Key Challenge**: The lack of a modality-agnostic representation prevents cross-modal learning and limits the diversity of training data from which physical properties can be learned.
- **Goal**: Inspired by the success of cross-modal unified modeling in structure prediction (e.g., AlphaFold3, RoseTTAFold), this work proposes using neural fields as a unified molecular representation — modeling molecules as continuous functions mapping 3D coordinates to atomic densities — enabling joint training across three drug modalities within a single model.

## Method

### Overall Architecture

FuncBind is a two-stage latent-space conditional generative model:
1. **Stage 1**: Train a neural field autoencoder (VAE) to learn latent representations of molecules.
2. **Stage 2**: Train a conditional denoiser in latent space for novel molecule generation.

At inference, given a target protein structure, noise is sampled and iteratively denoised to produce a latent vector, which is decoded into an atomic density field and post-processed to recover the molecular structure.

### Key Designs

1. **Spatial Feature Map Neural Field Representation**: Unlike FuncMol, which uses a global embedding, FuncBind organizes the latent variable $z$ as a spatial feature map ($C \times L^3$), where features at each spatial location capture local information. The encoder $E_\psi$ is a 3D CNN that maps low-resolution molecular voxels into feature map space. The decoder $D_\phi$ is a multiplicative filter network based on Gabor filters, extracting position-dependent embeddings $z_x$ from the feature map via nearest-neighbor interpolation to decode the atomic density field: $D_\phi(x, z_x) \to \mathbb{R}^n$. This design captures local atomic detail while remaining compatible with expressive denoising architectures such as U-Nets. The VAE training loss combines reconstruction loss with KL regularization:

$$\mathcal{L}_{AE} = \sum_{v \in \mathcal{D}} \mathbb{E}_{z \sim q_\psi(z|v)} \left[\int \|D_\phi(x,z) - v(x)\|_2^2 dx\right] + \beta \text{KL}(q_\psi(z|v) \| \mathcal{N}(0,I_d))$$

2. **Conditional Denoiser**: The denoiser receives a noisy latent vector $y = z + \sigma\varepsilon$, conditioned on the target structure $z^{tar}$, molecular modality $c$, and noise level $\sigma$. The target encoder $E_{\psi'}$ shares architectural similarity with the molecular encoder but has independent parameters. The core network $U_\theta$ adopts a 3D U-Net with the Karras preconditioning scheme:

$$\hat{z}_\theta(y|z^{tar}, \sigma, c) = \frac{1}{\sigma^2+1}y + \frac{\sigma}{\sqrt{\sigma^2+1}} U_\theta\left(\frac{y}{\sqrt{\sigma^2+1}}, z^{tar}, \frac{1}{4}\log\sigma, c\right)$$

A key architectural decision is to forgo SE(3) equivariance constraints in favor of data augmentation (rotations and translations).

3. **Sampling Strategies**: Two score-based sampling methods are supported — diffusion models (via reverse SDE integration) and Walk-Jump Sampling (single noise-level sampling, simpler to train and faster to mix). Both leverage the Tweedie–Miyasawa formula to relate the denoiser to the conditional score function.

### Loss & Training

- **Autoencoder**: Reconstruction loss + KL divergence; coordinates near atomic centers are upsampled during training for focused optimization.
- **Denoiser**: MSE denoising loss across multiple noise levels with Karras adaptive reweighting.
- **Post-processing**: Latent codes are rendered into 0.25 Å resolution voxels; peak detection and gradient ascent localize atomic coordinates; OpenBabel infers chemical bonds and residue identities.
- The model comprises **5B parameters** and is jointly trained across all three modalities.

## Key Experimental Results

### Main Results: Small Molecule Generation (CrossDocked2020)

| Metric | FuncBind | VoxBind (SOTA) | MolCraft | TargetDiff |
|--------|----------|----------------|----------|------------|
| VinaScore ↓ | -5.71 | **-6.94** | -6.59 | -5.47 |
| VinaDock ↓ | -7.26 | **-8.30** | -7.92 | -7.80 |
| QED ↑ | 0.50 | **0.57** | 0.50 | 0.48 |
| SA ↑ | 0.65 | 0.70 | **0.69** | 0.58 |
| Diversity ↑ | **0.70** | 0.73 | 0.72 | 0.72 |
| Strain Energy ↓ | 217 | **162** | **195** | 1243 |
| # Atoms | 19.0 | 23.4 | 22.7 | 24.2 |

### Main Results: Antibody CDR Loop Redesign (SAbDab)

| Method | H3-AAR ↑ | H3-RMSD ↓ | H1-AAR ↑ | H1-RMSD ↓ |
|--------|----------|-----------|----------|-----------|
| FuncBind | **47.5%** | **2.04 Å** | **86.9%** | **0.41 Å** |
| AbDiffuser | 34.1% | 3.35 Å | 76.3% | 1.58 Å |
| DiffAb† | 26.8% | 3.60 Å | 65.8% | 1.19 Å |
| RAbD† | 22.1% | 2.90 Å | 22.9% | 2.26 Å |

### Main Results: Macrocyclic Peptide Generation

| Method | TS ↑ | L-RMSD ↓ | I-RMSD ↓ | TM-Score ↑ | Vina dock ↑ |
|--------|------|----------|----------|------------|-------------|
| FuncBind | 0.33 | **2.6 Å** | **1.8 Å** | **0.36** | **41%** |
| AfCycDesign | **0.34** | 7.6 Å | 3.7 Å | 0.33 | 29% |
| RFPeptide | 0.31 | 12 Å | 3.3 Å | 0.33 | 8.8% |

### Ablation Study

| Configuration | Observation |
|---------------|-------------|
| Unified model vs. single-modality model | Comparable performance, but the unified model achieves **significantly higher uniqueness** |
| Diffusion vs. Walk-Jump Sampling | Diffusion outperforms on most metrics; WJS is simpler to train and mixes faster |
| Global embedding vs. spatial feature map | Spatial feature maps scale better to large molecules and are compatible with U-Net architectures |

### Key Findings

- FuncBind surpasses all baselines on CDR loop redesign AAR and RMSD by a margin of 1.5–3×.
- Wet-lab validation: H3 loop redesign achieves a **45% binding rate** on rigid epitopes.
- The model generates novel, chemically valid non-canonical amino acids, with fewer than 1% implausible structures.
- 41% of generated macrocyclic peptide samples achieve Vina docking scores superior to the seed molecule.

## Highlights & Insights

- **Power of unified representation**: Unifying three structurally diverse molecular modalities via neural fields is an elegant design choice; the spatial feature map enables both fine local structural capture and compatibility with established vision network architectures.
- **Data augmentation over equivariance**: Replacing SE(3) equivariance constraints with rotation/translation augmentation simplifies the architecture without sacrificing performance.
- **Closed-loop wet-lab validation**: The full pipeline from generation to experimental validation is demonstrated for CDR design, with a 45% binding rate indicating genuine translational value.
- **New benchmark contribution**: A dataset of ~190K synthetic macrocyclic peptide–protein complexes is introduced.

## Limitations & Future Work

- Small-molecule metrics remain slightly below those of specialized models such as VoxBind and MolCraft.
- The approach relies on accurate 3D structures of molecular interfaces, which are costly to obtain in practical drug discovery settings.
- Practical constraints such as synthesizability (small molecules) and developability (antibodies) are not addressed.
- The 5B-parameter model incurs substantial training and inference costs, posing a high deployment barrier.
- Cross-modal transfer learning has not been thoroughly explored.

## Related Work & Insights

- **FuncMol**: The direct predecessor; this work extends it by introducing spatial feature maps and conditional generation.
- **VoxBind**: The SOTA voxel-based method; FuncBind can be viewed as its continuous counterpart.
- **AlphaFold3/RoseTTAFold**: Successful precedents for cross-modal unified structure prediction.
- **Insight**: Neural field representations hold considerable potential for molecular science and could be extended to a broader range of biological systems.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The neural field approach unifying three molecular modalities is novel, though the core techniques (VAE + diffusion) are well-established.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across three modalities, wet-lab validation, and a new dataset contribution.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with complete technical descriptions.
- **Value**: ⭐⭐⭐⭐⭐ A unified molecular generation framework of significant practical importance; wet-lab results substantially strengthen the claims.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number](prior-guided_flow_matching_for_target-aware_molecule_design_with_learnable_atom_.md)
- [\[NeurIPS 2025\] Atomic Diffusion Models for Small Molecule Structure Elucidation from NMR Spectra](atomic_diffusion_models_for_small_molecule_structure_elucidation_from_nmr_spectr.md)
- [\[ICLR 2026\] Unified Biomolecular Trajectory Generation via Pretrained Variational Bridge](../../ICLR2026/computational_biology/unified_biomolecular_trajectory_generation_via_pretrained_variational_bridge.md)
- [\[NeurIPS 2025\] Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling](towards_unified_and_lossless_latent_space_for_3d_molecular_latent_diffusion_mode.md)
- [\[NeurIPS 2025\] Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings](generative_modeling_of_full-atom_protein_conformations_using_latent_diffusion_on.md)

</div>

<!-- RELATED:END -->
