---
title: >-
  [Paper Note] Atomic Diffusion Models for Small Molecule Structure Elucidation from NMR Spectra
description: >-
  [NeurIPS 2025][Medical Imaging][NMR spectroscopy] This paper proposes ChefNMR, the first end-to-end framework based on 3D atomic diffusion models that directly predicts the molecular structure of unknown small molecules…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "NMR spectroscopy"
  - "molecular structure elucidation"
  - "diffusion models"
  - "natural products"
  - "3D molecule generation"
date: 2026-05-08
content_hash: d62d85e301266d84
---

# Atomic Diffusion Models for Small Molecule Structure Elucidation from NMR Spectra

**Conference**: NeurIPS 2025
**arXiv**: [2512.03127](https://arxiv.org/abs/2512.03127)
**Code**: [GitHub](https://github.com/zhonge/ChefNMR)
**Area**: Medical Imaging
**Keywords**: NMR spectroscopy, molecular structure elucidation, diffusion models, natural products, 3D molecule generation

## TL;DR

This paper proposes ChefNMR, the first end-to-end framework based on 3D atomic diffusion models that directly predicts the molecular structure of unknown small molecules (especially complex natural products) from 1D NMR spectra and molecular formulae alone, achieving state-of-the-art performance on both synthetic and experimental datasets.

## Background & Motivation

**Importance of Natural Products**: Natural products (secondary metabolites) have contributed to more than half of all FDA-approved small-molecule drugs, including landmark compounds such as penicillin and paclitaxel, whose biological functions are intimately tied to their molecular structures.

**NMR as the Core Technology**: Nuclear magnetic resonance (NMR) spectroscopy is the cornerstone technique for small molecule structure elucidation, encoding the local chemical environment and connectivity of atoms through chemical shifts, peak intensities, and J-coupling patterns.

**The Inverse Problem is Extremely Difficult**: Inferring molecular structure from NMR spectra is a highly challenging inverse problem. Conventional approaches rely heavily on expert knowledge and are time-consuming, remaining inefficient even with computational assistance.

**Limitations of Prior Work**: Existing machine learning methods either predict only substructures rather than complete molecules, require richer inputs such as 2D NMR data, or can only handle simple molecules with no more than 101 atoms, and thus cannot scale to complex natural products.

**Rise of Diffusion Models**: Diffusion generative models have demonstrated strong capabilities in tasks such as 3D molecule generation and protein structure prediction, and non-equivariant Transformers offer better scalability than equivariant GNNs.

**Core Motivation**: The paper frames NMR structure elucidation as a conditional 3D atomic diffusion generation problem, while simultaneously constructing a large-scale NMR dataset of natural products to cover complex chemical diversity.

## Method

### Overall Architecture

ChefNMR represents molecule–spectrum pairs as $(\mathbf{A}, \mathbf{X}, \mathcal{S})$, where $\mathbf{A}$ is a one-hot encoding of atom types, $\mathbf{X} \in \mathbb{R}^{N\times3}$ denotes 3D coordinates, and $\mathcal{S}=(\mathbf{s}_H, \mathbf{s}_C)$ denotes the ¹H and ¹³C NMR spectra. The objective is to learn the conditional distribution $p(\mathbf{X}|\mathbf{A}, \mathcal{S})$.

### Three Core Designs

**1. NMR-ConvFormer (Spectral Encoder)**

- **Convolutional Tokenizer**: Two layers of 1D convolution followed by ReLU and max-pooling process the ¹H and ¹³C spectra independently, capturing local features such as peak intensities and splitting patterns while reducing sequence length.
- **Transformer Encoder**: After adding positional and type embeddings to the token sequences, multi-head self-attention and feed-forward networks model both within-spectrum and cross-spectrum global dependencies (e.g., the corresponding signals of the same chemical group across both spectra).
- **Multi-head Attention Pooling (MAP)**: A learnable [CLS] token aggregates information through a final self-attention layer, producing a fixed-size conditioning vector $\mathbf{z}_\mathcal{S}$ via LayerNorm and linear projection.

**2. Conditional 3D Atomic Diffusion Model (DiT)**

- Built on the EDM framework, the model $D_\theta$ predicts clean coordinates $\hat{\mathbf{X}}_0$ from noisy coordinates $\mathbf{X}_\sigma = \mathbf{X}_0 + \mathbf{n}$.
- Input tokens are obtained by concatenating noisy coordinates $\mathbf{X}_\sigma$ with atom types $\mathbf{A}$ and projecting through an MLP.
- The noise level $\sigma$ is embedded via frequency encoding and an MLP, then added to the spectral embedding $\mathbf{z}_\mathcal{S}$ to form the conditioning vector.
- Conditioning information is injected into DiT blocks via adaLN-Zero (adaptive layer normalization).
- Two variants: ChefNMR-S (134M parameters) and ChefNMR-L (462M parameters).

**3. Random Coordinate Augmentation + Classifier-Free Guidance**

- During training, one conformation is randomly sampled from each molecule's conformer ensemble, and random rigid-body transformations (translation + rotation) are applied, encouraging the model to learn SE(3)-invariant representations.
- Classifier-Free Guidance (CFG): During training, the ¹H spectrum, the ¹³C spectrum, or both are dropped with probabilities $p_H=0.1$, $p_C=0.1$, and $p_{both}=0.1$, respectively. At inference, conditional and unconditional predictions are combined via a guidance weight $\omega$.

### Loss & Training

$$\mathcal{L} = \lambda(\sigma)\mathcal{L}_{\text{MSE}}(\hat{\mathbf{X}}_0, \mathbf{X}_0) + \mathcal{L}_{\text{smooth\_lddt}}(\hat{\mathbf{X}}_0, \mathbf{X}_0)$$

- **MSE Loss**: Global structural alignment, $\|\hat{\mathbf{X}}_0 - \mathbf{X}_0\|_2^2$.
- **Smooth LDDT Loss**: Inspired by AlphaFold3, this metric measures local geometric accuracy based on pairwise atomic distance deviations, smoothed via sigmoid functions with thresholds $t_k \in \{0.5, 1.0, 2.0, 4.0 \text{Å}\}$. This loss enforces local chemical validity (e.g., bond lengths), compensating for MSE's insufficient sensitivity to local geometry.

## Key Experimental Results

### Datasets

| Dataset | Type | # Molecules | Atom Range | Notes |
|---------|------|-------------|------------|-------|
| SpectraBase | Synthetic | 141K | [3, 59] | Simple molecules |
| USPTO | Synthetic | 745K | [8, 101] | Molecules from chemical reactions |
| SpectraNP | Synthetic | 111K | [4, 274] | Complex natural products (constructed in this work) |
| SpecTeach | Experimental | 238 | — | Simple molecules, 2 solvents |
| NMRShiftDB2 | Experimental | 23K | — | ¹³C, >7 solvents |

### Main Results on Synthetic Data (Table 2)

| Dataset | Method | Top-1 Acc% | Top-10 Acc% | Top-1 Sim |
|---------|--------|-----------|------------|-----------|
| SpectraBase | Hu et al. | 45.24 | 67.38 | 0.686 |
| SpectraBase | **ChefNMR-L** | **72.04** | **88.20** | **0.833** |
| USPTO | Alberts et al. | 73.38 | 89.98 | N/A |
| USPTO | **ChefNMR-L** | **81.57** | **93.01** | **0.912** |
| SpectraNP | Hu et al. | 19.26 | 39.87 | 0.585 |
| SpectraNP | **ChefNMR-L** | **40.15** | **65.74** | **0.631** |

### Zero-Shot Generalization to Experimental Spectra

- SpecTeach: ChefNMR achieves 56% top-1 accuracy, substantially outperforming Hu et al. and NMR-DiGress.
- NMRShiftDB2: ChefNMR achieves 21% top-1 accuracy, correctly generating structures even from noisy experimental spectra containing solvent peaks.

### Ablation Study

| Ablation | Top-1 Acc% | Impact |
|----------|-----------|--------|
| Full ChefNMR-S | 69.15 | Baseline |
| w/o coordinate augmentation | 49.75 | **−19.4%, most critical component** |
| w/o Smooth LDDT | 68.31 | Minor decrease |
| w/o convolutional tokenizer | 61.78 | −7.4% |
| w/o MAP pooling | 62.97 | −6.2% |
| w/o dropout | 65.48 | −3.7% |

**Key Findings**: (1) Coordinate augmentation contributes most (+20%), confirming that learning SE(3) invariance is critical; (2) 3D diffusion representation comprehensively outperforms chemical language models and graph diffusion models; (3) performance improves with model and data scale, and expanding SpectraNP is expected to further improve natural product elucidation; (4) incorrect predictions still maintain chemical validity and structural similarity.

## Highlights & Insights

1. **Pioneering Work**: The first study to apply 3D atomic diffusion models to NMR structure elucidation, overcoming the representational bottlenecks of chemical language models and graph-based models.
2. **Scalability to Complex Natural Products**: Capable of handling molecules with up to 274 atoms, far exceeding the 101-atom limit of prior methods.
3. **SpectraNP Dataset**: A large-scale synthetic NMR dataset of 111K natural products is constructed, filling a critical data gap in the field.
4. **Generalization to Experimental Spectra**: Trained exclusively on synthetic data, ChefNMR achieves strong zero-shot transfer to real experimental spectra containing noise and solvent peaks.
5. **Well-Supported Design Decisions**: The contributions of coordinate augmentation, the LDDT loss, and individual ConvFormer components are clearly quantified through ablation studies.

## Limitations & Future Work

1. **No Stereochemistry Prediction**: The current method discards stereochemical information and cannot distinguish enantiomers or diastereomers, which is critical for medicinal chemistry.
2. **Reliance on Synthetic Spectra for Training**: Spectra are simulated using MestReNova; a synthetic-to-experimental domain gap remains.
3. **Lack of Confidence Estimation**: The model cannot communicate to chemists how confident it is in a predicted structure, limiting practical deployment.
4. **Only 1D NMR Is Used**: Richer structural information from 2D NMR experiments (e.g., COSY, HSQC) is not exploited.
5. **Low Accuracy on SpectraNP**: Top-1 accuracy on complex natural products is only 40%, still far from practical utility.

## Related Work & Insights

- **NMR Spectrum Prediction (Forward Problem)**: Quantum chemical simulations and ML methods can predict NMR spectra from molecular structures, used here for training data generation.
- **Chemical Language Models**: Hu et al. use a multi-task Transformer to predict SMILES; Alberts et al. generate SMILES from NMR peak lists — both rely on sequence-based representations.
- **Graph Diffusion Models**: Discrete graph diffusion methods such as DiGress generate molecular graphs via Markov chains but scale poorly to large molecules.
- **3D Molecular Diffusion Models**: Equivariant GNNs (EDM, GeoDiff) vs. non-equivariant Transformers (EQGAT-diff, AlphaFold3); the latter offer superior scalability. ChefNMR is the first to introduce DiT for NMR-conditioned molecular generation.

## Rating

- Novelty: ⭐⭐⭐⭐ — First application of 3D atomic diffusion + DiT to NMR structure elucidation; the problem formulation offers a fresh perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple datasets (synthetic + experimental), multiple baselines, and detailed ablations; lacks evaluation by practicing chemists.
- Writing Quality: ⭐⭐⭐⭐ — Background and motivation are clearly articulated; methodology is thoroughly described; figures and tables are of high quality.
- Value: ⭐⭐⭐⭐ — Significant potential to accelerate drug discovery and natural product research, though accuracy on complex molecules still requires improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] One Small Step with Fingerprints, One Giant Leap for De Novo Molecule Generation from Mass Spectra](one_small_step_with_fingerprints_one_giant_leap_for_de_novo_molecule_generation_.md)
- [\[NeurIPS 2025\] Posterior Sampling by Combining Diffusion Models with Annealed Langevin Dynamics](posterior_sampling_by_combining_diffusion_models_with_annealed_langevin_dynamics.md)
- [\[NeurIPS 2025\] Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models](consistent_sampling_and_simulation_molecular_dynamics_with_energy-based_diffusio.md)
- [\[NeurIPS 2025\] Unified All-Atom Molecule Generation with Neural Fields](unified_all-atom_molecule_generation_with_neural_fields.md)
- [\[NeurIPS 2025\] Semantic and Visual Crop-Guided Diffusion Models for Heterogeneous Tissue Synthesis in Histopathology](semantic_and_visual_crop-guided_diffusion_models_for_heterogeneous_tissue_synthe.md)

</div>

<!-- RELATED:END -->
