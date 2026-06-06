---
title: >-
  [Paper Note] Distributional Priors Guided Diffusion for Generating 3D Molecules in Low Data Regimes
description: >-
  [AAAI 2026][Computational Biology][3D molecule generation] This paper proposes GODD (Geometric OOD Diffusion Model), which captures distributional structural priors via an equivariant asymmetric autoencoder to guide the…
tags:
  - "AAAI 2026"
  - "Computational Biology"
  - "3D molecule generation"
  - "diffusion models"
  - "out-of-distribution generalization"
  - "structural priors"
  - "drug design"
date: 2026-05-08
content_hash: 75ebdd20139847d9
---

# Distributional Priors Guided Diffusion for Generating 3D Molecules in Low Data Regimes

**Conference**: AAAI 2026
**arXiv**: [2404.00962](https://arxiv.org/abs/2404.00962)  
**Code**: N/A  
**Area**: Medical Imaging / Molecule Generation
**Keywords**: 3D molecule generation, diffusion models, out-of-distribution generalization, structural priors, drug design

## TL;DR

This paper proposes GODD (Geometric OOD Diffusion Model), which captures distributional structural priors via an equivariant asymmetric autoencoder to guide the generation process of a diffusion model, enabling models trained on data-rich molecular distributions to generalize to data-scarce distributions, achieving a 12.6% improvement in success rate on OOD structural shift benchmarks.

## Background & Motivation

**Background**: Deep learning-based 3D molecule generation is a frontier direction in drug discovery. Diffusion models (e.g., EDM, GeoDiff) have demonstrated strong capabilities in 3D molecule generation, producing molecular geometries that conform to chemical rules.

**Limitations of Prior Work**: (1) 3D molecule generation models typically assume that training and test data share the same distribution; however, in drug design, target molecules often differ significantly from the training distribution—for instance, requiring the generation of molecules with novel scaffolds or functional groups. (2) This out-of-distribution (OOD) generation problem is particularly acute in molecular design, as valuable novel drug molecules are, almost by definition, "previously unseen." (3) Existing OOD molecule generation research has focused primarily on property shifts (e.g., generating molecules with higher activity) while neglecting structural shifts (e.g., generating molecules with different scaffolds), the latter being a more fundamental and challenging problem.

**Key Challenge**: Data-scarce regions are precisely where generative models are most needed for exploration, yet models cannot effectively generate in those regions due to the lack of training data—a chicken-and-egg dilemma.

**Goal**: (1) Train a generative model on data-rich molecular distributions. (2) Enable the model to generalize to data-scarce molecular distributions. (3) Handle structural-level distribution shifts (rather than property shifts alone).

**Key Insight**: The authors observe that although different molecular distributions vary greatly in their specific structures, their distributional priors can be encoded as structural-grained representations. By learning prior representations that distinguish different distributions, a diffusion model can be guided to generate toward a target distribution even when that distribution has almost no training data.

**Core Idea**: An equivariant asymmetric autoencoder is used to extract "distributional structural priors." A small number of samples from the target distribution encode its prior features, which are then injected into the diffusion denoising process to guide generation toward sparse regions of the target distribution.

## Method

### Overall Architecture

GODD takes atomic coordinates and types of 3D molecules as input. The overall framework consists of two main components: (1) an equivariant asymmetric autoencoder that learns distributional structural priors from molecular data, and (2) a prior-guided diffusion model that leverages structural priors during the denoising process to guide generation. Training is performed on data-rich distributions; at inference time, OOD generation is achieved by injecting prior encodings from the target distribution.

### Key Designs

1. **Equivariant Asymmetric Autoencoder**:

    - **Function**: Extracts structural prior representations that distinguish different distributions from molecular data.
    - **Mechanism**: The "asymmetry" refers to the difference in capacity between the encoder and decoder—the encoder is larger while the decoder is smaller. The encoder maps molecules to distributional prior representations in a latent space, from which the decoder reconstructs the molecules. This asymmetric design compels the latent representation to capture high-level structural information at the distribution level (e.g., scaffold patterns, ring structure characteristics) rather than individual molecular details. The entire architecture satisfies SE(3) equivariance, ensuring that prior representations are invariant to rotations and translations.
    - **Design Motivation**: A symmetric autoencoder tends to learn per-molecule reconstruction rather than distribution-level features. The asymmetric design enforces an information bottleneck that drives the model to extract more abstract distributional priors, which is key to achieving cross-distribution generalization.

2. **Prior-Guided Diffusion Denoising**:

    - **Function**: Injects structural priors into the denoising process of the diffusion model to guide the generation direction.
    - **Mechanism**: The denoising network of a standard diffusion model predicts noise as $\epsilon_\theta(x_t, t)$; GODD extends this to $\epsilon_\theta(x_t, t, z)$, where $z$ is a prior representation encoded from a small number of samples from the target distribution. The prior representation is injected into intermediate layers of the denoising network via cross-attention or additive injection, causing each denoising step to shift toward structural characteristics of the target distribution.
    - **Design Motivation**: The generation direction of an unconditional diffusion model is determined by the training data distribution and cannot transcend its range. By injecting prior encodings from the target distribution, effective guidance is achievable even when that distribution has very limited data.

3. **Training Strategy Under Distribution Shift**:

    - **Function**: Ensures that the learned prior representations remain effective for unseen distributions.
    - **Mechanism**: Distribution shift scenarios are simulated during training—available data are partitioned into multiple sub-distributions that alternately serve as "source distributions" and "target distributions." This curriculum learning strategy trains the model to extract useful distributional priors from a small number of samples rather than memorizing specific distributions.
    - **Design Motivation**: Training on a single distribution may cause the model to overfit to that distribution's prior features. Multi-distribution simulation training enhances the generalizability of prior extraction.

### Loss & Training

The total loss comprises: (1) the standard denoising loss of the diffusion model $\mathcal{L}_{\text{diff}} = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t, z)\|^2]$; (2) the reconstruction loss of the autoencoder; and (3) a contrastive learning loss on prior representations—prior representations of molecules from the same distribution should be close, while those from different distributions should be distant.

## Key Experimental Results

### Main Results

Evaluated on a standard OOD structural shift benchmark measuring the combined success rate of molecular validity, uniqueness, and novelty.

| Evaluation Scenario | Metric | GODD | Prev. SOTA | Gain |
|---|---|---|---|---|
| Scaffold shift | Success rate | Best | -- | +12.6% |
| Ring structure shift | Success rate | Best | -- | Significant |
| Fragment-based drug design | Combined metric | Competitive | -- | Good generalization |

### Ablation Study

| Configuration | Success Rate | Notes |
|---|---|---|
| GODD (Full) | Best | Complete model |
| w/o asymmetric design | Degraded | Symmetric AE fails to learn distributional priors |
| w/o prior guidance | Significantly degraded | Unconditional diffusion cannot perform OOD generation |
| w/o equivariance | Degraded | Disrupts geometric symmetry of molecules |
| w/o multi-distribution training | Degraded | Reduced generalizability of prior extraction |

### Key Findings

- The asymmetric autoencoder design is critical for learning effective distributional priors—the symmetric variant tends to memorize individual molecules rather than distribution-level features.
- Prior guidance is essential for OOD generation—an unguided diffusion model suffers a substantial drop in success rate under OOD settings.
- Equivariance guarantees are necessary—breaking equivariance leads to geometrically unreasonable generated molecules.
- The model also demonstrates strong performance on fragment-based drug design tasks, validating its practical utility.

## Highlights & Insights

- The design of the **asymmetric autoencoder for capturing distributional priors** is particularly elegant—the information bottleneck forces the model to learn distribution-level rather than instance-level representations, which is the key insight for addressing generation in data-scarce distributions.
- **Reformulating OOD generation as a conditional generation problem** has broad transfer value—the same paradigm can be applied to OOD generation in other domains such as protein design and materials design.
- A 12.6% improvement in success rate represents significant progress in molecule generation and has practical potential for AI-driven drug discovery.

## Limitations & Future Work

- Experiments are conducted primarily on relatively small-scale molecular datasets; scalability to large-scale pharmaceutical databases remains to be verified.
- The definition of "a small number of target distribution samples" is vague—how many samples are sufficient to effectively encode distributional priors?
- Pharmacological property constraints (e.g., ADMET properties) are not considered; generated molecules may be structurally novel but pharmacologically infeasible.
- Integration with virtual screening pipelines could filter out molecules that fail to satisfy drug-likeness constraints after generation.

## Related Work & Insights

- **vs. EDM/GeoDiff**: Standard 3D molecular diffusion models excel at in-distribution generation but cannot handle structural shifts. GODD extends the applicability of diffusion models through prior guidance.
- **vs. property-guided generation methods**: Prior OOD molecule generation research has focused mainly on property shifts (e.g., MOOD); GODD is the first to systematically address structural shifts, tackling a more fundamental challenge.
- **vs. VAE-based methods**: VAEs achieve a degree of OOD generation through latent space interpolation but lack explicit modeling of structural priors. GODD's prior encoding is more controllable and interpretable.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of asymmetric AE and prior-guided diffusion is novel; the problem formulation of OOD structural shift is innovative
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation across multiple shift types + drug design application validation + ablation study
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear; method description is detailed
- Value: ⭐⭐⭐⭐ Has practical application value for AI-driven drug discovery

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Apo2Mol: 3D Molecule Generation via Dynamic Pocket-Aware Diffusion Models](apo2mol_3d_molecule_generation_via_dynamic_pocket-aware_diff.md)
- [\[NeurIPS 2025\] Pharmacophore-Guided Generative Design of Novel Drug-Like Molecules](../../NeurIPS2025/computational_biology/pharmacophore-guided_generative_design_of_novel_drug-like_molecules.md)
- [\[ICLR 2026\] Protein Counterfactuals via Diffusion-Guided Latent Optimization](../../ICLR2026/computational_biology/protein_counterfactuals_via_diffusion-guided_latent_optimization.md)
- [\[NeurIPS 2025\] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](../../NeurIPS2025/computational_biology/uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)
- [\[NeurIPS 2025\] Manipulating 3D Molecules in a Fixed-Dimensional E(3)-Equivariant Latent Space](../../NeurIPS2025/computational_biology/manipulating_3d_molecules_in_a_fixed-dimensional_e3-equivariant_latent_space.md)

</div>

<!-- RELATED:END -->
