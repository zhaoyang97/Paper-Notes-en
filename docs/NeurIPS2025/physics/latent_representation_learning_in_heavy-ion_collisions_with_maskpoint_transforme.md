---
title: >-
  [Paper Note] Latent Representation Learning in Heavy-Ion Collisions with MaskPoint Transformer
description: >-
  [NEURIPS2025 (Workshop: ML and Physical Sciences)][Physics & Scientific Computing][heavy-ion collisions] This work introduces a masked point cloud Transformer autoencoder to heavy-ion collision analysis. Through a two-st…
tags:
  - "NEURIPS2025 (Workshop: ML and Physical Sciences)"
  - "Physics & Scientific Computing"
  - "heavy-ion collisions"
  - "masked autoencoder"
  - "Transformer"
  - "self-supervised pre-training"
  - "quark-gluon plasma"
date: 2026-05-08
content_hash: 7844e02d0cf4326c
---

# Latent Representation Learning in Heavy-Ion Collisions with MaskPoint Transformer

**Conference**: NEURIPS2025 (Workshop: ML and Physical Sciences)
**arXiv**: [2510.06691](https://arxiv.org/abs/2510.06691)  
**Code**: [https://github.com/Giovanni-Sforza/MaskPoint-AMPT](https://github.com/Giovanni-Sforza/MaskPoint-AMPT)  
**Area**: Physics
**Keywords**: heavy-ion collisions, masked autoencoder, Transformer, self-supervised pre-training, quark-gluon plasma

## TL;DR
This work introduces a masked point cloud Transformer autoencoder to heavy-ion collision analysis. Through a two-stage paradigm of self-supervised pre-training followed by supervised fine-tuning, the model learns nonlinear latent representations substantially stronger than those of PointNet—reducing PC1 distribution overlap from 2.42% to 0.27%—providing a general feature learning framework for studying QGP properties.

## Background & Motivation

**Background**: Relativistic heavy-ion collisions are the sole experimental means of studying QCD phase transitions and quark-gluon plasma (QGP) properties. Traditional analyses rely on hand-crafted observables (particle spectra, anisotropic flow, etc.), but these scalar quantities fail to fully exploit the information contained in high-dimensional final-state data.

**Limitations of Prior Work**:
   - Traditional observables are hand-selected and may miss physically important yet subtle data structures.
   - Deep learning methods such as PointNet have been preliminarily applied to collision data, but the representations they learn are essentially linear copies of individual physical observables (e.g., $\sigma_\eta$).
   - Systematic application of self-supervised pre-training in high-energy nuclear physics remains lacking.

**Key Challenge**: Final-state particle data constitute high-dimensional unordered point clouds. Capturing global inter-particle correlations is essential, yet PointNet's global pooling discards fine-grained interaction information between particles.

**Goal**:
   - Introduce a Transformer autoencoder to learn richer representations of collision events.
   - Verify whether self-supervised pre-training can capture nonlinear physical structures beyond individual observables.

**Key Insight**: The three-momenta $(p_x, p_y, p_z)$ of final-state particles are treated as a 3D point cloud, leveraging mature masked point cloud modeling techniques from computer vision.

**Core Idea**: A self-supervised pre-trained Transformer autoencoder learns nonlinear physical features from heavy-ion collision point clouds, significantly outperforming the linear representations of PointNet.

## Method

### Overall Architecture

Two-stage paradigm:
- **Stage 1 — Self-supervised Pre-training**: Mask 25% of the point cloud → Transformer encoder extracts a 96-dimensional feature vector $\mathbf{f}$ → Transformer decoder discriminates "real particles vs. fake particles."
- **Stage 2 — Supervised Fine-tuning**: Freeze the encoder → MLP classifier performs collision system identification (Pb+Pb vs. p+Pb).

### Key Designs

1. **Masked Discriminative Pre-training**:

    - Function: Learn the intrinsic physical structure of collision events without labels.
    - Mechanism: Farthest Point Sampling (FPS) masks 25% of the point cloud; the remaining 96 particles pass through PointNet for local feature extraction → 6-layer Transformer encodes global correlations → produces a 96-dimensional feature $\mathbf{f}$. The decoder uses cross-attention to fuse $\mathbf{f}$ with either "truly masked points" or "randomly sampled fake points," and an MLP performs binary classification (real vs. fake) trained with cross-entropy loss.
    - Design Motivation: The discriminative objective (rather than reconstruction) compels the encoder to learn high-quality physical features by distinguishing collision-generated real particles from randomly sampled fake ones.

2. **Transformer Encoder Architecture**:

    - Function: Capture long-range correlations between particles.
    - Mechanism: PointNet first extracts local features for each patch; a 6-layer Transformer (self-attention) then models global particle–particle interactions.
    - Design Motivation: PointNet's global max-pooling discards inter-particle relational information, whereas Transformer self-attention preserves it.

3. **PCA + SHAP Interpretability Analysis**:

    - Function: Investigate what physical information the learned features encode.
    - Mechanism: PCA reduces the 96-dimensional features to principal components; their linear correlations with traditional physical observables ($\sigma_\eta$, $\langle p_T \rangle$, etc.) are computed. Random Forest + SHAP reveals nonlinear associations.
    - Key Findings: PointNet's PC1 correlates linearly with $\sigma_\eta$, indicating it merely "fits" a known observable. The autoencoder's PC1 shows near-zero linear correlation with $\sigma_\eta$, yet SHAP identifies $\sigma_\eta$ as the most important contributor—evidence of nonlinear encoding.

### Loss & Training

- Data: AMPT-simulated Pb+Pb and p+Pb collision events; 128 particles per event ($|\eta|<2.4$, $p_T > 0.4$ GeV/c).
- Pre-training and fine-tuning each run for 300 epochs; AdamW optimizer with cosine learning rate decay.
- Masking ratio: 25%.

## Key Experimental Results

### Main Results — Collision System Classification

| Method | PC1 Distribution Overlap | Classification Accuracy (Full Multiplicity Range) | Notes |
|--------|--------------------------|--------------------------------------------------|-------|
| $\sigma_\eta$ (physical observable) | 2.71% | — | Theoretical optimum for a single variable |
| PointNet | 2.42% | Lower | Approaches the theoretical limit of $\sigma_\eta$ |
| **MaskPoint Transformer** | **0.27%** | **Significantly higher** | Breaks the single-variable limit |

### Ablation Study

| Configuration | Classification Accuracy | Notes |
|---------------|------------------------|-------|
| With PointNet preprocessing | Higher | Local feature extraction is beneficial |
| Without PointNet preprocessing | Lower | Direct Transformer on raw point clouds performs poorly |
| Masking ratio 25% | Optimal | Best performance in experiments |
| Masking ratio 50% / 75% | Degraded | Too few visible particles for the encoder to learn sufficient information |

### Key Findings

- **Breaking the single-variable limit**: PointNet's PC1 overlap (2.42%) is close to $\sigma_\eta$ (2.71%), indicating it essentially learns a linear representation of $\sigma_\eta$. The autoencoder's PC1 overlap is only 0.27%—stronger than any single known observable—suggesting it captures novel physical information.
- **Unsupervised PCA space naturally separates collision systems**: Figure 2 shows that even after unsupervised pre-training alone, the PC1–PC2 space already clearly distinguishes Pb+Pb from p+Pb, indicating the encoder spontaneously learns intrinsic differences between the two collision systems.
- **Direct evidence of nonlinear encoding**: Near-zero linear correlation with $\sigma_\eta$ yet highest SHAP importance confirms that the information is encoded as a nonlinear combination rather than a simple copy.

## Highlights & Insights

- **"Low linear correlation + high SHAP importance" interpretability paradigm**: This analytical approach elegantly distinguishes whether a model is merely performing linear fitting. High linear correlation implies the model is copying the observable; low linear correlation with high SHAP importance indicates a deeper nonlinear structure has been learned. This paradigm is broadly transferable to other AI4Science tasks.
- **Demonstration of self-supervised pre-training value in particle physics**: The work demonstrates that "pre-train for representations, then fine-tune for tasks" significantly improves performance in high-energy nuclear physics, providing a basis for building foundation models in particle physics.
- **Cross-domain transfer of the point cloud perspective**: The direct transfer of masked point cloud modeling from CV/3D domains to physics confirms the cross-domain applicability of these methods.

## Limitations & Future Work

- **Simulation data only**: Events are generated with AMPT; validation on real experimental data has not been performed.
- **Limited input features**: Only three-momenta $(p_x, p_y, p_z)$ are used; four-momenta, charge, spin, and other particle attributes are not exploited.
- **Simple downstream task**: Collision system identification (Pb+Pb vs. p+Pb) is not a genuine practical challenge and serves only as a proxy for evaluating representation quality.
- **Absence of physical priors**: Conservation laws, Lorentz symmetry, and other physical constraints are not incorporated.
- **Future directions**:
    - Validate on real RHIC/LHC data.
    - Incorporate Lorentz-equivariant network architectures.
    - Apply to more physically meaningful downstream tasks: chiral magnetic effect (CME) detection, nuclear deformation studies.
    - Integrate particle species information ($\pi, K, p$, etc.).

## Related Work & Insights

- **vs. PointNet**: PointNet can only learn linear approximations of physical observables; the Transformer learns nonlinear combinations, representing a qualitative leap in representational capacity.
- **vs. OmniJet-α / Particle Transformer**: These are foundation models for LHC jet physics; this work transplants analogous ideas into the heavy-ion collision domain.
- **vs. traditional observable-based analysis**: Traditional methods rely on domain experts to design a small set of observables, potentially missing important information. Self-supervised methods can discover new "data-driven observables."

## Rating

- Novelty: ⭐⭐⭐⭐ First application of self-supervised Transformers to heavy-ion collision feature learning; interpretability analysis paradigm is elegant.
- Experimental Thoroughness: ⭐⭐⭐ Workshop-paper scale; only one downstream task; simulation data only.
- Writing Quality: ⭐⭐⭐⭐ Well-balanced presentation of physical background and ML methodology; interpretability analysis is clearly articulated.
- Value: ⭐⭐⭐⭐ Lays groundwork for AI foundation models in high-energy nuclear physics; methodology is transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] POLARIS: A High-contrast Polarimetric Imaging Benchmark Dataset for Exoplanetary Disk Representation Learning](polaris_a_high-contrast_polarimetric_imaging_benchmark_dataset_for_exoplanetary_.md)
- [\[AAAI 2026\] SAOT: An Enhanced Locality-Aware Spectral Transformer for Solving PDEs](../../AAAI2026/physics/saot_an_enhanced_locality-aware_spectral_transformer_for_solving_pdes.md)
- [\[NeurIPS 2025\] Transfer Learning Beyond the Standard Model](transfer_learning_beyond_the_standard_model.md)
- [\[NeurIPS 2025\] Integration Matters for Learning PDEs with Backward SDEs](integration_matters_for_learning_pdes_with_backward_sdes.md)
- [\[NeurIPS 2025\] Toward Complete Merger Identification at Cosmic Noon with Deep Learning](toward_complete_merger_identification_at_cosmic_noon_with_deep_learning.md)

</div>

<!-- RELATED:END -->
