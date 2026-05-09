---
title: >-
  [Paper Note] Toward Complete Merger Identification at Cosmic Noon with Deep Learning
description: >-
  [NeurIPS 2025 (ML4PS Workshop)][Galaxy merger identification] A ResNet18 is trained on simulated HST CANDELS images generated from IllustrisTNG50, demonstrating for the first time that deep learning can successfully identify galaxy mergers at high redshift $1<z<1.5$, including minor mergers ($\mu \geq 1/10$) and low-mass galaxies ($M_\star > 10^8 M_\odot$), achieving an overall accuracy of ~73%. Model behavior is further analyzed through Grad-CAM and UMAP.
tags:
  - NeurIPS 2025 (ML4PS Workshop)
  - Galaxy merger identification
  - ResNet18
  - Cosmic noon
  - IllustrisTNG
  - Grad-CAM
date: 2026-05-08
content_hash: 44361e6e154784d8
---

# Toward Complete Merger Identification at Cosmic Noon with Deep Learning

**Conference**: NeurIPS 2025 (ML4PS Workshop)
**arXiv**: [2511.15006](https://arxiv.org/abs/2511.15006)
**Code**: [https://github.com/alschechter/NeurIPSCosmicNoonMergerID](https://github.com/alschechter/NeurIPSCosmicNoonMergerID)
**Area**: Astrophysics / Deep Learning
**Keywords**: Galaxy merger identification, ResNet18, Cosmic noon, IllustrisTNG, Grad-CAM

## TL;DR
A ResNet18 is trained on simulated HST CANDELS images generated from IllustrisTNG50, demonstrating for the first time that deep learning can successfully identify galaxy mergers at high redshift $1<z<1.5$, including minor mergers ($\mu \geq 1/10$) and low-mass galaxies ($M_\star > 10^8 M_\odot$), achieving an overall accuracy of ~73%. Model behavior is further analyzed through Grad-CAM and UMAP.

## Background & Motivation

**Background**: Galaxy mergers are a key driver of galaxy evolution, influencing star formation, AGN activity, and morphological transformation. Traditional identification methods include non-parametric approaches (measuring asymmetry, concentration, and clumpiness) and close-pair analyses, which are primarily effective at low redshift for massive, major mergers.

**Limitations of Prior Work**:
   - Non-parametric methods are calibrated at low redshift and are unreliable at high redshift ($z>1$)
   - Most approaches target major mergers ($\mu \geq 1/4$) and massive galaxies ($M_\star > 10^{10} M_\odot$)
   - Close-pair selection captures only early-stage mergers, missing roughly half the merger population (post-coalescence mergers)
   - Existing CNN-based studies are confined to $z<1$ and $M_\star > 10^9 M_\odot$

**Key Challenge**: A comprehensive understanding of the role of mergers in galaxy evolution requires large merger catalogs spanning all merger stages, mass ratios, and mass ranges; however, existing methods suffer from severe selection bias at high redshift and the low-mass end.

**Goal**: To extend CNN-based merger identification into a more challenging parameter space: high redshift ($1<z<1.5$), low stellar mass ($M_\star > 10^8 M_\odot$), and minor mass ratios ($\mu \geq 1/10$).

**Key Insight**: IllustrisTNG50 high-resolution simulations (~0.1 kpc spatial resolution) are used to generate mock HST observations with full physical self-consistency, providing accurate merger/non-merger ground-truth labels and avoiding biases inherent in visual classification.

**Core Idea**: High-resolution cosmological simulations + radiative transfer + realistic background injection = training data; ResNet18 + transfer learning = merger identification across the full parameter space at high redshift.

## Method

### Overall Architecture
Data pipeline: IllustrisTNG50 simulation → SKIRT radiative transfer (including dust and AGN) → mock HST CANDELS three-channel images (F606W/F814W/F160W) → injection into real CANDELS mosaic backgrounds → ResNet18 binary classification (merger/non-merger). Each galaxy is observed from 6 viewing angles; images are normalized with logarithmic stretching.

### Key Designs

1. **Simulated Data Construction**:

    - Function: Generate mock HST observational images with precise merger ground-truth labels.
    - Mechanism: TNG50 snapshots at $z=1$ and $z=1.5$ are used, with merger events defined within a 500 Myr time window. Each merger galaxy is paired with a mass-matched non-merger. Multi-band images are generated using the SKIRT radiative transfer code, with PSF convolution and real CANDELS background noise injected (5$\sigma$ limiting magnitude 26.5).
    - Design Motivation: Cosmological simulations provide merger ground truth independent of any identification method, avoiding circular bias from visual classification.

2. **ResNet18 + Zoobot Transfer Learning**:

    - Function: Leverage pretrained weights to accelerate convergence on simulated data.
    - Mechanism: ResNet18 is initialized with Zoobot 2.0.2 pretrained weights, trained with a learning rate of $10^{-5}$, exponential decay of 0.5, Adam optimizer, and cross-entropy loss. The output head is modified to 2 nodes (merger/non-merger).
    - Design Motivation: Zoobot is pretrained on large-scale galaxy morphology classification data; its learned feature representations transfer effectively to merger identification.

3. **Multi-Viewing-Angle Analysis**:

    - Function: Assess the effect of viewing angle on identification accuracy using 6 orientations per galaxy.
    - Mechanism: All viewing angles of the same galaxy are assigned to the same data split (train/validation/test) to prevent information leakage. The number of correctly identified viewing angles per galaxy is recorded and analyzed as a function of mass ratio and stellar mass.
    - Design Motivation: Merger features such as tidal tails are visible only from certain orientations, potentially setting a theoretical upper bound on identification accuracy.

### Loss & Training
- Cross-entropy loss, Adam optimizer
- Data augmentation: rotation $\pm 30°$, horizontal/vertical flipping
- Training set: ~5,900 mergers + ~5,900 non-mergers
- Early stopping: training halts when validation loss improvement is less than 0.0005 over 5 epochs

## Key Experimental Results

### Main Results
Average performance across 3 random seeds:

| Metric | Value |
|--------|-------|
| Accuracy | 73.0 $\pm$ 0.4% |
| Purity | 74.0 $\pm$ 0.01% |
| Completeness | 72.0 $\pm$ 0.01% |
| Brier Score | 0.19 $\pm$ 0.01 |
| ECE | 0.08 $\pm$ 0.03 |
| AUC | 0.8 $\pm$ 0.01 |

### Ablation Study: Accuracy by Merger Subclass

| Subclass | Accuracy |
|----------|----------|
| All mergers | 71.9 $\pm$ 1.0% |
| Major mergers ($\mu \geq 1/4$) | 75.8 $\pm$ 0.8% |
| Minor mergers ($1/10 < \mu < 1/4$) | 68.3 $\pm$ 0.7% |
| Pre-merger | 79.6 $\pm$ 0.7% |
| Post-merger | 66.0 $\pm$ 0.01% |
| Non-merger | 74.0 $\pm$ 0.01% |

### Key Findings
- **First demonstration that CNNs can identify minor and low-mass mergers at $z>1$**: All prior CNN merger identification work was limited to $z<1$ or $M_\star > 10^9 M_\odot$.
- **Post-coalescence mergers are the most difficult to identify** (66%), while pre-merger pairs are the easiest (79.6%): morphological disturbances near final coalescence are less pronounced than during the double-nucleus phase.
- **Viewing angle sets a theoretical accuracy ceiling**: nearly all mergers are correctly identified from at least one viewing angle, but not all orientations yield correct predictions—even some major mergers are identified from $\leq 3$ viewing angles.
- **UMAP reveals that the network has learned physically meaningful quantities**: the latent space exhibits clear gradients in stellar mass and specific star formation rate (sSFR), but not in merger stage or mass ratio—indicating the network relies on morphological features rather than merger-specific signatures.
- **Grad-CAM confirms the network attends to galaxies rather than backgrounds**: the central galaxy is highlighted rather than background noise or unrelated sources.

## Highlights & Insights
- The finding that **viewing angle limits identifiability** applies broadly to all image-based merger identification methods: even a perfect classifier cannot detect merger features that are geometrically hidden, setting a theoretical accuracy ceiling of approximately 85%.
- The **sSFR gradient revealed by UMAP** suggests that some misclassifications arise from non-merger galaxies with high sSFR whose clumpy star formation mimics merger morphology; future work should incorporate SFR-matched negative samples.
- The paradigm of using cosmological simulations to generate training data is generalizable to other astronomical classification tasks.

## Limitations & Future Work
- The TNG50 box volume is small (50 Mpc), resulting in insufficient high-mass galaxy samples and limited training data for massive major mergers.
- Non-merger samples are only mass-matched without SFR matching—high-SFR non-mergers may be systematically misclassified as mergers.
- An accuracy of 73% remains insufficient for constructing a reliable merger catalog; further improvements are needed.
- The model has not been validated on real observational data (simulation-only evaluation).

## Related Work & Insights
- **vs. Margalef-Bentabol et al. (2024)**: That work achieves ~73% accuracy at $0.1<z<1$ and $M_\star > 10^9 M_\odot$. The present work matches this accuracy in a significantly more challenging parameter space (higher redshift, lower mass).
- **vs. Bickley et al. (2024)**: That work identified the influence of viewing angle on accuracy; the present work validates and extends this finding to high redshift.
- **vs. Rose et al. (2024)**: That work applies CNN identification at $3<z<5$ using CEERS data but does not cover the low-mass regime.

## Rating
- Novelty: ⭐⭐⭐⭐ First validation of CNN-based merger identification at high redshift and low stellar mass
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-seed evaluation, multi-angle analysis, and complete Grad-CAM/UMAP interpretability analysis
- Writing Quality: ⭐⭐⭐⭐ Workshop paper with clear methodology and conclusions
- Value: ⭐⭐⭐ Practically significant for the astronomical community, though methodological novelty is limited

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Transfer Learning Beyond the Standard Model](transfer_learning_beyond_the_standard_model.md)
- [\[NeurIPS 2025\] Latent Representation Learning in Heavy-Ion Collisions with MaskPoint Transformer](latent_representation_learning_in_heavy-ion_collisions_with_maskpoint_transforme.md)
- [\[NeurIPS 2025\] POLARIS: A High-contrast Polarimetric Imaging Benchmark Dataset for Exoplanetary Disk Representation Learning](polaris_a_high-contrast_polarimetric_imaging_benchmark_dataset_for_exoplanetary_.md)
- [\[NeurIPS 2025\] Multi-Modal Masked Autoencoders for Learning Image-Spectrum Associations for Galaxy Evolution and Cosmology](multi-modal_masked_autoencoders_for_learning_image-spectrum_associations_for_gal.md)
- [\[CVPR 2026\] QKD: Quantum-Gated Task-interaction Knowledge Distillation for Class-Incremental Learning](../../CVPR2026/physics/qkd_quantum_gated_incremental_learning.md)

</div>

<!-- RELATED:END -->
