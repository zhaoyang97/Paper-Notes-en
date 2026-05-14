---
title: >-
  [Paper Note] Vision Transformers for Cosmological Fields: Application to Weak Lensing Mass Maps
description: >-
  [NeurIPS 2025][Physics][Vision Transformer] This work presents the first systematic application of Vision Transformers (ViT and Swin Transformer) to constraining cosmological parameters ($\Omega_m$ and $S_8$) from weak l…
tags:
  - "NeurIPS 2025"
  - "Physics"
  - "Vision Transformer"
  - "Weak Lensing"
  - "Cosmological Parameters"
  - "Swin Transformer"
  - "Simulation-Based Inference"
date: 2026-05-08
content_hash: 54bd74815537a1c0
---

# Vision Transformers for Cosmological Fields: Application to Weak Lensing Mass Maps

**Conference**: NeurIPS 2025
**arXiv**: [2512.07125](https://arxiv.org/abs/2512.07125)
**Code**: None
**Area**: Cosmology, Deep Learning, Weak Gravitational Lensing
**Keywords**: Vision Transformer, Weak Lensing, Cosmological Parameters, Swin Transformer, Simulation-Based Inference

## TL;DR
This work presents the first systematic application of Vision Transformers (ViT and Swin Transformer) to constraining cosmological parameters ($\Omega_m$ and $S_8$) from weak lensing convergence maps, comparing attention-based architectures against CNNs within a simulation-based inference framework.

## Background & Motivation

### State of the Field

**Background**: Weak gravitational lensing serves as a key probe of cosmic structure formation, with small-scale nonlinear structures encoding rich non-Gaussian information.

### Limitations of Prior Work

**Limitations of Prior Work**: Traditional two-point statistics capture only Gaussian features, while CNNs have been shown to extract non-Gaussian information for cosmological parameter constraints.

### Root Cause

**Key Challenge**: Vision Transformers have achieved breakthroughs in computer vision, yet remain systematically unevaluated in the weak lensing domain.

### Starting Point

**Key Insight**: ViTs directly capture global context via attention mechanisms, without requiring the hierarchical aggregation of CNNs, and offer greater interpretability.

## Method

### Overall Architecture
- Simulation-Based Inference (SBI) framework: trains a neural density estimator (NDE) to approximate $p(d|\theta)$
- Visual models as feature compressors: map 512×512 convergence fields to low-dimensional data vectors
- NDE ensemble: 3 MAFs + 3 MDNs; NDEs with bias exceeding 5% are discarded, and the remaining posterior samples are merged

### Key Designs
1. **Simulation Data**:

    - Convergence maps generated from the DarkGridV1 N-body simulation suite
    - Four tomographic bins adopting the DES-Y3 redshift distribution
    - Three test configurations: noiseless single-channel, LSST-Y1 single-channel, and LSST-Y1 four-channel (full tomography)
    - 13,680 convergence maps of size 512×512 in total

2. **Model Architecture Coverage**:

    - CNN family: Baseline CNN (500K), ResNet-18/34/50/101 (11M–44M)
    - ViT family: ViT-B (86M), ViT-L (307M), ViT-H (632M)
    - Swin family: Swin-T (29M), Swin-S (50M), Swin-B (88M), Swin-L (197M)

3. **Pretraining Strategy**:

    - Synthetic data with statistical properties similar to weak lensing fields are efficiently generated using analytic models
    - Models are pretrained on synthetic data and then fine-tuned on realistic simulations
    - Pretraining yields significant gains for Transformer architectures, with minimal impact on CNNs

### Loss & Training
- L2 loss (RMSE) for model training
- AdamW optimizer
- ReduceLROnPlateau scheduler (patience=10, factor=0.3)
- Learning rate $10^{-3}$ for CNNs; $10^{-5}$ for ViT/Swin
- 80:10:10 train/validation/test split; early stopping patience=30

## Key Experimental Results

### Main Results (Model Performance Overview)

| Model | Parameters | Noiseless RMSE ($S_8$) | LSST-Y1 RMSE ($S_8$) |
|-------|------------|------------------------|----------------------|
| Baseline CNN | 500K | Good | Good |
| ResNet-50 | 24M | Good | Good |
| ViT-B | 86M | Poor | Poor |
| ViT-L | 307M | Poor | Poor |
| Swin-T | 29M | Moderate | Comparable to CNN |
| Swin-L | 197M | **Best** (noiseless) | Comparable to CNN |

### Ablation Study on Pretraining

| Setting | Swin (w/o pretraining) | Swin (w/ pretraining) | CNN (w/o pretraining) |
|---------|------------------------|------------------------|------------------------|
| 25% training data | Significant degradation | Near full-data performance | Stable |
| 50% training data | Moderate degradation | Near full-data performance | Stable |
| 100% training data | Baseline | Best | Baseline |

### Key Findings
- **In the noiseless regime**, Swin Transformers demonstrate marginal superiority over CNNs, benefiting from greater model flexibility.
- **Upon inclusion of realistic noise**, Swin and CNN performance become comparable, and the Transformer advantage disappears.
- Vanilla ViT consistently underperforms, likely due to low training efficiency on small datasets and difficulty in capturing fine-scale features.
- Pretraining has a pronounced effect on Transformers but negligible impact on CNNs.
- Posterior estimates from both model families are validated as well-calibrated via TARP coverage tests.

## Highlights & Insights
- This is the first systematic application of attention-based architectures to cosmological parameter constraints from weak lensing mass maps.
- Swin Transformers achieve comparable performance to CNNs (similar Figure of Merit), but offer no advantage under noisy conditions.
- The pretraining strategy reveals Transformers' dependence on training data volume: pretraining can effectively compensate for data scarcity.
- The interpretability of Transformer attention weights represents a promising direction for future investigation.

## Limitations & Future Work
- Only $\Omega_m$ and $\sigma_8$ are varied; systematic effects such as photometric redshift uncertainties are not considered.
- The limited dataset size (13,680 maps) constrains the full potential of Transformer architectures.
- More advanced pretraining schemes, such as masked autoencoding for ViT/Swin, have not been explored.
- The physical interpretability of Transformer attention weights (e.g., identifying regions contributing the most information) warrants further study.

## Related Work & Insights
- The importance of pretraining for Transformers is once again confirmed in the context of scientific data.
- The windowed attention of Swin performs better on scientific images, possibly because weak lensing signals exhibit both local and global characteristics.
- For scientific data with limited signal-to-noise ratios, Transformers do not necessarily outperform CNNs; stronger inductive biases (e.g., translation invariance) may be more beneficial.
- The inference framework combining NDE ensembles (MAF + MDN) with coverage validation (TARP) is worth adopting in related work.
- The synthetic data pretraining strategy demonstrates practical value for addressing data scarcity in astronomy.

## Rating
- Novelty: ⭐⭐⭐ (Novel at the application level; methodology is relatively standard)
- Technical Contribution: ⭐⭐⭐ (Systematic comparison offers useful reference value)
- Experimental Thoroughness: ⭐⭐⭐⭐ (12 models × 3 settings + SBI validation)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, objective conclusions)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Neural Deprojection of Galaxy Stellar Mass Profiles](neural_deprojection_of_galaxy_stellar_mass_profiles.md)
- [\[NeurIPS 2025\] Quantum Doubly Stochastic Transformers](quantum_doubly_stochastic_transformers.md)
- [\[NeurIPS 2025\] AstroCo: Self-Supervised Conformer-Style Transformers for Light-Curve Embeddings](astroco_self-supervised_conformer-style_transformers_for_light-curve_embeddings.md)
- [\[NeurIPS 2025\] Simulation-Based Inference for Neutrino Interaction Model Parameter Tuning](simulation-based_inference_for_neutrino_interaction_model_parameter_tuning.md)
- [\[NeurIPS 2025\] Dynamic Diffusion Schrödinger Bridge in Astrophysical Observational Inversions](dynamic_diffusion_schrödinger_bridge_in_astrophysical_observational_inversions.md)

</div>

<!-- RELATED:END -->
