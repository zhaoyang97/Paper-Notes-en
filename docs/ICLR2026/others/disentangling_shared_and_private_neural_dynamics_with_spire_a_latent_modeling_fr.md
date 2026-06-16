---
title: >-
  [Paper Note] Disentangling Shared and Private Neural Dynamics with SPIRE: A Latent Modeling Framework for Deep Brain Stimulation
description: >-
  [ICLR2026][latent variable model] This paper proposes SPIRE (Shared–Private Inter-Regional Encoder), a nonlinear dual-latent-space autoencoder framework that decomposes intracranial recordings from multiple brain regions…
tags:
  - "ICLR2026"
  - "latent variable model"
  - "shared-private disentanglement"
  - "deep brain stimulation"
  - "multi-region neural dynamics"
  - "autoencoder"
date: 2026-05-08
content_hash: 0c485e3aaccf6580
---

# Disentangling Shared and Private Neural Dynamics with SPIRE: A Latent Modeling Framework for Deep Brain Stimulation

**Conference**: ICLR2026
**arXiv**: [2510.25023](https://arxiv.org/abs/2510.25023)  
**Code**: [GitHub](https://github.com/Rahil-Soroush/spire-iclr2026)  
**Area**: LLM Evaluation
**Keywords**: latent variable model, shared-private disentanglement, deep brain stimulation, multi-region neural dynamics, autoencoder

## TL;DR

This paper proposes SPIRE (Shared–Private Inter-Regional Encoder), a nonlinear dual-latent-space autoencoder framework that decomposes intracranial recordings from multiple brain regions into shared and private subspaces via cross-region alignment and orthogonal disentanglement losses. Trained exclusively on baseline data, SPIRE detects frequency-dependent network reorganization induced by DBS stimulation.

## Background & Motivation

**Background**: Movement disorders such as dystonia and Parkinson's disease involve dysfunction in the basal ganglia–thalamo–cortical circuit. Deep brain stimulation (DBS) targeting the globus pallidus internus (GPi) and subthalamic nucleus (STN) demonstrates substantial clinical efficacy, yet the network-level mechanisms by which it modulates cross-regional neural dynamics remain poorly understood.

**Limitations of Prior Work**: Most DBS analyses focus on local features (spectral power, evoked potentials), overlooking changes in cross-regional coordination patterns. Existing latent variable models suffer from critical limitations: (1) GPFA and CCA assume linearity and cannot capture the nonlinear structure of real neural data; (2) DLAG can decompose shared/private components but is constrained to a linear-Gaussian framework and is primarily designed for spike data; (3) multimodal models such as SharedAE and MMVAE align shared spaces but are not designed for intracranial stimulation recordings and lack explicit shared-private disentanglement mechanisms.

**Key Challenge**: No existing framework simultaneously satisfies three requirements: nonlinear modeling capacity, explicit shared vs. private decomposition, and applicability to human LFP data under external perturbation. Understanding how stimulation reorganizes intrinsic cross-regional coordination patterns is essential for elucidating the circuit-level mechanisms of DBS.

**Key Insight**: The authors design a dual-latent-space framework following a "train on baseline, infer under perturbation" paradigm—establishing a reference model of intrinsic coordination on stimulation-free data, then observing the reorganization of the shared latent space under DBS conditions to reveal stimulation's effects on network-level dynamics.

## Method

### Overall Architecture

SPIRE assigns an independent GRU encoder-decoder to each brain region $r$. The encoder maps multi-channel input $x^{(r)} \in \mathbb{R}^{B \times T \times C_r}$ to a hidden state $h^{(r)}$, which is then linearly projected to yield shared latent variables $z_{\text{sh}}^{(r)} \in \mathbb{R}^{B \times T \times d_{\text{sh}}}$ and private latent variables $z_{\text{pr}}^{(r)} \in \mathbb{R}^{B \times T \times d_{\text{pr}}}$. The decoder reconstructs the original signal from the concatenated latent variables: $\hat{x}^{(r)} = f_{\text{dec}}^{(r)}([z_{\text{sh}}^{(r)}, z_{\text{pr}}^{(r)}])$.

### Key Designs

1. **Cross-Region Alignment Module**:

    - A lightweight linear mapping $M^{(s \to r)}$ (initialized as the identity matrix) and a deep 1D convolutional ConvAlign module (initialized with impulse responses) are used to spatiotemporally align shared latent variables across regions.
    - ConvAlign maintains one filter per shared dimension, permitting small phase shifts to model cross-regional conduction delays.
    - Mappings are directional ($s \to r$ and $r \to s$ are learned independently), imposing no symmetry assumption, thereby reflecting the directionality of inter-regional signal propagation.
    - **Design Motivation**: Real neural signals exhibit millisecond-level temporal offsets and nonlinear subspace rotations across brain regions that cannot be captured by a pure matrix mapping.

2. **Nine-Term Multi-Objective Training Loss**:

    - **Reconstruction objectives**: $\mathcal{L}_{\text{rec}}$ (self-reconstruction using shared + private), $\mathcal{L}_{\text{cross}}$ (reconstruction using the other region's shared variables), and $\mathcal{L}_{\text{self}}$ (reconstruction using only the region's own shared variables), ensuring the shared latent variables carry meaningful variance.
    - **Alignment objective**: $\mathcal{L}_{\text{align}}$ applies VICReg regularization (variance–invariance–covariance) to align shared latent variables across regions while preserving region-specific perspective differences.
    - **Disentanglement objectives**: $\mathcal{L}_{\text{orth}}$ penalizes cross-covariance between shared and private components; variance guards $\mathcal{L}_{\text{var-sh}}$ and $\mathcal{L}_{\text{var-pr}}$ prevent degenerate solutions (shared collapse or private vanishing).
    - **Alignment module regularization**: $\mathcal{L}_{\text{mapid}}$ biases the linear mapping toward the identity matrix; $\mathcal{L}_{\text{align-reg}}$ regularizes ConvAlign filters toward impulse responses, ensuring interpretability.

3. **"Train on Baseline, Infer under Perturbation" Paradigm**:

    - The model is trained exclusively on off-stimulation baseline data, establishing a reference framework for intrinsic cross-regional coordination.
    - At inference time, DBS stimulation condition data are fed to the model, enabling observation of how the shared latent space is reorganized by stimulation and how the private latent space reflects local effects.
    - This approach prevents stimulation artifacts from directly contaminating model parameters while using the baseline reference framework as an anchor for quantifying the degree of reorganization.

## Key Experimental Results

### Synthetic Data Validation

Three synthetic datasets of progressively increasing complexity (from linear to nonlinear) were constructed, each containing 100 trials × 250 time steps (0.5 s @ 500 Hz), with 3 shared + 3 private dimensions:

| Dataset | Mixing | Noise | SPIRE CCA (shared) | DLAG CCA (shared) |
|---------|--------|-------|-------------------|-------------------|
| D0 | Linear + Gaussian noise | Gaussian | Comparable to DLAG | Linear-friendly baseline |
| D1 | Nonlinear distortion + bilinear mixing | 1/f + AR(1) | (0.92, 0.91, 0.71) | (0.86, 0.79, 0.60) |
| D2 | D1 + time-varying sinusoidal delays | Same as D1 | Outperforms DLAG | Further degradation |

SPIRE statistically significantly outperforms DLAG in recovering private latent variables ($p < 0.05$). Under nonlinear (D1) and time-varying delay (D2) conditions, SPIRE also surpasses DLAG in shared latent variable recovery. D0 is a linear scenario favorable to DLAG; SPIRE shows no disadvantage but no significant advantage either.

### Human DBS Recording Data

- Intracranial LFP from 10 pediatric dystonia patients (ages 5–23) with electrodes covering GPi and STN across 17 hemispheres.
- Stimulation conditions: GPi at 85/185/250 Hz, STN at 85/185 Hz, and off-stimulation.
- Preprocessing: bipolar referencing, downsampling to 500 Hz, 50 Hz low-pass Butterworth filtering to remove high-frequency stimulation artifacts.
- Segmented into 0.5 s non-overlapping windows with 0–3rd order temporal delay feature augmentation.

### Disentanglement and Reconstruction Validation

| Metric | GPi | STN |
|--------|-----|-----|
| Shared GPi/STN CCA (median) | ≈1.0 | ≈1.0 |
| Shared–Private CCA (median) | 0.55–0.65 | 0.55–0.65 |
| Full (shared+private) reconstruction MSE | 0.00211 | 0.000983 |
| Private-only reconstruction MSE | 0.544 | 0.391 |
| Shared-only (same region) reconstruction MSE | 0.0462 | 0.0178 |

The shared subspace exhibits high cross-regional consistency (CCA ≈ 1.0), while shared–private correlation is only weak (0.55–0.65), indicating effective disentanglement. Reconstruction analysis reveals that the majority of recoverable neural dynamics reside in the shared manifold—using only private variables increases reconstruction error by two orders of magnitude, whereas using only shared variables yields results close to full reconstruction.

### Stimulation Frequency Decoding

Random forests decode stimulation frequency from various latent variables (4-class for GPi / 3-class for STN). Shared latent variables significantly outperform private latent variables in both decoding tasks ($p < 0.001$), with no significant difference between GPi-shared and STN-shared—indicating that the shared space encodes cross-regionally generalizable stimulation feature signatures, and that stimulation systematically reorganizes cross-regional coordination patterns in a frequency-dependent manner.

### Comparison with Baseline Methods

| Method | Nonlinear | Shared/Private Decomposition | LFP-Compatible | DBS Scenario | Reconstruction MSE |
|--------|-----------|------------------------------|----------------|--------------|-------------------|
| GPFA / CCA | ✗ | ✗ | ✓ | ✗ | — |
| DLAG | ✗ | ✓ | ✗ (spike) | ✗ (fails to converge) | Numerically unstable |
| SharedAE | ✓ | Partial (no temporal resolution) | ✗ | ✗ | Higher than SPIRE |
| MMVAE | ✓ | ✗ (no explicit disentanglement) | ✗ | ✗ | Higher than SPIRE |
| **SPIRE** | **✓** | **✓** | **✓** | **✓** | **Lowest** |

DLAG fails to converge on real intracranial data due to numerical instability in Gaussian process optimization. SPIRE achieves significantly lower reconstruction error than both SharedAE and MMVAE on GPi and STN.

## Highlights & Insights

- **First nonlinear shared-private decomposition framework** for human multi-region intracranial recordings, bridging the methodological gap between linear models and real nonlinear LFP data.
- **Elegant training paradigm**: training exclusively on baseline data prevents stimulation artifacts from contaminating model parameters while establishing a reference anchor for quantifying stimulation-induced reorganization. This "train on control, infer on treatment" paradigm has broad transferability.
- **Nine loss terms, each physically motivated**: the combination of VICReg alignment, orthogonal disentanglement, and ConvAlign temporal alignment is carefully designed; variance guards prevent degenerate solutions, and alignment module regularization ensures interpretability.
- The paper provides the first demonstration on pediatric DBS data that shared latent variables encode frequency-dependent network reorganization, offering quantitative evidence for the distributed network modulation theory of DBS.
- The shared-private disentanglement idea is transferable to multimodal AI models, such as decomposing modality-shared and modality-specific representations in vision-language models.

## Limitations & Future Work

- Validation is limited to shorter timescales of stimulation; long-term chronic stimulation effects and plasticity changes have not been examined.
- Only LFP signals are used; spike data and behavioral modalities are not integrated, and multimodal fusion is a natural direction for extension.
- The absence of a probabilistic objective (e.g., VAE ELBO) precludes uncertainty quantification over latent variables.
- Validation is currently limited to the two-region (GPi–STN) setting and has not been extended to multi-region scenarios including cortex, thalamus, and other structures.
- Shared latent variables are statistical abstractions; assigning precise biophysical interpretations requires complementary experiments and multimodal validation.
- The sample size is small (10 patients, 17 hemispheres); generalizability across etiologies and age groups requires validation on larger-scale datasets.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First application of nonlinear shared-private decomposition to human intracranial DBS recordings; the "train on baseline, infer under perturbation" paradigm is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Synthetic benchmarks + real clinical data + multiple baseline comparisons + disentanglement validation + decoding analysis, though sample size is limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, complete loss function definitions, and rich, intuitive figures.
- **Value**: ⭐⭐⭐⭐ — Substantive contribution to computational neuroscience and the mechanistic understanding of DBS; methodology is transferable to other multi-view dynamical systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] TDSNNs: Competitive Topographic Deep Spiking Neural Networks for Visual Cortex Modeling](../../AAAI2026/others/tdsnns_competitive_topographic_deep_spiking_neural_networks_for_visual_cortex_mo.md)
- [\[ICLR 2026\] LPWM: Latent Particle World Models for Object-Centric Stochastic Dynamics](latent_particle_world_models_self-supervised_object-centric_stochastic_dynamics_.md)
- [\[ICLR 2026\] Out of the Shadows: Exploring a Latent Space for Neural Network Verification](out_of_the_shadows_exploring_a_latent_space_for_neural_network_verification.md)
- [\[ICLR 2026\] Latent Fourier Transform](latent_fourier_transform.md)
- [\[ICLR 2026\] An Information-Theoretic Framework For Optimizing Experimental Design To Distinguish Probabilistic Neural Codes](an_information-theoretic_framework_for_optimizing_experimental_design_to_disting.md)

</div>

<!-- RELATED:END -->
