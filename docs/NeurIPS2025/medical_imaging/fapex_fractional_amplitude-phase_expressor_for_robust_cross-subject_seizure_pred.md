---
title: >-
  [Paper Note] FAPEX: Fractional Amplitude-Phase Expressor for Robust Cross-Subject Seizure Prediction
description: >-
  [NEURIPS2025][Medical Imaging][seizure prediction] This paper proposes FAPEX, a framework that achieves adaptive time-frequency decomposition via a learnable Fractional Neural Frame Operator (FrNFO)…
tags:
  - "NEURIPS2025"
  - "Medical Imaging"
  - "seizure prediction"
  - "EEG"
  - "fractional Fourier transform"
  - "state-space model"
  - "phase-amplitude coupling"
date: 2026-05-08
content_hash: e2ecc9f5a751fcd5
---

# FAPEX: Fractional Amplitude-Phase Expressor for Robust Cross-Subject Seizure Prediction

**Conference**: NEURIPS2025 Spotlight  
**arXiv**: [2511.03263](https://arxiv.org/abs/2511.03263)  
**Code**: To be confirmed  
**Area**: Medical Imaging  
**Keywords**: seizure prediction, EEG, fractional Fourier transform, state-space model, phase-amplitude coupling  

## TL;DR
This paper proposes FAPEX, a framework that achieves adaptive time-frequency decomposition via a learnable Fractional Neural Frame Operator (FrNFO), combined with Amplitude-Phase Cross-Encoding (APCE) and Spatial Correlation Aggregation (SCA). FAPEX comprehensively outperforms 33 baseline methods across 12 cross-species, cross-modality seizure prediction benchmarks.

## Background & Motivation
- Epilepsy affects over 50 million people worldwide; accurate seizure prediction is critical for timely clinical intervention.
- Most existing methods are **subject-specific**, requiring large amounts of labeled data for each new patient and failing to generalize across patients, which severely hinders large-scale clinical deployment.
- **Subject-Agnostic Seizure Prediction (SASP)** is a clinically more valuable setting, yet it faces three core challenges:
    1. Conventional CNN/Transformer architectures exhibit **spectral bias**, favoring low-frequency components and struggling to capture high-frequency oscillations (HFOs) as key biomarkers.
    2. Seizures involve abnormal **phase-amplitude coupling (PAC)**, but existing models typically process time-domain and frequency-domain amplitude information separately, neglecting phase–amplitude interactions.
    3. Substantial inter-patient variability in electrode placement, implantation strategies, and brain region coverage leads to severe **channel heterogeneity**.

## Core Problem
How to design a unified, subject-agnostic seizure prediction model that generalizes robustly across species (human/rat/canine/macaque) and acquisition modalities (Scalp-EEG/SEEG/ECoG/LFP), while effectively capturing both high- and low-frequency biomarkers as well as phase-amplitude coupling relationships?

## Method

### Overall Architecture
FAPEX consists of three core modules: FrNFO backbone encoder → Amplitude-Phase Cross-Encoding (APCE) → Spatial Correlation Aggregation (SCA).

### 1. Input Patchification
- Multi-channel neural signals $\boldsymbol{X} \in \mathbb{R}^{C \times T}$ are segmented into non-overlapping patches of fixed length $\tau$.
- Each patch is projected into a $d_{\text{model}}$-dimensional feature space via a shared linear embedding.
- This design renders the model independent of electrode count and spatial arrangement.

### 2. Fractional Neural Frame Operator (FrNFO)
- **Core innovation**: Combines the Fractional Fourier Transform (FrFT) with learnable Weyl-Heisenberg frames.
- FrFT continuously interpolates between the time domain ($\theta=0$) and frequency domain ($\theta=\pi/2$) via the fractional parameter $\theta$, but conventional FrFT is constrained by fixed chirp functions and is sensitive to deformations.
- FrNFO addresses these limitations by:
    - Generating **adaptive window functions** via an implicit MLP parameterized with Hermite polynomials and sinusoidal activations.
    - Introducing **learnable fractional orders** $\boldsymbol{\theta} \in (0, \pi)^{d_{\text{model}}}$, where each feature channel independently controls time-frequency resolution.
    - Performing filtering in the fractional domain via the fractional convolution theorem: $\hat{\boldsymbol{X}}_{:,k} = \exp(-\pi i \omega^2 \cot\theta_k) \odot \mathcal{F}_{\theta_k}(\boldsymbol{X}_{:,k}) \odot \mathcal{F}_{\theta_k}(\boldsymbol{\Psi}_{:,k})$
- The output naturally decomposes into **amplitude** and **phase** components in complex-valued representation.
- Theoretical guarantee: From a scattering transform perspective, the amplitude representation of FrNFO is shown to possess provable robustness.

### 3. Amplitude-Phase Cross-Encoding (APCE)
- A bidirectional state-space model (Bidirectional SSM) is employed to construct a cross-attention mechanism.
- **Phase BSSM**: Takes phase embeddings as input while amplitude embeddings provide state-space parameters ($\boldsymbol{B}, \boldsymbol{C}$), capturing the modulation of amplitude by phase.
- **Amplitude BSSM**: Roles are reversed — amplitude serves as the query while phase provides context.
- Features are fused via residual connections, yielding representations that encode phase-amplitude coupling information.

### 4. Spatial Correlation Aggregation (SCA)
- Linear attention is used to model global cross-electrode dependencies with $O(C)$ complexity (linear in the number of channels).
- The feature map $\phi$ is implemented as a single-layer MLP: $\phi_{\text{MLP}}(\boldsymbol{x}) = \exp(\boldsymbol{W}_1^\top \boldsymbol{x})$
- Local spatiotemporal patterns are aggregated via a $3 \times 3$ depthwise convolution gating mechanism.

## Key Experimental Results

### Experimental Scale
- **12 benchmark datasets**: Covering 4 species (human, rat, canine, macaque) and 4 acquisition modalities (Scalp-EEG, SEEG, ECoG, LFP).
- **33 baseline methods**: 23 supervised + 10 self-supervised.
- Evaluation protocol: Subject-Agnostic Nested Cross-Validation (SANCV).

### Core Results (Supervised FAPEX-Base)

| Dataset | SEN | F1 | AUROC |
|--------|-----|-----|-------|
| Beirut (Scalp-EEG, Human) | 84.7 | 84.3 | 85.8 |
| Canine (ECoG, Canine) | 86.0 | 84.7 | 74.5 |
| FMCE (ECoG/SEEG, Human) | 88.8 | 90.7 | 97.2 |
| cTLE-RatLFP (LFP, Rat) | 81.8 | 83.2 | 91.2 |
| KAIME (EEG+SEEG, Macaque) | 87.0 | 95.6 | 90.1 |
| PCS (Scalp-EEG, Human) | 91.5 | 91.5 | 96.3 |

### Self-Supervised Pretraining Gains (FAPEX-Base SSL)
- After pretraining, F1 improves by a further 2–10 percentage points on most datasets.
- On the IESS dataset, F1 increases from 72.4 → 84.9, a gain of 12.5 percentage points.
- On the PCS dataset, F1 improves from 91.5 → 95.0.

### Cross-Domain Transfer
- In source-only transfer settings, FAPEX achieves relative F1 improvements typically exceeding 30% over Neuro-BERT and CBraMod.
- Even under target-domain-labeled CDAC/MME protocols, FAPEX matches or surpasses most baselines across scenarios.

## Highlights & Insights
1. **Theoretical rigor**: FrNFO is not merely an engineering innovation but possesses provable robustness from a scattering transform perspective.
2. **Exceptionally comprehensive experiments**: 12 datasets, 4 species, 33 baselines, and multiple transfer learning settings constitute one of the largest comparative studies in the seizure prediction field.
3. **Interpretability**: Visualization of FrNFO frequency responses shows progressive sub-band refinement with increasing depth, with both high- and low-frequency components effectively amplified, overcoming the low-frequency bias of conventional models.
4. **Compact architecture**: Rather than scaling model size, high performance is achieved through mathematically principled design.

## Limitations & Future Work
1. Channel alignment preprocessing maps all data to 64 channels; this hard constraint may discard original channel layout information.
2. Several proprietary datasets (AGS, ATLE, IESS, etc.) are not publicly reproducible.
3. The paper does not report detailed comparisons of inference latency or parameter counts, and discussion of feasibility for wearable device deployment is insufficient.
4. Ablation studies are only briefly mentioned in the appendix; the independent contributions of core modules require more transparent quantification.

## Related Work & Insights
- **vs. Medformer / TSLANet** (multi-scale temporal models): FAPEX exceeds F1 by 5–10 percentage points on the vast majority of datasets, with particularly notable advantages in cross-species scenarios.
- **vs. EEG foundation models (Neuro-BERT / CBraMod / LaBraM)**: FAPEX substantially outperforms these models even with identical pretraining data, indicating that gains stem from architectural design rather than pretraining scale.
- **vs. SeizureFormer** (epilepsy-specific model): FAPEX achieves 10–20 percentage point advantages in both SEN and F1.
- **vs. frequency-domain methods (FreTS / NFM / ATFNet)**: The fractional-order design of FrNFO significantly outperforms fixed-frequency basis transforms on non-stationary signals.

## Inspiration & Connections
- The learnable fractional-order time-frequency decomposition underlying FrNFO can be extended to other biosignal analysis tasks such as ECG and EMG.
- The SSM-based APCE architecture provides a paradigm for other signal processing tasks requiring joint modeling of amplitude and phase.
- The success of cross-species generalization implies that pre-ictal neural dynamics may share universal patterns across species.

## Rating
- **Novelty**: 9/10 — An innovative combination of fractional frame theory and deep learning, with solid theoretical foundations.
- **Experimental Thoroughness**: 10/10 — 12 datasets, 4 species, 33 baselines, and multiple transfer settings; exceptionally comprehensive.
- **Writing Quality**: 8/10 — Generally clear, but notation is mathematically dense with occasional inconsistencies.
- **Value**: 9/10 — A significant contribution to the seizure prediction field with high potential for clinical translation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Zebra: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding](zebra_towards_zero-shot_cross-subject_generalization_for_universal_brain_visual_.md)
- [\[ICLR 2026\] COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics](../../ICLR2026/medical_imaging/compass_robust_feature_conformal_prediction_for_medical_segmentation_metrics.md)
- [\[NeurIPS 2025\] MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding](more-brain_routed_mixture_of_experts_for_interpretable_and_generalizable_cross-s.md)
- [\[AAAI 2026\] MindCross: Fast New Subject Adaptation with Limited Data for Cross-subject Video Reconstruction from Brain Signals](../../AAAI2026/medical_imaging/mindcross_fast_new_subject_adaptation_with_limited_data_for_cross-subject_video_.md)
- [\[NeurIPS 2025\] EEGReXferNet: A Lightweight Gen-AI Framework for EEG Subspace Reconstruction via Cross-Subject Transfer Learning and Channel-Aware Embedding](eegrexfernet_a_lightweight_gen-ai_framework_for_eeg_subspace_reconstruction_via_.md)

</div>

<!-- RELATED:END -->
