---
title: >-
  [Paper Note] FDP: A Frequency-Decomposition Preprocessing Pipeline for Unsupervised Anomaly Detection in Brain MRI
description: >-
  [AAAI 2026][Medical Imaging][Unsupervised Anomaly Detection] This work presents the first systematic frequency-domain analysis of brain MRI anomalies, demonstrating that lesions are predominantly concentrated in low-frequency components. Based on this finding, the authors propose the **Frequency Decomposition Preprocessing (FDP)** framework, which reconstructs low-frequency signals via a learnable prior context bank to suppress lesions while preserving anatomical structures. As a plug-and-play module, FDP consistently improves detection performance across multiple UAD baselines (achieving a 17.63% DICE gain on LDM).
tags:
  - AAAI 2026
  - Medical Imaging
  - Unsupervised Anomaly Detection
  - Brain MRI
  - Frequency Domain Analysis
  - Frequency Decomposition
  - Diffusion Models
date: 2026-05-08
content_hash: bebef47f9d81664c
---

# FDP: A Frequency-Decomposition Preprocessing Pipeline for Unsupervised Anomaly Detection in Brain MRI

**Conference**: AAAI 2026
**arXiv**: [2511.12899](https://arxiv.org/abs/2511.12899)
**Code**: [github](https://github.com/ls1rius/MRI_FDP)
**Area**: Medical Imaging
**Keywords**: Unsupervised Anomaly Detection, Brain MRI, Frequency Domain Analysis, Frequency Decomposition, Diffusion Models

## TL;DR

This work presents the first systematic frequency-domain analysis of brain MRI anomalies, demonstrating that lesions are predominantly concentrated in low-frequency components. Based on this finding, the authors propose the **Frequency Decomposition Preprocessing (FDP)** framework, which reconstructs low-frequency signals via a learnable prior context bank to suppress lesions while preserving anatomical structures. As a plug-and-play module, FDP consistently improves detection performance across multiple UAD baselines (achieving a 17.63% DICE gain on LDM).

## Background & Motivation

### The Challenge of Unsupervised Anomaly Detection in Brain MRI

Due to the high diversity of brain anatomy and the scarcity of annotated data, supervised anomaly detection faces significant bottlenecks. The standard paradigm for existing unsupervised anomaly detection (UAD) methods is:
1. Train a generative model (VAE/DDPM/LDM, etc.) on healthy MRI scans to learn representations of normal anatomy
2. At inference time, input scans potentially containing anomalies; the model attempts to reconstruct a "normal" version
3. Detect anomalies via pixel-wise residuals between the original and reconstructed images

A key limitation is that the **artificially synthesized noise** used during training (e.g., random masking, Gaussian noise) lacks the **biophysical fidelity and morphological complexity** of real clinical lesions, restricting the model's generalization to genuine pathologies.

### Inspiration from a Frequency-Domain Perspective

MRI data is inherently acquired in the frequency domain (k-space), where each spatial frequency component encodes distinct structural information. Nevertheless, existing UAD methods operate almost exclusively in the spatial domain, overlooking the diagnostic potential of frequency-domain analysis. This paper presents the first systematic frequency-domain characterization of MRI anomalies.

### Two Key Observations

**Lesions are predominantly concentrated in low frequencies**: Applying high-pass filters with varying thresholds to lesion-containing MRI scans and computing DICE against ground truth reveals that DICE drops below 0.1 when the threshold $m$ reaches 0.2, indicating that lesion signals are primarily carried by low-frequency components.

**Low-frequency components of healthy scans exhibit high consistency**: Normal MRI scans show highly consistent real-part signal patterns in low-frequency regions ($m \leq 0.1$), whereas lesion-containing scans exhibit markedly greater dispersion in the same frequency band; both groups show comparable signal variability in high-frequency regions ($m > 0.1$).

## Method

### Overall Architecture

FDP consists of two main modules serving as a preprocessing pipeline upstream of the generative model:

**Training phase**: Healthy MRI → FFT to frequency domain → high-pass filtering to separate high-frequency $f_h$ and low-frequency $f_l$ → FRM reconstructs low-frequency components via a prior context bank → merge and IFFT back to spatial domain → feed into generative model for training; the high-frequency image $I_h$ simultaneously serves as an auxiliary structural prior (HFSup).

**Inference phase**: Lesion-containing MRI → same frequency decomposition → FRM reconstructs low-frequency components (lesion signals in the low-frequency band are replaced by prior context) → merge with high-frequency components and feed into generative model → produce a cleaner "healthy" reconstruction.

### Key Designs

#### 1. **Frequency Decomposition**

An ideal high-pass filter is applied to decompose the frequency-domain representation of an MRI image into high-frequency and low-frequency components:

$$H_{hp}(u,v) = \begin{cases} 0 & \text{if } D(u,v) \leq \mathfrak{D}_0 \\ 1 & \text{if } D(u,v) > \mathfrak{D}_0 \end{cases}$$

where $D(u,v) = \sqrt{(u-H/2)^2 + (v-W/2)^2}$, $\mathfrak{D}_0 = \min(m*H, m*W)$, and $m$ is the high-pass filter threshold.

The low-frequency component $f_l$ captures global anatomical structure (including lesion signals), while the high-frequency component $f_h$ captures fine details, edges, and textures (containing minimal lesion information).

Design Motivation: Since lesions are concentrated in the low-frequency band, reconstructing only the low-frequency component suffices to eliminate lesions while preserving anatomical detail.

#### 2. **Frequency Reconstruction Module (FRM)**

Core Idea: A learnable prior context bank $\boldsymbol{P} = [p_1, p_2, \dots, p_k]$, learned from a healthy MRI training set, is used to reconstruct the low-frequency signal.

- Prior context initialization: k-means++ clustering applied to low-frequency signals from the training set
- Reconstruction via an attention retrieval mechanism:

$$\hat{f_l} = \text{ATTN}(f_l, P, P)$$
$$\hat{f} = \text{MERGE}(\hat{f_l}, f_h)$$
$$\hat{I} = \text{IDFT}(\hat{f})$$

- L1 loss supervises low-frequency reconstruction: $L1(\hat{f_l}, f_l)$

Design Motivation: Verified through PCA, t-SNE, and maximum likelihood estimation, the low-frequency signals of healthy MRI scans exhibit low variance and approximately lie on a low-dimensional manifold, enabling effective approximation via linear combinations of prior context vectors. At inference time, lesion-containing low-frequency signals are "pulled back" onto the healthy distribution.

#### 3. **High-Frequency Supplement (HFSup)**

The high-frequency signal is transformed back to the spatial domain to obtain $I_h$, which is fed into the generative model as an auxiliary structural prior to enhance anatomical structure preservation and edge sharpness.

Design Motivation: High-frequency information alone is insufficient for MRI reconstruction (lacking global structure), but as a supplementary feature it enhances the structural preservation capability of the baseline model.

### Loss & Training

- FRM is trained with L1 loss for low-frequency reconstruction
- Generative models (e.g., LDM, VAE) retain their original training strategies
- Adam optimizer, learning rate 2e-5, batch size 32, 800 epochs
- 4 × NVIDIA V100 GPUs
- Default prior context size: 128; default $m$: 0.10

## Key Experimental Results

### Main Results

Results on the BraTS20 dataset (T2-weighted):

| Model | DICE | AUPRC | AUROC |
|------|------|-------|-------|
| VAE | 34.90 | 29.95 | 94.46 |
| **FDP + VAE** | **46.32 (+11.42)** | **41.32 (+11.37)** | 92.16 |
| LDM | 35.02 | 30.75 | 91.62 |
| **FDP + LDM** | **52.66 (+17.63)** | **51.67 (+20.92)** | 93.12 |
| AnoDDPM | 36.19 | 32.01 | 91.37 |
| **FDP + AnoDDPM** | **48.24 (+12.05)** | **47.56 (+15.55)** | 92.97 |
| DAE (coarse) | 56.87 | 43.23 | 95.71 |
| **FDP + DAE (coarse)** | **63.03 (+6.16)** | **61.41 (+18.18)** | 93.95 |
| pDDPM | 46.15 | 45.67 | 92.01 |
| **FDP + pDDPM** | **54.09 (+8.04)** | **51.03 (+5.36)** | 93.72 |

Cross-dataset generalization (LDM baseline):

| Dataset | LDM DICE | LDM+FDP DICE | Gain |
|--------|---------|-------------|------|
| BraTS21 | 29.53 | 45.06 | +15.53 |
| MSLUB | 8.35 | 13.06 | +4.71 |
| MSSEG-2 | 20.15 | 34.63 | +14.48 |

### Ablation Study

| FRM | HFSup | DICE | AUPRC | AUROC |
|-----|-------|------|-------|-------|
| ✗ | ✗ | 35.02 | 30.75 | 91.62 |
| ✗ | ✓ | 42.18 | 40.55 | 92.25 |
| ✓ | ✗ | 50.00 | 45.86 | 92.97 |
| ✓ | ✓ | **52.66** | **51.67** | **93.12** |

Sensitivity analysis of $m_{\text{FRM}}$:

| $m_{\text{FRM}}$ | DICE | AUPRC | AUROC |
|-----------------|------|-------|-------|
| 0.01 | 39.45 | 37.13 | 90.54 |
| 0.05 | 50.00 | 44.86 | 92.97 |
| **0.10** | **52.66** | **51.67** | **93.12** |
| 0.15 | 45.24 | 43.56 | 91.78 |
| 0.20 | 43.89 | 41.87 | 93.08 |

### Key Findings

1. **FRM is the core contribution**: FRM alone yields a +15.0 DICE improvement (35.02 → 50.00), with HFSup contributing an additional +2.66.
2. **Consistent improvements**: All 7 baseline methods integrated with FDP achieve at least a 6.16% DICE gain and at least a 5.36% AUPRC gain.
3. **$m=0.10$ is the optimal threshold**: Values that are too small (0.01) fail to effectively suppress lesions, while values that are too large (>0.15) result in overly dispersed low-frequency components that cannot be well approximated by the prior context bank.
4. **Limited AUROC improvement**: A marginal decrease is observed in certain methods, which the authors attribute to predicted regions being slightly smaller than the ground truth (high precision, slightly reduced recall).

## Highlights & Insights

1. **First systematic frequency-domain analysis**: The intuition that "MRI lesions are predominantly low-frequency" is experimentally validated and quantified.
2. **Plug-and-play design**: FDP is a preprocessing module that requires no modification to the downstream generative model architecture, functioning directly as a "frequency-domain lesion suppression" front-end.
3. **Elegant design of the prior context bank**: The low-dimensional manifold structure of healthy low-frequency signals is exploited, enabling low-frequency reconstruction via attention-based retrieval.
4. **Complementary use of frequency and spatial domains**: The strategy of low-frequency reconstruction combined with high-frequency preservation simultaneously eliminates lesions and preserves structural integrity.

## Limitations & Future Work

1. **Assumption that lesions are predominantly low-frequency**: Certain subtle texture-type anomalies (e.g., early demyelination changes) may partially manifest in the mid-frequency range.
2. **Ringing artifacts from ideal high-pass filtering**: Gaussian or Butterworth filters could be considered to mitigate such effects.
3. **Evaluation primarily on T2-weighted images**: Although generalization tests on FLAIR/T1 datasets are included, the main conclusions are based on T2-weighted data.
4. **Fixed prior context size**: Whether 128 prior context vectors remain sufficient for larger-scale or multi-protocol datasets warrants further investigation.
5. **Lack of analysis on small lesions**: The tendency for predicted regions to be smaller than ground truth may adversely affect the detection rate of small lesions.

## Rating

- Novelty: ⭐⭐⭐⭐ — First frequency-domain perspective on MRI UAD; observations are novel and the methodology is well-motivated
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 7 baselines, 4 datasets, detailed ablation and hyperparameter analysis
- Writing Quality: ⭐⭐⭐⭐ — Clear analysis, rich figures and tables
- Value: ⭐⭐⭐⭐ — Provides a general-purpose preprocessing enhancement strategy with strong practical utility

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Unsupervised Motion-Compensated Decomposition for Cardiac MRI Reconstruction via Neural Representation](unsupervised_motion-compensated_decomposition_for_cardiac_mri_reconstruction_via.md)
- [\[AAAI 2026\] Neural Bandit Based Optimal LLM Selection for a Pipeline of Tasks](neural_bandit_based_optimal_llm_selection_for_a_pipeline_of_tasks.md)
- [\[CVPR 2026\] InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](../../CVPR2026/medical_imaging/invad_inversion-based_reconstruction-free_anomaly_detection_with_diffusion_model.md)
- [\[AAAI 2026\] WDT-MD: Wavelet Diffusion Transformers for Microaneurysm Detection in Fundus Images](wdt-md_wavelet_diffusion_transformers_for_microaneurysm_detection_in_fundus_imag.md)
- [\[CVPR 2026\] Virtual Full-stack Scanning of Brain MRI via Imputing Any Quantised Code](../../CVPR2026/medical_imaging/virtual_full-stack_scanning_of_brain_mri_via_imputing_any_quantised_code.md)

<!-- RELATED:END -->
