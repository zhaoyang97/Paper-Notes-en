---
title: >-
  [Paper Note] Functional MRI Time Series Generation via Wavelet-Based Image Transform and Spectral Flow Matching for Brain Disorder Identification
description: >-
  [ICLR 2026][Medical Imaging][Paper Note] DSFM converts fMRI BOLD time series into multi-scale time-frequency scalogram images via Discrete Wavelet Transform (DWT), compresses them into a low-frequency sparse domain via block DCT, and performs class-conditional generation using a "heat-diffusion-style" flow matching in the DCT domain. The synthesized signals a
tags:
  - ICLR 2026
  - Medical Imaging
date: 2026-05-08
content_hash: 9a7d5110fbc549cb
---
# Functional MRI Time Series Generation via Wavelet-Based Image Transform and Spectral Flow Matching for Brain Disorder Identification

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Dgphd9qizu](https://openreview.net/forum?id=Dgphd9qizu)  
**Code**: [https://github.com/htew0001/DSFM](https://github.com/htew0001/DSFM)  
**Area**: Medical Imaging / fMRI Generation / Brain Disorder Classification  
**Keywords**: fMRI Generation, BOLD Signal, Discrete Wavelet Transform (DWT), Discrete Cosine Transform (DCT), Flow Matching, Brain Network Classification  

## TL;DR
DSFM converts fMRI BOLD time series into multi-scale time-frequency scalogram images via Discrete Wavelet Transform (DWT), compresses them into a low-frequency sparse domain via block DCT, and performs class-conditional generation using a "heat-diffusion-style" flow matching in the DCT domain. The synthesized signals are then transformed back to the time domain for data augmentation, enhancing downstream brain functional connectivity (FC) classification performance.

## Background & Motivation
- **Background**: fMRI observes brain dynamics non-invasively through Blood-Oxygen-Level-Dependent (BOLD) signals and is a key modality for diagnosing neuropsychiatric disorders such as depression and autism. However, fMRI acquisition is expensive, leading to small datasets and class imbalances that restrict the generalization of data-driven brain analysis models. Consequently, generative models are employed for data augmentation.
- **Limitations of Prior Work**: ① Prevailing methods generate directly in the **functional connectivity (FC) matrix space** (e.g., DCGAN, BrainFC-CGAN), but FC compresses signal dependencies into a static correlation matrix, losing transient brain states and cross-frequency interactions. ② Methods shifting to **time-domain generation** (e.g., Diffusion-TS, FM-TS) struggle to decouple physiological fluctuations (heartbeat, respiration, motion artifacts) and fail to restore multi-scale oscillations. ③ T2I-Diff, which treats time series as **image generation**, uses fixed-resolution STFT, resulting in lost fine-grained transients and frequency modulation attenuation, which causes artifacts during signal reconstruction and limited gains.
- **Key Challenge**: To faithfully replicate the non-stationarity, spatio-temporal dynamics, and physiological variations of BOLD signals, a single representation of FC, raw time series, or fixed-resolution time-frequency maps is insufficient. A representation that characterizes both global multi-scale trends and local energy compression is required.
- **Goal**: Construct a dual-spectral representation that balances global and local features, and design an efficient generation process aligned with frequency hierarchies in the spectral domain to enhance downstream brain disorder classification.
- **Key Insight**: **[Dual-Spectral Cascade]** Use DWT to capture global multi-scale transients and block DCT to capture local low-frequency energy compression. **[Spectral Flow Matching]** Map the frequency-domain autoregressive property of diffusion models—where high frequencies are removed before low frequencies—to a heat diffusion process in the DCT domain, using ODE flow matching for efficient, coarse-to-fine generation aligned with frequency hierarchies.

## Method

### Overall Architecture
DSFM (Dual-Spectral Flow Matching) is a six-step pipeline: extracting BOLD time series from ROIs → multi-resolution DWT decomposition into time-scale scalogram images → block 2D DCT for local spectral encoding → learning velocity fields via U-ViT and ODE sampling in the DCT domain for class-conditional samples → inverse transformation (IDCT+IDWT) back to time-domain BOLD signals → data augmentation for FC construction and classifier evaluation. The core idea is reshaping "time-series generation" into "dual-spectral image generation" to integrate structured frequency priors throughout the process.

```mermaid
flowchart LR
    A[ROI BOLD Time Series<br/>D×T] --> B[DWT Multi-resolution<br/>Time-scale Scalogram]
    B --> C[Block 2D DCT<br/>Local Low-freq Compression]
    C --> D[Zig-zag Flattening<br/>Low→High Freq Sorting]
    D --> E[Spectral Flow Matching<br/>U-ViT Velocity Field + ODE Sampling]
    E --> F[IDCT + IDWT<br/>Inverse Transform to BOLD]
    F --> G[Augmentation → FC Construction → Classification]
```

### Key Designs

**1. Dual-Spectral Image Transform (DWT → Block DCT): Mapping Time Series to Global-Local Frequency Maps.** Given $x_s \in \mathbb{R}^{D\times T}$ ($D$ ROIs, $T$ time points), Discrete Wavelet Transform $W(k,j)=\sum_n x(n)\,\psi_{j,k}[n]$ (using Haar basis, 5-level decomposition) decomposes each BOLD signal into multi-scale subbands. These subbands are upsampled to the original duration and stacked to form a tensor $W(i,j,k)\in\mathbb{R}^{D\times T_\psi\times C}$, capturing both low-frequency trends and high-frequency transients. Component-wise normalization is applied to enhance contrast. Each subband is then segmented into non-overlapping $B\times B$ blocks, and a 2D type-II DCT is applied: $D^{(k)}(u,v)=\alpha(u)\alpha(v)\sum_{x,y}W^{(k)}(x,y)\cos\!\frac{(2x+1)u\pi}{2B}\cos\!\frac{(2y+1)v\pi}{2B}$. This leverages DCT's low-frequency energy compression to preserve primary structures while filtering high-frequency noise.

**2. DCT-Domain Spectral Flow Matching: Efficient, Coarse-to-Fine Frequency-Aligned Generation.** Diffusion models act as frequency-domain autoregressors—the forward process removes high frequencies before low frequencies. DSFM implements this via a heat diffusion SPDE $\mathrm{d}x_t(c)=\eta(t)\Delta_c x_t(c)\,\mathrm{d}t+G(t)\,\mathrm{d}W(t)$. Using the fact that the Laplacian operator is diagonalized by the DCT basis $\Delta_c=V\Lambda V^T$, the process is transformed into the DCT domain as a mode-wise decoupled equation $\mathrm{d}z_t=-\eta(t)\Lambda z_t\,\mathrm{d}t+G(t)\,\mathrm{d}W(t)$. The reverse-time probability flow ODE is then decomposed as $\frac{\mathrm{d}z_t[k]}{\mathrm{d}t}=-\eta(t)\lambda_k z_t[k]-\frac12 g(t,k)^2\nabla_{z_t[k]}\log p(z_t)$. By establishing equivalence between this ODE and the conditional velocity field $v(z_t|z_0;t,k)=\dot\mu(t,k)z_0[k]+\dot\sigma(t,k)\epsilon$, the model is trained by minimizing the Conditional Spectral Flow Matching loss $\mathcal{L}_{\text{CSFM}}(\theta)=\mathbb{E}\,\lVert v_\theta(z_t;t,k)-v(z_t|z_0;t,k)\rVert^2$.

**3. Class-Conditional Generation + U-ViT Velocity Field: Enhancing Clinical Discriminability.** The velocity field $v_\theta$ is parameterized by a U-ViT. Class labels $c$ (Healthy Control/Patient) are injected via classifier-free guidance. During training, $c$ is replaced by a null token $\varnothing$ with probability $p_\varnothing$. During sampling, an adaptive ODE solver integrates the velocity field to generate DCT samples. Synthesized BOLD signals are reconstructed via inverse transforms, and FC matrices are built using Ledoit-Wolf shrinkage (retaining the top 40% strongest connections) to feed the downstream classifier.

## Key Experimental Results

### Main Results
- **Datasets**: NetSim (Simulated, 50 channels), MDD (REST-meta-MDD, 250 HC / 227 MDD, AAL 116 ROI), ABIDE (488 ASD / 537 NC, Schaefer 100 ROI).
- **Unconditional Generation (NetSim, lower is better)**: DSFM achieved a cFID of **0.105±.006** (compared to Diffusion-TS at 0.193 and T2I-Diff at 1.384).

**Downstream Classification (MDD, AAL Atlas)**

| Method | Context-FID ↓ | Accuracy ↑ | F1 ↑ | ROC ↑ |
|---|---|---|---|---|
| Real (No Aug) | — | 58.90 | 58.39 | 59.00 |
| 2D-DCGAN (FC) | — | 62.88 | 62.48 | 62.67 |
| TimeGAN | 4.98 | 66.78 | 66.48 | 67.26 |
| Diffusion-TS | 2.06 | 67.29 | 67.21 | 64.57 |
| T2I-Diff | 7.45 | 66.87 | 66.83 | 67.26 |
| **DSFM (Ours)** | **1.51** | **70.84** | **70.77** | **71.49** |

**Downstream Classification (ABIDE, Schaefer Atlas)**

| Method | Context-FID ↓ | Accuracy ↑ | F1 ↑ | ROC ↑ |
|---|---|---|---|---|
| Real (No Aug) | — | 64.67 | 64.12 | 67.28 |
| Diffusion-TS | 0.51 | 66.60 | 66.58 | 68.85 |
| T2I-Diff | 0.82 | 69.69 | 69.65 | 71.88 |
| **DSFM (Ours)** | **0.07** | **71.54** | **70.98** | **73.78** |

### Ablation Study (MDD Frequency Subband Analysis)

| Setting | Subbands Used | Accuracy ↑ | ROC Drop |
|---|---|---|---|
| Full-band | LH1–LH5 + LL | 70.84 | — |
| Low-pass | LH3–5 + LL | 66.89 | -7.97% |
| Mid-pass | LH1,2,5 + LL | 63.30 | -15.50% |
| High-pass | LH1–4 | 65.40 | -11.0% |

### Key Findings
- **Consistent Gains from Dual-Spectral Representation**: DSFM outperforms FC-based, time-domain, and time-frequency image-based baselines in both generative fidelity (FID/cFID) and classification metrics across all datasets.
- **Criticality of All Frequency Bands**: Removing any subband leads to performance degradation. The "Mid-pass" setting (keeping only some mid and low frequencies) showed the largest drop (ROC -15.5%), indicating that joint full-band modeling is essential for brain disorder identification.
- **Sampling Efficiency**: The ODE flow matching generates high-quality samples in 20–100 steps, avoiding the thousand-step overhead of traditional diffusion SDEs.

## Highlights & Insights
- **Reshaping "Time-Series Generation"**: Combining DWT for global multi-scale structures and block DCT for local compression imposes a structured frequency prior that is more nuanced than a standard STFT.
- **Theoretical Bridge**: The use of Laplacian diagonalization via the DCT basis to decouple the heat diffusion SPDE into mode-wise ODEs provides a rigorous flow-matching implementation for the "frequency autoregressive" property.
- **Clinical Utility**: The improvement is verified not just by FID scores but by direct gains in downstream clinical classification, validating the goal of generating data for better diagnosis.

## Limitations & Future Work
- **Heuristic Parameters**: The choice of Haar basis, 5-level decomposition, and block size $B$ are manually set; robustness across different acquisition protocols and atlases needs further validation.
- **Data Scale**: Experiments were conducted on datasets with hundreds of cases. Multi-site heterogeneity and multi-class/multi-disorder generalization have not been fully explored.
- **Physiological Interpretability**: While emphasizing physiological dynamics, the model has not yet systematically verified the realism of specific components like cardiac or respiratory signals in the generated output.

## Related Work & Insights
- **FC Space Generation**: DCGAN and BrainFC-CGAN preserve connectome structures but lose transient network states.
- **Time-Domain Generation**: Diffusion-TS and FM-TS focus on raw sequences but struggle with physiological decoupling.
- **Frequency-Domain Diffusion**: This work capitalizes on the observation that diffusion functions as a frequency-domain autoregressor, explicitly encoding the "frequency hierarchy = generation order" prior into the generative dynamics.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The combination of DWT+DCT dual-spectral cascade and DCT-domain heat diffusion flow matching is a first in fMRI generation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers simulation and two real brain disorder datasets with comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, logical progression of formulas, and comprehensive pipeline diagrams.
- **Value**: ⭐⭐⭐⭐ — Directly addresses fMRI data scarcity with measurable improvements in clinical classification.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
</div>
<!-- RELATED:END -->

## Related Papers

- [\[ICLR 2026\] Learning Patient-Specific Disease Dynamics with Latent Flow Matching for Longitudinal Imaging Generation](learning_patient-specific_disease_dynamics_with_latent_flow_matching_for_longitu.md)
- [\[CVPR 2026\] Continual Learning for fMRI-Based Brain Disorder Diagnosis via Functional Connectivity Matrices Generative Replay](../../CVPR2026/medical_imaging/forge_continual_learning_for_fmri_based_brain_disorder_diagnosis.md)
- [\[ICLR 2026\] CRONOS: Continuous time reconstruction for 4D medical longitudinal series](cronos_continuous_time_reconstruction_for_4d_medical_longitudinal_series.md)
- [\[NeurIPS 2025\] Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data](../../NeurIPS2025/medical_imaging/self-supervised_learning_via_flow-guided_neural_operator_on_time-series_data.md)
- [\[AAAI 2026\] Ambiguity-aware Truncated Flow Matching for Ambiguous Medical Image Segmentation](../../AAAI2026/medical_imaging/ambiguity-aware_truncated_flow_matching_for_ambiguous_medica.md)

</div>

<!-- RELATED:END -->
