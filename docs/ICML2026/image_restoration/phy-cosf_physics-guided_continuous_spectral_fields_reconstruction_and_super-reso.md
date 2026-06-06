---
title: >-
  [Paper Note] Phy-CoSF: Physics-Guided Continuous Spectral Fields Reconstruction and Super-Resolution for Snapshot Compressive Imaging
description: >-
  [ICML 2026][Image Restoration][CASSI] A two-stage train-render deep unfolding framework for Coded Aperture Snapshot Spectral Imaging (CASSI) is proposed, allowing arbitrary wavelength queries. Within each unfolding stage…
tags:
  - "ICML 2026"
  - "Image Restoration"
  - "CASSI"
  - "Hyperspectral Reconstruction"
  - "Deep Unfolding Network"
  - "Implicit Neural Representation"
  - "Continuous Spectral Super-resolution"
date: 2026-05-08
content_hash: 2bb143e5aae96308
---

# Phy-CoSF: Physics-Guided Continuous Spectral Fields Reconstruction and Super-Resolution for Snapshot Compressive Imaging

**Conference**: ICML 2026  
**arXiv**: [2605.13583](https://arxiv.org/abs/2605.13583)  
**Code**: [github.com/PaiDii/Phy-CoSF](https://github.com/PaiDii/Phy-CoSF)  
**Area**: Image Restoration / Hyperspectral Imaging / Implicit Neural Representation  
**Keywords**: CASSI, Hyperspectral Reconstruction, Deep Unfolding Network, Implicit Neural Representation, Continuous Spectral Super-resolution

## TL;DR
A two-stage train-render deep unfolding framework for Coded Aperture Snapshot Spectral Imaging (CASSI) is proposed, allowing arbitrary wavelength queries. Within each unfolding stage, a Continuous Spectral Field (CoSF) prior module—comprising a Fourier-Mamba-driven triple-branch cross-domain feature mixer, random frequency encoding, and a spectral synthesis head—is embedded. While trained on discrete wavelengths, the model can synthesize hyperspectral images (HSI) at any continuous wavelength during inference, achieving continuous spectral reconstruction and zero-shot spectral super-resolution.

## Background & Motivation

**Background**: CASSI systems compress 3D HSI into a single snapshot via a physical mask, a disperser, and a 2D sensor. Reconstructing the full HSI from a single frame is a severely ill-posed inverse problem. Leading approaches have evolved from model-driven priors (sparsity, low-rank) to E2E CNN/Transformers (TSA-Net, MST++) and currently to Deep Unfolding Networks (DUNs) like ADMM-Net, GAP-Net, DAUHST, DERNN-LNLT, LADE-DUN, and MiJUN, which combine physical interpretability with data-driven performance.

**Limitations of Prior Work**: All mainstream methods (whether E2E or DUN) rely on a **fixed discrete wavelength** input-output assumption: they are bound to specific channels (e.g., 28) during both training and inference. However, the physical principle of CASSI is continuous dispersion. This "discrete-to-discrete" setting contradicts the physical nature of light and precludes high-value capabilities such as "inference at new wavelengths" or "spectral super-resolution." Expanding to new wavelengths requires re-collecting training data and retraining the entire model.

**Key Challenge**: The strength of DUNs stems from learning denoiser/deblurring operators for discrete channels at each stage. To enable continuous wavelength queries, the prior itself must be a continuous function of wavelength. A critical challenge lies in embedding the "coordinate-based query" capability of Implicit Neural Representations (INR) into an unfolding network without compromising physical consistency.

**Goal**: (1) Enable a single model to perform both high-fidelity HSI reconstruction and spectral super-resolution at arbitrary target wavelengths; (2) Retain the physically interpretable structure of DUN (via A-HQS algorithm unfolding) while replacing discrete prior modules with a decoupled "wavelength-independent content + continuous spectral synthesis" form; (3) Fully exploit the complementary structures of HSI in spatial, frequency, and channel domains.

**Key Insight**: The authors recognize that spectral synthesis is fundamentally identical to "querying color by coordinates" in NeRF. By using a wavelength-independent content representation $f$ and a continuous wavelength embedding $e_\lambda$ for implicit decoding in each DUN stage, the model can be trained on discrete samples and rendered at any $\lambda$.

**Core Idea**: Transforming DUN into a **train-render two-stage paradigm**. The training phase queries discrete wavelengths with Ground Truth (GT) to calculate $L1$ loss, while the rendering phase allows the same model to freely query any continuous wavelength for "zero-shot spectral super-resolution."

## Method

### Overall Architecture
Phy-CoSF unfolds the A-HQS algorithm into $K$ stages. Given the **forward physical model** $y = \Phi x + n$, each stage executes three steps: (i) **Data Fidelity Sub-problem** $x_{k+1} = (\Phi^T \Phi + \mu I)^{-1}(\Phi^T y + \mu \hat z_k)$, explicitly calculated by a degradation-aware network (DAN) where the physical mask $\Phi$ enters the computation graph; (ii) **Prior Sub-problem** $z_{k+1} = \text{CoSF}(x_{k+1}, \eta)$ performed by the CoSF module, with $\eta = \sqrt{\tau/\mu}$ as a learnable noise level; (iii) **Acceleration Step** $\hat z_{k+1} = z_{k+1} + \beta_k(z_{k+1} - z_k)$. The training phase queries a set of randomly sampled discrete wavelengths, and the rendering phase feeds arbitrary wavelengths $\lambda$ into the spectral synthesis head to render single-wavelength slices $HSI(\lambda) \in \mathbb{R}^{1\times H\times W}$.

### Key Designs

1.  **Continuous Spectral Field Prior Module (CoSF)**:
    - **Function**: Replaces the discrete fixed-channel priors in traditional DUNs, making the prior a continuous field regarding wavelength.
    - **Mechanism**: The prior is split into two parts. The **Triple-Branch Cross-Domain Feature Mixer** extracts a wavelength-independent multi-scale content representation $f \in \mathbb{R}^{C \times H \times W}$. It uses $3\times 3$ convolutions to obtain fine-grained features $f_H$, followed by downsampling and CDFE processing to get meso-scale $f_M$ and coarse-scale $f_L$. Branches are interpolated back to the original resolution and concatenated. The **Spectral Synthesis Head (SSH)** normalizes target $\lambda$ to $[-1, 1]$, applies random frequency encoding $\gamma(\lambda) = [\sin(2\pi\lambda b_1), \dots, \cos(2\pi\lambda b_m)]$ (fixed $b_i \sim \mathcal{N}(0,\sigma^2)$), and projects it via MLP to $e_\lambda \in \mathbb{R}^D$. This is concatenated with $f$ and synthesized into an intensity map $HSI(\lambda) = \text{SH}(\text{Concat}(e_\lambda, f))$ through convolutions.
    - **Design Motivation**: (a) Triple-branch multi-scale captures local textures and global contexts simultaneously; (b) Random Fourier Encoding (RFE) provides a "high-frequency inductive bias" to counter the low-frequency preference of deep networks; (c) The decoupling of content and wavelength-dependent embeddings is the physical basis for continuous queries.

2.  **Cross-Domain Feature Encoder (CDFE) (Spatial $\rightarrow$ Frequency $\rightarrow$ Channel)**:
    - **Function**: Refines HSI features complementarily across three domains using a single sequence backbone, overcoming the local bias of convolutions.
    - **Mechanism**: CDFE is a serial structure. **Spatial Domain**: Uses GLAM-Net (global-local attention mechanism) for local textures, $f_{spatial} = f_{in} + \text{GLAM}(f_{in})$. **Frequency Domain**: Employs a 2D-FFT to map $f_{spatial}$ to the frequency domain, where every coefficient contains global structural info. This is flattened and fed into a Mamba block for long-range dependency modeling. **Channel Domain**: Uses a GDFN module for channel-wise recalibration.
    - **Design Motivation**: HSI spatial structure, spectral correlation, and cross-channel dependencies are naturally decoupled signals. Using Mamba instead of Transformer for frequency-domain modeling avoids $O(N^2)$ complexity, making it suitable for high-resolution HSI.

3.  **Train-Render Two-Stage Paradigm**:
    - **Function**: Enables discrete supervision during training and zero-shot arbitrary wavelength rendering during inference.
    - **Mechanism**: During training, discrete wavelengths corresponding to GT are randomly sampled, and CoSF queries these coordinates to compute the $L1$ reconstruction loss $\mathcal{L}_{rec}$. Although the model only sees $\lambda$ with GT, the SSH learns a continuous function of wavelength. During inference, this constraint is removed: arbitrary $\lambda$ (unseen during training) can be fed into the SSH to obtain high-fidelity spectral slices.
    - **Design Motivation**: achieves continuous rendering with minimal training changes while avoiding dependence on rare and hard-to-acquire ground truth for spectral super-resolution.

### Loss & Training
The model is trained using $L1$ reconstruction loss $\mathcal{L}_{rec} = \|HSI_{pred}(\lambda) - HSI_{gt}(\lambda)\|_1$, with discrete wavelengths sampled per batch. The number of unfolding stages is $K=9$. The Fourier-Mamba within CDFE uses direction-independent 1D Mamba with a sequence length of $H \times W$. Evaluation is conducted on 10 scenes from the ICVL dataset using SAM, PSNR, and SSIM.

## Key Experimental Results

### Main Results
Continuous spectral reconstruction (Comparison under uniform discrete wavelength settings):

| Method | Params (M) | FLOPs (G) | Avg SAM ↓ | Avg PSNR (dB) ↑ | Avg SSIM ↑ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| MST++ | 0.07 | 1.18 | 2.43 | 34.48 | 0.884 |
| CST-L+ | 0.15 | 3.94 | 2.41 | 34.39 | 0.882 |
| GAP-Net | 4.21 | 65.73 | 2.38 | 36.01 | 0.915 |
| DAUHST-9stg | 2.42 | 6.68 | 2.32 | 35.76 | 0.911 |
| RDLUF-MixS2-9stg | 0.11 | 31.49 | 2.47 | 35.03 | 0.900 |
| DERNN-LNLT*-9stg | 0.93 | 122.14 | 2.33 | 35.72 | 0.911 |
| LADE-DUN-10stg | 1.23 | 8.34 | 2.16 | 35.79 | 0.914 |
| MiJUN-9stg | 0.04 | 6.01 | 2.37 | 35.26 | 0.901 |
| **Phy-CoSF-9stg** | 0.27 | 801.38 | **1.14** | **36.45** | **0.915** |

Phy-CoSF achieves a SAM of 1.14 (nearly half that of the strongest baseline LADE-DUN at 2.16) and a PSNR of 36.45 dB. However, its FLOPs reach 801G, an order of magnitude higher than other unfolding networks.

### Ablation Study

| Configuration | Key Metrics (Avg) | Description |
| :--- | :--- | :--- |
| Full Phy-CoSF-9stg | SAM 1.14 / PSNR 36.45 | All three modules included |
| w/o CoSF (Discrete Prior) | PSNR ~35.7 | Loses continuous rendering; drops 0.7 dB+ |
| w/o Fourier-Mamba | Significant drop in SSIM/PSNR | Lack of global dependency modeling |
| Single-scale vs Triple-branch | Loss of multi-scale details | Validates fine/meso/coarse necessity |
| Fixed Encoding vs RFE | Insufficient high-freq detail | RFE provides necessary inductive bias |

### Key Findings
- **Significant SAM Advantage**: SAM dropping from ~2.4 to 1.14 indicates halved spectral angular error, primarily because continuous $\lambda$ encoding allows SSH to accurately characterize spectral shapes.
- **Parameter Efficient but High FLOPs**: 0.27M parameters is low, but 801G FLOPs is high since SSH must be executed per stage and per wavelength—a hallmark of compute-intensive implicit representations.
- **Zero-shot Spectral Super-resolution**: This is achieved for the first time within a CASSI DUN framework, enabling the rendering of high-fidelity slices for $\lambda$ not present during training.
- **Fourier-Mamba Efficiency**: Using Mamba for 1D global signal modeling in the frequency domain avoids $O(N^2)$ attention costs, proving its suitability for high-resolution structured spectral signals.

## Highlights & Insights
- Combining INR with each stage of a DUN is a natural yet underrated synergy: DUN provides physical interpretability, while INR provides continuous query capability.
- The decoupled "wavelength-independent content + wavelength-dependent embedding" design is transferable to any inverse problem involving continuous coordinate queries (e.g., temporal video interpolation or spatial SR).
- Processing flattened 2D spectra via 1D Mamba effectively captures global context while controlling memory usage relative to Transformers.
- Seamlessly migrating NeRF’s RFE + MLP projection to 1D spectral coordinates validates the "implicit field" concept for spectral data.

## Limitations & Future Work
- FLOPs (801G) are significantly higher than baselines, making it less suitable for real-time applications.
- Spectral super-resolution is shown qualitatively; quantitative evaluation on "ground truth at new wavelengths" is missing.
- Evaluation is limited to ICVL; robustness across datasets like KAIST/CAVE or real hardware data remains to be seen.
- $L1$ loss might lead to over-smoothing; perceptual or spectral-angle losses could be explored.
- Rendering a full spectrum with 200+ bands requires sequential query iterations; batch rendering optimizations are needed.

## Related Work & Insights
- **vs LADE-DUN**: While LADE-DUN uses latent diffusion as a prior (PSNR 35.79), Phy-CoSF uses a continuous field prior (PSNR 36.45) with added rendering flexibility.
- **vs MiJUN**: MiJUN introduced Mamba to CASSI DUN via tensor mode-k unfolding; Phy-CoSF applies Mamba to the frequency domain with INR decoupling.
- **vs DERNN-LNLT**: While DERNN-LNLT compresses models via weight sharing, Phy-CoSF keeps parameters low while using INR to expand model functionality.
- **vs NeRF**: This work successfully extends the "coordinates $\rightarrow$ color" concept along the spectral axis rather than the spatial axis.

## Rating
- Novelty: ⭐⭐⭐⭐ (Successful integration of INR + DUN + Fourier-Mamba).
- Experimental Thoroughness: ⭐⭐⭐ (Strong comparisons but lacks multi-dataset and quantitative SR tests).
- Writing Quality: ⭐⭐⭐⭐ (Clear physical derivations and modular breakdown).
- Value: ⭐⭐⭐⭐ (Introduces "on-demand wavelength query" capability with potential for remote sensing and medical imaging).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Statistical Characteristic-Guided Denoising for Rapid High-Resolution Transmission Electron Microscopy Imaging](../../CVPR2026/image_restoration/statistical_characteristic-guided_denoising_for_rapid_high-resolution_transmissi.md)
- [\[CVPR 2026\] Spectral Super-Resolution via Adversarial Unfolding and Data-Driven Spectrum Regularization](../../CVPR2026/image_restoration/spectral_super-resolution_via_adversarial_unfolding_and_data-driven_spectrum_reg.md)
- [\[CVPR 2026\] UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution](../../CVPR2026/image_restoration/ucan_unified_convolutional_attention_lightweight_sr.md)
- [\[ICML 2026\] Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner](coevolutionary_continuous_discrete_diffusion_make_your_diffusion_language_model_.md)
- [\[ICML 2026\] Semi-Supervised Neural Super-Resolution for Mesh-Based Simulations](semi-supervised_neural_super-resolution_for_mesh-based_simulations.md)

</div>

<!-- RELATED:END -->
