---
title: >-
  [Paper Note] Phy-CoSF: Physics-Guided Continuous Spectral Fields Reconstruction and Super-Resolution for Snapshot Compressive Imaging
description: >-
  [ICML 2026][Scientific Computing][CASSI] A train-render two-phase deep unfolding framework for snapshot compressive spectral imaging (CASSI)…
tags:
  - "ICML 2026"
  - "Scientific Computing"
  - "CASSI"
  - "Hyperspectral Reconstruction"
  - "Deep Unfolding Network"
  - "Implicit Neural Representation"
  - "Continuous Spectral Super-Resolution"
date: 2026-05-08
content_hash: b021b857cfde3c78
---

# Phy-CoSF: Physics-Guided Continuous Spectral Fields Reconstruction and Super-Resolution for Snapshot Compressive Imaging

**Conference**: ICML 2026  
**arXiv**: [2605.13583](https://arxiv.org/abs/2605.13583)  
**Code**: [github.com/PaiDii/Phy-CoSF](https://github.com/PaiDii/Phy-CoSF)  
**Area**: Image Restoration / Hyperspectral Imaging / Implicit Neural Representation  
**Keywords**: CASSI, Hyperspectral Reconstruction, Deep Unfolding Network, Implicit Neural Representation, Continuous Spectral Super-Resolution

## TL;DR
A train-render two-phase deep unfolding framework for snapshot compressive spectral imaging (CASSI), enabling arbitrary wavelength querying. Each unfolding stage incorporates a continuous spectral field (CoSF) prior module, consisting of a Fourier-Mamba-driven triple-branch cross-domain feature mixer, random frequency encoding, and a spectral synthesis head. Training on discrete wavelengths enables inference at any continuous wavelength, achieving continuous spectral reconstruction and zero-shot spectral super-resolution.

## Background & Motivation

**Background**: The CASSI system compresses a 3D hyperspectral image (HSI) into a single snapshot via a physical mask, disperser, and 2D sensor. Reconstructing the full HSI from a single frame is a severely underdetermined inverse problem. Mainstream solutions have evolved from "model-driven priors (sparsity, low-rank) → E2E CNN/Transformer (TSA-Net, MST++) → Deep Unfolding Networks (DUN: ADMM-Net, GAP-Net, DAUHST, DERNN-LNLT, LADE-DUN, MiJUN)", with DUNs now dominant due to their balance of physical interpretability and data-driven learning.

**Limitations of Prior Work**: All mainstream methods (E2E or DUN) are based on the assumption of **fixed discrete wavelengths** for input and output: training is tied to 28 wavelength channels, and inference can only output these 28. However, CASSI's physical imaging is inherently continuous in wavelength. This "discrete in both training and inference" setup contradicts the physics and precludes valuable capabilities like "inference at new wavelengths" or "spectral super-resolution." Extending to new wavelengths requires new data collection and retraining the entire model.

**Key Challenge**: The strong prior in DUNs comes from "learning a denoising/deblurring operator for each discrete channel at each stage," but to enable continuous wavelength querying, the prior itself must be a continuous function of wavelength. The key challenge is how to embed the "coordinate-based arbitrary output querying" capability of implicit neural representations (INR) into the unfolding network without breaking physical consistency.

**Goal**: (1) Enable a single model to perform high-fidelity HSI reconstruction and spectral super-resolution at arbitrary target wavelengths; (2) Retain the physically interpretable structure of DUNs (A-HQS algorithm unfolding), but decouple the prior module into "wavelength-independent content + continuous spectral synthesis"; (3) Fully exploit the complementary structure of HSI in spatial, frequency, and channel domains.

**Key Insight**: The authors recognize that spectral synthesis is essentially the same as "coordinate-based color querying" in NeRF—by using "wavelength-independent content representation $f$" plus "continuous wavelength embedding $e_\lambda$" for implicit decoding at each DUN stage, the model can be trained on discrete wavelengths and render at arbitrary $\lambda$ during inference.

**Core Idea**: Transform the DUN into a **train-render two-phase paradigm**—the training phase queries only at discrete wavelengths with ground truth and computes L1 loss; the rendering phase allows the same model to freely query any continuous wavelength, achieving "zero-shot spectral super-resolution."

## Method

### Overall Architecture
Phy-CoSF unfolds the A-HQS algorithm (accelerated half-quadratic splitting) into $K$ stages. The **forward physical model** is $y = \Phi x + n$. Each stage performs three steps: (i) **Data fidelity subproblem** $x_{k+1} = (\Phi^T \Phi + \mu I)^{-1}(\Phi^T y + \mu \hat z_k)$ is explicitly computed by a DAN (degradation-aware network), with the physical mask $\Phi$ included in the computation graph; (ii) **Prior subproblem** $z_{k+1} = \text{CoSF}(x_{k+1}, \eta)$ is handled by the CoSF module, where $\eta = \sqrt{\tau/\mu}$ is a learnable noise level; (iii) **Acceleration step** $\hat z_{k+1} = z_{k+1} + \beta_k(z_{k+1} - z_k)$. The training phase queries a randomly sampled set of discrete wavelengths and computes L1 loss; the inference phase feeds any wavelength $\lambda$ into the spectral synthesis head in CoSF to render a single-wavelength slice $HSI(\lambda) \in \mathbb{R}^{1\times H\times W}$.

### Key Designs

1. **Continuous Spectral Field Prior Module (CoSF)**:

    - **Function**: Replaces the traditional discrete prior trained for fixed channels in DUNs, making the prior itself a continuous field over wavelength.
    - **Mechanism**: The prior is split into two parts. The **Triple-Branch Cross-Domain Feature Mixer** extracts a "wavelength-independent" multi-scale content representation $f \in \mathbb{R}^{C \times H \times W}$: a $3\times 3$ convolution increases channels to obtain fine-grained features $f_H \in \mathbb{R}^{C/12 \times H \times W}$; a $4\times 4$ convolution with downsampling and CDFE yields meso-scale $f_M \in \mathbb{R}^{C/6 \times H/2 \times W/2}$ and coarse-scale $f_L \in \mathbb{R}^{C/3 \times H/4 \times W/4}$, each branch processed by CDFE. After upsampling to the original resolution and $1\times 1$ refinement convolution, channel concatenation yields $f$. The **Spectral Synthesis Head (SSH)** normalizes the target $\lambda$ to $[-1, 1]$, applies random frequency encoding $\gamma(\lambda) = [\sin(2\pi\lambda b_1), \dots, \cos(2\pi\lambda b_m)]$ ($b_i \sim \mathcal{N}(0,\sigma^2)$ fixed), and MLP projection to obtain $e_\lambda \in \mathbb{R}^D$. Concatenated with $f$, two $3\times 3$ and one $1\times 1$ convolutions synthesize the intensity map for wavelength $HSI(\lambda) = \text{SH}(\text{Concat}(e_\lambda, f))$.
    - **Design Motivation**: (a) The three-branch multi-scale structure captures local textures, meso-scale structures, and global context; (b) Random Fourier encoding provides a "high-frequency inductive bias" to counteract the low-frequency bias of deep networks; (c) The decoupling of "wavelength-independent content + wavelength-dependent embedding" is the physical basis for continuous querying.

2. **Cross-Domain Feature Encoder CDFE (Spatial → Frequency → Channel)**:

    - **Function**: Uses a single sequential backbone to complementarily refine HSI features across three domains, compensating for the locality of convolution.
    - **Mechanism**: CDFE is a three-stage structure. **Spatial domain**: GLAM-Net (global-local attention mechanism) extracts local texture details, $f_{spatial} = f_{in} + \text{GLAM}(f_{in})$. **Frequency domain** (novelty of this work): 2D-FFT maps $f_{spatial}$ to the frequency domain, where each coefficient contains global structure; flatten to a 1D sequence and input to a Mamba block for long-range modeling $f_{freq} = f_{spatial} + i\text{FFT}(\text{Mamba}(\text{FFT}(f_{spatial})))$; finally, iFFT back to spatial domain with residual connection. **Channel domain**: GDFN module recalibrates by channel, $f_{out} = f_{freq} + \text{GDFN}(f_{freq})$. Feature dimensions are consistent throughout for residual connections.
    - **Design Motivation**: The spatial structure, spectral correlation, and cross-channel relationships in HSI are naturally decoupled signals; using Mamba instead of Transformer for long-range modeling in the frequency domain avoids $O(N^2)$ attention overhead, especially suitable for high-resolution hyperspectral scenarios.

3. **Train-Render Two-Phase Paradigm**:

    - **Function**: Enables training under discrete supervision and inference with arbitrary wavelength zero-shot rendering.
    - **Mechanism**: In training, a random set of ground-truth discrete wavelengths is sampled, and CoSF queries only these coordinates to obtain slices, computing L1 reconstruction loss $\mathcal{L}_{rec}$ to update the network. The model thus only "sees" ground-truth $\lambda$, but since SSH takes $\lambda$ as a continuous input condition, it essentially learns a continuous function over wavelength. In inference, this restriction is lifted: any $\lambda$ can be fed to SSH to obtain a high-fidelity spectral slice, achieving super-resolution in the spectral direction.
    - **Design Motivation**: Achieve continuous rendering with minimal training changes (only change query coordinates); also avoids reliance on real spectral super-resolution data (which is extremely hard to obtain).

### Loss & Training
Training uses L1 reconstruction loss $\mathcal{L}_{rec} = \|HSI_{pred}(\lambda) - HSI_{gt}(\lambda)\|_1$, with a random set of discrete wavelengths sampled per batch. The number of unfolding stages $K = 9$, each stage contains 1 DAN + 1 CoSF + acceleration update. The Fourier-Mamba block in CDFE is a direction-agnostic 1D Mamba, with input sequence length equal to the flattened frequency domain $H \times W$. All evaluations are on 10 scenes from the ICVL dataset, with metrics SAM (spectral angle), PSNR, and SSIM.

## Key Experimental Results

### Main Results
Continuous spectral reconstruction (compared with various mainstream DUN/E2E methods under the same discrete wavelength setting):

| Method | Params (M) | FLOPs (G) | Avg SAM ↓ | Avg PSNR (dB) ↑ | Avg SSIM ↑ |
|---|---|---|---|---|---|
| MST++ | 0.07 | 1.18 | 2.43 | 34.48 | 0.884 |
| CST-L+ | 0.15 | 3.94 | 2.41 | 34.39 | 0.882 |
| GAP-Net | 4.21 | 65.73 | 2.38 | 36.01 | 0.915 |
| DAUHST-9stg | 2.42 | 6.68 | 2.32 | 35.76 | 0.911 |
| RDLUF-MixS2-9stg | 0.11 | 31.49 | 2.47 | 35.03 | 0.900 |
| DERNN-LNLT*-9stg | 0.93 | 122.14 | 2.33 | 35.72 | 0.911 |
| LADE-DUN-10stg | 1.23 | 8.34 | 2.16 | 35.79 | 0.914 |
| MiJUN-9stg | 0.04 | 6.01 | 2.37 | 35.26 | 0.901 |
| **Phy-CoSF-9stg** | 0.27 | 801.38 | **1.14** | **36.45** | **0.915** |

Phy-CoSF achieves a SAM of only 1.14 (the strongest baseline LADE-DUN is 2.16, nearly double the advantage), and a PSNR of 36.45 dB, also the best; especially, on several scenes (Scene1/2/4/7/10), SAM drops to around 1, far below the 2–3.5 of other methods. The trade-off is a FLOPs of 801G, an order of magnitude higher than other unfolding networks.

### Ablation Study

| Configuration | Key Metrics (Avg) | Notes |
|---|---|---|
| Full Phy-CoSF-9stg | SAM 1.14 / PSNR 36.45 / SSIM 0.915 | All three modules |
| w/o CoSF module (revert to discrete prior) | Close to LADE-DUN baseline (PSNR ~35.7) | Loses continuous rendering, performance drops by 0.7 dB+ |
| w/o Fourier-Mamba (frequency branch) | SSIM/PSNR drops significantly | Lacks global dependency modeling |
| Single-scale instead of triple-branch | Multi-scale details lost | Validates necessity of fine/meso/coarse |
| Fixed wavelength encoding instead of RFE | Lacks high-frequency details | Random Fourier encoding provides necessary inductive bias |
| No train-render separation | Cannot perform zero-shot spectral super-resolution | Paradigm itself determines capability ceiling |

(Note: Detailed ablation table in the paper appendix.)

### Key Findings
- **Significant SAM advantage**: Phy-CoSF greatly improves spectral fidelity: SAM drops from ~2.4 to 1.14, meaning spectral angle error is halved, mainly due to RFE + SE's continuous $\lambda$ encoding enabling SSH to finely model spectral shapes.
- **Parameter efficient but FLOPs intensive**: 0.27M parameters (second only to MST++/MiJUN), but 801G FLOPs—SSH requires per-stage, per-wavelength execution, a typical "small parameter, computation-intensive" implicit representation architecture.
- **Zero-shot spectral super-resolution** is achieved for the first time in a CASSI DUN framework: training uses only discrete wavelengths, inference directly queries any $\lambda$ for high-fidelity slices (see Figure 1 bottom).
- **Fourier-Mamba is more efficient than frequency-domain Transformer**: The authors emphasize using Mamba for 1D sequence modeling in the frequency domain, expanding context while controlling memory, validating SSM's suitability for structured spectral signals.

## Highlights & Insights
- Embedding INR concepts into each DUN stage is a natural yet rarely explored combination: DUN provides physical interpretability and strong priors, INR enables continuous querying, making them highly complementary.
- The decoupled design of "wavelength-independent content $f$ + wavelength-dependent embedding $e_\lambda$" can be transferred to any inverse problem requiring "output querying by continuous coordinates," such as temporal video frame rate upsampling, spatial super-resolution, or dynamic range HDR rendering.
- The Fourier-Mamba block is inspired by "frequency domain coefficients are themselves 1D global signals"—flattening the 2D spectrum and processing with Mamba cleverly bypasses the complexity of Transformers on large images, providing a clean case for SSM in frequency domain processing.
- The combination of random Fourier encoding and learnable MLP projection (fixed high-frequency inductive bias, then task adaptation via MLP) has been validated in NeRF series; this work seamlessly transfers it to 1D wavelength coordinates.

## Limitations & Future Work
- FLOPs of 801G far exceed baselines (recent DAUHST only 6.68G); per-image inference speed is not reported, making it unfriendly for real-time hyperspectral reconstruction.
- Continuous spectral super-resolution is only qualitatively demonstrated under the train-render paradigm (Figure 1); lacks quantitative evaluation on "ground truth at new wavelengths," so the upper bound of super-resolution quality needs further analysis.
- Evaluation is only on ICVL, not covering more challenging datasets like KAIST, CAVE, or real instrument data; robustness to different dispersion parameters is unknown.
- Training uses L1 loss, which may bias toward over-smoothing in noisy/high dynamic range scenarios; perceptual or spectral angle loss could be considered.
- The SSH in CoSF must query wavelengths sequentially; rendering a full set of 200+ wavelengths requires multiple forward passes. Batch-parallel querying optimization could be explored.

## Related Work & Insights
- **vs LADE-DUN**: LADE-DUN uses pretrained latent diffusion as a generative prior, PSNR 35.79; Phy-CoSF replaces the prior with a continuous field, achieves PSNR 36.45, and gains continuous rendering capability.
- **vs MiJUN (Mamba + tensor mode-k unfolding)**: MiJUN is among the first to use Mamba in CASSI DUN; Phy-CoSF applies Mamba in the frequency domain rather than time/space, and adds INR decoupling.
- **vs DERNN-LNLT**: Cross-stage parameter sharing compresses the model; Phy-CoSF takes the opposite route—parameters remain small but INR enables more functions (continuous rendering).
- **vs NeRF/INR series**: This work successfully extends NeRF's "coordinate → color" concept along the spectral axis (rather than spatial) for HSI reconstruction, suggesting future deployment in more inverse problems.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of INR + DUN + Fourier-Mamba is among the first to move CASSI reconstruction from "discrete wavelengths" to "continuous spectra."
- Experimental Thoroughness: ⭐⭐⭐ Compared with 9 mainstream methods on ICVL, but lacks multi-dataset and quantitative continuous super-resolution evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear physical derivation and module decomposition, well-supported by Figures 3/4.
- Value: ⭐⭐⭐⭐ Introduces "on-demand wavelength querying" to hyperspectral imaging, with significant practical value for downstream applications in remote sensing, medical imaging, agriculture, etc.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Spectral Super-Resolution via Adversarial Unfolding and Data-Driven Spectrum Regularization](../../CVPR2026/image_restoration/spectral_super-resolution_via_adversarial_unfolding_and_data-driven_spectrum_reg.md)
- [\[CVPR 2026\] Statistical Characteristic-Guided Denoising for Rapid High-Resolution Transmission Electron Microscopy Imaging](../../CVPR2026/image_restoration/statistical_characteristic-guided_denoising_for_rapid_high-resolution_transmissi.md)
- [\[CVPR 2025\] A Physics-Informed Blur Learning Framework for Imaging Systems](../../CVPR2025/image_restoration/a_physics-informed_blur_learning_framework_for_imaging_systems.md)
- [\[CVPR 2026\] UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution](../../CVPR2026/image_restoration/ucan_unified_convolutional_attention_lightweight_sr.md)
- [\[ECCV 2024\] Learning Exhaustive Correlation for Spectral Super-Resolution: Where Spatial-Spectral Attention Meets Linear Dependence](../../ECCV2024/image_restoration/learning_exhaustive_correlation_for_spectral_super-resolution_where_spatial-spec.md)

</div>

<!-- RELATED:END -->
