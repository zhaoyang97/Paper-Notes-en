---
title: >-
  [Paper Note] PIDSR: Complementary Polarized Image Demosaicing and Super-Resolution
description: >-
  [CVPR 2025][Image Restoration][Polarized image demosaicing] PIDSR proposes a framework for joint complementary optimization of polarized image demosaicing (PID) and polarization image super-resolution (PISR). Utilizing a two-stage recurrent pipeline (spatial-physical coherent reconstruction + polarization-aware resolution enhancement) and a Stokes-assisted network, it directly reconstructs high-quality high-resolution polarization images from CPFA raw images…
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Polarized image demosaicing"
  - "Super-resolution"
  - "Stokes parameters"
  - "Two-stage recurrence"
  - "Degree of Polarization (DoP) and Angle of Polarization (AoP)"
date: 2026-05-08
content_hash: f47808c1e2880bfd
---

# PIDSR: Complementary Polarized Image Demosaicing and Super-Resolution

**Conference**: CVPR 2025  
**arXiv**: [2504.07758](https://arxiv.org/abs/2504.07758)  
**Code**: [https://github.com/PRIS-CV/PIDSR](https://github.com/PRIS-CV/PIDSR)  
**Area**: Image Restoration / Polarization Imaging  
**Keywords**: Polarized image demosaicing, Super-resolution, Stokes parameters, Two-stage recurrence, Degree of Polarization (DoP) and Angle of Polarization (AoP)

## TL;DR

PIDSR proposes a framework for joint complementary optimization of polarized image demosaicing (PID) and polarization image super-resolution (PISR). Utilizing a two-stage recurrent pipeline (spatial-physical coherent reconstruction + polarization-aware resolution enhancement) and a Stokes-assisted network, it directly reconstructs high-quality high-resolution polarization images from CPFA raw images, significantly reducing errors in DoP and AoP.

## Background & Motivation

**Background**: Polarized cameras capture images in four polarization orientations (0°, 45°, 90°, 135°) in a single exposure using Division-of-Focal-Plane (DoFP) technology, facilitating polarized vision tasks (such as shape estimation, reflection removal, dehazing, and HDR). However, the direct output of these cameras is a CPFA (Color Polarization Filter Array) raw image, where each pixel contains information for only one color channel and one polarization orientation, necessitating demosaicing to reconstruct full polarized images.

**Limitations of Prior Work**: (1) Demosaicing inevitably introduces artifacts, and due to the non-linear relationship between DoP/AoP and polarized images ($p = \sqrt{S_1^2+S_2^2}/S_0$, $\theta = \frac{1}{2}\arctan(S_2/S_1)$), demosaicing errors are significantly amplified in DoP and AoP; (2) The resolution of polarized cameras is heavily constrained by hardware compared to standard RGB cameras; (3) Existing PID methods cannot improve resolution, whereas PISR methods assume input without demosaicing artifacts (which actually always exist) — executing PID $\rightarrow$ PISR sequentially leads to error accumulation.

**Key Challenge**: PID and PISR are treated as independent tasks in a serial pipeline, yet they are actually complementary — higher resolution can alleviate demosaicing artifacts (validation experiments show the error rate decreases as resolution increases), while fewer demosaicing artifacts can improve super-resolution performance. Serial pipelines fail to exploit this complementarity.

**Goal**: To design a joint framework $\mathcal{D}^{\uparrow}$ that directly and simultaneously outputs demosaiced and high-resolution polarization images from CPFA raw images, achieving more accurate DoP/AoP than sequential methods.

**Key Insight**: The authors observe that CPFA raw images can be approximately converted into 4 half-resolution full-color polarized images (by separating polarization directions followed by simple RGB demosaicing). This allows PID to be decomposed into two sub-problems: "spatial discontinuity restoration" and "resolution enhancement", aligning with the sub-structural components of PISR.

**Core Idea**: To unify PID and PISR into a recurrent structure. Each iteration consists of two phases: spatial-physical coherent reconstruction (intra-resolution) and resolution enhancement (cross-resolution). Incorporating physical priors via Stokes parameter injection, a recurrency of n iterations achieves $2^n\times$ super-resolution.

## Method

### Overall Architecture

The input CPFA raw image $R$ is first converted into 4 half-resolution full-color polarized images $R_{\alpha_{1,2,3,4}}$ (by separating pixels according to polarization directions followed by bilinear RGB demosaicing). Then, the processing alternates between two stages: Stage 1 ($f$, intra-resolution spatial-physical coherent reconstructor) is used to restore spatial discontinuity and physical correlation, and Stage 2 ($g$, polarization-aware resolution enhancer) performs 2× super-resolution. The first recurrent cycle completes demosaicing to obtain full-resolution images, and each subsequent cycle performs another 2× super-resolution. An $n$-cycle iteration achieves $k = 2^n\times$ super-resolution.

### Key Designs

1. **Recurrent PIDSR Pipeline**:

    - **Function**: Unify demosaicing and super-resolution into a recurrently executable structure to enable complementary optimization.
    - **Mechanism**: The key observation is that a CPFA raw image can yield 4 half-resolution polarized images, meaning PID is equivalent to "restoring spatial discontinuity" (intra-resolution) + "2× resolution enhancement" (cross-resolution). Similarly, PISR can be decomposed into "restoring physical correlation" (intra) + "resolution enhancement" (cross). Decoupling intra and cross and unifying them into two stages establishes a negative feedback loop: improved demosaicing reduces input errors for SR, and improved SR reduces demosaicing artifacts.
    - **Design Motivation**: A naive serial pipeline of $\mathcal{D}$ followed by $\uparrow$ has two fatal issues: (1) the two stages are independent, leading to one-way accumulated errors without negative feedback; (2) it cannot exploit complementarity. The recurrent architecture resolves both problems.

2. **Stokes Feature Injection (SFI) Block**:

    - **Function**: Explicitly inject physical priors of polarization (Stokes parameters $S_1, S_2$) into network features to preserve polarization properties.
    - **Mechanism**: The SFI block contains two branches — an input feature branch (with an MDTA attention module) and a Stokes feature branch. The outputs of both branches are multiplied to generate biases that adjust the input features. This block is embedded into a modified U-Net, replacing the standard convolution block. Denoting high-frequency physical details, the Stokes parameters ($S_1$ describing horizontal/vertical polarization differences and $S_2$ describing 45°/135° differences) complement the low-frequency structure of image features.
    - **Design Motivation**: Directly concatenating image features and Stokes features exhibits a large domain gap — image features primarily capture low-frequency structures, while Stokes features contain high-frequency information. The multiplicative modulation in SFI bridges this domain gap more effectively, similar to FiLM.

3. **Polarization-Aware Resolution Enhancer (Stage g)**:

    - **Function**: Perform 2× super-resolution while preserving polarization properties.
    - **Mechanism**: The encoder features from the coarsest level of Stage f are directly reused (avoiding extracting features from intermediate results to save computation) and upsampled through another decoder and a Stokes feature injection head ($\mathcal{F}_s^g$ processing the more accurate reconstructed Stokes parameters $S_{1,2}^b$). Output features then pass through a feature refinement block $\mathcal{A}^g$ and an upsampling block $\mathcal{U}$ to reconstruct residuals.
    - **Design Motivation**: Since Stage f has already restored spatial discontinuity and physical correlation, the resulting Stokes parameters are more reliable and can better guide the super-resolution process. Skipping redundant feature extraction also improves efficiency.

### Loss & Training

Total loss $L = \lambda_1 L_{img} + \lambda_2 L_{Stokes} + \lambda_3 L_{pol}$, with $\lambda_1 = 1.0, \lambda_2 = 10.0, \lambda_3 = 10.0$.

- **Image Loss** $L_{img}$: $L_1(I_{\alpha_1}+I_{\alpha_3}, I_{\alpha_2}+I_{\alpha_4})$ (utilizing polarization identity constraints) + gradient loss.
- **Stokes Loss** $L_{Stokes}$: Gradient loss of $S_0$ + L1 loss of $S_{1,2}$.
- **Polarization Loss** $L_{pol}$: L1 loss of DoP and AoP.

Trained on a synthetic dataset rendered using Mitsuba 3, using Adam optimizer, with a learning rate of 0.005 for 100 epochs on NVIDIA A800 GPUs. Both stages are trained simultaneously.

## Key Experimental Results

### Main Results

| Method | Demosaiced $S_0$ PSNR↑ | DoP PSNR↑ | AoP MAE↓ | 2×SR $S_0^{HR}$ PSNR↑ | 2×SR DoP PSNR↑ |
|------|---------------------|-----------|----------|----------------------|----------------|
| Polanalyser | 33.28 | 26.68 | 17.87 | - | - |
| IGRI2 | 35.50 | 27.78 | 16.58 | - | - |
| TCPDNet | 38.65 | 32.26 | 13.19 | - | - |
| **PIDSR (demosaic)** | **40.24** | **33.33** | **12.24** | - | - |
| PSRNet (2×) | - | - | - | 36.46 | 32.01 |
| CPSRNet (2×) | - | - | - | 33.60 | 24.14 |
| **PIDSR (2×)** | - | - | - | **37.44** | **32.97** |

### Ablation Study

| Configuration | $S_0$ PSNR↑ | DoP PSNR↑ | AoP MAE↓ |
|------|-------------|-----------|----------|
| Sequential $\mathcal{D}$ and $\uparrow$ | 32.32 | 23.78 | 19.29 |
| Single-stage pipeline | 34.61 | 27.95 | 38.22 |
| Without SFI blocks | 37.18 | 32.73 | 13.12 |
| Ours (demosaic) → PSRNet | 40.24 | 33.33 | 12.24 |
| TCPDNet → ours (SR only) | 38.65 | 32.26 | 13.19 |
| **Complete PIDSR** | **40.24** | **33.33** | **12.24** |

### Key Findings

- For demosaicing, PIDSR outperforms the strongest baseline TCPDNet by 1.59 dB in $S_0$ and 1.07 dB in DoP, while reducing the AoP error by 0.95°.
- The serial method (Sequential $\mathcal{D}$ and $\uparrow$) performs the worst, demonstrating severe error accumulation.
- The AoP MAE of the single-stage pipeline surges to 38.22° — because spatial discontinuities are super-resolved directly without being restored, destroying physical correlation.
- Removing SFI blocks results in a 0.6 dB drop in DoP, indicating that physical prior injection from Stokes parameters indeed helps preserve polarization properties.
- Complementarity verification: Performing demosaicing with PIDSR followed by SR using PSRNet approximates the joint optimization of the complete PIDSR — demonstrating that the demosaicing quality of PIDSR is already exceptionally high. Conversely, demosaicing with TCPDNet followed by SR with PIDSR performs worse than the complete PIDSR — confirming that the SR stage indeed benefits from superior demosaicing.
- On real-world data, PIDSR's AoP and DoP do not exhibit distinct jagged artifacts, whereas other methods do.

## Highlights & Insights

- **Dual Theoretical and Experimental Validation of Complementarity**: Rather than simply claiming complementarity between PID and PISR, the authors quantitatively prove that higher resolutions reduce demosaicing error rates via controlled experiments, and demonstrate that joint optimization outperforms sequential methods via ablation. This validation approach is highly rigorous.
- **Equivalent Transformation of CPFA $\rightarrow$ Half-Resolution Polarized Images**: This insight serves as the foundation of the methodological framework — redefining PID as "spatial restoration + 2× super-resolution", thereby naturally unifying it with PISR. This represents an elegant problem reformulation.
- **Stokes Parameters as Physical Prior Injection**: The approach leverages polarization-specific physical constraints (e.g., $I_{\alpha_1}+I_{\alpha_3} = I_{\alpha_2}+I_{\alpha_4}$) as loss regularization and injects them into the network through SFI blocks. This physics-informed design is inherently more reliable than purely data-driven methods.

## Limitations & Future Work

- Processes only single-frame CPFA raw images and does not support polarized video sequences.
- Cannot process non-polarized CFA raw images (as it requires Stokes parameters as inputs).
- A potential domain gap exists between synthetic training data and real polarized camera data.
- Running multiple recurrent cycles (e.g., 2 cycles for 4× SR) increases inference latency.
- Larger super-resolution factors (e.g., 8×) or joint tasks with other low-level vision tasks (e.g., denoising, HDR) have not yet been explored.

## Related Work & Insights

- **vs TCPDNet**: The state-of-the-art polarization demosaicing method that reconstructs polarized images directly from CPFA using a CNN. PIDSR achieves superior demosaicing results via joint super-resolution, proving that inter-task complementarity is indeed effective.
- **vs PSRNet/CPSRNet**: Existing polarization super-resolution methods assume artifact-free inputs, which compromises their performance in real scenes where demosaicing artifacts are inevitable. PIDSR fundamentally addresses this invalid assumption through joint optimization.
- **vs Traditional RGB Demosaicing + SR**: Polarization scenarios are more complex than RGB (12 channels vs. 3 channels, with non-linear DoP/AoP computations), preventing direct application of RGB methods. The integration of Stokes physical priors in PIDSR is a design unique to polarization imaging.

## Rating

- Novelty: ⭐⭐⭐⭐ The observation of complementarity between PID and PISR is novel, and the two-stage recurrent pipeline is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on both synthetic and real-world data, with comprehensive ablation studies and downstream task verification.
- Writing Quality: ⭐⭐⭐⭐ Problem modeling and motivational derivations are highly clear.
- Value: ⭐⭐⭐⭐ Highly valuable to the polarization imaging community; the concept of complementary joint optimization is generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Augmenting Perceptual Super-Resolution via Image Quality Predictors](augmenting_perceptual_super-resolution_via_image_quality_predictors.md)
- [\[CVPR 2025\] Progressive Focused Transformer for Single Image Super-Resolution](progressive_focused_transformer_for_single_image_super-resolution.md)
- [\[CVPR 2025\] QMambaBSR: Burst Image Super-Resolution with Query State Space Model](qmambabsr_burst_image_super-resolution_with_query_state_space_model.md)
- [\[CVPR 2025\] AdcSR: Adversarial Diffusion Compression for Real-World Image Super-Resolution](adversarial_diffusion_compression_for_real-world_image_super-resolution.md)
- [\[NeurIPS 2025\] FIPER: Factorized Features for Robust Image Super-Resolution and Compression](../../NeurIPS2025/image_restoration/fiper_factorized_features_for_robust_image_super-resolution_and_compression.md)

</div>

<!-- RELATED:END -->
