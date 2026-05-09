---
title: >-
  [Paper Note] Leveraging Multispectral Sensors for Color Correction in Mobile Cameras
description: >-
  [CVPR 2026][Image Generation][Multispectral sensors] This paper proposes a unified end-to-end color correction framework that jointly fuses data from a high-resolution RGB sensor and an auxiliary low-resolution multispectral (MS) sensor, integrating illuminant estimation, illuminant compensation, and color space conversion into a single model. The proposed approach reduces color error ($\Delta E_{00}$) by up to 50% compared to RGB-only and MS-only baselines.
tags:
  - CVPR 2026
  - Image Generation
  - Multispectral sensors
  - color correction
  - automatic white balance
  - sensor fusion
  - mobile cameras
date: 2026-05-08
content_hash: 7bdc3ac8381ec9f5
---

# Leveraging Multispectral Sensors for Color Correction in Mobile Cameras

**Conference**: CVPR 2026
**arXiv**: [2512.08441](https://arxiv.org/abs/2512.08441)
**Code**: [Project Page](https://lucacogo.github.io/Mobile-Spectral-CC/)
**Area**: Image Generation / Image Processing
**Keywords**: Multispectral sensors, color correction, automatic white balance, sensor fusion, mobile cameras

## TL;DR

This paper proposes a unified end-to-end color correction framework that jointly fuses data from a high-resolution RGB sensor and an auxiliary low-resolution multispectral (MS) sensor, integrating illuminant estimation, illuminant compensation, and color space conversion into a single model. The proposed approach reduces color error ($\Delta E_{00}$) by up to 50% compared to RGB-only and MS-only baselines.

## Background & Motivation

- **Background**: Color correction is a fundamental component of the camera imaging pipeline, responsible for converting raw sensor measurements into a device-independent standard color space (e.g., CIE XYZ). Conventional pipelines adopt a modular design: automatic white balance (AWB, comprising illuminant estimation and illuminant compensation) followed by color space transformation (CST), with each step processed independently, leading to cascading error propagation.

- **Key Challenge**: RGB sensors possess only three broadband channels, providing severely limited spectral information that makes it unreliable to disentangle surface reflectance $R(\lambda)$ from illuminant power distribution $E(\lambda)$ — an inherently underdetermined problem. Although recent nanophotonic advances have enabled compact snapshot multispectral sensors (e.g., with 15 narrowband channels), existing methods typically employ MS data only at the illuminant estimation stage and subsequently discard it, wasting valuable spectral information.

- **Key Insight**: This work propagates multispectral information throughout all stages of the color correction pipeline (illuminant estimation, illuminant compensation, and CST), rather than restricting its use to the first step. **Core Idea**: End-to-end joint modeling of dual RGB+MS inputs within a unified framework that implicitly performs all correction steps, leveraging spectral constraints to improve color accuracy.

## Method

### Overall Architecture

The paper proposes a dual-input end-to-end color correction framework: a high-resolution RGB image ($512 \times 512$) serves as the primary input, while a low-resolution MS image ($64 \times 64$, 15 channels) serves as the auxiliary input. Features extracted by separate encoders for each modality are fused, and the model directly outputs results normalized to the D65 illuminant in the CIE XYZ color space. Two lightweight image-to-image architectures (LPIENet and cmKAN) are adapted to demonstrate the generality of the framework.

### Key Designs

1. **LPIENet Adaptation (U-Net Structure)**:
   - **Function**: An image restoration architecture based on U-Net topology, suited for tasks requiring spatially consistent transformations.
   - **Mechanism**: A dedicated spectral encoder branch (3 IRA blocks without downsampling) is added to the original 3-encoder-2-decoder structure; RGB and MS features are fused via element-wise addition at the skip connections.
   - **Design Motivation**: IRA blocks integrate MobileNet-style convolutions with parallel channel/spatial attention, maintaining lightweight design (220K parameters in the standard version and 60K in the small version).

2. **cmKAN Adaptation (KAN Hypernetwork Structure)**:
   - **Function**: Uses a hypernetwork to generate spatially varying KAN layer parameters, enabling smooth nonlinear color transformations.
   - **Mechanism**: A 3-layer convolutional spectral encoder is added to the generator; its outputs are fused with the illuminant estimator (IE) features at two feature levels via element-wise addition.
   - **Design Motivation**: KAN layers (3rd-order B-spline, grid size 5) are inherently well-suited for modeling smooth color mappings; the entire architecture contains only 18K parameters, making it extremely compact.

3. **Physics-Driven Simulated Dataset**:
   - **Function**: Provides RGB–MS–Ground Truth triplets under realistic illumination conditions for training and evaluation.
   - **Mechanism**: Starting from reflectance spectra in two public hyperspectral datasets, the dataset is synthesized by combining 102 illuminant SPDs with 7 camera spectral sensitivity functions (covering both mobile and mirrorless cameras), yielding 116,688 image triplets. A misaligned version is also constructed.
   - **Design Motivation**: No real-world paired RGB+MS dataset with the required hyperspectral reflectance ground truth exists; physics-based simulation is adopted while retaining realistic acquisition noise.

### Loss & Training

- End-to-end training directly optimizing the $\Delta E_{00}$ color difference between the output and the D65 ground truth in CIE XYZ.
- Scene-level data splitting: 80% training / 20% testing, with an additional 20% of the training set held out for validation to prevent data leakage.
- For the misaligned dataset version, only the spectral encoder is fine-tuned to adapt to geometric misalignment.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (best cmKAN-light) | Best Traditional (FC4) | Gain |
|---------|--------|------------------------|------------------------|------|
| Aligned – Mirrorless | $\Delta E_{00}$ Mean | 1.60 | 3.28 | −51.2% |
| Aligned – Mirrorless | $\Delta E_{00}$ Median | 1.30 | 2.86 | −54.5% |
| Aligned – Mobile | $\Delta E_{00}$ Mean | 1.47 | 3.16 | −53.5% |
| Aligned – Mobile | $\Delta E_{00}$ Median | 1.19 | 2.75 | −56.7% |
| Misaligned – Mirrorless | $\Delta E_{00}$ Mean | 1.76 | 3.25 (SpectralFC4) | −45.8% |
| Misaligned – Mobile | $\Delta E_{00}$ Mean | 1.62 | 3.19 (SpectralFC4) | −49.2% |

### Ablation Study

| Configuration | Key Metric ($\Delta E_{00}$ Mean) | Notes |
|---------------|----------------------------------|-------|
| cmKAN-light (RGB+MS) | 1.60 | Best performance, only 18K parameters |
| LPIENet (RGB+MS) | 1.74 | 220K parameters, slightly below cmKAN |
| LPIENet-small (RGB+MS) | 2.09 | 60K parameters |
| SpectralFC4 (traditional pipeline + MS) | 3.25 | MS used for illuminant estimation only |
| FC4 (traditional pipeline + RGB) | 3.28 | RGB-only method |
| Gray World assumption | 7.80 | Statistical method, largest error |

### Key Findings

- End-to-end methods achieve a qualitative leap over traditional modular pipelines: $\Delta E_{00}$ drops from 3+ to approximately 1.5.
- cmKAN-light achieves the best performance with only 18K parameters, demonstrating the natural suitability of KAN structures for color mapping tasks.
- On the misaligned dataset, fine-tuning only the spectral encoder suffices to maintain high performance, indicating robustness to geometric misalignment.
- Utilizing MS data throughout the entire pipeline yields substantially greater gains than restricting its use to illuminant estimation (SpectralFC4 vs. the proposed method).

## Highlights & Insights

- The strategy of replacing modular pipelines with end-to-end joint modeling is direct and effective, eliminating cascading error propagation across stages.
- The extremely small model size (18K–220K parameters) is well-suited for mobile deployment, which is precisely the most likely application scenario for MS sensors.
- Demonstrating framework generality by adapting two architecturally distinct backbones (U-Net and KAN) strengthens the credibility of the approach.
- The physics-driven dataset construction methodology is principled: it combines real hyperspectral data, real camera spectral sensitivity functions, and real illuminant SPDs.

## Limitations & Future Work

- The dataset is entirely simulation-based; although the RGB and MS images are physically driven, a domain gap with real sensor pairs remains.
- Only static scenes are evaluated; temporal consistency in video mode is not considered.
- The low spatial resolution of the MS sensor ($64 \times 64$) may limit color correction quality in regions with fine textures.
- Evaluation is restricted to single-illuminant scenes; multi-illuminant mixed scenarios (e.g., indoor artificial and natural light) are not covered.
- Although the KAN model has few parameters, inference latency is not sufficiently discussed in the paper.

## Related Work & Insights

- **vs. FC4/ConvMean (traditional pipeline)**: Traditional methods process color correction in separate stages using only RGB data, yielding errors of approximately $\Delta E_{00}$ 3.2–3.5; the proposed end-to-end RGB+MS fusion reduces this to around 1.5.
- **vs. SpectralFC4/SpectralConvMean**: These methods also use MS data but only for illuminant estimation, discarding MS information thereafter; leveraging MS throughout the full pipeline in this work further reduces error by approximately 50%.
- **vs. QU**: A cross-camera adaptation method that requires fine-tuning and achieves limited performance ($\Delta E_{00}$ 5.3–6.4).
- **Insights**: The cost and form factor of multispectral sensors are now compatible with integration into mobile devices; end-to-end utilization of their information is more promising than conventional step-by-step approaches.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The concept of end-to-end RGB+MS fusion for color correction is novel; dual-architecture validation reinforces persuasiveness.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple camera sensitivity functions, both aligned and misaligned settings, and comprehensive statistical metrics.
- **Writing Quality**: ⭐⭐⭐⭐ Physical modeling is clearly presented; related work is thoroughly surveyed.
- **Value**: ⭐⭐⭐⭐ Provides practical guidance for the application of multispectral sensors in mobile computational photography.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Too Vivid to Be Real? Benchmarking and Calibrating Generative Color Fidelity](too_vivid_to_be_real_benchmarking_and_calibrating_generative_color_fidelity.md)
- [\[CVPR 2026\] Ar2Can: An Architect and an Artist Leveraging a Canvas for Multi-Human Generation](ar2can_an_architect_and_an_artist_leveraging_a_canvas_for_multi-human_generation.md)
- [\[CVPR 2026\] Physics-Consistent Diffusion for Efficient Fluid Super-Resolution via Multiscale Residual Correction](physics-consistent_diffusion_for_efficient_fluid_super-resolution_via_multiscale.md)
- [\[ICLR 2026\] Test-Time Iterative Error Correction for Efficient Diffusion Models](../../ICLR2026/image_generation/test-time_iterative_error_correction_for_efficient_diffusion_models.md)
- [\[CVPR 2026\] FRAMER: Frequency-Aligned Self-Distillation with Adaptive Modulation Leveraging Diffusion Priors for Real-World Image Super-Resolution](framer_frequency-aligned_self-distillation_with_adaptive_modulation_leveraging_d.md)

</div>

<!-- RELATED:END -->
