---
title: >-
  [Paper Note] Learning Phase Distortion with Selective State Space Models for Video Turbulence Mitigation
description: >-
  [CVPR 2025][Physics & Scientific Computing][Atmospheric Turbulence Mitigation] MambaTM is proposed as the first Mamba-based video atmospheric turbulence mitigation network. It reparameterizes the phase distortion traditionally represented by Zernike polynomials into Latent Phase Distortion (LPD) via a VAE, using LPD to guide the state transitions of SSMs. While maintaining linear complexity and a global receptive field, it achieves state-of-the-art restoration quality and nea…
tags:
  - "CVPR 2025"
  - "Physics & Scientific Computing"
  - "Atmospheric Turbulence Mitigation"
  - "Mamba"
  - "State Space Models"
  - "Latent Phase Distortion"
  - "Degradation-Aware Restoration"
date: 2026-05-08
content_hash: 98f8a1e2c3d41898
---

# Learning Phase Distortion with Selective State Space Models for Video Turbulence Mitigation

**Conference**: CVPR 2025  
**arXiv**: [2504.02697](https://arxiv.org/abs/2504.02697)  
**Code**: [https://xg416.github.io/MambaTM](https://xg416.github.io/MambaTM)  
**Area**: Image/Video Restoration / Atmospheric Turbulence Mitigation  
**Keywords**: Atmospheric Turbulence Mitigation, Mamba, State Space Models, Latent Phase Distortion, Degradation-Aware Restoration

## TL;DR
MambaTM is proposed as the first Mamba-based video atmospheric turbulence mitigation network. It reparameterizes the phase distortion traditionally represented by Zernike polynomials into Latent Phase Distortion (LPD) via a VAE, using LPD to guide the state transitions of SSMs. While maintaining linear complexity and a global receptive field, it achieves state-of-the-art restoration quality and nearly 2× inference speedup (55.4 FPS vs 32.7 FPS).

## Background & Motivation

1. **Background**: Atmospheric turbulence causes spatially and temporally varying pixel displacements and blur in long-range imaging, severely affecting downstream tasks such as detection and recognition. Existing deep learning methods are categorized into single-frame and multi-frame approaches. Multi-frame methods are generally superior due to their ability to exploit the temporal "lucky effect" (where certain frames are less affected by turbulence).
2. **Limitations of Prior Work**:
    - **Spatial Dimension**: The limited receptive field of CNNs struggles to handle the wide-range spatial dependencies of turbulence.
    - **Temporal Dimension**: Self-attention can theoretically aggregate long temporal information, but its quadratic complexity is difficult to scale to multiple frames. Recurrent aggregation methods suffer from parallelization issues and training instability.
    - **Degradation Representation**: The traditional representation of phase distortion using Zernike polynomials suffers from severe ill-posedness, where multiple sets of Zernike coefficients can yield the same degradation pattern, and it requires non-differentiable PSF sizes.
3. **Key Challenge**: Simultaneously achieving a large spatial and long temporal receptive field, low complexity, and an accurately estimable degradation representation.
4. **Goal**: (1) To design an efficient turbulence mitigation backbone with a global receptive field; (2) To propose a learnable, one-to-one degradation representation to replace Zernike polynomials.
5. **Key Insight**: Utilizing Mamba (selective State Space Models) to replace attention/RNNs to achieve linear complexity and a global receptive field, and employing a VAE to compress the Zernike representation into LPD in the latent space, thereby eliminating the ill-posedness.
6. **Core Idea**: LPD makes degradation estimation more well-posed + Mamba makes spatiotemporal modeling more efficient = degradation-aware, efficient video turbulence mitigation.

## Method

### Overall Architecture
The input consists of a degraded video of $T$ frames, $I \in \mathbb{R}^{T \times H \times W \times 3}$. A multi-scale encoder extracts frame-by-frame features. After processing through $N_1$ groups of Mamba modules, the LPD decoder estimates a 4-channel phase distortion map (2-channel tilt + 2-channel blur). The LPD encoder then compresses this map into guided features, which are fed into the subsequent $N_2$ groups of guided Mamba modules. Finally, a multi-scale decoder outputs the restored image. ReBlurNet (frozen and pre-trained) is responsible for re-degrading the restoration results with LPD, achieving joint optimization.

### Key Designs

1. **Latent Phase Distortion (LPD)**:

    - **Function**: Replaces traditional Zernike coding with a learnable low-dimensional representation to achieve a one-to-one mapping with degradation patterns.
    - **Mechanism**: A conditional VAE is used to encode the Zernike coefficient field $\mathbf{a}$ into a latent space $\tilde{\mathbf{a}} \sim \mathcal{N}(\mu, \sigma^2)$, where $(\mu, \log\sigma)$ represents the LPD. The decoder of the VAE (ReBlurNet) generates blur patterns conditioned on the LPD, replacing the original large-kernel depthwise convolutions. LPD is 50× faster than Zernike (0.02s vs. 0.16–6.10s), fully differentiable, and improves the re-degradation PSNR from 25.84/31.17 to 34.08 dB.
    - **Design Motivation**: The mapping from Zernike coefficients to degradation is many-to-one (meaning the phase retrieval problem is ill-posed), and training directly to estimate Zernike coefficients may fail to converge. The VAE maps infinite possibilities into a unique Gaussian distribution, fundamentally solving the ill-posedness.

2. **Multi-Scan Hybrid Mamba Module**:

    - **Function**: Achieves global spatiotemporal modeling of video features with linear complexity.
    - **Mechanism**: Each Mamba group comprises three bidirectional Mamba blocks utilizing different scanning orders: Space-First (SFMB, width-height-time axis), Time-First (TFMB, time-height-width axis), and Local Hilbert scan (LHMB, a space-filling curve that preserves multi-dimensional locality). The three scanning schemes provide complementary connection strengths along different axes. Bidirectional scanning ensures unbiased modeling of non-causal visual data.
    - **Design Motivation**: When flattening a 3D tensor into a 1D sequence, a single scanning order inevitably breaks connections along certain axes. The Hilbert curve is particularly effective at maintaining post-flattening local neighborhood, compensating for the limitations of standard raster scans.

3. **Guided SSM (GSSM)**:

    - **Function**: Explicitly guides the state transitions of the state space model using the estimated LPD information.
    - **Mechanism**: In a standard Mamba, the control parameters $\Delta, B, C$ depend solely on the input feature $x$. GSSM concatenates the LPD encoding $r$ with $x$ to jointly determine these parameters: $\Delta = s_\Delta(x; r), B = s_B(x; r), C = s_C(x; r)$. In addition, the LPD is used to modulate the gating mechanism at the output of the Mamba layer. This couples the state evolution and transition with degradation information.
    - **Design Motivation**: While ReBlurNet implicitly injects degradation awareness via the re-degradation loss, GSSM further explicitly guides the aggregation mode at each spatiotemporal location—severely degraded regions might require aggregating information from more frames.

### Loss & Training
Two-stage training:
- **Stage 1** (ReBlurNet): VAE loss $\mathcal{L}_{VAE} = \mathcal{L}_{returb} + \alpha_k \mathcal{L}_{KL}$, where $\mathcal{L}_{returb}$ is the L1 reconstruction loss, and $\mathcal{L}_{KL}$ is the KL divergence regularization.
- **Stage 2** (MambaTM): Freeze ReBlurNet, with the total loss $\mathcal{L} = \mathcal{L}_{restore} + 0.2 \mathcal{L}_{returb}$. $\mathcal{L}_{restore}$ consists of Charbonnier loss + 0.01× perceptual loss. $\mathcal{L}_{returb}$ contains tilt reconstruction, blur reconstruction, and KL divergence.
- **Progressive Training**: Gradually increases the configuration from batch size 16, patch size 192, and 18 frames to batch size 4, patch size 256, and 36 frames.

## Key Experimental Results

### Main Results

**ATSyn-dynamic Dynamic Scenes (PSNR / SSIM / LPIPS):**

| Method | Weak | Medium | Strong | Overall |
|------|------|--------|--------|---------|
| DATUM (prev SOTA) | 30.21 / 0.887 / 0.179 | 29.62 / 0.878 / 0.183 | 28.26 / 0.846 / 0.219 | 29.42 / 0.871 / 0.192 |
| **MambaTM** | **30.87 / 0.899 / 0.143** | **30.08 / 0.890 / 0.143** | **28.61 / 0.860 / 0.172** | **29.92 / 0.884 / 0.152** |

**TMT Dynamic Scenes + Speed:**

| Method | PSNR | SSIM | LPIPS | FPS |
|------|------|------|-------|-----|
| DATUM | 28.60 | 0.844 | 0.225 | 32.7 |
| **MambaTM** | **28.90** | **0.856** | **0.200** | **55.4** |

**ATSyn-static + Turb-Text Real World:**

| Method | PSNR (static) | CRNN/DAN/ASTER Recognition Accuracy |
|------|-------------|----------------------|
| DATUM | 26.76 | 93.55 / 97.95 / 97.25 |
| **MambaTM** | **27.01** | **97.80 / 99.35 / 98.15** |

### Ablation Study

**LPD vs. Zernike Representation:**

| Representation | Speed (s) | PSNR_returb | Differentiability |
|------|---------|-------------|--------|
| Zernike (Strict Supervision) | 0.16~6.10 | 25.84 | Partial |
| Zernike (Relaxed Supervision) | 0.16~6.10 | 31.17 | Partial |
| **LPD** | **0.02** | **34.08** | **Fully** |

**Scanning Strategy Ablation (ATSyn-dynamic Overall):**

| Configuration | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| SF + TF Only | 29.13 | 0.872 | 0.167 |
| SF + TF + LH | 29.67 | 0.883 | 0.157 |
| SF + TF + LH + LPD Guidance | **29.75** | 0.881 | **0.153** |

### Key Findings
- **LPD is the Core Design**: Translates the ill-posed Zernike estimation problem into a well-posed latent space estimation, improving the re-degradation PSNR by 3 to 8 dB.
- **Mamba's Linear Complexity** brings real-time performance of 55.4 FPS, which is 1.7× faster than DATUM, and provides greater stability (SSMs are linear recurrent, avoiding the inference instability issues inherent in non-linear recurrences of DATUM).
- **The Three Scanning Strategies are Complementary but Not Mandatory**: Removing any single scan does not lead to a collapse in performance, demonstrating the robustness of SSMs, although the hybrid scan provides more uniform connectivity across all axes.
- **ReBlurNet Design**: NAFNet with multi-scale modulation outperforms PlainUNet (46.72 vs. 43.26 dB), whereas an excessively large NAFNet yields only marginal returns.

## Highlights & Insights
- **Exquisite LPD Reparameterization Trick**: It utilizes a VAE to convert the ill-posed estimation of physical parameters into a well-posed latent code estimation while preserving physical meaning. This trick can be transferred to any degradation estimation scenario with a "many-parameters-to-one-effect" challenge (e.g., optical aberration or motion blur kernel estimation).
- **Degradation-Guided Mamba Design**: Degradation information is injected into the $\Delta, B, C$ parameters of the SSM, making state evolution and information aggregation adaptive. This "conditional SSM" concept can be transferred to other conditional generation or restoration tasks.
- **Progressive Training Strategy**: Progressive training from small patches with large batch sizes to larger patches with smaller batch sizes ensures rapid convergence in the early stages while enabling the model to learn long-range dependencies in later stages. This is a highly practical training tip for video restoration tasks.

## Limitations & Future Work
- **Trained Only on Synthetic Data**: Although the qualitative results on real-world data are impressive, there is a lack of quantitative evaluation on real turbulence data due to the absence of ground truth.
- **The VAE for LPD Requires Pre-training**: The two-stage training increases complexity; whether the VAE and restoration network can be trained end-to-end jointly is worth exploring.
- **Limited Mitigation in Dynamic Scenes**: The paper assumes that dynamic regions undergo rigid motion, which may fail for non-rigid motion (such as pedestrian limbs).
- **Limitations of the Hilbert Scan**: The Hilbert curve inevitably introduces jumps near borders, and the optimal flattening strategy for 3D tensors remains underexplored.

## Related Work & Insights
- **vs. DATUM [Zhang et al.]**: DATUM utilizes a non-linear recurrence structure, which offers global temporal receptive fields but suffers from unstable training and slow inference. MambaTM replaces it with a linear SSM, yielding a substantial boost in stability and speed.
- **vs. TMT [Zhang et al.]**: TMT utilizes temporal-channel self-attention, whose quadratic complexity limits the number of processed frames. MambaTM's linear complexity allows it to handle more frames.
- **vs. DRBNet [Chen et al.]**: DRBNet employs a differentiable simulator for degradation awareness but still relies on Zernike polynomials. LPD fundamentally resolves the ill-posedness of Zernike representation.

## Rating
- Novelty: ⭐⭐⭐⭐ Both the LPD reparameterization and degradation-guided Mamba are meaningful and novel designs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Synthetic and real, static and dynamic, multiple benchmarks, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Structurally organized, though the physical background section might be somewhat challenging for non-optical readers.
- Value: ⭐⭐⭐⭐ First application of Mamba to turbulence mitigation, with the LPD design being easily transferable to other degradation estimation scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Physics-Guided Machine Learning for Uncertainty Quantification in Turbulence Models](../../NeurIPS2025/physics/physics-guided_machine_learning_for_uncertainty_quantification_in_turbulence_mod.md)
- [\[CVPR 2025\] Towards Faithful Multimodal Concept Bottleneck Models](towards_faithful_multimodal_concept_bottleneck_models.md)
- [\[ICML 2026\] MōLe-Λ: Learning the Coupled-Cluster Response State for Energies, Gradients, and Properties](../../ICML2026/physics/mōle-λ_learning_the_coupled-cluster_response_state_for_energies_gradients_and_pr.md)
- [\[CVPR 2025\] KAC: Kolmogorov-Arnold Classifier for Continual Learning](kac_kolmogorov-arnold_classifier_for_continual_learning.md)
- [\[NeurIPS 2025\] GyroSwin: 5D Surrogates for Gyrokinetic Plasma Turbulence Simulations](../../NeurIPS2025/physics/gyroswin_5d_surrogates_for_gyrokinetic_plasma_turbulence_simulations.md)

</div>

<!-- RELATED:END -->
