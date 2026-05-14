---
title: >-
  [Paper Note] Audio Super-Resolution with Latent Bridge Models
description: >-
  [NeurIPS 2025][Image Restoration][Audio super-resolution] This paper proposes AudioLBM, which compresses audio waveforms into a continuous latent space and employs a bridge model to realize a latent-to-latent generation…
tags:
  - "NeurIPS 2025"
  - "Image Restoration"
  - "Audio super-resolution"
  - "latent bridge model"
  - "frequency-aware training"
  - "cascaded super-resolution"
  - "any-to-192kHz"
date: 2026-05-08
content_hash: 5e3ab61e57f13f87
---

# Audio Super-Resolution with Latent Bridge Models

**Conference**: NeurIPS 2025
**arXiv**: [2509.17609](https://arxiv.org/abs/2509.17609)
**Code**: Available (Demo: [https://AudioLBM.github.io/](https://AudioLBM.github.io/))
**Area**: Audio Super-Resolution / Generative Models
**Keywords**: Audio super-resolution, latent bridge model, frequency-aware training, cascaded super-resolution, any-to-192kHz

## TL;DR

This paper proposes AudioLBM, which compresses audio waveforms into a continuous latent space and employs a bridge model to realize a latent-to-latent generation process from low-resolution to high-resolution. Combined with frequency-aware training for broader data utilization and a cascaded design to surpass the 48kHz ceiling, AudioLBM comprehensively outperforms methods such as AudioSR across speech, sound effects, and music, while achieving any-to-192kHz audio super-resolution for the first time.

## Background & Motivation

**Background**: Audio super-resolution (SR) aims to upsample low-resolution waveforms to high resolution. Existing approaches include mapping-based methods, GANs, diffusion models, and bridge models. AudioSR is currently the most representative cross-domain any-to-48kHz method, based on a diffusion model operating in a mel-spectrogram latent space. A2SB applies a Schrödinger Bridge in the STFT domain for music bandwidth extension.

**Limitations of Prior Work**: The generation quality of existing methods is constrained by the mismatch between generative priors and the super-resolution task: (1) AudioSR generates high-resolution content from Gaussian noise (noise-to-latent), neglecting the rich prior information in the LR waveform; (2) A2SB treats the high-frequency region in the STFT domain as missing and fills it with Gaussian noise, yielding an equally uninformative prior. Furthermore, all existing methods are bounded by the 48kHz ceiling, whereas professional audio production requires 96kHz or even 192kHz.

**Key Challenge**: The LR waveform itself is a highly informative prior for the HR target, yet existing frameworks fail to exploit it effectively. The generation process should be a conditional transformation from LR to HR rather than a generation from noise. Scarcity of high-sampling-rate training data further limits scalability.

**Goal**: (1) Design a generative framework that fully leverages LR prior information; (2) Address the scarcity of high-sampling-rate training data; (3) Break through the 48kHz super-resolution ceiling to reach 96kHz and 192kHz.

**Key Insight**: Directly compress audio waveforms into a continuous latent space (preserving LR prior information) and connect LR and HR latent representations via a bridge model. Frequency-aware training is introduced to enable any-to-any upsampling, and a cascaded design with prior augmentation is employed to surpass the sampling-rate ceiling.

**Core Idea**: Replace the noise-to-data diffusion paradigm with a bridge model operating in waveform latent space, complemented by frequency-aware training and a cascaded design for high-quality audio super-resolution.

## Method

### Overall Architecture

Input: LR waveform $\bm{x}^{LR}$ → waveform VAE encoder → LR latent $\bm{z}^{LR}$ → bridge model reverse sampling → HR latent $\bm{z}^{HR}$ → VAE decoder → HR waveform $\bm{x}^{HR}$. During training, LR–HR latent pairs serve as the two boundary distributions of the bridge model, which learns the generative path from LR to HR.

### Key Designs

1. **Waveform Latent Space Bridge Model (AudioLBM)**:

    - **Function**: Establishes a generative path from LR to HR in a continuous latent space, fully exploiting the LR prior.
    - **Mechanism**: A convolutional VAE is trained to compress waveforms into $\bm{z} \in \mathbb{R}^{c \times l}$. Using $\bm{z}^{LR}$ as the prior ($t=T$, Dirac distribution) and $\bm{z}^{HR}$ as the target ($t=0$), a bridge process is established. A noise prediction network $\epsilon_\theta(\bm{z}_t, t, \bm{z}_T)$ is trained with loss $\|\epsilon_\theta - (\bm{z}_t - \alpha_t \bm{z}_0)/(\alpha_t \sigma_t)\|_2^2$. During inference, first-order SDE reverse sampling (50 steps) is performed starting from $\bm{z}^{LR}$.
    - **Design Motivation**: Unlike AudioSR's noise-to-latent paradigm, the bridge model's latent-to-latent path naturally inherits the spectral structure and energy distribution of the LR waveform. Compared to STFT-domain methods, directly compressing the waveform avoids frequency-band misalignment.

2. **Frequency-Aware Training (Frequency-Aware LBMs)**:

    - **Function**: Overcomes scarcity of high-sampling-rate training data and enables any-to-any super-resolution.
    - **Mechanism**: During training, LR/HR sampling rate pairs are dynamically sampled: first, filtering yields an HR version (with sampling rate $SR_{HR}$ lower than the original but retaining core frequency bands); then $SR_{LR} \sim \mathcal{U}(0, SR_{HR})$ is randomly sampled to generate the LR version. The prior frequency $f_{prior}$ and target frequency $f_{target}$ are encoded as sinusoidal embedding tokens prepended to the DiT input. A constant scaling factor $s$ is also used to rescale latent vectors for training stability.
    - **Design Motivation**: Training at fixed sampling rates wastes a large portion of non-48kHz data. Frequency-aware conditioning allows the model to explicitly learn different frequency-band mappings, and the target frequency can be specified at inference time. Experiments confirm that training data diversity is far more important than using only 48kHz data.

3. **Cascaded LBMs + Prior Augmentation**:

    - **Function**: Overcomes single-model capacity limitations to achieve progressive super-resolution from 48→96→192kHz.
    - **Mechanism**: Multiple AudioLBMs are trained in stages. To mitigate cascading errors, two forms of prior augmentation are proposed: (i) waveform-domain degradation — randomly removing some high-frequency details of the HR prior near the Nyquist boundary; (ii) latent-space blurring — applying dynamic Gaussian smoothing along the time axis with ratio $b_r \sim \mathcal{U}(0, b_r^{max})$. The training objective becomes generating the UHR target from the degraded or blurred prior.
    - **Design Motivation**: Unlike the noise augmentation used in diffusion models, the bridge model's boundary is a Dirac distribution, making blurring and degradation more natural. Exposing the model to degraded priors during training makes it more robust to artifacts in the previous stage's output at inference time.

### Loss & Training

The base loss is noise prediction MSE. The frequency-aware variant adds frequency-conditional inputs. The cascaded variant incorporates blurred priors and degradation condition $b_r$. Training data comprises approximately 5,000 hours (speech, sound effects, and music), with an effective batch size of 128 and 1M iterations. Inference uses 50-step first-order SDE sampling.

## Key Experimental Results

### Main Results

| Setting | Metric | AudioSR | Ours (zero-shot) | Gain |
|---------|--------|---------|-----------------|------|
| VCTK 8→48kHz | LSD↓ | 0.940 | 0.753 | 19.9% |
| VCTK 8→48kHz | SSIM↑ | 0.809 | 0.893 | +0.084 |
| VCTK 8→48kHz | SigMOS↑ | 2.846 | 3.023 | +0.177 |
| 48Audio 8→48kHz | LSD↓ | 1.468 | 1.066 | 27.4% |
| ESC-50 16→44.1kHz | LSD↓ | 1.292 | 0.999 | 22.7% |
| SDS 16→44.1kHz | LSD↓ | 1.352 | 1.160 | 14.2% |

### Ablation Study

| Configuration | ESC-50 LSD↓ | SDS LSD↓ | Note |
|---------------|-------------|----------|------|
| w/o Filter | 1.366 | 1.461 | No filtering of low-sampling-rate data |
| w/o Input-A | 1.052 | 1.187 | No input frequency awareness |
| w/o Target-A | 1.022 | 1.166 | Input frequency awareness only |
| Full (Ours) | 0.994 | 1.124 | Bidirectional frequency awareness |
| only 48kHz | 1.127 | 1.198 | Trained on 48kHz data only |

### Key Findings

- **Incremental contributions of frequency-aware training are clear**: Data filtering, input frequency awareness, and output frequency awareness each successively improve performance, yielding a total LSD reduction of approximately 20%.
- **Cascaded system substantially outperforms direct training**: In the 16→96kHz setting, the cascaded model reduces LSD(0–48) by 0.415 and improves ViSQOL by 0.32 compared to a direct any-to-96kHz model — demonstrating that specializing each stage on a specific frequency band is more effective.
- **Noise prediction outperforms data prediction**: In the latent space, the noise prediction objective outperforms the data prediction objective commonly used in bridge model literature.
- A model fine-tuned on VCTK further surpasses the zero-shot version, achieving SigMOS of 3.095, exceeding the GAN-based method AP-BWE (3.082).
- 192kHz super-resolution is achieved for the first time: LSD decreases from 1.913 (direct training) to 1.365 (cascaded).

## Highlights & Insights

- **Paradigm shift in LR→HR prior exploitation**: The bridge model's latent-to-latent path is fundamentally aligned with the nature of super-resolution — the LR waveform is not noise but an informationally degraded version of the HR target. This principle transfers to other conditional generation tasks such as image and video super-resolution.
- **Frequency-aware any-to-any training** is particularly elegant: fixed conditioning is replaced by learnable conditioning, enabling training on all available data while granting the model stronger frequency comprehension. This represents a general paradigm for overcoming data scarcity.
- **Alignment between prior augmentation strategy and bridge model properties**: Diffusion models use noise addition for cascaded augmentation, whereas the Dirac-boundary nature of bridge models makes blurring more natural. The idea of simulating the artifacts of a previous stage through degradation is applicable to any cascaded generative system.
- Breaking the 48kHz ceiling to reach 192kHz for the first time opens new possibilities for professional audio production.

## Limitations & Future Work

- In speech scenarios, the zero-shot model occasionally misidentifies low-frequency noise as sound-effect textures; domain adaptation could alleviate this.
- Training data for the 192kHz stage is extremely scarce, necessitating reliance on fine-tuning and data augmentation, leaving this stage undertrained.
- The 50-step SDE sampling is relatively slow; consistency distillation or flow matching could be explored for acceleration.
- VAE compression loss constitutes the system's upper bound — VAE reconstruction quality limits the final performance ceiling.
- The method has not been systematically evaluated under real-world degradation scenarios (reverberation, compression artifacts).

## Related Work & Insights

- **vs. AudioSR**: Performs noise-to-latent diffusion in a mel-spectrogram latent space, with LR serving only as a condition. This work performs latent-to-latent bridge modeling in a waveform latent space, enabling more natural and effective prior exploitation.
- **vs. Bridge-SR**: Applies a bridge model directly in the waveform domain (WaveNet architecture) with weak generalization. This work advances to the latent space with a DiT backbone.
- **vs. A2SB**: Applies a bridge model in the STFT domain but fills the high-frequency region with noise. This work avoids the "excavate-and-fill" approach entirely.

## Rating

- Novelty: ⭐⭐⭐⭐ — A systematic design integrating bridge models, waveform latent space, and frequency-aware training.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers three domains, multiple sampling rates, complete ablations, and cascaded validation.
- Writing Quality: ⭐⭐⭐⭐ — Clear diagrams and complete method descriptions.
- Value: ⭐⭐⭐⭐ — New state of the art in audio super-resolution, opening the direction beyond 48kHz.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Latent Harmony: Synergistic Unified UHD Image Restoration via Latent Space Regularization and Controllable Refinement](latent_harmony_synergistic_unified_uhd_image_restoration_via_latent_space_regula.md)
- [\[NeurIPS 2025\] Implicit Augmentation from Distributional Symmetry in Turbulence Super-Resolution](implicit_augmentation_from_distributional_symmetry_in_turbulence_super-resolutio.md)
- [\[NeurIPS 2025\] FIPER: Factorized Features for Robust Image Super-Resolution and Compression](fiper_factorized_features_for_robust_image_super-resolution_and_compression.md)
- [\[CVPR 2026\] RAW-Domain Degradation Models for Realistic Smartphone Super-Resolution](../../CVPR2026/image_restoration/rawdomain_degradation_models_smartphone_sr.md)
- [\[NeurIPS 2025\] Adaptive Discretization for Consistency Models](adaptive_discretization_for_consistency_models.md)

</div>

<!-- RELATED:END -->
