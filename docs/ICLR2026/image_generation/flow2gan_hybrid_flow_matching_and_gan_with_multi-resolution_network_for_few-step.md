---
title: >-
  [Paper Note] Flow2GAN: Hybrid Flow Matching and GAN with Multi-Resolution Network for Few-step High-Fidelity Audio Generation
description: >-
  [ICLR 2026][Image Generation][Flow Matching] This paper proposes Flow2GAN, a two-stage training framework that first employs an improved Flow Matching objective to learn generative capabilities…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Flow Matching"
  - "GAN"
  - "Audio Generation"
  - "Multi-Resolution"
  - "Few-step Inference"
date: 2026-05-08
content_hash: 2808901cd675a3a3
---

# Flow2GAN: Hybrid Flow Matching and GAN with Multi-Resolution Network for Few-step High-Fidelity Audio Generation

**Conference**: ICLR 2026
**arXiv**: [2512.23278](https://arxiv.org/abs/2512.23278)
**Code**: [GitHub](https://github.com/k2-fsa/Flow2GAN)
**Area**: Diffusion Models / Audio Generation
**Keywords**: Flow Matching, GAN, Audio Generation, Multi-Resolution, Few-step Inference

## TL;DR
This paper proposes Flow2GAN, a two-stage training framework that first employs an improved Flow Matching objective to learn generative capabilities, then fine-tunes with a GAN to achieve few-step (1/2/4 steps) high-fidelity audio generation, combined with a multi-resolution network architecture that processes Fourier coefficients at different time-frequency resolutions.

## Background & Motivation

**Background**: Audio generation is dominated by two paradigms — GANs (e.g., HiFi-GAN, BigVGAN), which capture multi-granularity audio details via carefully designed discriminators and enable single-step efficient inference, and diffusion/Flow Matching models, which offer stable training and high generation quality but require multi-step sampling.

**Limitations of Prior Work**: GAN training converges slowly and is prone to mode collapse; diffusion-based methods incur high computational cost due to multi-step inference. Existing acceleration approaches (distillation, consistency training, etc.) often sacrifice quality or require expensive retraining.

**Key Challenge**: The unique properties of audio signals pose additional challenges for Flow Matching — (a) silence regions and zero-energy frequency bins require precise noise cancellation, making velocity estimation difficult; (b) MSE loss treats all regions uniformly, which misaligns with auditory perception where errors in quiet regions are more perceptible.

**Goal**: To simultaneously achieve the stable training properties of FM and the efficient few-step inference capability of GANs, while optimizing the FM training objective for audio-specific characteristics.

**Key Insight**: Reformulating the FM objective from velocity estimation to endpoint estimation to avoid difficulty in velocity estimation over silent regions; introducing spectral energy-adaptive loss scaling; and applying GAN fine-tuning to accelerate inference.

**Core Idea**: Improved FM for pre-training (endpoint estimation + energy-adaptive loss scaling) followed by GAN fine-tuning for few-step high-fidelity generation.

## Method

### Overall Architecture
Two-stage training: Stage 1 trains the generative model with improved Flow Matching (mapping conditional input representations to Fourier coefficients of audio waveforms); Stage 2 constructs a few-step generator from the pre-trained model and fine-tunes it with GAN discriminators. A multi-branch ConvNeXt network processes STFT coefficients at three time-frequency resolutions.

### Key Designs

1. **Endpoint Estimation**:

    - Function: Reformulates the FM objective from predicting velocity $v_t = x_1 - x_0$ to predicting the endpoint $\hat{x}_1 = g_\theta(x_t, t|\mathbf{c})$.
    - Mechanism: The loss is rewritten as $\mathcal{L}'_{\text{FM}} = \mathbb{E}_{t,x_0,x_1}[\|g_\theta(x_t,t|\mathbf{c}) - x_1\|^2]$, and the Euler step during inference becomes $x_{t_{i+1}} = x_{t_i} + (t_{i+1}-t_i)\frac{g_\theta(x_{t_i},t_i|\mathbf{c}) - x_{t_i}}{1-t_i}$.
    - Design Motivation: Velocity estimation in silent regions requires the model to simultaneously learn $x_1 - x_0$ and $-x_0$, whereas endpoint estimation unifies the target as reconstructing the clean signal, yielding a more consistent learning objective. Removing the weighting factor $\frac{1}{(1-t)^2}$ directs the model's focus toward small $t$ values.

2. **Spectral Energy-Adaptive Loss Scaling**:

    - Function: Scales the prediction error by the inverse of the reference spectral energy, emphasizing quiet regions.
    - Mechanism: $\mathcal{L}''_{\text{FM}} = \mathbb{E}[\sum_{i,j}(\frac{\mathcal{S}(g_\theta - x_1)}{\sqrt{\mathcal{S}(x_1)+\epsilon}})_{i,j}]$, where $\mathcal{S}(x) = \text{LinFB}(|\text{STFT}(x)|^2)$.
    - Design Motivation: Unlike per-frame scaling in prior work, this approach accounts for energy variation across both time and frequency dimensions, better aligning with human auditory perception.

3. **GAN Fine-tuning**:

    - Function: Constructs an $N$-step generator $G_\theta^N$ from the pre-trained FM model and fine-tunes it with GAN discriminators.
    - Mechanism: Adversarial training with MPD and MRD discriminators; the loss combines HingeGAN loss, L1 feature matching loss, and multi-scale Mel reconstruction loss. Gradients are back-propagated end-to-end through the multi-step generator.
    - Design Motivation: FM pre-training provides a strong initialization, enabling the GAN to rapidly improve perceptual detail quality with minimal fine-tuning. Single-step models fine-tuned from standard FM pre-training perform significantly worse than those from the improved FM pre-training.

4. **Multi-Resolution Network Architecture**:

    - Function: Three branches process Fourier coefficients from STFT at different resolutions.
    - Mechanism: Each branch uses ConvNeXt to process complex STFT coefficients at a specific time-frequency resolution; the waveforms recovered via ISTFT from each branch are summed to yield the final output. Branches with lower frame rates use larger embedding dimensions.
    - Design Motivation: Compared to the single-resolution design of Vocos, the multi-resolution approach better captures the complexity of audio signals.

### Loss & Training
- Stage 1: Improved FM loss (endpoint estimation + energy-adaptive scaling), optimized with the ScaledAdam optimizer.
- Stage 2: HingeGAN loss + L1 feature matching + multi-scale Mel reconstruction loss (7 window sizes).

## Key Experimental Results

### Main Results
LibriTTS test set (Mel-spectrogram conditioned):

| Model | Params | PESQ↑ | ViSQOL↑ | V/UV F1↑ | Periodicity↓ | FSD↓ |
|-------|--------|-------|---------|----------|-------------|------|
| BigVGAN-v2* | 112.4M | 4.379 | 4.971 | 0.978 | 0.055 | 0.014 |
| PeriodWave-Turbo (4 steps) | 70.2M | 4.434 | 4.965 | 0.958 | 0.096 | 0.020 |
| WaveFM (1 step) | 19.5M | 3.540 | 4.894 | 0.943 | 0.124 | 0.098 |
| Flow2GAN 1 step | 78.9M | 4.189 | 4.957 | 0.975 | 0.063 | 0.028 |
| Flow2GAN 2 steps | 78.9M | 4.440 | 4.979 | 0.983 | 0.044 | 0.023 |
| Flow2GAN 4 steps | 78.9M | **4.484** | **4.986** | **0.985** | **0.037** | 0.016 |

### Ablation Study

| Configuration | PESQ↑ | ViSQOL↑ | Notes |
|---------------|-------|---------|-------|
| Standard FM + GAN fine-tuning (1 step) | Significant drop | Drop | Insufficient pre-training from standard FM |
| Improved FM (no GAN), 2 steps | Moderate | Moderate | Improved but lacks fine detail |
| Improved FM + GAN, 1 step | 4.189 | 4.957 | Full pipeline |
| Improved FM + GAN, 4 steps | 4.484 | 4.986 | More steps yield better quality |

### Key Findings
- The improved FM objective (endpoint estimation + energy-adaptive scaling) is a critical prerequisite for successful GAN fine-tuning — single-step GAN fine-tuning from standard FM pre-training performs substantially worse than from the improved FM variant.
- Applying spectral energy scaling across both time and frequency dimensions outperforms scaling along the time dimension alone.
- GAN fine-tuning requires only a small number of iterations to achieve rapid quality gains, constituting an efficient "free lunch."
- The multi-resolution architecture yields a clear improvement over the single-resolution design of Vocos.

## Highlights & Insights
- **Elegance of Endpoint Estimation**: A simple reformulation of the training objective effectively circumvents velocity estimation difficulties in silent regions without requiring complex architectural changes, yet yields significant improvements.
- **Complementarity of the Two-Stage Strategy**: FM provides stable training and a strong initialization, while GAN contributes perceptual detail enhancement and few-step inference capability — each stage contributes its strengths. This hybrid approach is generalizable to other generative domains.
- **Perceptual Alignment via Energy Scaling**: Using the inverse of signal energy as a loss weight directs the model's attention toward quiet regions that are more perceptible to human listeners — a simple yet effective design choice.

## Limitations & Future Work
- The model size of 78.9M parameters is substantially larger than Vocos (13.5M), warranting attention to computational efficiency.
- The clamping range (0.01–100) for energy scaling is empirically determined and lacks in-depth analysis of the optimal range.
- The selection of three STFT resolutions lacks a systematic ablation, and a more optimal configuration may exist.
- Evaluation is limited to speech and general audio; extension to other audio domains such as music generation remains unexplored.

## Related Work & Insights
- **vs. BigVGAN**: BigVGAN relies on large-scale data for pure GAN training; Flow2GAN's two-stage strategy achieves comparable performance on the same data as BigVGAN's large-data-trained version.
- **vs. PeriodWave-Turbo**: Both share the idea of GAN fine-tuning over FM pre-training; however, Flow2GAN's single-step model — built on the improved FM — significantly outperforms the counterpart built on standard FM.
- **vs. WaveFM (Consistency Distillation)**: Flow2GAN's GAN fine-tuning approach substantially surpasses consistency distillation in single-step generation quality.

## Rating
- Novelty: ⭐⭐⭐⭐ The combined design of endpoint estimation and energy-adaptive scaling is tailored to audio-specific properties; the two-stage training strategy is practically motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across Mel-conditioned, token-conditioned, TTS vocoder settings, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Well-structured, with thorough comparisons and an online demo for listening evaluation.
- Value: ⭐⭐⭐⭐ Directly applicable to the audio generation field with open-sourced code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Multi-agent Coordination via Flow Matching](multi-agent_coordination_via_flow_matching.md)
- [\[ICLR 2026\] Laplacian Multi-scale Flow Matching for Generative Modeling](laplacian_multi-scale_flow_matching_for_generative_modeling.md)
- [\[ICLR 2026\] DistillKac: Few-Step Image Generation via Damped Wave Equations](distillkac_few-step_image_generation_via_damped_wave_equations.md)
- [\[CVPR 2026\] Frequency-Aware Flow Matching for High-Quality Image Generation](../../CVPR2026/image_generation/freqflow_frequency_aware_flow_matching.md)
- [\[CVPR 2026\] Few-shot Acoustic Synthesis with Multimodal Flow Matching](../../CVPR2026/image_generation/few-shot_acoustic_synthesis_with_multimodal_flow_matching.md)

</div>

<!-- RELATED:END -->
