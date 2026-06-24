---
title: >-
  [Paper Note] BinauralFlow: A Causal and Streamable Approach for High-Quality Binaural Speech Synthesis with Flow Matching Models
description: >-
  [ICML 2025][Audio & Speech][Flow Matching] This work proposes BinauralFlow, a streamable binaural speech synthesis framework based on conditional Flow Matching. Incorporating a causal U-Net architecture and a continuous inference pipeline, it produces high-fidelity, streamable binaural audio. In perceptual evaluations, a 42% confusion rate demonstrates that the synthesized audio is virtually indistinguishable from real recordings.
tags:
  - "ICML 2025"
  - "Audio & Speech"
  - "Flow Matching"
  - "Binaural Audio Synthesis"
  - "Causal U-Net"
  - "Streamable Inference"
  - "Spatial Audio"
date: 2026-05-08
content_hash: bf834d78e4088dc6
---

# BinauralFlow: A Causal and Streamable Approach for High-Quality Binaural Speech Synthesis with Flow Matching Models

**Conference**: ICML 2025  
**arXiv**: [2505.22865](https://arxiv.org/abs/2505.22865)  
**Code**: [Project Page](https://liangsusan-git.github.io/project/binauralflow/)  
**Area**: Image Generation  
**Keywords**: Flow Matching, Binaural Audio Synthesis, Causal U-Net, Streamable Inference, Spatial Audio

## TL;DR

This work proposes BinauralFlow, a streamable binaural speech synthesis framework based on conditional Flow Matching. Incorporating a causal U-Net architecture and a continuous inference pipeline, it produces high-fidelity, streamable binaural audio. In perceptual evaluations, a 42% confusion rate demonstrates that the synthesized audio is virtually indistinguishable from real recordings.

## Background & Motivation

Binaural rendering aims to synthesize binaural audio (one channel for each ear) simulated to mimic natural hearing, based on mono audio and the positions of the speaker/listener. This is crucial for immersive applications such as VR/AR/MR, games, and films.

Prior work faces two core challenges:

**High-Quality Rendering Challenge**: Generating binaural audio indistinguishable from real recordings requires precise modeling of binaural cues (ILD/ITD), room reverberation, and ambient noise. Traditional DSP methods (e.g., SoundSpaces) rely on simplified geometric simulations and non-personalized HRTFs, resulting in limited quality. Regression-based neural network methods (e.g., WarpNet) fail to synthesize reverberation and noise that are absent in the input signal.

**Streamable Inference Challenge**: Real-time applications require continuous, low-latency streaming generation. However, existing generative models (such as diffusion models) employ non-causal architectures (non-causal convolutions + global attention) and multi-step iterative inference, failing to support streaming processing.

**Key Insight**: Transforming binaural rendering from a regression problem into a generative problem—reverberation and ambient noise are stochastic in nature and difficult for regression methods to fit, whereas generative models can naturally model this stochasticity.

## Method

### Overall Architecture

BinauralFlow consists of three core components:

1. **Conditional Flow Matching Model**: Generates binaural audio via flow matching in the complex STFT spectrogram domain, conditioned on the speaker/listener poses and the mono input.
2. **Causal U-Net Architecture**: Ensures temporal causality, where the prediction for each frame relies solely on historical information.
3. **Continuous Inference Pipeline**: Includes streamable STFT/ISTFT, a buffer bank, a midpoint solver, and an early skip schedule to achieve seamless streamable generation.

### Key Designs

#### Conditional Flow Matching

**STFT Domain Modeling**: Mono audio $x$ and binaural audio $y$ are transformed into the time-frequency domain via STFT, where the mono channel is duplicated along the channel dimension to form a two-channel input.

**Noise Sampling Strategy**: Noise is sampled with the mono input as the center (instead of zero-centered), ensuring that the starting point of the flow already contains rich speech information.

**Optimal Transport Flow Formulation**: The flow is formulated as $\phi_t(z) = t \cdot y + (1-t) \cdot x + (1-t) \cdot \sigma \cdot \varepsilon$. When $t=0$, the distribution revolves around the mono input; when $t=1$, it collapses to the target binaural audio.

**Conditional Vector Field Matching**: The network $u_t$ is trained to match the vector field $v_t = y - z$ using an $L_1$ loss. The model takes four conditions as inputs: time step $t$, transmitter pose $p_{tx}$ (position + quaternion rotation, 7-dimensional), receiver pose $p_{rx}$, and the mono spectrogram $x$.

**Differences from Simplified Flow Matching**: (1) Simplified FM uses a minimal perturbation ($1e-4$), almost degrading into a deterministic task; Ours uses normal-scale Gaussian noise ($\sigma=0.5$) to maintain generative stochasticity. (2) Ours employs mono audio as a generation condition to enhance robustness, which Simplified FM cannot do as it leads to model collapse.

#### Causal U-Net Architecture

To achieve streamable rendering, a comprehensive causal transformation is applied to the standard U-Net:

- **Causal Convolutional Layers**: $3 \times 3$ convolutions with stride=1 and single-sided padding=2, restricting the receptive field to historical information only.
- **Causal Down/Upsampling**: Downsampling uses $4 \times 4$ convolutions (stride=2), and upsampling uses $4 \times 4$ transposed convolutions.
- **Causal Normalization**: GroupNorm is restricted to being computed within each independent frame, rather than across frames.
- **Activation Function**: SiLU (Sigmoid Linear Unit).
- **Condition Injection**: Time steps and poses are encoded via Random Gaussian Fourier Embedding + MLP, and then added to the hidden features.
- **Complex Number Processing**: The real and imaginary parts of the complex spectrogram are split into independent input channels, and merged back into the complex space after output.

The network contains a total of 7 causal 2D convolutional blocks (7 each for the contracting and expanding paths), performing 4 downsampling/upsampling operations.

#### Continuous Inference Pipeline

A causal backbone alone is insufficient for streaming inference—generative models require multi-step iteration, making it essential to ensure temporal causality across all inference steps.

- **Streamable STFT/ISTFT**: Adapted for streaming processing by adding buffers and adjusting padding methods. The buffer content is prepended to each chunk before processing, and the buffer is updated afterwards.
- **Buffer Bank**: Maintains buffers for each causal convolutional layer. Key design: different denoising steps cannot share the same buffer (otherwise, historical information would be overwritten). Therefore, a dictionary-like buffer bank $B = \{B_t\}$ indexed by time step $t$ is constructed.
- **Midpoint Solver**: A second-order ODE solver that achieves the best balance between accuracy and efficiency—generating more realistic background noise compared to Euler, and requiring fewer function evaluations compared to Heun.
- **Early Skip Schedule**: Skips the first half of the time steps and only performs denoising in the second half. Flow matching can correct errors from the first half during the second half, reducing the inference steps from 12 to 6 (compared to 30 steps required by SGMSE).

### Loss & Training

- **Loss Function**: Conditional Flow Matching $L_1$ loss
- **Optimizer**: Adam, learning rate $1e-4$, weight decay $1e-5$
- **STFT Parameters**: Window length 512, hop size 128, Hann window
- **Input Length**: 32768 samples (approx. 0.683 seconds at 48kHz), spectrogram size $256 \times 257$
- **Noise Standard Deviation**: $\sigma = 0.5$
- **Inference Steps**: 6 steps (Midpoint Solver + Early Skip)
- **Large-scale Pre-training Strategy**: Pre-trained on 7700+ hours of data collected with an artificial binaural dummy head and loudspeakers, and then fine-tuned with a small amount of real human data, significantly improving data efficiency.

## Key Experimental Results

### Main Results

Dataset: 10 hours of self-collected, high-quality paired mono/binaural data (48kHz) from real human speakers and listeners in a non-anechoic chamber environment. Train/val/test split: 8.47/0.86/1.33 hours. The test set contains male and female speakers unseen during training.

| Method | Type | NFE | Speed(ms) | $L_2$ ($\times 1e-5$)↓ | Mag↓ | Phase↓ |
|------|------|-----|----------|-------------|------|--------|
| SoundSpaces 2.0 | DSP | 1 | - | 4.91 | 0.0129 | 1.58 |
| 2.5D Visual Sound | R | 1 | 1.1 | 2.78 | 0.0174 | 1.56 |
| WaveNet | R | 1 | 21.0 | 2.79 | 0.0175 | 1.57 |
| WarpNet | R | 1 | 21.9 | 2.79 | 0.0176 | 1.57 |
| BinauralGrad | G | 6 | 221.1 | 2.93 | 0.0143 | 1.33 |
| SGMSE | G | 30 | 770.2 | 1.55 | 0.0076 | 1.43 |
| **BinauralFlow** | **G** | **6** | **163.0** | **1.00** | **0.0071** | **1.33** |

BinauralFlow comprehensively outperforms the SOTA on all metrics, achieving the fastest inference speed among generative models (163ms vs 770ms for SGMSE), and requiring an NFE of only 6 (vs 30 for SGMSE).

### Ablation Study

| Configuration | $L_2$ ($\times 1e-5$)↓ | Mag↓ | Phase↓ | Description |
|------|-------------|------|--------|------|
| Simplified FM | 1.86 | 0.0101 | 1.35 | Degrades into a near-deterministic task |
| **BinauralFlow** | **1.00** | **0.0071** | **1.33** | Maintains generative stochasticity + mono conditioning |
| Euler Solver (NFE=6) | 0.90 | 0.0066 | 1.24 | Low numerical values but poor noise quality |
| Midpoint Solver (NFE=6) | 1.00 | 0.0071 | 1.33 | Best balance of quality and efficiency |
| Heun Solver (NFE=6) | 16.86 | 0.0499 | 1.44 | Poor performance due to insufficient steps |
| Heun Solver (NFE=30) | 1.27 | 0.0087 | 1.36 | Requires 30 steps to be usable |

Real-time factor (RTF): RTF = 0.239 at NFE = 6 (on an RTX 4090 GPU), and RTF = 0.04 at NFE = 1, demonstrating potential for real-time streaming generation.

### Key Findings

1. **Generative vs. Regression**: Modeling binaural rendering as a generative task is key—regression-based methods cannot synthesize reverberation and noise absent in the input, whereas generative methods comprehensively outperform regression methods.
2. **Outstanding Perceptual Evaluation**: The confusion rate in A-B testing is 42% (with a theoretical limit of 50%), showing that humans can barely distinguish synthetic audio from real recordings, whereas SGMSE achieves only 21% and BinauralGrad only 3%.
3. **Effectiveness of Early Skip**: Skipping the first half of the time steps does not degrade quality, as flow matching can automatically correct errors in the second half; in contrast, skipping the second half leads to a degradation in background noise modeling.
4. **Pre-training Significantly Boosts Data Efficiency**: After pre-training on 7,700 hours of artificial dummy head data, zero-shot performance already matches the efficacy of training from scratch using 1%-5% of real human data.
5. **Indispensability of Continuous Inference Pipeline**: Processing each chunk independently using a non-streaming pipeline produces obvious block-boundary artifacts, whereas the buffer bank design ensures smooth transition across chunks.

## Highlights & Insights

1. **The Power of Redefining the Problem**: The perspective shift from regression to generation is the core contribution of this work—since the stochasticity of reverberation and noise is inherently a generative task, regression methods are fundamentally limited.
2. **Systemic Nature of Causal Transformation**: Rather than simply replacing standard convolutions with causal ones, the work causalizes the entire pipeline—from convolutions, normalization, downsampling/upsampling, and buffer management to the inference pipeline—forming a complete streaming solution.
3. **Empirical Evidence of Flow Matching Outperforming Diffusion Models**: 6-step flow matching comprehensively outperforms 30-step SGMSE (a diffusion model), further validating the efficiency advantages of flow matching in the audio generation domain.
4. **Ingenuity of Noise Design**: Sampling noise centered on the mono input rather than standard Gaussian noise ensures that the starting point of generation already contains speech structural information, reducing the difficulty of generation.

## Limitations & Future Work

1. **High Data Collection Cost**: Gathering paired data requires professional equipment (binaural microphones, OptiTrack motion capture systems) and controlled environments. Although pre-training strategies alleviate this issue, a high barrier to entry remains.
2. **Large Model Size**: The model has 314.5 MB in parameter size, exceeding BinauralGrad (52.9 MB) and SGMSE (273.6 MB).
3. **Single-Environment Generalization**: The training data originates from a single room, and generalization capability across different rooms or acoustic environments has not been fully verified.
4. **Deepfake Risk**: Highly realistic audio synthesis carries a potential risk of being abused for audio deepfakes.
5. **Contradiction Between Numerical Metrics and Perceptual Quality**: The Euler solver yields better (lower) numerical metrics but worse perceptual quality (realism of background noise), suggesting that current evaluation metrics may not fully align with human perception.

## Related Work & Insights

- **BinauralGrad (NeurIPS 2022)**: A two-stage diffusion model, previous SOTA but slow in inference, employing a hybrid regression-generation strategy.
- **SGMSE (2023)**: A diffusion model in the complex STFT domain, which performs well in speech enhancement but requires 30 steps and lacks streaming support.
- **Flow Matching (Lipman et al., 2022)**: Optimal transport formulation, more efficient than diffusion models.
- **CosyVoice 2 (2024)**: A chunk-aware causal flow matching TTS model, which lacks buffer design for convolutional layers, potentially leading to inter-block discontinuities.
- **Insights**: The causal transformation and buffer bank design in this work can be transferred to other generative tasks requiring real-time streaming inference (e.g., real-time speech synthesis, real-time video generation, etc.).

## Rating

| Dimension | Score (1-5) | Description|
|------|-----------|------|
| Novelty | 4 | Generative perspective + full-pipeline causal transformation + buffer bank design |
| Technical Depth | 5 | Highly complete and systematic from mathematical formulation to engineering implementation |
| Experimental Thoroughness | 5 | Quantitative + qualitative + perceptual tests + ablations + public datasets |
| Writing Quality | 4 | Clear structure with adequate details |
| Value | 4 | Directly addresses real-time application needs in VR/AR |
| **Overall Score** | **4.4** | A highly solid and systematic work |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Shallow Flow Matching for Coarse-to-Fine Text-to-Speech Synthesis](../../NeurIPS2025/audio_speech/shallow_flow_matching_for_coarse-to-fine_text-to-speech_synthesis.md)
- [\[NeurIPS 2025\] LeVo: High-Quality Song Generation with Multi-Preference Alignment](../../NeurIPS2025/audio_speech/levo_high-quality_song_generation_with_multi-processing_refined_supervision.md)
- [\[ICLR 2026\] Flow2GAN: Hybrid Flow Matching and GAN with Multi-Resolution Network for Few-step High-Fidelity Audio Generation](../../ICLR2026/audio_speech/flow2gan_hybrid_flow_matching_and_gan_with_multi-resolution_network_for_few-step.md)
- [\[ICML 2025\] Sortformer: A Novel Approach for Permutation-Resolved Speaker Supervision in Speech-to-Text Systems](sortformer_a_novel_approach_for_permutation-resolved_speaker_supervision_in_spee.md)
- [\[ICLR 2026\] AlignSep: Temporally-Aligned Video-Queried Sound Separation with Flow Matching](../../ICLR2026/audio_speech/alignsep_temporally-aligned_video-queried_sound_separation_with_flow_matching.md)

</div>

<!-- RELATED:END -->
