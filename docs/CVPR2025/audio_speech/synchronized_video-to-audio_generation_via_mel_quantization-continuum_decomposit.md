---
title: >-
  [Paper Note] Synchronized Video-to-Audio Generation via Mel Quantization-Continuum Decomposition
description: >-
  [CVPR 2025][Audio & Speech][Video-to-Audio Generation] Mel-QCD is proposed to decompose Mel spectrograms into three signals: semantic vectors (quantized), energy, and standard deviation (continuous). By predicting these signals from video via a V2X predictor, and combining ControlNet with textual inversion, this approach achieves comprehensive SOTA video-to-audio generation across eight metrics on VGGSound.
tags:
  - "CVPR 2025"
  - "Audio & Speech"
  - "Video-to-Audio Generation"
  - "Mel Spectrogram Decomposition"
  - "Vector Quantization"
  - "ControlNet"
  - "Diffusion Models"
date: 2026-05-08
content_hash: f851fe4ecff479bd
---

# Synchronized Video-to-Audio Generation via Mel Quantization-Continuum Decomposition

**Conference**: CVPR 2025  
**arXiv**: [2503.06984](https://arxiv.org/abs/2503.06984)  
**Code**: [Project Page](https://wjc2830.github.io/MelQCD/)  
**Area**: Image Generation  
**Keywords**: Video-to-Audio Generation, Mel Spectrogram Decomposition, Vector Quantization, ControlNet, Diffusion Models

## TL;DR

Mel-QCD is proposed to decompose Mel spectrograms into three signals: semantic vectors (quantized), energy, and standard deviation (continuous). By predicting these signals from video via a V2X predictor, and combining ControlNet with textual inversion, this approach achieves comprehensive SOTA video-to-audio generation across eight metrics on VGGSound.

## Background & Motivation

Video-to-audio (V2A) generation aims to synthesize audio that is semantically and temporally synchronized with silent video. Existing methods face a key challenge:

1. **Trade-off between Signal Completeness and Complexity**: The more detailed the control signal (e.g., full Mel spectrogram), the better the semantic and temporal alignment, but the more difficult it is to predict from video.
2. **Limitations of Prior Work**: FoleyCrafter only extracts onset signals, losing significant semantic detail; ReWaS utilizes energy signals, which also contain limited information.
3. **Infeasibility of Direct Mel Spectrogram Prediction**: The high dimensionality and continuous distribution of Mel spectrograms make direct prediction from video highly impractical.

The Core Problem of this work: How to better balance the completeness and prediction complexity of control signals?

## Method

### Overall Architecture

Mel-QCD consists of two stages: pre-training and training. In the pre-training stage, the Mel-QCD signal decomposition method is derived from audio, and the SVQ codebook is constructed. In the training phase, a V2X signal predictor is trained to predict the decomposed signals from video, which then control an Auffusion-based T2A diffusion model via ControlNet to generate audio.

### Key Designs

**1. Mel Signal Decomposition and Quantization-Continuum Separation**

- **Function**: Decomposes Mel spectrograms based on informational properties, employing different representation strategies for different components.
- **Mechanism**: The Mel signal $\mathbf{M}_{k,t}$ at each time slot $t$ is decomposed into energy $\mathbf{E}_t$ (mean), standard deviation $\mathbf{D}_t$, and a normalized semantic vector $\mathbf{S}_{.,t}$. A key finding is that semantic vectors $\mathbf{S}$ cluster within sound events (making them quantizable), whereas $\mathbf{E}$ and $\mathbf{D}$ are continuously distributed across sound events (requiring them to remain continuous).
- **Design Motivation**: Quantizing semantic vectors converts continuous prediction into classification tasks (reducing complexity from $\mathcal{O}(N)$ to $\mathcal{O}(M)$) while maintaining completeness. The frequency dimension is downsampled to $K'=8$, $\lambda=1$, resulting in a codebook size of $3^8=6561$, which is further decomposed into two $3^4=81$ classifications.

**2. V2X Multi-Signal Predictor**

- **Function**: Simultaneously predicts three signals—quantized semantic vectors, energy, and standard deviation—from the input video.
- **Mechanism**: The video is resampled to $\frac{T \times f_{mel}}{4}$ frames, and visual encoders extract frame features. The SVQ predictor uses a Transformer + MLP for classification (into two 81-class outputs), while energy and standard deviation are obtained via continuous regression using a Transformer + MLP. The three predicted signals are recombined into Mel-QCD: $\mathbf{M}^{qcd}_{k,t} = \hat{\mathbf{E}}_t + \hat{\mathbf{S}}_{k,t} \times \hat{\mathbf{D}}_t$.
- **Design Motivation**: Tailoring the prediction method (classification vs. regression) to each signal type leads to more accurate results than a unified prediction approach.

**3. Text Inversion for Enhanced Semantic Consistency**

- **Function**: Mitigates semantic shift caused by inaccurate Mel-QCD predictions.
- **Mechanism**: Predefined sound event text prompts are used, and an Inversion Adapter maps the video's CLIP visual embedding into pseudo-word tokens $\{V_1, ..., V_n\}$. These are concatenated with text tokens and fed into the CLIP text encoder to generate semantically enhanced textual guidance $C_T$.
- **Design Motivation**: Deviations are inevitable in localized Mel-QCD time slots; textual inversion provides global semantic correction.

### Loss & Training

The standard diffusion model denoising loss is used:

$$\mathcal{L} = \mathbb{E}_{\mathbf{z}_0, t, \mathbf{C}_S, \mathbf{C}_T, \epsilon \sim \mathcal{N}(0,1)} [\|\epsilon - \epsilon_\theta(\mathbf{z}_t, t, \mathbf{C}_S, \mathbf{C}_T)\|_2^2]$$

## Key Experimental Results

### Main Results: Comprehensive Comparison on VGGSound Test Set

| Method | FID↓ | MKL↓ | Class ACC↑ | W-Dis↓ | JS-Div↓ | IB-AA↑ | IB-AV↑ |
|------|------|------|-----------|--------|--------|--------|--------|
| SpecVQGAN | 19.31 | 6.47 | 5.64 | 0.45 | 0.10 | 0.18 | 0.13 |
| DiffFoley | 15.15 | 6.47 | 23.27 | 0.49 | 0.14 | 0.32 | 0.23 |
| VTA-LDM | 11.77 | 4.72 | 27.72 | 0.37 | 0.11 | 0.44 | 0.28 |
| FoleyCrafter | 13.11 | 4.14 | 31.54 | 0.43 | 0.13 | 0.48 | 0.29 |
| **Mel-QCD (Ours)** | **11.73** | **2.96** | **45.91** | **0.33** | **0.11** | **0.52** | 0.31 |

### Control Signal Comparison (AvSync15 Dataset)

| Control Signal | Proposer | GT FID↓ | GT Cls ACC↑ | Pred FID↓ | Pred Cls ACC↑ |
|---------|--------|---------|-----------|-----------|-------------|
| Mel-QCD | Ours | **47.57** | **66.67** | **61.00** | **64.67** |
| Onset | FoleyCrafter | 65.38 | 56.67 | 68.72 | 56.67 |
| Energy | ReWaS | 57.21 | 62.67 | - | - |

### Key Findings

- Mel-QCD achieves the best performance in 6 out of 8 metrics, with Class ACC improving by 14.37% (from 31.54 to 45.91).
- MKL decreases significantly from the second-best 4.14 to 2.96, indicating that the generated distribution is closer to the ground truth distribution.
- Even under high compression ($K'=8$, $\lambda=1$), the quantized semantic vector loses only a small amount of information.

## Highlights & Insights

1. **Insightful Spectrogram Decomposition**: Discovering that semantic vectors can be clustered and quantized while energy must remain continuous represents a profound understanding of audio representations.
2. **Converting Continuous Prediction to Classification**: The SVQ codebook converts high-dimensional regression into low-dimensional classification, significantly reducing prediction difficulty.
3. **Decompose-Recompose-Control Paradigm**: Provides a novel paradigm for signal representation in V2A tasks.

## Limitations & Future Work

- The SVQ codebook size ($3^8$) remains relatively large, and decomposing it into two $3^4$ classifications may introduce errors.
- Textual inversion is dependent on predefined sound event labels.
- Sensitive to the quality of training data (requiring 55K carefully filtered synchronized videos).

## Related Work & Insights

- **FoleyCrafter**: Uses onset signals to control T2A via ControlNet, but the information volume is too limited.
- **Auffusion**: A foundational T2A model; this work builds the V2A pipeline on top of it.
- **ReWaS**: Controls with energy signals, which provides more information than onset but is still insufficient.

## Rating

⭐⭐⭐⭐ — The core concept of signal decomposition is highly novel, and the quantization-continuum separation strategy is elegant. Comprehensive SOTA results on VGGSound validate the effectiveness of the method, and the analysis experiments are thorough.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VinTAGe: Joint Video and Text Conditioning for Holistic Audio Generation](vintage_joint_video_and_text_conditioning_for_holistic_audio_generation.md)
- [\[CVPR 2026\] OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text](../../CVPR2026/audio_speech/omnisonic_towards_universal_and_holistic_audio_generation_from_video_and_text.md)
- [\[CVPR 2025\] Towards Lossless Implicit Neural Representation via Bit Plane Decomposition](towards_lossless_implicit_neural_representation_via_bit_plane_decomposition.md)
- [\[CVPR 2025\] MultiFoley: Video-Guided Foley Sound Generation with Multimodal Controls](video-guided_foley_sound_generation_with_multimodal_controls.md)
- [\[ICML 2026\] Towards Streaming Synchronized Spatial Audio Generation via Autoregressive Diffusion Transformer](../../ICML2026/audio_speech/towards_streaming_synchronized_spatial_audio_generation_via_autoregressive_diffu.md)

</div>

<!-- RELATED:END -->
