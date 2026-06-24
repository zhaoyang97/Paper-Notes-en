---
title: >-
  [Paper Note] OmniAudio: Generating Spatial Audio from 360-Degree Video
description: >-
  [ICML 2025][Audio & Speech][Spatial Audio Generation] This work proposes the OmniAudio framework, which achieves first-of-its-kind spatial audio generation in First-order Ambisonics (FOA) format from 360-degree panoramic videos. By incorporating a coarse-to-fine self-supervised pre-training paradigm and a dual-branch video encoding architecture, OmniAudio achieves state-of-the-art (SOTA) performance on the self-collected Sphere360 dataset.
tags:
  - "ICML 2025"
  - "Audio & Speech"
  - "Spatial Audio Generation"
  - "360-Degree Video"
  - "First-order Ambisonics"
  - "Flow Matching"
  - "Self-Supervised Pre-training"
date: 2026-05-08
content_hash: 95e9955be346a594
---

# OmniAudio: Generating Spatial Audio from 360-Degree Video

**Conference**: ICML 2025  
**arXiv**: [2504.14906](https://arxiv.org/abs/2504.14906)  
**Code**: [github.com/liuhuadai/OmniAudio](https://github.com/liuhuadai/OmniAudio)  
**Area**: Audio & Speech  
**Keywords**: Spatial Audio Generation, 360-Degree Video, First-order Ambisonics, Flow Matching, Self-Supervised Pre-training

## TL;DR

This work proposes the OmniAudio framework, which achieves first-of-its-kind spatial audio generation in First-order Ambisonics (FOA) format from 360-degree panoramic videos. By incorporating a coarse-to-fine self-supervised pre-training paradigm and a dual-branch video encoding architecture, OmniAudio achieves state-of-the-art (SOTA) performance on the self-collected Sphere360 dataset.

## Background & Motivation

Traditional video-to-audio methods face two key limitations: (1) they only generate non-spatial audio (such as mono or stereo), lacking 3D directional information; (2) they only process perspective videos with a limited field of view (FoV), omitting out-of-view sound sources. For instance, when a train passes through a panoramic video but is not visible in the frontal perspective view, traditional methods fail to capture this sound source.

Spatial audio (specifically in the FOA format) can preserve 3D auditory localization cues, but existing methods (such as ViSAGe and Diff-SAGe) still rely on fixed perspective inputs. 360-degree panoramic videos naturally provide complete spherical visual coverage, enabling the simultaneous observation of all sounding objects and their spatial relationships.

This paper defines the novel task of **360V2SA** (360-degree Video to Spatial Audio), which presents three main challenges: (1) scarcity of paired data; (2) precise audio-visual synchronization on a sphere; and (3) complexity in generating high-fidelity spatial audio.

## Method

### Overall Architecture

OmniAudio consists of two core stages:

1. **Coarse-to-Fine Self-Supervised Pre-training**: Utilizes large-scale non-spatial audio and FOA spatial audio for two-stage pre-training to learn general audio patterns.
2. **Spatially-Aware Supervised Fine-Tuning**: Fine-tunes a Diffusion Transformer (DiT) conditioned on dual-branch video representations to generate FOA audio.

The generative backbone adopts **Conditional Flow Matching**, which generates samples by learning vector fields of velocities mapping from noise to data, offering more stable training compared to DDPM.

### Key Designs

#### 1. Spatial Audio VAE

FOA audio comprises four channels (W/X/Y/Z), which respectively encode omnidirectional acoustic pressure, front-back, left-right, and up-down sound pressure gradients. Conventional VAEs only support stereo audio; this work proposes the following modifications:

- Initialize the four-channel FOA VAE with pre-trained stereo VAE weights.
- Remove the Mid-Side STFT loss tailored for stereo audio, and instead apply equal-weighted (1/4 each) reconstruction loss individually to each of the W/X/Y/Z channels.
- Utilize the Snake activation function and the Descript Audio Codec architecture based on the Stable Audio framework to achieve high-quality reconstruction under high compression ratios.

#### 2. Dual-Branch Video Representation

To encode both global scene context and local fine-grained details:

- **Global Branch**: Pads the 360-degree Equirectangular Projection (ERP) video into a 1:1 square crop and feeds it into a frozen MetaCLIP-Huge image encoder to extract global panoramic features.
- **Local Branch**: Extracts the frontal 120° perspective video from the 360-degree video, which is then projected linearly and fed into the same MetaCLIP encoder to capture local detailed features.

Fusion mechanism: The local FoV features are upsampled to match the sequence length of the audio latent representation and added element-wise; the global 360 features are max-pooled and utilized as the global conditioning input for the DiT.

#### 3. Coarse-to-Fine Self-Supervised Pre-training

- **Coarse-grained Stage**: Trained on ~2M non-spatial audio samples (FreeSound + AudioSet + VGGSound). Non-spatial audio is first converted into FOA format (setting Y/Z to zero, W = Left + Right, X = Left - Right) and compressed into latent representations via the Spatial VAE. A flow matching model is then trained with token masking to reconstruct the masked portions.
- **Fine-grained Stage**: Pre-trained exclusively on FOA spatial audio, allowing the model to learn spatial dynamic features specific to FOA.

Masking strategy: Applies conditioning masks to audio latent representations with a probability of $p_{cond}=0.1$, randomly masking frames with a designated minimum mask span.

### Loss & Training

**Pre-training Loss**: Standard Conditional Flow Matching objective (computed only on the masked portions).

**Fine-tuning Loss**: Flow Matching objective incorporating dual-branch video conditions, with timesteps sampled from a logit-normal distribution. CFG-Scale = 5 is applied during inference.

**VAE Loss**: Weighted four-channel multi-resolution STFT loss + KL divergence loss + discriminator loss.

**Training Details**:

- VAE: 24× A800 GPUs, batch size 144, 500K steps + 300K additional steps with frozen encoder
- Pre-training: 8× A100 GPUs, batch size 256, 100K steps
- Fine-tuning: 8× A100 GPUs, batch size 256, 50K steps, learning rate 5e-5 (AdamW)
- DiT Architecture (Large): 1536 embedding dimension, 24 layers, 24 attention heads, totaling 1.2B parameters

### Sphere360 Dataset

This work introduces the first large-scale 360V2SA dataset, featuring 103K video clips (10 seconds each), totaling 288 hours and covering 288 distinct audio events.

Data collection pipeline: YouTube keyword search $\rightarrow$ 360°/FOA technical filtering $\rightarrow$ channels-level and video-level two-stage scraping $\rightarrow$ semi-automatic cleaning (removal of static videos, silence detection, speech filtering, and ImageBind audio-visual alignment checks).

## Key Experimental Results

### Main Results

| Model | Params | FD ↓ | KL ↓ | ΔAngular ↓ | MOS-SQ ↑ | MOS-AF ↑ | Inference Time |
|------|--------|------|------|------------|----------|----------|---------|
| **Sphere360-Bench (In-distribution)** | | | | | | | |
| GT | - | - | - | - | 88.41 | 90.12 | - |
| Diff-Foley + AS | 0.94B | 331.05 | 3.56 | - | 69.87 | 71.12 | 2.40s |
| MMAudio + AS | 1.03B | 271.15 | 2.39 | - | 75.34 | 77.56 | 3.01s |
| ViSAGe (FoV) | 0.36B | 210.87 | 2.90 | 1.49 | 73.45 | 74.89 | 22.37s |
| ViSAGe (360) | 0.36B | 219.66 | 2.96 | 1.51 | 74.12 | 75.34 | 22.37s |
| **OmniAudio** | **1.22B** | **88.30** | **1.58** | **1.28** | **84.67** | **87.23** | **0.92s** |
| **YT360-Test (Out-of-distribution)** | | | | | | | |
| Diff-Foley + AS | 0.94B | 361.65 | 2.22 | - | 67.21 | 70.34 | 2.40s |
| MMAudio + AS | 1.03B | 190.40 | 1.71 | - | 73.25 | 76.77 | 3.01s |
| ViSAGe (FoV) | 0.36B | 199.09 | 1.86 | 1.99 | 71.82 | 72.17 | 22.37s |
| ViSAGe (360) | 0.36B | 225.52 | 1.95 | 1.98 | 72.45 | 72.96 | 22.37s |
| **OmniAudio** | **1.22B** | **92.57** | **1.64** | **1.27** | **80.37** | **83.49** | **0.92s** |

### Ablation Study

**Ablation Study on Self-Supervised Pre-training Strategy**:

| Configuration | FD ↓ | KL ↓ | ΔAngular ↓ | Description |
|------|------|------|------------|------|
| Coarse-to-Fine | **88.30** | **1.58** | **1.28** | Full two-stage pre-training |
| w/ fine only | 97.57 | 1.82 | 1.28 | Pre-trained with FOA only |
| w/ coarse only | 97.26 | 1.78 | 1.30 | Pre-trained with non-spatial audio only |
| w/o PT | 104.57 | 1.83 | 1.32 | Without pre-training |

**Ablation Study on Dual-Branch Design**:

| Configuration | FD ↓ | KL ↓ | ΔAngular ↓ | Description |
|------|------|------|------------|------|
| ERP + Per (Dual-branch) | **88.30** | **1.58** | 1.28 | Panoramic + Perspective dual-branch |
| w/ Per only | 88.80 | 1.87 | 1.33 | Perspective video only |
| w/ EAC only | 93.37 | 1.84 | 1.30 | Equi-angular cubemap (EAC) only |
| w/ ERP only | 97.83 | 1.87 | 1.28 | Equirectangular projection (ERP) only |

**Ablation Study on Model Scale**:

| Scale | Params | FD ↓ | KL ↓ | ΔAngular ↓ |
|------|--------|------|------|------------|
| Large | 1.2B | **88.30** | **1.58** | **1.26** |
| Medium | 472M | 104.19 | 1.82 | 1.28 |
| Small | 291M | 108.50 | 1.91 | 1.29 |

### Key Findings

1. **OmniAudio significantly outperforms all baselines**: FD decreases from the best baseline's 210.87 to 88.30 (on Sphere360), and the inference time of 0.92s is substantially faster (24 times faster) than ViSAGe's 22.37s.
2. **360-degree video is more critical than perspective video**: Panoramic inputs consistently outperform perspective-only inputs across both spatial and non-spatial metrics.
3. **Coarse-to-fine pre-training is indispensable**: Removing either stage leads to a degradation in FD by 9 to 16 points.
4. **Strong OOD generalization**: OmniAudio maintains a significant advantage on the out-of-distribution (OOD) YT360 test set.
5. **Cascaded approaches (V2A + spatialization) yield inferior results**: End-to-end generation of FOA spatial audio is superior to cascaded pipelines.

## Highlights & Insights

1. **Forward-looking task definition**: The 360V2SA task is overlooked yet highly critical; VR/AR scenarios inherently require panoramic video paired with spatial audio.
2. **Clever domain transfer strategy**: Using Spatial VAE to convert non-spatial audio to FOA format for participation in pre-training effectively compensates for the scarcity of spatial audio data.
3. **Complementary dual-branch design**: The global branch provides scene context ("what sound sources exist"), whereas the local branch provides fine-grained details ("where the sound originates"). The fusion strategy is elegant and highly effective.
4. **Engineering value of the dataset**: The semi-automatic collection and cleaning pipeline for Sphere360 exhibits outstanding reproducibility, incorporating critical steps such as static frame removal, silence detection, speech filtering, and audio-visual alignment validation.
5. **Inference efficiency advantage**: The integration of Flow Matching and DiT enables inference in just 0.92s, rendering it significantly faster than autoregressive competitors.

## Limitations & Future Work

1. **Difficulty in multi-source scenarios**: When multiple sound sources are present simultaneously, the model tends to confuse auditory events (e.g., misclassifying instrument sounds as applause).
2. **Limited scale of data**: 103K samples remain insufficient to fully model the complexities of real-world 360V2SA scenarios.
3. **FOA is limited to first-order Ambisonics**: Higher-order Ambisonics could be modeled in the future to further enhance spatial resolution and precision.
4. **Static FoV extraction strategy**: Utilizing only the frontal 120° field of view for the local branch could be improved by adaptively tracking and selecting views containing the primary acoustic sources.
5. **Lack of temporal dynamic modeling**: Frame-by-frame visual feature extraction cannot explicitly capture the moving trajectories of dynamic sound sources.

## Related Work & Insights

- **Diff-Foley / MMAudio**: Representative traditional V2A methods, serving as the non-spatial audio generation baselines for OmniAudio.
- **ViSAGe**: The closest competing framework, which is still constrained to perspective video inputs and runs 24 times slower during inference.
- **SpeechFlow**: A pioneer in self-supervised Flow Matching pre-training, which inspired the pre-training strategy of OmniAudio.
- **Stable Audio / Audiobox**: Sources of inspiration for the audio VAE and masked pre-training paradigms.
- **MetaCLIP-Huge**: Frozen visual encoder, eliminating visual-side training overhead.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | ⭐⭐⭐⭐⭐ | First to define the 360V2SA task, with a complete end-to-end framework design |
| Technical Depth | ⭐⭐⭐⭐ | High cohesion between Spatial VAE, dual-branch representations, and coarse-to-fine pre-training |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | Highly comprehensive with main experiments, three ablation sets, and subjective evaluations |
| Dataset Contribution | ⭐⭐⭐⭐⭐ | High value to the community with the 103K Sphere360 dataset and standardized benchmarks |
| Writing Quality | ⭐⭐⭐⭐ | Well-structured with rich illustrations and compelling motivations |
| Overall Rating | ⭐⭐⭐⭐⭐ | Novel task + comprehensive pipeline + valuable dataset = High-impact work |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Towards Streaming Synchronized Spatial Audio Generation via Autoregressive Diffusion Transformer](../../ICML2026/audio_speech/towards_streaming_synchronized_spatial_audio_generation_via_autoregressive_diffu.md)
- [\[CVPR 2026\] AudioStory: Generating Long-Form Narrative Audio with Large Language Models](../../CVPR2026/audio_speech/audiostory_generating_long-form_narrative_audio_with_large_language_models.md)
- [\[ICLR 2026\] OWL: Geometry-Aware Spatial Reasoning for Audio Large Language Models](../../ICLR2026/audio_speech/owl_geometry-aware_spatial_reasoning_for_audio_large_language_models.md)
- [\[ICLR 2026\] Query-Guided Spatial-Temporal-Frequency Interaction for Music Audio-Visual Question Answering](../../ICLR2026/audio_speech/query-guided_spatial-temporal-frequency_interaction_for_music_audio-visual_quest.md)
- [\[ICLR 2026\] AC-Foley: Reference-Audio-Guided Video-to-Audio Synthesis with Acoustic Transfer](../../ICLR2026/audio_speech/ac-foley_reference-audio-guided_video-to-audio_synthesis_with_acoustic_transfer.md)

</div>

<!-- RELATED:END -->
