---
title: >-
  [Paper Note] MultiFoley: Video-Guided Foley Sound Generation with Multimodal Controls
description: >-
  [CVPR 2025][Audio & Speech][Foley Sound Generation] The paper presents MultiFoley, a video-guided Foley sound generation system based on Diffusion Transformers. It supports textual semantic controls and reference audio style controls. By jointly training on video-audio and text-audio datasets, it achieves 48kHz high-quality audio generation, outperforming existing methods with a 90% win rate in human evaluations.
tags:
  - "CVPR 2025"
  - "Audio & Speech"
  - "Foley Sound Generation"
  - "Video-to-Audio"
  - "Diffusion Model"
  - "Multimodal Control"
  - "Audio-Visual Synchronization"
date: 2026-05-08
content_hash: f3ed3d94b822415a
---

# MultiFoley: Video-Guided Foley Sound Generation with Multimodal Controls

**Conference**: CVPR 2025  
**arXiv**: [2411.17698](https://arxiv.org/abs/2411.17698)  
**Code**: [https://ificl.github.io/MultiFoley/](https://ificl.github.io/MultiFoley/)  
**Area**: Audio and Speech / Generative Models  
**Keywords**: Foley Sound Generation, Video-to-Audio, Diffusion Model, Multimodal Control, Audio-Visual Synchronization

## TL;DR

The paper presents MultiFoley, a video-guided Foley sound generation system based on Diffusion Transformers. It supports textual semantic controls and reference audio style controls. By jointly training on video-audio and text-audio datasets, it achieves 48kHz high-quality audio generation, outperforming existing methods with a 90% win rate in human evaluations.

## Background & Motivation

**Background**: Video-to-Audio (V2A) generation aims to synthesize audio automatically for silent videos. Most existing methods (e.g., FoleyCrafter, Frieren, VAB) operate at a low sampling rate of 16kHz and accept only visual input—lacking the ability to specify the desired audio semantics via text or style via reference audio.

**Limitations of Prior Work**: Three key issues exist: (1) Low audio quality—the 16kHz sampling rate loses substantial high-frequency information, whereas professional Foley workflows require 48kHz; (2) Limited controllability—users cannot specify nuances such as "footsteps on wooden floor" vs "footsteps on gravel"; (3) Data mismatch—high-quality sound libraries (e.g., HQ-SFX) only contain text labels without video, while video datasets (e.g., VGGSound) often contain poor audio quality.

**Key Challenge**: Achieving visual synchronization, semantic controllability, and high audio quality simultaneously, which require different types of data and distinct model capabilities.

**Goal**: Unify three generation modes—video guidance, textual control, and reference audio transfer—to achieve high-quality, controllable Foley sound generation at 48kHz.

**Key Insight**: Joint training—mixing VGGSound (video-audio pairs) and HQ-SFX (text-audio pairs) during training, distinguishing audio quality distributions with quality tags, and achieving flexible multimodal inference through random condition dropout.

**Core Idea**: DiT + Joint training on mixed data + Random multi-conditional dropout = Foley generation that is visually synchronized, textually controllable, and high-fidelity.

## Method

### Overall Architecture

The input can include: (1) a silent query video $v_q$ (encoded with CAVP, 8FPS, 64 frames); (2) a text prompt $t_c$ (encoded with T5-base); and (3) a reference audio-video pair $(a_c, v_c)$. The audio is encoded as a 40Hz latent variable (dimension 64) using DAC-VAE. A 12-layer DiT (332M parameters) performs diffusion denoising in the latent space, where conditions are injected via concatenation (video) and cross-attention (text).

### Key Designs

1. **Mixed Data Joint Training + Quality Tags**:

    - **Function**: To simultaneously learn temporal alignment between video and audio, and semantic correspondence between text and audio.
    - **Mechanism**: VGGSound (168K video-audio samples) is labeled as "low quality," while HQ-SFX (400K text-audio pairs) is labeled as "high quality." They are sampled with a 60:40 mixture ratio during training. During inference, the "high quality" tag is used to guide generation toward the high-quality audio distribution.
    - **Design Motivation**: Resolves the data conflict where VGGSound provides visual synchronization supervision but has poor audio quality, whereas HQ-SFX provides high-quality audio but lacks video. Quality tags allow the model to distinguish and separate these two distributions.

2. **Multi-Conditional Random Dropout**:

    - **Function**: To enable the model to flexibly handle any combination of conditioning signals.
    - **Mechanism**: During training, text, video, and audio conditions are randomly dropped with a probability of 0.25. Additionally, sections of 0-2 seconds of the audio latents are randomly masked. This allows the model to accept any subset of conditions during inference, such as video-only, video+text, or video+reference audio.
    - **Design Motivation**: Prevents the model from over-relying on any single condition. Ablation results show that removing text causes a sharp drop in semantic metrics (CLAP 34.4% → 19.4%), while synchronization is maintained (0.80s → 0.77s), demonstrating successful condition decoupling.

3. **Classifier-Free Guidance (CFG) with Negative Prompts**:

    - **Function**: To precisely control the generated semantics using positive and negative text prompts.
    - **Mechanism**: During inference, both positive prompts (desired sounds) and negative prompts (undesired sounds) are input. The CFG formulation is $\hat\epsilon = (\gamma+1) \cdot \epsilon_\theta(z_t, v_q, t_{\text{pos}}) - \gamma \cdot \epsilon_\theta(z_t, \varnothing_v, t_{\text{neg}})$, where the negative prompt replaces simple unconditional noise estimation.
    - **Design Motivation**: In text-controlled Foley generation, excluding unwanted sound categories (e.g., "no background music") is more effective than only specifying positive requirements.

### Loss & Training

The standard LDM denoising loss is formulated as $\mathcal{L}_{\text{LDM}} = \mathbb{E}_{\epsilon, t}[\|\epsilon - \hat\epsilon\|_2^2]$. Loss is not computed on masked conditional regions. Inference uses DDIM with 100 sampling steps and a CFG scale of 3.0. The large model is first trained on the mixed dataset, and then fine-tuned on a high audio-visual alignment subset of VGGSound.

## Key Experimental Results

### Main Results

Comparison of V2A generation on the VGGSound test set (8702 videos):

| Method | ImageBind↑ | CLAP↑ | AV-Sync↓ | Sampling Rate |
|------|-----------|-------|---------|--------|
| FoleyCrafter | 30.2% | 25.3% | 0.87s | 16kHz |
| Frieren | 26.1% | 27.6% | 0.87s | 16kHz |
| **Ours** | 28.0% | **34.4%** | **0.80s** | **48kHz** |
| Oracle (DAC-VAE) | 35.4% | 28.2% | 0.62s | 48kHz |

Human evaluation (400 cases vs FoleyCrafter):

| Dimension | MultiFoley Win Rate |
|------|----------------|
| Semantic Alignment | **85.8%** |
| Synchronization | **94.5%** |
| Audio Quality | **86.5%** |
| Overall | **90.2%** |

### Ablation Study

| Configuration | ImageBind | CLAP | AV-Sync |
|------|-----------|------|---------|
| Full Model (VT2A) | 28.0% | 34.4% | 0.80s |
| No Text (V2A) | 22.4% | 19.4% | 0.77s |
| No Subset Fine-Tuning | 27.3% | 33.8% | 0.81s |
| Low quality tag | - | 34.4% | - |
| High quality tag | - | 34.9% | - |

### Key Findings
- **Text condition is crucial for semantics**: Removing text drops CLAP from 34.4% to 19.4% (-15%), while AV-Sync remains almost unchanged (0.77s vs 0.80s). This proves that visual conditions guide temporal synchronization, whereas text conditions guide semantic content.
- **Quality tags effectively separate distributions**: The High Quality tag improves FAD@AUD (closer to professional audio distribution) but decreases FAD@VGG (deviating from web video distribution), showing that quality control is indeed effective.
- **CFG scale of 3.0 is optimal**: Scales that are too low (1.0) yield unclear semantics, while scales that are too high (7.0) trigger overfitting artifacts.
- **Condition overlay is effective in audio-visual extension tasks**: Under full conditions (video + audio + video conditions + text), CLAP reaches 64.3% compared to 55.4% for video-only.

## Highlights & Insights

- **An elegant solution for mixed-data training**: Quality tags provide a simple yet elegant solution to resolve the conflict between "good data lacking videos" and "video data lacking high-quality sound." Switching tags during inference controls the audio style/quality profile without requiring complex data filtering or multi-stage training pipelines.
- **48kHz full-bandwidth generation**: Three times the bandwidth of 16kHz baselines, providing practical value for professional Foley production. The efficient compression of DAC-VAE (48kHz to 40Hz latents) serves as a key enabling technology.
- **Natural implementation of condition decoupling**: Random dropout of different conditioning signals enables the model to automatically learn the independent role of each condition—video for synchronization, text for semantics, and reference audio for style—without requiring explicit decoupled architectures.

## Limitations & Future Work

- **ImageBind score is lower than some baselines and the Oracle**: 28.0% vs FoleyCrafter's 30.2% and Oracle's 35.4%, indicating room for improvement in cross-modal semantic alignment.
- **Difficulty handling multiple events**: When multiple sound events overlap in a scene, the model struggles to differentiate the temporal relationships between them.
- **Limited scale of training data**: Operating on VGGSound (168K) + HQ-SFX (400K) only; scaling to larger high-quality datasets could significantly boost performance.
- **Inference overhead**: The computational cost of 100-step DDIM diffusion sampling combined with 48kHz audio decoding is relatively heavy.
- **Potential abuse risks**: Extremely realistic sound generation for arbitrary videos poses deepfake and misuse risks.

## Related Work & Insights

- **vs FoleyCrafter**: FoleyCrafter scores higher on ImageBind but lower on CLAP, indicating solid video alignment but weaker semantic comprehension. MultiFoley substantially enhances semantic control via text conditioning.
- **vs Frieren**: Frieren achieves relatively good FAD but suffers from poor synchronization. MultiFoley ensures better temporal alignment by directly concatenating video features into the DiT input.
- **vs Make-An-Audio**: Pure text-to-audio methods cannot guarantee video synchronization. MultiFoley unifies both text and video controls.

## Rating
- Novelty: ⭐⭐⭐⭐ The unified framework for multimodal control and the quality tagging mechanism are key innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation involving automatic metrics, human evaluation, extensive ablations, and extension tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear methodological descriptions, though some mathematical notations could be further simplified.
- Value: ⭐⭐⭐⭐ Immediate practical utility for professional Foley and video post-production.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AC-Foley: Reference-Audio-Guided Video-to-Audio Synthesis with Acoustic Transfer](../../ICLR2026/audio_speech/ac-foley_reference-audio-guided_video-to-audio_synthesis_with_acoustic_transfer.md)
- [\[NeurIPS 2025\] MGAudio: Model-Guided Dual-Role Alignment for High-Fidelity Open-Domain Video-to-Audio Generation](../../NeurIPS2025/audio_speech/model-guided_dual-role_alignment_for_high-fidelity_open-domain_video-to-audio_ge.md)
- [\[CVPR 2025\] Synchronized Video-to-Audio Generation via Mel Quantization-Continuum Decomposition](synchronized_video-to-audio_generation_via_mel_quantization-continuum_decomposit.md)
- [\[CVPR 2025\] VinTAGe: Joint Video and Text Conditioning for Holistic Audio Generation](vintage_joint_video_and_text_conditioning_for_holistic_audio_generation.md)
- [\[NeurIPS 2025\] Node-Based Editing for Multimodal Generation of Text, Audio, Image, and Video](../../NeurIPS2025/audio_speech/node-based_editing_for_multimodal_generation_of_text_audio_image_and_video.md)

</div>

<!-- RELATED:END -->
