---
title: >-
  [Paper Note] TCSinger 2: Customizable Multilingual Zero-shot Singing Voice Synthesis
description: >-
  [ACL2025][Audio & Speech][Singing Voice Synthesis] TCSinger 2 is proposed as a multi-task multilingual zero-shot singing voice synthesis model. Through a blurred boundary encoder, a contrastive learning audio encoder, and a Flow-based customized Transformer (incorporating Cus-MOE), it achieves style transfer and multi-level style control based on singing, speech, or text prompts.
tags:
  - "ACL2025"
  - "Audio & Speech"
  - "Singing Voice Synthesis"
  - "Zero-shot"
  - "Multilingual"
  - "Style Transfer"
  - "Style Control"
  - "Flow Matching"
  - "Mixture of Experts"
date: 2026-05-08
content_hash: 112c28eda7bd8fca
---

# TCSinger 2: Customizable Multilingual Zero-shot Singing Voice Synthesis

**Conference**: ACL2025  
**arXiv**: [2505.14910](https://arxiv.org/abs/2505.14910)  
**Code**: [AaronZ345/TCSinger2](https://github.com/AaronZ345/TCSinger2)  
**Area**: Audio & Speech  
**Keywords**: Singing Voice Synthesis, Zero-shot, Multilingual, Style Transfer, Style Control, Flow Matching, Mixture of Experts

## TL;DR

TCSinger 2 is proposed as a multi-task multilingual zero-shot singing voice synthesis model. Through a blurred boundary encoder, a contrastive learning audio encoder, and a Flow-based customized Transformer (incorporating Cus-MOE), it achieves style transfer and multi-level style control based on singing, speech, or text prompts.

## Background & Motivation

### Background

Customizable multilingual zero-shot singing voice synthesis (SVS) holds broad application prospects in music creation and short-video dubbing. Existing SVS models face two core challenges:

**Over-reliance on precise phoneme and note boundary annotations**: Datasets (such as OpenCpop) rely on MFA and manual alignment, resulting in significant boundary annotation errors; in zero-shot scenarios, transitions between phonemes and notes are particularly unnatural.

**Lack of effective multi-level style control**: Existing models cannot flexibly control singing style (including multiple levels such as timbre, singing method, emotion, and technique) through diverse prompts like natural language text, speech, or singing.

### Limitations of Prior Work

- TCSinger (prior work) only supports style control via labels or audio prompts, failing to handle natural language text prompts.
- Choi and Nam (2022) proposed a melody-unsupervised model to reduce reliance on boundary annotations, but this degraded synthesis quality and could not guarantee smooth transitions.
- Speech synthesis models like StyleTTS 2 and CosyVoice cannot model multi-level singing styles.

## Method

### Overall Architecture

TCSinger 2 consists of three core modules. The inputs are lyrics $l$, musical score $n$, and prompt $P$ (one of singing, speech, or text), and the output is the synthesized singing voice.

### 1. Blurred Boundary Content Encoder (BBC Encoder)

- **Design Motivation**: Existing models rely on precise phoneme/note boundaries, but annotation errors are common, especially in multilingual datasets.
- **Mechanism**: After separately encoding lyrics and musical scores, the duration is predicted to expand them into frame-level sequences. At each phoneme and note boundary, $m=8$ tokens are randomly masked to generate blurred boundaries.
- **Effect**: Forces the model to learn implicit alignment paths, improving transition naturalness and robustness in zero-shot generation, while augmenting the training data.

### 2. Custom Audio Encoder

- **VAE-based Singing/Speech Encoder**: Extracts style representations from singing prompts and speech prompts, respectively.
- **Text Encoder**: Fuses music scores and text prompts through cross-attention to obtain representations containing content and multi-level styles.
- **Contrastive Learning Alignment**: Three contrastive types are designed: (1) same content with different styles, (2) similar style with different content, and (3) different styles and content. Triplet pairs across the three modalities are aligned using the InfoNCE objective function.
- **Reconstruction Training**: Uses $L_2$ loss and LSGAN adversarial loss to train the audio decoder, ensuring that the singing representations do not lose integrity.

### 3. Flow-based Custom Transformer

- **Flow Matching Generation**: Concatenates Gaussian noise with content embeddings and prompt embeddings, learning content and style via Transformer self-attention. It takes 1000 steps for training and only 25 steps for inference (using an Euler ODE solver).
- **Cus-MOE (Customized Mixture of Experts)**:
    - **Lingual-MOE**: Selects experts based on the lyric language, where each expert focuses on a specific language family (e.g., Romance/Latin languages), improving multilingual generation quality.
    - **Stylistic-MOE**: Selects experts based on audio or text prompts to match fine-grained styles (e.g., "mezzo-soprano + cheerful pop falsetto").
    - The routing strategy employs dense-to-sparse Gumbel-Softmax with a load-balancing loss.
- **$F_0$ Supervision**: Predicts $F_0$ using the output of the first Transformer block to provide pitch supervision for subsequent blocks.
- **CFG Strategy**: Randomly drops prompts with a probability of 0.2 during training, and uses classifier-free guidance with $\gamma=3$ during inference to improve generation quality and style controllability.

### Loss & Training

- Audio Encoder stage: contrastive loss + $L_2$ reconstruction loss + LSGAN adversarial loss
- TCSinger 2 main model: duration loss + pitch loss + load-balancing loss + Flow Matching loss

### Supported Inference Tasks

- Zero-shot style transfer (intra-lingual / cross-lingual)
- Multi-level text style control (timbre, singing method, emotion, technique)
- Speech-to-singing (STS) style transfer

## Key Experimental Results

### Dataset

- Self-collected 50 hours of clean singing voice + multiple open-source datasets (Opencpop, M4Singer, OpenSinger, PopBuTFy, GTSinger), covering 9 languages (Chinese, English, French, Spanish, German, Italian, Japanese, Korean, Russian).
- Part of the data is manually annotated with multi-level style labels; 30 unseen singers are reserved as the test set.
- Model configuration: 4 Transformer blocks, hidden size 768, 8 attention heads, 4 experts per group.
- Training hardware: 8x NVIDIA RTX-4090

### Table 1: Zero-shot Style Transfer (Parallel / Cross-Lingual)

| Method | MOS-Q ↑ | MOS-S ↑ | FFE ↓ | Cos ↑ | MOS-Q (Cross-Lingual) ↑ | MOS-S (Cross-Lingual) ↑ |
|------|---------|---------|-------|-------|------------------|------------------|
| GT | 4.58 | - | - | - | - | - |
| GT (vocoder) | 4.36 | 4.41 | 0.04 | 0.95 | - | - |
| StyleTTS 2 | 3.71 | 3.79 | 0.42 | 0.71 | 3.58 | 3.63 |
| CosyVoice | 3.74 | 3.93 | 0.33 | 0.87 | 3.63 | 3.77 |
| VISinger 2 | 3.79 | 3.88 | 0.31 | 0.83 | 3.69 | 3.72 |
| TCSinger | 3.94 | 4.01 | 0.26 | 0.91 | 3.77 | 3.87 |
| **TCSinger 2** | **4.13** | **4.27** | **0.21** | **0.93** | **3.96** | **4.09** |

### Table 2: Multi-level Style Control with Text Prompts

| Method | MOS-Q ↑ | MOS-C ↑ | FFE ↓ | MOS-Q (Non-parallel) ↑ | MOS-C (Non-parallel) ↑ |
|------|---------|---------|-------|------------------|------------------|
| GT | 4.56 | - | - | - | - |
| GT (vocoder) | 4.26 | 4.32 | 0.06 | - | - |
| StyleTTS 2 | 3.61 | 3.67 | 0.43 | 3.51 | 3.59 |
| CosyVoice | 3.72 | 3.73 | 0.37 | 3.60 | 3.67 |
| VISinger 2 | 3.81 | 3.81 | 0.30 | 3.69 | 3.75 |
| TCSinger | 3.99 | 3.97 | 0.27 | 3.90 | 3.93 |
| **TCSinger 2** | **4.07** | **4.19** | **0.22** | **3.98** | **4.11** |

### Table 3: Speech-to-Singing (STS) Style Transfer

| Method | FFE ↓ | Cos ↑ | MOS-Q ↑ | MOS-S ↑ |
|------|-------|-------|---------|---------|
| GT (vocoder) | 0.06 | 0.93 | 4.21 | 4.20 |
| StyleTTS 2 | 0.41 | 0.71 | 3.60 | 3.52 |
| CosyVoice | 0.39 | 0.79 | 3.66 | 3.65 |
| VISinger 2 | 0.32 | 0.75 | 3.72 | 3.59 |
| TCSinger | 0.28 | 0.82 | 3.89 | 3.84 |
| **TCSinger 2** | **0.24** | **0.89** | **3.97** | **3.96** |

### Table 4: Ablation Study (CMOS Changes)

| Setting | CMOS-Q (Transfer) | CMOS-S (Transfer) | CMOS-Q (Control) | CMOS-C (Control) |
|------|-------------|-------------|-------------|-------------|
| TCSinger 2 (Full) | 0.00 | 0.00 | 0.00 | 0.00 |
| w/o BBC Encoder | -0.36 | -0.23 | -0.39 | -0.26 |
| w/o Custom Audio Encoder | -0.21 | -0.37 | -0.19 | -0.41 |
| w/o $F_0$ Supervision | -0.33 | -0.24 | -0.31 | -0.27 |
| w/o CFG | -0.26 | -0.22 | -0.25 | -0.31 |
| w/o Cus-MOE | -0.31 | -0.32 | -0.38 | -0.35 |
| w/o Lingual-MOE | -0.29 | -0.17 | -0.32 | -0.21 |
| w/o Stylistic-MOE | -0.21 | -0.26 | -0.23 | -0.33 |

**Key Ablation Findings**: BBC Encoder has the greatest impact on synthesis quality (CMOS-Q -0.36/-0.39); Custom Audio Encoder has the greatest impact on style control (CMOS-C -0.41); Cus-MOE has an overall comprehensive and significant impact.

## Highlights & Insights

1. **Novel and Practical Blurred Boundary Strategy**: By randomly masking boundary tokens instead of pursuing precise alignment, it addresses both label-error sensitivity and unnatural transitions while also augmenting the training data.
2. **Trimodal Contrastive Learning for a Unified Style Space**: Singing, speech, and text prompts are aligned into the same representation space, enabling the model to support flexible multimodal inputs and multi-task inference.
3. **Exquisite Cus-MOE Design**: Separately routes linguistic conditions and stylistic conditions to different expert groups, achieving fine-grained decoupled control over quality and style.
4. **Comprehensive Evaluation across Multi-Task and Multilingual Scenarios**: Covering 9 languages and 4 tasks, it comprehensively outperforms baselines in all tasks.

## Limitations & Future Work

1. **Reliance on Manually Annotated Style Labels**: Multi-level styles (emotion, singing method, technique, etc.) still require manual annotation, which is costly and prone to error. The authors plan to utilize automatic annotation tools in the future.
2. **Insufficient Inference Speed**: Although Flow Matching is faster than diffusion models (25 inference steps), it still does not meet the requirements of industrial-grade real-time streaming generation.
3. **Limited Dataset Scale**: The total training data is approximately 268 hours, which is still insufficient for zero-shot scenarios covering 9 languages, potentially limiting the generalization capability.

## Related Work & Insights

- **Singing Voice Synthesis**: VISinger 2 (high-fidelity SVS), SiFiSinger (pitch control), TCSinger (prior work, style transfer + label control)
- **Style Modeling**: StyleTTS 2 (prosody prediction), CosyVoice (x-vector + LLM decoupled style), PromptSinger (identity control via text description)
- **Zero-shot Speech Synthesis**: Attentron (style extraction via attention mechanism), ZSM-SS (wav2vec 2.0 external encoder), MegaTTS 3 (sparse alignment diffusion)
- **MoE Related**: Switch Transformer (load-balancing loss), Gumbel-Softmax (differentiable routing)

## Rating

- Novelty: ⭐⭐⭐⭐ — The design of BBC Encoder and Cus-MOE is original in the SVS field, and unifying the style space through trimodal contrastive learning is a meaningful exploration.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation covering 4 tasks, 9 languages, subjective/objective metrics, and complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, detailed methodology description, and rich tables/figures.
- Value: ⭐⭐⭐⭐ — The first multilingual zero-shot SVS system to support singing, speech, and text prompts, demonstrating high practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Adapting Speech Language Model to Singing Voice Synthesis](../../NeurIPS2025/audio_speech/adapting_speech_language_model_to_singing_voice_synthesis.md)
- [\[AAAI 2026\] HQ-SVC: Towards High-Quality Zero-Shot Singing Voice Conversion in Low-Resource Scenarios](../../AAAI2026/audio_speech/hq-svc_towards_high-quality_zero-shot_singing_voice_conversion_in_low-resource_s.md)
- [\[ACL 2025\] ControlSpeech: Towards Simultaneous and Independent Zero-shot Speaker Cloning and Zero-shot Language Style Control](controlspeech_zero_shot.md)
- [\[ACL 2025\] Zero-Shot Text-to-Speech for Vietnamese](zero-shot_text-to-speech_for_vietnamese.md)
- [\[ACL 2026\] ReStyle-TTS: Relative and Continuous Style Control for Zero-Shot Speech Synthesis](../../ACL2026/audio_speech/restyle-tts_relative_and_continuous_style_control_for_zero-shot_speech_synthesis.md)

</div>

<!-- RELATED:END -->
