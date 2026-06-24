---
title: >-
  [Paper Note] HQ-SVC: Towards High-Quality Zero-Shot Singing Voice Conversion in Low-Resource Scenarios
description: >-
  [AAAI 2026][Audio & Speech][Singing Voice Conversion] This work proposes the HQ-SVC framework, which jointly extracts content and speaker features based on a decoupled audio codec (FACodec). Combined with an Enhanced Voice Adapter (EVA) to fuse acoustic features such as pitch and energy, progressive optimization is performed using DDSP and a diffusion model. Using a single RTX 3090 and less than 80 hours of singing voice data, the framework achieves zero-shot singing voice co…
tags:
  - "AAAI 2026"
  - "Audio & Speech"
  - "Singing Voice Conversion"
  - "Zero-shot"
  - "Low-resource"
  - "Diffusion Model"
  - "DDSP"
  - "Audio Codecs"
date: 2026-05-08
content_hash: b14bc3c01bf45b3c
---

# HQ-SVC: Towards High-Quality Zero-Shot Singing Voice Conversion in Low-Resource Scenarios

**Conference**: AAAI 2026  
**arXiv**: [2511.08496](https://arxiv.org/abs/2511.08496)  
**Code**: [ShawnPi233/HQ-SVC](https://github.com/ShawnPi233/HQ-SVC)  
**Area**: Image Restoration  
**Keywords**: Singing Voice Conversion, Zero-shot, Low-resource, Diffusion Model, DDSP, Audio Codecs  

## TL;DR

This work proposes the HQ-SVC framework, which jointly extracts content and speaker features based on a decoupled audio codec (FACodec). Combined with an Enhanced Voice Adapter (EVA) to fuse acoustic features such as pitch and energy, progressive optimization is performed using DDSP and a diffusion model. Using a single RTX 3090 and less than 80 hours of singing voice data, the framework achieves zero-shot singing voice conversion quality that outperforms large-scale training baselines, while additionally supporting speech super-resolution tasks.

## Background & Motivation

### Task Definition

The goal of Zero-Shot Singing Voice Conversion (Zero-shot SVC) is to convert the source singer's timbre to an unseen target speaker's timbre while preserving the original melody and lyric content, without requiring fine-tuning on the target speaker. This has broad applications in music production and virtual singer domains.

### Limitations of Prior Work

**Poor Generalization**: Traditional SVC methods (e.g., DiffSVC, SoVITS-SVC) rely on explicit speaker IDs and independent content encoders (HuBERT, ContentVec), making them unable to generalize to unseen speakers.

**High Resource Overhead**: High-quality zero-shot methods such as LDM-SVC require two-stage adversarial training. SaMoye-SVC relies on 1700 hours of large-scale singing data and 7 days of training on A100 GPUs, resulting in extremely high resource consumption.

**Low Synthesis Quality**: Although lightweight schemes like FastSVC can achieve zero-shot conversion, there is a significant gap in generation quality.

**Information Loss**: Modeling speaker and content features separately leads to the loss of acoustic information, making feature fusion and natural speech reconstruction difficult.

### Key Insight

Speech and singing both belong to human vocals, and large-scale speech data can provide effective priors for singing tasks. FACodec in NaturalSpeech3 can simultaneously decouple content and speaker features within a unified framework, reducing information loss compared to separate modeling. However, FACodec itself cannot completely capture the complex acoustic variations required for high-quality synthesis, necessitating additional acoustic feature enhancement and progressive synthesis optimization.

## Method

### Overall Architecture

The pipeline of HQ-SVC is divided into four stages:

1. **Decoupled Feature Extraction**: The encoder and decoder of the pre-trained FACodec are frozen, and content features $x_{\text{con}}$ (256-dimensional) and speaker features $x_{\text{spk}}$ are extracted from the intermediate layers.
2. **EVA Multi-Feature Fusion**: Acoustic features such as F0 pitch (extracted by RMVPE), energy, and phase are introduced and deeply fused with content/speaker features.
3. **Progressive Synthesis**: A preliminary waveform is first generated via DDSP and converted to a Mel-spectrogram, which is then refined and optimized by a diffusion model.
4. **Vocoder Generation**: NSF-HiFiGAN is utilized to convert the Mel-spectrogram + F0 into the final 44.1kHz audio.

### Decoupled Codec

FACodec is adopted as the unified decoupler, with both its encoder and decoder completely frozen. Internally, FACodec utilizes three groups of Factorized Vector Quantization (FVQ):

- Content quantizer $Q^c$: 2 quantizers
- Pitch quantizer $Q^p$: 1 quantizer
- Detail quantizer $Q^d$: 3 quantizers
- All codebook sizes are 1024

Freezing the pre-trained model avoids retraining, which significantly reduces the demand for training resources.

### Enhanced Voice Adapter (EVA)

This is the core contribution module of this work, addressing the issue that relying solely on content and speaker features makes it difficult to capture the rich melody and energy dynamics in singing voices:

**Feature Extraction and Mapping**:

- Speaker feature $x_{\text{spk}}$ is mapped to speaker embedding $e_{\text{spk}}$ via an MLP, while the residual part serves as the style feature $e_{\text{sty}}$.
- F0 is logarithmically transformed as $x_{f_0} = \ln(f_0/700 + 1)$ and then processed by an MLP to obtain the pitch embedding $e_{f_0}$.
- Amplitude (volume) and phase are processed by MLPs to obtain $e_{\text{vol}}$ and $e_{\text{pha}}$, respectively.
- All MLPs consist of two linear layers with SiLU activations, with an output dimension of 256.

**Feature Fusion Strategy**:

Considering the strong coupling between pitch and timbre, the speaker and F0 embeddings are added together and then concatenated with the remaining features to form a 1024-dimensional style embedding:

$$e_s = \text{Concat}(e_{\text{spk}} + e_{f_0},\ e_{\text{sty}},\ e_{\text{vol}},\ e_{\text{pha}})$$

After being compressed to 256 dimensions via a 1D convolution, it is fused with the content features using the FiLM mechanism:

$$\text{FiLM}(e_c, e_s) = f_\alpha(e_s) \cdot e_c + f_\beta(e_s)$$

The FiLM output is then processed by an 8-head self-attention Conformer and LayerNorm to generate the final embedding $e$ for subsequent synthesis modules.

**Speaker Contrastive Loss $\mathcal{L}_{\text{spk}}$**:

Based on the InfoNCE loss, it pulls the embeddings of the same speaker closer and pushes those of different speakers apart within a batch, enhancing zero-shot generalization capabilities. The temperature parameter is set to $\tau = 0.1$.

**Speaker-F0 Predictor (SFP)**:

To address the issue where sufficient pitch statistics of the target speaker cannot be obtained in zero-shot scenarios, an MLP module is designed to predict the mean and variance of F0 from the speaker embedding, supervised by the L1 loss:

$$\mathcal{L}_{f_0} = \mathbb{E}[\|\mu_{x_{f_0}} - \hat{\mu}_{x_{f_0}}\|_1 + \|\sigma^2_{x_{f_0}} - \hat{\sigma}^2_{x_{f_0}}\|_1]$$

### Progressive Singing Voice Reconstruction

**DDSP Synthesis**: Utilizes harmonic and noise synthesizers to generate periodic and aperiodic components respectively, introducing strong inductive bias to improve audio fidelity. The DDSP output is converted to a Mel-spectrogram, followed by calculating the MSE loss $\mathcal{L}_{\text{ddsp}}$.

**Diffusion Model Refinement**: Employs WaveNet as the denoiser (128-dimensional input, 20 residual blocks, 512 convolutional channels, 256-dimensional hidden layers) to further fill in the acoustic details missed by DDSP. During the inference phase, DPM-Solver++ is used with 100 diffusion steps and $10\times$ acceleration.

**Total Loss**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ddsp}} + \mathcal{L}_{\text{diff}} + \mathcal{L}_{\text{spk}} + \mathcal{L}_{f_0}$

## Key Experimental Results

### Table 1: Main Results for Zero-Shot Singing Voice Conversion

| Method | Training Config | Data Volume | STOI↑ | SECS↑ | F0 RMSE↓ | FPC↑ | NISQA↑ | NMOS↑ | SMOS↑ |
|------|---------|--------|-------|-------|----------|------|--------|-------|-------|
| FACodec-SVC | RTX 3090 (1h) | <80h | 0.533 | 0.074 | 77.798 | 0.601 | 1.791 | 2.391 | 2.740 |
| SaMoye-SVC | A100 (7 days) | 1700h | 0.724 | **0.647** | 17.418 | 0.617 | 3.528 | 3.958 | 3.569 |
| **HQ-SVC** | RTX 3090 (11h) | <80h | **0.799** | 0.627 | **8.681** | **0.891** | **3.841** | **4.215** | 3.578 |

HQ-SVC comprehensively outperforms SaMoye-SVC (trained for 7 days on over 1700 hours of data) in terms of STOI (+10%), F0 RMSE (reduced by 50%), FPC (+44%), NISQA (+9%), and subjective NMOS (+6.5%).

### Table 2: Comparison of Speech Super-Resolution

| Method | Data Volume | LSD↓ | NISQA↑ | NMOS↑ | SMOS↑ |
|------|--------|------|--------|-------|-------|
| AudioSR | 7000h | 2.087 | 4.094 | 4.188 | 4.235 |
| **HQ-SVC** | <80h | **1.842** | **4.193** | **4.332** | **4.479** |

Even when compared with the specialized audio super-resolution model AudioSR (trained on 7000 hours of data), HQ-SVC achieves superior results across LSD, NISQA, NMOS, and SMOS, showcasing outstanding cross-task generalization capabilities.

## Highlights & Insights

1. **Extreme Resource Efficiency**: With a single RTX 3090, <6GB of VRAM, 11 hours of training, and <80h of data, it outperforms SaMoye-SVC (which requires A100 $\times$ 7 days and 1700h of data), lowering the barrier for zero-shot SVC.
2. **Unified Decoupling Outperforms Separate Encoding**: Utilizing FACodec to simultaneously decouple content and speaker features within a unified framework performs better in tempo control and information fusion compared to the HQ-SVC-SE variant using CAM++ + ContentVec.
3. **Progressive Synthesis Pipeline**: DDSP provides preliminary synthesis with strong inductive biases, while the diffusion model fills in details. Ablation studies prove that removing either module significantly decreases quality (removing diffusion drops NISQA from 3.841 to 3.175).
4. **Two-in-One Super-Resolution**: Since the model takes 16kHz features as input and optimizes 44.1kHz Mel-spectrograms during training, it naturally supports speech super-resolution, with subjective quality surpassing specialized methods.
5. **Ingenious Speaker-F0 Predictor Design**: Predicting F0 statistics from speaker embeddings compensates for the deficiency of being unable to obtain target pitch distributions in zero-shot scenarios.

## Limitations & Future Work

1. **Room for Improvement in Timbre Similarity**: The SECS metric of HQ-SVC is slightly lower than that of SaMoye-SVC (0.627 vs. 0.647). The unified encoder is less effective in speaker identity separation than the independent encoder variant, HQ-SVC-SE (SECS of 0.668).
2. **Side Effects of Pitch and Timbre Coupling**: Incorporating $\mathcal{L}_{\text{spk}}$ and $\mathcal{L}_{f_0}$ results in a slight drop in SMOS, indicating that enhanced pitch consistency may conversely hinder human perception of timbre shifts when there is a significant range discrepancy between target singers.
3. **Intelligibility Loss**: In speech super-resolution tasks, STOI (0.841 vs. 0.986) and FPC (0.868 vs. 0.998) are significantly lower than those of AudioSR, showing that feature decoupling-based reconstruction introduces minor pronunciation and pitch errors.
4. **Only Chinese Singing Verified**: The training set comprises Opensinger and M4Singer Chinese singing voice datasets, leaving cross-lingual generalization capability insufficiently validated.
5. **Style Transfer Unaddressed**: Currently, only timbre conversion is performed. Singing style conversion (e.g., singing techniques, emotional expressions) is left for future work.

## Related Work & Insights

- **Traditional SVC**: GMM methods require parallel data, and GAN methods (e.g., FastSVC) suffer from unstable training and limited quality.
- **Low-Resource SVC**: DDSP-SVC and CoMoSVC leverage DDSP/diffusion models to achieve high-quality low-resource conversion, but they rely on explicit speaker IDs.
- **Zero-shot SVC**: LDM-SVC achieves high-quality zero-shot conversion based on VITS but is computationally expensive; SaMoye-SVC requires large-scale data and adversarial training.
- **Audio Codec**: FACodec from NaturalSpeech3 achieves content/speaker decoupling, on top of which HQ-SVC enhances acoustic detail modeling.
- **Speech Super-Resolution**: AudioSR is trained on 7000h of data based on LDM, focusing on preserving low-frequency details; HQ-SVC outperforms it in terms of naturalness under a unified framework.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The design of the EVA module and the progressive DDSP+diffusion pipeline are innovative, and the Speaker-F0 Predictor is simple and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Main experiments, super-resolution, sampler comparison, and ablation studies provide comprehensive subjective and objective evaluations; however, baselines are limited (with SaMoye-SVC being the only strong baseline).
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, complete equations, and in-depth ablation analysis.
- **Value**: ⭐⭐⭐⭐ — Significantly lowers the barrier for zero-shot SVC, carrying substantial practical value for low-resource scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TCSinger 2: Customizable Multilingual Zero-shot Singing Voice Synthesis](../../ACL2025/audio_speech/tcsinger_2_customizable_multilingual_zero-shot_singing_voice_synthesis.md)
- [\[ACL 2026\] Multimodal In-Context Learning for ASR of Low-Resource Languages](../../ACL2026/audio_speech/multimodal_in-context_learning_for_asr_of_low-resource_languages.md)
- [\[ACL 2026\] Hard to Be Heard: Phoneme-Level ASR Analysis of Phonologically Complex, Low-Resource Endangered Languages](../../ACL2026/audio_speech/hard_to_be_heard_phoneme-level_asr_analysis_of_phonologically_complex_low-resour.md)
- [\[ICLR 2026\] TVTSyn: Content-Synchronized Time-Varying Timbre for Streaming Voice Conversion and Anonymization](../../ICLR2026/audio_speech/tvtsyn_content-synchronous_time-varying_timbre_for_streaming_voice_conversion_an.md)
- [\[ICML 2026\] VocSim: A Training-Free Benchmark for Zero-Shot Content Identity Recognition for Single-Source Audio](../../ICML2026/audio_speech/vocsim_a_training-free_benchmark_for_zero-shot_content_identity_in_single-source.md)

</div>

<!-- RELATED:END -->
