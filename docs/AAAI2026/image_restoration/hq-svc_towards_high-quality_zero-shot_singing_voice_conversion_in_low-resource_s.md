---
title: >-
  [Paper Note] HQ-SVC: Towards High-Quality Zero-Shot Singing Voice Conversion in Low-Resource Scenarios
description: >-
  [AAAI 2026][Image Restoration][Singing Voice Conversion] This paper proposes HQ-SVC, a framework that leverages a disentangled audio codec (FACodec) to jointly extract content and speaker features, integrates an Enhanced Voice Adaptor (EVA) to fuse acoustic features such as pitch and energy, and employs a progressive synthesis pipeline combining DDSP and a diffusion model. Trained on a single RTX 3090 with fewer than 80 hours of singing data, HQ-SVC achieves zero-shot singing voice conversion quality surpassing large-scale training baselines, and additionally supports speech super-resolution.
tags:
  - AAAI 2026
  - Image Restoration
  - Singing Voice Conversion
  - Zero-Shot
  - Low-Resource
  - Diffusion Model
  - DDSP
  - Audio Codec
date: 2026-05-08
content_hash: 778e84b117903c92
---

# HQ-SVC: Towards High-Quality Zero-Shot Singing Voice Conversion in Low-Resource Scenarios

**Conference**: AAAI 2026
**arXiv**: [2511.08496](https://arxiv.org/abs/2511.08496)
**Code**: [ShawnPi233/HQ-SVC](https://github.com/ShawnPi233/HQ-SVC)
**Area**: Image Restoration
**Keywords**: Singing Voice Conversion, Zero-Shot, Low-Resource, Diffusion Model, DDSP, Audio Codec

## TL;DR

This paper proposes HQ-SVC, a framework that leverages a disentangled audio codec (FACodec) to jointly extract content and speaker features, integrates an Enhanced Voice Adaptor (EVA) to fuse acoustic features such as pitch and energy, and employs a progressive synthesis pipeline combining DDSP and a diffusion model. Trained on a single RTX 3090 with fewer than 80 hours of singing data, HQ-SVC achieves zero-shot singing voice conversion quality surpassing large-scale training baselines, and additionally supports speech super-resolution.

## Background & Motivation

### Task Definition

Zero-shot Singing Voice Conversion (SVC) aims to transfer the timbre of a source singer to that of an unseen target speaker while preserving the original melody and lyric content, without requiring fine-tuning on the target speaker. This has broad applications in music production and virtual singer systems.

### Limitations of Prior Work

**Poor generalization**: Traditional SVC methods (DiffSVC, SoVITS-SVC, etc.) rely on explicit speaker IDs and separate content encoders (HuBERT, ContentVec), and cannot generalize to unseen speakers.

**High resource consumption**: High-quality zero-shot methods such as LDM-SVC require two-stage adversarial training, while SaMoye-SVC depends on 1,700 hours of singing data and 7 days of A100 training.

**Low synthesis quality**: Lightweight zero-shot approaches such as FastSVC exhibit a noticeable quality gap.

**Information loss**: Modeling speaker and content features separately leads to acoustic information loss, making feature fusion and natural voice reconstruction difficult.

### Key Insights

Speech and singing both belong to the human voice domain, and large-scale speech data can provide effective priors for singing tasks. FACodec from NaturalSpeech3 can simultaneously disentangle content and speaker features within a unified framework, reducing information loss from separate modeling. However, FACodec alone cannot fully capture the complex acoustic variations required for high-quality synthesis, necessitating additional acoustic feature enhancement and progressive synthesis optimization.

## Method

### Overall Architecture

The HQ-SVC pipeline consists of four stages:

1. **Disentangled feature extraction**: The pre-trained FACodec encoder and decoder are frozen; content features $x_{\text{con}}$ (256-dim) and speaker features $x_{\text{spk}}$ are extracted from intermediate layers.
2. **EVA multi-feature fusion**: F0 pitch (extracted via RMVPE), energy, and phase are introduced and deeply fused with content/speaker features.
3. **Progressive synthesis**: A preliminary waveform is first generated via DDSP and converted to a Mel spectrogram, which is then refined by a diffusion model.
4. **Vocoder generation**: NSF-HiFiGAN converts the Mel spectrogram and F0 into the final 44.1 kHz audio output.

### Disentangled Codec

FACodec is adopted as a unified disentangler, with both encoder and decoder fully frozen. FACodec internally employs three groups of factorized vector quantization (FVQ):

- Content quantizer $Q^c$: 2 quantizers
- Pitch quantizer $Q^p$: 1 quantizer
- Detail quantizer $Q^d$: 3 quantizers
- All codebook sizes are 1024

Freezing the pre-trained model avoids retraining and substantially reduces training resource requirements.

### Enhanced Voice Adaptor (EVA)

This is the core contribution module of the paper, addressing the limitation that content and speaker features alone are insufficient to capture the rich melodic and energy dynamics in singing.

**Feature extraction and projection**:

- Speaker feature $x_{\text{spk}}$ is projected via an MLP to obtain speaker embedding $e_{\text{spk}}$, with the residual serving as style feature $e_{\text{sty}}$
- F0 is log-transformed as $x_{f_0} = \ln(f_0/700 + 1)$ and projected via an MLP to obtain pitch embedding $e_{f_0}$
- Volume and phase are projected via separate MLPs to obtain $e_{\text{vol}}$ and $e_{\text{pha}}$
- All MLPs consist of two linear layers with SiLU activation, with output dimension 256

**Feature fusion strategy**:

Considering the strong coupling between pitch and timbre, the speaker and F0 embeddings are summed and concatenated with the remaining features to form a 1024-dim style embedding:

$$e_s = \text{Concat}(e_{\text{spk}} + e_{f_0},\ e_{\text{sty}},\ e_{\text{vol}},\ e_{\text{pha}})$$

After compression to 256 dimensions via 1D convolution, the FiLM mechanism is applied to fuse with content features:

$$\text{FiLM}(e_c, e_s) = f_\alpha(e_s) \cdot e_c + f_\beta(e_s)$$

The FiLM output is further processed through an 8-head self-attention Conformer with LayerNorm to produce the final embedding $e$ for subsequent synthesis modules.

**Speaker contrastive loss $\mathcal{L}_{\text{spk}}$**:

Based on InfoNCE loss, this term pulls together embeddings from the same speaker and pushes apart those from different speakers within a batch, enhancing zero-shot generalization. The temperature parameter is $\tau = 0.1$.

**Speaker-F0 Predictor (SFP)**:

To address the lack of sufficient pitch statistics for target speakers in zero-shot scenarios, an MLP module is designed to predict the mean and variance of F0 from the speaker embedding, supervised with an L1 loss:

$$\mathcal{L}_{f_0} = \mathbb{E}[\|\mu_{x_{f_0}} - \hat{\mu}_{x_{f_0}}\|_1 + \|\sigma^2_{x_{f_0}} - \hat{\sigma}^2_{x_{f_0}}\|_1]$$

### Progressive Singing Voice Reconstruction

**DDSP synthesis**: Harmonic and noise synthesizers generate periodic and aperiodic components respectively, introducing strong inductive biases to improve audio fidelity. The DDSP output is converted to a Mel spectrogram, and an MSE loss $\mathcal{L}_{\text{ddsp}}$ is computed.

**Diffusion model refinement**: A WaveNet is used as the denoiser (128-dim input, 20 residual blocks, 512 convolutional channels, 256-dim hidden layer) to recover acoustic details missed by DDSP. During inference, DPM-Solver++ is applied with 100 diffusion steps and 10× acceleration.

**Total loss**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ddsp}} + \mathcal{L}_{\text{diff}} + \mathcal{L}_{\text{spk}} + \mathcal{L}_{f_0}$

## Key Experimental Results

### Table 1: Main Results on Zero-Shot Singing Voice Conversion

| Method | Training Setup | Data | STOI↑ | SECS↑ | F0 RMSE↓ | FPC↑ | NISQA↑ | NMOS↑ | SMOS↑ |
|--------|---------------|------|-------|-------|----------|------|--------|-------|-------|
| FACodec-SVC | RTX 3090 (1h) | <80h | 0.533 | 0.074 | 77.798 | 0.601 | 1.791 | 2.391 | 2.740 |
| SaMoye-SVC | A100 (7 days) | 1700h | 0.724 | **0.647** | 17.418 | 0.617 | 3.528 | 3.958 | 3.569 |
| **HQ-SVC** | RTX 3090 (11h) | <80h | **0.799** | 0.627 | **8.681** | **0.891** | **3.841** | **4.215** | 3.578 |

HQ-SVC comprehensively outperforms SaMoye-SVC—trained for 7 days on 1,700 hours of data—across STOI (+10%), F0 RMSE (50% reduction), FPC (+44%), NISQA (+9%), and subjective NMOS (+6.5%).

### Table 2: Speech Super-Resolution Comparison

| Method | Data | LSD↓ | NISQA↑ | NMOS↑ | SMOS↑ |
|--------|------|------|--------|-------|-------|
| AudioSR | 7000h | 2.087 | 4.094 | 4.188 | 4.235 |
| **HQ-SVC** | <80h | **1.842** | **4.193** | **4.332** | **4.479** |

Even compared to the dedicated audio super-resolution model AudioSR (trained on 7,000 hours), HQ-SVC achieves superior results on LSD, NISQA, NMOS, and SMOS, demonstrating strong cross-task generalization.

## Highlights & Insights

1. **Extreme resource efficiency**: A single RTX 3090 with <6 GB VRAM, 11 hours of training, and <80 hours of data suffices to surpass SaMoye-SVC (A100 × 7 days + 1,700 hours), substantially lowering the barrier for zero-shot SVC.
2. **Unified disentanglement outperforms separate encoding**: Using FACodec to jointly disentangle content and speaker features within a unified framework yields better rhythm control and feature integration than the HQ-SVC-SE variant, which uses separate CAM++ + ContentVec encoders.
3. **Progressive synthesis pipeline**: DDSP provides an initial synthesis with strong inductive biases, while the diffusion model recovers fine-grained details—ablation experiments confirm that removing either component leads to significant quality degradation (removing diffusion drops NISQA from 3.841 to 3.175).
4. **Incidental super-resolution capability**: Because the model is trained on 16 kHz input features while optimizing 44.1 kHz Mel spectrograms, it natively supports speech super-resolution and surpasses dedicated methods in perceptual quality.
5. **Elegant Speaker-F0 Predictor design**: Predicting F0 statistics from the speaker embedding elegantly compensates for the unavailability of target pitch distributions in zero-shot scenarios.

## Limitations & Future Work

1. **Speaker similarity has room for improvement**: HQ-SVC's SECS is marginally lower than SaMoye-SVC (0.627 vs. 0.647); the unified codec is less effective at speaker identity separation than the separate encoder variant HQ-SVC-SE (SECS 0.668).
2. **Side effects of pitch-timbre coupling**: Adding $\mathcal{L}_{\text{spk}}$ and $\mathcal{L}_{f_0}$ slightly reduces SMOS, suggesting that enhanced pitch consistency can negatively affect human perception of timbre change when target and source singers have substantially different vocal ranges.
3. **Intelligibility loss**: In the speech super-resolution task, STOI (0.841 vs. 0.986) and FPC (0.868 vs. 0.998) are notably lower than AudioSR, indicating that the feature disentanglement and reconstruction process introduces minor articulation and pitch errors.
4. **Validation limited to Chinese singing**: Training data consists of Opensinger and M4Singer Chinese singing datasets; cross-lingual generalization has not been thoroughly validated.
5. **Style conversion not addressed**: The current work focuses solely on timbre conversion; singing style transfer (e.g., vocal technique, emotional expression) is left for future work.

## Related Work & Insights

- **Traditional SVC**: GMM-based methods require parallel data; GAN-based methods (FastSVC) suffer from training instability and limited quality.
- **Low-resource SVC**: DDSP-SVC and CoMoSVC achieve high-quality low-resource conversion via DDSP/diffusion models but rely on explicit speaker IDs.
- **Zero-shot SVC**: LDM-SVC achieves high-quality zero-shot conversion based on VITS but is computationally expensive; SaMoye-SVC requires large-scale data and adversarial training.
- **Audio codecs**: FACodec from NaturalSpeech3 enables content/speaker disentanglement; HQ-SVC extends this with enhanced acoustic detail modeling.
- **Speech super-resolution**: AudioSR is trained on 7,000 hours via LDM and focuses on low-frequency detail preservation; HQ-SVC surpasses it in naturalness using a general-purpose framework.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The EVA module design and the progressive DDSP+diffusion pipeline are innovative; the Speaker-F0 Predictor is concise and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Main experiments, super-resolution, sampler comparison, and ablation studies with complete objective and subjective evaluations; however, the number of baselines is limited (only one strong baseline, SaMoye-SVC).
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, complete formulations, and thorough ablation analysis.
- **Value**: ⭐⭐⭐⭐ — Significantly lowers the barrier for zero-shot SVC and offers substantial practical value for low-resource scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Temporal Inconsistency Guidance for Super-resolution Video Quality Assessment](temporal_inconsistency_guidance_for_super-resolution_video_quality_assessment.md)
- [\[ICLR 2026\] Breaking Scale Anchoring: Frequency Representation Learning for Accurate High-Resolution Inference from Low-Resolution Training](../../ICLR2026/image_restoration/breaking_scale_anchoring_frequency_representation_learning_for_accurate_high-res.md)
- [\[ICCV 2025\] Decouple to Reconstruct: High Quality UHD Restoration via Active Feature Disentanglement and Reversible Fusion](../../ICCV2025/image_restoration/decouple_to_reconstruct_high_quality_uhd_restoration_via_active_feature_disentan.md)
- [\[AAAI 2026\] ICLR: Inter-Chrominance and Luminance Interaction for Natural Color Restoration in Low-Light Image Enhancement](iclr_inter-chrominance_and_luminance_interaction_for_natural_color_restoration_i.md)
- [\[NeurIPS 2025\] Real-World Adverse Weather Image Restoration via Dual-Level Reinforcement Learning with High-Quality Cold Start](../../NeurIPS2025/image_restoration/real-world_adverse_weather_image_restoration_via_dual-level_reinforcement_learni.md)

<!-- RELATED:END -->
