---
title: >-
  [Paper Note] DiFlowDubber: Discrete Flow Matching for Automated Video Dubbing via Cross-Modal Alignment and Synchronization
description: >-
  [CVPR 2026][Image Generation][Video Dubbing] Ours proposes DiFlowDubber, an automated video dubbing framework based on **Discrete Flow Matching (DFM)**. Through a two-stage training pipeline (Zero-shot TTS pre-training → Video dubbing adaptation), large-scale TTS knowledge is transferred to video-driven dubbing. The framework features a FaPro module to capture facial expression-prosody mapping and a Synchronizer module for precise lip-sync.
tags:
  - CVPR 2026
  - Image Generation
  - Video Dubbing
  - Discrete Flow Matching
  - Cross-Modal Alignment
  - Speech Synthesis
  - Lip-Sync
date: 2026-05-08
content_hash: 53c859ff39a3bc75
---

# DiFlowDubber: Discrete Flow Matching for Automated Video Dubbing via Cross-Modal Alignment and Synchronization

**Conference**: CVPR 2026  
**arXiv**: [2603.14267](https://arxiv.org/abs/2603.14267)  
**Code**: [Demo](https://nngocson2002.github.io/projects/diflowdubber)  
**Area**: Image Generation / Multimodal Speech Generation  
**Keywords**: Video Dubbing, Discrete Flow Matching, Cross-Modal Alignment, Speech Synthesis, Lip-Sync  

## TL;DR

Ours proposes DiFlowDubber, an automated video dubbing framework based on **Discrete Flow Matching (DFM)**. Through a two-stage training pipeline (Zero-shot TTS pre-training → Video dubbing adaptation), large-scale TTS knowledge is transferred to video-driven dubbing. The framework features a FaPro module to capture facial expression-prosody mapping and a Synchronizer module for precise lip-sync.

## Background & Motivation

**Background**: Video-to-Speech (V2C) dubbing has extensive applications in film production, multimedia creation, and assistive voice technologies. It requires generating natural speech that preserves the speaker's timbre while ensuring synchrony with lip movements and conveying visual emotions.

**Limitations of Prior Work**:
   - Methods trained directly on limited dubbing datasets struggle to produce expressive prosody and high audio quality.
   - Two-stage methods (TTS pre-training followed by adaptation) can leverage large-scale corpora but fail to simultaneously guarantee prosodic expression and lip-sync.

**Key Challenge**: Speaker2Dubber only adapts the phoneme encoder, failing to fully utilize the prosody and acoustic modeling capabilities of TTS. ProDubber introduces prosody enhancement but relies on a duration predictor to estimate lip movement, which is **not constrained by actual video length**, leading to poor synchronization (low LSE scores).

**Goal**: Design a framework that fully utilizes the prosody, content, and acoustic modeling capabilities of large-scale TTS pre-training, adapting them to video dubbing to ensure accurate pronunciation, naturalness, and precise lip-sync.

**Key Insight**: Use FACodec to decompose speech into three types of discrete tokens: prosody, content, and acoustics. Different attributes are modeled separately—prosody and acoustics via generative DFM, and content via a deterministic architecture.

**Core Idea**: Two-stage training pipeline + Discrete Flow Matching backbone + Facial expression-to-prosody mapping + Dual-alignment synchronizer.

## Method

### Overall Architecture

Two-stage pipeline (Figure 2):
- **Stage 1 (Zero-shot TTS Pre-training)**: Train a zero-shot TTS system on LibriTTS to learn prosody/content/acoustic representations of speech.
    - **Content Modeling**: A deterministic architecture directly predicts content tokens.
    - **Prosody-Acoustic Modeling**: A Discrete Flow Matching Prosody-Acoustic (DFPA) module generates expressive prosody and acoustic tokens.
- **Stage 2 (Video Dubbing Adaptation)**: Adapt the pre-trained TTS to the dubbing task.
    - FaPro module extracts global prosody priors from facial expressions.
    - Synchronizer achieves text-video-speech tri-modal alignment.
    - CCTA ensures content consistency.
    - DFPA is adapted for vision-conditioned generation.

### Key Designs

#### 1. Discrete Flow Matching Prosody-Acoustic (DFPA) Module

- **Function**: Jointly models the distribution of prosody and acoustic tokens.
- **Mechanism**: A denoiser based on the DiT architecture takes the current denoising target $\mathbf{x}_t$ (concatenated prosody and acoustic tokens) as input, with conditions including the content latent representation $\tilde{\mathbf{h}}_c$ and speaker embeddings. The optimization objective is:
  $$\mathcal{L}_{\text{DFM}} = -\sum_{i \in \mathcal{T}} \mathbb{E}_{t \sim \mathcal{U}[0,1]} [\log p_{1|t}(\mathbf{x}_1^i | \mathbf{x}_t, \mathbf{c}; \theta)]$$
- **Design Motivation**: FACodec decouples speech attributes into independently controllable tokens; DFM captures expressive prosodic variations better than autoregressive models.

#### 2. FaPro (Facial Expression to Prosody Mapping)

- **Function**: Extracts global prosody priors $\tilde{\mathbf{z}}_p \in \mathbb{R}^{m \times L \times D}$ from facial video frames.
- **Mechanism**: Facial features $\mathbf{v}_{\text{face}}$ → Upsampled to align with speech length → ConvNeXt V2 for enhanced temporal modeling → $m$-layer FFT iterations to extract prosody representations for each RVQ codebook.
- **Design Motivation**: Facial expressions are strongly correlated with speech prosody; global prosody (emotion, speaking rate, intonation) is largely determined by facial expressions.

#### 3. Synchronizer (Dual-Alignment Synchronizer)

- **Function**: Bridges the gaps between text, video, and speech modalities.
- **Mechanism**:
    - **Video-Text Alignment**: Cross-attention with lip features as query and phoneme features as key/value, applying a contrastive loss $\mathcal{L}_{VT}$ using the MFA alignment matrix $\mathcal{M}_{VT}$.
    - **Duration Regularization**: Attention weights are converted to phoneme-frame durations via MAS; phoneme embeddings are replicated by duration and upsampled to speech length.
    - **Speech-Text Alignment**: A second layer of contrastive alignment $\mathcal{L}_{ST}$ is applied to the upsampled representation to correct minor misalignments.
    - ConvNeXt V2 refinement yields the final aligned representation $\mathbf{h}_{\text{sync}} \in \mathbb{R}^{L \times D}$.
- **Design Motivation**: ProDubber's duration predictor is unconstrained by video length; the dual-alignment mechanism explicitly forces temporal consistency between lip shapes and speech.

#### 4. CCTA (Content Consistency Temporal Adaptation)

- **Function**: Transfers semantic content knowledge from the TTS domain to maintain linguistic consistency.
- **Mechanism**: Initialized with pre-trained TTS weights, freezing projection layers and the content head, and replacing the TTS duration predictor with the Synchronizer. A distillation loss maintains consistency between teacher and student features:
  $$\mathcal{L}_{\text{distill}} = \frac{1}{B} \sum_{i=1}^{B} [1 - \cos(\phi(\mathbf{z}_t^{(i)}), \phi(\mathbf{z}_c^{(i)}))]$$

### Loss & Training

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{align}} + \lambda_1 \mathcal{L}_c + \lambda_2 \mathcal{L}_{\text{CTC}} + \lambda_3 \mathcal{L}_{\text{distill}} + \lambda_4 \mathcal{L}_{\text{DFM}}$$

where $\mathcal{L}_{\text{align}} = \lambda_5 \mathcal{L}_{VT} + \lambda_6 \mathcal{L}_{ST}$.

## Key Experimental Results

### Main Results: Chem Dataset (Setting 1.0)

| Method | LSE-C↑ | LSE-D↓ | WER↓ | SECS↑ | UTMOS↑ |
|------|--------|--------|------|-------|--------|
| HPMDubbing (CVPR'23) | 7.85 | 7.19 | 16.05 | 85.09 | 2.16 |
| EmoDubber (CVPR'25) | 8.11 | 6.92 | 11.72 | 90.62 | 3.82 |
| ProDubber (CVPR'25) | 2.58 | 12.54 | **9.45** | 72.13 | 3.85 |
| **DiFlowDubber** | **8.31** | **6.73** | 9.65 | 84.59 | **4.02** |

### Main Results: GRID Dataset (Setting 1.0)

| Method | LSE-C↑ | LSE-D↓ | WER↓ | UTMOS↑ |
|------|--------|--------|------|--------|
| EmoDubber (CVPR'25) | 7.12 | 6.82 | 18.53 | 3.83 |
| **DiFlowDubber** | **7.32** | **6.73** | **16.79** | **3.95** |

### Ablation Study

| Configuration | LSE-C↑ | WER↓ | UTMOS↑ |
|------|--------|------|--------|
| Full Model | **8.31** | **9.65** | **4.02** |
| w/o TTS Pre-training | 8.17 | 12.04 | 3.53 |
| w/o $\mathcal{L}_{VT}+\mathcal{L}_{ST}$ | 8.26 | 17.15 | 3.90 |
| w/o $\mathcal{L}_{VT}$ | 8.36 | 16.57 | 3.87 |
| w/o $\mathcal{L}_{ST}$ | 8.31 | 12.60 | 3.93 |
| w/o $\mathcal{L}_{\text{distill}}$ | 8.33 | 12.62 | 3.97 |

### Key Findings

1. **Synchronization**: DiFlowDubber significantly leads in LSE-C/LSE-D, proving the effectiveness of the Synchronizer. Although ProDubber has a low WER, its synchrony is extremely poor (LSE-C of only 2.58), behaving more like TTS than dubbing.
2. **Value of TTS Pre-training**: Removing pre-training causes WER to rise from 9.65 to 12.04 and UTMOS to drop from 4.02 to 3.53, proving the necessity of transferring large-scale TTS knowledge.
3. **Dual-Alignment Complementarity**: $\mathcal{L}_{VT}$ primarily aids synchronization, while $\mathcal{L}_{ST}$ primarily aids pronunciation accuracy; both are indispensable.
4. MOS-S subjective evaluations surpass all recent baselines, proving the best perceptual quality.

## Highlights & Insights

1. **Elegant FACodec Decomposition Strategy**: Decoupling speech into prosody/content/acoustic tokens allows different modeling strategies (deterministic vs. generative) for each attribute and facilitates visual condition injection.
2. **Synchronizer's Double Insurance**: Performing video-text alignment followed by speech-text alignment after upsampling is more robust than single-layer alignment.
3. **First Application of Discrete Flow Matching (DFM) to Video Dubbing**, demonstrating DFM's effectiveness in cross-modal speech generation.

## Limitations & Future Work

1. **Inconspicuous SECS Scores**: Speaker similarity metrics are not as high as some baselines. While the authors attribute this to evaluation bias (baseline models share speaker encoders with SECS calculation), it remains an area for improvement.
2. **Dependency on MFA External Tools**: Requires Montreal Forced Aligner to provide phoneme-frame alignment information as training supervision.
3. **Evaluation Limited to Chem and GRID**: Both are controlled environment datasets; generalization to in-the-wild scenarios is unknown.
4. Inference Speed: Discrete Flow Matching with 128 NFEs might be slow; no detailed speed comparison was provided.

## Related Work & Insights

- **FACodec Speech Decomposition**: Borrowed from NaturalSpeech 3, the idea of factorizing speech attributes is worth adapting for other cross-modal tasks.
- **Contrastive Alignment Loss**: Dual contrastive alignment for video-text and speech-text can be generalized to any scenario requiring multimodal temporal alignment.
- **Two-stage Pre-training-Adaptation Paradigm**: Learning unimodal capabilities on large-scale data first before introducing new modal conditions via adaptation is an effective general strategy for cross-modal generation.

## Rating

⭐⭐⭐⭐ — Strong systematicity, well-designed modules, significant synchronization improvements, though evaluation scale is limited and the text is somewhat verbose.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Cross-Modal Emotion Transfer for Emotion Editing in Talking Face Video](cross-modal_emotion_transfer_for_emotion_editing_in_talking_face_video.md)
- [\[CVPR 2026\] MOS: Mitigating Optical-SAR Modality Gap for Cross-Modal Ship Re-Identification](mos_mitigating_optical-sar_modality_gap_for_cross-modal_ship_re-identification.md)
- [\[CVPR 2026\] Score2Instruct: Scaling Up Video Quality-Centric Instructions via Automated Dimension Scoring](score2instruct_scaling_up_video_quality-centric_instructions_via_automated_dimen.md)
- [\[ICLR 2026\] Discrete Adjoint Matching](../../ICLR2026/image_generation/discrete_adjoint_matching.md)
- [\[CVPR 2026\] VeCoR — Velocity Contrastive Regularization for Flow Matching](vecor_--_velocity_contrastive_regularization_for_flow_matching.md)

<!-- RELATED:END -->
