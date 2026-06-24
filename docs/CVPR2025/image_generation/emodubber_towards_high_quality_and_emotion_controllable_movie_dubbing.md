---
title: >-
  [Paper Note] EmoDubber: Towards High Quality and Emotion Controllable Movie Dubbing
description: >-
  [CVPR 2025][Image Generation][Movie Dubbing] This paper proposes EmoDubber, an emotion-controllable movie dubbing framework. By aligning lip movements and prosody using duration-level contrastive learning, enhancing speech clarity via a pronunciation enhancement strategy, and controlling emotion categories and intensity through a flow matching-based positive-negative guidance mechanism, EmoDubber comprehensively outperforms existing methods in lip-sync and pronunciation clari…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Movie Dubbing"
  - "Emotion Controllable"
  - "Lip Synchronization"
  - "Flow Matching"
  - "Positive-Negative Guidance"
date: 2026-05-08
content_hash: 5271dcd2b0c2137b
---

# EmoDubber: Towards High Quality and Emotion Controllable Movie Dubbing

**Conference**: CVPR 2025  
**arXiv**: [2412.08988](https://arxiv.org/abs/2412.08988)  
**Code**: [https://github.com/GalaxyCong/DubFlow](https://github.com/GalaxyCong/DubFlow)  
**Area**: Diffusion Models  
**Keywords**: Movie Dubbing, Emotion Controllable, Lip Synchronization, Flow Matching, Positive-Negative Guidance

## TL;DR

This paper proposes EmoDubber, an emotion-controllable movie dubbing framework. By aligning lip movements and prosody using duration-level contrastive learning, enhancing speech clarity via a pronunciation enhancement strategy, and controlling emotion categories and intensity through a flow matching-based positive-negative guidance mechanism, EmoDubber comprehensively outperforms existing methods in lip-sync and pronunciation clarity.

## Background & Motivation

**Background**: Movie dubbing (Visual Voice Cloning, V2C) aims to convert text into speech synchronized with video lip movements while matching the specified speaker's voiceprint. Existing methods are divided into two categories: those focusing on speaker style representation (V2C-Net, StyleDubber), and those utilizing video information for prosody modeling (HPMDubbing, MCDubber).

**Limitations of Prior Work**: (1) Audio-visual synchronization and clear pronunciation are hard to guarantee simultaneously—existing methods operate at the video frame and mel-spectrogram levels, ignoring phone-level pronunciation information, which leads to mumbled generated speech; (2) Emotional expression is stiff and lacks controllability—users cannot specify emotional categories and intensity, which is a critical requirement in movie post-production.

**Key Challenge**: The dubbing task requires simultaneous optimization across four dimensions—lip-sync, pronunciation clarity, speaker cloning, and emotional control—whereas existing methods only address a subset of the first three.

**Goal**: (1) How to achieve precise lip-movement and prosody alignment? (2) How to improve pronunciation clarity? (3) How to allow users to flexibly control emotion categories and intensity?

**Key Insight**: The authors observe that emotions in human speech are often mixed rather than single. Therefore, a positive-negative guidance mechanism is designed to enhance the target emotion while suppressing other emotions, achieving more precise emotional control.

**Core Idea**: Align lip movements and prosody through duration-level contrastive learning, integrate phoneme sequences using pronunciation enhancement, and achieve flexible emotional intensity control through flow-matching guided by positive and negative classifiers.

## Method

### Overall Architecture

EmoDubber takes four inputs: silent video $V_l$, reference audio $R_a$, text $T_p$, and user emotion instruction $E = \{c, \alpha, \beta\}$. The output is the emotion-controllable dubbing audio $\hat{Y}$. The process is handled sequentially by four modules: (1) LPA aligns lip movements with prosody; (2) PE enhances pronunciation information; (3) SIA injects speaker style to generate acoustic priors; (4) FUEC generates mel-spectrograms via flow matching with emotions injected. Finally, a vocoder converts the output into waveforms.

### Key Designs

1. **Lip-related Prosody Aligning (LPA)**:

    - **Function**: Learn the inherent consistency between lip movements and speech prosody, establishing accurate temporal alignment.
    - **Mechanism**: Use lip embedding $\mathcal{E}$ as Query and phone prosody embedding $\mathcal{O}_p$ (comprising style phonemes + pitch + energy) as Key/Value to obtain the lip-prosody context sequence $C_{pho}$ through multi-head attention. The key innovation is Duration-Level Contrastive Learning (DLCL): using the "0-1" duration matrix $M^{gt}_{lip,pho}$ forced-aligned by MFA as positive sample pairs, encouraging attention weights at correct temporal positions to be higher than elsewhere. The loss is defined as $\mathcal{L}_{cl} = -\log \frac{\sum\exp(\text{sim}^+/\tau)}{\sum\exp(\text{sim})}$, guaranteeing monotonicity and surjectivity.
    - **Design Motivation**: Previous methods used simple MSE or unconstrained attention, which failed to guarantee precise temporal correspondence between lip movements and phonemes. DLCL explicitly enforces monotonic alignment via contrastive learning of positive-negative pairs, which is more flexible and precise compared to diagonal constraints.

2. **Pronunciation Enhancing (PE)**:

    - **Function**: Expand phone-level information to the video-frame level and fuse it with the lip-prosody sequence to enhance speech clarity.
    - **Mechanism**: Extract the explicit duration $D_p$ of each phoneme from the attention matrix using Monotonic Alignment Search (MAS), and expand the phone embedding $\mathcal{O}_s \in \mathbb{R}^{P \times d_m}$ to the video level $\mathcal{O}^v_s \in \mathbb{R}^{F \times d_m}$ using a length regulator. Then, use Audio-Visual Efficient Conformer (AVEC, 5 Conformer blocks + CTC layer) to fuse two features: lip-prosody context $C_{pho}$ and pronunciation enhancement sequence $\mathcal{O}^v_s$. The CTC layer guarantees correct pronunciation by maximizing the probability of correct phonemes.
    - **Design Motivation**: Existing dubbing methods operate at frame or mel-spectrogram levels, neglecting phoneme-level pronunciation details. Explicitly extending phone sequences and fusing them via Conformer provides phoneme-level supervision to avoid "muffled" speech.

3. **Flow-based User Emotion Controlling (FUEC)**:

    - **Function**: Inject user-specified emotion categories and intensities during the flow matching generation process.
    - **Mechanism**: Train a Flow Matching Prediction Network (FMPN) using OT-CFM to generate mel-spectrograms. During inference, a Positive-Negative Guidance Mechanism (PNGM) is introduced: a pre-trained emotion classifier $\psi$ predicts the emotion distribution of the current intermediate state $\phi_t(x)$, modifying the velocity field to $\tilde{v}_{t,i} = v_t + \gamma(\alpha \nabla\log p_\psi(c_i|\phi_t) - \beta \nabla\log p_\psi(\sum_{j\neq i} l_j c_j|\phi_t))$. Here, $\alpha$ represents positive guidance to enhance the target emotion $c_i$, and $\beta$ represents negative guidance to suppress the weighted mix of other emotions. Users can flexibly control emotion by adjusting $\alpha \in [0,9]$ and $\beta \in [0,2]$.
    - **Design Motivation**: Traditional classifier guidance only enhances the target emotion, but human speech features mixed emotions. PNGM achieves more precise emotional styling by performing simultaneous enhancement and suppression: increasing $\alpha$ emphasizes the target emotion, while increasing $\beta$ dampens other emotions, allowing both to be independently adjusted.

### Loss & Training

The total training loss comprises: flow matching loss $\mathcal{L}_\theta$ (MSE of predicted vs. target vector fields), contrastive learning loss $\mathcal{L}_{cl}$ (DLCL in LPA), and CTC loss (ensuring pronunciation correctness in PE). The emotion classifier $\psi$ is pre-trained on 13 emotion datasets from Emobox (50,000+ recordings). Phoneme encoder, USL, and flow decoder are pre-trained on LibriSpeech. During inference, $\gamma=15$.

## Key Experimental Results

### Main Results

Chem benchmark (Setting 1.0 + 2.0):

| Method | LSE-C↑ | LSE-D↓ | WER↓ | SECS↑ |
|------|--------|--------|------|-------|
| GT | 8.12 | 6.59 | 3.85 | 100.0 |
| HPMDubbing | 7.85 | 7.19 | 16.05 | 85.09 |
| StyleDubber | 3.87 | 10.92 | 13.14 | 87.72 |
| Speaker2Dub | 3.76 | 10.56 | 16.98 | 74.73 |
| **EmoDubber** | **8.11** | **6.92** | **11.72** | **90.62** |

Zero-shot setting (Dub 3.0):

| Method | LSE-C↑ | LSE-D↓ | WER↓ | MOS-S↑ | MOS-N↑ |
|------|--------|--------|------|--------|--------|
| StyleDubber | 6.17 | 9.11 | 15.10 | 4.03 | 3.85 |
| Speaker2Dub | 4.83 | 10.39 | 15.91 | 3.98 | 4.01 |
| **EmoDubber** | **7.40** | **6.65** | **14.03** | **4.07** | **4.05** |

### Ablation Study

| Configuration | LSE-C↑ | LSE-D↓ | WER↓ | Description |
|------|--------|--------|------|------|
| Full EmoDubber | 8.11 | 6.92 | 11.72 | Full model |
| w/o DLCL | ~5.5 | ~9.0 | ~14.0 | Removing contrastive learning leads to a significant decrease in lip-sync |
| w/o PE | ~7.0 | ~7.5 | ~15.0 | Removing pronunciation enhancement increases WER |
| w/o PNGM | - | - | - | Emotion intensity becomes uncontrollable |

### Key Findings

- EmoDubber almost matches the ground truth in lip-sync metrics (LSE-C: 8.11 vs. GT: 8.12), significantly outperforming all baseline methods.
- WER is reduced from 16.05% in HPMDubbing and 13.14% in StyleDubber to 11.72%, demonstrating a notable improvement in pronunciation clarity.
- Speaker similarity SECS reaches 90.62%, outperforming all baselines including StyleDubber's 87.72%.
- It maintains superior performance in zero-shot multi-speaker settings, indicating strong generalization capability.
- In PNGM, $\alpha$ and $\beta$ can be independently controlled: increasing $\alpha$ enhances the Intensity Score of the target emotion, while increasing $\beta$ suppresses alternative emotions.

## Highlights & Insights

- **Breakthrough in near-GT lip-sync**: LSE-C of 8.11 vs. GT 8.12 shows the extreme efficacy of DLCL's explicit duration-level alignment. This is more direct than prior hierarchical modeling in HPMDubbing.
- **Novel Positive-Negative Guidance concept (PNGM)**: Going beyond mere target enhancement, it actively dampens extraneous emotional components, aligning better with the mixed nature of human expressions. This design can generalize to any flow-matching/diffusion generation tasks requiring precise attribute control.
- **End-to-end phone-to-waveform pipeline**: Fully integrates alignment, enhancement, style injection, and emotional control, formulating the most comprehensive dubbing system to date.

## Limitations & Future Work

- The emotion classifier is fixed and pre-trained, limiting the number of treatable emotion classes.
- Evaluated only on English datasets; cross-lingual adaptability is unexplored.
- Training requires additional annotated MFA alignments and emotion labels.
- The interactive cost is high since $\gamma$, $\alpha$, and $\beta$ in PNGM must be manually configured by the user.
- It is not compared with the latest large-scale speech generation models (e.g., VALL-E, VoiceCraft).

## Related Work & Insights

- **vs HPMDubbing**: HPM models visual-to-prosody mapping hierarchically using lip, face, and scene tracks. EmoDubber explicitly aligns at the duration level via DLCL, massively leading in lip-sync (LSE-C 8.11 vs. 7.85) and additionally offering emotion control.
- **vs StyleDubber**: StyleDubber leverages a multi-scale style adapter to enhance speaker characteristics. EmoDubber's SIA module serves a similar purpose but incorporates an emotional control dimension via FUEC.
- **vs Emotional TTS (e.g., EmoSphere)**: Conventional emotional TTS relies only on positive guidance for emotional enhancement. EmoDubber's PNGM incorporates negative guidance to suppress non-target emotions, securing finer control.

## Rating

- Novelty: ⭐⭐⭐ vanity positive-negative guidance mechanism and duration-level contrastive learning display novelty, though the system is functionally a multi-module pipeline.
- Experimental Thoroughness: ⭐⭐⭐⭐ Diverse dubbing datasets, zero-shot setup, and extensive MOS reviews offer solid coverage.
- Writing Quality: ⭐⭐⭐⭐ The architecture is clearly detailed with excellent formula-to-figure correspondence.
- Value: ⭐⭐⭐⭐ Successfully unifies lip-syncing, pronunciation validity, and emotional control in a dubbing framework, proving highly applicable to movie post-production.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ArtiFade: Learning to Generate High-quality Subject from Blemished Images](artifade_learning_to_generate_high-quality_subject_from_blemished_images.md)
- [\[CVPR 2025\] OmniStyle: Filtering High Quality Style Transfer Data at Scale](omnistyle_filtering_high_quality_style_transfer_data_at_scale.md)
- [\[CVPR 2025\] StableAnimator: High-Quality Identity-Preserving Human Image Animation](stableanimator_high-quality_identity-preserving_human_image_animation.md)
- [\[CVPR 2025\] 3DTopia-XL: Scaling High-Quality 3D Asset Generation via Primitive Diffusion](3dtopia-xl_scaling_high-quality_3d_asset_generation_via_primitive_diffusion.md)
- [\[CVPR 2025\] EDEN: Enhanced Diffusion for High-quality Large-motion Video Frame Interpolation](eden_enhanced_diffusion_for_high-quality_large-motion_video_frame_interpolation.md)

</div>

<!-- RELATED:END -->
