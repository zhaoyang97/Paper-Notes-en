---
title: >-
  [Paper Note] MoEE: Mixture of Emotion Experts for Audio-Driven Portrait Animation
description: >-
  [CVPR 2025][Human Understanding][mixture of emotion experts] This paper proposes the Mixture of Emotion Experts (MoEE) model, which trains an individual expert network for each of the 6 basic emotions and dynamically combines them through a Soft MoE gating mechanism. In conjunction with a 150-hour professional emotional talking-head dataset and a multi-modal emotion conditioning module, MoEE achieves precise and natural control over both single and compound emotions.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "mixture of emotion experts"
  - "audio-driven portrait animation"
  - "compound emotion"
  - "Action Units"
  - "DH-FaceEmoVid-150"
date: 2026-05-08
content_hash: d9b910fbaf46d1df
---

# MoEE: Mixture of Emotion Experts for Audio-Driven Portrait Animation

**Conference**: CVPR 2025  
**arXiv**: [2501.01808](https://arxiv.org/abs/2501.01808)  
**Code**: To be confirmed (dataset will be publicly released)  
**Area**: Human Understanding  
**Keywords**: mixture of emotion experts, audio-driven portrait animation, compound emotion, Action Units, DH-FaceEmoVid-150

## TL;DR

This paper proposes the Mixture of Emotion Experts (MoEE) model, which trains an individual expert network for each of the 6 basic emotions and dynamically combines them through a Soft MoE gating mechanism. In conjunction with a 150-hour professional emotional talking-head dataset and a multi-modal emotion conditioning module, MoEE achieves precise and natural control over both single and compound emotions.

## Background & Motivation

**Background**: Audio-driven portrait animation has made significant progress in lip synchronization (e.g., Hallo, AniPortrait), but emotional control remains weak. Existing methods (e.g., EAT, StyleTalk) either support limited emotion categories or require target reference videos to transfer emotional styles, lacking flexible emotional control capabilities.

**Core Problem**:
1. **Lack of a basic emotion modeling framework**: The lack of precise modeling for individual basic emotions prevents compound emotions (e.g., "angry disgust", "sadly surprised") from being synthesized through combination.
2. **Lack of high-quality emotion datasets**: Existing emotional datasets (e.g., MEAD) are small in scale, limited in emotion categories, and lack fine-grained annotations (such as AU labels and textual descriptions).

**Motivation**: Inspired by the MoE architecture, if each basic emotion has its corresponding expert model, compound emotions can be synthesized through the soft combination of expert weights, resembling a "palette" mechanism.

## Method

### Overall Architecture

MoEE is based on the denoising U-Net architecture of Stable Diffusion and adopts a two-stage training strategy:
1. **Stage 1**: Fine-tune the Reference Net and Denoising U-Net on the entire emotion dataset to learn rich expression priors.
2. **Stage 2**: Freeze the spatial, cross, audio, and temporal modules, and only train the Emotion MoE module and the Emotion-to-Latents module.

Input: A portrait image $\mathbf{I}$, an audio sequence $\mathbf{A}$, and an emotion condition $\mathbf{C}$ (text/label/audio). Output: A talking head video with the target emotion.

### Key Designs

#### Design 1: Mixture of Emotion Experts

- **6 Experts**: Corresponding to happiness, sadness, anger, disgust, fear, and surprise, respectively. Each expert is a cross-attention module trained on single-emotion data.
- **Soft MoE Gating**: Unlike Hard MoE (which selects only one expert), it employs soft assignment to allow multiple experts to process the input concurrently.
    - **Local Assignment**: A gating layer $s = \text{sigmoid}(G(X, \phi))$ is learned, where $s \in \mathbb{R}^{n \times 6}$. Weights are independently assigned per token to achieve fine-grained control over detailed local facial expressions.
    - **Global Assignment**: $g = \text{softmax}(G(\text{Pool}(X), \omega))$, where $g \in \mathbb{R}^6$, using 6 global scalars to control the overall emotional tone.
- **Combination Formula**: $X' = X + \sum_{i=1}^6 g_i \cdot E_i(X \cdot s_i)$

**Single Emotion**: Activates only the corresponding expert | **Compound Emotion**: Soft combination of multiple experts

#### Design 2: Emotion-to-Latents Module

- **Multimodal Input Alignment**: Three modalities—text (T5 encoder), audio (emotion2vec), and label (self-trained MLP)—are encoded individually and then mapped to the same dimension via a fully connected (FC) layer.
- **Learnable Embeddings**: A set of learnable tokens is maintained as the key/value for attention, transforming multi-modal features into a unified emotion latent.
- **Injection Mechanism**: The emotion latent is injected into the U-Net as the key/value of the cross-attention inside the Emotion MoE Module.

#### Design 3: Masked Noisy Emotion Sampling

- **Problem**: The sub-datasets for single emotions are relatively small, which makes it easy for experts to overfit and learn information unrelated to the target emotion.
- **Solution**: Mix data from other emotions/neutral expressions into the training with a certain probability (adding noise) to expand the diversity of person identities.
- **Mouth Masking**: Different emotions originate from different talking videos, and major differences in lip shapes can distract the model. MediaPipe is used to locate and mask the mouth area, forcing the model to focus on general expression changes rather than the lip shapes.

### Loss & Training

$$L = L_{latent} + \lambda L_{spatial}$$

- $L_{latent}$: Standard diffusion denoising loss $\mathbb{E}_{t,c,z_t,\epsilon}[\|\epsilon - \epsilon_\theta(z_t, t, c)\|^2]$
- $L_{spatial}$: Timestep-aware pixel-level loss, which computes L1 + Perceptual Loss in the decoded image space.
    - $L_{spatial} = w(t)(||I_p, I_{GT}|| + ||V(I_p), V(I_{GT})||^2)$
    - $w(t) = \cos(t \cdot \pi / 2T)$, assigning a smaller weight to larger timesteps (as pixel-level constraints are less meaningful when noise levels are high).

## Key Experimental Results

### Main Results

**HDTF Dataset (Table 2)**:

| Method | FID↓ | FVD↓ | LPIPS↓ | Sync-C↑ |
|---|---|---|---|---|
| AniPortrait | 36.83 | 476.82 | 0.211 | 5.977 |
| Hallo | 28.61 | 343.02 | 0.167 | 6.254 |
| EAT | 81.25 | 545.27 | 0.357 | 5.012 |
| **MoEE** | **28.83** | **322.63** | **0.152** | 6.114 |

**DH-FaceEmoVid-150 Dataset (Table 4)**:

| Method | FID↓ | FVD↓ | LPIPS↓ | AKD↓ |
|---|---|---|---|---|
| AniPortrait | 66.03 | 712.29 | 0.323 | 20.654 |
| Hallo | 72.35 | 702.84 | 0.329 | 16.444 |
| EAT | 48.01 | 467.74 | 0.260 | 14.109 |
| **MoEE** | **39.62** | **402.80** | **0.182** | **4.028** |

### Ablation Study (Table 5)

| Variant | FID↓ | FVD↓ | LPIPS↓ | AKD↓ |
|---|---|---|---|---|
| w/o MoEE | 58.41 | 655.33 | 0.325 | 14.959 |
| w/o Global Soft Assignment | 46.33 | 447.81 | 0.194 | 10.652 |
| w/o Masked Noisy Sampling | 51.79 | 489.24 | 0.211 | 4.591 |
| w/o DH-FaceEmoVid-150 | 52.41 | 511.93 | 0.275 | 7.885 |
| **Full MoEE** | **39.62** | **402.80** | **0.182** | **4.028** |

### Key Findings

1. **MoEE module contributes the most**: Without it, FID degrades from 39.62 to 58.41, and AKD degrades from 4.03 to 14.96.
2. **Global soft assignment is indispensable**: Removing it causes AKD to degrade from 4.03 to 10.65, demonstrating that global emotional tone control is critical for facial expression accuracy.
3. **Significant dataset impact**: Excluding only DH-FaceEmoVid-150 degrades FID from 39.62 to 52.41.
4. Masked Noisy Sampling significantly improves FID/FVD (from 51.79 to 39.62) while keeping AKD stable.
5. Visualization of the emotional latent space shows that the distribution of different emotions becomes more separated with MoEE.

## Highlights & Insights

1. **Palette Analogy**: Modeling compound emotions as a soft combination of basic emotions is conceptually intuitive and practically effective, highly aligning with Ekman's basic emotion theory in psychology.
2. **Local + Global Gating**: Local assignment regulates local facial regions (micro-expressions like eyebrows and mouth corners) while global assignment governs the overall emotional tone. This dual-level control is much more precise than a single gating mechanism.
3. **Ingenious Masked Noisy Sampling**: It addresses the overfitting problem of expert training on small datasets while shifting the focus from mouth shapes to expression variations through mouth masking.
4. **High Value of the DH-FaceEmoVid-150 Dataset**: Offering 150 hours of 1080p video with compound emotions and AU annotations, it substantially fills the data vacancy in this domain.

## Limitations & Future Work

1. It only covers 6 basic emotions and 4 compound emotions; subtle emotions such as sarcasm, embarrassment, and helplessness remain unaddressed.
2. Training with Chinese speech may result in evaluations of Sync-C/Sync-D that are not completely objective.
3. Action Unit (AU) annotations rely on automatic generation via ME-GraphAU + GPT-4V, which may introduce noise.
4. Inference speed is limited by the diffusion model, making it difficult to achieve real-time application.

## Related Work & Insights

- **Hallo/AniPortrait/EchoMimic**: State-of-the-art (SOTA) methods for audio-driven portrait animation, but lacking emotional control.
- **EAT/StyleTalk**: Representative emotional control methods, but the expressions are either unnatural or require reference videos.
- **Soft MoE**: An architecture borrowed from the NLP domain, which this paper successfully applies to emotional decoupling.
- **Insights**: The MoE architecture is not only suitable for model capacity expansion but also effective for decoupling semantic dimensions (such as emotion, style, etc.). This paradigm of "one expert dedicated to one semantic aspect" is worth extending to other controlled generation tasks.

## Rating

⭐⭐⭐⭐ — The integration of MoE with emotional decoupling is natural and effective. The dataset contribution substantially fills the domain gap, and the method significantly outperforms existing approaches in both naturalness and accuracy of emotional control. One star is deducted because the variety of emotions is still limited, and the inference efficiency does not support real-time applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Sonic: Shifting Focus to Global Audio Perception in Portrait Animation](sonic_shifting_focus_to_global_audio_perception_in_portrait_animation.md)
- [\[CVPR 2025\] KeyFace: Expressive Audio-Driven Facial Animation for Long Sequences via KeyFrame Interpolation](keyface_expressive_audio-driven_facial_animation_for_long_sequences_via_keyframe.md)
- [\[AAAI 2026\] Spatiotemporal-Untrammelled Mixture of Experts for Multi-Person Motion Prediction](../../AAAI2026/human_understanding/spatiotemporal-untrammelled_mixture_of_experts_for_multi-person_motion_predictio.md)
- [\[CVPR 2026\] DeX-Portrait: Disentangled and Expressive Portrait Animation via Explicit and Latent Motion Representations](../../CVPR2026/human_understanding/dex-portrait_disentangled_and_expressive_portrait_animation_via_explicit_and_lat.md)
- [\[CVPR 2025\] X-Dyna: Expressive Dynamic Human Image Animation](x-dyna_expressive_dynamic_human_image_animation.md)

</div>

<!-- RELATED:END -->
