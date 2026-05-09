---
title: >-
  [Paper Note] Cross-Modal Emotion Transfer for Emotion Editing in Talking Face Video
description: >-
  [CVPR 2026][Image Generation][Emotion Editing] This paper proposes C-MET (Cross-Modal Emotion Transfer), which models the mapping of emotion semantic vectors between speech and facial expression spaces, achieving for the first time speech-driven talking face video generation for extended emotions (e.g., sarcasm, charisma), surpassing the state of the art in emotion accuracy by 14%.
tags:
  - CVPR 2026
  - Image Generation
  - Emotion Editing
  - Cross-Modal Transfer
  - Talking Face Generation
  - Emotion Semantic Vector
  - Extended Emotions
date: 2026-05-08
content_hash: 916a082f697981db
---

# Cross-Modal Emotion Transfer for Emotion Editing in Talking Face Video

**Conference**: CVPR 2026
**arXiv**: [2604.07786](https://arxiv.org/abs/2604.07786)
**Code**: [https://chanhyeok-choi.github.io/C-MET/](https://chanhyeok-choi.github.io/C-MET/)
**Area**: Image Generation / Talking Face
**Keywords**: Emotion Editing, Cross-Modal Transfer, Talking Face Generation, Emotion Semantic Vector, Extended Emotions

## TL;DR
This paper proposes C-MET (Cross-Modal Emotion Transfer), which models the mapping of emotion semantic vectors between speech and facial expression spaces, achieving for the first time speech-driven talking face video generation for extended emotions (e.g., sarcasm, charisma), surpassing the state of the art in emotion accuracy by 14%.

## Background & Motivation

**State of the Field**: Emotional talking face generation is a core application of generative models, aiming to convert neutral talking videos into videos with target emotions. Existing methods are categorized into three types based on emotion source: label-driven, speech-driven, and image-driven.

**Limitations of Prior Work**: (1) Label-driven methods support only predefined discrete emotion categories (e.g., 8 basic emotions) and cannot represent complex or nuanced emotions; (2) In speech-driven methods, emotion and linguistic content are entangled and cannot be disentangled; (3) Image-driven methods require high-quality frontal reference images, and reference data for extended emotions (e.g., sarcasm) is difficult to obtain.

**Root Cause**: How to leverage rich speech emotion information to drive facial expression generation without collecting additional annotated data, especially for extended emotions unseen during training?

**Paper Goals**: Achieve cross-modal (speech→visual) emotion transfer while supporting zero-shot generation of extended emotions.

**Starting Point**: Rather than directly predicting facial expressions, the paper learns the mapping of "emotion semantic vectors"—defined as the difference between two emotion embeddings—between the speech space and the visual space.

**Core Idea**: Emotion Semantic Vector = target emotion embedding − input emotion embedding; a cross-modal Transformer learns the mapping from speech semantic vectors to visual semantic vectors.

## Method

### Overall Architecture
C-MET consists of three components: (a) pretrained encoders that extract speech/visual embeddings and compute semantic vectors; (b) multimodal token contrastive learning for representation space alignment; (c) a Transformer encoder that regresses the target visual semantic vector, and a decoder that reconstructs the emotional video.

### Key Designs

1. **Emotion Semantic Vector**:

    - **Function**: Given input emotion $i$ and target emotion $j$, the speech-space semantic vector is computed as $f_a^{i \to j} = f_a^j - f_a^i$, and the visual-space semantic vector as $f_v^{i \to j} = f_v^j - f_v^i$.
    - **Mechanism**: Difference vectors represent the direction of emotion change rather than modeling absolute emotions, enabling the model to learn "emotion transfer" rather than "emotion recognition."
    - **Design Motivation**: Inspired by EmoKnob's speech emotion control. Difference vectors exhibit good composability in continuous space, allowing the model to train on basic emotions and generalize to unseen extended emotions at inference.

2. **Multimodal Token Contrastive Learning**:

    - **Function**: A 1D convolution-based visual tokenizer $T_v$ and a projection-based audio tokenizer $T_a$ are used; bidirectional contrastive loss aligns token representations across modalities.
    - **Mechanism**:
    $L_{\text{cnt}} = \frac{L_{v \to a} + L_{a \to v}}{2}$
    - **Design Motivation**: Reduces the representational gap between speech and facial expression modalities to improve cross-modal regression accuracy.

3. **Cross-Modal Transformer Encoder**:

    - **Function**: Reference visual semantic vector $z_r$, target speech semantic vector $z_a$, and input visual embedding $z_v$ are concatenated into a token sequence and fed into a Transformer encoder to predict the target visual semantic vector:
    $\hat{f}_{v,t}^{i \to j} = P_v(TE(\{z_{r,t'}\} \| \{z_a\} \| \{z_{v,t}\}))$
    - **Mechanism**: The predicted semantic vector is added to the input visual embedding to obtain the target embedding, which is then fed into a pretrained decoder to generate the emotional video.
    - **Design Motivation**: Three types of type embeddings are introduced to distinguish different sources, enabling the Transformer to effectively model cross-modal dependencies.

### Loss & Training
- Reconstruction loss (bidirectional): $L_{\text{recon}} = L_{i \to j} + L_{j \to i}$
- Direction loss: $L_{\text{dir}} = 1 + \frac{\langle \hat{f}_v^{i \to j}, \hat{f}_v^{j \to i} \rangle}{\|\hat{f}_v^{i \to j}\| \|\hat{f}_v^{j \to i}\|}$ (ensures forward and reverse vectors point in opposite directions)
- Total loss: $L = L_{\text{recon}} + \lambda_{\text{cnt}} \cdot L_{\text{cnt}} + \lambda_{\text{dir}} \cdot L_{\text{dir}}$
- $\lambda_{\text{cnt}} = 0.1$, $\lambda_{\text{dir}} = 0.05$

## Key Experimental Results

### Main Results

| Method | Emotion Source | Acc_emo↑ (MEAD) | Acc_emo↑ (CREMA-D) | FID↓ | AITV↓ |
|--------|---------------|----------------|-------------------|------|-------|
| EAMM | Image | 18.81 | 19.15 | 161.6 | 3.745 |
| EAT | Label | 41.56 | 39.97 | 91.0 | 12.575 |
| EDTalk | Image | 41.99 | 29.69 | **76.4** | 2.827 |
| FLOAT | Speech | 13.21 | 29.11 | 92.8 | 1.434 |
| **C-MET** | **Speech** | **55.91** | **43.47** | 90.8 | 2.643 |

### Ablation Study

| Loss Configuration | Acc_emo↑ (MEAD) | Note |
|-------------------|----------------|------|
| $L_{\text{recon}}$ only | 49.43 | Baseline |
| + $L_{\text{cnt}}$ | 53.46 | Contrastive learning contributes +4% |
| + $L_{\text{cnt}}$ + $L_{\text{dir}}$ | **55.91** | Direction loss further contributes +2.4% |

Plug-and-play validation:

| Backbone | Original Acc_emo | + C-MET Acc_emo | AITV Change |
|----------|-----------------|----------------|-------------|
| PD-FGC | 33.36 | 36.82 (+3.46) | 1.247→1.180 (faster) |
| EDTalk | 41.99 | **55.91 (+13.92)** | 2.827→2.643 (faster) |

### Key Findings
- Emotion accuracy improves substantially (14% above SOTA), with a slight trade-off in visual quality metrics (FID/FVD), reflecting an inherent trade-off between emotion expressiveness and visual fidelity.
- In user studies, C-MET achieves overwhelming preference (>75%) under both basic and extended emotion settings.
- C-MET can serve as a plug-and-play module replacing heavy expression encoders while also reducing inference time.

## Highlights & Insights
- **Pioneering Contribution**: The first method to explicitly model speech-to-visual emotion semantic vector mapping.
- **Zero-Shot Extended Emotions**: Training uses only 8 basic emotions from MEAD; at inference, the model generalizes to unseen extended emotions such as sarcasm and charisma.
- **Plug-and-Play**: Seamlessly integrates into existing disentangled networks (EDTalk, PD-FGC) as a replacement for heavyweight expression encoders.

## Limitations & Future Work
- Visual quality metrics (FID, FVD) are slightly inferior to image-driven methods; strong emotion expression leads to larger facial motion deviations.
- The model depends on pretrained emotion2vec+large and EDTalk encoder/decoder components, and its performance ceiling is bounded by these components.
- Quantitative evaluation of extended emotions lacks a standardized benchmark; validation currently relies solely on user studies.

## Related Work & Insights
- Comparison with FLOAT (speech-driven but with entangled emotion and content) validates the necessity of disentangled design.
- The speech emotion control idea from EmoKnob is effectively generalized to visual generation.
- The strategy of using contrastive learning for modality alignment is transferable to other cross-modal transfer tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The emotion semantic vector mapping idea is novel; zero-shot generation of extended emotions is pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative + qualitative + user study + ablation are complete, but extended emotions lack standardized evaluation.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear, though notation is somewhat heavy.
- Value: ⭐⭐⭐⭐⭐ Practically valuable; addresses a key bottleneck in emotional talking face generation.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] DiFlowDubber: Discrete Flow Matching for Automated Video Dubbing via Cross-Modal Alignment and Synchronization](diflowdubber_discrete_flow_matching_for_automated_video_dubbing_via_cross-modal_.md)
- [\[CVPR 2026\] MOS: Mitigating Optical-SAR Modality Gap for Cross-Modal Ship Re-Identification](mos_mitigating_optical-sar_modality_gap_for_cross-modal_ship_re-identification.md)
- [\[CVPR 2026\] Preserving Source Video Realism: High-Fidelity Face Swapping for Cinematic Quality](preserving_source_video_realism_high-fidelity_face_swapping_for_cinematic_qualit.md)
- [\[AAAI 2026\] Multi-Aspect Cross-modal Quantization for Generative Recommendation](../../AAAI2026/image_generation/multi-aspect_cross-modal_quantization_for_generative_recommendation.md)
- [\[CVPR 2026\] FDeID-Toolbox: Face De-Identification Toolbox](fdeid-toolbox_face_de-identification_toolbox.md)

<!-- RELATED:END -->
