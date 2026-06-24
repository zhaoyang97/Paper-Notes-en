---
title: >-
  [Paper Note] Improving Sound Source Localization with Joint Slot Attention on Image and Audio
description: >-
  [CVPR 2025][Audio & Speech][Sound Source Localization] Proposes a joint slot attention mechanism to decompose both images and audio into target/non-target representations, achieving precise sound source localization through cross-modal attention matching and contrastive learning, resulting in SOTA performance of 65.16% AUC and 86.00% cIoU on Flickr-SoundNet.
tags:
  - "CVPR 2025"
  - "Audio & Speech"
  - "Sound Source Localization"
  - "Slot Attention"
  - "Cross-modal Contrastive Learning"
  - "Attention Matching"
  - "Audio-visual Alignment"
date: 2026-05-08
content_hash: acf8bda7fea76e76
---

# Improving Sound Source Localization with Joint Slot Attention on Image and Audio

**Conference**: CVPR 2025  
**arXiv**: [2504.15118](https://arxiv.org/abs/2504.15118)  
**Code**: Yes  
**Area**: Audio & Speech / Sound Source Localization  
**Keywords**: Sound Source Localization, Slot Attention, Cross-modal Contrastive Learning, Attention Matching, Audio-visual Alignment

## TL;DR

Proposes a joint slot attention mechanism to decompose both images and audio into target/non-target representations, achieving precise sound source localization through cross-modal attention matching and contrastive learning, resulting in SOTA performance of 65.16% AUC and 86.00% cIoU on Flickr-SoundNet.

## Background & Motivation

**Background**: Sound Source Localization (SSL) localizes sounding objects in images. Existing methods perform localization by contrasting global audio features with global/local visual features, but ignore scenarios where multiple objects may be present but only some are emitting sound.

**Limitations of Prior Work**: Global contrastive learning aligns all visual information with audio, causing non-sounding objects to receive high activations as well. A method is needed to separate "sounding objects" from "background/non-sounding objects" in both modalities.

**Key Challenge**: The audio signal is global (microphones capture all sounds), while the visual scene is spatial (objects have distinct positions). How to separate them without prior knowledge of which object is sounding?

**Key Insight**: Utilize slot attention to perform competitive decomposition simultaneously in both modalities—two learnable slots competitively attend to input features, naturally forming a separation between "target" and "non-target".

**Core Idea**: Bi-modal joint slot attention to separate target/non-target + cross-modal attention matching = precise sound source localization.

## Method

### Key Designs

1. **Joint Slot Attention Decomposition**:

    - **Function**: Decomposes images and audio individually into target and non-target slots.
    - **Mechanism**: Two learnable queries (target $\mathbf{p}$, non-target $\mathbf{r}$) interact with visual/audio features through cross-attention to competitively distribute attention. A divergence loss $\mathcal{L}_{div}$ ensures that the two slots do not degenerate into the same representation.
    - **Design Motivation**: Two slots are sufficient (experiments show more slots do not help), naturally leading to functional differentiation.

2. **Cross-modal Attention Matching**:

    - **Function**: Uses the self-attention map of one modality to guide the cross-modal attention of the other modality.
    - **Mechanism**: $\mathcal{L}_{match} = \|\text{ca}^{a,v} - \text{sg}(\text{ia}^{v,v})\|_2^2 + \|\text{ca}^{v,a} - \text{sg}(\text{ia}^{a,a})\|_2^2$, transferring internal visual attention patterns to audio-to-visual attention.
    - **Design Motivation**: Self-attention has already learned to focus on important regions, so cross-modal attention should target the same areas.

3. **Target-Slot-Only Contrastive Learning**:

    - **Function**: Aligns only the target slots of both modalities, ignoring non-target ones.
    - **Mechanism**: $\mathcal{L}_{cotr}$ is computed as the contrastive loss only between target slots $\mathbf{p}^v, \mathbf{p}^a$.
    - **Design Motivation**: Non-target slots containing noise/background information should not be involved in alignment, otherwise, localization accuracy will be degraded.

### Loss & Training

$\mathcal{L} = \mathcal{L}_{cotr} + \lambda_1\mathcal{L}_{match} + \lambda_2\mathcal{L}_{div} + \lambda_3\mathcal{L}_{recon}$. The reconstruction loss $\mathcal{L}_{recon}$ reconstructs original features from the two slots to ensure decomposition integrity.

## Key Experimental Results

### Main Results

| Dataset | cIoU | AUC |
|--------|------|-----|
| Flickr-SoundNet | **86.00%** | **65.16%** |
| VGG-Sound | **86.00%** | **64.90%** |
| Prev. SOTA (FNAC) | 85.74% | 63.66% |

### Ablation Study

- Cross-modal attention matching is the most critical for localization.
- The two-slot design is optimal (more slots yield no gains).
- False negative mitigation (k-reciprocal NN) improves robustness.

### Key Findings
- **Effective slot attention decomposition**: Target slots naturally focus on sounding objects, while non-target slots focus on background.
- **Attention matching > Simple feature alignment**: Directly matching attention patterns is more effective than feature-level contrast.

## Highlights & Insights
- **Novel application of slot attention in audio-visual tasks**—originally used for unsupervised object discovery, here applied to cross-modal sound source separation.
- **Decompose-then-align paradigm**—separating target/non-target first and then aligning only targets is cleaner than global alignment.

## Limitations & Future Work
- Single sound source assumption (fails in multi-source scenarios).
- Requires paired audio-visual data.
- Performance degradation under noise/complex background audio.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of joint slot attention and attention matching is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple datasets and multiple metrics.
- Writing Quality: ⭐⭐⭐⭐ Clear description of the methodology.
- Value: ⭐⭐⭐⭐ Advances the sound source localization SOTA.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Object-aware Sound Source Localization via Audio-Visual Scene Understanding](object-aware_sound_source_localization_via_audio-visual_scene_understanding.md)
- [\[ICLR 2026\] Physics-Informed Audio-Geometry-Grid Representation Learning for Universal Sound Source Localization](../../ICLR2026/audio_speech/physics-informed_audio-geometry-grid_representation_learning_for_universal_sound.md)
- [\[CVPR 2026\] How Far Can We Go With Synthetic Data for Audio-Visual Sound Source Localization?](../../CVPR2026/audio_speech/how_far_can_we_go_with_synthetic_data_for_audio-visual_sound_source_localization.md)
- [\[CVPR 2025\] Towards Open-Vocabulary Audio-Visual Event Localization](towards_open-vocabulary_audio-visual_event_localization.md)
- [\[NeurIPS 2025\] Seeing Sound, Hearing Sight: Uncovering Modality Bias and Conflict of AI Models in Sound Localization](../../NeurIPS2025/audio_speech/seeing_sound_hearing_sight_uncovering_modality_bias_and_conflict_of_ai_models_in.md)

</div>

<!-- RELATED:END -->
