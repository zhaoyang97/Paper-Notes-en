---
title: >-
  [Paper Note] MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing
description: >-
  [ICCV 2025][Audio & Speech][Audio-visual video parsing] This paper proposes the MUG framework, which simultaneously improves segment-level and event-level prediction in weakly supervised audio-visual video parsing (AVVP) through a pseudo label-augmented cross-modal random combination data augmentation strategy and an audio-visual Mamba network.
tags:
  - "ICCV 2025"
  - "Audio & Speech"
  - "Audio-visual video parsing"
  - "Mamba"
  - "pseudo labels"
  - "data augmentation"
  - "weakly supervised learning"
date: 2026-05-08
content_hash: 8899fe3a995daf86
---

# MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing

**Conference**: ICCV 2025
**arXiv**: [2507.01384](https://arxiv.org/abs/2507.01384)  
**Code**: [https://github.com/WangLY136/MUG](https://github.com/WangLY136/MUG)  
**Area**: Audio-Visual Understanding
**Keywords**: Audio-visual video parsing, Mamba, pseudo labels, data augmentation, weakly supervised learning

## TL;DR

This paper proposes the MUG framework, which simultaneously improves segment-level and event-level prediction in weakly supervised audio-visual video parsing (AVVP) through a pseudo label-augmented cross-modal random combination data augmentation strategy and an audio-visual Mamba network.

## Background & Motivation

The AVVP task aims to predict modality-specific events (visual/auditory/audio-visual) and localize their temporal boundaries within each segment of a video under weak supervision (video-level labels only). Existing methods face three main challenges:

**Difficulty of data augmentation**: Due to weak supervision constraints, the absence of single-modality label information makes it difficult for models to learn a large variety of segment-level event combinations.

**Cross-modal noise**: Events in the two modalities may be entirely unrelated, and naive cross-modal interaction introduces noise interference.

**Long-sequence modeling**: The length of visual input token sequences approaches the threshold of ViT-S (approximately 2000), making Transformer-based processing of long sequences inefficient, while vanilla Mamba exhibits limitations in single-frame image recognition.

## Method

### Overall Architecture

MUG consists of two components: (1) Cross-Modal Random Combination data augmentation (CMRC); and (2) the AV-Mamba network, comprising four modules — MBA, AMF, MFE, and PLSIM. Input audio-visual features are processed through Mamba-based attention, followed by adaptive fusion and feature enhancement, with textual semantic information introduced to constrain predictions.

### Key Designs

1. **Cross-Modal Random Combination (CMRC)**: The core idea is to leverage refined pseudo labels to enable data augmentation for the AVVP task. Based on pseudo labels extracted by VALOR, empty labels are manually annotated (videos rarely lack visual content), and visual and audio tracks from two different videos are randomly combined to generate new samples. The new label is the union of the visual and auditory pseudo labels. Data is generated across five batches (1,585–12,096 samples) following the actual distribution; experiments show Batch 4 (1× original data volume) achieves the optimal trade-off.

2. **Mamba-Based Attention (MBA)**: Inspired by CBAM but implemented with Mamba. Global max pooling and average pooling are applied to each segment feature, which are then fed into a shared Mamba block to obtain channel attention weights $W_t^m$ and spatial attention weights $S_t^m$; enhanced features $\hat{f}_t^m$ are obtained via element-wise multiplication. The causal modeling capability of Mamba facilitates the capture of causal dependencies across multiple frames.

3. **Adaptive Mamba Fusion (AMF)**: Forward/backward SSM scanning and dynamic scanning are applied independently to visual and audio features. The key innovation is that the two modalities **share the state transition matrix B** (governing hidden state evolution) while retaining independent input transition matrix A and output transition matrix C, thereby reducing parameters while preserving single-modality independence and capturing cross-modal similarity. After fusion, a gating strategy yields $f_m^{AMF}$ and the mixed feature $f_{mix}^{AMF}$.

4. **Mamba Feature Enhancement (MFE)**: Receives three streams of input — visual, audio, and mixed. Average pooling is first applied to reduce the dimensionality of both modality features, then element-wise multiplication is performed on the two modality features at the same time step, and a channel enhancement vector is generated via Sigmoid. This amplifies cross-modally similar features while suppressing dissimilar ones.

5. **Pseudo Label Semantic Interaction Module (PLSIM)**: Introduces the text modality to eliminate cross-modal noise. Semantic features are extracted from event categories corresponding to pseudo labels using CLIP/CLAP text encoders, then mapped via multiple MLPs to scaling factors $\gamma$ and bias factors $\rho$, which are adaptively fused with the audio-visual features: $F_{audio} = f_a^{MFE} \odot \gamma_{a1} + \gamma_{a2} + f_a^{MFE}$.

### Loss & Training

- Inherits the multi-modal multiple instance learning loss from the HAN model
- Pseudo labels provide fine-grained supervision
- AdamW optimizer, batch size 64, learning rate $3 \times 10^{-4}$, trained for 20 epochs
- Training conducted on an NVIDIA RTX A6000 GPU

## Key Experimental Results

### Main Results

F1-score comparison with state-of-the-art methods on the LLP dataset:

| Method | Seg-A | Seg-V | Seg-AV | Seg-Type | Seg-Event | Evt-A | Evt-V | Evt-AV | Evt-Type | Evt-Event |
|--------|-------|-------|--------|----------|-----------|-------|-------|--------|----------|-----------|
| HAN | 60.1 | 52.9 | 48.9 | 54.0 | 55.4 | 51.3 | 48.9 | 43.0 | 47.7 | 48.0 |
| VALOR | 61.8 | 65.9 | 58.4 | 62.0 | 61.5 | 55.4 | 62.6 | 52.2 | 56.7 | 54.2 |
| CoLeaF | 64.2 | 64.4 | 59.3 | 62.6 | 62.5 | 57.6 | 63.2 | 54.2 | 57.9 | 55.6 |
| **MUG** | **65.4** | **66.5** | **59.9** | **63.9** | **64.7** | **59.5** | **63.9** | **55.3** | **59.6** | **57.7** |

MUG achieves the best performance across all metrics. Compared to CoLeaF, it improves segment-level visual F1 by +2.1%, segment-level audio F1 by +1.2%, segment-level Event@AV by +2.2%, and event-level Event@AV by +2.1%.

### Ablation Study

Module-wise ablation (Segment-level Event@AV):

| Configuration | Seg-A | Seg-V | Seg-Event@AV |
|---------------|-------|-------|--------------|
| MUG (full) | 65.4 | 66.5 | 64.7 |
| w/o CMRC | 62.7 | 65.2 | 62.2 |
| w/o MBA | 64.5 | 66.1 | 63.9 |
| w/o AMF | 64.1 | 66.3 | 63.4 |
| w/o MFE | 63.8 | 64.6 | 62.8 |
| w/o PLSIM | 64.8 | 66.5 | 64.0 |

CMRC contributes the most (Event@AV drops 2.5% upon removal), followed by MFE.

Comparison with Transformer/CNN architectures: MUG (7.6M parameters) vs. Transformer (19.3M) vs. CNN (6.5M) — MUG achieves the best performance with significantly fewer parameters than the Transformer.

### Key Findings

- CMRC data augmentation reaches its optimum at Batch 4 (1× original data volume); generating excessive data introduces noise and overfitting.
- CMRC generalizes to multiple baseline models (consistent improvements are observed across HAN, MGN, and JoMoLD).
- PLSIM degrades or even harms performance when the augmentation volume is too large (12,000+ samples).
- The Mamba architecture demonstrates substantially superior parameter efficiency compared to the Transformer.

## Highlights & Insights

- **Innovation in data augmentation**: This work is the first to design a cross-modal data augmentation strategy for the AVVP task, cleverly exploiting single-modality pseudo label information to enable free combination of visual and audio tracks.
- **Elegant parameter sharing strategy**: In AMF, only the state transition matrix B is shared while A and C remain independent, striking a balance between cross-modal information sharing and single-modality independence.
- **Complementary Mamba-Transformer hybrid architecture**: Mamba handles temporal causal dependencies while HAN's Transformer captures spatial global dependencies, achieving a mutually complementary design.

## Limitations & Future Work

- Relies on the LLP dataset (only 25 categories with considerable annotation noise), which is relatively small in scale.
- Pseudo label quality remains imperfect, and the manual annotation of empty labels has limited scalability.
- Larger-scale pretraining or multi-dataset joint training has not been explored.
- The implementation details of the dynamic scanning branch are insufficiently described.

## Related Work & Insights

- The pseudo label strategy from VALOR provides the foundation for data augmentation in this work.
- Vision Mamba and VMamba demonstrate the potential of Mamba for visual tasks.
- The cross-modal parameter sharing paradigm introduced in this framework can be extended to other multimodal tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Both the data augmentation strategy and the Mamba fusion architecture are innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive ablation study with multi-baseline validation of CMRC effectiveness.
- **Writing Quality**: ⭐⭐⭐ Structure is clear, but certain implementation details are insufficiently described.
- **Value**: ⭐⭐⭐⭐ Introduces an effective new data augmentation paradigm for the AVVP task.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] UWAV: Uncertainty-Weighted Weakly-Supervised Audio-Visual Video Parsing](../../CVPR2025/audio_speech/uwav_uncertainty-weighted_weakly-supervised_audio-visual_video_parsing.md)
- [\[ECCV 2024\] CoLeaF: A Contrastive-Collaborative Learning Framework for Weakly Supervised Audio-Visual Video Parsing](../../ECCV2024/audio_speech/coleaf_a_contrastive-collaborative_learning_framework_for_weakly_supervised_audi.md)
- [\[ECCV 2024\] Label-Anticipated Event Disentanglement for Audio-Visual Video Parsing](../../ECCV2024/audio_speech/label-anticipated_event_disentanglement_for_audio-visual_video_parsing.md)
- [\[ICCV 2025\] VGGSounder: Audio-Visual Evaluations for Foundation Models](vggsounder_audio-visual_evaluations_for_foundation_models.md)
- [\[ICCV 2025\] Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations](zero-avsr_zero-shot_audio-visual_speech_recognition_with_llms_by_learning_langua.md)

</div>

<!-- RELATED:END -->
