---
title: >-
  [Paper Note] Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach
description: >-
  [CVPR2025][Audio & Speech][valence-arousal estimation] This work proposes a continuous emotion estimation method that integrates three modalities: facial expressions (GRADA+Transformer), behavioral descriptions (Qwen3-VL+Mamba), and audio (WavLM). By employing two fusion strategies—Directed Cross-Modal MoE and Reliability-Aware Audio-Visual—the approach achieves a CCC of 0.6576 (dev) / 0.62 (test) on the Aff-Wild2 dataset.
tags:
  - "CVPR2025"
  - "Audio & Speech"
  - "valence-arousal estimation"
  - "multimodal fusion"
  - "mixture of experts"
  - "affective computing"
  - "ABAW"
date: 2026-05-08
content_hash: c26a7895132aea78
---

# Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach

**Conference**: CVPR2025  
**arXiv**: [2603.13056](https://arxiv.org/abs/2603.13056)  
**Code**: [GitHub](https://github.com/SMIL-SPCRAS/CVPRW-26)  
**Area**: Audio and Speech  
**Keywords**: valence-arousal estimation, multimodal fusion, mixture of experts, affective computing, ABAW

## TL;DR

This work proposes a continuous emotion estimation method that integrates three modalities: facial expressions (GRADA+Transformer), behavioral descriptions (Qwen3-VL+Mamba), and audio (WavLM). By employing two fusion strategies—Directed Cross-Modal MoE and Reliability-Aware Audio-Visual—the approach achieves a CCC of 0.6576 (dev) / 0.62 (test) on the Aff-Wild2 dataset.

## Background & Motivation

**Continuous Emotion Estimation Challenges**: Continuous valence-arousal estimation in in-the-wild conditions faces difficulties such as variations in appearance, head poses, illumination, occlusions, and individual differences.

**Visual Modality Dominance**: Existing methods in the ABAW competition mainly rely on visual features (ResNet, ViT, EfficientNet), but unimodal information remains insufficient.

**Unexplored VLM Potential**: The application of multimodal Vision-Language Models (VLMs) in sentiment analysis remains to be fully explored, despite their capability to capture contextual and situational emotional cues.

**Unreliability of Audio Modality**: Since Aff-Wild2 is collected with a focus on videos, the audio is usually noisy and unreliable, containing a large number of non-speech segments.

**Modality Fusion Complexity**: Different modalities exhibit varying reliability at different moments, requiring adaptive weighting instead of simple concatenation.

**New Dimension of Behavioral Descriptions**: Utilizing VLMs to extract behavioral-level semantic information (such as facial expressions, head movements, hand gestures, and body poses) as a supplement to emotion estimation.

## Method

### Overall Architecture

Tri-modal pipeline: Face (frame-level features + temporal regression), Behavior (VLM segment-level embedding + temporal modeling), and Audio (speech features + chunk pooling) are processed, yielding valence/arousal predictions via one of the two fusion strategies.

### Facial Model

- **Feature Extraction**: YOLO face detection $\rightarrow$ GRADA encoder (based on EfficientNet-B1, multi-task fine-tuned on 10 emotion datasets, outputting 256-dimensional embeddings).
- **Temporal Modeling**: Transformer-based sequence regression model: projection layer $\rightarrow$ multi-layer Transformer $\rightarrow$ regression head (FCL + LN + GELU + Dropout + FCL).
- **Sliding Window Processing**: Overlapping temporal windows ($L=400, S=150$), using the nearest frame for padding when no face is detected.

### Behavioral Description Model

- **Feature Encoding**: Qwen3-VL-4B-Instruct processes segments of 16 uniformly sampled frames combined with an emotion-oriented text prompt, extracting the last token representation from the final hidden layer.
- **Two Settings**: Visual-only embeddings vs. multimodal (video+text) embeddings, with the latter being significantly superior.
- **Temporal Modeling**: Segment-level embeddings are processed by stacked Mamba blocks to model temporal dynamics (4–12 layers, hidden dimension of 128–256, state size of 8).
- **Frame-level Expansion**: Segment-level predictions are expanded to the corresponding frame intervals, with overlapping segments averaged.

### Audio Model

- **Preprocessing**: 4-second segments with 2-second overlaps, mono, 16kHz.
- **Cross-Modal Filtering**: Filters non-speech segments based on MediaPipe mouth opening/closing detection (temporal smoothing + opening duration threshold).
- **Feature Extraction**: WavLM-Large (pretrained on MSP-Podcast), fine-tuning only the top 4 Transformer layers.
- **Chunk Pooling**: Hidden representations are divided into 4 temporal chunks, each aggregated using attention-statistics pooling (weighted mean + weighted standard deviation).

### Fusion Strategies

**DCMMOE (Directed Cross-Modal Mixture-of-Experts)**:
- All ordered modality pairs form cross-attention experts $|\mathcal{E}| = M(M-1)$ to explicitly model asymmetric cross-modal interactions.
- A learnable gating network adaptively assigns expert weights to each frame.
- Uses a 5-layer cross-attention Transformer with 16 heads.

**RAAV (Reliability-Aware Audio-Visual)**:
- Facial and behavioral features are fused at the frame level via masked reliability-aware gating (learnable reliability gating + modality priors).
- Audio provides auxiliary context through learnable bottleneck latent representations (cross-attention).
- Asymmetric design: visual modalities determine the temporal resolution, while audio provides window-level supplementation.

### Loss & Training

A hybrid loss based on CCC, optioned with an MAE term to enhance training stability. Valence and arousal loss weights can be tuned independently.

## Key Experimental Results

| ID | Model | Valence | Arousal | Avg (dev) | Avg (test) |
|----|------|---------|---------|-----------|------------|
| 1 | Face: GRADA+Transformer | 0.587 | 0.651 | 0.619 | 0.54 |
| 3 | Behavior: Qwen3-multimodal+Mamba | 0.429 | 0.648 | 0.539 | – |
| 4 | Audio: WavLM+Pooling | 0.342 | 0.464 | 0.403 | – |
| 5 | Face+Audio (DCMMOE) | 0.625 | 0.667 | 0.646 | 0.58 |
| 7 | Face+Behavior+Audio (DCMMOE) | 0.610 | 0.688 | 0.649 | 0.61 |
| **8** | **Face+Behavior+Audio (RAAV)** | **0.608** | **0.707** | **0.658** | **0.62** |

**Key Findings**:
- Multimodal fusion consistently outperforms single modalities (test CCC: 0.54 $\rightarrow$ 0.62).
- Qwen3 multimodal embeddings (0.539) significantly outperform visual-only embeddings (0.401), confirming the value of VLM behavioral embeddings.
- Arousal estimation is more reliable than valence across all models.
- The RAAV fusion strategy achieves the highest score of 0.7073 on arousal, with an overall best of 0.6576.
- Cross-modal mouth opening/closing filtering effectively improves the quality of the audio modality.

## Highlights & Insights

1. **Innovative Application of VLM Behavioral Embeddings**: This work is the first to utilize Qwen3-VL to extract behavioral-level semantic embeddings in the ABAW VA task, filling the gap of applying VLMs to continuous emotion estimation.
2. **Two Complementary Fusion Strategies**: DCMMOE models interactions among all ordered modality pairs, while RAAV adopts an asymmetric frame-centric design, each offering unique advantages.
3. **Cross-Modal Audio Filtering**: Relying on visual cues (mouth opening and closing) to filter unreliable audio segments, which specifically adapts to the characteristics of video-oriented datasets.
4. **Systematic Modality Analysis**: Clearly demonstrates the individual contribution of each modality alongside the benefits gained from their fusion.

## Limitations & Future Work

1. As a challenge paper (CVPR Workshop), the approach is highly engineering-oriented with limited theoretical innovation.
2. Qwen3 behavioral embeddings require manually designed prompts; their generalization and robustness remain to be validated.
3. Mouth opening/closing filtering is a crude approximation of voice presence, which can lead to misses or false positives.
4. Validation is conducted purely on the single Aff-Wild2 dataset; cross-dataset generalization has not been evaluated.
5. Face detection requires manual identity correction, limiting scalability.

## Related Work & Insights

- **Unimodal Methods**: ResNet baseline (ABAW), MobileViT, DDAMFN, MAE+CLIP+TCN (Zhou et al.)
- **Multimodal Fusion**: PDEM audio features + visual backbone (Dresvyanskiy et al.), MAE+VGGish ensemble (Zhang et al.), GR-JCA gating (Praveen et al.), TAGF temporal gating (Lee et al.)
- **Challenge Winners**: ResNet+VGGish+LogMel+TCN+cross-modal attention (Yu et al., 8th ABAW)
- This work is the first to use VLM behavioral embeddings in the ABAW VA task.

## Rating

- Novelty: ⭐⭐⭐⭐ — VLM behavioral embedding is a highlight, though the overall solution is a module-stacking engineering approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Thorough ablation studies across modalities, but verified on only a single dataset with no horizontal comparison against other challenge teams.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, but features numerous mathematical notations, and some details are wordy.
- Value: ⭐⭐⭐⭐ — Demonstrates a feasible path for VLM integration to the affective computing community, showing competitive challenge performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Sortformer: A Novel Approach for Permutation-Resolved Speaker Supervision in Speech-to-Text Systems](../../ICML2025/audio_speech/sortformer_a_novel_approach_for_permutation-resolved_speaker_supervision_in_spee.md)
- [\[ICML 2025\] BinauralFlow: A Causal and Streamable Approach for High-Quality Binaural Speech Synthesis with Flow Matching Models](../../ICML2025/audio_speech/binauralflow_a_causal_and_streamable_approach_for_high-quality_binaural_speech_s.md)
- [\[CVPR 2025\] Contextual AD Narration with Interleaved Multimodal Sequence](contextual_ad_narration_with_interleaved_multimodal_sequence.md)
- [\[CVPR 2025\] MultiFoley: Video-Guided Foley Sound Generation with Multimodal Controls](video-guided_foley_sound_generation_with_multimodal_controls.md)
- [\[CVPR 2025\] HOP: Heterogeneous Topology-based Multimodal Entanglement for Co-Speech Gesture Generation](hop_heterogeneous_topology-based_multimodal_entanglement_for_co-speech_gesture_g.md)

</div>

<!-- RELATED:END -->
