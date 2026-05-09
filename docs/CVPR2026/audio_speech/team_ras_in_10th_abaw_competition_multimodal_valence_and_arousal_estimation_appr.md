---
title: >-
  [Paper Note] Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach
description: >-
  [CVPR 2026][Audio & Speech][Affective Computing] This paper proposes a multimodal approach combining facial visual features, VLM-based behavioral description embeddings, and audio features for continuous valence-arousal (VA) estimation. Two fusion strategies—DCMMOE and RAAV—are explored, achieving competitive results on the Aff-Wild2 dataset.
tags:
  - CVPR 2026
  - "Audio & Speech"
  - Affective Computing
  - Valence-Arousal Estimation
  - Multimodal Fusion
  - Mixture of Experts
  - VLM Behavioral Description
date: 2026-05-08
content_hash: 4ec7aebd8af3ed62
---

# Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach

**Conference**: CVPR 2026
**arXiv**: [2603.13056](https://arxiv.org/abs/2603.13056)
**Code**: [GitHub](https://github.com/SMIL-SPCRAS/CVPRW-26)
**Area**: Audio & Speech
**Keywords**: Affective Computing, Valence-Arousal Estimation, Multimodal Fusion, Mixture of Experts, VLM Behavioral Description

## TL;DR

This paper proposes a multimodal approach combining facial visual features, VLM-based behavioral description embeddings, and audio features for continuous valence-arousal (VA) estimation. Two fusion strategies—DCMMOE and RAAV—are explored, achieving competitive results on the Aff-Wild2 dataset.

## Background & Motivation

**Background**: Continuous emotion recognition under in-the-wild conditions is an important research problem in artificial intelligence. Valence reflects the pleasantness of an emotional state, while arousal reflects its intensity; together they constitute a continuous dimensional representation of affect. The ABAW Challenge series is a benchmark competition in this field, now in its 10th edition.

**Limitations of Prior Work**: Existing methods primarily rely on fusing facial visual and audio features, with insufficient utilization of behavioral contextual information such as body pose, gestures, and scene context. Although vision-language models (VLMs) have demonstrated strong performance across various visual tasks, they have not been fully exploited for continuous VA estimation.

**Key Challenge**: Single-modality information is insufficient to accurately capture complex emotional expressions; existing multimodal fusion strategies inadequately model asymmetric cross-modal interactions.

**Goal**: To leverage VLMs for extracting behavioral description embeddings and to design effective multimodal fusion strategies that integrate facial, audio, and behavioral information for continuous VA estimation.

**Key Insight**: Qwen3-VL-4B-Instruct is introduced to extract behavioral semantic embeddings from video, combined with GRADA facial features and WavLM audio features, and two distinct fusion schemes are explored.

**Core Idea**: VLM-derived behavioral description embeddings serve as a third modality, integrated via directed cross-modal mixture-of-experts fusion and a reliability-aware audio-visual fusion strategy for VA estimation.

## Method

### Overall Architecture

Three modality branches independently extract features and are subsequently integrated through fusion strategies:
- **Facial Branch**: YOLO face detection → GRADA (EfficientNet-B1) extracts 256-dimensional frame-level affective embeddings → Transformer-based temporal regression
- **Behavioral Description Branch**: Qwen3-VL-4B-Instruct processes 16-frame video segments with emotion-oriented text prompts → segment-level embeddings → Mamba temporal modeling
- **Audio Branch**: WavLM-Large processes 4-second audio segments → attentive statistics pooling → cross-modal lip dynamics filtering

### Key Designs

1. **GRADA Facial Feature Extractor**: Based on EfficientNet-B1, pre-trained in a multi-task fashion on 10 public affective datasets (expression classification + VA regression), using discriminative learning rates and progressive unfreezing. With only 7.9M parameters, it balances generalization and efficiency. The 256-dimensional embedding layer serves as frame-level features.

2. **Qwen3-VL Behavioral Description Model**: A multimodal VLM processes video frames alongside emotion-oriented prompts (attending to facial expressions, head movements, gestures, posture, scene context, etc.), and the final valid token representation from the last hidden layer is extracted as the segment-level behavioral embedding. Two extraction modes are evaluated—visual-only and multimodal—with experiments showing the multimodal mode (CCC=0.5385) significantly outperforms the visual-only mode (CCC=0.4007).

3. **Cross-Modal Audio Filtering**: Mouth-opening dynamics extracted via MediaPipe approximate speech presence detection, filtering out audio segments that contain no speech to reduce the impact of noise and unreliable segments. Only the top 4 Transformer layers of WavLM-Large are fine-tuned.

### Loss & Training

- A CCC-based composite loss is used; some experiments incorporate a MAE term to improve training stability.
- Different modalities employ separate hyperparameter searches: facial model with $L=400$, $S=150$; Mamba model with window length 16 and stride 8.
- The audio model uses a backbone learning rate of $5 \times 10^{-6}$ and a head learning rate of $2 \times 10^{-4}$, following a differential learning rate strategy.

## Key Experimental Results

### Main Results

CCC results of various model configurations on Aff-Wild2:

| Model Configuration | Valence CCC | Arousal CCC | Mean CCC (dev) | Mean CCC (test) |
|---|---|---|---|---|
| Facial GRADA + Transformer | 0.5869 | 0.6508 | 0.6189 | 0.54 |
| Visual Qwen3 + Mamba | 0.2499 | 0.5515 | 0.4007 | — |
| Multimodal Qwen3 + Mamba | 0.4290 | 0.6480 | 0.5385 | — |
| Audio WavLM | 0.3415 | 0.4636 | 0.4025 | — |
| Facial + Audio DCMMOE | 0.6252 | 0.6671 | 0.6461 | 0.58 |
| Facial + Multimodal Qwen3 + Audio DCMMOE | 0.6100 | 0.6875 | 0.6487 | 0.61 |
| Facial + Multimodal Qwen3 + Audio RAAV | **0.6078** | **0.7073** | **0.6576** | **0.62** |

### Ablation Study

Comparison of Qwen3 embedding extraction modes:

| Extraction Mode | Valence CCC | Arousal CCC | Mean CCC |
|---|---|---|---|
| Visual-only | 0.2499 | 0.5515 | 0.4007 |
| Multimodal | 0.4290 | 0.6480 | 0.5385 |
| **Gain** | +0.1791 | +0.0965 | +0.1378 |

### Key Findings

- **Multimodal fusion consistently outperforms unimodal baselines**: The three-modality RAAV fusion (CCC=0.6576) improves over the best single-modality result (facial model CCC=0.6189) by approximately 6%.
- **Value of VLM behavioral embeddings**: Qwen3 multimodal embeddings substantially outperform visual-only embeddings (+13.8% CCC), validating the advantage of joint vision-language understanding for affective analysis.
- **RAAV outperforms DCMMOE**: RAAV achieves the best results (arousal CCC=0.7073) through its frame-centric fusion design and asymmetric treatment of audio as auxiliary context.
- **Arousal is consistently easier to estimate than valence**: Arousal CCC exceeds valence CCC across all unimodal models.
- **Consistent improvement from development to test set**: Test set scores progress from unimodal 0.54 → two-modality 0.58 → three-modality 0.62.

## Highlights & Insights

- **First introduction of VLM behavioral descriptions for VA estimation**: Qwen3 is used to extract behavioral semantic embeddings as an independent modality, compensating for the inability of traditional facial/audio features to capture high-level behavioral semantics.
- **Directed cross-modal MoE**: DCMMOE explicitly models directed cross-modal interactions (A→B and B→A are handled by distinct experts) with adaptive gating-based weighting.
- **Cross-modal audio filtering**: Visual modality information (lip dynamics) is leveraged to filter unreliable audio segments, representing an elegant cross-modal collaborative preprocessing design.
- **Highly lightweight behavioral modeling**: Mamba replaces Transformer for temporal modeling, offering superior parameter efficiency.

## Limitations & Future Work

- As a competition system paper, the depth of methodological innovation is limited; the contribution is primarily an engineering combination and optimization.
- Extraction of Qwen3 behavioral embeddings incurs high computational cost (4B-parameter model inference), which may constrain practical deployment.
- Cross-modal audio filtering relies on lip dynamics as a proxy for speech presence, which is not fully reliable.
- Validation is conducted solely on the Aff-Wild2 dataset; generalizability remains unknown.
- More advanced end-to-end multimodal fusion architectures are not explored.

## Related Work & Insights

- Compared to methods from previous ABAW editions (e.g., Zhang et al.'s Masked AE + VGGish, Yu et al.'s ResNet + TCN + cross-modal attention), the key distinction of this work lies in the introduction of VLM behavioral embeddings.
- The asymmetric fusion design of RAAV—where vision determines temporal resolution and audio provides window-level evidence—represents a noteworthy modality role allocation strategy.
- Mamba for affective temporal modeling is an emerging direction worthy of further investigation.

## Rating

- **Novelty**: ⭐⭐⭐ — Introducing VLM behavioral embeddings is a meaningful contribution, though the overall approach is a modular combination
- **Experimental Thoroughness**: ⭐⭐⭐ — Ablations are reasonably systematic, but evaluation is limited to a single dataset without cross-dataset validation
- **Writing Quality**: ⭐⭐⭐⭐ — Structure is clear with sufficient technical detail; strong for a competition paper
- **Value**: ⭐⭐⭐ — Serves as a useful reference as a competition solution; the exploration of VLMs for affective computing merits attention

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Solution for 10th Competition on Ambivalence/Hesitancy (AH) Video Recognition Challenge using Divergence-Based Multimodal Fusion](solution_for_10th_competition_on_ambivalencehesitancy_ah_video_recognition_chall.md)
- [\[CVPR 2026\] OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text](omnisonic_towards_universal_and_holistic_audio_generation_from_video_and_text.md)
- [\[CVPR 2026\] Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis](tri-subspaces_disentanglement_for_multimodal_sentiment_analysis.md)
- [\[CVPR 2026\] UniM: A Unified Any-to-Any Interleaved Multimodal Benchmark](unim_a_unified_any-to-any_interleaved_multimodal_benchmark.md)
- [\[CVPR 2026\] ViDscribe: Multimodal AI for Customizing Audio Description and Question Answering in Online Videos](vidscribe_multimodal_ai_for_customizing_audio_description_and_question_answering.md)

</div>

<!-- RELATED:END -->
