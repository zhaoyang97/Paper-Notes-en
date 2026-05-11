---
title: >-
  [Paper Note] Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach
description: >-
  [CVPR 2026 (ABAW Workshop)][Audio & Speech][Valence-Arousal Estimation] This work is the first to incorporate behavioral description embeddings extracted by a VLM (Qwen3-VL-4B-Instruct) as an independent third modality…
tags:
  - "CVPR 2026 (ABAW Workshop)"
  - "Audio & Speech"
  - "Valence-Arousal Estimation"
  - "Multimodal Fusion"
  - "VLM Behavioral Description"
  - "Mamba"
  - "ABAW Competition"
date: 2026-05-08
content_hash: b6bd2e80d92e8bca
---

# Team RAS in 10th ABAW Competition: Multimodal Valence and Arousal Estimation Approach

**Conference**: CVPR 2026 (ABAW Workshop)
**arXiv**: [2603.13056](https://arxiv.org/abs/2603.13056)
**Code**: [GitHub](https://github.com/SMIL-SPCRAS/CVPRW-26)
**Area**: Audio & Speech
**Keywords**: Valence-Arousal Estimation, Multimodal Fusion, VLM Behavioral Description, Mamba, ABAW Competition

## TL;DR

This work is the first to incorporate behavioral description embeddings extracted by a VLM (Qwen3-VL-4B-Instruct) as an independent third modality, combining them with GRADA facial encodings and WavLM audio features via two fusion strategies—DCMMOE and RAAV—achieving a continuous VA estimation CCC of 0.658 (dev) / 0.62 (test) on Aff-Wild2, demonstrating the value of VLM behavioral semantics for continuous emotion recognition.

## Background & Motivation

**Background**: Continuous valence-arousal (VA) estimation under in-the-wild conditions remains challenging due to large appearance variation, diverse head poses, frequent occlusions, and noisy audio. The ABAW Challenge is the most authoritative benchmark in this domain; prior state-of-the-art methods predominantly employ face + audio + cross-attention fusion pipelines.

**Limitations of Prior Work**: Existing multimodal methods rely solely on conventional feature extractors (EfficientNet for vision, VGGish/WavLM for audio), and are unable to capture rich behavior-level semantics—such as trends in facial expression change, gesture meaning, or the relationship between body posture and context. While VLMs have demonstrated strong contextual understanding in video comprehension, they have not yet been applied to continuous VA estimation.

**Key Challenge**: Frame-level visual features encode only appearance and lack understanding of behavioral semantics and situational context. VLMs can provide such high-level semantics, but their outputs are segment-level rather than frame-level, and differ substantially from traditional modalities in temporal resolution and information density, making effective integration a key challenge.

**Goal**: (1) How can segment-level VLM outputs be aligned with frame-level visual/audio features? (2) How can heavily noisy in-the-wild audio be reliably exploited? (3) How can three modalities with vastly different temporal resolutions and information densities be adaptively fused?

**Key Insight**: Qwen3-VL processes video segments with emotion-oriented prompts to extract behavior-level embeddings → Mamba models segment-level temporal dynamics → frame-level expansion; mouth open/close detection filters audio reliability; two fusion strategies, DCMMOE and RAAV, are designed to handle modality asymmetry.

**Core Idea**: VLM behavioral description embeddings serve as a third modality, combined with two asymmetric fusion strategies (DCMMOE/RAAV), injecting VLM behavioral understanding into continuous emotion estimation.

## Method

### Overall Architecture

Three independent unimodal encoders (GRADA face + Qwen3-VL behavioral description + WavLM audio) feed into one of two fusion strategies (DCMMOE or RAAV) for frame-level VA prediction. Each modality has its own temporal modeler (Transformer / Mamba / block-level pooling), with projection to a shared latent space prior to fusion.

### Key Designs

1. **GRADA Facial Model + Transformer Temporal Regression**

    - **Function**: Extracts frame-level affective representations and models short- to mid-term temporal dynamics.
    - **Mechanism**: EfficientNet-B1 is multi-task fine-tuned on 10 emotion datasets (7.9M parameters), producing 256-dimensional frame-level affective embeddings. YOLO face detection combined with manual identity annotation ensures single-target tracking. A Transformer regression model operates on sliding windows of length $L=400$ with stride $S=150$, comprising a projection block (FC+LN+Dropout), multi-layer Transformer ($N=5, H=16$), and a regression head (FC+LN+GELU+Dropout+FC).
    - **Design Motivation**: EfficientNet-B1 demonstrates the best generalization/efficiency trade-off across multi-architecture evaluations. Transformer sliding windows maintain temporal continuity while increasing training samples.

2. **Qwen3-VL Behavioral Description Model + Mamba Temporal Encoding**

    - **Function**: Leverages VLM to capture behavior-level semantics that conventional feature extractors cannot encode.
    - **Mechanism**: Qwen3-VL-4B-Instruct processes 16-frame video segments with an emotion-oriented prompt (directing attention to facial expressions, head movements, gestures, posture, and scene), extracting last-hidden-layer tokens as segment-level embeddings $e \in \mathbb{R}^d$. Two settings are evaluated: vision-only embeddings (visual tokens only) and multimodal embeddings (video + text jointly). Segment-level embeddings are passed through a Mamba encoder for temporal modeling (vision: 4 layers, hidden=128, state=8, kernel=3; multimodal: 12 layers, hidden=256, state=8, kernel=5); frame-level predictions are obtained via segment-to-frame expansion with overlap averaging.
    - **Design Motivation**: Multimodal embeddings (CCC 0.539) substantially outperform vision-only (0.401), demonstrating that prompt-guided textual context is critical for behavioral understanding.

3. **WavLM Audio Model + Cross-Modal Reliability Filtering**

    - **Function**: Extracts affective cues from audio while filtering heavily noisy non-speech segments.
    - **Mechanism**: Four-second segments with 2-second overlap are processed; MediaPipe detects mouth open/close states for cross-modal filtering—only segments where the duration of mouth opening and annotation coverage exceed thresholds are retained. WavLM-Large top 4 layers are fine-tuned (pre-trained on MSP-Podcast). Each 4-second segment is divided into 4 temporal blocks; each block is aggregated via attentive statistical pooling (weighted mean + weighted standard deviation), and a regression head outputs VA.
    - **Design Motivation**: Aff-Wild2 in-the-wild audio is extremely noisy; mouth open/close detection serves as a simple and effective proxy for speech presence.

4. **DCMMOE Fusion Strategy**

    - **Function**: Models all directed pairwise modality interactions with adaptive weighted fusion.
    - **Mechanism**: Each modality is projected into a shared $d_h$-dimensional space. All ordered pairs $(q,k)$ form cross-attention experts ($|\mathcal{E}|=M(M-1)$ experts), where each expert applies $N$ layers of $H$-head cross-attention (query = modality $q$, key/value = modality $k$). A gating network computes expert weights from the averaged multimodal state: $\mathbf{g}_l = \mathbf{W}_g \bar{\mathbf{h}}_l + \mathbf{b}_g$, with fusion $\mathbf{z}_l = \sum_{(q,k)} \text{softmax}(\mathbf{g}_{(q,k),l}) \mathbf{Z}_{(q,k),l}$.
    - **Design Motivation**: Explicitly models directed cross-modal interactions (asymmetric query and context roles) with data-dependent expert weighting.

5. **RAAV Fusion Strategy**

    - **Function**: Frame-centric asymmetric fusion—visual modalities determine temporal resolution while audio provides supplementary context.
    - **Mechanism**: Facial and behavioral features are fused at each frame via masked reliability-aware gating: $\mathbf{z}_{\text{vis},l} = \sum_m \alpha_l^{(m)} \mathbf{h}_l^{(m)}$, where $\alpha$ is determined by learned scoring functions and modality priors. The fused visual sequence then extracts context from audio $\mathbf{B}_a$ via bottleneck cross-attention: $\mathbf{Z}_0 = \text{LN}(\mathbf{Z}_\text{vis} + \text{CrossAttn}(\mathbf{Z}_\text{vis}, \mathbf{B}_a, \mathbf{B}_a))$.
    - **Design Motivation**: Reflects the task-inherent property of VA estimation where vision is dominant and audio is supplementary.

### Loss & Training

- Hybrid CCC loss with optional MAE term; valence and arousal can be weighted independently.
- AdamW, lr=1e-4, batch=8, ReduceLROnPlateau.
- Facial backbone lr=5e-6, head lr=2e-4; WavLM top 4 layers fine-tuned; 50 epochs.

## Key Experimental Results

### Main Results

| ID | Configuration | Valence CCC | Arousal CCC | Avg CCC | Test Avg |
|----|---------------|-------------|-------------|---------|----------|
| 1 | Face GRADA+Transformer | 0.587 | 0.651 | 0.619 | 0.54 |
| 2 | Behavior Qwen3 Vision+Mamba | 0.250 | 0.552 | 0.401 | - |
| 3 | Behavior Qwen3 Multimodal+Mamba | 0.429 | 0.648 | **0.539** | - |
| 4 | Audio WavLM+Block Pooling | 0.342 | 0.464 | 0.403 | - |
| 5 | Face+Audio DCMMOE | 0.625 | 0.667 | 0.646 | 0.58 |
| 7 | Face+Behavior(Multi)+Audio DCMMOE | 0.610 | 0.688 | 0.649 | 0.61 |
| 8 | Face+Behavior(Multi)+Audio **RAAV** | 0.608 | **0.707** | **0.658** | **0.62** |

### Ablation Study

| Comparison | Avg CCC | Difference | Note |
|------------|---------|------------|------|
| Qwen3 Multimodal vs. Vision-Only | 0.539 vs 0.401 | +0.138 | Prompt-guided textual context is critical |
| Trimodal vs. Bimodal (Face+Audio) | 0.649 vs 0.646 | +0.003 | VLM modality provides consistent but marginal gain |
| RAAV vs. DCMMOE (Trimodal) | 0.658 vs 0.649 | +0.009 | RAAV has clear advantage on arousal |
| Fusion vs. Best Unimodal | 0.658 vs 0.619 | +0.039 | Fusion consistently outperforms unimodal |

### Key Findings

- Qwen3 multimodal embeddings (0.539) substantially outperform vision-only (0.401) by 0.138 CCC—direct regression from VLM visual-only features performs poorly; textual context guidance is essential.
- Trimodal fusion consistently outperforms bimodal and unimodal baselines, though the marginal gain from the VLM modality is only +0.003, likely limited by temporal resolution loss in segment-to-frame expansion.
- RAAV is particularly strong on arousal (0.707 vs. 0.688), while DCMMOE is slightly better on valence (0.625 vs. 0.608), reflecting different dimensional preferences of the two fusion strategies.
- The gap from dev (0.658) to test (0.62) of 0.038 suggests generalization remains an open challenge.

## Highlights & Insights

- **First use of VLM behavioral descriptions as an independent modality** for continuous VA estimation—the large gap between multimodal and vision-only embeddings (0.539 vs. 0.401) clearly demonstrates the value of prompt-guided behavioral semantics. This approach is generalizable to action recognition, social signal processing, and related tasks.
- RAAV's asymmetric design (vision determines temporal resolution, audio provides supplementary context) appropriately reflects the task characteristics of VA estimation.
- Mouth open/close detection as an audio reliability filter is a simple yet effective cross-modal strategy—using visual signals to pre-screen audio quality at near-zero computational cost.
- DCMMOE's $M(M-1)$ directed pairwise experts with adaptive gating model asymmetric cross-modal interactions more precisely than naive concatenation.

## Limitations & Future Work

- Segment-level Qwen3 embeddings expanded to frame level incur a hard temporal resolution loss; token-level embeddings may improve this.
- High inference cost of VLM (Qwen3-VL-4B) makes real-time deployment impractical.
- Aff-Wild2 contains approximately 3M frames but only 584 subjects; individual differences may dominate results.
- The dev (0.658) to test (0.62) drop indicates insufficient cross-subject generalization.
- The use of VLM outputs directly as frame-level descriptions rather than segment-level embeddings remains unexplored.

## Related Work & Insights

- **vs. Yu et al. (9th ABAW Winner)**: Uses ResNet + VGGish/LogMel + TCN + cross-modal attention. This work adds the VLM behavioral modality and replaces TCN with Mamba, achieving competitive CCC (0.62 test).
- **vs. Praveen et al. (8th ABAW)**: GR-JCA performs bimodal fusion. This work's DCMMOE models all directed pairs with adaptive gating at finer fusion granularity.
- **vs. Lee et al. (9th ABAW)**: Time-aware Gated Fusion. This work's key innovation is the introduction of the VLM behavioral modality.

## Rating

⭐⭐⭐⭐

- **Novelty** ⭐⭐⭐⭐: First application of VLM behavioral descriptions to continuous VA; the multimodal vs. vision-only embedding comparison yields clear insights.
- **Experimental Thoroughness** ⭐⭐⭐⭐: Systematic comparison across 8 configurations, two fusion strategies, and full coverage of uni-/bi-/trimodal settings.
- **Writing Quality** ⭐⭐⭐⭐: Clear structure with complete formalization of fusion strategies.
- **Value** ⭐⭐⭐⭐: Valuable reference for the intersection of affective computing and VLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Solution for 10th Competition on Ambivalence/Hesitancy (AH) Video Recognition Challenge using Divergence-Based Multimodal Fusion](solution_for_10th_competition_on_ambivalencehesitancy_ah_video_recognition_chall.md)
- [\[CVPR 2026\] Echoes Over Time: Unlocking Length Generalization in Video-to-Audio Generation Models](echoes_over_time_unlocking_length_generalization_in_video-to-audio_generation_mo.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [\[CVPR 2026\] Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis](tri-subspaces_disentanglement_for_multimodal_sentiment_analysis.md)
- [\[CVPR 2026\] UniM: A Unified Any-to-Any Interleaved Multimodal Benchmark](unim_a_unified_any-to-any_interleaved_multimodal_benchmark.md)

</div>

<!-- RELATED:END -->
