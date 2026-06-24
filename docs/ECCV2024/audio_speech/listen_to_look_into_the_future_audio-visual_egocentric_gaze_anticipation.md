---
title: >-
  [Paper Note] Listen to Look into the Future: Audio-Visual Egocentric Gaze Anticipation
description: >-
  [ECCV 2024][Audio & Speech][Egocentric Gaze Anticipation] This paper proposes CSTS (Contrastive Spatial-Temporal Separable), an audio-visual fusion method that introduces audio signals to egocentric gaze anticipation for the first time. It models spatial co-occurrence and temporal correlation of audio-visual signals separately through spatial and temporal separable fusion modules, and enhances representations using post-fusion contrastive learning…
tags:
  - "ECCV 2024"
  - "Audio & Speech"
  - "Egocentric Gaze Anticipation"
  - "Audio-Visual Fusion"
  - "Contrastive Learning"
  - "Spatial-Temporal Separable Fusion"
  - "Augmented Reality"
date: 2026-05-08
content_hash: 74ec630b770c1051
---

# Listen to Look into the Future: Audio-Visual Egocentric Gaze Anticipation

**Conference**: ECCV 2024  
**arXiv**: [2305.03907](https://arxiv.org/abs/2305.03907)  
**Code**: [https://github.com/bolinlai/CSTS-EgoGazeAnticipation](https://github.com/bolinlai/CSTS-EgoGazeAnticipation)  
**Area**: Audio & Speech / Egocentric Vision  
**Keywords**: Egocentric Gaze Anticipation, Audio-Visual Fusion, Contrastive Learning, Spatial-Temporal Separable Fusion, Augmented Reality

## TL;DR

This paper proposes CSTS (Contrastive Spatial-Temporal Separable), an audio-visual fusion method that introduces audio signals to egocentric gaze anticipation for the first time. It models spatial co-occurrence and temporal correlation of audio-visual signals separately through spatial and temporal separable fusion modules, and enhances representations using post-fusion contrastive learning, surpassing SOTA on Ego4D and Aria datasets.

## Background & Motivation

**Background**: Egocentric Gaze Anticipation aims to predict the wearer's future gaze targets from egocentric videos, which is a key building block for augmented reality and wearable computing. Existing methods mostly focus on gaze estimation (gaze location in the current frame), while gaze anticipation (future frames) remains under-explored.

**Limitations of Prior Work**: All existing methods rely solely on visual modalities, completely ignoring the influence of audio signals on gaze behavior. However, in daily activities (especially social scenarios), sound is an important stimulus driving gaze shifts—for instance, hearing someone speak and then turning the gaze towards the speaker.

**Key Challenge**: The spatial-temporal relationship between audio and vision in egocentric video is fundamentally different from third-person video: (1) The wearer's reaction to audio stimuli causes dramatic camera movements (head rotation shifts the viewpoint), and the spatial position of the sound source changes over time; (2) There is a natural time delay between the audio stimulus and the gaze response, leading to asynchrony. Existing joint fusion methods (fusing spatial and temporal dimensions simultaneously) cannot handle these characteristics effectively.

**Goal**: How to effectively fuse audio and visual signals in egocentric scenarios to predict the wearer's future gaze targets.

**Key Insight**: Decompose the correlation of audio and vision into the spatial dimension (where the sound source is) and the temporal dimension (how audio drives viewpoint changes and gaze shifts), and model them using two independent modules.

**Core Idea**: Design spatial-temporal separable audio-visual fusion modules to capture intra-frame spatial co-occurrence and cross-frame temporal correlation of audio-visual signals respectively, combined with post-fusion contrastive learning to achieve egocentric gaze anticipation.

## Method

### Overall Architecture

The model consists of the following components:
1. **Visual Encoder** $\phi(x)$: MViT (Multi-scale Vision Transformer) extracts video features, outputting $T \times H \times W$ tokens of $D$ dimensions.
2. **Audio Encoder** $\psi(a)$: A lightweight Transformer processes audio spectrograms, outputting $T \times M$ tokens of $D$ dimensions.
3. **Spatial Fusion Module**: Models the spatial co-occurrence relationship of audio-visual tokens within each frame.
4. **Temporal Fusion Module**: Models cross-frame audio-visual temporal correlation.
5. **Channel Reweighting Merger**: Merges the outputs of the two fusion branches.
6. **Post-Fusion Contrastive Learning**: Applied contrastive loss on the post-fusion representations.
7. **Decoder**: Predicts the probability distribution heatmap of future gaze.

The input is an 8-frame video of a 3-second observation segment + corresponding audio, predicting the gaze location distribution for the future 2 seconds.

### Key Designs

1. **Spatial Fusion Module (Spatial Fusion)**:

    - **Function**: Models the spatial co-occurrence of audio signals and visual regions within each frame, identifying which visual areas are most relevant to the current sound (e.g., sound source location).
    - **Mechanism**: For each timestep $i$, the audio embedding is pooled via convolution to a single token $z_{a,s}^{(i)} \in \mathbb{R}^{1 \times 1 \times D}$, and concatenated with the $N = H \times W$ visual tokens of that frame to form $z_s^{(i)} \in \mathbb{R}^{1 \times (N+1) \times D}$, which is fed into an intra-frame self-attention layer:
    $\sigma(z_s^{(i)}) = \text{Softmax}\left(\frac{Q_s^{(i)} K_s^{(i)T}}{\sqrt{D}}\right) V_s^{(i)}$
   Crucially, an **intra-frame mask** is used to completely block cross-frame connections, computing each frame independently.
    - **Design Motivation**: In egocentric videos, the viewpoint changes drastically due to head motion, and the spatial position of the same object varies significantly across frames. Independent intra-frame fusion avoids confusion of spatial positions across frames.

2. **Temporal Fusion Module (Temporal Fusion)**:

    - **Function**: Models the correlation of audio and visual signals in the temporal dimension, capturing how audio drives viewpoint changes and gaze shifts.
    - **Mechanism**: For each modality at each timestep, all tokens are pooled via convolution into a single token (visual $z_{v,t} \in \mathbb{R}^{T \times 1 \times D}$, audio $z_{a,t} \in \mathbb{R}^{T \times 1 \times D}$), concatenated to form $z_t \in \mathbb{R}^{2T \times 1 \times D}$, and fed into a cross-frame self-attention layer:
    $\pi(z_t) = \text{Softmax}\left(\frac{Q_t K_t^T}{\sqrt{D}}\right) V_t$
   Free interaction is allowed among all $2T$ tokens.
    - **Design Motivation**: There is a temporal delay between audio stimuli and gaze response (e.g., gaze shifts after a specific reaction time after hearing a sound). Cross-frame attention can capture this delay effect and temporal dynamics.

3. **Channel Reweighting Merger + Post-Fusion Contrastive Learning**:

    - **Function**: Merges the spatial and temporal fusion results, and enhances the alignment of audio-visual representations using a contrastive loss.
    - **Mechanism**:
        - **Merger**: The output of spatial fusion is channel-reweighted using temporal fusion weights: $u_v = u_{v,s} \otimes u_{v,t}$ (element-wise multiplication + broadcasting).
        - **Contrastive Learning**: Unlike traditional methods that calculate contrastive loss on raw embeddings, this paper computes it on post-fusion reweighted features: $u_v$ and $u_a$ (reweighted audio representation) are globally average-pooled and projected to a low-dimensional space, using the InfoNCE contrastive loss:
    $\mathcal{L}_{cntr}^{v2a} = -\frac{1}{|\mathcal{B}|}\sum_{i=1}^{|\mathcal{B}|}\log\frac{\exp(w_v^{(i)T} w_a^{(i)} / \mathcal{T})}{\sum_{j \in \mathcal{B}}\exp(w_v^{(i)T} w_a^{(j)} / \mathcal{T})}$
    - **Design Motivation**: Calculating contrastive loss on post-fusion features is more effective than on raw features because post-fusion features already contain spatial-temporal correlation information, and contrastive learning can further reinforce this association.

### Loss & Training

The final loss is a linear combination of two terms:
$$\mathcal{L} = \mathcal{L}_{kld} + \alpha \mathcal{L}_{cntr}$$

- $\mathcal{L}_{kld}$: KL divergence loss, supervising the match between predicted and ground-truth gaze heatmap distributions.
- $\mathcal{L}_{cntr}$: Bidirectional contrastive loss (video-to-audio + audio-to-video).
- The visual encoder uses MViT (pre-trained on Kinetics-400), and the audio encoder is a lightweight Transformer.
- Input video resolution is $256 \times 256$, audio sampling rate is 24kHz, STFT window is 10ms, hop is 5ms, spectrogram size is $256 \times 256$.
- The decoder has skip connections from the video encoder to the decoder.

## Key Experimental Results

### Main Results

| Method | Ego4D F1 | Ego4D Recall | Ego4D Precision | Aria F1 | Aria Recall | Aria Precision |
|------|---------|-------------|----------------|---------|------------|---------------|
| MViT (Vision only) | 37.2 | 54.1 | 28.3 | 57.5 | 62.4 | 53.3 |
| GLC (Prev. Gaze Estimation SOTA) | 37.8 | 52.9 | 29.4 | 58.3 | 65.4 | 52.6 |
| DFG+ (Prev. Gaze Anticipation SOTA) | 37.3 | 52.3 | 29.0 | 57.6 | 65.5 | 51.3 |
| **CSTS (Ours)** | **39.7** | 53.3 | **31.6** | **59.9** | 66.8 | **54.3** |

Gain: +2.4% / +2.3% F1 over DFG+ (gaze anticipation SOTA), and +1.9% / +1.6% F1 over GLC (gaze estimation SOTA).

### Ablation Study

| Configuration | Ego4D F1 | Aria F1 | Description |
|------|---------|---------|------|
| Vision only | 37.2 | 57.5 | Visual-only baseline |
| S-fusion | 38.6 | 58.6 | Spatial fusion only (+1.4 / +1.1) |
| T-fusion | 38.7 | 58.6 | Temporal fusion only (+1.5 / +1.1) |
| STS (S+T) | 39.2 | 59.3 | Spatial-temporal separable fusion (+2.0 / +1.8) |
| **CSTS (STS + Contrastive)** | **39.7** | **59.9** | Add contrastive learning (+2.5 / +2.4) |

Fusion strategy comparison:

| Fusion Method | Ego4D F1 | Aria F1 | Description |
|---------|---------|---------|------|
| Linear | 38.2 | 58.1 | Linear fusion |
| Bilinear | 37.6 | 57.7 | Bilinear fusion |
| Concat. | 38.1 | 58.0 | Concatenation fusion |
| Vanilla SA | 38.5 | 58.0 | Standard joint self-attention |
| **STS (Ours)** | **39.2** | **59.3** | Optimal spatial-temporal separable fusion |

Post-fusion contrastive vs traditional contrastive:

| Method | Ego4D F1 | Aria F1 |
|------|---------|---------|
| STS (Without contrastive) | 39.2 | 59.3 |
| STS + Vanilla Contr (Raw feature contrastive) | 39.0 | 59.1 |
| **STS + Post Contr (Post-fusion contrastive)** | **39.7** | **59.9** |

### Key Findings

- **Audio significantly improves gaze anticipation**: Adding audio improves F1 by +2.5% (Ego4D) and +2.4% (Aria), validating the heavy driving effect of audio on egocentric gaze behavior.
- **Separable fusion outperforms joint fusion**: STS is more effective than all joint fusion strategies (Linear, Bilinear, Concat, Vanilla SA), validating the need to handle spatial and temporal audio-visual correlations separately in egocentric scenarios.
- **Post-fusion contrastive learning is more effective than traditional contrastive learning**: Applying contrastive loss on post-fusion features is superior to doing so on raw embeddings, where traditional contrastive learning even slightly degrades performance.
- **The model consistently outperforms the baseline across all prediction timesteps**: Although the task becomes more difficult as the prediction timestep moves further from the current moment, the model maintains its advantage.
- Spatial fusion visualization shows that the model accurately localizes speaker positions and tracks changes in sound sources.

## Highlights & Insights

- **First to introduce audio for egocentric gaze anticipation**: This is a highly intuitive but long-neglected direction, filling an important research gap.
- **Strong motivation for spatial-temporal separable fusion**: Instead of being a purely engineering-driven design, it stems from the inherent characteristics of egocentric videos (viewpoint shifts, reaction delays) and is backed by strong cognitive science foundations.
- **Post-Fusion Contrastive Loss**: This is a novel contribution. Conducting contrastive learning on post-fusion rather than raw features is more effective because the post-fusion features already contain spatial-temporal correlation, which contrastive loss further strengthens.
- **Convincing visualization analysis**: Gaze spatial association weight maps clearly demonstrate how the model tracks changes in sound sources and speaker positions.

## Limitations & Future Work

- Validated only on social scenarios (Ego4D, Aria); other activity scenarios (e.g., cooking, sports) might display different audio characteristics, calling for further generalization validation.
- A lightweight Transformer is used as the audio encoder; stronger pre-trained audio models (e.g., BEATs, AudioMAE) might bring further improvements.
- The merging method of spatial and temporal fusion (channel reweighting) is relatively simple; more complex interaction mechanisms could be explored.
- The granularity of gaze heatmap prediction is limited to a $256 \times 256$ resolution; high-resolution scenarios may require finer-grained predictions.
- The model does not explicitly model gaze-action associations; incorporating action prediction could further improve gaze anticipation performance.

## Related Work & Insights

- **vs DFG/DFG+**: Prior SOTA in gaze anticipation generates future frames with CNNs before predicting gaze. This paper directly predicts future gaze from current audio-visual features without video generation steps.
- **vs GLC**: Current SOTA in gaze estimation uses global-local correlation modeling. This paper extends it to anticipation tasks and incorporates the audio modality.
- **vs Audio-Visual Saliency Prediction**: Traditional saliency prediction targets fixed-viewpoint third-person videos and uses joint fusion. This paper tackles the unique challenges of egocentric videos (viewpoint shifts, reaction delays) using a separable fusion strategy.

## Rating

- Novelty: ⭐⭐⭐⭐ First to introduce audio for egocentric gaze anticipation. Both spatial-temporal separable fusion and post-fusion contrastive learning are novel designs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on two datasets with highly detailed ablations (module ablations, fusion strategy comparisons, contrastive learning strategies), frame-by-frame analysis, and visualizations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, systematic description of methods, and complete mathematical derivations of the fusion mechanism.
- Value: ⭐⭐⭐⭐ Provides an important multimodal baseline for egocentric perception and AR applications. The post-fusion contrastive learning strategy holds general reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Siamese Vision Transformers are Scalable Audio-Visual Learners](siamese_vision_transformers_are_scalable_audio-visual_learners.md)
- [\[ECCV 2024\] Action2Sound: Ambient-Aware Generation of Action Sounds from Egocentric Videos](action2sound_ambientaware_generation_of_action_sounds_from_e.md)
- [\[ECCV 2024\] CoLeaF: A Contrastive-Collaborative Learning Framework for Weakly Supervised Audio-Visual Video Parsing](coleaf_a_contrastive-collaborative_learning_framework_for_weakly_supervised_audi.md)
- [\[ECCV 2024\] Label-Anticipated Event Disentanglement for Audio-Visual Video Parsing](label-anticipated_event_disentanglement_for_audio-visual_video_parsing.md)
- [\[ICML 2026\] Do Audio LLMs Listen or Read? Analyzing and Mitigating Paralinguistic Failures with VoxParadox](../../ICML2026/audio_speech/do_audio_llms_listen_or_read_analyzing_and_mitigating_paralinguistic_failures_wi.md)

</div>

<!-- RELATED:END -->
