---
title: >-
  [Paper Note] Label-Anticipated Event Disentanglement for Audio-Visual Video Parsing
description: >-
  [ECCV2024][Audio & Speech][Audio-Visual Video Parsing] This paper proposes the LEAP (Label semantic-based Projection) decoding paradigm, which utilizes the text embeddings of event categories as semantic anchors. Using a cross-modal attention mechanism, potentially overlapping event semantics within audio/visual latent features are disentangled into independent label embeddings. Combined with an EIoU-based audio-visual semantic similarity loss…
tags:
  - "ECCV2024"
  - "Audio & Speech"
  - "Audio-Visual Video Parsing"
  - "Event Disentanglement"
  - "Label Semantic Projection"
  - "Weakly Supervised"
date: 2026-05-08
content_hash: a8ada4c6541b0616
---

# Label-Anticipated Event Disentanglement for Audio-Visual Video Parsing

**Conference**: ECCV2024  
**arXiv**: [2407.08126](https://arxiv.org/abs/2407.08126)  
**Authors**: Jinxing Zhou, Dan Guo, Yuxin Mao, Yiran Zhong, Xiaojun Chang, Meng Wang (Hefei University of Technology, Northwestern Polytechnical University, Shanghai AI Lab, USTC, MBZUAI)  
**Code**: To be confirmed  
**Area**: Audio & Speech  
**Keywords**: Audio-Visual Video Parsing, Event Disentanglement, Label Semantic Projection, Weakly Supervised  

## TL;DR

This paper proposes the LEAP (Label semantic-based Projection) decoding paradigm, which utilizes the text embeddings of event categories as semantic anchors. Using a cross-modal attention mechanism, potentially overlapping event semantics within audio/visual latent features are disentangled into independent label embeddings. Combined with an EIoU-based audio-visual semantic similarity loss, LEAP achieves SOTA performance on the AVVP task.

## Background & Motivation

The Audio-Visual Video Parsing (AVVP) task requires identifying and temporally localizing all audio events, visual events, and audio-visual events from audible videos. This task is conducted under a weakly supervised setting, where only video-level event labels are available during training.

Existing methods mostly focus on improving audio-visual encoders to obtain better feature representations, but pay insufficient attention to the decoding stage. The dominant MMIL (Multi-modal Multi-Instance Learning) decoding strategy relies on a simple linear layer to directly map latent features to the event category space, which suffers from two key challenges:

1. **Insufficient semantic disentanglement**: When a temporal segment contains multiple overlapping events, the linear layer struggles to clearly show how overlapping semantics are separated from the mixed features.
2. **Poor interpretability**: The decoding process lacks intuitive semantic guidance, making it difficult to trace how events are recognized.

## Core Problem

How to design a more interpretable event decoding paradigm that allows multiple potentially overlapping event semantics in audio-visual latent features to be explicitly disentangled and recognized?

## Method

### 1. LEAP Decoding Paradigm (Label Semantic-based Projection)

Core Idea: Utilize natural language text of event categories (e.g., "dog", "guitar") to obtain semantically independent label embeddings, which serve as semantic anchors for decoding.

**Label Embedding Acquisition**: A pre-trained GloVe model is used to encode the text of $C$ event categories into a label semantic matrix $F^l \in \mathbb{R}^{C \times d}$.

**Cross-modal Projection**: A Transformer cross-attention mechanism is employed to achieve projection:

- **Query**: Label embedding $F^l$ (representing each event semantic)
- **Key/Value**: Audio or visual features $F^m$ ($m \in \{a, v\}$)
- Calculate the cross-attention matrix $A^{lm} \in \mathbb{R}^{C \times T}$, which reflects the similarity between each event category and each temporal segment.
- Aggregate relevant semantic information based on attention weights to enhance the label embedding.

**Iterative Refinement**: The LEAP module can be stacked iteratively $N$ times (experimentally $N=2$), repeatedly utilizing the encoded features to step-by-step enhance the label embeddings corresponding to actual events, making them more discriminative.

**Event Prediction**:

- **Segment-level Prediction**: Directly apply sigmoid to the cross-attention matrix of the last round $A_N^{lm}$ to obtain segment-level event probabilities $P^m \in \mathbb{R}^{T \times C}$.
- **Video-level Prediction**: Pass the enhanced label embeddings $F_N^{lm}$ through a linear layer + sigmoid to obtain video-level event probabilities $p^m \in \mathbb{R}^{1 \times C}$.

### 2. Semantic-Aware Optimization Strategy

**Basic Loss $\mathcal{L}_{basic}$**: Combines video-level weak labels and segment-level pseudo-labels (from the VALOR method) to impose BCE constraints on audio and visual event predictions.

**Audio-Visual Semantic Similarity Loss $\mathcal{L}_{avss}$**:

- Propose the EIoU (Event Intersection over Union) metric: Calculates the IoU of event category sets between each pair of audio-visual segments, serving as the calibration value for cross-modal semantic similarity.
- For instance, if an audio segment contains events $\{c_1, c_2, c_3\}$ and a visual segment contains $\{c_1, c_2\}$, then $\text{EIoU} = 2/3$.
- Construct the EIoU matrix $r \in \mathbb{R}^{T \times T}$ as the supervision target.
- Calculate the cosine similarity matrix $s \in \mathbb{R}^{T \times T}$ of the encoded features, and use MSE loss to make $s$ approach $r$.

**Total Loss**: $\mathcal{L} = \mathcal{L}_{basic} + \lambda \mathcal{L}_{avss}$, with $\lambda = 1$.

### 3. Compatibility with Existing Encoders

As a decoder, LEAP can be integrated plug-and-play with any audio-visual encoder (e.g., HAN, MM-Pyr) to replace the original MMIL decoding strategy.

## Key Experimental Results

**Dataset**: LLP (Look, Listen, and Parse), containing 11,849 YouTube videos across 25 event categories.

**LEAP vs MMIL Comparison (MM-Pyr Encoder)**:

| Metric | MMIL | LEAP | Gain |
|------|------|------|------|
| Segment Type@AV | 62.2 | 64.8 | +2.6 |
| Segment Event@AV | 60.6 | 63.6 | +3.0 |
| Event Type@AV | 57.1 | 60.2 | +3.1 |
| Event Event@AV | 53.0 | 57.4 | +4.4 |

**Comparison with SOTA**: Achieves optimal performance on all event parsing metrics, outperforming methods such as CMPAE (CVPR'23) and VALOR (NeurIPS'23).

**Overlapping Event Processing**: On the overlapping event subset, LEAP improves by an average of 1.7% over MMIL (using the MM-Pyr encoder).

**Ablation Study**:

- The trade-off between performance and computation is optimal when the number of LEAP modules is $N=2$ (Avg. 61.3%).
- Among the label embedding strategies, GloVe is optimal, while BERT and CLIP are also effective (showing the method is robust to the choice of embedding).
- $\mathcal{L}_{avss}$ brings an additional gain of approximately 1.0% on the MM-Pyr encoder.

## Highlights & Insights

1. **Novel decoding paradigm**: Incorporates label text semantics into the decoding process and treats semantically independent label embeddings as "projection targets" to disentangle overlapping events. The idea is intuitive and effective.
2. **Strong interpretability**: The cross-attention matrix directly reflects event-segment mapping, making the decoding process trackable.
3. **Plug-and-play**: LEAP can replace the MMIL decoder in any AVVP method, offering excellent generalizability.
4. **EIoU Metric**: Uses the IoU of event sets as a cross-modal semantic similarity metric, elegantly addressing the variation in event density across different modalities.

## Limitations & Future Work

1. The label embedding utilizes simple GloVe word embeddings and does not leverage richer semantic descriptions (such as acoustic or visual descriptive features of events), which limits its semantic expressiveness.
2. Segment-level pseudo-labels depend on external methods like VALOR for generation, and the quality of these pseudo-labels has a significant impact on LEAP's performance.
3. The EIoU matrix is calculated based on pseudo-labels; thus, noise in the pseudo-labels will propagate to the similarity supervision signal.
4. The experiments are only validated on a single dataset (LLP), lacking validation on larger-scale datasets or datasets with more categories.
5. LEAP introduces additional Transformer decoding modules, which increase computational overhead compared to simple linear layers.

## Related Work & Insights

| Method | Key Improvements | Event@AV (Event) |
|------|-----------|-----------------|
| HAN (ECCV'20) | Baseline Encoder + MMIL | 48.0 |
| VALOR (NeurIPS'23) | Segment-level Pseudo-labels + MMIL | 54.2 |
| CMPAE (CVPR'23) | Stronger Encoder + Class-Adaptive Threshold | 55.7 |
| **LEAP (Ours)** | **Label Semantic Projection Decoding** | **57.4** |

Key Difference: Previous works mainly improve the encoder or label generation, whereas this work is the first to systematically improve the decoding stage, which is orthogonally complementary to encoder-side improvements.

## Related Work & Insights

1. The concept of **label-semantic guided decoding** can be generalized to other multi-label classification scenarios (such as multi-label image classification and action recognition), especially those involving overlapping labels.
2. EIoU, as a cross-modal semantic alignment metric, can be adapted for other tasks requiring heterogeneous modality alignment.
3. The paradigm of "projecting latent features onto semantically independent anchors" aligns with the concept of object queries in DETR, indicating potential of further exploration in combination with query-based detection frameworks.

## Rating

- Novelty: ⭐⭐⭐⭐ — Tackles the problem from a decoding paradigm perspective; label semantic projection is a novel design.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablations and detailed comparison with MMIL, though limited to a single dataset.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, intuitive diagrams, and complete mathematical derivations.
- Value: ⭐⭐⭐⭐ — A plug-and-play decoding improvement with high practicality and generalizable ideas.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] CoLeaF: A Contrastive-Collaborative Learning Framework for Weakly Supervised Audio-Visual Video Parsing](coleaf_a_contrastive-collaborative_learning_framework_for_weakly_supervised_audi.md)
- [\[ICCV 2025\] MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing](../../ICCV2025/audio_speech/mug_pseudo_labeling_augmented_audio-visual_mamba_network_for_audio-visual_video_.md)
- [\[CVPR 2025\] UWAV: Uncertainty-Weighted Weakly-Supervised Audio-Visual Video Parsing](../../CVPR2025/audio_speech/uwav_uncertainty-weighted_weakly-supervised_audio-visual_video_parsing.md)
- [\[CVPR 2025\] Towards Open-Vocabulary Audio-Visual Event Localization](../../CVPR2025/audio_speech/towards_open-vocabulary_audio-visual_event_localization.md)
- [\[ECCV 2024\] Siamese Vision Transformers are Scalable Audio-Visual Learners](siamese_vision_transformers_are_scalable_audio-visual_learners.md)

</div>

<!-- RELATED:END -->
