---
title: >-
  [Paper Note] CoLeaF: A Contrastive-Collaborative Learning Framework for Weakly Supervised Audio-Visual Video Parsing
description: >-
  [ECCV2024][Audio & Speech][audio-visual video parsing] This work proposes the CoLeaF dual-branch learning framework, which explicitly optimizes the integration of cross-modal context through event-aware contrastive learning, achieving an average improvement of 1.9% F-score on the weakly supervised audio-visual video parsing task.
tags:
  - "ECCV2024"
  - "Audio & Speech"
  - "audio-visual video parsing"
  - "weakly supervised learning"
  - "contrastive learning"
  - "knowledge distillation"
  - "cross-modal learning"
date: 2026-05-08
content_hash: d85cb5cbb4296ad4
---

# CoLeaF: A Contrastive-Collaborative Learning Framework for Weakly Supervised Audio-Visual Video Parsing

**Conference**: ECCV2024  
**arXiv**: [2405.10690](https://arxiv.org/abs/2405.10690)  
**Code**: [GitHub](https://github.com/faeghehsardari/coleaf)  
**Area**: Audio & Speech  
**Keywords**: audio-visual video parsing, weakly supervised learning, contrastive learning, knowledge distillation, cross-modal learning

## TL;DR

This work proposes the CoLeaF dual-branch learning framework, which explicitly optimizes the integration of cross-modal context through event-aware contrastive learning, achieving an average improvement of 1.9% F-score on the weakly supervised audio-visual video parsing task.

## Background & Motivation

The Audio-Visual Video Parsing (AVVP) task requires simultaneously detecting three types of events in a video: audible-only, visible-only, and audible-visible co-occurring events. Since frame-by-frame labeling of events in each modality is extremely costly, this task is conducted under a weakly supervised setting—where only video-level labels are available during training.

Existing methods (e.g., HAN, CMPAE) widely utilize uni-modal and cross-modal contexts simultaneously through self-attention and cross-attention. However, the authors identify a key contradiction: **while cross-modal learning benefits the detection of co-occurring audio-visual events, it degrades the performance on unaligned events (audible-only/visible-only)** because cross-modal attention introduces irrelevant modal information. Experimental verification shows that when CMPAE uses only uni-modal information, its true positive rate on unaligned events is actually higher.

Furthermore, complex class co-occurrence relationships (such as playing an instrument accompanied by singing) often exist in videos. Explicitly modeling these relationships can improve performance, but existing methods (e.g., CVCMS) incur a heavy computational overhead of $T \times C \times C$ for modeling inter-class relationships.

## Core Problem

1. **The dilemma of cross-modal information integration**: How can the network utilize cross-modal context for audio-visual co-occurring events while filtering out cross-modal interference for unaligned events under a weakly supervised scenario?
2. **Efficiency of inter-class relationship modeling**: How can class co-occurrence relationships be leveraged without increasing computational overhead during inference?

## Method

### Overall Architecture: Dual-Branch Design

CoLeaF contains two parallel branches that share the same input audio-visual features:

- **Reference branch** (used only during training): utilizes only uni-modal information and explicitly models inter-class relationships.
- **Anchor branch** (training + inference): utilizes uni-modal and cross-modal contexts (can embed any AVVP method such as HAN or CMPAE).

### Reference Branch

By concatenating the audio/visual input features $F^a, F^v$ with learnable class tokens $C^a, C^v$ (one token per class) and feeding them into the self-attention layers of their respective modalities, this design offers the following advantages:

- Self-attention learns uni-modal context in the temporal dimension.
- The interaction between class tokens and input tokens explicitly models inter-class co-occurrence relationships.
- It avoids cross-modal interference by not utilizing cross-attention.

The output temporal tokens predict segment-level event probabilities via an FC layer, and the class tokens, processed through AvgPool + Sigmoid, also participate in supervised learning via the BCE loss.

### Anchor Branch

Any AVVP method (such as the self-attention + cross-attention structure of HAN) can be reused to generate feature representations integrated with cross-modal context and event probability predictions.

### Event-Aware Contrastive Loss

Core Innovation: adaptively adjusts the strength of contrastive learning according to the proportion of unaligned events in the video.

$$\mathcal{L}_{Evt}^{Anch} = -\frac{1}{T} \sum_{\phi \in \{a,v\}} \vartheta^\phi \sum_{t=1}^{T} \log \frac{\exp(\hat{f}_t^{\phi\top} \cdot \ddot{x}_t^\phi / \tau)}{\sum_{n \neq t} \exp(\hat{f}_t^{\phi\top} \cdot \ddot{x}_n^\phi / \tau)}$$

Where the weight $\vartheta^\phi$ reflects the degree of unalignment:

- Pseudo-labels are extracted from the predictions of the Reference branch to count the number of audible-only events $N^a$, visible-only events $N^v$, and co-occurring events $N^{av}$.
- $\vartheta^a = N^a / (N^a + N^{av})$: The more unaligned events there are, the stronger the contrastive learning is encouraged, forcing the Anchor to align closer to the Reference's uni-modal representations.
- When all events are audio-visual co-occurring, $\vartheta = 0$, and no contrastive constraints are applied.

### Self-Modality-Aware KD

Because the Reference is trained solely on modality-agnostic video-level labels, its representation capability is limited. To address this, the Anchor reversely distills modality-aware pseudo-labels to the Reference:

$$\mathcal{L}_{SelfMo}^{Ref} = \sum_{\phi \in \{a,v\}} BCE(G^\phi, \ddot{\mathcal{P}}^\phi)$$

Where $G^\phi$ represents the pseudo-labels obtained by thresholding the Anchor's predicted audio/visual probabilities. This forms a **collaborative closed loop**: the Anchor provides modality-aware supervision for the Reference, while the Reference optimizes the Anchor's cross-modal integration via contrastive learning.

### Co-occurrence Class KD

Inter-class relationships are transferred via a class correlation matrix:

$$\mathcal{L}_{CoCls}^{Anch} = \sum_{\phi \in \{a,v\}} MSE(\ddot{M}^\phi, M^\phi)$$

Where $\ddot{M}^\phi_{i,j} = \ddot{\mathcal{P}}_i^\phi \cdot \ddot{\mathcal{P}}_j^\phi$ is the inter-class correlation matrix of the Reference. During inference, the class tokens and the Reference branch are completely discarded, incurring no extra computational cost.

### New Evaluation Metrics

The authors point out the limitations of traditional A/V metrics: comparing uni-modal predictions only against uni-modal labels misclassifies co-occurring predictions as true positives for audible-only/visible-only. They propose the Ao/Vo metrics which consider both modalities simultaneously:

$$\hat{y}_t^{ao} = \hat{y}_t^a \odot (1 - \hat{y}_t^v)$$

An event is counted as audible-only if and only if the prediction is "audio present and video absent".

## Key Experimental Results

**LLP Dataset** (standard setting, Segment-level / Event-level):

| Method | Ao | Vo | AV | Ao | Vo | AV |
|------|-----|-----|-----|-----|-----|-----|
| HAN | 33.1 | 50.7 | 48.9 | 31.0 | 50.1 | 43.0 |
| JoMoLD | 46.2 | 58.8 | 57.2 | 40.9 | 59.0 | 49.6 |
| CMPAE | 48.2 | 57.9 | 57.5 | 43.6 | 57.5 | 49.6 |
| **CoLeaF** | **49.3** | **62.4** | **58.6** | **44.1** | **62.2** | **52.1** |

Key ablation results:

- Event-aware contrastive loss $\rightarrow$ Ao improved by 2.1% (segment) / 2.5% (event).
- Self-modality distillation $\rightarrow$ all event types improved by 0.7% ~ 1.6%.
- Class token + co-occurrence distillation $\rightarrow$ comprehensive gains of 1.1% ~ 2.3%.
- Framework generality: embedding into HAN and CMPAE achieves an average gain of 2.4% F-score.

## Highlights & Insights

1. **Precise Problem Insight**: For the first time, the negative impact of cross-modal learning on unaligned events is quantitatively analyzed, and a targeted solution is proposed.
2. **Event-Aware Contrastive Learning**: Adaptively regulates the strength of cross-modal constraints based on video content, showing a clever design.
3. **Dual-Branch Collaborative Mechanism**: The bidirectional knowledge transfer between Anchor $\leftrightarrow$ Reference forms a positive feedback loop.
4. **Zero Inference Overhead**: The inter-class relationship modeling and the Reference branch are used exclusive to training, incurring absolutely no additional computational cost during inference.
5. **Strong Generalizability**: Any AVVP method can be embedded as the Anchor, providing a plug-and-play solution.
6. **Contribution of New Metrics**: The Ao/Vo metrics reflect the detection capability for unaligned events more accurately, highlighting the misleading nature of traditional metrics.

## Limitations & Future Work

1. Evaluated only on LLP and UnAV-100 datasets, lacking evaluation on larger-scale datasets.
2. The quality of pseudo-labels relies on the preset threshold $\theta$, which may require hyperparameter tuning for different datasets.
3. The architectural choices for the Reference and Anchor branches are relatively simple, and stronger backbones have not been explored.
4. Cross-modal alignment of temporal contexts is not considered, as the contrastive strength is only adjusted globally.
5. Class tokens increase the FLOPs of the Reference by 41.7%, which does not affect inference but increases the training cost.

## Related Work & Insights

| Method | Cross-modal Optimization | Inter-class Relationship | Inference Overhead |
|------|--------------|---------|---------|
| HAN | Direct fusion of self+cross attention | None | Baseline |
| CVCMS | Learn inter-class dependencies | Explicit modeling | $T \times C^2$ extra overhead |
| CMPAE | Subjective logic theory regulation | None | Baseline |
| JoMoLD | CLIP/CLAP offline labels | None | Requires two-stage training |
| **CoLeaF** | **Explicit contrastive optimization in embedding space** | **Modeled during training, distilled during inference** | **Zero inference overhead** |

## Insights & Inspiration

1. The paradigm of "complex training, simple inference" is highly worth referencing—introducing an auxiliary branch during the training phase to learn specific knowledge, and then transferring it to the main network via distillation.
2. The adaptive mechanism for contrastive learning strength can be transferred to other multimodal scenarios: dynamically adjusting cross-modal fusion according to the degree of modality alignment.
3. The deficiency analysis of traditional metrics provides an important reminder: the design of evaluation metrics should be strictly consistent with the task definition.

## Rating
- Novelty: ⭐⭐⭐⭐ (The designs of event-aware contrastive learning and collaborative distillation mechanism are novel)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensively ablated but evaluated on relatively few datasets)
- Writing Quality: ⭐⭐⭐⭐ (Clear motivation and complete formulas)
- Value: ⭐⭐⭐⭐ (General framework + new metrics, advancing the field)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Label-Anticipated Event Disentanglement for Audio-Visual Video Parsing](label-anticipated_event_disentanglement_for_audio-visual_video_parsing.md)
- [\[CVPR 2025\] UWAV: Uncertainty-Weighted Weakly-Supervised Audio-Visual Video Parsing](../../CVPR2025/audio_speech/uwav_uncertainty-weighted_weakly-supervised_audio-visual_video_parsing.md)
- [\[ICCV 2025\] MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing](../../ICCV2025/audio_speech/mug_pseudo_labeling_augmented_audio-visual_mamba_network_for_audio-visual_video_.md)
- [\[ECCV 2024\] Siamese Vision Transformers are Scalable Audio-Visual Learners](siamese_vision_transformers_are_scalable_audio-visual_learners.md)
- [\[ECCV 2024\] Listen to Look into the Future: Audio-Visual Egocentric Gaze Anticipation](listen_to_look_into_the_future_audio-visual_egocentric_gaze_anticipation.md)

</div>

<!-- RELATED:END -->
