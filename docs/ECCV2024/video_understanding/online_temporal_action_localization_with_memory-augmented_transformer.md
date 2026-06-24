---
title: >-
  [Paper Note] Online Temporal Action Localization with Memory-Augmented Transformer
description: >-
  [ECCV 2024][Video Understanding][Online Temporal Action Localization] This paper proposes MATR (Memory-Augmented Transformer), which models long-term context by selectively storing historical segment features in a memory queue, and employs a dual Transformer decoder to locate the end and start times of actions respectively. It achieves new state-of-the-art results on two online temporal action localization benchmarks, THUMOS14 and MUSES, even comparable to some offline method…
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Online Temporal Action Localization"
  - "Memory-Augmented Transformer"
  - "Long-term Context Modeling"
  - "End-to-end Detection"
  - "Sliding Window"
date: 2026-05-08
content_hash: 1b06ea893bdeda68
---

# Online Temporal Action Localization with Memory-Augmented Transformer

**Conference**: ECCV 2024  
**arXiv**: [2408.02957](https://arxiv.org/abs/2408.02957)  
**Code**: [https://cvlab.postech.ac.kr/research/MATR/](https://cvlab.postech.ac.kr/research/MATR/)  
**Area**: Others (Video Understanding)  
**Keywords**: Online Temporal Action Localization, Memory-Augmented Transformer, Long-term Context Modeling, End-to-end Detection, Sliding Window

## TL;DR
This paper proposes MATR (Memory-Augmented Transformer), which models long-term context by selectively storing historical segment features in a memory queue, and employs a dual Transformer decoder to locate the end and start times of actions respectively. It achieves new state-of-the-art results on two online temporal action localization benchmarks, THUMOS14 and MUSES, even comparable to some offline methods.

## Background & Motivation
Temporal Action Localization (TAL) aims to detect the start time, end time, and category of each action instance in untrimmed videos. Online TAL (On-TAL) requires inference using only the frames up to the current time, and predictions already output cannot be modified. Applicable scenarios include video surveillance, sports analysis, and video summarization. Early On-TAL methods based on Online Action Detection (OAD) classify each frame first and then aggregate them into instances, but this frame-level supervision is sub-optimal. Subsequent method OAT introduced a sliding window and anchor mechanism to exploit instance-level supervision, but still exhibits two key limitations: (1) each iteration only processes a fixed-size video segment, failing to model long-term actions beyond the window; (2) performance is highly sensitive to the input segment size, requiring careful parameter tuning for each dataset. The key challenge lies in how to effectively utilize long-term context to locate action instances spanning multiple segments under the online setting. The core idea of this paper is to selectively store past segment features using a FIFO memory queue and employ a dual decoder—detecting the action end first, then retrieving the action start from memory—to achieve precise localization.

## Method

### Overall Architecture
MATR consists of four parts: (1) Feature Extractor: extracts frame-level features of the current input segment using a pre-trained backbone network (TSN/I3D); (2) Memory-Augmented Video Encoder: encodes intra-segment temporal context using a Transformer encoder, and selectively stores segment features into a memory queue via a flag token mechanism; (3) Instance Decoding Module: composed of two Transformer decoders, the end decoder and the start decoder, which utilize the current segment features to locate the action end and the memory queue to locate the action start, respectively; (4) Prediction Heads: respectively predict the end offset, the start area + offset, and the action category. The model infers frame-by-frame in a sliding window manner and is trained end-to-end.

### Key Designs
1. **Memory Queue and Flag Token Mechanism**:
    - **Function**: Selectively store past segment features and provide long-term context for the model.
    - **Mechanism**: The memory queue is managed in a FIFO manner, discarding the oldest when the queue is full. The key innovation is introducing a learnable flag token, which is fed into the encoder along with the segment features, and then a flag prediction head determines whether the current segment is associated with an action instance. During training, the ground truth FLAG label is used, while during inference, it is determined by $\text{sigmoid}(\hat{g}) > \theta$. Segments are stored in the memory only when FLAG=1, which effectively filters out background frames and improves memory utilization efficiency.
    - **Design Motivation**: Unlike memory modules in OAD, TAL requires preserving temporal position information to predict time precisely, and directly compressing the memory would discard crucial temporal positions. Selective storage is more efficient than storing everything, avoiding interference from background frames.

2. **End-Start Dual Decoder Localization**:
    - **Function**: Locate the end and start times of actions respectively.
    - **Mechanism**: The End Decoder utilizes the encoded current segment features to locate the action end near the current time through cross-attention. The Start Decoder receives the output embeddings of the End Decoder and utilizes the memory queue concatenated with the current segment features as long-term context to find the action start via cross-attention. The two decoders share the architecture but use different information sources. A 2D temporal position encoding (relative segment position + relative frame position) is adopted to support streaming videos of unpredictable length.
    - **Design Motivation**: The end of an action is typically near the current segment (which can be derived from short-term features), while the start might have occurred long ago (requiring long-term memory). Since their information requirements differ, modeling them separately is more reasonable. Experiments demonstrate that the dual-decoder design improves by 6.8 mAP compared to a single decoder.

3. **Class-Boundary Decoupled Queries**:
    - **Function**: Decouple the two sub-tasks of action classification and boundary localization.
    - **Mechanism**: For each instance, a pair of queries is set up: a class query $Q_\text{class}$ is responsible for action classification, and a boundary query $Q_\text{bound}$ is responsible for boundary localization. Both share the same positional encoding $E_\text{pos}$ to associate with the same instance. The classification head concatenates the class embeddings from the End Decoder and the Start Decoder to output the category probability. The start prediction head adopts a hierarchical strategy of region classification + offset regression, dividing the temporal range into $L_m + 2$ regions.
    - **Design Motivation**: Inspired by the practice of decoupling classification and localization in object detection (such as the DETR series), allowing different queries to focus on different sub-tasks reduces interference between tasks.

### Loss & Training
The model is trained end-to-end using the Hungarian algorithm to match predictions with ground truth. The total loss is $L = L_\text{class} + L_\text{start} + L_\text{end} + L_\text{diou} + L_\text{flag}$, where Focal Loss is used for classification, cross-entropy for the start region, L1 loss for start and end offsets, DIoU Loss for instance-level supervision, and BCE Loss for the flag token. All loss weights are set to 1 without requiring extra balancing. During inference, NMS is applied at each time step, and instances with predicted end times exceeding the current time are removed.

## Key Experimental Results

### Main Results
Comparison of mAP (%) on THUMOS14 and MUSES datasets:

| Method | Type | THUMOS14 Avg mAP | MUSES Avg mAP |
|------|------|-----------------|--------------|
| SimOn | OAD-based Online | 34.4 | - |
| CAG-QIL | OAD-based Online | 29.7 | 4.8 |
| OAT-OSN | Instance Online | 44.6 | 13.7 |
| **MATR** | **Instance Online** | **49.5** | **14.4** |
| G-TAD | Offline | 39.9 | 11.4 |
| MUSES | Offline | 53.4 | 18.6 |
| ActionFormer | Offline | 66.8 | - |

MATR outperforms the previous state-of-the-art OAT-OSN by 4.9 percentage points on THUMOS14 and 0.7 percentage points on MUSES, even surpassing some offline methods (e.g., G-TAD, P-GCN).

### Ablation Study

| Configuration | Avg mAP | Description|
|------|---------|------|
| Full model (MATR) | 49.5 | Full model |
| w/o flag token | 47.4 | Without selective storage, -2.1 |
| w/o segment encoder | 46.6 | Without segment encoder, -2.9 |
| Single decoder | 42.7 | Single decoder predicting start and end simultaneously, -6.8 |
| w/o splitting query | 47.9 | Without decoupling class/boundary queries, -1.6 |
| w/o sampling | 47.2 | Without sampling the memory, -2.3 |
| w/o DIoU loss | 41.4 | Without DIoU loss, -8.1 |
| w/o memory | 46.0 | Completely without memory queue, -3.5 |
| memory size=7 (best) | 49.5 | Best memory size for THUMOS14 |
| memory size=15 (best) | 14.4 | Best memory size for MUSES |

### Key Findings
- The dual-decoder design is the most critical component; using a single decoder drops performance by 6.8 mAP.
- DIoU instance-level supervision is crucial for online TAL (-8.1 mAP), indicating that frame-level supervision is insufficient for learning precise boundary localization.
- The size of the memory queue only needs to cover the duration of 99% of the instances in the training set (about 7 segments for THUMOS14); larger sizes do not necessarily yield better results.
- MATR is robust to segment sizes: when reducing the segment size from 64 to 8, the performance only drops by 9.1%, whereas OAT-OSN drops from 44.6% to 25.8%.
- Compared to OAD memory modules, MATR requires only 24M parameters and a 167ms inference time, whereas MAT requires 40M/192ms and E2E-LOAD requires 53M/196ms.
- Using region classification + offset regression for start prediction (49.5) outperforms pure offset regression (46.7).

## Highlights & Insights
- The detection paradigm of "locating the action end first, then retrieving the action start from memory" is intuitive and reasonable — human judgment of action ending is immediate, while recalling the start requires accessing long-term memory.
- The selective storage using flag tokens is both simple and effective. It represents an elegant memory management scheme that avoids complex memory compression and attentional selection.
- The decoupled design of class query and boundary query successfully borrows experiences from the field of object detection.
- The 2D temporal position encoding (segment-level + frame-level) cleverly addresses the position encoding issue for streaming video with unpredictable durations.
- End-to-end training with all loss coefficients set to 1 is simple and requires no hyperparameter tuning.

## Limitations & Future Work
- When multiple action instances coexist in the memory queue, matching errors for start points may occur.
- When storing segments, only whether they are related to actions is considered, without utilizing the contextual relationships within the already stored memory.
- The performance gain is limited on datasets with frequent shot cuts like MUSES, which may require stronger cross-shot modeling capabilities.
- NMS post-processing is still required during inference, which is not fully end-to-end.
- The model parameter size (192.8M) is significantly larger than OAT-OSN (128.7M), primarily due to the dual-decoder design.

## Related Work & Insights
- **DETR/ActionFormer**: Paradigms for end-to-end Transformer detection and temporal localization. This paper introduces a similar idea into the online setting.
- **OAT**: The first On-TAL method using instance-level supervision, but it is limited by fixed windows and fails to model long-term actions.
- **Stream Buffer / MAT**: Memory modules in OAD that are inapplicable to TAL scenarios because compression loses temporal position information.
- **Insight**: Memory management is a core problem in online video understanding, where selective storage (when to store/when to discard) is more critical than simple compression.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of end-to-end online TAL + memory queue + dual decoders is novel, though the individual components are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on two datasets with detailed ablation studies (modules, memory size, segment size, prediction head, memory compression, and inference time).
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich illustrations, and well-articulated methodology.
- Value: ⭐⭐⭐⭐ Actively advances the online TAL task and fills the gap in long-term context modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] HAT: History-Augmented Anchor Transformer for Online Temporal Action Localization](hat_history-augmented_anchor_transformer_for_online_temporal_action_localization.md)
- [\[ECCV 2024\] Bayesian Evidential Deep Learning for Online Action Detection](bayesian_evidential_deep_learning_for_online_action_detection.md)
- [\[CVPR 2025\] Context-Enhanced Memory-Refined Transformer for Online Action Detection](../../CVPR2025/video_understanding/context-enhanced_memory-refined_transformer_for_online_action_detection.md)
- [\[ECCV 2024\] Optimizing Factorized Encoder Models: Time and Memory Reduction for Scalable and Efficient Action Recognition](optimizing_factorized_encoder_models_time_and_memory_reduction_for_scalable_and_.md)
- [\[ECCV 2024\] Leveraging Temporal Contextualization for Video Action Recognition](leveraging_temporal_contextualization_for_video_action_recognition.md)

</div>

<!-- RELATED:END -->
