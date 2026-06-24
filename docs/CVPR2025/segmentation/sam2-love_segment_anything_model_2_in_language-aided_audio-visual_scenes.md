---
title: >-
  [Paper Note] SAM2-LOVE: Segment Anything Model 2 in Language-Aided Audio-Visual Scenes
description: >-
  [CVPR 2025][Segmentation][Audio-Visual Segmentation] SAM2-LOVE designs a multimodal fusion Transformer to compress text, audio, and visual tri-modal information into a learnable token to prompt SAM2. Combined with token propagation and accumulation strategies to enhance spatiotemporal consistency, it outperforms the state-of-the-art (EEMC) by 8.5 percentage points on the Ref-AVS benchmark with a $\mathcal{J\&F}$ score of 58.5%.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Audio-Visual Segmentation"
  - "SAM2"
  - "Multimodal Fusion"
  - "Referring Segmentation"
  - "Spatiotemporal Consistency"
date: 2026-05-08
content_hash: df2a1779c855ee71
---

# SAM2-LOVE: Segment Anything Model 2 in Language-Aided Audio-Visual Scenes

**Conference**: CVPR 2025  
**arXiv**: [2506.01558](https://arxiv.org/abs/2506.01558)  
**Code**: [https://github.com/yuji-wang/SAM2-LOVE](https://github.com/yuji-wang/SAM2-LOVE)  
**Area**: Segmentation  
**Keywords**: Audio-Visual Segmentation, SAM2, Multimodal Fusion, Referring Segmentation, Spatiotemporal Consistency

## TL;DR

SAM2-LOVE designs a multimodal fusion Transformer to compress text, audio, and visual tri-modal information into a learnable token to prompt SAM2. Combined with token propagation and accumulation strategies to enhance spatiotemporal consistency, it outperforms the state-of-the-art (EEMC) by 8.5 percentage points on the Ref-AVS benchmark with a $\mathcal{J\&F}$ score of 58.5%.

## Background & Motivation

**Background**: Referring Audio-Visual Segmentation (Ref-AVS) is an emerging task requiring models to continuously segment target objects from videos based on textual expressions and audio signals, which demands pixel-level scene understanding in "Language-Aided Audio-Visual Scenes" (LAVS). Existing methods are categorized into bi-modal approaches (text-visual or audio-visual) and tri-modal approaches (EEMC).

**Limitations of Prior Work**: (1) Bi-modal methods fail to locate targets accurately due to the lack of the third modality information—text-visual methods (such as EVF-SAM) cannot distinguish sounding objects in silent scenes, and audio-visual methods (such as GAVS) cannot comprehend dynamic control signals in user text. (2) Although the existing tri-modal method EEMC models all three modalities simultaneously, it suffers from insufficient spatiotemporal consistency. Despite having a memory cache, the model still fails to continuously track target positions and shapes, causing the segmented areas to drift over time.

**Key Challenge**: The Ref-AVS task demands both tri-modal comprehension and video-level spatiotemporal consistency. EEMC performs well in tri-modal understanding but lacks robust video tracking capabilities, whereas SAM2 possesses powerful video segmentation and tracking abilities but lacks text and audio understanding.

**Goal**: To integrate the powerful video segmentation capability of SAM2 with tri-modal understanding, achieving pixel-level comprehension in LAVS.

**Key Insight**: SAM2 follows a "prompt-then-propagate" paradigm—using prompts to locate targets in keyframes and then propagating the results throughout the video. The key lies in how to compress tri-modal information into effective prompts to drive SAM2.

**Core Idea**: To design a fusion Transformer that compresses tri-modal information into a learnable [seg] token, which prompts the first frame of SAM2, then utilizing the zero-shot VOS capabilities of SAM2 to propagate across the entire video. Spatiotemporal understanding is enhanced through token propagation (forward knowledge transfer) and token accumulation (backward knowledge transfer) strategies.

## Method

### Overall Architecture

The pipeline of SAM2-LOVE consists of three main components: (1) **Multimodal Encoder**: VGGish encodes audio, ViT encodes video frames, and DistilRoBERTa encodes text, with each projected to a unified dimension via MLPs; (2) **Multimodal Fusion Transformer**: A 6-layer bidirectional Transformer that fuses the tri-modal sequences and the learnable [seg] token, outputting a compressed multimodal representation; (3) **SAM2**: Prompts the first frame with the [seg] token to locate the target and propagates it to the entire video via memory attention. During training, the loss is computed only on the first frame, while during inference, SAM2 processes the entire video sequence.

### Key Designs

1. **Multimodal Fusion Module**:

    - **Function**: Compresses textual, audio, and visual tri-modal information into a single learnable token to prompt SAM2.
    - **Mechanism**: Defines a learnable [seg] token and prepends it to the multimodal sequence: $F_M^i = \text{Concat}([[seg]; \hat{F}_A; [aud]; \hat{F}_T; [vis]; \hat{F}_V^i])$, where [aud] and [vis] are fixed modality indicator tokens. This sequence is fed into a 6-layer bidirectional Transformer encoder. The [seg] token interacts with all modality features via self-attention, and the first element is extracted upon output as the updated [seg] embedding, containing compressed tri-modal information.
    - **Design Motivation**: SAM2's prompt interface is designed to accept sparse prompts (points/boxes/masks), making a single token the most natural adaptation method. Bidirectional attention allows the [seg] token to perceive information from all modalities simultaneously.

2. **Token Propagation Strategy (Forward Knowledge Transfer)**:

    - **Function**: Propagates the [seg] token across video frames to adaptively capture intra-frame spatial features and inter-frame continuity.
    - **Mechanism**: When processing the $i$-th frame, the [seg] token output from the previous frame is used as the input for the current frame. Audio and text features remain unchanged, while only the visual features are updated to the current frame. Thus, the [seg] token propagates frame-by-frame, continuously accumulating temporal information. Ultimately, the [seg] token, fully propagated across the entire video, prompts the first frame of SAM2.
    - **Design Motivation**: Conventional approaches independently fuse tri-modal information for each frame, which ignores temporal relationships across frames. Through propagation, the token can model the spatiotemporal dynamics of the video.

3. **Token Accumulation Strategy (Backward Knowledge Transfer)**:

    - **Function**: Prevents the [seg] token from forgetting information of earlier frames during propagation in long videos.
    - **Mechanism**: Maintains a sequence of historical tokens [his]. After processing each frame, the global [cls] token from that frame's ViT is appended to [his]: $[his]^i = \text{Concat}([[cls]^0; [cls]^1; ...; [cls]^i])$. When processing the $(i+1)$-th frame, [his] is appended to $F_M^{i+1}$, allowing the [seg] token to review the global representation of all historical frames via attention.
    - **Design Motivation**: As the number of propagation steps increases, information from early frames gets diluted. The accumulation strategy provides a "replay" mechanism that complements propagation—propagation acts as forward knowledge transfer, whereas accumulation acts as backward knowledge transfer.

### Loss & Training

- **Loss**: $\mathcal{L}_{mask} = \lambda_{bce} \text{BCE}(\hat{M}, M) + \lambda_{dice} \text{DICE}(\hat{M}, M)$, where $\lambda_{bce}=\lambda_{dice}=1.0$.
- **Training Strategy**: The fusion module and [seg] token access the complete video sequence, while SAM2 is supervised only on the first frame. During inference, SAM2 receives the full video leveraging its zero-shot VOS capability.
- **Optimization**: DeepSpeed ZeRO-2, AdamW, lr=1e-4, batch=8, grad accumulation=2, 2×A100.
- **Freezing Strategy**: Audio/visual/text encoders and SAM2 image encoder/memory attention are frozen; fusion Transformer and SAM2 prompt encoder/mask decoder are trainable.

## Key Experimental Results

### Main Results

| Method | Seen $\mathcal{J\&F}$ | Unseen $\mathcal{J\&F}$ | Mix $\mathcal{J\&F}$ |
|------|----------------------|------------------------|---------------------|
| GAVS+text | 39.4 | 39.8 | 39.6 |
| ReferFormer+audio | 40.7 | 39.6 | 40.2 |
| EEMC | 42.8 | 57.2 | 50.0 |
| **SAM2-LOVE** | **47.7** | **69.4** | **58.5** |
| vs EEMC | +4.9 | +12.2 | **+8.5** |

### Ablation Study

| Design | Seen $\mathcal{J\&F}$ | Unseen $\mathcal{J\&F}$ |
|------|----------------------|------------------------|
| CLIP encoder (w/o [cls] accumulation) | 46.8 | 68.2 |
| RoBERTa+ViT (w/o [cls] accumulation) | 46.7 | 69.0 |
| RoBERTa+ViT (w/ [cls] accumulation) | **47.7** | **69.4** |
| 1-layer Transformer | 45.4 | 68.1 |
| 6-layer Transformer | 46.7 | 69.0 |
| 12-layer Transformer | 46.4 | 70.5 |

### Key Findings

- SAM2's zero-shot VOS capability is key to the performance leap—achieving a 12.2% gain on the Unseen split, far exceeding the 4.9% gain on the Seen split.
- RoBERTa is more suitable for LAVS than the CLIP text encoder—as the expressions in Ref-AVS represent control signals (e.g., "the one making sound") rather than specific semantics (e.g., "cat").
- The [cls] token accumulation strategy stably improves performance with virtually zero additional computational overhead.
- A single [seg] token is sufficiently effective; increasing the token count to 4/8 results in a decline on the Seen split but improves performance on the Unseen split—aligning with the insight that "extra tokens acting as registers to store global information benefit novel tasks but potentially harm known tasks".

## Highlights & Insights

- **Simple and Effective Design**: The core innovation lies in "fusion into token + prompting SAM2", establishing a clean design without complex multimodal fusion architectures.
- **Unified View of Knowledge Transfer**: Conceptually elegant, interpreting token propagation as forward knowledge transfer and token accumulation as backward knowledge transfer.
- **Asymmetric Training and Inference**: Supervised only on the first frame during training, while during inference SAM2 propagates across the entire video in a zero-shot manner, fully exploiting SAM2's pre-trained capabilities.
- **Substantial Gains in Unseen Categories** (+12.2%), which demonstrates that the generalization ability of SAM2 is successfully unlocked.

## Limitations & Future Work

- Performance under NULL settings (expressions pointing to non-existent objects) is poor (0.23), indicating the model struggles with "no target" scenarios.
- Supervised training is only applied to the first frame, which may lead to error accumulation across subsequent frames.
- The VGGish audio encoder is relatively legacy, which might limit audio understanding capabilities.
- Stronger audio encoders (e.g., AudioMAE) and end-to-end training of the SAM2 image encoder can be explored.

## Related Work & Insights

- **EEMC**: The only previous tri-modal Ref-AVS method, modeling all three modalities simultaneously but suffering from poor spatiotemporal consistency.
- **EVF-SAM**: A work extending SAM to text prompting, but limited to static images and lacking audio support.
- **GAVS**: A work extending SAM to audio-visual scenarios, but lacking text comprehension.
- **SAM2**: A powerful foundation for video segmentation, which this work extends into tri-modal scenarios.

## Rating

- **Novelty**: 7/10 — The core idea (compressing features into a token to prompt SAM2) is intuitive, though the token propagation/accumulation strategies show originality.
- **Experimental Thoroughness**: 8/10 — Evaluated comprehensively on the Ref-AVS benchmark with ablation studies covering each component, though conducted on only a single benchmark dataset.
- **Writing Quality**: 7/10 — The methodology is explained clearly, though certain symbols and mathematical concepts could be further refined.
- **Value**: 8/10 — Significantly pushes the SOTA on the Ref-AVS task, demonstrating the potential of SAM2 in multimodal scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Robust Audio-Visual Segmentation via Audio-Guided Visual Convergent Alignment](robust_audio-visual_segmentation_via_audio-guided_visual_convergent_alignment.md)
- [\[CVPR 2025\] Dynamic Derivation and Elimination: Audio Visual Segmentation with Enhanced Audio Semantics](dynamic_derivation_and_elimination_audio_visual_segmentation_with_enhanced_audio.md)
- [\[CVPR 2025\] A Distractor-Aware Memory for Visual Object Tracking with SAM2](a_distractor-aware_memory_for_visual_object_tracking_with_sam2.md)
- [\[CVPR 2025\] Revisiting Audio-Visual Segmentation with Vision-Centric Transformer](revisiting_audio-visual_segmentation_with_vision-centric_transformer.md)
- [\[ICCV 2025\] OmniSAM: Omnidirectional Segment Anything Model for UDA in Panoramic Semantic Segmentation](../../ICCV2025/segmentation/omnisam_omnidirectional_segment_anything_model_for_uda_in_panoramic_semantic_seg.md)

</div>

<!-- RELATED:END -->
