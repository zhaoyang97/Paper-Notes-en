---
title: >-
  [Paper Note] The Devil is in Temporal Token: High Quality Video Reasoning Segmentation
description: >-
  [CVPR 2025][Segmentation][Video Reasoning Segmentation] VRS-HQ proposes hierarchical temporal token encoding (frame-level `<SEG>` + video-level `<TAK>`) and a token-driven keyframe selection strategy. Incorporating SAM2, it achieves end-to-end video reasoning segmentation, outperforming VISA by 9.1% on ReVOS.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Video Reasoning Segmentation"
  - "Temporal Token Aggregation"
  - "Keyframe Selection"
  - "SAM2"
  - "Multimodal Large Language Models"
date: 2026-05-08
content_hash: 497270b9c4fd2601
---

# The Devil is in Temporal Token: High Quality Video Reasoning Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2501.08549](https://arxiv.org/abs/2501.08549)  
**Code**: [VRS-HQ](https://github.com/sitong-gong/VRS-HQ)  
**Area**: Image Segmentation  
**Keywords**: Video Reasoning Segmentation, Temporal Token Aggregation, Keyframe Selection, SAM2, Multimodal Large Language Models

## TL;DR

VRS-HQ proposes hierarchical temporal token encoding (frame-level `<SEG>` + video-level `<TAK>`) and a token-driven keyframe selection strategy. Incorporating SAM2, it achieves end-to-end video reasoning segmentation, outperforming VISA by 9.1% on ReVOS.

## Background & Motivation

Video Reasoning Segmentation (VRS) requires models to generate video-level segmentation masks based on complex implicit textual intent. Existing methods face three major bottlenecks:
- **Limited Temporal Context**: Methods like VISA and VideoLISA rely on a single `<SEG>` token to represent target objects in keyframes or the entire video, failing to fully capture inter-frame variations and spatio-temporal features.
- **Inaccurate Keyframe Detection**: VISA utilizes an external model (LLaMA-VID) to detect keyframes, which may yield inaccurate keyframe predictions in complex temporal reasoning scenarios, thereby impairing downstream segmentation.
- **Decoupled Segmentation and Propagation**: VISA separately employs SAM (for keyframe segmentation) and XMem (for mask propagation), preventing end-to-end training and inference.
- The representation capability of a single token is insufficient to simultaneously encode intra-frame spatial details and inter-frame temporal dynamics.
- A unified solution is required to seamlessly perform temporal reasoning, keyframe selection, segmentation, and propagation within a single pipeline.

## Method

### Overall Architecture

VRS-HQ consists of four modules: (1) MLLM-based temporal token encoding using Chat-UniVi (generating frame-level `<SEG>` and video-level `<TAK>` tokens); (2) Temporal Dynamic Aggregation (TDA) to fuse frame-level features into temporal tokens; (3) Token-driven Keyframe Selection (TKS) utilizing token similarity and SAM2 occlusion scores to select keyframes; (4) SAM2 to execute keyframe segmentation and mask propagation.

### Key Designs

#### Key Design 1: Hierarchical Temporal Token Encoding

**Function**: Captures frame-level spatial information and video-level temporal semantics respectively, providing rich spatial-temporal features for the segmentation model.

**Mechanism**: The MLLM vocabulary is expanded to introduce two special tokens: frame-level `<SEG>` and video-level `<TAK>`. A structured conversation template is designed: "Please find {expression} in the Reference Video and segment it in each frame and the entire video respectively." The MLLM learns via autoregression to generate a response containing multiple `<SEG>` tokens and one `<TAK>` token. The embeddings from the last layer of the MLLM, $\bar{h}_{seg} \in \mathbb{R}^{T' \times d'}$ and $\bar{h}_{tak} \in \mathbb{R}^{1 \times d'}$, are extracted and mapped to the feature space of SAM2 via MLPs.

**Design Motivation**: A single token is insufficient to encode both spatial details and temporal coherence. Hierarchical tokens allow the model to learn local and global information separately, combining the strengths of both through fusion.

#### Key Design 2: Temporal Dynamic Aggregation (TDA)

**Function**: Performs weighted fusion based on cosine similarity to integrate frame-level spatial features into the temporal token while maintaining the temporal consistency of the target.

**Mechanism**: The cosine similarity between each frame-level `<SEG>` token and the video-level `<TAK>` token is computed and normalized into weights $\lambda_i$. The fusion formula is $h'_{tak} = h_{tak} + \alpha \sum_{i=1}^{T'} \lambda_i h_{seg}[i]$, where $\alpha$ is the fusion coefficient. Higher-similarity frames contribute larger weights, naturally injecting spatial details from representative frames into the temporal token. During training, the frame with the highest similarity is selected as the keyframe.

**Design Motivation**: Simple average pooling blurs target spatial details, whereas similarity-based weighted fusion biases temporal tokens toward the most representative frames while preserving global semantics.

#### Key Design 3: Token-driven Keyframe Selection (TKS)

**Function**: Eliminates the need for external keyframe detection models during inference, utilizing SAM2 occlusion scores to assist in accurate keyframe selection.

**Mechanism**: In the inference phrase, the CLIP model is employed to locate the frame that matches the textual expression best as the global sampling anchor. Each sampled frame is treated as a candidate keyframe and passed to SAM2 along with the fused `<TAK>` embedding to calculate the occlusion score $S_o = \mathcal{MD}(\mathcal{E}(\mathcal{X}_V^f), h'_{tak})$. The final keyframe is determined by combining the softmax-normalized occlusion score $S'_o$ and the token similarity score $S_t$.

**Design Motivation**: Errors from external keyframe detection models propagate to segmentation results; SAM2's occlusion scores naturally reflect the target's visibility and confidence in the current frame, making them an ideal metric for keyframe selection.

### Loss & Training

End-to-end training loss: $L_{total} = \lambda_{txt} L_{txt} + \lambda_{mask} L_{mask}$, where $L_{mask} = \lambda_{bce} L_{bce} + \lambda_{dice} L_{dice}$. The weight settings are: $\lambda_{txt}=1, \lambda_{mask}=1, \lambda_{bce}=2, \lambda_{dice}=0.5$.

## Key Experimental Results

### Main Results: ReVOS Video Reasoning Segmentation Benchmark

| Method | Backbone | Referring J&F | Reasoning J&F | Overall J&F |
|------|----------|---------------|---------------|-------------|
| **VRS-HQ (7B)** | Chat-UniVi-7B | **62.1** | **56.1** | **59.1** |
| VISA (13B) | Chat-UniVi-13B | 57.4 | 44.3 | 50.9 |
| VISA (7B) | Chat-UniVi-7B | 50.9 | 43.0 | 46.9 |
| TrackGPT (13B) | LLaVA-13B | 49.5 | 40.5 | 45.0 |
| LISA (13B) | LLaVA-13B | 46.6 | 36.7 | 41.6 |

### Ablation Study: Contribution of Components (ReVOS Overall J&F)

| Setup | J&F |
|------|-----|
| VRS-HQ (Full) | **59.1** |
| Only `<SEG>` token (w/o TDA) | Lower |
| w/o TKS (External keyframe selection) | Lower |
| w/o SAM2 propagation (Using XMem) | Lower |

### Key Findings

- VRS-HQ (7B) outperforms VISA-13B by **9.1%** J&F on ReVOS, improving by 5.9%/12.5%/9.1% on referring/reasoning/overall subsets respectively.
- It also outperforms other models on three standard RVOS datasets by 7.3%/5.6%/6.5% respectively.
- The weighted fusion of TDA is significantly better than simple average fusion.
- SAM2's occlusion score serves as an effective cue for keyframe selection, eliminating reliance on external models.

## Highlights & Insights

- **Hierarchical Token Design**: The "frame-level + video-level" hierarchical encoding concept can be generalized to other video understanding tasks.
- **End-to-End Inference**: Utilizing SAM2 for unified segmentation and propagation eliminates the multi-model cascading issues found in VISA.
- **Token-Driven Keyframe Selection**: Leveraging the model's own temporal understanding rather than relying on external models is both more elegant and robust.

## Limitations & Future Work

- Relies on the quality of SAM2's occlusion scores; when SAM2's understanding of the target object is inaccurate, keyframe selection may be affected.
- The number of candidate keyframes is constrained by the number of sampled frames, potentially missing optimal keyframes.
- In ultra-long video scenarios, the processing capacity of the MLLM for a large number of frames might be limited.
- Multi-target simultaneous reasoning and segmentation can be explored in future work.

## Related Work & Insights

- Comparisons with VISA indicate that a single token is the core bottleneck in video reasoning segmentation, and hierarchical tokens serve as an effective solution.
- SAM2's promptable segmentation + cross-frame propagation capabilities make it an ideal backend for video segmentation.
- Enhancing keyframe selection strategies significantly impacts the final segmentation quality, which is worth exploring in other video tasks.

## Rating

⭐⭐⭐⭐ — Clearly identifies the core bottlenecks of existing VRS methods (single token + external keyframe detection), and proposes hierarchical token and TKS strategies that simplify the inference pipeline while dramatically boosting performance. Achieving these results at a 7B model scale, outperforming the 13B baseline, is particularly impressive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ROS-SAM: High-Quality Interactive Segmentation for Remote Sensing Moving Object](ros-sam_high-quality_interactive_segmentation_for_remote_sensing_moving_object.md)
- [\[CVPR 2025\] Exploiting Temporal State Space Sharing for Video Semantic Segmentation](exploiting_temporal_state_space_sharing_for_video_semantic_segmentation.md)
- [\[CVPR 2025\] Mask-Adapter: The Devil is in the Masks for Open-Vocabulary Segmentation](mask-adapter_the_devil_is_in_the_masks_for_open-vocabulary_segmentation.md)
- [\[CVPR 2025\] Image Quality Assessment: From Human to Machine Preference](image_quality_assessment_from_human_to_machine_preference.md)
- [\[CVPR 2025\] The Devil is in Low-Level Features for Cross-Domain Few-Shot Segmentation](the_devil_is_in_low-level_features_for_cross-domain_few-shot_segmentation.md)

</div>

<!-- RELATED:END -->
