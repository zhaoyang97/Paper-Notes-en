---
title: >-
  [Paper Note] GLUS: Global-Local Reasoning Unified into A Single Large Language Model for Video Segmentation
description: >-
  [CVPR 2025][Segmentation][Video Object Segmentation] This paper proposes the GLUS framework, which unifies global understanding and local temporal consistency into a single MLLM through a frame partition strategy of "context frames (global reasoning) + query frames (local tracking)". Combined with an end-to-end trained VOS memory bank module, it significantly outperforms all MLLM-based methods on MeViS (J&F 51.3%).
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Video Object Segmentation"
  - "Multimodal Large Language Model"
  - "Global-Local Reasoning"
  - "Memory Bank"
  - "Referring Video Object Segmentation"
date: 2026-05-08
content_hash: 4ba203a2bdbe6212
---

# GLUS: Global-Local Reasoning Unified into A Single Large Language Model for Video Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2504.07962](https://arxiv.org/abs/2504.07962)  
**Code**: [https://glus-video.github.io/](https://glus-video.github.io/) (Project Page)  
**Area**: Segmentation  
**Keywords**: Video Object Segmentation, Multimodal Large Language Model, Global-Local Reasoning, Memory Bank, Referring Video Object Segmentation

## TL;DR
This paper proposes the GLUS framework, which unifies global understanding and local temporal consistency into a single MLLM through a frame partition strategy of "context frames (global reasoning) + query frames (local tracking)". Combined with an end-to-end trained VOS memory bank module, it significantly outperforms all MLLM-based methods on MeViS (J&F 51.3%).

## Background & Motivation

1. **Background**: Referring Video Object Segmentation (RefVOS) requires locating and consistently tracking target objects throughout an entire video based on linguistic descriptions. Recent approaches (such as VISA, VideoLISA, and ViLLa) introduce multimodal large language models (MLLMs) to RefVOS, aiming to leverage LLM reasoning capabilities to handle complex linguistic expressions.
2. **Limitations of Prior Work**: MLLMs have limited context windows (typically $N \approx 16$ frames), while the video frame count $T$ is much larger than $N$. Existing methods face a dilemma between "Ref" and "VOS"—using $N$ frames for global semantic understanding (uniform sampling) versus using $N$ frames for local continuous tracking (continuous sampling), making it impossible to satisfy both simultaneously. To compensate for the deficiency of either side, they rely on external VOS models or keyframe selectors.
3. **Key Challenge**: Global reasoning requires sparse frames covering the entire video to capture actions/attributes in descriptions, whereas local reasoning requires continuous frames to ensure temporal consistency. Within a limited context window, these two requirements present a direct conflict in frame allocation.
4. **Goal**: (1) How to achieve both global understanding and local tracking within a single MLLM? (2) How to eliminate reliance on external VOS models? (3) How to maximize information utilization efficiency within a limited context window?
5. **Key Insight**: The autoregressive nature of MLLMs is naturally compatible with streaming inference in VOS, where the prediction of the current frame is based on the results of preceding frames. Therefore, frames can simply be partitioned into two groups: sparse context frames providing global information (akin to a human quickly skimming the video), and continuous query frames that are labeled frame-by-frame for segmentation (akin to a human drawing masks for individual frames).
6. **Core Idea**: Explicitly partition the $N$ frames into "global context frames" and "local query frames", and integrate a pre-trained VOS memory bank in an end-to-end manner, enabling a single MLLM to possess both global understanding and local tracking capabilities.

## Method

### Overall Architecture
The input consists of $T$ video frames and a linguistic expression $R$, and the output is the segmentation masks for all $T$ frames. Based on the LISA-7B framework, GLUS partitions $N=8$ frames into 4 context frames (uniformly sampled from the video) and 4 query frames (continuously sampled video segments). Context frames are placed at the beginning to provide global semantics, while query frames follow to autoregressively generate $\langle\text{SEG}\rangle$ tokens. During inference, a sliding window is used to process all $T$ frames segment by segment, with the context frames remaining fixed across the entire video.

### Key Designs

1. **Global-Local Unification Strategy with Context and Query Frames**:
    - **Function**: Accommodate both global and local information simultaneously within the limited context window of a single MLLM.
    - **Mechanism**: Explicitly partition the $N$ frames into $N_C$ context frames (uniformly sampled across the entire video to cover global semantics) and $N_Q$ query frames (continuously sampled to enable local temporal reasoning). Within the LLM, context frames are positioned first, and query frames follow, interleaved with $\langle\text{SEG}\rangle$ tokens. The segmentation token for the $t$-th frame is generated as $\langle\text{SEG}\rangle_t = \text{LLM}([R, I^C_{1:N_C}, I^Q_1, \langle\text{SEG}\rangle_1, ..., I^Q_t])$. Training and inference are fully aligned: during training, context frames are uniformly sampled at random and query frames are sampled as random short segments; during inference, context frames are chosen as the central frames of each video segment, and query frames are traversed using a sliding window with a stride of 1.
    - **Design Motivation**: Prior methods often exhibit a discrepancy between training and inference strategies (e.g., VISA uses random sampling for training and uniform + continuous sampling for inference), leading to distribution mismatch. The training-inference alignment of GLUS serves as a critical advantage.

2. **End-to-End VOS Memory Bank Integration**:
    - **Function**: Overcome the context window limitation of MLLMs, store and utilize long-term historical information, and eliminate reliance on external VOS models.
    - **Mechanism**: Integrate SAM-2's memory attention module into the segmentation decoder in an end-to-end manner. Decoding the $t$-th query frame becomes $M_t = \text{Dec}(I^Q_t, \langle\text{SEG}\rangle_t, \text{MemBank})$, allowing gradients to backpropagate to the stored features in the memory bank and the pre-trained memory reading attention. During training, $N_Q$ query frames are utilized to simulate the streaming inference behavior of VOS.
    - **Design Motivation**: Prior methods (e.g., VISA, VideoLISA) invoke external VOS models for mask propagation only during inference, causing inconsistent training and inference paradigms. End-to-end training enables co-optimization of the MLLM and the VOS memory bank, aligning the training and inference distributions.

3. **Object Contrastive Loss**:
    - **Function**: Enhance the MLLM's ability to differentiate between objects with similar appearances at a fine-grained level, reducing mismatch cases where incorrect objects are aligned with linguistic descriptions.
    - **Mechanism**: Positive pairs are formed by $\langle\text{SEG}\rangle$ tokens generated from different language descriptions pointing to the same object (positive pairs can be sampled in approximately 91.5% of batches in MeViS), whereas tokens for different objects serve as negative pairs. A SimCLR-style contrastive loss is applied to maximize the distance between the $\langle\text{SEG}\rangle$ tokens of different objects, alongside maintaining a segmentation token bank to ensure a sufficient number of negative samples.
    - **Design Motivation**: In RefVOS, many videos contain multiple objects with similar appearances (e.g., "the elephant being attacked" needs to be distinguished from other elephants), and MLLMs easily generate similar $\langle\text{SEG}\rangle$ tokens for them. This loss is plug-and-play; it is applied solely to MeViS data but still offers performance gains on Ref-Youtube-VOS.

### Loss & Training
Standard cross-entropy loss is used for textual supervision (forcing the generation of the $\langle\text{SEG}\rangle$ token), and mask supervision follows SAM-2's per-pixel BCE + DICE loss, supplemented by the optional object contrastive loss. The MLLM is fine-tuned using LoRA, the SAM-2 decoder is trainable, and the backbone is frozen. Training is conducted on 4×A100 GPUs for about 25 hours (3000 steps, with batch size = 2 and gradient accumulation steps = 10). The context window is set to $N=8$ (4 context + 4 query frames), with 64 vision tokens per frame after 4x downsampling.

## Key Experimental Results

### Main Results

| Method | MeViS J&F | MeViS J | Ref-YT-VOS J&F |
|------|-----------|---------|-----------------|
| VISA-13B | 44.5 | 41.8 | 63.0 |
| VideoLISA-3.8B | 44.4 | 41.3 | 63.7 |
| ViLLa | - | - | 66.5 |
| DsHmp (non-LLM) | 46.4 | 43.0 | 67.1 |
| **GLUS-S (ours)** | **50.3** | **47.5** | **66.6** |
| **GLUS-A (ours)** | **51.3** | **48.5** | **67.3** |

On the highly challenging MeViS dataset, GLUS-A achieves a J&F of 51.3%, outperforming the previous state-of-the-art MLLM-based method by approximately **7 percentage points** and exceeding the non-LLM SOTA (DsHmp) by about 5 percentage points.

### Ablation Study

| Configuration | MeViS J&F | Explanation |
|------|-----------|------|
| Global-only (uniform) | 44.3 | Global frames only, without local reasoning |
| Local-only (continuous) | 44.4 | Local frames only, without global context |
| **GLUS (context+query)** | **48.2** | Unified global + local |
| + Memory Bank | **49.3** | Improvement of +1.1 with Memory Bank |
| + Contrastive Loss | **50.3** | Improvement of +1.0 with Contrastive Loss |
| + Key Frame Selector | 50.3→refined | Self-refinement framework |

### Key Findings
- **Unified global-local reasoning is the core contribution**: Jumping from 44.3/44.4 to 48.2 (+3.8/3.9) demonstrates that both styles of reasoning are indispensable.
- **End-to-end training of the memory bank is a significant enhancement**: An increase of +1.1 J&F indicates that the MLLM successfully learns to coordinate with the VOS memory module, making external VOS models obsolete.
- **Contrastive loss is extremely effective on MeViS**: Yielding +1.0 J&F, this is because scenarios in MeViS featuring multiple objects and dynamic descriptions especially demand fine-grained differentiation. Although the contrastive loss is applied strictly to MeViS, it also brings indirect gains to Ref-Youtube-VOS.
- GLUS uses less SFT data than VISA/ViLLa (only MeViS + Ref-YT-VOS) yet yields superior performance, showcasing the high efficiency of the architectural design.

## Highlights & Insights
- **Importance of Training-Inference Alignment**: GLUS represents the first MLLM method in RefVOS to achieve fully aligned frame-sampling strategies between training and inference. This seemingly simple design principle yields significant performance gains (compared to VISA's random training / uniform + continuous inference).
- **VOS Memory Bank as a Plug-and-Play Enhancement for MLLMs**: The scheme of end-to-end integrating SAM-2's memory module is elegant—it leverages the temporal tracking capability of VOS pre-training while allowing the MLLM to learn how to "utilize" memory through gradient backpropagation. This paradigm can be extended to other MLLM tasks requiring long-sequence reasoning.
- **"Human Simulation" Design Intuition**: The workflow of first scanning quickly (context frames) and then labeling frame-by-frame (query frames) aligns with human annotation habits, offering a clear and intuitive design philosophy.

## Limitations & Future Work
- The context window remains constrained to $N=8$ frames ($4+4$). In longer videos, the sampling density of context frames is insufficient, potentially leading to missed key actions.
- Only 64 vision tokens are allocated per frame (due to 4x downsampling), limiting the spatial resolution and constraining the segmentation quality of fine boundaries.
- Computational resource constraints meant that the Additional-SFT version did not incorporate the contrastive loss and keyframe selector, leaving the performance gains of fully stacked components under-explored.
- The contrastive loss depends on the characteristic of multiple referring expressions pointing to the same object in MeViS, making it difficult to directly apply to datasets lacking multi-description annotations.
- The pseudo-labels of the keyframe selector originate from GLUS's own predicted IoUs, posing a risk of self-reinforcement bias.

## Related Work & Insights
- **vs VISA**: VISA uses random sampling for training, and uniform + continuous + external VOS for inference. GLUS unifies training and inference and operates without external VOS, resulting in a cleaner system and superior performance.
- **vs ViLLa**: ViLLa focuses on local reasoning (continuous sampling) supplemented by a context-aggregation module. GLUS introduces global information via explicit context frames, showing a distinct edge on MeViS.
- **vs VideoLISA**: VideoLISA tracks the entire video utilizing a single token, which provides strong global reasoning but weak local reasoning. GLUS's multi-token + memory bank scheme yields better temporal consistency.
- The concept of end-to-end integrating a VOS memory bank can be generalized to other video-level MLLM tasks (e.g., VideoQA, video understanding), making it a valuable direction to follow.

## Rating
- Novelty: ⭐⭐⭐⭐ The design of context frames + query frames is simple yet highly effective, and the end-to-end integration of the VOS memory bank is a meaningful contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across multiple datasets including MeViS, Ref-YT-VOS, ReVOS, and ReasonVOS, with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Thorough analysis of the problem, and Table 1 provides a clear, at-a-glance comparison of the frame sampling strategies of prior methods.
- Value: ⭐⭐⭐⭐ Establishes a strong baseline for MLLM-based methods in RefVOS, delivering substantial performance improvements on the most challenging MeViS dataset.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] VISA: Reasoning Video Object Segmentation via Large Language Models](../../ECCV2024/segmentation/visa_reasoning_video_object_segmentation_via_large_language_models.md)
- [\[CVPR 2025\] StoryGPT-V: Large Language Models as Consistent Story Visualizers](storygpt-v_large_language_models_as_consistent_story_visualizers.md)
- [\[CVPR 2025\] Uni4D: Unifying Visual Foundation Models for 4D Modeling from a Single Video](uni4d_unifying_visual_foundation_models_for_4d_modeling_from_a_single_video.md)
- [\[CVPR 2025\] The Devil is in Temporal Token: High Quality Video Reasoning Segmentation](the_devil_is_in_temporal_token_high_quality_video_reasoning_segmentation.md)
- [\[CVPR 2025\] DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment](dinov2_meets_text_a_unified_framework_for_image-_and_pixel-level_vision-language.md)

</div>

<!-- RELATED:END -->
