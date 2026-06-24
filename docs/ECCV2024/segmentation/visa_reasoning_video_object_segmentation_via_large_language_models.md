---
title: >-
  [Paper Note] VISA: Reasoning Video Object Segmentation via Large Language Models
description: >-
  [ECCV 2024][Segmentation][Reasoning Segmentation] Proposes the new ReasonVOS task and the VISA model, utilizing the world knowledge reasoning capabilities of multi-modal LLMs to achieve video object segmentation and tracking based on implicit text queries.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Reasoning Segmentation"
  - "Video Object Segmentation"
  - "Multi-Modal LLM"
  - "SAM"
  - "Video Reasoning"
date: 2026-05-08
content_hash: 1e3c411a113a85ec
---

# VISA: Reasoning Video Object Segmentation via Large Language Models

**Conference**: ECCV 2024  
**arXiv**: [2407.11325](https://arxiv.org/abs/2407.11325)  
**Code**: [https://github.com/cilinyan/VISA](https://github.com/cilinyan/VISA)  
**Area**: Image Segmentation  
**Keywords**: Reasoning Segmentation, Video Object Segmentation, Multi-Modal LLM, SAM, Video Reasoning  

## TL;DR

Proposes the new ReasonVOS task and the VISA model, utilizing the world knowledge reasoning capabilities of multi-modal LLMs to achieve video object segmentation and tracking based on implicit text queries.

## Background & Motivation

**Background**: Existing video object segmentation (VOS) methods rely on explicit user instructions—categories, masks, or phrases (e.g., "a running car"), and their capabilities are limited to intuitively visible feature descriptions.

**Limitations of Prior Work**: Actual user requirements are often implicit (e.g., "find my favorite mug"), which requires the model to possess common-sense reasoning and video content understanding capabilities—abilities that existing methods completely lack.

**Key Challenge**: While reasoning segmentation has been explored in the image domain by works such as LISA, the video domain requires simultaneous processing of temporal information and spatial details. Since each frame requires a large number of visual tokens, a direct extension to the video domain is computationally infeasible.

**Goal**: How to achieve object segmentation in videos that requires world knowledge reasoning—given implicit, complex text instructions (e.g., "electric car", "which car is most likely to win the race"), and outputting the corresponding sequence of object masks.

**Key Insight**: Instead of processing all frames, a Text-guided Frame Sampler is first used to select keyframes, reducing the number of tokens. Then, a multi-modal LLM simultaneously processes multiple frames for reasoning, combined with a SAM decoder for segmentation and a tracker for propagation.

**Core Idea**: Through a pipeline consisting of text-guided keyframe sampling + multi-modal LLM reasoning + SAM decoding + object tracking, video reasoning segmentation is efficiently achieved.

## Method

### Overall Architecture

VISA consists of three core components, forming an elegant pipeline:

1. **Text-guided Frame Sampler (TFS)**: Inputs the full video + text description $\rightarrow$ outputs a target frame $f_{tgt}$ and $T_r$ reference frames $\mathbf{x}_r$.
2. **Multi-Modal LLM**: Concatenates the target frame, reference frames, and text tokens as input to the LLM $\rightarrow$ outputs text containing a special `<Seg>` token $\rightarrow$ extracts the hidden embedding of `<Seg>`, which is projected via an MLP to the prompt embedding of SAM $\rightarrow$ the SAM decoder decodes the segmentation mask $m_{tgt}$ of the target frame.
3. **Object Tracker (XMem)**: Propagates $m_{tgt}$ bidirectionally to all remaining frames $\rightarrow$ yields the complete mask sequence $\mathcal{M}$.

### Key Designs

**1. Text-guided Frame Sampler (TFS) —— Mechanism**

- **Function**: Selects the target frame and reference frames most relevant to the text query from a long video.
- **Mechanism**: Utilizes LLaMA-VID (which compresses each frame into 2 tokens to process long videos) as the sampler. A prompt template is designed: "To find {description}, which percentage mark of the video should I check?", allowing LLaMA-VID to generate Top-K=10 responses, and takes the average of the percentages to locate the target frame $f_{tgt}$.
- **Design Motivation**: Most frames in a video are irrelevant to the query (e.g., asking "who will take the baton" only requires inspecting the last few frames). Processing all frames directly with $T \times L$ tokens is computationally infeasible. TFS preserves spatial details (without spatial pooling) while drastically reducing the number of frames to be processed.
- **Reference Frame Sampling Strategy**: Samples $T_r$ reference frames based on $f_{tgt}$, supporting three strategies: Global (uniform sampling across the entire video), Local (consecutive sampling centered around $f_{tgt}$), and Global-Local (a combined strategy taking half from each).

**2. Multi-Modal LLM + `<Seg>` Token Mechanism —— Mechanism**

- **Function**: Performs joint vision-language reasoning on the selected frames and outputs an embedding that can be decoded into a segmentation mask.
- **Mechanism**: Employs Chat-UniVi (a multi-modal LLM supporting flexible visual token counts) as the backbone. Each frame is encoded by ViT and then processed by Spatial Merging to obtain $L=112$ visual tokens. The target frame, reference frames, and text tokens are concatenated and input into the LLM to generate a response containing `<Seg>`. The last-layer embedding of `<Seg>` is extracted and projected to $h_{seg}$ via an MLP, serving as the prompt embedding for the SAM decoder.
- **Design Motivation**: Following LISA's `<Seg>` token design, this design decouples the reasoning capability of the language model from the segmentation capability—the LLM is responsible for reasoning "what it is," and the SAM decoder is responsible for precisely segmenting "where it is." Simultaneously processing multiple frames captures temporal information, which is superior to the frame-by-frame processing in TrackGPT.

**3. Object Tracker (XMem) —— Mechanism**

- **Function**: Propagates the segmentation mask of the target frame to all frames in the video.
- **Mechanism**: Uses a frozen XMem for bidirectional propagation: $\mathcal{M} = \text{OT}(m_{tgt}, \mathbf{x}_v)$.
- **Design Motivation**: Accurate segmentation is only required for a single frame, leveraging a mature VOS tracker to complete temporally consistent segmentation across the whole video, thereby avoiding the high computational overhead of reasoning on every frame.

### Loss & Training

**Loss Function**: $\mathcal{L} = \lambda_{txt} \mathcal{L}_{txt} + \lambda_{mask} \mathcal{L}_{mask}$

- $\mathcal{L}_{txt}$: Autoregressive cross-entropy loss (text generation)
- $\mathcal{L}_{mask} = \lambda_{bce} \text{BCE}(\hat{m}, m) + \lambda_{dice} \text{DICE}(\hat{m}, m)$
- Weight settings: $\lambda_{txt}=1.0$, $\lambda_{mask}=1.0$, $\lambda_{bce}=2.0$, $\lambda_{dice}=0.5$

**Training Strategy**:

- Frozen modules: TFS, Vision Backbone, and Object Tracker are all frozen.
- Trainable modules: Multi-Modal LLM (efficiently fine-tuned via LoRA) + SAM Decoder.
- During training, the target frame and 8-12 reference frames are randomly sampled (without using TFS), and TFS is only used for frame selection during inference.
- Training data mixture: Referring VOS (Ref-YouTube-VOS, MeViS, Ref-DAVIS17) + Video QA + Image data + ReVOS.
- Hardware: 8×A100 80G, batch size 128, 10 epochs, AdamW + cosine schedule, lr=2e-5.

## Key Experimental Results

### Main Results

**Table 1: Performance comparison on the ReVOS dataset**

| Method | Backbone | Referring J&F | Reasoning J&F | Overall J&F | R (Robustness) |
|------|----------|:---:|:---:|:---:|:---:|
| ReferFormer | Video-Swin-B | 32.7 | 23.4 | 28.1 | 8.8 |
| LISA | LLaVA-7B | 45.7 | 36.1 | 40.9 | 9.3 |
| TrackGPT(IT) | LLaVA-7B | 48.2 | 39.0 | 43.6 | 11.6 |
| TrackGPT(IT) | LLaVA-13B | 49.5 | 40.5 | 45.0 | 12.8 |
| **VISA(IT)** | Chat-UniVi-7B | 50.9 | 43.0 | 46.9 | 15.5 |
| **VISA(IT)** | Chat-UniVi-13B | **57.4** | **44.3** | **50.9** | 14.5 |

**Table 2: Performance comparison on Referring VOS datasets**

| Method | Backbone | MeViS J&F | Ref-YT-VOS J&F | Ref-DAVIS17 J&F |
|------|----------|:---:|:---:|:---:|
| ReferFormer | Video-Swin-B | 31.0 | 62.9 | 61.1 |
| LISA | LLaVA-13B | 37.9 | 54.4 | 66.0 |
| TrackGPT | LLaVA-13B | 41.2 | 59.5 | 66.5 |
| **VISA** | Chat-UniVi-7B | 43.5 | 61.5 | 69.4 |
| **VISA** | Chat-UniVi-13B | **44.5** | **63.0** | **70.4** |

### Ablation Study

**Table 5: Ablation of target frame selection and reference frame sampling strategies (ReVOS Overall J&F)**

| Target Frame | $T_r$ | No Sampling | Global | Local | Global-Local |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $f_0$ (First frame) | 0 | 42.6 | - | - | - |
| $f_0$ | 12 | - | 44.5 | 44.9 | 45.0 |
| $f_{tgt}$ (TFS) | 0 | 44.3 | - | - | - |
| $f_{tgt}$ | 12 | - | 46.7 | 46.3 | **46.9** |

- Using TFS for frame selection improves over using the first frame directly by ~2%.
- The Global-Local combination strategy is the most optimal.
- 12 reference frames perform better than 6 frames.

**Table 4: Ablation on training datasets (ReVOS, Chat-UniVi-7B)**

| ReferVOS | VQA | Image | ReVOS | Referring J&F | Reasoning J&F |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ✗ | ✓ | ✓ | ✓ | 47.6 | 39.9 |
| ✓ | ✓ | ✗ | ✓ | 34.2 | 33.5 |
| ✓ | ✓ | ✓ | ✗ | 52.9 | 39.2 |
| ✓ | ✓ | ✓ | ✓ | 50.9 | **43.0** |

### Key Findings

- Traditional methods (e.g., ReferFormer) achieve only ~23% J&F on reasoning tasks, whereas VISA(IT) reaches 44.3%, representing an improvement of over 20 points.
- VISA's multi-frame processing is superior to TrackGPT's frame-by-frame processing, achieving a 3.3% higher overall J&F (7B model).
- Image datasets are crucial for training—performance drops drastically by 16.7% (referring) / 9.5% (reasoning) when removed, as image data scale is much larger than video data.
- Instruction tuning on the ReVOS training set brings a 3.8% improvement to reasoning, while the referring performance remains almost unchanged.
- Having $L=112$ and $L=256$ visual tokens per frame yields comparable performance, but $L=56$ leads to a significant decline.
- Training with negative samples that contain non-existent targets increases the robustness score R to 15.5 (vs TrackGPT's 11.6).

## Highlights & Insights

- **Forward-looking Task Definition**: ReasonVOS is a valuable new task definition—extending VOS from explicit instructions to implicit instructions that require world knowledge reasoning, which aligns better with the real-world requirements of Embodied AI.
- **Elegant and Decoupled Architecture Design**: TFS frame selection $\rightarrow$ LLM reasoning $\rightarrow$ SAM segmentation $\rightarrow$ Tracker propagation. The four modules play their respective roles. Freezing the modules that do not require training (TFS, Backbone, Tracker), and only training the LLM (LoRA) and SAM decoder, makes training extremely efficient.
- **`<Seg>` Token Bridging Mechanism**: Inheriting the design of LISA, a special token is used to seamlessly bridge the reasoning capability of the LLM to the segmentation capability, avoiding the issue where the segmentation model itself needs to reason.
- **Solid Dataset Contributions**: ReVOS contains implicit reasoning descriptions (14,678), explicit descriptions (20,071), and non-existent target descriptions (325), providing comprehensive coverage, and the introduction of negative samples effectively reduces hallucinations.
- **TFS Substitution Strategy during Training**: Randomly sampling frames instead of using TFS during training exposes the model to more combinations of frames. TFS is then used for precise frame selection during inference—this discrepancy in train/test strategy is reasonably designed.

## Limitations & Future Work

- **Small Object Issue**: With only 112 visual tokens per frame, the segmentation of small targets (e.g., a paddle) is poor, whereas increasing the token count would introduce computational burdens.
- **Strong Dependency on Keyframe Localization**: Inaccurate localization by TFS directly leads to segmentation failure, e.g., targets that only appear in a single frame (like a person holding a fire extinguisher) are difficult to locate.
- **Limited Long-term Temporal Modeling**: Although multiple reference frames are sampled, 12 frames are still limited, making the model inadequate for queries that require extremely long-term temporal understanding.
- **Replaceable Backbone**: Currently based on Chat-UniVi, the backbone can be replaced with stronger multi-modal LLMs (such as InternVL, Qwen-VL) to achieve continuous improvements in the future.
- **Error Accumulation in Tracker**: The bidirectional propagation of XMem may drift in long videos or complex scenes, resulting in a decline in segmentation quality over propagation distance.

## Related Work & Insights

- **vs LISA**: VISA is a natural extension of LISA in the video domain. The core difference lies in the introduction of TFS and Tracker to handle temporal issues; VISA also achieves performance comparable to LISA on image tasks.
- **vs TrackGPT**: TrackGPT processes frame-by-frame and lacks temporal information, whereas VISA processes multiple frames simultaneously for reasoning, achieving an overall J&F that is 5.9 points higher (13B model).
- **vs LLaMA-VID + LMPM**: The issue with the VQA+ReferVOS pipeline is that the low token count of VQA models is insufficient to accurately locate targets, leading to error accumulation across the two stages.
- **vs Video-ChatGPT**: Token compression methods via spatial/temporal pooling lose the spatial details required for segmentation, whereas VISA preserves spatial resolution via TFS frame selection.

## Rating

- ⭐⭐⭐⭐ **Novelty**: The new ReasonVOS task definition is valuable, and the TFS + LLM + SAM + Tracker pipeline is simple and effective. However, individual components are combinations of existing modules.
- ⭐⭐⭐⭐ **Experimental Thoroughness**: Extensive experiments across 8 datasets and multiple ablation groups (datasets/frame sampling/token count/sampling strategy), with comprehensive horizontal and vertical comparisons.
- ⭐⭐⭐⭐ **Writing Quality**: The structure is clear, the task definition is distinct, the figures and tables are informative, and the motivation is well-articulated.
- ⭐⭐⭐⭐⭐ **Value**: Opens up the Reasoning VOS direction, fills a gap with the ReVOS dataset, and matches the release of open-source code and datasets, providing a significant boost to subsequent research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] CoReS: Orchestrating the Dance of Reasoning and Segmentation](cores_orchestrating_the_dance_of_reasoning_and_segmentation.md)
- [\[CVPR 2025\] GLUS: Global-Local Reasoning Unified into A Single Large Language Model for Video Segmentation](../../CVPR2025/segmentation/glus_global-local_reasoning_unified_into_a_single_large_language_model_for_video.md)
- [\[ECCV 2024\] ActionVOS: Actions as Prompts for Video Object Segmentation](actionvos_actions_as_prompts_for_video_object_segmentation.md)
- [\[ECCV 2024\] OpenPSG: Open-set Panoptic Scene Graph Generation via Large Multimodal Models](openpsg_open-set_panoptic_scene_graph_generation_via_large_multimodal_models.md)
- [\[ECCV 2024\] A Semantic Space is Worth 256 Language Descriptions: Make Stronger Segmentation Models with Descriptive Properties](a_semantic_space_is_worth_256_language_descriptions_make_str.md)

</div>

<!-- RELATED:END -->
