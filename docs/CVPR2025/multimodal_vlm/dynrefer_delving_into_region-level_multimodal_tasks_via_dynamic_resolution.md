---
title: >-
  [Paper Note] DynRefer: Delving into Region-level Multimodal Tasks via Dynamic Resolution
description: >-
  [CVPR 2025][Multimodal VLM][Region-level understanding] Modeling the dynamic resolution mechanism of human "fixation + saccade", this work constructs multi-level nested views around target regions with random sampling during training and selective combination during inference based on task or image priors, outperforming 7B+ models on region captioning, attribute detection, and dense captioning with only 4.2B parameters.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Region-level understanding"
  - "dynamic resolution"
  - "multi-view fusion"
  - "human gaze bionics"
  - "dense captioning"
date: 2026-05-08
content_hash: 7e1d9e6819feceac
---

# DynRefer: Delving into Region-level Multimodal Tasks via Dynamic Resolution

**Conference**: CVPR 2025  
**arXiv**: [2405.16071](https://arxiv.org/abs/2405.16071)  
**Code**: [https://github.com/callsys/DynRefer](https://github.com/callsys/DynRefer)  
**Area**: Multimodal VLMs  
**Keywords**: Region-level understanding, dynamic resolution, multi-view fusion, human gaze bionics, dense captioning

## TL;DR
Modeling the dynamic resolution mechanism of human "fixation + saccade", this work constructs multi-level nested views around target regions with random sampling during training and selective combination during inference based on task or image priors, outperforming 7B+ models on region captioning, attribute detection, and dense captioning with only 4.2B parameters.

## Background & Motivation

**Background**: Region-level multimodal tasks (region captioning, attribute detection, region identification) are critical applications of VLMs. Existing methods such as GLaMM, RegionGPT, and Alpha-CLIP employ fixed resolutions to process target regions, feeding either cropped region images or the entire image uniformly into the visual encoder.

**Limitations of Prior Work**: Different tasks exhibit widely varying requirements for the contextual information surrounding a region—attribute detection requires focusing on fine-grained details of the region itself (e.g., color, texture), whereas region captioning necessitates context to understand spatial relationships and scene semantics. Fixed-resolution strategies fail to satisfy both requirements simultaneously: cropping too tightly discards context, while cropping too widely introduces irrelevant noise.

**Key Challenge**: The trade-off between "fine-grained detail focus" and "contextual understanding" in region-level tasks. While the human eye resolves this naturally via high-resolution foveal vision and low-resolution peripheral vision, existing models lack such adaptive capabilities.

**Goal**: To enable the model to dynamically adjust its "viewing mode" for target regions based on task demands, resembling human vision—focusing closely when details are needed, and zooming out when context is required.

**Key Insight**: Building multiple nested views around target regions using an interpolation coefficient $t \in [0,1]$ ($t=0$ denotes the region only, and $t=1$ denotes the full image), randomly sampling views during training to simulate dynamic foveation, and selecting the optimal combination of views during inference based on task types or image information density.

**Core Idea**: Replacing fixed-resolution region processing with multi-level nested view random sampling training and selective inference, achieving task-adaptive region understanding with dynamic resolution.

## Method

### Overall Architecture
The input consists of an image and a bounding box of the target region. Multiple nested views (with varying scale factors) are constructed around the region. Each view is resized to 224×224 and encoded by a frozen ViT. Region representations are extracted using RoI-Align, aligned via the Align Module, concatenated and fused, and then compressed into a fixed-length region representation $x_v$ using a Q-Former. These representations are simultaneously fed into three decoders (tagging, contrastive, LLM) to align vision and language.

### Key Designs

1. **Nested View Construction and Randomly Sampled Training**:

    - **Function**: Simulates the dynamic resolution foveation mechanism of the human eye.
    - **Mechanism**: The interpolation coefficient $t$ controls the crop range, where $t=0$ yields a tight crop of only the target region, and $t=1$ yields the full image. During training, $n$ views are randomly sampled (always including the region view with $t_1=0$) to form the multi-view input. This randomness forces the model to learn to extract useful features from different scale levels.
    - **Design Motivation**: Training with fixed dual-views is less effective than random sampling (Ablation Study line 5 vs 6), as randomness increases training data diversity and enhances model robustness across varying context levels.

2. **Dynamic Convolutional Spatial Alignment Module (Align Module)**:

    - **Function**: Corrects spatial misalignment between different views caused by cropping and scaling.
    - **Mechanism**: The region feature $r_i$ from each view is concatenated with the reference feature $r_1$ ($t=0$). A convolutional layer calculates a 2D offset map, and features are resampled based on these offsets. Inspired by dynamic convolution, this module adaptively aligns features from different zoom levels using learned offsets.
    - **Design Motivation**: Since the same region occupies entirely different scales and locations within the 224×224 input under different $t$ values, directly concatenating features would cause spatial semantic misalignment.

3. **Selectively Multimodal Referring**:

    - **Function**: Selects the optimal combination of views during inference based on priors.
    - **Mechanism**: Offers two modes—(1) *Task Prior*: When the task type is known, empirical optimal $t$ values are selected (e.g., tight views with $t_2=0.1$ for attribute detection, medium context with $t_2=0.4-0.5$ for captioning); (2) *Image Prior*: When the task is unknown, Perceptual Hashing (pHASH) greedy search is used to maximize the informational discrepancy between views:
      $$\frac{\sum \text{pHASH}(x(t_1)) \oplus \text{pHASH}(x(t_i))}{t_i}$$
      where $1/t_i$ downweights overly contextualized views.
    - **Design Motivation**: Different tasks require different levels of context, making adaptive selection during inference more effective than a uniform strategy.

### Loss & Training
Three decoders are jointly trained: $D_{tag}$ conducts multi-label tag prediction using asymmetric loss (4585 predefined tags); $D_{rtc}$ performs region-text alignment using a Sigmoid contrastive loss; and $D_{llm}$ generates region captions using FlanT5-XL with a cross-entropy loss. These three components mutually reinforce each other—removing any single decoder degrades performance on the remaining tasks, as shown in the ablation studies.

## Key Experimental Results

### Main Results

| Method | Params | RefCOCOg CIDEr | VG CIDEr | OVAD mAP | COCO Acc |
|------|------|---------------|----------|----------|----------|
| GLaMM | 7.4B | 106.0 | 180.5 | - | - |
| RegionGPT | 7.4B | 109.9 | 145.6 | - | 80.6 |
| ControlCap | 4.2B | 111.4 | 181.9 | - | - |
| Alpha-CLIP | 7.4B | 109.2 | 160.3 | - | - |
| **DynRefer** | **4.2B** | **115.7** | **190.9** | **29.2** | **89.4** |

### Ablation Study

| Configuration | OVAD mAP | COCO Acc | VG-COCO mAP | RefCOCOg CIDEr |
|------|----------|----------|-------------|---------------|
| Single Crop | 23.0 | 77.0 | 40.0 | 107.3 |
| Increased Resolution 448 | 22.7 | 81.2 | 41.8 | 113.0 |
| Fixed 2 Views | 25.4 | 85.4 | 45.8 | 114.2 |
| Random 2 Views | 26.1 | 87.8 | 46.6 | 114.4 |
| Random 3 Views + Image Prior | **28.7** | **90.3** | **47.4** | **118.6** |

### Key Findings
- **Multi-view vastly outperforms high resolution**: Increasing resolution from 224 to 448 yields limited gains (OVAD 22.7 vs 23.0), whereas dual views boost performance to 25.4, indicating multi-view representations are more effective than single-view high resolution.
- **3 views are optimal**: Moving from 2 to 3 views brings significant improvement, but 4 views lead to a performance drop—the combination space of $C_{10}^3$ makes the representation manifold too complex to optimize.
- **The region reference view ($t_1=0$) is indispensable**: Removing it causes COCO Acc to plunge from 90.3 to 74.0, demonstrating that fine-grained details of the region itself are fundamental.
- **Excessive context is detrimental**: Performance on all tasks decreases when $t_2 > 0.5$. Attribute detection performs best at $t_2=0.1$ (almost exclusively referencing the target region), while captioning peaks at $t_2=0.4-0.5$.

## Highlights & Insights
- **Engineering realization of a biomimetic concept**: The foveation and saccade dynamic resolution mechanism of the human eye is elegantly mapped into a "nested view + random sampling + selective inference" workflow. Such biomimetic designs are highly relevant for vision backbone research.
- **4.2B model outperforms 7B+ models**: Proves that smart input processing strategies are more effective than brute-forcing model parameters: small model + robust strategy > large model + fixed strategy.
- **pHASH-based image prior selection**: Automatically selects the optimal views without requiring explicit task-type inputs, realizing true task-agnostic inference.

## Limitations & Future Work
- Each view requires a separate forward pass through the ViT encoder; 3 views imply 3$\times$ the encoding cost, which limits real-time applications.
- The nested view strategy assumes prior knowledge of target regions (requiring a bounding box input) and cannot handle bounding-box-free region understanding.
- The model uses only FlanT5-XL as its LLM backbone; scaling up to stronger LLMs (e.g., LLaMA-7B) could yield further improvements.
- The pHASH method for image priors is highly heuristic; learnable view-selection strategies warrant future exploration.

## Related Work & Insights
- **vs Alpha-CLIP**: Alpha-CLIP marks regions via an alpha channel but still relies on single-resolution input. DynRefer's multi-resolution strategy provides much richer visual information.
- **vs ControlCap**: ControlCap (also 4.2B) enhances output quality via controllable caption generation, while DynRefer intervenes at the input stage via dynamic resolution. Their methodologies are orthogonal and can be combined.
- **vs GLaMM**: GLaMM is a 7.4B grounding model that underperforms compared to the 4.2B DynRefer in region captioning, demonstrating that intelligent input strategies can compensate for parameter scale gaps.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The biomimetic concept of dynamic resolution is novel, and the implementation combining nested views with random sampling is clean and simple.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ High coverage with 5 tasks and thorough ablation studies (over 18 rows of comparison), with empirical support for every design choice.
- **Writing Quality**: ⭐⭐⭐⭐ The method explanation is clear, the biological motivation is highly engaging, and the ablation tables are well-organized.
- **Value**: ⭐⭐⭐⭐ Significant contribution to region-level multimodal understanding, with dynamic resolution paradigms generalizable to adjacent domains like video understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Mimic In-Context Learning for Multimodal Tasks](mimic_in-context_learning_for_multimodal_tasks.md)
- [\[CVPR 2025\] HiFICL: High-Fidelity In-Context Learning for Multimodal Tasks](hificl_high-fidelity_in-context_learning_for_multimodal_tasks.md)
- [\[CVPR 2025\] DocoPilot: Improving Multimodal Models for Document-Level Understanding](docopilot_improving_multimodal_models_for_document-level_understanding.md)
- [\[CVPR 2025\] VideoGLaMM: A Large Multimodal Model for Pixel-Level Visual Grounding in Videos](videoglamm_a_large_multimodal_model_for_pixel-level_visual_grounding_in_videos.md)
- [\[CVPR 2026\] DRS-GUI: Dynamic Region Search for Training-Free GUI Grounding](../../CVPR2026/multimodal_vlm/drs-gui_dynamic_region_search_for_training-free_gui_grounding.md)

</div>

<!-- RELATED:END -->
