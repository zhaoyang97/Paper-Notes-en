---
title: >-
  [Paper Note] Omni-RGPT: Unifying Image and Video Region-level Understanding via Token Marks
description: >-
  [CVPR 2025][Video Understanding][Region-level Understanding] Omni-RGPT proposes the Token Mark mechanism to directly mark target regions in the visual feature space, unifying regional-level understanding for both images and videos. Together with the RegVID-300k dataset containing 300,000 region-level video instructions, it achieves SOTA performance on tasks such as commonsense reasoning.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Region-level Understanding"
  - "Multimodal LLM"
  - "Token Mark"
  - "Region-level Instruction Dataset"
date: 2026-05-08
content_hash: 58667df0e2cbf974
---

# Omni-RGPT: Unifying Image and Video Region-level Understanding via Token Marks

**Conference**: CVPR 2025  
**arXiv**: [2501.08326](https://arxiv.org/abs/2501.08326)  
**Code**: [Project Page](https://miranheo.github.io/omni-rgpt)  
**Area**: Video Understanding / Multimodal LLM  
**Keywords**: Region-level Understanding, Multimodal LLM, Token Mark, Video Understanding, Region-level Instruction Dataset

## TL;DR

Omni-RGPT proposes the Token Mark mechanism to directly mark target regions in the visual feature space, unifying regional-level understanding for both images and videos. Together with the RegVID-300k dataset containing 300,000 region-level video instructions, it achieves SOTA performance on tasks such as commonsense reasoning.

## Background & Motivation

Multimodal Large Language Models (MLLMs) have made significant progress in global visual understanding, but region-level understanding still faces challenges:
- **Textual coordinate methods** (KOSMOS-2, Shikra) encode bounding box coordinates as text, but the number of tokens scales linearly with the frame count in video, limiting scalability.
- **RoI feature methods** (GPT4RoI, RegionGPT) extract regional visual features from each frame, but suffer from temporal drift—where RoI features for the same object are inconsistent across different frames.
- **Visual marking methods** (SoM, ViP-LLaVA) overlay markers on images, which can alter the original visual appearance of the image.
- Methods relying solely on initial frames (Elysium, Merlin) lack robust references to target objects in subsequent frames.
- Lack of large-scale region-level video instruction datasets—existing datasets either have short descriptions (Elysium only contains nouns/phrases) or limited sources.
- Unifying multi-frame target representations into a single consistent vector remains an open challenge.

## Method

### Overall Architecture

Omni-RGPT is based on the LLaVA architecture and introduces Token Marks as region identifiers. Input images/videos are encoded by a visual encoder to obtain visual tokens $V \in \mathbb{R}^{T \times D \times H \times W}$. Token Marks are embedded into corresponding spatial locations within the regions while being injected into the text prompt. An auxiliary Temporal Region Guide Head guides region consistency in videos during training, introducing no extra overhead during inference.

### Key Design 1: Token Mark Region Representation

**Function**: Assigns a unique learnable token identifier to each target region, establishing a direct connection between visual regions and textual references.

**Mechanism**: Defines a set of learnable tokens $F \in \mathbb{R}^{N_F \times C}$ (analogous to different colors on a palette). For $N$ input regions $\{m_i\}_{i=1}^N$, $N$ tokens $R = \{r_i\}_{i=1}^N$ are uniformly sampled from $F$. A Spatial Token Mark is constructed as $S_{:,h,w} = \frac{\sum_{i=1}^N m_{i,h,w} \cdot r_i}{\epsilon + \sum_{i=1}^N m_{i,h,w}}$. After downsampling to the visual token resolution, it is mapped to the LLM embedding space via a shared projection layer and added to the visual tokens as a residual: $\hat{V} = V + \hat{S}$. Meanwhile, the sampled tokens also replace the `<region>` placeholders in the text.

**Design Motivation**: (1) Scalability: Target tokens are shared across frames, meaning the number of text tokens is independent of the frame count; (2) Temporal Consistency: Fixed tokens ensure consistent referencing across frames; (3) Retaining Global Alignment: Adding them as residuals avoids destroying the base model's vision-language alignment.

### Key Design 2: Temporal Region Guide Head

**Function**: Guides the model to learn cross-frame consistency of target regions in videos during the training phase, without relying on explicit trajectory annotations.

**Mechanism**: An auxiliary classification head $\mathcal{F}_{\text{aux}}$ acts on the visual tokens output by the LLM, classifying each visual token into $N_F + 1$ classes ($N_F$ Token Mark categories + background), using soft labels to handle multi-region overlap. Region prompts $\hat{V}_1$ are only provided in the first frame, and the remaining frames $V_2, ..., V_T$ have no regional annotations. The auxiliary head implicitly guides the model to track target objects via the training-time consistency of Token Marks. The loss function is $\mathcal{L} = \mathcal{L}_{\text{LLM}} + \alpha\mathcal{L}_{\text{aux}}$.

**Design Motivation**: In practical applications, users usually annotate regions in only one frame. The auxiliary head enables the model to learn to automatically extend the first-frame annotation to the entire video, introducing no additional overhead during inference.

### Key Design 3: RegVID-300k Region-level Video Instruction Dataset

**Function**: Provides the first large-scale, diverse region-level video instruction data, containing 98k videos, 214k regions, and 294k instruction samples.

**Mechanism**: A three-stage pipeline construction: (1) GPT-4o assisted detailed region-level description—overlaying mask indices on video frames using the SoM technique, feeding them into GPT-4o to generate fine-grained descriptions of about 60 words; (2) Visual hallucination mitigation—filtering out hallucinated content from the descriptions generated by GPT-4o; (3) Description-guided instruction sample generation—generating multiple dialogue-format QA pairs from detailed descriptions. Data is sourced from 10 public video datasets.

**Design Motivation**: Existing video instruction data lacks region-level annotations, which restricts the development of region-level video understanding capabilities.

### Loss & Training

$\mathcal{L} = \mathcal{L}_{\text{LLM}} + \alpha\mathcal{L}_{\text{aux}}$, where $\mathcal{L}_{\text{LLM}}$ is the standard cross-entropy language modeling loss, and $\mathcal{L}_{\text{aux}}$ is the soft-label cross-entropy loss of the auxiliary classification head.

## Key Experimental Results

### Main Results: Causal-VidQA Video Commonsense Reasoning

| Method | LLM | Acc@D | Acc@E | Acc@P | Acc@C | Acc@All |
|------|-----|-------|-------|-------|-------|---------|
| **Omni-RGPT** | 7B | **84.0** | **84.6** | **84.2** | **85.4** | - |
| MotionEpic | 7B | 81.2 | 83.0 | 74.3 | 73.7 | 69.4 |
| Video-LLaVA | 7B | 73.7 | 74.4 | 67.6 | 65.4 | 61.8 |
| Video-ChatGPT | 7B | 73.1 | 75.1 | 66.0 | 63.9 | 61.1 |

### Ablation Study: Token Mark Key Components

| Setting | VCR Q→A | VCR QA→R | Description |
|------|---------|----------|------|
| Full Model | **Best** | **Best** | Token Mark + Guide Head |
| Textual Coordinates Only | Worse | Worse | Traditional Coordinate Encoding |
| RoI Features Only | Worse | Worse | Suffer from Temporal Drift |
| w/o Guide Head | Slightly Worse | Slightly Worse | Degraded Video Consistency |

### Key Findings

- Token Mark achieves SOTA performance on both image (VCR) and video (Causal-VidQA) commonsense reasoning benchmarks.
- The auxiliary guide head enables the model to stably track target objects in videos even when region annotations are only provided in the first frame.
- The RegVID-300k dataset leads to a significant improvement in video description tasks.
- The number of text tokens in the Token Mark method is independent of the number of video frames, showing excellent scalability.

## Highlights & Insights

- **Reverse Thinking**: Unlike traditional methods that generate region embeddings from visual features, Token Mark reverses the process by marking regions with predefined tokens, offering a novel perspective.
- **Minimalist Unification**: It handles both image and video region-level understanding concurrently using the exact same Token Mark mechanism.
- **Data Contribution**: RegVID-300k fills the gap in region-level video instruction data.

## Limitations & Future Work

- It relies on first-frame region annotations, making it less robust to scenarios where the target object is occluded or invisible in the first frame.
- The number of Token Marks $N_F$ is fixed, which may limit the capability to handle dense-region scenes.
- The tracking capability of the auxiliary guide head depends on the visual attention of the LLM, which may degenerate in fast-motion scenes.
- Future work can explore the integration with explicit tracking models.

## Related Work & Insights

- Token Mark is similar to the concept of learnable queries in DETR but is applied to mark specific spatial regions.
- Compared to the visual marking approach of SoM (Set-of-Mark), Token Mark operates in the latent space rather than the pixel space, thereby preserving the original image.
- The GPT-4o-assisted annotation and hallucination mitigation pipeline of RegVID-300k can be adapted to construct other video understanding datasets.

## Rating

⭐⭐⭐⭐ — The design of Token Mark is simple yet effective, unifying the methodological framework of image and video region understanding. The contribution of the RegVID-300k dataset is also highly valuable. The SOTA performance across multiple benchmarks validates the effectiveness of the proposed approach.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MMVU: Measuring Expert-Level Multi-Discipline Video Understanding](mmvu_measuring_expert-level_multi-discipline_video_understanding.md)
- [\[ICML 2025\] Unifying Specialized Visual Encoders for Video Language Models](../../ICML2025/video_understanding/unifying_specialized_visual_encoders_for_video_language_models.md)
- [\[NeurIPS 2025\] AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding](../../NeurIPS2025/video_understanding/adavideorag_omnicontextual_adaptive_retrievalaugmented_effic.md)
- [\[ICML 2026\] OmniSIFT: Modality-Asymmetric Token Compression for Efficient Omni-modal Large Language Models](../../ICML2026/video_understanding/omnisift_modality-asymmetric_token_compression_for_efficient_omni-modal_large_la.md)
- [\[CVPR 2025\] DivPrune: Diversity-Based Visual Token Pruning for Large Multimodal Models](divprune_diversity-based_visual_token_pruning_for_large_multimodal_models.md)

</div>

<!-- RELATED:END -->
