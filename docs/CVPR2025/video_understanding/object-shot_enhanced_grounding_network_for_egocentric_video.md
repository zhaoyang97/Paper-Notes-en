---
title: >-
  [Paper Note] Object-Shot Enhanced Grounding Network for Egocentric Video
description: >-
  [CVPR 2025][Video Understanding][Ego4D NLQ] OSGNet targets the two major shortcomings of ego-centric Natural Language Queries (NLQ)—namely, visual features lacking fine-grained object information and neglecting attention switching implied by head-mounted camera motion. To address this, it proposes a two-branch architecture consisting of an "object branch (Co-DETR + CLIP text encoder) + shot branch (shot segmentation based on head turn + shot-level contrastive learning)"…
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Ego4D NLQ"
  - "Object-aware grounding"
  - "shot contrastive learning"
  - "Mamba"
date: 2026-05-08
content_hash: 993198b044c97637
---

# Object-Shot Enhanced Grounding Network for Egocentric Video

**Conference**: CVPR 2025  
**arXiv**: [2505.04270](https://arxiv.org/abs/2505.04270)  
**Code**: [https://github.com/Yisen-Feng/OSGNet](https://github.com/Yisen-Feng/OSGNet)  
**Area**: Video Understanding / Egocentric Video / Video Temporal Grounding  
**Keywords**: Ego4D NLQ, Object-aware grounding, shot contrastive learning, Mamba

## TL;DR
OSGNet targets the two major shortcomings of ego-centric Natural Language Queries (NLQ)—namely, visual features lacking fine-grained object information and neglecting attention switching implied by head-mounted camera motion. To address this, it proposes a two-branch architecture consisting of an "object branch (Co-DETR + CLIP text encoder) + shot branch (shot segmentation based on head turn + shot-level contrastive learning)", achieving state-of-the-art (SOTA) performance on Ego4D-NLQ, Goal-Step, and TACoS.

## Background & Motivation
1. **Background**: The Ego4D NLQ task requires localizing the temporal segments of answers in long egocentric videos based on query sentences (e.g., "How many drill bits did I remove from the drill before I moved the yellow carton?"), which is a core capability of embodied AI and intelligent assistants.
2. **Limitations of Prior Work**:
    - Pre-trained features of egocentric videos (using clip-narration contrastive learning) mostly learn "actions" and lose the fine-grained information of "background small objects" (e.g., measuring tape) that the query cares about;
    - Third-person video localization methods (e.g., Moment-DETR, SnAG) perform poorly when directly transferred to this task, as they ignore the "high-frequency camera shot transitions" of head-mounted cameras.
3. **Key Challenge**: The training objective of general video-text backbones (action alignment) is inconsistent with the goal of the NLQ task (fine-grained memory retrieval of background objects).
4. **Goal**:
    - Compensate for the "loss of fine-grained object information";
    - Leverage the hidden signal "wearer's head movement = attention switching".
5. **Key Insight**: Explicitly inject object category information into video tokens using off-the-shelf detectors combined with CLIP-based textification. Meanwhile, segment videos into shot segments based on "head-turn points" and apply shot-query contrastive learning.
6. **Core Idea**: A two-branch framework: the main branch uses parallel cross-attention to fuse video, query, and object features, while the shot branch aligns shot-query pairs using a contrastive loss.

## Method

### Overall Architecture
- **Feature Extraction**: (a) Object Extraction: Co-DETR (pretrained on LVIS) detects objects in each frame, filters them based on query nouns and a confidence threshold $\theta$, and encodes class names via CLIP ViT-B/32 to obtain $\mathbf{O}_{clip} \in \mathbb{R}^{T \times N_o \times D_o}$; (b) Video backbone (EgoVLP/InternVideo) yields $\mathbf{V}_{clip} \in \mathbb{R}^{T \times D}$; (c) Text encoder yields $\mathbf{Q}_F \in \mathbb{R}^{L \times D}$.
- **Object Encoder**: Employs objects as queries and text queries as keys/values for cross-attention, obtaining query-aligned object features $\mathbf{O}_F$.
- **Main Branch**: Stacks multiple layers of [BiMamba video $\rightarrow$ CA(video, query) $\parallel$ CA(video, object) $\rightarrow$ gating fusion] to get fused features. A multi-scale transformer is then utilized to generate a feature pyramid, followed by classification and regression heads to output temporal intervals.
- **Shot Branch**: Separately uses video features, conducts shot segmentation based on head movements, and performs shot-level contrastive learning.
- **Inference**: Selects the top-K confidence intervals from the main branch.

### Key Designs

1. **Object Injection and Parallel Cross-Attention**

    - **Function**: Introduces fine-grained objects of interest from the query as an additional modality to compensate for the missing information in the video backbone.
    - **Mechanism**: Conducts separate cross-attention operations between visual features and queries/objects: $\mathbf{V}_Q^{(i)} = \hat{\mathbf{V}}^{(i)} + CA(\hat{\mathbf{V}}^{(i)}, \mathbf{Q}_F, \mathbf{Q}_F)$ and $\mathbf{V}_O^{(i)} = \hat{\mathbf{V}}^{(i)} + CA(\hat{\mathbf{V}}^{(i)}, \mathbf{O}_F, \mathbf{O}_F)$, respectively. They are then fused using a gating mechanism $\mathbf{A} = \sigma(\text{MLP}(\hat{\mathbf{V}}_Q \| \hat{\mathbf{V}}_O))$: $\mathbf{V}^{(i+1)} = \mathbf{A}\cdot\hat{\mathbf{V}}_Q + (1-\mathbf{A})\cdot\hat{\mathbf{V}}_O$.
    - **Design Motivation**: Query keywords ("drill bit", "yellow carton") are not necessarily captured within the video features. Explicitly exposing objects via an object detector and employing parallel cross-attention prevents query information from overshadowing object details.

2. **BiMamba Long Video Modeling**

    - **Function**: Replaces traditional self-attention to process long sequences (egocentric videos can span up to several hours).
    - **Mechanism**: Employs bidirectional Mamba within the fusion block, achieving linear complexity and saving memory compared to transformers.
    - **Design Motivation**: The length of videos in NLQ tasks far exceeds that of typical moment retrieval, making self-attention prohibitively expensive in memory.

3. **Shot Branch: Head-Turn Segmentation + Contrastive Learning**

    - **Function**: Leverages the unique egocentric signal "head-mounted camera movement corresponds to wearer attention shift" to automatically segment the video into semantically independent shots.
    - **Mechanism**: Estimates the head rotation amplitude from camera trajectory/motion (optical flow or camera-pose variations) and segments shots based on peak values. A representation is extracted for each shot and aligned with the query text feature using a contrastive loss, mapping mutually relevant shots and queries closer in the embedding space.
    - **Design Motivation**: NLQ tasks often contain temporal structures like "do action A first, then B, and finally return to C". Head turns correspond to transitions in user attention or task boundaries, which serve as natural, weakly-supervised segmentation cues.

### Loss & Training
- Localization loss: $\mathcal{L}_{ML} = \mathcal{L}_{cls} + \mathcal{L}_{reg}$. Focal loss is used for classification, while Distance-IoU loss (only for positive samples) is used for regression.
- Contrastive loss $\mathcal{L}_{shot}$ (InfoNCE) for the shot branch.
- Total loss: $\mathcal{L} = \mathcal{L}_{ML} + \lambda \mathcal{L}_{shot}$.

## Key Experimental Results

### Main Results
**Ego4D-NLQ v1 Test (R@1, IoU=0.5)**:

| Method | Feature | R@1@0.5 |
|------|---------|---------|
| InternVideo | E+I | 10.06 |
| CONE | E | 7.84 |
| SnAG | E | 10.29 |
| RGNet | E | 10.61 |
| RGNet† (NaQ pretrain) | E | 11.69 |
| **OSGNet** | E | 10.71 |
| **OSGNet†** | E | **15.46** |

OSGNet† improves upon RGNet† by +3.77 points in R@1@0.5 and +6.74 points in R@5@0.5.

**Ego4D-Goal-Step**: R@1@0.3 is +3.65 higher than BayesianVSLNet.
**TACoS** (third-person comparison): R@1@0.5 is +3.32 higher than SnAG, proving effectiveness on generic videos as well.
**vs GroundVQA on NLQ**: R@1@0.5 +2.15.

### Ablation Study

| Configuration | R@1@0.5 (Ego4D-NLQ) |
|------|--------------------|
| Baseline (no object, no shot) | ~13 |
| + Object branch | ~14 |
| + Shot branch | ~14.3 |
| Full (object + shot) | 15.46 |
| Replacing BiMamba with self-attn | Out of memory (OOM) or performance degradation |

### Key Findings
- Object information brings the most significant improvement (~+5 R@1) for "background object" queries (e.g., "where is the screwdriver").
- The shot branch significantly benefits long videos (>5min) but shows almost no gain for short videos.
- The object filtering threshold $\theta$ should be chosen carefully: too high results in missing small objects, while too low introduces noise, indicating a clear sweet spot.

## Highlights & Insights
- **First explicit modeling of "head-mounted camera motion = attention signal"**: This is a unique, previously overlooked supervision signal in egocentric videos, which can potentially be extended to egocentric action recognition and action anticipation.
- **Lightweight injection of Object-as-text**: Utilizing the detector to generate labels instead of extraction-heavy region features, and then encoding class names through CLIP, avoids introducing a large number of extra parameters. This technique of "textifying detection results" can be readily adopted in any video QA task.
- **Parallel CA + gating** is more stable than sequential CA because it avoids modality order bias, making it a reusable multimodal fusion design.
- **Replacing transformers with BiMamba** validates the utility of state space models (SSMs) in long video modeling.

## Limitations & Future Work
- The object branch heavily relies on the detection capability of Co-DETR on the LVIS vocabulary, which may still miss long-tail objects.
- Shot segmentation depends on low-level optical flow/motion estimation, which is prone to false cuts under dramatic lighting changes or unstable camera shake.
- Textifying objects discards geometric information (position/shape/size), which might be insufficient for queries involving spatial relationships; future research could incorporate bounding box coordinates as auxiliary tokens.
- Multi-modality expansion (e.g., audio) has not been explored; egocentric videos often contain verbal instructions that can serve as additional cues.
- Future work: Feeding shot segments as "chapters" into an LLM for retrieval-based reading could further boost performance in long videos.

## Related Work & Insights
- **vs SnAG (CVPR 24)**: Single-branch without object information; OSGNet adds both the object and shot branches.
- **vs NaQ (CVPR 23)**: A pure data-augmentation approach; OSGNet is orthogonal to and can be combined with NaQ pretraining.
- **vs GroundVQA**: GroundVQA adopts LLMs for video grounding, whereas this work follows a specialized model approach, making it more computationally efficient.
- **Inspiration**: Tasks struggling with features missing critical fine-grained details can benefit from injecting text tokens generated by specialized expert models.

## Rating
- **Novelty**: ⭐⭐⭐ The combination of object injection and shot contrastive learning fits the requirements of the NLQ task beautifully, though neither technique is entirely new on its own.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on three major datasets with detailed ablation studies and benchmarking against NaQ pretraining.
- **Writing Quality**: ⭐⭐⭐⭐ The motivation behind the dual-branch architecture is clearly articulated.
- **Value**: ⭐⭐⭐⭐ Achieves noticeable gains on egocentric NLQ, presenting a solid SOTA baseline; the shot segmentation idea holds transfer value for general long video processing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] HierarQ: Task-Aware Hierarchical Q-Former for Enhanced Video Understanding](hierarq_task-aware_hierarchical_q-former_for_enhanced_video_understanding.md)
- [\[ICCV 2025\] Fine-grained Spatiotemporal Grounding on Egocentric Videos](../../ICCV2025/video_understanding/fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)
- [\[CVPR 2025\] DynFocus: Dynamic Cooperative Network Empowers LLMs with Video Understanding](dynfocus_dynamic_cooperative_network_empowers_llms_with_video_understanding.md)
- [\[ICCV 2025\] Frequency-Semantic Enhanced Variational Autoencoder for Zero-Shot Skeleton-based Action Recognition](../../ICCV2025/video_understanding/frequency-semantic_enhanced_variational_autoencoder_for_zero-shot_skeleton-based.md)
- [\[CVPR 2025\] EgoLife: Towards Egocentric Life Assistant](egolife_towards_egocentric_life_assistant.md)

</div>

<!-- RELATED:END -->
