---
title: >-
  [Paper Note] Learning Precise Affordances from Egocentric Videos for Robotic Manipulation
description: >-
  [ICCV 2025][Segmentation][Affordance Learning] This paper presents a complete affordance learning system comprising: (1) an automatic pipeline for extracting precise graspable and functional affordance segmentation annotations from egocentric videos; (2) a Geometry-guided Affordance Transformer (GAT) based on DINOv2 with depth-geometric guidance for cross-domain affordance segmentation (mIoU improved by 13.8%); and (3) the Aff-Grasp framework, which achieves a 77.1% grasping success rate across 179 real robot trials.
tags:
  - ICCV 2025
  - Segmentation
  - Affordance Learning
  - Egocentric Video
  - Robotic Manipulation
  - Affordance Segmentation
  - Tool Grasping
date: 2026-05-08
content_hash: 63353b625b08a512
---

# Learning Precise Affordances from Egocentric Videos for Robotic Manipulation

**Conference**: ICCV 2025
**arXiv**: [2408.10123](https://arxiv.org/abs/2408.10123)
**Code**: [https://reagan1311.github.io/affgrasp](https://reagan1311.github.io/affgrasp)
**Area**: Image Segmentation
**Keywords**: Affordance Learning, Egocentric Video, Robotic Manipulation, Affordance Segmentation, Tool Grasping

## TL;DR

This paper presents a complete affordance learning system comprising: (1) an automatic pipeline for extracting precise graspable and functional affordance segmentation annotations from egocentric videos; (2) a Geometry-guided Affordance Transformer (GAT) based on DINOv2 with depth-geometric guidance for cross-domain affordance segmentation (mIoU improved by 13.8%); and (3) the Aff-Grasp framework, which achieves a 77.1% grasping success rate across 179 real robot trials.

## Background & Motivation

Affordance—the set of potential actions an object offers—is a core concept in embodied intelligence. For example, when cutting, one grasps the handle of a knife; when handing it to another person, one grasps the blade. Current research faces three major challenges:

**Data Scarcity**: Large-scale, precisely annotated affordance datasets are lacking, and labeling fine-grained object parts (e.g., spoon handles) is extremely difficult.

**Poor Generalization**: Existing models struggle to generalize across domains or to unseen objects and affordance categories.

**Limited Real-World Deployment**: Few works validate their approaches on physical robots.

These three issues are interrelated: insufficient large-scale diverse data → poor generalization → unreliable deployment.

Existing methods for learning affordances from video (e.g., VRB, Robo-ABC) have two key limitations:
- They focus solely on "how humans grasp objects" (graspable affordance), neglecting "which part of a tool is in use" (functional affordance).
- Affordances are represented as coarse Gaussian heatmaps rather than precise segmentation masks.

## Method

### Overall Architecture

The system consists of three components:
1. **Automatic Data Collection Pipeline**: Egocentric video → precise affordance segmentation annotations.
2. **GAT Model**: Geometry-guided affordance segmentation.
3. **Aff-Grasp Framework**: Affordance-driven robotic manipulation.

### Key Designs

1. **Automatic Affordance Data Collection Pipeline**:

    - **Graspable Point Localization**: Contact frames are extracted from hand-object interaction videos. A hand-object detector localizes the interaction region, and contact points are sampled at the intersection of the hand mask and the object bounding box. A pre-contact frame (where the object is fully visible) is identified, and contact points are projected onto it via homography transformation.
    - **Functional Point Localization**: Functional regions are identified from tool-object interaction videos. A pre-contact frame is selected where the IoU between the tool and target object is minimized; the point within the tool mask that is closest to the target object mask is computed. When no action video is available, the point farthest from the grasp point is used as a substitute.
    - **Data Generation**: Functional points are mapped to pre-contact frames via point correspondence, and SAM is then used with these points as prompts to generate precise segmentation masks.

   *Design Motivation*: To simultaneously obtain precise segmentation annotations for both graspable and functional affordances without manual labeling.

2. **Geometry-guided Affordance Transformer (GAT)**:

    - **DINOv2 Encoder + LoRA**: The self-supervised vision foundation model DINOv2 is used as the feature extractor, with LoRA fine-tuning to prevent overfitting and to adapt to multi-domain data. LoRA decomposes the trainable matrix as $W_0 + \Delta W = W_0 + BA$, where $r \ll \min(d,k)$.
    - **Depth Feature Injector (DFI)**: Pseudo depth maps are generated using Depth-Anything, and geometric features are injected into image features via cross-attention: $\hat{F}_i = \beta \cdot \text{softmax}(QK^T/\sqrt{d_k}) \cdot V + F_i$, where $\beta$ is initialized to 0 to prevent depth features from dominating at the start of training. Even when DFI is used only during training and discarded at inference, it still yields improvements—functioning as a regularizer.
    - **Cosine Similarity Classifier**: Segmentation is performed using the cosine similarity between learnable embeddings $M \in \mathbb{R}^{L \times C}$ and upsampled features, with no explicit background classifier (pixels below threshold $\tau$ are classified as background). This is more robust than a linear layer.

   *Design Motivation*: Given low-resolution training data and large domain gaps, DINOv2 enhances cross-domain capability, while DFI leverages shape information (e.g., cylindrical shape → graspable; sharp edge → cutting) to compensate for insufficient texture cues.

3. **Aff-Grasp Robotic Manipulation Framework**:

    - An open-vocabulary detector localizes objects in the scene → GAT predicts affordances for each object → the object with the desired affordance is selected.
    - Contact-GraspNet generates 6-DoF grasp poses within the graspable region → the highest-scoring pose is selected.
    - Affordance-specific sequential motion primitives are executed (tool use / handover).
    - CLIP text embeddings can replace learnable embeddings for open-vocabulary support.

### Loss & Training

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{focal} + \mathcal{L}_{dice}$$

A combination of focal loss and dice loss is used to handle severe class imbalance. Four DFI modules are placed at the beginning of four blocks throughout the model. DFI can optionally be removed at inference to accelerate computation.

## Key Experimental Results

### Main Results

**Visual Evaluation — Affordance Evaluation Dataset (AED):**

| Pretrain | Method | mIoU | F1 | Acc |
|----------|--------|------|-----|-----|
| ImageNet | DeepLabV3+ | 13.46 | 22.27 | 23.05 |
| ImageNet | PSPNet | 16.90 | 27.32 | 26.46 |
| ImageNet | SegFormer | 23.72 | 36.86 | 37.19 |
| Foundation | ZegCLIP | 18.33 | 26.41 | 25.55 |
| Foundation | DINOv2 | 46.16 | 62.49 | 63.61 |
| Foundation | ViT-Adapter | 50.86 | 66.88 | 65.21 |
| Foundation | OOAL | 54.82 | 70.58 | 68.00 |
| Foundation | **GAT (Ours)** | **68.62** | **81.09** | **83.51** |

**Robot Accuracy Evaluation:**

| Method | Correct Affordance | Successful Grasp | Successful Interaction |
|--------|--------------------|------------------|------------------------|
| LOCATE | 42/72 (58.3%) | 33/72 (45.8%) | n/a |
| Robo-ABC | 62/72 (86.1%) | 44/72 (61.1%) | n/a |
| **Aff-Grasp** | **70/72 (97.2%)** | **57/72 (80.6%)** | **47/72 (65.3%)** |

### Ablation Study

**Component Ablation of GAT (AED):**

| Configuration | mIoU | F1 | Acc |
|---------------|------|-----|-----|
| Baseline (DeiT III + linear + BCE) | 31.02 | 44.55 | 35.85 |
| + DINOv2 | 45.45 | 61.78 | 70.86 |
| + embedder | 48.83 | 65.10 | 71.07 |
| + embedder & 4× upsample | 51.41 | 64.26 | 67.27 |
| + focal loss | 50.70 | 66.97 | 70.12 |
| + focal & dice loss | 53.12 | 69.13 | 74.55 |
| cosine sim w/o bg | 56.70 | 72.00 | 71.22 |
| + DFI (training only) | 60.15 | 74.92 | 79.87 |
| + DFI (full) | 64.66 | 78.35 | 79.74 |
| + LoRA (full GAT) | **68.62** | **81.09** | **83.51** |

**DFI Computational Overhead:**

| Inference Setting | #Params (M) | GFLOPs | Inference Time (ms) |
|-------------------|-------------|--------|---------------------|
| w/ DFI | 96.9 (↓5.4%) | 204.9 (↓9.5%) | 10.1 (↓37.6%) |
| w/o DFI | — | — | — |

**Generalization Evaluation (Unseen Objects):**

| Method | Correct Affordance | Successful Grasp | Inference Time (s) |
|--------|--------------------|------------------|--------------------|
| LOCATE | 20/35 (57.1%) | 15/35 (42.9%) | 0.0047 |
| Robo-ABC | 24/35 (68.6%) | 21/35 (60.0%) | 12.92 |
| **Aff-Grasp** | **32/35 (91.4%)** | **28/35 (80.0%)** | **0.0063** |

### Key Findings

1. **Foundation models substantially outperform ImageNet-pretrained counterparts**: DINOv2 alone surpasses SegFormer (ImageNet) by 22.44 mIoU, confirming the importance of cross-domain representation capacity.
2. **DFI contributes substantially**: Using DFI only during training improves mIoU by 3.58; full DFI improves by 7.96, demonstrating that depth-geometric information serves as effective regularization.
3. **LoRA improves mIoU by 3.96**: LoRA fine-tuning enables efficient adaptation without modifying the original DINOv2 parameters.
4. **Cosine similarity is more robust than linear classification**: The combination of cosine similarity with no background classifier outperforms the linear w/o bg variant by 1.74 mIoU.
5. **Aff-Grasp comprehensively outperforms baselines in robot experiments**: Affordance prediction accuracy reaches 97.2% (11.1% above Robo-ABC), grasping success rate reaches 80.6% (19.5% above), and the framework is the first to support tool-object interaction.
6. **Strong generalization to unseen objects**: Affordance prediction accuracy of 91.4%, with inference taking only 6.3 ms.

## Highlights & Insights

- **First complete pipeline for jointly annotating precise graspable and functional affordance segmentations**: A qualitative leap from coarse heatmaps to precise masks.
- **Discovery of DFI as a training-time regularizer**: Discarding DFI at inference still yields improvements while accelerating inference by 37.6%, making the design highly practical.
- **Full closed-loop validation from perception to manipulation**: 179 real robot trials covering 7 tasks and 34 objects, including complex scenarios such as handover.
- **Open-vocabulary capability**: CLIP text embeddings can replace learnable embeddings to support unseen affordance categories.

## Limitations & Future Work

1. Data collection relies on the temporal ordering of hand-object and tool-object interactions in video; heuristic substitutes are required when certain interaction types are absent.
2. Segmentation quality depends on SAM prompt accuracy, which may be unreliable under heavy occlusion.
3. The "farthest point" substitution strategy for functional affordance assumes that the functional part is opposite the grasp point, which does not hold for all tools.
4. Motion primitives in real robot experiments still require pre-recording, limiting task complexity.
5. Low training data resolution (cropped regions typically <100 px) restricts recognition of fine-grained parts.

## Related Work & Insights

- **Learning affordances from human video**: VRB and Robo-ABC are predecessors; this work improves upon them in data quality (precise masks vs. heatmaps) and affordance type coverage (adding functional affordance).
- **Applying visual foundation models**: The cross-domain feature representation of DINOv2 is a key enabler for affordance generalization.
- **SAM for prompt-based segmentation**: Using point prompts to generate precise masks bridges the gap between sparse localization and dense segmentation.
- Provides an efficient alternative to task-oriented grasping approaches based on VLMs/LLMs, without requiring per-inference language model queries.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First complete system jointly learning precise graspable and functional affordance segmentation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Dual visual and robot evaluation, 179 real-world trials, comprehensive coverage of cross-domain, zero-shot, and cluttered scenarios.
- **Writing Quality**: ⭐⭐⭐⭐ Systematic and complete, with clear illustrations and a coherent narrative from data collection to modeling to deployment.
- **Value**: ⭐⭐⭐⭐⭐ Addresses all three major pain points in affordance research (data, generalization, deployment) with strong practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] 2HandedAfforder: Learning Precise Actionable Bimanual Affordances from Human Videos](2handedafforder_learning_precise_actionable_bimanual_affordances_from_human_vide.md)
- [\[ICCV 2025\] O-MaMa: Learning Object Mask Matching between Egocentric and Exocentric Views](o-mama_learning_object_mask_matching_between_egocentric_and_exocentric_views.md)
- [\[ICCV 2025\] ReferEverything: Towards Segmenting Everything We Can Speak of in Videos](refereverything_towards_segmenting_everything_we_can_speak_of_in_videos.md)
- [\[ICCV 2025\] Open-World Skill Discovery from Unsegmented Demonstration Videos](open-world_skill_discovery_from_unsegmented_demonstration_videos.md)
- [\[ICCV 2025\] UniGlyph: Unified Segmentation-Conditioned Diffusion for Precise Visual Text Synthesis](uniglyph_unified_segmentation-conditioned_diffusion_for_precise_visual_text_synt.md)

</div>

<!-- RELATED:END -->
