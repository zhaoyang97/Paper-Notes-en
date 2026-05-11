---
title: >-
  [Paper Note] REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting
description: >-
  [CVPR 2026][LLM Agent][3D Reasoning Segmentation] REALM is proposed as a framework that leverages MLLM reasoning capabilities to perform open-world 3D reasoning segmentation on 3DGS via a global-to-local spatial groundin…
tags:
  - "CVPR 2026"
  - "LLM Agent"
  - "3D Reasoning Segmentation"
  - "MLLM-Agent"
  - "3D Gaussian Splatting"
  - "Global-to-Local Spatial Grounding"
  - "3D Scene Editing"
date: 2026-05-08
content_hash: ecd75a8d41f4d379
---

# REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting

**Conference**: CVPR 2026
**arXiv**: [2510.16410](https://arxiv.org/abs/2510.16410)
**Code**: [https://ChangyueShi.github.io/REALM](https://ChangyueShi.github.io/REALM)
**Area**: LLM Agent / 3D Vision
**Keywords**: 3D Reasoning Segmentation, MLLM-Agent, 3D Gaussian Splatting, Global-to-Local Spatial Grounding, 3D Scene Editing

## TL;DR
REALM is proposed as a framework that leverages MLLM reasoning capabilities to perform open-world 3D reasoning segmentation on 3DGS via a global-to-local spatial grounding strategy, handling implicit instructions without 3D post-training. It achieves 92.88% mIoU on LERF, surpassing baseline methods by 40+ percentage points, and supports editing tasks including object removal, replacement, and style transfer.

## Background & Motivation

**Background**: Enabling AI systems to understand complex human instructions and precisely localize target objects in 3D scenes is a foundational capability for robotics and human-computer collaboration. Existing 3D open-vocabulary segmentation methods (e.g., LERF, LangSplat, GS-Grouping) can handle explicit category queries (e.g., "segment the cup"), but perform poorly on implicit instructions requiring reasoning (e.g., "segment the object between the lamp and the book," "make the table tidier").

**Limitations of Prior Work**: (1) 3D segmentation methods lack reasoning capabilities—they can only perform explicit keyword matching and cannot understand spatial relationships, semantic attributes, or commonsense reasoning. (2) While 2D MLLMs excel at reasoning, they inherently lack 3D spatial understanding—directly feeding rendered views to an MLLM is highly sensitive to viewpoint selection, and different angles may produce contradictory results. (3) Existing attempts (e.g., ScanReason, ReasonGrounder) are limited by bounding box prediction or dependence on top-down views, resulting in insufficient precision and applicability.

**Key Challenge**: MLLMs possess strong 2D reasoning capabilities but lack 3D spatial awareness. The core challenge is how to reliably lift 2D reasoning results into 3D space and obtain precise segmentation masks without 3D-specific post-training of the MLLM.

**Goal**: To realize an open-world framework that understands implicit reasoning instructions, requires no 3D post-training, and generates precise 3D segmentation masks.

**Key Insight**: 3DGS serves as a high-fidelity proxy for the 3D world—it can render photorealistic novel views for MLLM comprehension. A two-stage multi-view reasoning strategy aggregates MLLM responses across viewpoints to eliminate single-view sensitivity.

**Core Idea**: Render multi-view images from 3DGS for MLLM reasoning, then obtain precise 3D masks through a two-stage strategy of global coarse localization followed by local fine segmentation.

## Method

### Overall Architecture
The REALM pipeline consists of three core modules: (1) **3D Feature Field Construction**—starting from 2D instance masks from SAM, consistent instance IDs are established through cross-view tracking, and instance features $f_i \in \mathbb{R}^D$ are learned for each 3D Gaussian, enabling mapping to instance IDs via a classifier $\mathcal{C}LS$; (2) **LMSeg (MLLM-Based Visual Segmenter)**—given an image and a language query, an MLLM is called to obtain reasoning results (bounding box + category + explanation), SAM then generates a 2D mask, and the target instance ID is retrieved via the feature field; (3) **GLSpaG (Global-to-Local Spatial Grounding)**—coarse localization is first performed from global viewpoints, followed by fine segmentation and 3D mask refinement from local viewpoints. The resulting segmentation can be directly applied to 3D editing tasks such as object removal, replacement, and style transfer.

### Key Designs

1. **3D Feature Field and Instance Identification**:

    - **Function**: Learn instance features for each Gaussian primitive, enabling stable mapping of 2D segmentation results into 3D space.
    - **Mechanism**: SAM is used to extract instance masks frame by frame; a temporal propagation model associates instances across views to obtain consistent IDs. Each Gaussian $G_i$ is assigned a feature $f_i$, and a 2D feature map is rendered via alpha blending: $F = \sum_i f_i \alpha_i \prod_{j<i}(1-\alpha_j)$. A classifier then predicts the instance ID: $\hat{id}(u,v) = \arg\max_k (CLS(F)_{u,v,k})$.
    - **Design Motivation**: Establish a stable bridge from 2D to 3D—LMSeg reasons in 2D to identify the target, and the feature field directly determines the corresponding set of 3D Gaussians without complex multi-view 3D fusion.

2. **Global Spatial Grounding (GLSpaG-Global)**:

    - **Function**: Perform parallel reasoning from multiple global viewpoints and determine coarse localization of the target instance through voting aggregation.
    - **Mechanism**: K-means clustering is applied to training camera poses to obtain $N^{cluster}$ representative viewpoints, from which $N^{global}=8$ views containing the most instances are selected as global cameras. LMSeg is called for each global view to obtain a target instance ID, and a majority vote determines the final target: $ID^q = \arg\max_{c} |\{i: ID_i^q = c\}|$. A coarse 3D segmentation mask $M^{3D}$ is generated from the 3D feature field accordingly.
    - **Design Motivation**: Single-view reasoning is highly sensitive to viewpoint selection (Fig. 2 demonstrates large variance across 10 random views); multi-view voting significantly reduces this randomness.

3. **Local Spatial Grounding and Refinement (GLSpaG-Local)**:

    - **Function**: Based on the global localization result, sample local cameras near the target object, obtain fine-grained 2D masks, and align the 3D mask with local masks through optimization.
    - **Mechanism**: Viewpoints containing the target ID are selected from clustered representative cameras as local cameras. LMSeg is called for each local view to obtain a 2D mask. The 3D mask $M^{3D}$ is rendered to each local view via differentiable rasterization, and boundary precision is optimized using an L1 loss $\mathcal{L}_{local} = \|\hat{M}_i - M_i^{2D\text{-Local}}\|_1$ over 50 iterations.
    - **Design Motivation**: The coarse mask from the global stage has imprecise boundaries; close-up local views provide finer segmentation information, and optimization-based alignment substantially improves mask quality.

### Loss & Training
During feature field training, a cross-entropy loss aligns rendered instance IDs with SAM ground-truth IDs. During inference, the local refinement stage uses an L1 loss to align the 3D rendered mask with LMSeg's 2D mask, requiring only 50 optimization steps (3.67s). The overall framework requires no 3D-specific post-training; both the MLLM (Qwen-2.5-VL) and SAM are off-the-shelf pretrained models.

## Key Experimental Results

### Main Results

| Dataset | Metric | REALM | Prev. SOTA | Gain |
|---------|--------|-------|------------|------|
| LERF | mIoU | 92.88% | 44.82% (Gaga) | +48.06% |
| LERF | mBIoU | 90.12% | 42.37% (Gaga) | +47.75% |
| 3D-OVS | mIoU | 93.68% | 58.46% (GAGS) | +35.22% |
| 3D-OVS | mBIoU | 86.02% | 50.34% (GAGS) | +35.68% |
| REALM3D | mIoU | 82.30% | 65.55% (GS-Group) | +16.75% |
| REALM3D | mBIoU | 70.37% | 55.99% (GS-Group) | +14.38% |

Note: All comparisons are conducted under **implicit query** conditions. Baseline methods primarily rely on CLIP keyword matching and cannot effectively handle implicit instructions that require reasoning.

### Ablation Study

| Configuration | mIoU | mBIoU | Notes |
|---------------|------|-------|-------|
| GS-Group (Baseline) | 0.32 | 0.30 | No reasoning capability |
| +Qwen2.5-VL | 0.78 | 0.77 | MLLM reasoning added but unstable |
| +Global Reasoning | 0.89 | 0.88 | Multi-view voting eliminates randomness |
| +Local Refinement | 0.95 | 0.94 | Boundary refinement |

Ablation of global camera sampling strategies (LERF Figurines):

| Strategy | mIoU |
|----------|------|
| No K-means | 0.38 |
| K-means + Random selection | 0.76 |
| Fully random | 0.59 |
| K-means + Top-K-ID (final) | 0.95 |

Inference efficiency: rendering speed 354.72 FPS; total inference time <10s (global MLLM 2.53s + local MLLM 2.48s + local refinement 3.67s).

### Key Findings
- REALM's advantage on implicit queries is substantial (mIoU exceeds baselines by 48% on LERF), as baseline methods fundamentally cannot handle reasoning-type instructions.
- The global camera sampling strategy is critical: K-means clustering ensures viewpoint diversity, and Top-K-ID selection ensures coverage of more objects—both steps are indispensable.
- 50 refinement steps is optimal—too few (10) yields insufficient precision, while too many (500/1000) causes overfitting and degradation.
- Cluster count $N^{cluster}=24$ performs best; too few (2) results in insufficient coverage while too many (128) introduces noise.

## Highlights & Insights
- The approach of using 3DGS as a "viewpoint factory" is particularly elegant—MLLMs excel at understanding photorealistic 2D images, and 3DGS can render exactly such images, making the two naturally complementary.
- The two-stage global-to-local strategy is essentially a coarse-to-fine spatial attention mechanism, progressively reducing uncertainty from coarse scene-level retrieval to fine local segmentation.
- The voting aggregation mechanism is simple yet effective, converting the high variance of single-view reasoning into stable low-variance multi-view output.
- The newly introduced REALM3D benchmark (100+ scenes, 1444 prompt-mask pairs, including implicit instructions) fills a gap in evaluation for 3D reasoning segmentation.
- A single agent framework simultaneously supports multiple 3D interaction tasks: segmentation, removal, replacement, and style transfer.

## Limitations & Future Work
- The framework depends on 3DGS reconstruction quality—poor reconstruction (missing textures, geometric errors) degrades rendered view quality and cascades into degraded MLLM comprehension.
- The reasoning ceiling of the MLLM determines the system's upper bound—highly complex reasoning chains or highly abstract instructions may cause failures.
- Multi-view reasoning introduces latency (8 global views + N local views), with total time approximately 8.68s, making real-time interaction difficult.
- Local refinement is an L1 optimization performed per view; when the number of viewpoints is limited, the 3D mask may be insufficiently precise in uncovered regions.
- Although the REALM3D dataset is relatively large (100+ scenes), the difficulty and diversity of implicit instructions still have room for improvement.

## Related Work & Insights
- **vs. LERF/LangSplat**: These methods embed CLIP language features in radiance fields/3DGS and can only handle explicit category queries (e.g., "cup"); they cannot understand reasoning instructions such as "find me something that emits light." REALM overcomes this limitation through MLLM reasoning.
- **vs. GS-Grouping/Gaga**: These methods group 3D instances via contrastive learning, with queries also limited to explicit vocabulary; REALM's mIoU exceeds Gaga by 48% on implicit queries.
- **vs. ReasonGrounder**: Conceptually closest to the proposed approach, but ReasonGrounder relies on top-down views and only predicts bounding boxes, whereas REALM provides fine-grained masks without being constrained to a specific viewpoint.
- **vs. 2D Reasoning Segmentation (LISA)**: Methods such as LISA perform reasoning segmentation in 2D but lack 3D consistency; REALM ensures 3D consistency through multi-view aggregation and the feature field.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of MLLM-Agent and 3DGS is novel, and the global-to-local spatial grounding strategy is cleverly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three benchmarks, detailed ablations, efficiency analysis, and multiple downstream tasks—extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated; the viewpoint sensitivity visualization in Fig. 2 is intuitive and compelling.
- **Value**: ⭐⭐⭐⭐ Pioneering the direction of 3D reasoning segmentation; the REALM3D benchmark offers long-term value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation](sceneassistant_a_visual_feedback_agent_for_openvoc.md)
- [\[CVPR 2026\] WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning](worldmm_dynamic_multimodal_memory_agent_for_long_video_reasoning.md)
- [\[CVPR 2026\] Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code](nerfify_multiagent_nerf_paper_to_code.md)
- [\[CVPR 2026\] CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare](carepilot_a_multi-agent_framework_for_long-horizon_computer_task_automation_in_h.md)
- [\[CVPR 2026\] Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding](think_then_verify_a_hypothesis-verification_multi-agent_framework_for_long_video.md)

</div>

<!-- RELATED:END -->
