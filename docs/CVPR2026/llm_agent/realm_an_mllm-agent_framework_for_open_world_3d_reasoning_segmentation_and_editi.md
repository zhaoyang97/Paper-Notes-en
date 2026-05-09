---
title: >-
  [Paper Note] REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting
description: >-
  [CVPR 2026][LLM Agent][3D reasoning segmentation] This paper proposes REALM, a framework that leverages an MLLM agent to perform reasoning segmentation on views rendered by 3D Gaussian Splatting (3DGS), and introduces a Global-Local Spatial Grounding strategy (GLSpaG) to aggregate multi-view MLLM reasoning results. REALM substantially outperforms existing methods on implicit-instruction 3D segmentation (mIoU 92.88% vs. baseline 44.82% on LERF) and supports downstream 3D editing.
tags:
  - CVPR 2026
  - LLM Agent
  - 3D reasoning segmentation
  - multimodal large language models
  - Gaussian splatting
  - global-local grounding
  - 3D editing
date: 2026-05-08
content_hash: b3352edd8b49382d
---

# REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting

**Conference**: CVPR 2026
**arXiv**: [2510.16410](https://arxiv.org/abs/2510.16410)
**Code**: [Project Page](https://ChangyueShi.github.io/REALM)
**Area**: LLM Agent
**Keywords**: 3D reasoning segmentation, multimodal large language models, Gaussian splatting, global-local grounding, 3D editing

## TL;DR
This paper proposes REALM, a framework that leverages an MLLM agent to perform reasoning segmentation on views rendered by 3D Gaussian Splatting (3DGS), and introduces a Global-Local Spatial Grounding strategy (GLSpaG) to aggregate multi-view MLLM reasoning results. REALM substantially outperforms existing methods on implicit-instruction 3D segmentation (mIoU 92.88% vs. baseline 44.82% on LERF) and supports downstream 3D editing.

## Background & Motivation

**Background**: 3D open-vocabulary segmentation methods (e.g., LERF, GS-Group) can handle explicit queries ("segment the cup") but fail to understand implicit instructions that require reasoning (e.g., "segment the object between the lamp and the book"). MLLMs excel at 2D reasoning but lack 3D spatial understanding.

**Limitations of Prior Work**: Feeding one or a few rendered views directly into an MLLM is highly sensitive to viewpoint selection—a suboptimal view may occlude the target object. Feeding a large number of views simultaneously overwhelms the MLLM and prevents consistent 3D understanding.

**Key Challenge**: MLLMs possess strong 2D reasoning capabilities but no 3D perception; 3D segmentation methods have spatial understanding but no reasoning ability. The core challenge is bridging these two.

**Goal**: To exploit the reasoning capabilities of off-the-shelf MLLMs for open-world 3D reasoning segmentation, without any 3D-specific post-training.

**Key Insight**: 3DGS is used as a high-fidelity proxy for the 3D world, rendering photorealistic novel views for MLLM understanding. A two-stage global-local strategy aggregates multi-view reasoning results.

**Core Idea**: A two-stage strategy of global multi-view coarse localization followed by local close-up fine segmentation, lifting the 2D reasoning capability of MLLMs into the 3D domain.

## Method

### Overall Architecture
**Input**: A 3DGS scene and a natural language instruction (which may be an implicit reasoning query). **Output**: A 3D segmentation mask and optional 3D editing results. The method consists of three components: (1) 3D feature field construction (SAM + cross-view propagation → per-Gaussian instance features); (2) LMSeg (an MLLM-based image-level reasoning segmentation agent); (3) GLSpaG (the global-local spatial grounding strategy).

### Key Designs

1. **3D Feature Field**:

    - **Function**: Assigns a consistent instance ID to each Gaussian primitive.
    - **Mechanism**: SAM extracts per-frame instance masks; a temporal propagation model associates instances across views to obtain consistent $id_i$. Each Gaussian is assigned a feature $f_i \in \mathbb{R}^D$, which is rendered into a 2D feature map $F$ via alpha blending. A classifier $\mathcal{CLS}$ maps features to instance IDs.
    - **Design Motivation**: MLLM reasoning results must be linked back to 3D space; the instance feature field serves as the bridge between 2D reasoning and 3D segmentation.

2. **MLLM-Based Visual Segmenter (LMSeg)**:

    - **Function**: Performs language-guided reasoning segmentation on a single view.
    - **Mechanism**: Given a rendered image $\mathcal{I}$ and query $q$, the MLLM returns a bounding box $\mathcal{B}$, category $\mathcal{C}$, and explanation $\mathcal{E}$. $\mathcal{B}$ is fed into SAM to obtain a 2D mask $M^{2D}$. The mask is cross-referenced with instance IDs $\hat{id}$ in the feature field to determine the target instance ID.
    - **Design Motivation**: Combines the reasoning capability of MLLMs with the precise segmentation capability of SAM—the MLLM handles "understanding" while SAM handles "mask drawing."

3. **Global Stage (GLSpaG - Global)**:

    - **Function**: Coarsely localizes the target object from multiple global viewpoints.
    - **Mechanism**: K-means clustering on training camera poses yields candidate viewpoints; the top-$N^{\text{global}}$ views containing the most instances are selected. LMSeg is run in parallel on each global view, and results are aggregated via voting to obtain the target instance ID: $ID^q = \arg\max_c |\{i: ID_i^q = c\}|$. The classifier groups matching 3D Gaussians into a coarse 3D mask.
    - **Design Motivation**: Parallel multi-view reasoning with voting is far more robust than relying on a single viewpoint. K-means view selection ensures coverage.

4. **Local Stage (GLSpaG - Local)**:

    - **Function**: Refines the 3D mask using close-up views of the target object.
    - **Mechanism**: Views from the clustered candidates that contain the target ID are selected as local cameras. LMSeg is run to obtain a fine 2D mask $M_i^{2D-Local}$. The 3D mask is projected to 2D via differentiable rendering and aligned with the LMSeg mask through optimization: $\mathcal{L}_{\text{local}} = \|\hat{M}_i - M_i^{2D-Local}\|_1$, iterating for 50 steps.
    - **Design Motivation**: The coarse mask from the global stage may contain noise; fine segmentation from close-up views combined with rendering alignment corrects boundary errors.

### Loss & Training
Feature field training: supervised with 2D instance labels. Local refinement: L1 rendering mask alignment loss, optimized for 50 steps. No MLLM fine-tuning is required (Qwen-2.5-VL is used off-the-shelf). The full pipeline runs on a single RTX 3090.

## Key Experimental Results

### Main Results

**3D Segmentation under Implicit Instructions (mIoU %)**

| Method | LERF | 3D-OVS | REALM3D |
|--------|------|--------|---------|
| Gaga | 44.82 | 42.53 | 58.56 |
| GAGS | 17.84 | 58.46 | 52.24 |
| GS-Group | 42.43 | 41.79 | 65.55 |
| **REALM** | **92.88** | **93.68** | **82.30** |

### Ablation Study

| Configuration | mIoU | Notes |
|---------------|------|-------|
| GS-Group baseline | ~43% | No reasoning capability |
| + LMSeg (single view) | Improved but unstable | View-sensitive |
| + Global stage | Significant improvement | Robust multi-view voting |
| + Local stage | **92.88%** | Refined boundary correction |

### Key Findings
- REALM achieves mIoU of 92.88% on LERF under implicit instructions, more than doubling the baseline, demonstrating the substantial value of MLLM reasoning for 3D segmentation.
- The global-local strategy substantially outperforms directly feeding multiple views to an MLLM, as the latter fails to resolve ambiguities when given many views simultaneously.
- The newly proposed REALM3D dataset, containing 100+ scenes and 1,444 implicit prompt-mask pairs, fills the evaluation gap in 3D reasoning segmentation.
- Support for downstream 3D editing (removal, replacement, style transfer) demonstrates practical utility.

## Highlights & Insights
- **The two-stage global-local design is highly intuitive**: first "looking from afar" to identify the target (robust via voting), then "looking up close" for fine segmentation—closely mirroring human visual search strategies.
- **No 3D-specific training required**: the method relies entirely on off-the-shelf MLLMs and SAM, solving the problem through system design rather than model training. This implies that REALM's performance will improve automatically as VLM capabilities advance.
- **The REALM3D dataset is a significant contribution**: 100+ scenes with 1,444 implicit prompt-mask pairs, providing the first quantitative evaluation benchmark for 3D reasoning segmentation.

## Limitations & Future Work
- The pipeline depends on the quality of pre-trained 3DGS reconstruction—poor reconstruction quality will degrade overall performance.
- MLLM inference requires multiple calls ($N^{\text{global}}$ global + $N^{\text{local}}$ local), resulting in relatively high latency.
- The voting mechanism assumes that the majority of viewpoints can correctly identify the target, which may fail for heavily occluded objects.
- Performance in large-scale outdoor scenes has not been evaluated.

## Related Work & Insights
- **vs. ReasonGrounder**: ReasonGrounder relies on bird's-eye views, limiting its applicability in complex 3D environments. REALM's multi-view global-local strategy is more general.
- **vs. ScanReason/VGMamba**: These methods only predict 3D bounding boxes, whereas REALM outputs fine-grained 3D masks.
- **vs. SceneAssistant (same batch)**: Both employ VLM agents for 3D scene understanding, but address different tasks—REALM targets segmentation and editing, while SceneAssistant targets generation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The global-local two-stage strategy is elegantly designed, though the overall framework is a composition of existing components.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three benchmarks (including the newly proposed REALM3D), complete ablations, and rich editing demonstrations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear and systematic, with well-formalized notation.
- **Value**: ⭐⭐⭐⭐⭐ Overwhelming performance gains (2×+ mIoU), a new dataset, and practical 3D editing capability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation](sceneassistant_a_visual_feedback_agent_for_openvoc.md)
- [\[CVPR 2026\] WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning](worldmm_dynamic_multimodal_memory_agent_for_long_video_reasoning.md)
- [\[CVPR 2026\] Nerfify: A Multi-Agent Framework for Turning NeRF Papers into Code](nerfify_a_multi-agent_framework_for_turning_nerf_papers_into_code.md)
- [\[CVPR 2026\] CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare](carepilot_a_multi-agent_framework_for_long-horizon_computer_task_automation_in_h.md)
- [\[CVPR 2026\] Think, Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding](think_then_verify_a_hypothesis-verification_multi-agent_framework_for_long_video.md)

</div>

<!-- RELATED:END -->
