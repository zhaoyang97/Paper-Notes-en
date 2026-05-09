---
title: >-
  [Paper Note] LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D Capabilities
description: >-
  [ICCV 2025][3D Vision][3D Scene Understanding] This paper proposes LLaVA-3D, which constructs "3D Patches" by injecting 3D positional embeddings into 2D CLIP patch features, extending a 2D LMM (LLaVA-Video) into a unified 2D/3D understanding model with minimal architectural modifications. The approach achieves 3.5× faster training convergence than existing 3D LMMs, reaches state-of-the-art performance on multiple 3D benchmarks, and preserves 2D capabilities without degradation.
tags:
  - ICCV 2025
  - 3D Vision
  - 3D Scene Understanding
  - Large Multimodal Models
  - 3D Patch
  - Positional Encoding
  - Visual Grounding
date: 2026-05-08
content_hash: 82888ef685edc6ea
---

# LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D Capabilities

**Conference**: ICCV 2025
**arXiv**: [2409.18125](https://arxiv.org/abs/2409.18125)
**Code**: [https://zcmax.github.io/projects/LLaVA-3D](https://zcmax.github.io/projects/LLaVA-3D)
**Area**: 3D Vision / Multimodal VLM
**Keywords**: 3D Scene Understanding, Large Multimodal Models, 3D Patch, Positional Encoding, Visual Grounding

## TL;DR

This paper proposes LLaVA-3D, which constructs "3D Patches" by injecting 3D positional embeddings into 2D CLIP patch features, extending a 2D LMM (LLaVA-Video) into a unified 2D/3D understanding model with minimal architectural modifications. The approach achieves 3.5× faster training convergence than existing 3D LMMs, reaches state-of-the-art performance on multiple 3D benchmarks, and preserves 2D capabilities without degradation.

## Background & Motivation

**State of the Field**: 2D large multimodal models (LMMs) such as the LLaVA series have achieved remarkable progress in image and video understanding, yet they remain confined to virtual-level visual interaction and lack the ability to engage with the 3D physical world. Dominant approaches in existing 3D LMMs include: (a) using 3D point cloud encoders (e.g., LL3DA, LEO), (b) leveraging offline 3D instance segmentation to extract object features (e.g., Chat-Scene), and (c) aggregating CLIP features from multi-view images via manually designed 2D segmentation to construct 3D representations (e.g., 3D-LLM).

**Limitations of Prior Work**: (a) Large-scale 3D vision-language datasets are scarce, far smaller in scale than their 2D counterparts; (b) no pretrained 3D encoder comparable to CLIP exists; (c) existing 3D LMM pipelines are complex and computationally intensive—e.g., requiring 3D instance segmentation followed by feature aggregation, with scene feature construction taking up to 900 seconds per scene; (d) 3D visual grounding relies on offline 3D segmentation models, limiting practical applicability.

**Root Cause**: The central challenge is how to efficiently transfer the powerful 2D understanding priors of a 2D LMM to 3D scene understanding without sacrificing them. Directly training 3D LMMs faces the dual bottleneck of insufficient data and weak encoders.

**Paper Goals**: (a) How to efficiently transfer 2D visual understanding priors from a 2D LMM to 3D scene understanding? (b) How to construct a unified 2D+3D architecture that supports both modalities simultaneously? (c) How to directly predict accurate 3D spatial outputs (e.g., 3D bounding boxes) without relying on time-consuming offline 3D segmentation?

**Starting Point**: Inspired by ODIN—which achieves unified segmentation by applying different positional encodings for 2D and 3D—LLaVA-3D similarly bridges the 2D and 3D worlds by appending 3D positional embeddings to 2D CLIP patches, requiring minimal modifications to the original architecture.

**Core Idea**: Learnable 3D positional embeddings are directly added onto LLaVA's 2D CLIP patch features to construct 3D Patches that jointly encode semantic and spatial information, upgrading a 2D LMM to a 3D LMM at minimal cost.

## Method

### Overall Architecture

The model is built upon LLaVA-Video. The input consists of multi-view RGB-D images. A CLIP encoder extracts 2D patch features, which are projected into the LLM's embedding space and augmented with 3D positional embeddings encoded from 3D coordinates to form "3D Patches." These patches undergo optional pooling-based compression before being fed into the LLM for multimodal reasoning. For tasks requiring 3D coordinate inputs or outputs, 3D Coordinate Tokens and a dedicated Grounding Decoder are employed, respectively.

### Key Designs

1. **3D Patch Construction**:

    - Function: Augments 2D visual patch features with 3D spatial information to form 3D patch representations.
    - Mechanism: Given multi-view 2D patch features $X'_p \in \mathbb{R}^{V \times d \times w \times h}$ (where $V$ denotes the number of views), the 3D position $P \in \mathbb{R}^{V \times 3 \times w \times h}$ of each patch is obtained using known camera intrinsics, extrinsics, and depth maps. A two-layer MLP encodes these positions into 3D positional embeddings $P'$, which are then added directly: $X'_{3D} = X'_p + P'$.
    - Design Motivation: Additive fusion, rather than complex feature aggregation, maximally preserves the rich semantic information in 2D CLIP patches while injecting 3D spatial position information. This design also allows the same model to switch between 2D and 3D modes by toggling the 3D positional embedding.

2. **3D Patch Pooling**:

    - Function: Reduces the number of 3D Patches to accommodate the context length constraints of the LLM.
    - Mechanism: Two parameter-free pooling strategies are proposed: (a) Voxelization Pooling, which discretizes 3D space into a voxel grid and applies average pooling over all 3D patches within each occupied voxel, with the token count determined by voxel size rather than the number of input images; (b) FPS Pooling, which selects a fixed number of representative 3D patches via farthest point sampling.
    - Design Motivation: Unlike spatial or temporal pooling commonly used in 2D settings, 3D-position-based pooling ensures that the compressed features cover the entire scene structure as completely as possible.

3. **3D Coordinate Token Encoding & Grounding Decoder**:

    - Function: Handles 3D coordinate inputs (e.g., "describe the object at position [x, y, z]") and outputs (3D bounding box prediction).
    - Mechanism: On the input side, a 3D Position Encoding Layer encodes 3D coordinates into 3D Coordinate Tokens, which are concatenated with 3D patch tokens and text tokens before being fed into the LLM. On the output side, a Grounding Decoder is designed: instance queries are sampled from 3D patches via FPS, and cross-attention aggregates geometric information from 3D patches and semantic information from the LLM. Multi-scale 3D k-NN attention models local geometric structures, ultimately predicting 3D bounding boxes.
    - Design Motivation: Directly prompting the LLM to output 3D coordinates yields poor results (only 7.8 Acc@0.25 on ScanRefer), necessitating a dedicated decoder to extract geometric information from 3D patches. This design also eliminates the dependency on offline 3D segmentation models required by two-stage approaches.

### Loss & Training

Two-stage training:
- **Stage 1 — Multi-task Instruction Fine-tuning**: Joint fine-tuning on 2D data (LLaVA-Video data) and 3D data (LLaVA-3D-Instruct-86K), training the 3D positional encoding layers and other modules. The 3D data covers tasks including 3D QA, 3D dense captioning, and 3D visual grounding.
- **Stage 2 — Decoder-only Fine-tuning**: As the Grounding Decoder does not fully converge within the single epoch of Stage 1, all other components are frozen and only the decoder and location tokens are further trained using 3D visual grounding data.

## Key Experimental Results

### Main Results

| Benchmark | Metric | Ours | Prev. SOTA | Gain |
|-----------|--------|------|------------|------|
| ScanQA (val) | CIDEr | **103.1** | 101.4 (LEO) | +1.7 |
| ScanQA (val) | EM@1 | **30.6** | 27.2 (Scene-LLM) | +3.4 |
| SQA3D (test) | EM@1 | **60.1** | 54.7 (Chat-3D v2) | +5.4 |
| MMScan QA | EM@1 | **50.1 (54.9)** | 36.6 (44.5) (LEO) | +13.5 |
| OpenEQA | Accuracy | 53.2 | 55.3 (GPT-4V, 50 frames) | Competitive |
| Scan2Cap | C@0.5 | **84.1** | 77.2 (ChatScene) | +6.9 |
| Multi3DRefer | Acc@0.25 | **49.8** (single-stage) | 57.1 (ChatScene, two-stage) | — |
| MVBench (2D) | — | 58.1 | 58.6 (LLaVA-Video) | −0.5 |

### Ablation Study

| Configuration | ScanQA EM@1 | SQA3D EM@1 | ScanRefer Acc@0.25 | Inference Time |
|---------------|-------------|------------|-------------------|----------------|
| SAM+CLIP w/ PE + Q-Former + Vicuna-7B | 21.9 | 49.3 | — | 900s |
| CLIP w/ PE + Pooling+MLP + Vicuna-7B | 23.4 | 51.2 | 43.8 | 0.2s |
| CLIP w/ PE + Pooling+MLP + LLaVA-1.5-7B | 27.0 | 55.6 | 47.9 | 0.2s |
| CLIP w/ PE + MLP + LLaVA-Video-7B | **30.6** | **60.1** | **50.1** | 0.2s |
| 2D Patch (without 3D positional embedding) | — | 59.8 | — | — |
| 3D Patch | — | 60.1 (+0.3) | — | — |
| 3D Patch (MMScan QA) | **55.4** vs 42.1 | — | — | — |
| 3D Patch (Scan2Cap) | **84.1** vs 29.7 | — | — | — |

### Key Findings

- **The benefit of 3D Patches is task-dependent**: On simple 3D QA tasks primarily relying on linguistic descriptions such as ScanQA and SQA3D, 3D positional embeddings yield only marginal gains (+0.3–0.4). However, on tasks requiring genuine 3D spatial understanding (e.g., MMScan QA +13.3, Scan2Cap +54.4), 3D Patches are critical.
- **Building from a 2D LMM rather than a plain LLM provides clear advantages**: Starting from LLaVA-1.5 instead of Vicuna-7B improves ScanRefer by 4.1%, validating the value of 2D pretraining priors.
- **Video LMMs are superior base models**: LLaVA-Video serves as a stronger backbone than LLaVA-1.5 and InternVL2.5, as multi-view 3D scene representations are inherently consistent with the temporal structure of video.
- **Inference speed is dramatically improved**: Compared to the SAM+CLIP-based approach (900s), the CLIP-only approach requires only 0.2s per scene while achieving better performance.
- **2D capabilities are largely preserved**: LLaVA-3D underperforms LLaVA-Video by only 0.5 points on MVBench and VideoMME, validating the effectiveness of the joint training strategy.

## Highlights & Insights

- **Minimalist yet effective 3D extension**: The simple additive operation "2D patch + 3D positional embedding = 3D Patch" is sufficient to upgrade a 2D LMM to a 3D LMM, obviating complex 3D encoders and feature aggregation pipelines. This design philosophy of "minimal modification, maximal reuse" is broadly instructive.
- **Necessity of the Grounding Decoder**: Experiments demonstrate that directly prompting the LLM to output 3D coordinates yields drastically inferior results (7.8 vs. 50.1), indicating that 3D spatial perception requires a dedicated decoder to extract geometric information from visual features and cannot be reduced to the LLM's numerical generation capability.
- **General extensibility**: LLaVA-3D serves as a general framework adaptable to different 2D LMMs (LLaVA-1.5, InternVL2.5, LLaVA-Video), with performance scaling alongside the strength of the 2D backbone.

## Limitations & Future Work

- The approach requires known camera parameters and depth information to construct 3D positional embeddings, limiting applicability to scenes without depth data.
- Validation is currently restricted to indoor scene datasets such as ScanNet; generalization to large-scale outdoor environments remains unexplored.
- The Grounding Decoder requires a dedicated Stage 2 training procedure to converge, adding training complexity.
- 3D Patch pooling strategies such as voxelization introduce information loss, which may adversely affect tasks requiring fine-grained spatial understanding.

## Related Work & Insights

- **vs. LEO / Chat-Scene**: These methods rely on offline 3D instance segmentation to extract object features, resulting in complex and time-consuming pipelines. LLaVA-3D directly constructs 3D representations from multi-view images, bypassing the 3D segmentation bottleneck.
- **vs. 3D-LLM / Scene-LLM**: These approaches require elaborate 2D segmentation and feature aggregation pipelines, consuming 900 seconds per scene. LLaVA-3D reduces this to 0.2 seconds.
- **vs. pure 2D LMMs (e.g., GPT-4V)**: GPT-4V and similar models achieve competitive results on 3D QA benchmarks—occasionally surpassing some 3D-specialized methods on ScanQA—yet still exhibit a clear performance gap compared to LLaVA-3D on tasks requiring precise 3D spatial understanding.

## Rating

- Novelty: ⭐⭐⭐⭐ — The idea is simple yet highly effective; the 3D Patch concept is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers 6+ 3D benchmarks with detailed ablations and 2D capability retention analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear structure, carefully designed figures, and well-motivated arguments.
- Value: ⭐⭐⭐⭐⭐ — Provides a simple and efficient pathway for extending 2D LMMs to 3D; expected to have broad impact.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation](easy3d_a_simple_yet_effective_method_for_3d_interactive_segmentation.md)
- [\[ICCV 2025\] A Simple yet Mighty Hartley Diffusion Versatilist for Generalizable Dense Vision Tasks](a_simple_yet_mighty_hartley_diffusion_versatilist_for_genera.md)
- [\[ICCV 2025\] Ross3D: Reconstructive Visual Instruction Tuning with 3D-Awareness](ross3d_reconstructive_visual_instruction_tuning_with_3d-awareness.md)
- [\[ICCV 2025\] GaussianProperty: Integrating Physical Properties to 3D Gaussians with LMMs](gaussianproperty_integrating_physical_properties_to_3d_gaussians_with_lmms.md)
- [\[ICCV 2025\] Describe, Adapt and Combine: Empowering CLIP Encoders for Open-set 3D Object Retrieval](describe_adapt_and_combine_empowering_clip_encoders_for_open-set_3d_object_retri.md)

<!-- RELATED:END -->
