---
title: >-
  [Paper Note] Watch and Learn: Learning to Use Computers from Online Videos
description: >-
  [CVPR 2026][LLM Pretraining][Computer-Using Agent] Watch & Learn proposes using an inverse dynamics model (IDM) to automatically convert YouTube tutorial videos into executable UI trajectory data (53K+ trajectories without manual annotation), enhancing CUA capabilities with +11.1% improvement for Qwen 2.5VL-7B and +3.8% for UI-TARS-1.5-7B on OSWorld.
tags:
  - CVPR 2026
  - LLM Pretraining
  - Computer-Using Agent
  - Inverse Dynamics Model
  - Trajectory Generation
  - YouTube Tutorials
  - GUI Automation
date: 2026-05-08
content_hash: 3c9abb643e87ebf9
---

# Watch and Learn: Learning to Use Computers from Online Videos

**Conference**: CVPR 2026  
**arXiv**: [2510.04673](https://arxiv.org/abs/2510.04673)  
**Code**: N/A  
**Area**: Pretraining  
**Keywords**: Computer-Using Agent, Inverse Dynamics Model, Trajectory Generation, YouTube Tutorials, GUI Automation

## TL;DR
Watch & Learn proposes using an inverse dynamics model (IDM) to automatically convert YouTube tutorial videos into executable UI trajectory data (53K+ trajectories without manual annotation), enhancing CUA capabilities with +11.1% improvement for Qwen 2.5VL-7B and +3.8% for UI-TARS-1.5-7B on OSWorld.

## Background & Motivation
**State of the Field**: Computer-using agents (CUA) require massive high-quality UI operation trajectory data for training, but manual annotation costs approximately $1.45/task.

**Limitations of Prior Work**: (a) Manual annotation is not scalable; (b) heuristic parsing methods have low accuracy (TongUI: 72.3%); (c) YouTube tutorial videos are a rich but underutilized data source.

**Root Cause**: The web contains vast tutorial videos demonstrating real computer operations, but no tool can accurately extract action sequences from videos.

**Paper Goals**: Build a high-accuracy IDM that automatically extracts UI operation trajectories from videos, and validate that these trajectories effectively train CUAs.

**Starting Point**: Use a SigLIP-2 visual encoder + Transformer inverse dynamics model to infer user actions from consecutive screenshots.

**Core Idea**: An inverse dynamics model with 91.7% accuracy automatically annotates UI operations in YouTube tutorial videos, converting them into 53K executable trajectories.

## Method

### Overall Architecture
Four stages: (1) Build 600K+ state transition corpus to train the IDM; (2) retrieve and filter tutorial videos from YouTube; (3) use IDM to predict action sequences in videos; (4) use generated trajectories for ICL or SFT to enhance CUA.

### Key Designs

1. **Inverse Dynamics Model (IDM)**:

    - Function: Infer the user action from two consecutive screenshot frames
    - Mechanism: SigLIP-2 visual encoder + 4-layer Transformer + 3 specialized heads (action classification/coordinate prediction/text generation)
    - Accuracy: 91.7% (vs TongUI 72.3%), directly determining downstream performance

2. **Task-Aware Video Retrieval**:

    - Function: Retrieve tutorial videos related to target tasks from YouTube
    - Mechanism: Covers 69 applications across 7 categories; uses optical flow to filter static frames

3. **Trajectory Generation and Quality Control**:

    - Function: Convert videos into structured (state, action) trajectories
    - Scale: 53,125 high-quality trajectories

### Loss & Training
- IDM is trained on 600K human interaction data
- Downstream usage: ICL (providing example trajectories) or SFT (fine-tuning models)

## Key Experimental Results

### Main Results: OSWorld-Verified

| Model | Method | Baseline | +W&L | Gain |
|-------|--------|----------|------|------|
| Gemini 2.5 Flash | ICL | 19.0% | 22.0% | +3.0% |
| Claude 4 Sonnet | ICL | 43.9% | 45.5% | +1.6% |
| Qwen 2.5VL 7B | SFT | 1.9% | 13.0% | +11.1% |
| UI-TARS-1.5-7B | SFT | 27.3% | 31.1% | +3.8% |

### Ablation Study: ICL Components (Gemini)

| Config | Success Rate |
|--------|-------------|
| Frames only | 19.0% |
| + Actions | 20.1% |
| + Actions + Reasoning | 22.0% |

### Key Findings
- IDM accuracy (91.7% vs 72.3%) directly determines downstream performance
- SFT gains far exceed ICL (Qwen: +11.1% vs best ICL +3.0%)
- Auto-generated trajectories generalize across OS platforms; other methods degrade

## Highlights & Insights
- **YouTube tutorials as free CUA training data**: No manual annotation needed; only an accurate IDM is required
- **IDM accuracy is critical**: The 19.4% accuracy gap directly causes downstream performance differences
- **ICL + reasoning chain is most effective**: Providing not only example frames and actions but also reasoning processes yields the best results

## Limitations & Future Work
- YouTube video quality varies; optical flow filtering may discard valid content
- IDM training relies on existing 600K annotated data; not fully zero-annotation
- Only covers desktop OS operations; not extended to mobile platforms

## Related Work & Insights
- **vs ShowUI/OS-Atlas**: Relies on manual annotation with high cost; W&L automates annotation for scalability
- **vs TongUI**: Heuristic parsing accuracy is only 72.3%; IDM achieves 91.7%

## Rating
- Novelty: ⭐⭐⭐⭐ — Using IDM to automatically extract trajectories from videos is a novel data acquisition paradigm
- Experimental Thoroughness: ⭐⭐⭐⭐ — OSWorld + multi-model validation + ICL/SFT dual modes
- Writing Quality: ⭐⭐⭐⭐ — Clear framework
- Value: ⭐⭐⭐⭐⭐ — Solves the CUA training data bottleneck with high practical value

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] MXNorm: Reusing MXFP Block Scales for Efficient Tensor Normalisation](mxnorm_reusing_mxfp_block_scales_for_efficient_ten.md)
- [\[CVPR 2026\] Model Merging in the Essential Subspace](model_merging_in_the_essential_subspace.md)
- [\[CVPR 2026\] Evidential Transformation Network: Turning Pretrained Models into Evidential Models for Post-hoc Uncertainty Estimation](evidential_transformation_network_post_hoc_uncertainty_estimation.md)
- [\[CVPR 2026\] FlowMotion: Training-Free Flow Guidance for Video Motion Transfer](flowmotion_training-free_flow_guidance_for_video_motion_transfer.md)
- [\[CVPR 2026\] Defending Unauthorized Model Merging via Dual-Stage Weight Protection](defending_unauthorized_model_merging_via_dual-stage_weight_protection.md)

<!-- RELATED:END -->
