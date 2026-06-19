---
title: >-
  [Paper Note] iFinder: Structured Zero-Shot VLM Grounding for Dash-Cam Video Reasoning
description: >-
  [NeurIPS 2025][Multimodal VLM][dash-cam video analysis] This paper proposes iFinder, a modular training-free framework that decouples dash-cam video understanding into perception (structured scene representation) and rea…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "dash-cam video analysis"
  - "LLM grounding"
  - "structured reasoning"
  - "zero-shot"
  - "vision-language models"
date: 2026-05-08
content_hash: 085d62826e3e6d5f
---

# iFinder: Structured Zero-Shot VLM Grounding for Dash-Cam Video Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2509.19552](https://arxiv.org/abs/2509.19552)  
**Code**: None  
**Area**: Multimodal VLM
**Keywords**: dash-cam video analysis, LLM grounding, structured reasoning, zero-shot, vision-language models

## TL;DR

This paper proposes iFinder, a modular training-free framework that decouples dash-cam video understanding into perception (structured scene representation) and reasoning (LLM). Through a hierarchical data structure and a three-block prompting strategy, iFinder endows LLMs with interpretable spatiotemporal reasoning capabilities, achieving zero-shot superiority over end-to-end V-VLMs across four driving video benchmarks, with accident reasoning accuracy gains of up to 39%.

## Background & Motivation

Applying general-purpose LLMs to post-hoc analysis of driving videos presents three key challenges:

**Weak spatial reasoning**: End-to-end V-VLMs lack structured inductive biases, leading to misinterpretation of critical visual cues such as object orientation and distance changes.

**Difficulty in causal inference**: V-VLMs rely on implicit visual features and lack verifiable reasoning chains.

**Single-modality constraint**: Dash-cam videos typically only provide a forward-facing camera, with no auxiliary sensors such as LiDAR or GPS.

Existing driving-specific V-VLMs (DriveMM, WiseAD) are designed for real-time driving decisions rather than post-hoc video analysis. General-purpose V-VLMs (VideoLLaMA2, VideoLLaVA) produce plausible-sounding responses but frequently fail at fine-grained scene understanding.

The central thesis is that **perception should be decoupled from reasoning**—specialized vision models should extract structured scene information, while LLMs perform symbolic reasoning, rather than requiring V-VLMs to handle both tasks simultaneously.

## Method

### Overall Architecture

iFinder is an 8-step pipeline that converts raw video into hierarchical structured data $\mathcal{D}$, which is then fed to an LLM via a three-block prompt to generate the final reasoning output:

$$\mathcal{F}: \mathbb{R}^{T \times H \times W \times 3} \to \mathcal{D}$$

Each step employs an independent pretrained vision model, and no training or fine-tuning is required throughout the pipeline.

### Key Designs

#### Step 1: Frame Distortion Correction
GeoCalib estimates camera intrinsics and distortion coefficients, and OpenCV's undistort corrects lens distortion from the front-facing camera, ensuring optimal input for downstream perception models.

#### Step 2: Global Scene Understanding
An image VLM (InternVL) extracts environmental information (weather, road structure, day/night), while a video VLM (VideoLLaMA2) generates event descriptions, yielding $D_{scene}$ and $D_{video}$ respectively.

#### Step 3: Ego-Vehicle State Estimation
DROID-SLAM is used for camera pose estimation, and the following quantities are computed from the translation vector sequence:
- **Steering estimation**: heading angle change $\Delta\theta_t$ between frames classifies motion as straight/left-turn/right-turn.
- **Motion estimation**: displacement speed $s_t = \|T_{t+g} - T_t\| / g$ within a temporal window determines stopped/moving status.

#### Step 4: 2D Object Detection and Tracking
OWL-V2 detects 18 driving-related object classes (vehicles, pedestrians, traffic signs, etc.), and ByteTracker assigns unique IDs for frame-by-frame tracking.

#### Step 5: Object Lane Localization
The OMR lane detection model obtains lane lines, divides the road into lane segments, and matches each object to a corresponding lane $\lambda_{t,i}$ using the midpoint of the bounding box bottom edge.

#### Step 6: Object Distance Estimation
Metric3D predicts a metric depth map $D_t$, SAM generates object segmentation masks $M_{t,i}$, and object distance is computed as $d_{t,i} = \text{mean}(D_{t,i} \odot M_{t,i})$.

#### Step 7: Object Attribute Extraction
InternVL extracts attributes (color, type, etc.) from cropped object regions to enrich the LLM's interpretable reasoning.

#### Step 8: 3D Detection for Object Orientation
CenterTrack performs 3D detection to extract yaw angle $\theta_{t,i} \in [-\pi, \pi]$, and the Hungarian algorithm associates 3D bounding boxes with 2D detections.

#### Hierarchical Data Structure Design

All information is organized into a JSON-format hierarchical structure:
- **Video-level**: environmental information, ego-vehicle state, event descriptions, peer VLM responses.
- **Frame-level**: per-frame object lists (ID, bounding box, category, distance, attributes, orientation, lane).

#### Three-Block Prompting Strategy

1. **Key Explanation**: Precisely explains the semantics of structured data to eliminate ambiguity.
2. **Step Instructions**: Decomposes the reasoning task into explicit sub-goals in a chain-of-thought manner.
3. **Peer Instruction**: Informs the LLM that peer VLM responses may be unreliable, encouraging independent reasoning.

### Loss & Training

**Fully training-free**—all modules use pretrained weights without fine-tuning. Inference can be parallelized across modules. GPT-4o-mini serves as $\mathcal{F}_{LLM}$ for final reasoning.

## Key Experimental Results

### Main Results

Zero-shot evaluation across four driving video benchmarks:

| Method | MM-AU (%) | SUTD (%) | LingoQA (Lingo-J) | Nexar (Acc%) |
|------|-----------|----------|--------------------|--------------|
| VideoLLaMA2 | 52.89 | 47.51 | 36.00 | 50.0 |
| VideoChat2 | 49.56 | 42.17 | 41.20 | 58.0 |
| DriveMM | 24.22 | 43.90 | — | 49.0 |
| **iFinder** | **63.39** | **50.93** | **44.20** | **62.0** |

iFinder surpasses the driving-specific model DriveMM by **39 percentage points** on MM-AU.

| SUTD Subcategory | U | F | R | C | I | A |
|-----------|------|------|------|------|------|------|
| VideoLLaMA2 | 49.2 | 39.0 | 48.5 | 53.5 | 35.8 | 45.2 |
| **iFinder** | **52.2** | **43.5** | **50.2** | **56.8** | **39.2** | **49.6** |

iFinder achieves the best performance across all six cognitive ability dimensions of SUTD.

### Ablation Study

Contribution of each visual module to MM-AU accuracy (accuracy drop upon removal):

| Removed Component | Accuracy (%) | Drop |
|----------|-----------|------|
| Full iFinder | 63.39 | — |
| w/o scene understanding | 57.81 | -5.58 |
| w/o orientation estimation | 58.83 | -4.56 |
| w/o object attributes | 59.04 | -4.35 |
| w/o distortion correction | 60.47 | -2.92 |
| w/o distance estimation | 60.62 | -2.77 |
| w/o lane detection | 61.80 | -1.59 |

### Key Findings

1. **Object orientation and global environmental context are the most critical**—surprisingly, they outweigh distance and lane information in importance.
2. Under extreme weather (fog/rain/snow) and nighttime conditions, iFinder still maintains the highest accuracy (Foggy: 75%, Rainy: 65.52%).
3. Even retaining <1% of detected objects (high-confidence threshold), accuracy remains at 58.27%, demonstrating robustness to error propagation.
4. Among the prompting blocks, the Key Explanation block is the most critical; its absence causes the LLM to produce invalid formatted outputs.

## Highlights & Insights

1. **Perception–reasoning decoupling** is the core innovation: decomposing the black-box V-VLM into "expert perception modules + symbolic reasoning LLM," where each component can be independently upgraded.
2. The combination of **structured representation and prompt engineering** enables a general-purpose LLM to surpass specialized models on domain-specific tasks.
3. **Surprising finding**: object orientation (rot_y) and global context are more important than distance and lane information, providing new insight into information prioritization for driving scene understanding.
4. **Fully zero-shot**: no domain fine-tuning is required, and the modular design allows upgrading any individual component to improve overall performance.

## Limitations & Future Work

1. Inference efficiency is constrained by multi-module sequential execution (scene understanding ~67s, attribute estimation ~67s), making real-time deployment infeasible.
2. The framework lacks reasoning capabilities for ambiguous or socially-motivated driving behaviors (e.g., yielding intent, courtesy behavior).
3. Reliance on GPT-4o-mini as the reasoning core introduces dependency on a closed-source model.
4. CenterTrack's 3D detection is adapted only to NuScenes categories, limiting orientation estimation for novel object classes.

## Related Work & Insights

- **V-VLM driving analysis**: DriveMM and WiseAD target real-time driving, while iFinder is positioned for post-hoc analysis; the two are complementary.
- **Modular vs. end-to-end**: iFinder demonstrates that modular approaches can surpass end-to-end methods in both interpretability and accuracy.
- **Insight**: The "structured grounding" paradigm of this framework can be generalized to other domains such as medical image analysis and industrial video surveillance.

## Rating

- Novelty: ⭐⭐⭐⭐ — Perception–reasoning decoupling combined with structured grounding constitutes a new paradigm for driving video analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 4 benchmarks + multi-dimensional ablations + extreme condition analysis + error propagation analysis.
- Writing Quality: ⭐⭐⭐⭐ — Pipeline description is clear with abundant illustrations.
- Value: ⭐⭐⭐⭐ — The modular training-free design offers strong practicality and scalability.
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GUI-Rise: Structured Reasoning and History Summarization for GUI Navigation](gui-rise_structured_reasoning_and_history_summarization_for_gui_navigation.md)
- [\[NeurIPS 2025\] Training-free Online Video Step Grounding](training-free_online_video_step_grounding.md)
- [\[CVPR 2026\] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](../../CVPR2026/multimodal_vlm/docseeker_long_document_understanding.md)
- [\[NeurIPS 2025\] Video-R1: Reinforcing Video Reasoning in MLLMs](video-r1_reinforcing_video_reasoning_in_mllms.md)
- [\[NeurIPS 2025\] TOMCAT: Test-time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning](tomcat_test-time_comprehensive_knowledge_accumulation_for_compositional_zero-sho.md)

</div>

<!-- RELATED:END -->
