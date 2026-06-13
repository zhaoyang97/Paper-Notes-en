---
title: >-
  [Paper Note] rPPG-VQA: A Video Quality Assessment Framework for Unsupervised rPPG Training
description: >-
  [CVPR 2026][Human Understanding][Remote Photoplethysmography] rPPG-VQA proposes the first video quality assessment framework tailored for remote heart rate detection (rPPG)…
tags:
  - "CVPR 2026"
  - "Human Understanding"
  - "Remote Photoplethysmography"
  - "Video Quality Assessment"
  - "Unsupervised Learning"
  - "Multimodal Large Language Models"
  - "Data Curation"
date: 2026-05-08
content_hash: da95250e92ff106d
---

# rPPG-VQA: A Video Quality Assessment Framework for Unsupervised rPPG Training

**Conference**: CVPR 2026
**arXiv**: [2604.11156](https://arxiv.org/abs/2604.11156)  
**Code**: [https://github.com/Tianyang-Dai/rPPG-VQA](https://github.com/Tianyang-Dai/rPPG-VQA)  
**Area**: Human Understanding
**Keywords**: Remote Photoplethysmography, Video Quality Assessment, Unsupervised Learning, Multimodal Large Language Models, Data Curation

## TL;DR
rPPG-VQA proposes the first video quality assessment framework tailored for remote heart rate detection (rPPG), combining signal-level multi-method consensus SNR with scene-level MLLM disturbance recognition, along with a two-stage adaptive sampling strategy to curate in-the-wild training data.

## Background & Motivation

**Background**: Unsupervised rPPG aims to learn contactless heart rate detection from unannotated video data, but existing research focuses primarily on methodological innovation while neglecting data quality issues.

**Limitations of Prior Work**: (1) Motion, illumination, and other noise in in-the-wild videos may overwhelm weak physiological signals; (2) AI-generated videos lack any real physiological basis; (3) conventional VQA evaluates human perceptual quality, which is misaligned with rPPG requirements; (4) a single SNR metric is easily fooled by periodic non-physiological signals (e.g., strobe lights).

**Key Challenge**: Videos with high visual quality may contain no extractable physiological signal, while visually degraded videos may still carry valid signals — a distinction that conventional VQA cannot make.

**Core Idea**: A dual-branch assessment — the signal-level branch applies multi-method consensus SNR to eliminate algorithmic bias, while the scene-level branch employs an MLLM to identify disturbances such as motion and illumination variation.

## Method

### Overall Architecture
In-the-wild video input → Signal-level branch (multiple rPPG algorithms extract signals → consensus SNR score) + Scene-level branch (MLLM evaluates motion/illumination/compression disturbances) → Fusion into a unified quality score → Two-stage adaptive sampling → Target training set construction.

### Key Designs

1. **Signal-Level Multi-Method Consensus SNR**:

    - **Function**: Evaluates the integrity of physiological signals in a video while eliminating single-algorithm bias.
    - **Mechanism**: Multiple traditional rPPG algorithms (GREEN, ICA, CHROM, POS, etc.) independently extract signals and estimate SNR. If a genuine physiological signal is present, all methods should yield consistently high SNR (method-agnostic property); inconsistency indicates unreliable signal.
    - **Design Motivation**: A single SNR metric is susceptible to periodic noise (e.g., strobe lights producing heartbeat-like signals); multi-method consensus filters out such false positives.

2. **Scene-Level MLLM Disturbance Recognition**:

    - **Function**: Identifies scene-level disturbances that signal-level metrics cannot capture.
    - **Mechanism**: An MLLM performs human-like scene reasoning over video frames to detect complex disturbances such as unstable illumination, severe motion, and camera artifacts, producing a disturbance score.
    - **Design Motivation**: Signal-level metrics cannot distinguish the physiological origin of a signal and lack scene context to differentiate genuine biosignals from confounding artifacts.

3. **Two-Stage Adaptive Sampling (TAS)**:

    - **Function**: Constructs an optimal training set from a large-scale uncurated video pool.
    - **Mechanism**: Stage 1 applies quality thresholds to filter low-quality videos; Stage 2 employs duration-aware probabilistic sampling to balance quality, diversity, and efficiency.
    - **Design Motivation**: Naive filtering may yield insufficiently diverse training sets; probabilistic sampling maintains data diversity while ensuring quality.

### Loss & Training
The curated training set is used to train existing unsupervised rPPG methods (e.g., ContrastPhys, SiNC), validating the effectiveness of the VQA framework.

## Key Experimental Results

### Main Results

| Sampling Strategy | PURE Test Set HR MAE | Note |
|------------------|----------------------|------|
| All (full data) | High error | Low-quality data degrades training |
| Random | Moderate error | Better than using all data |
| rPPG-VQA | Lowest error | Quality curation is highly effective |

### Ablation Study

| Configuration | HR MAE | Note |
|--------------|--------|------|
| Signal-level + Scene-level | Best | Dual branches are complementary |
| Signal-level only | Second best | Scene disturbances overlooked |
| Scene-level only | Moderate | Signal quality assessment missing |

### Key Findings
- Training on all in-the-wild videos performs worse than training on a quality-curated subset.
- The dual-branch design shows clear complementary effects; each branch alone has blind spots.
- The TAS strategy maintains training set diversity while ensuring data quality.

## Highlights & Insights
- **First systematic study of data quality for rPPG**: Addresses the data-side gap in unsupervised rPPG research.
- **Method-agnostic signal quality metric**: Multi-algorithm consensus eliminates bias and the idea is transferable to other signal processing tasks.

## Limitations & Future Work
- MLLM inference incurs substantial computational cost.
- Quality threshold selection requires manual tuning.
- Future work may explore end-to-end quality-aware training frameworks.

## Related Work & Insights
- **vs. Conventional VQA (PSNR/SSIM)**: Designed for human perceptual quality, misaligned with rPPG requirements.
- **vs. Post-hoc signal evaluation**: Requires prior signal extraction and cannot pre-filter raw videos.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic treatment of data quality for rPPG
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison across multiple sampling strategies
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is precise and well-motivated
- Value: ⭐⭐⭐⭐ Unlocks the potential of in-the-wild data for unsupervised rPPG

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback](reference-free_image_quality_assessment_for_virtual_try-on_via_human_feedback.md)
- [\[CVPR 2026\] AVATAR: Reinforcement Learning to See, Hear, and Reason Over Video](avatar_reinforcement_learning_to_see_hear_and_reason_over_video.md)
- [\[CVPR 2026\] From Intuition to Investigation: A Tool-Augmented Reasoning MLLM Framework for Generalizable Face Anti-Spoofing](from_intuition_to_investigation_a_tool-augmented_reasoning_mllm_framework_for_ge.md)
- [\[CVPR 2026\] COG: Confidence-aware Optimal Geometric Correspondence for Unsupervised Single-reference Novel Object Pose Estimation](cog_confidence-aware_optimal_geometric_correspondence_for_unsupervised_single-re.md)
- [\[CVPR 2026\] Unleashing Vision-Language Semantics for Deepfake Video Detection](unleashing_vision-language_semantics_for_deepfake_video_detection.md)

</div>

<!-- RELATED:END -->
