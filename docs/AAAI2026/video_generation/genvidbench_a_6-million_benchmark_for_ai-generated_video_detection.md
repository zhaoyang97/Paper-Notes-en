---
title: >-
  [Paper Note] GenVidBench: A 6-Million Benchmark for AI-Generated Video Detection
description: >-
  [AAAI 2026][Video Generation][AI-generated video detection] This paper introduces GenVidBench—the first large-scale AI-generated video detection dataset with 6.78 million videos…
tags:
  - "AAAI 2026"
  - "Video Generation"
  - "AI-generated video detection"
  - "benchmark dataset"
  - "cross-source cross-generator"
  - "video forensics"
  - "deepfake detection"
date: 2026-05-08
content_hash: 1a8f809a50d4d759
---

# GenVidBench: A 6-Million Benchmark for AI-Generated Video Detection

**Conference**: AAAI 2026  
**arXiv**: [2501.11340](https://arxiv.org/abs/2501.11340)  
**Code**: [Project Page](https://genvidbench.github.io/)  
**Area**: Video Generation  
**Keywords**: AI-generated video detection, benchmark dataset, cross-source cross-generator, video forensics, deepfake detection

## TL;DR

This paper introduces GenVidBench—the first large-scale AI-generated video detection dataset with 6.78 million videos, featuring cross-source and cross-generator properties, covering 11 state-of-the-art video generators, and providing rich semantic annotations.

## Background & Motivation

### State of the Field
Video generation models (e.g., Sora, Kling) are advancing rapidly, with generated video quality approaching photorealism, making the boundary between real and fake videos increasingly blurred. This raises serious concerns regarding misinformation propagation, reputational harm, and cybersecurity threats, underscoring the urgent need for effective AI-generated video detectors.

### Limitations of Prior Work

**Development of high-performance detectors is constrained by the lack of large-scale, high-quality dedicated datasets.** Existing datasets suffer from the following issues:

| Dataset | Scale | Prompt/Image | Video Pairs | Semantic Labels | Cross-Source |
|--------|------|-----------|--------|----------|------|
| GVD | 11k | ✗ | ✗ | ✗ | ✗ |
| GVF | 2.8k | ✓ | ✓ | ✓ | ✗ |
| GenVideo | 2.27M | ✗ | ✗ | ✗ | ✗ |
| GenVidDet | 2.66M | ✗ | ✗ | ✗ | ✗ |
| **GenVidBench** | **6.78M** | **✓** | **✓** | **✓** | **✓** |

- GVD and GenVideo lack original prompts, video pairs, and semantic labels, making it impossible to prevent content overlap between training and test sets.
- GVF provides prompts and semantic labels but contains only 2.8k samples and lacks a cross-source setting.
- Existing datasets use the same generators for both training and testing, resulting in low detection difficulty that does not reflect real-world scenarios.

### Core Idea
Construct a large-scale dataset with **cross-source and cross-generator** properties: training and test sets use **different video generators** and **different content sources**, compelling detectors to learn intrinsic features of generated videos rather than relying on generator-specific or content-specific biases.

## Method

### Overall Architecture

GenVidBench is designed around three core principles: **large scale, cross-source and cross-generator design, and coverage of state-of-the-art generators**.

### Key Designs

#### 1. **Dataset Composition and Pairing Strategy**

The dataset is organized into two groups of paired videos (Video Pairs), where videos within each pair share the same text prompt or source image:

**Training Set (Video Pair 1)**:

| Source | Type | Task | Resolution | Count |
|------|------|------|--------|------|
| Vript | Real | - | - | 417,566 |
| Pika | Fake | T2V & I2V | 1088×560 | 1,670,465 |
| VideoCraftV2 | Fake | T2V & I2V | 512×320 | 1,672,242 |
| ModelScope | Fake | T2V | 256×256 | 1,672,242 |
| T2V-Zero | Fake | T2V | 512×512 | 1,268,595 |

**Test Set (Video Pair 2)**:

| Source | Type | Task | Resolution | Count |
|------|------|------|--------|------|
| HD-VG-130M | Real | - | 1280×720 | 13,853 |
| MuseV | Fake | I2V | 1210×576 | 13,853 |
| SVD | Fake | I2V | 1024×576 | 13,853 |
| Mora | Fake | T2V | 1024×576 | 13,853 |
| CogVideo | Fake | T2V | 480×480 | 13,853 |
| Sora | Fake | T2V | 1920×1080 | 51 |
| Kling | Fake | T2V & I2V | - | 264 |

- **Design Motivation**: Training and test sets use entirely different generators and content sources to prevent models from distinguishing real from fake based on content or video quality alone.

#### 2. **Cross-Source and Cross-Generator Task Design**

- **Same-generator detection**: Training and testing on the same subset yields accuracy generally >97.4%, indicating the task is trivial.
- **Cross-generator detection**: Performance drops dramatically when the generator changes (e.g., training on Pika and testing on SVD yields only 54.66%), revealing the challenges of real-world scenarios.
- **Same-source vs. cross-source**: Videos generated from the same content source are easier to classify (Pair1 average 61.81% vs. cross-source 56.71%), confirming that detectors are heavily dependent on the generation source.

#### 3. **Semantic Content Annotation**

Videos are semantically categorized along three dimensions:
- **Object category**: Person, animal, architecture, nature, plant, cartoon, food, game, vehicle, other
- **Action**: Reflects temporal properties (static poses, display/exhibition, etc.)
- **Scene**: Indicates scene complexity (natural landscape vs. traffic scene)

Topics are extracted using an LLM and aggregated into an abstract classification tree (≤10 classes per dimension), providing a foundation for scene-specific analysis.

#### 4. **Lightweight Version: GenVidBench-143k**

A carefully sampled subset of 143,400 videos is drawn from the 6.78 million total, preserving representativeness and diversity while substantially reducing computational requirements.

### Loss & Training

This paper is primarily a dataset contribution. Standard video classification models are used for benchmarking, following MMAction2 default configurations with 8-frame sampling, 224×224 resolution, and batch size 8.

## Key Experimental Results

### Main Results (Cross-Source Cross-Generator Detection)

| Method | Type | MuseV | SVD | CogV | Mora | Sora | Kling | HD | Top-1 |
|------|------|-------|-----|------|------|------|-------|----|-------|
| I3D | CNN | 32.72 | 12.04 | 76.44 | 72.30 | 41.18 | 46.22 | 95.04 | 60.21 |
| SlowFast | CNN | 87.14 | 29.80 | 93.07 | 55.23 | 23.53 | 58.33 | 96.61 | 70.06 |
| TSM | CNN | 95.94 | 73.16 | 36.44 | 91.72 | 33.34 | 71.60 | 96.30 | 73.88 |
| VideoSwin | Trans. | 90.24 | 27.72 | 91.64 | 88.14 | 19.60 | 50.76 | 99.10 | 80.39 |
| MViTv2-S | Trans. | 77.08 | 44.89 | 99.91 | 76.77 | 61.36 | 31.37 | 86.98 | 80.45 |
| **DeMamba** | **Mamba** | **85.04** | **48.81** | **98.66** | **90.23** | **1.96** | **33.71** | **99.86** | **85.47** |

DeMamba achieves the highest Top-1 accuracy of 85.47%, yet the detection rate for Sora-generated videos is only 1.96%, rendering them nearly undetectable.

### Cross-Dataset Comparison

| Dataset | SlowFast | I3D | F3Net |
|--------|----------|-----|-------|
| NeuralTextures | 82.55 | - | - |
| GVF | - | 61.88 | - |
| GenVideo | - | - | 51.83 |
| **GenVidBench** | **70.06** | **60.21** | **42.52** |

All detectors achieve significantly lower performance on GenVidBench compared to other datasets, confirming its greater difficulty.

### Key Findings

1. **Cross-generator detection is extremely challenging**: Same-generator train/test accuracy exceeds 97%, whereas cross-generator accuracy can drop to ~50%.
2. **Substantial quality variation across generators**: Sora-generated videos are nearly undetectable (1.96%), while CogVideo is the easiest to detect due to poor temporal continuity.
3. **SVD-generated videos are most difficult to distinguish**: Hard-case analysis reveals that SVD produces the most severe blurring artifacts.
4. **Transformer-based models outperform CNNs**: Transformer and Mamba architectures perform better on this task.
5. **Semantic category affects detection difficulty**: Cartoons are the easiest to detect (Mean=0.209), while vehicles are the hardest (Mean=0.308).

### Scene-Specific Analysis (Plants Category)

| Method | MuseV | SVD | CogV | Mora | HD | Mean |
|------|-------|-----|------|------|-----|------|
| TimeSformer | 77.96 | 29.80 | 96.30 | 93.44 | 87.14 | **75.09** |
| VideoSwin | 57.96 | 7.35 | 92.59 | 47.88 | 98.76 | 52.86 |

Different models exhibit substantial performance variation on specific semantic categories.

## Highlights & Insights

1. **The cross-source design is the most significant contribution**: By ensuring that training and test sets share the same content source but differ in generator, content bias is effectively eliminated, forcing detectors to learn generation artifacts.
2. **Unprecedented scale**: With 6.78 million videos, GenVidBench is 2.5× larger than the previous largest dataset, GenVidDet (2.66M).
3. **Practical value of semantic annotations**: Researchers can extract targeted subsets by scene type (e.g., person, action) for focused investigation.
4. **The 143k lightweight version lowers the barrier to entry**: It substantially reduces computational demands and accelerates model iteration.

## Limitations & Future Work

1. **Small test set scale**: Each test generator contains only ~14k videos, which is highly asymmetric compared to the millions of training samples.
2. **Lower generation quality in the training set**: Pika, ModelScope, and T2V-Zero exhibit noticeably lower resolution and quality than MuseV, SVD, and Mora in the test set.
3. **Limited coverage of the latest generators**: Sora contributes only 51 samples and Kling only 264.
4. **Lack of in-depth analysis of temporal artifacts**: Frame-level temporal consistency and other temporal forensic cues are not explored.
5. **Semantic annotation relies on LLMs**: Classification quality is bounded by LLM capability.

## Related Work & Insights

- The dataset design philosophy is consistent with GenImage for image generation detection, extended to the video domain with an added temporal dimension.
- The cross-source design principle is generalizable to other generated content detection tasks (audio, 3D models, etc.).
- DeMamba's superior performance suggests the potential of SSM architectures in video forensics.

## Rating

- **Novelty**: ⭐⭐⭐ — Primarily a dataset contribution; the cross-source design is a highlight, but technical innovation is limited.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multi-model benchmarking, cross-validation, hard-case analysis, and scene-specific analysis are comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with abundant tables.
- **Value**: ⭐⭐⭐⭐⭐ — Fills a critical gap in large-scale, high-quality generated video detection datasets, offering long-term value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] D3: Training-Free AI-Generated Video Detection Using Second-Order Features](../../ICCV2025/video_generation/d3_training-free_ai-generated_video_detection_using_second-order_features.md)
- [\[CVPR 2026\] ActivityForensics: A Comprehensive Benchmark for Localizing Manipulated Activity in Videos](../../CVPR2026/video_generation/activityforensics_a_comprehensive_benchmark_for_localizing_manipulated_activity_.md)
- [\[ACL 2026\] Self-Correcting Text-to-Video Generation with Misalignment Detection and Localized Refinement](../../ACL2026/video_generation/self-correcting_text-to-video_generation_with_misalignment_detection_and_localiz.md)
- [\[CVPR 2026\] SWIFT: Sliding Window Reconstruction for Few-Shot Training-Free Generated Video Attribution](../../CVPR2026/video_generation/swift_sliding_window_reconstruction_for_few-shot_training-free_generated_video_a.md)
- [\[AAAI 2026\] OmniVDiff: Omni Controllable Video Diffusion for Generation and Understanding](omnivdiff_omni_controllable_video_diffusion_for_generation_and_understanding.md)

</div>

<!-- RELATED:END -->
