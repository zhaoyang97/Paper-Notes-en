---
title: >-
  [Paper Note] HOIGen-1M: A Large-Scale Dataset for Human-Object Interaction Video Generation
description: >-
  [CVPR 2025][Video Generation][Human-Object Interaction] HOIGen-1M is the first million-scale high-quality dataset designed for Human-Object Interaction (HOI) video generation. It addresses HOI video data scarcity and description hallucination through an efficient data filtering pipeline and a Mixture-of-Multimodal-Experts (MoME) captioning strategy, while introducing two evaluation metrics, CoarseHOIScore and FineHOIScore, to quantify the quality of interaction in generated v…
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Human-Object Interaction"
  - "Text-to-Video Generation"
  - "Large-Scale Dataset"
  - "Video Captioning"
  - "Multimodal Large Language Models"
date: 2026-05-08
content_hash: 775d21a6777cea15
---

# HOIGen-1M: A Large-Scale Dataset for Human-Object Interaction Video Generation

**Conference**: CVPR 2025  
**arXiv**: [2503.23715](https://arxiv.org/abs/2503.23715)  
**Code**: [https://liuqi-creat.github.io/HOIGen.github.io](https://liuqi-creat.github.io/HOIGen.github.io)  
**Area**: Video Understanding/Video Generation  
**Keywords**: Human-Object Interaction, Text-to-Video Generation, Large-Scale Dataset, Video Captioning, Multimodal Large Language Models

## TL;DR

HOIGen-1M is the first million-scale high-quality dataset designed for Human-Object Interaction (HOI) video generation. It addresses HOI video data scarcity and description hallucination through an efficient data filtering pipeline and a Mixture-of-Multimodal-Experts (MoME) captioning strategy, while introducing two evaluation metrics, CoarseHOIScore and FineHOIScore, to quantify the quality of interaction in generated videos.

## Background & Motivation

### Background

**Background**: Text-to-video (T2V) generation has made tremendous progress; models like Sora and Kling 1.5 can generate complex scenes. However, Human-Object Interaction (HOI), a fundamental component of the physical world, remains a major pain point for current T2V models—even models with over 10B parameters struggle to accurately generate simple HOI videos (e.g., "loading a suitcase onto a bus").

**Limitations of Prior Work**: (1) Lack of large-scale HOI video data—WebVid-10M contains low-quality watermarked videos, and Panda-70M has a large amount of static/blurry videos, most of which do not contain HOI; (2) Existing HOI-aware datasets (such as CAD-120, BEHAVE, etc.) are too small in scale (ranging from thousands to tens of thousands), far below the million-scale required for training T2V models; (3) Existing captioning methods are either too brief (12-13 words) or not specifically designed for HOI, losing interaction details; (4) There is a lack of dedicated metrics to evaluate the quality of HOI video generation.

**Key Challenge**: The performance of T2V models in HOI scenarios is far inferior to that in general scenarios. The root cause is the lack of large-scale, high-quality, and precisely described HOI videos in the training data.

**Goal**: To build a million-scale high-quality HOI video dataset, design an accurate video captioning method, and propose an evaluation framework for HOI video generation.

## Method

### Overall Architecture

The construction of HOIGen-1M consists of three core components: (1) A video filtering pipeline that automatically filters high-quality HOI videos from 80 million raw videos; (2) A MoME captioning strategy that utilizes multiple multimodal large models to validate each other, eliminating hallucinations and generating precise descriptions; (3) An evaluation framework that proposes CoarseHOIScore and FineHOIScore metrics to evaluate the quality of interaction in generated videos from coarse to fine grains.

### Key Designs

1. **Efficient Video Filtering Pipeline**:
    - **Function**: Efficiently filter high-quality videos containing HOI from 80 million raw videos.
    - **Mechanism**: Five-stage cascaded filtering—(a) Metadata filtering (duration > 1s, resolution $\ge$ 720p, frame rate $\ge$ 20 FPS); (b) OCR filtering to remove text-heavy videos; (c) Aesthetic score filtering to ensure visual quality; (d) Optical flow score filtering to ensure moderate motion (excluding excessively high or low motion); (e) MLLM + LLM to determine whether HOI is present (PLLaVA generates captions + Qwen2.5 judges interaction). Ultimately, 1.5 million videos proceed to the manual verification stage.
    - **Manual verification**: Seven annotators spent eight weeks checking each video to ensure interactions are distinct and objects are visible, ultimately obtaining 1.1 million videos.

2. **Mixture-of-Multimodal-Experts (MoME) Captioning Strategy**:
    - **Function**: Generate accurate and hallucination-free HOI video descriptions.
    - **Mechanism**: (a) Two captioning experts (PLLaVA and Qwen2-VL) generate descriptions individually; (b) A decision expert (Llama 3.1) judges whether the two descriptions are consistent—inconsistency indicates a detected hallucination; (c) Upon hallucination detection, a third captioning expert is introduced to focus on the disputed areas, and then the decision expert integrates them to generate an edited description; (d) In the absence of hallucination, the decision expert selects the more informative description.
    - **Design Motivation**: A single MLLM inevitably suffers from hallucinations during video captioning. Multi-expert cross-validation can systematically detect and eliminate hallucinations.

3. **CoarseHOIScore and FineHOIScore Evaluation Metrics**:
    - **CoarseHOIScore**: Uses an HOI detector to detect the presence of HOI triplets (human, object, action) in the generated videos, calculating the frame-wise proportion that exceeds a confidence threshold.
    - **FineHOIScore**: Based on MLLM scoring, it comprehensively evaluates multiple dimensions such as interaction plausibility, motion smoothness, and human physical realism, providing a more fine-grained quality assessment.

### Loss & Training

The primary contribution of this work is the dataset rather than the models. When fine-tuning T2V models to validate the dataset's efficacy, the native training strategies of each model are utilized (such as the original training loss of CogVideoX-5B).

## Key Experimental Results

### Main Results

| Metric | Data |
|------|------|
| Dataset Scale | 1.1 million+ video clips |
| Video Duration | 2,200+ hours in total |
| Resolution | $\ge$ 720p |
| Average Caption Length | 153.8 words (WebVid-10M: 12.0, Panda-70M: 13.2) |
| Object Categories | 15,000+ |
| Interaction Action Categories | 7,000+ |
| Fine-tuned CogVideoX-5B CoarseHOIScore | Close to the level of the commercial software Kling 1.5 |
| Best Commercial Model (Kling 1.5) CoarseHOIScore | 42.72% |
| Best Open-Source Model (CogVideoX-5B) CoarseHOIScore | 32.84% |
| Hailuo CoarseHOIScore | 39.56% |
| Dreamina CoarseHOIScore | 36.36% |
| Number of Evaluation Prompts | 306 (musical instruments, vehicles, kitchenware, etc.) |
| Manual Verification Time | 7 annotators $\times$ 8 weeks |

## Highlights & Insights

1. **First Million-Scale HOI Video Generation Dataset**: It fills the data gap in HOI within the T2V field, with all videos verified manually.
2. **Hallucination Elimination Concept of MoME Strategy**: Cross-validating with multiple MLLMs rather than relying on a single model is an effective paradigm for addressing hallucinations in large-scale automatic annotating.
3. **Design of HOI Evaluation Metrics**: Introducing an HOI detector to generative evaluation is a clever cross-task transfer—it captures the core quality of interaction generation better than general-purpose metrics.
4. **Revealed Gap by Experiments**: Even the state-of-the-art commercial model, Kling 1.5, only achieves 42.72% on CoarseHOIScore, which demonstrates that HOI video generation remains a far-from-resolved challenge.

## Limitations & Future Work

1. CoarseHOIScore and FineHOIScore rely on the capabilities of current HOI detectors and MLLMs, which may fail to capture subtle differences in interaction quality.
2. The dataset is mainly sourced from public videos, where scenes and filming conditions may introduce biases.
3. Although manual verification guarantees quality, it limits the efficiency of further dataset expansion.
4. Although the average caption length of 153.8 words is much longer than existing datasets, it may still not be sufficiently detailed for complex HOI scenarios.

## Related Work & Insights

- **T2V Datasets**: WebVid-10M (10 million videos, short captions, watermarked), Panda-70M (70 million videos, short captions, highly static), OpenVid-1M (1 million videos, long captions, general scenarios).
- **HOI-Aware Datasets**: BEHAVE (15.2K frames, RGBD+SMPL), HOI4D (4,000 first-person 4D videos), GRAB (1,334 full-body + hand motion capture sequences), MPHOI-72 (72 multi-person activity videos), PVSG (400 first/third-person videos).
- **T2V Models**: Sora (minute-level video generation), CogVideoX (open-source million-scale training), Kling 1.5 (commercial video generation), OpenSora/OpenSoraPlan (open-source community solutions).
- **Video Captioning Methods**: PLLaVA (video multimodal understanding), Qwen2-VL (video captioning generation), MoME strategy (multi-expert cross-validation to eliminate hallucinations).
- **Video Quality Assessment**: VBench (16-dimensional video quality evaluation framework), FID/FVD (general generative quality metrics).

## Rating

- Novelty: ⭐⭐⭐⭐ (First million-scale HOI video generation dataset + MoME hallucination elimination + dedicated evaluation metrics)
- Value: ⭐⭐⭐⭐⭐ (Directly applicable to boost the HOI generation capabilities of T2V models)
- Technical Depth: ⭐⭐⭐ (Dataset paper is engineering-oriented, with moderate methodological innovation)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure and comprehensive analysis)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DH-FaceVid-1K: A Large-Scale High-Quality Dataset for Face Video Generation](../../ICCV2025/video_generation/dh-facevid-1k_a_large-scale_high-quality_dataset_for_face_video_generation.md)
- [\[CVPR 2026\] CineBrain: A Large-Scale Multi-Modal Audiovisual Brain Dataset for Brain-Conditioned Video Generation](../../CVPR2026/video_generation/cinebrain_a_large-scale_multi-modal_audiovisual_brain_dataset_for_brain-conditio.md)
- [\[CVPR 2025\] IDOL: Instant Photorealistic 3D Human Creation from a Single Image](idol_instant_photorealistic_3d_human_creation_from_a_single_image.md)
- [\[CVPR 2025\] MovieBench: A Hierarchical Movie Level Dataset for Long Video Generation](moviebench_a_hierarchical_movie_level_dataset_for_long_video_generation.md)
- [\[CVPR 2025\] Video-Bench: Human-Aligned Video Generation Benchmark](video-bench_human-aligned_video_generation_benchmark.md)

</div>

<!-- RELATED:END -->
