---
title: >-
  [Paper Note] DH-FaceVid-1K: A Large-Scale High-Quality Dataset for Face Video Generation
description: >-
  [ICCV 2025][Video Generation][face video dataset] This paper introduces DH-FaceVid-1K, a large-scale high-quality face video dataset comprising 1,200+ hours, 270,043 video clips, and 20,000+ unique identities. It specifically addresses the severe underrepresentation of Asian faces in existing datasets and empirically validates scaling laws with respect to data volume and model parameter count through systematic experiments.
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "face video dataset"
  - "text-to-video"
  - "image-to-video"
  - "diffusion models"
date: 2026-05-08
content_hash: e906d575bbc3b749
---

# DH-FaceVid-1K: A Large-Scale High-Quality Dataset for Face Video Generation

**Conference**: ICCV 2025
**arXiv**: [2410.07151](https://arxiv.org/abs/2410.07151)  
**Code**: [Project Page](https://luna-ai-lab.github.io/DH-FaceVid-1K/)  
**Area**: Image Generation / Face Video Generation
**Keywords**: face video dataset, video generation, text-to-video, image-to-video, diffusion models

## TL;DR

This paper introduces DH-FaceVid-1K, a large-scale high-quality face video dataset comprising 1,200+ hours, 270,043 video clips, and 20,000+ unique identities. It specifically addresses the severe underrepresentation of Asian faces in existing datasets and empirically validates scaling laws with respect to data volume and model parameter count through systematic experiments.

## Background & Motivation

Face video generation is one of the most active tasks in video generation, underpinning applications such as talking-head synthesis and text-driven video generation. However, many state-of-the-art methods rely on proprietary private data, while publicly available datasets suffer from three core limitations:

**Insufficient total duration**: CelebV-HQ (68h) and CelebV-Text (279h) are far from adequate for pretraining needs.

**Quality–quantity trade-off**: VoxCeleb2 offers 2,400h but at only 224×224 resolution; TalkingHead-1KH is similarly resolution-constrained.

**Lack of diversity**: Asian faces are severely underrepresented in existing datasets, limiting model generalization across ethnic groups.

The paper further identifies common quality issues in existing public datasets, including low sharpness/resolution, multiple faces per frame, hand/object occlusion, and overlaid captions or noise artifacts, all of which substantially degrade training effectiveness.

## Method

### Overall Architecture

The construction of DH-FaceVid-1K proceeds in four key stages:
1. **Raw video collection**: Interview programs and vlog-style videos are collected from crowdsourcing platforms (single subject, professional environment, high-quality equipment), yielding over 2,000 hours of raw footage.
2. **Face detection and cropping**: Clips are cropped to include the full face and upper-shoulder region, ensuring a minimum face region of 256×256.
3. **Noise filtering**: Subtitle detection (OCR), letterbox detection, multi-face exclusion, manual screening for hand/occlusion artifacts, and blur enhancement via CodeFormer.
4. **Annotation generation**: DWPose is used to extract facial keypoints; PLLaVA generates initial text descriptions; 100+ human annotators perform cross-validation over six months.

### Key Designs

1. **Multi-ethnic coverage and Asian face supplementation**: Approximately 1,000 hours of in-house collected data (95% Asian faces) are combined with 200 hours of cleaned data from CelebV-HQ, CelebV-Text, and TalkingHead-1KH. The final dataset comprises 80% Asian, 11% Caucasian, 4% African, and 5% other ethnicities, addressing the demographic bias prevalent in existing datasets.

2. **Rigorous data quality control pipeline**: (1) Subtitle detection: five frames are randomly sampled per clip for OCR; clips with more than 10 characters are flagged; (2) Letterbox detection: continuous black borders exceeding 20 pixels are identified; (3) Face filtering: OpenCV discards clips with multiple faces; FaceXFormer filters out individuals under 22 years of age; (4) Two-stage manual verification: Stage 1 cross-validates static and dynamic attributes per ISO 2859 standards; Stage 2 further reviews flagged problematic samples.

3. **Audio filtering and lip-audio synchronization**: To address SyncNet score bias for non-English speech, a SyncNet model is retrained to generate synchronization scores for each clip, filtering out samples with poor lip-audio alignment and ensuring dataset suitability for audio-driven talking-head generation.

### Loss & Training

As a dataset paper, no training strategy is proposed per se. However, the following empirical guidelines are provided for downstream model training:
- The optimal data scale for a 2B-parameter DiT model is approximately 600 hours.
- Models with 5B/6B parameters require the full 1,000+ hours to achieve their performance ceiling.
- Small models overfit on large datasets; large models exhibit training instability on small datasets.

## Key Experimental Results

### Main Results (Text-to-Video)

| Method | Dataset | FVD (↓) | FID (↓) | CLIP (↑) |
|---|---|---|---|---|
| CogVideoX | HDTF | 127.88 | 17.83 | 0.9247 |
| CogVideoX | CelebV-HQ | 137.62 | 17.31 | 0.9305 |
| CogVideoX | CelebV-Text | 129.59 | 15.82 | 0.9388 |
| CogVideoX | **DH-FaceVid-1K** | **98.01** | **11.73** | **0.9401** |
| EasyAnimate | CelebV-Text | 121.22 | 16.53 | 0.9274 |
| EasyAnimate | **DH-FaceVid-1K** | **113.27** | **13.91** | 0.9240 |

### Ablation Study (Data Scaling Laws, CogVideoX T2V)

| Data Scale | 2B FVD | 2B FID | 5B FVD | 5B FID |
|---|---|---|---|---|
| 100h | 215.06 | 17.01 | 237.52 | 18.17 |
| 200h | 185.06 | 16.23 | 203.52 | 16.37 |
| 400h | 177.18 | 14.16 | 180.99 | 14.25 |
| 600h | 145.14 | 12.50 | 143.25 | 12.83 |
| 800h | 148.82 | 13.15 | 121.63 | 12.05 |
| 1000h | 150.27 | 13.31 | **98.01** | **11.73** |

The 2B model reaches a performance inflection point at 600h (additional data yields marginal or slight degradation), whereas the 5B model continues to benefit monotonically from more data.

### Key Findings

1. **All models trained on DH-FaceVid-1K significantly outperform those trained on public datasets**: CogVideoX T2V FVD decreases from 129.59 to 98.01 (−24.3%) and FID from 15.82 to 11.73 (−25.9%).
2. **Scaling laws**: The 2B model achieves optimal cost-effectiveness at ~600h; models with 5B+ parameters require the full dataset to realize their potential.
3. **DiT vs. UNet**: DiT architectures (Latte, CogVideoX) generally outperform UNet architectures (AnimateDiff), though at higher training resource cost.
4. **Models trained on CelebV-Text exhibit visible artifacts when generating Asian faces** (random hand-like noise, multi-face artifacts); models trained on DH-FaceVid-1K do not exhibit these issues.
5. **I2V tasks also benefit**: CogVideoX I2V achieves FVD 92.31 on DH-FaceVid-1K vs. 123.25 on CelebV-Text.

## Highlights & Insights

- **Exceptional data engineering value**: A two-stage cross-validation quality control pipeline with 100+ annotators over six months ensures industry-leading data quality.
- **Scaling law experiments** provide actionable guidance: different model scales correspond to distinct optimal data volumes, preventing blind data accumulation.
- **Empirical DiT vs. UNet comparison** offers a valuable reference for backbone selection.
- **Addresses the practical gap of Asian face data scarcity**, with significant implications for real-world deployment.

## Limitations & Future Work

- The 80% proportion of Asian faces leaves other ethnicities still underrepresented, potentially introducing a new demographic bias.
- The strict dataset access process (application form submission and agreement signing) may limit broad adoption by the academic community.
- The effect of generating longer videos (>15s) remains unexplored.
- Direct comparison with recent dedicated talking-head methods such as Hallo3 is absent.
- Experiments on audio-driven generation tasks are limited.

## Related Work & Insights

- The results validate the importance of domain-specific high-quality datasets for fine-tuning pretrained models, a conclusion generalizable to other vertical domains.
- The data processing pipeline (subtitle detection → letterbox removal → multi-face filtering → manual review → audio synchronization) can serve as a standard reference for video dataset construction.
- The scaling law analysis framework (model parameters × data scale × performance metrics) provides an evaluation methodology for future dataset papers.

## Rating

- **Novelty**: ⭐⭐⭐ The dataset construction pipeline is well-established but methodologically incremental; the primary contribution lies in the data itself.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Systematic experiments across 5 T2V models, 5 I2V models, 6 data scales, and 2 model parameter scales.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, comprehensive statistics, and in-depth scaling law analysis.
- **Value**: ⭐⭐⭐⭐ Fills a critical gap in large-scale high-quality face video datasets; the scaling law analysis offers practical guidance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] TIP-I2V: A Million-Scale Real Text and Image Prompt Dataset for Image-to-Video Generation](tip-i2v_a_million-scale_real_text_and_image_prompt_dataset_for_image-to-video_ge.md)
- [\[CVPR 2025\] HOIGen-1M: A Large-Scale Dataset for Human-Object Interaction Video Generation](../../CVPR2025/video_generation/hoigen-1m_a_large-scale_dataset_for_human-object_interaction_video_generation.md)
- [\[ICCV 2025\] Dual-Expert Consistency Model for Efficient and High-Quality Video Generation](dual-expert_consistency_model_for_efficient_and_high-quality_video_generation.md)
- [\[ICLR 2026\] Self-Forcing++: Towards Minute-Scale High-Quality Video Generation](../../ICLR2026/video_generation/self-forcing_towards_minute-scale_high-quality_video_generation.md)
- [\[CVPR 2026\] Scaling Instruction-Based Video Editing with a High-Quality Synthetic Dataset](../../CVPR2026/video_generation/scaling_instruction-based_video_editing_with_a_high-quality_synthetic_dataset.md)

</div>

<!-- RELATED:END -->
