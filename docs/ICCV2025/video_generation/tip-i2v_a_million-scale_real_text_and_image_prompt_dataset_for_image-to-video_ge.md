---
title: >-
  [Paper Note] TIP-I2V: A Million-Scale Real Text and Image Prompt Dataset for Image-to-Video Generation
description: >-
  [ICCV 2025][Video Generation][Image-to-video] This paper constructs TIP-I2V, the first million-scale real-user text and image prompt dataset for image-to-video (I2V) generation (1,701,935 unique prompt pairs), accompanied by generated videos from five state-of-the-art I2V models. Built upon this dataset, the paper introduces TIP-Eval, a large-scale evaluation benchmark, alongside studies on user preference analysis and AI-generated video detection.
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "Image-to-video"
  - "prompt dataset"
  - "user preference analysis"
  - "video generation evaluation"
  - "deepfake video detection"
date: 2026-05-08
content_hash: 65c1e3b14fa5128d
---

# TIP-I2V: A Million-Scale Real Text and Image Prompt Dataset for Image-to-Video Generation

**Conference**: ICCV 2025
**arXiv**: [2411.04709](https://arxiv.org/abs/2411.04709)  
**Code**: [GitHub](https://tip-i2v.github.io/)  
**Area**: Video Generation / Dataset
**Keywords**: Image-to-video, prompt dataset, user preference analysis, video generation evaluation, deepfake video detection

## TL;DR

This paper constructs TIP-I2V, the first million-scale real-user text and image prompt dataset for image-to-video (I2V) generation (1,701,935 unique prompt pairs), accompanied by generated videos from five state-of-the-art I2V models. Built upon this dataset, the paper introduces TIP-Eval, a large-scale evaluation benchmark, alongside studies on user preference analysis and AI-generated video detection.

## Background & Motivation

Image-to-video (I2V) generation is an important direction in video generation research, offering greater controllability, temporal consistency, and practical utility compared to text-to-video (T2V). However, a fundamental resource gap exists: **no dedicated I2V prompt dataset is available**.

Limitations of existing prompt datasets:

1. **VidProM** (T2V) and **DiffusionDB** (T2I) contain only text prompts, lacking paired image prompts.
2. The prompt semantics of T2V and T2I differ fundamentally from those of I2V: the former describe scenes to be generated (e.g., "a dragon flies over a city"), whereas I2V text prompts are action instructions applied to objects already present in the conditioning image (e.g., "make the hair dance," "the flowers fall").
3. Existing I2V evaluation benchmarks (VBench-I2V, I2V-Bench, etc.) cover a limited number of topics (only 11–16 categories), and their prompts are expert-designed rather than sourced from real users.

These gaps prevent researchers from accurately understanding real user needs, and model evaluations fail to reflect actual usage scenarios. Accordingly, a large-scale, real-user I2V prompt dataset is urgently needed.

## Method

### Overall Architecture

The TIP-I2V construction pipeline proceeds as follows: data collection (Pika Discord channel) → prompt extraction and deduplication → image prompt recovery → metadata annotation (UUID / UserID / timestamp / subject / direction / NSFW / embeddings) → multi-model video generation → downstream research applications.

### Key Designs

1. **Large-scale real prompt collection**: Chat messages from the official Pika Discord channel spanning July 2023 to October 2024 are collected using DiscordChatExporter. Regular expressions are applied to extract text prompts and corresponding video links, yielding **1,701,935** unique text prompts after deduplication. Since the original image prompts are not directly accessible, they are recovered by parsing the first frame of each Pika-generated video, as Pika renders the user-supplied image as the video's initial frame.

2. **Rich metadata annotation**: Each data point includes:
    - UUID and anonymized UserID
    - Timestamp
    - **Subject and direction**: GPT-4o is employed to infer the target subject and the intended motion direction from each prompt
    - Text embeddings (text-embedding-3-large) and image embeddings (CLIP)
    - NSFW status: text is screened via Detoxify; images are screened via nsfw_image_detection

3. **Multi-model video generation**: Beyond the original Pika videos, 100K randomly sampled prompts are used to generate videos with the following models:
    - Stable Video Diffusion
    - Open-Sora
    - I2VGen-XL
    - CogVideoX-5B

   Each model produces 100,000 videos, enabling cross-model comparative studies.

4. **TIP-Eval evaluation benchmark**: From the 1,000 most popular subjects, 10 real user text–image prompt pairs per subject are selected, yielding 10,000 prompts for model evaluation. Compared to existing benchmarks (VBench-I2V covers only 11 subjects / 355 prompts), TIP-Eval offers substantially broader coverage and greater ecological validity.

### Loss & Training

- This paper presents a dataset contribution; no new model training is involved.
- The AI-generated video detector follows a standard classification training scheme, using a linear classifier on CLIP features (strong detector) and a CNN (for generalization experiments).

## Key Experimental Results

### Main Results

Multi-dimensional evaluation of five I2V models on TIP-Eval (10 normalized evaluation dimensions):

| Dimension | Pika | SVD | Open-Sora | I2VGen-XL | CogVideoX-5B |
|-----------|------|-----|-----------|-----------|---------------|
| Aesthetic Quality | High | Medium | Low | Low | Medium |
| Consistency | High | High | Low | Low | Medium |
| Dynamics | Low | Low | High | Medium | High |
| Video–Text Alignment | 0.26 (highest) | — | — | — | Slightly lower |

Key finding: **the earlier commercial model Pika outperforms the more recent open-source CogVideoX-5B on 8 out of 10 evaluation dimensions**, a conclusion that diverges from evaluations based on expert-designed prompts.

### Ablation Study

Generalization of existing deepfake detection methods to the I2V scenario:

| Method | Pika | SVD | Open-Sora | I2VGen-XL | CogVideoX | Avg. |
|--------|------|-----|-----------|-----------|-----------|------|
| Blind Guess | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 | 50.0 |
| CNNSpot | 50.7 | 50.3 | 50.7 | 50.3 | 50.3 | 50.5 |
| UnivFD | 48.5 | 52.0 | 53.4 | 60.9 | 50.7 | 53.1 |
| Strong Detector (in-domain) | 93.2 | 97.3 | 96.9 | 97.9 | 96.2 | **96.3** |
| Strong Detector (cross-domain) | 84.5 | 92.1 | 93.4 | 73.6 | 92.2 | **87.2** |

Existing fake-image detection methods fail entirely to generalize to I2V video detection (near random-chance performance), underscoring the necessity of training dedicated I2V detectors.

### Key Findings

- **Highly skewed user preferences**: The top-3 subjects (person, astronaut, portrait painting) are all human-related; the most frequent direction is "move."
- **Pronounced long-tail distribution**: Only 2,721 subjects and 309 directions are needed to cover 80% of user preferences.
- **Distinctive I2V prompt semantics**: WizMap visualization shows that TIP-I2V text prompt distributions differ markedly from those of VidProM and DiffusionDB.
- **Three-class video discrimination** (real / T2V-generated / I2V-generated): the strong detector achieves 96.3% in-domain accuracy and 87.2% cross-domain accuracy, confirming that I2V-generated videos exhibit detectable artifacts.

## Highlights & Insights

- TIP-I2V is the first dedicated I2V prompt dataset, filling a critical research gap.
- The 1.7M-scale real-user data provides unique insights into user behavior in practice.
- TIP-Eval substantially surpasses existing benchmarks in both coverage (1,000 subjects) and ecological validity (real user prompts).
- The finding that user-perspective evaluation can overturn expert-perspective conclusions carries meaningful implications for model development.
- The dataset is designed for extensibility — researchers can generate additional videos for new models using the existing prompts.

## Limitations & Future Work

- Image prompts are recovered from video first frames, which may introduce quality degradation (compression artifacts, etc.).
- Data sourced exclusively from Pika Discord may introduce user population bias.
- Subject/direction labels inferred by GPT-4o exhibit semantic overlap (e.g., person / people / man).
- Due to computational constraints, only 100K videos are generated per model (approximately 6% of total prompts).
- Human annotation of video quality is absent.

## Related Work & Insights

- The data collection paradigm extends VidProM (T2V) and DiffusionDB (T2I) into the I2V domain, establishing a new research direction.
- The "instructional" nature of I2V prompts stands in stark contrast to the "descriptive" prompts of T2V and T2I settings.
- The methodology provides a foundation for constructing future user-oriented video training datasets.
- Video provenance tracking (tracing generated videos back to their source images) represents a practically valuable direction for safety research.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First I2V prompt dataset with a clearly defined problem formulation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multi-model comparison, user behavior analysis, and detection experiments comprehensively.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with rich visualizations.
- **Value**: ⭐⭐⭐⭐⭐ The dataset–benchmark combination offers lasting community contributions; the 1.7M-scale prompts support a wide range of research applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] RealCam-I2V: Real-World Image-to-Video Generation with Interactive Complex Camera Control](realcam-i2v_real-world_image-to-video_generation_with_interactive_complex_camera.md)
- [\[ICCV 2025\] DH-FaceVid-1K: A Large-Scale High-Quality Dataset for Face Video Generation](dh-facevid-1k_a_large-scale_high-quality_dataset_for_face_video_generation.md)
- [\[ICCV 2025\] STiV: Scalable Text and Image Conditioned Video Generation](stiv_scalable_text_and_image_conditioned_video_generation.md)
- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)
- [\[ICCV 2025\] Versatile Transition Generation with Image-to-Video Diffusion](versatile_transition_generation_with_image-to-video_diffusion.md)

</div>

<!-- RELATED:END -->
