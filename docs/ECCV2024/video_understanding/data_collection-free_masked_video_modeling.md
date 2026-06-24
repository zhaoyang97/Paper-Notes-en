---
title: >-
  [Paper Note] Data Collection-Free Masked Video Modeling
description: >-
  [ECCV 2024][Video Understanding][Self-Supervised Learning] This paper proposes a Pseudo-Motion Generator (PMG) to recursively generate pseudo-motion videos from static images. Combined with Masked Video Modeling (VideoMAE) for self-supervised pre-training, it entirely eliminates the collection costs, privacy, and copyright concerns of real video data, and even enables effective video Transformer pre-training using only synthetic images.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Self-Supervised Learning"
  - "Masked Video Modeling"
  - "Pseudo-Motion Video"
  - "Synthetic Data"
  - "VideoMAE"
date: 2026-05-08
content_hash: 0942f73af253c82f
---

# Data Collection-Free Masked Video Modeling

**Conference**: ECCV 2024  
**arXiv**: [2409.06665](https://arxiv.org/abs/2409.06665)  
**Code**: None  
**Area**: Video Understanding  
**Keywords**: Self-Supervised Learning, Masked Video Modeling, Pseudo-Motion Video, Synthetic Data, VideoMAE

## TL;DR

This paper proposes a Pseudo-Motion Generator (PMG) to recursively generate pseudo-motion videos from static images. Combined with Masked Video Modeling (VideoMAE) for self-supervised pre-training, it entirely eliminates the collection costs, privacy, and copyright concerns of real video data, and even enables effective video Transformer pre-training using only synthetic images.

## Background & Motivation

Pre-training video Transformers faces multiple data-related challenges:

**High Collection Cost**: Video data is massive, making downloading, storing, and pre-processing extremely resource-intensive.

**Copyright Risks**: Videos on platforms like YouTube are prohibited from downloading by default, and datasets like Kinetics pose potential legal issues.

**Privacy Concerns**: Videos often contain personally identifiable information (PII) such as faces.

**Bias and Ethical Issues**: Large-scale datasets may contain biases regarding nationality, gender, and age.

**Data Accessibility**: Some datasets are only accessible to specific research groups.

Limitations of prior work:
- VPN (Perlin Noise videos) and SynAPT still require real videos for cooperative use.
- MoSI (generating pseudo-motion from images) is only applicable to CNN-based architectures and **fails to train ViTs effectively**.
- No prior work has successfully pre-trained video Transformers using only synthetic images.

## Method

### Overall Architecture

1. PMG generates pseudo-motion videos from static images.
2. Training VideoMAE with the generated videos (masking ratio of 0.75).
3. Fine-tuning on downstream action recognition tasks.

Core Hypothesis: VideoMAE primarily learns low-level features (e.g., temporal patch correspondences across frames) rather than high-level semantics. Therefore, as long as patches in the pseudo-motion video are traceable, effective training is achievable.

### Pseudo-Motion Generator (PMG)

PMG generates videos by **recursively applying image transformations**: it randomly selects a transformation $f$ and an intensity parameter $\theta$ from a transformation set, and recursively applies them to the initial image: $I_{i+1} = f(I_i)$, finally concatenating them into a video $V = [I_1; I_2; ...; I_T]$.

### Eight Candidate Image Transformations

| Transformation | UCF101 | HMDB51 | Characteristics |
|------|--------|--------|------|
| Identity (Baseline) | 72.7 | 35.6 | No motion |
| Sliding Window | 75.1 | 40.5 | Random window shifting |
| **Zoom-in/out** | **81.2** | **44.5** | Window scaling |
| Fade-in/out | 76.3 | 34.1 | Fade-in/out |
| Affine | 80.5 | 43.2 | Affine transformation |
| **Perspective** | **82.7** | **45.9**| Perspective transformation |
| Color Jitter | 76.2 | 38.7 | Color jittering |
| CutMix | 76.8 | 45.1 | Mixing + shifting |

Key Finding: Transformations where patches remain traceable across frames (Zoom/Affine/Perspective) perform well, whereas transformations that only modify color/brightness yield poor performance on motion-sensitive datasets.

### Optimal Combination of Transformations

Based on experiments, **Zoom-in/out + Affine** is selected as the optimal combination (HMDB51: 51.8%), whereas CutMix degrades performance due to temporal discontinuity.

### Video-level Augmentation: Mixup

Applying Mixup frame-by-frame to the generated pseudo-motion videos significantly enhances diversity:

| Augmentation Method | HMDB51 | UCF101 |
|----------|--------|--------|
| No Augmentation | 51.8 | 83.8 |
| **Mixup** | **55.9** | **87.3** |
| VideoMix | 53.0 | 85.2 |

### Integration with Synthetic Images

Three types of synthetic image datasets are utilized: FractalDB (fractal geometry), Shaders1k (OpenGL shaders), and Visual Atom (sine waves).

### Loss & Training

- Backbone: ViT-Base, masking ratio 0.75, 2000 epochs
- Video frames: 16 frames, 224x224 resolution
- 8x A100 GPUs

## Key Experimental Results

### Main Results

| Method | Data Source | Data Amount | UCF101 | HMDB51 | Diving48 |
|------|--------|--------|--------|--------|----------|
| Scratch ViT-B | - | - | 51.4 | 18.0 | 17.9 |
| VideoMAE(FT data) | Real Video | - | 91.3 | 62.6 | 79.3 |
| VideoMAE(K400) | Real Video | 260k | 96.1 | 73.3 | - |
| MoSI(ViT-B) | Real Image | - | 48.0 | 27.3 | 14.2 |
| PPMA | Real + Synthetic Video | 300k | 92.5 | 71.2 | 64.0 |
| **Ours(FT frames)** | **Real Image** | **-** | **87.3** | **55.9** | **68.3** |
| **Ours(PASS)** | **Real Image** | **100k** | **89.3** | **60.0** | **69.2** |
| **Ours(Shaders1k)** | **Synthetic Image** | **100k** | **89.4** | **59.7** | **72.3** |

### Synthetic Image Pre-training

| Synthetic Dataset | Data Amount | UCF101 | HMDB51 |
|------------|--------|--------|--------|
| FractalDB | 100k | 78.1 | 41.1 |
| **Shaders1k** | **100k** | **89.6** | **59.7** |
| Visual Atom | 100k | 82.6 | 48.2 |

### PMG as Video Augmentation

| Pre-training Data | HMDB51 | UCF101 |
|------------|--------|--------|
| Real Video Only | 62.6 | 91.3 |
| Pseudo-Motion Only | 55.9 | 87.3 |
| **Real + Pseudo-Motion** | **64.6** | **92.2** |

### Key Findings

- **Synthetic images can totally substitute real data**: The accuracy of Shaders1k on UCF101 (89.4) outperforms using real video frames (87.3).
- Data diversity is more critical than semantic relevance: PASS (without humans) achieves comparable results to action video frames.
- MoSI fails completely on ViTs (48.0 vs. 87.3 for Ours), demonstrating that PMG is a vital innovation.
- On Diving48, Shaders1k (72.3) significantly outperforms FT data (68.3), showing that synthetic data offers greater advantages in motion-intensive tasks.
- PMG can also serve as a data augmentation method for real videos (+2% Gain).

## Highlights & Insights

1. **Complete Decoupling of Data and Pre-training**: It is demonstrated for the first time that video Transformers can be effectively pre-trained using only synthetic images.
2. **Reveals the Learning Essence of VideoMAE**: VideoMAE primarily learns low-level spatio-temporal correspondences between patches rather than high-level semantic information.
3. **Simple and Effective PMG Design**: PMG recursively applies standard image transformations without requiring complex generative models.
4. **Dual-use Utility**: It can be used both as an independent pre-training strategy (obviating data collection) and as a robust augmentation for real videos.

## Limitations & Future Work

- A performance gap still exists between purely synthetic pre-training and real video pre-training (UCF: 89.4 vs. 96.1), leaving significant room for improvement.
- The pseudo-motion patterns are relatively simple (affine/zoom-in/out) and cannot simulate complex human movements or object interactions.
- Only ViT-Base has been validated; whether larger models can benefit further remains unexplored.
- The poor performance of FractalDB and Visual Atom implies constraints on the properties of synthetic images, limiting their general applicability.

## Related Work & Insights

- The limitations of MoSI, which only applies to CNNs, motivated the exploration of ViT-friendly designs in this study.
- The characteristic of VideoMAE learning low-level features is cleverly exploited; what was previously considered a limitation has become the foundation of this method's success.
- The integration with synthetic image pre-training opens up possibilities for complete data-free pre-training.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First to achieve synthetic image-to-video Transformer pre-training, opening a new paradigm)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (6 downstream datasets + multiple synthetic/real image sources + extensive ablation studies)
- Writing Quality: ⭐⭐⭐⭐ (Well-motivated objectives, clearly organized experiments)
- Value: ⭐⭐⭐⭐ (High practical value, though a performance gap with real data remains)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition](masked_video_and_body-worn_imu_autoencoder_for_egocentric_action_recognition.md)
- [\[ICCV 2025\] Towards Efficient General Feature Prediction in Masked Skeleton Modeling](../../ICCV2025/video_understanding/towards_efficient_general_feature_prediction_in_masked_skeleton_modeling.md)
- [\[ECCV 2024\] Text-Guided Video Masked Autoencoder](text-guided_video_masked_autoencoder.md)
- [\[ECCV 2024\] Rethinking Video-Text Understanding: Retrieval from Counterfactually Augmented Data](rethinking_video-text_understanding_retrieval_from_counterfactually_augmented_da.md)
- [\[ECCV 2024\] Exploring the Feature Extraction and Relation Modeling For Light-Weight Transformer Tracking](exploring_the_feature_extraction_and_relation_modeling_for_light-weight_transfor.md)

</div>

<!-- RELATED:END -->
