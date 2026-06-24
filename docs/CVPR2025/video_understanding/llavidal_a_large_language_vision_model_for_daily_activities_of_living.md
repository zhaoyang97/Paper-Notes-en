---
title: >-
  [Paper Note] LLAVIDAL: A Large Language Vision Model for Daily Activities of Living
description: >-
  [CVPR 2025][Video Understanding][Daily activity understanding] To address daily activities of living (ADL) understanding, a multi-view multimodal instruction-tuning dataset ADL-X is constructed, and the LLAVIDAL model is proposed to integrate video, 3D skeleton, and HOI cues, achieving SOTA performance through the MMPro progressive training strategy.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Daily activity understanding"
  - "large language-vision models"
  - "multimodal fusion"
  - "skeleton features"
  - "human-object interaction"
date: 2026-05-08
content_hash: 703141e234acad44
---

# LLAVIDAL: A Large Language Vision Model for Daily Activities of Living

**Conference**: CVPR 2025  
**arXiv**: [2406.09390](https://arxiv.org/abs/2406.09390)  
**Code**: [https://adl-x.github.io/](https://adl-x.github.io/)  
**Area**: Video Understanding  
**Keywords**: Daily activity understanding, large language-vision models, multimodal fusion, skeleton features, human-object interaction

## TL;DR

To address daily activities of living (ADL) understanding, a multi-view multimodal instruction-tuning dataset ADL-X is constructed, and the LLAVIDAL model is proposed to integrate video, 3D skeleton, and HOI cues, achieving SOTA performance through the MMPro progressive training strategy.

## Background & Motivation

Existing large language-vision models (LLVMs) are primarily trained on web videos, excelling in sports, movies, and similar scenarios, but they suffer from significant limitations in understanding **activities of daily living (ADL)**:

1. **Difficulty in fine-grained action differentiation**: ADL involves subtle action differences (e.g., "picking up a cup" vs. "putting down a cup") that models trained on web videos struggle to capture.
2. **Lack of unstructured temporal processing capability**: Unlike cooking tutorials, ADL videos lack strict temporal structures and may contain irrelevant, interrupting actions (e.g., answering a phone call while cooking).
3. **Lack of crucial modalities**: 3D skeletons (viewpoint-invariant representations) and Human-Object Interactions (HOI) are critical cues for understanding ADL, yet existing LLVMs do not utilize them.
4. **Absence of ADL-specific instruction-tuning datasets**

## Method

### Overall Architecture

LLAVIDAL consists of three primary components:

- **ADL-X Dataset**: A dataset of 100K video-instruction pairs based on NTU RGB+D 120, containing multi-view RGBS (RGB + Skeleton) data.
- **Multimodal Feature Extraction**: Three-way features including video (CLIP-L/14), skeleton (SkeletonCLIP), and HOI (OWLv2).
- **MMPro Progressive Training**: A three-stage pipeline to progressively integrate each modality into the LLM embedding space.

### Key Designs

1. **ADL-X Dataset Construction**: A semi-automated framework consisting of three core strategies:
    - **Person Augmented Generation (PAG)**: Utilizes skeleton data to crop human regions, reducing background interference and forcing the AI annotator to focus on human actions.
    - **Temporal Stitching (TS)**: Stitches short video clips into longer videos to simulate unstructured action patterns in ADL, using GPT to generate 160 combined action sequences.
    - **Weakly Supervised Descriptions (WS)**: CogVLM is first used to generate frame-level descriptions at 0.5 fps, followed by GPT-3.5 utilizing weakly supervised action labels to generate coherent video descriptions (limited to 300 words), thereby reducing hallucinations.

2. **Multimodal Feature Integration**: Explores three ways to inject skeleton/HOI information into LLVMs:
    - **ℳ as features**: Skeletons use SkeletonCLIP to extract language-aligned features $\mathcal{X}_s \in \mathbb{R}^{F_s \times D_s}$; HOI detects action-related objects via BLIP2 and performs localization and tracking via OWLv2, extracting features $\mathcal{X}_o^j \in \mathbb{R}^{8 \times D_o}$.
    - **ℳ as QA**: Converts skeleton/HOI coordinates into natural language QA pairs and adds them to the training set.
    - **ℳ as context**: Appends description of skeleton/HOI motion into the text query.

3. **MMPro Progressive Training**: A three-stage curriculum learning strategy to address the difficulty of synchronous multimodal alignment:
    - **Stage 1**: Each modality is aligned independently to the LLM embedding space via a linear projection layer $Q_m = \mathcal{T}_m(\mathcal{X}_m; \theta_m)$.
    - **Stage 2**: Joint alignment of video + skeleton (the simpler modality is integrated first).
    - **Stage 3**: Integrating the HOI modality so all three are combined. The order is determined by curriculum loss, since skeleton features are easier to align than HOI.
    - During inference, only video input is required, without needing crops or extra modalities.

### Loss & Training

- Standard causal language modeling loss: $\min_\theta L_{CE}(\text{LLM}(\mathcal{T}_v(\mathcal{X}_i^v)), y_i)$
- 8 × NVIDIA RTX A6000, batch size 32, learning rate $2e^{-5}$, 3 epochs
- Video input $T=100$ frames, resolution $224 \times 224$
- Vision feature dimension $D_v=1024$, skeleton feature dimension $D_s=216$, object feature dimension $D_o=512$, LLM embedding dimension $K=4096$

## Key Experimental Results

### Main Results

| Dataset/Task | Metric | LLAVIDAL | Prev. SOTA | Gain |
|------------|------|----------|----------|------|
| Charades (MCQ-AR) | Accuracy | **55.2** | 53.1 (ChatUniVi) | +2.1 |
| Smarthome (MCQ-AR)| Accuracy | **48.1** | 48.1 (ChatUniVi) | Tie |
| LEMMA (MCQ-TC) | Accuracy | **34.3** | 32.6 (VideoLlama)| +1.7 |
| TSU (MCQ-TC) | Accuracy | **38.2** | 36.4 (ChatUniVi) | +1.8 |
| Charades (Description) | Avg | **48.6** | 42.4 (ADL-X ChatGPT) | +6.2 |
| TSU (Description) | Avg | **70.8** | 64.8 (ADL-X ChatGPT) | +6.0 |

### Ablation Study

| Configuration | Char AR | SH AR | TSU TC | TSU Desc | Description |
|------|---------|-------|--------|----------|------|
| ADL-X ChatGPT baseline | 51.0 | 44.5 | 29.5 | 64.8 | Baseline |
| + Skeleton Features (SF) | 52.7 | 42.6 | 30.3 | 66.5 | Skeleton is effective |
| + HOI Features (OF) | 53.8 | 48.0 | 37.1 | 68.0 | HOI is most critical |
| Joint (SF+OF) | 53.8 | 40.7 | 33.1 | 65.8 | Direct joint is suboptimal |
| MMPro Prog.A (Token) | **55.2** | **48.1** | **38.2** | **70.8** | Progressive is optimal |
| MMPro Prog.B (Token) | 52.8 | 49.4 | 33.0 | 69.2 | Different order is worse |
| X-InstructBLIP | 49.0 | 45.6 | 29.9 | 65.5 | Other methods underperform |

### Key Findings

- **Validation of ADL-X's Three Strategies**: Human evaluation on 100 QA pairs yields an average score of 4.1/5.0, proving high data quality.
- **HOI features are far more effective than HOI QA/context**: QA format (50.4) and context format (50.3) are significantly inferior to direct features (53.8).
- **Joint Skeleton Features + Skeleton Context (SC+SF)** is optimal before Stage 2, but after the final MMPro integration, skeleton context is no longer needed.
- **MMPro order is key**: Prog.A (video $\rightarrow$ skeleton $\rightarrow$ HOI) outperforms Prog.B (video $\rightarrow$ HOI $\rightarrow$ skeleton), as the sparser HOI features are harder to align.
- LLAVIDAL only requires video input during inference, without extra modalities, rendering it highly practical.
- Outperforms all LLVMs trained on 10x larger datasets (e.g., CogVLM trained on 1.5B images).

## Highlights & Insights

- **The first ADL-oriented LLVM**, filling the gap in daily activity understanding in the era of large models.
- **The semi-automated data construction pipeline** is highly generalizable and can be extended to other domain-specific datasets.
- **MMPro progressive training** represents an elegant solution to address conflicting gradients during simultaneous multimodal alignment, utilizing curriculum learning principles to determine the integration order of modalities.
- Dispensing with skeleton/HOI during inference is a highly practical design — utilizing multimodal synergy to boost learning during training while keeping inference unimodal and simple.

## Limitations & Future Work

- Modest data diversity, as data is sourced solely from NTU RGB+D 120, limiting background and subject variety.
- Skeleton data is derived from ground-truth annotations; in real-world scenarios, skeleton estimation may be noisy or inaccurate.
- HOI extraction relies on BLIP2 + GPT-3.5 filtering, risking the introduction of noise.
- Only Vicuna 13B is validated as the LLM backbone, leaving stronger base models unexplored.
- QA pair generation relies heavily on GPT-3.5, presenting uncontrollable hallucination issues.

## Related Work & Insights

- Compared to VideoLLaVA, VideoChat, etc., LLAVIDAL's innovation lies in its domain-focused approach and the introduction of extra modalities.
- SkeletonCLIP's method of encoding skeletons into language space is highly instructive: cross-modal alignment is achieved via dual-encoder contrastive learning.
- The Temporal Stitching strategy simulating ADL's unstructured temporality is simple yet highly effective.
- MMPro can be generalized to any multimodal alignment scenario, such as video + audio + text.

## Rating

- Novelty: ⭐⭐⭐⭐ First ADL LLVM + MMPro strategy is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple benchmarks and comprehensive ablation studies, though from a single data source.
- Writing Quality: ⭐⭐⭐⭐ Detailed methodology explanation, albeit slightly wordy.
- Value: ⭐⭐⭐⭐ Tailored for practical ADL scenarios, a trinity of data, model, and benchmark.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking](mambavlt_time-evolving_multimodal_state_space_model_for_vision-language_tracking.md)
- [\[CVPR 2025\] VoCo-LLaMA: Towards Vision Compression with Large Language Models](voco-llama_towards_vision_compression_with_large_language_models.md)
- [\[CVPR 2025\] Video Summarization with Large Language Models](video_summarization_with_large_language_models.md)
- [\[CVPR 2025\] PAVE: Patching and Adapting Video Large Language Models](pave_patching_and_adapting_video_large_language_models.md)
- [\[CVPR 2025\] On the Consistency of Video Large Language Models in Temporal Comprehension](on_the_consistency_of_video_large_language_models_in_temporal_comprehension.md)

</div>

<!-- RELATED:END -->
