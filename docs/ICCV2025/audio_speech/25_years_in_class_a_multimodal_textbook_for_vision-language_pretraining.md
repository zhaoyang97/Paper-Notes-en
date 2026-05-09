---
title: >-
  [Paper Note] 2.5 Years in Class: A Multimodal Textbook for Vision-Language Pretraining
description: >-
  [ICCV 2025][Audio & Speech][vision-language pretraining] This work extracts keyframes and text (via ASR and OCR) from YouTube instructional videos to construct a high-quality interleaved image-text "multimodal textbook" dataset for VLM pretraining, achieving substantial improvements over web-crawled interleaved datasets on knowledge-intensive and reasoning benchmarks.
tags:
  - ICCV 2025
  - "Audio & Speech"
  - vision-language pretraining
  - interleaved image-text dataset
  - instructional video
  - multimodal textbook
  - in-context learning
date: 2026-05-08
content_hash: dc383c6dbecdcafc
---

# 2.5 Years in Class: A Multimodal Textbook for Vision-Language Pretraining

**Conference**: ICCV 2025
**arXiv**: [2501.00958](https://arxiv.org/abs/2501.00958)
**Code**: [https://github.com/DAMO-NLP-SG/multimodal_textbook](https://github.com/DAMO-NLP-SG/multimodal_textbook)
**Area**: Audio & Speech
**Keywords**: vision-language pretraining, interleaved image-text dataset, instructional video, multimodal textbook, in-context learning

## TL;DR

This work extracts keyframes and text (via ASR and OCR) from YouTube instructional videos to construct a high-quality interleaved image-text "multimodal textbook" dataset for VLM pretraining, achieving substantial improvements over web-crawled interleaved datasets on knowledge-intensive and reasoning benchmarks.

## Background & Motivation

Existing interleaved image-text datasets (e.g., MMC4, OBELICS) are collected via web crawling and suffer from three core issues: (1) **loose image-text alignment** — webpage images may be irrelevant to the surrounding context (e.g., advertisements, logos); (2) **lack of logical coherence across image sequences** — multiple images within a webpage bear no explicit reasoning relationship; (3) **low knowledge density** — news and entertainment content constitutes a large proportion, with minimal coverage of foundational subject knowledge.

Meanwhile, the internet hosts a vast amount of high-quality instructional videos (e.g., mathematics and geometry courses), in which instructors' step-by-step visual demonstrations are naturally paired with detailed verbal explanations, forming a structure that is tightly image-text aligned and logically coherent — akin to a textbook. Yet these resources remain underutilized in VLM training. Microsoft's Phi series has also demonstrated the critical role of textbook-quality data in LLM training.

## Method

### Overall Architecture

The proposed approach consists of two stages: **systematic collection of instructional videos** and **a multi-level extraction and filtering pipeline from video to textbook**. The final output is an interleaved image-text dataset comprising 6.5 million keyframes and 750 million text tokens, covering 6 foundational disciplines including mathematics, physics, and chemistry.

### Key Designs

1. **LLM-Driven Knowledge Taxonomy and Video Collection**:

    - GPT-4o is used to construct a four-level knowledge taxonomy: discipline → course → sub-course → knowledge point, covering 6 disciplines, 55 courses, and 3,915 knowledge points.
    - Each knowledge point serves as a search query on YouTube; the top-50 results are retrieved and deduplicated.
    - An LLM reviews video metadata (title, description, comments) to filter irrelevant or non-compliant content, yielding 159,565 videos.
    - Design motivation: A standardized taxonomy ensures broad coverage and prevents omission of important subject areas.

2. **Multi-Level Knowledge Extraction and Filtering Pipeline (Video → Textbook)**:

    - **Video level**: FFmpeg extracts audio → whisper-large-v3 transcribes ASR → Qwen2-72B rewrites and refines ASR fluency; an LLM filters low-quality videos along three dimensions — relevance, knowledge density, and transcription quality — retaining 75K videos.
    - **Segment level**: ASR timestamps segment long videos into 10–20 second clips; VideoLlama2 generates segment descriptions, and segments are filtered by text similarity with ASR to remove uninformative clips (e.g., scenes showing only the instructor's face).
    - **Keyframe level**: SSIM is used to detect significant changes between consecutive frames for keyframe extraction, removing redundancy; InternVL2-40B performs OCR on keyframes to extract text, formulas, and symbols, filtering low-information keyframes and duplicate OCR outputs.
    - Design motivation: The coarse-to-fine multi-level filtering progressively removes noise — video-level filtering removes irrelevant content, segment-level filtering removes visually uninformative scenes, and keyframe-level filtering removes redundant frames and low-quality OCR.

3. **Organization of the Interleaved Textbook Format**:

    - Keyframes, OCR text, and refined ASR text are interleaved chronologically.
    - Even when the visual content of a segment is filtered out, its ASR text is retained — preserving valuable narrated knowledge.
    - Final format: $\{\text{frame}_1^{k_1}, \text{frame}_1^{k_2}, \text{ocr}_1, \text{asr}_1, \text{asr}_2, \text{asr}_3, \text{frame}_4^{k_1}, \text{ocr}_4, \text{asr}_4, \ldots\}$

### Loss & Training

Continual pretraining is performed on LLaVA-1.5-7B (following 558K paired-data alignment), and Idefics2-8B is trained both from scratch and via continual pretraining. For fair comparison, equal-volume samples (610K) are drawn from MMC4 and OBELICS using identical training hyperparameters.

## Key Experimental Results

### Main Results

| Dataset / Benchmark | Setting | MMC4 | OBELICS | Textbook-6.5M | Gain |
|---------------------|---------|------|---------|---------------|------|
| ScienceQA-IMG | 0-shot | - | - | 26.3 | - |
| ScienceQA-IMG | 4-shot | 11.6 | 16.4 | **37.3** | +20.9 vs MMC4 |
| MathVista | 0-shot | 20.4 | 21.6 | **24.3** | +2.7 |
| MathVista | 1-shot | 30.0 | 28.5 | **43.4** | +14.9 vs OBELICS |
| OKVQA | 4-shot | 28.7 | 37.5 | **39.9** | +2.4 vs OBELICS |
| TextVQA | 4-shot | 20.9 | 32.2 | **33.5** | +1.3 vs OBELICS |
| Avg. over 7 benchmarks | 0–4-shot | 10.9–21.9 | 10.7–26.2 | **15.5–30.8** | +3.2~+8.3 |

After continual pretraining on Idefics2, MathVista improves from 27.6 to 29.7, and MathVision from 14.3 to 16.2.

### Ablation Study

| Configuration | 1-shot Avg. Accuracy | Note |
|---------------|----------------------|------|
| Full method (SSIM + ASR refinement + OCR) | 31.1 | Best |
| w/o ASR refinement | 26.2 (↓4.9) | Raw ASR is colloquial; PPL reaches 16.86, degrading language ability |
| w/o OCR | 28.8 (↓2.3) | OCR supplies additional knowledge via formulas and symbols |
| SSIM → pixel-level keyframe extraction | 22.1 (↓9.0) | Extracts too many frames (18M) with heavy redundancy |
| SSIM → CLIP semantic-level extraction | 24.6 (↓6.5) | Extracts too few frames (1.7M), missing critical keyframes |

### Key Findings

- A **"cheating test"** validates context awareness: placing the test sample itself within the few-shot context, the Textbook model achieves 94.1% on MathVista (vs. 72.6% for MMC4), demonstrating that VLMs pretrained on the Textbook dataset effectively attend to information in interleaved contexts.
- **Image order shuffling** experiments show that shuffling has almost no effect on MMC4, a moderate drop on OBELICS, and a substantial performance degradation on Textbook — confirming that image sequences derived from video sources exhibit strong logical dependencies, which are critical for learning complex knowledge and reasoning.
- After instruction fine-tuning (LLaVA-665K), Textbook yields an additional 5.5% gain on MathVista, more than double that of OBELICS (+2.4%).

## Highlights & Insights

- **A paradigm shift in data sourcing**: moving from web crawling to instructional video mining, leveraging the natural temporal consistency and lecture-demonstration alignment inherent in video.
- The in-sample image similarity metric shows that intra-sample image relevance in the Textbook dataset (0.686) is twice that of OBELICS (0.345), and remains stable as the number of images increases.
- Each sample contains on average 10.7 images and 1,297 tokens, far exceeding MMC4 (5.7 images / 417 tokens).

## Limitations & Future Work

- The dataset primarily covers foundational academic instructional videos, offering limited zero-shot gains on general-domain VQA tasks (few-shot settings are required to demonstrate advantages).
- ASR refinement relies on a large LLM (Qwen2-72B), incurring substantial computational cost for processing 75K videos.
- Only English instructional videos are included; multilingual extension remains unexplored.
- Training has not been combined with large-scale (billion-scale) web data; the optimal data mixture ratio warrants further investigation.

## Related Work & Insights

- The Phi series validates the importance of "textbook-quality" data; this work extends that insight from pure text to the multimodal setting.
- The key distinction from multi-source datasets such as OmniCorpus lies in the temporal coherence of video; future work could explore incorporating additional video types such as conference talks and laboratory recordings.
- The "cheating test" methodology can serve as a general evaluation tool for interleaved context awareness.

## Rating
- Novelty: ⭐⭐⭐⭐ — The idea of constructing interleaved textbook data from instructional videos is original and fills a gap in video-sourced pretraining data.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Multi-model, multi-benchmark, extensive ablations, and cleverly designed cheating and shuffling experiments.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with detailed pipeline descriptions.
- Value: ⭐⭐⭐⭐ — Open-sourced dataset and reusable pipeline offer practical value for knowledge-intensive VLM pretraining.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] How Would It Sound? Material-Controlled Multimodal Acoustic Profile Generation for Objects](how_would_it_sound_material-controlled_multimodal_acoustic_profile_generation_fo.md)
- [\[ICCV 2025\] Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations](zero-avsr_zero-shot_audio-visual_speech_recognition_with_llms_by_learning_langua.md)
- [\[ICCV 2025\] Latent Swap Joint Diffusion for 2D Long-Form Latent Generation](latent_swap_joint_diffusion_for_2d_long-form_latent_generation.md)
- [\[ICCV 2025\] Everything is a Video: Unifying Modalities through Next-Frame Prediction](everything_is_a_video_unifying_modalities_through_next-frame_prediction.md)
- [\[ICCV 2025\] MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing](mug_pseudo_labeling_augmented_audio-visual_mamba_network_for_audio-visual_video_.md)

<!-- RELATED:END -->
