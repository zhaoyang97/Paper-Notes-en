---
title: >-
  [Paper Note] 2.5 Years in Class: A Multimodal Textbook for Vision-Language Pretraining
description: >-
  [ICCV 2025][Audio & Speech][multimodal textbook] This work collects 2.5 years (22,000 hours) of instructional videos from YouTube and constructs a high-quality interleaved image-text "multimodal textbook" corpus (6.5M ke…
tags:
  - "ICCV 2025"
  - "Audio & Speech"
  - "multimodal textbook"
  - "interleaved image-text corpus"
  - "instructional videos"
  - "VLM pretraining"
  - "in-context learning"
date: 2026-05-08
content_hash: 008f0ce09cacc1b8
---

# 2.5 Years in Class: A Multimodal Textbook for Vision-Language Pretraining

**Conference**: ICCV 2025
**arXiv**: [2501.00958](https://arxiv.org/abs/2501.00958)
**Code**: [GitHub](https://github.com/DAMO-NLP-SG/multimodal_textbook) (open-sourced)
**Authors**: Wenqi Zhang, Hang Zhang, Xin Li, Jiashuo Sun, Yongliang Shen, Weiming Lu, Deli Zhao, Yueting Zhuang, Lidong Bing (Zhejiang University & Alibaba DAMO Academy)
**Area**: Audio & Speech
**Keywords**: multimodal textbook, interleaved image-text corpus, instructional videos, VLM pretraining, in-context learning

## TL;DR

This work collects 2.5 years (22,000 hours) of instructional videos from YouTube and constructs a high-quality interleaved image-text "multimodal textbook" corpus (6.5M keyframes + 0.75B text tokens) via an LLM-driven multi-level extraction and filtering pipeline. The resulting dataset significantly improves VLM pretraining on knowledge-intensive and reasoning tasks, yielding substantial gains on ScienceQA and MathVista in particular.

## Background & Motivation

**Value of interleaved image-text corpora**: Compared to image-text pair data, interleaved image-text corpora enable VLMs to understand the world in a manner closer to human cognition, while also unlocking advanced capabilities such as in-context learning and multi-image comparison.

**Three key limitations of existing datasets**:
1. **Loose image-text alignment**: Web-crawled interleaved datasets (e.g., MMC4, OBELICS) exhibit weak associations between images and text, and contain irrelevant images such as logos and advertisements.
2. **Lack of logical coherence in image sequences**: Most webpages contain few images with ambiguous inter-image logical relationships, making it difficult to learn complex visual reasoning.
3. **Low knowledge density**: Crawled webpages inevitably include low-knowledge-density content such as news, entertainment, and advertisements.

**Underutilized instructional video resources**: The internet hosts a wealth of instructional videos (e.g., geometry lessons on YouTube) that people commonly use to acquire foundational knowledge and professional skills, yet these valuable resources have been largely overlooked for VLM training. Instructional videos naturally feature frame-by-frame demonstrations accompanied by detailed verbal explanations from instructors, making them ideal training data sources.

**Importance of textbook-quality data**: Microsoft's Phi series has demonstrated that high-quality textbook-grade data is critical for LLM training.

## Core Problem

How to systematically extract high-quality, textbook-grade interleaved image-text datasets from large-scale instructional videos on the internet, in order to enhance knowledge acquisition and reasoning capabilities during VLM pretraining?

## Method

### Overall Architecture

A **Video-to-Textbook** pipeline is constructed, consisting of two major stages:
1. **Instructional video collection**: LLM-generated knowledge taxonomy → taxonomy-guided video retrieval → metadata filtering.
2. **Video-to-textbook conversion**: Coarse-to-fine knowledge extraction and filtering at multiple levels (video-level → segment-level → keyframe-level).

### Key Designs

#### 1. LLM-Driven Knowledge Taxonomy
- Four-level hierarchy: **Subject → Course → Sub-course → Knowledge Point**
- Constructed using GPT-4o, covering multiple educational stages from elementary to secondary school
- Final output: **6 subjects** (Mathematics, Physics, Chemistry, Earth Science, Engineering, Computer Science) → **55 courses** → **3,915 knowledge points**
- Each knowledge point serves as a query keyword to retrieve relevant videos via the YouTube Search API, retaining Top-50 results per knowledge point

#### 2. Video Collection and Metadata Filtering
- Retrieved videos are first deduplicated by video ID
- **LLM-based metadata review**: An LLM examines the title, description, and comments of each video to exclude irrelevant, pornographic, or illegal content
- Final collection: **159,565 videos**

#### 3. Multi-Level Video-to-Textbook Pipeline

**Video-level — ASR Extraction and Rewriting**:
- Audio is extracted using FFmpeg; speech-to-text transcription is performed with Whisper-large-v3 (ASR)
- Raw ASR text is colloquial and exhibits high perplexity (PPL=16.8 vs. 11.2 for standard corpora); **Qwen2-72B-Instruct** is used to rewrite the transcripts, reducing PPL to 13.9 and improving fluency and coherence

**Video-level — Low-Quality Video Filtering**:
- Rule-based filtering: non-English, shorter than 10 seconds, or near-empty ASR transcripts
- LLMs (DeepSeek-V2 + Llama3-70B-Instruct) evaluate ASR along three dimensions:
    - **Relevance**: whether the ASR aligns with the target knowledge point
    - **Knowledge density**: whether the transcript contains excessive filler expressions such as "um" or "then we get this"
    - **Transcription quality**: whether the transcript contains repetitions or errors
- A video is discarded only when both LLMs deem it unqualified → **75,000 high-quality videos** retained

**Segment-level — Long Video Segmentation**:
- ASR timestamps are used to segment long videos into short clips of 10–20 seconds
- Fragmented ASR segments are first merged into semantically coherent paragraphs before timestamp-based splitting

**Segment-level — Filtering of Visually Uninformative Segments**:
- VideoLlama2 generates detailed descriptions for each segment
- Text similarity between segment descriptions and ASR transcripts is computed (gte-Qwen2-7B-instruct); segments with uninformative scenes (transitions, instructor close-ups, etc.) are filtered out
- ASR text from discarded segments is retained, as it may still carry useful information

**Keyframe-level — SSIM-Based Keyframe Extraction**:
- SSIM is used to compare consecutive frames; starting from the first frame, only frames exhibiting significant change are retained as keyframes → **6.5M keyframes** extracted
- Comparison: pixel-level method (OpenCV absdiff) extracts too many frames (18M, −9% performance); semantic-level method (CLIP-ViT-L) extracts too few (1.7M, −6.5% performance); SSIM achieves the optimal balance for instructional video scenarios

**Keyframe-level — OCR Extraction**:
- InternVL2-40B performs OCR on each keyframe to extract on-screen text, mathematical symbols, and formulas
- InternVL2 simultaneously scores keyframe quality and filters low-information frames
- Highly similar OCR results across consecutive keyframes are deduplicated

#### 4. Interleaved Format Organization
- Keyframes, OCR text, and ASR text are interleaved in chronological order: `{frame1_k1, frame1_k2, ocr1, asr1, asr2, asr3, frame4_k1, ocr4, asr4, ...}`
- Multiple segments are concatenated up to the VLM's maximum context length, with an `End of Video` token appended at the end of each video
- Final corpus: **610K interleaved samples**, averaging 10.7 keyframes + 1,297 text tokens per sample

### Loss & Training

- Standard VLM pretraining loss; no loss is computed on image tokens
- LLaVA-1.5-7B continued pretraining: training resumes from the pretrained checkpoint (post 558K alignment data)
- Idefics2-8B under two settings: training from scratch / continued pretraining on the OBELICS-pretrained base
- Fair comparison: 610K samples are drawn from MMC4 and OBELICS respectively; identical training hyperparameters are used across all datasets
- Evaluation: RICES-based few-shot prompting (retrieving the $k$ most similar training samples as few-shot demonstrations)

## Key Experimental Results

### LLaVA-1.5-7B Continued Pretraining (Few-shot, Average over 7 Benchmarks)

| Setting | Metric | Textbook-6.5M | OBELICS | MMC4 | Gain (vs. OBELICS) |
|---------|--------|--------------|---------|------|--------------------|
| 0-shot | 7B Avg | 26.3 | — | — | +3.2% |
| 1-shot | ScienceQA | 29.4 | 2.8 | 1.6 | Large margin |
| 1-shot | MathVista | 43.4 | 28.5 | 30.0 | +14.9 |
| 4-shot | 7B Avg | — | — | — | +4.6% |

### Idefics2-8B Results

| Setting | Metric | Textbook | OBELICS | MMC4-cf |
|---------|--------|----------|---------|---------|
| Continued | MathVista | 29.7 | 27.6 | 27.8 |
| Continued | MathVision | 16.2 | 14.3 | 14.0 |
| From Scratch | MathVista | 26.1 | 24.2 | 24.0 |

### Zero-Shot Results After SFT (LLaVA-665K Fine-tuning)

| Dataset | Metric | Textbook+SFT | Original LLaVA-1.5 | OBELICS+SFT |
|---------|--------|--------------|--------------------|-------------|
| MathVista | Acc | 28.7 | 23.2 | 25.6 |
| General VQA | Avg | 62.2 | 61.1 | 61.8 |

### Cheat Test — Verifying Context Awareness

| Dataset | Metric | Textbook | OBELICS | MMC4-cf |
|---------|--------|----------|---------|---------|
| MathVista | 1-shot cheat | **94.1** | 67.7 | 72.6 |
| MathVision | 1-shot cheat | **98.4** | 66.5 | 69.3 |
| OKVQA | 2-shot cheat | **84.3** | 71.3 | 53.5 |

### Ablation Study

1. **ASR rewriting is critical**: Skipping ASR rewriting leads to a 4.9% average drop across 7 benchmarks; original ASR PPL = 16.86, rewritten PPL = 13.92.
2. **OCR provides additional gains**: Removing OCR causes a 2.3% average drop, particularly affecting TextVQA and MathVista.
3. **Keyframe extraction algorithm comparison**:
    - Pixel-level (OpenCV absdiff): 18M frames, −9% performance (excessive redundant frames)
    - Semantic-level (CLIP-ViT-L): 1.7M frames, −6.5% performance (missing key frames)
    - **SSIM**: 6.5M frames, best performance (optimal balance between quantity and quality)
4. **Image order shuffling experiment**: Shuffling the image order in Textbook significantly degrades performance, while MMC4 is largely unaffected — demonstrating that the logical coherence of image sequences in Textbook is genuinely learned and utilized by the model.

## Highlights & Insights

1. **Novel data source**: This is the first work to systematically transform YouTube instructional videos into interleaved image-text corpora for VLM pretraining, circumventing all three major drawbacks of conventional web-crawled data.
2. **Complete and well-motivated pipeline**: From LLM-generated knowledge taxonomy to multi-level filtering, each step is supported by clear design rationale and ablation validation.
3. **Elegant Cheat Test design**: By placing test samples directly into the few-shot context, the experiment examines whether the model genuinely attends to the interleaved context, revealing that VLMs trained on conventional datasets largely ignore their context.
4. **InSI-SIM metric**: An in-sample image similarity metric is designed to quantitatively assess the degree of image association within interleaved datasets; Textbook achieves 0.686, far exceeding OBELICS at 0.345.
5. **Knowledge transfer to SFT**: Knowledge acquired during pretraining transfers to the downstream SFT stage, yielding gains on MathVista twice as large as those from OBELICS.

## Limitations & Future Work

1. **Residual noise**: Despite multi-level filtering, the corpus may still contain redundant keyframes and low-quality text.
2. **Understanding tasks only**: Image token loss is not computed during training, precluding use for image generation tasks (extendable to omni-modal models).
3. **Limited subject coverage**: Only 6 STEM disciplines are included; humanities, social sciences, and medicine are not covered.
4. **English-centric**: Only English instructional videos are collected, resulting in insufficient multilingual coverage.
5. **Heavy reliance on external tools**: The pipeline depends on multiple large models including GPT-4o, Qwen2-72B, and InternVL2-40B, making reproduction costly.

## Related Work & Insights

| Dimension | Ours (Textbook) | MMC4 / OBELICS | OmniCorpus |
|-----------|----------------|----------------|------------|
| Data Source | Instructional videos | Web (Common Crawl) | Multi-source |
| Avg. Images/Sample | **10.7** | 5.7 / 2.5 | 3.9 |
| Avg. Text Tokens/Sample | **1,297** | 417 / 816 | 574 |
| InSI-SIM | **0.686** | 0.319 / 0.345 | 0.321 |
| Knowledge Density | High (STEM instruction) | Low (mixed news, etc.) | Medium |
| Image Sequence Coherence | Strong (temporal video frames) | Weak (random web images) | Weak |

This work aligns with the philosophy of Phi's "Textbooks Are All You Need," extending it to the **multimodal** domain: not only is the text of textbook quality, but the images — derived from video frames — also provide high-quality, temporally coherent visual knowledge.

The following additional insights are noteworthy:
- **Video as a pretraining data source**: The temporal structure of instructional videos naturally provides high-quality interleaved image-text data; this paradigm is extendable to other video types (science communication, documentaries, etc.).
- **Data quality over data quantity**: 610K high-quality samples significantly outperform million-scale low-quality web data, reaffirming the "Textbooks Are All You Need" philosophy.
- **Cultivating context awareness**: The issues revealed by the Cheat Test deserve attention — existing VLMs may not truly leverage few-shot context, and high-quality interleaved data can mitigate this problem.
- **Extendable to other domains**: The same pipeline can be applied to medical instructional videos, legal lectures, programming tutorials, and other specialized domains.

## Rating

- **Novelty**: ★★★★☆ (4/5) — Both the data source and pipeline design are innovative, though the core idea of "using instructional videos for pretraining" is relatively intuitive.
- **Technical Depth**: ★★★★☆ (4/5) — The pipeline design is meticulous and ablations are thorough, though no model architecture innovation is involved.
- **Experimental Thoroughness**: ★★★★★ (5/5) — Multiple VLMs, diverse settings, Cheat Test, image shuffling experiments, and ablation studies — highly comprehensive.
- **Writing Quality**: ★★★★☆ (4/5) — Clear structure with rich figures and tables.
- **Impact**: ★★★★☆ (4/5) — Open-sourced dataset and pipeline provide practical guidance for VLM pretraining data construction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] BabyVLM-V2: Toward Developmentally Grounded Pretraining and Benchmarking of Vision Foundation Models](../../CVPR2026/audio_speech/babyvlm-v2_toward_developmentally_grounded_pretraining_and_benchmarking_of_visio.md)
- [\[AAAI 2026\] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering](../../AAAI2026/audio_speech/end-to-end_contrastive_language-speech_pretraining_model_for_long-form_spoken_qu.md)
- [\[ICCV 2025\] How Would It Sound? Material-Controlled Multimodal Acoustic Profile Generation for Objects](how_would_it_sound_material-controlled_multimodal_acoustic_profile_generation_fo.md)
- [\[ICCV 2025\] Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations](zero-avsr_zero-shot_audio-visual_speech_recognition_with_llms_by_learning_langua.md)
- [\[NeurIPS 2025\] LUMIA: A Handheld Vision-to-Music System for Real-Time, Embodied Composition](../../NeurIPS2025/audio_speech/lumia_a_handheld_vision-to-music_system_for_real-time_embodied_composition.md)

</div>

<!-- RELATED:END -->
