---
title: >-
  [Paper Note] Harnessing PDF Data for Improving Japanese Large Multimodal Models
description: >-
  [ACL 2025][Multimodal VLM][Japanese LLM] Proposes a fully automated PDF data extraction pipeline to extract image-text pairs from Japanese PDFs and generate instruction data. By continually fine-tuning the LLaVA1.5 framework, it significantly improves the performance of Japanese multimodal models, achieving a 2.1%–13.8% gain on Heron-Bench.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Japanese LLM"
  - "PDF data"
  - "multimodal training"
  - "data pipeline"
  - "continual fine-tuning"
date: 2026-05-08
content_hash: 6a6e20d57a8f8c6d
---

# Harnessing PDF Data for Improving Japanese Large Multimodal Models

**Conference**: ACL 2025  
**arXiv**: [2502.14778](https://arxiv.org/abs/2502.14778)  
**Code**: [https://github.com/ku21fan/PDF-JLMM](https://github.com/ku21fan/PDF-JLMM)  
**Area**: Multimodal / VLM  
**Keywords**: Japanese LLM, PDF data, multimodal training, data pipeline, continual fine-tuning

## TL;DR

Proposes a fully automated PDF data extraction pipeline to extract image-text pairs from Japanese PDFs and generate instruction data. By continually fine-tuning the LLaVA1.5 framework, it significantly improves the performance of Japanese multimodal models, achieving a 2.1%–13.8% gain on Heron-Bench.

## Background & Motivation

Large Multimodal Models (LMMs) perform exceptionally well in English, but are constrained by the scarcity of high-quality training data in non-English languages such as Japanese. The key challenges currently faced by Japanese LMMs include:

**Single Data Source**: Most open-source Japanese LMMs rely on translated English datasets (e.g., the Japanese-translated version of LLaVA). Consequently, models primarily learn Western cultural content and lack Japan-specific cultural knowledge (such as cherry blossoms, Japanese architecture, etc.).

**Unutilized PDF Data**: Unlike web-crawled image-text data, PDFs contain a wealth of high-value but untapped information from books, reports, brochures, etc. To the authors' knowledge, no prior research has leveraged PDF data to enhance Japanese LMMs.

**High Manual Annotation Cost**: Manual image-text annotation of large-scale PDFs is impractical.

Core Problem: **Can PDF data effectively enhance Japanese LMMs? How can useful training data be automatically extracted from PDFs?**

## Method

### Overall Architecture

A three-stage training pipeline: Stage 1 Pre-training (558K Japanese image-text pairs) $\rightarrow$ Stage 2 Instruction Tuning (620K Japanese instruction data) $\rightarrow$ Stage 3 Continual Fine-Tuning CFT (362K PDF-derived instruction data). The core innovation lies in the Stage 3 data construction pipeline.

### Key Designs

1. **PDF Collection and Filtering**: Over 51.38 million PDFs were obtained from the National Diet Library Web Archiving Project. Filtering strategy: select only PDFs with 5 pages or fewer (longer ones resemble books and contain fewer images); extract only the first page (images typically appear on the first page); use PyMuPDF to detect PDFs containing images. Ultimately, 200K PDF pages were selected. **Design Motivation**: Empirical findings from manually observing hundreds of PDFs, enabling efficient filtering of image-free PDFs using simple rules.

2. **Layout Analysis & OCR Extraction**: Instead of reading data directly from PDFs (as PyMuPDF might extract invisible images or falsely split images), PDFs are first converted into JPEG images, and then layout analysis and OCR are completed using the Surya tool. Surya is based on a pre-trained deep learning model supporting over 90 languages. However, its performance is not perfect—it occasionally misidentifies Japanese characters as Hindi, and filters out "images" with width or height under 50 pixels.

3. **Image-Text Pairing**: Japanese-Cloob (a widely used CLIP-like model in Japan with 300K monthly active users) is utilized to compute the cosine similarity between image embeddings and OCR text embeddings, selecting the most similar text as the pair. **Key Findings**: Due to imperfect OCR quality (line breaks, misidentification of complex Kanji), training models directly with the extracted image-text pairs yields poor results.

4. **PDF-Style Text Generation**: To explore "what if the image-text extraction were more accurate," GPT-4o-mini is used to generate "PDF-style text" for each image—instead of directly describing the image, it simulates the indirect explanatory text surrounding the image in a PDF. Experiments show this method is much more effective than the raw extracted image-text pairs.

5. **Instruction Data Generation**: Images are fed directly into GPT-4o-mini to generate Japanese instruction data (dialogue format), with the paired text as optional context. **Key Findings**: When the quality of the paired text is low, generating instruction data solely based on the image yields better results. Hence, in the final scheme, all 362K instruction data points are generated based solely on the images.

6. **NSFW and PII Filtering**: GPT-4o-mini is employed to detect and filter out unsafe content and personally identifiable information (PII).

### Loss & Training

- Based on the LLaVA1.5 framework, the visual encoder is replaced from CLIP to SigLIP.
- Parameter-efficient fine-tuning is conducted using LoRA.
- The main model PDF-JLMM uses Swallow (a Japanese fine-tuned version of Llama3-8B) as the base LLM.
- Training time (4×A100): Stage 1 takes ~11h, Stage 2 takes ~42h, Stage 3 takes ~19h.

## Key Experimental Results

### Main Results

| Model | JA-LLaVA-Bench(COCO) | JA-LLaVA-Bench(Wild) | Heron-Bench |
|------|---------------------|---------------------|-------------|
| GPT-4V | 90.1 | 94.1 | 79.7 |
| Qwen-VL 7B | 80.4 | 54.0 | 49.7 |
| Heron BLIP v1 7B | 89.5 | 45.1 | 45.4 |
| EvoVLM-JP-v1 7B | 69.2 | 56.4 | 45.1 |
| **PDF-JLMM 8B** | **88.2** | **65.8** | **65.8** |
| LLaVA1.5-Llama3 8B | 86.9 | 56.9 | 61.6 |
| LLaVA1.5-Phi3-medium 14B | 86.8 | 74.1 | 57.4 |

### Impact of PDF Data Volume (Heron-Bench)

| LLM | Stages 1&2 | +50K PDF | +100K PDF | +150K PDF | +200K PDF |
|-----|-----------|---------|----------|----------|----------|
| Swallow 8B | 54.7 | 65.7 | **65.8** | 63.8 | 64.6 |
| Llama3 8B | 54.8 | 58.7 | 61.0 | **61.8** | 61.6 |
| Phi3-mini 3.8B | 43.3 | 51.9 | 53.0 | **57.1** | 54.3 |
| Phi3-medium 14B | 54.2 | **58.8** | 56.3 | 57.4 | 58.1 |

### Raw Image-Text Pairs vs. Instruction Data

| Training Data | L-COCO | L-Wild | Heron |
|---------|--------|--------|-------|
| Stages 1&2 (Baseline) | 84.0 | 59.8 | 54.7 |
| Top 1 Image-Text Pairs | 77.0 | 37.4 | 40.0 |
| PDF-style text | 81.5 | 56.5 | 65.5 |
| **Instruction Data (Image Only)** | **87.3** | **61.6** | **65.7** |

### PDF Data vs. Translated English Data

| Stage 2 Data Source | L-COCO | L-Wild | Heron |
|----------|--------|--------|-------|
| LLaVA-v1.5-Instruct-620K-JA | 84.0 | 59.8 | 54.7 |
| **Instruct-from-200K PDF (362K)** | **88.1** | **72.7** | **70.0** |

### Key Findings

- **PDF-derived data is effective across all model sizes (3.8B–14B) and all LLM backbones (Japanese/Non-Japanese)**.
- **Achieves a maximum gain of 13.8% on Heron-Bench** (Phi3-mini), with a minimum gain of 2.1% (Phi3-medium).
- **Direct usage of raw image-text pairs actually leads to performance degradation** (Heron dropped from 54.7 to 40.0), indicating that OCR noise is detrimental to training.
- **Instruction data generated using images alone outperforms using images together with paired text**, as noisy text degrades instruction quality.
- **PDF data outperforms translated English data**: Even with a smaller sample size (362K vs. 620K), it scores 15.3% higher on Heron-Bench, demonstrating the value of culturally grounded content.
- **Translating other English datasets (such as Vision-Flan and Image-Textualization) even leads to a decrease in Japanese performance**.
- The size of the Japanese vocabulary has a significant impact: Phi3 has only 837 Japanese vocabulary tokens, causing the 14B model to underperform compared to the 8B Llama3.
- Diminishing returns with non-linear growth in data volume: Performance begins to saturate beyond 100K–150K PDFs.

## Highlights & Insights

- **First to demonstrate the value of PDFs as a multimodal training resource**: PDFs contain large volumes of culturally specific content that cannot be replaced by translated datasets.
- **"Generating instructions using only images" is superior to "using images and noisy text"**: This is counter-intuitive yet crucial—low-quality paired text is worse than none at all.
- **Importance of cultural knowledge**: Post-training, the model successfully identifies "cherry blossoms" rather than just stating "white flowers," which translated data fails to provide.
- **Automation pipeline is imperfect yet effective**: Even with errors in OCR and matching, as long as the subsequent instruction generation strategy is sound, high-quality training data can still be generated.
- **Ingenious PDF-style text experiment**: By using GPT to generate idealized paired text as an "upper-bound experiment," it proves that if extraction technology improves further, the image-text pair approach holds great potential.

## Limitations & Future Work

- Reliance on GPT-4o-mini for generating instruction data and PDF-style text introduces API costs and potential quality biases.
- Experiments are only conducted on the LLaVA1.5 framework; generalizability to more advanced architectures (such as Qwen-VL2, InternVL) remains unverified.
- Surya's OCR and layout analysis perform limitedly on general PDFs, restricting the possibility of direct image-text pair application.
- Tested solely on Japanese; although generalizability to other languages is claimed, it is not verified.
- Evaluation is limited to the Heron-Bench series; performance on newer benchmarks such as JDocQA and JMMMU is not evaluated.

## Related Work & Insights

- Unlike existing PDF data extraction efforts (which focus heavily on extracting table/figure-caption pairs from academic papers), this work extends the scope to general PDFs and explores the pairing of non-caption text.
- LLaVA's instruction data generation strategy is successfully adapted across languages here.
- While VILA-jp enhances Japanese LMMs using interleaved data, this work provides a complementary data source by utilizing PDF data.
- Insights for low-resource language LMM training: Rather than translating English data, it is more beneficial to tap into indigenous content sources of the native language.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to utilize PDF data for enhancing Japanese LMMs; the fully automated pipeline design is highly creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Extremely thorough ablation studies: comparisons are conducted across model scale, LLM selection, data volume, image-text pairs vs. instructions, and translated vs. native data.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured; each experiment is designed around a clear research question.
- **Value**: ⭐⭐⭐⭐ — Provides a replicable methodology for training LMMs in low-resource languages; the idea of utilizing PDF data sources offers widespread inspiration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Improving Medical Large Vision-Language Models with Abnormal-Aware Feedback](improving_medical_large_vision-language_models_with_abnormal-aware_feedback.md)
- [\[ACL 2025\] Error-driven Data-efficient Large Multimodal Model Tuning](error-driven_data-efficient_large_multimodal_model_tuning.md)
- [\[ICCV 2025\] CompCap: Improving Multimodal Large Language Models with Composite Captions](../../ICCV2025/multimodal_vlm/compcap_improving_multimodal_large_language_models_with_composite_captions.md)
- [\[CVPR 2025\] Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment](../../CVPR2025/multimodal_vlm/task_preference_optimization_improving_multimodal_large_language_models_with_vis.md)
- [\[ICCV 2025\] Effective Training Data Synthesis for Improving MLLM Chart Understanding](../../ICCV2025/multimodal_vlm/effective_training_data_synthesis_for_improving_mllm_chart_understanding.md)

</div>

<!-- RELATED:END -->
