---
title: >-
  [Paper Note] MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering
description: >-
  [ACL 2025][Multimodal VLM][Multilingual VQA] This work introduces MTVQA, the first multilingual text-centric visual question answering benchmark covering 9 languages. It resolves the "vision-text misalignment" issue of translation-based approaches through human expert annotation. Evaluations reveal a substantial performance gap between the best MLLM (InternVL-2.5, 32.2%) and the human baseline (79.7%), highlighting the severe challenges of multilingual text understanding.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Multilingual VQA"
  - "Text-Centric Visual Question Answering"
  - "MLLM Evaluation"
  - "Low-Resource Languages"
  - "Benchmark Dataset"
date: 2026-05-08
content_hash: efdf4f5f2f3eaad8
---

# MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering

**Conference**: ACL 2025  
**arXiv**: [2405.11985](https://arxiv.org/abs/2405.11985)  
**Code**: [Available](https://github.com/bytedance/MTVQA)  
**Area**: Multilingual Translation  
**Keywords**: Multilingual VQA, Text-Centric Visual Question Answering, MLLM Evaluation, Low-Resource Languages, Benchmark Dataset

## TL;DR

This work introduces MTVQA, the first multilingual text-centric visual question answering benchmark covering 9 languages. It resolves the "vision-text misalignment" issue of translation-based approaches through human expert annotation. Evaluations reveal a substantial performance gap between the best MLLM (InternVL-2.5, 32.2%) and the human baseline (79.7%), highlighting the severe challenges of multilingual text understanding.

## Background & Motivation

Text-Centric Visual Question Answering (TEC-VQA) serves as an important proxy task for evaluating AI's understanding capabilities in text-rich scenarios. However, two core limitations exist in current research:

**Language coverage bias**: Existing TEC-VQA benchmarks focus almost exclusively on high-resource languages like English and Chinese, while low-resource languages (e.g., Arabic, Thai, Vietnamese) are severely neglected.

**Fundamental flaws of translation-based approaches**: Previous multilingual VQA works (e.g., xGQA, MaXM) scale up QA pairs using translation engines. However, this incurs a severe "vision-text misalignment" issue in TEC-VQA scenarios, where translation only processes the QA texts but neglects the visual text embedded within the images.

For instance, with an image of a Russian menu, a translation engine might incorrectly translate dish names, leading to a mismatch between the QA pair and the actual text displayed in the image. While such misalignment might not be problematic in general VQA (where questions do not rely on visual text), it is fatal for TEC-VQA.

## Method

### Overall Architecture

The construction of MTVQA follows three phases: (1) multi-source text-rich image collection; (2) two-round human expert annotation (maker-checker paradigm); and (3) large-scale MLLM benchmarking.

### Key Designs

1. **Image Collection and Cleaning**:

    - Sources comprise three channels: public datasets (e.g., ICDAR MLT19, 30%), web crawling (Common Crawl, 20%), and on-site photography (50%).
    - On-site photography was conducted in countries/regions where each language is natively spoken, ensuring the authenticity and high quality of the images.
    - Multilingual OCR engines were used to filter images containing text, and algorithms filtered out sensitive/inappropriate content.
    - The collection covers over 20 fine-grained scenarios: menus, maps, bills, slides, academic papers, etc.
    - Resulting in a final set of 2,116 test images and 6,678 training images.

2. **Two-Round Human Expert Annotation**:

    - **Annotator Qualifications**: Native speakers of the target language for over 10 years with a bachelor's degree or higher.
    - **Round 1 (Generation)**: For each image, 3 annotators generate 5 QA pairs. The first 3 pairs require directly reading the text in the image to answer, while the remaining 2 require reasoning based on the visual text.
    - **Round 2 (Verification)**: Another group of 2 annotators independently reviews each QA pair for relevance, accuracy, conciseness, and ethical compliance.
    - A 10% random sampling check was implemented; failed batches were sent back for re-annotation.
    - Although this design was costly (approximately \$90,000 and taking over 5 months), it ensured high annotation quality.

3. **9-Language Coverage**:

    - Arabic (AR), Korean (KO), Japanese (JA), Thai (TH), Vietnamese (VI), Russian (RU), French (FR), German (DE), and Italian (IT).
    - Covers multiple writing systems (Arabic, Hangul, Japanese scripts, Thai, Latin, and Cyrillic).
    - Final dataset contains 28,607 QA pairs across 8,794 images.

### Evaluation Design

- Accuracy is adopted as the primary metric instead of ANLS, as ANLS cannot accurately reflect the correctness of textual content in images.
- A unified prompt format is used to constrain output length, ensuring concise and evaluable answers.
- Human baselines are evaluated using 10 native speakers per language.

## Key Experimental Results

### Main Results—MLLM Multilingual TEC-VQA Performance (Accuracy %)

| Model | AR | DE | FR | IT | JA | KO | RU | TH | VI | Avg |
|------|------|------|------|------|------|------|------|------|------|------|
| **Human** | 76.9 | 80.2 | 84.1 | 78.0 | 79.1 | 81.7 | 76.3 | 78.4 | 82.8 | **79.7** |
| InternVL2.5-78B | 15.9 | 39.0 | 45.6 | 42.9 | 21.1 | 33.9 | 12.2 | 23.8 | 41.5 | **32.2** |
| Qwen2-VL-72B | 20.7 | 36.5 | 44.1 | 42.8 | 21.6 | 37.4 | 15.6 | 17.7 | 41.6 | 30.9 |
| GPT-4o | 20.2 | 34.2 | 41.2 | 32.7 | 20.0 | 33.9 | 11.5 | 22.5 | 34.2 | 27.8 |
| Claude3 Opus | 15.1 | 33.4 | 40.6 | 34.4 | 19.4 | 27.2 | 13.0 | 19.5 | 29.1 | 25.7 |
| TextSquare | 3.7 | 27.0 | 30.8 | 26.7 | 3.2 | 7.2 | 6.7 | 5.2 | 12.4 | 13.6 |

### Key Contrastive Analysis

| Dimension | Finding |
|------|------|
| Human vs. Best MLLM | 79.7% vs. 32.2%, a 47.5% gap, indicating huge room for improvement |
| Latin vs. Non-Latin Scripts | DE/FR/IT are generally higher than AR/JA/TH/RU, due to training data bias |
| Text-Specific vs. General MLLMs | TextMonkey (9.9%) < MiniCPM (17.3%) < InternVL (32.2%); text-specific models lag behind because they focus only on English and Chinese |
| OCR+GPT-4 vs. GPT-4V | 21.6% vs. 22.0%, each with its own pros and cons |
| OCR+GPT-4V | 28.3%, the best combination |
| Instruction Tuning Gain | Xcomposer-4KHD: 11.2% → 19.7% (+8.5%) |

### Error Analysis Statistics

| Error Type | Ratio |
|---------|------|
| OCR Recognition Failure | 39% |
| Insufficient Reasoning | 34% |
| Language Bias | 15% |
| Hallucination | 12% |

### Key Findings

1. All models perform significantly worse on non-Latin script languages, especially Arabic and Russian.
2. Open-source models in the Qwen2-VL and InternVL series are already capable of outperforming GPT-4V/GPT-4o.
3. Text-specific MLLMs lag behind general models in multilingual scenarios due to their excessive focus on English and Chinese.
4. Few-shot improvements are limited and tend to saturate (zero-shot 22.0% → 5-shot 24.8%).
5. Asking questions in English versus the native language makes almost no difference, indicating the bottleneck lies in visual text perception rather than language understanding.

## Highlights & Insights

1. **Precise Problem Definition**: The paper clearly points out the fundamental flaw of translation-based approaches in TEC-VQA (vision-text misalignment) and addresses it with high-quality manual annotation.
2. **Broad Coverage**: Spanning 9 languages, 20+ scenario types, and dual coverage of both document and natural scenes, it represents the most comprehensive multilingual TEC-VQA benchmark to date.
3. **Revealing Gap**: The massive gap of 32.2% vs. 79.7% clearly demonstrates that MLLMs are still far from passing in multilingual text understanding.
4. **OCR Failures Account for 39%**: This suggests that improving visual text perception (rather than language understanding) is the true key.

## Limitations & Future Work

1. **Still Limited Language Coverage**: The 9 covered languages still leave out many low-resource languages (e.g., Hindi, Swahili).
2. **Single Answer Format**: Only short answers are requested, lacking reasoning questions that require longer explanations.
3. **Limitations of Evaluation Metric**: Accuracy requires exact matching, which might be overly strict for morphologically rich languages such as Arabic.
4. **High Cost Barrier**: The translation and annotation cost of approximately \$90,000 makes rapid expansion of language coverage difficult.
5. **Small Training Set Size**: The scale of 6,678 training images limits the effectiveness of instruction tuning.

## Related Work & Insights

- **General Multilingual VQA**: xGQA (7 languages) and MaXM (7 languages) employ translation schemes, which are only suitable for scenarios that do not rely on visual text in images.
- **TEC-VQA Benchmarks**: TextVQA, DocVQA, and OCRBench are predominantly in English.
- Core Insight from MTVQA: Constructing TEC-VQA benchmarks for low-resource languages requires native annotation; translation-based approaches do not work.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first natively annotated multilingual TEC-VQA benchmark.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Evaluates over 20 models (including both closed-source and open-source), complete with human baselines and detailed error analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Detailed description of the dataset construction process and deep experimental analysis.
- **Value**: ⭐⭐⭐⭐⭐ — Fills an important gap in multilingual TEC-VQA and reveals massive room for improvement in MLLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MAGIC-VQA: Multimodal and Grounded Inference with Commonsense Knowledge for Visual Question Answering](magic-vqa_multimodal_and_grounded_inference_with_commonsense_knowledge_for_visua.md)
- [\[NeurIPS 2025\] Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering](../../NeurIPS2025/multimodal_vlm/benchmarking_retrievalaugmented_multimodal_generation_for_do.md)
- [\[ACL 2025\] WikiMixQA: A Multimodal Benchmark for Question Answering over Tables and Charts](wikimixqa_a_multimodal_benchmark_for_question_answering_over_tables_and_charts.md)
- [\[NeurIPS 2025\] Are Vision Language Models Ready for Clinical Diagnosis? A 3D Medical Benchmark for Tumor-centric Visual Question Answering](../../NeurIPS2025/multimodal_vlm/are_vision_language_models_ready_for_clinical_diagnosis_a_3d_medical_benchmark_f.md)
- [\[NeurIPS 2025\] FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering](../../NeurIPS2025/multimodal_vlm/focus_internal_mllm_representations_for_efficient_fine-grained_visual_question_a.md)

</div>

<!-- RELATED:END -->
