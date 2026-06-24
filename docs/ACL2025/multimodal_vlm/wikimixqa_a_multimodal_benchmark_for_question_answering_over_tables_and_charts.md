---
title: >-
  [Paper Note] WikiMixQA: A Multimodal Benchmark for Question Answering over Tables and Charts
description: >-
  [ACL 2025][Multimodal VLM][Multimodal Question Answering] This paper proposes WikiMixQA, a benchmark consisting of 1,000 multiple-choice questions that require cross-modal reasoning over tables and charts. Evaluating 12 VLLMs reveals that while closed-source models achieve around 70% accuracy when provided with precise context, their performance drops drastically when retrieval from long documents is required. Open-source models reach a maximum accuracy of only 27%…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Multimodal Question Answering"
  - "Document Understanding"
  - "Tables"
  - "Charts"
  - "Vision-Language Models"
  - "Long Context"
  - "Cross-Modal Reasoning"
date: 2026-05-08
content_hash: 65e61bdec31f6eb2
---

# WikiMixQA: A Multimodal Benchmark for Question Answering over Tables and Charts

**Conference**: ACL 2025  
**arXiv**: [2506.15594](https://arxiv.org/abs/2506.15594)  
**Code**: [GitHub](https://github.com/negar-foroutan/WikiMixQA)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Question Answering, Document Understanding, Tables, Charts, Vision-Language Models, Long Context, Cross-Modal Reasoning

## TL;DR

This paper proposes WikiMixQA, a benchmark consisting of 1,000 multiple-choice questions that require cross-modal reasoning over tables and charts. Evaluating 12 VLLMs reveals that while closed-source models achieve around 70% accuracy when provided with precise context, their performance drops drastically when retrieval from long documents is required. Open-source models reach a maximum accuracy of only 27%, highlighting the severe shortcomings of existing vision-language models in long-context multimodal document understanding.

## Background & Motivation

Document understanding is a core task in NLP. Real-world documents contain not only plain text but also a large number of embedded structured or visual elements, such as tables and charts, making automatic comprehension challenging. While Vision-Large Language Models (VLLMs) have demonstrated capabilities across various tasks, they still exhibit significant deficiencies in the following aspects:

- **Single-page limitations**: Existing VQA benchmarks mostly focus on single-page documents, failing to test cross-page reasoning capabilities.
- **Lack of cross-modal reasoning**: Most dataset questions involve only a single modality (plain text or a single chart) and do not require models to synthesize information across multiple modalities.
- **Insufficient evaluation granularity**: Existing datasets lack precise control over the "modality types required to answer the question," making detailed analysis of model bottlenecks difficult.
- **Weak long-context processing**: Models perform decently when given precise information, but their performance drops significantly when they need to locate and extract scattered information from long documents.

WikiMixQA is designed to fill this gap: using Wikipedia as the document source, it constructs a multiple-choice question-answering benchmark that requires reasoning across table-table, chart-chart, and table-chart pairs.

## Method

### 1. Data Construction Pipeline (Four-stage Pipeline)

**(1) Document Collection**: Starting from the English Wikipedia dump of March 2022 (over 4 million articles), articles containing ≥3 tables were filtered (392,223 articles). Then, over a million images were downloaded, and a fine-tuned ViT model was used to filter out non-chart images, retaining articles with valid charts (15,164 articles). Using Wikipedia's "Instance of" attribute, documents were categorized into seven categories: Economy, Geography, History, Politics, Science, Sport, and Wikimedia, ultimately retaining 7,258 documents.

**(2) Modality Pair Selection**: To ensure question quality and diversity, modality pairs are selected based on semantic similarity rather than random pairing. For tables lacking captions, Llama-3-8B-Instruct is used to generate descriptions. For chart images, GPT-4-turbo is utilized to verify if they are charts and to extract key information. Then, BAAI/bge-reranker-v2-m3 cross-encoder is utilized to calculate similarity scores for all modality pairs, filtering semantically related table-table, chart-chart, and table-chart pairs.

**(3) Question Generation**: Pairs with similarities between the macro-mean and 0.9 are retained, while tables with less than 512 characters are excluded. For each pair, three types of multiple-choice questions are generated (based on the first modality, the second modality, and synthesizing both). Each question contains four options, the correct answer, and an explanation. A total of 3,528 questions are generated using GPT-4-turbo. Subsequently, InternVL2-Llama3-76B is employed for a two-stage quality filtering: first determining whether the modality information is sufficient to answer the question, and second verifying if the answer provided by GPT-4 is correct.

**(4) Human Annotation**: Out of the 3,528 questions, 2,001 questions (including 938 passed by AI filtering and 1,063 randomly sampled) are selected for annotation by three computer science master's students. The annotation follows a two-step process: first, verifying whether the question strictly requires integrating both modalities (validity check), and second, assessing the correctness of the answer (Correct / Wrong / Small Edit). Through majority voting, 595 questions are labeled Correct, and 405 questions are integrated after revision, culminating in a final benchmark of 1,000 questions.

### 2. Three Evaluation Settings

| Setting | Provided Context | Tested Ability |
|------|-------------|---------|
| **Blind** | No context provided | Model internal knowledge and reasoning |
| **Oracle** | Precise relevant tables/charts | Structured data reasoning and interpretation |
| **Wikidoc** | Wikipedia page snapshot | Long-context retrieval + reasoning |

### 3. Dataset Characteristics

- **Size**: 1,000 MCQs from 526 unique Wikipedia documents (~4,000 pages)
- **Average document length**: 24.18 pages, $1,815 \pm 2,825$ tokens
- **Seven main topics**: Economy, Geography, History, Politics, Science, Sport, and Wikimedia
- **Three modality combinations**: Around 50% are table-table, and the rest are chart-chart and table-chart
- **Quality assurance**: Pre-filtered by AI + two rounds of human annotation, ensuring that questions strictly require cross-modal reasoning to be answered

## Key Experimental Results

### Table 1: Model Accuracy Under Three Evaluation Settings (%)

| Model | Blind | Oracle | Wikidoc |
|------|-------|--------|---------|
| GPT-4o | 33.46 | **71.42** | **55.24** |
| Gemini-2.0-pro | 22.67 | 69.53 | 23.47 |
| Gemini-2.0-flash | 23.27 | 67.52 | 24.47 |
| Claude-3.5-Sonnet | 11.28 | 70.82 | 35.56 |
| InternVL2.5-78B | 3.29 | 27.67 | — |
| Qwen2.5-VL-72B | 0.39 | 23.17 | — |
| Llama-3.2-11B | 10.68 | 14.08 | — |
| Human Expert | — | 87.50 | — |

**Key Findings**: Under the Oracle setting, closed-source models achieve around 70% accuracy, but under the Wikidoc setting, only GPT-4o exceeds 50% (55.24%), while other closed-source models drop to nearly random levels. The top open-source model in the Oracle setting achieves only 27.67% (InternVL2.5-78B), lagging significantly behind human performance (87.50%).

### Table 2: Breakdown by Question Type (Modality Combination) Under the Oracle Setting

| Model | 2 Charts | 2 Tables | 1 Chart + 1 Table |
|------|----------|----------|-------------------|
| GPT-4o | 71.31 | 71.63 | 71.15 |
| Gemini-2.0-pro | 54.65 | **77.43** | 69.61 |
| Claude-3.5-Sonnet | 66.66 | 73.29 | 70.38 |
| InternVL2.5-78B | 24.41 | 30.02 | 26.53 |
| Qwen2.5-VL-72B | 22.48 | 24.22 | 21.92 |

**Key Findings**: GPT-4o performs evenly across the three combinations (~71%); Gemini-2.0-pro performs best on twin-table questions (77.43%) but scores only 54.65% on twin-chart questions, showing a significant discrepancy. Chart interpretation remains more challenging for all models than table understanding.

### Table 3: Breakdown by Topic Under the Oracle Setting (Closed-Source Models)

| Model | History | Politics | Geography | Sports | Science | Economy | Wikimedia |
|------|---------|----------|-----------|--------|---------|---------|-----------|
| GPT-4o | 74.12 | 68.61 | 76.32 | 75.96 | 72.94 | 58.25 | 72.48 |
| Claude-3.5-Sonnet | 67.74 | **76.68** | 69.47 | 71.15 | 68.24 | 61.17 | 74.50 |
| Gemini-2.0-pro | 75.81 | 71.75 | 72.11 | 69.23 | 69.41 | 52.43 | 72.48 |

**Key Findings**: The Economy topic is the most challenging for all models (GPT-4o scores only 58.25%), largely because economic questions frequently involve comparative analysis of bar charts and line graphs.

## Key Findings

1. **Severe Degradation in Long Context**: Moving from Oracle to Wikidoc, GPT-4o's accuracy drops from 71.42% to 55.24% (a 16 percentage point decrease), while other closed-source models degenerate even more severely (Gemini-2.0-pro drops from 69.53% to 23.47%). This indicates that locating relevant multimodal information in long documents is a critical bottleneck for current VLLMs.
2. **Open-Closed Source Gap**: Open-source models score at most 27.67% under the Oracle setting, which is close to a random guess level (25%), exhibiting a gap of approximately 43 percentage points compared to closed-source counterparts.
3. **Significant Human-Model Gap**: Human experts achieve 87.50% in the Oracle setting, leading the best model by 16 percentage points. This indicates that cross-modal reasoning remains highly difficult for current models.
4. **Charts are Harder than Tables**: Twin-chart questions are generally more difficult than twin-table questions, likely due to charts demanding dual capabilities in both visual perception and numerical reasoning.
5. **Validation of the Blind Setting**: All models perform near or below the default random level in the Blind setting, proving that WikiMixQA's questions indeed require contextual information and cannot be solved merely via guessing or pre-trained knowledge.

## Highlights & Insights

- **Systematic pipeline**: The four-stage construction process—from document collection, modality pairing, and AI generation to human annotation—is well-organized and strictly quality-controlled.
- **Cross-modal reasoning design**: Every question strictly requires synthesis across two modalities (table-table / chart-chart / table-chart) to be answered, filling a critical gap in existing benchmarks.
- **Three evaluation granularities**: The Blind, Oracle, and Wikidoc settings precisely isolate the models' execution capabilities across knowledge retrieval, reasoning, and search.
- **Broad model coverage**: Evaluates 12 VLLMs (4 open-source + 8 closed-source), presenting a comprehensive performance landscape.
- **Human comparison**: Evaluates a human expert baseline (87.50%), quantifying the substantial gap between models and human performance.

## Limitations & Future Work

1. **Single data source**: Only Wikipedia is used, limiting the style and content distribution of the documents and omitting professional documents like scientific papers or financial reports.
2. **Limited reasoning depth**: Questions primarily require single-hop or simple multi-hop cross-modal reasoning, lacking deeper, more complex multi-step reasoning chains.
3. **Limitations of long-context evaluation**: The Wikidoc setting uses page screenshots as inputs without incorporating text representations, potentially underestimating the model's capabilities in text-vision hybrid input formats.
4. **Incomplete open-source evaluation**: Due to resource/computational constraints, the Wikidoc setting is only evaluated on closed-source models.
5. **Broad chart category definitions**: Classifying maps, chord diagrams, etc., collectively as "charts" may introduce task conflation among different difficulty levels.
6. **Expandable topic coverage**: Among the seven categories, Economy contains relatively few documents (only 80 articles), which may undermine its representativeness.

## Related Work & Insights

| Benchmark | Document Source | No. Questions | Cross-page Reasoning | Avg. Tokens | Type of Evidence |
|------|---------|--------|---------|------------|---------|
| MMLongBench-Doc | Multi-source | 1k | ✓ | 21,214 | Table/Chart/Map |
| DUDE | Multi-source | 41k | ✓ | 1,832 | Table/Chart/Map |
| MP-DocVQA | Industrial Documents | 50k | ✗ | 2,027 | Table/Chart |
| InfographicsVQA | Infographics | 30k | ✗ | 288 | Table/Chart/Map |
| TAT-DQA | Financial Reports | 16k | ✗ | 577 | Table |
| **WikiMixQA** | **Wikipedia** | **1k** | **✓** | **1,815** | **Table/Chart/Map** |

The unique values of WikiMixQA include: (1) Each question strictly demands cross-modal reasoning; (2) The required modality types are controlled to enable fine-grained analysis; (3) It provides a three-tiered evaluation including Blind, Oracle, and Wikidoc; (4) It spans seven distinct topics, displaying superior domain diversity compared to expert-domain benchmarks.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to build a document understanding benchmark strictly requiring reasoning across tables and charts, featuring sound designs.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Evaluates 12 models in three settings, split by topic / modality types, providing a comprehensive analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear description of the construction pipeline, rich figures and tables, and detailed appendices.
- Value: ⭐⭐⭐⭐ — Exposes critical vulnerabilities of VLLMs in long-context cross-modal reasoning, offering practical guidance for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MAGIC-VQA: Multimodal and Grounded Inference with Commonsense Knowledge for Visual Question Answering](magic-vqa_multimodal_and_grounded_inference_with_commonsense_knowledge_for_visua.md)
- [\[ACL 2025\] MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering](mtvqa_benchmarking_multilingual_text-centric_visual_question_answering.md)
- [\[AAAI 2026\] Format Matters: The Robustness of Multimodal LLMs in Reviewing Evidence from Tables and Charts](../../AAAI2026/multimodal_vlm/format_matters_the_robustness_of_multimodal_llms_in_reviewing_evidence_from_tabl.md)
- [\[NeurIPS 2025\] Are Vision Language Models Ready for Clinical Diagnosis? A 3D Medical Benchmark for Tumor-centric Visual Question Answering](../../NeurIPS2025/multimodal_vlm/are_vision_language_models_ready_for_clinical_diagnosis_a_3d_medical_benchmark_f.md)
- [\[NeurIPS 2025\] WearVQA: A Visual Question Answering Benchmark for Wearables in Egocentric Authentic Real-world scenarios](../../NeurIPS2025/multimodal_vlm/wearvqa_a_visual_question_answering_benchmark_for_wearables_in_egocentric_authen.md)

</div>

<!-- RELATED:END -->
