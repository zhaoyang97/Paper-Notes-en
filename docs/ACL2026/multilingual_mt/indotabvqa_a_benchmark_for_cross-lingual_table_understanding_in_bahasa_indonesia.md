---
title: >-
  [Paper Note] IndoTabVQA: A Benchmark for Cross-Lingual Table Understanding in Bahasa Indonesia Documents
description: >-
  [ACL 2026][Multilingual & Machine Translation][Cross-lingual table understanding] Proposes IndoTabVQA, a cross-lingual visual question answering benchmark for Indonesian (Bahasa Indonesia) document tables. It contains 1…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Cross-lingual table understanding"
  - "Visual Question Answering"
  - "Indonesian documents"
  - "Spatial priors"
  - "Low-resource languages"
date: 2026-05-08
content_hash: a4c0ef44be4461e9
---

# IndoTabVQA: A Benchmark for Cross-Lingual Table Understanding in Bahasa Indonesia Documents

**Conference**: ACL 2026  
**arXiv**: [2604.11970](https://arxiv.org/abs/2604.11970)  
**Code**: [https://huggingface.co/datasets/NusaBharat/INDOTABVQA](https://huggingface.co/datasets/NusaBharat/INDOTABVQA)  
**Area**: Document Understanding / Cross-lingual VQA  
**Keywords**: Cross-lingual table understanding, Visual Question Answering, Indonesian documents, Spatial priors, Low-resource languages

## TL;DR

Proposes IndoTabVQA, a cross-lingual visual question answering benchmark for Indonesian (Bahasa Indonesia) document tables. It contains 1,593 document images and QA annotations in four languages (Indonesian/English/Hindi/Arabic), revealing significant performance gaps of VLMs in low-resource languages and cross-lingual table understanding. Fine-tuning combined with spatial priors achieves up to 48.5% In-Match accuracy.

## Background & Motivation

**Background**: Visual Language Models (VLMs) demonstrate excellent performance in text-dense visual understanding tasks, with benchmarks like TextVQA and DocVQA driving progress. Specialized datasets like TableVQA-Bench further evaluate structure-aware numerical reasoning capabilities.

**Limitations of Prior Work**: Existing benchmarks share a critical limitation—they are English-centric and monolingual, failing to reveal the true capabilities of VLMs in low-resource languages. Languages such as Indonesian, Hindi, and Arabic cover billions of users worldwide, but VLMs may fail significantly on documents in these languages. For table VQA, models must simultaneously handle linguistic variation and structural complexity, a combined challenge that remains understudied.

**Key Challenge**: Existing VQA benchmarks cannot test two critical capabilities: (1) whether VLMs can understand tables in low-resource languages? (2) whether VLMs can answer correctly when the document and question are in different languages? This gap limits our understanding of real-world multilingual capabilities.

**Goal**: Construct a cross-lingual table visual question answering benchmark to systematically evaluate VLM capabilities in low-resource language document understanding and cross-lingual visual reasoning.

**Key Insight**: Utilize Indonesian documents as visual content (representing over 200 million speakers but severely underrepresented in vision-language research), paired with QA annotations in four languages, to isolate two challenges: vision-language understanding (monolingual setting) and cross-lingual alignment (cross-lingual setting).

**Core Idea**: Build a benchmark using real-world Indonesian document tables with four-language QA annotations. Introduce spatial priors (table detection coordinates) as additional inputs to demonstrate that targeted fine-tuning and spatial information can significantly improve VLM performance on specialized document tasks.

## Method

### Overall Architecture

The evaluation pipeline of IndoTabVQA includes three settings: (1) Zero-shot evaluation—direct inference on the test set using pre-trained VLMs; (2) Fine-tuning evaluation—evaluation on 1,043 test images after fine-tuning on 500 training images; (3) Fine-tuning + Spatial Priors—using YOLOv9 to detect table regions to obtain bounding box coordinates, incorporating this spatial information into the prompt before VLM processing. The input is a document image $I$ + a question $Q$ (in one of four languages), and the output is a short text or numerical answer $A$.

### Key Designs

1.  **Diverse Dataset Construction**:
    *   Function: Provide evaluation resources covering various table visual styles and document domains.
    *   Mechanism: 1,593 document images were collected from Indonesian government reports, educational records, business documents, and public health data, categorized into three visual styles: bordered tables (500), borderless tables (602), and colored tables (491). QA annotations were manually written in Indonesian and expanded to English, Hindi, and Arabic via automatic translation + native speaker verification. Each QA pair underwent double quality checks for internal consistency and cross-lingual equivalence.
    *   Design Motivation: Borderless tables require inferring structure from whitespace and alignment, while colored tables introduce visual distractions. This diversity ensures the benchmark exposes different failure modes of VLMs.

2.  **Spatial Priors Augmented Input**:
    *   Function: Help VLMs focus on relevant regions by providing table location information.
    *   Mechanism: A two-stage process—Stage 1 uses YOLOv9 (pre-trained on TableBank+PubLayNet) to detect table regions in documents, outputting bounding box coordinates and table counts; Stage 2 feeds the original input + bounding box coordinates + table count as an augmented prompt into the VLM. Knowing precise table locations allows the model to concentrate attention on relevant content.
    *   Design Motivation: Practical document processing systems typically detect document regions before specialized processing. Spatial priors simulate this workflow and isolate the specific impact of spatial localization on performance.

3.  **Dual-Metric Evaluation**:
    *   Function: Simultaneously evaluate exact matching and semantic understanding capabilities.
    *   Mechanism: (a) In-Match Accuracy—relaxed matching, considered correct if the normalized ground truth appears as a substring in the prediction, addressing redundant context generated by VLMs; (b) STS Accuracy—uses a multilingual sentence embedding model to calculate the cosine similarity between predicted and ground truth answers to measure semantic alignment.
    *   Design Motivation: VLMs often generate answers containing extra context; In-Match avoids false negatives from strict matching, while STS captures semantic equivalence across different expressions.

### Loss & Training

Full instruction fine-tuning was performed on Qwen2.5-VL 3B, while parameter-efficient fine-tuning via LoRA was used for the 7B version. Each language variant was trained independently to isolate language-specific patterns. The training set contains 500 images, the validation set 50, and the test set 1,043.

## Key Experimental Results

### Main Results

Cross-lingual In-Match Accuracy (%):

| Model | Indonesian | English | Hindi | Arabic | Average |
|------|--------|------|--------|---------|------|
| GPT-4o (Zero-shot) | 72.2 | 44.6 | 26.0 | 21.4 | 41.1 |
| Qwen2.5-VL 7B | 54.8 | 36.2 | 17.3 | 23.0 | 32.9 |
| LLaMA-3.2 11B | 57.4 | 30.8 | 15.5 | 19.4 | 30.7 |
| IndoTabVQA 7B+SP | **78.3** | **58.4** | **29.4** | **32.8** | **48.5** |
| IndoTabVQA 3B+SP | 73.1 | 54.8 | 27.2 | 31.1 | 46.6 |
| GPT-4o+SP | 72.6 | 52.7 | 27.2 | 25.5 | 44.6 |

### Ablation Study

| Configuration | In-Match Avg | STS Avg | Description |
|------|-------------|---------|------|
| Qwen2.5-VL 3B Zero-shot | 21.9% | 26.5% | Baseline |
| Fine-tuned 3B | 39.7% | 46.7% | +17.8% Gain |
| Fine-tuned 3B + Spatial Priors | 46.6% | 53.1% | Further +6.9% |
| Fine-tuned 7B | 44.5% | 54.9% | Larger model |
| Fine-tuned 7B + Spatial Priors | 48.5% | 58.3% | Optimal config |

### Key Findings
- **Severe decline in cross-lingual performance**: GPT-4o falls from 72.2% in Indonesian to 26.0% in Hindi and 21.4% in Arabic, a gap of 30-50 percentage points.
- **Hindi is the most difficult**: Lowest accuracy across almost all models (4-29%), due to Devanagari script tokenization difficulties and training data scarcity.
- **Targeted fine-tuning with 500 images yields significant gains**: +28.6 points for Indonesian, +17.4 for English.
- **Spatial priors are effective across all model scales**: GPT-4o +3.5%, 3B +6.9%, 7B +4.0%.
- **Fine-tuned 7B+SP outperforms GPT-4o+SP**: At 48.5% vs 44.6%, it indicates that domain adaptation + spatial information is more important than pure model scale.
- **Borderless tables are the hardest** (requiring structural inference), bordered tables are the easiest, and colored tables benefit larger models (color provides visual grouping cues).

## Highlights & Insights
- **Quantification of the cross-lingual gap**: Systematically quantifies performance loss in cross-lingual transfer for table VQA for the first time. The 30-50 point gap is a warning that current VLM multilingual capabilities are severely overestimated.
- **Effectiveness of small-data fine-tuning**: Only 500 training images yield 17-28 point improvements, proving high marginal benefits for domain adaptation. This is encouraging for low-resource language research.
- **Simplicity and effectiveness of spatial priors**: Using off-the-shelf object detection models to provide table coordinates is a zero-extra-training-cost strategy that consistently brings 4-7% gains. This approach can generalize to other document understanding tasks requiring spatial localization.

## Limitations & Future Work
- Small dataset size (1,593 images) may not fully cover the diversity of Indonesian documents.
- Each image has only one QA pair, limiting evaluation of complex multi-hop reasoning.
- While manually verified, absolute cross-lingual semantic equivalence in translated QA is difficult to guarantee.
- Spatial priors rely on the accuracy of external object detection models; detection failures propagate errors.
- Future work could extend to more low-resource languages (e.g., Burmese, Khmer) and more complex document types.

## Related Work & Insights
- **vs TableVQA-Bench**: Only supports English; IndoTabVQA extends to four languages and cross-lingual settings.
- **vs DocVQA**: Focuses on general document understanding; IndoTabVQA focuses on the more challenging subtask of table structure reasoning.
- **vs TabComp**: Focuses on comparative table reasoning but remains English-centric; IndoTabVQA fills the gap for low-resource languages.

## Rating
- Novelty: ⭐⭐⭐⭐ First cross-lingual table VQA benchmark for Indonesian, addressing low-resource language representation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation over six models, three settings, table types, and language analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, deep analysis, and well-designed research questions.
- Value: ⭐⭐⭐⭐ Provides vital evaluation resources for cross-lingual document AI and highlights deficiencies in multilingual VLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Efficient Training for Cross-lingual Speech Language Models](efficient_training_for_cross-lingual_speech_language_models.md)
- [\[CVPR 2026\] SEA-Vision: A Multilingual Benchmark for Document and Scene Text Understanding in Southeast Asia](../../CVPR2026/multilingual_mt/sea-vision_a_multilingual_benchmark_for_comprehensive_document_and_scene_text_un.md)
- [\[ACL 2026\] LLM-XTM: Enhancing Cross-Lingual Topic Models with Large Language Models](llm-xtm_enhancing_cross-lingual_topic_models_with_large_language_models.md)
- [\[ACL 2026\] XQ-MEval: A Dataset with Cross-lingual Parallel Quality for Benchmarking Translation Metrics](xq-meval_a_dataset_with_cross-lingual_parallel_quality_for_benchmarking_translat.md)
- [\[ACL 2026\] Why Low-Resource NLP Needs More Than Cross-Lingual Transfer: Lessons Learned from Luxembourgish](why_low-resource_nlp_needs_more_than_cross-lingual_transfer_lessons_learned_from.md)

</div>

<!-- RELATED:END -->
