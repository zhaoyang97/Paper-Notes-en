---
title: >-
  [Paper Note] IndoTabVQA: A Benchmark for Cross-Lingual Table Understanding in Bahasa Indonesia Documents
description: >-
  [ACL 2026][Multilingual & Machine Translation][Cross-lingual table understanding] This paper presents IndoTabVQA, a cross-lingual visual question answering benchmark for table understanding in Bahasa Indonesia documents.…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Cross-lingual table understanding"
  - "visual question answering"
  - "Bahasa Indonesia documents"
  - "spatial priors"
  - "low-resource languages"
date: 2026-05-08
content_hash: 256115242cd38f18
---

# IndoTabVQA: A Benchmark for Cross-Lingual Table Understanding in Bahasa Indonesia Documents

**Conference**: ACL 2026  
**arXiv**: [2604.11970](https://arxiv.org/abs/2604.11970)  
**Code**: [https://huggingface.co/datasets/NusaBharat/INDOTABVQA](https://huggingface.co/datasets/NusaBharat/INDOTABVQA)  
**Area**: Document Understanding / Cross-Lingual VQA  
**Keywords**: Cross-lingual table understanding, visual question answering, Bahasa Indonesia documents, spatial priors, low-resource languages

## TL;DR

This paper presents IndoTabVQA, a cross-lingual visual question answering benchmark for table understanding in Bahasa Indonesia documents. The dataset comprises 1,593 document images annotated with QA pairs in four languages (Indonesian, English, Hindi, and Arabic). The benchmark reveals substantial performance gaps in VLMs for low-resource languages and cross-lingual table understanding, with fine-tuning combined with spatial priors achieving up to 48.5% In-Match accuracy.

## Background & Motivation

**Background**: Vision-language models (VLMs) have demonstrated strong performance on text-rich visual understanding tasks, with benchmarks such as TextVQA and DocVQA driving substantial progress. Table-specific datasets such as TableVQA-Bench further evaluate structure-aware numerical reasoning capabilities.

**Limitations of Prior Work**: Existing benchmarks share a critical limitation—they are English-centric and monolingual, making them unable to reveal the true capabilities of VLMs on low-resource languages. Languages such as Indonesian, Hindi, and Arabic serve billions of users worldwide, yet VLMs may fail significantly on documents in these languages. Table VQA requires models to handle both linguistic variation and structural complexity simultaneously, a combinatorial challenge that remains insufficiently studied.

**Key Challenge**: Existing VQA benchmarks fail to assess two critical capabilities: (1) whether VLMs can understand tables in low-resource languages, and (2) whether VLMs can correctly answer questions when the document and question are in different languages. This gap limits our understanding of true multilingual capabilities.

**Goal**: To construct a cross-lingual table visual question answering benchmark that systematically evaluates VLMs on low-resource language document understanding and cross-lingual visual reasoning.

**Key Insight**: By using Indonesian-language documents as visual content (representing over 200 million speakers yet severely underrepresented in vision-language research) paired with QA annotations in four languages, the benchmark disentangles two challenges: visual-language understanding (monolingual setting) and cross-lingual alignment (cross-lingual setting).

**Core Idea**: The benchmark is constructed from real-world Indonesian document tables with four-language QA annotations. Spatial priors (table detection bounding box coordinates) are introduced as additional input, demonstrating that targeted fine-tuning and spatial information can substantially improve VLM performance on specialized document tasks.

## Method

### Overall Architecture

The IndoTabVQA evaluation pipeline comprises three settings: (1) zero-shot evaluation—direct inference on the test set using pretrained VLMs; (2) fine-tuned evaluation—models fine-tuned on 500 training images and evaluated on 1,043 test images; (3) fine-tuning with spatial priors—YOLOv9 is first applied to detect table regions and obtain bounding box coordinates, which are incorporated into the prompt before VLM inference. The input consists of a document image $I$ and a question $Q$ in one of four languages, and the output is a short textual or numerical answer $A$.

### Key Designs

1. **Diverse Dataset Construction**:

    - Function: Provides an evaluation resource covering diverse table visual styles and document domains.
    - Mechanism: 1,593 document images are collected from Indonesian government reports, educational records, business documents, and public health data. Images are categorized into three visual styles: bordered tables (500 images), borderless tables (602 images), and colored tables (491 images). QA annotations are authored manually in Indonesian and then extended to English, Hindi, and Arabic via automatic translation followed by native-speaker verification, with each QA pair undergoing dual quality checks for internal consistency and cross-lingual equivalence.
    - Design Motivation: Borderless tables require structural inference from whitespace and alignment, while colored tables introduce visual distractions. This diversity ensures the benchmark exposes distinct failure modes of VLMs.

2. **Spatial Prior-Augmented Input**:

    - Function: Helps VLMs focus on relevant regions by providing table location information.
    - Mechanism: A two-stage pipeline is employed. In Stage 1, YOLOv9 (pretrained on TableBank and PubLayNet) detects table regions in the document and outputs bounding box coordinates along with the number of tables. In Stage 2, the original input, bounding box coordinates, and table count are combined into an augmented prompt for the VLM. Knowledge of precise table locations allows the model to concentrate attention on relevant content.
    - Design Motivation: Practical document processing systems typically detect document regions before performing specialized analysis. Spatial priors simulate this real-world workflow and enable isolation of the specific contribution of spatial localization to performance.

3. **Dual-Metric Evaluation**:

    - Function: Simultaneously assesses exact matching and semantic comprehension.
    - Mechanism: (a) In-Match accuracy—a relaxed matching criterion where a prediction is considered correct if the normalized ground-truth answer appears as a substring of the prediction, accommodating VLMs that generate redundant context; (b) STS accuracy—a multilingual sentence embedding model computes cosine similarity between predictions and ground-truth answers to measure semantic alignment.
    - Design Motivation: VLMs frequently generate answers containing additional contextual information; In-Match avoids false negatives caused by strict matching. STS captures semantic equivalence across different surface forms.

### Loss & Training

Qwen2.5-VL 3B undergoes full-parameter instruction fine-tuning, while the 7B variant is fine-tuned using LoRA for parameter efficiency. Each language variant is trained independently to isolate language-specific learning patterns. The training set consists of 500 images, the validation set 50 images, and the test set 1,043 images.

## Key Experimental Results

### Main Results

Cross-lingual In-Match accuracy (%):

| Model | Indonesian | English | Hindi | Arabic | Average |
|-------|-----------|---------|-------|--------|---------|
| GPT-4o (zero-shot) | 72.2 | 44.6 | 26.0 | 21.4 | 41.1 |
| Qwen2.5-VL 7B | 54.8 | 36.2 | 17.3 | 23.0 | 32.9 |
| LLaMA-3.2 11B | 57.4 | 30.8 | 15.5 | 19.4 | 30.7 |
| IndoTabVQA 7B+SP | **78.3** | **58.4** | **29.4** | **32.8** | **48.5** |
| IndoTabVQA 3B+SP | 73.1 | 54.8 | 27.2 | 31.1 | 46.6 |
| GPT-4o+SP | 72.6 | 52.7 | 27.2 | 25.5 | 44.6 |

### Ablation Study

| Configuration | In-Match Avg. | STS Avg. | Notes |
|--------------|--------------|---------|-------|
| Qwen2.5-VL 3B zero-shot | 21.9% | 26.5% | Baseline |
| Fine-tuned 3B | 39.7% | 46.7% | +17.8% gain |
| Fine-tuned 3B + spatial priors | 46.6% | 53.1% | Additional +6.9% |
| Fine-tuned 7B | 44.5% | 54.9% | Larger model |
| Fine-tuned 7B + spatial priors | 48.5% | 58.3% | Best configuration |

### Key Findings
- Severe cross-lingual performance degradation: GPT-4o drops from 72.2% on Indonesian to 26.0% on Hindi and 21.4% on Arabic, a gap of 30–50 percentage points.
- Hindi proves the most challenging: nearly all models achieve the lowest accuracy on Hindi (4–29%), attributable to Devanagari script tokenization difficulties and scarce training data.
- Targeted fine-tuning on as few as 500 images yields substantial gains: +28.6 percentage points on Indonesian and +17.4 on English.
- Spatial priors consistently benefit all model scales: +3.5% for GPT-4o, +6.9% for 3B, and +4.0% for 7B.
- Fine-tuned 7B+SP achieves 48.5%, surpassing GPT-4o+SP at 44.6%, demonstrating that domain adaptation combined with spatial information outweighs raw model scale.
- Borderless tables are the most challenging (requiring structural inference), bordered tables are the simplest, and colored tables favor larger models (color aids visual grouping).

## Highlights & Insights
- **Quantification of cross-lingual gaps**: This work provides the first systematic quantification of cross-lingual transfer performance degradation in the table VQA setting. The 30–50 percentage point gap is striking and suggests that the multilingual capabilities of current VLMs are substantially overestimated.
- **Effectiveness of small-data fine-tuning**: Gains of 17–28 percentage points from only 500 training images demonstrate an exceptionally high marginal return on domain adaptation, which is encouraging for low-resource language research operating under constrained resources.
- **Simplicity and effectiveness of spatial priors**: Providing table bounding box coordinates from an off-the-shelf object detector as additional input is a zero-extra-training-cost strategy that consistently yields 4–7% improvements. This approach is generalizable to other document understanding tasks requiring spatial localization.

## Limitations & Future Work
- The dataset scale is relatively small (1,593 images), which may be insufficient to capture the full diversity of Indonesian-language documents.
- Each image contains only a single QA pair, limiting the assessment of complex multi-hop reasoning.
- Although translated QA pairs undergo human verification, guaranteeing complete semantic equivalence across languages remains difficult.
- Spatial priors depend on the accuracy of the external object detection model; detection failures propagate errors downstream.
- Future work could extend the benchmark to additional low-resource languages (e.g., Burmese, Khmer) and more complex document types.

## Related Work & Insights
- **vs. TableVQA-Bench**: Supports English only; IndoTabVQA extends coverage to four languages and a cross-lingual evaluation setting.
- **vs. DocVQA**: Focuses on general document understanding; IndoTabVQA specifically targets table structure reasoning, a more challenging sub-task.
- **vs. TabComp**: Addresses tabular comparison reasoning but remains English-centric; IndoTabVQA fills the gap for low-resource languages.

## Rating
- Novelty: ⭐⭐⭐⭐ First cross-lingual table VQA benchmark targeting Bahasa Indonesia, addressing the representational gap for low-resource languages.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers six models across three evaluation settings with table-type and language-level analysis.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with in-depth analysis and well-motivated research questions.
- Value: ⭐⭐⭐⭐ Provides an important evaluation resource for cross-lingual document AI research and highlights deficiencies in multilingual VLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Efficient Training for Cross-lingual Speech Language Models](efficient_training_for_cross-lingual_speech_language_models.md)
- [\[CVPR 2026\] SEA-Vision: A Multilingual Benchmark for Document and Scene Text Understanding in Southeast Asia](../../CVPR2026/multilingual_mt/sea-vision_a_multilingual_benchmark_for_comprehensive_document_and_scene_text_un.md)
- [\[AAAI 2026\] GloCTM: Cross-Lingual Topic Modeling via a Global Context Space](../../AAAI2026/multilingual_mt/gloctm_cross-lingual_topic_modeling_via_a_global_context_space.md)
- [\[ACL 2026\] MORPHOGEN: A Multilingual Benchmark for Evaluating Gender-Aware Morphological Generation](morphogen_a_multilingual_benchmark_for_evaluating_gender-aware_morphological_gen.md)
- [\[ACL 2026\] The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models](the_gaoyao_benchmark_a_comprehensive_framework_for_evaluating_multilingual_and_m.md)

</div>

<!-- RELATED:END -->
