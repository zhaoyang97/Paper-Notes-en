---
title: >-
  [Paper Note] Relation-Rich Visual Document Generator for Visual Information Extraction
description: >-
  [CVPR 2025][Multimodal VLM][Document Understanding] This paper proposes RIDGE, a relation-rich visual document generator. It leverages LLMs to generate hierarchically structured text content combined with self-supervised content-driven layout generation. By synthesizing document images annotated with entity categories and linkage relations, RIDGE significantly enhances the performance of VIE models across multiple benchmarks.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Document Understanding"
  - "Synthetic Data Generation"
  - "Visual Information Extraction"
  - "Layout Generation"
  - "Hierarchical Structure Learning"
date: 2026-05-08
content_hash: 5e6ea16d29583cc3
---

# Relation-Rich Visual Document Generator for Visual Information Extraction

**Conference**: CVPR 2025  
**arXiv**: [2504.10659](https://arxiv.org/abs/2504.10659)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Document Understanding, Synthetic Data Generation, Visual Information Extraction, Layout Generation, Hierarchical Structure Learning

## TL;DR

This paper proposes RIDGE, a relation-rich visual document generator. It leverages LLMs to generate hierarchically structured text content combined with self-supervised content-driven layout generation. By synthesizing document images annotated with entity categories and linkage relations, RIDGE significantly enhances the performance of VIE models across multiple benchmarks.

## Background & Motivation

Visual Information Extraction (VIE) is a core task in document understanding, requiring the identification of entity categories (such as header/key/value) and entity linkage relations from document images. However, this task faces severe data scarcity issues:

1. **Extremely high annotation costs**: Open-set data require annotating $O(n^2)$ entity linkage relations, incurring prohibitive human labor costs. FUNSD contains only 199 document images, and XFUND contains only 199 images per language.
2. **Privacy constraints**: Form-like documents often contain personal information, making public acquisition highly challenging.
3. **Limitations of existing synthetic methods**: DocSynth relies on handcrafted layouts and suffers from low resolution. SynthDoG generates unnatural layouts by randomly placing grids based on rules. DocILE relies on 100 human-annotated templates, limiting diversity.
4. **Disconnection between content and layout in layout generation**: Existing layout generation methods focus solely on topological structures (bounding boxes + categories) while ignoring the correlation between text content and layout, failing to support VIE training.

## Method

### Overall Architecture

RIDGE employs a two-stage generation pipeline: **Stage 1** utilizes an LLM to generate document text content annotated with hierarchical structures (HST format); **Stage 2** adopts a Content-driven Layout Generation Model (CLGM) to place text entities into plausible bounding boxes, which are ultimately rendered into document images. In addition, a Hierarchical Structure Learning (HSL) training paradigm is introduced to enhance the performance of document understanding models.

### Key Designs

1. **Hierarchical Structured Text (HST) Generation**:
    - **Function**: Utilizes an LLM to generate document content with rich annotations of entity categories and linkage relations.
    - **Mechanism**: Designs a structured text format called HST, where the entire document is wrapped in a `<content>` tag, paragraphs are organized using hierarchical tags like `<h1>/<h2>`, key-value pairs are directly associated using colons `:`, and hierarchical nesting is represented by varying numbers of hyphens `-`. This format allows automatic parsing of entity texts, categories (header/key/value/other), and linkage relations.
    - **Design Motivation**: Conventional synthesis methods either do not generate text content or lack relation annotations in the content. HST enables the LLM to implicitly encode relationship metadata while generating content, providing complete entity annotations without human labor. Few-shot exemplars are used to guide the LLM to understand the format.

2. **Content-driven Layout Generation Model (CLGM)**:
    - **Function**: Automatically generates diverse and reasonable document layouts based on text content.
    - **Mechanism**: Serializes the document layout into a JSON format (which is more compact than HTML/CSS) and employs a masking strategy to replace bounding box coordinates with a special token `<FILL_i>`, training the LLM to predict these coordinates. Self-supervised training only requires OCR results (text + bboxes) without category or linkage labels. During training, entity sequences are randomly shuffled to force the model to infer layout relationships from text content; during inference, the original reading order of the HST is preserved. The layout generation formulation is $S(M) = f_\theta(S(D_{\backslash M}))$.
    - **Design Motivation**: Traditional layout generation is category-driven, but category annotations in form documents are scarce. CLGM uses content-driven generation combined with self-supervision, requiring only easily obtainable OCR results for training. Shuffling the training sequence forces the model to understand semantic text content to infer plausible placements, thereby automatically learning content-layout relationships.

3. **Hierarchical Structure Learning (HSL) Training Paradigm**:
    - **Function**: Promotes downstream models' capability of understanding hierarchical relations in documents using generated HST.
    - **Mechanism**: Outlines three training tasks—(a) HSP: parsing the entire document into HST format; (b) HSP with Localization: parsing local HST given specified regional bboxes; (c) VIE with HSP: outputting the HST of relevant sections first, then answering information extraction queries (similar to Chain-of-Thought).
    - **Design Motivation**: Existing methods (e.g., DocOwl-1.5 using newline characters to simulate layout, LayoutLLM using box+text format) only learn spatial locations without capturing hierarchical relationships. HSL enhances structural comprehension by encouraging the model to 'understand structure before answering,' while the CoT style improves interpretability.

### Loss & Training

- CLGM is trained with a standard autoregressive cross-entropy loss:
$$\mathcal{L} = -\sum_{k=1}^{K} \log P(S(M)^k | S(M)^{<k}, S(D_{\backslash M}), \theta)$$
- The backbone network is LLaMA-3.1-8B, fine-tuned using LoRA, with a maximum sequence length of 8000.
- Training data: OCR annotations of approximately 100K document images (originating from form/specification/resume/memo categories in RVL-CDIP + FUNSD/XFUND/HUST-CELL).
- Synthetic documents include around 3K English + 3K Chinese documents, generating 444K instruction samples.

## Key Experimental Results

### Main Results

| Dataset | Metric | Qwen2-VL-7B | +RIDGE | Gain |
|--------|------|-------------|--------|------|
| FUNSD (Open-set) | F1% | 59.89 | 66.48 | +6.59 |
| XFUND-ZH (Open-set) | F1% | 62.08 | 69.84 | +7.76 |
| CORD (Closed-set) | F1% | 82.71 | 84.47 | +1.76 |
| CORD– | ANLS% | 80.40 | 85.53 | +5.13 |
| EPHOIE | Acc% | 76.91 | 77.89 | +0.98 |
| POIE | ANLS% | 96.01 | 96.71 | +0.70 |

### Ablation Study

| Configuration | FUNSD F1% | XFUND-ZH F1% | CORD F1% | Description |
|------|-----------|-------------|----------|------|
| Qwen2-VL-7B Baseline | 59.89 | 62.08 | 82.71 | No extra training |
| +VIE Data | 64.87 | 67.32 | 83.77 | Only VIE data generated by RIDGE |
| +VIE + HSL | 66.48 | 69.84 | 84.47 | Adding hierarchical structure learning |

Ablation on Domain-Specific Generation:

| Configuration | SROIE– ANLS% | EPHOIE Acc% | EPHOIE ANLS% |
|------|-------------|------------|-------------|
| +RIDGE | 97.74 | 77.89 | 87.79 |
| +RIDGE+RIDGE-DS | 98.05 | 80.91 | 89.82 |

### Key Findings

- The improvements on open-set VIE are significantly larger than on closed-set tasks (FUNSD +6.59% vs SROIE +0.24%), as RIDGE mainly simulates open-set scenarios.
- RIDGE, relying solely on synthetic data, outperforms models specifically trained for VDU, such as DocOwl-1.5.
- In zero-shot scenarios for LayoutLMv3, pre-training with RIDGE achieves SER performances of 62.77% (FUNSD) and 69.75% (XFUND-ZH), demonstrating the reliability of generated semantic category labels.
- Domain-specific document generation (e.g., receipts/exam cover pages) yields an additional improvement of ~4% on closed-set tasks.

## Highlights & Insights

- **Elegant Two-Stage Decoupled Design**: Separating content and layout generation allows layout generation to rely solely on OCR results without human labels, substantially lowering the barrier for data acquisition.
- **Random Shuffling Strategy in Self-Supervised Training**: Randomly shuffling entity sequences during training forces the model to infer layouts purely from textual content, which is key to learning content-layout relationships automatically.
- **Explainability By-product**: The VIE with HSP training task instructs the model to output the hierarchical structure before answering, serving as both a training strategy and an interpretability mechanism.
- The HST format design is succinct and effective, elegantly utilizing existing HTML tags and indentation to represent hierarchy.

## Limitations & Future Work

- Currently, the model is primarily trained on form-like documents. Significantly different document types (such as academic papers, invoices, etc.) require additional training.
- The size of the synthetic data (6K documents) is relatively small; scaling up could potentially yield greater improvements.
- An end-to-end joint content-layout generation pipeline has not yet been explored.
- The gain on closed-set VIE is limited, and domain-specific generation requires additional prompt engineering.

## Related Work & Insights

- This work extends the line of research using LLMs for layout generation (e.g., LayoutNUWA) but innovatively introduces text-content dependency.
- Compared to synthesis methods like SynthDoG and DocILE, RIDGE is the first to automatically generate synthetic documents with relation annotations.
- The Chain-of-Thought style of VIE with HSP can be extended to other tasks requiring structured reasoning.

## Rating

- Novelty: ⭐⭐⭐⭐ Content-driven combined with self-supervised layout generation offers a novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 7 VIE benchmarks across open/closed sets, multiple downstream models, and thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The manuscript is well-structured and the motivations are clearly articulated.
- Value: ⭐⭐⭐⭐ Effectively addresses the critical pain point of data scarcity in the VIE domain; the synthetic data solution is highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MARTEN: Visual Question Answering with Mask Generation for Multi-Modal Document Understanding](marten_visual_question_answering_with_mask_generation_for_multi-modal_document_u.md)
- [\[ICCV 2025\] DOGR: Towards Versatile Visual Document Grounding and Referring](../../ICCV2025/multimodal_vlm/dogr_towards_versatile_visual_document_grounding_and_referring.md)
- [\[CVPR 2025\] DocoPilot: Improving Multimodal Models for Document-Level Understanding](docopilot_improving_multimodal_models_for_document-level_understanding.md)
- [\[AAAI 2026\] Exploring LLMs for Scientific Information Extraction using the SciEx Framework](../../AAAI2026/multimodal_vlm/exploring_llms_for_scientific_information_extraction_using_the_sciex_framework.md)
- [\[ICLR 2026\] LiveWeb-IE: A Benchmark For Online Web Information Extraction](../../ICLR2026/multimodal_vlm/liveweb-ie_a_benchmark_for_online_web_information_extraction.md)

</div>

<!-- RELATED:END -->
