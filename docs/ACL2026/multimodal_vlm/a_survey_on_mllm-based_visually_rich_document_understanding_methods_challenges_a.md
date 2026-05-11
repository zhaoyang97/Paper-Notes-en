---
title: >-
  [Paper Note] A Survey on MLLM-based Visually Rich Document Understanding: Methods, Challenges, and Emerging Trends
description: >-
  [ACL 2026][Multimodal VLM][visually rich document understanding] This paper presents a systematic survey of Multimodal Large Language Model (MLLM)-based Visually Rich Document Understanding (VRDU)…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "visually rich document understanding"
  - "multimodal large language models"
  - "OCR-free"
  - "document information extraction"
  - "retrieval-augmented generation"
date: 2026-05-08
content_hash: 4406f43d9bd4dead
---

# A Survey on MLLM-based Visually Rich Document Understanding: Methods, Challenges, and Emerging Trends

**Conference**: ACL 2026
**arXiv**: [2507.09861](https://arxiv.org/abs/2507.09861)
**Code**: None
**Area**: Document Understanding / Multimodal LLM
**Keywords**: visually rich document understanding, multimodal large language models, OCR-free, document information extraction, retrieval-augmented generation

## TL;DR

This paper presents a systematic survey of Multimodal Large Language Model (MLLM)-based Visually Rich Document Understanding (VRDU), organizing OCR-based and OCR-free methods along two dimensions—feature representation/fusion and training paradigms—while discussing emerging directions such as data scarcity, multi-page documents, multilingual support, RAG, and agent-based frameworks.

## Background & Motivation

**Background**: Visually Rich Document Understanding (VRDU) aims to automatically extract and comprehend information from documents containing complex visual, textual, and layout elements, with broad applications in finance, healthcare, and education. The rapid advancement of MLLMs is driving a paradigm shift from traditional to MLLM-based approaches in this field.

**Limitations of Prior Work**: (1) Early methods rely on OCR pipelines, leading to cascading error propagation; (2) the multimodal nature of documents (text, vision, layout) increases the complexity of feature fusion; (3) the scarcity of annotated data constrains supervised learning approaches; (4) multi-page and multilingual document processing remain open challenges.

**Key Challenge**: While MLLMs excel at general vision-language tasks, document understanding poses unique demands—precise comprehension of layout relationships, table structures, and printed/handwritten text—which general-purpose MLLMs cannot directly handle.

**Goal**: To provide a comprehensive survey of MLLM-based VRDU, covering method taxonomies, training strategies, challenges, and future directions, serving as a systematic roadmap for researchers.

**Key Insight**: The survey is organized along two core dimensions: (1) techniques for representing and fusing textual, visual, and layout features; and (2) pre-training, instruction tuning, and training strategies.

**Core Idea**: MLLM-based VRDU is evolving from OCR-dependent to OCR-free paradigms, while expanding from single-page static understanding to multi-page dynamic interaction via RAG and agent frameworks.

## Method

### Overall Architecture

The survey categorizes MLLM-based VRDU methods into two major classes: OCR-Dependent (requiring external OCR output as text input) and OCR-Free (end-to-end understanding directly from document images). Each class is further subdivided by feature fusion strategy and LLM backbone.

### Key Designs

1. **OCR-Dependent Methods**:

    - Function: Leverage text and layout information provided by external OCR engines, combined with LLMs for document understanding.
    - Mechanism: Representative methods include DocLLM (cross-attention for text-layout fusion), ICL-D3IE (in-context learning with GPT-3 for document IE), and LayoutLLM (integrating LayoutLMv3's layout encoding with Vicuna's generative capability).
    - Design Motivation: OCR supplies precise textual content, but introduces a trade-off between OCR error propagation and pipeline complexity.

2. **OCR-Free Methods**:

    - Function: Understand documents end-to-end directly from document images without external OCR.
    - Mechanism: Representative methods include the mPLUG-DocOwl series (perceiving document images via visual encoders), TextMonkey (sliding window processing for high-resolution documents), and InternVL-based methods (dynamic resolution handling).
    - Design Motivation: Eliminating OCR error propagation enables truly end-to-end processing, but requires visual encoders capable of recognizing fine-grained text within documents.

3. **Training Paradigm Taxonomy**:

    - Function: Systematize the training pipeline for MLLMs.
    - Mechanism: A three-stage paradigm—pre-training (PT) to learn foundational document representations, instruction tuning (IT) to align task instruction understanding, and downstream fine-tuning (FT) to adapt to specific tasks. Different methods adopt varying combinations of these stages.
    - Design Motivation: Understanding how training strategy choices affect final performance enables more informed design decisions for researchers.

### Loss & Training

Methods covered in the survey employ diverse training strategies: standard autoregressive language modeling loss, contrastive learning (e.g., CLIP-style), and text-layout alignment losses. Pre-training typically uses large-scale document-text pairs, while instruction tuning adopts structured QA formats.

## Key Experimental Results

### Main Results

| Method | Type | Tasks | Modalities | LLM Backbone | Multi-page |
|--------|------|--------|------------|--------------|------------|
| DocLLM | OCR-Dep | KIE, QA, DC | T, L | Custom | Single-page |
| LayoutLLM | OCR-Dep | KIE, QA | T, V, L | Vicuna-7B | Single-page |
| mPLUG-DocOwl | OCR-Free | QA | V | mPLUG-Owl | Single-page |
| TextMonkey | OCR-Free | QA | V | Qwen-VL | Single-page |
| InternVL-Doc | OCR-Free | QA, KIE | V | InternVL | Multi-page |
| DocThinker | OCR-Free | QA, KIE | T, V | Qwen2.5-VL | Single-page |

### Challenges & Trends

| Challenge | Current Status | Future Directions |
|-----------|---------------|-------------------|
| Data scarcity | Synthetic data + transfer learning | Self-supervised pre-training + few-shot learning |
| Multi-page documents | Supported by only a few methods | Dynamic page selection + retrieval augmentation |
| Multilingual support | English-dominant | Multilingual pre-training + cross-lingual transfer |
| RAG integration | Preliminary exploration | Document retrieval + generation pipeline |
| Agent frameworks | Emerging direction | Multi-tool collaborative document understanding agents |

### Key Findings
- OCR-Free methods are rapidly closing the gap with OCR-Dependent methods, particularly with the support of high-resolution visual encoders.
- Multi-page document understanding remains the most significant bottleneck, as most methods still support only single-page inputs.
- The introduction of RAG and agent frameworks opens a new pathway from document "understanding" to document "application."

## Highlights & Insights
- The survey's taxonomy dimensions are clearly designed, constructing a complete methodological space along three axes: OCR dependency × feature fusion × training paradigm.
- The model summary tables are highly informative, covering key attributes such as LLM backbone, visual encoder, training stages, multi-page support, and prompt format.
- The forward-looking discussion of emerging directions (RAG, agents) provides clear guidance for future research.

## Limitations & Future Work
- The survey covers methods up to mid-2025; the rapid pace of MLLM development may render portions of the content outdated.
- The absence of unified benchmark comparisons makes it difficult to directly assess the relative performance of different methods.
- Discussion of computational cost and efficiency is insufficiently in-depth.
- Future directions include: (1) a unified multi-task, multi-page document understanding framework; (2) trustworthy document understanding with hallucination control; and (3) integration of document understanding with knowledge graphs.

## Related Work & Insights
- **vs. Traditional document understanding surveys**: This survey focuses on MLLM-era methods, encompassing a broader range of OCR-free and generative approaches.
- **vs. General MLLM surveys**: This work delves into the specific requirements of document understanding, including layout awareness, table comprehension, and high-resolution processing.
- **vs. Document AI application surveys**: Greater emphasis is placed on technical method taxonomy rather than application scenarios.

## Rating
- Novelty: ⭐⭐⭐ Survey paper; contribution lies in systematization rather than originality.
- Experimental Thoroughness: ⭐⭐⭐ Comprehensive method coverage, but lacks unified quantitative comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-justified taxonomy dimensions, and information-dense tables.
- Value: ⭐⭐⭐⭐ Significant reference value for researchers in the VRDU field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PinPoint: Focus, Don't Prune — Identifying Instruction-Relevant Regions for Information-Rich Image Understanding](../../CVPR2026/multimodal_vlm/focus_dont_prune_identifying_instruction-relevant_regions_for_information-rich_i.md)
- [\[ICLR 2026\] Index-Preserving Lightweight Token Pruning for Efficient Document Understanding](../../ICLR2026/multimodal_vlm/index-preserving_lightweight_token_pruning_for_efficient_document_understanding_.md)
- [\[CVPR 2026\] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](../../CVPR2026/multimodal_vlm/docseeker_long_document_understanding.md)
- [\[AAAI 2026\] URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding](../../AAAI2026/multimodal_vlm/urag_unified_retrieval_and_generation_in_multimodal_llms_for.md)
- [\[ACL 2026\] Making MLLMs Blind: Adversarial Smuggling Attacks in MLLM Content Moderation](making_mllms_blind_adversarial_smuggling_attacks_in_mllm_content_moderation.md)

</div>

<!-- RELATED:END -->
