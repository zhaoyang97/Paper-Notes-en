---
title: >-
  [Paper Note] A Survey on MLLM-based Visually Rich Document Understanding: Methods, Challenges, and Emerging Trends
description: >-
  [ACL 2026][Multimodal VLM][OCR-free] A systematic survey of Visually Rich Document Understanding (VRDU) based on Multimodal Large Language Models (MLLMs), categorizing OCR-based and OCR-free methods across feature representation/fusion and training paradigms, while discussing emerging trends such as data scarcity, multi-page documents, multilingual suppor
tags:
  - ACL 2026
  - Multimodal VLM
  - OCR-free
date: 2026-05-08
content_hash: d79e611adb14865d
---
# A Survey on MLLM-based Visually Rich Document Understanding: Methods, Challenges, and Emerging Trends

**Conference**: ACL 2026 Findings  
**arXiv**: [2507.09861](https://arxiv.org/abs/2507.09861)  
**Code**: None  
**Area**: Document Understanding / Multimodal LLM  
**Keywords**: Visually Rich Document Understanding, Multimodal Large Language Model, OCR-free, Document Information Extraction, Retrieval-Augmented Generation

## TL;DR

A systematic survey of Visually Rich Document Understanding (VRDU) based on Multimodal Large Language Models (MLLMs), categorizing OCR-based and OCR-free methods across feature representation/fusion and training paradigms, while discussing emerging trends such as data scarcity, multi-page documents, multilingual support, RAG, and agents.

## Background & Motivation

**Background**: Visually Rich Document Understanding (VRDU) aims to automatically extract and understand information from documents containing complex visual, textual, and layout elements, with broad applications in finance, healthcare, and education. With the rapid development of MLLMs, the field is undergoing a paradigm shift from traditional methods to MLLM-based approaches.

**Limitations of Prior Work**: (1) Early methods relied on OCR pipelines, where errors propagated cumulatively; (2) The multimodality of documents (text, vision, layout) increases the complexity of feature fusion; (3) Scarcity of labeled data constrains supervised learning methods; (4) Processing multi-page and multilingual documents remains a challenge.

**Key Challenge**: While MLLMs perform excellently on general vision-language tasks, document understanding has unique characteristics—requiring precision in layout relationships, tabular structures, and printed/handwritten text—making general MLLMs difficult to apply directly.

**Goal**: To provide a comprehensive survey of MLLM-based VRDU, covering method classification, training strategies, challenges, and future directions, serving as a systematic roadmap for researchers.

**Key Insight**: Organizing the survey from two core dimensions: (1) Representation and fusion techniques for text, vision, and layout features; (2) Pre-training, instruction tuning, and training strategies.

**Core Idea**: MLLM-based VRDU is evolving from OCR-dependent to OCR-free approaches, while expanding from single-page static understanding to multi-page dynamic interaction (RAG, agents).

## Method

### Overall Architecture

The survey categorizes MLLM-based VRDU methods into two major types: OCR-Dependent (requiring external OCR output as text input) and OCR-Free (end-to-end understanding directly from document images). Each category is further subdivided by feature fusion methods and LLM backbones.

### Key Designs

**1. OCR-Dependent Methods: Feeding External OCR Text/Layout to LLMs, High Recognition Accuracy but with Cascading Errors**

This category follows the logic of traditional document IE—first using an OCR engine to extract text and coordinates from document images, then passing them to the LLM for understanding. Representative works have different focuses: DocLLM uses cross-attention to fuse text and layout features, ICL-D3IE leverages the in-context learning of GPT-3 for document information extraction, and LayoutLLM connects LayoutLMv3's layout encoding with Vicuna's generative capabilities. Their common advantage is that OCR provides precise text content, relieving the model of the burden of recognizing text from pixels; the cost is that if the OCR fails, errors propagate through the pipeline, and there is an external dependency—the exact pain point OCR-Free routes aim to bypass.

**2. OCR-Free Methods: Visual Encoders Directly Reading Images, End-to-End but Requiring Fine-Grained Recognition**

To discard the OCR pipeline and eliminate error propagation, OCR-Free methods allow visual encoders to perceive document images directly and output understanding results end-to-end. The difficulty lies in the fact that document text is often small and dense, so the main evolutionary line in this category is "how to feed high-resolution documents into visual encoders": the mPLUG-DocOwl series models document images directly, TextMonkey uses sliding windows to process high-resolution page blocks, and InternVL-based methods rely on dynamic resolution to adapt to different document sizes. This achieves true end-to-end processing but places the entire burden of character recognition on the visual encoder, demanding higher fine-grained recognition capabilities.

**3. Training Paradigms: The Three Stages (Pre-training → Instruction Tuning → Downstream Fine-tuning) and Their Combinations**

The survey breaks down the MLLM training process into three stages: Pre-training (PT) for learning basic document representations, Instruction Tuning (IT) for aligning understanding of task instructions, and Downstream Fine-tuning (FT) for specific task adaptation. Different methods use various combinations of these stages. Explicitly segmenting the training process allows readers to see "where the performance difference of a method actually comes from," enabling grounded trade-offs when designing new methods rather than treating training as a black box.

### Loss & Training

The methods covered in the survey utilize multiple training strategies: standard autoregressive language modeling loss, contrastive learning (e.g., CLIP-style), and text-layout alignment loss. Pre-training typically uses large-scale document-text pairs, and instruction tuning uses structured QA formats.

## Key Experimental Results

### Main Results

| Method | Type | Task | Modality | LLM Backbone | Multi-page |
|------|------|------|------|---------|------|
| DocLLM | OCR-Dep | KIE, QA, DC | T, L | Custom | Single-page |
| LayoutLLM | OCR-Dep | KIE, QA | T, V, L | Vicuna-7B | Single-page |
| mPLUG-DocOwl | OCR-Free | QA | V | mPLUG-Owl | Single-page |
| TextMonkey | OCR-Free | QA | V | Qwen-VL | Single-page |
| InternVL-Doc | OCR-Free | QA, KIE | V | InternVL | Multi-page |
| DocThinker | OCR-Free | QA, KIE | T, V | Qwen2.5-VL | Single-page |

### Challenges & Trends

| Challenge | Current State | Future Work |
|------|---------|---------|
| Data Scarcity | Synthetic data + Transfer learning | Self-supervised pre-training + Few-shot learning |
| Multi-page Documents | Supported by a few methods | Dynamic page selection + Retrieval augmentation |
| Multilingualism | Primarily English | Multilingual pre-training + Cross-lingual transfer |
| RAG Integration | Preliminary exploration | Document retrieval + Generation pipeline |
| Agent Framework | Emerging direction | Multimodal agent for document understanding with tool orchestration |

### Key Findings
- OCR-Free methods are rapidly catching up to OCR-Dependent methods, especially with the support of high-resolution visual encoders.
- Multi-page document understanding is currently the biggest bottleneck, with most methods still only supporting single pages.
- The introduction of RAG and agent frameworks provides new paths from "understanding" to "application" in document understanding.

## Highlights & Insights
- The classification dimensions of the survey are clearly designed: a complete method space is constructed from three dimensions: OCR dependency × feature fusion × training paradigm.
- The model summary tables are highly practical, covering key information such as LLM backbones, visual encoders, training stages, multi-page support, and prompt formats.
- Forward-looking discussions on emerging directions (RAG, agents) provide clear paths for future research.

## Limitations & Future Work
- The survey covers methods up to mid-2025; the rapid development of MLLMs may soon render some content outdated.
- Lack of quantitative comparison on a unified benchmark makes it difficult to directly compare the performance of different methods.
- The discussion on computational cost and efficiency is not deep enough.
- Future directions: (1) Unified multi-task multi-page document understanding frameworks; (2) Trustworthy document understanding (hallucination control); (3) Integration of document understanding with knowledge graphs.

## Related Work & Insights
- **vs. Traditional document understanding surveys**: Focuses on new methods in the MLLM era, covering more OCR-free and generative approaches.
- **vs. General MLLM surveys**: Deeply explores the special requirements of document understanding (layout awareness, table understanding, high resolution).
- **vs. Document AI application surveys**: Focuses more on technical method classification rather than application scenarios.

## Rating
- Novelty: ⭐⭐⭐ Survey paper, emphasizes systematicity rather than originality.
- Experimental Thoroughness: ⭐⭐⭐ Comprehensive method coverage, but lacks unified quantitative comparison.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, reasonable classification dimensions, high information density in tables.
- Value: ⭐⭐⭐⭐ Provides significant reference value for researchers in the VRDU field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Relation-Rich Visual Document Generator for Visual Information Extraction](../../CVPR2025/multimodal_vlm/relation-rich_visual_document_generator_for_visual_information_extraction.md)
- [\[ACL 2026\] A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning](a_survey_of_multimodal_mathematical_reasoning_from_perception_alignment_to_reaso.md)
- [\[ACL 2026\] SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding](slideagent_hierarchical_agentic_framework_for_multi-page_visual_document_underst.md)
- [\[AAAI 2026\] Exo2Ego: Exocentric Knowledge Guided MLLM for Egocentric Video Understanding](../../AAAI2026/multimodal_vlm/exo2ego_exocentric_knowledge_guided_mllm_for_egocentric_vide.md)
- [\[CVPR 2026\] PinPoint: Focus, Don't Prune — Identifying Instruction-Relevant Regions for Information-Rich Image Understanding](../../CVPR2026/multimodal_vlm/focus_dont_prune_identifying_instruction-relevant_regions_for_information-rich_i.md)

</div>

<!-- RELATED:END -->
