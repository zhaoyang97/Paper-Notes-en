---
title: >-
  [Paper Note] A Survey on MLLM-based Visually Rich Document Understanding: Methods, Challenges, and Emerging Trends
description: >-
  [ACL 2026 Findings][Multimodal VLM][Visually Rich Document Understanding] This paper systematically reviews Visually Rich Document Understanding (VRDU) based on Multimodal Large Language Models (MLLMs), categorizing OCR-based and OCR-free methods from two dimensions: feature representation/fusion and training paradigms, while discussing emerging directions such as data scarcity, multi-page documents, multilingual support, RAG, and agents.
tags:
  - "ACL 2026 Findings"
  - "Multimodal VLM"
  - "Visually Rich Document Understanding"
  - "Multimodal Large Language Models"
  - "OCR-free"
  - "Document Information Extraction"
  - "Retrieval-Augmented Generation"
date: 2026-05-08
content_hash: 21c6238919ed217c
---

# A Survey on MLLM-based Visually Rich Document Understanding: Methods, Challenges, and Emerging Trends

**Conference**: ACL 2026 Findings  
**arXiv**: [2507.09861](https://arxiv.org/abs/2507.09861)  
**Code**: None  
**Area**: Document Understanding / Multimodal LLM  
**Keywords**: Visually Rich Document Understanding, Multimodal Large Language Models, OCR-free, Document Information Extraction, Retrieval-Augmented Generation

## TL;DR

This paper systematically reviews Visually Rich Document Understanding (VRDU) based on Multimodal Large Language Models (MLLMs), categorizing OCR-based and OCR-free methods from two dimensions: feature representation/fusion and training paradigms, while discussing emerging directions such as data scarcity, multi-page documents, multilingual support, RAG, and agents.

## Background & Motivation

**Background**: Visually Rich Document Understanding (VRDU) aims to automatically extract and understand information from documents containing complex visual, textual, and layout elements, with wide applications in finance, healthcare, and education. With the rapid development of MLLMs, the field is undergoing a paradigm shift from traditional methods to MLLM-based methods.

**Limitations of Prior Work**: (1) Early methods relied on OCR pipelines, where errors propagate cumulatively; (2) The multimodality of documents (text, vision, layout) increases the complexity of feature fusion; (3) Scarcity of labeled data constrains supervised learning methods; (4) Handling multi-page and multilingual documents remains challenging.

**Key Challenge**: While MLLMs perform well on general vision-language tasks, document understanding has specificities—requiring the understanding of precise layout relationships, table structures, and printed/handwritten text, which general MLLMs struggle to handle directly.

**Goal**: Ours aims to provide a comprehensive survey of MLLM-based VRDU, covering method classification, training strategies, challenges, and future directions, offering a systematic roadmap for researchers.

**Key Insight**: Ours is organized from two core dimensions—(1) representation and fusion techniques for text, visual, and layout features; (2) pre-training, instruction tuning, and training strategies.

**Core Idea**: MLLM-based VRDU is evolving from OCR-dependent to OCR-free, and expanding from single-page static understanding to multi-page dynamic interaction (RAG, Agents).

## Method

### Overall Architecture

The survey categorizes MLLM-based VRDU methods into two categories: OCR-Dependent (requiring external OCR output as text input) and OCR-Free (end-to-end understanding directly from document images). Each category is further subdivided by feature fusion methods and LLM backbones.

### Key Designs

**1. OCR-Dependent Methods: Feeding external OCR text/layout to the LLM; high recognition accuracy but suffers from cascading errors**

This category follows the logic of traditional document IE—first extracting text and coordinates via OCR engines, then passing them to the LLM for understanding. Representative works have different focuses: DocLLM uses cross-attention to fuse text and layout features, ICL-D3IE directly leverages GPT-3's in-context learning to handle document information extraction, and LayoutLLM connects LayoutLMv3's layout encoding to Vicuna's generation capability. Their common benefit is that OCR provides precise text content, sparing the model from the burden of character recognition from pixels; the cost is that once OCR fails, errors propagate through the pipeline, and it introduces an external dependency that OCR-free routes aim to avoid.

**2. OCR-Free Methods: Visual encoders read images directly; end-to-end but requires fine-grained recognition**

To eliminate the OCR pipeline and error propagation, OCR-Free methods allow visual encoders to perceive document images directly and output understanding results end-to-end. The challenge lies in document text being dense and small, so the evolution focuses on "how to feed high-resolution documents into visual encoders": the mPLUG-DocOwl series models document images directly, TextMonkey uses a sliding window to process high-resolution pages, and InternVL-based methods rely on dynamic resolutions to adapt to different document sizes. This achieves true end-to-end processing but shifts the recognition burden entirely to the visual encoder, demanding high fine-grained recognition capabilities.

**3. Training Paradigms: Three stages (Pre-training → Instruction Tuning → Downstream Fine-tuning) and their combinations**

The survey deconstructs the MLLM training process into three stages: Pre-training (PT) to learn the foundation of document representation, Instruction Tuning (IT) to align understanding of task instructions, and Downstream Fine-tuning (FT) to adapt to specific tasks. Different methods use various combinations of these stages. Explicitly deconstructing the training process allows readers to see where performance differences originate, making design choices more informed rather than treating training as a black box.

### Loss & Training

Methods covered in the survey use various training strategies: standard autoregressive language modeling loss, contrastive learning (e.g., CLIP-style), and text-layout alignment loss. Pre-training usually employs large-scale document-text pairs, while instruction tuning uses structured QA formats.

## Key Experimental Results

### Main Results

| Method | Type | Task | Modality | LLM Backbone | Multi-page |
|------|------|------|------|---------|------|
| DocLLM | OCR-Dep | KIE, QA, DC | T, L | Custom | Single |
| LayoutLLM | OCR-Dep | KIE, QA | T, V, L | Vicuna-7B | Single |
| mPLUG-DocOwl | OCR-Free | QA | V | mPLUG-Owl | Single |
| TextMonkey | OCR-Free | QA | V | Qwen-VL | Single |
| InternVL-Doc | OCR-Free | QA, KIE | V | InternVL | Multi-page |
| DocThinker | OCR-Free | QA, KIE | T, V | Qwen2.5-VL | Single |

### Challenges & Trends

| Challenge | Current Status | Future Direction |
|------|---------|---------|
| Data Scarcity | Synthetic data + Transfer learning | Self-supervised PT + Few-shot learning |
| Multi-page Documents | Supported by few methods | Dynamic page selection + RAG |
| Multilingual | Primarily English | Multilingual PT + Cross-lingual transfer |
| RAG Integration | Preliminary exploration | Doc retrieval + Generation pipeline |
| Agent Frameworks | Emerging direction | Multi-tool collaborative document agents |

### Key Findings
- OCR-Free methods are rapidly catching up with OCR-Dependent methods, especially with the support of high-resolution visual encoders.
- Multi-page document understanding is currently the largest bottleneck, with most methods only supporting single pages.
- The introduction of RAG and agent frameworks provides new paths for document understanding from "understanding" to "application".

## Highlights & Insights
- The classification dimensions are clearly designed: a complete method space is built from OCR dependency $\times$ feature fusion $\times$ training paradigm.
- The model summary tables are practical, covering key information such as LLM backbone, visual encoder, training stages, multi-page support, and prompt formats.
- Forward-looking discussions on emerging directions (RAG, agents) provide guidance for subsequent research.

## Limitations & Future Work
- The survey covers methods up to mid-2025; the rapid development of MLLMs may quickly make some content outdated.
- There is a lack of quantitative comparison on unified benchmarks, making it difficult to directly compare the performance of different methods.
- The discussion on computational cost and efficiency is not deep enough.
- Future directions: (1) Unified multi-task multi-page document understanding frameworks; (2) Trustworthy document understanding (hallucination control); (3) Combining document understanding with knowledge graphs.

## Related Work & Insights
- **vs Traditional Document Understanding Surveys**: Focuses on new methods in the MLLM era, covering more OCR-free and generative methods.
- **vs General MLLM Surveys**: Dives deeper into specific requirements for document understanding (layout awareness, table understanding, high resolution).
- **vs Document AI Application Surveys**: Focuses more on technical method classification rather than application scenarios.

## Rating
- Novelty: ⭐⭐⭐ Survey paper, emphasizing systematicity over originality.
- Experimental Thoroughness: ⭐⭐⭐ Comprehensive method coverage, but lacks unified quantitative comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, reasonable classification dimensions, and high table information density.
- Value: ⭐⭐⭐⭐ Important reference value for researchers in the VRDU field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] A Survey of Deep Learning for Geometry Problem Solving](a_survey_of_deep_learning_for_geometry_problem_solving.md)
- [\[CVPR 2025\] Relation-Rich Visual Document Generator for Visual Information Extraction](../../CVPR2025/multimodal_vlm/relation-rich_visual_document_generator_for_visual_information_extraction.md)
- [\[ACL 2026\] SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding](slideagent_hierarchical_agentic_framework_for_multi-page_visual_document_underst.md)
- [\[ACL 2026\] Towards Visually Grounded Multimodal Summarization via Cross-Modal Transformer and Gated Attention](towards_visually_grounded_multimodal_summarization_via_cross-modal_transformer_a.md)
- [\[ACL 2026\] Prune-then-Merge: Towards Efficient Multi-Vector Visual Document Retrieval](sculpting_the_vector_space_towards_efficient_multi-vector_visual_document_retrie.md)

</div>

<!-- RELATED:END -->
