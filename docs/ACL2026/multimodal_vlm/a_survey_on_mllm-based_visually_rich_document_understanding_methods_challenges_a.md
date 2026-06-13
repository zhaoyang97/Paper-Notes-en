---
title: >-
  [Paper Note] A Survey on MLLM-based Visually Rich Document Understanding: Methods, Challenges, and Emerging Trends
description: >-
  [ACL 2026][Multimodal VLM][Visually Rich Document Understanding] A systematic survey of Visually Rich Document Understanding (VRDU) based on Multimodal Large Language Models (MLLM)…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Visually Rich Document Understanding"
  - "Multimodal Large Language Models"
  - "OCR-free"
  - "Document Information Extraction"
  - "Retrieval-Augmented Generation"
date: 2026-05-08
content_hash: 5c9562ea17e8dfa3
---

# A Survey on MLLM-based Visually Rich Document Understanding: Methods, Challenges, and Emerging Trends

**Conference**: ACL 2026  
**arXiv**: [2507.09861](https://arxiv.org/abs/2507.09861)  
**Code**: None  
**Area**: Document Understanding / Multimodal LLM  
**Keywords**: Visually Rich Document Understanding, Multimodal Large Language Models, OCR-free, Document Information Extraction, Retrieval-Augmented Generation

## TL;DR

A systematic survey of Visually Rich Document Understanding (VRDU) based on Multimodal Large Language Models (MLLM), categorizing OCR-based and OCR-free methods from the dimensions of feature representation/fusion and training paradigms, while discussing emerging directions such as data scarcity, multi-page documents, multilingual support, RAG, and agents.

## Background & Motivation

**Background**: Visually Rich Document Understanding (VRDU) aims to automatically extract and understand information from documents containing complex visual, textual, and layout elements, with wide applications in finance, healthcare, and education. With the rapid development of MLLMs, the field is undergoing a paradigm shift from traditional methods to MLLM-based approaches.

**Limitations of Prior Work**: (1) Early methods rely on OCR pipelines, leading to cascading error propagation; (2) The multimodality of documents (text, vision, layout) increases the complexity of feature fusion; (3) Scarcity of labeled data constrains supervised learning methods; (4) Multi-page and multilingual document processing remains challenging.

**Key Challenge**: While MLLMs perform excellently on general vision-language tasks, document understanding has specific requirements—such as understanding precise layout relationships, table structures, and printed/handwritten text—which general MLLMs struggle to handle directly.

**Goal**: To provide a comprehensive survey of MLLM-based VRDU, covering method classification, training strategies, challenges, and future directions, thereby offering a systematic roadmap for researchers.

**Key Insight**: The survey is organized around two core dimensions: (1) representation and fusion techniques for text, vision, and layout features; (2) pre-training, instruction tuning, and training strategies.

**Core Idea**: MLLM-based VRDU is evolving from OCR-dependent to OCR-free approaches, while expanding from single-page static understanding to multi-page dynamic interaction (RAG, agents).

## Method

### Overall Architecture

The survey categorizes MLLM-based VRDU methods into two primary types: OCR-Dependent (requiring external OCR output as text input) and OCR-Free (end-to-end understanding directly from document images), with further subdivisions based on feature fusion techniques and LLM backbones.

### Key Designs

1.  **OCR-Dependent Methods**:

    - **Function**: Utilizes text and layout information provided by external OCR engines, combined with LLMs for document understanding.
    - **Mechanism**: Typical examples include DocLLM (using cross-attention to fuse text and layout), ICL-D3IE (leveraging in-context learning of GPT-3 for document IE), and LayoutLLM (combining layout encoding from LayoutLMv3 with Vicuna’s generative capabilities).
    - **Design Motivation**: OCR provides precise textual content, but introduces a trade-off between OCR error propagation and pipeline complexity.

2.  **OCR-Free Methods**:

    - **Function**: End-to-end document understanding directly from document images without the need for external OCR.
    - **Mechanism**: Representative methods include the mPLUG-DocOwl series (perceiving document images directly via visual encoders), TextMonkey (employing sliding windows for high-resolution documents), and InternVL-based methods (dynamic resolution processing).
    - **Design Motivation**: To eliminate OCR error propagation and achieve true end-to-end processing, though this requires visual encoders capable of precisely identifying fine-grained text.

3.  **Training Paradigm Classification**:

    - **Function**: Systematizes the MLLM training workflow.
    - **Mechanism**: A three-stage paradigm—Pre-training (PT) for learning document representation foundations, Instruction Tuning (IT) for aligning task instruction understanding, and Downstream Fine-tuning (FT) for specific task adaptation. Different methods adopt various combinations of these stages.
    - **Design Motivation**: To understand how training strategy choices impact final performance, assisting researchers in making informed design decisions.

### Loss & Training

The methods covered utilize diverse training strategies: standard autoregressive language modeling loss, contrastive learning (e.g., CLIP-style), and text-layout alignment loss. Pre-training typically involves large-scale document-text pairs, while instruction tuning employs structured QA formats.

## Key Experimental Results

### Main Results

| Method | Type | Task | Modality | LLM Backbone | Multi-page |
| :--- | :--- | :--- | :--- | :--- | :--- |
| DocLLM | OCR-Dep | KIE, QA, DC | T, L | Custom | Single-page |
| LayoutLLM | OCR-Dep | KIE, QA | T, V, L | Vicuna-7B | Single-page |
| mPLUG-DocOwl | OCR-Free | QA | V | mPLUG-Owl | Single-page |
| TextMonkey | OCR-Free | QA | V | Qwen-VL | Single-page |
| InternVL-Doc | OCR-Free | QA, KIE | V | InternVL | Multi-page |
| DocThinker | OCR-Free | QA, KIE | T, V | Qwen2.5-VL | Single-page |

### Challenges & Emerging Trends

| Challenge | Current Status | Future Direction |
| :--- | :--- | :--- |
| Data Scarcity | Synthetic data + Transfer learning | Self-supervised pre-training + Few-shot learning |
| Multi-page Documents | Supported by a few methods | Dynamic page selection + Retrieval augmentation |
| Multilingual | Primarily English | Multilingual pre-training + Cross-lingual transfer |
| RAG Integration | Preliminary exploration | Document retrieval + Generation pipeline |
| Agent Frameworks | Emerging direction | Multi-tool collaborative document understanding agents |

### Key Findings

- OCR-Free methods are rapidly catching up with OCR-Dependent methods, particularly with the support of high-resolution visual encoders.
- Multi-page document understanding is currently the most significant bottleneck, as most methods still only support single pages.
- The introduction of RAG and agent frameworks offers a new path for document understanding, moving from "understanding" to "application."

## Highlights & Insights

- The classification dimensions are clearly defined: a complete method space is constructed across OCR dependency, feature fusion, and training paradigms.
- The model summary tables are highly practical, covering key information such as LLM backbones, visual encoders, training stages, multi-page support, and prompt formats.
- Forward-looking discussions on emerging directions (RAG, agents) provide a clear trajectory for subsequent research.

## Limitations & Future Work

- The survey covers methods up to mid-2025; the rapid evolution of MLLMs may soon render some parts obsolete.
- There is a lack of quantitative comparisons on unified benchmarks, making it difficult to directly compare the performance of different methods.
- The discussion on computational cost and efficiency is not sufficiently in-depth.
- Future directions: (1) Unified multi-task multi-page document understanding frameworks; (2) Trustworthy document understanding (hallucination control); (3) Integration of document understanding with knowledge graphs.

## Related Work & Insights

- **vs Traditional Document Understanding Surveys**: Focuses on new methods in the MLLM era, covering more OCR-free and generative approaches.
- **vs General MLLM Surveys**: Dives deeper into the specific requirements of document understanding, such as layout awareness, table understanding, and high resolution.
- **vs Document AI Application Surveys**: Prioritizes technical method classification over specific application scenarios.

## Rating

- Novelty: ⭐⭐⭐ Survey article, prioritizing systematicity over originality.
- Experimental Thoroughness: ⭐⭐⭐ Comprehensive method coverage, but lacks unified quantitative comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, reasonable classification dimensions, and high information density.
- Value: ⭐⭐⭐⭐ Significant reference value for researchers in the VRDU field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning](a_survey_of_multimodal_mathematical_reasoning_from_perception_alignment_to_reaso.md)
- [\[ACL 2026\] SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding](slideagent_hierarchical_agentic_framework_for_multi-page_visual_document_underst.md)
- [\[ICLR 2026\] Index-Preserving Lightweight Token Pruning for Efficient Document Understanding](../../ICLR2026/multimodal_vlm/index-preserving_lightweight_token_pruning_for_efficient_document_understanding_.md)
- [\[CVPR 2026\] PinPoint: Focus, Don't Prune — Identifying Instruction-Relevant Regions for Information-Rich Image Understanding](../../CVPR2026/multimodal_vlm/focus_dont_prune_identifying_instruction-relevant_regions_for_information-rich_i.md)
- [\[ACL 2026\] Towards Visually Grounded Multimodal Summarization via Cross-Modal Transformer and Gated Attention](towards_visually_grounded_multimodal_summarization_via_cross-modal_transformer_a.md)

</div>

<!-- RELATED:END -->
