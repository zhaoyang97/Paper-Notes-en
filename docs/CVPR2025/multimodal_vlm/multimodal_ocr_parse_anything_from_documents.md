---
title: >-
  [Paper Note] Multimodal OCR: Parse Anything from Documents
description: >-
  [CVPR 2025][Multimodal VLM][Document Parsing] This work proposes the Multimodal OCR (MOCR) paradigm, which unifies the parsing of text and graphics (charts, icons, UI, etc.) in documents into a structured text representation (including SVG code). The 3B model achieves a SOTA score of 83.9 on olmOCR-Bench, outperforming Gemini 3 Pro in graphics parsing.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Document Parsing"
  - "OCR"
  - "SVG"
  - "Graphical Reconstruction"
  - "Multimodal Pretraining"
date: 2026-05-08
content_hash: 11e4a7f87d561385
---

# Multimodal OCR: Parse Anything from Documents

**Conference**: CVPR 2025  
**arXiv**: [2603.13032](https://arxiv.org/abs/2603.13032)  
**Code**: [https://github.com/rednote-hilab/dots.mocr](https://github.com/rednote-hilab/dots.mocr)  
**Area**: Multimodal VLM  
**Keywords**: Document Parsing, OCR, SVG, Graphical Reconstruction, Multimodal Pretraining

## TL;DR

This work proposes the Multimodal OCR (MOCR) paradigm, which unifies the parsing of text and graphics (charts, icons, UI, etc.) in documents into a structured text representation (including SVG code). The 3B model achieves a SOTA score of 83.9 on olmOCR-Bench, outperforming Gemini 3 Pro in graphics parsing.

## Background & Motivation

**Background**: Current document parsing is primarily text-centric—identifying and organizing textual content while cropping graphical regions (such as charts and icons) into pixel images, discarding their structural and semantic information.

**Limitations of Prior Work**: A large amount of information in documents is conveyed via graphics (charts, flowcharts, chemical structures, etc.), but traditional OCR pipelines treat these as "black boxes." This makes document parsing inherently "lossy," limiting the volume of structured supervision signals extracted from documents.

**Key Challenge**: Unlike text, which has standardized character representations, graphics lack a standard textual representation. It is crucial to find a unified, executable, editable, and verifiable representation format to capture their structural information.

**Goal**: To design a unified architecture that parses both text and graphics in documents simultaneously, converting graphics into reusable, structured outputs.

**Key Insight**: Modern VLMs already possess the capability to generate executable representations from images (such as SVG). Document graphics can thus be parsed into renderable code instead of pixel crops.

**Core Idea**: Document parsing should not stop at text recognition; graphics should also be treated as first-class citizens—represented uniformly using SVG code to achieve "parse anything."

## Method

### Overall Architecture

dots.mocr is a 3B-parameter end-to-end model comprising: a 1.2B high-resolution vision encoder trained from scratch (supporting ~11M pixel inputs) + a lightweight multimodal connector + a Qwen2.5-1.5B decoder. Given an input document image, the model outputs an ordered structured sequence $\mathbf{S} = [(\mathcal{B}_k, c_k, p_k)]$, where $\mathcal{B}_k$ represents the spatial bounding box, $c_k$ represents the element category, and $p_k$ is the content payload (text $\rightarrow$ plain text / table markup / LaTeX; graphics $\rightarrow$ SVG code).

### Key Designs

1. **Unified Parsing Format**:

    - **Function**: Text and graphics are represented in the same sequence format and generated according to the reading order.
    - **Mechanism**: The payload for text regions is the text transcription (plain text / table markup / LaTeX), while the payload for graphical regions is the SVG code.
    - **Design Motivation**: A unified format makes end-to-end training possible, allowing the model to exploit the semantic relationships between text and graphics.

2. **Four-Stage Training Recipe**:

    - **Function**: To transition gradually from general visual understanding to specialized MOCR capabilities.
    - **Stage 1**: General vision training to establish a stable vision-language interface.
    - **Stage 2**: Broad pre-training (general vision + text-only document parsing) to build a foundation for text OCR.
    - **Stage 3**: Specialized MOCR training, reducing the proportion of general data while increasing graphics parsing (image-to-SVG).
    - **Stage 4**: Instruction tuning refined using high-quality supervised data.
    - Resolution is scaled up stage-by-stage to match the increasing difficulty of the tasks.

3. **Multi-Source Data Engine**:

    - PDF documents: Using dots.ocr as an auto-labeling engine with stratified sampling (language/domain/complexity).
    - Web rendering: HTML/DOM provides aligned structured signals, naturally containing SVG icons and charts.
    - Native SVG resources: Normalized using svgo $\rightarrow$ deduplicated $\rightarrow$ domain-balanced $\rightarrow$ complexity-sampled.
    - General vision data: Retained to maintain broad capabilities.
    - Key steps for SVG data processing: canonicalization, viewBox normalization, and complexity reduction.

4. **OCR Arena Automatic Evaluation**:

    - **Function**: Pairwise comparisons using LLM-as-a-Judge (Gemini 3 Flash) to calculate Elo ratings.
    - **Mechanism**: Traditional metrics such as WER/NED are overly sensitive to surface-form differences, whereas Elo ratings better reflect true quality.
    - Anti-position bias: Each pairwise comparison is conducted twice (with order swapped); inconsistent decisions are treated as a tie.

### Loss & Training

A unified autoregressive objective: predicting structured parsing sequences conditioned on input images + task instructions. Optimization stability is managed via mixture reweighting and curriculum scheduling.

## Key Experimental Results

### Main Results

| Model | olmOCR-Bench | OmniDocBench1.5 (Elo) | XDocParse (Elo) | Avg Elo |
|------|-------------|---------------------|----------------|---------|
| Gemini 3 Pro | — | 1128.0 | 1323.7 | **1210.7** |
| **dots.mocr** | **83.9** | 1059.0 | 1210.7 | **1124.7** |
| dots.ocr | 79.1 | 1027.2 | 1190.3 | 1086.2 |
| HunyuanOCR | — | 1003.9 | 951.1 | 984.2 |

### Graphics Parsing

| Method | UniSVG Score | ChartMimic | Design2Code | ChemDraw |
|------|-------------|-----------|-------------|----------|
| Gemini 3 Pro | 0.735 | 0.788 | 0.760 | 0.839 |
| OCRVerse | 0.763 | 0.799 | — | 0.881 |
| **dots.mocr-svg** | **0.902** | **0.905** | **0.834** | **0.901** |

### Key Findings

- **Open-source SOTA**: dots.mocr ranks second only to Gemini 3 Pro in Elo ratings and sets a new SOTA (83.9) on olmOCR-Bench.
- **Graphics parsing significantly outperforms closed-source models**: Surpasses Gemini 3 Pro by +0.167 on UniSVG and by +0.117 on ChartMimic. The 3B model outperforms large closed-source models in graphic reconstruction.
- **Training the vision encoder from scratch is superior to using a pretrained encoder**: It tailors feature representations specifically for document parsing tasks.
- **Mutual benefit of unified training**: Joint representation learning for text and graphics parsing mutually enhances each other.

## Highlights & Insights

- **The paradigm shift of "parse anything" is highly visionary**: Converting previously discarded graphics into reusable, structured supervision signals is not just about better OCR; it opens up a new data source for multimodal pre-training. Every chart in a PDF can become an image-to-code training sample.
- **SVG as the "textual" representation of graphics**: SVG is renderable, editable, and hierarchical, making it the optimal structured carrier for graphic information. This choice is more general-purpose than Python or HTML.
- **Vision encoder trained from scratch**: For fine-grained tasks like document parsing, the feature distribution of documents differs significantly from ImageNet; training from scratch yields superior results.

## Limitations & Future Work

- **Does not support simultaneous text and SVG output in a single pass**: Currently requires separate runs for page-level text parsing and region-level SVG decoding.
- **Complex real-world photos cannot be represented using SVG**: The method is only applicable to graphics that can be "algorithmically described."
- **Non-uniqueness of SVG code**: The same image can be generated by multiple different SVG codes, which increases training difficulty.

## Related Work & Insights

- **vs GOT-OCR / DeepSeek-OCR**: These are end-to-end text OCR models that do not handle graphics. MOCR expands its scope to graphics, making it more comprehensive.
- **vs UniSVG / StarVector**: These models focus on the single task of image-to-SVG. MOCR performs both text OCR and graphics parsing within a unified framework.
- **Insights**: The concept of "parsing everything into executable code" shares commonalities with CodePercept—using code/SVG as representations that are more precise than natural language.

## Rating

- Novelty: ⭐⭐⭐⭐ The paradigm of elevating graphics to a first-class parsing target is novel, though the technical components themselves are relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Dual evaluation on document parsing and graphics parsing, leveraging the Elo system, olmOCR-Bench breakdown, and 6 SVG benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Highly systematic, though the paper is quite long.
- Value: ⭐⭐⭐⭐⭐ Extremely significant contribution to the Document AI community, open-sourcing a 3B model, data engine, and a new evaluation paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Describe Anything Anywhere At Any Moment](../../CVPR2026/multimodal_vlm/describe_anything_anywhere_at_any_moment.md)
- [\[ICML 2026\] Very Efficient Listwise Multimodal Reranking for Long Documents](../../ICML2026/multimodal_vlm/very_efficient_listwise_multimodal_reranking_for_long_documents.md)
- [\[NeurIPS 2025\] MME-VideoOCR: Evaluating OCR-Based Capabilities of Multimodal LLMs in Video Scenarios](../../NeurIPS2025/multimodal_vlm/mme-videoocr_evaluating_ocr-based_capabilities_of_multimodal_llms_in_video_scena.md)
- [\[AAAI 2026\] OIDA-QA: A Multimodal Benchmark for Analyzing the Opioid Industry Documents Archive](../../AAAI2026/multimodal_vlm/oida-qa_a_multimodal_benchmark_for_analyzing_the_opioid_industry_documents_archi.md)
- [\[ACL 2025\] Improving MLLM's Document Image Machine Translation via Synchronously Self-reviewing Its OCR Proficiency](../../ACL2025/multimodal_vlm/improving_mllms_document_image_machine_translation_via_synchronously_self-review.md)

</div>

<!-- RELATED:END -->
