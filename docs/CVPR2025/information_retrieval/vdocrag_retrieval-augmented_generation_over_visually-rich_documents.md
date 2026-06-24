---
title: >-
  [Paper Note] VDocRAG: Retrieval-Augmented Generation over Visually-Rich Documents
description: >-
  [CVPR 2025][Information Retrieval & RAG][Document RAG] Proposes the first RAG framework that directly takes document images (rather than parsed text) as input, utilizing an LVLM as a dual-encoder retriever plus two self-supervised pre-training tasks (contrastive + generative) to achieve document image retrieval, outperforming text RAG by 24 points on ChartQA.
tags:
  - "CVPR 2025"
  - "Information Retrieval & RAG"
  - "Document RAG"
  - "Visual Retrieval"
  - "LVLM Retriever"
  - "Self-Supervised Pre-training"
  - "OpenDocVQA"
date: 2026-05-08
content_hash: cb66bacdadc7f015
---

# VDocRAG: Retrieval-Augmented Generation over Visually-Rich Documents

**Conference**: CVPR 2025  
**arXiv**: [2504.09795](https://arxiv.org/abs/2504.09795)  
**Code**: [https://vdocrag.github.io](https://vdocrag.github.io)  
**Area**: Information Retrieval  
**Keywords**: Document RAG, Visual Retrieval, LVLM Retriever, Self-Supervised Pre-training, OpenDocVQA

## TL;DR
Proposes the first RAG framework that directly takes document images (rather than parsed text) as input, utilizing an LVLM as a dual-encoder retriever plus two self-supervised pre-training tasks (contrastive + generative) to achieve document image retrieval, outperforming text RAG by 24 points on ChartQA.

## Background & Motivation

**Background**: RAG systems for Document QA typically parse documents into plain text first using OCR, and then perform retrieval and generation using text retrievers. However, a significant amount of non-textual information (charts, table layouts, images) in the document is lost during text parsing.

**Limitations of Prior Work**: (1) Text parsing loses the visual structure of charts (e.g., bar chart height contrasts, table row-column structures). (2) Existing visual retrievers (e.g., VisRAG-Ret) are based on CLIP-like models, which have insufficient understanding of document-specific layouts and text. (3) There is a lack of standard benchmarks requiring reasoning across multiple document pages.

**Key Challenge**: Document understanding requires joint visual and textual comprehension; however, the retrieval step in RAG typically relies only on text, discarding critical visual information.

**Goal**: Design an end-to-end visual document RAG framework that directly operates on document images, from retrieval to generation.

**Key Insight**: Utilize an LVLM (InternVL2) as both the document image encoder and query encoder, teaching the LVLM to compress document images into retrievable embeddings through two self-supervised pre-training tasks (RCR contrastive learning + RCG generative learning).

**Core Idea**: Use an LVLM as a dual-encoder retriever for document images, achieving self-supervision via contrastive learning with OCR text and generative pre-training with a customized attention mask to build an end-to-end visual document RAG.

## Method

### Overall Architecture
Query → VDocRetriever (LVLM encoder, using the `<EOS>` token as the embedding) → retrieves the most relevant document images → VDocGenerator (the same LVLM) receives the query + retrieved document images → generates the answer.

### Key Designs

1. **VDocRetriever (LVLM Dual-Encoder)**:

    - **Function**: Encodes document images and queries independently into comparable embeddings.
    - **Mechanism**: Uses InternVL2-4B to separately encode document images and query text, taking the hidden state of the `<EOS>` token as the embedding. It supports dynamic high-resolution image encoding (multiple patches per page).
    - **Design Motivation**: LVLMs understand documents better than CLIP as they have encountered a large volume of text-dense images and layout structures during pre-training.

2. **RCR Pre-training (Representation Compression via Retrieval)**:

    - **Function**: Teaches the LVLM to compress document images into retrievable embeddings using contrastive learning.
    - **Mechanism**: Extracts text from documents via OCR → performs text-image contrastive learning (using InfoNCE loss). The OCR text serves as the "natural positive sample" for the document image.
    - **Design Motivation**: The text inside a document image and the image itself form a natural cross-modal pair, requiring no human annotation.

3. **RCG Pre-training (Representation Compression via Generation)**:

    - **Function**: Enables the `<EOS>` token to learn better global representations via a custom attention mask.
    - **Mechanism**: Uses OCR text as the decoding target but forces the generation process to attend only to the `<EOS>` token (preventing direct access to image patch tokens). This forces the `<EOS>` token to compress all document information.
    - **Design Motivation**: Ablation studies show that RCG has a larger impact than RCR (removing RCG drops retrieval by 5.6 points vs. removing RCR drops it by 1.4 points), indicating that generative compression enhances retrieval quality more effectively than contrastive learning.

### Loss & Training
RCR: InfoNCE contrastive loss. RCG: next-token generation loss + custom attention mask. OpenDocVQA dataset: 43K QA over 206K images, 7+ document types, including multi-hop reasoning questions.

## Key Experimental Results

### Main Results

| Method | ChartQA | SlideVQA | InfoVQA | DUDE |
|------|---------|----------|---------|------|
| Text RAG | 28.0 | 28.6 | 40.5 | 40.1 |
| **VDocRAG** | **52.0** | **44.2** | **56.2** | **48.5** |
| Gain | **+24.0** | **+15.6** | **+15.7** | **+8.4** |

### Ablation Study

| Pre-training | SlideVQA Retrieval | InfoVQA Retrieval |
|--------|-------------|-------------|
| Full | 77.3 | 72.9 |
| w/o RCR | 75.9 (-1.4) | 71.1 (-1.8) |
| w/o RCG | 71.7 (-5.6) | 68.8 (-4.1) |
| w/o Both | 71.0 (-6.3) | 66.8 (-6.1) |

### Key Findings
- **Visual RAG $\gg$ Text RAG**: An absolute gain of +24 points on ChartQA, as the visual structure of charts is completely lost during text parsing.
- **RCG is More Critical than RCR**: Generative pre-training forces the `<EOS>` token to compress global information, having a larger impact on retrieval quality.
- **OpenDocVQA Fills the Gap**: The first open-domain document QA dataset requiring multi-page visual reasoning.

## Highlights & Insights
- **The "look directly at the image without reading parsed text" document understanding paradigm** holds great potential—avoiding OCR errors and format loss.
- **The attention mask design of RCG** is elegant—forcing `<EOS>` to learn better representation by limiting the information source during generation.

## Limitations & Future Work
- The LVLM encoder is relatively heavy (4.2B), requiring offline encoding + indexing for large-scale retrieval scenarios.
- Only static documents are supported; dynamic web pages/interactive documents remain unverified.
- OpenDocVQA is biased towards English, and multi-lingual document support remains unexplored.

## Related Work & Insights
- **vs. VisRAG**: VisRAG utilizes SigLIP for visual retrieval. VDocRetriever employs LVLMs + self-supervised pre-training, achieving comparable or superior results on multiple benchmarks.
- **vs. Text RAG (e.g., E5-Mistral)**: Text RAG shows advantages on text-dense documents, but lags behind VDocRAG significantly on chart/table documents.

## Rating
- Novelty: ⭐⭐⭐⭐ Utilizing LVLMs for document retrieval + RCG pre-training is a novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on both retrieval and generation, with clear pre-training ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The framework is clearly described, and the contribution of the OpenDocVQA dataset is valuable.
- Value: ⭐⭐⭐⭐⭐ Direct engineering value for document understanding RAG.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Hybrid-Vector Retrieval for Visually Rich Documents: Combining Single-Vector Efficiency and Multi-Vector Accuracy](../../ACL2026/information_retrieval/hybrid-vector_retrieval_for_visually_rich_documents_combining_single-vector_effi.md)
- [\[ACL 2025\] Re-identification of De-identified Documents with Autoregressive Infilling](../../ACL2025/information_retrieval/reidentification_deidentified.md)
- [\[CVPR 2025\] Few-Shot Recognition via Stage-Wise Retrieval-Augmented Finetuning](few-shot_recognition_via_stage-wise_retrieval-augmented_finetuning.md)
- [\[CVPR 2025\] RANGE: Retrieval Augmented Neural Fields for Multi-Resolution Geo-Embeddings](range_retrieval_augmented_neural_fields_for_multi-resolution_geo-embeddings.md)
- [\[ACL 2025\] GeAR: Generation Augmented Retrieval](../../ACL2025/information_retrieval/gear_generation_augmented_retrieval.md)

</div>

<!-- RELATED:END -->
