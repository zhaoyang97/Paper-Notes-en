---
title: >-
  [Paper Note] MARTEN: Visual Question Answering with Mask Generation for Multi-Modal Document Understanding
description: >-
  [CVPR 2025][Multimodal VLM][Document Understanding] The VQAMask pre-training paradigm is proposed, which introduces an auxiliary mask generation task (discarded during inference) on top of VQA text parsing. Through explicit spatial alignment supervision, it enhances the vision encoder's perception of text regions in document images. The resulting Marten model achieves state-of-the-art (SOTA) performance among 8B-level MLLMs across multiple document understanding tasks.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Document Understanding"
  - "VQAMask"
  - "Spatially-Aware Alignment"
  - "Mask Generation"
  - "OCR-free"
  - "Visual Text Recognition"
date: 2026-05-08
content_hash: d2843d6e597220fe
---

# MARTEN: Visual Question Answering with Mask Generation for Multi-Modal Document Understanding

**Conference**: CVPR 2025  
**arXiv**: [2503.14140](https://arxiv.org/abs/2503.14140)  
**Code**: [GitHub](https://github.com/PriNing/Marten)  
**Area**: Multimodal VLM  
**Keywords**: Document Understanding, VQAMask, Spatially-Aware Alignment, Mask Generation, OCR-free, Visual Text Recognition

## TL;DR

The VQAMask pre-training paradigm is proposed, which introduces an auxiliary mask generation task (discarded during inference) on top of VQA text parsing. Through explicit spatial alignment supervision, it enhances the vision encoder's perception of text regions in document images. The resulting Marten model achieves state-of-the-art (SOTA) performance among 8B-level MLLMs across multiple document understanding tasks.

## Background & Motivation

**Background**:
Multimodal Large Language Models (MLLMs) have been widely applied to document understanding tasks, including document VQA, table VQA, and chart VQA. Existing methods are categorized into OCR-dependent and OCR-free approaches. The latter, which directly predicts answers from images in an end-to-end manner, is the mainstream development direction.

**Limitations of Prior Work**:
1. Existing pre-training tasks (e.g., full-text recognition, text grounding) of OCR-free MLLMs primarily perform implicit alignment at the semantic level.
2. They lack explicit supervision of the spatial positions of visual text, which may cause models to over-rely on the language capabilities of LLMs rather than truly "seeing" the image.
3. The absence of spatially-aware supervision leads to model hallucinations—outputs rely on the semantic speculation of the LLM rather than actual observation by the vision encoder.
4. Dense and small text in high-resolution document images cannot be accurately located and recognized solely through semantic alignment.

**Key Challenge**:
Existing VQA-style pre-training only implicitly teaches the model the correspondence between images and text, lacking explicit spatial position supervision, which leads to vision encoder features that are insufficiently sensitive to space.

**Goal**:
Design a pre-training method that balances both semantic and spatial alignment, enabling the vision encoder to truly learn "where to look".

**Key Insight**:
Introduce an auxiliary mask generation module in the intermediate layers of the LLM, leveraging the cross-attention between vision tokens and language tokens to generate masks of text regions, thereby explicitly guiding spatial alignment. This module is discarded during inference, resulting in zero extra overhead.

**Core Idea**:
Add an auxiliary mask generation task during pre-training to force the alignment of visual features with spatial text positions, and discard the mask module during inference to avoid adding any inference cost.

## Method

### Overall Architecture

The training of Marten consists of two stages:
1. **VQAMask Alignment Training**: Simultaneously optimizes the VQA text parsing and mask generation tasks, with the vision encoder and MLP trainable, while the LLM is frozen.
2. **Visual-Language Genenerative Training**: Discards the mask module, unfreezes the LLM, and performs SFT on a variety of document VQA data.

Model architecture: Visual Foundation Model (VFM) → Modality Connector (MLP) → LLM → Language Output + Mask Generation Module (MGM, training only)

### Key Designs

### Key Design 1: VQAMask Pre-training Paradigm

**Function**: Achieves both semantic-level and spatial-level vision-language alignment simultaneously.

**Mechanism**: Joint optimization of two sub-tasks:
- **VQA Text Parsing**: Six types of OCR-related QA tasks (full-text recognition, coordinate recognition, text grounding, formula/table/chart conversion), optimized via the autoregressive loss $\mathcal{L}_{vqa}$ of the LLM's output layer.
- **Mask Generation**: Extracts the cross-attention maps between visual tokens (queries) and language tokens (keys) at the intermediate layers of the LLM, restores them to the original image resolution via transposed convolutions, and supervises using Dice Loss + Cross-Entropy Loss:
$$\mathcal{L}_{mask} = l_{\text{DICE}}(\tilde{\mathbf{M}}, \mathbf{M}) + l_{\text{CE}}(\tilde{\mathbf{M}}, \mathbf{M})$$

**Design Motivation**: Mask generation serves as an explicit spatial supervision signal, forcing visual tokens to establish accurate spatial correspondences with the corresponding text regions within the intermediate layers of the LLM. The MGM is discarded during inference, introducing zero extra overhead.

### Key Design 2: Mask Generation Module (MGM)

**Function**: Extracts spatial attention information from the LLM's hidden states and generates text region masks.

**Mechanism**:
1. Extract the hidden states $(\mathbf{V}^k, \mathbf{Q}^k, \mathbf{A}^k)$ from the $k-1$-th layer of the LLM.
2. Concatenate the question and answer tokens into $\mathbf{H}^k = [\mathbf{Q}^k, \mathbf{A}^k]$.
3. Use a 4-layer Transformer to compute cross-attention from visual tokens (queries) to language tokens (keys/values).
4. Reorganize 1D visual tokens into 2D space, and restore them to the original image resolution using transposed convolutions $\phi$.
5. Output the predicted mask $\tilde{\mathbf{M}} = \phi(\mathbf{V}_{attn})$.

**Pixel Shuffle Improvement**: Employs pixel shuffle with local windows (4×4) instead of global operations to keep spatial structures from being destroyed.

**Design Motivation**: Utilizing cross-attention allows visual tokens to highlight corresponding text regions under the semantic guidance of language tokens, achieving accurate spatial alignment.

### Key Design 3: Unlabeled Mask Acquisition Pipeline

**Function**: Automatically generates large-scale text region mask labels without manual annotation.

**Mechanism**: A three-stage pipeline:
1. **Detection**: Use PaddleOCR to detect all text regions in the image and crop individual text instance images.
2. **Clustering**: Perform K-means clustering on each cropped image to divide pixels into two categories. Pixels with smaller distances to the center are designated as foreground (text), followed by validation of edge pixel values.
3. **Stitching**: Stitch all cropped masks back into the full image using their original coordinates.

The **MTMask6M** dataset was built based on this pipeline: 6 million image-mask pairs, covering documents (3.36M), tables (600K), charts (475K), math formulas (200K), scene text (396K), etc.

**Design Motivation**: The boundary between text and background in document scenarios is usually clear. Simple clustering binarization is sufficient to obtain high-quality masks, avoiding expensive manual annotations.

## Key Experimental Results

### Main Results: Comparison with 8B-level MLLMs

| Method | DocVQA | InfoVQA | DeepForm | KLC | WTQ | TabFact | FUNSD | SROIE |
|------|--------|---------|----------|-----|-----|---------|-------|-------|
| DocOwl-1.5 | 82.2 | 50.7 | 68.8 | 38.7 | 40.6 | 80.2 | - | - |
| Mini-Monkey | 87.4 | 60.1 | - | - | - | - | 42.9 | 70.3 |
| MM1.5 | 88.1 | 59.5 | - | - | 46.0 | 75.9 | - | - |
| **Marten** | **88.5** | **60.5** | **75.0** | **40.5** | **46.2** | **84.2** | **44.4** | **80.4** |

- Achieves a 0.4% gain on DocVQA/InfoVQA over the previous SOTA.
- Delivers a 6.2% improvement on DeepForm and a 10.1% increase on SROIE.
- Demonstrates particularly pronounced advantages in KIE (Key Information Extraction) tasks.

### Ablation Study

| Setting | DocVQA | InfoVQA | TextVQA |
|------|--------|---------|---------|
| VQA Only | 87.1 | 58.3 | 78.6 |
| VQA + Mask (VQAMask) | **88.5** | **60.5** | **79.8** |

- VQAMask brings consistent improvements across all tasks.
- The mask generation task is effective across different vision encoders (SigLIP, InternViT) and LLMs (Qwen2, InternLM2).

### Key Findings

- The local window strategy for pixel shuffle outperforms the global strategy.
- Performing mask generation using the LLM's intermediate layers (rather than the final layer) yields the best results.
- The data diversity of MTMask6M is crucial for performance.

## Highlights & Insights

1. **Elegant Auxiliary Task Concept**: Mask generation is an auxiliary training-time task that is completely discarded during inference, achieving "free" performance gains.
2. **Spatial Alignment vs. Semantic Alignment**: Exposes the limitation of existing methods' over-reliance on the semantic capabilities of LLMs, returning to the nature of visual perception through explicit spatial supervision.
3. **Automatic Mask Generation Pipeline**: Exploits the clear text-to-background contrast in document scenarios to obtain annotations simply and efficiently.
4. **Strong Generalization**: VQAMask is insensitive to the choice of vision encoder and LLM, serving as a general pre-training enhancement strategy.
5. **Large-scale 6M Dataset**: MTMask6M spans 5 major categories of document scenarios, making it the largest mask-annotated dataset in this domain.

## Limitations & Future Work

1. The mask acquisition pipeline relies on OCR tools (PaddleOCR), which might fail in extreme scenarios.
2. K-means clustering assumes a distinct color contrast between text and background, which may not apply to low-contrast documents.
3. Validated only on 8B-level models; the performance on larger or smaller models remains unknown.
4. Mask supervision only focuses on text regions, providing no direct assistance for non-text visual elements (graphics, icons, etc.).
5. The two-stage training workflow increases training complexity.

## Related Work & Insights

- **KOSMOS-2.5** [Peng et al.]: Proposes a visual text grounding task, but only outputs bounding boxes instead of pixel-level masks.
- **DocOwl 1.5** [Hu et al.]: Multi-task structured document parsing, providing the DocStruct4M dataset.
- **UReader** [Ye et al.]: Promoted document understanding via the Read Full Text task.
- **Insights**: The role of auxiliary tasks in pre-training deserves deeper exploration. The strategy of "use during training, discard during inference" can be extended to other fields. Spatial alignment is a critical bottleneck for document MLLMs.

## Rating

⭐⭐⭐⭐ (4/5)

**Reason**: The VQAMask pre-training paradigm is novel, and the concept of "enhanced during training, free during inference" is highly practical. The 6-million-sample dataset and automatic annotation pipeline represent significant contributions. It achieves consistent SOTA performance across multiple document understanding tasks. The limitation lies in the somewhat bounded depth of innovation regarding the core method (auxiliary mask task), and the experimental analysis could be more exhaustive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering](../../NeurIPS2025/multimodal_vlm/benchmarking_retrievalaugmented_multimodal_generation_for_do.md)
- [\[CVPR 2025\] Relation-Rich Visual Document Generator for Visual Information Extraction](relation-rich_visual_document_generator_for_visual_information_extraction.md)
- [\[CVPR 2025\] DocoPilot: Improving Multimodal Models for Document-Level Understanding](docopilot_improving_multimodal_models_for_document-level_understanding.md)
- [\[ACL 2025\] WikiMixQA: A Multimodal Benchmark for Question Answering over Tables and Charts](../../ACL2025/multimodal_vlm/wikimixqa_a_multimodal_benchmark_for_question_answering_over_tables_and_charts.md)
- [\[ACL 2025\] MTVQA: Benchmarking Multilingual Text-Centric Visual Question Answering](../../ACL2025/multimodal_vlm/mtvqa_benchmarking_multilingual_text-centric_visual_question_answering.md)

</div>

<!-- RELATED:END -->
