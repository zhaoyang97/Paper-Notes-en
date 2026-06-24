---
title: >-
  [Paper Note] mPLUG-DocOwl2: High-resolution Compressing for OCR-free Multi-page Document Understanding
description: >-
  [ACL 2025][Model Compression][Document Understanding] This paper proposes a layout-aware High-resolution DocCompressor module, which employs global low-resolution visual features as queries and sub-image features as keys/values for grouped cross-attention. This compresses each high-resolution document image from thousands of tokens down to 324 tokens. Combined with a three-stage training framework, it achieves SOTA performance in multi-page document understanding while reduci…
tags:
  - "ACL 2025"
  - "Model Compression"
  - "Document Understanding"
  - "Visual Token Compression"
  - "Layout-aware"
  - "Multi-page Document"
  - "OCR-free"
date: 2026-05-08
content_hash: 4b54c6768bd371e4
---

# mPLUG-DocOwl2: High-resolution Compressing for OCR-free Multi-page Document Understanding

**Conference**: ACL 2025  
**arXiv**: [2409.03420](https://arxiv.org/abs/2409.03420)  
**Code**: [GitHub](https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/DocOwl2)  
**Area**: Model Compression / Multimodal VLM  
**Keywords**: Document Understanding, Visual Token Compression, Layout-aware, Multi-page Document, OCR-free

## TL;DR

This paper proposes a layout-aware High-resolution DocCompressor module, which employs global low-resolution visual features as queries and sub-image features as keys/values for grouped cross-attention. This compresses each high-resolution document image from thousands of tokens down to 324 tokens. Combined with a three-stage training framework, it achieves SOTA performance in multi-page document understanding while reducing First Token Latency by over 50%.

## Background & Motivation

**Background**: OCR-free document understanding has made significant progress (e.g., InternVL 2 achieving 91.6% on DocVQA) by cropping high-resolution document images into multiple low-resolution sub-images to capture fine-grained textual details. However, this strategy comes at the cost of requiring thousands of visual tokens per document image (averaging over 3k tokens for InternVL 2), leading to high GPU memory consumption and slow inference speeds.

**Limitations of Prior Work**: The massive amount of visual tokens makes multi-page document understanding nearly impractical—a 10-page document requires over 30k tokens, which far exceeds the context window of most LLMs. Existing compression methods have notable drawbacks: (1) compressing each sub-image independently still yields a large token count (e.g., TokenPacker still requires over 1.8k tokens); (2) learnable queries (e.g., Resampler/Q-former) lack layout priors, making it difficult to efficiently compress dense textual information in documents; (3) methods based on token similarity selection (e.g., TextMonkey) may overlook certain regions.

**Key Challenge**: The density of textual information in document images is much higher than that in natural images, causing simple visual token compression methods to suffer from severe textual information loss; however, avoiding compression altogether precludes joint understanding of multi-page or multi-image documents.

**Goal**: How to drastically compress high-resolution document images while fully preserving layout and textual information?

**Key Insight**: Two key observations: (1) the NLP field has demonstrated that text paragraphs can be compressed into a few vectors while retaining most of the semantic meaning; (2) visual tokens aligned by the vision-to-text module are essentially "text tokens" that encode textual information from different regions of the image, allowing them to be treated similarly to text compression. The global low-resolution image naturally encodes the overall layout information, which can serve as semantic guidance for compression.

**Core Idea**: Utilizing global low-resolution features as queries to guide the cross-attention compression of high-resolution sub-image features, utilizing spatial correspondence for grouped attention, and performing compression after vision-to-text alignment to better preserve text semantics.

## Method

### Overall Architecture

Encoding workflow of DocOwl2: High-resolution image $\to$ Shape-adaptive Cropping into $R \times C$ sub-images + a global image $\to$ ViT independently encodes each sub-image and global image $\to$ H-Reducer (convolution + FC) performs vision-to-text alignment and reduces the token count of each sub-image to 1/4 $\to$ **High-resolution DocCompressor** further compresses the tokens to match the size of the global feature map (324 tokens) $\to$ The compressed tokens of multiple images are concatenated and fed into the LLM. The model undergoes a three-stage training process: Single-image Pretraining $\to$ Multi-image Continue-Pretraining $\to$ Multi-task Finetuning.

### Key Designs

1. **Layout-aware Grouped Cross-Attention Compression**:

    - **Function**: Compress each high-resolution document image from $(R \times C + 1) \times h \times w/4$ tokens to $h \times w/4$ tokens (e.g., from 2560 down to 324).
    - **Mechanism**: Each token $\hat{v}_{ij}^g$ in the global feature map $\hat{V}^g$ acts as a query, and the corresponding $R \times C$ high-resolution sub-image tokens $\hat{v}_{ij}^s$ serve as keys/values for grouped cross-attention: $\bar{v}_{ij} = \text{softmax}(\frac{W^q \hat{v}_{ij}^g \cdot W^k \hat{v}_{ij}^s}{\sqrt{d_k}}) W^v \hat{v}_{ij}^s + \hat{v}_{ij}^g$ (including residual connection). The spatial correspondence is naturally determined by the spatial mapping of image cropping.
    - **Design Motivation**: Unlike allowing each query to attend to all high-resolution tokens (which is computationally expensive and makes information compression harder), grouped attention takes advantage of the natural spatial correspondence between the global image and the sub-images. Each query only needs to focus on the $R \times C$ corresponding tokens of the same physical region, making it easier to aggregate semantic information based on layout areas.

2. **Compression Position: Post Vision-to-Text Alignment**:

    - **Function**: Place the DocCompressor after the H-Reducer (V2T module) rather than between the ViT and H-Reducer.
    - **Mechanism**: The model first aggregates 4 horizontal features via the convolutional layer of the H-Reducer and aligns them with the LLM feature space via the FC layer so that the visual features are already encoded as "text-like tokens" before compression.
    - **Design Motivation**: Ablation studies (r4 vs. r3) demonstrate that compressing after V2T alignment outperforms compressing the raw ViT output. The intuition is that compressing already aligned features is analogous to text summarization in NLP (operating within the semantic space), whereas directly compressing raw visual features results in much higher loss of textual details.

3. **Three-Stage Training Framework**:

    - **Function**: Step-by-step endowment of single-image understanding, multi-image correlation, and multi-task generalization capabilities to the model.
    - **Mechanism**: **Stage 1** (Single-image Pretraining): Learning structural parsing of documents, tables, and charts on DocStruct4M to ensure compressed tokens encode sufficient information. **Stage 2** (Multi-image Continue-Pretraining): Learning two symmetric tasks—multi-page text parsing and text locating—on MP-DocStruct1M. **Stage 3** (Multi-task Finetuning): Training on a mixture of single-image (DocDownstream-1.0, DocReason25K) and multi-image (MP-DocVQA, DUDE, NewsVideoQA, MP-DocReason51K) instruction-tuning datasets.
    - **Design Motivation**: The two symmetric tasks in Stage 2 (parsing text given page numbers + locating page numbers given text) are foundational capabilities for multi-page understanding—the model must establish bidirectional mapping between page numbers and content across multiple images. Ablation studies (r3 vs. r2) demonstrate that this stage is crucial for understanding documents containing over 10 pages.

### Loss & Training

Initialized based on mPLUG-Owl2. Stage 1 trained for 12k steps (batch 1024, lr 1e-4), freezing the main LLM parameters and only tuning MAM. Stage 2 trained for 2.4k steps (batch 1024, lr 2e-5), freezing the ViT. Stage 3 trained for 9k steps (batch 256, lr 2e-5), with all parameters trainable except the ViT. DocCompressor contains only 2 cross-attention layers.

## Key Experimental Results

### Main Results

Single-page document understanding (visual tokens < 1k groups):

| Model | TokenV | DocVQA | ChartQA | TextVQA | InfoVQA |
|------|--------|--------|---------|---------|---------|
| TextMonkey | 768 | 73.0 | 66.9 | 65.9 | 28.6 |
| UReader | ~841 | 65.4 | 59.3 | 57.6 | 42.2 |
| **DocOwl2** | **324** | **80.7** | **70.0** | **66.7** | **46.4** |

Multi-page document understanding:

| Model | TokenV | MP-DocVQA (ANLS) | DUDE (ANLS) | FTL(s)↓ |
|------|--------|------------------|-------------|---------|
| LongVA-7B | ~2029 | 60.80 | 38.37 | 2.13 |
| Idefics3-8B | ~838 | 67.15 | 38.65 | 2.26 |
| **DocOwl2-8B** | **324** | **69.42** | **46.77** | **0.95** |

### Ablation Study

Compression architecture comparison (same training process, evaluated on single-page):

| Method | Compression Method | TokenV | DocVQA | WTQ | ChartQA |
|------|---------|--------|--------|-----|---------|
| Resampler | Learnable query | 256 | 69.0 | 29.4 | 66.6 |
| CAbstractor | Adaptive average pooling | 256 | 73.0 | 32.6 | 67.6 |
| **DocCompressor** | Grouped cross-attention | 256 | **76.1** | **35.1** | **69.2** |
| DocCompressor (after ViT) | Grouped cross-attention | 256 | 75.7 | 33.3 | 68.7 |
| DocCompressor (Full attention) | Global cross-attention | 256 | 74.4 | 33.7 | 68.2 |
| DocCompressor (Average) | Grouped average pooling | 256 | 74.6 | 31.9 | 68.2 |

### Key Findings

- **10x Compression with Only 2% Performance Drop**: Compared to DocOwl 1.5 (~1698 tokens), DocOwl2 (324 tokens) retains 98% of the performance on DocVQA while reducing First Token Latency by 55% (0.26s vs. 0.58s).
- **Grouped Attention Outperforms Global Attention**: Grouped attention using spatial correspondence (r3) improves performance by 2 percentage points compared to global attention (r5), with lower computational overhead.
- **Post-V2T Compression is Better**: Placing compression after the H-Reducer (r3) performs 0.4-1.8 percentage points better than placing it before (r4).
- **Multi-Image Training Stage Contributes Significantly**: Incorporating Stage 2 boosts the accuracy on 10+ page documents from 5.8% to 37.9% (r4 vs. r1).
- **Compared with SOTA utilizing >1k tokens**: Using <20% of the tokens, it achieves over 80% of the performance of InternVL 2 / IXC 2.5 on 7 out of 10 benchmarks.

## Highlights & Insights

- **Precise Intuition Behind Layout-Aware Compression**: Text within the same layout region of a document is semantically coherent. Using the spatial structure of the global feature map to guide compression is highly natural. This design aligns much better with the physical layout of documents than learnable queries or token selection methods.
- **Insight of "Visual Token = Text Token"**: Once visual tokens are aligned via V2T, they enter the textual semantic space; thus, compressing them is equivalent to text summarization rather than lossy encoding of visual information. This insight clarifies why compression after V2T yields superior results.
- **Symmetric Task Design in Three-Stage Training**: Multi-page text parsing (page number $\to$ text) and text locating (text $\to$ page number) are inverse tasks. Training them concurrently establishes a stronger bidirectional association between page numbers and content.

## Limitations & Future Work

- **Outdated Base Model**: Based on LLaMA-1 and mPLUG-Owl2, which limits the performance ceiling. Utilizing stronger base models (e.g., LLaMA-3 / Qwen2) may yield significant improvements.
- **Fixed Compression Ratio**: All images are uniformly compressed to 324 tokens despite large differences in information density across different documents. Adaptive compression ratios represent a promising research direction.
- **Remaining SOTA Performance Gap**: On tasks requiring fine-grained OCR (e.g., DocVQA), a performance gap of over 10 percentage points still exists compared to InternVL 2 (91.6%).
- **Evaluation Limited to Document Images**: Compression performance on natural images has not been sufficiently verified.

## Related Work & Insights

- **vs. TokenPacker**: TokenPacker uses downsampled features as queries to compress each sub-image, but still requires concatenating all sub-image outputs (1.8k tokens). DocOwl2 leverages global features to guide the compression more thoroughly (324 tokens).
- **vs. TextMonkey**: TextMonkey selects valuable tokens based on token similarity as compression guidance, which may miss certain regions. DocOwl2's global feature map covers all regions.
- **vs. Mini-Gemini**: Mini-Gemini requires an additional high-resolution encoder; DocOwl2 reuses the same ViT to encode both global and sub-images, making it more elegant.
- **Inspirations**: This paradigm of "utilizing low-resolution global features to guide high-resolution detail compression" can be mapped to video understanding (e.g., keyframe-guided neighboring frame compression) and ultra-long document processing.

## Rating

- Novelty: ⭐⭐⭐⭐ The layout-aware compression design is natural and effective, and compressing after V2T offers a deep insight.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on 10 single-page and 3 multi-page benchmarks, featuring comprehensive ablation studies, latency analysis, and qualitative examples.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, systematic ablation design, and well-articulated motivations.
- Value: ⭐⭐⭐⭐ Provides a practical and reproducible solution addressing the token efficiency problem in multi-page document understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Understanding Multi-layered Transmission Matrices](../../CVPR2025/model_compression/understanding_multi-layered_transmission_matrices.md)
- [\[ACL 2025\] TaDA: Training-free recipe for Decoding with Adaptive KV Cache Compression and Mean-centering](tada_training-free_recipe_for_decoding_with_adaptive_kv_cache_compression_and_me.md)
- [\[CVPR 2026\] High Resolution Neural Video Coding with Bi-directional Confidence-Guided Reference Information Modeling](../../CVPR2026/model_compression/high_resolution_neural_video_coding_with_bi-directional_confidence-guided_refere.md)
- [\[ICML 2026\] Hierarchical Image Tokenization for Multi-Scale Image Super Resolution](../../ICML2026/model_compression/hierarchical_image_tokenization_for_multi-scale_image_super_resolution.md)
- [\[ICCV 2025\] Variance-Based Pruning for Accelerating and Compressing Trained Networks](../../ICCV2025/model_compression/variance-based_pruning_for_accelerating_and_compressing_trained_networks.md)

</div>

<!-- RELATED:END -->
