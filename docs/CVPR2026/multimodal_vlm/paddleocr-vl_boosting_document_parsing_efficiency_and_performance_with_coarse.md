---
title: >-
  [Paper Note] PaddleOCR-VL: Boosting Document Parsing Efficiency and Performance with Coarse-to-Fine Visual Processing
description: >-
  [CVPR 2026][Multimodal VLM][Document Parsing] PaddleOCR-VL introduces a coarse-to-fine document parsing framework that first employs a lightweight VRFM module to detect valid regions and predict reading order, then applies a compact 0.9B VLM for fine-grained recognition, achieving state-of-the-art document parsing performance with minimal visual tokens and parameters.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Document Parsing
  - Coarse-to-Fine
  - Visual Redundancy
  - OCR
  - Vision-Language Model
date: 2026-05-08
content_hash: 0c38f65b3e829bd0
---

# PaddleOCR-VL: Boosting Document Parsing Efficiency and Performance with Coarse-to-Fine Visual Processing

**Conference**: CVPR 2026
**arXiv**: [2603.24326](https://arxiv.org/abs/2603.24326)
**Code**: [https://github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
**Area**: Multimodal VLM / Document Understanding
**Keywords**: Document Parsing, Coarse-to-Fine, Visual Redundancy, OCR, Vision-Language Model

## TL;DR

PaddleOCR-VL introduces a coarse-to-fine document parsing framework that first employs a lightweight VRFM module to detect valid regions and predict reading order, then applies a compact 0.9B VLM for fine-grained recognition, achieving state-of-the-art document parsing performance with minimal visual tokens and parameters.

## Background & Motivation

Document parsing requires recognizing elements such as text, formulas, and tables while determining the correct reading order. Existing approaches fall into three categories: pipeline methods (prone to error propagation), general-purpose VLMs (hallucination-prone and computationally expensive), and specialized VLMs (large parameter counts or coordinate drift in end-to-end settings).

**Core Problem**: High-resolution input is critical for document parsing, yet the computational cost of visual encoding grows quadratically with resolution. Visual information in document images is highly non-uniform — valid regions occupy only approximately 39% of a typical slide and approximately 60% even in information-dense newspapers.

**Starting Point**: Given the substantial redundant background in documents, the paper proposes first rapidly localizing valid regions and then performing fine-grained recognition exclusively on those regions. This decoupled design allows each module to specialize in its own task while significantly reducing the number of visual tokens fed into the VLM.

## Method

### Overall Architecture

A two-stage pipeline: the coarse stage employs VRFM to detect document elements, classify them, and predict reading order → valid regions are cropped → the fine stage uses PaddleOCR-VL-0.9B for fine-grained recognition of each region → results are reassembled into a structured document according to reading order.

### Key Designs

1. **Valid Region Focusing Module (VRFM)**:

    - **Function**: Efficiently localizes valid visual elements in documents and predicts reading order.
    - **Mechanism**: Built upon the RT-DETR detector for layout element detection and classification, extended with a Pointer Network to predict an $N \times N$ reading order matrix. Training proceeds in two stages: RT-DETR is trained first (100 epochs), followed by freezing the backbone and training only the Pointer Network (200 epochs, using noise-robust Generalized Cross Entropy Loss).
    - **Design Motivation**: Unifies region detection and reading order prediction within a lightweight framework, avoiding the propagation of redundant background content into the downstream VLM.

2. **PaddleOCR-VL-0.9B Element Recognition Model**:

    - **Function**: Performs fine-grained multi-type recognition (text / table / formula / figure) on cropped valid regions.
    - **Mechanism**: Adopts a NaViT-style visual encoder (initialized from Keye-VL) + a 2-layer MLP projector + ERNIE-4.5-0.3B language model (with 3D-RoPE). A key characteristic is native dynamic-resolution processing, which avoids distortion and hallucination introduced by fixed-resolution or tiling strategies.
    - **Design Motivation**: The 0.9B parameter count is extremely compact; however, because the model processes only cropped valid regions rather than full pages, information density is higher, yielding superior recognition performance.

3. **High-Quality Data Pipeline**:

    - **Function**: Constructs a multi-source training dataset of 30M+ samples.
    - **Mechanism**: Four data sources (open-source + synthetic + web-crawled + internal) combined with automatic annotation (PP-StructureV3 for initial labeling → VLM refinement → hallucination filtering) and hard-example mining (fine-grained evaluation to identify weaknesses → targeted synthetic data generation).
    - **Design Motivation**: Data quality is a critical performance factor; hard-example mining establishes a closed loop of "evaluation → synthesis → training."

### Loss & Training

The VLM undergoes two-stage training: Stage 1 pre-trains on 29M samples for alignment (1 epoch, LR 5e-5→5e-6); Stage 2 performs instruction fine-tuning on 2.7M samples (2 epochs, higher resolution limit of 2048, LR 5e-6→5e-7).

## Key Experimental Results

### Main Results

| Method | Params | Visual Tokens | Overall↑ | Text↓ | Formula↑ | Table↑ | ReadOrder↓ |
|--------|--------|---------------|----------|-------|----------|--------|------------|
| Gemini-2.5 Pro | - | - | 88.03 | 0.075 | 85.82 | 85.71 | 0.097 |
| Qwen2.5-VL-72B | 72B | 5626 | 87.02 | 0.094 | 88.27 | 82.15 | 0.102 |
| PaddleOCR-VL | **0.9B** | **Fewest** | **91.32** | **0.046** | **90.98** | **85.77** | **0.050** |

On OmniDocBench v1.5, the proposed method surpasses all baselines with the fewest parameters and visual tokens.

### Ablation Study

| Configuration | Overall | Notes |
|---------------|---------|-------|
| End-to-end VLM (w/o VRFM) | Lower | Processing full pages is inefficient and yields inferior performance |
| VRFM + PaddleOCR-VL-0.9B | 91.32 | Full coarse-to-fine strategy |
| w/o Hard-Example Mining | Noticeable drop | The data quality closed loop is critical |

### Key Findings

- The 0.9B model outperforms 72B/241B general-purpose VLMs across all four core metrics, demonstrating that "what the model sees" matters more than "how large the model is."
- The system supports 109 languages and remains robust on challenging scenarios such as handwritten and historical documents.
- Inference latency and throughput are substantially better than competing approaches.

## Highlights & Insights

- **Small but precise design philosophy**: By cropping to valid regions and eliminating irrelevant pixels from VLM input, a 0.9B model outperforms a 72B one — underscoring that input quality outweighs model scale.
- **Engineering value of coarse-fine decoupling**: VRFM and the VLM can be independently optimized and upgraded, reducing maintenance overhead.
- **Lessons from the data closed loop**: The "evaluation → hard-example mining → synthetic data" cycle offers a broadly applicable training paradigm for domain-specific VLMs.

## Limitations & Future Work

- Detection accuracy of VRFM has a cascading effect on downstream recognition — missed regions result in direct information loss.
- Reading order prediction relies on a Pointer Network, which may lack robustness for highly complex cross-page layouts.
- The two-stage design introduces additional latency; although overall throughput is superior, the system is less elegant than a truly end-to-end approach.
- Future work may explore joint training or end-to-end optimization of VRFM and the VLM.

## Related Work & Insights

- **vs. MinerU/Dolphin**: End-to-end VLM approaches carry large parameter counts and are prone to reading order confusion; PaddleOCR-VL avoids these issues through decoupling.
- **vs. DeepSeek-OCR**: DeepSeek-OCR employs unified visual token compression, but coarse-grained compression sacrifices layout precision; PaddleOCR-VL's selective focusing is more accurate.
- **vs. PP-StructureV3**: A conventional pipeline approach; PaddleOCR-VL introduces VLM-based recognition to achieve stronger semantic understanding.

## Rating

- Novelty: ⭐⭐⭐⭐ The coarse-to-fine idea is not new, but its specific instantiation for the document domain is well-crafted.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive multi-benchmark comparisons covering diverse document types.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed quantitative evidence.
- Value: ⭐⭐⭐⭐⭐ Open-source with outstanding performance; highly valuable for real-world industrial applications.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Efficient Document Parsing via Parallel Token Prediction](efficient_document_parsing_via_parallel_token_prediction.md)
- [\[CVPR 2026\] Towards Real-World Document Parsing via Realistic Scene Synthesis and Document-Aware Training](towards_real-world_document_parsing_via_realistic_scene_synthesis_and_document-a.md)
- [\[CVPR 2026\] Multimodal OCR: Parse Anything from Documents](multimodal_ocr_parse_anything_from_documents.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)

<!-- RELATED:END -->
