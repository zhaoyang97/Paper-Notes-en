---
title: >-
  [Paper Note] Efficient Document Parsing via Parallel Token Prediction
description: >-
  [CVPR 2026][Multimodal VLM][Document Parsing] This paper proposes PTP (Parallel Token Prediction), a model-agnostic plug-and-play acceleration method that enables parallel multi-token prediction by inserting learnable register tokens into training sequences, achieving 1.6×–2.2× throughput gains on OmniDocBench without accuracy loss.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Document Parsing
  - Parallel Token Prediction
  - Register Token
  - VLM Acceleration
  - OCR
date: 2026-05-08
content_hash: e665319a03338b02
---

# Efficient Document Parsing via Parallel Token Prediction

**Conference**: CVPR 2026
**arXiv**: [2603.15206](https://arxiv.org/abs/2603.15206)
**Code**: [GitHub](https://github.com/flow3rdown/PTP-OCR)
**Area**: Multimodal VLM
**Keywords**: Document Parsing, Parallel Token Prediction, Register Token, VLM Acceleration, OCR

## TL;DR

This paper proposes PTP (Parallel Token Prediction), a model-agnostic plug-and-play acceleration method that enables parallel multi-token prediction by inserting learnable register tokens into training sequences, achieving 1.6×–2.2× throughput gains on OmniDocBench without accuracy loss.

## Background & Motivation

**Practical demands of document parsing**: Document parsing converts unstructured documents into machine-readable output, serving as a cornerstone for RAG, document analysis, and related applications, with high requirements for both speed and accuracy.

**VLMs have transformed document parsing**: End-to-end and pipeline-based VLM approaches have substantially improved parsing quality, but autoregressive (AR) decoding has become the primary speed bottleneck.

**Fundamental tension in AR decoding**: Document parsing is inherently a high-determinism transcription task rather than open-ended generation; the output is uniquely determined by the input image, making it naturally amenable to parallelization.

**Limitations of existing acceleration methods**: Output compression, visual token pruning, and parameter pruning all fail to address the AR bottleneck fundamentally.

**Constraints of non-autoregressive approaches**: CTC-based NAR models exhibit limited performance and are restricted to span-level OCR.

**Key insight**: An image can be decomposed into multiple patches that are independently recognizable, and this parallelism can be internalized within the model.

## Method

### Overall Architecture

PTP augments standard NTP training in VLMs by inserting learnable register tokens and designing corresponding training objectives and attention masks, endowing the model with parallel decoding capability, together with a high-quality data generation pipeline.

### Key Designs

#### Register Token Design

$n$ register tokens are inserted after each token in the training sequence. All register tokens share the same token ID and learnable embedding, distinguished solely by positional encoding. The $i$-th register token learns to predict the token at position $i+1$ ahead:

$$\hat{X}_a = (x_1, [r_2, r_3], x_2, [r_3, r_4], \ldots, x_l)$$

#### Attention Mask Design

Three constraints are enforced: (1) regular tokens attend only to preceding regular tokens, isolated from registers; (2) register tokens attend to all preceding regular tokens and registers within the same group; (3) registers from different groups are mutually isolated. This ensures that standard NTP training is entirely unaffected by the register tokens.

#### Positional Encoding Adjustment

The position ID of register $r_i$ equals the position of the preceding regular token $x_{i-1}$ plus 1, incremented sequentially.

### Loss & Training

$$\mathcal{L}_{\text{PTP}} = \alpha \cdot \mathcal{L}_{\text{NTP}} + (1-\alpha) \cdot \mathcal{L}_{\text{reg}}$$

$\mathcal{L}_{\text{reg}} = -\sum_i \sum_j \log P_\theta(x_{i+j+1} | X_{a,\leq i}, r_{i+j})$

### Data Generation Pipeline

200k pages of diverse documents → layout-analysis-based sub-region segmentation → multi-model collaborative annotation (strong VLM + open-source VLM + specialized models) → majority voting + LLM post-processing → CLIP deduplication + pHash deduplication → final 1.8M high-quality samples.

## Key Experimental Results

### Main Results: OmniDocBench

| Model Type | Representative Model | Overall Edit Distance↓ |
|------------|---------------------|------------------------|
| Pipeline | PP-StructureV3 | 0.0695 |
| General VLM | Gemini-2.5 Pro | 0.0734 |
| General VLM | GPT-4o | 0.2297 |
| PTP Method | PTP-1 | 1.6× speedup |
| PTP Method | PTP-2 | 2.2× speedup |

### Ablation Study

| Configuration | Throughput Gain | Accuracy Impact |
|---------------|----------------|-----------------|
| PTP-0 (NTP baseline) | 1.0× | baseline |
| PTP-1 (1 register) | 1.6× | No loss / reduced hallucination |
| PTP-2 (2 registers) | 2.2× | No loss |
| Combined with speculative decoding | 82% acceptance rate | — |

### Key Findings

- PTP not only accelerates inference but also **reduces** model hallucination, as register tokens impose additional predictive constraints.
- The method generalizes to general visual language understanding (VLU) tasks.
- PTP is orthogonal to and compatible with speculative decoding, achieving an 82% acceptance rate when combined.
- The estimated speedup ratio is: $\text{SR} \approx ((1+n) \times L_\theta) / L'_\theta$

## Highlights & Insights

- **Exceptional plug-and-play applicability**: Model-agnostic, architecture-preserving, requiring only the addition of register tokens and modification of the attention mask.
- During training, register tokens do not affect regular tokens (via mask isolation), establishing a guaranteed lower bound on NTP performance.
- The incidental effect of reduced hallucination is noteworthy—multi-token prediction provides implicit regularization constraints.
- The data pipeline is comprehensive: multi-source collection + multi-model annotation + multi-stage filtering.

## Limitations & Future Work

- At inference time, the KV cache of register tokens must be removed at each step, increasing implementation complexity.
- Register prediction accuracy for distant future tokens degrades with increasing distance.
- Training sequence length increases by a factor of $(1+n)$, raising training cost.
- Validation is currently focused on document parsing; effectiveness in open-domain generation remains to be explored.

## Related Work & Insights

- Conceptually similar to the MTP head in DeepSeek-V3 but differs in implementation: PTP uses register tokens rather than additional prediction heads.
- The register token concept is inspired by the role of high-norm outlier absorbers in ViT (DINOv2), though the application is entirely different.
- The method is orthogonal to output compression and visual token pruning, and can be used in conjunction with them.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Towards Real-World Document Parsing via Realistic Scene Synthesis and Document-Aware Training](towards_real-world_document_parsing_via_realistic_scene_synthesis_and_document-a.md)
- [\[CVPR 2026\] PaddleOCR-VL: Boosting Document Parsing Efficiency and Performance with Coarse-to-Fine Visual Processing](paddleocr-vl_boosting_document_parsing_efficiency_and_performance_with_coarse.md)
- [\[CVPR 2026\] ApET: Approximation-Error Guided Token Compression for Efficient VLMs](apet_approximation-error_guided_token_compression_for_efficient_vlms.md)
- [\[ICLR 2026\] Index-Preserving Lightweight Token Pruning for Efficient Document Understanding](../../ICLR2026/multimodal_vlm/index-preserving_lightweight_token_pruning_for_efficient_document_understanding_.md)
- [\[CVPR 2026\] Parallel In-context Learning for Large Vision Language Models](parallel_in-context_learning_for_large_vision_language_models.md)

<!-- RELATED:END -->
