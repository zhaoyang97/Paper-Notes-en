---
title: >-
  [Paper Note] DocVLM: Make Your VLM an Efficient Reader
description: >-
  [CVPR 2025][Multimodal VLM][Document Understanding] Proposes a model-agnostic OCR encoding module that compresses OCR-extracted text and layout information into 64 learned query tokens and injects them into a frozen VLM, significantly improving document understanding capabilities under extremely low visual token counts (up to +30.6 points on DocVQA) and generalizing zero-shot to multi-page documents.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Document Understanding"
  - "OCR Encoder"
  - "Token Compression"
  - "Model-Agnostic"
  - "Multi-Page Documents"
date: 2026-05-08
content_hash: 9a73e049f86c890c
---

# DocVLM: Make Your VLM an Efficient Reader

**Conference**: CVPR 2025  
**arXiv**: [2412.08746](https://arxiv.org/abs/2412.08746)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Document Understanding, OCR Encoder, Token Compression, Model-Agnostic, Multi-Page Documents

## TL;DR
Proposes a model-agnostic OCR encoding module that compresses OCR-extracted text and layout information into 64 learned query tokens and injects them into a frozen VLM, significantly improving document understanding capabilities under extremely low visual token counts (up to +30.6 points on DocVQA) and generalizing zero-shot to multi-page documents.

## Background & Motivation

**Background**: Document understanding tasks (DocVQA, InfoVQA, etc.) require models to recognize text in documents and understand their layout relations. Current mainstream VLMs (LLaVA-OneVision, InternVL2, Qwen2-VL) primarily rely on high-resolution visual tokens to "see" the text, but high resolution implies a massive amount of visual tokens.

**Limitations of Prior Work**: When the number of visual tokens is limited (due to low resolution or token budgets), VLM performance on document tasks drops drastically—InternVL2 achieves only 56.0% on DocVQA with 256 tokens, compared to 85.7% with 1280 tokens. On the other hand, simply feeding OCR text as text tokens into the LLM is effective, but 800+ OCR tokens incur massive inference overhead.

**Key Challenge**: Document understanding requires precise text and layout information, but extracting this information from pixels requires a vast number of visual tokens. While OCR can directly provide text, it lacks an efficient way to encode and compress this information to integrate it into VLMs.

**Goal**: To design an efficient OCR encoding module that injects document text and layout information into a frozen VLM using a minimal number of tokens (64), significantly enhancing document understanding without modifying the VLM's weights.

**Key Insight**: Utilize a DocFormerV2 encoder to process OCR text along with 2D position information, and compress 800+ OCR tokens into 64 using an instruction-aware learned query compression mechanism, which is then concatenated with visual tokens and fed into the VLM.

**Core Idea**: Use a learned-query-compressed OCR encoder as a plug-and-play document understanding enhancement module, achieving document understanding capabilities close to a full 800-token encoding at the cost of only 64 tokens.

## Method

### Overall Architecture
Input document image $\rightarrow$ OCR system extracts text and bounding boxes $\rightarrow$ DocFormerV2 encoder processes OCR embeddings + instruction embeddings + 64 learned queries $\rightarrow$ Keep only the query outputs as the compressed representation $\rightarrow$ Linear projection to the hidden dimension of the VLM $\rightarrow$ Concatenate with visual tokens $\rightarrow$ Feed into the frozen LLM decoder.

### Key Designs

1. **OCR Encoder (Based on DocFormerV2)**:

    - Function: Encodes OCR text and 2D layout information into continuous representations.
    - Mechanism: Employs the T5 encoder of DocFormerV2 (344M parameters) while discarding the visual branch to avoid redundancy with the VLM's visual encoder. The inputs are the embeddings of OCR text tokens and 2D bounding box position encodings, and the output is a continuous representation containing textual semantics and spatial layout information.
    - Design Motivation: This is much more efficient than directly feeding raw OCR text into the LLM. The encoder captures layout context (e.g., table structures, paragraph hierarchies). Ablation studies show that the encoder output yields a 9+ points improvement on DocVQA compared to raw OCR text.

2. **Instruction-Aware Query Compression**:

    - Function: Compresses hundreds of OCR tokens into a fixed number of compact representations (64).
    - Mechanism: Initializes $M=64$ learnable queries whose distribution matches the OCR encoder embeddings. The queries, OCR embeddings, and instruction embeddings are jointly processed through cross-attention in the encoder, and only the 64 features corresponding to the queries are retained in the encoder output. The queries not only aggregate OCR information but also adaptively focus on relevant document regions based on the instruction content.
    - Design Motivation: Similar to the Q-Former concept in BLIP-2. However, the key difference lies in the instruction-awareness: the queries "see" the question during the encoding phase, allowing them to selectively retain information relevant to the question, which is more efficient than unconditional compression.

3. **Two-Stage Training (VLM Fully Frozen)**:

    - Function: Progressively aligns the representation spaces of the OCR encoder and the VLM.
    - Mechanism: Stage I (OCR-LLM Alignment): No image input is used, forcing the model to rely solely on the OCR encoding. The encoder is frozen for the first 10K steps to train only the queries and the projection layer, then the encoder is unfrozen and trained for 130K steps. Stage II (Visual Alignment): Visual features are reintroduced, followed by 100K steps of fine-tuning to learn the complementarity of OCR and vision.
    - Design Motivation: Stage II is particularly critical for compressed representations—with 16 queries, Stage II brings a +6.2 improvement, whereas with 800 full tokens, the improvement is only +0.7. Compressed representations require more training to learn complementarity with visual features.

### Loss & Training
Standard next-token prediction loss. VLM weights are fully frozen throughout, with only the OCR encoder, learned queries, and projection layer being trained. Training data consists of various document understanding datasets.

## Key Experimental Results

### Main Results

| Method | Token Count | DocVQA | InfoVQA | MP-DocVQA | TextVQA |
|------|---------|--------|---------|-----------|---------|
| InternVL2 baseline | 256 | 56.0 | 38.4 | 51.0 | 65.7 |
| DocVLM+InternVL2 | 320 | **86.6** | **57.6** | **76.2** | **71.2** |
| Qwen2-VL baseline | 320 | 84.4 | 54.1 | 73.0 | 78.0 |
| DocVLM+Qwen2-VL | 320 | **91.2** | **61.2** | **81.7** | **79.6** |
| Qwen2-VL baseline | 576 | 91.5 | 65.3 | 82.1 | 82.3 |
| DocVLM+Qwen2-VL | 576 | **92.8** | **66.8** | **84.5** | **82.8** |

### Ablation Study

| Configuration | DocVQA | Explanation |
|------|--------|------|
| Raw OCR Text (800 tok) | 76.4 | Direct OCR text input |
| OCR Encoding (800 tok) | 89.2 | Substantial improvement after encoding |
| 64 Compressed Queries | 85.5 | Retains most gains with only 8% tokens |
| 64 Compression + Visual Features | 90.2 | Visual complementarity yields +4.7 |
| 800 Encoding + Visual Features | 91.9 | Full encoding + visual complementarity yields +2.7 |
| 16 queries (Stage I only) | 81.7 | Low query counts |
| 16 queries (+ Stage II) | 87.9 | Stage II is key (+6.2) |

### Key Findings
- **Largest gains under low token budgets**: InternVL2 jumps from 56.0 on 256 tokens to 86.6 on 320 tokens (+64 OCR tokens), representing an absolute improvement of 30.6 points, which is the most striking result.
- **64 queries is the optimal sweet spot**: It retains ~97% of the information from the full 800-token encoding, using only 8% of the tokens.
- **High complementarity between vision and OCR**: The improvement for compressed encodings (+4.7) is larger than that for full encodings (+2.7), indicating that the information lost during compression is accurately compensated for by visual features.
- **Zero-shot multi-page generalization**: Trained only on single-page data, the model achieves 86.3% ANLS on MP-DocVQA (surpassing the previous SOTA GRAM's 80.3%), verifying the generalization ability of the compressed encodings.

## Highlights & Insights
- **The design concept of a "plug-and-play OCR module" is highly valuable for engineering**: The VLM is completely frozen, keeping its original general capabilities unaffected, while document understanding is enhanced solely through an external OCR encoder. This modular design can be extended to augment other specialized capabilities (e.g., table understanding, chart understanding).
- **Extreme compression with 64 tokens**: Similar to BLIP-2's Q-Former but with the addition of instruction-awareness, making the compression smarter. This idea can be transferred to video VLMs to compress tokens of long videos.
- **Zero-shot multi-page scaling**: Trained only on a single page but capable of handling multiple pages, flexibly responding to different requirements via global and page-wise encoding strategies.

## Limitations & Future Work
- Out-of-the-box accuracy is dependent on the quality of the external OCR system, as OCR errors propagate directly to the encoder.
- The OCR encoder adds an extra 344M parameters, which, combined with the inference overhead of the 64 queries, might not be negligible in extremely real-time scenarios.
- Validated only on document understanding tasks; whether it affects general image understanding tasks is not discussed.
- Instruction-aware compression means that different questions require re-encoding, so KV cache cannot be reused as easily as with fixed visual tokens.

## Related Work & Insights
- **vs GRAM**: The previous SOTA for multi-page documents used an external text module but in a more complex manner. DocVLM outperforms it in a simpler way (86.3 vs 80.3).
- **vs TextMonkey / UReader**: These methods enhance OCR capabilities by modifying the visual encoder, which requires retraining the VLM. DocVLM keeps the VLM frozen, making it more flexible.
- **vs Direct OCR text input**: The ablation studies clearly demonstrate that the encoder outperforms raw text by 9+ points, indicating that layout encoding is the key differentiator.

## Rating
- Novelty: ⭐⭐⭐⭐ Instruction-aware query compression is an effective improvement over the Q-Former method; the plug-and-play design is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated with 3 VLM baselines, 7 benchmarks, and detailed ablation studies on compression rates and training stages.
- Writing Quality: ⭐⭐⭐⭐ Clear methodological explanations and well-organized experiments, though some tables contain redundant information.
- Value: ⭐⭐⭐⭐⭐ Holds immediate engineering value for VLM document understanding; the modular design is particularly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant](lamra_large_multimodal_model_as_your_advanced_retrieval_assistant.md)
- [\[CVPR 2025\] FastVLM: Efficient Vision Encoding for Vision Language Models](fastvlm_efficient_vision_encoding_for_vision_language_models.md)
- [\[AAAI 2026\] PatientVLM Meets DocVLM: Pre-Consultation Dialogue Between Vision-Language Models for Efficient Diagnosis](../../AAAI2026/multimodal_vlm/patientvlm_meets_docvlm_pre-consultation_dialogue_between_vision_language_models.md)
- [\[CVPR 2026\] Is your VLM Sky-Ready? A Comprehensive Spatial Intelligence Benchmark for UAV Navigation](../../CVPR2026/multimodal_vlm/is_your_vlm_sky-ready_a_comprehensive_spatial_intelligence_benchmark_for_uav_nav.md)
- [\[CVPR 2025\] Your Large Vision-Language Model Only Needs a Few Attention Heads for Visual Grounding](your_large_vision-language_model_only_needs_a_few_attention_heads_for_visual_gro.md)

</div>

<!-- RELATED:END -->
