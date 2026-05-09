---
title: >-
  [Paper Note] A Token-level Text Image Foundation Model for Document Understanding (TokenFD/TokenVL)
description: >-
  [ICCV 2025][Self-Supervised Learning][token-level alignment] This paper proposes TokenFD, the first token-level text image foundation model, pre-trained on 20 million images and 1.8 billion BPE token-mask pairs via token-level vision-language alignment to achieve image-as-text semantic understanding. Built upon TokenFD, TokenVL is introduced as a document understanding MLLM, achieving a score of 860 on OCRBench (highest among 8B-class models) and an average improvement of 8.8% across ten VQA benchmarks including DocVQA.
tags:
  - ICCV 2025
  - Self-Supervised Learning
  - token-level alignment
  - visual foundation model
  - document understanding
  - OCR-free
  - multimodal large language model
date: 2026-05-08
content_hash: 99478d7589868884
---

# A Token-level Text Image Foundation Model for Document Understanding (TokenFD/TokenVL)

**Conference**: ICCV 2025
**arXiv**: [2503.02304](https://arxiv.org/abs/2503.02304)
**Code**: [Token-family/TokenFD](https://github.com/Token-family/TokenFD)
**Area**: Self-supervised Learning / Representation Learning
**Keywords**: token-level alignment, visual foundation model, document understanding, OCR-free, multimodal large language model

## TL;DR

This paper proposes TokenFD, the first token-level text image foundation model, pre-trained on 20 million images and 1.8 billion BPE token-mask pairs via token-level vision-language alignment to achieve image-as-text semantic understanding. Built upon TokenFD, TokenVL is introduced as a document understanding MLLM, achieving a score of 860 on OCRBench (highest among 8B-class models) and an average improvement of 8.8% across ten VQA benchmarks including DocVQA.

## Background & Motivation

1. **State of the Field**: General-purpose visual foundation models (VFMs) such as CLIP, DINO, and SAM are widely adopted as visual encoders in multimodal large language models. However, these models are trained under image-level (CLIP/DINO) or pixel-level (SAM) supervision.

2. **Limitations of Prior Work**: For document images containing dense small text, image-level VFMs fail to perceive fine-grained textual content precisely, leading to fundamental perceptual errors in downstream OCR-related tasks. Some methods attempt to incorporate SAM as an additional high-resolution encoder, but the dual-VFM combination doubles the token count, incurring high cost and reduced flexibility.

3. **Root Cause**: No token-granularity fine-grained text image foundation model currently exists. A critical gap lies between image-level and pixel-level supervision—namely, token-level alignment, which refers to the precise mapping of each BPE subword to its corresponding spatial region in the image.

4. **Paper Goals**: (1) Construct the first token-level image-text dataset; (2) Train the first token-level VFM; (3) Apply it to the construction of a document understanding MLLM.

5. **Starting Point**: BPE tokenizers are used to decompose text into subwords, and pixel-level masks are constructed for each subword to achieve token-granularity vision-language alignment—substantially finer than CLIP's image-level alignment and semantically richer than SAM's pixel-level segmentation.

6. **Core Idea**: Align visual features and language embeddings at the token (BPE subword) granularity, endowing the VFM with "image-as-text" semantic capability.

## Method

### Overall Architecture

The framework comprises three layers: (1) **TokenIT** dataset—20 million images with 1.8 billion token-mask pairs; (2) **TokenFD** foundation model—ViT backbone with deconvolution upsampling and token-level contrastive learning for image-as-text alignment; (3) **TokenVL** MLLM—TokenFD as the visual encoder combined with InternLM as the LLM, trained in two stages (token-alignment pre-training followed by instruction fine-tuning).

### Key Designs

1. **TokenIT Dataset Construction**:

    - Function: Construct the first token-level image-text dataset.
    - Mechanism: A four-step pipeline—① text image segmentation (fine-tuned SAM for natural scenes; unsupervised clustering for documents); ② text recognition (transcription via state-of-the-art OCR); ③ BPE tokenizer subword decomposition; ④ merging character-level masks into token-level masks. Each sample contains the original image, mask image, and a JSON file recording BPE token information.
    - Design Motivation: Coverage spans natural scenes, documents, tables, charts, code, GUI, and more. Three rounds of human verification over four months ensure data quality.

2. **TokenFD Pre-training**:

    - Function: Achieve token-level vision-language alignment.
    - Mechanism: Input images are encoded by a ViT backbone; two deconvolution layers upsample features by 4× to higher resolution, followed by a linear projection to match the language embedding dimension. For each BPE token-mask pair, token-level visual features $\mathbf{t}_i$ are extracted via mean pooling over the masked region, while language embeddings $\mathbf{e}_i$ are obtained from a simple token embedding layer (no complex text encoder required). Three loss functions are jointly optimized: distance loss $\mathcal{L}_{dis}$ (L1 distance), similarity loss $\mathcal{L}_{sim}$ (cosine similarity), and sigmoid contrastive loss $\mathcal{L}_{sig}$ (SigLIP-style).
    - Design Motivation: Unlike CLIP, which requires a sophisticated text encoder, TokenFD directly aligns using a token embedding layer—since the granularity is already at the BPE subword level, contextual encoding is unnecessary.

3. **TokenVL (MLLM)**:

    - Function: Construct a document understanding MLLM.
    - Mechanism: Two-stage training. **Stage 1 — LLM-guided Token Alignment**: autoregressive VQA training (implicit alignment) combined with a token alignment branch (explicit spatial alignment, extracting vision-language features from intermediate LLM layers for token-level alignment). **Stage 2 — SFT**: the token alignment branch is removed to eliminate inference overhead, and full-parameter fine-tuning is performed on VQA data. A token abstractor is also designed to adaptively compress visual features within each window using learnable tokens.
    - Design Motivation: During training, the token alignment branch compels the LLM to rely more on image content rather than semantic context inference; it is removed at inference time without additional overhead.

### Loss & Training

- TokenFD pre-training: AdamW with cosine schedule, base lr = 5e-4, trained for 2 epochs on TokenIT using 64 H800 GPUs.
- TokenVL Stage 1: InternLM frozen; TokenFD and token abstractor trained, lr = 2e-4, 1 epoch.
- TokenVL Stage 2: All parameters trainable, lr = 1e-5.

## Key Experimental Results

### Main Results (TokenFD Zero-shot / Linear Probing)

| Task | Method | Zero-shot avg | Linear Probing avg |
|------|--------|---------------|--------------------|
| Text Segmentation | CLIP-L-1024px | 15.81 | - |
| | SAM-H | - | 34.51 |
| | InternViT2.5 | - | 42.21 |
| | **TokenFD** | **34.59** | **48.77** |
| Text Retrieval | CLIP-L | - | 3.62 |
| | InternViT2.5 | - | 13.29 |
| | **TokenFD** | - | **63.62** |

### TokenVL Document Understanding

| Model | Params | DocVQA | InfoVQA | ChartQA | TextVQA | OCRBench |
|-------|--------|--------|---------|---------|---------|----------|
| InternVL2.5-2B | 2B | 88.7 | 60.9 | 79.2 | 74.3 | 804 |
| **TokenVL-2B** | **2B** | **89.9** | **61.0** | **81.1** | **76.4** | **821** |
| InternVL2.5-8B | 8B | 93.0 | 77.6 | 84.8 | 79.1 | 822 |
| **TokenVL-8B** | **8B** | **94.2** | **76.5** | **86.6** | **79.9** | **860** |

### Ablation Study

| Configuration | DocVQA | ChartQA | Note |
|---------------|--------|---------|------|
| w/o token abstractor, w/o TA | 93.1 | 86.5 | Baseline |
| w/ token abstractor, w/o TA | 93.8 | 86.5 | + token abstractor |
| w/ token abstractor, w/ TA | **94.2** | **86.6** | Full model |

### Key Findings

- TokenFD surpasses CLIP by 18.78% on zero-shot text segmentation and exceeds InternViT2.5 by over 50% on text retrieval, demonstrating the overwhelming advantage of token-level alignment on text-related tasks.
- TokenVL-8B achieves an OCRBench score of 860, outperforming InternVL2.5 by 38 points and TextHawk2 by 76 points, confirming that a token-level VFM substantially enhances document understanding capability.
- The token alignment branch significantly reduces edit distance in full-image text recognition, validating the effectiveness of explicit spatial alignment.

## Highlights & Insights

- **Completing the VFM Granularity Spectrum**: From CLIP (image-level) → SAM (pixel-level) → TokenFD (token-level), a complete granularity spectrum of visual foundation models is established. The token level precisely fills the critical gap between semantics and spatial precision.
- **Simple yet Effective Language Encoding**: No complex text encoder akin to CLIP is needed; a simple token embedding layer suffices—since the granularity is already at the BPE subword level, contextual understanding is unnecessary.
- **The Importance of Data Engineering**: Constructing 1.8 billion high-quality token-mask pairs required four months and three rounds of human review, underscoring the critical role of data quality in foundation model research.

## Limitations & Future Work

- Pre-training requires 64 H800 GPUs, imposing a high computational resource threshold.
- Only 2B and 8B variants of TokenVL are provided; larger-scale models remain unexplored.
- The quality of token-level masks depends on the accuracy of upstream OCR and segmentation models.
- Evaluation is limited to document/OCR-related tasks; the impact on general visual understanding has not been explored.

## Related Work & Insights

- **vs. CLIP**: CLIP's image-level alignment lacks precision for dense-text scenarios; TokenFD's token-level alignment achieves zero-shot text segmentation performance more than double that of CLIP.
- **vs. InternVL2.5**: InternVL2.5 employs InternViT as a general-purpose VFM, whereas TokenFD is specifically designed for text images, consistently outperforming InternVL2.5 on document tasks.
- **vs. SAM**: SAM provides pixel-level segmentation but lacks semantic capability; TokenFD achieves both spatial precision and semantic alignment.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First token-level text image foundation model, filling a critical gap in the VFM granularity spectrum.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage of text segmentation, retrieval, VQA, and OCRBench, with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-organized presentation of the three-layer framework.
- Value: ⭐⭐⭐⭐⭐ Significant advancement for the document understanding field; dataset, models, and code are fully open-sourced.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](../../NeurIPS2025/self_supervised/tabstar_a_tabular_foundation_model_for_tabular_data_with_text_fields.md)
- [\[CVPR 2026\] D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping](../../CVPR2026/self_supervised/d2dewarp_dual_dimensions_geometric_representation_learning_based_document_image_.md)
- [\[CVPR 2026\] Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval](../../CVPR2026/self_supervised/text-phase_synergy_network_with_dual_priors_for_unsupervised_cross-domain_image_.md)
- [\[NeurIPS 2025\] Manifolds and Modules: How Function Develops in a Neural Foundation Model](../../NeurIPS2025/self_supervised/manifolds_and_modules_how_function_develops_in_a_neural_foundation_model.md)
- [\[NeurIPS 2025\] Understanding Ice Crystal Habit Diversity with Self-Supervised Learning](../../NeurIPS2025/self_supervised/understanding_ice_crystal_habit_diversity_with_self-supervised_learning.md)

<!-- RELATED:END -->
