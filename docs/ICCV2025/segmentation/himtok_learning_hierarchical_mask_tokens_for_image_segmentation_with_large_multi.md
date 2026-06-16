---
title: >-
  [Paper Note] HiMTok: Learning Hierarchical Mask Tokens for Image Segmentation with Large Multimodal Model
description: >-
  [ICCV 2025][Segmentation][Hierarchical Mask Tokenization] This paper proposes HiMTok (Hierarchical Mask Tokenizer), which represents segmentation masks as up to 32 coarse-to-fine discrete tokens…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Hierarchical Mask Tokenization"
  - "Large Multimodal Model"
  - "Vector Quantization"
  - "Visual Grounding"
date: 2026-05-08
content_hash: fdb07da5caef5bed
---

# HiMTok: Learning Hierarchical Mask Tokens for Image Segmentation with Large Multimodal Model

**Conference**: ICCV 2025
**arXiv**: [2503.13026](https://arxiv.org/abs/2503.13026)  
**Code**: [github.com/yayafengzi/LMM-HiMTok](https://github.com/yayafengzi/LMM-HiMTok)  
**Area**: Image Segmentation / Large Multimodal Models
**Keywords**: Hierarchical Mask Tokenization, Large Multimodal Model, Segmentation, Vector Quantization, Visual Grounding

## TL;DR

This paper proposes HiMTok (Hierarchical Mask Tokenizer), which represents segmentation masks as up to 32 coarse-to-fine discrete tokens, enabling LMMs to directly generate segmentation results in the same manner as text generation — without any additional image-conditioned mask decoder — achieving state-of-the-art performance across multiple segmentation benchmarks.

## Background & Motivation

Existing LMM-driven segmentation approaches fall into three main paradigms, each with notable shortcomings:

**Boundary point sequences** (e.g., PolyFormer, VistaLLM): Masks are represented as sequences of polygon vertices, but a limited number of vertices cannot adequately capture complex shapes or multi-region objects.

**Hidden states + mask decoder** (e.g., LISA, PixelLM, PSALM): The LMM outputs hidden states of special tokens, which are then decoded by an external SAM/Mask2Former module. This introduces three limitations:
   - The LLM does not sufficiently learn precise spatial localization.
   - Mask representations are inconsistent between input and output (special tokens serve merely as identifiers, discarding the corresponding hidden state information).
   - The architecture is complex, as the mask decoder requires re-access to the original image.

**Image generation approaches** (VQ-GAN quantized into 2D tokens): These are overly redundant and fail to achieve competitive performance.

**Core Problem**: Can LMMs natively acquire segmentation capability — generating masks as they generate text — without relying on external segmentation models?

## Method

### Overall Architecture

The HiMTok system consists of three components:
- **Mask Tokenizer (MT)**: Encodes segmentation masks into a set of 1D latent tokens.
- **Vector Quantization layer (VQ)**: Discretizes the latent tokens.
- **Mask Detokenizer (MD)**: Reconstructs segmentation masks from the discrete tokens.

A three-stage training scheme progressively integrates segmentation capability into the LMM:
- Stage 1: Train HiMTok (unimodal mask reconstruction).
- Stage 2: Joint training of LMM + HiMTok (low-resolution images; aligns vision–language and mask tokens).
- Stage 3: Train LMM only (high-resolution images; fine-tuning).

### Key Designs

1. **Hierarchical Mask Tokenization**: Inspired by TiTok, masks are compressed into 32 1D discrete tokens. The key innovation is a **causal attention mechanism** in which each latent token is conditioned on the input mask patches and all preceding tokens, enforcing a coarse-to-fine hierarchical structure:

$$p(m_1,...,m_K|\mathcal{M}) = \prod_{k=1}^K p(m_k|\mathcal{M}, m_1,...,m_{k-1})$$

   Early tokens encode coarse spatial locations and prototypes, while later tokens capture local fine-grained details. This design is naturally aligned with the autoregressive principle of LLMs.

2. **Hierarchical Mask Loss (HML)**: Explicit supervision at different levels enforces the hierarchical property. At level $l$, the first $l$ mask tokens are used by the MD to independently reconstruct $\hat{M}^{(l)}$, supervised against a Gaussian-blurred mask label $M^{(l)}$ with a kernel size corresponding to the level:

$$\mathcal{L}_{mask} = \sum_l \mathcal{L}_{mask}^{(l)}(\hat{M}^{(l)}, M^{(l)})$$

   Each level's loss combines BCE Loss and Dice Loss. Fewer tokens correspond to coarser Gaussian distributions, while more tokens correspond to finer boundaries. During training, a subset of levels is sampled according to an inverse power-law distribution for efficiency.

3. **Bidirectional Information Flow**: Training data includes both box→mask and mask→box conversion directions, enabling the LMM to learn the intrinsic relationship between detection and segmentation. Bounding boxes are generated directly by the LMM rather than parsed from masks. A notable finding is that outputting mask tokens before bounding boxes (a form of visual chain-of-thought) improves visual grounding accuracy.

### Loss & Training

- Stage 1: Mask reconstruction task; HiMTok fully trained; 256×256 resolution; codebook size 1024; 32 latent tokens.
- Stage 2: Joint optimization with cross-entropy loss + HML; LMM (InternVL 2.5 backbone) + HiMTok partially trained; low-resolution 448×448 input; 7.1M data samples.
- Stage 3: Cross-entropy loss only; LMM training only; high-resolution input; 5.0M data samples (segmentation data ratio reduced to 0.24).
- Compute: 2,752 A800 GPU-hours in total (192 + 1,920 + 640).

## Key Experimental Results

### Main Results

**Referring Expression Segmentation (RefCOCO/+/g, cIoU)**:

| Method | w/ SFM | RefCOCO val | RefCOCO+ val | RefCOCOg val |
|--------|--------|-------------|--------------|--------------|
| LISA-7B(ft) | ✓ | 74.9 | 65.1 | 67.9 |
| PixelLM-7B | ✓ | 73.0 | 66.3 | 69.3 |
| PSALM | ✓ | 83.6 | 72.9 | 73.8 |
| u-LLaVA | ✓ | 83.0 | 77.1 | 77.1 |
| **LMM_HiMTok-8B** | **✗** | 81.1 | 77.1 | 75.8 |
| **LMM_HiMTok-8B(ft)** | **✗** | **85.0** | **79.7** | **80.0** |
| LMM_HiMTok-8B(ft)+SAM | ✓ | 85.9 | 80.5 | 80.1 |

Without relying on any segmentation foundation model (SFM), the proposed method achieves state-of-the-art performance, substantially outperforming both prior SFM-free and SFM-based approaches.

**Open-Vocabulary Segmentation (mIoU)**:

| Method | ADE20K (A-150) | PASCAL Context | PASCAL VOC |
|--------|----------------|----------------|------------|
| PSALM | 18.2 | 48.5 | 81.3 |
| LaSagnA | 14.3 | 46.1 | 69.8 |
| **LMM_HiMTok-8B** | **25.0** | 43.9 | **82.0** |

### Ablation Study

**Effect of Hierarchical Mask Loss (HML)** (RefCOCO val / RefCOCO+ val / RefCOCOg val):

| HML | RefCOCO | RefCOCO+ | RefCOCOg |
|-----|---------|----------|----------|
| ✗ | 79.2 | 64.7 | 63.9 |
| ✓ | **81.1** | **77.1** | **75.8** |

Without HML, RefCOCO+/g scores drop substantially (−12.4/−11.9), and the model requires the full 32-token sequence to function; with HML, flexible token lengths are supported.

**Effect of Mask Token Length on REC (Visual Grounding)**:

| Token count → box | Acc@0.5 | Acc@0.9 |
|-------------------|---------|---------|
| 0 (direct box prediction) | ~90.3 | ~57 |
| 16 → box | ~92 | ~73 |
| 32 → box | **~93** | **~78** |

Mask tokens serving as a visual chain-of-thought significantly improve high-precision localization (Acc@0.9).

### Key Findings

- 16 mask tokens already achieve 82.8% cIoU; 32 tokens provide an additional 2.5% gain.
- In the bidirectional information flow, the mask→box direction is more beneficial, as preceding mask tokens are easier to generate than directly predicting bounding boxes.
- Nearly identical scores on ReasonSeg val and test (60.7 vs. 60.8 gIoU) indicate strong textual reasoning capability.
- General image understanding ability is largely preserved, with results on MME comparable to InternVL2.5-8B.
- Small object segmentation remains the primary challenge, with cIoU significantly lower than the overall performance.

## Highlights & Insights

- **Paradigm shift**: This work is the first to achieve high-quality LMM segmentation without relying on external segmentation models; mask tokens are learned by the LLM as a new form of language.
- **Input–output consistency**: Mask tokenization and detokenization are consistent across LLM input and output — a property that prior hidden-state-based approaches cannot guarantee.
- **Hierarchical design naturally aligns with LLM autoregression**: The coarse-to-fine token hierarchy perfectly matches next-token prediction.
- **Visual CoT effect of bidirectional information flow**: The segment-then-localize paradigm is a novel contribution that offers a new perspective on visual reasoning in LMMs.
- **Architectural simplicity**: The detokenizer is a lightweight transformer that does not require the original image at inference time.

## Limitations & Future Work

- The mask token length must be predefined and cannot adapt to the shape complexity of individual objects.
- The current model is passive in nature, requiring the user to specify a referring expression rather than proactively segmenting all objects of interest.
- The absence of multi-scale feature design limits performance on fine-grained regional segmentation.
- A notable performance gap exists between small object segmentation and overall results.
- Stage 2 training requires 1,920 A800 GPU-hours, incurring non-trivial computational cost.

## Related Work & Insights

- TiTok demonstrated that natural images can be compressed into a small number of 1D tokens, inspiring the compact representation of masks in this work.
- The coarse-to-fine next-scale prediction paradigm from VAR is adapted for mask hierarchy.
- InternVL 2.5 serves as the backbone LMM, providing a strong vision–language foundation.
- The limitations of hidden-state-based approaches such as LISA and PSALM are clearly analyzed, motivating the proposed new paradigm.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ A fundamentally new mask representation paradigm with an elegant hierarchical design and insightful bidirectional information flow.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers RES/GRES/ReasonSeg/OVS/REC/general understanding with detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ The three-paradigm comparison figure is clear, and the method is described comprehensively.
- **Value**: ⭐⭐⭐⭐⭐ Opens a new direction for LMM-based segmentation; open-sourced code; high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Hierarchical Visual Prompt Learning for Continual Video Instance Segmentation](hierarchical_visual_prompt_learning_for_continual_video_instance_segmentation.md)
- [\[ICCV 2025\] Advancing Visual Large Language Model for Multi-granular Versatile Perception](advancing_visual_large_language_model_for_multi-granular_versatile_perception.md)
- [\[ICCV 2025\] O-MaMa: Learning Object Mask Matching between Egocentric and Exocentric Views](o-mama_learning_object_mask_matching_between_egocentric_and_exocentric_views.md)
- [\[ICCV 2025\] Dynamic Dictionary Learning for Remote Sensing Image Segmentation](dynamic_dictionary_learning_for_remote_sensing_image_segmentation.md)
- [\[ICCV 2025\] Enhancing Transformers Through Conditioned Embedded Tokens](enhancing_transformers_through_conditioned_embedded_tokens.md)

</div>

<!-- RELATED:END -->
