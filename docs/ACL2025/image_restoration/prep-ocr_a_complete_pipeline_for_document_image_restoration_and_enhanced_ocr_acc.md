---
title: >-
  [Paper Note] PreP-OCR: A Complete Pipeline for Document Image Restoration and Enhanced OCR Accuracy
description: >-
  [ACL 2025][Image Restoration][Document Image Restoration] Proposes the PreP-OCR two-stage pipeline: first restoring historical document images using a ResShift model trained on synthetically degraded data (employing multi-directional patch extraction and median fusion), and then applying ByT5 for semantic post-OCR error correction. It reduces CER by 63.9-70.3% across 13,831 pages of real-world historical documents.
tags:
  - "ACL 2025"
  - "Image Restoration"
  - "Document Image Restoration"
  - "OCR Post-Correction"
  - "Synthetic Data"
  - "Historical Document Digitization"
  - "ByT5"
date: 2026-05-08
content_hash: 0ac081011ce49313
---

# PreP-OCR: A Complete Pipeline for Document Image Restoration and Enhanced OCR Accuracy

**Conference**: ACL 2025  
**arXiv**: [2505.20429](https://arxiv.org/abs/2505.20429)  
**Code**: [https://github.com/NikoGuan/PreP-OCR](https://github.com/NikoGuan/PreP-OCR)  
**Area**: Document Image Processing / OCR  
**Keywords**: Document Image Restoration, OCR Post-Correction, Synthetic Data, Historical Document Digitization, ByT5

## TL;DR
Proposes the PreP-OCR two-stage pipeline: first restoring historical document images using a ResShift model trained on synthetically degraded data (employing multi-directional patch extraction and median fusion), and then applying ByT5 for semantic post-OCR error correction. It reduces CER by 63.9-70.3% across 13,831 pages of real-world historical documents.

## Background & Motivation

**Background**: During digitization, a vast amount of historical documents suffer from degradation issues (such as blur, noise, ink stains, and aging), which severely impacts the quality of OCR text extraction. Existing approaches generally follow two technical routes: image preprocessing (such as contrast enhancement and denoising) or post-OCR error correction, but they are rarely integrated systematically.

**Limitations of Prior Work**: (1) Image restoration triggers improvements in visual quality but cannot eliminate all OCR errors—morphological ambiguities still persist. (2) Post-OCR error correction relies on predictable error patterns, but when restoration is insufficient, the error types become diverse and unpredictable. (3) While LLM-based OCR (such as GPT-4o) achieves high raw accuracy, it suffers from unstable outputs and "hallucination" problems.

**Key Challenge**: Paired real degraded document data (degraded images and clean counterparts) is virtually impossible to acquire, preventing the direct training of supervised restoration models.

**Goal**: How to train an image restoration model using purely synthetic data that generalizes well to real-world historical documents? How to make image restoration and semantic error correction collaborate effectively?

**Key Insight**: Programmatically simulating document degradation (by applying noise, blur, and morphological operations in a random sequence) to train the image restoration model, and synthesizing training pairs based on OCR error distributions to train the ByT5 correction model.

**Core Idea**: A synthetic-data-driven two-stage pipeline where image restoration resolves pixel-level ambiguity and text correction tackles semantic-level errors.

## Method

### Overall Architecture
Input degraded historical document image → Image Restoration (ResShift + Multi-directional Patch Extraction + Median Fusion) → OCR (Tesseract) → Post-Correction (ByT5) → Clean Text Output. The two stages process visual ambiguity and semantic errors respectively.

### Key Designs

1. **Synthetic Degradation Data Generation**:

    - **Function**: Starts from raw text, renders it into clean document images, and applies various degradation operations to generate paired training data.
    - **Mechanism**: Text is first rendered in multiple fonts (with random indentation, character offsets, rotations, warping, and line spacing variations) to generate clean base images. Then, degradation operations are applied in a **random sequence**: random noise, downsampling, Gaussian blur, black/white blocks, lines, texture overlays, ink stains, and morphological dilation/erosion. Four noise levels are predefined, and 10% undergo Otsu binarization.
    - **Design Motivation**: The **random sequence of operations** is key—identical operations yield different degradation effects when applied in different sequences, drastically increasing data diversity. A total of 100k image pairs were generated for training.

2. **Multi-directional Patch Extraction and Median Fusion**:

    - **Function**: Scans the image from 4 different directions to extract 256×256 patches, and fuses them using median values after restoration.
    - **Mechanism**: Scans the image in 4 directions: top-left to bottom-right, top-right to bottom-left, bottom-left to top-right, and bottom-right to top-left with a stride of 128. After restoring each patch, its outer 64-pixel border is discarded, retaining only the central 128×128 region. Finally, each pixel is reconstructed using the median of 4 independent predictions: $\hat{I}[r,c,\chi] = \text{median}(\hat{I}_k[r,c,\chi] | k \in \{1,2,3,4\})$
    - **Design Motivation**: Restoration quality is poor at patch borders (lower PSNR than center areas). Multi-directional scanning places the same region at different relative positions within different patches, and median fusion suppresses outliers/artifacts.

3. **ByT5 OCR Post-Correction**:

    - **Function**: Trains a ByT5-base model to correct OCR error text into clean text.
    - **Mechanism**: Extracts OCR error distributions (e.g., "m" → {"n": 0.001, "rn": 0.002, ...}) from ICDAR 2017 and injects errors into clean text based on these distributions to generate synthetic training pairs. ByT5 performs byte-level sequence-to-sequence translation.
    - **Design Motivation**: Byte-level tokenization is naturally suited for handling rare characters and diacritics in historical documents. The synthetic data approach avoids reliance on large amounts of human-annotated real post-OCR error datasets.

### Loss & Training
- Image Restoration: ResShift is trained on 90k synthetic image pairs with 256×256 patches.
- Post-Correction: ByT5-base is fine-tuned on around 890k synthetic text pairs, with a maximum length of 512 characters.
- Inference: Under the Multi-Median-64 configuration, rendering takes approximately 45 seconds per page on an RTX 4090.

## Key Experimental Results

### Main Results (English, Tesseract OCR)

| Pipeline | CER% (All) | CER% (No Outliers) | Relative Reduction from Raw |
|----------|------------|--------------------|-----------------------------|
| Raw (Original Image) | 5.91% | 5.87% | - |
| Pre (Image Restoration) | 2.81% | 1.99% | -66.1% |
| **PreP (Restoration + Correction)** | **2.00%** | **1.30%** | **-77.9%** |

### Cross-lingual Generalization (Tesseract, Restoration Model Trained on English)

| Language | Raw CER | Pre CER | PreP CER | Total Reduction |
|----------|---------|---------|----------|-----------------|
| English | 5.91% | 2.81% | 2.00% | -66.2% |
| French | 5.16% | 2.89% | 1.53% | -70.3% |
| Spanish | 7.12% | 3.42% | 2.57% | -63.9% |

### Ablation Study (Image Restoration Configuration, English CER%)

| Configuration | ResShift | DiffIR | MIMO-UNet+ |
|---------------|----------|--------|------------|
| Single-0 (No border cropping) | 4.43% | 3.77% | 4.65% |
| Single-64 (64px border cropping) | 3.17% | 3.12% | 3.70% |
| Multi-Median-64 | **2.81%** | **2.94%** | **3.65%** |

### Key Findings
- **Two-phase complementarity**: Image restoration primarily resolves character morphological ambiguity (reducing CER from 5.91% to 2.81%), while post-correction deals with systematic OCR errors (further reducing it from 2.81% to 2.00%). Each plays an irreplaceable role.
- **GPT-4o is unexpectedly unsuitable for this pipeline**: GPT-4o's "hallucinations" (generating plausible-looking but incorrect text) are easily misjudged as correct by the post-correction model, causing the post-correction CER to rise instead. In contrast, character-level errors from traditional Tesseract are more predictable and easier to rectify.
- **Effective cross-lingual zero-shot transfer**: The ResShift model trained on English directly applies to French/Spanish, achieving a 44-52% reduction in CER due to the structural similarities of Latin alphabets.
- **Significant patch boundary effects**: Discarding the outer 64px reduces CER from 4.43% to 3.17%, while multi-directional scanning plus median fusion further drives it down to 2.81%.

## Highlights & Insights
- **A successful case of "Synthetic-to-Real" generalization**: The model trained purely on synthetic data performs exceptionally well on 13,831 pages of real-world historical documents. The random generation sequence of degradation operations is the key to ensuring generalization.
- **Median operation in multi-directional patch fusion**: Simply and elegantly resolves patch boundary artifacts, mimicking a multi-view consensus mechanism.
- **An counter-intuitive finding that "traditional OCR outperforms LLM-OCR"**: Under pipeline scenarios, the predictable error patterns of Tesseract enable better synergy with the post-correction model, whereas GPT-4o's hallucinations are particularly hard to correct.

## Limitations & Future Work
- It is only validated on Latin-based alphabet systems (English/French/Spanish), while non-Latin scripts (Arabic, Chinese, etc.) remain untested.
- The parameter ranges for synthetic degradation operations are empirically set and might not cover all types of real-world degradations.
- The error distribution for the post-correction model is obtained from ICDAR 2017, which might not fully match the error patterns of LLM-based OCR.
- Inference speed is relatively slow (approx. 45 seconds per page), requiring efficiency considerations for large-scale digitization projects.
- The restoration model might struggle with extremely unconventional fonts.

## Related Work & Insights
- **vs. Pure Post-Correction Approaches**: Post-correction alone lacks visual information, rendering it limited for severely degraded documents. PreP-OCR first restores the image to enhance OCR input quality, followed by semantic correction.
- **vs. Diffusion Model-based Restoration**: DiffIR achieves the highest patch-level AMP (25.64 dB) but yields worse full-page OCR CER than ResShift, indicating that patch-level metrics do not perfectly align with downstream task-level performance.
- **vs. GPT-4o OCR**: GPT-4o achieves the lowest raw CER (2.34%), but is less suitable for pipeline hybridization due to the difficulty of rectifying its hallucinations during downstream correction.

## Rating
- Novelty: ⭐⭐⭐ Each component (image restoration, synthetic data, post-correction) has precedents; the innovation lies in the systematic integration and the multi-directional fusion strategy.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive ablation studies with 13,831 pages of real-world data, 3 languages, 6 restoration models, and 3 OCR systems.
- Writing Quality: ⭐⭐⭐⭐ Clear logic and rigorous problem formulation.
- Value: ⭐⭐⭐⭐ Holds direct practical significance for historical document digitization and is fully open-source.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MMDIR: Multimodal Instruction-Driven Framework for Mixed-Degradation Document Image Restoration](../../CVPR2026/image_restoration/mmdir_multimodal_instruction-driven_framework_for_mixed-degradation_document_ima.md)
- [\[CVPR 2026\] Beyond the Ground Truth: Enhanced Supervision for Image Restoration](../../CVPR2026/image_restoration/beyond_the_ground_truth_enhanced_supervision_for_image_restoration.md)
- [\[CVPR 2026\] ShiftLUT: Spatial Shift Enhanced Look-Up Tables for Efficient Image Restoration](../../CVPR2026/image_restoration/shiftlut_spatial_shift_enhanced_look-up_tables_for_efficient_image_restoration.md)
- [\[ECCV 2024\] Contourlet Residual for Prompt Learning Enhanced Infrared Image Super-Resolution](../../ECCV2024/image_restoration/contourlet_residual_for_prompt_learning_enhanced_infrared_image_super-resolution.md)
- [\[CVPR 2026\] LRHDR: Learning Representation-enhanced HDR Video Reconstruction](../../CVPR2026/image_restoration/lrhdr_learning_representation-enhanced_hdr_video_reconstruction.md)

</div>

<!-- RELATED:END -->
