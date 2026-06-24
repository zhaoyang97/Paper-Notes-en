---
title: >-
  [Paper Note] WISE: A Framework for Gigapixel Whole-Slide-Image Lossless Compression
description: >-
  [CVPR 2025][Medical Imaging][Lossless Compression] To address the failure of existing lossless compression methods caused by the "information irregularity" (widespread high-frequency signals and high volatility) of WSI images, this paper proposes the WISE three-step compression framework (hierarchical projection coding $\rightarrow$ bitmap encoding $\rightarrow$ dictionary coding), achieving an average of 36x and up to 136x lossless compression.
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Lossless Compression"
  - "Whole Slide Images"
  - "Pathology Images"
  - "Dictionary Coding"
  - "Information Irregularity"
date: 2026-05-08
content_hash: 7d2c9c7f9316f133
---

# WISE: A Framework for Gigapixel Whole-Slide-Image Lossless Compression

**Conference**: CVPR 2025  
**arXiv**: [2503.18074](https://arxiv.org/abs/2503.18074)  
**Code**: None  
**Area**: Medical Imaging  
**Keywords**: Lossless Compression, Whole Slide Images, Pathology Images, Dictionary Coding, Information Irregularity

## TL;DR
To address the failure of existing lossless compression methods caused by the "information irregularity" (widespread high-frequency signals and high volatility) of WSI images, this paper proposes the WISE three-step compression framework (hierarchical projection coding $\rightarrow$ bitmap encoding $\rightarrow$ dictionary coding), achieving an average of 36x and up to 136x lossless compression.

## Background & Motivation
1. **Background**: Whole Slide Images (WSIs) are the core data format in digital pathology, with a single WSI reaching up to several gigabytes (width $\times$ height $\times$ color channels $\times$ multi-resolution pyramid), incurring extremely high storage and transmission costs. In practice, hospitals even ship hard drives via FedEx to transfer WSI data.
2. **Limitations of Prior Work**: Existing lossy compression methods (JPEG-2000, VQVAE) introduce distortions that affect clinical diagnosis. Conversely, lossless methods (PNG, Huffman, Gzip, and even NN-based methods) exhibit poor performance on WSIs—PNG achieves almost no compression on WSIs (~1.01x), and even the best-performing LZMA only achieves ~2x.
3. **Key Challenge**: WSIs possess unique frequency domain characteristics, where the proportion of high-frequency signals is significantly higher than in natural images, and local extrema occur frequently (referred to as information irregularity). This renders compression methods based on entropy coding and pixel prediction utterly ineffective.
4. **Goal**: Design a lossless compression method specifically tailored to the characteristics of information irregularity in WSIs.
5. **Key Insight**: It is observed that dictionary-based methods (e.g., LZMA) outperform image-specific methods (e.g., PNG) on WSIs, as dictionary matching is more tolerant of highly volatile data than pixel prediction. The problem lies in how to first reduce the information entropy of the data before letting dictionary-based methods achieve their maximum efficacy.
6. **Core Idea**: Reduce entropy layer by layer through a three-step process: narrowing the pixel value range via hierarchical differential coding $\rightarrow$ aggregating active bits using bitmap transposition $\rightarrow$ capturing repetitive patterns via dictionary coding.

## Method

### Overall Architecture
The WISE framework processes the base level in the multi-resolution pyramid (other levels can be generated via downsampling) in a patch-by-patch manner. It consists of four steps: (1) **Preprocessing**: removing extensive blank areas and the alpha channel in the WSI; (2) **Hierarchical Projection Coding**: line $\rightarrow$ column $\rightarrow$ channel three-directional differential coding to narrow the value range; (3) **Bitmap Encoding**: transposing the differential results bit by bit to aggregate active bits; (4) **Dictionary Coding**: leveraging the LZW algorithm to capture long repetitive patterns.

### Key Designs

1. **Hierarchical Projection Coding**

    - **Function**: Significantly reduce the information entropy of pixel values through three-directional differential coding.
    - **Mechanism**: For each pixel $(m,n,c)$, the row-wise difference $\Delta X_{m,n,c} = X_{m,n,c} - X_{m-1,n,c}$, column-wise difference $\Delta^r X_{m,n,c} = \Delta X_{m,n,c} - \Delta X_{m,n-1,c}$, and channel-wise difference $Y_{m,n,c} = \Delta^r X_{m,n,c} - \Delta^r X_{m,n,1}$ are calculated sequentially. Each step leverages the physical similarity of the nearest neighbors to compress the original range $[0, 255]$ of pixel values into a narrow range centered around 0. In the illustrative example, the entropy drops from 7.29 to 5.13.
    - **Design Motivation**: The high-frequency fluctuations in WSIs render long-range predictions ineffective, yet nearest-neighbor differencing remains effective. Hierarchical three-directional prediction ensures that local correlation is maximized across the row, column, and channel dimensions.

2. **Bitmap Encoding**

    - **Function**: Reorganize the bit structure of the differentially coded values to aggregate active bits, creating more repetitive patterns.
    - **Mechanism**: The differential values are mostly close to zero, meaning their binary representations contain a large number of 0s or sign bits in the high-order positions, while only the low-order bits carry active information. The encoding is transposed by bit position—repacking the data by grouping the $i$-th bit of all bytes together. Consequently, the high-order bits form numerous repeating bytes of 0x00 or 0xFF, while the low-order bits, though more random, still exhibit local patterns. Although the byte-level entropy may temporarily rise (from 5.13 to 5.56), a large number of long repeating patterns are successfully generated.
    - **Design Motivation**: Dictionary methods rely on repeating pattern matching, and bit transposition aggregates scattered "inactive bits" into long repeating sequences, creating ideal inputs for the subsequent dictionary coding.

3. **LZW Dictionary Coding**

    - **Function**: Capture the long repeating patterns generated after bitmap encoding to output the final compressed data.
    - **Mechanism**: The LZW algorithm dynamically constructs a dictionary online, substituting frequently occurring byte sequences with short indices. Uniquely, it does not require prior knowledge of the probability distribution (unlike arithmetic coding), making it highly suitable for WSIs with irregular data distributions. The massive repeating byte sequences (e.g., continuous 0x00s) produced by bitmap encoding are compressed with extreme efficiency. In the illustrative example, the final entropy drops from 5.56 to 2.54.
    - **Design Motivation**: The long repeating patterns arising after bitmap encoding present an ideal input for dictionary methods. The online nature of LZW avoids dependency on global statistics, making it perfectly suited for the high variability observed in WSIs.

### Loss & Training
WISE is a purely algorithmic, training-free compression method that does not involve any deep learning training. All steps consist of deterministic encoding and decoding.

## Key Experimental Results

### Main Results

| Method | C16 Avg. Compression Ratio↑ | C17 Avg. Compression Ratio↑ | Type |
|------|----------------|----------------|------|
| Huffman | ~2.0 | ~4.4 | Entropy Coding |
| PNG | ~4.4 | ~9.5 | Image Coding |
| Gzip | ~5.3 | ~19.0 | Dictionary Coding |
| LZMA | ~8.2 | ~27.1 | Dictionary Coding |
| Zstd-22 | ~7.7 | ~24.2 | Hybrid Coding |
| **WISE** | **~12.6** | **~37.2** | Ours |

Peak compression ratio on a single image: **136.15x** on Img5 of the C17 dataset.

### Ablation Study

| Configuration | Entropy (Example Matrix) | Description |
|------|-------------|------|
| Original Pixel Values | 7.29 | Unprocessed |
| +Row Projection | 6.01 | Row differencing reduces entropy |
| +Column Projection | 5.32 | Column differencing further reduces entropy |
| +Channel Projection | 5.13 | Channel differencing continues reduction |
| +Bitmap Encoding | 5.56 (Byte entropy increases) | but generates abundant repeating patterns |
| +Dictionary Coding | **2.54** | Final substantial entropy reduction |

### Key Findings
- Standard image compression methods fail completely on WSIs: PNG achieves only ~1.01x on WSIs compared to ~2.06x on Kodak.
- Dictionary-based methods are naturally superior to entropy and image-specific methods on WSIs, as the high-frequency fluctuations in WSIs violate the fundamental assumptions of entropy and predictive coding.
- WISE outperforms Zstd-22 (the strongest dictionary baseline) by 70-80%, proving the significant boost provided by the first two encoding steps to dictionary-based methods.
- The proportion of blank areas heavily impacts the final compression ratio; WSIs containing large blank regions can achieve a compression ratio exceeding 100x.

## Highlights & Insights
- **In-depth failure analysis** is the highlight of this paper—rather than proposing a method upfront, the authors thoroughly analyze why existing methods fail (due to information irregularity) and customize their solution accordingly. This "diagnosis before prescription" style of research methodology is highly worth standardizing.
- **Minimalist yet highly effective**: Without any deep learning components, this purely algorithmic approach accomplishes a 36x average compression ratio, offering outstanding engineering utility.
- **Bitmap transposition** is a general compression preprocessing trick that can be applied to any data where active bits are concentrated in the lower bits.

## Limitations & Future Work
- Only the base level of the pyramid is processed; joint compression of multi-resolution levels could potentially yield even higher compression ratios.
- Comparisons against deep learning-based lossless compression methods (e.g., L3C, ArIB-BPS) using optimized WSI representations are absent.
- Analysis of compression and decompression speeds lacks fine-grained details.
- Future studies could consider integrating a DNN projection head to replace the fixed differential scheme, thereby providing better-suited inputs for dictionary-based approaches.

## Related Work & Insights
- **vs. PNG/TIFF**: Predictive coding-based image compression methods fail on WSIs due to high-frequency fluctuations disrupting spatial prediction.
- **vs. ArIB-BPS**: Deep learning-based lossless compression methods fundamentally rely on entropy coding, which is similarly constrained by the irregular distribution of WSIs.
- **vs. LZMA**: Standard dictionary methods applied directly without information reorganization yield compression ratios only about half of what WISE achieves.

## Rating
- Novelty: ⭐⭐⭐⭐ WSI lossless compression is studied profoundly for the first time, and the encoding pipeline design is logical and well-justified.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 6 datasets alongside various baselines, with clear step-by-step ablation analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ In-depth and thorough analysis with seamless logic transitions from failure analysis to method design.
- Value: ⭐⭐⭐⭐⭐ Highly practical engineering value, addressing genuine storage and transmission bottlenecks in digital pathology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TopoSlide: Topologically-Informed Histopathology Whole Slide Image Representation Learning](../../CVPR2026/medical_imaging/toposlide_topologically-informed_histopathology_whole_slide_image_representation.md)
- [\[CVPR 2025\] CARL: A Framework for Equivariant Image Registration](carl_a_framework_for_equivariant_image_registration.md)
- [\[CVPR 2026\] MLLM-HWSI: A Multimodal Large Language Model for Hierarchical Whole Slide Image Understanding](../../CVPR2026/medical_imaging/mllm-hwsi_a_multimodal_large_language_model_for_hierarchical_whole_slide_image_u.md)
- [\[CVPR 2026\] Act Like a Pathologist: Tissue-Aware Whole Slide Image Reasoning](../../CVPR2026/medical_imaging/act_like_a_pathologist_tissue-aware_whole_slide_image_reasoning.md)
- [\[CVPR 2026\] Turning Pre-Trained Vision Transformers into End-to-End Histopathology Whole Slide Image Models for Survival Prediction](../../CVPR2026/medical_imaging/turning_pre-trained_vision_transformers_into_end-to-end_histopathology_whole_sli.md)

</div>

<!-- RELATED:END -->
