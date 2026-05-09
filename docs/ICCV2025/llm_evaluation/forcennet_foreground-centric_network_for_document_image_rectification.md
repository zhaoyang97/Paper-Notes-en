---
title: >-
  [Paper Note] ForCenNet: Foreground-Centric Network for Document Image Rectification
description: >-
  [ICCV 2025][LLM Evaluation][document image rectification] This paper proposes ForCenNet, a foreground-centric document rectification network featuring three key contributions: foreground label generation, a mask-guided Transformer decoder, and a curvature consistency loss. The method requires only undistorted images for training and achieves state-of-the-art performance on four benchmarks: DocUNet, DIR300, WarpDoc, and DocReal.
tags:
  - ICCV 2025
  - LLM Evaluation
  - document image rectification
  - foreground guidance
  - curvature consistency loss
  - mask guidance
  - deformation field prediction
date: 2026-05-08
content_hash: bb156f354b39558d
---

# ForCenNet: Foreground-Centric Network for Document Image Rectification

**Conference**: ICCV 2025
**arXiv**: [2507.19804](https://arxiv.org/abs/2507.19804)
**Code**: [https://github.com/caipeng328/ForCenNet](https://github.com/caipeng328/ForCenNet)
**Area**: Document Analysis / Image Rectification
**Keywords**: document image rectification, foreground guidance, curvature consistency loss, mask guidance, deformation field prediction

## TL;DR

This paper proposes ForCenNet, a foreground-centric document rectification network featuring three key contributions: foreground label generation, a mask-guided Transformer decoder, and a curvature consistency loss. The method requires only undistorted images for training and achieves state-of-the-art performance on four benchmarks: DocUNet, DIR300, WarpDoc, and DocReal.

## Background & Motivation

Document image rectification aims to remove geometric distortions from photographed documents to facilitate downstream tasks such as OCR. Existing methods face several critical challenges:

**Foreground neglect**: As shown in Figure 1, readable regions in documents (text, table lines) occupy only a small fraction of pixels, yet the most prominent distortions concentrate in the background. Existing methods (CGU-Net, DocRes, etc.) predict deformation fields uniformly over the entire image, leading to a misalignment between the task objective (foreground readability) and the optimization objective (per-pixel accuracy over the full image).

**Incomplete foreground definition**: DocGeoNet focuses on text-line masks, while FTDR uses frozen detection models to coarsely extract text-line information. However: (a) conventional detection models struggle to accurately recognize distorted text lines; (b) document foreground encompasses not only text but also table lines, figures, and other elements.

**Scarcity of annotated data**: Fine-grained annotations for document rectification are difficult to obtain. Weakly supervised methods circumvent this issue but sacrifice readability.

**Core Idea**: Detailed foreground elements (text, straight lines, figures) are extracted from undistorted images to construct a foreground-centric training framework that requires only undistorted reference images, enabling efficient iterative training.

## Method

### Overall Architecture

ForCenNet consists of three stages: (a) foreground label generation — foreground elements are extracted from undistorted images to produce training data; (b) foreground-centric network architecture — feature extraction + Transformer encoder + foreground segmentation + mask-guided decoder; (c) foreground-centric optimization — mask loss + deformation field regression + curvature consistency loss.

### Key Designs

1. **Foreground Label Generation**: Derived entirely from undistorted images without manual annotation:

    - **Character-level foreground segmentation**: Hi-SAM is fine-tuned to uniformly segment text regions, line elements, and figures, yielding the foreground mask $M_u$.
    - **Line element extraction**: An OCR engine extracts text lines (represented by the centerline of bounding boxes); a dedicated LSD-based document line detection algorithm (Algorithm 1) detects table lines, filtering non-horizontal/non-vertical lines and suppressing duplicate detections.
    - **Distortion field generation**: Native backward mappings $\mathcal{BM}$ are obtained from DOC3D, and the corresponding forward mappings $\mathcal{FM}$ are computed. Distortion fields are augmented via random cropping and pairwise overlap; $\mathcal{FM}$ is then applied to undistorted images, masks, and line elements to generate training samples.

2. **Mask-Guided Transformer Decoder**: Predicted foreground masks are used to guide feature extraction:

    - Feature extraction module: large-kernel convolution + residual layers, outputting $F_u \in \mathbb{R}^{H/8 \times W/8 \times 256}$.
    - Efficient Transformer encoder: 3-layer vanilla Transformer with overlapping patch embedding (kernel=3, stride=2) and an SPW strategy to reduce attention complexity.
    - Foreground segmentation module: a lightweight network predicts a binary mask $M$, smoothed via softmax to obtain probability density $\tilde{M}$.
    - Mask-guided self-attention: $\text{MSA}(Q,K,V) = \text{Softmax}(\frac{QK^T + \sigma \text{Seq}(\tilde{M})\text{Seq}(\tilde{M})^T}{\sqrt{d_{\text{head}}}})V$, directing attention toward foreground regions.
    - Encoder–decoder cross-attention enables multi-scale information fusion.

3. **Curvature Consistency Loss**: A geometry-aware loss designed for fine-grained elements such as table lines:

    - **Motivation**: Table line pixels are sparse, making L1 supervision weak; moreover, L1 captures only pixel offsets without encoding geometric structure.
    - Points are sampled every 4 pixels along line elements to form the set $P$; bilinear interpolation projects them onto the predicted and ground-truth deformation fields to obtain control points $Cp$ and $Cp_{gt}$.
    - Curvature is computed via central differences: $\kappa_i = \frac{|x'_i y''_i - y'_i x''_i|}{(x'^2_i + y'^2_i)^{3/2} + \varepsilon}$
    - The predicted curvature trend is constrained to match the ground-truth curvature: $\mathcal{L}_k = \frac{1}{N-1}\sum_i^{N-1}(\hat{k_i} - k_i)$

### Loss & Training

The total loss comprises three terms:
- $\mathcal{L}_{seg}$: L1 loss on the foreground mask.
- $\mathcal{L}_{map}$: L1 regression loss on the backward mapping, $\|\hat{\mathcal{BM}} - \mathcal{BM}\|_1$.
- $\mathcal{L}_k$: curvature consistency loss.

Training details:
- Input resolution: 288×288; AdamW optimizer; batch size 32.
- OneCycle learning rate schedule, maximum lr=10⁻⁴, 10% warmup.
- 2× NVIDIA A100; trained for 30 epochs.
- Two training variants: ForCenNet (365 undistorted images from DocUNet+DIR300) and ForCenNet-DOC3D (undistorted images from DOC3D).

## Key Experimental Results

### Main Results

Comparison on the DocUNet benchmark:

| Type | Method | MS-SSIM↑ | LD↓ | AD↓ | ED↓ | CER↓ |
|------|--------|----------|-----|-----|-----|------|
| Weakly-sup. | PaperEdge | 0.470 | 8.49 | 0.39 | 825.48 | 0.211 |
| Weakly-sup. | FDRNet | 0.542 | 8.21 | – | 829.78 | 0.206 |
| Foreground | DocGeoNet | 0.504 | 7.71 | 0.38 | 713.94 | 0.182 |
| Foreground | FTDR | 0.497 | 8.43 | 0.37 | 697.52 | 0.170 |
| Foreground | LA-DocFlatten | 0.526 | 6.72 | 0.30 | – | – |
| Other | CGU-Net | 0.557 | 6.83 | 0.31 | 513.76 | 0.178 |
| **Foreground** | **ForCenNet** | **0.582** | **4.82** | **0.19** | **571.40** | **0.136** |

On the DIR300 benchmark, ForCenNet achieves MS-SSIM=0.713 and LD=4.65, with ED dropping below 400 for the first time (390.61).

Cross-domain generalization (WarpDoc and DocReal, without additional fine-tuning):

| Method | WarpDoc MS-SSIM↑ | WarpDoc LD↓ | DocReal MS-SSIM↑ | DocReal LD↓ |
|--------|-----------------|------------|-----------------|------------|
| DocTr | 0.39 | 27.01 | 0.550 | 12.60 |
| CGU-Net | 0.35 | 26.28 | 0.549 | 11.33 |
| DocRes | 0.50 | 12.86 | 0.550 | 11.52 |
| **ForCenNet** | **0.54** | **8.10** | **0.595** | **6.95** |

### Ablation Study

Module ablation on the DocUNet benchmark:

| ID | Mask-Guided (MGD) | Curvature Loss (CL) | MS-SSIM↑ | LD↓ | CER↓ |
|----|------------------|---------------------|----------|-----|------|
| D | ✗ | ✗ | 0.530 | 7.06 | 0.198 |
| A | ✗ | ✓ | 0.558 | 5.44 | 0.173 |
| B | ✓ | ✗ | 0.565 | 5.10 | 0.169 |
| **C** | **✓** | **✓** | **0.571** | **4.95** | **0.141** |

Data scale ablation: using 65 undistorted images with varying distortion field augmentation multipliers:

| Multiplier | MS-SSIM↑ | LD↓ | CER↓ |
|-----------|----------|-----|------|
| ×1 | 0.449 | 10.745 | 0.291 |
| ×100 | 0.530 | 5.348 | 0.208 |
| ×500 | 0.566 | 4.892 | 0.149 |
| ×1000 | 0.571 | 4.950 | 0.141 |
| ×2000 | 0.567 | 4.965 | 0.147 |

Performance saturates at approximately ×500–×1000.

### Key Findings

- The mask-guided module (MGD) contributes more substantially (MS-SSIM: 0.530→0.565), while the curvature consistency loss yields significant CER improvement (0.169→0.141).
- A strong model can be trained using only 65 undistorted images combined with distortion field augmentation, demonstrating high label generation efficiency.
- Freezing the foreground segmentation model causes a sharp performance drop (MS-SSIM: 0.571→0.468), confirming the necessity of differentiable end-to-end training.
- ForCenNet's line rectification outperforms DocRes, yielding greater total line segment length on 65% of DocReal and 69% of WarpDoc samples.
- Robust cross-domain performance demonstrates the strong generalization capability of the foreground-guided strategy.

## Highlights & Insights

- **Precise problem framing**: The paper identifies the fundamental contradiction in document rectification — foreground regions most in need of correction occupy the fewest pixels — and proposes a systematic solution.
- **Elegant label generation**: Full training data can be generated automatically from undistorted reference images combined with random distortion fields, substantially reducing annotation costs.
- **Comprehensive foreground definition**: Text, table lines, and figures are treated as a unified set of foreground elements, providing broader coverage than DocGeoNet or FTDR.
- **Novel curvature consistency loss**: Constraining the deformation of line elements from a geometric perspective captures structural information beyond what pure L1 loss can express.
- **Differentiable foreground segmentation**: Ablation experiments validate the necessity of end-to-end learning; freezing the segmentation module leads to a large performance degradation.

## Limitations & Future Work

- Foreground segmentation relies on fine-tuned Hi-SAM, which may not be robust to rare characters or extreme distortions.
- The fixed input resolution of 288×288 may lose fine details for high-resolution documents.
- Distortion fields sourced from DOC3D may not cover all real-world distortion types.
- Data scale augmentation exhibits a saturation point; more diverse distortion field designs may yield further gains.
- Illumination correction (shadows, uneven lighting) is not addressed; joint optimization with illumination rectification is a natural extension.
- Foreground masks could be further refined (e.g., distinguishing text, tables, and figures) to enable differentiated loss weighting.

## Related Work & Insights

- DocGeoNet and FTDR impose text-line foreground constraints; ForCenNet extends this to line elements and figures, replacing coarse detection-based constraints with mask guidance and a curvature loss.
- Weakly supervised approaches (PaperEdge, FDRNet, DRNet) avoid manual annotation at the cost of readability; ForCenNet's label generation strategy combines the advantages of both paradigms.
- The curvature consistency loss concept is generalizable to other tasks requiring geometric consistency, such as map rectification and architectural facade correction.
- The mask-guided attention design is applicable to other vision tasks that require region-level focus.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The foreground label generation pipeline and curvature consistency loss are novel contributions, though the overall framework (encoder–decoder + deformation field) follows established paradigms.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four real-world benchmarks, cross-domain evaluation, comprehensive ablations, rich visualizations, and detailed data-scale and structural ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with complete mathematical derivations; occasional symbol rendering artifacts (LaTeX artifacts) slightly impede readability.
- **Value**: ⭐⭐⭐⭐ Delivers practical value to the document rectification community; the label generation scheme lowers the barrier for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] PHATNet: A Physics-guided Haze Transfer Network for Domain-adaptive Real-world Image Dehazing](phatnet_a_physics-guided_haze_transfer_network_for_domain-adaptive_real-world_im.md)
- [\[ICCV 2025\] OmniDiff: A Comprehensive Benchmark for Fine-grained Image Difference Captioning](omnidiff_a_comprehensive_benchmark_for_fine-grained_image_difference_captioning.md)
- [\[NeurIPS 2025\] Document Summarization with Conformal Importance Guarantees](../../NeurIPS2025/llm_evaluation/document_summarization_with_conformal_importance_guarantees.md)
- [\[AAAI 2026\] Axis-Aligned Document Dewarping](../../AAAI2026/llm_evaluation/axis-aligned_document_dewarping.md)
- [\[AAAI 2026\] HybriDLA: Hybrid Generation for Document Layout Analysis](../../AAAI2026/llm_evaluation/hybridla_hybrid_generation_for_document_layout_analysis.md)

</div>

<!-- RELATED:END -->
