---
title: >-
  [Paper Note] Revisiting Image Fusion for Multi-Illuminant White-Balance Correction
description: >-
  [ICCV 2025][White Balance] This paper addresses white-balance (WB) correction under multi-illuminant scenes by proposing an efficient Transformer-based fusion model to replace conventional linear fusion…
tags:
  - "ICCV 2025"
  - "White Balance"
  - "Multi-Illuminant"
  - "Image Fusion"
  - "Transformer"
  - "Dataset"
date: 2026-05-08
content_hash: dffd62e762c6bb7a
---

# Revisiting Image Fusion for Multi-Illuminant White-Balance Correction

**Conference**: ICCV 2025
**arXiv**: [2503.14774](https://arxiv.org/abs/2503.14774)  
**Code**: N/A  
**Area**: Image Processing
**Keywords**: White Balance, Multi-Illuminant, Image Fusion, Transformer, Dataset

## TL;DR

This paper addresses white-balance (WB) correction under multi-illuminant scenes by proposing an efficient Transformer-based fusion model to replace conventional linear fusion, alongside a large-scale multi-illuminant WB dataset containing 16,000+ images. The proposed method achieves a 100% improvement in correction quality over existing methods on the new dataset.

## Background & Motivation

**Background**: White balance is a core step in camera ISP pipelines, aiming to eliminate the influence of illuminant color temperature on image appearance. Most WB methods assume a single dominant light source, under which a global color temperature is estimated and a correction matrix is applied. Recent fusion-based WB methods render the same image under multiple predefined WB presets (e.g., daylight, fluorescent, cloudy) and learn pixel-wise fusion weights via neural networks.

**Limitations of Prior Work**: (1) Existing fusion methods perform only linear weighted fusion — the output at each pixel is a convex combination of WB-preset versions. While sufficient for single-illuminant scenes, this is severely limited for multi-illuminant scenes, where different regions require entirely different correction strategies that linear fusion cannot capture. (2) Existing WB datasets (e.g., WB-sRGB, Rendered WB) lack dedicated multi-illuminant images, leading to insufficient training and evaluation of models in multi-illuminant scenarios.

**Key Challenge**: Multi-illuminant scenes demand spatially adaptive, nonlinear correction, which neither the linear fusion assumption nor single-illuminant datasets can support.

**Goal**: (1) Design a nonlinear fusion model capable of capturing spatial dependencies; (2) Construct a dedicated large-scale multi-illuminant WB dataset.

**Key Insight**: WB correction in multi-illuminant scenes is inherently a spatially dependent fusion problem — different image regions are affected by different light sources, requiring global context to determine the optimal WB strategy per region. The long-range dependency modeling capability of Transformers is naturally suited for this task.

**Core Idea**: Replace linear fusion with an efficient Transformer to fuse multiple WB-preset versions of an image, enabling the model to leverage global context for more accurate spatially adaptive white-balance decisions.

## Method

### Overall Architecture

The input consists of five WB-preset versions of a single sRGB image (D65 daylight, cloudy, fluorescent-A, fluorescent-CWF, and incandescent), rendered from the RAW image using known WB preset matrices. The model outputs a WB-corrected image. The overall architecture follows an encoder–decoder design: the encoder extracts and merges features from the five versions; Transformer modules model cross-preset and cross-spatial dependencies; the decoder generates per-pixel fusion coefficient maps.

### Key Designs

1. **Cross-Preset Transformer Fusion**:

    - **Function**: Models dependencies across different WB preset versions and across spatial locations.
    - **Mechanism**: Feature maps from the five WB preset versions are concatenated along the channel dimension and fed into efficient Transformer blocks. Window Attention combined with a Shifted Window mechanism (analogous to Swin Transformer) controls computational complexity, computing self-attention within local windows while enabling cross-window information flow via shifts. Attention operates jointly over spatial and preset dimensions — for each pixel location, the model can attend to all preset versions of the same region as well as the same preset of other regions.
    - **Design Motivation**: The fundamental limitation of linear fusion is that fusion weights for each WB version are computed independently, without leveraging global context. The attention mechanism allows the model to infer the optimal fusion strategy for a local region based on the global illuminant distribution inferred from the entire image.

2. **Lightweight Multi-Scale Encoder**:

    - **Function**: Efficiently extracts multi-scale features.
    - **Mechanism**: A lightweight CNN with shared weights (a MobileNetV2-based variant) extracts features from each of the five WB versions independently, producing feature maps at three scales (1/4, 1/8, and 1/16 resolution). Weight sharing reduces parameter count to 1/5 of the non-shared counterpart while ensuring all versions are represented in a consistent feature space. Multi-scale features are passed to the decoder via skip connections, preserving low-level color detail while capturing high-level semantics.
    - **Design Motivation**: Multi-illuminant scenes require both fine-grained pixel-level color information (to distinguish illuminant sources in shadows vs. highlights) and coarse-grained scene understanding (to infer the global illuminant distribution); the multi-scale design satisfies both requirements.

3. **Large-Scale Multi-Illuminant WB Dataset**:

    - **Function**: Provides dedicated training and evaluation data for multi-illuminant WB.
    - **Mechanism**: Constructed via a physics-based rendering pipeline: (a) RAW images of multi-illuminant scenes with known illuminant information are collected; (b) each RAW image is rendered into sRGB under five standard WB presets; (c) ground-truth images are generated by applying the correct WB matrix to each pixel according to its dominant illuminant. The resulting dataset contains 16,000+ image pairs (5 presets + 1 GT), covering diverse multi-illuminant scenarios including indoor mixed lighting (window daylight + artificial illumination), colored lighting, and daylight–shadow mixtures.
    - **Design Motivation**: Existing datasets such as Set1/Set2 contain only hundreds of multi-illuminant images and lack corresponding multi-preset renderings. The new dataset fills this gap and enables, for the first time, systematic study of multi-illuminant WB in the community.

### Loss & Training

The primary loss is an L1 color reconstruction loss $\mathcal{L}_{color} = \|I_{pred} - I_{gt}\|_1$, supplemented by an angular loss $\mathcal{L}_{ang}$ that constrains the accuracy of chromaticity direction by computing the angular error between the predicted and ground-truth RGB vectors at each pixel. The total loss is $\mathcal{L} = \mathcal{L}_{color} + \alpha \mathcal{L}_{ang}$ with $\alpha = 0.5$. Training uses AdamW with cosine annealing learning rate scheduling and patch-based training with patch size 256×256.

## Key Experimental Results

### Main Results

Comparison on the proposed multi-illuminant dataset:

| Method | Type | MAE↓ | $\Delta E$↓ | PSNR↑ | SSIM↑ |
|--------|------|------|------------|-------|-------|
| Deep WB (Afifi 2020) | Non-fusion | 5.82 | 7.34 | 23.1 | 0.841 |
| Mixed-Ill WB | Non-fusion | 4.95 | 6.21 | 24.3 | 0.867 |
| WB-sRGB Fusion | Linear fusion | 4.12 | 5.38 | 25.7 | 0.889 |
| CLCC | Linear fusion | 3.87 | 4.91 | 26.2 | 0.895 |
| Ours | Transformer fusion | 1.93 | 2.46 | 31.4 | 0.952 |

### Ablation Study

| Configuration | MAE↓ | PSNR↑ | Note |
|---------------|------|-------|------|
| Full model | 1.93 | 31.4 | Complete Transformer fusion |
| Transformer replaced by linear fusion | 3.72 | 26.5 | Degrades to conventional scheme; −48% |
| w/o angular loss | 2.31 | 30.1 | Chromaticity constraint is important |
| w/o multi-scale skip connections | 2.48 | 29.6 | Detail recovery degraded |
| 3 presets (2 removed) | 2.67 | 28.9 | Preset count matters, but with diminishing returns |
| Non-shared encoder weights | 1.98 | 31.2 | Marginally better but 5× more parameters |

### Key Findings

- **Linear fusion is the primary bottleneck**: Replacing linear fusion with a Transformer reduces MAE from 3.72 to 1.93, demonstrating that spatial dependency modeling is critical.
- **Dataset matters significantly**: The same method trained on single-illuminant datasets and applied to multi-illuminant scenes performs substantially worse than when trained on the proposed dataset.
- Five WB presets outperform three by a clear margin, while increasing to seven yields only marginal improvement.
- The angular loss contributes more to color fidelity than L1 — L1 optimization tends to favor low-luminance regions, whereas the angular loss treats all luminance levels equally.

## Highlights & Insights

- **Revealing the fundamental limitation of linear fusion**: Through theoretical analysis and experiments, the paper demonstrates that linear fusion is principally inadequate for multi-illuminant scenes — the correct correction for regions under different illuminants is not a simple convex combination of preset-rendered versions. This finding motivates a paradigm shift in fusion-based WB research.
- **Significant dataset contribution**: The 16,000+ multi-illuminant image dataset is the first large-scale benchmark specifically targeting multi-illuminant WB scenarios, filling a critical data gap in the field.
- **Efficient and practical method design**: Window attention based on the Swin Transformer maintains linear computational complexity, making the approach applicable in practical ISP pipelines or post-processing workflows.

## Limitations & Future Work

- The method relies on five predefined WB presets; performance may degrade when the true illuminant color temperature falls outside the range covered by these presets.
- The dataset is constructed via physics-based rendering, which may introduce a domain gap relative to real camera ISP pipeline rendering.
- Video scenarios are not considered — temporally consistent multi-illuminant WB correction remains a more challenging open problem.
- Adaptive preset selection — dynamically determining which and how many WB preset versions to use — is a promising direction for future exploration.
- Integration with RAW-domain WB methods could be explored, where an initial correction is performed in the RAW domain followed by fine-grained refinement in the sRGB domain via the fusion approach.

## Related Work & Insights

- **vs. Deep WB (Afifi et al.)**: Deep WB directly predicts WB correction parameters from a single sRGB image without employing a multi-preset fusion strategy. It fails in multi-illuminant scenes due to globally estimating a single color temperature; the proposed local fusion scheme is more appropriate.
- **vs. CLCC**: CLCC is the strongest linear fusion baseline, improving fusion weights through chromaticity constraints. However, it remains bounded by the linear fusion limitation; the proposed Transformer-based approach achieves a 50% improvement in MAE.
- **vs. Traditional AWB Methods (Gray World / White Patch)**: Traditional methods rely entirely on statistical assumptions that break down under multiple illuminants. The proposed data-driven approach circumvents these assumption-based limitations.

## Rating

- Novelty: ⭐⭐⭐⭐ — Introducing Transformers into WB fusion and constructing a dedicated dataset constitutes a solid combinatorial contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Ablations are comprehensive and comparisons are thorough; cross-validation on existing datasets is lacking.
- Writing Quality: ⭐⭐⭐⭐ — Problem motivation is clearly articulated; dataset construction process is described in detail.
- Value: ⭐⭐⭐⭐ — The dataset provides long-term value to the community; the method has practical application potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] EDFFDNet: Towards Accurate and Efficient Unsupervised Multi-Grid Image Registration](edffdnet_towards_accurate_and_efficient_unsupervised_multi-grid_image_registrati.md)
- [\[NeurIPS 2025\] Depth-Supervised Fusion Network for Seamless-Free Image Stitching](../../NeurIPS2025/others/depth-supervised_fusion_network_for_seamless-free_image_stitching.md)
- [\[NeurIPS 2025\] Revisiting Agnostic Boosting](../../NeurIPS2025/others/revisiting_agnostic_boosting.md)
- [\[CVPR 2026\] OmniFood8K: Single-Image Nutrition Estimation via Hierarchical Frequency-Aligned Fusion](../../CVPR2026/others/omnifood8k_nutrition_estimation.md)
- [\[ICCV 2025\] Learning Visual Hierarchies in Hyperbolic Space for Image Retrieval](learning_visual_hierarchies_in_hyperbolic_space_for_image_retrieval.md)

</div>

<!-- RELATED:END -->
