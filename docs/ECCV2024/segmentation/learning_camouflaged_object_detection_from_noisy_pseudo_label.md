---
title: >-
  [Paper Note] Learning Camouflaged Object Detection from Noisy Pseudo Label
description: >-
  [ECCV 2024][Segmentation][Camouflaged Object Detection] The first weakly semi-supervised camouflaged object detection method (WSSCOD) is proposed, achieving comparable performance to fully supervised SOTA using only 20% pixel-level annotations + 80% box annotations. The core contribution is an adaptive noise correction loss $\mathcal{L}_{NC}$, which can be optimized separately in the early learning and memorization phases.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Camouflaged Object Detection"
  - "Weakly Semi-Supervised Learning"
  - "Noisy Labels"
  - "Box Prompt"
  - "Noise Correction Loss"
date: 2026-05-08
content_hash: 6ddaee9d22678abb
---

# Learning Camouflaged Object Detection from Noisy Pseudo Label

**Conference**: ECCV 2024  
**arXiv**: [2407.13157](https://arxiv.org/abs/2407.13157)  
**Code**: [zhangjinCV/Noisy-COD](https://github.com/zhangjinCV/Noisy-COD)  
**Area**: Camouflaged Object Detection / Semantic Segmentation  
**Keywords**: Camouflaged Object Detection, Weakly Semi-Supervised Learning, Noisy Labels, Box Prompt, Noise Correction Loss

## TL;DR

The first weakly semi-supervised camouflaged object detection method (WSSCOD) is proposed, achieving comparable performance to fully supervised SOTA using only 20% pixel-level annotations + 80% box annotations. The core contribution is an adaptive noise correction loss $\mathcal{L}_{NC}$, which can be optimized separately in the early learning and memorization phases.

## Background & Motivation

Camouflaged Object Detection (COD) aims to segment targets highly integrated with the environment, which is highly time-consuming to annotate (about 60 minutes per image). Existing weakly supervised methods (point or scribble annotations) perform far design behind fully supervised methods because of the ambiguous visual boundaries between foreground and background. The authors observe that:
- **Sparse annotations** (points, scribbles) make it difficult for classifiers to distinguish camouflaged objects, leading to high false negatives.
- **Dense annotations** (boxes) cause obvious false positives but provide rich target location information.
- **Box annotations** can 1) mask complex backgrounds to reduce the degree of camouflage, and 2) indicate the rough location of targets to simplify search.

Therefore, the authors propose leveraging box annotations as prompts combined with a very small amount of pixel-level annotations, establishing an economical and efficient new training paradigm.

## Method

### Overall Architecture

WSSCOD is a two-stage method:
1. **Stage 1**: Train an auxiliary network (ANet) using $M$ fully annotated images and box annotations to generate pseudo labels for the remaining $N$ images.
2. **Stage 2**: Merge fully annotated images and pseudo-labeled images into $\mathcal{D}_t$, and train the primary network (PNet) using the noise correction loss $\mathcal{L}_{NC}$.

The ratio of $M$ is set to $\{1\%, 5\%, 10\%, 20\%\}$; the corresponding models are named PNetF1, PNetF5, PNetF10, and PNetF20, respectively.

### Key Designs

1. **Auxiliary Network ANet (Dual-branch Encoder + Reverse Fusion Decoder)**:

    - **Dual-branch Encoder**: Uses two ConvNeXt-B encoders to encode the original image $x_m$ and the box proposal $\tilde{b}_m = x_m \cdot b_m$ separately, extracting complementary multi-scale features.
    - **Frequency Transformer**: Utilizes Discrete Wavelet Transform (DWT) to extract low-frequency and high-frequency components. High-frequency components are fused with shallow-layer features to capture details, while low-frequency components are fused with deep-layer features to enhance semantics.
    - **Reverse Fusion Decoder**: UNet-style multi-layer feature fusion, introducing a reverse mask $Rev(p) = -\sigma(p) + 1$ to correlate background with hard areas, magnifying their difference from correct pixels.
    - **Design Motivation**: Information inside the box bounding is not always reliable; hence, the original image and box proposals must complement each other. Additionally, frequency-domain information can reveal subtler target structures in camouflaged scenes.

2. **Primary Network PNet (Single-branch Structure)**:

    - Retains modules of ANet but uses only the image branch, replacing the encoder with a stronger PVTv2-B4.
    - Requires only the input image during inference, without requiring box annotations.
    - $p_{pn}^k = \Pi(\Phi_t(\mathbf{F}_t^k), ASPP(\mathbf{F}_t^4))$

3. **Noise Correction Loss $\mathcal{L}_{NC}$**:

    - Core formula: $\mathcal{L}_{NC} = \frac{\sum_{i=1}^{H \times W} |p_i - g_i|^q}{\sum_{i=1}^{H \times W}(p_i + g_i) - \sum_{i=1}^{H \times W} p_i \cdot g_i}$
    - **Early Learning Phase** ($q=2$): Equivalent to IoU-form loss, accelerating model convergence to correct pixels.
    - **Memorization Phase** ($q=1$): Shifts to MAE-form loss. The gradient magnitude is identical for every pixel ($\frac{\partial \mathcal{L}_{NC}}{\partial p_i} = \frac{sign(p_i - g_i)}{分母}$), preventing the loss from being dominated by noisy pixels.
    - Compared to pure MAE: $\mathcal{L}_{NC}$ is area-dependent, leveraging spatial correlation between pixels to achieve faster and better convergence.
    - Tolerates up to a 50% noise rate.
    - **Design Motivation**: Traditional CE and IoU losses are more sensitive to hard pixels, which is beneficial for clean labels but leads to severe misguidance when training on noisy labels.

### Loss & Training

- Both ANet and PNet are trained for 100 epochs.
- The switching point of $q$ changes with the noise rate: PNetF1 / F5 / F10 / F20 switch from $q=2$ to $q=1$ at the 20th / 20th / 40th / 60th epoch, respectively.
- Besides $\mathcal{L}_{NC}$, DICE loss is also utilized to assist in boundary learning.
- Adam optimizer with an initial lr of 1e-7, linearly warmed up to 1e-4, then cosine annealed.
- Image augmentation: Random cropping, blurring, brightness adjustment, flipping, and resizing to 384×384.
- All random seeds are fixed to 2024.

## Key Experimental Results

### Main Results

| Method | Annotation | CAMO $F_\beta$↑ | COD10K $F_\beta$↑ | NC4K $F_\beta$↑ | CHAMELEON $F_\beta$↑ |
|------|--------|---------|----------|---------|-----------|
| SCWS (Weakly Supervised) | Scribble 100% | 0.651 | 0.644 | 0.713 | 0.721 |
| CamoFormer (Fully Supervised) | Pixel 100% | 0.854 | 0.811 | 0.868 | 0.880 |
| **PNetF1** | **Pixel 1% + Box 99%** | **0.835** | **0.745** | **0.831** | **0.812** |
| **PNetF20** | **Pixel 20% + Box 80%** | **0.856** | **0.792** | **0.857** | **0.861** |
| PNet†F20 | Pixel 20% + Box 240% | 0.870 | 0.857 | 0.888 | 0.886 |

### Ablation Study

| Configuration | $F_\beta$↑ | $S_\alpha$↑ | Description |
|------|---------|---------|------|
| CE + IoU | 0.780 | 0.844 | Traditional losses are sensitive to noise |
| $\mathcal{L}_{NC}^{q=2.0}$ (IoU-form only) | 0.778 | 0.849 | No noise correction |
| $\mathcal{L}_{NC}^{q=1.0}$ (MAE-form only) | 0.780 | 0.855 | Insufficient convergence capability |
| GCE | 0.759 | 0.835 | Classification noise methods are not applicable to segmentation |
| **$\mathcal{L}_{NC}$ (Full)** | **0.792** | **0.860** | **Optimal two-stage adaptation** |

Ablation of Box Annotation vs Other Prompts: Box annotation improves performance by 7.2% compared to scribbles, and by 14.5% compared to no prompts (measured by $F_\beta$ index).

### Key Findings

- Utilizing only 1% full annotation (40 images) exceeds all weakly supervised methods.
- Using 20% full annotation achieves performance comparable to fully supervised SOTA (gap < 1%).
- $\mathcal{L}_{NC}$ also improves fully supervised methods (SINetv2 $F_\beta$ +2.1%, SCOD $F_\beta$ +5.9%).
- WSSCOD is scalable: continuously improving performance by incorporating only additional box annotations.

## Highlights & Insights

- Introduces the "early learning-memorization" phenomenon of noise label learning into pixel-level segmentation tasks and designs a targeted stage-wise loss-switching strategy.
- The unified formula of $\mathcal{L}_{NC}$ is elegant, smoothly transitioning between IoU-form and MAE-form via a single parameter $q$.
- Box annotation possesses unique advantages in camouflaged scenarios: it can directly mask complex backgrounds and reduce the level of camouflage.
- Confirms the universality of $\mathcal{L}_{NC}$: directly replacing loss functions in existing methods yields improvements.

## Limitations & Future Work

- Accuracy of box annotations has a certain impact on final results (similar to multimodal bias issues).
- Dual-branch fusion utilizes only simple channel concatenation; a better fusion strategy could yield further improvements.
- WSSCOD is a two-stage pipeline (training ANet followed by training PNet), which is relatively tedious.
- The possibility of using foundation models such as SAM as ANet has not been explored.

## Related Work & Insights

- Comparison with SAM: Even with box/point prompts, SAM lags significantly behind the proposed method, demonstrating that general foundation segmentation models lack specialized capabilities in camouflaged scenarios.
- Porting noisy label learning from classification tasks to pixel-level segmentation reveals fundamental differences: in segmentation, every image contains noisy pixels that exhibit spatial correlation.
- Extending $\mathcal{L}_{NC}$ to areas such as medical image segmentation where annotation noise is similarly severe could be considered.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first weakly semi-supervised COD method; the stage-wise design of $\mathcal{L}_{NC}$ is simple and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Competes with 16 SOTA methods across 4 datasets, providing thorough ablations (box types, loss functions, training strategies).
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, solid mathematical derivation, and rich illustrations.
- **Value**: ⭐⭐⭐⭐ — Significantly reduces COD annotation costs, with $\mathcal{L}_{NC}$ holding generalizable value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Frequency-Spatial Entanglement Learning for Camouflaged Object Detection](frequency-spatial_entanglement_learning_for_camouflaged_object_detection.md)
- [\[CVPR 2026\] EReCu: Pseudo-label Evolution Fusion and Refinement with Multi-Cue Learning for Unsupervised Camouflage Detection](../../CVPR2026/segmentation/erecu_pseudolabel_evolution_unsupervised_camouflage.md)
- [\[NeurIPS 2025\] Towards Robust Pseudo-Label Learning in Semantic Segmentation: An Encoding Perspective](../../NeurIPS2025/segmentation/towards_robust_pseudo-label_learning_in_semantic_segmentation_an_encoding_perspe.md)
- [\[CVPR 2026\] Beyond Appearance: Camouflaged Object Detection via Geometric Structure](../../CVPR2026/segmentation/beyond_appearance_camouflaged_object_detection_via_geometric_structure.md)
- [\[CVPR 2026\] SDDF: Specificity-Driven Dynamic Focusing for Open-Vocabulary Camouflaged Object Detection](../../CVPR2026/segmentation/sddf_specificity-driven_dynamic_focusing_for_open-vocabulary_camouflaged_object.md)

</div>

<!-- RELATED:END -->
