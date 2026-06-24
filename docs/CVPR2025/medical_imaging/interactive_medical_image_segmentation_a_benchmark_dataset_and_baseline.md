---
title: >-
  [Paper Note] Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline
description: >-
  [CVPR 2025][Medical Imaging][Interactive medical image segmentation] This paper proposes IMed-361M, a large-scale interactive medical image segmentation benchmark dataset containing 6.4 million images and 361 million masks (averaging 56 masks per image) covering 14 imaging modalities and 204 segmentation targets. Based on this, the authors develop an IMIS baseline network that supports click, bounding box, text, and combined interactions, which outperforms existing vision fou…
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Interactive medical image segmentation"
  - "large-scale dataset"
  - "SAM fine-tuning"
  - "dense annotation"
  - "multimodal medical imaging"
date: 2026-05-08
content_hash: 41618d712a0d3e72
---

# Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline

**Conference**: CVPR 2025  
**arXiv**: [2411.12814](https://arxiv.org/abs/2411.12814)  
**Code**: [https://github.com/uni-medical/IMIS-Bench](https://github.com/uni-medical/IMIS-Bench)  
**Area**: Medical Image / Interactive Segmentation  
**Keywords**: Interactive medical image segmentation, large-scale dataset, SAM fine-tuning, dense annotation, multimodal medical imaging

## TL;DR
This paper proposes IMed-361M, a large-scale interactive medical image segmentation benchmark dataset containing 6.4 million images and 361 million masks (averaging 56 masks per image) covering 14 imaging modalities and 204 segmentation targets. Based on this, the authors develop an IMIS baseline network that supports click, bounding box, text, and combined interactions, which outperforms existing vision foundation models across multiple scenarios.

## Background & Motivation

1. **Background**: SAM and its derivatives (MedSAM, SAM-Med2D) have advanced interactive segmentation, but their "segment everything" capability in medical imaging remains limited. While the natural image dataset SA-1B contains an average of 100 masks per image, medical datasets like COSMOS average only 5.7.
2. **Limitations of Prior Work**: (a) Existing medical segmentation datasets are sparsely annotated, with only a few masks per image, which limits dense segmentation and fine-grained interaction capabilities; (b) Datasets are typically confined to specific modalities or tasks, resulting in poor generalization; (c) The evaluation of different IMIS methods lacks a uniform standard, leading to poor comparability and reliability.
3. **Key Challenge**: Medical image foundation models require large-scale, densely annotated datasets similar to SA-1B, yet medical annotation costs are extremely high, and the annotation density of existing consolidated datasets is highly insufficient.
4. **Goal**: (a) Build a large-scale, diverse, and densely annotated IMIS benchmark dataset; (b) Establish a unified baseline model supporting multiple interaction strategies; (c) Provide a fair and consistent evaluation framework for IMIS models.
5. **Key Insight**: Leverage SAM's object identification capability to automatically generate dense interactive masks, and employ ground truth (GT)-guided quality control to ensure annotation quality.
6. **Core Idea**: Consolidate 110+ public datasets and use SAM to automatically generate high-density interactive annotations (56 masks per image) to construct a benchmark dataset with 361 million masks across 14 modalities, followed by training a unified multi-strategy interactive segmentation baseline.

## Method

### Overall Architecture
Data construction pipeline: Collect 110+ datasets $\rightarrow$ Standardized preprocessing $\rightarrow$ Conflict disambiguation $\rightarrow$ Automatic SAM generation of interactive masks $\rightarrow$ GT-guided quality control $\rightarrow$ Final 6.4 million images + 361 million masks. The model architecture adopts a ViT-Base encoder + multi-strategy prompt encoder (points + bounding boxes + text) + Transformer decoder, trained end-to-end via an interactive simulation strategy.

### Key Designs

1. **Automatic Mask Generation + GT-guided Quality Control**:

    - **Function**: Generate dense, high-quality interactive masks for each image.
    - **Mechanism**: Candidate masks are generated using SAM's 32×32 grid of points and filtered through a triple screening process: keeping those with IoU confidence > 0.85, NMS (IoU > 0.7 duplication removal), and removing background masks with coverage > 80%. A critical quality control step uses original GT to correct the generated masks: for multi-connected regions in the GT, the corresponding generated masks are directly replaced; for single-connected regions, if the minimal bounding box of the generated mask overlaps with the GT by > 95%, the generated mask is retained, otherwise, it is replaced by the GT. Finally, morphological operations are applied for denoising and hole-filling.
    - **Design Motivation**: SAM's automatically generated masks often fail to correctly separate structures with ambiguous boundaries (e.g., atria and ventricles of the heart), and scattered structures (e.g., intestines) are often identified as multiple separate objects. The GT-guided correction resolves these specific medical segmentation granularity issues.

2. **Multi-strategy Prompt Encoder**:

    - **Function**: Support click, bounding box, text inputs, and their combinations.
    - **Mechanism**: Points and bounding boxes are represented by positional encodings plus learnable embeddings, while text is encoded via a CLIP text encoder using templates like "A segmentation area of a [category]", covering over 200 organ and lesion categories. The three types of prompts can be combined freely—text provides global semantic guidance, while clicks or boxes provide local spatial localization.
    - **Design Motivation**: Prior methods typically support only a single interaction strategy, which fails to evaluate the impact of different interaction modalities. A unified multi-strategy design provides a benchmark for future multimodal segmentation research.

3. **Simulated Continuous Interaction Training Strategy**:

    - **Function**: Train the model to progressively improve segmentation results through multi-round interactions.
    - **Mechanism**: For each image and target mask, an initial interaction is simulated (random foreground sampling point, target bounding box + 5px offset) to yield an initial prediction. Then, $K=8$ iterations are performed: corrected interactions (new positive/negative clicks) are generated based on the erroneous regions between prediction and GT, while the low-resolution prediction mask from the previous round is provided as an additional cue. The image encoder is run only once, and subsequent iterations update only the parameters of the prompt encoder and decoder.
    - **Design Motivation**: In real clinical workflows, clinicians progressively correct segmentation results through multiple clicks. Simulating this process enables the model to learn from errors and improves interaction efficiency.

### Loss & Training
A linear combination of Focal Loss + Dice Loss is utilized (ratio of 20:1). Training uses the Adam optimizer with a learning rate of $2 \times 10^{-5}$ on 72 RTX 4090 GPUs with a batch size of 2. For each image, 5 targets are randomly selected (with replacement if insufficient). Images are standardized to 1024×1024, with pixel values randomly scaled and shifted with a 20% probability. The training spans 12 epochs. Performance is evaluated using the Dice score.

## Key Experimental Results

### Main Results

**Evaluation on External Datasets (bbox prompt):**

| Dataset | SAM | SAM-2 | MedSAM | SAM-Med2D | **IMIS-Net** |
|--------|-----|-------|--------|-----------|-------------|
| ISLES (Stroke) | 55.92 | 60.14 | 59.90 | 68.22 | **71.78** |
| SegThor (Avg) | 84.46 | 85.86 | 60.55 | 86.43 | **89.27** |
| TotalSeg MRI (Avg) | 75.45 | 77.62 | 59.52 | 75.92 | **79.06** |

**For the internal test set (image-level/mask-level statistics), IMIS-Net achieves the best performance under both click and bbox interactions.**

### Ablation Study

| Decoder Dim | Image Res | Click Dice | Box Dice | Trainable Params |
|-----------|----------|----------|---------|-----------|
| 768 | 256×256 | 0.8214 | 0.8469 | 29.68M |
| 768 | 512×512 | 0.8673 | 0.8968 | 29.68M |
| 256 | 1024×1024 | 0.8366 | 0.8497 | 5.52M |
| 512 | 1024×1024 | 0.8563 | 0.8729 | 15.19M |
| **768** | **1024×1024** | **0.8848** | **0.9060** | 29.68M |

**Text + Combined Prompts Performance:**

| Interaction Strategy | Dice |
|---------|------|
| Text Only | 76.30% |
| Text + Click | 88.25% (+11.95%) |
| Text + Click + 3 rounds correction | 89.69% |

### Key Findings
- Dense masks are crucial for performance: training solely on GT yields sub-optimal performance, while integrating interactive masks accelerates Dice improvements.
- Data scalability is significant: increasing training data volume consistently improves performance, demonstrating the scalability of the method.
- Input resolution has the greatest impact: scaling resolution from 256 to 1024 improves Dice from 84.69% to 90.60%.
- Increasing decoder dimension from 256 to 768 adds only 24.16M parameters but improves Dice by approximately 5.6%, indicating that model capacity becomes the bottleneck when data is sufficiently large.
- Bounding box interactions consistently outperform clicks (by providing more boundary information); multi-round interactions progressively narrow the performance gap between different models.
- Clicks positioned closer to the centroid yield superior results, with SAM-2 showing a 2.84% Dice improvement; bounding box offsets lead to a performance drop of 0.85%–3.94% across all methods.

## Highlights & Insights
- **Using SAM to automatically generate dense annotations for medical images** is a clever strategy: it leverages the object perception capability of general-domain models to mitigate the scarcity of medical annotations, while maintaining quality through GT correction. This methodology can be transferred to other annotation-scarce professional domains.
- **GT-guided granularity management** addresses the core challenge of SAM in medical imaging: the substitution strategy for multi-connected regions and ambiguous boundaries is highly practical, reflecting a deep understanding of the unique characteristics of medical imaging.
- **Scaling analysis on data size + decoder capacity** offers clear practical guidance: even when model performance approaches saturation, expanding the decoder still yields improvements, charting a simple and effective optimization path for future work.

## Limitations & Future Work
- Some interactive masks lack direct clinical significance (e.g., background letters in X-rays, hair in dermoscopy images); although they increase diversity, they may introduce noise.
- Only the ViT-Base encoder is utilized; larger encoders could further improve performance.
- The semantic information of automatically generated masks is missing, and future exploration is needed to obtain semantic labels for interactive masks.
- 3D medical images are processed only as 2D slices without fully exploiting volumetric context.
- Quality evaluation indicates poor mask quality in 18 sub-datasets, which may still contain residual errors despite cleaning.

## Related Work & Insights
- **vs MedSAM**: MedSAM fine-tunes SAM on large-scale medical data but supports only bounding box interactions with insufficient annotation density. IMIS-Net supports multi-strategy interactions and increases mask density to 56/image vs. MedSAM's ~5/image.
- **vs SAM-Med2D**: SAM-Med2D lacks bone structure data, leading to poor segmentation of such anatomical structures. IMIS-Net resolves this bias through more comprehensive data coverage.
- **vs SAM/SAM-2**: Models pre-trained on natural images lag significantly behind on medical images (with single-click Dice at only ~60%), demonstrating the necessity of domain adaptation.

## Rating
- Novelty: ⭐⭐⭐⭐ Dataset construction methodology is innovative (SAM automatic generation + GT correction), though the model architecture mostly adheres to standard SAM fine-tuning.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-dimensional evaluation across modalities, anatomical structures, interaction strategies, click positions, and box offsets; external dataset verification is highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The dataset construction process is clearly described, though some sections could be more concise.
- Value: ⭐⭐⭐⭐⭐ The benchmark dataset with 361 million masks will serve as a vital resource for medical imaging foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Interactive Medical Image Analysis with Concept-based Similarity Reasoning](interactive_medical_image_analysis_with_concept-based_similarity_reasoning.md)
- [\[CVPR 2025\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [\[CVPR 2025\] Show and Segment: Universal Medical Image Segmentation via In-Context Learning](show_and_segment_universal_medical_image_segmentation_via_in-context_learning.md)
- [\[CVPR 2025\] Revisiting MAE Pre-Training for 3D Medical Image Segmentation](revisiting_mae_pre-training_for_3d_medical_image_segmentation.md)
- [\[CVPR 2025\] Noise-Consistent Siamese-Diffusion for Medical Image Synthesis and Segmentation](noise-consistent_siamese-diffusion_for_medical_image_synthesis_and_segmentation.md)

</div>

<!-- RELATED:END -->
