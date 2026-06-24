---
title: >-
  [Paper Note] Rethinking Image Super-Resolution from Training Data Perspectives
description: >-
  [ECCV 2024][Image Restoration][Super-Resolution] This paper rethinks image super-resolution (SR) from the perspective of training data. It proposes an automated data evaluation pipeline to construct the DiverSeg dataset, which features low-resolution but high-quality and object-diverse images. The authors demonstrate that SR models trained on this dataset can outperform those trained on traditional high-resolution datasets (such as DF2K and LSDIR).
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "Super-Resolution"
  - "Training Data"
  - "Dataset Construction"
  - "Image Quality Assessment"
  - "Object Diversity"
date: 2026-05-08
content_hash: f2503f1d1e90f21a
---

# Rethinking Image Super-Resolution from Training Data Perspectives

**Conference**: ECCV 2024  
**arXiv**: [2409.00768](https://arxiv.org/abs/2409.00768)  
**Code**: [https://github.com/gohtanii/DiverSeg-dataset](https://github.com/gohtanii/DiverSeg-dataset)  
**Area**: Image Restoration  
**Keywords**: Super-Resolution, Training Data, Dataset Construction, Image Quality Assessment, Object Diversity

## TL;DR

This paper rethinks image super-resolution (SR) from the perspective of training data. It proposes an automated data evaluation pipeline to construct the DiverSeg dataset, which features low-resolution but high-quality and object-diverse images. The authors demonstrate that SR models trained on this dataset can outperform those trained on traditional high-resolution datasets (such as DF2K and LSDIR).

## Background & Motivation

The field of image super-resolution (SR) has achieved significant progress over the past decade, yet research has primarily focused on updating network architectures. From a training data perspective, conventional methods rely on high-resolution datasets like DIV2K and Flickr2K (collectively referred to as DF2K). Recently, LSDIR further expanded the data scale to 84,991 high-resolution images.

Two core criteria established for existing dataset constructions are:

**Resolution and Quality**: Demanding HD/2K/4K high resolution, and manually excluding compression artifacts.

**Diversity**: Containing various scenes, illuminations, and textures.

**Key Challenge**: Collecting uncompressed, high-resolution images is both difficult and expensive, making it hard to scale datasets up. For instance, ImageNet contains 1.28 million images but includes low-quality and JPEG-compressed images.

**Core Problem**: What criteria does training data actually require? Is high resolution truly necessary?

**Key Findings**: Three factors positively influence SR performance: (i) low compression artifacts, (ii) high intra-image diversity (more objects), and (iii) large-scale datasets. Low-resolution images that satisfy these conditions can even outperform high-resolution data during SR training.

## Method

### Overall Architecture

An automated image evaluation pipeline is proposed to filter and construct the SR training dataset DiverSeg from large-scale, low-resolution datasets (ImageNet, Places365, and PASS). The pipeline consists of two stages: **Source Selection** and **Object-based Filtering**.

### Key Designs

1. **Source Selection — Blockiness Distribution-based Quality Estimation**:

    - **Mechanism**: Screen high-quality data sources by estimating the JPEG compression quality of datasets.
    - Calculate the blockiness value $B(x)$ for each image using a **blockiness metric**, and obtain the dataset-level blockiness distribution $p_{X,q}(b)$ through kernel density estimation.
    - Blockiness is quantified using variations in subband DCT coefficients:
    $$B(x) = \sum_{i=1}^{P}\sum_{j=1}^{P}\left|\frac{\bar{V}_{crop}(i,j) - \bar{V}(i,j)}{\bar{V}(i,j)}\right|$$
    - Compare the distribution of the target dataset with the baseline distribution of a reference dataset (DF2K) at different quality levels using KL divergence to estimate dataset quality:
    $$\hat{q}_X = \sum_{q \in S} q \frac{\exp(-D_{KL}(p_{X,1.0} || p_{Z,q}))}{\sum_{q' \in S} \exp(-D_{KL}(p_{X,1.0} || p_{Z,q'}))}$$
    - Results: ImageNet quality is estimated at 95.5%, Places365 at 75.0% (filtered out), and PASS at 99.8%.
    - **Design Motivation**: Traditional methods require point-by-point manual checks of image quality, whereas this method automatically estimates the quality of the entire dataset based on statistical distributions, avoiding manual assessment.

2. **Object-based Filtering — Image Diversity Screening**:

    - **Core Hypothesis**: Images containing more object regions are more effective for SR training.
    - Two filtering methods:
        - **Segmentation-based filtering**: Uses SAM (ViT-H) to calculate the number of segmentation masks $R(x)$, with a threshold of $\theta = 100$, filtering out 260K images from ImageNet.
        - **Detection-based filtering**: Uses Detic (ViT-B) to calculate the number of detected objects $R(x)$, with a threshold of $\theta = 18$, also yielding 260K images.
    - **Design Motivation**: During manual quality evaluation, human annotators tend to favor detail-rich images, which implicitly filters out images with few objects. This method explicitly formulates this implicit preference.

3. **DiverSeg Dataset**:

    - DiverSeg-I: 259K images filtered from ImageNet.
    - DiverSeg-P: 267K images filtered from PASS.
    - DiverSeg-IP: A combination of both, totaling 527K images.
    - Characteristics: Low resolution (averaging 233K pixels vs. 2.8M for DF2K) but high quality (low blockiness) and high diversity (averaging 146 segmentation masks vs. 103 for DF2K).

### Loss & Training

The models are trained using the original configurations from their respective papers (MSRResNet, EDSR, RCAN, SwinIR, and HAT), with the only difference being the training dataset. Standard $L_1$ or $L_2$ losses are utilized. The core objective is to validate the impact of dataset quality rather than to modify the training strategy.

## Key Experimental Results

### Main Results

Comparison of $\times 4$ SR performance (PSNR/SSIM, on 5 benchmark datasets):

| Model | Training Data | Set5 | BSD100 | Urban100 | Manga109 |
|------|---------|------|--------|----------|----------|
| SwinIR | DF2K | 32.92/0.9044 | 27.92/0.7489 | 27.45/0.8254 | 32.03/0.9260 |
| SwinIR | LSDIR | 32.86/0.9036 | 27.92/0.7492 | 27.79/0.8331 | 31.98/0.9262 |
| SwinIR | **DiverSeg-I** | **32.97/0.9053** | **27.98/0.7508** | **27.83/0.8336** | **32.34/0.9283** |
| HAT | DF2K | 33.03/0.9056 | 27.99/0.7514 | 27.93/0.8365 | 32.44/0.9292 |
| HAT | LSDIR | 32.93/0.9053 | 28.01/0.7525 | 28.45/0.8469 | 32.57/0.9306 |
| HAT | **DiverSeg-I** | **33.15/0.9071** | **28.07/0.7542** | **28.51/0.8477** | **32.90/0.9325** |
| RCAN | DF2K | 32.50/0.8990 | 27.75/0.7421 | 26.73/0.8058 | 31.17/0.9165 |
| RCAN | **DiverSeg-I** | **32.70/0.9012** | **27.81/0.7443** | **27.03/0.8116** | **31.58/0.9210** |

### Ablation Study

| Configuration | Key Metric (Urban100 PSNR) | Description |
|------|---------|------|
| Full ImageNet (1.28M) | Lower | Contains a large number of low-quality compressed images |
| ImageNet Filtered (260K) | Gain | Performance improves after removing low-quality images |
| DiverSeg-I (260K, $\theta=100$) | **Best** | Dual filtering of high quality + high diversity |
| Places365 | Worst | Quality is only 75%, containing significant compression artifacts |
| PASS (1.44M) | Good | Quality is 99.8%, but diversity is insufficient |
| DiverSeg-P (267K) | Outperforms DF2K | Diversity improved after filtering from PASS |
| Threshold $\theta=0$ (No filtering) | Baseline | Compared to the full dataset |
| Threshold $\theta=100$ | **Best** | Sweet spot for object diversity filtering |

### Key Findings

- **High resolution is not required**: High-quality datasets with low resolution (~233K pixels) can outperform high-resolution datasets (DF2K ~2.8M pixels).
- **Compression artifacts are harmful**: The low quality (75.0%) of Places365 leads to the worst SR performance, demonstrating the negative impact of compression artifacts on SR training.
- **Object diversity is crucial**: More objects in an image $\rightarrow$ more textures and edges $\rightarrow$ better SR performance.
- **Scaling effect**: Under equal quality, larger quantities of images generally bring better performance.
- **DiverSeg-I outperforms DF2K across all 5 SR models**, proving effective for both CNN and Transformer architectures.

## Highlights & Insights

1. **Reversing traditional perceptions**: Proving that high-resolution images are not a necessity for SR training, which dramatically lowers the barrier to constructing SR datasets.
2. **Automated pipeline**: A fully automated dataset construction workflow that eliminates time-consuming manual quality assessments.
3. **Blockiness-based quality estimation**: Cleverly leveraging KL divergence to compare blockiness distributions and estimate dataset quality, avoiding frame-by-frame analysis.
4. **High generalizability**: The method is applicable to all tested SR models (3 CNNs + 2 Transformers), without relying on specific architectures.
5. **Significant practical value**: It enables effortless automated screening of SR training data from any large-scale image dataset in the future.

## Limitations & Future Work

- The threshold for object filtering ($\theta = 100/18$) is manually set, without an automated selection strategy.
- Validated only on $\times 4$ SR, lacking coverage of other scales such as $\times 2$ and $\times 8$.
- The evaluation models (SAM and Detic) themselves incur high computational overhead, requiring substantial resources to process million-scale datasets.
- The impact of semantic category distributions (e.g., natural scenes vs. urban scenes) on SR has not been analyzed.
- The interaction effects with data augmentation strategies have not been investigated.

## Related Work & Insights

- Provides a deeper understanding of the role of ImageNet pre-training in SR (e.g., HAT uses ImageNet pre-training, but it might not be the optimal SR data).
- The blockiness-based quality estimation method can be generalized to dataset construction for other low-level vision tasks.
- The finding of "object diversity" implies that SR models may benefit more from rich local texture patterns rather than global high resolution.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Outlier-Aware Post-Training Quantization for Image Super-Resolution](../../ICCV2025/image_restoration/outlier-aware_post-training_quantization_for_image_super-resolution.md)
- [\[ECCV 2024\] Accelerating Image Super-Resolution Networks with Pixel-Level Classification](accelerating_image_super-resolution_networks_with_pixel-level_classification.md)
- [\[ECCV 2024\] Overcoming Distribution Mismatch in Quantizing Image Super-Resolution Networks](overcoming_distribution_mismatch_in_quantizing_image_super-resolution_networks.md)
- [\[CVPR 2026\] FoundIR-v2: Optimizing Pre-Training Data Mixtures for Image Restoration Foundation Model](../../CVPR2026/image_restoration/foundir-v2_optimizing_pre-training_data_mixtures_for_image_restoration_foundatio.md)
- [\[ICCV 2025\] FoundIR: Unleashing Million-scale Training Data to Advance Foundation Models for Image Restoration](../../ICCV2025/image_restoration/foundir_unleashing_million-scale_training_data_to_advance_foundation_models_for_.md)

</div>

<!-- RELATED:END -->
