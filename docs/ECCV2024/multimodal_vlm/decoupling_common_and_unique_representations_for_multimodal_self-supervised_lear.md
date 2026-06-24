---
title: >-
  [Paper Note] Decoupling Common and Unique Representations for Multimodal Self-supervised Learning
description: >-
  [ECCV2024][Multimodal VLM][Multimodal self-supervised learning] Proposes DeCUR, which explicitly splits embedding dimensions into cross-modal common and modality-unique parts in multimodal self-supervised learning. Alignment and decoupling are driven by the cross-correlation matrix, respectively. Intramodal training is introduced to ensure that the unique dimensions learn meaningful information. DeCUR outperforms baselines like Barlow Twins and CLIP in three multimodal scenar…
tags:
  - "ECCV2024"
  - "Multimodal VLM"
  - "Multimodal self-supervised learning"
  - "representation decoupling"
  - "redundancy reduction"
  - "Barlow Twins"
  - "deformable attention"
date: 2026-05-08
content_hash: 3b4b703f9e9a2706
---

# Decoupling Common and Unique Representations for Multimodal Self-supervised Learning

**Conference**: ECCV2024  
**arXiv**: [2309.05300](https://arxiv.org/abs/2309.05300)  
**Code**: [zhu-xlab/DeCUR](https://github.com/zhu-xlab/DeCUR)  
**Area**: Other  
**Keywords**: Multimodal self-supervised learning, representation decoupling, redundancy reduction, Barlow Twins, deformable attention  

## TL;DR

Proposes DeCUR, which explicitly splits embedding dimensions into cross-modal common and modality-unique parts in multimodal self-supervised learning. Alignment and decoupling are driven by the cross-correlation matrix, respectively. Intramodal training is introduced to ensure that the unique dimensions learn meaningful information. DeCUR outperforms baselines like Barlow Twins and CLIP in three multimodal scenarios: SAR-Optical, RGB-DEM, and RGB-Depth.

## Background & Motivation

### Background

Multimodal self-supervised learning represents a rapidly developing field. Mainstream approaches (e.g., SimCLR-cross, CLIP, ImageBind) treat different modalities as augmented views of the same scene and perform cross-modal contrastive learning in a shared latent space.

### Limitations of Prior Work

**Learning only commonality, ignoring uniqueness**: Existing methods bet all embedding dimensions on cross-modal common features, forcing potentially orthogonal, modality-unique information into the same representation space, which limits the models' in-depth understanding of each modality.

**Lack of intra-modal training**: Cross-modal alignment cannot guarantee representation quality within a single modality. When a modality is missing (e.g., optical images obscured by clouds), single-modality performance drops significantly.

**Dependency on negative samples**: Methods like SimCLR/CLIP require a large number of negative samples, resulting in heavy training overhead and high sensitivity to batch size.

### Key Challenge

Cross-modal alignment requires cross-modal similarity, but each modality naturally possesses unique information that other modalities cannot provide (e.g., SAR can penetrate clouds, Depth contains geometric structures). These two objectives conflict with each other.

### Key Insight

Starting from the redundancy-reduction framework of Barlow Twins, the embedding dimensions are naturally split into common and unique groups, with different optimization objectives applied to each. This achieves joint alignment of common information and decoupling of unique information without requiring negative samples.

### Core Idea

> Explicitly split embedding dimensions into $K_c$ (common) and $K_u$ (unique), where the cross-correlation matrix of cross-modal common dimensions is pushed toward the identity matrix (alignment), and the cross-correlation matrix of unique dimensions is pushed toward the zero matrix (decoupling). This is combined with intra-modal Barlow Twins training to prevent unique dimensions from collapsing.

## Method

### Overall Architecture

DeCUR uses a dual-encoder, dual-projector joint-embedding architecture:

1. Each modality has an independent Encoder + MLP Projector (3 layers, output dimension 8192).
2. Each modality generates embedding vectors for two augmented views.
3. Embedding dimensions are proportionally divided into **common** and **unique** parts.
4. Cross-modal: common dimensions' cross-correlation $\to$ identity matrix; unique dimensions' cross-correlation $\to$ zero matrix.
5. Intra-modal: full-dimension Barlow Twins training (performed for each modality).
6. (Optional) Add Residual Deformable Attention (RDA) in the last two stages of the ConvNet encoders.

### Key Designs

| Component | Design Details | Function |
|------|---------|------|
| Embedding Split | $K = K_c + K_u$, with common ratio of 87.5% for SAR-Optical and 75% for RGB-DEM/Depth | Controls the capacity allocation for common and unique representations |
| Cross-modal common loss $L_{com}$ | Cross-correlation matrix: diagonal $\to 1$, non-diagonal $\to 0$ (same as Barlow Twins) | Cross-modal alignment + redundancy reduction |
| Cross-modal unique loss $L_{uni}$ | Cross-correlation matrix **all $\to 0$** (including diagonal elements) | Forces decorrelation of unique dimensions across modalities |
| Intra-modal loss $L_{M1} / L_{M2}$ | Full-dimension Barlow Twins for two augmented views of the same modality | Prevents unique dimensions from collapsing, enhancing unimodal representations |
| Deformable Attention (RDA) | DAT++ in the last two stages of ResNet-50 + residual connections | Data-driven focus on key regions within modalities |
| Batch Normalization | Mean-centering the embeddings along the batch dimension | Stabilizes the computation of cross-correlation matrices |

### Loss & Training

**Total Loss**:

$$\mathcal{L} = \mathcal{L}_{com} + \mathcal{L}_{uni} + \mathcal{L}_{M1} + \mathcal{L}_{M2}$$

Each term balances the invariance term and redundancy-reduction term via a trade-off coefficient $\lambda$, which defaults to 0.0051.

**Training Strategy**:

| Parameter | SAR-Optical / RGB-DEM | RGB-Depth |
|------|----------------------|-----------|
| Epochs | 100 | 200 |
| Batch size | 256 | 128 |
| GPU | 4× NVIDIA A100 | 4× NVIDIA A100 |
| Backbone | ResNet-50 | ResNet-50 / MiT-B2/B5 |
| Projector dimension | 8192 | 8192 |
| Training duration | SAR-opt 35h / GeoNRW 6h | SUN-RGBD 6h |

## Key Experimental Results

### Main Results

**SAR-Optical Scene Classification (BigEarthNet-MM, mAP)**

| Method | Multimodal 1% | Multimodal 100% | SAR-only 1% | SAR-only 100% |
|------|----------|------------|-------------|--------------|
| SimCLR-cross | 77.4/78.7 | 82.8/89.6 | 68.1/70.4 | 71.7/83.7 |
| CLIP | 77.4/78.7 | 82.8/89.6 | 68.0/70.2 | 71.7/83.4 |
| Barlow Twins | 78.7/80.3 | 83.2/89.5 | 72.3/73.7 | 77.8/83.6 |
| **DeCUR** | **79.8/81.5** | **86.2/89.8** | **74.4/76.0** | **79.5/84.0** |

> Format: linear-probing / fine-tuning

**RGB-DEM Semantic Segmentation (GeoNRW, mIoU)**

| Method | Multimodal 1% frozen/FT | Multimodal 100% frozen/FT | RGB-only 1% frozen/FT | RGB-only 100% frozen/FT |
|------|------|------|------|------|
| SimCLR-cross | 23.0/30.2 | 35.2/47.3 | 20.1/25.9 | 29.6/42.5 |
| Barlow Twins | 31.2/33.6 | 43.0/48.4 | 29.4/33.4 | 38.0/45.9 |
| **DeCUR** | **34.7/36.6** | **44.7/48.9** | **32.2/35.7** | **40.8/46.7** |

**RGB-Depth Semantic Segmentation (SUN-RGBD / NYUDv2, mIoU)**

| Model | SUN-RGBD mIoU | NYUDv2 mIoU |
|------|--------------|-------------|
| FCN (CLIP) | 30.5 | 30.4 |
| FCN (DeCUR) | **34.5** (+4.0) | **31.2** (+0.8) |
| CMX-B2 | 49.7 | - |
| CMX-B2 (DeCUR) | **50.6** (+0.9) | - |
| CMX-B5 | - | 56.9 |
| CMX-B5 (DeCUR) | - | **57.3** (+0.4) |

### Ablation Study

**Loss Terms Ablation (1% labels fine-tuning)**

| Configuration | SAR-optical (mAP) | RGB-DEM (mIoU) |
|------|-------------------|----------------|
| DeCUR (Full) | **81.7** | **36.9** |
| w/o Intra-modal & w/o Decoupling (Pure cross-modal BT) | 80.3 | 33.6 |
| w/o Intra-modal training (Decoupling only) | 80.1 | 34.3 |
| w/o Decoupling (Intra-modal only) | 81.1 | 35.2 |

**Deformable Attention Ablation (frozen encoder)**

| Configuration | BigEarthNet-MM 1%/100% | GeoNRW-MM 1%/100% |
|------|----------------------|-------------------|
| w/o DA | 79.4/85.4 | 34.9/43.9 |
| w/ DA (w/o residual) | −0.1/− | −0.6/− |
| w/ RDA (w/ residual) | **+0.4/+0.8** | **−0.2/+0.8** |

### Key Findings

1. **Decoupling and intra-modal training are both indispensable**: Decoupling alone without intra-modal training causes the unique dimensions to collapse to random values, leading to unstable downstream performance.
2. **The ratio of common dimensions varies by modality**: The optimal common dimension ratio is 87.5% for SAR-Optical and 75% for RGB-DEM/Depth, which aligns with domain intuition (where DEM possesses more unique information).
3. **The common ratio is robust to embedding dimensions**: The optimal ratio remains consistent under both 512 and 8192 dimensions.
4. **Residual connections are crucial for deformable attention**: Deformable Attention (DA) without residual connections degrades performance in few-label scenarios instead.
5. **Significant advantage in scenarios with missing modalities**: SAR-only achieves a 2.0–3.2% improvement compared to Barlow Twins (BT) SAR, demonstrating that joint pre-training assists unimodal understanding.

## Highlights & Insights

1. **Simple yet effective design**: It only requires dividing dimensions based on Barlow Twins and modifying a single line of the loss objective (diagonal $\to 0$), with zero external dependencies.
2. **Excellent interpretability analysis**: GradCAM and Integrated Gradients confirm that the spatial saliency of unique dimensions is indeed more orthogonal, and spectral saliency aligns with domain knowledge (e.g., Near-Infrared is important, while water vapor/cirrus bands are not).
3. **Deformable attention visualization**: The optical model learns to ignore clouds, whereas the SAR model focuses on cloud areas because radar can penetrate clouds.
4. **Good cross-architecture generalization**: Improvements are achieved on both ResNet-50 and MiT-B2/B5, and the pre-trained weights can be directly migrated to the SOTA supervised model CMX.
5. **Embedding space sparsity**: Performance does not decrease significantly even when decoupled to 50% unique dimensions, suggesting a large amount of redundancy exists in the common space.

## Limitations & Future Work

1. **Global fixed ratio**: The entire dataset shares the same common/unique ratio without considering variation in the amount of modality-unique information across different samples (e.g., scenes with heavy cloud cover should have a higher ratio of unique information).
2. **Grid search required for the optimal ratio**: Search costs on large datasets are high. Although ~80% is generally effective, an adaptive strategy is lacking.
3. **Limited to dual modalities**: The current framework has not been extended to joint decoupling for three or more modalities.
4. **Limited downstream tasks**: Verification is mainly conducted on classification and semantic segmentation, lacking evaluation on tasks like detection or generation.
5. **No comparison with generative methods like MAE**: Unimodal MAE or MultiMAE are not included in the baselines.

## Related Work & Insights

| Related Method | Relationship |
|---------|------|
| Barlow Twins | Direct upstream of DeCUR. DeCUR = Multimodal version + Dimension decoupling |
| VICReg | Also falls under redundancy reduction. DeCUR outperforms VICReg in all scenarios |
| CLIP / SimCLR-cross | Cross-modal contrastive learning baselines; rely on negative samples, whereas DeCUR does not |
| FactorCL | Concurrent work decomposing common/unique features, but uses modality-specific augmentations. DeCUR is simpler as it operates directly on embedding dimensions |
| ImageBind | Multimodal joint embedding. DeCUR's decoupling approach can be extended to the shared space of ImageBind |
| CMX | DeCUR's pre-trained weights can directly boost the RGBD segmentation performance of CMX |

**Insights**: The decoupling idea can be generalized to any joint-embedding framework—reserving a portion of dimensions in the shared space for modality-unique information with almost zero increased complexity. Adaptive common/unique ratios (per-sample or per-region) represent a clear direction for improvement.

## Rating

- ⭐⭐⭐⭐ **Novelty**: Splitting Barlow Twins dimensions into common/unique and optimizing them separately is a natural yet clever extension. The idea is clear and elegant.
- ⭐⭐⭐⭐⭐ **Experimental Thoroughness**: Three types of multimodal scenarios + multi-label/multi-architecture/missing modality/ablation/interpretability, offering very comprehensive coverage.
- ⭐⭐⭐⭐ **Writing Quality**: Clearly structured and richly visualized (t-SNE, GradCAM, spectral saliency, deformable points), making it easy to understand.
- ⭐⭐⭐⭐ **Value**: Simple and easy to reproduce, with direct application value for the remote sensing/RGB-D communities; however, verification on mainstream multimodal scenarios like NLP/vision-language is missing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] SOLAR: Self-supervised Joint Learning for Symmetric Multimodal Retrieval](../../ICML2026/multimodal_vlm/solar_self-supervised_joint_learning_for_symmetric_multimodal_retrieval.md)
- [\[ICLR 2026\] SPECS: Decoupling Multimodal Learning via Self-distilled Preference-based Cold Start](../../ICLR2026/multimodal_vlm/specs_decoupling_multimodal_learning_via_self-distilled_preference-based_cold_st.md)
- [\[CVPR 2025\] Self-Supervised Spatial Correspondence Across Modalities](../../CVPR2025/multimodal_vlm/self-supervised_spatial_correspondence_across_modalities.md)
- [\[CVPR 2026\] Finding Distributed Object-Centric Properties in Self-Supervised Transformers](../../CVPR2026/multimodal_vlm/finding_distributed_object-centric_properties_in_self-supervised_transformers.md)
- [\[CVPR 2025\] BadVision: Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models](../../CVPR2025/multimodal_vlm/stealthy_backdoor_attack_in_self-supervised_learning_vision_encoders_for_large_v.md)

</div>

<!-- RELATED:END -->
