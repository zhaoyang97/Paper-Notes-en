---
title: >-
  [Paper Note] STARC-9: A Large-scale Dataset for Multi-Class Tissue Classification for CRC Histopathology
description: >-
  [NeurIPS 2025][Medical Imaging][Colorectal Cancer] This paper introduces STARC-9, a large-scale colorectal cancer (CRC) tissue classification dataset comprising 630K patches across 9 tissue classes…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Colorectal Cancer"
  - "Tissue Classification"
  - "Dataset"
  - "Histopathology"
  - "Deep Clustering"
date: 2026-05-08
content_hash: 85c5f4c9b9883a52
---

# STARC-9: A Large-scale Dataset for Multi-Class Tissue Classification for CRC Histopathology

**Conference**: NeurIPS 2025
**arXiv**: [2511.00383](https://arxiv.org/abs/2511.00383)  
**Code**: [GitHub](https://github.com/Path2AI/STARC-9)  
**Area**: Medical Imaging
**Keywords**: Colorectal Cancer, Tissue Classification, Dataset, Histopathology, Deep Clustering

## TL;DR

This paper introduces STARC-9, a large-scale colorectal cancer (CRC) tissue classification dataset comprising 630K patches across 9 tissue classes, along with its construction framework DeepCluster++. The framework combines domain-specific autoencoder feature extraction, K-means clustering, and equal-frequency binning sampling to ensure morphological diversity. Models trained on STARC-9 significantly outperform those trained on NCT and HMU.

## Background & Motivation

Multi-class tissue classification is a foundational task in computational pathology, supporting downstream applications such as tissue segmentation, biomarker prediction, and survival analysis. However, existing public CRC datasets suffer from three core issues:

**Insufficient morphological diversity**: Samples fail to adequately represent the broad appearance variation within each tissue class.

**Class imbalance**: Dominant classes such as tumor epithelium vastly outnumber clinically important but rare classes such as mucus or necrosis.

**Low-quality patches**: Datasets contain misclassified patches and artifact-contaminated patches that impair model learning.

For instance, the widely used NCT-CRC-HE-100K contains only 100K images and exhibits JPEG compression artifacts. HMU-GC-HE-30K includes non-representative (misclassified) patches. TCGA datasets provide only whole-slide images (WSIs) without patch-level annotations.

**Deeper issues**: Existing dataset construction methods are each limited — manual annotation relies on subjective pathologist judgment and is time-consuming; random sampling tends to miss rare morphologies; conventional deep clustering oversamples near cluster centroids, reducing diversity. No unified framework currently exists for large-scale, balanced, and diverse dataset construction.

## Method

### Overall Architecture

DeepCluster++ is a semi-automated dataset construction framework operating in three stages: (1) training a domain-specific autoencoder on histopathology images; (2) extracting features with the frozen encoder → K-means clustering → equal-frequency binning sampling to ensure diversity; and (3) pathologist verification.

### Key Designs

1. **Domain-specific autoencoder (AE_CRC)**: Trained on 100K CRC patches, the encoder consists of 6 convolutional layers with BatchNorm and LeakyReLU, producing a 32,768-dimensional latent vector, trained with SSIM loss.

   The rationale for choosing an autoencoder over pretrained foundation models (e.g., UNI, CTransPath): (a) reconstruction-loss-driven features preserve structural-level morphological similarity, yielding more coherent clusters; (b) models trained with classification or contrastive objectives tend to over-separate biologically related tissues, reducing intra-class consistency; (c) the autoencoder is lightweight and efficient for processing 630K patches.

2. **Clustering and diversity sampling**: For each WSI, frozen encoder features are extracted → global average pooling reduces dimensionality to 512 → PCA reduces to 256 → K-means clustering (with $m=400$ samples per cluster). The key innovation is **Equal-Frequency Binning**:

    - Compute the Euclidean distance from each patch to its cluster centroid: $d_i = \|v_i - c\|$
    - Sort patches by distance and divide into $g=5$ equal-frequency bins
    - Sample 20% of patches from each bin

   This approach ensures uniform representation from near-centroid (homogeneous) to cluster-boundary (diverse) patches, avoiding oversampling in dense regions. Compared to equal-width binning, equal-frequency binning maintains consistent patch counts across bins.

3. **Efficient cluster label propagation**: After annotating a seed cluster, the local continuity of the embedding space is leveraged to propagate labels to neighboring clusters. For example, labeling cluster_48 as "TUM" reveals that adjacent clusters 2, 97, 53, and 112 contain the same tissue morphology, enabling batch annotation and substantially reducing manual effort.

### Loss & Training

The autoencoder is trained with SSIM loss, which outperforms MSE loss on the validation set (SSIM: 0.9262 vs. 0.8863, PSNR: 32.48 dB vs. 28.53 dB). Downstream models are uniformly trained with batch size 32, lr=0.0001, weight decay=1e-5, Adam optimizer, 10 epochs, with data augmentation including flipping, rotation, and color jitter.

## Key Experimental Results

### Main Results (Multi-class Tissue Classification, 7 Common Tissue Classes)

Evaluated on STANFORD-CRC-HE-VAL-LARGE (54,000 patches):

| Model | NCT-trained Acc | HMU-trained Acc | STARC-9-trained Acc | Gain (vs. NCT) |
|-------|----------------|----------------|--------------------:|---------------|
| ResNet50 | 62.59% | 85.71% | **98.64%** | +36.05% |
| EfficientNet-B7 | 82.47% | 84.45% | **98.80%** | +16.33% |
| ViT-base | 84.25% | 90.29% | **98.09%** | +13.84% |
| DeiT-B | 81.63% | 90.05% | **98.65%** | +17.02% |
| Swin Trans-base | 79.05% | 91.88% | **98.79%** | +19.74% |
| CTransPath | 79.05% | 91.88% | **99.00%** | +19.95% |
| UNI | 80.43% | 91.80% | **98.26%** | +17.83% |

All models trained on STARC-9 exceed 98% accuracy, with CTransPath reaching 99%.

### Tumor Segmentation Experiments

| Validation Set | NCT IoU | HMU IoU | STARC-9 IoU | NCT Dice | HMU Dice | STARC-9 Dice |
|----------------|---------|---------|-------------|----------|----------|-------------|
| Stanford | 67.19±21.53 | 64.68±24.21 | **89.33±8.76** | 78.20±17.01 | 75.49±21.01 | **90.47±8.14** |
| TCGA-CRC | 51.94±37.94 | 58.89±29.42 | **88.81±10.90** | 58.90±31.38 | 68.85±22.10 | **89.38±9.14** |

Models trained on STARC-9 achieve approximately 14% higher Dice on Stanford and approximately 30% higher on TCGA-CRC, with substantially smaller standard deviations.

### Ablation Study

| Ablation | Evaluation Method | Result |
|----------|------------------|--------|
| SSIM vs. MSE autoencoder | Reconstruction quality | SSIM superior (0.9262 vs. 0.8863) |
| Encoder comparison in latent space | Feature visualization | AE_CRC yields most coherent clusters; UNI over-separates tissues |
| Equal-frequency vs. equal-width binning | Diversity | Equal-frequency binning better preserves intra-class heterogeneity |
| Samples per cluster | Cluster purity | $m=400$ optimal ($m=800$ mixes tissues; $m=100$ insufficient variation) |

### Key Findings

- **Data quality is decisive**: Replacing only the training data (STARC-9 vs. NCT/HMU) under identical model architectures and training configurations yields 14–36% accuracy improvements.
- **High accuracy without pretraining**: CNNs trained from scratch on STARC-9 achieve 97.81% accuracy, demonstrating that data quality can reduce reliance on pretrained models.
- **Mixed-tissue patches are a critical challenge**: Models trained on STARC-9 achieve 85% accuracy on mixed-tissue patches, far surpassing HMU (55%) and NCT (42%).
- **Cross-dataset generalization**: High performance is maintained on external TCGA-CRC and IMP-CRS10K datasets.
- Classification accuracy for necrosis (NCS) improves most markedly (>90% gain vs. NCT; >45% gain vs. HMU).

## Highlights & Insights

- **Generalizability of DeepCluster++**: While demonstrated on CRC, the framework is directly applicable to WSI dataset construction for other cancer types.
- **Equal-frequency binning ensures diversity**: Unlike the common practice of sampling near cluster centroids, equal-frequency binning uniformly covers from center to boundary, providing more comprehensive morphological representation.
- **Strong evidence that data > model**: Consistent superiority of STARC-9-trained models across all architectures underscores the decisive impact of data quality and diversity on model performance.
- **630K patches, ~70K per class across 9 categories**: Near-perfect class balance eliminates class imbalance as a confounding factor during training.

## Limitations & Future Work

- Data originates from a single institution (Stanford), limiting demographic diversity (underrepresentation of Black and Indigenous populations).
- The 9 tissue classes may not fully cover all tissue types present in CRC resection specimens.
- The framework is currently limited to CRC; applicability of DeepCluster++ to other cancer types (e.g., CNS tumors) requires validation.
- AE_CRC may require retraining for new datasets.
- The absence of image–text pairs precludes support for multimodal applications.

## Related Work & Insights

Compared to NCT-CRC-HE-100K (100K patches, JPEG artifacts) and HMU-GC-HE-30K (30K patches, erroneous labels), STARC-9 is comprehensively superior in scale, quality, and diversity. Compared to QuPath-based manual annotation, DeepCluster++ substantially reduces human labor. Key insight: when constructing large-scale datasets, the sampling strategy (equal-frequency binning) and the choice of feature extractor (reconstruction loss vs. discriminative loss) are critical determinants of final data quality.

## Rating

- Novelty: ⭐⭐⭐⭐ DeepCluster++ features original design; the equal-frequency binning sampling strategy is practically valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 21 model benchmarks, multiple validation sets, both classification and segmentation tasks — exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Framework is clearly presented and experiments are thorough, though the paper is somewhat lengthy.
- Value: ⭐⭐⭐⭐⭐ Dataset, framework, and benchmark models are fully open-sourced, representing a major contribution to the computational pathology community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Semantic and Visual Crop-Guided Diffusion Models for Heterogeneous Tissue Synthesis in Histopathology](semantic_and_visual_crop-guided_diffusion_models_for_heterogeneous_tissue_synthe.md)
- [\[NeurIPS 2025\] MATCH: Multi-faceted Adaptive Topo-Consistency for Semi-Supervised Histopathology Segmentation](match_multi-faceted_adaptive_topo-consistency_for_semi-supervised_histopathology.md)
- [\[NeurIPS 2025\] RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis](ram-w600_a_multi-task_wrist_dataset_and_benchmark_for_rheumatoid_arthritis.md)
- [\[ICLR 2026\] Omni-iEEG: A Large-Scale, Comprehensive iEEG Dataset and Benchmark for Epilepsy Research](../../ICLR2026/medical_imaging/omni-ieeg_a_large-scale_comprehensive_ieeg_dataset_and_benchmark_for_epilepsy_re.md)
- [\[NeurIPS 2025\] PhysioWave: A Multi-Scale Wavelet-Transformer for Physiological Signal Representation](physiowave_a_multi-scale_wavelet-transformer_for_physiological_signal_representa.md)

</div>

<!-- RELATED:END -->
