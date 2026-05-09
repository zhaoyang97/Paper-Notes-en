---
title: >-
  [Paper Note] MozzaVID: Mozzarella Volumetric Image Dataset
description: >-
  [CVPR 2026][Medical Imaging][volumetric image dataset] This paper introduces MozzaVID — a mozzarella cheese microstructure volumetric image classification dataset based on synchrotron X-ray CT — comprising 591–37,824 samples of size $192^3$, with classification targets spanning 25 cheese types and 149 individual cheese specimens. The dataset bridges the large gap in scale and task design between 3D volumetric and 2D datasets, and experiments demonstrate that 3D models significantly outperform their 2D counterparts.
tags:
  - CVPR 2026
  - Medical Imaging
  - volumetric image dataset
  - 3D classification
  - X-ray CT
  - food microstructure
  - deep learning benchmark
date: 2026-05-08
content_hash: 44f4f8cc50b36797
---

# MozzaVID: Mozzarella Volumetric Image Dataset

**Conference**: CVPR 2026
**arXiv**: [2412.04880](https://arxiv.org/abs/2412.04880)
**Code**: [https://papieta.github.io/MozzaVID/](https://papieta.github.io/MozzaVID/) (available; dataset publicly released)
**Area**: Medical Imaging / Dataset
**Keywords**: volumetric image dataset, 3D classification, X-ray CT, food microstructure, deep learning benchmark

## TL;DR

This paper introduces MozzaVID — a mozzarella cheese microstructure volumetric image classification dataset based on synchrotron X-ray CT — comprising 591–37,824 samples of size $192^3$, with classification targets spanning 25 cheese types and 149 individual cheese specimens. The dataset bridges the large gap in scale and task design between 3D volumetric and 2D datasets, and experiments demonstrate that 3D models significantly outperform their 2D counterparts.

## Background & Motivation

1. **Background**: Volumetric images (3D CT, MRI, etc.) are widely used in medicine, materials science, and food science, with growing interest in deep learning for these domains. In the 2D domain, standard benchmarks such as MNIST (60K images) and ImageNet (14M images) have driven substantial architectural innovation.

2. **Limitations of Prior Work**: Volumetric datasets suffer from critical shortcomings — (a) **insufficient scale**: the largest volumetric datasets (e.g., BugNIST with 9,154 samples, PN9 with 8,798) are far smaller than 2D datasets; (b) **poor accessibility**: many medical datasets require registration, data-use agreements, or direct contact with curators; (c) **overly specialized tasks**: most datasets target specific diagnostic problems and are unsuitable as general-purpose benchmarks; (d) **lack of classification-oriented benchmarks**: the majority of volumetric datasets focus on segmentation or detection with few classification targets.

3. **Key Challenge**: The absence of a large-scale, general-purpose volumetric benchmark prevents 3D deep learning researchers from comparing architectures under a unified standard, as is routinely done in 2D. Consequently, new models are often evaluated on a single specialized dataset, limiting generalizability and cross-study comparability. Many 3D methods simply adapt 2D architectures to 3D, potentially missing opportunities for volumetric-specific optimization.

4. **Goal**: To create a large-scale, clean, multipurpose, and publicly available volumetric image classification benchmark that closes the scale gap between 2D and 3D datasets.

5. **Key Insight**: The microstructure of mozzarella cheese is anisotropic and highly disordered — properties that allow arbitrary sub-volume extraction without introducing bias. This enables derivation of up to 37,824 samples from 591 original scans, a unique advantage offered by food microstructure.

6. **Core Idea**: By exploiting the disorder and divisibility of mozzarella microstructure, the authors construct an unprecedented-scale 3D classification benchmark (37K+ volumes) and simultaneously validate the indispensability of 3D representations for volumetric tasks.

## Method

### Overall Architecture

The construction pipeline of MozzaVID proceeds as follows: (1) six specimens are cut from each of 25 mozzarella cheese formulations (150 specimens total), and each specimen undergoes four local tomographic scans (600 scans total; 9 are discarded, yielding 591); (2) high-resolution CT scans are acquired at the DanMAX beamline of the MAX IV synchrotron at 20 keV with a pixel size of 0.55 μm; (3) after preprocessing, the raw volumes are cropped and downsampled into three configurations, each producing $192^3$ sub-volumes.

### Key Designs

1. **Data Acquisition and Preprocessing**:

    - Function: Obtain high-quality, low-noise 3D microstructural images.
    - Mechanism: A synchrotron X-ray source is used (offering higher flux and coherence than laboratory micro-CT). Each scan acquires 2,601 projections at 1.5 ms exposure with a $2356 \times 2688$ detector. Preprocessing includes discarding 9 scans with artifacts, cropping to $1601 \times 1601 \times 2156$ voxels, and histogram alignment (intensity normalization after segmenting fat and protein phases).
    - Design Motivation: The X-ray attenuation coefficients of protein and fat in mozzarella are similar, making laboratory CT imaging noisy with poor contrast. Synchrotron radiation resolves this limitation, and rapid scanning also mitigates thermal instability of the sample.

2. **Three Dataset Configurations (Small / Base / Large)**:

    - Function: Support evaluation at different scales, ranging from "typical volumetric dataset size" to "approaching 2D dataset scale."
    - Mechanism: Starting from a centered $1536^3$ voxel cube — the 8X-1X (Small) configuration downsamples by $8\times$ to retain 591 volumes; 4X-2X (Base) downsamples by $4\times$ and subdivides each into 8 sub-volumes, yielding 4,728; 2X-4X (Large) downsamples by $2\times$ and subdivides each into 64 sub-volumes, yielding 37,824. All configurations produce $192^3$ output volumes.
    - Design Motivation: Since mozzarella microstructure has no specific macroscopic shape or boundary, subdivision introduces no bias. The three configurations allow researchers to establish baseline performance on Large while exploring data-limited scenarios with Small/Base.

3. **Dual-Granularity Classification Targets**:

    - Function: Provide two classification difficulty levels — coarse-grained (25 cheese types) and fine-grained (149 cheese specimens).
    - Mechanism: The 25 cheese types are produced with different formulations (cooking temperature, screw speed, additives), representing distinct structural variants. Each type has 6 specimens, with subtle intra-type variation arising from spatial position differences. The hierarchy is: inter-type similarity (shallow) < inter-specimen similarity within the same type (moderate) < inter-scan similarity within the same specimen (strong).
    - Design Motivation: Coarse-grained classification achieves up to 97.3% accuracy on Large (feasible but non-trivial), while fine-grained classification reaches only 73.3% on Base (challenging). Classification on smaller configurations is more demanding, reflecting realistic volumetric data scenarios.

### Loss & Training

Cross-entropy loss is used, and test-set accuracy is reported as the evaluation metric. The AdamW optimizer is employed with an effective batch size of 32 and a learning rate of $10^{-4}$ for large models. Data augmentation consists solely of random flips along the XY axes. CNN models use early stopping at 30 epochs; Transformer models at 50 epochs. If convergence is not reached within 5 days, the best checkpoint is retained.

## Key Experimental Results

### Main Results

| Model | Coarse-3D-Large | Coarse-2D-Large | Fine-3D-Large | Fine-2D-Large |
|-------|----------------|-----------------|---------------|---------------|
| ResNet50 | **0.973** | 0.777 | **0.935** | 0.770 |
| MobileNetV2 | 0.909 | 0.775 | 0.895 | 0.857 |
| ConvNeXt-S | 0.806 | 0.621 | 0.877* | 0.652 |
| ViT-B/16 | 0.731 | 0.474 | 0.855 | 0.442 |
| Swin-S | 0.896* | 0.620 | 0.922* | 0.686 |
| **Average** | **0.863** | **0.653** | **0.905** | **0.681** |

On average, 3D models outperform 2D models by 21–22 percentage points, a highly significant margin.

### Effect of Dataset Scale

| Configuration | Coarse-3D | Coarse-2D | Notes |
|---------------|-----------|-----------|-------|
| Small (591) | 0.614 (avg) | 0.367 (avg) | Very limited data; severe overfitting |
| Base (4,728) | 0.799 (avg) | 0.625 (avg) | Moderate scale; gap narrows |
| Large (37,824) | 0.863 (avg) | 0.653 (avg) | Largest scale; 3D advantage most pronounced |

### Key Findings

- **3D representations are indispensable**: 3D models on Coarse-Base (4,728 samples) generally outperform 2D models on Large (37,824 samples), indicating that 3D representation is far more important than data volume for volumetric tasks. However, this trend is less clear for fine-grained classification, suggesting that textural details can be partially captured through 2D slices.
- **ResNet50 is the most robust across all 3D configurations**, outperforming more modern architectures such as ConvNeXt and ViT. This suggests that state-of-the-art architectures may be over-optimized for 2D data, and that 3D-specific architecture design remains an open opportunity.
- **Swin Transformer performs competitively**, achieving reasonable results even on Small/Base despite Transformers' known data hunger, making it a promising starting point for 3D architecture development.
- **UMAP embeddings reveal meaningful structural representations** — cheeses of similar formulations cluster more closely in the embedding space, and the cluster distribution is highly consistent with the PCA space of chemical parameters.

## Highlights & Insights

- **Exploiting the "disorder" of food microstructure as an asset** — precisely because mozzarella lacks regular repeating patterns, sub-volumes can be freely extracted without information loss. This insight elegantly converts a material property into an inherent advantage for dataset design.
- **The three-configuration design spans the full spectrum from "typical volumetric scenarios" to "near-2D benchmark scale"**, enabling a single dataset to serve diverse research needs simultaneously.
- **Classification as a tool for structural analysis** — the embedding space of a classifier can quantify and visualize microstructural variability, offering direct utility for food science research beyond serving as a methodological benchmark.
- **First 3D deep learning dataset for food microstructure** — filling a gap at the intersection of food science and computer vision.

## Limitations & Future Work

- **Classification only** — no segmentation or detection ground truth is provided. Although this is a prerequisite of the subdivision strategy (classification does not depend on macroscopic structural integrity), it limits the diversity of use cases.
- **Accuracy on Large is already very high (97.3%)**, leaving limited room for future improvement; Small/Base configurations are more challenging targets.
- **Single data source** — limited to mozzarella. Although the authors argue for visual similarity with other organic/medical volumetric data, the benchmark value for other materials requires independent validation.
- **Scan orientation may introduce bias** — fiber orientation differences between cheese types could be exploited by models as a shortcut for classification. Ablation results suggest limited impact but do not fully rule this out.
- **Absence of pretraining** — all models are trained from scratch, which may disadvantage data-hungry architectures such as ViT. Self-supervised pretraining on volumetric data is a promising future direction.

## Related Work & Insights

- **vs. BugNIST**: Both are non-medical 3D classification datasets. BugNIST contains 9,544 samples but presents an overly simple classification task. MozzaVID offers $4\times$ more samples in the Large configuration with a more challenging classification objective.
- **vs. MedMNIST 3D**: MedMNIST 3D contains 1,633–1,908 samples at a low resolution of $64^3$ with only 2–11 classes. MozzaVID is better suited as a benchmark in terms of both sample size and number of categories.
- **vs. 2D food datasets (FoodSeg103 / Recipe1M+)**: These are 2D datasets of food photographs targeting a fundamentally different problem (finished product recognition vs. microstructure analysis).

## Rating

- Novelty: ⭐⭐⭐⭐ — First large-scale 3D food microstructure dataset with creative material selection; however, the primary contribution is the dataset rather than a methodological advance.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage across 5 architectures × 3 configurations × 2 granularities × 2D/3D, with in-depth UMAP analysis.
- Writing Quality: ⭐⭐⭐⭐ — An exemplary dataset paper with an exceptionally thorough survey of related work.
- Value: ⭐⭐⭐⭐ — Fills an important gap in 3D benchmarking and has the potential to become a standard test bed for volumetric deep learning.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] GaussianPile: A Unified Sparse Gaussian Splatting Framework for Slice-based Volumetric Reconstruction](gaussianpile_a_unified_sparse_gaussian_splatting_framework_for_slice-based_volum.md)
- [\[CVPR 2026\] Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study](are_general-purpose_vision_models_all_we_need_for_2d_medical_image_segmentation_.md)
- [\[CVPR 2026\] RDFace: A Benchmark Dataset for Rare Disease Facial Image Analysis under Extreme Data Scarcity and Phenotype-Aware Synthetic Generation](rdface_a_benchmark_dataset_for_rare_disease_facial_image_analysis_under_extreme_.md)
- [\[CVPR 2026\] GLEAM: A Multimodal Imaging Dataset and HAMM for Glaucoma Classification](gleam_a_multimodal_imaging_dataset_and_hamm_for_gl.md)
- [\[CVPR 2026\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)

<!-- RELATED:END -->
