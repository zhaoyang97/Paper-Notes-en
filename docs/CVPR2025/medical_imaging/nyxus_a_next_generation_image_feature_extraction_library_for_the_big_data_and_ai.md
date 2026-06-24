---
title: >-
  [Paper Note] Nyxus: A Next Generation Image Feature Extraction Library for the Big Data and AI Era
description: >-
  [CVPR2025][Medical Imaging][feature extraction] Nyxus is a next-generation image feature extraction library designed for the big data and AI era. It supports out-of-core scalable extraction of 2D/3D data, covering 261+ features across both radiomics and cell profiling domains, enabling speedups of 3–131× over CellProfiler and several to hundreds of times over PyRadiomics/MITK.
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "feature extraction"
  - "image analysis"
  - "scalability"
  - "radiomics"
  - "cell profiling"
  - "GPU acceleration"
date: 2026-05-08
content_hash: 1c0c87f3ef0eeb1e
---

# Nyxus: A Next Generation Image Feature Extraction Library for the Big Data and AI Era

**Conference**: CVPR2025  
**arXiv**: [2603.12016](https://arxiv.org/abs/2603.12016)  
**Code**: [GitHub (MIT)](https://github.com/PolusAI/nyxus)  
**Area**: Medical Image  
**Keywords**: feature extraction, image analysis, scalability, radiomics, cell profiling, GPU acceleration

## TL;DR

Nyxus is a next-generation image feature extraction library designed for the big data and AI era. It supports out-of-core scalable extraction of 2D/3D data, covering 261+ features across both radiomics and cell profiling domains, enabling speedups of 3–131× over CellProfiler and several to hundreds of times over PyRadiomics/MITK.

## Background & Motivation

- Modern imaging devices generate terabyte to petabyte-scale experimental data, which conventional feature extraction methods struggle to handle due to limitations in efficiency, scalability, and memory management.
- Advances in high-resolution, high-dimensional (temporal and volumetric), and multiplexing/hyperplexing imaging technologies have led to an exponential growth in data volume.
- The two major communities in image analysis—radiomics and cell profiling—have historically developed independently, leading to inconsistent feature implementation standards. Identical features (e.g., perimeter) yield disparate results across different libraries, which hinders reproducibility.
- Existing tools generally suffer from the following drawbacks: (1) inability to handle out-of-core large images; (2) non-configurable feature hyperparameters; (3) lack of GPU acceleration; (4) incomplete feature coverage or bias towards a single domain; (5) limited user interfaces.
- Supports two primary use cases: image features (whole image) and regional features (ROI features).
- Although deep learning has significantly improved segmentation accuracy, subsequent feature extraction still lacks unified, efficient, and cross-domain tools.

## Method

### Design Philosophy
Nyxus is redesigned from the ground up with the core objectives of: out-of-core processing (handling images exceeding physical memory), harmonization of cross-domain features, configurable hyperparameters, and multi-platform, multi-interface delivery.

### Feature System
- Covers 5 major feature categories: morphological, intensity, texture, volumetric, and miscellaneous.
- Total of 261+ features, exceeding all compared libraries (CellProfiler 122, PyRadiomics, MITK, etc.), with the largest number of features in all categories except miscellaneous.
- Supports IBSI standard configuration profiles to ensure radiomics features comply with international standards.
- Programmable optimization of feature hyperparameters: provides two modes, untargeted (default parameters) and targeted (speed-optimized parameters).

### Scalable Architecture
- **Out-of-core processing**: Supports tiling-based loading and computation of ultra-large images, free from memory capacity constraints.
- **Multi-threaded CPU parallelism**: The optimal thread count is approximately 6–10 threads, with diminishing returns beyond this range.
- **GPU (CUDA) acceleration**: Provides over 3× speedup for large ROIs (>5000 pixels); GPU acceleration is not recommended for small ROIs due to data transfer bottlenecks.
- **Embarrassingly parallel**: Well-suited for distributed architectures; a parallel strategy that allocates a small number of threads per image is optimal.

### Delivery Formats
- Python packages (PyPI/Conda), Command Line Interface (CLI), Napari plugin (low-code GUI), OCI containers (cloud/HPC deployment), and CWL workflow tools.
- Licensed under the MIT open-source license, suitable for both public and commercial applications.

### Accuracy Verification
- Feature accuracy validated using standard IBSI datasets.
- Numerical accuracy validated against MATLAB's image feature extraction libraries.
- Supports configurable feature hyperparameter profiles to adapt to future standards.
- Evaluated against 9 libraries: Nyxus, CellProfiler, PyRadiomics, MITK, RadiomicsJ, WND-CHARM, NIST WIPP, Imea, and MATLAB.
- Unlike radiomics, the cell profiling community currently lacks an IBSI-like standard, but the architecture of Nyxus is designed to easily integrate future standards.

## Key Experimental Results

### Speed Comparison (TissueNet, 2498 Microscopic Images)

| Compared Library | Untargeted Speedup | Targeted Speedup |
|--------|-------------------|-----------------|
| CellProfiler | 3–35× | 58–131× |
| MITK | 100–1000× | Faster |
| RadiomicsJ | 100–1000× | Faster |

### Speed Comparison (Medical Decathlon, Clinical Data)

| Compared Library | Untargeted | Targeted |
|--------|-----------|---------|
| RadiomicsJ | >5× | Faster |
| MITK | ~2× (Intensity) | 1.46–357× |
| CellProfiler | 20–198× | Faster |

### Hardware Scalability
- Multi-threading gains saturate after 6–10 threads, indicating that high-performance computing clusters are not strictly necessary for feature extraction.
- Executes efficiently on Apple Silicon (M1 MacBook Pro), showing excellent performance on ARM architectures.
- GPU acceleration: achieves >3× speedup with 100 ROIs of 100K pixels each.
- Computation time scales linearly for ROIs >500 pixels, demonstrating high predictability.
- Out-of-core processing maintains the aforementioned performance trends even on memory-constrained nodes.

### Feature Coverage
- Nyxus (261 features) vs. CellProfiler (122) vs. WND-CHARM (122) vs. PyRadiomics/MITK/RadiomicsJ (fewer).
- Possesses the highest number of features across all four categories: intensity, shape, texture, and volumetric.

## Highlights & Insights

1. **Unified Feature Library**: Integrates features from both the radiomics and cell profiling communities into a single framework for the first time, harmonizing standards.
2. **Extreme Scalability**: Co-design of out-of-core processing, GPU acceleration, and multi-threaded CPU execution allows handling of petabyte-scale data.
3. **Overwhelming Speed Advantage**: Up to 131× faster than CellProfiler in targeted mode, which is a conservative estimate given that Nyxus computes a larger number of features.
4. **Multi-Interface Delivery**: From Python APIs to no-code GUIs and cloud-native containers, catering to users of all skill levels.
5. **Rigorous Accuracy Validation**: Numerical accuracy is verified against the IBSI standard and MATLAB, ensuring scientific reliability.
6. **Configurable Hyperparameter Profiles**: Supports on-demand tuning of feature calculation parameters, leaving room for downstream ML/DL application optimization.
7. **Robust Cross-Hardware Performance**: Runs efficiently on environments ranging from Linux EC2 to Apple Silicon M1, demonstrating high adaptability.

## Limitations & Future Work

1. **Leans Towards System Engineering Rather Than Methodological Innovation**: From a computer vision perspective, this is more of a system/tool paper than a methodology paper, introducing no new algorithms.
2. **Lack of Standard in Cell Profiling**: IBSI only covers radiomics, and the cell profiling domain lacks a similar standardized configuration, limiting cross-domain validation.
3. **Conditional GPU Acceleration**: Small ROIs (<5000 pixels) are slower on GPUs than on CPUs due to data transfer overheads, which limits comprehensive acceleration in scenarios such as whole slide imaging (WSI).
4. **Partially Unfair Comparison**: The compared libraries extract different numbers of features, and the execution times were not normalized by feature count. Additionally, some libraries run in worst-case scenarios when handling non-native data formats.
5. **Limited Deep Learning Integration**: Does not study deep integration with end-to-end deep learning pipelines, such as feature selection or differentiable feature extraction.
6. **Insufficient Evaluation of 3D Features**: The evaluation focuses primarily on 2D scenarios, with limited large-scale performance benchmarking on 3D volumetric features.

## Rating
- Novelty: ⭐⭐ (Outstanding engineering contribution but no new algorithm)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Highly comprehensive comparison across datasets, hardware, and libraries)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, rich figures/tables)
- Value: ⭐⭐⭐⭐ (High practical value for the biomedical image analysis community)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Boundaries of Fair AI in Medical Image Prognosis: A Causal Perspective](../../NeurIPS2025/medical_imaging/the_boundaries_of_fair_ai_in_medical_image_prognosis_a_causal_perspective.md)
- [\[CVPR 2026\] Dual-Level Hypergraph Generation for Addressing Feature Scarcity in Whole-Slide Image Classification](../../CVPR2026/medical_imaging/dual-level_hypergraph_generation_for_addressing_feature_scarcity_in_whole-slide_.md)
- [\[CVPR 2025\] Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation](enhanced_contrastive_learning_with_multi-view_longitudinal_data_for_chest_x-ray_.md)
- [\[NeurIPS 2025\] Interpretable Next-token Prediction via the Generalized Induction Head](../../NeurIPS2025/medical_imaging/interpretable_next-token_prediction_via_the_generalized_induction_head.md)
- [\[AAAI 2026\] Rethinking Bias in Generative Data Augmentation for Medical AI: a Frequency Recalibration Approach](../../AAAI2026/medical_imaging/rethinking_bias_in_generative_data_augmentation_for_medical_ai_a_frequency_recal.md)

</div>

<!-- RELATED:END -->
