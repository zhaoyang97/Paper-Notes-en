---
title: >-
  [Paper Note] Toward Generalizable Whole Brain Representations with High-Resolution Light-Sheet Data
description: >-
  [CVPR 2026][Object Detection][Light-Sheet Fluorescence Microscopy] This paper introduces CANVAS — the first large-scale, subcellular-resolution Light-Sheet Fluorescence Microscopy (LSFM) whole-brain benchmark dataset, encompassing 6 cell markers, approximately 93,000 annotated cells, and a public leaderboard. It reveals critical generalization failures of existing detection models across markers and brain regions, and explores the potential of 3D Masked Autoencoders (MAE) for self-supervised representation learning.
tags:
  - CVPR 2026
  - Object Detection
  - Light-Sheet Fluorescence Microscopy
  - Whole-Brain Imaging
  - Cell Detection Benchmark
  - Self-Supervised Learning
  - Foundation Models
date: 2026-05-08
content_hash: a33cb57310d39a28
---

# Toward Generalizable Whole Brain Representations with High-Resolution Light-Sheet Data

**Conference**: CVPR 2026  
**arXiv**: [2603.29842](https://arxiv.org/abs/2603.29842)  
**Code**: [https://canvas.lightsheetdata.com](https://canvas.lightsheetdata.com)  
**Area**: Object Detection  
**Keywords**: Light-Sheet Fluorescence Microscopy, Whole-Brain Imaging, Cell Detection Benchmark, Self-Supervised Learning, Foundation Models

## TL;DR

This paper introduces CANVAS — the first large-scale, subcellular-resolution Light-Sheet Fluorescence Microscopy (LSFM) whole-brain benchmark dataset, encompassing 6 cell markers, approximately 93,000 annotated cells, and a public leaderboard. It reveals critical generalization failures of existing detection models across markers and brain regions, and explores the potential of 3D Masked Autoencoders (MAE) for self-supervised representation learning.

## Background & Motivation

1. **Background**: Advances in tissue clearing and LSFM have made it possible to acquire complete 3D mouse brain data at subcellular resolution. A single whole-brain dataset can reach approximately 100 GB (compressed), containing 1,600–1,850 z-slices, each approximately 7,000×10,000 pixels.
2. **Limitations of Prior Work**: Despite dramatic improvements in data acquisition capability, scalable processing methods and standardized benchmarks for petabyte-scale LSFM data remain absent. Existing CV models (e.g., U-Net, ResNet, ViT) are primarily designed for CT, fMRI, and X-ray modalities and cannot directly generalize to LSFM data.
3. **Key Challenge**: Different cell-type markers exhibit highly heterogeneous morphological characteristics across brain regions (e.g., astrocytes vs. dopaminergic neurons), making it difficult for a single model to generalize across markers and regions. Additionally, LSFM annotation is extremely costly (167,950 predictions required individual human verification).
4. **Goal**: To provide the first publicly available whole-brain LSFM benchmark dataset, establish cell detection evaluation standards, reveal generalization bottlenecks of existing models, and explore self-supervised methods to address annotation scarcity.
5. **Key Insight**: Construct a comprehensive benchmark covering 6 functionally distinct cell markers (NeuN, cFos, PV, TH, Iba1, GFAP), annotate ROIs from diverse brain regions, and systematically evaluate cross-dataset and cross-region generalization of baseline models.
6. **Core Idea**: Advance the field through a standardized benchmark and rigorous evaluation, rather than proposing a new detection algorithm.

## Method

### Overall Architecture

The CANVAS dataset construction pipeline proceeds as follows: mouse brain tissue → SHIELD preservation → delipidation → SmartBatch+ fluorescence labeling → EasyIndex tissue clearing → SmartSPIM light-sheet microscopy imaging (voxel size 1.8×1.8×4 µm) → destriping + stitching → Zarr format storage → Neuroglancer visualization. The detection benchmark uses ConvMixer as the baseline backbone, with a FindMaxima layer converting probability heatmaps into discrete 3D cell positions.

### Key Designs

1. **Multi-Marker Dataset Design**:

    - Function: Covers 6 distinct cell types to capture morphological heterogeneity.
    - Mechanism: Six markers are selected — NeuN (neuronal nuclear protein, distributed throughout the brain), cFos (neuronal activity marker), PV (parvalbumin interneurons), TH (dopaminergic neurons), Iba1 (microglia), and GFAP (astrocytes). Three training ROIs and three test ROIs are selected per marker, yielding approximately 93,000 annotated cell centroids in total (45,745 training + 47,301 test).
    - Design Motivation: These 6 markers span neurons and immune cells, with morphological variation ranging from spherical nuclei (NeuN) to complex stellate structures (GFAP), systematically challenging model adaptation to diverse cell morphologies.

2. **ConvMixer + FindMaxima Baseline**:

    - Function: Reformulates cell detection in LSFM volumetric data as probability heatmap prediction followed by non-maximum suppression.
    - Mechanism: A ConvMixer backbone outputs a 3D probability heatmap $H$. The FindMaxima layer performs 3D non-maximum suppression: a voxel $H(x,y,z)$ is classified as a cell centroid if it is the local maximum within a $d_{\min}$ neighborhood and exceeds a threshold $\tau$. Separate models are trained per marker; evaluation uses kd-tree-based matching with tolerance thresholds equal to the mean cell radius (NeuN/cFos: 6 pixels; TH/PV/GFAP: 8 pixels; Iba1: 5 pixels).
    - Design Motivation: ConvMixer combines the patch embedding concept of ViT with the simplicity of CNNs, offering high computational efficiency suitable for large-scale data baselines.

3. **3D-MAE Self-Supervised Representation Learning**:

    - Function: Learns transferable volumetric features from large quantities of unlabeled LSFM data.
    - Mechanism: A DINOv2-style ViT is adapted into a 3D MAE with two key modifications: (1) optimized crop/patch sizes tuned to cell morphology (optimal configuration: 16×32×32 crop, 4×8×8 patch); (2) content-aware reconstruction weighting $w_i = \alpha + \gamma \cdot \min(1, \text{Var}(\mathbf{x}_i) / \bar{\sigma}^2)$, where background patches receive weight $\alpha=1$ and cell-containing patches receive weight $\alpha+\gamma=10$. The optimal mask ratio is 0.15, substantially lower than the 0.75 used in natural-image MAE.
    - Design Motivation: LSFM annotation is extremely costly; self-supervised methods can leverage vast amounts of unlabeled data to learn general representations. 3D microscopy images have high semantic density, and aggressive masking disrupts critical spatial relationships.

### Loss & Training

- **Baseline Detection Model**: Binary Focal Loss; models trained separately per marker; convergence within one day on NVIDIA RTX 3090/4090.
- **3D-MAE**: Content-aware MSE reconstruction loss; AdamW optimizer ($\eta=1.5 \times 10^{-4}$); cosine annealing schedule; trained for 700 epochs.
- **All-Marker Model**: Joint training on approximately 60k patches from all 6 markers; reconstruction loss within 15% of single-marker optimum; effective transfer on 10× data.

## Key Experimental Results

### Main Results

| Training Model | cFos F1 | NeuN F1 | TH F1 | PV F1 | GFAP F1 | Iba1 F1 |
|----------------|---------|---------|-------|-------|---------|---------|
| cFos model | **0.78** | 0.02 | 0.04 | 0.28 | 0.00 | 0.03 |
| NeuN model | 0.76 | **0.81** | 0.41 | **0.89** | 0.05 | 0.43 |
| TH model | 0.74 | 0.21 | **0.57** | 0.68 | 0.04 | 0.56 |
| PV model | 0.29 | 0.73 | 0.20 | 0.63 | 0.01 | 0.06 |
| GFAP model | 0.74 | 0.04 | 0.14 | 0.21 | **0.33** | 0.43 |
| Iba1 model | 0.74 | 0.57 | 0.28 | 0.62 | 0.61 | **0.81** |

### Ablation Study (3D-MAE Configuration)

| Configuration | Optimal Mask Ratio | Reconstruction Loss | Notes |
|---------------|--------------------|---------------------|-------|
| 16×32×32 / 4×8×8 | 0.15 | **Best** | Best across 5/6 markers |
| 24×48×48 / 6×12×12 | 0.15–0.35 | Second | Best only for GFAP |
| 32×64×64 / 8×16×16 | 0.35–0.55 | Worst | Oversized receptive field introduces noise |
| All-marker joint | 0.15 | 0.0070 vs 0.0061 | Effective transfer on 10× data |

### Key Findings

- **Severe Generalization Gap**: Most models perform well only on their own marker; cross-dataset F1 drops dramatically (e.g., cFos model achieves only F1=0.02 on NeuN).
- **GFAP Detection is Hardest**: Even the in-domain model achieves only F1=0.33; the Iba1 model (0.61) outperforms the GFAP model on GFAP data.
- **NeuN Model Generalizes Best**: Achieves F1=0.89 on the PV dataset, likely due to morphological similarity of nuclear signals.
- **MAE Features Effectively Improve Detection**: As a post-processing classifier, MAE features yield a mean F1 improvement of 22.9% on GFAP (86.3% on region 3).
- **Optimal Mask Ratio Far Below Natural Images**: 0.15 vs. 0.75, reflecting the high semantic density of 3D biological images.

## Highlights & Insights

- **First Whole-Brain LSFM Benchmark**: CANVAS fills the gap of standardized benchmarks in biological volumetric imaging. Coverage of 6 markers spanning neurons to immune cells enables systematic evaluation of model generalization, and the dataset serves as a valuable resource for 3D foundation model development.
- **Content-Aware MAE Weighting**: A simple yet effective technique — assigning 10× weight to cell-containing patches in sparse biological signals. This "up-weighting informative regions" strategy is broadly applicable to self-supervised methods dealing with sparse data.
- **Cross-Modality Complementary Perspective**: The paper clearly positions LSFM within the fMRI (macroscale) → LSFM (mesoscale) → EM (nanoscale) imaging spectrum, providing a conceptual framework for multimodal brain science research.

## Limitations & Future Work

- Annotation volume remains small (93k annotations vs. approximately 87 million cells in the mouse brain), covering only 3 ROIs per marker.
- No new detection algorithm is proposed; the baseline model (ConvMixer) is relatively simple.
- The 6 markers cover only a fraction of all cell types; future work should expand marker coverage.
- The 3D-MAE is currently used only as a post-processing TP/FP classifier, not integrated end-to-end into detection.
- Future work could explore deep integration of pretrained MAE with detection backbones, rather than feature concatenation followed by classification.

## Related Work & Insights

- **vs. BrainSeg / CellPose, etc.**: Existing methods primarily target specific modalities or cell types; CANVAS provides a systematic platform for cross-marker generalization evaluation.
- **vs. Natural-Image MAE**: The optimal mask ratio for LSFM (0.15) is substantially lower than for natural images (0.75), indicating that pretraining strategies must be adapted to data characteristics.
- **vs. fMRI / EM Datasets**: LSFM achieves a unique balance between whole-organ coverage and subcellular resolution, serving as a critical bridge between macroscale functional imaging and nanoscale structural imaging.

## Rating

- Novelty: ⭐⭐⭐ The core contribution is a benchmark dataset rather than a new method; technical innovation is limited (ConvMixer and MAE are both existing architectures).
- Experimental Thoroughness: ⭐⭐⭐⭐ Cross-evaluation across 6 markers is highly systematic; 84-configuration MAE hyperparameter search is thorough; however, detection baselines are limited.
- Writing Quality: ⭐⭐⭐⭐ Background introduction is detailed and dataset design motivation is clear; the main text is lengthy due to extensive biological background.
- Value: ⭐⭐⭐⭐ As a benchmark dataset, CANVAS provides critical infrastructure for the LSFM community; findings on generalization gaps offer actionable guidance for designing improved whole-brain analysis models.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] MRD: Multi-resolution Retrieval-Detection Fusion for High-Resolution Image Understanding](mrd_multi-resolution_retrieval-detection_fusion_for_high-resolution_image_unders.md)
- [\[CVPR 2026\] HeROD: Heuristic-inspired Reasoning Priors Facilitate Data-Efficient Referring Object Detection](herod_heuristic_inspired_reasoning_data_efficient_rod.md)
- [\[ICLR 2026\] CORDS: Continuous Representations of Discrete Structures](../../ICLR2026/object_detection/cords_continuous_representations_of_discrete_structures.md)
- [\[NeurIPS 2025\] Generalizable Insights for Graph Transformers in Theory and Practice](../../NeurIPS2025/object_detection/generalizable_insights_for_graph_transformers_in_theory_and_practice.md)
- [\[CVPR 2026\] CD-Buffer: Complementary Dual-Buffer Framework for Test-Time Adaptation in Adverse Weather Object Detection](cd-buffer_complementary_dual-buffer_framework_for_test-time_adaptation_in_advers.md)

<!-- RELATED:END -->
