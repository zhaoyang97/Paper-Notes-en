---
title: >-
  [Paper Note] TopoCellGen: Generating Histopathology Cell Topology with a Diffusion Model
description: >-
  [CVPR 2025][Medical Imaging][Digital Pathology] This paper proposes TopoCellGen, the first diffusion model for generating multi-class cell topological layouts in digital pathology. It introduces intra-class spatial consistency and inter-class structural regularization constraints via persistent homology, and proposes the Topological Fréchet Distance (TopoFD) evaluation metric.
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Digital Pathology"
  - "Topological Constraints"
  - "Diffusion Models"
  - "Cell Layout Generation"
  - "Persistent Homology"
date: 2026-05-08
content_hash: 083e39b52e175acf
---

# TopoCellGen: Generating Histopathology Cell Topology with a Diffusion Model

**Conference**: CVPR 2025  
**arXiv**: [2412.06011](https://arxiv.org/abs/2412.06011)  
**Code**: [GitHub](https://github.com/Melon-Xu/TopoCellGen)  
**Area**: Medical Imaging  
**Keywords**: Digital Pathology, Topological Constraints, Diffusion Models, Cell Layout Generation, Persistent Homology

## TL;DR

This paper proposes TopoCellGen, the first diffusion model for generating multi-class cell topological layouts in digital pathology. It introduces intra-class spatial consistency and inter-class structural regularization constraints via persistent homology, and proposes the Topological Fréchet Distance (TopoFD) evaluation metric.

## Background & Motivation

- **Importance of Multi-class Cell Topology**: The spatial organization of different cell types (lymphocytes, epithelial cells, stromal cells, etc.) in tissues is crucial for understanding the tumor microenvironment, disease progression, and diagnosis. For instance, the density of tumor-infiltrating lymphocytes (TILs) is closely associated with clinical prognosis.
- **Limitations of Prior Work**: Existing pathology image diffusion models directly generate images, failing to explicitly model spatial layouts of cells. This makes them difficult to align with pathologists' domain knowledge, as well as hard to control and validate.
- **Value of Cell Layout Generation**: (1) Aligns directly with pathologists' domain knowledge, (2) controllable generation allows generalization to unseen scenarios, and (3) generated layouts can conditionally produce H&E images for data augmentation.
- **Topological Relations are Key**: Topological patterns like clustering, mixing, and connectivity among cells provide deep insight into cell communication, structural changes, and morphological abnormalities. For example, epithelial cells arrange in ring/tubular structures in healthy tissues, whereas immune cells cluster around tumors.
- **Lack of Evaluation Metrics**: Traditional FID focuses solely on visual similarity, failing to evaluate topological structural fidelity.

## Method

### Overall Architecture

TopoCellGen adds three topology-aware constraints on top of DDPM:
1. **Cell Counting Loss**: Precisely controls the quantity of each cell class.
2. **Intra-class Spatial Consistency**: Maintains intra-class spatial distribution patterns.
3. **Inter-class Structural Regularization**: Preserves cross-class topological relationships.
During inference, cell layouts are generated and then converted into H&E stained images via a conditional generative model.

### Key Designs

**1. Differentiable Cell Counting Loss**
- **Function**: Precisely control the number of cells in each class in the generated layout, addressing the cell count bias of prior models.
- **Mechanism**: Uses conditional vector $c = [c_1, c_2, ..., c_n]$ to specify the cell count for each class. During training, the predicted noise-free layout $\hat{x}_0^t$ is obtained via Eq. 2, and Straight-Through Estimator (STE) is used to make the binarization operation differentiable. The loss is computed as $\mathcal{L}_{\text{count}} = \frac{1}{n}\sum_{i=1}^n |\frac{\sum b(\hat{x}_0^t)^{(i)}}{\delta} - \frac{\sum x_0^{(i)}}{\delta}|$.
- **Design Motivation**: Relying solely on the conditional vector is insufficient for precise control over cell counts; STE allows discrete counting operations to be trained end-to-end, where $\delta$ represents the area of a single cell ($3 \times 3$).

**2. Intra-class Spatial Consistency based on Persistent Homology**
- **Function**: Maintain the spatial distribution pattern of each cell class itself (e.g., ring-like clusters of epithelial cells).
- **Mechanism**: Computes the Euclidean Distance Transform (EDT) map (distance from each pixel to the nearest cell) for both predicted and ground-truth layouts. Then, the 1D persistent homology persistence diagram is computed on the EDT map, measuring the difference between the two diagrams using the Wasserstein distance: $\mathcal{L}_{\text{intra}} = \frac{1}{n}\sum_{i=1}^n \mathcal{L}_{\text{spc}}(Dgm((\hat{x}_t^{edt})^{(i)}), Dgm((x_0^{edt})^{(i)}))$.
- **Design Motivation**: Persistent homology captures topological features (connected components, loops, voids) in a multi-scale manner, describing cell distribution patterns more comprehensively than simple spatial statistics.

**3. Inter-class Structural Regularization**
- **Function**: Capture spatial relationships between different cell types (e.g., clustering patterns of immune cells around tumors).
- **Mechanism**: Merges all channels into a single-channel aggregated layout $x_0^{agg} = Agg(x_0)$, similarly computes distance transform and persistent homology, and enforces constraints on overall topology via $\mathcal{L}_{\text{inter}} = \mathcal{L}_{\text{spc}}(Dgm(\hat{x}_{t,agg}^{edt}), Dgm(x_{0,agg}^{edt}))$.
- **Design Motivation**: While intra-class loss only examines the distribution of single cell types, the inter-class loss captures cross-type spatial interactions (e.g., contact patterns between immune cells and tumor cells) via the aggregated view.

### Loss & Training

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{simple}} + \lambda_c \mathcal{L}_{\text{count}} + \lambda_{\text{intra}} \mathcal{L}_{\text{intra}} + \lambda_{\text{inter}} \mathcal{L}_{\text{inter}}$$

Where $\mathcal{L}_{\text{simple}} = \mathbb{E}_{t,x_0,\epsilon}[\|\epsilon - \epsilon_\theta(x_t, c, t)\|^2]$ is the standard DDPM objective.

### Evaluation Metric: Topological Fréchet Distance (TopoFD)

Extracts features from persistence diagrams of both real and generated layouts, and computes topological similarity using Fréchet distance similar to FID, overcoming FID's limitations in topological evaluation.

## Key Experimental Results

### Main Results: Cell Layout Generation on CoNSeP Dataset

| Method | FID ↓ | TopoFD ↓ | Count Error ↓ |
|------|:---:|:---:|:---:|
| DDPM (baseline) | High | High | High |
| + Cell Count Loss | Lower | Moderate | **Substantially Reduced** |
| + Intra-class Topo | Lower | Lower | Lower |
| + Inter-class Topo (Full) | **Lowest** | **Lowest** | **Lowest** |

### Downstream Tasks: Cell Detection and Classification (Data Augmentation)

| Augmentation Strategy | Detection F1 | Classification F1 |
|------|:---:|:---:|
| No Augmentation | baseline | baseline |
| GAN Augmentation | Slight Improvement | Slight Improvement |
| DDPM Augmentation | Moderate Improvement | Moderate Improvement |
| **TopoCellGen Augmentation** | **Largest Improvement** | **Largest Improvement** |

### Key Findings

- The combination of the three topological constraints yields complementary effects, with the full model achieving optimal performance across all metrics.
- Using generated layouts for data augmentation significantly improves downstream cell detection and classification tasks.
- TopoFD reflects topological quality differences in layouts better than FID.
- Cell Counting Loss is critical for precisely controlling cell density—relying solely on the conditional vector is insufficient.
- The model successfully captures key topological patterns in real tissues (e.g., glandular ring structures, immune infiltration patterns).

## Highlights & Insights

1. **Topological Constraints in Generative Models**: For the first time, persistent homology is systematically applied to multi-class cell layout generation, focusing on both intra-class distribution and inter-class interaction.
2. **TopoFD Evaluation Metric**: Fills the gap in topological fidelity evaluation, establishing a standard benchmark for future work.
3. **Controllable and Interpretable**: The generated layouts correspond directly to cell distribution patterns understandable by pathologists, providing greater interpretability than directly generated images.
4. **Practical Downstream Value Chain**: Layout generation -> conditional image synthesis -> data augmentation -> enhanced detection/classification performance, forming a complete application pipeline.

## Limitations & Future Work

- The computation of distance transforms requires STE approximation in backpropagation, which might lack precision.
- Current cells are represented as fixed-size squares ($3 \times 3$), without modeling morphological diversity.
- Persistent homology computations are relatively slow on large-scale layouts.
- Future work can extend this to 3D histology and introduce fine-grained cell morphology and functional states.

## Related Work & Insights

- **TopoDiffusionNet**: Integrates topology and diffusion models in natural images, but does not address multi-class interactions.
- **TopoGAN**: A pioneer of topological loss in GAN frameworks.
- **Abousamra et al.**: GAN-based cell layout generation using spatial statistics and topological descriptors.
- **Persistent Homology**: A multi-scale topological analysis theory proposed by Edelsbrunner et al.
- **Insight**: Explicitly introducing domain-specific structural constraints inside generative models guarantees correct key properties better than purely data-driven methods.

## Rating

⭐⭐⭐⭐ — First to systematically integrate persistent homology into cell layout diffusion generation, featuring a precise problem definition and clear clinical value. The TopoFD metric addresses an important gap. The design of intra- and inter-class topological constraints is elegant and complementary. The primary limitations lie in the computational overhead and simplified cell representation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DiN: Diffusion Model for Robust Medical VQA with Semantic Noisy Labels](din_diffusion_model_for_robust_medical_vqa_with_semantic_noisy_labels.md)
- [\[NeurIPS 2025\] Semantic and Visual Crop-Guided Diffusion Models for Heterogeneous Tissue Synthesis in Histopathology](../../NeurIPS2025/medical_imaging/semantic_and_visual_crop-guided_diffusion_models_for_heterogeneous_tissue_synthe.md)
- [\[AAAI 2026\] Graph-Theoretic Consistency for Robust and Topology-Aware Semi-Supervised Histopathology Segmentation](../../AAAI2026/medical_imaging/graph-theoretic_consistency_for_robust_and_topology-aware_semi-supervised_histop.md)
- [\[ECCV 2024\] Co-synthesis of Histopathology Nuclei Image-Label Pairs using a Context-Conditioned Joint Diffusion Model](../../ECCV2024/medical_imaging/co-synthesis_of_histopathology_nuclei_image-label_pairs_using_a_context-conditio.md)
- [\[CVPR 2025\] SeaLion: Semantic Part-Aware Latent Point Diffusion Models for 3D Generation](sealion_semantic_part-aware_latent_point_diffusion_models_for_3d_generation.md)

</div>

<!-- RELATED:END -->
