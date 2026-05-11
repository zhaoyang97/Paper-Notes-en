---
title: >-
  [Paper Note] Vector Contrastive Learning for Pixel-wise Pretraining in Medical Vision
description: >-
  [ICCV 2025][Medical Imaging][Contrastive Learning] This paper proposes Vector Contrastive Learning (Vector CL), which reformulates standard contrastive learning from a binary optimization problem into a vector regression problem. By modeling feature distances to quantify the degree of dispersion, it addresses the over-dispersion problem in pixel-wise medical vision pretraining, achieving significant improvements over 17 methods across 8 downstream tasks.
tags:
  - ICCV 2025
  - Medical Imaging
  - Contrastive Learning
  - Pixel-wise Pretraining
  - Medical Vision Foundation Model
  - Displacement Vector Regression
  - Over-dispersion Problem
date: 2026-05-08
content_hash: 114a0ab2b6904232
---

# Vector Contrastive Learning for Pixel-wise Pretraining in Medical Vision

**Conference**: ICCV 2025
**arXiv**: [2506.20850](https://arxiv.org/abs/2506.20850)
**Code**: [GitHub](https://github.com/YutingHe-list/COVER)
**Area**: Medical Imaging
**Keywords**: Contrastive Learning, Pixel-wise Pretraining, Medical Vision Foundation Model, Displacement Vector Regression, Over-dispersion Problem

## TL;DR

This paper proposes Vector Contrastive Learning (Vector CL), which reformulates standard contrastive learning from a binary optimization problem into a vector regression problem. By modeling feature distances to quantify the degree of dispersion, it addresses the over-dispersion problem in pixel-wise medical vision pretraining, achieving significant improvements over 17 methods across 8 downstream tasks.

## Background & Motivation

Contrastive learning (CL) is a core paradigm for self-supervised pretraining; however, extending it to pixel-level representations—which is critical for medical vision—remains an open problem. The central obstacle is the **over-dispersion problem**:

Standard binary CL formulates pretraining as a binary optimization (pulling positive pairs together and pushing negative pairs apart) without modeling the degree of dispersion, causing features to be excessively scattered. This is particularly severe for pixel-level tasks, where pixel-wise features are naturally distributed across image grids with semantically continuous and intrinsically correlated variations. Over-dispersion in binary CL destroys these correlations, disrupts intra-class distributions, and makes it difficult for the model to disentangle underlying semantics.

**Core Insight**: Feature distances inherently encode semantic correspondences and can be represented by displacement vectors in image space. Rather than directly minimizing $|\alpha - d'|$ to model the embedding-space distance $d'$, one can construct a function $\mathcal{V}$ that associates distances with vectors $v'$ in image space, reformulating CL as a vector regression problem $|v - \mathcal{V}(d')|$, where $v$ is an accessible ground-truth vector.

## Method

### Overall Architecture

The COVER (**CO**ntrast in **VE**ctor **R**egression) framework implements Vector CL through three key innovative modules:

1. **SeVR (Self-Vector Regression)**: Establishes a scalable self-learning paradigm.
2. **MoV (Mixture of Vectors)**: Constructs a consistent optimization flow from vector regression to distance modeling.
3. **VPA (Vector Pyramid Aggregation)**: Enables pyramid-based multi-scale correspondence modeling.

### Key Designs

1. **SeVR — Self-Vector Regression**: Given a medical image $x$, two views $x_a = t(x)$ and $x_b = \psi_{ab}(x)$ are generated via random spatial transformations $\mathcal{T}_{sp}$. The spatial transformation itself yields a displacement vector field (DVF) $\psi_{ab} = \{v^i\}_{i \in \Omega}$ as annotation-free ground truth. A shared-weight network $\mathcal{N}_\theta$ extracts multi-scale features $F_a, F_b$, and function $\mathcal{V}$ predicts DVF $\psi'_{ab}$, jointly optimized via vector loss and consistency loss:

    - $\mathcal{L}_{vec} = \sum_{i \in \{\epsilon_{ab}=1\}} |\psi^i_{ab} - \psi'^i_{ab}|$
    - $\mathcal{L}_{con}$: maintains semantic invariance under spatial transformation via cosine similarity.

2. **MoV — Mixture of Vectors**: Comprises two sub-modules:

    - **VEU (Vector Embedding Unit)**: Within an $N \times N$ receptive field, scaled dot-product attention is computed between the center feature $f^i_a$ and the target feature set $f^{N \times N}_b$ to obtain a distance map $D^{N \times N}$. A **fixed vector template matrix** $\mathbb{V}^{N \times N}$ is designed to encode spatially continuous relationships, mapping distances to vectors via $v'_{ab} = \text{softmax}(f^i_a f^{\top}_{b} / \tau) \mathbb{V}^{N \times N}$, thereby avoiding artificial partitioning while preserving feature correlations.
    - **MVI (Multi-Vector Integration)**: The $C$-channel features are split into $J$ groups, each independently generating a vector; the results are averaged to accommodate correspondence ambiguity and enhance bias adaptability.

3. **VPA — Vector Pyramid Aggregation**: Stacks MoV modules within a pyramid architecture with coarse-to-fine chained computation: $\psi'^0_{ab} = \mathcal{M}(f^0_a, f^0_b)$, $\psi'^l_{ab} = \mathcal{M}(\psi'^{l-1}_{ab}(f^l_a), f^l_b) \bigodot \psi'^{l-1}_{ab}$. Higher levels capture global correspondences while lower levels refine local ones, achieving a large effective receptive field under a small local receptive field with high computational efficiency.

### Loss & Training

- Overall objective: $\mathcal{L}_{COVER} = \mathcal{L}_{con} + \mathcal{L}_{vec}$
- Optimized with SGD, learning rate $10^{-4}$, for $2 \times 10^5$ iterations.
- Theoretical basis: Vector CL has a tighter generalization bound than binary CL: $\delta_{VCL} \leq \tau \log(1/\alpha_{min}) \ll \Delta$.

## Key Experimental Results

### Main Results

| Method | Type | SCR(S) | PDCXR(C) | KiPA22(S) | FIVES(S) | CANDI(S) | FeTA21(S) | KiPA22-3D(S) | STOIC(C) | Avg. |
|--------|------|--------|----------|-----------|----------|----------|-----------|--------------|----------|------|
| Scratch | - | 81.8 | 90.4 | 74.1 | 79.4 | 84.0 | 56.9 | 72.4 | 72.0 | 76.4 |
| SimCLR | BCL | 89.0 | 94.7 | 74.4 | 84.5 | 89.2 | 53.4 | 78.9 | 60.7 | 78.1 |
| PixPro | DBCL | 91.5 | 93.0 | 73.6 | 84.3 | 89.9 | 60.7 | 80.0 | 75.1 | 81.0 |
| GEMINI | VR | 92.4 | 92.9 | 79.1 | 85.3 | 90.0 | 61.7 | 85.0 | 79.5 | 83.2 |
| **COVER** | **VCL** | **94.0** | **95.9** | **80.0** | **87.2** | **89.9** | **63.6** | **85.2** | **80.4** | **84.5** |

COVER outperforms Scratch on all 8 tasks with an average gain of 8.1%, and is the only method that achieves positive gains across all tasks.

### Ablation Study

| Component | SCR DSC% |
|-----------|----------|
| Base ($\mathcal{L}_{con}$ only) | 91.8 |
| + VEU (SeVR) | 92.9 (+1.1) |
| + VPA | 93.4 (+0.5) |
| + MVI | **94.0 (+0.6)** |

Hyperparameter ablation: receptive field $N=7\times7$ is optimal (54.8%); VEU count $J=[4,4,4,1,1]$ is optimal (56.3%).

### Key Findings

- **Cross-scale transferability**: Significant improvements are observed for both small targets (vessels, brain tissue) and large targets (chest, kidneys).
- **Cross-scenario adaptability**: Even when the pretraining data domain is inconsistent with the downstream task (e.g., chest X-ray pretraining → kidney CT), COVER still transfers effectively.
- Using only 5% of training data, COVER approaches the performance of GVSL trained on 25% of data.
- VPA achieves an effective receptive field of 121×121 with only 1/52 of the computation of a direct approach.
- t-SNE visualization shows that COVER features exhibit a smooth and continuous distribution, effectively aggregating semantically similar features.

## Highlights & Insights

- **Paradigm Innovation**: This work is the first to reformulate contrastive learning from a binary problem into a vector regression problem, with rigorous mathematical equivalence proofs.
- The design of the fixed vector template matrix $\mathbb{V}$ is elegant—fixed, parameter-free, and naturally encoding spatial continuity.
- The self-spatial-transformation mechanism in SeVR eliminates dependence on paired data (unlike GVSL and GEMINI), scaling to arbitrary medical images.
- Theoretical analysis demonstrates that Vector CL yields a tighter Rademacher complexity generalization bound than binary CL.

## Limitations & Future Work

- Only U-Net is used as the backbone; exploring larger-scale architectures (e.g., ViT) is a natural next step.
- The pretraining data scale is limited (~112k 2D images, 837 3D volumes); scaling up is expected to yield further gains.
- DVF generation via affine transformations may be insufficiently diverse; non-rigid transformations could potentially improve pretraining quality.
- The equivalence between vector regression and distance modeling is approximate rather than exact (relying on a weight normalization distribution).

## Related Work & Insights

- Compared to GVSL and GEMINI, COVER is the first to explicitly establish a mapping function from distance to vector, achieving a consistent optimization flow.
- Comparisons with dense binary CL methods such as DenseCL reveal the severity of the over-dispersion problem.
- The proposed methodology is generalizable to domains requiring pixel-level understanding, such as remote sensing and satellite imagery.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Novel CL paradigm with rigorous theoretical foundation)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (8 tasks, 4 modalities, 17 competing methods, and comprehensive ablation studies)
- Writing Quality: ⭐⭐⭐⭐⭐ (Complete mathematical derivations and clear motivation)
- Value: ⭐⭐⭐⭐⭐ (Significant advance for medical vision foundation models)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GECKO: Gigapixel Vision-Concept Contrastive Pretraining in Histopathology](gecko_gigapixel_vision-concept_contrastive_pretraining_in_histopathology.md)
- [\[ICCV 2025\] An OpenMind for 3D Medical Vision Self-Supervised Learning](an_openmind_for_3d_medical_vision_selfsupervised_learning.md)
- [\[CVPR 2026\] MedKCO: Medical Vision-Language Pretraining via Knowledge-Driven Cognitive Orchestration](../../CVPR2026/medical_imaging/medkco_medical_vision-language_pretraining_via_knowledge-driven_cognitive_orches.md)
- [\[ICCV 2025\] Boosting Vision Semantic Density with Anatomy Normality Modeling for Medical Vision-language Pre-training](boosting_vision_semantic_density_with_anatomy_normality_modeling_for_medical_vis.md)
- [\[NeurIPS 2025\] VQ-Seg: Vector-Quantized Token Perturbation for Semi-Supervised Medical Image Segmentation](../../NeurIPS2025/medical_imaging/vq-seg_vector-quantized_token_perturbation_for_semi-supervised_medical_image_seg.md)

</div>

<!-- RELATED:END -->
