---
title: >-
  [Paper Note] Beyond Single Images: Retrieval Self-Augmented Unsupervised Camouflaged Object Detection
description: >-
  [ICCV 2025][Segmentation][camouflaged object detection] This paper proposes RISE — a retrieval self-augmented unsupervised camouflaged object detection paradigm that constructs foreground/background prototype libraries f…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "camouflaged object detection"
  - "unsupervised segmentation"
  - "retrieval-augmented"
  - "KNN"
  - "prototype library"
date: 2026-05-08
content_hash: d9690d714199169e
---

# Beyond Single Images: Retrieval Self-Augmented Unsupervised Camouflaged Object Detection

**Conference**: ICCV 2025
**arXiv**: [2510.18437](https://arxiv.org/abs/2510.18437)  
**Code**: [https://github.com/xiaohainku/RISE](https://github.com/xiaohainku/RISE)  
**Area**: Segmentation / Camouflaged Object Detection / Unsupervised
**Keywords**: camouflaged object detection, unsupervised segmentation, retrieval-augmented, KNN, prototype library

## TL;DR

This paper proposes RISE — a retrieval self-augmented unsupervised camouflaged object detection paradigm that constructs foreground/background prototype libraries from the training set itself and leverages KNN retrieval to generate pseudo-labels, substantially outperforming existing unsupervised and prompt-based methods without any annotations.

## Background & Motivation

**Background**: Camouflaged Object Detection (COD) aims to segment target objects from highly similar backgrounds. Mainstream fully supervised methods rely on dense pixel-level annotations, where annotating a single image can take up to one hour. Weakly and semi-supervised approaches reduce annotation burden but still require partial labels.

**Limitations of Prior Work**: (a) Unsupervised methods (TokenCut, MaskCut, ProMerge, etc.) primarily exploit intra-image feature similarity to separate foreground from background, but camouflaged objects and backgrounds share highly similar features, causing single-image methods to perform poorly; (b) prompt-based methods combining SAM with task-specific prompts still require some form of supervision and offer limited context-specific understanding of COD; (c) methods that generate pseudo-labels via diffusion models or multimodal LLMs (GenSAM, ProMac) require days of computation and significant GPU resources.

**Key Challenge**: Within a single image, the DINOv2 features of camouflaged objects and backgrounds are nearly indistinguishable (almost overlapping in t-SNE visualizations) — intra-image similarity alone cannot effectively separate them. However, at the dataset level, foreground objects exhibit higher similarity to a foreground prototype library than to a background prototype library.

**Goal**: To leverage dataset-level contextual information to distinguish camouflaged foregrounds from backgrounds in a fully annotation-free setting.

**Key Insight**: Mining prototypes directly from the dataset through a coarse-to-fine strategy — first obtaining coarse masks via clustering, then refining prototypes through retrieval — to construct high-quality foreground/background prototype libraries, followed by KNN-based retrieval to classify each feature as foreground or background.

**Core Idea**: Rather than relying on intra-image similarity, RISE leverages a dataset-level prototype library combined with KNN retrieval to distinguish camouflaged objects from backgrounds, enabling unsupervised COD.

## Method

### Overall Architecture

RISE operates in two stages: (1) **Clustering-then-Retrieval (CR)** — spectral clustering is applied to each image to generate coarse masks; foreground/background global features are extracted; high-confidence prototypes are selected via cross-category retrieval and aggregated into a prototype library; (2) **Multi-View KNN Retrieval (MVKR)** — DINOv2 features are extracted per image; each local feature retrieves the top-K most similar prototypes from the library and votes for foreground or background; multi-view fusion eliminates artifacts; the resulting pseudo-masks are used to train SINet-V2.

### Key Designs

1. **Clustering-then-Retrieval (CR) — Prototype Library Construction**:

    - **Function**: Construct a high-quality foreground/background prototype library from unannotated COD data.
    - **Mechanism**:
        - **Spectral clustering for coarse masks**: A feature similarity graph $\mathcal{G}$ is constructed with adjacency matrix $\mathbf{W}_{i,j} = \max(\text{cos}(\mathbf{F}'_i, \mathbf{F}'_j), 0)$; the normalized Laplacian $\mathbf{L} = \mathbf{D}^{-1/2}(\mathbf{D}-\mathbf{W})\mathbf{D}^{-1/2}$ is computed; eigenvectors are used for KMeans binary classification; the cluster with a lower boundary pixel ratio is assigned as foreground.
        - **Cross-Category Retrieval**: Rather than selecting the foreground prototype most similar to the foreground global feature, the method selects the one least similar to the background global feature: $\mathbf{P}^f = \arg\min_{\mathbf{s} \in \mathbf{S}_f} \text{cos}(\mathbf{s}, \mathbf{F}^g_b)$. This enhances discriminability between foreground and background prototypes.
        - **Histogram-Adaptive Filtering**: The distribution of foreground–background global feature similarities across all images is computed; images of poor quality are filtered out using the histogram peak as a threshold.
    - **Design Motivation**: Cross-category retrieval is the critical component — intuitively, "least similar to the other class" provides stronger discriminative guarantees than "most similar to own class." Ablation studies confirm a 5–8% improvement attributable to this design.

2. **Multi-View KNN Retrieval (MVKR)**:

    - **Function**: Generate high-quality pseudo-masks for each image using the prototype library.
    - **Mechanism**: For each feature $\mathbf{F}_{i,j}$, the top-K ($K=512$) most similar prototypes are retrieved from both foreground and background libraries, and a vote determines the class assignment. To eliminate artifacts in DINOv2 feature maps, multiple views (flips and rotations) of the same image are generated; each view is independently retrieved and inverse-transformed before aggregated voting.
    - **Design Motivation**: DINOv2 feature maps contain position-dependent artifacts whose locations shift across different views. Multi-view fusion eliminates these artifacts without requiring additional model fine-tuning.
    - **Implementation Details**: FAISS is used to accelerate retrieval; all images are uniformly resized to $476 \times 476$.

3. **Pseudo-Label Training**:

    - **Function**: Train a standard COD model using the generated pseudo-masks.
    - **Mechanism**: The generated pseudo-masks are directly used as ground truth to train SINet-V2 following the standard fully supervised training pipeline.
    - **Design Motivation**: RISE focuses on pseudo-label generation quality; the downstream training component is orthogonal and interchangeable with existing methods.

### Loss & Training

RISE itself requires no training — it only performs pseudo-label generation. The downstream SINet-V2 follows the standard COD training strategy. The feature extractor is DINOv2-ViT-L14 (frozen).

## Key Experimental Results

### Main Results

Comparison with unsupervised methods on four COD benchmarks (DINOv2-ViT-L14 feature extractor):

| Method | CHAMELEON $S_\alpha$↑ | COD10K $S_\alpha$↑ | COD10K $F^\omega_\beta$↑ | NC4K $S_\alpha$↑ |
|------|--------|---------|---------|--------|
| RISE | **0.822** | **0.763** | **0.600** | **0.805** |
| ProMerge | 0.741 | 0.674 | 0.435 | 0.726 |
| TokenCut | 0.708 | 0.637 | 0.370 | 0.697 |
| VoteCut | 0.679 | 0.645 | 0.390 | 0.674 |
| DiffCut | 0.574 | 0.628 | 0.372 | 0.693 |

Comparison with prompt-based methods (with SAM integration):

| Method | CHAMELEON $S_\alpha$↑ | COD10K $S_\alpha$↑ | COD10K $F^\omega_\beta$↑ | NC4K $S_\alpha$↑ |
|------|--------|---------|---------|--------|
| RISE+SAM | **0.823** | **0.790** | **0.643** | **0.825** |
| WS-SAM* | 0.795 | 0.787 | 0.622 | 0.829 |
| ProMac | 0.786 | 0.774 | 0.609 | 0.812 |
| GenSAM | 0.659 | 0.641 | 0.390 | 0.702 |

### Ablation Study

| Configuration | COD10K $S_\alpha$ | COD10K $E_\phi$ | COD10K $F^\omega_\beta$ | COD10K $M$ |
|------|--------|---------|---------|------|
| (e) Full RISE | **0.763** | **0.840** | **0.600** | **0.049** |
| (a) Image-level only (spectral clustering) | 0.641 | 0.662 | 0.414 | 0.169 |
| (b) Without cross-category retrieval | 0.710 | 0.781 | 0.518 | 0.065 |
| (c) Without histogram filtering | 0.744 | 0.822 | 0.575 | 0.055 |
| (d) Without multi-view retrieval | 0.759 | 0.832 | 0.584 | 0.052 |

### Key Findings

- **Dataset-level information is critical**: Moving from image-level-only modeling to full RISE yields over 12% improvement in $S_\alpha$, demonstrating the decisive advantage of cross-image information over single-image similarity.
- **Cross-category retrieval contributes most**: Removing it causes a 5.3% drop in $S_\alpha$ and 8.2% drop in $F^\omega_\beta$ on COD10K.
- RISE surpasses WS-SAM, which uses manually annotated weak supervision signals, while reducing inference time from days to hours.
- The method is robust across different DINO variants: DINO-ViT-S16/B16 and DINOv2-S14/B14/L14 all yield effective results.

## Highlights & Insights

- **Retrieval self-augmentation paradigm**: Rather than relying on external data sources, RISE constructs a prototype library from the dataset itself — a "self-bootstrapping" strategy particularly valuable for COD where annotation costs are extremely high.
- **Counterintuitive cross-category retrieval**: Selecting prototypes by "least similar to the opposing class" rather than "most similar to own class" significantly improves discriminability — a trick transferable to any scenario requiring contrastive prototype construction.
- **Multi-view artifact elimination**: By exploiting the property that DINOv2 artifact locations shift across views, simple flip/rotation augmentation combined with voting removes artifacts far more efficiently than model fine-tuning.
- **Histogram-adaptive thresholding**: Using the peak of the similarity distribution for adaptive filtering eliminates the need for manual threshold selection.

## Limitations & Future Work

- The quality of coarse masks from spectral clustering is a bottleneck — poor initial segmentation degrades prototype quality.
- The top-K parameter ($K=512$) requires tuning (sensitivity analysis provided in Figure 5 of the paper).
- The current formulation supports only binary classification (foreground/background) and does not support multi-instance detection.
- Detection of extremely small objects remains challenging, though qualitative results show improvements over baselines.

## Related Work & Insights

- **vs. TokenCut/VoteCut**: These methods rely on Normalized Cut within individual images; the high foreground–background similarity in camouflaged scenes causes failure. RISE overcomes this limitation through dataset-level prototypes.
- **vs. ProMac/GenSAM**: These methods use diffusion models or multimodal LLMs to generate prompts, requiring days of computation. RISE achieves superior results in only a few hours.
- **vs. Retrieval-Augmented Semantic Segmentation (RASS)**: RASS relies on external models to generate prototype libraries, whereas RISE mines prototypes from the dataset itself, avoiding out-of-domain bias.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The retrieval self-augmentation paradigm is pioneered in COD; the cross-category retrieval strategy is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers four datasets, eight unsupervised methods, three prompt-based baselines, comprehensive ablations, and sensitivity analysis.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated; t-SNE visualizations are highly persuasive.
- Value: ⭐⭐⭐⭐⭐ Sets a new benchmark for unsupervised COD with ideas generalizable to other fine-grained segmentation tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SDDF: Specificity-Driven Dynamic Focusing for Open-Vocabulary Camouflaged Object Detection](../../CVPR2026/segmentation/sddf_specificity-driven_dynamic_focusing_for_open-vocabulary_camouflaged_object.md)
- [\[ICCV 2025\] Ensemble Foreground Management for Unsupervised Object Discovery](ensemble_foreground_management_for_unsupervised_object_discovery.md)
- [\[CVPR 2026\] FCL-COD: Weakly Supervised Camouflaged Object Detection with Frequency-aware and Contrastive Learning](../../CVPR2026/segmentation/fcl-cod_weakly_supervised_camouflaged_object_detection_with_frequency-aware_and_.md)
- [\[ICCV 2025\] Rethinking Detecting Salient and Camouflaged Objects in Unconstrained Scenes](rethinking_detecting_salient_and_camouflaged_objects_in_unconstrained_scenes.md)
- [\[NeurIPS 2025\] TabRAG: Improving Tabular Document Question Answering for Retrieval Augmented Generation via Structured Representations](../../NeurIPS2025/segmentation/tabrag_improving_tabular_document_question_answering_for_retrieval_augmented_gen.md)

</div>

<!-- RELATED:END -->
