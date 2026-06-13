---
title: >-
  [Paper Note] GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis
description: >-
  [CVPR 2026][Medical Imaging][Multi-heterogeneous graph] This paper proposes the GIIM framework, which constructs a Multi-Heterogeneous Graph (MHG) with four types of edge relations to simultaneously model the dynamic cha…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Multi-heterogeneous graph"
  - "multi-view diagnosis"
  - "intra-/inter-view dependencies"
  - "missing view handling"
  - "CADx"
date: 2026-05-08
content_hash: e83b7387b3547867
---

# GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis

**Conference**: CVPR 2026
**arXiv**: [2603.09446](https://arxiv.org/abs/2603.09446)  
**Code**: Unavailable  
**Area**: Medical Image Analysis / Graph Neural Networks / Computer-Aided Diagnosis
**Keywords**: Multi-heterogeneous graph, multi-view diagnosis, intra-/inter-view dependencies, missing view handling, CADx

## TL;DR

This paper proposes the GIIM framework, which constructs a Multi-Heterogeneous Graph (MHG) with four types of edge relations to simultaneously model the dynamic changes of individual lesions across imaging phases and the spatial associations among different lesions. Four missing-view imputation strategies are designed. GIIM achieves significant improvements over existing methods on three modalities: liver CT, breast mammography, and breast MRI.

## Background & Motivation

**Background**: Clinical diagnosis requires integrating complex dependencies among abnormalities across multiple views — including the dynamic enhancement pattern of a single lesion across multi-phase CT and the spatial co-occurrence among different lesions. CNN, Transformer, and GNN-based methods have made progress in single-view or simple multi-view fusion settings.

**Limitations of Prior Work**:

1. Existing CADx methods typically process each view independently or simply concatenate features, ignoring intra-view multi-lesion relationships and inter-view temporal/spatial dynamics.
2. Attention-based methods require fixed-size inputs and cannot flexibly handle a variable number of lesions.
3. Missing views are common in clinical practice due to protocol constraints, technical failures, or patient-related factors, yet existing methods lack robustness to such scenarios.

**Key Challenge**: There is a need to simultaneously model four types of dependencies (same lesion across views, different lesions within a single view, different lesions across multiple views, and single-to-multi-view aggregation), while maintaining robustness under missing-view conditions.

**Goal**: Reformulate multi-view medical diagnosis as a relational modeling problem, comprehensively capturing four types of dependencies via heterogeneous graphs while handling missing data.

**Key Insight**: GNNs are naturally suited for variable-size node sets and heterogeneous relational modeling, enabling different node and edge types to encode distinct levels of clinical relationships.

**Core Idea**: Construct each patient's multi-lesion, multi-view data as a multi-heterogeneous graph and perform type-aware message passing to jointly reason over all four dependency types.

## Method

### Overall Architecture

Two-stage training: (1) A ConvNeXt feature extractor is independently trained for each view, leveraging 7×7 large-kernel convolutions and depth-wise separable convolutions to capture morphological and intensity details. (2) The feature extractor is frozen, and multi-lesion, multi-view features are organized into a Multi-Heterogeneous Graph (MHG), over which a heterogeneous message-passing GNN performs relational reasoning and classification.

### Key Designs

1. **Dual-type Node Representation**

    - **Single-view node** $N_{single}^v = f_v(l_v)$: the feature of a lesion under a specific view.
    - **Multi-view node** $M_{multi} = \|_{v=1}^V N_{single}^v$: an aggregated node formed by concatenating features from all views.
    - For breast mammography, where lesion correspondences between CC and MLO views are uncertain, single-view nodes are replaced by the mean of all lesion features within that view.

2. **Four Types of Edge Relations**

    - $E_{intra}$: Same lesion across different views (e.g., arterial → venous → delayed phase) → captures temporal enhancement dynamics.
    - $E_{s-m}$: Single-view nodes to their corresponding multi-view aggregation node → integrates information across phases.
    - $E_{inter-s}$: Different lesions within the same view → models spatial co-occurrence (e.g., HCC commonly co-occurs with satellite nodules).
    - $E_{inter-m}$: Between aggregated nodes of different lesions → captures high-level inter-lesion contextual relationships, allowing small lesions to leverage context from nearby larger lesions.

3. **Heterogeneous Message Passing**

    - For each node, messages are aggregated separately from single-view neighbors and multi-view neighbors (each with independent weight matrices $\mathbf{W}_{single}^k$ and $\mathbf{W}_{multi}^k$), concatenated with the node's previous state, and updated via a nonlinear transformation: $h_n^k = \sigma(\mathbf{W}^k \cdot \text{CONCAT}(h_n^{k-1}, h_{N_{single}(n)}^k, h_{M_{multi}(n)}^k))$
    - Five layers of SAGEConv (512→256→128→64→number of classes), with the final layer directly outputting classification probabilities.

4. **Four Missing-View Imputation Strategies**

    - **Constant**: Zero-vector padding — simple, but encourages the model to learn to ignore missing nodes.
    - **Learnable**: Learnable parameters normalized via the Frobenius norm.
    - **RAG-based**: Retrieves the most similar complete sample from a database using available features, and borrows its missing-view features.
    - **Covariance-based**: Computes inter-view feature-difference covariance to measure sample similarity, then selects the most similar complete sample for imputation.

### Loss & Training

- Single-view stage: Standard cross-entropy classification loss; ConvNeXt is trained independently and then frozen.
- Graph model stage: MHG is trained end-to-end; graphs are constructed per patient (one patient graph per batch).

## Key Experimental Results

### Main Results

| Dataset | Method | Accuracy (%) | AUC (%) |
|--------|------|-------------|---------|
| Liver CT | NN-based (multi-view) | 75.45 | 89.09 |
| | Attention-based | 73.41 | 88.53 |
| | **GIIM** | **78.20** | **91.05** |
| VinDr-Mammo | NN-based | 67.48 | 82.21 |
| | Attention-based | 68.09 | 81.00 |
| | **GIIM** | **71.17** | **82.54** |
| BreastDM (MRI) | NN-based | 80.85 | 87.35 |
| | Attention-based | 85.10 | 76.37 |
| | **GIIM** | **87.23** | **89.02** |

Multi-view vs. single-view: approximately +12% accuracy on Liver CT and +7.8% on Mammography.

### Ablation Study

**Missing-view strategy comparison (Liver, 100% missing-view test)**

| Strategy | 100% miss-view | Full-view |
|------|----------------|-----------|
| NN-based | 70.00 | 75.45 |
| GIIM (Constant) | **72.27** | 78.20 |
| GIIM (Learnable) | 72.05 | 77.05 |
| GIIM (RAG) | 71.59 | **78.41** |
| GIIM (Covariance) | 72.05 | 78.18 |

**Edge type ablation**: Removing any of the four edge types leads to performance degradation; $E_{intra}$ (same lesion across phases) has the largest impact.

### Key Findings

- Zero-vector imputation is the most stable under missing-view testing (serving as a unique "missing indicator" that trains the model to rely on other views), while RAG/Covariance imputation performs better on complete data.
- Multi-view consistency yields the greatest gains on Liver CT (4-phase CT with significant enhancement pattern changes across phases for the same lesion).
- For BI-RADS classification, mean pooling is adopted instead of per-lesion graph construction due to the uncertain correspondence between CC and MLO view lesions.

## Highlights & Insights

- The four-type edge design comprehensively covers the relational reasoning patterns employed by clinical radiologists, offering greater interpretability than simple attention decomposition.
- The practical trade-off finding for missing-view strategies is actionable: generative imputation performs better on complete data, while zero-vector imputation is more robust under missing-view conditions.
- The heterogeneous message-passing aggregation scheme — aggregating separately from single-view and multi-view neighbors — prevents information loss due to edge-type conflation.
- The flexibility of GNNs allows the framework to handle an arbitrary number of lesions and views, outperforming CNN/Transformer architectures that require fixed-size inputs.

## Limitations & Future Work

- The single-view feature extractor and graph model are trained in separate stages; joint end-to-end training may yield further improvements.
- The graph structure is hard-coded by data (determined by the number of lesions and views); dynamic graph construction or attention-weighted edges remain unexplored.
- ConvNeXt serves as a relatively conservative backbone; stronger alternatives such as ViT or SAM may further improve performance.
- The three datasets are relatively small in scale (the largest contains 920 cases), limiting the generalizability of the validation.

## Related Work & Insights

- **vs. Phase Attention (Wang et al. 2022)**: Performs intra-phase and inter-phase attention but handles fixed-size inputs and ignores inter-lesion relationships; GIIM uses GNNs to flexibly accommodate variable numbers of lesions.
- **vs. SSL-MNGCN (Ibrahim et al. 2022)**: Applies GCN to texture/spatial feature maps in mammography but does not model cross-view temporal relationships.
- **vs. mmFormer (Zhang et al. 2022)**: A multi-modal Transformer for incomplete brain tumor segmentation, targeting voxel-level tasks rather than lesion-level classification.
- Insight: The heterogeneous graph relational modeling paradigm is generalizable to other scenarios requiring joint multi-view/multi-modal reasoning.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of four heterogeneous edge types and missing-view imputation strategies constitutes a complete and well-motivated design.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers three modalities, missing-view ablation, and comparisons of four imputation strategies.
- Writing Quality: ⭐⭐⭐ — Content is detailed but the structure is somewhat complex.
- Value: ⭐⭐⭐ — Provides a general framework for multi-view medical diagnosis, though the limited dataset scale constrains its persuasiveness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding](medgrpo_multi-task_reinforcement_learning_for_heterogeneous_medical_video_unders.md)
- [\[CVPR 2026\] RelativeFlow: Taming Medical Image Denoising Learning with Noisy Reference](relativeflow_taming_medical_image_denoising_learning_with_noisy_reference.md)
- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)
- [\[CVPR 2026\] Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision](learning_generalizable_3d_medical_image_representations_from_mask-guided_self-su.md)
- [\[CVPR 2026\] Continual Learning for fMRI-Based Brain Disorder Diagnosis via Functional Connectivity Matrices Generative Replay](forge_continual_learning_for_fmri_based_brain_disorder_diagnosis.md)

</div>

<!-- RELATED:END -->
