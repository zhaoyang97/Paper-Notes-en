---
title: >-
  [Paper Note] Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference
description: >-
  [CVPR 2026][Medical Imaging][Spatial transcriptomics] This paper proposes SpaHGC, a multimodal heterogeneous graph framework that constructs three types of subgraphs—intra-target-slice (TS), cross-slice (CS), and intra-reference-slice (RS)—and integrates masked graph contrastive learning with a cross-node dual attention mechanism to predict spatial gene expression from H&E histopathology images, achieving PCC improvements of 7.3%–27.1% across seven datasets.
tags:
  - CVPR 2026
  - Medical Imaging
  - Spatial transcriptomics
  - heterogeneous graph learning
  - cross-slice knowledge transfer
  - contrastive learning
  - gene expression prediction
date: 2026-05-08
content_hash: 238d68990a0f2901
---

# Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference

**Conference**: CVPR 2026
**arXiv**: [2603.22821](https://arxiv.org/abs/2603.22821)
**Code**: [https://github.com/wenwenmin/SpaHGC](https://github.com/wenwenmin/SpaHGC)
**Area**: Medical Image Analysis / Spatial Transcriptomics
**Keywords**: Spatial transcriptomics, heterogeneous graph learning, cross-slice knowledge transfer, contrastive learning, gene expression prediction

## TL;DR
This paper proposes SpaHGC, a multimodal heterogeneous graph framework that constructs three types of subgraphs—intra-target-slice (TS), cross-slice (CS), and intra-reference-slice (RS)—and integrates masked graph contrastive learning with a cross-node dual attention mechanism to predict spatial gene expression from H&E histopathology images, achieving PCC improvements of 7.3%–27.1% across seven datasets.

## Background & Motivation

**State of the Field**: Spatial transcriptomics (ST) technology enables precise quantification of the spatial distribution of gene expression in tissues, but its high experimental cost limits large-scale application. Predicting ST gene expression from H&E histopathology images has emerged as a promising alternative.

**Limitations of Prior Work**: (1) ST data are sparse and noisy—gene expression at certain spots may be missing or near zero; (2) existing methods model spatial structure within a single slice only, ignoring expression patterns shared across slices; (3) inter-individual variability and disease progression introduce cross-sample heterogeneity that single-slice models struggle to capture in a generalizable manner.

**Root Cause**: Tissues of the same type or disease typically share common expression patterns, yet individual differences make direct cross-slice alignment difficult—how can shared information be effectively integrated while accommodating individual heterogeneity?

**Paper Goals**: To model cross-slice spatial relationships and transfer prior knowledge from multiple reference slices to improve gene expression prediction on a target slice.

**Starting Point**: Constructing a multimodal heterogeneous graph that connects cross-slice spots via image embeddings from a pathology foundation model (UNI), and leveraging contrastive learning to enhance the robustness of learned representations.

**Core Idea**: Heterogeneous graph + cross-slice knowledge transfer + masked contrastive learning = more accurate gene expression prediction.

## Method

### Overall Architecture
SpaHGC operates in four stages: (1) extracting patch embeddings using the UNI pathology foundation model; (2) constructing three types of subgraphs—intra-target-slice (TS), cross-slice (CS), and intra-reference-slice (RS); (3) applying complementary masking to generate two augmented views, which are trained via a heterogeneous graph encoder with contrastive learning; and (4) producing gene expression predictions.

### Key Designs

1. **Multimodal Heterogeneous Graph Construction**:

    - **Target-slice (TS) graph**: Connects the $Q$ nearest-neighbor spots within the target slice based on Euclidean distance, capturing local spatial continuity.
    - **Cross-slice (CS) graph**: For each target patch embedding $\mathbf{z}_t^{(i)}$, retrieves the Top-K patches from the reference slice by cosine similarity, forming cross-slice edges. Reference nodes carry joint features $\mathbf{h}_r = [\mathbf{z}_r \| \mathbf{y}_r]$ (visual + gene expression).
    - **Reference-slice (RS) graph**: Top-K connections among reference nodes based on cosine similarity of their joint features, forming a global semantic scaffold.
    - **Design Motivation**: The three edge types respectively capture local spatial semantics, cross-slice morphological similarity, and global expression relationships within the reference slice—enabling multi-level information fusion.

2. **Cross Node Dual Attention (CNDA)**:

    - **Function**: A bidirectional attention mechanism in which target nodes attend to reference nodes to acquire visual and gene expression knowledge, while reference nodes attend to target nodes to update their own representations.
    - **Core equations**: $\mathbf{A}_{t \leftarrow r} = \text{softmax}(\frac{\mathbf{Q}_t \mathbf{K}_r^\top}{\sqrt{d'}})$, $\bar{\mathbf{L}}_t = \mathbf{A}_{t \leftarrow r} \mathbf{V}_r$
    - **Design Motivation**: Selective transfer—dynamic attention weights allow the model to automatically select the most relevant morphological and expression information from the reference slice while suppressing irrelevant cross-slice noise.

3. **Cross Node Attention Pooling (CNAP)**:

    - **Function**: Multi-head unidirectional cross-node attention that aggregates reference node representations into target node representations via cross-attention.
    - **Design Motivation**: Compared with simple exemplar retrieval, CNAP dynamically aggregates auxiliary information according to the contextual semantics of each target node, enabling more flexible adaptation across different tissue regions.

4. **Complementary Masking Contrastive Learning**:

    - **Function**: Node-type-specific feature masking is applied separately to target and reference nodes to generate two complementary views ($\mathbf{M}_t^{(1)} + \mathbf{M}_t^{(2)} = \mathbf{1}$), and the model is trained with a cosine-distance contrastive loss.
    - **Design Motivation**: Simulates real-world feature dropout and sequencing noise present in ST data, forcing the model to learn consistent representations that are robust to such corruption.

### Loss & Training
- Contrastive loss: $\mathcal{L}_{\text{con}} = \frac{1}{N} \sum_j (2 - 2 \cdot \text{Cos}(\hat{L}_j^1, \hat{L}_j^2))$
- A regression loss is used for gene expression prediction.
- An asymmetric contrastive design is adopted in which one view is stop-gradient to serve as a stable target.

## Key Experimental Results

### Main Results (7 public ST datasets)

| Method | HER+ PCC% | cSCC PCC% | Lymph Node PCC% | Pancreas2 PCC% |
|--------|-----------|-----------|-----------------|----------------|
| STNet | 5.61 | 9.2 | 3.4 | 31.56 |
| HisToGene | 7.89 | 17.56 | 19.24 | 26.13 |
| mclSTExp | 23.15 | 31.88 | 21.64 | 31.61 |
| M2OST | 18.24 | 24.88 | 30.97 | 38.35 |
| **SpaHGC** | **27.86** | **38.79** | **35.02** | **41.36** |

### Ablation Study

Component contributions are validated by progressively removing modules from the full SpaHGC:
- Removing CNDA → notable PCC drop, confirming the critical role of cross-slice attention.
- Removing the CS graph → significant PCC drop, confirming the necessity of cross-slice connectivity.
- Removing masking → reduced robustness, confirming the contribution of contrastive learning.
- Replacing UNI with ResNet → clear PCC decrease, confirming the importance of a strong pathology foundation model.

### Key Findings
- SpaHGC achieves PCC gains of 7.3%–27.1% across all seven datasets, representing substantial improvements.
- Predicted expressions show significant enrichment in multiple cancer-related pathways, validating biological relevance.
- Cross-slice knowledge transfer generalizes across different platforms (10x Visium, ST 1000, etc.), tissue types, and cancer subtypes.

## Highlights & Insights
- **Cross-slice knowledge transfer**: Unlike existing methods that operate on a single slice, leveraging prior knowledge from multiple reference slices represents an important paradigm shift.
- **Deep integration of the pathology foundation model (UNI)**: Establishing cross-slice connections via strong pretrained embeddings fully exploits large-scale pretraining knowledge.
- **Biological downstream validation**: Beyond numerical metrics, the work includes pathway enrichment analysis and other biological validations.

## Limitations & Future Work
- Multiple reference slices are required as training data, which may be limiting for tissue types with extremely scarce samples.
- The Top-K connections in graph construction depend on the quality of UNI embeddings.
- Fine-grained incorporation of spatial positional information during heterogeneous graph construction has not been explored.

## Related Work & Insights
- The contrastive alignment ideas in BLEEP and mclSTExp are instructive, but SpaHGC's heterogeneous graph framework is more flexible.
- The exemplar retrieval concept in EGGN bears similarity to SpaHGC's CS graph, but SpaHGC adopts a more systematic graph-structured approach.
- The complementary masking strategy is generalizable to other multimodal graph learning tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of modeling cross-slice relationships via heterogeneous graphs is valuable, though the core components (GraphSAGE, attention, contrastive learning) are combinations of established techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Seven datasets, nine baselines, and biological downstream analysis—highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with rich figures and tables.
- Value: ⭐⭐⭐⭐ Makes a significant contribution to the spatial transcriptomics field; cross-slice knowledge transfer is a promising direction.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] HiFusion: Hierarchical Intra-Spot Alignment and Regional Context Fusion for Spatial Gene Expression Prediction from Histopathology](../../AAAI2026/medical_imaging/hifusion_hierarchical_intra-spot_alignment_and_regional_context_fusion_for_spati.md)
- [\[NeurIPS 2025\] Learning Relative Gene Expression Trends from Pathology Images in Spatial Transcriptomics](../../NeurIPS2025/medical_imaging/learning_relative_gene_expression_trends_from_pathology_images_in_spatial_transc.md)
- [\[CVPR 2026\] Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization](human_knowledge_integrated_multi-modal_learning_for_single_source_domain_general.md)
- [\[AAAI 2026\] Dual-Path Knowledge-Augmented Contrastive Alignment Network for Spatially Resolved Transcriptomics](../../AAAI2026/medical_imaging/dual-path_knowledge-augmented_contrastive_alignment_network_for_spatially_resolv.md)
- [\[CVPR 2026\] Forecasting Epileptic Seizures from Contactless Camera via Cross-Species Transfer Learning](forecasting_epileptic_seizures_from_contactless_ca.md)

<!-- RELATED:END -->
