---
title: >-
  [Paper Note] GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis
description: >-
  [CVPR2025][Medical Imaging][graph neural network] This paper proposes GIIM, a multi-view medical image classification framework based on Multi-Heterogeneous Graphs (MHG). It simultaneously models intra-view and inter-view lesion dependencies, significantly outperforming existing multi-view methods across three modalities—liver CT, breast mammography, and breast MRI—while maintaining robustness to missing views.
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "graph neural network"
  - "multi-view"
  - "heterogeneous graph"
  - "missing view"
  - "medical image classification"
date: 2026-05-08
content_hash: a59a525cda4e5dae
---

# GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis

**Conference**: CVPR2025  
**arXiv**: [2603.09446](https://arxiv.org/abs/2603.09446)  
**Code**: To be confirmed  
**Area**: Medical Image  
**Keywords**: graph neural network, multi-view, heterogeneous graph, missing view, medical image classification

## TL;DR

This paper proposes GIIM, a multi-view medical image classification framework based on Multi-Heterogeneous Graphs (MHG). It simultaneously models intra-view and inter-view lesion dependencies, significantly outperforming existing multi-view methods across three modalities—liver CT, breast mammography, and breast MRI—while maintaining robustness to missing views.

## Background & Motivation

Clinical diagnosis requires a comprehensive analysis of the interrelationships among abnormalities across multi-view or multi-phase medical images. For instance, differences in enhancement patterns across phases in multi-phase CT serve as crucial clues for liver lesion classification, while the CC and MLO views in breast mammography provide complementary information. However, existing multi-view computer-aided diagnosis (CADx) methods suffer from two core limitations: (1) they analyze each lesion independently, ignoring spatial relationships among multiple lesions within the same view (intra-view) and dynamic changes of the same lesion across different views (inter-view); and (2) missing views frequently occur in clinical practice (due to protocol differences or technical issues), which severely degrades prediction reliability. Furthermore, CNNs and Transformers require fixed-size inputs, making it difficult to flexibly model a variable number of lesions and their complex relationships.

## Method

### Overall Architecture (Two-Stage)

**Stage 1: Single-View Feature Extraction**
- Independently train a ConvNeXt feature extractor $f_v$ for each view $v$.
- Freeze the pre-trained backbone and extract lesion features for each view.

**Stage 2: Multi-Heterogeneous Graph Construction and Inference**
- Construct a heterogeneous graph for each patient and perform classification via message passing.

### Graph Structure Design

**Node Types**:
- **Single-view nodes** $N_{single}$: Lesion features of each view, with one node for each of the $V$ views.
- **Multi-view nodes** $M_{multi}$: Concatenation of all single-view node features to aggregate global information.

**Four Edge Types (Core Innovations)**:
1. **Intra-tumor, Inter-view ($E_{intra}$)**: Connects different view nodes of the same lesion to capture temporal or viewpoint variations.
2. **Single-to-Multi-view ($E_{s-m}$)**: Connects single-view nodes to their corresponding multi-view aggregated nodes.
3. **Inter-tumor, Single-view ($E_{inter-s}$)**: Connects different lesions within the same view to model spatial relationships.
4. **Inter-tumor, Multi-view ($E_{inter-m}$)**: Connects multi-view aggregated nodes of different lesions to model high-level context.

### Heterogeneous Message Passing

For each node $n$ at layer $k$, information is aggregated from two types of neighbors:
- Aggregate from single-view neighbors: $h_{N_{single}(n)}^k = \frac{1}{|N_{single}(n)|} \sum_{u} \mathbf{W}_{single}^k h_u^{k-1}$
- Aggregate from multi-view neighbors: $h_{M_{multi}(n)}^k = \frac{1}{|M_{multi}(n)|} \sum_{u} \mathbf{W}_{multi}^k h_u^{k-1}$

Final update: $h_n^k = \sigma(\mathbf{W}^k \cdot \text{CONCAT}(h_n^{k-1}, h_{N_{single}(n)}^k, h_{M_{multi}(n)}^k))$

The architecture utilizes 5 layers of SAGEConv (512 $\rightarrow$ 256 $\rightarrow$ 128 $\rightarrow$ 64 $\rightarrow$ 32 $\rightarrow$ number of classes).

### Missing View Handling (Four Strategies)

1. **Constant**: Zero-vector padding, simple and effective.
2. **Learnable**: Randomly initialized learnable tensor with Frobenius normalization.
3. **RAG-based**: Retrieval-based, finding the corresponding view features of the most similar sample from the database using cosine similarity.
4. **Covariance-based**: Retrieval based on covariance similarity, leveraging the statistical relationships of feature differences between views.

## Key Experimental Results

**Multi-view Classification (Three Datasets)**:

| Dataset | Method | Acc (%) | AUC (%) |
|--------|------|---------|---------|
| Liver (877 cases) | Attention-based | 73.41 | 88.53 |
| | **GIIM** | **78.20** | **91.05** |
| VinDr-Mammo | Attention-based | 68.09 | 81.00 |
| | **GIIM** | **71.17** | **82.54** |
| BreastDM (232 cases) | Attention-based | 85.1 | 76.37 |
| | **GIIM** | **87.23** | **89.02** |

**Key Findings**:
- Multi-view methods achieve an improvement of approximately 12% in Acc and 8.3% in AUC compared to single-view methods on the Liver dataset.
- GIIM improves upon the second-best multi-view method by 3% in Acc and 2% in AUC on the Liver dataset.
- Missing view scenarios: GIIM (constant) still maintains an accuracy of 72-74% on the test set with completely missing views (Liver), significantly outperforming other methods.
- Performance gap between full views vs. missing views: approximately 4% for the liver and 5% for breast mammography.

## Highlights & Insights

- The heterogeneous graph design with four edge types comprehensively captures all key relationships in multi-view medical images.
- GNNs are naturally suited to modeling a variable number of lesions, offering greater flexibility than CNNs or Transformers.
- Validated effectively across three different modalities (CT, X-ray, MRI), demonstrating strong generalizability.
- The four missing-view handling strategies offer flexible choices for practical deployment (RAG/Covariance for full views, Constant for missing views).
- The two-stage training strategy is simple yet effective, decoupling feature extraction from relationship modeling.

## Limitations & Future Work

- The feature extractor and graph model are trained in separate stages rather than being optimized end-to-end, which may limit overall performance.
- The mapping of lesions between CC and MLO views in breast mammography experiments is simplified (using mean pooling instead of precise matching).
- The liver dataset is private, limiting reproducibility.
- Graph construction relies on precise lesion detection/annotation; the impact of detection errors on downstream classification is not discussed.
- The method was only evaluated on classification tasks and has not been extended to segmentation or detection.

## Related Work & Insights

- **Phase Attention** (Wang et al., 2022): Uses attention mechanisms to weight different view features but does not model relationships between lesions.
- **SSL-MNGCN/MMGCN** (Ibrahim et al., 2022): Uses GCNs for breast mammography but only models texture and spatial features.
- **ConvNeXt** (Liu et al., 2022): The single-view feature extraction backbone used in this study.
- **Multi-phase fusion**: Traditional methods combine multi-view images through alignment and fusion but ignore dependencies between lesions.

## Rating

- Novelty: ⭐⭐⭐⭐ (The heterogeneous graph design with four edge types is highly systematic, and the missing view handling is comprehensive)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Three modalities, missing view tests, and comparisons with multiple baselines)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure and comprehensive description of medical background)
- Value: ⭐⭐⭐⭐ (A general framework applicable to various multi-view medical diagnosis scenarios)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation](enhanced_contrastive_learning_with_multi-view_longitudinal_data_for_chest_x-ray_.md)
- [\[CVPR 2025\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)
- [\[CVPR 2025\] UltrasoundAgents: Hierarchical Multi-Agent Evidence-Chain Reasoning for Breast Ultrasound Diagnosis](ultrasoundagents_hierarchical_multi-agent_evidence-chain_reasoning_for_breast_ul.md)
- [\[CVPR 2025\] Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization](human_knowledge_integrated_multi-modal_learning_for_single_source_domain_general.md)
- [\[CVPR 2025\] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](semitooth_a_generalizable_semi-supervised_framework_for_multi-source_tooth_segme.md)

</div>

<!-- RELATED:END -->
