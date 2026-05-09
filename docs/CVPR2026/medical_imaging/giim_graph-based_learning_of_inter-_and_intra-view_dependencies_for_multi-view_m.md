---
title: >-
  [Paper Note] GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis
description: >-
  [CVPR 2026][Medical Imaging][multi-view medical imaging] This paper proposes the GIIM framework, which constructs a Multi-Heterogeneous Graph (MHG) to simultaneously model intra-view and inter-view dependencies among lesions in multi-view medical images, and achieves robust diagnosis on incomplete data through four missing-view representation strategies.
tags:
  - CVPR 2026
  - Medical Imaging
  - multi-view medical imaging
  - graph neural networks
  - heterogeneous graph
  - missing views
  - multimodal diagnosis
date: 2026-05-08
content_hash: 5d016ba17d845c6b
---

# GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis

**Conference**: CVPR 2026
**arXiv**: [2603.09446](https://arxiv.org/abs/2603.09446)
**Code**: N/A
**Area**: Medical Imaging
**Keywords**: multi-view medical imaging, graph neural networks, heterogeneous graph, missing views, multimodal diagnosis

## TL;DR

This paper proposes the GIIM framework, which constructs a Multi-Heterogeneous Graph (MHG) to simultaneously model intra-view and inter-view dependencies among lesions in multi-view medical images, and achieves robust diagnosis on incomplete data through four missing-view representation strategies.

## Background & Motivation

Multi-view medical imaging (multi-phase CT, multi-view mammography, multi-sequence MRI) provides radiologists with complementary information and is critical for clinical diagnosis. However, existing multi-view CADx methods suffer from two fundamental limitations: (1) they neglect relationships among multiple lesions within a single view (e.g., tumor size, location, and spatial relationships are crucial for cancer diagnosis); and (2) they fail to model the dynamic changes of lesions across different views. Furthermore, missing views are common in clinical practice due to protocol differences, technical errors, or patient refusal, further reducing prediction reliability. Existing CNN-, Transformer-, and simple GNN-based methods typically analyze each lesion independently, overlooking important diagnostic cues.

## Method

### Overall Architecture

A two-stage pipeline: Stage 1 trains a ConvNeXt feature extractor for each view independently; Stage 2 constructs a Multi-Heterogeneous Graph (MHG) and performs classification by learning complex inter-lesion relationships through heterogeneous message passing.

### Key Designs

1. **Node Representations**: Two types of nodes — single-view nodes $N_{single_v}$ representing view-specific features, and multi-view nodes $M_{multi} = \|_{v=1}^{V}(N_{single_v})$ as the concatenation of all view features. This design separately captures local view-specific information and global aggregated information.

2. **Four Types of Edge Relations**: (a) **Intra-tumor Inter-view** $E_{intra}$: connects different views of the same lesion to capture temporal changes; (b) **Single-to-Multi** $E_{s-m}$: connects single-view nodes to the corresponding multi-view aggregation node to fuse fine-grained and global information; (c) **Inter-tumor Single-view** $E_{inter-s}$: connects different lesions within the same view to model co-occurrence relationships; (d) **Inter-tumor Multi-view** $E_{inter-m}$: connects aggregation nodes of different lesions to provide high-level contextual information (certain tumor types frequently co-occur). This design comprehensively covers diagnostic cues across lesions and views.

3. **Heterogeneous Message Passing**: For each node, features are aggregated separately from single-view neighbors and multi-view neighbors using independent trainable weight matrices $\mathbf{W}_{single}^k$ and $\mathbf{W}_{multi}^k$. The node representation is then updated by concatenating the original features with both aggregated features: $h_n^k \leftarrow \sigma(\mathbf{W}^k \cdot \text{CONCAT}(h_n^{k-1}, h_{N_{single}(n)}^k, h_{M_{multi}(n)}^k))$. Five SAGEConv layers (512→256→128→64→32) are used, with the final layer directly outputting classification probabilities.

### Missing-View Handling Strategies

Missing views are a common clinical challenge (incomplete protocols, technical failures, etc.). The paper proposes four strategies to handle missing view features $\mathcal{F}_v^m$:
- **Constant**: zero vector $[0.0]^{1 \times c}$
- **Learnable**: a trainable parameter tensor normalized by Frobenius norm
- **RAG-based**: retrieves the corresponding view features from the most similar sample in the database using cosine similarity
- **Covariance-based**: constructs a covariance matrix $\mathbf{\Sigma} = \frac{1}{n-1}\sum_{i=1}^n (\Delta_i - \mu)(\Delta_i - \mu)^T$ from available features and retrieves the missing features from the most similar sample via covariance similarity

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (GIIM) | NN-based | ML-based | Attention-based |
|--------|------|-----------|----------|----------|-----------------|
| Liver Tumor (4-class) | Acc / AUC | **78.20 / 91.05** | 75.45 / 89.09 | 73.63 / 88.00 | 73.41 / 88.53 |
| VinDr-Mammo (BI-RADS) | Acc / AUC | **71.17 / 82.54** | 67.48 / 82.21 | 66.87 / 80.86 | 68.09 / 81.00 |
| BreastDM (MRI) | Acc / AUC | **87.23 / 89.02** | 80.85 / 87.35 | 82.98 / 79.41 | 85.1 / 76.37 |

### Ablation Study

| Configuration | Liver (100% missing) | Liver (complete views) | Note |
|------|-------------------|-----------------|------|
| GIIM (constant) | 72.27 | 78.20 | Simple yet effective missing-view baseline |
| GIIM (learnable) | 73.64 | 78.20 | Slightly better in missing-view setting |
| GIIM (RAG-based) | 74.31 | 78.20 | Best in missing-view setting |
| GIIM (Covariance) | 70.91 | 78.20 | Strong performance after training with complete views |
| NN-based | 70.00 | 75.45 | Reference |

### Key Findings

- Multi-view methods consistently outperform single-view methods by approximately 5–12% in accuracy and 4–8% in AUC across all three datasets
- GIIM achieves the best performance among all multi-view methods, surpassing the second-best by ~3% in accuracy and ~2% in AUC on the liver dataset
- The performance gain of GIIM is most pronounced on BreastDM (Acc: 87.23 vs. 85.1; AUC: 89.02 vs. 87.35)
- An interesting trade-off exists in the missing-view setting: RAG-based performs best on the missing test set, while Covariance-based excels on the complete test set
- The Constant strategy proves surprisingly effective in the missing-view setting, as it renders missing nodes "transparent" to the graph, encouraging the model to rely more on available nodes

## Highlights & Insights

- Multi-view medical diagnosis is reformulated as a relational modeling problem; the four-type edge design comprehensively covers clinically relevant diagnostic logic
- GNNs are naturally suited for variable numbers of lesions and flexible relational modeling, offering advantages over CNN/Transformer architectures that require fixed-size inputs
- The comparative analysis of four missing-view strategies reveals a counterintuitive finding: simpler methods can be more robust in the missing-view scenario
- Extensive validation across modalities (CT, mammography, MRI) demonstrates the generalizability of the proposed framework

## Limitations & Future Work

- The ablation study is primarily qualitative, lacking quantitative ablation of individual components (e.g., each edge type)
- No direct quantitative comparison with Transformer-based multi-view methods (e.g., Phase Attention) is provided
- The RAG-based missing-view strategy depends on the quality and scale of the retrieval database, which may degrade on small datasets
- Adaptive edge weight learning or attention mechanisms are not explored; the current SAGEConv uses uniform aggregation weights
- The ConvNeXt feature extractor is frozen after Stage 1 training, potentially limiting end-to-end optimization
- On VinDr-Mammo, the AUC improvement is marginal (82.21→82.54, +0.33%), indicating limited advantage on certain datasets

## Related Work & Insights

- The Phase Attention Model (Wang et al., 2022) is the primary comparison method; this paper replaces attention-based view relation modeling with a graph-structured approach
- The inductive learning capability of SAGEConv (Hamilton et al., 2017) enables the model to generalize across varying numbers of lesions
- The missing-data handling strategies relate to the incomplete multimodal learning in mmFormer (Zhang et al., 2022)
- The LightGBM-based multi-view classifier from Nguyen et al. (2022) serves as the ML-based baseline
- Heterogeneous Graph Attention Networks (HAN, Wang et al., 2019) provide the theoretical foundation for heterogeneous message passing

## Rating

- **Novelty**: ⭐⭐⭐⭐ The heterogeneous graph architecture with four-type edge design is novel, reformulating diagnosis as relational modeling
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three datasets spanning different modalities with both complete and missing-view settings provide comprehensive evaluation
- **Writing Quality**: ⭐⭐⭐ The method description is thorough but verbose, with some redundant notation definitions
- **Value**: ⭐⭐⭐⭐ Practically significant for clinical multi-view diagnosis; the missing-view handling strategies address a real-world challenge

## Additional Remarks

The primary practical contributions of this work are: (1) the graph structure naturally accommodates the real-world clinical scenario of multiple lesions per patient without requiring fixed input sizes; (2) the missing-view handling strategies directly address the practical problem of incomplete CT/MRI protocols; and (3) cross-modality validation (CT/mammography/MRI) demonstrates the generalizability of the framework. However, the two-stage training procedure (freezing the backbone before training the graph network) may limit end-to-end optimization potential, and a joint training scheme warrants future exploration. Furthermore, the effectiveness of the Constant strategy in the missing-view setting is a phenomenon that merits further investigation.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] RelativeFlow: Taming Medical Image Denoising Learning with Noisy Reference](relativeflow_taming_medical_image_denoising_learning_with_noisy_reference.md)
- [\[CVPR 2026\] MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding](medgrpo_multi-task_reinforcement_learning_for_heterogeneous_medical_video_unders.md)
- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)
- [\[CVPR 2026\] Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision](learning_generalizable_3d_medical_image_representations_from_mask-guided_self-su.md)
- [\[CVPR 2026\] Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference](cross-slice_knowledge_transfer_via_masked_multi-modal_heterogeneous_graph_contra.md)

<!-- RELATED:END -->
