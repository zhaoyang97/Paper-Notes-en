---
title: >-
  [Paper Note] GROVER: Graph-guided Representation of Omics and Vision with Expert Regulation for Cancer Survival Prediction
description: >-
  [AAAI 2026][Medical Imaging][Spatial omics] This paper proposes GROVER, a spatial multi-omics framework that captures nonlinear spatial-feature dependencies via a KAN-GCN encoder…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Spatial omics"
  - "multimodal fusion"
  - "graph convolutional network"
  - "mixture of experts"
  - "contrastive learning"
date: 2026-05-08
content_hash: a2272b5215eaa055
---

# GROVER: Graph-guided Representation of Omics and Vision with Expert Regulation for Cancer Survival Prediction

**Conference**: AAAI 2026
**arXiv**: [2511.11730](https://arxiv.org/abs/2511.11730)  
**Code**: [GitHub](https://github.com/Xubin-s-Lab/GROVER)  
**Area**: Computational Biology / Spatial Multi-omics Integration
**Keywords**: Spatial omics, multimodal fusion, graph convolutional network, mixture of experts, contrastive learning

## TL;DR

This paper proposes GROVER, a spatial multi-omics framework that captures nonlinear spatial-feature dependencies via a KAN-GCN encoder, aligns heterogeneous modalities through spot-feature-pair contrastive learning, and dynamically routes and filters low-quality signals via a self-adaptive Mixture of Experts (MoE). GROVER achieves superior clustering performance over existing methods on four real-world spatial omics datasets.

## Background & Motivation

Spatial transcriptomics and spatial proteomics were respectively recognized by *Nature* as Methods of the Year in 2021 and 2024. These technologies extend single-cell analysis to the spatial dimension, offering unprecedented insights into tissue organization. However, spatial multi-modal omics analysis faces a core challenge: how to effectively integrate features from heterogeneous modalities (transcriptomics, proteomics, and histology images) to produce coherent low-dimensional representations for downstream tasks such as spatial domain identification and cell type annotation.

Three key limitations exist in prior work: (1) Most methods integrate only transcriptomic and proteomic data, neglecting the morphological context provided by histology images. Methods such as SpatialGlue and PRAGA handle only dual modalities. Although MISO attempts to incorporate histology images, it relies on simple outer-product interaction with limited effectiveness. (2) Existing frameworks treat all modalities uniformly across all spatial locations, ignoring substantial variation in data quality. In practice, spatial omics data are frequently affected by technical noise (e.g., dropout events in single-cell sequencing) and biological/experimental artifacts (e.g., tissue sectioning errors), leading to large signal-to-noise ratio disparities across spots and modalities. (3) A significant semantic gap exists between omics data and histology images, and the many-to-many mapping between image patches and sequenced spots makes precise cross-modal alignment extremely challenging.

The paper addresses these issues with a three-tier strategy: KAN-GCN encoders for spatially-aware modal representations, spot-level contrastive learning to bridge the cross-modal semantic gap, and a self-adaptive MoE to dynamically select reliable modal signals at each spot.

## Method

### Overall Architecture

GROVER takes three modality inputs: transcriptomics (RNA), proteomics (ADT), and histology images. For each modality, two graphs are constructed: a spatial graph $\mathcal{G}_S$ based on spatial coordinates and a modality-specific feature graph $\mathcal{G}_F^{(m)}$ based on feature similarity. After KAN-GCN encoding, attention-weighted fusion yields a unified per-modality representation. Spot-level contrastive learning then aligns cross-modal semantics, and a self-adaptive MoE performs dynamic fusion to produce a unified embedding $Z$ for downstream analyses such as clustering.

### Key Designs

1. **KAN-based Graph Convolutional Encoder (KAN-GCN)**:

    - Function: Replaces fixed linear transformations in standard GCNs to enhance the expressive capacity of graph convolution.
    - Mechanism: Standard GCN layer propagation follows $H^{(l+1)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)})$, where $W^{(l)}$ is a fixed linear transformation. GROVER replaces this with a Kolmogorov-Arnold Network (KAN), where each layer contains a matrix of trainable univariate functions $\varphi_{q,p}^{(l)}$. The transformed features are computed as $\mathcal{F}^{(l)}(H^{(l)})_{i,q} = \sum_{p=1}^{d_l}\varphi_{q,p}^{(l)}(H^{(l)}_{i,p})$, and node updates become $H^{(l+1)} = \sigma(\hat{A}\cdot\mathcal{F}^{(l)}(H^{(l)}))$.
    - Design Motivation: KAN learns nonlinear transformations in a kernel-function manner, capturing complex nonlinear patterns in spatial omics data (e.g., nonlinear relationships between gene expression and spatial location) more effectively than the linear projections of standard GCNs.

2. **Spatial-Feature Attention Fusion**:

    - Function: Adaptively fuses spatial-graph and feature-graph encodings for each modality.
    - Mechanism: For spot $i$ and modality $m$, attention scores are computed via a shared linear transformation applied to the spatial encoding $e_i^S$ and feature encoding $e_i^{F(m)}$: $e_i^{(t)} = \mathbf{q}^\top \tanh(\mathbf{W}e_i^{(t)} + \mathbf{b})$. After softmax normalization, the fused representation is $\tilde{e}_i^{(m)} = \alpha_i^{(S)}e_i^S + \alpha_i^{(F)}e_i^{F(m)}$.
    - Design Motivation: The spatial graph encodes physical proximity while the feature graph encodes functional similarity; their relative importance varies across spots, necessitating adaptive weighting.

3. **Spot-Feature-Pair Contrastive Learning**:

    - Function: Aligns semantic representations across modalities prior to fusion.
    - Mechanism: A masked bidirectional InfoNCE loss is employed. For each modality, a cosine similarity matrix $S_{i,j}^{(m)}$ is computed, and a binary mask $M^{(m)}$ is constructed to exclude highly similar negative samples (preventing biologically similar but differently labeled spots from serving as hard negatives). Bidirectional contrastive losses are computed for all three modality pairs (RNA–ADT, RNA–Image, ADT–Image): $\mathcal{L}_{contrast}^{m_1,m_2} = \frac{1}{2}(\ell_{masked}(\tilde{E}^{(m_1)}, \tilde{E}^{(m_2)}, M^{(m_1)}) + \ell_{masked}(\tilde{E}^{(m_2)}, \tilde{E}^{(m_1)}, M^{(m_2)}))$.
    - Design Motivation: The distribution and semantic gap between omics data and histology images is substantial. The masking mechanism addresses a spatial-omics-specific problem: neighboring spots are often biologically highly similar, and treating them as negatives introduces a large number of false negatives that degrade training.

4. **Self-adaptive Mixture of Experts (MoE)**:

    - Function: Dynamically adjusts the contribution of each modality at the individual spot level.
    - Mechanism: The aligned embeddings from three modalities are averaged to form the gating input $x_i = \frac{1}{3}(\hat{e}_i^{(R)} + \hat{e}_i^{(A)} + \hat{e}_i^{(I)})$. The gating network outputs normalized weights $\beta_i^{(m)}$. A key innovation is threshold filtering: when $\beta_i^{(m)} < \gamma$ (with $\gamma=0.3$), the corresponding modality weight is set to zero and the remaining weights are renormalized. Each modality is processed by a dedicated FFN expert and the results are aggregated as $z_i = \sum_m s_i^{(m)} \cdot h_i^{(m)}$. In the degenerate case where all modalities fall below the threshold, the single modality with the highest confidence is used.
    - Design Motivation: Modality quality varies substantially across spots in spatial omics data — some spots may have extremely poor RNA signal due to sequencing dropout, or distorted image patches due to sectioning artifacts. MoE enables the model to automatically down-weight unreliable modalities at such spots.

### Loss & Training

The overall training objective is a weighted sum of per-modality reconstruction losses and cross-modal contrastive losses: $\mathcal{L}_{total} = \sum_{m}\mathcal{L}_{rec}^{(m)} + \lambda\sum_{m_i \neq m_j}\mathcal{L}_{contrast}^{m_i, m_j}$, where $\lambda=2$. Reconstruction losses recover per-modality features from the fused embedding $Z$ via graph decoders that leverage the spatial adjacency structure. The model is trained on dual NVIDIA RTX A5000 GPUs and converges within 300 epochs.

## Key Experimental Results

### Main Results (Clustering Performance on Four Datasets)

| Dataset | Metric | GROVER | MISO | SpatialGlue | COSMOS |
|--------|------|--------|------|-------------|--------|
| Human Tonsil | ARI (%) | **45.2±7.8** | 41.3±6.7 | 43.3±6.7 | 19.8±6.7 |
| Human Tonsil | SC (%) | **31.6±3.9** | 7.0±1.6 | 23.8±3.2 | 20.0±0.7 |
| Human Tonsil | CHI | **2494.4±285.5** | 244.4±14.6 | 1063.6±123.6 | 937.4±99.6 |
| Human Breast Cancer | ARI (%) | **44.1±10.7** | 37.5±3.0 | 43.0±6.9 | 25.6±2.2 |
| Human Breast Cancer | SC (%) | **36.3±7.7** | 11.0±0.6 | 20.2±0.8 | 24.8±0.8 |
| Human Glioblastoma | ARI (%) | 40.8±6.6 | **43.5±6.9** | 40.1±7.6 | 32.0±6.9 |
| Human Glioblastoma | NMI (%) | **53.9±4.1** | 49.2±2.2 | 53.8±7.3 | 48.6±4.3 |
| Tonsil Add-on | ARI (%) | **46.5±5.6** | 44.6±11.9 | 45.3±7.3 | 24.6±4.3 |
| Tonsil Add-on | SC (%) | **38.2±1.2** | 8.3±0.5 | 21.4±1.1 | 18.4±2.5 |

### Ablation Study (Human Tonsil with Add-on Antibodies)

| Configuration | ARI (%) | NMI (%) | SC (%) | Note |
|------|---------|---------|--------|------|
| GROVER (full) | 46.5±5.6 | 59.0±4.8 | 38.2±1.2 | All components enabled |
| w/o MoE | 42.5±4.3 | 56.8±3.0 | 21.8±1.2 | Expert routing replaced by simple summation; ARI drops 4.0% |
| w/o $\mathcal{L}_{contrast}$ | 45.5±7.2 | 57.8±4.3 | 21.6±2.6 | Contrastive loss removed; SC drops 16.6% |
| w/o KAN-GCN | 42.7±6.7 | 55.9±5.2 | 52.6±1.1 | Standard GCN substituted; supervised metrics decrease but SC improves |

### Key Findings
- GROVER's advantage is most pronounced on SC (Silhouette Coefficient) — outperforming MISO by 24.6 percentage points on Human Tonsil — indicating substantially cleaner cluster structure in the fused embeddings.
- The dual-modality method SpatialGlue outperforms the tri-modal method MISO on several datasets, suggesting that naively adding modalities with uniform fusion can introduce noise, corroborating the necessity of adaptive fusion.
- Removing MoE leads to significant performance degradation, validating the importance of per-spot dynamic weighting.
- Replacing KAN-GCN with standard GCN improves unsupervised metrics (SC, DBI) but degrades supervised metrics (ARI, NMI), suggesting that KAN's nonlinear modeling confers advantages under label-based evaluation but may yield less geometrically smooth embedding spaces.
- $\gamma=0.3$ (close to the reciprocal of the number of experts, $1/3$) is the optimal threshold — too low fails to filter noisy modalities, while too high over-relies on a single modality and discards complementary information.

## Highlights & Insights
- Integrating KAN into GCN is a novel combination: replacing fixed linear transformations with trainable nonlinear univariate functions effectively grants each edge in message passing its own independent nonlinear transformation capacity.
- The spot-level MoE design precisely matches the characteristics of spatial omics data, where data quality varies dramatically across spatial locations, making a globally uniform weighting scheme inadequate.
- The masking mechanism in contrastive learning elegantly addresses the spatial-data-specific problem of "biologically similar false negatives."
- The framework is highly modular and can seamlessly incorporate any state-of-the-art pathology foundation model (e.g., OmiCLIP), offering strong extensibility.

## Limitations & Future Work
- Experiments are validated only on clustering tasks; performance on additional downstream tasks such as survival prediction and cell type annotation is not demonstrated (despite the title referencing cancer survival prediction, survival analysis is absent from the experimental section).
- KAN-GCN underperforms standard GCN on unsupervised metrics, hinting at potential overfitting risk or less regular embedding geometry.
- All four datasets originate from the 10x Genomics platform; cross-platform generalizability has not been evaluated.
- The current framework handles only three modalities (RNA, ADT, Image); extension to additional omics types (e.g., epigenomics, metabolomics) requires further validation.
- The MoE gating network is relatively simple (single linear layer); more sophisticated gating mechanisms may yield additional gains.

## Related Work & Insights
- SpatialGlue employs dual-attention GNN to integrate transcriptomics and proteomics, representing the current dual-modality state of the art.
- MISO introduces histology images via outer-product interaction, but its uniform fusion strategy limits performance.
- The MoE concept originates from classical mixture-of-experts models and has been widely adopted in NLP (e.g., Switch Transformer); applying it to spatial omics fusion represents a valuable cross-domain transfer.
- KAN (Kolmogorov-Arnold Network) is a recently prominent architecture; this work is among the earliest to incorporate it into GCN-based frameworks.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MUST: Modality-Specific Representation-Aware Transformer for Diffusion-Enhanced Survival Prediction with Missing Modality](../../CVPR2026/medical_imaging/must_modality-specific_representation-aware_transformer_for_diffusion-enhanced_s.md)
- [\[AAAI 2026\] Expert-Guided Prompting and Retrieval-Augmented Generation for Emergency Medical Service Question Answering](expert-guided_prompting_and_retrieval-augmented_generation_for_emergency_medical.md)
- [\[AAAI 2026\] DW-DGAT: Dynamically Weighted Dual Graph Attention Network for Neurodegenerative Disease Diagnosis](dw-dgat_dynamically_weighted_dual_graph_attention_network_for_neurodegenerative_.md)
- [\[AAAI 2026\] Small but Mighty: Dynamic Wavelet Expert-Guided Fine-Tuning of Large-Scale Models for Optical Remote Sensing Object Segmentation](small_but_mighty_dynamic_wavelet_expert-guided_fine-tuning_of_large-scale_models.md)
- [\[AAAI 2026\] Personality-guided Public-Private Domain Disentangled Hypergraph-Former Network for Multimodal Depression Detection](personality-guided_public-private_domain_disentangled_hypergraph-former_network_.md)

</div>

<!-- RELATED:END -->
