---
title: >-
  [Paper Note] SUICA: Learning Super-high Dimensional Sparse Implicit Neural Representations for Spatial Transcriptomics
description: >-
  [ICML 2025][Computational Biology][spatial transcriptomics] The authors propose SUICA, which compresses super-high dimensional sparse spatial transcriptomic data into a compact embedding space via a graph-enhanced autoencoder. It then utilizes Implicit Neural Representations (INR) to model the continuous mapping from coordinates to embeddings, achieving spatial imputation, gene imputation, and denoising across various ST platforms.
tags:
  - "ICML 2025"
  - "Computational Biology"
  - "spatial transcriptomics"
  - "implicit neural representations"
  - "graph autoencoder"
  - "gene expression"
  - "spatial imputation"
date: 2026-05-08
content_hash: efc53ce1bc66cb64
---

# SUICA: Learning Super-high Dimensional Sparse Implicit Neural Representations for Spatial Transcriptomics

---

**Conference**: ICML 2025  
**arXiv**: [2412.01124](https://arxiv.org/abs/2412.01124)  
**Code**: [https://github.com/Szym29/SUICA](https://github.com/Szym29/SUICA)  
**Area**: Computational Biology  
**Keywords**: spatial transcriptomics, implicit neural representations, graph autoencoder, gene expression, spatial imputation

## TL;DR

The authors propose SUICA, which compresses super-high dimensional sparse spatial transcriptomic data into a compact embedding space via a graph-enhanced autoencoder. It then utilizes Implicit Neural Representations (INR) to model the continuous mapping from coordinates to embeddings, achieving spatial imputation, gene imputation, and denoising across various ST platforms.

---

## Background & Motivation

**Background**: Spatial Transcriptomics (ST) is a technology capable of quantifying gene expression while preserving tissue spatial context. Current mainstream ST platforms (e.g., Stereo-seq, Visium, Slide-seqV2) collect mRNA transcript counts at various spatial locations, generating super-high dimensional expression matrices (typically with $>20,000$ gene channels). Deep learning methods such as SpaGCN, STAGATE, and GraphST have been utilized to enhance spatial resolution and perform denoising.

**Limitations of Prior Work**: ST data faces a threefold challenge. First, limited spatial sampling density (high-resolution ST is very expensive, around $\$3,500/\text{cm}^2$) leads to insufficient spatial resolution. Second, the low mRNA capture rate coupled with distinct expression patterns across different cell states results in a highly zero-inflated distribution (with zeros accounting for up to 90% of the data), i.e., the dropout problem. Third, different ST platforms vary significantly in spatial regularity, sequencing depth, and dropout rates, making it difficult to model them within a unified framework.

**Key Challenge**: Implicit Neural Representations (INR) possess excellent properties such as continuous modeling and inherent smoothness, making them naturally suited for spatial interpolation of discretely sampled points. However, applying INR to ST data faces two fundamental challenges: (1) existing INR applications are low-dimensional to low-dimensional mappings (e.g., $\mathbb{R}^2 \to \mathbb{R}^3$), whereas ST requires mapping from $\mathbb{R}^2$ to a super-high dimensional space ($>20,000$ channels), where simply widening and deepening MLPs cannot overcome the curse of dimensionality; (2) the output of INR tends to be smoothly distributed like a normal distribution, whereas ST data is highly zero-inflated and extremely sparse, making the standard regression paradigm unable to preserve sparsity.

**Goal**: To design an INR variant tailored specifically for the characteristics of ST data, enabling spatial imputation, gene imputation, and denoising within a continuous compact representation, while preserving both the sparsity and numerical fidelity of super-high dimensional outputs.

**Key Insight**: Instead of forcing the INR to directly map to the super-high dimensional raw space, a Graph Autoencoder (GAE) is first employed to compress the super-high dimensional sparse data into a low-dimensional dense embedding. The INR is then tasked only with learning the mapping from coordinates to this low-dimensional embedding, shifting the burden of the "curse of dimensionality" to graph neural networks, which are better suited for this task.

**Core Idea**: Bridge the gap between INR and super-high dimensional ST data using a Graph Autoencoder, enabling the INR to operate in a compact embedding space, and reconstruct the raw space while enforcing sparsity through a decoder head trained with Dice Loss.

## Method

### Overall Architecture

The pipeline of SUICA consists of three stages: (1) Pre-training the Graph Autoencoder (GAE): based on a GCN encoder and an MLP decoder, trained in an auto-associative manner on ST slices to obtain the low-dimensional embedding $z_{gt}$ for all sampled points; (2) INR Training: optimizing the neural mapping from spatial coordinates to the embeddings $z_{gt}$; (3) Decoder Head Fine-tuning: freezing the INR, attaching the pre-trained GAE decoder to the output of the INR, and fine-tuning the decoder to map the embeddings back to the raw gene expressions using a joint loss function that includes Dice Loss to preserve sparsity.

### Key Designs

1. **Graph-Enhanced Autoencoder (GAE)**:

    - **Function**: Compresses super-high dimensional sparse ST data into low-dimensional dense embeddings while preserving spatial context.
    - **Mechanism**: The encoder employs a GCN to capture the contextual information of neighboring spots based on a $k$-NN graph ($k=5$), rendering the embedding structure-aware. The decoder is a standard MLP (as newly interpolated points lack an available graph structure). The training loss is standard MSE: $\mathcal{L}_{gae} = \frac{1}{|M_y|}\sum_{M_y}(\hat{y} - y_{gt})^2$.
    - **Design Motivation**: Spectral analysis experiments show that embeddings generated by GAE express higher Graph Total Variation and larger inter-channel variance compared to a standard AE, indicating greater structural discriminative power and informational content. These decoupled embeddings are better suited for INR modeling.

2. **Embedding Mapping (INR)**:

    - **Function**: Learns the continuous mapping from spatial coordinates $x$ to compact embeddings $z$.
    - **Mechanism**: different INR architectures are selected based on the spatial density of the ST data—using SIREN (periodic activation functions) for spatially sparse data and FFN (Fourier Feature Networks) for spatially dense data. Training utilizes MSE loss: $\mathcal{L}_{embd} = \frac{1}{|M_z|}\sum_{M_z}(\hat{z} - z_{gt})^2$. Operating in the embedding space avoids the curse of dimensionality (reducing from $>20\text{K}$ dims to a low-dimensional state), greatly simplifying the fitting task for the INR.
    - **Design Motivation**: The inherent smoothness of INR naturally enables interpolation capabilities. When operating in a low-dimensional embedding space, this smoothness prior is not disrupted by the super-high dimensional sparsity.

3. **Decoder Head with Dice Loss**:

    - **Function**: Decodes the INR-generated embeddings back into the super-high dimensional raw gene expression space while preserving sparsity.
    - **Mechanism**: The INR is first warmed up with $\mathcal{L}_{embd}$ to reach stability, then frozen while the pre-trained decoder is attached and fine-tuned independently. The loss function consists of three terms: $\mathcal{L}_{recons} = \frac{1}{|M_y^+|}\sum_{M_y^+}(\hat{y}-y_{gt})^2 + \frac{1}{|M_y|}\sum_{M_y}|\hat{y}-y_{gt}| + \lambda\mathcal{L}_{dice}$. Among them, Dice Loss transforms the regression into a quasi-classification problem, using $\tanh$ to map the output to a pseudo-probability space $[0,1)$ and utilizing intersection-over-union (IoU) to optimize the overlap between predicted non-zero patterns and ground truth non-zero patterns.
    - **Design Motivation**: (1) Stage-wise training prevents the domain shift of INR mapping errors from directly harming decoding performance, and also prevents the local optima of the pre-trained decoder from hindering INR optimization; (2) calculating MSE only on non-zero values ($M_y^+$) avoids the issue where all-zero predictions receive low loss; (3) Dice Loss is sensitive to class imbalance, successfully forcing the model to retain the inherent sparse patterns of ST data.

### Loss & Training

Three-stage sequential training: GAE pre-training (Adam, $lr=10^{-5}$, 200 epochs) $\to$ INR training (Adam, $lr=10^{-4}$, 1K epochs) $\to$ decoder head fine-tuning (same $lr$, 1K epochs). All experiments were completed on a single RTX 4090 GPU.

## Key Experimental Results

### Main Results — Spatial Imputation

| Method | MAE↓ | MSE↓ | Cosine↑ | Pearson↑ | Spearman↑ | ARI↑ |
|------|------|------|---------|----------|-----------|------|
| FFN | 6.51 | 1.20 | 0.706 | 0.718 | 0.400 | 0.143 |
| SIREN | 7.21 | 1.31 | 0.661 | 0.678 | 0.247 | 0.289 |
| STAGE (SOTA) | 6.52 | 1.11 | 0.732 | 0.747 | 0.365 | 0.139 |
| **SUICA** | **5.66** | **0.85** | **0.797** | **0.792** | **0.447** | **0.343** |

*\*Stereo-seq MOSTA dataset, MAE/MSE $\times10^{-2}$. Reference ARI = 0.312*

### Ablation Study

| Configuration | MSE↓ (E16.5) | Cosine↑ (E16.5) | MSE↓ (Brain) | Cosine↑ (Brain) |
|------|-------------|-----------------|-------------|-----------------|
| Vanilla INR | 2.35 | 0.668 | 9.33 | 0.756 |
| +AE | 1.60 | 0.789 | 11.27 | 0.695 |
| +Dice | 1.48 | 0.806 | 7.05 | 0.826 |
| +Graph (SUICA) | **1.47** | **0.807** | **5.67** | **0.860** |

### Key Findings

- The ARI of SUICA even surpasses the reference value calculated with the ground truth by 3.9%, demonstrating that the smooth prior of INR actually enhances biological signals.
- SUICA successfully predicts the spatial expression patterns of specific genes (e.g., localized expression of AFP in the liver, recovery of low expression signals of SEPT3 in brain regions).
- The performance of GAE vs. AE varies by ST platform: graph-enhanced features contribute the most in spatially sparse data (Human Brain, Visium), whereas AE+Dice contributes more in spatially dense data (MOSTA).
- SUICA consistently outperforms STAGE in gene imputation and denoising tasks, with a particularly pronounced advantage in denoising scenarios (Cosine 0.733 vs. 0.606).

## Highlights & Insights

- It extends INR from traditional low-dimensional to low-dimensional mappings to a low-dimensional to super-high-dimensional ($>20\text{K}$ channels) scenario, elegantly resolving the curse of dimensionality via an intermediate embedding space. This represents a significant expansion of INR applicability.
- The "treating regression as classification" philosophy of Dice Loss is highly suited for zero-inflated data—this strategy could be generalized to other sparse data modeling tasks, such as point clouds and event camera data.
- SUICA is degradation-agnostic; the same framework can simultaneously handle spatial missingness, gene dropout, and noise without requiring any prior knowledge.
- The discovery that biological fidelity even surpasses that of the raw data (ARI exceeding ground truth by 3.9%) indicates that the smooth prior of INR is essentially performing implicit denoising, thereby enhancing real biological signals.
- Spectral analysis of the graph autoencoder (Figure 3) intuitively demonstrates why GAE outperforms a standard AE—a higher GTV indicates that the embeddings retain richer spatial structural information.
- The paper validates the approach across multiple ST platforms (Stereo-seq, Visium, Slide-seqV2, MERFISH), fully demonstrating the generalizability of the proposed method.

## Limitations & Future Work

- The three-stage sequential training pipeline is somewhat tedious; end-to-end joint training might be superior, though the paper explains the necessity of stage-wise training (due to domain shift and local optima issues).
- The $k$-NN graph structure used by the GAE encoder may be unstable when spot densities vary dramatically, making adaptive graph construction worth exploring.
- The current evaluation is primarily focused on mouse tissues, and generalizability to human tissues has not been fully verified, where gene expression distributions may differ.
- The training speed of INR is limited, and its scalability to larger-scale ST datasets (e.g., million-level spots in whole-brain MERFISH data) remains undiscussed.
- The approximation error bound for INR mapping from low-dimensional to super-high-dimensional spaces is not theoretically analyzed, leaving a lack of convergence guarantees.
- The choice of embedding dimensionality significantly affects performance, but the paper does not sufficiently discuss how to automatically determine the optimal dimension.
- Comparisons with super-resolution methods assisted by reference histology images (such as Hist2ST) are missing.

## Related Work & Insights

- STAGE (Li et al., 2024) is the current SOTA coordinate-based ST enhancement method, utilizing a position-supervised autoencoder. However, it fails to leverage graph structures and sparsity priors, resulting in limited performance on spatially sparse data.
- SIREN and FFN are classic INR architectures. SUICA flexibly selects either as its backbone and demonstrates general suitability.
- NeRF-like works also employ INR decoding in an embedding space, but their mapping target dimension is far lower than the $20\text{K}+$ of ST. SUICA for the first time pushes INR to super-high-dimensional outputs.
- Graph-based methods like SpaGCN, STAGATE, and GraphST focus on spatial domain clustering, whereas SUICA concentrates on continuous signal reconstruction, rendering them complementary.
- The GAE+INR paradigm in this study can be generalized to other spatial omics data types (e.g., spatial proteomics, spatial metabolomics) simply by updating the input dimensions of the GAE.
- Dice Loss is widely used in medical image segmentation; this work novelly introduces it to regression tasks to handle zero-inflated distributions.

## Rating

⭐⭐⭐⭐ The method design is highly targeted, systematically resolving the challenges of modeling ST data across three layers: GAE dimension reduction, INR continuous modeling, and Dice Loss sparsity preservation. The experiments comprehensively cover multiple platforms (Stereo-seq, Visium, Slide-seqV2) and diverse tasks (spatial imputation, gene imputation, denoising), supported by thorough ablation studies. The finding that ARI can surpass the ground truth is also interesting. However, the three-stage training is somewhat complex, comparison with histology-image-assisted methods (e.g., Hist2ST, TRIPLEX) is missing, and discussion on scalability is insufficient.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Relative Gene Expression Trends from Pathology Images in Spatial Transcriptomics](../../NeurIPS2025/computational_biology/learning_relative_gene_expression_trends_from_pathology_images_in_spatial_transc.md)
- [\[ICML 2025\] SToFM: a Multi-scale Foundation Model for Spatial Transcriptomics](stofm_a_multi-scale_foundation_model_for_spatial_transcriptomics.md)
- [\[CVPR 2026\] Predicting Spatial Transcriptomics from Histology Images via High-Order Multi-Cell Interaction Modeling](../../CVPR2026/computational_biology/predicting_spatial_transcriptomics_from_histology_images_via_high-order_multi-ce.md)
- [\[CVPR 2026\] Multi-View Hierarchical Alignment Learning for Spatial Transcriptomics](../../CVPR2026/computational_biology/multi-view_hierarchical_alignment_learning_for_spatial_transcriptomics.md)
- [\[CVPR 2026\] HyperST: Hierarchical Hyperbolic Learning for Spatial Transcriptomics Prediction](../../CVPR2026/computational_biology/hyperst_hierarchical_hyperbolic_learning_for_spatial_transcriptomics_prediction.md)

</div>

<!-- RELATED:END -->
