---
title: >-
  [Paper Note] DW-DGAT: Dynamically Weighted Dual Graph Attention Network for Neurodegenerative Disease Diagnosis
description: >-
  [AAAI 2026][Medical Imaging][Graph Attention Network] To address three core challenges in early diagnosis of neurodegenerative diseases (PD/AD)—multi-indicator data fusion, heterogeneous information extraction, and class imbalance—this paper proposes DW-DGAT, a dynamically weighted dual graph attention network. By introducing a universal data fusion strategy, micro-macro dual-level graph feature learning, and a dynamic class weight generation mechanism…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Graph Attention Network"
  - "Neurodegenerative Disease"
  - "Multimodal Fusion"
  - "Class Imbalance"
  - "Parkinson's/Alzheimer's"
date: 2026-05-08
content_hash: 27fed791f82c184b
---

# DW-DGAT: Dynamically Weighted Dual Graph Attention Network for Neurodegenerative Disease Diagnosis

**Conference**: AAAI 2026  
**arXiv**: [2601.10001](https://arxiv.org/abs/2601.10001)  
**Code**: [github.com/AlexanderLeung9/DW-DGAT](https://github.com/AlexanderLeung9/DW-DGAT)  
**Area**: Neurodegenerative Disease Diagnosis / Medical Imaging  
**Keywords**: Graph Attention Network, Neurodegenerative Disease, Multimodal Fusion, Class Imbalance, Parkinson's/Alzheimer's

## TL;DR

To address three core challenges in early diagnosis of neurodegenerative diseases (PD/AD)—multi-indicator data fusion, heterogeneous information extraction, and class imbalance—this paper proposes DW-DGAT, a dynamically weighted dual graph attention network. By introducing a universal data fusion strategy, micro-macro dual-level graph feature learning, and a dynamic class weight generation mechanism, DW-DGAT substantially outperforms 14 baseline methods on the PPMI and ADNI3 datasets.

## Background & Motivation

Parkinson's disease (PD) and Alzheimer's disease (AD) are the two most prevalent incurable neurodegenerative diseases worldwide, and **early diagnosis** is critical for slowing disease progression. However, brain imaging changes in early stages are extremely subtle and difficult for clinicians to detect visually. Specifically:
- PD has a **prodromal stage (PRO)** between healthy controls (HC) and PD
- AD has an **early mild cognitive impairment (EMCI)** stage between cognitively normal (CN) and AD

### Three Core Challenges

#### Challenge 1: High-Dimensional Heterogeneity of Multi-Indicator Data

DTI-derived indicators include FA, MD, LDH-S/K, AXD, RDD, eigenvectors V1–V3, etc., existing in **three structural forms**:
- **3D indicators**: Raw voxel-level diffusion metrics
- **2D indicators**: Brain region connectivity network matrices (FA/FN/FL deterministic networks)
- **1D indicators**: Brain region statistics (surface size, voxel count)

Using all indicators directly incurs severe memory and computational overhead. Existing studies typically employ only limited subsets, wasting the complementary information across indicators.

#### Challenge 2: Heterogeneity Between Neuroimaging and Phenotypic Data

Phenotypic data (sex, age, years of education, MoCA score, etc.) are **low-dimensional yet highly informative**, and direct fusion with high-dimensional neuroimaging data compromises their diagnostic value. Feature extraction is required at both the micro (brain region) and macro (inter-subject relationship) levels.

#### Challenge 3: Class Imbalance

Medical datasets are inherently class-imbalanced (e.g., only 37 AD samples vs. 234 CN samples in ADNI3). Conventional oversampling (complex data generation) and undersampling (prone to overfitting) strategies are both suboptimal for multi-indicator medical data.

## Method

### Overall Architecture

DW-DGAT comprises four core modules (Fig. 1):
1. **DF (Data Fusion)**: Unifies 1D/2D/3D multi-indicator data into an ROI × feature matrix
2. **SGA (Single Graph Attention)**: Extracts micro-level graph features at the brain region level
3. **GGA (Global Graph Attention)**: Extracts macro-level relational features across subjects
4. **CWG (Class Weight Generator)**: Dynamically generates class weights to alleviate imbalance

### Key Designs

#### 1. **Universal Data Fusion (DF)**

All three structural forms are unified into $\mathbf{X} \in \mathbb{R}^{R \times F}$ (R = 90 ROIs, F = feature dimension):

**3D → 1D conversion** (4 values extracted per ROI per indicator):
- Centroid coordinates $(\bar{x}, \bar{y}, \bar{z})_r$: weighted centroid
- Centroid weight $w_r$: voxel value at the centroid location
- Mean weight $\bar{w}_r$: mean of all voxel values within the ROI
- Maximum weight $\hat{w}_r$: maximum value within the ROI

**2D → 1D conversion**: For each deterministic network matrix, min-max normalization to [0,1] is applied, followed by computing the L1 norm of each row to obtain an R-dimensional vector.

**1D processing**: The ratio of surface voxel count to total voxel count is computed.

All 1D/2D/3D features are finally concatenated per ROI to form a unified matrix.

**Design Motivation**: This is the first universal strategy capable of fusing multi-indicator data in all three structural forms (1D, 2D, 3D), without being restricted to specific indicators or data formats.

#### 2. **Single Graph Attention (SGA) — Micro-Level Features**

ROI graph features are extracted in two steps:

**Graph Pooling**:
- Compute pairwise Euclidean distance matrix between ROIs
- Compute centrality distance for each ROI (sum of distances to all other ROIs)
- Zero out features of the 50% ROIs with the largest centrality distances (weakly connected, low-contribution ROIs)
- Compute Gaussian kernel-based mean similarity among remaining ROIs as supplementary features

**ViT Encoder**:
- Project the pooled $\mathbf{X}_1 \in \mathbb{R}^{R \times F'}$ to $E=384$ dimensions
- Append a learnable CLS token and positional encoding
- Extract global ROI relationships through 12 MHSA blocks
- Read out the CLS token as the subject representation

#### 3. **Global Graph Attention (GGA) — Macro-Level Features**

Subject-level relationships are modeled based on phenotypic data:

**Adjacency Graph Construction**:
- Compute inter-subject distances using transformed cosine similarity $d_{i,j} = 1 - \frac{\mathbf{p}_i \cdot \mathbf{p}_j}{\|\mathbf{p}_i\|_2 \cdot \|\mathbf{p}_j\|_2}$
- Generate a similarity matrix via Gaussian kernel ($\sigma$ = median distance)
- Remove self-loops and re-normalize

**MHSA Graph Convolution Layer (Core Innovation)**:
- Replace the affine weight matrix in conventional GCN with MHSA
- For each subject $k$, first aggregate neighbor features weighted by the adjacency matrix to obtain $\mathbf{H}_k$
- Then compute $Q, K, V$ via multi-head self-attention on $\mathbf{H}_k$
- This exploits graph structure (adjacency matrix) while adaptively adjusting edge weights

Two MHSA-GC layers are stacked with progressively doubled output feature dimensions, followed by a FC layer for dimensionality reduction.

#### 4. **Class Weight Generator (CWG)**

Adversarial training analogous to GANs is adopted, with two key modifications to address GAN training instability:

**Architecture**: CWG shares a similar structure with DGAT, comprising C GGAs (one per class), each focusing on intra-class subject relationships via class-masked adjacency graphs.

**DGAT Loss $L_1$**:

$$L_2 = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} \mathbf{Y}_{i,j} \cdot \mathbf{R}_{i,j} \cdot \log \mathbf{O}_{i,j}$$

where $\mathbf{R}_{i,j}$ is the class weight after **weight inversion and normalization**, implemented via $\exp(-(\mathbf{W}_{i,j} - \min(\mathbf{w}_i)))$.

**CWG Loss $L_3$**: Extends $L_2$ with a weight entropy regularization term $-\frac{\alpha}{N \cdot C} \sum \mathbf{R}_{i,j} \log \mathbf{R}_{i,j}$ to prevent weight degeneration.

**Stability Guarantees**:
- Subtract the extreme value inside Softmax to prevent numerical overflow
- Add machine precision $\epsilon$ to prevent gradient vanishing
- Penalize only the classifier predictions on incorrect classes, leaving correct classes unaffected

### Loss & Training

- Adam optimizer, learning rate 0.001, dropout 0.5
- 500 training epochs, 10-fold cross-validation
- Batch size 64
- CWG and DGAT updated alternately (CWG updated first to generate weights, then DGAT updated using the weights)
- Strict data leakage prevention: data from different time points of the same subject are placed in the same fold

## Key Experimental Results

### Main Results

| Method Category | Method | PD ACC (%) | AD ACC (%) |
|----------------|--------|------------|------------|
| Vision Networks | ViT-small | 66.99 | 64.03 |
| Vision Networks | VGG-19-BN | 63.96 | 60.64 |
| Public GNNs | ChebNetII | 66.99 | 62.90 |
| Public GNNs | GATv2 | 65.08 | 62.07 |
| ND-Specific | LG-GNN | 66.04 | 61.64 |
| ND-Specific | RA-GCN | 61.54 | 55.07 |
| ND-Specific | BrainGNN | 64.15 | 56.90 |
| **Ours** | **DW-DGAT** | **74.56±5.99** | **68.65±4.35** |

DW-DGAT surpasses the second-best method (ViT/ChebNetII) by **7.57%** on PD diagnosis and by **4.62%** on AD diagnosis. RA-GCN collapses on the PD task due to GAN training instability (BA = 33.33% = random chance).

### Ablation Study

| Module Combination | PD ACC (%) | AD ACC (%) | Cumulative Gain (%) |
|-------------------|-----------|-----------|-------------------|
| Baseline (MLP + 3 networks) | 63.05 | 56.03 | - |
| +DF | 65.41 | 60.78 | +7.11 |
| +DF+SGA | 67.45 | 61.42 | +9.79 |
| +DF+SGA+GGA | 71.70 | 65.09 | +17.71 |
| **Full DW-DGAT** | **74.56** | **68.65** | **24.13** |

The largest contributing module is **GGA** (+7.92%), followed by **DF** (+7.11%). The success of GGA is attributed to the combination of cosine similarity-based graph construction and MHSA-GC layers.

### Key Findings

1. **GGA is the most critical module**: The adjacency graph construction method and MHSA-GC layers jointly enhance the correlation between phenotypic features and subject labels, enabling more precise message propagation.
2. **CWG training stability**: Comparing the training loss of RA-GCN (stagnant) against DW-DGAT (steadily decreasing) validates the stability advantage of the improved loss function in adversarial training.
3. **t-SNE visualization**: DW-DGAT produces the most compact intra-class clusters and identifies the greatest number of minority-class samples.
4. **ROC curves**: DW-DGAT's ROC curves dominate all other methods across all classes.
5. **Computational complexity**: Classifier 139.02 GFLOPs, generator 169.29 GFLOPs; classifier 2728 MB GPU memory, generator 3288 MB GPU memory.

## Highlights & Insights

- **Universal data fusion strategy**: For the first time, this work elegantly resolves the unified fusion of multi-indicator data in all three structural forms (1D/2D/3D), extracting centroid coordinates + centroid weight + mean + maximum (4 features per ROI) in a concise yet effective manner.
- **Micro-macro dual-level design**: SGA focuses on structural relationships among brain regions (micro-level), while GGA captures phenotypic relationships among subjects (macro-level), with the two being complementary.
- **MHSA-based GC layer replacing affine transformations**: Retains the inductive bias of graph structure (adjacency matrix) while adaptively learning edge weights, offering greater flexibility than conventional GCN and GAT.
- **Stabilized adversarial training in CWG**: Through three simple modifications—extreme value shifting, epsilon addition, and selective penalization—the training collapse observed in RA-GCN is resolved.

## Limitations & Future Work

- **Limitations of 3D data fusion**: Only 4 statistical values are extracted per ROI, without accounting for ROI size differences (larger ROIs may contain richer latent features).
- **Inductive learning constraints**: GGA requires loading all samples in the current batch to construct the adjacency graph; while superior to transductive settings, batch size still affects performance.
- **Limited data scale**: PPMI has 636 samples and ADNI3 has 464 samples; small-sample issues persist even with 10-fold cross-validation.
- **High computational cost**: GGA has time complexity $O(N^3 \cdot E^2)$, which scales rapidly with batch size.
- **Validation limited to MRI+DTI**: The framework has not been extended to other modalities such as fMRI or PET.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combined design of data fusion + dual graph attention + dynamic weighting is novel, though individual components build on existing techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 14 baseline methods, two datasets, and comprehensive ablation and visualization analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with detailed formulations, though the complexity analysis is overly verbose.
- **Value**: ⭐⭐⭐⭐ — Meaningful advancement for early ND diagnosis; the 7.57% accuracy gain is substantial.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] NutriScreener: Retrieval-Augmented Multi-Pose Graph Attention Network for Malnourishment Screening](nutriscreener_retrieval-augmented_multi-pose_graph_attention_network_for_malnour.md)
- [\[AAAI 2026\] MAPI-GNN: Multi-Activation Plane Interaction Graph Neural Network for Multimodal Medical Diagnosis](mapi-gnn_multi-activation_plane_interaction_graph_neural_network_for_multimodal_.md)
- [\[AAAI 2026\] A Disease-Aware Dual-Stage Framework for Chest X-ray Report Generation](a_disease-aware_dual-stage_framework_for_chest_x-ray_report_.md)
- [\[AAAI 2026\] CAT-Net: A Cross-Attention Tone Network for Cross-Subject EEG-EMG Fusion Tone Decoding](cat-net_a_cross-attention_tone_network_for_cross-subject_eeg-emg_fusion_tone_dec.md)
- [\[AAAI 2026\] GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis](giim_graph-based_learning_of_inter-_and_intra-view_dependencies_for_multi-view_m.md)

</div>

<!-- RELATED:END -->
