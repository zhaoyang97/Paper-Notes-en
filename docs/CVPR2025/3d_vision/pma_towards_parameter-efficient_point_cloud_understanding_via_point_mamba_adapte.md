---
title: >-
  [Paper Note] PMA: Towards Parameter-Efficient Point Cloud Understanding via Point Mamba Adapter
description: >-
  [CVPR 2025][3D Vision][Point Cloud Understanding] This paper proposes the Point Mamba Adapter (PMA), which structures and fuses complementary features from all intermediate layers of a pre-trained point cloud model into an ordered sequence using the Mamba architecture. Combined with a Geometrically-Constrained Gated Prompt Generator (G2PG) to dynamically optimize sequence ordering in 3D space, PMA achieves or surpasses full fine-tuning performance while updating only 1% of th…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Point Cloud Understanding"
  - "Parameter-Efficient Fine-Tuning"
  - "Mamba"
  - "Intermediate Feature Fusion"
  - "State Space Models"
date: 2026-05-08
content_hash: 9ce61761bb504aab
---

# PMA: Towards Parameter-Efficient Point Cloud Understanding via Point Mamba Adapter

**Conference**: CVPR 2025  
**arXiv**: [2505.20941](https://arxiv.org/abs/2505.20941)  
**Code**: [https://github.com/zyh16143998882/PMA](https://github.com/zyh16143998882/PMA)  
**Area**: 3D Vision  
**Keywords**: Point Cloud Understanding, Parameter-Efficient Fine-Tuning, Mamba, Intermediate Feature Fusion, State Space Models

## TL;DR

This paper proposes the Point Mamba Adapter (PMA), which structures and fuses complementary features from all intermediate layers of a pre-trained point cloud model into an ordered sequence using the Mamba architecture. Combined with a Geometrically-Constrained Gated Prompt Generator (G2PG) to dynamically optimize sequence ordering in 3D space, PMA achieves or surpasses full fine-tuning performance while updating only 1% of the parameters.

## Background & Motivation

**Background**: The field of point cloud understanding has transitioned from end-to-end supervised learning to a "self-supervised pre-training + downstream fine-tuning" paradigm, with pre-trained models like Point-MAE, Point-BERT, and PointGPT-L becoming mainstream. Fine-tuning strategies are categorized into Full Fine-Tuning (FFT) and Parameter-Efficient Fine-Tuning (PEFT).

**Limitations of Prior Work**: Existing PEFT methods (such as IDPT, DAPT, and PointGST) introduce a small number of learnable parameters in each layer of the pre-trained model for adaptation but ultimately feed only the output features of the last layer into the downstream task head, **completely discarding the rich representation in intermediate layers**. This is especially detrimental for tasks like segmentation that require fine-grained, point-level understanding.

**Key Challenge**: The authors experimentally observed that the intermediate layer features of pre-trained models contain semantic information almost equivalent to the final layer (e.g., using only the first 3 layers exhibits only a 2.6% drop in classification accuracy). However, existing PEFT methods completely ignore this complementary information. The difficulty lies in: (1) how to efficiently fuse features across all layers, and (2) how to construct ordered sequences given the isotropic nature of 3D space.

**Goal**: To design a PEFT method orthogonal to the pre-trained backbone that can efficiently fuse all intermediate layer features while addressing the sequence ordering challenge in 3D point clouds.

**Key Insight**: Intermediate layer features exhibit "temporal orderliness" (monotonically increasing with layer depth), and the total number of tokens ($L \times M$) far exceeds a single layer. Standard Attention is infeasible due to its quadratic complexity, whereas Mamba's linear complexity and sequence modeling capabilities are a perfect fit.

**Core Idea**: Utilize Mamba as an adapter to concatenate tokens from all layers of the pre-trained model into an ordered sequence for comprehensive feature fusion, and dynamically optimize sequence ordering via geometric constraints.

## Method

### Overall Architecture

The pipeline of PMA is as follows: The input point cloud is divided into point patches via FPS+KNN. PointNet is utilized to extract embeddings and positional encodings, which are combined with CLS tokens and fed into the frozen L-layer Transformer backbone. The output token features from each layer are passed to a shared G2PG module to generate geometric prompts and sorting indices. Subsequently, tokens from all layers are concatenated chronologically into a long sequence and fed into the Mamba Adapter for thorough fusion. Finally, the fused features from the first N-1 layers, the last layer's features, and the CLS token are concatenated and sent to the task head. During training, only the CLS token, G2PG, Mamba Adapter, and task head are updated.

### Key Designs

1. **Mamba Adapter (Orthogonal Adapter)**:

    - **Function**: Fuses intermediate layer features of the pre-trained model into a unified representation.
    - **Mechanism**: Concatenates $L$ layers $\times$ $M$ tokens into a single sequence of length $L \times M$, utilizing Mamba's State Space Model (SSM) for sequence modeling. Mamba's linear complexity $O(L \times M)$ makes handling such long sequences viable. The output matrix $C$ is augmented as $C + P$, where $P$ represents the geometric prompts.
    - **Design Motivation**: Compared to the $O((L \times M)^2)$ complexity of standard Attention, Mamba's linear complexity makes full-layer fusion feasible. Moreover, Mamba's sequential dependency modeling is inherently suited for progressive semantic information across layers.

2. **Geometrically-Constrained Gated Prompt Generator (G2PG)**:

    - **Function**: Generates geometric prompts for the output gates of Mamba and produces token sorting indices.
    - **Mechanism**: For the output tokens of each layer, a connected graph is constructed based on center coordinates via KNN. Neighborhood features are aggregated using Down Linear + Max Pooling to strengthen geometric constraints, which are then mapped through Up Linear to match the dimensionality of the Mamba output matrix $C$ (e.g., $S=128$). Softmax is applied to obtain the probability distribution $T_i^D$. $T_i^D$ serves two purposes: (1) mapping each token to a unique index via One-hot + Argmax to achieve geometry-aware semantic sorting; and (2) generating geometric prompts $P_i$ to inject into the Mamba output matrix.
    - **Design Motivation**: 3D space is isotropic and lacks a natural orientation, making simple position-based sorting (as in NLP) impossible. G2PG leverages the spatial neighborhood constraints of point clouds to learn sorting and prompts, enabling Mamba to tune its outputs based on spatial structure rather than merely preceding inputs.

3. **Feature Aggregation and Task Head**:

    - **Function**: Concatenates the fused multi-layer features from Mamba with the final layer features and feeds them into downstream task heads.
    - **Mechanism**: The final prediction is $y = f([C_N; F_{last}; F_{pre}])$, where $F_{pre}$ represents the fused features of the first N-1 layers, $F_{last}$ represents the last layer's features, and $C_N$ is the final CLS token. This design preserves both full-layer fused information and the final layer's global semantics.
    - **Design Motivation**: It leverages the full-layer fusion capabilities of Mamba while keeping the high-quality representations of the final layer itself, creating a complementary effect.

### Loss & Training

- Standard cross-entropy loss is employed for classification, and standard segmentation loss is used for segmentation.
- During training, all parameters of the Transformer backbone are frozen, and only the CLS token, G2PG, Mamba Adapter, and downstream task heads are updated.
- Compared to the 360.5M parameters of PointGPT-L, PMA only requires 4.9M trainable parameters (a 99% reduction).

## Key Experimental Results

### Main Results

| Dataset | Metric | PMA (Ours) | PointGPT-L (FFT) | PointGST | Gain |
|--------|------|-----------|------------------|----------|------|
| ScanObjectNN OBJ-BG | OA(%) | 98.97 | 97.2 | 98.97 | +1.77 vs FFT |
| ScanObjectNN PB-T50-RS | OA(%) | 95.18 | 93.4 | 94.83 | +1.78 vs FFT |
| ModelNet40 (w/ Vote) | OA(%) | 95.4 | 94.9 | 95.3 | +0.5 vs FFT |
| ShapeNetPart | mIoU_C | 84.52 | - | 83.87 | +0.65 vs PointGST |

### Ablation Study

| Configuration | ScanObjectNN PB-T50-RS | Description |
|------|----------------------|------|
| Point-MAE + FFT | 85.18 | Full fine-tuning baseline |
| Point-MAE + IDPT | 84.94 | PEFT baseline comparison |
| Point-MAE + DAPT | 85.08 | PEFT baseline comparison |
| Point-MAE + PointGST | 85.29 | PEFT baseline comparison |
| Point-MAE + PMA | **86.43** | Our method, +1.25 vs FFT |

### Key Findings

- The value of intermediate layer features is heavily underestimated: using only the first 3 layers results in a classification accuracy just 2.6% lower than all 12 layers, indicating that intermediate layers carry substantial complementary information.
- The geometric sorting of G2PG is crucial to Mamba's efficacy: it resolves the challenge of building sequential representations stemming from 3D isotropy.
- The improvement of PMA is particularly prominent in segmentation tasks (e.g., mIoU_I increases from 85.7 to 86.1 on Point-BERT), validating the importance of intermediate feature fusion for fine-grained understanding.
- Surpassing full fine-tuning with only 1% of the parameters significantly reduces deployment costs.

## Highlights & Insights

- **A new paradigm of using Mamba as a feature fuser**: Rather than utilizing Mamba for standard sequence modeling, it is used as a tool for cross-layer feature fusion. This novel perspective elevates PEFT from "inserting small modules into each layer" to "using a global module to fuse all layers".
- **Dual-function design of G2PG**: A single module simultaneously addresses both sorting and prompting while sharing parameters across layers, making it highly efficient.
- **Intermediate feature experiments**: The experiments in Figure 1 (gradually increasing the number of active layers) provide direct evidence for the value of intermediate layer features. This observation can be generalized to PEFT research in 2D vision and NLP.

## Limitations & Future Work

- Currently, validation is limited to Transformer-based point cloud pre-trained models. Its applicability to other architectures (such as PointMamba, which inherently uses SSMs) remains to be explored.
- The KNN graph construction in G2PG introduces some computational overhead, and its efficiency in ultra-large-scale point cloud scenarios needs to be evaluated.
- The sequence sorting strategy relies on Softmax + Argmax discretization, which might exhibit gradient discontinuity issues.
- The paper does not investigate weighting strategies for different layer features within the Mamba Adapter; whether certain layers are more critical warrants further exploration.

## Related Work & Insights

- **vs IDPT**: IDPT first introduced instance-aware dynamic prompts to point cloud PEFT but still only utilized the final layer's features. PMA substantially outperforms it via full-layer fusion.
- **vs PointGST**: PointGST achieves performance close to PMA on some datasets with fewer parameters (0.6M vs 4.9M), but PMA shows a clear advantage on harder datasets such as PB-T50-RS.
- **vs PointMamba/Mamba3D**: These works utilize Mamba as the backbone network, whereas PMA uses Mamba as an adapter. These two paradigms could potentially be integrated.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of intermediate layer fusion and the Mamba Adapter is refreshing, and the G2PG is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensively covers classification, segmentation, and few-shot learning, evaluated across multiple pre-trained models.
- Writing Quality: ⭐⭐⭐⭐ The motivation is clear, and the observation experiments in Figure 1 are highly convincing.
- Value: ⭐⭐⭐⭐ Offers a new avenue for 3D PEFT with a 99% parameter reduction, making it highly valuable for physical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Spectral Informed Mamba for Robust Point Cloud Processing](spectral_informed_mamba_for_robust_point_cloud_processing.md)
- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](../../ICCV2025/3d_vision/efficient_spiking_point_mamba_for_point_cloud_analysis.md)
- [\[ICCV 2025\] StruMamba3D: Exploring Structural Mamba for Self-supervised Point Cloud Representation Learning](../../ICCV2025/3d_vision/strumamba3d_exploring_structural_mamba_for_self-supervised_point_cloud_represent.md)
- [\[CVPR 2026\] Mamba Learns in Context: Structure-Aware Domain Generalization for Multi-Task Point Cloud Understanding](../../CVPR2026/3d_vision/mamba_learns_in_context_structure-aware_domain_generalization_for_multi-task_poi.md)
- [\[CVPR 2025\] PointLoRA: Low-Rank Adaptation with Token Selection for Point Cloud Learning](pointlora_low-rank_adaptation_with_token_selection_for_point_cloud_learning.md)

</div>

<!-- RELATED:END -->
