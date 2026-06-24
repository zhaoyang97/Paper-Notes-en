---
title: >-
  [Paper Note] Self-Disentanglement and Re-Composition for Cross-Domain Few-Shot Segmentation
description: >-
  [ICML 2025][Segmentation][Cross-Domain Few-Shot Segmentation] This paper identifies a feature entanglement issue in distance-comparison-based methods for Cross-Domain Few-Shot Segmentation (CD-FSS), which stems from the equal-weighted cross-matching of ViT layer outputs during distance computation. Consequently, the authors propose to address this issue through Self-Disentanglement and Re-Composition by learning the comparison weights among ViT components.
tags:
  - "ICML 2025"
  - "Segmentation"
  - "Cross-Domain Few-Shot Segmentation"
  - "ViT Feature Disentanglement"
  - "Orthogonal Space Disentanglement"
  - "Cross-Pattern Comparison"
  - "Adaptive Fusion Weights"
date: 2026-05-08
content_hash: 2a42141c77e4e6c9
---

# Self-Disentanglement and Re-Composition for Cross-Domain Few-Shot Segmentation

**Conference**: ICML 2025  
**arXiv**: [2506.02677](https://arxiv.org/abs/2506.02677)  
**Code**: None  
**Area**: Image Segmentation  
**Keywords**: Cross-Domain Few-Shot Segmentation, ViT Feature Disentanglement, Orthogonal Space Disentanglement, Cross-Pattern Comparison, Adaptive Fusion Weights

## TL;DR

This paper identifies a feature entanglement issue in distance-comparison-based methods for Cross-Domain Few-Shot Segmentation (CD-FSS), which stems from the equal-weighted cross-matching of ViT layer outputs during distance computation. Consequently, the authors propose to address this issue through Self-Disentanglement and Re-Composition by learning the comparison weights among ViT components.

## Background & Motivation

Cross-Domain Few-Shot Semantic Segmentation (CD-FSS) aims to transfer knowledge from a source domain dataset to an unseen target domain with only a few annotations. Existing methods typically perform mask prediction by comparing feature distances between support and query sets.

**Limitations of Prior Work**: The authors identify a **feature entanglement issue** in this widely adopted approach—models tend to bind multiple patterns from the source domain (e.g., wings and body) together, making it difficult to transfer each pattern individually. For instance, when a model entangles the "wings + body" patterns, if a target domain image contains only wings while the body differs from the training data (e.g., a different bat species), the model fails to capture the wings, leading to segmentation errors.

**Key Challenge**: In the CD-FSS scenario, a substantial domain gap and semantic gap exist between the source and target domains. Transferring entangled patterns is much more difficult than transferring disentangled ones.

**Key Insight**: Leveraging interpretability studies of ViTs, the authors notice that residual connections and consistent spatial dimensions align the outputs of each ViT component (MSA, MLP) within the **same feature space**. Thus, the final ViT output can be naturally viewed as a cumulative sum of all components. Based on this structural decomposition, the authors observe that different layers capture distinct semantic patterns. However, equal weights are assigned to all cross-layer comparisons during distance computation—meaningful comparisons (wing vs. wing) and meaningless ones (body vs. wing) are mixed equally, causing feature entanglement.

**CKA Validation**: The authors validate this hypothesis using Centered Kernel Alignment (CKA) similarity experiments. Domain similarities across different layers vary significantly: the average CKA for layer-wise matching (diagonal) is much higher than the CKA of the final output (e.g., 0.6107 vs. 0.4288 on FSS-1000). Interestingly, simple shift matching (Top-12 average 0.8126) even outperforms diagonal matching, suggesting that learnable cross-matching could outperform naive layer-to-layer matching.

**Core Idea**: Learn the weights of all comparisons among ViT components to self-disentangle the ViT output features and cross-recompose them, thereby down-weighting meaningless comparisons and up-weighting meaningful ones.

## Method

### Overall Architecture

The proposed framework, named SDRC (Self-Disentanglement and Re-Composition), operates as follows:

1. Extract $L$ sets of support/query features from different ViT layers and concatenate them along the channel dimension.
2. Feed them into the **Orthogonal Space Disentanglement (OSD)** module for weight allocation and semantic disentanglement.
3. Pass the OSD outputs to the **Cross-Pattern Comparison (CPC)** module, where disentangled patterns are cross-compared to generate $L^2$ sets of score maps.
4. During source-domain training, combine the score maps using uniform weights; during target-domain fine-tuning, introduce **Adaptive Fusion Weights (AFW)** to dynamically learn the comparison weights.
5. Obtain the final prediction via bilinear interpolation back to the original image size, followed by an argmax operation.

### Key Designs

1. **Orthogonal Space Disentanglement (OSD) Module**: Concatenates features from all layers and projects them into a low-dimensional orthogonal space to explicitly disentangle different semantic patterns and allocate weights.

   Specific process: Concatenate $L$ groups of features along the channel dimension to obtain $F_{con}^* \in \mathbb{R}^{Ld \times n \times n}$, then process them through a three-layer structure:
    - A fully connected layer $W_{in} \in \mathbb{R}^{Ld \times r}$ for dimensionality reduction to a low-rank space.
    - A convolutional layer $W_{orth} \in \mathbb{R}^{r \times r \times 1 \times 1}$ to impose orthogonal constraints.
    - A fully connected layer $W_{out} \in \mathbb{R}^{r \times Ld}$ to map back to the original space and split.

   Orthogonal regularization loss (calculated after reshaping $F_{orth}$ to $\mathbb{R}^{r \times n^2}$):
    $L_{orth} = \|F_{orth} F_{orth}^T - I\|_F^2$

   **Design Motivation**: Promote independence among features of different channels via orthogonal constraints to achieve semantic disentanglement. Mutual Information (MI) experiments validate this design—using OSD significantly reduces MI between support/query features (e.g., from 0.91 to 0.65 on Chest X-ray). The rank $r$ is set to 8 by default to balance performance and parameter count. During source training, $W_{in}$ and $W_{out}$ are co-trained with the encoder; during target fine-tuning, only the compact $W_{orth}$ (only 64 parameters) is fine-tuned, while the rest are frozen.

2. **Cross-Pattern Comparison (CPC) Module**: Cross-compares the disentangled support prototypes with query features to generate $L^2$ sets of score maps for re-composition.

   First, obtain $L$ foreground prototypes $P_{fg} \in \mathbb{R}^{L \times d \times 1 \times 1}$ and background prototypes $P_{bg} \in \mathbb{R}^{L \times d \times 1 \times 1}$ from the support features via Mask Average Pooling. Then, cross-compare the $L$ sets of query features with the $L$ sets of prototypes:
    $C_{bg/fg} = distance(F^q, P_{bg/fg}), \quad C = concat(C_{bg}, C_{fg})$

   where $C$ is reshaped to $\mathbb{R}^{L^2 \times 2 \times n \times n}$, and 2 represents background and foreground. Cosine similarity is used by default:
    $distance_{cos} = \frac{F^q \cdot P_{bg/fg}}{\|F^q\| \|P_{bg/fg}\|}$

   **Design Motivation**: Due to the dynamic nature of ViT's self-attention mechanism, features extracted from different layers may exhibit appropriate cross-layer semantic correspondences. Therefore, cross-comparison is more effective than element-wise alignment (experiment: 59.50% vs. 55.14%).

3. **Adaptive Fusion Weights (AFW)**: A highly lightweight parameter matrix $W_{AFW} \in \mathbb{R}^{L^2 \times 2}$ (only 288 parameters for ViT-B) to dynamically learn the re-composition weights for different target domains.

   During source training, uniform weights are applied: $C_{fusion} = \frac{\sum_{l=0}^{L^2} C(l)}{L^2}$

   During target fine-tuning, AFW is introduced: $C_{fusion} = \frac{W_{AFW} \otimes C}{L^2}$

   **Design Motivation**: Since the parameter count of AFW is extremely small, co-training it with the encoder on the source domain leads to overfitting on source data (experimentally verified: source-trained 61.01% vs. target-introduced 63.22%). Visualizations demonstrate that AFW learns distinctly different weight distributions across different domains, and foreground/background weights automatically exhibit a mutually exclusive trend.

### Loss & Training

The overall loss function consists of the standard BCE loss and the orthogonal regularization:
$$L = L_{BCE} + \lambda L_{orth}$$

where $\lambda = 0.1$, and the performance variation is less than 1% across the range of 0.01 to 0.5, indicating low sensitivity to this hyperparameter.

**Two-stage Training Strategy**:
- **Source-domain Training**: Trained on PASCAL VOC 2012+SBD. OSD is jointly optimized with the ViT encoder, and score maps are combined using uniform weights.
- **Target-domain Fine-tuning**: Freeze $W_{in}$ and $W_{out}$, and fine-tune only $W_{orth}$ and AFW. Since query labels are unavailable in this phase, the support set is used as the query to calculate $L_{BCE}$ and $L_{orth}$.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (1-shot) | Prev. SOTA | Gain |
|--------|------|-------------|----------|------|
| FSS-1000 | mIoU | 80.31 | 79.71 (APSeg) | +0.60 |
| Deepglobe | mIoU | 43.15 | 42.60 (ABCDFSS) | +0.55 |
| ISIC | mIoU | 46.57 | 45.43 (APSeg) | +1.14 |
| Chest X-ray | mIoU | 82.86 | 84.10 (APSeg) | -1.24 |
| **Average** | **mIoU** | **63.22** | **61.30 (APSeg)** | **+1.92** |

| Dataset | Metric | Ours (5-shot) | Prev. SOTA | Gain |
|--------|------|-------------|----------|------|
| FSS-1000 | mIoU | 82.55 | 81.90 (APSeg) | +0.65 |
| Deepglobe | mIoU | 46.83 | 50.12 (DRA) | -3.29 |
| ISIC | mIoU | 55.02 | 53.98 (APSeg) | +1.04 |
| Chest X-ray | mIoU | 84.79 | 84.50 (APSeg) | +0.29 |
| **Average** | **mIoU** | **67.30** | **65.42 (DRA)** | **+1.88** |

Note: Ours uses a ViT-B encoder-only architecture (FLOPs 18.86G). APSeg also uses ViT-B but adopts a SAM-based encoder-decoder architecture, whose parameters and computation cost are far larger than ours.

### Ablation Study

| Configuration | 1-shot Avg mIoU | 5-shot Avg mIoU | Description |
|------|---------------|----------------|------|
| Baseline | 49.88 | 53.64 | Without any modules |
| +CPC | 59.50 | 62.68 | +9.62%, largest source of improvement |
| +CPC+AFW | 61.32 | 65.22 | AFW yields another ~1.8% improvement |
| +CPC+OSD | 60.75 | 64.45 | OSD yields another ~1.3% improvement |
| +CPC+AFW+OSD (Full) | **63.22** | **67.30** | Three modules combined are optimal |

| Distance Metric | Baseline (1-shot) | Ours (1-shot) | Baseline (5-shot) | Ours (5-shot) |
|----------|-----------------|-------------|-----------------|-------------|
| Euclidean | 48.92 | 62.49 | 53.07 | 66.53 |
| Dot | 49.18 | 62.75 | 53.03 | 66.58 |
| EMD | 50.02 | **63.37** | 53.23 | 67.01 |
| Cosine | 49.88 | 63.22 | 53.64 | **67.30** |

| OSD rank | 2 | 4 | 8 | 16 | 32 | 64 |
|----------|------|------|------|------|------|------|
| 1-shot mIoU | 60.39 | 61.73 | 63.22 | 63.25 | 63.43 | 62.61 |

### Key Findings

- **CPC is the Core Contribution**: Introducing CPC improves mIoU by 9.62% (1-shot), demonstrating that cross-comparison is critical for disentangling features.
- **Cross-Comparison Beats Element-wise Matching**: Cross-layer comparison (59.50%) outperforms same-layer comparison (55.14%) by 4.36%, verifying the existence of effective cross-layer semantic correspondences inside ViT.
- **OSD Successfully Reduces Mutual Information**: Across 4 target domains, the MI between support/query features consistently decreases after applying OSD.
- **AFW Learns Domain-Specific Weight Distributions**: Visualizations show that the heatmaps of AFW vary significantly across target domains; the maximum weights do not necessarily lie on the diagonal, and foreground/background weights automatically exhibit mutual exclusion.
- **AFW Should Not Be Trained on the Source Domain**: Jointly training AFW on the source domain yields inferior performance (61.01%) compared to directly introducing it in the target domain (63.22%).
- **Insensitivity to Orthogonal Loss Wight**: $\lambda$ is robust, with mIoU varying by less than 1% across the range of 0.01–0.5.
- **Optimal Computational Efficiency**: The FLOPs count is only 18.86G, which is lower than PATNet (22.63G), HSNet (20.11G), and SSP (18.97G).
- **Robustness to Distance Metrics**: Regardless of the distance metric used, the proposed method consistently outperforms the baseline by large margins (+12–14%).
- **Limited Gain from Multiple Background Prototypes**: Using clustering to obtain multiple background prototypes yields only a minor increase from 63.22% to 63.59%, which does not justify the extra computation.

## Highlights & Insights

1. **Explaining Feature Entanglement via ViT Architecture**: Utilizing the cumulative summation property of the residual joints in ViTs, the paper attributes feature entanglement to the equal-weighted treatment of cross-layer comparisons—where meaningful and meaningless comparisons in $$S = \sum_i \sum_j (Layer_s^i \cdot Layer_q^j)$$ are mixed with identical weights. This analysis is not only intuitive but also rigorously verified by CKA experiments.
2. **"Self-Disentanglement" Design**: Unlike traditional feature disentanglement methods that require auxiliary VAE/GAN networks, this work leverages the structural properties of ViT itself to achieve disentanglement without adding extra network branches, keeping it elegant and simple.
3. **Theoretical Analysis**: By employing the $\mathcal{H}$-divergence domain adaptation theory, the paper demonstrates the mechanism of simultaneously reducing the source domain risk $\epsilon_\mathcal{S}(h)$ and the domain gap $d_\mathcal{H}(\mathcal{S}, \mathcal{T})$.
4. **Extremely Lightweight**: Fine-tuning OSD requires only 64 parameters ($W_{orth}$) and AFW requires only 288 parameters, leading to a total FLOPs count lower than all competing methods.
5. **AFW Foreground/Background Mutual Exclusion Phenomenon**: On the Deepglobe and ISIC datasets, the adaptively learned foreground and background weights exhibit a mutually exclusive relationship, which is an intriguing emergent behavior.

## Limitations & Future Work

1. **Inferior 1-shot Performance on Chest X-ray Compared to APSeg**: In the chest X-ray domain (82.86% vs. 84.10%), APSeg's SAM-based architecture still holds an advantage.
2. **Analysis Limited to ViT Architectures**: The analysis relies heavily on the residual accumulation structure of ViTs. Extending these findings to CNNs or non-standard residual structures remains to be explored.
3. **Single Source Domain**: The method only uses PASCAL VOC as the source domain, without exploring multi-source domains or larger-scale pre-trained data settings.
4. **Annotation Dependency for Target Domain Fine-tuning**: Despite the few-shot setting, obtaining target-domain support-set annotations remains costly in certain application fields.
5. **Fixed Choice of Rank**: Setting rank=8 is a globally optimal empirical value; different target domains might require different optimal ranks.
6. **Scalability Directions**: Combining the method with more powerful pre-trained models like DINOv2 or SAM2, and adapting CPC and AFW designs to multi-class scenarios.

## Related Work & Insights

- **PATNet (ECCV 2022)**: Established the CD-FSS benchmark and evaluation protocol, mapping domain-specific features into domain-agnostic features using a feature transformation layer.
- **APSeg (CVPR 2024)**: A SAM-based automatic prompt network, which also uses ViT-B but with parameter and computation sizes much larger than ours.
- **DRA (CVPR 2024)**: Employs compact adapters to align features across different domains; its 5-shot performance on Deepglobe outperforms ours.
- **ABCDFSS (CVPR 2024)**: Introduces micro-adapters at test time for feature refinement, which is conceptually similar to our target-domain AFW.
- **APM (NeurIPS 2024)**: Uses a lightweight frequency masker for feature extraction; complementary to our structural disentanglement concept.
- **Gandelsman et al. (ICLR)**: Found that the outputs of various ViT components lie in the same feature space, serving as the theoretical foundation for our analysis.
- **Inspirations**: The concept of ViT structural decomposition can be extended to other areas requiring feature disentanglement, such as transfer learning, domain adaptation, and multi-task learning.

## Rating

- Novelty: ⭐⭐⭐⭐ Analyzing feature entanglement through the lens of ViT structural decomposition is a novel perspective. CKA validation and theoretical analysis bolster the persuasiveness of this view; however, the high-level concept of disentanglement and re-composition is relatively common in transfer learning.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The ablation study is extremely comprehensive—encompassing module ablations, distance metric comparisons, rank sensitivity analysis, robustness of the orthogonal weight, comparison strategy comparisons, mutual information validation, CKA domain similarity, AFW visualizations, computational efficiency comparisons, and theoretical analyses.
- Writing Quality: ⭐⭐⭐⭐ The logical flow moves organically from problem identification to mathematical analysis, CKA verification, and then to method design; however, the mathematical formulas appear somewhat densely formatted.
- Value: ⭐⭐⭐⭐ Achieves state-of-the-art results on several CD-FSS benchmarks while maintaining lightweight efficiency; the ViT structural disentanglement perspective is highly generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Adapter Naturally Serves as Decoupler for Cross-Domain Few-Shot Semantic Segmentation](adapter_naturally_serves_as_decoupler_for_cross-domain_few-shot_semantic_segment.md)
- [\[CVPR 2025\] The Devil is in Low-Level Features for Cross-Domain Few-Shot Segmentation](../../CVPR2025/segmentation/the_devil_is_in_low-level_features_for_cross-domain_few-shot_segmentation.md)
- [\[CVPR 2025\] Dual-Agent Optimization framework for Cross-Domain Few-Shot Segmentation](../../CVPR2025/segmentation/dual-agent_optimization_framework_for_cross-domain_few-shot_segmentation.md)
- [\[CVPR 2026\] Cross-Domain Few-Shot Segmentation via Multi-view Progressive Adaptation](../../CVPR2026/segmentation/cross-domain_few-shot_segmentation_via_multi-view_progressive_adaptation.md)
- [\[AAAI 2026\] Bridging Granularity Gaps: Hierarchical Semantic Learning for Cross-Domain Few-Shot Segmentation](../../AAAI2026/segmentation/bridging_granularity_gaps_hierarchical_semantic_learning_for_cross-domain_few-sh.md)

</div>

<!-- RELATED:END -->
