---
title: >-
  [Paper Note] Neighbor-aware Instance Refining with Noisy Labels for Cross-Modal Retrieval
description: >-
  [AAAI 2026][Cross-modal retrieval] This paper proposes NIRNL, a framework that enhances sample discriminability via Cross-modal Margin Preserving (CMP) and employs Neighbor-aware Instance Refining (NIR) to partition training data into clean, hard, and noisy subsets, each with a tailored optimization strategy. The framework unifies three paradigms—robust learning, label calibration, and instance selection—achieving state-of-the-art cross-modal retrieval performance under high noise rates.
tags:
  - AAAI 2026
  - Cross-modal retrieval
  - noisy labels
  - neighbor-aware
  - instance refining
  - robust learning
date: 2026-05-08
content_hash: 87a0df37eba283f9
---

# Neighbor-aware Instance Refining with Noisy Labels for Cross-Modal Retrieval

**Conference**: AAAI 2026
**arXiv**: [2512.24064](https://arxiv.org/abs/2512.24064)
**Code**: [GitHub](https://github.com/perquisite/NIRNL)
**Area**: Information Retrieval
**Keywords**: Cross-modal retrieval, noisy labels, neighbor-aware, instance refining, robust learning

## TL;DR

This paper proposes NIRNL, a framework that enhances sample discriminability via Cross-modal Margin Preserving (CMP) and employs Neighbor-aware Instance Refining (NIR) to partition training data into clean, hard, and noisy subsets, each with a tailored optimization strategy. The framework unifies three paradigms—robust learning, label calibration, and instance selection—achieving state-of-the-art cross-modal retrieval performance under high noise rates.

## Background & Motivation

Cross-Modal Retrieval (CMR) aims to retrieve semantically relevant samples across different modalities (e.g., images and text). Most existing CMR methods rely on precisely annotated data to learn representations in a shared multimodal semantic space. However, collecting large-scale, high-quality annotations is both costly and time-consuming, and multimodal annotations inevitably contain noise. Noisy labels can severely degrade learned models and impair retrieval performance.

Existing robust CMR methods fall into three broad categories, each with notable limitations:

**Robust Learning** (e.g., RONO): Designs robust loss functions to tolerate the influence of noise. However, it relies on prior assumptions about the noise distribution and can only "tolerate" noise without eliminating its ceiling effect on performance.

**Label Calibration** (e.g., UOT-RCL): Directly corrects noisy labels. However, when class boundaries are ambiguous or the noise distribution heavily overlaps with the true distribution, this approach may introduce new noise or amplify existing errors.

**Instance Selection** (e.g., RSHNL, NRCH): Filters noisy samples and trains on clean data only. However, it is sensitive to preset thresholds, prone to either discarding clean samples or retaining noisy ones, and wastes a substantial amount of training data.

**Core Challenge**: Under complex noise scenarios, how can one dynamically balance **model performance ceiling**, **calibration reliability**, and **data utilization**? NIRNL is specifically designed to address all three dimensions in a unified manner.

## Method

### Overall Architecture

NIRNL comprises two core modules operating in parallel:

1. **CMP (Cross-modal Margin Preserving)**: Constrains the relative distances between positive and negative sample pairs in the embedding space to enhance representational discriminability.
2. **NIR (Neighbor-aware Instance Refining)**: Leverages cross-modal neighborhood consensus to generate soft labels and partitions the dataset into three subsets—clean, hard, and noisy—each with a customized optimization strategy.

### Key Designs

1. **Cross-modal Margin Preserving (CMP)**: CMP applies a triplet-style hinge loss to constrain the relative distances between positive and negative pairs, making intra-class samples more compact and inter-class samples more dispersed:

$$\mathcal{L}_{CMP} = \frac{1}{N} \sum_{i=1}^{N} \sum_{j \neq i}^{N} |\Gamma(f_i^{\mathcal{V}}, f_j^{\mathcal{T}}) - \Gamma(f_i^{\mathcal{V}}, f_i^{\mathcal{T}}) + \mathcal{M}|_+$$

plus the symmetric text→image direction. Here $\mathcal{M}$ is a predefined margin and $|\cdot|_+$ denotes the hinge function. CMP imposes constraints on all samples and serves as a global structural regularizer.

2. **Neighbor-aware Instance Refining (NIR)**: The core mechanism of NIR is to assess label reliability via KNN neighborhood consensus. The process consists of the following steps:

    - **Soft Label Generation**: For each sample $i$, the $K$ nearest neighbors are identified in both the visual and textual modalities, and their class distribution is aggregated as a soft label:
    $\hat{p}(c|\mathcal{V}_i) = \frac{1}{K} \sum_{k=1, \mathcal{V}_k \in \mathcal{N}_i^{\mathcal{V}}}^{K} \mathbb{I}[y_k^c = 1]$

    - **Three-way Dataset Partition**: Samples are divided into three categories based on the consistency between soft labels and ground-truth labels:
      - **Clean subset** $\mathcal{D}_P$: Soft labels from both modalities agree with the ground truth (highly reliable labels).
      - **Hard subset** $\mathcal{D}_H$: Only one modality agrees (uncertain label reliability).
      - **Noisy subset** $\mathcal{D}_N$: Soft labels from both modalities disagree with the ground truth (labels are likely incorrect).

    - **Wasserstein Barycenter Extraction**: An EM algorithm is used to compute the semantic barycenter $\bar{u}_c$ of each class in the shared space, which is subsequently used in the loss computation for each subset.

3. **Differentiated Optimization Strategies for Three Subsets**:

    - **Clean subset**: Optimized directly with cross-entropy loss $\mathcal{L}_P$, fully exploiting reliable supervision signals.
    - **Hard subset**: Optimized with a weighted cross-entropy loss $\mathcal{L}_H$, where the weight $\ell_i = 1 - (1-s(\mathcal{V}_i))(1-s(\mathcal{T}_i))$ reflects the probability of a sample belonging to the correct barycenter, assigning smaller weights to potentially corrupted annotations.
    - **Noisy subset**: The soft labels from both modalities are first fused for label correction $\hat{y}_i = \arg\max_c \hat{p}_i^c$, followed by a robust MAE loss $\mathcal{L}_N$ to mitigate potential biases introduced by label correction.

### Loss & Training

The overall training objective is:

$$\mathcal{L} = (\mathcal{L}_P + \mathcal{L}_H + \mathcal{L}_N) + \alpha \mathcal{L}_{CMP}$$

where the first three terms apply exclusively to their corresponding subsets, and $\mathcal{L}_{CMP}$ applies to all samples. $\alpha$ is a balancing coefficient. The three-way partition is updated dynamically during training—as the quality of model representations improves, neighborhood structures become more accurate, leading to more precise data partitioning and forming a virtuous cycle.

## Key Experimental Results

### Main Results

Evaluation is conducted on three benchmark datasets (Wikipedia, XMedia, INRIA-Websearch) under four noise rates: 0.2, 0.4, 0.6, and 0.8.

**Wikipedia Dataset (MAP%)**:

| Noise Rate | Metric | NIRNL | RSHNL (AAAI'25) | RONO (CVPR'23) | Gain |
|--------|------|-------|-----------------|----------------|------|
| 0.2 | I2T / T2I | 51.6 / 46.6 | 49.1 / 45.4 | 50.5 / 47.1 | +2.5 / +1.2 |
| 0.4 | I2T / T2I | 51.7 / 46.5 | 44.3 / 41.6 | 48.8 / 45.8 | +7.4 / +4.9 |
| 0.6 | I2T / T2I | 49.2 / 46.1 | 38.3 / 36.4 | 45.3 / 41.8 | +10.9 / +9.7 |
| 0.8 | I2T / T2I | 41.7 / 39.4 | 27.8 / 26.8 | 41.6 / 38.2 | +13.9 / +12.6 |

**XMedia Dataset (Mean MAP%)**:

| Method | Noise Rate=0.2 Mean | Noise Rate=0.8 Mean | Overall Mean |
|------|---------------|---------------|--------|
| NIRNL | 92.3 | 91.3 | 91.8 |
| RSHNL | 91.2 | 85.6 | 88.6 |
| RONO | 91.2 | 87.5 | 89.5 |

**INRIA-Websearch Dataset (Mean MAP%)**:

| Method | Noise Rate=0.2 Mean | Noise Rate=0.8 Mean | Overall Mean |
|------|---------------|---------------|--------|
| NIRNL | 53.1 | 50.4 | 52.0 |
| RSHNL | 53.1 | 42.9 | 49.5 |
| NRCH | 43.0 | 41.3 | 42.2 |

NIRNL achieves the best results across all datasets and noise rates, with particularly pronounced advantages at high noise rates (0.6 and 0.8).

### Ablation Study

Ablation analysis at noise rate 0.6:

| Variant | Wikipedia Mean | XMedia Mean | Websearch Mean | Description |
|------|--------------|-------------|---------------|------|
| NIRNL-1 | 24.4 | 40.8 | 8.3 | Remove CMP |
| NIRNL-2 | 44.8 | 88.6 | 46.7 | Discard noisy subset |
| NIRNL-3 | 47.1 | 90.3 | 51.1 | No weighting on hard subset |
| NIRNL-4 | 40.5 | 90.8 | 50.4 | Remove barycenter alignment |
| **NIRNL** | **47.7** | **91.8** | **52.1** | Full framework |

### Key Findings

- **CMP has the largest impact on performance**: Removing CMP causes a dramatic drop to 24.4 on Wikipedia (a 49% decrease), indicating that a structured embedding space is fundamental to noise robustness.
- **Information in the noisy subset cannot be ignored**: Discarding the noisy subset (NIRNL-2) leads to performance degradation, demonstrating that label correction can recover useful information from "bad labels."
- **The weighting strategy for the hard subset is effective**: Without weighting (NIRNL-3), the model becomes overly sensitive to noisy labels.
- **NIRNL correctly identifies the majority of clean samples in later training stages**: As training progresses, the proportion of truly clean samples in the clean subset consistently increases.
- **RSHNL exhibits overfitting on Wikipedia**: This is attributed to its failure to capture the global neighborhood distribution structure.

## Highlights & Insights

1. **Unification of three paradigms**: The paper's most significant contribution is organically unifying robust learning, label calibration, and instance selection within a single framework. The "partition-then-customize" approach avoids the performance ceiling of robust learning, the risk of introducing new noise via label calibration, and the data waste of instance selection.
2. **Cross-modal neighborhood consensus**: Cross-validating with neighborhood information from two distinct modalities provides more reliable noise detection than single-modal approaches. A sample is considered clean only when both modalities "vote" in agreement.
3. **Elegant use of Wasserstein barycenters**: Wasserstein barycenters from optimal transport theory are employed to extract class-level semantic centers, offering greater robustness than simple mean-based centroids.
4. **Well-motivated differentiated loss design**: CE for clean, weighted CE for hard, and MAE for noisy samples—the theoretical guarantee of MAE's robustness to noisy labels is cleverly leveraged.
5. **Comprehensive experimental setup**: The comparison across four noise rates × three datasets × bidirectional retrieval × ten baseline methods is highly thorough.

## Limitations & Future Work

1. **Fixed backbone networks**: Experiments keep feature extractors (VGG-19, AlexNet) frozen without exploring end-to-end fine-tuning—adopting stronger pretrained models (e.g., CLIP) may yield further improvements.
2. **Only symmetric noise evaluated**: Experiments exclusively use symmetric label noise, leaving asymmetric or instance-dependent noise unexplored.
3. **Threshold setting for the three-way partition**: Although neighborhood consensus is used in lieu of hard thresholds, the choice of $K$ in KNN still requires tuning, and the paper does not provide a detailed sensitivity analysis of $K$.
4. **Computational overhead**: Computing global KNN and Wasserstein barycenters at each epoch may introduce additional overhead for large-scale datasets.
5. **Limited to image-text retrieval**: Although the framework is generalizable in principle, it is only validated on image-text bimodal settings and has not been extended to additional modalities such as video or audio.

## Related Work & Insights

- **RSHNL (AAAI'25)**: An instance selection method using a self-paced learning strategy and the most direct baseline in this work; however, it neglects global neighborhood distribution structure and wastes noisy data.
- **RONO (CVPR'23)**: A robust method employing discriminative center learning that suppresses but does not correct noise.
- **GNN4CMR (TPAMI'23)**: A graph neural network-based method whose performance degrades sharply under high noise.
- **Insights**: The neighborhood consensus combined with three-way partition strategy is potentially transferable to other multimodal learning tasks affected by noise, such as noisy pair problems in visual question answering and image-text generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The unified three-paradigm framework design is novel, and the neighborhood-consensus-based three-way partition is distinctively insightful.
- **Technical Depth**: ⭐⭐⭐⭐ — The theoretical foundations of Wasserstein barycenters and differentiated loss functions are solid.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three datasets, four noise rates, ten baselines, complete ablation, and robustness analysis.
- **Value**: ⭐⭐⭐⭐ — Code is open-sourced; the practical demand for noisy-label scenarios is well-motivated.
- **Writing Quality**: ⭐⭐⭐⭐ — Structure is clear, mathematical derivations are complete, and figures are informative.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ReFeed: Retrieval Feedback-Guided Dataset Construction for Style-Aware Query Rewriting](refeed_retrieval_feedback-guided_dataset_construction_for_style-aware_query_rewr.md)
- [\[AAAI 2026\] Knowledge Completes the Vision: A Multimodal Entity-aware Retrieval-Augmented Generation Framework for News Image Captioning](knowledge_completes_the_vision_a_multimodal_entity-aware_retrieval-augmented_gen.md)
- [\[ACL 2026\] From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines](../../ACL2026/information_retrieval/from_relevance_to_authority_authority-aware_generative_retrieval_in_web_search_e.md)
- [\[ACL 2026\] Region-R1: Reinforcing Query-Side Region Cropping for Multi-Modal Re-Ranking](../../ACL2026/information_retrieval/region-r1_reinforcing_query-side_region_cropping_for_multi-modal_re-ranking.md)
- [\[ICLR 2026\] Embedding-Based Context-Aware Reranker](../../ICLR2026/information_retrieval/embedding-based_context-aware_reranker.md)

</div>

<!-- RELATED:END -->
