---
title: >-
  [Paper Note] Bipartite Mode Matching for Vision Training Set Search from a Hierarchical Data Server
description: >-
  [AAAI 2026][training set search] This paper proposes a hierarchical data server combined with a Bipartite Mode Matching (BMM) framework. It organizes large-scale source data via multi-granularity hierarchical clustering…
tags:
  - "AAAI 2026"
  - "training set search"
  - "domain adaptation"
  - "bipartite matching"
  - "hierarchical clustering"
  - "data-centric approach"
date: 2026-05-08
content_hash: 196a05068e2898f5
---

# Bipartite Mode Matching for Vision Training Set Search from a Hierarchical Data Server

**Conference**: AAAI 2026
**arXiv**: [2601.09531](https://arxiv.org/abs/2601.09531)  
**Code**: [github.com/yorkeyao/BMM](https://github.com/yorkeyao/BMM)  
**Area**: Other
**Keywords**: training set search, domain adaptation, bipartite matching, hierarchical clustering, data-centric approach

## TL;DR

This paper proposes a hierarchical data server combined with a Bipartite Mode Matching (BMM) framework. It organizes large-scale source data via multi-granularity hierarchical clustering and employs the Hungarian algorithm to perform one-to-one matching between semantic modes of the source and target domains, thereby retrieving a training set that minimizes the distributional gap to the target domain. The approach significantly outperforms existing training set search methods on person re-identification and object detection tasks.

## Background & Motivation

### Problem Setting

In practical applications such as autonomous driving, medical imaging, and large-scale surveillance, target-domain data is available but real-time annotation is infeasible due to high cost and the need for expert knowledge. A viable alternative is to search a large-scale pre-annotated data server for a suitable training set.

### Core Challenge — Mode Granularity Mismatch

The target domain typically contains multiple distinct semantic modes. When a retrieved training set fails to cover these modes, model performance degrades. The key issue is **granularity mismatch**: for instance, if the target domain contains an "apple" mode, a naive matching may retrieve only "real apples" or the broader "fruit" category, rather than "all forms of apples" at a semantically commensurate level.

### Limitations of Prior Work

Existing training set search methods (e.g., SnP, NDS) primarily reduce distributional discrepancy through optimization or model feedback, but **overlook the potential of optimizing the structure of the data server itself**. Moreover, these methods are often task-specific (e.g., SnP targets re-ID only).

### Inspiration

Drawing an analogy to the hierarchical organization of search engines (e.g., Google's PageRank constructs a page hierarchy via link analysis), the authors propose organizing the data server as a hierarchical structure to support mode matching at varying granularities.

## Method

### Overall Architecture

The BMM framework consists of three core steps:
1. **Hierarchical Data Server Construction**: Balanced k-means followed by bottom-up agglomerative clustering on source data to build a multi-level semantic mode tree.
2. **Target Domain Mode Extraction**: Flat clustering on target data to extract distinct semantic modes.
3. **Bipartite Matching**: FID distances between source and target modes serve as edge weights; the Hungarian algorithm finds the optimal one-to-one matching.

### Key Designs

#### 1. **Hierarchical Data Server**

**Function**: Organizes source data into a multi-level semantic mode tree, providing matching candidates at different granularities.

**Mechanism**:
- A pretrained ImageNet feature extractor is used to obtain features $\mathcal{F}_s = \{F(x_i)\}_{i \in [m_s]}$ for all source images.
- **Balanced k-means** partitions the feature space into $J$ equal-sized base clusters $\{C^1, \ldots, C^J\}$ by minimizing SSE:

$$SSE = \sum_{k=1}^{J} \sum_{x_i \in C_k} \|x_i - \theta_k\|^2 \quad \text{s.t.} \quad |C_k| = \frac{m_t}{K}, \forall k$$

- **Agglomerative clustering** then merges the closest cluster pairs bottom-up to produce a dendrogram.
- All intermediate clusters of varying sizes generated during merging are treated as candidate modes, yielding $H$ modes in total.

**Design Motivation**: Balanced k-means (versus standard k-means) is critical to prevent long-tail distributions caused by large size disparities among same-level modes, thereby ensuring matching quality. Ablations confirm this improvement reduces FID from 51.93 to 50.48 and raises mAP from 26.08% to 28.16%.

#### 2. **Target Mode Extraction and Bipartite Graph Construction**

**Function**: Extracts semantic modes from the target domain and constructs a complete bipartite graph between source and target modes.

**Mechanism**:
- **Flat clustering** is applied to the target domain to extract $L$ modes $\{T^1, \ldots, T^L\}$. Flat (rather than hierarchical) clustering is used because hierarchical clustering produces highly correlated high- and low-level modes that hinder mode selection.
- A bipartite graph $G = (V, E)$ is constructed with all source and target modes as vertices; edge weights are FID distances between source–target mode pairs:

$$\text{FID}(x, y) = \|\mu_x - \mu_y\|_2^2 + \text{Tr}(\Sigma_x + \Sigma_y - 2(\Sigma_x \Sigma_y)^{1/2})$$

where $\mu$ and $\Sigma$ denote the mean and covariance of image descriptors within each cluster.

#### 3. **Optimal Matching via the Hungarian Algorithm**

**Function**: Finds the globally optimal one-to-one mode matching in the bipartite graph.

**Mechanism**: The optimal permutation $\sigma^*$ minimizing total matching cost is sought:

$$\sigma^* = \arg\min_{\sigma \in \mathfrak{S}_L} \sum_{i}^{L} \text{FID}(T^i, S^{\sigma(i)})$$

The matched source clusters are then merged (with deduplication) to form the retrieved training set $S^* = \{S^{\sigma(i)}\}_{i=1}^L$.

**Why Hungarian over greedy matching**: Direct greedy matching can cause multiple target modes to match the same source mode (occurring in 4 out of 20 cases in experiments), reducing data diversity. Allowing duplicates lowers SSIM (15.85 vs. 20.45) and mAP (20.14 vs. 26.08) despite lower FID.

### Loss & Training

BMM is a data retrieval framework that does not involve neural network training. The retrieved training set can be directly used to train task models (e.g., IDE for re-ID, RetinaNet for detection), or combined with pseudo-label UDA methods for further gains.

**Time Complexity**: Overall $\mathcal{O}(J^3)$; the hierarchical construction is a one-time offline process, and per-target-domain matching costs $\mathcal{O}(\log J \cdot J \cdot L)$, making deployment practical.

## Key Experimental Results

### Main Results

**Person Re-ID (5% pruning ratio, IDE model)**

| Method | Target: AlicePerson | Target: Market |
|--------|---|---|
| | FID↓ / R1↑ / mAP↑ | FID↓ / R1↑ / mAP↑ |
| Random Sampling | 81.41 / 33.16 / 14.49 | 39.65 / 47.39 / 23.97 |
| NDS | 61.01 / 44.63 / 22.81 | 31.63 / 49.17 / 24.77 |
| SnP | 60.64 / 47.26 / 25.45 | 30.37 / 51.96 / 26.56 |
| CCDR | 60.52 / 48.47 / 25.04 | 31.07 / 50.87 / 27.08 |
| **BMM** | **51.93 / 49.28 / 26.08** | **27.05 / 53.03 / 28.39** |

**Vehicle Detection (5% pruning ratio, RetinaNet model)**

| Method | Target: ExDark | Target: Region100 |
|--------|---|---|
| | FID↓ / mAP↑ / mAP@50↑ | FID↓ / mAP↑ / mAP@50↑ |
| Random Sampling | 105.74 / 23.50 / 56.13 | 251.90 / 11.38 / 25.30 |
| SnP | 59.78 / 32.15 / 67.18 | 153.82 / 15.07 / 34.65 |
| CCDR | 57.98 / 32.96 / 67.67 | 142.48 / 22.25 / 44.49 |
| **BMM** | **56.34 / 34.83 / 76.57** | **140.07 / 23.08 / 46.34** |

BMM achieves the lowest FID and the highest (or second-highest) accuracy across all target domains and tasks.

### Ablation Study

| Configuration | FID↓ | mAP↑ | Note |
|---|---|---|---|
| Baseline (flat clustering + greedy search) | 60.64 | 25.45 | SnP method |
| Hierarchical clustering only (no mode matching) | ~60 | ~25 | No significant gain alone |
| Mode matching only (flat clustering) | ~60 | ~25 | No significant gain alone |
| **Hierarchical clustering + mode matching** | **51.93** | **26.08** | Effective only when combined |
| Standard k-means | 51.93 | 26.08 | Baseline |
| **Balanced k-means** | **50.48** | **28.16** | Simple modification yields notable gains |
| Direct matching (duplicates allowed) | 51.07 | 20.14 | Low SSIM, insufficient diversity |
| Direct matching (duplicates removed) | 60.84 | 22.07 | Some target modes unmatched |
| **BMM (Hungarian matching)** | **51.93** | **26.08** | Balances diversity and domain gap |

### Key Findings

1. **Hierarchical structure eliminates hyperparameter tuning**: Flat clustering requires precisely identifying the "sweet spot" for the number of clusters, whereas hierarchical clustering maintains stable performance as long as the cluster count is not too small (source clusters >200, target clusters >20 suffice).
2. **Orthogonal complementarity with UDA methods**: BMM + MMT pseudo-labels raises mAP from 26.08% to 78.95% (targeting Market), far exceeding either method alone.
3. **Cross-task generalization**: The same framework is effective for both re-ID and detection without task-specific modifications.

## Highlights & Insights

1. **Data-centric vs. model-centric**: In contrast to the predominant UDA literature focused on model improvements, BMM optimizes data server structure, offering a complementary path to performance gains.
2. **Elegance of hierarchical structure**: The hierarchy is built offline once and reused across different target domains, enabling efficient deployment.
3. **Interpretability of mode matching**: Visualizations of matching results intuitively demonstrate how the retrieved training set covers diverse target-domain modes (e.g., low-light scenes).
4. **Balanced k-means as a principled design**: The simple equal-size constraint yields meaningful improvements, embodying the principle of task-aware data structure design.

## Limitations & Future Work

1. **Feature extractor dependency**: The current approach relies on ImageNet-pretrained features, which may be less effective for domains with large semantic divergence from ImageNet.
2. **Limitations of FID as a mode distance**: FID assumes Gaussian feature distributions, which may be inaccurate for complex multimodal distributions.
3. **Extension to denser prediction tasks**: Applicability to segmentation, keypoint detection, and other dense prediction tasks remains to be validated.
4. **Scalability of the data server**: The Hungarian algorithm has complexity $O(J^3)$, which may require approximation algorithms for very large-scale servers.
5. **Dynamic updates**: Adding new data to the server requires rebuilding the hierarchy; incremental update mechanisms warrant further investigation.

## Related Work & Insights

- **Training Set Search**: SnP (Yao et al. 2023) is the most closely related predecessor, proposing a search-and-prune framework; BMM substantially improves the search stage.
- **Unsupervised Domain Adaptation (UDA)**: BMM is orthogonal and complementary to UDA methods such as MMT and Adaptive Teacher.
- **Data Valuation**: Data Shapley (Ghorbani & Zou 2019) selects data based on sample-level value, complementing BMM's mode-matching perspective.
- **Implications for deployment**: In real-world scenarios, constructing a structured data server may be a cost-effective, high-return strategy for improving transfer learning performance.

## Rating

- Novelty: ⭐⭐⭐⭐ (novel combination of hierarchical server and bipartite matching; data structure optimization perspective is distinctive)
- Experimental Thoroughness: ⭐⭐⭐⭐ (multi-task, multi-target-domain validation; comprehensive ablations; joint experiments with UDA methods)
- Writing Quality: ⭐⭐⭐⭐ (clear structure, intuitive figures)
- Value: ⭐⭐⭐⭐ (data-centric approach has practical value for transfer learning)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning-Augmented Online Bipartite Fractional Matching](../../NeurIPS2025/others/learning-augmented_online_bipartite_fractional_matching.md)
- [\[CVPR 2026\] ViT3: Unlocking Test-Time Training in Vision](../../CVPR2026/others/vit3_unlocking_test_time_training_in_vision.md)
- [\[AAAI 2026\] Scalable Vision-Guided Crop Yield Estimation](scalable_vision-guided_crop_yield_estimation.md)
- [\[AAAI 2026\] CAE: Hierarchical Semantic Alignment for Image Clustering](hierarchical_semantic_alignment_for_image_clustering.md)
- [\[AAAI 2026\] Forget Less by Learning from Parents Through Hierarchical Relationships](forget_less_by_learning_from_parents_through_hierarchical_relationships.md)

</div>

<!-- RELATED:END -->
