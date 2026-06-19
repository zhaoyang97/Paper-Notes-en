---
title: >-
  [Paper Note] Learning Anchor in Dual Orthogonal Space for Fast Multi-view Clustering
description: >-
  [CVPR 2026][Others][Paper Note] This paper proposes DOSFMVC, which extends anchor learning for large-scale multi-view clustering from a "single space" to a "dual orthogonal space." It jointly learns anchors in a space spanned by the anchors themselves and an additional orthogonal space based on "anchored clustering centers." By replacing traditional
tags:
  - CVPR 2026
  - Others
date: 2026-05-08
content_hash: b0606ad9df5cc5f8
---
# Learning Anchor in Dual Orthogonal Space for Fast Multi-view Clustering

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Qin_Learning_Anchor_in_Dual_Orthogonal_Space_for_Fast_Multi-view_Clustering_CVPR_2026_paper.html)  
**Code**: TBD  
**Area**: Multi-view Clustering / Unsupervised Learning  
**Keywords**: Multi-view clustering, anchor learning, orthogonal space, anchored clustering centers, large-scale clustering  

## TL;DR
This paper proposes DOSFMVC, which extends anchor learning for large-scale multi-view clustering from a "single space" to a "dual orthogonal space." It jointly learns anchors in a space spanned by the anchors themselves and an additional orthogonal space based on "anchored clustering centers." By replacing traditional consensus anchor graphs with cluster indicator matrices for anchors and raw data, it achieves state-of-the-art (SOTA) performance across ACC/NMI/Purity/F1 on 7 datasets (up to ~300k samples) while maintaining linear complexity.

## Background & Motivation
**Background**: Multi-view clustering aims to group data by fusing complementary and consistent information from multiple views (e.g., images, text, and videos of the same news) unsupervised. Traditional methods (co-training, subspace, graph-based) often require eigen-decomposition of an $n\times n$ affinity matrix, resulting in $O(n^2)$ or even $O(n^3)$ complexity, which is prohibitive for large-scale data. Consequently, two types of fast methods have emerged: matrix factorization and **anchor-based methods**. The latter learns a small set of representative anchors to construct a small anchor graph, reducing complexity to linear.

**Limitations of Prior Work**: Existing anchor-based methods **learn anchors only in a single space**, relying on orthogonality or other constraints to derive anchors from multi-view data. The authors identify two deficiencies: (1) Anchors can belong to multiple spaces simultaneously, but current methods fail to exploit **complementary information** across spaces, leading to suboptimal anchor quality. (2) A highly valuable space—the **space based on "anchored clustering centers"**—is ignored by almost all existing works and is not used to assist anchor learning.

**Key Challenge**: Anchor quality determines clustering performance. Single-space constraints provide limited information, leading to "suboptimal" anchors. Furthermore, the use of a "consensus anchor graph" as an intermediary lack clear discriminative structure.

**Goal**: (1) Jointly learn anchors in two orthogonal spaces to utilize complementary information; (2) replace the consensus anchor graph with "anchored clustering centers + anchor clustering indicators" to introduce a second space for anchor learning; (3) integrate anchor learning and partitioning into a unified model while maintaining linear complexity.

**Core Idea**: By constraining anchors simultaneously in the "space spanned by the anchors themselves" and the "space spanned by anchored clustering centers," the model utilizes complementary information from both orthogonal spaces to learn higher-quality, more discriminative anchors and clustering partitions.

## Method

### Overall Architecture
DOSFMVC is a pure optimization-based (non-neural network) multi-view clustering model. Inputs include large-scale data from $v$ views $\{X_p\}_{p=1}^v$ and the number of clusters $k$; the output is the clustering indicator matrix $R$ (final cluster assignments). It integrates "anchor learning in dual orthogonal spaces" and "clustering partition using anchors" into a **single** joint objective function. The variables are solved iteratively via alternating optimization, and clustering results are obtained directly from $R$. The method is formalized as a matrix factorization problem with orthogonal constraints, solved without a multi-stage serial pipeline.

Key notation: $A_p \in \mathbb{R}^{d_p \times m}$ is the **anchor basis** ( $m$ anchors) for the $p$-th view with orthogonality $A_p^T A_p = I$ for discriminative power; $W \in \mathbb{R}^{c\times m}$ is the **clustering indicator for anchors** (non-negative); $R \in \mathbb{R}^{c\times n}$ is the **clustering indicator for raw data** ($R\in\mathrm{Ind}$, one "1" per column); $U_p$ is the **anchored clustering center matrix** serving as the basis for the additional orthogonal space ($U_p^T U_p = I$); and $\alpha$ balances the two terms.

### Key Designs

**1. Joint Anchor Learning in Dual Orthogonal Spaces: Utilizing two complementary spaces instead of one**

Addressing the limitation of single-space information sources, the first space reconstructs data using the anchor basis $A_p$. A small number of anchors linearly represent each view of the large-scale data:

$$\min_{A_p, W, R} \sum_{p=1}^{v} \|X_p - A_p W^T R\|_F^2, \quad \text{s.t. } A_p^T A_p = I,\ W \ge 0,\ R \in \mathrm{Ind}$$

Unlike traditional approaches using a "consensus subspace representation" or "consensus anchor graph," this model introduces anchor indicators $W$ and data indicators $R$, which are **learned simultaneously to reinforce each other**. The second space approximates the anchor basis $A_p$ using anchored clustering centers $U_p$ as a basis: $\min \alpha\|A_p - U_p W\|_F^2$. The combined objective Eq.(3) is:

$$\min_{A_p, U_p, W, R} \sum_{p=1}^{v}\|X_p - A_p W^T R\|_F^2 + \alpha\sum_{p=1}^{v}\|A_p - U_p W\|_F^2$$

The first term learns anchors in their own spanned space, while the second learns them in the space of anchored clustering centers. By utilizing complementary information, "suboptimal anchors" are suppressed. Experiments show optimal performance when $\alpha$ is $10^{-1}$ or $1$, with a significant drop when $\alpha$ is too small, confirming the information gain from the second space.

**2. Additional Space Based on "Anchored Clustering Centers": Replacing consensus graphs for clearer structure**

This design targets the omission of the anchored cluster center space in prior work. Instead of a consensus anchor graph, $U_p$ is introduced as an **orthogonal basis matrix** to express and constrain anchors. The constraint $U_p^T U_p = I$ ensures the space is "fully spanned" and subproblems have finite solutions. Intuitively, anchored clustering centers are closer to cluster semantics than a consensus graph, pulling anchors toward a clustering structure and enhancing discriminability. Removing this space (DOSFMVC-m) leads to decreased performance across all metrics.

**3. Unified Model + Alternating Optimization + Linear Complexity: Four closed-form subproblems solved iteratively**

To resolve the separation of anchor learning and partitioning, the model unifies them and uses alternating optimization to split the non-convex problem into four subproblems with closed-form solutions:

- **Update $A_p$**: Solving independently for each view by setting the derivative to zero: $A_p = (\alpha U_p W + X_p R^T W)(W^T R R^T W + \alpha I)^{-1}$.
- **Update $R$**: Since each column has one non-zero entry, it is solved sample-wise by assigning each sample to the optimal cluster $i^* = \arg\min_i \|(X_p)_{:,j} - A_p W^T_{:,i}\|^2$.
- **Update $U_p$**: Formulated as $\max_{U_p}\mathrm{Tr}(U_p A_p W^T)$, solved via SVD of $A_p W^T$ as $U_p = UV^T$.
- **Update $W$**: Derivative set to zero: $W = (A_p U_p^T + X^T A_p R)(R^T A_p^T A_p R + \alpha U_p U_p^T)^{-1}$.

The total complexity is $O(m^3 + d_p m n + n c^2 d^2 + c^3 d^3 + 2nmd + mnk)$. Given $n \gg k, d, m$ in large-scale scenarios, the complexity is **linear** $O(n)$ relative to the number of samples. The objective value is monotonically non-increasing and bounded below by 0, ensuring convergence.

### Loss & Training
The model has no separate training loss; the objective is Eq.(3). Following Algorithm 1: Initialize $U_p, W, R$ and iteratively update $A_p \to R \to U_p \to W$ until convergence. The hyperparameter $\alpha$ is grid-searched in $\{10^{-4},...,1\}$, and the number of anchors is searched in $\{c, 2c, ..., 5c\}$ (typically $4c$ is optimal).

## Key Experimental Results

### Main Results
Evaluation on 7 multi-view datasets (BBCSport, Notting-Hill, VGGFace2-50, YTF-10/20/100/200, up to 286k samples) against 9 SOTA baselines. ACC(%) results (selected):

| Method | BBCSport | Notting-Hill | VGGFace2-50 | YTF-10 | YTF-20 | YTF-100 | YTF-200 |
|------|----------|--------------|-------------|--------|--------|---------|---------|
| LMVSC | 75.35 | 91.30 | 11.10 | 75.60 | 71.00 | 60.00 | 50.30 |
| CFMC | 75.20 | 91.50 | 12.00 | 86.00 | 74.80 | 65.50 | 61.80 |
| ALPC | 79.00 | 92.30 | 15.25 | 88.27 | 77.50 | 67.28 | 62.20 |
| **Ours** | **82.50** | **94.78** | **17.50** | **90.87** | **81.00** | **70.40** | **64.00** |

NMI, Purity, and F1 metrics also show clear leads (e.g., NMI 86.27 vs. ALPC 84.00 on YTF-200). Traditional methods like SwMC/GMC/PMSC suffer from Out-Of-Memory (OOM) errors on large datasets, highlighting the necessity of anchor-based linear methods.

### Ablation Study
DOSFMVC-m denotes the version without the "anchored clustering centers" space (single-space anchor learning):

| Dataset | Configuration | ACC | NMI | Purity | F1 |
|--------|------|-----|-----|--------|-----|
| BBCSport | DOSFMVC-m | 80.00 | 69.27 | 76.50 | 69.75 |
| BBCSport | **Ours** | 82.50 | 72.00 | 79.28 | 71.40 |
| YTF-200 | DOSFMVC-m | 61.70 | 84.00 | 69.29 | 54.37 |
| YTF-200 | **Ours** | 64.00 | 86.27 | 72.00 | 57.00 |

Indicators consistently dropped across all datasets without the second space, validating the "dual orthogonal space" contribution. Runtime analysis shows DOSFMVC is competitive with the fastest anchor methods (1520s on YTF-200 vs ALPC 1760s).

### Key Findings
- The second space (anchored clustering centers) is the primary performance driver.
- The parameter $\alpha$ has a clear "sweet spot" at $10^{-1}$ or $1$.
- The model is robust to the number of anchors; performance remains stable within the $\{c,...,5c\}$ range.
- Scalability is verified: DOSFMVC runs linearly and maintains SOTA where traditional methods fail due to OOM.

## Highlights & Insights
- The "dual space" concept is simple and effective. It recognizes that anchors can belong to multiple spaces and uses a second orthogonal constraint to mine complementary information.
- Using "anchored clustering centers" as a basis instead of a consensus graph is a key innovation. It aligns the intermediate representation with cluster semantics, and $U_p$ can be solved efficiently via SVD.
- The unified optimization with four closed-form subproblems ensures robustness and eliminates the need for learning rate tuning.
- Accuracy gains do not come at the cost of efficiency; the linear complexity enables processing of hundreds of thousands of samples.

## Limitations & Future Work
- The pure linear and orthogonal model may have limited expressive power compared to deep multi-view clustering for highly non-linear structures.
- Predefining the number of clusters $k$ and anchors remains necessary, lacking an automatic determination mechanism.
- Typos in the original manuscript (e.g., DOSFMNVC-m vs DOSFMVC) and table alignment issues add reading overhead.
- Evaluation is primarily on facial, video, and text benchmarks; performance on highly heterogeneous or incomplete views is not yet explored.

## Related Work & Insights
- **vs. ALPC / CFMC (SOTA Anchor-based)**: These rely on single-space learning and consensus graphs; the dual-space approach in this paper outperforms them across all metrics.
- **vs. LMVSC / SMVSC / FPMVS (Early Anchor Methods)**: This method significantly improves accuracy while maintaining comparable linear complexity.
- **vs. Matrix Factorization (OPMC, etc.)**: While those factorize raw data directly, this method builds a small graph via anchors, with dual-space constraints further refining anchor quality.
- **vs. Traditional Multi-view (SwMC/GMC/PMSC)**: These are $O(n^2)\sim O(n^3)$ and fail on large data, whereas this design is $O(n)$.

## Rating
- Novelty: ⭐⭐⭐⭐ The extension to dual orthogonal spaces is a meaningful advancement over traditional anchor learning.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across 7 datasets and 4 metrics, though ablation could further decompose specific constraints.
- Writing Quality: ⭐⭐⭐ Solid theoretical derivation and complexity analysis, but technical inconsistencies in naming and tables exist.
- Value: ⭐⭐⭐⭐ Highly practical for large-scale multi-view clustering with a balance of precision and efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Scalable Multi-View Subspace Clustering with Tensorized Anchor Guidance](scalable_multi-view_subspace_clustering_with_tensorized_anchor_guidance.md)
- [\[CVPR 2026\] Cluster-aware Anchor Learning for Multi-View Clustering](cluster-aware_anchor_learning_for_multi-view_clustering.md)
- [\[CVPR 2026\] Anti-Degradation Lifelong Multi-View Clustering](anti-degradation_lifelong_multi-view_clustering.md)
- [\[CVPR 2026\] Multi-Hierarchical Contrastive Spectral Fusion for Multi-View Clustering](multi-hierarchical_contrastive_spectral_fusion_for_multi-view_clustering.md)
- [\[CVPR 2026\] Reliable Clustering Number Estimation for Contrastive Multi-View Clustering](reliable_clustering_number_estimation_for_contrastive_multi-view_clustering.md)

</div>

<!-- RELATED:END -->
