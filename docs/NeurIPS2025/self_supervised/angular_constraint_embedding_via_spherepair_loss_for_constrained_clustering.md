---
title: >-
  [Paper Note] Angular Constraint Embedding via SpherePair Loss for Constrained Clustering
description: >-
  [NeurIPS 2025][Self-Supervised Learning][Constrained clustering] This paper proposes the SpherePair loss function, which performs pairwise constraint embedding learning in angular space (rather than Euclidean space)…
tags:
  - "NeurIPS 2025"
  - "Self-Supervised Learning"
  - "Constrained clustering"
  - "angular embedding"
  - "SpherePair loss"
  - "deep constrained clustering"
  - "spherical representation"
date: 2026-05-08
content_hash: 76d2bbd372403d51
---

# Angular Constraint Embedding via SpherePair Loss for Constrained Clustering

**Conference**: NeurIPS 2025
**arXiv**: [2510.06907](https://arxiv.org/abs/2510.06907)
**Code**: Available (repository mentioned in the paper)
**Area**: Clustering / Representation Learning
**Keywords**: Constrained clustering, angular embedding, SpherePair loss, deep constrained clustering, spherical representation

## TL;DR

This paper proposes the SpherePair loss function, which performs pairwise constraint embedding learning in angular space (rather than Euclidean space), enabling a deep constrained clustering method that requires neither anchors nor prior knowledge of the number of clusters, while providing rigorous theoretical guarantees for determining optimal hyperparameters.

## Background & Motivation

**Background**: Deep constrained clustering (DCC) improves unsupervised clustering by integrating domain knowledge in the form of pairwise constraints (must-link / cannot-link). Existing approaches fall into two categories: (1) end-to-end DCC, which introduces anchors to model clustering as a pseudo-classification task; and (2) deep constrained embedding, which learns constraint-aware representations before applying a clustering algorithm.

**Limitations of Prior Work**: End-to-end methods (e.g., VanillaDCC, VolMaxDCC, CIDEC) rely on anchors to indirectly infer global cluster assignments from local pairwise relations, but anchors may be misaligned with true cluster centers, and the number of clusters $K$ must be known in advance. Deep constrained embedding methods (e.g., AutoEmbedder, CPAC) learn representations in Euclidean space; however, because Euclidean distances are unbounded $[0, +\infty)$, margins must be set manually, and distance control between positive and negative pairs can be conflicting.

**Key Challenge**: Both the anchor mechanism in end-to-end methods and Euclidean distance in embedding methods suffer from fundamental deficiencies — the former requires global information but has access only to local constraints, while the latter's unbounded distance space leads to hyperparameter sensitivity and potential constraint conflicts.

**Goal**: (1) How can positive and negative constraints be encoded fairly without anchors? (2) How can the need for manual tuning of margins and embedding dimensions be eliminated? (3) How can the number of clusters be inferred without knowing $K$?

**Key Insight**: The authors observe that angular space (cosine similarity) is naturally bounded $[0, \pi]$, and that the geometric properties of the hypersphere can guarantee equidistant cluster distributions without constraint conflicts.

**Core Idea**: Perform constraint embedding learning in angular space rather than Euclidean space, exploit the closedness of spherical geometry to eliminate constraint conflicts, and automatically determine optimal hyperparameters through theoretical derivation.

## Method

### Overall Architecture

Given a dataset and pairwise constraints (positive pairs: same cluster; negative pairs: different clusters), a deep autoencoder is trained to learn an angular embedding space in which same-cluster samples have small angles and different-cluster samples are pushed into the negative zone. After the embedding is learned, a standard clustering algorithm such as K-means is applied to the normalized spherical representations $\mathcal{Z}_{\text{sphere}}$ to produce the final clustering, without requiring end-to-end joint training.

### Key Designs

1. **SpherePair Loss Function**:

   - **Function**: Encodes pairwise constraints in angular space, driving positive-pair angles toward 0 and negative-pair angles toward the negative zone boundary $\pi/\omega$.
   - **Mechanism**: For each constraint pair $(a_i, b_i, y_i)$, the angle $\theta$ between embedding vectors is computed and mapped via cosine to a similarity score $\text{Sim} \in [0,1]$; a logistic regression loss $\mathcal{L}_{\text{ang}}$ is then minimized. Positive pairs use $\cos(\theta)$ directly; negative pairs use $\cos(\min(\omega\theta, \pi))$, where $\omega$ is an angular factor controlling the size of the negative zone.
   - **Design Motivation**: The bounded angular distance $[0, \pi]$ avoids the normalization issues and margin sensitivity caused by unbounded Euclidean distances. The negative zone mechanism $\pi/\omega$ ensures sufficient separation between different clusters.

2. **Theoretically Determined Optimal Hyperparameters**:

   - **Function**: Theoretically determines the optimal values of the angular factor $\omega$ and embedding dimension $D$, eliminating manual tuning.
   - **Mechanism**: Theorem 4.4 proves that when $D \geq K$, any $\omega \geq \pi/\arccos(-1/(K-1))$ is valid; Corollary 4.5 further establishes that when $D$ is sufficiently large, $\omega = 2$ is the universally optimal choice (effective for any $K$). Under this setting, the $K$ clusters in the embedding form a regular simplex in a $(K-1)$-dimensional subspace.
   - **Design Motivation**: The theoretical guarantees ensure that SpherePair requires no adjustment of $\omega$ for different datasets or values of $K$; fixing $\omega = 2$ suffices. The requirement $D \geq K$ is easily satisfied in practice, even when $K$ is unknown.

3. **PCA-Based Cluster Count Inference**:

   - **Function**: Infers the number of clusters from the learned spherical representations when $K$ is unknown.
   - **Mechanism**: Based on the angular invariance property in Theorem 4.6 — when the PCA projection dimension $d \geq K-1$, inter-cluster angles are preserved — the method computes the minimum inter-cluster angle $\delta_d$ for samples involved in cannot-link constraints across subspaces of increasing dimension, and identifies the plateau onset $d^* = K-1$ in the sequence $\{\bar{\delta}_d\}$, yielding $\hat{K} = d^* + 1$.
   - **Design Motivation**: The geometric structure of the spherical representations enables direct inference of $K$ without retraining or the cumbersome posterior cluster validation required by end-to-end methods.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_{\text{ang}} + \lambda \mathcal{L}_{\text{recon}}$, where the reconstruction loss $\mathcal{L}_{\text{recon}}$ prevents representation collapse. Embeddings are normalized before decoding to preserve angular properties. Default hyperparameters are $\lambda = 0.02$, $\omega = 2$ (theoretically determined), and $\rho = 0.05$. An Adam optimizer is used, with a stacked denoising autoencoder for initialization.

## Key Experimental Results

### Main Results

SpherePair is compared against 7 DCC baselines on 8 benchmarks using ACC/NMI/ARI under constraint counts of 1k/5k/10k:

| Dataset | Constraints | Metric | SpherePair | Runner-up | Gain |
|---------|-------------|--------|-----------|-----------|------|
| CIFAR100-20 | 10k | ACC | 62.8/62.6 | 54.6 (VanillaDCC) | +8.2 |
| CIFAR10 | 10k | ACC | 90.5/89.9 | 90.1 (CIDEC) | +0.4 |
| FashionMNIST | 10k | ACC | 84.8/83.6 | 81.5 (SDEC) | +3.3 |
| ImageNet10 | 1k | ACC | 92.1/91.7 | 94.3 (SDEC) | Near SOTA |

### Ablation Study

| Configuration | Key Feature | Notes |
|---------------|-------------|-------|
| SpherePair (full) | $\omega=2$, $D=10/20$ | Full model; best across all datasets |
| SpherePair† (no pretraining) | Random initialization | Still outperforms most pretrained baselines, demonstrating the strength of angular embedding learning |
| AutoEmbedder | Euclidean distance + manual margin | Among the weakest embedding methods; requires extensive tuning |
| Imbalanced constraints (IMB1/IMB2) | Skewed constraint distribution | SpherePair remains stable; end-to-end methods degrade severely |

### Key Findings

- **SpherePair achieves optimal or near-optimal results across all datasets and constraint scales**, with particularly pronounced advantages under low constraint counts (1k) and imbalanced constraint conditions.
- **Strong generalization**: The performance gap between training and test sets is minimal (typically <1%), indicating that the learned representations generalize well to unseen samples.
- **Accurate cluster count inference**: The PCA plateau detection method correctly infers $K$ on most datasets.
- **SpherePair† without pretraining still outperforms most pretrained baselines**, confirming that the methodological advantages of angular constraint embedding are independent of initialization.

## Highlights & Insights

- **Paradigm shift from Euclidean to angular space**: The central insight is that the boundedness of angular space $[0, \pi]$ naturally resolves the margin sensitivity and constraint conflict issues inherent to Euclidean space. This perspective is transferable to other metric learning tasks, such as semi-supervised learning and prototypical networks for few-shot learning.
- **Theory-driven hyperparameter elimination**: Fixing $\omega = 2$ and requiring $D \geq K$ via rigorous mathematical derivation is uncommon in deep learning research. The conditions derived from regular simplex geometry eliminate the most burdensome tuning step in conventional approaches.
- **Complete decoupling of representation learning and clustering**: After learning the embedding, K-means is applied directly; cosine and Euclidean distances are equivalent on normalized features, making this highly practical. This design has important implications for scalability to large-scale datasets.

## Limitations & Future Work

- **Theoretical guarantees depend on constraint quality**: If cannot-link constraints do not cover all true clusters, cluster count inference will fail. Whether the sampling distribution of constraints is appropriate in practice remains a challenge.
- **Fully connected encoder architecture**: For fair comparison, the experiments use simple fully connected encoders. Whether stronger backbones are needed for more complex data (e.g., high-resolution images) has not been investigated.
- **Assumes clear cluster structure**: For data with continuous distributions or hierarchical structure, the assumption of equidistant spherical embeddings may not hold.
- **Constraint acquisition cost not discussed**: The paper assumes constraints are available, but obtaining high-quality pairwise constraints may itself be a bottleneck in practice.

## Related Work & Insights

- **vs. VanillaDCC/VolMaxDCC**: These end-to-end methods use anchors to bridge representations and cluster assignments, but anchor misalignment propagates errors. SpherePair removes anchors entirely, avoiding this issue through a two-stage pipeline of angular embedding followed by K-means.
- **vs. AutoEmbedder**: Both are anchor-free constrained embedding approaches, but AutoEmbedder operates in Euclidean space and requires manual margin setting. SpherePair replaces the margin with the natural boundedness of angular space, with $\omega$ determined theoretically.
- **vs. supervised angular learning (ArcFace, etc.)**: Angular losses in face recognition rely on class labels and class center prototypes. SpherePair is the first to apply angular learning to the weakly supervised setting with only pairwise constraints, without requiring class centers.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Angular-space constraint embedding is a first in this field, with a solid theoretical foundation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Eight datasets, seven baselines, multiple constraint conditions, and comprehensive hyperparameter robustness analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are clear, though the dense mathematical notation may challenge non-specialist readers.
- **Value**: ⭐⭐⭐⭐ Introduces a new paradigm for constrained clustering with both theoretical contributions and practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] You Can Trust Your Clustering Model: A Parameter-free Self-Boosting Plug-in for Deep Clustering](you_can_trust_your_clustering_model_a_parameter-free_self-boosting_plug-in_for_d.md)
- [\[NeurIPS 2025\] Hybrid Autoencoders for Tabular Data: Leveraging Model-Based Augmentation in Low-Label Settings](hybrid_autoencoders_for_tabular_data_leveraging_model-based_augmentation_in_low-.md)
- [\[NeurIPS 2025\] Soft Task-Aware Routing of Experts for Equivariant Representation Learning](soft_task-aware_routing_of_experts_for_equivariant_representation_learning.md)
- [\[NeurIPS 2025\] Understanding Ice Crystal Habit Diversity with Self-Supervised Learning](understanding_ice_crystal_habit_diversity_with_self-supervised_learning.md)
- [\[NeurIPS 2025\] Know Thyself by Knowing Others: Learning Neuron Identity from Population Context](know_thyself_by_knowing_others_learning_neuron_identity_from_population_context.md)

</div>

<!-- RELATED:END -->
