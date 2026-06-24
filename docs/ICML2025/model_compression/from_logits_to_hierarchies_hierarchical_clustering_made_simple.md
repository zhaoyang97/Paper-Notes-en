---
title: >-
  [Paper Note] From Logits to Hierarchies: Hierarchical Clustering made Simple
description: >-
  [ICML 2025][Model Compression][Hierarchical Clustering] The L2H (Logits to Hierarchies) algorithm is proposed, which utilizes only the logits output from pre-trained flat clustering models. Through masked softmax and iterative merging strategies, it constructs high-quality hierarchical clusterings without fine-tuning, significantly outperforming dedicated deep hierarchical clustering models while running on a CPU in less than a minute on ImageNet-scale datasets.
tags:
  - "ICML 2025"
  - "Model Compression"
  - "Hierarchical Clustering"
  - "Logits"
  - "Pre-trained Models"
  - "Dendrogram"
  - "Unsupervised Learning"
date: 2026-05-08
content_hash: 66c4dbdee732e5cc
---

# From Logits to Hierarchies: Hierarchical Clustering made Simple

**Conference**: ICML 2025  
**arXiv**: [2410.07858](https://arxiv.org/abs/2410.07858)  
**Code**: Complete Python implementation (NumPy + SciPy) provided in the paper appendix  
**Area**: Human Understanding  
**Keywords**: Hierarchical Clustering, Logits, Pre-trained Models, Dendrogram, Unsupervised Learning

## TL;DR

The L2H (Logits to Hierarchies) algorithm is proposed, which utilizes only the logits output from pre-trained flat clustering models. Through masked softmax and iterative merging strategies, it constructs high-quality hierarchical clusterings without fine-tuning, significantly outperforming dedicated deep hierarchical clustering models while running on a CPU in less than a minute on ImageNet-scale datasets.

## Background & Motivation

Hierarchical clustering is a classic problem in machine learning with wide applications in fields such as phylogenetic trees, tumor subtyping, and social networks. Compared to flat clustering, hierarchical clustering provides a multi-granularity perspective and enhances interpretability without requiring a pre-specified number of clusters.

Recently, several deep-learning-based specialized hierarchical clustering methods have emerged, such as DeepECT (hierarchical clustering based on autoencoder embedding space) and TreeVAE (hierarchical generative model based on VAE latent space). However, the authors identified **two key limitations** in these approaches:

**Poor scalability**: Computational requirements are extremely high on large-scale datasets, making it difficult to handle a large number of clusters. TreeVAE adds a decoder for each leaf node, causing the computational cost to explode as the number of categories increases.

**Degraded leaf-level performance**: Introducing hierarchical structures sacrifices leaf-level clustering quality, showing a significant gap compared to non-hierarchical models. This implies that hierarchy comes at an unnecessary cost.

These findings prompt the authors to reconsider: **Is it truly necessary to design specialized deep hierarchical clustering architectures**? Can hierarchical structures be extracted from existing high-quality flat clustering models with minimal overhead?

## Method

### Overall Architecture

The core idea of L2H is remarkably elegant: **it exploits the relationships between clusters inherent in the logits of pre-trained flat clustering models to construct a complete hierarchical tree via iterative bottom-up merging**. The entire process does not require access to internal model representations, fine-tuning, and can even operate on black-box models (such as commercial models accessed via APIs).

**Input**: Dataset $\mathcal{D} = \{x_1, \dots, x_N\}$, logits ($N \times K$ matrix) from a pre-trained $K$-clustering model $f_\theta$.

**Output**: A complete hierarchical clustering tree $\mathcal{H}$ (represented as a list of merge steps).

### Key Designs

#### 1. Definition of Base Functions

Two key functions are defined:

- **Cluster assignment function** $h_\theta(x) = \arg\max_k \text{softmax}_k(f_\theta(x))$: returns the cluster assignment of data point $x$
- **Prediction probability function** $g_\theta(x) = \max_k \text{softmax}_k(f_\theta(x))$: returns the maximum assignment probability

#### 2. Masked Softmax — Core Innovation

To measure the association between cluster groups, the authors introduce masked softmax:

$$\text{m\_softmax}_k(\mathbf{v}; G) = \begin{cases} \frac{\exp(v_k)}{\sum_{j \notin G} \exp(v_j)} & \text{if } k \notin G \\ 0 & \text{if } k \in G \end{cases}$$

Given a set of clusters $G$, the masked softmax excludes the clusters in $G$ from the softmax computation. This allows answering a crucial question: **If a certain group of clusters did not exist, where would its data points be reassigned?**

Based on this, the masked version of assignment and probability functions are defined:

- $h_\theta^m(x; G)$: Reassignment over the remaining clusters after excluding $G$
- $g_\theta^m(x; G)$: The corresponding reassignment probability

#### 3. Iterative Merging Algorithm

The L2H algorithm starts with $K$ singleton cluster groups, merging two groups at each step, and constructs a complete hierarchical tree after $K-1$ steps. Each step consists of two phases:

**Phase 1: Selecting the Group to Merge**

Compute a score for each group $G$:

$$s(G) = \Lambda_{x \in \mathcal{D}_c, c \in G} g_\theta(x)$$

where $\Lambda$ is an aggregation function (the sum of cluster-wise means is used in experiments). The group $G^*$ with the lowest score is selected for merging—intuitively, a lower score implies that the data points in this group are "less confident" in their current cluster assignments, indicating a higher likelihood of association with other groups.

**Phase 2: Finding the Most Associated Group**

For all data points in $G^*$, the reassignment after excluding $G^*$ is recomputed using masked softmax. The total reassignment probability received by each external cluster $c$ is accumulated:

$$rp(c) = \sum_{x \in \mathcal{D}_{G^*}, h_\theta^m(x; G^*) = c} g_\theta^m(x; G^*)$$

Then, the group $G^\dagger$ with the highest average reassignment probability is selected for merging:

$$G^\dagger \in \arg\max_{G \in \mathcal{G} \setminus \{G^*\}} \frac{1}{|G|} \sum_{c \in G} rp(c)$$

#### 4. Key Advantages of the Method

- **No fine-tuning required**: Directly utilizes the logits of the pre-trained model, preserving leaf-level clustering performance.
- **Black-box applicability**: Requires no internal representations of the model, relying only on the logits output.
- **Avoids pairwise distances**: Does not compute pairwise distances between data points, significantly reducing computational complexity.
- **High universality**: Class-agnostic and applicable to any clustering or classification model that outputs logits.
- **Supervised extension**: Can be directly applied to supervised classifiers to recover meaningful category hierarchies.

### Loss & Training

L2H itself **requires no training**, meaning there are no loss functions or learnable parameters. It is purely a post-processing algorithm in the inference phase. The workflow is as follows:

1. Perform inference on the dataset using a pre-trained flat model (e.g., TURTLE, TEMI) to obtain the $N \times K$ logits matrix.
2. Run the L2H algorithm on the logits, iteratively merging for $K-1$ steps.
3. Output the hierarchical structure (the list of merge orders of the tree).

The aggregation function $\Lambda$ is the only design choice (hyperparameter). By default, the experiments use the sum of cluster-wise means:

$$\Lambda_{x \in \mathcal{D}_c, c \in G} g_\theta(x) = \sum_{c \in G} \frac{1}{|\mathcal{D}_c|} \sum_{x \in \mathcal{D}_c} g_\theta(x)$$

## Key Experimental Results

### Main Results

A comprehensive comparison is conducted against specialized deep hierarchical clustering methods on CIFAR-10, CIFAR-100, and Food-101.

| Dataset | Method | NMI | ARI | ACC | DP↑ | LHD↓ |
|---------|------|-----|-----|-----|-----|------|
| CIFAR-10 | DeepECT | 0.006 | 0.002 | 0.110 | 0.101 | 0.369 |
| CIFAR-10 | TreeVAE | 0.414 | 0.313 | 0.497 | 0.341 | 0.410 |
| CIFAR-10 | **L2H-TEMI** | 0.907 | 0.906 | 0.956 | 0.902 | 0.348 |
| CIFAR-10 | **L2H-TURTLE** | **0.985** | **0.989** | **0.995** | **0.988** | **0.277** |
| CIFAR-100 | DeepECT | 0.016 | 0.005 | 0.070 | 0.052 | 0.121 |
| CIFAR-100 | TreeVAE | 0.199 | 0.098 | 0.228 | 0.103 | 0.484 |
| CIFAR-100 | **L2H-TEMI** | 0.778 | 0.565 | 0.682 | 0.502 | 0.298 |
| CIFAR-100 | **L2H-TURTLE** | **0.917** | **0.831** | **0.896** | **0.803** | **0.235** |
| Food-101 | DeepECT | 0.003 | 0.000 | 0.011 | 0.010 | 0.333 |
| Food-101 | TreeVAE | 0.114 | 0.017 | 0.057 | 0.016 | 0.483 |
| Food-101 | **L2H-TEMI** | **0.917** | **0.841** | **0.904** | **0.801** | **0.270** |
| Food-101 | **L2H-TURTLE** | 0.894 | 0.800 | 0.876 | 0.758 | 0.297 |

L2H completely dominates specialized deep hierarchical clustering models on all datasets, with NMI improvements typically exceeding **5 to 10-fold**.

### Ablation Study

| Configuration | DP (CIFAR-10) | LHD (CIFAR-10) | DP (CIFAR-100) | LHD (CIFAR-100) | Description |
|------|--------------|----------------|----------------|-----------------|------|
| Aggregation function $\sum_c \sum_x$ | 0.988 | 0.258 | 0.801 | 0.244 | Simple sum |
| Aggregation function $\sum_c \text{mean}_x$ | 0.988 | 0.248 | 0.793 | 0.283 | Sum of within-group means |
| Aggregation function Default choice | 0.988 | **0.277** | **0.803** | **0.235** | Sum of cluster-wise means (Optimal) |
| Alternative merging strategy (All-pairs) | - | - | 0.797 | 0.246 | Considers all group pairs at each step; high computational overhead with no gain |
| L2H-TCL (Weak backbone) | 0.733 | 0.398 | 0.218 | 0.351 | Non-CLIP backbone, still outperforms specialized models |

### Key Findings

1. **Fundamental flaws of deep custom models**: DeepECT undergoes tree collapse on all color datasets (yielding only 2-3 leaf nodes), while TreeVAE can only model 20 leaf nodes on CIFAR-100 (where 100 classes actually exist), restricted by its one-decoder-per-leaf architecture design.

2. **ImageNet Case Study**: Applying L2H to the logits of a pre-trained InternImage classifier successfully reconstructs a category tree highly aligned with the WordNet hierarchy from 1000 classes. An interesting discovery is that snow-related man-made objects (e.g., bobsleds, snowmobiles) are merged with Arctic dog breeds, revealing **spurious correlations** within the model.

3. **Robustness of hyperparameter $K$**: Varying $K$ from 85 to 115 on CIFAR-100 (true value is 100) results in graceful performance degradation, demonstrating that the method is insensitive to the estimation of the number of clusters.

4. **INaturalist21 Multi-level Experiment**: Training TURTLE on 2.7M images with 1103 family-level classes, L2H achieves clustering performance at coarser hierarchical levels (order/class/phylum/kingdom) that **outperforms** TURTLE models trained separately for each level.

## Highlights & Insights

- **An examplar of "Simple is Beautiful"**: The entire L2H algorithm is implemented in fewer than 50 lines of NumPy + SciPy code, yet significantly outperforms specialized deep models requiring hours of GPU training.
- **Ingenious design of masked softmax**: It measures cluster kinship using counterfactual reasoning ("where would data points go if these clusters did not exist"), which is both intuitive and highly efficient.
- **Preserving leaf-level performance**: Since L2H is a pure post-processing method, it does not affect the clustering quality of the original model, which is a fundamental advantage over end-to-end hierarchical models.
- **Cross-paradigm generality**: The same algorithm applies to both unsupervised clustering and supervised classification, providing a new tool for model interpretability.
- **Extreme efficiency**: Runs in <1 minute on CPU for ImageNet-1K-scale datasets, whereas TreeVAE takes hours of GPU training.

## Limitations & Future Work

1. **Dependency on logit quality**: The performance of L2H depends on the informativeness of the pre-trained model's logits. If the model is poorly calibrated, the logits may not accurately reflect the true relationships between clusters.
2. **No automatic level selection**: After producing a complete tree, human inspection is still needed to extract meaningful levels of granularity, lacking an automated selection mechanism.
3. **Validation limited to visual data**: Experiments focus solely on visual datasets, and performance on other modalities like NLP or graph data has not yet been verified.
4. **Choice of aggregation function**: Although ablation studies show limited impact from different aggregation functions, the optimal choice might vary across datasets, lacking an adaptive mechanism.
5. **Insufficient comparison with traditional agglomerative clustering in representation space**: Agglomerative clustering using CLIP embeddings already shows promising performance (e.g., NMI=0.799 on CIFAR-10); further comparisons could analyze the source of information gain in L2H.

## Related Work & Insights

- **TURTLE / TEMI**: Serving as the backbone models for L2H, they represent the current state-of-the-art flat clustering methods using CLIP/DINOv2 pre-trained representations. L2H demonstrates that high-quality flat clusterings already contain rich hierarchical information.
- **TreeVAE / DeepECT**: Representing two pathways of deep hierarchical clustering (generative VAE vs. embedded AE), this paper reveals their fundamental limitations in real-world scenarios.
- **HypHC**: Based on Dasgupta cost optimization in hyperbolic embeddings, it is theoretically elegant but exhibits poor practical clustering performance.
- **Dasgupta Cost**: The theoretical foundation of hierarchical clustering evaluation, to which the DP and LHD metrics of L2H are closely related.
- **Insights**: This work inspires us that in many scenarios, the combination of **lightweight post-processing + strong pre-trained models** may outperform specially designed end-to-end approaches. This design philosophy can be generalized to other structured prediction tasks.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | The method itself is simple but the perspective is novel, posing a strong challenge to the assumption that "specialized architectures are needed". |
| Technical Depth | 3 | The algorithm design is ingenious but the technical barrier is low, with masked softmax + greedy merging at its core. |
| Experimental Thoroughness | 5 | Very solid, featuring multiple datasets, multiple backbones, multiple metrics, comprehensive ablations, and sensitivity analyses. |
| Practicality | 5 | Requires no training, runs on CPU, features concise code, and is plug-and-play. |
| Writing Quality | 4 | Clear exposition, rich figures/tables, and a convincing critical analysis of existing methods. |
| **Total Score** | **4.2** | A highly practical work that achieves compelling SOTA results with an extremely minimalist approach. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Cross-Architecture Distillation Made Simple with Redundancy Suppression](../../ICCV2025/model_compression/cross-architecture_distillation_made_simple_with_redundancy_suppression.md)
- [\[ICCV 2025\] Knowledge Distillation with Refined Logits](../../ICCV2025/model_compression/knowledge_distillation_with_refined_logits.md)
- [\[ICML 2025\] TreeLoRA: Efficient Continual Learning via Layer-Wise LoRAs Guided by a Hierarchical Gradient-Similarity Tree](treelora_efficient_continual_learning_via_layer-wise_loras_guided_by_a_hierarchi.md)
- [\[CVPR 2025\] Logits DeConfusion with CLIP for Few-Shot Learning](../../CVPR2025/model_compression/logits_deconfusion_with_clip_for_few-shot_learning.md)
- [\[ACL 2025\] Beyond Logits: Aligning Feature Dynamics for Effective Knowledge Distillation](../../ACL2025/model_compression/beyond_logits_aligning_feature_dynamics_for_effective_knowledge_distillation.md)

</div>

<!-- RELATED:END -->
