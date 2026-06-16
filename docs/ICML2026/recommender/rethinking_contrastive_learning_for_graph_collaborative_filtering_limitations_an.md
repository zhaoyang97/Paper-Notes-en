---
title: >-
  [Paper Note] Rethinking Contrastive Learning for Graph Collaborative Filtering: Limitations and a Simple Remedy
description: >-
  [ICML 2026][Recommender Systems][Sampled Softmax] The authors decompose the forward prediction of LightGCN into a "sum of learnable weights of multi-hop neighbor pairs." They find that the Sampled Softmax (SSM) loss only weights based on the structural similarity of the item-side neighbors and treats all four types of neighbor pairs (UU/II/UI/IU) indiscriminately. Con
tags:
  - ICML 2026
  - Recommender Systems
  - Sampled Softmax
date: 2026-05-08
content_hash: 97aadec5135ad0e0
---
# Rethinking Contrastive Learning for Graph Collaborative Filtering: Limitations and a Simple Remedy

**Conference**: ICML 2026  
**arXiv**: [2605.24015](https://arxiv.org/abs/2605.24015)  
**Code**: https://github.com/geon0325/NT-SSM  
**Area**: Recommender Systems / Graph Collaborative Filtering  
**Keywords**: Graph Collaborative Filtering, Contrastive Learning, Sampled Softmax, Multi-hop Neighbors, Type-aware

## TL;DR
The authors decompose the forward prediction of LightGCN into a "sum of learnable weights of multi-hop neighbor pairs." They find that the Sampled Softmax (SSM) loss only weights based on the structural similarity of the item-side neighbors and treats all four types of neighbor pairs (UU/II/UI/IU) indiscriminately. Consequently, they propose NT-SSM, which incorporates user-side structural similarity into the gradient and calibrates weighting strategies according to neighbor pair types, consistently outperforming SSM across four datasets and various GCF backbones.

## Background & Motivation

**Background**: Graph Collaborative Filtering (GCF), represented by LightGCN, utilizes multi-layer linear propagation on user-item bipartite graphs as the predictive backbone, trained with contrastive learning losses like Sampled Softmax (SSM). This is currently a mainstream paradigm for industrial recommender systems. Most existing analyses of SSM's benefits remain at the representation geometry level, focusing on alignment and uniformity.

**Limitations of Prior Work**: These "representation-perspective" explanations are disconnected from the actual scoring mechanism of GCF—they fail to answer which learnable parameters SSM updates and which sample pairs it promotes. Consequently, SSM remains a black box: it improves performance, but the reasons why and the potential for further enhancement are unclear.

**Key Challenge**: The GCF prediction score $\hat{r}_{ui}$ appears to be the inner product of the final user and item embeddings. However, when expanded, it is a double summation over multi-hop neighbors where only the inner products of ID embeddings are truly learnable; representation-layer metrics are entirely oblivious to this fact.

**Goal**: To fully expand the GCF forward prediction to clarify which neighbor pair inner products are "rewarded" during training. By analyzing the SSM gradient, the study aims to identify what SSM does correctly and what it misses regarding neighbor pair weighting.

**Key Insight**: The authors use the symmetric normalized adjacency matrix $\widetilde{\mathbf{A}}$ to represent $L$-layer propagation as a single structural similarity matrix $\widetilde{\mathbf{S}}=\frac{1}{L+1}\sum_{\ell=0}^{L}\widetilde{\mathbf{A}}^{\ell}$. By rewriting $\hat{r}_{ui}$ as a weighted sum of inner products of multi-hop neighbor pairs, "learning" is naturally mapped to "deciding which neighbor pairs to up-weight."

**Core Idea**: Enable the contrastive loss to simultaneously perceive user-side structural similarity and determine up-weighting strategies according to the four types of neighbor pairs (UU/II/UI/IU). Both aspects are missing in SSM, and addressing them leads to immediate performance gains.

## Method

### Overall Architecture
The method proceeds in three steps: first, rewriting the LightGCN prediction score as a double summation of structural weights and learnable weights; second, performing a closed-form analysis of the SSM gradient to clarify update rules for neighbor pair inner products; and finally, designing the NT-SSM loss by explicitly incorporating "user-side similarity" and "type-awareness" into the gradient. Notably, NT-SSM only modifies the loss function without changing the GCF backbone, allowing it to be directly applied to models like LightGCN, SimGCL, and XSimGCL.

Specifically, after $L$ layers of propagation, the final representations of users and items are $\mathbf{E}=\widetilde{\mathbf{S}}\mathbf{E}^{(0)}$. The prediction score expands to $\hat{r}_{ui}=\sum_{v\in\widetilde{\mathcal{N}}_u}\sum_{v'\in\widetilde{\mathcal{N}}_i}\widetilde{\mathbf{S}}_{uv}\cdot\widetilde{\mathbf{S}}_{iv'}\cdot(\mathbf{e}_v^{(0)\top}\mathbf{e}_{v'}^{(0)})$, where $\widetilde{\mathbf{S}}$ contains no learnable parameters, and only $\mathbf{e}_v^{(0)\top}\mathbf{e}_{v'}^{(0)}$ is trained. The authors categorize neighbor pairs into four types (UU/II/UI/IU) and conduct experiments by controlling the retention ratios $q, q'$. They found that "retaining only a few neighbor pairs with the highest structural similarity" improves NDCG@20 by 35.17% compared to "weighting all pairs," with optimal retention ratios varying across types. These findings serve as the foundation for the loss design.

### Key Designs

**1. Aligning SSM Gradient to "Neighbor Pair Weighting Dynamics": Identifying what is being updated**

To improve SSM, one must first know which parameters are pushed up during training. The authors derive the closed-form update rule for the standard SSM loss $\mathcal{L}(i;u)=-\log\frac{\exp(s(u,i)/\tau)}{\exp(s(u,i)/\tau)+\sum_{j\in\mathcal{B}_u}\exp(s(u,j)/\tau)}$ with respect to the learnable weights $\mathbf{e}_v^{(0)\top}\mathbf{e}_{v'}^{(0)}$: $\partial\mathcal{L}/\partial(\mathbf{e}_v^{(0)\top}\mathbf{e}_{v'}^{(0)})=\frac{\widetilde{\mathbf{S}}_{uv}}{\tau}(\mathbb{E}_{x\sim\pi_u}[\widetilde{\mathbf{S}}_{xv'}]-\widetilde{\mathbf{S}}_{iv'})$. If the structural similarity $\widetilde{\mathbf{S}}_{iv'}$ between the positive item $i$ and neighbor $v'$ exceeds the negative sample expectation, the inner product of this neighbor pair is up-weighted. This explains from a geometric perspective why SSM implicitly performs "structure-aware selective weighting" and repositions the contrastive loss from "representation distribution" to "parameter update rules."

**2. Introducing User-side Structural Similarity: Breaking the "Item-only" Limitation**

The gradient criterion above only depends on the item-side $\widetilde{\mathbf{S}}_{iv'}$. Empirical evidence (Observation 1) shows that the user side must also focus only on structurally similar neighbors for performance gains—original SSM only considers half of the problem. NT-SSM modifies the similarity function $s(u, i)$ to introduce normalized structural weights for both users and items. This ensures that $\widetilde{\mathbf{S}}_{uv}$ no longer degrades into a simple multiplier in the gradient expansion but actively participates in determining whether a neighbor pair should be up-weighted or down-weighted.

**3. Node Type-aware (NT) Neighbor Pair Weighting: Customizing Up-weighting for UU/II/UI/IU**

Observation 2 reveals that the four types of neighbor pairs (UU, II, UI, IU) have vastly different optimal retention ratios across datasets. A uniform rule inevitably sacrifices one category for another. NT-SSM buckets the negative sample expectation $\mathbb{E}_x[\widetilde{\mathbf{S}}_{xv'}]$ by the node types of $v, v'$, effectively calibrating independent up-weighting thresholds for each type. This ensures that even if one type (e.g., II) has lower overall similarity values, it is not suppressed by another type (e.g., UI) with higher values.

### Loss & Training
NT-SSM is a "plug-and-play" replacement for SSM with no additional regularization or two-stage scheduling. The temperature $\tau$ follows standard SSM conventions, and negative sampling can remain in-batch or uniform. It is also compatible with BPR; the authors provide an NT-BPR variant that achieves significant improvements by replacing the sigmoid term of BPR with a type-aware version.

## Key Experimental Results

### Main Results
The authors compared BPR/SSM with their node type-aware versions (NT-BPR/NT-SSM) on four public datasets using LightGCN as the backbone.

| Dataset | Metric | BPR | NT-BPR | Gain |
|--------|------|------|--------|----------|
| LastFM | NDCG@20 | $0.2530\pm0.0016$ | $0.2654\pm0.0016$ | +4.90% |
| MovieLens | NDCG@20 | $0.2953\pm0.0009$ | $0.3154\pm0.0009$ | +6.80% |
| Yelp | NDCG@20 | $0.0449\pm0.0003$ | $0.0480\pm0.0004$ | +6.90% |
| Amazon-Book | Recall@20 | $0.0356\pm0.0002$ | $0.0393\pm0.0003$ | +10.39% |

NT-BPR outperforms BPR across all datasets; the advantage of NT-SSM over SSM shows a consistent trend, with the most significant gains on sparse large-scale graphs like Amazon-Book.

### Ablation Study

| Configuration | Key Finding | Description |
|------|---------|------|
| Full NT-SSM | Best across all datasets | Both user-side and type-awareness enabled |
| "User-side similarity" only | Better than SSM but inferior to full | Validates fix for Limitation 1 |
| "Type-awareness" only | Better than SSM but inferior to full | Validates fix for Limitation 2 |
| Retention ratio $q=q'=100$ | NDCG@20 35.17% lower than optimal | Confirms Observation 1 |

### Key Findings
- The number of multi-hop neighbor pairs reaches billions at 3-hops on Amazon-Book, but most are useless for prediction; weighting only the most structurally similar pairs is optimal.
- SSM's gradient is "half-correct"—it up-weights neighbor pairs similar to the item side but is entirely deaf to the user side; this explains why NT-SSM gains are larger on user-cold-start datasets.
- Replacing the loss with NT-SSM in various GCF backbones (LightGCN, SimGCL, XSimGCL) yields improvements, indicating that the enhancement is decoupled from the backbone and serves as a general loss-layer plugin.

## Highlights & Insights
- Shifting from a "geometric perspective" to a "parameter perspective" provides a gradient-level handle for analysis. This approach of "expanding the sum to see learnable terms" is applicable to all propagation models and can be transferred to Knowledge Graph or multi-behavior recommendation.
- A single gradient formula explains both why SSM is effective and why it is insufficient—it is a rare instance of contrastive learning analysis that both validates and challenges existing methods.
- NT-SSM is a pure loss replacement with zero additional inference overhead and near-zero deployment cost, making it highly friendly for industrial teams.

## Limitations & Future Work
- The analysis holds under the linear propagation of LightGCN; verification is needed for models with non-linear activations like NGCF and PinSage.
- Structural similarity $\widetilde{\mathbf{S}}$ is an offline geometric quantity that may have high estimation bias for long-tail nodes, potentially leading to persistent neglect of niche users.
- The negative sampling strategy remains uniform/in-batch; its coupling with the NT gradient has not been jointly optimized. Type-aware negative samplers could be explored.

## Related Work & Insights
- **vs SSM (Wu et al., 2024)**: This paper provides a "diagnosis + fix" for SSM. The losses share the same origin, but this work explicitly introduces user-side similarity and type-awareness into the gradient.
- **vs SimGCL / XSimGCL (Yu et al., 2022/2023)**: Those works use noise in the representation space for regularization. This work modifies weighting dynamics instead of representations; the two approaches are orthogonal and can be combined.
- **vs BPR (Rendle et al., 2012)**: By upgrading BPR to NT-BPR, the authors demonstrate that the type-aware concept is not limited to the softmax family.

## Rating
- Novelty: ⭐⭐⭐⭐ Redefining contrastive learning from "representation geometry" to "neighbor pair weighting dynamics" is a unique perspective shift.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive validation across four datasets, multiple backbones, and both BPR/SSM families with thorough ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear derivation from expansion to gradient to improvement. Figures 1-3 illustrate insights intuitively.
- Value: ⭐⭐⭐⭐ A zero-overhead loss replacement that serves as a plug-and-play upgrade for industrial recommendation stacks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1"></div>

## Related Papers

- [\[ICML 2026\] GCIB: Graph Contrastive Information Bottleneck for Multi-Behavior Recommendation](gcib_graph_contrastive_information_bottleneck_for_multi-behavior_recommendation.md)
- [\[ACL 2026\] ClusterRAG: Cluster-Based Collaborative Filtering for Personalized Retrieval-Augmented Generation](../../ACL2026/recommender/clusterrag_cluster-based_collaborative_filtering_for_personalized_retrieval-augm.md)
- [\[NeurIPS 2025\] FACE: A General Framework for Mapping Collaborative Filtering Embeddings into LLM Tokens](../../NeurIPS2025/recommender/face_a_general_framework_for_mapping_collaborative_filtering_embeddings_into_llm.md)
- [\[NeurIPS 2025\] Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation](../../NeurIPS2025/recommender/semantic_retrieval_augmented_contrastive_learning_for_sequential_recommendation.md)
- [\[ICLR 2026\] C2AL: Cohort-Contrastive Auxiliary Learning for Large-scale Recommendation Systems](../../ICLR2026/recommender/c2al_cohort-contrastive_auxiliary_learning_for_large-scale_recommendation_system.md)

</div>

<!-- RELATED:END -->
