---
title: >-
  [Paper Note] QuRe: Query-Relevant Retrieval through Hard Negative Sampling in Composed Image Retrieval
description: >-
  [ICML2025][Recommender Systems][Composed Image Retrieval] Proposes QuRe, which improves user satisfaction in Composed Image Retrieval (CIR) by simultaneously retrieving target images and other relevant images through a hard negative sampling strategy based on steep drops in relevance scores and a reward model optimization objective.
tags:
  - "ICML2025"
  - "Recommender Systems"
  - "Composed Image Retrieval"
  - "Hard Negative Sampling"
  - "Reward Model"
  - "Bradley-Terry"
  - "Human Preference Alignment"
date: 2026-05-08
content_hash: 9a2272d9c14190e4
---

# QuRe: Query-Relevant Retrieval through Hard Negative Sampling in Composed Image Retrieval

**Conference**: ICML2025  
**arXiv**: [2507.12416](https://arxiv.org/abs/2507.12416)  
**Code**: [jackwaky/QuRe](https://github.com/jackwaky/QuRe)  
**Area**: Image Retrieval  
**Keywords**: Composed Image Retrieval, Hard Negative Sampling, Reward Model, Bradley-Terry, Human Preference Alignment

## TL;DR

Proposes QuRe, which improves user satisfaction in Composed Image Retrieval (CIR) by simultaneously retrieving target images and other relevant images through a hard negative sampling strategy based on steep drops in relevance scores and a reward model optimization objective.

## Background & Motivation

Composed Image Retrieval (CIR) retrieves a target image using a reference image combined with a text description. Existing methods suffer from a key limitation:

- **Focusing solely on target image retrieval**: Datasets usually annotate only one target per query, treating all other images as negative samples.
- **Contrastive learning causing false negatives**: Training treats all images in a batch except the target as negative samples, mistakenly pushing away unannotated images that are highly relevant to the query.
- **User satisfaction is neglected**: Even if the target is retrieved, the rest of the top-k results are often flooded with irrelevant images, resulting in a poor user experience.

Core Problem: Standard Recall@k only measures whether the target appears in the top-k, failing to reflect the overall quality of the retrieved set.

## Method

### Overall Architecture

QuRe is based on the BLIP-2 architecture (ViT-L image encoder + Q-Former) and mainly consists of two innovations:

1. **Reward model training objective**: Replaces traditional contrastive loss with the Bradley-Terry preference model.
2. **Hard negative sampling strategy**: Locates the hard negative sample interval based on double steep drops in relevance scores.

### Relevance Score

For each image $I$ in the corpus, the relevance score is defined as the inner product of the multimodal query embedding and the image embedding:

$$s(x_I, x_T, I) = \frac{Q(E_{img}(x_I), x_T) \cdot Q(E_{img}(I))}{\tau}$$

where $E_{img}$ is the BLIP-2 image encoder, $Q$ is the Q-Former, and $\tau$ is a learnable temperature parameter.

### Training Objective (Bradley-Terry Preference Model)

Unlike contrastive learning, which treats all non-target samples in a batch as negatives, QuRe adopts a reward model objective, pairing only one positive sample and one negative sample at a time:

$$p^*(I_p \succ I_n \mid x_I, x_T) = \sigma(s(x_I, x_T, I_p) - s(x_I, x_T, I_n))$$

The objective function minimizes the negative log-likelihood (equivalent to minimizing the KL divergence):

$$\mathcal{L} = -\mathbb{E}_{(x_I, x_T, I_p, I_n) \sim \mathbb{D}^*} [\log(p^*(I_p \succ I_n \mid x_I, x_T))]$$

where $I_p = y_I$ (target image), and $I_n$ is sampled from the hard negative sample set $\mathbb{H}$.

### Hard Negative Sample Set Sampling (Core Contribution)

**Two conditions**:

- C1: Negative samples should have lower relevance to the query than the target image.
- C2: Relevance scores of negative samples should be close to the target image (challenging).

**Specific steps**:

1. Sort all images in the corpus in descending order of relevance score: $\mathbb{S}_i = \{s_{i,1}, \ldots, s_{i,N_{img}}\}$
2. Take the subset of scores lower than the target: $\mathbb{S}_i^{<targ} = \{s_{i,j} \mid s_{i,j} < s(x_{I_i}, x_{T_i}, y_i)\}$
3. Find the **two positions with the largest adjacent score differences** $k_1, k_2$ in this subset (i.e., the two steepest score drop points).
4. The hard negative sample set is defined as the images between the two steep drops:

$$\mathbb{H}_i = \{I_j \mid j \in [\min(k_1,k_2)+1,\ \max(k_1,k_2)],\ s_{i,j} < s(x_{I_i}, x_{T_i}, y_i)\}$$

**Intuition**: The first steep drop excludes false negatives (highly similar to the target), and the second steep drop excludes negatives that are too easy. The intermediate region contains exactly the hard negative samples that "differ from the query in at least one key attribute (e.g., color, shape)".

### Training Details

- Re-update the hard negative sample set every $\lfloor n_{epoch} / n_{def} \rfloor$ epochs ($n_{def}=6$).
- Initial warm-up phase: The hard negative sample set contains the entire corpus except the target.
- Uniformly sample one negative from $\mathbb{H}$ in each epoch to ensure diversity.

## Key Experimental Results

### FashionIQ Validation Set (Recall@10 / Recall@50)

| Method | Dress R@10 | Shirt R@10 | Toptee R@10 | Average R@10 | Average Avg |
|------|-----------|-----------|------------|----------|---------|
| CLIP4CIR | 38.32 | 44.31 | 47.27 | 43.30 | 55.03 |
| SPRC | 45.71 | 51.37 | 55.48 | 50.86 | 62.13 |
| **QuRe** | **46.80** | **53.53** | **57.47** | **52.60** | **63.04** |

### CIRR Test Set

| Method | R@1 | R@5 | R@10 | R_s@1 | R@5+R_s@1 |
|------|-----|-----|------|-------|-----------|
| SPRC | 50.75 | 80.58 | 88.72 | 79.57 | 80.07 |
| **QuRe** | **52.22** | **82.53** | **90.31** | 78.51 | **80.52** |

- Average R@10 on FashionIQ improves by **+1.74%** compared to SPRC.
- R@1 on CIRR improves by **+1.47%**, and R@5 improves by **+1.95%**.

### HP-FashionIQ Human Preference Alignment

QuRe demonstrates the best human preference alignment on the newly constructed HP-FashionIQ dataset, indicating that its retrieval results overall are more aligned with user expectations.

## Highlights & Insights

1. **Precise Problem Definition**: For the first time, it recourse to the objective of "not only retrieving the target but also ensuring other top-k results are relevant" in CIR.
2. **Ingenious Hard Negative Strategy**: Adaptively locates hard negative sample intervals using the two steep drop points of relevance scores without requiring additional annotations.
3. **Reward Model Objective**: Borrows the Bradley-Terry model from RLHF to compare only one positive and negative pair at a time, naturally avoiding false negative issues.
4. **HP-FashionIQ Dataset**: Fills the gap of lacking a human preference evaluation benchmark in the CIR field (61 participants, 2,715 valid queries).
5. **Resource Efficient**: Can be trained on a single RTX 3090 GPU, offering strong practicality.

## Limitations & Future Work

1. **Slightly Lower R_s@K Metric than SPRC**: Since QuRe allows false negatives to receive high scores, there is a minor regression in subset recall metrics.
2. **Periodic Reconstruction of Hard Negative Sets Required**: Calculating steep drop points requires sorting the entire corpus every $\lfloor n/n_{def} \rfloor$ epochs, and the computational cost grows with the size of the corpus.
3. **Robustness of the Double Steep Drop Hypothesis**: When the distribution of relevance scores is smooth with no obvious steep drops, the definition of the hard negative sample interval may be unstable.
4. **Validated Only in Fashion and General Domains**: Lacks experimental validation in complex scenarios such as medical or remote sensing domains.
5. **Limited Scale of HP-FashionIQ**: Covers only two categories, shirts and toptees, and its generalization capability remains to be verified.

## Related Work & Insights

*   **CoVR-BLIP** / **SPRC**: Strongest existing CIR baselines, both adopting contrastive learning.
*   **HCL** (Robinson et al., 2020): Classic definition of hard negatives — different categories + small embedding distance.
*   **RLHF** (Ouyang et al., 2022): The source of inspiration for the Bradley-Terry preference model.
*   **FNC** (Huynh et al., 2022): Filters false negatives using a threshold, whereas QuRe achieves an adaptive alternative via steep drop points.

## Rating

*   Novelty: ⭐⭐⭐⭐ — Introducing the RLHF preference model to CIR training and double steep drop hard negative sampling are both novel ideas.
*   Experimental Thoroughness: ⭐⭐⭐⭐ — Two standard datasets plus a newly constructed human preference dataset, with a complete ablation study.
*   Writing Quality: ⭐⭐⭐⭐ — Clear motivation, rigorous formula derivation, and intuitive illustrations.
*   Value: ⭐⭐⭐⭐ — Trainable on a single GPU, open-source code, advancing CIR research from the perspective of user satisfaction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation](../../NeurIPS2025/recommender/semantic_retrieval_augmented_contrastive_learning_for_sequential_recommendation.md)
- [\[AAAI 2026\] Inductive Generative Recommendation via Retrieval-based Speculation](../../AAAI2026/recommender/inductive_generative_recommendation_via_retrieval-based_speculation.md)
- [\[ACL 2026\] ClusterRAG: Cluster-Based Collaborative Filtering for Personalized Retrieval-Augmented Generation](../../ACL2026/recommender/clusterrag_cluster-based_collaborative_filtering_for_personalized_retrieval-augm.md)
- [\[AAAI 2026\] CroPS: Improving Dense Retrieval with Cross-Perspective Positive Samples in Short-Video Search](../../AAAI2026/recommender/crops_improving_dense_retrieval_with_cross-perspective_positive_samples_in_short.md)
- [\[ACL 2026\] Bridging Language and Items for Retrieval and Recommendation: Benchmarking LLMs as Semantic Encoders](../../ACL2026/recommender/bridging_language_and_items_for_retrieval_and_recommendation_benchmarking_llms_a.md)

</div>

<!-- RELATED:END -->
