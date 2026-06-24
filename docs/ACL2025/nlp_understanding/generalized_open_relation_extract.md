---
title: >-
  [Paper Note] Towards a More Generalized Approach in Open Relation Extraction
description: >-
  [ACL 2025][NLP Understanding][Open Relation Extraction] This paper proposes the MixORE framework, which operates under a highly generalized Open Relation Extraction setting (where unlabeled data simultaneously contains both known and novel relations, without making any long-tail or pre-segmentation assumptions). By utilizing a Semantic Autoencoder to detect novel relations, combined with open-world semi-supervised joint learning, MixORE comprehensively outperforms state-of-th…
tags:
  - "ACL 2025"
  - "NLP Understanding"
  - "Open Relation Extraction"
  - "Generalized OpenRE"
  - "Semi-supervised Learning"
  - "Contrastive Learning"
  - "Novel Relation Detection"
date: 2026-05-08
content_hash: 978525308cec90ee
---

# Towards a More Generalized Approach in Open Relation Extraction

**Conference**: ACL 2025  
**arXiv**: [2505.22801](https://arxiv.org/abs/2505.22801)  
**Code**: [https://github.com/qingwang-isu/MixORE](https://github.com/qingwang-isu/MixORE)  
**Area**: NLP Understanding  
**Keywords**: Open Relation Extraction, Generalized OpenRE, Semi-supervised Learning, Contrastive Learning, Novel Relation Detection  

## TL;DR

This paper proposes the MixORE framework, which operates under a highly generalized Open Relation Extraction setting (where unlabeled data simultaneously contains both known and novel relations, without making any long-tail or pre-segmentation assumptions). By utilizing a Semantic Autoencoder to detect novel relations, combined with open-world semi-supervised joint learning, MixORE comprehensively outperforms state-of-the-art (SOTA) methods on FewRel, TACRED, and Re-TACRED.

## Background & Motivation

### Development of Open Relation Extraction

Traditional Relation Extraction (RE) relies heavily on a pre-defined set of relations and abundant labeled data, making it incapable of handling emerging relation types. Open Relation Extraction (OpenRE) aims to actively discover novel relations from unlabeled data.

### Limitations of Prior Work in OpenRE

**Setting 1 (Unsupervised RE)**: Assumes that the unlabeled data consists exclusively of novel relations, thus ignoring the valuable information from known relations.

**Setting 2 (Semi-supervised OpenRE)**: Assumes that the unlabeled data is pre-segmented into known and novel subsets, which is unrealistic to assume beforehand in real-world scenarios.

**The "Long-tail" Assumption in KNoRD**: Assumes that novel relations are rare, belong to a long-tail distribution, and tend to be expressed explicitly. However, novel relations do not necessarily follow a long-tail distribution (e.g., when a concept in a newly-emerged domain first appears, the number of its instances can be quite substantial).

### Our Generalized Setting

This work relaxes the "long-tail" assumption, assuming only that the unlabeled data contains both known and novel relation instances, without imposing any constraints on relation distributions. This is significantly closer to practical application scenarios.

## Method

### Overall Architecture

MixORE is a two-stage framework:

**Phase 1: Novel Relation Detection**
- Goal: Identify potential novel relation instances from unlabeled data.
- Output: A weak-label set $\mathcal{D}_w$ for novel relations.

**Phase 2: Open-World Semi-Supervised Joint Learning (OW-SS Joint Learning)**
- Goal: Jointly optimize both known relation classification and novel relation discovery.
- Input: Labeled data $\mathcal{D}_l$ + weakly-labeled data $\mathcal{D}_w$.

### Key Designs

#### 1. Relation Encoder

A BERTbase model is utilized as the encoder. Typed entity markers (e.g., `<e1:type>`, `</e1:type>`) are inserted into the input sentence, and the hidden vectors at the positions of the two entity markers are concatenated to form the relation representation:

$$h_r = [h_{<e1:type>} | h_{<e2:type>}]$$

#### 2. Semantic Autoencoder (SAE) for Novel Relation Detection

The core idea is that known relation instances will cluster around their corresponding one-hot vectors in the latent space, while novel relation instances will emerge as outliers due to their mismatch with any known relations.

- Each known relation is represented as a one-hot vector. An SAE is trained to map the feature space to a $|\mathcal{C}_{known}|$-dimensional latent space.
- The SAE utilizes tied weights (where the transposed weight matrix serves as the decoder), with the following objective function:
  $$\min_W \|X_l - W^\top S_l\|_F^2 + \lambda \|WX_l - S_l\|_F^2$$
- The Bartels-Stewart algorithm is applied to compute a closed-form solution, eliminating the need for iterative updates.
- During inference, unlabeled data is projected into the latent space, and the cosine similarity between the projected representation and each known relation's one-hot vector is calculated.
- **Instances falling into the lowest 5% of projection scores are labeled as outliers** (candidates for novel relations).

#### 3. GMM Clustering for Weak Labels

A Gaussian Mixture Model (GMM) is applied to cluster the detected outliers into $|\mathcal{C}_{novel}|$ novel relation groups. Instances with a GMM posterior probability exceeding 0.95 are retained as high-quality weak labels $\mathcal{D}_w$.

#### 4. OW-SS Joint Learning

A continual learning strategy is adopted, warming up on $\mathcal{D}_l$ first, and then continually training on $\mathcal{D}_l \cup \mathcal{D}_w$.

### Loss & Training

The total loss consists of three components: $\mathcal{L} = \mathcal{L}_c + \mathcal{L}_{lm} + \mathcal{L}_e$

**1. Classification Loss $\mathcal{L}_c$ (Cross-Entropy)**:
$$\mathcal{L}_c = -\frac{1}{D_c}\sum_{i=1}^{D_c}\sum_{r=1}^{|\mathcal{C}_u|} y_r^i \log(\hat{y_r^i})$$

**2. Triplet Margin Loss for Labeled Data $\mathcal{L}_{lm}$**:
- Positive sample pairs are constructed solely from $\mathcal{D}_l$ (to avoid propagating noise from the weak labels).
- The number of positive pairs is fixed at $D_m = 5D_c$ to ensure uniform sampling of each relation.
- Triplet margin loss is implemented, measured via cosine distance.

**3. Clustering Exemplar Loss $\mathcal{L}_e$**:
- K-Means is used to compute relation exemplars at multiple levels of granularity.
- Instance representations are encouraged to align with their respective cluster centers.
- The exemplars are dynamically updated at the end of each training epoch.

**Inference Phase**: Known relations are identified via classification results, while novel relations are categorized using Faiss K-Means clustering.

## Key Experimental Results

### Main Results

**Dataset Settings**: FewRel (41 relations, 6 novel), TACRED (41 relations, 6 novel), Re-TACRED (39 relations, 6 novel)

**FewRel Results**:

| Method | Known F1 | B³ F1 | V-measure F1 | ARI |
|------|----------|-------|-------------|-----|
| ORCA | 0.6210 | 0.5481 | 0.5492 | 0.4318 |
| KNoRD | 0.7738 | 0.7318 | 0.7297 | 0.6945 |
| **MixORE** | **0.8328** | **0.8968** | **0.8802** | **0.8817** |

**TACRED Results**:

| Method | Known F1 | B³ F1 | V-measure F1 | ARI |
|------|----------|-------|-------------|-----|
| KNoRD | 0.8519 | 0.7680 | 0.7883 | 0.7193 |
| **MixORE** | **0.8833** | **0.8682** | **0.8599** | **0.8473** |

**Re-TACRED Results**:

| Method | Known F1 | B³ F1 | V-measure F1 | ARI |
|------|----------|-------|-------------|-----|
| KNoRD | 0.8669 | 0.6389 | 0.7306 | 0.5081 |
| **MixORE** | **0.9156** | **0.8750** | **0.8613** | **0.8925** |

### Key Findings

1. **MixORE comprehensively outperforms all baselines across all datasets**, demonstrating substantial advantages on both known and novel relations.
2. **Comparison with KNoRD**: On Re-TACRED, the ARI increases by 0.3844 (from 0.5081 to 0.8925), which is a massive improvement.
3. **Unsupervised methods like HiURE/AugURE perform decently in novel clustering but yield very poor results for known classification** (with F1 scores around 0.43-0.49). This indicates that the "treat all as novel" strategy is unsuitable for the generalized setting.
4. **Ablation studies show that**: Removing the NRD phase (predicting all samples as known) drops the known F1 from 0.8606 to 0.7374 (a 12.3% decrease); eliminating continual learning reduces the novel B³ F1 from 0.8968 to 0.8134.

## Highlights & Insights

1. **Precise Problem Definition**: Methodically structures existing OpenRE assumptions and proposes a more reasonable, generalized setup.
2. **Elegant Novelty Detection via SAE**: Utilizes one-hot vectors of known relations to constrain the latent space structure, causing novel relations to naturally emerge as outliers.
3. **Lightweight First Phase**: Freezes BERT parameters and solves the SAE using the Bartels-Stewart closed-form solution, achieving high efficiency.
4. **The 5% Significance Threshold is Theoretically Grounded**: Aligns with conventions in statistical hypothesis testing.
5. **Triplet Loss Constructs Positive Pairs Solely from Labeled Data**: Avoids incorporating false positive pairs caused by weak-label noise.

## Limitations & Future Work

1. **Requires Prior Knowledge of the Number of Novel Relations $|\mathcal{C}_{novel}|$**: This is typically unknown in real-world scenarios. Approaches to automatically determine the number of clusters (such as BIC/AIC) could be considered.
2. **The 5% Threshold May Lack Robustness**: The proportion of novel relations varies significantly across different datasets.
3. **BERTbase Limitation**: Larger pre-trained models such as DeBERTa-v3 could be explored.
4. **Simplicity of Data Augmentation Strategies**: Relying on off-the-shelf intra-sentence/inter-sentence augmentations, where more advanced methods like LLM-based generation could be considered.
5. **Validation Only on English Datasets**: Cross-lingual scenarios remain unexplored.

## Related Work & Insights

- **Integration of Open-World Learning and Continual Learning**: MixORE organically integrates Open-world SSL (Cao et al., 2022 ORCA) and Continual RE (Cui et al., 2021).
- **New Application Scenarios for SAE**: Transfers the Semantic Autoencoder from zero-shot learning to outlier-based novel relation detection in relation extraction.
- **Evolution of Contrastive Learning in RE**: Advances from instance-level (Liu et al., 2022) to exemplar-level multi-granularity contrastive learning.
- **Insight**: Proposing a generalized setting propels the OpenRE field toward more practical scenarios; similar ideas can be extended to Open NER and Open Event Extraction.

## Rating

| Dimension | Score (1-10) | Description |
|------|-------------|------|
| Novelty | 8 | Precise problem definition; novel approach using SAE for novel relation detection. |
| Experimental Thoroughness | 8 | Evaluated on three datasets, supported by rich ablation studies and baseline comparisons. |
| Writing Quality | 8 | Clear logic and highly detailed methodology. |
| Value | 7 | The generalized setup is very close to real-world applications, though assuming a known number of novel relations is still a limitation. |
| Overall Score | 8 | High-quality work that substantially propels the OpenRE field forward. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Variational Approach for Mitigating Entity Bias in Relation Extraction](a_variational_approach_for_mitigating_entity_bias_in_relation_extraction.md)
- [\[ACL 2025\] Generating Diverse Training Samples for Relation Extraction with Large Language Models](generating_diverse_training_samples_for_relation_extraction_with_large_language_.md)
- [\[ACL 2025\] Beyond Prompting: An Efficient Embedding Framework for Open-Domain Question Answering](embqa_embedding_odqa.md)
- [\[ACL 2025\] RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering](rise_reasoning_enhancement_via_iterative_self-exploration_in_multi-hop_question_.md)
- [\[ACL 2026\] HCRE: LLM-based Hierarchical Classification for Cross-Document Relation Extraction](../../ACL2026/nlp_understanding/hcre_llm-based_hierarchical_classification_for_cross-document_relation_extractio.md)

</div>

<!-- RELATED:END -->
