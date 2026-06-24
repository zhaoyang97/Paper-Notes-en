---
title: >-
  [Paper Note] Hierarchical Level-Wise News Article Clustering via Multilingual Matryoshka Embeddings
description: >-
  [ACL 2025][Multilingual & Machine Translation][Matryoshka embeddings] A method leveraging multilingual Matryoshka embeddings is proposed to achieve hierarchical news clustering. Different dimensional subsets of the embeddings correspond to semantic similarities of various granularities (theme $\rightarrow$ topic $\rightarrow$ event). Combined with an improved hierarchical agglomerative clustering algorithm, this approach achieves state-of-the-art (SOTA) performance on SemEval…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Matryoshka embeddings"
  - "multilingual"
  - "hierarchical clustering"
  - "news similarity"
  - "agglomerative clustering"
date: 2026-05-08
content_hash: 6d1f34d58d715205
---

# Hierarchical Level-Wise News Article Clustering via Multilingual Matryoshka Embeddings

**Conference**: ACL 2025  
**arXiv**: [2506.00277](https://arxiv.org/abs/2506.00277)  
**Code**: [GitHub](https://github.com/hanshanley/multilingual-matryoshka-news)  
**Area**: Multilingual / News Clustering  
**Keywords**: Matryoshka embeddings, multilingual, hierarchical clustering, news similarity, agglomerative clustering  

## TL;DR

A method leveraging multilingual Matryoshka embeddings is proposed to achieve hierarchical news clustering. Different dimensional subsets of the embeddings correspond to semantic similarities of various granularities (theme $\rightarrow$ topic $\rightarrow$ event). Combined with an improved hierarchical agglomerative clustering algorithm, this approach achieves state-of-the-art (SOTA) performance on SemEval 2022 Task 8 (Pearson $\rho = 0.816$).

## Background & Motivation

- **Problem Definition**: In a globalized news ecosystem, analyzing and clustering multilingual news at different granularities (event/topic/theme) is essential to understanding media coverage patterns and cross-lingual information dissemination.
- **Limitations of Prior Work**: (1) Existing LLM-based methods are mostly monolingual, non-scalable, or unable to differentiate between different granularities of similarity; (2) Decoder-based LLMs (e.g., GPT-4) incur prohibitively high costs when processing large-scale documents; (3) Encoder-based models only measure "similarity" via cosine similarity, which has an ambiguous definition; (4) Clustering methods often require prior knowledge of the target cluster count.
- **Key Insight**: The nested hierarchical structure of Matryoshka Representation Learning is naturally suited to encoding semantic information of different granularities—lower dimensions capture coarse-grained themes, whereas higher dimensions capture fine-grained events.
- **Goal**: Train multilingual Matryoshka embeddings and design a hierarchical agglomerative clustering algorithm to automatically discover news groups at various granularities without requiring prior knowledge of the cluster count.

## Method

### Overall Architecture

A two-stage approach:
1. **Embedding Training**: Train multilingual Matryoshka embeddings based on a modified AngIE loss to encode different granularities of similarity in different dimensional subsets.
2. **Hierarchical Clustering**: Deploy an improved algorithm based on Reciprocal Agglomerative Clustering (RAC) to perform layer-wise clustering using different dimensional subsets of the embeddings.

### Key Designs

- **Hierarchical Matryoshka Training**: Impose distinct similarity thresholds at different dimensions—the $d/4$ dimension distinguishes "very dissimilar" vs. others; the $d/2$ dimension distinguishes "somewhat dissimilar" vs. others; and the $d$ dimension distinguishes all four levels of similarity. This forces the embeddings to learn coarse-grained concepts in lower dimensions and fine-grained details in higher dimensions.
- **Modified AngIE Loss**: $$\mathcal{L}_{mat} = \mathcal{L}_{\text{AngIE}_{diss}}(\mathbf{H}_{d/4}) + \mathcal{L}_{\text{AngIE}_{somewhat}}(\mathbf{H}_{d/2}) + \mathcal{L}_{\text{AngIE}_{same}}(\mathbf{H}_{d})$$, which combines three sub-objectives: cosine, contrastive, and angle constraints.
- **SimCSE Enhancement**: During training, encode each sample twice using different dropout masks to generate implicit positive pairs, which strengthens the quality of the monolingual embedding space.

### Clustering Algorithm

Three-layer hierarchical clustering:
1. **Layer 1 (Theme)**: Employs $d/4$-dimensional embeddings + RNN merging, with threshold $\lambda_1$.
2. **Layer 2 (Topic)**: Within each Layer 1 cluster, applies $d/2$-dimensional embeddings + RNN merging, with threshold $\lambda_2$.
3. **Layer 3 (Event)**: Within each Layer 2 cluster, uses full $d$-dimensional embeddings + RNN merging, with threshold $\lambda_3$.

### Data Augmentation

- **Style Augmentation**: GPT-4o generates 3 paraphrased versions of each article in different styles.
- **Entity Sensitivity**: Use Spacy + T5 to swap named entities to generate "somewhat similar" samples.
- **Language Expansion**: Expand the original 10 languages to 54, scaling the final training set to 4.1 million article pairs.

## Experiments

### Main Results (SemEval 2022 Task 8)

| Model | SE-22 (Pearson $\rho$) | SE-22 Extended |
|------|:-:|:-:|
| mE5-base (baseline) | 0.604 | 0.582 |
| fine-mE5-base (Ours) | 0.817 | 0.812 |
| mat-mE5-base-192 (Ours) | 0.799 | 0.808 |
| mat-mE5-base-384 (Ours) | 0.792 | **0.816** |
| GateNLP-UShef (Prev. SOTA) | 0.801 | – |

### Ablation Study

| Ablation Option | SE-22 $\rho$ (192d) | SE-22 Ext $\rho$ (192d) |
|--------|:-:|:-:|
| Full Model | 0.799 | 0.808 |
| Remove SimCSE dropout | 0.693 | 0.733 |
| Remove contrastive loss | $\approx 0$ | $\approx 0$ |
| Train only on original SE-22 data | 0.828 | 0.706 |

### Clustering Performance (Miranda Dataset, BERTopic F1)

| Model | Precision | Recall | F1 |
|------|:-:|:-:|:-:|
| mE5-base | 0.8507 | 0.3715 | 0.5171 |
| mat-mE5-base-192 | 0.7895 | 0.8971 | **0.8399** |
| fine-mE5-base | 0.7791 | 0.5735 | 0.6607 |

### Key Findings

- Matryoshka embeddings significantly outperform traditional fine-tuned embeddings in distinguishing different similarity levels (achieving the highest AUROC across all levels).
- The dropout positive pairs from SimCSE are crucial for training, with performance dropping from 0.799 to 0.693 upon removal.
- Data augmentation is indispensable for multilingual generalization; training solely on the original data drops performance to 0.706 on the extended test set.
- In BERTopic clustering, the Matryoshka-192d embedding alone reaches an F1 score of 0.84, substantially outperforming the full-dimensional fine-tuned model (0.66).
- Multilingual alignment: The average relational similarity with English reaches 0.753, with Portuguese being the highest (0.839) and Burmese being the lowest (0.452).

## Highlights & Insights

- Redefining Matryoshka representation learning from "learning the same information at different dimensions" to "learning different granularities of information at different dimensions" is conceptually innovative and highly intuitive.
- The hierarchical clustering algorithm naturally aligns with the hierarchical structure of the embeddings, eliminating the need to predefine the number of clusters.
- Supports large-scale multilingual evaluation across 54 languages.
- The training data augmentation strategies (style paraphrasing, entity replacement, and translation) are systematically designed.

## Limitations & Future Work

- The clustering thresholds $\lambda_1, \lambda_2, \lambda_3$ require empirical adjustment on development sets; different data distributions might require hyperparameter retuning.
- Only a 512-token context window is supported, potentially losing information for long-form news articles.
- Heavily relies on GPT-4o for data augmentation and translation, which incurs high costs and introduces translation biases.
- Embedding alignment quality remains weak for low-resource languages (e.g., Burmese, Kannada).

## Related Work & Insights

- **Semantic Embeddings**: SimCSE (Gao et al., 2021), E5 (Wang et al., 2022), AngIE (Li & Li, 2024)
- **Matryoshka Learning**: Nested representation learning of MRL (Kusupati et al., 2022)
- **News Clustering**: BERTopic (Grootendorst, 2022), multilingual news datasets from Miranda et al. (2018)
- **SemEval Task**: Multilingual news similarity evaluation of SemEval 2022 Task 8 (Chen et al., 2022)
- **Agglomerative Clustering**: Reciprocal nearest neighbor agglomerative algorithm of RAC (Sumengen et al., 2021)

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Less, but Better: Efficient Multilingual Expansion for LLMs via Layer-wise Mixture-of-Experts](less_but_better_efficient_multilingual_expansion.md)
- [\[ACL 2025\] M2rc-Eval: Massively Multilingual Repository-level Code Completion Evaluation](m2rc-eval_massively_multilingual_repository-level_code_completion_evaluation.md)
- [\[ACL 2025\] Context Augmented Token-Level Post-Editing for Human Interpreting](context_augmented_token-level_post-editing_for_human_interpreting.md)
- [\[ACL 2025\] THOR-MoE: Hierarchical Task-Guided and Context-Responsive Routing for Neural Machine Translation](thor-moe_hierarchical_task-guided_and_context-responsive_routing_for_neural_mach.md)
- [\[ACL 2025\] Memorization Inheritance in Sequence-Level Knowledge Distillation for Neural Machine Translation](memorization_inheritance_seqkd.md)

</div>

<!-- RELATED:END -->
