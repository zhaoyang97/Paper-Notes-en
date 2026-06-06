---
title: >-
  [Paper Note] Hierarchical Retrieval: The Geometry and a Pretrain-Finetune Recipe
description: >-
  [NeurIPS 2025][Information Retrieval & RAG][hierarchical retrieval] This paper investigates the feasibility of Dual Encoders (DE) for Hierarchical Retrieval (HR)…
tags:
  - "NeurIPS 2025"
  - "Information Retrieval & RAG"
  - "hierarchical retrieval"
  - "dual encoder"
  - "embedding geometry"
  - "pretrain-finetune"
  - "long-distance retrieval"
date: 2026-05-08
content_hash: 9d5bef2d0725670b
---

# Hierarchical Retrieval: The Geometry and a Pretrain-Finetune Recipe

**Conference**: NeurIPS 2025
**arXiv**: [2509.16411](https://arxiv.org/abs/2509.16411)  
**Code**: To be confirmed  
**Area**: Information Retrieval
**Keywords**: hierarchical retrieval, dual encoder, embedding geometry, pretrain-finetune, long-distance retrieval

## TL;DR
This paper investigates the feasibility of Dual Encoders (DE) for Hierarchical Retrieval (HR), theoretically proving that embedding dimensionality need only grow linearly with hierarchy depth and logarithmically with document count. After identifying the "lost-in-the-long-distance" phenomenon, the paper proposes a pretrain-finetune strategy that improves long-distance recall from 19% to 76% on WordNet.

## Background & Motivation
Document collections in information retrieval often exhibit hierarchical structure. In online advertising keyword matching, for example, the query "kids sandals" must match the exact keyword "kids sandals" as well as more general ancestor keywords such as "sandals" and "footwear." This is **Hierarchical Retrieval (HR)**—retrieving all reachable ancestors of a query node within a DAG-described hierarchy.

**Dual Encoders (DE)** are the most popular retrieval architecture due to their simplicity and scalability, yet the properties of Euclidean geometry limit their expressive power:
- HR relevance is **asymmetric** ("kids sandals" → "sandals" matches, but not vice versa)
- A single document embedding must simultaneously satisfy conflicting distance requirements (needing to be both close to and far from certain documents for different queries)

This gives rise to two core questions:
- **Q1**: Do DE embeddings that solve HR exist?
- **Q2**: Can such embeddings be learned from training data?

## Method

### Overall Architecture
1. **Theoretical geometric analysis**: Proves that asymmetric DE can solve HR with embedding dimensionality $d = O(s \cdot \log m)$
2. **Empirical validation**: Verifies on synthetic trees and real hierarchies that learned DE embeddings can indeed solve HR
3. **Problem identification**: Reveals the "lost-in-the-long-distance" phenomenon—documents farther apart in the hierarchy are harder to retrieve
4. **Proposed solution**: A pretrain-finetune strategy to improve long-distance retrieval quality

### Key Designs

#### 1. Geometric Feasibility of HR (Theorem 3.1)

**Problem definition**: Given DAG $G=(V,E)$, document set $D=\{x_j\}$, and query set $Q=\{q_i\}$, the HR objective is to retrieve all nodes reachable from $E(q_i)$ for each $q_i$.

**Construction algorithm (Algorithm 1)**:
- Sample document embeddings $\hat{x}_j \in \mathbb{R}^d$ i.i.d. from a Gaussian distribution
- Normalize to obtain $x_j = \hat{x}_j / \|\hat{x}_j\|$
- Query embedding is the normalized sum of matched document embeddings: $q_i = \left(\sum_{j \in S(q_i)} \hat{x}_j\right) / \left\|\sum \hat{x}_j\right\|$

**Theorem 3.1**: For the HR problem, if hierarchy $G$ satisfies $|S(q_i)| \leq s$ (at most $s$ matched documents per query), then there exists dimensionality:

$$d = O\!\left(\max\!\left\{s \log m,\; \frac{1}{\epsilon^2} \log m\right\}\right)$$

such that a threshold $r$ and embedding set exist satisfying:
- Matched pairs: $\langle q_i, x_j \rangle \geq r + \epsilon$
- Unmatched pairs: $\langle q_i, x_j \rangle \leq r - \epsilon$

For a perfect tree with $W$ children per internal node and height $H$, $s=H$ and $m=O(W^{H-1})$, yielding $d=O(H^2 \cdot \log W)$.

#### 2. Lost-in-the-Long-Distance Phenomenon

Training a DE with $d=3$ on a perfect tree with $H=4$ and $W=5$:
- Distance-0 matched pairs (self): recall near 100%
- Distance-1 matched pairs (parent): recall drops significantly
- Distance-2 matched pairs (grandparent): recall drops further

**Resampling fails**: Increasing the sampling ratio of long-distance pairs (heavy-tail sampling) improves long-distance recall but severely degrades short-distance recall. Adjusting the mixing ratio $p$ cannot satisfy both objectives simultaneously.

#### 3. Pretrain-Finetune Strategy

The core idea is straightforward:
1. **Pretraining phase**: Train the DE on a standard training set containing matched pairs at all distances
2. **Finetuning phase**: Finetune exclusively on long-distance data

Key findings:
- During finetuning, recall at distances 1 and 2 rapidly approaches 100%
- Meanwhile, recall at distance 0 is maintained near 100%
- Overall recall reaches 97%, far exceeding standard training (66%) and resampling (70%)
- Prolonged finetuning degrades distance-0 quality, which can be avoided via early stopping

### Loss & Training
The DE is trained with softmax cross-entropy loss:

$$L(\theta_q, \theta_x) = \frac{1}{N} \sum_{k=1}^{N} CE\!\left(\sigma\!\left(\frac{D(\theta_x)^T \cdot f_q(q^{(k)}, \theta_q)}{T}\right), \mathbf{1}_k\right)$$

where $T=20$ is a fixed temperature. During finetuning:
- Learning rate is reduced to $1/1000$ of pretraining
- Temperature is raised from 20 to 500
- The best checkpoint is selected via a validation set

## Key Experimental Results

### Main Results
**WordNet hierarchical retrieval** (82,115 noun synsets, distances $\leq 8$):

| Method | Dist-0 | Dist-1 | Dist-2 | Dist-3 | Dist-4 | Dist-5 | Dist-6 | Dist-7 | Dist-8 | Min | Overall |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|-----|---------|
| **$d$=32, Standard** | 100.0 | 90.8 | 79.2 | 62.3 | 46.4 | 31.8 | 20.1 | 14.8 | 8.4 | 8.4 | 61.8 |
| **$d$=32, Pretrain+Finetune** | 100.0 | 77.3 | 76.5 | 80.4 | 83.5 | 84.2 | 84.3 | 80.1 | 67.3 | **67.3** | **87.3** |
| **$d$=64, Standard** | 100.0 | 93.9 | 86.9 | 76.8 | 60.2 | 46.8 | 36.1 | 28.8 | 19.4 | 19.4 | 71.4 |
| **$d$=64, Pretrain+Finetune** | 100.0 | 90.8 | 91.6 | 92.7 | 92.6 | 91.8 | 90.9 | 87.3 | 75.7 | **75.7** | **92.3** |

At $d=64$, pretrain+finetune improves overall recall from 71.4% to **92.3%**, and worst-distance recall (distance 8) from 19.4% to **75.7%**.

**Semantic similarity evaluation on HyperLex** (5-dimensional embeddings):

| Method | Spearman $\rho$ |
|--------|----------------|
| OrderEmb | 0.195 |
| WN-Basic | 0.240 |
| WN-Euclidean | 0.389 |
| Standard Training | 0.350 |
| **Pretrain+Finetune** | **0.415** |

### Ablation Study
**Method comparison on synthetic tree ($H=4$, $W=5$, $d=3$)**:

1. **Standard sampling**: Distance-0 perfect; distance-1/2 poor; overall 66%
2. **Resampling ($p=0.03$)**: Distance-1/2 improved but distance-0 collapses to 2.8%; overall 40.7%
3. **Pretrain+Finetune**: All distances near 100%; overall **97%**

**Relationship between embedding dimensionality and hierarchy parameters**:
- Varying $H$ with fixed $W=2$: log-log fit slope 2.29 (theoretical prediction: 2, i.e., $d \propto H^2$)
- Varying $W$ with fixed $H=4$: fit slope 16.5 in $\log(W)$ space (theoretical prediction: 16)
- Learned embeddings require smaller $d$ than manually constructed ones

### Key Findings
1. **Lost-in-the-long-distance is systematic**: It appears consistently not only on synthetic trees but also on the real WordNet hierarchy
2. **Simple resampling cannot resolve it**: There is a fundamental trade-off between short-distance and long-distance quality
3. **Pretrain-finetune decouples the two objectives**: Learning near-distance representations first, then adapting for long-distance, avoids mutual interference
4. **Exact distance labels are unnecessary**: Only a coarse "near" vs. "far" partition is needed to apply the strategy

## Highlights & Insights
1. **Seamless integration of theory and practice**: The paper first proves feasibility, then identifies problems empirically, and finally proposes a solution
2. The **lost-in-the-long-distance phenomenon** is a novel and practically significant finding with broad implications for retrieval system design
3. The **pretrain-finetune strategy is remarkably simple** yet highly effective—requiring no modifications to model architecture or loss function
4. The theoretical bound $d = O(s \cdot \log m)$ is logarithmic in $m$, implying that low-dimensional embeddings suffice for large-scale document collections
5. The comparison with the multi-label classification bound of Guo et al. (2019) is clearly articulated; this paper generalizes to arbitrary margin $2\epsilon$

## Limitations & Future Work
1. **Constructing long-distance datasets**: When hierarchies are unavailable in practice, obtaining long-distance pairs requires domain knowledge
2. **Early stopping required during finetuning**: Prolonged finetuning degrades short-distance quality
3. **Only lookup-table DE considered**: Behavioral differences in Transformer-based encoders are not explored
4. **Single softmax loss**: Contrastive learning and other retrieval losses are not investigated
5. **DAG structure assumption**: Real-world retrieval hierarchies may be more complex (e.g., multi-label, partial ordering)
6. **Preliminary ESCI experiments**: Only two levels—Exact vs. Substitute—are validated on the shopping dataset

## Related Work & Insights
- **Graph Embedding**: Traditional approaches (Poincaré embeddings, Box Embeddings) can model hierarchies but are not suitable for large-scale retrieval
- **Ou et al. (2016)**: Uses asymmetric embeddings but requires known graph structure
- **Pretrain-finetune paradigm**: This paper adapts it to the long-distance problem in retrieval
- Insights: (1) Distance-sensitive quality analysis is essential in retrieval systems; (2) Staged training strategies have broad applicability in multi-objective optimization

## Rating
- Novelty: ⭐⭐⭐⭐ (Formalization of HR problem + lost-in-the-long-distance discovery are novel; pretrain-finetune strategy is relatively straightforward)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Three-tier validation: synthetic + WordNet + ESCI, with sufficient ablations)
- Writing Quality: ⭐⭐⭐⭐⭐ (Theoretical proofs are clear, figures are intuitive, narrative is well-structured)
- Value: ⭐⭐⭐⭐ (Offers practical guidance for the retrieval community with solid theoretical contributions)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] HiFi-RAG: Hierarchical Content Filtering and Two-Pass Generation for Open-Domain RAG](hifi-rag_hierarchical_content_filtering_and_two-pass_generation_for_open-domain_.md)
- [\[ICLR 2026\] Hierarchical Concept-based Interpretable Models](../../ICLR2026/information_retrieval/hierarchical_concept-based_interpretable_models.md)
- [\[ICML 2026\] Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation](../../ICML2026/information_retrieval/hierarchical_abstract_tree_for_cross-document_retrieval-augmented_generation.md)
- [\[ACL 2026\] Why These Documents? Explainable Generative Retrieval with Hierarchical Category Paths](../../ACL2026/information_retrieval/why_these_documents_explainable_generative_retrieval_with_hierarchical_category_.md)
- [\[ACL 2026\] FinRAG-12B: A Production-Validated Recipe for Grounded Question Answering in Banking](../../ACL2026/information_retrieval/finrag-12b_a_production-validated_recipe_for_grounded_question_answering_in_bank.md)

</div>

<!-- RELATED:END -->
