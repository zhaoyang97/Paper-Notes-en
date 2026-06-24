---
title: >-
  [Paper Note] Predicate-Conditional Conformalized Answer Sets for Knowledge Graph Embeddings
description: >-
  [ACL 2025 (Findings)][Graph Learning][knowledge graph embedding] This paper proposes CondKGCP, a predicate-conditional conformal prediction method for uncertainty quantification in knowledge graph embeddings. By merging similar predicates to expand the calibration set and employing dual calibration (score + rank) to reduce prediction set size, it outputs tighter answer sets while guaranteeing predicate-level conditional coverage, outperforming five baselines on multiple KGE b…
tags:
  - "ACL 2025 (Findings)"
  - "Graph Learning"
  - "knowledge graph embedding"
  - "conformal prediction"
  - "conditional coverage"
  - "uncertainty quantification"
  - "link prediction"
date: 2026-05-08
content_hash: 37a3e02ea85fc91f
---

# Predicate-Conditional Conformalized Answer Sets for Knowledge Graph Embeddings

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2505.16877](https://arxiv.org/abs/2505.16877)  
**Code**: Yes  
**Area**: Graph Learning/Knowledge Graph  
**Keywords**: knowledge graph embedding, conformal prediction, conditional coverage, uncertainty quantification, link prediction

## TL;DR
This paper proposes CondKGCP, a predicate-conditional conformal prediction method for uncertainty quantification in knowledge graph embeddings. By merging similar predicates to expand the calibration set and employing dual calibration (score + rank) to reduce prediction set size, it outputs tighter answer sets while guaranteeing predicate-level conditional coverage, outperforming five baselines on multiple KGE benchmarks.

## Background & Motivation

**Background**: KGE methods (such as TransE and RotatE) encode entities and relations in a vector space for link prediction, but lack uncertainty quantification for predictions. Zhu et al. (2025) proposed KGCP to apply conformal prediction to KGEs to generate prediction sets with guaranteed coverage of the true answers.

**Limitations of Prior Work**: KGCP only provides marginal coverage guarantees (averaged over all queries), but uncertainty varies significantly across different predicates—e.g., "contraindicated_for" in a medical KG requires stricter guarantees than "has_symptom". Marginal coverage can cause the coverage rate of certain predicates to fall far below the target.

**Key Challenge**: Predicate-level conditional coverage requires performing conformal prediction independently for each predicate (Mondrian CP). However, most predicates in a KG have very few triplets, resulting in calibration sets that are too small, leading to unstable thresholds and excessively large prediction sets or coverage failures.

**Goal**: To achieve predicate-conditional coverage guarantees in KGEs while maintaining compact prediction sets.

**Key Insight**: Merging predicates with similar vector representations to enlarge the calibration set and introducing rank information for dual calibration.

**Core Idea**: CondKGCP = Predicate Merging (to increase calibration data) + Dual Calibration (score threshold $\cap$ rank threshold), backed by theoretical guarantees and experimental validation.

## Method

### Overall Architecture
Given a KGE model $M_\theta$, a query $q = \langle h, r, ? \rangle$, and a target coverage rate $1-\epsilon$. Output: A compact set of entities $\hat{C}(q)$ containing the true answer.

### Key Designs

1. **Predicate Merging**

    - **Function**: Merges predicates with sparse calibration data into the partition of the most semantically similar predicate with sufficient data.
    - **Mechanism**: Algorithm 1 divides the predicate set $R$ into "sufficiently-sampled" ($|\mathcal{T}_{cal}[\{r\}]| \geq \phi$) and "sparsely-sampled" groups. Sparsely-sampled predicates are merged with their nearest sufficiently-sampled neighbor based on Manhattan distance.
    - **Design Motivation**: Predicates with similar vector representations exhibit similar nonconformity score distributions, making calibration more stable after merging.
    - **Theory**: After merging, the calibration set $\mathcal{T}_g$ for each partition $g$ is large enough to determine reliable thresholds.

2. **Dual Calibration**

    - **Function**: Constrains the prediction set using both score and rank thresholds simultaneously.
    - **Mechanism**: $\hat{C}_{CondKGCP}(q) = \{e \in E_q[S \leq \hat{s}_{\epsilon'}] : \text{rank}(q,e) \leq \hat{k}(g)\}$
        - **Score Calibration**: Identify the $\epsilon'$-quantile of the nonconformity scores as the threshold.
        - **Rank Calibration**: Find the minimum $k$ such that $P(\text{rank} > k | \text{pred} \in g) < \epsilon$.
    - **Design Motivation**: Score-only calibration produces overly large prediction sets at the subgroup level; incorporating rank constraints excludes noisy entities with high scores but low ranks.
    - **Error Rate Allocation**: $\epsilon = \epsilon_{rank} + \epsilon_{score}$, guaranteeing the total coverage rate.

3. **Theoretical Guarantees**

    - Theorem 2: The conditional coverage probability of CondKGCP is tightly bounded around $1-\epsilon$ (with an upper/lower bound difference of $O(1/|\mathcal{T}_g|)$).
    - Theorem 3: Under certain conditions, the size of the prediction set generated by dual calibration is strictly smaller than that of score-only calibration.

## Key Experimental Results

### Conditional Coverage vs. Prediction Set Size (FB15k-237, $\epsilon=0.1$)

| Method | Mean Conditional Coverage | Coverage Std↓ | Mean Prediction Set Size↓ |
|------|-------------|------------|--------------|
| KGCP (Marginal) | 0.90 | 0.15 | 245 |
| MCP (Mondrian) | 0.91 | 0.08 | 892 |
| ClusterCP | 0.90 | 0.09 | 456 |
| **CondKGCP** | **0.91** | **0.05** | **312** |

### Comparison across Datasets

| Dataset | CondKGCP Coverage | CondKGCP Set Size | Best Baseline Coverage | Best Baseline Set Size |
|--------|---------------|---------------|--------------------|--------------------|
| FB15k-237 | 0.91 | **312** | 0.91 | 456 |
| WN18RR | 0.90 | **89** | 0.90 | 178 |
| YAGO3-10 | 0.91 | **205** | 0.90 | 367 |

### Key Findings
- **Conditional coverage significantly outperforms marginal coverage**: The coverage standard deviation decreases from 0.15 (KGCP) to 0.05 (CondKGCP), indicating that the coverage rates for all predicates are close to the target.
- **Prediction set is 2-3x smaller than Mondrian CP**: Predicate merging effectively increases the volume of calibration data.
- **Dual calibration is critical**: Removing rank calibration increases the prediction set size by 40-60%.
- **Merging strategy outperforms random merging**: Merging based on vector similarity achieves a 50% lower coverage standard deviation compared to random grouping.
- **Robust to different KGE models**: Effective across TransE, RotatE, and ComplEx.

## Highlights & Insights
- **Key upgrade from "average coverage" to "conditional coverage"**: For high-risk applications like medical KGs, "average 90% coverage" is insufficient—every predicate must achieve 90% to be safe. CondKGCP fills this gap.
- **Elegant design of predicate merging**: Leveraging pre-existing KGE vector representations for similarity calculation avoids the need for auxiliary models, provided that the KGE vectors successfully encode the semantic similarities of predicates.
- **Dual calibration (score $\cap$ rank)**: Constraints on two independent dimensions are more effective than a single threshold; rank eliminates noisy entities that have high scores but are semantically irrelevant.
- **Advancing uncertainty quantification in the KG domain**: Extending conformal prediction from classification to KGE link prediction represents a valuable cross-domain transfer.

## Limitations & Future Work
- **i.i.d. Assumption**: Conformal prediction assumes that the calibration and test sets are identically and independently distributed; the temporal evolution of KGs might violate this.
- **Hyperparameter $\phi$ for Predicate Merging**: Requires manual selection of the minimum calibration set size threshold.
- **Limited to 1-hop link prediction**: Not yet extended to multi-hop queries (e.g., EPFO queries).
- **Computational Cost**: Requires separate calibration for each predicate partition, which incurs high overhead when the number of predicates is large.
- **Choice of Manhattan distance**: Other similarity metrics (such as cosine or Euclidean) might be more suitable for certain KGEs.

## Related Work & Insights
- **vs KGCP (Zhu et al., 2025)**: KGCP only provides marginal coverage, whereas CondKGCP scales this up to predicate-conditional coverage, which is more appropriate for high-risk scenarios.
- **vs Ding et al. (2024)**: While they applied subgroup conformal prediction in classification, CondKGCP transfers this idea to KGE link prediction.
- **vs Shi et al. (2024)**: They used rank information to reduce the prediction set in classification, whereas CondKGCP introduces rank calibration to KGEs.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of predicate merging and dual calibration is novel in KGEs, backed by deep theoretical guarantees.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 datasets $\times$ 3 KGE models $\times$ 5 baselines + ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous definitions, clear presentation of Algorithm 1, and natural progression from theory to experiments.
- Value: ⭐⭐⭐⭐ Represents a significant advancement in KGE uncertainty quantification, with direct implications for high-risk applications such as medicine.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Aitchison Embeddings for Learning Compositional Graph Representations](../../ICML2026/graph_learning/aitchison_embeddings_for_learning_compositional_graph_representations.md)
- [\[ICML 2025\] WILTing Trees: Interpreting the Distance Between MPNN Embeddings](../../ICML2025/graph_learning/wilting_trees_interpreting_the_distance_between_mpnn_embeddings.md)
- [\[ACL 2025\] Croppable Knowledge Graph Embedding](croppable_knowledge_graph_embedding.md)
- [\[ACL 2025\] A Mutual Information Perspective on Knowledge Graph Embedding](a_mutual_information_perspective_on_knowledge_graph_embedding.md)
- [\[ACL 2026\] ARK: Answer-Centric Retriever Tuning via KG-augmented Curriculum Learning](../../ACL2026/graph_learning/ark_answer-centric_retriever_tuning_via_kg-augmented_curriculum_learning.md)

</div>

<!-- RELATED:END -->
