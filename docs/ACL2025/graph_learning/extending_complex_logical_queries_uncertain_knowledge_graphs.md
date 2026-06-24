---
title: >-
  [Paper Note] Extending Complex Logical Queries on Uncertain Knowledge Graphs
description: >-
  [Graph Learning] This paper proposes a formal framework of "soft queries" to extend complex logical queries to uncertain knowledge graphs containing confidence values, and designs the SRC method combining forward reasoning and backward calibration to answer soft queries efficiently, with theoretical proofs that errors do not catastrophically cascade.
tags:
  - "Graph Learning"
date: 2026-05-08
content_hash: ecdf93cfce678341
---

# Extending Complex Logical Queries on Uncertain Knowledge Graphs

| Conference | arXiv | Code | Area | Keywords |
|------|-------|------|------|--------|
| ACL 2025 | 2403.01508 | - | Graph Learning / Knowledge Graph Reasoning | Uncertain Knowledge Graphs, Complex Logical Queries, Soft Queries, Neuro-Symbolic Methods, Confidence Calibration |

## TL;DR

This paper proposes a formal framework of "soft queries" to extend complex logical queries to uncertain knowledge graphs containing confidence values, and designs the SRC method combining forward reasoning and backward calibration to answer soft queries efficiently, with theoretical proofs that errors do not catastrophically cascade.

## Background & Motivation

**Research Problem:** How to perform complex logical reasoning on incomplete, uncertain knowledge graphs (Uncertain KGs) with confidence values?

**Limitations of Prior Work:** 
- Existing complex query answering (CQA) methods are based on Boolean logic and cannot handle the uncertainty of knowledge (e.g., "the confidence of someone knowing Python is 0.8").
- Probabilistic databases assume uniform uncertainty for all unobserved facts, failing to leverage the generalization ability of machine learning.
- First-order logic is insufficient to describe preferences and necessity in practical reasoning (e.g., leadership is more important for a PI than for a junior developer in recruitment).

**Core Motivation:** Real-world knowledge is inherently uncertain (e.g., O*NET, STRING, ConceptNet), necessitating a new query language and reasoning method to handle both knowledge uncertainty and incompleteness simultaneously.

## Method

### Overall Architecture

The **Soft Queries on Uncertain KG (SQUK)** setting is proposed, comprising three levels:
1. **Soft Query Language:** Introduces α-necessity and β-importance parameters on top of existential first-order logic.
2. **SRC Forward Reasoning:** A symbolic reasoning algorithm based on UKGE confidence functions.
3. **Backward Calibration:** Reduces reasoning errors through Debiasing and Learning strategies.

### Key Designs

**Soft Query Definition:** In each soft atomic formula $(h, r, t, α, β)$, $α$ acts as the necessity threshold (filtering out substandard candidates), and $β$ acts as the importance weight (adjusting the relative importance of different conditions). The semantics are based on the semiring $(+, \max, -\infty)$, where conjunction corresponds to summation and disjunction corresponds to taking the maximum.

**SRC Forward Reasoning Algorithm:** 
- Maintains a state vector $C_z \in \mathbb{R}^{|E|}$ for each variable node.
- Progressively prunes query graph edges via two equivalent transformations: RemoveConstNode ($O(|E|)$) and RemoveLeafNode ($O(|E|^2)$).
- The final state vector of the free variable is the utility vector.

**Error Analysis (Core Theoretical Contribution):**
- Theorem 1: The reasoning error of a single soft atomic query is bounded by the UKGE error bound function.
- Theorem 2: The cumulative error of conjunctive queries grows at most linearly, **without catastrophic cascading errors**.

### Loss & Training

The calibration Learning strategy calibrates confidence values via a learnable affine transformation $\hat{P}_c(s,r,o) = \hat{P}(s,r,o)(1+\rho_\theta) + \lambda_\theta$, optimizing the MSE loss:

$$\mathcal{L} = \sum_{u(s)>0} (U(\phi(s), \hat{P}_c) - U(\phi(s), \mathcal{P}))^2$$

## Experiments

### Main Results

| Uncertain KG | Method | τ AVG | ρ AVG | MAP AVG | NDCG |
|----------|------|-------|-------|---------|------|
| CN15k | LogicE+NE | 4.8 | 5.8 | 7.0 | 11.2 |
| CN15k | ConE+NE | 8.1 | 10.0 | 7.7 | 13.2 |
| CN15k | SRC | 5.9 | 8.9 | 9.2 | 15.5 |
| CN15k | **SRC(D)** | **10.5** | - | - | - |
| CN15k | **SRC(D+L)** | **Best** | **Best** | **Best** | **Best** |

> SRC(D+L) consistently outperforms query embedding and neuro-symbolic baselines across 12 query types on three datasets (CN15k, PPI5k, O*NET20k).

### Ablation Study

| Ablation Item | Effect |
|--------|------|
| α=0 (no necessity filtering) | Error integration range expands to [0,1], leading to performance degradation |
| Debiasing Strategy | Compensates for the zero-bias of UKGE by shifting the α threshold, significantly improving performance |
| Learning Strategy | Training the calibration transformation only when α=0 simplifies training but yields better generalization |
| Comparison with LLMs | SRC outperforms commercial LLMs on soft queries described in natural language |

### Key Findings

- The reasoning complexity of SRC is identical to state-of-the-art symbolic methods ($O(n_e|E|^2 + n_r|E|)$), and leveraging sparsity can yield a further speedup of approximately 97%.
- The backward calibration strategy is key to performance improvement; forward reasoning alone is insufficient to cope with the prediction bias of UKGE.
- Soft queries can model preferences and necessity distinctions that traditional FOL queries cannot express, showing clear application value in scenarios such as recruitment and recommendation.
- Even advanced LLMs struggle to rival dedicated neuro-symbolic methods in structured knowledge reasoning.

## Highlights & Insights

- Formally defines the "soft query" language, elegantly integrating uncertainty and preferences into logical queries.
- Rigorous error bound analysis (Theorems 1 & 2) provides theoretical guarantees for algorithmic reliability.
- Constructs a comprehensive SQUK benchmark dataset covering three domains (commonsense, biology, employment) and various query structures.

## Limitations & Future Work

- The prediction error of UKGE (MAE is around 0.2 on CN15k) remains the bottleneck for reasoning accuracy.
- Experiments are restricted to three medium-sized uncertain KGs; scalability to ultra-large-scale graphs has not been verified.
- The α and β parameters for soft requirements need to be manually set or heuristically chosen based on statistical quantiles.
- Currently, only basic query types (1P, 2P, etc.) are used during training; generalization to complex query structures requires further validation.

## Related Work & Insights

- **Complex Query Answering:** Query embedding and symbolic methods on deterministic KGs, such as BetaE (Ren & Leskovec, 2020), FIT (Yin et al., 2024), and QTO (Bai et al., 2023).
- **Uncertain Knowledge Graph Embedding:** UKGE (Chen et al., 2019) and its subsequent improvements for learning confidence predictions.
- **Probabilistic Databases:** Traditional probabilistic databases (Suciu et al., 2022) and open-world probabilistic databases (Ceylan et al., 2021).
- **Soft Constraint Satisfaction:** Possibilistic CSP (Schiex, 1992) and weighted CSP (Bistarelli et al., 1999) provide inspiration for the α and β designs in this study.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Practicality | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can LLMs Evaluate Complex Attribution in QA? Automatic Benchmarking using Knowledge Graphs](paper_2401_14640.md)
- [\[ICLR 2026\] Controllable Logical Hypothesis Generation for Abductive Reasoning in Knowledge Graphs](../../ICLR2026/graph_learning/controllable_logical_hypothesis_generation_for_abductive_reasoning_in_knowledge_.md)
- [\[NeurIPS 2025\] Uncertain Knowledge Graph Completion via Semi-Supervised Confidence Distribution Learning](../../NeurIPS2025/graph_learning/uncertain_knowledge_graph_completion_via_semi-supervised_confidence_distribution.md)
- [\[ICML 2025\] Is Complex Query Answering Really Complex?](../../ICML2025/graph_learning/is_complex_query_answering_really_complex.md)
- [\[ACL 2025\] Cross-Document Contextual Coreference Resolution in Knowledge Graphs](cross-document_contextual_coreference_resolution_in_knowledge_graphs.md)

</div>

<!-- RELATED:END -->
