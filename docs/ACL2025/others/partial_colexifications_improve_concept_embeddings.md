---
title: >-
  [Paper Note] Partial Colexifications Improve Concept Embeddings
description: >-
  This work introduces partial colexification (affix/overlap colexification) into concept embedding training for the first time, consistently outperforming baselines that rely solely on full colexification across three tasks: semantic similarity modeling, semantic shift prediction, and word association prediction.
tags:

date: 2026-05-08
content_hash: 9c9cf3d838e5962d
---

# Partial Colexifications Improve Concept Embeddings

- **Conference**: ACL 2025
- **arXiv**: [2502.09743](https://arxiv.org/abs/2502.09743)
- **Code**: [GitHub](https://github.com/calc-project/concept-embeddings)
- **Area**: Computational Linguistics / Lexical Semantics
- **Keywords**: Concept Embeddings, Colexification, Partial Colexification, Graph Embeddings, Cross-lingual Semantics, Node2Vec, ProNE

## TL;DR

This work introduces partial colexification (affix/overlap colexification) into concept embedding training for the first time, consistently outperforming baselines that rely solely on full colexification across three tasks: semantic similarity modeling, semantic shift prediction, and word association prediction.

## Background & Motivation

- **Core Problem**: Concept embeddings provide language-agnostic semantic representations for cross-lingual NLP tasks. However, existing methods solely utilize full colexification (where the same word form expresses two meanings) to construct colexification networks, neglecting the semantic relationships embedded within word parts (such as shared affixes or stems).
- **Limitations of Prior Work**:
    - **Full Colexification Only**: Harvill et al. 2022 and Chen et al. 2023 rely purely on graph embeddings based on full colexification, missing implicit conceptual relations in morphological derivation or compounding (e.g., "tree" and "bark" are related via a shared root, but are rarely fully colexified).
    - **Noisy Automatically Constructed Colexification Graphs**: Liu et al. 2023 automatically infer colexification from parallel corpora, but the quality is inferior to hand-annotated data.
    - **Limited Coverage**: The number of colexifications in any single language is very small, requiring cross-lingual aggregation to construct a meaningful network.
- **Design Motivation**: By leveraging the partial colexification inference method (affix + overlap) proposed by List (2023), this work constructs richer colexification networks on the hand-annotated IDS dataset (329 languages) to train superior concept embeddings.

## Method

### Overall Architecture

Three types of colexification networks are inferred from the IDS dataset (329 languages, 1,310 concepts): full colexification (identical word forms), affix colexification (one word is a prefix/suffix of another), and overlap colexification (sharing a substring). Three graph embedding methods (SDNE, Node2Vec, and ProNE) are applied to learn 128-dimensional concept vectors, which are then combined across different embedding types using concatenation followed by PCA.

### Key Designs

1. **Three Colexification Networks**: Full colexification (1,246 nodes / 4,008 edges), affix colexification (1,308 nodes / 38,215 edges, which is directional but converted to an undirected graph), and overlap colexification (926 nodes / 12,974 edges). Edge weights are weighted by the number of language families, where colexifications spanning more language families receive higher weights.
2. **Embedding Combination Strategy**: Embeddings are trained separately on each network type, then concatenated and reduced back to 128 dimensions via PCA. Six combinations are evaluated: full, affix, overlap, full+affix, full+overlap, and full+affix+overlap.
3. **Three Evaluation Tasks**: (a) Semantic similarity modeling (Multi-SimLex, 538 pairs, Spearman correlation); (b) Semantic shift prediction (DatSemShift, 547 pairs, logistic regression binary classification); (c) Word association prediction (EAT, 780 edges, link prediction binary classification).

### Loss & Training

- **SDNE**: Autoencoder reconstruction loss + first-order and second-order proximity preservation loss
- **Node2Vec**: Random walk-based Skip-gram (Word2Vec) objective
- **ProNE**: Sparse matrix factorization + spectral propagation

## Experiments

### Main Results

**Task (a) Semantic Similarity (Spearman Correlation Coefficient)**:

| Method | full | affix | full+affix | full+affix+overlap |
|------|------|-------|-----------|-------------------|
| ProNE | 0.64 | 0.63 | **0.72** | 0.66 |
| Node2Vec | 0.64 | 0.58 | 0.69 | 0.66 |
| fastText-ZH (best) | 0.44 | — | — | — |

**Task (b) Semantic Shift Prediction (Accuracy)**:

| Method | full | full+affix | full+affix+overlap |
|------|------|-----------|-------------------|
| Node2Vec | 0.79 | **0.83** | 0.82 |
| ProNE | 0.78 | 0.82 | **0.83** |
| fastText-ET (best) | 0.82 | — | — |

**Task (c) Word Association Prediction (Accuracy)**:

| Method | full | full+affix | full+affix+overlap |
|------|------|-----------|-------------------|
| ProNE | 0.71 | 0.80 | **0.81** |
| Node2Vec | 0.71 | 0.78 | 0.79 |
| fastText-EN (best) | 0.87 | — | — |

### Ablation Study

| Colexification Type | Semantic Similarity | Semantic Shift Prediction | Word Association Prediction |
|-----------|-----------|------------|---------|
| full alone | 0.64 | 0.79 | 0.71 |
| +affix | **0.72 (+0.08)** | **0.83 (+0.04)** | 0.80 (+0.09) |
| +overlap | 0.62 (-0.02) | 0.80 (+0.01) | 0.77 (+0.06) |
| +affix+overlap | 0.66 (+0.02) | 0.83 (+0.04) | **0.81 (+0.10)** |

### Key Findings

1. **Affix colexification is the most valuable supplementary information**: full+affix significantly outperforms full alone on all three tasks, improving the correlation coefficient by 0.08 and accuracy by 4 to 9 percentage points.
2. **The value of overlap colexification varies by task**: It is detrimental in the semantic similarity task (-0.02) but beneficial in word association prediction (+0.06), potentially because overlap colexification captures more distant semantic associations.
3. **Concept embeddings vastly outperform word embeddings in semantic similarity**: ProNE full+affix (0.72) versus the best fastText baseline (0.44) indicates that cross-lingual colexification captures semantic similarity far better than monolingual distributional information.
4. **Word embeddings still hold an advantage in word association prediction**: fastText-EN (0.87) > ProNE best (0.81), likely because EAT is a monolingual English association dataset where distributional information from word embeddings is more direct.
5. **SDNE exhibits the poorest performance**: It underperforms Node2Vec and ProNE across all tasks, indicating it is not suitable for embedding colexification graphs.

## Highlights & Insights

- This study introduces partial colexification (affix + overlap) into concept embeddings for the first time and proposes a systematic evaluation framework.
- The approach relies on high-quality, hand-annotated cross-lingual data (329 languages from the IDS database), which is more reliable than automatically inferred colexifications.
- Concept embeddings substantially outperform traditional word embeddings in cross-lingual semantic similarity modeling.

## Limitations & Future Work

- The coverage is restricted to approximately 1,000 core concepts, limited by comparative lists of basic vocabulary.
- Isolated nodes in the graph (concepts without colexification) cannot be embedded.
- The underlying mechanism of overlap colexification remains unclear, necessitating further research.
- Assessment tasks are primarily based on English and a few high-resource languages, without directly verifying the utility for low-resource languages.

## Related Work & Insights

- **Concept Embeddings**: Harvill et al. 2022 (BabelNet graph embeddings), Chen et al. 2023 (Colex2Lang), Liu et al. 2023
- **Colexification Studies**: François 2008, List 2023 (partial colexification), CLICS database
- **Graph Embeddings**: Node2Vec (Grover & Leskovec 2016), ProNE (Zhang et al. 2019), SDNE (Wang et al. 2016)
- **Word Embeddings**: Word2Vec, fastText, GloVE

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Applying partial colexification to concept embeddings is a natural yet unexplored direction.
- **Utility**: ⭐⭐⭐ — The concept set is limited, and direct NLP applications must wait for coverage expansion.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three complementary evaluation tasks, multiple baselines, and a balanced negative sampling design.
- **Overall**: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Polaris: Coupled Orbital Polar Embeddings for Hierarchical Concept Learning](../../ICML2026/others/polaris_coupled_orbital_polar_embeddings_for_hierarchical_concept_learning.md)
- [\[ACL 2025\] MEXMA: Token-level Objectives Improve Sentence Representations](mexma_token-level_objectives_improve_sentence_representations.md)
- [\[ACL 2025\] Improve Rule Retrieval and Reasoning with Self-Induction and Relevance ReEstimate](improve_rule_retrieval_and_reasoning_with_self-induction_and_relevance_reestimat.md)
- [\[ACL 2025\] Capacity Matters: A Proof-of-Concept for Transformer Memorization on Real-World Data](capacity_matters_a_proof-of-concept_for_transformer_memorization_on_real-world_d.md)
- [\[ACL 2025\] GPT-4 as a Homework Tutor can Improve Student Engagement and Learning Outcomes](gpt-4_as_a_homework_tutor_can_improve_student_engagement_and_learning_outcomes.md)

</div>

<!-- RELATED:END -->
