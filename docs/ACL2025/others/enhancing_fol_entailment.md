---
title: >-
  [Paper Note] Enhancing Transformers for Generalizable First-Order Logical Entailment
description: >-
  [ACL 2025][First-order logical entailment] A systematic study on the generalizable reasoning capability of Transformers in first-order logical entailment tasks, revealing the impacts of query syntax, token embeddings, and Transformer architectures (especially position embeddings), and proposing TEGA (Transformer Encoder with Guided Attention) to significantly improve logical reasoning performance under relative position encoding settings.
tags:
  - "ACL 2025"
  - "First-order logical entailment"
  - "Transformer"
  - "Knowledge graph query answering"
  - "OOD generalization"
  - "Position embeddings"
date: 2026-05-08
content_hash: 7fb7e5f69301318a
---

# Enhancing Transformers for Generalizable First-Order Logical Entailment

**Conference**: ACL 2025  
**arXiv**: [2501.00759](https://arxiv.org/abs/2501.00759)  
**Code**: [https://github.com/HKUST-KnowComp/TEGA](https://github.com/HKUST-KnowComp/TEGA)  
**Area**: Others  
**Keywords**: First-order logical entailment, Transformer, Knowledge graph query answering, OOD generalization, Position embeddings  

## TL;DR

A systematic study on the generalizable reasoning capability of Transformers in first-order logical entailment tasks, revealing the impacts of query syntax, token embeddings, and Transformer architectures (especially position embeddings), and proposing TEGA (Transformer Encoder with Guided Attention) to significantly improve logical reasoning performance under relative position encoding settings.

## Background & Motivation

**Background**:
   - Transformers have demonstrated powerful reasoning capabilities in tasks such as arithmetic reasoning, symbolic reasoning, and theorem proving.
   - Knowledge Graph Query Answering (KGQA) is an important application of first-order logical entailment, with various dedicated methods available like BetaE, ConE, and CQD.
   - Prior work has studied the in-context reasoning ability of Transformers, but reasoning under parametric knowledge and OOD generalization remain under-explored.

**Limitations of Prior Work**:
   - Existing analyses are limited to in-context knowledge reasoning, failing to cover parametric knowledge scenarios.
   - There is a lack of research that explicitly links the two distribution shifts of OOD generalization (concept shift and covariate shift) with the KGQA task.
   - Existing benchmark datasets suffer from incomplete coverage of query types and characteristics (with at most 10 unseen query types).
   - Prior inductive bias designs are only effective under absolute position encoding (APE) and actually fail under the superior relative position encoding (RPE).

**Key Challenge**:
   - Existing research does not fully understand the design space of Transformers in first-order logical entailment.
   - RPE significantly outperforms APE, but prior architectural enhancements are tailored for APE and yield no benefit under RPE.

**Goal**:
   - Establish a comprehensive benchmark to evaluate the generalization capability of Transformers in first-order logical entailment.
   - Systematically study the impact of design choices such as query syntax, embeddings, and architecture on reasoning.
   - Propose effective inductive biases under the RPE setting.

**Key Insight**:
   - Format KGQA as an instance of first-order logical entailment, and decompose OOD generalization into the knowledge dimension (concept shift) and the query type dimension (covariate shift).
   - Determine optimal design choices through large-scale ablation studies, and then propose targeted architectural improvements.

**Core Idea**:
   - Systematically reveal the advantages of RPE in logical reasoning through experiments, and propose the TEGA architecture, which introduces logic-aware guided attention under RPE to enhance generalization capability.

## Method

### Overall Architecture

The study covers three core phases of KGQA modeling:
1. **Query Syntax** (input representation): Lisp-like vs. EFO syntax
2. **Token Embeddings**: Random initialization vs. pre-trained KG embeddings (TransE/DistMult/ComplEx)
3. **Transformer Architecture**: APE/DPE/RoPE/RPE + TEGA inductive bias

### Key Designs

1. **Formalization of Two Types of Distribution Shifts**:

    - Function: Decompose the OOD problem in KGQA into concept shift (unobserved knowledge $\mathcal{G}_o \to \mathcal{G}$) and covariate shift (unseen query types).
    - Mechanism: $P_{\text{train}}(Y|X) \cdot P_{\text{train}}(X) \neq P_{\text{test}}(Y|X) \cdot P_{\text{test}}(X)$
    - Design Motivation: Provide a clear theoretical framework for evaluating the generalization capability of Transformers.

2. **Comprehensive Benchmark Dataset**:

    - Function: Construct a benchmark containing 55 query types (23 seen + 32 unseen) covering all characteristics such as projection, intersection, union, negation, existential, multi-hop, and cyclic.
    - Mechanism: Sample on three knowledge graphs: FB15k, FB15k-237, and NELL995.
    - Design Motivation: Existing benchmarks have incomplete coverage (BetaE has only 4 unseen types, SQE has only 29).

3. **TEGA (Transformer Encoder with Guided Attention)**:

    - Function: Introduce inductive bias via logic-aware guided attention under the RPE setting.
    - Mechanism: Guide the self-attention pattern based on the logical relations between tokens in the query (such as belonging to the same atomic formula, sharing variables, etc.).
    - Design Motivation: Prior inductive biases (such as the structured encoding of SQE) are effective under APE but ineffective under RPE, requiring a specialized design for RPE.

### Loss & Training

- **Task**: Rank all entities and predict the answer set using embedding similarity.
- **Evaluation Metric**: MRR (Mean Reciprocal Rank)
- **Four-dimensional Evaluation**: ID(K)/OOD(K) × ID(Q)/OOD(Q)
- The knowledge graph is not directly accessible to the model during the training/testing phase; knowledge must be parameterized into the model.

## Key Experimental Results

### Main Results

MRR(%) results on FB15k:

| Method | ID(Q)/ID(K) | ID(Q)/OOD(K) | OOD(Q)/ID(K) | OOD(Q)/OOD(K) |
|------|------------|-------------|-------------|-------------- |
| BetaE | 26.9 | 18.5 | 22.4 | 13.5 |
| ConE | 35.5 | 22.0 | 27.2 | 15.6 |
| SQE-LSTM | 39.9 | 26.3 | 31.5 | 18.5 |
| Trans.+APE | 46.9 | 31.9 | 21.8 | 13.2 |
| Trans.+RPE | 48.1 | 32.3 | **35.4** | **21.5** |
| Trans.+RoPE | 50.1 | 32.7 | 34.6 | 20.8 |

- **Transformers comprehensively outperform dedicated methods**: Even a simple APE Transformer outperforms all baselines in the ID setting.
- **RPE leads by a large margin on OOD(Q)**: RPE's OOD(Q) is 13.6% higher than APE (35.4 vs 21.8), proving that relative position encoding is crucial for generalizing logical structures.

Query syntax experiments (FB15k-237):

| Setting | Lisp-like OOD(Q)/ID(K) | EFO OOD(Q)/ID(K) |
|------|----------------------|-----------------|
| APE | 10.0 | 10.4 |
| RPE | 22.1 | **35.4** |

- The combination of EFO syntax + RPE far outperforms Lisp-like + RPE in OOD generalization (35.4 vs 22.1).

Pre-trained embedding experiments:
- ComplEx and DistMult can improve performance, while TransE is worse than random initialization.
- Reason: Embedding learning during training is implicitly equivalent to KG-BERT-style link prediction.

### Key Findings

1. **RPE >> APE**: Relative position encoding has a massive advantage in generalizing to OOD query types.
2. **EFO syntax + RPE is optimal**: The parallel structure makes the distance of logical relationships between tokens more consistent, which RPE can learn more easily.
3. **APE is not robust to permutation**: After reversing the query permutation, APE performance plummets ($54.1 \to 27.8$), whereas RPE remains unchanged ($54.3 \to 54.5$).
4. **TEGA is effective under RPE**: It provides an effective inductive bias under the RPE setting.
5. **Transformers can perform logical entailment**: Under parametric knowledge, Transformers can execute first-order logical entailment.

## Highlights & Insights

- **Most thorough and extensive study**: 55 query types × 3 KGs × 4 PEs × 2 syntaxes × 4 embeddings.
- **Clear formalization of OOD generalization**: Naturally maps concept shift and covariate shift to KGQA.
- **Reveals the significant phenomenon of RPE > APE**: This finding offers insightful inspiration for the broader field of Transformer reasoning.
- **Discovers that existing inductive biases fail under RPE**: Points out an overlooked design blind spot.
- **Benchmark dataset contribution**: An evaluation framework with 32 unseen query types + two OOD dimensions.

## Limitations & Future Work

1. Only the KGQA scenario was studied; other forms of first-order logical entailment (such as natural language logical reasoning) are not covered.
2. The specific architectural details and performance improvements of TEGA are not sufficiently described in the paper.
3. The scale of the knowledge graphs is limited (FB15k, FB15k-237, and NELL995 are all small-to-medium-sized KGs).
4. No direct comparison with the logical reasoning capabilities of LLMs (such as GPT-4).
5. Limitations of parametric knowledge: The model needs to memorize all knowledge during training and cannot be dynamically updated.

## Related Work & Insights

- **BetaE (Ren & Leskovec, 2020)**: Probabilistic distribution-based embedding method; most commonly used dataset but has few query types.
- **SQE (Bai et al., 2023b)**: LSTM-based structured query encoding with 29 unseen types.
- **FIT (Yin et al., 2023b)**: Most comprehensive query feature coverage but has only 10 unseen types.
- Insight: The choice of position encoding has a decisive impact on structural generalization. RPE should be used as the default choice in reasoning tasks that require capturing structural relationships.

## Rating

| Dimension | Score (1-10) |
|------|-----------|
| Novelty | 7 |
| Technical Depth | 9 |
| Experimental Thoroughness | 9 |
| Writing Quality | 8 |
| Value | 7 |
| **Overall Score** | **8.0** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SpatialFormer: Towards Generalizable Vision Transformers with Explicit Spatial Understanding](../../ECCV2024/others/spatialformer_towards_generalizable_vision_transformers_with_explicit_spatial_un.md)
- [\[ICML 2025\] GPU-friendly and Linearly Convergent First-order Methods for Certifying Optimal $k$-sparse GLMs](../../ICML2025/others/gpu-friendly_and_linearly_convergent_first-order_methods_for_certifying_optimal_.md)
- [\[ACL 2025\] RoToR: Towards More Reliable Responses for Order-Invariant Inputs](rotor_towards_more_reliable_responses_for_order-invariant_inputs.md)
- [\[ACL 2025\] Implicit Reasoning in Transformers is Reasoning through Shortcuts](implicit_reasoning_in_transformers_is_reasoning_through_shortcuts.md)
- [\[AAAI 2026\] Tractable Weighted First-Order Model Counting with Bounded Treewidth Binary Evidence](../../AAAI2026/others/tractable_weighted_first-order_model_counting_with_bounded_treewidth_binary_evid.md)

</div>

<!-- RELATED:END -->
