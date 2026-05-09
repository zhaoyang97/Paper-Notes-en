---
title: >-
  [Paper Note] RFKG-CoT: Relation-Driven Adaptive Hop-count Selection and Few-Shot Path Guidance for Knowledge-Aware QA
description: >-
  [AAAI 2026][Graph Learning][Knowledge Graph Question Answering] This paper proposes RFKG-CoT, which enhances LLM reasoning over knowledge graphs via two components: relation-driven adaptive hop-count selection (dynamically adjusting reasoning steps using KG relation activation masks) and few-shot path guidance (in-context examples in a Question-Paths-Answer format). Evaluated on 4 KGQA benchmarks, the method achieves significant improvements — GPT-4 reaches 91.5% (+6.6pp) on WebQSP, and Llama2-7B gains up to +14.7pp.
tags:
  - AAAI 2026
  - Graph Learning
  - Knowledge Graph Question Answering
  - Chain-of-Thought
  - Relation-Aware
  - Adaptive Hop-count
  - Few-Shot Guidance
date: 2026-05-08
content_hash: 014267add6bdc44d
---

# RFKG-CoT: Relation-Driven Adaptive Hop-count Selection and Few-Shot Path Guidance for Knowledge-Aware QA

**Conference**: AAAI 2026
**arXiv**: [2512.15219](https://arxiv.org/abs/2512.15219)
**Code**: N/A
**Area**: Graph Learning
**Keywords**: Knowledge Graph Question Answering, Chain-of-Thought, Relation-Aware, Adaptive Hop-count, Few-Shot Guidance

## TL;DR
This paper proposes RFKG-CoT, which enhances LLM reasoning over knowledge graphs via two components: relation-driven adaptive hop-count selection (dynamically adjusting reasoning steps using KG relation activation masks) and few-shot path guidance (in-context examples in a Question-Paths-Answer format). Evaluated on 4 KGQA benchmarks, the method achieves significant improvements — GPT-4 reaches 91.5% (+6.6pp) on WebQSP, and Llama2-7B gains up to +14.7pp.

## Background & Motivation

**Background**: Methods such as KG-CoT integrate knowledge graph paths into LLM reasoning to mitigate hallucination, but bottlenecks remain in both path selection and path utilization.

**Limitations of Prior Work**:
- **Rigid hop-count selection**: Existing methods select hop counts based solely on question features, ignoring KG relational structure. For example, "Who is Justin Bieber's brother?" requires only 1 hop via a direct "brother" relation, but an indirect "father-son" chain requires multiple hops.
- **Insufficient path utilization**: KG paths are concatenated directly into LLM prompts without guidance on how to interpret and use them.

**Key Challenge**: The quality of KG paths depends on hop-count selection, which should be jointly determined by both the question and the KG relational structure rather than fixed uniformly.

**Key Insight**: Use relation activation masks to capture KG relational semantics for dynamic hop-count selection; use few-shot "Think" templates to teach LLMs how to extract answers from paths.

**Core Idea**: Relation masks make hop-count selection structure-aware + few-shot path guidance teaches LLMs how to leverage retrieved paths.

## Method

### Overall Architecture
Initialize topic entities → compute relation scores via MLP → dynamically select hop counts using relation activation masks → generate KG paths → submit to LLM reasoning with few-shot guidance (Question-Paths-Think-Answer template).

### Key Designs

1. **Relation-Driven Adaptive Hop-count Selector**:

   - **Function**: Dynamically selects the number of reasoning steps based on KG relation activation patterns rather than question features alone.
   - **Mechanism**: Records which relations are activated at each reasoning step via a relation activation mask, and uses this to determine whether additional hops are needed. Selects 1 hop when a direct relation exists; automatically increases hop count for indirect chains.
   - **Design Motivation**: The same question may require different hop counts under different KG topologies; relation masks capture this structural information.

2. **Few-Shot Path Guidance**:

   - **Function**: Uses structured in-context examples to teach LLMs how to interpret and utilize KG paths.
   - **Mechanism**: Each example contains: a query, serialized KG paths (Entity→Relation→Entity), a symbolic "Think" template mapping path elements to answer constraints, and an explicit answer format. The optimal number of examples is $E=3$.
   - **Design Motivation**: LLMs receiving KG paths lack guidance on how to translate path information into reasoning steps; the "Think" template serves as a bridge.

### Loss & Training
- The graph reasoning module learns relation scores via an MLP, optimized on the training set.
- LLM inference is performed in a zero-shot/few-shot manner without fine-tuning.

## Key Experimental Results

### Main Results

| Dataset | LLM | RFKG-CoT | KG-CoT | Gain |
|--------|-----|----------|--------|------|
| WebQSP | GPT-4 | **91.5%** | 84.9% | +6.6pp |
| CompWebQ | GPT-4 | **65.1%** | 62.3% | +2.8pp |
| WebQuestions | GPT-4 | **78.2%** | 68.0% | +10.2pp |
| WebQSP | ChatGPT | **89.9%** | 82.1% | +7.8pp |
| WebQSP | Llama2-7B | **87.1%** | 72.4% | +14.7pp |

### Ablation Study

| Component | WebQSP (ChatGPT) | CompWebQ | Notes |
|------|-----------------|----------|------|
| KG-CoT baseline | 82.1% | 51.6% | No improvement |
| + Relation mask | 85.5% | 59.8% | +3.4/+8.2pp |
| + Few-shot guidance | 87.7% | 57.8% | +5.6/+6.2pp |
| Full RFKG-CoT | **89.9%** | **61.4%** | +7.8/+9.8pp |

### Key Findings
- **Complementary components**: The relation mask improves path quality and selection, while few-shot guidance improves path utilization; their combination exceeds the sum of individual contributions.
- **Smaller models benefit more**: Llama2-7B gains +14.7pp vs. +6.6pp for GPT-4, as smaller models have less parametric knowledge and rely more heavily on external paths.
- **Non-monotonic effect of few-shot count**: $E=3$ is optimal; $E=5$ degrades performance, likely due to the cognitive load on the transformer.

## Highlights & Insights
- The **relation activation mask** is an elegant design — encoding KG topological information as a binary mask to guide hop-count decisions offers greater flexibility than question classifiers.
- **Inverse scaling finding**: Smaller models benefit disproportionately more, suggesting that KG path guidance is most valuable when compensating for limited parametric knowledge.

## Limitations & Future Work
- The method has not been evaluated on state-of-the-art reasoning models (e.g., o1, DeepSeek-R1).
- Learning of relation masks depends on the coverage of relation types in the training data.
- The selection strategy for few-shot examples could be further optimized.

## Related Work & Insights
- **vs. KG-CoT**: Improvements are made at both critical stages — hop-count selection and path utilization.
- **vs. ToG**: ToG performs dynamic navigation over KGs but provides no path guidance; RFKG-CoT offers a more structured reasoning framework.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of relation masks and few-shot path guidance constitutes effective incremental innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 4 datasets, 3 LLMs, detailed ablations, and hyperparameter analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Method motivation and design logic are clearly articulated.
- **Value**: ⭐⭐⭐⭐ Offers practical improvements for KGQA; the substantial gains on smaller models have notable application value.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] PKD: Preference-driven Knowledge Distillation for Few-shot Node Classification](../../NeurIPS2025/graph_learning/preference-driven_knowledge_distillation_for_few-shot_node_classification.md)
- [\[AAAI 2026\] Commonality in Few: Few-Shot Multimodal Anomaly Detection via Hypergraph-Enhanced Memory](commonality_in_few_few-shot_multimodal_anomaly_detection_via_hypergraph-enhanced.md)
- [\[AAAI 2026\] MUG: Meta-path-aware Universal Heterogeneous Graph Pre-Training](mug_meta-path-aware_universal_heterogeneous_graph_pre-training.md)
- [\[AAAI 2026\] Enhancing Logical Expressiveness in GNNs via Path-Neighbor Aggregation](enhancing_logical_expressiveness_in_graph_neural_networks_via_path-neighbor_aggr.md)
- [\[AAAI 2026\] Adaptive Riemannian Graph Neural Networks](adaptive_riemannian_graph_neural_networks.md)

<!-- RELATED:END -->
