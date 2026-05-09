---
title: >-
  [Paper Note] Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs
description: >-
  [NeurIPS 2025][Graph Learning][Knowledge Graph Question Answering] This paper proposes DP (Deliberation on Priors), a framework that leverages structural priors from knowledge graphs via progressive knowledge distillation to generate faithful relational paths, and validates reasoning reliability through a reasoning introspection strategy based on constraint priors, achieving new state-of-the-art performance on KGQA benchmarks.
tags:
  - NeurIPS 2025
  - Graph Learning
  - Knowledge Graph Question Answering
  - Trustworthy Reasoning
  - Structural Priors
  - Constrained Reasoning
  - Knowledge Distillation
date: 2026-05-08
content_hash: 05bdaab3fba8ed94
---

# Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs

**Conference**: NeurIPS 2025
**arXiv**: [2505.15210](https://arxiv.org/abs/2505.15210)
**Code**: [GitHub](https://github.com/mira-ai-lab/Deliberation-on-Priors)
**Area**: Graph Learning / KGQA
**Keywords**: Knowledge Graph Question Answering, Trustworthy Reasoning, Structural Priors, Constrained Reasoning, Knowledge Distillation

## TL;DR

This paper proposes DP (Deliberation on Priors), a framework that leverages structural priors from knowledge graphs via progressive knowledge distillation to generate faithful relational paths, and validates reasoning reliability through a reasoning introspection strategy based on constraint priors, achieving new state-of-the-art performance on KGQA benchmarks.

## Background & Motivation

Knowledge graph retrieval-augmented generation (KG-RAG) aims to supply LLMs with up-to-date external knowledge to reduce hallucinations. However, existing methods fail to fully exploit the prior knowledge embedded in knowledge graphs, manifesting in two aspects:

(1) **Structural information is underutilized**: Relational paths from topic entities to answer entities can enhance LLM awareness of KG structural patterns and improve reasoning faithfulness. (2) **Explicit and implicit constraints are neglected**: Constraints such as type constraints, multi-entity constraints, and temporal constraints can be used to filter candidate paths and guide backtracking, improving reasoning reliability.

Existing methods either retrieve relevant triples end-to-end before generating answers, or decompose questions step by step before retrieval, but neither adequately exploits these priors. This is especially problematic for complex questions (e.g., CWQ requires 4-hop reasoning), where the reasoning reliability of LLMs is severely insufficient.

## Method

### Overall Architecture

The DP framework consists of four core modules: **Distillation** → **Planning** → **Instantiation** → **Introspection**. It operates in two phases: an offline phase that distills structural priors into LLMs via SFT + KTO, and an online phase that achieves trustworthy reasoning through path generation, selection, instantiation, and constraint verification.

### Key Designs

**1. Progressive Knowledge Distillation: SFT + KTO**

- **Function**: Inject KG structural patterns into LLMs to enable generation of faithful relational paths.
- **Mechanism**: Weak supervision signals are first collected by extracting question-to-relational-path mappings from the training set: $\boldsymbol{\mathcal{P}}_w(q) = \text{ShortestPath}_{\text{Dijkstra}}(\mathcal{G}_k(e_s), e_s, e_t)$. SFT is then applied to maximize the conditional log-likelihood $\mathcal{L}_{\text{SFT}} = \sum_{t=1}^{T} \log P_\theta(r_t^* | r_{<t}^*, q, e_s)$. KTO (Kahneman-Tversky Optimization) is subsequently applied for further refinement by constructing positive/negative path pairs via three perturbations—path truncation, entity-path swapping, and relation deletion—to address severe positive-to-negative class imbalance (1:3 ratio).
- **Design Motivation**: SFT alone is insufficient for LLMs to distinguish semantically valid from invalid paths. KTO is better suited than DPO for handling class-imbalanced data, and its utility-maximization objective grounded in Kahneman-Tversky prospect theory is more robust.

**2. Reasoning Introspection Strategy: Constraint Extraction + Verification + Backtracking**

- **Function**: Ensure that the final reasoning paths satisfy the constraints of the question, improving answer reliability.
- **Mechanism**: Five categories of constraints are predefined (type, multi-entity, explicit temporal, implicit temporal, and ordinal). During online inference, an LLM first extracts constraints $\mathcal{C}(q)$ from the question, then verifies whether the instantiated reasoning path $\mathbb{P}$ satisfies them: $\mathcal{J}(q, e_s, \mathbb{P}) = 1$ when $\mathbb{P} \models \mathcal{C}(q)$. If not, a backtracking mechanism is triggered: the violated constraints are fed back, and path selection and verification are restarted.
- **Design Motivation**: Existing methods directly generate answers once a reasoning path is obtained, lacking any verification mechanism. Constraint priors provide executable checking conditions, and the backtracking mechanism reduces the negative impact of false-positive reasoning paths.

**3. One-to-Many Path Mapping**

- **Function**: Improve the coverage and diversity of generated paths.
- **Mechanism**: Unlike the one-to-one mapping in RoG, DP collects **all shortest paths** from topic entities to answer entities as weak supervision signals, forming a one-to-many question-to-path mapping. For questions with multiple topic entities, candidate paths are generated independently for each entity and then merged.
- **Design Motivation**: One-to-one mapping may miss valid alternative paths. The one-to-many mapping improves path generation F1 on WebQSP from 59.3% to 76.7% (a relative gain of 29.3%).

### Loss & Training

The path generator is built on LLaMA3.1-8B-Instruct with LoRA for efficient adaptation. SFT is trained for 2 epochs and KTO for 1 epoch. The KTO loss is: $\mathcal{L}_{\text{KTO}} = \mathbb{E}_{(x,y) \sim D}[\lambda_y - v(x,y)]$, where the value function $v(x,y)$ is computed separately for positive and negative samples. During online inference, path selection and constraint verification employ few-shot prompting.

## Key Experimental Results

### Main Results

| Method | Type | WebQSP H@1 | WebQSP F1 | CWQ H@1 | CWQ F1 |
|--------|------|-----------|-----------|---------|---------|
| RoG | SL | 80.8 | 70.8 | 57.8 | 56.2 |
| GNN-RAG | SL | 82.8 | 73.5 | 62.8 | 60.4 |
| DoG (GPT-4) | ICL | 65.4 | 55.6 | 41.0 | 46.4 |
| LightPROF | HL | 83.8 | - | 59.3 | - |
| **DP (GPT-4.1)** | HL | **86.7** | **80.1** | **75.8** | **69.4** |

### Ablation Study

| Setting | WebQSP H@1 | WebQSP F1 | CWQ H@1 | CWQ F1 |
|---------|-----------|-----------|---------|---------|
| DP (GPT-4.1) | **86.7** | **80.1** | **75.8** | **69.4** |
| GPT-4.1 (vanilla) | 71.0 | 54.6 | 53.0 | 48.9 |
| w/o KTO | 84.7 | 77.3 | 74.6 | 67.3 |
| w/o Introspection | 82.0 | 75.7 | 70.8 | 65.2 |
| w/o Constraint Predefining | 83.4 | 76.4 | 74.4 | 68.5 |

### Key Findings

1. **13% H@1 gain on CWQ**: Compared to the previous SOTA LightPROF, DP improves H@1 on CWQ from 59.3 to 75.8.
2. **Introspection is the most critical component**: Removing introspection causes the largest performance drop (WebQSP H@1 ↓4.7%, CWQ H@1 ↓5.0%).
3. **Only 2.5–2.9 LLM interactions required**: Far fewer than ToG (15.9–22.6) and PoG (9.0–13.3), with the lowest token consumption overall.
4. **Gap between H and H@1 is only ~10%**: Far superior to ToG's 46.2% gap, demonstrating the reasoning reliability of DP.
5. **GPT-4.1 triggers backtracking more frequently than GPT-3.5**: Stronger instruction-following ability leads to stricter constraint checking.

## Highlights & Insights

- Systematically integrates KG prior knowledge (structural + constraint) into LLM reasoning in an elegant and effective manner.
- The application of KTO under severe positive-negative class imbalance represents a meaningful technical contribution.
- The backtracking mechanism introduces self-correction capability into LLM reasoning, improving overall system reliability.
- The extremely low LLM interaction count (2–3 calls) makes the approach practically viable for real-world deployment.
- The paper identifies the widespread conflation of H and H@1 metrics in prior work, which has value for academic rigor.

## Limitations & Future Work

- Constraint categories require manual predefinition (5 types); extending to vertical domains demands additional human effort.
- The path generator relies on the quality of weak supervision signals; inaccurate training annotations may propagate errors.
- Future work plans to investigate automatic extraction and summarization of constraint types to reduce manual intervention.
- The introspection mechanism may be explored for other LLM tasks requiring trustworthy reasoning.

## Related Work & Insights

- **KG-RAG methods**: RoG, ToG, DoG, and related approaches have progressively refined LLM–KG interaction but lack sufficient exploitation of prior knowledge.
- **KTO optimization**: Compared to DPO, KTO is better suited for imbalanced data; this paper's application introduces a new alignment paradigm for the KGQA domain.
- **Insight**: Knowledge graphs are not merely external knowledge sources for LLMs—their structure itself encodes rich reasoning priors. Systematically exploiting these priors can significantly improve both faithfulness and reliability of reasoning.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Systematic exploitation of prior knowledge + innovative application of KTO for path generation + constraint-driven introspection with backtracking
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three datasets + multi-LLM integration validation + comprehensive ablation + efficiency analysis + error analysis
- **Writing Quality**: ⭐⭐⭐⭐ Method description is thorough and formulations are rigorous
- **Value**: ⭐⭐⭐⭐⭐ New SOTA on KGQA + efficient and practical + broadly applicable trustworthy reasoning framework

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Dynamic Bundling with Large Language Models for Zero-Shot Inference on Text-Attributed Graphs](dynamic_bundling_with_large_language_models_for_zero-shot_inference_on_text-attr.md)
- [\[CVPR 2026\] Mario: Multimodal Graph Reasoning with Large Language Models](../../CVPR2026/graph_learning/mario_multimodal_graph_reasoning_with_large_language_models.md)
- [\[AAAI 2026\] PathMind: A Retrieve-Prioritize-Reason Framework for Knowledge Graph Reasoning with Large Language Models](../../AAAI2026/graph_learning/pathmind_a_retrieve-prioritize-reason_framework_for_knowledge_graph_reasoning_wi.md)
- [\[NeurIPS 2025\] Reasoning Meets Representation: Envisioning Neuro-Symbolic Wireless Foundation Models](reasoning_meets_representation_envisioning_neuro-symbolic_wireless_foundation_mo.md)
- [\[NeurIPS 2025\] SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs](sstag_structure-aware_self-supervised_learning_method_for_text-attributed_graphs.md)

</div>

<!-- RELATED:END -->
