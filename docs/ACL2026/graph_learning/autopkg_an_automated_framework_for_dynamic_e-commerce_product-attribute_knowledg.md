---
title: >-
  [Paper Note] AutoPKG: An Automated Framework for Dynamic E-commerce Product-Attribute Knowledge Graph Construction
description: >-
  [ACL 2026][Graph Learning][Knowledge Graph Construction] AutoPKG is a multi-agent LLM framework that automatically constructs Product-Attribute knowledge graphs (PKGs) from multimodal e-commerce content…
tags:
  - "ACL 2026"
  - "Graph Learning"
  - "Knowledge Graph Construction"
  - "E-commerce Attribute Extraction"
  - "Multi-Agent LLM"
  - "Dynamic Ontology"
  - "Multimodal"
content_hash: fd4006b3b4cbf5ef
---

# AutoPKG: An Automated Framework for Dynamic E-commerce Product-Attribute Knowledge Graph Construction

**Conference**: ACL 2026
**arXiv**: [2604.16950](https://arxiv.org/abs/2604.16950)
**Code**: [GitHub](https://github.com/Product-Understanding-Lazada-Alibaba/AutoPKG)
**Area**: Knowledge Graph
**Keywords**: Knowledge Graph Construction, E-commerce Attribute Extraction, Multi-Agent LLM, Dynamic Ontology, Multimodal

## TL;DR
AutoPKG is a multi-agent LLM framework that automatically constructs Product-Attribute knowledge graphs (PKGs) from multimodal e-commerce content, using Type Induction Agent, Attribute Key Discovery Agent, Attribute Value Extraction Agent, and centralized KGD Decision Agent, achieving 0.953 WKE for types and +7.89% recommendation GMV in online A/B tests on Lazada.

## Method

### Key Designs

1. **PKG Schema (3 node types + 4 edge types)**: Separates schema structure from instance facts, enabling type checking and incremental extension.

2. **KGD: Centralized Write Interface**: All upstream agents can only "propose"; KGD is the sole writer with constrained edit actions (ADD/MERGE/REPLACE/DISCARD), serving as a gatekeeper ensuring deduplication and normalization.

3. **Dynamic Ontology Population**: Type Induction and Key Discovery agents propose from zero, all normalized through KGD.

## Key Experimental Results

- Type induction WKE: 0.953 (Qwen3-4B)
- KGD decision accuracy: 0.764 (Qwen3-Next-80B-A3B)
- Online A/B recommendation GMV: +7.89%
- Small models fail catastrophically at KGD decisions (Llama-3.2-3B: 0.384)

## Highlights & Insights
- KGD's "propose → normalize → write" paradigm elegantly decouples free-text generation chaos from knowledge graph consistency needs
- WKE evaluation protocol balances multiple dimensions, preventing single-metric gaming
- End-to-end industrial validation from offline metrics to online A/B tests

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FALCON: An ML Framework for Fully Automated Layout-Constrained Analog Circuit Design](../../NeurIPS2025/graph_learning/falcon_an_ml_framework_for_fully_automated_layout-constrained_analog_circuit_des.md)
- [\[AAAI 2026\] PathMind: A Retrieve-Prioritize-Reason Framework for Knowledge Graph Reasoning with Large Language Models](../../AAAI2026/graph_learning/pathmind_a_retrieve-prioritize-reason_framework_for_knowledge_graph_reasoning_wi.md)
- [\[AAAI 2026\] NOTAM-Evolve: A Knowledge-Guided Self-Evolving Optimization Framework with LLMs for NOTAM Interpretation](../../AAAI2026/graph_learning/notam-evolve_a_knowledge-guided_self-evolving_optimization_framework_with_llms_f.md)
- [\[ICLR 2026\] Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding](../../ICLR2026/graph_learning/entropy-guided_dynamic_tokens_for_graph-llm_alignment_in_molecular_understanding.md)
- [\[NeurIPS 2025\] Unifying and Enhancing Graph Transformers via a Hierarchical Mask Framework](../../NeurIPS2025/graph_learning/unifying_and_enhancing_graph_transformers_via_a_hierarchical_mask_framework.md)

</div>

<!-- RELATED:END -->
