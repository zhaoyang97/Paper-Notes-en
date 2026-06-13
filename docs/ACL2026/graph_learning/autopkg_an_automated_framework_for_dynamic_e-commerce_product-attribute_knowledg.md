---
title: >-
  [Paper Note] AutoPKG: An Automated Framework for Dynamic E-commerce Product-Attribute Knowledge Graph Construction
description: >-
  [ACL 2026][Graph Learning][Knowledge Graph Construction] AutoPKG is proposed as a multi-agent LLM framework to automatically construct Product-Attribute Knowledge Graphs (PKG) from multimodal e-commerce product content.…
tags:
  - "ACL 2026"
  - "Graph Learning"
  - "Knowledge Graph Construction"
  - "E-commerce Attribute Extraction"
  - "Multi-agent LLM"
  - "Dynamic Ontology"
  - "Multimodal"
date: 2026-05-08
content_hash: c137fc7f46197394
---

# AutoPKG: An Automated Framework for Dynamic E-commerce Product-Attribute Knowledge Graph Construction

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.16950](https://arxiv.org/abs/2604.16950)  
**Code**: [GitHub](https://github.com/Product-Understanding-Lazada-Alibaba/AutoPKG)  
**Area**: Knowledge Graph  
**Keywords**: Knowledge Graph Construction, E-commerce Attribute Extraction, Multi-agent LLM, Dynamic Ontology, Multimodal

## TL;DR
AutoPKG is proposed as a multi-agent LLM framework to automatically construct Product-Attribute Knowledge Graphs (PKG) from multimodal e-commerce product content. Through specialized agents for Type Induction, Attribute Key Discovery, Attribute Value Extraction, and a centralized KGD decision agent, it enables continuous evolution and normalization of dynamic ontologies. It achieves $0.953$ WKE (Type) and $0.724$ WKE (Key) on the Lazada dataset, with a $7.89\%$ GMV increase in online A/B testing for recommendations.

## Background & Motivation

**Background**: Product attributes are core to e-commerce infrastructure, supporting faceted navigation, search ranking, and recommendation systems. Industrial attribute extraction pipelines typically rely on human-maintained product taxonomies and perform extraction against fixed attribute lists.

**Limitations of Prior Work**: (1) Human-maintained taxonomies are often inconsistent across markets, incomplete for long-tail products, and costly to maintain under continuous distribution shifts. (2) Even powerful Product Attribute Value Extraction (PAVE) models are restricted to outdated or narrow attribute lists, limiting coverage. (3) Existing frameworks lack integrated support for automatic type induction, attribute key discovery, and multimodal extraction.

**Key Challenge**: The e-commerce product space is dynamic, long-tail, and multimodal, whereas existing knowledge graph construction (KGC) methods assume fixed, human-managed schemas that fail to adaptively evolve.

**Goal**: To build a framework that requires no predefined taxonomy and can automatically construct and continuously evolve a PKG from scratch while supporting multimodal attribute extraction from text and images.

**Key Insight**: PKG construction is decomposed into a collaboration of four specialized agents. The core innovation is the Knowledge Graph Decision (KGD) agent—a centralized authority through which all write operations must pass. It ensures global consistency using a restricted set of editing actions (ADD/MERGE/REPLACE/DISCARD).

**Core Idea**: A multi-agent LLM "propose-normalize-write" paradigm for incremental PKG construction is implemented. Centralized KGD unifies all updates into constrained editing decisions to achieve continuous deduplication and normalization.

## Method

### Overall Architecture
For each new product listing: the Type Induction Agent infers the product type from text $\rightarrow$ the Attribute Key Discovery Agent generates a specific attribute key list based on the type $\rightarrow$ the Attribute Value Extraction Agent extracts values from text and images $\rightarrow$ the KGD Agent performs normalized edits (ADD/MERGE/REPLACE/DISCARD) on all proposals to maintain a unified canonical graph.

### Key Designs

1.  **PKG Schema Design (3 Node Types + 4 Edge Types)**:
    -   **Function**: Supports a minimal graph structure for continuous expansion.
    -   **Mechanism**: Nodes include Product, ProductType (canonical), and AttributeKey. Edges are divided into schema edges (has_key, has_value) defining the type-attribute structure, and instance edges (of_type, has_attribute) recording product-level facts. Attribute values are typed via AttributeKeys to enable canonical value reuse across products.
    -   **Design Motivation**: Separates schema structure from instance facts to enable type checking (values must be permitted by the ProductType via its keys) while supporting incremental expansion.

2.  **KGD: Centralized Writing Interface**:
    -   **Function**: Ensures global consistency and normalization of all graph updates.
    -   **Mechanism**: Upstream agents only "propose," while KGD is the sole agent permitted to "write." For each proposal, KGD receives candidate content and a retrieved local graph neighborhood to select a constrained edit action: ADD (create new node/edge), MERGE (merge candidate into existing canonical node), REPLACE (promote candidate surface form to a new canonical label), or DISCARD (reject invalid proposal). Dense retrieval is performed using Qwen3-Embedding-0.6B on canonical names.
    -   **Design Motivation**: Decentralized writing leads to synonym explosion and normalization inconsistency. KGD acts as a "gatekeeper," forcing every update through deduplication and normalization logic.

3.  **Dynamic Ontology Population (Types & Keys)**:
    -   **Function**: Automatically discovers product types and attribute keys from scratch.
    -   **Mechanism**: The Type Induction Agent proposes canonical product types and brief descriptions from listings; the Attribute Key Discovery Agent proposes attribute keys (including definitions and example values) based on inferred types. All proposals are normalized by KGD—deciding whether to create a new type/key or merge into an existing one.
    -   **Design Motivation**: Eliminates dependency on predefined taxonomies and supports automatic coverage of long-tail categories.

### Evaluation Protocol
A new metric, WKE (Weighted Knowledge Efficiency), is proposed. It is a weighted harmonic mean of Acceptance, Compression, and Coverage (for types) or Acceptance, P-Precision, and P-Recall (for keys). This prevents aggressive merging strategies from appearing high-performing through single-metric gains.

## Key Experimental Results

### Main Results (Product Type Induction, WKE)

| Model | Acceptance | Compression | Coverage | WKE |
| :--- | :--- | :--- | :--- | :--- |
| Qwen3-4B | 0.952 | 0.948 | 0.960 | **0.953** |
| Qwen3-30B | 0.954 | 0.936 | 0.960 | 0.952 |
| Gemma-3n-E4B | 0.936 | 0.967 | 0.986 | 0.952 |
| SmolLM3-3B | 0.646 | 0.933 | 0.613 | 0.682 |

### KGD Decision Accuracy

| Model | Accuracy |
| :--- | :--- |
| Qwen3-Next-80B-A3B | **0.764** |
| Qwen3-30B-A3B | 0.734 |
| Llama-3.2-3B | 0.384 |
| SmolLM3-3B | 0.123 |

### Online A/B Test (GMV Gain)

| Scenario | GMV Gain |
| :--- | :--- |
| Badge | +3.81% |
| Search | +5.32% |
| Recommendation | **+7.89%** |
| Filter | +0.26% (not sig.) |

### Key Findings
- **KGD normalization requires strong semantic reasoning**: Small models (Llama-3.2-3B: 0.384) frequently fail the MERGE vs. ADD decision, leading to persistent downstream losses.
- **Moderate-sized models suffice for Type Induction**: Qwen3-4B reached the highest WKE of $0.953$, though KGD requires larger models.
- **Attribute Key Discovery bottleneck is long-tail coverage**: High precision (0.93-0.99) but low recall (0.27-0.47) suggests rare keys remain difficult to discover automatically.
- **Multimodal extraction F1 is moderate ($0.531$)**: Reflects real-world challenges of noisy seller text and non-diagnostic images.
- **Online A/B tests prove production value**: Significant GMV gains observed, especially in recommendation scenarios (+7.89%).

## Highlights & Insights
- The **"propose-normalize-write" paradigm of KGD** is highly practical—it decouples the chaos of free-text generation from the consistency required by a KG. Using a constrained action space (4 options) allows LLMs to make structured decisions. This pattern is transferable to any scenario requiring continuous construction of canonical knowledge bases.
- The **WKE evaluation protocol** effectively balances multiple dimensions, preventing the gaming of single metrics (e.g., achieving high compression by only merging without adding, which kills coverage).
- **Industrial validation** from offline metrics to online A/B testing provides a comprehensive blueprint for both academic research and industrial implementation.

## Limitations & Future Work
- High cost due to dependence on powerful LLMs as the KGD backbone (Qwen3-Next-80B inference latency is significant).
- Long-tail coverage of attribute keys is not fully resolved (max recall only $0.474$).
- The four KGD actions are manually designed; future work could explore learning even more flexible editing strategies.
- Dataset originates from the Southeast Asian market (Lazada); generalization across markets and languages needs further validation.
- Lacks direct comparison with industrial systems like AutoKnow or AliCG due to data unavailability.

## Related Work & Insights
- **vs. AutoKnow (Dong et al., 2020)**: AutoKnow constructs e-commerce KGs but assumes a human-managed schema. AutoPKG enables automatic schema induction and continuous evolution.
- **vs. AutoSchemaKG (Bai et al., 2025)**: AutoSchemaKG performs general domain schema induction but lacks multimodal support and e-commerce specialization.
- **vs. PAVE methods**: Traditional PAVE extracts values against fixed attribute lists. AutoPKG unifies schema evolution with value extraction in a single framework.

## Rating
- Novelty: ⭐⭐⭐⭐ First open framework to support type induction, key discovery, and multimodal extraction simultaneously.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation via offline, online, and public benchmarks with a well-designed evaluation protocol.
- Writing Quality: ⭐⭐⭐⭐ Complete system paper structure, though high information density requires careful reading.
- Value: ⭐⭐⭐⭐⭐ Significant contribution to the e-commerce KG community through industrial systems, open data, and evaluation protocols.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ComplianceNLP: Knowledge-Graph-Augmented RAG for Multi-Framework Regulatory Gap Detection](compliancenlp_knowledge-graph-augmented_rag_for_multi-framework_regulatory_gap_d.md)
- [\[NeurIPS 2025\] FALCON: An ML Framework for Fully Automated Layout-Constrained Analog Circuit Design](../../NeurIPS2025/graph_learning/falcon_an_ml_framework_for_fully_automated_layout-constrained_analog_circuit_des.md)
- [\[ICML 2026\] DTKG: Dual-Track Knowledge Graph-Verified Reasoning Framework for Multi-Hop QA](../../ICML2026/graph_learning/dtkg_dual-track_knowledge_graph-verified_reasoning_framework_for_multi-hop_qa.md)
- [\[AAAI 2026\] PathMind: A Retrieve-Prioritize-Reason Framework for Knowledge Graph Reasoning with Large Language Models](../../AAAI2026/graph_learning/pathmind_a_retrieve-prioritize-reason_framework_for_knowledge_graph_reasoning_wi.md)
- [\[ICLR 2026\] Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding](../../ICLR2026/graph_learning/entropy-guided_dynamic_tokens_for_graph-llm_alignment_in_molecular_understanding.md)

</div>

<!-- RELATED:END -->
