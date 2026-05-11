---
title: >-
  [Paper Note] AnchorMem: Anchored Facts with Associative Contexts for Building Memory in Large Language Models
description: >-
  [ACL 2026][LLM Evaluation][LLM Memory] This paper proposes AnchorMem, a memory framework inspired by the Proustian phenomenon in cognitive science. It decouples retrieval units (atomic facts) from generation contexts (or…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "LLM Memory"
  - "Atomic Facts"
  - "Associative Event Graph"
  - "Retrieval Augmentation"
  - "Long-term Dialogue"
date: 2026-05-08
content_hash: cf63c856f3468211
---

# AnchorMem: Anchored Facts with Associative Contexts for Building Memory in Large Language Models

**Conference**: ACL 2026
**arXiv**: [2604.17377](https://arxiv.org/abs/2604.17377)
**Code**: [https://github.com/RayNeo-AI-2025/AnchorMem](https://github.com/RayNeo-AI-2025/AnchorMem)
**Area**: LLM Evaluation
**Keywords**: LLM Memory, Atomic Facts, Associative Event Graph, Retrieval Augmentation, Long-term Dialogue

## TL;DR

This paper proposes AnchorMem, a memory framework inspired by the Proustian phenomenon in cognitive science. It decouples retrieval units (atomic facts) from generation contexts (original interactions) and connects fragmented memories via an associative event graph, achieving substantial improvements over existing memory systems such as A-Mem and Mem0 on the LoCoMo benchmark.

## Background & Motivation

**Background**: LLMs require memory systems to leverage historical experience in long-term multi-turn interactions. Existing approaches fall into two main categories: generative memory systems (A-Mem, Mem0), which integrate new information through frequent rewriting, and graph-indexing methods (GraphRAG, HippoRAG 2), which structure knowledge via entity-relation graphs.

**Limitations of Prior Work**: Generative memory systems suffer from "context blurring"—frequent rewriting dilutes or loses the nuanced context of original interactions. Graph-indexing methods suffer from "spurious connections"—high-frequency but semantically general entities (e.g., "oil") create misleading bridges that cause retrieval paths to incorrectly conflate unrelated events.

**Key Challenge**: There exists a fundamental tension between fine-grained retrieval (requiring precise retrieval units) and contextual integrity (requiring preservation of original interaction context). Existing methods either sacrifice contextual fidelity for retrieval efficiency or introduce noisy connections through over-structuring.

**Goal**: Design a memory framework that achieves precise atomic-level retrieval while preserving the integrity of original interaction context.

**Key Insight**: Inspired by the "Proustian phenomenon" in cognitive science—whereby a specific sensory cue triggers complete episodic recall—the paper explicitly decouples retrieval anchors from generation contexts.

**Core Idea**: Extract atomic facts as retrieval anchors, preserve original interactions as immutable context, and establish cross-memory logical connections through an associative event graph.

## Method

### Overall Architecture

AnchorMem is constructed as a heterogeneous graph $\mathcal{G}=(\mathcal{V}, \mathcal{E})$, where nodes comprise three types: interaction contexts $\mathcal{V}_C$, atomic facts $\mathcal{V}_F$, and associative events $\mathcal{V}_E$. The edge set includes mapping edges (surjective mappings from facts to contexts) and associative edges (semantic connections between facts). The overall pipeline consists of three stages: fact-context construction → associative event graph construction → memory retrieval and generation.

### Key Designs

1. **Fact-Context Units**:

    - **Function**: Decouple each interaction into retrievable atomic facts and immutable original context.
    - **Mechanism**: For each interaction segment $C_i$, an LLM extracts a set of atomic facts $\mathcal{F}_i = \{f_{i,1}, ..., f_{i,n}\}$, where each fact is a concise, self-contained statement. A surjective mapping $\mathcal{M}(f) = C_i$ is established so that any retrieved fact immediately localizes its original context. The original interaction content remains immutable and is never rewritten or compressed.
    - **Design Motivation**: Avoid the context blurring caused by frequent rewriting while enabling precise retrieval through atomic facts.

2. **Associative Event Graph**:

    - **Function**: Establish cross-memory logical connections without relying on generalized entity bridges.
    - **Mechanism**: Semantically related atomic facts are aggregated into "associative event" nodes. Higher-order event linking binds related fact sets into a shared event representation, rather than using generic entities as bridges. This preserves the ability to integrate fragmented information while avoiding the spurious connections caused by high-frequency entities in graph indexing.
    - **Design Motivation**: Address the imprecision of entity-level connections in traditional graph indexing by achieving more reliable cross-memory integration through event-level aggregation.

3. **Memory Retrieval and Context Reconstruction**:

    - **Function**: Anchor queries to specific facts and events to reconstruct complete generation context.
    - **Mechanism**: During retrieval, queries are anchored to specific atomic facts and associative events to locate relevant memories; during generation, the associated original interaction segments and events are used to reconstruct context. Retrieval precision derives from exact matching of atomic facts, while generation quality derives from the complete preservation of original context.
    - **Design Motivation**: Pursue precision at the retrieval stage and completeness at the generation stage, seamlessly bridged through the mapping relation.

### Loss & Training

AnchorMem is a training-free framework requiring no additional training. Fact extraction and event construction are accomplished through LLM prompt engineering.

## Key Experimental Results

### Main Results

Evaluated on the LoCoMo benchmark using GPT-4o-mini:

| Method | Avg F1 | Avg BLEU | Avg ACC |
|--------|--------|----------|---------|
| NaiveRAG | 33.06 | 25.90 | 47.92 |
| HippoRAG 2 | 43.11 | 30.52 | 61.82 |
| Mem0 | 29.07 | 23.34 | 39.31 |
| A-Mem | 32.39 | 26.57 | 51.35 |
| LightMem | 41.71 | 32.14 | 61.95 |
| **AnchorMem** | **49.87** | **38.85** | **70.52** |
| **AnchorMem†** | **51.91** | **40.25** | **73.83** |

### Ablation Study

| Configuration | Key Advantage | Description |
|---------------|--------------|-------------|
| vs Mem0 | +22.80 F1 | Avoids context loss from frequent rewriting |
| vs HippoRAG 2 | +6.76 F1 | Avoids spurious entity-level connections |
| vs LightMem | +8.16 F1 | Atomic facts + event graph outperform topic summarization |
| AnchorMem† (enhanced) | +2.04 F1 | Further optimized retrieval strategy |

### Key Findings

- AnchorMem outperforms all baselines across all four question types (single-hop, multi-hop, open-domain, and temporal), achieving an F1 of 55.84 on single-hop questions—more than 11 points ahead of LightMem.
- Consistent advantages are observed on open-source models such as Qwen2.5-32B, demonstrating model-agnosticism.
- By avoiding frequent rewriting, AnchorMem achieves the fastest memory construction speed among all evaluated methods.

## Highlights & Insights

- The Proustian phenomenon analogy is particularly apt: just as the taste of a madeleine evokes a complete childhood memory, atomic facts serve as "anchors" that trigger recall of full interaction episodes. This cognitive science inspiration translates directly into an effective technical design.
- The decoupling of retrieval units from generation context is the core innovation: retrieval handles what it does best (precise matching), and generation leverages what it needs most (complete context), connected through a mapping relation that is more flexible than end-to-end compression.
- The associative event graph uses "events" rather than "entities" as cross-memory bridges, effectively avoiding spurious associations caused by generic entities (e.g., person names, common nouns)—a design transferable to other RAG scenarios.

## Limitations & Future Work

- The quality of atomic fact extraction depends on LLM prompt engineering, and different LLMs may yield facts of varying quality.
- As interaction history grows, the number of atomic fact and event nodes scales linearly, necessitating long-term maintenance strategies.
- Evaluation is currently limited to the LoCoMo benchmark; validation in real-world application scenarios (e.g., personal assistants, customer service) is absent.
- The strategy for constructing associative events (i.e., determining which facts belong to the same event) warrants further investigation.

## Related Work & Insights

- **vs A-Mem**: A-Mem links and evolves notes through retrieval-driven mechanisms, but frequent rewriting degrades context; AnchorMem keeps original context immutable.
- **vs Mem0**: Mem0 performs fine-grained extraction and state updates, but incurs significant information loss during updates; AnchorMem achieves an F1 gain of 22.8 points.
- **vs HippoRAG 2**: HippoRAG 2 uses entity graphs as associative signals for reranking, but entity-level connections are insufficiently reliable; AnchorMem's event-level connections are more precise.

## Rating
- Novelty: ⭐⭐⭐⭐ — Novel design combining retrieval-generation decoupling with cognitive science inspiration.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive comparisons across multiple models and question types.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear cognitive science motivation and rigorous formal definitions.
- Value: ⭐⭐⭐⭐ — Provides a practical and effective new paradigm for long-term LLM memory systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](closing_the_modality_reasoning_gap_for_speech_large_language_models.md)
- [\[ACL 2026\] E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task](e2edev_benchmarking_large_language_models_in_end-to-end_software_development_tas.md)
- [\[ACL 2026\] Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models](attribution_citation_and_quotation_a_survey_of_evidence-based_text_generation_wi.md)
- [\[ACL 2026\] Modeling Multi-Dimensional Cognitive States in Large Language Models under Cognitive Crowding](modeling_multi-dimensional_cognitive_states_in_large_language_models_under_cogni.md)
- [\[ACL 2026\] Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation: A Case Study on Tang Poetry](capabilities_and_evaluation_biases_of_large_language_models_in_classical_chinese.md)

</div>

<!-- RELATED:END -->
