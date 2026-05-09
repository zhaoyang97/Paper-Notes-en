---
title: >-
  [Paper Note] Privacy-protected Retrieval-Augmented Generation for Knowledge Graph Question Answering
description: >-
  [AAAI 2026][LLM Safety][Privacy-preserving RAG] This work is the first to explore privacy-protected RAG for Knowledge Graph Question Answering (KGQA). It proposes ARoG (Abstraction Reasoning on Graph), a framework that employs two strategies—relation-centric abstraction and structure-oriented abstraction—to enable effective retrieval and utilization of knowledge graphs for question answering even when entities are anonymized (replaced with semantically meaningless MIDs).
tags:
  - AAAI 2026
  - LLM Safety
  - Privacy-preserving RAG
  - Knowledge Graph Question Answering
  - Entity Anonymization
  - Abstract Reasoning
  - LLM Privacy
date: 2026-05-08
content_hash: 81a581bd3ff4bcbb
---

# Privacy-protected Retrieval-Augmented Generation for Knowledge Graph Question Answering

**Conference**: AAAI 2026
**arXiv**: [2508.08785](https://arxiv.org/abs/2508.08785)
**Code**: [https://github.com/NLPGM/ARoG](https://github.com/NLPGM/ARoG)
**Area**: AI Safety
**Keywords**: Privacy-preserving RAG, Knowledge Graph Question Answering, Entity Anonymization, Abstract Reasoning, LLM Privacy

## TL;DR

This work is the first to explore privacy-protected RAG for Knowledge Graph Question Answering (KGQA). It proposes ARoG (Abstraction Reasoning on Graph), a framework that employs two strategies—relation-centric abstraction and structure-oriented abstraction—to enable effective retrieval and utilization of knowledge graphs for question answering even when entities are anonymized (replaced with semantically meaningless MIDs).

## Background & Motivation

RAG (Retrieval-Augmented Generation) enhances LLM output quality by retrieving factual information from external knowledge sources such as knowledge graphs (KGs), alleviating hallucination and knowledge staleness. However, many real-world KGs contain **sensitive private information** (personal data, corporate secrets, etc.).

**Core Privacy Risk**: When using a private KG to answer questions, existing RAG systems must expose relevant triples to the LLM. For example, answering "Where does Bronny live?" requires sending the triple (Bronny, lives in, L.A.) to the LLM. Due to the **black-box nature** of LLMs and the potential insecurity of data transmission, this poses a serious privacy leakage risk, especially when using third-party LLM APIs.

**Privacy-Protected RAG Scenario**: KG entities are anonymous to the LLM—entities are replaced with encrypted unique machine identifiers (MIDs) that carry no semantic information. The LLM has no access to entity types, names, or descriptions.

**Two Core Challenges**:

**How to transform anonymous entities into retrievable information?** MIDs carry no semantics; the LLM cannot retrieve relevant knowledge by matching MIDs to the question.

**How to retrieve anonymous entities relevant to the question?** Questions are in natural language, whereas KG entities are meaningless identifiers, leaving no semantic bridge between the two.

**Key Insight**: Although entities are anonymized, the **relations** in a KG define schema-level patterns rather than sensitive information and can be safely shared with the LLM. The abstract concept of an entity can be inferred from its surrounding relations (e.g., an entity that serves as subject of *time_zones*, *contained_by*, and *population*, and as object of *citytown*, can be abstracted as "geographic location").

## Method

### Overall Architecture

ARoG adopts a **Retrieval-then-Generation** pipeline consisting of four modules:

1. **Relation-centric Abstraction (RA)**: Abstracts anonymous entities into high-level concepts based on their neighboring relations.
2. **Structure-oriented Abstraction (SA)**: Transforms unstructured questions into structured abstract concept paths.
3. **Abstraction-driven Retrieval (AR)**: Performs multi-round iterative retrieval guided by both abstractions.
4. **Generator**: Infers the answer from the retrieved triples.

### Key Designs

#### 1. **Relation-centric Abstraction Module**

Relations are treated as "predicate verbs" and entities as "subject/object nouns"; entity concepts are inferred through their relations. The module proceeds in three steps:

**Step 1: Relation Retrieval** — Starting from $n$ topic entities, neighboring relations are extracted, and the LLM selects $W$ most relevant relations:

$$R_{\text{t,opt}} = \arg\max_{S \subseteq R_t, |S| \leq W} \text{LLM}(R_t, q, Inst_{rr}, E_{rr})$$

**Step 2: Relation Filtering** — For each entity in the candidate set, a SentenceTransformer selects the top-$K$ ($K=5$) relations most relevant to the question by cosine similarity:

$$R_{v,opt} = \arg\max_{S \subseteq R_v, |S| \leq K} \sum_{r \in S} \cos(\text{Emb}(q), \text{Emb}(r))$$

**Step 3: Entity Abstraction** — The LLM infers an abstract concept for each entity based on the filtered relations, which is then appended to its MID:

$$e^{abs} = \text{LLM}(R_{v,opt}, e, Inst_{ea}, E_{ea})$$

Key design choices: entities within the same cluster share a single concept (reducing LLM calls); multiple concepts are joined by commas.

#### 2. **Structure-oriented Abstraction Module**

Unstructured questions are converted into structured **abstract concept paths** $P_q$. A key advantage is that **path validity does not depend on correct entity names**.

For example, the question *"What is the daughter of the artist who had The Mrs. Carter Show World Tour?"* yields the abstract concept path:

```
Nicki Minaj (artist) → had → The Mrs. Carter Show World Tour
Nicki Minaj (artist) → has daughter named → Chiara Fattorini (person)
```

Although both "Nicki Minaj" and "Chiara Fattorini" are incorrect entities, the path—containing the concepts "artist" and "person"—remains semantically aligned with the abstracted triples in the KG.

Generation uses a Chain-of-Thought (CoT) approach:

$$Rat, P_q = \text{LLM}(q, Inst_{sa}, E_{sa})$$

#### 3. **Abstraction-driven Retrieval Module**

Multi-round iterative retrieval with width $W$ and depth $D$ (both defaulting to 3):

In each iteration, candidate triples $T_q^{abs}$ are obtained via relation-centric abstraction and then matched against triples in the abstract concept path $P_q$ by cosine similarity, selecting $W$ most relevant ones:

$$T_{q,\text{opt}}^{abs} = \arg\max_{S \subseteq T_q^{abs}, |S| \leq W} \sum_{t^{abs} \in S} \max_{t_p \in P_q} \cos(\text{Emb}(t^{abs}), \text{Emb}(t_p))$$

Newly discovered entities replace the topic entities in the next iteration.

#### 4. **Generator Module**

The accumulated relevant triples $T_{q,all}^{abs}$ and the original question $q$ are fed to the LLM to generate an answer. The output includes a Flag: if positive, the answer is returned directly; otherwise, the next retrieval round is triggered.

**MIDs in the answer are replaced with real names on the user side** (the user retains the MID-to-name mapping), completing the full privacy-protection loop.

### Loss & Training

ARoG is a purely inference-time framework with no model training. Key configurations:

- **LLM**: gpt-4o-mini-2024-07-18
- **Temperature**: 0 for SA and Generator (deterministic output); 0.4 for RA (introducing diversity)
- **Retrieval parameters**: width $W = 3$, depth $D = 3$
- **Embedding model**: SentenceTransformer for relation filtering and triple matching
- **Results averaged over 3 runs**

## Key Experimental Results

### Main Results

| Type | Method | WebQSP #Tot | WebQSP #Fil | CWQ #Tot | CWQ #Fil | GrailQA #Tot | GrailQA #Fil |
|---|---|---|---|---|---|---|---|
| Pure | CoT | 67.2 | 0.0 | 55.1 | 0.0 | 35.5 | 0.0 |
| RAG | ToG | 64.9 | 8.2 | 54.1 | 4.9 | 38.9 | 17.0 |
| RAG | PoG | 61.4 | 12.6 | 49.8 | 16.3 | 49.7 | 36.1 |
| RAG | GoG | 62.3 | 26.9 | 48.1 | 16.5 | 29.1 | 12.4 |
| **RAG** | **ARoG** | **74.7** | **58.9** | **60.0** | **36.3** | **78.7** | **71.8** |
| | Gain | +6.1 | +5.8 | +4.9 | +19.8 | +16.4 | +9.0 |

ARoG's advantage is especially pronounced under the **#Filtered setting** (questions that cannot be answered by the LLM's internal knowledge alone). ToG achieves only 8.2% on WebQSP #Fil, while ARoG reaches 58.9%. On GrailQA, ARoG achieves 78.7% (#Tot) and 71.8% (#Fil), substantially outperforming all baselines.

### Ablation Study

| Relation Retrieval | Relation Filtering | Entity Abstraction | Structure Abstraction | WebQSP #Tot | CWQ #Tot | GrailQA #Tot |
|---|---|---|---|---|---|---|
| ✓ | × | × | × | 63.9 | 36.4 | 72.6 |
| ✓ | × | ✓ | × | 68.3 | 47.9 | 76.9 |
| ✓ | × | ✓ | ✓ | 72.9 | 59.5 | 78.3 |
| ✓ | ✓ | ✓ | × | 68.1 | 50.3 | 76.3 |
| ✓ | ✓ | ✓ | ✓ | **74.7** | **60.0** | **78.7** |

- Removing RA: #Filtered performance drops by at least 3.8% (private knowledge retrieval depends on RA).
- Removing SA: the largest drop occurs on CWQ (multi-hop reasoning relies on structured paths).
- The two abstraction strategies are complementary; combining both yields the best results.

### Key Findings

1. **Existing RAG methods nearly fail under privacy-protected settings**: Methods such as ToG and PoG rely heavily on entity semantics; anonymization causes substantial performance degradation.
2. **Stability of SP methods**: Semantic parsing methods do not depend on entity information within the KG and thus exhibit relatively stable performance as the setting transitions from #Total to #Filtered.
3. **Abstract concept paths outperform CoT and question decomposition**: Compared to CoT rationales and sub-question decomposition, abstract concept paths are more effective for retrieval because they incorporate concept inference over anonymous entities.
4. **Retrieval efficiency**: ARoG achieves the lowest total token usage on GrailQA (5,605), demonstrating superior efficiency.

## Highlights & Insights

- **Novel problem formulation**: This is the first work to formally define and systematically address the privacy-protected RAG scenario, offering a new perspective on KG privacy protection.
- **Relations as semantic bridges**: Leveraging KG relations (schema-level information) to infer the semantics of anonymous entities is an elegant design—relations are not themselves private data, yet they can be used to reconstruct semantic meaning.
- **Error tolerance of concept paths**: Abstract concept paths remain effective even when entity names are incorrect, greatly improving system robustness on open-domain questions.
- **Complete privacy loop**: MIDs are only resolved to real entity names on the user side; the LLM never has access to entity semantics at any stage.

## Limitations & Future Work

1. **Dependence on relation information**: If relations in the KG also need to be anonymized, the current approach would fail.
2. **LLM API call overhead**: Each retrieval round requires multiple LLM calls (12–25 on average), which may be prohibitive in cost-sensitive scenarios.
3. **Evaluation limited to Freebase**: Validation on other KG types (e.g., Wikidata, domain-specific KGs) is absent.
4. **Topic entity assumption**: The approach assumes that topic entity names in questions are available, which may not hold in certain privacy settings.
5. **Abstraction quality depends on LLM capability**: The quality of entity abstraction is bounded by the LLM's reasoning ability and may be insufficient for specialized domains.

## Related Work & Insights

- The evolution of RAG systems from "direct knowledge exposure" to "privacy-protected retrieval" is an inevitable trend.
- The decoupling strategy between relations and entities in KGs can be generalized to other privacy-protection scenarios.
- The abstract reasoning paradigm (from concrete to abstract and back to concrete) can be applied to federated KG querying.
- Future work could incorporate secure multi-party computation or homomorphic encryption to further strengthen privacy guarantees.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (first to define privacy-protected RAG; abstraction reasoning strategy is highly original)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (3 datasets × 2 settings; ablation, efficiency, quantitative, and in-depth analysis; very comprehensive)
- Writing Quality: ⭐⭐⭐⭐⭐ (problem definition is clear; method description is detailed; figures are intuitive)
- Value: ⭐⭐⭐⭐⭐ (pioneers a new direction in privacy-protected RAG with high practical value)

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Resource-Adaptive Federated Text Generation with Differential Privacy](../../ICLR2026/llm_safety/resource-adaptive_federated_text_generation_with_differential_privacy.md)
- [\[AAAI 2026\] PRISM: Privacy-Aware Routing for Adaptive Cloud-Edge LLM Inference via Semantic Sketch Collaboration](prism_privacy-aware_routing_for_adaptive_cloud-edge_llm_inference_via_semantic_s.md)
- [\[NeurIPS 2025\] InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy](../../NeurIPS2025/llm_safety/invisibleink_high-utility_and_low-cost_text_generation_with_differential_privacy.md)
- [\[AAAI 2026\] Beyond Superficial Forgetting: Thorough Unlearning through Knowledge Density Estimation and Block Re-insertion](beyond_superficial_forgetting_thorough_unlearning_through_knowledge_density_esti.md)
- [\[CVPR 2026\] Learning from Oblivion: Predicting Knowledge-Overflowed Weights via Retrodiction of Forgetting](../../CVPR2026/llm_safety/learning_from_oblivion_predicting_knowledge_overflowed_weights_via_retrodiction_.md)

<!-- RELATED:END -->
