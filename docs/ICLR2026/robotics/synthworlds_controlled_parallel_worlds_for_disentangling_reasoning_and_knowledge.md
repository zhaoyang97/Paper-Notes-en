---
title: >-
  [Paper Note] SynthWorlds: Controlled Parallel Worlds for Disentangling Reasoning and Knowledge in Language Models
description: >-
  [ICLR 2026][Robotics][Knowledge Advantage Gap] This paper constructs structurally identical parallel corpora in which entities are mapped to either real or synthetic names…
tags:
  - "ICLR 2026"
  - "Robotics"
  - "Knowledge Advantage Gap"
  - "Reasoning vs Memorization"
  - "Parallel Corpora"
  - "Multi-hop QA"
  - "RAG Evaluation"
date: 2026-05-08
content_hash: 7feb28b32d734802
---

# SynthWorlds: Controlled Parallel Worlds for Disentangling Reasoning and Knowledge in Language Models

**Conference**: ICLR 2026
**arXiv**: [2510.24427](https://arxiv.org/abs/2510.24427)  
**Code**: [GitHub](https://github.com/behavioral-data/synthworlds)  
**Area**: Robotics
**Keywords**: Knowledge Advantage Gap, Reasoning vs Memorization, Parallel Corpora, Multi-hop QA, RAG Evaluation

## TL;DR

This paper constructs structurally identical parallel corpora in which entities are mapped to either real or synthetic names, and quantifies the Knowledge Advantage Gap (KA) — the contribution of parametric knowledge — by comparing model performance across the two "parallel worlds." The results show that this gap persists even when models are augmented with RAG and CoT.

## Background & Motivation

**Background**: Language models have achieved impressive results on complex tasks such as multi-hop QA and web navigation. However, because training data is rarely disclosed, it remains difficult to determine whether performance gains stem from genuine reasoning ability or from memorization of factual knowledge in the training corpus. Existing benchmarks increasingly lose diagnostic validity as training data scales — for instance, MuSiQue (released in 2021) was designed so that models could not answer questions without supporting documents, yet Llama-3.3-70B now achieves over 26% F1 in the closed-book setting.

**Limitations of Prior Work**:
1. **Manually curated evaluation sets**: Costly, difficult to scale, require continuous updating, and will eventually be subsumed by model training data.
2. **Synthetic data approaches**: Either reuse existing content (e.g., novels), allowing parametric knowledge leakage, or rely on overly simple templates (e.g., "The job of David is a farmer") that fail to test complex relational reasoning.
3. **Evaluating on synthetic tasks alone**: Success only demonstrates that a model can reason, while failure is ambiguous — it may reflect overly complex reasoning chains or the absence of background knowledge the model typically relies on.

**Key Challenge**: Existing evaluation methods cannot simultaneously control for reasoning difficulty and the contribution of parametric knowledge. Without experimentally decoupling reasoning from memorization, the fundamental question — "Is the model reasoning or recalling?" — cannot be answered.

**Goal**: The paper proposes SynthWorlds, a framework that automatically generates two structurally identical parallel corpora from a knowledge graph:
- **Real-Mapped (RM)**: Entities use real names (e.g., Geoffrey Hinton, University of Toronto), where parametric knowledge may be beneficial.
- **Synth-Mapped (SM)**: Entities use synthetic names (e.g., Caleb Ardent, University of Metrovale), where parametric knowledge is entirely inapplicable.

Parallel tasks of identical difficulty are constructed on both corpora, and the performance gap $\text{KA} = P_R - P_S$ precisely quantifies the contribution of parametric knowledge.

## Method

### Overall Architecture

The SynthWorlds generation pipeline proceeds in three stages:

1. **Universe Construction**: Connected subgraphs are sampled from the Wikidata knowledge graph, comprising relational triples (subject → relation → object).
2. **Entity Renaming**: All named entities are systematically renamed to synthetic counterparts while preserving type consistency (person → person, city → city, derived names remain consistent, e.g., University of Toronto → University of Metrovale).
3. **Document Generation**: An LLM generates documents from the synthetic triples; the corresponding RM documents are then produced via symbolic reference substitution.

The final output consists of two parallel corpora, each containing 6,290 documents, approximately 1.5 million tokens, 161K facts, 956 entity types, and 354 relation types.

### Key Design 1: Type-Consistent Entity Renaming

Renaming is not a simple random substitution; it preserves **semantic consistency** — surface forms must be compatible with the ontological type of each entity:
- Person names → person names (Geoffrey Hinton → Caleb Ardent)
- City names → city names (Toronto → Metrovale)
- Derived names remain consistent (University of Toronto → University of Metrovale, not University of Grandvale)
- Library names remain library names (Central Library → Oakwood Public Library, not Central Stadium)

This ensures that surface-form differences do not introduce additional signals between RM and SM corpora, so that performance differences genuinely reflect the role of parametric knowledge. Common-sense knowledge (e.g., "hospitals have doctors") and domain-general knowledge (e.g., laws of physics) are preserved; only entity-specific factual knowledge is eliminated.

### Key Design 2: Parallel Task Construction and Difficulty Control

Two types of parallel tasks are constructed on both corpora:

**Multi-hop QA**:
- Subgraphs matching reasoning motifs are sampled from the fact graph $G_{facts}$.
- Single-hop questions are generated for each triple and then composed into multi-hop questions.
- Difficulty is precisely controlled via 6 reasoning motifs spanning 2–4 hops.
- Each reasoning step is drawn from a different document, requiring cross-document reasoning.
- A total of 1,200 parallel QA pairs are produced.

**Page Navigation**:
- Symbolic references between documents serve as hyperlinks, forming a document graph $G_{doc}$.
- An agent must navigate from a source page to a target page by clicking links or backtracking.
- Expected random-walk distance serves as a difficulty proxy, bucketed into 5 difficulty levels (50–10M).
- A total of 1,000 parallel navigation pairs are produced.

### Key Design 3: Knowledge Advantage Gap Measurement Framework

A quantitative framework is defined as follows:
- **Baseline KA**: $\text{KA}^{base} = P_R^{base} - P_S^{base}$, the contribution of pure parametric knowledge.
- **Augmented KA**: $\text{KA}^{ext} = P_R^{ext} - P_S^{ext}$, the gap after knowledge augmentation.
- **Gap reduction**: $\text{KA}^{base} - \text{KA}^{ext}$, measuring how much the augmentation closes the gap.

In the baseline setting, $P_S^{base}$ approaches chance (parametric knowledge is inapplicable), so $\text{KA}^{base}$ directly reflects the degree to which models rely on memorization.

## Key Experimental Results

### Main Results

Six models are evaluated: GPT-5-mini, Gemini-2.0-Flash, gpt-oss-20B, gpt-oss-120B, Kimi-K2-Instruct, and Kimi-K2-Thinking.

**Multi-hop QA (F1 Score)**:

| Setting | GPT-5-mini RM | GPT-5-mini SM | KA |
|---------|:---:|:---:|:---:|
| Closed-book | ~20 | ~0 | **~20** |
| One-step RAG | Improved | Improved but less | Widened (−4.0) |
| IRCoT + RAG | Further improved | Substantially improved | **Narrowed (+5.2)** |
| Reading Comprehension | High | Comparable or higher | ~0 |

Key findings:
- SM accuracy is near zero in the closed-book setting, validating the synthetic world design.
- **One-step RAG widens the KA gap** — the gain for RM exceeds that for SM, indicating that the retriever itself depends on parametric knowledge.
- IRCoT + RAG, which interleaves retrieval and reasoning, narrows the gap: by 5.2 points for GPT-5-mini and 10.3 points for Gemini-2.0-Flash.

**Page Navigation (Success Rate)**:

| Setting | GPT-5-mini RM | GPT-5-mini SM | KA |
|---------|:---:|:---:|:---:|
| Links Only | High | Low | **~30** |
| Content + Links | High | Moderately improved | ~20.7 (narrowed by 9.3) |

- Providing page content substantially improves SM performance, yet a gap persists.
- Under the Links Only condition, 48% (GPT-5-mini) and 60% (Gemini-2.0-Flash) of reasoning steps reference entities absent from the current page, indicating reliance on external parametric knowledge.

### Ablation Study

**Effect of Reasoning Difficulty on KA**:

| Task Difficulty | QA 2-hop KA | QA 4-hop KA | Nav Easy KA | Nav Hard KA |
|----------------|:---:|:---:|:---:|:---:|
| Closed-book / Links Only | Larger | Smaller (RM also drops) | Smaller | Larger |
| With augmentation | Narrowed | Partially narrowed | Substantially narrowed | Partially narrowed |

- On simple QA tasks, the RM advantage is larger (direct recall is easier); on harder tasks, RM performance also degrades.
- In the Reading Comprehension setting, SM performance matches or exceeds RM, suggesting that parametric knowledge may **interfere** with context-based reasoning.
- On navigation tasks, harder paths yield larger KA — models rely on parametric knowledge to take "shortcuts."

## Highlights & Insights

1. **Elegant experimental design**: The parallel-world construction genuinely decouples reasoning from memorization in a way not previously achieved in the literature.
2. **Surprising and insightful findings**: The discovery that one-step RAG widens the KA gap reveals that LM-based retrievers are themselves dependent on parametric knowledge.
3. **Fully automatic and scalable framework**: New corpora can be generated on demand, preventing evaluation sets from being subsumed by model training data.
4. **Comprehensive coverage**: 6 models × multiple augmentation strategies × 2 task types.
5. **The Knowledge Advantage Gap framework** is clear and practically applicable to other evaluation scenarios.

## Limitations & Future Work

1. Validation is currently limited to corpora constructed from Wikidata; generalizability to other knowledge graphs or domains (e.g., code, mathematics) remains to be demonstrated.
2. Although synthetic names are type-consistent, they may introduce subtle distributional shifts (e.g., differing statistical properties of names) that affect some results.
3. The paper is categorized under robotics, but its content is squarely within LLM evaluation — a misclassification.

## Rating

⭐⭐⭐⭐⭐ — The experimental design is exceptionally elegant, achieving for the first time a controlled decoupling of reasoning and memorization in LLMs. The KA framework and the finding that one-step RAG widens the gap have far-reaching implications for the evaluation of RAG systems and agent architectures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ExoPredicator: Learning Abstract Models of Dynamic Worlds for Robot Planning](exopredicator_learning_abstract_models_of_dynamic_worlds_for_robot_planning.md)
- [\[ICML 2026\] WestWorld: Scalable Trajectory World Models with Knowledge Encoding](../../ICML2026/robotics/westworld_a_knowledge-encoded_scalable_trajectory_world_model_for_diverse_roboti.md)
- [\[ICLR 2026\] Experience-based Knowledge Correction for Robust Planning in Minecraft](experience-based_knowledge_correction_for_robust_planning_in_minecraft.md)
- [\[ICLR 2026\] RoboPARA: Dual-Arm Robot Planning with Parallel Allocation and Recomposition Across Tasks](robopara_dual-arm_robot_planning_with_parallel_allocation_and_recomposition_acro.md)
- [\[ICLR 2026\] JULI: Jailbreak Large Language Models by Self-Introspection](juli_jailbreak_large_language_models_by_self-introspection.md)

</div>

<!-- RELATED:END -->
