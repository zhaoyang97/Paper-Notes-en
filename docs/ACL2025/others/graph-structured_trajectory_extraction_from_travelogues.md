---
title: >-
  [Paper Note] Graph-Structured Trajectory Extraction from Travelogues
description: >-
  [ACL 2025][trajectory extraction] This paper proposes the "Visiting Order Graph" to unify the geographic containment hierarchy and chronological transition relations in travel trajectories. It constructs the ATD-VSO benchmark dataset covering 100 Japanese travelogues (consisting of 3,354 geographic entities and 3,369 relations). Baseline experiments reveal that geographic inclusion relation prediction ($F1=0.355$) is the primary bottleneck, pointing out a key direction for in…
tags:
  - "ACL 2025"
  - "trajectory extraction"
  - "travelogue"
  - "graph structure"
  - "visiting order"
  - "geographic inclusion"
date: 2026-05-08
content_hash: 65fa2d5a33b57cf6
---

# Graph-Structured Trajectory Extraction from Travelogues

**Conference**: ACL 2025  
**arXiv**: [2410.16633](https://arxiv.org/abs/2410.16633)  
**Code**: To be released  
**Area**: Information Extraction / Tourism Informatics  
**Keywords**: trajectory extraction, travelogue, graph structure, visiting order, geographic inclusion

## TL;DR

This paper proposes the "Visiting Order Graph" to unify the geographic containment hierarchy and chronological transition relations in travel trajectories. It constructs the ATD-VSO benchmark dataset covering 100 Japanese travelogues (consisting of 3,354 geographic entities and 3,369 relations). Baseline experiments reveal that geographic inclusion relation prediction ($F1=0.355$) is the primary bottleneck, pointing out a key direction for integrating geographic knowledge in this field.

## Background & Motivation

Travelogues are important resources for analyzing human travel behavior, finding extensive applications in tourism informatics (travel recommendation, tour planning) and human geography. Automatically extracting travel trajectories from travelogues is of great value, yet it faces two key challenges:

First, **insufficient trajectory representation**. Existing studies represent trajectories as sequences of locations (Ishino et al. 2012; Wagner et al. 2023), but sequences fail to characterize geographic inclusion relations—for instance, "Kyoto City" contains "Kyoto Station". Instead of being simply arranged in a sequence at the same level, they exhibit a nested hierarchical relationship.

Second, **lack of open benchmark datasets**. Prior works rely on private datasets, which prevents fair comparison and research replication, severely hindering research accumulation in this field.

This work addresses both issues simultaneously: it proposes a graph-structured trajectory representation and constructs a high-quality, publicly available benchmark dataset.

## Method

### Overall Architecture

The trajectory extraction task is decomposed into two cascaded subtasks: (1) **Visiting Status Prediction (VSP)**: determining whether each geographic entity mentioned in a travelogue was actually visited by the traveler; (2) **Visiting Order Prediction (VOP)**: constructing a visiting order graph among the visited entities, which includes Inclusion Relation Prediction (IRP) and Transition Relation Prediction (TRP).

### Key Designs

1. **Visiting Order Graph**:

    - **Function**: Unifying the geographic hierarchy and chronological order in trajectories.
    - **Mechanism**: Nodes represent geographic entities, and edges represent two types of directed relations: **Inclusion** (e.g., "Nara City" contains "Todai-ji") and **Transition** (e.g., the traveler moves directly from "Nara Station" to "Todai-ji"). Transition relations are only established between sibling entities sharing the same parent node.
    - **Design Motivation**: By restricting transition relations to be established only among sibling nodes, the visiting order between any two entities can be inferred by traversing the inclusion and transition relations—even if they are not directly connected (e.g., the order between "Kyoto Station" and "Nara City" can be indirectly determined via the transition between their respective parents "Kyoto City" $\rightarrow$ "Nara City").
    - **Special Case Handling**: Multiple visits (splitting entities into sub-entities), temporal ambiguity (excluding with an UnknownTime tag), and geographic overlaps (using an Overlap relation, such as "Honshu" and "Tokyo Prefecture").

2. **Fine-Grained Visiting Status Label System**:

    - **Function**: Finely distinguishing different types of "mentions".
    - **Mechanism**: Entity-level 2-label (Visit/Other) + Mention-level 6-label (Visit/PlanToVisit/See/Visit-Past/Visit-Future/UnkOrNotVisit). Entity labels are derived using the Mention Label Aggregation (MLA) rule: if at least one mention is labeled as Visit or PlanToVisit, the entity is labeled as Visit.
    - **Design Motivation**: Places are mentioned in diverse ways in travelogues—"arrived at Nara Station" (Visit) vs. "JR Nara Station is a bit far from Kintetsu Nara Station" (factual statement, does not imply a visit) vs. "want to visit Kiyomizu-dera next time" (Visit-Future). This necessitates a fine-grained distinction.

3. **Sequence Sorting Decoding**:

    - **Function**: Ensuring global consistency in transition relation prediction.
    - **Mechanism**: Constrained decoding based on greedy search—selecting the highest-scoring transition pair $\langle e_a, e_b \rangle$ from all candidate entity pairs, then pruning all conflicting pairs (reverse pairs, other predecessors of $e_b$, other successors of $e_a$) to guarantee that sibling nodes under the same parent node form a unique, valid sequence.
    - **Design Motivation**: Naive pairwise prediction may yield inconsistent results (such as $A \rightarrow B$ and $B \rightarrow A$ both holding true), which requires global constraints.

### Baseline Systems

- **VSP**: LUKE (LukeForEntityClassification) + Mention Label Aggregation (MLA) rules.
- **IRP**: LUKE (LukeForEntityPairClassification) to predict the parent entity of each entity.
- **TRP**: LUKE + Sequence Sorting Decoding to predict subsequently visited entities.
- **Simple Rule Baselines**: Majority Label (for VSP), Random/Flat (for IRP), and OccOrder (for TRP, ordering by textual occurrence).

## Key Experimental Results

### Main Results: Visiting Status Prediction (VSP)

| Level | Method | Accuracy | Macro F1 |
|------|------|----------|----------|
| Mention-level | Majority Label | 0.679 | 0.135 |
| Mention-level | LUKE | **0.789** | **0.468** |
| Entity-level | Majority Label | 0.823 | 0.451 |
| Entity-level | LUKE + MLA | **0.862** | **0.740** |

F1 for each label: Visit reaches 0.872–0.918; UnkOrNotVisit is only 0.482–0.561 (the most common error is misclassifying it as Visit).

### Main Results: Inclusion Relation Prediction (IRP)

| Method | All F1 | Depth=1 F1 | Depth $\ge$ 2 F1 |
|------|--------|-----------|-----------|
| Random | 0.043 | 0.057 | 0.038 |
| Flat (all predicting ROOT) | 0.244 | **1.000** | 0 |
| **LUKE** | **0.355** | 0.058 | **0.425** |

### Main Results: Transition Relation Prediction (TRP)

| Method | All F1 | Forward F1 | Reverse F1 |
|------|--------|-----------|-----------|
| Random | 0.190 | 0.247 | 0.061 |
| OccOrder (visit status) | **0.758** | 0.794 | **0.386** |
| LUKE (Naive Decoding) | 0.680 | 0.737 | 0.298 |
| LUKE (Sequence Sorting Decoding) | 0.748 | **0.796** | 0.366 |

### Ablation Study

| Configuration | Key Metrics | Note |
|------|---------|------|
| LUKE Naive Decoding vs. Sequence Sorting Decoding | TRP F1: 0.680 $\rightarrow$ 0.748 (+0.068) | Constrained decoding effectively eliminates inconsistent predictions. |
| OccOrder-EM vs. OccOrder-VS | TRP F1: 0.730 $\rightarrow$ 0.758 | Representative mention selection based on visiting status outperforms the earliest occurring mention. |
| Depth=1 vs. Depth $\ge$ 2 (IRP) | F1: 0.058 vs. 0.425 | Determining whether an entity is at the top level (having no parent) is extremely difficult. |

### Key Findings

1. **VSP is relatively mature**: The F1 score for the Visit label reaches 0.872–0.918, indicating that identifying "visited" locations is fairly straightforward.
2. **IRP is the core bottleneck**: The overall F1 is only 0.355 because the pre-trained LUKE model lacks containment knowledge between geographical entities—knowledge such as "Nara City contains Todai-ji" requires external geographic knowledge injection.
3. **Textual order $\approx$ visiting order**: The simple occurrence order rule (OccOrder) achieves an F1 of 0.758, even outperforming LUKE naive decoding (0.680). This suggests that the description order in travelogues is highly consistent with the actual visiting order.
4. **Reverse relations are the most challenging**: The F1 score for predicting travelers backtracking routes (mentioning B first but actually visiting A first) is only 0.298–0.386.
5. **High inter-annotator agreement (IAA)**: Entity-level IAA F1 = 0.89, $\kappa$ = 0.81; relation-level F1 = 0.85.

## Highlights & Insights

- The **Visiting Order Graph** unifies two inherently different types of relations (spatial hierarchy vs. temporal sequence). This graph-structured design represents a conceptual contribution to the field.
- Restricting transition relations to sibling nodes at the same level is highly elegant, enabling the visiting order between any two nodes to be inferred via graph traversal, which guarantees the completeness of the representation.
- Rigorous dataset construction process: using the online whiteboard tool Miro to annotate inclusion/transition relations is much more intuitive than pure text-based annotation.
- The discovery of the strong baseline effect of textual occurrence order suggests that future improvements should focus on handling "non-sequential description" cases (such as backtracking or plans).
- Clear future directions for geographic knowledge injection: GeoLM pre-training, geocoding features (coordinates/boundary shapes), and geographic knowledge graphs.

## Limitations & Future Work

- **Small dataset scale**: Only 100 Japanese travelogues, leaving generalizability to be validated. It needs to be extended to other languages and larger scales.
- **Cascaded evaluation setup**: Currently using gold labels as inputs for downstream tasks (e.g., IRP uses gold-standard visiting status), without evaluating end-to-end error propagation.
- **Weak baseline models**: Relying solely on LUKE without attempting advanced LLMs such as GPT-4 or Llama for relation extraction, which might possess stronger geographic common sense.
- The performance of IRP is severely insufficient (F1=0.355), which is still far from practical application.
- Cross-document trajectory merging (e.g., multiple travelogues describing the same destination) remains unaddressed.
- Culturally specific to Japanese travelogues, where language-specific stylistic features (e.g., honorifics, omitted subjects) may affect the cross-lingual transferability of the method.

## Related Work & Insights

- **vs. Ishino et al. (2012) / Wagner et al. (2023)**: Previous works adopt sequence representations for trajectories, which fail to capture geographic inclusion relations. The proposed graph-structured representation offers a qualitative leap in expressiveness.
- **vs. ATD-MCL (Higashiyama et al. 2023)**: This work builds upon the geographic entity annotations of ATD-MCL by adding visiting status and visiting order annotations, forming a complete trajectory extraction benchmark.
- **vs. GeoLM (Li et al. 2023)**: GeoLM pre-training integrates geospatial information, representing a promising direction for improving IRP.
- **vs. General Relation Extraction**: The relations in this work (inclusion/transition) are domain-specific—inclusion relations require geographic knowledge rather than pure textual inference.

## Rating

- Novelty: ⭐⭐⭐⭐ Graph-structured trajectory representation is a meaningful conceptual innovation; the task decomposition (VSP $\rightarrow$ IRP $\rightarrow$ TRP) is clear and reasonable.
- Experimental Thoroughness: ⭐⭐⭐ The baseline methods are relatively simple (only LUKE), lacking LLM baselines and end-to-end experiments; however, it is sufficient as a dataset contribution paper.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, diagrams are intuitive, and the annotation process is described in detail.
- Value: ⭐⭐⭐ The dataset and benchmark are valuable for the tourism NLP subdomain, though the application scope is relatively narrow; the identified bottleneck in IRP guides future research directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] DRS: Deep Question Reformulation With Structured Output](drs_deep_question_reformulation_with_structured_output.md)
- [\[ACL 2025\] Mapping the Podcast Ecosystem with the Structured Podcast Research Corpus](mapping_the_podcast_ecosystem_with_the_structured_podcast_research_corpus.md)
- [\[NeurIPS 2025\] FACE: Faithful Automatic Concept Extraction](../../NeurIPS2025/others/face_faithful_automatic_concept_extraction.md)
- [\[ACL 2025\] TARGA: Targeted Synthetic Data Generation for Practical Reasoning over Structured Data](targa_targeted_synthetic_data_generation_for_practical_reasoning_over_structured.md)
- [\[ACL 2025\] STRICTA: Structured Reasoning in Critical Text Assessment for Peer Review and Beyond](stricta_structured_reasoning_in_critical_text_assessment_for_peer_review_and_bey.md)

</div>

<!-- RELATED:END -->
