---
title: >-
  [Paper Note] Ontology-Guided Reverse Thinking Makes Large Language Models Stronger on Knowledge Graph Question Answering
description: >-
  [ACL 2025][Graph Learning][Knowledge Graph Question Answering] Proposes ORT (Ontology-Guided Reverse Thinking), which leverages the ontology structure of Knowledge Graphs to construct label reasoning paths backward from the target to guide knowledge retrieval, significantly enhancing the KGQA capabilities of LLMs.
tags:
  - "ACL 2025"
  - "Graph Learning"
  - "Knowledge Graph Question Answering"
  - "Ontology-Guided"
  - "Reverse Reasoning"
  - "LLM"
  - "Label Reasoning Path"
date: 2026-05-08
content_hash: 5258e29dc65628f2
---

# Ontology-Guided Reverse Thinking Makes Large Language Models Stronger on Knowledge Graph Question Answering

**Conference**: ACL 2025  
**arXiv**: [2502.11491](https://arxiv.org/abs/2502.11491)  
**Code**: None  
**Area**: Graph Learning  
**Keywords**: Knowledge Graph Question Answering, Ontology-Guided, Reverse Reasoning, LLM, Label Reasoning Path

## TL;DR

Proposes ORT (Ontology-Guided Reverse Thinking), which leverages the ontology structure of Knowledge Graphs to construct label reasoning paths backward from the target to guide knowledge retrieval, significantly enhancing the KGQA capabilities of LLMs.

## Background & Motivation

The core challenge of Knowledge Graph Question Answering (KGQA) lies in efficiently establishing reasoning paths from question conditions to answer targets. Existing methods primarily fall into two categories: (1) Fine-tuning methods (e.g., RoG, KD-CoT), which require a large amount of high-quality training data and are computationally expensive; (2) Embedding + Search methods (e.g., MindMap, Think-on-Graph), which rely on entity vector matching and graph traversal but fail to handle conceptual targets (e.g., "stadium" is a concept rather than a concrete entity in the Knowledge Graph).

The core problem is that traditional forward reasoning methods are entity-centric, expanding neighbors after finding entities through vector matching. However, the targets of questions are often abstract and difficult to match directly with concrete entities. For instance, when asking for the venue of "1995 Rugby World Cup", "stadium" is merely a concept label, and entity matching cannot directly locate "Ellis Park Stadium".

Inspired by human reverse thinking, the authors propose constructing reasoning paths backward from the target to the known conditions, utilizing the ontology structure of the Knowledge Graph to achieve concept-level reasoning navigation.

## Method

### Overall Architecture

ORT consists of three phases: (1) extracting conditions, targets, and their labels from the question; (2) constructing reverse label reasoning paths based on the Knowledge Graph ontology; (3) utilizing the label reasoning paths to guide knowledge retrieval and generate answers.

### Key Designs

1. **Aim and Condition Recognition**: Utilizes an LLM to extract the condition entities $\mathcal{C}_E$, condition labels $\mathcal{C}_L$, target entities $\mathcal{A}_E$, and target labels $\mathcal{A}_L$ from the question. By providing a Label List of the Knowledge Graph, the LLM is guided to map the entities in the question to the labels in the Knowledge Graph ontology, addressing the limitations of pure vector matching.

2. **Neighbor Label Dictionary Construction**: Traverses relation triples in the Knowledge Graph ontology, collecting all other labels appearing in the same triple for each label $l_i$, to construct a neighbor label dictionary $\mathcal{D} = \{l_i: \mathcal{N}(l_i)\}$. This serves as the fundamental data structure for constructing the reverse reasoning tree.

3. **Reverse Reasoning Tree Construction**: Takes the target labels $\mathcal{A}_L$ as root nodes (unifying multiple target labels via a virtual root), and recursively queries the neighbor label dictionary to expand child nodes until reaching the maximum recursion depth (determined by the number of question hops). This backward construction starts from the target and proceeds towards the conditions, naturally filtering out a vast number of irrelevant paths.

4. **Triple Pruning Strategy**:

    - **Prune by Conditions**: Performs DFS traversal on all paths to remove those that do not contain any condition label; for paths containing condition labels, it only retains the segment up to the last condition label.
    - **Prune Cycle Sub-paths**: Detects and removes cycles in the paths using DFS, preventing infinite loops caused by bidirectional relations.
    - **Prune by Semantics**: Inverts the remaining paths to forward paths and feeds them along with the question into an LLM, which filters out paths beneficial for answering the question, eliminating semantically irrelevant and distracting paths.

5. **Guided Answer Mining**: Utilizes the label reasoning paths to guide forward queries on the Knowledge Graph. Starting from the condition nodes, it progressively queries neighbor entities satisfying the label constraints along the label path to construct an entity reasoning path tree. Finally, all entity paths are collected via DFS and fed into the LLM to aggregate and generate the final answer.

### Loss & Training

ORT is a plug-and-play method that requires no fine-tuning, directly leveraging the capabilities of existing LLMs. It mainly relies on several LLM prompts: condition/target extraction, semantic pruning, and answer generation.

## Key Experimental Results

### Main Results

| Dataset | Metric | ORT (DeepSeek-v3) | RoG (Fine-tuned) | MindMap | Pure LLM (DeepSeek-v3) |
|--------|------|------|----------|------|------|
| WebQSP | Hit@1 | **89.43** | 85.7 | 64.92 | 64.0 |
| WebQSP | F1 | **71.83** | 70.8 | 47.14 | 43.9 |
| CWQ | Hit@1 | **72.91** | 62.6 | 48.83 | 41.12 |
| CWQ | F1 | **62.63** | 56.2 | 43.30 | 33.80 |

### Ablation Study

| Configuration | WebQSP Hit@1 | CWQ Hit@1 | Description |
|------|---------|------|------|
| Full ORT | 89.43 | 72.91 | All components |
| w/o LLM Filter | 86.58 | 62.58 | Removes semantic pruning, CWQ drops by 10+ |
| Trace Forward | 77.82 | 60.73 | Forward reasoning replaces reverse reasoning |
| w/o Rules | 64.00 | 41.12 | No label paths constructed, degrades to pure LLM |

### Key Findings

- ORT outperforms the fine-tuned method RoG without any fine-tuning (WebQSP: 89.43 vs 85.7).
- ORT consistently brings a 25%+ improvement in Hit@1 across three different LLMs (GPT-4o, DeepSeek-v3, Qwen-max).
- Reverse reasoning outperforms forward reasoning by 11.61% in Hit@1 on WebQSP, demonstrating the effectiveness of reverse thinking.
- All methods perform generally worse on the CWQ dataset (which contains more multi-hop questions) than on WebQSP, indicating that multi-hop reasoning remains challenging.

## Highlights & Insights

- **Innovative Application of Reverse Thinking**: First to introduce human reverse thinking into KGQA. Reasoning backward from targets is highly more efficient than searching forward from conditions, inherently filtering out a significant number of irrelevant paths.
- **Utilization of Ontology Structure**: Conducts reasoning at the label/concept level using the KG ontology, addressing the limitations of traditional methods that perform matching purely at the entity-level.
- **Plug-and-Play**: Requires no fine-tuning. As a general enhancement strategy, it directly boosts the KGQA capabilities of various LLMs.
- **Multi-layer Pruning Strategy**: The combination of condition, cycle, and semantic pruning ensures both the quality and efficiency of the reasoning paths.

## Limitations & Future Work

- When querying the KG, if too many entities satisfy the label constraints, a large volume of irrelevant results may be introduced.
- Feeding all entity paths into the LLM may introduce distracting information, thereby reducing answer accuracy.
- Highly dependent on the label extraction capability of the LLM; label identification may be inaccurate for complex questions.
- Only validated on the Freebase KG; the transferability to other knowledge graphs (e.g., Wikidata) remains to be verified.

## Related Work & Insights

- Compared to forward-search methods like MindMap, the search space for reverse reasoning is significantly reduced.
- Utilizing the ontology structure adds a new dimension to KGQA, bridging the abstract intent of questions and the structured data of knowledge graphs.
- Compared to fine-tuning methods like RoG, ORT achieves superior performance without fine-tuning, suggesting that leveraging structural information in KGs is more critical than optimizing model parameters.
- **Insight**: The concept of reverse reasoning can be extended to other structured reasoning tasks, such as database query optimization, causal reasoning, and planning problems.
- WebQSP is dominated by single-hop questions (65.49%), while CWQ contains more multi-hop questions (20.75% $\ge$ 3 hops); the discrepancy in hop distribution across datasets influences method performance.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of reverse thinking and KG ontology is a novel and intuitively sound idea.
- Experimental Thoroughness: ⭐⭐⭐⭐ Uses two standard datasets, multiple LLMs, and detailed ablation studies, but lacks validation on more diverse knowledge graphs.
- Writing Quality: ⭐⭐⭐⭐ Method descriptions are clear, diagrams are intuitive, and the algorithm pseudocode is comprehensive.
- Value: ⭐⭐⭐⭐ Plug-and-play and highly effective, offering significant practical value to the KGQA field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering](fidelis_faithful_reasoning_in_large_language_model_for_knowledge_graph_question_.md)
- [\[ACL 2025\] The Role of Exploration Modules in Small Language Models for Knowledge Graph Question Answering](the_role_of_exploration_modules_in_small_language_models_for_knowledge_graph_que.md)
- [\[ACL 2025\] Agent Steerable Search for Knowledge Graph Question Answering](agent_steerable_search_for_knowledge_graph_question_answering.md)
- [\[ACL 2025\] Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study Over Open-ended Question Answering](kg_llm_trustworthy_qa.md)
- [\[ICML 2026\] KBQA-R1: Reinforcing Large Language Models for Knowledge Base Question Answering](../../ICML2026/graph_learning/kbqa-r1_reinforcing_large_language_models_for_knowledge_base_question_answering.md)

</div>

<!-- RELATED:END -->
