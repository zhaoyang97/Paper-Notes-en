---
title: >-
  [Paper Note] DQA: Diagnostic Question Answering for IT Support
description: >-
  [ACL 2026][Information Retrieval & RAG][Diagnostic Question Answering] This paper proposes the DQA framework, which achieves systematic troubleshooting in corporate IT support scenarios by maintaining persistent diagnost…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Diagnostic Question Answering"
  - "IT Support"
  - "RAG"
  - "Root Cause Analysis"
  - "Diagnostic State Tracking"
date: 2026-05-08
content_hash: 0c8c91233dd70df2
---

# DQA: Diagnostic Question Answering for IT Support

**Conference**: ACL 2026  
**arXiv**: [2604.05350](https://arxiv.org/abs/2604.05350)  
**Code**: None  
**Area**: Information Retrieval / Dialogue Systems  
**Keywords**: Diagnostic Question Answering, IT Support, RAG, Root Cause Analysis, Diagnostic State Tracking

## TL;DR
This paper proposes the DQA framework, which achieves systematic troubleshooting in corporate IT support scenarios by maintaining persistent diagnostic states and aggregating retrieval evidence at the root-cause level (rather than processing documents individually). The success rate increases from a 41.3% baseline to 78.7%, and the average number of turns decreases from 8.4 to 3.9.

## Background & Motivation

**Background**: Corporate IT support interactions are inherently diagnostic—users submit vague symptom reports, and support agents need to iteratively collect evidence to identify root causes. Retrieval-Augmented Generation (RAG) is the mainstream knowledge grounding method, and multi-turn RAG further improves retrieval robustness via dialogue query rewriting.

**Limitations of Prior Work**: Standard multi-turn RAG systems lack explicit diagnostic state representations. Retrieved documents are consumed independently in each turn, making it difficult to accumulate evidence across turns, reconcile conflicting signals, or maintain awareness of unresolved hypotheses. Large-scale ticket repository retrieval also produces numerous near-duplicate redundant results, wasting context windows and latency budgets.

**Key Challenge**: Diagnostic dialogues require tracking competing hypotheses, interpreting partial signals, and deciding when to ask questions vs. when to provide solutions. However, existing RAG systems conflate "dialogue coherence" with "diagnostic progress" and lack explicit modeling of diagnostic progress.

**Goal**: Design a troubleshooting framework that maintains explicit diagnostic states, aggregates evidence at the root-cause level, and supports state-based action selection.

**Key Insight**: Drawing inspiration from Case-Based Reasoning (CBR)—learning from similar resolved cases, but instead of adapting a single case, aggregating distributional information (such as cluster prevalence) of the entire retrieval neighborhood to guide action selection.

**Core Idea**: Cluster retrieved tickets by root-cause descriptions and maintain a hypothesis-weight vector as the diagnostic state, which is dynamically updated with new evidence each turn to guide a strategy shift from "broad questioning" to "precise troubleshooting" and then to "proposing solutions."

## Method

### Overall Architecture
DQA consists of four core components: (1) RAggG (Retrieval-Aggregated Generation) aggregates retrieval results at the root-cause level; (2) Retrieval-induced diagnostic state tracks the support for competing hypotheses; (3) Action-aware diagnostic strategy guides clarification questions, investigation steps, or solution proposals; (4) State-conditioned response generation. Each dialogue turn triggers: Query Rewriting → Retrieval → Aggregation → State Update → Action Selection → Response Generation.

### Key Designs

1. **RAggG: Retrieval-Aggregated Generation**:

    - **Function**: Clusters a large number of retrieved tickets by root cause, compressing them into compact diagnostic signals to replace document-by-document processing.
    - **Mechanism**: Given a user description, retrieve Top-K similar tickets, encode the `resolution` fields using a sentence vector encoder, and then cluster (using mini-batch k-means or hierarchical clustering). Each cluster represents a candidate root cause, outputting aggregated evidence $\mathcal{E} = \{(n_j, R_j)\}_{j=1}^{J}$, where $n_j$ is the evidence count and $R_j$ is a representative case. The query-conditioned hypothesis distribution is $h_k = \frac{n_k(x)}{\sum_{k'} n_{k'}(x)}$.
    - **Design Motivation**: The large volume of near-duplicate tickets returned by standard RAG wastes context windows. Aggregation preserves distributional information (e.g., which root cause is most common) rather than simple deduplication, providing stronger signals for downstream action selection.

2. **Retrieval-induced Diagnostic State**:

    - **Function**: Tracks the support for each candidate root cause, collected evidence, and symptoms persistently across turns.
    - **Mechanism**: Maintains a structured state $s_t$, including a hypothesis weight vector $\mathbf{h}_t \in \mathbb{R}^K$ (each element corresponding to a root-cause cluster), as well as associated symptoms, KB articles, and typical solutions. The state is updated each turn through re-retrieval and re-aggregation; retrieval-induced weights are recomputed from fresh evidence, while structured state fields persist across turns.
    - **Design Motivation**: Unlike explicit probabilistic reasoning, DQA updates beliefs implicitly through re-retrieval. This avoids the complexity of manually designing probabilistic models while remaining responsive to current evidence.

3. **Action-aware Diagnostic Strategy**:

    - **Function**: Selects appropriate action types—clarification questions, investigation steps, or solution proposals—based on the diagnostic state.
    - **Mechanism**: Models troubleshooting as a policy over three diagnostic actions: Clarification Questions (collecting discriminative evidence), Investigation Steps (verifying potential causes), and Solution Proposals (proposing fixes when uncertainty decreases). As evidence accumulates and support converges on a few root causes, the strategy automatically shifts from broad questioning to precise investigation and resolution.
    - **Design Motivation**: Unconstrained free-text generation cannot clearly reflect diagnostic progress. Categorizing actions into three types makes the diagnostic advancement process traceable and interpretable.

### Loss & Training
DQA is a system-level design using a playback-based evaluation protocol. The evaluation is based on 150 anonymized corporate IT support scenarios, each involving multi-turn interactions between a user simulator and the DQA agent.

## Key Experimental Results

### Main Results

| Method | Success Rate | Average Turns |
|------|--------|---------|
| Multi-turn RAG Baseline | 41.3% | 8.4 |
| DQA | **78.7%** | **3.9** |

### Ablation Study

| Configuration | Success Rate | Description |
|------|--------|------|
| DQA Full | 78.7% | Complete framework |
| w/o Aggregation | ~55% | Significant degradation after removing root-cause aggregation |
| w/o Diagnostic State | ~50% | No cross-turn state tracking |
| w/o Action Strategy | ~60% | No explicit action selection |

### Key Findings
- DQA nearly doubles the success rate (41.3% → 78.7%) while reducing average turns by more than half (8.4 → 3.9).
- Root-cause level aggregation is more effective than document-by-document retrieval because it compresses redundant information while preserving distributional signals.
- Explicit diagnostic states allow the system to accumulate evidence between turns, avoiding repetitive questioning.
- Transitions in action selection (Question → Investigate → Resolve) naturally correspond to changes in diagnostic confidence.

## Highlights & Insights
- **Paradigm shift from document retrieval to root-cause aggregation**: Traditional RAG operates at the document level, while DQA elevates aggregation to the level of semantic concepts (root causes). This idea can be generalized to any retrieval scenario requiring structural insights from many similar cases.
- **Implicit belief updating**: Updating diagnostic states via per-turn re-retrieval + re-aggregation avoids the complexity of explicit probabilistic models. This is a "retrieval-as-inference" strategy.
- **Formalization of diagnostic actions**: Constraint-based dialogue using three action types makes system behavior interpretable and controllable.

## Limitations & Future Work
- Evaluation based on a playback protocol with 150 anonymized scenarios is small and may not fully reflect actual deployment performance.
- Clustering quality depends on the quality of ticket `resolution` fields; noisy or incomplete resolution descriptions may affect performance.
- The current strategy uses three manually defined action types; future work could explore learned policies.
- Latency and scalability issues for integration with real-time systems were not discussed.

## Related Work & Insights
- **vs Standard Multi-turn RAG**: Multi-turn RAG improves retrieval robustness but does not represent diagnostic states. DQA explicitly tracks hypotheses and evidence.
- **vs Case-Based Reasoning (CBR)**: CBR adapts from a few cases, whereas DQA aggregates distributional information from a large neighborhood.
- **vs Medical Diagnostic Dialogue**: Similar uncertainty reduction logic, but IT scenarios have higher heterogeneity and faster-changing failure modes.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of root-cause aggregation and diagnostic states is a novel design in RAG systems.
- Experimental Thoroughness: ⭐⭐⭐ Significant performance gains, but the evaluation scale of 150 scenarios is small.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear, and the method design is systematic.
- Value: ⭐⭐⭐⭐ Directly useful for corporate IT support scenarios; the aggregation concept is generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits](mab-dqa_addressing_query_aspect_importance_in_document_question_answering_with_m.md)
- [\[ACL 2026\] FinRAG-12B: A Production-Validated Recipe for Grounded Question Answering in Banking](finrag-12b_a_production-validated_recipe_for_grounded_question_answering_in_bank.md)
- [\[ACL 2026\] ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering](chatr1_reinforcement_learning_for_conversational_reasoning_and_retrieval_augment.md)
- [\[ACL 2026\] CounterRefine: Answer-Conditioned Counterevidence Retrieval for Inference-Time Knowledge Repair in Factual Question Answering](counterrefine_answer-conditioned_counterevidence_retrieval_for_inference-time_kn.md)
- [\[ACL 2026\] Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches](is_agentic_rag_worth_it_an_experimental_comparison_of_rag_approaches.md)

</div>

<!-- RELATED:END -->
