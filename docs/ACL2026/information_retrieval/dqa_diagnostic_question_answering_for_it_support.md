---
title: >-
  [Paper Note] DQA: Diagnostic Question Answering for IT Support
description: >-
  [ACL 2026][Information Retrieval & RAG][Diagnostic Question Answering] This paper proposes the DQA framework, which achieves systematic fault diagnosis in enterprise IT support by maintaining persistent diagnostic states…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Diagnostic Question Answering"
  - "IT Support"
  - "RAG"
  - "Root Cause Analysis"
  - "Diagnostic State Tracking"
date: 2026-05-08
content_hash: 41e9c95e3623d03d
---

# DQA: Diagnostic Question Answering for IT Support

**Conference**: ACL 2026
**arXiv**: [2604.05350](https://arxiv.org/abs/2604.05350)
**Code**: None
**Area**: Information Retrieval / Dialogue Systems
**Keywords**: Diagnostic Question Answering, IT Support, RAG, Root Cause Analysis, Diagnostic State Tracking

## TL;DR
This paper proposes the DQA framework, which achieves systematic fault diagnosis in enterprise IT support by maintaining persistent diagnostic states and aggregating retrieved evidence at the root-cause level rather than processing documents individually. The success rate improves from a baseline of 41.3% to 78.7%, while the average number of turns decreases from 8.4 to 3.9.

## Background & Motivation

**Background**: Enterprise IT support interactions are inherently diagnostic — users submit vague symptom reports, and support agents must iteratively gather evidence to identify the root cause. Retrieval-Augmented Generation (RAG) is the predominant knowledge grounding approach, and multi-turn RAG further improves retrieval robustness through conversational query rewriting.

**Limitations of Prior Work**: Standard multi-turn RAG systems lack an explicit representation of diagnostic state. Retrieved documents are consumed independently at each turn, making it difficult to accumulate evidence across turns, reconcile conflicting signals, or maintain awareness of unresolved hypotheses. Retrieval over large-scale ticket repositories also yields numerous near-duplicate results, wasting context window capacity and increasing latency.

**Key Challenge**: Diagnostic conversations require tracking competing hypotheses, interpreting partial signals, and deciding when to ask questions versus when to propose a solution. However, existing RAG systems conflate "conversational coherence" with "diagnostic progress," lacking explicit modeling of diagnostic advancement.

**Goal**: To design a fault-diagnosis framework that maintains explicit diagnostic states, aggregates evidence at the root-cause level, and supports state-conditioned action selection.

**Key Insight**: The framework draws inspiration from Case-Based Reasoning (CBR) — learning from similar resolved cases — but rather than adapting individual cases, it aggregates distributional information (e.g., cluster prevalence) across the entire retrieved neighborhood to guide action selection.

**Core Idea**: Retrieved tickets are clustered by root-cause description, and a hypothesis weight vector is maintained as the diagnostic state. This vector is dynamically updated with each new piece of evidence, guiding a strategic transition from broad inquiry to targeted investigation to solution proposal.

## Method

### Overall Architecture
DQA consists of four core components: (1) RAggG (Retrieval-Aggregated Generation), which aggregates retrieval results at the root-cause level; (2) retrieval-induced diagnostic state tracking that monitors the support level for competing hypotheses; (3) action-aware diagnostic strategy that guides clarification questions, investigation steps, or solution proposals; and (4) state-conditioned response generation. Each dialogue turn triggers: query rewriting → retrieval → aggregation → state update → action selection → response generation.

### Key Designs

1. **RAggG: Retrieval-Aggregated Generation**

    - **Function**: Clusters a large number of retrieved tickets by root cause and compresses them into compact diagnostic signals, replacing per-document processing.
    - **Mechanism**: Given a user description, the top-$K$ similar tickets are retrieved; sentence embeddings of the resolution fields are computed, followed by clustering (mini-batch k-means or hierarchical clustering). Each cluster represents a candidate root cause, yielding aggregated evidence $\mathcal{E} = \{(n_j, R_j)\}_{j=1}^{J}$, where $n_j$ is the evidence count and $R_j$ is a representative case. The query-conditioned hypothesis distribution is $h_k = \frac{n_k(x)}{\sum_{k'} n_{k'}(x)}$.
    - **Design Motivation**: Standard RAG returns numerous near-duplicate tickets that waste context window capacity. Aggregation preserves distributional information (e.g., which root causes are most prevalent) rather than merely deduplicating, providing stronger signals for downstream action selection.

2. **Retrieval-Induced Diagnostic State**

    - **Function**: Persistently tracks the support level, collected evidence, and symptoms for each candidate root cause across turns.
    - **Mechanism**: A structured state $s_t$ is maintained, containing a hypothesis weight vector $\mathbf{h}_t \in \mathbb{R}^K$ (each element corresponding to a root-cause cluster), along with associated symptoms, KB articles, and representative solutions. The state is updated each turn via re-retrieval and re-aggregation: retrieval-induced weights are recomputed from fresh evidence, while structured state fields persist across turns.
    - **Design Motivation**: Unlike explicit probabilistic inference, DQA implicitly updates beliefs through re-retrieval, avoiding the complexity of hand-crafted probabilistic models while remaining responsive to current evidence.

3. **Action-Aware Diagnostic Strategy**

    - **Function**: Selects the appropriate action type based on the diagnostic state — clarification questions, investigation steps, or solution proposals.
    - **Mechanism**: Fault diagnosis is modeled as a policy over three diagnostic action types: clarification (collecting discriminative evidence), investigation (verifying likely causes), and solution proposal (suggesting a fix when uncertainty has sufficiently decreased). As evidence accumulates and support concentrates on a small number of root causes, the strategy automatically transitions from broad inquiry to targeted investigation and resolution.
    - **Design Motivation**: Unconstrained free-text generation cannot explicitly reflect diagnostic progress. Categorizing actions into three types makes the diagnostic process trackable and interpretable.

### Loss & Training
DQA is a system-level design evaluated using a replay-based protocol. Evaluation is conducted on 150 anonymized enterprise IT support scenarios, each involving multi-turn interactions between a user simulator and the DQA agent.

## Key Experimental Results

### Main Results

| Method | Success Rate | Avg. Turns |
|--------|-------------|------------|
| Multi-turn RAG Baseline | 41.3% | 8.4 |
| DQA | **78.7%** | **3.9** |

### Ablation Study

| Configuration | Success Rate | Note |
|---------------|-------------|------|
| DQA (Full) | 78.7% | Complete framework |
| w/o Aggregation | ~55% | Significant degradation without root-cause aggregation |
| w/o Diagnostic State | ~50% | No cross-turn state tracking |
| w/o Action Strategy | ~60% | No explicit action selection |

### Key Findings
- DQA nearly doubles the success rate (41.3% → 78.7%) while reducing the average number of turns by more than half (8.4 → 3.9).
- Root-cause-level aggregation is more effective than per-document retrieval, as it compresses redundancy while retaining distributional signals.
- Explicit diagnostic state enables the system to accumulate evidence across turns and avoid repetitive questioning.
- The action strategy transition (inquiry → investigation → resolution) naturally corresponds to changes in diagnostic confidence.

## Highlights & Insights
- **Paradigm Shift from Document Retrieval to Root-Cause Aggregation**: Conventional RAG operates at the document level, whereas DQA elevates retrieval to aggregate at the semantic concept (root-cause) level. This approach is generalizable to any retrieval scenario requiring structured insights from large collections of similar cases.
- **Implicit Belief Updating**: Diagnostic state is updated through per-turn re-retrieval and re-aggregation, avoiding the complexity of explicit probabilistic models. This constitutes a "retrieval-as-reasoning" strategy.
- **Formalization of Diagnostic Actions**: Constraining open-ended dialogue to three action types renders system behavior interpretable and controllable.

## Limitations & Future Work
- Evaluation relies on a replay protocol over 150 anonymized scenarios, which may not fully reflect real-world deployment performance.
- Clustering quality depends on the quality of ticket resolution fields; noisy or incomplete resolution descriptions may degrade performance.
- The current strategy employs three manually defined action types; future work could explore learned policies.
- Latency and scalability issues associated with integration into real-time systems are not discussed.

## Related Work & Insights
- **vs. Standard Multi-turn RAG**: Multi-turn RAG improves retrieval robustness but does not represent diagnostic state. DQA explicitly tracks hypotheses and evidence.
- **vs. Case-Based Reasoning (CBR)**: CBR adapts from a small number of cases, whereas DQA aggregates distributional information from a large retrieved neighborhood.
- **vs. Medical Diagnostic Dialogue**: The underlying uncertainty-reduction logic is similar, but IT support scenarios exhibit greater heterogeneity and faster-changing failure patterns.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of root-cause aggregation and diagnostic state tracking is a novel design within RAG systems.
- Experimental Thoroughness: ⭐⭐⭐ Improvements are substantial, but the evaluation scale of 150 scenarios is relatively small.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear and the method is systematically designed.
- Value: ⭐⭐⭐⭐ Directly applicable to enterprise IT support scenarios; the aggregation approach is broadly generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits](mab-dqa_addressing_query_aspect_importance_in_document_question_answering_with_m.md)
- [\[ACL 2026\] CounterRefine: Answer-Conditioned Counterevidence Retrieval for Inference-Time Knowledge Repair in Factual Question Answering](counterrefine_answer-conditioned_counterevidence_retrieval_for_inference-time_kn.md)
- [\[ACL 2026\] Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches](is_agentic_rag_worth_it_an_experimental_comparison_of_rag_approaches.md)
- [\[AAAI 2026\] N2N-GQA: Noise-to-Narrative for Graph-Based Table-Text Question Answering Using LLMs](../../AAAI2026/information_retrieval/n2n-gqa_noise-to-narrative_for_graph-based_table-text_question_answering_using_l.md)
- [\[AAAI 2026\] MAVIS: A Benchmark for Multimodal Source Attribution in Long-form Visual Question Answering](../../AAAI2026/information_retrieval/mavis_a_benchmark_for_multimodal_source_attribution_in_long-form_visual_question.md)

</div>

<!-- RELATED:END -->
