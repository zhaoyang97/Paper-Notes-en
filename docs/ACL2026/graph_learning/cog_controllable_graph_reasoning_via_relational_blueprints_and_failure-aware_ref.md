---
title: >-
  [Paper Note] CoG: Controllable Graph Reasoning via Relational Blueprints and Failure-Aware Refinement over Knowledge Graphs
description: >-
  [ACL 2026][Graph Learning][Knowledge Graph Question Answering] CoG is a training-free KGQA framework that implements Kahneman's Dual-Process Theory for KG reasoning. System 1 distills SPARQL from training sets into a "re…
tags:
  - "ACL 2026"
  - "Graph Learning"
  - "Knowledge Graph Question Answering"
  - "dual-process"
  - "relational blueprints"
  - "failure-aware backtracking"
  - "training-free agent"
date: 2026-05-08
content_hash: df9361e0651fad25
---

# CoG: Controllable Graph Reasoning via Relational Blueprints and Failure-Aware Refinement over Knowledge Graphs

**Conference**: ACL 2026  
**arXiv**: [2601.11047](https://arxiv.org/abs/2601.11047)  
**Code**: https://github.com/zjukg/CoG (Available)  
**Area**: Graph Learning / KG Reasoning  
**Keywords**: Knowledge Graph Question Answering, dual-process, relational blueprints, failure-aware backtracking, training-free agent  

## TL;DR
CoG is a training-free KGQA framework that implements Kahneman's Dual-Process Theory for KG reasoning. System 1 distills SPARQL from training sets into a "relational blueprint" template library offline, serving as soft structural constraints to guide the reranking and pruning of candidate relations online. System 2 triggers evidence-conditioned reflection and targeted backtracking when search stalls, correcting early erroneous decisions. On three multi-hop KGQA benchmarks (CWQ, WebQSP, GrailQA), it achieves SOTA accuracy (GPT-4 backbone: CWQ 77.8, WebQSP 89.7, GrailQA 86.4) while maintaining the lowest cost (13% fewer tokens and 12% fewer calls than PoG on CWQ).

## Background & Motivation

**Background**: The mainstream LLM-driven agent paradigm for KGQA (ToG / PoG / KG-Agent) follows a "plan → retrieve → generate" cycle, expanding evidence chains step-by-step from topic entities. However, these methods exhibit high instability in complex multi-hop settings due to interference from neighborhood noise.

**Limitations of Prior Work**: This instability is attributed to "cognitive rigidity," categorized into two types: (1) Error Cascading from Indiscriminate Exploration — choosing the wrong relation early on (e.g., selecting `contains` instead of `adjoins`) pulls the agent into a massive noisy candidate set, causing errors to snowball; (2) Structural Misalignment from Myopic Decisions — relying solely on local semantic matching leads to local optima (e.g., selecting `actor` instead of `director`), causing downstream constraints (runtime checks, temporal filters) to fail and forcing premature trajectory termination.

**Key Challenge**: Current agents lack a bridge between "local semantic relevance" and "global structural consistency across hops." They lack both empirical structural priors and the ability to diagnose or backtrack at dead-ends. Fine-tuning methods (RoG, KG-Agent) can learn structural priors at a high cost, while zero-shot agents are too unconstrained. Hard-constraint methods like GCR suffer from branch collapse if a single edge is missing in an incomplete KG.

**Goal**: (1) Introduce "cheap, interpretable" structural priors that constrain without locking the search direction; (2) Enable the agent to identify "where it went wrong" and backtrack when search stalls or evidence is insufficient; (3) Ensure the entire mechanism is training-free and independent of parameter fine-tuning.

**Key Insight**: Kahneman's Dual-Process Theory is mapped directly to KG reasoning: System 1 (fast, intuitive) handles blueprint-guided candidate filtering, while System 2 (slow, analytical) handles failure diagnosis and backtracking. This division of labor naturally separates "experience" from "reflection."

**Core Idea**: Relation-only blueprints are distilled offline from training set SPARQL queries as soft priors (storing relation chains without entities). Online, agents use these blueprints to rerank and prune candidate relations. Upon failure, evidence-conditioned reflection and targeted backtracking are activated, integrating "experience" and "reflection" into a single training-free framework.

## Method

### Overall Architecture
A three-stage pipeline (Figure 2):

1.  **Offline blueprint construction**: Parse training set SPARQL, strip entities to retain only relation sequences $\mathcal{S}(q)=\langle r_1,\ldots,r_L\rangle$, and deduplicate. For each unique template, the longest question is selected as a semantic anchor $q^*$ and encoded into a vector index using SentenceTransformer.
2.  **System 1 - Online blueprint-guided KG exploration**: Mask topic entities → retrieve top-$K$ neighboring blueprints → copy (if similarity $\ge \tau_{\text{copy}}$) or adapt via LLM → obtain a query-specific blueprint $S_{\text{BP}}=\langle r_1^{\text{BP}},\ldots,r_L^{\text{BP}}\rangle$. At each step $t$, candidate relations $\mathcal{R}_{\text{cand}}$ are collected from the frontier $E_{t-1}$ and reranked using three merged scores: local relevance $\phi_{\text{loc}}$, step-wise alignment $\phi_{\text{step}}$, and global compatibility $\phi_{\text{glob}}$. The LLM prunes the shortlist and explicitly includes the step-wise top-1 candidate (Structure-Consistency Safeguard).
3.  **System 2 - Failure-Aware Refinement**: Detect stagnation or insufficient evidence → invoke LLM for evidence-conditioned reflection on the trajectory $\mathcal{T}=[e_0,r_1,\ldots]$ to locate the erroneous decision point $t_{\text{err}}$ → backtrack the frontier, recall structural candidates that were prematurely pruned, and resume expansion. If evidence is completely missing, fallback to grounded inference (generating an answer using verified paths and unsatisfied constraints).

### Key Designs

1.  **Offline relational blueprint library + Hybrid Copy-Adapt Mechanism**:
    *   **Function**: Distills globally reusable "structural compasses" from training data and decides whether to copy or adapt based on similarity to avoid overfitting.
    *   **Mechanism**: Deterministic rules (regex) strip Freebase IDs and non-structural elements from SPARQL, leaving only relation sequences. During querying, the top-$K$ neighbors are retrieved; if similarity $\ge \tau_{\text{copy}}=0.92$, the top-1 is copied directly. Otherwise, the top-2 neighbors and the original question are sent to the LLM for generation or light rewriting.
    *   **Design Motivation**: Compressing 3,098 WebQSP queries into 569 templates (18.4%) and 44k GrailQA queries into 3.7k (8.3%) demonstrates that KG reasoning logic is more finite than natural language syntax. With $\tau_{\text{copy}}=0.92$, only 8.7% are copied while 91.3% are adapted, ensuring generalization.

2.  **Three-signal fusion blueprint-guided rerank + Structure-Consistency Safeguard**:
    *   **Function**: Considers local semantics, step-wise alignment, and global compatibility simultaneously, ensuring the LLM does not overlook structurally correct relations.
    *   **Mechanism**: Monotonic slot-alignment $\pi(t)=\arg\max_j \text{sim}(h(o_t), h(r_j^{\text{BP}}))$ finds the blueprint position. Reranking uses $\text{Score}(r)=\lambda_{\text{loc}}\phi_{\text{loc}}+\lambda_{\text{step}}\phi_{\text{step}}+\lambda_{\text{glob}}\phi_{\text{glob}}$ with weights $(0.6, 0.25, 0.15)$. The final set includes LLM-selected candidates and the top-1 step-wise aligned relation.
    *   **Design Motivation**: Local relevance alone falls into local optima, while global structure alone ignores sparse KG regions. The safeguard uses a dual-source selection to treat the LLM as a semantic expert and $\phi_{\text{step}}$ as a structural expert.

3.  **Failure-Aware Refinement (System 2) - Diagnosis + Targeted Backtracking + Grounded Inference**:
    *   **Function**: Switches to correction mode when exploration is stuck to avoid stochastic stagnation.
    *   **Mechanism**: The LLM reviews the trajectory $\mathcal{T}$ and pruned branches to pinpoint $t_{\text{err}}$. The agent rolls back the frontier and recalls previously pruned structurally relevant candidates. If edges are missing, the LLM synthesizes an answer based on verified path segments and remaining constraints.
    *   **Design Motivation**: Existing agents like ToG/PoG lack explicit diagnosis, leading to loops or hallucinations. System 2 is critical; ablation shows accuracy on CWQ drops from 66.9 to 58.5 (-8.4%) without it.

### Loss & Training
Completely training-free with no gradient updates: (1) Blueprint encoder uses a pre-trained SentenceTransformer without fine-tuning; (2) Agents use fixed LLM APIs (GPT-3.5 Turbo / GPT-4 / Qwen2.5-7B) with temperature 0.3; (3) Exploration depth is capped at 4.

## Key Experimental Results

### Main Results (Hits@1 / F1)

| Method | CWQ Hits@1 | CWQ F1 | WebQSP Hits@1 | WebQSP F1 | GrailQA Hits@1 | GrailQA Zero-shot |
|---|---|---|---|---|---|---|
| ToG (GPT-4) | 67.6 | 47.6 | 82.6 | 58.9 | 81.4 | 86.5 |
| PoG (GPT-4) | 75.0 | 42.1 | 87.3 | 59.8 | 84.7 | 88.6 |
| **CoG (GPT-4)** | **77.8** | **69.2** | **89.7** | **75.5** | **86.4** | **89.1** |
| ToG (GPT-3.5) | 57.1 | 41.9 | 76.2 | 50.9 | 68.7 | 72.7 |
| PoG (GPT-3.5) | 63.2 | 43.7 | 82.0 | 58.1 | 76.5 | 81.7 |
| **CoG (GPT-3.5)** | **66.9** | **59.9** | **86.8** | **74.3** | **79.2** | **83.6** |

CoG (GPT-4) F1 on CWQ is +27.1 percentage points higher than PoG, indicating better retrieval of complete answer sets.

### Ablation Study (CWQ Hits@1)

| Configuration | CWQ | WebQSP | GrailQA |
|---|---|---|---|
| **Full CoG** | **66.9** | **86.8** | **79.2** |
| w/o Failure-Aware Refinement | 58.5 | 79.9 | 75.3 |
| w/o Blueprint Guidance | 61.5 | 82.2 | 76.4 |
| w/o Blueprint Adaptation | 62.4 | 83.5 | 77.5 |

### Key Findings
- **System 2 is the most vital component**: Removing Failure-Aware Refinement results in an 8.4-point drop on CWQ, nearly 1.6x the impact of removing blueprint guidance (-5.4).
- **Strong Zero-shot Generalization**: On GrailQA's zero-shot split, the GPT-3.5 backbone achieves 83.6%, proving blueprints capture abstract reasoning patterns rather than rote memorization.
- **Efficiency Gains**: CoG requires 11.7 calls and 7,075 tokens on CWQ, compared to 13.3 calls and 8,156 tokens for PoG. It represents a Pareto improvement in both performance and cost.
- **Cross-KG Transferability**: Mapping Freebase entities to Wikidata QIDs shows CoG still leads PoG by 2.7 points on WebQSP.

## Highlights & Insights
- **Dual-Process Theory is an Apt Metaphor**: System 1 suppresses "early error amplification" through intuition, while System 2 resolves "local optima dead-ends" through analysis. This is more systematic than simple retries.
- **Zero-Cost Distillation**: The rule-based blueprint extraction requires no LLM calls or fine-tuning, allowing training data to inform the agent cheaply.
- **Dual-Source Selection Pattern**: Combining the LLM as a semantic expert with structural alignment as a safeguard is a transferable solution for RAG and tool selection.
- **Diagnostic Refinement over Blind Retries**: Case studies show PoG wasting 14k tokens on 26 retries at the same node, while CoG succeeds via a single diagnostic reroute.

## Limitations & Future Work
- **Limitations**: KG incompleteness provides a hard ceiling that refinement cannot bypass. Blueprint library coverage depends on the training set, which may be sparse for niche domains. Backtracking increases latency.
- **Future Work**: Upgrading linear blueprints to typed graph templates; introducing online learning for incremental blueprint evolution; using a learned detector for refinement triggers.

## Related Work & Insights
- **vs ToG**: ToG uses LLM-driven beam search. CoG adds structural soft guidance and self-correction, improving CWQ Hits@1 by 9.8 points with 27% fewer tokens.
- **vs PoG**: PoG uses heuristic retries. CoG uses evidence-conditioned reflection for targeted fixes, avoiding infinite loops.
- **vs GCR / KG-Tries**: GCR uses hard constraints. CoG uses soft blueprints and refinement, making it more robust to missing KG edges.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First systematic application of Dual-Process Theory to KG agents. 
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 3 datasets, 3 backbones, efficiency metrics, and transfer tests.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and well-integrated metaphors.
- **Value**: ⭐⭐⭐⭐ Highly practical for industrial KGQA systems due to being training-free and Pareto-efficient.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion](../../ICML2026/graph_learning/generative_representation_learning_on_hyper-relational_knowledge_graphs_via_mask.md)
- [\[AAAI 2026\] PCoKG: Personality-aware Commonsense Reasoning with Debate](../../AAAI2026/graph_learning/pcokg_personality-aware_commonsense_reasoning_with_debate.md)
- [\[ACL 2026\] Which bird does not have wings: Negative-constrained KGQA with Schema-guided Semantic Matching and Self-directed Refinement](which_bird_does_not_have_wings_negative-constrained_kgqa_with_schema-guided_sema.md)
- [\[NeurIPS 2025\] Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs](../../NeurIPS2025/graph_learning/deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)
- [\[ICLR 2026\] Relational Graph Transformer](../../ICLR2026/graph_learning/relational_graph_transformer.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2025\] Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models](../../ICML2025/graph_learning/graph-constrained_reasoning_faithful_reasoning_on_knowledge_graphs_with_large_la.md)
- [\[ICML 2026\] Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion](../../ICML2026/graph_learning/generative_representation_learning_on_hyper-relational_knowledge_graphs_via_mask.md)
- [\[AAAI 2026\] PCoKG: Personality-aware Commonsense Reasoning with Debate](../../AAAI2026/graph_learning/pcokg_personality-aware_commonsense_reasoning_with_debate.md)
- [\[ACL 2026\] Which bird does not have wings: Negative-constrained KGQA with Schema-guided Semantic Matching and Self-directed Refinement](which_bird_does_not_have_wings_negative-constrained_kgqa_with_schema-guided_sema.md)
- [\[ICLR 2026\] Relational Graph Transformer](../../ICLR2026/graph_learning/relational_graph_transformer.md)

</div>

<!-- RELATED:END -->
