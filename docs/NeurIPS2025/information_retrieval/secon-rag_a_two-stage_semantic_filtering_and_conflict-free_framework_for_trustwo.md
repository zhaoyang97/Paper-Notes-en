---
title: >-
  [Paper Note] SeCon-RAG: A Two-Stage Semantic Filtering and Conflict-Free Framework for Trustworthy RAG
description: >-
  [NeurIPS 2025][Information Retrieval & RAG][RAG] This paper proposes SeCon-RAG, a two-stage defense framework. The first stage employs clustering combined with semantic graph filtering to remove poisoned documents…
tags:
  - "NeurIPS 2025"
  - "Information Retrieval & RAG"
  - "RAG"
  - "corpus poisoning defense"
  - "semantic filtering"
  - "conflict-aware reasoning"
  - "entity-intent-relation"
date: 2026-05-08
content_hash: ea7028db2d2a7dc2
---

# SeCon-RAG: A Two-Stage Semantic Filtering and Conflict-Free Framework for Trustworthy RAG

**Conference**: NeurIPS 2025
**arXiv**: [2510.09710](https://arxiv.org/abs/2510.09710)  
**Code**: [GitHub](https://github.com/) (mentioned in paper)  
**Area**: NLP Understanding / RAG Security / Adversarial Robustness
**Keywords**: RAG, corpus poisoning defense, semantic filtering, conflict-aware reasoning, entity-intent-relation

## TL;DR
This paper proposes SeCon-RAG, a two-stage defense framework. The first stage employs clustering combined with semantic graph filtering to remove poisoned documents, while the second stage performs conflict-aware filtering at inference time. SeCon-RAG comprehensively outperforms existing RAG defense methods across 5 LLMs and 3 QA datasets, maintaining high accuracy and near-zero attack success rates even under 100% poisoning rates.

## Background & Motivation

**Background**: RAG has become a mainstream paradigm for enhancing LLM factual accuracy by incorporating external documents. However, RAG systems are highly dependent on external corpora, making them inherently vulnerable to corpus poisoning and retrieval contamination attacks.

**Limitations of Prior Work**:
   - Existing defenses such as TrustRAG apply clustering to detect poisoned documents, but rely solely on statistical features in vector space, constituting coarse-grained filtering that tends to incorrectly remove legitimate documents whose topics overlap with adversarial ones.
   - Majority voting mechanisms fail under high poisoning rates.
   - Methods such as AstuteRAG use heuristic rules to fuse internal and external knowledge but do not resolve semantic conflicts among retrieved documents or between documents and the model's parametric knowledge.
   - Explicit detection and filtering of conflicting information is absent at the inference stage.

**Key Challenge**: Aggressive filtering risks losing valuable information and reducing recall, whereas lenient filtering retains poisoned content and reduces reliability—a fundamental trade-off between information preservation and security filtering.

**Goal**: (1) Replace coarse-grained statistical filtering with fine-grained semantic information to precisely identify poisoned documents; (2) Introduce conflict detection at the inference stage to ensure that final generation is grounded in consistent and trustworthy knowledge.

**Key Insight**: The paper introduces a structured semantic extraction module, EIRE, which extracts entity, intent, and relation triples from documents to construct semantic graphs. Semantic graphs of clean documents exhibit dense and connected structures, whereas those of poisoned documents are sparse and fragmented—a structural discrepancy that enables precise identification.

**Core Idea**: Incorporate structured semantic information (entity-intent-relation) into both the retrieval and inference stages of RAG defense, enabling fine-grained poisoning detection and conflict resolution.

## Method

### Overall Architecture
SeCon-RAG operates in two stages. **Stage 1, SCF (Semantic and Clustering-Based Filtering)**, applies dual filtering—clustering and semantic graph analysis—to candidate documents during retrieval to remove poisoned content. **Stage 2, CAF (Conflict-Aware Filtering)**, performs semantic consistency verification on documents that pass Stage 1, filtering out residual conflicts at inference time. Both stages share a common semantic extraction module, EIRE.

### Key Designs

1. **EIRE (Entity-Intent-Relation Extractor)**:

    - Function: Extracts structured semantic triples $(E_d, I_d, R_d)$ from documents, corresponding respectively to entity sets, intents, and inter-entity relations.
    - Mechanism: A prompt-driven LLM performs structured information extraction, converting free-text documents into comparable semantic frames. For instance, given a document about a pilot, EIRE extracts "Nungesser" (entity), "answering who piloted the aircraft" (intent), and "Nungesser → piloted → L'Oiseau Blanc" (relation).
    - Design Motivation: Provides a unified structured representation for downstream semantic graph comparison and conflict detection, removing reliance on uninterpretable vector distances.

2. **SCF Stage 1: Joint Clustering and Semantic Graph Filtering**:

    - Function: Cleans the corpus prior to top-k retrieval by removing poisoned documents.
    - **Clustering Filter**: Documents are embedded in vector space and clustered via K-means; documents whose cosine similarity to the cluster centroid exceeds threshold $\tau_{\text{cluster}}$ are filtered out (poisoned documents tend to cluster tightly due to template-based generation).
    - **Semantic Graph Filter**: EIRE constructs a semantic graph $G_d = (V_i, E_{ij})$ for each candidate document, with nodes as entity embeddings and edges as semantic relations. Each candidate graph is compared against a small set of manually verified correct document graphs $G_{\text{cor}}$, and an LLM evaluates the graph structural similarity score $ssG(d, D_{\text{cor}}) \in [0,1]$. Clean documents yield dense and connected graphs, while poisoned documents yield sparse and fragmented ones.
    - **Conservative AND Logic**: A document is removed only when flagged by both filters simultaneously: $\widetilde{\mathcal{D}} = \mathcal{D}' \setminus (\mathcal{D}_{\text{cluster}} \cap \mathcal{D}_{\text{semantic}})$, maximizing retention of valuable information.
    - Design Motivation: Clustering captures statistical anomalies (template-based poisoning), while semantic graphs capture content anomalies (semantically incoherent poisoning); the two are complementary.

3. **CAF Stage 2: Conflict-Aware Filtering**:

    - Function: Performs final filtering on top-k documents that pass SCF at inference time.
    - Mechanism: EIRE extracts semantic information from each candidate document, and trustworthiness is assessed along three dimensions:
        - **Q (Query Consistency)**: Whether the document's intent and entities align with the user query.
        - **C (Corpus Consistency)**: Whether relational contradictions exist between the document and other retrieved documents.
        - **M (Model Consistency)**: Whether the document's key entities are consistent with the LLM's parametric knowledge.
    - The LLM classifies each document as poisoned / conflicting / irrelevant / trustable; only trustable documents are retained for final generation.
    - Design Motivation: SCF removes statistical and structural anomalies, but may retain documents that are semantically relevant yet factually contradictory (e.g., conflicting with the model's parametric knowledge). CAF provides fine-grained factual-level filtering.

## Key Experimental Results

### Main Results
Evaluation is conducted on HotpotQA, NQ, and MS-MARCO across 5 LLMs (Mistral-12B, Qwen-7B, LLaMA-3.1-8B, GPT-4o, DeepSeek-R1) under four settings: Clean / PIA / PoisonedRAG-20% / PoisonedRAG-100%.

| Model + Dataset | Setting | SeConRAG ACC/ASR | TrustRAG ACC/ASR | Gain |
|----------------|---------|-----------------|-----------------|------|
| GPT-4o + HotpotQA | 100% Poisoning | 83.6% / 2.4% | 80.9% / 2.7% | +2.7% ACC |
| GPT-4o + NQ | 100% Poisoning | 81.8% / 0.0% | 80.0% / 0.1% | +1.8% ACC |
| GPT-4o + MS-MARCO | PIA | 93.6% / 0.0% | 89.1% / 1.3% | +4.5% ACC |
| DeepSeek-R1 + NQ | 100% Poisoning | 96.4% / 0.0% | 88.2% / 0.0% | +8.2% ACC |
| DeepSeek-R1 + MS-MARCO | Clean | 94.0% | 91.0% | +3.0% ACC |
| LLaMA-3.1-8B + MS-MARCO | 100% Poisoning | 89.1% / 0.0% | 84.5% / 6.4% | +4.6% ACC |

### Ablation Study

| Configuration | HotpotQA (100% Poisoning) ACC/ASR | Notes |
|--------------|----------------------------------|-------|
| Full SeCon-RAG | 74.0% / 8.0% | Complete model (Mistral-12B) |
| w/o SCF | 71.0% / 25.0% | Removing Stage 1; ASR spikes +17% |
| w/o CAF | 68.0% / 56.0% | Removing Stage 2; ASR spikes +48% |
| Clustering filter only | Slightly below joint | Lacks semantic graph complement |
| Semantic filter only | Slightly below joint | Lacks clustering statistical complement |
| w/o EIRE | ACC degraded | Loss of fine-grained semantic reasoning |

### Key Findings
- **CAF is the core module**: Removing CAF causes ASR to surge from 8% to 56%, demonstrating that conflict filtering at the inference stage is critical for defense.
- **Clustering and semantic filtering are complementary**: Either filter alone underperforms their combination; the AND logic effectively filters poisoned content while preserving information.
- **Robust across embedding models**: Accuracy remains >75% and ASR <10% across MiniLM, SimCSE, BERT, and BGE embeddings.
- **Threshold insensitivity**: Performance is stable within $\tau_{\text{cluster}} \in [0.86, 0.90]$ and $\tau_{\text{semantic}} \in [0.2, 0.4]$, with ±2% variance.
- **Moderate runtime overhead**: 1.21–1.45 minutes per batch, approximately 10 seconds slower than TrustRAG but with significantly improved robustness.

## Highlights & Insights
- **The observation of semantic graph structural discrepancy is insightful**: Clean documents yield dense and connected EIRE semantic graphs, while poisoned documents yield sparse and fragmented ones—this structural signal is more interpretable than vector distances and harder for adversaries to circumvent.
- **Conservative AND logic**: Filtering only documents flagged by both filters simultaneously reduces false positives while maintaining high detection rates, representing a practical engineering design.
- **The three-dimensional consistency check is transferable**: The Q/C/M verification dimensions (query alignment, inter-document consistency, and model knowledge consistency) can be applied to any NLP system requiring multi-source information fusion.

## Limitations & Future Work
- **Computational overhead**: EIRE requires LLM inference to extract semantic triples from each candidate document, and SCF's semantic graph comparison also involves LLM calls, which may become a bottleneck at corpus scale.
- **Dependence on a small set of verified documents**: SCF's semantic filtering requires 10 manually verified correct documents as a reference baseline, which may be difficult to obtain in cold-start scenarios.
- **Adaptive attacks not evaluated**: Current experiments use standard poisoning strategies; if adversaries are aware of EIRE's semantic graph structure and deliberately construct densely connected semantic graphs, defense performance may degrade.
- **Reliability of CAF's model knowledge dimension**: When the model's parametric knowledge is itself erroneous, the M-dimension check may produce adverse effects.

## Related Work & Insights
- **vs TrustRAG**: TrustRAG relies solely on clustering for coarse-grained filtering combined with simple consistency voting. SeCon-RAG extends this with semantic graph filtering and inference-stage conflict detection, showing clear advantages under high poisoning rates.
- **vs AstuteRAG**: AstuteRAG fuses internal and external knowledge heuristically without explicit conflict detection. SeCon-RAG's CAF module provides a more systematic conflict-aware mechanism.
- **vs InstructRAG**: InstructRAG uses self-synthesized rationales to guide retrieval but remains vulnerable under high poisoning rates. SeCon-RAG's two-stage design demonstrates greater robustness.

## Rating
- Novelty: ⭐⭐⭐⭐ The observation regarding semantic graph structural discrepancy is inspiring, though the overall framework represents a combination of known techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 5 LLMs × 3 datasets × 4 attack settings, with detailed ablations, embedding model analysis, and threshold sensitivity analysis—highly comprehensive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] HiFi-RAG: Hierarchical Content Filtering and Two-Pass Generation for Open-Domain RAG](hifi-rag_hierarchical_content_filtering_and_two-pass_generation_for_open-domain_.md)
- [\[NeurIPS 2025\] Retrieval is Not Enough: Enhancing RAG Reasoning through Test-Time Critique and Optimization](retrieval_is_not_enough_enhancing_rag_reasoning_through_test-time_critique_and_o.md)
- [\[NeurIPS 2025\] RAG-IGBench: Innovative Evaluation for RAG-based Interleaved Generation in Open-domain Question Answering](rag-igbench_innovative_evaluation_for_rag-based_interleaved_generation_in_open-d.md)
- [\[NeurIPS 2025\] RMIT-ADM+S at the MMU-RAG NeurIPS 2025 Competition](rmit-adms_at_the_mmu-rag_neurips_2025_competition.md)
- [\[NeurIPS 2025\] Worse than Zero-shot? A Fact-Checking Dataset for Evaluating the Robustness of RAG Against Misleading Retrievals](worse_than_zero-shot_a_fact-checking_dataset_for_evaluating_the_robustness_of_ra.md)

</div>

<!-- RELATED:END -->
