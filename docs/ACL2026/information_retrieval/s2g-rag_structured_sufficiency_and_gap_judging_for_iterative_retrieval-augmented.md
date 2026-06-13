---
title: >-
  [Paper Note] S2G-RAG: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA
description: >-
  [ACL2026][Information Retrieval & RAG][Iterative Retrieval] S2G-RAG explicitly models "evidence sufficiency" and "next-step gaps" in iterative RAG as a structured controller…
tags:
  - "ACL2026"
  - "Information Retrieval & RAG"
  - "Iterative Retrieval"
  - "Sufficiency Judgment"
  - "Information Gap"
  - "Evidence Compression"
  - "Multi-hop QA"
date: 2026-05-08
content_hash: 572a81223b94e5de
---

# S2G-RAG: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA

**Conference**: ACL2026  
**arXiv**: [2604.23783](https://arxiv.org/abs/2604.23783)  
**Code**: Not declared in the paper  
**Area**: Information Retrieval / RAG / Multi-hop QA  
**Keywords**: Iterative Retrieval, Sufficiency Judgment, Information Gap, Evidence Compression, Multi-hop QA

## TL;DR
S2G-RAG explicitly models "evidence sufficiency" and "next-step gaps" in iterative RAG as a structured controller, S2G-Judge. Combined with gap-guided queries and sentence-level evidence extraction to reduce noise, it improves F1 on HotpotQA from 43.3 (SIM-RAG) to 56.5 under the BM25 setting.

## Background & Motivation
**Background**: RAG has become the standard solution for knowledge-intensive QA. While single-hop questions are typically resolved with one retrieval step, multi-hop questions require multiple rounds of retrieval and intermediate reasoning. Existing methods include IR-CoT, FLARE, Self-RAG, SIM-RAG, and RAG-Critic, which control retrieval through reasoning traces, uncertainty, reflection tokens, or critic signals.

**Limitations of Prior Work**: The critical bottleneck of iterative RAG is control rather than single-step retrieval. Systems must decide in each round whether the current evidence memory is sufficient; if not, they must specify what to search for next. When control is unreliable, models either answer prematurely from incomplete chains or continue retrieving, accumulating redundant and distractor text.

**Key Challenge**: Free-text reasoning is flexible but difficult to audit and stabilize into next-hop retrieval requirements. Moreover, multiple rounds of retrieval lead to increasingly long contexts, causing subsequent judgments and answers to be affected by distractors.

**Goal**: The authors aim to transform retrieval control into an explicit, diagnostic, and trainable structured prediction problem while controlling context bloating through sentence-level evidence extraction.

**Key Insight**: S2G-RAG designs a lightweight S2G-Judge that outputs a sufficiency decision and structured gap items in each round. Gap items are mapped to the next query, and retrieved documents are processed by a sentence-level extractor to append only key sentences to the evidence memory.

**Core Idea**: The RAG controller is tasked with first answering "is it enough?" and, if not, structurally identifying "what information is missing," thereby turning multi-hop retrieval from a free-form drift into an auditable gap-filling process.

## Method
S2G-RAG can be understood as a judge-first iterative RAG. It does not modify the search engine or retrain the answer generator; instead, it inserts a structured control layer and an evidence extraction layer between the retriever and the reasoner to prevent memory bloat.

### Overall Architecture
Given a question $q$ and a corpus $D$, the system maintains an accumulated evidence context $C_t$. Initial $C_0$ is empty. In each round, S2G-Judge reads $(q, C_t)$ and outputs a binary sufficiency flag $s_t$ and a set of gap items $G_t$. If $s_t$ is true, retrieval stops and the answer reasoner is called. If false, a query is constructed based on $G_t$ to retrieve top-k documents, from which an evidence extractor selects sentence-level evidence blocks $E_t$ to append to $C_t$.

In the default configuration, the answer reasoner is Llama-3-8B-Instruct, the S2G-Judge is a LoRA fine-tuned Llama-3.2-3B-Instruct, with a maximum of 4 retrieval rounds, top-k=6 per round, and title-based de-duplication. Sparse retrieval uses BM25, and dense retrieval uses E5-base-v2.

### Key Designs
1.  **Structured Sufficiency and Gap Judging**:
    - Function: Enables the controller to explicitly judge whether current evidence is sufficient and identify missing information if it is not.
    - Mechanism: Each gap item contains four fields: `category`, `target`, `slot`, and `description`. Categories include bridge_entity, attribute, relation, evidence_span, and other. If sufficiency is true, the gap set is empty.
    - Design Motivation: Common failures in multi-hop QA involve finding a bridge entity but lacking attributes, or finding an entity but lacking relational evidence. Structured gaps are easier to convert into stable retrieval queries than free-text reflections.

2.  **Distilling Process Supervision from Execution Trajectories**:
    - Function: Trains a lightweight judge to make decisions based on real multi-round intermediate states rather than idealized evidence chains.
    - Mechanism: The authors run the iterative retrieval process and record $(q, C_t)$ for each round. A strong teacher (GPT-4o-mini) labels sufficiency and gap items under a context-only constraint. Conflicting samples with low confidence are filtered, and S2G-Judge is trained via LoRA SFT.
    - Design Motivation: Intermediate contexts in real iterative RAG often contain redundancy and distractors. Distilling from execution trajectories allows the judge to adapt to the state distribution encountered in practice.

3.  **Gap-aware Sentence-level Evidence Context**:
    - Function: Avoids concatenating full documents into the context over multiple rounds to control noise and latency.
    - Mechanism: Retrieved documents are split into sentence candidates with title sources. An LLM extractor outputs sentence indices based on the question and gap items. The system maps these back to original sentences and appends them to the context. The extractor performs pointer selection rather than rewriting.
    - Design Motivation: Pointer-based extraction preserves provenance and reduces hallucinated evidence; gap items help the extractor prioritize sentences that fill the current gap.

### Loss & Training
S2G-Judge is trained using standard autoregressive cross-entropy on structured outputs. The input is $(q, C_t)$, and the target is the sufficiency and gap schema labeled by the teacher. Supervision comes from multi-round trajectories rather than manual step-by-step labels. During inference, query construction prioritizes concatenating the `target` and `slot` of a gap item, falling back to the `description` if missing. By default, only the first valid gap item is used to keep queries concise.

## Key Experimental Results

### Main Results
Main results report EM/F1 on TriviaQA, HotpotQA, and 2WikiMultiHopQA, comparing BM25 and E5 retrieval settings.

| Retrieval | Method | TriviaQA EM/F1 | HotpotQA EM/F1 | 2Wiki EM/F1 | Main Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BM25 | IR-CoT | 56.9 / 68.9 | 28.6 / 41.5 | 23.5 / 32.4 | Multi-hop reasoning baseline |
| BM25 | SIM-RAG | 70.7 / 75.6 | 32.7 / 43.3 | 34.1 / 40.2 | Strong critic control |
| BM25 | S2G-RAG | 72.0 / 77.9 | 43.3 / 56.5 | 41.7 / 48.6 | Maximum multi-hop gain |
| E5 | Standard RAG | 58.8 / 68.3 | 25.1 / 35.3 | 10.6 / 21.0 | Single-hop retrieval insufficient |
| E5 | RAG-Critic | 65.0 / 75.9 | 40.0 / 51.2 | 27.9 / 34.0 | Strong learned-control baseline |
| E5 | S2G-RAG | 71.1 / 78.0 | 42.0 / 53.5 | 39.0 / 45.3 | Effective across retrievers |

Compared to SIM-RAG, S2G-RAG achieves gains of +10.6 EM and +13.2 F1 on HotpotQA (BM25) and +7.6 EM and +8.4 F1 on 2Wiki. Gains on TriviaQA (primarily single-hop) are smaller but present, indicating that sentence-level context mitigates noise from additional retrieval.

### Ablation Study

| Variant | HotpotQA EM | HotpotQA F1 | Description |
| :--- | :--- | :--- | :--- |
| Full S2G-RAG | 43.3 | 56.5 | Complete system |
| w/o SFT | 39.2 | 50.8 | Using zero-shot judge; -4.1 EM / -5.7 F1 |
| w/o S2G-Judge | 27.5 | 37.6 | Largest drop without structured control |
| w/o Extractor | 39.5 | 52.5 | Direct full-text concatenation; increased noise/latency |

| Evidence Compression | EM | F1 | Ratio | Description |
| :--- | :--- | :--- | :--- | :--- |
| LLM summarization | 36.9 | 48.1 | 0.3816 | Summaries lose/rewrite evidence |
| ReComp extractive | 41.4 | 53.8 | 0.4948 | Better fidelity but lower than ours |
| ReComp abstractive | 34.9 | 46.4 | 0.1917 | Shortest but significant QA loss |
| Sentence Pointer | 43.3 | 56.5 | 0.3461 | Best accuracy-compression trade-off |

### Key Findings
- S2G-Judge is the most critical module. Removing it causes HotpotQA F1 to drop from 56.5 to 37.6, demonstrating that multi-hop retrieval control is more important than simply retrieving more documents.
- Sufficiency judgment is conservative; the false positive rate is 6.44%, but 31.60% of samples that already satisfy retrieval truth are still judged as insufficient. This reduces the risk of premature answering but suggests room for calibration.
- Sentence-level evidence context provides 4.5x-6.4x compression compared to full-text concatenation. On HotpotQA (BM25), per-sample latency dropped from 1.9552s to 1.6085s while F1 improved.

## Highlights & Insights
- "Sufficiency + Gap" is an effective interface for iterative RAG. It is more diagnostic than simple query rewriting and more constrained than free-form generator reflection.
- The use of trajectory distillation is practical. The controller observes intermediate evidence memory generated by real system rollouts rather than idealized labels, aligning it with deployment states.
- Sentence pointer extraction balances compression and auditability. Unlike abstractive summarization, it does not rewrite evidence, reducing the risk of hallucination during compression.
- The paper demonstrates a modular RAG direction: retriever, controller, extractor, and reasoner can be decoupled and improved independently without forcing all capabilities into a single LLM.

## Limitations & Future Work
- The gap schema sacrifices expressivity for stability. Complex questions may require multi-entity joins, temporal constraints, or more structured programs that the four-field schema might not cover.
- Sentence-level extraction involves a compactness-recall trade-off, potentially missing cross-sentence evidence or disambiguation context.
- The sufficiency judgment is conservative, which increases retrieval rounds and latency despite reducing premature answers. Better calibration is needed.
- The system is not trained end-to-end using RL or actor-critic methods. Future work could integrate S2G outputs as auxiliary rewards or latent variables in learned retrieval strategies.

## Related Work & Insights
- **vs IR-CoT**: IR-CoT uses intermediate reasoning as retrieval cues, whereas S2G-RAG explicitly predicts gap fields, making retrieval needs more structured.
- **vs FLARE / Self-RAG**: These methods control retrieval via uncertainty or reflection tokens; S2G-RAG's sufficiency and gap schemas are easier to audit and train.
- **vs SIM-RAG**: SIM-RAG evaluates if a draft answer is acceptable. S2G-RAG goes further by requiring the identification of missing information to generate the next query.
- **vs RAG-Critic**: RAG-Critic provides error feedback and correction. S2G-RAG is more lightweight, emphasizing structured control of the multi-round evidence state.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The sufficiency/gap schema is straightforward but addresses a core bottleneck in iterative RAG.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers three datasets, two retriever types, multiple strong baselines, and extensive ablations. Inclusion of real-world long-document tasks would have been more complete.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear progression from motivation and system design to analysis experiments.
- Value: ⭐⭐⭐⭐⭐ Highly relevant for engineering multi-hop RAG systems, particularly regarding auditable control and evidence memory compression.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] UnIte: Uncertainty-based Iterative Document Sampling for Domain Adaptation in Information Retrieval](unite_uncertainty-based_iterative_document_sampling_for_domain_adaptation_in_inf.md)
- [\[ACL 2026\] CRAFT: Training-Free Cascaded Retrieval for Tabular QA](craft_training-free_cascaded_retrieval_for_tabular_qa.md)
- [\[ACL 2026\] VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG](videostir_understanding_long_videos_via_spatio-temporally_structured_and_intent-.md)
- [\[ACL 2026\] Disco-RAG: Discourse-Aware Retrieval-Augmented Generation](disco-rag_discourse-aware_retrieval-augmented_generation.md)
- [\[ACL 2026\] Navigating Large-Scale Document Collections: MuDABench for Multi-Document Analytical QA](navigating_large-scale_document_collections_mudabench_for_multi-document_analyti.md)

</div>

<!-- RELATED:END -->
