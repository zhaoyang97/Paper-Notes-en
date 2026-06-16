---
title: >-
  [Paper Note] S2G-RAG: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA
description: >-
  [ACL 2026][Information Retrieval & RAG][Paper Note] S2G-RAG explicitly models "evidence sufficiency" and "next-step gaps" in iterative RAG as a structured controller named S2G-Judge. Using gap-guided queries and sentence-level evidence extraction to mitigate noise, it improves F1 from 43.3 (SIM-RAG) to 56.5 under the HotpotQA BM25 setting.
tags:
  - ACL 2026
  - Information Retrieval & RAG
date: 2026-05-08
content_hash: f514e4b4118f2c01
---
# S2G-RAG: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA

**Conference**: ACL2026  
**arXiv**: [2604.23783](https://arxiv.org/abs/2604.23783)  
**Code**: Not declared in the paper  
**Area**: Information Retrieval / RAG / Multi-hop QA  
**Keywords**: Iterative Retrieval, Sufficiency Judgment, Information Gap, Evidence Compression, Multi-hop QA

## TL;DR
S2G-RAG explicitly models "evidence sufficiency" and "next-step gaps" in iterative RAG as a structured controller named S2G-Judge. Using gap-guided queries and sentence-level evidence extraction to mitigate noise, it improves F1 from 43.3 (SIM-RAG) to 56.5 under the HotpotQA BM25 setting.

## Background & Motivation
**Background**: RAG has become the standard for knowledge-intensive QA. While single-hop questions are often solved with one retrieval step, multi-hop questions require iterative retrieval and intermediate reasoning. Existing methods like IR-CoT, FLARE, Self-RAG, SIM-RAG, and RAG-Critic control retrieval through reasoning traces, uncertainty, reflection tokens, or critic signals.

**Limitations of Prior Work**: The key bottleneck in iterative RAG is control rather than single-step retrieval. Systems must decide in each round whether the current evidence memory is sufficient; if not, they must specify what to search for next. When control is unreliable, models either answer prematurely from incomplete chains or continue retrieving, accumulating redundant and distractor text.

**Key Challenge**: Free-text reasoning is flexible but difficult to audit and stabilize into specific next-hop retrieval needs. Furthermore, multi-round retrieval leads to context bloat, causing subsequent judgments and answers to be affected by distractors.

**Goal**: The authors aim to transform retrieval control into an explicit, diagnosable, and trainable structured prediction problem while leveraging sentence-level evidence context to control multi-round context expansion.

**Key Insight**: S2G-RAG designs a lightweight S2G-Judge that outputs a sufficiency decision and structured gap items in each round. Gap items are mapped to the next query, and retrieved documents are filtered by a sentence-level extractor to append only key sentences to the evidence memory.

**Core Idea**: By forcing the RAG controller to answer "is it enough" and structurally identify "what is missing," multi-hop retrieval is transformed from a free-form drift into an auditable gap-filling process.

## Method
The design of S2G-RAG can be understood as judge-first iterative RAG. It does not modify the search engine or retrain the answer generator but introduces a structured control layer and a sentence extraction layer between the retriever and reasoner.

### Overall Architecture

Given a question $q$ and a corpus $D$, the system maintains an accumulating evidence context $C_t$, where $C_0$ is initially empty. In each round, the lightweight S2G-Judge reads $(q, C_t)$ and outputs a binary sufficiency flag $s_t$ and a set of structured gap items $G_t$. If $s_t$ is true, retrieval stops and the answer reasoner generates a response. If false, $G_t$ is translated into the next query, top-k documents are retrieved, and a sentence-level extractor selects key evidence blocks $E_t$ to be appended to $C_t$ for the next round. This pipeline turns iterative control into an auditable judge-fill loop. The default answer reasoner is Llama-3-8B-Instruct, the S2G-Judge is a LoRA-tuned Llama-3.2-3B-Instruct, with a maximum of 4 rounds, top-k=6 per round, and title-based de-duplication. BM25 is used for sparse retrieval and E5-base-v2 for dense retrieval.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    Q["Question q + Evidence Context C_t (initially empty)"] --> JUDGE["Structured Sufficiency and Gap Judge (S2G-Judge)<br/>Outputs Binary Flag + Four-field Gap Items"]
    TRAIN["Distilled Process Supervision from Execution Trajectories<br/>Real Rollout States → GPT-4o-mini Labels → LoRA SFT"] -. Offline Training .-> JUDGE
    JUDGE -->|Sufficient| ANS["Answer Reasoner (Llama-3-8B)<br/>Answers based on C_t"]
    JUDGE -->|Insufficient| QRY["Gap-to-Query Translation<br/>Concatenation of Gap Target + Slot"]
    QRY --> RET["Retrieve top-k Documents<br/>BM25 / E5, Title De-duplication"]
    RET --> EXT["Gap-aware Sentence-level Evidence Extraction<br/>Pointer-based Selection, No Rewriting"]
    EXT --> CUP["Append Extracted Sentence Blocks to C_t"]
    CUP -->|Next Round (max 4)| JUDGE
```

### Key Designs

**1. Structured Sufficiency and Gap Judging: Formatting "Enough/Missing" as fillable fields.** The controller explicitly answers two questions per round: whether the current evidence is sufficient for an answer, and if not, what category of information is missing. Gaps are regularized into gap items with four fields: `category`, `target`, `slot`, and `description`. Categories include bridge_entity, attribute, relation, evidence_span, and other. If sufficiency is true, the gap set is empty.

This structured design directly addresses typical multi-hop failure modes, such as finding a bridge entity but missing its attributes. Compared to free-text reflection, gap items with categories and slots are easier to map to stable queries and diagnose system bottlenecks.

**2. Distillation from Execution Trajectories: Adapting the judge to real intermediate states.** Training data is not derived from idealized evidence chains. Instead, real rollouts of the iterative retrieval process are recorded as $(q, C_t)$ pairs. A strong teacher (GPT-4o-mini) then labels these states with sufficiency and gap items under context-only constraints. After filtering low-confidence or contradictory samples, S2G-Judge is trained via LoRA SFT.

Training on execution trajectories rather than clean annotations is vital because real RAG contexts contain noise and distractors. Learning the distribution of states encountered during deployment ensures the judge remains accurate when evaluating noisy evidence.

**3. Gap-aware Sentence-level Evidence Context: Suppressing noise via pointer extraction.** To prevent the context from bloating with full documents, the extractor treats retrieved documents as a pool of candidate sentences with title provenance. The LLM outputs sentence indices based on the question and current gaps. The system retrieves these original sentences to append to the context—extractor acts as a pointer selector without rewriting.

Pointer-based extraction preserves evidence provenance while avoiding hallucinations introduced by summarization. Providing gap items to the extractor allows it to prioritize sentences that fill specific missing information, keeping the context compact yet relevant.

### Loss & Training
S2G-Judge is trained using standard autoregressive cross-entropy on structured outputs. The input is $(q, C_t)$, and the target is the teacher-labeled schema. Supervision signals are derived entirely from multi-round execution trajectories. During inference, query construction prioritizes the concatenation of the gap item’s target and slot, falling back to the description if necessary. By default, only the first valid gap item is used to keep the query concise.

## Key Experimental Results

### Main Results
The main results report EM/F1 on TriviaQA, HotpotQA, and 2WikiMultiHopQA, comparing BM25 and E5 retrieval settings.

| Retrieval | Method | TriviaQA EM/F1 | HotpotQA EM/F1 | 2Wiki EM/F1 | Key Conclusion |
|-----------|--------|----------------|----------------|-------------|----------------|
| BM25      | IR-CoT | 56.9 / 68.9    | 28.6 / 41.5    | 23.5 / 32.4 | Multi-hop reasoning baseline |
| BM25      | SIM-RAG| 70.7 / 75.6    | 32.7 / 43.3    | 34.1 / 40.2 | Strong critic control |
| BM25      | S2G-RAG| 72.0 / 77.9    | 43.3 / 56.5    | 41.7 / 48.6 | Maximum gain in multi-hop |
| E5        | Std RAG| 58.8 / 68.3    | 25.1 / 35.3    | 10.6 / 21.0 | Single-turn insufficient |
| E5        | RAG-Critic| 65.0 / 75.9 | 40.0 / 51.2    | 27.9 / 34.0 | Strong learned-control baseline |
| E5        | S2G-RAG| 71.1 / 78.0    | 42.0 / 53.5    | 39.0 / 45.3 | Robust across retrievers |

Compared to SIM-RAG, S2G-RAG achieves a +10.6 EM and +13.2 F1 improvement on HotpotQA with BM25. TriviaQA gains are smaller due to its single-hop nature, but sentence-level evidence context still provides benefits by mitigating noise from unnecessary retrieval.

### Ablation Study

| Variant | HotpotQA EM | HotpotQA F1 | Description |
|---------|-------------|-------------|-------------|
| Full S2G-RAG | 43.3 | 56.5 | Complete system |
| w/o SFT | 39.2 | 50.8 | Untuned judge; -4.1 EM / -5.7 F1 |
| w/o S2G-Judge | 27.5 | 37.6 | No structured control; largest drop |
| w/o Extractor | 39.5 | 52.5 | Concatenating full text; higher noise/lat. |

| Compression Method | EM | F1 | Comp. Ratio | Description |
|--------------------|----|----|-------------|-------------|
| LLM summarization  | 36.9 | 48.1 | 0.3816 | Summarization loses evidence |
| ReComp extractive  | 41.4 | 53.8 | 0.4948 | Better fidelity but lower than ours |
| ReComp abstractive | 34.9 | 46.4 | 0.1917 | Shortest but significant QA loss |
| Sentence Pointer   | 43.3 | 56.5 | 0.3461 | Best accuracy/compression trade-off|

### Key Findings
- S2G-Judge is the most critical module. Removing it causes HotpotQA F1 to drop from 56.5 to 37.6, emphasizing that control is more important than sheer retrieval volume in multi-hop tasks.
- Sufficiency judgment is conservative; the false positive rate is 6.44%, but 31.60% of samples that meet the retrieval truth are still judged as insufficient. This reduces premature answering but suggests room for calibration.
- Sentence-level evidence context provides a 4.5x-6.4x compression ratio compared to full documents. In HotpotQA BM25, the system reduced per-sample latency from 1.9552s to 1.6085s while achieving higher F1.

## Highlights & Insights
- "Sufficiency + Gap" is a highly effective interface for iterative RAG. It is more diagnosable than query rewriting and easier to constrain than free-form reflection.
- Trajectory distillation is a practical training design. The controller learns from real system rollouts rather than idealized annotations, better fitting the deployment distribution.
- Sentence pointer extraction balances compression and auditability. Unlike abstractive summarization, it preserves original text, reducing hallucination risk during the compression phase.
- The paper advocates for a modular RAG direction: the retriever, controller, extractor, and reasoner can be decoupled and optimized independently.

## Limitations & Future Work
- The gap schema sacrifices expressiveness for stability; complex questions might need multi-entity joins, temporal constraints, or more structured intermediate programs.
- Sentence-level extraction involves a compactness-recall trade-off, potentially missing cross-sentence evidence or context required for disambiguation.
- Conservative sufficiency judgment increases retrieval rounds and latency. Better calibration is needed.
- The system uses distillation rather than end-to-end RL/actor-critic optimization. Future work could use S2G outputs as auxiliary rewards.

## Related Work & Insights
- **vs IR-CoT**: While IR-CoT uses reasoning traces as retrieval cues, S2G-RAG uses structured gap fields to make retrieval demands more systematic.
- **vs FLARE / Self-RAG**: These methods control retrieval via uncertainty or reflection tokens; S2G-RAG’s sufficiency and gap schema are easier to audit and train.
- **vs SIM-RAG**: SIM-RAG evaluates if a draft answer is acceptable; S2G-RAG goes further by identifying missing information categories and converting them to specific queries.
- **vs RAG-Critic**: RAG-Critic provides error feedback; S2G-RAG is more lightweight, focusing on the structured control of multi-round evidence states.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The schema is straightforward but targets the core bottleneck of iterative RAG control.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers three datasets, two retriever types, and strong baselines.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear progression from motivation to design and analysis.
- Value: ⭐⭐⭐⭐⭐ Highly relevant for RAG system engineering, particularly in auditable control and memory compression.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[ACL 2025\] Mitigating Lost-in-Retrieval Problems in RAG Multi-Hop QA](../../ACL2025/information_retrieval/mitigating_lost-in-retrieval_problems_in_retrieval_augmented_multi-hop_question_.md)
- [\[ACL 2026\] VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG](videostir_understanding_long_videos_via_spatio-temporally_structured_and_intent-.md)
- [\[ACL 2026\] CRAFT: Training-Free Cascaded Retrieval for Tabular QA](craft_training-free_cascaded_retrieval_for_tabular_qa.md)
- [\[ACL 2026\] UnIte: Uncertainty-based Iterative Document Sampling for Domain Adaptation in Information Retrieval](unite_uncertainty-based_iterative_document_sampling_for_domain_adaptation_in_inf.md)
- [\[ACL 2025\] SGIC: A Self-Guided Iterative Calibration Framework for RAG](../../ACL2025/information_retrieval/sgic_a_self-guided_iterative_calibration_framework_for_rag.md)

</div>

<!-- RELATED:END -->
