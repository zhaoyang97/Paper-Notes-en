---
title: >-
  [Paper Note] CORAL: Adaptive Retrieval Loop for Culturally-Aligned Multilingual RAG
description: >-
  [ACL 2026][Information Retrieval & RAG][Cross-cultural RAG] CORAL reframes multilingual RAG failures as "retrieval condition misalignment"—requiring not just query rewriting but the dynamic switching of retrieval corpora…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Cross-cultural RAG"
  - "Planner-critic loop"
  - "Dynamic corpus selection"
  - "Query rewriting"
  - "Low-resource languages"
date: 2026-05-08
content_hash: ed49f8df3e68e195
---

# CORAL: Adaptive Retrieval Loop for Culturally-Aligned Multilingual RAG

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.25676](https://arxiv.org/abs/2604.25676)  
**Code**: No link provided (Experimental details in appendix)  
**Area**: Information Retrieval / Multilingual RAG / Agentic RAG  
**Keywords**: Cross-cultural RAG, Planner-critic loop, Dynamic corpus selection, Query rewriting, Low-resource languages

## TL;DR
CORAL reframes multilingual RAG failures as "retrieval condition misalignment"—requiring not just query rewriting but the dynamic switching of retrieval corpora. Through a planner + critic agent loop forming a closed-loop of "corpus selection → retrieval → scoring & filtering → sufficiency check → corpus/query modification," it achieves a 3.58pp gain over the strongest baseline for low-resource languages on two cultural benchmarks and a 3.91pp gain on CLIcK (Korean cultural QA).

## Background & Motivation

**Background**: Multilingual RAG (mRAG) typically employs query translation or multilingual embeddings for a shared retrieval space, assuming that "language alignment is sufficient."

**Limitations of Prior Work**: (1) Culture-bound queries (e.g., Korean holidays, Indonesian customs) often retrieve "semantically related but culturally misaligned" evidence—for example, a question about traditional Korean festivals might retrieve generic "Korean tourism" info from English Wikipedia. (2) Existing agentic mRAG (Self-RAG, ReAct, IRCoT) only focuses on "how to search"—query reformulation and multi-hop reasoning—but the **retrieval space remains fixed**. Even clever query rewriting cannot compensate for "searching in the wrong corpus." (3) Simply stacking more languages into the corpus pool (multiRAG) introduces excessive noise.

**Key Challenge**: Current mRAG treats multilingualism primarily as a representation problem (mapping all languages to the same space). However, cultural QA lacks **locale-specific knowledge**—mixing it with globally aggregated content causes the retrieval phase to be overwhelmed by mainstream cultures.

**Goal**: (1) Treat "retrieval condition" (corpus scope + query expression) as a first-class decision that can be modified at test time. (2) Use a feedback loop to let evidence quality drive updates to retrieval conditions. (3) Prove that this "adaptive retrieval condition" is more effective than "adaptive query" particularly for low-resource languages.

**Key Insight**: Rather than letting the planner rephrase queries within a fixed corpus, the planner should simultaneously decide "which corpora to search + how to search," while a critic judges evidence sufficiency and provides feedback to the planner for corpus re-selection if necessary.

**Core Idea**: Dynamic corpus selection + critique-guided query rewriting + explicit sufficiency check, iterating within a planner-critic loop.

## Method

### Overall Architecture
CORAL is a test-time agent framework requiring no training. It relies on two LLM agents (planner and critic; implemented using the same model like GPT-OSS-120B or Qwen3-235B in experiments) coupled with an external retriever (Qwen3-Embedding-8B + FAISS cosine nearest neighbor) to implement a 5-step loop: (1) The planner selects a small subset of culturally/linguistically relevant corpora based on the query. (2) Retrieve top-5 documents from each selected corpus. (3) The critic scores and filters each document using a 4-dimensional metric. (4) The critic determines if the cumulative evidence is sufficient. (5) If insufficient, the critique is fed back to the planner to re-select corpora and rewrite the query, returning to step (1). After termination, the top 5 documents with the highest $s_{tot}$ from the cumulative validated pool are fed to the generator (LLaMA-3.2-3B-Instruct, etc.).

### Key Designs

1.  **Query-conditioned Dynamic Corpus Selection vs. Fixed Corpus Pool**:
    *   **Function**: Allows the planner to dynamically pick 1–N target corpora (subsets of Wikipedia in 13 languages) based on query language/cultural cues, rather than using a rigid multilingual pool.
    *   **Mechanism**: The planner usually starts with the corpus corresponding to the query language but actively expands to culturally adjacent corpora when cultural cues (local institutions, customs, regional entities) are detected. After receiving critic feedback, it can expand (adding regional high-resource neighbors) or contract (removing irrelevant sources). For instance, a query in BLEnD-su (Sundanese) may trigger both Sundanese and Indonesian corpora.
    *   **Design Motivation**: The core argument is that the bottleneck of mRAG lies in the **retrieval space**, not just query phrasing. Figure 3 shows the planner's selected language distribution extends far beyond the query language (English queries trigger su/id/fa/ar corpora), proving the planner performs "cultural routing" rather than simple language matching.

2.  **Multi-dimensional Scoring + Cumulative Evidence Pool + Explicit Sufficiency Check**:
    *   **Function**: Ensures the critic looks beyond mere relevance to utility, clarity, and cultural compatibility; triggers explicit iterations when information is lacking.
    *   **Mechanism**: The critic assigns a 0–5 score across four dimensions: relevance $s_\text{rel}$, usefulness $s_\text{use}$, specificity $s_\text{spec}$, and compatibility $s_\text{comp}$. Scores are aggregated as $s_{tot} = s_{rel} + 0.5(s_{use} + s_{spec} + s_{comp})$. A document is "validated" only if every dimension $\geq 2$ and $s_{tot} \geq 6$. The critic also provides a binary sufficiency decision each round.
    *   **Design Motivation**: Simple relevance is insufficient for cultural tasks where content may be "semantically correct but contextually wrong." Adding the "compatibility" dimension specifically captures alignment in language, culture, and domain. Structured feedback enables the planner to make precise improvements in the next round.

3.  **Critique-guided Query Rewriting (paraphrase / narrow / expand)**:
    *   **Function**: The planner goes beyond translation, performing three types of rewriting based on the critic’s failure analysis.
    *   **Mechanism**: Analysis of 158 rewrites for 100 CLIcK samples showed 53.8% are "narrowing" (adding constraints/disambiguation), 32.9% are paraphrasing, and the rest are expanding. Narrowing typically occurs when "retrieved topics are relevant but information is insufficient," where the planner adds context cues identified by the critic.
    *   **Design Motivation**: Pure translation-based rewriting (e.g., tRAG / crossRAG) fails on cultural tasks because it modifies the language but not the "information demand structure." CORAL drives rewriting via failure signals, translating information gaps into specific retrieval constraints.

### Loss & Training
Purely inference-time: Planner/critic temperature 0.6 + reasoning effort high; generator temperature 0 + reasoning effort low. The critic handles 4-dimensional scoring and sufficiency decisions; the planner handles corpus selection and query rewrite decisions. Iterations are capped, stopping once sufficiency is met (average 1.34 rounds for BLEnD, 1.52 for CLIcK).

## Key Experimental Results

### Main Results (Cultural QA Accuracy, Generator = LLaMA-3.2-3B-Instruct)

| Method | BLEnD-low | BLEnD-mid | BLEnD-high | BLEnD-avg | CLIcK |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Non-RAG | 58.04 | 55.65 | 62.09 | 62.13 | 48.10 |
| monoRAG | 57.69 | 56.80 | 65.03 | 63.93 | 53.53 |
| tRAG (translate-then-retrieve) | – | – | – | – | 56.06 |
| multiRAG (all corpus pool) | 61.89 | 56.48 | 67.97 | 63.49 | 50.78 |
| crossRAG (multi + translated docs) | 62.59 | 57.83 | 67.32 | 64.27 | 53.75 |
| **CORAL (GPT-OSS-120B)** | **68.18** | 60.47 | 70.92 | 67.14 | 58.66 |
| **CORAL (Qwen3-235B)** | 66.78 | **61.83** | **72.22** | **67.84** | **58.88** |

Max improvement: +3.58pp on BLEnD-low (+5.59pp specifically for Sundanese), +3.91pp on CLIcK; outperforms Self-RAG by up to +12.14pp.

### Ablation Study

**(a) Fixed Corpus Scope vs. CORAL (testing if rigid English fallback is enough):**

| Method | BLEnD-low | BLEnD-mid | BLEnD-high | CLIcK |
| :--- | :--- | :--- | :--- | :--- |
| Non-RAG | 55.65 | 63.06 | 69.29 | 48.10 |
| RAG-$C_\text{own}$ (Oracle monolingual) | 51.89 | 60.77 | 67.43 | 53.53 |
| RAG-$C_\text{all}$ (Full pool) | 56.55 | 65.92 | 69.84 | 50.78 |
| RAG-$C_\text{own} \cup C_\text{en}$ | 56.06 | 65.94 | 71.22 | 54.20 |
| **CORAL** | **61.83** | **70.41** | **72.78** | **58.88** |

**(b) Component Ablation (Adding CORAL components to multiRAG baseline, GPT-OSS-120B planner):**

| Configuration | BLEnD-low | BLEnD-mid | BLEnD-high | CLIcK |
| :--- | :--- | :--- | :--- | :--- |
| multiRAG (fixed pool + original query) | 56.55 | 65.92 | 69.84 | 50.78 |
| + Dynamic Corpus Selection | 58.11 | 70.06 | 72.76 | 57.25 |
| **+ Query Rewriting (full CORAL)** | **60.47** | 69.10 | **73.51** | **58.66** |

### Key Findings
*   Simply adding an English fallback ($C_\text{own} \cup C_\text{en}$) is far inferior to dynamic selection—proving CORAL's gains come from query-conditioned "cultural routing," not just "relying on English."
*   Oracle fixed corpus $C_\text{own}$ (retrieving only the target language) actually underperforms Non-RAG on BLEnD, suggesting cultural QA cannot rely on a single source—many answers require proxy evidence.
*   **Dynamic Corpus Selection** is the largest single contributor (multiRAG → +5.78pp on BLEnD-mid, +6.47pp on CLIcK), proving "where to search" is more critical than "how to search."
*   Query rewriting adds another +1–3pp on top of corpus selection, with 53.8% being "narrowing"—its main function is filling contextual gaps identified by the critic rather than mere translation.
*   Consistency across 3 generators (Llama-3.2-3B, Ministral-3-8B, Qwen3-1.7B) confirms improvements stem from retrieval conditions rather than generator capacity.

## Highlights & Insights
*   "Retrieval condition misalignment" is a clear, actionable failure mode name—it shifts the mRAG failure explanation from vague "hallucination" to "searching the wrong corpus for semantically correct but culturally wrong content," providing a specific path for fixes.
*   The critic's "compatibility" dimension explicitly targets cultural alignment, formalizing the weakness of previous relevance-only ranking into a computable signal.
*   The planner's corpus distribution (Fig 3) shows English queries triggering Sundanese/Indonesian corpora—evidence that the planner is performing "cultural inference" rather than "language matching."
*   The framework requires no model training and is a pure LLM agent loop, allowing immediate deployment on any LLM; this inference-time agentic design is a practical solution for low-resource language scenarios.

## Limitations & Future Work
*   Wikipedia subsets limit corpus diversity—procedural, experiential, or local policy knowledge may not be in Wikipedia, leaving blind spots for real-world cultural QA.
*   Currently only evaluates Multiple Choice Questions (MCQ); does not cover open-ended generation or multi-turn failure modes.
*   The planner and critic are the same model; the impact of decoupling them hasn't been ablated. A strong planner with a lightweight critic might be more efficient.
*   High-conflict scenarios (like NQ-Swap) were not tested; it's unknown if CORAL needs a suppression mechanism.
*   Iterative inference cost: CLIcK averages 21,548 tokens/sample, which is a cost factor for large-scale deployment, necessitating caching or early-stopping optimizations.

## Related Work & Insights
*   **vs. Self-RAG / IRCoT / Self-Ask**: These use iterative query rewriting but keep the corpus fixed; CORAL treats the "retrieval space" as an adaptive variable, outperforming Self-RAG by 12.14pp on BLEnD.
*   **vs. multiRAG / crossRAG (Ranaldi et al. 2026)**: These merge or translate corpora; CORAL uses selective retrieval and critique filtering to avoid the noise of "indiscriminate corpus expansion."
*   **vs. MAFeRw / RQ-RAG (Query experts)**: These only optimize queries; CORAL optimizes both corpus and query, feeding information gaps back into query rewriting.
*   **vs. MIRAGE-bench / BLEnD (Cultural benchmarks)**: This work goes beyond evaluation to provide a retrieval-side solution, proving that "the solution to issues identified in cultural benchmarks should lie in retrieval, not just generation."

## Rating
*   **Novelty**: ⭐⭐⭐⭐⭐ The reframing of "retrieval condition as a first-class decision" and the evidence of cultural routing are novel contributions.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 6 generators × 2 benchmarks × 13 languages + 3 types of ablation + manual annotation of rewrite types, very solid.
*   **Writing Quality**: ⭐⭐⭐⭐ The story is clear and ablations are thorough, though some prompt details and algorithmic steps are tucked into the appendix.
*   **Value**: ⭐⭐⭐⭐⭐ Immediately applicable for low-resource RAG deployment without training; high value for internationalized product teams.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] The Dilemma of Low-Resource Languages in Multilingual Retrieval: Evidence from Amharic](the_multilingual_curse_at_the_retrieval_layer_evidence_from_amharic.md)
- [\[ACL 2026\] All Languages Matter: Understanding and Mitigating Language Bias in Multilingual RAG](all_languages_matter_understanding_and_mitigating_language_bias_in_multilingual_.md)
- [\[ACL 2026\] Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion](enhancing_multilingual_rag_systems_with_debiased_language_preference-guided_quer.md)
- [\[ACL 2026\] Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation](language-coupled_reinforcement_learning_for_multilingual_retrieval-augmented_gen.md)
- [\[ICML 2026\] Retriever Portfolios: A Principled Approach to Adaptive RAG](../../ICML2026/information_retrieval/retriever_portfolios_a_principled_approach_to_adaptive_rag.md)

</div>

<!-- RELATED:END -->
