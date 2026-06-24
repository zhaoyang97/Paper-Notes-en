---
title: >-
  [Paper Note] AntiLeakBench: Preventing Data Contamination by Automatically Constructing Benchmarks with Updated Real-World Knowledge
description: >-
  [ACL 2025][LLM Evaluation][Data Contamination] Proposed AntiLeakBench, an automated anti-leakage benchmark framework that identifies new knowledge post-LLM cutoff dates by tracking Wikidata knowledge update histories, and automatically constructs single-/multi-hop QA test samples (with real-world Wikipedia supporting documents) to ensure strict knowledge-level zero contamination. Large-scale experiments on 12 LLMs demonstrate a pervasive post-cutoff performance decline (with…
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Data Contamination"
  - "Benchmark Construction"
  - "Knowledge Update"
  - "Wikidata"
  - "Automated Evaluation"
date: 2026-05-08
content_hash: bc0d4015a3f9f776
---

# AntiLeakBench: Preventing Data Contamination by Automatically Constructing Benchmarks with Updated Real-World Knowledge

**Conference**: ACL 2025  
**arXiv**: [2412.13670](https://arxiv.org/abs/2412.13670)  
**Code**: [https://github.com/bobxwu/AntiLeakBench](https://github.com/bobxwu/AntiLeakBench)  
**Area**: LLM Evaluation / Data Contamination  
**Keywords**: Data Contamination, Benchmark Construction, Knowledge Update, Wikidata, Automated Evaluation

## TL;DR

Proposed AntiLeakBench, an automated anti-leakage benchmark framework that identifies new knowledge post-LLM cutoff dates by tracking Wikidata knowledge update histories, and automatically constructs single-/multi-hop QA test samples (with real-world Wikipedia supporting documents) to ensure strict knowledge-level zero contamination. Large-scale experiments on 12 LLMs demonstrate a pervasive post-cutoff performance decline (with significant EM drop), validating the framework's effectiveness.

## Background & Motivation

**Background**: Static benchmarks like MMLU and GSM8K are the foundation of LLM evaluation, but their public availability leads to the risk of test data leaking into training sets—the data contamination issue. Overfitting has already been confirmed in certain LLMs on GSM8K. Dynamic benchmarks (e.g., LiveBench, RealTimeQA) attempt to address this by gathering newly released data, but the core issue remains unresolved.

**Limitations of Prior Work**: (1) **Weak immunity to contamination**—"newly released" does not equate to "new knowledge". LeetCode programming tasks or exam questions might have their solutions already covered in LLM training data before formal release. (2) **High manual maintenance cost**—human annotation is required for newly gathered data, leading to low update frequencies. RealTimeQA and KoLA have recently almost ceased updates.

**Key Challenge**: How to balance "strict guarantee of zero contamination" with "sustainable low-cost updates"?

**Goal**: Build a knowledge-level strictly contamination-free evaluation benchmark while achieving a fully automated, zero-human-labor update process.

**Key Insight**: Instead of directly using newly released data, identify knowledge triples that are genuinely updated in a knowledge base (Wikidata) after the cutoff, and construct QA samples based on them.

**Core Idea**: Use the edit history of the knowledge base to track "new knowledge" (rather than "new data"), automatically constructing evaluation samples guaranteed to be contamination-free at the knowledge level.

## Method

### Overall Architecture

A four-step automated pipeline: (1) Prepare Wikidata triple data, (2) Identify knowledge updated after the cutoff time, (3) Build supporting documents from Wikipedia, (4) Construct contamination-free QA samples based on the new knowledge. The entire process requires no human intervention.

### Key Designs

1. **Data Preparation**:

    - **Function**: Retrieve entity-relation-entity triples and their temporal qualifiers from Wikidata.
    - **Mechanism**: Extract relations associated with physical entities (e.g., member of sports team) while excluding relations of virtual entities (e.g., coordinates). Each triple is accompanied by start_time and end_time qualifiers, marking the valid period of the knowledge.
    - **Design Motivation**: Wikidata provides structured, temporally annotated knowledge, making it an ideal data source for tracking knowledge changes.

2. **Identifying Updated Knowledge**:

    - **Function**: Identify knowledge that changed after the LLM cutoff time $t_1$ and before the current time $t_2$.
    - **Mechanism**: Group all triples by subject and relation, sorting them chronologically by start_time. If a new value of a triple appears after $t_1$ (i.e., the object changes), it is flagged as updated knowledge. For example: (Messi, member of sports team) from PSG → Inter Miami, occurring after the cutoff.
    - **Key Details**: Exclude cases of "reverting to previous values" (e.g., a player returning to a former club) to confirm that the new value is indeed different from the old one.
    - **Design Motivation**: This is the core insight of the framework—only when the knowledge itself is updated after the cutoff can we strictly guarantee that the knowledge does not exist in the LLM's training set.

3. **Building Supporting Documents**:

    - **Function**: Provide real-world context for each piece of new knowledge.
    - **Mechanism**: Retrieve the revision history of Wikipedia pages, locate revisions subsequent to the start_time of the new knowledge, and extract article summaries containing the subject and object (or their aliases) as supporting documents.
    - **Design Motivation**: Avoid using LLMs to generate documents (to prevent hallucination); instead, utilize Wikipedia, a well-maintained real-world data source. Furthermore, the revised documents are also generated after the cutoff, ensuring they are absent from the training set.

4. **Constructing Contamination-Free Samples**:

    - **Function**: Build evaluation samples based on new knowledge and supporting documents.
    - **Mechanism**: Support four task formats—
        - **Single-Hop Gold**: Directly query new knowledge, with the context containing only the supporting document (e.g., "Which team does Messi play for?", with his Wikipedia page as context).
        - **Single-Hop $N_d$**: Add $N_d$ distractor documents to test long-context localization capability.
        - **Multi-Hop Gold**: Construct an $H$-hop chain of knowledge ($o_i = s_{i+1}$), such as "Who is the coach of Messi's team?", requiring two-step reasoning.
        - **Multi-Hop $N_d$**: Multi-hop with distractor documents.
    - **Question Formats**: Generation (open-ended generation) and Multi-Choice (four options: Correct / Unknown / Outdated / Noise).
    - **Design Motivation**: Diverse task formats evaluate different capability dimensions of LLMs—knowledge retrieval, long-context understanding, and multi-hop reasoning.

### Benchmark Maintenance

The benchmark can be updated simply by downloading the latest Wikidata dump and running the automated pipeline. It naturally supports multiple languages (leveraging the multilingual features of Wikidata/Wikipedia) with zero manual cost throughout the process.

## Key Experimental Results

### Main Results (12 LLMs × 8 Task Settings, EM/F1)

| Model | Single-Hop Gold EM | Multi-Hop Gold EM | Average EM | Average F1 |
|------|-------------------|-------------------|---------|---------|
| Gemma-2-9B | **85.0** | 57.7 | — | — |
| Mistral-Nemo-12B | 82.7 | **57.7** | **53.9** | **62.0** |
| LongChat-v1.5-7B | 75.5 | 38.8 | 36.4 | 48.9 |
| GPT-4o | High | High | One of the best | One of the best |
| Llama-2-7B | 40.6 | 33.6 | 19.9 | 36.7 |

### Ablation Study (Performance Comparison Before/After Cutoff)

| Observation | Explanation |
|------|------|
| General EM decline post-cutoff | Performance of all LLMs on post-cutoff samples is significantly lower than on pre-cutoff ones, directly confirming the existence of data contamination. |
| Increased selection of outdated answers | In Multi-Choice, the selection rate of outdated answers (correct pre-cutoff but updated post-cutoff) rises significantly. |
| Performance drop with more distractor documents | EM continuously declines from $N_d=3$ to $N_d=7$, reflecting difficulties under long-context retrieval. |

### Key Findings

- Data Quality Verification: Human evaluation indicates context accuracy of 97.3% (single-hop)/98.7% (multi-hop), and answer accuracy of 96.7%/97.3%.
- The performance comparison before and after the cutoff is the most direct evidence of the framework's effectiveness—if the benchmark is truly uncontaminated, performance on post-cutoff knowledge should drop (since LLMs have not encountered it).
- Multi-hop tasks are significantly more difficult than single-hop tasks, and distractor documents further exacerbate this difficulty.

## Highlights & Insights

- Clarifying the distinction between "new knowledge" and "new data" is a core contribution. While LiveBench collects new problems from LeetCode whose underlying knowledge might still be old, AntiLeakBench ensures that the fundamental knowledge itself is generated post-cutoff.
- The fully automated, zero-human pipeline ensures sustainable maintainability; downloading Wikidata dumps periodically is sufficient to generate customized benchmarks for new LLMs.
- The inclusion of an outdated option (the correct answer pre-cutoff) in Multi-Choice tasks cleverly detects whether LLMs rely on outdated knowledge for answering.

## Limitations & Future Work

- Only covers knowledge-based QA tasks—reasoning, mathematics, and code generation tasks cannot be evaluated using this framework.
- Relies on the update frequency and coverage of Wikidata/Wikipedia; knowledge updates in certain domains might lag.
- The knowledge cutoff dates of LLMs might not be entirely accurate, as models might encounter post-cutoff knowledge through other channels.
- Automatically generated questions tend to be highly templated, offering lower diversity compared to human-written ones.
- Although multilingual support is claimed, experiments are primarily validated on English.

## Related Work & Insights

- **vs LiveBench**: Collects newly released data, but the underlying knowledge may be old; AntiLeakBench ensures the knowledge itself is new.
- **vs RealTimeQA**: Requires manual maintenance and has rarely been updated recently; AntiLeakBench is fully automated and zero-cost.
- **vs ADU (Ying et al. 2024)**: Uses LLMs to rewrite existing benchmarks, which risks introducing bias; AntiLeakBench operates based on real-world knowledge updates.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The concept of "new knowledge" and the fully automated knowledge-tracking construction pipeline are novel and significant contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated 12 LLMs under 8 task settings, with strong before/after cutoff comparisons and thorough human quality verification.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and detailed methodology, though some mathematical notations are dense.
- Value: ⭐⭐⭐⭐⭐ A major contribution to LLM evaluation infrastructure, with its core practical value lying in fully automated and sustainable updating.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TripTailor: A Real-World Benchmark for Personalized Travel Planning](triptailor_a_real-world_benchmark_for_personalized_travel_planning.md)
- [\[ICML 2025\] How Much Can We Forget about Data Contamination?](../../ICML2025/llm_evaluation/how_much_can_we_forget_about_data_contamination.md)
- [\[ICLR 2026\] Detecting Data Contamination in LLMs via In-Context Learning](../../ICLR2026/llm_evaluation/detecting_data_contamination_in_llms_via_in-context_learning.md)
- [\[ACL 2025\] Something's Fishy In The Data Lake: A Critical Re-evaluation of Table Union Search Benchmarks](somethings_fishy_in_the_data_lake_a_critical_re-evaluation_of_table_union_search.md)
- [\[ACL 2025\] RuleArena: A Benchmark for Rule-Guided Reasoning with LLMs in Real-World Scenarios](rulearena_rule_guided_reasoning.md)

</div>

<!-- RELATED:END -->
