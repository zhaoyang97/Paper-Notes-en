---
title: >-
  [Paper Note] SPARTA: Scalable and Principled Benchmark of Tree-Structured Multi-hop QA over Text and Tables
description: >-
  [ICLR 2026][Audio & Speech][Multi-hop Reasoning] This paper presents SPARTA, an end-to-end framework for automatically constructing large-scale table-text multi-hop QA benchmarks. By leveraging a reference fact database…
tags:
  - "ICLR 2026"
  - "Audio & Speech"
  - "Multi-hop Reasoning"
  - "Table-Text QA"
  - "Benchmark Construction"
  - "SQL"
  - "Cross-modal Reasoning"
date: 2026-05-08
content_hash: 44bd8548d0acfc70
---

# SPARTA: Scalable and Principled Benchmark of Tree-Structured Multi-hop QA over Text and Tables

**Conference**: ICLR 2026
**arXiv**: [2602.23286](https://arxiv.org/abs/2602.23286)  
**Code**: [github.com/pshlego/SPARTA](https://github.com/pshlego/SPARTA)  
**Area**: Audio & Speech
**Keywords**: Multi-hop Reasoning, Table-Text QA, Benchmark Construction, SQL, Cross-modal Reasoning

## TL;DR

This paper presents SPARTA, an end-to-end framework for automatically constructing large-scale table-text multi-hop QA benchmarks. By leveraging a reference fact database, provenance-based refinement, and realistic structural constraints to generate high-quality nested SQL queries, SPARTA reduces the F1 of state-of-the-art models by over 30 points.

## Background & Motivation

- **Three major limitations of existing benchmarks**:
  1. **Limited question types and shallow reasoning**: Most benchmarks require ≤2-hop reasoning and do not support advanced operations such as aggregation and grouping.
  2. **Severe annotation noise**: An audit of 100 HybridQA samples reveals that 21% contain errors (redundant modality 52.4%, incomplete answers 23.8%, incorrect/unanswerable 23.8%).
  3. **Reliance on small-scale web tables**: Average ~15 rows, far from the thousands of rows found in real-world databases.
- The complexity of manual annotation limits benchmark scale and quality, motivating the need for automated approaches.

## Method

### Overall Architecture: Three-Stage Pipeline

1. **Reference Fact Database Construction** — Source tables and grounding tables are merged into a unified relational database.
2. **Query Generation** — An LLM generates nested SQL queries whose depth matches the target number of hops.
3. **Question Verbalization** — Validated SQL queries are converted into fluent natural language questions.

### Reference Fact Database

- **Source table $\mathcal{S}_T$**: Retains the original relational tables (e.g., 6 publicly available Kaggle tables covering NBA salaries, awards, drafts, etc.).
- **Grounding table $\mathcal{G}_T$**: Decomposes unstructured text into atomic fact tuples stored in SQL-queryable relational tables.
- Two grounding approaches: (1) using verified corpora such as ROTOWIRE; (2) template-based table-to-text conversion.
- Shared entity attributes (e.g., PLAYER_NAME) are linked via primary–foreign key constraints to ensure join reachability.

### Key Designs in Query Generation

#### Post-order Traversal + Realistic Structural Constraints

- Nested SQL is modeled as a query graph $G=(V,E)$, where nodes represent query blocks and edges represent nesting predicates.
- **Post-order traversal** is adopted for construction: leaf queries are generated and validated first, then recursively encapsulated into higher-level queries.
- This approach outperforms top-down or breadth-first strategies, as post-order guarantees that each intermediate block can be executively validated at construction time.

#### Provenance-based Refinement

When a query returns empty results:
1. Predicates are stripped in reverse order until a non-empty result is obtained.
2. Tuples are sampled from the non-empty result.
3. A why-not provenance tool is run to identify blocking predicates.
4. The diagnostic report is fed back to the LLM to rewrite the problematic clause.

### Question Verbalization

- AST-ICL (a state-of-the-art SQL-to-text model) is used to convert SQL into fluent natural language.
- Three CS graduate students perform lightweight validation; annotation efficiency is **4×** that of HybridQA.

### Domain-Agnostic Design

The framework is extensible to arbitrary domains: given source and grounding tables, and after applying table-to-text generation, the query generation pipeline remains unchanged. Extensions to the movie and medical domains are demonstrated.

## Key Experimental Results

### Benchmark Comparison

| Benchmark | Table Scale | Question Generation | GROUP BY/HAVING | >3-Hop | Annotation Error Rate |
|-----------|-------------|---------------------|-----------------|--------|-----------------------|
| HybridQA | 4.4 cols × 15.7 rows | Manual | ✗ | ✗ | 21% |
| OTT-QA | 4.4 cols × 15.7 rows | Manual | ✗ | ✗ | 21% |
| TAT-QA | 4.0 cols × 9.4 rows | Manual | ✗ | ✗ | 30% |
| **SPARTA (NBA)** | **12.2 cols × 3280 rows** | **Automatic + lightweight validation** | **✓** | **✓** | **0%** |

### Performance of SOTA Models on SPARTA

| Model | HybridQA F1 | SPARTA F1 | Drop |
|-------|-------------|-----------|------|
| Best existing model | >70 | <40 | >30↓ |
| Best OTT-QA model | >50 | <20 | >30↓ |

### Ablation Study: Query Generation Strategies

| Method | Execution Success Rate | Query Diversity |
|--------|------------------------|-----------------|
| One-Shot (no verification) | Low | Low |
| Post-Order (no Provenance) | Medium | Medium |
| **Post-Order + Provenance** | **High** | **High** |

### Key Findings

1. SOTA models (GPT-4, Claude, etc.) suffer substantial F1 drops on SPARTA, exposing fundamental weaknesses in cross-modal reasoning.
2. The combination of post-order traversal and provenance-based refinement significantly improves query execution rate and diversity.
3. Lightweight human validation requires only 1/4 of the annotation time needed for HybridQA.
4. Successful extension to the movie and medical domains validates the domain-agnostic design.

## Highlights & Insights

- **Fundamental redesign of Table-Text QA benchmarks**: The SQL-centric pipeline addresses three core issues simultaneously—scale, noise, and logical depth.
- **Provenance-based refinement as a key innovation**: Database techniques (why-not provenance) are introduced into NLP benchmark construction.
- **High discriminative power**: F1 drops of 30+ points for SOTA models clearly indicate fundamental deficiencies in existing cross-modal reasoning capabilities.
- **Reproducible and extensible**: Code, data, and models are fully open-sourced to facilitate future research.

## Limitations & Future Work

- Atomic fact extraction for grounding tables relies on specific corpora (e.g., ROTOWIRE); extending to new domains requires manual template design.
- Question verbalization depends on LLMs, which may introduce subtle semantic drift.
- Only extractive and generative QA models are evaluated; agent-based and tool-augmented methods have not yet been tested.

## Related Work & Insights

- Table-Text QA benchmarks: HybridQA, OTT-QA, TAT-QA, FinQA, MultiHiertt, etc.
- Synthetic benchmark generation: ERBench, TDBench, etc. (mostly single-modal or shallow).
- PEEL: Template-based NL–nested SQL pair generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The SQL-centric approach to automated benchmark construction is innovative.
- **Technical Depth**: ⭐⭐⭐⭐ — Provenance-based refinement and post-order traversal constraints are elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive coverage across multiple domains, models, and ablations.
- **Practical Value**: ⭐⭐⭐⭐⭐ — Directly exposes fundamental weaknesses of SOTA models, offering high community value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion](scalable_multilingual_multimodal_machine_translation_with_speech-text_fusion.md)
- [\[ACL 2026\] Jamendo-MT-QA: A Benchmark for Multi-Track Comparative Music Question Answering](../../ACL2026/audio_speech/jamendo-mt-qa_a_benchmark_for_multi-track_comparative_music_question_answering.md)
- [\[ICLR 2026\] MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark](mmsu_a_massive_multi-task_spoken_language_understanding_and_reasoning_benchmark.md)
- [\[ICLR 2026\] EchoMind: An Interrelated Multi-level Benchmark for Evaluating Empathetic Speech Language Models](echomind_an_interrelated_multi-level_benchmark_for_evaluating_empathetic_speech_.md)
- [\[ICLR 2026\] VowelPrompt: Hearing Speech Emotions from Text via Vowel-level Prosodic Augmentation](vowelprompt_hearing_speech_emotions_from_text_via_vowel-level_prosodic_augmentat.md)

</div>

<!-- RELATED:END -->
