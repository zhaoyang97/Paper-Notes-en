---
title: >-
  [Paper Note] EXECUTE: A Multilingual Benchmark for LLM Token Understanding
description: >-
  [ACL 2025][Multilingual & Machine Translation][multilingual] This paper extends the character understanding benchmark CUTE to 8 languages and multiple writing systems, proposing the EXECUTE framework. The study demonstrates that LLM performance varies dramatically across character, word, and sub-character levels in different languages, and counter-intuitively finds that LLMs perform better on token understanding tasks for less familiar languages.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "multilingual"
  - "tokenization"
  - "character understanding"
  - "benchmark"
  - "writing system"
date: 2026-05-08
content_hash: 2eaf3181a2d82086
---

# EXECUTE: A Multilingual Benchmark for LLM Token Understanding

**Conference**: ACL 2025  
**arXiv**: [2505.17784](https://arxiv.org/abs/2505.17784)  
**Code**: [https://github.com/Leukas/EXECUTE](https://github.com/Leukas/EXECUTE)  
**Area**: Multilingual Translation  
**Keywords**: multilingual, tokenization, character understanding, benchmark, writing system

## TL;DR
This paper extends the character understanding benchmark CUTE to 8 languages and multiple writing systems, proposing the EXECUTE framework. The study demonstrates that LLM performance varies dramatically across character, word, and sub-character levels in different languages, and counter-intuitively finds that LLMs perform better on token understanding tasks for less familiar languages.

## Background & Motivation

**Background**: The CUTE benchmark demonstrates that LLMs perform poorly on English character manipulation. However, variations in writing systems (alphabets vs. abugidas vs. logographs) across different languages may introduce distinct challenges.

**Limitations of Prior Work**: CUTE only covers English and Russian, neglecting differences in writing systems. Furthermore, multilingual LLMs exhibit highly uneven token allocation across different languages.

**Key Challenge**: Are the challenges of token understanding in LLMs correlated with the character-word-token (CWT) statistical characteristics of the language?

**Goal**: Build a multilingual token understanding benchmark across different writing systems.

**Key Insight**: A unified framework covering alphabets (English/Russian), abugidas (Amharic/Hindi), abjads (Arabic), logographs (Chinese), mixed systems (Japanese), and featural alphabets (Korean).

**Core Idea**: LLM token understanding difficulties depend on the CWT statistics of the language—the higher the character-to-word ratio, the harder the character-level tasks; the higher the token-to-word ratio, the harder the word-level tasks.

## Method

### Overall Architecture
Select 8 languages (covering all major writing systems) -> Generate dataset for each language using translation models -> Retain CUTE's composition and manipulation tasks (removing similarity tasks) -> Add CJK sub-character tasks -> Evaluate 5 multilingual LLMs.

### Key Designs

1. **CWT Statistical Framework**

    - Character/word ratio (c/w): Chinese 1.51 (extremely low), English 4.04, Russian 5.06 (high)
    - Token/word ratio (t/w): Amharic 7.69 (extremely high, byte-level encoding), Chinese 1.25 (low)
    - Character/token ratio (c/t): English 3.05 (high), Chinese 1.20 (low)
    - **Design Motivation**: These three ratios predict LLM performance across different task granularities.

2. **Simplified Scalable Framework**

    - Eliminate similarity tasks that require static embeddings and native speakers.
    - Adding a new language only requires an English-to-target translation model.
    - **Design Motivation**: Drastically lowers the barrier to scaling.

3. **CJK Sub-character Tasks**

    - Test LLM understanding of radicals and strokes in Chinese/Japanese characters.
    - **Design Motivation**: Logographic writing systems possess unique sub-character structures.

## Key Experimental Results

### Main Results

| Language | Character-level Tasks | Word-level Tasks | Sub-character Tasks |
|------|-----------|---------|-----------|
| English (c/w=4.04) | ~45% | ~85% | N/A |
| Chinese (c/w=1.51) | ~75% | ~60%* | ~25% |
| Arabic | ~40% | ~70% | N/A |
| Amharic (t/w=7.69) | ~55% | ~45% | N/A |
| Korean | ~50% | ~75% | ~30% |
*Chinese word-level tasks involve single-character words.

### Key Findings Matrix

| Finding | Description |
|------|------|
| CWT Statistics Predict Performance | Lower c/w (e.g., Chinese) makes character-level tasks easier. |
| Unfamiliar Languages Perform Better | Low-resource languages perform better on character manipulation (because byte-level tokenization retains more raw character information). |
| Sub-character Tasks are Extremely Difficult | CJK radical understanding achieves only ~25% accuracy. |
| Different Language Challenges at Different Granularities | English struggles at the character level, Chinese at the word level, and Amharic at the word level. |

### Key Findings
- **CWT statistics are strong predictors**: Languages with higher character/word ratios present harder character-level tasks, which aligns with intuition.
- **Counter-intuitive finding: Unfamiliar languages perform better**: This is likely because low-resource languages are encoded as byte-level tokens, which preserves more raw character structure.
- **Sub-character understanding remains an LLM blind spot**: Performance on radical/stroke identification is extremely low.
- **Languages within the same writing system show highly correlated performance** (e.g., English and Russian, which both use alphabets).

## Highlights & Insights
- **The CWT statistical framework** bridges the gap between linguistic features and LLM behaviors, offering a quantitative tool for understanding multilingual LLMs.
- **The counter-intuitive "the less familiar, the better" finding** reveals the profound impact of tokenization strategies on downstream capabilities.
- **The scalable framework design** lowers the barriers to entry, making the addition of new languages practically effortless.

## Limitations & Future Work
- Each language only serves as a single representative of its writing system.
- Translation quality might affect the experimental results.
- The potential of fine-tuning to improve token understanding remains unexplored.
- Future Directions: Benchmarking against character-level LLMs, optimizing tokenization strategies.

## Related Work & Insights
- **vs CUTE**: EXECUTE extends the scope to 8 languages and a diverse set of writing systems.
- **vs Spelling Correction Research**: Prior works often conflate character-level and semantic knowledge, whereas EXECUTE purely assesses the understanding of token structures.

## Rating
- Novelty: ⭐⭐⭐⭐ The CWT framework and cross-writing-system analysis are highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ 8 languages × 5 models × multiple task types.
- Writing Quality: ⭐⭐⭐⭐ Meticulous and rigorous analysis.
- Value: ⭐⭐⭐⭐ Strongly guides the development of multilingual LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CruxEval-X: A Benchmark for Multilingual Code Reasoning, Understanding and Execution](cruxeval-x_a_benchmark_for_multilingual_code_reasoning_understanding_and_executi.md)
- [\[ACL 2025\] Code-Switching Red-Teaming: LLM Evaluation for Safety and Multilingual Understanding](code-switching_red-teaming_llm_evaluation_for_safety_and_multilingual_understand.md)
- [\[ACL 2025\] Are Rules Meant to be Broken? Understanding Multilingual Moral Reasoning as a Computational Pipeline with UniMoral](are_rules_meant_to_be_broken_understanding_multilingual_moral_reasoning_as_a_com.md)
- [\[ACL 2025\] Context Augmented Token-Level Post-Editing for Human Interpreting](context_augmented_token-level_post-editing_for_human_interpreting.md)
- [\[AAAI 2026\] X-MuTeST: A Multilingual Benchmark for Explainable Hate Speech Detection and A Novel LLM-consulted Explanation Framework](../../AAAI2026/multilingual_mt/x-mutest_a_multilingual_benchmark_for_explainable_hate_speech_detection_and_a_no.md)

</div>

<!-- RELATED:END -->
