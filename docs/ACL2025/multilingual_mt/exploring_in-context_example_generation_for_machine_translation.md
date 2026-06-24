---
title: >-
  [Paper Note] Exploring In-context Example Generation for Machine Translation
description: >-
  [ACL 2025][Multilingual & Machine Translation][Machine Translation] Proposes DAT (Demonstration Augmentation for Translation), which enables LLMs to automatically generate relevant and diverse source-target sentence pairs as in-context demonstrations **without any external resources**. DAT outperforms zero-shot and fixed-demonstration few-shot baselines across five low-resource language translation tasks.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Machine Translation"
  - "In-context Learning"
  - "Low-resource Languages"
  - "Example Generation"
  - "Maximal Marginal Relevance"
date: 2026-05-08
content_hash: 1e829a3348ca51bc
---

# Exploring In-context Example Generation for Machine Translation

**Conference**: ACL 2025  
**arXiv**: [2506.00507](https://arxiv.org/abs/2506.00507)  
**Code**: [GitHub](https://github.com/aiclaudev/DAT)  
**Area**: Multilingual Translation  
**Keywords**: Machine Translation, In-context Learning, Low-resource Languages, Example Generation, Maximal Marginal Relevance

## TL;DR

Proposes DAT (Demonstration Augmentation for Translation), which enables LLMs to automatically generate relevant and diverse source-target sentence pairs as in-context demonstrations **without any external resources**. DAT outperforms zero-shot and fixed-demonstration few-shot baselines across five low-resource language translation tasks.

## Background & Motivation

**Background**: In-context learning (ICL) in LLMs has demonstrated powerful capabilities in machine translation. Prior work (e.g., R-BM25, CTQScorer) has focused on **selecting the optimal demonstrations from an existing human-annotated corpus pool**, achieving excellent results for high-resource languages.

**Limitations of Prior Work**: The aforementioned methods rely on a crucial assumption—the existence of a large-scale, human-annotated source-target parallel corpus pool. For **low-resource languages** (such as Nepali, Khmer, Pashto, Zulu, and Swahili), this assumption does not hold because public datasets and human annotators are extremely scarce.

**Key Challenge**: Low-resource languages need ICL the most to compensate for the lack of data, yet they are precisely the ones lacking the demonstration pools required for ICL. While El Mekki & Abdul-Mageed (2025) attempted to use LLMs to generate synthetic parallel data, their approach still requires bilingual lexicons and unannotated texts in the target language.

**Goal**: Is it possible to dynamically generate high-quality in-context demonstrations for each translation request solely relying on the linguistic capabilities of LLMs, **without depending on any external resources**?

**Key Insight**: Based on two key priors identified in previous research—**relevance** and **diversity**—LLMs can be guided to generate source-side sentences, which are then filtered using MMR and translated to obtain demonstration pairs.

**Core Idea**: Let the LLM "create its own tests"—automatically generating relevant and diverse translation prompt demonstrations for each sentence to be translated.

## Method

### Overall Architecture

Given a user query $q$ (the sentence to be translated), the process consists of four steps:
1. Generate $m$ relevant source-side sentences using the LLM (satisfying both relevance and diversity).
2. Select $k$ optimal sentences using MMR filtering.
3. Translate these $k$ sentences using the LLM to obtain the target-side responses.
4. Use the $k$ source-target pairs as in-context demonstrations to translate $q$.

### Key Designs

1. **Source-side Generation**:

    - Uses zero-shot prompting to guide the LLM to generate $m=10$ sentences that are relevant to the query $q$ yet distinct from each other.
    - The relevance and diversity priors are embedded directly into the prompt.
    - **Design Motivation**: Relevant sentences provide translation clues (shared vocabulary/syntax), while diversity avoids redundancy.

2. **Filtering using MMR**:

    - Relevance metric—n-gram recall: $\alpha(q, x_i) = \frac{1}{4}\sum_{n=1}^{4} R_n(q, x_i)$
    - Where $R_n$ is the n-gram recall score between $q$ and $x_i$.
    - MMR selection formula: $\arg\max_{x_i \in X \setminus X^*}\left[\alpha(q, x_i) - \frac{\lambda}{|X^*|}\sum_{x_j \in X^*}\alpha(x_j, x_i)\right]$
    - Iteratively selects $k=4$ sentences, ensuring each chosen sentence is both relevant to $q$ and distinct from the already selected ones.
    - **Design Motivation**: Directly generated $m$ sentences may vary in quality; filtering ensures a high upper bound of quality.

3. **Target-side Generation**:

    - Performs zero-shot translation using the LLM for each selected source sentence.
    - Resulting in $k$ source-target pairs: $D^* = \{(x_i^*, \text{LLM}(x_i^*))\}_{i=1}^k$.

4. **Query Translation**:

    - $\hat{y} = \text{LLM}(I, D^*, q)$
    - Where $I$ is the translation instruction.

### Extension: Accumulation Setting

- Gradually accumulates the generated translation pairs during testing into a demonstration pool.
- Subsequent translations can retrieve demonstrations from the pool via R-BM25, reducing real-time generation overhead.

## Key Experimental Results

### Main Results (COMET Scores, English $\rightarrow$ Low-Resource Languages)

| Fixed Pairs | Model | Method | Nepali | Khmer | Pashto | Zulu | Swahili |
|-------|------|------|---------|--------|---------|--------|-----------|
| ✘ | Llama-3.1-8B | Zero-shot | 72.1 | 62.0 | 53.9 | 23.3 | 60.6 |
| ✘ | Llama-3.1-8B | **DAT** | **74.9** | **64.4** | **54.6** | 22.3 | **61.8** |
| ✘ | Llama-3.1-70B | Zero-shot | 79.8 | 72.7 | 67.5 | 37.8 | 72.9 |
| ✘ | Llama-3.1-70B | **DAT** | **81.1** | 72.4 | **68.3** | 38.3 | **73.4** |
| ✔ | Llama-3.1-70B | Few-shot | 80.6 | 51.1 | 65.7 | 38.9 | 71.6 |
| ✔ | Llama-3.1-70B | **DAT** | **81.5** | **52.9** | **68.5** | 39.2 | **72.7** |

- Without fixed pairs: DAT improves by 2.8 COMET points on Nepali (8B) and significantly outperforms zero-shot across most languages.
- With fixed pairs: DAT outperforms traditional few-shot methods in most languages.

### In-Context Demonstration Quality Analysis (Llama-3.1-70B)

| Method | Nepali Relevance↑ | Nepali Quality↑ | Nepali COMET↑ | Khmer Relevance↑ | Khmer COMET↑ |
|------|-----------------|---------------|-------------|---------------|-----------|
| Retrieval(src) | 7.5 | 80.7 | 80.5 | 7.5 | 61.4 |
| Fixed set(pair) | 3.9 | **89.4** | 80.6 | 3.9 | 51.1 |
| **DAT** | **25.9** | 82.5 | **81.1** | **25.9** | **72.4** |

- DAT's Relevance (25.9) far exceeds Retrieval (7.5) and Fixed set (3.9).
- Although the Fixed set has the highest Quality (89.4), its COMET is not the best—**relevance is more important than absolute quality**.

### Ablation Study (Effect of MMR Filtering)

| Method | $m$ | $k$ | Khmer | Pashto | Swahili |
|------|-----|-----|--------|---------|-----------|
| No Filtering₄ | 4 | 4 | 63.8 | 54.2 | 61.9 |
| No Filtering₁₀ | 10 | 10 | 63.4 | 53.6 | 61.7 |
| **DAT** | 10 | 4 | **64.4** | **54.6** | **62.3** |

- 10 unfiltered demonstrations (63.4) perform worse than 4 (63.8)—additional demonstrations acts as noise.
- DAT's generate-10-and-select-4 strategy is optimal.

### Key Findings

- **High-quality fixed pairs can "backfire"**: In English $\rightarrow$ Khmer translation, after using fixed human-annotated pairs, Llama-3.1-70B's COMET score crashed by 21.6 points, and the model generated abnormally long, repetitive outputs.
- **Relevance is the most crucial prior**: It is highly predictive of final translation performance, even more so than absolute translation quality.
- **Self-generated demonstrations are effective**: LLMs can create useful translation clues for low-resource languages entirely relying on their own internal capabilities.
- **Accumulation setting holds potential**: Translation quality steadily improves as seed data increases, though it has not yet fully closed the gap with real-time generation.

## Highlights & Insights

- **Resolves the "chicken-or-egg" dilemma**: Low-resource languages lack demonstration pools $\rightarrow$ Let the LLM generate its own demonstration pool.
- **The discovery of the "backfire phenomenon" is highly valuable**: It reveals that high-quality but irrelevant fixed demonstrations can severely impair ICL performance.
- **Extremely straightforward methodology**: The entire pipeline requires only zero-shot prompting + n-gram matching + MMR filtering.
- **Insight of Relevance > Quality**: Challenges the intuition that "higher demonstration quality is always better," emphasizing the match with the query instead.

## Limitations & Future Work

1. **Evaluated only on the English $\rightarrow$ Low-Resource direction**: Reverse translation (low-resource $\rightarrow$ English) was not explored due to expected limitations in source-side generation quality.
2. **Llama-3.1 exclusive**: Other multilingual models (e.g., Qwen, Gemma, etc.) were not tested.
3. **Accumulation setting has not converged**: An accumulated pool of 500 seed data instances still cannot fully match the performance of on-the-fly generation.
4. **Target side depends entirely on LLM translation**: Combing the approach with NMT models to generate higher-quality target-side text is a potential direction.
5. **Computational overhead**: Each test input requires multiple LLM calls (source sentence generation + translation + final translation), presenting real-time efficiency challenges.

## Related Work & Insights

- **Relationship to R-BM25 (Agrawal et al., 2023)**: While R-BM25 retrieves demonstrations from an existing corpus pool, DAT does not require one. The two can be combined: DAT generation $\rightarrow$ accumulation $\rightarrow$ R-BM25 retrieval.
- **Comparison to Self-ICL (Chen et al., 2023)**: Self-ICL also self-generates demonstrations but targets general tasks; DAT is specifically designed for MT with a dual prior of relevance + diversity.
- **Insight**: In any ICL scenario where demonstrations are scarce, the paradigm of "letting the model generate its own reference first, then using it for assistance" might prove effective.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to explore in-context example generation in the MT field without relying on any external resources.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Five low-resource languages, multi-dimensional analysis (quality/relevance/diversity), and complete ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation and in-depth analysis (the discovery and explanation of the "backfire" phenomenon are excellent).
- **Value**: ⭐⭐⭐⭐ — Direct practical value for low-resource translation, with a simple and reproducible method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] GrammaMT: Improving Machine Translation with Grammar-Informed In-Context Learning](grammamt_improving_machine_translation_with_grammar-informed_in-context_learning.md)
- [\[ACL 2025\] Exploring In-Image Machine Translation with Real-World Background](exploring_in-image_machine_translation_with_real-world_background.md)
- [\[ACL 2025\] Understanding In-Context Machine Translation for Low-Resource Languages: A Case Study on Manchu](understanding_in-context_machine_translation_for_low-resource_languages_a_case_s.md)
- [\[ACL 2025\] THOR-MoE: Hierarchical Task-Guided and Context-Responsive Routing for Neural Machine Translation](thor-moe_hierarchical_task-guided_and_context-responsive_routing_for_neural_mach.md)
- [\[ACL 2025\] Accessible Machine Translation Evaluation For Low-Resource Languages](accessible_machine_translation_evaluation_for_low-resource_languages.md)

</div>

<!-- RELATED:END -->
