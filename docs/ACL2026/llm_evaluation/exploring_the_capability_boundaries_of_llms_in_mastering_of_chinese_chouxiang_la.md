---
title: >-
  [Paper Note] Exploring the Capability Boundaries of LLMs in Mastering of Chinese Chouxiang Language
description: >-
  [ACL 2026][LLM Evaluation][Chouxiang language] This paper introduces Chinese internet subculture language "Chouxiang" (抽象话) to the NLP community…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Chouxiang language"
  - "internet subculture language"
  - "LLM benchmarking"
  - "Chinese internet slang"
  - "cross-cultural understanding"
date: 2026-05-08
content_hash: f6870b4410ab34e7
---

# Exploring the Capability Boundaries of LLMs in Mastering of Chinese Chouxiang Language

**Conference**: ACL 2026
**arXiv**: [2604.15841](https://arxiv.org/abs/2604.15841)  
**Code**: [GitHub](https://github.com/csdq777/Mouse)  
**Area**: LLM Evaluation
**Keywords**: Chouxiang language, internet subculture language, LLM benchmarking, Chinese internet slang, cross-cultural understanding

## TL;DR
This paper introduces Chinese internet subculture language "Chouxiang" (抽象话) to the NLP community, constructs the first evaluation benchmark Mouse — comprising six tasks: translation (TR), representation classification (RC), intent recognition (IR), toxicity detection (TD), meaning selection (MS), and cloze completion (CC) — and finds that state-of-the-art LLMs perform reasonably well on contextual semantic understanding but exhibit significant limitations across other tasks.

## Background & Motivation

**Background**: LLMs demonstrate strong performance on standard NLP tasks, yet their ability to handle non-standard linguistic varieties — particularly non-Western subculture languages — remains almost entirely unexplored. Existing research on Chinese internet language is largely confined to toxicity detection and perturbed language identification.

**Limitations of Prior Work**: (1) Existing LLMs and benchmarks exhibit a pronounced Western-centric bias, with Chinese subculture language severely underrepresented; (2) Chouxiang, the most widely used internet subculture language among Chinese youth, integrates complex mechanisms such as homophonic substitution, visual-symbolic analogy, and semantic mapping, far exceeding the complexity of memes and viral internet expressions; (3) Prior research has framed such linguistic phenomena exclusively in negative terms (e.g., toxicity), overlooking their evolved neutral or even positive functions.

**Key Challenge**: Chouxiang constructs a non-standard semantic space deviating from standard Chinese through multi-layered mappings — homophones → grapheme decomposition → semantic metaphor → dialect borrowing — while LLM pretraining is predominantly grounded in standard text, resulting in systematic capability deficits when interpreting and processing this subculturally encoded language.

**Goal**: To formally define a taxonomy of Chouxiang, construct the Mouse evaluation benchmark, and systematically assess the capability boundaries of state-of-the-art LLMs on Chouxiang.

**Key Insight**: Chouxiang is analyzed from a semiotic representation perspective and decomposed into three core dimensions — phonetic, visual, and semantic — which, combined with intent classification, form a dual-dimensional taxonomy.

**Core Idea**: The first NLP benchmark for Chouxiang, comprehensively evaluating LLMs' ability to understand, translate, and produce subculture language through six distinct tasks.

## Method

### Overall Architecture
The Mouse benchmark comprises 1,099 Chouxiang Evaluation Instances (CXEIs), each containing the original text, a standard Chinese reference translation, a representation classification label, an intent label, and a toxicity label. Six tasks are designed: Translation (TR), Representation Classification (RC), Intent Recognition (IR), Toxicity Detection (TD), Meaning Selection (MS), and Cloze Completion (CC).

### Key Designs

1. **Three-Dimensional Representation Taxonomy for Chouxiang**:

    - Function: Systematically characterizes the linguistic composition of Chouxiang expressions.
    - Mechanism: The phonetic dimension exploits the phonological redundancy of Chinese through homophonic substitution (e.g., "主包" → "主播"); the visual dimension leverages the pictographic nature of Chinese characters and the iconic properties of emoji (e.g., "彳亍口巴" → "行吧" via radical decomposition); the semantic dimension performs meaning mapping through direct symbolic literalization or dialect borrowing (e.g., "踩到皮" → "踩到香蕉皮"). A single utterance may simultaneously span multiple dimensions.
    - Design Motivation: Prior taxonomies based on origin (symbols / homophones / dialects / memes) suffer from feature overlap; a semiotic representation perspective yields a more systematic and operationalizable classification.

2. **Dual-Dimensional Evaluation Instance (CXEI)**:

    - Function: Provides a comprehensive unit of evaluation.
    - Mechanism: Each CXEI contains five attributes — original text (Chouxiang text mixing emoji, Chinese characters, and Latin letters), reference translation (pure standard Chinese), representation dimension (phonetic / visual / semantic), intent (commentary / emotional expression / statement / sexual innuendo / meme play / urging / group identity, etc.), and toxicity (binary label).
    - Design Motivation: Chouxiang functions not only as a linguistic phenomenon but also as a social interaction tool, necessitating simultaneous evaluation of linguistic decoding ability and pragmatic comprehension.

3. **Six-Task Evaluation Design**:

    - Function: Comprehensively assesses LLM mastery of Chouxiang from multiple dimensions.
    - Mechanism: Translation (decoding ability) → Representation Classification (linguistic analysis) → Intent Recognition (pragmatic understanding) → Toxicity Detection (safety identification) → Meaning Selection (core semantic comprehension) → Cloze Completion (productive use of Chouxiang). The sequence forms a capability gradient from passive understanding to active production.
    - Design Motivation: No single task suffices for comprehensive evaluation — LLMs may grasp contextual meaning yet fail at translation, or detect toxicity yet fail at intent recognition.

### Loss & Training
This paper is a purely evaluative study and does not involve model training. Translation quality is scored using LLM-as-Judge; all other tasks are evaluated using accuracy or balanced accuracy.

## Key Experimental Results

### Main Results

| Task | SOTA LLM Performance | Notes |
|------|---------------------|-------|
| Meaning Selection (MS) | Good | Contextual semantic understanding is adequate |
| Cloze Completion (CC) | Good | Contextual adaptation is adequate |
| Translation (TR) | Poor | Cannot decode complex phonetic/visual mappings |
| Representation Classification (RC) | Poor | Cannot identify linguistic composition |
| Intent Recognition (IR) | Moderate | Some intents can be inferred from context |
| Toxicity Detection (TD) | Poor | Chouxiang encoding effectively bypasses conventional filters |

### Ablation Study

| Analysis Dimension | Finding |
|-------------------|---------|
| Phonetic vs. Visual vs. Semantic | The visual dimension is the most challenging |
| LLM-as-Judge vs. human agreement | Moderate agreement in translation evaluation |
| Key factors affecting translation | Complexity of the homophonic chain is the primary factor |

### Key Findings
- State-of-the-art LLMs perform adequately on tasks involving contextual semantic understanding (MS, CC), but poorly on tasks requiring decoding of Chouxiang-specific encoding mechanisms (TR, RC).
- This indicates that LLMs grasp *what is being said* but not *how it is said* — they lack awareness of the formal layer of Chouxiang, such as grapheme decomposition and multi-level homophonic chains.
- The visual dimension (grapheme decomposition, emoji metaphor) represents the largest blind spot for LLMs.
- Poor toxicity detection performance demonstrates that Chouxiang can effectively circumvent existing safety filters.

## Highlights & Insights
- **Formally introducing subculture language into NLP evaluation** carries significant sociolinguistic value — internet language is living and continuously evolving, and NLP systems must keep pace.
- The six-task design spanning "comprehension" to "production" constitutes a complete capability evaluation spectrum, analogous to proficiency-graded testing in second-language instruction.
- A pronounced asymmetry is identified between LLMs' understanding of *form* and *content* — models can grasp meaning without understanding the encoding mechanism, offering insight into the fundamental nature of LLM linguistic competence.

## Limitations & Future Work
- The dataset scale is limited (1,099 instances) and may not cover the full range of Chouxiang variants.
- The study focuses exclusively on Chinese Chouxiang; analogous subculture phenomena in other languages (e.g., Japanese internet slang, Korean abbreviations) remain unexplored.
- The reliability of LLM-as-Judge in translation evaluation is limited.
- Toxicity annotation may be subject to annotator subjectivity bias.

## Related Work & Insights
- **vs. Chinese Meme Research (Xie et al., 2025)**: Memes constitute a subset of Chouxiang; the semantic structure of Chouxiang is considerably more complex.
- **vs. Perturbed Language Detection (Xiao et al., 2024)**: That work focuses on toxicity bypass, whereas this paper treats Chouxiang as a neutral subcultural phenomenon.
- **vs. Cross-Cultural LLM Evaluation (Cao et al., 2023)**: That work addresses value alignment bias; this paper focuses on linguistic encoding capability.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First NLP benchmark for Chouxiang, a uniquely novel research perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive six-task evaluation with multi-model comparisons.
- Writing Quality: ⭐⭐⭐⭐ Detailed linguistic analysis with abundant illustrative examples.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](challenging_the_boundaries_of_reasoning_an_olympiad-level_math_benchmark_for_lar.md)
- [\[ACL 2026\] TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice](taxpraben_a_scalable_benchmark_for_structured_evaluation_of_llms_in_chinese_real.md)
- [\[AAAI 2026\] Beyond Accuracy: A Cognitive Load Framework for Mapping the Capability Boundaries of Tool-use Agents](../../AAAI2026/llm_evaluation/beyond_accuracy_a_cognitive_load_framework_for_mapping_the_c.md)
- [\[ACL 2026\] Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation: A Case Study on Tang Poetry](capabilities_and_evaluation_biases_of_large_language_models_in_classical_chinese.md)
- [\[ACL 2026\] Evaluating Memory Capability in Continuous Lifelog Scenario](evaluating_memory_capability_in_continuous_lifelog_scenario.md)

</div>

<!-- RELATED:END -->
