---
title: >-
  [Paper Note] Exploring the Capability Boundaries of LLMs in Mastering of Chinese Chouxiang Language
description: >-
  [ACL 2026 Findings][LLM Evaluation][Chouxiang Language] This paper introduces "Chouxiang Language," a Chinese internet subculture language, to the NLP community and constructs Mouse, the first evaluation benchmark (comprising six tasks: translation, representation classification, intent recognition, toxicity detection, meaning selection, and cloze test). It discovers that while SOTA LLMs perform reasonably in contextual semantic understanding…
tags:
  - "ACL 2026 Findings"
  - "LLM Evaluation"
  - "Chouxiang Language"
  - "Internet Subculture Language"
  - "LLM Benchmark"
  - "Chinese Internet Slang"
  - "Cross-cultural Understanding"
date: 2026-05-08
content_hash: 5e83b42ea7ddf0b5
---

# Exploring the Capability Boundaries of LLMs in Mastering of Chinese Chouxiang Language

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.15841](https://arxiv.org/abs/2604.15841)  
**Code**: [GitHub](https://github.com/csdq777/Mouse)  
**Area**: LLM Evaluation  
**Keywords**: Chouxiang Language, Internet Subculture Language, LLM Benchmark, Chinese Internet Slang, Cross-cultural Understanding

## TL;DR
This paper introduces "Chouxiang Language," a Chinese internet subculture language, to the NLP community and constructs Mouse, the first evaluation benchmark (comprising six tasks: translation, representation classification, intent recognition, toxicity detection, meaning selection, and cloze test). It discovers that while SOTA LLMs perform reasonably in contextual semantic understanding, they exhibit significant limitations in other tasks.

## Background & Motivation

**Background**: LLMs perform excellently on standard NLP tasks, but their ability to process non-standard linguistic variants—particularly non-Western subculture languages—remains largely unexplored. Existing research on Chinese internet language is primarily limited to toxicity detection and perturbed language detection.

**Limitations of Prior Work**: (1) Existing LLMs and benchmarks exhibit a significant Western-centric bias, largely ignoring Chinese subculture languages; (2) As the most widely used internet subculture language among Chinese youth, Chouxiang Language integrates complex mechanisms such as homophone substitution, visual symbol analogy, and semantic mapping, far exceeding the complexity of emojis and memes; (3) Previous studies restricted such linguistic phenomena to negative dimensions (toxicity), overlooking their evolved neutral or even positive functions.

**Key Challenge**: Chouxiang Language creates a non-standard semantic space deviating from standard Chinese through multi-layer mappings (homophones $\rightarrow$ glyph decomposition $\rightarrow$ semantic metaphors $\rightarrow$ dialect borrowing). However, LLM pre-training is based mainly on standard text, resulting in systematic capability deficiencies when understanding and processing this subculture encoding.

**Goal**: To formally define the classification system of Chouxiang Language, construct the Mouse evaluation benchmark, and systematically evaluate the capability boundaries of SOTA LLMs on Chouxiang Language.

**Key Insight**: From a semiological perspective, Chouxiang Language is categorized into three core dimensions: homophonic, visual, and semantic, combined with a dual-dimension classification system for intent recognition.

**Core Idea**: The first NLP benchmark for Chouxiang Language, comprehensively evaluating LLMs' ability to understand, translate, and utilize subculture language across six tasks.

## Method

### Overall Architecture
Mouse decomposes the "understanding of Chouxiang Language" into a capability spectrum spanning from passive decoding to active generation. First, 1,099 Chouxiang Evaluation Instances (CXEI) are collected and annotated from real Chinese internet corpora, each with standard Chinese reference translations, representation dimensions, intent, and toxicity labels. Subsequently, six tasks—translation, representation classification, intent recognition, toxicity detection, meaning selection, and cloze test—are designed around these instances. Translation is scored using LLM-as-Judge, while other tasks are evaluated via accuracy/balanced accuracy, ultimately mapping the capability from "grasping meaning" to "practical usage."

### Key Designs

**1. Three-dimensional representation classification system: Deconstructing the linguistic composition of Chouxiang Language from a semiological perspective.**

Previous studies categorized Chouxiang Language by origin (symbols/homophones/dialects/emojis), where features often overlapped and were difficult to operationalize. This paper adopts a semiological approach, merging word-formation mechanisms into three orthogonal dimensions: the Phonetic dimension uses Chinese phonetic redundancy for homophone substitution (e.g., "Zhubao" $\rightarrow$ "Streamer"), the Visual dimension leverages the pictographic nature of Chinese characters and the graphic attributes of emojis (e.g., "彳亍口巴" restored to "行吧" via radical assembly), and the Semantic dimension realizes mapping through literalization of symbols or dialect borrowing (e.g., "stepping on skin" $\rightarrow$ "stepping on a banana peel"). These dimensions can overlap in a single sentence, allowing multi-label annotation for each instance.

**2. Dual-dimension evaluation instances (CXEI): Packaging linguistic decoding and pragmatic understanding into a single evaluation unit.**

Chouxiang Language is not merely a word game but an interactive tool carrying social intent; focusing solely on translation accuracy misses its pragmatic layer. Each CXEI is fixed with five attributes: the original text (mixing emojis/characters/Latin letters), a standard Chinese reference translation, representation dimensions (phonetic/visual/semantic), intent (comment/emotional expression/statement/memeing/urging/group identity, etc.), and a binary toxicity label. This allows an instance to support both "decodability" and "pragmatic intent" queries, preventing the decoupling of linguistic ability from social semantics.

**3. Six-task ability gradient: Progressing from passive understanding to active usage.**

A single task cannot characterize the mastery of Chouxiang Language—a model might understand context but fail to translate the original, or detect toxicity but fail to recognize intent. This paper ranks six tasks into a hierarchy: Translation tests decoding; Representation Classification tests linguistic analysis; Intent Recognition tests pragmatics; Toxicity Detection tests safety identification; Meaning Selection tests core semantic understanding; and Cloze Test tests the ability to actively generate Chouxiang Language. This forms a continuous capability gradient from "passive comprehension" to "active production."

## Key Experimental Results

### Main Results

| Task | SOTA LLM Performance | Note |
|------|-------------|------|
| Meaning Selection (MS) | Good | Adequate contextual semantic understanding |
| Cloze Test (CC) | Good | Adequate contextual adaptability |
| Translation (TR) | Poor | Failure to decode complex homophones/visual mappings |
| Representation Classification (RC) | Poor | Failure to identify linguistic components |
| Intent Recognition (IR) | Moderate | Some intents can be inferred from context |
| Toxicity Detection (TD) | Poor | Chouxiang toxicity bypasses traditional filters |

### Ablation Study

| Analysis Dimension | Finding |
|---------|------|
| Phonetic vs. Visual vs. Semantic | Visual dimension is the most difficult to process |
| LLM-as-Judge vs. Human Agreement | Moderate agreement in translation evaluation |
| Key Factors Affecting Translation | Complexity of homophone chains is the primary factor |

### Key Findings
- SOTA LLMs perform reasonably on tasks involving contextual semantic understanding (MS, CC) but poorly on tasks requiring the decoding of specific encoding mechanisms (TR, RC).
- This indicates that LLMs understand "what is being said" but not "how it is said"—they lack awareness of the formal level of Chouxiang Language (e.g., radical decomposition, multi-level homophone chains).
- The visual dimension (radical decomposition, emoji metaphors) represents the largest blind spot for LLMs.
- Poor performance in toxicity detection suggests that Chouxiang Language can effectively bypass existing safety filters.

## Highlights & Insights
- **Formally introducing subculture language into NLP evaluation** holds significant sociolinguistic value—internet language is living and evolving, and NLP systems must keep pace.
- The six-task design forms a complete capability assessment spectrum from "understanding" to "usage," similar to proficiency testing in second language acquisition.
- A clear asymmetry is observed between LLMs' understanding of "form" versus "content"—they can grasp meaning without understanding the encoding mechanism, providing insights into the nature of LLM linguistic capabilities.

## Limitations & Future Work
- The dataset scale is limited (1,099 instances), which may not cover all variants of Chouxiang Language.
- The study focuses only on Chinese; similar subculture phenomena in other languages (e.g., Japanese netlogue, Korean abbreviations) remain to be explored.
- The reliability of LLM-as-Judge in translation evaluation remains limited.
- Toxicity annotation may be subject to the subjective bias of annotators.

## Related Work & Insights
- **vs. Chinese Emoji Research (Xie et al., 2025)**: Emojis are a subset of Chouxiang Language; the latter possesses a more complex semantic structure.
- **vs. Perturbed Language Detection (Xiao et al., 2024)**: While they focus on toxicity bypass, this study treats Chouxiang Language as a neutral subcultural phenomenon.
- **vs. Cross-cultural LLM Evaluation (Cao et al., 2023)**: While they focus on value alignment bias, this study focuses on linguistic encoding capabilities.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First Chouxiang Language NLP benchmark, unique research perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across six tasks with multiple model comparisons.
- Writing Quality: ⭐⭐⭐⭐ Detailed linguistic analysis and rich examples.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Beyond Accuracy: A Cognitive Load Framework for Mapping the Capability Boundaries of Tool-use Agents](../../AAAI2026/llm_evaluation/beyond_accuracy_a_cognitive_load_framework_for_mapping_the_c.md)
- [\[ACL 2026\] SCAN: Structured Capability Assessment and Navigation for LLMs](scan_structured_capability_assessment_and_navigation_for_llms.md)
- [\[ACL 2026\] Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination](can_llms_act_as_historians_evaluating_historical_research_capabilities_of_llms_v.md)
- [\[ACL 2026\] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](challenging_the_boundaries_of_reasoning_an_olympiad-level_math_benchmark_for_lar.md)
- [\[ACL 2026\] Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation: A Case Study on Tang Poetry](capabilities_and_evaluation_biases_of_large_language_models_in_classical_chinese.md)

</div>

<!-- RELATED:END -->
