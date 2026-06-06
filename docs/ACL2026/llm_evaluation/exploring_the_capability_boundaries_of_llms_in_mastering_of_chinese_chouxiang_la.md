---
title: >-
  [Paper Note] Exploring the Capability Boundaries of LLMs in Mastering of Chinese Chouxiang Language
description: >-
  [ACL 2026][LLM Evaluation][Chouxiang language] This paper introduces "Chouxiang language," a Chinese internet subculture language, to the NLP community and constructs the first evaluation benchmark…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Chouxiang language"
  - "Internet subculture language"
  - "LLM benchmark"
  - "Chinese internet slang"
  - "Cross-cultural understanding"
date: 2026-05-08
content_hash: 6c25b96aca727428
---

# Exploring the Capability Boundaries of LLMs in Mastering of Chinese Chouxiang Language

**Conference**: ACL 2026  
**arXiv**: [2604.15841](https://arxiv.org/abs/2604.15841)  
**Code**: [GitHub](https://github.com/csdq777/Mouse)  
**Area**: LLM Evaluation  
**Keywords**: Chouxiang language, Internet subculture language, LLM benchmark, Chinese internet slang, Cross-cultural understanding

## TL;DR
This paper introduces "Chouxiang language," a Chinese internet subculture language, to the NLP community and constructs the first evaluation benchmark, Mouse (comprising six tasks: translation, representation classification, intent recognition, toxicity detection, meaning selection, and cloze). The study finds that while SOTA LLMs perform acceptably in contextual semantic understanding, they exhibit significant limitations across other tasks.

## Background & Motivation

**Background**: LLMs demonstrate exceptional performance on standard NLP tasks, yet their ability to process non-standard language variants—particularly non-Western subculture languages—remains largely unexplored. Existing research on Chinese internet language is primarily confined to toxicity detection and perturbed language detection.

**Limitations of Prior Work**: (1) Existing LLMs and benchmarks exhibit a pronounced Western-centric bias, with Chinese subculture languages being severely neglected; (2) Chouxiang language, as the most widely used online subculture language among Chinese youth, incorporates complex mechanisms such as homophonic substitution, visual symbolic analogy, and semantic mapping, far exceeding the complexity of standard emojis or internet slang; (3) Previous studies restricted such linguistic phenomena to negative dimensions (toxicity), overlooking their evolved neutral and even positive functions.

**Key Challenge**: Chouxiang language creates a non-standard semantic space that deviates from standard Chinese through multi-layered mappings (homophony $\rightarrow$ glyph decomposition $\rightarrow$ semantic metaphor $\rightarrow$ dialect borrowing). Conversely, LLM pre-training primarily relies on standard text, resulting in systemic capability deficits when decoding such subculture-specific encodings.

**Goal**: To formally define the classification system of Chouxiang language, construct the Mouse evaluation benchmark, and systematically evaluate the capability boundaries of SOTA LLMs regarding Chouxiang language.

**Key Insight**: From a semiotic perspective, Chouxiang language is categorized into three core dimensions—homophonic, visual, and semantic—integrated with intent classification to form a dual-dimensional taxonomy.

**Core Idea**: The first NLP benchmark for Chouxiang language, comprehensively evaluating LLM capabilities in understanding, translating, and utilizing subculture language across six tasks.

## Method

### Overall Architecture
The authors construct the Mouse benchmark containing 1,099 Chouxiang Evaluation Instances (CXEI), where each instance includes the original text, a standard Chinese reference translation, representation classification labels, intent labels, and toxicity labels. Six tasks are designed: Translation (TR), Representation Classification (RC), Intent Recognition (IR), Toxicity Detection (TD), Meaning Selection (MS), and Cloze (CC).

### Key Designs

1.  **Three-dimensional Representation Classification System for Chouxiang Language**:
    - **Function**: To systematically describe the linguistic composition of Chouxiang language.
    - **Mechanism**: The **homophonic dimension** utilizes Chinese phonetic redundancy for homophone substitution (e.g., "主包" $\rightarrow$ "主播" [Anchor]); the **visual dimension** exploits the pictographic nature of Chinese characters and the graphic attributes of emojis (e.g., "彳亍口巴" $\rightarrow$ "行吧" [Okay] via radical decomposition); the **semantic dimension** maps meanings through direct symbolic literalization or dialect borrowing (e.g., "踩到皮" $\rightarrow$ "踩到香蕉皮" [Slipped on a banana skin]). A single sentence can span multiple dimensions.
    - **Design Motivation**: Previous classifications based on origin (symbols/homophones/dialects/emojis) suffered from overlapping features; a semiotic perspective is more systematic and operational.

2.  **Dual-dimensional Evaluation Instances (CXEI)**:
    - **Function**: To provide comprehensive evaluation units.
    - **Mechanism**: Each CXEI contains five attributes—original text (mixed emoji/Hanzi/Latin script), reference translation (standard Chinese), representation dimensions (homophonic/visual/semantic), intent (commentary, emotional expression, statement, sexual innuendo, memetic usage, urgency, group identity, etc.), and toxicity (binary label).
    - **Design Motivation**: Chouxiang language is not only a linguistic phenomenon but also a social interaction tool, necessitating the evaluation of both linguistic decoding and pragmatic understanding.

3.  **Six-task Evaluation Design**:
    - **Function**: To assess LLM mastery of Chouxiang language across different levels.
    - **Mechanism**: Translation (decoding capability) $\rightarrow$ Representation Classification (linguistic analysis) $\rightarrow$ Intent Recognition (pragmatic understanding) $\rightarrow$ Toxicity Detection (safety recognition) $\rightarrow$ Meaning Selection (core semantic understanding) $\rightarrow$ Cloze (active usage). This forms a capability gradient from passive understanding to active generation.
    - **Design Motivation**: Single tasks cannot provide a holistic evaluation—an LLM might understand context but fail to translate, or detect toxicity while failing to recognize intent.

### Loss & Training
This work is purely evaluative and does not involve model training. Translation quality is scored using LLM-as-Judge, while other tasks use Accuracy or Balanced Accuracy.

## Key Experimental Results

### Main Results

| Task | SOTA LLM Performance | Description |
| :--- | :--- | :--- |
| Meaning Selection (MS) | Good | Adequate contextual semantic understanding |
| Cloze (CC) | Good | Adequate contextual adaptability |
| Translation (TR) | Poor | Failure to decode complex homophonic/visual mappings |
| Representation Classification (RC) | Poor | Inability to identify linguistic components |
| Intent Recognition (IR) | Moderate | Some intents can be inferred from context |
| Toxicity Detection (TD) | Poor | Chouxiang language successfully bypasses traditional filters |

### Ablation Study

| Analysis Dimension | Finding |
| :--- | :--- |
| Homophonic vs. Visual vs. Semantic | The visual dimension is the most difficult to process |
| LLM-as-Judge vs. Human Consistency | Medium consistency in translation evaluation |
| Key Factors Affecting Translation | Complexity of homophonic chains is the primary factor |

### Key Findings
- SOTA LLMs perform acceptably in tasks involving contextual semantic understanding (MS, CC), but poorly in tasks requiring the decoding of unique encoding mechanisms (TR, RC).
- This indicates that LLMs understand "what is being said" but not "how it is being said"—they lack awareness of the formal level of Chouxiang language (e.g., glyph decomposition, multi-level homophonic chains).
- The **visual dimension** (glyph decomposition, emoji metaphors) represents the most significant blind spot for LLMs.
- Poor performance in toxicity detection suggests that Chouxiang language can effectively bypass current safety filters.

## Highlights & Insights
- **Formally introducing subculture language into NLP evaluation** holds significant sociolinguistic value—internet language is living and evolving, and NLP systems must keep pace.
- The six-task design provides a complete capability evaluation spectrum from "understanding" to "usage," similar to proficiency testing in second language acquisition.
- The discovery of a clear asymmetry between the understanding of "form" and "content" in LLMs—where they grasp meaning but not the encoding mechanism—offers insights into the nature of LLM linguistic competence.

## Limitations & Future Work
- The dataset scale is limited (1,099 instances), which may not cover all variants of Chouxiang language.
- The study focuses solely on Chinese; similar subculture phenomena in other languages (e.g., Japanese netlspeak, Korean abbreviations) remain to be explored.
- The reliability of LLM-as-Judge in translation evaluation is somewhat limited.
- Toxicity labeling may be subject to the subjective bias of annotators.

## Related Work & Insights
- **vs. Chinese Meme Research (Xie et al., 2025)**: Memes are a subset of Chouxiang language; however, the semantic structure of Chouxiang language is more complex.
- **vs. Perturbed Language Detection (Xiao et al., 2024)**: While they focus on toxicity evasion, this paper treats Chouxiang language as a neutral subcultural phenomenon.
- **vs. Cross-cultural LLM Evaluation (Cao et al., 2023)**: They focus on value bias, whereas this paper emphasizes linguistic encoding capabilities.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First NLP benchmark for Chouxiang language, unique research perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across six tasks and multiple models.
- **Writing Quality**: ⭐⭐⭐⭐ Detailed linguistic analysis with rich examples.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination](can_llms_act_as_historians_evaluating_historical_research_capabilities_of_llms_v.md)
- [\[ACL 2026\] SCAN: Structured Capability Assessment and Navigation for LLMs](scan_structured_capability_assessment_and_navigation_for_llms.md)
- [\[ACL 2026\] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](challenging_the_boundaries_of_reasoning_an_olympiad-level_math_benchmark_for_lar.md)
- [\[ACL 2026\] TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice](taxpraben_a_scalable_benchmark_for_structured_evaluation_of_llms_in_chinese_real.md)
- [\[AAAI 2026\] Beyond Accuracy: A Cognitive Load Framework for Mapping the Capability Boundaries of Tool-use Agents](../../AAAI2026/llm_evaluation/beyond_accuracy_a_cognitive_load_framework_for_mapping_the_c.md)

</div>

<!-- RELATED:END -->
