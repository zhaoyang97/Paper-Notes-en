---
title: >-
  [Paper Note] Language Models Entangle Language and Culture
description: >-
  [ACL 2026][Multilingual & Machine Translation][Multilingual LLMs] This paper evaluates multilingual LLMs through general advisory questions constructed from the WildChat dataset. It finds systematic differences in respon…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Multilingual LLMs"
  - "cultural bias"
  - "language-culture entanglement"
  - "LLM evaluation"
  - "fairness"
date: 2026-05-08
content_hash: 721b81c2a6e5c1d6
---

# Language Models Entangle Language and Culture

**Conference**: ACL 2026  
**arXiv**: [2601.15337](https://arxiv.org/abs/2601.15337)  
**Code**: None  
**Area**: Multilingual / Cultural Bias  
**Keywords**: Multilingual LLMs, cultural bias, language-culture entanglement, LLM evaluation, fairness

## TL;DR

This paper evaluates multilingual LLMs through general advisory questions constructed from the WildChat dataset. It finds systematic differences in response quality and cultural context caused by different language queries—response quality for low-resource languages is significantly lower than for English, and the choice of language implicitly shifts the cultural information used in responses. These findings validate the entanglement of language and culture in LLMs using a translated version of CulturalBench.

## Background & Motivation

**Background**: LLMs such as ChatGPT are utilized by hundreds of millions of people for daily queries (health, finance, education, etc.) across multiple languages. Existing multilingual evaluations like MMMLU and BenchMAX focus primarily on MCQ tasks such as knowledge QA and mathematical reasoning, evaluating only accuracy while ignoring variations in response style and cultural context.

**Limitations of Prior Work**: (1) Existing multilingual benchmarks check for "correctness" rather than "quality"—lacking evaluation of response quality for open-ended advisory questions; (2) existing bias research triggers bias by embedding cultural cues (names, nationalities, etc.) in prompts, which does not reflect actual user query patterns; (3) no work has systematically established the relationship between language choice and cultural context.

**Key Challenge**: LLMs implicitly bind language and culture during the training process—when querying in a certain language, the model might not only produce lower-quality responses but also utilize the cultural framework associated with that language. This results in the same question receiving fundamentally different advice across different languages, causing systematic disadvantages for users of low-resource languages.

**Goal**: (1) Construct a set of general advisory questions to evaluate response quality differences across languages; (2) verify if language choice changes the cultural context of responses; (3) further validate the language-culture entanglement hypothesis through a translated version of CulturalBench.

**Key Insight**: By using culturally neutral open-ended questions (containing no cultural cues) and observing whether changing only the query language leads to shifts in the response's cultural context—this more authentically reflects real user interaction scenarios compared to existing methods that embed cultural cues.

**Core Idea**: Language and culture are entangled within LLMs—selecting different languages affects not only response quality but also implicitly activates different cultural information, leading even culture-neutral general questions to produce culturally biased responses.

## Method

### Overall Architecture

The evaluation is divided into three parts: (1) Construction of 20 culture-neutral advisory questions based on WildChat, translated into 6 languages (English, Chinese, Hindi, Brazilian Portuguese, Swahili, Hebrew); (2) Generation of responses from 5 multilingual LLMs for each language, using LLM-as-Judge to evaluate quality differences; (3) Cultural classification of responses and validation of language-culture entanglement on a translated CulturalBench.

### Key Designs

1.  **WildChat-based Culture-Neutral Question Construction**:
    - **Function**: Generate an evaluation question set representative of the real user query distribution.
    - **Mechanism**: Filtered English queries from the WildChat dataset, removed programming-related queries (due to over-concentration), retained lengths of 40-400 characters, deduplicated using fuzzywuzzy (threshold 60), generated embeddings with Qwen3-0.6b followed by HDBSCAN clustering, and manually constructed 20 questions covering domains such as health, education, investment, and job searching. All questions were intentionally designed to be culture-neutral—containing no references to countries, ethnicities, or specific cultures.
    - **Design Motivation**: Existing bias studies using prompts with cultural cues do not reflect real user behavior; culture-neutral questions more accurately reveal the model's inherent cultural bias.

2.  **LLM-as-Judge Evaluation Configuration Optimization**:
    - **Function**: Ensure the reliability of cross-lingual evaluation.
    - **Mechanism**: Tested 6 judging configurations (original/translation + varying numbers of reference answers), measuring consistency with human labels using Pearson correlation and Cohen's Kappa. The configuration "original language query + original language response + 8 random reference answers" was selected. Cohere Command-A was used as the judge model. It was also verified that the judge model lacks language bias: scores for English responses translated into Hindi remained higher than those for native Hindi responses translated into English.
    - **Design Motivation**: LLM judges may themselves have language biases, necessitating strict variable control—experiments proved quality differences stem from the responses themselves rather than judging bias.

3.  **Dual Validation of Cultural Entanglement**:
    - **Function**: Establish a causal relationship between language choice and cultural context.
    - **Mechanism**: First, after translating all non-English responses into English, an LLM-as-Judge categorized them into 6 cultures (Western, Indian, Chinese, African, Latin American, Jewish), finding that models could still identify the cultural origin of the response even after translation. Second, evaluating Qwen3-14B on a translated CulturalBench (750+ questions covering 29 regions) revealed that accuracy for the same cultural knowledge question differed significantly across languages (Kruskal-Wallis: $H=45.52$, $p=1.14\times10^{-8}$). The influence of random perturbations was excluded: performance changes with appended random strings were not significant ($H=1.02$, $p=0.80$).
    - **Design Motivation**: Quality differences alone are insufficient to prove entanglement—it must be shown that language indeed changes the cultural content of the response. Post-translation classification and CulturalBench provide two independent validation dimensions.

### Loss & Training

This is an evaluation study and does not involve model training. The Kruskal-Wallis non-parametric test was used to verify the statistical significance of cross-lingual differences.

## Key Experimental Results

### Main Results

**Kruskal-Wallis Test for Cross-Lingual Quality Differences**

| Model | H-statistic | p-value | Significance |
|------|---------|------|-----------|
| Cohere-Aya-32B | 712.80 | $8.39\times10^{-152}$ | Highly Significant |
| Cohere-Aya-8B | 721.13 | $1.33\times10^{-153}$ | Highly Significant |
| Magistral-Small | 610.81 | $9.33\times10^{-130}$ | Highly Significant |
| Qwen3-14B | 928.91 | $1.48\times10^{-198}$ | Highly Significant |
| Sarvam-m | 899.84 | $2.89\times10^{-192}$ | Highly Significant |

All models performed best in English, while Hindi, Swahili, and Hebrew consistently showed poorer performance.

### Ablation Study

**Translated CulturalBench vs. Random Perturbation Control (Qwen3-14B)**

| Condition | H-statistic | p-value | Conclusion |
|------|---------|------|------|
| Cross-lingual | 45.52 | $1.14\times10^{-8}$ | Significant Difference |
| Random String | 1.02 | 0.80 | No Significant Difference |

### Key Findings

- All five models performed significantly worse in at least one language, with English consistently being the best.
- Cohere-Aya-32B showed better cross-lingual consistency than Cohere-Aya-8B, suggesting larger models are more stable across languages.
- Although Sarvam-m and Magistral are based on the same backbone (Mistral-small-3.1-24B), they showed different strengths across languages due to different fine-tuning strategies—Sarvam-m was stronger in English and Hindi, while Magistral was stronger in Chinese and Portuguese.
- Cultural classification experiments showed: Hindi queries → highest proportion of responses classified as Indian culture; Chinese → Chinese culture. Cultural characteristics remained identifiable even after translation into English.

## Highlights & Insights

- Using culture-neutral questions to reveal language-culture entanglement is an ingenious experimental design—it eliminates the confounding factor of manually injected cultural cues, making the findings more persuasive.
- The control experiment for judge model bias (translating responses before judging) is a methodological merit—many multilingual evaluations overlook this potential confounder.
- The discovery of language-culture entanglement has direct practical implications for LLM deployment: users may receive culturally biased advice simply by using their native language; for example, investment advice might implicitly favor the investment habits of the culture associated with that language.

## Limitations & Future Work

- Evaluation was limited to small-to-medium-sized open-source models (up to 32B); larger models might exhibit different behaviors.
- The 20 questions have limited coverage; while based on real distributions, the sample size is small.
- Reliance on LLM-as-Judge evaluation, while validated, may still contain systematic biases.
- Coverage was limited to only 6 languages; the performance of more low-resource languages remains to be explored.
- Mechanism not explored—the root causes of language-culture entanglement (training data distribution? tokenizer?) require interpretability analysis.

## Related Work & Insights

- **vs MMMLU/BenchMAX**: These evaluate MCQ accuracy, whereas this paper evaluates open-ended response quality and cultural context—revealing a significant dimension missed by existing benchmarks.
- **vs Bąk et al. / Schlicht et al.**: These papers evaluate multilingual bias in specific domains (email/medical), whereas this study covers a broader range of general queries.
- **vs IndQA (OpenAI)**: Similar but focused only on Indian languages; this work covers languages from multiple regions and establishes a general conclusion of language-culture entanglement.

## Rating

- Novelty: ⭐⭐⭐⭐ First to systematically reveal language-culture entanglement using culture-neutral questions, though the method is primarily evaluation rather than proposing a solution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive with multiple models, multiple languages, statistical testing, judge bias control, and random perturbation controls.
- Writing Quality: ⭐⭐⭐⭐ Logical clarity with a step-by-step experimental design.
- Value: ⭐⭐⭐⭐ Direct guidance for multilingual LLM fairness and deployment, although the lack of a solution reduces practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models](the_gaoyao_benchmark_a_comprehensive_framework_for_evaluating_multilingual_and_m.md)
- [\[ACL 2026\] LLM-XTM: Enhancing Cross-Lingual Topic Models with Large Language Models](llm-xtm_enhancing_cross-lingual_topic_models_with_large_language_models.md)
- [\[ACL 2026\] Efficient Training for Cross-lingual Speech Language Models](efficient_training_for_cross-lingual_speech_language_models.md)
- [\[ACL 2026\] Multilingual Language Models Encode Script Over Linguistic Structure](multilingual_language_models_encode_script_over_linguistic_structure.md)
- [\[ACL 2026\] Why Do Multilingual Reasoning Gaps Emerge in Reasoning Language Models?](why_do_multilingual_reasoning_gaps_emerge_in_reasoning_language_models.md)

</div>

<!-- RELATED:END -->
