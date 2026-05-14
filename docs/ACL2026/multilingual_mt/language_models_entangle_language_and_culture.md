---
title: >-
  [Paper Note] Language Models Entangle Language and Culture
description: >-
  [ACL 2026][Multilingual & Machine Translation][Multilingual LLM] This paper evaluates multilingual LLMs on culturally neutral, open-ended advice-seeking questions derived from the WildChat dataset. It finds that query la…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Multilingual LLM"
  - "Cultural Bias"
  - "Language-Culture Entanglement"
  - "LLM Evaluation"
  - "Fairness"
date: 2026-05-08
content_hash: 75768a0f6d330629
---

# Language Models Entangle Language and Culture

**Conference**: ACL 2026
**arXiv**: [2601.15337](https://arxiv.org/abs/2601.15337)
**Code**: None
**Area**: Multilingual / Cultural Bias
**Keywords**: Multilingual LLM, Cultural Bias, Language-Culture Entanglement, LLM Evaluation, Fairness

## TL;DR

This paper evaluates multilingual LLMs on culturally neutral, open-ended advice-seeking questions derived from the WildChat dataset. It finds that query language systematically affects both response quality and cultural context — low-resource language queries yield notably lower quality responses than English, and language choice implicitly shifts the cultural framing of responses. A translated version of CulturalBench further validates the entanglement between language and culture in LLMs.

## Background & Motivation

**Background**: LLMs such as ChatGPT are used by hundreds of millions of people for everyday queries spanning health, finance, and education, with users interacting across a wide range of languages. Existing multilingual evaluations — including MMMLU and BenchMAX — predominantly focus on multiple-choice tasks such as knowledge QA and mathematical reasoning, assessing accuracy while neglecting variation in response style and cultural context.

**Limitations of Prior Work**: (1) Existing multilingual benchmarks assess correctness but not quality — open-ended advice-seeking responses remain unevaluated. (2) Prior bias studies elicit cultural bias by embedding explicit cues (e.g., names, nationalities) in prompts, which does not reflect real user query behavior. (3) No prior work has systematically established a relationship between language choice and the cultural context of model responses.

**Key Challenge**: LLMs implicitly bind language to culture during training. Querying in a given language may not only degrade response quality but also activate a cultural framework associated with that language, causing the same question to yield substantively different advice across languages — systematically disadvantaging users of low-resource languages.

**Goal**: (1) Construct a culturally neutral set of advice-seeking questions to evaluate response quality variation across languages; (2) determine whether language choice alters the cultural context of responses; (3) further validate the language-culture entanglement hypothesis using a translated version of CulturalBench.

**Key Insight**: Culturally neutral open-ended questions — containing no cultural cues — are used to observe whether changing only the query language shifts the cultural context of responses. This better reflects real user interaction than the existing approach of injecting explicit cultural markers.

**Core Idea**: Language and culture are entangled in LLMs — language choice not only affects response quality but also implicitly activates cultural information, causing culturally biased responses even to culturally neutral, general-purpose questions.

## Method

### Overall Architecture

The evaluation consists of three components: (1) constructing 20 culturally neutral advice-seeking questions from WildChat, translated into 6 languages (English, Chinese, Hindi, Brazilian Portuguese, Swahili, and Hebrew); (2) generating responses from 5 multilingual LLMs in each language and evaluating quality differences using LLM-as-Judge; (3) classifying responses by cultural origin and validating language-culture entanglement on a translated version of CulturalBench.

### Key Designs

1. **Culturally Neutral Question Construction from WildChat**

    - **Function**: Generate an evaluation question set representative of the real-world user query distribution.
    - **Mechanism**: English queries are filtered from WildChat by removing programming-related queries (over-represented), retaining queries of 40–400 characters, and deduplicating with fuzzywuzzy (threshold 60). Qwen3-0.6b is used to generate embeddings, followed by HDBSCAN clustering. After manual analysis, 20 questions spanning health, education, investment, and job-seeking are constructed. All questions are deliberately designed to be culturally neutral, containing no references to any country, ethnicity, or culture.
    - **Design Motivation**: Existing bias studies use prompts with embedded cultural cues, which does not reflect authentic user behavior. Culturally neutral questions more accurately reveal the model's intrinsic cultural tendencies.

2. **LLM-as-Judge Configuration Optimization**

    - **Function**: Ensure the reliability of cross-lingual evaluation.
    - **Mechanism**: Six judging configurations are tested (original vs. translated queries/responses, varying numbers of reference responses), with Pearson correlation and Cohen's Kappa used to measure agreement with human annotations. The final configuration — original-language query, original-language response, and 8 randomly sampled reference responses — is selected. Cohere Command-A serves as the judge model. The absence of language bias in the judge is verified: English responses translated into Hindi receive higher scores than native Hindi responses translated into English.
    - **Design Motivation**: LLM judges may themselves exhibit language bias; rigorous variable control is necessary. The experiment confirms that observed quality differences stem from the responses themselves rather than from judge bias.

3. **Dual Validation of Cultural Entanglement**

    - **Function**: Establish a causal relationship between language choice and cultural context.
    - **Mechanism**: In the first step, all non-English responses are translated into English and classified by an LLM-as-Judge into one of six cultural categories (Western, Indian, Chinese, African, Latin American, Jewish); the model is found to reliably identify the cultural origin of responses even after translation. In the second step, Qwen3-14B is evaluated on a translated version of CulturalBench (750+ questions covering 29 regions); accuracy on the same cultural knowledge questions varies significantly across languages (Kruskal-Wallis: $H=45.52$, $p=1.14\times10^{-8}$). The effect of random perturbations is also ruled out: appending random character strings produces no significant performance change ($H=1.02$, $p=0.80$).
    - **Design Motivation**: Quality differences alone are insufficient to establish entanglement — it must be demonstrated that language genuinely alters the cultural content of responses. Cultural classification after translation and CulturalBench evaluation provide two independent dimensions of validation.

### Loss & Training

This is an evaluation study and involves no model training. Statistical significance of cross-lingual differences is assessed using the Kruskal-Wallis non-parametric test.

## Key Experimental Results

### Main Results

**Kruskal-Wallis Test for Cross-Lingual Quality Differences**

| Model | H Statistic | p-value | Significance |
|-------|-------------|---------|--------------|
| Cohere-Aya-32B | 712.80 | $8.39\times10^{-152}$ | Highly significant |
| Cohere-Aya-8B | 721.13 | $1.33\times10^{-153}$ | Highly significant |
| Magistral-Small | 610.81 | $9.33\times10^{-130}$ | Highly significant |
| Qwen3-14B | 928.91 | $1.48\times10^{-198}$ | Highly significant |
| Sarvam-m | 899.84 | $2.89\times10^{-192}$ | Highly significant |

All models perform best in English, with Hindi, Swahili, and Hebrew consistently underperforming.

### Ablation Study

**Translated CulturalBench vs. Random Perturbation Control (Qwen3-14B)**

| Condition | H Statistic | p-value | Conclusion |
|-----------|-------------|---------|------------|
| Cross-lingual | 45.52 | $1.14\times10^{-8}$ | Significant difference |
| Random string | 1.02 | 0.80 | No significant difference |

### Key Findings

- All 5 models perform significantly worse in at least one language; English consistently yields the highest quality.
- Cohere-Aya-32B exhibits greater cross-lingual consistency than Cohere-Aya-8B, suggesting that larger models are more stable across languages.
- Sarvam-m and Magistral share the same base model (Mistral-small-3.1-24B) but differ in cross-lingual strengths due to distinct fine-tuning strategies — Sarvam-m performs better in English and Hindi, while Magistral performs better in Chinese and Portuguese.
- Cultural classification experiments show that Hindi queries yield responses most frequently categorized as Indian culture, and Chinese queries yield responses most frequently categorized as Chinese culture — cultural characteristics remain identifiable even after translation into English.

## Highlights & Insights

- Using culturally neutral questions to reveal language-culture entanglement is an elegant experimental design — it eliminates the confound of artificially injected cultural cues, making the findings more convincing.
- The control experiment for judge model bias (re-evaluating translated responses) is a methodological strength that many multilingual evaluation studies overlook.
- The language-culture entanglement finding has direct practical implications for LLM deployment: users may receive culturally biased advice simply by querying in their native language — for instance, investment advice may implicitly reflect the financial norms associated with the query language's culture.

## Limitations & Future Work

- Only small-to-medium-scale open-source models (up to 32B parameters) are evaluated; results may differ for larger models.
- The 20-question set offers limited coverage; while grounded in a real-world query distribution, the sample size remains small.
- Evaluation relies on LLM-as-Judge; despite validation, systematic bias may persist.
- Only 6 languages are covered; the behavior of additional low-resource languages remains unexplored.
- The underlying mechanism is not investigated — the root causes of language-culture entanglement (training data distribution? tokenizer design?) require interpretability analysis.

## Related Work & Insights

- **vs. MMMLU/BenchMAX**: These benchmarks assess MCQ accuracy; this paper evaluates open-ended response quality and cultural context, revealing an important dimension that existing benchmarks miss.
- **vs. Bąk et al. / Schlicht et al.**: These works evaluate multilingual bias in specific domains (email / medical); this paper covers a broader range of general-purpose queries.
- **vs. IndQA (OpenAI)**: A related effort but limited to Indian languages; this paper spans multiple regional languages and establishes a more general conclusion about language-culture entanglement.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic use of culturally neutral questions to reveal language-culture entanglement, though the contribution is primarily evaluative rather than solution-oriented.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers multiple models and languages with statistical testing, judge bias control, and random perturbation baselines.
- **Writing Quality**: ⭐⭐⭐⭐ — Logically clear with a well-structured, progressive experimental design.
- **Value**: ⭐⭐⭐⭐ — Directly relevant to multilingual LLM fairness and deployment, though the absence of proposed solutions reduces its practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Focusing on Language: Revealing and Exploiting Language Attention Heads in Multilingual Large Language Models](../../AAAI2026/multilingual_mt/focusing_on_language_revealing_and_exploiting_language_attention_heads_in_multil.md)
- [\[ACL 2026\] The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models](the_gaoyao_benchmark_a_comprehensive_framework_for_evaluating_multilingual_and_m.md)
- [\[ACL 2026\] Efficient Training for Cross-lingual Speech Language Models](efficient_training_for_cross-lingual_speech_language_models.md)
- [\[ACL 2026\] Multilingual Language Models Encode Script Over Linguistic Structure](multilingual_language_models_encode_script_over_linguistic_structure.md)
- [\[ACL 2026\] Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality](language_on_demand_knowledge_at_core_composing_llms_with_encoder-decoder_transla.md)

</div>

<!-- RELATED:END -->
