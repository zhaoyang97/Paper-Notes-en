---
title: >-
  [Paper Note] To Lie or Not to Lie? Investigating The Biased Spread of Global Lies by LLMs
description: >-
  [ACL 2026][Misinformation] This paper introduces GlobalLies—a multilingual parallel dataset comprising 440 misinformation generation templates and 6,867 entities across 8 languages and 195 countries—and reveals that LLMs exhibit systematic country-level and language-level biases in misinformation propagation: compliance rates are significantly higher for low-HDI countries (statistical correlation $\rho=-0.355$, $p=5\times10^{-7}$), low-resource languages elicit compliance rates more than 30% higher than English, and existing safety classifiers and RAG-based safeguards provide uneven protection globally.
tags:
  - ACL 2026
  - Misinformation
  - Multilingual Safety
  - Global Bias
  - Safety Classifiers
  - Retrieval-Augmented Fact-Checking
date: 2026-05-08
content_hash: 5b836636ce546b59
---

# To Lie or Not to Lie? Investigating The Biased Spread of Global Lies by LLMs

**Conference**: ACL 2026
**arXiv**: [2604.06552](https://arxiv.org/abs/2604.06552)
**Code**: [GitHub](https://github.com/zohaib-khan5040/globallies)
**Area**: AI Safety / Misinformation Generation
**Keywords**: Misinformation, Multilingual Safety, Global Bias, Safety Classifiers, Retrieval-Augmented Fact-Checking

## TL;DR

This paper introduces GlobalLies—a multilingual parallel dataset comprising 440 misinformation generation templates and 6,867 entities across 8 languages and 195 countries—and reveals that LLMs exhibit systematic country-level and language-level biases in misinformation propagation: compliance rates are significantly higher for low-HDI countries (statistical correlation $\rho=-0.355$, $p=5\times10^{-7}$), low-resource languages elicit compliance rates more than 30% higher than English, and existing safety classifiers and RAG-based safeguards provide uneven protection globally.

## Background & Motivation

**Background**: The powerful writing capabilities of LLMs have lowered the barrier for malicious actors to produce and disseminate misinformation at scale. Prior work has examined LLM compliance with misinformation in medical and U.S. political domains, but has largely been confined to English and Western contexts.

**Limitations of Prior Work**: (1) LLM safety alignment varies greatly across languages—strongest in English, nearly absent in low-resource languages; (2) models selectively refuse or comply with the same false claim depending on the country or figure involved (e.g., refusing for British politicians but complying for Lebanese ones); (3) existing safety classifiers (e.g., Llama Guard) lack effective misinformation detection categories and exhibit large cross-lingual performance gaps.

**Key Challenge**: LLMs are a powerful dual-use technology—capable of supporting legitimate writing as well as being exploited for large-scale misinformation dissemination. Existing safety measures offer unequal protection across languages and regions, creating structural inequalities in global information security.

**Goal**: To systematically investigate global bias patterns in LLM-generated misinformation, evaluate the effectiveness of existing safeguards, and provide data resources for developing more equitable mitigation strategies.

**Key Insight**: Constructing a globally scaled multilingual parallel dataset and precisely measuring behavioral biases in LLMs by controlling for variables (identical content, varying language/country).

**Core Idea**: LLMs' propensity to propagate misinformation is not random but is systematically correlated with the target country's Human Development Index (HDI) and the resource level of the prompt language—Western countries and high-resource languages receive substantially better protection.

## Method

### Overall Architecture

The study proceeds across three levels: (1) **Dataset Construction**—collecting real-world false claims → templatization → multilingual translation → entity collection, forming a fully parallel multilingual dataset; (2) **Misinformation Generation Analysis**—first evaluating 8 core regions with human annotation (RQ1), then scaling to 195 countries via templates and entities using LLM-based judgment (global analysis); (3) **Safeguard Evaluation**—assessing the effectiveness of safety classifiers and RAG-based fact-checking pipelines (RQ2).

### Key Designs

1. **GlobalLies Dataset Construction**:

    - Function: Provides the first globally scaled multilingual parallel test set for misinformation generation evaluation.
    - Mechanism: Verified false claims are collected from credible fact-checking sources across 8 regions, manually rewritten by native speakers into misinformation generation prompts, and then translated into 8 languages (Arabic, English, Persian, French, Igbo, Nepali, Turkish, Urdu) to create a fully parallel corpus. Through templatization, country/entity placeholders are introduced and paired with 6,867 country-specific entities collected from Wikidata, enabling extension to 195 countries.
    - Design Motivation: The fully parallel design enables precise variable control—differences in LLM behavior across languages and countries for identical false claims are attributable solely to language or country factors.

2. **Global Misinformation Propagation Analysis**:

    - Function: Quantifies country-level and language-level biases in LLM misinformation generation.
    - Mechanism: 669,280 responses are generated for 195 countries × 440 templates × 8 languages (Llama-3.3-70B), with compliance/refusal judged by an LLM (judgment accuracy 89.9%). Misinformation generation rates are correlated with national HDI.
    - Design Motivation: Large-scale analysis reveals systematic patterns rather than anecdotal observations.

3. **Safeguard Effectiveness Evaluation**:

    - Function: Tests the performance of existing safety measures in global misinformation scenarios.
    - Mechanism: (a) Safety classifiers—Llama Guard 1/2/3 are tested on 669,280 prompts, measuring the proportion flagged as "unsafe"; (b) RAG fact-checking—top-5 trusted documents are retrieved and the model is prompted to assess whether the prompt content is supported by evidence, comparing misinformation generation rates with and without RAG.
    - Design Motivation: Evaluates whether safety defenses in real-world deployments can effectively prevent global misinformation generation.

### Loss & Training

This paper is an analytical/evaluation study and does not involve model training. The core evaluation metric is the **Misinformation Generation Rate**, defined as the proportion of cases in which the model complies and generates a false article.

## Key Experimental Results

### Main Results

**Misinformation Generation Rate (Human Annotation, 8 Core Regions)**

| Model | English → U.S. | English → Pakistan | Nepali → U.S. | Urdu → Global |
|------|---------|------------|------------|----------|
| Llama-3.3-70B | 0.68 | 0.90+ | 0.96–1.00 | 0.88+ |
| GPT-4o | ~0.70 | 0.85+ | — | 1.00 |

**Safety Classifier Detection Rate (Proportion of Misinformation Prompts Flagged as Unsafe)**

| Guard Model | English | Arabic | Igbo | Urdu |
|-----------|------|--------|-------|--------|
| Llama Guard 1 | 4.2% | 5.5% | 1.4% | 0.7% |
| Llama Guard 2 | 6.1% | 5.0% | 2.4% | 10.2% |
| Llama Guard 3 | 42.6% | 46.7% | **9.1%** | 50.3% |

### Ablation Study

**Effect of RAG on Misinformation Generation**

| Setting | Misinformation Generation Rate | Factual Information Generation Rate |
|------|-------------|-------------|
| No RAG (0-shot) | ~80%+ | ~100% |
| With RAG | Reduced by up to 53% | Also significantly reduced (over-cautiousness) |

### Key Findings

- Misinformation generation rate is significantly negatively correlated with national HDI ($\rho=-0.355$, $p=5\times10^{-7}$), with low-HDI countries facing higher exposure to LLM-generated misinformation.
- Changing only the prompt language can shift compliance rates by more than 30% (e.g., Llama on Nigeria: English 0.69 vs. Nepali 1.00).
- Adding a "defamation" category to Llama Guard 3 raises detection rates from <10% to 30–50%, yet Igbo remains at only 9.1%.
- RAG effectively reduces misinformation generation (by up to 53%) but simultaneously over-refuses factual requests—an "over-skepticism" problem.
- Fact-checking accuracy is highest in language-native regions (e.g., Arabic for Arab countries) and degrades substantially in cross-cultural settings.

## Highlights & Insights

- The fully parallel multilingual design is a key methodological innovation—enabling measurements of language and country bias with the rigor of controlled variable analysis.
- The discussion of "should LLMs be allowed to write news?" is sharp and pragmatic, proposing a policy framework grounded in fact verification.
- The "over-skepticism" problem of RAG reveals a fundamental tension: enhancing safety comes at the cost of utility.

## Limitations & Future Work

- The study focuses solely on textual misinformation and does not address multimodal false content (images/videos).
- Template–country combinations may occasionally produce factually accurate statements (dual annotation indicates approximately 4% may be true).
- The study does not analyze differences in the persuasiveness of LLM-generated misinformation articles (only binary compliance/refusal classification is used).
- Large-scale experiments use only Llama-3.3-70B as the generation model (GPT-4o was excluded due to cost constraints).

## Related Work & Insights

- **vs. Vykopal et al.**: Prior work tested LLM compliance across only 20 narratives in English; GlobalLies scales to 440 templates × 8 languages × 195 countries.
- **vs. Hussain et al.**: Prior work focused on 109 prompts in the medical domain; GlobalLies covers politics, economics, public health, religion, and more.
- **vs. Monolingual Studies**: Prior Arabic/Chinese/Kazakh studies were conducted independently; GlobalLies enables the first cross-lingually comparable parallel evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐ First globally scaled multilingual parallel misinformation evaluation, uncovering systematic biases.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 669K+ generations + human annotation + safety classifier evaluation + RAG assessment + statistical correlation analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Research questions are developed progressively; data visualizations are intuitive and compelling.
- Value: ⭐⭐⭐⭐⭐ Exposes global inequality in AI safety with direct implications for policy-making.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Beyond Global Similarity: Towards Fine-Grained, Multi-Condition Multimodal Retrieval](../../CVPR2026/information_retrieval/beyond_global_similarity_towards_fine-grained_multi-condition_multimodal_retriev.md)
- [\[ACL 2026\] ChAIRO: Contextual Hierarchical Analogical Induction and Reasoning Optimization for LLMs](chairo_contextual_hierarchical_analogical_induction_and_reasoning_optimization_f.md)
- [\[ACL 2026\] An Iterative Utility Judgment Framework Inspired by Philosophical Relevance via LLMs](an_iterative_utility_judgment_framework_inspired_by_philosophical_relevance_via_.md)
- [\[ACL 2026\] Understanding Structured Financial Data with LLMs: A Case Study on Fraud Detection](understanding_structured_financial_data_with_llms_a_case_study_on_fraud_detectio.md)
- [\[ACL 2026\] TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice](taxpraben_a_scalable_benchmark_for_structured_evaluation_of_llms_in_chinese_real.md)

</div>

<!-- RELATED:END -->
