---
title: >-
  [Paper Note] CuLEmo: Cultural Lenses on Emotion - Benchmarking LLMs for Cross-Cultural Emotion Understanding
description: >-
  [LLM Evaluation] This paper proposes CuLEmo, the first multilingual benchmark dataset for evaluating culture-aware emotion prediction. Spanning 6 languages/cultures (Amharic, Arabic, English, German, Hindi, and Spanish), it evaluates the cross-cultural emotion understanding capabilities of LLMs across 400 culturally-relevant scenarios, revealing significant cultural variations in emotional expression and highly varying performance among LLMs.
tags:
  - "LLM Evaluation"
date: 2026-05-08
content_hash: 3d587ac8d080cd14
---

# CuLEmo: Cultural Lenses on Emotion - Benchmarking LLMs for Cross-Cultural Emotion Understanding

## General Information

- **Conference**: ACL2025
- **arXiv**: [2503.10688](https://arxiv.org/abs/2503.10688)
- **Code**: [https://github.com/llm-for-emotion/culemo](https://github.com/llm-for-emotion/culemo)
- **Area**: LLM Evaluation
- **Keywords**: Cross-Cultural Emotion Understanding, Multilingual LLM Evaluation, Culture Awareness, Emotion Prediction, Sentiment Analysis

## TL;DR

This paper proposes CuLEmo, the first multilingual benchmark dataset for evaluating culture-aware emotion prediction. Spanning 6 languages/cultures (Amharic, Arabic, English, German, Hindi, and Spanish), it evaluates the cross-cultural emotion understanding capabilities of LLMs across 400 culturally-relevant scenarios, revealing significant cultural variations in emotional expression and highly varying performance among LLMs.

## Background & Motivation

### Problem Definition
Emotion is language- and culture-dependent. The same event can evoke completely different emotional responses across different cultures. For instance, not leaving a tip at a restaurant might trigger guilt in the US, but it is normal in China, and tipping might even be considered offensive in Japan. Do current LLMs possess culture-aware emotion understanding capabilities?

### Limitations of Prior Work

**Keyword Dependency**: Most existing emotion benchmarks rely on keyword-based emotion recognition, ignoring deeper cultural dimensions.

**Translation Bias**: Many cross-lingual emotion datasets are obtained by translating English annotated data, which may introduce incomplete or misleading insights.

**Insufficient Cultural Coverage**: Previous studies were restricted to limited culture/language combinations (e.g., only US vs. Japan) and limited emotion categories (e.g., only pride/shame).

**Annotation Inequity**: Labels for translated data originate from an Anglocentric cultural perspective, failing to reflect authentic local cultural standpoints.

### Research Questions
- RQ1: Can LLMs provide culture-aware emotional responses?
- RQ2: Which cultures are better represented in LLMs?
- RQ3: Can LLMs identify national culture solely through the prompt language?
- RQ4: How does prompt language affect culture-aware emotion understanding in LLMs?

## Method

### Overall Architecture

CuLEmo comprises 400 culturally-relevant events/scenarios, with each scenario presented in the form of "How would you feel when...", spanning 6 languages and cultures.

### Dataset Construction

**Event Collection**:
- Human-created scenarios + web search + LLM prompting.
- Covers traditions, events, norms, and behaviors in each target country.
- Does not contain explicit emotional keywords (distinguishing it from traditional emotion datasets).
- References the ISEAR data format: "When I … situations that cause a specific emotion".

**Ten Categories**:

| Category | Number of Scenarios |
|---|---|
| Family Relations | 45 |
| Social Etiquette and Interaction | 65 |
| Personal Appearance and Dress Code | 32 |
| Cultural and Religious Practices | 62 |
| Sex and Intimacy | 38 |
| Professional Scenarios | 28 |
| Dining Etiquette | 35 |
| Personal Privacy | 25 |
| Emotional and Psychological Situations | 40 |
| Public Behavior and Norms | 30 |

**Translation Pipeline**:
- Initial translation into 5 target languages (Arabic, Amharic, German, Hindi, Spanish) using Google Translate.
- Review and correction by native speakers (translation completed prior to annotation to avoid label bias).

**Annotation**:
- Amazon Mechanical Turk + custom POTATO annotation tool.
- At least 5 native annotators from the target country for each instance.
- An additional 2 annotators were recruited when there was no majority vote.
- 6 emotion categories: joy, fear, sadness, anger, guilt, neutral.
- Compensation: \$12/hour.

### Key Designs

**Language and Cultural Coverage Considerations**:
- Typological diversity: 5 languages, 4 writing systems.
- Geographical diversity: East vs. West.
- Resource availability: Low-resource vs. high-resource languages.
- 6 target countries: UAE, USA, Germany, Ethiopia, India, Mexico.

**Prompt Construction**:
- English prompt: All instructions, inputs, and expected outputs are in English.
- In-language prompt: All components in the target language.
- With/without country context: "You live in <<country name>>,".

**Evaluation Tasks**:
1. Emotion prediction (6-class classification: joy, fear, sadness, anger, guilt, neutral)
2. Sentiment analysis (3-class classification: positive, negative, neutral)

## Experiments

### Model Selection
- **Open-source**: LLaMA-3 (3.2-3B, 3.1-8B), Gemma (2B, 9B), Aya (expanse-8b, 101-13B), Ministral (3B, 8B)
- **Closed-source**: GPT (3.5, 4), Gemini-1.5, Claude (3.5-sonnet, 3-opus)

### Main Results (Emotion Prediction, with Country Context)

| LLM | USA(EN) | UAE(EN) | UAE(AR) | Germany(EN) | Germany(DE) | India(EN) | India(HI) | Mexico(ES) |
|---|---|---|---|---|---|---|---|---|
| GPT-4 | 0.60 | 0.55 | 0.48 | 0.51 | 0.50 | 0.40 | 0.40 | 0.65 |
| Ministral-8B | 0.65 | 0.61 | 0.06 | 0.58 | **0.72** | 0.39 | 0.19 | 0.32 |
| Gemini1.5-flash | 0.56 | 0.56 | 0.56 | 0.46 | 0.48 | 0.41 | 0.41 | 0.64 |
| Claude-3.5-sonnet | 0.57 | 0.48 | 0.54 | 0.46 | 0.42 | 0.40 | 0.36 | 0.61 |
| Gemma-2-2B | 0.62 | 0.59 | 0.45 | **0.64** | 0.54 | 0.38 | 0.29 | 0.56 |

### Cultural Differences in Annotations

**Distribution of Emotion Labels**:
- German annotations exhibit the highest proportion of neutral (no emotion) labels (87%).
- Amharic (29.5%) and Arabic (22%) have the highest proportion of the anger category.
- Indian annotations show the greatest divergence from other countries (with only 29% agreement with Germany).

**Typical Cases**:
- "Not leaving a tip for a waiter": USA = guilt, UAE = neutral, Ethiopia = guilt.
- "Wearing black clothes to a wedding": USA = sadness, Ethiopia = anger, others = neutral.
- "A woman wearing shorts in the street": Ethiopia = anger, others = neutral.

### Cultural Representation Bias

Without country context, LLMs perform more accurately on USA, Mexico, and Germany, while dropping significantly on UAE, Ethiopia, and India, which implies certain cultures are more prevalent in the training data.

### Impact of Prompt Language

- Accuracy drops significantly after removing the country context.
- Incorporating "You live in <<country name>>" consistently improves performance (GPT-4 Hindi +21%, Amharic Claude-3.5 +9%).
- **Language alone cannot reliably convey cultural context**.
- For low-resource languages, an English prompt + country context often outperforms in-language prompting.

### Sentiment Analysis (3-class)

- Coarse-grained classification performs better than fine-grained emotion prediction.
- GPT-4 Hindi observes an improvement of +22%.
- Both Claude-3-opus and GPT-4 achieve 75% on Mexican culture (Spanish).

### Key Findings
1. Emotions are indeed culture-dependent, with identical events evoking drastically different emotions across cultures.
2. LLMs perform better on high-resource cultures (USA, Western Europe) and degrade significantly on low-resource cultures (India, Ethiopia).
3. Explicitly providing country context is critical for culture-aware emotion understanding.
4. Larger models are not always better: Gemma-2-2B and Ministral-8B compete with or even outperform closed-source models in certain cultures.

## Highlights & Insights

1. **First Culture-Aware Emotion Benchmark**: Rather than relying on translated annotations, it performs independent annotations across cultures for the same scenarios.
2. **Key Discovery of "Language $\neq$ Culture"**: Prompt language alone cannot convey cultural context; the country must be explicitly specified.
3. **Performance Gap in Low-Resource Cultures**: This reveals cultural bias inherent in the training data of LLMs.
4. **Practical Recommendation**: For low-resource languages, using English prompting combined with specifying the target country is often the optimal strategy.
5. **Ingenious Annotation Design**: Scenarios contain no explicit emotion keywords, demanding genuine cultural reasoning rather than simple keyword matching.

## Limitations & Future Work

1. **Limited Number of Events**: With only 400 questions per language, it is difficult to comprehensively cover cultural variations.
2. **Limited Emotion Categories**: Only 6 categories are included, lacking fine-grained categories such as surprise or disgust.
3. **Drawbacks of Majority Voting**: It excludes minority perspectives, potentially losing cultural diversity.
4. **Constrained Model Scale**: Only small to medium-sized open-source models are evaluated; the performance of larger models remains unknown.
5. **Coverage of Only 6 Languages/Countries**: Generalizability is limited.

## Related Work

- **Culture-Aware NLP**: CulturalBench (Liu et al., 2024), NORMSAGE (Fung et al., 2023)
- **Cross-Lingual Emotion**: XLM-EMO (Bianchi et al., 2022), De Bruyne (2023)
- **Cultural Sentiment Analysis**: Havaldar et al. (2023) — US/Japan pride/shame, Ahmad et al. (2024) — Hausa
- **Multilingual LLM Evaluation**: MEGA (Ahuja et al., 2023)

## Rating ⭐⭐⭐⭐

- Novelty: ⭐⭐⭐⭐ — First culture-aware emotion benchmark, with a clearly defined problem.
- Value: ⭐⭐⭐⭐ — Directly guides the construction of culture-sensitive AI systems.
- Methodological Novelty: ⭐⭐⭐ — Primarily focused on benchmark construction and evaluation, without introducing new models or methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive comparison featuring 11 models, multiple prompt configurations, and dual tasks (emotion prediction and sentiment analysis).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents](androidlab_autonomous_agent.md)
- [\[ACL 2025\] Atomic Calibration of LLMs in Long-Form Generations](atomic_calibration_of_llms_in_long-form_generations.md)
- [\[ACL 2025\] Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models](retrieval_models_arent_tool-savvy_benchmarking_tool_retrieval_for_large_language.md)
- [\[ACL 2025\] EcomScriptBench: A Multi-task Benchmark for E-commerce Script Planning via Step-wise Intention-Driven Product Association](ecomscriptbench.md)
- [\[ACL 2025\] A Conformal Risk Control Framework for Granular Word Assessment and Uncertainty Calibration of CLIPScore Quality Estimates](a_conformal_risk_control_framework_for_granular_word_assessment_and_uncertainty_.md)

</div>

<!-- RELATED:END -->
