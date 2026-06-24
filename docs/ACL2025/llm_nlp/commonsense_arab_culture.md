---
title: >-
  [Paper Note] Commonsense Reasoning in Arab Culture
description: >-
  [ACL 2025][LLM (Other)][Arabic Culture] This paper proposes the ArabCulture dataset (3,482 MSA questions covering 13 Arab countries, 4 regions, and 54 cultural subdomains) to systematically evaluate the Arabic cultural commonsense reasoning capabilities of multiple LLMs. The results show that even GPT-4o only achieves 90%, while most models score between 40% and 80%, revealing significant deficiencies of LLMs in understanding non-Western cultures.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Arabic Culture"
  - "Commonsense Reasoning"
  - "LLM Evaluation"
  - "Multilingual"
  - "Cultural Benchmark"
  - "ArabCulture"
date: 2026-05-08
content_hash: e7d8d99d82480c36
---

# Commonsense Reasoning in Arab Culture

**Conference**: ACL 2025  
**arXiv**: [2502.12788](https://arxiv.org/abs/2502.12788)  
**Code**: None  
**Area**: Other / Cultural Commonsense Reasoning  
**Keywords**: Arabic Culture, Commonsense Reasoning, LLM Evaluation, Multilingual, Cultural Benchmark, ArabCulture  

## TL;DR

This paper proposes the ArabCulture dataset (3,482 MSA questions covering 13 Arab countries, 4 regions, and 54 cultural subdomains) to systematically evaluate the Arabic cultural commonsense reasoning capabilities of multiple LLMs. The results show that even GPT-4o only achieves 90%, while most models score between 40% and 80%, revealing significant deficiencies of LLMs in understanding non-Western cultures.

## Background & Motivation

**Background**: Commonsense reasoning is a fundamental human cognitive ability shaped by culture. With the rapid advancement of LLMs, their commonsense reasoning capabilities have attracted significant attention. However, existing commonsense reasoning benchmarks (such as Winograd Schema, WinoGrande, and PIQA) are almost entirely grounded in Western cultural assumptions, failing to evaluate model understanding of non-Western cultures.

**Limitations of Prior Work**: (1) Existing Arabic commonsense reasoning datasets are mostly translated from English datasets via machine translation (e.g., AraDiCE-WinoGrande, AlGhafa-COPA), and translation cannot convey culturally specific knowledge (e.g., Ramadan traditions, wedding customs in different countries). (2) The ChatGPT-generated ACVA dataset (2,486 instances) is not designed for reasoning evaluation and lacks fine-grained regional information. (3) The only manually constructed AraDiCE-Culture dataset contains only 180 samples and covers just one country. Considering the Arab world's population of approximately 456 million and its significant cultural diversity, existing datasets are too small and narrow in coverage.

**Key Challenge**: The Arab world exhibits extreme cultural diversity (with unique traditions across 13 major countries and 4 major regions), yet the evaluation of LLM cultural understanding relies on translation datasets from a Western perspective. These datasets cannot capture region-specific knowledge, nor can they differentiate model understanding of cultures across different countries/regions, potentially leading to systematically biased evaluation conclusions.

**Goal**: (1) Construct a large-scale Arabic cultural commonsense reasoning benchmark manually created from scratch by native speakers, covering 13 countries, 4 regions, and 54 subdomains; (2) Systematically evaluate the Arabic cultural commonsense reasoning performance of over 30 LLMs; (3) Analyze the impacts of positional context, evaluation formats, and prompt languages on model performance.

**Key Insight**: A sentence completion task is adopted, where a premise is provided and the model must select the culturally correct ending from three syntactically and logically plausible candidate completions. The key design is that all three options are grammatically and logically valid, forcing the model to rely solely on cultural commonsense to answer correctly, thereby eliminating shortcuts through grammatical or logical cues. The dataset is entirely authored from scratch by native speakers from 13 countries, without relying on translation or web scraping.

**Core Idea**: The first Arabic cultural commonsense reasoning benchmark created from scratch by native speakers, covering 13 countries and 54 domains, with systematic evaluation exposing LLM cultural blind spots.

## Method

### Overall Architecture

ArabCulture is a sentence completion/multiple-choice question (MCQ) dataset containing 3,482 instances, all written in Modern Standard Arabic (MSA). Each instance consists of a premise and three candidate completions, with only one being culturally correct. Construction workflow: (1) Recruit 26 annotators (13 countries $\times$ 2 annotators/country) under strict screening criteria to ensure cultural representativeness; (2) Each annotator drafts 150 instances, covering 12 main topics and 54 subdomains (Food, Weddings, Celebrations, Everyday Activities, Habits, Traditional Games, Death and Mourning, Art, Childrearing, Agriculture, Family Relationships, Idioms); (3) Two-stage quality control—national representative review + peer cross-verification by another annotator from the same country (instances are discarded if answered incorrectly); (4) Country-specificity labeling (CS vs. ¬CS) to distinguish between country-specific and shared cultural knowledge. Out of the initial 3,900 instances, 3,482 were retained after two rounds of filtering.

### Key Designs

1. **Strict Annotator Screening and Training**:
    - **Function**: To ensure the cultural authenticity and representativeness of the data
    - **Mechanism**: Five strict criteria (native speaker, resident for $\ge 10$ years, deep understanding of local culture, parents from the country, and high school education or above); online training and a pilot study are conducted to ensure task understanding
    - **Design Motivation**: The "correctness" of cultural commonsense highly depends on local knowledge, and the cultural representativeness of annotators directly determines data quality

2. **Two-Stage Quality Control**:
    - **Function**: To ensure data quality and eliminate culturally ambiguous or incorrect instances
    - **Mechanism**: Stage 1—manual review of linguistic errors and guideline compliance by national representatives (paper authors); Stage 2—peer cross-verification in MCQ format by an annotator from the same country, where instances are discarded upon errors (indicating cultural ambiguity)
    - **Design Motivation**: The exclusion rate of the two rounds of filtering is approximately 10.7% (from 3,900 to 3,482), ensuring that each retained instance has a clear cultural consensus

3. **Three-Tier Positional Context Evaluation**:
    - **Function**: To analyze the model's ability to utilize geographical and cultural cues
    - **Mechanism**: Three prompt settings—no location context, region-only (e.g., 'Gulf region'), and region + country (e.g., 'Gulf region - Saudi Arabia')
    - **Design Motivation**: To test whether the model can effectively retrieve corresponding cultural knowledge based on positional information

### Evaluation Setup

- Two modes: Sentence completion (probability-based judgment) and multiple-choice questions (MCQ, instruction following)
- Zero-shot evaluation of 31 models: 20 multilingual models + 10 Arabic-specific models + GPT-4o
- Comparison between Arabic and English prompts

## Key Experimental Results

### Main Results (MCQ, ℓ=R+C)

| Model | Size | MCQ Accuracy |
|------|--------|-----------|
| Human Performance | - | **100.0** |
| **GPT-4o** | - | **90.0** |
| Qwen2.5 Instruct | 72B | 80.0 |
| AceGPT-v2 Chat | 32B | 79.6 |
| Qwen2.5 Instruct| 32B | 76.5 |
| SILMA Instruct | 9B | 72.0 |
| Llama-3.3 Instruct | 70B | 71.2 |
| Gemma-2 Instruct | 27B | 64.2 |
| Jais Chat | 13B | 54.4 |
| Llama-3.1 Instruct | 8B | 49.1 |
| DeepSeek-R1-Distill-Llama | 70B | **34.5** |
| Random Baseline | - | 33.3 |

### MCQ vs. Completion Mode Comparison

| Model | Completion Accuracy | MCQ Accuracy | Gain |
|------|-----------|----------|------|
| Qwen2.5 Instruct 32B | 38.6 | 76.5 | **+37.9** |
| Llama-3.3 Instruct 70B | 41.1 | 71.2 | **+30.1** |
| Gemma-2 Instruct 27B | 39.8 | 64.2 | **+24.4** |

### Ablation Study

| Analysis Dimension | Key Findings |
|----------|---------|
| CS (country-specific) vs. ¬CS | Accuracy on shared cultural questions is higher |
| English vs. Arabic Prompt | English prompts **consistently outperform** Arabic (reflecting training data bias) |
| Positional Context | Inconsistent effects, benefiting some models while degrading others |
| Adding Cultural Facts to Prompts | Small models benefit partially; not a universal solution |

### Key Findings

- **GPT-4o Leads by a Wide Margin**: 90.0% accuracy, but still 10 percentage points behind humans.
- **Arabic-Specific Models Show No Advantage**: Jais Chat 13B scores only 54.4%, far below the general-purpose Qwen2.5 32B at 76.5%, indicating that "tailored for Arabic" $\neq$ "understanding Arabic culture".
- **Reasoning Models Fail Completely**: DeepSeek-R1-Distill-Llama 70B achieved only **34.5%** on MCQ, close to the random baseline of 33.3%.
- **46% of Instances are Country-Specific**: Proves that while the Arab world shares cultural foundations, nearly half of the cultural knowledge is specific to individual countries.
- **MCQ Far Outperforms Completion**: The gain can reach up to +37.9%, showing that instruction-tuned models are better at structured selection.

## Highlights & Insights

- The first large-scale Arabic cultural commonsense reasoning benchmark constructed from scratch by native speakers.
- Covers 13 countries and 54 subdomains, exhibiting extreme cultural diversity.
- Country-specificity labels (CS/¬CS) offer a unique perspective for analyzing cultural distribution.
- The near-random performance of reasoning models (DeepSeek-R1) on cultural tasks is an important and counterintuitive finding.
- English prompts outperforming Arabic prompts reflects the language imbalance in LLM training data.

## Limitations & Future Work

- Although covering 13 countries, it still fails to represent all 22 Arab countries.
- With only 2 annotators per country, individual biases might affect data quality and representativeness.
- Evaluates only zero-shot performance, without exploring few-shot or fine-tuning scenarios.
- The sentence completion format might not be the optimal way to evaluate cultural understanding (which in practice involves more open-ended reasoning).
- The performance variations of models across different themes (food, weddings, celebrations, etc.) were not analyzed.

## Related Work & Insights

- **English Commonsense Reasoning**: Winograd Schema, WinoGrande, PIQA — grounded in Western cultural assumptions, inapplicable to Arab culture.
- **Arabic NLP Benchmarks**: ArabicMMLU, LaraBench, DOLPHIN — knowledge/language tasks but not cultural reasoning.
- **Cultural AI Evaluation**: Koto et al. 2024b proposed a cross-cultural evaluation framework, which inspired the thematic classification design in this paper.
- **Insights**: Similar approaches can be extended to the evaluation of commonsense reasoning in other non-Western cultural regions (South Asia, Southeast Asia, Sub-Saharan Africa).

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Cultural Learning-Based Culture Adaptation of Language Models](cultural_learning-based_culture_adaptation_of_language_models.md)
- [\[ACL 2025\] Culture is Not Trivia: Sociocultural Theory for Cultural NLP](culture_is_not_trivia_sociocultural_theory_for_cultural_nlp.md)
- [\[ACL 2025\] GAMEBoT: Transparent Assessment of LLM Reasoning in Games](gamebot_transparent_assessment_of_llm_reasoning_in_games.md)
- [\[ACL 2025\] CER: Confidence Enhanced Reasoning in LLMs](cer_confidence_enhanced_reasoning.md)
- [\[ACL 2025\] Are Your LLMs Capable of Stable Reasoning?](are_your_llms_capable_of_stable_reasoning.md)

</div>

<!-- RELATED:END -->
