---
title: >-
  [Paper Note] Leveraging Large Language Models to Measure Gender Representation Bias in Gendered Language Corpora
description: >-
  [LLM (Other)] This paper proposes leveraging the contextual understanding capabilities of LLMs to detect and quantify gender representation bias in the training corpora of grammatically gendered languages (e.g., Spanish and Valencian). A severe male-dominated imbalance is identified, and continuous pre-training using reverse-biased data is verified to effectively mitigate bias in model outputs.
tags:
  - "LLM (Other)"
date: 2026-05-08
content_hash: 104a43084ceacf74
---

# Leveraging Large Language Models to Measure Gender Representation Bias in Gendered Language Corpora

| Information | Content |
|------|------|
| Conference | ACL 2025 |
| arXiv | [2406.13677](https://arxiv.org/abs/2406.13677) |
| Code | [GitHub](https://github.com/ellisalicante/grb-corpora) |
| Area | NLP / Bias Detection / Multilingualism |
| Keywords | Gender Representation Bias, Grammatically Gendered Languages, LLM Bias Detection, Corpus Analysis, Bias Mitigation |

## TL;DR

This paper proposes leveraging the contextual understanding capabilities of LLMs to detect and quantify gender representation bias in the training corpora of grammatically gendered languages (e.g., Spanish and Valencian). A severe male-dominated imbalance is identified, and continuous pre-training using reverse-biased data is verified to effectively mitigate bias in model outputs.

## Background & Motivation

- **Problem Definition**: Gender representation bias refers to the unequal frequency of reference to individuals of different genders in text. This upstream bias in training data is the root cause of bias propagation and amplification in downstream models.
- **Limitations of Prior Work**: 
  1. Existing studies primarily focus on **stereotyping bias** (associating specific roles with genders) rather than **representation bias** (frequency inequality).
  2. Current methods (such as gender polarity) are designed specifically for English, relying on string matching of predefined gendered word lists. These approaches cannot handle **grammatically gendered languages**, where all nouns have grammatical gender (e.g., Spanish "el coche" is the masculine noun "car," but is a non-human reference).
- **Importance**: Approximately 38% of the global population speaks grammatically gendered languages. In these languages, the masculine plural form typically defaults to representing mixed-gender groups (e.g., Spanish "los profesores" refers to both male teachers and teachers in general), which itself constitutes an implicit gender representation bias.
- **Core Motivation**: There is a need for a method capable of distinguishing "nouns/pronouns referring to humans" from "non-human nouns" and correctly classifying their grammatical gender—which extends beyond simple vocabulary list matching and requires semantic understanding.

## Method

### Overall Architecture

A three-step pipeline leveraging the contextual understanding capabilities of LLMs:

1. **Identification**: Identify all nouns and pronouns in a given text.
2. **Classification**: Determine whether each noun/pronoun refers to a human (P) or non-human (N).
3. **Gender Determination**: Determine the grammatical gender of each word—masculine (M) or feminine (F).

Finally, the ratio $L_{P,M} : L_{P,F}$ is calculated to quantify gender representation bias.

### Key Designs

1. **LLM-based Method**: Through meticulously designed prompts and few-shot examples, the LLM is enabled to complete word identification, human-reference determination, and grammatical gender classification in a single query. The text is processed sentence-by-sentence to fully utilize the LLM's contextual semantic understanding.
2. **Exclusion of Adjectives**: Since the gender marking of adjectives typically depends on their associated nouns and does not independently convey human-reference information, excluding them reduces complexity.
3. **Validation of Bias Propagation via Continuous Pre-training**: Three synthetic datasets of 5,000 sentences each (male-biased, female-biased, and balanced) are constructed to perform QLoRA continuous pre-training (<20 steps) on three open-source LLMs, validating how training data bias propagates to model outputs.

### Loss & Training

Continuous pre-training utilizes the standard language modeling loss (next token prediction) combined with QLoRA for parameter-efficient training. The core objective is not to train a new model, but to validate the bias propagation hypothesis.

## Key Experimental Results

### Main Results: Spanish-English Corpus Bias

| Dataset | English GM:GF | Spanish L_{P,M}:L_{P,F} |
|--------|-----------|------------------------|
| Europarl | 1.39:1 ~ 1.46:1 | **3.94:1 ~ 3.98:1** |
| CCAligned | 1.07:1 | **4.03:1 ~ 4.54:1** |
| Global Voices | 1.43:1 | **4.39:1 ~ 4.48:1** |
| WMT-News | 3.08:1 ~ 3.44:1 | **5.22:1 ~ 6.04:1** |

### Valencian Corpus Bias

| Dataset | L_{P,M}:L_{P,F} |
|--------|-----------------|
| BOUA | 2.21:1 ~ 2.88:1 |
| DOGV+DOGCV | 2.41:1 ~ 2.72:1 |
| DSCV+DSCCV | 2.03:1 ~ 2.38:1 |

### Ablation Study: Bias Propagation Experiment

| Training Data | Model Output Male-to-Female Ratio (Example: llama3.1-8B Valencian) |
|---------|------|
| Base model (no continuous pre-training) | 3.21:1 |
| Male-biased training | 6.63:1 ↑ |
| Balanced training | Close to 1:1 |
| **Female-biased training** | **≈1:1** ✓ |

### Key Findings

1. **Spanish Bias is Significantly Larger than English**: In the same parallel corpus, the masculine representation bias in Spanish is 3 to 4 times higher than in English, which relates to grammatical gender conventions.
2. **Valencian Bias is Lower**: 2:1 to 3:1, potentially due to more formal conventions of inclusive language in official administrative documents.
3. **Bias is Propagated**: Training on male-biased data significantly increases the proportion of masculine references in model outputs.
4. **Reverse Bias Mitigates Effectively**: Fine-tuning with as few as 5,000 female-biased sentences via continuous pre-training reduces the model output bias from 3:1 to nearly 1:1.
5. **Method Validation**: GPT-4-Turbo achieves F-scores of 90.24% and 84.43% on the Spanish and Valencian validation sets, respectively, proving the stability and reliability of the proposed method.

## Highlights & Insights

- Fills the gap in quantifying gender representation bias in training corpora for grammatically gendered languages, utilizing a simple yet effective methodology.
- Distinguishing between "nouns referring to humans" and "all nouns" is a critical design choice—preventing non-human references like "table (feminine)" and "car (masculine)" from being erroneously counted as bias.
- The discovery of the reverse-bias mitigation strategy is highly practical: it does not require vast amounts of balanced data, as a mere 5,000 sentences of counter-biased data can significantly correct the bias.
- The method is generalizable to other grammatically gendered languages (such as French, German, and Czech), providing a valuable utility for fairness research in multilingual NLP.

## Limitations & Future Work

- Only validated on Spanish and Valencian, without testing on other grammatically gendered languages (e.g., French, German).
- The method relies on high-end LLMs (GPT-4-Turbo), leading to higher computational costs.
- Only 1,000 sentences were sampled per subset; while statistically sound, this might miss the long-tail distribution.
- Continuous pre-training experiments use synthetic data, which may differ from the real-world distribution.
- Only binary genders (masculine/feminine) are considered, without covering non-binary gender expressions.

## Related Work & Insights

- **Gender polarity** (Dhamala et al., 2021): An English gender polarity method based on predefined vocabulary lists, inapplicable to grammatically gendered languages.
- **BOLD** (Dhamala et al., 2021): A benchmark for social bias, with a primary focus on stereotyping bias.
- **Biesialska et al. (2024)**: Reports the correlation between stereotyping bias and representation bias.
- Implications for multilingual model fairness: highlights the need to design distinct bias-detection methodologies based on different linguistic typologies.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Practical Value | ⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Classifying Unreliable Narrators with Large Language Models](classifying_unreliable_narrators.md)
- [\[ACL 2025\] KazMMLU: Evaluating Language Models on Kazakh, Russian, and Regional Knowledge of Kazakhstan](kazmmlu_evaluating_language_models_on_kazakh_russian_and_regional_knowledge_of_k.md)
- [\[ACL 2025\] Culture is Not Trivia: Sociocultural Theory for Cultural NLP](culture_is_not_trivia_sociocultural_theory_for_cultural_nlp.md)
- [\[ACL 2025\] PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](plangenllms_planning_survey.md)
- [\[ACL 2025\] Explicit and Implicit Data Augmentation for Social Event Detection](explicit_and_implicit_data_augmentation_for_social_event_detection.md)

</div>

<!-- RELATED:END -->
