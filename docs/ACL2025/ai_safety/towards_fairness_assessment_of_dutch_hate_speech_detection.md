---
title: >-
  [Paper Note] Towards Fairness Assessment of Dutch Hate Speech Detection
description: >-
  [ACL 2025][AI Safety][Hate speech detection] This paper systematically evaluates the counterfactual fairness of Dutch hate speech detection models, proposes four counterfactual data generation methods (LLMdef, LLMlist, SLL, MGS), and validates the improvement of counterfactual data augmentation on model performance and fairness through fine-tuning on the BERTje model.
tags:
  - "ACL 2025"
  - "AI Safety"
  - "Hate speech detection"
  - "counterfactual fairness"
  - "Dutch NLP"
  - "bias mitigation"
  - "data augmentation"
date: 2026-05-08
content_hash: cc5cb89faa3bc1c5
---

# Towards Fairness Assessment of Dutch Hate Speech Detection

**Conference**: ACL 2025  
**arXiv**: [2506.12502](https://arxiv.org/abs/2506.12502)  
**Code**: [https://github.com/Juulba/Dutch-counterfactual-fairness](https://github.com/Juulba/Dutch-counterfactual-fairness)  
**Area**: AI Safety  
**Keywords**: Hate speech detection, counterfactual fairness, Dutch NLP, bias mitigation, data augmentation  


## TL;DR

This paper systematically evaluates the counterfactual fairness of Dutch hate speech detection models, proposes four counterfactual data generation methods (LLMdef, LLMlist, SLL, MGS), and validates the improvement of counterfactual data augmentation on model performance and fairness through fine-tuning on the BERTje model.

## Background & Motivation

There are three core issues in the field of hate speech detection:

**Language Bias**: The vast majority of research focuses on English, while studies on European languages such as Dutch are severely lacking.

**Lack of Fairness**: Existing work on Dutch hate speech detection primarily focuses on dataset construction and model development, with almost no exploration of model fairness issues.

**Counterfactual Evaluation Gap**: Research and relevant datasets on counterfactual fairness specifically for Dutch are completely missing.

The core of the fairness issue is that if sensitive attributes in a sentence (such as nationality, race, or religion) are replaced with other attributes, the model's prediction results should not change. For example, "All Moroccans are troublemakers" and "All Dutch are troublemakers" should receive the same probability of hate speech classification. However, actual models often predict a 98% probability of hate speech for the former and only 10% for the latter, exposing severe bias.

This paper fills the gap in the fairness evaluation of Dutch hate speech detection, presenting the first work to provide a counterfactual fairness dataset and systematic evaluation for Dutch.

## Method

### Overall Architecture

The overall pipeline consists of four stages:

1. **Social Group Term Construction**: Manually curate a list of 85 Dutch Social Group Terms (SGTs).
2. **Counterfactual Data Generation**: Generate counterfactual sentences using four methods.
3. **Model Fine-tuning**: Fine-tune the BERTje model using the counterfactual data.
4. **Evaluation**: Comprehensively evaluate the models across both performance and fairness dimensions.

### Dutch Social Group Term (SGT) Construction

Unlike simply translating English SGT lists, the authors considered the specificities of the Dutch language:

- **Social Context Adaptation**: For example, "Moroccan" is a prominent minority group in Dutch society but not in the US; "Sikhs" are less relevant in Dutch hate speech.
- **Grammatical Variation Handling**: Dutch nouns have corresponding adjective forms, and adjectives require inflections (e.g., "Nederlands" as an adjective and "Nederlander" as a noun).
- Ultimately, 85 SGTs were curated, covering nationality, skin color, immigration, gender, sexual orientation, religion, age, and ideology.

### Four Counterfactual Data Generation Methods

**1. LLMdef (Implicit LLM Generation)**

- Instructs the LLM to automatically identify and replace SGTs in sentences based on identity attribute categories (gender, race, religion, etc.).
- Replacement terms are autonomously determined by the LLM, making them more context-aware.
- Generated 15,175 counterfactual data points.

**2. LLMlist (Explicit LLM Generation)**

- Built upon LLMdef but provides a predefined list of SGTs.
- The LLM selects the replacement terms from the provided list, offering stronger controllability.
- Generated 21,562 data points.

**3. SLL (Sentence Log-Likelihood)**

- Computes sentence log-likelihood based on GPT-2: $f(x) = \log(P(x)) = \sum_{i=1}^{n} \log P(x_i | x_0, x_1, \ldots, x_{i-1})$
- Replaces the SGT in the original sentence with all other SGTs from the list, retaining only replacements where the likelihood is greater than or equal to that of the original sentence.
- Generated 49,104 counterfactual data points.

**4. MGS (Manual Group Substitution)**

- Constructs substitution dictionaries based on identity groups and grammatical functions.
- Substitutions occur only within the same category: for instance, "woman" is only replaced with other gender/noun terms (e.g., "transgender", "man").
- Ensures grammatical correctness, generating 20,393 counterfactual data points.

### Loss & Training

- Base Model: BERTje (a pre-trained Dutch BERT model)
- Training Data: IMSyPP Dutch hate speech dataset (25,720 training + 2,858 validation)
- Four-class Labels: appropriate, inappropriate, offensive, violent
- Fine-tuned BERTje individually using each of the four counterfactual datasets

### Fairness Evaluation Metrics

- **CTF (Counterfactual Token Fairness)**: $CTF(X, X_{cf}) = \sum_{x \in X} \sum_{x' \in X_{cf}} |g(x) - g(x')|$, where lower values represent greater fairness.
- **DPD (Demographic Parity Difference)**: Measures the difference in predicted positive rates across different identity groups.
- **EOD (Equalized Odds Difference)**: Measures the differences in true positive rates and false positive rates across different groups.

## Key Experimental Results

### Overall Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|-----|
| Baseline BERTje | 0.75 | 0.78 | 0.75 | 0.75 |
| BERTje + LLMdef | 0.79 | 0.61 | 0.62 | 0.61 |
| BERTje + LLMlist | 0.77 | 0.65 | 0.52 | 0.61 |
| BERTje + SLL | **0.79** | **0.79** | **0.79** | **0.79** |
| BERTje + MGS | **0.79** | **0.79** | **0.79** | **0.79** |

### Counterfactual Fairness Evaluation (CTF, lower is fairer)

| Model | Toxic | Non-Toxic | Average |
|------|-------|-----------|---------|
| Baseline | 0.11 | 0.36 | 0.24 |
| BERTje + LLMdef | 0.26 | 0.011 | 0.13 |
| BERTje + LLMlist | 0.32 | 0.001 | 0.16 |
| BERTje + SLL | 0.20 | 0.001 | **0.10** |
| BERTje + MGS | 0.28 | 0.003 | 0.14 |

### Group Fairness Evaluation (lower is fairer)

| Model | DPD | EOD |
|------|-----|-----|
| Baseline | 0.38 | 0.53 |
| BERTje + LLMdef | 0.09 | 0.18 |
| BERTje + LLMlist | 0.13 | 0.25 |
| BERTje + SLL | **0.06** | **0.11** |
| BERTje + MGS | 0.18 | 0.36 |

### Detection Performance per Category (F1)

| Category | Baseline | LLMdef | LLMlist | SLL | MGS |
|------|----------|--------|---------|-----|-----|
| Appropriate | 0.81 | 0.85 | 0.84 | 0.85 | 0.85 |
| Inappropriate | 0.37 | 0.38 | 0.36 | 0.34 | 0.35 |
| Offensive | 0.75 | 0.78 | 0.78 | 0.79 | 0.79 |
| Violent | 0.53 | 0.43 | 0.45 | 0.50 | 0.52 |

## Highlights & Insights

1. **SLL Method is Comprehensively Optimal**: Although the counterfactual sentences generated by SLL are of lower grammatical and semantic quality than those from LLM-based methods, they achieve the best results in both performance (F1=0.79) and fairness (CTF=0.10, DPD=0.06, EOD=0.11)—revealing a counter-intuitive relationship between data quality and model effectiveness.
2. **Fairness "Seesaw Effect"**: All counterfactual models drastically improve fairness on non-toxic templates (CTF drops from 0.36 to approximately 0.001) but degrade indeed on toxic templates (rising from 0.11 to 0.20–0.32), indicating that counterfactual data augmentation may introduce new biases.
3. **LLMs Generate High-Quality Sentences but Yield Poor Effects**: While LLMs generate more grammatically correct counterfactual sentences, the noise they introduce leads to worse classification performance and fairness compared to traditional methods.
4. **Specific Challenges of Dutch**: Grammatical behaviors such as adjective inflection and noun/adjective dual part-of-speech make simple substitution methods difficult to generate correct Dutch sentences, which is a non-issue in English-centric research.
5. **First Counterfactual Fairness Dataset for Dutch**: Fills a significant gap in the study of fairness in Dutch hate speech detection.

## Limitations & Future Work

1. **Class Imbalance**: There are relatively few samples in the "inappropriate" and "violent" categories, leading to persistently low detection performance for these classes (F1 < 0.40 and < 0.55). Oversampling or class weighting strategies can be explored in future work.
2. **Degradation of Counterfactual Fairness in Toxic Classes**: The CTF of counterfactual models on toxic templates unexpectedly increases; the root cause remains unexplored and might relate to the noise introduced by unrealistic counterfactuals.
3. **Single Model Evaluation**: Only the BERTje model was evaluated, with no coverage of multilingual models (e.g., mBERT, XLM-R) or larger-scale LLMs.
4. **Lack of Ablation Studies**: The impact of varying training data volumes on performance and fairness was not investigated, making it difficult to disentangle the contributions of "data volume increase" from "counterfactual data quality."
5. **Limitations of Template-Based Evaluation**: The template data utilized for fairness evaluation consists of only 34 short sentence templates, which limits the coverage of diverse scenarios.
6. **Scalability**: The proposed method can be extended to other under-resourced European languages (e.g., Danish, Swedish) to validate its cross-lingual generalization capability.

## Related Work & Insights

- **Davani et al. (2021)**: Pioneering work on counterfactual fairness in English hate speech detection, from which the SGT construction and SLL methods in this paper are directly borrowed.
- **Garg et al. (2019)**: Proposed the CTF metric to quantify counterfactual fairness.
- **Kusner et al. (2017)**: The theoretical foundation of counterfactual fairness, defining that "a decision is fair if it is the same in the actual world and a counterfactual world where the individual belonged to a different demographic group."
- **Caselli et al. (2021)**: DALC v1.0 Dutch abusive language corpus, providing a foundation for hate speech research in Dutch.
- The core insight of this paper: **The quality of data augmentation is not linearly and positively correlated with downstream task performance**. "Lower-quality" data generated by traditional methods might foster model fairness by introducing more diverse distributional shifts.

## Rating

- Novelty: ⭐⭐⭐ — The method itself is not original (porting English approaches to Dutch), but it fills the gap in Dutch fairness evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive comparison among the four methods, including performance and multi-dimensional fairness evaluations, although ablation studies are missing.
- Writing Quality: ⭐⭐⭐⭐ — The structure is clear, the case analysis is detailed, and qualitative and quantitative evaluations are well integrated.
- Value: ⭐⭐⭐ — Releases the first Dutch counterfactual dataset, but technical contributions are limited, mainly representing the application and comparison of existing methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Bridging Fairness and Explainability: Can Input-Based Explanations Promote Fairness in Hate Speech Detection?](../../ICLR2026/ai_safety/bridging_fairness_and_explainability_can_input-based_explanations_promote_fairne.md)
- [\[ACL 2025\] SpeechFake: A Large-Scale Multilingual Speech Deepfake Dataset Incorporating Cutting-Edge Generation Methods](speechfake_a_largescale_multilingual_speech_deepfake.md)
- [\[ACL 2025\] Gender Inclusivity Fairness Index (GIFI): A Multilevel Framework](gifi_gender_fairness.md)
- [\[ACL 2025\] FairI Tales: Evaluation of Fairness in Indian Contexts with a Focus on Bias and Stereotypes](fairi_tales_evaluation_of_fairness_in_indian_contexts_with_a_focus_on_bias_and_s.md)
- [\[ACL 2025\] Quantifying Misattribution Unfairness in Authorship Attribution](quantifying_misattribution_unfairness_in_authorship_attribution.md)

</div>

<!-- RELATED:END -->
