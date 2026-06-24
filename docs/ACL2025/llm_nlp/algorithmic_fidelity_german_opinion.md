---
title: >-
  [Paper Note] Algorithmic Fidelity of Large Language Models in Generating Synthetic German Public Opinions: A Case Study
description: >-
  [ACL 2025][LLM (Other)][algorithmic fidelity] Based on open-ended question data from the German Longitudinal Election Study (GLES), this study systematically evaluates the algorithmic fidelity of three open-source LLMs (Llama2, Gemma, Mixtral) in generating synthetic German public opinions using demographic persona prompting. The findings show that Llama2 performs best in sub-population representativeness (JS distance of 0.28), yet all models exhibit a left-leaning political…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "algorithmic fidelity"
  - "opinion simulation"
  - "persona prompting"
  - "German survey"
  - "Jensen-Shannon distance"
  - "political bias"
date: 2026-05-08
content_hash: fa3ef706e3c38b89
---

# Algorithmic Fidelity of Large Language Models in Generating Synthetic German Public Opinions: A Case Study

**Conference**: ACL 2025  
**arXiv**: [2412.13169](https://arxiv.org/abs/2412.13169)  
**Code**: [soda-lmu/llm-opinion-german](https://github.com/soda-lmu/llm-opinion-german)  
**Institution**: LMU Munich, Munich Center for Machine Learning, University of Maryland  
**Area**: LLM/NLP  
**Keywords**: algorithmic fidelity, opinion simulation, persona prompting, German survey, Jensen-Shannon distance, political bias

## TL;DR

Based on open-ended question data from the German Longitudinal Election Study (GLES), this study systematically evaluates the algorithmic fidelity of three open-source LLMs (Llama2, Gemma, Mixtral) in generating synthetic German public opinions using demographic persona prompting. The findings show that Llama2 performs best in sub-population representativeness (JS distance of 0.28), yet all models exhibit a left-leaning political bias and a reduction in within-group diversity.

## Background & Motivation

**Background**: In recent years, various attempts have been made to use LLMs for synthetic survey data generation, representing a "silicon sampling" approach where LLMs play personas with specific demographic backgrounds to simulate human survey responses. Argyle et al. (2023) introduced the concept of "algorithmic fidelity" to measure the ability of LLMs to replicate socioeconomic backgrounds and opinion differences among various human sub-populations, demonstrating a certain degree of feasibility on US election data.

**Limitations of Prior Work**: Existing studies have three notable limitations. First, the vast majority of research focuses on the English language and US contexts, leaving evaluations of non-English countries (such as the German multi-party system) extremely scarce. Second, existing experiments almost entirely use closed-ended multiple-choice questions (e.g., single-choice voting intentions), whereas open-ended free-text responses are closer to real surveys but harder to evaluate. Third, von der Heyde et al. (2025) investigated German voting behavior using GPT-3.5 and found a bias toward the Green Party and Left Party, but this was limited to a multiple-choice setting and could not reveal the fine-grained biases of LLMs in free-text generation.

**Key Challenge**: The diversity and semantic richness of open-ended responses far exceed those of closed-ended questionnaires, yet this is precisely the scenario where LLMs are most likely to generate stereotyped, low-diversity outputs. There is a tension between two goals: the model needs to understand the conditional distribution of demographic backgrounds while maintaining opinion diversity within groups.

**Goal**: This study aims to systematically evaluate the algorithmic fidelity of several open-source LLMs on German open-ended survey questions for the first time, answering three specific questions: (1) Which model has the best representativeness at the group level? (2) How does representativeness vary across different sub-populations and temporal waves? (3) Which demographic variables included in the prompt have the greatest impact on fidelity?

**Key Insight**: This work utilizes 21 waves of panel data from the GLES Panel (German Longitudinal Election Study), selecting the open-ended question "What is the most important problem currently facing Germany?" spanning 2019–2021 (including before and after COVID-19). Information-theoretic metrics (Jensen-Shannon distance, conditional entropy, mutual information, and Cramér's V) are used to comprehensively quantify distribution alignment and variable associations, alongside ablation experiments to decouple the contributions of individual variables.

**Core Idea**: The core idea is to have LLMs simulate German citizens responding to open-ended political questions using persona prompting, map the free text into 16 coded categories using a classifier, and then systematically evaluate distribution fidelity and sub-population bias via information-theoretic metrics.

## Method

### Overall Architecture

The entire pipeline consists of four steps: (1) extracting respondents' demographic features (age, gender, political party preference, region, education, occupational qualification) from the GLES dataset to construct German persona prompts; (2) generating text responses using three open-source LLMs (Llama2-13B, Gemma-7B, Mixtral-8x7B); (3) training a BERT classifier to map LLM outputs to 16 coarse-grained categories; (4) comparing the distribution alignment between synthetic and real survey data using metrics such as JS distance, conditional entropy, information gain, and Cramér's V.

### Key Designs

1. **Persona Prompt Construction and Multi-Variable Coding**:

    - **Function**: Translates demographic profiles of actual survey respondents into German prompts understandable by LLMs.
    - **Mechanism**: A German template is used, filling placeholders with 6 demographic variables (age, gender, political party preference, region, education, occupational qualification) to instruct the LLM to answer "What is the most important problem currently facing Germany?" in character. The prompt language is set to German instead of English to match the original GLES survey.
    - **Design Motivation**: Using German prompts ensures domain consistency and avoids translation bias. The six variables cover core socioeconomic dimensions, ensuring sufficient information without overly constraining the model.

2. **LLM Output Classification Pipeline (Annotation → BERT → Full Inference)**:

    - **Function**: Classifies open-ended texts generated by LLMs into 16 predefined categories, making distribution comparison possible.
    - **Mechanism**: First, 500 outputs are randomly sampled from each of the three LLMs for manual labeling (1,500 in total). Then, a German BERT classifier is fine-tuned, reaching a weighted F1-score of 0.93 on the test set, before being used for full-scale automatic classification. The 16-category coding scheme follows GESIS coding standards, merging 50+ original fine-grained categories into 16 coarse-grained ones.
    - **Design Motivation**: Open-ended responses cannot be directly compared in terms of distributions; they must first be standardized into the same category space. Training a classifier is more stable than using an LLM directly for classification, and it avoids the methodological issue of circularly using the evaluated object as the evaluation tool.

3. **Information-Theoretic Evaluation Metric System**:

    - **Function**: Quantifies the degree of distribution alignment between synthetic and real survey data from multiple dimensions.
    - **Mechanism**: JS distance measures overall distribution alignment (ranging from 0 to 1, lower is better). Conditional entropy measures the remaining uncertainty of responses given sub-populations. Information gain (mutual information) measures the predictive power of demographic variables on responses. Cramér's V tests whether the LLM preserves the association patterns between input variables and output categories.
    - **Design Motivation**: A single metric cannot comprehensively evaluate fidelity. JS distance measures group-level distribution alignment; conditional entropy and information gain reveal modeling quality at the sub-population level; Cramér's V tests whether the association patterns between variables are distorted.

## Key Experimental Results

### Main Results

**Experiment 1: Comparison of Three Models (Wave 12, Pre-COVID)**

| Metric | Gemma | Llama2 | Mixtral | Real Survey |
|------|-------|--------|---------|----------|
| COVID Regex Match Rate | **0.42** | 0.03 | 0.002 | 0 |
| JS Distance (↓) | 0.62 | **0.28** | 0.29 | - |
| Response Entropy | 2.26 | **2.90** | 2.56 | 2.93 |
| Non-German Response Rate | 0.02 | 0.06 | 0.03 | - |
| No Response Rate | 0 | 0 | 0.05 | 0.04 |

**Experiment 2: Longitudinal Analysis of Llama2 (Wave 12-21)**

| Wave | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | Mean |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|
| LLM Entropy | 2.90 | 0.58 | 1.67 | 1.31 | 2.12 | 2.20 | 2.27 | 2.46 | 2.46 | 2.49 | 2.04 |
| Survey Entropy | 2.93 | 2.02 | 2.24 | 2.31 | 2.53 | 2.82 | 2.75 | 2.85 | 2.92 | 2.19 | 2.55 |
| JS Distance | 0.29 | 0.29 | 0.24 | 0.22 | 0.20 | 0.23 | 0.23 | 0.22 | 0.24 | 0.30 | 0.24 |

### Ablation Study

**Experiment 3: Variable Ablation (Cramér's V Comparison)**

| Prompt Variable | Real Survey | LLM - Single Variable | LLM - All Variables |
|------------|----------|----------------|-------------|
| Age | 0.09 | 0.09 | 0.07 |
| Education | 0.06 | **0.25** | 0.05 |
| Gender | 0.08 | **0.20** | 0.16 |
| Party Preference | 0.16 | **0.35** | 0.17 |
| Region | 0.06 | **0.42** | 0.15 |
| Occupational Qualification | 0.08 | 0.12 | 0.07 |

### Key Findings

- **Severe Hallucination in Gemma**: 42% of responses involved COVID-19 (whereas data was collected in November 2019, prior to the COVID outbreak), leading to its exclusion from subsequent analyses. This indicates that temporal misalignment in LLM knowledge can cause severe distribution shifts.
- **Negative Correlation between Diversity and Representativeness**: The Pearson correlation between Survey Entropy and JS distance is $r=-0.35$; the more diverse the survey responses, the lower the representativeness of the LLM. Models are more accurate in scenarios with high consensus (e.g., during the COVID outbreak, where 92.4% of responses concentrated on health policy).
- **Systemic Left-leaning Bias**: The JS distance of Llama2's modeling for the Green Party and the Left Party is significantly lower than that for AfD (right-wing). This bias cannot be fully explained by differences in within-subpopulation diversity, pointing to systemic leanings in model training data and RLHF.
- **Single-Variable Overfocusing Effect**: When only a single variable is provided, the model over-relies on it (Cramér's V jumps from 0.16 to 0.35). When multiple variables are provided together, the correlation returns to a reasonable range.
- **Party Preference Variable has the Greatest Impact**: In the ablation analysis, adding only the party preference variable leads to the largest drop in JS distance, while excluding only the party preference variable results in the most severe performance degradation.

## Highlights & Insights

- **Innovative Evaluation Paradigm for Open-Ended Responses**: By establishing an "open-ended text -> manual annotation -> BERT classification -> distribution metrics" pipeline, this work addresses the difficulty of directly comparing distributions of free text. This workflow holds direct methodological value for other open-ended survey simulation studies.
- **Information Gain Analysis Reveals Stereotyping Mechanisms**: The conditional entropy of Mixtral drops sharply for specific subpopulations (especially the Green Party and AfD), meaning that the model generates highly homogeneous responses for particular groups. This method of quantifying the "degree of stereotyping" via information-theoretic metrics can be transferred to other demographic bias evaluation scenarios.
- **Variable Interaction Effects in Ablation Experiments**: It reveals a counterintuitive phenomenon—providing the model with fewer demographic variables actually leads to a stronger variable-response association. This indicates that the conditional generation of LLMs is not a simple Bayesian posterior sampling, but is strongly influenced by the prompt structure.

## Limitations & Future Work

- **Limited Model Coverage**: Only three open-source LLMs were tested, excluding closed-source models such as GPT-4 and Claude, which might exhibit different behaviors due to larger-scale training and more detailed RLHF.
- **Loss of Classification Granularity**: Coded representation into 16 coarse-grained categories inevitably loses subtle semantic nuances in free-text responses. Future work can incorporate continuous distance metrics in embedding space.
- **Zero-Shot Limitation**: Only zero-shot prompting was used, leaving the potential of few-shot exemplars or fine-tuning to improve fidelity unexplored.
- **Generalizability of a Single Question**: The evaluation is based on only one survey question ("most important problem"). The fidelity of responses might differ for other types of questions (e.g., values, policy preferences).

## Related Work & Insights

- **vs Argyle et al. (2023)**: They evaluated algorithmic fidelity using GPT-3's Silicon Sampling approach on US election data, but as restricted to English and closed-ended questions. This work extends the paradigm to a German open-ended scenario, revealing greater fidelity challenges.
- **vs von der Heyde et al. (2025)**: They observed left-wing bias in GPT-3.5's single-choice voting predictions but could not reveal the extent of stereotyping. This work provides a more granular quantification of bias through conditional entropy and information gain.
- **vs Santurkar et al. (2023)**: They found significant discrepancies between LLM opinion distributions and US survey responses. This study corroborates similar findings in a non-English context and provides evidence from variable ablation.

## Rating
- Novelty: ⭐⭐⭐⭐ The first study to systematically evaluate the algorithmic fidelity of LLMs on German open-ended surveys, bringing novelty to both the scenario and methodology.
- Experimental Thoroughness: ⭐⭐⭐⭐ The three experiments (model comparison + longitudinal analysis + ablation) are rigorously designed, supported by a complete system of information-theoretic metrics.
- Writing Quality: ⭐⭐⭐⭐ The data presentation is clear, with rich tables and figures, and the discussions and conclusions are logically coherent.
- Value: ⭐⭐⭐ Primarily analytical work, lacking proposed improvement methods, which limits its guidance for practical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Language Model Fine-Tuning on Scaled Survey Data for Predicting Distributions of Public Opinions](language_model_fine-tuning_on_scaled_survey_data_for_predicting_distributions_of.md)
- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)
- [\[ACL 2025\] FoodTaxo: Generating Food Taxonomies with Large Language Models](foodtaxo_generating_food_taxonomies_with_large_language_models.md)
- [\[ACL 2025\] Is It JUST Semantics? A Case Study of Discourse Particle Understanding in LLMs](is_it_just_semantics_a_case_study_of_discourse_particle_understanding_in_llms.md)
- [\[ACL 2025\] An Empirical Study of Large Language Models for Automated Review Generation](an_empirical_study_of_large_language_models_for_automated_review_generation.md)

</div>

<!-- RELATED:END -->
