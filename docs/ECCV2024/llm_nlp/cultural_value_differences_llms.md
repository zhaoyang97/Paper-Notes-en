---
title: >-
  [Paper Note] Cultural Value Differences of LLMs: Prompt, Language, and Model Size
description: >-
  [ECCV 2024][LLM (Other)][Cultural Values] This paper systematically investigates the behavioral patterns of LLMs in expressing cultural values utilizing the Hofstede cultural dimensions questionnaire. It finds that prompt language (Chinese vs. English) and model size have a far greater impact on cultural value disparities than differences in model architecture and question order.
tags:
  - "ECCV 2024"
  - "LLM (Other)"
  - "Cultural Values"
  - "Large Language Models"
  - "Hofstede"
  - "Multilingual Bias"
  - "Model Behavioral Analysis"
date: 2026-05-08
content_hash: 79f67f180fda4309
---

# Cultural Value Differences of LLMs: Prompt, Language, and Model Size

**Conference**: ECCV 2024  
**arXiv**: [2407.16891](https://arxiv.org/abs/2407.16891)  
**Code**: None  
**Area**: LLM / NLP / AI Safety  
**Keywords**: Cultural Values, Large Language Models, Hofstede, Multilingual Bias, Model Behavioral Analysis

## TL;DR

This paper systematically investigates the behavioral patterns of LLMs in expressing cultural values utilizing the Hofstede cultural dimensions questionnaire. It finds that prompt language (Chinese vs. English) and model size have a far greater impact on cultural value disparities than differences in model architecture and question order.

## Background & Motivation

**Background**: While LLMs generate human-like text, they also internalize cultural biases and values present in their training corpora. Prior studies have employed human social science tools, such as political compass tests and the World Values Survey, to evaluate the value orientations of LLMs.

**Limitations of Prior Work**: (1) Most studies evaluate models solely in English, ignoring the impact of language on value expression (the Sapir-Whorf hypothesis); (2) there is a lack of systematic research on the impact of prompt variations, such as option ordering and identity simulation; (3) cultural value differences between models of different scales within the same family remain under-explored.

**Key Challenge**: Is the cultural value expression of LLMs an inherent characteristic determined by training data, or a superficial phenomenon easily manipulated by prompt engineering?

**Goal**: To systematically identify the key factors—prompt variations, language, and model size—that influence the cultural value expression of LLMs.

**Key Insight**: Utilizing the Hofstede VSM2013 questionnaire as a standardized tool to conduct comprehensive experiments across 6 LLMs with 54 simulated identities $\times$ multiple languages $\times$ shuffled option orders.

**Core Idea**: Through controlled variable experiments, this study reveals that language and model size are the primary drivers of cultural value disparities in LLMs, rather than contextual identity information.

## Method

### Overall Architecture

The VSM2013 questionnaire is used (consisting of 24 questions mapping to 6 cultural dimensions: Power Distance Index PDI, Individualism IDV, Uncertainty Avoidance Index UAI, Masculinity MAS, Long-Term Orientation LTO, and Indulgence versus Restraint IVR). Each experimental set is defined by a triplet of LLM $\times$ Language $\times$ Option Order. Within each set, 54 simulated identities (9 nationalities $\times$ 2 genders $\times$ 3 age groups) are tested, with each question repeated 10 times.

### Key Designs

1. **Multidimensional Experimental Design**:

    - Function: Comprehensive control of variables to identify the independent impacts of each factor.
    - Mechanism: Fixed model and language while varying option order (RQ1); fixed model while varying language (RQ2); fixed language while varying model (RQ3). Six models (Llama2-7B/13B/70B, Qwen-14B/72B, Mixtral-8x7B), two languages (Chinese and English), and two option orders are examined.
    - Design Motivation: Adopting standardized control variable methodology from social sciences to ensure causal interpretability of the conclusions.

2. **Three-Tier Evaluation Metric System**:

    - Function: To quantify cultural value disparities at different granularities.
    - Mechanism: (1) Pearson correlation coefficients of raw VSM scores measure within-set consistency; (2) standard deviation $\sigma_m(v_i)$ measures cross-country cultural variance; (3) Model Cultural Disparity (MCD) = $D_m/D_h$ is proposed to normalize cross-country variations in model performance against human data. Inter-set comparisons utilize Davies-Bouldin Index (DBI), Silhouette Score, and a newly proposed $SS_h$ (human-referenced Silhouette Score).
    - Design Motivation: A single metric cannot fully capture the multi-level characteristics of cultural values, necessitating both within-set and inter-set perspectives.

3. **Human-Referenced Silhouette Index $SS_h$**:

    - Function: To measure the "absolute scale" of inter-set differences in models, using human cultural disparities as a baseline.
    - Mechanism: The denominator of the standard Silhouette Score is modified by replacing the within-set distance $a(n_i)$ with the average human cross-country distance $a_h(n_i)$. Consequently, $SS_h > 1$ indicates that the inter-set disparity of the model exceeds the disparity between human nations.
    - Design Motivation: Standard clustering evaluation metrics only measure relative separation, whereas $SS_h$ provides an absolute scale anchored on humans.

### Loss & Training

This is a pure inference evaluation study with no training involved. All models are evaluated using default parameters, with response generation repeated 10 times per question to compute the average.

## Key Experimental Results

### Main Results

| Factor | Impact Level | Evidence |
|------|---------|------|
| Prompt Language | **Largest** | The $SS_h$ between Chinese and English for the same model is generally $>1$, far exceeding the impact of option shuffling. |
| Model Size | **Significant** | The cultural disparity of Llama2 7B vs. 70B is greater than that of Llama2 vs. Mixtral. |
| Option Shuffling | Moderate | Mixtral $SS_h=0.680$ is the most sensitive, while Llama2-13B $SS_h=0.228$ is the most stable. |
| Simulated Identity | **Smallest** | The MCD across the 54 identities is much less than 1; models do not alter their values based on simulated nationalities. |

### Ablation Study

| Model | English w/o shuffle vs. w/ shuffle DBI↓ | Chinese vs. English $SS_h$↑ |
|------|-------------------------------------|---------------------|
| Llama2-7B | 1.837 | 1.52 |
| Llama2-70B | 0.658 | 0.87 |
| Qwen-14B | 0.981 | 1.33 |
| Mixtral-8x7B | 0.542 | 0.68 |

### Key Findings

- **Language is the most critical factor**: When prompted in Chinese versus English, the same model exhibits starkly different cultural value tendencies in dimensions like PDI and IDV, with variations that occasionally exceed those found between different human countries.
- **Model Size > Model Architecture**: The cultural value disparity between Llama2-7B and Llama2-70B is larger than the disparity between Llama2 and Mixtral.
- **Models are largely unaffected by identity simulation**: Whether simulating a Japanese, American, or Chinese persona, the same model yields nearly identical responses under the same language (MCD << 1).
- **Option position bias exists but is manageable**: Large-scale models (70B+) demonstrate higher robustness against option shuffling.

## Highlights & Insights

- **Systematic Experimental Design**: A large-scale controlled experiment involving 54 simulated identities $\times$ 10 repetitions $\times$ multiple languages $\times$ multiple models, totaling 12960 $\times$ N responses, ensuring comprehensive coverage.
- **Design of the $SS_h$ Metric**: Normalizing against human cultural differences establishes an absolute scale for inter-set comparisons. This methodology is transferable to other LLM behavioral evaluation scenarios.
- **The "Language Determines Culture" Finding**: Consistent with the Sapir-Whorf hypothesis, this provides key insights for the deployment and safety auditing of multilingual LLMs—highlighting that model value alignments cannot be assessed solely in English.

## Limitations & Future Work

- Only six models and two languages are tested; evaluations on more languages (e.g., Arabic, Japanese) and model families are still lacking.
- The psychological validity of the VSM questionnaire itself has faced criticism, and it may not fully translate to LLMs.
- The causal relationship between training data composition and resulting cultural values is not analyzed.
- Directly assigning neutral values to health-related questions (Q15, Q18) is overly simplistic.
- Future work can investigate how RLHF/DPO alignment modifies the expression of cultural values in models.

## Related Work & Insights

- **vs. Arora et al.**: Explores the cultural values of multilingual MLMs using VSM + WVS but relies solely on masked models without controlling for model scale; this work extends the evaluation to generative LLMs and adds the dimension of scale.
- **vs. Kovač et al.**: Evaluates LLM "personality" using psychological questionnaires and finds high context dependency; this paper further demonstrates that language has a more profound impact than context.
- **vs. Feng et al.**: Evaluates bias using political compass tests restricted to single English models; this study is more comprehensive across multiple dimensions and models.

## Rating

- Novelty: ⭐⭐⭐ The experimental design is creative, but the research framework (evaluating LLMs using human questionnaires) is not entirely brand new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale systematic experiments with strict control variables.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, but excessive mathematical notation increases cognitive load.
- Value: ⭐⭐⭐⭐ Possesses practical utility for multilingual LLM deployment and safety evaluations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Convert Language Model into a Value-based Strategic Planner](../../ACL2025/llm_nlp/convert_language_model_into_a_value-based_strategic_planner.md)
- [\[ACL 2025\] Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](../../ACL2025/llm_nlp/unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)
- [\[ECCV 2024\] Propose, Assess, Search: Harnessing LLMs for Goal-Oriented Planning in Instructional Videos](propose_assess_search_harnessing_llms_for_goal-oriented_planning_in_instructiona.md)
- [\[ECCV 2024\] APL: Anchor-based Prompt Learning for One-stage Weakly Supervised Referring Expression Comprehension](apl_anchor-based_prompt_learning_for_one-stage_weakly_supervised_referring_expre.md)
- [\[ACL 2025\] Cultural Learning-Based Culture Adaptation of Language Models](../../ACL2025/llm_nlp/cultural_learning-based_culture_adaptation_of_language_models.md)

</div>

<!-- RELATED:END -->
