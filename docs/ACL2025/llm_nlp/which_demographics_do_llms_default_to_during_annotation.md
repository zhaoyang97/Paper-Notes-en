---
title: >-
  [Paper Note] Which Demographics Do LLMs Default to During Annotation?
description: >-
  [ACL2025][LLM (Other)][LLM annotation] By comparing the annotation behavior of LLMs under three prompt conditions—no demographic information (N), socio-demographic (SD), and placebo information (P)—this study reveals that in subjective annotation tasks (offensiveness/politeness), LLMs default to annotation patterns that align more closely with white, young, and highly educated cohorts. Furthermore, socio-demographic prompting indeed exerts a more systematic influence compared…
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "LLM annotation"
  - "demographic bias"
  - "socio-demographic prompting"
  - "perspectivism"
  - "placebo prompting"
  - "offensiveness"
  - "politeness"
date: 2026-05-08
content_hash: f71e44235fb8e902
---

# Which Demographics Do LLMs Default to During Annotation?

**Conference**: ACL2025  
**arXiv**: [2410.08820](https://arxiv.org/abs/2410.08820)  
**Code**: [uni-bamberg.de/nlproc/resources/llms-default-demographics](https://www.uni-bamberg.de/en/nlproc/resources/llms-default-demographics/)  
**Area**: LLM/NLP  
**Keywords**: LLM annotation, demographic bias, socio-demographic prompting, perspectivism, placebo prompting, offensiveness, politeness

## TL;DR

By comparing the annotation behavior of LLMs under three prompt conditions—no demographic information (N), socio-demographic (SD), and placebo information (P)—this study reveals that in subjective annotation tasks (offensiveness/politeness), LLMs default to annotation patterns that align more closely with white, young, and highly educated cohorts. Furthermore, socio-demographic prompting indeed exerts a more systematic influence compared to placebo prompting.

## Background & Motivation

**Background**: LLMs are increasingly deployed for automated data annotation, especially in zero-shot/few-shot scenarios. However, annotation is inherently subjective—annotators from different demographic backgrounds evaluate the offensiveness and politeness of the same text differently (e.g., older females might consider the word "bro" offensive, while teenage males find it normal).

**Limitations of Prior Work**: LLM annotations lack the diversity of human annotators. Previous studies (Beck et al. 2024, Mukherjee et al. 2024) have attempted to inject diversity via socio-demographic prompting, but **failed to identify consistent patterns**, and even argued that the effect of demographic information in prompts is negligible.

**Key Challenge**: LLMs inevitably possess a certain "default persona"—when not provided with socio-demographic details, their annotation behavior aligns closer to certain groups over others. This implicit bias can marginalize minority viewpoints, yet there is currently a **lack of systematic empirical research to quantify this default bias**.

**Goal**: (RQ1) Which demographic groups do LLMs default to mimicking? (RQ2) Is the impact of socio-demographic prompting more significant than placebo information? (RQ3) How do task attributes (offensiveness vs. politeness rating) affect the role of demographic information? (RQ4) Are different models consistent in their behaviors?

**Key Insight**: Utilizing the Popquorn dataset (designed specifically to study relationships between annotator metadata and annotation variations, containing detailed socio-demographic information) and introducing placebo prompting as a control group, this study systematically compares LLM annotation behaviors under three prompting conditions.

**Core Idea**: In the absence of socio-demographic details, LLMs implicitly "default" to perspectives closer to white and younger cohorts during annotation, while socio-demographic prompting indeed triggers more systematic behavioral shifts compared to placebo prompting.

## Method

### Overall Architecture

This work designs three categories of prompts (SD/P/N). Across two subjective rating tasks (offensiveness 1-5, politeness 1-5), two LLMs (GPT-4o, Claude 3.5 Sonnet) are used to perform annotation. The LLM annotations are then compared and analyzed against human annotations that contain detailed demographic metadata.

### Key Design 1: Three Prompt Types Control

**Function**: Design three categories of prompt templates: socio-demographic (SD), placebo (P), and non-informative (N).
**Design Motivation**: SD vs N uncovers the impact of demographic information; P vs N checks whether any arbitrary extra information can alter model behavior; SD vs P contrasts the unique effects of demographic information.
**Mechanism**:

- **SD prompt**: Injects actual annotator attributes such as gender, race, age, occupation, and education, e.g., "You are a person of gender [gender], race [race], age [age]..."
- **P prompt**: Injects unrelated attributes (height, zodiac sign, house number, hobbies, favorite color), e.g., "You are a person of height [height], Zodiac sign [zodiac sign]..."
- **N prompt**: Contains no demographic information, or uses generic descriptions such as "any gender, any race..."
- Two template variations are designed for each type to evaluate template stability.

### Key Design 2: Dataset Selection — Popquorn

**Function**: Conducts experiments using the Popquorn dataset.
**Design Motivation**: This dataset is specifically tailored to investigate the impact of annotator demographic attributes on annotations. It comprises 45,000 annotations from 1,484 annotators, complete with metadata such as gender, race, age, occupation, and education level.
**Mechanism**:

- Utilizes two subsets: offensiveness ratings (4,500 annotations / 1,500 instances) and politeness ratings (11,151 annotations / 3,717 instances).
- Uniformly samples 3 annotators per instance to prevent uneven annotation distributions from introducing confounding variables.
- Excludes annotators who declined to disclose demographic metadata.

### Key Design 3: Analysis Methods

**Function**: Uses two analytical methodologies to infer the "default persona" of LLMs.
**Design Motivation**: A single analytical dimension might miss key insights.
**Mechanism**:

- **Mixed-effects regression analysis** (Table 3): Takes the distance between LLM and human annotations as the dependent variable, and annotator demographics as independent variables, incorporating random intercepts (annotators and instances). This tests which groups' annotations deviate the most from the LLM's default output.
- **Mean distance comparison analysis** (Table 4): Calculates the mean distance $\Delta\mu$ between the SD prompt output and the N prompt output. A larger distance indicates that the demographic value deviates further from the model's default behavior.

### Models and Cost

- GPT-4o: \$54; Claude 3.5 Sonnet: \$60
- Each experiment is run only once due to cost constraints.
- Extremely low parsing failure rate (22 instances for GPT-4o, 3 for Claude).

## Key Experimental Results

### Main Results: Regression Analysis of LLM's Default Persona (Table 3)

| Demographic Attribute | Offensiveness (GPT-4o) | Offensiveness (Claude) | Politeness (GPT-4o) | Politeness (Claude) |
|---|---|---|---|---|
| **Age** (Per year increase) | **+0.01** | **+0.01** | 0.00 | 0.00 |
| **Race (Baseline: White)** | | | | |
| Black/African American | ***+0.22** | **+0.19** | **+0.14** | **+0.15** |
| Asian | +0.09 | +0.03 | -0.08 | 0.00 |
| Hispanic/Latino | -0.11 | -0.05 | +0.09 | +0.12 |
| **Education (Baseline: < High School)** | | | | |
| College Degree | +0.05 | -0.09 | *-0.43 | *-0.48 |
| Graduate Degree | +0.06 | -0.01 | *-0.36 | *-0.44 |
| **Gender (Baseline: Male)** | | | | |
| Female | 0.00 | -0.03 | -0.05 | -0.05 |
| Non-binary | -0.06 | -0.01 | -0.05 | -0.05 |

(Positive coefficient = LLM is further from this group = does not default to this group; * $p \le 0.05$, ** $p \le 0.01$, *** $p \le 0.001$)

### Ablation Study: Comparison of Socio-Demographic Prompting Effects (Summary of Table 4)

| Attribute | $\Delta\mu$ GPT-4o Offensiveness | $\Delta\mu$ Claude Offensiveness |
|---|---|---|
| Male | 0.18 | 0.17 |
| Female | 0.20 | 0.15 |
| Non-binary | **0.29** | 0.17 |
| White | 0.18 | 0.16 |
| Black/African American | 0.22 | 0.15 |
| 18-24 Years Old | 0.21 | 0.16 |
| >65 Years Old | 0.20 | 0.15 |

(Placebo prompt scores remain consistently stable across attribute values without systematic differences)

### Key Findings

1. **Race Bias is Most Pronounced**: The distance between LLM and Black/African American annotators is significantly larger than that for White annotators (+0.19 to 0.22 Likert points in the offensiveness task), suggesting that LLMs default closer to a white perspective.
2. **Age Effect**: In offensiveness tasks, LLMs display larger distances from older annotators ($p \le 0.01$), defaulting to younger perspectives.
3. **Education Effect is Only Significant in Politeness Tasks**: LLMs display larger distances from lower-education cohorts (up to 0.57 points), defaulting towards highly educated perspectives.
4. **No Significant Gender or Occupational Effects**: Models do not exhibit defined preferences along these dimensions.
5. **Socio-demographic Prompting Outperforms Placebo**: Socio-demographic prompting induces systematic predictive changes, whereas variations under placebo prompting lack directional patterns.
6. **Politeness is More Susceptible to Demographic Influences than Offensiveness**: The average prediction variance of GPT-4o in politeness tasks (0.25) is higher than that in offensiveness tasks (0.19).
7. **Claude-Specific Pattern**: With advancing age, Claude's predictive variation on politeness tasks increases from 0.16 to 0.26.

## Highlights & Insights

- **Placebo-Controlled Design**: Drawing from the placebo concept proposed by Mukherjee et al. (2410.08820) (e.g., zodiac signs, house numbers), this study cleverly disentangles the effects of socio-demographic prompting from those of generic "extra information."
- **Complementary Dual Analysis**: Regression analysis (evaluating the distance between LLMs and humans) and mean distance analysis (evaluating SD vs. N prompt differences) provide cross-validation from two perspectives, enhancing the credibility of the findings.
- **Contradicting Prior Work**: While Beck et al. (2024) and Mukherjee et al. (2024) reported inconsistent effects of demographic prompting, this study observes significant effects by employing a more suitable dataset (Popquorn), illustrating that dataset quality is crucial for deriving robust insights.
- **Cautious Discussion of Practical Impact**: Despite statistical significance, the effect sizes are relatively small relative to the 1-5 Likert scale (< 0.5 points), and the authors remain cautious about over-interpretation.

## Limitations & Future Work

1. **Limited to 2 Models and 1 Run**: Due to budget constraints (totaling \$114), each experiment was executed only once, limiting statistical power.
2. **Dataset Cultural Bias**: Popquorn's demographic variables are centered on US society (such as racial classifications) and may not generalize to other cultures.
3. **Imbalanced Demographic Subgroups**: For instance, non-binary gender matches only 124 offensiveness annotation samples, making it difficult to draw reliable conclusions.
4. **Sampling May Introduce Bias**: Downsampling to standardize at exactly 3 annotators per instance may skew the representation of certain demographic cohorts.
5. **Lack of Post-Hoc Statistical Testing**: The regression models were not subjected to post-hoc tests for homoscedasticity and collinearity.
6. **Small Effect Sizes**: Observed variances range between 0.1 to 0.5 Likert points; their significance in real-world applications remains to be verified.
7. **Limited to 2 Tasks**: Both offensiveness and politeness are affective/social judgment tasks; future work should extend the scope to other NLP tasks like Named Entity Recognition (NER) or general sentiment analysis.

## Related Work & Insights

### vs. Beck et al. (2024) — Socio-Demographic Prompting Effects

Beck et al. discovered that the impact of demographic information in prompts is far weaker than prompting framing techniques (e.g., wording details). In contrast, utilizing the Popquorn dataset tailored specifically for perspectivism research, this work observes significant socio-demographic effects for race and age. The key differentiator lies in dataset selection—Popquorn’s controlled design makes it highly suitable for detecting subtle demographic variances.

### vs. Mukherjee et al. (2024) — Placebo Prompting

Mukherjee et al. proposed the concept of placebo prompting, showing that most LLMs demonstrate substantial response variability, thereby questioning the reliability of socio-demographic prompting. Adopting and expanding upon this placebo paradigm, this work reveals that socio-demographic prompting indeed prompts more systematic shifts than placebo runs (where placebo scores remain stable across attribute values, while SD prompts show directional differentiation). This partially alleviates the concerns raised by Mukherjee et al.

### vs. Sun et al. (2025) — LLM Demographic Bias

Sun et al. concluded that LLMs tend to align more closely with the perceptions of white participants. This study validates this finding (demonstrating that white annotators are closest to the default LLM outputs) and extends it to broader socio-demographic dimensions (age, education, occupation), providing a more comprehensive profile of demographic bias.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The dual-control paradigm employing both placebo and demographic configurations is ingenious; Popquorn is utilized for LLM analysis for the first time.
- **Experimental Thoroughness**: ⭐⭐⭐ — Small-scale, evaluating only two models, two tasks, and a single run; however, the multi-faceted, complementary analytical methods yield robust findings.
- **Writing Quality**: ⭐⭐⭐⭐ — Structured logically around Research Questions (RQs); covers related work comprehensively and addresses limitations with candor.
- **Value**: ⭐⭐⭐⭐ — Offers practical warnings regarding LLM annotation fairness; the placebo control methodology is highly recommended for future studies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Which of These Best Describes Multiple Choice Evaluation with LLMs?](multiple_choice_eval.md)
- [\[ACL 2025\] Do LLMs Give Psychometrically Plausible Responses in Educational Assessments?](do_llms_give_psychometrically_plausible_responses_in_educational_assessments.md)
- [\[ACL 2025\] Do Language Models Mirror Human Confidence? Exploring Psychological Insights to Address Overconfidence in LLMs](do_language_models_mirror_human_confidence_exploring_psychological_insights_to_a.md)
- [\[ACL 2025\] Beyond Demographics: Fine-tuning Large Language Models to Predict Individuals' Subjective Text Perceptions](beyond_demographics_fine-tuning_large_language_models_to_predict_individuals_sub.md)
- [\[ACL 2025\] Do Language Models Understand Honorific Systems in Javanese?](do_language_models_understand_honorific_systems_in_javanese.md)

</div>

<!-- RELATED:END -->
