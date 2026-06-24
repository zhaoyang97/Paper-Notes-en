---
title: >-
  [Paper Note] Building Better: Avoiding Pitfalls in Developing Language Resources when Data is Scarce
description: >-
  [ACL 2025][Multilingual & Machine Translation][low-resource languages] By surveying 81 low-resource language NLP researchers and annotators, this paper reveals quality issues (unnatural data, cultural misalignment) and ethical concerns (exploitation of annotators' labor, unfair authorship attribution) in low-resource language data construction, proposing six actionable recommendations for improvement.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "low-resource languages"
  - "data annotation ethics"
  - "participatory research"
  - "language resources"
  - "NLP practices"
date: 2026-05-08
content_hash: 687fe8aeaaed7a16
---

# Building Better: Avoiding Pitfalls in Developing Language Resources when Data is Scarce

**Conference**: ACL 2025  
**arXiv**: [2410.12691](https://arxiv.org/abs/2410.12691)  
**Code**: None  
**Area**: Other  
**Keywords**: low-resource languages, data annotation ethics, participatory research, language resources, NLP practices

## TL;DR

By surveying 81 low-resource language NLP researchers and annotators, this paper reveals quality issues (unnatural data, cultural misalignment) and ethical concerns (exploitation of annotators' labor, unfair authorship attribution) in low-resource language data construction, proposing six actionable recommendations for improvement.

## Background & Motivation

**Background**: The NLP community maintains a growing interest in low-resource languages, with participatory research frameworks (such as Masakhane) emerging. High-resource languages have established relatively comprehensive data quality standards and ethical guidelines (such as Datasheets for Datasets), but these standards have not been consistently extended to low-resource language contexts.

**Limitations of Prior Work**: Low-resource language NLP faces unique challenges: (1) data scarcity drives researchers to use any accessible data without verifying its quality; (2) it is difficult to find native speakers of specific languages on common annotation platforms (e.g., AMT, Prolific), leading researchers to turn to personal connections or online communities, which lack standards to protect annotator labor rights; (3) many low-resource languages are spoken rather than written, posing fundamental difficulties for text data collection.

**Key Challenge**: Speakers of the studied languages should be the primary beneficiaries of NLP tools, yet they are often marginalized in practice—relegated to unpaid annotation labor, while the resulting tools fail to meet real needs and the data fails to reflect cultural characteristics. Participatory research, which is intended to empower communities, can conversely become a new form of exploitation in the absence of standards.

**Goal**: (1) Empirically investigate practical issues and bad practices in low-resource language NLP; (2) propose actionable improvement recommendations based on first-hand data.

**Key Insight**: Directly distribute questionnaires to and gather personal experiences from "stakeholders" of low-resource language NLP, including researchers, annotators, and community members.

**Core Idea**: By directly listening to the voices of low-resource language practitioners, this study reveals systemic issues in data quality and labor ethics and proposes pathways for improvement.

## Method

### Overall Architecture

From June to October 2024, surveys were distributed to the *CL community via channels such as X, LinkedIn, Slack, and email, resulting in 81 valid responses covering over 70 low-resource languages. The questionnaire comprised four parts: (1) basic information (languages, project types); (2) motivation and shortcomings (reasons for doing low-resource NLP, identified issues); (3) labor recognition (whether compensated or credited); and (4) experiences with participatory research. Both quantitative statistical and qualitative (thematic analysis) analyses were conducted.

### Key Designs

1. **Multidimensional Questionnaire Design**:

    - Function: Comprehensively reveal practical issues in low-resource language NLP.
    - Mechanism: Addresses both ethical and technical aspects by asking not only technical questions but also questions like "Is your labor recognized?" and "Is participatory research fair?". It allows respondents to self-define "low-resource language" to avoid definitional disputes, covering over 70 languages across multiple regions, including Africa, South Asia, the Middle East, Eastern Europe, and Southeast Asia.
    - Design Motivation: Existing literature mostly discusses the topic from an external perspective, lacking first-hand data directly from participants.

2. **Dual-path Quantitative and Qualitative Analysis**:

    - Function: Use numbers to demonstrate scale and narratives to explain the essence.
    - Mechanism: Quantitative analysis: data scarcity at 78%, unrepresentative data at 58%, poor tool performance at 54%, and mismatch with user needs at 54%. Qualitative analysis delves into case studies: crudely categorizing all Arabic dialects together, replacing monetary compensation with company merchandise, and exploiting junior researchers for free labor under the pretext of "resume building."
    - Design Motivation: Statistical data reveals the prevalence of issues, while individual stories uncover their severity.

3. **Six Actionable Recommendations**:

    - Function: Directly derive improvement pathways from survey findings.
    - Mechanism: (1) Human-centeredness—engage language speakers in decision-making rather than just annotation; (2) Give fair recognition—annotators should receive compensation and authorship; (3) Use terminology with care—avoid classification from a colonial perspective; (4) Set realistic expectations—low-resource tools should not be expected to match high-resource performance; (5) Verify data sources—do not waive quality checks due to scarcity; (6) State researcher positionality—describe the researchers' relationship to the studied language.
    - Design Motivation: Each recommendation directly addresses specific problems uncovered in the survey.

### Loss & Training

This paper is a survey study and does not involve model training.

## Key Experimental Results

### Main Results

| Survey Dimension | Proportion |
|---------|------|
| Data scarcity as the primary limitation | 78% |
| Data is unrepresentative/unnatural | 58% |
| Poor performance of existing tools | 54% |
| Mismatch between tools and user needs | 54% |
| Low annotation quality | 25% |
| Data is impractical | 18% |
| Unfair recognition experienced at least once | >67% |

### Respondent Motivation Analysis

| Motivation | Proportion |
|------|------|
| Scientific interest/curiosity | 81% |
| Building language technologies | 72% |
| Obvious limitations in existing resources | 60% |
| Building technology for their own language | 60% |
| Contributing to LLM research | 59% |
| Building technology for as many languages as possible | 38% |

### Key Findings

- **Double-edged sword of participatory research**: 40% of respondents who spent from a day to over a month on annotation reported negative experiences—their labor was not properly compensated or recognized. Some junior researchers had their monthly salary replaced with "company merchandise."
- **Cultural misalignment is a systemic issue**: grouping all Arabic dialects into a single category, analyzing non-Western cultures using Western frameworks, and using religious texts to represent everyday language—leading to severe cultural misrepresentations.
- **Junior researchers are the primary victims**: they are told that "participating in the community is an honor and a resume builder," but they receive no authorship despite contributing substantial annotation labor.
- 60% of respondents work on low-resource NLP for their own language—language acts as a form of "symbolic capital" that deeply drives researchers.
- Relying on machine translation and LLMs to generate synthetic data is particularly harmful in low-resource scenarios due to the inability to effectively verify quality.
- The majority of respondents (>90%) chose to leave their contact information, reflecting their urgent desire for these issues to be addressed.

## Highlights & Insights

- **Highly persuasive first-hand survey data**: Unlike papers that discuss low-resource NLP ethics from an external viewpoint, this work directly gives participants a voice. Stories like "receiving company merchandise instead of a monthly salary" carry much more impact than abstract discussions of "annotator rights."
- **Linking ethics to data quality issues**: Unfair annotation practices (e.g., using non-native speakers for annotation, lack of pay leading to perfunctory work) directly damage data quality—the two are inseparable.
- **Highly actionable authorship recommendations**: The paper concretely lists contributions through which annotators can earn authorship (e.g., running language-specific ablation experiments, selecting culturally representative samples, writing specific sections), rather than offering vague suggestions.

## Limitations & Future Work

- The 81 responses may suffer from selection bias—those who actively participated in the survey might be individuals who are more sensitive to these issues.
- There is no tracking of the actual adoption of the proposed recommendations.
- The recommendations are primary oriented toward academia; their applicability to low-resource product development in industry requires further discussion.
- Specific technical solutions—how to build better tools with less data—are not addressed.
- The survey was distributed via English-language platforms, which may have missed low-resource language practitioners who are not active in the English NLP community.

## Related Work & Insights

- **vs Joshi et al. (2020)**: They statistically analyzed the research status of various languages in NLP from the perspective of language coverage, whereas this paper reveals practical-level issues from the perspective of participant experiences.
- **vs Bird & Yibarbuk (2024)**: They focused on participatory collaborative models between linguists and communities, whereas this paper has a broader scope (industry + online communities + academia).
- **vs Gebru et al. (2021) Datasheet**: Datasheets provided data documentation standards but did not specifically consider the unique needs for annotator protection in low-resource language contexts. This work can be seen as an ethical supplement tailored as a low-resource version of Datasheets.

## Rating

- Novelty: ⭐⭐⭐⭐ First large-scale collection of first-hand data from low-resource language NLP practitioners, uncovering issues such as exploitation in participatory research that are under-discussed in literature.
- Experimental Thoroughness: ⭐⭐⭐ The sample size of 81 responses is limited, but the qualitative analysis is comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, discoveries and recommendations map one-to-one, and the positionality statement serves as an excellent exemplar.
- Value: ⭐⭐⭐⭐ Holds direct practical guiding significance for the low-resource language NLP community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models](crosslingual_pitfalls.md)
- [\[ACL 2025\] Data Quality Issues in Multilingual Speech Datasets: The Need for Sociolinguistic Awareness and Proactive Language Planning](multilingual_speech_data_quality.md)
- [\[ACL 2025\] Less, but Better: Efficient Multilingual Expansion for LLMs via Layer-wise Mixture-of-Experts](less_but_better_efficient_multilingual_expansion.md)
- [\[ACL 2025\] Alleviating Distribution Shift in Synthetic Data for Machine Translation Quality Estimation](alleviating_distribution_shift_in_synthetic_data_for_machine_translation_quality.md)
- [\[ACL 2025\] LACA: Improving Cross-lingual Aspect-Based Sentiment Analysis with LLM Data Augmentation](laca_crosslingual_absa.md)

</div>

<!-- RELATED:END -->
