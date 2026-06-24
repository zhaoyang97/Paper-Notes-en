---
title: >-
  [Paper Note] The AI Gap: How Socioeconomic Status Affects Language Technology Interactions
description: >-
  [ACL 2025][LLM (Other)][Socioeconomic status] Through a large-scale survey of 1,000 users of different socioeconomic status (SES) and an analysis of 6,482 real LLM prompts, this study reveals significant, systematic differences between high- and low-SES groups in terms of language technology usage frequency, interaction styles, and topic choices, calling for the development of more inclusive NLP technologies to narrow the AI gap.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Socioeconomic status"
  - "language technology"
  - "digital divide"
  - "LLM interaction"
  - "user survey"
date: 2026-05-08
content_hash: da55b427d650eda2
---

# The AI Gap: How Socioeconomic Status Affects Language Technology Interactions

**Conference**: ACL 2025  
**arXiv**: [2505.12158](https://arxiv.org/abs/2505.12158)  
**Code**: [https://huggingface.co/datasets/MilaNLProc/survey-language-technologies](https://huggingface.co/datasets/MilaNLProc/survey-language-technologies)  
**Area**: Other  
**Keywords**: Socioeconomic status, language technology, digital divide, LLM interaction, user survey

## TL;DR

Through a large-scale survey of 1,000 users of different socioeconomic status (SES) and an analysis of 6,482 real LLM prompts, this study reveals significant, systematic differences between high- and low-SES groups in terms of language technology usage frequency, interaction styles, and topic choices, calling for the development of more inclusive NLP technologies to narrow the AI gap.

## Background & Motivation

**Background**: With the widespread popularization of large language models like ChatGPT, AI technology is profoundly reshaping how people interact with technology. However, the adoption rates and usage patterns of AI tools vary significantly across different social groups, a disparity already noted by organizations such as UNESCO and The Economist.

**Limitations of Prior Work**: Previous studies on the relationship between SES and language technology relied on proxy metrics (e.g., educational level, income) and synthetic data, lacking direct user survey data. Existing large-scale prompt datasets (e.g., ShareGPT, LMSYS-Chat-1M), despite their massive size, do not collect socioeconomic background information of users, making it impossible to analyze the impact of SES on LLM usage.

**Key Challenge**: While language technologies appear accessible to everyone, according to the Technology Acceptance Model (TAM), technology adoption is influenced by perceived usefulness and ease of use. Disparities in digital literacy, device access, and cultural capital among different SES groups may further widen the "digital divide" in the AI era—referred to as the AI Gap.

**Goal**: (1) How does SES influence the adoption rates and use cases of language technologies? (2) What are the differences in linguistic characteristics when different SES groups interact with LLMs? (3) How do topics discussed and perceptions of AI systems differ across groups?

**Key Insight**: The authors recruited 1,000 participants from the US and UK directly via the Prolific platform to collect their socioeconomic backgrounds and real interaction prompts with LLMs, combining quantitative statistical analysis and qualitative clustering to address the aforementioned questions.

**Core Idea**: This study is the first to directly investigate the impact of SES on language technology usage and LLM interaction styles through a large-scale user survey, revealing that SES differences are systematically reflected in usage frequency, linguistic style, and topic selection.

## Method

### Overall Architecture

The methodological framework of this paper is a large-scale user survey study including three core parts: (1) sociodemographic information collection (17 questions, including self-reported SES via the MacArthur scale), (2) language technology usage habits survey (frequency, task types, usage scenarios), and (3) LLM prompt collection (participants were requested to provide their 10 most recent interaction records with AI chatbots). The survey was conducted in two phases on the Prolific platform, with 501 participants in the first phase, and a targeted addition of 380 low- and high-SES participants in the second phase.

### Key Designs

1. **SES Measurement and Grouping**:
    - **Function**: Accurately measure the socioeconomic status of participants
    - **Mechanism**: The MacArthur scale (1-10 points) is utilized to let participants self-rate their socioeconomic status, which is then mapped to the Western class system: 1-3 as low, 4-7 as middle, and 8-10 as high. Objective indicators such as education, parental occupation, and housing are also collected for cross-validation.
    - **Design Motivation**: Self-reported SES reflects subjective psychological perceptions better than objective indicators, and research shows that subjective class perception has a significant impact on behavior.

2. **Linguistic Analysis of Prompts**:
    - **Function**: Quantify style differences in interaction across different SES groups
    - **Mechanism**: 6,482 real prompts are analyzed across multiple dimensions: length (word count), concreteness (using Brysbaert et al.'s 40K word concreteness ratings), and anthropomorphism (usage rates of polite words, greetings, professional terminology vs. metaphorical language), alongside training a Bag-of-Words (BoW) classifier to verify the predictability of prompts across different groups.
    - **Design Motivation**: Bernstein's linguistic coding theory predicts that high-SES groups use more abstract language; this classic hypothesis is verified here within LLM interaction scenarios.

3. **Topic Clustering and Qualitative Analysis**:
    - **Function**: Discover topical differences that concern different SES groups
    - **Mechanism**: Prompts are encoded using SentenceTransformer and M3-Embedding, clustered via UMAP+HDBSCAN, and then assigned descriptive labels using GPT-4o, followed by manual evaluation of cluster quality. Different "framings" of unique and shared topics among the three SES groups are compared.
    - **Design Motivation**: The same topics (e.g., finance, job-seeking, food) present drastically different demand framings across different SES groups, and this discrepancy reveals deeper social inequalities.

### Loss & Training

This paper is a survey study and does not involve model training. Statistical testing uses Chi-square tests of independence and bootstrap resampling significance tests, and the classifier employs Bag-of-Words (BoW) paired with logistic regression.

## Key Experimental Results

### Main Results

| Analysis Dimension | Metric | Low SES | Mid SES | High SES | Statistical Significance |
|----------|------|-------|-------|-------|------------|
| Average Prompt Length | Word Count | 27.0 | 22.3 | 18.4 | p < 0.05 |
| Linguistic Concreteness | Concreteness score (1-5) | 2.66 | 2.63 | 2.57 | p < 0.05 |
| AI Chatbot Usage Frequency | Daily use ratio | Decreasing from low to high | Moderate | Increasing from low to high | χ²=67.79, p<0.001 |
| Device Access | Daily use of multiple devices | Less | Moderate | More | χ²=55.11, p<0.001 |

### Ablation Study

| Analysis Configuration | Key Metric | Description |
|----------|---------|------|
| BoW Classifier | Macro-F1=39.25 | Far exceeds the majority class baseline of 25.02, demonstrating the distinguishability of prompts from different SES groups |
| Technical Term Usage Rate | Low 3.32% / Mid 4.16% / High 4.94% | High SES uses technical terms more frequently |
| Anthropomorphism (Greetings) | Low 6.34% / Mid 5.08% / High 4.29% | Low SES is more prone to anthropomorphizing LLMs |
| Search-style Query Ratio | Low 46.6% / Mid 43.5% / High 45.4% | All groups use LLMs to substitute search engines |

### Key Findings

- **Significant Differences in Usage Scenarios**: High-SES groups use LLMs more in work, study, and technical scenarios for advanced tasks like programming, data analysis, and professional writing, whereas low-SES groups utilize them more for entertainment and general Q&A.
- **Different Framings of the Same Topic**: In finance, low-SES users ask about saving money, while high-SES users ask about investment strategies. In job seeking, low-SES users look for remote jobs not requiring a degree, whereas high-SES users focus on cover letters for management positions.
- **High-SES Users' Prompts are Shorter but More Abstract**: This is potentially because they possess a richer vocabulary, allowing them to express precise needs in fewer words.
- **Risk of Evaluation Benchmark Bias**: Tasks commonly used by high-SES users (summarization, math problems) are easier to evaluate using ground-truth, while tasks preferred by low-SES users rely more on human preference evaluations, potentially leading to unfairness in existing evaluations.

## Highlights & Insights

- **First Real-World Prompt Dataset with SES Annotations**: This fills the gap where existing large-scale prompt datasets lack socioeconomic background information, providing a valuable resource for subsequent research.
- **New Validation Scenario for Classic Sociolinguistic Theory**: Bernstein's restricted/elaborated code theory is validated in LLM interaction, demonstrating that social class differences likewise persist in human-computer interaction.
- **Causal Inference from Usage Disparities to the AI Gap**: Rather than merely describing differences, this work analyzes how these differences can exacerbate the digital divide via a feedback loop: low-SES users using simple language → poor system performance → lower satisfaction → less usage.

## Limitations & Future Work

- **Sample Limitations**: The study is restricted to US/UK Prolific platform users. Crowdsourced workers may be more tech-savvy than the general population, and the SES distribution leans middle-to-low.
- **Self-Reporting Bias**: Participants provided their 10 most recent prompts voluntarily, introducing potential selective reporting bias. In addition, demographics of 2.5% of the participants were inconsistent with their Prolific profiles.
- **Unverified Causal Chain**: The paper speculates that SES differences exacerbate inequality through NLP system performance disparities, but it does not empirically test performance differences across different prompt styles in current systems.
- **Lack of Longitudinal Analysis**: Cross-sectional data cannot track changes in usage habits over time, nor can it observe whether AI technology will gradually bridge the gap.

## Related Work & Insights

- **vs Cercas Curry et al. (2024a)**: They used movie and TV dialogue as proxy data to study the impact of SES on NLP performance; this work directly collects real user data, offering greater ecological validity.
- **vs Daepp and Counts (2024)**: They analyzed differences in ChatGPT usage intentions across different regions of the US, whereas this work collects finer-grained SES information at the individual level.
- **vs Kirk et al. (2024) / ShareGPT Dataset**: These large-scale prompt datasets are larger in size but lack socioeconomic background annotations, precluding the type of analysis performed in this study.

## Rating

- Novelty: ⭐⭐⭐⭐ It is the first to systematically study the impact of SES on LLM interactions, pioneering an important research direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Anchored by a 1,000-person survey and 6,482 real prompts, the research offers multi-dimensional analysis supported by rich statistical testing.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear and the motivation is fully articulated, though some analyses remain relatively shallow.
- Value: ⭐⭐⭐⭐ Offers important insights for AI fairness research and suggests practically actionable improvements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] How Numerical Precision Affects Arithmetical Reasoning Capabilities of LLMs](how_numerical_precision_affects_arithmetical_reasoning_capabilities_of_llms.md)
- [\[ACL 2025\] Retrospective Learning from Interactions](retrospective_learning_from_interactions.md)
- [\[ACL 2025\] Mind Your Tone: Investigating How Prompt Politeness Affects LLM Accuracy](mind_your_tone_investigating_how_prompt_politeness_affects_llm_accuracy_short_pa.md)
- [\[ACL 2026\] Mind the Gap: How Elicitation Protocols Shape the Stated-Revealed Preference Gap in Language Models](../../ACL2026/llm_nlp/mind_the_gap_how_elicitation_protocols_shape_the_stated-revealed_preference_gap_.md)
- [\[ACL 2025\] PiFi: Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](plugin_finetuning_bridge.md)

</div>

<!-- RELATED:END -->
