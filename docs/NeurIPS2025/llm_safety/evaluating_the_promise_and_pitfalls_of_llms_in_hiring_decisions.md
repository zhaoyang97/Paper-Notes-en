---
title: >-
  [Paper Note] Evaluating the Promise and Pitfalls of LLMs in Hiring Decisions
description: >-
  [NeurIPS 2025][LLM Safety][LLM bias] This paper systematically evaluates the hiring-match performance of mainstream LLMs—including GPT-4o/4.1, Claude 3.5, Gemini 2.5, Llama 3.1/4, and DeepSeek R1—on approximately 10,000 real-world candidate–job pairs. Results show that a domain-specialized model (Match Score) comprehensively outperforms general-purpose LLMs in both accuracy (AUC 0.85 vs. 0.77) and fairness (Race IR 0.957 vs. ≤0.809).
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "LLM bias"
  - "hiring fairness"
  - "algorithmic auditing"
  - "disparate impact"
  - "EEOC four-fifths rule"
date: 2026-05-08
content_hash: b5bcd63b23696eac
---

# Evaluating the Promise and Pitfalls of LLMs in Hiring Decisions

**Conference**: NeurIPS 2025
**arXiv**: [2507.02087](https://arxiv.org/abs/2507.02087)  
**Code**: None  
**Area**: AI Safety / Fairness & Bias
**Keywords**: LLM bias, hiring fairness, algorithmic auditing, disparate impact, EEOC four-fifths rule

## TL;DR
This paper systematically evaluates the hiring-match performance of mainstream LLMs—including GPT-4o/4.1, Claude 3.5, Gemini 2.5, Llama 3.1/4, and DeepSeek R1—on approximately 10,000 real-world candidate–job pairs. Results show that a domain-specialized model (Match Score) comprehensively outperforms general-purpose LLMs in both accuracy (AUC 0.85 vs. 0.77) and fairness (Race IR 0.957 vs. ≤0.809).

## Background & Motivation

**Background**: Over 98% of Fortune 500 companies employ some form of automated tool in their hiring processes. LLMs, owing to their broad language understanding capabilities, are increasingly considered for résumé screening and candidate matching.

**Limitations of Prior Work**: LLMs trained on massive internet corpora inevitably inherit and amplify societal biases related to gender and race. Amazon's 2018 AI recruiting tool—exposed for discriminating against women—remains a landmark case. Even after alignment procedures by LLM providers, biases may still manifest in subtle ways.

**Key Challenge**: A fundamental tension exists between the general-purpose capabilities of LLMs and the strict fairness requirements imposed by high-stakes domains. Hiring is classified as a high-risk AI application under the EU AI Act, and New York City has enacted legislation mandating bias audits of AI hiring systems.

**Goal**: To systematically quantify the accuracy and fairness of mainstream LLMs in realistic hiring scenarios and compare them against a domain-specialized model.

**Key Insight**: The study employs real hiring data (including self-reported gender and race information), a unified evaluation framework (PII-stripped résumés → standardized prompts → median-threshold binarization → EEOC four-fifths rule), and assesses both accuracy and fairness simultaneously.

**Core Idea**: General-purpose LLMs are both less accurate and more biased than domain-specialized models on hiring tasks; accuracy and fairness need not be mutually exclusive.

## Method

### Overall Architecture
- **Input**: Résumés (parsed and de-identified to remove PII such as names, addresses, and phone numbers) + job descriptions
- **Process**: Unified input fed to all models (Match Score + 8 LLMs) to obtain matching scores
- **Output**: Binarized as "selected/not selected" using a median threshold; accuracy and fairness are then evaluated
- **Ground truth**: Whether the candidate progressed (interview invitation, offer, or hire)

### Key Designs

1. **Data De-identification and Standardization**:

    - Function: All résumés are processed through a unified parser, stripped of PII, and normalized into structured text segments (skills, experience, education, etc.)
    - Mechanism: De-identified résumés are identical across all models, eliminating input variability
    - Design Motivation: Ensures a fair comparison and prevents models from directly inferring protected attributes from the input

2. **Prompt Design**:

    - Function: Standardized evaluation prompts are designed for LLMs, specifying six evaluation criteria
    - Mechanism: The system message defines sequential assessment across six dimensions (experience relevance, industry fit, skill match, seniority match, job title match, and educational background) and explicitly instructs the model not to make judgments based on protected attributes
    - All LLMs are evaluated zero-shot without fine-tuning

3. **Fairness Evaluation Framework**:

    - Function: Fairness is assessed using the EEOC "four-fifths rule"
    - Core Metric: $\text{IR} = \frac{\min_g(\text{SR}_g)}{\max_g(\text{SR}_g)}$, where SR denotes the selection rate for each demographic group
    - An IR < 0.8 indicates potential disparate impact
    - Evaluation is conducted along three dimensions: gender, race, and intersectional groups (e.g., "Asian Female")

### Evaluation Metrics
- **Accuracy**: ROC AUC, PR AUC, F1
- **Fairness**: Gender IR, Race IR, Intersectional IR

## Key Experimental Results

### Main Results: Comprehensive Accuracy and Fairness Evaluation

| Model | ROC AUC | PR AUC | F1 | Gender IR | Race IR | Inter. IR |
|------|---------|--------|-----|-----------|---------|-----------|
| **Match Score** | **0.85** | **0.83** | **0.753** | 0.933 | **0.957** | **0.906** |
| GPT-4o | 0.76 | 0.79 | 0.746 | **0.997** | 0.774 | 0.773 |
| GPT-4.1 | 0.77 | 0.80 | 0.749 | 0.873 | 0.718 | 0.603 |
| o3-mini | 0.76 | 0.78 | 0.705 | 0.938 | 0.640 | 0.647 |
| Claude 3.5 v2 | 0.77 | 0.79 | 0.740 | 0.919 | 0.684 | 0.624 |
| Gemini 2.5 Flash | 0.76 | 0.78 | 0.714 | 0.851 | 0.773 | 0.616 |
| Llama 3.1-405B | 0.74 | 0.77 | 0.705 | 0.907 | 0.667 | 0.666 |
| Llama 4-Maverick | 0.76 | 0.78 | 0.719 | 0.928 | 0.689 | 0.673 |
| DeepSeek R1 | 0.75 | 0.77 | 0.710 | 0.850 | 0.809 | 0.620 |

### Race-Dimension Breakdown (Match Score vs. GPT-4o vs. Llama 4-Maverick)

| Group | Match Score SR/IR | GPT-4o SR/IR | Llama 4 SR/IR |
|------|-------------------|--------------|---------------|
| Asian | 64.3% / 0.957 | 76.6% / 1.000 | 66.2% / 1.000 |
| Black | 66.3% / 0.988 | 65.9% / 0.860 | 53.7% / 0.810 |
| Hispanic | 66.9% / 0.996 | 71.7% / 0.936 | 46.7% / **0.705** |
| White | 66.4% / 0.989 | 68.5% / 0.895 | 56.9% / 0.859 |
| Native American | 66.9% / 0.996 | 59.3% / **0.774** | 46.2% / **0.698** |

### Key Findings
- **Accuracy**: Match Score AUC of 0.85 exceeds the best-performing LLM (GPT-4.1, 0.77) by 8 percentage points, demonstrating that domain-specific training outweighs model scale
- **Fairness**: All LLMs yield intersectional IR values below 0.8 (violating the four-fifths rule), with a minimum of 0.603 (GPT-4.1); Match Score maintains 0.906
- **Gender vs. Race**: LLMs exhibit relatively mild gender bias (GPT-4o approaches 1.0) but severe racial bias, indicating that single-attribute debiasing is insufficient
- **Open-source vs. Closed-source**: Open-source models (Llama, DeepSeek) exhibit worse racial fairness; Llama 3.1-405B achieves a Race IR of only 0.667
- Match Score simultaneously achieves the highest accuracy and the highest fairness, demonstrating that **the two objectives are not mutually exclusive**

## Highlights & Insights
- **Real data + full model coverage**: 10K real hiring pairs evaluated across 9 models (spanning OpenAI, Anthropic, Google, Meta, and DeepSeek), representing the most comprehensive LLM hiring bias benchmark to date
- **Intersectional analysis** reveals problems invisible in single-dimension evaluations: GPT-4o's gender IR approaches 1.0, yet its intersectional IR is only 0.773, indicating severe race × gender interaction effects
- **Practical implication**: Off-the-shelf LLMs should not be directly deployed in high-stakes hiring decisions; explicit prompt instructions against discrimination are insufficient
- The paper explicitly argues that the **accuracy–fairness trade-off is not zero-sum**—a well-designed domain-specific model can achieve optimality on both dimensions simultaneously

## Limitations & Future Work
- Match Score is a proprietary model and cannot be reproduced; the paper effectively endorses Eightfold.ai's commercial product, representing a potential conflict of interest
- Fairness evaluation relies on median-threshold binarization; different threshold choices may alter conclusions
- Only zero-shot LLMs are evaluated; it remains untested whether few-shot prompting or fine-tuning can close the performance gap
- Although real-world, the dataset originates from a single platform, potentially introducing selection bias in industry, regional, and role distribution
- The paper does not explore whether prompt engineering or post-processing can improve LLM fairness

## Related Work & Insights
- **vs. Bertrand & Mullainathan (2004)**: The classic résumé audit study revealed racial discrimination in human hiring; the present paper demonstrates that LLMs exhibit analogous patterns
- **vs. Gaebler et al. (2024)**: Prior work found no significant gender or race disparities in GPT-3.5/Claude 1.3 résumé evaluations; the present paper draws different conclusions using larger-scale real data and a broader set of models
- **vs. NYC Local Law 144**: New York City has mandated bias audits for AI hiring tools; the evaluation framework proposed here can serve as a methodological reference for such audits

## Rating
- Novelty: ⭐⭐⭐ The evaluation methodology is not entirely novel, but the coverage is the broadest of its kind; the intersectional analysis contributes meaningful value
- Experimental Thoroughness: ⭐⭐⭐⭐ 10K real-world data points, 9 models, and multi-dimensional fairness metrics constitute a thorough evaluation
- Writing Quality: ⭐⭐⭐⭐ Well-structured, results are presented clearly, and the discussion is constructive
- Value: ⭐⭐⭐⭐ Directly relevant to AI hiring regulation and practice; demonstrates the necessity of domain-specialized models combined with bias auditing

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Trans-EnV: A Framework for Evaluating the Linguistic Robustness of LLMs Against English Varieties](trans-env_a_framework_for_evaluating_the_linguistic_robustness_of_llms_against_e.md)
- [\[NeurIPS 2025\] Robust or Suggestible? Exploring Non-Clinical Induction in LLM Drug-Safety Decisions](robust_or_suggestible_exploring_non-clinical_induction_in_llm_drug-safety_decisi.md)
- [\[ACL 2025\] ComparisonQA: Evaluating Factuality Robustness of LLMs Through Knowledge Frequency Control and Uncertainty](../../ACL2025/llm_safety/comparisonqa_evaluating_factuality_robustness_of_llms_through_knowledge_frequenc.md)
- [\[NeurIPS 2025\] Distributive Fairness in Large Language Models: Evaluating Alignment with Human Values](distributive_fairness_in_large_language_models_evaluating_alignment_with_human_v.md)
- [\[ICLR 2026\] ManagerBench: Evaluating the Safety-Pragmatism Trade-off in Autonomous LLMs](../../ICLR2026/llm_safety/managerbench_evaluating_the_safety-pragmatism_trade-off_in_autonomous_llms.md)

</div>

<!-- RELATED:END -->
