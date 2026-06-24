---
title: >-
  [Paper Note] Mind Your Tone: Investigating How Prompt Politeness Affects LLM Accuracy
description: >-
  [ACL 2025 (Findings, Short Paper)][LLM (Other)][Prompt tone] This paper systematically investigates how prompt politeness levels influence LLM response accuracy. By constructing 250 multiple-choice prompts across 5 tone gradients (ranging from "Very Polite" to "Very Rude") and testing them on ChatGPT 4o, the authors counterintuitively find that rude prompts achieve significantly higher accuracy (84.8%) than polite prompts (80.8%).
tags:
  - "ACL 2025 (Findings, Short Paper)"
  - "LLM (Other)"
  - "Prompt tone"
  - "Politeness"
  - "LLM accuracy"
  - "Human-computer interaction"
  - "Prompt engineering"
date: 2026-05-08
content_hash: a833e4876a7ebeea
---

# Mind Your Tone: Investigating How Prompt Politeness Affects LLM Accuracy

**Conference**: ACL 2025 (Findings, Short Paper)  
**arXiv**: [2510.04950](https://arxiv.org/abs/2510.04950)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Prompt tone, Politeness, LLM accuracy, Human-computer interaction, Prompt engineering

## TL;DR

This paper systematically investigates how prompt politeness levels influence LLM response accuracy. By constructing 250 multiple-choice prompts across 5 tone gradients (ranging from "Very Polite" to "Very Rude") and testing them on ChatGPT 4o, the authors counterintuitively find that rude prompts achieve significantly higher accuracy (84.8%) than polite prompts (80.8%).

## Background & Motivation

**Background**: Prompt engineering has become a critical means to improve LLM performance, with researchers extensively exploring the impacts of prompt formatting, instruction structures, and few-shot examples on model outputs. However, at the pragmatic level, the influence of prompt "tone" and "politeness" on model performance remains largely under-studied.

**Limitations of Prior Work**: A small body of prior research (e.g., Yin et al. 2024) has explored the impact of emotional factors on LLMs, with some finding that rude expressions lead to degraded model performance. However, these conclusions may become outdated as models iterate, and they lack a fine-grained analysis across different politeness gradients. Differences in alignment training strategies across various LLM versions could result in entirely distinct tone preferences.

**Key Challenge**: On one hand, alignment training like RLHF encourages models to provide more helpful answers to polite users; on the other hand, rude or direct expressions are often more precise and less ambiguous at the instruction level, which might actually facilitate task completion. Redundant information introduced by polite expressions (e.g., "I would be grateful if you could take the time to...") may distract the model's attention.

**Goal**: To quantify the impact of different tone levels on LLM multiple-choice question-answering accuracy, and to test whether this difference is statistically significant.

**Key Insight**: Designing controlled experiments that maintain identical question content while altering only the tone wrapping (Very Polite $\rightarrow$ Polite $\rightarrow$ Neutral $\rightarrow$ Rude $\rightarrow$ Very Rude), thereby isolating the independent effect of the tone variable.

**Core Idea**: Validating the causal effect of tone on LLM accuracy via paired sample t-tests, discovering that next-generation LLMs perform better under rude prompts.

## Method

### Overall Architecture

The experiments adopt a classic controlled variable design: first, 50 basic multiple-choice questions (covering three domains: mathematics, science, and history) are constructed; then, each question is rewritten into 5 tone variants (Very Polite / Polite / Neutral / Rude / Very Rude), yielding a total of $50 \times 5 = 250$ prompts. All prompts are input into ChatGPT 4o to record accuracy rates, and paired sample t-tests are finally applied to evaluate the statistical significance of the differences.

### Key Designs

1. **Five-level Tone Gradient Construction**:

    - Function: To generate tone-different but content-equivalent prompt variants for the same question.
    - Mechanism: "Very Polite" employs honorifics and expressions of gratitude (e.g., "I would be incredibly grateful if you could..."), "Polite" uses "please" and "could you", "Neutral" directly poses questions without embellishment, "Rude" utilizes imperative and impatient tones, and "Very Rude" incorporates insulting and aggressive expressions (e.g., "Are you stupid? Just answer this..."). Each variant preserves the exact same core question content and options.
    - Design Motivation: A 5-level gradient reveals finer-grained trends than a binary division (polite vs. rude), such as the existence of a "sweet spot" tone.

2. **Multi-domain Base Question Set**:

    - Function: To ensure that conclusions are free from domain-specific biases.
    - Mechanism: Sampling approximately 17 questions from each of the three domains (mathematics, science, and history), with each question formatted as a standard multiple-choice QA with a clear, unambiguous answer.
    - Design Motivation: Different domains exhibit varying semantic complexities; multi-domain coverage enhances the generalizability of the findings.

3. **Paired Sample t-test**:

    - Function: To statistically test whether accuracy differences between different tone pairs are significant.
    - Mechanism: For each pair of tone conditions (e.g., VP vs. VR), accuracy is calculated respectively across the 50 questions, followed by a paired t-test. Since the same underlying question naturally forms pairs across different tones, the premise of paired tests is satisfied.
    - Design Motivation: To eliminate confounding variables introduced by differences in question difficulty, directly comparing the performance of the same question under different tones.

### Loss & Training

This is an empirical study and does not involve model training or loss functions. The core analytical method is the paired sample t-test, with the null hypothesis being "there is no difference in accuracy between tone conditions."

## Key Experimental Results

### Main Results

| Tone Condition | Accuracy | Difference from Neutral |
|----------|--------|--------------|
| Very Polite | 80.8% | -1.2% |
| Polite | 81.2% | -0.8% |
| Neutral | 82.0% | Baseline |
| Rude | 83.6% | +1.6% |
| Very Rude | 84.8% | +2.8% |

### Statistical Significance Test

| Comparison Condition | t-value | p-value | Significance |
|----------|-----|-----|--------|
| Very Polite vs. Very Rude | - | $<0.05$ | Significant |
| Polite vs. Rude | - | $<0.05$ | Significant |
| Neutral vs. Very Rude | - | $<0.05$ | Significant |

### Key Findings

- Accuracy increases monotonically as politeness decreases, ranging from 80.8% for "Very Polite" to 84.8% for "Very Rude", showing a clear linear trend.
- This contradicts the findings of earlier studies like Yin et al. (2024), which reported that rude prompts reduced performance on older models.
- Paired t-tests confirm that the differences are statistically significant ($p < 0.05$), ruling out the possibility of random fluctuation.
- Possible explanations: rude prompts are often phrased more concisely and directly, reducing interference from redundant information; furthermore, the alignment training of next-generation models might have weakened their sensitivity to tone.

## Highlights & Insights

- The **counterintuitive discovery** is the major highlight: it challenges the naive assumption that "polite prompts yield better results," showing that RLHF alignment of LLMs does not equate to higher-quality responses under polite phrasing. This finding has direct implications for prompting strategies.
- The **experimental design is simple yet effective**: confounding variables are elegantly controlled via a paired design, yielding credible conclusions with a minimized experimental scale. This method of "rewriting the same question" can be transferred to research other pragmatic dimensions (such as formality, sentiment, and euphemism).
- It **implies the limitations of alignment training**: RLHF might teach the model "not to generate offensive responses due to rude tones," but it does not compel the model to allocate more "reasoning capacity" under polite requests.

## Limitations & Future Work

- **Only one model tested (ChatGPT 4o)**: Whether findings extend to open-source models (such as LLaMA, Mistral, etc.) or models with different RLHF strategies remains unknown.
- **Small data scale**: $50 \text{ questions} \times 5 \text{ tones}$, totaling 250 samples, which limits statistical power. Scaling up to 500+ questions would yield more robust conclusions.
- **Limited to multiple-choice QA tasks**: The impact of tone may differ completely in open-ended generation, chain-of-thought reasoning, code generation, and other tasks.
- **Difficulty in controlling the quality of tone rewrites**: Perfect semantic equivalence across different tone variants is difficult to guarantee; rude expressions might unintentionally simplify the question phrasing.
- Future work can scale to larger-scale studies involving multi-lingual, multi-model, and multi-task setups.

## Related Work & Insights

- **vs. Earlier studies such as Yin et al. (2024)**: They found that rude expressions degrade LLM performance, whereas this work arrives at the opposite conclusion on the newer GPT-4o, indicating that iterations of alignment strategies may alter the models' tone sensitivity.
- **vs. EmotionPrompt (Li et al. 2023)**: They studied the influence of emotional phrases (such as "this is important for my career") on LLMs and found that positive emotions lead to improvements. This paper focuses on tone rather than emotional intent—distinct yet complementary dimensions.
- Although this short paper is modest in scale, the phenomenon it sheds light on warrants deeper exploration: if rude tones are indeed effective, should prompt design aim for "concise and direct" rather than "polite and considerate"?

## Rating

- Novelty: ⭐⭐⭐⭐ The research angle is novel and the findings are counterintuitive, but it represents an empirical observation rather than algorithmic innovation.
- Experimental Thoroughness: ⭐⭐⭐ Acceptable as a short paper, but the scale is too small (50 questions, single model) and lacks ablation or cross-model validation.
- Writing Quality: ⭐⭐⭐⭐ Standard short paper structure, clear and logical argumentation.
- Value: ⭐⭐⭐ Highlights an interesting phenomenon, but requires larger-scale replication to guide practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] How Numerical Precision Affects Arithmetical Reasoning Capabilities of LLMs](how_numerical_precision_affects_arithmetical_reasoning_capabilities_of_llms.md)
- [\[ACL 2025\] The AI Gap: How Socioeconomic Status Affects Language Technology Interactions](the_ai_gap_how_socioeconomic_status_affects_language_technology_interactions.md)
- [\[ACL 2025\] LLM-AT: Automatic Transmission for LLM Tiers Optimizing Cost and Accuracy](automatic_transmission_for_llm_tiers_optimizing_cost_and_accuracy_in_large_langu.md)
- [\[ACL 2025\] A Survey of LLM-based Agents in Medicine: How Far Are We from Baymax?](a_survey_of_llm-based_agents_in_medicine_how_far_are_we_from_baymax.md)
- [\[ACL 2025\] Why Prompt Design Matters and Works: A Complexity Analysis of Prompt Search Space in LLMs](why_prompt_design_matters_and_works_a_complexity_analysis_of_prompt_search_space.md)

</div>

<!-- RELATED:END -->
