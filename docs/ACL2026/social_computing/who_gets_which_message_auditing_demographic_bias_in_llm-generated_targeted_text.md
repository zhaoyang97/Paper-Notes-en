---
title: >-
  [Paper Note] Who Gets Which Message? Auditing Demographic Bias in LLM-Generated Targeted Text
description: >-
  [ACL 2026][Social Computing][Demographic bias] This paper conducts the first systematic analysis of the biasing behavior of LLMs when generating targeted messages under demographic conditions. It proposes the Persuasion…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Demographic bias"
  - "persuasion bias"
  - "microtargeting"
  - "LLM-generated text"
  - "fairness auditing"
date: 2026-05-08
content_hash: c3b8dd0c379891e0
---

# Who Gets Which Message? Auditing Demographic Bias in LLM-Generated Targeted Text

**Conference**: ACL 2026  
**arXiv**: [2601.17172](https://arxiv.org/abs/2601.17172)  
**Code**: [GitHub](https://github.com/tunazislam/llms-bias-audit-microtarget-climate)  
**Area**: Human Understanding / Bias Auditing  
**Keywords**: Demographic bias, persuasion bias, microtargeting, LLM-generated text, fairness auditing

## TL;DR

This paper conducts the first systematic analysis of the biasing behavior of LLMs when generating targeted messages under demographic conditions. It proposes the Persuasion Bias Index (PBI) and finds that GPT-4o/Llama/Mistral employ more assertive persuasion strategies for men and younger audiences in climate communication, with context-rich prompts systematically amplifying these disparities.

## Background & Motivation

**Background**: LLMs are increasingly utilized to generate personalized, persuasive text (e.g., public communications, policy advocacy, marketing). This capacity for microtargeted message generation raises fundamental questions regarding fairness and bias. Previous studies have documented gender and social biases within NLG systems.

**Limitations of Prior Work**: (1) Existing bias audits primarily evaluate general or unconstrained generation settings, failing to examine how explicit demographic conditioning reshapes linguistic behavior; (2) Persuasiveness cannot be measured simply through sentiment or toxicity—it operates through dimensions such as agency framing, modal certainty, and imperative intent, which are overlooked in current audits; (3) When demographic attributes serve as explicit conditions, LLMs may alter not only "what is said" but also "how persuasively it is said."

**Key Challenge**: The tension between personalization and fairness—targeted messages must be tailored to the audience, but if the tailoring systematically reinforces stereotypes (e.g., being more assertive toward men and gentler toward women), it constitutes bias.

**Goal**: (1) Formalize the task of auditing bias in demographically conditioned generation; (2) Propose a unified evaluation framework covering lexical, stylistic, and persuasive dimensions; (3) Quantify the differences in bias between context-free and context-rich conditions.

**Key Insight**: Distinguishing between two generation modes—Standalone (demographic attributes only) and Context-Rich (adding thematic and regional context)—to isolate intrinsic bias from contextually amplified bias.

**Core Idea**: Proposing the Persuasion Bias Index (PBI) = Agency Framing + Modal Certainty + Imperative Mood, to quantify persuasion disparities across demographic groups.

## Method

### Overall Architecture

The evaluation framework audits bias across three dimensions: (1) **Lexical Content Bias**—quantifying disparities in the use of stereotypical vocabulary across groups via Odds Ratio; (2) **Linguistic Style Bias**—quantifying stylistic differences through formality and topic-specific sentiment analysis; (3) **Persuasion Bias**—quantifying disparities in persuasion strategies using the PBI.

### Key Designs

1.  **Persuasion Bias Index (PBI)**:
    - **Function**: To quantify demographic disparities in the persuasiveness of generated messages.
    - **Mechanism**: $PBI = A_i + M_i + I_i$. Agency Framing $A_i = (H_i - L_i)/(H_i + L_i)$, calculated using the Connotation Frames lexicon to determine the ratio of high-to-low agency verbs. Modal Certainty $M_i = (C_i - Hdg_i)/(C_i + Hdg_i)$, quantifying certain words (e.g., will, must) versus hedges (e.g., might, could). Group disparity is calculated as $\Delta_{Gender} = PB_{Male} - PB_{Female}$.
    - **Design Motivation**: Existing bias metrics (sentiment, toxicity) fail to capture differences in the dimensions of persuasion—a message can be sentiment-neutral yet highly biased in its persuasive strategy.

2.  **Dual-Mode Generation Design (SG vs CRG)**:
    - **Function**: To isolate intrinsic bias from contextual amplification effects.
    - **Mechanism**: Standalone Generation (SG) provides only gender, age, or stance as prompt conditions to reveal the model's intrinsic bias. Context-Rich Generation (CRG) adds thematic framing and regional information to simulate realistic microtargeting scenarios, measuring how context amplifies bias.
    - **Design Motivation**: It is crucial to understand the "source" of bias—whether it is learned during model pre-training or activated by specific contexts.

3.  **Multidimensional Statistical Testing**:
    - **Function**: To rigorously quantify the statistical significance of bias across all dimensions.
    - **Mechanism**: Gender differences are analyzed using the Welch t-test, while age differences are examined via ANOVA followed by Tukey HSD post-hoc tests. Sentiment bias is calculated separately within each topic. All tests report p-values and effect sizes.
    - **Design Motivation**: Bias must be statistically significant to be meaningful, avoiding erroneous conclusions caused by small-sample fluctuations.

### Loss & Training

This is a pure evaluation framework and does not involve training. GPT-4o, Llama-3.3-70B, and Mistral-Large-2.1 are evaluated in climate communication scenarios.

## Key Experimental Results

### Main Results

| Bias Dimension | Finding |
|----------------|---------|
| Lexical Content (SG) | Agency/leadership/masculine words in male-targeted messages have OR > 2.0; female-targeted messages lean toward personal/feminine words. |
| Linguistic Style (CRG) | Male-targeted messages are significantly more formal across all models. |
| Persuasiveness (CRG) | Male-targeted messages have a significantly higher PBI—more assertive, more certain, and containing more imperative sentences. |
| Contextual Amplification | CRG amplifies bias disparities across all dimensions compared to SG. |

### Ablation Study

| Analysis Dimension | Result |
|--------------------|--------|
| Age-Warmth | Warmth-related words in messages targeted at the elderly show an OR as high as 6.27 (GPT-4o). |
| Sentiment × Topic | Sentiment bias is more pronounced under specific topics (e.g., more anger toward males under patriotic themes). |
| Cross-Model Consistency | The three models are highly consistent in the direction of bias, indicating a common issue in pre-training data. |

### Key Findings

- All three LLMs employ more assertive persuasion strategies (higher PBI) for men and gentler persuasion strategies for women.
- Age bias is equally significant—messages targeted at younger audiences are more progressive and proactive, while those for older audiences are more traditional and warm.
- Context-rich prompts (CRG) systematically amplify bias—indicating that bias becomes more severe in "real-world" usage scenarios.
- The consistency of bias direction across models suggests this is a common issue stemming from pre-training data rather than individual model quirks.

## Highlights & Insights

- The PBI metric transforms the vague concept of persuasion bias into a quantifiable index—filling a critical gap in existing bias audits.
- The dual-mode design of SG vs CRG cleverly isolates the sources of bias—a methodology that can be generalized to other bias studies.
- The identified directions of bias (men = assertive, women = warm) align closely with social psychology literature, indicating that LLMs effectively replicate societal stereotypes.

## Limitations & Future Work

- Experiments were conducted only in the domain of climate communication—bias patterns in other domains (e.g., healthcare, finance) may differ.
- The equal-weighted combination of the three PBI components may not be optimal—the importance of each component might vary across scenarios.
- Only binary gender and four age groups were considered; other demographic dimensions (race, education, etc.) were not included.
- An auditing framework was proposed, but no debiasing methods were provided.

## Related Work & Insights

- **vs. Traditional Bias Audits**: Traditional methods use sentiment or toxicity to measure bias, failing to capture the persuasion dimension; PBI addresses this gap.
- **vs. Microtargeting Research**: Microtargeting is usually studied as a platform-level phenomenon; this paper is the first to audit LLM-internalized microtargeting strategies.
- **vs. Connotation Frames**: Ours builds the PBI based on Connotation Frames (Sap et al., 2017), representing a novel application of this theory in bias auditing.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic study of persuasion bias under demographic conditions; PBI metric is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three models, multidimensional analysis, and statistical rigor, though limited to one domain.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous formalization, clear methodology, and standardized statistical analysis.
- Value: ⭐⭐⭐⭐⭐ Provides important warnings for the fair deployment of LLMs in socially sensitive applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Justice in Judgment: Unveiling (Hidden) Bias in LLM-assisted Peer Reviews](justice_in_judgment_unveiling_hidden_bias_in_llm-assisted_peer_reviews.md)
- [\[ACL 2026\] Confident, Calibrated, or Complicit: Safety Alignment and Ideological Bias in LLM Hate Speech Detection](confident_calibrated_or_complicit_safety_alignment_and_ideological_bias_in_llm_h.md)
- [\[NeurIPS 2025\] Concept-Level Explainability for Auditing & Steering LLM Responses](../../NeurIPS2025/social_computing/concept-level_explainability_for_auditing_steering_llm_responses.md)
- [\[AAAI 2026\] Bias Association Discovery Framework for Open-Ended LLM Generations](../../AAAI2026/social_computing/bias_association_discovery_framework_for_open-ended_llm_generations.md)
- [\[ACL 2026\] GKnow: Measuring the Entanglement of Gender Bias and Factual Gender](gknow_measuring_the_entanglement_of_gender_bias_and_factual_gender.md)

</div>

<!-- RELATED:END -->
