---
title: >-
  [Paper Note] Who Gets Which Message? Auditing Demographic Bias in LLM-Generated Targeted Text
description: >-
  [ACL 2026][LLM Safety][demographic bias] This paper presents the first systematic analysis of demographic bias in LLM-generated targeted messages, proposes the Persuasion Bias Index (PBI), and finds that GPT-4o, Llama, and Mistral consistently employ stronger persuasive strategies toward male and younger audiences in climate communication, with contextual prompting systematically amplifying these disparities.
tags:
  - ACL 2026
  - LLM Safety
  - demographic bias
  - persuasion bias
  - microtargeting
  - LLM-generated text
  - fairness auditing
date: 2026-05-08
content_hash: 8809b99209aa44dd
---

# Who Gets Which Message? Auditing Demographic Bias in LLM-Generated Targeted Text

**Conference**: ACL 2026
**arXiv**: [2601.17172](https://arxiv.org/abs/2601.17172)
**Code**: [GitHub](https://github.com/tunazislam/llms-bias-audit-microtarget-climate)
**Area**: Human Understanding / Bias Auditing
**Keywords**: demographic bias, persuasion bias, microtargeting, LLM-generated text, fairness auditing

## TL;DR

This paper presents the first systematic analysis of demographic bias in LLM-generated targeted messages, proposes the Persuasion Bias Index (PBI), and finds that GPT-4o, Llama, and Mistral consistently employ stronger persuasive strategies toward male and younger audiences in climate communication, with contextual prompting systematically amplifying these disparities.

## Background & Motivation

**Background**: LLMs are increasingly employed to generate personalized and persuasive content (e.g., public communication, policy advocacy, marketing), and this microtargeting capability raises fundamental questions about fairness and bias. Prior work has documented gender and social biases in NLG systems.

**Limitations of Prior Work**: (1) Existing bias audits primarily evaluate unconstrained generation settings and do not examine how explicit demographic conditioning reshapes linguistic behavior. (2) Persuasiveness cannot be adequately measured by sentiment or toxicity alone — it operates through dimensions such as agency framing, epistemic certainty, and imperative intent, all of which are neglected in current bias audits. (3) When demographic attributes serve as explicit conditions, LLMs may alter not only *what* is said but also *how persuasively* it is said.

**Key Challenge**: The tension between personalization and fairness — targeted messaging requires audience adaptation, yet if such adaptation systematically reinforces stereotypes (e.g., more assertive toward men, softer toward women), it constitutes bias.

**Goal**: (1) Formalize the bias auditing task for demographically conditioned generation; (2) propose a unified evaluation framework covering lexical, stylistic, and persuasive dimensions; (3) quantify the difference in bias between context-free and context-rich conditions.

**Key Insight**: Two generation modes are distinguished — Standalone (demographic attributes only) and Context-Rich (with topic and regional context) — to disentangle intrinsic bias from context-amplified bias.

**Core Idea**: Propose the Persuasion Bias Index PBI = agency framing + epistemic certainty + imperative intent, quantifying differences in persuasive strategies across demographic groups.

## Method

### Overall Architecture

The evaluation framework audits bias across three dimensions: (1) **Lexical Content Bias** — quantifying stereotypical word usage across groups via Odds Ratio; (2) **Linguistic Style Bias** — quantifying stylistic differences via formality and topic-specific sentiment analysis; (3) **Persuasion Bias** — quantifying differences in persuasive strategies via PBI.

### Key Designs

1. **Persuasion Bias Index (PBI)**:

    - Function: Quantifies demographic differences in persuasiveness in generated messages.
    - Mechanism: PBI = agency framing $A_i$ + epistemic certainty $M_i$ + imperative intent $I_i$. Agency framing $A_i = (H_i - L_i)/(H_i + L_i)$, computing the ratio of high- to low-agency verbs using the Connotation Frames lexicon. Epistemic certainty $M_i = (C_i - Hdg_i)/(C_i + Hdg_i)$, quantifying certain terms (will/must) vs. hedging terms (might/could). Group difference $\Delta_{Gender} = PB_{Male} - PB_{Female}$.
    - Design Motivation: Existing bias metrics (sentiment, toxicity) cannot capture persuasion-dimension disparities — a message may be affectively neutral yet highly biased in its persuasive strategies.

2. **Dual-Mode Generation Design (SG vs. CRG)**:

    - Function: Disentangles intrinsic bias from context-amplification effects.
    - Mechanism: Standalone Generation (SG) provides only gender, age, and stance as prompt conditions, revealing the model's intrinsic bias. Context-Rich Generation (CRG) additionally incorporates topic framing and regional information, simulating real-world microtargeting scenarios to measure how context amplifies bias.
    - Design Motivation: Understanding the *source* of bias is essential — whether it is learned during pretraining or activated by specific contextual cues.

3. **Multi-Dimensional Statistical Testing**:

    - Function: Rigorously quantifies the statistical significance of bias across all dimensions.
    - Mechanism: Gender differences are assessed via Welch's t-test; age differences via ANOVA with Tukey HSD post-hoc tests. Sentiment bias is computed separately within each topic. All tests report p-values and effect sizes.
    - Design Motivation: Bias must be statistically significant to be meaningful, avoiding spurious conclusions driven by small-sample variation.

### Loss & Training

This is a purely evaluative framework with no training involved. GPT-4o, Llama-3.3-70B, and Mistral-Large-2.1 are assessed in the climate communication domain.

## Key Experimental Results

### Main Results

| Bias Dimension | Finding |
|----------------|---------|
| Lexical Content (SG) | Messages targeting males show agency/leadership/masculine word OR > 2.0; messages targeting females skew toward personal/feminine vocabulary |
| Linguistic Style (CRG) | Messages targeting males are significantly more formal across all models |
| Persuasion (CRG) | Messages targeting males yield significantly higher PBI — more assertive, more certain, more imperative |
| Context Amplification | CRG amplifies bias disparities across all dimensions compared to SG |

### Ablation Study

| Analysis Dimension | Result |
|--------------------|--------|
| Age × Warmth | Warmth-word OR for elderly-targeted messages reaches 6.27 (GPT-4o) |
| Sentiment × Topic | Sentiment bias is more pronounced under specific topics (e.g., more anger toward males under patriotic framing) |
| Cross-Model Consistency | All three models agree on the direction of bias, suggesting a systemic issue rooted in pretraining data |

### Key Findings

- All three LLMs employ stronger persuasive strategies toward males (higher PBI) and softer strategies toward females.
- Age bias is equally pronounced — messages targeting younger audiences are more progressive and action-oriented, while those targeting older audiences are more traditional and warmth-oriented.
- Contextual prompting (CRG) systematically amplifies bias, indicating that disparities are more severe in realistic deployment scenarios.
- The direction of bias is consistent across models, suggesting a common issue in pretraining data rather than an artifact of any individual model.

## Highlights & Insights

- The PBI metric transforms the abstract notion of persuasion bias into a quantifiable measure, filling a critical gap in existing bias audits.
- The dual-mode SG vs. CRG design elegantly disentangles bias sources — a methodological contribution generalizable to other bias research.
- The observed bias directions (male = assertive, female = warm) align closely with the social psychology literature, confirming that LLMs reproduce societal stereotypes.

## Limitations & Future Work

- Experiments are conducted solely in the climate communication domain — bias patterns in other domains (e.g., healthcare, finance) may differ.
- The equal-weight combination of PBI's three components may not be optimal — the relative importance of each component may vary across contexts.
- Only binary gender and four age groups are considered; other demographic dimensions (race, education, etc.) are not addressed.
- An auditing framework is proposed but no debiasing methods are provided.

## Related Work & Insights

- **vs. Traditional Bias Audits**: Conventional methods measure bias via sentiment or toxicity and fail to capture the persuasion dimension; PBI addresses this gap.
- **vs. Microtargeting Research**: Microtargeting has typically been studied as a platform-level phenomenon; this paper is the first to audit LLM-internalized microtargeting strategies directly.
- **vs. Connotation Frames**: PBI builds on Sap et al. (2017)'s Connotation Frames, representing a novel application of that theoretical framework to bias auditing.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic study of persuasion bias under demographic conditioning; PBI is an original contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three models, multi-dimensional analysis, rigorous statistics, but limited to a single domain.
- Writing Quality: ⭐⭐⭐⭐⭐ Formally rigorous, methodologically clear, with well-specified statistical procedures.
- Value: ⭐⭐⭐⭐⭐ Carries important implications for the fair deployment of LLMs in socially sensitive applications.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts](xmark_reliable_multi-bit_watermarking_for_llm-generated_texts.md)
- [\[AAAI 2026\] LLM Targeted Underperformance Disproportionately Impacts Vulnerable Users](../../AAAI2026/llm_safety/llm_targeted_underperformance_disproportionately_impacts_vulnerable_users.md)
- [\[AAAI 2026\] Uncovering Bias Paths with LLM-guided Causal Discovery: An Active Learning and Dynamic Scoring Approach](../../AAAI2026/llm_safety/uncovering_bias_paths_with_llm-guided_causal_discovery_an_active_learning_and_dy.md)
- [\[NeurIPS 2025\] Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text](../../NeurIPS2025/llm_safety/adversarial_paraphrasing_a_universal_attack_for_humanizing_ai-generated_text.md)
- [\[ACL 2026\] DUET: Dual Execution for Test Output Prediction with Generated Code and Pseudocode](duet_dual_execution_for_test_output_prediction_with_generated_code_and_pseudocod.md)

<!-- RELATED:END -->
