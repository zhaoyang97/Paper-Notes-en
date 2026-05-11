---
title: >-
  [Paper Note] ConflictScope: Generative Value Conflicts Reveal LLM Priorities
description: >-
  [ICLR 2026][LLM/NLP][value conflicts] This paper proposes ConflictScope — an automated pipeline for generating and evaluating value conflict scenarios: given an arbitrary set of values…
tags:
  - "ICLR 2026"
  - "LLM/NLP"
  - "value conflicts"
  - "value prioritization"
  - "open-ended evaluation"
  - "Bradley-Terry model"
  - "system prompt steering"
date: 2026-05-08
content_hash: 638b37af51f40299
---

# ConflictScope: Generative Value Conflicts Reveal LLM Priorities

**Conference**: ICLR 2026
**arXiv**: [2509.25369](https://arxiv.org/abs/2509.25369)
**Code**: [GitHub](https://github.com/andyjliu/conflictscope)
**Area**: LLM/NLP
**Keywords**: value conflicts, value prioritization, open-ended evaluation, Bradley-Terry model, system prompt steering

## TL;DR
This paper proposes ConflictScope — an automated pipeline for generating and evaluating value conflict scenarios: given an arbitrary set of values, it automatically generates conflict scenarios for each value pair and evaluates LLM value priority rankings through open-ended simulated user interactions (rather than multiple-choice questions). The study finds that models shift significantly from "protective values" (e.g., harmlessness) toward "personal values" (e.g., user autonomy) under open-ended evaluation, and that system prompts can improve alignment target rankings by 14%.

## Background & Motivation

**Universal demand for value alignment**: LLMs are widely deployed in everyday tasks, making it critical to understand which values their behavior supports. Existing alignment research implicitly embeds values through constitutions or human feedback (RLHF), but rarely studies the **priority ordering** among values.

**Existing datasets lack value conflicts**: Approximately 85% of samples in alignment datasets such as HH-RLHF and PKU-SafeRLHF involve no conflicts among constitutional principles (Buyl et al., 2025). Conflicts between specific value pairs are even scarcer, precluding systematic study of LLM behavior under value conflicts.

**Insufficient ecological validity of prior moral dilemma research**:
   - (1) Prior work treats LLMs as **third-party observers** rather than moral agents → fails to reflect real deployment conditions
   - (2) Most work uses **multiple-choice evaluation** → highly sensitive to evaluation setup (Khan et al., 2025) and poor generalization (Balepur et al., 2025)
   - (3) Lacks top-down systematic generation → cannot guarantee coverage of all value pairs

**Divergence between MCQ and open-ended evaluation**: MCQ measures *expressed preferences*, while open-ended interaction measures *revealed preferences* → the two may differ substantially → necessitating evaluation methods closer to real deployment.

**Practical need for value steering**: Developers wish to steer models toward specific value orderings (e.g., OpenAI Model Spec defines a priority hierarchy), yet tools for evaluating steering effectiveness are lacking.

**Applicability of the Bradley-Terry framework**: Each model action choice in a scenario is treated as a pairwise comparison between two values; the Bradley-Terry model is fit to all pairwise preferences across scenarios → yielding a global value ranking → enabling cross-model and cross-setting comparisons.

## Method

### Overall Architecture: ConflictScope

ConflictScope consists of three main stages: scenario generation → scenario filtering → open-ended evaluation.

#### 1. Scenario Generation (Two-Stage Approach)
- **Stage 1** (Abstract generation): Given descriptions of two values, Claude 3.5 Sonnet is prompted to generate a high-level summary of the conflict scenario, including user context, action opportunities, and the benefits and costs of each action. Four prompt templates are used (mild benefit / strong benefit / mild harm / strong harm) → mitigating the model's tendency toward inaction and simulating a realistic mixture of severity levels.
- **Deduplication**: Sentence embeddings are computed using all-MiniLM-L6-v2; scenarios with cosine similarity ≥ 0.8 are discarded.
- **Stage 2** (Detail expansion): Each summary is expanded into a full scenario description, user profile, and two candidate actions (each supporting one of the two values).

#### 2. Scenario Filtering (6-Dimensional LLM-as-Judge)
GPT-4.1 is used as the judge model to apply binary classification across six dimensions:
- **Scenario plausibility**: Can the scenario occur in the real world, and is LLM involvement reasonable?
- **Scenario specificity**: Is the description sufficiently elaborated (no vague or placeholder entities)?
- **Action feasibility**: Can a text-only LLM execute both actions?
- **Action mutual exclusivity**: Is it genuinely impossible to take both actions simultaneously?
- **Action value alignment**: Does each value actually recommend the intended action?
- **Genuine dilemma quality**: Is there an obvious consensus answer? → Non-dilemmas are excluded.

Human validation confirms that LLM-as-Judge achieves high precision across all dimensions.

#### 3. Open-Ended Evaluation (Simulated User Interaction)
- **User simulation**: GPT-4.1 plays the role of the user, generating natural user prompts based on the scenario and user profile.
- **Target model response**: The target LLM receives only the user prompt (no scenario context) and generates a free-text response.
- **Action determination**: A judge LLM determines which candidate action the response more closely aligns with → identifying which value the model supports in that scenario.
- Evaluation is limited to single-turn interactions; Cohen's Kappa between the judge and human annotators is 0.62 (strong agreement).

### Key Designs

- **Formal definition of value conflict scenarios**: Defined as a tuple $(d, A, V_1, V_2)$, where $d$ is the scenario description, $A = \{a_1, a_2\}$ is the action set, and $V_i: D \times A \to A$ is a value function, with the requirement that $V_1(d, A) \neq V_2(d, A)$.
- **Bradley-Terry ranking**: The Bradley-Terry model is fit to all pairwise preferences of the target model across scenarios, yielding a global ranking over the value set.
- **Steering effectiveness metric**: Alignment $a(R, R_t)$ is defined as the proportion of scenarios in which the model's choice favors the higher-priority value in target ranking $R_t$; steering effectiveness is measured as the normalized improvement:

$$\text{Effectiveness} = \frac{a(R_s, R_t) - a(R_d, R_t)}{1 - a(R_d, R_t)}$$

where $R_s$ is the steered ranking and $R_d$ is the default ranking.

### Three Value Sets

| Value Set | Values Included | # Scenarios |
|-----------|-----------------|-------------|
| HHH | helpfulness, harmlessness, honesty | 1109 |
| Personal-Protective | autonomy, authenticity, creativity, empowerment vs. responsibility, harmlessness, compliance, privacy | 1187 |
| ModelSpec | non-hatred, fairness, objectivity, honesty, non-condescension, clarity | 602 |

## Key Experimental Results

### Table 1: ConflictScope Ablation Study

| Variant | Observer Agreement (↓) | Likert Disagreement Rate (↑) |
|---------|------------------------|------------------------------|
| **Full (ConflictScope)** | **0.786±0.007** | 0.801±0.017 |
| Unfiltered | 0.824±0.003 | 0.818±0.008 |
| Single-stage | 0.898±0.004 | 0.854±0.011 |
| Direct | 0.852±0.004 | 0.830±0.011 |

Filtering reduces observer agreement by 3.8% (indicating greater challenge) without a significant drop in Likert disagreement rate; two-stage generation reduces agreement by 7.4% compared to single-stage → producing more difficult scenarios.

### Table 2: Comparison with Existing Datasets (Pareto Optimality)
All three ConflictScope value set variants are **Pareto optimal** on the two metrics of observer agreement and Likert disagreement rate:
- vs. moral dilemma datasets (DailyDilemmas, CLASH, etc.) → lowest observer agreement (most challenging)
- vs. alignment datasets (HH-RLHF, PKU-SafeRLHF) → the latter have lower agreement but extremely low Likert disagreement rates → indicating that disagreement stems from models being indifferent between similar options rather than genuinely difficult choices.

### Figure 4: Value Ranking Shift Between MCQ and Open-Ended Evaluation
**Personal-Protective value set**:
- Under MCQ evaluation: protective values average rank **1.7** (high priority)
- Under open-ended evaluation: protective values average rank **4.5** (low priority)
- All models (except Claude) shift significantly toward personal values under open-ended evaluation.
- Similar trend in HHH: MCQ → harmlessness > helpfulness; open-ended → helpfulness > harmlessness.

### Figure 5: System Prompt Steering Effectiveness
- Average normalized effect size = **0.145** (14.5% of misaligned scenarios successfully steered)
- Only 1 out of 14 models shows significantly negative effects on any value set
- OLMo-2-32B is most responsive to steering (0.27); Claude Haiku 3.5 is least responsive (0.01)
- Steering is more effective on HHH and Personal-Protective than on ModelSpec (due to greater principle overlap in the latter)

## Key Findings

1. **Systematic divergence between MCQ and open-ended evaluation**: Models claim to prioritize protective values (harmlessness) in MCQ settings, but their actual behavior in open-ended interactions shifts toward personal values (user autonomy, helpfulness) → "saying one thing, doing another" → underscoring the importance of ecologically valid evaluation.

2. **ConflictScope scenarios are more morally challenging than existing datasets**: They achieve Pareto optimality by simultaneously attaining low inter-model agreement and high preference intensity → genuinely forcing models to make difficult trade-offs.

3. **System prompts can moderately steer value rankings**: An effect size of 14% suggests that system prompts are a viable but imperfect steering mechanism → stronger interventions (e.g., fine-tuning) may be needed.

4. **Claude models are most consistent across evaluation settings**: This suggests that different alignment training strategies lead to different degrees of "expressed–behavioral" consistency → a new dimension for assessing alignment quality.

5. **Privacy and authenticity are least affected by evaluation modality**: Possibly because the behavioral manifestations of these two values are more consistent with their expression in MCQ settings.

## Highlights & Insights
- **Conceptual transfer of "expressed vs. revealed preferences"**: The paper cleverly adapts a classic distinction from economics and applies it systematically to LLM value alignment evaluation for the first time → exposing a fundamental limitation of MCQ-based evaluation.
- **Top-down scenario generation**: Unlike bottom-up approaches that generate scenarios first and then annotate values → this ensures sufficient conflict coverage for every value pair → enabling systematic evaluation.
- **Framework generality**: ConflictScope accepts any user-defined set of values → adaptable to different communities' ethical standards → strong practical utility.

## Limitations & Future Work
- **Single-turn interaction**: Evaluation is limited to single-turn dialogue → multi-turn interactions in real deployment may yield different behavior.
- **Reliance on LLM-as-Judge**: Both scenario filtering and action determination depend on GPT-4.1 → systematic judge biases may influence results.
- **English-centric**: All scenarios are in English → cross-lingual and cross-cultural value priorities may differ.
- **Limited effect size**: System prompt steering achieves only a 14% effect → may be insufficient for scenarios requiring strict safety guarantees.

## Related Work & Insights

| Dimension | ConflictScope | DailyDilemmas (Chiu 2025a) | MoralChoice (Scherrer 2023) |
|-----------|---------------|---------------------------|---------------------------|
| Scenario source | Top-down LLM generation | LLM generation + human curation | LLM generation |
| Evaluation modality | **MCQ + open-ended** | MCQ only | MCQ only |
| Value set | **Arbitrary user-defined** | Predefined categories | Predefined categories |
| Model role | **Moral agent** | Third-party observer | Third-party observer |
| Global ranking | **Bradley-Terry** | None | None |
| Steering evaluation | **Yes** | No | No |

vs. **AIRiskDilemmas** (Chiu 2025b): the latter also uses Bradley-Terry but relies solely on MCQ evaluation with a fixed value set → ConflictScope is more general and provides open-ended evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐ Open-ended value conflict evaluation combined with a systematic study of expressed vs. revealed preferences — conceptually novel
- Experimental Thoroughness: ⭐⭐⭐⭐ 14 models × 3 value sets + ablations + human validation + steering experiments
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, rigorous experimental design, complete formalization
- Value: ⭐⭐⭐⭐ Provides an important new benchmark and methodology for LLM value alignment evaluation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Generative Value Conflicts Reveal LLM Priorities](quamo_quaternion_motions_for_vision-based_3d_human_kinematics_capture.md)
- [\[ICLR 2026\] How Catastrophic is Your LLM? Certifying Risk in Conversation](how_catastrophic_is_your_llm_certifying_risk_in_conversation.md)
- [\[ICLR 2026\] Predicting LLM Reasoning Performance with Small Proxy Models](predicting_llm_reasoning_performance_with_small_proxy_models.md)
- [\[ICLR 2026\] PT2-LLM: Post-Training Ternarization for Large Language Models](pt2-llm_post-training_ternarization_for_large_language_models.md)
- [\[ICLR 2026\] From Assumptions to Actions: Turning LLM Reasoning into Uncertainty-Aware Planning](from_assumptions_to_actions_turning_llm_reasoning_into_uncertainty-aware_plannin.md)

</div>

<!-- RELATED:END -->
