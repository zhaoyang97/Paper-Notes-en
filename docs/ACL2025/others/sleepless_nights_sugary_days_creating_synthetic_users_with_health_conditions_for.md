---
title: >-
  [Paper Note] Sleepless Nights, Sugary Days: Creating Synthetic Users with Health Conditions for Realistic Coaching Agent Interactions
description: >-
  [ACL 2025][Synthetic users] Proposes an end-to-end framework to generate synthetic users with health conditions (covering sleep and diabetes management) based on real demographic, health/lifestyle, and behavioral/psychological profile data. This framework is used to evaluate the interaction quality of health coaching agents, and is validated through human expert evaluation to significantly outperform generic synthetic users.
tags:
  - "ACL 2025"
  - "Synthetic users"
  - "health condition modeling"
  - "coaching conversational agents"
  - "user simulation"
  - "LLM evaluation"
date: 2026-05-08
content_hash: fd229bedd1de429f
---

# Sleepless Nights, Sugary Days: Creating Synthetic Users with Health Conditions for Realistic Coaching Agent Interactions

**Conference**: ACL 2025  
**arXiv**: [2502.13135](https://arxiv.org/abs/2502.13135)  
**Code**: [Yes](https://anonymous.4open.science/r/sleepless_nights)  
**Area**: Other  
**Keywords**: Synthetic users, health condition modeling, coaching conversational agents, user simulation, LLM evaluation

## TL;DR

Proposes an end-to-end framework to generate synthetic users with health conditions (covering sleep and diabetes management) based on real demographic, health/lifestyle, and behavioral/psychological profile data. This framework is used to evaluate the interaction quality of health coaching agents, and is validated through human expert evaluation to significantly outperform generic synthetic users.

## Background & Motivation

Interactive health coaching agents require interaction with users to evaluate their effectiveness, but collecting and evaluating diverse, long-term human interactions is both expensive and time-consuming. LLMs-generated synthetic users offer the potential for automated evaluation, but prior methods exhibit key limitations:

**Lack of Grounding in Real Health Conditions**: Generic synthetic users cannot accurately reflect the needs and challenges of users under specific health conditions.

**Demographic Bias**: LLM training data is biased toward English-speaking cultures and highly active online populations, which does not represent the actual patient distribution.

**Lack of Contextualized Knowledge**: LLMs can reference phenomena like difficulty sleeping, but these do not substitute for contextualized knowledge rooted in lived experiences.

**Risk of Causal Implication**: Presenting LLMs with specific advice might unintentionally alter other implicit characteristics of the synthetic user.

Core Idea: Synthetic users should be generated based on real data—constructed from actual demographic, health, and behavioral/psychological profiles, rather than relying entirely on free-form LLM generation.

## Method

### Overall Architecture

Two-stage construction of synthetic users:

1. **Structured Data Generation**: Generate structured attributes based on real demographic, health/lifestyle, and behavioral/psychological data.
2. **Complete Profile Generation**: Generate complete user "vignettes" using LLMs based on the structured data.

Then, simulated interactions are conducted between the synthetic users and the coaching agent using either the Concordia system or direct LLM calls.

### Key Designs

1. **Attribute Grounding Based on Real Data**

    - Sleep Scenario: Uses the LifeSnaps public dataset (68 participants, including demographics, sleep data, Big Five personality, etc.)
    - Diabetes Scenario: Uses the PBHS longitudinal cohort (345 patients with Type 2 diabetes, containing detailed demographic, socioeconomic, and clinical data)
    - Design Motivation: Directly sample the distribution of real data to avoid LLM distributional biases.

2. **Multi-level User Modeling**

   For the sleep scenario:
    - Basic Attributes: Age, gender, BMI, sleep duration and efficiency, Big Five personality.
    - LLM-generated Sleep Profile: Primary sleep concerns, sleep goals, reasons for goals, barriers.
    - Optional Extensions: Challenges from the COM-B behavior model framework, rich backstory.

   For the diabetes scenario:
    - Sample barriers from 246 real challenges according to the COM-B model distribution.
    - Build vignettes based on patients' demographics, socioeconomic, and clinical data.
    - Generate communication styles (tone, verbosity, level of confidence).

3. **Interaction Simulation**

    - Instantiate synthetic users using the Concordia generative agent framework.
    - Concordia provides associative memory, chain-of-thought reasoning, and modular architecture.
    - The Sleep Agent employs a "Talker-Reasoner" dual-agent architecture (System 1 + System 2).
    - Uses Gemini 1.5 Pro as the underlying LLM.

4. **Multi-dimensional Evaluation Strategy**

    - Automated Evaluation: Compare the coaching agent's internal user model with the ground-truth user profile.
    - Expert Evaluation: Trained human evaluators blindly assess interaction quality.
    - Comparative Evaluation: Full synthetic users vs. demographic-only baseline users.

### Loss & Training

This is a framework-oriented work that does not involve model training. The core lies in the design of synthetic data generation and evaluation pipelines.

## Key Experimental Results

### Sleep Coaching Experiment (68 synthetic users, 10-turn interactions)

| Evaluation Dimension | Metric |
|----------|------|
| Accuracy of sleep concern identification | **89.7%** |
| Recall of barriers | 71.4% |
| Precision of barriers | 72.5% |
| Recall of sleep goals | 66.4% |
| Precision of sleep goals | 84.2% |

### Human Expert Evaluation (Sleep Scenario)

| Evaluation Item | Preference of Full User vs. Baseline | Inter-annotator Agreement |
|--------|---------------------|-------------|
| Overall Preference | Full User Wins Significantly | Fleiss' $\kappa = 0.67$ |
| p-value | $3.7 \times 10^{-12}$ | |
| 5/5 Perfect Agreement Rate | 64% | |
| $\ge 4/5$ Agreement Rate | 91% | |

### Diabetes Coaching Experiment (200 synthetic users)

| Evaluation Dimension | Expert Rating |
|----------|---------|
| User Consistency | **92%** |
| Fidelity of Barrier Representation | **100%** |

### Key Findings

1. Coaching agents can identify synthetic users' primary sleep concerns with 89.7% accuracy, rendering evidence that synthetic users indeed convey assigned health attributes effectively during interactions.
2. Synthetic users based on complete health/behavioral attributes significantly outperform baseline users based solely on demographics ($p < 10^{-12}$).
3. Inter-rater agreement is high ($\kappa = 0.67$), indicating that quality differences are distinct and easy to judge.
4. The framework is validated across two independently developed agents and scenarios, demonstrating its generalizability.

## Highlights & Insights

1. **End-to-End Framework Design**: A complete workflow covering real data sampling $\rightarrow$ attribute generation $\rightarrow$ vignette construction $\rightarrow$ interaction simulation $\rightarrow$ multi-dimensional evaluation.
2. **Crucial Role of Grounding in Real Data**: Experiments forcefully demonstrate that demographic information alone is far from sufficient; health conditions and behavioral profiles are critical to generating realistic synthetic users.
3. **Independent Validation in Two Scenarios**: The sleep and diabetes scenarios were developed independently by different teams, enhancing the reliability of the conclusions.
4. **Systematic Reflection on LLM Biases**: Explicitly identifies and mitigates multiple sources of bias when using LLMs as synthetic users.
5. **Privacy Preservation Support**: Synthetic users can generate novel individuals, reducing direct reliance on real patients' private data.

## Limitations & Future Work

1. Synthetic users may still lack the depth and nuance of real-life lived experiences.
2. Only the elicitation of goals and barriers was evaluated, without assessing the subsequent behavior change process.
3. Reliance on the Gemini family of models: different LLMs may yield varying qualities of synthetic users.
4. Performance declines when substituting with open-source models (e.g., Gemma 2-27B).
5. The effectiveness of long-term interactions (beyond 10 turns) has not been verified.

## Related Work & Insights

- AMIE (Tu et al., 2024): A conversational agent for medical diagnosis, but its synthetic patient design suffers from limitations such as demographic bias.
- Yu et al. (2024): A knowledge graph-based patient LLM, suitable for clinical but not health coaching scenarios.
- Castricato et al. (2024): Synthetic users based on US census statistics, but considering only demographics without health conditions.
- Concordia (Vezhnevets et al., 2023): The generative agent framework used in this work.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to systematically integrate real health data into synthetic user generation for coaching agent evaluation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Well-designed dual-scenario validation, automated + expert evaluation, and comparative experiments.
- **Writing Quality**: ⭐⭐⭐⭐ Clear description of the framework, comprehensive background review.
- **Value**: ⭐⭐⭐⭐ Provides a practical methodology for agent evaluation in the health AI domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CoachMe: Decoding Sport Elements with a Reference-Based Coaching Instruction Generation Model](coachme_sport_instruction.md)
- [\[ACL 2025\] CONFETTI: Conversational Function-Calling Evaluation Through Turn-Level Interactions](confetti_conversational_function-calling_evaluation_through_turn-level_interacti.md)
- [\[ACL 2025\] Using Shapley Interactions to Understand How Models Use Structure](using_shapley_interactions_to_understand_how_models_use_structure.md)
- [\[ACL 2025\] KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding](kodcode_a_diverse_challenging_and_verifiable_synthetic_dataset_for_coding.md)
- [\[ACL 2025\] LaTIM: Measuring Latent Token-to-Token Interactions in Mamba Models](latim_measuring_latent_token-to-token_interactions_in_mamba_models.md)

</div>

<!-- RELATED:END -->
