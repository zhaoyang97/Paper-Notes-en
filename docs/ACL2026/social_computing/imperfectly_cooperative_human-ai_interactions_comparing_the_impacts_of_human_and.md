---
title: >-
  [Paper Note] Imperfectly Cooperative Human-AI Interactions: Comparing the Impacts of Human and AI Attributes in Simulated and User Studies
description: >-
  [ACL 2026][Social Computing][Human-AI Interaction] Through a dual-framework experiment involving 2000 LLM simulations and a 290-person user study…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Human-AI Interaction"
  - "Imperfect Cooperation"
  - "Personality Traits"
  - "AI Transparency"
  - "Simulation vs. User Study"
date: 2026-05-08
content_hash: 5286e671adbc48d1
---

# Imperfectly Cooperative Human-AI Interactions: Comparing the Impacts of Human and AI Attributes in Simulated and User Studies

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.15607](https://arxiv.org/abs/2604.15607)  
**Code**: None  
**Area**: Human-AI Interaction / AI Safety  
**Keywords**: Human-AI Interaction, Imperfect Cooperation, Personality Traits, AI Transparency, Simulation vs. User Study

## TL;DR

Through a dual-framework experiment involving 2000 LLM simulations and a 290-person user study, this work compares the impacts of human personality traits and AI design attributes in imperfectly cooperative scenarios (recruitment negotiation, partially honest transactions). It finds that personality traits dominate in simulations, whereas AI transparency is the key driving factor in real-world user experiments.

## Background & Motivation

**Background**: Human-AI interaction research primarily focuses on fully cooperative scenarios where humans and AI pursue common goals. Significant research already exists on the impacts of AI transparency and individual user differences in these contexts.

**Limitations of Prior Work**: (1) Real-world AI deployments increasingly involve imperfectly cooperative scenarios (e.g., an AI recruitment manager whose goals partially conflict with a candidate, or AI customer service withholding information), yet research in this area is insufficient; (2) Human traits and AI attributes are typically studied in isolation, and their joint effects remain unexplored; (3) It is questionable whether LLM simulations can replace human experiments to validate conclusions.

**Key Challenge**: Simulation experiments allow for controlled variables but may not reflect real human behavior; human experiments are costly but more reliable. Do the conclusions from these two methods align?

**Goal**: To simultaneously investigate the joint effects of human personality and AI attributes in imperfectly cooperative scenarios and compare the differences between simulated and real-world user experiments.

**Key Insight**: Using the Sotopia-S4 platform to construct parallel simulation/user studies, manipulating extraversion/agreeableness (human) and transparency/adaptability/professionalism/warmth/Theory of Mind (AI), and comparing influence factors through causal discovery analysis.

**Core Idea**: In imperfectly cooperative scenarios, the impact of AI attributes (especially transparency) on real users is far greater than predicted by simulations, highlighting the necessity of human-in-the-loop validation.

## Method

### Overall Architecture

A two-phase experiment: (1) Simulation study: 5 scenarios $\times$ 5 AI interventions $\times$ 4 personality configurations $\times$ 10 repetitions = 2000 dialogues; (2) User study: 290 Prolific participants interacting with the same AI configurations, completing a personality test before the dialogues.

### Key Designs

1.  **Imperfectly Cooperative Scenario Design**:
    *   **Function**: To create an experimental environment where human and AI goals partially conflict.
    *   **Mechanism**: Recruitment negotiation (high/low stakes versions, where points for salary and start date are zero-sum or non-zero-sum) + AI-LieDar scenarios (where the AI is incentivized to withhold information for self-interest, public image, or emotional management).
    *   **Design Motivation**: To cover both explicit conflict (negotiation) and implicit conflict (information withholding) to better reflect real-world AI deployment scenarios.

2.  **AI Attribute Ablation Design**:
    *   **Function**: To quantify the causal effect of each AI attribute.
    *   **Mechanism**: The baseline features all 5 attributes at high levels, then sets one attribute to low in each trial—transparency (whether the thinking process is shown), warmth, professionalism, adaptability, and Theory of Mind (ToM). Causal discovery analysis (PC algorithm) is used to determine the influence paths rather than simple correlation.
    *   **Design Motivation**: A controlled variable approach ensures the independent effect of each attribute can be isolated.

3.  **Multi-dimensional Evaluation System**:
    *   **Function**: To comprehensively measure interaction outcomes.
    *   **Mechanism**: Includes outcome metrics (agreement reached, points, goal attainment), process metrics (interaction depth, linguistic fairness, communicative adaptability, transparency), relationship metrics (warmth, ToM, relational impact), and informational norm metrics (reliability, factual alignment).
    *   **Design Motivation**: To look beyond "task completion" and evaluate "interaction quality" and "relational impact."

### Loss & Training

Simulations were driven by GPT-4o (temperature 0.7), and user studies were conducted on the Prolific platform. Causal analysis was performed using the PC algorithm.

## Key Experimental Results

### Main Results

Ranking of causal influence factors (simplified):

| Dataset | Strongest Influence Factor | Note |
|-------|-----------|------|
| Simulation (Recruitment) | Agreeableness > Extraversion > AI Attributes | Personality-led |
| Simulation (LieDar) | Extraversion > Agreeableness > AI Attributes | Personality-led |
| User Study (Recruitment) | AI Transparency > Adaptability > Personality | AI-attribute-led |
| User Study (LieDar) | AI Transparency > Personality | AI-attribute-led |

### Ablation Study

| AI Attribute Ablation | Simulation Impact | User Study Impact |
|-----------|---------|-----------|
| Low Transparency | Slight | **Significant Negative** |
| Low Adaptability | Medium | Medium |
| Low Professionalism | Slight | Slight |
| Low Warmth | Slight | Slight |

### Key Findings

*   **Key discrepancy between simulations and humans**: Personality traits are the primary drivers in simulations, whereas AI attributes (especially transparency) are the critical factors in human experiments. LLM simulations may overestimate the impact of personality and underestimate user sensitivity to AI attributes.
*   Transparency (displaying the thinking process) is the most consistent positive factor in human experiments.
*   Scenario type (negotiation vs. information withholding) moderates the relative importance of various factors.

## Highlights & Insights

*   The **simulation-human comparison methodology** is highly valuable—it reveals systematic biases in LLM simulations and provides an important warning for future research using LLMs to simulate human behavior.
*   The **central role of AI transparency in conflict scenarios** has direct implications for AI design.
*   The **experimental framework for imperfectly cooperative scenarios** can be reused for other human-AI interaction studies.

## Limitations & Future Work

*   The user study scale (290 people) is limited, and participants were all native English speakers from the US.
*   Personality traits were treated as covariates rather than controlled variables in the user study.
*   Simulations were driven only by GPT-4o; different models might exhibit different biases.

## Related Work & Insights

*   **vs. Pure Simulation Studies (Park et al., 2024)**: This work identifies significant discrepancies through parallel human validation.
*   **vs. Fully Cooperative Scenario Research**: In imperfectly cooperative scenarios, the importance of AI attributes is magnified.

## Rating

*   Novelty: ⭐⭐⭐⭐ The combination of imperfectly cooperative scenarios and simulation/human comparison is novel.
*   Experimental Thoroughness: ⭐⭐⭐⭐ 2000 simulations + 290 human participants, with rigorous causal analysis.
*   Writing Quality: ⭐⭐⭐⭐ Detailed description of the experimental design.
*   Value: ⭐⭐⭐⭐⭐ Provides critical insights for both AI design and LLM simulation research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Persona-E2: A Human-Grounded Dataset for Personality-Shaped Emotional Responses to Textual Events](persona-e2_a_human-grounded_dataset_for_personality-shaped_emotional_responses_t.md)
- [\[NeurIPS 2025\] Policy-as-Prompt: Turning AI Governance Rules into Guardrails for AI Agents](../../NeurIPS2025/social_computing/policy-as-prompt_turning_ai_governance_rules_into_guardrails_for_ai_agents.md)
- [\[ACL 2026\] Reheat Nachos for Dinner? Evaluating AI Support for Cross-Cultural Communication of Neologisms](reheat_nachos_for_dinner_evaluating_ai_support_for_cross-cultural_communication_.md)
- [\[ICLR 2026\] Propaganda AI: An Analysis of Semantic Divergence in Large Language Models](../../ICLR2026/social_computing/propaganda_ai_an_analysis_of_semantic_divergence_in_large_language_models.md)
- [\[ICLR 2026\] Human or Machine? A Preliminary Turing Test for Speech-to-Speech Interaction](../../ICLR2026/social_computing/human_or_machine_a_preliminary_turing_test_for_speech-to-speech_interaction.md)

</div>

<!-- RELATED:END -->
