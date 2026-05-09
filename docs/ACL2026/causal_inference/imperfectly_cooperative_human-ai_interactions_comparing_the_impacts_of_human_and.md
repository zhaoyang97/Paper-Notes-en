---
title: >-
  [Paper Note] Imperfectly Cooperative Human-AI Interactions: Comparing the Impacts of Human and AI Attributes in Simulated and User Studies
description: >-
  [ACL 2026][Causal Inference][Human-AI Interaction] Through 2000 LLM simulations and a 290-person user study, discovers that personality traits dominate in simulations while AI transparency is the key driver in real user experiments for imperfectly cooperative scenarios.
tags:
  - ACL 2026
  - Causal Inference
  - Human-AI Interaction
  - Imperfect Cooperation
  - Personality Traits
  - AI Transparency
content_hash: e39f5d132dfdbd05
---

# Imperfectly Cooperative Human-AI Interactions: Comparing the Impacts of Human and AI Attributes in Simulated and User Studies

**Conference**: ACL 2026
**arXiv**: [2604.15607](https://arxiv.org/abs/2604.15607)
**Code**: N/A
**Area**: Human-AI Interaction / AI Safety
**Keywords**: Human-AI Interaction, Imperfect Cooperation, Personality Traits, AI Transparency, Simulation vs User Study

## TL;DR
Through 2000 LLM simulations and a 290-person user study in a dual-framework experiment, this paper compares the impacts of human personality traits and AI design attributes in imperfectly cooperative scenarios (hiring negotiation, partially honest trading), finding that personality traits dominate in simulations while AI transparency is the key driver in real user experiments.

## Method

### Key Designs

1. **Imperfectly Cooperative Scenario Design**: Hiring negotiations (high/low risk with zero-sum and non-zero-sum point allocations) + AI-LieDar scenarios (AI has incentives to conceal information).

2. **AI Attribute Ablation Design**: Baseline with all 5 attributes high, then each set to low individually — transparency, warmth, expertise, adaptability, theory of mind. Causal discovery analysis (not simple correlation).

3. **Multi-Dimensional Evaluation**: Outcome metrics (agreement, points), process metrics (interaction depth, verbal fairness), relationship metrics (warmth, theory of mind), and information norm metrics (credibility, factual alignment).

## Key Experimental Results

| Dataset | Strongest Factor |
|---------|-----------------|
| Simulation (Hiring) | Agreeableness > Extraversion > AI Attributes |
| User Study (Hiring) | **AI Transparency** > Adaptability > Personality |

## Highlights & Insights
- The simulation-real divergence methodology is valuable — reveals systematic biases of LLM simulation, providing important warnings for future LLM-as-human-proxy research
- AI transparency's central role in conflict scenarios provides direct guidance for AI design

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] CausalDetox: Causal Head Selection and Intervention for Language Model Detoxification](causaldetox_causal_head_selection_and_intervention_for_language_model_detoxifica.md)
- [\[ACL 2026\] Dialectic-Med: Mitigating Diagnostic Hallucinations via Counterfactual Adversarial Multi-Agent Debate](dialectic-med_mitigating_diagnostic_hallucinations_via_counterfactual_adversaria.md)
- [\[ACL 2026\] ClimateCause: Complex and Implicit Causal Structures in Climate Reports](climatecause_complex_and_implicit_causal_structures_in_climate_reports.md)
- [\[ACL 2026\] Better and Worse with Scale: How Contextual Entrainment Diverges with Model Size](better_and_worse_with_scale_how_contextual_entrainment_diverges_with_model_size.md)
- [\[ACL 2026\] Parallel Universes, Parallel Languages: A Comprehensive Study on LLM-based Multilingual Counterfactual Example Generation](parallel_universes_parallel_languages_a_comprehensive_study_on_llm-based_multili.md)

<!-- RELATED:END -->
