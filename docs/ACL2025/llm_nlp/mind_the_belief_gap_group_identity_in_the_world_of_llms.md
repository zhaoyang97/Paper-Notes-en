---
title: >-
  [Paper Note] Mind the (Belief) Gap: Group Identity in the World of LLMs
description: >-
  [ACL 2025][LLM (Other)][Belief congruence] By simulating Belief Congruence theory within a multi-agent LLM framework, this work reveals that LLMs exhibit a stronger belief congruence bias than humans, which increases misinformation propagation and impairs learning capability. The authors propose three mitigation strategies based on social psychology.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Belief congruence"
  - "multi-agent systems"
  - "social bias"
  - "misinformation propagation"
  - "group psychology"
date: 2026-05-08
content_hash: ee884b6dba94051a
---

# Mind the (Belief) Gap: Group Identity in the World of LLMs

**Conference**: ACL 2025  
**arXiv**: [2503.02016](https://arxiv.org/abs/2503.02016)  
**Code**: [GitHub](https://github.com/MichiganNLP/BeliefCongruenceSim)  
**Area**: LLM/NLP  
**Keywords**: Belief congruence, multi-agent systems, social bias, misinformation propagation, group psychology

## TL;DR

By simulating Belief Congruence theory within a multi-agent LLM framework, this work reveals that LLMs exhibit a stronger belief congruence bias than humans, which increases misinformation propagation and impairs learning capability. The authors propose three mitigation strategies based on social psychology.

## Background & Motivation

First proposed by Rokeach (1960), belief congruence theory states that individuals tend to align with others who hold similar beliefs, playing a critical role in shaping group interactions and social biases. As LLMs are increasingly deployed in multi-agent systems and social simulations, understanding whether they exhibit similar group psychological traits is essential.

However, existing research has rarely explored intergroup dynamics in LLMs. If LLMs exhibit belief congruence like humans, it could lead to echo-chamber effects in multi-agent interactions, amplifying misinformation and hindering learning. This paper proposes three research questions: (1) Do LLMs exhibit belief congruence? (2) Does this behavior affect downstream tasks? (3) Can their negative impacts be mitigated?

This is the first work to simulate belief congruence using LLM agents and examine its impact on downstream tasks.

## Method

### Overall Architecture

The study consists of three main parts: (1) belief congruence simulation experiments replicating classic psychological experiments; (2) downstream impact evaluations, including misinformation propagation and LLM learning; and (3) mitigation strategies based on social psychological theories. Three models are utilized: Llama-3-70B, Qwen-2.5-72B, and GPT-3.5.

### Key Designs

**1. Belief Congruence Simulation: Replicating the Rokeach & Mezei (1966) Experiment**

- **Function**: Validate whether LLMs display similar (or even stronger) belief congruence behavior to humans.
- **Mechanism**: In campus experiments, white LLM participants discuss topics with four confederates (2 white + 2 Black, one agreeing and one disagreeing in each group) and then select two to have coffee with. In field experiments, Black or white LLM participants discuss a psychiatric patient scenario with four confederates and select future colleagues. The order of participants is randomized in each round.
- **Design Motivation**: Utilize classic psychological paradigms to ensure a solid theoretical foundation, and directly compare the results with original human experimental data to quantify the strength of belief congruence in LLMs.

**2. Downstream Task Evaluation: Misinformation Propagation + LLM Learning**

- **Function**: Quantify the negative impact of belief congruence on downstream application scenarios.
- **Mechanism**: **Misinformation propagation**: Homogeneous or heterogeneous LLM groups with Democrat/Republican political personas discuss the truthfulness of news, and the echo-chamber effect is measured by changes in accuracy rates. **LLM learning**: A two-stage framework (learning stage + selection stage) from Marks et al. (2019) is adopted to test whether LLMs prefer politically aligned information sources even when those sources are less accurate.
- **Design Motivation**: Misinformation and impaired learning are the most critical negative impacts of belief congruence in human society; this work transfers them to the LLM agent setting to validate their presence.

**3. Three Mitigation Strategies**

- **Function**: Reduce the negative consequences of belief congruence.
- **Mechanism**: (1) **Contact Hypothesis**: Introduce heterogeneous groups consisting of both Democrat and Republican agents. (2) **Accuracy Nudge**: Prompt agents during interaction to verify information accuracy (analogous to self-reflection). (3) **Global Political Citizenship**: Adapt the Global Citizenship framework of Reysen & Katzarska-Miller to a political context, adding survey questions that emphasize cross-political inclusivity into the system prompt.
- **Design Motivation**: Each strategy targets a different level: the Contact Hypothesis alters group composition, Accuracy Nudge alters individual cognitive processes, and Global Citizenship alters the identity framework.

### Loss & Training

As an evaluation study, this paper does not involve model training. Evaluation metrics include belief congruence selection frequency, misinformation accuracy ($\text{correctness rate} = \sum_{i=1}^{N} \mathbb{I}(f(x_i) = y_i) / n$), source selection ratios, and changes in confidence behavior.

## Key Experimental Results

### Main Results: Misinformation Propagation

| Model | Initial Accuracy (Dem) | Final Accuracy (Dem) | Initial Accuracy (Rep) | Final Accuracy (Rep) |
|------|----------------|----------------|----------------|----------------|
| gpt-35 | ~0.65 | 0.598 | ~0.65 | 0.601 |
| llama-3 | ~0.50 | 0.444 | ~0.50 | 0.377 |
| qwen-2.5 | ~0.45 | 0.400 | ~0.45 | 0.390 |

### Ablation Study: Mitigation Strategy Effects

| Mitigation Strategy | GPT-3.5 Final Accuracy (Dem/Rep) | Llama-3 | Qwen-2.5 |
|---------|--------------------------|---------|----------|
| No Mitigation | 0.598 / 0.601 | 0.444 / 0.377 | 0.400 / 0.390 |
| Accuracy Nudge | 0.664 / 0.632 | 0.501 / 0.450 | 0.520 / 0.460 |
| Global Political Citizenship | **0.678 / 0.661** | **0.581 / 0.550** | **0.525 / 0.516** |
| Contact Hypothesis (Heterogeneous) | 0.674 | 0.544 | 0.493 |

### Key Findings

1. **Amplified Belief Congruence in LLMs**: All models exhibit stronger belief congruence than humans across both campus and field experiments (human selection rates range from 0.2 to 0.6, while all LLMs score $\ge 0.5$, with GPT-3.5 averaging 0.93).
2. **Homogeneous Groups Exacerbate Misinformation**: Post-interaction accuracy rates generally decrease, with Republican personas exhibiting weaker performance.
3. **Global Political Citizenship represents the most effective mitigation strategy for misinformation**: Leading to an accuracy improvement of up to 37%.
4. **Accuracy Nudges are most effective for learning tasks**: Reducing the propensity of agents to choose politically aligned but inaccurate sources ($\downarrow 11\%$).
5. **Belief Congruence is ubiquitous across all demographic dimensions**: The effects remain consistent when substituted with alternative dimensions such as age, gender, or minimal groups.

## Highlights & Insights

- First to systematically introduce the classic group psychology theory of Belief Congruence into LLM multi-agent systems and validate it.
- Rigorous experimental design, directly replicating classic psychological experiments and benchmarking against human data.
- Highly practical mitigation strategies, with the cross-domain adaptation of the Global Citizenship framework being particularly impressive.
- Reveals the risks of collective behavioral biases in LLM multi-agent systems, providing crucial insights for AI safety.

## Limitations & Future Work

- Simulations are based on a single paradigm (Rokeach 1960); further validation across more diverse psychological paradigms is needed.
- LLM simulation of belief systems might oversimplify complex human socio-cultural and emotional dynamics.
- Primarily situated in the US political context, leaving cross-cultural applicability to be validated.
- Only two downstream tasks are explored; belief congruence might yield positive impacts in specialized tasks such as mental health support.
- Future research can explore other group psychology theories such as Social Identity Theory and Realistic Group Conflict Theory.

## Related Work & Insights

- **Multi-agent LLM Interaction**: Previous work of social simulations by Park et al. (2023) and ChatEval have demonstrated the emergent behaviors of LLM agents, but systematic studies on group-level psychology are still lacking.
- **Insights**: LLM-internalized biases are not only reflected in individual outputs but are amplified and reinforced during multi-agent interactions. Designers of multi-agent systems should actively incorporate group diversity and self-reflection mechanisms.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First to systematically introduce belief congruence theory into the LLM domain, framing an important and interdisciplinary problem.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three models + diverse experimental setups + demographic ablations + three mitigation strategies.
- **Writing Quality**: ⭐⭐⭐⭐ Thorough introduction of psychological theories and clear experimental logic.
- **Value**: ⭐⭐⭐⭐⭐ Provides significant guidance for AI safety and multi-agent system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](zero-shot_belief_a_hard_problem_for_llms.md)
- [\[NeurIPS 2025\] Mind the Gap: Removing the Discretization Gap in Differentiable Logic Gate Networks](../../NeurIPS2025/llm_nlp/mind_the_gap_removing_the_discretization_gap_in_differentiable_logic_gate_networ.md)
- [\[ACL 2025\] Theory of Mind in Large Language Models: Assessment and Enhancement](theory_of_mind_llm.md)
- [\[ACL 2026\] Mind the Gap: How Elicitation Protocols Shape the Stated-Revealed Preference Gap in Language Models](../../ACL2026/llm_nlp/mind_the_gap_how_elicitation_protocols_shape_the_stated-revealed_preference_gap_.md)
- [\[ACL 2025\] LLM as Effective Streaming Processor: Bridging Streaming-Batch Mismatches with Group Position Encoding](llm_as_effective_streaming_processor_bridging_streaming-batch_mismatches_with_gr.md)

</div>

<!-- RELATED:END -->
