---
title: >-
  [Paper Note] PoliticsBench: Benchmarking Political Values in Large Language Models with Multi-Stage Roleplay
description: >-
  [ICML 2026][LLM Evaluation][Political Bias] PoliticsBench is a novel benchmark based on **multi-stage roleplay**—evaluating the expression of political values in LLMs through 20 political scenarios and 4 stages of intera…
tags:
  - "ICML 2026"
  - "LLM Evaluation"
  - "Political Bias"
  - "Values"
  - "Multi-turn Dialogue"
  - "Roleplay"
date: 2026-05-08
content_hash: 34c32d69ce11671b
---

# PoliticsBench: Benchmarking Political Values in Large Language Models with Multi-Stage Roleplay

**Conference**: ICML 2026  
**arXiv**: [2603.23841](https://arxiv.org/abs/2603.23841)  
**Code**: To be confirmed  
**Area**: Social Computing / LLM Value Alignment / Bias Evaluation  
**Keywords**: Political Bias, LLM Evaluation, Values, Multi-turn Dialogue, Roleplay

## TL;DR
PoliticsBench is a novel benchmark based on **multi-stage roleplay**—evaluating the expression of political values in LLMs through 20 political scenarios and 4 stages of interaction. Findings show that 7 mainstream LLMs lean left (19-39 points), while only Grok leans right (-22.7) but exhibits the highest volatility; **situational prompts stimulate value dimensions more effectively than direct questioning** (+0.48 in feature activation, +1.39 in commitment).

## Background & Motivation

**Background**: LLMs are increasingly used as information sources and decision-support tools, yet their potential political biases may affect decision-making fairness. Existing LLM social bias benchmarks mainly focus on demographic stereotypes, and evaluations of political bias often remain at a coarse-grained level (left/right tilt), ignoring the specific values that drive political reasoning.

**Limitations of Prior Work**:
- Existing political evaluation benchmarks use single-step/isolated Q&A pairs with low information density.
- System prompts of closed-source models prevent direct answers to political questions.
- Evaluation dimensions are too coarse (binary left/right classification) to characterize the specific value dimensions of models.

**Key Challenge**: There is a conflict between the need for fine-grained evaluation of political values and the security alignment mechanisms of models that prevent direct political questioning.

**Goal**: Design a high-fidelity benchmark that can bypass the limitations of safety alignment and evaluate the expression of LLM political values across multiple dimensions ($\geq 3$ dimensions).

**Key Insight**: Drawing inspiration from EQ-Bench (emotional intelligence evaluation) and ethical benchmarks, the study utilizes multi-stage roleplaying with gradually increasing pressure to force models out of superficial neutrality and reveal their underlying value systems.

**Core Idea**: Instead of asking "What is your political stance?", ask "What are your trade-offs in this political dilemma?"—inducing deep-seated values under adversarial pressure through 4-stage roleplay across 20 real-world political scenarios.

## Method

### Overall Architecture
A three-layer evaluation—(1) Scenario Design Layer: 20 roleplay scenarios based on actual political topics (unionization, free healthcare, gender policy); (2) Interaction Stage Layer: 4 progressive stages + 1 reflection stage for each scenario, where the model outputs "thought" and "response" at each stage; (3) Scoring Layer: Three judge LLMs with balanced political spectra (Grok as right-leaning, GPT-4.1-mini as left-leaning, Claude-3.7-Sonnet as neutral) score each response across 10 political value dimensions and commitment levels.

### Key Designs

1.  **Four-Stage Progressive Scenarios**:
    - **Function**: Gradually applies pressure to focus on different value conflicts in each stage, forcing the model to expose its latent value system.
    - **Mechanism**: Stage 1 (Initial Conflict) $\rightarrow$ Stage 2 (Conflicting Loyalties, where the model must weigh two conflicting values—**this is the key stimulation point**) $\rightarrow$ Stage 3 (External Pressure, introducing urgent deadlines where the model states "non-negotiable bottom lines") $\rightarrow$ Stage 4 (Resolution and Cost, where the model reflects on "what was sacrificed") $\rightarrow$ Bonus (Self-reflection).
    - **Design Motivation**: Similar to behavioral performance testing under pressure in psychology; gradually increasing pressure elevates the model from "expressing an opinion" to a behavioral commitment of "accepting costs for a stance."

2.  **10-Dimensional Balanced Political Value System**:
    - **Function**: Uses specific value dimensions that drive political reasoning instead of binary left/right classification.
    - **Mechanism**: 5 left-leaning dimensions (Progressivism, Egalitarianism, Openness/Inclusion, Collective Responsibility, Pragmatism) + 5 right-leaning dimensions (Traditionalism, Authority/Deference, Risk Aversion, Personal Responsibility, Moral Certainty). Each value is scored 0-20, standardized to $[-10, 10]$, weighted by $w_i \in \{-1.125, -0.875, \ldots, +1.125\}$, and averaged to a total alignment score of $[-100, 100]$.
    - **Design Motivation**: Avoids anthropomorphizing the model while accurately characterizing value preferences reflected in the human language patterns the model has absorbed.

3.  **Three-Judge System + Chain-of-Thought**:
    - **Function**: Prevents the political bias of a single model from monopolizing the evaluation results.
    - **Mechanism**: Three judges with different political leanings score independently, and each judge must provide Chain-of-Thought (CoT) reasoning. Final reports use the average score of the three judges, with pairwise quadratic weighted Cohen's $\kappa = 0.84-0.91$ to measure consistency.
    - **Design Motivation**: Avoids single-judge bias; high $\kappa$ indicates a clear scoring signal. The conflict of Claude acting as both subject and judge is partially mitigated through majority voting.

## Key Experimental Results

### Main Results: Comparison of Political Leanings

| Model | Average Score | Std Dev | Statistical Significance |
| :--- | :--- | :--- | :--- |
| Claude | 24.79 | 12.98 | $\checkmark$ ($p < 0.0001$) |
| Deepseek | 37.32 | 25.38 | $\checkmark$ |
| Gemini | 28.43 | 15.82 | $\checkmark$ |
| GPT-5.4-mini | 29.11 | 8.13 | $\checkmark$ |
| **Grok** | **-7.81** | **30.83** | $\times$ ($p = 0.27$) |
| Llama | 38.64 | 19.84 | $\checkmark$ |
| Qwen Base | 25.71 | 8.22 | $\checkmark$ |
| Qwen-IT | 26.10 | 17.02 | $\checkmark$ |

Seven models exhibit a left-leaning tendency (19-39), whereas only Grok leans right (-22.7) but with the largest standard deviation (30.83, nearly 4 times that of other models).

### Ablation Study

| Configuration | Activated Features | Commitment | Description |
| :--- | :--- | :--- | :--- |
| Baseline (Direct Questioning) | 4.42 | 3.08 | Model leans toward superficial neutrality |
| Stage 1 | — | +0.29 | Initial reaction |
| Stage 2 (Conflicting Loyalty) | **+0.48** | +1.39 | **Peak activation** |
| Stage 3 (External Pressure) | +0.41 | **+1.67** | **Peak commitment** |
| Stage 4 (Cost) | +0.23 | +1.28 | Commitment slightly drops as stages progress |
| Average Across Stages | 4.90 | 4.47 | Significant overall improvement |

### Key Findings
- Stage 2 activates the most features (5.15 vs baseline 4.42) when forcing trade-offs—multi-value conflicts induce more behavior than single questions.
- Stage 3 shows the highest commitment under external pressure (4.75/5)—models are most likely to take a clear side under an ultimatum.
- The average change in political score across 4 stages is only 3.63 points (1.8% of the 200-point range)—core values remain relatively stable.

## Highlights & Insights
- **Clever Multi-Stage Progressive Design**: Gradually applying pressure through 4 stages focusing on different value conflicts can be repurposed for other evaluation scenarios (ethical decision-making, risk preference).
- **"Thought + Response" Binary Separation**: Unlike other benchmarks, PoliticsBench requires the model to output both a "thought" (internal reasoning) and a "response" (external action) at each stage—enabling observation of both reasoning and stance commitment.
- **Conversion of Value Dimensions vs. Political Labels**: Avoids "left/right" labels for LLMs by decomposing them into 10 specific value dimensions—accurate characterization without anthropomorphizing.
- **"Scenarios Stimulate Values Better than Direct Questions"**: Situational immersion effectively pushes the model from "expressing an opinion" to a behavioral commitment of "paying a price for a stance."

## Limitations & Future Work
- PoliticsBench evaluates "political value expression in constrained interactions" rather than "fixed internal beliefs"—scenario intensity is limited and cannot distinguish between the model's inherent bias and a roleplaying persona.
- Robustness to paraphrase decreases: Models become more sensitive to phrasing changes in later stages (difference increases by 1.1 points).
- Conflict of interest exists as Claude is both an evaluated subject and a judge.
- Improvements: Symmetry testing (pairing each scenario with its opposite); reversing scoring scales; separating model values from character values.

## Related Work & Insights
- **vs. MIT Truth-Political Bias** (Single-step direct questions): Single-step information density is low; multi-stage scenarios stimulate 35.3% higher commitment.
- **vs. PoliTune** (Textbook-style questions): Direct questions activate at most 4.42 value dimensions, whereas implicit scenarios reach 4.90.
- **vs. EQ-Bench** (Emotional Intelligence benchmark): Adapts the multi-stage roleplay framework of EQ-Bench to the political domain; unlike EQ-Bench, this work necessitates balancing three judges with different political stances to avoid single bias.

## Rating
- **Novelty**: ⭐⭐⭐⭐  The idea of evaluating political values through multi-stage scenarios is unprecedented, though adapted from EQ-Bench.
- **Experimental Thoroughness**: ⭐⭐⭐⭐  8 models $\times$ 20 scenarios $\times$ 4 stages $\times$ three judges + paraphrasing + ablation; however, LLM-as-a-judge remains controversial.
- **Writing Quality**: ⭐⭐⭐⭐⭐  Clear logic, sufficient motivation, detailed tabular data, and honest discussion of limitations.
- **Value**: ⭐⭐⭐⭐  Fills a fine-grained gap in LLM political value evaluation; real-world value depends on whether "scenario-induced values" truly represent inherent model bias.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Benchmarking LLMs for Political Science: A United Nations Perspective](../../AAAI2026/llm_evaluation/benchmarking_llms_for_political_science_a_united_nations_perspective.md)
- [\[ICML 2026\] Investigating Advanced Reasoning of Large Language Models via Black-Box Environment Interaction](investigating_advanced_reasoning_of_large_language_models_via_black-box_environm.md)
- [\[ACL 2026\] E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task](../../ACL2026/llm_evaluation/e2edev_benchmarking_large_language_models_in_end-to-end_software_development_tas.md)
- [\[ACL 2026\] Modeling Multi-Dimensional Cognitive States in Large Language Models under Cognitive Crowding](../../ACL2026/llm_evaluation/modeling_multi-dimensional_cognitive_states_in_large_language_models_under_cogni.md)
- [\[ACL 2026\] PolitNuggets: Benchmarking Agentic Discovery of Long-Tail Political Facts](../../ACL2026/llm_evaluation/politnuggets_benchmarking_agentic_discovery_of_long-tail_political_facts.md)

</div>

<!-- RELATED:END -->
