---
title: >-
  [Paper Note] Position: Theory of Mind Benchmarks are Broken for Large Language Models
description: >-
  [ICML 2025][LLM Evaluation][Theory of Mind] This position paper points out that most current LLM Theory of Mind (ToM) benchmarks only evaluate whether models can "predict others' behavior" (Literal ToM), but fail to test whether they can "take optimal responses based on that prediction" (Functional ToM). Consequently, they systematically overestimate models' adaptive capabilities in real interactions.
tags:
  - "ICML 2025"
  - "LLM Evaluation"
  - "Theory of Mind"
  - "Multi-Agent Interaction"
  - "Functional Adaptation"
  - "Benchmark Reliability"
date: 2026-05-08
content_hash: e11e27d4d9a14f70
---

# Position: Theory of Mind Benchmarks are Broken for Large Language Models

**Conference**: ICML 2025  
**arXiv**: [2412.19726](https://arxiv.org/abs/2412.19726)  
**Code**: No public repository (not provided in the paper)  
**Area**: LLM Evaluation  
**Keywords**: Theory of Mind, LLM Evaluation, Multi-Agent Interaction, Functional Adaptation, Benchmark Reliability

## TL;DR

This position paper points out that most current LLM Theory of Mind (ToM) benchmarks only evaluate whether models can "predict others' behavior" (Literal ToM), but fail to test whether they can "take optimal responses based on that prediction" (Functional ToM). Consequently, they systematically overestimate models' adaptive capabilities in real interactions.

## Background & Motivation

Over the past two years, numerous works have evaluated the ToM capabilities of LLMs using classic psychology questions (e.g., Sally-Anne) or question-answering tasks. These tasks generally report high accuracy, leading to the conclusion that "LLMs possess strong ToM."

The authors argue that a key assumption has been taken for granted here: for humans, predicting others' behavior and making decisions based on that prediction typically exhibit "procedural consistency." However, for current LLMs, this consistency does not hold.

The core counterexample presented in the paper is very straightforward: in a 100-round game of Rock-Paper-Scissors, the opponent always plays Rock. Many LLMs can quickly predict that "the opponent will play Rock in the next round" (very high Literal ToM). But their own actions remain close to uniform random, rather than choosing Paper in the long run to maximize payoffs.

Therefore, the authors propose that if evaluation only looks at prediction and ignores behavior adaptation, it mistakes "knowing what to say" for "knowing what to do." This is the fundamental reason why this paper refers to current benchmarks as "broken."

## Method

### Overall Architecture

Under the framework of Partially Observable Stochastic Games (POSG), the paper distinguishes and formalizes two categories of ToM metrics:

1. **Literal ToM**: The error in predicting other agents' behaviors.
2. **Functional ToM**: The regret of one's own strategy relative to the optimal response, given the opponent's strategy.

The evaluation workflow does not involve training new models; instead, it freezes the model weights and performs in-context adaptation solely through prompts and interaction history. This aligns with real-world deployment scenarios: most online LLM services do not continuously fine-tune weights for every individual user.

### Key Designs 1: Literal ToM (Definition 2.1)

Literal ToM is defined as the distance between the opponent's actual sequence of actions and the model's predicted sequence of actions.

Intuitive understanding: if the model can always guess what the opponent will do next, the Literal ToM score is high. However, this metric inherently does not involve "how the model itself should act in the next step." Thus, it can only answer "whether it understands others" but not "whether it makes the correct decisions for itself."

### Key Designs 2: Functional ToM (Definition 2.2)

Functional ToM is measured using the $T$-step regret:

$$\Delta_{\text{Functional}} = \sum_{t=1}^{T}(r_t^{i^*}-r_t^{i})$$

where:
$r_t^{i^*}$ is the reward obtainable at step $t$ by the optimal response to the current opponent strategy, and $r_t^{i}$ is the actual strategy reward.

Intuitive understanding: if the model truly "understands and exploits" the patterns of the opponent's behavior, the regret should approach 0. The larger the regret, the weaker the ToM in terms of action execution.

### Key Designs 3: Bridging Metric $\Delta_{\text{ToM}}$

The authors also construct an upper bound analysis for "rational action following predictions": first predicting the opponent's action using Literal ToM, and then assuming the decision-maker always rationally maximizes payoff, yielding the corresponding regret $\Delta_{\text{ToM}}$.

If it occurs that:

$$\Delta_{\text{ToM}} \ll \Delta_{\text{Functional}}$$

it indicates that the issue is not "failing to predict the opponent accurately," but "failing to translate predictions into actions." This is precisely the phenomenon repeatedly observed in the paper.

### Tasks and Game Settings

The paper utilizes three types of classic repeated games covering different incentive structures:

1. **RPS (Rock-Paper-Scissors)**: Competitive.
2. **IBS (Iterated Battle of the Sexes)**: Coordination/Cooperative.
3. **IPD (Iterated Prisoner's Dilemma)**: Mixed incentive.

Opponent strategies include:

1. **Single Action** (constant action).
2. **Tit-for-Tat** (a more dynamic responsive strategy, see the appendix for extended analysis).

The interaction length is set to exactly 100 steps, which is significantly longer than many prior works, aiming to observe whether behaviors truly converge toward adaptation.

### Prompt Families and Comparison Dimensions

The paper systematically compares multiple prompting strategies:

1. **Generic**: LM, QA, CoT.
2. **In-context RL style**: Plans+Insights, Reflexion.
3. **Explicit ToM**: Social QA, Social LM (predicting the opponent's action first, then making a decision).
4. **Long-context diagnostics**: Oracle, Oracle+Max, No History, No Payoffs, S2A, CoT 3-shot.

The core question remains identical: even when provided with more "modes of thinking" or stronger "prompt structures," can the model stably translate predictions into payoff-optimal actions?

## Key Experimental Results

### Main Results Table (RPS, Single Action, corresponding to Table 1 in the paper)

| Model | Functional Regret $\Delta_{\text{Functional}}/T$ | Rationalized Regret $\Delta_{\text{ToM}}/T$ | ToM Accuracy |
|------|---:|---:|---:|
| Tabular (RMax) | 0.083 ± 0.004 | 0.039 ± 0.006 | 97.4% ± 0.4 |
| LLaMA-2 70B | 0.971 ± 0.067 | 0.048 ± 0.003 | 96.8% ± 0.2 |
| LLaMA-2 70B Chat | 0.857 ± 0.142 | 0.119 ± 0.006 | 92.1% ± 0.4 |
| LLaMA-2 13B | 1.015 ± 0.044 | 0.049 ± 0.003 | 96.7% ± 0.2 |
| Falcon 40B | 0.973 ± 0.040 | 0.056 ± 0.005 | 96.2% ± 0.3 |
| Mixtral 8x7B Instruct | 0.542 ± 0.070 | 0.050 ± 0.005 | 96.7% ± 0.3 |

**Interpretation**: Many models achieve near 97% ToM accuracy, but their functional regret is close to 1.0 (almost the worst possible in RPS). This directly validates that "high Literal ToM does not equal high Functional ToM."

### Prompting Strategy Analysis Table (Mixtral 8x7B, extracted from Table 2)

| Prompt | Game | $\Delta_{\text{Functional}}/T$ | $\Delta_{\text{ToM}}/T$ | ToM% |
|------|------|---:|---:|---:|
| LM | RPS | 0.542 | 0.050 | 96.7 |
| Social LM | RPS | 0.643 | 0.119 | 92.0 |
| LM | IBS | 2.055 | 0.216 | 97.3 |
| Social LM | IBS | 2.082 | 0.182 | 97.4 |
| LM | IPD | 0.949 | 0.098 | 97.8 |
| Social LM | IPD | 1.098 | 0.110 | 97.4 |

**Interpretation**: Although Social Prompting is often helpful, it does not cure the "prediction-action disconnect." Even when "opponent prediction" is explicitly incorporated into the decision pipeline, the functional performance still lags far behind simple tabular strategies.

### Long-Context Ablation Table (LLaMA-3 70B, extracted from Table 3)

| Configuration | RPS | IBS | IPD |
|------|---:|---:|---:|
| Tabular | 0.083 | 0.211 | 0.086 |
| QA | 0.444 | 1.391 | 0.996 |
| CoT | 0.213 | 2.475 | 0.892 |
| CoT + 3-shot | 0.121 | 0.526 | 2.773 |
| Social QA | 0.256 | 1.613 | 0.550 |
| Oracle | 0.238 | 1.343 | 0.839 |
| Oracle + Max | 0.153 | 0.785 | 0.767 |
| Oracle + Max - History | 1.275 | 2.060 | 0.206 |
| Oracle + Max - Payoffs | 0.103 | 0.883 | 0.241 |

**Interpretation**:
1. Even when directly informed of the opponent's actions (Oracle), performance is still significantly worse than Tabular.
2. Removing history significantly deteriorates performance in RPS/IBS, indicating that historical information is crucial for adaptation.
3. In some tasks, removing payoffs unexpectedly improves performance, suggesting that joint reasoning within long contexts might introduce interference for the model.

### Supplementary Analysis: Reasoning Model DeepSeek-R1 Distill 32B (Table 4)

| Game-Opponent | Tabular Functional Regret | DeepSeek-R1 Functional Regret | DeepSeek-R1 ToM% |
|------|---:|---:|---:|
| RPS - Single Action | 0.083 | 0.074 | 63.7 |
| RPS - Tit-for-Tat | 0.224 | 0.906 | 38.6 |
| IBS - Single Action | 0.211 | 0.126 | 97.2 |
| IBS - Tit-for-Tat | 0.468 | 0.045 | 94.5 |
| IPD - Single Action | 0.086 | 0.121 | 83.1 |
| IPD - Tit-for-Tat | 0.248 | 4.789 | 88.9 |

**Interpretation**: The reasoning model exhibits an anomalous combination of "strong functional performance but not always strong literal prediction," further proving that these two capabilities do not have a unidirectional entailment relationship.

## Highlights & Insights

1. The most significant conceptual contribution is upgrading ToM from a "predictive capability" to an "actionable capability."
2. The paper presents a powerful refutation of existing evaluation paradigms using a minimal counterexample (a constant-action opponent in RPS).
3. Introducing $\Delta_{\text{ToM}}$ as a diagnostic bridge is highly practical, helping to distinguish whether the failure lies in "incorrect prediction" or "faulty decision-making."
4. It has direct implications for practical deployment: if a model is to be granted action autonomy, its assessment must focus on Functional ToM rather than mere question-answering scores.
5. It aligns clearly with the predict-then-optimize literature: prediction error and decision-making error are not equivalent objectives.

## Limitations & Future Work

1. The tasks are still dominated by matrix games with limited environmental complexity; extrapolation to real-world collaborative systems requires further validation.
2. The primary evaluation targets are open-source models; similar evaluations on closed-source frontier models remain insufficient.
3. The focus of the paper is on "critiquing evaluation frameworks" rather than proposing a new training algorithm that can scale up Functional ToM.
4. Although prompt and contextual factors are analyzed, the mechanistic explanations regarding internal neural representations (i.e., why predictions fail to translate to behavior) remain weak.
5. Future research could explore the combination of "structured memory + short planner + action-value head" to mitigate decision-making degradation caused by long contexts.

## Related Work & Insights

### Relation to Human-like ToM Benchmarks

Works by Bubeck, Kosinski, Street, Strachan, etc., emphasize signs of ToM in question-answering tasks. This paper points out that these tasks are mostly static predictions and fail to reflect interactive adaptation.

### Relation to Multi-Agent RL & Game Theory

Lowe et al., Akata et al., etc., have focused on joint behavioral modeling. The point of divergence in this paper is explicitly decoupling "predicting others" from "acting based on predictions," unifying them under a regret perspective.

### Relation to Mutual ToM in HCI

The paper echoes the direction of "mutual modeling to improve collaborative outcomes." That is, the value of ToM lies not in the mentalistic labeling itself, but in whether it can improve joint task performance.

### Insights for My Own Research

If designing future LLM agent benchmarks, three types of metrics should be included by default:

1. Opponent modeling accuracy (Literal).
2. Payoff-oriented online adaptation (Functional).
3. Prediction-decision coupling efficiency (which can be measured by the difference between $\Delta_{\text{ToM}}$ and $\Delta_{\text{Functional}}$).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐  
  *Reason*: Upgrades ToM evaluation from a single predictive dimension to a dual "prediction + action" framework, with clear conceptual innovation and broad impact.

- **Experimental Thoroughness**: ⭐⭐⭐⭐☆  
  *Reason*: Systems comparison across multiple models, games, and prompts is highly systematic, though coverage of complex real-world tasks and closed-source models could be further enhanced.

- **Writing Quality**: ⭐⭐⭐⭐⭐  
  *Reason*: Starting with counterexamples, leading into formal definitions, and concluding with empirical diagnostics, the narrative arc is complete and arguments are well-structured.

- **Value**: ⭐⭐⭐⭐⭐  
  *Reason*: Provides actionable evaluation principles for all scenarios involving LLM decision-making, directly influencing the design of benchmarks and deployment thresholds.

## Overall Judgment

The most valuable aspect of this paper is not "yet another benchmark result," but rather correcting the evaluation objective itself. If a system is to represent a user or collaborate with a user in multi-round interactions, having a high Literal ToM score alone is far from sufficient. If future ToM benchmarks do not explicitly incorporate the Functional ToM dimension, they will continue to overestimate the true social interactive capabilities of LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Dynamic Theory of Mind: Evaluating LLM Adaptation to Temporal Evolution of Human States](../../ACL2025/llm_evaluation/towards_dynamic_theory_of_mind_evaluating_llm_adaptation_to_temporal_evolution_o.md)
- [\[ICML 2025\] Correlated Errors in Large Language Models](correlated_errors_in_large_language_models.md)
- [\[AAAI 2026\] Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory](../../AAAI2026/llm_evaluation/lost_in_benchmarks_rethinking_large_language_model_benchmarking_with_item_respon.md)
- [\[ACL 2025\] PapersPlease: A Benchmark for Evaluating Motivational Values of Large Language Models Based on ERG Theory](../../ACL2025/llm_evaluation/papersplease_a_benchmark_for_evaluating_motivational_values_of_large_language_mo.md)
- [\[ICML 2025\] G-Sim: Generative Simulations with Large Language Models and Gradient-Free Calibration](g-sim_generative_simulations_with_large_language_models_and_gradient-free_calibr.md)

</div>

<!-- RELATED:END -->
