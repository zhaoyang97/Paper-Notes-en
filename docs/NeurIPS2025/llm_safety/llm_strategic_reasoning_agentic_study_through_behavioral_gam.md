---
title: >-
  [Paper Note] LLM Strategic Reasoning: Agentic Study through Behavioral Game Theory
description: >-
  [NeurIPS 2025][LLM Safety][Strategic Reasoning] This paper proposes an LLM strategic reasoning evaluation framework grounded in behavioral game theory. It employs Truncated Quantal Response Equilibrium (TQRE) to quantify reasoning depth τ, evaluates 22 state-of-the-art models across 13 matrix games, and reveals differences in reasoning styles as well as biases induced by demographic personas.
tags:
  - NeurIPS 2025
  - LLM Safety
  - Strategic Reasoning
  - Behavioral Game Theory
  - TQRE
  - Reasoning Depth
  - Demographic Bias
date: 2026-05-08
content_hash: bcaa8542f137bcb5
---

# LLM Strategic Reasoning: Agentic Study through Behavioral Game Theory

**Conference**: NeurIPS 2025
**arXiv**: [2502.20432](https://arxiv.org/abs/2502.20432)
**Code**: None
**Area**: AI Safety / LLM Evaluation
**Keywords**: Strategic Reasoning, Behavioral Game Theory, TQRE, Reasoning Depth, Demographic Bias

## TL;DR

This paper proposes an LLM strategic reasoning evaluation framework grounded in behavioral game theory. It employs Truncated Quantal Response Equilibrium (TQRE) to quantify reasoning depth τ, evaluates 22 state-of-the-art models across 13 matrix games, and reveals differences in reasoning styles as well as biases induced by demographic personas.

## Background & Motivation

LLMs are increasingly deployed in interactive scenarios requiring strategic decision-making (e.g., procurement negotiation, advertising auctions), yet existing evaluations suffer from significant limitations:

1. **Over-reliance on Nash Equilibrium (NE)**: Most studies merely check whether LLMs converge to NE, but NE assumes perfect rationality, which is ill-suited for probabilistic LLMs.
2. **Binary evaluation**: Simply judging "reached/not reached NE" fails to quantify the depth of reasoning capability.
3. **Neglect of mechanisms**: No investigation into why LLMs deviate from optimal strategies.
4. **Circular reasoning**: Using a framework that assumes perfect rationality to test whether an agent is rational is logically flawed.

Core Problem: How can one go beyond NE and use cognitive science tools to quantify the strategic reasoning depth, style, and latent biases of LLMs?

## Method

### Overall Architecture

1. Design a library of 13 matrix games spanning 7 game categories.
2. Collect choice frequencies from 30 independent trials per LLM per game.
3. Fit choice distributions using the TQRE model to estimate reasoning depth τ and decision precision γ.
4. Analyze reasoning chains to reveal reasoning styles.
5. Inject demographic personas to study bias.

### Key Designs

1. **TQRE Evaluation Framework**:
    - Function: Quantifies the strategic reasoning depth of LLMs using the Truncated Quantal Response Equilibrium model.
    - Mechanism: Assumes that agents' reasoning levels follow $k \sim \text{Poisson}(\tau)$; a level-$k$ agent computes expected utilities based on beliefs over the mixed strategies of levels 0 through $k{-}1$, and translates utilities into probabilistic behavior via a logit choice rule.
    - Design Motivation: (1) The parameter τ of the Poisson distribution quantifies reasoning depth, replacing the binary judgment of NE; (2) the logit choice rule captures "near-optimal but noisy" behavior; (3) the hierarchical belief model reflects recursive reasoning of the form "I think my opponent thinks I will…"
    - Estimation: Maximum likelihood fitting $\max_{\tau,\gamma} \sum_{i,j} c_{ij} \ln p_{ij}(\tau,\gamma)$

2. **Reasoning Chain Analysis and Demographic Experiments**:
    - Function: Analyzes reasoning chains of top models to reveal distinct reasoning styles; injects gender/age/ethnicity/sexual orientation personas to study bias.
    - Mechanism: Classifies reasoning chains into styles such as maximin (worst-case optimization), belief-based (iterative belief reasoning), and mixed (hybrid strategies).
    - Design Motivation: Reasoning chain analysis explains "why different models perform differently across different games"; persona experiments examine whether "technical capability is equivalent to fairness."
    - Key Finding: Longer reasoning chains do not necessarily yield better decisions.

### Loss & Training

This paper presents an evaluation framework rather than a training method. Core mathematical tools:
- Level-$k$ choice probability: $p_{ij}^{(k)} = \frac{\exp(\lambda_k U_{ij}^{(k)})}{\sum_a \exp(\lambda_k U_{ia}^{(k)})}$, where $\lambda_k = \gamma \cdot k$
- Overall choice probability: $p_{ij} = \sum_k f_k \cdot p_{ij}^{(k)}$, $f_k = \frac{\tau^k e^{-\tau}}{k!}$
- Baseline reference: log-likelihood under uniform random selection $\text{MLL}_{chance} = -\ln(mn)$

## Key Experimental Results

### Main Results (Table)

Reasoning depth τ by model (selected):

| Model | Competitive (BL) | Cooperative (BL) | Mixed-Motive (BL) | Bayesian | Signaling |
|-------|-----------------|-----------------|------------------|----------|-----------|
| GPT-o3-mini | 1.31 | 3.55 | 3.35 | 4.23 | 3.08 |
| GPT-o1 | 4.74 | 2.80 | 0.14 | 4.23 | 3.98 |
| DeepSeek-R1 | - | - | - | - | - |
| GPT-4o | 1.54 | 0.60 | 1.67 | 1.97 | 3.59 |
| Gemma-V2-27B | 0.32 | 0.18 | 0.98 | 1.83 | 3.07 |

GPT-o3-mini and GPT-o1 lead across most games, though the dominant model varies by game type.

### Ablation Study

- Reasoning chain length vs. decision quality: Longer chains inconsistently improve decisions and sometimes introduce overthinking.
- Model size vs. reasoning depth: A nonlinear relationship; smaller R1-distilled models can outperform larger ones.
- Complete vs. incomplete information: Performance gaps between models widen under incomplete information games.

### Key Findings

- **Reasoning style differences**: GPT-o1 favors maximin (conservative); DeepSeek-R1 favors belief-based (iterative reasoning); GPT-o3-mini balances both.
- **Demographic bias**:
    - GPT-4o and Claude-3-Opus show improved reasoning under a female persona.
    - Gemini 2.0 exhibits significant reasoning degradation under minority sexuality personas.
    - DeepSeek-R1 produces inconsistent results under certain ethnic personas.
- **Model size is not determinative**: Smaller models can match or surpass larger models on specific games.
- **Longer reasoning chains ≠ better decisions**: Self-interference during reasoning can lead to suboptimal choices.

## Highlights & Insights

- **Evaluation paradigm upgrade**: Shifts from "whether NE is reached" to "how deep is the reasoning," providing a continuous and cognitively grounded assessment.
- **Excellent interdisciplinary integration**: Introduces the TQRE model from behavioral game theory into LLM evaluation, opening a novel perspective.
- **Reasoning style analysis is informative**: Explains why the same model performs markedly differently across game types.
- **Warning significance of bias findings**: Strong reasoning capability does not imply fairness; persona-induced biases persist even in advanced models.

## Limitations & Future Work

- Only one-shot games are evaluated; repeated games and dynamic strategy adaptation are not addressed.
- Coverage of 13 games is limited; more complex forms such as auctions and bargaining are not included.
- The Poisson distribution assumption over reasoning levels in TQRE may not precisely characterize LLM behavior.
- The prompt design in demographic experiments may introduce additional confounding factors.
- Generalizing results from matrix games to real-world decision-making scenarios requires further validation.

## Related Work & Insights

- Relationship to the Cognitive Hierarchy (CH) model: TQRE combines the hierarchical reasoning of CH with the stochastic choice of QRE.
- Distinction from robustness evaluations such as PromptBench: This work focuses on strategic reasoning capability rather than general performance.
- Direct implications for LLM-as-agent deployment: Strong reasoning capability may be accompanied by latent biases.

## Rating

⭐⭐⭐⭐ — The evaluation framework is cleverly designed, the interdisciplinary integration is outstanding, and the demographic bias findings carry significant safety implications.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] When AI Democratizes Exploitation: LLM-Assisted Strategic Manipulation of Fair Division Algorithms](when_ai_democratizes_exploitation_llm-assisted_strategic_manipulation_of_fair_di.md)
- [\[NeurIPS 2025\] TRAP: Targeted Redirecting of Agentic Preferences](trap_targeted_redirecting_of_agentic_preferences.md)
- [\[NeurIPS 2025\] Probabilistic Reasoning with LLMs for K-Anonymity Estimation](probabilistic_reasoning_with_llms_for_k-anonymity_estimation.md)
- [\[NeurIPS 2025\] Contextual Integrity in LLMs via Reasoning and Reinforcement Learning](contextual_integrity_in_llms_via_reasoning_and_reinforcement_learning.md)
- [\[ICLR 2026\] From Static Benchmarks to Dynamic Protocol: Agent-Centric Text Anomaly Detection for Evaluating LLM Reasoning](../../ICLR2026/llm_safety/from_static_benchmarks_to_dynamic_protocol_agent-centric_text_anomaly_detection_.md)

<!-- RELATED:END -->
