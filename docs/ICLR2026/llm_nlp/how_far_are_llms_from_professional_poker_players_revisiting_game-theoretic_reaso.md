---
title: >-
  [Paper Note] How Far Are LLMs from Professional Poker Players? Revisiting Game-Theoretic Reasoning with Agentic Tool Use
description: >-
  [ICLR 2026][LLM/NLP][LLM poker] This paper systematically analyzes three core reasoning deficiencies of LLMs in poker (heuristic reasoning, factual misunderstanding, and knowing-doing gap), and proposes ToolPoker — the first tool-integrated LLM reasoning system for incomplete information games. By incorporating an external CFR solver to provide game-theoretically optimal action guidance, a 7B model approaches Nash equilibrium performance in Limit Hold'em.
tags:
  - ICLR 2026
  - LLM/NLP
  - LLM poker
  - game-theoretic reasoning
  - tool-augmented LLM
  - CFR solver
  - incomplete information game
date: 2026-05-08
content_hash: cb1d5ccc721ec562
---

# How Far Are LLMs from Professional Poker Players? Revisiting Game-Theoretic Reasoning with Agentic Tool Use

**Conference**: ICLR 2026
**arXiv**: [2602.00528](https://arxiv.org/abs/2602.00528)
**Code**: To be confirmed
**Area**: LLM/NLP
**Keywords**: LLM poker, game-theoretic reasoning, tool-augmented LLM, CFR solver, incomplete information game

## TL;DR
This paper systematically analyzes three core reasoning deficiencies of LLMs in poker (heuristic reasoning, factual misunderstanding, and knowing-doing gap), and proposes ToolPoker — the first tool-integrated LLM reasoning system for incomplete information games. By incorporating an external CFR solver to provide game-theoretically optimal action guidance, a 7B model approaches Nash equilibrium performance in Limit Hold'em.

## Background & Motivation
**State of the Field**: LLMs have achieved breakthroughs in mathematical reasoning and programming, yet perform far below traditional methods in incomplete information games. Poker requires tight integration of Bayesian belief updating, game-theoretic reasoning, and strategic execution.

**Limitations of Prior Work**: LLMs exhibit three core reasoning deficiencies — ① Heuristic Reasoning: reliance on shallow heuristics rather than game-theoretic principles; ② Factual Misunderstanding: misjudging objective quantities such as hand strength and pot odds; ③ Knowing-Doing Gap: correct reasoning but action deviating from conclusions.

**Root Cause**: LLMs can generate text that "sounds like" valid game-theoretic analysis but cannot execute the underlying computations precisely.

**Paper Goals**: Enable LLMs to perform game-theoretically sound reasoning and decision-making in incomplete information games.

**Starting Point**: Integrating a CFR solver as an external tool into the LLM reasoning pipeline.

**Core Idea**: ToolPoker = LLM language understanding + precise game-theoretic computation from a CFR solver.

## Method

### Overall Architecture
A three-stage progressive design: ① Vanilla LLM evaluation (exposing problems) → ② BC-RIRL (preliminary attempt, limited effectiveness) → ③ ToolPoker (final solution).

### Key Designs

1. **Quantitative Analysis of Three Reasoning Deficiencies**:

    - HR (Heuristic Reasoning): Reliance on shallow rules such as "raise with big hands," without precise range analysis.
    - FA (Factual Accuracy): Misjudging computable quantities such as win rates and pot odds — the most critical bottleneck.
    - AC (Action Consistency): Reasoning concludes fold but the selected action is call.
    - Evaluation method: LLM-as-Judge scoring (0–2 scale).

2. **BC-RIRL (Preliminary Approach, Limited Effectiveness)**:

    - BC: Behavioral cloning on 5k expert reasoning trajectories; RIRL: RL using CFR cumulative regret signals.
    - Improves reasoning style but FA remains low (~1.12/2.0); overall performance is even worse (−77.5 vs. −53.5 chips).

3. **ToolPoker (Final Solution)**:

    - Unified tool interface: CFR solver and equity calculator merged into a single API, returning GTO action + equity + range + pot odds.
    - BC phase: Programmatically generated tool-call datasets to teach the model when and how to invoke the solver.
    - RL phase: Composite reward $R = R_{answer} + \alpha_f R_{format} + \alpha_t R_{tool}$
    - Design Motivation: The solver compensates for LLMs' computational limitations while preserving their reasoning and explanation capabilities.

## Key Experimental Results

### Main Results — vs. Traditional Methods (chip differential, positive = winning)

| Method | vs NFSP | vs DQN | vs DMC | vs DeepCFR |
|------|---------|--------|--------|------------|
| Vanilla Qwen-7B | −53.5 | −48.0 | −49.5 | −62.0 |
| BC-RIRL | −77.5 | −74.0 | −76.0 | −85.5 |
| **ToolPoker** | **+60.5** | **+63.0** | **+61.5** | **−5.0** |

### Reasoning Quality Evaluation (0–2 scale)

| Method | HR | FA | AC |
|------|-------|-------|------|
| Vanilla Qwen-7B | 1.34 | 1.06 | 1.56 |
| BC-RIRL | 1.60 | 1.12 | 1.68 |
| **ToolPoker** | **1.92** | **1.90** | **1.94** |
| o4-mini | 1.80 | 1.56 | 1.85 |

### Key Findings
- Vanilla LLMs, including GPT-4o, are entirely unable to defeat CFR+.
- BC-RIRL performs worse than Vanilla — imitation + RL alone is insufficient to remedy computational deficiencies.
- ToolPoker FA improves from 1.12 to 1.90 — the solver fully resolves factual misunderstanding.
- Against DeepCFR, the chip differential is only −5.0, approaching Nash equilibrium.

## Highlights & Insights
- The **HR/FA/AC diagnostic framework** is transferable to other tasks requiring precise reasoning.
- The **knowing-doing gap** is an underappreciated problem in LLMs.
- The **complementarity of tools and LLMs** yields significant gains in game-theoretic settings.
- The counterintuitive finding that BC-RIRL underperforms Vanilla suggests that imitation + RL alone may reinforce erroneous patterns.

## Limitations & Future Work
- Reliance on an external CFR solver introduces latency and deployment complexity.
- Validation is limited to two-player poker; extension to multi-player games remains unexplored.
- Tool calls occasionally produce formatting errors.
- Fine-tuning is conducted only on Qwen2.5-7B.

## Related Work & Insights
- **vs. Pluribus/Libratus**: Traditional AI poker relies on pure CFR; ToolPoker positions the LLM as a "user interface" for the CFR solver.
- **vs. ReAct**: Similar tool-use paradigm but designed specifically for game-theoretic settings.
- Implication: LLM applications requiring precise computation should consistently consider tool augmentation.

## Supplementary Technical Details

### Introduction to CFR (Counterfactual Regret Minimization)
CFR is the standard algorithm for solving Nash equilibria in incomplete information games. It converges to an equilibrium strategy by iteratively minimizing counterfactual regret at each information set. In poker, CFR computes GTO (Game Theory Optimal) strategies at each decision point, including precise probabilities for each action.

### Details of the Composite Reward Design
The design of $R_{tool}$ is particularly important: it rewards not merely "invoking the tool" but "correctly utilizing the tool's returned results" — preventing the model from learning to call the tool while ignoring the solver's output. $R_{format}$ ensures parseable output and prevents tool-call failures due to formatting errors.

### Why Does BC-RIRL Perform Worse Than Vanilla?
A plausible explanation is that the BC phase imitates the expert's reasoning *style* without replicating precise computational ability, causing the model to "fail confidently" — wrapping incorrect factual judgments in expert-sounding language, which is more dangerous than honestly admitting uncertainty. This finding serves as a warning for all LLM fine-tuning approaches based on behavioral cloning: imitating "talking like an expert" and "thinking like an expert" are fundamentally distinct objectives.

### The Unique Challenges of Poker
The key distinction between poker and chess or Go lies in information incompleteness: players cannot observe opponents' cards. This requires strategies that not only account for current hand strength but also maintain and update belief distributions over opponents' hand ranges after each action. LLMs are substantially less capable at this form of Bayesian reasoning than at deterministic reasoning. ToolPoker addresses this computational bottleneck precisely through the CFR solver.

## Rating
- Novelty: ⭐⭐⭐⭐ First CFR+LLM integrated system; deficiency analysis is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation against multiple traditional methods combined with reasoning quality assessment.
- Writing Quality: ⭐⭐⭐⭐ Three-stage progressive structure is clearly presented.
- Value: ⭐⭐⭐⭐ A paradigmatic example of tool-augmented LLMs in game-theoretic settings.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Perception Programs: Unlocking Visual Tool Reasoning in Language Models](../../CVPR2026/llm_nlp/perception_programs_visual_tool_reasoning.md)
- [\[ACL 2026\] How Do Answer Tokens Read Reasoning Traces? Self-Reading Patterns in Thinking LLMs](../../ACL2026/llm_nlp/how_do_answer_tokens_read_reasoning_traces_self-reading_patterns_in_thinking_llm.md)
- [\[ICLR 2026\] How Catastrophic is Your LLM? Certifying Risk in Conversation](how_catastrophic_is_your_llm_certifying_risk_in_conversation.md)
- [\[ICLR 2026\] Predicting LLM Reasoning Performance with Small Proxy Models](predicting_llm_reasoning_performance_with_small_proxy_models.md)
- [\[AAAI 2026\] Understanding Syllogistic Reasoning in LLMs from Formal and Natural Language Perspectives](../../AAAI2026/llm_nlp/understanding_syllogistic_reasoning_in_llms_from_formal_and_natural_language_per.md)

<!-- RELATED:END -->
