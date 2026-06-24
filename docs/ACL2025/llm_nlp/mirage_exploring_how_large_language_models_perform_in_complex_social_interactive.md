---
title: >-
  [Paper Note] MIRAGE: Exploring How Large Language Models Perform in Complex Social Interactive Environments
description: >-
  [ACL 2025][LLM (Other)][Social Interaction Evaluation] This paper proposes MIRAGE, an evaluation framework that systematically assesses the performance of LLMs in complex social interactive environments through eight carefully designed Murder Mystery game scenarios and four core metrics (Trust Inclination Index [TII], Clue Investigation Capability [CIC], Interaction Capability Index [ICI], and Script Compliance Index [SCI]). The findings reveal that even GPT-4 faces severe ch…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Social Interaction Evaluation"
  - "Role-Playing"
  - "Murder Mystery Games"
  - "LLM Evaluation"
  - "Trust and Deception"
date: 2026-05-08
content_hash: 1248a6855ee78f22
---

# MIRAGE: Exploring How Large Language Models Perform in Complex Social Interactive Environments

**Conference**: ACL 2025  
**arXiv**: [2501.01652](https://arxiv.org/abs/2501.01652)  
**Code**: [GitHub](https://github.com/) (The paper states that the dataset and simulation code have been open-sourced)  
**Area**: LLM/NLP  
**Keywords**: Social Interaction Evaluation, Role-Playing, Murder Mystery Games, LLM Evaluation, Trust and Deception

## TL;DR

This paper proposes MIRAGE, an evaluation framework that systematically assesses the performance of LLMs in complex social interactive environments through eight carefully designed Murder Mystery game scenarios and four core metrics (Trust Inclination Index [TII], Clue Investigation Capability [CIC], Interaction Capability Index [ICI], and Script Compliance Index [SCI]). The findings reveal that even GPT-4 faces severe challenges in these scenarios.

## Background & Motivation

LLMs have demonstrated significant potential in environmental perception, reasoning, decision-making, and simulating complex human behavior, particularly in role-playing scenarios. However, existing evaluations suffer from several key limitations:

**Focusing on Agents while neglecting the LLM itself**: Prior works such as Sotopia and Lyfe Agents focus on the social interaction capabilities of workflow-enhanced agent systems, but overlook a crucial fact—the foundational social interaction capability originates from the underlying LLM itself.

**Overly simplified game scenarios**: Board games like Werewolf and Avalon are constrained by rigid decision-making workflows and limited scenario diversity, failing to fully test advanced social behaviors of LLMs (such as deception and leadership).

**Insufficient existing murder mystery simulations**: Pioneering work by Wu et al. (2023) is limited by a narrow scope of game scripts, simplistic evaluation methods, and a lack of human validation for the datasets.

Murder Mystery Games serve as an ideal evaluation paradigm: participants portray specific characters and engage in semi-structured narrative interactions, requiring extensive background knowledge, socially-driven decision-making, and open-ended interactions. These characteristics make it exceptionally suitable for evaluating how LLMs handle complex human social behaviors.

## Method

### Overall Architecture

MIRAGE consists of three components: (1) an evaluation environment composed of eight carefully crafted murder mystery scenarios; (2) a three-stage simulation workflow driving game execution; and (3) four objective evaluation metrics to quantify LLM performance.

### Key Designs

1. **Script Construction (8 Experimental Scenarios)**: Each script contains six parts: character background story (contextual info), character script (event timeline), character relationships (initial social network), character performance (persona and speaking style), character goals (tasks and objectives), and other rules of the game. The eight scripts cover multiple dimensions:

    - Structural dimension: Single-stage vs. Multi-stage (character information revealed all at once or in phases)
    - Genre dimension: Traditional (realistic settings) vs. Non-traditional (supernatural/fantasy)
    - Ending dimension: Closed-ended (fixed ending) vs. Open-ended (ending changes according to player actions)
    - Scale dimension: 5-10 characters, 14-82 clues, and 3K-62K words

2. **Three-stage Simulation Workflow**: All characters are divided into two factions: murderers and civilians.

    - **Phase A - Open Dialogue**: Players engage in turn-based open-ended dialogues based on their script roles.
    - **Phase B - Environmental Interaction**: Players choose either to "Inquire" (ask a specific character a question, which they must answer) or "Investigate" (publicly reveal a clue to all characters).
    - **Phase C - Voting**: Players accuse suspects and vote. Civilians win if the murderer receives the most voting accusations.

3. **Auxiliary Modules**:

    - **Summary Module**: Compresses historical context when input exceeds the LLM's context limit.
    - **Suspicion Module**: Records suspicion scores toward other characters after each dialogue round.
    - **Trust Module**: Records trust scores toward other characters after each dialogue round.
    - **Retry Module**: Resubmits the prompt when the LLM's output fails to parse.

4. **Four Evaluation Metrics**:

    - **TII (Trust Inclination Index)**: Integrates suspicion and trust scores to measure the LLM's ability to balance trust and suspicion during social interactions.
    - **CIC (Clue Investigation Capability)**: The ratio of investigated clues to total clues, measuring information collection and problem-solving abilities.
    - **ICI (Interaction Capability Index)**: Evaluated by a strong neutral LLM, encompassing reasoning and analysis, communication and cooperation, observation, and creative thinking.
    - **SCI (Script Compliance Index)**: Measures character role-playing fidelity via the average of two evaluations: direct scoring and Rouge-L-based consistency of script reconstruction.

### Loss & Training

MIRAGE is an evaluation framework and does not involve model training. In the evaluation, GPT-4-Turbo is employed as the neutral evaluation model (its stability was verified to be superior to GPT-4 and GPT-4o through ablation experiments). Each character participates in an alternating cycle of five rounds of open dialogue and interaction phases.

## Key Experimental Results

### Main Results

| Model | Victory (MRR) | TII | CIC | ICI | SCI |
|------|--------------|-----|-----|-----|-----|
| GPT-3.5 | 29.11 | 47.13 | 27.46 | 70.06 | 49.10 |
| GPT-4 | 34.69 | 76.32 | 19.01 | 76.54 | 50.42 |
| GPT-4o | 47.01 | 78.69 | 35.92 | 76.80 | 51.29 |
| Qwen-2-7B | **51.81** | 75.78 | 18.66 | 74.92 | 50.57 |
| GLM-4-9B | 31.89 | 53.85 | 20.07 | 71.60 | 48.13 |

Trust Bias Experiment (Changes in TII after the murderer self-discloses their identity):

| Model | TII (No Self-Disclosure) | TII (After Self-Disclosure) | Change Δ |
|------|-------------|-------------|--------|
| Qwen-1-7B | 51.02 | 50.69 | -0.33 |
| Qwen-1.5-7B | 73.00 | 69.14 | -3.86 |
| Yi-1.5-9B | 55.73 | 57.57 | **+1.84 (Only one to correctly increase suspicion)** |
| GLM-4-9B | 57.82 | 55.94 | -1.88 |

### Ablation Study

Evaluation Model Selection Ablation:

| Evaluation Model | Stability of ICI Scoring | Description |
|---------|-------------|------|
| GPT-4 | Unstable, strong bias | Scores certain models excessively high |
| GPT-4-Turbo | **Most stable** | Most consistent scores, ultimately adopted |
| GPT-4o | Low | Systematic underestimation |

Token Usage Statistics:

| Model | Environment Input Tokens | Model Output Tokens | Environment Requests | User Completions |
|------|-------------|-------------|----------|----------|
| GPT-3.5 | 2,719,895 | 121,378 | 883 | 580 |
| GPT-4o | 6,252,580 | 204,772 | 1,328 | 574 |
| Qwen-2-7B | 2,204,029 | 192,158 | 743 | 588 |

### Key Findings

1. **GPT-4o is the strongest overall but does not have the highest win rate**: GPT-4o performs best across CIC, ICI, and SCI, but is surpassed by Qwen-2-7B in actual win rate (51.81% vs. 47.01%), suggesting that being "smarter" does not equate to "winning more."

2. **LLMs generally exhibit an over-trust bias**: Most LLMs maintain trust even in the extreme scenario where the murderer self-discloses their identity. Only Yi-1.5-9B increases its suspicion in this condition, which explains its superior performance in the Victory metric.

3. **Clue discovery patterns**: LLMs show high enthusiasm for environmental exploration in early rounds (rapidly increasing CIC), but the discovery of key clues is unstable. This indicates that while models actively explore, they lack the capability to identify critical information.

4. **Open-source models can rival some closed-source counterparts**: Qwen-2-7B (7B parameters) is close to GPT-4 in ICI and even surpasses GPT-4 in SCI, demonstrating that model size is not the sole determinant of social interaction capabilities.

5. **Complex social scenarios remain a bottleneck for LLMs**: The SCI of all models hovers only around 48-51%, indicating a substantial gap between current LLMs and the ideal performance in faithfully executing complex role-playing.

## Highlights & Insights

- The choice of using Murder Mystery Games as an evaluation tool for LLM social intelligence is highly creative—it is closer to real-world social scenarios than Werewolf or Avalon, requiring a comprehensive application of knowledge understanding, reasoning, deception, and cooperation.
- The design of the eight scenarios is highly deliberate, covering multiple dimensions including single/multi-stage, traditional/non-traditional, and closed/open endings, ensuring the diversity of the evaluation.
- The over-trust phenomenon revealed by the TII metric is highly insightful—LLMs might struggle to appropriately express suspicion in adversarial scenarios due to biases toward "friendly" and "cooperative" behaviors induced during their alignment training.
- The misalignment between win rates and capability metrics (where Qwen-2-7B achieves the highest win rate but is not the "smartest") offers an interesting insight: in game-theoretic scenarios, rational analytical capability does not necessarily translate directly into actual victory.

## Limitations & Future Work

- **Context Length Limitations**: The context limitations of LLMs necessitate the use of a summary module to compress historical information, which compromises decision-making quality.
- **Limited Scenario Scale**: The data volume of eight scripts is relatively limited, and the generalizability of the findings requires verification across more scenarios.
- **Unfair Impact of Safety Alignment**: Safety-oriented LLMs may refuse to respond to sensitive topics involving murder and deception, putting them at a disadvantage during simulation.
- **Evaluation Subjectivity**: ICI and SCI rely on LLM evaluators, and different evaluation models may yield divergent conclusions.
- **Lack of Human Baseline**: There is no comparison with human players, making it difficult to judge the actual gap between LLMs and humans.

## Related Work & Insights

- **SOTOPIA** (Zhou et al., 2023): Evaluates the social intelligence of LLM agents but focuses on workflow-enhanced systems rather than raw LLMs. MIRAGE focuses on the inherent social capabilities of the LLM itself.
- **Generative Agents** (Park et al., 2023): Generative agents that simulate human behavior, proving the feasibility of LLMs in social simulations.
- **Werewolf Game Studies** (Xu et al., 2023; Wu et al., 2024): Evaluate the deception and reasoning capabilities of LLMs using the Werewolf game, but the scenarios are rigid and lack diversity.
- This paper establishes the boundaries of LLM social capabilities in more complex narrative interaction scenarios, providing a benchmark for subsequent social AI research.

## Rating

- Novelty: ⭐⭐⭐⭐ The selection of Murder Mystery Games as an LLM evaluation tool is novel, and the eight-scenario, four-metric system is systematically designed.
- Experimental Thoroughness: ⭐⭐⭐ The evaluation scale across five models is moderate, but it lacks comparison with human baselines and testing on a wider range of open-source models.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear and the scenario design details are sufficient, although the metric formulas are scattered in the appendix, making them less intuitive.
- Value: ⭐⭐⭐⭐ Fills the evaluation gap in complex social interaction scenarios, and the over-trust phenomenon revealed by TII is also significant for LLM safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ACT: Knowledgeable Agents to Design and Perform Complex Tasks](act_knowledgeable_agents_to_design_and_perform_complex_tasks.md)
- [\[ACL 2025\] AfroBench: How Good are Large Language Models on African Languages?](afrobench_how_good_are_large_language_models_on_african_languages.md)
- [\[ACL 2025\] Large Language Models for Predictive Analysis: How Far Are They?](large_language_models_for_predictive_analysis_how_far_are_they.md)
- [\[ACL 2025\] Interactive and Expressive Code-Augmented Planning with Large Language Models](interactive_and_expressive_code-augmented_planning_with_large_language_models.md)
- [\[ACL 2025\] SocialEval: Evaluating Social Intelligence of Large Language Models](socialeval_evaluating_social_intelligence_of_large_language_models.md)

</div>

<!-- RELATED:END -->
