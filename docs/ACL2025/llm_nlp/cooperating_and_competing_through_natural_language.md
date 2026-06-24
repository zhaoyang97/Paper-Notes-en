---
title: >-
  [Paper Note] Cooperating and Competing Through Natural Language
description: >-
  [ACL 2025][LLM (Other)][Natural Language Games] This paper investigates the cooperative and competitive behaviors of LLM agents in natural language interaction environments. By designing multi-player game scenarios, it analyzes how linguistic strategies (such as persuasion, deception, and negotiation) affect game outcomes, revealing the emergent strategic capabilities and limitations of LLMs in social interactions.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Natural Language Games"
  - "Cooperation and Competition"
  - "Multi-agent Dialogue"
  - "Language Strategies"
  - "Social Interaction"
date: 2026-05-08
content_hash: e691e6b78b7b4f0a
---

# Cooperating and Competing Through Natural Language

**Conference**: ACL 2025  
**Area**: Other  
**Keywords**: Natural Language Games, Cooperation and Competition, Multi-agent Dialogue, Language Strategies, Social Interaction

## TL;DR
This paper investigates the cooperative and competitive behaviors of LLM agents in natural language interaction environments. By designing multi-player game scenarios, it analyzes how linguistic strategies (such as persuasion, deception, and negotiation) affect game outcomes, revealing the emergent strategic capabilities and limitations of LLMs in social interactions.

## Background & Motivation

**Background**: With the advancement of LLM capabilities, there is a growing demand to deploy AI agents in social interaction scenarios—such as business negotiation assistance, multi-party coordination, and debate systems. The core characteristic of these scenarios is that agents must interact strategically with other participants through natural language, involving both cooperation and competition.

**Limitations of Prior Work**: (1) Most existing multi-agent LLM research focuses on purely cooperative scenarios (e.g., multi-agent programming), with insufficient study on competitive and mixed-motive scenarios; (2) Classical game theory assumes participants share common knowledge and make rational decisions, whereas in natural language interactions, information is transmitted through language, which itself serves as a strategic tool (potentially misleading, persuasive, or ambiguous), rendering classical analytical frameworks inapplicable; (3) There is a lack of a systematic evaluation framework to measure LLM capabilities in strategic linguistic interactions.

**Key Challenge**: In natural language interactions, cooperation requires information sharing and trust-building, while competition requires information hiding and strategic deception. Both occur through the same linguistic channel, forming a unique "language strategy space." How to understand and evaluate LLM behavior within this space remains an open question.

**Goal**: (1) Design a research framework that translates classical game-theoretic scenarios into natural language; (2) Systematically evaluate the strategic behaviors of mainstream LLMs in cooperative, competitive, and mixed-motive scenarios; (3) Analyze the types, effectiveness, and emergent patterns of linguistic strategies.

**Key Insight**: The authors select three classes of classical game scenarios—Prisoner's Dilemma (pure competition), Public Goods Game (cooperation), and Bargaining Game (mixed-motive)—and convert them into natural language dialogue tasks, allowing LLM agents to play games through multi-turn dialogues.

**Core Idea**: Convert classical scenarios from game theory into natural language multi-turn dialogue tasks. By enabling LLM agents to interact in these verbalized games, the study systematically investigates the emergence, effectiveness, and impact of language strategies on game outcomes.

## Method

### Overall Architecture
The research framework consists of three levels: (1) Game scenario design—converting classical games into natural language dialogue scenarios, where each scenario has clear rules, roles, and payoff structures; (2) Agent interaction—allowing different LLMs (or LLM against human) to engage in multi-turn dialogue games within the scenarios; (3) Strategy analysis—conducting multi-level analysis of interaction logs, including outcome analysis (payoffs), strategy analysis (classification of linguistic actions), and dynamic analysis (how strategies evolve over time).

### Key Designs

1. **Verbalized Game Scenario Design**:

    - Function: Translates abstract game-theoretic scenarios into operational natural language interaction tasks.
    - Mechanism: Detailed scenarios and role prompts are designed for each game. For example, in the Bargaining Game: the buyer and the seller each have a reservation price, and negotiate the final price through multi-turn dialogue; the payoff is the difference between the final price and the reservation price. The key innovation lies in allowing participants to freely use various linguistic strategies in conversation—they can lie (claiming a lower budget), persuade (highlighting product defects), threaten (claiming to have alternatives), etc. Each scenario has a maximum limit of 10 dialogue turns.
    - Design Motivation: Free-form linguistic interaction is closer to real-world social interaction than formalized games, revealing the capability boundaries of LLMs in the "language strategy" dimension.

2. **Strategy Annotation and Classification System**:

    - Function: Systematically classifies and quantifies the verbal behaviors of LLMs.
    - Mechanism: Based on pragmatics and game theory, the taxonomy divides linguistic strategies into six major categories: (a) Information strategies—truthful information sharing, selective information disclosure, false information; (b) Persuasion strategies—logical argument, emotional appeal, authority citation; (c) Commitment strategies—promises, threats, ultimatums; (d) Cooperative strategies—proposals, compromises, reciprocity; (e) Competitive strategies—insistence, rejection, pressure; (f) Meta-strategies—topic switching, delaying, obfuscation. A trained classifier is used to automatically annotate strategy types for each dialogue turn.
    - Design Motivation: Fine-grained strategy classification enables quantitative analysis—allowing statistics on which strategies different models tend to use in various situations, and which strategies are most effective.

3. **Dynamic Strategy Analysis Framework**:

    - Function: Tracks the usage patterns and evolution of strategies during the game process.
    - Mechanism: The multi-turn game process is modeled as a sequence of strategies, and strategy transition probability matrices (the probability of transitioning from Strategy A to Strategy B) are computed. By comparing these with Nash equilibrium strategies in game theory, the framework analyzes whether LLM strategies are rational and whether they adaptively change in response to the opponent's strategies. It also introduces a "strategy diversity index" to quantify the flexibility of LLMs in utilizing strategies.
    - Design Motivation: Static analysis only shows the overall distribution of strategies, whereas dynamic analysis reveals temporal patterns—such as whether a "cooperate-first, compete-later" strategy evolution path exists.

### Loss & Training
This is an analytical paper that does not train new models. The strategy classifier is fine-tuned on DeBERTa and trained on approximately 5,000 human-annotated strategy label data.

## Key Experimental Results

### Main Results (Performance of Different Models in Three Games)

| Model | Bargaining Payoff ↑ | Prisoner's Dilemma Cooperation Rate | Public Goods Contribution Rate | Strategy Diversity ↑ |
|------|-------------|-------------|-------------|-----------|
| GPT-4 | 72.3 | 43% | 61% | 0.82 |
| Claude-3 | 68.5 | 52% | 68% | 0.78 |
| LLaMA-3-70B | 61.2 | 38% | 54% | 0.65 |
| GPT-3.5 | 55.4 | 35% | 48% | 0.54 |
| Human Baseline | 70.1 | 47% | 58% | 0.91 |

### Strategy Usage Analysis

| Strategy Type | GPT-4 Usage | Claude-3 Usage | Effectiveness (Payoff Correlation) | Description |
|---------|-----------|-------------|----------------|------|
| Truthful Information Sharing | 28% | 35% | +12% | Claude tends to be more honest |
| Selective Disclosure | 22% | 18% | +18% | Most effective strategy |
| False Information | 8% | 3% | +5% | Effective in the short term but harmful long term |
| Emotional Appeal | 15% | 20% | +8% | Used more frequently by Claude |
| Compromise Proposal | 18% | 16% | +15% | Most effective in cooperative scenarios |
| Pressure/Threat | 9% | 8% | -3% | Usually counterproductive |

### Key Findings
- GPT-4 performs best in bargaining, reaching near-human levels, primarily due to the high-frequency use of the "selective disclosure" strategy.
- The Claude series exhibits a stronger "cooperative tendency"—yielding the highest cooperation and contribution rates. However, this becomes a disadvantage in purely competitive scenarios.
- All LLMs exhibit lower strategy diversity than humans, showing obvious "strategic rigidness"—tending to repeatedly use a few pre-defined strategies.
- "Threat" and "pressure" strategies are almost always counterproductive in LLM-to-LLM interactions.

## Highlights & Insights
- Combining game theory with natural language interaction is an excellent research direction that bridges theory and practice—revealing the real ability boundaries of LLMs as participants in social interaction.
- The design of the strategy classification system is very systematic. The six categories are both theoretically grounded (pragmatics, game theory) and highly operational.
- The "strategic rigidness" phenomenon is an important finding—indicating that the adaptive capacity of current LLMs in strategic language interaction is still far inferior to that of humans.

## Limitations & Future Work
- Although the experimental scenarios are classical, they are relatively simple. Real-world social interactions usually involve more participants and more complex payoff structures.
- The strategic behaviors of LLMs might be influenced by human preferences during RLHF training (e.g., a tendency to cooperate and avoid deception), rather than purely "emerging naturally."
- Reputation and trust accumulation effects in long-term repeated games are not considered.
- Future work can explore how to improve the strategy diversity and adaptability of LLMs through prompt engineering or fine-tuning.

## Related Work & Insights
- **vs CICERO (Meta)**: CICERO trains strategic AI in the game of Diplomacy but uses a formalized action space; in contrast, this paper operates purely through natural language interaction, which is closer to real-world social scenarios.
- **vs Avalon/Werewolf Studies**: Social deduction game research focuses on deception detection, whereas this work systematically analyzes the usage patterns of six categories of linguistic strategies.
- **vs Multi-Agent Debate**: The debate scenario is purely competitive; this paper covers cooperative and mixed-motive scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The research framework combining game theory and natural language interaction is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-scenario, multi-model, and quantitative strategy analysis, yet lacks larger-scale experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear interdisciplinary writing with a complete framework description.
- Value: ⭐⭐⭐⭐⭐ Offers important insights for understanding the social interaction capabilities of LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LLMs Know Their Vulnerabilities: Uncover Safety Gaps through Natural Distribution Shifts](llms_know_their_vulnerabilities_uncover_safety_gaps_through_natural_distribution.md)
- [\[ACL 2025\] What Makes a Good Natural Language Prompt?](good_natural_language_prompt.md)
- [\[ACL 2025\] Internal and External Impacts of Natural Language Processing Papers](internal_and_external_impacts_of_natural_language_processing_papers.md)
- [\[ACL 2025\] Enhancing Transformation from Natural Language to Signal Temporal Logic Using LLMs with Diverse External Knowledge](enhancing_transformation_from_natural_language_to_signal_temporal_logic_using_ll.md)
- [\[ACL 2025\] QualiSpeech: A Speech Quality Assessment Dataset with Natural Language Reasoning](qualispeech_a_speech_quality_assessment_dataset_with_natural_language_reasoning_.md)

</div>

<!-- RELATED:END -->
