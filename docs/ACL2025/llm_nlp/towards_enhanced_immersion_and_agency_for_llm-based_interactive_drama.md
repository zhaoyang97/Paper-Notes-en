---
title: >-
  [Paper Note] Towards Enhanced Immersion and Agency for LLM-based Interactive Drama
description: >-
  [ACL 2025][LLM (Other)][Interactive Drama] This work proposes an Immersion-Agency paradigm to conceptualize LLM-based interactive drama, and designs two methods—Playwriting-guided Generation and Plot-based Reflection—to enhance story generation quality and player agency, respectively.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Interactive Drama"
  - "Immersion"
  - "Agency"
  - "Script Generation"
  - "Role-Playing Agent"
date: 2026-05-08
content_hash: 1bdbf90abe0112ae
---

# Towards Enhanced Immersion and Agency for LLM-based Interactive Drama

**Conference**: ACL 2025  
**arXiv**: [2502.17878](https://arxiv.org/abs/2502.17878)  
**Code**: [GitHub](https://github.com/gingasan/interactive-drama)  
**Area**: LLM NLP  
**Keywords**: Interactive Drama, Immersion, Agency, Script Generation, Role-Playing Agent

## TL;DR

This work proposes an Immersion-Agency paradigm to conceptualize LLM-based interactive drama, and designs two methods—Playwriting-guided Generation and Plot-based Reflection—to enhance story generation quality and player agency, respectively.

## Background & Motivation

LLM-based interactive drama is an emerging genre of AI conversational application, where players role-play and interact with LLM agents representing other characters to experience an unfolding narrative. However, existing works suffer from the following limitations:

1. **Lack of Theoretical Framework**: Previous studies focused primarily on general architecture design, without deeply exploring the core dimensions of the interactive experience. This paper introduces two key concepts from classical interactive narrative theory: Immersion (the feeling of being absorbed in the story) and Agency (the player's ability to influence the story world).
2. **Insufficient Story Generation Quality**: Although LLMs are exposed to a vast amount of literary works during pre-training, fine-tuning processes lack emphasis on playwriting techniques. Consequently, the generated stories often lack fundamental dramatic structures and compelling conflicts. Experiments demonstrate that GPT-4o and Qwen2.5-72b rarely employ any narrative techniques without explicit prompting.
3. **Neglect of Character Agency**: Previous character agent architectures rarely considered how player actions could meaningfully influence character reactions and the narrative trajectory.

## Method

### Overall Architecture

The system consists of two major modules: (1) Script Generation, which utilizes Playwriting-guided Generation to generate high-quality dramatic stories (including plot structures and narrative techniques) from a premise provided by the player; (2) Character Agents, which employ Plot-based Reflection to allow NPCs to dynamically adjust the plot chain based on player actions, thereby enhancing agency.

### Key Designs

1. **Playwriting-guided Generation**:
    - Defines 8 classic dramatic situations (such as love, phoenix rebirth, Cinderella, vengeance, etc.), described based on Aristotle’s three-act structure (setup, confrontation, resolution).
    - Summarizes 6 micro-narrative techniques (suspense, twists, non-linear narrative, multiple-narrative, irony, symbolism).
    - Generation workflow: Sample 1 dramatic situation + 3 narrative techniques $\rightarrow$ Writer LLM generates the story $\rightarrow$ Critic LLM evaluates and provides improvement feedback $\rightarrow$ Writer revises $\rightarrow$ Repeat 3 times to select the best formulation $\rightarrow$ Progressive refinement of details.
    - Effect: Narrative technique utilization rate increases from $6\%\text{--}12\%$ in the baseline to $28\%\text{--}74\%$ (based on GPT-4o).

2. **Plot-based Reflection**:
    - The character agent performs a reflection every $k=5$ interaction steps to analyze memories of player activities (emotions, intentions) and dynamically adjust the plot chain.
    - Each reflection is constrained to adjusting at most one incomplete plot point or inserting at most one new plot point, preventing incoherent narratives caused by excessive LLM modifications.
    - This enables characters to exhibit meaningful shifts in reactions driven by player behavior, such as leaking secrets, offering companionship, or advancing the plot in specific directions.

3. **Hybrid Agent Architecture**:
    - Director-Actor Architecture: A Director Agent coordinates globally while individual Actor Agents play their respective characters, which is suitable for high-interaction scenarios.
    - One-for-All Architecture: A single global agent plays all characters, yielding higher efficiency, suitable for narrative-centric scenarios.
    - The hybrid approach dynamically switches between these two architectures based on scenario characteristics, balancing performance and efficiency (accelerating inference by $1.49\times$).

### Loss & Training

This work does not involve model training; all agents are built on prompt engineering using GPT-4o. The key strategies include:
- A "Sampling-Critic-Revise" loop to ensure the correct application of playwriting techniques.
- Progressive generation (adding details from coarse to fine).
- A memory system that preserves all dialogue history within the prompt.

## Key Experimental Results

### Main Results

Story generation evaluation (50 premise paragraphs, human annotator win rates):

| Method | Conflict (Best ↑ / Worst ↓) | Suspense | Emotional Tension | Character Arc | Technique Adherence Rate |
|------|--------|------|---------|--------|-----------|
| Outline-First | 18%/34% | 10%/28% | 10%/50% | 18%/36% | - |
| Playwriting-Guided | **32%/24%** | **32%/22%** | **48%/16%** | **34%/20%** | 92% |
| w/o Critic & Revise | 24%/24% | 26%/34% | 18%/28% | 12%/32% | 66% |
| w/o Refinement | 26%/18% | 32%/26% | 24%/6% | 36%/12% | - |

Character Agent evaluation (5-point scale, hand-crafted script "Seven at the Station", 10 human players + 10 agent players):

| Architecture | Character Consistency | Attractiveness | Narrative Completeness | Progress | Influence | Intention Following | Speedup Ratio |
|------|----------|--------|---------|------|--------|---------|--------|
| Director-Actor | 3.9 | 4.2 | 3.8 | 3.6 | 4.2 | 3.9 | 1.00x |
| Hybrid Architecture | 4.1 | 3.9 | 4.3 | **4.3** | 4.0 | 4.0 | 1.49x |
| w/o Reflection | 4.0 | 3.5 | 4.2 | 3.9 | 3.5 | 3.3 | 1.90x |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| w/o Critic & Revise | Technique adherence rate $92\% \rightarrow 66\%$ | Critic LLM is crucial for ensuring the correct application of playwriting techniques. |
| w/o Refinement | Emotional tension $48\% \rightarrow 24\%$ | Progressive refinement contributes the most to emotional details. |
| w/o Plot-based Reflection | Influence $4.0 \rightarrow 3.5$, Intention following $4.0 \rightarrow 3.3$ | The reflection mechanism is the core of agency. |
| Pure Director-Actor | Progress 3.6 vs. Hybrid 4.3 | Multi-agent communication causes information loss, affecting narrative progress. |

### Key Findings

1. **Progressive refinement contributes the most to emotional tension** ($10\% \rightarrow 48\%$), as emotions typically derive from subtle details in the text.
2. **Plot-based Reflection not only enhances agency but also increases character attractiveness** ($3.5 \rightarrow 3.9$), presumably because reflection encourages characters to display stronger empathy.
3. **Aggressive agent players paradoxically scored higher in character attractiveness and influence dimensions**, likely because agents interact more actively, and high-quality responses in turn left a deep impression on the annotators.
4. **The algorithm can automatically match appropriate dramatic situations to different themes** (e.g., romance $\rightarrow$ love, crime $\rightarrow$ deliverer/savior).

## Highlights & Insights

- **Theoretical Contribution**: First to establish an Immersion-Agency evaluation paradigm for LLM-based interactive drama, providing an analytical framework from the perspectives of narratology and psychology.
- **Systematization of Dramatic Techniques**: Combining classical dramatic theories (Polti’s 36 dramatic situations, Aristotle’s three-act structure) with modern LLM prompt engineering, presenting an interesting practice of computational creativity.
- **Human-Centric Evaluation**: Rejecting automated LLM evaluators and insisting on evaluations conducted by annotators with training in the humanities, as the evaluation of literary works requires accuracy and empathy.
- **Practicality of Hybrid Architecture**: Dynamically selecting the architecture based on scenario characteristics achieves a favorable balance between efficiency and quality.

## Limitations & Future Work

1. **Dependency on GPT-4o**: All agents rely on the same closed-source model, which is costly and uncontrollable.
2. **Efficiency Issues**: Playwriting-guided Generation is 10-12 times slower than vanilla prompting.
3. **Reflection Boundary Control**: LLMs tend to over-adjust the narrative. Currently, this is addressed via hard constraints, which may restrict more creative adaptation.
4. **Solely Focused on Dialogue Formats**: Exploration of scene generation (multimodal elements like visuals and music) for enhancing immersion is not conducted.
5. **Limited Evaluation Scale**: Tested with only 10 human players and one hand-crafted script; generalizability remains to be verified.

## Related Work & Insights

- **Mateas (2000)**'s theory of Interactive Drama provides the original definitions of Immersion and Agency, which this work operationalizes into evaluable dimensions.
- **Park et al. (2023)**'s memory-based reflection focuses on memory synthesis, whereas Plot-based Reflection focuses on plot adaptation; the two are complementary, parallel techniques.
- **Wu et al. (2024)** first defined the six elements and plot chain mechanism of LLM interactive drama, upon which this work adds reflection and generation enhancement.
- This paper demonstrates the possibility of integrating classical dramatic theory into AI systems, providing broad insights for NPC AI, game narrative, educational simulation, and other fields.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Theoretical Depth | 4 |
| Experimental Thoroughness | 3 |
| Practical Value | 4 |
| Writing Quality | 4 |
| Overall Score | 3.8 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Cross-Modal Alignment for LLM-Enhanced Spoken Language Understanding](cross-modal_alignment_for_llm-enhanced_spoken_language_understanding.md)
- [\[ACL 2025\] Boosting LLM's Molecular Structure Elucidation with Knowledge Enhanced Tree Search Reasoning](boosting_llms_molecular_structure_elucidation_with_knowledge_enhanced_tree_searc.md)
- [\[ACL 2025\] CER: Confidence Enhanced Reasoning in LLMs](cer_confidence_enhanced_reasoning.md)
- [\[ACL 2025\] Alignment Drift in CEFR-prompted LLMs for Interactive Spanish Tutoring](alignment_drift_in_cefr-prompted_llms_for_interactive_spanish_tutoring.md)
- [\[ACL 2025\] Interactive and Expressive Code-Augmented Planning with Large Language Models](interactive_and_expressive_code-augmented_planning_with_large_language_models.md)

</div>

<!-- RELATED:END -->
