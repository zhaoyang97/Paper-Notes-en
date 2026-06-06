---
title: >-
  [Paper Note] HumanLLM: Benchmarking and Improving LLM Anthropomorphism via Human Cognitive Patterns
description: >-
  [ACL 2026][LLM Evaluation][Anthropomorphism] This paper proposes the HumanLLM framework, which models 244 psychological patterns (100 personality traits + 144 social cognitive patterns) as interacting causal forces rathe…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Anthropomorphism"
  - "Cognitive Patterns"
  - "Multi-pattern Dynamics"
  - "Role-playing Agent"
  - "Psychological Modeling"
date: 2026-05-08
content_hash: c81b661473a255d7
---

# HumanLLM: Benchmarking and Improving LLM Anthropomorphism via Human Cognitive Patterns

**Conference**: ACL 2026  
**arXiv**: [2601.10198](https://arxiv.org/abs/2601.10198)  
**Code**: [GitHub](https://github.com/YJGoodbye2024/HumanLLM)  
**Area**: Role-playing / Personality Simulation  
**Keywords**: Anthropomorphism, Cognitive Patterns, Multi-pattern Dynamics, Role-playing Agent, Psychological Modeling

## TL;DR

This paper proposes the HumanLLM framework, which models 244 psychological patterns (100 personality traits + 144 social cognitive patterns) as interacting causal forces rather than isolated labels. It constructs a dataset of 11,359 scenarios and multi-turn dialogues containing interactions of 2-5 patterns. Through a dual-layer checklist evaluation, it achieves high alignment with human judgment ($r=0.90$). HumanLLM-8B outperforms Qwen3-32B in multi-pattern dynamics with 4x fewer parameters.

## Background & Motivation

**Background**: Role-playing language agents (RPLAs) have evolved from conceptual frameworks to practical applications such as digital clones, AI companions, and social simulations. Existing personality injection methods include: (1) Prompting—assigning personality labels via instructions; (2) Fine-tuning—training on character-specific data; (3) Activation steering—manipulating internal representations via persona vectors.

**Limitations of Prior Work**: (1) Existing methods model personality as isolated label-to-behavior mappings ("extroverted" $\rightarrow$ "talkative"), ignoring the dynamic interactions between multiple cognitive patterns—in reality, a talkative person might remain silent when the "spotlight effect" is activated; (2) This leads to "personality hallucinations"—models claim a trait in self-reports but exhibit inconsistent behavior; (3) Existing evaluations use holistic metrics (e.g., CoSER's Anthropomorphism), which implicitly equate "good anthropomorphism" with "pro-social behavior," penalizing realistic but negative human traits (e.g., defensive attribution).

**Key Challenge**: Human behavior is the product of dynamic interactions among multiple cognitive patterns—a confident person may yield under conformity pressure, and a talkative person may become silent when under scrutiny. Existing methods only simulate single traits and fail to capture this "inter-pattern tension and modulation."

**Goal**: (1) Construct a large-scale psychological pattern dataset (each pattern based on approximately 50 academic papers); (2) Design multi-pattern interaction scenarios for the model to learn dynamic relationships between patterns; (3) Propose an evaluation framework that distinguishes "simulation accuracy" from "social desirability."

**Key Insight**: Grounded in Lewin's field theory—human cognition consists of two dimensions: stable personality traits (Person) and context-triggered social cognitive patterns (Environment). By treating patterns as interacting causal forces rather than isolated labels, the model can implicitly learn reinforcement, conflict, and conditional modulation between patterns through training in multi-pattern scenarios.

**Core Idea**: Modeling psychological patterns as interacting causal forces. By training LLMs in multi-pattern interaction scenarios, the model learns "not just what humans do, but the psychological processes that generate these behaviors"—upgrading from behavior mimicry to cognitive modeling.

## Method

### Overall Architecture

HumanLLM consists of three core components: (1) Pattern data construction—extracting structured representations (definitions + mechanisms + manifestations) of 244 psychological patterns from approximately 12,000 academic papers; (2) Scenario and dialogue generation—constructing 11,359 scenarios with 2-5 interacting patterns, each involving multi-turn dialogues for 2-6 characters (including internal thoughts, actions, and utterances); (3) Dual-layer checklist evaluation—pattern-level (12-15 general behavioral indicators) + scenario-level (2-6 context-specific behavioral expectations).

### Key Designs

1.  **Literature-based Psychological Pattern Construction**:
    *   Function: Provides a scientifically rigorous structured representation for each pattern.
    *   Mechanism: Personality traits use Goldberg's 100 unipolar markers (20 descriptors per Big Five dimension). Social cognitive patterns are curated from cognitive biases (Tversky & Kahneman), social influence (Cialdini), evolutionary psychology, and motivation research, filtering 144 from 232 candidates (requiring empirical validation and non-redundancy). Each pattern is synthesized from ~50 academic papers via Gemini Deep Search and Gemini 2.5 Pro into a three-layer structure: definition, core mechanism, and real-world manifestation. Manual verification showed average scores of 3.20-3.70 (4-point scale) with Krippendorff's $\alpha = 0.58-0.76$.
    *   Design Motivation: Unlike existing work that generates character descriptions from model parametric knowledge, each pattern in this work is supported by ~50 academic papers—ensuring psychological rigor and scientific validity.

2.  **Multi-pattern Interaction Scenario Generation**:
    *   Function: Enables the model to learn dynamic relationships between patterns (reinforcement/conflict/conditional modulation).
    *   Mechanism: Each scenario contains combinations of 2-5 patterns covering three interaction types: reinforcement (e.g., "self-serving bias" strengthening "overconfidence effect"), conflict (e.g., "confidence" vs. "conformity"), and conditional modulation (e.g., "talkativeness" suppressed by "spotlight effect"). Character design includes self-perception and other-perception to support information asymmetry. The DIAMONDS model ensures situational diversity. For each scenario, expected behavioral tendencies are generated as evaluation criteria. Dialogues (12-20 turns) are generated by Claude Sonnet 4.5, with each turn including three-dimensional expression: internal thoughts [brackets], physical actions (parentheses), and linguistic utterances.
    *   Design Motivation: The key innovation lies in the diversity of pattern combinations—not simply stacking traits, but building scenarios that require "negotiation" between patterns. The three-dimensional expression design allows the model to learn the separation between surface behavior and internal cognitive processes.

3.  **Dual-layer Checklist Evaluation**:
    *   Function: Decouples simulation accuracy from social desirability.
    *   Mechanism: The pattern-level checklist includes 12-15 universal behavioral indicators for each pattern (e.g., Spotlight Effect: "Overestimating others' attention to one's appearance"), derived from definitions and applicable across scenarios. The scenario-level checklist includes 2-6 context-specific behavioral expectations per character (e.g., "Persistence in conceptual integrity despite deadline pressure"). GPT-5-mini serves as a ternary judge (+1 satisfied / 0 not shown / -1 violated). Metrics: IPE (Individual Pattern Expression) measures single pattern fidelity, and MPD (Multi-Pattern Dynamics) measures emergent behavior from multi-pattern interactions.
    *   Design Motivation: Traditional holistic metrics (e.g., CoSER's Anthropomorphism) correlate poorly with human judgment ($r=0.43$) and suffer from "normative confusion"—LLM judges rate "defensive attribution" as low anthropomorphism due to "lack of empathy." The checklist achieves $r=0.90$ human alignment via value-neutral behavioral indicators.

### Loss & Training

Supervised Fine-Tuning (SFT): Dialogues for each character were converted into ShareGPT format, producing 30,543 HumanLLM samples. Mixed training data: HumanLLM + OpenThoughts-114k (instruction following) + CoSER (role-playing) in a 4:4:2 ratio, totaling 76,358 samples. Base models are Qwen3-8B/32B.

## Key Experimental Results

### Main Results

**IPE and MPD Evaluation (%, Mean ± SD of 3 trials)**

| Model | IPE | MPD |
| :--- | :--- | :--- |
| GPT-5 | 15.5±0.4 | 43.4±1.1 |
| Claude Sonnet 4.5 | 34.8±0.3 | 79.5±0.4 |
| Gemini 3 Pro | 41.3±0.3 | 85.1±0.4 |
| Qwen3-8B | 18.6±0.7 | 54.4±2.1 |
| Qwen3-32B | 26.0±0.4 | 65.8±0.7 |
| DeepSeek-R1 | 23.3±0.6 | 69.0±0.5 |
| **HumanLLM-8B** | **25.7±0.4** | **70.3±0.6** |
| **HumanLLM-32B** | **32.8±0.3** | **73.6±0.4** |

### Ablation Study

**Data Ablation (8B Variant)**

| Configuration | IPE | MPD |
| :--- | :--- | :--- |
| Qwen3-8B (base) | 18.6 | 54.4 |
| Qwen3-8B (OT+CoSER, w/o HumanLLM data) | 9.1 | 31.3 |
| HumanLLM-8B (Full) | 25.7 | 70.3 |

**Evaluation Framework Alignment Validation (100 Scenarios)**

| Metric Type | Human | LLM | $\Delta$ | Correlation $r$ |
| :--- | :--- | :--- | :--- | :--- |
| Anthropomorphism (Holistic) | 84.6 | 53.8 | -30.8 | 0.43 |
| Character Fidelity (Holistic) | 83.1 | 65.4 | -17.7 | 0.61 |
| **IPE (checklist)** | **38.4** | **37.8** | **-0.6** | **0.90** |
| **MPD (checklist)** | **72.1** | **75.8** | **+3.7** | **0.88** |

### Key Findings

*   HumanLLM-8B outperforms Qwen3-32B in MPD (70.3% vs. 65.8%); the 4x parameter difference demonstrates that psychological training data is more critical than model scale.
*   GPT-5 performed unexpectedly low (IPE: 15.5%); analysis shows its strong instruction-following tendency leads to overly literal role-playing—general capabilities do not automatically transfer to psychological simulation.
*   Negative Transfer: Training only on OpenThoughts+CoSER caused a significant drop in performance (IPE: 18.6 $\rightarrow$ 9.1); general data inhibited the model's ability to express psychological patterns. HumanLLM data not only compensated for this negative transfer but also produced synergistic effects.
*   Traditional holistic metrics suffer from "normative confusion"—LLM judges equate social desirability with simulation accuracy. The checklist method effectively decouples the two.

## Highlights & Insights

*   Modeling psychological patterns as "interacting causal forces" rather than "isolated labels" is a significant conceptual breakthrough—this perspective is generalizable to any application requiring multi-dimensional personality simulation (game NPCs, social simulation, psychological counseling training).
*   The discovery of normative confusion has methodological value—revealing systematic biases in LLM-as-Judge when evaluating human behavior simulation; the checklist method provides a reusable solution.
*   The discovery of negative transfer has direct implications for SFT data ratios—general data may "drown out" domain-specific capabilities, requiring anchoring data (like HumanLLM) to maintain performance.

## Limitations & Future Work

*   Dialogues average 16.4 turns; long-term character consistency (50+ turns) was not evaluated.
*   Psychological theories are primarily derived from WEIRD populations; cross-cultural applicability is uncertain—e.g., conformity pressure may manifest differently in collectivist cultures.
*   Training data is entirely LLM-synthesized; a gap still exists between synthetic data and real human interaction.
*   High-fidelity simulation of negative traits (e.g., manipulation, bias) poses safety and ethical risks—deployment requires additional safety layers.

## Related Work & Insights

*   **vs. CoSER**: CoSER extracts dialogues from 771 books and uses holistic metrics. HumanLLM constructs psychological patterns from academic papers and uses checklists—achieving $r=0.90$ human alignment vs. CoSER's $r=0.43$.
*   **vs. Character-LLM**: Trains historical figure agents via experience reconstruction, focusing on single characters. HumanLLM focuses on general cognitive patterns rather than specific characters, offering better generalizability.
*   **vs. Persona Vectors (Chen et al.)**: Manipulates traits via activation steering but cannot handle multi-trait conflicts. HumanLLM implicitly learns multi-pattern dynamics through scenario-based training.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ Framing cognitive patterns as interacting causal forces + pattern library supported by 12,000 papers + discovery of normative confusion.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-baseline comparisons + ablation + external benchmarks + human alignment validation + normative confusion case studies.
*   Writing Quality: ⭐⭐⭐⭐ Clear framework, natural connection between psychological theory and engineering implementation, though the paper is long.
*   Value: ⭐⭐⭐⭐⭐ Provides a paradigm shift for LLM personality simulation from label mapping to cognitive modeling; the dataset and evaluation framework are independently reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] The Silent Vote: Improving Zero-Shot LLM Reliability by Aggregating Semantic Neighborhoods](the_silent_vote_improving_zero-shot_llm_reliability_by_aggregating_semantic_neig.md)
- [\[ACL 2026\] Collapse of Correct Beliefs: A Study on LLM Cognitive Resilience Under Clinical Stress](when_correct_beliefs_collapse_epistemic_resilience_of_llms_under_clinical_pressu.md)
- [\[ACL 2026\] Modeling Multi-Dimensional Cognitive States in Large Language Models under Cognitive Crowding](modeling_multi-dimensional_cognitive_states_in_large_language_models_under_cogni.md)
- [\[ACL 2026\] HoWToBench: Holistic Evaluation for LLM's Capability in Human-level Writing using Tree of Writing](howtobench_holistic_evaluation_for_llms_capability_in_human-level_writing_using_.md)
- [\[ACL 2026\] Fin-Bias: Comprehensive Evaluation for LLM Decision-Making under human bias in Finance Domain](fin-bias_comprehensive_evaluation_for_llm_decision-making_under_human_bias_in_fi.md)

</div>

<!-- RELATED:END -->
