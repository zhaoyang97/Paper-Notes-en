---
title: >-
  [Paper Note] Theory of Mind in Large Language Models: Assessment and Enhancement
description: >-
  [ACL 2025][LLM (Other)][Theory of Mind] This paper presents a systematic survey of evaluation benchmarks (10+ story-based benchmarks) and enhancement strategies (prompt-only and fine-tuning methods) for the Theory of Mind (ToM) capabilities of LLMs, highlighting that current LLMs still fall significantly short in ToM reasoning and outlining future directions.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Theory of Mind"
  - "ToM"
  - "benchmark survey"
  - "belief reasoning"
  - "LLM cognitive abilities"
date: 2026-05-08
content_hash: dd1f5dd550ae8f37
---

# Theory of Mind in Large Language Models: Assessment and Enhancement

**Conference**: ACL 2025  
**arXiv**: [2505.00026](https://arxiv.org/abs/2505.00026)  
**Code**: None  
**Area**: LLM / NLP  
**Keywords**: Theory of Mind, ToM, benchmark survey, belief reasoning, LLM cognitive abilities

## TL;DR
This paper presents a systematic survey of evaluation benchmarks (10+ story-based benchmarks) and enhancement strategies (prompt-only and fine-tuning methods) for the Theory of Mind (ToM) capabilities of LLMs, highlighting that current LLMs still fall significantly short in ToM reasoning and outlining future directions.

## Background & Motivation
**Background**: Theory of Mind (ToM) is the cornerstone of human social intelligence, referring to the ability to attribute mental states such as beliefs, intentions, and emotions to others. As LLMs become increasingly integrated into daily life, evaluating their ToM capability is crucial.

**Limitations of Prior Work**: There is an ongoing controversy regarding whether LLMs truly possess ToM. Some studies suggest that LLMs exhibit signs of ToM (Kosinski 2024), while more recent research indicates that these capabilities are superficial and unstable (Shapira et al. 2024; Ullman 2023).

**Key Challenge**: A proliferation of ToM evaluation benchmarks and enhancement methods emerged during 2023-2024, yet there is a lack of a unified survey to systematically organize them. Previous surveys (Ma et al. 2023b) only covered benchmarks up to 2023 and did not encompass enhancement strategies.

**Goal**: (1) Systematically survey story-based ToM benchmarks and multimodal benchmarks from the past two years; (2) Categorize and summarize strategies for enhancing LLM ToM capabilities; (3) Outline future research directions.

**Key Insight**: A unified comparative analysis of benchmarks and methods is conducted based on the ATOMS framework (7 mental states: beliefs, intentions, desires, emotions, knowledge, percepts, non-literal communication).

**Core Idea**: This is the first survey to systematically cover both evaluation and enhancement of ToM in LLMs, revealing that current research disproportionately focuses on belief reasoning while studies on other mental states remain severely insufficient.

## Method

### Overall Architecture
This paper is a survey organized in the structure of "Evaluation $\rightarrow$ Enhancement $\rightarrow$ Future Directions". The evaluation section covers 10+ story-based benchmarks (textual + multimodal), compared and categorized under the 7 ATOMS mental states. The enhancement section is divided into prompt-only methods and methods integrated with extra techniques (such as fine-tuning).

### Key Designs

1. **Evaluation Benchmark Analysis**:

    - **Function**: Systematically compares 10+ benchmarks such as ToMi, HI-TOM, TOMBENCH, BigToM, and OpenToM across multiple dimensions, including mental state coverage, reasoning order (first/second-order), and data formats.
    - **Key Findings**: The majority of benchmarks focus on belief reasoning, with TOMBENCH being the most comprehensive (covering 5 out of 7 mental states). Multimodal benchmarks (MMToM-QA, MuMA-ToM) are restricted to synthetic videos set in household environments.
    - **Key Issue**: Most existing benchmarks are 'passive' evaluations (where the LLM acts as an observer) and lack assessments in active decision-making scenarios.

2. **Prompt-only Enhancement Strategies**:

    - **Function**: Reviews 4 methods that enhance ToM exclusively through prompt engineering.
    - **SymbolicToM**: Constructs a belief graph for each character and retrieves relevant belief subgraphs as prompts during inference. It theoretically handles belief problems of any order, but memory consumption scales exponentially with the order of reasoning.
    - **SimToM**: Inspired by human "Simulation Theory", it uses a two-stage framework: first performing perspective-taking (filtering information known to the character), and then answering questions based on the filtered story.
    - **PercepToM**: A three-stage pipeline: identifying the perceiver of information $\rightarrow$ extracting information perceivable by the target character $\rightarrow$ answering based on this information.
    - **TimeToM**: Introduces a timeline to build a "Temporal Belief State Chain" (TBSC), converting higher-order reasoning into first-order reasoning through the intersection of temporal beliefs.
    - **Design Motivation**: The common intuition of all these methods is to first identify the target character's scope of perception/knowledge and then answer based on this restricted information; the core challenge lies in accurately modeling the character's perspective.

3. **Enhancement Strategies with Extra Techniques**:

    - **Function**: Identifies 3 methods that introduce fine-tuning or symbolic reasoning.
    - **ToM-LM**: Uses LLMs for semantic parsing to translate ToM questions into symbolic forms, which are then verified using a model checker (SMCDEL), enhancing interpretability.
    - **BIP-ALM**: Integrates Bayesian Inverse Planning with LLMs to extract symbolic information from video and text, fine-tuning the LLM to predict action likelihood.
    - **LIMP**: Appraises multi-agent scenarios, utilizing VLMs to extract video information and LLMs to extract textual information to infer mental states via inverse multi-agent planning.
    - **Design Motivation**: Symbolic reasoning improves verifiability but depends heavily on fine-tuning data, and currently has only been validated in multiple-choice settings.

### Key Assessment Conclusions on ToM Capabilities
- Most assessments indicate that LLMs still lack robust ToM capabilities.
- Belief is the most researched mental state, while studies on other mental states (intentions, desires, emotions, etc.) are severely lacking.
- Higher-order ToM reasoning (second-order and above) remains a significantly harder challenge for LLMs.
- Multimodal ToM evaluations are in their infancy, largely restricted to synthetic household environments.

## Key Experimental Results

### Main Results — Comparison of Benchmark Coverage

| Benchmark | Beliefs | Intentions | Desires | Emotions | Knowledge | Percepts | Non-literal |
|------|---------|-----------|---------|----------|-----------|---------|-------------|
| ToMi | ✓ | | | | | | |
| HI-TOM | ✓ | | | | | | |
| TOMBENCH | ✓ | ✓ | ✓ | ✓ | ✓ | | ✓ |
| BigToM | ✓ | ✓ | ✓ | | | | |
| OpenToM | ✓ | | ✓ | | ✓ | | |
| SimpleToM | ✓ | | ✓ | | ✓ | ✓ | |

### Comparison of Enhancement Methods

| Method | Type | Supported Order | Requires Fine-tuning | Applicable Scenario |
|------|------|---------|---------|---------|
| SymbolicToM | Prompt-only | Any order | No | Textual ToM |
| SimToM | Prompt-only | Primarily first-order | No | Textual ToM |
| TimeToM | Prompt-only | Higher-order | No | Textual ToM |
| ToM-LM | Extra Technique | First-order | Yes | Symbolic ToM |
| BIP-ALM | Extra Technique | First-order | Yes | Multimodal ToM |
| LIMP | Extra Technique | Second-order | No | Multi-agent |

### Key Findings
- All prompt-only methods adopt pipeline architectures, which are susceptible to error propagation.
- SymbolicToM's memory consumption grows exponentially with reasoning order; for TimeToM, the construction accuracy of the TBSC is the key bottleneck.
- Symbolic methods (e.g., ToM-LM) increase transparency and verifiability but require logic expertise to curate training data.
- Hallucinations of VLMs in action recognition constitute the main source of error for LIMP.

## Highlights & Insights
- **The ATOMS framework as a unified analytical lens**: Evaluating all benchmarks and methods across 7 mental state dimensions clearly reveals research biases—belief reasoning is dominant while other mental states are nearly blank. This points out clear directions for future research.
- **The reduction paradigm of 'higher-order reasoning $\rightarrow$ first-order reasoning' (TimeToM)** is highly clever: by utilizing the intersection of temporal beliefs, it decomposes $n$-order reasoning into a combination of multiple first-order reasonings, offering a highly reusable trick.
- **The distinction between passive and active evaluation** is valuable: all current benchmarks are passive (where the LLM acts as an observer), but true ToM requires the LLM to make decisions as an active agent in social scenarios.

## Limitations & Future Work
- The survey does not include experimental comparisons of various enhancement methods on a unified benchmark, as different methods utilize different benchmarks and base models.
- There is limited discussion on enhancement strategies for mental states other than belief (e.g., emotions, intentions).
- Multimodal ToM is currently limited to VirtualHome synthetic videos, presenting a substantial gap from real-world scenarios.
- The relationship between the ToM capabilities of LLMs and model scale is not discussed (from a scaling law perspective).

## Related Work & Insights
- **Comparison with the survey by Ma et al. (2023b)**: The previous survey only covered benchmarks prior to 2023 and excluded enhancement strategies. This paper provides a more comprehensive coverage, including 2023-2024 new benchmarks, enhancement methods, and multimodal aspects.
- **Connection to LLM Agent research**: ToM is a core capability for LLMs acting as social agents; however, current agent research primarily focuses on tool-use and planning, leaving the ToM dimension neglected.
- **Connection to Alignment research**: Understanding the LLM's capacity to reason about user intentions (a form of ToM) provides direct utility for improving alignment quality.

## Rating
- Novelty: ⭐⭐⭐ A survey paper with novel content organization but without technical innovation.
- Experimental Thoroughness: ⭐⭐⭐ Systematically organizes various methods but lacks unified experimental comparisons.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, tabular comparisons are intuitive, and future directions are valuable.
- Value: ⭐⭐⭐⭐ Serves as an excellent primer and reference resource for ToM researchers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] EnigmaToM: Improve LLMs' Theory-of-Mind Reasoning Capabilities with Neural Knowledge Base of Entity States](enigmatom_improve_llms_theory-of-mind_reasoning_capabilities_with_neural_knowled.md)
- [\[ACL 2026\] CoSToM: Causal-oriented Steering for Intrinsic Theory-of-Mind Alignment in Large Language Models](../../ACL2026/llm_nlp/costomcausal-oriented_steering_for_intrinsic_theory-of-mind_alignment_in_large_l.md)
- [\[ACL 2025\] Clue Guided Re-Assessment to Improve Reasoning in Large Language Models](clue_guided_re-assessment_to_improve_reasoning_in_large_language_models.md)
- [\[ACL 2025\] Enhancing Conversational Agents with Theory of Mind: Aligning Beliefs, Desires, and Intentions for Human-Like Interaction](enhancing_conversational_agents_with_theory_of_mind_aligning_beliefs_desires_and.md)
- [\[ACL 2025\] Knockout LLM Assessment: Using Large Language Models for Evaluations through Iterative Pairwise Comparisons](knockout_llm_assessment_using_large_language_models_for_evaluations_through_iter.md)

</div>

<!-- RELATED:END -->
