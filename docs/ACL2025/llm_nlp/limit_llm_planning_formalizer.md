---
title: >-
  [Paper Note] On the Limit of Language Models as Planning Formalizers
description: >-
  [ACL 2025][LLM (Other)][PDDL] This study systematically evaluates the limits of the "LLM-as-Formalizer" methodology. For the first time, LLMs are required to generate complete PDDL representations (rather than partial ones) to formalize planning domains from textual descriptions of varying levels of naturalness. The strongest models (GPT-4o/o3-mini/DeepSeek-R1) can effectively formalize, outperforming direct planning, but performance decreases as descriptions become more natu…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "PDDL"
  - "planning formalization"
  - "LLM reasoning"
  - "neuro-symbolic"
  - "natural language to formal language"
date: 2026-05-08
content_hash: d84039ce980dd6df
---

# On the Limit of Language Models as Planning Formalizers

**Conference**: ACL 2025  
**arXiv**: [2412.09879](https://arxiv.org/abs/2412.09879)  
**Code**: [https://github.com/CassieHuang22/llm-as-pddl-formalizer](https://github.com/CassieHuang22/llm-as-pddl-formalizer)  
**Area**: LLM/NLP  
**Keywords**: PDDL, planning formalization, LLM reasoning, neuro-symbolic, natural language to formal language  

## TL;DR
This study systematically evaluates the limits of the "LLM-as-Formalizer" methodology. For the first time, LLMs are required to generate complete PDDL representations (rather than partial ones) to formalize planning domains from textual descriptions of varying levels of naturalness. The strongest models (GPT-4o/o3-mini/DeepSeek-R1) can effectively formalize, outperforming direct planning, but performance decreases as descriptions become more natural. Weak models struggle with syntax errors, whereas strong models face semantic errors.

## Background & Motivation
**Background**: LLMs can perform informal planning (e.g., "party recommendations") but cannot generate executable and verifiable formal plans. Recent research utilizes LLMs as "formalizers"—translating natural language descriptions into PDDL (Planning Domain Definition Language) and then using deterministic solvers to find plans.

**Limitations of Prior Work**: (a) Prior work only required LLMs to generate partial PDDL (e.g., generating only problem files given a domain file), which is unrealistic since real-world users do not provide existing PDDL components. (b) The textual descriptions used are highly templated, serving almost as verbatim translations of PDDL, which does not represent how users actually describe tasks. (c) "Success" under these simplified settings might be overestimated.

**Key Challenge**: LLMs exhibit strong information extraction capabilities but weak formal reasoning capabilities. Formalization (extracting information into structures) should theoretically suit LLMs better than direct planning (reasoning and searching). However, how well can they actually perform, and at what level of "naturalness" in descriptions do they remain effective?

**Goal**: To systematically evaluate the capabilities and limits of the LLM-as-Formalizer paradigm under reference-level rigorous and realistic settings (generating complete PDDL from descriptions of varying naturalness).

**Key Insight**: By constructing descriptions at three levels of naturalness (highly templated, moderately templated, natural language) and evaluating multiple LLMs across four planning domains.

**Core Idea**: Using LLMs for formalization outperforms direct planning, but natural descriptions remain a bottleneck. Weak models suffer from syntax errors, whereas strong models encounter semantic errors.

## Method

### Overall Architecture
(1) Construct datasets with three levels of description naturalness (highly templated → moderately templated → natural) across four planning domains (BlocksWorld, Logistics, Barman, and MysteryBlocksWorld). (2) Compare two paradigms of LLM usage: LLM-as-Planner (directly generating plans) and LLM-as-Formalizer (generating PDDL and utilizing a solver to find plans). (3) Evaluate multiple LLMs regarding solvability and correctness.

### Key Designs

1. **Construction of Descriptions with Three Levels of Naturalness**:

    - **Highly Templated**: Almost verbatim translations of PDDL—listing all preconditions and effects while utilizing predicate terminology.
    - **Moderately Templated**: More natural but still explicitly discusses preconditions and effects.
    - **Natural Language**: Simulates real-world users—generated through human-model collaboration, describing general rules instead of listing rules in exhaustive sequences, and omitting readily inferable information.
    - Design Motivation: To test the formalization ability of LLMs under different levels of information expliciteness.

2. **Complete PDDL Generation**:

    - Function: Requiring LLMs to generate both domain files ($\mathbb{DF}$) and problem files ($\mathbb{PF}$).
    - Difference from prior work: Prior work only generated partial files (e.g., generating only $\mathbb{PF}$), assuming $\mathbb{DF}$ was already provided.
    - Design Motivation: Complete generation is closer to real-world usage scenarios.

3. **MysteryBlocksWorld Control Experiment**:

    - Function: Replacing all entities, predicates, and action names with meaningless words to test whether LLM performance relies on lexical memorization.
    - Design Motivation: Successful formalization on the obfuscated version demonstrates that the model understands structure rather than merely memorizing PDDL for BlocksWorld.

### Loss & Training
- **Pure evaluation study**—no training involved.
- Evaluate 11 LLMs: Gemma-2 (9B/27B), LLaMA-3.1 (8B/70B/405B), DeepSeek-R1 series, GPT-4o, and o3-mini.
- Metrics: Solvability (whether the generated PDDL can be parsed by a solver) and correctness (correctness of the final plan).

## Key Experimental Results

### Main Results (BlocksWorld-100, Highly Templated)

| Model | Planner Accuracy | Formalizer Accuracy | Note |
|------|-------------|----------------|------|
| Gemma-2-9B/27B | ~0% | ~0% | Too weak; both failed |
| LLaMA-3.1-8B/70B | ~0% | ~0-5% | Insufficient capability |
| GPT-4o-mini | Low-Medium | Medium-High | Formalizer performs better |
| **GPT-4o** | Medium | **High** | Formalizer significantly outperforms Planner |
| **o3-mini** | High | **Highest** | Strongest reasoning model |
| **DeepSeek-R1** | High | **High** | Strongest open-source model |

### Impact of Naturalness

| Description Type | Strongest Model Accuracy | Note |
|---------|-------------|------|
| Highly Templated | Highest | Complete and explicit information |
| Moderately Templated | Medium-High | Partial information requires inference |
| **Natural Language** | **Significant Decline** | Substantial information requires inference |

### Key Findings
- **Formalizer often outperforms Planner**—but not always: on certain model-data combinations, Planner may perform better.
- **Natural descriptions significantly increase difficulty**—the performance drop from templated to natural descriptions reflects the challenge of "information inference."
- **Formalizer is more robust to lexical distributions**—in MysteryBlocksWorld, Planner performance drops sharply while Formalizer maintains better performance.
- Weak models (<20B) are largely unable to complete full PDDL generation, getting stuck on syntax errors.
- Errors in strong models are primarily semantic—the generated code is syntactically correct, but preconditions/effects are inaccurately defined.
- Reasoning models (o3-mini, DeepSeek-R1) perform best under both paradigms.

## Highlights & Insights
- **"LLMs perform better at formalization than planning"** is verified under rigorous settings—but this superiority is conditional on model capability and description quality.
- **The gradient decline in performance relative to naturalness** quantifies the cost of "information inference"—this serves as an important warning for practical applications, as real users rarely write in PDDL-like styles.
- **The MysteryBlocksWorld control experiment** is neat and effective, confirming that the Formalizer relies more on structural understanding than lexical memory.
- **The stratification of model capabilities** into syntax versus semantic errors offers practical value, helping diagnose bottlenecks in the formalization pipeline.
- This work provides significant reference value for neuro-symbolic AI research combining LLMs with symbolic planning.

## Limitations & Future Work
- The four planning domains evaluated are classic IPC domains, whereas real-world planning is significantly more complex and open-ended.
- The "natural" descriptions were still model-generated and human-verified, which might not fully mirror real-world user phrasing.
- Only zero-shot generation is evaluated, without exploring iterative corrections (e.g., self-correcting PDDL based on solver feedback).
- Cost-efficiency is not evaluated—Formalizer requires additional execution steps with solvers.

## Related Work & Insights
- **vs Liu et al. (2023) / Guan et al. (2023)**: Prior works only generated partial PDDL; this work is the first to require full PDDL generation.
- **vs Valmeekam et al. (2024)**: Valmeekam et al. demonstrated the weak direct planning capability of LLMs; this work proves that LLM formalization capability is significantly better.
- **vs DPT-Agent**: DPT-Agent uses FSMs for fast decision-making without involving formal planning; this work focuses on translating natural language into formal planning representations, addressing different levels of "planning."
- This offers direct guidance for developing LLM-assisted planning systems: prioritizing the Formalizer pipeline over the Planner pipeline.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic evaluation under complete PDDL and multiple levels of naturalness, with rigorous experimental design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 domains × 3 levels of naturalness × 11 models × 2 paradigms + control experiment + detailed error analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Precise problem formulation, clear methodology, and well-structured findings.
- Value: ⭐⭐⭐⭐⭐ Substantial contribution to the fundamental understanding of neuro-symbolic AI and LLM planning capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Interactive and Expressive Code-Augmented Planning with Large Language Models](interactive_and_expressive_code-augmented_planning_with_large_language_models.md)
- [\[ACL 2025\] Planning-Driven Programming: A Large Language Model Programming Workflow](planning-driven_programming_a_large_language_model_programming_workflow.md)
- [\[ACL 2025\] PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](plangenllms_planning_survey.md)
- [\[ACL 2025\] ASPERA: A Simulated Environment to Evaluate Planning for Complex Action Execution](aspera_a_simulated_environment_to_evaluate_planning_for_complex_action_execution.md)
- [\[ACL 2025\] Can We Further Elicit Reasoning in LLMs? Critic-Guided Planning with Retrieval-Augmentation for Solving Challenging Tasks](can_we_further_elicit_reasoning_in_llms_critic-guided_planning_with_retrieval-au.md)

</div>

<!-- RELATED:END -->
