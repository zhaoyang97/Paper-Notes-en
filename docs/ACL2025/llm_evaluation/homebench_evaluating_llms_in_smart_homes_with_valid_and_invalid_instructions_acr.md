---
title: >-
  [Paper Note] HomeBench: Evaluating LLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multiple Devices
description: >-
  [ACL 2025][LLM Evaluation][Smart Home] This paper introduces HomeBench, the first evaluation benchmark for smart-home LLMs that incorporates both valid and invalid instructions across single- and multi-device scenarios. The study reveals that even GPT-4o achieves a success rate of only 0.0% in multi-device scenarios involving invalid instructions.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Smart Home"
  - "Invalid Instructions"
  - "Multi-device Control"
  - "Benchmark"
date: 2026-05-08
content_hash: d9db65e93ddedd11
---

# HomeBench: Evaluating LLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multiple Devices

**Conference**: ACL 2025  
**arXiv**: [2505.19628](https://arxiv.org/abs/2505.19628)  
**Code**: [https://github.com/BITHLP/HomeBench](https://github.com/BITHLP/HomeBench)  
**Area**: LLM Evaluation  
**Keywords**: Smart Home, LLM Evaluation, Invalid Instructions, Multi-device Control, Benchmark

## TL;DR

This paper introduces HomeBench, the first evaluation benchmark for smart-home LLMs that incorporates both valid and invalid instructions across single- and multi-device scenarios. The study reveals that even GPT-4o achieves a success rate of only 0.0% in multi-device scenarios involving invalid instructions.

## Background & Motivation

**Background**: Large Language Models (LLMs) are being utilized to build smarter home assistant systems. Related research has explored how to integrate LLMs into smart homes to understand user needs and respond appropriately. Existing smart-home datasets and evaluation benchmarks primarily focus on valid, single-device operation instructions, such as "turn on the living room light."

**Limitations of Prior Work**: Real-world smart home scenarios are far more complex than those covered by existing benchmarks. Users frequently issue invalid instructions (e.g., "set the AC to 50 degrees" — exceeding device capability; "turn on the second-floor light" — when there is no second floor) and control multiple devices simultaneously (e.g., "turn off all lights and set the AC to 25 degrees"). Existing benchmarks completely ignore these realistic challenges.

**Key Challenge**: LLMs must not only correctly execute valid instructions but also possess the ability to identify and correct invalid instructions. In multi-device scenarios, this challenge is magnified: the model must independently judge the validity of each instruction and execute it correctly. An error in any single sub-instruction leads to overall failure.

**Goal**: To build a comprehensive smart-home evaluation benchmark covering four scenarios: (1) valid single-device instructions, (2) invalid single-device instructions, (3) valid multi-device instructions, and (4) invalid multi-device instructions, in order to systematically evaluate the performance of current LLMs across all scenarios.

**Key Insight**: Extracting typical patterns of invalid instructions from user daily interactions, combined with the demand for concurrent multi-device control, to construct a hierarchical evaluation framework.

**Core Idea**: By introducing invalid instructions and multi-device dimensions, this work exposes the true performance ceiling of current LLMs in smart home applications.

## Method

### Overall Architecture

The construction process of HomeBench consists of three stages: (1) defining smart home devices and their attribute schemas (device types, controllable attributes, and valid ranges); (2) generating four categories of instructions based on device schemas—valid single-device, invalid single-device, valid multi-device, and invalid multi-device; and (3) designing an evaluation protocol that requires models to convert user instructions into structured device operation sequences and evaluates their correctness. The evaluation metric employs a strict exact match Success Rate, meaning a trial is considered successful only if all sub-operations are correct.

### Key Designs

1. **Classification System of Invalid Instructions**:

    - **Function**: To systematically cover the types of invalid instructions that may occur in smart homes.
    - **Mechanism**: Classifying invalid instructions into multiple types, including: (a) exceeding device capabilities (e.g., temperature out of adjustable range); (b) referencing non-existent devices or rooms; (c) logically contradictory instructions (e.g., asking to turn a light on and off at the same time); and (d) incorrect attribute value types. For each type of invalid instruction, the correct response that the model should generate is defined (e.g., refusing execution and providing correction suggestions).
    - **Design Motivation**: User errors are ubiquitous in real-world scenarios. A qualified smart home assistant must handle these situations gracefully rather than executing them blindly.

2. **Multi-device Instruction Generation and Evaluation**:

    - **Function**: To evaluate the model's capability to process multiple instructions simultaneously.
    - **Mechanism**: Multi-device instructions consist of 2–5 independent sub-instructions, each targeting a different device. In invalid multi-device scenarios, the set of sub-instructions contains at least one invalid instruction. The evaluation adopts an exact-match standard: the model must correctly handle all sub-instructions (including correctly executing valid ones and correctly rejecting invalid ones) to be counted as a success.
    - **Design Motivation**: Although strict, the exact-match standard reflects the real-world demand of smart homes: any erroneous operation can lead to poor user experience or safety hazards.

3. **Evaluation of Enhancement Strategies (ICL/RAG/Fine-tuning)**:

    - **Function**: To assess whether existing enhancement strategies can compensate for the shortcomings of LLMs on this task.
    - **Mechanism**: Beyond zero-shot evaluation, three strategies are systematically tested: In-Context Learning (ICL, providing examples), Retrieval-Augmented Generation (RAG, retrieving similar instruction cases), and Fine-tuning. ICL provides a few labeled instruction-operation pairs as context; RAG constructs an instruction library and retrieves the most relevant cases during inference; Fine-tuning directly trains the model on labeled data.
    - **Design Motivation**: If enhancement strategies can resolve the issue, it is beneficial; if not, it further highlights the challenging nature of the task.

### Loss & Training

Fine-tuning experiments utilize the standard instruction fine-tuning paradigm, training the models on annotated instruction-operation pairs to generate correct operation sequences.

## Key Experimental Results

### Main Results

| Model | Valid Single-Device | Invalid Single-Device | Valid Multi-Device | Invalid Multi-Device |
|------|-----------|-----------|-----------|-----------|
| GPT-4o | High | Medium | Medium-Low | **0.0%** |
| GPT-4 | High | Medium | Medium-Low | Extremely Low |
| Claude 3.5 | High | Low | Low | Extremely Low |
| LLaMA 3.1 | Medium | Low | Low | Extremely Low |
| Qwen2 | Medium | Low | Low | Extremely Low |
| 8 Other Models | Medium-Low | Low | Low | Extremely Low |

### Ablation Study

| Strategy | Valid Single-Device | Invalid Single-Device | Valid Multi-Device | Invalid Multi-Device |
|------|-----------|-----------|-----------|-----------|
| Zero-shot | Baseline | Baseline | Baseline | Baseline (0%) |
| + ICL | Marginal Improvement | Marginal Improvement | Limited Improvement | Extremely Low |
| + RAG | Marginal Improvement | Limited Improvement | Limited Improvement | Extremely Low |
| + Fine-tuning | Significant Improvement | Moderate Improvement | Moderate Improvement | Still Extremely Low |

### Key Findings

- **GPT-4o achieves a 0.0% success rate in invalid multi-device scenarios**: This is the most striking finding. Even the strongest commercial model fails completely when faced with multi-device control scenarios containing invalid instructions. The reason is that the model needs to independently judge the validity of each sub-instruction while generating a fully correct operation sequence, and a failure in any single step leads to an overall failure.
- **Identifying invalid instructions is the core bottleneck**: All models suffer a significant performance drop when transitioning from valid to invalid scenarios, indicating that LLMs lack a deep understanding of device constraints and physical common sense.
- **Enhancement strategies show limited effectiveness**: Although ICL, RAG, and fine-tuning help in simple scenarios, they have negligible impact in invalid multi-device scenarios, indicating that this is not a simple data problem but a fundamental defect in model capability.
- **A clear gap exists between open-source and commercial models**: The gap is even more pronounced in complex scenarios.

## Highlights & Insights

- **The shocking result of 0% success rate** makes the contribution of this work clear at a glance: it explicitly informs the community that current LLMs still have a long way to go before becoming reliable smart home assistants. This "problem-exposing" type of benchmark is often more driving than "progress-showing" work.
- **The completeness of the evaluation design** is worth learning from: the four-quadrant division (valid/invalid $\times$ single-device/multi-device) is both concise and comprehensive, serving as a paradigm for constructing evaluation benchmarks in similar domains.
- **The sharp drop in performance from valid to invalid scenarios** reveals a deeper issue: LLMs are good at pattern matching but lack constraint reasoning capabilities. This insight can be transferred to other NLP tasks requiring rule compliance.

## Limitations & Future Work

- **Evaluation metrics are excessively strict**: While exact-match success rate reflects real-world needs, it might underestimate the model's partially correct capabilities. Introducing partial-match metrics could provide a more fine-grained analysis.
- **Limited device and scenario coverage**: Smart homes involve a wide variety of devices, and the current dataset might not cover all typical devices and interaction patterns.
- **Multi-turn dialogue is not considered**: In real-world scenarios, users may clarify their requirements step-by-step through dialogue; the current evaluation only considers single-turn instructions.
- **Lack of detailed analysis on error types**: Which types of invalid instructions are the hardest to detect? What is the distribution of different models' performance across various error types?
- **Future Directions**: Combining device knowledge graphs or formal constraint rules to enhance LLMs' reasoning capabilities regarding entity constraints.

## Related Work & Insights

- **vs Existing Smart Home Datasets**: Prior datasets (such as SmartHome-QA, etc.) only contain valid instructions and mostly single-device operations. HomeBench's introduction of the invalid instruction dimension is its core contribution.
- **vs General Tool-use Evaluations** (e.g., ToolBench): The difference between smart home tasks and general tool use lies in the need to understand physical constraints and device capability boundaries, which is not covered by existing tool-use evaluations.
- The evaluation framework proposed in this paper can be transferred to other LLM application scenarios requiring constraint satisfaction, such as database operations, API calls, robot control, etc.

## Rating

- Novelty: ⭐⭐⭐⭐ Mentions invalid instructions and multi-device joint control dimensions in smart home evaluation for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation of 13 models alongside comparisons of various enhancement strategies.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definitions and intuitive, impactful presentation of results.
- Value: ⭐⭐⭐⭐ Exposes critical capability deficiencies of LLMs in practical applications, offering direct reference value to the industry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] MultiCogEval: Evaluating LLMs Across Multi-Cognitive Levels](../../ICML2025/llm_evaluation/evaluating_llms_across_multi-cognitive_levels_from_medical_knowledge_mastery_to_.md)
- [\[ACL 2025\] EvoWiki: Evaluating LLMs on Evolving Knowledge](evowiki_evaluating_llms_on_evolving_knowledge.md)
- [\[ACL 2025\] From Tools to Teammates: Evaluating LLMs in Multi-Session Coding Interactions](from_tools_to_teammates_evaluating_llms_in_multi-session_coding_interactions.md)
- [\[ACL 2025\] WiCkeD: A Simple Method to Make Multiple Choice Benchmarks More Challenging](wicked_a_simple_method_to_make_multiple_choice_benchmarks_more_challenging.md)
- [\[ACL 2025\] WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models](wximpactbench_a_disruptive_weather_impact_understanding_benchmark_for_evaluating.md)

</div>

<!-- RELATED:END -->
