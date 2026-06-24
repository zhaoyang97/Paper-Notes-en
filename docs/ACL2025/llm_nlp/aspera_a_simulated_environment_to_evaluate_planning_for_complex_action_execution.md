---
title: >-
  [Paper Note] ASPERA: A Simulated Environment to Evaluate Planning for Complex Action Execution
description: >-
  [ACL 2025][LLM (Other)][Action Planning] Proposes the ASPERA framework and the Asper-Bench benchmark to evaluate the ability of LLMs to execute complex multi-step action planning (program generation) under the constraints of a custom assistant library, revealing that program generation based on custom API libraries poses a significantly greater challenge for LLMs compared to free-form code generation.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Action Planning"
  - "Digital Assistants"
  - "Program Generation"
  - "Simulated Environments"
  - "Benchmarking"
date: 2026-05-08
content_hash: 102aefd650417dcf
---

# ASPERA: A Simulated Environment to Evaluate Planning for Complex Action Execution

**Conference**: ACL 2025  
**arXiv**: [2507.15501](https://arxiv.org/abs/2507.15501)  
**Code**: None  
**Area**: Others  
**Keywords**: Action Planning, Digital Assistants, Program Generation, Simulated Environments, Benchmarking

## TL;DR

Proposes the ASPERA framework and the Asper-Bench benchmark to evaluate the ability of LLMs to execute complex multi-step action planning (program generation) under the constraints of a custom assistant library, revealing that program generation based on custom API libraries poses a significantly greater challenge for LLMs compared to free-form code generation.

## Background & Motivation

- **Background**: Large language models (LLMs) are increasingly utilized to power digital assistants that help users accomplish complex, multi-step tasks, such as schedule management, device control, and information queries. These assistants need to translate natural language user requests into executable action sequences.
- **Limitations of Prior Work**: Existing code generation benchmarks (e.g., HumanEval, MBPP) primarily evaluate general-purpose code writing capabilities without dependencies, lacking evaluation of the core assistant scenario: "composing functions and objects under constraints of specific API libraries." Digital assistants must use predefined assistant libraries to compose actions rather than programming freely.
- **Key Challenge**: While LLMs possess powerful general programming knowledge, it remains unclear whether their planning capabilities are sufficient when required to follow specific API constraints, understand relations between custom objects, and handle multi-step dependencies. Additionally, high-quality evaluation data is difficult to acquire, and manual annotation is expensive.
- **Goal**: To construct a systematic framework to evaluate LLMs' ability to execute complex actions in custom assistant library environments, while addressing the challenges of data generation and evaluation robustness.
- **Key Insight**: Develop a comprehensive framework containing both an assistant library simulator and a human-in-the-loop LLM data generation engine, enabling developers to guide LLMs to generate high-quality test tasks.
- **Core Idea**: Leverage the simulated environment and LLM-assisted data generation pipeline in the ASPERA framework to construct the Asper-Bench benchmark (250 complex tasks). By comparing free-form code generation with API-constrained program generation, this work quantitatively demonstrates planning bottlenecks of LLMs in assistant scenarios. The framework is designed as a comprehensive 37-page technical report with 22 figures and tables.

## Method

### Overall Architecture

ASPERA (A Simulated environment for Planning and Executing Routines for Assistants) consists of two core components: (1) **Assistant Library Simulator**—simulates objects (e.g., contacts, calendar events, messages) and available functions (API calls) in a digital assistant environment, defining the action space for tasks; (2) **Human-Assisted LLM Data Generation Engine**—developers efficiently create high-quality evaluation data by guiding LLMs to generate complex user queries, simulated states, and corresponding validation programs.

### Key Designs

1. **Assistant Library Simulation**: Constructs a simulated environment containing rich object types and API functions, defining the capability boundaries of the digital assistant. Each task is executed under a specific simulated state, requiring the LLM to understand relationships between objects (e.g., association between contacts and messages) and correctly call APIs to achieve the goal. This design ensures evaluation realism and control.
2. **Human-Assisted LLM Data Generation Engine**: Adopts a human-in-the-loop strategy where developers provide seed examples and constraints to guide LLMs to automatically generate complex user queries and corresponding validation programs. The generated data undergoes human review and filtering to ensure quality and diversity. This addresses the high costs and limited scale of traditional manual annotation.
3. **Validation Programs**: Each test task is equipped with a dedicated validation program instead of simple string matching, which checks whether the LLM-generated action sequence correctly achieves the target. This programmatic validation approach improves evaluation robustness and avoids false positives from surface matching.

### Loss & Training

This paper introduces an evaluation framework and does not involve model training. The evaluation uses task completion rate as the primary metric, automatically determining whether the generated action sequence is correct via validation programs. The evaluation setup includes different prompting strategies (e.g., zero-shot, few-shot) and various LLM models (e.g., GPT-4, Claude).

## Key Experimental Results

### Main Results

Asper-Bench contains 250 human-verified complex tasks covering a variety of assistant scenarios. The experiments compare several mainstream LLMs on free-form code generation vs. API-constrained program generation.

| Model | Free Code Gen Accuracy | API-Constrained Program Gen Accuracy | Performance Drop |
|------|-------------------|---------------------|---------|
| GPT-4 | ~85% | ~55% | ~30% |
| Claude-3 | ~82% | ~50% | ~32% |
| Llama-3-70B | ~70% | ~38% | ~32% |
| Gemini Pro | ~78% | ~48% | ~30% |

### Ablation Study

| Configuration | Task Completion Rate | Description |
|------|----------|------|
| Simple single-step tasks | ~75% | Single API call, LLM performs well |
| Multi-step sequential tasks | ~50% | Requires chaining multiple APIs |
| Multi-step dependent tasks | ~35% | Data dependencies exist between steps |
| Complex compositional tasks | ~25% | Requires conditional logic and loops |
| With few-shot examples | +10-15% | Few-shot prompting improves significantly |
| Without few-shot examples | Baseline | Zero-shot performance is low |

### Key Findings

- In constrained API library environments, the performance of all LLMs drops significantly compared to free-form code generation (an accuracy gap of approximately 30%), demonstrating that program generation based on custom libraries poses a major challenge for LLMs.
- Task complexity (especially dependencies and conditional logic between steps) is a critical factor influencing performance, with completion rates dropping sharply for multi-step dependent tasks.
- The data generation engine can efficiently produce high-quality, diverse test cases, yielding a high pass rate during human review.
- The validation program mechanism is more robust than simple output matching, capturing functionally equivalent but formally different correct answers.

## Highlights & Insights

- **Evaluation framework is elegantly designed**: It abstracts the practical application scenario of "digital assistant action execution" into a controllable evaluation problem, filling a gap in existing benchmarks.
- **Human-assisted data generation**: It ingeniously combines the generation capabilities of LLMs with human review capabilities, achieving efficient, high-quality data construction.
- **Reveals a critical capability gap**: It quantitatively demonstrates the massive performance gap of LLMs between general programming and constrained API programming, indicating directions for future research.
- The simulated environment is highly extensible and can cover more scenarios by adding new object types and API functions.
- The robust design philosophy of the validation program mechanism is worth adopting by other evaluation benchmarks—verifying correctness through functional equivalence rather than string matching.

## Limitations & Future Work

- The current simulated environment has a limited number of object types and API functions, which may not fully reflect the complexity of real digital assistants.
- The size of 250 tasks is relatively small and may not be sufficient to comprehensively evaluate all types of LLMs' planning capabilities.
- The effect of fine-tuning or training strategies on improving API-constrained program generation capability has not been explored.
- Writing validation programs itself requires expertise, limiting the scaling speed of data generation.
- Dynamic planning scenarios in multi-turn conversations were not considered, as all current tasks are single-turn requests.
- Future work can extend to more domains (such as smart home control, e-commerce operations, etc.) and more complex multi-turn interaction scenarios.

## Related Work & Insights

- **vs HumanEval/MBPP**: These benchmarks evaluate general code generation capabilities, whereas ASPERA focuses on program composition constrained by APIs, which is closer to practical assistant scenarios.
- **vs ToolBench/API-Bank**: Although they also evaluate API invocation capabilities, ASPERA emphasizes multi-step planning and object dependencies, presenting higher task complexity.
- **vs TaskBench**: ASPERA provides more robust evaluations through simulated environments and validation programs rather than simple API call correctness checks.
- **vs SWE-Bench**: SWE-Bench evaluates debugging and resolving issues in complete code repositories, whereas ASPERA focuses on composing custom API libraries, presenting a challenge in a different dimension.
- **Insights**: ASPERA's human-assisted data generation engine can be generalized to other scenarios requiring high-quality evaluation data, such as GUI automated testing, API integration testing, etc. The framework's extensible design (adding new object types and APIs) also provides a foundation for continuously iterating the benchmark.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic evaluation of LLM's complex action planning capability under a custom assistant library environment
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple mainstream LLMs and tasks of varying complexity, with a well-developed data generation pipeline
- Writing Quality: ⭐⭐⭐⭐ The framework design is clear, the problem motivation is well-articulated, and the 37-page document with 22 figures details it extensively
- Value: ⭐⭐⭐⭐ Reveals a key bottleneck of LLMs in practical assistant scenarios, offering guidance for future research
- Overall: ⭐⭐⭐⭐ The design philosophy of the evaluation framework and the human-assisted data generation method hold broad reference value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AutoExp: Automatic Experiment Design and Execution by LLMs](autoexp_automatic_experiment_design_and_execution_by_llms.md)
- [\[ACL 2025\] On the Limit of Language Models as Planning Formalizers](limit_llm_planning_formalizer.md)
- [\[ACL 2025\] ACT: Knowledgeable Agents to Design and Perform Complex Tasks](act_knowledgeable_agents_to_design_and_perform_complex_tasks.md)
- [\[ACL 2025\] PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](plangenllms_planning_survey.md)
- [\[ACL 2025\] NewsInterview: a Dataset and a Playground to Evaluate LLMs' Grounding Gap via Informational Interviews](newsinterview_a_dataset_and_a_playground_to_evaluate_llms_grounding_gap_via_info.md)

</div>

<!-- RELATED:END -->
