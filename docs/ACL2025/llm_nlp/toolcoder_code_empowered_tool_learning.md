---
title: >-
  [Paper Note] ToolCoder: A Systematic Code-Empowered Tool Learning Framework for Large Language Models
description: >-
  [ACL 2025][LLM (Other)][Tool learning] This paper proposes the ToolCoder framework, which reformulates tool learning as a code generation task. By drawing on software engineering principles (requirements analysis → modular design → implementation & execution → error debugging → code reuse), it enables LLMs to perform multi-step tool calls by generating and executing Python code, comprehensively outperforming baselines such as ReAct and CodeAct on RestBench and API-Bank.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Tool learning"
  - "Code generation"
  - "Software engineering"
  - "Function calling"
  - "Error reflection"
date: 2026-05-08
content_hash: 89cc71e242bff4a5
---

# ToolCoder: A Systematic Code-Empowered Tool Learning Framework for Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2502.11404](https://arxiv.org/abs/2502.11404)  
**Code**: Yes (mentioned in the paper, see the original text for the specific link)  
**Area**: LLM/NLP  
**Keywords**: Tool learning, Code generation, Software engineering, Function calling, Error reflection

## TL;DR
This paper proposes the ToolCoder framework, which reformulates tool learning as a code generation task. By drawing on software engineering principles (requirements analysis → modular design → implementation & execution → error debugging → code reuse), it enables LLMs to perform multi-step tool calls by generating and executing Python code, comprehensively outperforming baselines such as ReAct and CodeAct on RestBench and API-Bank.

## Background & Motivation

**Background**: The dominant paradigm in LLM tool learning is the "plan-execute-observe" cycle (e.g., ReAct), where LLMs perform step-by-step reasoning in natural language, selecting and executing one tool call at each step and observing the result. Code-based approaches (e.g., CodeAct) have emerged, letting LLMs generate code snippets as actions.

**Limitations of Prior Work**: (a) **Weak Planning**: relying on manual prompts and natural language reasoning for multi-step planning, which is highly error-prone in complex tasks; (b) **Imprecise Error Diagnosis**: inability to locate the exact cause of execution failures, leading to a lack of targeted corrections; (c) **Lack of Experience Reuse**: initiating each query from scratch without utilizing successfully executed experience, resulting in redundant resolution of similar sub-problems.

**Key Challenge**: Natural language reasoning excels at flexible expression but lacks structured constraints, while code excels at structured execution but requires precise planning. How to combine the strengths of both?

**Goal**: To design a systematic framework that leverages the code generation capabilities of LLMs and software engineering methodologies to comprehensively enhance the planning, execution, error-correction, and reuse capabilities in tool learning.

**Key Insight**: Mapping the entire software development process to tool learning—requirements analysis corresponds to intent understanding, modular design to task decomposition, code implementation to tool invocation, debugging to error reflection, and the code repository to experience reuse.

**Core Idea**: Treat tool learning as software development—transform queries into function signatures, subtasks into code comments, tool invocations into function implementations, and Python tracebacks into error diagnosis.

## Method

### Overall Architecture
Input: Natural language query $q$ + toolset $\mathcal{T}$ (including documentation $\mathcal{D}$). Output: Execution result $r$ as the final answer. The workflow consists of four stages: Task-to-Code Transformation → Subtask Planning and Tool Selection → Code Implementation and Execution → Error Reflection and Correction. Additionally, a reusable function repository $\mathcal{F}$ is integrated throughout the entire process.

### Key Designs

1. **Task-to-Code Transformation**:

    - **Function**: Transform natural language queries into a structured Python function scaffold $c$, including the function name, parameter list, docstring, and the main function call.
    - **Mechanism**: Analogous to the requirements analysis phase in software engineering. For example, for the query "How many movies did Sofia Coppola direct?", it generates `def get_directed_movie_count(director_name: str) -> int:` along with a complete docstring, keeping the function body empty.
    - **Design Motivation**: The structured scaffold provides an explicit input-output specification for subsequent planning, activating the code reasoning ability of LLMs. Experiments indicate that the correct path rate drops significantly when the scaffold is removed.

2. **Subtask Planning & Tool Selection**:

    - **Function**: Decompose the scaffold into a modular sequence of subtasks, select appropriate tools for each subtask, and generate pseudocode.
    - **Mechanism**: The planning module analyzes the scaffold and generates a sequence of subtasks $\{s_1, ..., s_m\}$, which are embedded as code comments. The tool selection module references the tool documentation to map subtasks to concrete API calls, generating pseudocode $c_p$ that contains the tool invocation sequence and data flow.
    - **Design Motivation**: Code comments are naturally suited for expressing structured task decomposition, and data flow is seamlessly connected via variable assignments. This is more precise than natural language planning and reduces ambiguity.

3. **Implementation & Execution + Reusable Function Repository**:

    - **Function**: Generate executable implementations for each sub-function placeholder in the pseudocode, and store successfully executed sub-functions in the repository $\mathcal{F}$.
    - **Mechanism**: The code generation module references the selected tool documentation $\mathcal{T}_s$ and the existing reusable repository $\mathcal{F}$ to generate Python implementations for each sub-function. Upon successful execution, the sub-functions are extracted and added to the repository; they are directly reused when similar subtasks are encountered in the future.
    - **Design Motivation**: (a) Code reuse avoids repeatedly solving the same sub-problems, enhancing efficiency; (b) Code snippets verified by execution are more reliable than those generated from scratch; (c) As reasoning progresses, the repository is continuously enriched, forming a positive feedback loop.

4. **Error Reflection**:

    - **Function**: Utilize Python traceback information to precisely locate errors and iteratively repair them, including two strategies: planning reconstruction and code review.
    - **Mechanism**: **Planning Reconstruction**—when a non-existent tool is detected, it is cross-validated against the available toolset to guide the LLM to select an alternative tool. **Code Review**—upon execution failure, Python's detailed traceback precisely pinpoints the error line and cause, allowing the LLM to modify the code and re-execute, iterating up to 3 times.
    - **Design Motivation**: Python's traceback mechanism dynamically provides precise error diagnosis information, which is far superior to vague descriptions in natural language. This is one of the core advantages of code-based methods over text-based methods.

### Loss & Training
ToolCoder is a pure prompting framework and requires no training. It is mainly implemented using GPT-4o-mini, and its generalization capabilities are also tested on the Qwen2.5 series of models.

## Key Experimental Results

### Main Results
Evaluated on RestBench (TMDB + Spotify) and API-Bank (Level 1/2).

| Method | TMDB Success↑ | TMDB Acc↑ | TMDB Path↑ | Spotify Success↑ | API-Bank L2↑ |
|------|---------------|-----------|------------|-------------------|-------------|
| ReAct | 76.0 | 48.0 | 50.0 | 68.42 | 56.30 |
| CodeAct | 80.0 | 56.0 | 67.0 | 71.93 | 54.07 |
| **ToolCoder** | **85.0** | **78.0** | **83.0** | **87.72** | **62.41** |

### Ablation Study

| Configuration | TMDB Success↑ | TMDB Acc↑ | TMDB Path↑ | Description |
|------|---------------|-----------|------------|------|
| ToolCoder Full | 85.0 | 78.0 | 83.0 | Full framework |
| w/o Reusable Repository | 83.0 | 71.0 | 78.0 | After removing reuse, Acc -7%, Path -5% |
| w/o Error Reflection | 75.0 | 65.0 | 77.0 | After removing reflection, Success -10%, Acc -13% |
| w/o Code Scaffold | - | - | Significant Decrease | Planning capability is severely compromised |

### Key Findings
- **Code-based methods comprehensively outperform text-based methods**: ToolCoder leads across all metrics, with the structured advantage of code being particularly prominent in complex multi-step tasks.
- **Error reflection contributes the most**: Removing it leads to a 10 percentage point decrease in the success rate, indicating that the precise error information provided by Python tracebacks is extremely critical for error correction.
- **Code scaffolds activate reasoning ability**: Excluding the Python function scaffold leads to a significant decrease in the correctness of the planning path, proving that structured code templates help LLMs better understand task intent.
- **The cumulative effect of the reusable repository is distinct**: As inference samples increase, the cumulative success rate steadily improves (approaching 90% on TMDB), demonstrating the effectiveness of the positive feedback loop in experience reuse.
- **Code LLMs benefit more**: Qwen2.5-Coder shows larger improvements than the base Qwen2.5, indicating that code pre-training is highly complementary to the ToolCoder framework.

## Highlights & Insights
- **Systematic mapping of software engineering methodologies**: Mapping the complete software development life cycle (requirements analysis → design → implementation → testing → maintenance) to every step of tool learning is not simply "replacing text with code," but a systematic methodology. This mindset is transferable to any Agent task requiring structured reasoning.
- **Python traceback as an error diagnostic signal**: Utilizing the built-in error mechanism of programming languages for precise debugging is more reliable and concrete than letting the LLM "reflect" in natural language. This trick is applicable across all code-generating Agents.
- **Cumulative learning via reusable function repositories**: Similar to a programmer's code snippet library, it gradually strengthens by accumulating experience through use, offering a clean implementation of "continuous improvement" in tool learning.

## Limitations & Future Work
- **Dependence on the LLM's code generation quality**: If the base model has weak coding capabilities, the entire framework may perform poorly (though it remains effective on Qwen2.5-14B).
- **Applicable only to programmable tool calls**: For tool interaction scenarios that require human judgment or cannot be programmed, code-based approaches may not be applicable.
- **Maintenance of the reusable repository**: As the repository grows, redundant or obsolete function snippets may appear, and there is a lack of cleanup/update mechanisms.
- **Limited evaluation scenarios**: RestBench and API-Bank are relatively small in scale with a limited number of tools; performance on large-scale toolsets (thousands of APIs) remains to be validated.

## Related Work & Insights
- **vs ReAct**: ReAct alternates reasoning and acting in natural language, which limits its planning capability. ToolCoder uses code for structured planning, offering higher accuracy.
- **vs CodeAct**: CodeAct also generates code as actions, but performs single-turn generation without iterative corrections or experience accumulation. ToolCoder incorporates error reflection and code reuse, making it more systematic.
- **vs ATC**: ATC uses a black-box probing method to assist in toolchain construction, whereas ToolCoder is more transparent and controllable through code scaffolding and modular design.

## Rating
- Novelty: ⭐⭐⭐⭐ Mapping software engineering methodologies systematically to tool learning is novel, though the general direction of code-based tool use is not unprecedented.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive across multiple benchmarks, models, and ablations, but dataset scales are relatively small.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear, the software engineering analogy is appropriate, and figures/tables are rich.
- Value: ⭐⭐⭐⭐ Provides a complete framework for code-empowered tool learning, with well-designed and reusable components.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] To Code or not to Code? Adaptive Tool Integration for Math Language Models via Expectation-Maximization](to_code_or_not_to_code_adaptive_tool_integration_for_math_language_models_via_ex.md)
- [\[ACL 2025\] WarriorCoder: Learning from Expert Battles to Augment Code Large Language Models](warriorcoder_learning_from_expert_battles_to_augment_code_large_language_models.md)
- [\[ACL 2025\] Divide-Then-Aggregate: An Efficient Tool Learning Method via Parallel Tool Invocation](dta_llama_parallel_tool_invocation.md)
- [\[ACL 2025\] ToolSpectrum: Towards Personalized Tool Utilization for Large Language Models](toolspectrum_towards_personalized_tool_utilization_for_large_language_models.md)
- [\[ACL 2025\] Interactive and Expressive Code-Augmented Planning with Large Language Models](interactive_and_expressive_code-augmented_planning_with_large_language_models.md)

</div>

<!-- RELATED:END -->
