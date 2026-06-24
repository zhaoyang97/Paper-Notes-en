---
title: >-
  [Paper Note] SceneGenAgent: Precise Industrial Scene Generation with Coding Agent
description: >-
  [ACL 2025][Code Intelligence][Industrial Scene Generation] This work proposes SceneGenAgent, an LLM-based code-generation agent. Through structured layout planning, layout verification, and iterative refinement, it utilizes C# code to generate industrial scenes with high precision. It achieves an 81% success rate on real industrial tasks and constructs the SceneInstruct dataset, enabling open-source LLMs to perform closely to GPT-4o.
tags:
  - "ACL 2025"
  - "Code Intelligence"
  - "Industrial Scene Generation"
  - "LLM Code Generation"
  - "Layout Planning"
  - "Iterative Refinement"
  - "Scene Modeling"
date: 2026-05-08
content_hash: 03d0472a4e2e37bc
---

# SceneGenAgent: Precise Industrial Scene Generation with Coding Agent

**Conference**: ACL 2025  
**arXiv**: [2410.21909](https://arxiv.org/abs/2410.21909)  
**Code**: [https://github.com/THUDM/SceneGenAgent](https://github.com/THUDM/SceneGenAgent)  
**Area**: Code Intelligence  
**Keywords**: Industrial Scene Generation, LLM Code Generation, Layout Planning, Iterative Refinement, Scene Modeling  

## TL;DR

This work proposes SceneGenAgent, an LLM-based code-generation agent. Through structured layout planning, layout verification, and iterative refinement, it utilizes C# code to generate industrial scenes with high precision. It achieves an 81% success rate on real industrial tasks and constructs the SceneInstruct dataset, enabling open-source LLMs to perform closely to GPT-4o.

## Background & Motivation

**Background**: Industrial scene modeling is a fundamental requirement for simulation and manufacturing pipelines. Recently, Large Language Models (LLMs) have achieved significant progress in generating 3D scenes from textual descriptions, demonstrating strong performance particularly in general scenarios such as domestic and indoor environments.

**Limitations of Prior Work**: However, industrial scenes differ fundamentally from general scenes. Industrial scenes require precise measurements and positioning, imposing strict requirements on dimensions, spacing, and spatial layout. Directly applying general scene generation methods fails to meet industrial-grade precision requirements, where LLMs' deficiencies in spatial reasoning and precise numerical calculation become the primary bottleneck.

**Key Challenge**: LLMs excel at understanding natural language descriptions and reasoning about high-level semantics, but perform poorly in scenarios requiring precise numerical calculation and spatial geometric layout. Directly prompting LLMs to output coordinates and sizes often leads to inaccuracies, whereas industrial scenes tolerate zero layout errors.

**Goal**: To design an LLM-based agent system that: (1) formulates scene generation as a structured code generation task to ensure precision; (2) introduces automated verification and iterative refinement mechanisms to improve generation quality; and (3) constructs a training dataset to empower open-source models with industrial scene generation capabilities.

**Key Insight**: The authors observe that code naturally supports precise numerical calculations and logical control. By translating the scene generation task into a C# code generation problem, the executability of code is leveraged to guarantee layout precision, while compiler feedback is used to automatically verify and refine the generation.

**Core Idea**: Generate C# code via LLMs to describe industrial scene layouts, integrated with a "planning-verification-refinement" closed-loop agent to guarantee high precision.

## Method

### Overall Architecture

SceneGenAgent is a multi-stage LLM agent framework. The overall workflow is as follows: input natural language description of an industrial scene $\to$ LLM performs structured layout planning (generating a calculable layout format) $\to$ translate the layout plan into C# code $\to$ execute the code and perform layout verification $\to$ if verification fails, enter iterative refinement until requirements are met $\to$ output final industrial scene. The core of the system relies on three modules: structured layout planning, automated layout verification, and iterative optimization/refinement.

### Key Designs

1. **Structured & Calculable Layout Planning**:

    - **Function**: Translates natural language scene descriptions into structured, precisely calculable layout representations.
    - **Mechanism**: Defines a structured layout description format so that LLMs output structured data containing precise numerical values (such as object types, locations, and dimension parameters) instead of free-form text. This format is naturally expressible and calculable via code, avoiding the ambiguity of natural language. While generating C# code, the LLM can leverage the programming language's computational capacity for precise spatial layout calculations.
    - **Design Motivation**: Industrial scenes require millimeter-level precision, which is impossible to convey accurately through natural language descriptions. The structured format enables programmatic layout verification, while the code representation guarantees the accuracy of numerical coordinates.

2. **Layout Verification**:

    - **Function**: Automatically detects whether the generated scene layout meets industrial constraints.
    - **Mechanism**: After compiling and executing the generated C# code, the system automatically checks for object collisions, clearance constraints, boundary conditions, etc. It judges layout validity and correctness using compiler error messages and runtime check results. The verification module generates structured error reports, providing precise feedback signals for subsequent corrections.
    - **Design Motivation**: In industrial scenarios, object interpenetrations and violations of safety clearance are strictly forbidden. Automatic verification replaces manual inspection to achieve an end-to-end automated generation pipeline while offering reliable feedback for iterative corrections.

3. **Iterative Refinement**:

    - **Function**: Automatically corrects layout errors based on verification feedback.
    - **Mechanism**: When layout verification detects issues, the system feeds error messages back to the LLM. The LLM then adjusts the code based on the specific error types and spatial information. This process can run for multiple rounds, with verification repeated after each correction until all constraints are met or the maximum iteration threshold is reached.
    - **Design Motivation**: Generating a complex industrial scene correctly on the first attempt is exceptionally challenging. The iterative correction mechanism allows the model to refine the layout step-by-step, mimicking the iterative adjustment process of human engineers. It leverages the strength of coding agents—namely, that code can be precisely modified and locally adjusted without regenerating the entire scene.

### Loss & Training

To equip open-source models with industrial scene generation capabilities, the authors constructed the SceneInstruct dataset. This dataset contains pairs of industrial scene descriptions and their corresponding high-quality C# code, which are used to fine-tune open-source LLMs (such as Llama3.1-70B). The goal of fine-tuning is to enable the model to directly generate structured layout code from scene descriptions and comprehend spatial constraints in industrial contexts.

## Key Experimental Results

### Main Results

Experiments are evaluated on real-world industrial scene generation tasks, where engineers author scene descriptions and manually inspect the correctness of generated outputs.

| Model | Method | Success Rate |
|---|---|---|
| GPT-4o | Direct Generation (w/o Agent) | ~50% |
| GPT-4o | + SceneGenAgent | **81.0%** |
| Llama3.1-70B | Direct Generation | Low |
| Llama3.1-70B | + SceneInstruct SFT + SceneGenAgent | Close to GPT-4o |
| Other Open-Source LLMs | + SceneInstruct SFT | Significant Gain |

### Ablation Study

| Configuration | Success Rate | Description |
|---|---|---|
| Full SceneGenAgent | 81.0% | Complete framework |
| w/o Layout Verification | Obvious decline | Cannot guarantee precision without verification |
| w/o Iterative Refinement | Moderate decline | Only a single attempt to generate the correct layout |
| w/o Structured Format | Significant decline | Free-text layout precision cannot be guaranteed |
| SceneInstruct SFT | Significant increase | Open-source models benefit the most |

### Key Findings

- The structured layout format is the most critical design for performance gain. Removing it leads to a dramatic drop in success rate, as accurate numerical computation relies entirely on the executability of code.
- The iterative refinement mechanism typically converges within 2-3 rounds, demonstrating that most errors can be resolved with minor corrections.
- SceneInstruct fine-tuning significantly boosts open-source models; fine-tuned Llama3.1-70B under the SceneGenAgent framework performs comparably to GPT-4o, validating the immense value of domain-specific data for open-source models.
- Engineer evaluations indicate that scenes generated by SceneGenAgent fulfill most practical industrial requirements.

## Highlights & Insights

- **Code as a Guarantee of Precision**: Formulating scene generation as code generation is an extremely clever design. Code not only expresses precise numerical relationships but also allows a compiler to automatically catch errors—something natural language outputs cannot achieve. This methodology can be transferred to any task requiring precise numerical outputs (e.g., circuit design, architectural layout).
- **Verification-Correction Closed Loop**: The core of an agent lies in its feedback loop. Unlike purely generative approaches, SceneGenAgent leverages compilers and runtime checks to construct automated feedback mechanisms, making the correction process precise down to specific lines of code and numerical values.
- **Data Flywheel Effect**: SceneInstruct demonstrates a highly practical knowledge distillation pathway of "strong model annotation $\to$ weak model learning", utilizing high-quality code generated by GPT-4o to train open-source models.

## Limitations & Future Work

- The current evaluation relies heavily on manual inspections and lacks an automated metric system, resulting in high scalability costs.
- The C# code is tied to specific industrial software (such as Unity), and generalising to other industrial tools requires adaptation.
- The scale and diversity of SceneInstruct may be limited, potentially lacking coverage of highly complex or rare industrial scenes.
- The temporal overhead of multi-round iterative refinement may become a bottleneck in latency-sensitive applications.
- Future work can explore multimodal inputs (e.g., schematics + text descriptions) to enhance scene understanding capabilities.

## Related Work & Insights

- **vs General 3D Scene Generation Methods**: Methods like SceneScape and Text2Room target general scenarios and do not focus on precision. SceneGenAgent is tailored specifically for industrial precision, marking a key advancement in scene generation from "looking correct" to "measuring correct".
- **vs Code Generation Agents (e.g., SWE-Agent)**: While SWE-Agent tackles software engineering issues, SceneGenAgent transfers the code-agent paradigm to scene generation, validating the generalizability of coding agents in precision-control tasks.
- This paper inspires a broader concept: for tasks requiring precise outputs, incorporating a paradigm of "code as intermediate representation + compiler verification" is highly promising.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing code generation into industrial scene modeling offers a fresh perspective, though the closed-loop agent framework itself is not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐ Evaluation on real-world industrial tasks is a highlight, but the scale is relatively small and heavily dependent on human assessment.
- Writing Quality: ⭐⭐⭐⭐ Clear methodological descriptions and well-articulated motivation.
- Value: ⭐⭐⭐⭐ Offers direct application value for industrial scene generation, with a highly transferable code-agent paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Self-Improving Coding Agent](../../NeurIPS2025/code_intelligence/a_selfimproving_coding_agent.md)
- [\[ACL 2025\] DARS: Dynamic Action Re-Sampling to Enhance Coding Agent Performance by Adaptive Tree Traversal](dars_dynamic_action_re-sampling_to_enhance_coding_agent_performance_by_adaptive_.md)
- [\[ACL 2025\] UTBoost: Rigorous Evaluation of Coding Agents on SWE-Bench](utboost_rigorous_evaluation_of_coding_agents_on_swe-bench.md)
- [\[ACL 2025\] CompileAgent: Automated Real-World Repo-Level Compilation with Tool-Integrated LLM-based Agent System](compileagent_automated_real-world_repo-level_compilation_with_tool-integrated_ll.md)
- [\[ACL 2025\] Rethinking Repetition Problems of LLMs in Code Generation](rethinking_repetition_problems_of_llms_in_code_generation.md)

</div>

<!-- RELATED:END -->
