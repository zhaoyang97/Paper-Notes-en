---
title: >-
  [Paper Note] ComfyBench: Benchmarking LLM-based Agents in ComfyUI for Autonomously Designing Collaborative AI Systems
description: >-
  [CVPR 2025][Multi-Agent][LLM Agent] ComfyBench proposes the first comprehensive benchmark (200 tasks, 3205 node documentations, and 20 curriculum workflows) to evaluate the capability of LLM-based agents to autonomously design collaborative AI systems in ComfyUI. It also introduces the ComfyAgent framework, which leverages code-based workflow representation and multi-agent collaboration to achieve a resolve rate comparable to o1-preview. However…
tags:
  - "CVPR 2025"
  - "Multi-Agent"
  - "LLM Agent"
  - "Workflow Generation"
  - "ComfyUI"
  - "Collaborative AI System"
  - "Benchmark"
date: 2026-05-08
content_hash: 151b1e90905c86df
---

# ComfyBench: Benchmarking LLM-based Agents in ComfyUI for Autonomously Designing Collaborative AI Systems

**Conference**: CVPR 2025  
**arXiv**: [2409.01392](https://arxiv.org/abs/2409.01392)  
**Code**: [https://github.com/xxyQwQ/ComfyBench](https://github.com/xxyQwQ/ComfyBench)  
**Area**: LLM Evaluation  
**Keywords**: LLM Agent, Workflow Generation, ComfyUI, Collaborative AI System, Benchmark

## TL;DR

ComfyBench proposes the first comprehensive benchmark (200 tasks, 3205 node documentations, and 20 curriculum workflows) to evaluate the capability of LLM-based agents to autonomously design collaborative AI systems in ComfyUI. It also introduces the ComfyAgent framework, which leverages code-based workflow representation and multi-agent collaboration to achieve a resolve rate comparable to o1-preview. However, it resolves only 15% of helper creative tasks, highlighting a significant gap in autonomous system design for LLM agents.

## Background & Motivation

**Background**: Prior AI research has mainly focused on developing massive monolithic models to maximize task-specific intelligence. However, another perspective is to use LLM agents to **autonomously design collaborative AI systems**—namely, pipelines or workflows composed of multiple AI models.

**Limitations of Prior Work**: (1) There lacks a standardized benchmark to evaluate the ability of LLM agents to design collaborative AI systems; (2) Even with powerful LLMs as agents, it remains challenging to understand and combine complex node systems (e.g., ComfyUI contains thousands of nodes with distinct functions); (3) Existing agent frameworks lack mechanisms to learn from historical workflows.

**Key Challenge**: Although visual workflow systems like ComfyUI are flexible and powerful, the combinatorial space of node graphs is extremely vast. The permutations and configurations of 3205 nodes constitute a complex design space that far exceeds the context comprehension capacity of a single LLM.

**Goal**: (1) To construct a benchmark evaluating workflow design capabilities of agents; (2) To develop an agent framework capable of utilizing node documentation and existing workflows to generate new workflows.

**Key Insight**: Transforming workflows into code representations (instead of visual node graphs) allows LLMs to better understand and generate workflows. Multi-agent collaboration is employed so that different agents manage document retrieval, workflow learning, and code generation.

**Core Idea**: Representing workflows as code enables LLM comprehension and generation, while multi-agent collaboration enables learning from existing workflows to design new collaborative AI systems.

## Method

### Overall Architecture

ComfyBench consists of two components: (1) **Benchmark**: 200 diverse task instructions (covering text-to-image, image editing, style transfer, super-resolution, object removal, etc.), detailed documentation for 3205 ComfyUI nodes, and 20 reference workflows for agent learning; (2) **ComfyAgent**: a multi-agent framework capable of autonomously reading node documentation and reference workflows to generate new workflow code for a given task, which can be inversely parsed into ComfyUI workflows and executed.

### Key Designs

1. **Code-based Workflow Representation**:

    - **Function**: Translates the visual node graphs of ComfyUI into a Python code format understandable by LLMs.
    - **Mechanism**: Each ComfyUI workflow is represented in JSON (comprising node IDs, types, parameters, connections). This is translated into equivalent Python code containing node instantiations and link relations. The code can be inversely parsed into JSON workflows by an interpreter for execution in ComfyUI.
    - **Design Motivation**: LLMs inherently understand code. Code representation is more compact and structured than JSON node descriptions, facilitating easier generation and comprehension by LLMs.

2. **Multi-Agent Collaboration System**:

    - **Function**: Collaboratively performs the entire pipeline from task comprehension to workflow generation.
    - **Mechanism**: Multiple agents handle distinct responsibilities: the document retrieval agent extracts relevant node information from the 3205 node docs; the curriculum learning agent identifies the most relevant historical solutions from the 20 reference workflows as demonstrations; and the code generation agent generates the new workflow code based on the retrieved node info and reference workflows.
    - **Design Motivation**: Breaking down a complex system design task into manageable sub-tasks mitigates the individual limitations of a single LLM through agent collaboration.

3. **Dual Metric Evaluation System**:

    - **Function**: Comprehensively evaluates the workflow design capabilities of the agents.
    - **Mechanism**: **Pass Rate**: whether the generated workflow can execute correctly in ComfyUI (free of syntax or connection errors); **Resolve Rate**: whether the execution results satisfy the task requirements (assessed through human or automated evaluation).
    - **Design Motivation**: Pass Rate measures the agent's understanding of node systems, while Resolve Rate measures its semantic task comprehension and creative design capabilities.

### Loss & Training

Training-free—ComfyAgent is an inference-time agent framework built upon the in-context learning and tool-calling capabilities of LLMs.

## Key Experimental Results

### Main Results

| Agent | Pass Rate | Resolve Rate |
|-------|-------------------|---------------------|
| Baseline LLM Agent | Low | Low |
| **ComfyAgent** | **Comparable to o1-preview** | **Comparable to o1-preview** |
| o1-preview | Baseline | Baseline |

- ComfyAgent significantly outperforms other agent baselines.
- Resolves only 15% of creative tasks.

### Ablation Study

| Configuration | Description |
|------|------|
| Complete ComfyBench Setup | 200 tasks, 3205 nodes, 20 reference workflows |
| Different Task Categories | Significant gap between creative and standard tasks (15% vs higher) |

### Key Findings

- **Creative tasks serve as the primary bottleneck**: ComfyAgent only resolves 15% of creative tasks, demonstrating that LLM agents still have immense room for improvement in scenarios requiring compositional innovation.
- **Code representation significantly outperforms JSON representation**: LLMs find workflow represented in Python code format easier to understand and generate.
- **Learning from existing workflows is critical**: The curriculum learning mechanism drastically improves generation quality by providing reference solutions.
- **Alignment with o1-preview**: ComfyAgent (based on open-source LLMs) reaches par with o1-preview, demonstrating the critical importance of proper agent framework design.

## Highlights & Insights

- **Forward-looking problem formulation**: Shifting from "making AI smart" to "allowing AI to design AI systems" represents a paradigm shift. Utilizing LLM agents to replace humans in assembling AI systems is a key stepping stone toward AGI.
- **Code as a universal interface**: The concept of unifying complex systems operation into a code generation task provides valuable insights, serving as a key to general-purpose LLM agents.
- **Practical benchmark design**: Featuring 200 real-world image generation tasks and 3205 authentic node documents evaluated directly within the ComfyUI ecosystem, the results are both executable and verifiable.

## Limitations & Future Work

- The resolve rate on creative tasks is only 15%, indicating that the agent struggles significantly with brand-new compositions (rather than imitating existing designs).
- The benchmark is constrained to the image generation domain of ComfyUI, without covering other collaborative systems such as video or audio.
- The evaluation criteria for the resolve rate might incur subjective bias.
- Lacks a detailed per-category analysis when compared with o1-preview.

## Related Work & Insights

- **vs. General-purpose Agents (e.g., AutoGPT, MetaGPT)**: General-purpose agents focus on general task decomposition, whereas ComfyAgent optimizes for workflow design, displaying superior efficiency in this specific domain.
- **vs. Code Generation (e.g., Copilot)**: Code generation focuses on execution code, while ComfyAgent generates a "blueprint" for a collaborative system, requiring an understanding of components interaction.

## Rating

- Novelty: ⭐⭐⭐⭐ The first benchmark to evaluate agents designing collaborative AI systems, featuring a pioneering problem setting.
- Experimental Thoroughness: ⭐⭐⭐ Core metrics are sufficiently reported, but detailed per-model/per-category data is lacking.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation of the problem, with logical system design.
- Value: ⭐⭐⭐⭐ The combination of the benchmark and the agent framework significantly advances LLM agent research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MAS-ZERO: Designing Multi-Agent Systems with Zero Supervision](../../NeurIPS2025/multi_agent/maszero_designing_multiagent_systems_with_zero_supervision.md)
- [\[CVPR 2025\] Collaborative Tree Search for Enhancing Embodied Multi-Agent Collaboration](collaborative_tree_search_for_enhancing_embodied_multi-agent_collaboration.md)
- [\[AAAI 2026\] COACH: Collaborative Agents for Contextual Highlighting -- A Multi-Agent Framework for Sports Video Analysis](../../AAAI2026/multi_agent/coach_collaborative_agents_for_contextual_highlighting_--_a_multi-agent_framewor.md)
- [\[ICLR 2026\] When Agents "Misremember" Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems](../../ICLR2026/multi_agent/when_agents_misremember_collectively_exploring_the_mandela_effect_in_llm-based_m.md)
- [\[ICLR 2026\] WideSearch: Benchmarking Agentic Broad Info-Seeking](../../ICLR2026/multi_agent/widesearch_benchmarking_agentic_broad_info-seeking.md)

</div>

<!-- RELATED:END -->
