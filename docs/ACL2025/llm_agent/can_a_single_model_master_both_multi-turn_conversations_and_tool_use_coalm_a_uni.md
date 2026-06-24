---
title: >-
  [Paper Note] Can a Single Model Master Both Multi-turn Conversations and Tool Use? CoALM: A Unified Conversational Agentic Language Model
description: >-
  [ACL 2025][LLM Agent][Conversational Agent] This paper proposes CoALM (Conversational Agentic Language Model). By constructing a multi-task training dataset, CoALM-IT, that integrates multi-turn ReAct reasoning and complex API calls, the authors train a unified model that excels in both traditional task-oriented dialogue (TOD) and language agent (LA) tool use, outperforming specialized models such as GPT-4o on three benchmarks: MultiWOZ, BFCL V3, and API-Bank.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "Conversational Agent"
  - "Tool Calling"
  - "Multi-turn Dialogue"
  - "Unified Model"
  - "Task-oriented Dialogue"
date: 2026-05-08
content_hash: 72e141058003b868
---

# Can a Single Model Master Both Multi-turn Conversations and Tool Use? CoALM: A Unified Conversational Agentic Language Model

**Conference**: ACL 2025  
**Link**: [ACL Anthology](https://aclanthology.org/2025.acl-long.605/)  
**Code**: None  
**Area**: LLM Agent / Dialogue Systems / Tool Use  
**Keywords**: Conversational Agent, Tool Calling, Multi-turn Dialogue, Unified Model, Task-oriented Dialogue  

## TL;DR

This paper proposes CoALM (Conversational Agentic Language Model). By constructing a multi-task training dataset, CoALM-IT, that integrates multi-turn ReAct reasoning and complex API calls, the authors train a unified model that excels in both traditional task-oriented dialogue (TOD) and language agent (LA) tool use, outperforming specialized models such as GPT-4o on three benchmarks: MultiWOZ, BFCL V3, and API-Bank.

## Background & Motivation

**Background**: Current AI conversational assistants require two core capabilities: (1) traditional task-oriented dialogue (TOD) capability—maintaining user intents across multi-turn conversations, managing dialogue states, and operating a finite set of APIs (e.g., restaurant reservation systems); (2) language agent (LA) capability—using an open set of tools/APIs to perform complex multi-step reasoning tasks.

**Limitations of Prior Work**: TOD and LA systems each have distinct strengths but mutual shortcomings: TOD systems excel at multi-turn dialogue management (maintaining context, handling user intent changes) but can only operate a predefined, finite set of APIs, requiring re-collection of training data to integrate new services; LA systems excel at flexible tool calling and reasoning but fail to effectively maintain user intent in multi-turn dialogues, easily "forgetting" prior dialogue context.

**Key Challenge**: Multi-turn dialogue management and flexible tool use are both necessary conditions for building excellent conversational agents, but existing methods often compromise one capability when optimizing for the other. The training data formats and objectives of the two capabilities differ, making simple mixed training ineffective.

**Goal**: (1) Quantitatively analyze the gap between the two types of methods in cross-domain evaluation; (2) design a unified training scheme to enable a single model to possess both TOD and LA capabilities.

**Key Insight**: By carefully constructing a multi-task training dataset (CoALM-IT), multi-turn ReAct reasoning and API calls are interwoven throughout the dialogue flow, exposing the model to the fusion of both capabilities during training.

**Core Idea**: Integrate TOD dialogue state management and LA ReAct reasoning into a unified data format, training a single model to master both capabilities simultaneously.

## Method

### Overall Architecture

The core of CoALM is the construction of the CoALM-IT training dataset, followed by fine-tuning LLaMA series models (8B/70B/405B) on it. The input consists of the multi-turn dialogue history and available tool descriptions, and the output is a multi-step response that integrates reasoning, tool calling, and natural language responses.

### Key Designs

1. **CoALM-IT Multi-task Training Data Construction**:
    - **Function**: Create mixed training data with a unified format covering both TOD and LA capabilities.
    - **Mechanism**: Starting from existing TOD datasets (MultiWOZ, etc.) and LA datasets (ToolBench, etc.), convert them into a unified dialogue format. The key innovation lies in converting dialogue state update operations in TOD into API-like calling formats and introducing a ReAct-style Thought-Action-Observation loop. This unifies TOD and LA data in format, allowing the model to learn complementary capabilities from both. Meanwhile, data augmentation is employed to introduce more diverse API descriptions for TOD scenarios and richer multi-turn dialogue contexts for LA scenarios.
    - **Design Motivation**: Formatting unification is the foundation of capability unification—if TOD and LA are trained on completely different formats, the model cannot switch flexibly during inference.

2. **Integration of Multi-turn ReAct Reasoning and Dialogue Management**:
    - **Function**: Embed the ReAct reasoning process within multi-turn dialogues.
    - **Mechanism**: In each dialogue turn, the model first performs a "Thought" (reflecting on current user intent and dialogue history), then decides on an "Action" (calling an API or replying directly). If an API is called, the model waits for an "Observation" (API return result), and finally generates a "Response" (a natural language response based on the reasoning and API results). This format naturally unifies TOD dialogue state management (tracking intent changes via Thought) and LA tool calling (executing APIs via Action).
    - **Design Motivation**: The explicit reasoning steps of ReAct enable the model to "reflect" on context changes in each dialogue turn, maintaining dialogue coherence better than end-to-end generation.

3. **Multi-scale Model Training Strategy**:
    - **Function**: Verify the scalability of the method across different LLaMA model sizes.
    - **Mechanism**: CoALM-8B, CoALM-70B, and CoALM-405B are trained respectively using the same CoALM-IT dataset but different training hyperparameters. Evaluation is conducted on three complementary benchmarks: MultiWOZ 2.4 (TOD, testing dialogue management), BFCL V3 (LA, testing function calling), and API-Bank (LA, testing multi-step API usage).
    - **Design Motivation**: To verify the effectiveness of the unified training scheme across different model scales while exploring scaling laws.

### Loss & Training

A standard instruction-tuning workflow is used to perform full-parameter fine-tuning of LLaMA models on CoALM-IT. The training loss is the standard next-token prediction loss.

## Key Experimental Results

### Main Results

| Benchmark | Metric | CoALM-8B | CoALM-70B | CoALM-405B | GPT-4o | Specialized SOTA |
|---|---|---|---|---|---|---|
| MultiWOZ 2.4 | Joint Goal Acc | High | Higher | Optimal | Lower | TOD-specific Model |
| BFCL V3 | Function Call Acc | High | Higher | Optimal | Baseline | LA-specific Model |
| API-Bank | API Call Success | High | Higher | Optimal | Baseline | LA-specific Model |

### Ablation Study

| Configuration | MultiWOZ | BFCL V3 | API-Bank | Description |
|---|---|---|---|---|
| CoALM-IT (Full Data) | Optimal | Optimal | Optimal | Full Model |
| Trained on TOD data only | Good | Poor | Poor | LA capability missing |
| Trained on LA data only | Poor | Good | Good | TOD capability missing |
| Simple mixture (no format unification) | Moderate | Moderate | Moderate | Inconsistent formatting affects performance |
| Without ReAct | Significant Decrease | Decrease | Decrease | Reasoning steps missing |

### Key Findings
- A single CoALM model surpasses the best domain-specific methods in both TOD and LA domains, proving the feasibility of a unified model.
- CoALM even outperforms GPT-4o, indicating that the quality of targeted training data is more important than model scale.
- Dataset formatting unification (rather than simple mixing) is crucial—when the same data is not unified in format, performance drops significantly.
- The scaling effect is prominent: improvements from $8\text{B} \rightarrow 70\text{B} \rightarrow 405\text{B}$ are consistent across all three benchmarks.

## Highlights & Insights
- **The research perspective of "bridging TOD and LA"** is highly valuable. Existing studies isolate the two, but real-world dialogue assistants need to possess both capabilities simultaneously. CoALM systematically validates the feasibility of a unified model for the first time.
- **The power of data engineering** is demonstrated once again. The design of CoALM-IT (format unification + ReAct integration) is the core of the method's success, proving more critical than architectural innovations.
- This method provides a clear technical roadmap for building practical conversational agents: high-quality unified training data + instruction tuning.

## Limitations & Future Work
- The construction process of CoALM-IT requires extensive manual design and quality control, limiting scalability.
- Evaluation is only conducted in English scenarios; the performance of multi-lingual conversational agents remains unclear.
- Safety and hallucination control are not considered—incorrect tool calling can lead to severe consequences in real-world deployment.
- Future work can explore incorporating more dimensions of capability (e.g., knowledge QA, code execution) into the unified framework.

## Related Work & Insights
- **vs ToolLLM**: ToolLLM focuses on fine-tuning tool-use capabilities and does not involve multi-turn dialogue management; CoALM achieves comparable performance in tool use while retaining TOD capabilities.
- **vs SOLOIST/SimpleTOD**: Traditional TOD fine-tuning methods can only handle predefined APIs, whereas CoALM achieves open API processing through a unified format.
- **vs ReAct**: ReAct proposes a thought-action-observation reasoning framework, which CoALM embeds into multi-turn dialogue scenarios, representing a natural extension of ReAct in conversational agents.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The unified modeling idea bridging TOD and LA is pioneering, though the technical implementation (format unification + SFT) is straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extremely thorough, featuring three complementary benchmarks, multiple model scales, and detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem definition and convincing analysis.
- **Value**: ⭐⭐⭐⭐⭐ Establishes a benchmark for building unified conversational agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] T1: A Tool-Oriented Conversational Dataset for Multi-Turn Agentic Planning](../../NeurIPS2025/llm_agent/t1_a_tool-oriented_conversational_dataset_for_multi-turn_agentic_planning.md)
- [\[ACL 2025\] GUICourse: From General Vision Language Model to Versatile GUI Agent](guicourse_from_general_vision_language_model_to_versatile_gui_agent.md)
- [\[ACL 2025\] Bel Esprit: Multi-Agent Framework for Building AI Model Pipelines](bel_esprit_multi-agent_framework_for_building_ai_model_pipelines.md)
- [\[ACL 2025\] ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use](toolhop_multi_hop_tool_use.md)
- [\[ACL 2025\] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?](repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_research_claims.md)

</div>

<!-- RELATED:END -->
