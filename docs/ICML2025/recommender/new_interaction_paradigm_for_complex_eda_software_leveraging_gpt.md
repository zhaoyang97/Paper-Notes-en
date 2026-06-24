---
title: >-
  [Paper Note] New Interaction Paradigm for Complex EDA Software Leveraging GPT
description: >-
  [ICML 2025][Recommender Systems][LLM-assisted interaction] This work proposes the SmartonAI system, which integrates Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) into the EDA tool KiCad. It achieves task decomposition, document retrieval, and intelligent plugin recommendation and execution through natural language interaction, significantly reducing the learning curve for complex engineering software.
tags:
  - "ICML 2025"
  - "Recommender Systems"
  - "LLM-assisted interaction"
  - "Electronic Design Automation (EDA)"
  - "Retrieval-Augmented Generation (RAG)"
  - "Task decomposition"
  - "Plugin recommendation"
date: 2026-05-08
content_hash: ca5c66b4464b2d23
---

# New Interaction Paradigm for Complex EDA Software Leveraging GPT

**Conference**: ICML 2025  
**arXiv**: [2307.14740](https://arxiv.org/abs/2307.14740)  
**Code**: None  
**Area**: Recommender Systems  
**Keywords**: LLM-assisted interaction, Electronic Design Automation (EDA), Retrieval-Augmented Generation (RAG), Task decomposition, Plugin recommendation

## TL;DR

This work proposes the SmartonAI system, which integrates Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) into the EDA tool KiCad. It achieves task decomposition, document retrieval, and intelligent plugin recommendation and execution through natural language interaction, significantly reducing the learning curve for complex engineering software.

## Background & Motivation

While EDA (Electronic Design Automation) tools such as KiCad, Cadence, and Altium Designer are powerful, they suffer from several core limitations:

**Steep learning curve**: Novice users must spend significant time familiarizing themselves with complex interfaces and workflows, resulting in extremely high entry barriers.

**Fragmented documentation**: Official documentation, community forums, and tutorials are scattered across various platforms, often outdated or written primarily for expert users.

**Lack of adaptive support**: Existing assistance methods, such as GUI scripting and design wizards, lack flexibility and interactivity, failing to adapt dynamically to user contexts.

**High trial-and-error costs**: Users must engage in repetitive trial-and-error when encountering specific scenario-based issues, leading to low efficiency.

Meanwhile, LLMs (such as GPT-4, Claude 3, Gemini 2.5, LLaMA-3, Qwen2.5, etc.) have demonstrated powerful capabilities in natural language understanding, tool calling, and code generation, offering a new technical pathway to address these issues. However, the application of LLMs in domain-specific engineering tools remains insufficient—EDA workflows involve complex procedural logic, structured design hierarchies, and tightly coupled UI operations, presenting unique challenges for the deployment of general-purpose LLMs.

## Method

### Overall Architecture

SmartonAI adopts a modular design, primarily consisting of two core components:

| Component | Function | Key Technology |
|------|------|----------|
| **Chat Plugin** | Multi-turn conversational task decomposition and document retrieval | Hierarchical LLM classification + RAG |
| **OneCommandLine Plugin** | Intent-based plugin recommendation and automatic execution | Semantic matching + Parameterized execution |
| **DocHelper** | Retrieval-augmented document localization subsystem | Hybrid retrieval (BM25 + Dense retrieval) |

The entire system translates user intents into specific design operations through a unified natural language interface, supporting multi-language interaction and feedback-based incremental learning.

### Key Designs

#### 1. Chat Plugin: Interactive Task Decomposition

The core of the Chat Plugin is a two-stage cascaded LLM architecture: **Main-Sub GPT** and **QA GPT**.

**Main-Sub GPT Hierarchical Classification Module**:

- **MainGPT**: Receives user natural language queries and classifies intents into 20 predefined macro-task categories (e.g., "netlist validation", "footprint adjustment", etc.). It is based on instruction-tuned LLMs (such as Qwen2.5, LLaMA-3), combined with custom prompt templates and task ontologies.
- **SubGPT**: Receives the macro-task prediction results and selects one or multiple domain-specific subtasks from a curated task database via dense retrieval. This modular decomposition enables interpretable routing, few-shot prompt specialization, and plug-play extensibility.

**QA GPT Multi-turn Q&A Module**:

- Input: The selected subtask triggers the document synthesis pipeline, gathering context from DocHelper and formatting it into a structured prompt template along with the user query.
- Maintains conversation state across multi-turn user interactions.
- Utilizes a hybrid strategy of retrieval-based prompting, RAG localization, and constrained decoding to enhance factuality and clarity.
- Supports dynamic feedback injection: Users can flag responses as "unsatisfied" and provide clarifying constraints, allowing the system to modify the retrieval or generation pipeline via structured system prompts.

#### 2. OneCommandLine Plugin: Intelligent Plugin Recommendation and Execution

The workflow of the OneCommandLine Plugin is as follows:

1. **Intent Understanding**: The user inputs requirements in natural language.
2. **Semantic Matching**: The system recommends appropriate plugins from the KiCad plugin library through semantic matching.
3. **Parameter Collection**: Necessary parameter inputs are collected interactively.
4. **Automatic Execution**: The plugin is executed with minimal manual intervention.
5. **Feedback Correction**: A feedback-based correction loop is supported.

This module supports multilingual input and plugin autocompletion, achieving end-to-end automation from user intent to tool execution.

#### 3. DocHelper: Retrieval-Augmented Document Localization

DocHelper serves as the knowledge foundation of the entire system, providing retrieval-augmented localization for task-aware Q&A:

**Document Preprocessing**:

- Chunks HTML and Markdown files, using semantic boundary-based dynamic window sizes to generate overlapping segments.
- Embeds each segment using a Transformer encoder (such as BGE-M3, E5-large), which is then stored in a FAISS vector database.
- Appends metadata tags (e.g., tool version, component type).

**Retrieval Pipeline**:

- Query construction: Formulated via a learned retriever-query generator pipeline or statically templated from subtask descriptors.
- Hybrid ranking: Combines BM25 sparse scoring and dense similarity scoring.
- Context filtering: Filters based on task types and user contexts.
- Assembly injection: Assembles the retrieved results into a unified context block to inject into the prompt.

**Advanced Features**:

| Function | Description |
|------|------|
| Context Distillation | Uses GPT-4 or compression-enhanced LLMs for concise prompt-level summarization. |
| Source Attribution | Inline citations or reference links to the original HTML source. |
| Incremental Retrieval | Feedback-based query rewriting to overcome initial retrieval failures. |

### Loss & Training

The repository system primarily focuses on engineering integration and does not involve traditional end-to-end training. Its core strategies include:

1. **Instruction Tuning**: MainGPT and SubGPT are instruction-tuned based on pre-trained LLMs (such as Qwen2.5, LLaMA-3, etc.), combined with custom prompt templates.
2. **Few-Shot Learning**: Achieves few-shot generalization through carefully designed prompt templates and task ontologies.
3. **Incremental Learning**: The system continuously optimizes through user feedback, supporting dynamic feedback injection and query rewriting.
4. **Hybrid Retrieval Optimization**: A fused ranking strategy of BM25 + dense retrieval, balancing exact matching and semantic understanding.

## Key Experimental Results

### Main Results

The paper reports preliminary experimental results, evaluated primarily from the perspective of user experience and efficiency gains:

| Evaluation Dimension | Metric | SmartonAI | Traditional Method | Gain |
|----------|------|-----------|----------|------|
| Novice Onboarding Time | Completing the first PCB design task | Significantly shortened | Baseline | Large Gain |
| Plugin Discovery Efficiency | Time to find the correct plugin | Achieved via natural language | Manual search | Qualitative leap |
| Multi-turn Dialogue Satisfaction | User feedback score | High | N/A | — |
| Multilingual Support | Language coverage | Multilingual | English-only docs | Extended coverage |

### Ablation Study

| Configuration | Key Metric | Description |
|------|----------|------|
| Without DocHelper (Pure LLM) | Accuracy drop | RAG localization is critical for domain-specific Q&A |
| Without Hierarchical Decomposition (Single GPT) | Reduced routing accuracy | The two-stage classification improves the accuracy of task understanding |
| Without Feedback Mechanism | Lower user satisfaction | Dynamic feedback injection improves the interaction experience |
| BM25 Retrieval Only | Insufficient semantic recall | The hybrid retrieval strategy outperforms the single method |
| Dense Retrieval Only | Insufficient exact matching | The hybrid strategy balances precision and semantics |

### Key Findings

1. **RAG is critical**: Pure LLMs show limited accuracy on EDA domain-specific questions; the retrieval-augmented capabilities of DocHelper significantly improve the factuality and operability of responses.
2. **Hierarchical decomposition is effective**: The two-stage architecture of 20 macro-task categories and subtask refinement enables interpretable intent routing, outperforming direct answers from a single model.
3. **Advantages of multi-turn interaction**: Maintaining conversation states allows the system to process progressively complex design problems rather than relying on one-off answers.
4. **Feedback loop improvement**: The query rewriting and constraint injection mechanisms triggerable upon user dissatisfaction significantly improve the quality of subsequent responses.

## Highlights & Insights

1. **Pioneering deep integration of LLM + EDA**: To the best of the authors' knowledge, SmartonAI is the first system to tightly integrate multi-turn conversational LLMs with real-time KiCad tool calling, plugin discovery, RAG document retrieval, and adaptive HTML document assembly.
2. **Significance of engineering paradigm**: It not only addresses EDA challenges but also provides a blueprint for an AI-assisted interaction paradigm that can be generalized to other complex domain-specific software.
3. **Hybrid retrieval strategy**: The combination of BM25 + dense retrieval is highly effective in engineering documentation scenarios, where BM25 handles precise terminology matching while dense retrieval captures semantic similarity.
4. **Modular design philosophy**: The decoupled design of the Chat Plugin and the OneCommandLine Plugin allows the system to easily scale to new EDA tools or other domains.
5. **Closed-loop from recommendation to execution**: The system not only recommends plugins but also automatically collects parameters and executes them, achieving genuine end-to-end automation.

## Limitations & Future Work

1. **Limited experimental scale**: Currently, only preliminary results are provided, lacking large-scale quantitative evaluations and user studies.
2. **Exclusively integrated with KiCad**: The system is only integrated with KiCad at present, and its migration to commercial tools like Cadence or Altium remains unverified.
3. **Limitations of 20 macro-task categories**: Predefined categories might not cover all EDA scenarios and require continuous expansion.
4. **LLM hallucination issues**: Although mitigated by RAG, LLM hallucinations remain a potential risk in engineering scenarios that demand high precision.
5. **Real-time challenges**: The latency induced by multiple LLM calls and retrieval steps might impact user experience, especially during complex task decomposition.
6. **Lack of quantitative comparison with baseline systems**: There is a lack of direct comparative experiments against existing EDA assistant tools or general LLM agent frameworks.

## Related Work & Insights

- **GitHub Copilot / Codex**: Pioneers in natural language programming. SmartonAI extends similar ideas to the domain of engineering tools.
- **HuggingGPT / LangChain / Auto-GPT**: Representative LLM orchestration frameworks. SmartonAI customizes upon them specifically for GUI-intensive tools.
- **RAG series works (REALM, Atlas, RETRO)**: The theoretical foundations of retrieval enhancement. DocHelper serves as their practical implementation in engineering documentation scenarios.
- **DeepPCB / DreamPlace**: Pioneers of ML applications in the EDA field but focused on back-end optimization, whereas SmartonAI addresses front-end human-computer interaction.
- **Insights**: The approach of this work can be migrated to the design of intelligent assistant systems for other complex software (e.g., CAD, simulation tools, IDEs). The core is a trinity architecture of "task decomposition + domain knowledge retrieval + tool execution closed-loop."

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|----------|
| Novelty | 4 | First to deeply integrate LLM+RAG into EDA tools, pioneering a new interaction paradigm |
| Technical Depth | 3 | Primarily focused on system integration, with limited individual technical innovations |
| Experimental Thoroughness | 2 | Only preliminary results are provided, lacking quantitative evaluation and comparative experiments |
| Value | 4 | Directly addresses usability pain points of engineering software, and the paradigm is highly generalizable |
| Writing Quality | 3 | Clearly structured, but the experimental section is quite thin |
| **Overall** | **3.2** | The direction is valuable and the system design is reasonable, but the experimental validation is insufficient |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Aligning LLMs by Predicting Preferences from User Writing Samples](aligning_llms_by_predicting_preferences_from_user_writing_samples.md)
- [\[ICML 2025\] RLTHF: Targeted Human Feedback for LLM Alignment](rlthf_targeted_human_feedback_for_llm_alignment.md)
- [\[ICML 2025\] How to Set AdamW's Weight Decay as You Scale Model and Dataset Size](how_to_set_adamws_weight_decay_as_you_scale_model_and_dataset_size.md)
- [\[ICML 2025\] Position: The Right to AI](the_right_to_ai.md)
- [\[ICML 2025\] PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model](parm_multi-objective_test-time_alignment_via_preference-aware_autoregressive_rew.md)

</div>

<!-- RELATED:END -->
