---
title: >-
  [Paper Note] Bel Esprit: Multi-Agent Framework for Building AI Model Pipelines
description: >-
  [ACL 2025][LLM Agent][AI pipeline construction] Proposes Bel Esprit, a multi-agent conversational framework. Through a four-step collaboration of Mentalist (requirement clarification) $\rightarrow$ Builder (pipeline construction) $\rightarrow$ Inspector (validation) $\rightarrow$ Matchmaker (model mapping), it automatically transforms vague natural language requirements from users into multi-model AI pipeline graphs, achieving 25.2% EM and 37.0 GED (with GPT-4o Builder) on 44…
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "AI pipeline construction"
  - "Multi-Agent framework"
  - "graph generation"
  - "pipeline validation"
  - "model orchestration"
date: 2026-05-08
content_hash: 8328af4f09cc96e9
---

# Bel Esprit: Multi-Agent Framework for Building AI Model Pipelines

**Conference**: ACL 2025  
**arXiv**: [2412.14684](https://arxiv.org/abs/2412.14684)  
**Code**: [https://belesprit.aixplain.com](https://belesprit.aixplain.com)  
**Area**: LLM Agent / AI Pipeline Automation  
**Keywords**: AI pipeline construction, Multi-Agent framework, graph generation, pipeline validation, model orchestration

## TL;DR

Proposes Bel Esprit, a multi-agent conversational framework. Through a four-step collaboration of Mentalist (requirement clarification) $\rightarrow$ Builder (pipeline construction) $\rightarrow$ Inspector (validation) $\rightarrow$ Matchmaker (model mapping), it automatically transforms vague natural language requirements from users into multi-model AI pipeline graphs, achieving 25.2% EM and 37.0 GED (with GPT-4o Builder) on 441 pipeline test cases.

## Background & Motivation

**Background**: Complex AI tasks (e.g., multimodal content moderation, multilingual video dubbing) often require chaining multiple models into pipelines, such as Automatic Speech Recognition $\rightarrow$ Translation $\rightarrow$ Text-to-Speech. Existing AutoML efforts focus on single-model selection, architecture search, and hyperparameter tuning, but lack a systematic approach for automatic orchestration of multi-model pipelines.

**Limitations of Prior Work**: Existing agentic workflow generation methods mainly focus on writing LLM prompts or sorting simple tool functions. Evaluations are restricted to classic reasoning tasks like math, programming, or QA, and do not involve combining cross-modal AI models. Furthermore, user requirements are often ambiguous (e.g., unspecified input languages or output formats), making direct pipeline generation highly error-prone.

**Key Challenge**: Pipeline construction is inherently a scientific-reasoning-driven graph generation problem—requiring an understanding of AI functions' input/output specifications, modal compatibility, and task-decomposition logic—whereas LLMs are prone to errors in long-context scientific reasoning.

**Goal**: To automatically generate correct multi-model AI pipelines starting from vague user natural language queries.

**Key Insight**: To design a multi-agent collaborative framework that first clarifies user requirements, then progressively constructs the pipeline, and finally validates and populates the models.

**Core Idea**: To decompose pipeline construction into four sequential phases: requirement clarification, branch-based graph generation, dual syntactic/semantic validation, and model matchmaking, each managed by a dedicated sub-agent.

## Method

### Overall Architecture

The system comprises four sub-agents: Mentalist (requirement analysis) $\rightarrow$ Builder (pipeline construction) $\rightarrow$ Inspector (pipeline validation, which can loop back to Builder for revisions) $\rightarrow$ Matchmaker (model mapping). The core workflow progressively refines the user query into structured specifications, generates a pipeline DAG based on these specifications, and finally assigns concrete models to each functional node.

### Key Designs

1. **Mentalist (Requirement Clarification Agent)**:

    - **Function**: Eliminates ambiguity in user queries through conversational interaction to extract structured input/output specifications.
    - **Mechanism**: Consists of three sub-modules—Query Clarifier (identifies missing information via dialogue), Specification Extractor (extracts parameters like name, modality, and language from refined queries to form a tabular specification), and Attachment Matcher (maps user-uploaded files to the correct input nodes in the pipeline).
    - **Design Motivation**: User requests are often incomplete (e.g., "dub my video into French" without specifying the input language); generating a pipeline without clarification leads to numerous errors.

2. **Builder (Pipeline Construction Agent + Chain-of-Branches)**:

    - **Function**: Generates pipeline graphs (nodes = AI functions/input/output, edges = data flow) based on refined queries and structured specifications.
    - **Mechanism**: Proposes the Chain-of-Branches strategy—for a pipeline with $ outputs, it generates $ branches one by one. Each branch represents a path from input to output, and new branches can reuse existing nodes to reduce redundancy. It also introduces three special nodes: Router (routing by modality), Decision (conditional routing), and Script (running Python code).
    - **Design Motivation**: Generating the entire graph in one step is prone to hallucinations and structural inconsistency; generating branch-by-branch reduces the complexity of a single step.

3. **Inspector (Pipeline Validation Agent)**:

    - **Function**: Performs dual syntactic and semantic validation on the Builder's output, feeding errors back to the Builder for iterative correction.
    - **Mechanism**: Syntactic verification validates graph constraints (e.g., modality matching—audio cannot directly connect to translation nodes); some errors can be corrected automatically, while complex errors require reconstruction. Semantic verification generates natural language summaries for each branch, and the LLM determines whether they satisfy user specifications.
    - **Design Motivation**: LLMs often make mistakes in long-context reasoning (e.g., missing translation steps, leading to language mismatches), which necessitates an independent validation step.

### Evaluation Protocol Design

Defines two pipeline evaluation metrics: Exact Match (EM, determining exact match based on the VF2 graph isomorphism algorithm) and Graph Edit Distance (GED, calculating node/edge insertion/deletion/substitution operations, with each operational weight set to 1.0). A dataset of 441 pipeline scenarios is established (82 manually created + 359 synthetically expanded).

## Key Experimental Results

### Main Results

| Framework Configuration (GPT-4o Builder) | EM (%) | GED (%) |
|---------------------------|--------|---------|
| Builder only | 15.7 | 65.1 |
| + Query Clarifier | 25.1 | 44.4 |
| + Specification Extractor | 26.0 | 41.4 |
| + Chain-of-Branches | 25.2 | 40.3 |
| + Syntactic Inspector | 25.6 | 38.3 |
| + Semantic Inspector | 25.2 | 37.0 |

### Comparison of Different Builder LLMs

| Builder LLM | EM (%) | GED (%) |
|-------------|--------|---------|
| GPT-4o (Full Config) | 25.2 | 37.0 |
| Llama 3.1 405B (Full Config) | 20.3 | 48.9 |
| Llama 3.1 70B (Full Config) | 19.4 | 53.9 |
| Llama 3.1 8B | <3.0 | — |

### Key Findings
- Compared to Builder only, the complete framework improves EM by +9.5% and reduces GED by -28.1%.
- Mentalist yields the largest improvement for ambiguous queries, while Chain-of-Branches performs best on large-scale pipelines.
- Semantic inspection occasionally introduces negative impacts for weaker models (due to unnecessary graph duplication).
- Errors mainly stem from node substitution (parameter mismatch or incorrect node type), which accounts for the highest proportion.
- As the pipeline scale increases, generation becomes more difficult, but Chain-of-Branches effectively mitigates this issue.

## Highlights & Insights
- **Formalization of pipeline construction**—Formalizing the multi-model orchestration problem as a scientific-reasoning-driven graph generation task is an early systematic work in this direction. Chain-of-Branches effectively reduces the single-step generation complexity through branch decomposition.
- **Elegant multi-agent collaborative architecture**—Each sub-agent solves a specific challenge in pipeline construction (ambiguity, construction, validation, matchmaking), adhering to the design principle of separation of concerns.
- **Practical evaluation system**—The dual metrics of EM + GED combined with the VF2 graph isomorphism algorithm establish a reusable evaluation standard for pipeline generation tasks.

## Limitations & Future Work
- Highly ambiguous queries remain a bottleneck: even with Mentalist, it still fails when critical input/output specifications are missing.
- Limited AI function pool: with 70+ predefined functions, expanding it increases prompt length and inference costs.
- Inspector does not validate the code generated by the Script node.
- Only generates static pipelines, without extending to dynamic workflows of autonomous agents.
- Performance of small models (8B) is unacceptable, indicating a heavy reliance on strong LLMs.

## Related Work & Insights
- **vs HuggingGPT**: HuggingGPT lets LLMs call HuggingFace models but lacks a validation mechanism; Bel Esprit features the Inspector for dual-checking.
- **vs AutoAgents**: Focuses on automated agent generation, but evaluations are limited to QA/math; Bel Esprit is tailored for cross-modal pipelines.
- **vs TaskWeaver/LangGraph**: Provides frameworks but requires manual pipeline design; Bel Esprit automatically generates them from natural language.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of pipeline construction formalization + Chain-of-Branches + multi-agent validation is highly novel.
- Experimental Thoroughness: ⭐⭐⭐ 441 data entries + systematic ablation + qualitative analysis, though the scale is somewhat limited.
- Writing Quality: ⭐⭐⭐⭐ Clear formalization, intuitive examples, and expressive illustrations.
- Value: ⭐⭐⭐⭐ Holds direct engineering value for automatic AI pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Table-Critic: A Multi-Agent Framework for Collaborative Criticism and Refinement in Table Reasoning](table_critic_multi_agent.md)
- [\[ACL 2025\] Select, Read, and Write: A Multi-Agent Framework of Full-Text-based Related Work Generation](select_read_and_write_a_multi-agent_framework_of_full-text-based_related_work_ge.md)
- [\[ACL 2025\] MIND: A Multi-agent Framework for Zero-shot Harmful Meme Detection](mind_a_multi-agent_framework_for_zero-shot_harmful_meme_detection.md)
- [\[ACL 2025\] AndroidGen: Building an Android Language Agent under Data Scarcity](androidgen_agent_data_scarcity.md)
- [\[ACL 2025\] METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling](metal_a_multi-agent_framework_for_chart_generation_with_test-time_scaling.md)

</div>

<!-- RELATED:END -->
