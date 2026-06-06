---
title: >-
  [Paper Note] FS-Researcher: Test-Time Scaling for Long-Horizon Research Tasks with File-System-Based Agents
description: >-
  [ACL 2026][LLM Reasoning][Deep Research] This paper proposes FS-Researcher, a dual-agent deep research framework based on a file system. Through a Context Builder to construct a hierarchical knowledge base and a Report W…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Deep Research"
  - "File System"
  - "Test-Time Scaling"
  - "Knowledge Base Construction"
  - "Dual-Agent Framework"
date: 2026-05-08
content_hash: 4260c2b098bd4f86
---

# FS-Researcher: Test-Time Scaling for Long-Horizon Research Tasks with File-System-Based Agents

**Conference**: ACL 2026  
**arXiv**: [2602.01566](https://arxiv.org/abs/2602.01566)  
**Code**: [https://github.com/Ignoramus0817/FS-Researcher](https://github.com/Ignoramus0817/FS-Researcher)  
**Area**: LLM Reasoning  
**Keywords**: Deep Research, File System, Test-Time Scaling, Knowledge Base Construction, Dual-Agent Framework

## TL;DR

This paper proposes FS-Researcher, a dual-agent deep research framework based on a file system. Through a Context Builder to construct a hierarchical knowledge base and a Report Writer to compose reports by section, it utilizes a persistent workspace to bypass context window limitations. It achieves 53.94 RACE (SOTA) on the DeepResearch Bench and demonstrates a positive test-time scaling effect between context-building computation and report quality.

## Background & Motivation

**Background**: Deep Research is a frontier representative task for LLM Agents, requiring agents to systematically gather evidence from the internet and synthesize it into long-form reports. Commercial deep research products from OpenAI, Google, and Anthropic have demonstrated human-level performance.

**Limitations of Prior Work**: (1) Limited model context lengths cause long-trajectory deep research tasks to easily exceed context capacity, leading to agent execution failure; (2) In existing methods (static pipelines, single-agent workflows), thoughts, tool observations, and report drafts compete for a limited token budget, resulting in incomplete coverage and premature synthesis; (3) Current compression strategies (e.g., summarizing tool observations) extend trajectories but introduce lossy bottlenecks—fine-grained evidence and sources may be lost, and hard context limits still remain.

**Key Challenge**: There is a fundamental contradiction between the volume of information required for deep research tasks (hundreds of webpages, reports with tens of thousands of tokens) and the capacity of the model's context window. Existing methods either truncate information or use lossy compression, failing to truly achieve test-time scaling (allocating more computation to improve quality).

**Goal**: (1) Design a deep research framework scalable beyond the context window; (2) Verify whether the framework can continuously improve report quality by increasing computation; (3) Outperform both closed-source and open-source SOTA on multiple benchmarks.

**Key Insight**: Inspired by programming agents and AI IDEs (Cursor, Claude Code), file-system workspaces serve as effective infrastructure for long-duration tool use and iterative development. This paradigm is migrated to deep research, using the file system as persistent external memory.

**Core Idea**: Replace the context window with a file system as the agent's memory infrastructure—storing information in files rather than retaining it in context, loading as needed, and supporting infinite expansion and cross-session iterative optimization.

## Method

### Overall Architecture

FS-Researcher is a dual-agent framework divided into two stages: (1) The Context Builder receives a research topic, browses the internet like a librarian, writes structured notes, and archives raw webpages to build a hierarchical knowledge base; (2) The Report Writer uses the knowledge base as the sole source of truth to write reports section by section. Both agents share the same file-system workspace and support independent iterative optimization. The workspace includes deliverables (knowledge base/reports) and control files (TODO, Checklist, Log).

### Key Designs

1. **File-System Workspace**:
    - **Function**: Provides persistent external memory to bypass context window limitations.
    - **Mechanism**: The workspace contains two types of files: deliverables (index.md, knowledge_base/, sources/, report.md) and control files (todos, checklist, logs). All files are stored in Markdown format. The agent checks the workspace state at the start of each session, formulates a plan, and executes it. At the end of the session, it reviews status according to the checklist, marking incomplete items as [IN-PROGRESS]. The toolset includes file-system tools (ls, grep, read_file, insert/delete/replace) and web browsing tools (search_web, read_webpage).
    - **Design Motivation**: The file system offers three advantages: (a) mirroring the native environment humans use for complex tasks; (b) storage capacity far exceeding the context window with on-demand access without overflow; (c) persistent and traceable intermediate products, supporting iterative optimization across sessions.

2. **Context Builder**:
    - **Function**: Systematically collects, distills, and archives information into a knowledge base.
    - **Mechanism**: Deliverables include index.md (table of contents with topic breakdown and KB structure), knowledge_base/ (tree-structured directory of notes, each statement cited back to sources/), and sources/ (archived raw webpages). The workflow is non-linear—index.md and knowledge_base/ update dynamically during browsing. Self-checks at the end of each session identify errors, gaps, or conflicts in the knowledge base, marking them for processing. It can run iteratively until the session budget is reached or it passes review.
    - **Design Motivation**: Unlike accumulating facts directly in the context, externalizing information to the file system allows the knowledge base to grow beyond context capacity, while structured organization facilitates on-demand retrieval by the Report Writer.

3. **Report Writer**:
    - **Function**: Writes high-quality research reports section by section based on the knowledge base.
    - **Mechanism**: Web browsing tools are removed, allowing facts to be read only from the knowledge base. It adopts a multi-session writing process: the first session creates an outline (serving as TODOs), and each subsequent session focuses on one section. Section-level reviews (per checklist) occur after each section, followed by a report-level review once completed. Issues result in re-marking sections as [IN-PROGRESS]. There is no session budget limit.
    - **Design Motivation**: Generating an entire report at once often results in a list of facts lacking deep analysis. Sectional writing provides frequent re-anchoring opportunities, combining the knowledge base for local planning and self-correction.

### Loss & Training

As a framework-oriented work, this paper does not involve model training. Standard ReAct architectures are used to drive the two agents: $T_i, A_i = M_\theta(T_{j<i}, A_{j<i}, O_{j<i}, P)$, $O_i = Execute(A_i)$. Various backbone models like GPT-5, Claude-Sonnet-4.5, and Gemini-2.5-Pro are supported. File I/O latency is negligible (<0.03% of total time).

## Key Experimental Results

### Main Results

**DeepResearch Bench Performance Comparison**

| Method | Backbone | Comp. | Insight | Instr. | Read. | RACE |
|------|---------|-------|---------|--------|-------|------|
| OpenAI-DeepResearch | - | 46.46 | 43.73 | 49.39 | 47.22 | 46.45 |
| Gemini-2.5-Pro-DR | - | 49.51 | 49.45 | 50.12 | 50.00 | 49.71 |
| WebWeaver | Qwen3-235B | 51.45 | 51.39 | 50.26 | 48.98 | 50.80 |
| RhinoInsight | Gemini-2.5-Pro | 50.51 | 51.45 | 51.72 | 50.00 | 50.92 |
| **FS-Researcher** | Claude-Sonnet-4.5 | **54.25** | **55.85** | **52.47** | **51.54** | **53.94** |
| **FS-Researcher** | GPT-5 | 51.96 | 54.44 | 52.14 | 51.26 | 52.76 |

**DeepConsult Performance Comparison**

| Method | Win% | Tie% | Lose% | Avg Score |
|------|------|------|-------|-----------|
| OpenAI-DeepResearch | 0.00 | 100.00 | 0.00 | 5.00 |
| WebWeaver | 66.16 | 12.14 | 21.68 | 6.94 |
| **FS-Researcher** (Claude) | **80.00** | 10.42 | 9.58 | **8.33** |

**BrowseComp Accuracy**

| Method | Accuracy |
|------|-------|
| Claude-Sonnet-4.5 (Official) | 43.9% |
| FS-Researcher (Claude) | **55.0%** |
| GPT-5 (Official) | 54.9% |
| FS-Researcher (GPT-5) | **68.0%** |

### Ablation Study

**Module Ablation (GPT-5 Backbone, 10 Sample Queries)**

| Configuration | Comp. | Insight | Instr. | Read. | RACE |
|------|-------|---------|--------|-------|------|
| FS-Researcher (Full) | 51.96 | 54.44 | 52.14 | 51.26 | 52.76 |
| - Persistent Workspace | 48.38(-3.58) | 46.49(-7.95) | 50.78 | 49.92 | 48.69(-4.07) |
| - Dual-Agent→Single-Agent | 40.90(-11.06) | 37.55(-16.89) | 46.30 | 44.78 | 42.41(-10.35) |
| - Section-writing→One-shot | 47.06(-4.90) | 45.64(-8.80) | 50.50 | 46.46 | 47.63(-5.13) |

### Key Findings

- FS-Researcher consistently outperforms closed-source and open-source SOTA across three benchmarks, proving that the framework-level advantages of the file-system paradigm are independent of the backbone model.
- The dual-agent ablation has the largest impact (RACE -10.35), indicating that the separation of evidence collection and report writing is a core design choice.
- Increasing Context Builder turns (3→5→10) continuously improves report quality (Insight from 49.48 to 55.88), though readability slightly declines after 5 turns as increased information density results in a more technical writing style.
- Persistent workspace has the greatest impact on Insight (-7.95), showing that a structured knowledge base is critical for deep analysis.
- Using smaller summarization models to compress context reduces Context Builder costs by 47% with negligible quality loss.

## Highlights & Insights

- The paradigm shift of utilizing a file system as agent external memory—from "putting information in context" to "putting information in files and loading as needed"—is a simple yet profound architectural innovation.
- Dual-agent separation solves a fundamental problem: information gathering and report writing require different cognitive modes; mixing them leads to premature synthesis and shallow exploration.
- Successful verification of the test-time scaling effect (more computation → better reports) provides preliminary evidence for scaling laws in agent systems.

## Limitations & Future Work

- The framework relies on strong backbone models—it requires robust multi-turn planning, web search, and long-form writing capabilities; smaller models may suffer from frequent premature termination.
- There is a trade-off between readability and comprehensiveness—richer knowledge bases lead to more technical writing styles.
- Multi-agent collaboration (e.g., multiple Context Builders searching different sub-topics in parallel) has not been studied.
- Storing raw webpages may involve copyright and privacy concerns.

## Related Work & Insights

- **vs OpenAI/Google Deep Research**: Commercial product technologies are opaque; FS-Researcher serves as a reproducible open-source alternative and outperforms them on multiple benchmarks.
- **vs LangChain Open Deep Research**: Under the same GPT-5 backbone, FS-Researcher improves RACE by +2.16, proving framework contributions are independent of the model.
- **vs Summarization/Compression Methods**: Summarization is lossy and still context-constrained; the file-system approach is lossless and has no upper bound.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The paradigm innovation of using a file system as agent memory is simple and effective; the validation of the test-time scaling effect is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three benchmarks, three backbone models, three ablation sets, scaling analysis, and case studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, detailed methodological description, and logical ablation design.
- Value: ⭐⭐⭐⭐⭐ Provides a reproducible SOTA framework and design principles for deep research agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks](sppo_sequence-level_ppo_for_long-horizon_reasoning_tasks.md)
- [\[ACL 2026\] Efficient Test-Time Scaling via Temporal Reasoning Aggregation](efficient_test-time_scaling_via_temporal_reasoning_aggregation.md)
- [\[ACL 2026\] Parallel Test-Time Scaling for Latent Reasoning Models](parallel_test-time_scaling_for_latent_reasoning_models.md)
- [\[ACL 2026\] Scaling Test-Time Compute to Achieve IOI Gold Medal with Open-Weight Models](scaling_test-time_compute_to_achieve_ioi_gold_medal_with_open-weight_models.md)
- [\[ICLR 2026\] ATTS: Asynchronous Test-Time Scaling via Conformal Prediction](../../ICLR2026/llm_reasoning/atts_asynchronous_test-time_scaling_via_conformal_prediction.md)

</div>

<!-- RELATED:END -->
