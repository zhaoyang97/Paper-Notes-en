---
title: >-
  [Paper Note] FS-Researcher: Test-Time Scaling for Long-Horizon Research Tasks with File-System-Based Agents
description: >-
  [ACL 2026][LLM Reasoning][Deep Research] This paper proposes FS-Researcher, a file-system-based dual-agent framework for deep research. A Context Builder constructs a hierarchical knowledge base while a Report Writer com…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Deep Research"
  - "File System"
  - "Test-Time Scaling"
  - "Knowledge Base Construction"
  - "Dual-Agent Framework"
date: 2026-05-08
content_hash: cb97585618bde299
---

# FS-Researcher: Test-Time Scaling for Long-Horizon Research Tasks with File-System-Based Agents

**Conference**: ACL 2026
**arXiv**: [2602.01566](https://arxiv.org/abs/2602.01566)  
**Code**: [https://github.com/Ignoramus0817/FS-Researcher](https://github.com/Ignoramus0817/FS-Researcher)  
**Area**: LLM Reasoning
**Keywords**: Deep Research, File System, Test-Time Scaling, Knowledge Base Construction, Dual-Agent Framework

## TL;DR

This paper proposes FS-Researcher, a file-system-based dual-agent framework for deep research. A Context Builder constructs a hierarchical knowledge base while a Report Writer composes reports section by section. By leveraging a persistent workspace to overcome context window limitations, the framework achieves 53.94 RACE (SOTA) on DeepResearch Bench and demonstrates a positive test-time scaling effect between context-building compute and report quality.

## Background & Motivation

**State of the Field**: Deep Research represents a frontier task for LLM agents, requiring agents to systematically gather evidence from the internet and synthesize it into long-form reports. OpenAI, Google, and Anthropic have released commercial deep research products exhibiting human-level performance.

**Limitations of Prior Work**: (1) Model context lengths are finite, and the long-trajectory nature of deep research tasks frequently exceeds context capacity, causing agent execution to be interrupted. (2) In existing approaches (static pipelines, single-agent workflows), thoughts, tool observations, and report drafts compete for a limited token budget, leading to incomplete coverage and premature synthesis. (3) Current compression strategies (e.g., summarizing tool observations) extend trajectories but introduce lossy bottlenecks—fine-grained evidence and source attribution may be lost—while still being subject to hard context limits.

**Root Cause**: There is a fundamental conflict between the volume of information required for deep research tasks (hundreds of web pages, reports spanning tens of thousands of tokens) and the capacity of model context windows. Existing methods either truncate information or apply lossy compression, making genuine test-time scaling (allocating more compute to improve quality) infeasible.

**Paper Goals**: (1) Design a deep research framework that scales beyond context window constraints. (2) Verify whether the framework can continuously improve report quality by increasing compute. (3) Surpass both closed-source and open-source SOTA on multiple benchmarks.

**Starting Point**: Inspired by coding agents and AI IDEs (Cursor, Claude Code)—file-system workspaces serve as effective infrastructure for long-horizon tool use and iterative development. This paradigm is transferred to deep research, using the file system as persistent external memory.

**Core Idea**: Replace the context window with a file system as the agent's memory infrastructure—information is written to files rather than retained in context, loaded on demand, and supports unlimited scaling and cross-session iterative refinement.

## Method

### Overall Architecture

FS-Researcher is a dual-agent framework operating in two phases: (1) the **Context Builder** receives a research topic and, acting like a librarian, browses the internet, writes structured notes, and archives raw web pages to construct a hierarchical knowledge base; (2) the **Report Writer** uses the knowledge base as its sole source of ground truth and composes the report section by section. Both agents share the same file-system workspace and support independent iterative refinement. The workspace contains deliverables (knowledge base / report) and control files (TODO, Checklist, Log).

### Key Designs

1. **File-System Workspace**:

    - **Function**: Provides persistent external memory, overcoming context window limitations.
    - **Mechanism**: The workspace contains two categories of files: deliverables (`index.md`, `knowledge_base/`, `sources/`, `report.md`) and control files (todos, checklist, logs). All files are stored in Markdown format. At the start of each session, the agent inspects the workspace state, formulates a plan, and executes it. At the end of each session, the agent reviews progress against the checklist and marks unsatisfied items as `[IN-PROGRESS]`. The tool set includes file-system tools (`ls`, `grep`, `read_file`, `insert/delete/replace`) and web browsing tools (`search_web`, `read_webpage`).
    - **Design Motivation**: The file system offers three key advantages: (a) it mirrors the native environment humans use to handle complex tasks; (b) its storage capacity far exceeds the context window, with on-demand access and no overflow; (c) intermediate artifacts persist and are traceable, enabling cross-session iterative refinement.

2. **Context Builder**:

    - **Function**: Systematically collects, distills, and archives information into the knowledge base.
    - **Mechanism**: Deliverables include `index.md` (a table of contents with topic decomposition and KB structure), `knowledge_base/` (a tree-structured note directory where each statement carries a citation pointing to `sources/`), and `sources/` (archived raw web pages). The workflow is non-linear—`index.md` and `knowledge_base/` are updated dynamically during browsing. At the end of each session, the agent performs self-inspection to identify errors, gaps, or conflicts in the knowledge base and flags them for follow-up. The agent can run iteratively until the session budget is exhausted or the self-review passes.
    - **Design Motivation**: Unlike accumulating facts directly in context, externalizing information to the file system allows the knowledge base to grow far beyond context capacity, while structured organization enables the Report Writer to retrieve content on demand.

3. **Report Writer**:

    - **Function**: Composes a high-quality research report section by section, drawing solely from the knowledge base.
    - **Mechanism**: Web browsing tools are removed; the agent is only permitted to read facts from the knowledge base. A multi-session writing workflow is adopted: the first session creates an outline (which also serves as the TODO), and each subsequent session selects one section to write. After completing each section, a section-level review is conducted (based on the checklist); once all sections are complete, a report-level review is performed. If issues are identified, relevant sections are re-marked as `[IN-PROGRESS]`. There is no session budget limit.
    - **Design Motivation**: Generating the entire report in a single pass tends to produce fact enumeration lacking analytical depth. Section-by-section writing provides frequent re-anchoring opportunities, enabling local planning and self-correction in conjunction with the knowledge base.

### Loss & Training

This paper presents a framework contribution and does not involve model training. Both agents are driven by the standard ReAct architecture: $T_i, A_i = M_\theta(T_{j<i}, A_{j<i}, O_{j<i}, P)$, $O_i = Execute(A_i)$. Multiple backbone models are supported, including GPT-5, Claude-Sonnet-4.5, and Gemini-2.5-Pro. File I/O latency is negligible (<0.03% of total time).

## Key Experimental Results

### Main Results

**DeepResearch Bench Performance Comparison**

| Method | Backbone | Comp. | Insight | Instr. | Read. | RACE |
|--------|----------|-------|---------|--------|-------|------|
| OpenAI-DeepResearch | - | 46.46 | 43.73 | 49.39 | 47.22 | 46.45 |
| Gemini-2.5-Pro-DR | - | 49.51 | 49.45 | 50.12 | 50.00 | 49.71 |
| WebWeaver | Qwen3-235B | 51.45 | 51.39 | 50.26 | 48.98 | 50.80 |
| RhinoInsight | Gemini-2.5-Pro | 50.51 | 51.45 | 51.72 | 50.00 | 50.92 |
| **FS-Researcher** | Claude-Sonnet-4.5 | **54.25** | **55.85** | **52.47** | **51.54** | **53.94** |
| **FS-Researcher** | GPT-5 | 51.96 | 54.44 | 52.14 | 51.26 | 52.76 |

**DeepConsult Performance Comparison**

| Method | Win% | Tie% | Lose% | Avg Score |
|--------|------|------|-------|-----------|
| OpenAI-DeepResearch | 0.00 | 100.00 | 0.00 | 5.00 |
| WebWeaver | 66.16 | 12.14 | 21.68 | 6.94 |
| **FS-Researcher** (Claude) | **80.00** | 10.42 | 9.58 | **8.33** |

**BrowseComp Accuracy**

| Method | Accuracy |
|--------|----------|
| Claude-Sonnet-4.5 (official) | 43.9% |
| FS-Researcher (Claude) | **55.0%** |
| GPT-5 (official) | 54.9% |
| FS-Researcher (GPT-5) | **68.0%** |

### Ablation Study

**Module Ablation (GPT-5 backbone, 10 sampled queries)**

| Configuration | Comp. | Insight | Instr. | Read. | RACE |
|---------------|-------|---------|--------|-------|------|
| FS-Researcher (Full) | 51.96 | 54.44 | 52.14 | 51.26 | 52.76 |
| − Persistent Workspace | 48.38 (−3.58) | 46.49 (−7.95) | 50.78 | 49.92 | 48.69 (−4.07) |
| − Dual-Agent → Single-Agent | 40.90 (−11.06) | 37.55 (−16.89) | 46.30 | 44.78 | 42.41 (−10.35) |
| − Section-by-Section → One-Shot | 47.06 (−4.90) | 45.64 (−8.80) | 50.50 | 46.46 | 47.63 (−5.13) |

### Key Findings

- FS-Researcher consistently surpasses both closed-source and open-source SOTA across three benchmarks, demonstrating that the file-system paradigm confers framework-level advantages independent of the backbone model.
- The dual-agent ablation yields the largest performance drop (RACE −10.35), indicating that separating evidence collection from report writing is the core design decision.
- Increasing the number of Context Builder sessions (3→5→10) continuously improves report quality (Insight increases from 49.48 to 55.88), though readability slightly declines after 5 sessions as increased information density shifts writing style toward a more technical register.
- The persistent workspace has the largest impact on Insight (−7.95), underscoring the importance of a structured knowledge base for in-depth analysis.
- Compressing context with a smaller summarization model reduces Context Builder costs by 47% with negligible quality loss.

## Highlights & Insights

- The paradigm shift of using the file system as agent external memory—from "information kept in context" to "information stored in files and loaded on demand"—represents a concise yet profound architectural innovation.
- The dual-agent separation addresses a fundamental problem: information gathering and report writing require different cognitive modes, and conflating them leads to premature synthesis and shallow exploration.
- The successful verification of the test-time scaling effect (more compute → better reports) provides preliminary evidence for scaling laws in agent systems.

## Limitations & Future Work

- The framework depends on strong backbone models—robust multi-turn planning, web search, and long-form writing capabilities are required; smaller models may terminate early frequently.
- A trade-off exists between readability and comprehensiveness—a richer knowledge base tends to produce a more technical writing style.
- Multi-agent collaboration (e.g., multiple Context Builders searching different subtopics in parallel) has not been explored.
- Archiving raw web pages may raise copyright and privacy concerns.

## Related Work & Insights

- **vs. OpenAI/Google Deep Research**: Commercial products lack technical transparency; FS-Researcher is a reproducible open-source alternative that outperforms them on multiple benchmarks.
- **vs. LangChain Open Deep Research**: Under the same GPT-5 backbone, FS-Researcher achieves a RACE improvement of +2.16, demonstrating that the framework's contribution is independent of the model.
- **vs. Summarization-Based Compression**: Summarization compression is lossy and still subject to context limits, whereas the file-system approach is lossless and unbounded.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The paradigm innovation of using the file system as agent memory is concise and effective; the verified test-time scaling effect is valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three benchmarks, three backbone models, three ablation studies, scaling analysis, and case studies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clearly articulated, method descriptions are thorough, and ablation design is well-reasoned.
- **Value**: ⭐⭐⭐⭐⭐ Provides a reproducible SOTA framework and design principles for deep research agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Efficient Test-Time Scaling via Temporal Reasoning Aggregation](efficient_test-time_scaling_via_temporal_reasoning_aggregation.md)
- [\[ACL 2026\] Parallel Test-Time Scaling for Latent Reasoning Models](parallel_test-time_scaling_for_latent_reasoning_models.md)
- [\[ICLR 2026\] ATTS: Asynchronous Test-Time Scaling via Conformal Prediction](../../ICLR2026/llm_reasoning/atts_asynchronous_test-time_scaling_via_conformal_prediction.md)
- [\[ICLR 2026\] Understanding the Role of Training Data in Test-Time Scaling](../../ICLR2026/llm_reasoning/understanding_the_role_of_training_data_in_test-time_scaling.md)
- [\[ACL 2026\] Scaling Test-Time Compute to Achieve IOI Gold Medal with Open-Weight Models](scaling_test-time_compute_to_achieve_ioi_gold_medal_with_open-weight_models.md)

</div>

<!-- RELATED:END -->
