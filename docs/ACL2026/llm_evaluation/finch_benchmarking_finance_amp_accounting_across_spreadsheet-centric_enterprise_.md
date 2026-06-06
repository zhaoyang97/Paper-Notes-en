---
title: >-
  [Paper Note] Finch: Benchmarking Finance & Accounting across Spreadsheet-Centric Enterprise Workflows
description: >-
  [ACL 2026][LLM Evaluation][Finance & Accounting] This paper introduces Finch (FinWorkBench), a finance and accounting workflow benchmark constructed from real-world enterprise environments (such as the Enron dataset). It…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Finance & Accounting"
  - "Spreadsheets"
  - "Enterprise Workflows"
  - "Agent Evaluation"
  - "Long-horizon Tasks"
date: 2026-05-08
content_hash: e87a5cc7f943717b
---

# Finch: Benchmarking Finance & Accounting across Spreadsheet-Centric Enterprise Workflows

**Conference**: ACL 2026  
**arXiv**: [2512.13168](https://arxiv.org/abs/2512.13168)  
**Code**: [HuggingFace](https://huggingface.co/FinWorkBench)  
**Area**: LLM Evaluation  
**Keywords**: Finance & Accounting, Spreadsheets, Enterprise Workflows, Agent Evaluation, Long-horizon Tasks

## TL;DR

This paper introduces Finch (FinWorkBench), a finance and accounting workflow benchmark constructed from real-world enterprise environments (such as the Enron dataset). It includes 172 composite workflows and 1,710 spreadsheets (27 million cells). Even the strongest GPT 5.1 Pro, spending an average of 16.8 minutes, passed only 38.4% of the workflows, revealing severe deficiencies of cutting-edge AI agents in authentic enterprise scenarios.

## Background & Motivation

**Background**: Frontier AI systems (Claude, ChatGPT, Gemini, Copilot) are increasingly embedded in daily enterprise workflows. Finance and Accounting (F&A) is a high-risk, knowledge-intensive domain critical to every organization. AI assistant tools have a growing impact on document drafting, data exploration, and spreadsheet operations.

**Limitations of Prior Work**: (1) Real F&A work is inherently messy—artifacts are interconnected across heterogeneous spreadsheets, PDFs, and other documents, undergoing collaborative editing across multiple versions; (2) Spreadsheets contain complex structures—cross-sheet references, irregular layouts, merged cells, implicit formula chains, and charts; (3) Workflows are long-horizon—requiring multi-step reasoning that covers data entry, editing, retrieval, calculation, modeling, validation, and report generation; (4) Existing benchmarks often use clean, single-table inputs, failing to reflect real-world complexity.

**Key Challenge**: Can today's frontier AI agents truly handle the messy, long-horizon, knowledge-intensive workflows that professionals face daily?

**Goal**: Construct the first truly enterprise-grade F&A workflow benchmark sourced from real enterprise environments, maintaining original multimodal complexity.

**Key Insight**: Workflows should be observed in real enterprise environments before being formally defined, rather than being manually designed. "Existence precedes essence."

**Core Idea**: The benchmark is constructed through three paths: email thread extraction, version difference analysis, and expert annotation.

## Method

### Overall Architecture

The Finch dataset is obtained via three construction paths: (1) Mining workflows from enterprise email threads—where business objectives and attachments are naturally described; (2) Deriving workflows from differences in versioned spreadsheets—analyzing continuous version changes to infer underlying goals; (3) Designing workflows from final deliverables and reports—where expert instructions are written based on high-quality files. All workflows underwent over 700 hours of expert annotation and multiple rounds of quality control.

### Key Designs

1.  **Mining Workflows from Email Threads**:
    *   **Function**: Captures workflow intent and context within real collaboration.
    *   **Mechanism**: From the Enron email corpus (15,000 files + 500,000 emails), GPT-5 identifies collaborative messages that meet two conditions: (a) explicitly stated business goals, and (b) reference to one or more spreadsheet attachments. In strongly grounded cases, both input and reference artifacts are in the attachments; in weakly grounded cases, only partial artifacts are available and require expert supplementation.
    *   **Design Motivation**: Email threads contain "natural documentation" of workflows—collaborators naturally describe, discuss, and track work during daily communication.

2.  **Deriving Workflows from Version Differences**:
    *   **Function**: Discovers workflows latent within spreadsheet modification histories.
    *   **Mechanism**: Versioned workbook families are collected, and LLM-based diff programs identify continuous versions to infer workflow types (e.g., "date versioning, assumption updates, error corrections") and detailed change descriptions. Human experts verify and refine these—confirming that differences constitute meaningful workflows rather than accidental changes.
    *   **Design Motivation**: Many workflows are not explicitly described in emails but can be "excavated" through version history—a unique data source.

3.  **Multidimensional Evaluation Framework**:
    *   **Function**: Supports reliable evaluation of complex spreadsheet artifacts.
    *   **Mechanism**: (a) Human Evaluation—experts compare inputs, references, and model outputs per workflow for a binary pass/fail; (b) LLM-as-Judge—supports automated evaluation for three task types: modification (structured diff + compact snapshots + screenshots), generation (full value/formula extraction + screenshots), and QA. Evaluation focuses on completeness, numerical/logical correctness, avoidance of over-editing, and format readability.
    *   **Design Motivation**: Spreadsheet evaluation cannot be a simple cell-by-cell comparison—multiple reasonable solutions such as equivalent formulas or alternative layouts may exist.

### Loss & Training

Finch is an evaluation benchmark. Products/Agents evaluated: ChatGPT (GPT 5.1 Pro), Claude (Sonnet/Opus 4.5 Thinking Mode). API models: GPT 5.1, Claude Sonnet/Opus 4.5, Gemini 3 Pro, Grok 4, Qwen 3 Max. SpreadsheetBench was used as the baseline code generation framework.

## Key Experimental Results

### Main Results

| Model/Agent | Workflow Pass Rate |
| :--- | :--- |
| GPT 5.1 Pro (Human Eval) | 38.4% |
| Claude Opus 4.5 | Second strongest but <50% |
| Gemini 3 Pro | Significantly lower than GPT 5.1 |
| GPT 5.1 Pro ≤ 2 tasks | 44.3% |
| GPT 5.1 Pro > 2 tasks | 23.5% |
| GPT 5.1 Pro (incl. PDF/Image) | 35.0% |

### Ablation Study

| Complexity Dimension | Impact |
| :--- | :--- |
| Task Composability | ≤2 tasks 44.3% → >2 tasks 23.5%; severe error accumulation |
| Multimodal Artifacts | Drops to 35.0% when including PDF/Images |
| Spreadsheet Complexity | Median 15K cells, max 3.7 million cells |
| Tool Call Frequency | Median 16 times, range 6–107 times |
| Long-horizon Dependency | Cross-sheet references and implicit formula chains cause frequent failures |

### Key Findings

*   Even the strongest agent (GPT 5.1 Pro) passed only 38.4% on a benchmark with over 700 hours of expert annotation.
*   Composability is a major bottleneck—pass rates for multi-task workflows are nearly half those of single-task workflows.
*   Messy spreadsheet structures (merged cells, nested headers, irregular layouts) frequently lead to data retrieval errors.
*   Agents struggle to reconstruct implicit business logic encoded in spreadsheet formulas.
*   LLM-as-Judge shows high alignment with human evaluation, providing a scalable evaluation solution.

## Highlights & Insights

*   The "existence precedes essence" philosophy for dataset construction is compelling—mining workflows from real enterprise emails and version histories is more authentic than manual design.
*   With 92.4% of workflows involving multiple spreadsheets and an average scale of 8 sheets, it far exceeds existing benchmarks—reflecting actual enterprise scenarios.
*   The 38.4% pass rate is a sobering reminder for the industry—AI is far from achieving "automation" in enterprise F&A work.
*   The investment of over 700 hours in annotation and multiple rounds of quality control ensures the high quality of the benchmark.

## Limitations & Future Work

*   Primarily English-based, without coverage of multilingual enterprise scenarios.
*   Enron data, while authentic, is dated (2000s); some business practices may be obsolete.
*   Binary pass/fail for workflow evaluation may be unfair to high-quality work that is only partially completed.
*   Real-time collaboration and multi-agent scenarios are not yet covered.

## Related Work & Insights

*   **vs SpreadsheetBench**: The latter is designed for smaller, cleaner spreadsheet tasks; Finch extends to large, messy enterprise-grade artifacts.
*   **vs DABStep**: The latter focuses on data analysis steps; Finch covers end-to-end composite workflows.
*   **vs WideSearch**: The latter focuses on web search tasks; Finch integrates search as a component of larger workflows.

## Rating

*   **Novelty**: ⭐⭐⭐⭐⭐ First real-world enterprise F&A workflow benchmark; novel methodology for mining workflows from email/version history.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multiple frontier models/agents, human + automated evaluation, detailed complexity analysis.
*   **Writing Quality**: ⭐⭐⭐⭐⭐ Transparent and detailed dataset construction process, comprehensive statistical analysis.
*   **Value**: ⭐⭐⭐⭐⭐ Provides a much-needed high-quality authentic benchmark for evaluating enterprise AI agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] AgentEval: DAG-Structured Step-Level Evaluation for Agentic Workflows with Error Propagation Tracking](agenteval_dag-structured_step-level_evaluation_for_agentic_workflows_with_error_.md)
- [\[ACL 2026\] Fin-Bias: Comprehensive Evaluation for LLM Decision-Making under human bias in Finance Domain](fin-bias_comprehensive_evaluation_for_llm_decision-making_under_human_bias_in_fi.md)
- [\[ACL 2026\] AJ-Bench: Benchmarking Agent-as-a-Judge for Environment-Aware Evaluation](aj-bench_benchmarking_agent-as-a-judge_for_environment-aware_evaluation.md)
- [\[ACL 2026\] Personalized Benchmarking: Evaluating LLMs by Individual Preferences](personalized_benchmarking_evaluating_llms_by_individual_preferences.md)
- [\[ICCV 2025\] ForCenNet: Foreground-Centric Network for Document Image Rectification](../../ICCV2025/llm_evaluation/forcennet_foreground-centric_network_for_document_image_rectification.md)

</div>

<!-- RELATED:END -->
