---
title: >-
  [Paper Note] Finch: Benchmarking Finance & Accounting across Spreadsheet-Centric Enterprise Workflows
description: >-
  [ACL 2026][LLM Evaluation][Finance & Accounting] This paper introduces Finch (FinWorkBench), a finance and accounting workflow benchmark constructed from real enterprise environments (e.g., the Enron dataset), comprising 172 composite workflows and 1,710 spreadsheets (27 million cells). Even the strongest model, GPT 5.1 Pro, spending an average of 16.8 minutes per workflow, passes only 38.4% of the workflows, revealing critical gaps in frontier AI agents under realistic enterprise conditions.
tags:
  - ACL 2026
  - LLM Evaluation
  - "Finance & Accounting"
  - Spreadsheets
  - Enterprise Workflows
  - Agent Evaluation
  - Long-Horizon Tasks
date: 2026-05-08
content_hash: a1b84d70a300d251
---

# Finch: Benchmarking Finance & Accounting across Spreadsheet-Centric Enterprise Workflows

**Conference**: ACL 2026
**arXiv**: [2512.13168](https://arxiv.org/abs/2512.13168)
**Code**: [HuggingFace](https://huggingface.co/FinWorkBench)
**Area**: LLM Evaluation
**Keywords**: Finance & Accounting, Spreadsheets, Enterprise Workflows, Agent Evaluation, Long-Horizon Tasks

## TL;DR

This paper introduces Finch (FinWorkBench), a finance and accounting workflow benchmark constructed from real enterprise environments (e.g., the Enron dataset), comprising 172 composite workflows and 1,710 spreadsheets (27 million cells). Even the strongest model, GPT 5.1 Pro, spending an average of 16.8 minutes per workflow, passes only 38.4% of the workflows, revealing critical gaps in frontier AI agents under realistic enterprise conditions.

## Background & Motivation

**Background**: Frontier AI systems (Claude, ChatGPT, Gemini, Copilot) are increasingly embedded in enterprise daily workflows. Finance & Accounting (F&A) is a high-stakes, knowledge-intensive domain critical to every organization. AI-assisted tools are having a growing impact on document drafting, data exploration, and spreadsheet manipulation.

**Limitations of Prior Work**: (1) Real-world F&A tasks are inherently messy — artifacts are interconnected across heterogeneous spreadsheets, PDFs, and other documents, and undergo multi-version collaborative editing; (2) Spreadsheets contain complex structures — cross-sheet references, irregular layouts, merged cells, implicit formula chains, and charts; (3) Workflows are long-horizon — requiring multi-step reasoning spanning data entry, editing, retrieval, computation, modeling, validation, and report generation; (4) Existing benchmarks typically use clean, single-table inputs that fail to reflect real-world complexity.

**Key Challenge**: Can today's frontier AI agents genuinely handle the messy, long-horizon, knowledge-intensive workflows that domain professionals face daily?

**Goal**: To construct the first truly enterprise-grade F&A workflow benchmark, sourced from real enterprise environments and preserving their original multimodal complexity.

**Key Insight**: Mining real workflows from collaborative threads and spreadsheet version histories in the Enron email corpus — "existence precedes essence."

**Core Idea**: Workflows should be observed from real enterprise environments and then formally defined, rather than artificially designed. The benchmark is constructed via three pathways: email thread extraction, version diff analysis, and expert annotation.

## Method

### Overall Architecture

The Finch dataset is constructed through three pathways: (1) mining workflows from enterprise email threads — emails naturally describe business objectives and reference attached files; (2) deriving workflows from diffs between versioned spreadsheets — analyzing changes across consecutive versions to infer underlying goals; (3) designing workflows from final deliverables and reports — expert-authored workflow instructions based on high-quality documents. All data underwent 700+ hours of expert annotation and multiple rounds of quality control.

### Key Designs

1. **Mining Workflows from Email Threads**:

    - Function: Capture workflow intent and context from real collaborative communication.
    - Mechanism: From the Enron email corpus (15,000 files + 500,000 emails), GPT-5 identifies collaborative messages satisfying two conditions — (a) an explicit statement of a business objective, and (b) reference to one or more attached spreadsheets. In strongly grounded cases, both input and reference artifacts are present in the attachments; in weakly grounded cases, only partial artifacts are available and experts supplement the remainder.
    - Design Motivation: Email threads contain the "natural documentation" of workflows — collaborators naturally describe, discuss, and track their work in everyday communication.

2. **Deriving Workflows from Version Diffs**:

    - Function: Discover workflows implicitly encoded in spreadsheet modification histories.
    - Mechanism: Versioned workbook families are collected; an LLM-based diff procedure identifies consecutive versions and infers workflow types (e.g., "date versioning, assumption updates, error correction") along with detailed change descriptions. Human experts verify and refine the results, confirming that the diffs constitute meaningful workflows rather than incidental changes.
    - Design Motivation: Many workflows are not explicitly described in emails but can be "archaeologically" recovered from version histories — a unique data source.

3. **Multi-Dimensional Evaluation Framework**:

    - Function: Enable reliable evaluation of complex spreadsheet artifacts.
    - Mechanism: (a) Human evaluation — experts compare input/reference/model output per workflow and assign binary pass/fail judgments; (b) LLM-as-Judge — supports automated evaluation across three task types: modification tasks (structured diff + compact snapshot + screenshot), generation tasks (full value/formula extraction + screenshot), and QA tasks. Evaluation criteria cover completeness, numerical/logical correctness, avoidance of over-editing, and formatting readability.
    - Design Motivation: Spreadsheet evaluation cannot rely on simple cell-by-cell comparison — equivalent formulas, alternative layouts, and other valid solutions must be accommodated.

### Loss & Training

Finch is an evaluation benchmark. Product-level agents evaluated include ChatGPT (GPT 5.1 Pro) and Claude (Sonnet/Opus 4.5 thinking mode). API-level models include GPT 5.1, Claude Sonnet/Opus 4.5, Gemini 3 Pro, Grok 4, and Qwen 3 Max. SpreadsheetBench is used as the baseline code generation framework.

## Key Experimental Results

### Main Results

| Model / Agent | Workflow Pass Rate |
|---|---|
| GPT 5.1 Pro (human eval) | 38.4% |
| Claude Opus 4.5 | 2nd best, but <50% |
| Gemini 3 Pro | Substantially below GPT 5.1 |
| GPT 5.1 Pro ≤2 tasks | 44.3% |
| GPT 5.1 Pro >2 tasks | 23.5% |
| GPT 5.1 Pro (with PDF/images) | 35.0% |

### Ablation Study

| Complexity Dimension | Impact |
|---|---|
| Task compositionality | ≤2 tasks: 44.3% → >2 tasks: 23.5%; severe error accumulation |
| Multimodal artifacts | Drop to 35.0% when PDFs/images are present |
| Spreadsheet complexity | Median 15K cells; maximum 3.7M cells |
| Tool call count | Median 16; range 6–107 |
| Long-horizon dependencies | Cross-sheet references and implicit formula chains cause frequent failures |

### Key Findings

- Even the strongest agent (GPT 5.1 Pro) passes only 38.4% of workflows on a benchmark requiring 700+ hours of expert annotation.
- Compositionality is the key bottleneck — pass rates for multi-task workflows are nearly half those of single-task workflows.
- Messy spreadsheet structures (merged cells, nested headers, irregular layouts) frequently cause data retrieval errors.
- Agents struggle to reconstruct implicit business logic encoded in spreadsheet formulas.
- LLM-as-Judge shows high agreement with human evaluation, providing a scalable assessment approach.

## Highlights & Insights

- The "existence precedes essence" philosophy of dataset construction is compelling — mining workflows from real enterprise emails and version histories is more authentic than artificial design.
- 92.4% of workflows involve multiple spreadsheets with an average of 8 sheets, far exceeding the scale of existing benchmarks — this reflects genuine enterprise scenarios.
- The 38.4% pass rate is a sobering reminder for the field: AI is far from achieving "automation" in enterprise F&A work.
- The 700+ hours of annotation investment and multiple rounds of quality control ensure high benchmark quality.

## Limitations & Future Work

- Primarily English-language; multilingual enterprise scenarios are not covered.
- Although authentic, the Enron data originates from the early 2000s, and some business practices it reflects may be outdated.
- Binary pass/fail evaluation may be unfair to high-quality partial completions.
- Real-time collaboration and multi-agent scenarios are not covered.

## Related Work & Insights

- **vs. SpreadsheetBench**: The latter is designed for smaller, cleaner spreadsheet tasks; Finch extends to large, messy, enterprise-grade artifacts.
- **vs. DABStep**: The latter focuses on data analysis steps; Finch covers end-to-end composite workflows.
- **vs. WideSearch**: The latter focuses on web search tasks; Finch incorporates such tasks as components of larger workflows.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First truly enterprise-grade F&A workflow benchmark; the methodology of mining workflows from emails and version histories is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple frontier models/agents, human + automated evaluation, detailed complexity analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Dataset construction process is transparent and thorough; statistical analysis is comprehensive.
- Value: ⭐⭐⭐⭐⭐ Provides a much-needed high-quality real-world benchmark for enterprise AI agent evaluation.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] TingIS: Real-time Risk Event Discovery from Noisy Customer Incidents at Enterprise Scale](tingis_real-time_risk_event_discovery_from_noisy_customer_incidents_at_enterpris.md)
- [\[ICCV 2025\] ForCenNet: Foreground-Centric Network for Document Image Rectification](../../ICCV2025/llm_evaluation/forcennet_foreground-centric_network_for_document_image_rectification.md)
- [\[ACL 2026\] LexRel: Benchmarking Legal Relation Extraction for Chinese Civil Cases](lexrel_benchmarking_legal_relation_extraction_for_chinese_civil_cases.md)
- [\[ACL 2026\] ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition](researchbench_benchmarking_llms_in_scientific_discovery_via_inspiration-based_ta.md)
- [\[ACL 2026\] E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task](e2edev_benchmarking_large_language_models_in_end-to-end_software_development_tas.md)

<!-- RELATED:END -->
