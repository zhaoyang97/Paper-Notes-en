---
title: >-
  [Paper Note] Towards Robust Real-World Spreadsheet Understanding with Multi-Agent Multi-Format Collaboration
description: >-
  [ACL 2026][LLM/NLP][spreadsheet understanding] This paper proposes SpreadsheetAgent, a two-stage multi-agent framework that achieves robust real-world spreadsheet understanding through progressive region-based reading and cross-validation across three formats—code execution, vision, and LaTeX—without exceeding LLM context limits.
tags:
  - ACL 2026
  - LLM/NLP
  - spreadsheet understanding
  - multi-agent framework
  - multi-format reasoning
  - structured information extraction
  - progressive reading
date: 2026-05-08
content_hash: e0d2857dcfdcd54b
---

# Towards Robust Real-World Spreadsheet Understanding with Multi-Agent Multi-Format Collaboration

**Conference**: ACL 2026
**arXiv**: [2604.12282](https://arxiv.org/abs/2604.12282)
**Code**: [github](https://github.com/renhouxing/SpreadsheetAgent)
**Area**: LLM / Natural Language Processing
**Keywords**: spreadsheet understanding, multi-agent framework, multi-format reasoning, structured information extraction, progressive reading

## TL;DR

This paper proposes SpreadsheetAgent, a two-stage multi-agent framework that achieves robust real-world spreadsheet understanding through progressive region-based reading and cross-validation across three formats—code execution, vision, and LaTeX—without exceeding LLM context limits.

## Background & Motivation

**Background**: Spreadsheets are among the most widely used data formats in enterprise reporting, financial auditing, and scientific data management. Prior work on table understanding with LLMs—including Chain-of-Table, TableGPT, and SheetAgent—predominantly treats tables as plain text (Markdown/HTML/LaTeX), largely ignoring layout semantics.

**Limitations of Prior Work**: (1) Real-world spreadsheets contain rich visual cues such as hierarchical headers, multiple sheets, font colors, and merged cells that plain-text formats cannot fully capture; (2) practical spreadsheets are often large-scale (thousands of rows and columns), exceeding the effective context capacity of LLMs; (3) existing methods tend to lose structural information when loading an entire spreadsheet at once.

**Key Challenge**: The structural complexity and scale of real-world spreadsheets far exceed what LLMs can process in a single pass, making it difficult to faithfully preserve layout semantics within a limited context budget.

**Goal**: To design a progressive reading-and-reasoning paradigm that incrementally parses spreadsheets through multi-agent collaboration.

**Key Insight**: An iterative extract-then-verify loop is adopted—an extraction agent incrementally parses local regions using three tools (code execution, vision, and LaTeX), while a verification agent cross-validates extraction fidelity through visual and LaTeX channels.

**Core Idea**: YAML is used as an intermediate representation to preserve structural semantics; multi-format redundant verification reduces error propagation, enabling downstream reasoning to operate on faithful structured representations.

## Method

### Overall Architecture

The framework consists of two stages: (1) **Structure Extraction Stage**—an Extraction Agent coordinates a Visual Range Agent and a LaTeX Range Agent to construct structured sketches and row/column summaries through iterative region inputs; (2) **Solving Stage**—task-driven reasoning is performed over the extracted intermediate representations. The two stages are tightly coupled through an iterative extract–verify–correct loop.

### Key Designs

1. **Three-Tool Extraction Module**: The Extraction Agent has access to three auxiliary tools—code execution (for precise parsing of raw values and numerical computation), a Visual Range Description Agent (which renders a selected region as an image and uses a VLM to extract visual cues such as color and borders), and a LaTeX Range Description Agent (which converts the region to a LaTeX table to recover hierarchical headers and alignment structures). Collaborative division of labor among these tools produces compact intermediate representations.

2. **Dual-Channel Cross-Verification Module**: Rather than reprocessing the entire spreadsheet, the Verification Agent selectively focuses on uncertain or structurally complex regions. The Visual Verification Agent renders a region as an image and checks whether the extracted result matches the visual layout; the LaTeX Verification Agent renders the region as LaTeX and checks structural fidelity. If both channels pass, the extraction is accepted; otherwise, correction suggestions are returned to trigger the next extraction round.

3. **YAML Intermediate Representation**: YAML (rather than JSON or free text) is chosen as the output format for structured extraction results, owing to its human readability, support for nested structures, and ease of parsing. Experiments show that the output format significantly affects downstream performance—YAML reduces ambiguity compared to JSON, stabilizes parsing, and improves compatibility with the task reasoner.

### Loss & Training

This work proposes an inference-time framework and involves no model training. GLM-4.5V is used as the VLM and Qwen3-Coder-480B as the LLM, with greedy decoding (temperature=0), a maximum context of 4K tokens per round, and up to 20 tool-call rounds.

## Key Experimental Results

### Main Results (SpreadsheetBench)

| Model | Soft Cell | Soft Sheet | Soft Overall | Hard Cell | Hard Sheet | Hard Overall |
|---|---|---|---|---|---|---|
| GPT-4o | 13.49 | 22.51 | 16.96 | 10.52 | 17.66 | 13.27 |
| ChatGPT Agent | 38.27 | 30.48 | 35.27 | - | - | - |
| GPT-OSS-120B | 30.78 | 27.64 | 29.57 | 24.96 | 23.93 | 24.56 |
| + SpreadsheetAgent | **41.30** | **33.14** | **38.16** | **32.80** | **29.34** | **31.47** |
| Qwen3-Coder-480B | 30.36 | 31.05 | 30.63 | 22.82 | 27.07 | 24.45 |
| + SpreadsheetAgent | **45.63** | **35.33** | **41.67** | **36.90** | **31.05** | **34.65** |
| Human Performance | 75.56 | 65.00 | 71.33 | 66.67 | 55.00 | 62.00 |

### Ablation Study (Qwen3-30B)

| Configuration | Soft Overall | Hard Overall |
|---|---|---|
| SpreadsheetAgent (full) | 22.37 | 18.42 |
| w/o Tools & Verify | 20.18 | 16.01 |
| w/ JSON (replace YAML) | 20.76 | 16.34 |
| w/o Structure | 19.70 | 15.46 |
| w/o Verify | 21.45 | 17.54 |
| w/o Vision Tool | 21.45 | 16.89 |
| w/o LaTeX Tool | 20.32 | 16.23 |
| w/o All (baseline) | 16.41 | 12.83 |

### Key Findings

- SpreadsheetAgent enables GPT-OSS-120B to surpass ChatGPT Agent by 2.89 absolute percentage points (38.16% vs. 35.27%).
- Qwen3-Coder-480B + SpreadsheetAgent achieves the highest overall score of 41.67%, still well below the human performance of 71.33%.
- The verification module contributes approximately 1 percentage point of improvement; structure extraction contributes approximately 2.7 percentage points.
- The LaTeX tool contributes more than the vision tool (removing each leads to drops of 2.05 vs. 0.92 points, respectively).
- YAML outperforms JSON by 1.61 percentage points.

## Highlights & Insights

- **Progressive Reading Paradigm**: In contrast to one-shot loading, iterative region-based reading addresses the scale problem while preserving layout semantics.
- **Verification Is Easier Than Generation**: The design philosophy of the verification module holds that having a model check an existing result is more reliable than generating from scratch.
- **Multi-Format Redundancy**: Code execution, vision, and LaTeX formats are complementary; no single format alone can fully capture the semantics of a spreadsheet.

## Limitations & Future Work

- A substantial gap remains compared to human performance (71.33%), indicating that spreadsheet understanding is far from solved.
- Multi-round tool invocations introduce considerable computational overhead.
- The quality of correction suggestions from the verification module is bounded by the capabilities of the underlying VLM/LLM.
- Future work may explore reinforcement learning to optimize tool-calling strategies.

## Related Work & Insights

- Systematic improvements over prior multi-agent spreadsheet frameworks such as SheetAgent and SheetMind.
- The stepwise reasoning philosophy of Chain-of-Table is reflected in the structure extraction stage.
- The verification module design is generalizable to other information extraction tasks requiring fidelity guarantees.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The multi-format progressive reading and cross-validation framework design is novel and well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-model comparisons, detailed ablations, and multi-benchmark validation are thorough.
- **Writing Quality**: ⭐⭐⭐⭐ Framework description is clear and algorithmic pseudocode is well-structured.
- **Value**: ⭐⭐⭐⭐ Meaningfully advances the practical needs of real-world spreadsheet understanding.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data](memory-augmented_llm-based_multi-agent_system_for_automated_feature_generation_o.md)
- [\[NeurIPS 2025\] Large Language Models Miss the Multi-Agent Mark](../../NeurIPS2025/llm_nlp/large_language_models_miss_the_multi-agent_mark.md)
- [\[NeurIPS 2025\] SYMPHONY: Synergistic Multi-agent Planning with Heterogeneous Language Model Assemblies](../../NeurIPS2025/llm_nlp/symphony_synergistic_multi-agent_planning_with_heterogeneous_language_model_asse.md)
- [\[ACL 2026\] MulDimIF: A Multi-Dimensional Constraint Framework for Evaluating and Improving Instruction Following in Large Language Models](muldimif_a_multi-dimensional_constraint_framework_for_evaluating_and_improving_i.md)
- [\[ACL 2026\] Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models](style_amnesia_investigating_speaking_style_degradation_and_mitigation_in_multi-t.md)

<!-- RELATED:END -->
