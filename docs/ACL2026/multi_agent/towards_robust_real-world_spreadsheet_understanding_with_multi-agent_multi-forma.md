---
title: >-
  [Paper Note] Towards Robust Real-World Spreadsheet Understanding with Multi-Agent Multi-Format Collaboration
description: >-
  [ACL 2026][Multi-Agent][Spreadsheet understanding] Proposes SpreadsheetAgent, a two-stage multi-agent framework that achieves robust real-world spreadsheet understanding without exceeding LLM context limits through progr…
tags:
  - "ACL 2026"
  - "Multi-Agent"
  - "Spreadsheet understanding"
  - "multi-agent framework"
  - "multi-format reasoning"
  - "structured information extraction"
  - "progressive reading"
date: 2026-05-08
content_hash: 219dfbdc99946c2f
---

# Towards Robust Real-World Spreadsheet Understanding with Multi-Agent Multi-Format Collaboration

**Conference**: ACL 2026  
**arXiv**: [2604.12282](https://arxiv.org/abs/2604.12282)  
**Code**: [github](https://github.com/renhouxing/SpreadsheetAgent)  
**Area**: LLM / Natural Language Processing  
**Keywords**: Spreadsheet understanding, multi-agent framework, multi-format reasoning, structured information extraction, progressive reading

## TL;DR

Proposes SpreadsheetAgent, a two-stage multi-agent framework that achieves robust real-world spreadsheet understanding without exceeding LLM context limits through progressive regional reading and cross-verification using code execution, vision, and LaTeX formats.

## Background & Motivation

**Background**: Spreadsheets are the most commonly used data format in corporate reporting, financial auditing, and scientific data management. Existing works in LLM-based table understanding, such as Chain-of-Table, TableGPT, and SheetAgent, mostly treat tables as plain text (Markdown/HTML/LaTeX), ignoring layout semantics.

**Limitations of Prior Work**: (1) Real-world spreadsheets contain rich visual cues like hierarchical headers, multiple sheets, font colors, and merged cells, which plain text formats fail to capture completely; (2) Practical spreadsheets are massive (thousands of rows/columns), exceeding the effective context processing capacity of LLMs; (3) Existing methods tend to lose structural information when loading the entire table at once.

**Key Challenge**: The structural complexity and scale of spreadsheets far exceed the single-pass processing capacity of LLMs. The core issue is how to faithfully preserve layout semantics under a limited context budget.

**Goal**: Design a progressive reading-reasoning paradigm to incrementally parse spreadsheets through multi-agent collaboration.

**Key Insight**: Adopt an "extraction-verification" iterative loop—an extraction agent incrementally parses local regions via code execution, vision, and LaTeX tools, while a verification agent cross-validates the faithfulness of extracted results through vision and LaTeX channels.

**Core Idea**: Use YAML as an intermediate representation to preserve structural semantics and utilize multi-format redundancy to reduce error propagation, ensuring downstream reasoning is based on faithful structured representations.

## Method

### Overall Architecture

A two-stage pipeline: (1) Structure Extraction Stage—the Extraction Agent coordinates Vision Range Agents and LaTeX Range Agents to construct structural sketches and row/column summaries through iterative regional inputs; (2) Solving Stage—performing task-driven reasoning on the extracted intermediate representation. The two stages are tightly coupled through an iteration loop of extraction, verification, and correction.

### Key Designs

1.  **Extraction Module with Three-Tool Collaboration**: The extraction agent possesses three auxiliary tools: code execution (precise parsing of raw values and numerical calculations), a vision range description agent (converting selected regions into images for VLM to extract visual cues like colors/borders), and a LaTeX range description agent (converting regions into LaTeX tables to recover hierarchical headers and alignment structures). This division of labor produces a compact intermediate representation.

2.  **Dual-Channel Cross-Verification Module**: Instead of re-processing the entire table, the verification agent selectively focuses on uncertain or structurally complex regions. The vision verification agent renders the region as an image to check if the extraction matches the visual layout; the LaTeX verification agent renders the region as LaTeX to check structural faithfulness. Results are passed only after both channels verify them; otherwise, correction suggestions are returned for the next round of extraction.

3.  **YAML Intermediate Representation**: YAML is chosen (instead of JSON or free text) for outputting structured extraction results because it is human-readable, supports nested structures, and is simple to parse. Experiments show that the output format significantly affects downstream performance—YAML reduces ambiguity, stabilizes parsing, and improves compatibility with task reasoners compared to JSON.

### Loss & Training

This is an inference framework and does not involve training. GLM-4.5V is used as the VLM and Qwen3-Coder-480B as the LLM, with greedy decoding (temperature=0), a maximum of 4K tokens context per round, and up to 20 tool calls.

## Key Experimental Results

### Main Results (SpreadsheetBench)

| Model | Soft Cell | Soft Sheet | Soft Overall | Hard Cell | Hard Sheet | Hard Overall |
|---|---|---|---|---|---|---|
| GPT-4o | 13.49 | 22.51 | 16.96 | 10.52 | 17.66 | 13.27 |
| ChatGPT Agent | 38.27 | 30.48 | 35.27 | - | - | - |
| GPT-OSS-120B | 30.78 | 27.64 | 29.57 | 24.96 | 23.93 | 24.56 |
| + Ours | **41.30** | **33.14** | **38.16** | **32.80** | **29.34** | **31.47** |
| Qwen3-Coder-480B | 30.36 | 31.05 | 30.63 | 22.82 | 27.07 | 24.45 |
| + Ours | **45.63** | **35.33** | **41.67** | **36.90** | **31.05** | **34.65** |
| Human Performance | 75.56 | 65.00 | 71.33 | 66.67 | 55.00 | 62.00 |

### Ablation Study (Qwen3-30B)

| Configuration | Soft Overall | Hard Overall |
|---|---|---|
| SpreadsheetAgent (Full) | 22.37 | 18.42 |
| w/o Tools & Verify | 20.18 | 16.01 |
| w/ JSON (vs YAML) | 20.76 | 16.34 |
| w/o Structure | 19.70 | 15.46 |
| w/o Verify | 21.45 | 17.54 |
| w/o Vision Tool | 21.45 | 16.89 |
| w/o LaTeX Tool | 20.32 | 16.23 |
| w/o All (baseline) | 16.41 | 12.83 |

### Key Findings

-   SpreadsheetAgent allows GPT-OSS-120B to outperform ChatGPT Agent by 2.89 absolute percentage points (38.16% vs 35.27%).
-   Qwen3-Coder-480B + SpreadsheetAgent achieves a peak of 41.67%, which remains significantly lower than the human performance of 71.33%.
-   The verification module contributes approximately 1 percentage point to the improvement, while structural extraction contributes about 2.7 points.
-   The LaTeX tool contributes more than the vision tool (decreases of 2.05 vs 0.92 respectively when removed).
-   The YAML format provides a 1.61 percentage point improvement over JSON.

## Highlights & Insights

-   **Progressive Reading Paradigm**: Unlike one-shot loading, this approach solves scale issues through iterative regional reading while preserving layout semantics.
-   **Verification is Easier than Solving**: The design philosophy of the verification module is that having a model check existing results is more reliable than generating from scratch.
-   **Multi-format Redundancy**: The three formats (code, vision, and LaTeX) are complementary; no single format can completely capture spreadsheet semantics.

## Limitations & Future Work

-   A massive gap remains compared to human performance (71.33%), indicating that spreadsheet understanding is far from solved.
-   Multi-turn tool calls introduce high computational overhead.
-   The quality of correction suggestions from the verification module depends on the upper-bound capabilities of the VLM/LLM.
-   Future work could explore Reinforcement Learning to optimize tool-calling strategies.

## Related Work & Insights

-   Systematic improvement over multi-agent spreadsheet frameworks like SheetAgent and SheetMind.
-   Incorporates the step-wise reasoning ideas from Chain-of-Table into the structural extraction phase.
-   The design of the verification module can be generalized to other information extraction tasks requiring faithfulness guarantees.

## Rating

-   **Novelty**: ⭐⭐⭐⭐ The framework design of multi-format progressive reading and cross-validation is novel and sound.
-   **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-model comparisons, detailed ablations, and verification across multiple benchmarks are thorough.
-   **Writing Quality**: ⭐⭐⭐⭐ Clear framework descriptions and standardized algorithmic pseudo-code.
-   **Value**: ⭐⭐⭐⭐ Significant contribution to the practical demand for real-world spreadsheet understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking](../../ICLR2026/multi_agent/uis-digger_towards_comprehensive_research_agent_systems_for_real-world_unindexed.md)
- [\[ACL 2026\] LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey](llm-based_human-agent_collaboration_and_interaction_systems_a_survey.md)
- [\[ACL 2026\] Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration](scaling_external_knowledge_input_beyond_context_windows_of_llms_via_multi-agent_.md)
- [\[ACL 2026\] ConSensus: Multi-Agent Collaboration for Multimodal Sensing](consensus_multi-agent_collaboration_for_multimodal_sensing.md)
- [\[ACL 2026\] Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs](topology_matters_measuring_memory_leakage_in_multi-agent_llms.md)

</div>

<!-- RELATED:END -->
