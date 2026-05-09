---
title: >-
  [Paper Note] Automating Complex Document Workflows via Stepwise and Rollback-Enabled Operations
description: >-
  [AAAI 2026][LLM Agent][document workflows] This paper proposes AutoDW, a framework that automates complex document workflows through stepwise planning (generating one API call at a time) combined with adaptive rollback (parameter-level and API-level). On DWBench—a benchmark of 250 sessions and 1,708 instructions—AutoDW achieves 90% instruction-level and 62% session-level completion rates, surpassing the strongest baseline by 40% and 76%, respectively.
tags:
  - AAAI 2026
  - LLM Agent
  - document workflows
  - multi-step operations
  - rollback mechanism
  - error recovery
  - AutoDW
date: 2026-05-08
content_hash: c4bfc1d14135c485
---

# Automating Complex Document Workflows via Stepwise and Rollback-Enabled Operations

**Conference**: AAAI 2026
**arXiv**: [2512.04445](https://arxiv.org/abs/2512.04445)
**Code**: None
**Area**: LLM Agents
**Keywords**: document workflows, multi-step operations, rollback mechanism, error recovery, AutoDW

## TL;DR

This paper proposes AutoDW, a framework that automates complex document workflows through stepwise planning (generating one API call at a time) combined with adaptive rollback (parameter-level and API-level). On DWBench—a benchmark of 250 sessions and 1,708 instructions—AutoDW achieves 90% instruction-level and 62% session-level completion rates, surpassing the strongest baseline by 40% and 76%, respectively.

## Background & Motivation

- **Background**: LLMs have demonstrated automation capabilities in code generation, data science, and web tasks, yet long-chain workflow automation for document processing (Word editing, format conversion, etc.) remains challenging. Existing document agents perform poorly—e.g., GPT-4 in PPTC achieves only a 6% session completion rate.
- **Limitations of Prior Work**: Real-world document workflows involve multi-step, interdependent instructions (e.g., "set table header → fill second row → merge cells"). Existing agents generate all API calls at once via a predefined plan without adapting to evolving document states, causing cascading failures whenever a single step errs.
- **Key Challenge**: A fundamental gap exists between the ambiguity of natural-language instructions (e.g., "add a header" may correspond to multiple APIs) and the precision required by document operations (API parameters must exactly match the current document state).
- **Key Insight**: Decompose workflows into atomic operations executed and verified step by step, coupled with a two-layer rollback mechanism for automatic error correction to prevent error cascades.

## Method

### Overall Architecture

AutoDW consists of three core modules: (1) **Stepwise Planning**—generating one sub-instruction and its corresponding API call at a time; (2) **API Execution & State Tracking**—executing APIs in a Python runtime and extracting the document state; (3) **Adaptive Rollback**—verifying whether execution results align with user intent and triggering parameter-level or API-level rollback upon mismatch.

### Key Designs

1. **Stepwise Planning**

   - Two-stage generation: first decompose the user instruction into atomic sub-instructions (each completable by a single API call), then generate the specific API call.
   - Sub-instructions bridge the semantic gap between natural language and API functionality while enabling intent classification to narrow the API search space.
   - Intent classification: a fine-tuned 178M BERT model classifies instructions into 8 categories (content creation / modification / table / image / chart / formatting / document structure / document lifecycle), achieving 98% test accuracy.
   - Top-3 intents are retained rather than top-1, improving robustness to ambiguous instructions.

2. **Document State Tracking**

   - Document state is modeled as a 7-tuple: document metadata, paragraph elements, table elements, image elements, page layout, interactive elements, and document styles.
   - After each API execution, the complete document state is programmatically extracted, providing precise change descriptions for subsequent verification.
   - State parsing failures are treated as invalid executions and trigger API-level rollback, preventing further planning based on erroneous states.

3. **Adaptive Rollback**

   - **Change Analysis**: Compares document states before and after execution, detecting changes across six dimensions—structure, content, formatting, style, tables, and hyperlinks.
   - **Alignment Verification**: An LLM verifier assesses whether the state changes align with the sub-instruction, returning a binary decision, confidence score, and explanation.
   - **Parameter-Level Rollback**: Retains the selected API but updates its parameters based on the verifier's explanation.
   - **API-Level Rollback**: Completely reselects the API; triggered when parameter-level rollback also fails.
   - Single-round rollback (parameter-level → API-level) is the default; experiments confirm diminishing marginal returns beyond one round.

4. **DWBench Benchmark Construction**

   - 250 multi-turn sessions, 1,708 manually annotated instructions, and 74 APIs.
   - Average of 34.8 API calls per session (range: 15–75) and 5.1 API calls per instruction.
   - Correctness metric: an LLM judge evaluates semantic equivalence between the post-execution document state and the ground-truth state.

### Loss & Training

- The BERT intent classifier is fine-tuned on 3,315 instruction–intent pairs with no overlap with DWBench.
- The verifier confidence threshold of 0.6 is determined via sensitivity analysis as the optimal point balancing false negatives and false positives.
- The rollback strategy requires no additional training—it relies entirely on the LLM's reasoning capability.

## Key Experimental Results

### Main Results

| Method | Instruction-Level Accuracy | Session-Level Accuracy | # APIs | Token Usage |
|---|---|---|---|---|
| Retrieval-only | 13.84% | 4.40% | 4.82 | 29.6k |
| Reasoning-only | 39.93% | 25.20% | 5.12 | 31.6k |
| Hybrid (PPTC) | 64.46% | 35.20% | 5.30 | 36.5k |
| AutoDW | **90.33%** | **62.00%** | 5.21 | 42.8k |

### Ablation Study

| LLM Backbone | Instruction-Level Accuracy | Session-Level Accuracy | Easy / Medium / Hard |
|---|---|---|---|
| Qwen-Plus | 82.82% | 53.60% | 86.3 / 83.1 / 79.0 |
| DeepSeek-v3 | 90.33% | 62.00% | 94.5 / 90.0 / 86.3 |
| Gemini-2.5-Pro | Among best | Among best | High / High / High |
| GPT-4.1 | Among best | Among best | High / High / High |

### Key Findings

- **76% improvement in session-level completion**: from 35.2% (Hybrid) to 62.0% (AutoDW), at the cost of only 25.6% additional token usage.
- **Hard tasks (>6 APIs) lag the overall average by only 4.4%**: demonstrating AutoDW's stability on long-chain complex tasks.
- **Strong cross-LLM robustness**: all four LLM backbones perform well; even the weakest, Qwen-Plus, achieves 82.8% instruction-level accuracy.
- **Cost-effectiveness of rollback**: single-round two-layer rollback is the optimal strategy; additional rounds yield diminishing returns.
- **~60% of rollbacks occur at format-conversion steps**: document format handling remains a weak point for LLMs.

## Highlights & Insights

- **Generality of the "stepwise + rollback" paradigm**: beyond document automation, this paradigm is transferable to any multi-step execution task such as code generation and data pipelines.
- **Completeness of the 7-tuple document state representation**: precise state tracking is the foundation of the rollback mechanism—accurate verification is impossible without accurate state.
- **Efficiency of the 178M BERT intent classifier**: the design principle of delegating fixed classification to a small model and flexible reasoning to a large model is worth emulating.

## Limitations & Future Work

- Currently limited to Word documents (.docx); Excel, PowerPoint, PDF, and other formats are not covered.
- The 74 APIs cover common operations but fall far short of the full complexity of real-world Office APIs.
- The verifier's confidence threshold calibration relies on empirical tuning; adaptive thresholding warrants exploration.
- The session-level completion rate of 62%, while substantially ahead of baselines, still leaves considerable room for improvement.

## Related Work & Insights

- **vs. PPTC (PPT automation baseline)**: PPTC uses a predefined plan with a rule-based mapper and has no error recovery capability; AutoDW's stepwise planning and adaptive rollback yield robust performance across varying task complexity.
- **vs. DocPilot / TableTalk (human-in-the-loop)**: these systems rely on human verification at each step; AutoDW replaces human validation with an LLM verifier to achieve full automation.

## Rating

| Dimension | Score | Rationale |
|---|---|---|
| Novelty | ⭐⭐⭐⭐ | Stepwise planning combined with two-layer rollback is a novel and practical design for document agents. |
| Technical Depth | ⭐⭐⭐⭐ | The 7-tuple state tracking, 6-dimensional change analysis, and two-layer rollback constitute a complete system design. |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | Large-scale benchmark of 250 sessions, 4 LLM backbones, difficulty gradients, and ablation studies. |
| Value | ⭐⭐⭐⭐⭐ | Directly addresses practical pain points in office automation; a 90% instruction completion rate approaches production readiness. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] A2Flow: Automating Agentic Workflow Generation via Self-Adaptive Abstraction Operators](a2flow_automating_agentic_workflow_generation_via_self-adaptive_abstraction_oper.md)
- [\[AAAI 2026\] Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning](beyond_react_a_planner-centric_framework_for_complex_tool-au.md)
- [\[NeurIPS 2025\] Automated Multi-Agent Workflows for RTL Design](../../NeurIPS2025/llm_agent/automated_multi-agent_workflows_for_rtl_design.md)
- [\[AAAI 2026\] MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models](medla_a_logic-driven_multi-agent_framework_for_complex_medic.md)
- [\[ICLR 2026\] FeatureBench: Benchmarking Agentic Coding for Complex Feature Development](../../ICLR2026/llm_agent/featurebench_benchmarking_agentic_coding_for_complex_feature_development.md)

</div>

<!-- RELATED:END -->
