---
title: >-
  [Paper Note] EDIT-Bench: Evaluating LLM Abilities to Perform Real-World Instructed Code Edits
description: >-
  [ICLR 2026][Code Intelligence][LLM Evaluation] EDIT-Bench transforms in-the-wild instructed code editing requests from nearly 500 real developers—captured via an in-house VSCode plugin—into 540 challenging problems with test harnesses. Evaluating 40 LLMs reveals that this is a difficult benchmark, with only one SOTA model exceeding a 60% success rate.
tags:
  - ICLR 2026
  - Code Intelligence
  - LLM Evaluation
date: 2026-05-08
content_hash: a4606d86a9a0daa6
---
# EDIT-Bench: Evaluating LLM Abilities to Perform Real-World Instructed Code Edits

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=FtL9eEmU6v](https://openreview.net/forum?id=FtL9eEmU6v)  
**Code**: [https://github.com/waynchi/editbench](https://github.com/waynchi/editbench) (Leaderboard: [https://waynechi.com/edit-bench/](https://waynechi.com/edit-bench/))  
**Area**: Code Intelligence / Benchmark  
**Keywords**: Instructed Code Editing, Real-world Data, LLM Evaluation, Context Dependency, Multilingual  

## TL;DR
EDIT-Bench transforms in-the-wild instructed code editing requests from nearly 500 real developers—captured via an in-house VSCode plugin—into 540 challenging problems with test harnesses. Evaluating 40 LLMs reveals that this is a difficult benchmark, with only one SOTA model exceeding a 60% success rate.

## Background & Motivation
- **Background**: Instructed code editing (where a developer highlights a code block and uses natural language to request direct modifications) has become a mainstream interaction mode in AI programming assistants like Copilot, Cursor, and Continue, alongside autocomplete and chat.
- **Limitations of Prior Work**: Existing benchmarks rarely evaluate this capability directly. Code generation benchmarks (HumanEval, MBPP) only test writing code from scratch. The few editing-related datasets either consist of templated problems written by annotators (CanItEdit, EditEval) or are derived from LeetCode/educational problems (Aider Polyglot), which do not reflect the diversity of real-world software development. While "arena-style" evaluations like Chatbot Arena are realistic, they require massive human voting, making them costly to scale.
- **Key Challenge**: Real editing requests are **vague, informal, and context-dependent**. A developer might highlight code and simply say "fix this," expecting the model to infer intent. Conversely, existing benchmarks provide well-defined tasks (e.g., "modify function X to state Y"), creating a significant distribution gap.
- **Goal**: Construct an **automatically evaluable (with test cases)** instructed code editing benchmark rooted in **real-world usage**. The goal is to force models to synthesize multi-source information including user instructions, full source files, highlighted regions, and cursor positions, mimicking a real IDE environment.
- **Core Idea**: **Collect in-the-wild data using a real plugin**. The authors developed an open-source VSCode plugin that mimics Copilot/Cursor editing features. Real editing data was collected from users in exchange for free SOTA model access. Human experts then refined these requests into **difficult problems with test harnesses**, balancing "realism" with "automatic grading."

## Method

### Overall Architecture
The construction of EDIT-Bench follows a pipeline: "Real-world Collection → Manual Filtering → Test Harnessing → Translation Expansion." 2,672 accepted edits were collected from 458 users, filtered down to ~470 challenging problems. Experts then wrote generalizable test cases for 109 of these (EDIT-Bench-core). Finally, comments were translated into five natural languages to expand the set to 540 problems (EDIT-Bench-complete) for evaluating 40 LLMs.

```mermaid
flowchart LR
    A[In-house VSCode Plugin] -->|458 Users| B[2672 Real Accepted Edits]
    B -->|Filtering: Py/JS only / De-duplication / Noise Removal| C[~470 Challenging Problems]
    C -->|Expert Test Harnesses + Dual Review| D[EDIT-Bench-core 109 tasks]
    D -->|GPT-4o Translation of Comments| E[EDIT-Bench-complete 540 tasks]
    E -->|pass@1| F[Evaluation of 40 LLMs]
```

### Key Designs
**1. Collecting in-the-wild editing data via a real plugin.** Instead of hiring annotators to synthesize problems, the authors released an open-source VSCode plugin focused on instructed editing. Users highlight code and write a task description; the plugin records the instruction, code context (highlighted segment, cursor position, prefix, and suffix), the model response, and whether it was accepted. This captured the natural informality of real development—e.g., a bug fix might be requested as "fix this," a pasted error trace, or a natural language description.

**2. Context-dependent problem design.** EDIT-Bench is the first benchmark to combine "user instruction + full source file + highlighted region + cursor position." Real instructions are often too vague to be interpreted without context. While source files can be long ($\ge 10\text{k}$ characters, median $\approx 4.5\text{k}$ tokens), the highlighted parts are small (median 138 tokens). Models must accurately locate the target within a long context while utilizing comments and highlights. Models are required to **regenerate the entire file** to complete the edit.

**3. Converting real requests to evaluable tasks: Expert test harnesses + Dual review.** Since raw data lacks test cases, a team of five senior programmers wrote **environment configurations and test cases** for each problem. Tests were designed to be implementation-agnostic, and overly ambiguous problems were discarded. GPT-4o/Sonnet 3.7 generated reference solutions to screen for PII and logic. While coding agents helped with environment setup (e.g., `conftest.py` or `jest-config.js`), the actual test logic was manually authored to avoid the low-quality pattern-matching often seen in agent-generated tests.

**4. Multilingual expansion.** The core 109 problems initially covered five languages (EN, RU, ZH, PL, ES) unevenly. Borrowing from HumanEval-XL, the authors used GPT-4o to translate comments into five languages (English, Spanish, Russian, Chinese, Portuguese) across Python and Javascript, forming the 540-problem EDIT-Bench-complete set. Translation quality was sampled and verified by native speakers.

## Key Experimental Results

### Comparison with Existing Benchmarks

| Benchmark | # Problems | Source | # NL | Instruction Length | # PL | Code Context Length | Highlight |
|---|---|---|---|---|---|---|---|
| CanItEdit | 105 | Annotators | 1 | $140 \pm 105$ | 3 | $1309 \pm 1116$ | No |
| EditEval | 194 | Annotators | 1 | $99.9 \pm 49.3$ | 1 | $258 \pm 185$ | No |
| Aider Polyglot | 225 | Exercises | 1 | $606 \pm 885$ | 5 | $6184 \pm 6452$ | No |
| **EDIT-Bench** | **540** | **In-the-wild** | **5** | **$238 \pm 738$** | **2** | **$5642 \pm 7567$** | **Yes** |

### Main Results (pass@1 for 40 LLMs)
- Only **1 out of 40** models exceeded 60%: **claude-sonnet-4 reached 66.67%**, ranking first.
- Closed-source models generally outperformed open-source ones: only 4 open-source models appeared in the top 15. The strongest open-source model, **glm-4.6, scored 56.48%**, followed by kimi-k2-0905 and deepseek-chat-v3.1.
- Counter-intuitively, **gpt-5 (medium reasoning) lagged behind gpt-5-mini**, often failing on "simple" tasks like indentation formatting and edge case handling.

### Ablation Study (Effect of Contextual Information)

| Model | Code Only | +Highlight | +Cursor | +Highlight +Cursor |
|---|---|---|---|---|
| claude-sonnet-4 | 62.41 | 64.81 (+2.40) | 63.15 (+0.74) | 64.26 (+1.85) |
| deepseek-chat-v3.1 | 51.48 | 54.26 (+2.78) | 53.15 (+1.67) | 52.78 (+1.30) |
| gemini-2.5-flash | 52.59 | 52.96 (+0.37) | 52.41 (−0.18) | 56.30 (+3.71) |
| kimi-k2-0905 | 54.63 | 56.48 (+1.85) | 52.22 (−2.41) | 58.15 (+3.52) |
| glm-4.6 | 52.96 | 56.48 (+3.52) | 52.22 (−0.74) | 44.81 (−8.15) |

Adding highlighted code improved performance for 5/7 models, but adding cursor position yielded mixed results.

### Key Findings
- **High Variance in Difficulty**: "Hard" problems (solved by $\le 20$ models) had instructions nearly 5x shorter than "easy" ones, forcing models to rely on reasoning over copying.
- **Task Category Performance**: Models were strongest at **bug fixing (avg 52.2%)** but weakest at **optimization (44.6%)** and **feature addition (39.6%)**.
- **Weak Correlation with Existing Benchmarks**: Low correlation with Aider Polyglot ($r=0.24$) and Chatbot Arena Coding ($r=0.11$) suggests that real-world data captures a unique set of challenges.

## Highlights & Insights
- **Balancing Realism and Automation**: Trading free model access for real-world data and then manually harnessing it is a clever mechanism to avoid the costs of human-voting arenas while maintaining authentic distributions.
- **Human-Agent Collaboration**: The insight that agents excel at environment setup but fail at generating rigorous test logic provides a practical methodology for future benchmark construction.
- **Contextual Quantization**: Ablations demonstrate that while highlights are generally useful, the utility of cursor positions is unstable, providing evidence for optimizing context injection in coding assistants.

## Limitations & Future Work
- **Small Scale**: Only 109 unique core problems; expansion to 540 relies heavily on comment translation.
- **Limited Languages**: Focused on Python/JS; others like PHP (18% of original collection) and HTML (7%) were discarded.
- **Expansion Bias**: Translating the same problems into multiple languages may lead to high correlation between test cases, potentially diluting the effective count of independent challenges.

## Related Work & Insights
- **Static vs. Live Benchmarks**: Complements zero-shot generation benchmarks (HumanEval) and live benchmarks (LiveCodeBench) by focusing on editing.
- **Methodological Insight**: Systematizing real user interaction traces into automated benchmarks is a methodology that can be extended to completion, refactoring, and debugging scenarios.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First work to transform in-the-wild instructed editing (with highlights/cursor/multilingual) into an auto-evaluable benchmark.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation of 40 models, including context ablations and difficulty/category breakdowns.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, transparent construction process, and insightful interpretation of results.
- **Value**: ⭐⭐⭐⭐ Addresses a major evaluation gap in AI programming assistants with an extensible leaderboard.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[ICLR 2026\] CodeSense: a Real-World Benchmark and Dataset for Code Semantic Reasoning](codesense_a_real-world_benchmark_and_dataset_for_code_semantic_reasoning.md)
- [\[ACL 2026\] AutoMonitor-Bench: Evaluating the Reliability of LLM-Based Misbehavior Monitor](../../ACL2026/code_intelligence/automonitor-bench_evaluating_the_reliability_of_llm-based_misbehavior_monitor.md)
- [\[ACL 2026\] ReFEree: Reference-Free and Fine-Grained Method for Evaluating Factual Consistency in Real-World Code Summarization](../../ACL2026/code_intelligence/referee_reference-free_and_fine-grained_method_for_evaluating_factual_consistenc.md)
- [\[ACL 2026\] LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software](../../ACL2026/code_intelligence/logiceval_a_systematic_framework_for_evaluating_automated_repair_techniques_for_.md)
- [\[ACL 2025\] CompileAgent: Automated Real-World Repo-Level Compilation with Tool-Integrated LLM-based Agent System](../../ACL2025/code_intelligence/compileagent_automated_real-world_repo-level_compilation_with_tool-integrated_ll.md)

</div>

<!-- RELATED:END -->
