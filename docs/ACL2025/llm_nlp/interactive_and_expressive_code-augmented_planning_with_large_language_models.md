---
title: >-
  [Paper Note] Interactive and Expressive Code-Augmented Planning with Large Language Models
description: >-
  [ACL 2025][LLM (Other)][REPL-Plan] This paper proposes REPL-Plan, a top-down planning approach that enables LLMs to interact with an extended REPL (Read-Eval-Print Loop). This method leverages the full expressiveness of code while enabling dynamic error correction and handling of vague subproblems, achieving robust performance on ALFWorld, WebShop, and real-world web navigation tasks.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "REPL-Plan"
  - "LLM-REPL"
  - "code-augmented planning"
  - "interactive decision-making"
  - "task decomposition"
date: 2026-05-08
content_hash: ddc8749315d6eea1
---

# Interactive and Expressive Code-Augmented Planning with Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2411.13826](https://arxiv.org/abs/2411.13826)  
**Area**: LLM/NLP  
**Keywords**: REPL-Plan, LLM-REPL, code-augmented planning, interactive decision-making, task decomposition  

## TL;DR

This paper proposes REPL-Plan, a top-down planning approach that enables LLMs to interact with an extended REPL (Read-Eval-Print Loop). This method leverages the full expressiveness of code while enabling dynamic error correction and handling of vague subproblems, achieving robust performance on ALFWorld, WebShop, and real-world web navigation tasks.

## Background & Motivation

**LLM Planning Challenges**: While LLMs exhibit strong capabilities in commonsense reasoning and interactive decision-making, they often make mistakes in complex, long-horizon planning tasks—frequently resulting in hallucinations and incorrect short-term decisions.

**Limitations of Code-Augmented Planning**: Prior works structure LLM outputs using code to improve planning, including utilizing variables to track information and decomposing subtasks with functions. However, pure code-based approaches suffer from three inherent issues:
   - **Vague Subproblems**: Many tasks require handling unstructured data or subjective judgments (e.g., "purchase the item that best matches the user's description"), which are difficult to address directly with code.
   - **Bottom-Up Coding Style**: Coding typically requires writing subprocess functions before dealing with the main task. This demands precise advance planning, which places high demands on generating accurate code in a single trial.
   - **Coding Errors**: Even for skilled human programmers, writing bug-free code in one go is extremely challenging.

**Inspiration from Human Developers**: Real-world developers use REPLs (such as IPython or Jupyter Notebooks) for interactive programming: entering code line-by-line, inspecting results, and debugging errors. This interactive mode naturally supports dynamic error correction and exploratory development.

## Method

### Overall Architecture

The core of REPL-Plan is **LLM-REPL**—a recursive extension of REPL, where the LLM writes code line-by-line and interacts with the planning environment via code.

### LLM-REPL Design

LLM-REPL adds three key primitive functions to the standard REPL:

| Primitive | Function |
|------|------|
| `[subtask](args)` | Generates a child LLM-REPL to handle the subtask, reusing the historical execution state if it already exists |
| `get_args()` | Retrieves arguments passed by the parent REPL |
| `answer(a)` | Returns the result to the parent REPL, handing execution control back to the parent |

Environment Interaction Primitives:

| Primitive | Function |
|------|------|
| `act(a)` | Executes action $a$ in the environment |
| `get_obs()` | Retrieves the current observation text |

### Key Designs

1. **Recursive Subtask Generation**: When the code calls an undefined function (triggering a `NameError`), a new child LLM-REPL is automatically created to handle this subtask. The child REPL has an independent variable space and communicates local context via `get_args()` and `answer()`.

2. **Top-Down Planning**: Unlike traditional bottom-up coding, the LLM starts with the main task and recursively creates child REPLs to resolve subproblems as they arise—akin to the real-world software engineering practice of "designing the interface first, then implementing the details."

3. **Dynamic Error Correction**: The line-by-line execution nature of REPL allows the LLM to inspect execution feedback and error logs at each step, enabling immediate code revision without rewriting the entire program.

4. **k-Shot REPL Pool**: Keeps a global pool of REPLs that stores code and environments generated in previous tasks and demonstrations. When a new task invokes a REPL of the same name, it can reuse the code/output history, facilitating in-context learning.

### Walkthrough Example

Taking e-commerce search as an example, the main REPL invokes `filter_search(description)` to iterate through search pages. This function in turn calls `filter_page(description)` to filter matching items on the current page, which then calls `parse_items()` to parse page elements and `item_matches(item, description)` to check for matches—the entire process is recursively generated from the top down.

## Key Experimental Results

### ALFWorld (Household Simulation Environment)

| Method | Success Rate (%) | External Memory |
|------|-----------|----------|
| ReAct | 53.7 | No |
| ADaPT | 82.1 | No |
| THREAD | 95.5 | No |
| **REPL-Plan** | **97.0** | No |
| Reflexion | 76.1 | Yes |
| RAP | 85.8 | Yes |

REPL-Plan achieves a 97.0% success rate without using external memory, outperforming all baselines.

### WebShop (E-commerce Navigation)

| Model | Setting | Strategy | Method | Success Rate (%) | Score (%) |
|------|------|------|------|-----------|---------|
| GPT-3.5 | k=3 | Top-3 | THREAD | 49 | 76.3 |
| GPT-3.5 | k=3 | Top-3 | REPL-Plan | 47 | 74.2 |
| GPT-4o-mini | k=10 | Top-3 | THREAD | 21 | 42.1 |
| GPT-4o-mini | k=10 | Top-3 | REPL-Plan | 37 | 69.9 |
| GPT-4o-mini | k=10 | Top-20 | **REPL-Plan** | **52** | **77.1** |

In the simple setting ($k=3$), REPL-Plan performs comparably to THREAD, but exhibits a significant advantage in larger search spaces ($k=10$) and more complex strategies (Top-20).

### Real-World Web Tasks

| Method | Simple Tasks (%) | Complex Tasks (%) |
|------|-------------|-------------|
| ReACT | 86.7 | 17.6 |
| THREAD | 13.3 | 0.0 |
| **REPL-Plan** | **86.7** | **39.6** |

In complex real-world web tasks involving loops, REPL-Plan significantly systemizes and outperforms the baselines (where web observations range from 4k to 20k tokens).

### Ablation Study (WebShop $k=3$, $n=25$)

| Ablation | GPT-3.5 SR (%) | GPT-4o-mini SR (%) |
|------|---------------|-------------------|
| Full Model | 52 | 44 |
| Buggy Demonstrations | 52 | 40 |
| W/o Subtask REPL | 24 | 20 |
| Zero-shot Sub-REPL | 28 | 16 |

- **Recursive subtask REPL is critical**: Performance drops by half after removal.
- **Robust error correction**: Performance remains almost unchanged even after injecting code bugs.
- **Limited zero-shot capability**: Only about half of the trials can correctly infer when demonstrations for a sub-REPL are removed.

## Highlights & Insights

1. **Unifying Code Expressiveness and Dynamics**: REPL-Plan represents the first planning framework that combines full code expressiveness (loops, variables, functions) with dynamic error correction and the capability to handle vague problems.
2. **Top-Down Recursive Decomposition**: Through the recursive generation of LLM-REPL, it naturally realizes an "API-first, implementation-later" development pattern, bypassing the challenges of bottom-up planning.
3. **Robustness to Hallucinations**: Code control flow mitigates the impact of LLM hallucinations—even if an incorrect element ID is predicted, loop structures guarantee that the agent continues searching instead of getting permanently stuck.
4. **Scalability**: The advantages of REPL-Plan are particularly pronounced in tasks with long observations (4k-20k tokens) and high complexity.

## Limitations & Future Work

1. **Reliance on High-Quality Demonstrations**: The capacity to infer sub-REPLs zero-shot is limited, leading to a substantial drop in performance when no demonstrations are provided.
2. **API Call Overhead**: Since each line of code requires LLM inference, recursively generating multiple child REPLs significantly increases API call frequency and costs.
3. **Environmental Constraints**: Currently validated only in text environments; generalizability to multimodal environments remains to be explored.
4. **Quality of Sub-REPL Task Descriptions**: The system relies on the LLM to generate task descriptions for child REPLs; inaccurate descriptions can lead to cascading failures.

## Related Work & Insights

- **LLM Agents**: ReAct (Yao et al., 2022b) and Reflexion (Shinn et al., 2023) primarily use observation-action sequences; THREAD (Schroeder et al., 2024) employs textual divide-and-conquer strategies.
- **Code-Augmented LLMs**: PAL (Gao et al., 2023) leverages code to assist reasoning; ProgPrompt (Singh et al., 2022) utilizes program structures for planning.
- **Recursive Task Decomposition**: ADaPT (Prasad et al., 2024) performs adaptive decomposition; THREAD utilizes tree-like subtask structures.

## Rating

⭐⭐⭐⭐ — Elegant design with clean intuition, systematically integrating human interactive programming experience (REPL) into LLM planning, and showing a distinct edge in real-world web navigation tasks. The recursive REPL design is highly elegant, though warning points include API invocation costs and demo reliance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] INTERACT: Enabling Interactive, Question-Driven Learning in Large Language Models](interact_enabling_interactive_question-driven_learning_in_large_language_models.md)
- [\[ACL 2025\] WarriorCoder: Learning from Expert Battles to Augment Code Large Language Models](warriorcoder_learning_from_expert_battles_to_augment_code_large_language_models.md)
- [\[ACL 2025\] On the Limit of Language Models as Planning Formalizers](limit_llm_planning_formalizer.md)
- [\[ACL 2025\] ToolCoder: A Systematic Code-Empowered Tool Learning Framework for Large Language Models](toolcoder_code_empowered_tool_learning.md)
- [\[ACL 2025\] MIRAGE: Exploring How Large Language Models Perform in Complex Social Interactive Environments](mirage_exploring_how_large_language_models_perform_in_complex_social_interactive.md)

</div>

<!-- RELATED:END -->
