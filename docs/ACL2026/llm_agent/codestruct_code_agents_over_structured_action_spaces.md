---
title: >-
  [Paper Note] CodeStruct: Code Agents over Structured Action Spaces
description: >-
  [ACL 2026][LLM Agent][Code Agent] This paper proposes CodeStruct, a framework that redefines code repositories as AST-based structured action spaces, enabling LLM code agents to read and edit code via named program entities rather than raw text fragments. CodeStruct achieves 1.2–5.0% accuracy improvements on SWE-Bench Verified while reducing token consumption by 12–38%.
tags:
  - ACL 2026
  - LLM Agent
  - Code Agent
  - AST-based Structured Actions
  - Code Editing
  - SWE-Bench
  - Action Space
date: 2026-05-08
content_hash: 0f4fff05c1e5ea07
---

# CodeStruct: Code Agents over Structured Action Spaces

**Conference**: ACL 2026
**arXiv**: [2604.05407](https://arxiv.org/abs/2604.05407)
**Code**: [https://github.com/amazon-science/CodeStruct](https://github.com/amazon-science/CodeStruct)
**Area**: LLM Agent / Code Intelligence
**Keywords**: Code Agent, AST-based Structured Actions, Code Editing, SWE-Bench, Action Space

## TL;DR
This paper proposes CodeStruct, a framework that redefines code repositories as AST-based structured action spaces, enabling LLM code agents to read and edit code via named program entities rather than raw text fragments. CodeStruct achieves 1.2–5.0% accuracy improvements on SWE-Bench Verified while reducing token consumption by 12–38%.

## Background & Motivation

**Background**: LLM code agents (e.g., SWE-Agent) are already capable of handling complex repository-level software engineering tasks. Mainstream approaches interact with code through file-reading and text-editing tools, with some systems augmented by repository maps or symbolic indices to improve navigation.

**Limitations of Prior Work**: Existing agents treat code as flat text rather than a structured artifact, resulting in a fundamental abstraction mismatch: reading code either loads entire files and introduces irrelevant context, or slices by line numbers and truncates functions; editing code relies on string-matching substitution, where format drift causes "no match found" errors and repeated patterns cause "multiple matches" errors.

**Key Challenge**: Source code has precise syntactic structure by nature—functions, classes, and methods are all named program entities—yet LLM agents are forced to manipulate these structured objects indirectly via line numbers and string patterns. Enhancement strategies improve "where to look" but do not change the fundamental "how to interact."

**Goal**: To design an AST-based structured action space that allows agents to read and modify code directly through named semantic entities.

**Key Insight**: Human developers reference and modify code by function and class names, not by line numbers. CodeStruct exposes this natural working style directly to LLM agents.

**Core Idea**: Parse the code repository into an AST and provide two structure-aware primitive operations—`readCode` and `editCode`—through which agents locate and manipulate program entities using selectors such as `file.py::ClassName::method`.

## Method

### Overall Architecture
CodeStruct represents a code repository as an AST-driven structured environment. The agent's action space consists of two primitives: `readCode` (structure-aware code retrieval) and `editCode` (structure-aware code modification). Each operation identifies a target AST node via a selector with fuzzy-matching support. The primitives are exposed as standard tool interfaces via the MCP protocol, enabling plug-and-play integration into any agent framework.

### Key Designs

1. **readCode: Structure-Aware Code Retrieval**

    - **Function**: Provides coarse-to-fine code navigation across three modes—directory browsing, file summarization, and entity-level retrieval.
    - **Mechanism**: When the input is a directory, a file listing is returned; when the input is a file without a selector, small files return their full content while large files return a structural summary (top-level entity signatures and scope names); when a selector $\sigma$ is provided, the matching entity node is located in the AST and its complete implementation is returned. Selectors support both unscoped (e.g., `load`) and scoped (e.g., `User.load`) forms, resolved via deterministic name-based fuzzy matching.
    - **Design Motivation**: Conventional line-number-based reading either introduces excessive irrelevant context or truncates functions. Selector-based retrieval guarantees that complete syntactic units are returned, eliminating fragile dependencies on line numbers.

2. **editCode: Structure-Aware Code Modification**

    - **Function**: Performs insert, replace, or delete operations on named AST nodes, automatically maintaining formatting and validating syntactic correctness.
    - **Mechanism**: Given an operation type $\omega \in \{\text{insert}, \text{replace}, \text{removal}\}$ and a selector $\sigma$, the target AST node is located, local indentation context is computed, the transformation is applied, and the modified code is validated via AST parsing—edits with syntax errors are rejected. In replace operations, the agent only needs to supply the signature and new content, without redundantly regenerating unchanged code.
    - **Design Motivation**: The primary problems with text-level editing are the fragility of string matching and redundant generation. `editCode` decouples semantic intent from textual realization—the agent specifies *what* to change, and the tool handles *how* to change it.

3. **Formalization of the AST Action Space**

    - **Function**: Models the multi-step code editing process as a structured action trajectory over AST states, enabling fine-grained behavioral analysis.
    - **Mechanism**: Each `editCode` operation transforms the current AST into a new syntactically valid AST, and multi-step edits form an explicit, analyzable sequence of state transitions.
    - **Design Motivation**: Structured state transitions make agent behavior traceable and debuggable, providing a better analytical foundation for understanding and improving code agents.

### Loss & Training
CodeStruct does not involve model training—it is a tool interface applied at inference time. Exposed as a standard tool via the MCP protocol, it integrates directly with any LLM without modification.

## Key Experimental Results

### Main Results (SWE-Bench Verified, 500 tasks)

| Model | Text Pass@1 | CodeStruct Pass@1 | Gain | Token Reduction |
|-------|-------------|-------------------|------|-----------------|
| GPT-5-nano | 17.2% | 38.0% | +20.8pp | Increased |
| Claude-3.5-Sonnet | 49.0% | 50.2% | +1.2% | 12% |
| GPT-4o | 33.2% | 38.2% | +5.0% | 38% |
| Claude-3.7-Sonnet | 57.4% | 59.4% | +2.0% | 24% |

CodeAssistBench (135 multi-turn programming tasks): all models improve by 0.8–4.4%, with cost reductions of up to 33%.

### Ablation Study

| Analysis Dimension | Finding |
|--------------------|---------|
| Empty patch rate (GPT-5-nano) | Text: 46.6% → CodeStruct: 7.2% (84.5% reduction) |
| Edit failure types | "No match" and "multiple match" errors substantially reduced |
| Per-step token consumption | Reduction more pronounced for read operations (only target entities retrieved) |

### Key Findings
- CodeStruct yields the greatest gains when text-interface fragility—rather than insufficient reasoning capacity—is the primary bottleneck for code agents.
- The reduction in GPT-5-nano's empty patch rate from 46.6% to 7.2% provides the strongest evidence.
- For stronger models (e.g., Claude-3.7-Sonnet), CodeStruct still delivers stable but smaller accuracy gains while significantly reducing token consumption.
- GPT-5-nano exhibits increased token consumption under CodeStruct, because structured operations enable sustained exploration that previously terminated upon edit failures.

## Highlights & Insights
- **Abstraction Alignment Principle**: The abstraction level of a tool interface should align with that of the artifact being manipulated. Code is structured; tools that operate on code should be structured as well. This principle generalizes to agent design in other domains.
- **Tool Design Over Model Capacity**: The 20.8pp improvement on GPT-5-nano demonstrates that, in certain settings, improving tool design is more effective than switching to a larger model.
- **Plug-and-Play Integration via MCP**: By exposing functionality through a standard tool protocol, CodeStruct requires no modification to agent planning or execution logic, substantially lowering the barrier to adoption.

## Limitations & Future Work
- AST parsing currently supports Python only and has not been extended to other programming languages.
- Fuzzy matching may introduce ambiguity in large repositories.
- Syntax validation operates only at the AST level and does not guarantee semantic correctness.
- Integration with agent training remains unexplored—training agents with structured tools from the outset may yield further improvements.

## Related Work & Insights
- **vs. SWE-Agent**: SWE-Agent provides file maps and text-editing tools; CodeStruct upgrades the underlying operations from text-level to AST-level.
- **vs. GumTree**: GumTree computes AST edit scripts for offline comparison; CodeStruct exposes AST operations as real-time decision primitives for agents.
- **vs. Code2Vec**: Code2Vec uses ASTs for code representation learning in single-pass prediction; CodeStruct uses ASTs as the action space for multi-turn interaction.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Using ASTs as the agent action space is a concise yet far-reaching design choice.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Six LLMs, two benchmarks, and detailed failure analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Problem formulation is clear, method description is precise, and experimental analysis is thorough.
- **Value**: ⭐⭐⭐⭐⭐ Highly practical—zero training cost, plug-and-play, and significant performance gains.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] SecureVibeBench: Evaluating Secure Coding Capabilities of Code Agents with Realistic Vulnerability Scenarios](securevibebench_evaluating_secure_coding_capabilities_of_code_agents_with_realis.md)
- [\[ACL 2026\] StructMem: Structured Memory for Long-Horizon Behavior in LLMs](structmem_structured_memory_for_long-horizon_behavior_in_llms.md)
- [\[AAAI 2026\] Reflection-Driven Control for Trustworthy Code Agents](../../AAAI2026/llm_agent/reflection-driven_control_for_trustworthy_code_agents.md)
- [\[ACL 2026\] EA-Agent: A Structured Multi-Step Reasoning Agent for Entity Alignment](ea-agent_a_structured_multi-step_reasoning_agent_for_entity_alignment.md)
- [\[ACL 2026\] From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation](from_query_to_counsel_structured_reasoning_with_a_multi-agent_framework_and_data.md)

<!-- RELATED:END -->
