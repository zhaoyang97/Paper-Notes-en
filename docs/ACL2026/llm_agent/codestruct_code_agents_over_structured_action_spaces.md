---
title: >-
  [Paper Note] CodeStruct: Code Agents over Structured Action Spaces
description: >-
  [ACL 2026][LLM Agent][Code Agent] This paper proposes the CodeStruct framework, which redefines code repositories as AST-based structured action spaces. By allowing LLM code agents to perform read and edit operations thr…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Code Agent"
  - "AST-structured operations"
  - "code editing"
  - "SWE-Bench"
  - "action space"
date: 2026-05-08
content_hash: e249ee1d717d26cd
---

# CodeStruct: Code Agents over Structured Action Spaces

**Conference**: ACL 2026  
**arXiv**: [2604.05407](https://arxiv.org/abs/2604.05407)  
**Code**: [https://github.com/amazon-science/CodeStruct](https://github.com/amazon-science/CodeStruct)  
**Area**: LLM Agent / Code Intelligence  
**Keywords**: Code Agent, AST-structured operations, code editing, SWE-Bench, action space

## TL;DR
This paper proposes the CodeStruct framework, which redefines code repositories as AST-based structured action spaces. By allowing LLM code agents to perform read and edit operations through named program entities rather than text fragments, the framework improves accuracy by 1.2-5.0% on SWE-Bench Verified and reduces token consumption by 12-38%.

## Background & Motivation

**Background**: LLM code agents (such as SWE-Agent) have demonstrated the capability to handle complex repository-level software engineering tasks. Current mainstream methods interact with code through file reading and text editing tools, with some systems utilizing repository maps or symbol indexes to improve navigation.

**Limitations of Prior Work**: Existing agents treat code as flat text rather than structured artifacts, resulting in a fundamental abstraction mismatch. When reading code, they either load entire files (introducing irrelevant context) or slice by line numbers (causing function truncation). When editing code, they rely on string pattern matching; formatting drift leads to "no match found" errors, while repetitive patterns cause "multiple matches" errors.

**Key Challenge**: Source code naturally possesses a precise syntactic structure—functions, classes, and methods are named program entities—yet LLM agents are forced to manipulate these structured objects indirectly through line numbers and string patterns. Current enhancement schemes only improve "where to look" without changing the fundamental "how to interact."

**Goal**: To design an AST-based structured action space that allows agents to directly read and modify code via named semantic entities.

**Key Insight**: Human developers reference and modify code through function and class names rather than line numbers. CodeStruct exposes this natural workflow directly to LLM agents.

**Core Idea**: The code repository is parsed into an AST, providing two structure-aware primitive operations: `readCode` and `editCode`. Agents locate and manipulate program entities using selectors such as `file.py::ClassName::method`.

## Method

### Overall Architecture
CodeStruct represents the code repository as an AST-driven structured environment. The agent's action space consists of two primitives: `readCode` (structure-aware code retrieval) and `editCode` (structure-aware code modification). Each operation identifies target AST nodes via a selector, which supports fuzzy matching. Exposed through the Model Context Protocol (MCP) as a standard tool interface, it can be integrated plug-and-play into any agent framework.

### Key Designs

1.  **readCode: Structure-aware Code Retrieval**:
    - **Function**: Provides code navigation from coarse to fine—directory browsing, file summaries, and entity-level retrieval modes.
    - **Mechanism**: If the input is a directory, it returns a file list. If the input is a file without a selector, it returns the full text for small files or a structural summary (top-level entity signatures and scope names) for large files. When a selector $\sigma$ is provided, it locates the matching entity node in the AST and returns the complete implementation. Selectors support both unscoped (e.g., `load`) and scoped (e.g., `User.load`) entities using deterministic name-based fuzzy matching.
    - **Design Motivation**: Traditional line-number reading either introduces excessive irrelevant context or truncates functions. Selector-based retrieval ensures the return of complete syntactic units, avoiding fragile dependencies on line numbers.

2.  **editCode: Structure-aware Code Modification**:
    - **Function**: Executes insertion, replacement, or removal operations on named AST nodes while automatically maintaining formatting and validating syntactic validity.
    - **Mechanism**: Given an operation type $\omega \in \{\text{insert}, \text{replace}, \text{removal}\}$ and a selector $\sigma$, the tool locates the target AST node, calculates local indentation context, applies the transformation, and verifies the modified code via AST parsing for syntax errors—rejecting the edit if errors occur. In replacement operations, agents only need to provide the signature and new content, without redundantly re-generating unchanged code.
    - **Design Motivation**: The primary issues with text-level editing are the fragility of string matching and redundant generation. `editCode` decouples semantic intent from textual implementation—the agent specifies "what to change," and the tool handles "how to change."

3.  **Formalization of AST Action Space**:
    - **Function**: Models multi-step code editing processes as structured action trajectories over AST states, supporting fine-grained behavior analysis.
    - **Mechanism**: Each `editCode` operation transforms the current AST into a new syntactically valid AST, forming an explicit, analyzable sequence of state transitions.
    - **Design Motivation**: Structured state transitions make agent behavior traceable and debuggable, providing a superior analytical foundation for understanding and improving code agents.

### Loss & Training
CodeStruct does not involve model training; it serves as a tool interface at inference time. It is exposed as a standard tool via the MCP protocol and can be directly integrated with any LLM.

## Key Experimental Results

### Main Results (SWE-Bench Verified, 500 tasks)

| Model | Text Pass@1 | CodeStruct Pass@1 | Gain | Token Reduction |
| :--- | :--- | :--- | :--- | :--- |
| GPT-5-nano | 17.2% | 38.0% | +20.8pp | Increase |
| Claude-3.5-Sonnet | 49.0% | 50.2% | +1.2% | 12% |
| GPT-4o | 33.2% | 38.2% | +5.0% | 38% |
| Claude-3.7-Sonnet | 57.4% | 59.4% | +2.0% | 24% |

On CodeAssistBench (135 multi-turn programming tasks), all models improved by 0.8-4.4% with cost reductions up to 33%.

### Ablation Study

| Analysis Dimension | Findings |
| :--- | :--- |
| Empty Patch Rate (GPT-5-nano) | Text: 46.6% → CodeStruct: 7.2% (84.5% reduction) |
| Edit Failure Types | "No match" and "multiple matches" errors significantly decreased |
| Token Consumption per Step | Retrieval operations saw more significant reductions (only retrieving target entities) |

### Key Findings
- When text interface fragility (rather than lack of reasoning capability) is the primary bottleneck for code agents, CodeStruct yields the greatest benefits.
- The reduction of GPT-5-nano's empty patch rate from 46.6% to 7.2% is the strongest evidence of its effectiveness.
- For stronger models (e.g., Claude-3.7-Sonnet), it still provides stable but smaller improvements while significantly reducing token consumption.
- Token consumption for GPT-5-nano actually increased with CodeStruct because the structured operations allowed it to continue exploration in cases that would previously have terminated due to failure.

## Highlights & Insights
- **Abstraction Alignment Principle**: The abstraction level of the tool interface should align with the abstraction level of the manipulated object. Code is structured; therefore, tools manipulating code should also be structured. This principle can be generalized to agent design in other domains.
- **Tool Design Superiority over Model Capability**: The 20.8pp improvement for GPT-5-nano demonstrates that in certain scenarios, improving tool design is more effective than switching to a larger model.
- **Plug-and-play Integration via MCP**: Exposure through a standard tool protocol allows integration without modifying the agent's planning or execution logic, significantly lowering the barrier to adoption.

## Limitations & Future Work
- Currently only supports AST parsing for Python and has not been extended to other programming languages.
- Fuzzy matching may produce ambiguities in very large repositories.
- Syntax validation only checks correctness at the AST level and does not guarantee semantic correctness.
- Integration with agent training has not been explored; using structured tools during training might yield even better results.

## Related Work & Insights
- **vs SWE-Agent**: SWE-Agent provides file maps and text editing tools; CodeStruct upgrades low-level operations from the text level to the AST level.
- **vs GumTree**: GumTree calculates AST edit scripts for offline comparison, whereas CodeStruct exposes AST operations as real-time decision primitives for the agent.
- **vs Code2Vec**: Code2Vec utilizes ASTs for code representation learning (single prediction); CodeStruct utilizes ASTs for the action space of multi-turn interactions.

## Rating
- Novelty: ⭐⭐⭐⭐ Using AST as an agent action space is a simple yet far-reaching design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Includes 6 LLMs, 2 benchmarks, and detailed failure analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition, precise methodology description, and in-depth experimental analysis.
- Value: ⭐⭐⭐⭐⭐ High practicality—zero training cost, plug-and-play, and significant improvements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] PersonaAgent: Bridging Memory and Action for Personalized LLM Agents](personaagent_bridging_memory_and_action_for_personalized_llm_agents.md)
- [\[ACL 2026\] StructMem: Structured Memory for Long-Horizon Behavior in LLMs](structmem_structured_memory_for_long-horizon_behavior_in_llms.md)
- [\[AAAI 2026\] Reflection-Driven Control for Trustworthy Code Agents](../../AAAI2026/llm_agent/reflection-driven_control_for_trustworthy_code_agents.md)
- [\[ACL 2026\] Context-Value-Action Architecture for Value-Driven Large Language Model Agents](context-value-action_architecture_for_value-driven_large_language_model_agents.md)
- [\[ACL 2026\] Don't Act Blindly: Robust GUI Automation via Action-Effect Verification and Self-Correction](don39t_act_blindly_robust_gui_automation_via_action-effect_verification_and_self.md)

</div>

<!-- RELATED:END -->
