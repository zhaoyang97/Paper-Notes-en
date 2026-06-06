---
title: >-
  [Paper Note] PROTEA: Offline Evaluation and Iterative Refinement for Multi-Agent LLM Workflows
description: >-
  [ACL2026][Multi-Agent][Multi-agent workflows] PROTEA is an offline debugging platform for multi-agent LLM workflows. It localizes degradation in final answers to specific nodes through node-level evaluation…
tags:
  - "ACL2026"
  - "Multi-Agent"
  - "Multi-agent workflows"
  - "Offline evaluation"
  - "Node evaluation"
  - "Prompt iteration"
  - "LLM-as-a-judge"
date: 2026-05-08
content_hash: ba0b4c1595d04d3a
---

# PROTEA: Offline Evaluation and Iterative Refinement for Multi-Agent LLM Workflows

**Conference**: ACL2026  
**arXiv**: [2605.18032](https://arxiv.org/abs/2605.18032)  
**Code**: Undisclosed  
**Area**: LLM Agent / Multi-Agent Workflows / Prompt Debugging  
**Keywords**: Multi-agent workflows, Offline evaluation, Node evaluation, Prompt iteration, LLM-as-a-judge

## TL;DR
PROTEA is an offline debugging platform for multi-agent LLM workflows. It localizes degradation in final answers to specific nodes through node-level evaluation, backward generation of intermediate expectations, and editable prompt revisions, creating a closed-loop verification process.

## Background & Motivation
**Background**: Increasingly, LLM applications are not single prompt calls but workflow graphs composed of specialized LLM roles for intent analysis, retrieval, planning, ranking, and generation. Frameworks like AutoGen and LangGraph have facilitated the development of these graph-based systems, enhancing output controllability.

**Limitations of Prior Work**: The cost of multi-node decomposition is debugging complexity. When the final answer is incorrect, the root cause often lies in an upstream intermediate output that is subsequently amplified by downstream nodes. Developers must manually parse long traces, guess which node to modify, manually edit prompts, and re-run the system.

**Key Challenge**: Existing evaluation frameworks largely provide end-to-end scores, and observability platforms record traces, but the closed loop from "evaluation evidence to local repair to regression verification" still relies on manual effort. In real-world products, reference labels are often available only for the final answer, not for every intermediate node.

**Goal**: To build a unified interface that allows developers to run multi-agent workflows on fixed offline test sets, evaluate intermediate nodes, locate bottlenecks, view editable prompt revisions, and immediately compare behavior and score trajectories before and after modification.

**Key Insight**: PROTEA does not attempt fully autonomous workflow structure optimization. Instead, it focuses on a developer-in-the-loop debugging experience: the system is responsible for generating evidence and candidate modifications, while humans are responsible for inspecting, editing, accepting, or rolling back changes.

**Core Idea**: Invert the logic from the final answer reference to infer "what each intermediate node should have produced," use node-level rubrics for scoring and highlight problematic nodes on the workflow graph, then convert evaluator rationales into local prompt revisions.

## Method
The core of PROTEA is an evaluate → inspect → revise → re-evaluate loop on a fixed test set. It treats a multi-agent workflow as a DAG, where each node has its own prompt, input/output, and evaluation criteria. After execution, it saves complete traces, node scores, rationales, generated references, and prompt versions for comparison and playback across iterations.

### Overall Architecture
A PROTEA project consists of three parts: workflow specifications (which can be imported from saved projects or LangGraph), an offline test set (potentially with final answer references), and evaluator settings for each node (including rubrics, judge prompts, and thresholds). After loading a project, the developer runs the current workflow, triggering Auto Evaluate. The system displays the pass/warn/fail status of each node on the graph. Upon selecting a node, the right panel displays node outputs, references, evaluation rationales, suggested revisions, and a before/after prompt diff. Once the developer accepts or edits a modification, the system re-runs the test set and displays the score history.

### Key Designs
1. **Backward Node Evaluation**:
	- **Function**: Generates inspectable candidate references for intermediate nodes when only final answer references are available.
	- **Mechanism**: For a node $v$ in the DAG, the system uses the node's instructions, output format, position in the graph, direct child node requirements, and the final answer reference to generate $\hat{y}_v$ as the expected output. Final nodes use the final answer reference directly; intermediate nodes prioritize manual node references if available, falling back to backward generation or format-based templates otherwise.
	- **Design Motivation**: Real teams rarely maintain labels for all intermediate nodes. While backward generation is not equivalent to human ground truth, it decomposes end-to-end supervision into inspectable local goals, significantly lowering the barrier to diagnosis.

2. **Graph-level Diagnosis & Node Sorting**:
	- **Function**: Allows developers to see potential failure points without reading the entire trace from scratch.
	- **Mechanism**: Each node is assigned a score $\sigma_d(v) \in [0, 1]$ by an evaluator based on multiple criteria, which are weighted to get $s(v) \in [0, 1]$. Default thresholds are set to pass $\ge 0.8$, warn $\ge 0.55$, and fail otherwise. The interface sorts nodes by fail, warn, and pass status, with lower scores prioritized within the same status.
	- **Design Motivation**: Workflow debugging requires "suggestions on where to look," without stripping developers of their judgment. Graph status and rationales provide a starting point, while humans decide whether to implement changes.

3. **Local Prompt Revision and Automated Re-testing**:
	- **Function**: Converts evaluator rationales into adoptable, editable, and reversible prompt modifications.
	- **Mechanism**: The prompt-revision module receives the current node's instructions, evaluation rationales, and improvement suggestions to generate revised instructions and brief explanations. The system requires maintaining variable names and output formats while avoiding the inclusion of specific test case content. After accepting a modification, the system re-runs the same offline suite to compare behaviors.
	- **Design Motivation**: Autonomous prompt optimization can easily become a black-box search. PROTEA limits modifications to selected nodes and presents them as before/after diffs, which is better suited for team reviews and regression management.

### Loss & Training
PROTEA is a system tool and does not train a new model. Its optimization objectives are derived from node-level judge scores and final task metrics on offline test sets. The Auto Loop mode repeats the evaluate → revise → re-evaluate cycle for a fixed number of rounds, retaining prompt revisions only when repeated checks show improvement and stable behavior.

The evaluation strategy is rubric-based LLM-as-a-judge. Comparing node outputs with references produces criterion scores, an overall score, a short rationale, and an improvement direction. For binary-like metrics such as numerical exact-match, the paper notes that evaluator feedback can be too sparse, necessitating "partial-credit" criteria like intermediate facts, reasoning setup, and format validity.

## Key Experimental Results

### Main Results
PROTEA was evaluated through developer-in-the-loop iterations on two near-production internal workflows and a small-scale quantitative assessment via an automated iteration stress test.

| Scenario | Workflow Scale | Metric | Initial Performance | After PROTEA | Note |
|------|------------|------|----------|-----------|------|
| Enterprise Document Review | 5 nodes | item-level accuracy | 64.3% | 83.9% | Revisions focused on explicit intermediate outputs and tighter rubrics |
| Conversational Rec/Match | 6 nodes | Hit@5 | 0.30 | 0.38 | Final references helped track constraint propagation via backward eval |
| User Study | 6 developers | Qualitative feedback | N/A | Approved | Key value: graph localization, rationales, before/after revisions |

### Ablation Study
The automated iteration stress test used 11 workflows independently generated by LLMs based on documentation; all 11 could run end-to-end. The quantitative table focuses on 5 workflows where initial prompts were intentionally weak, leaving room for improvement, comparing the no-rewrite baseline with the best scores from three Auto Loop rounds.

| Workflow | No-rewrite baseline | Auto Loop best | Gain | Note |
|--------|---------------------|----------------|------|------|
| HTTP log triage | 0.307±0.029 | 0.648 | +0.341 | Automatic revisions brought significant improvement |
| Course scheduling | 0.186±0.001 | 0.800 | +0.614 | Largest gain observed |
| Incident ticket | 0.333±0.110 | 0.840 | +0.507 | Significant improvement in structure/constraints |
| Refuse/clarify | 0.208±0.027 | 0.390 | +0.182 | Moderate improvement |
| Word problem | 0.000±0.000 | 0.000 | 0.000 | Exact-match metrics failed to provide partial feedback |

### Key Findings
- In internal document review tasks, PROTEA increased accuracy from 64.3% to 83.9%, demonstrating that node-level evidence effectively supports manual prompt refinement.
- In recommendation workflows, Hit@5 increased from 0.30 to 0.38. While the gain was smaller, the task was closer to final recommendation quality, showing that backward references help in analyzing constraint propagation.
- In the auto-iteration of 5 minimal-prompt workflows, 4 exceeded the no-rewrite baseline, with 3 showing a gain > 0.3. However, the word problem task failed completely, exposing limitations of near-binary evaluation signals.
- In the user study, 6 experienced developers valued graph-level localization, per-node rationales, and editable before/after revisions, indicating that the paper's contribution leans toward "development loop design" rather than a standalone algorithm.

## Highlights & Insights
- **Practicality of Backward Node Evaluation**: It acknowledges that missing intermediate references is the norm and uses final answers and graph structure to generate "good enough, inspectable" local expectations rather than requiring perfect initial labels.
- **Workflow-Centric UI**: PROTEA integrates evaluation, traces, prompt versions, and re-run comparisons into a single interface, reducing the time-consuming context switching inherent in multi-agent debugging.
- **Human-Friendly Design**: The pass/warn/fail threshold design is intuitive for developers; it does not disguise a continuous judge score as absolute truth but instead flags areas "worth looking at."
- **Guided Automation**: The automatic mode maintains human-in-the-loop principles. Even with Auto Loop, the system emphasizes fixed suites, stable behavior, and regression comparison rather than unconstrained prompt searching.

## Limitations & Future Work
- PROTEA currently primarily supports fixed DAG workflows and local prompt revisions, not handling cyclic control flows, supervisor-based coordination, or long-term interactive agents.
- **Calibration of LLM-as-a-judge**: This remains a core risk. Different judges, rubrics, or randomness can affect node status; production deployment requires multi-judge agreement and manual auditing.
- **Reference Bias**: Backward-generated node references are only candidate expectations and may project biases from the final answer back to upstream nodes. Manual editing is supported, but the paper lacks large-scale calibration experiments.
- **Future Directions**: Support for architecture-level edits (e.g., adding nodes, reconnecting dependencies), automatic generation of partial-credit rubrics, and exporting prompt versions to CI regression suites.

## Related Work & Insights
- **vs OpenAI Evals / promptfoo**: These tools excel at end-to-end measurement and regression but typically do not localize failures to workflow nodes. PROTEA focuses on graph-level diagnosis and local repair.
- **vs LangSmith / Langfuse**: Observability platforms provide trace and prompt management; PROTEA goes further by integrating node evaluation, backward references, and revision suggestions into a closed loop.
- **vs DSPy / OPRO**: Automatic prompt optimization focuses more on search under an objective function; PROTEA focuses on how developers understand, audit, and compare changes in specific multi-node workflows.
- **Insight**: Evaluation for complex LLM applications should not just report final scores but should push diagnostic evidence down to the node level and ensure every prompt modification includes before/after regression records on the same suite.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Backward node evaluation and the graph-level prompt revision loop address real pain points in multi-agent debugging.
- **Experimental Thoroughness**: ⭐⭐⭐☆☆ Includes two production-adjacent cases, an automated stress test, and a user study, though some internal task details are not fully reproducible.
- **Writing Quality**: ⭐⭐⭐⭐☆ The system process is clear, and limitations are addressed honestly. Technical formulas serve the interface design without over-complication.
- **Value**: ⭐⭐⭐⭐⭐ Highly practical for teams maintaining multi-node LLM workflows, particularly as an offline regression and prompt review tool.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] AgenticEval: Toward Agentic and Self-Evolving Safety Evaluation of Large Language Models](agenticeval_toward_agentic_and_self-evolving_safety_evaluation_of_large_language.md)
- [\[ACL 2026\] Seeing the Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent Systems](seeing_the_whole_elephant_a_benchmark_for_failure_attribution_in_llm-based_multi.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2026\] CIA: Inferring the Communication Topology from LLM-based Multi-Agent Systems](cia_inferring_the_communication_topology_from_llm-based_multi-agent_systems.md)
- [\[ACL 2026\] Social Dynamics as Critical Vulnerabilities that Undermine Objective Decision-Making in LLM Collectives](social_dynamics_as_critical_vulnerabilities_that_undermine_objective_decision-ma.md)

</div>

<!-- RELATED:END -->
