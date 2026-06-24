---
title: >-
  [Paper Note] GuardAgent: Safeguard LLM Agents via Knowledge-Enabled Reasoning
description: >-
  [ICML2025][LLM Agent][Agent Security] GuardAgent is the first "Agent-safeguarding-Agent" framework that dynamically converts safety rules into executable guardrail code to verify if the actions of a target Agent violate safety policies. It achieves guardrail accuracies of over 98% and 83% on new benchmarks for medical access control and web safety control, respectively.
tags:
  - "ICML2025"
  - "LLM Agent"
  - "Agent Security"
  - "Guardrail Agent"
  - "Code Generation & Execution"
  - "Access Control"
  - "Security Policy"
date: 2026-05-08
content_hash: 10ccd25a07d38307
---

# GuardAgent: Safeguard LLM Agents via Knowledge-Enabled Reasoning

**Conference**: ICML2025  
**arXiv**: [2406.09187](https://arxiv.org/abs/2406.09187)  
**Code**: [GuardAgent Project Page](https://guardagent.github.io/)  
**Area**: LLM Agent  
**Keywords**: Agent Security, Guardrail Agent, Code Generation & Execution, Access Control, Security Policy

## TL;DR
GuardAgent is the first "Agent-safeguarding-Agent" framework that dynamically converts safety rules into executable guardrail code to verify if the actions of a target Agent violate safety policies. It achieves guardrail accuracies of over 98% and 83% on new benchmarks for medical access control and web safety control, respectively.

## Background & Motivation

### Limitations of Prior LLM Guardrails
Traditional LLM guardrails (such as NVIDIA NeMo Guardrails, Llama Guard, etc.) primarily focus on text toxicity detection (categories such as violence, pornography, and hate speech) and rely on classifiers for output content moderation.

However, the outputs of LLM Agents extend beyond mere text:
- Web Agents may click buttons and fill out forms.
- Medical Agents may query databases and modify patient records.
- Autonomous driving Agents may generate trajectory commands.

These output modalities go far beyond the scope of pure text, and their safety requirements are highly domain-specific. Traditional text-based guardrails are entirely incapable of handling requirements such as "prohibiting unauthorized personnel from viewing laboratory data of specific patients."

### Key Challenge
Existing Agent safety controls are either hard-coded inside the target Agent (lacking transferability) or cover only general text risks (lacking fine granularity). Hence, there is a need for an Agent-level guardrail solution that is non-intrusive, transferable, and programmable.

## Method

### Overall Architecture
GuardAgent is itself an LLM Agent designed to monitor another target Agent:
1. Receives safety rules (as plain text descriptions).
2. Receives inputs, outputs, and logs from the target Agent.
3. Generates an action plan $\rightarrow$ generates guardrail code $\rightarrow$ executes the code $\rightarrow$ returns pass/deny.

### Key Designs

#### Key Design 1: Task Plan Generation
The LLM analyzes the safety rules and the action logs of the target Agent to generate a structured inspection plan. This step is driven by in-context learning (ICL), retrieving relevant historical examples from the memory module to serve as context.

#### Key Design 2: Guardrail Code Generation and Execution
The plan is translated into executable Python code, which can invoke predefined tool functions (such as database queries, field comparisons, and permission verification). Code execution is deterministic, thereby avoiding the unreliability of pure natural language (NL) reasoning.

#### Key Design 3: Memory Module
Stores previously successfully handled safety rules and their corresponding plans/code to provide few-shot examples for new rules. It guides the current task by retrieving similar historical cases.

### Three Major Advantages
- Non-intrusive: Deployed in parallel with the target Agent without modifying its internal logic.
- Code-driven: More reliable and auditable than pure natural language (NL) judgment.
- Training-free: Based on in-context learning (ICL), utilizing off-the-shelf LLMs directly.

## Key Experimental Results

### Two Newly Proposed Benchmarks

| Benchmark | Scenario | Safety Requirement Type | Scale |
|------|------|------------|------|
| EICU-AC | Medical Agent (EHRAgent) | Access Control | Multi-role permission combinations |
| Mind2Web-SC | Web Agent (SeeAct) | Security Policy | Multi-website operation types |

### Main Results: Guardrail Accuracy

| Method | EICU-AC Accuracy | Mind2Web-SC Accuracy |
|------|---------------|-------------------|
| LLM-based guardrail (pure NL) | ~85% | ~65% |
| GuardAgent (GPT-4) | **98%+** | **83%+** |
| GuardAgent (Claude) | 97%+ | 82%+ |
| GuardAgent (GPT-3.5) | 95%+ | 78%+ |

### Key Findings
1. "Agent-guarding-Agent" significantly outperforms "Model-guarding-Agent" (pure NL reasoning approaches).
2. The determinism of code execution is a key factor in accuracy improvement.
3. Various backbone LLMs can successfully drive GuardAgent, demonstrating the universality of the framework.
4. GuardAgent does not affect the task performance of the target Agent (performing filtering only).
5. The retrieval quality of the memory module directly impacts the processing effectiveness of complex rules.

## Highlights & Insights

1. Pioneeringly proposes the "Agent-guarding-Agent" paradigm, filling a critical gap in the LLM Agent safety domain.
2. The elegant design of code generation + execution effectively converts ambiguous safety rules into deterministic inspection logic.
3. The two new benchmarks demonstrate strong practical significance; both medical access control and web safety policies represent high-frequency, real-world requirements.
4. Excellent extensibility: introducing new safety rules only requires updating the toolbox and memory, with no retraining needed.
5. Industry implication: any scenario deploying an Agent requires a matching guardrail Agent.

## Limitations & Future Work

1. Code generation relies heavily on the capabilities of the underlying LLM, and complex rules may result in incorrect code generation.
2. The accuracy on Mind2Web-SC is lower than on EICU-AC, indicating that the high diversity of web scenarios is more challenging to cover.
3. The toolbox currently requires manual pre-definition; automated tool discovery represents a potential future direction.
4. The handling of conflicting safety rules has not been thoroughly discussed.
5. Attackers may bypass guardrails keying on indirect methods (adversarial robustness requires further evaluation).

## Related Work & Insights

- The fundamental difference from NeMo Guardrails and Llama Guard lies in the fact that GuardAgent processes structured actions rather than pure text.
- In contrast to agent-internal safety mechanisms, GuardAgent is external and generalized.
- Inspiration for future research: The guardrail Agent can be combined with formal verification, or hierarchical guardrail systems can be explored.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (5.0/5) — The first Agent-guarding-Agent framework
- Experimental Thoroughness: ⭐⭐⭐⭐☆ (4.0/5) — Two new benchmarks, but scenarios could be more diverse
- Writing Quality: ⭐⭐⭐⭐☆ (4.0/5)
- Value: ⭐⭐⭐⭐⭐ (5.0/5) — Agent safety is a critical demand

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Process Reward Agents for Steering Knowledge-Intensive Reasoning](../../ICML2026/llm_agent/process_reward_agents_for_steering_knowledge-intensive_reasoning.md)
- [\[NeurIPS 2025\] LLM Agents for Knowledge Discovery in Atomic Layer Processing](../../NeurIPS2025/llm_agent/llm_agents_for_knowledge_discovery_in_atomic_layer_processing.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Multimodal Knowledge-Induced Cognitive Reasoning for Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_multimodal_knowledge-induced_cognitive_reasoning_for_web.md)
- [\[ICML 2025\] KBQA-o1: Agentic Knowledge Base Question Answering with Monte Carlo Tree Search](kbqa-o1_agentic_knowledge_base_question_answering_with_monte_carlo_tree_search.md)
- [\[ACL 2025\] Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools](../../ACL2025/llm_agent/agentic_reasoning_tools.md)

</div>

<!-- RELATED:END -->
