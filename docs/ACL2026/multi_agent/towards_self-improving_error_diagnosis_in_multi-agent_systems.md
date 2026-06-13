---
title: >-
  [Paper Note] Towards Self-Improving Error Diagnosis in Multi-Agent Systems
description: >-
  [ACL 2026][Multi-Agent][Multi-agent Fault Attribution] Ours proposes the ErrorProbe framework, which implements self-improving semantic fault attribution in multi-agent systems through MAST taxonomy-driven structured dec…
tags:
  - "ACL 2026"
  - "Multi-Agent"
  - "Multi-agent Fault Attribution"
  - "Error Localization"
  - "Self-Improving Diagnosis"
  - "Verifiable Memory"
  - "Backward Tracing"
date: 2026-05-08
content_hash: 81fc9aa0c1911069
---

# Towards Self-Improving Error Diagnosis in Multi-Agent Systems

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.17658](https://arxiv.org/abs/2604.17658)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: Multi-agent Fault Attribution, Error Localization, Self-Improving Diagnosis, Verifiable Memory, Backward Tracing

## TL;DR

Ours proposes the ErrorProbe framework, which implements self-improving semantic fault attribution in multi-agent systems through MAST taxonomy-driven structured decomposition, symptom-driven backward tracing, and a verifiable memory mechanism, significantly outperforming baselines particularly in step-level error localization.

## Background & Motivation

**Background**: LLM-based Multi-Agent Systems (MAS) have demonstrated strong capabilities in software engineering, web navigation, and scientific reasoning. However, debugging remains a prominent issue. When a system collaborative task involving multiple roles (architect, engineer, tester, etc.) fails, it is critical to determine "which agent caused the error?" and "at which step did the error originate?"

**Limitations of Prior Work**: Existing diagnosis methods suffer from three types of defects: (1) Taxonomy-based manual annotation methods (e.g., MAST) require extensive expert effort and are difficult to scale; (2) Training-based specialized trackers rely on expensive data generation pipelines and requiring continuous retraining; (3) The LLM-as-a-Judge paradigm performs poorly in step-level localization within long contexts, especially in scenarios with delayed error manifestation.

**Key Challenge**: Fault attribution in MAS faces multiple challenges—extremely long interaction trajectories (dozens to hundreds of rounds), delayed error manifestation (early errors only surfacing in later stages), complex causal dependencies between agents, and diverse failure patterns. These factors prevent a single LLM inference from effectively penetrating long contexts to locate root causes.

**Goal**: To design a self-improving multi-agent fault attribution framework that requires no manual annotation and can precisely identify the responsible agent and the error origin step.

**Key Insight**: Simulate the debugging process of human experts—first decompose the problem into multiple specialized roles (hypothesis generation, verification execution, arbitration decision), prune irrelevant context through backward tracing, and utilize a verified memory bank to achieve cross-domain pattern reuse.

**Core Idea**: Operationalize the MAST taxonomy into lightweight detectors to provide local anomaly clues, combine this with symptom-driven backward tracing to compress context, and then employ a "Strategist-Investigator-Arbiter" team to verify hypotheses using tools. Finally, update the memory bank through a verification gate to achieve self-improvement.

## Method

### Overall Architecture

ErrorProbe is a three-stage pipeline: the input consists of failed multi-agent interaction trajectories and descriptions of fault symptoms; the output is the responsible agent, the error origin step, and the fault type. First, local anomaly labels are detected via MAST taxonomy; then, the context is pruned via backward tracing starting from the symptom; finally, three specialized agents collaborate to diagnose and update the verifiable memory.

### Key Designs

1.  **MAST-guided Structured Decomposition**:
    - **Function**: Transforms raw interaction trajectories into structured representations and detects local anomaly signals.
    - **Mechanism**: First parses trajectories to extract agent identities, roles, and action types for each step, then uses taxonomy-conditioned prompts to detect step-level deviations (e.g., "tool output ignored", "reasoning-action mismatch"). These weak signals serve as heuristic priors to narrow the search space from $L$ steps to a few candidate regions.
    - **Design Motivation**: Raw trajectories are noisy and unstructured, making direct analysis prone to confusion. The 14 error patterns of MAST (specification issues, alignment failures, verification defects) provide semantic anchors.

2.  **Symptom-driven Backward Tracing**:
    - **Function**: Reconstructs causal chains backward from the failure symptoms to compress irrelevant context.
    - **Mechanism**: Constructs a dependency graph $G=(V,E)$ between messages and performs breadth-first search from the symptom node $v_L$ to determine the effective receptive field of the error, masking irrelevant parallel branches. This compresses the original long trajectory $x$ into a causal subset $x' \subset x$.
    - **Design Motivation**: Root causes (e.g., incorrect parameters at step 5) and symptoms (e.g., crash at step 50) may be separated by dozens of steps; processing the entire history directly leads to the "lost in the middle" phenomenon.

3.  **Verifiable Memory and Three-Agent Diagnosis Team**:
    - **Function**: Executes diagnosis through a "Strategist-Investigator-Arbiter" team and maintains a memory bank of verified error patterns.
    - **Mechanism**: The Strategist retrieves historical patterns from the memory bank and generates a set of hypotheses; the Investigator must provide executable evidence for each hypothesis through tools (CodeExec sandbox, LogicProbe conditional verification); the Arbiter aggregates evidence to make a final judgment and decides whether to write the pattern into memory. Memory updates must satisfy a strict verification gate: $\text{Verify}(E_t) \land c_t > \tau$, preventing hallucination pollution.
    - **Design Motivation**: Pure LLM judgments are prone to fault attribution hallucinations; tool-based verification provides objective evidence, while the verification gate prevents memory corruption under distribution shifts.

### Loss & Training

ErrorProbe is a training-free inference-time framework. By processing failed tasks in a stream, it selectively updates the memory state $\mathcal{M}_i \leftarrow \text{Update}(\mathcal{M}_{i-1}, x_i, \hat{y}_i, \text{Verify}(\hat{y}_i))$ based on verification results after each diagnosis, achieving self-improvement. Memory retrieval uses a combination of structural matching and quality-weighted RFI-Δ scores, degrading to first-principles reasoning during cold starts.

## Key Experimental Results

### Main Results

| Benchmark | Method | Agent Accuracy | Step Accuracy |
|------|------|-------------|------------|
| TracerTraj | LLM-as-a-Judge (Claude) | 67.7% | 8.7% |
| TracerTraj | ErrorProbe+Memory (Claude) | 73.2% | 39.4% |
| Who&When-Algo | LLM-as-a-Judge (Claude) | 55.6% | 41.3% |
| Who&When-Algo | ErrorProbe+Memory (Claude) | 60.3% | 59.5% |
| Average | ErrorProbe+Memory (Claude) | 59.6% | 42.7% |
| Average | LLM-as-a-Judge (Claude) | 57.0% | 21.3% |

### Ablation Study

| Configuration | Agent Avg | Step Avg | Description |
|------|-----------|----------|------|
| LLM-as-a-Judge | 57.0% | 21.3% | Single-inference baseline |
| Agent-as-a-Judge (Baseline) | 46.4% | 24.7% | Tool-enhanced but unstructured |
| ErrorProbe (No Memory) | 56.3% | 41.9% | Decomposition + Tracing |
| ErrorProbe (With Memory) | 59.6% | 42.7% | Full framework |

### Key Findings

- Step-level localization is the biggest highlight: ErrorProbe increases the Step Accuracy of Claude from 21.3% to 42.7%, a more than twofold improvement.
- The memory module benefits weaker models more: GPT-OSS-120B improved from 25.8% to 31.1%, and Qwen3-32B from 29.2% to 34.9%.
- Cross-domain transfer is effective: Patterns learned from KodCode improved diagnosis on TracerTraj, with the verification gate successfully filtering out domain-specific noise.
- GSM8K showed the largest in-domain memory gain (Step +35%), due to the high repeatability of error patterns in that domain.

## Highlights & Insights

- **Sophisticated Verification Gate**: Only diagnostic patterns confirmed by tool execution are written to memory, avoiding memory corruption under distribution shifts—a concept transferable to other LLM agent systems requiring experience accumulation.
- **Backward Tracing Resolves "Lost in the Middle"**: Compressing long trajectories into causal subsets via dependency graph pruning is applicable to all scenarios requiring causal localization in long contexts.
- **Three-Agent Team Mimics Human Debugging**: The division of labor—hypothesis generation, evidence collection, and arbitration decision—allows for independent optimization of each stage.

## Limitations & Future Work

- Reliance on explicit failure signals; unable to detect "silent failures" (technically correct but semantically wrong outputs).
- High inference overhead of the multi-agent diagnosis team makes it unsuitable for ultra-low latency scenarios.
- Validated only on three model families; coverage of more architectures is needed.
- Future work could introduce test-time oracle feedback mechanisms to expose latent errors.

## Related Work & Insights

- **vs LLM-as-a-Judge**: LLM-as-a-Judge is severely deficient in step localization (<10% on TracerTraj). ErrorProbe solves the causal localization puzzle in long contexts through structured decomposition and backward tracing.
- **vs TracerTraj Trained Trackers**: Training-based methods rely on expensive counterfactual replay data and require continuous retraining. ErrorProbe is training-free and achieves progressive improvement through verifiable memory.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of verifiable memory and backward tracing is innovative, though the core idea (Multi-agent collaboration) is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three benchmarks, three models, intensive ablations, and memory scaling analysis provide comprehensive evidence.
- Writing Quality: ⭐⭐⭐⭐ Problem definitions are clear and methods are detailed, though some sections are slightly verbose.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Stochastic Self-Organization in Multi-Agent Systems](../../ICLR2026/multi_agent/stochastic_self-organization_in_multi-agent_systems.md)
- [\[ACL 2026\] AgenticEval: Toward Agentic and Self-Evolving Safety Evaluation of Large Language Models](agenticeval_toward_agentic_and_self-evolving_safety_evaluation_of_large_language.md)
- [\[AAAI 2026\] Conversational Learning Diagnosis via Reasoning Multi-Turn Interactive Learning](../../AAAI2026/multi_agent/conversational_learning_diagnosis_via_reasoning_multi-turn_interactive_learning.md)
- [\[AAAI 2026\] LungNoduleAgent: A Collaborative Multi-Agent System for Precision Diagnosis of Lung Nodules](../../AAAI2026/multi_agent/lungnoduleagent_a_collaborative_multi-agent_system_for_precision_diagnosis_of_lu.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)

</div>

<!-- RELATED:END -->
