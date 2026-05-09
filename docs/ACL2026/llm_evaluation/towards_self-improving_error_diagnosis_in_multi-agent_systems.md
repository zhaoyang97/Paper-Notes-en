---
title: >-
  [Paper Note] Towards Self-Improving Error Diagnosis in Multi-Agent Systems
description: >-
  [ACL 2026][LLM Evaluation][To be supplemented] To be supplemented after a thorough reading of the paper.
tags:
  - ACL 2026
  - LLM Evaluation
  - To be supplemented
date: 2026-05-08
content_hash: e7ac778c7d94295e
---

# Towards Self-Improving Error Diagnosis in Multi-Agent Systems

**Conference**: ACL 2026
**arXiv**: [2604.17658](https://arxiv.org/abs/2604.17658)
**Code**: None
**Area**: LLM Evaluation
**Keywords**: multi-agent fault attribution, error localization, self-improving diagnosis, verified memory, backward tracing

## TL;DR

This paper proposes ErrorProbe, a framework that achieves self-improving semantic fault attribution in multi-agent systems through MAST taxonomy-driven structured decomposition, symptom-driven backward tracing, and a verified memory mechanism. The approach substantially outperforms baselines, particularly in step-level error localization.

## Background & Motivation

**State of the Field**: LLM-based multi-agent systems (MAS) have demonstrated strong capabilities in software engineering, web navigation, and scientific reasoning, yet their debugging challenges are increasingly prominent. When multiple agents (architect, engineer, tester, etc.) collaborate on a task and the system fails, one must answer: "Which agent caused the error, and at which step did it originate?"

**Limitations of Prior Work**: Existing diagnostic approaches suffer from three categories of deficiencies: (1) taxonomy-based manual annotation methods (e.g., MAST) require substantial expert effort and do not scale; (2) training-data-driven specialized trackers rely on expensive data generation pipelines and require continuous retraining; (3) the LLM-as-a-Judge paradigm performs poorly at step-level localization over long contexts, especially in scenarios where errors manifest with delay.

**Root Cause**: Fault attribution in MAS faces multiple compounding challenges — interaction trajectories are extremely long (tens to hundreds of turns), errors manifest with delay (early-stage errors surface only later), causal dependency chains across agents are complex, and failure modes are highly diverse. These factors render single-pass LLM judgment ineffective at penetrating long contexts to identify root causes.

**Paper Goals**: Design a multi-agent fault attribution framework that requires no manual annotation and is capable of self-improvement, accurately identifying the responsible agent and the originating error step.

**Starting Point**: The approach simulates the debugging process of a human expert — decomposing the problem into specialized roles (hypothesis generation, verification execution, arbitration decision), pruning irrelevant context via backward tracing, and enabling cross-domain pattern reuse through a verified memory repository.

**Core Idea**: The MAST taxonomy is operationalized as lightweight detectors that provide local anomaly signals. These are combined with symptom-driven backward tracing for context compression. A "Strategist–Investigator–Arbitrator" trio then validates hypotheses through tool-assisted execution, and self-improvement is achieved by updating the memory repository through a verification-gated mechanism.

## Method

### Overall Architecture

ErrorProbe is a three-stage pipeline: it takes as input a failed multi-agent interaction trajectory and a fault symptom description, and outputs the responsible agent, the originating error step, and the fault type. The pipeline first applies the MAST taxonomy to detect local anomaly labels, then performs backward tracing from the symptom to prune the context, and finally has three specialized agents collaboratively diagnose the fault and update the verified memory.

### Key Designs

1. **MAST-Guided Structured Decomposition**:

    - Function: Transforms raw interaction trajectories into structured representations and detects local anomaly signals.
    - Mechanism: The trajectory is parsed to extract each step's agent identity, role, and action type; taxonomy-conditioned prompts then detect step-level deviations (e.g., "tool output ignored," "reasoning–action mismatch"). These weak signals serve as heuristic priors, narrowing the search space from $L$ steps to a small set of candidate regions.
    - Design Motivation: Raw trajectories are noisy and unstructured; direct analysis is prone to losing focus. The 14 error patterns in MAST (specification issues, alignment failures, verification defects) provide semantic anchors.

2. **Symptom-Driven Backward Tracing**:

    - Function: Reconstructs the causal chain backward from the fault symptom, compressing irrelevant context.
    - Mechanism: A dependency graph $G=(V,E)$ is constructed over messages; breadth-first search from symptom node $v_L$ determines the effective receptive field of the error and masks unrelated parallel branches, compressing the original long trajectory $x$ into a causal subset $x' \subset x$.
    - Design Motivation: The root cause (e.g., an erroneous parameter at step 5) and the symptom (e.g., a crash at step 50) may be separated by dozens of steps; processing the full history directly leads to the "lost-in-the-middle" phenomenon.

3. **Verified Memory and Three-Agent Diagnostic Team**:

    - Function: Executes diagnosis through a "Strategist–Investigator–Arbitrator" team and maintains a repository of verified error patterns.
    - Mechanism: The Strategist retrieves historical patterns from the memory repository and generates a hypothesis set; the Investigator must provide executable evidence for each hypothesis via tools (CodeExec sandbox, LogicProbe conditional verification); the Arbitrator aggregates evidence to render a final judgment and decides whether to commit the pattern to memory. Memory updates must satisfy a strict verification gate: $\text{Verify}(E_t) \land c_t > \tau$, preventing hallucination-induced corruption.
    - Design Motivation: Pure LLM-based judgment is prone to spurious attribution hallucinations; tool-executed verification provides objective evidence, and the verification gate prevents memory degradation under distributional shift.

### Loss & Training

ErrorProbe is a training-free inference-time framework. By processing failed tasks in a streaming fashion, the memory state is selectively updated after each diagnosis based on verification outcomes: $\mathcal{M}_i \leftarrow \text{Update}(\mathcal{M}_{i-1}, x_i, \hat{y}_i, \text{Verify}(\hat{y}_i))$, enabling self-improvement. Memory retrieval uses a combination of structural matching and quality-weighted RFI-Δ scoring; under cold-start conditions, the system degrades gracefully to first-principles reasoning.

## Key Experimental Results

### Main Results

| Benchmark | Method | Agent Acc. | Step Acc. |
|-----------|--------|------------|-----------|
| TracerTraj | LLM-as-a-Judge (Claude) | 67.7% | 8.7% |
| TracerTraj | ErrorProbe+Memory (Claude) | 73.2% | 39.4% |
| Who&When-Algo | LLM-as-a-Judge (Claude) | 55.6% | 41.3% |
| Who&When-Algo | ErrorProbe+Memory (Claude) | 60.3% | 59.5% |
| Avg. (3 benchmarks) | ErrorProbe+Memory (Claude) | 59.6% | 42.7% |
| Avg. (3 benchmarks) | LLM-as-a-Judge (Claude) | 57.0% | 21.3% |

### Ablation Study

| Configuration | Agent Avg. | Step Avg. | Notes |
|---------------|-----------|----------|-------|
| LLM-as-a-Judge | 57.0% | 21.3% | Single-pass baseline |
| Agent-as-a-Judge (baseline) | 46.4% | 24.7% | Tool-augmented but unstructured |
| ErrorProbe (w/o memory) | 56.3% | 41.9% | With decomposition + tracing |
| ErrorProbe (w/ memory) | 59.6% | 42.7% | Full framework |

### Key Findings

- Step-level localization is the most significant contribution: ErrorProbe raises Claude's Step accuracy from 21.3% to 42.7%, more than doubling the baseline.
- The memory module provides greater benefit to weaker models: GPT-OSS-120B improves from 25.8% to 31.1%; Qwen3-32B improves from 29.2% to 34.9%.
- Cross-domain transfer is effective: patterns learned from KodCode improve diagnosis on TracerTraj, and the verification gate successfully filters domain-specific noise.
- In-domain memory gains are largest on GSM8K (Step +35%), owing to the high repetitiveness of error patterns in that domain.

## Highlights & Insights

- **Elegant verification gate design**: Only diagnostic patterns confirmed through tool execution are written to memory, avoiding the memory corruption under distributional shift that would afflict naive caching. This design principle is transferable to other LLM agent systems requiring accumulated experience.
- **Backward tracing addresses "lost-in-the-middle"**: Dependency-graph-based pruning compresses long trajectories into causal subsets, a method applicable to any scenario requiring causal localization in long contexts.
- **Three-agent team simulates human debugging workflows**: The division of labor among hypothesis generation, evidence collection, and arbitration allows each stage to be optimized independently.

## Limitations & Future Work

- The framework depends on explicit failure signals and cannot detect "silent failures" (outputs that are technically correct but semantically wrong).
- The multi-agent diagnostic team incurs non-trivial inference overhead, making it unsuitable for ultra-low-latency scenarios.
- Validation is limited to three model families, leaving broader architectural coverage unexplored.
- Future work could incorporate test-time oracle feedback mechanisms to surface latent errors.

## Related Work & Insights

- **vs. LLM-as-a-Judge**: LLM-as-a-Judge performs poorly on step-level localization (below 10% on TracerTraj); ErrorProbe addresses causal localization in long contexts through structured decomposition and backward tracing.
- **vs. TracerTraj training-based trackers**: Training-based methods depend on expensive counterfactual replay data and require continuous retraining. ErrorProbe requires no training and achieves progressive improvement through verified memory.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of verified memory and backward tracing is highly original, though the core idea of multi-agent collaborative diagnosis is not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three benchmarks, three model families, rich ablations, and memory scaling analysis constitute a thorough evaluation.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and method description is detailed, though some sections are slightly verbose.
**Code**: To be confirmed
**Area**: llm_evaluation
**Keywords**: To be supplemented

## TL;DR
To be supplemented after a thorough reading of the paper.

## Background & Motivation
To be supplemented after a thorough reading of the paper.

## Method
To be supplemented after a thorough reading of the paper.

## Key Experimental Results
To be supplemented after a thorough reading of the paper.

## Highlights & Insights
To be supplemented after a thorough reading of the paper.

## Limitations & Future Work
To be supplemented after a thorough reading of the paper.

## Related Work & Insights
To be supplemented after a thorough reading of the paper.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] MultiFileTest: A Multi-File-Level LLM Unit Test Generation Benchmark and Impact of Error Fixing Mechanisms](multifiletest_a_multi-file-level_llm_unit_test_generation_benchmark_and_impact_o.md)
- [\[ICLR 2026\] Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis](../../ICLR2026/llm_evaluation/talk_evaluate_diagnose_user-aware_agent_evaluation_with_automated_error_analysis.md)
- [\[ICLR 2026\] Which LLM Multi-Agent Protocol to Choose?](../../ICLR2026/llm_evaluation/which_llm_multi-agent_protocol_to_choose.md)
- [\[ICLR 2026\] UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking](../../ICLR2026/llm_evaluation/uis-digger_towards_comprehensive_research_agent_systems_for_real-world_unindexed.md)
- [\[AAAI 2026\] MAPS: Multi-Agent Personality Shaping for Collaborative Reasoning](../../AAAI2026/llm_evaluation/maps_multi-agent_personality_shaping_for_collaborative_reaso.md)

<!-- RELATED:END -->
