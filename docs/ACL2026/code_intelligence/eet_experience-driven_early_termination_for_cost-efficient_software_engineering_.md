---
title: >-
  [Paper Note] EET: Experience-Driven Early Termination for Cost-Efficient Software Engineering Agents
description: >-
  [ACL 2026][Software Engineering Agent] This paper proposes EET, an experience-driven early termination method that identifies unproductive iterations during patch generation and patch selection phases, reducing the total cost of SE agents by 19%–55% (32% on average) with negligible performance degradation (at most 0.2%).
tags:
  - ACL 2026
  - Software Engineering Agent
  - Cost Optimization
  - Experience-Driven
  - Early Termination
  - SWE-bench
date: 2026-05-08
content_hash: 4d4bb648bba8f0ce
---

# EET: Experience-Driven Early Termination for Cost-Efficient Software Engineering Agents

**Conference**: ACL 2026
**arXiv**: [2601.05777](https://arxiv.org/abs/2601.05777)
**Code**: [GitHub](https://github.com/IanWalls/EET)
**Area**: Code Intelligence
**Keywords**: Software Engineering Agent, Cost Optimization, Experience-Driven, Early Termination, SWE-bench

## TL;DR

This paper proposes EET, an experience-driven early termination method that identifies unproductive iterations during patch generation and patch selection phases, reducing the total cost of SE agents by 19%–55% (32% on average) with negligible performance degradation (at most 0.2%).

## Background & Motivation

**State of the Field**: LLM-based software engineering (SE) agents have achieved remarkable progress in automated issue resolution, with systems such as Agentless, Mini-SWE-Agent, and Trae Agent demonstrating strong performance on SWE-bench.

**Limitations of Prior Work**: The high monetary cost of SE agents is a major barrier to practical deployment (53% of developers cite cost as a key obstacle). The "token snowball" effect causes costs to grow super-linearly as conversation histories accumulate; unproductive iterations on hard or unsolvable problems further amplify waste.

**Root Cause**: Existing cost reduction methods (e.g., turn-control) can lower costs but significantly degrade task performance (an average drop of 10.7%). Achieving substantial cost savings while preserving performance remains the central challenge.

**Paper Goals**: To propose a general-purpose early termination optimization method that integrates seamlessly into diverse SE agents and significantly reduces cost without sacrificing task performance.

**Starting Point**: Inspired by the intuition that experienced developers can directly locate solutions without extensive trial and error, the method uses structured historical experience to guide agents in skipping redundant iterations.

**Core Idea**: Distill historical issue-resolution experience into structured knowledge objects (task abstraction + trajectory summary + confidence assessment), and use retrieved relevant experience at both the patch generation and patch selection stages to determine whether early termination is warranted.

## Method

### Overall Architecture

EET consists of two major components: (1) **Experience Generation** — distilling structured experience objects from historical issue-resolution records and storing them in an experience repository; and (2) **Early Termination Mechanism** — leveraging retrieved relevant experiences to make early termination decisions at milestone checkpoints during patch generation and via confidence thresholds during patch selection.

### Key Designs

1. **Structured Experience Representation and Retrieval**:
    - Function: Compress raw execution trajectories into compact, reusable experience objects.
    - Mechanism: Each experience object contains `task_description` (issue abstraction), `execution_summary` (trajectory summary), `evaluation_result` (all passing), `confidence`, and `confidence_reason` (quality assessment). Only successfully resolved experiences are retained.
    - Design Motivation: Raw trajectories are noisy and token-expensive; excessive compression discards useful signals. Structured representation strikes a balance between information density and utility.

2. **Milestone-Based Early Termination during Patch Generation**:
    - Function: Identify the point within a single patch generation process at which further iteration is unnecessary.
    - Mechanism: Code modification and test execution are defined as milestones. After each milestone, a confidence score is evaluated; if it exceeds threshold $\tau^{gen}$, generation is terminated early.
    - Design Motivation: Signals of patch quality may emerge after code modification (structural alignment) or after test execution (dynamic feedback); the dual-milestone design covers both scenarios.

3. **Dual-Threshold Early Termination during Patch Selection**:
    - Function: Dynamically control the number of candidate patches that need to be generated.
    - Mechanism: After each patch is generated, its confidence is assessed using the patch content, execution trajectory, and historical experience. If confidence exceeds $\tau_{upper}^{sel}$, generation stops (the patch is sufficiently good); if it falls below $\tau_{lower}^{sel}$, generation also stops (the problem is unlikely to be solved).
    - Design Motivation: Avoids the inefficiency of a fixed patch count — simple problems do not require multiple patches, and generating more patches for hard problems yields diminishing returns.

### Loss & Training

EET is an inference-time optimization method and involves no training. Key hyperparameters include: TF-IDF similarity threshold $\tau_{sim}$, generation early termination threshold $\tau^{gen}$, and selection upper/lower thresholds $\tau_{upper}^{sel}$ / $\tau_{lower}^{sel}$, tuned on 100 held-out validation samples from SWE-bench. The experience repository is constructed from SWE-bench Lite (207 deduplicated problems).

## Key Experimental Results

### Main Results

| Agent + Backend | Resolve Rate Δ | API Calls | Input Tokens | Output Tokens | Total Cost Δ |
|----------------|---------------|-----------|--------------|---------------|--------------|
| Agentless + GPT-5-mini | +7.8% | -26.4% | -51.8% | -51.0% | **-55.1%** |
| Agentless + DeepSeek-V3.2 | +7.2% | -25.5% | -31.9% | -35.0% | -32.2% |
| Mini-SWE + GPT-5-mini | +1.0% | -7.9% | -13.7% | -3.7% | -19.4% |
| Mini-SWE + DeepSeek-V3.2 | +0.6% | -8.4% | -13.6% | -4.4% | -19.3% |
| Trae + GPT-5-mini | 0.0% | -29.9% | -30.4% | -28.0% | -28.2% |
| Trae + DeepSeek-V3.2 | -0.2% | -26.5% | -37.7% | -28.2% | -36.7% |
| **Average** | **+2.7%** | **-20.8%** | **-29.9%** | **-25.1%** | **-31.8%** |

### Ablation Study

| Variant (Trae + GPT-5-mini) | Resolve Rate Δ | Total Cost Δ |
|-----------------------------|---------------|--------------|
| Full EET | 0.0% | -28.2% |
| w/o experience injection | -10.4% | -58.9% |
| w/o early termination mechanism | +0.4% | +3.1% |

### Key Findings

- EET achieves early termination on an average of 11.3% of issues (ranging from 8.6% to 14.0%), with the most significant cost savings concentrated in this subset.
- The largest gains are observed for Agentless (resolve rate even improves by 7.2–7.8%), as experience guidance compensates for the limitations of its fixed pipeline.
- Compared to turn-control: turn-control reduces costs more aggressively (−41.4%) but causes a substantial drop in resolve rate (−10.7%).
- LLM confidence scores are well-calibrated: patches with confidence >90 have pass rates of 63.6%–92.6%, while those with confidence <40 achieve only 8.7%–13.8%.
- Cross-repository transfer experiments demonstrate that the captured experience reflects general debugging patterns rather than repository-specific cues.

## Highlights & Insights

- The method is highly general and can be plug-and-play integrated into SE agents of different paradigms (fixed pipeline / autonomous planning / generate-then-select).
- The design of the "experience" object is elegant: rather than naive RAG retrieval of raw trajectories, it distills information into structured knowledge with confidence assessments.
- The dual-threshold design covers both "good enough to stop" and "too hard to continue" scenarios, making it more principled than a single threshold.
- Ablation studies clearly reveal the complementary roles of experience injection and the early termination mechanism.

## Limitations & Future Work

- The method relies on historical data to build the experience repository, posing a cold-start problem for entirely novel domains.
- Evaluation is conducted solely on SWE-bench Verified; generalization to industrial settings remains to be validated.
- Early termination decisions depend on LLM confidence outputs, and calibration quality may vary substantially across models.
- The current scope is limited to SE agents, but the underlying design philosophy — experience-driven early termination — is domain-agnostic and can be extended to general multi-step reasoning agents.

## Related Work & Insights

- **Distinction from RAG-based agent memory** (e.g., MetaGPT, MemoryBank): EET's experience is specifically designed to serve cost optimization, rather than to improve task performance.
- Fan et al.'s "token snowball" analysis identifies the root cause of the cost problem; EET addresses it from the perspective of experience reuse.
- **Implication for agent system design**: Cost optimization should be treated as a first-class concern rather than a secondary consideration subordinate to performance.

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic application of experience-driven early termination to the SE agent cost problem; the perspective is novel and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 3 agents × 2 LLM backends, with baseline comparisons, ablation studies, and cross-repository transfer analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, precise method description, and comprehensive experimental design.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering](../../ICLR2026/code_intelligence/ambig-swe_interactive_agents_to_overcome_underspecificity_in_software_engineerin.md)
- [\[NeurIPS 2025\] SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents](../../NeurIPS2025/code_intelligence/swe-rebench_an_automated_pipeline_for_task_collection_and_decontaminated_evaluat.md)
- [\[ACL 2026\] CollabCoder: Plan-Code Co-Evolution via Collaborative Decision-Making for Efficient Code Generation](collabcoder_plan-code_co-evolution_via_collaborative_decision-making_for_efficie.md)
- [\[ACL 2026\] LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software](logiceval_a_systematic_framework_for_evaluating_automated_repair_techniques_for_.md)
- [\[AAAI 2026\] MoSE: Hierarchical Self-Distillation Enhances Early Layer Embeddings](../../AAAI2026/code_intelligence/mose_hierarchical_self-distillation_enhances_early_layer_embeddings.md)

<!-- RELATED:END -->
