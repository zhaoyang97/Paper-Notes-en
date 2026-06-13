---
title: >-
  [Paper Note] Self-Awareness before Action: Mitigating Logical Inertia via Proactive Cognitive Awareness
description: >-
  [ACL 2026][LLM Reasoning][Self-aware Reasoning] This paper proposes the SABA reasoning framework, which follows a "perceive-before-act" paradigm to explicitly construct and audit knowledge states before making final deci…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Self-aware Reasoning"
  - "Non-interactive Narrative Reasoning"
  - "Structured State Management"
  - "Information Fusion"
  - "Logical Inertia"
date: 2026-05-08
content_hash: bb14a8f47fb9c3e8
---

# Self-Awareness before Action: Mitigating Logical Inertia via Proactive Cognitive Awareness

**Conference**: ACL 2026  
**arXiv**: [2604.20413](https://arxiv.org/abs/2604.20413)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: Self-aware Reasoning, Non-interactive Narrative Reasoning, Structured State Management, Information Fusion, Logical Inertia

## TL;DR
This paper proposes the SABA reasoning framework, which follows a "perceive-before-act" paradigm to explicitly construct and audit knowledge states before making final decisions. By utilizing Information Fusion (IF) to integrate narratives into verifiable baseline states and Query-driven Structured Reasoning (QSR) to recursively identify and resolve missing premises, SABA achieves state-of-the-art performance on both detective reasoning and general reasoning benchmarks.

## Background & Motivation

**Background**: Large Language Models (LLMs) have demonstrated powerful capabilities in multi-step reasoning and narrative understanding. In interactive scenarios (e.g., social deduction games), agents can acquire new information and revise beliefs through dialogue. However, in non-interactive puzzle scenarios, the narrative is fixed, and the model must reconstruct hidden truths solely from long texts containing implicit clues, missing links, and distractor information.

**Limitations of Prior Work**: Existing reasoning paradigms exhibit systematic flaws in non-interactive long narrative reasoning: (1) Chain-of-Thought (CoT) tends to commit to an early hypothesis and expand upon it, even if the initial premise is weak (logical inertia); (2) decomposition methods (e.g., Least-to-Most) introduce intermediate steps but lose global coherence when narratives are long and evidence is scattered; (3) refinement methods (e.g., Self-Refine) revise after generating an answer, but often defend the same early error rather than triggering a comprehensive re-evaluation (confirmation bias).

**Key Challenge**: Once a model forms an early hypothesis under incomplete premises, the error propagates throughout the reasoning process, leading to unstable conclusions. The root cause is the model's lack of awareness regarding whether its own knowledge or reasoning state is complete before "acting" (providing an answer). Existing methods focus on "answer then correct" rather than "check completeness then answer."

**Goal**: Design a reasoning framework that shifts the focus from "direct prediction" to "state assessment"—explicitly auditing whether the current understanding is complete and consistent before any decision is made.

**Key Insight**: Redefine reasoning as a progressive state construction process rather than a single-step inference. The model should act like a system auditor, first inspecting its own knowledge state to identify missing premises (obstacles), then incrementally filling them through hypothesis generation and state updates until a reasoning foundation sufficient to support the final conclusion is built.

**Core Idea**: Alternating between "structured state construction" and "obstacle-driven reasoning" via a recursive control loop—first integrating the narrative into a verifiable baseline, then transforming missing or ambiguous premises into explicit obstacles and queries, and resolving them recursively until logical closure is achieved.

## Method

### Overall Architecture
SABA consists of two stages: Stage 1 is Information Fusion (IF), which transforms the raw narrative into a structured and verified baseline state; Stage 2 is Query-driven Structured Reasoning (QSR), which recursively identifies reasoning obstacles, decomposes them into queries, generates hypotheses, and updates the state until no obstacles remain or the maximum depth is reached. An adaptive gating mechanism exists between the two stages: if the metrics for conflicts and doubts in the baseline state are below a threshold, the iterative loop is skipped, and the answer is synthesized directly.

### Key Designs

1.  **Information Fusion (IF)**:
    - **Function**: Transforms scattered, weak-signal raw narratives into dense, structured evidence representations.
    - **Mechanism**: Conducted in two steps. First is **event alignment**: the narrative is decomposed into a core event skeleton $S = \{s_1, ..., s_m\}$ and a heterogeneous attribute set $A = \{a_1, ..., a_p\}$ (actions, object states, locations, evidence descriptions, etc.). Then, through an alignment mapping $\Phi_{\text{map}}: A \to 2^S$, each attribute is bound to one or more backbone events, making implicit associations explicitly retrievable. Second is **consistency checking**: for each alignment unit, a verification annotation $b_i = \psi_{\text{vfy}}(d_i, D_{\text{aligned}} \setminus d_i)$ is computed to check temporal, entity state, and causal consistency, marking potential conflicts and uncertainties.
    - **Design Motivation**: Scattered clues in long narratives lead to the "lost-in-the-middle" effect and information forgetting. IF establishes a verified cognitive baseline by pre-associating scattered attributes, ensuring key evidence remains highly available throughout the reasoning trajectory. Consistency annotations do not discard information but mark uncertainty, making it explicitly processable in subsequent reasoning.

2.  **Query-driven Structured Reasoning (QSR)**:
    - **Function**: Progressively builds reasoning support by recursively identifying and resolving missing premises.
    - **Mechanism**: In each iteration, it first performs **obstacle identification** $\Omega_t = \mathcal{M}(p_{\text{aware}} | D_t, T)$, where each obstacle is represented as $\omega = (\tau(\omega), \text{dim}(\omega), \text{req}(\omega))$ (type, blocked dimension, missing requirement). Then, it performs **query decomposition** $Q_{i,t} = \mathcal{M}(p_{\text{dec}} | \omega_i, D_t)$, converting abstract reasoning gaps into specific information needs. Finally, **hypothesis generation** $h = \mathcal{M}(p_{\text{hypo}} | q, D_t)$ acts as a temporary logical bridge to fill the gap. The state update $D_{t+1} = D_t \cup Q_t \cup H_t$ continues recursively until $\Omega_t = \emptyset$ or the maximum depth is reached.
    - **Design Motivation**: The core insight is that "missing premises should be explicitly exposed and handled, rather than ignored or implicitly skipped." Shifting reasoning from "direct inference" to "progressive gap detection and filling" reduces logical leaps and unsupported hypotheses.

3.  **Adaptive Gating**:
    - **Function**: Prevents redundant computation on simple tasks.
    - **Mechanism**: Evaluates the density of logical conflicts $\mathbb{C}$ and doubts $\mathbb{D}$ in the baseline state. If both are below pre-defined thresholds $x$ and $y$, the QSR iteration loop is skipped to synthesize the answer directly.
    - **Design Motivation**: Not all tasks require recursive reasoning; the gating mechanism avoids wasting reasoning budget on simple tasks.

### Loss & Training
SABA is a pure prompting framework and requires no training. DeepSeek-V3 and Gemini-1.5-Flash are used as backbone models, with the decoding temperature set to 0.0 for reproducibility. Semantic similarity is measured using all-MiniLM-L6-v2.

## Key Experimental Results

### Main Results (DeepSeek-V3)

| Method | DP-Complex SA | DP-Complex CCR | StrategyQA | BBH | Reasoning Cost T |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Direct | 40.7±0.9 | 58.7±1.0 | 82.0±0.4 | 78.7±0.5 | 1.0 |
| CoT | 45.4±1.1 | 61.9±1.2 | 87.6±0.5 | 86.0±0.6 | 2.5 |
| GoT | 69.8±1.6 | 77.3±1.7 | 91.7±0.8 | 90.7±0.9 | 35.7 |
| **SABA** | **79.3±1.2** | **83.3±0.6** | **94.4±0.4** | **93.2±0.5** | 9.2 |

### Ablation Study (DeepSeek-V3, DP-Complex)

| Configuration | SA | CCR | StrategyQA | Description |
| :--- | :--- | :--- | :--- | :--- |
| SABA (Full) | 79.3±1.2 | 83.3±0.6 | 94.4±0.4 | Full model |
| w/o IF | 69.8±1.1 | 70.7±0.9 | 82.2±0.6 | SA drops 12.0% without Information Fusion |
| Self-assess-only| 65.8±1.3 | 65.9±1.1 | 79.1±0.8 | Keeps only gap awareness |
| w/o Awareness | 61.7±1.5 | 62.2±1.2 | 76.7±0.9 | SA drops 22.2% without obstacle identification |

### Key Findings
- SABA improves SA from 69.8 (the strongest baseline, GoT) to 79.3 (+9.5 points) on the most difficult DP-Complex, while the reasoning cost is only 25.8% of GoT (9.2 vs 35.7).
- **Obstacle identification is the most critical component**: removing it leads to the largest drop in SA (22.2%), demonstrating that explicit diagnosis of missing premises is vital to preventing premature commitment.
- Information Fusion contributes significantly (SA drops 12.0%, CCR drops 15.1% without it), showing that pre-integrating scattered clues into a grounded intermediate state benefits subsequent reasoning.
- Reasoning efficiency is a clear advantage: SABA's reasoning cost (9.2) is 23.3% lower than SC (12.0) and 74.2% lower than GoT (35.7), thanks to adaptive gating and targeted computation allocation.
- Cross-model generalization: Stable performance is maintained on Llama-3.1-70B, proving the framework does not rely on a specific backbone.

## Highlights & Insights
- The **"perceive-before-act" paradigm shift** is highly insightful: shifting reasoning from "answer → correction" to "audit → construction → answer" fundamentally addresses confirmation bias. This concept can be transferred to any scenario requiring reasoning under incomplete information.
- The **formal representation of obstacles** $\omega = (\tau, \text{dim}, \text{req})$ makes missing premises first-class citizens—instead of just "feeling something is wrong," it precisely specifies "what is missing, in which dimension, and what is needed." This explicitness supports subsequent systematic processing.
- The **full traceability** of the reasoning trajectory (recording obstacles, queries, hypotheses, and state changes at each step) makes the reasoning process auditable, which is highly valuable for Explainable AI (XAI).
- Adaptive gating is a pragmatic engineering decision—avoiding over-computation for tasks where complex reasoning is unnecessary.

## Limitations & Future Work
- SABA relies on the self-assessment capability of the backbone model; obstacle detection quality may be limited for smaller models.
- The recursive process introduces high latency, which may affect real-time applications.
- The structured input processing in the IF module depends on the model's instruction-following capability; end-to-end clue extraction remains an open problem.
- Evaluation was limited to detective reasoning and general QA; it has not been verified in other reasoning types like code generation or mathematics.
- Fixed depth limits $t_{\max}$ and gating thresholds require manual tuning.

## Related Work & Insights
- **vs CoT**: CoT is a linear reasoning chain prone to early hypothesis commitment. SABA explicitly checks completeness before reasoning to avoid error propagation.
- **vs Self-Refine/Reflexion**: These are "answer-first" methods prone to confirmation bias. SABA shifts the refinement target from the candidate answer to the underlying knowledge state, forcing an audit of completeness and consistency before commitment.
- **vs GoT (Graph-of-Thought)**: GoT externalizes reasoning trajectories but operates on unstructured text, lacking explicit representation of missing or inconsistent information. SABA formalizes reasoning as iterative structured state construction and verification.
- **Insight**: The "state-first" reasoning concept may be valuable for RAG systems—first constructing and verifying the retrieved knowledge state before reasoning based on it.

## Rating
- Novelty: ⭐⭐⭐⭐ The "perceive-before-act" concept is novel, though specific techniques (IF + QSR) show moderate innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Thorough verification across multiple benchmarks, ablations, and models, though the detective reasoning dataset consists of only 31 cases.
- Writing Quality: ⭐⭐⭐⭐ Formal definitions are clear and visualizations are good, though some formula notations are somewhat heavy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] The Reasoning Trap — Logical Reasoning as a Mechanistic Pathway to Situational Awareness](../../ICLR2026/llm_reasoning/the_reasoning_trap_--_logical_reasoning_as_a_mechanistic_pathway_to_situational_.md)
- [\[ICML 2026\] Verifying Meta-Awareness via Predictive Rewards in Reasoning Models](../../ICML2026/llm_reasoning/verifying_meta-awareness_via_predictive_rewards_in_reasoning_models.md)
- [\[ACL 2026\] Logical Phase Transitions: Understanding Collapse in LLM Logical Reasoning](logical_phase_transitions_understanding_collapse_in_llm_logical_reasoning.md)
- [\[ACL 2026\] Discovering a Shared Logical Subspace: Steering LLM Logical Reasoning via Alignment of Natural-Language and Symbolic Views](discovering_a_shared_logical_subspace_steering_llm_logical_reasoning_via_alignme.md)
- [\[ICLR 2026\] Conflict-Aware Fusion: Resolving Logic Inertia in Large Language Models via Structured Cognitive Priors](../../ICLR2026/llm_reasoning/conflict-aware_fusion_resolving_logic_inertia_in_large_language_models_via_struc.md)

</div>

<!-- RELATED:END -->
