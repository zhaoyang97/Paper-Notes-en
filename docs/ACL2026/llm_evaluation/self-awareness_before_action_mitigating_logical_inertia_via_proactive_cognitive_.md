---
title: >-
  [Paper Note] Self-Awareness before Action: Mitigating Logical Inertia via Proactive Cognitive Awareness
description: >-
  [ACL 2026][LLM Evaluation][self-aware reasoning] This paper proposes SABA, a reasoning framework that adopts a "perceive before act" paradigm, explicitly constructing and auditing knowledge states prior to any final decision. It employs Information Fusion (IF) to consolidate narratives into a verifiable baseline state, and Query-driven Structured Reasoning (QSR) to recursively identify and resolve missing premises, achieving state-of-the-art performance on both detective reasoning and general reasoning benchmarks.
tags:
  - ACL 2026
  - LLM Evaluation
  - self-aware reasoning
  - non-interactive narrative reasoning
  - structured state management
  - information fusion
  - logical inertia
date: 2026-05-08
content_hash: 1e4a11fb47fb6f51
---

# Self-Awareness before Action: Mitigating Logical Inertia via Proactive Cognitive Awareness

**Conference**: ACL 2026
**arXiv**: [2604.20413](https://arxiv.org/abs/2604.20413)
**Code**: None
**Area**: LLM Evaluation
**Keywords**: self-aware reasoning, non-interactive narrative reasoning, structured state management, information fusion, logical inertia

## TL;DR
This paper proposes SABA, a reasoning framework that adopts a "perceive before act" paradigm, explicitly constructing and auditing knowledge states prior to any final decision. It employs Information Fusion (IF) to consolidate narratives into a verifiable baseline state, and Query-driven Structured Reasoning (QSR) to recursively identify and resolve missing premises, achieving state-of-the-art performance on both detective reasoning and general reasoning benchmarks.

## Background & Motivation

**Background**: Large language models have demonstrated strong capabilities in multi-step reasoning and narrative comprehension. In interactive settings (e.g., social deduction games), agents can acquire new information and revise beliefs through dialogue. In non-interactive puzzle scenarios, however, the narrative is fixed, and models must reconstruct hidden truths solely from long-form text containing implicit clues, missing links, and distracting information.

**Limitations of Prior Work**: Existing reasoning paradigms exhibit systematic deficiencies in non-interactive long-narrative reasoning: (1) Chain-of-Thought tends to commit to an early hypothesis and elaborate upon it even when the initial premise is weak (*logical inertia*); (2) decomposition methods such as Least-to-Most introduce intermediate steps but lose global coherence when narratives are lengthy and evidence is scattered; (3) refinement methods such as Self-Refine revise outputs post-hoc but often rationalize the same early error rather than triggering comprehensive re-evaluation (*confirmation bias*).

**Key Challenge**: Once a model forms an early hypothesis under incomplete premises, the error propagates throughout the entire reasoning process, yielding unstable conclusions. The fundamental cause is the model's lack of awareness regarding whether its current knowledge or reasoning state is complete before acting. Existing methods follow an "answer then correct" paradigm rather than "verify completeness then answer."

**Goal**: To design a reasoning framework that shifts focus from "direct prediction" to "state assessment"—explicitly auditing whether the current understanding is complete and consistent before any decision is made.

**Key Insight**: Reasoning is reformulated as a progressive state-construction process rather than a single-step inference. The model should act as a systematic auditor: first examining its own knowledge state, identifying missing premises (obstacles), then incrementally filling them through hypothesis generation and state updates until a reasoning foundation sufficient to support the final conclusion is established.

**Core Idea**: Alternating between "structured state construction" and "obstacle-driven reasoning" via a recursive control loop—first integrating the narrative into a verifiable baseline, then converting missing or ambiguous premises into explicit obstacles and queries, resolving them recursively until logical closure is achieved.

## Method

### Overall Architecture
SABA consists of two stages. Stage 1, Information Fusion (IF), transforms raw narratives into a structured and validated baseline state. Stage 2, Query-driven Structured Reasoning (QSR), recursively identifies reasoning obstacles, decomposes them into queries, generates hypotheses, and updates the state until no obstacles remain or the maximum depth is reached. An adaptive gating mechanism bridges the two stages: if conflict and uncertainty indicators in the baseline state fall below predefined thresholds, the iterative loop is bypassed and the answer is synthesized directly.

### Key Designs

1. **Information Fusion (IF)**:

    - **Function**: Transforms dispersed, weak-signal raw narratives into dense, structured evidence representations.
    - **Mechanism**: Proceeds in two steps. First, **event alignment**: the narrative is decomposed into a core event skeleton $S = \{s_1, ..., s_m\}$ and a heterogeneous attribute set $A = \{a_1, ..., a_p\}$ (actions, object states, locations, evidence descriptions, etc.); an alignment mapping $\Phi_{\text{map}}: A \to 2^S$ then binds each attribute to one or more backbone events, rendering implicit associations explicitly retrievable. Second, **consistency checking**: a validation annotation $b_i = \psi_{\text{vfy}}(d_i, D_{\text{aligned}} \setminus d_i)$ is computed for each aligned unit, examining temporal, entity-state, and causal consistency, and flagging potential conflicts and uncertainties.
    - **Design Motivation**: Dispersed clues in long narratives cause the "lost-in-the-middle" effect and information forgetting. IF pre-associates scattered attributes to establish a validated cognitive baseline, ensuring critical evidence remains highly accessible throughout the reasoning trajectory. Consistency annotations do not discard information but mark uncertainty, making it explicitly tractable in subsequent reasoning.

2. **Query-driven Structured Reasoning (QSR)**:

    - **Function**: Progressively constructs the reasoning foundation by recursively identifying and resolving missing premises.
    - **Mechanism**: At each iteration, the model first **identifies obstacles** $\Omega_t = \mathcal{M}(p_{\text{aware}} | D_t, T)$, where each obstacle is represented as $\omega = (\tau(\omega), \text{dim}(\omega), \text{req}(\omega))$ (type, blocked dimension, missing requirement). It then **decomposes queries** $Q_{i,t} = \mathcal{M}(p_{\text{dec}} | \omega_i, D_t)$, converting abstract reasoning gaps into concrete information needs. Finally, it **generates hypotheses** $h = \mathcal{M}(p_{\text{hypo}} | q, D_t)$ as provisional logical bridges to fill the gaps. The state is updated as $D_{t+1} = D_t \cup Q_t \cup H_t$, and the recursion continues until $\Omega_t = \emptyset$ or the maximum depth is reached.
    - **Design Motivation**: The core insight is that "missing premises should be explicitly exposed and addressed rather than ignored or implicitly skipped." This transforms reasoning from "direct inference" to "progressive gap detection and filling," reducing logical leaps and unsupported assumptions.

3. **Adaptive Gating**:

    - **Function**: Prevents redundant computation on simple tasks.
    - **Mechanism**: The densities of logical conflicts $\mathbb{C}$ and uncertainties $\mathbb{D}$ in the baseline state are evaluated; if both fall below predefined thresholds $x$ and $y$, the QSR iterative loop is bypassed and the answer is synthesized directly.
    - **Design Motivation**: Not all tasks require recursive reasoning; the gating mechanism avoids wasting inference budget on straightforward tasks.

### Loss & Training
SABA is a pure prompting framework requiring no training. DeepSeek-V3 and Gemini-1.5-Flash are used as backbone models with decoding temperature set to 0.0 for reproducibility. Semantic similarity is computed using all-MiniLM-L6-v2.

## Key Experimental Results

### Main Results (DeepSeek-V3)

| Method | DP-Complex SA | DP-Complex CCR | StrategyQA | BBH | Inference Cost T |
|--------|--------------|----------------|------------|-----|-----------------|
| Direct | 40.7±0.9 | 58.7±1.0 | 82.0±0.4 | 78.7±0.5 | 1.0 |
| CoT | 45.4±1.1 | 61.9±1.2 | 87.6±0.5 | 86.0±0.6 | 2.5 |
| GoT | 69.8±1.6 | 77.3±1.7 | 91.7±0.8 | 90.7±0.9 | 35.7 |
| SABA | **79.3±1.2** | **83.3±0.6** | **94.4±0.4** | **93.2±0.5** | 9.2 |

### Ablation Study (DeepSeek-V3, DP-Complex)

| Configuration | SA | CCR | StrategyQA | Notes |
|---------------|----|-----|------------|-------|
| SABA (Full) | 79.3±1.2 | 83.3±0.6 | 94.4±0.4 | Full model |
| w/o IF | 69.8±1.1 | 70.7±0.9 | 82.2±0.6 | SA drops 12.0% without IF |
| Self-assess-only | 65.8±1.3 | 65.9±1.1 | 79.1±0.8 | Gap awareness only |
| w/o Awareness | 61.7±1.5 | 62.2±1.2 | 76.7±0.9 | SA drops 22.2% without obstacle identification |

### Key Findings
- On the most challenging DP-Complex split, SABA improves SA from 69.8 (GoT, strongest baseline) to 79.3 (+9.5 points) while incurring only 25.8% of GoT's inference cost (9.2 vs. 35.7).
- **Obstacle identification is the most critical component**: its removal causes the largest SA drop (22.2%), demonstrating that explicitly diagnosing missing premises is essential for preventing premature commitment.
- Information Fusion also contributes substantially (SA drops 12.0% and CCR drops 15.1% without it), confirming that pre-integrating dispersed clues into a grounded intermediate state benefits downstream reasoning.
- SABA demonstrates a clear efficiency advantage: its inference cost (9.2) is 23.3% lower than SC (12.0) and 74.2% lower than GoT (35.7), owing to adaptive gating and directed computation allocation.
- Cross-model generalization holds on Llama-3.1-70B, confirming that the framework is not backbone-specific.

## Highlights & Insights
- The **"perceive before act" paradigm shift** is highly insightful: reframing reasoning from "answer → correct" to "audit → construct → answer" addresses confirmation bias at its root. This principle is transferable to any scenario requiring inference under incomplete information.
- The **formal representation of obstacles** $\omega = (\tau, \text{dim}, \text{req})$ elevates missing premises to first-class citizens—not merely a vague sense that something is wrong, but a precise specification of what is missing, in which dimension, and what is required. This explicitness enables systematic downstream processing.
- The **full traceability** of reasoning trajectories (each step logs obstacles, queries, hypotheses, and state changes) makes the reasoning process auditable, which is highly valuable for explainable AI.
- Adaptive gating is a pragmatic engineering decision that avoids over-computation on tasks that do not require recursive reasoning.

## Limitations & Future Work
- SABA relies on the backbone model's self-assessment capability; obstacle detection quality may degrade for smaller models.
- The recursive process introduces substantial latency, which may hinder real-time applications.
- The IF module's structured input processing depends on the model's instruction-following ability; fully end-to-end clue extraction remains an open problem.
- Evaluation is limited to detective reasoning and general QA; performance on other reasoning types such as code generation and mathematics has not been validated.
- The maximum depth $t_{\max}$ and gating thresholds require manual configuration.

## Related Work & Insights
- **vs. CoT**: CoT produces a linear reasoning chain that tends to commit to early hypotheses and extend them. SABA explicitly verifies completeness before reasoning, preventing error propagation.
- **vs. Self-Refine / Reflexion**: These methods follow an "answer then revise" paradigm and are susceptible to confirmation bias. SABA shifts the revision target from candidate answers to the underlying knowledge state, enforcing an audit of completeness and consistency before any commitment.
- **vs. GoT (Graph-of-Thought)**: GoT externalizes reasoning trajectories but operates on unstructured text, lacking explicit representations of missing or inconsistent information. SABA formalizes reasoning as iterative structured state construction and verification.
- **Insight**: The state-first reasoning philosophy may offer value for RAG systems—constructing and validating the retrieved knowledge state before reasoning over it.

## Rating
- Novelty: ⭐⭐⭐⭐ The "perceive before act" concept is novel, though the specific techniques (IF + QSR) represent moderate incremental innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-benchmark, ablation, and cross-model validation are thorough, though the detective reasoning dataset contains only 31 examples.
- Writing Quality: ⭐⭐⭐⭐ Formal definitions are clear and visualizations are effective, though some notation is unnecessarily heavy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Modeling Multi-Dimensional Cognitive States in Large Language Models under Cognitive Crowding](modeling_multi-dimensional_cognitive_states_in_large_language_models_under_cogni.md)
- [\[ACL 2026\] Towards Self-Improving Error Diagnosis in Multi-Agent Systems](towards_self-improving_error_diagnosis_in_multi-agent_systems.md)
- [\[ACL 2026\] Mitigating Catastrophic Forgetting in Target Language Adaptation of LLMs via Source-Shielded Updates](mitigating_catastrophic_forgetting_in_target_language_adaptation_of_llms_via_sou.md)
- [\[ICLR 2026\] Mitigating Spurious Correlation via Distributionally Robust Learning with Hierarchical Ambiguity Sets](../../ICLR2026/llm_evaluation/mitigating_spurious_correlation_via_distributionally_robust_learning_with_hierar.md)
- [\[AAAI 2026\] Beyond Accuracy: A Cognitive Load Framework for Mapping the Capability Boundaries of Tool-use Agents](../../AAAI2026/llm_evaluation/beyond_accuracy_a_cognitive_load_framework_for_mapping_the_c.md)

</div>

<!-- RELATED:END -->
