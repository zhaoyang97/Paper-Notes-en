---
title: >-
  [Paper Note] Accurate Legal Reasoning at Scale: Neuro-Symbolic Offloading and Structural Auditability for Robust Legal Adjudication
description: >-
  [ACL 2026][LLM Reasoning][Neuro-symbolic] This paper proposes the Amortized Intelligence paradigm: treating the LLM as a "one-time compiler" to compile legal contracts into a deterministic Directed Acyclic Graph (DAG) in…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Neuro-symbolic"
  - "Legal Reasoning"
  - "DACL"
  - "Amortized Intelligence"
  - "Auditability"
date: 2026-05-08
content_hash: 47b86e1f1e06569f
---

# Accurate Legal Reasoning at Scale: Neuro-Symbolic Offloading and Structural Auditability for Robust Legal Adjudication

**Conference**: ACL 2026  
**arXiv**: [2605.02472](https://arxiv.org/abs/2605.02472)  
**Code**: None (Industrial deployment system)  
**Area**: LLM Reasoning / Neuro-Symbolic  
**Keywords**: Neuro-symbolic, Legal Reasoning, DACL, Amortized Intelligence, Auditability

## TL;DR
This paper proposes the Amortized Intelligence paradigm: treating the LLM as a "one-time compiler" to compile legal contracts into a deterministic Directed Acyclic Graph (DAG) intermediate representation called DACL. At runtime, a lightweight agent schedules a symbolic engine for execution, achieving 99.5% accuracy on 400 real-world contract events. Compared to large reasoning models (LRMs) like GPT-5.2, Claude, and Gemini, the accuracy on complex contracts jumps from 22-46% to 98%, while token consumption is reduced by 9.9x.

## Background & Motivation
**Background**: Legal AI has evolved from judgment prediction and contract review to the automated execution of LLM-driven "computational legal clauses." These clauses are common in high-frequency, high-value business scenarios such as logistics billing, energy procurement, taxation, and insurance, potentially generating thousands of repeated executions per month.

**Limitations of Prior Work**: Existing LLM solutions have two fatal flaws in high-risk contract execution: **Reliability**: Chain-of-Thought (CoT) reasoning is often unfaithful to arithmetic, and identical inputs can yield different outputs, which is unacceptable in industry. **Economy**: Running heavy LLM inference for every execution leads to costs that scale linearly with event volume. Benchmark experiments show that top-tier LRMs like GPT-5.2, Claude 4.5 Sonnet, and Gemini 3 Pro collapse to 22-46% accuracy on structurally complex contracts (e.g., a Logistics Master Service Agreement with 76 decision states). These are not arithmetic errors but "structural failures"—calculating correctly but using the wrong variables.

**Key Challenge**: Legal execution requires "absolute deterministic output + auditable trails + affordable costs," whereas general LLMs provide "probabilistic output + unfaithful CoT + costs linearly tied to traffic." Essentially, "general reasoning engines" are being used in the wrong context—legal terms provide deterministic logic once signed and do not require re-interpretation for every execution.

**Goal**: To decouple legal automation into two stages: "Understanding" (high-difficulty but one-time) and "Execution" (high-frequency but simple). The LLM is responsible only for the former, leaving the latter to a deterministic symbolic engine.

**Key Insight**: The authors draw inspiration from compiler architecture—treating the LLM as a "front-end compiler" to translate contracts into an intermediate representation (IR) and using the symbolic execution engine as the "backend" runtime. This "amortizes" the inference cost over the initial compilation, making subsequent executions near-zero cost.

**Core Idea**: Use "DACL IR + Neuro-Symbolic Agent" to offload probabilistic reasoning from runtime to compile-time, simultaneously achieving auditability, determinism, and economy.

## Method

### Overall Architecture
The system consists of two pipelines: **Compilation Pipeline** (one-time)—an LLM agent segments and classifies natural language contracts to generate a DACL graph, which undergoes type checking and scenario testing before being reviewed by a lawyer and indexed. **Execution Pipeline** (per event)—user events $F_i$ and queries $Q$ enter a neuro-symbolic agent. The agent uses gpt-5-mini for semantic routing to identify relevant clause IDs and invokes the DACL symbolic engine to perform calculations, returning the result $v$ and audit trail $\tau$. Finally, the agent wraps the results into a natural language response. All business-critical logic remains internal to the symbolic engine and external to probabilistic reasoning.

### Key Designs
1.  **DACL IR and Four Clause Primitives**:
    - **Function**: Models contract logic using a strongly-typed Directed Acyclic Graph (DAG) to ensure referential transparency—the same input must produce the same output.
    - **Mechanism**: Variables are categorized into three types: External (runtime inputs like shipment weight), Const (contract constants with validity windows), and Derived (intermediate calculations). Logic is expressed via four recursive primitives: **Procedure** (sequential pipeline with conditional early exit), **Logical Clause** (first-order Boolean logic with short-circuit evaluation and enforced priority), **Range Clause** (maps continuous variables to discrete buckets, validating non-overlap and coverage during loading), and **Pricing Formula** (sandboxed arithmetic expressions whitelisting `ceil/floor/round/sqrt/exp/log` to prevent arbitrary code execution). Clause-level `validity_start_date/end_date` allows incremental recompilation of only affected clauses during contract amendments.
    - **Design Motivation**: General logic programming (e.g., Prolog) is oriented toward theorem proving and is not direct enough for the arithmetic, temporal, and range combinations found in commercial contracts. DACL elevates these patterns to first-class primitives, reducing the LLM's task to syntactic mapping rather than semantic reasoning.

2.  **Amortized Intelligence (Reason Once, Execute Many)**:
    - **Function**: Separates the distinct tasks of "understanding contracts" and "executing contracts."
    - **Mechanism**: Baseline solutions are $O(N)$ models—each event $e_i$ requires re-reading contract $C$ and facts $F_i$ for full LLM reasoning. The proposed $O(1)$ model performs the contract-to-DACL translation only once. Subsequent events use cheap gpt-5-mini calls for semantic routing and symbolic execution. For 400 events, token consumption dropped from 13.44M to 1.35M (a 9.9x reduction).
    - **Design Motivation**: Business scenarios naturally exhibit an asymmetry where contracts are signed once but events repeat. Moving the expensive work to the one-time phase is a classic software engineering trade-off applied here to legal AI.

3.  **Neuro-Symbolic Agent's Three-Stage Scheduling**:
    - **Function**: Enables a lightweight LLM agent to safely delegate natural language queries to the symbolic engine and package deterministic results into responses.
    - **Mechanism**: The agent uses gpt-5-mini with the OpenAI Agents SDK for ReAct-style scheduling, exposing only one tool: `evaluate_clauses_tool(K, F_i)`. The three stages are: (1) **Semantic Mapping** $K = \mathcal{M}_{\theta_{small}}(Q, F_i)$—constrained semantic parsing maps the query to clause IDs without executing logic; (2) **Symbolic Delegation** $(v, \tau) = \Phi_{DACL}(K, F_i)$—invoking the deterministic engine where $P(v|K, F_i) = 1$; (3) **Synthesized Response** $y = \mathcal{S}_{\theta_{small}}(Q, v, \tau)$—combining results and audit trails into natural language.
    - **Design Motivation**: Ensures business-critical calculations never pass through probabilistic LLM reasoning. LLMs handle selection and expression, while numerical and conditional logic are offloaded to symbols.

### Loss & Training
The study does not train new models. It uses zero-shot LRM baselines (GPT-5.2 with reasoning_effort=none/medium, Claude 4.5 Sonnet with Extended Thinking, and Gemini 3 Pro with thinking_level=high) and gpt-5-mini for agent orchestration. All baselines are constrained to output a strict schema: a `reasoning` field (for auditing) and a `result` field (for automated scoring).

## Key Experimental Results

### Main Results: Accuracy Across 400 Events in Four Contract Types (Selected)

| Contract | GPT-5.2 (none) | GPT-5.2 (med) | Claude Sonnet 4.5 | Gemini 3 Pro | DACL Agent |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Health-PPO | 74% | 91% | 73% | 69% | **100%** |
| Energy-Sup | 100% | 99% | 100% | 91% | **100%** |
| Logistics-MSA | 22% | 46% | 45% | 30% | **98%** |
| Muni-IFB | 36% | 95% | 93% | 96% | **100%** |
| **Overall** | 58.0% | 82.8% | 77.8% | 71.5% | **99.5%** |

### Ablation Study: Error Analysis and Computational Cost

| Dimension | GPT-5.2 Medium Baseline | DACL Agent | Gain |
| :--- | :--- | :--- | :--- |
| Total Tokens (400 events) | 13.44M | 1.35M | 9.9x ↓ |
| Avg. Latency (Logistics-MSA) | ~164s | 26.8s | 6.1x ↓ |
| Var. Dependency Error Rate | 71% | ~0 (only 2 orchestrator errors) | — |
| Arithmetic Hallucination | <1 case/model | 0 | — |

### Key Findings
- **"Reasoning Cliff"**: GPT-5.2 is 100% accurate on Energy-Sup (1 decision state) but collapses to 46% on Logistics-MSA (76 states). This reveals that while LLMs handle arithmetic depth (e.g., date lookups in Muni-IFB), they cannot manage state width (maintaining state across multi-branch decision trees).
- **Structural vs. Arithmetic Errors**: 71% of errors stem from Variable Dependency (VD). Arithmetic Hallucination (AH) accounted for <1% of errors. Models have mastered "arithmetic primitives" but lack "structural fidelity"—applying the right algorithm to the wrong variables.
- **Controllable DACL Errors**: The two errors in the DACL Agent for Logistics-MSA were semantic routing failures by the orchestrator (gpt-5-mini selecting the wrong clause). The symbolic engine itself is error-free by construction; these can be eliminated via stricter tool input schemas.
- **Production Validation**: The system has been live for 12 months, processing ~1,000 billing events per month across 150+ commercial contracts.
- **"Medium" Reasoning helps Depth, not Width**: Enabling medium reasoning effort for GPT-5.2 raised Muni-IFB (temporal logic) from 36% to 95%, but Logistics-MSA only rose from 22% to 46%. Long CoT aids single-path depth but fails at cross-branch state tracking.

## Highlights & Insights
- **Paradigm Shift (Compile-time vs. Runtime)**: The most profound insight is redefining legal AI as a "compiler" rather than a "runtime interpreter." This boundary shift is a highly transferable engineering concept.
- **Error Taxonomy**: By categorizing errors into VD, DH, and AH, the authors prove that LRM failures are about "state tracking," not "calculation." This suggests future LRM training should focus on long-horizon state tracking.
- **Minimalist Tool Design**: Restricting the neuro-symbolic agent to a single tool (`evaluate_clauses_tool`) limits the LLM's room for error—a counter-narrative to "more tools are better," leading to higher safety in high-risk scenarios.
- **Incremental Versioning**: The use of `validity_start_date/end_date` in DACL allows incremental compilation, solving the scalability problem of recompiling entire contracts for minor edits.

## Limitations & Future Work
- **Limitations**: (1) DACL currently lacks support for defeasible reasoning, open semantic standards (e.g., "reasonable care"), and higher-order logic; (2) Establishing the Gold Standard requires manual implementation, limiting evaluation scale; (3) Traffic is synthetic, which may not cover all long-tail production errors; (4) Tested only on English contracts.
- **Future Work**: Extending DACL to defeasible/probabilistic formalisms (e.g., ProbLog); developing automated compilation error detection to reduce human review; expanding to public law scenarios (tax, social security) to enhance "Access to Justice."

## Related Work & Insights
- **vs. Prolog-based ProSLM/SOLAR**: Unlike general logic programming, DACL uses primitives specifically designed for commercial contract arithmetic and temporal logic.
- **vs. LexGLUE/LegalBench**: While prior work evaluates language understanding, this paper focuses on deterministic output in high-risk execution scenarios.
- **vs. Program-of-Thoughts (PoT)**: While PoT uses LLMs to generate Python for one-off reasoning, this work formalizes output into a persistent, versioned, and type-safe DAG for industrial reuse.

## Rating
- Novelty: ⭐⭐⭐⭐ (Redirecting the LLM toward one-time compilation is a significant paradigm shift).
- Experimental Thoroughness: ⭐⭐⭐⭐ (Strong real-world data and error analysis, though lacks RAG-LRM baselines).
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear concepts like "Reasoning Cliff" and excellent architectural diagrams).
- Value: ⭐⭐⭐⭐⭐ (A rare ACL paper describing a deployed industrial system with 12 months of production data).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LegalDrill: Diagnosis-Driven Synthesis for Legal Reasoning in Small Language Models](legaldrill_diagnosis-driven_synthesis_for_legal_reasoning_in_small_language_mode.md)
- [\[ACL 2026\] LePREC: Reasoning as Classification over Structured Factors for Assessing Relevance of Legal Issues](leprec_reasoning_as_classification_over_structured_factors_for_assessing_relevan.md)
- [\[AAAI 2026\] NeSTR: A Neuro-Symbolic Abductive Framework for Temporal Reasoning in Large Language Models](../../AAAI2026/llm_reasoning/nestr_a_neuro-symbolic_abductive_framework_for_temporal_reasoning_in_large_langu.md)
- [\[ACL 2026\] Discovering a Shared Logical Subspace: Steering LLM Logical Reasoning via Alignment of Natural-Language and Symbolic Views](discovering_a_shared_logical_subspace_steering_llm_logical_reasoning_via_alignme.md)
- [\[AAAI 2026\] In-Token Rationality Optimization: Towards Accurate and Concise LLM Reasoning via Self-Feedback](../../AAAI2026/llm_reasoning/in-token_rationality_optimization_towards_accurate_and_concise_llm_reasoning_via.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2026\] LegalDrill: Diagnosis-Driven Synthesis for Legal Reasoning in Small Language Models](legaldrill_diagnosis-driven_synthesis_for_legal_reasoning_in_small_language_mode.md)
- [\[ACL 2026\] LePREC: Reasoning as Classification over Structured Factors for Assessing Relevance of Legal Issues](leprec_reasoning_as_classification_over_structured_factors_for_assessing_relevan.md)
- [\[ACL 2026\] Discovering a Shared Logical Subspace: Steering LLM Logical Reasoning via Alignment of Natural-Language and Symbolic Views](discovering_a_shared_logical_subspace_steering_llm_logical_reasoning_via_alignme.md)
- [\[ICLR 2026\] Are Reasoning LLMs Robust to Interventions on Their Chain-of-Thought?](../../ICLR2026/llm_reasoning/are_reasoning_llms_robust_to_interventions_on_their_chain-of-thought.md)
- [\[ICLR 2026\] Compositional Generalization from Learned Skills via CoT Training: A Theoretical and Structural Analysis for Reasoning](../../ICLR2026/llm_reasoning/compositional_generalization_from_learned_skills_via_cot_training_a_theoretical_.md)

</div>

<!-- RELATED:END -->
