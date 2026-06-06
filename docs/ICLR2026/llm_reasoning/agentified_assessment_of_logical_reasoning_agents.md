---
title: >-
  [Paper Note] Agentified Assessment of Logical Reasoning Agents
description: >-
  [ICLR 2026][LLM Reasoning][Logical Reasoning Evaluation] This paper proposes an agent-based evaluation framework (AAA) that encapsulates assessment logic as an assessor agent and interacts with the agent under test via a…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "Logical Reasoning Evaluation"
  - "Agent-to-Agent Assessment"
  - "First-Order Logic"
  - "Automatic Formalization"
  - "SMT Solving"
date: 2026-05-08
content_hash: e3311351b0da939d
---

# Agentified Assessment of Logical Reasoning Agents

**Conference**: ICLR 2026
**arXiv**: [2603.02788](https://arxiv.org/abs/2603.02788)  
**Code**: [HuggingFace Dataset](https://huggingface.co/datasets/yfxiao/folio-refined)  
**Area**: LLM Reasoning
**Keywords**: Logical Reasoning Evaluation, Agent-to-Agent Assessment, First-Order Logic, Automatic Formalization, SMT Solving

## TL;DR

This paper proposes an agent-based evaluation framework (AAA) that encapsulates assessment logic as an assessor agent and interacts with the agent under test via a standard A2A interface. On a FOLIO dataset systematically cleaned using the Vampire theorem prover, an auto-formalization agent (NL→Z3Py + SMT solving) achieves 86.70% accuracy, substantially outperforming the CoT baseline at 73.89%, with a particularly notable gain of 32.79 percentage points on contradiction detection (False class).

## Background & Motivation

- **Evaluation Pain Point 1 — Conflation of Failure Modes**: When evaluating reasoning agents, execution failures (timeouts / parsing errors / runtime exceptions) and reasoning errors are often conflated into a single accuracy figure, making it difficult to distinguish "the model cannot reason" from "the model's tools malfunctioned."
- **Evaluation Pain Point 2 — Linear Integration Cost**: Traditional evaluation harnesses tightly couple benchmark logic with agent implementations; adding each new benchmark requires re-integration, resulting in O(n) cost.
- **Data Quality Issues**: The FOLIO dataset contains potential label errors (3.8% in the training set, 1.5% in the validation set) as well as NL-FOL translation quality issues, making it inherently unreliable to assess reasoning capabilities on flawed data.
- **Lack of Standardized Interfaces**: Different agents have different input/output formats, execution environments, and error-handling mechanisms, with no unified plug-and-play interface available.
- **Value of Formal Verification**: First-order logic reasoning is an important capability for LLMs, yet CoT methods cannot guarantee logical validity; formal verification via SMT solving provides deterministic guarantees.

## Method

### Overall Architecture

Agentified Agent Assessment (AAA) = Assessor Agent (evaluation protocol) + Agent Under Test + Standard A2A Interface (AgentBeats/A2A Protocol). Evaluation logic is agentified and communicates with the agent under test through a standard interface.

### Key Designs

**1. AAA Evaluation Framework**:
- The assessor agent manages the full evaluation pipeline: task dispatch → execution budget control (timeouts / retry limits) → output parsing → structured failure-type logging → machine-readable evaluation report generation.
- Failure types are categorized as: Timeout (execution timeout), RuntimeError (runtime exception), and ParseError (output parsing failure) — distinguished from reasoning errors.
- Core value: integration cost is reduced from O(n) to O(1) — an agent implements the A2A interface once and can participate in any assessor's evaluation.
- No failure discarded: traditional harnesses count unparseable outputs as errors; AAA distinguishes execution failures from reasoning errors, supporting post-hoc auditing.

**2. FOLIO Data Cleaning Pipeline**:
- Step 1: Formal verification of FOL representations using the Vampire theorem prover — satisfiability of $\bigwedge_i \phi_i \wedge \neg\varphi$ is checked to validate True/False/Uncertain labels.
- Step 2: When verification results conflict with labels, a critique agent diagnoses translation errors (bracket mismatches, naming inconsistencies, etc.), and a refiner agent performs targeted repairs.
- Step 3: Iterative verify-repair until consistency is achieved; samples exceeding the threshold are flagged for human review.
- Results: 674 training samples (67.3%) passed verification directly, 23 (2.3%) passed after repair, and 304 (30.4%) remain problematic.

**3. Auto-formalization Agent**:
- Stage 1 (Code Generation): An LLM translates natural-language premises and conclusions into executable Z3Py programs.
- Stage 2 (Execution and Verification): Sandboxed execution (60s timeout); satisfiability checking determines True/False/Uncertain.
- Self-repair loop: up to 3 iterations; upon syntax errors, error messages are extracted for targeted code repair before retrying.

**4. CoT Baseline**: Standard chain-of-thought prompting requiring step-by-step reasoning before outputting a final label.

## Key Experimental Results

### Main Results

| Method | True Acc. | False Acc. | Uncertain Acc. | Overall Acc. |
|--------|:---------:|:----------:|:--------------:|:------------:|
| Chain-of-Thought | 89.04% | 44.26% | 84.06% | 73.89% |
| **Auto-formalization** | **90.41%** | **77.05%** | **91.30%** | **86.70%** |

### Ablation Study

| Analysis Dimension | Key Findings |
|-------------------|-------------|
| False class gain | +32.79 pp; contradiction detection is CoT's weakest point, where formal verification yields the most significant advantage. |
| Uncertain class gain | +7.24 pp; the SMT solver handles logical uncertainty more effectively. |
| True class | 89.04%→90.41%; already high, marginal improvement. |
| Effect of data cleaning | Cleaned labels are more reliable, yielding more faithful evaluation results. |

### Key Findings

- The substantial gain on the False class (44.26%→77.05%) reveals a systematic weakness of CoT in logical contradiction reasoning: models struggle to derive contradictions from premises (which requires proving that $\phi \wedge \neg\varphi$ is unsatisfiable).
- Formal verification replaces "seemingly correct" reasoning with deterministic logical guarantees.
- The backbone model is Gemini 2.5 Flash (T=0.0), ensuring deterministic outputs.
- Data cleaning exposes implicit quality issues in existing benchmarks — evaluating on incorrect labels systematically underestimates model capability.

## Highlights & Insights

- **Paradigm Innovation in Agentified Evaluation**: Transforming evaluation itself into an agent decouples assessment logic from agent implementation, reducing integration cost.
- **Structured Failure Logging**: Distinguishing Timeout / RuntimeError / ParseError from reasoning errors makes evaluations auditable and traceable.
- **Lessons from Data Cleaning**: Using a formal theorem prover to verify NLI dataset labels is more reliable and scalable than manual annotation.
- **Dramatic Gain on False Class**: Reveals a fundamental limitation of CoT in logical negation and contradiction reasoning — informal methods face a hard ceiling in this setting.

## Limitations & Future Work

- Validation is conducted on a single dataset (FOLIO validation set, 203 examples), with extremely small scale and limited statistical power.
- Only two agents are compared (CoT and Auto-formalization), lacking broader comparisons with methods such as LINC, Logic-LM, and SymbCoT.
- 30.4% of training samples remain flagged as "problematic" after the cleaning pipeline, leaving its completeness to be improved.
- Practical interoperability, communication latency, and overhead of the A2A interface are not analyzed in detail.
- The quality of auto-formalized Z3Py code is heavily dependent on the capabilities of the backbone LLM.
- Performance on more complex reasoning tasks (e.g., higher-order logic, probabilistic reasoning) is not evaluated.

## Related Work & Insights

- Compared to traditional static evaluation harnesses, AAA decouples assessment logic from agent implementation, representing an advancement in evaluation methodology.
- Built on the AgentBeats framework and A2A protocol, it aligns with the trend toward agent interoperability standards.
- The FOLIO cleaning work uses the Vampire theorem prover to verify NL-FOL alignment, serving as an exemplar of data quality assurance.
- The auto-formalization approach (NL→Z3Py) is in the same vein as LINC and Logic-LM, but adds a self-repair loop to improve robustness.

## Rating

- Novelty: ⭐⭐⭐ (The AAA framework concept is novel, though the auto-formalization technique is relatively straightforward.)
- Experimental Thoroughness: ⭐⭐ (Single dataset, few comparison methods, extremely small scale.)
- Writing Quality: ⭐⭐⭐⭐ (Clear and well-structured, with precise problem definitions.)
- Value: ⭐⭐⭐ (The agentified evaluation paradigm is thought-provoking; the data cleaning pipeline has practical value.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Scaling Generalist Data-Analytic Agents](scaling_generalist_data-analytic_agents.md)
- [\[ACL 2026\] Logical Phase Transitions: Understanding Collapse in LLM Logical Reasoning](../../ACL2026/llm_reasoning/logical_phase_transitions_understanding_collapse_in_llm_logical_reasoning.md)
- [\[ICLR 2026\] Estimating the Empowerment of Language Model Agents](estimating_the_empowerment_of_language_model_agents.md)
- [\[ICLR 2026\] Adaptive Social Learning via Mode Policy Optimization for Language Agents](adaptive_social_learning_via_mode_policy_optimization_for_language_agents.md)
- [\[ACL 2026\] When Is Thinking Enough? Early Exit via Sufficiency Assessment for Efficient Reasoning](../../ACL2026/llm_reasoning/when_is_thinking_enough_early_exit_via_sufficiency_assessment_for_efficient_reas.md)

</div>

<!-- RELATED:END -->
