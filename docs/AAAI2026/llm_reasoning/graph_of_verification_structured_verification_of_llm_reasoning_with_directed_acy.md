---
title: >-
  [Paper Note] Graph of Verification: Structured Verification of LLM Reasoning with Directed Acyclic Graphs
description: >-
  [AAAI 2026][LLM Reasoning][reasoning verification] This paper proposes Graph of Verification (GoV), a structured verification framework that models LLM reasoning processes as directed acyclic graphs (DAGs). Through a fle…
tags:
  - "AAAI 2026"
  - "LLM Reasoning"
  - "reasoning verification"
  - "directed acyclic graph"
  - "multi-granularity verification"
  - "decomposed verification"
  - "training-free method"
date: 2026-05-08
content_hash: 0a3293919b6fb107
---

# Graph of Verification: Structured Verification of LLM Reasoning with Directed Acyclic Graphs

**Conference**: AAAI 2026
**arXiv**: [2506.12509](https://arxiv.org/abs/2506.12509)  
**Code**: [Frevor/Graph-of-Verification](https://github.com/Frevor/Graph-of-Verification)  
**Area**: LLM Reasoning
**Keywords**: reasoning verification, directed acyclic graph, multi-granularity verification, decomposed verification, training-free method

## TL;DR

This paper proposes Graph of Verification (GoV), a structured verification framework that models LLM reasoning processes as directed acyclic graphs (DAGs). Through a flexible Node Block architecture, GoV enables multi-granularity verification—ranging from atomic-level steps in formal tasks to paragraph-level verification in natural language narratives—and substantially outperforms both holistic verification and other decomposed verification methods on both structured and loosely structured reasoning benchmarks.

---

## Background & Motivation

**Core challenge in LLM reasoning verification**: LLMs frequently produce reasoning steps that appear plausible but contain subtle logical flaws; even when the final answer is correct, the underlying process may be invalid. This calls for rigorous evaluation of the intrinsic validity of reasoning processes.

**Limitations of prior work — holistic verification**: Presenting an entire reasoning chain to a verifier LLM in one pass imposes excessive cognitive load, making it difficult to detect local defects; the problem worsens as reasoning chains grow longer and more complex.

**Limitations of prior work — trained verifiers**: Process reward models (PRMs) require large amounts of human-annotated data, incur high training costs, and struggle to keep pace with the rapid iteration of frontier LLMs.

**Limitations of prior work — existing decomposed verification**: PARC and Deductive Verification pursue atomic-granularity verification with minimal premises, which is theoretically most precise but relies on fragile premise extraction or claim-step dependencies, making them unreliable for loosely structured natural language reasoning.

**Root cause — adaptability gap**: No unified framework currently exists to flexibly handle reasoning structures ranging from highly formal proofs to highly unstructured natural language narratives—a key obstacle to achieving truly reliable reasoning verification.

**Core Idea**: Inspired by human cognition—when confronted with complex arguments, people instinctively decompose them into dependency sequences for stepwise verification—this process can naturally be modeled as a DAG. Crucially, verification granularity should match the structure of the reasoning: atomic granularity for structured reasoning to maximize precision, and block granularity for natural language reasoning to ensure robustness.

---

## Method

### Overall Architecture: Four-Stage Verification Pipeline

1. **DAG Modeling**: The reasoning process is modeled as a directed acyclic graph $\mathcal{G} = (V, E)$.
2. **Topological Sorting**: A topological sort is applied to the DAG, ensuring premises are verified before conclusions.
3. **Sequential Verification**: An LLM verifies each unit in topological order, conditioned on already-verified predecessor information.
4. **Early Stopping & Error Localization**: Verification terminates at the first detected error, enabling precise fault localization.

### Key Design 1: Formalization of the Two-Dimensional Design Space

The paper is the first to formalize decomposed verification as a two-dimensional design space defined by **verification granularity** and **context scope**, providing a principled theoretical framework for selecting verification strategies.

**Dimension 1: Verification Granularity**
- **Atomic granularity**: Reasoning is decomposed into minimal indivisible logical units (e.g., a single arithmetic operation $a + b = c$), enabling the most precise error localization but requiring highly structured reasoning.
- **Block granularity**: Multiple related atomic steps are aggregated into semantically coherent units (e.g., complete paragraphs), offering greater robustness and adaptability to natural language at the cost of localization precision.

**Dimension 2: Context Scope**
- **Minimal context**: Only the direct premises of the current unit are provided, reducing cognitive load and noise, but dependent on potentially error-prone premise extraction.
- **Inclusive context**: All previously verified information is provided, which is safer and more robust but may introduce redundancy.

**Key Insight**: The optimal configuration depends on the inherent structure of the reasoning—structured reasoning favors "atomic granularity + minimal context," while natural language reasoning benefits from "block granularity + inclusive context."

### Key Design 2: DAG Modeling of the Reasoning Process

- **Nodes $V = \{v_1, v_2, \dots, v_n\}$**: The basic verification units of reasoning, classified into three types:
    - Foundational elements: premises, axioms, facts (root nodes of the DAG)
    - Derived statements: intermediate conclusions drawn from predecessor nodes via logical rules or computation
    - Terminal statements: the final conclusion/goal of the reasoning process
- **Directed edges $E \subseteq V \times V$**: $(v_i, v_j) \in E$ indicates that $v_i$ is a direct premise of $v_j$, and the validity of $v_j$ depends on $v_i$.
- **Theoretical advantages of the DAG**:
    - Directionality enforces the premise-to-conclusion direction of reasoning, eliminating ambiguity.
    - Acyclicity guarantees the absence of circular dependencies, preventing self-justifying reasoning.

### Key Design 3: Multi-Granularity Topological Units

**Topological sorting**: A topological ordering $\sigma: V \to \{1, 2, \dots, n\}$ is computed such that $\sigma(v_i) < \sigma(v_j)$ holds for all edges $(v_i, v_j)$, yielding the verification sequence $\mathcal{C}_{\text{verif}} = (c_1, c_2, \dots, c_n)$.

**Atomic nodes**: Each node $v_k$ serves as an independent verification unit, enabling the finest-grained error localization; suited for structured reasoning.

**Node Blocks**: Topologically contiguous nodes are grouped into logically coherent blocks $\mathcal{B} = (B_1, \dots, B_m)$, subject to two constraints:
- **Topological consistency**: $\max_{v \in B_j} \sigma(v) < \min_{v \in B_k} \sigma(v)$, $j < k$, preserving macro-level dependencies.
- **Semantic coherence**: Each block corresponds to a complete logical unit (e.g., a lemma in a proof or a paragraph in a narrative); arbitrary segmentation is prohibited.

Degenerate case: when $m = 1$, the framework reduces to conventional holistic verification.

### Key Design 4: Atomic Node Verification Mechanism

The verification function $\text{Verify}(c_k, Pred_{\text{prov}}(c_k))$ is applied to each node in topological order:

- **Foundational nodes** ($Pred(c_k) = \emptyset$): Verified directly for consistency with the problem statement or domain knowledge.
- **Derived nodes** ($Pred(c_k) \neq \emptyset$): Verification requires both conditions to hold simultaneously:
  1. All direct predecessors have been verified as True.
  2. The reasoning step from the predecessors to the current node is itself valid.

$$T(c_k) = (\forall c_i \in Pred(c_k), T(c_i) = \text{True}) \land \text{Verify}(c_k, Pred_{\text{prov}}(c_k))$$

### Key Design 5: Node Block Verification Mechanism

- **External premise set**: $Pred_{\text{ext}}(B_j) = (\bigcup_{v \in V(B_j)} Pred(v)) \setminus V(B_j)$, i.e., all external dependencies of internal nodes.
- **In practice**: For natural language paragraphs, $Pred_{\text{prov}}(B_j)$ typically includes the full content of all previously verified blocks $(B_1, \dots, B_{j-1})$.
- The LLM simultaneously verifies the internal coherence of the block and the validity of all statements within the block with respect to external premises.

### Early Stopping and Error Localization

Verification proceeds in topological order (or block order); upon detecting any unit as False, that unit is marked as the earliest failure point and verification halts. The overall reasoning process is deemed valid only if all units pass verification.

---

## Key Experimental Results

### Experiment 1: Number Triangle Summation (Structured Reasoning)

- **Task design**: Starting from $N$ initial numbers, adjacent numbers are summed layer by layer until a single number remains. Each reasoning process has a 50% probability of containing a unit error.
- **Configuration**: Atomic granularity + minimal context (precision mode).
- **Results** (Qwen2.5-72B-Instruct, F1 Score):

| Problem Scale $N$ | Holistic Verification | GoV |
|---|---|---|
| $N = 2$ | 99.6 | 98.7 |
| $N = 4$ | 93.3 | **97.9** |
| $N = 6$ | 46.3 | **98.1** |
| $N = 8$ | 49.5 | **98.1** |

As problem complexity increases, holistic verification F1 drops sharply (from 99.6 to 49.5), while GoV consistently maintains a high level above 97.

### Experiment 2: ProcessBench (Loosely Structured Reasoning)

- **Dataset**: Reasoning process verification on GSM8K, MATH, and OlympiadBench.
- **Configuration**: Block granularity + inclusive context (robust mode).
- **Results** (Qwen2.5-7B-Instruct, F1 Score):

| Dataset | Holistic Verification | PARC | GoV |
|---|---|---|---|
| GSM8K | 47.3 | 47.0 | **58.0** |
| MATH | 34.6 | 43.3 | **55.0** |
| OlympiadBench | 32.7 | - | **42.9** |

GoV significantly outperforms both holistic verification and PARC across all datasets, with F1 improvements exceeding 20 percentage points on MATH.

### Key Findings

1. **Scalability**: GoV's advantage becomes increasingly pronounced as problem complexity grows, addressing the critical weakness of holistic verification on long reasoning chains.
2. **Flexibility**: The same framework adapts to fundamentally different task types by configuring different granularities.
3. **Model generalization**: GoV is effective across models ranging from 7B to 72B parameters.
4. **Training-free**: No annotated data or fine-tuning is required; the method is ready to use out of the box.

---

## Highlights & Insights

**Strengths**

- The formalization of the two-dimensional design space is a significant theoretical contribution, providing clear principled guidance for decomposed verification.
- The Node Block architecture elegantly unifies atomic and block verification, spanning a continuum from precision to robustness within a single framework.
- The training-free design confers high flexibility and practical utility.

**Limitations & Future Work**

- DAG construction itself relies on the LLM's decomposition capability, which may be unreliable for extremely loosely structured reasoning.
- Number Triangle Summation is an artificially constructed benchmark with limited correspondence to real-world reasoning scenarios.
- The early stopping strategy may miss independent errors in subsequent steps.

---

## Related Work & Insights

- **Chain-of-Thought / Tree-of-Thoughts / Graph-of-Thoughts**: These focus on the **generation** of reasoning processes; GoV focuses on the **verification** of reasoning processes, representing a natural extension of structured reasoning generation.
- **PARC (Premise-Augmented Reasoning Chains)**: Pursues atomic-granularity + minimal-premise decomposed verification; fragile in natural language settings due to unreliable premise extraction. GoV addresses this by flexibly adjusting granularity via Node Blocks.
- **Deductive Verification**: Constrains the generation process to a "natural program" format with explicit premise declarations, limiting applicability; GoV verifies existing reasoning in a post-hoc manner, offering broader applicability.
- **Process Reward Model (PRM)**: A trained verifier with high precision but high cost that requires retraining to follow model updates; GoV is training-free and thus more flexible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Efficient PRM Training Data Synthesis via Formal Verification](../../ACL2026/llm_reasoning/efficient_prm_training_data_synthesis_via_formal_verification.md)
- [\[NeurIPS 2025\] Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling](../../NeurIPS2025/llm_reasoning/rethinking_optimal_verification_granularity_for_compute-efficient_test-time_scal.md)
- [\[ICLR 2026\] Verifying Chain-of-Thought Reasoning via Its Computational Graph](../../ICLR2026/llm_reasoning/verifying_chain-of-thought_reasoning_via_its_computational_graph.md)
- [\[NeurIPS 2025\] Clip-and-Verify: Linear Constraint-Driven Domain Clipping for Accelerated Neural Network Verification](../../NeurIPS2025/llm_reasoning/clip-and-verify_linear_constraint-driven_domain_clipping_for_accelerating_neural.md)
- [\[AAAI 2026\] Dropouts in Confidence: Moral Uncertainty in Human-LLM Alignment](dropouts_in_confidence_moral_uncertainty_in_human-llm_alignment.md)

</div>

<!-- RELATED:END -->
