---
title: >-
  [Paper Note] Circuit Stability Characterizes Language Model Generalization
description: >-
  [ACL 2025][LLM (Other)][Circuit stability] This paper proposes "circuit stability" as a novel approach to evaluate the generalization capabilities of language models. By mathematically formalizing soft circuits and circuit equivalence, it demonstrates across three case studies (arithmetic reasoning, boolean expressions, and sports understanding) that circuit stability can predict and characterize generalization behavior.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Circuit stability"
  - "mechanistic interpretability"
  - "generalization"
  - "soft circuits"
  - "persistent homology"
date: 2026-05-08
content_hash: 8ff4ecb621af79d4
---

# Circuit Stability Characterizes Language Model Generalization

**Conference**: ACL 2025  
**arXiv**: [2505.24731](https://arxiv.org/abs/2505.24731)  
**Code**: [github](https://github.com/alansun17904/circuit-stability)  
**Area**: LLM/NLP  
**Keywords**: Circuit stability, mechanistic interpretability, generalization, soft circuits, persistent homology

## TL;DR

This paper proposes "circuit stability" as a novel approach to evaluate the generalization capabilities of language models. By mathematically formalizing soft circuits and circuit equivalence, it demonstrates across three case studies (arithmetic reasoning, boolean expressions, and sports understanding) that circuit stability can predict and characterize generalization behavior.

## Background & Motivation

Evaluating the capabilities of language models faces two major hurdles: (1) existing benchmarks saturate rapidly; and (2) constructing more challenging datasets requires substantial human labor. While evaluation methods tailored for specific capabilities exist (such as needle-in-the-haystack for long-context recall), determining which capabilities are worth evaluating in the first place remains a non-trivial problem.

To address this, the authors propose three core insights:

**Circuit Perspective**: Instead of testing input-output pairs individually, the model's circuits (subgraphs of the reasoning process) are extracted and analyzed. These circuits can apply to an infinite set of samples.

**Simplifying Assumption**: Rather than pre-specifying skills of interest, it is assumed that a learned skill/circuit is only useful if it is consistently applied by the model.

**Continuous Relaxation**: Instead of extracting hard-to-find "hard" circuits (discrete subgraphs), the authors introduce "soft" circuits (continuous mappings), which retain rich structural information while facilitating computation.

## Method

### Overall Architecture

Every edge in the computation graph of a Transformer is assigned a continuous importance score (soft circuit). Circuit stability is then measured by comparing the similarities of soft circuits across different subtasks. The core conceptual chain is: Task $\to$ Subtask Partitioning $\to$ Soft Circuit $\to$ Circuit Stability/Equivalence.

### Key Designs

1. **Task and Subtask Definitions (Definition 1-2)**:

    - A task is defined as a distribution $D$ over $X \times Y$.
    - Subtasks are obtained by partitioning $X \times Y$ meaningfully.
    - For instance, two-digit addition can be partitioned by operand lengths: subtask $(o_1, o_2)$ represents instances where the first operand is of length $o_1$ and the second is $o_2$.

2. **Soft Circuit Definition (Definition 3)**:

    - Traditional hard circuits: $c: E^M \to \{0,1\}$ (binary), where searching is an NP-hard combinatorial optimization problem.
    - Soft circuits: $c: E^M \to \mathbb{R}$ (continuous), where $c(e)$ denotes the expected change in a performance metric $L$ upon ablating edge $e$.
    - $$c(e) = \mathbb{E}_{(x,y) \sim D}[L(M_{e}(x), y) - L(M(x), y)]$$
    - As long as $L$ is well-defined, $c$ always exists, avoiding the discrete search space difficulty of hard circuits.

3. **$\epsilon$-Circuit Stability (Definition 4)**:

    - For a model $M$, task distribution $D$, and a set of partitions $P$, the model is $\epsilon$-circuit stable if the expected similarity of soft circuits from two randomly sampled subtasks $s, s' \in P$ exceeds $\epsilon$.
    - Spearman's rank correlation coefficient $\rho$ is used as the similarity metric $K$.
    - Intuition: A stable model utilizes consistent reasoning processes across different subtasks.

4. **$\alpha$-Circuit Equivalence (Definition 5)**:

    - The soft circuits $c_s, c_{s'}$ of two subtasks are $\alpha$-equivalent if and only if $K(c_s, c_{s'}) \ge \alpha$.
    - This is used to identify which subtasks share similar circuits (for clustering analysis).

### Loss & Training

This work proposes an analysis framework rather than a training method. Circuit discovery is measured via next-token patching, implemented as noisy-to-clean patching for edge ablation. The primary computational overhead lies in circuit discovery for each subtask (requiring approximately 2 forward passes and 1 backward pass per subtask).

## Key Experimental Results

### Main Results

**Case Study 1: Arithmetic Reasoning (gemma-2-2b, 79k circuit edges)**

Partitioning two-digit addition by operand lengths $(o_1, o_2)$ yields 64 subtasks:

| Analysis | Finding |
|------|------|
| $\alpha=0.6$ equivalent clustering | 5 distinct circuit families emerge: equal-length, one-digit difference, first-operand heavy, single-digit, and second-operand heavy. |
| $\alpha=0.4$ | A single cluster (sharing core arithmetic components). |
| Commutativity violation | The first-operand-heavy and second-operand-heavy circuit families split at $\alpha=0.53$. |
| Performance discrepancy | Performance differences between $(o_1, o_2)$ and $(o_2, o_1)$ can exceed 20%, aligning with circuit non-equivalence. |
| Associativity violation | Adjacent subtasks such as $(6,6)$ vs $(6,7)$ belong to different circuit families. |

**Case Study 2: Boolean Expressions (phi-1.5, 128k circuit components)**

| Subtask | Within-parenthesis Stability | Without-parenthesis Stability | Cross-parenthesis Stability | Performance Change |
|------|------|----------|------|------|
| not | High | High | Significantly different* | Performance drops by 40% when parenthesized |
| not+and | Medium | Medium | Significantly different* | Stable performance |
| not+and+or | Medium | Medium | Significantly different* | Stable performance |

The circuit instability on the `not` subtask aligns with the performance drop (the model fails to understand associativity), whereas instability in `not+and` and `not+and+or` is "expected" because parentheses alter the order of operations.

**Case Study 3: Sports Understanding (Chain-of-Thought)**

| Model | Few-shot Accuracy | CoT Accuracy | Few-shot Circuit Stability | CoT Circuit Stability |
|------|------|----------|------|------|
| Llama-3.1-8b | ~75% | ~88%| ~0.55 | ~0.75 |
| Gemma-2-9b | ~80% | ~93% | ~0.60 | ~0.80 |

CoT significantly improves circuit stability ($p < 0.05$), supporting the hypothesis that CoT improves performance by facilitating subtask decomposition and circuit component reuse.

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| $\alpha$ varying from $0 \to 1$ | Number of clusters increases monotonically | 80% of circuit families emerge when $\alpha \in [0.58, 0.79]$. |
| t-SNE Visualization | Circuit families form clearly separated groups | Validates that the $\alpha=0.6$ clustering is not an artifact of a specific $\alpha$. |
| Hard Circuit Comparison | Subtasks in the same family share many components | Some hard circuits exhibit a sub-circuit relationship with each other. |

### Key Findings

1. **Circuit instability predicts generalization failure**: The split of circuit families in arithmetic tasks aligns with the model's generalization failure regarding commutativity and associativity.
2. **Higher circuit stability is not always better**: In boolean expressions, circuit instability when parentheses change the operation semantics is actually correct behavior.
3. **CoT induces circuit stability**: Chain-of-Thought prompting significantly enhances stability, supporting a mechanistic explanation where CoT facilitates component reuse.
4. **Critical transition threshold**: An explosive growth in the number of circuit families occurs around $\alpha=0.6$, revealing the dividing line of internal reasoning mechanisms within the model.

## Highlights & Insights

1. **Outstanding Mathematical Rigor**: The paper progresses step-by-step from definitions to experiments, offering mathematically precise and elegant formal definitions of soft circuits, $\epsilon$-stability, and $\alpha$-equivalence.
2. **Pragmatic Choice of Continuous Relaxation**: This avoids the NP-hard discrete search space of hard circuits while retaining rich structural information.
3. **Implicit Introduction of Occam's Razor**: Employing different circuits for different subtasks implies a longer minimum description length, hinting at poorer generalization.
4. **Method Generality**: The theoretical framework is agnostic to modalities and architectures, allowing extension to non-Transformer models.
5. **Actionability**: Circuit stability acts not only as a diagnostic tool but also points toward prospective improvements (such as CoT-induced stability or causal alignment during alignment training).

## Limitations & Future Work

1. Case studies are relatively limited, covering only three tasks and a few models (gemma-2-2b, phi-1.5, Llama-3.1-8b, Gemma-2-9b).
2. The construction of partitions relies on prior knowledge, making it difficult to determine meaningful partitions for complex tasks.
3. The choice of circuit abstraction granularity (MLP layers vs. attention heads vs. finer-grained components) may affect the conclusions.
4. Whether Spearman's $\rho$ is the optimal choice for the similarity metric $K$ remains theoretically unanalyzed.
5. The sports understanding task employs random partitioning, which does not represent true subtasks in a strict sense.

## Related Work & Insights

This work is closely related to mechanistic interpretability (Olah et al., 2020), circuit discovery (Conmy et al., 2023), and skill composition (Arora and Goyal, 2023). In contrast to existing ad-hoc circuit explanation works, this paper provides a general mathematical framework to compare and quantify cross-task circuit consistency. This approach can be extended to analyze the effects of RLHF alignment on circuits, circuit evolution during different training stages, and other directions.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The concept of circuit stability is novel, with an elegant mathematical framework and unique insights.
- **Experimental Thoroughness**: ⭐⭐⭐ The case studies are compelling but limited in number, and model coverage is somewhat restricted.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Precise mathematical definitions, deep case analyses, and highly coherent arguments.
- **Value**: ⭐⭐⭐⭐ Provides a fresh perspective on understanding LLM generalization mechanisms, offering substantial theoretical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Circuit Compositions: Exploring Modular Structures in Transformer-Based Language Models](circuit_compositions_modular_structures.md)
- [\[ACL 2025\] Systematic Generalization in Language Models Scales with Information Entropy](systematic_generalization_in_language_models_scales_with_information_entropy.md)
- [\[ACL 2025\] Computation Mechanism Behind LLM Position Generalization](computation_mechanism_behind_llm_position_generalization.md)
- [\[ACL 2025\] ConsistencyChecker: Tree-based Evaluation of LLM Generalization Capabilities](consistencychecker_tree_evaluation.md)
- [\[ACL 2025\] Revisiting Compositional Generalization Capability of Large Language Models Considering Instruction Following Ability](compositional_generalization_instruction.md)

</div>

<!-- RELATED:END -->
