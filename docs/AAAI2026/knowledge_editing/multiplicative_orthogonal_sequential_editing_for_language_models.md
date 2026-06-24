---
title: >-
  [Paper Note] Multiplicative Orthogonal Sequential Editing for Language Models (MOSE)
description: >-
  [AAAI 2026][Knowledge Editing][Orthogonal Transformation] This paper proposes MOSE (Multiplicative Orthogonal Sequential Editing), which injects new knowledge by left-multiplying the parameter matrix with an orthogonal matrix (rather than via additive updates), strictly preserving the Frobenius norm and condition number of the edited matrix. MOSE achieves a 12.08% performance improvement in sequential editing while retaining 95.73% of general capabilities.
tags:
  - "AAAI 2026"
  - "Knowledge Editing"
  - "Orthogonal Transformation"
  - "Sequential Editing"
  - "Numerical Stability"
  - "Model Editing"
date: 2026-05-08
content_hash: afa82472b5562c33
---

# Multiplicative Orthogonal Sequential Editing for Language Models (MOSE)

**Conference**: AAAI 2026
**arXiv**: [2601.07873](https://arxiv.org/abs/2601.07873)  
**Code**: [https://github.com/famoustourist/MOSE](https://github.com/famoustourist/MOSE)  
**Area**: Knowledge Editing
**Keywords**: Knowledge Editing, Orthogonal Transformation, Sequential Editing, Numerical Stability, Model Editing

## TL;DR

This paper proposes MOSE (Multiplicative Orthogonal Sequential Editing), which injects new knowledge by left-multiplying the parameter matrix with an orthogonal matrix (rather than via additive updates), strictly preserving the Frobenius norm and condition number of the edited matrix. MOSE achieves a 12.08% performance improvement in sequential editing while retaining 95.73% of general capabilities.

## Background & Motivation

### Limitations of Prior Work

**Background**: Knowledge editing aims to efficiently modify the internal knowledge of LLMs without impairing other capabilities. In sequential editing scenarios (i.e., continuous successive edits), all existing methods adopt an additive paradigm—adding an update matrix to the original parameter matrix: $W = W_0 + \Delta W_1 + \Delta W_2 + \cdots$.

**Root Cause of Additive Editing**: Repeated additive updates severely compromise the numerical stability of parameter matrices—the Frobenius norm grows continuously and the condition number rises sharply, causing degradation in both editing performance and general model capabilities. Although methods such as RECT and AlphaEdit partially alleviate this issue, they remain within the additive framework and merely delay the degradation.

**Core Observation**: It can be mathematically proven that left-multiplication by an orthogonal matrix preserves both the Frobenius norm and the condition number of a matrix: $\|RW\|_F = \|W\|_F$ and $\kappa_2(RW) = \kappa_2(W)$. This implies that encoding knowledge updates into an orthogonal matrix can fundamentally prevent numerical stability degradation.

## Method

### Overall Architecture

MOSE transforms knowledge editing from an "additive update" to a "multiplicative update": instead of $W' = W_0 + \Delta W$, it computes $W' = R \cdot W_0$, where $R$ is an orthogonal matrix ($R^\top R = I$). By encoding new knowledge into an orthogonal transformation, MOSE edits knowledge while strictly preserving the numerical stability of the parameter matrix.

### Key Designs

1. **Orthogonal Transformation Update**

    - Optimization objective: $\min_R \lambda \|RW_0K_0 - W_0K_0\|_F^2 + \|RW_0K_E - V_E\|_F^2$
    - The first term preserves existing knowledge representations (retention term); the second term aligns new knowledge representations with the target (editing term).
    - $\lambda$ controls the trade-off between retaining old knowledge and injecting new knowledge.
    - This is a standard orthogonal Procrustes problem with a closed-form solution: SVD of $M = BA^\top$ yields $R = UV^\top$.

2. **Orthogonal Projection for Attention Layers**

    - Orthogonal transformations are applied to the Q, K, and V projection matrices of attention layers.
    - Q and K layers share a single orthogonal matrix, since attention computation involves the inner product of Q and K and consistency must be maintained.
    - This ensures that the semantic structure of the attention computation is not disrupted.

3. **Activation Function Expansion for FFN Layers**

    - The nonlinear activation functions in FFN layers make straightforward application of orthogonal transformations difficult.
    - The activation function is approximated via polynomial expansion, and the orthogonal transformation is applied in the expanded space.
    - Higher-order remainder terms of the Taylor expansion act as noise but do not affect overall stability.

### Loss & Training

No additional training is required. $R$ is obtained in a single step via a closed-form solution (SVD). The construction of $K_0$ (keys for retained knowledge), $K_E$ (keys for edited knowledge), and $V_E$ (target values) follows the methodology of ROME/MEMIT.

## Key Experimental Results

### Main Results (Sequential Editing over 1,000 Steps)

| Method | CounterFact Reliability | Generalization | Locality | General Capability Retention |
|--------|------------------------|---------------|----------|------------------------------|
| ROME | 0.000 | 0.000 | 0.000 | Severe degradation |
| MEMIT | 0.000 | 0.000 | 0.000 | Severe degradation |
| RECT | 0.569 | 0.329 | 0.252 | Moderate |
| AlphaEdit | ~Good | ~Good | ~Good | Good |
| **MOSE** | **Best** | **Best** | **Best** | **95.73%** |

### Ablation Study (Numerical Stability)

| Method | Frobenius Norm Change after 1,000 Edits | Condition Number Change |
|--------|-----------------------------------------|------------------------|
| ROME/MEMIT | Surges (order-of-magnitude growth) | Surges |
| RECT/AlphaEdit | Slow growth | Moderate growth |
| **MOSE** | **Unchanged** | **Unchanged** |

### Key Findings

- **ROME and MEMIT collapse completely after approximately 100 sequential edits**, with reliability dropping to 0, demonstrating that the additive paradigm is infeasible for long-sequence editing.
- **The norm and condition number of MOSE remain entirely unchanged after 1,000 edits**, strictly validating the theoretical predictions.
- **Consistent effectiveness across three different models**: LLaMA3-8B, LLaMA2-13B, and Qwen2.5-7B.
- Retains 95.73% of general capabilities across 4 downstream tasks, substantially outperforming other editing methods.

## Highlights & Insights

- **Paradigm shift from additive to multiplicative updates**: All prior work patches the additive framework; MOSE is the first to break out of it and provides a mathematical proof of why multiplicative (orthogonal) updates are fundamentally superior.
- **Elegant solution via the orthogonal Procrustes problem**: Knowledge editing is reformulated as a classical mathematical problem with a closed-form solution, requiring no iterative optimization and thus computationally efficient.
- **Practical significance for long-sequence editing**: LLMs require continuous knowledge updates during deployment; MOSE makes it feasible to perform thousands of successive edits without performance degradation.

## Limitations & Future Work

- The polynomial expansion for FFN layers is approximate; it remains unclear whether noise from higher-order remainder terms accumulates over an extremely large number of edits.
- SVD-based solution of the orthogonal Procrustes problem incurs computational overhead for large matrices; acceleration strategies warrant investigation.
- Validation is limited to structured knowledge triples; effectiveness on more complex knowledge types (e.g., implicit knowledge, skills) remains to be explored.
- The constraint of sharing a single orthogonal matrix across Q and K may be overly restrictive and could potentially limit the expressive capacity of the edits.

## Related Work & Insights

- **vs. ROME/MEMIT (Meng 2022, 2023)**: Classic additive editing methods that are effective for small numbers of edits but collapse rapidly under sequential editing.
- **vs. AlphaEdit (Fang 2025)**: Performs additive updates under null-space constraints, partially alleviating stability issues but unable to resolve them fundamentally; performance still degrades after a large number of edits.
- **vs. PRUNE (Ma 2025)**: Mitigates the problem by constraining the condition number, but the constraint itself introduces additional computational overhead; MOSE preserves the condition number naturally.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The multiplicative orthogonal editing paradigm represents a genuine paradigm breakthrough, supported by rigorous mathematical proof.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across three models, two datasets, and six baselines.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from statistical analysis → mathematical proof → method design → experimental validation is seamless.
- Value: ⭐⭐⭐⭐⭐ Provides a fundamental solution to the practically important problem of continual knowledge editing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Neuron-Level Sequential Editing for Large Language Models](../../ACL2025/knowledge_editing/neuron-level_sequential_editing_for_large_language_models.md)
- [\[ICML 2026\] The Labyrinth and the Thread: Rethinking Regularizations in Sequential Knowledge Editing for Large Language Models](../../ICML2026/knowledge_editing/the_labyrinth_and_the_thread_rethinking_regularizations_in_sequential_knowledge_.md)
- [\[ICLR 2026\] Energy-Regularized Sequential Model Editing on Hyperspheres](../../ICLR2026/knowledge_editing/energy-regularized_sequential_model_editing_on_hyperspheres.md)
- [\[ICML 2026\] Revisiting Parameter-Based Knowledge Editing in Large Language Models: Theoretical Limits and Empirical Evidence](../../ICML2026/knowledge_editing/revisiting_parameter-based_knowledge_editing_in_large_language_models_theoretica.md)
- [\[ICML 2026\] Reverse-Engineering Model Editing on Language Models](../../ICML2026/knowledge_editing/reverse-engineering_model_editing_on_language_models.md)

</div>

<!-- RELATED:END -->
