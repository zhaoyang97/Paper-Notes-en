---
title: >-
  [Paper Note] Transformers Provably Learn Chain-of-Thought Reasoning with Length Generalization
description: >-
  [NeurIPS 2025][LLM Reasoning][CoT Theory] This paper provides the first optimization-theoretic proof that a one-layer Transformer trained via gradient descent can learn CoT reasoning on a synthetic state-tracking task an…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "CoT Theory"
  - "Length Generalization"
  - "Attention Concentration"
  - "State Tracking"
  - "Complexity Classes"
date: 2026-05-08
content_hash: b970fecf6eea7166
---

# Transformers Provably Learn Chain-of-Thought Reasoning with Length Generalization

**Conference**: NeurIPS 2025
**arXiv**: [2511.07378](https://arxiv.org/abs/2511.07378)
**Code**: None
**Area**: LLM Reasoning
**Keywords**: CoT Theory, Length Generalization, Attention Concentration, State Tracking, Complexity Classes

## TL;DR
This paper provides the first optimization-theoretic proof that a one-layer Transformer trained via gradient descent can learn CoT reasoning on a synthetic state-tracking task and achieve length generalization. It is the first work to establish convergence guarantees for constant-depth Transformers learning $\mathsf{NC}^1$-complete problems, going beyond prior theory that was limited to $\mathsf{TC}^0$.

## Background & Motivation

**Background**: Chain-of-thought (CoT) reasoning is a key technique enabling modern LLMs to tackle complex tasks by generating intermediate reasoning steps that exceed the model's single-step processing capacity. Theoretically, it is known that Transformers with CoT can express $\mathsf{NC}^1$-level problems, surpassing the $\mathsf{TC}^0$ limitation of CoT-free models.

**Limitations of Prior Work**: Existing theoretical work either studies expressiveness alone ("can it represent?") without addressing optimization ("can it be learned?"), or proves convergence guarantees only for simple $\mathsf{TC}^0$ tasks such as parity. A critical gap remains: no prior work has proven that Transformers can learn, through training, to produce CoT that generalizes to longer inputs.

**Key Challenge**: Expressiveness results do not imply learnability. Even if a set of weights exists that enables a Transformer with CoT to solve a problem, it is unclear whether gradient descent can find those weights and whether the resulting model generalizes to longer inputs.

**Goal**: To provide rigorous convergence and length generalization guarantees for Transformers learning CoT reasoning via gradient descent, at the complexity level of $\mathsf{NC}^1$.

**Key Insight**: The authors focus on LEGO (a group-operation-based state-tracking task) as the theoretical benchmark. LEGO is a canonical problem separating $\mathsf{TC}^0$ from $\mathsf{NC}^1$ and possesses rich algebraic structure that can be exploited analytically.

**Core Idea**: The *attention concentration* mechanism links the retrieval robustness of the attention layer to the algebraic structure of the state-tracking task, showing that the group structure determines the extent of length generalization.

## Method

### Overall Architecture

The analysis concerns a one-layer Transformer (softmax attention + smoothed ReLU MLP) without positional encoding (NoPE), trained with a CoT loss on the LEGO state-tracking task. The LEGO task requires the model to execute a sequence of group operations $g_1, g_2, \ldots, g_L$ and autoregressively output intermediate states $y_1, y_2, \ldots, y_L$, where $y_{i+1} = g_{i+1}(y_i)$.

### Key Designs

1. **Attention Concentration Mechanism**:

    - *Function*: Characterizes how the attention layer implements precise retrieval of relevant context.
    - *Mechanism*: The Transformer learns two key retrieval patterns via the attention matrix $Q$: (1) $Q_{4,3}$ retrieves the corresponding group operation $g_{i+1}$ from context; (2) $Q_{4,4}$ retrieves the preceding answer $y_i$. When both attention distributions are sufficiently "concentrated," the MLP can correctly compute $g_{i+1}(y_i)$.
    - *Design Motivation*: The key question for length generalization is whether attention remains focused on the correct positions as input length increases. This depends on the algebraic structure of the task.

2. **Generalization Analysis under Two Group Structures**:

    - **Simply transitive group operations (e.g., cyclic groups)**: The learned features exhibit high "purity," yielding highly concentrated attention (attention mass on the correct clause is $1 - 1/d^c$), enabling generalization to lengths $\text{poly}(d)$ times the training length.
    - **Symmetric group operations ($S_n$)**: The learned features are insufficiently "pure," so direct training generalizes only to a constant multiple of the training length. However, recursive self-training (progressive length doubling) can iteratively extend the solvable length.

3. **Recursive Self-Training Scheme**:

    - *Function*: Progressively extends the model's effective input length through curriculum learning.
    - *Mechanism*: The model is first trained to convergence on length $L$; its greedy outputs are then used to generate training data of length $2L$, and training continues. After each doubling round, model accuracy reaches $1 - O(2^{-k/\text{poly}(d)})$.
    - *Design Motivation*: For complex group structures that do not admit direct length generalization, self-training provides a principled path to progressively expanding capability beyond the coverage of the original training data.

### Loss & Training
- Two-stage curriculum learning: first train the MLP to learn single-step group operations (one-step reasoning), then freeze the MLP and train the attention to perform contextual retrieval.
- No positional encoding is used. Theoretical analysis shows that standard PE schemes (e.g., RoPE) inherently hinder length generalization due to their positional dependence.

## Key Experimental Results

### Main Results (Simply Transitive / Cyclic Group $C_6$)

| Training Length | Test L=5 | Test L=10 | Test L=20 | Test L=30 | Test L=40 | Test L=50 |
|----------------|----------|-----------|-----------|-----------|-----------|-----------|
| CoT (L=5) | 100% | 100% | 100% | 100% | 99.7% | 98.1% |
| Non-CoT (L=5) | 16.4% | 16.9% | 15.1% | 18.1% | 19.2% | 19.0% |
| Non-CoT (L=10) | 16.8% | 18.4% | 16.6% | 17.9% | 17.4% | 17.9% |

### Ablation Study

| Configuration | Key Finding |
|---------------|-------------|
| CoT vs. Non-CoT | Non-CoT models fail to learn the task even within the training distribution (near-random performance) |
| Cyclic group vs. Symmetric group | Cyclic group generalizes to 10× training length; symmetric group generalizes to only ~2–3× |
| No PE vs. with PE | Removing PE is necessary for length generalization (standard PE biases toward local positions) |

### Key Findings
- **CoT is necessary**: Transformers without CoT perform near chance (~17%) even within the training distribution, confirming that state tracking genuinely exceeds $\mathsf{TC}^0$ capability.
- **Algebraic structure governs generalization**: Simply transitive groups (cyclic groups) yield far better generalization than symmetric groups, consistent with theoretical predictions.
- **Recursive self-training is effective**: Under the $S_5$ setting, each length-doubling round successfully brings the model to high accuracy on the new target length.
- **Attention concentration is empirically verifiable**: Post-training attention weights are observed to concentrate on the correct clauses, as predicted by theory.

## Highlights & Insights
- **First optimization guarantee at the $\mathsf{NC}^1$ level**: Prior theory addressed only $\mathsf{TC}^0$ tasks such as parity. This work extends convergence guarantees to $\mathsf{NC}^1$-complete problems, representing a qualitative advance in theoretical complexity.
- **Attention concentration $\Leftrightarrow$ length generalization**: The paper presents a clean and elegant picture: length generalization depends on retrieval robustness of attention, which depends on feature "purity," which in turn depends on group structure.
- **Theoretical support for NoPE**: The paper provides a principled explanation for why removing positional encoding facilitates length generalization—the positional bias introduced by standard PE degrades the attention mechanism's long-range retrieval ability.
- **Self-training transcends training data**: The paper proves that self-training genuinely expands the model's capability boundary rather than merely reinforcing what has already been learned.

## Limitations & Future Work
- **Task specificity**: The analysis is restricted to LEGO state tracking. Although this is a standard theoretical benchmark, it remains distant from natural language reasoning.
- **Single-layer Transformer**: The analysis is limited to one layer. While the authors argue this represents the current theoretical frontier, conclusions may differ for multi-layer settings.
- **No positional encoding**: Practical LLMs employ PE schemes such as RoPE; whether theoretical insights derived under the NoPE assumption translate to practice remains an open question.
- **Curriculum learning assumption**: The two-stage curriculum (MLP first, then attention) differs from standard end-to-end training.
- **Restriction on symmetric groups**: Theorem 4.2 requires $|Y| < \log\log d$, leaving more general group structures unaddressed.

## Related Work & Insights
- **vs. Merrill et al. (ICLR 2024)**: They establish that CoT endows Transformers with $\mathsf{NC}^1$ (and even P) expressiveness, but do not address optimization. This paper is the first to provide learnability guarantees.
- **vs. Kim & Suzuki (ICLR 2025)**: They prove that Transformers with CoT can efficiently solve parity ($\mathsf{TC}^0$). This paper extends the complexity level to $\mathsf{NC}^1$.
- **vs. Wen et al.**: They analyze the benefit of CoT for sparse-dependency problems from a sample efficiency perspective. This paper focuses on length generalization, and the two lines of work are complementary.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First learnability and length generalization guarantee for CoT at the $\mathsf{NC}^1$ level; both the attention concentration framework and the theoretical treatment of recursive self-training are original contributions.
- Experimental Thoroughness: ⭐⭐⭐ — Experiments are adequate for a theory paper and validate the theoretical predictions, but are confined to synthetic tasks and the non-CoT baseline is somewhat underexplored.
- Writing Quality: ⭐⭐⭐⭐ — The paper is clear and rigorous with consistent notation, though reviewers noted that figures could be improved in readability.
- Value: ⭐⭐⭐⭐ — Deepens theoretical understanding of CoT reasoning; the insight linking attention concentration to length generalization carries practical implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unlabeled Data Can Provably Enhance In-Context Learning of Transformers](unlabeled_data_can_provably_enhance_in-context_learning_of_transformers.md)
- [\[NeurIPS 2025\] Large Language Models Can Learn and Generalize Steganographic Chain-of-Thought under Process Supervision](large_language_models_can_learn_and_generalize_steganographic_chain-of-thought_u.md)
- [\[NeurIPS 2025\] Exact Expressive Power of Transformers with Padding](exact_expressive_power_of_transformers_with_padding.md)
- [\[NeurIPS 2025\] I-RAVEN-X: Benchmarking Generalization and Robustness of Analogical and Mathematical Reasoning in Large Language and Reasoning Models](i-raven-x_benchmarking_generalization_and_robustness_of_analogical_and_mathemati.md)
- [\[NeurIPS 2025\] Latent Chain-of-Thought for Visual Reasoning](latent_chain-of-thought_for_visual_reasoning.md)

</div>

<!-- RELATED:END -->
