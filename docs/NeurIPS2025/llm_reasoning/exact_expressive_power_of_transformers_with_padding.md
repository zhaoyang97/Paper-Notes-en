---
title: >-
  [Paper Note] Exact Expressive Power of Transformers with Padding
description: >-
  [NeurIPS 2025][LLM Reasoning][Transformer expressive power] This paper provides an exact characterization of the expressive power of Transformers with padding: fixed depth combined with polynomial padding is precisely equivalent to $\mathsf{FO}$-uniform $\mathsf{TC}^0$; further combined with $O(\log^d n)$ looping, this is precisely equivalent to $\mathsf{FO}$-uniform $\mathsf{TC}^d$; and polylog looping converges to $\mathsf{NC}$. These results establish a complete theoretical foundation for padding and looping as parallel inference-time computation mechanisms.
tags:
  - NeurIPS 2025
  - LLM Reasoning
  - Transformer expressive power
  - padding
  - looping
  - circuit complexity
  - TC0
date: 2026-05-08
content_hash: 445b8970ab8bd09f
---

# Exact Expressive Power of Transformers with Padding

**Conference**: NeurIPS 2025
**arXiv**: [2505.18948](https://arxiv.org/abs/2505.18948)
**Code**: None
**Area**: LLM Reasoning
**Keywords**: Transformer expressive power, padding, looping, circuit complexity, TC0

## TL;DR
This paper provides an exact characterization of the expressive power of Transformers with padding: fixed depth combined with polynomial padding is precisely equivalent to $\mathsf{FO}$-uniform $\mathsf{TC}^0$; further combined with $O(\log^d n)$ looping, this is precisely equivalent to $\mathsf{FO}$-uniform $\mathsf{TC}^d$; and polylog looping converges to $\mathsf{NC}$. These results establish a complete theoretical foundation for padding and looping as parallel inference-time computation mechanisms.

## Background & Motivation

**Background**: Transformers have inherent computational limitations — fixed-depth Transformers are confined to $\mathsf{TC}^0$, the class of highly parallelizable problems. Chain-of-Thought (CoT) reasoning can extend beyond $\mathsf{TC}^0$, but at the cost of sequential decoding, sacrificing parallelism.

**Limitations of Prior Work**: Although CoT is expressively powerful, it is slow at inference time. The question remains whether there exist inference-time computation methods that extend the expressive power of Transformers without increasing parameters and while preserving parallelism.

**Key Challenge**: It was previously known that padded Transformers have an upper bound of $\mathsf{TC}^0$, but whether the entire class $\mathsf{TC}^0$ is achievable (i.e., whether the lower bound matches) remained an open problem.

**Goal**: (a) Precisely characterize the expressive power of padded Transformers; (b) Understand how the combination of padding and looping systematically extends Transformer capabilities.

**Key Insight**: Rather than following the standard $\mathsf{FO}[\mathsf{M}, \mathsf{bit}]$ route (where the `bit` predicate is difficult to simulate in Transformers), this work employs $\mathsf{FO}+\mathsf{M}^2$ — first-order logic with pairwise majority quantifiers — which is equivalent to $\mathsf{TC}^0$ but does not require `bit`.

**Core Idea**: A set of $n^k$ padding tokens provides the Transformer with sufficient "storage space" to enumerate all $n^k$ assignments of $k$ variables, thereby enabling the resolution of arbitrary $\mathsf{FO}+\mathsf{M}^2$ formulas.

## Method

### Overall Architecture
This is a purely theoretical work. The central contributions are two exact characterization theorems:
- Theorem 1: $\mathsf{AHAT}^0_* = \mathsf{FO}\text{-uniform } \mathsf{TC}^0$ (fixed depth + polynomial padding)
- Theorem 2: $\mathsf{AHAT}^d_* = \mathsf{FO}\text{-uniform } \mathsf{TC}^d$ ($O(\log^d n)$ looping + polynomial padding)

The computational model is the Averaging-Hard-Attention Transformer (AHAT): hard attention (attending only to positions achieving the maximum attention score), masked pre-norm, and causal masking.

### Key Designs

1. **Padding → Variable Enumeration Storage** (Lemma 2):

    - **Function**: Proves that $\mathsf{FO}+\mathsf{M}^2$ formulas with $k$ distinct variables and nesting depth $\ell$ are computable in $\mathsf{uAHAT}^0_k$.
    - **Mechanism**: The $n^k$ padding tokens enumerate all $n^k$ assignments of $k$ variables. Each token $v$ stores the truth values of subformulas under assignment $v$. By induction, each layer evaluates one level of the formula:
        - Constants/variables: positions are retrieved via the layer-norm hash $\phi(z) = \langle z,1,-z,-1\rangle / \sqrt{2z^2+2}$
        - Existential quantifier $\exists i. P$: token $v$ attends to all $v'$ agreeing with $v$ except on variable $i$; the mean yields $c/n$, compared against $1/(2n)$
        - Pairwise majority quantifier $\mathsf{M}^2(i,j).P$: similarly, attends to all $v'$ agreeing on all variables except $i,j$, yielding $c/n^2$, compared against $1/2$
    - **Design Motivation**: Since $\mathsf{FO}+\mathsf{M}^2$ is equivalent to $\mathsf{FO}\text{-uniform } \mathsf{TC}^0$, demonstrating that padding can simulate all operators of this logic establishes the lower bound.

2. **Reduction Framework for Transformer Analysis** (Lemma 3):

    - **Function**: Introduces the tools of reductions and complete problems from classical complexity theory into the analysis of Transformers.
    - **Mechanism**: If a class $\mathsf{C}$ has a complete problem $L$ under $\mathsf{R}$-reductions, and padded Transformers can recognize $L$ and compute all $\mathsf{R}$-reductions, then $\mathsf{C} \subseteq \mathsf{AHAT}^d_*$.
    - **Design Motivation**: This establishes a modular proof strategy — it suffices to show that a Transformer can solve one complete problem and implement the associated reductions to obtain recognizability for the entire class.

3. **Looping for Depth Extension** (Lemma 5):

    - **Function**: Proves that $O(\log^d n)$ looping combined with polynomial padding suffices to recognize all of $\mathsf{FO}\text{-uniform } \mathsf{TC}^d$.
    - **Mechanism**: Graph connectivity is $\mathsf{NL}$-complete under $\mathsf{FO}$ reductions, and log-looped padded Transformers can solve graph connectivity (a known result). This implies that padded Transformers can implement $\mathsf{L}$-reductions. The completeness of wide-$\mathsf{TC}^d$ circuit evaluation under $\mathsf{L}$-reductions then enables a chain of derivations.
    - **Design Motivation**: Formalizes the intuition that padding controls width while looping controls depth.

4. **Uniformity Collapse Theorem** (Theorem 3):

    - **Function**: Proves that for $d \geq 1$, $\mathsf{FO}\text{-uniform } \mathsf{TC}^d = \mathsf{L}\text{-uniform } \mathsf{TC}^d$.
    - **Mechanism**: Since $\mathsf{L}$ itself lies within $\mathsf{FO}\text{-uniform } \mathsf{AC}^d$, the $\mathsf{L}$-computation used to construct circuits can be simulated by $\mathsf{FO}$-uniform circuits and composed with the evaluation circuit.
    - **Design Motivation**: A new circuit complexity result obtained as a by-product, which strengthens Theorem 2.

### Loss & Training
Not applicable — this is a purely theoretical work with no training component.

## Key Experimental Results

### Main Results
This paper contains no experiments. The central results are mathematical theorems:

| Theorem | Model | Equivalent Complexity Class |
|---------|-------|-----------------------------|
| Thm 1 | Fixed depth + poly padding | $\mathsf{FO}$-uniform $\mathsf{TC}^0$ |
| Thm 2 | $O(\log^d n)$ loop + poly padding | $\mathsf{FO}$-uniform $\mathsf{TC}^d$ |
| Cor 2.1 | Polylog loop + poly padding | $\mathsf{FO}$-uniform $\mathsf{NC}$ |
| Thm 3 | Circuit complexity ($d \geq 1$) | $\mathsf{FO}$-uniform $\mathsf{TC}^d = \mathsf{L}$-uniform $\mathsf{TC}^d$ |

### Key Comparisons

| Inference-Time Computation | Expressive Power | Parallelism | Parameter Increase |
|---------------------------|-----------------|-------------|-------------------|
| None | $\mathsf{TC}^0$ | Fully parallel | None |
| Poly padding | $\mathsf{TC}^0$ (fills the entire class) | Fully parallel | None |
| Padding + log loop | $\mathsf{TC}^1 \supseteq \mathsf{NL}$ | Highly parallel | None |
| Padding + polylog loop | $\mathsf{NC}$ | Highly parallel | None |
| Chain of Thought | $\supseteq \mathsf{P}$ | Sequential | None |

### Key Findings
- Padding alone does not raise the complexity-class upper bound of Transformers (which remains $\mathsf{TC}^0$), but enables them to fill the entire class — previously only a proper subset was known to be reachable.
- Logarithmic precision and polynomial precision are equivalent in the padded/looped setting.
- Causal masking and unmasked Transformers are equivalent under padding (Proposition 1).
- Padding combined with polylog looping reaches $\mathsf{NC}$, which is the theoretical upper bound for parallelism-preserving computation (unless $\mathsf{NC} = \mathsf{P}$).

## Highlights & Insights
- **First exact characterization of the expressive power of padded Transformers**, resolving an open problem. The key technical insight is to bypass the hard-to-simulate `bit` predicate by instead working with the equivalent $\mathsf{FO}+\mathsf{M}^2$ logic.
- **Introducing reduction and completeness tools into Transformer theory** is an elegant methodological contribution: one complete problem plus reduction capability yields the entire complexity class. This methodology can be further applied to analyze new Transformer variants.
- **The dual perspective of padding as "computational width" and looping as "computational depth"** provides a clear conceptual framework for designing inference-time computation strategies.
- **The uniformity collapse theorem** is a novel circuit complexity result with independent value.

## Limitations & Future Work
- The AHAT model (hard attention + masked pre-norm) is an idealization of standard soft-attention Transformers; the expressive power of practical models may differ.
- While the theory shows that padding is effective, how to train models to "learn to exploit padding" in practice remains an open question (existing empirical results are mixed).
- The trainability of looping in practice is questionable — appropriate parameterization and training strategies have yet to be resolved.
- The memory overhead arising from the increased context length induced by padding is not discussed.
- A fine-grained characterization of $\mathsf{AHAT}^d_k$ for fixed $k$ (i.e., with bounded padding) remains open.

## Related Work & Insights
- **vs. Merrill & Sabharwal (2023)**: The prior work established the upper bound $\mathsf{AHAT} \subseteq \mathsf{TC}^0$; this paper completes the picture by establishing the matching lower bound.
- **vs. Pfau (2024)**: That work posed the open question of padded Transformers vs. $\mathsf{TC}^0$; this paper provides a complete answer.
- **vs. CoT theory**: CoT can reach beyond $\mathsf{P}$ but sacrifices parallelism; padding combined with looping achieves $\mathsf{NC}$ while preserving parallelism.
- **vs. Merrill (2025)**: The result on looped Transformers solving graph connectivity is used as a key lemma in this work.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Resolves an important open problem; introducing reduction/completeness tools into Transformer theory represents a methodological innovation.
- Experimental Thoroughness: ⭐⭐⭐ Purely theoretical with no experiments, but the theorem system is complete and self-consistent.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain is clear, theorem statements are precise, and proof intuitions are well articulated.
- Value: ⭐⭐⭐⭐ Carries significant implications for understanding the computational power of Transformers and for designing inference-time computation strategies.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Transformers Provably Learn Chain-of-Thought Reasoning with Length Generalization](transformers_provably_learn_chain-of-thought_reasoning_with_length_generalizatio.md)
- [\[NeurIPS 2025\] Unlabeled Data Can Provably Enhance In-Context Learning of Transformers](unlabeled_data_can_provably_enhance_in-context_learning_of_transformers.md)
- [\[NeurIPS 2025\] SafePath: Preventing Harmful Reasoning in Chain-of-Thought via Early Alignment](safepath_preventing_harmful_reasoning_in_chain-of-thought_via_early_alignment.md)
- [\[NeurIPS 2025\] Unlocking Multimodal Mathematical Reasoning via Process Reward Model](unlocking_multimodal_mathematical_reasoning_via_process_reward_model.md)
- [\[NeurIPS 2025\] Let LRMs Break Free from Overthinking via Self-Braking Tuning](let_lrms_break_free_from_overthinking_via_self-braking_tuning.md)

<!-- RELATED:END -->
