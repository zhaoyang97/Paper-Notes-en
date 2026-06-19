---
title: >-
  [Paper Note] Why Are Linear RNNs More Parallelizable?
description: >-
  [ICML 2026][LLM (Other)][Linear RNN] This paper provides a rigorous explanation through circuit complexity as to why Linear RNNs are more parallelizable like Transformers compared to traditional non-linear RNNs: LRNNs fall within arithmetic circuit classes with approximate log-depth, whereas non-linear RNNs can express harder-to-parallelize logspace / pol
tags:
  - ICML 2026
  - LLM (Other)
  - Linear RNN
date: 2026-05-08
content_hash: 7f0217bdb0da3725
---
# Why Are Linear RNNs More Parallelizable?

**Conference**: ICML2026  
**arXiv**: [2603.03612](https://arxiv.org/abs/2603.03612)  
**Code**: https://arg-git.informatik.uni-kl.de/pub/LinearRNN  
**Area**: LLM Efficiency / Theory of Sequence Models / Parallel Computing  
**Keywords**: Linear RNN, Parallelization, Circuit Complexity, Expressivity, Long Context Architectures  

## TL;DR
This paper provides a rigorous explanation through circuit complexity as to why Linear RNNs are more parallelizable like Transformers compared to traditional non-linear RNNs: LRNNs fall within arithmetic circuit classes with approximate log-depth, whereas non-linear RNNs can express harder-to-parallelize logspace / polynomial-time complete problems, forming a fundamental trade-off between expressivity and parallelizability.

## Background & Motivation
**Background**: Long-context LLM architectures are refocusing on RNNs and state-space/linear attention-style models. Linear RNN variants like Mamba, RWKV, and DeltaNet aim to combine the length generalization of recurrent states with high parallel throughput similar to Transformers. Thus, understanding why linear recurrence is easy to parallelize is no longer just a theoretical question but directly impacts long-sequence model design.

**Limitations of Prior Work**: It is widely recognized that traditional RNNs are sequential while Transformers can be parallelized, and that certain LRNNs can be parallelized via scan algorithms. However, these are algorithmic intuitions. Two specific questions remain unanswered: first, whether non-linear RNNs face unavoidable parallelization barriers; second, whether differences between LRNN variants are merely engineering implementations or involve rigorous expressivity hierarchies.

**Key Challenge**: Stronger model expressivity often leads to behavior resembling general sequential computation, making it harder to compress into shallow parallel circuits. Conversely, models that are easier to parallelize might sacrifice expressivity for certain algorithmic tasks. LRNNs sit in the middle ground: they are stronger than certain simple Transformer categories but seemingly less difficult to parallelize than traditional non-linear RNNs.

**Goal**: Establish a complexity map for RNNs/LRNNs: identifying which complexity classes non-linear RNNs can express, where the LRNN upper bound lies, and fine-grained differences between linear update parameterizations like DPLR, PD, and Mamba.

**Key Insight**: The problem of language recognition by neural networks is mapped to circuit complexity and automata theory. Non-linear RNNs demonstrate sequential computation capabilities through counter/stack machines; LRNNs demonstrate parallelizability through matrix multiplication and arithmetic circuits; different LRNN parameterizations correspond to different Weighted Finite Automata (WFA) capabilities.

**Core Idea**: Linear state updates can be expressed as matrix products and sums, thus allowing parallel simulation by log-depth arithmetic circuits. Non-linear recurrences can simulate stronger sequential machines, making them impossible to parallelize as efficiently under standard complexity-theoretic assumptions.

## Method
This paper does not propose a new model but categorizes the existing RNN family theoretically. The key is transforming "parallelizability" into "simulability by shallow bounded fan-in circuits" and "expressivity" into "completeness for specific complexity classes."

### Overall Architecture
The paper defines two major sequence layers. Non-linear RNN updates are $h_t=f(h_{t-1},x_t)$, where $f$ can include non-linearities like MLP/ReLU. Linear RNN updates are $S_t=A_t(x_t)S_{t-1}+b_t(x_t)$, where each step performs a linear transformation on the previous state plus an input-dependent term. Multi-layer models can interleave recurrent sublayers and feedforward sublayers similarly to Transformers.

Subsequently, complexity classes are introduced: Transformers and simple LRNNs usually fall near $\mathsf{TC}^0$ or $\mathsf{NC}^1$. The general upper bound for LRNNs is $\mathsf{PNC}^1$ (log-depth arithmetic circuits with positivity checks). Non-linear RNNs can solve $\mathsf{L}$-complete problems with log precision and $\mathsf{P}$-complete problems with polynomial precision. Finally, theoretical predictions are validated on two synthetic tasks: sorted deterministic graph connectivity and iterated $3\times3$ matrix multiplication.

### Key Designs

**1. Mapping parallelizability to provable depth bounds using circuit complexity**
Previous engineering experience showed RNNs are slow and Transformers are fast, but couldn't distinguish between implementation gaps and inherent barriers. The core method maps a layer's ability to "recognize a language" to standard complexity classes: if it can be simulated by bounded fan-in circuits of $O(\log n)$ or $O(\log n \log^* n)$ depth, it is naturally parallelizable like Transformers. If it expresses a complete problem for a class like $\mathsf{L}$ or $\mathsf{P}$, it cannot be compressed into shallow circuits under standard conjectures (e.g., $\mathsf{PNC}^1 \neq \mathsf{L}$), making it inherently sequential. This "coordinate system" elevates architectural comparison to provable asymptotic depth differences. This is crucial because for context lengths of 64K–1M, $\log n \approx 16$–$20$ while $\log^2 n$ can reach 256–400—theoretical depth differences translate directly into sequential time differences on hardware.

**2. Expressivity lower bound of non-linear RNNs: Difficulty in parallelization as a result of strength**
To explain why traditional RNNs are hard to parallelize, the authors prove an expressivity lower bound: MLP RNNs with log precision can simulate counter machines, solving the sorted deterministic graph connectivity task, which is $\mathsf{L}$-complete. With polynomial precision, they can even simulate multi-stack machines to recognize $\mathsf{P}$-complete languages. The key insight is that non-linear recurrence treats the state as a sequential memory with arbitrary read-write-update capabilities, providing algorithmic power but imposing a depth cost for parallel simulation (approx. $\Omega(\log^2 n)$ depth in log precision), adding an $O(\log n)$ factor over Transformers. Expressivity and parallelizability are thus fundamentally traded off.

**3. Fine-grained LRNN hierarchy: DPLR is strictly stronger than PD**
The authors distinguish between "Linear RNN" parameterizations, noting that specific choices alter expressivity upper bounds. General LRNN updates $S_t = A_t S_{t-1} + b_t$ expand into matrix products and sums, falling into $\mathsf{PNC}^1$. However, diagonal-plus-low-rank (DPLR) variants like RWKV-7 and DeltaNet can express iterated $3\times3$ matrix multiplication, reaching $\mathsf{PNC}^1$-completeness—the highest expressivity within the linearly parallelizable range. Permutation-diagonal (PD) parameterizations are restricted to $\mathsf{NC}^1$. Each RNN type is matched with an automaton model: LRNNs to Weighted Finite Automata (WFA) and PDs to Deterministic WFA (DWFA). For architectural design, DPLR is an attractive midpoint, offering better algebraic computation than PD or Mamba while maintaining near-logarithmic parallel depth.

### Loss & Training
Theoretical sections involve no training. Synthetic tasks use binary or step-wise classification. Models are trained with AdamW, BCEWithLogitsLoss, batch size 128, and gradient clipping at 1.0 for up to 60K steps. Comparison models include non-linear RNNs, Transformer, Mamba, RWKV-7, and DeltaNet. Training lengths are in range $[1,100]$, with testing covering extra-long sequences $[101,200]$ and $[201,300]$.

## Key Experimental Results

### Main Results
The main finding is the theoretical classification table. It determines the "maximum expressivity" and "minimum parallel depth" for different model families.

| Model Category | Complexity Class | Parallel Depth Implication | Representative Model/Task | Conclusion |
|----------|------------|--------------|---------------|------|
| Transformer / Simple LRNN | $\approx \mathsf{TC}^0 \subseteq \mathsf{NC}^1$ | $O(\log n)$ bounded fan-in depth | Transformer, Mamba-style structures | Easiest to parallelize, limited expressivity |
| General LRNN | $\mathsf{PNC}^1$ Upper Bound | $O(\log n \log^* n)$ depth simulation | Linear state update family | Negligible parallel overhead over Transformer |
| DPLR LRNN | $\mathsf{PNC}^1$-complete | Near LRNN upper bound | RWKV-7, DeltaNet | Strongest expressivity within linearly parallelizable range |
| PD LRNN | $\mathsf{NC}^1$-complete | log-depth | Permutation-diagonal LRNN | Stronger than simple finite states, weaker than DPLR |
| log-precision nonlinear RNN | $\mathsf{L}$-complete | Potential $\Omega(\log^2 n)$ depth | MLP RNN solving graph connectivity | Strong expressivity, higher parallel cost |
| poly-precision nonlinear RNN | $\mathsf{P}$-complete | Non-parallelizable in polylog-depth (standard assumptions) | MLP RNN simulating multi-stack machines | Most expressive but most sequential |

Synthetic experiments validate these predictions. While all models learn ID (In-Distribution) data, OOD (Out-of-Distribution) length extrapolation results are summarized below:

| Task | Theoretical Expectation | Strongest Model | Weaker Models | Observation |
|------|----------|--------------|--------------|------|
| Sorted deterministic graph connectivity | $\mathsf{L}$-complete; Non-linear RNN should solve it, LRNNs struggle | nonlinear RNN | Transformer, RWKV-7, Mamba, DeltaNet (degrade on OOD) | Only nonlinear RNN achieves near-perfect length extrapolation |
| Iterated matrix multiplication over $\mathbb{Z}_m$ | DPLR LRNN and nonlinear RNN should be stronger | RWKV-7, DeltaNet, nonlinear RNN | Transformer, Mamba | DPLR and non-linear models show only moderate OOD degradation |
| Iterated matrix multiplication over $\mathbb{Z}$ | Algebraic state growth; tests algebraic recursion | RWKV-7, DeltaNet, nonlinear RNN | Transformer significantly degrades; Mamba below top models | DPLR linear-algebraic structure excels at matrix products |

### Ablation Study

| Configuration | Key Metrics | Explanation |
|------|---------|------|
| nonlinear RNN on graph connectivity | OOD length score remains high | Consistent with $\mathsf{L}$-complete capability analysis |
| LRNN/Transformer on graph connectivity | Higher degradation with length | Theoretically difficult to cover sequential reachability issues |
| RWKV-7 / DeltaNet on IMM | Strong ID and OOD performance | DPLR expresses $\mathsf{PNC}^1$-complete matrix products |
| Mamba on IMM | Significantly weaker than DPLR models | Simple linear parameterization lacks expressivity |
| Transformer on IMM | Unstable in training; poor length extrapolation | Shared shallow parallel advantage $\neq$ algebraic recurrence capability |
| Unified training setup | AdamW, 60K steps, batch 128 | Differences stem primarily from architectural inductive bias |

### Key Findings
- The fundamental reason LRNNs are more parallelizable is that linear recurrence reduces to matrix products/scans, which have log-depth arithmetic circuit implementations.
- The parallelization difficulty of non-linear RNNs is not due to poor implementation but because they can simulate stronger sequential computation models; this expressivity yields a depth cost in the complexity sense.
- DPLR is a compelling midpoint: it is more expressive than simple Mamba/S4 structures for iterated algebraic computation while maintaining parallel depth close to Transformers.
- Synthetic experiments reflect theoretical predictions, showing that complexity results govern the length generalization behavior of trainable models.

## Highlights & Insights
- The strongest contribution is elevating the RNN debate from "empirically fast/slow" to "complexity classes and complete problems," providing a theoretical coordinate system for architectural design.
- It clarifies the trade-off: to obtain the sequential algorithmic power of non-linear RNNs, one must accept deeper parallel simulation; for Transformer-like efficiency, state updates must be restricted to linear/scannable forms.
- The distinction between DPLR and PD is insightful, showing that parameterization choices like low-rank terms or permutation structures significantly shift expressivity bounds.
- Synthetic tasks are well-aligned with theory: graph connectivity separates non-linear and linear models, while iterated matrix multiplication separates DPLR from simpler architectures.

## Limitations & Future Work
- Complexity analysis depends on formal assumptions (precision, uniformity, bounded fan-in). Asymptotic parallel depth does not directly equate to GPU kernel performance or memory bandwidth.
- Experiments are restricted to synthetic algorithmic tasks; while they validate expressivity, they do not directly prove superior large-scale language modeling.
- Theoretical analysis focuses on exact simulation; real networks might use approximate computation or hybrid structures to bypass single-layer limitations.
- While non-linear RNNs offer extra expressivity at a $\Theta(\log n)$ parallel cost, whether this trade-off is worthwhile for real-world tasks remains an open question.

## Related Work & Insights
- **vs Transformer Complexity**: Existing work places Transformers in the $\mathsf{TC}^0/\mathsf{NC}^1$ range; this work places LRNNs in the adjacent $\mathsf{PNC}^1$, explaining their extra expressivity with minimal parallel overhead.
- **vs Mamba/S4 Theory**: Simple linear RNNs are often structurally limited; this work highlights how DPLR structures like RWKV-7 and DeltaNet reach higher complexity classes.
- **vs Classical RNN Theory**: Re-applies early concepts (simulating stack/counter machines) to explain the parallelization boundaries of modern LLM architectures.
- **vs Parallelizing Non-linear RNNs**: Matches recent Newton-style methods that parallelize non-linear RNNs to $O(\log^2 n)$ depth, consistent with the $\mathsf{L}$-complete expectation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Systematically explains LRNN parallelizability via complexity theory and distinguishes DPLR/PD expressivity.
- Experimental Thoroughness: ⭐⭐⭐☆☆ Experiments align with theory but are restricted to synthetic validations.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure, though notation-dense for non-theory readers.
- Value: ⭐⭐⭐⭐☆ Highly instructive for long-context architecture choice, highlighting the importance of DPLR-style linear recurrence.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Language Models, Graph Searching, and Supervision Adulteration: When More Supervision is Less and How to Make More More](../../ACL2025/llm_nlp/lm_graph_search_supervision.md)
- [\[NeurIPS 2025\] Composing Linear Layers from Irreducibles](../../NeurIPS2025/llm_nlp/composing_linear_layers_from_irreducibles.md)
- [\[ICLR 2026\] Fine-Grained Activation Steering: Steering Less, Achieving More](../../ICLR2026/llm_nlp/fine-grained_activation_steering_steering_less_achieving_more.md)
- [\[ICLR 2026\] Weight Decay may matter more than μP for Learning Rate Transfer in Practice](../../ICLR2026/llm_nlp/weight_decay_may_matter_more_than_mup_for_learning_rate_transfer_in_practice.md)
- [\[NeurIPS 2025\] Linear Transformers Implicitly Discover Unified Numerical Algorithms](../../NeurIPS2025/llm_nlp/linear_transformers_implicitly_discover_unified_numerical_algorithms.md)

</div>

<!-- RELATED:END -->
