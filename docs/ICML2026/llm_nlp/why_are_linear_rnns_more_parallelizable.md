---
title: >-
  [Paper Note] Why Are Linear RNNs More Parallelizable?
description: >-
  [ICML 2026][LLM (Other)][Linear RNN] This paper uses circuit complexity to strictly explain why Linear RNNs are more easily parallelized like Transformers compared to traditional non-linear RNNs: LRNNs fall within arithmetic circuit classes of approximate log-depth, whereas non-linear RNNs can express harder-to-parallelize $\mathsf{logspace}$ / $\mathsf{p
tags:
  - ICML 2026
  - LLM (Other)
  - Linear RNN
date: 2026-05-08
content_hash: ffab9218a0bb2e24
---
# Why Are Linear RNNs More Parallelizable?

**Conference**: ICML2026  
**arXiv**: [2603.03612](https://arxiv.org/abs/2603.03612)  
**Code**: https://arg-git.informatik.uni-kl.de/pub/LinearRNN  
**Area**: LLM Efficiency / Sequence Model Theory / Parallel Computing  
**Keywords**: Linear RNN, Parallelization, Circuit Complexity, Expressivity, Long Context Architectures  

## TL;DR
This paper uses circuit complexity to strictly explain why Linear RNNs are more easily parallelized like Transformers compared to traditional non-linear RNNs: LRNNs fall within arithmetic circuit classes of approximate log-depth, whereas non-linear RNNs can express harder-to-parallelize $\mathsf{logspace}$ / $\mathsf{polynomial}$-time complete problems, forming a fundamental trade-off between expressivity and parallelizability.

## Background & Motivation
**Background**: Long-context LLM architectures are re-focusing on RNN and state-space / linear attention models. Linear RNN variants such as Mamba, RWKV, and DeltaNet aim to combine the length generalization of recurrent states with high parallel throughput similar to Transformers. Therefore, understanding "why linear recursion is easy to parallelize" is no longer just a theoretical question but directly relates to long-sequence model design.

**Limitations of Prior Work**: It is well-known that traditional RNNs update sequentially while Transformers can be parallelized; it is also known that certain LRNNs can be parallelized via scans. However, these are algorithmic intuitions that do not clearly answer two specific questions: first, whether non-linear RNNs face inevitable parallelization bottlenecks; second, whether the differences between LRNN variants are merely engineering details or involve strict hierarchies in expressivity.

**Key Challenge**: The more expressive a model is, the more it tends toward general sequential computation, making it harder to compress into shallow parallel circuits. Conversely, models that are easier to parallelize may sacrifice expressivity in certain algorithmic tasks. LRNNs sit in the middle ground: they are stronger than certain simple classes of Transformers but appear less difficult to parallelize than traditional non-linear RNNs.

**Goal**: The paper aims to establish a complexity map for RNNs/LRNNs: identifying which complexity classes non-linear RNNs can express, where the upper bound for LRNNs lies, and the fine-grained differences between linear update parameterizations like DPLR, PD, and Mamba.

**Key Insight**: The authors map the problem of neural network language recognition to circuit complexity and automata theory. Non-linear RNNs demonstrate sequential computation capabilities via counter machines / stack machines; LRNNs demonstrate parallelizability through matrix multiplication and arithmetic circuits; different LRNN parameterizations correspond to different Weighted Finite Automata (WFA) capabilities.

**Core Idea**: Linear state updates can be formulated as matrix products and sums, which can be simulated in parallel by log-depth arithmetic circuits. Non-linear recursions can simulate stronger sequential machines and thus cannot be parallelized with equal efficiency unless major collapses occur in complexity theory.

## Method
This paper does not propose a new model but rather provides a theoretical classification of the existing RNN family. The key is translating "ease of parallelization" into "simulability by shallow bounded fan-in circuits" and "expressivity" into "solving complete problems of specific complexity classes."

### Overall Architecture
The paper defines two major categories of sequence layers. Non-linear RNN state updates are $h_t=f(h_{t-1},x_t)$, where $f$ can include non-linearities like ReLU/MLPs. Linear RNN state updates are $S_t=A_t(x_t)S_{t-1}+b_t(x_t)$, representing a linear transformation of the previous state plus an input-dependent term at each step. Actual multi-layer models can stack recurrent sublayers and feedforward sublayers alternately, similar to Transformers.

The authors then introduce complexity classes: Transformers and simple LRNNs often fall near $\mathsf{TC}^0$ or $\mathsf{NC}^1$; the general upper bound for LRNNs is $\mathsf{PNC}^1$ (log-depth arithmetic circuits plus positivity check); non-linear RNNs can solve $\mathsf{L}$-complete problems under log precision and even $\mathsf{P}$-complete problems under polynomial precision. Finally, the paper validates theoretical predictions using two synthetic tasks: sorted deterministic graph connectivity and iterated $3\times3$ matrix multiplication.

### Key Designs

**1. Using circuit complexity to transform "parallelizability" into provable depth bounds**  
Previously, "RNNs are slow, Transformers are fast" was an engineering observation without a clear distinction between implementation lags and inherent bottlenecks. The core method of this paper maps a sequence layer's ability to "recognize a language" to standard complexity classes. If it can be simulated by bounded fan-in circuits of depth $O(\log n)$ or approximately $O(\log n \log^* n)$, it is inherently parallelizable like a Transformer. If it expresses a complete problem for a complexity class, then under standard conjectures (e.g., $\mathsf{PNC}^1 \neq \mathsf{L}$, $\mathsf{NC} \neq \mathsf{P}$), it cannot be compressed into such shallow circuits and must be more sequential. This provides a "coordinate system" for architecture comparison, elevating it from empirical experience to provable asymptotic differences in parallel depth. This is critical because at context lengths of 64K–1M, $\log n \approx 16$–$20$, whereas $\log^2 n$ can reach 256–400—theoretical depth differences translate directly into temporal differences on hardware.

**2. Expressivity lower bound of non-linear RNNs: Simulating stronger sequential machines makes them harder to parallelize**  
To explain why traditional RNNs are specifically hard to parallelize, the authors prove an expressivity lower bound for non-linear recursion: MLP RNNs with log precision can simulate counter machines, thereby solving the $\mathsf{L}$-complete task of sorted deterministic graph connectivity. With polynomial precision, they can simulate multi-stack machines to recognize $\mathsf{P}$-complete languages. The key insight is that non-linear recursion treats the recurrent state as sequential memory that can be read and written arbitrarily, which is the source of its algorithmic power. Conversely, "fully parallel simulation" of such inherently sequential computation requires deeper circuits (approximately $\Omega(\log^2 n)$ depth at log precision, an $O(\log n)$ factor more than Transformers) assuming $\mathsf{PNC}^1 \neq \mathsf{L}$. This solidifies the trade-off between expressivity and parallelizability.

**3. LRNNs are not monolithic: DPLR is strictly stronger than PD**  
The paper further deconstructs "Linear RNNs," noting that parameterization choices change the expressivity upper bound. General LRNN state updates $S_t=A_t S_{t-1}+b_t$ can be expanded into matrix products and sums, placing language recognition within $\mathsf{PNC}^1$. However, specific parameterizations determine how far they reach within this bound: diagonal-plus-low-rank (DPLR) variants like RWKV-7 and DeltaNet can express iterated $3\times3$ matrix multiplication, reaching $\mathsf{PNC}^1$-complete—the most expressive tier within the linear parallelizable range. In contrast, permutation-diagonal (PD) parameterizations maintain a permutation-diagonal structure in matrix products and are restricted to $\mathsf{NC}^1$ (though still capable of $\mathsf{NC}^1$-complete tasks). The authors pair each RNN class with an automata model: LRNNs correspond to Weighted Finite Automata (WFA), and PD corresponds to its deterministic version (DWFA). For architecture design, this serves as a "ruler": DPLR is more capable at expressing iterative algebraic computation than PD or Mamba/S4 while maintaining near-logarithmic parallel depth, making it an attractive midpoint.

### Loss & Training
The theoretical section involves no training loss. The experimental section uses synthetic algorithmic tasks for binary or step-by-step classification. All models use AdamW, `BCEWithLogitsLoss`, batch size 128, and gradient clipping 1.0, training for up to 60K steps. Baselines include non-linear RNNs, Transformers, Mamba, RWKV-7, and DeltaNet. Training lengths range in $[1,100]$, with testing including out-of-distribution (OOD) extrapolation for $[101,200]$ and $[201,300]$.

## Key Experimental Results

### Main Results
The primary result is the theoretical classification table. It identifies the "maximum expressivity" and "minimum parallel depth" for different model families rather than just benchmark scores.

| Model Category | Complexity Class | Parallel Depth Meaning | Representative Model/Task | Conclusion |
|----------|------------|--------------|---------------|------|
| Transformer / Simple LRNN | $\approx \mathsf{TC}^0 \subseteq \mathsf{NC}^1$ | $O(\log n)$ bounded fan-in depth | Transformer, Mamba-like simple structures | Easiest to parallelize, limited expressivity |
| General LRNN | $\mathsf{PNC}^1$ upper bound | $O(\log n \log^* n)$ simulation depth | Linear state update family | Small parallel overhead over Transformer |
| DPLR LRNN | $\mathsf{PNC}^1$-complete | Approaches LRNN upper bound | RWKV-7, DeltaNet | Strongest expressivity within linear parallel range |
| PD LRNN | $\mathsf{NC}^1$-complete | Log-depth | Permutation-diagonal LRNN | Stronger than simple finite state, weaker than DPLR |
| Log-precision non-linear RNN | $\mathsf{L}$-complete | Likely requires $\Omega(\log^2 n)$ depth | MLP RNN on graph connectivity | High expressivity, higher parallel cost |
| Poly-precision non-linear RNN | $\mathsf{P}$-complete | No polylog-depth parallelization under standard assumptions | MLP RNN simulating multi-stack machines | Strongest but most sequential |

Synthetic experiments validate these predictions. While all models learn In-Distribution (ID) tasks, their OOD behavior reveals architectural biases.

| Task | Theoretical Expectation | Strongest Models | Weaker Models | Observation |
|------|----------|--------------|--------------|------|
| Sorted deterministic graph connectivity | $\mathsf{L}$-complete, solvable by non-linear RNN, difficult for LRNN | non-linear RNN | Transformer, RWKV-7, Mamba, DeltaNet degrade on OOD | Only non-linear RNN achieves near-perfect OOD extrapolation |
| Iterated matrix multiplication over $\mathbb{Z}_m$ | DPLR LRNN and non-linear RNN should be stronger | RWKV-7, DeltaNet, non-linear RNN | Transformer, Mamba | DPLR and non-linear models are near-perfect ID; moderate OOD degradation |
| Iterated matrix multiplication over $\mathbb{Z}$ | Integer growth without modulus tests algebraic state | RWKV-7, DeltaNet, non-linear RNN | Transformer significantly degrades; Mamba below top models | DPLR's linear algebraic structure is ideal for matrix products |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| non-linear RNN on graph connectivity | OOD score remains near perfect | Matches $\mathsf{L}$-complete capability analysis |
| LRNN/Transformer on graph connectivity | Degradation increases with length | Theoretically incapable of covering sequential reachability |
| RWKV-7 / DeltaNet on IMM | Strong ID and OOD | DPLR can express $\mathsf{PNC}^1$-complete matrix products |
| Mamba on IMM | Significantly weaker than RWKV-7/DeltaNet | Simple linear parameterization lacks expressivity |
| Transformer on IMM | Unstable even during training, poor extrapolation | Shallow parallel advantage $\neq$ algebraic recursion capability |
| Unified training settings | AdamW, 60K steps, batch 128 | Ensures differences stem from architectural inductive bias |

### Key Findings
- The fundamental reason LRNNs are easier to parallelize is that linear recursion reduces to matrix products/scans, which have log-depth arithmetic circuit implementations.
- The "difficulty" in parallelizing non-linear RNNs is not an implementation failure but a result of their ability to simulate stronger sequential computation; this expressivity comes at a cost in circuit depth.
- DPLR is an attractive middle ground: it is more capable of expressing iterated algebraic computations than simple structures like Mamba/S4 while maintaining near-logarithmic parallel depth.
- Although experiments are small, they align with theoretical predictions, suggesting complexity results are not just abstract classifications but reflect training behavior on algorithmic tasks.

## Highlights & Insights
- The major contribution is elevating RNN architecture discussions from "empirical speed" to "complexity classes and complete problems." This provides a valuable theoretical map for long-context architecture design.
- It presents a clear trade-off: to obtain the sequential algorithmic power of a non-linear RNN, one must accept deeper parallel simulation; to achieve Transformer-like parallel efficiency, state updates must be restricted to linear/scannable forms.
- The distinction between DPLR and PD is insightful. While many papers group "Linear RNNs" together, this work shows that parameterization choices (low-rank terms vs. permutation-diagonal structures) change the expressivity ceiling.
- The synthetic tasks are well-chosen: graph connectivity isolates non-linear RNNs from LRNNs, while iterated matrix multiplication separates DPLR from simpler architectures.

## Limitations & Future Work
- Complexity analysis relies on formal assumptions like precision, uniformity, and bounded fan-in. Asymptotic parallel depth does not directly equate to GPU kernels, memory bandwidth, or training stability.
- Experiments are restricted to synthetic algorithmic tasks. While they validate expressivity tendencies, they do not directly prove superior performance in large-scale language modeling.
- The theory focuses on exact simulation and language recognition; real neural networks can perform approximate computation or use multi-layer hybrid structures to bypass the limitations of a single layer.
- While it is noted that non-linear RNN expressivity requires $\Theta(\log n)$ more parallel overhead, whether this cost is justified in real-world tasks remains an open question.

## Related Work & Insights
- **vs. Transformer Complexity**: Existing work places Transformers in the $\mathsf{TC}^0/\mathsf{NC}^1$ range; this paper places LRNNs in the adjacent $\mathsf{PNC}^1$, explaining their slight parallel overhead and additional expressivity.
- **vs. Mamba/S4 Theory**: Simple state-space/linear RNNs are often expressively limited; this work shows DPLR structures like RWKV-7 and DeltaNet reach higher complexity classes.
- **vs. Traditional RNN Theory**: Early ideas about RNNs simulating stack/counter machines are reused to explain the parallelization boundaries of LLM architectures.
- **vs. Parallelizing Non-linear RNNs**: Recent Newton-style methods can parallelize non-linear RNNs to $O(\log^2 n)$; these theoretical results show this is consistent with expectations for $\mathsf{logspace}$ complete problems.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Strong theoretical contribution systematically explaining LRNN parallelization and DPLR/PD expressivity.
- Experimental Thoroughness: ⭐⭐⭐☆☆ Experiments align well with theory but are restricted to synthetic validation.
- Writing Quality: ⭐⭐⭐⭐☆ Structure is clear, though complexity notation is dense for non-theoretical readers.
- Value: ⭐⭐⭐⭐☆ Highly instructive for choosing long-context LLM architectures, particularly for understanding why DPLR-style linear recursion is noteworthy.

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
