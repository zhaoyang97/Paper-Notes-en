---
title: >-
  [Paper Note] Exact Learning of Arithmetic with Differentiable Agents
description: >-
  [NeurIPS 2025 (MATH-AI Workshop)][differentiable FST] This paper proposes the Differentiable Finite-State Transducer (DFST), a Turing-complete and end-to-end differentiable model family that operates on a 2D symbol grid.…
tags:
  - "NeurIPS 2025 (MATH-AI Workshop)"
  - "differentiable FST"
  - "exact learning"
  - "arithmetic"
  - "length generalization"
  - "Turing-complete"
date: 2026-05-08
content_hash: 196c67394e59853e
---

# Exact Learning of Arithmetic with Differentiable Agents

**Conference**: NeurIPS 2025 (MATH-AI Workshop)
**arXiv**: [2511.22751](https://arxiv.org/abs/2511.22751)
**Code**: [GitHub](https://github.com/dngfra/differentiable-exact-algorithmic-learner)
**Area**: Neuro-Symbolic Computing / Length Generalization
**Keywords**: differentiable FST, exact learning, arithmetic, length generalization, Turing-complete

## TL;DR
This paper proposes the Differentiable Finite-State Transducer (DFST), a Turing-complete and end-to-end differentiable model family that operates on a 2D symbol grid. Trained via Policy-Trajectory Observations (PTOs) derived from expert arithmetic computations, DFST achieves perfect generalization to 3,850-digit binary addition and 2,450-digit decimal addition using only 20 training samples of up to 3-digit addition, with zero observed errors.

## Background & Motivation
**Background**: Length generalization is a central challenge in arithmetic reasoning. Transformers struggle to generalize precisely to longer inputs on simple arithmetic tasks (typically 2–3×), even with various techniques such as positional encoding improvements, scratchpads, and index hints, which yield high but imperfect exact-match accuracy.

**Limitations of Prior Work**: Existing universal computation model families either suffer from **computational time explosion** (unbounded precision in RNNs, growing context windows in Transformers) or **uninformative gradients** (non-differentiable adaptive computation time mechanisms). Although Turing-complete in theory, these models cannot be effectively trained via gradient-based methods in practice.

**Key Challenge**: Gold's theorem establishes that no algorithm can exactly learn the class of recursive functions from input–output samples alone. However, if intermediate computation steps are observable, the problem reduces from learning recursive functions to learning finite-state transducers—a tractable objective.

**Core Idea**: Design a model that simultaneously satisfies three conditions: (1) Turing-completeness, (2) constant-precision and constant-time output generation, and (3) end-to-end differentiability. The key insight is to decouple the differentiable controller from the external environment—DFST moves freely on a 2D grid and reads/writes symbols thereon.

## Method

### Overall Architecture
An expert Grid Agent (a hand-crafted arithmetic algorithm operating on a 2D grid) generates Policy-Trajectory Observations (PTOs)—observation sequences of the form (current symbol, action) = (written symbol, movement direction). The DFST is trained on next-action prediction via MSE loss and subsequently executes arithmetic operations autonomously on the grid.

### Key Designs

1. **Grid Agent and Symbol Universe**

    - **Function**: Defines the computational environment on an unbounded 2D grid $\mathbb{Z}^2$. The Grid Agent has a finite state set $Q$ and a transition function $\delta: Q \times \Sigma \to Q \times \Sigma \times \mathcal{M}$ (where $\mathcal{M} = \{U,D,L,R,S\}$).
    - **Mechanism**: At each step, the Grid Agent can only perceive the symbol at the current position, write one symbol, and move one cell. This is a natural planar generalization of the Turing machine and is therefore computationally universal.
    - **Design Motivation**: Simulates the process of a human teacher working through a calculation on a blackboard—the student observes chalk movements and writings but cannot directly access the teacher's internal state.

2. **Differentiable Finite-State Transducer (DFST)**

    - **Function**: Learns the Grid Agent's policy by parameterizing discrete state transitions as differentiable tensor operations.
    - **Mechanism**: DFST consists of three trainable tensors—$A \in \mathbb{R}^{|\Sigma| \times d \times d}$ (state transition), $B \in \mathbb{R}^{|\Sigma| \times |\Sigma| \times d}$ (symbol output), $C \in \mathbb{R}^{|\Sigma| \times |\mathcal{M}| \times d}$ (movement output)—plus an initial hidden state $h_0$. Update rules: $h_{t+1} = A(x_t) h_t$, $\hat{s}_t = B(x_t) h_t$, $\hat{m}_t = C(x_t) h_t$. Outputs are discretized via argmax.
    - **Universality Theorem**: Any Grid Agent with $d$ states can be exactly simulated by a DFST of dimension $d$ (by one-hot encoding the state transitions into the tensors).
    - **Design Motivation**: The linear structure enables parallel training in $O(\log T)$ via Blelloch scan, with no nonlinearities, yielding a simpler loss landscape.

3. **Policy-Trajectory Observations (PTOs)**

    - **Function**: Replaces pure input–output supervision with training data that includes intermediate computation steps.
    - **Mechanism**: Each training sample is a complete computation trajectory $(x_t, s_t, m_t)_{t=0}^T$—the currently observed symbol, the written symbol, and the movement direction. This is more structured than scratchpads, as it directly corresponds to the execution policy.
    - **Design Motivation**: Gold's theorem precludes exact learning of recursive functions from I/O samples; learning from PTOs reduces the problem to learning an FST, which is theoretically guaranteed to be feasible (Papazov et al.).

### Loss & Training
- **Loss**: MSE on next-action prediction (symbol token + movement token).
- **Optimizer**: Adam with cosine annealing, no warmup.
- **Initialization**: State transition matrices $A$ initialized as the identity matrix; $B$ and $C$ initialized to zero.
- **Precision**: float32.
- **Training duration**: 500K iterations for addition; 3M iterations for multiplication.

## Key Experimental Results

### Main Results (DFST vs. Neural GPU)

| Task | DFST Samples | DFST Parameters | Max Training Digits | Robust LG | Prob. LG | Neural GPU Samples |
|------|-------------|----------------|--------------------|-----------|---------|--------------------|
| add2 | 20 | 1,020 | 3 | ≥3850 | ≥3850 | ~200K |
| add10 | 225 | 8,900 | 3 | ≥2450 | ≥2450 | N/A |
| mult2 | 750 | 5,280 | 5 | ≥600 | ≥600 | ~200K |
| mult10 | 10,000 | 162,108 | 5 | ≥180 | ≥180 | N/A |

Note: All "≥" values indicate that longer inputs could not be tested due to GPU memory constraints; no computational errors were observed.

### Key Findings
- DFST learns perfect binary addition from 20 samples (up to 3 digits) and generalizes to 3,850 digits (1,283×)—a record-breaking degree of length generalization.
- Compared to Neural GPU: Neural GPU requires ~200K samples, is highly sensitive to random seeds, and fails on symmetric digit inputs; DFST performs perfectly on symmetric inputs as well.
- DFST training requires only a simple cosine scheduler, whereas Neural GPU demands curriculum learning, gradient noise injection, and relaxation-pull mechanisms.
- Longer training continuously improves Robust Length Generalization (Figures 2–5), with no saturation observed.
- Exact learning is itself undecidable in general (equivalent to deciding equivalence of two Turing machines); confidence can only be established through randomized testing.

## Highlights & Insights
- **The power of intermediate supervision**: The critical shift from "impossible" (Gold's theorem) to "readily achievable" lies in the form of training data—observing the computation process versus observing only results. This has profound implications for LLM chain-of-thought training: structured intermediate steps may be far more valuable than additional I/O samples.
- **Environment–controller decoupling**: The core innovation of DFST lies not in the model architecture per se, but in enabling the model to move freely within an external environment. This breaks the constraint of sequential models to unidirectional movement and is the key to achieving both Turing-completeness and differentiability.
- **The power of minimalist design**: Pure linear operations combined with a simple training strategy suffice for exact algorithm learning, with no need for elaborate techniques.

## Limitations & Future Work
- The Grid Agent must be hand-designed (i.e., expert algorithms are required); the problem of discovering algorithms from scratch remains unsolved.
- The hidden dimension of DFST must match the number of states of the target Grid Agent, requiring prior knowledge.
- The generalization ratio for multiplication (36× for mult2, 36× for mult10) falls far short of that for addition (1,283×), possibly due to the greater complexity of the multiplication algorithm's state space.
- Evaluation is limited to arithmetic tasks; generalizability to other algorithmic tasks such as sorting and search remains unknown.
- Verification of exact learning is itself undecidable—it is impossible to formally prove that the model is correct on all inputs.

## Related Work & Insights
- **vs. Transformer + scratchpad**: Transformers achieve only 2–3× length generalization with imperfect accuracy; DFST achieves 1,000×+ perfect generalization. The key differences lie in computational universality and the form of training data.
- **vs. Neural GPU**: Neural GPU is the closest prior work and can achieve length generalization on binary arithmetic, but is highly seed-dependent and fails on hard cases; DFST does not exhibit these failure modes.
- **Implications for LLMs**: Chain-of-thought is essentially a form of PTO—exposing the model to intermediate reasoning steps. However, Transformers are not Turing-complete under finite precision, which may be a fundamental upper bound on their arithmetic generalization capacity.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The design philosophy of DFST (Turing-complete + differentiable + environment decoupling) is highly elegant with a solid theoretical foundation.
- **Experimental Thoroughness**: ⭐⭐⭐ Limited to arithmetic tasks, but results within that scope are remarkable.
- **Writing Quality**: ⭐⭐⭐⭐ The formal framework is clearly presented, and comparisons with prior work are thorough.
- **Value**: ⭐⭐⭐⭐ Significant implications for both the theory and practice of exact algorithm learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Scalable GPU-Accelerated Euler Characteristic Curves: Optimization and Differentiable Learning for PyTorch](scalable_gpu-accelerated_euler_characteristic_curves_optimization_and_differenti.md)
- [\[NeurIPS 2025\] A Differentiable Model of Supply-Chain Shocks](a_differentiable_model_of_supply-chain_shocks.md)
- [\[AAAI 2026\] Whispering Agents: An Event-Driven Covert Communication Protocol for the Internet of Agents](../../AAAI2026/others/whispering_agents_an_event-driven_covert_communication_protocol_for_the_internet.md)
- [\[AAAI 2026\] Expandable and Differentiable Dual Memories with Orthogonal Regularization for Exemplar-free Continual Learning](../../AAAI2026/others/expandable_and_differentiable_dual_memories_with_orthogonal_regularization_for_e.md)
- [\[CVPR 2026\] DiffBMP: Differentiable Rendering with Bitmap Primitives](../../CVPR2026/others/diffbmp_differentiable_rendering_with_bitmap_primitives.md)

</div>

<!-- RELATED:END -->
