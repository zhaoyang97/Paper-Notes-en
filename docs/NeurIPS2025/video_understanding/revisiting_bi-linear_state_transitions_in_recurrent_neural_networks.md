---
title: >-
  [Paper Note] Revisiting Bi-Linear State Transitions in Recurrent Neural Networks
description: >-
  [NeurIPS 2025][Video Understanding][Bilinear RNN] This paper systematically revisits bilinear state transitions in RNNs—i.e., multiplicative interactions between the hidden state and the input—and theoretically proves th…
tags:
  - "NeurIPS 2025"
  - "Video Understanding"
  - "Bilinear RNN"
  - "State Tracking"
  - "Finite Automata"
  - "Multiplicative Interaction"
  - "Parity"
date: 2026-05-08
content_hash: 01a5a92bbab51105
---

# Revisiting Bi-Linear State Transitions in Recurrent Neural Networks

**Conference**: NeurIPS 2025
**arXiv**: [2505.21749](https://arxiv.org/abs/2505.21749)
**Code**: None
**Area**: Sequence Modeling / Recurrent Neural Networks
**Keywords**: Bilinear RNN, State Tracking, Finite Automata, Multiplicative Interaction, Parity

## TL;DR

This paper systematically revisits bilinear state transitions in RNNs—i.e., multiplicative interactions between the hidden state and the input—and theoretically proves that bilinear RNNs can simulate arbitrary finite-state machines. By removing additive terms, these models form a natural expressivity hierarchy ranging from diagonal to full-rank structures, revealing that popular linear RNNs such as Mamba occupy the lowest tier of this hierarchy.

## Background & Motivation

**Background**: State tracking is a fundamental requirement in sequential decision-making (multi-turn dialogue, robotic control, agent-LLM pipelines) and can be formalized as the simulation of finite automata or regular languages. However, many popular sequence models—including Transformers, Mamba, and mLSTM—fail to learn state-tracking tasks beyond their training length.

**Limitations of Prior Work**:
- **Failure of Transformers**: Unable to learn state tracking beyond training lengths, even with large-scale pretrained models and chain-of-thought reasoning.
- **Limitations of Linear RNNs**: Diagonal linear RNNs such as Mamba cannot learn state tracking at arbitrary lengths. Although it is known that state transition matrices must be input-dependent and permit negative eigenvalues, the set of learnable tasks remains highly restricted.
- **Historical Neglect of Bilinear Models**: Despite early work (Sutskever 2011), bilinear RNNs have not been widely adopted due to instability and optimization difficulties arising from three-way multiplicative interactions.

**Key Challenge**: State tracking inherently requires the hidden state to participate in *computation* (rather than mere *memory*), necessitating that the input *route* information flow through the hidden state. The additive structure of existing linear RNNs is fundamentally ill-suited for this purpose.

**Goal**: To characterize the state-tracking capacity of bilinear RNNs (with multiplicative interactions and no additive terms) and to determine what expressivity hierarchy different parameterizations induce.

**Key Insight**: The authors adopt a purely multiplicative interaction (no bias, no additive input term) and exploit scale invariance to address training stability.

**Core Idea**: A pure bilinear RNN with all additive terms removed can simultaneously simulate arbitrary finite-state machines and maintain training stability via runtime normalization enabled by scale invariance, yielding a well-defined expressivity hierarchy.

## Method

### Overall Architecture

The core recurrence (**pure bilinear, no additive terms**):

$$h_i^t = (h^{t-1})^\top \mathcal{W}_i x^t = \sum_{jk} \mathcal{W}_{ijk} x_k^t h_j^{t-1}$$

This is equivalent to an input-dependent state transition matrix: $h^t = \mathcal{A}_x h^{t-1}$, where $(\mathcal{A}_x)_{ij} = \sum_k \mathcal{W}_{ijk} x_k$.

Key insight: The **absence of any additive term** endows the hidden state with **scale invariance**—multiplying the hidden state by a constant at any timestep and dividing by that constant subsequently leaves the final output unchanged.

### Key Designs

#### 1. Expressivity Hierarchy of Bilinear Models

From highest to lowest expressivity:

| Model | Simulable Tasks | Parameter Count |
|-------|----------------|-----------------|
| Full Bilinear | Arbitrary finite-state machines | $H^2 D$ |
| CP-Factored Bilinear | Progressively approaches full model as $R$ increases | $R(2H+D)$ |
| Block-Diagonal Bilinear | Block size $\geq$ number of states | $H'^2 D \cdot B$ |
| $\mathcal{R}_2$ Block-Diagonal | Abelian groups only (commutative operations) | — |
| Real Diagonal | Parity only | $HD$ |
| Positive Diagonal (Mamba) | No state-tracking capacity | — |

#### 2. Full Bilinear RNN Simulates Arbitrary FSMs (Proposition 1)

The proof encodes states as one-hot vectors; the input-dependent transition matrix then directly encodes an arbitrary transition function $\delta(q, \sigma)$.

#### 3. CP Decomposition Reduces Parameter Count

$$\mathcal{A}_x = \mathcal{W}^{(h_1)} \text{diag}((\mathcal{W}^{(x)})^\top x) (\mathcal{W}^{(h_2)})^\top$$

This reduces parameters from $H^2 D$ to $R(2H+D)$, where $R$ is the decomposition rank. Experiments show that increasing $R$ progressively recovers the capacity of the full model.

#### 4. Complex Diagonal Bilinear RNNs Are Restricted to Abelian Groups (Proposition 2)

When $\mathcal{P}_x = \mathcal{P}$ (shared eigenbasis), all matrices $\mathcal{A}_x$ commute, implying that only commutative group operations can be simulated. This is a **fundamental negative result**: even complex diagonal ($2 \times 2$ rotation) blocks cannot learn general state machines.

#### 5. Real Diagonal Models Can Trivially Learn Parity (Proposition 3)

**Frozen random weights** combined with training only a linear readout layer suffice to learn parity at arbitrary lengths from as few as 2 training examples, with success probability $1 - 2^{-H}$. No training of recurrent parameters is required.

### Loss & Training

- **Exploiting Scale Invariance**: Hidden states are not normalized during training (preserving gradient flow) but are normalized at inference (preventing overflow); the purely multiplicative structure ensures these two regimes are mutually consistent.
- Data: training lengths 2–10, test length 500.
- 100,000 training steps, batch size 64.
- Best learning rate selected from 3 candidates.

## Key Experimental Results

### Main Results: Modular Addition (Length Generalization, OOD Length 500)

| Model | m=2 | m=3 | m=5 | m=10 | m=25 | m=50 |
|-------|-----|-----|-----|------|------|------|
| Bilinear (full) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Block-diag (size=1, real diag) | 1.00 | 0.00 | 0.00 | 0.10 | 0.00 | 0.02 |
| Block-diag (size=2) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| $\mathcal{R}_2$ Block-diag | 1.00 | 0.00 | 1.00 | 0.66 | 0.37 | 0.00 |
| LSTM | 1.00 | 1.00 | 0.98 | 1.00 | 0.00 | 0.02 |
| Mamba (1/2/4 layers) | 0.00 | 0.01 | 0.01 | 0.00 | 0.00 | 0.00 |
| Transformer (1/2/4 layers) | 0.03 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |

### Random State Machine (OOD Length 500)

| Model | m=2 | m=5 | m=10 | m=25 |
|-------|-----|-----|------|------|
| Bilinear (full) | 1.00 | 1.00 | 1.00 | 1.00 |
| Factored Bilinear | 1.00 | 1.00 | 1.00 | 0.50 |
| Block-diag (size=2) | 1.00 | 0.00 | 0.00 | 0.00 |
| LSTM | 1.00 | 1.00 | 0.00 | 0.00 |
| Mamba (4 layers) | 0.00 | 0.00 | 0.00 | 0.00 |

### Ablation Study: With vs. Without Additive Terms

| Model | Parity OOD (len 500) |
|-------|------|
| Pure Bilinear (no additive terms) | 1.00 |
| With additive terms (standard form) | Significantly lower |

### Key Findings

1. **Bilinear models consistently achieve the best results across all tasks**: The full bilinear model reaches 1.00 OOD accuracy on modular addition, random state machines, and modular arithmetic.
2. **Block size 2 is sufficient for modular addition** but insufficient for learning general state machines.
3. **Mamba fails completely on all OOD evaluations**: Even with 4 layers, OOD accuracy on modular addition is 0.
4. **Additive terms are harmful**: Removing biases and additive input contributions is critical for learning state tracking.
5. **Learning parity with frozen weights**: No recurrent parameter training is needed; 2 examples plus a linear readout suffice.

## Highlights & Insights

1. **Clear revelation of the expressivity hierarchy**: From full bilinear to factored, block-diagonal, real-diagonal, and positive-diagonal (Mamba), a well-defined chain of expressivity degradation is established.
2. **Importance of negative results**: $2 \times 2$ complex diagonal blocks can only learn Abelian groups—this implies that simply "allowing complex eigenvalues" cannot resolve general state tracking.
3. **"Input routing" perspective**: Viewing the hidden state as an active *computational participant* rather than passive memory, multiplicative interactions allow the input to determine how the hidden state is transformed.
4. **Elegant exploitation of scale invariance**: The apparent inconsistency between training (no normalization) and inference (normalization) is perfectly resolved by the purely multiplicative structure—runtime normalization does not affect outputs.

## Limitations & Future Work

1. **Large parameter count**: The third-order tensor $\mathcal{W} \in \mathbb{R}^{H \times H \times D}$ of the full bilinear model contains $H^2 D$ parameters; CP decomposition alleviates this but may sacrifice expressivity.
2. **Inefficient parallel training**: Three-way multiplicative interactions preclude direct application of standard parallel scan algorithms.
3. **Validation limited to toy tasks**: Large-scale experiments such as language modeling are absent.
4. **Gap from practical LLM training**: Hidden dimension 256 and training lengths of 2–10 are far smaller than real-world scales.
5. **Degradation of factored models in large state spaces**: Factored bilinear accuracy drops to 0.95 at $m=50$.

## Related Work & Insights

- **Mamba/GLA/mLSTM**: This work reveals that these models occupy the lowest tier of the bilinear hierarchy and lack state-tracking capacity.
- **DeltaNet/DeltaProduct**: Increase expressivity via Householder structures—an orthogonal yet complementary direction.
- **Grazzi et al. (ICLR 2025)**: Highlights the importance of negative eigenvalues; the present work provides a more complete theoretical account from the bilinear perspective.
- **Observable Operator Models**: Bilinear RNNs can be viewed as a continuous relaxation thereof.
- **Implications for linear RNN research**: The core expressivity bottleneck lies not in the range of eigenvalues but in the fundamental distinction between multiplicative and additive interactions.

## Rating

⭐⭐⭐⭐

The theoretical analysis is rigorous and insightful; the revealed hierarchy is a genuine contribution. However, the absence of large-scale experiments and efficient parallelization schemes creates a gap from practical deployment. As an analytical work, it is highly accomplished.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products](deltaproduct_improving_state-tracking_in_linear_rnns_via_householder_products.md)
- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)
- [\[NeurIPS 2025\] Neural Stochastic Flows: Solver-Free Modelling and Inference for SDE Solutions](neural_stochastic_flows_solver-free_modelling_and_inference_for_sde_solutions.md)
- [\[NeurIPS 2025\] PASS: Path-Selective State Space Model for Event-Based Recognition](pass_path-selective_state_space_model_for_event-based_recognition.md)
- [\[NeurIPS 2025\] Agentic Persona Control and Task State Tracking for Realistic User Simulation](agentic_persona_control_and_task_state_tracking_for_realistic_user_simulation_in.md)

</div>

<!-- RELATED:END -->
