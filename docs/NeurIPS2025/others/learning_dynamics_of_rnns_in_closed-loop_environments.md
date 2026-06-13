---
title: >-
  [Paper Note] Learning Dynamics of RNNs in Closed-Loop Environments
description: >-
  [NeurIPS 2025][RNN learning dynamics] This paper establishes a mathematical theory revealing that RNNs exhibit fundamentally different learning dynamics under closed-loop (agent–environment interaction) versus open-loop…
tags:
  - "NeurIPS 2025"
  - "RNN learning dynamics"
  - "closed-loop learning"
  - "open-loop vs. closed-loop"
  - "control theory"
  - "internal representations"
date: 2026-05-08
content_hash: 5897ec5310d2b5a8
---

# Learning Dynamics of RNNs in Closed-Loop Environments

**Conference**: NeurIPS 2025
**arXiv**: [2505.13567](https://arxiv.org/abs/2505.13567)  
**Code**: [GitHub](https://github.com/yoavger/closed_loop_rnn_learning_dynamics)  
**Area**: Theory / Recurrent Neural Networks / Control Theory
**Keywords**: RNN learning dynamics, closed-loop learning, open-loop vs. closed-loop, control theory, internal representations

## TL;DR

This paper establishes a mathematical theory revealing that RNNs exhibit fundamentally different learning dynamics under closed-loop (agent–environment interaction) versus open-loop (supervised learning) training. Closed-loop learning follows a three-phase process driven by the competition between short-term policy improvement and long-term stability.

## Background & Motivation

**Background**: RNNs are widely used in neuroscience modeling and sequence tasks. Existing theoretical work primarily analyzes RNN learning dynamics and solution properties in open-loop (supervised learning) settings.

**Limitations of Prior Work**: Biological learning occurs in closed-loop environments—where an agent's actions influence subsequent inputs—yet a theoretical account of closed-loop RNN learning dynamics is largely absent.

**Key Challenge**: Open-loop analyses assume i.i.d. inputs and ignore feedback loops. In closed-loop environments, outputs affect subsequent inputs, making the learning dynamics fundamentally different.

**Goal**: To establish a mathematical theory of closed-loop RNN learning dynamics and reveal why and how closed-loop learning differs from its open-loop counterpart.

**Key Insight**: The authors adopt the classical double integrator control task and obtain an analytically tractable theoretical framework via linearized RNNs and a rank-1 connectivity weight assumption.

**Core Idea**: Closed-loop RNN learning is governed by the eigenvalue evolution of the joint agent–environment system. The learning process unfolds in three stages, fundamentally driven by the competition between short-term policy improvement and long-term system stability.

## Method

### Overall Architecture

The research framework consists of:
- **Environment**: A discrete-time double integrator (position–velocity control task) with only position observed (partial observability).
- **Agent**: An RNN with $N=100$ neurons trained via policy gradient.
- A joint system matrix $\bm{P}$ is constructed to model the RNN and environment in a unified manner.

### Key Designs

1. **Joint Closed-Loop System**: The environmental state $\bm{x}_t$ and RNN hidden state $\bm{h}_t$ are concatenated into a joint state $\bm{s}_t = (\bm{x}_t, \bm{h}_t)^\top$, yielding a linear dynamical system:
$$\bm{s}_{t+1} = \bm{P} \bm{s}_t, \quad \bm{P} = \begin{bmatrix} \bm{A} & \bm{B}\bm{z}^\top \\ \bm{m}\bm{C}\bm{A} & \bm{W} \end{bmatrix}$$
System stability is determined by the eigenvalues of $\bm{P}$.

2. **Effective Low-Dimensional System**: Under the rank-1 connectivity assumption ($\bm{W} = \bm{u}\bm{v}^\top$), hidden states are confined to the subspace spanned by $\bm{m}$ and $\bm{u}$, reducing the system to 4 dimensions controlled by four scalar order parameters (overlaps): $\sigma_{\bm{z}\bm{m}}, \sigma_{\bm{z}\bm{u}}, \sigma_{\bm{v}\bm{m}}, \sigma_{\bm{v}\bm{u}}$.

3. **Effective Feedback Gain**: The high-dimensional nonlinear RNN policy is embedded into a 2D interpretable space $(k_1, k_2)$:
$$u_t \approx -k_1 x_t^{(1)} - k_2 x_t^{(2)}$$
Stability regions are analyzed via the closed-loop matrix $\bm{M}_{\text{cl}} = \bm{A} - \bm{B}\bm{K}$.

### Three-Phase Learning Dynamics

**Phase 1 — Negative-Position Policy**:
- Loss decreases rapidly; the RNN learns a proportional control strategy $u_t \propto -\text{position}$.
- The characteristic polynomial simplifies to $\chi_{\bm{P}}(\lambda) = \lambda^2 - 2\lambda + (1 - \sigma_{\bm{z}\bm{m}})$.
- An asymmetric loss landscape drives $\sigma_{\bm{z}\bm{m}}$ to a small negative value.
- The system is unstable ($\rho(\bm{P}) > 1$), manifesting as oscillatory divergence.

**Phase 2 — Building a World Model**:
- Loss enters a plateau; the RNN must learn to infer the latent variable (velocity).
- A surrogate loss is introduced: $\mathcal{L}_{\text{surrogate}} = \alpha \cdot \mathcal{L}_\infty + (1-\alpha) \cdot \mathcal{L}_2$.
- Gradients from the short-term control objective ($\mathcal{L}_2$) and the long-term stability objective ($\mathcal{L}_\infty$) point in nearly opposite directions, producing a zigzag trajectory.
- This phase ends when the dominant eigenvalue enters the unit circle (system stabilization).

**Phase 3 — Policy Refinement**:
- Loss decreases again; trajectories become rapid and non-oscillatory.
- A third real eigenvalue $\lambda_3$ grows, giving rise to a second slow mode.
- The low-dimensional effective model accurately reproduces the dynamics of this phase.

## Key Experimental Results

### Main Results: Closed-Loop vs. Open-Loop Learning

The authors compare two RNNs with identical architectures and initializations, trained in closed-loop and open-loop modes respectively:

| Training Mode | Initial Behavior | Mid-Training | Final Performance |
|--------------|-----------------|--------------|-------------------|
| Closed-loop | Similar → plateau | Three-phase progression | Stable convergence |
| Open-loop | Similar → loss spike | Closed-loop test loss deteriorates sharply | Eventually recovers but via a different path |

Key finding: The two modes traverse entirely different trajectories in the effective feedback gain space $(k_1, k_2)$.

### Validation on Multi-Frequency Tracking Task

| Feature | Observation |
|---------|-------------|
| Learning phases | Staircase-shaped loss decrease; each step corresponds to acquiring one frequency component |
| Order of frequency acquisition | Low → high frequency, consistent with human motor control experiments |
| Competition effect | Performance on previously acquired frequencies temporarily degrades when a new frequency is learned |

### Ablation Study

- **Linear vs. nonlinear RNNs**: Nonlinear RNNs qualitatively exhibit the same three-phase dynamics.
- **Episode length $T$**: Short $T$ drives eigenvalues upward along the imaginary axis; long $T$ drives them downward—validating the short-term/long-term competition theory.
- **Adam optimizer**: The zigzag trajectory is mitigated, as adaptive optimization alleviates gradient direction conflicts.

### Key Findings

- Closed-loop and open-loop training produce fundamentally different learning trajectories even with identical architectures and initializations.
- The plateau in closed-loop learning is not stagnation but a necessary phase during which the system builds an internal world model.
- Tracking the eigenvalues of the joint agent–environment system—rather than the RNN alone—is both necessary and sufficient for understanding closed-loop learning.

## Highlights & Insights

- **Strong theoretical contribution**: The first analytical theory of closed-loop RNN learning dynamics, filling an important gap in the literature.
- **Concise and illuminating physical picture**: The competition between short-term policy improvement and long-term stability provides a unified explanation for the plateau, zigzag trajectories, and related phenomena.
- **Connection to neuroscience**: The order in which the RNN acquires frequencies strikingly matches human motor learning experiments, suggesting shared inductive biases.
- **Low-dimensional effective model**: The 100-dimensional RNN dynamics are compressed into four scalar order parameters while preserving the key learning dynamics.

## Limitations & Future Work

- The theoretical analysis relies on linearization and rank-1 weight simplification; extension to fully nonlinear settings requires further work.
- Only direct gradient computation (policy gradient) is considered; more complex RL algorithms such as sparse rewards or actor-critic methods are not addressed.
- The effective system maintains precise spectral equivalence only within an episode; learning dynamics across episodes are not yet fully characterized.
- The control task is relatively simple (double integrator); applicability to higher-dimensional, nonlinear environments remains to be explored.

## Related Work & Insights

- This work is analogous to Saxe et al. 2013's analysis of feedforward network learning dynamics, serving as the corresponding theory for closed-loop RNNs.
- It complements Bordelon et al. 2025's analysis of open-loop RNN learning dynamics.
- It provides a theoretical perspective for understanding "phase transitions" in RL training (analogous to grokking?).
- Insight: RL algorithm design should account for the balance between short-term and long-term objectives, potentially mitigating competition via curriculum learning or episode-length scheduling.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First mathematical theory of closed-loop RNN learning dynamics, opening a new direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Theoretical validation is rigorous and the multi-frequency extension is convincing, though task complexity is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Elegant theoretical derivations, polished figures, and clear physical intuition.
- Value: ⭐⭐⭐⭐ Significant implications for understanding closed-loop and biological learning, though direct practical applicability is limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Fixed-Point RNNs: Interpolating from Diagonal to Dense](fixed-point_rnns_interpolating_from_diagonal_to_dense.md)
- [\[NeurIPS 2025\] Normalization in Attention Dynamics](normalization_in_attention_dynamics.md)
- [\[NeurIPS 2025\] RNNs Perform Task Computations by Dynamically Warping Neural Representations](rnns_perform_task_computations_by_dynamically_warping_neural_representations.md)
- [\[ICML 2026\] Learning Permutation-Invariant Macroscopic Dynamics](../../ICML2026/others/learning_permutation-invariant_macroscopic_dynamics.md)
- [\[NeurIPS 2025\] Recurrent Self-Attention Dynamics: An Energy-Agnostic Perspective from Jacobians](recurrent_self-attention_dynamics_an_energy-agnostic_perspective_from_jacobians.md)

</div>

<!-- RELATED:END -->
