---
title: >-
  [Paper Note] Structured Reinforcement Learning for Combinatorial Decision-Making
description: >-
  [NeurIPS 2025][Reinforcement Learning][combinatorial MDP] This paper proposes Structured Reinforcement Learning (SRL), which embeds a combinatorial optimization solver as a differentiable layer within the actor of an act…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "combinatorial MDP"
  - "structured actor"
  - "Fenchel-Young loss"
  - "COAML pipeline"
  - "primal-dual algorithm"
date: 2026-05-08
content_hash: d0e107bc4ef947ce
---

# Structured Reinforcement Learning for Combinatorial Decision-Making

**Conference**: NeurIPS 2025
**arXiv**: [2505.19053](https://arxiv.org/abs/2505.19053)
**Code**: [GitHub](https://github.com/tumBAIS/Structured-RL)
**Area**: Reinforcement Learning / Combinatorial Optimization / Operations Research
**Keywords**: combinatorial MDP, structured actor, Fenchel-Young loss, COAML pipeline, primal-dual algorithm

## TL;DR
This paper proposes Structured Reinforcement Learning (SRL), which embeds a combinatorial optimization solver as a differentiable layer within the actor of an actor-critic framework. End-to-end gradient propagation is achieved via Fenchel-Young loss with Gaussian perturbations, enabling purely online learning without expert demonstrations. SRL matches imitation learning and outperforms unstructured RL by up to 92% across six industrial-scale combinatorial decision-making problems.

## Background & Motivation
**Problem Definition — Combinatorial MDP (C-MDP)**: Given state $s \in \mathcal{S}$, action $a \in \mathcal{A}(s) \subset \mathbb{R}^{d(s)}$, where $\mathcal{A}(s)$ is the feasible solution set of a combinatorial problem (routing, scheduling, assortment selection, etc.), and its convex hull $\mathcal{C}(s) = \text{conv}(\mathcal{A}(s))$ is referred to as the marginal polytope. The objective is to find a policy that maximizes cumulative discounted reward under unknown transition dynamics $P(s', r | s, a)$.

**Three Key Limitations of Existing Approaches**:
- **Unstructured RL** (PPO/SAC): Treats the CO layer as part of the environment, inducing a piecewise-constant and highly non-smooth reward function with extremely high gradient variance, making convergence difficult.
- **Structured Imitation Learning (SIL)**: Requires pre-collected expert demonstrations, with performance upper-bounded by the expert policy; recovery from off-expert trajectories is infeasible.
- **Predict-then-Optimize**: Decouples prediction from optimization, precluding end-to-end learning of long-term returns.

**Core Insight**: By embedding the CO solver directly into the actor architecture (rather than treating it as a post-processing step in the environment), the actor naturally outputs combinatorially feasible solutions. Perturbation-based regularization simultaneously renders the CO layer differentiable.

## Method

### Overall Architecture: COAML Actor-Critic
SRL replaces the standard neural network actor in actor-critic with a COAML (Combinatorial Optimization-Augmented ML) pipeline:
- **Neural network $\phi_w$**: Maps state $s$ to a scoring vector $\theta = \phi_w(s) \in \mathbb{R}^d$, encoding contextual information, variable dependencies, and environment dynamics.
- **CO layer $f$**: Solves $f(\theta, s) = \arg\max_{a \in \mathcal{A}(s)} \langle \theta \mid a \rangle$ using $\theta$ as linear objective coefficients, ensuring feasibility and scalability.
- **Policy representation**: $\pi_w(\cdot \mid s) := \delta_{f(\phi_w(s), s)}$, a deterministic Dirac policy at inference time — no sampling or projection required.
- **Critic $\psi_\beta$**: Estimates $Q(s, a)$ via standard TD learning; double Q-learning is employed in complex environments to mitigate overestimation.

### Key Design 1: Fenchel-Young Loss for Non-Differentiability
The CO layer $f$ is piecewise-constant with respect to $\theta$ — within the normal cone of a vertex of the marginal polytope $\mathcal{C}(s)$, $f$ maps to the same vertex and the gradient is zero. SRL resolves this via Gaussian perturbation regularization:
- A perturbation $Z \sim \mathcal{N}(0, \varepsilon)$ is introduced, transforming the loss to $L_\Omega(\theta; \hat{a}) = \mathbb{E}[\max_{a \in \mathcal{A}(s)} (\theta + Z)^\top a] - \theta^\top \hat{a}$.
- This loss is both differentiable and convex in $\theta$; gradients are computed via a pathwise estimator with substantially lower variance than score-function estimators.
- A key property of the Fenchel-Young framework: the loss is zero if and only if $\hat{a}$ solves the regularized prediction problem, naturally measuring suboptimality.

### Key Design 2: Critic-Based Online Target Action Estimation
SRL constructs target actions $\hat{a}$ online without any expert supervision — the core distinction from SIL:
1. Gaussian perturbations $Z \sim \mathcal{N}(\theta, \sigma_b)$ are applied to the scoring vector $\theta$, sampling $m$ perturbed vectors $\eta$.
2. Each $\eta$ is passed through the CO layer to generate a candidate action $a' = f(\eta, s)$.
3. The Critic evaluates the $Q$-value of each candidate action.
4. Softmax-weighted aggregation: $\hat{a} = \sum a' \cdot \exp(Q/\tau) / \sum \exp(Q/\tau)$.
- As $\tau \to 0$, this approaches a greedy argmax; as $\tau \to \infty$, it approaches a uniform average.
- $\hat{a}$ need not be feasible (it is never executed) and serves solely as a training signal for the Fenchel-Young loss.
- Three benefits of the softmax estimator: credit assignment across actions encourages exploration; value averaging mitigates Critic overestimation bias; avoidance of extreme actions prevents premature convergence.

### Key Design 3: Triple Gaussian Perturbation Mechanism
SRL employs three independent Gaussian perturbations, each serving a distinct purpose:
- **$\sigma_f$ (forward pass)**: Explores the state space $\mathcal{S}$, enabling the agent to visit more diverse states.
- **$\sigma_b$ (target action selection)**: Explores the combinatorial action space $\mathcal{A}(s)$, generating candidates for Critic evaluation.
- **$\varepsilon$ (Fenchel-Young loss)**: Regularizes the CO layer to ensure differentiability and constructs a convex surrogate objective.

### Geometric Interpretation: Sampling-Based Primal-Dual Algorithm in Dual Space
The paper provides a rigorous geometric analysis (focused on the static case $T=1$, generalizable to multi-stage settings via Critic learning):
- The convex hull $\mathcal{C}(s)$ of the action space forms a marginal polytope; the scoring vector $\theta$ defines a normal fan decomposition.
- The policy can be expressed as a regularized optimization over the distributional simplex $\Delta^{\mathcal{A}(s)}$: $\pi_w(\cdot \mid s) = \arg\max_{q \in \Delta} \{\langle A^\top \phi_w(s) \mid q \rangle - \Omega(q)\}$.
- The SRL actor update is formally expressed as a five-step primal-dual iteration (Proposition 2): sampling → empirical distribution → subgradient → softmax aggregation → Fenchel-Young update.
- **Key distinction from Bouvier et al. (2025)**: That work requires solving $\max_{a \in \mathcal{A}(s)} r(s, a, \xi) + \langle \theta \mid a \rangle$, a nonlinear combinatorial optimization problem that is generally intractable. SRL replaces this with $m$ samples evaluated by the Critic, preserving computational tractability.

### Why Not SAC?
The paper provides a detailed justification for choosing PPO rather than SAC as the unstructured RL baseline:
- SAC requires the actor to output both mean and standard deviation, necessitating specialized network architectures.
- SAC requires the Critic to be differentiable with respect to the actor, but $Q(s, a)$ operates directly on combinatorial actions $a$, blocking gradient backpropagation.
- SAC's entropy regularization acts on the distribution over $\theta$, whereas what is needed is regularization over the distribution of combinatorial actions $a$.

## Key Experimental Results

### Six Test Environments
**Static ($T=1$)**: Warcraft shortest path (Dijkstra CO layer); single-machine scheduling (ranking CO layer, 50–100 jobs); stochastic vehicle scheduling (LP flow solver CO layer, 25 tasks).
**Dynamic ($T>1$)**:
- **Dynamic Vehicle Scheduling Problem (DVSP)** (exogenous uncertainty): 8 time steps, requests revealed over time, requiring real-time route planning.
- **Dynamic Assortment Problem (DAP)** (endogenous uncertainty): Customer choice follows an MNL model; item features evolve with decisions.
- **Grid-world Shortest Path Problem (GSPP)** (endogenous uncertainty): The goal moves upon arrival; path costs are influenced by historical decisions.

### Experimental Design
- All three algorithms share the same COAML pipeline; SRL and PPO share the same Critic architecture.
- The number of training episodes is determined by tuning PPO; SRL and SIL use a consistent number of update steps.
- Each algorithm is run with 10 random seeds; mean and standard deviation are reported.
- All experiments are conducted on a MacBook Air M3 (Julia language), with per-environment runtimes of 3–90 minutes.

### Main Results on Dynamic Environments

| Comparison | DVSP | DAP | GSPP |
|---|---|---|---|
| SRL vs SIL | Comparable | **+8%** | **+78%** |
| SRL vs PPO | **+16%** | **+77%** | **+92%** |

### Stability (Training Reward Std / Test Reward Std / Training Time)

| Algorithm | DVSP | DAP | GSPP |
|---|---|---|---|
| SIL | 0.3 / 0.4 / 12m | 0.8 / 11.9 / 3m | 39.3 / 1.1 / 11m |
| PPO | 5.8 / 5.6 / 3m | 5.4 / 13.5 / 5m | 105.8 / 47.0 / 10m |
| SRL | 0.3 / 0.3 / 31m | 1.8 / 1.9 / 31m | 72.1 / 0.6 / 34m |

- PPO's training variance is up to **80×** higher than SRL's; its test variance on GSPP is 78× higher (47.0 vs. 0.6).
- SRL outperforms the online-optimal expert policy by 79% on GSPP — SIL is bounded by the expert ceiling and cannot escape suboptimal regimes.
- Convergence speed: PPO requires 160–200 episodes to stabilize; SRL and SIL converge at comparable rates, with SIL converging 10–150 episodes faster on DAP/GSPP due to expert guidance.
- On static tasks, SRL matches SIL without requiring expert data and outperforms PPO by up to 54%.

### Computational Cost Analysis
- SRL invokes the CO layer at most 61 times per update ($m=40$ samples + 20 FY loss evaluations + 1 forward pass); SIL requires 20; PPO requires only 2.
- Computational cost scales linearly with CO layer complexity: for GSPP (simple CO layer), runtimes across all three methods are comparable; for DVSP (complex CO layer), SRL takes approximately 2.5× longer than SIL.

## Highlights & Insights
1. **Purely online learning without expert data** while maintaining extremely low variance — the structural constraints of the CO layer compress exploration over an exponentially large action space into the continuous space of scoring vectors $\theta$.
2. **Combinatorial feasibility is guaranteed by construction**: The CO layer directly outputs solutions satisfying all constraints, obviating the need for PPO-style continuous relaxation followed by projection.
3. **Greatest advantage in dynamic environments**: SIL only observes expert trajectories and cannot recover after deviating; SRL can discover policies superior to the expert through continuous online exploration.
4. **Theoretical and practical unity**: The primal-dual geometric interpretation is not merely elegant — it directly motivates the algorithmic design, replacing intractable nonlinear combinatorial optimization with sampling-based Critic evaluation.
5. **Lightweight experimental setup**: All experiments run on a MacBook Air M3; the neural networks in the COAML pipeline are typically small, underscoring the practical accessibility of the method.

## Limitations & Future Work
- **Computational overhead**: Frequent CO solver calls (up to 61 per update) incur significant cost when the CO layer is complex (e.g., VRP).
- **Convergence speed**: Purely online exploration without expert guidance is 10–150 episodes slower than SIL.
- **Critic limitations**: While the $Q$-function requires no decomposability (yielding flexibility), it cannot be directly optimized over actions, necessitating reliance on sampling.
- **Problem scale**: Validation is conducted on medium-scale instances (50–100 jobs for scheduling, 25 tasks for VSP); scalability to very large instances remains unexplored.
- **Static Critic assumption**: The geometric analysis assumes a fixed Critic in the static setting; theoretical guarantees for multi-stage dynamic settings have yet to be established.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Spatial-Aware Decision-Making with Ring Attractors in Reinforcement Learning Systems](spatial-aware_decision-making_with_ring_attractors_in_reinforcement_learning_sys.md)
- [\[NeurIPS 2025\] Prompt Tuning Decision Transformers with Structured and Scalable Bandits](prompt_tuning_decision_transformers_with_structured_and_scalable_bandits.md)
- [\[NeurIPS 2025\] Modulation of Temporal Decision-Making in a Deep Reinforcement Learning Agent under the Dual-Task Paradigm](modulation_of_temporal_decision-making_in_a_deep_reinforcement_learning_agent_un.md)
- [\[NeurIPS 2025\] Bandit and Delayed Feedback in Online Structured Prediction](bandit_and_delayed_feedback_in_online_structured_prediction.md)
- [\[NeurIPS 2025\] Complexity Scaling Laws for Neural Models using Combinatorial Optimization](complexity_scaling_laws_for_neural_models_using_combinatorial_optimization.md)

</div>

<!-- RELATED:END -->
