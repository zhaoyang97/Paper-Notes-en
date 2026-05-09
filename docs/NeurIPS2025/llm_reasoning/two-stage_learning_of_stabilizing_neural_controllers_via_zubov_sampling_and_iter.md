---
title: >-
  [Paper Note] Two-Stage Learning of Stabilizing Neural Controllers via Zubov Sampling and Iterative Domain Expansion
description: >-
  [NeurIPS 2025][LLM Reasoning][neural controller] A two-stage training framework is proposed: the first stage estimates the region of attraction (ROA) via Zubov-guided sampling and dynamic domain expansion, while the second stage refines the result through CEGIS-based counterexample-driven training. The framework jointly learns a neural network controller and a Lyapunov function, achieving ROA volumes 5 to $1.5 \times 10^5$ times larger than baselines and verification speeds 40–10000× faster than dReal.
tags:
  - NeurIPS 2025
  - LLM Reasoning
  - neural controller
  - Lyapunov function
  - region of attraction
  - Zubov theorem
  - neural network verification
date: 2026-05-08
content_hash: 56603a849c3b0cb2
---

# Two-Stage Learning of Stabilizing Neural Controllers via Zubov Sampling and Iterative Domain Expansion

**Conference**: NeurIPS 2025
**arXiv**: [2506.01356](https://arxiv.org/abs/2506.01356)
**Code**: [GitHub](https://github.com/Verified-Intelligence/Two-Stage_Neural_Controller_Training)
**Area**: LLM Reasoning
**Keywords**: neural controller, Lyapunov function, region of attraction, Zubov theorem, neural network verification

## TL;DR
A two-stage training framework is proposed: the first stage estimates the region of attraction (ROA) via Zubov-guided sampling and dynamic domain expansion, while the second stage refines the result through CEGIS-based counterexample-driven training. The framework jointly learns a neural network controller and a Lyapunov function, achieving ROA volumes 5 to $1.5 \times 10^5$ times larger than baselines and verification speeds 40–10000× faster than dReal.

## Background & Motivation
**Background**: Neural network-based control policies have demonstrated strong performance on complex systems, yet lack formal stability guarantees in safety-critical scenarios. Estimating the region of attraction (ROA) is a central approach for quantifying safety, typically achieved by searching for Lyapunov functions satisfying algebraic conditions.

**Limitations of Prior Work**:
   - Existing methods typically employ **fixed training domains**, resulting in overly conservative ROA estimates.
   - Sampling strategies (e.g., naive random sampling or pure CEGIS counterexample-driven approaches) are inefficient in high-dimensional systems — the ROA may occupy less than 1% of the training domain, making it nearly impossible to sample interior points randomly.
   - CEGIS methods depend on carefully tuned LQR/RL initialization and lack generalizability to new systems.
   - Formal verification of continuous-time systems relies on SMT solvers (e.g., dReal), which are extremely slow.

**Key Challenge**: Training domain selection and training data selection are the two key factors contributing to conservativeness, yet existing methods do not address both simultaneously.

**Goal**: (1) Design improved training sampling and domain selection strategies to reduce ROA estimation conservativeness; (2) Extend the α,β-CROWN verifier to support Jacobian operations for continuous-time systems.

**Key Insight**: Zubov's theorem provides a PDE characterization of the true ROA (where $W(x) \to 1$ at the ROA boundary), which can simultaneously guide data sampling (balanced sampling inside and outside the ROA) and domain updates (expansion along convergent trajectories).

**Core Idea**: Leverage Zubov's theorem to simultaneously guide training data sampling and dynamic training domain expansion, followed by CEGIS refinement to produce a formally verifiable controller.

## Method

### Overall Architecture
A two-stage training pipeline: the first stage (ROA estimation) uses Zubov-guided sampling and domain expansion to obtain a "near-ready" controller and Lyapunov function; the second stage (CEGIS refinement) applies counterexample-driven adversarial training to eliminate all violations of the Lyapunov conditions, yielding a formally verifiable result.

### Key Designs

1. **Zubov Guided Sampling**:

    - **Function**: Balances data sampling inside and outside the ROA during training.
    - **Mechanism**: Exploits the property of Zubov's theorem — $V_\theta < 1$ inside the ROA and $V_\theta \to 1$ at its boundary. Two objectives are optimized via PGD to select training points: $L_{\text{interior}} = \text{ReLU}(V_\theta(x) - c)$ projects points into the sublevel set $V_\theta^{\leq c}$; $L_{\text{outside}} = |V_\theta(x) - 1|$ pushes points toward the exterior of the ROA.
    - **Design Motivation**: Sampling purely from the interior prevents ROA expansion, while sampling purely from the exterior causes the sublevel set to collapse toward the origin. In high-dimensional settings, the ROA occupies a negligible fraction of the training domain (e.g., less than 1% for cartpole), making naive random sampling nearly incapable of drawing interior points.

2. **Dynamic Training Domain Expansion**:

    - **Function**: Starts from a small training domain and periodically expands it along convergent trajectories.
    - **Mechanism**: Trajectory simulations are initiated from points within the current ROA estimate $V_\theta^{\leq c}$, and the training domain is expanded to encompass the farthest states reached along convergent trajectories, followed by uniform scaling to ensure coverage of simpler systems.
    - **Design Motivation**: A fixed training domain is either too small (truncating the ROA) or too large (reducing sampling efficiency). Dynamic expansion simultaneously addresses conservativeness and forward invariance.

3. **Bellman-Type Data Loss**:

    - **Function**: Avoids computing the infinite-time integral $\int_0^\infty \|\phi(t,x)\|^p dt$.
    - **Mechanism**: Applies a dynamic programming decomposition $W(x) = \tanh\!\left(a\int_0^T \|\phi(t,x)\|^p dt + \tanh^{-1}(W(\phi(T,x)))\right)$, requiring simulation only to a short horizon $T=0.01\text{s}$.
    - **Design Motivation**: Convergence rates vary across trajectories, making infinite-time integration extremely slow to compute and difficult to batch.

4. **Jacobian Extension for α,β-CROWN**:

    - **Function**: Extends the state-of-the-art neural network verifier to support Jacobian-vector products for continuous-time dynamical systems.
    - **Mechanism**: New linear relaxations are designed for the Jacobian operators of activations such as Tanh and Sigmoid, along with a dynamically adjusted verification specification scheme to avoid expensive bisection operations.
    - **Design Motivation**: Existing verification of continuous-time systems relies on SMT solvers such as dReal, which can require tens of thousands of seconds.

### Loss & Training
The total training loss is a weighted combination of five terms: a zero-point loss $L_{\text{zero}}$ (enforcing $V(0) \approx 0$), a PDE residual loss $L_{\text{pde}}$ (Zubov equation), a data loss $L_{\text{data}}$ (Bellman-type ground-truth matching), a controller loss $L_{\text{controller}}$ (Sontag-style), and a boundary loss $L_{\text{boundary}}$ ($V \to 1$ at the training domain boundary). An actor-critic framework is used to decouple the learning of the Lyapunov function and the controller.

## Key Experimental Results

### Main Results (ROA Volume Comparison)

| System | FACTEST | Chang (CEGIS) | Yang (CEGIS) | Shi (FICL) | **Ours** |
|------|---------|--------------|-------------|-----------|---------|
| Van der Pol | 18.7 | 20.0 | 22.74 | 22.59 | **23.3** |
| Pendulum B | - | 285.2 | 217.3 | 306.3 | **1169.2** |
| Path Tracking B | - | 9.0 | 24.06 | 15.3 | **122.0** |
| Cartpole (4D) | - | 0.021 | - | 0.93 | **306.1** |
| PVTOL (6D) | - | - | - | 49.87 | **1.91×10⁴** |
| 2D Quadrotor (6D) | - | - | 2.33 | 44.53 | **6.64×10⁶** |
| Ducted Fan (6D) | - | - | - | - | **4.31×10⁴** |
| 3D Quadrotor (12D) | - | - | - | - | **1.17×10⁹** |

### Ablation Study

| Training Configuration | Formal Verification | PGD Verification | Trajectory Verification |
|----------|-----------|---------|---------|
| Two-stage + random sampling | 0% | 0% | 0% |
| ROA estimation stage only | 0% | 0% | 100% |
| CEGIS only | 0% | 0% | 0% |
| **Full pipeline** | **100%** | **100%** | **100%** |

### Verification Speed Comparison

| System | dReal | α,β-CROWN (Ours) | Speedup |
|------|-------|-----------------|--------|
| Van der Pol | 39265s | 3.94s | ~10000× |
| Pendulum B | 1480s | 3.64s | ~400× |
| Cartpole | Timeout | 76443s | - |

### Key Findings
- Zubov-guided sampling is critical to success: random sampling with the same two-stage architecture fails completely (0% success rate).
- The ROA estimation stage alone passes trajectory verification (the controller is "near-ready") but fails formal verification, demonstrating the necessity of CEGIS refinement.
- CEGIS alone, without a good initialization, also fails completely, confirming the complementarity of the two-stage design.
- This work is the first to handle 12D systems (3D quadrotor), on which all baselines fail.

## Highlights & Insights
- **Zubov's theorem serves three purposes at once**: it simultaneously addresses the sampling strategy, domain expansion, and physics-informed loss, serving as an excellent example of theory-guided practice.
- **Bellman decomposition avoids infinite-time integration**: the ground-truth computation for the Zubov function is reduced from an infinite-time integral to a short simulation ($T=0.01\text{s}$), greatly improving practical applicability.
- **Complementarity of the two-stage design**: the ablation study clearly demonstrates the necessity of "macro-level shaping + micro-level refinement," a paradigm transferable to other learning problems requiring formal verification.

## Limitations & Future Work
- **Formal verification of high-dimensional systems remains challenging**: bound propagation for Jacobian-vector products in systems of 6D and above is too loose; 12D systems can only be verified via trajectory-based methods.
- **Continuous-time systems only**: discrete-time, hybrid, and time-varying systems are not yet addressed.
- **No robustness guarantees**: robust controller synthesis under system disturbances is not considered.

## Related Work & Insights
- **vs Yang et al. (2024) CEGIS+α,β-CROWN**: The key contribution of this work is the Zubov sampling + domain expansion first stage, eliminating dependence on LQR initialization and achieving ROA improvements of several orders of magnitude.
- **vs Liu et al. (2025) Physics-informed Zubov**: This work additionally introduces a domain expansion mechanism and Bellman decomposition, and combines Zubov-based losses with CEGIS, achieving success on higher-dimensional systems.

## Rating
- Novelty: ⭐⭐⭐⭐ The systematic application of Zubov's theorem to controller synthesis is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 systems, 5 baselines, detailed ablations, and verification time comparisons.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear and experimental design is comprehensive.
- Value: ⭐⭐⭐⭐ Significant practical value in safety-critical control, with remarkable ROA improvement margins.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Clip-and-Verify: Linear Constraint-Driven Domain Clipping for Accelerated Neural Network Verification](clip-and-verify_linear_constraint-driven_domain_clipping_for_accelerating_neural.md)
- [\[NeurIPS 2025\] Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding](sampling-efficient_test-time_scaling_self-estimating_the_best-of-n_sampling_in_e.md)
- [\[NeurIPS 2025\] AbbIE: Autoregressive Block-Based Iterative Encoder for Efficient Sequence Modeling](abbie_autoregressive_block-based_iterative_encoder_for_efficient_sequence_modeli.md)
- [\[NeurIPS 2025\] DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning](dreamprm_domain-reweighted_process_reward_model_for_multimodal_reasoning.md)
- [\[AAAI 2026\] SERL: Self-Examining Reinforcement Learning on Open-Domain](../../AAAI2026/llm_reasoning/serl_self-examining_reinforcement_learning_on_open-domain.md)

<!-- RELATED:END -->
