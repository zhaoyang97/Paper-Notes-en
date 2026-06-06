---
title: >-
  [Paper Note] Randomized Advantage Transformation (RAT): Computing Natural Policy Gradients via Direct Backpropagation
description: >-
  [ICML2026][Reinforcement Learning][Natural Policy Gradient] By using the Woodbury identity, Tikhonov-regularized Natural Policy Gradient (NPG) is rewritten as a "standard policy gradient with transformed advantages." Thi…
tags:
  - "ICML2026"
  - "Reinforcement Learning"
  - "Natural Policy Gradient"
  - "Woodbury identity"
  - "Kaczmarz iteration"
  - "advantage transformation"
  - "on-policy RL"
date: 2026-05-08
content_hash: 6083af0274e2ac4c
---

# Randomized Advantage Transformation (RAT): Computing Natural Policy Gradients via Direct Backpropagation

**Conference**: ICML2026  
**arXiv**: [2605.18591](https://arxiv.org/abs/2605.18591)  
**Code**: https://github.com/agent-lab/ICML2026-RAT  
**Area**: reinforcement_learning  
**Keywords**: Natural Policy Gradient, Woodbury identity, Kaczmarz iteration, advantage transformation, on-policy RL  

## TL;DR
By using the Woodbury identity, Tikhonov-regularized Natural Policy Gradient (NPG) is rewritten as a "standard policy gradient with transformed advantages." This transformation is solved over mini-batches using randomized block Kaczmarz iterations, completely bypassing explicit Fisher matrix construction, conjugate gradient inner loops, and architecture-dependent curvature approximations like KFAC. It achieves natural policy gradients through a single standard backpropagation, matching or exceeding the performance of TRPO/ACKTR/KFAC on MuJoCo and Procgen.

## Background & Motivation

**Background**: Natural Policy Gradient (NPG) provides parameterization-invariant update directions by left-multiplying the standard policy gradient $\nabla^{\text{PG}}_{\bm{\theta}} J$ by the inverse Fisher matrix $\bm{F}^{-1}$. It serves as the theoretical foundation for TRPO, ACKTR, Natural Actor-Critic, and PPO-style updates. Theoretical analysis indicates that this "geometric correction" significantly improves convergence properties.

**Limitations of Prior Work**: However, the Fisher matrix $\bm{F}\in\mathbb{R}^{p\times p}$ scales with the number of parameters $p$ (often millions in deep policies), making explicit construction and inversion impractical. Two main workarounds exist:
- **Hessian-Free + CG** (e.g., TRPO): Converts inversion into Fisher-vector products solved via Conjugate Gradient iterations. Each step requires dozens of CG inner loops, which is computationally expensive and difficult to apply to shared actor-critic networks.
- **Structured Approximations** (e.g., KFAC): Assumes Fisher can be decomposed as layer-wise Kronecker products. This is faster but sacrifices accuracy and relies heavily on gradient independence assumptions, requiring re-derivation for different architectures.

**Key Challenge**: The benefit of NPG comes from "reshaping gradient directions with Fisher," but all existing implementations either spend excessive time precisely calculating $\bm{F}^{-1}\bm{g}$ or sacrifice accuracy for architecture-dependent approximations. Can natural gradients be computed using only "standard backpropagation" without explicit Fisher operations?

**Goal**: (1) Find a rewriting method such that NPG formally reduces to a standard policy gradient; (2) Provide a scalable finite-sample estimation algorithm for this rewriting; (3) Offer convergence guarantees and validation on large-scale benchmarks.

**Key Insight**: The authors noted a fact occasionally mentioned in literature—Tikhonov-regularized NPG is equivalent to a weighted least squares problem. Using the Woodbury identity $(\bm{I}+\bm{U}\bm{V})^{-1}\bm{U}=\bm{U}(\bm{I}+\bm{V}\bm{U})^{-1}$, the matrix inverse can be shifted from the "parameter dimension $p\times p$" to the "sample dimension $n\times n$." In batch RL settings where $B\ll p$, shifting inversion to sample space opens a new solution path.

**Core Idea**: Completely "absorb" $\bm{F}^{-1}$ into the transformation of the advantage function $A_\pi(s,a)$. NPG is written as $\bm{H}^\top\bm{\Sigma}\tilde{\bm{y}}$, where the only difference from standard PG is the advantage $\tilde{\bm{y}}=(\lambda\bm{I}_n+\bm{H}\bm{H}^\top\bm{\Sigma})^{-1}\bm{y}$. This transformation is then iteratively approximated on on-policy mini-batches using randomized block Kaczmarz.

## Method

### Overall Architecture

Let $n=|\mathcal{S}||\mathcal{A}|$ denote the "sample space dimension" and $p=|\bm{\theta}|$ the "parameter dimension." The rows of $\bm{H}\in\mathbb{R}^{n\times p}$ are $\partial_{\bm{\theta}}\log\pi$, $\bm{y}\in\mathbb{R}^n$ is the advantage vector, and $\bm{\Sigma}$ is the diagonal weighting matrix for $d_\pi(s)\pi(a|s)$. The pipeline follows three steps:

1.  **Rewriting**: Through two Woodbury transformations, Tikhonov-regularized NPG $\nabla^{\text{T-NPG}}=(\lambda\bm{I}_p+\bm{H}^\top\bm{\Sigma}\bm{H})^{-1}\bm{H}^\top\bm{\Sigma}\bm{y}$ is proven equivalent to $\bm{H}^\top\bm{\Sigma}\tilde{\bm{y}}$, where $\tilde{\bm{y}}=(\lambda\bm{I}_n+\bm{H}\bm{H}^\top\bm{\Sigma})^{-1}\bm{y}$. The matrix to be inverted shrinks from $p\times p$ to $n\times n$.
2.  **Approximation**: Since $n$ remains large in continuous action spaces, the $\bm{\Sigma}$ weighting is approximated via Monte Carlo sampling. Solving for $\tilde{\bm{y}}$ is rewritten as a regularized least squares problem: $\min_{\bm{g}}\|\bm{y}-\bm{H}\bm{g}\|_{\bm{\Sigma}}^2+\lambda\|\bm{g}\|_2^2$.
3.  **Solving**: Randomized block Kaczmarz iterations are used. Each step takes a mini-batch $\tau_j$, performs a $B\times B$ inversion, and after $K$ iterations, the resulting $\tilde{A}_j(s,a)$ is plugged into a PPO-style surrogate objective $J_{\text{RAT}}(\bm{\theta})=\mathbb{E}[\frac{\pi(a|s;\bm{\theta})}{\pi_{\text{old}}(a|s)}\tilde{A}_j(s,a)]$. A standard backpropagation on this target yields the natural policy gradient direction.

### Key Designs

1.  **Woodbury-form Advantage Rewriting**:
    - **Function**: Rewrites "Inverse Fisher × Standard Gradient" as "Standard Gradient form × Transformed Advantage," making NPG structurally identical to PG in code.
    - **Mechanism**: Applying the Woodbury identity $(\bm{I}+\bm{U}\bm{V})^{-1}\bm{U}=\bm{U}(\bm{I}+\bm{V}\bm{U})^{-1}$ twice to $(\lambda\bm{I}_p+\bm{H}^\top\bm{\Sigma}\bm{H})^{-1}\bm{H}^\top\bm{\Sigma}$ yields $\nabla^{\text{T-NPG}}_{\bm{\theta}} J=\bm{H}^\top\bm{\Sigma}(\lambda\bm{I}_n+\bm{H}\bm{H}^\top\bm{\Sigma})^{-1}\bm{y}$. This is interpreted as "standard policy gradient, but the advantage $A_\pi$ is replaced by $\bar{A}_\pi=[(\lambda\bm{I}_n+\bm{H}\bm{H}^\top\bm{\Sigma})^{-1}\bm{y}]_{(s,a)}$."
    - **Design Motivation**: Originally, inversion was required on $p\times p$ (crashing with millions of parameters); it is now moved to $n\times n$. As long as the batch size is smaller than the parameter count, the cost is manageable. All curvature information is compressed into a single scalar "advantage," leaving downstream optimizers and loss structures untouched.

2.  **Randomized Block Kaczmarz for Advantage Transformation**:
    - **Function**: Decomposes the $n\times n$ inversion for $\tilde{\bm{y}}$ into $K$ iterations of $B\times B$ inversions (where mini-batch size $B\ll p$), using only standard backpropagation.
    - **Mechanism**: Starting from $\bm{g}_0$, a mini-batch $\tau_j$ is randomly sampled for a regularized projection: $\bm{g}_j\leftarrow\arg\min_{\bm{g}}\|\bm{y}_{\tau_j}-\bm{H}_{\tau_j}\bm{g}\|_2^2+\lambda\|\bm{g}-\bm{g}_{j-1}\|_2^2$. This proximal subproblem has a closed-form update: $\bm{g}_j=\bm{g}_{j-1}+\bm{H}_{\tau_j}^\top\,[(\lambda\bm{I}+\bm{H}_{\tau_j}\bm{H}_{\tau_j}^\top)^{-1}(\bm{y}_{\tau_j}-\bm{H}_{\tau_j}\bm{g}_{j-1})]$. The $B\times B$ inversion in brackets is the "randomized advantage transformation" $\tilde{A}_j$. The paper uses `torch.linalg.solve` for numerical stability; $\bm{H}_{\tau}$ is obtained via PyTorch per-sample gradients.
    - **Design Motivation**: Hard constraint projections in classic Kaczmarz are unstable with batch noise and rank-deficient $\bm{H}_\tau$. The regularized Tikhonov proximal form ensures $(\lambda\bm{I}+\bm{H}_\tau\bm{H}_\tau^\top)$ is always invertible and keeps each step close to $\bm{g}_{j-1}$, gradually "seeping" curvature info into $\bm{g}$. Unlike SPRING, RAT refines within a single on-policy data batch to avoid stale gradients.

3.  **Architecture-Agnostic Shared Actor-Critic Adaptation (Pseudo-advantages)**:
    - **Function**: Enables RAT to use a unified target for actor and critic when they share a backbone (a scenario difficult for KFAC or Guzmán-Cordero).
    - **Mechanism**: Following ACKTR, the critic is treated as a Gaussian likelihood. An additional "pseudo-advantage" (e.g., an all-ones vector) is introduced for the critic. The actor advantage $\bm{y}^\pi$ and critic pseudo-advantage $\bm{y}^V$ are concatenated for RAT iterations to obtain shared surrogate losses. Gradients are handled via autograd without manual partitioning or weighting. Stability is ensured via $\ell_2$-norm gradient clipping $\alpha_j=\min(\eta,\nu/\|\bm{g}_j\|_2)$.
    - **Design Motivation**: KFAC-style methods typically calculate separate curvatures for each head, which is architecture-dependent. RAT places both actor and critic into the Woodbury-transformed "standard PG framework," making curvature handling via autograd "free."

### Loss & Training

The total objective is $J_{\text{RAT}}(\bm{\theta})=\mathbb{E}_{(s,a)\sim\mathcal{D}_k}\!\left[\frac{\pi(a|s;\bm{\theta})}{\pi_{\text{old}}(a|s)}\tilde{A}_j(s,a)\right]$, sharing PPO's importance sampling ratio structure. Within each on-policy rollout, $K$ inner Kaczmarz iterations refresh $\tilde{A}_j$, each corresponding to one standard backprop. The Tikhonov coefficient $\lambda$ ensures invertibility and controls convergence speed.

Theoretical results include: Under the "compatible advantage" assumption, $\mathbb{E}\|\bm{g}_j-\bm{g}^*\|_2^2\le(1-\mu)^j\|\bm{g}_0-\bm{g}^*\|_2^2$ (linear convergence, Theorem 1). With noise, an $\eta^2/\mu$ error floor exists (Theorem 2), justifying the use of gradient norm clipping.

## Key Experimental Results

### Main Results

MuJoCo continuous control (separate actor-critic) final rewards mean ± stderr (5 seeds, 10M steps), comparing PPO, TRPO/FVP+CG, KFAC, and Sophia. Key results for shared actor-critic are shown below:

| Task | State×Action | RAT (Ours) | ACKTR | PPO | Sophia |
|------|-----------|-----------|-------|------|--------|
| Swimmer | 8×2 | **271.6 ± 36.3** | 59.1 ± 13.0 | 191.3 ± 32.7 | 57.9 ± 5.9 |
| HalfCheetah | 17×6 | **4629.2 ± 287.4** | 3630.9 ± 282.6 | 4146.0 ± 107.5 | 899.5 ± 113.2 |
| Ant | 105×8 | **2926.6 ± 353.1** | 23.4 ± 3.2 | 1373.9 ± 26.0 | -7.0 ± 1.4 |
| Humanoid | 376×17 | **5382.7 ± 117.3** | 2571.7 ± 838.7 | 5357.9 ± 150.9 | 669.4 ± 56.2 |
| HumanoidStandup | 376×17 | **146529.7 ± 2317.6** | 127928.5 ± 5433.7 | 130014.2 ± 6463.7 | 111212.6 ± 13449.9 |

On high-dimensional Procgen (ResNet policy, 8 discrete tasks), RAT matches or exceeds baselines in all tasks. In low-dimensional parameter estimation visualizations, RAT's gradient directions nearly overlap with analytical natural gradients, whereas vanilla PG remains perpendicular to contours.

### Ablation Study (Wallclock time per step in ms)

| Method (HalfCheetah / Ant / Humanoid) | Separate Mode | Shared Mode | Description |
|--------|---------------|--------------|------|
| RAT (Ours) | 9.83 / 10.04 / 18.17 | 11.53 / 11.66 / 19.85 | ~2× faster than FVP+CG; supports shared backbone |
| FVP+CG (TRPO) | 19.86 / 19.95 / 19.81 | N/A | High CG inner loop overhead |
| KFAC / ACKTR | 5.60 / 5.61 / 6.57 | 6.92 / 6.85 / 7.87 | Fast but precision depends on architecture |
| Sophia (diag Fisher) | 3.92 / 3.98 / 5.71 | 5.97 / 6.03 / 7.58 | Fastest but worst rewards |
| PPO (vanilla) | 3.12 / 3.18 / 3.22 | 3.70 / 3.70 / 3.72 | Speed upper bound reference |

### Key Findings

- **Balance of Precision and Overhead**: RAT step time is roughly half of FVP+CG and twice that of KFAC, but it outperforms both in terms of reward. Sophia's diagonal approximation is fastest but collapses on Ant/Humanoid, proving that discarding off-diagonal curvature is unacceptable.
- **Significant Gains in Large Action Spaces**: In high-dimensional scenarios like Ant (105×8) and Humanoid (376×17), where ACKTR often plateaus or regresses, RAT shows stable improvement. This confirms the Theorem 1 analysis where $\mu$ governs convergence—RAT remains effective even as matrices become ill-conditioned.
- **Shared Actor-Critic is a Killer Feature**: While KFAC requires manual Fisher partitioning for shared networks, RAT's pseudo-advantage makes shared configurations "free" and at least as strong as separate ones in all tasks.

## Highlights & Insights

- **"Inversion Relocation" is the True Insight**: While most NPG improvements focus on *how* to solve $\bm{F}^{-1}$, this paper uses Woodbury to move inversion from the parameter dimension to the sample dimension. This is not just a math trick but a framework shift: Curvature info need not reside in a matrix; it can be stored in "transformed scalar advantages," making it naturally compatible with advantage-based interfaces (PPO, A2C, GAE).
- **Underestimated "Engineering-Friendliness" of Hiding Curvature in Advantages**: Modern RL frameworks (Stable-Baselines, CleanRL) are built around "calculate advantage → backprop." KFAC needs optimizer changes; TRPO needs CG loops; ACKTR needs manual head splits. RAT only modifies the advantage calculation function, allowing nearly zero-intrusion updates to any PPO implementation.
- **Reuse via Per-sample Gradient × NTK Perspective**: The authors identify $\bm{H}\bm{H}^\top$ as the Neural Tangent Kernel (NTK). This means NTK approximations (Random Features, Nyström) can directly reduce RAT's inner inversion cost, leaving a clear optimization path for scaling to LLM-level actor-critic training (e.g., RLHF).

## Limitations & Future Work

- **Sample vs. Parameter Relationship**: The method assumes efficiency gains only when $B\ll p$. If future models use tiny parameters with massive batches, $B\times B$ inversion could become a bottleneck.
- **Off-policy and Replay Buffer Adaptation**: Randomized Kaczmarz convergence relies on the on-policy distribution $d_\pi(s,a)$. Scaling to SAC or IMPALA requires addressing $\bm{\Sigma}$ re-weighting and sampling bias.
- **Realism of Theoretical Assumptions**: Theorem 1 assumes $\bm{H}$ is full rank ($p\ll n$), which is often violated with large policies and finite batches. Theorem 2's "error floor" addresses this partially, but more analysis of the $\eta^2/\mu$ magnitude and its relation to clipping threshold $\nu$ is needed.
- **Future Directions**: (1) Replacing inner Kaczmarz with variance reduction methods like SVRG/SARAH; (2) Adaptive $\lambda$ based on Fisher spectra; (3) Fusion with momentum variants like SPRING for ill-conditioned tasks.

## Related Work & Insights

- **vs FVP+CG (TRPO, Schulman 2015)**: Neither construct Fisher explicitly, but FVP+CG solves inversion in "parameter space CG" with dozens of loops. RAT uses "sample space Kaczmarz" with one $B\times B$ inversion + backprop, natively supporting shared actor-critic.
- **vs KFAC / ACKTR (Wu 2017)**: KFAC assumes layer-wise Kronecker decomposition and statistical independence. RAT makes no structural assumptions about $\bm{F}$, allowing easy migration to Transformer or CNN policies by changing only the per-sample gradient implementation.
- **vs Guzmán-Cordero et al. 2025**: Also uses Woodbury for NPG but approximates the Fisher inverse directly and requires manual gradient merging. RAT delegates the solution to a randomized iterator and unified autograd via pseudo-advantages.
- **vs SPRING (Goldshlager 2024)**: Both use Kaczmarz, but SPRING uses momentum, updates once per batch, and accumulates $\bm{g}$ across rollouts. RAT iterates multiple times within a single rollout without momentum, avoiding stale gradients in on-policy RL.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of Woodbury, randomized Kaczmarz, and advantage transformation is novel and elegantly reduces NPG to standard PG form.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers MuJoCo, Procgen, and low-dim Gaussian; however, lacks validation on LLM-level RLHF.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear step-by-step derivation, two convergence theorems, and explicit comparisons with KFAC/FVP/SPRING make it easily reproducible.
- Value: ⭐⭐⭐⭐⭐ Provides a nearly zero-intrusion natural gradient scheme for existing PPO codebases, opening doors for natural gradients in large-scale RL/RLHF.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DiffOP: Reinforcement Learning of Optimization-Based Control Policies via Implicit Policy Gradients](../../AAAI2026/reinforcement_learning/diffop_reinforcement_learning_of_optimization-based_control_policies_via_implici.md)
- [\[ICLR 2026\] Learning to Orchestrate Agents in Natural Language with the Conductor](../../ICLR2026/reinforcement_learning/learning_to_orchestrate_agents_in_natural_language_with_the_conductor.md)
- [\[ICML 2026\] RL4RLA: Teaching ML to Discover Randomized Linear Algebra Algorithms Through Curriculum Design and Graph-Based Search](rl4rla_teaching_ml_to_discover_randomized_linear_algebra_algorithms_through_curr.md)
- [\[ACL 2026\] Free Energy-Driven Reinforcement Learning with Adaptive Advantage Shaping for Unsupervised Reasoning in LLMs](../../ACL2026/reinforcement_learning/free_energy-driven_reinforcement_learning_with_adaptive_advantage_shaping_for_un.md)
- [\[ICML 2026\] ProRL: Effective Reinforcement Learning for Proactive Recommendation via Rectified Policy Gradient Estimation](prorl_effective_reinforcement_learning_for_proactive_recommendation_via_rectifie.md)

</div>

<!-- RELATED:END -->
