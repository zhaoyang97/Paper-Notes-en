---
title: >-
  [Paper Note] Sharp Description of Local Minima in the Loss Landscape of High-Dimensional Two-Layer ReLU Networks
description: >-
  [ICML2026][Optimization][Loss Landscape] Under the high-dimensional Gaussian input setting for teacher-student two-layer ReLU networks, this paper provides a hierarchical classification of all local minima of the population loss using a set of exact low-dimensional summary statistics equations regarding weight overlaps $(Q,R)$. It characterizes how over-parameterization transforms low-order spurious minima into saddle points while retaining high-order minima…
tags:
  - "ICML2026"
  - "Optimization"
  - "Loss Landscape"
  - "ReLU two-layer networks"
  - "Summary statistics"
  - "Over-parameterization"
  - "Fixed point"
date: 2026-05-08
content_hash: 05767ab453fe7628
---

# Sharp Description of Local Minima in the Loss Landscape of High-Dimensional Two-Layer ReLU Networks

**Conference**: ICML2026  
**arXiv**: [2604.09412](https://arxiv.org/abs/2604.09412)  
**Code**: To be confirmed  
**Area**: Optimization Theory / Neural Network Landscape / Mean-Field Analysis  
**Keywords**: Loss Landscape, ReLU two-layer networks, Summary statistics, Over-parameterization, Fixed point  

## TL;DR
Under the high-dimensional Gaussian input setting for teacher-student two-layer ReLU networks, this paper provides a hierarchical classification of all local minima of the population loss using a set of exact low-dimensional summary statistics equations regarding weight overlaps $(Q,R)$. It characterizes how over-parameterization transforms low-order spurious minima into saddle points while retaining high-order minima, thereby reconciling Safran–Shamir’s existence results, Arjevani–Field’s group-theoretic classification, and Safran et al.’s Hessian instability theory for the first time.

## Background & Motivation

**Background**: Training two-layer ReLU networks $\sum_{k=1}^{K}\mathrm{ReLU}(w_k^\top x)$ is a non-convex optimization problem that almost always converges in practice. This gap between "non-convexity" and "ease of optimization" has driven significant landscape theory research. Two main research lines exist: (i) Mean-field limits—proving that the landscape is asymptotically benign under infinite width (Chizat–Bach, Mei et al.); (ii) Finite-width counterexamples and algebraic characterizations—Safran–Shamir used computer-aided proofs to show the existence of spurious local minima, while Arjevani–Field used group theory to explain that these minima follow the "Principle of Minimal Symmetry Breaking."

**Limitations of Prior Work**: Mean-field results do not hold directly for any finite width and fail to specify at what width a landscape becomes "benign." Safran–Shamir only provided existence proofs without describing the global landscape structure. Subsequent work by Safran et al. used local Hessian analysis to argue that adding a single neuron turns spurious minima into saddle points, yet experiments clearly still observe high-order spurious minima—implying the local Hessian perspective misses underlying mechanisms.

**Key Challenge**: Existing tools are either purely asymptotic or purely local, leaving quantitative questions such as "why over-parameterization helps, to what extent, and which traps remains" unanswered. Notably, in the non-differentiable case of ReLU, Hessian-based arguments naturally fail.

**Goal**: (1) Provide an exact low-dimensional algebraic characterization of the population loss $\mathcal{L}(W;W^*)=\frac{1}{2}\mathbb{E}_x[(\phi(x,W)-\phi(x,W^*))^2]$; (2) Use this characterization to hierarchically classify all spurious minima by a discrete integer index; (3) Explain how over-parameterization simultaneously "eliminates some traps" while "retaining others."

**Key Insight**: Following the soft committee machine tradition from statistical physics, the paper introduces weight overlaps $Q_{ij}=\frac{1}{d}w_i^\top w_j$, $R_{im}=\frac{1}{d}w_i^\top w_m^*$, and $T_{mn}=\frac{1}{d}{w_m^*}^\top w_n^*$ as sufficient statistics. Under the orthogonal teacher assumption $T=I_M$, the entire population loss and the fixed-point structure of the gradient flow can be written in closed-form based on $(Q,R)$.

**Core Idea**: The fixed-point conditions $\mathcal{F}_R(Q,R)=0, \mathcal{F}_Q(Q,R)=0$ for ReLU networks are projected onto a block-symmetric ansatz. This allows each family of minima to be fully characterized by a single integer $k_1$ (the number of student neurons anti-aligned with the teacher), reducing the continuous non-convex landscape to a one-dimensional discrete family.

## Method

### Overall Architecture
Teacher $\phi(x,W^*)=\sum_{m=1}^M\mathrm{ReLU}(\frac{{w_m^*}^\top x}{\sqrt d})$, student $\phi(x,W)=\sum_{k=1}^K\mathrm{ReLU}(\frac{w_k^\top x}{\sqrt d})$, with $x\sim\mathcal{N}(0,I_d)$. Population gradient flow $\dot w_k=-\eta\mathbb{E}_x[\mathcal{G}_k]$, where $\mathcal{G}_k=(\phi(x,W)-\phi(x,W^*))H(\frac{w_k^\top x}{\sqrt d})\frac{x}{\sqrt d}$ and $H$ is the Heaviside step function. The authors employ three steps: (i) Project weight dynamics into an ODE of $(Q,R)$ and derive closed-form Gaussian expectations for ReLU; (ii) Solve for all fixed points satisfying $\mathcal{F}_Q=\mathcal{F}_R=0$ using a block-symmetric ansatz to reduce high-dimensional algebraic equations to a few scalars; (iii) Perform perturbation analysis (replacing Hessian for non-differentiable ReLU) to judge fixed-point stability, complemented by $10^4$ ODE simulations with random initializations to count attraction frequencies to each family.

### Key Designs

**1. Summary Statistics ODE and Fixed Point Conditions: Collapsing $Kd$-Dimensional Weight Trajectories into a Dimension-Independent Closed System**

Traditional local analysis faces geometric objects in $\mathbb{R}^{Kd}$ as $d \to \infty$, making it intractable. Following the soft committee machine tradition, weight overlaps $Q_{ij}=\frac{1}{d}w_i^\top w_j$ and $R_{im}=\frac{1}{d}w_i^\top w_m^*$ are used as sufficient statistics. For ReLU, the population gradient $\mathbb{E}_x[\mathcal{G}_k]$ can be expressed via bivariate/trivariate/quadrivariate Gaussian ReLU expectations as polynomials and inverse trigonometric functions of $(Q,R,T)$ (Appendix A.4 provides closed forms). Thus, gradient flow is equivalent to:

$$\dot Q=\mathcal{F}_Q(Q,R),\qquad \dot R=\mathcal{F}_R(Q,R),$$

where fixed points satisfy $\mathcal{F}_R(Q,R)=0, \mathcal{F}_Q(Q,R)=0$ (Result 1). This system is independent of the input dimension $d$. This approach preserves all generalization-relevant information (as loss is a function of $(Q,R)$) while reducing the search for minima from high-dimensional geometry to an algebraic problem of $O(K^2+KM)$ scalars.

**2. Block-symmetric Ansatz and $k_1$ Hierarchy: Reducing the Continuous Non-convex Landscape to One-dimensional Discrete Families**

Even with reduction to $(Q,R)$, searching for zeros remains an $O(K^2)$ algebraic problem. Leveraging permutation symmetry of student hidden units, the authors divide $K$ neurons into two groups: $|I_1|=k_1$ neurons anti-aligned with teachers ($R_{im}<0$) and $|I_2|=K-k_1$ neurons aligned. Under this ansatz, $R$ and $Q$ take block forms parameterized by $\mathbf{B}(x,y)=xI+y(J-I)$, and the coupled equations degenerate into a few scalar equations for $\{r_1^{\mathrm{diag}},r_1^{\mathrm{off}},q_1^{\mathrm{diag}},\dots\}$ (Result 2). Each family of spurious minima is fully characterized by a single integer $k_1\in[0,M]$. This structure macroscopically corresponds to Arjevani–Field’s "minimal symmetry breaking" principle: local errors from anti-aligned neurons are exactly compensated by directional adjustments of aligned ones, zeroing the gradient and trapping the optimization.

**3. Perturbation-based Stability Analysis and Over-parameterization Diagnosis: Replacing Hessian for Non-differentiable ReLU**

While ReLU lacks a Hessian, the population gradient flow is well-defined. The authors use perturbation analysis for stability: the system is initialized at a fixed point, weights are perturbed by $\xi\sim\mathcal{N}(0,\sigma^2 I)$, and GD is run for 1000 steps ($\eta=0.01$) to measure the return distance. In the well-specified case ($K=M$), the system returns even with large $\sigma$. In over-parameterized cases ($K \ge M+1$), even tiny $\sigma$ pushes the system away. Combined with the ansatz generalization for $K=M+1$, the authors formally prove that fixed-point equations for $k_1=1$ no longer have stable real solutions, while high-order $k_1 \ge 2$ families persist and are not simply zero-padded $K=M$ solutions. This diagnosis corrects the optimism of Safran et al. (that over-parameterization turns all minima into saddles) by identifying which families actually trap SGD.

### Loss & Training
Loss is $\mathcal{L}(W;W^*)=\frac{1}{2}\mathbb{E}_x[(\phi(x,W)-\phi(x,W^*))^2]$. Optimization uses population gradient flow, extended to normalized GD (on the sphere $\|w_k\|^2=d$), orthonormalized GD (Stiefel manifold $WW^\top=dI_K$), two-layer joint GD, and one-pass online SGD (Result 3 shows equivalence to GF when $\eta=o_d(1)$).

## Key Experimental Results

### Main Results: Frequency of Reaching Global Minima under Different Over-parameterization ($10^4$ trials, Orthogonal Teacher)

| Optimizer | $K=17,M=17$ | $K=18,M=17$ | $K=19,M=17$ |
|-----------|-------------|-------------|-------------|
| Gradient Descent | 13.25% | 64.18% | 77.50% |
| Two-layer Joint GD (2L-GD) | 13.24% | 67.91% | **99.48%** |
| Normalized GD | 14.12% | 58.35% | Failed to converge |
| Orthonormalized GD | Failed to converge | Failed to converge | Failed to converge |

### Ablation Study: Convergence Frequency to Each $k_1$ Family (10^4 GF runs)

| Order of Minima $k_1$ | $K=17,M=17$ | $K=18,M=17$ | $K=19,M=17$ |
|-----------------------|-------------|-------------|-------------|
| $k_1=0$ (Global Min) | 13.09% | 59.29% | 99.63% |
| $k_1=1$ | 27.52% | 0.00% | 0.00% |
| $k_1=2$ | 29.05% | 2.10% | 0.05% |
| $k_1=3$ | 18.94% | 10.83% | 0.31% |
| $k_1=4$ | 7.55% | 8.99% | 0% |

### Key Findings
- Under well-specified ($K=M$) conditions, loss distribution is strictly "quantized" into discrete plateaus, with positions accurately predicted by analytical formulas in Result 2.
- Adding 1 neuron eliminates the $k_1=1$ family entirely (frequency drops from 27.52% to 0), consistent with stability diagnostics for $K=M+1$. However, $k_1 \ge 2$ families persist with non-zero frequency and are not zero-padding solutions—they represent new coupled solutions in the over-parameterized space.
- Orthonormalized GD (onGD) prevents spurious families typical of ReLU networks because orthogonal constraints forbid "amplitude compensation" by aligned neurons. However, it converges extremely slowly and fails within $1.2\times 10^7$ steps.
- Result 3 indicates that under the scaling $\eta=o_d(1)$, one-pass SGD trajectories match GF, making all landscape conclusions applicable to common SGD settings.

## Highlights & Insights
- Reducing the non-convex landscape to calculable scalar equations via low-dimensional summary statistics and block ansatz provides a rare global structure analysis of finite-width ReLU networks—offering finer detail than mean-field (lacks quantitative depth) and local Hessian (misses high-order families).
- Rewriting Arjevani–Field’s group theory, Fukumizu–Amari’s symmetry-breaking plateaus, and Safran’s Hessian instability theory under a unified $(Q,R)$ ansatz integrates three independent toolsets.
- Incorporating normalized/orthonormalized/two-layer GD reveals a counter-intuitive phenomenon: retaining more degrees of freedom (unconstrained) makes it easier to escape spurious minima compared to spherical/Stiefel constraints, challenging the convention that constrained optimization is more stable.

## Limitations & Future Work
- Limited to two-layer ReLU with either single-layer (standard GD) or joint two-layer training. Deep structures and non-ReLU activations (Leaky ReLU, erf in Appendix E) have ODEs but lack large-scale experiments.
- Assumes Gaussian inputs and orthogonal teachers $T=I_M$. The precision of the ansatz under structured inputs or ill-conditioned teachers lacks quantitative bounds.
- Basin sizes for minima are not characterized—only "sampling frequency" is provided, which does not clarify what initialization avoids $k_1 \ge 2$ families.
- For practical engineering SGD with large mini-batches and $\eta=\Theta(1)$, the equivalence in Result 3 vanishes, requiring new diffusion term analysis.

## Related Work & Insights
- **vs Safran–Shamir 2018**: They provide existence proofs for spurious minima; Ours analytically parameterizes these minima into $k_1$-families and explains their fate under over-parameterization.
- **vs Safran et al. 2021**: They argue over-parameterization turns minima into saddles via Hessian, but overlook that $k_1 \ge 2$ remains alive; Ours provides a complete picture via ansatz + perturbation, correcting excessive optimism about "benign" over-parameterization.
- **vs Mean-field (Chizat–Bach / Mei et al.)**: Mean-field ensures global convergence as $K \to \infty$; Ours provides the discrete landscape structure for finite $K$ and quantitatively describes when "benign" effects take hold, bridging the gap between mean-field and finite-width analysis.

## Rating
- Novelty: ⭐⭐⭐⭐ Unifies different ReLU landscape schools under one ansatz and precisely classifies all spurious families.
- Experimental Thoroughness: ⭐⭐⭐⭐ $10^4$ simulations across multiple optimizers and over-parameterization levels, though limited to two-layer/Gaussian settings.
- Writing Quality: ⭐⭐⭐⭐ Results 1–3 and Figures 1–4 are self-contained and clearly cross-referenced with the appendix.
- Value: ⭐⭐⭐⭐ Provides the first quantitative, visualized, and reproducible finite-width characterization of how width "benign-izes" the landscape.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Deep-ICE: The First Globally Optimal Algorithm for Minimizing 0–1 Loss in Two-Layer ReLU and Maxout Networks](../../ICLR2026/optimization/deep-ice_the_first_globally_optimal_algorithm_for_empirical_risk_minimization_of.md)
- [\[ICLR 2026\] Directional Convergence, Benign Overfitting of Gradient Descent in leaky ReLU two-layer Neural Networks](../../ICLR2026/optimization/directional_convergence_benign_overfitting_of_gradient_descent_in_leaky_relu_two.md)
- [\[CVPR 2026\] Globscope: Toward a Global View of the Loss Landscape](../../CVPR2026/optimization/globscope_toward_a_global_view_of_the_loss_landscape.md)
- [\[ICML 2026\] Taming the Loss Landscape of PINNs with Noisy Feynman-Kac Supervision: Operator Preconditioning and Non-Asymptotic Error Bounds](taming_the_loss_landscape_of_pinns_with_noisy_feynman-kac_supervision_operator_p.md)
- [\[AAAI 2026\] On the Learning Dynamics of Two-Layer Linear Networks with Label Noise SGD](../../AAAI2026/optimization/on_the_learning_dynamics_of_two-layer_linear_networks_with_label_noise_sgd.md)

</div>

<!-- RELATED:END -->
