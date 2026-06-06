---
title: >-
  [Paper Note] Sharp Description of Local Minima in the Loss Landscape of High-Dimensional Two-Layer ReLU Networks
description: >-
  [ICML2026][Optimization][Loss landscape] Under the setting of high-dimensional Gaussian inputs for teacher-student two-layer ReLU networks…
tags:
  - "ICML2026"
  - "Optimization"
  - "Loss landscape"
  - "ReLU two-layer networks"
  - "summary statistics"
  - "overparameterization"
  - "fixed points"
date: 2026-05-08
content_hash: bb977c43c2de859a
---

# Sharp Description of Local Minima in the Loss Landscape of High-Dimensional Two-Layer ReLU Networks

**Conference**: ICML2026  
**arXiv**: [2604.09412](https://arxiv.org/abs/2604.09412)  
**Code**: TBD  
**Area**: Optimization Theory / Neural Network Landscape / Mean-field Analysis  
**Keywords**: Loss landscape, ReLU two-layer networks, summary statistics, overparameterization, fixed points  

## TL;DR
Under the setting of high-dimensional Gaussian inputs for teacher-student two-layer ReLU networks, this paper provides a hierarchical classification of all local minima of the population loss using a set of exact low-dimensional summary statistic equations regarding weight overlaps $(Q,R)$. It characterizes how overparameterization transforms low-order spurious minima into saddle points while preserving high-order minima, thereby reconciling the existence results of Safran–Shamir, the group-theoretic classification of Arjevani–Field, and the Hessian instability arguments of Safran et al. for the first time.

## Background & Motivation

**Background**: Training two-layer ReLU networks $\sum_{k=1}^{K}\mathrm{ReLU}(w_k^\top x)$ is a non-convex optimization problem, yet it almost always converges in practice. This gap between "non-convexity" and "good optimizability" has driven extensive landscape theory work. Two main approaches exist: (i) Mean-field limits—proving that the landscape is asymptotically benign in the infinite-width limit (Chizat–Bach, Mei et al.); (ii) Counterexamples and algebraic characterizations in the finite-width regime—Safran–Shamir used computer-aided proofs to show the existence of spurious local minima, while Arjevani–Field used group theory to show these minima follow the "Principle of Minimal Symmetry Breaking."

**Limitations of Prior Work**: Mean-field results do not hold directly for any finite width and do not specify at what width the benign property takes effect. Safran–Shamir only provides existence proofs without describing the global structure of the landscape. Subsequent work by Safran et al. used local Hessians to argue that adding one neuron can turn spurious minima into saddle points, but experiments clearly still show high-order spurious minima—implying that the local Hessian perspective misses critical mechanisms.

**Key Challenge**: Existing tools are either purely asymptotic or purely local, leaving quantitative questions unanswered: why does overparameterization help convergence, to what extent does it help, and what traps remain? Particularly, when ReLU is non-differentiable, Hessian-based arguments naturally fail.

**Goal**: (1) Provide an exact low-dimensional algebraic characterization of the population loss $\mathcal{L}(W;W^*)=\frac{1}{2}\mathbb{E}_x[(\phi(x,W)-\phi(x,W^*))^2]$; (2) Use this characterization to hierarchically classify all spurious minima using a discrete integer index; (3) Explain how overparameterization simultaneously "eliminates some traps" while "preserving others."

**Key Insight**: Drawing from the soft committee machine tradition in statistical physics, the weight overlaps $Q_{ij}=\frac{1}{d}w_i^\top w_j$, $R_{im}=\frac{1}{d}w_i^\top w_m^*$, and $T_{mn}=\frac{1}{d}{w_m^*}^\top w_n^*$ are introduced as sufficient statistics. Under the orthogonal teacher assumption $T=I_M$, the entire population loss and the fixed-point structure of the gradient flow can be written in closed-form based on $(Q,R)$.

**Core Idea**: The fixed-point conditions of the ReLU network $\mathcal{F}_R(Q,R)=0, \mathcal{F}_Q(Q,R)=0$ are projected onto a block-symmetric ansatz. This allow each family of minima to be fully characterized by a single integer $k_1$ (the number of student neurons anti-aligned with the teacher), thereby reducing the continuous non-convex landscape to a 1D discrete family.

## Method

### Overall Architecture
The teacher is $\phi(x,W^*)=\sum_{m=1}^M\mathrm{ReLU}(\frac{{w_m^*}^\top x}{\sqrt d})$, and the student is $\phi(x,W)=\sum_{k=1}^K\mathrm{ReLU}(\frac{w_k^\top x}{\sqrt d})$, where $x\sim\mathcal{N}(0,I_d)$. The population gradient flow is $\dot w_k=-\eta\mathbb{E}_x[\mathcal{G}_k]$, where $\mathcal{G}_k=(\phi(x,W)-\phi(x,W^*))H(\frac{w_k^\top x}{\sqrt d})\frac{x}{\sqrt d}$ and $H$ is the Heaviside function. The authors use three steps: (i) Project weight dynamics into ODEs of $(Q,R)$ and write the closed-form Gaussian expectations for ReLU; (ii) Solve for all fixed points satisfying $\mathcal{F}_Q=\mathcal{F}_R=0$, reducing high-dimensional algebraic equations to a few scalars using the block-symmetric ansatz; (iii) Use perturbation analysis (replacing the Hessian for non-differentiable ReLU) to determine the stability of each fixed point, complemented by statistical analysis of the frequency of being attracted to each family via $10^4$ random initialization ODE simulations.

### Key Designs

1.  **Summary Statistic ODEs and Fixed-Point Conditions**:
    - **Function**: Collapses the $Kd$-dimensional weight trajectories into a closed system of only $K^2+KM$ scalars, making the population landscape fully amenable to algebraic analysis.
    - **Mechanism**: Defining $Q_{ij}=\frac{1}{d}w_i^\top w_j$ and $R_{im}=\frac{1}{d}w_i^\top w_m^*$, $\mathbb{E}_x[\mathcal{G}_k]$ under ReLU can be expressed as polynomials and inverse trigonometric functions of $(Q,R,T)$ via binary/ternary/quaternary Gaussian expectations (Appendix A.4 provides closed forms). Thus, gradient flow is equivalent to $\dot Q=\mathcal{F}_Q(Q,R)$ and $\dot R=\mathcal{F}_R(Q,R)$. Fixed points satisfy $\mathcal{F}_R(Q,R)=0, \mathcal{F}_Q(Q,R)=0$ (Result 1), and these equations are independent of the input dimension $d$.
    - **Design Motivation**: Traditional local analysis struggles with geomtric objects in $\mathbb{R}^{Kd}$ as $d\to\infty$. Condensing dimensions to $O(K^2+KM)$ preserves all generalization-relevant information (as loss is a function of $(Q,R)$) while transforming the search for all minima into a solvable algebraic problem.

2.  **Block-Symmetric Ansatz and $k_1$ Hierarchy**:
    - **Function**: Groups all local minima into families based on a discrete integer $k_1\in[0,M]$ and provides analytical values for loss and overlaps for each family.
    - **Mechanism**: Leveraging the permutation symmetry of student hidden units, $K$ neurons are divided into two groups: $|I_1|=k_1$ neurons anti-aligned with the teacher ($R_{im}<0$), and $|I_2|=k_2=K-k_1$ neurons aligned. Under this ansatz, $R$ and $Q$ take block forms: each block is parameterized by $\mathbf{B}(x,y)=xI+y(J-I)$. The original coupled equations reduce to a few scalar equations regarding $\{r_1^{\mathrm{diag}},r_1^{\mathrm{off}},q_1^{\mathrm{diag}},\dots\}$ (Result 2). Each $k_1$ yields a family of spurious minima with a corresponding analytical loss and $(Q,R)$ template.
    - **Design Motivation**: Directly searching for zeros in the $(Q,R)$ space is an $O(K^2)$-dimensional algebraic problem. Symmetry reduces each family to $O(1)$ unknowns. This block structure is the macroscopic counterpart to the "minimal symmetry breaking" principle in Arjevani–Field's theory—the local error caused by anti-aligned neurons is exactly compensated by direction adjustments of aligned neurons, causing the gradient to vanish.

3.  **Perturbative Stability Analysis and Overparameterization Diagnosis**:
    - **Function**: Determines the stability of each fixed point family given the non-differentiability of ReLU and quantitatively explains how overparameterization destabilizes first-order families ($k_1=1$) while preserving high-order ones ($k_1\ge 2$).
    - **Mechanism**: The system is initialized at a fixed point, perturbed by $\xi\sim\mathcal{N}(0,\sigma^2 I)$, and followed by 1000 steps of GD ($\eta=0.01$) to measure the mean return distance. In the well-specified case ($K=M$), the system returns to $<10^{-3}$ even for large $\sigma$. In the overparameterized case ($K\ge M+1$), even minimal $\sigma$ pushes the system away. Combined with the generalized analysis of the ansatz for $K=M+1$, the authors formally prove that the fixed-point equations for $k_1=1$ no longer have stable real solutions, whereas $k_1\ge 2$ high-order families persist and are not simply produced by zero-padding.
    - **Design Motivation**: While ReLU lacks a computable Hessian, the population gradient flow is well-defined. Perturbation analysis bypasses non-differentiability and directly identifies which families truly trap SGD, serving as a natural diagnostic companion to the ansatz method.

### Loss & Training
The loss is $\mathcal{L}(W;W^*)=\frac{1}{2}\mathbb{E}_x[(\phi(x,W)-\phi(x,W^*))^2]$. Optimization uses population gradient flow, extended to normalized GD (on the sphere $\|w_k\|^2=d$), orthonormalized GD (Stiefel manifold $WW^\top=dI_K$), two-layer joint GD, and one-pass online SGD (Result 3 shows equivalence to GF when $\eta=o_d(1)$).

## Key Experimental Results

### Main Results: Frequency of Reaching Global Minima Under Different Overparameterization ($10^4$ Random Initializations, Orthogonal Teacher)

| Optimizer | $K=17,M=17$ | $K=18,M=17$ | $K=19,M=17$ |
| :--- | :--- | :--- | :--- |
| Gradient Descent | 13.25% | 64.18% | 77.50% |
| Two-layer Joint GD (2L-GD) | 13.24% | 67.91% | **99.48%** |
| Normalized GD | 14.12% | 58.35% | No Convergence |
| Orthonormalized GD | No Convergence | No Convergence | No Convergence |

### Ablation Study / Family Distribution: Frequency of GF Converging to Anti-aligned Neurons Count $k_1$ ($10^4$ Runs)

| Minima Order $k_1$ | $K=17,M=17$ | $K=18,M=17$ | $K=19,M=17$ |
| :--- | :--- | :--- | :--- |
| $k_1=0$ (Global Minima) | 13.09% | 59.29% | 99.63% |
| $k_1=1$ | 27.52% | 0.00% | 0.00% |
| $k_1=2$ | 29.05% | 2.10% | 0.05% |
| $k_1=3$ | 18.94% | 10.83% | 0.31% |
| $k_1=4$ | 7.55% | 8.99% | 0% |

### Key Findings
- In the well-specified case ($K=M$), the loss distribution is strictly "quantized" into several discrete plateaus, with each plateau's position accurately predicted by the analytical formula in Result 2 (the dashed lines in Figure 1b align with the histogram).
- Adding one neuron completely eliminates the $k_1=1$ family (frequency drops from 27.52% to 0), consistent with the instability of this family under $K=M+1$ in perturbation diagnosis. However, $k_1\ge 2$ families persist with non-zero frequency and cannot be derived via zero-padding $K=M$ solutions—they are new coupled solutions in the overparameterized space.
- Orthonormalized GD fails because the orthogonality constraint prohibits "amplitude compensation by aligned neurons," thus the typical spurious families of ReLU networks do not exist; the trade-off is extremely slow convergence, failing to converge within $1.2\times 10^7$ steps.
- Result 3 shows that one-pass SGD trajectories are consistent with GF under the $\eta=o_d(1)$ scaling, implying all landscape conclusions apply to common SGD settings.

## Highlights & Insights
- Using low-dimensional summary statistics + block ansatz transforms the problem of "all minima in a non-convex landscape" into a set of hand-solvable scalar equations. This is a rare finite-width ReLU landscape analysis that provides global structure—more precise than pure mean-field theory (lacks quantification) and pure local Hessian analysis (misses high-order families).
- Rewriting Arjevani–Field's discrete group-theoretic classification, Fukumizu–Amari’s symmetry-breaking plateaus, and Safran et al.'s Hessian instability theory using the same $(Q,R)$ ansatz unifies three independent toolsets into a single framework.
- Incorporating normalized/orthonormalized/two-layer GD into the same ODE system reveals a counter-intuitive phenomenon: preserving more degrees of freedom (unconstrained) makes it easier to escape spurious minima compared to spherical/Stiefel constraints. This challenges the "contrained optimization is more stable" convention and offers a reusable insight for deep network analysis.

## Limitations & Future Work
- Limited to two-layer ReLU with single-layer training (standard GD) or two-layer joint training. While ODEs for deep structures and non-ReLU activations (Leaky ReLU, erf in Appendix E) are provided, large-scale experiments are missing.
- Assumes Gaussian inputs and orthogonal teachers $T=I_M$. Whether the ansatz remains precise for structured inputs or ill-conditioned teachers lacks quantitative bounds.
- Does not characterize the "basin size" of each minima—only "sampled frequency" is provided, which cannot determine which initializations avoid $k_1\ge 2$ families.
- For real-world engineering SGD settings with large mini-batches or $\eta=\Theta(1)$, the equivalence in Result 3 no longer holds, requiring new diffusion term analysis.

## Related Work & Insights
- **vs Safran–Shamir 2018**: They provided computer-aided existence proofs for spurious minima; this paper analytically parameterizes all such minima into $k_1$-families and explains their fate under overparameterization.
- **vs Safran et al. 2021**: They used Hessians to argue that minima become saddle points under overparameterization but ignored that $k_1\ge 2$ remains alive. This paper uses the ansatz + perturbation to provide a complete picture, correcting the overly optimistic "overparameterization is benign" view.
- **vs Mean-field (Chizat–Bach / Mei et al.)**: Mean-field theory only guarantees global convergence as $K\to\infty$. This paper provides the discrete family structure of the landscape for finite $K$ and quantitatively describes when the benign property begins to take effect, filling the explanatory gap between mean-field and finite-width regimes.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Unifies different ReLU landscape analysis schools into one ansatz and precisely classifies all spurious families.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ $10^4$ simulations cover multiple optimizers and overparameterization levels, though limited to two-layer/Gaussian/orthogonal teacher settings.
- **Writing Quality**: ⭐⭐⭐⭐ The narration of Results 1–3 and Figures 1–4 is very self-contained, and appendix contents are clearly labeled.
- **Value**: ⭐⭐⭐⭐ Provides the first quantitative, visual, and reproducible finite-width characterization of how width "benignizes" the landscape.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Directional Convergence, Benign Overfitting of Gradient Descent in leaky ReLU two-layer Neural Networks](../../ICLR2026/optimization/directional_convergence_benign_overfitting_of_gradient_descent_in_leaky_relu_two.md)
- [\[AAAI 2026\] On the Learning Dynamics of Two-Layer Linear Networks with Label Noise SGD](../../AAAI2026/optimization/on_the_learning_dynamics_of_two-layer_linear_networks_with_label_noise_sgd.md)
- [\[ICML 2026\] Taming the Loss Landscape of PINNs with Noisy Feynman-Kac Supervision: Operator Preconditioning and Non-Asymptotic Error Bounds](taming_the_loss_landscape_of_pinns_with_noisy_feynman-kac_supervision_operator_p.md)
- [\[ICLR 2026\] Rolling Ball Optimizer: Learning by Ironing Out Loss Landscape Wrinkles](../../ICLR2026/optimization/rolling_ball_optimizer_learning_by_ironing_out_loss_landscape_wrinkles.md)
- [\[ICML 2026\] Dynamics and Representation Structure of Local Approximations to Gradient-Based Learning in Linear Recurrent Neural Networks](dynamics_and_representation_structure_of_local_approximations_to_gradient-based_.md)

</div>

<!-- RELATED:END -->
