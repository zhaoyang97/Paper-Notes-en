---
title: >-
  [Paper Note] Iterative Refinement Neural Operators are Learned Fixed-Point Solvers: A Principled Approach to Spectral Bias Mitigation
description: >-
  [ICML 2026][Physics & Scientific Computing][Neural Operators] The paper attaches a shared-weight U-Net correction module $\Phi_\theta$ to pre-trained neural operators (FNO/TFNO/WDSR, etc.). During inference…
tags:
  - "ICML 2026"
  - "Physics & Scientific Computing"
  - "Neural Operators"
  - "Fixed-Point Iteration"
  - "Spectral Bias"
  - "Inference-time Iteration"
  - "FNO"
date: 2026-05-08
content_hash: b16ded8cefea0425
---

# Iterative Refinement Neural Operators are Learned Fixed-Point Solvers: A Principled Approach to Spectral Bias Mitigation

**Conference**: ICML 2026  
**arXiv**: [2605.24041](https://arxiv.org/abs/2605.24041)  
**Code**: https://github.com/xiaotianliu-dartmouth/Iterative_Refinement_Neural_Operator (Available)  
**Area**: Scientific Computing / Neural Operators / PDE Surrogate Models  
**Keywords**: Neural Operators, Fixed-Point Iteration, Spectral Bias, Inference-time Iteration, FNO  

## TL;DR
The paper attaches a shared-weight U-Net correction module $\Phi_\theta$ to pre-trained neural operators (FNO/TFNO/WDSR, etc.). During inference, it iteratively updates the prediction via $h_{k+1}=h_k+\alpha\Phi_\theta(x,h_k)$, transforming a single-pass prediction into a "learned residual solver" that converges to a unique fixed point. This approach reduces errors by 34%–80% in tasks like turbulence, active matter, and ERA5 super-resolution, shifts the monolithic paradigm to a convergent dynamical system, and enables stable extrapolation to twice the number of training steps.

## Background & Motivation

**Background**: Neural operators such as FNO and DeepONet have become mainstream fast surrogate models for parameterized PDEs and multiphysics systems. They learn mappings $\mathcal{G}:\mathcal{X}\to\mathcal{H}$ in function spaces, providing entire solution fields in a single forward pass—orders of magnitude faster than traditional numerical methods.

**Limitations of Prior Work**: These operators commonly suffer from "spectral bias"—large-scale low-frequency structures are learned accurately, but medium-to-high frequency details (turbulent filaments, fine wind textures, orientation gradients in active matter) are significantly smoothed out. Figure 1 illustrates this in ERA5 16× super-resolution: FNO captures the general atmospheric structure but blurs small-scale kinetic energy vortices.

**Key Challenge**: Existing solutions rely on "scaling up" during training—widening models, using higher resolution data, or expanding datasets. This pushes the "monolithic forward" paradigm of regressing the entire solution at once to its limit. Conversely, classical numerical analysis offers another path: coarse solutions followed by residual iterative refinement (multigrid, defect correction, Krylov), a concept not yet systematically integrated into neural operators.

**Goal**: To transform the single forward pass into an iterative "test-time optimization" without retraining the base operator, decoupling accuracy gains from training resource consumption while providing theoretical convergence guarantees.

**Key Insight**: Reconceptualize the neural operator prediction process as a dynamical system in function space. A base operator provides a coarse initial value $h_0$, and a shared-weight correction operator $\Phi_\theta$ iteratively computes residual corrections. This corresponds to fixed-point iteration $h_{k+1}=T(h_k)$ in numerical analysis, allowing the use of the Banach Fixed-Point Theorem to prove convergence, extrapolation stability, and error lower bounds.

**Core Idea**: Replace the "monolithic single forward pass" with "learned residual iteration" to gradually eliminate spectral bias through repeated refinements, explicitly aligning each iteration step with different frequency bands via a progressive spectral loss.

## Method

### Overall Architecture

IRNO splits inference into two stages:

- **Initialization Stage**: A pre-trained and frozen base operator $T_{\text{base}}:\mathcal{X}\to\mathcal{H}$ (FNO / TFNO / WDSR) computes a coarse solution $h_0=T_{\text{base}}(x)$, handling large-scale low-frequency structures.
- **Iterative Correction Stage**: A shared-weight correction operator $\Phi_\theta:\mathcal{X}\times\mathcal{H}\to\mathcal{H}$ computes residual updates iteratively:

    $h_{k+1} = h_k + \alpha\cdot\Phi_\theta(x, h_k),\quad k=0,\dots,K-1$

    where $\alpha\in(0,1]$ is the step size controlling the trade-off between convergence speed and stability. At each step, the original input $x$ and current estimate $h_k$ are concatenated and fed into $\Phi_\theta$ to output the correction.

$\Phi_\theta$ is instantiated as a lightweight U-Net, but the framework is architecture-agnostic. The architecture must satisfy three requirements: (i) smoothness for iterative stability, (ii) multi-scale expressivity for spectral correction, and (iii) shared weights across iterations for computational efficiency. Critically, $\Phi_\theta$ learns an "iteration-invariant update rule," allowing more steps $k>K$ during inference than used in training.

### Key Designs

1.  **Function Space Fixed-Point Iteration + Inter-operator Transferability**:
    - **Function**: Reformulates single-pass prediction as an iterative process converging to a unique fixed point and allows the same $\Phi_\theta$ to refine outputs from different base operators.
    - **Mechanism**: Each iteration is $h_{k+1}=T(h_k)=h_k+\alpha\Phi_\theta(x,h_k)$, referencing the Banach Fixed-Point Theorem. The paper performs a first-order Taylor expansion around solution $y$: $\Phi(x,h)=b(x)+A(x,h)e+R(x,h)$, where $e=y-h$ is the residual. If the linearization $A(x,y)$ is strongly monotonic (there exist $m,M$ such that $\langle Ae,e\rangle\geq m\|e\|^2$ and $\|A\|_\text{op}\leq M$), then choosing $0<\alpha<2m/M^2$ ensures $q=\|I-\alpha A\|_\text{op}<1$. Theorem 3.1 gives error recursion $\|e_{k+1}\|\leq q\|e_k\|+c\|e_k\|^2+\alpha\|b\|$, Corollary 3.2 gives geometric convergence $\|e_k\|\lesssim q^k\|e_0\|$, and Corollary 3.3 gives the limit error bound under bias $\|e^*\|\leq \alpha\|b\|/(1-q)$.
    - **Design Motivation**: Breaks the "retrain for better accuracy" constraint. Accuracy gains are decoupled from retraining; increasing inference steps further reduces error. Since the corrector learns local residual geometry rather than the full mapping, it can migrate to other base operators seamlessly (often performing better as a $\Phi$ trained on a weak base encounters more diverse residual structures).

2.  **Multi-step Trajectory Supervision + Progressive Spectral Loss**:
    - **Function**: Explicitly supervises $\Phi_\theta$ at every iteration and forces focus on different frequency bands at different stages.
    - **Mechanism**: During training, the $K$-step trajectory is unrolled end-to-end with trajectory supervision $\mathcal{L}_{\text{spatial}}=\frac{1}{K}\sum_k\|h_k-y\|^2$ to prevent the network from deviating from the solution manifold. For spectral loss, the FFT magnitude difference between target and prediction is weighted by frequency $\rho(\omega,\lambda_k)=1+(|\omega|/|\omega|_{\text{nyq}})^{\lambda_k}$. The exponent $\lambda_k$ increases linearly from $\lambda_\text{start}$ to $\lambda_\text{end}$ (1.0 to 2.0) across iterations. Early steps focus on coarse structures, while later steps penalize high-frequency errors—analogous to the "coarse-to-fine" V-cycle in classical multigrid methods.
    - **Design Motivation**: Standard spatial L2 loss is insensitive to high frequencies (which contribute little to total energy). Fixed spectral weights might cause early iterations to be diverted by high-frequency noise. Progressive scheduling aligns training dynamics with fixed-point inference dynamics: large/coarse movements first, small/fine refinements later.

3.  **Fixed-Point Regularization to Compress Bias Error**:
    - **Function**: Directly minimizes the bias term $\|b\|=\|\Phi_\theta(x,y)\|$ from Theorem 3.1 to lower the limit error bound from Corollary 3.3.
    - **Mechanism**: Adds a term $\mathcal{L}_{\text{fp}}=\|\Phi_\theta(x,y)\|^2$, requiring the correction to be zero when the input is already the true solution $y$. This pins $y$ as a fixed point of the learned dynamical system, preventing the network from pushing the state away from the truth once converged. The total loss is $\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{spatial}}+\beta_{\text{spectral}}\mathcal{L}_{\text{spectral}}+\beta_{\text{fp}}\mathcal{L}_{\text{fp}}$. Figure 3 shows scatter plots of $\min_k\|e_k\|$ vs. $\|b\|$ on Active Matter and TR-2D with Pearson coefficients exceeding 0.93 ($p\ll 10^{-10}$), empirically validating that lower bias results in a lower error floor.
    - **Design Motivation**: Classical fixed-point solvers require the solution to be a fixed point; otherwise, they converge to the wrong location. Without this constraint, $\Phi_\theta$ might learn a degenerate solution that outputs non-zero corrections at $y$, causing the iteration to diverge even from a perfect initial value.

### Loss & Training
Total loss: $\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{spatial}}+\beta_{\text{spectral}}\mathcal{L}_{\text{spectral}}+\beta_{\text{fp}}\mathcal{L}_{\text{fp}}$. FNO base is trained with $K=6$ steps, TFNO/WDSR with $K=4$. Inference is evaluated up to $k=12$ and $k=8$ respectively (2× extrapolation). Step size $\alpha\in\{0.2, 0.25\}$ is most stable. The base operator is frozen during all training phases.

## Key Experimental Results

### Main Results

| Dataset | Metric | Base | Single-pass Baseline | IRNO | Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TR-2D | VRMSE ↓ | FNO | 0.2394 | 0.1309 | 45.32% |
| TR-2D | VRMSE ↓ | TFNO | 0.2371 | 0.1042 | **56.05%** |
| Active Matter | VRMSE ↓ | FNO | 0.1017 | 0.0501 | 50.73% |
| Active Matter | VRMSE ↓ | TFNO | 0.1981 | 0.0387 | **80.46%** |
| ERA5 16× | ACC ↑ | FNO | 0.7523 | 0.8919 | 18.56% |
| ERA5 16× | RFNE ↓ | FNO | 0.3247 | 0.2140 | 34.09% |
| ERA5 16× | ACC ↑ | WDSR | 0.9091 | 0.9104 | 0.14% |

On ERA5, IRNO (WDSR) outperforms recent spectral-specialized methods: HiNOTE (ACC 0.9055 / RFNE 0.2222) and HFS (ACC 0.8915 / RFNE 0.2253), achieving ACC 0.9104 / RFNE 0.1953. It is complementary to HFS—on Active Matter, HFS + IRNO reduces VRMSE from 0.0631 to 0.0486. Frequency analysis on Active Matter (FNO) shows high-frequency error reduced to 1.48–2.04% of the base, mid-frequency to 5.07–6.68%, and low-frequency to 27.72–36.10%.

### Ablation Study

| Configuration | VRMSE ↓ | Low-freq ratio | Mid-freq ratio | High-freq ratio | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Progressive Spectral $\lambda:1\to2$ | **0.0387** | 0.0551 | 0.0788 | **0.2393** | Full Model |
| Fixed $\lambda=1.00$ | 0.0509 | 0.0953 | 0.1067 | 0.6023 | Insufficient HF weight |
| Fixed $\lambda=1.25$ | 0.0695 | 0.1599 | 0.2101 | 0.8794 | Performance drop across bands |
| Fixed $\lambda=1.75$ | 0.0586 | 0.1124 | 0.1320 | 0.6949 | Excess early HF weight |
| Fixed $\lambda=2.00$ | 0.0666 | 0.2063 | 0.1578 | 0.7677 | Early steps diverted by noise |

In cross-operator transfer, IRNO$_{\text{TFNO}}$ refining FNO outputs reduces TR-2D VRMSE from 0.2396 to 0.0994 (58.53% gain), outperforming the same-operator IRNO$_{\text{FNO}}$ by 13 percentage points. On irregular grids (CE-Gauss with RIGNO base), it improves each step of a 7-step autoregressive rollout, with gains rising from 12.5% at $t=1$ to 21.3% at $t=7$, demonstrating suppression of error accumulation.

### Key Findings
- Step size $\alpha$ is critical for stability: $\alpha=0.1$ is slow but stable; $\alpha\in[0.2,0.4]$ converges quickly within training steps; $\alpha\geq 0.5$ diverges beyond $k=6$, aligning with the theoretical boundary $q=\|I-\alpha A\|_\text{op}<1$.
- Spectral error reduction is non-uniform: The largest drops occur near the Nyquist limit ($\omega=128$), effectively "reversing" the spectral bias of neural operators.
- Lower bias leads to a lower error floor (Pearson $r>0.93$), empirically confirming the effect of fixed-point regularization.
- Cross-architecture robustness: Using ResNet, ConvNext, or FNO as $\Phi_\theta$ backbones all yield >71% VRMSE reduction; choice of normalization (BN/LN/GN) has negligible impact.
- On the Inference Time—Performance Pareto, IRNO achieves ACC 0.84 at 1100 GFLOPs, while a monolithic 15× U-Net baseline at equal compute only achieves 0.79, suggesting gains stem from the iterative mechanism rather than parameter count.

## Highlights & Insights
- **Turning Spectral Bias into a Tunable Parameter**: Previously viewed as an "intrinsic flaw" of neural operators, IRNO treats spectral bias as "soft knowledge" recoverable by "running more iterations," shifting complexity from training time to inference-time graph depth.
- **Tight Theory-Experiment Loop**: Theorem 3.1 predicts an error lower bound $\propto\|b\|$ when bias is present; the paper validates this with scatter plots showing Pearson correlation >0.93. The critical $\alpha$ values identified in Figure 7 across 6 step sizes match the $\|I-\alpha A\|<1$ boundary. Using classical numerical analysis as a "compass" is a valuable methodology.
- **Cross-operator Transfer Outperforming Original Base**: IRNO$_{\text{TFNO}}$ refining FNO was better than IRNO$_{\text{FNO}}$. This suggests that when training refinement modules, **intentionally selecting a weaker base operator might be superior**, as it exposes the corrector to larger and more diverse residual structures.
- **Train Short, Test Long**: Stable performance at 2K inference steps after training for $K=4-6$ steps is a highly desirable property (similar to context window extrapolation in transformers) achieved through shared weights and strong convergent dynamics.

## Limitations & Future Work
- Inference computation increases linearly with $K$. While IRNO wins on the Pareto front against capacity-matched monolithic models, it remains disadvantaged for real-time scenarios (edge deployment, online control) sensitive to single-step latency.
- Convergence guarantees depend on the base operator's initial value falling within the basin of attraction (Assumption 3). The theory does not cover "completely wrong" guesses or severe base operator failures, and no detector for being "outside the basin" is provided.
- Spectral analysis is most detailed for Active Matter; TR-2D and ERA5 only have summarized data. Extremely long extrapolation (e.g., $8\times$) likely requires smaller step sizes or scheduling, which is mentioned in the appendix but not systematized.
- The corrector learns "residual geometry"; performance on PDE solutions with discontinuities (shocks, phase interfaces) was not specifically tested and might require replacing the spectral loss with wavelets or non-stationary bases.
- A natural extension is treating $\Phi_\theta$ as a learnable Krylov subspace generator combined with deflation or Anderson acceleration to accelerate convergence.

## Related Work & Insights
- **vs HiNOTE / HFS**: These improve spectral bias via architecture (hierarchical attention / frequency scaling). IRNO introduces iterative refinement at inference, which is orthogonal and additive—HFS+IRNO on Active Matter reduces error by an additional 23% over HFS alone.
- **vs F-Adapter (Parameter-Efficient Spectral Fine-tuning)**: F-Adapter offers a 2.31% VRMSE gain with low overhead, while IRNO offers 50.73% with higher compute. They target low-resource vs. accuracy-sensitive scenarios, respectively.
- **vs Classical Multigrid / Defect Correction**: IRNO is essentially a learned version of defect correction, replacing the smoother with a neural network. The "coarse-to-fine" spectral scheduling corresponds to V-cycle logic, providing a data-driven smoother to replace hand-designed Jacobi/GS methods.
- **vs Iterative Denoising in Diffusion Models**: Both involve $h_k\to h_{k+1}$ iterations, but DDPM uses noise scheduling rather than fixed-point theory. IRNO provides a formalization for learning iterators using Banach fixed points instead of stochastic differential equations, potentially inspiring deterministic sampler designs.

## Rating
- Novelty: ⭐⭐⭐⭐ Introduces classical defect correction to neural operators with contraction proofs; clean logic, though individual elements aren't entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 PDE systems × 4 base operators × cross-architecture / cross-step / cross-frequency ablations. Every theoretical prediction has an empirical counterpart.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical assumptions, visual proof for every corollary, well-structured tables. A model for scientific computing papers.
- Value: ⭐⭐⭐⭐⭐ Provides a universal "accuracy boost without retraining" path immediately applicable to deployed neural operators; the contraction perspective inspires future research on inference-time scaling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning to Refine: Spectral-Decoupled Iterative Refinement Framework for Precipitation Nowcasting](learning_to_refine_spectral-decoupled_iterative_refinement_framework_for_precipi.md)
- [\[ICML 2026\] Generative Neural Operators Through Diffusion Last Layer](generative_neural_operators_through_diffusion_last_layer.md)
- [\[ICML 2026\] EqGINO: Equivariant Geometry-Informed Fourier Neural Operators for 3D PDEs](eqgino_equivariant_geometry-informed_fourier_neural_operators_for_3d_pdes.md)
- [\[ICML 2026\] REX: A Family of Reversible Exponential Stochastic Runge-Kutta Solvers](rex_a_family_of_reversible_exponential_stochastic_runge-kutta_solvers.md)
- [\[AAAI 2026\] PhysicsCorrect: A Training-Free Approach for Stable Neural PDE Simulations](../../AAAI2026/physics/physicscorrect_a_training-free_approach_for_stable_neural_pde_simulations.md)

</div>

<!-- RELATED:END -->
