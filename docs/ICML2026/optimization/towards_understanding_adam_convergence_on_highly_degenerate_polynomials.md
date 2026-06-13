---
title: >-
  [Paper Note] Towards Understanding Adam Convergence on Highly Degenerate Polynomials
description: >-
  [ICML2026][Optimization][Adam] This paper selects a class of highly degenerate polynomials $L(x)=\tfrac{1}{k}x^k$ ($k\ge 4$ even) as minimal problem models. It proves that under constant learning rates…
tags:
  - "ICML2026"
  - "Optimization"
  - "Adam"
  - "degenerate minima"
  - "linear convergence"
  - "phase diagram"
  - "adaptive step size"
date: 2026-05-08
content_hash: 777017a4fc38c9ac
---

# Towards Understanding Adam Convergence on Highly Degenerate Polynomials

**Conference**: ICML2026  
**arXiv**: [2603.09581](https://arxiv.org/abs/2603.09581)  
**Code**: No public link provided  
**Area**: optimization  
**Keywords**: Adam, degenerate minima, linear convergence, phase diagram, adaptive step size  

## TL;DR
This paper selects a class of highly degenerate polynomials $L(x)=\tfrac{1}{k}x^k$ ($k\ge 4$ even) as minimal problem models. It proves that under constant learning rates, Adam achieves local linear convergence by exponentially amplifying the effective learning rate through a "decoupling" mechanism between $v_t$ and $g_t^2$. In contrast, GD and momentum only achieve a sub-linear rate of $\Theta(t^{-1/(k-2)})$ on the same problem. The study fully characterizes three phase regions on the $(\beta_1,\beta_2)$ plane: "stable convergence / spike / SignGD oscillation."

## Background & Motivation
**Background**: Adam is the default optimizer in deep learning, but theoretical analysis has long been limited to two types of settings: either requiring an external $\eta/\sqrt T$ decay schedule with $\beta_2$ close to 1 (e.g., the $O(\log T/\sqrt T)$ rate in Zhang et al. 2022a), or constructing counterexamples to show that Adam diverges on simple convex problems (Reddi et al. 2018).

**Limitations of Prior Work**: In practice, Adam works well with constant learning rates and $\beta_2\in[0.9,0.999]$. Existing theories neither explain this "bare Adam convergence" nor clarify Adam's specific advantages over GD/Momentum. Existing explanations are mostly scattered across inconsistent mechanisms like Hessian heterogeneity, coordinate $\ell_\infty$ geometry, or heavy-tailed noise assumptions.

**Key Challenge**: Theoretical analyses generally focus on strongly convex or general convex objectives, whereas deep network loss landscapes are empirically characterized as highly degenerate (Hessian spectrum concentrated near 0). Strong convexity theory happens to be the scenario where Adam is most prone to spikes, creating a mismatch with the empirical evidence where Adam far outperforms SGD in training Transformers.

**Goal**: To find a clean class of objective functions that isolates the "intrinsic adaptive" advantage of Adam; to achieve local stability analysis, convergence rates, and complexity separation from GD/Momentum on this class; and to align the hyperparameter phase diagram with experimental phenomena (spikes, SignGD oscillation).

**Key Insight**: Starting from the comparison in Fig. 1—on $L=\tfrac{1}{2}x^2$, constant step-size Adam eventually spikes, but on $L=\tfrac{1}{4}x^4$, it exhibits stable exponential descent. This suggests that a degeneracy order of $k\ge 4$ is the "comfortable" curvature structure for Adam and should be treated as the model problem for analysis.

**Core Idea**: Use "whether $v_t$ tracks $g_t^2$" as the phase transition anchor—when $g_t$ decays fast enough, $v_t\to\beta_2 v_{t-1}$ enters geometric decay, which is equivalent to equipping GD with an exponential learning rate schedule $\beta_2^{-t/2}$. This precisely transforms the sub-linear curse of degenerate polynomials back into a linear rate.

## Method

### Overall Architecture
The paper does not introduce new algorithms but applies a "model problem → state space → phase diagram" theoretical analysis pipeline to original Adam ($\varepsilon=0$, ignoring bias correction):

1.  **Model Problem Layer**: Fixed $L(x)=\tfrac{1}{k}x^k$, $k\ge 4$ even; this represents the "degenerate direction" in deep network loss landscapes.
2.  **State Space Layer**: Re-normalize Adam's three temporal variables using $\omega_t:=m_t/x_t^{k-1}$ and $\lambda_t:=x_t^{k-2}/\sqrt{v_t}$. This converts the dynamics from $(x_t,m_t,v_t)$ to $(\omega_t,\lambda_t,x_t)$, transforming the iteration into $x_{t+1}=(1-\eta\omega_t\lambda_t)x_t$.
3.  **Stability/Phase Diagram Layer**: Solve for non-trivial fixed points and analyze the Jacobian spectral radius to derive the stability condition $\beta_1<\beta_2^{k/(2(k-2))}$. This condition corresponds to the three empirical behaviors of Adam.
4.  **Acceleration Mechanism Layer**: Isolate adaptive effects in RMSProp ($\beta_1=0$), proving that $v_t$ geometric decay → exponential growth of the effective learning rate → upgrade from sub-linear to linear convergence on degenerate objectives.
5.  **Discretization Stability Layer**: Compress the "exponentially growing learning rate" back into discrete GD by introducing effective sharpness $u_t=\eta_t x_t^{k-2}$. Study the bifurcation of the 1D mapping $u_{t+1}=\gamma u_t(1-u_t)^{k-2}$ to obtain the global convergence threshold $\gamma<(\tfrac{k}{k-2})^{k-2}$.

### Key Designs

1.  **Normalized State Space $(\omega_t,\lambda_t,x_t)$ (Time-variant to Time-invariant)**:
    - **Function**: Compresses the Adam coupling system, which contains high powers of $x_t$, into a low-dimensional dynamic system independent of iteration scale.
    - **Mechanism**: After substituting $g_t=x_t^{k-1}$ and writing recurrences for $\{m_t,v_t,x_{t+1}\}$, the variables $\omega_t=m_t/x_t^{k-1}$ (normalized first moment) and $\lambda_t=x_t^{k-2}/\sqrt{v_t}$ (effective curvature) are introduced. These variables decouple $x_t$ decay, turning the iteration into $x_{t+1}=(1-\eta\omega_t\lambda_t)x_t$. The condition for monotonic loss descent is equivalent to $0\le\omega_t\lambda_t\le 2/\eta$.
    - **Design Motivation**: In the original variables, $x_t$ appears with multiple exponents, contaminating the spectral radius analysis with scale. After normalization, non-trivial fixed points appear at $x^\star=0$ while $(\omega^\star,\lambda^\star)$ take finite values independent of $x_t$, reducing stability analysis to a $2\times 2$ sub-Jacobian.

2.  **$v_t$–$g_t^2$ Decoupling Mechanism (Exponentially Amplifying Effective Learning Rate)**:
    - **Function**: Explains why Adam converts sub-linear convergence to linear on degenerate targets.
    - **Mechanism**: Under the RMSProp setting, Lemma 5.4 proves that if $x_t\to 0$, then $g_t/\sqrt{v_t}\to 0$, leading to $v_t/v_{t-1}\to\beta_2$, i.e., $v_t\sim\beta_2^t$. This implies the effective learning rate $\eta_{\mathrm{eff},t}=\eta/\sqrt{v_t}\propto\beta_2^{-t/2}$ grows exponentially. Lemma 5.5 applies this exponential schedule to the continuous GD flow $\dot x=-\eta(t)x^{k-1}$, yielding $x(t)\sim\exp\bigl(-\tfrac{\alpha}{k-2}t\bigr)$—elevating power-law decay to exponential decay.
    - **Design Motivation**: Previous work viewed Adam's advantage through the lens of SignGD, but SignGD is a $\beta_2=0$ limit that **does not converge** to 0. This paper argues that the true source of acceleration is the "geometric memory of $v_t$ lagging behind $g_t^2$," which is **mechanistically opposite** to SignGD. This mechanism only activates when $g_t$ decays fast enough, corresponding to $k\ge 4$ structures.

3.  **Three-Phase Diagram and Jacobian Spectral Condition (Theorem 4.1 + Theorem 6.1)**:
    - **Function**: Classifies all steady-state behaviors of Adam using $(\beta_1,\beta_2)$ into "stable convergence / spike / SignGD oscillation."
    - **Mechanism**: The existence condition for the non-trivial fixed point is $\beta_1<\beta_2^{(k-1)/(2(k-2))}$, and the stability condition is $\beta_1<\beta_2^{k/(2(k-2))}$. Putting these curves and the "no fixed point" region on the $(\beta_1,\beta_2)$ plane yields three regions: (I) Both conditions met → stable linear convergence at rate $x_{t+1}/x_t\to\beta_2^{1/(2(k-2))}$; (II) Fixed point exists but is unstable → early attraction followed by $\omega_t\lambda_t>2/\eta$ triggering a spike; (III) No fixed point → $v_t$ tracks $g_t^2$, equivalent to SignGD, loss oscillates near $L(\eta/2)$.
    - **Design Motivation**: Consolidates scattered observations of Adam failure modes (limit cycles, loss spikes, oscillations) into a single phase diagram using the same set of inequalities. It predicts a theoretical rate of $k\ln\beta_2/(2(k-2))$, which aligns perfectly with the empirical slopes in Fig. 2(a).

### Loss & Training
No training loss (theoretical paper), but three key quantitative predictions applicable to practice are provided: (a) Linear convergence rate $\beta_2^{1/(2(k-2))}$; (b) Complexity of GD on degenerate targets $T_\varepsilon\sim\varepsilon^{-(k-2)}$ vs. Adam's $T_\varepsilon\sim(k-2)\ln(1/\varepsilon)$; (c) Global stability threshold for the 1D mapping $\gamma_{\mathrm{crit}}=(\tfrac{k}{k-2})^{k-2}$, which for $k=4$ is equivalent to $\beta_2>0.0625$.

## Key Experimental Results

### Main Results
All experiments were conducted on the analytical minimal problem $L(x)=\tfrac{1}{k}x^k$ to verify theoretical predictions.

| Experiment | Setting | Key Observation | Corresponding Theory |
| :--- | :--- | :--- | :--- |
| Strong Convexity vs. Degenerate | $L=\tfrac{1}{2}x^2$ vs. $L=\tfrac{1}{4}x^4$, $\beta_1=0.9, \beta_2=0.99$ | Adam eventually spikes on strong convexity; stable exponential descent on $k=4$ with slope $\approx -0.0726$ | Confirms Adam's "degeneracy preference" |
| Convergence Rate Validation | $k=4,6$ Adam loss curves | $k=4$ measured slope $\approx -0.0726$, theory $-0.0726$; $k=6$ measured $\approx -0.0544$, theory $-0.0544$ | Perfect alignment |
| Evolution of $\lambda_t$ | Comparison of $k=2$ vs. $k=4$ | At $k=2$, $\lambda_t=1/\sqrt{v_t}$ grows boundlessly past $2/\eta$ (red zone); at $k>2$, $\lambda_t$ converges to a constant | Visualization of stability mechanism |
| Theory vs. Empirical Phase Diagram | $k=4, x_0=1, \eta=0.001$, 100k steps | Stable zone final loss $\approx 10^{-300}$ (machine epsilon); unstable zone significantly higher | Confirms stability inequality in Thm 4.1 |
| Three Typical Trajectories | $\beta_1=0.9, \beta_2\in\{0.91, 0.895, 0.8\}$ | Produces stable exponential, exponential + spike, and SignGD oscillation respectively | Matches three phases |
| Coupling Ratio $R_t^{(v)}=v_t/g_t^2$ | Same as above | Phase I/II shows very large $\max R_t^{(v)}$ (decoupling), Phase III $\approx 1$ (tight coupling) | Verifies decoupling is the root of acceleration |
| Mixed Degenerate/Non-degenerate | 4 types of coupled losses like $\tfrac{1}{4}(x-y)^2+\tfrac{1}{16}(x+y)^4$ | Adam is significantly faster than GD/Momentum in degenerate directions, but quadratic components introduce spikes | Explains why Adam is fast yet prone to spikes |

### Ablation Study

| Configuration | Key Change | Explanation |
| :--- | :--- | :--- |
| Full Adam | Both $\beta_1, \beta_2$ active | All three phases emerge in the diagram |
| $\beta_1=0$ (RMSProp) | First moment removed | State space drops from 3D to 2D; Thm 5.7 gives **global** linear convergence, proving acceleration comes from $v_t$ |
| $\beta_2=0$ and $\varepsilon\to 0$ (SignGD) | Second moment memory removed | Constant step size fails to converge, stalling at $O(L(\eta))$, proving "geometric memory" is key |
| $\tfrac{1}{k}x^k$ to $\tfrac{1}{k}x^k(1+h(x))$ | $h$ is analytic, $h(0)=0$ | Local stability conditions and rates remain identical | Universality of theory for general degenerate minima |

### Key Findings
- As the degeneracy order $k$ increases, the stable region in the $(\beta_1,\beta_2)$ plane **expands monotonically** (Remark 4.2), meaning harder degeneracy actually broadens the hyperparameter tolerance of Adam.
- The physical origin of Spikes is precisely located: in Phase II, the fixed point exists but is unstable. $v_t$ initially decouples to drive acceleration, but when $x_t$ suddenly rebounds, $v_t$ takes several steps to catch up with $g_t^2$—this "response delay" is the loss spike.
- The SignGD perspective (e.g., Kunstner et al. 2023) only explains Phase III. Linear acceleration in Phases I/II originates from the $v_t$–$g_t^2$ decoupling, which is the physical inverse of SignGD.

## Highlights & Insights
- **Selecting the Correct "Minimal Problem"**: By using $L=\tfrac{1}{k}x^k$, the paper preserves the critical degeneracy of deep network landscapes while allowing for analytical derivation. This paradigm of "isolating mechanisms with the simplest model problem" is highly valuable for future optimization theory.
- **Rigorous Complexity Separation**: Clearly establishes the exponential complexity explosion of GD on degenerate targets $T_\varepsilon\sim\varepsilon^{-(k-2)}$ versus Adam's $(k-2)\ln(1/\varepsilon)$ linear complexity, upgrading "Adam is faster" from an empirical observation to a complexity class separation without needing stochasticity assumptions.
- **Opposing Mechanism Insight**: Decoupling ($v_t$ lagging $g_t^2$) and SignGD ($v_t$ following $g_t^2$) are opposing physical processes. Distinguishing them helps diagnose whether Adam's advantage in a real task stems from one mechanism or the other.

## Limitations & Future Work
- The analysis is performed on 1D degenerate polynomials; while deep network landscapes share Hessian spectral similarities, their coupling is more complex.
- The assumptions ($\varepsilon=0$, ignoring bias correction, deterministic gradients) differ from training large models. The authors explicitly list stochastic batch settings as future work.
- There is no closed-form solution for the global basin of attraction; Theorem 4.1 only provides local stability.
- Improvement ideas: (a) Extend theorems to SDE settings with mini-batch noise; (b) Match $k$ estimates to the Hessian spectrum of real Transformers to predict stability zones; (c) Use the decoupling ratio $R_t^{(v)}$ as a health monitoring signal for training to trigger adaptive $\beta_2$ schedules.

## Related Work & Insights
- **vs. Zhang et al. (2022a)**: They require $\beta_2$ close to 1 and learning rate decay to prove $O(\log T/\sqrt T)$; this paper gives local linear convergence under constant learning rates across the $(\beta_1, \beta_2)$ domain, showing their conditions are essentially to avoid Phase III.
- **vs. Davis et al. (2025)**: They show GD + Polyak adaptive steps achieve linear convergence under quartic growth; this paper proves Adam achieves this **intrinsically** through $v_t$ geometric memory without external Polyak rules.
- **vs. SignGD Acceleration Theory (Kunstner et al. 2023)**: Phase III in this paper corresponds to SignGD and does not converge; linear acceleration comes from the opposite decoupling mechanism—completing the missing piece of Adam's advantage.
- **vs. Cohen et al. (2023) "Edge of Stability"**: They observe spikes near EoS; this paper provides a precise algebraic characterization via fixed-point instability in Phase II.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The opposition between decoupling and SignGD, along with the full phase diagram for Adam under constant step sizes, are clear characterizations previously absent in literature.
- Experimental Thoroughness: ⭐⭐⭐⭐ Achieved "perfect alignment of theory and empirical slopes" on analytic problems, but lacks end-to-end replication on large-scale real networks.
- Writing Quality: ⭐⭐⭐⭐⭐ The introduction of normalized state space, the matching tables (Table 1), and mechanism diagrams (Fig. 6) are exceptionally clear and well-integrated.
- Value: ⭐⭐⭐⭐ A significant completion for optimization theory with insights for hyperparameter and adaptive schedule design, though it does not directly propose a new pipeline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](../../NeurIPS2025/optimization/understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)
- [\[NeurIPS 2025\] Understanding Adam Requires Better Rotation Dependent Assumptions](../../NeurIPS2025/optimization/understanding_adam_requires_better_rotation_dependent_assumptions.md)
- [\[ICML 2026\] The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks](the_implicit_bias_of_adam_and_muon_on_smooth_homogeneous_neural_networks.md)
- [\[ICML 2026\] Balanced LoRA: Removing Parameter Invariance to Accelerate Convergence](balanced_lora_removing_parameter_invariance_to_accelerate_convergence.md)
- [\[ICML 2026\] Adaptive Preconditioners Trigger Loss Spikes in Adam](adaptive_preconditioners_trigger_loss_spikes_in_adam.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[NeurIPS 2025\] Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](../../NeurIPS2025/optimization/understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)
- [\[NeurIPS 2025\] Understanding Adam Requires Better Rotation Dependent Assumptions](../../NeurIPS2025/optimization/understanding_adam_requires_better_rotation_dependent_assumptions.md)
- [\[ICML 2026\] Balanced LoRA: Removing Parameter Invariance to Accelerate Convergence](balanced_lora_removing_parameter_invariance_to_accelerate_convergence.md)
- [\[NeurIPS 2025\] In Search of Adam's Secret Sauce](../../NeurIPS2025/optimization/in_search_of_adams_secret_sauce.md)
- [\[ICML 2026\] Towards Understanding Continual Factual Knowledge Acquisition of Language Models: From Theory to Algorithm](towards_understanding_continual_factual_knowledge_acquisition_of_language_models.md)

</div>

<!-- RELATED:END -->
