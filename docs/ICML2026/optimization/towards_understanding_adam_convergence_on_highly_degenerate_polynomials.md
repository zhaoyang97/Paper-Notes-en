---
title: >-
  [Paper Note] Towards Understanding Adam Convergence on Highly Degenerate Polynomials
description: >-
  [ICML 2026][Optimization & Theory][Adam] This paper selects a class of high-order degenerate polynomials $L(x)=\tfrac{1}{k}x^k$ (even $k\ge 4$) as a minimal problem model. It proves that under a constant learning rate, Adam achieves local linear convergence by exponentially amplifying the effective learning rate through a "decoupling" mechanism between $v_t$
tags:
  - ICML 2026
  - Optimization & Theory
  - Adam
date: 2026-05-08
content_hash: 48c80cd6b268d986
---
# Towards Understanding Adam Convergence on Highly Degenerate Polynomials

**Conference**: ICML2026  
**arXiv**: [2603.09581](https://arxiv.org/abs/2603.09581)  
**Code**: No public link provided  
**Area**: Optimization  
**Keywords**: Adam, Degenerate Minima, Linear Convergence, Phase Diagram, Adaptive Step Size  

## TL;DR
This paper selects a class of high-order degenerate polynomials $L(x)=\tfrac{1}{k}x^k$ (even $k\ge 4$) as a minimal problem model. It proves that under a constant learning rate, Adam achieves local linear convergence by exponentially amplifying the effective learning rate through a "decoupling" mechanism between $v_t$ and $g_t^2$. Meanwhile, GD and momentum only achieve a sublinear rate of $\Theta(t^{-1/(k-2)})$ on the same problem. The study comprehensively characterizes three phase regions of Adam—"stable convergence / spike / SignGD oscillation"—on the $(\beta_1,\beta_2)$ plane.

## Background & Motivation
**Background**: Adam is the default optimizer in deep learning, but theoretical analysis has long been restricted to two settings: either requiring an external $\eta/\sqrt T$ decay schedule with $\beta_2$ close to 1 (e.g., the $O(\log T/\sqrt T)$ rate in Zhang et al. 2022a), or constructing counterexamples showing Adam diverges on simple convex problems (Reddi et al. 2018).

**Limitations of Prior Work**: In practice, Adam works well with constant learning rates and $\beta_2\in[0.9, 0.999]$. Existing theories neither explain this "bare Adam convergence" nor clarify the specific targets where Adam holds a genuine advantage over GD/momentum—existing explanations are scattered across various mechanisms like Hessian heterogeneity, coordinate $\ell_\infty$ geometry, or heavy-tailed noise.

**Key Challenge**: Theoretical analyses generally focus on strongly convex or general convex targets, whereas empirical evidence suggests deep network loss landscapes are highly degenerate (Hessian spectrum concentrated near 0). Strong convexity is precisely where Adam is most prone to spikes, creating a mismatch with the phenomenon where Adam significantly outperforms SGD in training Transformers.

**Goal**: To find a clean class of objective functions that allows the "intrinsic adaptive" advantage of Adam to be purely isolated; to complete local stability analysis, convergence rates, and complexity separation from GD/Momentum on this class; and to align the hyperparameter phase diagram with empirical phenomena (spikes, SignGD oscillations).

**Key Insight**: Starting from the comparative observation in Fig. 1—Adam with a constant step size eventually spikes on $L=\tfrac{1}{2}x^2$, but achieves stable exponential decay on $L=\tfrac{1}{4}x^4$. This suggests that a degeneracy order of $k\ge 4$ is the "comfortable" curvature structure for Adam and should be analyzed as a model problem.

**Core Idea**: Use "whether $v_t$ tracks $g_t^2$" as the anchor for phase transitions. When $g_t$ decays sufficiently fast, $v_t\to\beta_2 v_{t-1}$ enters geometric decay. This is equivalent to equipping GD with an exponential learning rate schedule of $\beta_2^{-t/2}$, which pulls the sublinear curse of degenerate polynomials back to a linear rate.

## Method

### Overall Architecture
The paper does not introduce a new algorithm but performs a "Model Problem → State Space → Phase Diagram" theoretical analysis on original Adam (with $\varepsilon=0$ and bias correction ignored):

1.  **Model Problem Layer**: Fixed $L(x)=\tfrac{1}{k}x^k$, for even $k\ge 4$; this represents the minimal "degenerate direction" in deep network loss landscapes.
2.  **State Space Layer**: Uses $\omega_t:=m_t/x_t^{k-1}$ and $\lambda_t:=x_t^{k-2}/\sqrt{v_t}$ to re-normalize Adam's three temporal variables. The dynamics are rewritten from $(x_t,m_t,v_t)$ to $(\omega_t,\lambda_t,x_t)$, transforming the iteration into $x_{t+1}=(1-\eta\omega_t\lambda_t)x_t$.
3.  **Phase Diagram Layer**: Solves for non-trivial fixed points and analyzes their Jacobian spectral radii to obtain the stability condition $\beta_1<\beta_2^{k/(2(k-2))}$. The presence/stability of fixed points corresponds to three empirical Adam behaviors.
4.  **Acceleration Mechanism Layer**: Isolates the adaptive effect on RMSProp ($\beta_1=0$), proving that geometric decay of $v_t$ leads to exponential growth of the effective learning rate, upgrading convergence from sublinear to linear on degenerate targets.
5.  **Discretization Stability Layer**: Maps the "exponentially growing learning rate" back to discrete GD by introducing effective sharpness $u_t=\eta_t x_t^{k-2}$. Studying the bifurcation of the 1D map $u_{t+1}=\gamma u_t(1-u_t)^{k-2}$ yields a global convergence threshold $\gamma<(\tfrac{k}{k-2})^{k-2}$.

### Key Designs

**1. Normalized State Space $(\omega_t,\lambda_t,x_t)$: Compressing high-degree coupled systems into scale-invariant low-dimensional dynamics**

In the original variables $(x_t,m_t,v_t)$ for $L=\tfrac{1}{k}x^k$, $x_t$ appears with multiple powers, polluting direct spectral radius calculations with scale. By substituting $g_t=x_t^{k-1}$ and introducing two normalized quantities—$\omega_t=m_t/x_t^{k-1}$ (normalized first moment) and $\lambda_t=x_t^{k-2}/\sqrt{v_t}$ (effective curvature)—the decay of $x_t$ is decoupled. The simplified iteration $x_{t+1}=(1-\eta\omega_t\lambda_t)x_t$ implies monotonic loss descent if $0\le\omega_t\lambda_t\le 2/\eta$. Since non-trivial fixed points occur at $x^\star=0$ while $(\omega^\star,\lambda^\star)$ take finite values independent of $x_t$, stability reduces to a $2\times2$ sub-Jacobian problem, enabling analytical phase diagram derivation.

**2. $v_t$–$g_t^2$ Decoupling Mechanism: Exponentially amplifying the effective LR to upgrade sublinear to linear**

GD only achieves a sublinear rate of $\Theta(t^{-1/(k-2)})$ on degenerate targets. Why is Adam faster? The key is whether $v_t$ still tracks $g_t^2$. In the RMSProp setting, Lemma 5.4 proves that if $x_t\to 0$, then $g_t/\sqrt{v_t}\to 0$, leading to $v_t/v_{t-1}\to\beta_2$, i.e., $v_t\sim\beta_2^t$ decays geometrically. Consequently, the effective learning rate $\eta_{\mathrm{eff},t}=\eta/\sqrt{v_t}\propto\beta_2^{-t/2}$ grows exponentially. Lemma 5.5 feeds this exponential schedule into a continuous gradient flow $\dot x=-\eta(t)x^{k-1}$, yielding $x(t)\sim\exp(-\tfrac{\alpha}{k-2}t)$—power-law decay is upgraded to exponential decay. This acceleration source is mechanically opposite to the SignGD perspective: SignGD is the $\beta_2=0$ limit where $v_t$ tracks $g_t^2$ closely and fails to converge to 0; true acceleration comes from the "geometric memory of $v_t$ lagging behind $g_t^2$," a mechanism that only activates when $g_t$ decays fast enough, corresponding to $k\ge 4$ structures.

**3. Three-Phase Diagram and Jacobian Spectral Conditions (Theorem 4.1 + Theorem 6.1): Categorizing all steady-state behaviors using $(\beta_1,\beta_2)$**

The paper unifies scattered observations of Adam failure modes (limit cycles, loss spikes, oscillations) into one phase diagram. The existence condition for a non-trivial fixed point is $\beta_1<\beta_2^{(k-1)/(2(k-2))}$, and the stability condition is $\beta_1<\beta_2^{k/(2(k-2))}$. These define three regions on the $(\beta_1,\beta_2)$ plane: (I) Both conditions met → stable linear convergence with rate $x_{t+1}/x_t\to\beta_2^{1/(2(k-2))}$; (II) Fixed point exists but is unstable → early attraction produces exponential convergence, but later $\omega_t\lambda_t>2/\eta$ triggers a spike; (III) No fixed point exists → $v_t$ follows $g_t^2$ closely, equivalent to SignGD, causing loss oscillation around $L(\eta/2)$. This characterization using a single set of inequalities matches the theoretical rate $k\ln\beta_2/(2(k-2))$ with the empirical slope in Fig. 2(a).

### Loss & Training
No training loss (theoretical paper), but three key quantitative predictions are provided: (a) linear convergence rate $\beta_2^{1/(2(k-2))}$; (b) complexity of GD on degenerate targets $T_\varepsilon\sim\varepsilon^{-(k-2)}$ vs Adam's $T_\varepsilon\sim(k-2)\ln(1/\varepsilon)$; (c) global stability threshold for the auxiliary 1D map $\gamma_{\mathrm{crit}}=(\tfrac{k}{k-2})^{k-2}$, which for $k=4$ equals $\beta_2>0.0625$.

## Key Experimental Results

### Main Results
All experiments were conducted on the analytical minimal problem $L(x)=\tfrac{1}{k}x^k$ to verify theoretical predictions.

| Experiment | Setting | Key Observation | Corresponding Theory |
| :--- | :--- | :--- | :--- |
| Strong Convex vs. Degenerate | $L=\tfrac{1}{2}x^2$ vs $L=\tfrac{1}{4}x^4$, $\beta_1=0.9,\beta_2=0.99$ | Adam eventually spikes on strong convex; stable exponential decay on $k=4$, slope $\approx-0.0726$ | Confirms Adam's "degeneracy preference" |
| Convergence Rate Verification | $k=4,6$ Adam loss curve slopes | $k=4$ measured slope $\approx-0.0726$, theory $-0.0726$; $k=6$ measured $\approx-0.0544$, theory $-0.0544$ | Exact alignment |
| Evolution of $\lambda_t$ | Comparison $k=2$ vs $k=4$ | For $k=2$, $\lambda_t=1/\sqrt{v_t}$ grows unboundedly, crossing $2/\eta$ (red zone); for $k>2$, $\lambda_t$ converges to a constant | Visualizes stability mechanism |
| Theory vs. Empirical Phase Diagram | $k=4, x_0=1, \eta=0.001, 100k$ steps | Region I final loss $\approx 10^{-300}$ (machine epsilon), unstable regions significantly higher | Confirms stability inequalities in Theorem 4.1 |
| Three-Phase Trajectories | $\beta_1=0.9, \beta_2\in\{0.91, 0.895, 0.8\}$ | Generates stable exponential, exponential + spike, and SignGD oscillation respectively | Matches Three Phases |
| Phase Diagram Scan | $(\beta_1, \beta_2)$ grid of $\min_t L(x_t)$ and $L(x_T)$ | Regime II shows low $\min L$ but high $L_T$ (typical spike signature) | Distinguishes Phases I/II/III |
| Coupling Ratio $R_t^{(v)}=v_t/g_t^2$ | Same as above | In Phase I/II, $\max R_t^{(v)}$ is very large (decoupling occurs); in Phase III, $\approx 1$ (tight coupling) | Decoupling is the root of acceleration |
| Mixed Degeneracy | 4 coupled losses like $\tfrac{1}{4}(x-y)^2+\tfrac{1}{16}(x+y)^4$ | Adam is much faster than GD/Momentum in degenerate directions, but quadratic components introduce spikes | Explains why Adam is fast yet prone to spikes in real tasks |
| Architecture Correlation | ReLU vs softmax in MLP; Transformer vs CNN | Softmax and Transformers have Hessian spectra concentrated at 0 (higher degeneracy), making Adam advantages more prominent | Links theory to real models |

### Ablation Study

| Configuration | Key Change | Explanation |
| :--- | :--- | :--- |
| Full Adam | $\beta_1, \beta_2$ both active | All three phases appear |
| $\beta_1=0$ (RMSProp) | First moment removed | State space drops from 3D to 2D; Theorem 5.7 gives **global** linear convergence, proving acceleration originates from $v_t$ |
| $\beta_2=0$ & $\varepsilon\to 0$ (SignGD) | Second moment memory removed | Does not converge under constant step size, stalls at $O(L(\eta))$; proves "geometric memory" is the key to acceleration |
| Generalizing $\tfrac{1}{k}x^k$ to $\tfrac{1}{k}x^k(1+h(x))$ | $h$ is analytic and $h(0)=0$ | Local stability conditions and rates are identical—theory universal for general degenerate minima (Remark 4.4) |

### Key Findings
- Larger degeneracy order $k$ **monotonically expands** the stable region in the $(\beta_1,\beta_2)$ plane (Remark 4.2; the exponent in $\beta_1<\beta_2^{k/(2(k-2))}$ decreases with $k$), meaning harder degeneracy makes Adam hyperparameters more robust.
- The physical source of Spikes is precisely identified: In Phase II, the fixed point exists but is unstable; $v_t$ first decouples to drive acceleration, but when $x_t$ rebounds, $v_t$ takes several steps to catch up to $g_t^2$—this "response lag" is the loss spike.
- The SignGD perspective (e.g., Kunstner et al. 2023, 2024) only explains Phase III; the true acceleration in Phase I/II comes from $v_t$–$g_t^2$ decoupling, which is the mechanical opposite of SignGD.

## Highlights & Insights
- **Selecting the correct "minimal problem"**: Using $L=\tfrac{1}{k}x^k$ as the research object preserves the critical degeneracy of deep network landscapes while allowing for analytical solutions. This paradigm of isolating mechanisms via simple 1D model problems is highly valuable for future optimization theory.
- **Rigorous Complexity Separation**: Explicitly derives $T_\varepsilon\sim\varepsilon^{-(k-2)}$ for GD vs $(k-2)\ln(1/\varepsilon)$ for Adam on degenerate targets, elevating the "Adam is faster than GD" claim from empirical observation to complexity class separation without requiring stochasticity assumptions.
- **Insight into Mechanical Opposition**: Decoupling ($v_t$ lagging $g_t^2$) and SignGD mechanisms ($v_t$ tracking $g_t^2$) are opposing physical processes that can coexist in different training stages or coordinates. This distinction directly informs diagnostics for determining which mechanism drives Adam's advantage in real networks.

## Limitations & Future Work
- Analysis is on 1D degenerate polynomials; while deep network landscapes share Hessian spectral similarities, their coupling is more complex. Fig. 8 and Section 7 only demonstrate 2D toy losses.
- Assumes $\varepsilon=0$, ignores bias correction, and uses pure deterministic gradients (no mini-batch noise). These simplifications differ from training large models, and the authors list stochastic settings as future work.
- No closed-form solution for the global basin of attraction; Theorem 4.1 only provides local stability. The "broad empirical basin" in Fig. 3(b) is an observation, not a proof.
- Improvement ideas: (a) Extend current theorems to SDE settings with mini-batch noise; (b) Match $k$ estimates to Hessian blocks in real Transformers to predict stability zones; (c) Use the decoupling ratio $R_t^{(v)}$ as a training health monitor to trigger adaptive $\beta_2$ schedules.

## Related Work & Insights
- **vs Zhang et al. (2022a)**: They require $\beta_2 \to 1$ and learning rate decay for $O(\log T/\sqrt T)$; this paper provides local linear convergence across the $(\beta_1,\beta_2)$ plane with constant step sizes, showing Zhang's conditions essentially avoid Phase III.
- **vs Davis et al. (2025)**: They prove GD + Polyak adaptive step size achieves linear convergence under quartic growth; this paper proves Adam achieves the same via its **built-in** $v_t$ geometric memory without external Polyak rules.
- **vs SignGD Acceleration Theory (Kunstner et al. 2023, 2024)**: Phase III in this paper corresponds to SignGD and fails to converge; linear acceleration comes from the opposite decoupling mechanism—completing a missing piece of Adam's advantage theory.
- **vs Cohen et al. (2023) "edge of stability"**: They observe spikes in adaptive methods near EoS; this paper provides a precise algebraic characterization of spikes via fixed-point instability in Phase II.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The opposition of decoupling vs SignGD and the full phase diagram for Adam under constant step size are clear characterizations previously missing from literature.
- Experimental Thoroughness: ⭐⭐⭐⭐ Achieves exact alignment between theory and empirical slopes on model problems, but lacks end-to-end replication on real-world networks beyond preliminary architecture correlation.
- Writing Quality: ⭐⭐⭐⭐⭐ The introduction of normalized state spaces, the Phase Diagram table (Table 1), and mechanism diagrams (Fig. 6) are clear and well-coordinated with the formulas.
- Value: ⭐⭐⭐⭐ Significant theoretical completion; inspires practical improvements (dynamic $\beta_2$, adaptive schedules) although it doesn't directly propose a new pipe-line.

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
