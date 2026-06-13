---
title: >-
  [Paper Note] FOAM: Frequency and Operator Error-Based Adaptive Damping Method for Reducing Staleness-Oriented Error for Shampoo
description: >-
  [ICML2026][Optimization][Shampoo] FOAM couples the damping coefficient $\epsilon$ and the Eigenvalue Decomposition (EVD) trigger frequency of Shampoo into a feedback control loop via a "relative operator error proxy $h_t…
tags:
  - "ICML2026"
  - "Optimization"
  - "Shampoo"
  - "adaptive damping"
  - "eigenvalue decomposition frequency"
  - "preconditioner staleness"
  - "Kronecker factorization"
date: 2026-05-08
content_hash: d25ef684c440bb8d
---

# FOAM: Frequency and Operator Error-Based Adaptive Damping Method for Reducing Staleness-Oriented Error for Shampoo

**Conference**: ICML2026  
**arXiv**: [2606.02365](https://arxiv.org/abs/2606.02365)  
**Code**: https://github.com/REAL-KENTECH/FOAM.git (Available)  
**Area**: Optimizers / Second-order Preconditioning / Numerical Stability  
**Keywords**: Shampoo, adaptive damping, eigenvalue decomposition frequency, preconditioner staleness, Kronecker factorization

## TL;DR
FOAM couples the damping coefficient $\epsilon$ and the Eigenvalue Decomposition (EVD) trigger frequency of Shampoo into a feedback control loop via a "relative operator error proxy $h_t$" that can be cheaply estimated in the stale feature space. It reduces EVD calls by over 80% during large-scale model training while maintaining convergence quality.

## Background & Motivation

**Background**: In large-scale model training, pure first-order methods (SGD, Adam) are computationally cheap per step but exhibit slow convergence and sensitivity to ill-conditioned curvature. Kronecker-factored second-order preconditioners, specifically the Shampoo family, reshape gradients by computing the $-1/p$-th power of left and right factors $L_t, R_t$. These have outperformed AdamW on benchmarks like AlgoPerf. However, the core bottleneck of Shampoo is the massive computational overhead of performing EVD every step to obtain $L_t^{-1/p}$ and $R_t^{-1/p}$.

**Limitations of Prior Work**: Practical implementations commonly employ a "stale Shampoo" heuristic, where EVD is performed only every fixed $\mathbf{f}$ steps, with old inverse root factors reused in between. The selection of $\mathbf{f}$ is purely manual, lacking both theoretical guidance and the ability to adapt to the actual drift of gradient statistics during training.

**Key Challenge**: The authors point out that staleness is not merely a "minor convergence slowdown" but harms two critical aspects: (i) Convergence (a "staleness term" proportional to $(1-\beta^{\mathbf{f}}) R_{\mathrm{SG}}^4 / \epsilon_0^2$ appears in the discounted regret upper bound); (ii) Numerical stability (the Lipschitz constant of the inverse root mapping $A \mapsto A^{-1/p}$ is proportional to $1/(p \epsilon_0^{(p+1)/p})$—the smaller the $\epsilon_0$, the more sensitive it is to drift, making training prone to instability). In other words, staleness $\mathbf{f}$ and damping $\epsilon$ are coupled variables that must be controlled jointly.

**Goal**: Transform the decisions of "when to refresh EVD" and "what $\epsilon$ to use" from manual constants into interpretable feedback control. The requirements are: (a) the trigger criterion is based on actual operator error rather than an empirical schedule, (b) the decision overhead is much lower than a single EVD, and (c) training stability is maintained even with a significant reduction in EVD frequency.

**Key Insight**: The authors discovered that the full $mn \times mn$ preconditioner error $P_t - \hat{P}_t$ can be decomposed into an additive form of operator errors $\Delta_L, \Delta_R$ from the left and right factors using Kronecker identities. From a perturbation theory perspective, controlling the "relative error of each $m\times m$ and $n \times n$ factor" is a sufficient condition for overall stability, requiring only $O(m^2 + n^2)$ overhead. Combined with the observation that $\epsilon$ acts as a stabilizer while simultaneously suppressing useful preconditioning information, it becomes natural to **use $\epsilon$ as a feedback control variable**.

**Core Idea**: Define a "relative operator error proxy $h_t$" computable within the stale feature space and connect it to a multiplicative feedback loop: if $h_t > \tau$, increase $\epsilon$ to suppress error; if $h_t \le \tau$, decrease $\epsilon$ to release suppression. A true EVD is triggered only when $\epsilon$ reaches $\epsilon_{\max}$ and still cannot suppress the error.

## Method

### Overall Architecture
FOAM is embedded in the "update rule" $\mathcal{U}(\cdot)$ slot of the general Shampoo main loop (Algorithm 1). Each main loop step performs three cheap operations: computing the gradient $G_t$, updating second moments $L_t = \beta L_{t-1} + (1-\beta) G_t G_t^\top$ and $R_t = \beta R_{t-1} + (1-\beta) G_t^\top G_t$ via EMA, and updating parameters $W_{t+1} = W_t - \eta \hat{L}_t^{-1/p} G_t \hat{R}_t^{-1/p}$ using current (potentially stale) inverse root factors.

The core logic of FOAM occurs at "checkpoints" every $\mathbf{f}$ steps, executing a three-stage control flow for left and right factors independently: first, sensing the relative operator error under the current damping using the proxy $h_t$; second, updating $\epsilon_t$ via multiplicative rules; third, checking if the updated $\epsilon_t$ remains within $\epsilon_{\max}$. If so, the stale feature space is reused by merely updating the damping term (at $O(m^2)$ cost); otherwise, a full EVD is triggered, and $\epsilon$ is reset to the baseline $\epsilon_0$.

### Key Designs

1.  **Relative Operator Error Proxy $h_t$ (Sensing)**:
    *   **Function**: Provides a low-cost upper bound estimate of the "stale factor inverse root error relative to the true factor" without performing a new EVD.
    *   **Mechanism**: The authors prove $\|\Delta_L^{(1)}\|_F / \|\hat{L}_t^{-1/p}\|_F \le (\alpha / p) \cdot \text{RC}(\epsilon_{t-1})$, where $\text{RC}(\epsilon) = \|\hat{L}_t^{-1/2} (L_t - \hat{L}_t) \hat{L}_t^{-1/2}\|_F$ is the "whitened drift Frobenius norm" and $\alpha(\epsilon) = \|\hat{L}_t^{-1/p}\|_2 / \|\hat{L}_t^{-1/p}\|_F$ translates spectral to Frobenius norms. This $h_t$ is computed using cached eigenvectors $Q_L$ and eigenvalues $D_L$.
    *   **Design Motivation**: Prior works (e.g., Eschenhagen et al. using diagonalization residuals) measure how far $Q_L^\top L_t Q_L$ is from a diagonal matrix, which only reflects if the feature space can still approximate the current statistics, not the perturbation of the inverse root mapping itself. Relative error directly anchors to $\|\hat{L}_t^{-1/p}\|_F$ and anistropically amplifies drift in directions of small eigenvalues, which are exactly where the inverse root amplification is strongest.

2.  **Multiplicative Feedback Damping Control (Adapting)**:
    *   **Function**: Converts $h_t$ into the damping value $\epsilon_t$ for the next step, allowing $\epsilon$ to adaptively track real-time operator error changes.
    *   **Mechanism**: Uses the rule $\epsilon_t \leftarrow \max(\epsilon_0, \epsilon_{t-1} \cdot h_t / \tau)$, where $\tau \in (0,1)$ is the tolerance threshold. $\epsilon$ increases proportionally if $h_t > \tau$ (error exceeds threshold) and decreases if $h_t \le \tau$ (sufficient margin). $\epsilon_0$ serves as a floor to prevent $\epsilon$ from becoming too small and triggering Lipschitz sensitivity.
    *   **Design Motivation**: Fixed $\epsilon$ cannot adapt as $(1-\beta^{\mathbf{f}})$ increases with $\mathbf{f}$. Multiplicative feedback is more rational than additive because operator error depends on $\epsilon$ via a power law $\epsilon^{-(p+1)/p}$; multiplicative updates match this decay rate for stable regulation across wide dynamic ranges.

3.  **$\epsilon_{\max}$-based Delayed EVD Trigger (Frequency Control)**:
    *   **Function**: Determines when to stop "saving the situation with damping" and instead perform a full EVD refresh.
    *   **Mechanism**: After computing $\tilde{\epsilon}_t$, it checks if $\tilde{\epsilon}_t \le \epsilon_{\max}$. If true, $\hat{L}_t^{-1/p} = Q_L (D_L + \epsilon_t I_m)^{-1/p} Q_L^\top$ is reassembled with the new damping but old $Q_L, D_L$ (minimal cost). If $\tilde{\epsilon}_t > \epsilon_{\max}$, a full $\text{EVD}(L_t)$ is performed, and $\epsilon$ is reset to $\epsilon_0$.
    *   **Design Motivation**: Theorem 5.4 reveals that excessive $\epsilon_0$ causes regret to swell by $\epsilon_0 D^2$, meaning $\epsilon$ cannot increase indefinitely. $\epsilon_{\max}$ ensures damping doesn't suppress useful preconditioning information and serves as the sole trigger for EVD, ensuring EVD frequency adapts strictly to real drift rather than a blind schedule.

### Loss & Training
The original Shampoo loss objective is unchanged; only the update rule $\mathcal{U}(\cdot)$ is replaced. Key hyperparameters are sensing period $\mathbf{f}$, threshold $\tau$, max damping $\epsilon_{\max}$, and baseline $\epsilon_0$. Values used for ViT were $(20, 0.75, 3\times10^{-7}, 10^{-9})$ and for Conformer $(50, 0.4, 1\times10^{-7}, 10^{-9})$. $\mathbf{f}$ is merely the "check frequency" for $h_t$ and does not equal the EVD interval, which is extended by the $\epsilon_{\max}$ condition.

## Key Experimental Results

### Main Results
FOAM was compared against stale Shampoo across three large-scale tasks: (1) ViT-small @ ImageNet-1K (4× A6000); (2) Conformer @ LibriSpeech (4× RTX Pro 6000); (3) GPT-2 @ Wikitext-103. Metrics focused on wall-clock time required to reach equivalent/better performance.

| Task / Model | Metric | Stale Shampoo | FOAM | Key Finding |
|---|---|---|---|---|
| ImageNet-1K / ViT-small | Top-1 val. Acc. + wall-clock | baseline | Higher Acc / Lower loss / Less time | FOAM is in the top-left (superior) region |
| LibriSpeech / Conformer | WER + wall-clock | baseline | Lower WER / Less time | Similar advantage in the "blue zone" |
| Wikitext-103 / GPT-2 | PPL + wall-clock | baseline | Comparable or better than Shampoo | Consistent trends (Appendix J) |

### Ablation Study

| Configuration | Training Loss | Wall-clock Time | Description |
|---|---|---|---|
| Full FOAM (Adapting $\epsilon$ + Trigger EVD) | Lowest | Least | Pareto optimal (blue triangle) |
| Increase $\mathbf{f}$ w/o Adaptive $\epsilon$ | Significant Increase | Less | Stale factors allow spectral drift; training fails |
| Adaptive $\epsilon$ w/o EVD Refresh | Moderate Increase | Significant Increase | $\epsilon$ rises to suppress error but preconditioning quality drops |
| EVD Count (FOAM vs Stale) | — | — | EVDs for $L$ factor reduced to $\approx 10\%$, $R$ factor to $\approx 5\%$ on ViT |

### Key Findings
- **Co-dependence of components**: Both "proxy-controlled damping" and "$\epsilon_{\max}$ triggered EVD" are necessary to reduce loss and wall-clock time simultaneously, validating the theoretical claim that $\mathbf{f}$ and $\epsilon$ must be controlled together.
- **Robustness**: FOAM's advantages are robust to hyperparameter sweeps, with most configurations in the $(\mathbf{f}, \tau, \epsilon_{\max}, \epsilon_0)$ space falling into the superior performance zone.
- **Damping Dynamics**: During training, $\epsilon_t$ actively rises as staleness accumulates and drops to $\epsilon_0$ after an EVD refresh, exhibiting a saw-tooth pattern. The stale baseline remains at a tiny constant, lacking a buffer against large drifts.

## Highlights & Insights
- **Unified Feedback Loop**: It integrates "when to refresh" and "how much damping" into a single loop. Manual hyperparameters in Shampoo are replaced by a control mechanism where $\epsilon$ provides temporary stability and $\epsilon_{\max}$ triggers a clean slate, a strategy applicable to any "periodic expensive refresh" scenario.
- **Operator Error vs. Residual**: The authors distinguish between the geometry of diagonalization residuals and inverse root operator errors. The latter focuses on directions sensitive to perturbation (small eigenvalues), providing a more "symptomatic" health signal for stale feature reuse.
- **Theoretical Grounding**: The $r(1-\beta^{\mathbf{f}}) R_{\mathrm{SG}}^4 / \epsilon_0^2$ term in the discounted regret is not just a mathematical bound but a guide for setting multiplicative feedback, matching the physical relationship between damping and staleness.

## Limitations & Future Work
- **Theoretical Scope**: Full regret proof is provided for $p=2$ (Shampoo2), while $p=4$ is in the appendix. Assumptions include loss convexity and sub-Gaussian gradient tails, which are only coarse approximations in deep learning.
- **Hyperparameters**: While robust, there are still four parameters $(\mathbf{f}, \tau, \epsilon_{\max}, \epsilon_0)$ to select. Left and right factors are handled independently; joint scheduling via global signals remains unexplored.
- **Future Directions**: Making $\epsilon_{\max}$ adaptive (e.g., tied to regret estimates) could eliminate a manual hyperparameter. Additionally, modeling staleness induced by network latency in asynchronous distributed training could move the control loop to the system level.

## Related Work & Insights
- **vs Stale Shampoo (Gupta et al., 2018)**: Replaces fixed schedules and $\epsilon$ with closed-loop variables driven by the proxy $h_t$.
- **vs SOAP / Purifying Shampoo (Eschenhagen et al., 2026)**: While they use diagonalization residuals to trigger refreshes, FOAM argues these residuals aren't directly related to inverse root errors and uses the operator error proxy instead.
- **vs Damaskinos et al., 2018**: Unlike post-hoc weighting of stale gradients in asynchronous SGD, FOAM suppresses operator error within the second-order preconditioner itself.
- **vs SPlus (Frans et al., 2026)**: Where SPlus uses engineering stabilizers for stale reuse, FOAM derives a provable relative error upper bound to drive both stability and triggers.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Using $\epsilon$ as a feedback variable and operator error as a trigger represents a clear and useful shift in perspective for the Shampoo family.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Covers three modalities with sufficient ablation and robustness checks; lacks only a cross-comparison of $p$ values.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Strong link between theory, motivation, algorithm, and experiments. Clarifies why damping acts as a suppressor and why relative error is a better health signal.
- **Value**: ⭐⭐⭐⭐☆ Reducing EVD calls by 80%+ without loss in accuracy makes this a drop-in optimizer optimization for teams training large models with Kronecker-factored preconditioners.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Taming the Loss Landscape of PINNs with Noisy Feynman-Kac Supervision: Operator Preconditioning and Non-Asymptotic Error Bounds](taming_the_loss_landscape_of_pinns_with_noisy_feynman-kac_supervision_operator_p.md)
- [\[NeurIPS 2025\] Revisiting Orbital Minimization Method for Neural Operator Decomposition](../../NeurIPS2025/optimization/revisiting_orbital_minimization_method_for_neural_operator_decomposition.md)
- [\[NeurIPS 2025\] Purifying Shampoo: Investigating Shampoo's Heuristics by Decomposing its Preconditioner](../../NeurIPS2025/optimization/purifying_shampoo_investigating_shampoos_heuristics_by_decomposing_its_precondit.md)
- [\[ICML 2026\] Adaptive Preconditioners Trigger Loss Spikes in Adam](adaptive_preconditioners_trigger_loss_spikes_in_adam.md)
- [\[ICLR 2026\] Constraint Matters: Multi-Modal Representation for Reducing Mixed-Integer Linear Programming](../../ICLR2026/optimization/constraint_matters_multi-modal_representation_for_reducing_mixed-integer_linear_.md)

</div>

<!-- RELATED:END -->
