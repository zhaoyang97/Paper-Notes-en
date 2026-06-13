---
title: >-
  [Paper Note] On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective
description: >-
  [ICML 2026][Learning Theory / Test-Time Adaptation / Non-stationary Online Learning][TTA Learnability] This paper establishes the first theoretical framework for Test-Time Adaptation (TTA) learnability. It introduces $(\…
tags:
  - "ICML 2026"
  - "Learning Theory / Test-Time Adaptation / Non-stationary Online Learning"
  - "TTA Learnability"
  - "Recovery Complexity"
  - "Wasserstein Quantization"
  - "ϕ-mixing"
  - "minimax lower bound"
date: 2026-05-08
content_hash: 1db6141f93b925d0
---

# On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective

**Conference**: ICML 2026  
**arXiv**: [2605.28057](https://arxiv.org/abs/2605.28057)  
**Code**: None  
**Area**: Learning Theory / Test-Time Adaptation / Non-stationary Online Learning  
**Keywords**: TTA Learnability, Recovery Complexity, Wasserstein Quantization, ϕ-mixing, minimax lower bound

## TL;DR
This paper establishes the first theoretical framework for Test-Time Adaptation (TTA) learnability. It introduces $(\epsilon,\delta)$-Recovery Complexity to measure the time required for a model to reduce excess risk to $\epsilon$ after a distribution shift. By extending local recovery to non-stationary test streams via $(\epsilon,\rho)$-TTA Learnability, the authors derive matching minimax upper/lower bounds, revealing the intrinsic "adaptation speed—information constraint" trade-off in TTA.

## Background & Motivation

**Background**: TTA has achieved empirical success across vision, tabular, and NLP tasks (e.g., Tent, CoTTA, NOTE, ODS). The paradigm involves updating models online using only unlabeled test data guided by a proxy loss $\psi$. However, these methods often fail under complex distribution shifts, leading the community to question the conditions under which TTA is reliable.

**Limitations of Prior Work**: On one hand, online learning theory (regret framework) characterizes cumulative performance but fails to address instantaneous reliability questions such as "how long does recovery take after a shift?" On the other hand, existing TTA theoretical analyses (e.g., AdaNPC restricted to memory-based methods, ATTA assuming active labeling) impose excessively strong assumptions on algorithms or label availability, failing to support general unlabeled stream scenarios.

**Key Challenge**: The objective of TTA is essentially to maintain acceptable instantaneous risk at every time step. Current theoretical frameworks either do not characterize post-shift recovery or do not account for the information constraints caused by the mismatch between proxy tasks and true losses on unlabeled streams.

**Goal**: To construct a unified framework capable of expressing (i) both continuous and abrupt distribution shifts, (ii) temporal correlation, (iii) proxy-task mismatch, and (iv) post-shift recovery speed, while deriving matching minimax bounds under a stochastic proxy-gradient oracle.

**Key Insight**: The test stream is first discretized into a "piecewise stationary" approximation using Wasserstein-1 distance, allowing each segment to be analyzed as an independent recovery problem. Then, ϕ-mixing coefficients characterize intra-batch temporal dependencies. Finally, Le Cam’s two-point method is used to characterize information-theoretic lower bounds.

**Core Idea**: By defining "Recovery Complexity" $\tau(\epsilon,\delta)$, instantaneous reliability is transformed into an analyzable complexity measure. Multiple recovery behaviors are then aggregated into a global $(\epsilon,\rho)$-TTA Learnability, reducing the TTA learnability problem to the sample/batch complexity analysis of classical stochastic optimization.

## Method

### Overall Architecture
The theoretical framework consists of four components: (1) Test stream formalization—distribution shifts quantified by Wasserstein-1 + temporal dependencies via ϕ-mixing; (2) Competitive target—the proxy-optimal competitor $\theta_t^\star \in \arg\min_{\theta\in\mathcal{N}_r(\theta_1)} \psi_t(\theta)$ and its true task risk $R_t:=\ell_t(\theta_t^\star)$; (3) Local analysis—minimax upper/lower bounds for $(\epsilon,\delta)$-recovery complexity after a single shift; (4) Global aggregation—summarizing recovery behaviors into $(\epsilon,\rho)$-TTA Learnability and comparing it with dynamic regret. The input is a non-stationary stream $\mathcal{S}=\{D_t\}_{t=1}^T$, and the output provides provable bounds regarding the parameter regimes ($\alpha$ alignment, $\zeta$ mismatch bias, batch size $B$, mixing constant $C_\phi$, shift magnitude $\Delta_W$) under which TTA is learnable.

### Key Designs

1.  **Wasserstein-1 Quantized Distribution Shift Approximation**:
    - **Function**: Compresses a non-stationary trajectory $\{\mathcal{P}_t\}$ (mixing continuous and abrupt shifts) into piecewise stationary approximations $\{\tilde{\mathcal{P}}_t\}$, ensuring the approximation error does not exceed $\Delta_W/2$.
    - **Mechanism**: A greedy algorithm maintains an anchor; if the $W_1$ distance between the next step $\mathcal{P}_t$ and the anchor is $\le \Delta_W/2$, the anchor is inherited (shift indicator $\tilde{S}_t=0$); otherwise, a shift is declared, the anchor is reset, and $\tilde{S}_t=1$. The total number of shifts satisfies $\tilde{K}_S(T) \le \lceil 2V_T/\Delta_W\rceil$, where $V_T:=\sum_t W_1(\mathcal{P}_t,\mathcal{P}_{t+1})$.
    - **Design Motivation**: Choosing $W_1$ over KL/TV allows for a unified characterization of covariate + label shifts in the joint space $\mathcal{X}\times\mathcal{Y}$ and supports two-point constructions in lower bounds. Piecewise stationary approximation reduces "global non-stationary analysis" to "independent segment recovery analysis."

2.  **ϕ-mixing Temporal Dependency and Effective Batch Size**:
    - **Function**: Characterizes non-i.i.d. samples within a batch and reduces it to an equivalent "effective sample size" parameter.
    - **Mechanism**: Assuming geometric decay $\phi(i)\le \varrho^i$, the variance of the batch-mean gradient for a correlated batch of size $B$ is equivalent to an i.i.d. batch of size $B_{\text{eff}} = B/C_\phi$, where $C_\phi = 1 + 4\varrho^{1/2}/(1-\varrho^{1/2})$. When $\varrho=0$, it reduces to $B_{\text{eff}}=B$.
    - **Design Motivation**: This distills complex stochastic process properties into a scalar $C_\phi$ within the complexity bound, allowing the lower bound $\tau \gtrsim C_\phi/B \cdot 1/(\alpha(\sqrt{\zeta+2\alpha\epsilon}+\sqrt{\zeta})^2)$ to reflect the impact of both batch size and intra-batch independence.

3.  **Minimax upper/lower bounds of $(\epsilon,\delta)$-Recovery Complexity**:
    - **Function**: Formalizes "how fast the excess risk $\mathcal{E}_t:=\ell_t(\theta_t)-R_t$ can be controlled below $\epsilon$ with failure probability $\le \delta$ after a shift" as complexity $\tau(\epsilon,\delta):=\inf\{t: \sup_{u\ge t}\mathbb{P}(\mathcal{E}_u>\epsilon)\le \delta\}$.
    - **Mechanism**: The lower bound uses Le Cam’s method to construct two hard cases with a $W_1$ distance exactly $\Delta_W$, proving that any stochastic proxy-gradient oracle algorithm requires $\tau \ge \Omega\big(\frac{C_\phi}{B}\cdot \frac{1}{\alpha(\sqrt{\zeta+2\alpha\epsilon}+\sqrt{\zeta})^2}\big)$. The upper bound analyzes a simplified TTA baseline under $L$-smooth + PL conditions to obtain a matching order $\tau \le \tilde{O}(\cdot)$.
    - **Design Motivation**: The matching order is the most significant result, revealing three facts: (i) as $\epsilon\to 0$, complexity does not vanish if $\zeta>0$, indicating a proxy mismatch error floor; (ii) $\tau$ scales with $1/\alpha^2$, showing a quadratic amplification effect of proxy alignment; (iii) $\tau$ does not explicitly depend on $\Delta_W$, suggesting shift magnitude only determines "whether to adapt," while recovery difficulty is determined by $\alpha,\zeta,B,C_\phi$.

### Loss & Training
This is a theoretical analysis and does not introduce a new training algorithm. The TTA baseline used is stochastic proxy-gradient descent on $\psi$ with step size $\eta$, updating within a local neighborhood $\mathcal{N}_r(\theta_1)$. Key hyperparameters are $\eta$ and $B$. The analysis relies on (Assumption 2.1) $(\alpha,\zeta)$-Alignment, (Assumption 2.2) $L$-smooth + PL + bounded gradient variance $\sigma^2$, and (Assumption 2.4/2.7) Wasserstein Quantization + ϕ-mixing.

## Key Experimental Results

### Main Results
The "Main Results" focus on Theorem 3.3 (Lower Bound), Theorem 3.9 (Upper Bound), and the extension to TTA Learnability in Section 4. The following table compares this framework with existing theoretical tools across key TTA dimensions.

| Framework | Instantaneous Reliability | Unlabeled Proxy Mismatch | Non-stationary Stream Shape | Temporally Correlated Batch | Information Lower Bound |
|-----------|---------------------------|--------------------------|-----------------------------|-----------------------------|-------------------------|
| Static Online Regret (Shalev-Shwartz) | No | No | Fixed comparator only | Weak | No |
| Dynamic Regret (Zhao et al.) | Partial | No | Arbitrary variation | Weak | No |
| AdaNPC (Zhang et al.) | No | Memory-based only | Covariate only | No | No |
| ATTA (Gui et al.) | Partial | Requires active label | General | No | No |
| **Ours** | **Yes** | **Explicitly $\alpha,\zeta$** | **$W_1$ Quantization** | **ϕ-mixing → $C_\phi$** | **Le Cam Matching** |

### Ablation Study
The table below shows the degradation of the recovery complexity lower bound when various factors are disabled (corresponding to Remarks 3.4–3.7).

| Configuration | Lower Bound Magnitude | Description |
|---------------|-----------------------|-------------|
| Full bound | $\Omega\big(\frac{C_\phi}{B\alpha}(\sqrt{\zeta+2\alpha\epsilon}+\sqrt{\zeta})^{-2}\big)$ | Full matching order lower bound |
| $\zeta=0$ (Perfect alignment) | $\Omega(C_\phi/(B\alpha^2\epsilon))$ | Matching classical stochastic optimization bounds |
| $\varrho=0$ (Independent batch, $C_\phi=1$) | $\Omega(1/(B\alpha^2\epsilon))$ | Temporal correlation term disappears |
| $\alpha\downarrow$ | $\tau\uparrow$ as $1/\alpha^2$ | Weaker alignment leads to slower recovery |
| Changing $\Delta_W$ | Bound unchanged | Shift magnitude only triggers recovery, doesn't affect intrinsic difficulty |

### Key Findings
- **Proxy mismatch $\zeta$ creates an error floor**: The lower bound does not go to zero as $\epsilon\to 0$, meaning any unlabeled TTA has an irreducible minimum risk that cannot be suppressed by more samples.
- **Batch size $B$ and ϕ-mixing $C_\phi$ jointly dominate difficulty**: In practice, increasing batch size cannot compensate for high temporal correlation; one must consider the effective batch $B/C_\phi$.
- **Alignment intensity $\alpha$ has a quadratic impact**: This suggests that designing more aligned proxy losses (e.g., improved entropy, auxiliary supervision aligned to task gradients) yields high returns.
- **TTA Learnability vs. Dynamic Regret**: While formally related, they have different goals. The former requires every segment to recover below $\epsilon$, while the latter concerns cumulative gaps. This paper provides conversion conditions and counterexamples.

## Highlights & Insights
- Using Wasserstein-1 to discretize arbitrary non-stationary trajectories into finite stationary segments provides the "backbone" for reducing non-stationary analysis to "piecewise stationary + shift counting." This approach is transferable to other streaming learning scenarios like continual learning or online RL.
- Compressing ϕ-mixing into a scalar $C_\phi$ allows temporal correlation to be quantified alongside batch size and alignment, a technique valuable for future theoretical modeling.
- The error floor introduced by $\zeta$ conceptually matches noisy supervision, providing a formal explanation for the failure of self-training/pseudo-labeling TTA methods. This suggests that algorithm design should prioritize reducing $\zeta$ (e.g., better proxy design, label refinement) over pursuing $\epsilon\to 0$.

## Limitations & Future Work
- The entire analysis is conducted within a local neighborhood $\mathcal{N}_r(\theta_1)$, assuming PL + smoothness hold and that updates do not leave this region. This may not hold under large distribution shifts or long sequences.
- The upper bound depends on a simplified TTA baseline (stochastic proxy-gradient descent); a gap remains between this and practical methods like Tent/CoTTA which use BN-only updates or teacher-student mechanisms.
- The framework is more about "measurement" than "guided design." Future work could involve using estimates of $\alpha,\zeta,C_\phi$ to adapt step sizes online or decide reset timing.
- Lack of empirical validation (typical for theory papers); quantifying complexity factors on TTA benchmarks like CIFAR-C / ImageNet-C is a valuable next step.

## Related Work & Insights
- **vs. Dynamic Regret (Zhao et al., 2024)**: Dynamic regret focuses on cumulative gaps; this paper focuses on recovery time per segment. The authors provide conversion conditions and distinguish the different semantics of "non-stationary learning ability."
- **vs. AdaNPC (Zhang et al., 2023)**: AdaNPC provides bounds only for memory-based non-parametric classifiers; this paper is algorithm-agnostic by abstracting analysis to the algorithm-class level using a stochastic gradient oracle.
- **vs. ATTA (Gui et al., 2024)**: ATTA assumes active labeling to simplify the problem to supervised online learning. This paper maintains the unlabeled setting, explicitly modeling the proxy-task mismatch $\zeta$ as the fundamental source of difficulty.
- **vs. Tent/CoTTA/NOTE Empirical Work**: These methods handle covariate shifts, sequential shifts, and correlated streams. This framework explains their respective "difficulty factors" ($\Delta_W, V_T, \varrho$) and points to shared trade-offs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First theoretical framework for TTA learnability; innovatively decomposes non-stationary problems into analyzable recovery segments.
- Experimental Thoroughness: ⭐⭐ Pure theory paper without empirical validation; explanatory power for actual TTA algorithms needs follow-up work.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with logical flow between definitions, theorems, and remarks; however, high symbol density may be a barrier for non-theory readers.
- Value: ⭐⭐⭐⭐ Provides a provable language for the TTA community, establishing formal objectives for future proxy loss and reset strategy designs.

## Rating
- Novelty: To be rated
- Experimental Thoroughness: To be rated
- Writing Quality: To be rated
- Value: To be rated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MMD-Balls as Credal Sets: A PAC-Bayesian Framework for Epistemic Uncertainty in Test-Time Adaptation](mmd-balls_as_credal_sets_a_pac-bayesian_framework_for_epistemic_uncertainty_in_t.md)
- [\[ICML 2026\] Semi-Supervised Noise Adaptation: Transferring Knowledge from Noise Domain](semi-supervised_noise_adaptation_transferring_knowledge_from_noise_domain.md)
- [\[NeurIPS 2025\] The Parameterized Complexity of Computing the VC-Dimension](../../NeurIPS2025/learning_theory/the_parameterized_complexity_of_computing_the_vc-dimension.md)
- [\[NeurIPS 2025\] The Structural Complexity of Matrix-Vector Multiplication](../../NeurIPS2025/learning_theory/the_structural_complexity_of_matrix-vector_multiplication.md)
- [\[ICML 2026\] Correcting Split Selection in Online Decision Trees via Anytime-Valid Inference](correcting_split_selection_in_online_decision_trees_via_anytime-valid_inference.md)

</div>

<!-- RELATED:END -->
