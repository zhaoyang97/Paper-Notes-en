---
title: >-
  [Paper Note] Tighter Performance Theory of FedExProx
description: >-
  [ICLR 2026][Optimization & Theory][Paper Note] This paper revisits the federated extrapolated proximal method FedExProx, pointing out that its original theory was surprisingly no better than naive Gradient Descent (GD) on quadratic problems. By proposing a new analysis that "proves distance convergence first and then translates it to function value convergence," th
tags:
  - ICLR 2026
  - Optimization & Theory
date: 2026-05-08
content_hash: b4dde7ffe80d68e0
---
# Tighter Performance Theory of FedExProx

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=PHsFWKJhGv](https://openreview.net/forum?id=PHsFWKJhGv)  
**Code**: N/A  
**Area**: Federated Optimization / Convex Optimization Theory  
**Keywords**: Federated Learning, Extrapolation, Proximal Operator, Convergence Rate, Communication Complexity

## TL;DR
This paper revisits the federated extrapolated proximal method FedExProx, pointing out that its original theory was surprisingly no better than naive Gradient Descent (GD) on quadratic problems. By proposing a new analysis that "proves distance convergence first and then translates it to function value convergence," the authors bypass the $L_{\max}$ dependence that caused looseness. They provide the first rigorous proof that FedExProx's total time complexity can be **strictly superior to GD** in realistic federated scenarios where communication dominates computation, extending the conclusions to partial participation, adaptive extrapolation, and PŁ conditions.

## Background & Motivation

**Background**: Federated Learning aims to solve $\min_x f(x) = \frac{1}{n}\sum_{i=1}^n f_i(x)$ without centralizing data, where communication is the core bottleneck. A powerful class of methods is **proximal methods**: clients do not transmit gradients but compute the proximal operator $\mathrm{prox}_{\gamma f_i}(x) = \arg\min_z \{ f_i(z) + \frac{1}{2\gamma}\|z-x\|^2 \}$, followed by server aggregation. FedProx replaces local gradient steps in FedAvg with proximal steps, while **FedExProx** (Li et al., 2024) adds **extrapolation** on top of FedProx: it uses the update $x^{k+1} = x^k + \alpha_k(\frac{1}{n}\sum_i \mathrm{prox}_{\gamma f_i}(x^k) - x^k)$, where taking $\alpha_k > 1$ is faster than simple averaging. Its iteration complexity $O(L_\gamma(1+\gamma L_{\max})R^2/\varepsilon)$ was previously claimed to be superior to FedProx and FedExP.

**Limitations of Prior Work**: The authors calculated this seemingly elegant bound for quadratic optimization tasks and discovered an embarrassing fact—it is **no better than vanilla GD**. The communication complexity of GD is $O(LR^2/\varepsilon)$, whereas the FedExProx bound includes $L_{\max}$ (the maximum smoothness constant across all clients). Under heterogeneous data, $L_{\max} \gg L$ often holds. When considering "how much time one step takes," the optimal strategy under the old theory is to let $\gamma \to 0$, reducing FedExProx to GD and rendering the proximal operator useless.

**Key Challenge**: Is FedExProx itself insufficient, or is the analysis by Li et al. (2024) simply too loose? In other words, is the $L_{\max}$ dependence an inherent cost of the method or an artifact of the analysis?

**Goal**: (1) Rigorously characterize why the original analysis fails to show superiority over GD; (2) Construct a tighter analysis framework to theoretically prove FedExProx can outperform GD; (3) Extend the conclusions to partial participation, adaptive extrapolation, and PŁ function classes beyond quadratics.

**Key Insight**: The authors traced the $L_{\max}$ term back to a lemma (Lemma A.9) used in the original proof, which introduces a $\frac{1}{1+\gamma L_{\max}}$ factor when translating the descent of the Moreau envelope into the descent of the original function. By utilizing a proof path that avoids this lemma, it is possible to eliminate the $L_{\max}$ dependence.

**Core Idea**: **Prove convergence in terms of "distance" $\mathbb{E}\|\bar x - x^*\|^2 \le \varepsilon$ first, then use the $L$-smoothness of $f$ to translate this into function value convergence.** This bypasses the lemma introducing $L_{\max}$, resulting in a tighter linear rate depending only on $L_\gamma/\mu_\gamma^+$.

## Method

### Overall Architecture

This is a purely theoretical paper; the "Method" refers to a **new convergence analysis framework**. The argument proceeds in four steps: ① **Diagnosis**—translating the Li et al. (2024) bound to quadratic problems to prove it is never better than GD under a time model (Theorem 3.2), identifying $L_{\max}$ as the root cause; ② **Re-proof**—using the distance convergence path to establish a linear rate $O(\frac{L_\gamma}{\mu_\gamma^+}\log\frac{1}{\varepsilon})$ for non-strongly convex quadratic problems (Theorem 4.1), where $L_\gamma$ and $\mu_\gamma^+$ are the maximum and minimum non-zero eigenvalues of the average Moreau envelope $M^\gamma$; ③ **Time Modeling**—providing an explicit model for single-step cost $\pi(\gamma) = \eta + \tau(\gamma L_{\max}+1)$ (communication $\eta$ + computation $\tau$) to prove there exists $\gamma>0$ such that FedExProx is strictly faster than GD (Theorem 4.3); ④ **Generalization**—applying the same distance analysis to partial participation (nice sampling), adaptive extrapolation (GraDS / StoPS), and general functions satisfying the PŁ condition.

### Key Designs

**1. Diagnosing Old Analysis: FedExProx is Not Stronger than GD on Quadratics**

To prove the value of the new analysis, the authors first push the old analysis to its limit. They pair the iteration complexity $O(L_\gamma(1+\gamma L_{\max})R^2/\varepsilon)$ from Li et al. (2024) with a reasonable time-per-step assumption—"step time $\pi(\gamma)$ is non-decreasing with respect to $\gamma$" (Assumption 3.1, because larger $\gamma$ makes local proximal subproblems harder: the subproblem $f_i(z)+\frac{1}{2\gamma}\|z-x\|^2$ is $(L_i+1/\gamma)$-smooth and $1/\gamma$-strongly convex, requiring $\tilde O(\gamma L_i + 1)$ GD steps to solve). Thus, total time $T(\gamma) = \pi(\gamma)\cdot\frac{L_\gamma(1+\gamma L_{\max})R^2}{\varepsilon}$. Theorem 3.2 proves that for all $\gamma>0$:

$$T(\gamma) \;\ge\; \pi(0)\cdot\frac{L R^2}{\varepsilon},$$

where equality holds as $\gamma\to 0$ (FedExProx degrades to GD). This pinpointed that the root cause of the inequality is the $L_{\max}$ factor in $L_\gamma(1+\gamma L_{\max}) \ge L$, whereas GD only depends on $L$.

**2. Distance Convergence Bypassing $L_{\max}$: Establishing a Tighter Linear Rate**

The old analysis requires Lemma A.9 ($M^\gamma(x)-M^\gamma(x^*) \ge \frac{1}{1+\gamma L_{\max}}(f(x)-f(x^*))$) to prove $\mathbb{E}[f(\bar x)] - f(x^*)\le\varepsilon$ directly. The authors' masterstroke is **changing the convergence metric**: first proving that iterations contract linearly in distance $\mathbb{E}\|\bar x - x^*\|^2\le\varepsilon$, then using global $L$-smoothness of $f$ to translate this to function value differences. This path avoids Lemma A.9 entirely. Theorem 4.1 gives the new iteration complexity for non-strongly convex quadratics:

$$O\!\left(\frac{L_\gamma}{\mu_\gamma^+}\log\frac{1}{\varepsilon}\right),$$

where $\mu_\gamma^+$ is the **minimum non-zero eigenvalue** of $\nabla^2 M^\gamma$. For comparison, GD requires $O(\frac{L}{\mu^+}\log\frac{1}{\varepsilon})$. While structurally similar, the condition number $L_\gamma/\mu_\gamma^+$ can be much smaller than $L/\mu^+$—this represents the true gain from extrapolation and proximal operators.

**3. Computation + Communication Time Model: Proving Strict Superiority**

The authors decompose single-step time: local computation is constrained by the slowest client taking $\tilde O(\tau(\gamma L_{\max}+1))$, plus communication time $\eta$:

$$\pi(\gamma) = \eta + \tau(\gamma L_{\max}+1),\qquad T_\eta(\gamma) = \tilde O\!\left((\eta + \tau(\gamma L_{\max}+1))\cdot\frac{L_\gamma}{\mu_\gamma^+}\right).$$

Theorem 4.3 proves that the optimal $\gamma$ falls within an explicit interval and always satisfies $T_\eta(\gamma)\le T_{\mathrm{GD}}$. When communication dominates ($\eta/\tau \ge 2$), the optimal $\gamma$ is strictly greater than 0, making FedExProx strictly faster. Example 4.4 shows that for certain cases, FedExProx can be faster than GD by a factor of the condition number.

**4. Generalization to Partial Participation, Adaptive Extrapolation, and PŁ Conditions**

The same distance analysis is extended iteratively. **Partial Participation**: Using nice sampling, Theorem 5.1 gives the complexity $O(\frac{L_{\gamma,S}}{\mu_\gamma^+}\log\frac{1}{\varepsilon})$. **Adaptive Extrapolation** (Theorem 6.1): GraDS uses gradient diversity to set the extrapolation coefficient automatically, while StoPS uses a Polyak-style step size. Both are **semi-adaptive** to $\gamma$, converging for any $\gamma>0$ and approaching optimality if $\gamma$ is large enough without knowing optimal $\alpha$ beforehand. **Beyond Quadratics** (Theorem 7.2 / 7.5): By replacing $\mu_\gamma^+$ with the "PŁ constant," a similar linear rate is obtained for general functions satisfying the PŁ condition, requiring weaker assumptions than the original strong convexity requirement.

## Key Experimental Results

Experiments were designed to validate the theory. The objective function is a random quadratic $f(x)=\frac1n\sum_i\frac12 x^\top A_i x$ where $A_i$ has a minimum eigenvalue of 0 (non-strongly convex), $n=14$ clients. The core observation is how **total empirical time complexity changes with $\gamma$**.

### Main Results: U-shaped Curve of Optimal $\gamma$

| Communication Time $\eta$ | Optimal $\gamma$ Position | Theory Consistency | Meaning |
|------|------|------|------|
| Low ($\eta\propto 100$) | Near 0 | Consistent with Thm 4.3 | Computation is expensive $\rightarrow$ use fewer prox steps $\rightarrow$ approx GD |
| Medium | Shifted toward $>0.1$ | Consistent | Communication gets expensive $\rightarrow$ prox steps become profitable |
| High ($\eta\propto 10^5$) | Clearly $>0$ | Consistent | Communication dominates $\rightarrow$ extrapolated proximal steps significantly better |

As $\eta$ increases, the total time vs. $\gamma$ curve exhibits a clear **U-shape**. The optimal $\gamma$ moves from near 0 to above 0.1, always staying within the theoretical bounds predicted by Theorem 4.3. This directly validates the core claim: **The more communication dominates, the more advantageous a non-trivial $\gamma>0$ becomes.**

### Key Findings
- **U-shape is Critical Evidence**: If the old analysis were true, the optimal $\gamma$ would always be at 0. The measured U-shape proves an interior optimal $\gamma^*>0$ exists.
- **Tight Theoretical Bounds**: Empirical optimal $\gamma$ values fall within the narrow intervals predicted by Theorem 4.3.
- **Phenomena Beyond Quadratics**: Similar patterns were observed on real datasets like ARCENE (Appendix G).

## Highlights & Insights
- **"Changing Convergence Metric" is the Key**: Avoiding $L_{\max}$ by proving distance convergence first is a clever technical maneuver that provides a template for other federated/distributed analyses stuck with $L_{\max}$ dependencies.
- **Explicit Time Modeling**: By modeling $\pi(\gamma)=\eta+\tau(\gamma L_{\max}+1)$, the paper moves beyond simple iteration counts to meaningful federalized complexity comparisons.
- **Rigorous Self-Critique**: The paper starts by proving the existing analysis makes the algorithm look bad compared to GD, creating strong logical tension before providing the "redemption" proof.
- **Weaker Assumptions for Stronger Results**: Moving from strong convexity to PŁ allows for solutions in over-parameterized models while improving dependence on problem constants.

## Limitations & Future Work
- **Focus on Quadratic/PŁ**: The most refined results (explicit optimal $\gamma$ intervals) are for quadratics. General non-convex deep networks remain an open challenge.
- **Interpolation Assumption**: The theory relies on Assumption 1.3 (existence of $x^*$ such that all $\nabla f_i(x^*)=0$), which holds for over-parameterized models but not necessarily for general heterogeneous data.
- **Idealized Time Model**: The model assumes constant $\eta$ and linear computation scaling with $\gamma L_{\max}+1$, which might not capture network jitter or complex local solver behavior.
- **Cost of StoPS**: While adaptive, StoPS requires knowing the minimum value of the average Moreau envelope.

## Related Work & Insights
- **vs. Li et al. (2024)**: Same algorithm, but this paper provides a **tighter analysis** and weakens the requirement from strong convexity to PŁ.
- **vs. GD**: Proves FedExProx is strictly better in communication-dominated scenarios, quantifying gains by a factor of the condition number.
- **vs. FedProx / FedExP**: While FedExProx original theory claimed superiority over these, this paper reinforces that claim by showing it is even better than the stronger GD baseline.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[ICLR 2026\] Elastic Optimal Transport: Theory, Application, and Empirical Evaluation](elastic_optimal_transport_theory_application_and_empirical_evaluation.md)
- [\[ICLR 2026\] Bilevel Optimization with Lower-Level Uniform Convexity: Theory and Algorithm](bilevel_optimization_with_lower-level_uniform_convexity_theory_and_algorithm.md)
- [\[NeurIPS 2025\] Learning Theory for Kernel Bilevel Optimization](../../NeurIPS2025/optimization/learning_theory_for_kernel_bilevel_optimization.md)
- [\[ICML 2026\] Towards Understanding Continual Factual Knowledge Acquisition of Language Models: From Theory to Algorithm](../../ICML2026/optimization/towards_understanding_continual_factual_knowledge_acquisition_of_language_models.md)
- [\[ICML 2026\] Adaptive Sharpness-Aware Minimization with a Polyak-type Step size: A Theory-Grounded Scheduler](../../ICML2026/optimization/adaptive_sharpness-aware_minimization_with_a_polyak-type_step_size_a_theory-grou.md)

</div>

<!-- RELATED:END -->
