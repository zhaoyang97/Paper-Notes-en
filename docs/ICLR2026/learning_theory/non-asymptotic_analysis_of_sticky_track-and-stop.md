---
title: >-
  [Paper Note] Non-Asymptotic Analysis of (Sticky) Track-and-Stop
description: >-
  [ICLR 2026][Learning Theory][best-arm identification] This paper provides the first **non-asymptotic (finite-confidence) sample complexity upper bounds** for two classic algorithms in the pure exploration field: Track-and-Stop (TAS) and Sticky Track-and-Stop (S-TAS). It fills the theoretical gap where these algorithms were proven optimal only as $\delta\to0$, with unknown performance for finite $\delta$.
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Pure Exploration"
  - "Multi-Armed Bandits"
  - "best-arm identification"
  - "Track-and-Stop"
  - "Sticky Track-and-Stop"
  - "finite confidence"
  - "sample complexity"
  - "non-asymptotic analysis"
date: 2026-05-08
content_hash: 9c89cac6180b5cae
---

# Non-Asymptotic Analysis of (Sticky) Track-and-Stop

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=vebqP5aioj](https://openreview.net/forum?id=vebqP5aioj)  
**Code**: To be confirmed  
**Area**: Learning Theory / Pure Exploration / Multi-Armed Bandits  
**Keywords**: pure exploration, best-arm identification, Track-and-Stop, Sticky Track-and-Stop, finite confidence, sample complexity, non-asymptotic analysis  

## TL;DR
This paper provides the first **non-asymptotic (finite-confidence) sample complexity upper bounds** for two classic algorithms in the pure exploration field: Track-and-Stop (TAS) and Sticky Track-and-Stop (S-TAS). It fills the theoretical gap where these algorithms were proven optimal only as $\delta\to0$, with unknown performance for finite $\delta$.

## Background & Motivation
- **Background**: In the pure exploration problem, a statistician sequentially samples from $K$ unknown distributions (arms) to answer a question about the environment (e.g., identifying the best arm) with an error rate not exceeding a risk parameter $\delta$, using as few samples as possible. **Track-and-Stop (TAS)**, proposed by Garivier & Kaufmann (2016), is a foundational method that is **asymptotically optimal** as $\delta\to0$ when the answer mapping $i^\star(\mu)$ is single-valued (e.g., a unique best arm). Degenne & Koolen (2019) introduced **Sticky TAS (S-TAS)**, extending asymptotic optimality to multi-valued answer settings (e.g., $\epsilon$-best arm identification, where multiple arms are correct).
- **Limitations of Prior Work**: The optimality proofs for both are **asymptotic**, characterizing behavior in the limit $\delta\to0$, but providing no guarantees for the finite-confidence interval used in practice. Several works (e.g., Barrier 2023) pointed out that the root cause lies in the instability: when data is scarce, empirical means have not converged, causing the TAS sampling rule to fluctuate wildly, making it difficult to analyze; the issue is even more severe for S-TAS.
- **Key Challenge**: To bypass this instability, existing works either add confidence intervals to TAS (Degenne et al. 2019, which requires solving a difficult max-max-min problem without an efficient oracle) or "bias" the sampling rule toward uniform exploration (Barrier et al. 2022, at the cost of empirical performance). Yet, empirically, **original TAS consistently remains the fastest**. This leads to the question: Can finite-confidence guarantees be proven for the original TAS / S-TAS **without modifying the algorithms**?
- **Goal**: To characterize the non-asymptotic guarantees of TAS and S-TAS, ensuring the conclusions hold for **any structured problem and any answer correspondence**.
- **Core Idea**: **[Bypassing convergence arguments in favor of an "information accumulation" perspective]** Instead of analyzing where the sampling proportions $N(t)/t$ converge (the core of previous fragile asymptotic proofs), this work uses a sequence of "good events" $E_t$ to **lower-bound the stopping rule using the algorithm's own sampling rule**. By injecting the optimal rate $T^\star(\mu)^{-1}$ through a generalized oracle weight expression, a finite $\delta$ upper bound is directly derived.

## Method

### Overall Architecture
This paper does not propose a new algorithm but rather a **non-asymptotic proof framework**. The two main theorems (Theorem 1 for TAS, Theorem 2 for S-TAS) share a similar form: the expected stopping time is bounded by $\mathbb{E}_\mu[\tau_\delta]\le 2eK+10K^4+T_0(\delta)$, where $T_0(\delta)$ is the minimum time such that $\beta_{t,\delta}\le tT^\star(\mu)^{-1}-g(t)$, characterizing how fast the algorithm accumulates information to cross the stopping threshold $\beta_{t,\delta}$. The argument follows a three-step process: good event reduction → lower-bounding the stopping rule → injecting the optimal rate.

```mermaid
flowchart LR
  A[Good Event E_t<br/>Empirical Mean Concentration] --> B[Reduction:<br/>E[τ_δ] ≤ T̄ + 2eK]
  B --> C[Lower-bounding Stopping Rule<br/>≈ Sum of Sampling Rules]
  C --> D[Injecting Generalized Oracle Weight ω⋆<br/>Deriving T⋆μ⁻¹]
  D --> E[T_0δ is Finite<br/>and recovers Asymptotic Optimality as δ→0]
```

### Key Designs

**1. Good Event Reduction: Splitting expected stopping time into "bad event probability + deterministic stopping time."** The analysis starts by defining a family of good events $E_t=\{\forall s\in[\lceil\sqrt t\rceil,t]:\sum_k N_k(s)d(\hat\mu_k(s),\mu_k)\le 8K\log s\}$, meaning empirical means are sufficiently close to true values at non-early stages. This has two complementary properties: first, bad event probabilities are summable, $\sum_{t\ge3}\mathbb{P}_\mu(E_t^c)\le 2eK$ (contributing the $2eK$ in the bound); second, there exists a threshold $\bar T$ such that $t\ge\bar T$ under $E_t$ implies stopping, i.e., $E_t\subseteq\{\tau_\delta\le t\}$. Combining these yields $\mathbb{E}_\mu[\tau_\delta]\le\bar T+2eK$, converting the problem of finding expected stopping time into finding a deterministic $\bar T$. This technique draws from the finite-confidence analysis of Degenne et al. (2019) and Jourdan & Degenne (2023), diverging completely from the asymptotic approach dependent on $N(t)/t$ convergence.

**2. Coupling Stopping Rules and Sampling Rules: The technical core.** The challenge is that **stopping** depends on $\max_i\inf_\lambda\sum_k N_k(t)d(\hat\mu_k(t),\lambda_k)\ge\beta_{t,\delta}$, while **sampling** uses $\inf_{\lambda\in\neg i_s}\sum_k\omega_k(s)d(\tilde\mu_k(s),\lambda_k)$ at each step. The key observation is that under $E_t$, C-Tracking ensures $N(t)\approx\sum_{s}\omega(s)$ and $d(\hat\mu_k,\cdot)\approx d(\mu_k,\cdot)$. Thus, the stopping rule can be **lower-bounded** by the cumulative sum of the algorithm's own sampling objectives (up to a sub-linear term $\widetilde O(\sqrt t)$). Once coupled, for the single-valued TAS case, the bound can be narrowed using "if $i_s\ne i^\star(\mu)$ then $\mu\in\neg i_s$." This step explicitly relies on single-valued answers, explaining why TAS fails in multi-answer settings.

**3. Generalizing Oracle Weights beyond $M$: Enabling injection of the optimal rate.** In classic literature, the optimal rate is written as $T^\star(\mu)^{-1}=\sup_\omega\inf_{\lambda\in\neg i^\star(\mu)}\sum_k\omega_k d(\mu_k,\lambda_k)$ (Eq. 1), but $i^\star$ is only defined for $\mu\in M$, whereas the empirical estimate $\hat\mu(t)$ might fall outside $M$. This paper uses an equivalent rewrite $T^\star(\mu)^{-1}=\sup_\omega\max_{i\in I}\inf_{\lambda\in\neg i}\sum_k\omega_k d(\mu_k,\lambda_k)$ (Eq. 2/4), naturally extending the definition of oracle weights to any model. This rewrite allows the analysis to introduce an arbitrary $\omega^\star\in\omega^\star(\mu)$ in intermediate steps, ultimately quantifying accumulated information as $(t-\sqrt t-1)T^\star(\mu)^{-1}-\widetilde O(t^{3/4})$, ensuring $T_0(\delta)$ is finite and accurately recovers the $T^\star(\mu)\log(1/\delta)$ rate as $\delta\to0$. Two mild assumptions ($\sigma^2$-sub-Gaussian, bounded parameters $M\subset[\mu_{\min},\mu_{\max}]$) along with a projection $\tilde\mu$ for $\hat\mu$ are used to keep the KL divergence $d(\tilde\mu_k(s),\cdot)$ "well-behaved" for small $t$; the paper also explains how to remove the projection at the cost of an extra $T_M$ term.

**4. S-TAS Specific "Answer Set Collapse Time" $T_\mu$: Handling multi-answer difficulties.** In multi-answer settings, S-TAS has little control over the selected answer $i_s$ in early stages (relying on a fixed total order on $I$), and cannot directly switch from $\neg\bar\imath$ to $\neg i_s$ like TAS. This work uses the **upper semi-continuity** of $\mu\mapsto i_F(\mu)$ to prove (Lemma 4) the existence of a time $T_\mu$: for $t\ge T_\mu$, under $E_t$, all candidate models $\mu'$ in the confidence region $C_s$ satisfy $i_F(\mu')\subseteq i_F(\mu)\cup(I\setminus i^\star(\mu))$—meaning bad answers are excluded. $T_\mu$ is determined by $\epsilon_\mu$, the distance from $\mu$ to the answer-flipping boundary (in $\epsilon$-best-arm, $\epsilon_\mu=\Delta_\mu/2$). Consequently, $T_0(\delta)$ in Theorem 2 only differs from TAS by a problem-dependent constant $T_\mu T^\star(\mu)^{-1}$. Notably, this proof **does not rely** on where $N(t)/t$ converges or the convexity of $\omega^\star(\mu,\neg\bar\imath)$ (the pillars of the asymptotic proof in Degenne & Koolen 2019), proceeding entirely in the language of "information accumulation."

## Key Experimental Results

This is a **purely theoretical work with no experiments**. Its "key results" are presented as two sample complexity theorems, summarized below.

### Main Results: Two Non-Asymptotic Upper Bounds

| Algorithm | Setting | $\mathbb{E}_\mu[\tau_\delta]$ Upper Bound | $g(t)$ Magnitude |
|------|----------|-----------------------------------|-------------|
| TAS (Thm 1) | Single answer $\lvert i_F(\mu)\rvert=1$ | $2eK+10K^4+T_0(\delta)$, where $T_0(\delta)=\inf\{t:\beta_{t,\delta}\le tT^\star(\mu)^{-1}-g(t)\}$ | $64\sigma DLK^2\log K\sqrt{t}\log^2 t+16\sigma D\sqrt{K}t^{3/4}\log t$ |
| S-TAS (Thm 2) | Any multi-answer | $2eK+10K^4+T_0(\delta)$, where $T_0(\delta)=\inf\{t:\beta_{t,\delta}\le (t-T_\mu)T^\star(\mu)^{-1}-g(t)\}$ | $80\sigma DLK^2\log K\sqrt{t}\log^2 t+32\sigma D\sqrt{K}t^{3/4}\log t$ |

### Quantitative Comparison

| Dimension | Conclusion |
|------|------|
| Main term w.r.t. $\delta$ | $T_0(\delta)\sim T^\star(\mu)\log(1/\delta)$ (ignoring poly-log factors/constants), **precisely recovering asymptotic optimality** as $\delta\to0$. |
| $\delta$-independent terms | $2eK$ (sum of bad event probabilities) + $10K^4$ (technical requirement $t\ge10K^4$), purely artifacts of the analysis. |
| Sub-linear remainder | $\beta_{t,\delta}+g(t)$ grows as $O(\log(1/\delta)+t^{3/4})$, eventually surpassed by the linear term $tT^\star(\mu)^{-1}$, ensuring $T_0(\delta)$ is finite. |
| Projection-free cost | Without projection, for the original TAS by Garivier & Kaufmann (2016), the bound adds only a problem-dependent term $T_M$. |

### Key Findings
- TAS has been empirically fast at moderate $\delta$ (per previous benchmarks); this work completes the theoretical explanation that it indeed enjoys finite-confidence guarantees.
- S-TAS is the **only known algorithm** in literature that optimally solves pure exploration for any multi-answer setting; this work provides the **first** finite-confidence analysis for this general class of problems.
- In single-answer instances, the $T_\mu$ dependency in S-TAS can be removed, making it isomorphic to TAS (up to constant factors), though the proofs remain fundamentally different due to sampling rules.

## Highlights & Insights
- **"Modify the Proof, Not the Algorithm"**: Bypasses the two common routes of modifying algorithms—adding confidence intervals (Degenne et al. 2019, which introduces max-max-min issues) or biasing sampling (Barrier et al. 2022, which sacrifices performance)—to provide the first finite-confidence guarantee for the original TAS.
- **Methodological Shift**: Replaces "convergence of sampling proportions" with "information accumulation" as the object of analysis. This bypasses the fundamental hurdle where $N(t)/t$ converges to a convex hull rather than the oracle weights in multi-answer settings, which was the key reason TAS failed while S-TAS could be proven non-asymptotically.
- **Small Rewrite, Big Impact**: Rewriting $T^\star(\mu)^{-1}$ using a $\max$ over answers (Eq. 2) seems trivial but allows oracle weights to be defined outside $M$, enabling the proof to proceed even when empirical estimates stray from the model space.

## Limitations & Future Work
- **Upper bounds contain loose terms like $10K^4$ and $T_\mu$**: The authors admit $2eK$ and $10K^4$ are artifacts of the analysis, and $T_0(\delta)$ is only implicitly defined (though an explicit bound is in the appendix). These terms may not be tight when $K$ or $T_\mu$ is large, leaving a gap with lower bounds.
- **Dependency on Two Assumptions**: Sub-Gaussian exponential families and bounded parameters $M$ are mild but exclude heavy-tailed distributions. Removing the projection step adds the cost of $T_M$.
- **No Matching Lower Bounds**: This work provides only upper bounds and does not prove whether these non-asymptotic bounds are information-theoretically optimal regarding parameters like $K$ or $\Delta_\mu$, leaving this as an open question.
- **No Numerical Validation**: As a purely theoretical paper, it lacks experiments demonstrating the actual tightness of the bounds at finite $\delta$, making it hard for readers to gauge the magnitude of the constants.

## Related Work & Insights
- **Single-answer + TAS Lineage**: Founded by Garivier & Kaufmann (2016) and extended to structured settings like Lipschitz or unimodal (Wang et al. 2021; Poiani et al. 2024; Kaufmann & Koolen 2021), but optimality remained at the asymptotic level.
- **Bypassing Instability**: Optimistic TAS (Degenne et al. 2019) and biased sampling (Barrier et al. 2022) are the closest finite-confidence works; this paper contributes by analyzing the original versions directly.
- **Multi-answer + S-TAS**: Degenne & Koolen (2019) introduced S-TAS and its asymptotic optimality, proving the upper semi-continuity of $\mu\mapsto i_F(\mu)$—which became the pivot for the $T_\mu$ argument here. Previous multi-answer finite-confidence results were limited to subclasses like $\epsilon$-best-arm (Even-Dar 2002; Jourdan et al. 2023); this is the first to cover arbitrary multi-answer problems.
- **Insight**: When an asymptotic proof relies on a fragile "converges to where" argument, shifting to a lower-bounded perspective of "how much information was accumulated" can be more robust and extendable to general settings. This logic could potentially transition to the finite-sample analysis of other sequential decision algorithms based on lower-bound games.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Resolves two open problems (finite-confidence analysis of original TAS and general multi-answer S-TAS) with a clean methodological shift from "convergence" to "information accumulation."
- **Experimental Thoroughness**: ⭐⭐⭐ — Purely theoretical. The theorems are well-formed and quantitative scales are clear, but the lack of numerical validation for the tightness of constants is a minor drawback.
- **Writing Quality**: ⭐⭐⭐⭐ — Excellent background context. Provides detailed proof sketches for both theorems, highlighting the critical differences between TAS and S-TAS with a clear logical chain.
- **Value**: ⭐⭐⭐⭐ — Provides the first finite-confidence theoretical support for two widely used algorithms previously only guaranteed asymptotically, offering solid foundational value to the pure exploration community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Online Inventory Optimization in Non-Stationary Environment](online_inventory_optimization_in_non-stationary_environment.md)
- [\[ICLR 2026\] Stop Guessing: Choosing the Optimization-Consistent Uncertainty Measurement for Evidential Deep Learning](stop_guessing_choosing_the_optimization-consistent_uncertainty_measurement_for_e.md)
- [\[ICLR 2026\] Sharp Asymptotic Theory for Q-Learning with LD2Z Learning Rate and Its Generalization](sharp_asymptotic_theory_for_q-learning_with_textttld2z_learning_rate_and_its_gen.md)
- [\[ICLR 2026\] Towards a Theoretical Understanding of In-Context Learning: Stability and Non-i.i.d. Generalisation](towards_a_theoretical_understanding_of_in-context_learning_stability_and_non-iid.md)
- [\[ICLR 2026\] On Smoothness Bounds for Non-Clairvoyant Scheduling with Predictions](on_smoothness_bounds_for_non-clairvoyant_scheduling_with_predictions.md)

</div>

<!-- RELATED:END -->
