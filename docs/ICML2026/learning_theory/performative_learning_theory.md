---
title: >-
  [Paper Note] Performative Learning Theory
description: >-
  [ICML 2026][Learning Theory][Wasserstein distance] This paper embeds the "performative prediction" phenomenon—where predictions change the very outcomes they intend to forecast—into statistical learning theory for the first time. It proves upper bounds for generalization error, generalization gap, and excess risk under three scenarios: sample-only, population-only, and joint performative perturbations. The work reveals a fundamental tradeoff between "changing the world" and "…
tags:
  - "ICML 2026"
  - "Learning Theory"
  - "Generalization Bounds"
  - "Performative Prediction"
  - "Wasserstein distance"
  - "self-fulfilling/self-negating prediction"
  - "distributionally robust optimization"
date: 2026-05-08
content_hash: 0b42b261016f6701
---

# Performative Learning Theory

**Conference**: ICML 2026  
**arXiv**: [2602.04402](https://arxiv.org/abs/2602.04402)  
**Code**: https://github.com/rodemann/plt-jobseekers (Case study reproduction)  
**Area**: Learning Theory / Generalization Bounds / Performative Prediction  
**Keywords**: performative prediction, generalization bounds, Wasserstein distance, self-fulfilling/self-negating prediction, distributionally robust optimization

## TL;DR
This paper embeds the "performative prediction" phenomenon—where predictions change the very outcomes they intend to forecast—into statistical learning theory for the first time. It proves upper bounds for generalization error, generalization gap, and excess risk under three scenarios: sample-only, population-only, and joint performative perturbations. The work reveals a fundamental tradeoff between "changing the world" and "learning from the world," as well as an "empirical echo chamber" formed by self-negating populations and self-fulfilling samples in the worst case.

## Background & Motivation

**Background**: Machine learning systems have evolved from "analyzing the world" to "shaping the world." For instance, a navigation app predicting congestion may cause drivers to reroute, making the congestion disappear. Similarly, if employment centers allocate training slots based on "long-term unemployment risk," those predicted to be at high risk might find jobs faster due to the intervention. Perdomo et al. (2020) formalized this feedback loop as **performative prediction (PP)**. However, existing PP literature (Perdomo, Brown, Miller, etc.) almost exclusively defines "actual outcomes" at the **population level**, focusing on stability and optimality under repeated risk minimization while **evading the question of whether one can generalize from a finite sample to the population.**

**Limitations of Prior Work**: In reality, practitioners typically have access only to a **finite sample** of the population (e.g., a canary release in San Francisco or a pilot study in Bavaria). Performative effects may occur within the sample, the out-of-sample population, or both. Classical learning theory assumes fixed and independent training/test distributions. Once predictions react back upon the data distribution, standard conclusions regarding generalization from training to test sets no longer apply—a gap that has not been systematically addressed.

**Key Challenge**: In a performative world, the **learning target itself drifts with the prediction**. The more a model is used to intervene in the data (e.g., helping more people find jobs or rerouting more drivers), the more the sample deviates from the original population, paradoxically making it harder to reliably infer population properties. This creates an inherent tension between "intervention vs. inference."

**Goal**: To embed PP into statistical learning theory, clearly define what "generalization" means in a performative context, and provide computable generalization guarantees under three perturbation scenarios (sample-only, population-only, or both).

**Key Insight**: Without assuming any specific functional form for the transition map $\mathrm{Tr}$ (requiring only Wasserstein sensitivity), the authors incorporate performative drift into generalization bounds using covering numbers and Wasserstein distances. The worst-case scenarios are characterized as **min-max (population self-negation)** and **min-min (sample self-fulfillment)** risk functionals in Wasserstein space.

## Method

### Overall Architecture

The paper first conceptually decomposes "performative generalization" into four scenarios and proposes four research questions (RQ1–RQ4, see Table 1), corresponding to combinations of "retraining on samples / retraining on population / retraining on both." Technically, it uses a **general assumption** (Wasserstein sensitivity of the transition map $\mathrm{Tr}$) to embed performative drift into learning theory. The authors sequentially prove bounds for excess risk, performative excess risk, generalization gap, and cumulative performative excess risk. Finally, two structural insights—the "change vs. learn" tradeoff and the "empirical echo chamber"—are revealed, leading to a counter-intuitive but practical corollary: retraining on perturbed samples can actually tighten the bounds. The core proof technique involves empirical process theory and dual characterizations of inf-sup / inf-inf risk functionals in Wasserstein space (corresponding to Distributionally Robust Optimization, DRO, and Distributionally Favorable Optimization, DFO).

### Key Designs

**1. Embedding performative prediction into learning theory: Four generalization scenarios and Repeated Empirical Risk Minimization**

To address the gap where prior PP work ignores finite-sample generalization, the authors distinguish between **sample performativity** (the model only changes the sample/sub-population used for training, e.g., navigation visible only to canary users), **population performativity** (classical PP setting), and **full performativity** (both react). Accordingly, the classical PP concept of Repeated Risk Minimization (RRM, $\theta_{t+1}=G(d_t)$) is extended to **Repeated Empirical Risk Minimization** (RERM, $\widehat\theta_{t+1}=G(\widehat d_t)$, where $\widehat d_t=\mathrm{Tr}(\widehat d_{t-1},\widehat\theta_t)$). Table 1 uses a 2D matrix of "retraining target × performative effect location" to unify ERM, online learning, classical PP, and the four new RQs. The work adopts the **stateful** extension $d_t=\mathrm{Tr}(d_{t-1},\theta_t)$ from Brown et al. (2022), making the stateless case $d_t=\mathrm{Tr}_s(\theta_t)$ a special instance.

**2. Characterizing unknown drift with Wasserstein sensitivity to provide generalization bounds**

To provide bounds **without assuming the specific form of $\mathrm{Tr}$**, the authors require only a few conditions from the Perdomo/Brown framework: $\gamma$-strongly convex loss (Cond. 3.1), **jointly Wasserstein sensitive** transition mapping (Cond. 3.2, $W_p(\mathrm{Tr}(d,\theta),\mathrm{Tr}(d',\theta'))\le\varepsilon W_p(d,d')+\varepsilon\|\theta-\theta'\|_2$), and a loss that is Lipschitz with respect to $z$ and continuously differentiable with respect to $\theta$ (Cond. 3.3). The proof strategy bounds three segments of Wasserstein distance: $W_p(\widehat d_0,d_0)$ (Lemma 3.4, sample-population convergence), $W_p(\widehat d_0,\widehat d_T)$ (Lemma 3.5, in-sample performative drift), and $W_p(d_0,d_T)$ (Lemma 3.9, population performative drift), and converts them into expected differences via the Kantorovich–Rubinstein duality. Since drift might push evaluation points outside the support of $d_0$, the authors measure hypothesis class richness using the **covering number entropy integral** $\mathfrak C$ (rather than Rademacher complexity). This yields excess risk bounds under sample performativity (Theorem 3.7) and performative excess risk bounds under full performativity (Theorem 3.10). A key observable is the **performative response rate** $m/n$ (the proportion of units $m$ in the sample size $n$ that changed due to prediction); the bound grows with $m$.

**3. Self-negating and self-fulfilling: min-max / min-min in Wasserstein space and empirical echo chambers**

To provide **tighter** generalization gap bounds under slightly stronger regularity (Theorems 3.13/3.15), the authors reveal two directions in which generalization fails in a performative world: the population might **self-negate** the prediction—the worst case being $\sup_{d}\mathscr R(d,\widehat\theta_T)$, which corresponds to the inf-sup functional of **Distributionally Robust Optimization (DRO)** over a Wasserstein ball $\mathcal A=\{d:W_p(d_0,d)\le b\}$. Conversely, the sample might be **self-fulfilling**—RERM is equivalent to solving $\arg\inf_\theta\inf_{d\in\mathcal A'}\mathscr R(d,\theta)$ on the sample side, corresponding to **Distributionally Favorable Optimization (DFO)** via an inf-inf functional. The superposition creates an **empirical echo chamber**: the sample deceptively confirms the prediction while the population does the opposite, pulling the model toward "good on sample, bad on population." In the navigation example, San Francisco drivers (sample) might follow the app perfectly while Bay Area drivers (population) do the reverse.

**4. Change-vs-learn tradeoff and counter-intuitive inference: Retraining on perturbed samples tightens bounds**

The bounds reveal two insights. First, the **change-vs-learn tradeoff**: the performative term is dominated by $\varepsilon(1+L_a)$—growing exponentially with retraining steps $T$ when $\varepsilon(1+L_a)>1$ and linearly when $=1$; the bound also grows with $m$. Intuitively, if an employment center helps more people (increasing $m$), the cost is that the model becomes harder to generalize to unseen new clients. Second, a **counter-intuitive corollary** (Corollary 3.11): while naively retraining causes $\widehat\theta_t$ to deteriorate (the bound grows with $T$), these perturbed samples $\widehat d_1,\dots,\widehat d_T$ actually help **estimate $\mathrm{Tr}$** more efficiently, thereby tightening the bound in Theorem 3.10 (Lemma 3.9 conservatively uses the $m$ from the round with the most drastic reaction, whereas one actually observes $m_t$ each round). Practical conclusion: if retraining under performativity is necessary, **use the initial fit $\widehat\theta_0$ for out-of-sample prediction and estimate the induced population drift using observed sample drifts across rounds** to obtain the tightest guarantee.

## Key Experimental Results

As this is a theoretical work, "experiments" serve to **illustrate** the behavior of the bounds rather than as applications. The table below summarizes the four research questions and main results:

| Research Question | Scenario | Target Bound | Main Theorem |
|------|------|------|------|
| RQ1 | Sample-only performativity (retraining on sample only) | Classical excess risk | Theorem 3.7, Cor. 3.8 |
| RQ2 | Full performativity (both sample and population react) | Performative excess risk / Generalization gap | Theorems 3.10, 3.13, 3.15 |
| RQ3 | Sample retraining followed by population retraining | Cumulative performative excess risk | Theorem 3.16 |
| RQ4 | Inferential difference between RERM and hypothetical RRM | Inferential gap (statistical properties) | Partially answered by Li et al. (2025) CLT |

### Case Study: German Employment Agency Jobseeker Data

Data originates from German Federal Employment Agency administrative records (1975–2017, >60 million rows, 2% sample). The task is binary prediction of "whether a jobseeker will remain long-term unemployed," using L2-regularized logistic regression (satisfying strong convexity, Lipschitz, and differentiability) to simulate the performative effect of "allocating training based on risk."

| Setting | Sample Size n | Observed Drift m / Response Rate | Bound (95% Confidence) |
|------|------|------|------|
| Sample performativity (RQ1, Cor. 3.8) | 60,147 | m=1,816, $m/n\approx0.030$ | Generalization gap $\le 0.01+0.29\approx0.30$ nats |
| Full performativity (RQ2, Thm. 3.13, semi-simulated) | 41,585 | $m=\xi n,\ \xi\in\{0.01,\dots,0.5\}$ | Monotonic growth with allocation ratio $\xi$ (Figure 1.3) |

### Key Findings
- **Response rate directly controls the bound**: Under sample performativity, the bound is $\approx0.30$ nats. The sampling term is negligible ($\approx0.01$) due to large $n$, while the dominant term ($\approx0.29$) is entirely determined by the observable response rate $m/n=1816/60147$—turning an abstract bound into an actionable guarantee for the institution.
- **More interference leads to harder learning**: In semi-simulations, as the proportion $\xi$ of high-risk individuals assigned to training increases (helping more people/changing more units), the performative generalization gap grows, confirming the change-vs-learn tradeoff.
- **Decomposability of the bound**: The total bound is split into adaptive complexity, performative terms, and sampling terms, allowing practitioners to discern whether error stems from finite sampling or excessive intervention.

## Highlights & Insights
- **Translating phenomena into computable bounds**: While previous PP work focused on stability or optimality, this paper provides the first finite-sample generalization bounds from sample to population. The key variable $m/n$ is observable by institutions, making it highly practical.
- **Duality of min-max ↔ DRO and min-min ↔ DFO**: Mapping "population self-negation" and "sample self-fulfillment" to mature tools in distributionally robust/favorable optimization is an elegant bridge that allows future work to leverage progress in those fields.
- **Counter-intuitive "value of retraining"**: While retraining under performativity is often seen as detrimental, the authors show that perturbed samples provide information to estimate the unknown drift $\mathrm{Tr}$, thereby tightening bounds—a perspective applicable to any online/continual learning scenario with distribution drift.

## Limitations & Future Work
- **Strong convexity requirement**: Condition 3.1 (strong convexity) and Condition 3.12 (Lipschitz functions in $\mathcal F$) are somewhat restrictive; however, the authors emphasize these are verifiable assumptions about the model/loss we control, traded for zero assumptions on the unknown $\mathrm{Tr}$ we do not control.
- **Potential looseness**: To remain agnostic of the functional form of $\mathrm{Tr}$, the bounds may be looser than those assuming a specific drift form.
- **RQ4 only asymptotically addressed**: The inferential gap between RERM and RRM currently relies on $n\to\infty$ Central Limit Theorem results; finite-sample and stateful cases remain open.
- **Illustrative case study**: Real-world application requires historical prediction records (often unavailable); Section 4 is clearly intended as an illustration rather than a full deployment.

## Related Work & Insights
- **vs. Perdomo et al. (2020) / Brown et al. (2022)**: These study population-level stability and optimality assuming access to the full distribution; Ours introduces the finite-sample perspective and incorporates ERM into the performative framework via RERM.
- **vs. Kirev et al. (2025)**: The latter provides partial answers for RQ1(a) specifically for binary classification and linear drift; Ours is more general, covering all Lipschitz continuous transition maps and supporting drift across any subset of $X$ and $Y$.
- **vs. Distributionally Robust Performative Optimization (Jia et al. 2025, Xue & Sun 2024)**: These use Wasserstein ambiguity sets to robustify predictions a priori; This work studies **generalization**, and the ambiguity sets are **estimated** from the performative reactions of the samples rather than being pre-specified.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to embed performative prediction into statistical learning theory with finite-sample bounds.
- Experimental Thoroughness: ⭐⭐⭐⭐ Primarily a theoretical paper; real-world data is used effectively to illustrate bound behavior, though the authors acknowledge it is illustrative.
- Writing Quality: ⭐⭐⭐⭐ The conceptual framework (Table 1) and two running examples make the abstract bounds very clear.
- Value: ⭐⭐⭐⭐⭐ Provides the first set of generalization analysis tools for systems where predictions change data, with real-world relevance for high-stakes deployments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] AI4SLT: Empirical Processes in Lean 4 for Formal Statistical Learning Theory](ai4slt_empirical_processes_in_lean_4_for_formal_statistical_learning_theory.md)
- [\[ICLR 2026\] Learning Admissible Heuristics for A*: Theory and Practice](../../ICLR2026/learning_theory/learning_admissible_heuristics_for_a_theory_and_practice.md)
- [\[ICLR 2026\] Sharp Asymptotic Theory for Q-Learning with LD2Z Learning Rate and Its Generalization](../../ICLR2026/learning_theory/sharp_asymptotic_theory_for_q-learning_with_textttld2z_learning_rate_and_its_gen.md)
- [\[ICLR 2026\] Almost Bayesian: Dynamics of SGD Through Singular Learning Theory](../../ICLR2026/learning_theory/almost_bayesian_dynamics_of_sgd_through_singular_learning_theory.md)
- [\[ICML 2026\] Catastrophic Forgetting is Low-Rank: A Function-Space Theory for Continual Adaptation](catastrophic_forgetting_is_low-rank_a_function-space_theory_for_continual_adapta.md)

</div>

<!-- RELATED:END -->
