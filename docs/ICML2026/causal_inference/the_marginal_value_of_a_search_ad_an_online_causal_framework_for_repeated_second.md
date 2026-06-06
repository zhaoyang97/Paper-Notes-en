---
title: >-
  [Paper Note] The (Marginal) Value of a Search Ad: An Online Causal Framework for Repeated Second-price Auctions
description: >-
  [ICML 2026][Causal Inference][Second-price Auction] This paper models the true value of search ads as the treatment effect of "winning vs. losing." Under binary feedback in repeated second-price auctions (SPA)…
tags:
  - "ICML 2026"
  - "Causal Inference"
  - "Second-price Auction"
  - "treatment effect"
  - "contextual bandit"
  - "IPW"
  - "UCB"
date: 2026-05-08
content_hash: 96d61a853337bf4c
---

# The (Marginal) Value of a Search Ad: An Online Causal Framework for Repeated Second-price Auctions

**Conference**: ICML 2026  
**arXiv**: [2605.01756](https://arxiv.org/abs/2605.01756)  
**Code**: None  
**Area**: Causal Inference / Online Learning / Auctions & Advertising  
**Keywords**: Second-price Auction, treatment effect, contextual bandit, IPW, UCB

## TL;DR
This paper models the true value of search ads as the treatment effect of "winning vs. losing." Under binary feedback in repeated second-price auctions (SPA), it designs an online causal learning algorithm that utilizes payment rules to achieve a minimax optimal regret of $\widetilde\Theta(\sqrt{dT})$, which is strictly easier to learn than first-price auctions (FPA) under the same setting.

## Background & Motivation

**Background**: Search advertising (sponsored slots on Amazon / Google / Bing) almost exclusively uses second-price (Vickrey) auctions. While "truthful bidding" is theoretically optimal for SPA, advertisers actually care about the value of an impression (CTR / CVR), which requires online estimation. A recent line of work models "ad value" as a treatment effect: the revenue from a winning click $v_{t,1}$ minus the revenue $v_{t,0}$ the user might still generate via organic search after a loss. Wen et al. previously established an optimal regret of $\widetilde\Theta_d(T^{2/3})$ for FPA under binary feedback.

**Limitations of Prior Work**: Existing auto-bidding equates value with "revenue after winning," systematically overestimating bids. For brands already ranking high in organic results, the marginal benefit of winning a sponsored slot is near zero, yet traditional algorithms still treat it as a high-value opportunity. Furthermore, the regret gap between FPA and SPA has not been systematically characterized.

**Key Challenge**: Causal estimation requires observing outcomes for both "win" and "lose" treatments. However, a regret-minimizing bidder tends to win high-value rounds and lose low-value ones, violating the propensity overlap condition. Additionally, while the asymmetric feedback in SPA (winners see the HOB) facilitates learning, designing confidence widths remains non-trivial.

**Goal**: (i) Extend the treatment-effect perspective to SPA; (ii) Leverage HOB information from SPA payment rules to prove an optimal regret of $\widetilde\Theta(\sqrt{dT})$ instead of $T^{2/3}$; (iii) Relax propensity score estimation assumptions to allow arbitrary error forms and HOB distributions with atoms.

**Key Insight**: The authors capture the core information asymmetry of SPA: higher bids yield more information about the HOB CDF (winners see the exact HOB; losers infer $\mathbb{1}[b\geq m_t]$). By utilizing this "one-sided + inference" information structure and decomposing the HOB CDF into interval probabilities $p^i$ for binned estimation, one can obtain tighter confidence widths than directly estimating $G(b)$.

**Core Idea**: Translate the "information dividend of second-price payments" into bid-dependent confidence widths for propensity scores, then design a "better of two UCBs" decision rule to neutralize the potential high variance of IPW estimates. This pushes the SPA regret under binary feedback from $T^{2/3}$ to $\sqrt{T}$.

## Method

### Overall Architecture
The algorithm interacts online for $T$ rounds. In each round, it receives context $x_t\in\mathbb{R}^d$ and decides the bid $b_t$ based on a linear model $\mathbb{E}[\Delta v_t]=\theta_*^\top x_t$. The framework consists of three components: (1) **HOB Estimation Module**: Uses one-sided feedback from second-price payments to bin-estimate the CDF $\widehat G_t$; (2) **Value Estimation Module**: Solves for $\widehat\theta_t$ using modified IPW weighted least squares based on estimated propensities; (3) **Decision Module**: Constructs two equivalent reward rewritings $\widehat r_{t,0}, \widehat r_{t,1}$ with corresponding confidence widths $w_{t,0}, w_{t,1}$, and uses the "better of two UCBs" to select the tighter width for UCB bidding. The outer loop uses a master routine with $L=O(\log T)$ layers to categorize time steps by confidence width and periodically performs uniform exploration.

### Key Designs

1.  **Binned HOB Estimation + Bid-dependent Confidence Widths**:
    -   **Function**: Shifts CDF estimation from "individual estimation for each $b$" to "binned estimation of interval probabilities $p^i=\mathbb{P}(b^{i-1}<m_t\leq b^i)$," allowing lower bids to benefit from more observations and higher bids to have naturally lower variance.
    -   **Mechanism**: Bids are discretized into $\mathcal{B}=\{b^j=(j-1)/\sqrt{T}\}$. Define $\widehat p_t^i=\sum_{\tau\in\Phi_t}\mathbb{1}[b_\tau\geq b^i]\mathbb{1}[b^i<m_\tau\leq b^{i+1}]/\sum_{\tau\in\Phi_t}\mathbb{1}[b_\tau\geq b^i]$ and the CDF estimate $\widehat G_t(b^j)=\sum_{i\leq j}\widehat p_t^i$. Utilizing second-price payments: high bids reveal $m_t$ upon winning, while for low bids, any winner with a higher bid can provide the indicator $\mathbb{1}[b\geq m_t]$. Thus, each $p^i$ has $n_t^i\propto\sum\mathbb{1}[b_\tau\geq b^i]$ valid observations. The derived width (Lemma 1) $u_t(b^j)\propto\sqrt{\sum_{k\leq j}\log T/n_t^k\cdot(\widehat p_0^k+\log T/\sqrt T)}$ naturally achieves Bernstein-style concentration where smaller $p^i$ leads to more accurate estimates.
    -   **Design Motivation**: In FPA, binary feedback lacks payment information, making HOB estimation a $T^{2/3}$ bottleneck. In SPA, payment information accelerates HOB estimation, but only binning can fully release this dividend.

2.  **Error-Tolerant IPW + Weighted Least Squares**:
    -   **Function**: Constructs an IPW estimator $\widetilde e_t(b)$ robust to arbitrary error forms in $\widehat G_t$, then uses variance-weighted least squares to solve for $\widehat\theta_t$, avoiding dependence on Bernstein-type HOB error assumptions.
    -   **Mechanism**: Define $\widetilde e_t(b)=\mathbb{1}[b\geq m_t]v_{t,1}/\widehat G_t(b)-\mathbb{1}[b<m_t]v_{t,0}/(1-\widehat G_t(b))$. The bias and variance proxies are $u_t(b)\sigma_t(b)$ and $\sigma_t(b)^2$, where $\sigma_t(b)=1/(\widehat G_t(b)(1-\widehat G_t(b)))$. Solving $\widehat\theta_t=\arg\min_\theta\sum_{\tau\in\Phi_t}\sigma_\tau^{-2}(\widetilde e_\tau-\theta^\top x_\tau)^2+\|\theta\|_2^2$ yields a closed-form solution $A_t^{-1}z_t$. Lemma 3 provides the error bound $|\widehat\theta_t^\top x_t-\theta_*^\top x_t|\leq\gamma\|x_t\|_{A_t^{-1}}$.
    -   **Design Motivation**: Previous FPA algorithms relied on Bernstein assumptions (propensity errors are tighter in low-probability regions), which the binned HOB estimation does not satisfy. The new IPW design accepts arbitrary error forms $u_t(b)$, decoupling the HOB and value modules and improving generalizability.

3.  **"Better of Two UCBs" + Forced Exploration**:
    -   **Function**: Converts potential large errors in value estimation into bounded regret: automatically selects the tighter confidence width among two equivalent reward rewritings and triggers uniform exploration if both widths are too large.
    -   **Mechanism**: Reward $\bar r_t(b)$ can be rewritten as $\bar r_{t,0}(b)=G(b)(\theta_*^\top x_t-b)+\int_0^b G(m)\mathrm{d}m$ or $\bar r_{t,1}(b)=-(1-G(b))\theta_*^\top x_t-G(b)b+\int_0^b G(m)\mathrm{d}m$. Their dependence on $\theta_*^\top x_t$ is complementary: $r_{t,0}$ is tighter when $\widehat G_t(b)$ is small, while $r_{t,1}$ is tighter otherwise. Algorithm 2 selects between the two UCBs based on a threshold $1-\lambda/8$, constraining instantaneous regret to $\min\{w_{t,0}(b_t), w_{t,1}(b_t)\}\propto\sigma_t(b_t)^{-1}$. If estimates are poor (low probability per Lemma 6), forced exploration occurs.
    -   **Design Motivation**: Causal estimation struggles with win-loss imbalance. While traditional methods use forced randomization (costing $T^{2/3}$), this algorithm uses "dual UCBs" to exploit the negative correlation between value estimation error and $\sigma_t$, controlling regret in most rounds.

### Loss & Training
The algorithm performs online decision-making rather than model training. The master routine: (i) Uses initial $(L+1)T_0$ rounds with $b_t=1$ for HOB observations; (ii) Assigns current rounds to layers based on "wt > 2^{-ℓ}"; (iii) Uses the layered scheme from Lemma 5 to maintain conditional independence of observations across layers.

## Key Experimental Results

### Main Results

| Setting | Feedback | Upper Bound | Lower Bound | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| SPA + binary feedback (Ours Thm 1+2) | binary | $O(\sqrt{dT}\log^3 T)$ | $\Omega(\sqrt{dT})$ | Strictly better than $T^{2/3}$ in FPA |
| SPA + full-info feedback | full | $\widetilde O(\sqrt{dT})$ | $\Omega(\sqrt{dT})$ | Same order as binary |
| FPA + binary feedback (Wen et al. 2024) | binary | $\widetilde O_d(T^{2/3})$ | $\Omega_d(T^{2/3})$ | Comparison baseline |
| Empirical vs LinUCB | – | LinUCB linear regret | NFM-style $\sqrt{T}$ convergence | LinUCB overestimates by ignoring $v_{t,0}$ |

### Ablation Study

| Configuration | Phenomenon | Explanation |
| :--- | :--- | :--- |
| Algorithm 4 (Practical variant) vs Main | Extra $\sqrt d$ factor | Simplicity over layered structure trade-off |
| HOB Distribution with Atoms (Definition 1) | Regret stays $\sqrt{dT}$ | Generalization to point masses |
| Forced exploration only (no dual UCBs) | Regret degrades to $T^{2/3}$ | Validates "Better of Two UCBs" as key to $\sqrt T$ |
| Empirical LinUCB | Linear regret | Persistent overbidding when equating value to $v_{t,1}$ |

### Key Findings
- SPA payment rules provide critical "winner's HOB observation" information, pulling causal learning under binary feedback from $T^{2/3}$ to $\sqrt T$. This is the fundamental difference between SPA and FPA in marginal value bidding.
- Value estimation can be "inherently unreliable": the error in $\widehat\theta_t$ can be large, but if the decision module selects the tighter UCB, overall regret remains controlled.
- Forced randomization alone is suboptimal. Only the combination of binned HOB estimation, "better of two UCBs," and layered independence maintenance can match the lower bound.

## Highlights & Insights
- Modeling ad value as treatment effect rather than "winning revenue" has immediate industrial significance: it prevents overbidding for brands with high organic performance.
- "Better of two UCBs" is a clever technical trick: rewriting the reward function yields complementary widths relative to $\widehat G_t$, allowing the instantaneous regret to collapse to $\sigma_t^{-1}$ even with poor value estimates.
- Relaxing propensity error assumptions to arbitrary $u_t$ allows "non-Bernstein" HOB estimation in SPA to be coupled with causal learning, a methodological contribution suitable for any sponsored auction variant.

## Limitations & Future Work
- Assumes HOB distribution is i.i.d. stationary; in reality, competitors drift with trends. The authors mention contextual HOB in Appendix B but do not delve deep.
- The value model is limited to linear $\mathbb{E}[\Delta v_t]=\theta_*^\top x_t$; extension to kernel or neural representations for deep learning scenarios is an open direction.
- The multi-layer structure of Algorithm 3 is complex for engineering; the simpler Algorithm 4 incurs an extra $\sqrt d$ factor, necessitating trade-offs.
- Evaluation relies on synthetic data; the initial $O(\sqrt T\log T)$ rounds of high-bid exploration might be unacceptable in budget-constrained scenarios.

## Related Work & Insights
- **vs Wen et al. (2024, FPA + treatment effect)**: This work is a natural extension to SPA, with the key contribution of pushing regret from $T^{2/3}$ to $\sqrt T$ and relaxing propensity assumptions.
- **vs Han et al. 2020 / interval splitting (FPA)**: Binned HOB estimation is inspired by FPA interval splitting but combined with second-price asymmetric information to derive bid-dependent widths.
- **vs Incrementality / lift bidding empirical literature**: That line focuses on internal A/B tests with few theoretical guarantees; this paper provides the first minimax optimal algorithm not requiring "overlap."
- **vs Linear contextual bandits (LinUCB)**: Structurally similar to LinUCB but addresses heteroskedasticity from IPW and propensity uncertainty via "better of two UCBs."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to couple SPA payment rules with treatment-effect causal learning to achieve $\sqrt{dT}$ regret.
- Experimental Thoroughness: ⭐⭐⭐ Extremely complete theoretical analysis, but empirical tests are limited to synthetic data vs LinUCB.
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical structure and intuitive derivation of dual UCBs; high symbol density.
- Value: ⭐⭐⭐⭐ Methodological significance for "marginal value" bidding in search and recommendation systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Efficient Ensemble Conditional Independence Test Framework for Causal Discovery](../../ICLR2026/causal_inference/efficient_ensemble_conditional_independence_test_framework_for_causal_discovery.md)
- [\[NeurIPS 2025\] GST-UNet: A Neural Framework for Spatiotemporal Causal Inference with Time-Varying Confounding](../../NeurIPS2025/causal_inference/gst-unet_a_neural_framework_for_spatiotemporal_causal_inference_with_time-varyin.md)
- [\[ICML 2026\] Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference](tailoring_strictly_proper_scoring_rules_for_downstream_tasks_an_application_to_c.md)
- [\[NeurIPS 2025\] Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](../../NeurIPS2025/causal_inference/counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)
- [\[ICML 2026\] Evaluating Bivariate Causal Statements Based on Mutual Compatibility](evaluating_bivariate_causal_statements_based_on_mutual_compatibility.md)

</div>

<!-- RELATED:END -->
