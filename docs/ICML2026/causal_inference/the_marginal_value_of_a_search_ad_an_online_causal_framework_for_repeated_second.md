---
title: >-
  [Paper Note] The (Marginal) Value of a Search Ad: An Online Causal Framework for Repeated Second-price Auctions
description: >-
  [ICML 2026][Causal Inference][second-price auction] This paper models the true value of search ads as a treatment effect between “win” and “lose” outcomes. Under binary feedback in repeated second-price auctions (SPA)…
tags:
  - "ICML 2026"
  - "Causal Inference"
  - "second-price auction"
  - "treatment effect"
  - "contextual bandit"
  - "IPW"
  - "UCB"
date: 2026-05-08
content_hash: c5d4102f7d3213b9
---

# The (Marginal) Value of a Search Ad: An Online Causal Framework for Repeated Second-price Auctions

**Conference**: ICML 2026  
**arXiv**: [2605.01756](https://arxiv.org/abs/2605.01756)  
**Code**: None  
**Area**: Causal Inference / Online Learning / Auctions and Advertising  
**Keywords**: second-price auction, treatment effect, contextual bandit, IPW, UCB

## TL;DR
This paper models the true value of search ads as a treatment effect between “win” and “lose” outcomes. Under binary feedback in repeated second-price auctions (SPA), it designs an online causal learning algorithm that exploits the payment rule, achieving the minimax-optimal regret $\widetilde\Theta(\sqrt{dT})$, which is strictly easier than learning in the corresponding first-price auction setting.

## Background & Motivation

**Background**: Search ads (sponsored positions in Amazon / Google / Bing) are almost always sold via second-price (Vickrey) auctions. In bidding, truthful bidding is theoretically optimal for SPA; however, advertisers care about how much an impression is worth in practice, which requires online estimation of CTR / CVR. A recent line of work models “ad value” as a treatment effect: the gain from clicks after winning, $v_{t,1}$, minus the gain from organic clicks that may still occur after losing, $v_{t,0}$. Wen et al. previously established a $\widetilde\Theta_d(T^{2/3})$ optimal regret under binary feedback for FPA.

**Limitations of Prior Work**: Existing auto-bidding systems equate value with the post-win payoff, which systematically overestimates bids. For brands already ranking highly in organic search, the marginal gain from winning is nearly zero, yet conventional algorithms still treat such cases as high-value opportunities. On the theory side, the difference between the optimal regret rates for FPA and SPA had not been characterized systematically.

**Key Challenge**: Causal estimation requires observing both “win” and “lose” outcomes, but a regret-minimizing bidder tends to win on high-value instances and lose on low-value ones, thereby harming propensity overlap. Meanwhile, SPA provides asymmetric feedback because only the winner observes the HOB, which helps learning but makes confidence design nontrivial.

**Goal**: (i) Extend the treatment-effect perspective to SPA; (ii) leverage the extra HOB information induced by the SPA payment rule to prove that the optimal regret under SPA is $\widetilde\Theta(\sqrt{dT})$ rather than the FPA rate $\widetilde\Theta_d(T^{2/3})$; (iii) relax the propensity-score estimation assumption to allow arbitrary error forms and HOB distributions with atoms.

**Key Insight**: The authors exploit the information gap in SPA — higher bids reveal more about the HOB CDF (winning reveals the exact HOB, while losing still reveals $\mathbb{1}[b\geq m_t]$). By decomposing the HOB CDF into interval probabilities $p^i$ and estimating them in blocks under this “one-sided + inferential” information structure, tighter confidence widths are obtained than by directly estimating $G(b)$.

**Core Idea**: The information advantage of second-price payments is converted into bid-dependent confidence widths for the propensity score. A “better of two UCBs” decision rule then neutralizes the potentially large variance of the IPW estimator, pushing SPA regret under binary feedback from $T^{2/3}$ down to $\sqrt{T}$.

## Method

### Overall Architecture
The algorithm interacts online for $T$ rounds. At each round, it observes context $x_t\in\mathbb{R}^d$ and chooses bid $b_t$ under the linear model $\mathbb{E}[\Delta v_t]=\theta_*^\top x_t$. The framework has three components: (1) **HOB Estimation Module**: blockwise estimation of the CDF $\widehat G_t$ using one-sided feedback from second-price payments; (2) **Value Estimation Module**: a corrected IPW weighted least-squares estimator based on the estimated propensity; (3) **Decision Module**: two equivalent reward reformulations $\widehat r_{t,0}, \widehat r_{t,1}$ with corresponding confidence widths $w_{t,0}, w_{t,1}$, where the tighter one is selected via a “better of two UCBs” rule. An outer master routine with $L=O(\log T)$ layers partitions time according to confidence widths and periodically performs uniform exploration to satisfy the theoretical conditions.

### Key Designs

1. **Blockwise HOB estimation + bid-dependent confidence widths**:

    - Function: Replace “estimate each $b$ separately” with “estimate interval probabilities $p^i=\mathbb{P}(b^{i-1}<m_t\leq b^i)$ in blocks and then aggregate,” so that lower bids enjoy more observations while higher bids have naturally smaller variance.
    - Core Mechanism: Discretize bids as $\mathcal{B}=\{b^j=(j-1)/\sqrt{T}\}$. Define $\widehat p_t^i=\sum_{\tau\in\Phi_t}\mathbb{1}[b_\tau\geq b^i]\mathbb{1}[b^i<m_\tau\leq b^{i+1}]/\sum_{\tau\in\Phi_t}\mathbb{1}[b_\tau\geq b^i]$, and estimate the CDF via $\widehat G_t(b^j)=\sum_{i\leq j}\widehat p_t^i$. Using second-price payments, a higher bid reveals $m_t$ after winning; a lower bid can still infer $\mathbb{1}[b\geq m_t]$ from whether a higher bid would have won. Hence each $p^i$ receives $n_t^i\propto\sum\mathbb{1}[b_\tau\geq b^i]$ effective observations, with more observations for lower bids. The confidence bound in Lemma 1 gives $u_t(b^j)\propto\sqrt{\sum_{k\leq j}\log T/n_t^k\cdot(\widehat p_0^k+\log T/\sqrt T)}$, which naturally realizes the Bernstein-style phenomenon that smaller $p^i$ are estimated more accurately.
    - Design Motivation: In FPA, binary feedback provides no payment information, making HOB estimation the $T^{2/3}$ bottleneck. In SPA, payment information speeds up HOB estimation, but this advantage can only be fully exploited via interval-wise block estimation.

2. **IPW with arbitrary error tolerance + weighted least squares**:

    - Function: Construct an IPW estimator $\widetilde e_t(b)$ that is robust to arbitrary error forms in $\widehat G_t$, and then solve for $\widehat\theta_t$ via variance-weighted least squares, avoiding reliance on Bernstein-type HOB error assumptions.
    - Core Mechanism: Define $\widetilde e_t(b)=\mathbb{1}[b\geq m_t]v_{t,1}/\widehat G_t(b)-\mathbb{1}[b<m_t]v_{t,0}/(1-\widehat G_t(b))$. The computable proxies for bias and variance are $u_t(b)\sigma_t(b)$ and $\sigma_t(b)^2$, where $\sigma_t(b)=1/(\widehat G_t(b)(1-\widehat G_t(b)))$. Then solve $\widehat\theta_t=\arg\min_\theta\sum_{\tau\in\Phi_t}\sigma_\tau^{-2}(\widetilde e_\tau-\theta^\top x_\tau)^2+\|\theta\|_2^2$, with a closed-form solution of the form $A_t^{-1}z_t$. Lemma 3 yields the error bound $|\widehat\theta_t^\top x_t-\theta_*^\top x_t|\leq\gamma\|x_t\|_{A_t^{-1}}$, and this bound may be arbitrarily large.
    - Design Motivation: Earlier FPA algorithms relied on a Bernstein-style assumption that propensity errors are tighter in low-probability regions. The blockwise HOB estimator here does not satisfy that form. The new IPW design accepts arbitrary error forms $u_t(b)$, decoupling the HOB module from the value module and substantially improving generality.

3. **“better of two UCBs” + uniform exploration fallback**:

    - Function: Convert the potentially large value-estimation error into a bounded regret loss by automatically selecting the tighter confidence width among two equivalent reward representations, and triggering uniform exploration when both widths are too large.
    - Core Mechanism: The reward $\bar r_t(b)$ can be rewritten as $\bar r_{t,0}(b)=G(b)(\theta_*^\top x_t-b)+\int_0^b G(m)\mathrm{d}m$ or $\bar r_{t,1}(b)=-(1-G(b))\theta_*^\top x_t-G(b)b+\int_0^b G(m)\mathrm{d}m$. These two forms depend on $\theta_*^\top x_t$ with complementary coefficients: when $\widehat G_t(b)$ is small, selecting $r_{t,0}$ yields a smaller width proportional to $\widehat G_t(b)$; otherwise $r_{t,1}$ is preferred. Algorithm 2 compares $\widehat G_t(b_L), \widehat G_t(b_R)$ against the threshold $1-\lambda/8$ and selects $q$ from the two UCBs, compressing the instantaneous regret to $\min\{w_{t,0}(b_t), w_{t,1}(b_t)\}\propto\sigma_t(b_t)^{-1}$. When the estimates are too poor and both widths are large, the algorithm performs forced exploration by uniformly sampling $b^1$ or $b^J$; this event has small probability and is bounded in Lemma 6 by $|\Phi_{\text{exp}}|=O(d\log^5 T)$.
    - Design Motivation: The core difficulty in causal estimation is the imbalance between winning and losing observations. Traditional forced-randomization methods pay a $T^{2/3}$ price. Here, the “dual UCB” design exploits the negative correlation between value-estimation error and $\sigma_t$; after taking the minimum, the width becomes smaller rather than larger, allowing regret control on most rounds and only limited exploration at a small confidence threshold.

### Loss & Training
The algorithm does not train a machine-learning model; it makes online decisions. The master routine is as follows: (i) use the first $(L+1)T_0$ rounds with $b_t=1$ to collect initial HOB observations; (ii) in each later round, check confidence widths across layers $\ell=1,\ldots,L$, assign the current round to the first layer whose width satisfies “$wt > 2^{-ℓ}$,” and compute the corresponding bid by UCB or forced exploration; (iii) use the layered scheme in Lemma 5 to maintain conditional independence among observations across layers.

## Key Experimental Results

### Main Results

| Setting | Feedback | Upper Bound | Lower Bound | Conclusion |
|------|------|------|------|------|
| SPA + binary feedback (Thm 1+2 of this paper) | binary | $O(\sqrt{dT}\log^3 T)$ | $\Omega(\sqrt{dT})$ | Strictly better than the FPA rate of $T^{2/3}$ |
| SPA + full-info feedback | full | $\widetilde O(\sqrt{dT})$ | $\Omega(\sqrt{dT})$ | Same order as binary |
| FPA + binary feedback (Wen et al. 2024) | binary | $\widetilde O_d(T^{2/3})$ | $\Omega_d(T^{2/3})$ | Baseline for comparison |
| Empirical comparison vs LinUCB | – | LinUCB linear regret | NFM-style algorithm converges at $\sqrt{T}$ | LinUCB overestimates value by ignoring $v_{t,0}$ |

### Ablation Study

| Configuration | Phenomenon | Interpretation |
|------|------|------|
| Algorithm 4 (practical variant) vs. main algorithm | One extra $\sqrt d$ factor | Removing the layered structure trades performance for simplicity, matching the common linear-bandit trade-off |
| HOB distributions with atoms (Definition 1) | Regret remains $\sqrt{dT}$ | The $(\omega,\lambda)$-locally-bounded assumption is generalized to point masses |
| Forced exploration only, without better-of-two UCBs | Regret degrades to $T^{2/3}$ | Confirms that better-of-two UCBs are essential for the $\sqrt T$ rate |
| Empirical LinUCB | Linear regret | Continues overbidding when value is identified with $v_{t,1}$ |

### Key Findings
- The SPA payment rule provides critical “winner-observes-HOB” information, turning causal learning under binary feedback from $T^{2/3}$ to $\sqrt T$. This is the essential difference between SPA and FPA for marginal-value bidding.
- Value estimation can be intrinsically unreliable: $\widehat\theta_t$ may have arbitrarily large error, yet the overall regret remains controlled as long as the decision module selects the tighter of the two UCBs. Such a mechanism for absorbing estimation error through decision structure is rare in online causal learning.
- Forced randomization alone is clearly suboptimal; only the combination of blockwise HOB estimation, better-of-two UCBs, and layered independence maintenance achieves the desired lower bound rate.

## Highlights & Insights
- Modeling ad value as a treatment effect rather than as “win payoff” has immediate industrial significance: for brands already performing well in organic search, traditional algorithms keep overbidding, whereas this method naturally avoids that waste.
- “Better of two UCBs” is a highly elegant technical trick: after rewriting the same reward function in two forms, the confidence widths with respect to $\widehat G_t$ become complementary; taking the minimum causes the instantaneous regret to collapse to $\sigma_t^{-1}$ even when the value estimate is poor.
- Relaxing propensity estimation error to an arbitrary $u_t$ allows SPA-style, non-Bernstein HOB estimation to be combined with causal learning modules. This generalization itself has methodological value and is suitable for many sponsored-auction variants.

## Limitations & Future Work
- The HOB distribution is assumed i.i.d. and stationary, whereas real ad markets drift as competitors change with trends. The authors mention a contextual HOB extension in Appendix B, but it is not developed in depth in the main text.
- The value model is limited to the linear form $\mathbb{E}[\Delta v_t]=\theta_*^\top x_t$, which is not directly applicable to deep neural scoring systems. Extending the method to kernels or neural representations remains open.
- Algorithm 3’s multilayer structure is somewhat complex in practice. Algorithm 4 simplifies it but incurs an additional $\sqrt d$ factor in regret, so deployment would still require a trade-off between performance and complexity.
- Experiments are conducted only on synthetic data, without validation on real search-ad platform data. The initial $O(\sqrt T\log T)$ rounds of high-bid exploration may also be unacceptable under tight top-level budgets.

## Related Work & Insights
- **vs Wen et al. (2024, FPA + treatment effect)**: This paper is the natural SPA extension. Its key contribution is to use the payment rule to reduce regret from $T^{2/3}$ to $\sqrt T$ and to relax the propensity estimation assumption.
- **vs Han et al. 2020 / interval splitting (FPA)**: The blockwise HOB estimation is inspired by interval splitting in FPA. This paper combines that idea with the asymmetric information induced by second-price payments to obtain bid-dependent confidence widths.
- **vs incrementality / lift bidding empirical literature**: That line mainly relies on industrial A/B tests with limited theoretical guarantees. This paper provides the first minimax-optimal algorithm that does not depend on an overlap condition.
- **vs linear contextual bandits** (LinUCB): The structure resembles LinUCB, but it additionally handles the heteroskedasticity and propensity uncertainty introduced by IPW, and uses better-of-two UCBs to control the variance explosion in causal estimation.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The paper is the first to couple SPA payment rules with treatment-effect causal learning, obtaining the $\sqrt{dT}$ optimal regret and giving a clean characterization of the FPA–SPA complexity gap.
- **Experimental Thoroughness**: ⭐⭐⭐ The theoretical analysis is complete, but the empirical section only compares against LinUCB on synthetic data and lacks real auction-data validation.
- **Writing Quality**: ⭐⭐⭐⭐ The mathematical structure is clear and the derivation of better-of-two UCBs is intuitive; however, the notation is dense and the entry barrier is nontrivial.
- **Value**: ⭐⭐⭐⭐ It has methodological significance for marginal-value bidding in search advertising and recommender systems, and it can guide practitioners in redesigning existing auto-bidders.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Efficient Ensemble Conditional Independence Test Framework for Causal Discovery](../../ICLR2026/causal_inference/efficient_ensemble_conditional_independence_test_framework_for_causal_discovery.md)
- [\[ACL 2025\] IRIS: An Iterative and Integrated Framework for Verifiable Causal Discovery](../../ACL2025/causal_inference/iris_an_iterative_and_integrated_framework.md)
- [\[NeurIPS 2025\] Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](../../NeurIPS2025/causal_inference/counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)
- [\[NeurIPS 2025\] GST-UNet: A Neural Framework for Spatiotemporal Causal Inference with Time-Varying Confounding](../../NeurIPS2025/causal_inference/gst-unet_a_neural_framework_for_spatiotemporal_causal_inference_with_time-varyin.md)
- [\[ACL 2025\] FitCF: A Framework for Automatic Feature Importance-guided Counterfactual Example Generation](../../ACL2025/causal_inference/fitcf_a_framework_for_automatic_feature_importance-guided_counterfactual_example.md)

</div>

<!-- RELATED:END -->
