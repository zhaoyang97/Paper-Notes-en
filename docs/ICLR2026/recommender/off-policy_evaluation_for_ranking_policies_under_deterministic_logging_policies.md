---
title: >-
  [Paper Note] Off-Policy Evaluation for Ranking Policies under Deterministic Logging Policies
description: >-
  [ICLR 2026][Recommender Systems][Off-Policy Evaluation] Addressing the issue where "completely deterministic" logging policies in industrial ranking systems cause severe bias in traditional IPS-based estimators, this paper proposes the CIPS estimator (and its doubly robust extension CDR). By replacing the "policy probability ratio" with the "user click probability ratio" as
tags:
  - ICLR 2026
  - Recommender Systems
  - Off-Policy Evaluation
date: 2026-05-08
content_hash: 4b6d5fdd2549b813
---
# Off-Policy Evaluation for Ranking Policies under Deterministic Logging Policies

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=0ZkWWxcHKV](https://openreview.net/forum?id=0ZkWWxcHKV)  
**Code**: Provided as supplementary material  
**Area**: Recommender Systems / Off-Policy Evaluation (OPE) / Ranking  
**Keywords**: Off-Policy Evaluation, Ranking Policies, Deterministic Logging Policies, Click Importance Weights, Doubly Robust

## TL;DR
Addressing the issue where "completely deterministic" logging policies in industrial ranking systems cause severe bias in traditional IPS-based estimators, this paper proposes the CIPS estimator (and its doubly robust extension CDR). By replacing the "policy probability ratio" with the "user click probability ratio" as the importance weight, it relaxes the support condition required for unbiasedness from "stochastic logging policies" to "intrinsic randomness in click behavior," achieving low-bias or even unbiased evaluation under deterministic logging.

## Background & Motivation
**Background**: Systems like recommendation, search, and e-commerce are essentially "contextual bandit" processes—a ranking policy $\pi$ repeatedly observes context $x$, produces a ranking $A=(a_1,\dots,a_K)$, and observes rewards. Off-Policy Evaluation (OPE) aims to estimate the expected value $V(\pi)$ of a new policy $\pi$ using only data $D=\{(x_i,A_i,C_i,C_iR_i)\}$ collected by an old logging policy $\pi_0$, thereby avoiding risky A/B testing. Mainstream estimators in ranking include ranking-wise IPS, position-wise IIPS, and RIPS under the cascade model—all of which use importance weighting based on the "probability ratio of selecting the same ranking/action at the same position" between new and old policies.

**Limitations of Prior Work**: All these IPS-based estimators require the logging policy $\pi_0$ to be **sufficiently stochastic** to be unbiased. Ranking-wise IPS requires $\pi_0$ to have explored all rankings that the new policy might select (ranking-wise common support); IIPS requires $\pi_0$ to have selected all actions at every position (position-wise common support); RIPS requires prefix support under the cascade assumption. however, due to large action spaces, large-scale industrial systems often deploy **completely deterministic policies**—outputting a single unique ranking for a given $x$—as random policies are inefficient and risky. In such cases, support conditions are severely violated, leading to massive bias.

**Key Challenge**: Existing methods rely **solely on the logging policy** as the source of randomness for "importance weights." Once $\pi_0$ is deterministic, $\pi_0(A|x)\in\{0,1\}$, and the probability of most rankings the new policy wishes to evaluate is 0 in the logs, making weights undefined. Theorem 2.1 quantifies this bias: the bias of IPS equals $-\mathbb{E}_{p(x)}\big[\sum_{A\in U_0(x,\pi_0)}\pi(A|x)\sum_k q_k(x,A)\big]$, where $U_0$ is the set of unsupported rankings. Under deterministic logging, $U_0$ contains almost all rankings, resulting in huge bias (in a toy example, 5 out of 6 rankings were unsupported). This differs from the "high variance due to large action spaces" typically discussed in OPE literature: **under deterministic logging, bias, not variance, is the primary contradiction**.

**Goal**: Provide a ranking OPE estimator with controllable bias and theoretical unbiasedness under completely deterministic logging policies.

**Key Insight**: The authors observe a neglected source of randomness—**even if the logging policy deterministically provides a ranking, whether a user clicks on each action remains stochastic**. Each action in the ranking is viewed/clicked by the user with some non-zero click probability; this "intrinsic randomness in user click behavior" is independent of whether the logging policy is stochastic.

**Core Idea**: Use the "click probability ratio" instead of the "policy probability ratio" for importance weighting. Since $\pi_0$ no longer provides randomness, the click probability $p_c(x,a,\pi)$ is used as the new carrier for importance weights, shifting the support condition from "policy support" to "click support," which is easier to satisfy.

## Method

### Overall Architecture
The paper addresses how to unbiasedly estimate $V(\pi)=\mathbb{E}_{p(x)\pi(A|x)}\big[\sum_{k=1}^K q_k(x,A)\big]$ when $\pi_0$ is completely deterministic and traditional importance weights fail. The approach involves one key replacement, but the steps are interconnected:

First, **rewrite the value definition from "summation by position" to "summation by action"**. The original $V(\pi)=\mathbb{E}[\sum_k C(k)R(k)]$ accumulates by position $k$. The authors rewrite this equivalently as $V(\pi)=\mathbb{E}_{p(x)\pi(A|x)p(C,R|x,A)}\big[\sum_{a\in A}C(a)R(a)\big]$, where $C(a)$ and $R(a)$ are the click indicator and potential reward for action $a$. This is a change of variables for the same quantity (introducing no independence assumptions), aimed at making "clicks" the focus of the analysis.

Second, **define a new importance weight using marginal click probabilities**. Define $p_c(x,a,\pi)=\mathbb{E}_{\pi(A|x)}[\,\mathbb{E}[C(a)|x,A]\,]$ as the marginal probability of a user clicking action $a$ under context $x$ and policy $\pi$. Replacing $\pi(A|x)/\pi_0(A|x)$ with $p_c(x,a,\pi)/p_c(x,a,\pi_0)$ yields the CIPS estimator.

Third, **theoretically establish new unbiasedness conditions** (click-wise common support + potential reward independence) and extend the estimator to a doubly robust version, CDR, to reduce variance without introducing additional bias.

It is necessary to clarify the "two-stage reward" structure modeled: the user first generates an **exposure/click** (click vector $C$), and only after clicking is a **downstream potential reward** $R$ (e.g., purchase, watch time) generated. We can only observe $C$ and $CR$—meaning $R(k)$ is only visible if $C(k)=1$. $R$ can never be fully observed in isolation. This is the physical basis for CIPS: the click layer is naturally noisy and stochastic.

### Key Designs

**1. CIPS: Using Click Probability Ratio as the New Importance Weight**

The root of bias in traditional IPS is placing all randomness on the deterministic $\pi_0$. CIPS changes the weight carrier—instead of asking "with what probability did $\pi_0$ select this ranking," it asks "in the ranking produced by $\pi_0$, with what probability does the user click action $a$." The estimator is written as:

$$\hat{V}_{\text{CIPS}}(\pi;D)=\frac{1}{n}\sum_{i=1}^n\sum_{a\in A}\frac{p_c(x_i,a,\pi)}{p_c(x_i,a,\pi_0)}\,C_i(a)R_i(a),$$

where $p_c(x,a,\pi)=\mathbb{E}_{\pi(A|x)}[\mathbb{E}[C(a)|x,A]]$ is the marginal click probability. Intuitively, even if $\pi_0$ deterministically outputs $A_1$, the probabilities that a user clicks $a_1, a_2, a_3$ within $A_1$ (e.g., $0.8, 0.5, 0.2$) are non-zero. Since the new policy $\pi$ is a mixture of rankings, its marginal click probability $p_c(x,a,\pi)$ is also non-zero. Thus, the weight $p_c(x,a,\pi)/p_c(x,a,\pi_0)$ is defined almost everywhere (in the toy example, weights for $a_1, a_2, a_3$ were $0.6875, 0.78, 2.4$), whereas in the same example, IPS weights were undefined for 5/6 cases and IIPS for 6/9 cases. This is why CIPS can "survive" under deterministic logging.

**2. Click-wise common support + Potential reward independence: Relaxing unbiasedness to the click layer**

CIPS requires two assumptions much milder than the old ones. One is **click-wise common support** (Condition 3.1): $p_c(x,a,\pi)>0\Rightarrow p_c(x,a,\pi_0)>0$. This means that actions clicked under the new policy must have a non-zero click probability under the logging policy. Since click probabilities are inherently stochastic, this condition usually holds even if $\pi_0$ is deterministic—whereas ranking-wise/position-wise support "never" holds under deterministic logs. The second is **potential reward independence** (Condition 3.2): $\mathbb{E}[R(a)|x,A]=\mathbb{E}[R(a)|x]=q_r(x,a)$. This means the downstream reward after a click depends only on the action $a$ itself, not other actions in the ranking (e.g., in e-commerce, whether one buys an item after clicking depends on the item attributes). This is more realistic than the "click-layer independence" required by IIPS. Under these two conditions, Theorem 3.1 proves CIPS is unbiased, $\mathbb{E}_{p(D)}[\hat{V}_{\text{CIPS}}]=V(\pi)$, making it the **first known method to provide low-bias ranking policy evaluation under completely deterministic logging**.

In practice, true click probabilities are unknown and must be estimated ($\hat{p}_c$). Theorem 3.2 gives the bias in this case:

$$\text{Bias}[\hat{V}_{\text{CIPS}}]=\mathbb{E}_{p(x)}\Big[\sum_{a\in A}p_c(x,a,\pi_0)\Big(\frac{\hat{p}_c(x,a,\pi)}{\hat{p}_c(x,a,\pi_0)}-\frac{p_c(x,a,\pi)}{p_c(x,a,\pi_0)}\Big)q_r(x,a)\Big],$$

The key conclusion: bias depends on how accurately the **ratio** of click probabilities is estimated, rather than the absolute click probabilities. Even if $\hat{p}_c$ has errors, the bias remains small as long as the ratio between new and old is stable.

**3. CDR: Reducing Variance via Regression without Increasing Bias**

While CIPS solves bias, its variance (Theorem 3.3) depends on the magnitude of the click importance weights and uses "zero-padding" for non-clicked actions. The authors integrate CIPS into a doubly robust framework by introducing a potential reward regression model $\hat{q}_r(x,a)$, resulting in Click-based Doubly Robust (CDR):

$$\hat{V}_{\text{CDR}}(\pi;D)=\frac{1}{n}\sum_{i=1}^n\sum_{a\in A}\frac{p_c(x_i,a,\pi)}{p_c(x_i,a,\pi_0)}\big(C_i(a)R_i(a)-p_c(x_i,a,A)\hat{q}_r(x_i,a)\big)+\mathbb{E}_{\pi(A|x_i)}\big[p_c(x_i,a,A)\hat{q}_r(x_i,a)\big].$$.

Theorem 4.1/4.2 proves that CDR has the **exact same bias** as CIPS (regardless of regression accuracy), meaning the regression model does not ruin unbiasedness. Theorem 4.3 shows that the variance of CDR includes an additional term dominated by the regression error $\Delta_r(x,a)=q_r(x,a)-\hat{q}_r(x,a)$—when $\hat{q}_r$ is more accurate than zero-padding, CDR variance is strictly smaller than CIPS. This is the transfer of the classic DR "unbiased + variance reduction" idea to the click-weight framework.

### Loss & Training
The core of the paper is the estimator, not a training objective. The only components requiring fitting are the click probability model $\hat{p}_c$ (estimated via a 3-layer neural network on logged data $D$) and the potential reward regression $\hat{q}_r$ for CDR. CIPS/CDR themselves are closed-form weighted average estimators. If further variance reduction is needed, standard techniques like weight clipping or self-normalization can be applied.

## Key Experimental Results

Experiments focus on the "deterministic logging policy" scenario. Metrics are MSE, Squared Bias, and Variance normalized by the true value $V(\pi)$, averaged over 100 random seeds with bootstrap 95% confidence intervals. Synthetic settings: $n=1000$, 6 actions, $K=6$, independence violation $\lambda=0.5$. Logging policy uses the Plackett-Luce model with $\alpha=\infty$ (completely deterministic). Baselines include IPS, IIPS, and RIPS, with CIPS (true CTR) as a reference upper bound.

### Main Results (Synthetic Data, Deterministic Logs)

| Variable | Key Phenomenon | Conclusion |
|----------|----------------|------------|
| Log size $n$ (500→4000) | CIPS has significantly lower MSE than baselines across all scales, primarily by reducing bias. | CIPS dominates throughout; estimating click probability introduces minimal extra bias. |
| Ranking length $K$ | Baseline bias increases with $K$; CIPS bias remains low. | CIPS is robust to ranking length. |
| Independence violation $\lambda$ | MSE of CIPS increases only slightly as $\lambda$ grows (Condition 3.2 violation increases). | CIPS remains optimal even when Condition 3.2 is not strictly met. |
| Log stochasticity $\alpha$ | Deterministic user ratio from 0.07 to 0.93. Baseline bias surges as $\alpha$ increases (more deterministic). | CIPS is low-bias in deterministic settings; IIPS catches up only when the log is highly stochastic. |

### Real-world Data (KuaiRec)
Experiments were conducted on the KuaiRec dataset (dense interaction matrix). True interaction matrices served as $q_r$, with click probabilities defined by thresholds. Setting $\epsilon=0.0$ made both new and old policies **deterministic** (the hardest setting), with 10 actions, $K=6$, and $\alpha=\infty$.

| Config | Key Metric | Description |
|--------|------------|-------------|
| Variation in $n$ | CIPS MSE is consistently lower than baselines. | Despite non-negligible bias (due to partial click-wise support violation), CIPS far outperforms baselines. |
| Variation in $\alpha$ | The more deterministic the log, the larger the relative advantage of CIPS. | While all biases grow with $\alpha$, CIPS grows slowest and stays most effective. |

### Key Findings
- **Bias is the primary enemy under deterministic logs**: Baseline variance is small (variance is not a major issue in deterministic logs); MSE is almost entirely contributed by bias. CIPS gains come from suppressing this bias.
- **The "ratio" of click probabilities is more important than "absolute values"**: CIPS results are close to CIPS (true CTR), validating Theorem 3.2—as long as the ratio is stable, inaccuracies in absolute click probabilities are tolerable.
- **CDR further reduces variance** (Appendix D.2): Adding a regression model to CIPS further reduces variance without increasing bias and improves downstream policy selection.
- **Honest degradation boundaries**: When both new and old policies are deterministic, click-wise support may be partially violated, leading to non-zero bias for CIPS—yet CIPS remains the relative optimum in this extreme setting.

## Highlights & Insights
- **Shifting the "source of randomness" for importance weights is a transferable paradigm**: Traditional OPE assumes randomness must come from the behavior policy. This paper points out that in ranking/two-stage feedback scenarios, user click behavior itself is an independent source of randomness that can serve as a weight carrier.
- **"Lowering" the unbiasedness condition**: Shifting from ranking-layer support → position-layer support → click-layer support. Conditions become easier to satisfy closer to the noisy observation layer. This is a trade-off: use weaker but more realistic assumptions for usability in harsher environments.
- **Clean integration of the DR framework**: CDR proves that "changing the weight carrier" and "doubly robust variance reduction" are orthogonal. Bias is determined by weights, while variance is influenced by the regression model.
- **Challenging long-standing beliefs**: The author explicitly states that this work shakes the belief that accurate OPE is impossible under deterministic logs, making a previously "intractable" setting tractable.

## Limitations & Future Work
- **Dependence on click probability estimability**: If the click probability model has systematic errors that vary between old and new policies, Theorem 3.2 fails. Non-zero bias was observed when both policies were deterministic.
- **Potential reward independence (Condition 3.2) is a simplification**: In reality, downstream rewards might be affected by other actions in the ranking (e.g., price comparison). While degradation was mild in experiments, extreme violations aren't fully characterized.
- **Verified only on small-to-medium action spaces**: Synthetic experiments used 6 actions/$K=6$, and real data used 10. The scalability of the click model and weight stability in industrial spaces (millions of actions) remains to be tested.
- **Lack of direct comparison with "embedding-based marginalization"**: Those methods target variance reduction and require action embeddings. CIPS targets bias reduction without embeddings, but their combination in "large and deterministic" scenarios is worth exploring.

## Related Work & Insights
- **vs ranking-wise IPS / IIPS / RIPS**: These use policy probability ratios and rely on logging policy stochasticity. Under deterministic logs, support is "forever" violated; CIPS uses click probability ratios, pushing support to the click layer.
- **vs Deficient support methods**: These handle cases where the logging policy gives zero probability to some actions. This paper handles the extreme "completely deterministic" case—a strict subset of deficient support that requires specialized design.
- **vs Great action space embedding marginalization**: Both "marginalize importance weights," but the former uses action embeddings to prevent weight explosion (targeting variance/requiring extra info), whereas CIPS uses click probabilities (targeting bias/not requiring embeddings).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to push unbiased OPE into completely deterministic logs by replacing policy randomness with click randomness.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic verification across $n$/$K$/$\lambda$/$\alpha$ on synthetic data + KuaiRec, though action space scale is small.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic from motivation to theory to experiment; complete characterization of bias/variance.
- Value: ⭐⭐⭐⭐⭐ Directly addresses the industrial pain point of deterministic policies and breaks the "impossible to evaluate" dogma.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning Design Skills as Memory Policies for Agentic Photonic Inverse Design](../../ICML2026/recommender/learning_design_skills_as_memory_policies_for_agentic_photonic_inverse_design.md)
- [\[ICLR 2026\] GoalRank: Group-Relative Optimization for a Large Ranking Model](goalrank_group-relative_optimization_for_a_large_ranking_model.md)
- [\[ICML 2025\] SIMPLEMIX: Frustratingly Simple Mixing of Off- and On-policy Data in Language Model Preference Learning](../../ICML2025/recommender/simplemix_frustratingly_simple_mixing_of_off-_and_on-policy_data_in_language_mod.md)
- [\[ICLR 2026\] From Evaluation to Defense: Advancing Safety in Video Large Language Models](from_evaluation_to_defense_advancing_safety_in_video_large_language_models.md)
- [\[NeurIPS 2025\] Validating LLM-as-a-Judge Systems under Rating Indeterminacy](../../NeurIPS2025/recommender/validating_llm-as-a-judge_systems_under_rating_indeterminacy.md)

</div>

<!-- RELATED:END -->
