---
title: >-
  [Paper Note] Correcting Split Selection in Online Decision Trees via Anytime-Valid Inference
description: >-
  [ICML 2026][Online decision trees] The authors point out that the "fixed sample size" concentration inequality used by the classic Hoeffding Tree (HT) for splitting on data streams is invalidated by its own "data-depende…
tags:
  - "ICML 2026"
  - "Online decision trees"
  - "anytime-valid inference"
  - "testing-by-betting"
  - "Hoeffding Tree"
  - "Adaptive Random Forest"
date: 2026-05-08
content_hash: d1b8c2e056403532
---

# Correcting Split Selection in Online Decision Trees via Anytime-Valid Inference

**Conference**: ICML 2026  
**arXiv**: [2605.31239](https://arxiv.org/abs/2605.31239)  
**Code**: None  
**Area**: Data stream learning / Decision trees / Sequential inference  
**Keywords**: Online decision trees, anytime-valid inference, testing-by-betting, Hoeffding Tree, Adaptive Random Forest

## TL;DR
The authors point out that the "fixed sample size" concentration inequality used by the classic Hoeffding Tree (HT) for splitting on data streams is invalidated by its own "data-dependent stopping rule." They reformulate the split criterion using testing-by-betting + Universal Portfolio, allowing both single trees and Adaptive Random Forests to maintain controlled Type-I errors at any stopping time while achieving higher accuracy and smaller tree sizes across 12 real-world streams.

## Background & Motivation

**Background**: In data stream scenarios, bagging-based ensembles (especially Adaptive Random Forest, ARF) are the de facto standard; almost all such methods use the Hoeffding Tree (VFDT) as the base learner. The core mechanism of HT is: for each batch of data, use the Hoeffding inequality (or modified versions like McDiarmid or misclassification rate) to determine if the "current best candidate split" is significantly better than the second best. A split is committed once the threshold $\varepsilon(n_t(v),\delta)$ is exceeded.

**Limitations of Prior Work**: All existing modified versions (linearized impurity, McDiarmid bounds, etc.) can only guarantee that the probability of a correct split is at least $1-\delta$ "at a certain fixed sample size $n$." However, HT actually "stops upon seeing evidence"—this is a data-dependent stopping rule (optional continuation).

**Key Challenge**: Fixed-sample inequalities cannot control the error rate at a random stopping time $\tau$ triggered by data. Howard et al. (2021) demonstrated that under this misuse, the probability of an incorrect split can even reach 1. In other words, HT has lacked truly valid statistical guarantees to date; when embedded in ARF for non-stationary streams, the problem worsens as the i.i.d. assumption is also violated.

**Goal**: Construct a split criterion that controls Type-I errors under (i) data-dependent stopping and (ii) non-stationary and potentially dependent data streams, while ensuring a split is committed within finite time when a true predictive advantage exists.

**Key Insight**: Abandon the approach of "comparing which candidate split is optimal under a fixed sample size" and move toward anytime-valid inference (SAVI). By using Ville’s inequality to construct tests that "hold at any time," the method becomes naturally robust to optional stopping.

**Core Idea**: Reformulate "whether to split" as an online model comparison—treating the "unsplit leaf" as the incumbent and the "leaves after splitting by candidate $c$" as the challenger. The prequential predictive loss difference $\Delta_t^{v,c}$ is fed into a testing-by-betting wealth process, and a split is committed only when the wealth exceeds $1/\alpha^{v,c}$.

## Method

### Overall Architecture
The input is a sequence of arriving $(X_t,Y_t)$, and the output is a decision tree that expands over time. Each leaf $v$ maintains a set of candidate splits $C_v$. For each candidate $c=(j,s)$, two "shadow predictors" are run simultaneously: the incumbent $m^v$ predicts using the empirical class distribution on $v$; the challenger $m^{v_c}$ splits $v$ into left and right leaves along the candidate split, each maintaining its own empirical class distribution. Both predict based on past information, and the predictive loss difference $\Delta_t^{v,c}=\ell(m^v_{t-1}(X_t),Y_t)-\ell(m^{v_c}_{t-1}(X_t),Y_t)\in[-1,1]$ is calculated using the observed $Y_t$. Evidence is accumulated via an anytime-valid test, and $v$ is split using $c^\star$ when the evidence is strong enough; otherwise, observation continues.

### Key Designs

1. **Reconstructing "Impurity Maximization" as "Online Model Comparison"**:
    - **Function**: Replaces the split criterion from "finding the maximum impurity drop" to "determining if the challenger has lower predictive loss."
    - **Mechanism**: Instead of estimating the population impurity difference $\Delta^{v,c}=\mathcal{I}(p(v))-P_L\mathcal{I}(p(v_c^L))-P_R\mathcal{I}(p(v_c^R))$, where the target itself drifts over time in non-stationary streams, the authors test a clear null hypothesis: the strong null $H_0^{v,c}:\forall t,\;\delta_t^{v,c}=\mathbb{E}[\Delta_t^{v,c}\mid\mathcal{F}_{t-1}]\le 0$, meaning the challenger is no better than the incumbent at any time. Losses use log loss (entropy) or Brier score (Gini), normalized to $[0,1]$.
    - **Design Motivation**: Using prequential loss differences directly as evidence bypasses the non-linearity of impurity and eliminates the need for the concept of a "population optimal split," which is untenable in drifting scenarios.

2. **Testing-by-betting + Universal Portfolio**:
    - **Function**: Turns "whether to reject $H_0$" into a "betting game," where the rate of wealth growth determines when to split.
    - **Mechanism**: Starting with unit wealth $W_s=1$, a betting fraction $\beta_t\in[0,1]$ that is $\mathcal{F}_{t-1}$-measurable is chosen at each step, and wealth is updated via $W_t=W_{t-1}(1+\beta_t\Delta_t)$. Under $H_0$, $(W_t)$ is a non-negative supermartingale. By Ville’s inequality $\mathbb{P}_{H_0}(\sup_t W_t\ge 1/\alpha)\le\alpha$, wealth exceeding the threshold is a valid anytime-valid rejection rule. Instead of manual tuning, $\beta_t$ uses a parameter-free Universal Portfolio: a mixture over all constant rebalancing portfolios with a $\mathrm{Beta}(1/2,1/2)$ Jeffreys prior, $\beta_t=\frac{\int_0^1 \beta\prod_{u}(1+\beta\Delta_u)\,dF_+(\beta)}{\int_0^1 \prod_u(1+\beta\Delta_u)\,dF_+(\beta)}$. The commitment time is $\tau^{v,c}=\inf\{t:W_t^{v,c}\ge 1/\alpha^{v,c}\}$, picking the candidate with the highest wealth if multiple cross the threshold simultaneously.
    - **Design Motivation**: Testing-by-betting is currently the most powerful tool in the SAVI framework that does not rely on independence assumptions; UP achieves the optimal growth rate of the best constant portfolio in i.i.d. cases without hyperparameter tuning, preventing "cheating" by tuning $\beta_t$ based on test statistics.

3. **Confidence Sequence Variants + Global $\alpha$-Allocation**:
    - **Function**: (i) Provides an equivalent test for a "weak null" of average advantage; (ii) ensures the entire tree maintains an error probability of no more than $\alpha$ over its lifetime.
    - **Mechanism**: The weak null $H_{w,0}^{v,c}:\bar\delta_t^{v,c}=\frac{1}{t-s^v}\sum_u \delta_u^{v,c}\le 0$ is tested using an empirical Bernstein confidence sequence $(L_t,U_t)$, with stopping time $\tau_w^{v,c}=\inf\{t:L_t>0\}$. For global control, the budget $\alpha$ is distributed across all $(v,c)$: as long as $\sum_{v,c}\alpha^{v,c}\le\alpha$, the union bound plus anytime-validity yields $\mathbb{P}(\exists\text{false split ever})\le\alpha$. To test for a strictly positive advantage $\varepsilon>0$, wealth is replaced with an $\varepsilon$-shifted version $W_{\varepsilon,t}=\prod_u(1+\beta_u(\Delta_u-\varepsilon))$ and $\beta$ is constrained to $[0,1/(1+\varepsilon)]$.
    - **Design Motivation**: The authors found that the strong null splits earlier and performs better in practice, though theoretically, guaranteeing monotonic loss reduction at the commit time requires the "average advantage" semantics of the weak null. Providing both allows users to choose based on the scenario; global $\alpha$-allocation maintains the family-wise error rate even for a tree running indefinitely.

### Loss & Training
Log loss is used for classification and squared loss for regression (autonomously scaled to $[0,1]$ using maximum observed loss). Default hyperparameters are $n_{\min}=20$, $\varepsilon=0$, and $\alpha$ as the family-wise significance level. Three theoretical guarantees are provided: (i) Thm 4.1, global anytime validity; (ii) Thm 4.2, when a persistent advantage $\Delta>0$ exists, the commit time is finite and reaches $\tilde{\mathcal{O}}(\log(1/\alpha^{v,c})/\Delta^2)$ with high probability; (iii) Thm 4.3, under i.i.d.+convex loss, the expected loss of the deployed model is monotonically non-increasing between and at commit times.

## Key Experimental Results

### Main Results
The authors ran 10 iterations on 12 data streams (6 regression + 6 classification) using the prequential (test-then-train) protocol, comparing HT, ARF, and anytime-valid variants AVT_B (single tree) / AVF_B (forest). AVT_B beat HT on most streams (except abalone) and even approached or exceeded ARF on bike/fried/hyper100k; AVF_B achieved the best overall performance on almost all datasets. In terms of model scale, AVF_B was shallower and had fewer nodes than the single-tree AVT_B; AVT_B was slightly deeper than HT because it must handle drift without bagging.

| Setting | Gain Location | Key Phenomenon | Remarks |
|------|---------|---------|------|
| Single tree AVT_B vs HT | Majority of 12 streams | Prediction curves are stable and monotonic; HT shows frequent performance "collapses" on i.i.d. RandomTree | HT is slightly better on abalone as a counterexample |
| Forest AVF_B vs ARF | Almost all streams | AVF_B is best with the shallowest trees | Gap increases with drift intensity |

### Ablation Study
The cost-benefit trade-off between AVT_B / AVT_CS / AVF_B and HT/ARF is shown via warm-up (RandomTree, 10 numerical + 10 categorical features, depth 4) and time overhead comparisons.

| Configuration | Key Metric | Description |
|------|---------|------|
| HT (warm-up i.i.d.) | Multiple collapses + leaf count >> ground truth 8 | Failure of fixed-sample inequality + optional stopping |
| AVT_B (betting) | Stable monotonic improvement, leaves ≤ 8 | Strictly aligns with Thm 4.3 monotonicity |
| AVT_CS (weak null + Bernstein CS) | Slightly more conservative than AVT_B, later splits | Affected by historical average; slightly worse prediction |
| Inference Latency AVT_B vs HT | 0.06–0.12 ms vs 0.03–0.12 ms | Same magnitude; pure tree traversal |
| Update Latency AVT_B vs HT | 1–25 ms vs 0.07–0.6 ms | UP wealth maintenance is the main overhead |
| Update Latency AVF_B vs ARF | Within 1 order of magnitude | Shallower trees offset costs; naturally parallelizable |

### Key Findings
- **Strong null is more practical than weak null**: Betting-based tests split earlier with lower loss; the weak null is conservative as it is dragged down by historical averages.
- **AVF_B’s smaller tree size is a structural advantage**: Due to stricter split criteria, each ensemble member only grows new nodes when there is a true persistent advantage, making the forest more compact.
- **Extra overhead stems from wealth maintenance, not inference**: Update times are increased by an order of magnitude due to UP numerical integration (approximated discretely), but remain in the millisecond range for real-time use; candidate splits are naturally parallelizable.

## Highlights & Insights
- **Porting the SAVI stack to stream learning**: This is the first work to systematically graft testing-by-betting + UP + empirical Bernstein confidence sequences onto decision tree split criteria; previously, the stream learning community mostly patched the Hoeffding/McDiarmid framework.
- **"Shadow challenger" is a reusable design pattern**: Keeping the incumbent static while training candidate models in parallel for loss difference evaluation is a mechanism that can be migrated to other "online model selection" scenarios like streaming neural networks or streaming GBDT.
- **Clean alignment between theory and engineering**: The monotonicity in Thm 4.3 is almost a direct consequence of "convex loss + plug-in leaf updates." The authors isolate this to emphasize that anytime-valid tests ensure "structural updates (splitting) do not break monotonicity," allowing the methodology and theory to complement each other.

## Limitations & Future Work
- **Update costs remain 10–100x higher than a single tree**: UP numerical integration with a Beta prior becomes expensive as candidates increase; real-time performance depends on parallelism (GPU/multi-core), and single-thread pressure is evident on long streams like nzenergy.
- **Monotonicity at commit time under strong null still requires "persistent advantage" assumptions**: The authors state that ensuring $\mathbb{E}[\ell(\hat m_{\tau_k})]$ strictly decreases at commit times requires the weak null + $\varepsilon$-shifted test; for the strong null, this is only "empirically observed."
- **Granularity of $\alpha$-allocation is not deeply discussed**: All candidate splits share a global budget $\alpha$, but the candidate set $C_v$ can be large (high-dimensional + multiple thresholds). The paper uses a schedule in the appendix for allocating $\alpha^{v,c}$ but lacks sensitivity analysis on avoiding the dilution of detection power.
- **Extensions beyond stream scenarios**: While naturally suited for streaming GBDT or in-stream pruning, these are not covered; non-binary splits, categorical features, and regression tree extrapolation are only briefly mentioned as "standard adaptations."

## Related Work & Insights
- **vs Hoeffding Tree / VFDT (Domingos & Hulten 2000)**: HT is the direct subject of challenge—fixed-sample inequalities + data-dependent stopping = invalid nominal guarantees. This paper replaces the split criterion with SAVI while retaining HT’s "single-pass, incremental, leaf-wise split" skeleton, allowing for drop-in replacement.
- **vs McDiarmid-bound modifications (Rutkowski 2012; De Rosa & Cesa-Bianchi 2017)**: These works only replace Hoeffding with tighter non-linear concentration inequalities while still assuming a fixed $n$. This paper argues they do not solve optional stopping and have flawed theoretical foundations.
- **vs UP by Orabona & Jun (2023) / Sequential Forecaster Comparison by Choe & Ramdas (2024)**: This paper implements the latter's sequential forecaster comparison as a decision tree split criterion and highlights the empirical differences between "strong vs weak nulls" in tree learning.
- **vs Adaptive Random Forest (Gomes 2017/2018)**: This work does not replace ARF's bagging or drift detection layers but replaces the internal HT base learner, proving that AVF_B outperforms ARF in both performance and tree scale, suggesting ARF's advantages were previously hindered by the statistical invalidity of its base learner.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematically introduces SAVI/testing-by-betting to streaming decision trees, providing dual guarantees of global error control and monotonicity.
- Experimental Thoroughness: ⭐⭐⭐⭐ 12 real-world streams + warm-up synthetic experiments + time overhead tables, covering both single trees and forests. Comparison with recent streaming GBDT/online boosting is in the appendix.
- Writing Quality: ⭐⭐⭐⭐ Clear problem statement using Howard et al. (2021) to illustrate why HT is invalid; theorem statements correspond one-to-one with algorithm pseudocode.
- Value: ⭐⭐⭐⭐ A rare contribution where the high-level algorithm remains but the underlying criterion is fundamentally replaced; highly valuable for finance, network intrusion, and IoT stream scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Active Learning for Decision Trees with Provable Guarantees](../../ICLR2026/others/active_learning_for_decision_trees_with_provable_guarantees.md)
- [\[ICML 2026\] Inference of Online Newton Methods with Nesterov's Accelerated Sketching](inference_of_online_newton_methods_with_nesterovs_accelerated_sketching.md)
- [\[AAAI 2026\] From Decision Trees to Boolean Logic: A Fast and Unified SHAP Algorithm](../../AAAI2026/others/from_decision_trees_to_boolean_logic_a_fast_and_unified_shap_algorithm.md)
- [\[ICML 2026\] Decision Tree Learning on Product Spaces](decision_tree_learning_on_product_spaces.md)
- [\[NeurIPS 2025\] Improving Decision Trees through the Lens of Parameterized Local Search](../../NeurIPS2025/others/improving_decision_trees_through_the_lens_of_parameterized_local_search.md)

</div>

<!-- RELATED:END -->
