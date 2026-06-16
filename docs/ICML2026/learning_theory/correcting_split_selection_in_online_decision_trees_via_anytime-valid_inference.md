---
title: >-
  [Paper Note] Correcting Split Selection in Online Decision Trees via Anytime-Valid Inference
description: >-
  [ICML 2026][learning_theory][testing-by-betting] The authors point out that the "fixed sample size" concentration inequalities used by the classic Hoeffding Tree (HT) for splitting in data streams are violated by its own "data-dependent stopping rule." They reformulate the splitting criterion using testing-by-betting + Universal Portfolio, allowing both single trees
tags:
  - ICML 2026
  - learning_theory
  - testing-by-betting
  - Hoeffding Tree
  - Adaptive Random Forest
date: 2026-05-08
content_hash: d8aa53c8656c924c
---
# Correcting Split Selection in Online Decision Trees via Anytime-Valid Inference

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2605.31239](https://arxiv.org/abs/2605.31239)  
**Code**: None  
**Area**: Data Stream Learning / Decision Trees / Sequential Inference  
**Keywords**: Online Decision Trees, anytime-valid inference, testing-by-betting, Hoeffding Tree, Adaptive Random Forest

## TL;DR
The authors point out that the "fixed sample size" concentration inequalities used by the classic Hoeffding Tree (HT) for splitting in data streams are violated by its own "data-dependent stopping rule." They reformulate the splitting criterion using testing-by-betting + Universal Portfolio, allowing both single trees and Adaptive Random Forests to maintain controlled Type-I errors at any stopping time, while achieving higher accuracy and smaller tree sizes across 12 real-world streams.

## Background & Motivation

**Background**: In data stream scenarios, bagging-based ensembles (especially Adaptive Random Forest, ARF) are the de facto standard; almost all such methods use the Hoeffding Tree (VFDT) as the base learner. The core mechanism of HT is: for each batch of data, it uses Hoeffding's inequality (or variants like McDiarmid or misclassification rate-based bounds) to determine if the "current best candidate split" is significantly better than the second best. A split is committed once the difference exceeds a threshold $\varepsilon(n_t(v),\delta)$.

**Limitations of Prior Work**: All existing modifications (linearized impurity, McDiarmid bounds, etc.) can only guarantee that the probability of a correct split is at least $1-\delta$ "at a certain fixed sample size $n$." However, HT actually "stops as soon as evidence is observed"—this is a data-dependent stopping rule (optional continuation).

**Key Challenge**: Fixed sample size inequalities fail to control the error rate at a data-triggered random stopping time $\tau$. Howard et al. (2021) demonstrated that under this misuse, the probability of an erroneous split can even reach 1. In other words, HT has lacked truly valid statistical guarantees to date; when embedded in ARF to process non-stationary streams, the problem worsens as the i.i.d. assumption is also violated.

**Goal**: Construct a splitting criterion that controls Type-I error under (i) data-dependent stopping and (ii) non-stationary and potentially dependent data streams, while ensuring that a split is committed in finite time when a true predictive advantage exists.

**Key Insight**: Abandon the approach of "comparing candidate splits under a fixed sample size" and shift to anytime-valid inference (SAVI). Specifically, use Ville's inequality to construct tests that are "valid at any time," which are naturally robust to optional stopping.

**Core Idea**: Reformulate the question of "whether to split" as an online model comparison. Treat the "unsplit leaf" as the incumbent and the "leaf after splitting by candidate $c$" as the challenger. The prequential prediction loss difference $\Delta_t^{v,c}$ is fed into a testing-by-betting wealth process, and a split is committed only when the wealth exceeds $1/\alpha^{v,c}$.

## Method

### Overall Architecture
The input is a sequence of arriving data $(X_t,Y_t)$, and the output is a decision tree that expands over time. Each leaf $v$ maintains a set of candidate splits $C_v$. For each candidate $c=(j,s)$, two "shadow predictors" are maintained: the incumbent $m^v$ predicts using the empirical class distribution of $v$, while the challenger $m^{v_c}$ predicts by splitting $v$ into two leaves, each maintaining its own empirical class distribution. Both predict based on past information, and then calculate the prequential prediction loss difference $\Delta_t^{v,c}=\ell(m^v_{t-1}(X_t),Y_t)-\ell(m^{v_c}_{t-1}(X_t),Y_t)\in[-1,1]$. This difference is used to accumulate "evidence" through an anytime-valid test. When the evidence is strong enough, leaf $v$ is split using $c^\star$; otherwise, observation continues.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["Data stream arrives (X_t, Y_t)"] --> B["Leaf v + Candidate split c=(j,s)"]
    subgraph CMP["Online Model Comparison"]
        direction TB
        C1["Incumbent: Unsplit leaf<br/>Predicts with empirical distribution of v"]
        C2["Challenger: Split c into two leaves<br/>Each maintains empirical distribution"]
        C1 --> D["Prediction loss difference Δ_t ∈ [−1,1]"]
        C2 --> D
    end
    B --> CMP
    D --> E["Testing-by-betting + Universal Portfolio<br/>Wealth W_t = W_{t-1}(1+β_t·Δ_t), β_t is parameter-free"]
    G["Confidence sequence variant + Global α-allocation"] -.->|Set threshold 1/α^{v,c}| F
    E --> F{"Wealth crossing? W_t ≥ 1/α^{v,c}<br/>(Ville's Inequality)"}
    F -->|No, continue observing| A
    F -->|Yes| H["Commit split c⋆, split leaf v"]
    H --> A
```

### Key Designs

**1. Reframing "Impurity Maximization" as "Online Model Comparison": Replacing an unstable target under drift**

HT originally intended to estimate the population impurity difference $\Delta^{v,c}=\mathcal{I}(p(v))-P_L\mathcal{I}(p(v_c^L))-P_R\mathcal{I}(p(v_c^R))$. However, in non-stationary streams, this target itself drifts over time, making a "globally optimal split" non-existent. This paper adopts a clear, testable null hypothesis: the strong null $H_0^{v,c}:\forall t,\;\delta_t^{v,c}=\mathbb{E}[\Delta_t^{v,c}\mid\mathcal{F}_{t-1}]\le 0$, meaning the challenger is no better than the incumbent at any time. Evidence is derived from the prequential prediction loss difference $\Delta_t^{v,c}=\ell(m^v_{t-1}(X_t),Y_t)-\ell(m^{v_c}_{t-1}(X_t),Y_t)\in[-1,1]$, where the loss is log loss (corresponding to entropy) or Brier score (corresponding to Gini), normalized to $[0,1]$. Directly using the loss difference bypasses the non-linearity of impurity measures and eliminates dependence on a "population optimal split" in drifting environments.

**2. Testing-by-betting + Universal Portfolio: Making "whether to split" robust to optional stopping via betting**

Fixed sample size inequalities cannot withstand HT's "stop upon evidence" data-dependent stopping rule. Thus, the criterion is replaced with an anytime-valid wealth process. Starting from an initial wealth $W_s=1$, an $\mathcal{F}_{t-1}$-measurable betting fraction $\beta_t\in[0,1]$ is chosen at each step, and wealth is updated as $W_t=W_{t-1}(1+\beta_t\Delta_t)$. Under $H_0$, $(W_t)$ is a non-negative supermartingale. Ville’s inequality $\mathbb{P}_{H_0}(\sup_t W_t\ge 1/\alpha)\le\alpha$ ensures that "crossing the wealth threshold" is a valid rejection rule at any time. The split is committed at $\tau^{v,c}=\inf\{t:W_t^{v,c}\ge 1/\alpha^{v,c}\}$. Crucially, the betting fraction $\beta_t$ is not manually tuned—which would be equivalent to peeking at test statistics—but is determined via a parameter-free Universal Portfolio, mixing all constant rebalanced portfolios using a $\mathrm{Beta}(1/2,1/2)$ Jeffreys prior:

$$\beta_t=\frac{\int_0^1 \beta\prod_{u}(1+\beta\Delta_u)\,dF_+(\beta)}{\int_0^1 \prod_u(1+\beta\Delta_u)\,dF_+(\beta)}.$$

The UP achieves the optimal constant growth rate in i.i.d. cases without manual tuning, serving as a powerful tool in the SAVI framework that does not rely on independence assumptions.

**3. Confidence Sequence Variant + Global $\alpha$-allocation: Adding "average advantage" semantics and lifetime error control**

While the strong null leads to earlier splits and better performance in practice, a "weak null" $H_{w,0}^{v,c}:\bar\delta_t^{v,c}=\frac{1}{t-s^v}\sum_u \delta_u^{v,c}\le 0$ is also presented for those who require guaranteed monotonic loss reduction at the moment of commitment. This is constructed using an empirical Bernstein confidence sequence $(L_t,U_t)$, stopping at $\tau_w^{v,c}=\inf\{t:L_t>0\}$. For global control, the error budget $\alpha$ is decomposed across all candidates $(v,c)$. As long as $\sum_{v,c}\alpha^{v,c}\le\alpha$, the union bound combined with anytime-validity guarantees $\mathbb{P}(\exists\text{false split ever})\le\alpha$. This preserves the family-wise error rate even for a tree running indefinitely. To test for a strictly positive advantage $\varepsilon>0$, one simply uses an $\varepsilon$-shifted wealth process $W_{\varepsilon,t}=\prod_u(1+\beta_u(\Delta_u-\varepsilon))$ and constrains $\beta$ to $[0,1/(1+\varepsilon)]$.

### Loss & Training
The method uses log loss for classification and squared loss for regression (where the maximum observed loss is maintained online to scale values to $[0,1]$). Default hyperparameters are $n_{\min}=20$, $\varepsilon=0$, and $\alpha$ set as the family-wise significance level. Three theoretical guarantees are provided: (i) Thm 4.1, global anytime validity; (ii) Thm 4.2, when a persistent advantage $\Delta>0$ exists, the commitment time is finite and reaches $\tilde{\mathcal{O}}(\log(1/\alpha^{v,c})/\Delta^2)$ with high probability; (iii) Thm 4.3, under i.i.d. conditions with convex loss, the expected loss of the deployed model is non-increasing both between and at the moment of split commitments.

## Key Experimental Results

### Main Results
The authors evaluated the prequential (test-then-train) protocol over 10 runs on 12 data streams (6 regression, 6 classification). They compared HT and ARF against AVT_B (single tree) and AVF_B (forest) using the anytime-valid splitting criterion. AVT_B outperformed HT on most streams (except abalone), approaching or exceeding ARF on bike/fried/hyper100k. AVF_B achieved the best overall performance on nearly all datasets. In terms of model size, AVF_B was actually shallower with fewer nodes than the single-tree AVT_B; AVT_B was slightly deeper than HT because it must handle drift independently without bagging.

| Setting | Advantage | Key Phenomenon | Remarks |
|------|---------|---------|------|
| Single Tree AVT_B vs HT | Majority of 12 streams | Prediction curves are stable and monotonic; HT shows performance "collapses" on i.i.d. RandomTree. | abalone is a counter-example where HT is slightly better. |
| Forest AVF_B vs ARF | Almost all streams | AVF_B is the best and has the shallowest trees. | Gap increases with the intensity of drift. |

### Ablation Study
A warm-up experiment (RandomTree with 10 numerical and 10 categorical features, depth 4) and a time complexity table highlight the trade-offs between AVT_B / AVT_CS / AVF_B and HT/ARF.

| Configuration | Key Metric | Description |
|------|---------|------|
| HT (warm-up i.i.d.) | Multiple collapses + leaf count >> truth (8) | Failure of fixed-sample inequality + optional stopping. |
| AVT_B (betting) | Stable monotonic improvement, leaves ≤ 8 | Strictly aligns with Thm 4.3 monotonicity. |
| AVT_CS (weak null + CS) | More conservative than AVT_B, later splits | Affected by historical averages, slightly worse prediction. |
| Inference Latency AVT_B vs HT | 0.06–0.12 ms vs 0.03–0.12 ms | Same magnitude, pure tree traversal. |
| Update Latency AVT_B vs HT | 1–25 ms vs 0.07–0.6 ms | UP wealth maintenance is the main overhead. |
| Update Latency AVF_B vs ARF | Within 1 order of magnitude | Shallower trees offset some costs; naturally parallelizable. |

### Key Findings
- **Strong null is more practical than weak null**: Betting-based tests result in earlier splits and lower loss; the weak null is conservative as it is weighed down by historical averages.
- **Smaller tree size in AVF_B is a structural advantage**: Due to stricter splitting criteria, each ensemble member only grows new nodes when a true persistent advantage exists, making the forest more compact.
- **Overhead stems from wealth maintenance, not inference**: Update times are higher by an order of magnitude due to UP numerical integration (approximated discretely), but remain in the millisecond range for real-time use; candidate splits are naturally parallel.

## Highlights & Insights
- **Applying SAVI to Stream Learning**: This is the first work to systematically graft testing-by-betting, UP, and empirical Bernstein confidence sequences onto decision tree splitting; previous work in the stream learning community mainly patched the Hoeffding/McDiarmid framework.
- **"Shadow Challenger" is a Reusable Design Pattern**: Maintaining an incumbent and training candidate models in parallel only for loss evaluation can be directly transferred to other "online model selection" scenarios, such as streaming neural networks or streaming GBDT.
- **Clean Alignment of Theory and Engineering**: The monotonicity of Thm 4.3 is essentially a direct consequence of convex loss and plug-in leaf updates. By isolating this, the authors emphasize that anytime-valid tests only ensure that "structural updates (splits) do not break monotonicity," allowing the methodology and theory to complement each other.

## Limitations & Future Work
- **Update cost remains 10–100x higher than a single tree**: UP requires numerical integration over the Beta prior, which becomes expensive with more candidates. Real-time performance relies on parallelism (GPU/multi-core).
- **Monotonicity under the strong null still requires "persistent advantage"**: The authors state that ensuring $\mathbb{E}[\ell(\hat m_{\tau_k})]$ strictly decreases at commitment requires the weak null + $\varepsilon$-shifted test; for the strong null, it is only "empirically observed."
- **Coarseness of $\alpha$-allocation**: The global budget $\alpha$ is shared across all candidates, which may be many in high-dimensional settings. How to distribute $\alpha^{v,c}$ to prevent the power of the test from being diluted was not analyzed for sensitivity in the main text.
- **Expansion beyond current stream scenarios**: While the method is naturally suited for streaming GBDT or in-stream pruning, these were not explored. Non-binary splits and categorical feature handling were only briefly mentioned.

## Related Work & Insights
- **vs. Hoeffding Tree / VFDT (Domingos & Hulten 2000)**: HT is directly challenged—its nominal guarantees are void due to the combination of fixed-sample inequalities and data-dependent stopping. This work replaces the criterion while keeping the incremental growth structure.
- **vs. McDiarmid-bound Variants (Rutkowski 2012; De Rosa & Cesa-Bianchi 2017)**: These simply substitute tighter inequalities but still assume a fixed $n$. This paper argues they do not resolve the core optional stopping issue.
- **vs. Orabona & Jun (2023) / Choe & Ramdas (2024)**: This paper implements sequential forecaster comparison as a practical criterion for decision tree splitting and highlights the importance of "strong vs. weak null" in tree learning.
- **vs. Adaptive Random Forest (Gomes 2017/2018)**: Instead of replacing ARF layers, this work replaces the HT base learner, proving that ARF’s performance was partially hindered by its base learner's lack of statistical validity.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematically introduces SAVI/testing-by-betting to online trees with dual guarantees for error control and monotonicity.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 12 real streams, synthetic warm-ups, and timing overheads for both trees and forests.
- Writing Quality: ⭐⭐⭐⭐ Clear problem statement and correlation between theorems and algorithms.
- Value: ⭐⭐⭐⭐ A fundamental replacement of the core splitting criterion in stream learning; highly relevant for finance, security, and IoT.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Generalizing Analogical Inference from Boolean to Continuous Domains](../../AAAI2026/learning_theory/generalizing_analogical_inference_from_boolean_to_continuous_domains.md)
- [\[ICML 2026\] Parsimonious Learning-Augmented Online Metric Matching](parsimonious_learning-augmented_online_metric_matching.md)
- [\[AAAI 2026\] A Switching Framework for Online Interval Scheduling with Predictions](../../AAAI2026/learning_theory/a_switching_framework_for_online_interval_scheduling_with_pr.md)
- [\[NeurIPS 2025\] Computable Universal Online Learning](../../NeurIPS2025/learning_theory/computable_universal_online_learning.md)
- [\[AAAI 2026\] Streaming Generated Gaussian Process Experts for Online Learning and Control: Extended Version](../../AAAI2026/learning_theory/streaming_generated_gaussian_process_experts_for_online_learning_and_control_ext.md)

</div>

<!-- RELATED:END -->
