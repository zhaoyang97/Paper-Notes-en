---
title: >-
  [Paper Note] Correcting Split Selection in Online Decision Trees via Anytime-Valid Inference
description: >-
  [ICML 2026 Spotlight][Data Stream Learning][Online Decision Trees] The authors point out that the "fixed sample size" concentration inequalities used by the classic Hoeffding Tree (HT) for splitting on data streams are violated by its own "data-dependent stopping rule." They reformulate the split criterion using testing-by-betting + Universal Portfolio, allowing both single trees and Adaptive Random Forests to maintain controlled Type-I errors at any stopping time…
tags:
  - "ICML 2026 Spotlight"
  - "Data Stream Learning"
  - "Decision Trees"
  - "Sequential Inference"
  - "Online Decision Trees"
  - "anytime-valid inference"
  - "testing-by-betting"
  - "Hoeffding Tree"
  - "Adaptive Random Forest"
date: 2026-05-08
content_hash: a13ccad436a5f251
---

# Correcting Split Selection in Online Decision Trees via Anytime-Valid Inference

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2605.31239](https://arxiv.org/abs/2605.31239)  
**Code**: None  
**Area**: Data Stream Learning / Decision Trees / Sequential Inference  
**Keywords**: Online Decision Trees, anytime-valid inference, testing-by-betting, Hoeffding Tree, Adaptive Random Forest

## TL;DR
The authors point out that the "fixed sample size" concentration inequalities used by the classic Hoeffding Tree (HT) for splitting on data streams are violated by its own "data-dependent stopping rule." They reformulate the split criterion using testing-by-betting + Universal Portfolio, allowing both single trees and Adaptive Random Forests to maintain controlled Type-I errors at any stopping time, while achieving higher accuracy and smaller tree sizes across 12 real-world streams.

## Background & Motivation

**Background**: In data stream scenarios, bagging-based ensembles (especially Adaptive Random Forest, ARF) are the de facto standard; almost all such methods use the Hoeffding Tree (VFDT) as the base learner. The core mechanism of HT is: for each incoming batch of data, use Hoeffding's inequality (or revised versions like McDiarmid or misclassification rates) to judge if the "current best candidate split" is significantly better than the second best, committing the split once it exceeds a threshold $\varepsilon(n_t(v),\delta)$.

**Limitations of Prior Work**: All existing revised versions (linearized impurity, McDiarmid bounds, etc.) can only guarantee that the probability of a correct split is at least $1-\delta$ "at a fixed sample size $n$." However, HT effectively "stops when it sees evidence"—this is a data-dependent stopping rule (optional continuation).

**Key Challenge**: Fixed sample size inequalities cannot control error rates at a random stopping time $\tau$ triggered by data. Howard et al. (2021) demonstrated that under such misuse, the probability of a false split can even reach 1. In other words, HT has lacked truly valid statistical guarantees to date; the issue becomes more severe when embedded in ARF for non-stationary streams, where the i.i.d. assumption is also violated.

**Goal**: Construct a split criterion that controls Type-I error under (i) data-dependent stopping and (ii) non-stationary and potentially dependent data streams, while ensuring a split is committed in finite time when a true predictive advantage exists.

**Key Insight**: Abandon the comparison of candidate splits at a fixed sample size and pivot to anytime-valid inference (SAVI) — specifically, using Ville's inequality to construct tests that "hold at all times," which are naturally robust to optional stopping.

**Core Idea**: Reformulate "whether to split" as an online model comparison — treating the "unsplit leaf" as the incumbent and the "leaf split by candidate $c$" as the challenger. The prequential predictive loss difference $\Delta_t^{v,c}$ enters a testing-by-betting wealth process; a split is committed only when the wealth exceeds $1/\alpha^{v,c}$.

## Method

### Overall Architecture
The input is a sequentially arriving data stream $(X_t,Y_t)$, and the output is a decision tree that expands over time. Each leaf $v$ maintains a set of candidate splits $C_v$. For each candidate $c=(j,s)$, two "shadow predictors" are run simultaneously: the incumbent $m^v$ predicts using the empirical class distribution of $v$, while the challenger $m^{v_c}$ splits $v$ into left and right leaves, each maintaining its own empirical distribution. Both predict based on past information, and the predictive loss difference $\Delta_t^{v,c}=\ell(m^v_{t-1}(X_t),Y_t)-\ell(m^{v_c}_{t-1}(X_t),Y_t)\in[-1,1]$ is calculated using the observed $Y_t$. This difference accumulates "evidence" through an anytime-valid test; a split $c^\star$ is committed when evidence is sufficiently strong, otherwise the algorithm continues to observe.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["Data stream arrives sequentially (X_t, Y_t)"] --> B["Leaf v + Candidate split c=(j,s)"]
    subgraph CMP["Online Model Comparison"]
        direction TB
        C1["Incumbent: Unsplit leaf<br/>Predicts via v's empirical distribution"]
        C2["Challenger: Split c into L/R leaves<br/>Each maintains empirical distribution"]
        C1 --> D["Predictive loss difference Δ_t ∈ [−1,1]"]
        C2 --> D
    end
    B --> CMP
    D --> E["Testing-by-betting + Universal Portfolio<br/>Wealth W_t = W_{t-1}(1+β_t·Δ_t), β_t is parameter-free"]
    G["Confidence sequence variant + Global α-allocation"] -.->|Set threshold 1/α^{v,c}| F
    E --> F{"Wealth crosses? W_t ≥ 1/α^{v,c}<br/>(Ville's Inequality)"}
    F -->|No, continue| A
    F -->|Yes| H["Commit split c⋆, split leaf v"]
    H --> A
```

### Key Designs

**1. From "impurity maximization" to "online model comparison": Replacing a goal that fails under drift**

HT originally intended to estimate the population impurity difference $\Delta^{v,c}=\mathcal{I}(p(v))-P_L\mathcal{I}(p(v_c^L))-P_R\mathcal{I}(p(v_c^R))$, but in non-stationary streams, this target drifts over time; there is no "global optimal split." **Ours** replaces this with a clear, testable strong null hypothesis $H_0^{v,c}:\forall t,\;\delta_t^{v,c}=\mathbb{E}[\Delta_t^{v,c}\mid\mathcal{F}_{t-1}]\le 0$, meaning the challenger is never better than the incumbent. Evidence is derived from the prequential predictive loss difference $\Delta_t^{v,c}=\ell(m^v_{t-1}(X_t),Y_t)-\ell(m^{v_c}_{t-1}(X_t),Y_t)\in[-1,1]$, using log loss (entropy) or Brier score (Gini) scaled to $[0,1]$. Using loss differences directly bypasses the non-linear issues of impurity and no longer depends on the concept of a "population optimal split" in drifting scenarios.

**2. Testing-by-betting + Universal Portfolio: Turning "whether to split" into a bet naturally robust to optional stopping**

Fixed sample size inequalities cannot withstand HT's "stop when evidence appears" data-dependent stopping. Thus, the criterion is changed to an anytime-valid wealth process. Starting from unit wealth $W_s=1$, a betting fraction $\beta_t\in[0,1]$ that is $\mathcal{F}_{t-1}$-measurable is chosen at each step, and wealth is updated via $W_t=W_{t-1}(1+\beta_t\Delta_t)$. Under $H_0$, $(W_t)$ is a non-negative supermartingale, and Ville's inequality $\mathbb{P}_{H_0}(\sup_t W_t\ge 1/\alpha)\le\alpha$ ensures that "crossing the wealth line" is a valid rejection rule at any time. The commitment time is $\tau^{v,c}=\inf\{t:W_t^{v,c}\ge 1/\alpha^{v,c}\}$. To avoid manual tuning—which would violate the test by "peeking"—the parameter-free Universal Portfolio is used, mixing all constant rebalancing portfolios with a $\mathrm{Beta}(1/2,1/2)$ Jeffreys prior:

$$\beta_t=\frac{\int_0^1 \beta\prod_{u}(1+\beta\Delta_u)\,dF_+(\beta)}{\int_0^1 \prod_u(1+\beta\Delta_u)\,dF_+(\beta)}.$$

UP achieves the optimal growth rate of the best constant portfolio in i.i.d. cases without tuning, making it the strongest tool in the SAVI framework that does not rely on independence.

**3. Confidence sequence variant + global $\alpha$-allocation: Adding "average advantage" semantics and lifetime error control**

While the strong null leads to earlier splits and better performance in practice, ensuring a monotonic decrease in model loss at the commit time requires "average advantage" semantics. Thus, **Ours** also proposes a weak null $H_{w,0}^{v,c}:\bar\delta_t^{v,c}=\frac{1}{t-s^v}\sum_u \delta_u^{v,c}\le 0$, constructed using the empirical Bernstein confidence sequence $(L_t,U_t)$, stopping at $\tau_w^{v,c}=\inf\{t:L_t>0\}$. For global control, the budget $\alpha$ is distributed across all candidates $(v,c)$: as long as $\sum_{v,c}\alpha^{v,c}\le\alpha$, the union bound plus anytime-validity yields $\mathbb{P}(\exists\text{false split ever})\le\alpha$, maintaining family-wise error for a "tree running for a lifetime."

### Loss & Training
Log loss is used for classification and squared loss for regression (with online monitoring of max loss to scale to $[0,1]$). Default hyperparameters are $n_{\min}=20$, $\varepsilon=0$, and $\alpha$ as the family-wise significance level. There are three theoretical guarantees: (i) Thm 4.1, global anytime validity; (ii) Thm 4.2, when a persistent advantage $\Delta>0$ exists, the commitment time is finite and high-probability bounded by $\tilde{\mathcal{O}}(\log(1/\alpha^{v,c})/\Delta^2)$; (iii) Thm 4.3, under i.i.d. + convex loss, the expected loss of the deployed model is non-increasing between and at commit times.

## Key Experimental Results

### Main Results
The authors ran 10 trials on 12 data streams (6 regression + 6 classification) using a prequential (test-then-train) protocol, comparing HT, ARF, and anytime-valid tree variants: AVT_B (single tree) / AVF_B (forest). AVT_B outperformed HT on most streams (except abalone) and neared or exceeded ARF on bike/fried/hyper100k; AVF_B achieved global best results on nearly all datasets. Regarding model size, AVF_B was shallower and had fewer nodes than the single-tree AVT_B; AVT_B was slightly deeper than HT because it must handle drift without bagging.

| Setting | Performance Lead | Key Phenomenon | Note |
|------|---------|---------|------|
| Single Tree AVT_B vs HT | Majority of 12 streams | Prediction curve is stable and monotonic; HT repeatedly "collapses" on i.i.d. RandomTree | abalone is a slight counterexample |
| Forest AVF_B vs ARF | Almost all streams | AVF_B is best with the shallowest trees | Gap widens as drift intensity increases |

### Ablation Study
A warm-up experiment (RandomTree, 10 numerical + 10 categorical features, depth 4) + time overhead comparison for AVT_B / AVT_CS / AVF_B vs HT/ARF show the trade-offs.

| Configuration | Key Metrics | Description |
|------|---------|------|
| HT (warm-up i.i.d.) | Multiple collapses + leaf count >> true value 8 | Failure of fixed sample inequality + optional stopping |
| AVT_B (betting) | Stable monotonic improvement, leaves ≤ 8 | Strictly follows Thm 4.3 monotonicity |
| AVT_CS (weak null + Bernstein CS) | Slightly more conservative than AVT_B | Affected by historical average, slightly worse prediction |
| Inference Latency AVT_B vs HT | 0.06–0.12 ms vs 0.03–0.12 ms | Same magnitude, pure tree traversal |
| Update Latency AVT_B vs HT | 1–25 ms vs 0.07–0.6 ms | UP wealth maintenance is the main overhead |
| Update Latency AVF_B vs ARF | Within 1 order of magnitude | Shallower trees offset costs; naturally parallelizable |

### Key Findings
- **Strong null is more practical than weak null**: Betting-based tests split earlier with lower loss; the weak null is conservative as it is dragged down by historical averages.
- **AVF_B's smaller size is a structural advantage**: Due to a stricter split criterion, each ensemble member only grows new nodes when there is a true persistent advantage, making the forest more compact.
- **Extra overhead comes from wealth maintenance, not inference**: Update time is increased by 1 order of magnitude by UP integration (approximated discretely), but remains at the millisecond level; candidate splits can be processed in parallel.

## Highlights & Insights
- **Porting SAVI to stream learning**: This is the first work to systematically graft testing-by-betting + UP + empirical Bernstein confidence sequences onto decision tree split criteria; prior work in the stream community mostly patched the Hoeffding/McDiarmid framework.
- **"Shadow challenger" is a reusable design pattern**: Keeping the incumbent fixed and training candidate models in parallel just for loss evaluation is a mechanism that can be migrated to online neural networks, streaming GBDT, etc.
- **Clean alignment between theory and engineering**: The monotonicity of Thm 4.3 is a direct consequence of "convex loss + plug-in leaf updates." The anytime-valid test ensures that "structural updates (splits) do not break monotonicity," allowing the method and theory to complement each other.

## Limitations & Future Work
- **Update cost is 10–100x that of a single tree**: UP requires numerical integration (discrete approximation), which is expensive as candidates increase. Real-time performance relies on parallelization (GPU/multicore).
- **Commit time monotonicity under strong null requires "continuous advantage" assumption**: To strictly guarantee $\mathbb{E}[\ell(\hat m_{\tau_k})]$ decreases at commit time, one must use the weak null + $\varepsilon$-shifted test; for the strong null, this is only "empirically observed."
- **Granularity of $\alpha$-allocation**: The global budget $\alpha$ is shared across all candidates, but the set $C_v$ can be large. How to allocate $\alpha^{v,c}$ without diluting detection power was not analyzed in depth beyond a schedule in the appendix.
- **Extensions beyond streams**: The method is naturally suited for streaming GBDT and in-stream pruning, but **Ours** does not cover them; non-binary splits and categorical features are only mentioned as "standard adaptations."

## Related Work & Insights
- **vs Hoeffding Tree / VFDT (Domingos & Hulten 2000)**: HT is directly challenged—fixed sample size + data-dependent stopping = invalid nominal guarantees. **Ours** replaces the criterion but keeps the "single-pass, incremental, leaf-split" skeleton.
- **vs McDiarmid-bound corrections (Rutkowski 2012; De Rosa & Cesa-Bianchi 2017)**: These simply substitute Hoeffding for tighter bounds but still assume a fixed $n$. This paper argues they fail to solve optional stopping.
- **vs UP (Orabona & Jun 2023) / Sequential Comparison (Choe & Ramdas 2024)**: **Ours** lands the latter's sequential forecaster comparison as a decision tree split criterion, highlighting the empirical difference between strong and weak nulls in tree learning.
- **vs Adaptive Random Forest (Gomes 2017/2018)**: **Ours** does not replace ARF's bagging/drift detection but shows that AVF_B exceeds ARF in both performance and size, suggesting ARF was hampered by the statistical invalidity of its base learners.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematically introduces SAVI/testing-by-betting to online trees with global error control and monotonicity.
- Experimental Thoroughness: ⭐⭐⭐⭐ 12 real streams + warm-up experiments + time costs; covers both single trees and forests.
- Writing Quality: ⭐⭐⭐⭐ Very clear problem statement; theorems correspond directly to algorithm pseudocode.
- Value: ⭐⭐⭐⭐ A significant contribution replacing the fundamental split criterion for stream learning; directly valuable for finance, IoT, and security.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Online Decision-Focused Learning](../../ICLR2026/learning_theory/online_decision-focused_learning.md)
- [\[ICLR 2026\] Online Decision Making with Generative Action Sets](../../ICLR2026/learning_theory/online_decision_making_with_generative_action_sets.md)
- [\[ICML 2026\] When Sample Selection Bias Precipitates Model Collapse](when_sample_selection_bias_precipitates_model_collapse.md)
- [\[ICML 2026\] Active Learning with Low-Rank Structure for Data Selection](active_learning_with_low-rank_structure_for_data_selection.md)
- [\[ICLR 2026\] Multi-Condition Conformal Selection](../../ICLR2026/learning_theory/multi-condition_conformal_selection.md)

</div>

<!-- RELATED:END -->
