---
title: >-
  [Paper Note] Prediction with Expert Advice under Local Differential Privacy
description: >-
  [ICLR 2026][learning_theory][Paper Note] This paper investigates the online learning problem of "Prediction with Expert Advice" under Local Differential Privacy (LDP) constraints. It first notes that the classic random walk algorithm RW-FTPL naturally satisfies LDP. Building on this, it introduces two improvements: RW-AdaBatch, which utilizes the "few switche
tags:
  - ICLR 2026
  - learning_theory
date: 2026-05-08
content_hash: 656494cb4de91e5e
---
# Prediction with Expert Advice under Local Differential Privacy

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=B9H2705C7c](https://openreview.net/forum?id=B9H2705C7c)  
**Code**: https://github.com/ben-jacobsen/dp-online-learning  
**Area**: Online Learning / Learning Theory / Differential Privacy  
**Keywords**: Prediction with expert advice, local differential privacy, random walk, privacy amplification, dynamic environments

## TL;DR
This paper investigates the online learning problem of "Prediction with Expert Advice" under Local Differential Privacy (LDP) constraints. It first notes that the classic random walk algorithm RW-FTPL naturally satisfies LDP. Building on this, it introduces two improvements: RW-AdaBatch, which utilizes the "few switches" property for adaptive batching to achieve privacy amplification with minimal utility loss when data is "simple"; and RW-Meta, which uses a shared-noise mechanism to privately select from a set of data-dependent learning experts at zero additional privacy cost. RW-Meta outperforms classic baselines and SOTA centralized DP algorithms by 1.5–3 times on real COVID-19 hospital data.

## Background & Motivation

**Background**: Prediction with Expert Advice is a classic framework in online learning. In each step, a player receives advice from $n$ experts and must decide which one to follow. Subsequently, the environment reveals a gain vector $g_t \in [0,1]^n$, and the player receives the reward corresponding to their chosen action. The goal is to maximize cumulative reward or, equivalently, minimize the static regret relative to the best fixed action in hindsight. When these expert suggestions involve sensitive human behavior (e.g., medical diagnosis, population mobility, energy consumption, public health), algorithms purely seeking accuracy may leak data through their outputs, necessitating differential privacy.

**Limitations of Prior Work**: Previous research has almost exclusively followed the "Centralized DP (CDP)" route, which assumes a trusted central curator collects raw data before adding noise. However, establishing a trusted curator is difficult in highly sensitive scenarios. Local DP (LDP)—where each gain vector is noisy at the source and the algorithm only sees the noisy sequence—is more suitable and easier to implement (requiring only independent random sampling without complex multi-round secure aggregation protocols). Nevertheless, expert algorithms under LDP are largely unexplored, with only one known related work (aimed at federated environments and stochastic adversaries with a different technical approach).

**Key Challenge**: First, LDP adds noise to every data point rather than the final analysis, meaning noise scales are naturally larger; direct implementation often results in a utility gap of $\sqrt{n}/\mu$ compared to non-private baselines. How can this "pointwise noise" be used more efficiently without compromising privacy? Second, in non-private literature, "experts" are often adaptive learning algorithms themselves. In private settings, "privately selecting the best expert" and "actually executing that expert's advice" are distinct—the latter leaks privacy if the expert itself is not DP-compliant. Consequently, prior work has been limited to data-independent "static experts," which is a severe restriction in dynamic environments.

**Key Insight**: The authors observe that the classic algorithm RW-FTPL (Devroye et al., 2013), which they build upon, possesses an overlooked property: it was originally designed for "few switches" (infrequently changing predictions). This "limited switching" implies that the algorithm's output is nearly invariant to the permutation of adjacent data points, a core requirement for privacy amplification in the shuffle model. The authors apply this insight in two directions: for adaptive batching (RW-AdaBatch) and for constructing a shared-noise mechanism that allows multiple data-dependent experts to "hide behind the same noise" (RW-Meta).

**Core Idea**: Reinterpret the "few switches / permutation invariance" of RW-FTPL from a utility property into a privacy resource. In batching, data points find "safety in numbers" to hide (amplifying privacy). In expert selection, all experts reuse the same noisy gain vector (post-processing invariance ⇒ zero additional privacy cost).

## Method

### Overall Architecture

The study centers on the classic algorithm **RW-FTPL** and extends it. RW-FTPL operates by sampling a Gaussian random walk noise $z_0,\dots,z_T \stackrel{iid}{\sim} \mathcal{N}(0,\eta^2 I_n)$ at the start and choosing $x_t=\arg\max_i (G_{t-1}+S_{t-1})_i$ at each step, where $G_t=\sum_{s=0}^t g_s$ is the cumulative true gain and $S_t=\sum_{s=0}^t z_s$ is the cumulative noise. Viewed differently, by defining noisy gains $\tilde g_t := g_t + z_t$, RW-FTPL is simply post-processing these vectors. Since each $\tilde g_t$ satisfies local Gaussian DP with privacy parameter $\mu=\Delta/\eta$ (where sensitivity $\Delta=\max_{g,g'}\lVert g-g'\rVert_2\le\sqrt{n}$), the expected static regret is at most $(\eta+\tfrac{2}{\eta})\sqrt{2T\log n}$.

Building on this: **RW-AdaBatch** (for static environments) maintains the output of RW-FTPL while adaptively batching data when predictions are likely to remain constant, allowing points in large batches to benefit from stronger privacy amplification. **RW-Meta** (for dynamic environments) upgrades "experts" to learning algorithms (learners) and reuses a single noisy gain vector across all learners, enabling private expert selection and execution at zero extra privacy cost. A comparison of privacy/regret follows ($T$ steps, $n$ experts, $m$ learners):

| Algorithm | Privacy Guarantee | Asymptotic Regret | Regret Definition |
|-----------|-------------------|-------------------|-------------------|
| Agarwal & Singh (2017) | $(\varepsilon,\delta)$-CDP | $\frac{\sqrt{T\log n}+\sqrt{n\log n}\,\log^{1.5}T}{\varepsilon}$ | Best static action |
| RW-FTPL (Devroye et al., 2013) | $\mu$-GLDP | $\frac{\sqrt{Tn\log n}}{\mu}$ | Best static action |
| **RW-AdaBatch** (param $\alpha>0$) | $\mu$-GLDP; amplified GCDP | $\frac{(1+\alpha)\sqrt{Tn\log n}}{\mu}$ | Best static action |
| **RW-Meta** ($m$ learners) | $\mu$-GLDP | $\frac{\sqrt{Tnm\log m}}{\mu}$ | Best learner |

### Key Designs

**1. RW-FTPL Naturally Satisfies LDP: Rewriting "Random Walk Perturbation" as Post-processing**

The first contribution is the observation that the baseline RW-FTPL algorithm is inherently an LDP algorithm. By viewing the decision rule through the lens of noisy gains $\tilde g_t=g_t+z_t$, the term $\arg\max_i(G_{t-1}+S_{t-1})_i$ becomes a post-processing of the sequence $\tilde g_t$. Since Gaussian noise is added to the gain vector scaled by sensitivity $\Delta\le\sqrt{n}$, each point satisfies $\mu$-Gaussian LDP where $\mu=\Delta/\eta$. By the post-processing property of DP, the entire algorithm sequence is $\mu$-GLDP. This bridges a utility-focused algorithm with a privacy framework, providing the backbone for subsequent designs where no additional noise is needed.

**2. RW-AdaBatch: Transforming "Few Switches" into Adaptive Batching for Data-Driven Privacy Amplification**

RW-AdaBatch translates the "infrequent switching" of RW-FTPL into many data-dependent intervals of strict permutation invariance. If there is a high probability that predictions will not change for several future steps, those steps are batched (buffered and delayed $\tilde G$ updates). Points in large batches benefit from "hiding in the crowd," achieving privacy amplification similar to the local-to-central amplification in shuffle models. The amplification strength naturally increases with the "simplicity" of the data: when a clear optimal expert exists (non-zero gap), predictions are harder to flip, leading to larger batches and stronger privacy.

The authors use a Gaussian random walk theorem to characterize the utility cost. **Theorem 1** states that if the gap at starting point $v$ is $k$, the probability of the leader changing in a $B$-step random walk is at most:
$$2\Phi(-\sqrt2\,\beta)+2\sqrt{\pi}\,\varphi(-\beta)\bigl(\Phi(\beta)-\Phi(-\beta)\bigr),\quad \beta=\frac{k}{\eta\sqrt{2B}}-\sqrt{\log(2n-2)}.$$
The `ComputeDelay` subroutine in **Corollary 1** uses this to select the maximum batch size $B_t$ such that the switching probability is bounded by $\alpha\sqrt{\log n/(t+B_t)}$. **Corollary 2** proves the expected regret is at most $(1+\tfrac{\alpha}{2})(\eta+\tfrac{2}{\eta})\sqrt{2T\log n}$—practically zero utility cost for small $\alpha$. For privacy, points in a batch of size $B$ receive ex-post CDP of $\mu/\sqrt B$.

**3. RW-Meta: De-correlated Shared Noise for Zero-Cost Private Selection of Data-Dependent Experts**

In dynamic environments, selecting the best learner is insufficient; one must execute their advice. If learners are data-dependent and non-DP, this execution leaks privacy. Instead of splitting the privacy budget among $m$ learners (which would increase error by $\sqrt m$), RW-Meta lets all learners **share the same** noisy gain vector $\tilde g$. Since each learner only accesses $\tilde g$, the entire process remains a post-processing of the LDP data, satisfying $\mu$-GLDP while keeping error constant relative to $m$.

Specifically, $\langle x_{t,i},\tilde g_t\rangle$ is an unbiased estimate of learner $i$'s gain, but the cumulative term $\sum_s X_t\tilde g_t\sim\mathcal N(G^{(m)}_t,\Sigma_t)$ has a data-dependent covariance $\Sigma_t=\eta^2\sum_s X_tX_t^\top$. The authors introduce a **de-correlation step**: let
$$\Sigma_t^*=\Sigma_t-\tfrac{1}{m^2}(\mathbf 1^\top\Sigma_t\mathbf 1)\mathbf 1\mathbf 1^\top,$$
and sample $y_t\sim\mathcal N\!\bigl(0,\max(2t,\lambda_{\max}(\Sigma^*_{t-1}))I-\Sigma^*_{t-1}\bigr)$ to select $j_t=\arg\max_j(\tilde G^{(m)}_{t-1}+y_t)_j$. **Theorem 3** shows the expected regret relative to the best learner is:
$$\Bigl[\max\!\bigl(\sqrt2,\ \eta\,\lambda_{\max}(\Sigma_T^*/(\eta^2T))^{1/2}\bigr)+\sqrt2\Bigr]\sqrt{2T\log m}=O\!\Bigl(\frac{\sqrt{Tnm\log m}}{\mu}\Bigr).$$

## Key Experimental Results

### Main Results

RW-Meta was evaluated on COVID-19 hospital reporting data from the US DHHS. The task was to predict which hospitals reported the highest density of COVID patients. 13 learners were used (rolling Gaussian regressions with varied windows/regularization + basic RW-FTPL) across three states (NM, PA, CA) over 148 weeks. Comparison was made against the SOTA CDP expert algorithm by Agarwal & Singh (2017).

| State / Privacy | RW-Meta (LDP) | Agarwal Min Noise (CDP) | Agarwal Min Regret (CDP) | Best Linear Model |
|-----------------|---------------|--------------------------|---------------------------|-------------------|
| NM, $\mu=1$     | **36.87**     | 22.83                    | 13.41                     | 40.69             |
| NM, $\mu=0.25$  | **22.46**     | 15.58                    | 13.38                     | 25.85             |
| PA, $\mu=1$     | **40.32**     | 23.55                    | 15.38                     | 44.30             |
| CA, $\mu=1$     | **55.62**     | 25.09                    | 16.91                     | 61.16             |
| CA, $\mu=0.25$  | **33.34**     | 17.93                    | 16.84                     | 38.43             |

RW-Meta outperformed other algorithms in all settings, leading the strongest CDP baseline by approximately 1.5–3x.

### Ablation Study

| Configuration | Key Finding |
|---------------|-------------|
| RW-Meta vs Static Experts | Achieves **negative static regret** in low/medium privacy settings, which static experts cannot do. |
| RW-Meta vs Best Linear | Performance is approximately **90%** of the theoretical best, despite the "best" model shifting across states. |
| RW-AdaBatch Amplification | The analytical bound is "reasonable but not perfect" compared to Monte Carlo simulations, tightest at small $\delta$. |

### Key Findings
- **Data-dependent experts are critical**: RW-Meta's ability to achieve negative static regret means it out-predicts any single fixed action.
- **LDP can outperform CDP**: In lower-dimensional regimes ($T\gtrsim n/\varepsilon^2$), the benefits of "dynamic environment + zero-cost expert selection" outweigh the higher noise floor of LDP.
- **RW-AdaBatch amplification is "free"**: Utility cost is a negligible multiplier ($1+\alpha/2$), while privacy strengthens automatically on "easy" data.

## Highlights & Insights
- **Utility properties as privacy resources**: RW-FTPL's "few switches" was designed for low regret, but this paper reclaims it for privacy amplification and noise sharing.
- **Shared noise = Zero-cost selection**: Reusing noisy gains across learners bypasses the $\sqrt{m}$ increase in error typically caused by privacy budget splitting.
- **Adaptive Privacy**: The intuition that "easier data (pronounced gaps) yields stronger privacy" is formalized in an online LDP setting with computable bounds.
- **Transferable de-correlation trick**: Converting data-dependent covariance to a scaled identity matrix by exploiting noise-invariance in specific subspaces is a technique that could apply elsewhere.

## Limitations & Future Work
- **Oblivious adversary**: Regret bounds assume gain vectors do not adapt to the specific realization of the noise.
- **Ex-ante amplification lacks closed-form**: The bound for RW-AdaBatch requires numerical calculation and is somewhat loose in medium FPR ranges.
- **Learner diversity**: RW-Meta's performance depends on the learners not being too highly correlated.
- **Limited evaluation task**: Only one real-world dataset (COVID-19) was used; comparison with concurrent dynamic algorithms like Saha et al. (2025) is a future step.

## Related Work & Insights
- **vs RW-FTPL (Devroye et al., 2013)**: The foundation. This paper identifies its LDP nature and upgrades it for amplification and meta-learning.
- **vs Agarwal & Singh (2017)**: SOTA centralized DP expert algorithm. RW-Meta outperforms it in practical regimes despite using the weaker LDP model.
- **vs Asi et al. (2023)**: Also leverages limited switching but focuses on high-dimensional regimes in the centralized model.
- **vs Saha et al. (2025)**: Concurrent work for dynamic environments but uses an exponential candidate expert set, leading to runtime that scales poorly compared to the linear runtime of RW-Meta.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Asymptotic Optimality of the High-Dimensional Gaussian Mechanism and Improved Low-Dimensional Mechanisms for Differential Privacy](../../ICML2026/learning_theory/asymptotic_optimality_of_the_high-dimensional_gaussian_mechanism_and_improved_lo.md)
- [\[ICLR 2026\] Why We Need New Benchmarks for Local Intrinsic Dimension Estimation](why_we_need_new_benchmarks_for_local_intrinsic_dimension_estimation.md)
- [\[ICLR 2026\] Singleton-Optimized Conformal Prediction](singleton-optimized_conformal_prediction.md)
- [\[ICLR 2026\] Online Conformal Prediction with Adversarial Semi-bandit Feedback via Regret Minimization](online_conformal_prediction_with_adversarial_semi-bandit_feedback_via_regret_min.md)
- [\[ICLR 2026\] Distribution-informed Online Conformal Prediction](distribution-informed_online_conformal_prediction.md)

</div>

<!-- RELATED:END -->
