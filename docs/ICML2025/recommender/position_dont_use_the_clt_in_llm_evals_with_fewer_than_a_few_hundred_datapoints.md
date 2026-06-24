---
title: >-
  [Paper Note] Position: Don't Use the CLT in LLM Evals with Fewer Than a Few Hundred Datapoints
description: >-
  [ICML 2025 (Spotlight Position Paper)][Recommender Systems][Central Limit Theorem (CLT)] As a position paper, this work argues that when the sample size for LLM evaluation is fewer than a few hundred, confidence intervals based on the Central Limit Theorem (CLT) severely underestimate uncertainty. It recommends using Bayesian credible intervals or Wilson score intervals as alternative solutions.
tags:
  - "ICML 2025 (Spotlight Position Paper)"
  - "Recommender Systems"
  - "Central Limit Theorem (CLT)"
  - "Confidence Interval"
  - "Bayesian Credible Interval"
  - "LLM Evaluation"
  - "Small-sample Statistics"
  - "Wilson Score Interval"
  - "Bootstrap"
date: 2026-05-08
content_hash: 44878e87700dab35
---

tags:
  - ICML 2025
  - Recommender Systems
date: 2026-05-08
content_hash: 0c9fe6ee5402322a
---
# Position: Don't Use the CLT in LLM Evals with Fewer Than a Few Hundred Datapoints

**Conference**: ICML 2025 (Spotlight Position Paper)

**arXiv**: [2503.01747](https://arxiv.org/abs/2503.01747)

**Authors**: Sam Bowyer, Laurence Aitchison, Desi R. Ivanova

**Area**: Recommendation Systems / LLM Evaluation / Statistical Inference

**Keywords**: Central Limit Theorem (CLT), Confidence Interval, Bayesian Credible Interval, LLM Evaluation, Small-sample Statistics, Wilson Score Interval, Bootstrap

**Code**: [bayes_evals](https://github.com/sambowyer/bayes_evals)

---

## TL;DR

As a position paper, this work argues that when the sample size for LLM evaluation is fewer than a few hundred, confidence intervals based on the Central Limit Theorem (CLT) severely underestimate uncertainty. It recommends using Bayesian credible intervals or Wilson score intervals as alternative solutions.

---

## Background & Motivation

### Core Problem

Current statistical uncertainty quantification in LLM evaluations (evals) relies almost entirely on the CLT, but an increasing number of high-value benchmarks have very small data sizes:

- **FrontierMath**: ~300 questions, with some categories having fewer than 3 samples
- **AIME 2024**: Only 15 competition math problems
- **SWE-Bench Verified**: 500 samples
- **MLE-Bench**: 75 Kaggle competitions
- **LiveBench**: An average of only 55 samples per task
- **CUAD**: 510 legal documents, costing approximately 2 million USD

The scale of these professional benchmarks is far smaller than traditional large-scale benchmarks like MMLU and GSM8K (which range from thousands to tens of thousands of samples).

### Motivation

The CLT relies on asymptotic properties as $N \to \infty$. In small-sample scenarios, the confidence interval may:

1. When $\hat{\theta} = 0$ or $\hat{\theta} = 1$, the standard error becomes zero, and the interval collapses to a single point
2. The interval boundaries exceed the $[0,1]$ range
3. The actual coverage rate is far lower than the nominal coverage rate (e.g., set at 95% but actually far below 95%)

---

## Method

### Overall Architecture

This work systematically evaluates the failure modes of the CLT in **5 experimental scenarios** and compares Bayesian and frequentist alternatives:

| Scenario | Is CLT Applicable? | Recommended Method |
|------|-------------|---------|
| Single-model IID problems | Fails under small samples | Wilson score interval / Beta-Bernoulli Bayesian |
| Clustered problems | Fails under small samples | Bayesian importance sampling (Beta-Binomial) |
| Independent model comparison | Fails under small samples | Bayesian posterior sampling |
| Paired model comparison | Fails under small samples | Paired Bayesian importance sampling |
| Non-mean metrics (e.g., $F_1$) | Unusable | Bayesian Dirichlet-Categorical |

### Key Designs

#### 1. Bayesian Methods under the IID Setting

For Bernoulli data $y_i \sim \text{Bernoulli}(\theta)$, using a conjugate prior:

$$\theta | y_{1:N} \sim \text{Beta}\left(1 + \sum_{i=1}^{N} y_i, \; 1 + \sum_{i=1}^{N}(1-y_i)\right)$$

The exact credible interval is obtained directly from the Beta distribution quantiles, avoiding the need for asymptotic approximations.

#### 2. Generative Model under the Clustered Setting

$$d \sim \text{Gamma}(1,1), \quad \theta \sim \text{Beta}(1,1)$$
$$\theta_t \sim \text{Beta}(d\theta, d(1-\theta)), \quad y_{i,t} \sim \text{Bernoulli}(\theta_t)$$

where $d$ controls the difference in difficulty between task clusters, and $\theta_t$ represents the performance of each task/cluster. Integrating out $\theta_t$ yields:

$$Y_t \sim \text{BetaBin}(N_t, d\theta, d(1-\theta))$$

Inference is performed using importance sampling ($K=10000$) with the prior as the proposal distribution.

#### 3. Wilson Score Interval

$$\text{CI}_{1-\alpha}(\theta) = \frac{\hat{\theta} + \frac{z_{\alpha/2}^2}{2N}}{1 + \frac{z_{\alpha/2}^2}{N}} \pm \frac{z_{\alpha/2}}{2N\left(1 + \frac{z_{\alpha/2}^2}{N}\right)} \sqrt{4N\hat{\theta}(1-\hat{\theta}) + z_{\alpha/2}^2}$$

The center of the interval is no longer the sample mean $\hat{\theta}$, which avoids the issues of zero width and out-of-bound values.

### Loss & Training

This paper does not involve training losses; instead, it defines an experimental framework to evaluate coverage rates:

- For 100 $\theta$ values $\times$ 200 datasets $\times$ $N \in \{3, 10, 30, 100\}$
- Construct intervals for 100 nominal coverage levels for each dataset
- Calculate **coverage error** = average $|\text{actual coverage rate} - \text{nominal coverage rate}|$

---

## Key Experimental Results

### Main Results: Coverage Error across Scenarios

| Method | IID (N=10) | IID (N=100) | Clustered (N=10) | Paired (N=10) |
|------|-----------|------------|------------|------------|
| CLT | Coverage far below nominal | Close to nominal | Severe under-coverage | Severe under-coverage |
| Bootstrap (K=10000) | Coverage below nominal | Close to nominal | Under-coverage | Under-coverage |
| Wilson Score | Close to nominal ✓ | Close to nominal ✓ | N/A (IID only) | N/A |
| Clopper-Pearson | Overly conservative (too wide) | Close to nominal | N/A (IID only) | N/A |
| **Bayesian (Ours recommended)** | **Close to nominal ✓** | **Close to nominal ✓** | **Close to nominal ✓** | **Close to nominal ✓** |

### LangChain Tool Use Benchmark Empirical Testing (N=20)

| Model | CLT Interval Issues | Bayesian Interval |
|------|-------------|----------|
| GPT-4 | Interval exceeds [0,1] | Reasonable interval |
| Llama-2-70B | Interval collapses to zero | Reasonable interval |
| Mistral-7B | Interval exceeds [0,1] | Reasonable interval |
| GPT-3.5 | Severely underestimates uncertainty | Reasonable interval |

### Key Findings

1. **CLT systematically fails when N<100**: The actual coverage rate is far below the set nominal coverage rate.
2. **Bootstrap is similarly unreliable**: Even with K=10000 resamples, the coverage rate under small sample sizes remains insufficient.
3. **Wilson score interval** performs exceptionally well in the single-model IID scenario and is directly implemented in SciPy.
4. **Bayesian methods** are the only approach that achieves correct coverage rates across all scenarios.
5. **Non-linear metrics like $F_1$**: CLT is completely unusable, and only Bayesian methods are effective.
6. **Robustness to prior mismatch**: Bayesian methods still outperform CLT even under prior bias.

---

## Highlights & Insights

1. **Highly practical**: Python code snippets (3-10 lines) are provided for each scenario for direct reuse, making Bayesian methods extremely low-cost to implement.
2. **Relationship between Clopper-Pearson and Bayesian methods**: The CP exact interval is equivalent to a Bayesian credible interval with the uniform prior removed, explaining its over-conservativeness.
3. **Advantages of Bayesian model comparison**: It allows direct calculation of $\mathbb{P}(\theta_A > \theta_B | \text{data})$, a probability that frequentist methods cannot easily provide.
4. **Paired vs. unpaired**: The paired Bayesian method is more robust than the unpaired version when priors do not match.
5. **Negligible computational overhead**: The computational cost of Bayesian inference is trivial compared to the cost of constructing benchmarks and running LLM evaluations.

---

## Limitations & Future Work

1. **Position paper**: No new algorithms are proposed; the focus is mainly on methodological recommendations and systematic empirical validation.
2. **Only considers Bernoulli-type evaluations**: Applicability to continuous-valued metrics (e.g., BLEU, ROUGE) is not discussed.
3. **Prior selection**: Although robustness is verified, no systematic guidance for prior selection in specific scenarios is provided.
4. **Mainly simulated data**: Verification on real LLM evaluation scenarios is limited to a single LangChain case study.

---

## Related Work & Insights

- **Miller (2024)** proposes using CLT + clustered standard errors in LLM eval; Ours proves that this approach still fails under small sample sizes.
- **Madaan et al. (2024)** quantify variance in evaluation benchmarks but still utilize the CLT.
- **Dubey et al. (2024)** (Llama 3 report) acknowledge that the CLT is not suitable for metrics such as $F$-score, and thus completely omit reporting confidence intervals.
- **Insight**: In recommendation system evaluation, when A/B testing data scale is limited, Bayesian methods can provide more reliable statistical inference.

---

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 3 | The methods themselves are not new (Beta-Bernoulli and Wilson are classic methods); the contribution lies in the systematic empirical investigation. |
| Value | 5 | Provides directly reusable code, addressing a real pain point in LLM evaluation. |
| Experimental Thoroughness | 5 | Covers 5 scenarios × multiple sample sizes × various priors, with an extremely comprehensive ablation study (39 figures). |
| Writing Quality | 5 | Well-structured, strongly argued, blending code with mathematical derivations. |
| **Overall** | **4.5** | Makes significant methodological contributions to the LLM evaluation community and is highly recommended for wide adoption. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Position: The Right to AI](the_right_to_ai.md)
- [\[NeurIPS 2025\] Position: Towards Bidirectional Human-AI Alignment](../../NeurIPS2025/recommender/position_towards_bidirectional_human-ai_alignment.md)
- [\[ICLR 2026\] More Than What Was Chosen: LLM-based Explainable Recommendation Beyond Noisy User Preferences](../../ICLR2026/recommender/more_than_what_was_chosen_llm-based_explainable_recommendation_beyond_noisy_user.md)
- [\[AAAI 2026\] Preference is More Than Comparisons: Rethinking Dueling Bandits with Augmented Human Feedback](../../AAAI2026/recommender/preference_is_more_than_comparisons_rethinking_dueling_bandits_with_augmented_hu.md)
- [\[ICML 2025\] RLTHF: Targeted Human Feedback for LLM Alignment](rlthf_targeted_human_feedback_for_llm_alignment.md)

</div>

<!-- RELATED:END -->
