---
title: >-
  [Paper Note] Optimal Bayesian Stopping for Efficient Inference of Consistent LLM Answers
description: >-
  [ICML 2026][LLM Efficiency][Adaptive Sampling] This paper models the "Self-Consistency (SC) majority voting via multiple sampling" problem as a Bayesian optimal stopping problem with prior information. It proposes an $L$…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "Adaptive Sampling"
  - "Self-Consistency"
  - "Bayesian Stopping"
  - "Sequential Hypothesis Testing"
  - "Test-time Scaling"
date: 2026-05-08
content_hash: 75a5ffd73177daf7
---

# Optimal Bayesian Stopping for Efficient Inference of Consistent LLM Answers

**Conference**: ICML 2026  
**arXiv**: [2602.05395](https://arxiv.org/abs/2602.05395)  
**Code**: https://github.com/jh9959-afk/Paper (Available)  
**Area**: LLM Efficiency / Test-time Compute / Adaptive Self-Consistency  
**Keywords**: Adaptive Sampling, Self-Consistency, Bayesian Stopping, Sequential Hypothesis Testing, Test-time Scaling  

## TL;DR
This paper models the "Self-Consistency (SC) majority voting via multiple sampling" problem as a Bayesian optimal stopping problem with prior information. It proposes an $L$-aggregated posterior approximation that tracks only three categories of counts: "most frequent / second most frequent / others." The authors theoretically prove that $L=3$ achieves an asymptotic optimal stopping time identical to the exact posterior as $\delta \to 0$. Experimentally, it saves 30%–80% of LLM calls on GSM8K and CommonsenseQA at approximately 1.4x the speed of ASC.

## Background & Motivation

**Background**: A dominant lightweight approach to improving LLM accuracy on mathematical and reasoning tasks is Self-Consistency (SC) — sampling multiple Chain-of-Thought (CoT) paths for the same question and deciding the final answer by majority vote. While this "sample-and-vote" method has become a standard recipe for test-time scaling, it incurs significant computational waste due to fixed budgets (e.g., 40 samples per question). To mitigate this, Adaptive Self-Consistency (ASC) by Aggarwal et al. (2023) uses a Beta posterior for stopping decisions under uninformative priors, stopping "once the leading answer is strong enough."

**Limitations of Prior Work**: Works like ASC ignore a clearly available signal: the shape of the answer distribution for a specific LLM on tasks with the same distribution is itself a learnable prior. Easy questions yield "peaked" distributions (max probability near 1), while hard questions are "flat"; these shapes can be estimated from historical responses, yet they are completely unused in the ASC Bayesian update framework.

**Key Challenge**: Directly plugging the "answer frequency vector $\pi$" as a prior into the Bayesian framework and calculating the posterior for "mode $a_1$ identified" leads to a combinatorial explosion. Because answer labels themselves are unobserved (only pairwise equality is visible), one must enumerate all injections $\psi \in \mathfrak{S}_{M(n)}$ from $K$ distinct answers to latent labels. The complexity of the exact posterior is $\mathcal{O}(K!)$, which becomes intractable for open-ended reasoning tasks where $K$ is large.

**Goal**: (i) Provide an optimal stopping rule that strictly outperforms ASC statistically and runs in real-time computationally under both known and uncertain prior settings; (ii) Compare its asymptotic stopping time rigorously with prior-independent lower bounds.

**Key Insight**: The authors compress observations into "frequencies of frequencies" $\mathcal{C}_n = \{(v_i, c_i)\}$ (e.g., $\{(10,1),(3,2),(2,1)\}$ denotes "one answer appearing 10 times, two answers appearing 3 times each, and one answer appearing 2 times"). They further retain only Top-$(L-1)$ frequencies and merge the rest into "others," resulting in an $L$-aggregated state $\mathcal{C}_n^L$. This reduces posterior complexity from $\mathcal{O}(K!)$ to $\mathcal{O}(K^L \cdot \bar n^2)$.

**Core Idea**: Approximate the Bayesian posterior using $L=3$ (tracking only Top-1, Top-2, and Others) — "three is all you need." This still yields an asymptotic stopping time identical to the exact posterior ($L=K$) as $\delta \to 0$, while inference latency remains nearly on par with $L=2$.

## Method

### Overall Architecture
- **Input**: An IID sequence of LLM samples $(a^{(t)})_{t\ge 1}$ for a new question, target confidence $1-\delta$, and answer frequency prior $\pi$ (known setting) or a set of candidate priors $\Pi^M = \{\pi^1, \dots, \pi^M\}$ with hyper-prior weights $\lambda_m$ (uncertain setting).
- **Observation Compression**: As each answer arrives, update the counts $\mathcal{C}_n$ and compress them into $\mathcal{C}_n^L$ following the "Top-$(L-1)$ + Others" rule.
- **Stopping Criterion**: Calculate the approximate posterior $\mathbb{P}(H_1 \mid \mathcal{C}_n^L)$ — the posterior probability that the most frequent answer is the true mode $a_1$. Stop if this exceeds $1-\delta$ and output the current majority.
- **Output**: Stopping step $n^{\star,L}$ and the current most frequent answer.

### Key Designs

1. **From Exact Posterior to $L$-Aggregated Compression**:
    - **Function**: Reduces the $\mathcal{O}(K!)$ posterior complexity to $\mathcal{O}(K^L)$ for real-time execution.
    - **Mechanism**: The exact posterior $\mathbb{P}(H_1 \mid \mathcal{C}_n) = \frac{\sum_{\psi: \psi(1)=1} \prod_j p_{\psi(j)}^{n_j}}{\sum_{\psi \in \mathfrak{S}_{M(n)}} \prod_j p_{\psi(j)}^{n_j}}$ requires summation over all $\psi$. The aggregated state explicitly preserves only the Top-$(L-1)$ frequencies. The multinomial allocation $\mathbf{r}^{-\psi}$ of the remaining $K-L+1$ answers is summarized via $\tilde S_\psi = \sum_{\mathbf{r}^{-\psi}} w(\mathbf{r}) \cdot \frac{\bar n!}{\prod r_j!} \prod p_j^{r_j}$, where weights $w(\mathbf{r}) = \binom{c_d' + m(\mathbf{r})}{c_d'}^{-1}$ correct for double-counting caused by arbitrary assignments of boundary frequencies. A critical identity $\mathbb{P}(H_1 \mid \mathcal{C}_n^L) = \mathbb{E}[\mathbb{P}(H_1 \mid \mathcal{C}_n) \mid \mathcal{C}_n^L]$ ensures the aggregated posterior remains an unbiased coarsening of the exact posterior, introducing no systematic bias.
    - **Design Motivation**: The bottleneck of the exact posterior is $K!$, not the sample size. By fully retaining "head signals" and summarizing the tail with a statistic $\bar n_{L(n)}$, one preserves discriminative power while reducing complexity to an exponential in $L$. This serves as a key switch for the trade-off between statistical and computational costs.

2. **"Three is Enough" Asymptotic Optimality Theorem**:
    - **Function**: Rigorously characterizes the relationship between $L$ and stopping time, proving $L=3$ is the sweet spot.
    - **Mechanism**: In the $\delta \to 0$ limit, the asymptotic stopping rate for $L=2$ is $\lim \mathbb{E}[n^{\star,2}] / \log(1/\delta) = 1/D_{\mathrm{KL}}(p_1 \| p_2)$. For all $L \ge 3$, $\lim \mathbb{E}[n^{\star,L}] / \log(1/\delta) = 1 / ((p_1 - p_2) \log(p_1/p_2))$, matching the exact posterior $L=K$. Compared to the prior-free baseline $n^{\star,f}$ (whose rate denominator is given by the symmetrized Jensen-Shannon divergence corresponding to the PPR martingale confidence sequence lower bound in Shah et al. 2020 / Jain et al. 2022), it holds that $\mathbb{E}[n^{\star,f}] > \mathbb{E}[n^{\star,2}] > \mathbb{E}[n^{\star,3}] = \cdots = \mathbb{E}[n^{\star,K}]$. Under uncertain priors, $L=3$ remains strictly superior to ASC unless a candidate prior $\pi^{m^\dagger}$ satisfies $p_{1,m^\dagger} \approx p_{2,m^\dagger} \approx (p_{1,m^\star}+p_{2,m^\star})/2$.
    - **Design Motivation**: Intuitively, the "Top-1 frequency" measures "leader strength," while the "Top-1 and Top-2 difference" measures the "margin over the runner-up." Capturing these two statistics recovers the essence of Bayesian evidence. Further refinement (e.g., $L=4, 5$) provides redundant information. $L=2$ is slower because observing only Top-1 degrades the comparison to a Bernoulli hypothesis test, losing information about the magnitude of the second-largest answer.

3. **Hierarchical Bayesian Extension and Offline Learning for Uncertain Priors**:
    - **Function**: Relaxes "known $\pi$" to "$\pi$ is randomly drawn from a candidate set $\Pi^M$ with probability $\lambda_m$," making the method applicable in real-world scenarios without per-question priors.
    - **Mechanism**: Generalizes $\mathbb{P}(H_1 \mid \mathcal{C}_n^L)$ into a mixture weighted by $\lambda_m$: $\mathbb{P}_{\Pi^M}(H_1 \mid \mathcal{C}_n^L) = \frac{\sum_m \lambda_m \sum_{\psi:\psi(1)=1} (\prod p_{\psi(j),m}^{n_j}) \tilde S_\psi^m}{\sum_m \lambda_m \sum_\psi (\prod p_{\psi(j),m}^{n_j}) \tilde S_\psi^m}$. Complexity increases only by a factor of $M$. $\Pi^M$ is constructed by a 70/30 split of the dataset, using the empirical distributions from 40 LLM samples per training question, with $\lambda_m$ set as a uniform distribution.
    - **Design Motivation**: "True per-question priors" are inaccessible in practice. An effective substitute is a library of answer distribution shapes from historical questions for the same LLM, as shapes for hard/easy/multiple-choice/open-ended questions are stable at the distributional level. Even if $\Pi^M$ does not contain the ground truth, the mixture posterior maintains an asymptotic lower bound strictly better than prior-free ASC.

### Loss & Training
No training loss is used. At the algorithmic level: (i) Algorithm 1 (Full version in Appendix C) maintains the dynamic programming calculation for $\mathcal{C}_n \to \mathcal{C}_n^L$ and $\tilde S_\psi$; (ii) $L=3$ is used as the default configuration; (iii) The threshold confidence $1-\delta$ is specified by the user based on the desired mode estimation accuracy (values used in experiments: $1-\delta \in \{0.7, 0.8, 0.9, 0.95, 0.975, 0.99\}$).

## Key Experimental Results

### Main Results

Comparison of $L$-aggregation, exact posterior, and prior-free ASC on a synthetic dataset ($\pi = (0.5, 0.2, 0.1, 0.1, 0.05, 0.03, 0.01, 0.01)$, $K=8$, 10,000 trials) at $1-\delta = 0.99$:

| Method | Mode Accuracy | Avg. Samples | Posterior Latency (ms) |
|------|-----------|------------|-------------------|
| $L=2$ | 99.5% | 22.43 | 9.0 |
| $L=3$ | 99.2% | 18.07 | 14.2 |
| $L=4$ | 99.2% | 18.11 | 37.4 |
| Exact ($L=K=8$) | 99.2% | 18.13 | 29.8 |
| ASC (Prior-free) | 100.0% | 44.07 | — |

$L=3$ reduces the sample count from 44.07 (ASC) to 18.07 (**~59% saving**), while the posterior calculation latency is only ~5 ms more than $L=2$ and ~2.6× faster than $L=4$.

On the real-world dataset CommonsenseQA (FEval-TTC) with Qwen-2.5-72B and $1-\delta = 0.99$:

| Method | Answer Accuracy | Mode Accuracy | Avg. Samples |
|------|-----------|------------|-----------|
| $L=3$ (Known Prior) | 88.0% | 99.4% | 4.24 |
| $L=3$ (Uncertain Prior) | 88.1% | 99.5% | 6.23 |
| ASC | 87.6% | 100.0% | 8.04 |

Under uncertain priors, it still saves ~22% in calls compared to ASC with comparable accuracy.

### Ablation Study

| Configuration | Key Finding |
|------|---------|
| $L = 2$ vs $L = 3$ | With known priors, $L=2$ requires 24% more samples at $1-\delta=0.99$ (22.43 vs 18.07). With uncertain priors, $L=2$ degrades further and can theoretically perform worse than ASC. |
| $L = 3$ vs $L = K$ (Exact) | Asymptotically equivalent. Sample size differences across all $\delta$ are $\le 0.1$, but $L=3$ is ~2× faster. |
| Known vs Uncertain Prior | $L=3$ with uncertain priors requires ~50% more samples on Qwen-2.5-72B (6.23 vs 4.24) but remains superior to ASC. |
| $1-\delta = 0.95$ Extremes | With uncertain priors, $L=3$ on GPT-4o mini often stops after the first sample, **saving ~80% of calls** with 96.9% mode accuracy. |

## Highlights & Insights
- **Sequential Hypothesis Testing for LLM Self-Consistency**: Unlike previous mode identification literature (Shah et al. 2020; Jain et al. 2022) which assumes no priors because "knowing answer probabilities leaks the mode," this work defines the prior on the distribution of "Top-1/Top-2/..." frequencies rather than specific answers. This avoids leakage and opens a new class of solvable Bayesian mode identification problems.
- **"Three is Enough" as a Clean Asymptotic Phase Transition**: The jump from $L=2$ to $L=3$ is qualitative (moving from KL divergence to the exact JS-style lower bound), whereas $L=3$ to $L=K$ is only quantitative (they are asymptotically identical). This "low-dimensional sufficient statistic" concept can be transferred to other online posterior coarsening scenarios like best-arm identification or A/B test stopping.
- **Engineering of the Uncertain Prior "Candidate Pool"**: Using empirical distributions from a the training set (40 samples per question) as a library requires no parametric modeling. This makes it highly practical for deployment without needing new model training.

## Limitations & Future Work
- **Cold Start Problem**: In new domains or tasks with zero historical data, the method reverts to ASC with no gain.
- **Robustness under Prior Bias**: Theorem 4.1 holds if $\Pi^M$ contains the true $\pi^{m^\star}$. Asymptotic bounds for systematically biased candidate pools (e.g., training on hard tasks but testing on easy ones) are not provided.
- **Mode $\neq$ Correct Answer**: The observation that "Top-2 is sometimes the true answer" on CommonsenseQA suggests that mode identification might not be the optimal objective. Coupling the framework with ground-truth feedback to identify the "correct answer" rather than the "mode" is a promising direction.
- **Scalability of $K$**: While $\mathcal{O}(K^3)$ is better than $\mathcal{O}(K!)$, it may still be heavy if $K$ reaches the hundreds in long open-ended generation.

## Related Work & Insights
- **vs ASC (Aggarwal et al. 2023)**: ASC uses a Beta (uninformative) posterior. This work uses a multinomial posterior under arbitrary priors, providing strict Pareto improvements (fewer samples, better calibration) at the cost of requiring an offline prior pool.
- **vs Shah et al. 2020 / Jain et al. 2022 (PAC Mode Identification)**: These established the asymptotic optimality of prior-free PPR martingale stopping. This paper proves one can strictly beat that lower bound by introducing priors.
- **vs MCMC / Normal Approximation (Gelman et al. 1995)**: General Bayesian approximations are computationally unsuitable for real-time inference and discrete count structures. $L$-aggregation + DP offers a tailored lightweight alternative.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The $L=3$ aggregation and the "Three is Enough" theorem are elegant, reducing a complex combinatorial Bayesian problem to a real-time algorithm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers synthetic and FEval-TTC datasets with multiple LLMs and confidence levels across both known and uncertain prior settings. Could benefit from a mis-specified prior adversarial experiment.
- **Writing Quality**: ⭐⭐⭐⭐ Rigorous derivations and clear motivation; provides intuition for technical theorems.
- **Value**: ⭐⭐⭐⭐ Directly saves 30–80% of compute for SC-based test-time scaling. Highly practical for deployment with open-sourced code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference](obcache_optimal_brain_kv_cache_pruning_for_efficient_long-context_llm_inference.md)
- [\[ICML 2026\] Training-Inference Consistent Segmented Execution for Long-Context LLMs](training-inference_consistent_segmented_execution_for_long-context_llms.md)
- [\[ICML 2026\] Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving](theoretically_optimal_attentionffn_ratios_in_disaggregated_llm_serving.md)
- [\[ICML 2026\] Ekka: Automated Diagnosis of Silent Errors in LLM Inference](ekka_automated_diagnosis_of_silent_errors_in_llm_inference.md)
- [\[ICML 2026\] Variational Routing: A Scalable Bayesian Framework for Calibrated MoE Transformers](variational_routing_a_scalable_bayesian_framework_for_calibrated_mixture-of-expe.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference](obcache_optimal_brain_kv_cache_pruning_for_efficient_long-context_llm_inference.md)
- [\[ICML 2026\] Training-Inference Consistent Segmented Execution for Long-Context LLMs](training-inference_consistent_segmented_execution_for_long-context_llms.md)
- [\[ICML 2026\] DOT-MoE: 用可微 optimal transport 把 dense LLM 转成 MoE](dot-moe_differentiable_optimal_transport_for_moefication.md)
- [\[ICML 2026\] Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving](theoretically_optimal_attentionffn_ratios_in_disaggregated_llm_serving.md)
- [\[ICML 2026\] Ekka: Automated Diagnosis of Silent Errors in LLM Inference](ekka_automated_diagnosis_of_silent_errors_in_llm_inference.md)

</div>

<!-- RELATED:END -->
