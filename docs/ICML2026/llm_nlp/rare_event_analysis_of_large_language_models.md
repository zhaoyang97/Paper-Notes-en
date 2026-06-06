---
title: >-
  [Paper Note] Rare Event Analysis of Large Language Models
description: >-
  [ICML2026][LLM/NLP][Rare event analysis] This paper transfers mature Rare Event Analysis (REA) methods from statistical physics to LLMs. Using a toolkit of "exponential tilting + Transition Path Sampling + MBAR…
tags:
  - "ICML2026"
  - "LLM/NLP"
  - "Rare event analysis"
  - "importance sampling"
  - "MBAR"
  - "Transition Path Sampling"
  - "LLM safety"
date: 2026-05-08
content_hash: 94b01fe95015f89d
---

# Rare Event Analysis of Large Language Models

**Conference**: ICML2026  
**arXiv**: [2602.06791](https://arxiv.org/abs/2602.06791)  
**Code**: Yes (minimal implementation provided in paper appendix)  
**Area**: LLM Analysis / Rare Event Sampling / Statistical Physics Methods  
**Keywords**: Rare event analysis, importance sampling, MBAR, Transition Path Sampling, LLM safety

## TL;DR
This paper transfers mature Rare Event Analysis (REA) methods from statistical physics to LLMs. Using a toolkit of "exponential tilting + Transition Path Sampling + MBAR," the authors estimate rare completion probabilities on TinyStories that are several orders of magnitude smaller than those achievable by direct sampling, using affordable compute. Through EDA, they identify cheap runtime proxies (consecutive token repeats) to pre-screen high-ARI anomalous outputs.

## Background & Motivation
**Background**: LLMs are probabilistic models. As deployment scale increases, "events nearly unseen during training or testing" occur with non-negligible frequency online, such as harmful outputs suppressed to the distribution tails after alignment. Quantitative analysis of such "tail behaviors" is in its infancy: prior work either focuses on single-token probabilities (Wu & Hilton 2025) or extrapolates from a few test prompts to the deployment distribution (Jones et al. 2025).

**Limitations of Prior Work**: The default "direct sampling" method (i.e., temperature=1 autoregressive sampling) is extremely inefficient at the tails—observing a $10^{-9}$ event requires generating approximately $10^9$ completions on average. While feasible for small models, the cost for production-grade LLMs is staggering. Furthermore, many histogram bins remain at zero, precluding even basic point estimation.

**Key Challenge**: Rare events are "unsamplable" by definition, yet they are the most critical components for safety, compliance, and OOD behavior analysis. Systematically characterizing the tails of LLM output distributions without exhaustive compute requires specialized rare event sampling methods.

**Goal**: To build a deployable end-to-end framework for Rare Event Analysis (REA) in LLMs, divided into three stages: (1) Setup: Formalizing the LLM as a stochastic process and rare events as extreme values of observables; (2) Estimation: Estimating rare event probabilities; (3) Exploration: Analyzing the structure and properties of rare completions.

**Key Insight**: Decades of rare event tools from molecular dynamics and statistical physics (umbrella sampling, TPS, MBAR, bootstrap CI) are naturally suited for the "autoregressive sequence + scalar observable" setting. One simply needs to replace "particle trajectories" with "token trajectories."

**Core Idea**: Use exponentially tilted distributions $p_{\lambda}(\mathbf{x}) \propto e^{-\lambda \phi(\mathbf{x})} p_{\mathcal{M}}(\mathbf{x})$ to push sampling toward the tails, employ TPS-MCMC to traverse the sequence space, and utilize MBAR to combine biased samples from multiple $\lambda$ values back into probability estimates for the original distribution, providing bootstrap confidence intervals for each step.

## Method

### Overall Architecture
The framework treats the LLM as a stochastic process generating token sequences $\mathbf{x}_{1:T}$ and focuses on the probability of a scalar observable $\phi(\mathbf{x}_{1:T})$ taking extreme values. The pipeline is as follows:

1.  **Define Observables**: Select the target observable (the metric of interest) and the biasing observable (the metric used to guide sampling). In this study, both are identical—specifically, the Automated Readability Index (ARI) and the log-probability of the completion.
2.  **Construct Biased Distribution Family**: Select $K$ different values of $\lambda_k$, where each $\lambda_k$ corresponds to a tilted distribution $p_{\lambda_k}$ covering different regions of the original distribution's tail.
3.  **TPS Sampling**: Run MCMC chains using Transition Path Sampling under each $\lambda_k$, utilizing an annealing schedule to gradually increase $|\lambda|$ for improved convergence.
4.  **MBAR Inference**: aggregate samples from all $\lambda_k$ and solve $K$-component self-consistent equations to estimate normalization constants $Z(\lambda_k)$, thereby recovering the probability density under the original distribution $p_{\mathcal{M}}$.
5.  **Error & Diagnostics**: Discard the first 10% as burn-in, reject segments with Gelman-Rubin statistics $\ge 1.1$, and perform 100 bootstrap resamplings to obtain 95% CIs.
6.  **EDA Exploration**: Perform scatter and histogram analysis on the biased rare completions to identify cheap proxy metrics.

### Key Designs

1.  **Exponential Tilting + Umbrella Sampling + MBAR**:
    - **Function**: Combines multiple sets of "tail-biased" samples into an unbiased estimate of the original distribution.
    - **Mechanism**: For each $\lambda_k$, a biased PMF is defined as $p_{\lambda_k}(\mathbf{x}) = Z(\lambda_k)^{-1} e^{-\lambda_k \phi(\mathbf{x})} p_{\mathcal{M}}(\mathbf{x})$. The target expectation is expressed via mixture importance sampling: $\bar f = \sum_k \alpha_k \mathbb{E}_{p_{\lambda_k}}[w_{\text{Mix}} f]$. The weight $w_{\text{Mix}}(\mathbf{x}) = 1 / \sum_j \alpha_j Z(\lambda_j)^{-1} e^{-\lambda_j \phi(\mathbf{x})}$ effectively cancels out $p_{\mathcal{M}}$, meaning **the model's normalization constant for full sequences is not required**; only token-level log-probs are needed. The $K$ unknown $Z(\lambda_j)$ values are solved via MBAR equations with optimal weights $\alpha_k = N_k^{-1}$.
    - **Design Motivation**: Direct sampling has zero samples at the tail, causing variance to explode. Tilting shifts probability mass to rare regions, and MBAR merges multiple biased chains to maximize information reuse, providing globally consistent estimates across all bins.

2.  **Transition Path Sampling (TPS) + Annealing**:
    - **Function**: Constructs a Markov chain in the sequence space with $p_{\lambda_k}$ as the invariant distribution, modifying only the "tail" of completions at each step.
    - **Mechanism**: At step $i$ with trajectory $\mathbf{x}^{(i)}_{1:T}$, a random truncation point $\tau \in [1, T)$ is chosen. $x_{1:\tau-1}$ is kept, while $x_{\tau:T}$ is resampled autoregressively using the LLM to generate a candidate $\tilde{\mathbf{x}}$. Acceptance is determined by the Metropolis-Hastings rate (governed by $p_{\lambda_k}$). Annealing gradually increases the bias across 10 levels of $\lambda$ to allow the chain to transition from "typical" to "extreme" without getting stuck.
    - **Design Motivation**: For long sequences, independent resampling is almost always rejected (acceptance rates decay exponentially with length). TPS achieves $O(1)$ acceptance rates by retaining prefixes. Annealing solves the initialization problem where starting too far from the target distribution leads to excessive burn-in.

3.  **Cheap Proxy Discovery (EDA)**:
    - **Function**: Replaces expensive target observables (requiring full completions) with cheap proxies calculable in real-time for filtering rare adverse outputs.
    - **Mechanism**: Samples are pushed to the high-ARI extreme using large $\lambda$. Scatter plots of ARI vs. Log-Prob are analyzed and colored by $\text{Repeats}(\mathbf{x}) = \sum_t \mathbb{I}[x_{t+1}=x_t]$. For TinyStories, extreme ARI outputs often contain repeating tokens ("Trurururu..."). The repeat count is an $O(T)$ cheap statistic that can be calculated incrementally.
    - **Design Motivation**: Many metrics of interest (readability, toxicity, factuality) require full text or external models, which is too slow for deployment. Identifying cheap proxies that correlate strongly with the target in rare sub-distributions allows for early abortion of high-risk generations.

### Loss & Training
This work does not train models but performs sampling analysis on pre-trained TinyStories-8M. Key MCMC hyperparameters: $K=10$ levels of $\lambda$, $4 \times 10^4$ TPS steps per level, 10% burn-in, GR threshold 1.1, 100x bootstrap. ARI is truncated at 15 to prevent MCMC acceptance collapse from rare high-ARI-high-LogProb completions.

## Key Experimental Results

### Main Results: Tail Coverage and Sample Efficiency

Model: TinyStories-8M, fixed 16-token prompt, 100-token completion; total token budget approx. $4 \times 10^8$.

| Method | Total Completions | ARI / Log-Prob Tail Coverage | Histogram Tail Bin Counts |
| :--- | :--- | :--- | :--- |
| Direct Sampling (SOTA) | $4.1 \times 10^6$ (100 tokens each) | Limited to training data range | High frequency of zero counts |
| TPS + MBAR (Ours) | $\approx 7 \times 10^6$ effective / $8 \times 10^6$ gen | Far exceeds training range; estimates orders of magnitude smaller probabilities | Non-zero across entire range |

### Ablation Study (Error Analysis)

| Metric | Direct Sampling | MBAR | Description |
| :--- | :--- | :--- | :--- |
| Relative CI width (typical) | Small | Comparable | Both methods perform similarly in the middle |
| Relative CI width (tail) | Massive (many bins at 0) | Significantly smaller | MBAR CI is orders of magnitude narrower at the tail |
| Bin variance after doubling steps | — | Mostly < 1 | Suggests increasing steps is more efficient than more parallel chains |

### Key Findings
- **MBAR advantage in the tail is substantial**: While direct sampling yields no estimates when counts are zero, MBAR provides relative CI widths several orders of magnitude smaller, representing the core benefit for rare event estimation.
- **Annealing is critical for TPS convergence**: Running averages show that more extreme $\lambda$ values require longer burn-in; using GR $\ge 1.1$ to discard non-converged segments bounds the overall bias.
- **Model behavior in OOD regions is "mechanical"**: Forcing ARI beyond the training distribution results in high-probability completions consisting of highly repetitive tokens ("rururururu..."), suggesting the model falls back to a "repetition = high likelihood" mode when extrapolating.
- **The proxy variable "consecutive repeats" correlates with high ARI**: This allows for its use as a runtime early-filtering metric, saving the cost of full ARI calculation.

## Highlights & Insights
- **Transferring statistical physics tools to LLM analysis** is a natural progression. LLM autoregression corresponds to sequence trajectories, tokens to particle states, and log-probs to energy. The resulting framework is highly effective.
- **No model weights required, only token log-probs**: Since $p_{\mathcal{M}}$ cancels out in MBAR weights, the analysis can be performed via APIs (provided they return token-level logprobs), making it a practical entry point for third-party auditing of closed-source models.
- **"Cheap Proxy Discovery" as a transferable paradigm**: This framework can be used for any scenario where the target metric is expensive or requires online filtering (e.g., toxicity, jailbreaking, PII leakage) by first sampling the rare sub-distribution.
- **Statistical rigor**: The inclusion of GR diagnostics, burn-in, and bootstrap CIs represents a significant engineering contribution, providing transparency regarding the limits of the method.

## Limitations & Future Work
- **Limitations acknowledged by authors**: Experiments were limited to the toy-scale TinyStories-8M. Applying this to production LLMs requires compute scaling and algorithmic improvements (adaptive runtime, parallel tempering, etc.).
- **Single prompt**: Primary experiments used a single 16-token prompt. Addressing prompt diversity in deployment requires integration with extrapolation methods (e.g., Jones et al. 2025).
- **Dependence on observable smoothness**: When the target observable is sparse (e.g., presence of a specific bad token), a smooth biasing observable proxy must be constructed, which remains an open problem.
- **Future Directions**: (1) Using fine-tuned small models as TPS proposal distributions to improve acceptance rates; (2) incorporating RL-based variational methods or Doob transforms; (3) automating proxy discovery using sparse regression or Shapley values on rare sub-distributions.

## Related Work & Insights
- **vs. Wu & Hilton 2025**: They estimate the probability of single rare tokens; this work extends analysis to full completions and arbitrary scalar observables.
- **vs. Jones et al. 2025**: They focus on extrapolation across prompts; this work focuses on fine-grained rare event estimation for a single prompt. Both are complementary.
- **vs. "Distribution Sharpening"**: The authors frame these as special cases of TPS or Doob transforms within a unified REA framework, noting that RLHF/DPO KL-regularization objectives share a similar variational perspective.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Systematically porting statistical physics tools to LLM analysis is a high-depth application of method transfer.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Very thorough within the toy scale (CI, GR, trade-offs), though missing sanity checks on production models.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Self-contained, clear motivation, and honest about limitations.
- **Value**: ⭐⭐⭐⭐⭐ Provides a framework for quantifying rare events without requiring model weights, offering high utility for safety and alignment auditing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ProFuser: Progressive Fusion of Large Language Models](../../AAAI2026/llm_nlp/profuser_progressive_fusion_of_large_language_models.md)
- [\[ICLR 2026\] The Lattice Representation Hypothesis of Large Language Models](../../ICLR2026/llm_nlp/the_lattice_representation_hypothesis_of_large_language_models.md)
- [\[ICML 2026\] dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching](dllm-cache_accelerating_diffusion_large_language_models_with_adaptive_caching.md)
- [\[ICML 2026\] Resting Neurons, Active Insights: Robustify Activation Sparsity for Large Language Models](resting_neurons_active_insights_robustify_activation_sparsity_for_large_language.md)
- [\[ACL 2026\] Foresight Optimization for Strategic Reasoning in Large Language Models](../../ACL2026/llm_nlp/foresight_optimization_for_strategic_reasoning_in_large_language_models.md)

</div>

<!-- RELATED:END -->
