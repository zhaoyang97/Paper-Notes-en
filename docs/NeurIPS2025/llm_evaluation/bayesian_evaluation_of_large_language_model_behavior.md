---
title: >-
  [Paper Note] Bayesian Evaluation of Large Language Model Behavior
description: >-
  [NeurIPS 2025][LLM Evaluation][Bayesian inference] This paper proposes a Beta-Binomial Bayesian framework for evaluating LLM behavior. By modeling the posterior distribution of $\theta_m$ over stochastic generations for each prompt, the framework quantifies statistical uncertainty in evaluation metrics and introduces sequential sampling strategies such as Thompson sampling to achieve narrower credible intervals with fewer API calls.
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "Bayesian inference"
  - "uncertainty quantification"
  - "sequential sampling"
  - "Thompson sampling"
  - "binary metrics"
date: 2026-05-08
content_hash: cddc8937505455d7
---

# Bayesian Evaluation of Large Language Model Behavior

**Conference**: NeurIPS 2025
**arXiv**: [2511.10661](https://arxiv.org/abs/2511.10661)  
**Code**: To be confirmed  
**Area**: LLM Evaluation
**Keywords**: Bayesian inference, LLM evaluation, uncertainty quantification, sequential sampling, Thompson sampling, binary metrics

## TL;DR
This paper proposes a Beta-Binomial Bayesian framework for evaluating LLM behavior. By modeling the posterior distribution of $\theta_m$ over stochastic generations for each prompt, the framework quantifies statistical uncertainty in evaluation metrics and introduces sequential sampling strategies such as Thompson sampling to achieve narrower credible intervals with fewer API calls.

## Background & Motivation

**Background**: LLM evaluation typically involves a single generation per prompt using deterministic decoding (greedy) on fixed benchmarks, with metrics such as accuracy or refusal rate. In practice, however, LLMs are deployed with stochastic decoding (temperature > 0, top-p, etc.), and outputs for the same prompt may vary across generations.

**Limitations of Prior Work**: (a) Deterministic evaluation ignores the stochasticity of LLM outputs and cannot distinguish between a prompt that is refused with 99% probability and one refused with only 55% probability; (b) Evaluation metrics are typically reported without uncertainty quantification, making it impossible to determine whether performance differences between models are statistically significant; (c) Multi-sample evaluation is costly due to API charges, yet naively distributing samples uniformly is inefficient.

**Key Challenge**: Accurately evaluating the stochastic behavior of an LLM requires multiple samples per prompt, but API cost constraints limit the total sampling budget. The central challenge is how to maximize evaluation precision under a fixed budget.

**Goal**: (1) Provide a Bayesian framework with uncertainty quantification for binary LLM behavior evaluation; (2) Reduce evaluation cost through sequential sampling strategies.

**Key Insight**: The binary behavior of each prompt is modeled as a Bernoulli trial with unknown parameter $\theta_m$. Bayesian inference is performed using a Beta prior and Binomial likelihood, yielding a posterior distribution over $\theta_m$. Posterior distributions for aggregate metrics (mean, threshold counts, etc.) are then derived from these individual posteriors.

**Core Idea**: A Beta-Binomial conjugate model is used to characterize the stochastic binary behavior of each prompt, and Thompson sampling is employed to allocate the sampling budget preferentially to prompts with the highest uncertainty.

## Method

### Overall Architecture
The input consists of $M$ benchmark prompts. An LLM system $\pi$ generates $n_m$ stochastic responses for each prompt $x^{(m)}$, and each output $y$ is mapped to $b(y) \in \{0,1\}$ by a binary judge (e.g., a toxicity detector or preference comparator). The number of positive outcomes $r_m$ per prompt is recorded; a Beta posterior is used to infer $\theta_m$; and posterior inference is then performed over the aggregate function $W = g(\theta_1, \ldots, \theta_M)$.

### Key Designs

1. **Beta-Binomial Posterior Inference**:

    - **Function**: Estimates the binary behavior probability $\theta_m$ and its uncertainty for each prompt.
    - **Mechanism**: A prior $\theta_m \sim \text{Beta}(\alpha_m, \beta_m)$ is updated upon observing $r_m$ positive outcomes to yield a posterior $\text{Beta}(\alpha_m + r_m,\ \beta_m + n_m - r_m)$. Conjugacy guarantees closed-form updates and computational efficiency.
    - **Design Motivation**: The Beta-Binomial model is the classical approach to uncertainty modeling for binary data. Each prompt is modeled independently, and posteriors can be updated incrementally.

2. **Posterior Distribution of Aggregate Metrics**:

    - **Function**: Derives the distribution of evaluation metrics (e.g., mean $W_{\text{mean}}$, above-threshold count $W_{>\nu}$) from the individual $\theta_m$ posteriors.
    - **Mechanism**: $W_{\text{mean}} = \frac{1}{M}\sum \theta_m$ is approximated via Monte Carlo sampling; $W_{>\nu} = \sum \mathbf{I}(\theta_m > \nu)$ follows a Poisson Binomial distribution and can be computed exactly.
    - **Design Motivation**: Practitioners are typically interested in aggregate rather than per-prompt metrics. The Bayesian framework naturally propagates individual uncertainty to aggregate quantities.

3. **Sequential Thompson Sampling**:

    - **Function**: Dynamically determines which prompt to sample next under a limited API call budget.
    - **Mechanism**: At each step, Thompson sampling is applied to the current $\theta_m$ posteriors—a value of $\theta_m$ is drawn for each prompt, and the prompt whose sampled value contributes most to uncertainty in the aggregate metric is selected for the next generation. Maximum variance and other strategies are also explored.
    - **Design Motivation**: The uncertainty of $\theta_m$ varies across prompts (some are near 0 or 1 and require few samples; others are near 0.5 and require many). Sequential strategies enable intelligent budget allocation.

### Loss & Training
This paper involves no model training and constitutes a pure inference framework. All experiments use existing LLM APIs (GPT-4o-mini, GPT-4.1-nano) and an LM-as-judge (GPT-4.1-mini) for inference. Preference experiments use temperature = 1.0 and top-p = 0.9; refusal rate experiments employ the same stochastic decoding to capture output-level randomness. The prior is set to the uniform Beta(1, 1).

## Key Experimental Results

### Case Study 1: Pairwise Preference (GPT-4o-mini vs. GPT-4.1-nano)

| Method | Result |
|--------|--------|
| Greedy decoding | Model A preferred on 41/80 prompts |
| Bayesian ($n=50$) | $W_{\text{mean}}$ 95% CI: (51%, 53%) |
| Bayesian ($n=50$) | 23 prompts where Model A is preferred with >75% probability |

### Case Study 2: Jailbreak Refusal Rate

| Evaluation | Description |
|------------|-------------|
| Batch evaluation | Reveals substantial variation in per-prompt refusal probability $\theta_m$ |
| Sequential vs. batch | Thompson sampling achieves equivalent precision with fewer samples |

### Ablation Study: Sequential Sampling Strategy Comparison

| Strategy | Efficiency | Description |
|----------|------------|-------------|
| Uniform (batch) | Baseline | Equal number of samples per prompt |
| Max Variance | Better | Prioritizes prompts with highest posterior variance |
| Thompson Sampling | Best | Balances exploration and exploitation; fastest convergence of aggregate metric |

### Key Findings
- Greedy decoding conceals important information: two prompts may both favor Model A under greedy decoding, yet one does so with 99% probability and the other with only 55% under stochastic decoding.
- Sequential Thompson sampling reduces the credible interval of the aggregate metric by approximately 20–30% relative to uniform sampling under the same total budget.
- The incremental update property of the Beta posterior incurs virtually no additional computational overhead in the sequential setting.

## Highlights & Insights
- **Formalizing LLM evaluation as statistical inference**: Rather than ad-hoc averaging over multiple runs, the proposed framework provides a rigorous Bayesian formulation with statistically guaranteed posterior credible intervals.
- **Sequential sampling reduces cost**: The introduction of Thompson sampling enables intelligent evaluation—concentrating limited API calls on the most informative prompts rather than sampling uniformly.
- **Black-box compatibility**: The framework requires no internal information from the LLM (no logits, weights, or architectural details) and operates purely on observed outputs.

## Limitations & Future Work
- **Binary metrics only**: The current framework is restricted to $b(y) \in \{0,1\}$ and cannot directly accommodate continuous ratings (e.g., 1–5 quality scores) or multi-class judgments.
- **Deterministic judge assumption**: The binary judge $b(y)$ is assumed to be deterministic; in practice, LLM-as-judge is itself stochastic, introducing an additional source of uncertainty.
- **Independence assumption**: Each prompt's $\theta_m$ is modeled independently, ignoring potentially shared behavioral patterns among similar prompts (e.g., prompts on the same topic may have correlated refusal rates).
- **Computational cost**: Multiple samples per prompt require a large number of API calls (80 prompts × 50 samples = 4,000 calls), which is non-trivial for closed-source models.
- **Choice of aggregation function**: $W_{\text{mean}}$ and $W_{>\nu}$ are the most basic aggregation forms; more complex evaluation needs (e.g., conditional evaluation, subgroup comparison) are not addressed.
- **Future directions**: (1) Extension to continuous ratings via Beta regression or Gaussian models; (2) Hierarchical Bayesian models to share information across prompts; (3) Incorporation of judge uncertainty into the model; (4) Development of adaptive stopping criteria to reduce unnecessary sampling.

## Related Work & Insights
- **vs. Scholten et al. (2025)**: Also addresses output-level uncertainty, but employs frequentist methods; the Bayesian approach here naturally supports sequential updating and decision-making.
- **vs. Miller (2024)**: Discusses variance reduction through repeated generation but does not provide a formal statistical model; this paper contributes a complete Bayesian inference framework.
- **vs. Hariri et al. (2025)**: Applies Bayesian methods to multi-class evaluation; this paper focuses on binary behavior with sequential sampling.
- **Insights**: The framework is directly transferable to any setting requiring behavioral evaluation of stochastic systems (e.g., agents). The sequential Thompson sampling approach could also inform benchmark construction—identifying the most discriminative prompts.

## Rating
- **Novelty**: ⭐⭐⭐ The Beta-Binomial model itself is not new, but its systematic application to LLM evaluation combined with sequential sampling represents a meaningful contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two practical case studies (preference comparison + jailbreak refusal) with both batch and sequential evaluation modes compared.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Written in a pedagogical style suited to a statistical audience, with complete and clear derivations.
- **Value**: ⭐⭐⭐⭐ Provides a statistically rigorous uncertainty quantification tool for LLM evaluation with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation](../../ICLR2026/llm_evaluation/dont_passk_a_bayesian_framework_for_large_language_model_evaluation.md)
- [\[NeurIPS 2025\] Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training](exploiting_vocabulary_frequency_imbalance_in_language_model_pre-training.md)
- [\[NeurIPS 2025\] Hyperbolic Fine-Tuning for Large Language Models](hyperbolic_fine-tuning_for_large_language_models.md)
- [\[NeurIPS 2025\] LTD-Bench: Evaluating Large Language Models by Letting Them Draw](ltd-bench_evaluating_large_language_models_by_letting_them_draw.md)
- [\[NeurIPS 2025\] Can Large Language Models Master Complex Card Games?](can_large_language_models_master_complex_card_games.md)

</div>

<!-- RELATED:END -->
