---
title: >-
  [Paper Note] Valid Inference with Imperfect Synthetic Data
description: >-
  [NeurIPS 2025][LLM (Other)][synthetic data] A hyperparameter-free framework based on Generalized Method of Moments (GMM) is proposed to integrate imperfect LLM-generated synthetic data with real data for statistically valid inference. When the residuals of synthetic data are correlated with those of real data, the framework can substantially reduce estimation variance, while guaranteeing no harm to estimation quality in the worst case (i.e., when synthetic data is entirely un…
tags:
  - "NeurIPS 2025"
  - "LLM (Other)"
  - "synthetic data"
  - "statistical inference"
  - "GMM"
  - "prediction-powered inference"
  - "computational social science"
date: 2026-05-08
content_hash: 40e89192c0360c30
---

# Valid Inference with Imperfect Synthetic Data

**Conference**: NeurIPS 2025
**arXiv**: [2508.06635](https://arxiv.org/abs/2508.06635)  
**Code**: To be confirmed  
**Area**: LLM/NLP
**Keywords**: synthetic data, statistical inference, GMM, prediction-powered inference, computational social science

## TL;DR

A hyperparameter-free framework based on Generalized Method of Moments (GMM) is proposed to integrate imperfect LLM-generated synthetic data with real data for statistically valid inference. When the residuals of synthetic data are correlated with those of real data, the framework can substantially reduce estimation variance, while guaranteeing no harm to estimation quality in the worst case (i.e., when synthetic data is entirely uninformative).

## Background & Motivation

LLMs are increasingly applied in limited-data settings, particularly in computational social science and human-subjects research. Existing work primarily explores two paradigms:

**Proxy Labeling**: Using models to predict labels/covariates for unlabeled data.

**Synthetic Simulation**: Using models to generate entirely new synthetic samples (e.g., simulating survey responses).

However, **naively mixing synthetic data with real data leads to severely biased estimates**, undermining statistical validity. Existing debiasing methods (e.g., Prediction-Powered Inference, PPI) focus primarily on the proxy labeling setting; how to use **fully synthetic samples** (where both text and labels are model-generated) remains unresolved.

The core challenge is: how can synthetic data improve estimation efficiency while preserving consistency and correct asymptotic coverage?

## Method

### Overall Architecture

A three-tier data structure is constructed:
- **Labeled data** $\mathcal{D}_{\text{labeled}} = \{(T_i, X_i, Y_i)\}_{i=1}^n$: a small set of human annotations
- **Proxy data** $\mathcal{D}_{\text{proxy}}$: model $f$ predicts $(\hat{X}, \hat{Y})$ for all texts
- **Synthetic data** $\mathcal{D}_{\text{synthetic}}$: conditionally generated new samples $(\tilde{T}_k, \tilde{X}_k, \tilde{Y}_k)$

The key innovations lie in the **synthetic data generation strategy** and the **GMM-based integration of multi-source information**.

### Key Design 1: Conditional Synthetic Data Generation

A new sample $\tilde{T}_i$ is conditionally generated for each real text $T_i$:

$$\tilde{T}_k \sim \mathbb{P}(\cdot \mid T_i, X_i) \quad \text{(labeled samples)}$$
$$\tilde{T}_k \sim \mathbb{P}(\cdot \mid T_j, \hat{X}_j) \quad \text{(unlabeled samples)}$$

The model then extracts covariates and outcomes from synthetic samples: $\tilde{X}_k \sim \mathbb{P}(\cdot | \tilde{T}_k)$, $\tilde{Y}_k \sim \mathbb{P}(\cdot | \tilde{T}_k)$.

**Two motivations**:
- **Machine learning perspective**: analogous to in-context prompting, real samples serve as demonstrations to guide generation, and iterating over different samples improves diversity.
- **Statistical perspective**: introduces a **correlation structure** between real texts $T_i$ and synthetic samples $\tilde{T}_i$, which is critical for downstream GMM integration.

### Key Design 2: GMM Estimation with Augmented Moment Conditions

The target parameter $\theta^*$ is identified by the moment condition: $\mathbb{E}[\psi^{(\ell)}(\theta^*)] = 0$.

An auxiliary parameter $\eta_m$ is introduced for each auxiliary data source (proxy/synthetic), yielding an augmented moment vector:

$$g_t(\theta, \eta) = \begin{bmatrix} S_t \\ S_t \\ \vdots \\ 1 \\ \vdots \end{bmatrix} \odot \begin{bmatrix} \psi(\theta) \\ \psi(\eta_1) \\ \vdots \\ \psi(\eta_1) \\ \vdots \end{bmatrix} \in \mathbb{R}^{p + 2Mp}$$

where $S_t$ is a labeling indicator ($s=1$ for labeled, $s=0$ for unlabeled), and $M$ is the number of auxiliary data sources.

**Key structure**:
- $\theta$ appears only in moments evaluated on real labeled data → consistency is guaranteed regardless of synthetic data quality.
- Each $\eta_m$ has two sets of moments: one evaluated only on labeled data (capturing the real–auxiliary residual correlation), and one evaluated on all data (exploiting the larger sample size).

### Loss & Training: Two-Step GMM

A two-step GMM estimator is employed:

**Step 1**: Obtain initial parameter estimates $(\hat{\theta}_T^{(os)}, \hat{\eta}_T^{(os)})$ using the identity weight matrix $\mathbf{W}_T = \mathbf{I}$.

**Step 2**: Compute the optimal weight matrix (inverse of moment covariance):

$$\hat{\Omega}_T = \frac{1}{T} \sum_{t=1}^T g_t(\hat{\theta}^{(os)}, \hat{\eta}^{(os)}) g_t(\hat{\theta}^{(os)}, \hat{\eta}^{(os)})^\top$$

$$\hat{\mathbf{W}}_T = \hat{\Omega}_T^{-1}$$

Then minimize the weighted GMM objective:

$$\hat{\theta}_T, \hat{\eta}_T = \arg\min_{\theta, \eta} \left[\frac{1}{T} \sum_t g_t(\theta, \eta)\right]^\top \hat{\mathbf{W}}_T \left[\frac{1}{T} \sum_t g_t(\theta, \eta)\right]$$

**Core Mechanism**: In Step 1, synthetic data does not affect the estimation of $\theta$ (moment independence); in Step 2, the **off-diagonal entries** of the weight matrix capture the covariance between auxiliary and real data residuals, enabling auxiliary information to improve the estimation of $\theta$.

### Theoretical Guarantees

**Proposition 1**: The estimator $\hat{\theta}_T$ is consistent and asymptotically normal:

$$\sqrt{T}(\hat{\theta}_T - \theta) \xrightarrow{d} \mathcal{N}(0, V)$$

**Theorem 1** (Variance Analysis): Partitioning moments into real data residuals $m_t(\theta)$ and synthetic data residuals $h_t(\eta)$:

- **Worst case**: When synthetic residuals are independent of real residuals, the variance degenerates to the optimal variance of using real data alone → **no harm**.
- **Best case**: When synthetic residuals predict real residuals, the variance lower bound is proportional to the variance of the regression residual of real residuals on synthetic residuals → **substantial improvement**.

## Key Experimental Results

### Main Results: Four Computational Social Science Tasks

GPT-4o is used to generate proxy and synthetic data; evaluation is performed on logistic regression and OLS regression:

**Task 1**: Effect of hedging markers on perceived politeness (Stack Exchange / Wikipedia)
**Task 2**: Effect of first-person plural pronouns on perceived politeness
**Task 3**: Effect of affirmative language on media stance toward climate change (news headlines)
**Task 4**: Effect of legislator ideology on bill type (congressional bills)

| Metric | GMM-Synth vs. Baselines |
|--------|--------------------------|
| MSE | **Lowest in 8/8 tasks**; reduction >50% at low labeling rates |
| Coverage | Valid coverage maintained across all tasks |
| CI width | **Narrowest in 7/8 tasks** |
| Effective sample size | Saves >50% of human annotations |

### Ablation Study: GMM-Synth vs. GMM-Proxy

| Comparison | Finding |
|------------|---------|
| GMM-Synth vs. GMM-Proxy | GMM-Synth consistently outperforms GMM-Proxy on all tasks |
| PPI++Synth vs. PPI++Proxy | Gains from synthetic data in PPI++ are marginal (no improvement in 5/8 tasks) |

This demonstrates that the GMM framework integrates synthetic data more effectively than existing debiasing methods.

### Baseline Comparison

| Method | MSE | Coverage | Hyperparameters |
|--------|-----|----------|-----------------|
| Real Only | Baseline | ✓ | None |
| PPI++Proxy | Better than Real Only | ✓ | None |
| PPI++Synth | Sometimes better than Proxy, sometimes not | ✓ | Requires cross-validation to select α |
| RePPI | Moderate improvement | ✓ | Requires model fitting |
| **GMM-Synth** | **Best** | ✓ | **None** |

### Key Findings

1. Gains from synthetic data are most pronounced at **low labeling rates** — precisely the setting where synthetic data is most needed.
2. Using proxy or synthetic data alone leads to severely biased estimates.
3. Synthetic data generated by weaker open-source models (Llama-3-8B, Qwen-3-8B) also yields consistent gains.
4. The GMM approach requires no hyperparameter tuning, whereas PPI++ requires cross-validation to select α.

## Highlights & Insights

1. **Theoretical elegance**: The GMM framework naturally handles multi-source information; the optimal weight matrix automatically captures the synthetic–real residual correlation without manual tuning. The "worst-case harmless, best-case substantially improved" guarantee is highly practical.
2. **Innovation in generation strategy**: Conditioning synthetic data generation on real samples both achieves an in-context prompting effect and establishes a statistically meaningful correlation structure — a seamless integration of machine learning intuition and statistical theory.
3. **Extensibility**: The framework naturally supports multiple auxiliary data sources ($M$ synthetic datasets from different models), enabling plug-and-play extension.
4. **Broad practical significance**: In social science research where annotation is costly, the method preserves statistical validity while saving more than 50% of human labeling effort.

## Limitations & Future Work

1. **Asymptotic guarantees**: The theoretical guarantees are asymptotic; coverage may be insufficient at very small sample sizes.
2. **Model quality dependence**: Although poor-quality synthetic data causes no harm, it also yields no improvement.
3. **Task scope**: Validation is currently limited to regression-type tasks (GLM/OLS); applicability to more complex inference settings (e.g., causal inference, structural equation models) requires further investigation.
4. **Generation cost**: Conditional generation of synthetic data for each sample incurs substantial API costs for large-scale datasets.
5. **Text modality focus**: The framework is designed for text data; generalization to other modalities (images, tables) is not discussed.

## Related Work & Insights

- **PPI/PPI++** (Angelopoulos et al., 2023): Primarily addresses proxy labeling; the present work extends this to the fully synthetic data setting.
- **RePPI** (Ji et al., 2025): Maps proxy/synthetic losses to real losses via arbitrary model mappings, but requires additional modeling.
- **Debiased inference literature**: The design-based supervised learning framework of Egami et al. (2023); the proposed GMM approach is more flexible (supporting multiple proxy covariates and outcomes).
- **LLM simulation literature**: Works on social simulation (Park et al., 2023) and survey simulation demonstrate the potential of synthetic data but lack statistical validity guarantees — a gap the present paper fills.
- **Implication**: The GMM framework can provide statistical grounding for a wide range of "AI-assisted + human-verified" pipelines, with increasing value as LLM capabilities improve.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic application of the GMM framework to statistical inference integrating fully synthetic data, filling an important theoretical gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 4 real-world social science tasks × 2 regression types × multiple baselines × multiple LLMs; task variety is somewhat narrow.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear and well-motivated; the core insight of "residuals predicting residuals" is articulated with precision.
- **Value**: ⭐⭐⭐⭐ — Provides both theoretical foundations and a practical method for using LLM-generated synthetic data in rigorous statistical analysis; strong forward-looking relevance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Theorem Prover as a Judge for Synthetic Data Generation](../../ACL2025/llm_nlp/theorem_prover_as_a_judge_for_synthetic_data_generation.md)
- [\[ACL 2025\] Evaluating Language Models as Synthetic Data Generators](../../ACL2025/llm_nlp/evaluating_lms_synthetic_data_gen.md)
- [\[ACL 2025\] DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models](../../ACL2025/llm_nlp/difflm_controllable_synthetic_data_generation_via_diffusion_language_models.md)
- [\[ICLR 2026\] SIPDO: Closed-Loop Prompt Optimization via Synthetic Data Feedback](../../ICLR2026/llm_nlp/sipdo_closed-loop_prompt_optimization_via_synthetic_data_feedback.md)
- [\[ACL 2026\] Synthetic Eggs in Many Baskets: The Impact of Synthetic Data Diversity on LLM Fine-Tuning](../../ACL2026/llm_nlp/synthetic_eggs_in_many_baskets_the_impact_of_synthetic_data_diversity_on_llm_fin.md)

</div>

<!-- RELATED:END -->
