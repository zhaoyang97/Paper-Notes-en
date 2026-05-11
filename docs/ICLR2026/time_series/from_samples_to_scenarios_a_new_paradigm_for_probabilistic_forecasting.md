---
title: >-
  [Paper Note] From Samples to Scenarios: A New Paradigm for Probabilistic Forecasting
description: >-
  [ICLR 2026][Time Series][probabilistic forecasting] This paper proposes the Probabilistic Scenarios paradigm, in which a model directly outputs a finite set of {scenario, probability} pairs in place of sampling…
tags:
  - "ICLR 2026"
  - "Time Series"
  - "probabilistic forecasting"
  - "scenario generation"
  - "discrete probability"
  - "linear model"
date: 2026-05-08
content_hash: 88a7f02a3929b141
---

# From Samples to Scenarios: A New Paradigm for Probabilistic Forecasting

**Conference**: ICLR 2026
**arXiv**: [2509.19975](https://arxiv.org/abs/2509.19975)
**Code**: [GitHub](https://github.com/Fifthky/TimePrism)
**Area**: Time Series
**Keywords**: probabilistic forecasting, time series, scenario generation, discrete probability, linear model

## TL;DR
This paper proposes the Probabilistic Scenarios paradigm, in which a model directly outputs a finite set of {scenario, probability} pairs in place of sampling, and introduces TimePrism — a model consisting of only three parallel linear layers — that achieves 9/10 SOTA results across 5 benchmark datasets.

## Background & Motivation
**Background**: Probabilistic time series forecasting is fundamental to decision-making under uncertainty. Mainstream approaches include parametric distribution models, generative models (e.g., diffusion), and structured probabilistic models (flows/copulas), all of which rely on sampling to represent the predictive distribution.

**Limitations of Prior Work**: The sampling paradigm suffers from three inherent drawbacks: (i) **absence of probabilities** — generated trajectories carry no associated probability values; (ii) **insufficient coverage** — a limited number of samples fails to capture low-probability, high-impact tail events; (iii) **inference overhead** — the computational cost of generating multiple samples grows linearly with sample count.

**Key Challenge**: High-quality probabilistic forecasting requires a large number of samples to adequately approximate the distribution, yet large-scale sampling is computationally prohibitive and does not provide explicit probabilities.

**Goal**: To design a probabilistic forecasting paradigm that does not rely on sampling and can output a complete discrete probability distribution in a single forward pass.

**Key Insight**: Reformulate the learning objective from "approximating a continuous probability space" to "learning a probability distribution over a finite set of scenarios" — conceptually analogous to VQ-VAE but applied directly to the output trajectory space.

**Core Idea**: A simple linear model directly generates $N$ future scenarios together with their associated probabilities, entirely bypassing sampling.

## Method

### Overall Architecture
The Probabilistic Scenarios paradigm defines the model as:
$$f(\mathbf{x}) = (\mathcal{Y}_{\text{pred}}, \mathbf{p})$$
where $\mathcal{Y}_{\text{pred}} = \{\mathbf{y}_n\}_{n=1}^N$ is a set of $N$ predicted scenarios and $\mathbf{p} = (p_1, \dots, p_N)$ is the corresponding probability vector satisfying $\sum p_n = 1$.

### Key Designs
1. **Time Series Decomposition**: The input history $\mathbf{x} \in \mathbb{R}^{L \times D}$ is decomposed via moving average into a trend component $\mathbf{x}_{\text{trend}}$ and a seasonal component $\mathbf{x}_{\text{season}}$.
2. **Trend + Seasonal Linear Layers**: The trend layer produces $M$ trend predictions and the seasonal layer produces $K$ seasonal predictions; these are combined to yield $N = M \times K$ scenarios:
    $\mathcal{Y}_{\text{pred}} = \{\mathbf{y}_{t,m} + \mathbf{y}_{s,k} \mid m \in [M], k \in [K]\}$
   This compositional design reduces parameter complexity to $\mathcal{O}(\sqrt{N})$, far more efficient than directly generating $N$ scenarios.
3. **Probability Layer**: A third linear layer takes the original (non-decomposed) history as input and outputs an $N$-dimensional logit vector $\boldsymbol{\pi}$, which is converted to a probability vector via Softmax.

### Loss & Training
The total loss consists of two components:
$$\mathcal{L}_{\text{Prism}} = \mathcal{L}_{\text{recon}} + \lambda \cdot \mathcal{L}_{\text{prob}}$$

- **Scenario Reconstruction Loss** (WTA): The "winner" scenario closest to the ground truth is identified as $n^* = \arg\min_n \|\mathbf{y}_{gt} - \mathbf{y}_n\|_2^2$, and MSE is computed only for that scenario.
- **Probability Loss**: Cross-entropy is used to train the probability layer to assign the highest probability to the winner:
  $$\mathcal{L}_{\text{prob}} = -\log \frac{\exp(\pi_{n^*})}{\sum_j \exp(\pi_j)}$$
- In practice, a relaxed WTA is employed for training stability, with $\lambda=1$.

## Key Experimental Results

### Main Results
Weighted CRPS on 5 benchmark datasets (Electricity, Exchange, Solar, Traffic, Wikipedia):

| Model | Elec. | Exch. | Sol. | Traf. | Wiki. |
|------|-------|-------|------|-------|-------|
| TimeGrad | 0.232 | 0.845 | 0.241 | 0.162 | 0.517 |
| TACTiS-2 | 0.299 | 0.648 | 0.236 | 0.257 | 0.484 |
| TimeMCL | 0.370 | 1.12 | 0.290 | 0.262 | 0.640 |
| **TimePrism** | **0.133** | **0.468** | **0.085** | **0.111** | **0.506** |

On the Distortion metric, TimePrism achieves SOTA on all 5 datasets.

### Ablation Study
Effect of scenario count $N$ (Solar dataset):

| N | CRPS | Distortion | FLOPs (relative) |
|---|------|------------|-------------|
| 1 | 0.199 | 0.266 | 1.0x |
| 16 | 0.137 | 0.307 | 4.2x |
| 256 | 0.093 | 0.158 | 19.9x |
| 625 | 0.085 | 0.101 | 34.8x |
| 1024 | 0.082 | 0.092 | 48.3x |

Performance gains saturate around $N=625$.

### Key Findings
- TimePrism has constant inference FLOPs ($5.1 \times 10^5$) regardless of the number of scenarios, whereas TimeGrad requires $1.9 \times 10^{10}$ FLOPs for 100 samples.
- Visualizations show that TimePrism assigns high probability to common peak scenarios while assigning low probability to rare low-peak scenarios — a distinction that sampling-based models cannot make.
- The compositional architecture ($N = M \times K$) keeps parameter growth between $\mathcal{O}(\sqrt{N})$ and $\mathcal{O}(N)$.

## Highlights & Insights
- **Paradigm Innovation**: A fundamental shift from "sampling to approximate a continuous distribution" to "directly generating discrete scenarios with explicit probabilities" — conceptually simple yet highly effective.
- **Minimal Architecture as Proof of Concept**: Achieving SOTA with only three parallel linear layers (no nonlinear activations) demonstrates the remarkable potential of the paradigm itself.
- **Unified Evaluation Framework**: The paper proposes two complementary metrics — Weighted CRPS and Distortion — and derives fair, comparable computation formulas for both paradigms.
- **Efficiency Advantage**: A single forward pass suffices at inference; computational cost is 1–5 orders of magnitude lower than the strongest baselines.

## Limitations & Future Work
- The linear model may be ill-suited for very high-dimensional sequences or series without clear trend/seasonal patterns.
- The model assumes fixed input/output lengths, lacking flexibility for variable-length sequences.
- Multivariate modeling employs a weight-sharing strategy, resulting in relatively simple cross-variable dependency modeling.
- The optimal scenario count $N$ depends on data complexity and currently requires manual tuning.
- The WTA loss may cause some scenario heads to be neglected during early training (winner-takes-all effect); relaxed WTA only partially mitigates this.
- The approach has not been validated on larger-scale benchmarks such as GIFT-Eval.
- Sensitivity analysis across different forecasting horizons is absent.

## Related Work & Insights
- **vs. TimeMCL**: TimeMCL also outputs discrete scenarios but does not directly model probabilities, resulting in suboptimal CRPS; this work unifies scenario fidelity and probability matching via the probability layer.
- **Conceptual analogy to VQ-VAE**: Discretization is applied directly to the output trajectory space rather than the latent space.
- **vs. TACTiS-2**: TACTiS-2 can compute probability densities but still requires sampling to obtain trajectories; this work directly outputs discrete scenarios.
- **vs. TimeGrad**: Diffusion models require iterative sampling; 100-sample inference costs $10^4\times$ more FLOPs than TimePrism.
- Future work may integrate this paradigm with powerful backbones such as Transformers or Diffusion models to unlock stronger multivariate modeling capabilities.
- An adaptive mechanism for selecting the scenario count $N$ is also a valuable direction for future research.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ A fundamental innovation in the probabilistic forecasting paradigm, shifting from sampling to discrete scenario probabilities.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five datasets, multiple baselines, ablations, and visualizations — though limited to the time series domain.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clear and the logical chain from problem formulation to proposed solution is complete.
- Value: ⭐⭐⭐⭐⭐ Opens a new direction for probabilistic forecasting; the fact that a minimal model achieves SOTA is highly convincing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Scaling LLM Speculative Decoding: Non-Autoregressive Forecasting in Large-Batch Scenarios](../../AAAI2026/time_series/scaling_llm_speculative_decoding_non-autoregressive_forecasting_in_large-batch_s.md)
- [\[NeurIPS 2025\] Benchmarking Probabilistic Time Series Forecasting Models on Neural Activity](../../NeurIPS2025/time_series/benchmarking_probabilistic_time_series_forecasting_models_on_neural_activity.md)
- [\[NeurIPS 2025\] AERO: A Redirection-Based Optimization Framework Inspired by Judo for Robust Probabilistic Forecasting](../../NeurIPS2025/time_series/aero_a_redirection-based_optimization_framework_inspired_by_judo_for_robust_prob.md)
- [\[AAAI 2026\] ProbFM: Probabilistic Time Series Foundation Model with Uncertainty Decomposition](../../AAAI2026/time_series/probfm_probabilistic_time_series_foundation_model_with_uncertainty_decomposition.md)
- [\[ICLR 2026\] scits scientific time series understanding and generation with llms](scits_scientific_time_series_understanding_and_generation_with_llms.md)

</div>

<!-- RELATED:END -->
