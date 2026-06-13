---
title: >-
  [Paper Note] In-Context Learning of Stochastic Differential Equations with Foundation Inference Models
description: >-
  [NeurIPS 2025][Time Series][stochastic differential equations] This paper proposes FIM-SDE, a pretrained recognition model capable of zero-shot (in-context) estimation of drift and diffusion functions of low-dimensional…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "stochastic differential equations"
  - "in-context learning"
  - "foundation inference models"
  - "drift function estimation"
  - "diffusion function estimation"
date: 2026-05-08
content_hash: 0ef8ae4db90ef2a9
---

# In-Context Learning of Stochastic Differential Equations with Foundation Inference Models

**Conference**: NeurIPS 2025
**arXiv**: [2502.19049](https://arxiv.org/abs/2502.19049)  
**Authors**: Patrick Seifner, Kostadin Cvejoski, David Berghaus, Cesar Ojeda, Ramses J. Sanchez
**Code**: Available  
**Area**: Time Series / Stochastic Differential Equations
**Keywords**: stochastic differential equations, in-context learning, foundation inference models, drift function estimation, diffusion function estimation

## TL;DR

This paper proposes FIM-SDE, a pretrained recognition model capable of zero-shot (in-context) estimation of drift and diffusion functions of low-dimensional SDEs from noisy time series data, and further surpasses all baseline methods via rapid fine-tuning.

## Background & Motivation

Stochastic differential equations (SDEs) describe dynamical systems exhibiting a superposition of deterministic flow (governed by the drift function) and stochastic fluctuations (governed by the diffusion function):

$$dX_t = f(X_t, t)dt + g(X_t, t)dW_t$$

Accurately estimating the drift function $f$ and diffusion function $g$ from observational data is a central problem in machine learning with broad applications across the natural and social sciences. However, existing methods suffer from the following limitations:

**Dependence on prior knowledge**: Methods such as symbolic regression require presupposed functional forms.

**Training complexity**: Methods such as Neural SDEs require bespoke design and training for each individual dataset.

**Poor generalizability**: Existing models generally cannot generalize across SDE systems.

**Absence of foundation models**: The SDE discovery field lacks pretrained foundation models analogous to those in NLP/CV.

Core motivation: Can a foundation inference model be trained to accurately estimate drift and diffusion functions of arbitrary SDEs in a zero-shot setting, without requiring training on the target system?

## Method

### Overall Architecture

FIM-SDE is a Transformer-based foundation inference model consisting of three stages:

1. **Pretraining stage**: Supervised training on a large collection of synthetic SDE trajectories.
2. **In-context inference stage**: Zero-shot estimation of the target SDE's functions given new observation sequences.
3. **Fine-tuning stage** (optional): Rapid adaptation on the target dataset.

### Key Designs

#### 1. Amortized Inference

FIM-SDE draws on the principle of amortized inference: rather than training from scratch for each new problem, a general-purpose recognition network is pretrained to map observational data to the SDE parameter space.

Core advantages:
- Inference requires only a single forward pass (as opposed to iterative optimization).
- Naturally handles observation sequences of varying lengths and sampling rates.
- Natively supports in-context learning — estimation improves automatically as more observations are provided.

#### 2. Neural Operators

Leveraging the concept of neural operators, FIM-SDE learns a mapping from observed trajectories to function space:

$$\mathcal{F}: \{(t_i, X_{t_i})\}_{i=1}^N \mapsto (\hat{f}, \hat{g})$$

This enables the model to produce function-level estimates rather than merely finite-dimensional parameters.

#### 3. Transformer Architecture

Key architectural choices:
- **Input serialization**: Observed SDE trajectories $\{(t_i, X_{t_i})\}$ are encoded as token sequences.
- **Attention mechanism**: Captures long-range dependencies within trajectories.
- **Output decoding**: Transformer outputs are decoded into drift and diffusion function values at query points.
- **Multi-trajectory aggregation**: When multiple observed trajectories are available, information is aggregated naturally via the attention mechanism.

#### 4. Pretraining Data Generation

One of the core innovations lies in the construction of the pretraining dataset:

1. **SDE system sampling**: Drift and diffusion functions are sampled from a function space (e.g., a Gaussian process prior).
2. **Numerical simulation**: SDE trajectories are simulated using methods such as Euler–Maruyama.
3. **Observation noise injection**: Noise is added to simulate real-world measurement uncertainty.
4. **Discretization**: Observation points are sampled at random intervals.

The training set is large in scale and diverse, covering a wide range of SDE dynamical behaviors.

### Loss & Training

Supervised pretraining loss:

$$\mathcal{L} = \sum_{q \in Q} \left[ \|\hat{f}(q) - f^*(q)\|^2 + \|\hat{g}(q) - g^*(q)\|^2 \right]$$

where $Q$ denotes a set of query points, and $f^*$ and $g^*$ are the ground-truth drift and diffusion functions.

Fine-tuning strategy:
- The same loss function is employed.
- The learning rate is typically set to one-tenth of the pretraining learning rate.
- Significant improvement is achieved with only a small number of epochs.
- In the absence of ground-truth function labels, a reconstruction loss may be substituted.

## Key Experimental Results

### Main Results

#### Zero-Shot Estimation on Synthetic SDE Systems

| SDE System | FIM-SDE (Zero-Shot) | Symbolic Regression | GP Regression | Neural SDE | FIM-SDE (Fine-Tuned) |
|------------|---------------------|---------------------|---------------|------------|----------------------|
| Double-well dynamics | Near match | Requires prior | Match | Match | **Best** |
| Weakly perturbed Lorenz | Near match | Difficult | Match | Match | **Best** |
| Geometric Brownian motion | Near match | Match | Match | Match | **Best** |
| Ornstein–Uhlenbeck | Match | Match | Match | Match | **Best** |
| Nonlinear diffusion | Near match | Difficult | Near match | Near match | **Best** |

Note: "Match" indicates performance comparable to the best baseline; "Best" indicates outperformance of all baselines.

#### Real-World Datasets

| Dataset | Data Type | FIM-SDE (Zero-Shot) | FIM-SDE (Fine-Tuned) | Best Baseline |
|---------|-----------|---------------------|----------------------|---------------|
| Stock prices | Finance | Near baseline | **Surpasses** | GP / Neural SDE |
| Oil price volatility | Commodities | Near baseline | **Surpasses** | GP |
| Wind speed fluctuations | Meteorology | Near baseline | **Surpasses** | Neural SDE |

Key observations:
- In the zero-shot setting, FIM-SDE matches baseline methods trained individually on the target dataset.
- After fine-tuning, FIM-SDE consistently surpasses all baselines.

### Ablation Study

#### Effect of the Number of Observed Trajectories

| No. of Trajectories | Zero-Shot MSE | Fine-Tuned MSE | Improvement |
|---------------------|---------------|----------------|-------------|
| 1 | High | Moderate | Large |
| 5 | Moderate | Low | Moderate |
| 10 | Low | Low | Small |
| 50 | Low | Lowest | Marginal |

Observation: Zero-shot performance improves steadily as the number of observed trajectories increases, confirming that the model is genuinely performing in-context learning.

#### Effect of Pretraining Data Volume

| No. of Pretraining SDEs | Zero-Shot Generalization | Fine-Tuning Convergence Speed |
|-------------------------|--------------------------|-------------------------------|
| 1K | Poor | Moderate |
| 10K | Moderate | Fast |
| 100K | Good | Faster |
| 1M | Best | Fastest |

#### Robustness to Noise Level

| Observation Noise $\sigma$ | FIM-SDE MSE | Neural SDE MSE | GP MSE |
|----------------------------|-------------|----------------|--------|
| 0.01 | Low | Low | Low |
| 0.1 | Moderate | High | Moderate |
| 0.5 | Moderately high | High | Moderately high |
| 1.0 | High | Very high | High |

FIM-SDE performs no worse than the best baseline across all noise levels.

### Key Findings

1. **Zero-shot capability**: FIM-SDE provides meaningful function estimates without any training on the target system.
2. **Rapid fine-tuning**: A small number of fine-tuning epochs is sufficient to surpass all baselines trained on the target dataset.
3. **Real-world effectiveness**: The model is effective on real-world data from domains such as finance and meteorology.
4. **In-context learning phenomenon**: More observed trajectories yield better estimates — the model continues to improve at inference time.
5. **Robustness**: The model exhibits strong robustness to observation noise and irregular sampling.

## Highlights & Insights

1. **Foundation model paradigm for SDE discovery**: This work is the first to introduce the foundation model / in-context learning paradigm to SDE function estimation, establishing a new research direction.
2. **Efficiency of amortized inference**: Pretrain once, deploy repeatedly — substantially reducing the computational cost of SDE discovery.
3. **Function-space output**: The model outputs complete drift and diffusion functions (rather than parameters), affording greater flexibility.
4. **Cross-domain generalization**: A single pretrained model handles SDEs from domains ranging from physics to finance.
5. **Practical fine-tuning mechanism**: Building on already-competitive zero-shot performance, fine-tuning provides additional consistent gains.
6. **Title change across versions**: The v1 title was "Foundation Inference Models for SDEs: A Transformer-based Approach for Zero-shot Function Estimation," reflecting an evolution in the paper's emphasis.

## Limitations & Future Work

1. **Dimensionality constraint**: The current approach is limited to low-dimensional SDEs; high-dimensional SDEs (e.g., partial differential equations) require new methods.
2. **Pretraining distribution mismatch**: The SDE distribution used during pretraining may not align with that of real-world target systems.
3. **Absence of theoretical guarantees**: Theoretical guarantees for in-context learning (e.g., convergence rates, generalization bounds) have not yet been established.
4. **Jump diffusions and non-Markovian processes**: The current framework handles only standard SDEs and does not support jump processes or memory effects.
5. **Computational resources**: The pretraining stage requires substantial computational resources.
6. **Interpretability**: The black-box nature of the Transformer makes it difficult to assign physical meaning to the estimated functions.

## Related Work & Insights

- **Neural SDEs** (Kidger et al., 2021): Train a Neural SDE separately for each system; the pretraining approach proposed here eliminates this requirement.
- **SDE symbolic regression** (Brunton et al., 2016): Methods such as SINDy require presupposed functional forms.
- **In-context learning** (Brown et al., 2020): In-context learning demonstrated by the GPT series is introduced here into scientific computing.
- **Neural Operators** (Lu et al., 2021): Operator learning methods such as DeepONet provide the framework for function-space mapping adopted in this work.
- **Amortized inference** (Gershman & Goodman, 2014): The general framework of amortized inference.
- **Foundation models for science** (Bommasani et al., 2021): The broader trend of foundation models in scientific domains.

## Rating

- **Novelty**: ★★★★★ — Introducing foundation models to SDE discovery constitutes an entirely new paradigm.
- **Theoretical Depth**: ★★★☆☆ — Empirically oriented; theoretical analysis is limited.
- **Experimental Thoroughness**: ★★★★☆ — Covers synthetic and real-world data with comparisons against multiple baselines and comprehensive ablation studies.
- **Value**: ★★★★★ — Zero-shot SDE discovery carries significant practical value for scientific computing.
- **Writing Quality**: ★★★★☆ — Motivation is clearly articulated; experiments are comprehensive; structure is well-organized.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Neural Stochastic Flows: Solver-Free Modelling and Inference for SDE Solutions](neural_stochastic_flows_solver-free_modelling_and_inference_for_sde_solutions.md)
- [\[NeurIPS 2025\] TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning](tirex_zero-shot_forecasting_across_long_and_short_horizons_with_enhanced_in-cont.md)
- [\[AAAI 2026\] AirDDE: Multifactor Neural Delay Differential Equations for Air Quality Forecasting](../../AAAI2026/time_series/airdde_multifactor_neural_delay_differential_equations_for_air_quality_forecasti.md)
- [\[NeurIPS 2025\] Transformer Embeddings for Fast Microlensing Inference](transformer_embeddings_for_fast_microlensing_inference.md)
- [\[NeurIPS 2025\] Diffusion Transformers as Open-World Spatiotemporal Foundation Models](diffusion_transformers_as_open-world_spatiotemporal_foundation_models.md)

</div>

<!-- RELATED:END -->
