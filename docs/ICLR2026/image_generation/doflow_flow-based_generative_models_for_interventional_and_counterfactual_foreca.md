---
title: >-
  [Paper Note] DoFlow: Flow-based Generative Models for Interventional and Counterfactual Forecasting
description: >-
  [ICLR 2026][Image Generation][Causal Inference] This paper proposes DoFlow, a causal generative model based on continuous normalizing flows (CNF) that unifies observational, interventional…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Causal Inference"
  - "Continuous Normalizing Flows"
  - "Time Series Forecasting"
  - "Counterfactual Reasoning"
  - "Anomaly Detection"
date: 2026-05-08
content_hash: 03046bdd8e06f7d7
---

# DoFlow: Flow-based Generative Models for Interventional and Counterfactual Forecasting

**Conference**: ICLR 2026
**arXiv**: [2511.02137](https://arxiv.org/abs/2511.02137)  
**Code**: [Available](https://github.com/StatFusion/DoFlow_Causal_Time_Series)  
**Area**: Generative Modeling
**Keywords**: Causal Inference, Continuous Normalizing Flows, Time Series Forecasting, Counterfactual Reasoning, Anomaly Detection

## TL;DR

This paper proposes DoFlow, a causal generative model based on continuous normalizing flows (CNF) that unifies observational, interventional, and counterfactual time series forecasting over a causal DAG. The model additionally supports anomaly detection via explicit likelihood estimation, and is validated on both synthetic and real-world medical datasets.

## Background & Motivation

Time series forecasting is a central problem in statistics and machine learning. Conventional forecasting models (ARIMA, LSTM, Transformer, etc.) are **purely observational**—they learn historical correlations and extrapolate. However, practical applications often require answering causal "what if" questions:

**Interventional queries**: "How does the system evolve if a control variable is changed?" For example, in a hydropower plant, how does the power output change when the turbine control signal is modified? An observational predictor conditioned on fixed history cannot simulate the effect of different control strategies.

**Counterfactual queries**: "Had a different intervention been applied at the time, how would the already-observed trajectory have changed?" In healthcare, for instance, after observing a patient's treatment and outcome trajectory, one may ask whether a different medication regimen would have led to a better outcome for that specific patient.

**Core Challenge**: Existing causal generative models are primarily designed for static data, and no general framework exists for causal counterfactual forecasting over time series. A model that combines causal structure with generative capability is needed.

## Method

### Overall Architecture

DoFlow models a $K$-dimensional multivariate time series whose nodes are topologically ordered on a causal DAG. Each node $X_{i,t}$ depends on its own history $X_{i,t-}$ and the history of its parents $X_{\text{pa}(i),t-}$, defined via a structural causal model (SCM):

$$X_{i,t} := f_i(X_{i,t-}, X_{\text{pa}(i),t-}, U_{i,t})$$

where $U_{i,t}$ are independent exogenous noise variables.

The sequence is partitioned into a context window $\{1,\ldots,\tau\}$ (conditioning) and a forecast window $\{\tau+1,\ldots,T\}$ (prediction target).

### Key Designs

**1. Time-Conditioned Continuous Normalizing Flow (CNF)**

A CNF is learned for each DAG node $i$, shared across time steps, and defines a continuous transformation between the data distribution and a base distribution $\mathcal{N}(0,1)$ via a Neural ODE:

$$\frac{dx_{i,t}(s)}{ds} = v_i(x_{i,t}(s), s; H_{i,t-1}), \quad s \in [0,1]$$

where $H_{i,t-1}$ is the historical hidden state aggregated by an RNN:

$$H_{i,t-1} = \text{concat}(h_{i,t-1}, h_{\text{pa}(i),t-1})$$

**2. Forward Process (Encoding)**: Maps the observed value $x_{i,t}^F$ to a latent code $z_{i,t}^F$:

$$z_{i,t}^F = \Phi_\theta(x_{i,t}^F; H_{i,t-1}^F) = x_{i,t}^F + \int_0^1 v_i(x_{i,t}(s), s; H_{i,t-1}^F)\, ds$$

**3. Inverse Process (Decoding)**: Generates predictions from the latent space:

$$\hat{x}_{i,t} = \Phi_\theta^{-1}(z_{i,t}; \hat{H}_{i,t-1}) = z_{i,t} - \int_0^1 v_i(x_{i,t}(s), s; \hat{H}_{i,t-1})\, ds$$

**4. Three Forecasting Modes**:

- **Observational Forecasting**: Sample $z \sim \mathcal{N}(0,1)$ and decode in topological order.
- **Interventional Forecasting**: For $(i,t) \in \mathcal{I}$, directly set $\hat{x}_{i,t} \leftarrow \gamma_{i,t}$; non-intervened nodes are decoded normally, but their hidden states incorporate the intervention.
- **Counterfactual Forecasting** (three-step procedure):
    - **Abduction**: Encode factual observations into $z_{i,t}^F$ using factual hidden states $H^F$.
    - **Action**: Apply the intervention.
    - **Prediction**: Decode $z_{i,t}^F$ using counterfactual hidden states $\hat{H}^{CF}$ to obtain the counterfactual trajectory.

**5. Likelihood-Based Anomaly Detection**: The CNF provides explicit log-density:

$$\log p_{\theta}(\hat{x}_{\tau+1:T} | \hat{H}_\tau) = \sum_{t=\tau+1}^T \left[\log q(z_t) + \int_0^1 \nabla \cdot v_\theta(x_t(s), s; \hat{H}_{t-1})\, ds\right]$$

Anomalous contexts result in predicted trajectories with low density.

### Loss & Training

The model is trained using the **Conditional Flow Matching (CFM)** loss, with straight-line interpolation as the reference path:

$$\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}\left[\frac{1}{K(T-\tau)}\sum_{i,t} \|v_i(\phi(x_{i,t},z;s), s; H_{i,t-1}) - (z - x_{i,t})\|_2^2\right]$$

During training, hidden states are updated autoregressively using ground-truth observations (teacher forcing).

**Theoretical Guarantee** (Corollary 4.5): Under monotone SCM and exact training assumptions, DoFlow's counterfactual predictions exactly recover the true counterfactual trajectories.

## Key Experimental Results

### Main Results

**Synthetic Data: Observational / Interventional / Counterfactual Forecasting RMSE (Multiple DAG Structures)**

| Method | Tree-Obs | Tree-Int | Tree-CF | Diamond-Obs | Diamond-Int | Diamond-CF |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| DoFlow | **0.57** | **0.54** | **0.11** | **0.55** | **0.57** | **0.12** |
| GRU | 0.65 | 1.01 | NA | 0.58 | 0.94 | NA |
| TFT | 0.58 | 0.97 | NA | 0.63 | 1.18 | NA |
| TiDE | 0.60 | 1.15 | NA | 0.50 | 1.05 | NA |

**Key Observations**:
- DoFlow substantially outperforms baselines on interventional forecasting (RMSE gap ~0.5), as baselines lack causal structure.
- Counterfactual forecasting is a capability unique to DoFlow; baselines cannot perform it.
- DoFlow also performs robustly under nonlinear non-additive (NLNA) settings.

**Real-World Data: Hydropower Plant Interventional Forecasting**

DoFlow successfully predicts downstream signal changes under different turbine control strategies, with causal structure consistent with physical domain knowledge.

**Real-World Data: Cancer Treatment Effect Estimation**

| Method | RMSE of PEHE ↓ |
|--------|:---:|
| DoFlow | **Best** |
| CRN | Second best |

### Ablation Study

- Additive vs. nonlinear non-additive noise models: DoFlow performs well under both settings.
- Different DAG structures (Chain / Tree / Diamond / FC-Layer): consistently strong performance.
- Anomaly Detection AUROC: DoFlow effectively detects anomalies on both synthetic data and the real-world hydropower dataset.

### Key Findings

1. DoFlow unifies observational, interventional, and counterfactual queries within a single model, constituting the first general-purpose framework for counterfactual time series forecasting.
2. The invertibility of CNFs is central: encoding → modifying conditioning → decoding naturally supports counterfactual inference.
3. Intervention effects propagate through hidden states, enabling downstream nodes to naturally perceive upstream interventions.
4. Explicit likelihood density provides anomaly detection as an additional capability.

## Highlights & Insights

- **Unified Framework**: A single model simultaneously supports all three types of causal queries, resulting in a naturally elegant architectural design.
- **Causal Alignment of CNFs**: The invertibility of normalizing flows perfectly aligns with the abduction–action–prediction three-step procedure in causal inference.
- **Theoretical Support**: Exact counterfactual recovery is proven under monotone SCMs.
- **Explicit Likelihood**: Anomaly detection is obtained at no additional modeling cost, enhancing practical utility.
- **RNN + CNF Combination**: RNNs encode temporal context while CNFs handle uncertainty and invertible mappings.

## Limitations & Future Work

- The causal DAG is assumed to be known; in practice, causal discovery may be required.
- Instantaneous causal effects within the same time step are not modeled (all causal influences are assumed to have at least a one-step lag).
- The theoretical counterfactual recovery guarantee requires SCM monotonicity and exact training assumptions.
- Each node has an independent CNF; scalability to graphs with large node counts warrants further investigation.
- Counterfactual ground truth is unobservable in real-world settings, limiting quantitative evaluation to synthetic data.
- In-depth comparison with more sophisticated methods for causal effect estimation in time series remains to be conducted.

## Related Work & Insights

- **vs. Traditional Causal Effect Methods**: Those methods focus on short-term expected differences under discrete actions, whereas DoFlow supports continuous-variable interventions at arbitrary time steps.
- **vs. Static Causal Generative Models** (Javaloy et al.): DoFlow extends causal generation to time series, capturing cross-temporal causal dependencies.
- **vs. Modern Forecasters** (TFT / TiDE / TSMixer): These are observational models and cannot answer causal questions.
- **Healthcare Application Potential**: Individualized treatment comparison, drug dosage optimization, and clinical decision support.

## Rating

| Dimension | Score |
|-----------|:---:|
| Novelty | ★★★★★ |
| Theoretical Depth | ★★★★☆ |
| Experimental Thoroughness | ★★★★☆ |
| Value | ★★★★☆ |
| Writing Quality | ★★★★☆ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Conditionally Whitened Generative Models for Probabilistic Time Series Forecasting](conditionally_whitened_generative_models_for_probabilistic_time_series_forecasti.md)
- [\[ICLR 2026\] FlowCast: Trajectory Forecasting for Scalable Zero-Cost Speculative Flow Matching](flowcast_trajectory_forecasting_for_scalable_zero-cost_speculative_flow_matching.md)
- [\[ICLR 2026\] Laplacian Multi-scale Flow Matching for Generative Modeling](laplacian_multi-scale_flow_matching_for_generative_modeling.md)
- [\[ICML 2026\] Barriers to Counterfactual Credit Attribution for Autoregressive Models](../../ICML2026/image_generation/barriers_to_counterfactual_credit_attribution_for_autoregressive_models.md)
- [\[ICLR 2026\] Beyond Confidence: The Rhythms of Reasoning in Generative Models](beyond_confidence_the_rhythms_of_reasoning_in_generative_models.md)

</div>

<!-- RELATED:END -->
