---
title: >-
  [Paper Note] Controllable Financial Market Generation with Diffusion Guided Meta Agent
description: >-
  [AAAI 2026][financial market generation] This paper proposes the Diffusion Guided Meta Agent (DigMA), which formalizes controllable financial market generation as a conditional generation task. A conditional diffusion mo…
tags:
  - "AAAI 2026"
  - "financial market generation"
  - "diffusion model"
  - "limit order book"
  - "controllable generation"
  - "multi-agent simulation"
date: 2026-05-08
content_hash: 61bc7c7c528ac4cd
---

# Controllable Financial Market Generation with Diffusion Guided Meta Agent

**Conference**: AAAI 2026
**arXiv**: [2408.12991](https://arxiv.org/abs/2408.12991)  
**Code**: [microsoft/TimeCraft](https://github.com/microsoft/TimeCraft)  
**Area**: Others (Financial AI / Generative Models)
**Keywords**: financial market generation, diffusion model, limit order book, controllable generation, multi-agent simulation

## TL;DR

This paper proposes the Diffusion Guided Meta Agent (DigMA), which formalizes controllable financial market generation as a conditional generation task. A conditional diffusion model captures the dynamics of market states (time-varying distribution parameters of mid-price returns and order arrival rates), while a Meta Agent with financial economics priors generates order flow under the guidance of the controller. DigMA outperforms existing methods in both controllability and generation fidelity.

## Background & Motivation

### State of the Field
Financial markets are data-intensive and complex systems, where orders are the most fundamental event units (analogous to words in language or pixels in images). Through order flow, researchers can investigate market microstructure and interaction mechanisms. **Order flow modeling** thus constitutes the most fundamental generative task in finance.

### Limitations of Prior Work

**Rule-based multi-agent methods** (e.g., RFD, RMSC): rely on overly simplified market assumptions and hand-crafted rules; not trained on real data; limited simulation fidelity.

**Learning-based agent methods** (e.g., LOBGAN): predict the next order from historical order flow, but hundreds of orders may occur within a single minute and a trading day spans hundreds of minutes, making it difficult to capture long-range dependencies; tend to focus on local distributions while neglecting global dynamics.

**Complete absence of controllability**: no existing method allows specifying a target scenario (e.g., extreme events, high/low volatility) to generate corresponding order flow, precluding scenario experiments and counterfactual analysis.

### Root Cause
Establishing an effective connection between macro-level control objectives (e.g., daily return, volatility) and micro-level orders is extremely challenging because:
- Real order flow sequences are extremely long and irregular, making it impractical to apply diffusion models directly to raw order-level data.
- The signal-to-noise ratio of individual orders is very low, making it unrealistic to map macro-level control signals to each micro-level order.

## Method

### Overall Architecture

DigMA adopts a **two-stage design** to avoid applying the diffusion model directly to raw order flow:
- **Meta Controller**: a conditional diffusion model that learns the distribution $q(\mathbf{x}|c)$ of market state $\mathbf{x}$ given scenario $c$.
- **Order Generator**: comprises a simulated exchange and a Meta Agent that incorporates financial economics priors and generates orders via stochastic processes under the guidance of the Meta Controller.

### Key Designs

#### 1. **Problem Formulation: Controllable Financial Market Generation**

Formulated as a conditional generation task with two objectives:
- **Controllability objective**: minimize the discrepancy between the target metric $a$ and the metric $\tilde{a}$ computed from the generated order flow:
  $$\min_{\mathcal{M}} \mathbb{E}_{a,\tilde{a}}[\|\tilde{a} - a\|^2]$$
- **Fidelity objective**: minimize the divergence between the stylized facts distributions of real and generated order flows:
  $$\min_{\mathcal{M}} \mathcal{D}(p(\mathcal{F}'(\tilde{\boldsymbol{O}})) \| p(\mathcal{F}'(\boldsymbol{O}})))$$

Control metrics include: daily return, daily amplitude, and intraday volatility.

#### 2. **Meta Controller: Conditional Diffusion Model for Market State Dynamics**

**Market state definition**: **minute-level mid-price returns $\mathbf{r}$** and **order arrival rates $\boldsymbol{\lambda}$** are extracted from real order flow to define the market state $\mathbf{x} = \{\mathbf{r}, \boldsymbol{\lambda}\}$.

**Diffusion training**: adopts the $\epsilon$-parameterization of DDPM with training loss:
$$L_M = \mathbb{E}_{\mathbf{x},\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I}),n}[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_n, n)\|^2]$$

**Conditional control**: two types of control encoders are designed:
- **Discrete control encoder**: maps target conditions to categorical labels over predefined bins, embedded via a learnable embedding matrix.
- **Continuous control encoder**: maps real-valued conditions to latent vectors via a fully connected network.

**Classifier-Free Guidance** is employed for control: during training, condition information is randomly dropped (dropout probability 0.5); during sampling, the guided score is computed as:
$$\tilde{\boldsymbol{\epsilon}}_{\theta,\phi}(\mathbf{x}_n, n, \mathbf{c}) = (1-s)\boldsymbol{\epsilon}_\theta(\mathbf{x}_n, n) + s\boldsymbol{\epsilon}_\theta(\mathbf{x}_n, n, \boldsymbol{\phi}(\mathbf{c}))$$
where $s$ controls guidance strength.

**Model backbone**: a U-Net with 1D convolutional layers as the primary building block; parameters are shared across diffusion timesteps; DDIM sampling is used for efficiency.

#### 3. **Order Generator: Meta Agent with Financial Economics Priors**

The Meta Agent represents the aggregated behavior of all market participants. Core procedure:

**Wake-up process**: within each trading minute $t$, the agent "wakes up" at exponentially distributed intervals $\delta_i$ with parameter $\lambda_t$ provided by the Meta Controller.

**Actor Agent initialization**: each wake-up event instantiates a heterogeneous agent with three components (fundamentalist / chartist / noise), whose weights are independently sampled from exponential distributions with expected value ratios of 10:1.5:1.

**Return estimation**: a weighted average of the return signals from the three components:
$$\hat{r} = \frac{g_f r_t + g_c \bar{r} + g_n r_\sigma}{g_f + g_c + g_n}$$
where $r_t$ comes from the Meta Controller, $\bar{r}$ from the simulated exchange history, and $r_\sigma$ is Gaussian noise.

**CARA utility optimization**: based on the estimated return, a demand function $u(p) = \frac{\ln(\hat{p}_t/p)}{aVp}$ is computed ($a$: risk aversion coefficient, $V$: historical price volatility) to determine the minimum acceptable price $p_l$; the order price is then uniformly sampled as $p_i \sim \mathcal{U}(p_l, \hat{p}_t)$.

Design Motivation: this agent structure integrates the heterogeneous trader model from classical financial economics, allowing generated orders to naturally exhibit market microstructure properties.

### Loss & Training

- 10 training epochs per dataset; 200 diffusion steps.
- AdamW optimizer; learning rate $1\times10^{-5}$; mini-batch size 256.
- Condition dropout probability: 0.5.
- Data preprocessing: z-score normalization; minute-level mid-price returns and order arrival rates extracted from tick-by-tick data.

## Key Experimental Results

### Datasets
Two tick-by-tick order book datasets from the Chinese A-share market: A-Main (316K day×stock pairs) and ChiNext (122K day×stock pairs), covering the full year 2020 from the Shenzhen Stock Exchange.

### Main Results 1: Controllability Evaluation

| Target | Method | Control | A-Main Min MSE | ChiNext Min MSE |
|--------|--------|---------|---------------|----------------|
| Return | No Control | — | 0.529 | 0.684 |
| Return | Discrete | Discrete | 0.228 | 0.243 |
| Return | **Continuous** | Continuous | **0.161** | **0.342** |
| Amplitude | No Control | — | 0.268 | 0.427 |
| Amplitude | **Continuous** | Continuous | **0.054** | **0.110** |
| Volatility | No Control | — | 0.021 | 0.029 |
| Volatility | **Continuous** | Continuous | **0.011** | **0.028** |

### Main Results 2: Fidelity Evaluation (KL Divergence ↓)

| Model | MinR | RetAC | VolC | OIR |
|-------|------|-------|------|-----|
| RFD | 1.198 | 5.010 | 0.839 | 0.015 |
| RMSC | 2.640 | 10.170 | 1.237 | 0.563 |
| LOBGAN | 0.151 | **1.903** | 1.101 | 0.309 |
| **DigMA** | **0.084** | 2.781 | **0.273** | **0.009** |

### Ablation Study: High-Frequency Trading RL Evaluation

| Training Environment | Ret(%)↑ | Vol↓ | SR↑ | MDD(%)↓ |
|---------------------|---------|------|-----|---------|
| Replay | 0.009 | 0.413 | 0.014 | 1.133 |
| RFD | 0.000 | 0.159 | 0.011 | 0.803 |
| DigMA-c (no control) | 0.015 | **0.147** | 0.006 | **0.715** |
| **DigMA** | **0.029** | 0.411 | **0.049** | 1.313 |

### Key Findings

1. **Controllability**: the continuous control encoder achieves the lowest MSE on most metrics, reducing error by 3–10× compared to the no-control baseline.
2. **Fidelity**: DigMA achieves the lowest KL divergence on 3 out of 4 stylized facts; the sole exception, RetAC, is attributable to the autoregressive nature of LOBGAN.
3. **Downstream value**: RL trading agents trained in DigMA-generated environments achieve the highest daily return and Sharpe ratio (0.049 vs. best baseline 0.014).
4. **Computational efficiency**: DigMA generates each order in 0.017 ms, which is 100× faster than LOBGAN.

## Highlights & Insights

1. **The two-stage design elegantly bridges the macro–micro gap**: the diffusion model operates in a compressed market state space (minute-level returns and arrival rates) rather than raw order flow (tens of thousands of orders per day), neatly avoiding the issues of low signal-to-noise ratio and excessively long sequences.
2. **Integration of financial economics priors**: heterogeneous traders (fundamentalist / chartist / noise) + CARA utility + double-auction mechanism allow generated orders to naturally satisfy stylized facts.
3. **First application of Classifier-Free Guidance in finance**: enables the model to support both unconditional and conditional generation simultaneously.
4. **Downstream task validation confirms the practical value of the generated environment**: the performance gains of RL trading strategies demonstrate the quality of order flow generated by DigMA.

## Limitations & Future Work

1. **Single-asset generation only**: cross-asset correlations are not considered; future work should extend to joint multi-asset generation.
2. **Limited control metrics** (only return, amplitude, and volatility); finer-grained scenario control such as specific price trajectory patterns is not supported.
3. **Fixed Meta Agent structure**: the component weight distributions and utility function form of the heterogeneous traders are predefined.
4. **Data limited to Chinese A-shares**: generalizability to other markets (e.g., U.S. equities) has not been validated.

## Related Work & Insights

- DigMA represents a pioneering application of diffusion models to financial order flow generation.
- The evolution of financial market simulation from "rule-driven" → "data-driven" → the present paper's "diffusion-guided + economics priors" paradigm deserves attention.
- Insight: in complex system simulation, combining deep generative models with domain prior knowledge (rather than purely data-driven approaches) is an effective strategy for improving fidelity.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (first work combining diffusion models with financial economics priors for controllable order flow generation)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (comprehensive four-dimensional evaluation: controllability / fidelity / downstream tasks / efficiency)
- Writing Quality: ⭐⭐⭐⭐ (problem definition is clear, but methodological details are scattered across the main text and appendix)
- Value: ⭐⭐⭐⭐⭐ (significant practical implications for financial simulation research; code is open-sourced)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Compositional Diffusion with Guided Search for Long-Horizon Planning](../../ICLR2026/others/compositional_diffusion_long_horizon_planning.md)
- [\[AAAI 2026\] An Epistemic Perspective on Agent Awareness](an_epistemic_perspective_on_agent_awareness.md)
- [\[AAAI 2026\] HybriDLA: Hybrid Generation for Document Layout Analysis](hybridla_hybrid_generation_for_document_layout_analysis.md)
- [\[AAAI 2026\] Cash Flow Underwriting with Bank Transaction Data: Advancing MSME Financial Inclusion in Malaysia](cash_flow_underwriting_with_bank_transaction_data_advancing_msme_financial_inclu.md)
- [\[ICML 2026\] NonZero: Interaction-Guided Exploration for Multi-Agent Monte Carlo Tree Search](../../ICML2026/others/nonzero_interaction-guided_exploration_for_multi-agent_monte_carlo_tree_search.md)

</div>

<!-- RELATED:END -->
