---
title: >-
  [Paper Note] GST-UNet: A Neural Framework for Spatiotemporal Causal Inference with Time-Varying Confounding
description: >-
  [NeurIPS 2025][Causal Inference][Spatiotemporal causal inference] This paper proposes GST-UNet, which integrates a U-Net spatiotemporal encoder with iterative G-computation to estimate location-specific conditional avera…
tags:
  - "NeurIPS 2025"
  - "Causal Inference"
  - "Spatiotemporal causal inference"
  - "G-computation"
  - "UNet"
  - "time-varying confounding"
  - "potential outcomes"
date: 2026-05-08
content_hash: 49a341df3bbed16b
---

# GST-UNet: A Neural Framework for Spatiotemporal Causal Inference with Time-Varying Confounding

**Conference**: NeurIPS 2025
**arXiv**: [2502.05295](https://arxiv.org/abs/2502.05295)  
**Code**: [moprescu/GSTUNet](https://github.com/moprescu/GSTUNet)  
**Area**: Causal Inference
**Keywords**: Spatiotemporal causal inference, G-computation, UNet, time-varying confounding, potential outcomes

## TL;DR
This paper proposes GST-UNet, which integrates a U-Net spatiotemporal encoder with iterative G-computation to estimate location-specific conditional average potential outcomes (CAPOs) from a **single spatiotemporal observational trajectory**. The framework simultaneously handles interference, spatial confounding, temporal carry-over effects, and time-varying confounding, and is validated on a real-world causal analysis of wildfire smoke effects on respiratory hospitalization rates in California.

## Background & Motivation
**Background**: Causal effect estimation from spatiotemporal observational data is critical in public health, environmental science, and policy evaluation, where randomized experiments are typically infeasible, necessitating causal inference from observational data.

**Limitations of Prior Work**:
   - **Classical methods** (DID, Synthetic Control, MSM) rely on strong assumptions such as parallel trends and no-interference, and cannot handle spatial spillover effects.
   - **Deep learning predictive models** (CNN/RNN/Transformer) capture complex spatiotemporal patterns but lack causal adjustment and cannot estimate counterfactuals.
   - **Longitudinal causal inference methods** (MSM, G-computation, and RNN/Transformer-based extensions) assume independent time series (e.g., distinct patients) and cannot model cross-unit interactions.
   - **Existing spatiotemporal causal models** (e.g., Tec et al.) address only static exposures and do not handle time-varying confounding or interference.

**Key Challenge**: Spatiotemporal data simultaneously exhibit ① spatial interference (interventions in neighboring units affect local outcomes), ② spatial confounding, ③ temporal carry-over effects, and ④ time-varying confounding (covariates influenced by past interventions that in turn affect future interventions), with typically only a **single spatiotemporal trajectory** available.

**Goal**: To simultaneously address all four challenges under a single spatiotemporal trajectory and provide theoretically guaranteed causal effect estimates.

**Key Insight**: Combining classical iterative G-computation (recursive regression to eliminate time-varying confounding) with U-Net spatiotemporal representation learning, and exploiting a representation-based time invariance assumption to pool training samples from a single trajectory.

**Core Idea**: Learn a time-invariant history embedding $\phi(H_{1:t}, A_t)$ such that, conditioned on this embedding, the transition distribution is independent of $t$. This renders the single trajectory decomposable into exchangeable prefix segments, enabling iterative G-computation to recursively eliminate time-varying confounding.

## Method

### Overall Architecture
1. **Data structure**: Each location $s$ on an $N_X \times N_Y$ spatial grid has observations $(X_{s,t}, A_{s,t}, Y_{s,t})$ at each time step $t$, along with static features $V_s$. The goal is to estimate the CAPO $E[Y_{t+\tau}[a] \mid H_{1:t}]$ given history $H_{1:t}$ and intervention sequence $a_{t:t+\tau-1}$.
2. **Prefix construction**: From a single trajectory of length $T$, a prefix $P_t^\tau = (X_{1:t+\tau}, A_{1:t+\tau}, Y_{1:t+\tau}, V)$ is extracted for each $t \in \{1, \ldots, T-\tau\}$, yielding $T-\tau$ overlapping but non-independent training segments.
3. **Identification → Estimation → Inference**: Identification is established via Assumptions 1 and 2 (Theorem 1); estimation follows via iterative G-computation (Theorem 2); the GST-UNet architecture instantiates the estimator.

### Key Designs
1. **Representation-Based Time Invariance (Assumption 2)**: There exists an embedding function $\phi: \mathcal{H} \times \mathcal{A} \to \mathcal{Z}$ such that, conditioned on $z = \phi(H_{1:t}, A_t)$, the distribution of $(X_{t+1}, Y_{t+1})$ does not depend on $t$. This is weaker than classical stationarity—it does not require the marginal distribution to be time-invariant, only the conditional transition mechanism after embedding. This allows prefixes from different time points to be treated as conditionally exchangeable and pooled for regression training.

2. **Iterative G-Computation (Theorem 1)**: For intervention sequences with $\tau \geq 2$, directly conditioning on history and interventions introduces time-varying confounding bias. The following recursion is defined:

    - $Q_\tau(H, A) = E[Y_{t+\tau} \mid \phi(H_{1:t+\tau-1}, A_{t+\tau-1})]$ (direct regression on observed outcomes)
    - $Q_k(H, A) = E[Q_{k+1}(H^a, a_{t+k}) \mid \phi(H_{1:t+k-1}, A_{t+k-1})]$ (recursive regression on pseudo-outcomes)
    - Finally, $Q_1(h_{1:t}, a_t) = E[Y_{t+\tau}[a] \mid H_{1:t}]$

   The procedure proceeds backward from $Q_\tau$ to $Q_1$, using predictions from the previous step as pseudo-outcomes in forward predictions over intervened histories.

3. **U-Net Spatiotemporal Encoder**:

    - **Spatial module**: A U-Net encoder–decoder architecture with skip connections, progressively downsampling and upsampling the spatial grid.
    - **Temporal module**: ConvLSTM layers integrated within the encoder maintain hidden states across time steps while aggregating spatial information via convolution; static covariates $V$ are subsequently concatenated as additional channels.
    - **Attention gating**: Attention gates in the decoder selectively highlight relevant spatial regions, refining skip connections.
    - The encoder outputs a $d_h \times N_X \times N_Y$ feature map capturing interference, spatial confounding, and static features.

4. **Neural Causal Module (G-heads)**: $\tau$ G-computation heads are attached to the U-Net feature map; each $Q_k$ is a lightweight convolutional module or feedforward network. $Q_\tau$ is supervised against true observed outcomes; $Q_{k < \tau}$ is supervised against pseudo-outcomes generated by the preceding step (with detached forward passes to prevent gradient backpropagation).

### Loss & Training
**Joint loss**:
$$\mathcal{L}(\theta; e) = \frac{1}{\tau} \sum_{k=1}^{\tau} \alpha_k^{(e)} \mathcal{L}_k(\theta)$$
where $\mathcal{L}_k$ is the MSE loss for each G-head and $\alpha_k^{(e)}$ are epoch-dependent weights.

**Curriculum Training**:
- Phase $p(e) = \min\{\tau, \lceil e / e_c \rceil\}$, where $e_c$ is the curriculum cycle hyperparameter.
- Initial phase ($e \leq e_c$): only $Q_\tau$ is trained (with true supervision); $\alpha_\tau = 1$, all others are 0.
- Incremental activation: one additional G-head (from $Q_{\tau-1}$ forward) is activated every $e_c$ epochs.
- Final stage: all heads are weighted equally at $1/\tau$.
- **Design Motivation**: Prevents early inaccurate $Q_\tau$ predictions from propagating noisy pseudo-outcomes that cause preceding G-heads to overfit.

## Key Experimental Results

### Synthetic Experiments: RMSE ($\tau=5$, varying confounding strength $\beta_1$)

| Method | $\beta_1=0.0$ | $\beta_1=0.5$ | $\beta_1=1.0$ | $\beta_1=1.5$ | $\beta_1=2.0$ |
|--------|--------------|--------------|--------------|--------------|--------------|
| UNet+ (no causal adjustment) | **0.28** | 0.36 | 0.54 | 0.71 | 0.81 |
| STCINet | 0.29 | 0.38 | 0.62 | 0.80 | 0.90 |
| IPWUNet | 0.60 | 0.58 | 0.58 | 0.59 | 0.59 |
| GST-UNet w/o Attention | 0.50 | 0.46 | 0.51 | 0.45 | 0.47 |
| GST-UNet w/o Curriculum | 0.69 | 0.64 | 0.63 | 0.61 | 0.61 |
| **GST-UNet** | 0.33 | **0.35** | **0.40** | **0.44** | **0.40** |

### Synthetic Experiments: RMSE ($\tau=10$, varying confounding strength $\beta_1$)

| Method | $\beta_1=0.0$ | $\beta_1=0.5$ | $\beta_1=1.0$ | $\beta_1=1.5$ | $\beta_1=2.0$ |
|--------|--------------|--------------|--------------|--------------|--------------|
| UNet+ | **0.28** | 0.61 | 1.18 | 1.45 | 1.71 |
| STCINet | 0.31 | 0.68 | 1.25 | 1.47 | 1.60 |
| IPWUNet | 0.78 | 0.80 | 0.96 | 1.19 | 1.08 |
| **GST-UNet** | 0.38 | **0.55** | **0.68** | **0.73** | **0.85** |

### Ablation Study

| Component | Effect |
|-----------|--------|
| Remove Curriculum | RMSE increases by 50%–70% at $\tau=5$; particularly pronounced degradation under low confounding |
| Remove Attention | RMSE increases by 10%–30%; smaller impact when local dynamics dominate |
| Remove G-computation (i.e., UNet+) | Optimal under no confounding, but RMSE deteriorates sharply as confounding increases ($0.28 \to 1.71$) |
| IPW in place of G-computation | Exhibits substantial bias even without confounding due to inability to correct for spatial interference |

### Real-World Data: Causal Effect of the 2018 California Camp Fire on Respiratory Hospitalizations

| Method | Estimated Excess Hospitalizations (10 days) | 95% Bootstrap CI | Notes |
|--------|---------------------------------------------|------------------|-------|
| **GST-UNet** | ~4,650 (465/day) | [1888, 6535] | Stable and consistent with prior knowledge |
| UNet+ | ~3,981 | [−899, 5202] | CI includes negative values |
| STCINet | ~88 | [−3077, 3281] | Highly unstable, near zero |
| IPWUNet | ~20,500 | — | Implausibly large overestimate (insufficient rare-event support) |

Reference: Letellier et al. report 259 excess cases/day, but over a longer and lower-intensity exposure window (Nov 8–Dec 5).

### Key Findings
- When time-varying confounding is absent ($\beta_1=0$), a simple UNet+ predictive model suffices and G-computation introduces additional noise.
- As confounding strength increases, UNet+ and STCINet degrade substantially, while GST-UNet's advantage becomes increasingly pronounced (RMSE reduction of 21%–47% at $\tau=10$, $\beta_1=2.0$).
- Curriculum training is especially critical for long horizons ($\tau=10$).
- On the real wildfire data, GST-UNet is the only method yielding reasonable and stable estimates.

## Highlights & Insights
- **Theory–practice integration**: The paper provides a rare end-to-end connection among an identification theorem (Theorem 1), a consistency theorem (Theorem 2), and a neural network implementation.
- **Representation-Based Time Invariance**: Weaker than classical stationarity, permitting non-stationary processes as long as a time-invariant embedding exists.
- **Curriculum training**: A simple yet critical strategy that resolves training instability arising from the shared encoder and recursive pseudo-outcomes.
- **Single-trajectory causal inference**: The framework remains operational in the most data-scarce setting, where only one spatiotemporal trajectory is available.

## Limitations & Future Work
- **Verification of Assumption 2**: Representation-Based Time Invariance is difficult to validate empirically, and diagnostic tools to assess whether the learned $\phi$ genuinely satisfies the assumption are lacking.
- **Computational cost**: Joint training of $\tau$ G-heads with a shared encoder incurs substantially increasing computational cost as $\tau$ grows.
- **Spatial resolution**: Experiments use $40 \times 44$ and $64 \times 64$ grids; scalability of the U-Net to higher resolutions remains to be verified.
- **Binary treatment**: The current framework focuses on binary treatments; adaptation to continuous treatments requires additional modifications.
- **Causal assumptions**: Sequential unconfoundedness is not testable from observational data, and unmeasured confounding may still introduce bias.

## Related Work & Insights
- **vs. UNet+ (Tec et al. 2022)**: Shares the U-Net architecture but addresses only static exposures and spatial confounding, without handling time-varying confounding or interference.
- **vs. STCINet (Ali et al. 2024)**: Estimates direct and indirect effects but does not model time-varying confounding; the G-computation approach in this paper is more principled.
- **vs. longitudinal causal methods (Bica et al., Melnychuk et al.)**: Assume independent time series and cannot handle spatial spillovers; GST-UNet naturally captures spatial dependencies via the convolutional encoder.
- **vs. IPW methods**: Unstable under rare events and spatial interference; the regression-based approach of GST-UNet is more robust.
- **Broader inspiration**: The paradigm of combining iterative G-computation with deep learning can be generalized to other structured causal inference problems, such as graph-based causal inference.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First framework to end-to-end integrate G-computation with spatiotemporal deep learning, with solid theoretical contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Synthetic experiments provide well-controlled variable analysis; the real-world case study is practically meaningful, though only a single real dataset is used.
- **Writing Quality**: ⭐⭐⭐⭐ The theory-to-implementation connection is clear, with a complete notation system, though the density of mathematical exposition is high.
- **Value**: ⭐⭐⭐⭐ Directly applicable to environmental health, policy evaluation, and related domains, with strong generalizability of the framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Practical do-Shapley Explanations with Estimand-Agnostic Causal Inference](practical_do-shapley_explanations_with_estimand-agnostic_causal_inference.md)
- [\[ICML 2026\] Controllable Generative Sandbox for Causal Inference](../../ICML2026/causal_inference/controllable_generative_sandbox_for_causal_inference.md)
- [\[NeurIPS 2025\] Domain-Adapted Granger Causality for Real-Time Cross-Slice Attack Attribution in 6G Networks](domain-adapted_granger_causality_for_real-time_cross-slice_attack_attribution_in.md)
- [\[ICLR 2026\] Efficient Ensemble Conditional Independence Test Framework for Causal Discovery](../../ICLR2026/causal_inference/efficient_ensemble_conditional_independence_test_framework_for_causal_discovery.md)
- [\[NeurIPS 2025\] Conformal Prediction for Causal Effects of Continuous Treatments](conformal_prediction_for_causal_effects_of_continuous_treatments.md)

</div>

<!-- RELATED:END -->
