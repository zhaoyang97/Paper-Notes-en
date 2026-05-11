---
title: >-
  [Paper Note] Coherent Multi-Agent Trajectory Forecasting in Team Sports with CausalTraj
description: >-
  [AAAI 2026][Time Series][multi-agent trajectory forecasting] This paper proposes CausalTraj — a temporally causal, likelihood-based multi-agent trajectory forecasting model that autoregressively models spatio-temporal in…
tags:
  - "AAAI 2026"
  - "Time Series"
  - "multi-agent trajectory forecasting"
  - "causal autoregressive model"
  - "team sports analytics"
  - "joint metrics"
  - "Gaussian mixture model"
date: 2026-05-08
content_hash: 652f722adbb0472a
---

# Coherent Multi-Agent Trajectory Forecasting in Team Sports with CausalTraj

**Conference**: AAAI 2026
**arXiv**: [2511.18248](https://arxiv.org/abs/2511.18248)
**Author**: Wei Zhen Teoh
**Code**: [wezteoh/causaltraj](https://github.com/wezteoh/causaltraj)
**Area**: Time Series
**Keywords**: multi-agent trajectory forecasting, causal autoregressive model, team sports analytics, joint metrics, Gaussian mixture model

## TL;DR

This paper proposes CausalTraj — a temporally causal, likelihood-based multi-agent trajectory forecasting model that autoregressively models spatio-temporal interactions among agents step by step. CausalTraj achieves state-of-the-art results on joint metrics (minJADE/minJFDE) across NBA, basketball, and football datasets while maintaining competitive per-agent accuracy.

## Background & Motivation

### State of the Field
Multi-agent trajectory forecasting is a central task in sports analytics and equally critical in domains such as autonomous driving and crowd navigation. The task requires predicting the joint future motion trajectories of multiple agents from historical observations. Key challenges include the stochastic and multimodal nature of the future, and the dependence of each agent's motion on the collective configuration of all other agents.

### Limitations of Prior Work
Dominant approaches (e.g., GroupNet, LED, MoFlow) are primarily evaluated and optimized using **per-agent metrics** (minADE, minFDE). These metrics assess each agent independently, allowing different agents to select optimal trajectories from different predicted scenarios. This leads to two fundamental problems:

**Lack of joint modeling**: Training losses supervise multi-hypothesis trajectory selection independently for each agent, without modeling which trajectories can be combined into plausible joint futures. A model may excel on per-agent metrics while producing poor joint predictions.

**Scenario incoherence**: Generated trajectories may appear individually reasonable but fail to form coherent multi-agent evolutions — for instance, ball movement inconsistent with player movement, or players lacking coordinated positional formations.

### Design Motivation
The authors argue that **learning the true joint distribution** is the core objective, and that scenario coherence and per-agent accuracy should naturally emerge as byproducts. Inspired by the success of causal autoregressive architectures in language models and 3D environment generation, CausalTraj adopts a temporally causal framework: it evolves spatial and inter-agent dynamics step by step, rather than compressing the entire future into a single global latent variable. Compared to methods that predict all future timesteps in parallel (which require all outputs to be conditionally independent given the latent), stepwise causal modeling reduces demands on latent capacity and is better suited to capturing complex joint dependencies.

## Method

### Problem Formulation
Consider $N$ interacting agents moving in 2D coordinate space. Given historical trajectories $X_{1:P} \in \mathbb{R}^{N \times P \times 2}$ ($P$ historical frames), the goal is to predict the joint future trajectory $\hat{X}_{P+1:T} \in \mathbb{R}^{N \times F \times 2}$ ($F = T - P$ future frames). A joint prediction is referred to as a "scenario," and the model estimates the conditional distribution $p(X_{P+1:T} \mid X_{1:P})$.

### Causal Likelihood Modeling Framework
CausalTraj factorizes the joint conditional distribution as a causal product over timesteps:

$$p(X_{P+1:T} \mid X_{1:P}) = \prod_{t=P}^{T-1} p(X_{t+1} \mid X_{1:t})$$

The model predicts the conditional distribution of per-step displacements $\Delta X_{t+1} = X_{t+1} - X_t$ rather than absolute positions. At inference, displacements are sampled from the predicted distribution and positions are updated recursively; during training, teacher forcing is used to compute all timesteps in parallel.

### Gaussian Mixture Output
The displacement distribution at each step is modeled as a mixture of $M=8$ Gaussian components:

$$p(\Delta X_{t+1} \mid X_{1:t}) = \sum_{m=1}^{M} \pi_{t+1,m} \mathcal{N}(\Delta X_{t+1}; \mu_{t+1,m}, \Sigma_{t+1,m})$$

At each timestep, the network outputs: $M$ mixture weight logits, $M \times N \times 2$ means, and $M \times N \times 3$ Cholesky parameters. The covariance matrix is assumed block-diagonal (conditionally independent across agents), but shared mixture weights still couple the outputs of different agents, expressing joint dependency structure.

The training loss is the negative log-likelihood plus entropy regularization on mixture weights to prevent component collapse:

$$\mathcal{L} = \mathcal{L}_{\text{NLL}} - \lambda_{\text{ent}} \mathcal{L}_{\text{ent}}, \quad \lambda_{\text{ent}} = 0.05$$

### Model Architecture

**Agent History Encoder**: Two variants are provided:
- **Causal PointNet Encoder**: Replaces PointNet's global max-pooling with lookback max-pooling, aggregating only over timesteps $t' \leq t$ to maintain temporal causality. Implemented via zero-padding and sliding window pooling, balancing parallel training efficiency with hierarchical feature aggregation.
- **Mamba2 Encoder**: Initial features are projected by a 2-layer MLP, then each agent sequence is independently processed through Mamba2 layers. Mamba2 combines compact state representations with attention-style context modeling and achieves superior performance across multiple datasets.

**Agent Embeddings**: Learned embeddings distinguish only three roles (two teams + ball), concatenated to encoded features and fused via MLP.

**Inter-Agent Relation Encoder**: The core innovation is the **Spatial Relation Transformer Encoder (SRTE)**. Building on standard self-attention, it explicitly encodes pairwise spatial geometry. Specifically, a pairwise "mesh" tensor $M_t[q,k] = [x_{q,t} - x_{k,t}; z_{q,t}; z_{k,t}]$ is constructed, containing relative displacements and encoded features. This tensor is used to compute key/value projections, enabling the attention mechanism to directly leverage precise Euclidean displacement information.

**Scenario Aggregation and Prediction Head**: Features from all agents are concatenated with position/velocity information and compressed via MLP, then concatenated across timesteps to form a scenario-level representation, which is passed through a 3-layer MLP to output Gaussian mixture parameters.

### Joint Evaluation Metrics
The paper emphasizes the use of joint metrics (minJADE, minJFDE). The key distinction from traditional per-agent metrics minADE/minFDE is that per-agent metrics allow each agent to select its best prediction from different scenario samples, whereas joint metrics require selecting a single globally optimal scenario from $k$ complete scenarios for evaluation. This directly measures the model's ability to generate coherent joint configurations.

## Key Experimental Results

### Table 1: NBA SportVU Dataset Results (meters, minADE₂₀/minFDE₂₀ and minJADE₂₀/minJFDE₂₀)

| Time | Metric Type | GroupNet | LED | MoFlow(joint) | MoFlow | CausalTraj(C-PN) | CausalTraj(Mamba2) |
|------|-------------|----------|-----|---------------|--------|-----------------|-------------------|
| 1.0s | per-agent | 0.25/0.32 | 0.21/0.27 | 0.28/0.39 | 0.18/0.25 | 0.15/0.21 | **0.14/0.20** |
| 2.0s | per-agent | 0.47/0.68 | 0.44/0.56 | 0.48/0.71 | **0.34/0.47** | 0.34/0.50 | 0.33/0.49 |
| 4.0s | per-agent | 0.95/1.22 | 0.81/1.10 | 0.89/1.32 | **0.71/0.87** | 0.77/1.01 | 0.77/1.02 |
| 1.0s | joint | 0.50/0.77 | 0.34/0.64 | 0.40/0.67 | 0.37/0.68 | 0.28/0.50 | **0.27/0.49** |
| 2.0s | joint | 1.04/1.91 | 0.78/1.55 | 0.81/1.61 | 0.80/1.61 | **0.62/1.18** | 0.62/1.21 |
| 4.0s | joint | 2.12/3.72 | 1.63/2.99 | 1.72/3.33 | 1.69/3.31 | **1.34/2.47** | 1.38/2.57 |

CausalTraj outperforms all baselines on per-agent metrics at short horizons (≤2s) and **comprehensively surpasses** all baselines on joint metrics. For the 4s prediction horizon, minJADE drops from 1.63 (best baseline, LED) to 1.34, an 18% reduction.

### Table 2: Ablation Study (Basketball-U, 20 frames, joint metrics)

| Configuration | minJADE₂₀ | minJFDE₂₀ |
|---------------|-----------|-----------|
| CausalTraj (Mamba2) full model | **0.97** | **1.77** |
| w/o SRTE (standard Transformer) | 0.99 | 1.81 |
| Single Gaussian (no mixture) | 1.03 | 1.86 |
| Sample from component means only | 1.05 | 2.13 |

Ablation results indicate that SRTE provides measurable improvement; multimodal mixture modeling is critical — degrading from 8 Gaussian components to a single Gaussian increases minJADE by 6%; sampling from the distribution rather than using only the mean has a particularly large effect on minJFDE (1.77 → 2.13, +20%).

## Highlights & Insights

- **Contribution of the joint metric perspective**: This work is the first to systematically introduce joint metrics (minJADE/minJFDE) to sports trajectory forecasting, revealing blind spots in traditional per-agent metrics — many SOTA models perform substantially worse than expected on joint prediction.
- **Advantages of causal factorization**: Temporal causal factorization naturally supports stepwise modeling of evolving inter-agent interactions, avoiding the high latent capacity demands of parallel prediction methods. The straightforward MoG likelihood training requires no approximate inference.
- **Compelling qualitative evidence**: Visualizations clearly demonstrate that CausalTraj generates coordinated directional changes and realistic ball passes, while baseline models tend to produce smooth but poorly coordinated motion.
- **Lessons from MoFlow(joint obj.)**: Modifying only the loss function to a joint objective without changing the model architecture yields virtually no improvement on min-based joint metrics, demonstrating that joint modeling requires architectural support.

## Limitations & Future Work

- **Insufficient ball-player interaction modeling**: Qualitative results still show unrealistic distances between the ball and ball-handlers; the authors attribute this to the model's limited capacity to learn player-ball covariance.
- **Limitations of block-diagonal covariance**: Conditional independence among agents is assumed within each mixture component; although shared mixture weights provide some coupling, fine-grained inter-agent spatial covariance cannot be precisely modeled.
- **Absence of physical constraints**: Implausible behaviors such as the ball occasionally colliding with court boundaries persist, as the model incorporates no physical rules or boundary constraints.
- **Autoregressive inference efficiency**: Stepwise sampling is slower than parallel prediction methods and may be limiting in real-time prediction scenarios.
- **Limited dataset scale**: Validation is conducted on only three sports datasets, with no extension to other multi-agent domains such as autonomous driving or crowd navigation.

## Related Work & Insights

- **GroupNet (CVPR'22)**: Graph- and hypergraph-based spatial interaction modeling under a CVAE framework. CausalTraj substantially outperforms GroupNet on joint metrics (NBA 4s: 1.34 vs. 2.12 minJADE).
- **LED (CVPR'23)**: Latent diffusion-based trajectory prediction. LED achieves better per-agent metrics than GroupNet but worse than MoFlow, and similarly lags significantly behind CausalTraj on joint metrics.
- **MoFlow (CVPR'25)**: A flow matching-based single-step denoising model that remains the best on long-horizon per-agent metrics. However, its joint prediction capability is limited — even the variant trained with a joint loss shows almost no improvement on min-based joint metrics.
- **SportsTraj (ICLR'25)**: Previously state-of-the-art on Basketball-U/Football-U, using a joint training objective and a Mamba + graph network architecture. CausalTraj's Mamba2 variant surpasses SportsTraj on all joint metrics.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Combining a causal autoregressive framework with MoG likelihood for joint multi-agent trajectory forecasting, along with the SRTE design, is creative; overall, however, the work constitutes an effective combination of existing components.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three datasets, multiple baselines, complete ablation studies, and convincing qualitative visualizations; inference speed comparisons and larger-scale validation are absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated and the case for joint metrics is well argued. Open-source code and a project page enhance reproducibility.
- **Value**: ⭐⭐⭐⭐ — The joint metric perspective offers meaningful guidance to the sports trajectory forecasting community, and the model has clear practical value for applications such as tactical simulation and match analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MASFIN: A Multi-Agent System for Decomposed Financial Reasoning and Forecasting](../../NeurIPS2025/time_series/masfin_a_multi-agent_system_for_decomposed_financial_reasoning_and_forecasting.md)
- [\[AAAI 2026\] M2FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting](m2fmoe_multi-resolution_multi-view_frequency_mixture-of-experts_for_extreme-adap.md)
- [\[AAAI 2026\] FreqCycle: A Multi-Scale Time-Frequency Analysis Method for Time Series Forecasting](freqcycle_a_multi-scale_time-frequency_analysis_method_for_time_series_forecasti.md)
- [\[ICCV 2025\] V2XPnP: Vehicle-to-Everything Spatio-Temporal Fusion for Multi-Agent Perception and Prediction](../../ICCV2025/time_series/v2xpnp_vehicle-to-everything_spatio-temporal_fusion_for_multi-agent_perception_a.md)
- [\[AAAI 2026\] Mitigating Error Accumulation in Co-Speech Motion Generation via Global Rotation Diffusion and Multi-Level Constraints](mitigating_error_accumulation_in_co-speech_motion_generation_via_global_rotation.md)

</div>

<!-- RELATED:END -->
