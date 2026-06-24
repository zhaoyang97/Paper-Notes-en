---
title: >-
  [Paper Note] DeCaFlow: A Deconfounding Causal Generative Model
description: >-
  [NeurIPS 2025 (Spotlight)][Image Generation][Causal Inference] DeCaFlow, a deconfounding causal generative model, is proposed. Given a causal DAG and observational data, it can correctly estimate all do-calculus identifiable causal queries (including interventions and counterfactuals) under a single training run, even in the presence of hidden confounders.
tags:
  - "NeurIPS 2025 (Spotlight)"
  - "Image Generation"
  - "Causal Inference"
  - "Hidden Confounders"
  - "Normalizing Flow"
  - "Counterfactual Queries"
  - "do-calculus"
date: 2026-05-08
content_hash: 5d3834b5f80e1c3f
---

# DeCaFlow: A Deconfounding Causal Generative Model

**Conference**: NeurIPS 2025 (Spotlight)  
**arXiv**: [2503.15114](https://arxiv.org/abs/2503.15114)  
**Code**: [GitHub](https://github.com/aalmodovares/DeCaFlow)  
**Area**: Causal Inference / Generative Models  
**Keywords**: Causal Inference, Hidden Confounders, Normalizing Flow, Counterfactual Queries, do-calculus

## TL;DR

DeCaFlow, a deconfounding causal generative model, is proposed. Given a causal DAG and observational data, it can correctly estimate all do-calculus identifiable causal queries (including interventions and counterfactuals) under a single training run, even in the presence of hidden confounders.

## Background & Motivation

The core task of causal inference is to estimate interventional effects and counterfactual outcomes from observational data. However, **hidden confounders**—unobserved variables that simultaneously affect both treatment and outcome variables—are ubiquitous in the real world, making causal effect estimation extremely challenging.

**Limitations of Prior Work:**

**Symbolic methods based on do-calculus**: Can determine whether a causal query is identifiable, but do not directly provide numerical estimations.

**Frontdoor/Backdoor criterion methods**: Can only handle specific confounding structures, lacking generality.

**VAE/Causal representation learning methods**: Require separate training for each causal query, resulting in poor scalability.

**Causal Normalizing Flows (e.g., VACA)**: Do not handle hidden confounders, assuming all variables are observed.

**Key Insight of DeCaFlow**: By introducing an encoder-decoder structured Normalizing Flow, hidden confounders are modeled as latent variables and adjusted using proxy variables. It is trained only once to answer all identifiable causal queries, with no retraining required.

## Method

### Overall Architecture

DeCaFlow consists of three core components:

1. **Encoder**: Maps observational variables to the latent space, inferring the posterior distribution of hidden confounders.
2. **Decoder/Causal Flow**: Models the conditional distributions of variables based on the causal DAG structure.
3. **Identifiability Algorithm**: Determines which queries are identifiable in a given causal DAG.

### Key Designs

**1. Structured Normalizing Flow based on Causal Graph**

The decoder of DeCaFlow is a **Structural Causal Normalizing Flow**, whose architecture directly encodes the DAG structure of the causal graph. The conditional distribution of each observed variable $X_i$ is determined by its parents in the causal graph:

$$X_i = f_i(\text{Pa}(X_i), Z_i, U_i)$$

where $\text{Pa}(X_i)$ represents the causal parents of $X_i$, $Z_i$ is the hidden confounder associated with $X_i$, and $U_i$ is the exogenous noise variable.

The flow is implemented using Neural Spline Flow (NSF), ensuring the invertibility of the transformation.

**2. Encoder Modeling of Hidden Confounders**

The encoder $q_\phi(Z|X)$ maps observed data to the approximate posterior of hidden confounders:

$$q_\phi(Z|X) = \prod_{k=1}^{K} q_\phi(Z_k | \text{Proxy}(Z_k))$$

where $\text{Proxy}(Z_k)$ is a set of proxy variables for the hidden confounder $Z_k$—i.e., observed variables affected by $Z_k$ but free from other confounders in the causal graph.

**3. Proxy Variable Adjustment**

DeCaFlow extends the classical frontdoor criterion and leverages proxy variables to adjust for the causal effects of hidden confounders:

- When do-calculus is sufficient to identify the query, the causal flow is used directly for estimation.
- When do-calculus is insufficient, the proxy variables are leveraged to indirectly infer the values of hidden confounders, which are then conditioned on for adjustment.

**4. Identifiability Guarantees**

The paper proves two key theoretical results:
- **Theorem 1**: DeCaFlow can correctly estimate all do-calculus identifiable interventional queries.
- **Theorem 2**: If the interventional counterpart of a counterfactual query is identifiable, then the counterfactual query itself is also identifiable.

### Loss & Training

DeCaFlow is trained using the Evidence Lower Bound (ELBO):

$$\mathcal{L} = \mathbb{E}_{q_\phi(Z|X)}[\log p_\theta(X|Z)] - \beta \cdot \text{KL}(q_\phi(Z|X) \| p(Z))$$

The training strategy includes a warmup mechanism:
- `epoch < warmup`: $\beta = \text{KL weight}$ (a small value to encourage reconstruction)
- `epoch ≥ warmup`: $\beta = 1$ (standard VAE-ELBO)

An Adam optimizer + ReduceLROnPlateau learning rate scheduler are used.

Query mechanisms after training:
- **Observational sampling**: $x_{\text{gen}}, z_{\text{gen}} = \text{DeCaFlow.sample}(n)$
- **Interventional sampling**: $x_{\text{int}}, z_{\text{int}} = \text{DeCaFlow.sample\_interventional}(\text{index}, \text{value}, n)$
- **Counterfactuals**: $x_{\text{cf}}, z_{\text{cf}} = \text{DeCaFlow.compute\_counterfactual}(\text{factual}, \text{index}, \text{value})$

## Key Experimental Results

### Main Results

**Table 1: Napkin Graph (2 hidden confounders) — ATE Error**

| Method | ATE Error ↓ | CF Error ↓ | Supported Query Types |
|------|------------|------------|-------------|
| Causal Flow (No latent var) | 0.45 | 0.52 | Observational Only |
| VACA | 0.38 | 0.44 | Observational Only |
| DeCaFlow (Ours) | **0.08** | **0.12** | With Hidden Confounders |

DeCaFlow's ATE estimation error is reduced by approximately 5 times compared to methods that do not handle hidden confounders.

**Table 2: Ecoli70 Dataset (46 observed variables, 3 hidden confounders, hundreds of causal queries)**

| Method| Avg. ATE Error ↓ | Avg. CF Error ↓ | Handleable Queries |
|------|-----------------|----------------|-------------|
| Naive (No Adjustment) | 0.62 | N/A | All |
| Backdoor Adj. | 0.35 | N/A | Partial |
| Frontdoor Adj. | 0.28 | N/A | Few |
| DeCaFlow | **0.11**| **0.15**| All Identifiable |

On large-scale complex causal graphs, DeCaFlow significantly outperforms methods based on classical criteria.

### Ablation Study

**Contribution of the Encoder (With/Without Hidden Confounder Modeling)**

| Configuration | Napkin ATE ↓ | Sachs ATE ↓ |
|------|-------------|-------------|
| DeCaFlow (Full) | **0.08** | **0.14** |
| Without Encoder (= Causal Flow) | 0.45 | 0.39 |

The encoder (hidden confounder inference) is the core factor driving performance improvement.

**Impact of the Number of Proxy Variables**

| Number of Proxy Variables | ATE Error ↓ |
|-----------|------------|
| 1 | 0.22 |
| 2 | 0.12 |
| 3+ | **0.08** |

More proxy variables provide richer signals to infer the hidden confounders.

### Key Findings

1. **Single Training, Multiple Queries**: Training one DeCaFlow can answer all identifiable interventional and counterfactual queries on a given causal graph.
2. **Destructiveness of Hidden Confounders**: Not handling hidden confounders leads to severe bias, even in simple causal graphs (such as the Napkin graph).
3. **New Results on Counterfactual Identifiability**: Proves that interventional identifiability implies counterfactual identifiability.
4. **Generality**: DeCaFlow can be applied to arbitrary causal graph structures without requiring specific patterns (e.g., backdoor/frontdoor criteria).
5. **Large-scale Feasibility**: Demonstrates solid performance on Ecoli70 (46 variables, 3 latent variables, hundreds of queries).

## Highlights & Insights

- **Bridge between Theory and Practice**: Merges the theoretical identifiability of do-calculus with the practical implementation of deep generative models.
- **"Single Training" Paradigm**: Replaces the expensive cost of training separate models for each causal query.
- **Clever Use of Proxy Variables**: Indirectly inferring unobservable confounders via proxy variables is an elegant way to tackle hidden confounding.
- **NeurIPS 2025 Spotlight**: Reflects the high research interest in the direction of causal inference.

## Limitations & Future Work

1. **Assumption of Known Causal Graph**: Requires a pre-specified causal graph structure, while causal discovery remains a challenge in practice.
2. **Restriction to Continuous Variables**: Currently only supports continuous variables; extension is needed for discrete or mixed variable scenarios.
3. **Availability of Proxy Variables**: Relies on the existence of appropriate proxy variables, which may not exist in all causal graphs.
4. **Computational Complexity**: Training costs of Normalizing Flows can be high in large-scale causal graphs.
5. **Unidentifiable Queries**: DeCaFlow cannot estimate queries that are unidentifiable under do-calculus, requiring integration with partial identification methods.

## Related Work & Insights

- **VACA (Sánchez-Martín et al. 2022)**: VAE-based causal inference, which however does not handle hidden confounders.
- **Causal Normalizing Flows (Javaloy et al. 2024)**: Structured flow without latent variables.
- **Pearl's do-calculus**: The theoretical foundation of DeCaFlow.
- **Frontdoor/Backdoor criteria**: Causal identification methods under specific scenarios.
- **zuko library**: Underlying tools for constructing Normalizing Flows.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 5 — First to achieve "single-training" causal inference on arbitrary causal graphs with hidden confounders. |
| Technical Quality | 5 — Solid theoretical proofs and experimental validation. |
| Experimental Thoroughness | 4 — Multi-causal graph and large-scale validation. |
| Writing Quality | 4 — 55 pages containing thorough theoretical derivations. |
| Value | 5 — Spotlight paper, pushing the causal inference field forward significantly. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Coupling Generative Modeling and an Autoencoder with the Causal Bridge](coupling_generative_modeling_and_an_autoencoder_with_the_causal_bridge.md)
- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](../../ICCV2025/image_generation/motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)
- [\[NeurIPS 2025\] Non-Markovian Discrete Diffusion with Causal Language Models](non-markovian_discrete_diffusion_with_causal_language_models.md)
- [\[NeurIPS 2025\] Denoising Weak Lensing Mass Maps with Diffusion Model and Generative Adversarial Network](denoising_weak_lensing_mass_maps_with_diffusion_model_and_generative_adversarial.md)
- [\[NeurIPS 2025\] GeoRemover: Removing Objects and Their Causal Visual Artifacts](georemover_removing_objects_and_their_causal_visual_artifacts.md)

</div>

<!-- RELATED:END -->
