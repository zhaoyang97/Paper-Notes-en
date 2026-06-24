---
title: >-
  [Paper Note] Learning General Causal Structures with Hidden Dynamic Process for Climate Analysis
description: >-
  [ICML 2026][Image Generation][Causal Discovery] This paper proposes CaDRe, which utilizes a structurally constrained temporal VAE to jointly identify the "causal graph between observed variables" and the "latent dynamic processes driving observations" within a single non-parametric framework. It provides identifiability theorems for recovering both from temporal data simultaneously. The theory is validated on synthetic data, while the model achieves causal graphs consistent w…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Causal Discovery"
  - "Causal Representation Learning"
  - "Hidden Dynamical Systems"
  - "Nonlinear ICA"
  - "VAE"
date: 2026-05-08
content_hash: 8c35845149aefd2c
---

# Learning General Causal Structures with Hidden Dynamic Process for Climate Analysis

**Conference**: ICML 2026  
**arXiv**: [2501.12500](https://arxiv.org/abs/2501.12500)  
**Code**: https://github.com/MinghaoFu/CaDRe (Available)  
**Area**: Causal Inference / Causal Representation Learning / Time Series Generative Models / Climate Modeling  
**Keywords**: Causal Discovery, Causal Representation Learning, Hidden Dynamical Systems, Nonlinear ICA, VAE

## TL;DR
This paper proposes CaDRe, which utilizes a structurally constrained temporal VAE to jointly identify the "causal graph between observed variables" and the "latent dynamic processes driving observations" within a single non-parametric framework. It provides identifiability theorems for recovering both from temporal data simultaneously. The theory is validated on synthetic data, while the model achieves causal graphs consistent with domain experts and competitive temperature prediction accuracy on CESM2 climate data.

## Background & Motivation
**Background**: Understanding the causal structure of climate systems is both a foundation for scientific reasoning and a prerequisite for reliable prediction. Currently, mainstream research follows two paths: Causal Representation Learning (CRL), which typically uses nonlinear ICA to recover "latent driving factors" $\mathbf{z}_t$ from observations based on auxiliary variables, sparsity, or restricted generative functions; and classic Causal Discovery (CD), such as LiNGAM, Additive Noise Models, or PC/FCI, which construct DAGs among observed variables.

**Limitations of Prior Work**: Both paths have blind spots. CRL generally assumes no direct causal links between observed variables and that the generation $\mathbf{z}_t \to \mathbf{x}_t$ is noise-free and invertible—conditions climate data fails to meet (e.g., temperatures in adjacent regions affect each other via heat conduction, and observations contain strong stochastic perturbations). CD methods either rely on specific parameterizations (e.g., LiNGAM assumes linearity and non-Gaussianity) or, in non-parametric settings with latent confounding, can only provide PAG equivalence classes (e.g., FCI), failing to identify the latent variables themselves or their causal relationships.

**Key Challenge**: In climate systems, "unobservable atmospheric/radiative dynamics" and "observable regional spatial coupling" are intertwined—latent processes drive observations, while observations have their own causal graphs. Existing methods cannot handle both systems simultaneously.

**Goal**: To simultaneously identify (i) the causal graph $\mathbf{J}_g(\mathbf{x}_t)$ between observed variables $\mathbf{x}_t$; and (ii) the latent variables $\mathbf{z}_t$ along with their instantaneous and lagged dependencies, all within a single temporal non-parametric framework and an estimable generative model.

**Key Insight**: The authors provide two critical observations. First, the DAG structure ensures that the distribution transformation from noise sources $\mathbf{s}_t$ to observations $\mathbf{x}_t$ is injective—causal propagation between observations does not destroy the recoverability of the latent space. Second, the original SEM is equivalent in a data-generative sense to a nonlinear ICA with latent variables, allowing ICA identifiability tools to be applied to CD.

**Core Idea**: Use context information from three consecutive observations $\{\mathbf{x}_{t-1},\mathbf{x}_t,\mathbf{x}_{t+1}\}$ to recover the latent space at a point-wise level $\hat{\mathbf{z}}_t=h_z(\mathbf{z}_t)$. Then, leverage the SEM↔ICA equivalence to express the observation causal graph as $\mathbf{J}_g(\mathbf{x}_t)=\mathbf{I}-\mathbf{D}_m(\mathbf{s}_t)\mathbf{J}_m^{-1}(\mathbf{s}_t)$. Finally, perform unified estimation using a VAE with sparsity and DAG constraints.

## Method

### Overall Architecture
The data generation process is formulated as a temporal SEM with latent variables: observations $x_{t,i}=g_i(\mathbf{pa}_O(x_{t,i}),\mathbf{pa}_L(x_{t,i}),s_{t,i})$ are influenced by other observed components $\mathbf{x}_{t,\setminus i}$, current and historical latent variables $\mathbf{z}_t,\mathbf{z}_{t-1}$, and endogenous noise $s_{t,i}=g_{s_i}(\mathbf{z}_t,\epsilon^x_{t,i})$ which depends on $\mathbf{z}_t$. Latent variables $z_{t,j}=f_j(\mathbf{pa}_L(z_{t,j}),\epsilon^z_{t,j})$ exhibit both instantaneous and lagged dependencies.

CaDRe is a state-space VAE: the input is a time series $\mathbf{x}_{1:T}$. A z-encoder $\phi$ infers $\hat{\mathbf{z}}_t$ from observations, an s-encoder $\eta$ infers non-stationary noise $\hat{\mathbf{s}}_t$, and a decoder $\psi$ reconstructs $\hat{\mathbf{x}}_t$ from $(\hat{\mathbf{z}}_t,\hat{\mathbf{s}}_t)$. Two prior networks (normalizing flows) learn the temporal prior for $\hat{\mathbf{z}}_t$ and the conditional prior for $\hat{\mathbf{s}}_t$, respectively. Causal graphs are extracted from their Jacobians. The loss function consists of the ELBO, a sparsity penalty $\mathcal{L}_s$, and a DAG penalty $\mathcal{L}_d$.

### Key Designs

**1. Non-parametric Point-wise Identifiability of Latent Space (Theorem 3.2): Isolating latents from noise mixtures**

In climate data, $\mathbf{z}_t \to \mathbf{x}_t$ is noisy and non-invertible. Previous non-parametric identification (e.g., Hu & Schennach) could only recover distributions $p_{\hat{\mathbf{z}}_t}=p_{\mathbf{z}_t}$ and required partially known generative functions, which is insufficient for component-wise causal discovery. This work achieves point-wise mapping $\hat{\mathbf{z}}_t=h_z(\mathbf{z}_t)$ where $h_z$ is invertible and differentiable, without assuming noise-free invertibility. The proof uses Lemma 3.1 to show the operator $L_{\mathbf{x}_t\mid \mathbf{s}_t}$ is injective—the DAG ensures information flows uni-directionally, preventing the graph between observations from blocking distribution-level recovery. It then uses context $\{\mathbf{x}_{t-1},\mathbf{x}_t,\mathbf{x}_{t+1}\}$ and injectivity conditions on operators $L_{\mathbf{x}_{t+1}\mid \mathbf{z}_t}$ and $L_{\mathbf{x}_{t-1}\mid \mathbf{x}_{t+1}}$ (A2), along with latent drift (A3) and differentiability (A4), to prove points-wise recoverability. This serves as the foundation for Theorems 3.5 and 3.6.

**2. SEM ↔ Nonlinear ICA Equivalence and Closed-form Causal Graphs (Lemma 3.3 + Theorem 3.5): Transforming CD into ICA structure identification**

Non-parametric causal discovery among observations is typically challenging—FCI/CDNOD only provide PAG equivalence classes under latent confounding. This work bypasses this by proving that non-parametric causal discovery is equivalent to identifying the mixture structure of a nonlinear ICA with latents. By defining Jacobians $[\mathbf{J}_g(\mathbf{x}_t)]_{i,j}=\partial x_{t,i}/\partial x_{t,j}$ and $[\mathbf{J}_m(\mathbf{s}_t)]_{i,j}=\partial x_{t,i}/\partial s_{t,j}$, a closed-form expression $\mathbf{J}_g(\mathbf{x}_t)=\mathbf{I}-\mathbf{D}_m(\mathbf{s}_t)\mathbf{J}_m^{-1}(\mathbf{s}_t)$ is derived. Theorem 3.6 then proves that the support set $\text{supp}(\mathbf{J}_g(\mathbf{x}_t))$ is identifiable under A5 (generation variability). This succeeds because, unlike previous ICA-based CD, Ours treats $\mathbf{z}_t$ as a continuous conditional prior and removes invertibility assumptions via the DAG structure.

**3. Estimation Architecture via Normalizing Flows and Jacobian Extraction: Implementing theorems in a trainable VAE**

To implement the closed-form theory, the architecture extracts the latent instantaneous graph $\mathbf{J}_r(\hat{\mathbf{z}}_t)$, latent lagged graph $\mathbf{J}_r(\hat{\mathbf{z}}_{t-1})$, and observed causal graph $\mathbf{J}_{\hat g}(\hat{\mathbf{x}}_t)$. The z-prior uses a conditional normalizing flow to learn $\hat{\epsilon}^z_{t,i}=r_i(\hat{\mathbf{z}}_{t-1},\hat{\mathbf{z}}_t)$, where structures are read from the block Jacobian. The s-prior learns $\hat{\epsilon}^x_{t,i}=w_i(\hat{\mathbf{z}}_t,\hat{\mathbf{s}}_t)$. The observed graph $\mathbf{J}_{\hat g}(\hat{\mathbf{x}}_t)$ is calculated in one step from the decoder's $\mathbf{J}_{\hat m}(\hat{\mathbf{s}}_t)$ using Corollary 2, making the DAG time-varying with $\hat{\mathbf{z}}_t$. Sparsity and DAG constraints (Yu et al. 2019) are applied to prevent redundant edges and cycles. The use of flows ensures the "independent component" assumption holds, driving the KL term to accurately match priors for Theorem 3.6.

### Loss & Training
Total loss: $\mathcal{L}_{ALL}=\mathcal{L}_{ELBO}+\alpha\mathcal{L}_s+\beta\mathcal{L}_d$. ELBO KL coefficients: $\lambda_1=4\times10^{-3}$, $\lambda_2=1.0\times10^{-2}$. Structural coefficients: $\alpha=1.0\times10^{-4}$, $\beta=5.0\times10^{-5}$. All encoders, decoders, and priors utilize MLPs. A three-step window is used as context under the first-order Markov assumption.

## Key Experimental Results

### Main Results
CaDRe was evaluated on synthetic data for "latent representation recovery" and "observed causal graph recovery," and on CESM2 climate data for "causal graph quality" and "temperature prediction."

| Dataset | Metric | CaDRe | Best Prev. | Gain / Diff |
|--------|-------|-------|------------|-------------|
| Synthetic Independent ($d_z=3,d_x=10$) | MCC | **0.9811** | TDRL 0.9106 | +0.07, nearly perfect recovery |
| Synthetic Sparse | MCC | **0.9306** | G-CaRL 0.7701 | +0.16, closest to climate sparsity |
| Synthetic Dense | MCC | **0.6750** | G-CaRL 0.6714 | Comparable; all methods degrade |
| CESM2 (CD) | WSHD ↓ | **0.012** | LPCMCI 0.019 | Lower than all PC/FCI/PCMCI baselines |
| CESM2 (CD) | WTPR ↑ | **0.532** | TCDF 0.327 | 60% higher recall than runner-up |
| CESM2 (CD) | Latency (ms) ↓ | **1.095** | TDRL 0.974 | ~3000× faster than PCMCI/LPCMCI |
| CESM2 Forecast (H=96) | MSE | **0.410** | CARD 0.409 | Comparable to SOTA while adding interpretability |
| CESM2 Forecast (H=192) | MSE | **0.412** | CARD 0.422 | Slightly better long-term prediction |

### Ablation Study
Synthetic performance across three graph densities provides structural ablation:

| Configuration | MCC (Sparse) | $R^2$ (Sparse) | Description |
|---------------|--------------|----------------|-------------|
| CaDRe (Full) | 0.9306 | 0.9102 | Dual encoders + flow prior + Sparsity/DAG |
| iCITRIS (No explicit $\mathbf{s}_t$ modeling) | 0.4531 | 0.6326 | Failure to decouple noise leads to entanglement |
| TDRL / LEAP (No observed DAG constraint) | 0.6628 / 0.6453 | 0.6953 / 0.4637 | Treating causal links as indep. noise causes drops |
| G-CaRL (No flows + weak prior) | 0.7701 | 0.5443 | Insufficient prior capacity affects independence |

### Key Findings
- In sparse scenarios (most realistic for climate), CaDRe outperforms the runner-up by 0.16 in MCC and 0.22 in $R^2$, highlighting the impact of explicit $\mathbf{s}_t$ modeling and DAG constraints.
- Inference latency on CESM2 is approximately three orders of magnitude faster than classic CD (1.1 ms vs 3000+ ms) because $\mathbf{J}_g$ is obtained via a closed-form Jacobian step rather than repeated conditional independence tests.
- For temperature prediction, CaDRe matches pure forecasting models like CARD while providing a free causal graph consistent with domain expertise.

## Highlights & Insights
- **DAG saves ICA invertibility**: Traditional nonlinear ICA requires invertibility for $\mathbf{z}\to\mathbf{x}$. This work uses the DAG of the SEM to prove $\mathbf{J}_m$ is invertible (Corollary 1). This allows ICA to run in realistic scenarios with latent confounding and observation coupling.
- **Analytical expression for causal graphs**: Mapping $\mathbf{J}_g$ to $\mathbf{I}-\mathbf{D}_m\mathbf{J}_m^{-1}$ allows causal graphs to be generated millisecond-speed and time-varyingly once the VAE is trained. This is transferable to fields like fMRI or macroeconomics.
- **Context for Distribution-level ICA**: Using the $\{\mathbf{x}_{t-1},\mathbf{x}_t,\mathbf{x}_{t+1}\}$ window to construct operator injectivity conditions (A2) replaces the need for exogenous domain labels with temporal neighborhoods.

## Limitations & Future Work
- The model assumes first-order Markov and sparse latent structures by default. While extensions for higher-order Markov are discussed, identifying multi-scale climate couplings (daily, seasonal, annual) in one flow remains to be demonstrated.
- Assumption A5 (generation variability) is a strong non-degeneracy condition; its verifiability when $d_x$ is large and variables are highly collinear needs more discussion.
- Evaluated primarily on the CESM2 dataset; hasn't been systematically verified on other physical scenarios such as oceanic thermohaline circulation.

## Related Work & Insights
- **vs CRL (iCITRIS / TDRL / LEAP / G-CaRL)**: These focus on latent factors and treat observation dependencies as noise. Ours identifies both layers and cross-layer effects, leading to substantial MCC gains.
- **vs CD (FCI / LPCMCI)**: These produce PAGs and are slow at high dimensions. Ours transforms CD into ICA Jacobian reading, identifying specific directed graphs with millisecond latency.
- **vs CDSD**: While CDSD also attempts to unify CRL and CD, it does not identify observation causal graphs. Ours is the only method to achieve all five properties in Table 2 (Non-parametric, Latent Vars, Latent Graph, Observed Graph, No Equivalence Class).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First unified framework for joint component-wise identifiability of non-parametric latents and observed causal graphs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid synthetic and real-world results, though more real climate datasets would be beneficial.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical progression, though minor symbol overlaps exist in some formulas.
- Value: ⭐⭐⭐⭐⭐ Provides a universal "VAE + Causal Graph" template applicable to climate, neuroscience, and finance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Recovering Hidden Reward in Diffusion-Based Policies](recovering_hidden_reward_in_diffusion-based_policies.md)
- [\[ICML 2026\] Caracal: Causal Architecture via Spectral Mixing](caracal_causal_architecture_via_spectral_mixing.md)
- [\[ICLR 2026\] Embracing Discrete Search: A Reasonable Approach to Causal Structure Learning](../../ICLR2026/image_generation/embracing_discrete_search_a_reasonable_approach_to_causal_structure_learning.md)
- [\[CVPR 2026\] Spatiotemporal Pyramid Flow Matching for Climate Emulation](../../CVPR2026/image_generation/spatiotemporal_pyramid_flow_matching_for_climate_emulation.md)
- [\[ICML 2026\] Image Restoration via Diffusion Models with Dynamic Resolution](image_restoration_via_diffusion_models_with_dynamic_resolution.md)

</div>

<!-- RELATED:END -->
