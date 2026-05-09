---
title: >-
  [Paper Note] Coupling Generative Modeling and an Autoencoder with the Causal Bridge
description: >-
  [NeurIPS 2025][Image Generation][causal bridge] In the presence of unobserved confounders, this paper proposes coupling a generative model with an autoencoder to improve estimation of the causal bridge function—sharing statistical strength across treatment, control, and outcome variables via a shared encoder—and extends the framework to survival analysis.
tags:
  - NeurIPS 2025
  - Image Generation
  - causal bridge
  - proxy variable
  - unobserved confounder
  - autoencoder
  - treatment effect
  - survival analysis
date: 2026-05-08
content_hash: 43327e379d0ac2c0
---

# Coupling Generative Modeling and an Autoencoder with the Causal Bridge

**Conference**: NeurIPS 2025
**arXiv**: [2509.25599](https://arxiv.org/abs/2509.25599)
**Code**: To be confirmed
**Area**: Causal Inference / Generative Models / Proxy Variables
**Keywords**: causal bridge, proxy variable, unobserved confounder, autoencoder, treatment effect, survival analysis

## TL;DR
In the presence of unobserved confounders, this paper proposes coupling a generative model with an autoencoder to improve estimation of the causal bridge function—sharing statistical strength across treatment, control, and outcome variables via a shared encoder—and extends the framework to survival analysis.

## Background & Motivation

**Background**: Estimating the causal effect of a treatment $X$ on an outcome $Y$ is a central problem across many domains. When unobserved confounders $U$ are present, standard methods (unconfoundedness assumptions, instrumental variables) may be inapplicable. The proxy variable approach uses two sets of observable variables correlated with $U$—treatment proxies $Z$ and outcome proxies $W$—to estimate causal effects via a causal bridge function.

**Limitations of Prior Work**: (a) The causal bridge function $b(W,x)$ requires solving the Fredholm integral equation $\mathbb{E}(Y|x,z) = \mathbb{E}(b(W,x)|x,z)$, which is difficult in practice; (b) DFPV employs iterative two-stage learning without flexible conditional sampling; (c) CEVAE requires specifying a prior $p(U)$ and suffers from training instability due to the KL term; (d) existing methods do not handle survival outcomes.

**Key Challenge**: While the theoretical framework of proxy variable methods (Fredholm equations) is elegant, a systematic mechanism for sharing statistical strength when learning bridge functions from limited data—especially in small-sample regimes—is lacking.

## Method

### Causal Bridge Function

The causal graph has $U$ as an unobserved confounder affecting treatment $X$ and outcome $Y$; $Z$ is the treatment proxy and $W$ is the outcome proxy. The core equation is:

$$\mathbb{E}(Y|x,z) = \mathbb{E}(b(W,x)|x,z), \quad \forall x, z$$

If a solution exists, the causal effect satisfies $\mathbb{E}[Y|do(X=x)] = \mathbb{E}[b(W,x)]$.

### Theoretical Contributions

**Theorem 3 (Mean Error Bound for the Causal Bridge)**: Assuming $\mathbb{E}[Y|x,W,U]$ is $C$-Lipschitz in $U$ and $\|U\| \leq R$:

$$\mathbb{E}_{Z \sim p(Z|x)}\left[|\mathbb{E}[Y|x,Z] - \mathbb{E}_{W \sim p(W|x,Z)}[b(W,x)]|\right] \leq CR \cdot \sqrt{2I(U;Z|W,x)}$$

The bridge estimation error is controlled by the conditional mutual information $I(U;Z|W,x)$—when $W$ is a high-quality (low-noise) proxy for $U$, the error is small.

**Corollary 1**: If $W = \Psi(U) + \varepsilon$, where $\Psi$ is invertible and $\varepsilon$ is independent of $(U,Z,X)$, then $I(U;Z|W,x) \leq C_0 \sigma_\varepsilon^2$.

### Generative Model + Autoencoder Framework

**1. Generalized Bridge Function**:

$$b(W,x) = \int dU \; g(x, W, U) \; p(U|W,x)$$

where $g$ need not equal $\mathbb{E}[Y|x,W,U]$, allowing more flexible learning. A generator $U = h_{\theta_U}(W, x, \epsilon)$, $\epsilon \sim \mathcal{N}(0,I)$ is used for sampling.

**2. Outcome Bridge Loss**:

$$\mathcal{L}_{\theta_Y} = \sum_{i=1}^{N} \left(y_i - \mathbb{E}_{p(W|x_i,z_i)} \mathbb{E}_{p(\epsilon)} [g_{\theta_Y}(x_i, W, h_{\theta_U}(W, x, \epsilon))]\right)^2$$

**3. Autoencoder for Sharing Statistical Strength**: Treatment $X$ and its proxy $Z$ are reconstructed jointly:

$$\mathcal{L}_{\theta_X} = \sum_{i=1}^{N} \left(x_i - \mathbb{E}_{p(W|x_i,z_i)} \mathbb{E}_{p(\epsilon)} [g_{\theta_X}(h_{\theta_U}(W, x_i, \epsilon), z_i)]\right)^2$$

$$\mathcal{L}_{\theta_Z} = \sum_{i=1}^{N} \left(z_i - \mathbb{E}_{p(W|x_i,z_i)} \mathbb{E}_{p(\epsilon)} [g_{\theta_Z}(h_{\theta_U}(W, x_i, \epsilon))]\right)^2$$

The encoder $h_{\theta_U}$ is **shared** across all three losses $(Y, X, Z)$—joint optimization of $\mathcal{L}_{\theta_Y} + \mathcal{L}_{\theta_X} + \mathcal{L}_{\theta_Z}$ improves the quality of $h_{\theta_U}$, especially in small-sample regimes.

**4. Learning Procedure** (two-stage, non-iterative):
1. Learn the conditional generative model $p(W|x,z)$ from $\mathcal{D}_1 = \{(x_i, z_i, w_i)\}$
2. Jointly optimize the shared encoder $\theta_U$, bridge $\theta_Y$, and autoencoder $\{\theta_X, \theta_Z\}$ using $\mathcal{D}_2 = \{(x_i, z_i, y_i)\}$

### Survival Analysis Extension

For survival outcomes $(Y, E)$ (where $Y$ is the observed time and $E$ is the event indicator), a Cox proportional hazards model is employed:

$$\mathcal{L}_{\theta_Y} = \sum_{i: e_i=1} \rho_i - \log\left(\sum_{j: y_j > y_i} \exp(\rho_i)\right)$$

where $\rho_i = \mathbb{E}_{p(W|x_i,z_i)} \mathbb{E}_{p(\epsilon)} [g_{\theta_Y}(x_i, W, h_{\theta_U}(W, x, \epsilon))]$.

The causal estimand is the hazard ratio (HR): $\text{HR} = \exp(b(W, X=1)) / \exp(b(W, X=0))$.

## Key Experimental Results

### Synthetic Data: Demand & dSprite

| Method | Demand MSE (N=1k) | Demand MSE (N=5k) | dSprite MSE (N=1k) | dSprite MSE (N=5k) |
|------|-------------------|-------------------|---------------------|---------------------|
| DFPV | Baseline | Baseline | Baseline | Baseline |
| DFPV + Sampling | Significant improvement | Significant improvement | Improved | Improved |
| CB | Further improved | Further improved | Further improved | Further improved |
| **CB + AE** | **Best** | **Best** | **Best** | **Best** |

- Sampling from $p(W|x,z)$ via the generative model (100 samples) substantially outperforms DFPV's iterative learning
- The generalized bridge model $g_{\theta_Y}(x, W, h_{\theta_U})$ yields further improvements
- **The autoencoder yields the largest gains in the small-sample regime (N=1k)**—statistical strength is transferred through the shared $h_{\theta_U}$

### Real Data: Framingham Heart Study (Compared Against RCT)

| Method | HR Estimate | 95% CI | Consistency with RCT |
|------|---------|--------|-------------|
| CoxPH-Uniform | >1 (wrong direction) | Contains 1 | ✗ |
| CoxPH-IPW | >1 (wrong direction) | Contains 1 | ✗ |
| CoxPH-OW | <1 | Near 1 | Partial |
| CB | <1 | Wide | ✓ |
| **CB + AE** | **<1** | **Narrowest, far from 1** | **✓✓** |
| RCT (reference) | <1 | — | Gold standard |

- CoxPH-Uniform and CoxPH-IPW yield HR > 1 (implying statins increase CVD risk), which is entirely incorrect due to confounding by indication
- CB + AE produces results most consistent with the RCT gold standard, with the tightest 95% CI and clearest separation from HR = 1

## Highlights & Insights

- **A complete chain from theory to method to experiment**: from the information-theoretic error bound (Theorem 3) to design intuition ($W$ should be a low-noise proxy for $U$), to the shared-encoder architecture, to validation against the RCT gold standard
- **The autoencoder sharing mechanism is simple yet effective**: it avoids the KL term of VAEs (and the instability of CEVAE), regularizing the latent space through reconstruction losses alone
- **The survival analysis extension** represents a new application direction for the causal bridge framework with significant practical value in medical research
- **Validation against an RCT** is a rare gold-standard benchmark in causal inference papers

## Limitations & Future Work

- **Assumption verification is difficult**: the completeness assumption (A4) and the conditional independence of proxy variables are hard to test in practice
- **Proxy variable assignment**: partitioning covariates into $Z$ and $W$ requires domain knowledge or heuristic decisions
- **Theorem 3's bound may not be tight**: the constant $CR$ in the information-theoretic bound may be large
- **Deliberately simple architecture**: the authors intentionally keep the architecture simple to demonstrate the value of the method itself, though more complex architectures could yield further gains
- **Only binary treatment ($X \in \{0,1\}$) is evaluated**: extensions to continuous treatments remain unexplored

## Related Work & Insights

- **vs. DFPV (Xu et al. 2021)**: DFPV uses iterative two-stage learning; this paper adopts sequential two-stage learning with conditional sampling and an autoencoder, yielding significant improvements
- **vs. CEVAE (Louizos et al. 2017)**: CEVAE requires a prior $p(U)$ and a KL term, leading to training instability; this paper replaces the VAE with an autoencoder to avoid the KL difficulties
- **vs. CoxPH + IPW/OW**: traditional reweighting methods fail under strong confounding; the causal bridge approach is more robust
- **vs. Ying et al. (2022)**: they also model the hazard function via a bridge function, but impose rigid parametric constraints and lack RCT reference comparisons

## Rating
- Novelty: ⭐⭐⭐⭐ Coupling a generative model with an autoencoder for the causal bridge is a novel combination; the information-theoretic error bound constitutes a solid theoretical contribution
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Synthetic and real data, RCT gold-standard comparison, and ablation studies form a complete validation chain
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear, though the heavy notation occasionally requires cross-referencing
- Value: ⭐⭐⭐⭐ Provides a solid methodological contribution to proxy-variable causal inference; the survival analysis extension adds practical utility

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Diffusion Generative Modeling on Lie Group Representations](diffusion_generative_modeling_on_lie_group_representations.md)
- [\[NeurIPS 2025\] System-Embedded Diffusion Bridge Models](system-embedded_diffusion_bridge_models.md)
- [\[NeurIPS 2025\] Hephaestus: Mixture Generative Modeling with Energy Guidance for Large-scale QoS Degradation](hephaestus_mixture_generative_modeling_with_energy_guidance_for_large-scale_qos_.md)
- [\[NeurIPS 2025\] Latent Zoning Network: A Unified Principle for Generative Modeling, Representation Learning, and Classification](latent_zoning_network_a_unified_principle_for_generative_modeling_representation.md)
- [\[NeurIPS 2025\] Non-Markovian Discrete Diffusion with Causal Language Models](non-markovian_discrete_diffusion_with_causal_language_models.md)

<!-- RELATED:END -->
