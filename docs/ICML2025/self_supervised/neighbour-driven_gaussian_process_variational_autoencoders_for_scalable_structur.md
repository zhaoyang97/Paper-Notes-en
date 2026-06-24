---
title: >-
  [Paper Note] Neighbour-Driven Gaussian Process Variational Autoencoders for Scalable Structured Latent Modelling
description: >-
  [ICML 2025][Self-Supervised Learning][Gaussian Processes] Proposes two nearest-neighbour-based Gaussian process prior approximation methods (HPA and SPA) to introduce neighbour-driven sparsity into the latent space inference of GPVAEs. This enables scalable mini-batch training while retaining key latent dependencies, eliminating reliance on a large number of inducing points or restricted kernel functions.
tags:
  - "ICML 2025"
  - "Self-Supervised Learning"
  - "Gaussian Processes"
  - "Variational Autoencoders"
  - "Nearest Neighbour Approximation"
  - "Structured Latent Variables"
  - "Scalable Inference"
date: 2026-05-08
content_hash: 2f11cdefd80919fa
---

# Neighbour-Driven Gaussian Process Variational Autoencoders for Scalable Structured Latent Modelling

**Conference**: ICML 2025  
**arXiv**: [2505.16481](https://arxiv.org/abs/2505.16481)  
**Code**: None  
**Area**: Self-Supervised Learning  
**Keywords**: Gaussian Processes, Variational Autoencoders, Nearest Neighbour Approximation, Structured Latent Variables, Scalable Inference

## TL;DR

Proposes two nearest-neighbour-based Gaussian process prior approximation methods (HPA and SPA) to introduce neighbour-driven sparsity into the latent space inference of GPVAEs. This enables scalable mini-batch training while retaining key latent dependencies, eliminating reliance on a large number of inducing points or restricted kernel functions.

## Background & Motivation

Variational Autoencoders (VAEs) have achieved great success in representation learning and generative modelling. However, the standard VAE assumes that latent variables follow fully factorized Gaussian priors, failing to capture correlations among latent variables in structured data such as sequential or spatial data. Gaussian Process Variational Autoencoders (GPVAEs) model structured dependencies between latent variables by replacing the independent Gaussian prior with a GP prior. However, directly employing a GP introduces a cubic computational bottleneck of $\mathcal{O}(N^3)$.

Existing scalable GPVAE schemes mainly suffer from the following drawbacks:

**Restricted Kernel Assumptions**: Some methods (e.g., MGPVAE) only support specific Matérn kernels or low-rank kernels, limiting representation capacity.

**Inducing Point Methods**: Methods such as SVGPVAE use a small set of pseudo-points to approximate the posterior. However, they require a large number of inducing points when data changes rapidly, and optimizing the inducing point locations is inherently challenging.

**Sampling-based Methods**: Fully Bayesian methods (e.g., SGPBAE) offer well-calibrated results but suffer from time-consuming sampling.

The core insight of this work is that in many structured datasets (e.g., temporal proximity of video frames, local patterns in spatial areas), focusing on a few nearest neighbours is sufficient to capture most of the essential correlation structure. This idea aligns with Tobler's First Law of Geography and serves as the theoretical foundation for Nearest Neighbour Gaussian Processes (NNGP).

## Method

### Overall Architecture

This work inherits the standard GPVAE architecture: the encoder $q_\phi(\mathbf{Z}|\mathbf{Y})$ yields the mean and variance of the latent variables, the decoder $p_\theta(\mathbf{Y}|\mathbf{Z})$ reconstructs observations from latents, and the GP prior $p_\psi(\mathbf{Z}|\mathbf{X})$ imposes structured dependencies on the latent variables. Each latent dimension $l$ employs an independent kernel function $k_\psi^l$ to fully leverage the expressiveness of GPs.

The training objective is to maximize the ELBO:

$$\mathcal{L} = \mathbb{E}_{q_\phi(\mathbf{Z}|\mathbf{Y})}[\log p_\theta(\mathbf{Y}|\mathbf{Z})] - \text{KL}[q_\phi(\mathbf{Z}|\mathbf{Y}) \| p_\psi(\mathbf{Z}|\mathbf{X})]$$

The core problem is that the KL term involves a dense $N \times N$ covariance matrix $\mathbf{K_{XX}}$, preventing mini-batch decomposition. This paper proposes two neighbour-driven approximations to resolve this bottleneck.

### Key Designs

#### 1. Hierarchical Prior Approximation (HPA)

HPA introduces an auxiliary binary random vector $\mathbf{w} \in \{0,1\}^N$ to indicate the selection of latent variables, constructing a sparse covariance structure by "shutting off" interactions among non-neighbors:

- **Hierarchical Prior**: $p(\mathbf{Z}|\mathbf{w}) = \mathcal{N}(\mathbf{Z}|\mathbf{0}, \mathbf{D_w}\mathbf{K_{XX}}\mathbf{D_w})$, where $\mathbf{D_w} = \text{diag}(\mathbf{w})$
- **Variational Distribution**: $q(\mathbf{Z}|\mathbf{w}) = \mathcal{N}(\mathbf{Z}|\mathbf{D_w}\mu(\mathbf{Y}), \mathbf{D_w}\sigma^2(\mathbf{Y})\mathbf{D_w})$
- **Neighbourhood Sampling Strategy**: For each point $\mathbf{x}_i$ in the mini-batch, find the top-$H$ nearest neighbours in the entire dataset $\mathbf{X}$, with indices denoted as $n(i)$.

The mini-batch ELBO for HPA is:

$$\mathcal{L}_\text{HPA} \approx \frac{N}{|\mathcal{I}|}\sum_{i \in \mathcal{I}} \left\{ \mathbb{E}_{q(\mathbf{z}_i|\mathbf{y}_i)}[\log p(\mathbf{y}_i|\mathbf{z}_i)] - \frac{1}{N}\text{KL}[q(\mathbf{Z}_{n(i)}) \| p(\mathbf{Z}_{n(i)})] \right\}$$

The KL term decomposes into operations over several $H \times H$ low-dimensional covariance matrices. When $H=N$, the original full-batch ELBO is recovered.

#### 2. Sparse Precision Approximation (SPA)

SPA is based on the Vecchia approximation, which decomposes the GP joint distribution into a product of conditional distributions using the probability chain rule and imposes conditional independence:

- **Exact Decomposition**: $p(\mathbf{Z}) = p(\mathbf{z}_1)\prod_{j=2}^N p(\mathbf{z}_j|\mathbf{z}_{1:j-1})$
- **Neighbour Approximation**: $p(\mathbf{Z}) \approx p(\mathbf{z}_1)\prod_{j=2}^N p(\mathbf{z}_j|\mathbf{z}_{n(j)})$

where $n(j)$ denotes the $H$ nearest neighbours of $\mathbf{x}_j$ among the history of previous points $\{\mathbf{x}_h\}_{h=1}^{j-1}$. This is equivalent to performing a sparse Cholesky decomposition of the prior precision matrix $\mathbf{K_{XX}}^{-1}$.

The mini-batch ELBO of SPA is:

$$\mathcal{L}_\text{SPA} \approx \frac{N}{|\mathcal{I}|}\sum_{i \in \mathcal{I}} \mathbb{E}_{q(\mathbf{z}_i|\mathbf{y}_i)}[\log p(\mathbf{y}_i|\mathbf{z}_i)] - \frac{N}{|\mathcal{J}|}\sum_{j \in \mathcal{J}} \mathbb{E}_{q(\mathbf{Z}_{n(j)})}\text{KL}[q(\mathbf{z}_j) \| p(\mathbf{z}_j|\mathbf{Z}_{n(j)})]$$

Setting $H=N$ recovers the full ELBO, while $H=0$ degenerates into the standard VAE.

#### 3. Complementary Relationship between HPA and SPA

| Feature | HPA | SPA |
|------|-----|-----|
| Sparsification Target | Covariance Matrix | Precision Matrix |
| Sparsification Mechanism | Shuts off non-neighbour interactions via hierarchical selection of variables | Decomposes joint distribution into conditional distributions of neighbours via chain rule |
| Neighbour Search Range | Top-$H$ in global $\mathbf{X}$ | Top-$H$ in historical points |
| Theoretical Root | Hierarchical NNGP (Tran et al., 2021) | Vecchia Approximation (Vecchia, 1988) |
| Degeneration Condition | $H=N$ → Full-batch ELBO | $H=0$ → Standard VAE |

#### 4. Predictive Posterior

For a new input $\mathbf{x}_*$, prediction only requires considering its $H$ nearest neighbours in $\mathbf{X}$:

$$q(\mathbf{z}_*|\mathbf{Y}) = \int p(\mathbf{z}_*|\mathbf{Z}_{n(*)}) q(\mathbf{Z}_{n(*)}|\mathbf{Y}_{n(*)}) d\mathbf{Z}_{n(*)}$$

The predictive posterior is Gaussian and can be efficiently sampled for Monte Carlo estimation.

### Loss & Training

- **Training Objective**: Maximize the neighbour-driven approximated ELBO (either HPA or SPA form).
- **Joint Parameter Optimization**: The encoder $\phi$, decoder $\theta$, and kernel parameters $\psi$ are learned jointly via mini-batch SGD.
- **Kernel Flexibility**: Supports arbitrary kernel functions (RBF, Matérn, etc.) without being restricted to specific kernel assumptions.
- **Precomputing Nearest Neighbours**: Uses Faiss to accelerate nearest neighbour search on GPUs.
- **Computational Complexity**: Nearest neighbour search takes $\mathcal{O}(HN)$, and the Cholesky decomposition of the KL term requires $\mathcal{O}(LN_bH^3)$, where $N_b$ is the batch size, $L$ is the dimensionality of the latent space, and $H$ is the number of neighbours.

## Key Experimental Results

### Main Results

The paper conducts experiments on three types of tasks: representation learning, data imputation, and conditional generation.

| Dataset | Task | Metric | GPVAE-HPA/SPA | SVGPVAE (Inducing Points) | Gain |
|--------|------|------|---------------|-----------------|------|
| Moving Ball | Representation Learning | RMSE | Optimal (H=10) | Requires more inducing points | Lower reconstruction error |
| Time Series Data | Data Imputation | RMSE / NLL | Outperforms other GPVAEs | Moderate | Prediction accuracy + Training speed |
| Spatial Data | Conditional Generation | Log-likelihood | Competitive performance | Restricted by the number of inducing points | Flexible kernels + Faster convergence |

### Ablation Study

| Configuration | Key Metrics | Explanation |
|------|----------|------|
| $H=0$ (No neighbours) | Degenerates to standard VAE | Verification of SPA degeneration; cannot capture structured dependencies |
| $H=10$ | Near-optimal | A small number of neighbours is sufficient to capture core correlation structures |
| $H=N$ (All) | Recovers full-batch ELBO | Computational cost of $\mathcal{O}(N^3)$, not scalable |
| HPA vs SPA | Similar performance | HPA sparsifies covariance, SPA sparsifies precision; complementary |
| RBF vs Matérn Kernels | Both supported | Flexible kernel choice, not restricted to specific kernels |

### Key Findings

1. **High Efficiency with Few Neighbours**: An $H$ around 10 achieves performance close to full-batch GPs, reflecting the principle of local data correlation.
2. **Outperforming Inducing Point Methods**: Given the same accuracy, the proposed method requires fewer equivalent parameters and trains faster than SVGPVAE.
3. **Flexible Kernel Functions**: The approach is no longer restricted to low-rank or Matérn kernels, offering the freedom to select RBF, periodic kernels, etc.
4. **Strong Scalability**: The complexity is reduced from $\mathcal{O}(N^3)$ to $\mathcal{O}(LN_bH^3)$, making it suitable for large-scale datasets.

## Highlights & Insights

1. **Ingenious Problem Transformation**: Transplants NNGP from the observation space to the latent space of VAE, treating each data point as its own "inducing variable" and substituting global dependencies with neighbourhood relationships.
2. **Two Complementary Sparsification Strategies**: HPA and SPA provide sparse approximations from the covariance and precision matrix perspectives, respectively, offering diverse options.
3. **Profound Application of Tobler's First Law of Geography**: The intuition that "near things are more related" also holds in the latent space, providing solid theoretical support for the local approximation of GPs.
4. **Engineering Practicality**: Utilizes Faiss for fast nearest neighbour search, supports arbitrary kernel functions, and adopts standard encoder-decoder architectures, reducing the barrier to adoption.

## Limitations & Future Work

1. **The Number of Neighbours $H$ Requires Tuning**: Although experiments show $H=10$ is generally effective, the optimal $H$ may vary across datasets, and an adaptive selection mechanism is lacking.
2. **Reliance on Auxiliary Information**: Requires explicit auxiliary inputs $\mathbf{X}$ (such as timestamps or spatial coordinates) to define neighbourhood relationships, limiting applicability to datasets without a natural ordering.
3. **Sensitivity to Ordering**: SPA relies on the ordering of data points (the sequence of the chain decomposition), where different orderings may affect approximation quality.
4. **Potential Extensions**:
    - Adaptive neighbour number selection (e.g., dynamically adjusting $H$ based on the kernel's lengthscale).
    - Hybrid strategies combining inducing points and nearest neighbour methods.
    - Extending neighbourhood definitions to non-Euclidean spaces (such as graph-structured data).

## Related Work & Insights

- **GPVAE Family**: Casale et al. (2018) first proposed GPVAE but restricted it to low-rank kernels; Fortuin et al. (2020) applied it to time-series imputation but it was only suitable for short sequences.
- **SVGPVAE** (Jazbec et al., 2021): An inducing-point-based scalable scheme, where the difficulty of optimizing inducing points is the main bottleneck.
- **MGPVAE** (Zhu et al., 2023): Leverages state-space representations of Matérn kernels for Kalman filtering inference, but is limited in kernel choices.
- **NNGP** (Datta et al., 2016; Wu et al., 2022): Widely used in geostatistics, proving to outperform standard inducing point methods in large-scale tasks.
- **Insights**: The locality principle is central to GP scalability. Shifting this principle to the latent spaces of deep generative models is a promising direction that warrants further exploration.

## Rating

| Dimension | Score (1-5) | Explanation |
|------|-----------|------|
| Novelty | 4 | Porting NNGP concepts to the latent space of GPVAE is a novel transfer, and the two complementary approximations are elegantly designed. |
| Theoretical Rigor | 4 | The derivation is rigorous, and both HPA and SPA have clear degeneration and recovery conditions. |
| Practicality | 4 | Supports arbitrary kernels, Faiss acceleration, and standard architectures, making it engineering-friendly. |
| Writing Quality | 4 | Clear logic, unified notation, and well-articulated motivation. |
| Overall Rating | 4 | Provides a practical and elegant solution to the critical issue of GPVAE scalability. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning](../../CVPR2026/self_supervised/an_optimal_transport_driven_approach_for_cultivating_latent_space_in_online_incr.md)
- [\[ICML 2025\] AdaWorld: Learning Adaptable World Models with Latent Actions](adaworld_learning_adaptable_world_models_with_latent_actions.md)
- [\[CVPR 2025\] ScaleLSD: Scalable Deep Line Segment Detection Streamlined](../../CVPR2025/self_supervised/scalelsd_scalable_deep_line_segment_detection_streamlined.md)
- [\[ICLR 2026\] InfoNCE Induces Gaussian Distribution](../../ICLR2026/self_supervised/infonce_induces_gaussian_distribution.md)
- [\[NeurIPS 2025\] Curiosity-driven RL for Symbolic Equation Solving](../../NeurIPS2025/self_supervised/curiosity-driven_rl_for_symbolic_equation_solving.md)

</div>

<!-- RELATED:END -->
