---
title: >-
  [Paper Note] Slicing Wasserstein over Wasserstein via Functional Optimal Transport
description: >-
  [ICLR 2026][learning_theory][Wasserstein over Wasserstein] This paper proposes the **Double-Sliced Wasserstein (DSW)** distance, which efficiently approximates the costly Wasserstein over Wasserstein (WoW) distance using a two-layered slicing approach: "spherical domain slicing + $L^2$ Gaussian process slicing of quantile functions." It proves that minimizing DSW is equivalent
tags:
  - ICLR 2026
  - learning_theory
  - Wasserstein over Wasserstein
date: 2026-05-08
content_hash: af2d049b5fc4d42b
---
# Slicing Wasserstein over Wasserstein via Functional Optimal Transport

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=l3KtyVZde3](https://openreview.net/forum?id=l3KtyVZde3)  
**Code**: https://github.com/MoePien/slicing_wasserstein_over_wasserstein  
**Area**: Optimal Transport / Learning Theory  
**Keywords**: Optimal Transport, Wasserstein over Wasserstein, Sliced Wasserstein, Meta-measure, Gaussian Process

## TL;DR
This paper proposes the **Double-Sliced Wasserstein (DSW)** distance, which efficiently approximates the costly Wasserstein over Wasserstein (WoW) distance using a two-layered slicing approach: "spherical domain slicing + $L^2$ Gaussian process slicing of quantile functions." It proves that minimizing DSW is equivalent to minimizing WoW on discrete meta-measures, avoiding the numerical instabilities of existing sliced methods that rely on high-order moments. It serves as a scalable alternative for WoW in comparing datasets, shapes, and images.

## Background & Motivation

**Background**: Optimal Transport (OT) provides the Wasserstein distance to measure geometric differences between probability measures on any Polish space. Since $\mathcal{P}_2(X)$ is itself a complete separable metric space, one can define another layer of Wasserstein distance on "measures of measures" (meta-measures), resulting in the **Wasserstein over Wasserstein (WoW)** distance $\mathcal{W}(\cdot,\cdot;\mathcal{P}_2(X))$. This is particularly natural for comparing "distributions of non-Euclidean objects" such as image distributions, point cloud distributions, and labeled datasets (OTDD)—for instance, while Euclidean distances often fail for direct image comparison, representing images as patch distributions and comparing them via WoW is robust to small perturbations.

**Limitations of Prior Work**: WoW is extremely expensive. If a meta-measure is supported on $N$ measures, each with $n$ support points, calculating just the required distance matrix incurs a complexity of $O(N^2 n^2 \log n)$, as the inner Wasserstein must be solved repeatedly for the outer transport. To accelerate this, existing sliced methods either assume **parametric** forms like Gaussian mixtures or follow the **moment-based** approach of s-OTDD—the latter is well-defined only when finite-order moments exist, and practical implementations are restricted to the first few moments (original implementation only up to the 5th order) due to numerical instability, limiting accuracy in high dimensions.

**Key Challenge**: The power of Sliced Wasserstein comes from "reducing high-dimensional transport to analytically solvable 1D transport." However, the "underlying space" of a meta-measure is the infinite-dimensional Wasserstein space $\mathcal{P}_2(\mathbb{R})$, which lacks a Banach space structure and suitable quantile mappings for direct classical slicing. Conversely, bypassing this by describing this infinite-dimensional object via moment expansion leads to numerical instability for higher-order moments.

**Goal**: Construct a sliced distance that is well-defined, computable, and numerically stable for general meta-measures, serving as a "plug-and-play" replacement for WoW with theoretical guarantees of its validity (equivalence in minimization).

**Key Insight**: The authors leverage a clean, classical isometric fact: **the 1D Wasserstein space $(\mathcal{P}_2(\mathbb{R}),\mathcal{W})$ is isometrically isomorphic to the image of quantile functions in $L^2([0,1])$**. Thus, the "Wasserstein distance between 1D measures" becomes the "Euclidean distance between $L^2$ functions," allowing for linear projections in the function space for slicing, thereby completely avoiding moment expansions.

**Core Idea**: First, the underlying domain of the meta-measure is reduced to 1D using classical spherical slicing. Then, using quantile isometry, input 1D measures are moved into the function space $L^2([0,1])$. A second layer of slicing is performed using $L^2$ projections parameterized by Gaussian process samples—this "double-slicing" maintains slicing efficiency while replacing unstable high-order moments with functional projections.

## Method

### Overall Architecture

Given two empirical meta-measures $\mu,\nu\in\mathcal{P}_e(\mathcal{P}_e(\mathbb{R}^d))$, i.e., "distributions of distributions": $\mu=\frac{1}{N}\sum_{i=1}^N\delta_{\mu_i}$, where each $\mu_i=\frac{1}{n_i}\sum_k\delta_{x_{i,k}}$ is an empirical measure on $\mathbb{R}^d$. The goal is to efficiently compute a distance to replace WoW.

The computational pipeline consists of two concatenated slicing steps:

1.  **First Layer—Domain Slicing (Outer Spherical Slicing)**. A direction $\theta\in S^{d-1}$ is sampled from the Euclidean domain $\mathbb{R}^d$. Each underlying measure $\mu_i$ is pushed onto the real line using linear projection $\pi_\theta(x)=\langle\theta,x\rangle$, yielding a 1D meta-measure $\pi_{\theta,\sharp}\mu\in\mathcal{P}_2(\mathcal{P}_2(\mathbb{R}))$. This reduces a "distribution of distributions on $\mathbb{R}^d$" to a "distribution of distributions on $\mathbb{R}$."

2.  **Second Layer—Quantile Slicing (Inner Functional Slicing)**. For 1D meta-measures, the quantile isometry $q:\mathcal{P}_2(\mathbb{R})\to L^2([0,1]),\ \mu\mapsto Q_\mu$ is used to map each 1D underlying measure into a quantile function in $L^2([0,1])$. The 1D meta-measure thus becomes a "distribution over $L^2$ functions." Inner product projections $\pi_g$ are then performed using directions $g\in L^2([0,1])$ sampled from a Gaussian process, mapping these functions to the real line to obtain standard 1D measures. Finally, the analytical 1D Wasserstein distance is computed. This layer is termed **SQW (Sliced Quantile WoW)**.

3.  **Synthesis and Monte Carlo Estimation**. Both layers of projection directions (spherical direction $\theta$ and Gaussian process sample path $g$) are sampled simultaneously and averaged via Monte Carlo to obtain the **Double-Sliced Wasserstein (DSW)** distance. The entire pipeline requires only sorting for quantiles + 1D integration (numerical integration for inner products) + 1D Wasserstein distance, all of which are analytical or near-analytical, avoiding any high-order moments.

The theoretical framework first generalizes Sliced Wasserstein to "any Banach space" (Key Design 1), with SQW and DSW being special cases for $L^2([0,1])$ and meta-measures. Finally, an equivalence theorem (Key Design 4) ensures that "minimizing DSW" is consistent with "minimizing WoW" for discrete meta-measures.

### Key Designs

**1. $\xi$-Sliced Wasserstein on Banach Spaces: Replacing non-existent infinite-dimensional uniform distributions with reference measures**

Classical Sliced Wasserstein in $\mathbb{R}^d$ integrates directions $\theta$ according to a uniform distribution on the sphere $S^{d-1}$. However, applying slicing to infinite-dimensional function spaces poses a hurdle: **there is no uniform probability measure on an infinite-dimensional sphere**. The solution is to abandon finding a specific measure on the sphere and instead use an arbitrary reference measure $\xi\in\mathcal{P}_2(U^*)$ on the dual space $U^*$. The projection is defined as $\pi_v(u)=\langle v,u\rangle=v(u)$, yielding the $\xi$-Sliced Wasserstein distance:

$$\mathrm{SW}(\mu,\nu;\xi):=\Big(\int_{U^*}\mathcal{W}^2(\pi_{v,\sharp}\mu,\pi_{v,\sharp}\nu;\mathbb{R})\,d\xi(v)\Big)^{1/2}.$$

As long as the support of $\xi$ covers all directions (Theorem 3.1 provides the sufficient condition $\mathrm{supp}\,\xi\cap\mathrm{span}\,v\notin\{\emptyset,\{0\}\}$), it constitutes a true metric rather than just a pseudo-metric. This abstraction is the foundation: it allows slicing on any separable Banach space using only "easily sampled reference measures." Robustness is also proven—if $\xi_1,\xi_2$ are mutually absolutely continuous with bounded Radon–Nikodym derivatives, the induced distances are equivalent (Proposition 3.2).

**2. SQW: Moving 1D meta-measures into $L^2$ via 1D Wasserstein Isometry + Gaussian Processes**

To slice the infinite-dimensional "distribution of 1D measures," the leverage is the quantile isometry: for $\mu\in\mathcal{P}_2(\mathbb{R})$, the quantile function $Q_\mu(s)=\inf\{x\in\mathbb{R}\mid\mu((-\infty,x])\ge s\}$ ensures:

$$\mathcal{W}(\mu,\nu;\mathbb{R})=\Big(\int_0^1 |Q_\mu(s)-Q_\nu(s)|^2\,ds\Big)^{1/2},$$

meaning the mapping $q:\mu\mapsto Q_\mu$ is an **isometric embedding** from $\mathcal{P}_2(\mathbb{R})$ to $L^2([0,1])$. By pushing the meta-measure $\mu\in\mathcal{P}_2(\mathcal{P}_2(\mathbb{R}))$ to $q_\sharp\mu\in\mathcal{P}_2(L^2([0,1]))$, the WoW distance exactly equals the Wasserstein distance between meta-measures on $L^2$. Applying Design 1 to $U=L^2([0,1])$ with a fixed reference measure $\xi$ yields **Sliced Quantile WoW**:

$$\mathrm{SQW}(\mu,\nu;\xi):=\mathrm{SW}(q_\sharp\mu,q_\sharp\nu;\xi).$$

For an "easily sampled and fully supported" reference measure, the authors choose a **Gaussian measure**, utilizing the one-to-one correspondence between Gaussian measures on $L^2([0,1])$ and Gaussian processes $G$ with covariance kernel $k_\sigma(t,s)=\exp(-|t-s|^2/2\sigma^2)$. This replaces the s-OTDD moment method: **quantile isometry is a "lossless" way to describe 1D measures**, unlike truncation to the first few moments, which is both lossy and numerically unstable.

**3. DSW: Double Slicing via Domain and Quantile Slices to reduce multi-dimensional meta-measures to analytical 1D**

SQW only handles 1D meta-measures. Practical data (image patches, point clouds) reside in $\mathbb{R}^d$, where the multi-dimensional Wasserstein space $\mathcal{P}_2(\mathbb{R}^d)$ has neither a Banach structure nor a suitable quantile mapping. The solution is to **slice the domain first**: use $\pi_\theta:\mathcal{P}_2(\mathbb{R}^d)\to\mathcal{P}_2(\mathbb{R}),\ \mu\mapsto\pi_{\theta,\sharp}\mu$ to reduce the multi-dimensional meta-measure to a 1D meta-measure, then apply SQW and integrate $\theta$ over the uniform spherical distribution to get **Double-Sliced WoW**:

$$\mathrm{DSW}(\mu,\nu;\xi):=\Big(\int_{S^{d-1}}\mathrm{SQW}^2(\pi_{\theta,\sharp}\mu,\pi_{\theta,\sharp}\nu;\xi)\,dS^{d-1}(\theta)\Big)^{1/2}.$$

Numerically, both outer spherical and inner $\xi$ integrals are approximated **simultaneously** via Monte Carlo: sampling directions $\theta_s$ and GP paths $g_s$. Quantile functions are represented as piecewise constant from sorted support points. Function inner products are approximated via quadrature with weights $w_r$: $\widehat{\langle q(\pi_{\theta,\sharp}\mu_i),g\rangle}=\sum_{r=1}^R w_r\,q(\pi_{\theta,\sharp}\mu_i)(t_r)\,g(t_r)$. Finally:
$$\widehat{\mathrm{DSW}}(\mu,\nu):=\Big(\tfrac{1}{S}\sum_{s=1}^S \mathcal{W}^2(\widehat{\pi_{g_s,\sharp}q_\sharp\pi_{\theta_s,\sharp}\mu},\,\widehat{\pi_{g_s,\sharp}q_\sharp\pi_{\theta_s,\sharp}\nu};\mathbb{R})\Big)^{1/2}.$$

**4. WoW Equivalence Theorem: Ensuring DSW is a valid replacement for WoW**

The value of a sliced distance depends on whether it changes consistently with the original distance. The authors prove (Theorem 4.1): for a positive Gaussian reference $\xi\in\mathcal{P}_2(L^2([0,1]))$, DSW is a **metric** on the set of discrete empirical meta-measures $\mathcal{P}_e(\mathcal{P}_e(\mathbb{R}^d))$. Furthermore, for empirical meta-measures of fixed size $N, \tilde n$ on a compactly supported domain $X\subset\mathbb{R}^d$, there is convergence equivalence:

$$\mathrm{DSW}(\mu_n,\mu;\xi)\to 0\iff \mathrm{SW}(\mu_n,\mu;\xi)\to 0\iff \mathcal{W}(\mu_n,\mu;\mathbb{R}^d)\to 0\quad(n\to\infty).$$

This "topological metric equivalence" validates DSW as a replacement for WoW: **minimizing DSW is equivalent to minimizing WoW**.

### Loss & Training
This paper does not train a specific network but proposes a family of distance metrics. There is no specialized training objective. Key hyperparameters include: number of spherical projections $S$, $L^2$ quadrature points $R$, and Gaussian kernel bandwidth $\sigma$. Experiments on MNIST-2000 show low sensitivity to these parameters.

## Key Experimental Results

### Main Results

**Shape Classification (KNN, higher is better)**: Shapes are modeled as metric measure spaces and represented as meta-measures $\mathcal{P}_e(\mathcal{P}_e(\mathbb{R}))$. SQW is compared against TLB (exact 1D WoW), STLB, Anchor Energy (AE), and Gromov–Wasserstein (GW).

| Distance | 2D shapes Acc(%) | Animals Acc(%) | FAUST-1000 Acc(%) | MNIST-2000 Acc(%) | FAUST-1000 Time(ms) |
|--------|--------|--------|--------|--------|--------|
| SQW (Ours) | 99.5±1.2 | 99.1±1.3 | 42.7±5.9 | 84.8±4.7 | 13.8±15.1 |
| TLB (Exact 1D WoW) | 100.0±0.3 | 100.0±0.0 | 40.2±6.0 | 88.7±4.5 | 60.1±9.6 |
| STLB | 99.5±1.2 | 99.3±1.8 | 39.4±5.6 | 84.1±5.0 | 14.0±14.9 |
| AE | 99.7±0.9 | 97.8±1.8 | 41.8±5.3 | 88.1±4.5 | 25.2±12.0 |
| GW | 99.7±0.6 | 100.0±0.0 | 33.0±5.3 | —Timeout— | 1048.2±357.3 |

SQW achieves accuracy comparable to exact WoW while offering significant runtime advantages, especially compared to GW on large-scale datasets.

**Correlation with OTDD / s-OTDD**: Labeled datasets are represented as meta-measures. Using original OTDD as a baseline, DSW's correlation with OTDD is compared against s-OTDD. **DSW shows stronger correlation with OTDD across MNIST, FashionMNIST, and CIFAR-10**, indicating that quantile slicing replicates OTDD more faithfully than moment-based slicing.

### Ablation Study

| Analysis | Key Setting | Finding |
|------|---------|------|
| Parameter Sensitivity | Vary $S\in\{10^2,10^3\}$, $R\in\{10,10^2\}$, $\sigma$ | MNIST-2000 accuracy is robust, fluctuating only between 82.6%–85.0%. |
| SQW vs Exact WoW (TLB) | MNIST-2000 correlation | Pearson/Spearman ≥0.99; increasing $S$ pushes correlation to 1.0. |
| Point Cloud Generation | ModelNet-10 | DSW behaves consistently with WoW (captures mode collapse, monotonic with noise) and is much faster (~0.25s vs 4.5s). |
| Image Patch Distribution | Perlin Texture (patch=8) | Patch-based DSW and WoW both minimize at "true parameters," but DSW is more sensitive to texture variations and runs in 1s vs 40s. |

### Key Findings
- **Numerical Stability**: Unlike s-OTDD which is limited by the first few moments, DSW uses quantile isometry for a lossless 1D description, being robust to $S, R, \sigma$.
- **Reduced Computational Cost**: In point cloud and image patch scenarios, DSW is over an order of magnitude faster than WoW and OT-NNA.
- **Consistent Behavior**: DSW changes consistently with WoW under mode collapse, noise, and resolution perturbations.

## Highlights & Insights
- **"Isometric Mapping" over "Moment Expansion"**: Quantile mapping is used as a lever to transform 1D Wasserstein into $L^2$ distance, which is lossless and eliminates numerical instability.
- **Reusable $\xi$-Slicing Framework**: The abstraction of using a reference measure $\xi$ to bypass infinite-dimensional spherical search can be transferred to other infinite-dimensional generative models.
- **Decoupled Double Slicing**: Dimensions are handled by spherical slicing, while the "measures of measures" aspect is handled by functional quantile slicing.
- **Theory-Practice Loop**: The equivalence theorem validates DSW as a legal replacement in minimization tasks.

## Limitations & Future Work
- **Discrete Limitation**: Theorem 4.1 is proven for discrete empirical meta-measures; properties on continuous meta-measures remain to be analyzed.
- **Label Alignment**: Currently, label metrics are set to zero for comparison; future work could extend this to hybrid slicing including label spaces $Y$.
- **Sample Complexity**: Whether DSW inherits the better sample complexity of classical sliced Wasserstein is an open question.
- **Kernel Selection**: The RBF kernel is used exclusively; systematic kernel design for different modalities has not been explored.

## Related Work & Insights
- **vs s-OTDD (Nguyen et al., 2025)**: s-OTDD uses moment-based projection which is lossy and unstable; DSW uses functional quantile isometry which is lossless and stable.
- **vs SWBDG (Bonet et al., 2025c)**: They use Busemann level sets to slice $\mathcal{P}_2(\mathcal{P}_2(\mathbb{R}^d))$; DSW uses a domain + functional slicing approach without Gaussian approximations.
- **vs Han (2023)**: This work extends the Hilbert space slicing concept to general Banach spaces without requiring specific spherical measures.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Revisiting Tree-Sliced Wasserstein Distance through the Lens of the Fermat–Weber Problem](revisiting_tree-sliced_wasserstein_distance_through_the_lens_of_the_fermatweber_.md)
- [\[ICLR 2026\] Test-Time Verification via Optimal Transport: Coverage, ROC, & Sub-Optimality](test-time_verification_via_optimal_transport_coverage_roc_sub-optimality.md)
- [\[ICLR 2026\] SEINT: An Efficient SE(p)-Invariant Transport Metric Driven by Polar Transport Discrepancy-based Representation](an_efficient_sep-invariant_transport_metric_driven_by_polar_transport_discrepanc.md)
- [\[ICLR 2026\] A Statistical Learning Perspective on Semi-dual Adversarial Neural Optimal Transport Solvers](a_statistical_learning_perspective_on_semi-dual_adversarial_neural_optimal_trans.md)
- [\[ICLR 2026\] Curse of Slicing: Why Sliced Mutual Information is a Deceptive Measure of Statistical Dependence](curse_of_slicing_why_sliced_mutual_information_is_a_deceptive_measure_of_statist.md)

</div>

<!-- RELATED:END -->
