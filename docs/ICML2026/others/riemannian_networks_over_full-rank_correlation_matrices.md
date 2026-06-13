---
title: >-
  [Paper Note] Riemannian Networks over Full-Rank Correlation Matrices
description: >-
  [ICML 2026][Correlation Matrix Manifold] This paper systematically extends three fundamental layers—MLR, FC, and Conv—to five Riemannian geometries (ECM, LECM, OLM, LSM…
tags:
  - "ICML 2026"
  - "Correlation Matrix Manifold"
  - "Riemannian Networks"
  - "Log-Euclidean Metric"
  - "Cholesky Decomposition"
  - "Poincaré Ball"
date: 2026-05-08
content_hash: cc94d5b9d8748172
---

# Riemannian Networks over Full-Rank Correlation Matrices

**Conference**: ICML 2026  
**arXiv**: [2605.19073](https://arxiv.org/abs/2605.19073)  
**Code**: To be confirmed  
**Area**: Geometric Deep Learning / Manifold Neural Networks  
**Keywords**: Correlation Matrix Manifold, Riemannian Networks, Log-Euclidean Metric, Cholesky Decomposition, Poincaré Ball

## TL;DR
This paper systematically extends three fundamental layers—MLR, FC, and Conv—to five Riemannian geometries (ECM, LECM, OLM, LSM, PHCM) on the manifold of full-rank correlation matrices $\mathrm{Cor}^+(n)$. Precise backpropagation is derived for OLM and LSM. The constructed CorNet consistently outperforms SPDNet and Grassmann networks of similar size on Radar, HDM05, FPHA, and NTU120 datasets.

## Background & Motivation

**Background**: In tasks driven by covariance-like features (EEG, radar, skeletal motion), SPD manifold neural networks have formed a mature pipeline—ranging from SPDNet and SPDNetBN to various new layers based on gyrovectors and Riemannian geometry. Geometric priors have been repeatedly proven to enhance discriminative power.

**Limitations of Prior Work**: Although the correlation matrix $C = D(\Sigma)^{-1/2} \Sigma\, D(\Sigma)^{-1/2}$ is a normalized version of covariance and is statistically more compact, there are few deep networks specifically designed for such inputs. Directly feeding them into SPDNet ignores the core constraints that the diagonal elements are constantly 1 and the degrees of freedom are reduced to $n(n-1)/2$. Earlier attempts to treat correlation matrices as quotient manifolds of SPD lacked unique closed-form solutions for the Riemannian log and Fréchet mean.

**Key Challenge**: The geometry of the correlation matrix manifold $\mathrm{Cor}^+(n)$ has only recently been commoditized—ECM, LECM, PHCM (Thanwerdas & Pennec, 2022) and the permutation-invariant OLM and LSM (Thanwerdas, 2024) provide five sets of metrics with closed-form expressions. However, these tools have not yet been utilized in deep learning.

**Goal**: To transition the three most common components in Euclidean deep learning (MLR, FC, Conv) to $\mathrm{Cor}^+(n)$, covering four zero-curvature Log-Euclidean metrics and one non-zero curvature metric, PHCM, composed of Poincaré ball products, while solving for precise backpropagation of implicit operators under OLM and LSM.

**Key Insight**: All Log-Euclidean metrics are isometric to a Euclidean prototype space via a diffeomorphism $\phi$. By pulling back the "signed distance to a boundary hyperplane" form of MLR (Lebanon & Lafferty) via $\phi$, the FC layer can be implicitly defined from MLR. This avoids manually deriving separate layers for each geometry.

**Core Idea**: By writing Euclidean layers in the prototype space, the five corresponding correlation layers can be obtained via pullbacks of the five $\phi$ mappings. The entire CorNet training process is "Euclideanized" by using trivialization in the tangent space to prevent over-parameterization and by replacing Riemannian trigonometry approximations with closed-form expressions.

## Method

### Overall Architecture
The input is a set of covariance matrices, which are first projected onto $\mathrm{Cor}^+ (n)$ to obtain correlation matrices via $\mathrm{Cor}(\Sigma)$. Then, two segments are stacked: "Correlation Conv Layer $\to$ Correlation MLR Head." The Conv layer performs FC transformations on multi-channel correlation matrices concatenated within each receptive field, where the FC is implicitly defined by the MLR logit under the corresponding metric. All learnable parameters are parameterized in the tangent space $T_E M \cong \mathrm{Hol}(n)$ (symmetric matrices with zero diagonals), allowing direct training with standard PyTorch optimizers. Geometry is manifested only through forward $\phi$, $\phi^{-1}$, and necessary Newton/iterations.

### Key Designs

1.  **Unified MLR: Written once in the prototype space, obtained for five metrics simultaneously**:
    - **Function**: Expresses the manifold MLR from Chen et al. (2024c) as computable closed-form logits on $\mathrm{Cor}^+(n)$.
    - **Mechanism**: For each Log-Euclidean metric, it is proven that $\phi(I) = 0$, allowing the identity matrix $I$ to serve as the manifold origin. Using the isometry from Thm. 3.1, the logit is reduced to the familiar form in prototype space: $v_k(X) = \langle \phi(X), \phi_{*,E}(Z_k)\rangle - \gamma_k \|\phi_{*,E}(Z_k)\|$. Substituting the four derivatives given in Prop. 3.2 (strictly lower triangular $\lfloor V\rfloor$ for ECM/LECM, $V$ itself for OLM, and $V - \mathrm{diag}(V\mathbf{1})$ for LSM) yields the MLR under four Log-Euclidean geometries. PHCM uses Cholesky decomposition to equate to a product of $n-1$ Poincaré upper hemispheres $\mathrm{PHS}^{n-1}$, reusing Poincaré MLR from Ganea / Shimizu.
    - **Design Motivation**: To avoid manual derivation of MLR formulas for each geometry and to preclude Riemannian trigonometry approximations—all logits are closed-form, and parameters $(Z_k, \gamma_k) \in \mathrm{Hol}(n)\times\mathbb{R}$ always reside in Euclidean space.

2.  **FC / Conv Layers: Defined via MLR, providing closed-form solutions per metric**:
    - **Function**: Generalizes the perspective of "FC as a stack of signed distances from multiple MLRs" from Shimizu et al. (2021) on Poincaré balls to correlation manifolds.
    - **Mechanism**: The FC layer $F: \mathrm{Cor}^+(n)\to\mathrm{Cor}^+(m)$ is implicitly defined by $m(m-1)/2$ equations $s_k\, d(Y, H_{O_k, I}) = v_k(X; Z_k, \gamma_k)$. Closed-form solutions exist under Log-Euclidean metrics (Thm. 3.5); for example, under ECM, $Y = \mathrm{Cor}\circ \mathrm{Chol}^{-1}(V^{EC} + I_m)$. LECM adds another $\exp$ layer, OLM uses $\mathrm{Exp}^\circ$, and LSM uses $\mathrm{Cor}\circ\exp$. Matrix elements $V^{*}_{ij}$ are populated according to the subspace structures of $\lfloor\cdot\rfloor$ / $\mathrm{Hol}$ / $\mathrm{Row}_0$. The Conv layer concatenates $c$ correlation matrices in each receptive field into $(\mathrm{Cor}^+(n))^c$ and feeds them to the same FC, equivalent to "one affine transformation per receptive field" in Euclidean convolution.
    - **Design Motivation**: FC is the inverse process of the MLR form where "output coordinates are signed distances to orthogonal hyperplanes at the origin." Unified definitions ensure that FC, MLR, and Conv layers share the same metric and parameter space, preventing geometric mismatch. Reusing FC for Conv also eliminates the need for a separate convolution theory.

3.  **Precise Backpropagation for OLM/LSM: Implicit operators as explicit Jacobians**:
    - **Function**: In OLM and LSM, $D(H)$ (the unique diagonal correction that brings $\exp(D+H)$ back to a correlation matrix) and $D^\star(C)$ (the unique positive diagonal scaling that results in zero row sums after logging $D^\star C D^\star$) lack closed-form expressions. Originally, autograd would have to pass through iteratively convergent operations like $D_{k+1} = D_k - \log(D(\exp(D_k + H)))$ or damped Newton methods.
    - **Mechanism**: The authors apply the implicit function theorem to these two fixed-point conditions $f(D,H)=0$ and $g(D^\star,C)=0$ with respect to parameters, directly solving for closed-form Jacobians (Sec. E). During training, the explicit formula is called once after iterative convergence. Backpropagation precision no longer depends on the number of iteration steps, and the overhead of re-iterating during the backward pass is avoided.
    - **Design Motivation**: Transparently passing autograd through iterations is both inaccurate and slow. This is particularly critical for metrics like OLM/LSM that are permutation-invariant but require numerical root-finding. Precise backpropagation is a prerequisite for stable training of large networks using these two metrics.

### Loss & Training
Cross-entropy is used with the MLR head for classification. All learnable parameters are placed in the tangent space $\mathrm{Hol}(n)$ (or $\mathrm{Row}_0(n)$ for LSM) via trivialization and updated directly using standard Adam/SGD without needing Riemannian optimizers. Using the same metric for both Conv and MLR is the default configuration; mixed metrics are discussed in the ablation study.

## Key Experimental Results

### Main Results
Evaluation protocol: Four standard SPD tasks—Radar (3000 radar signals, 3 classes), HDM05 (MoCap skeletal motion), FPHA (hand actions), and NTU120 (large-scale skeletal motion). Five-fold average accuracy (%).

| Manifold | Method | Radar | HDM05 | FPHA | NTU120 |
|--------|------|------|------|------|------|
| Grassmann | GrNet | 90.48 | 63.19 | 85.31 | 57.59 |
| SPD | SPDNet | 93.25 | 64.57 | 85.59 | 51.25 |
| SPD | SPDNetBN | 94.85 | 71.28 | 89.33 | 54.35 |
| SPD | SPDNetLieBN-AIM | 95.47 | 71.83 | 90.39 | 58.20 |
| SPD | GyroSPD++ | 95.20 | 69.82 | 89.50 | 61.57 |
| Correlation | CorNet-ECM | **97.71** | 81.35 | **92.17** | **65.04** |
| Correlation | CorNet-LECM | **98.40** | 78.05 | 91.17 | 65.03 |
| Correlation | CorNet-OLM | 97.57 | 81.46 | 91.63 | 64.41 |
| Correlation | CorNet-PHCM | 96.56 | **82.26** | 90.03 | 60.01 |

Compared to the classic SPDNet, CorNets show gains of +5.15% / +17.69% / +6.58% / +13.79% across the four datasets. Performance is still comprehensively superior compared to GyroSPD++ (same architecture). On the largest dataset, NTU120, CorNet-ECM/LECM remains one of the top-2 fastest methods (approx. 12 s per epoch).

### Ablation Study
| Configuration | Key Observation | Description |
|------|---------|------|
| Same metric for Conv and MLR (Tab. 3 diagonal) | Almost always optimal on HDM05/FPHA | Mixing across metrics generally leads to drops; geometric consistency is important. |
| SPDNet input: Covariance vs. Correlation (Tab. 4) | HDM05: 64.57→66.81; FPHA: 85.59→83.37; Radar: 93.25→89.49 | Correlation is sometimes better, but **ignoring the geometry of the correlation manifold can lead to performance drops**, justifying the need for specialized CorNets. |
| CorMLR vs. SPDMLR-Trivlz (Tab. 5) | CorMLR leads on HDM05/FPHA; slightly trails on Radar; ECM/PHCM have clear speed advantages | Even a single-layer MLR reflects the discriminative power of correlation embedding, and ECM-like geometries are computationally cheaper. |
| Backprop: autograd vs. precise Jacobian (OLM/LSM) | Better precision and stability (Sec. E) | Necessary condition for training permutation-invariant metrics. |

### Key Findings
- **Optimal metrics vary by task**: Radar favors LECM, HDM05 favors PHCM, and FPHA/NTU120 favor ECM. This suggests that geometry is a type of "hyperparameter," and making it a switchable component in CorNet is a contribution in itself.
- **Interpretable gains of Correlation vs. Covariance**: On HDM05, the coefficient of variation for diagonal covariance is large, and diagonal values far exceed off-diagonal ones. Correlation matrices flatten the diagonal to 1, forcing the network to focus on off-diagonal correlation terms with higher information density, yielding the largest gains.
- **Efficiency is not compromised**: CorNet-ECM/LECM is approx. 17× faster than GyroSPD++ and approx. 8× faster than GyroAI on NTU120, showing that the geometrically "lighter" correlation manifold is not a burden.

## Highlights & Insights
- **The paradigm of "prototype space once + five pullbacks"**: Previously, every new manifold metric required hand-crafting a set of layers. Here, all Log-Euclidean geometries are merged into a single proof of Euclidean MLR via $\phi$-isometry. Adding a new metric only requires calculating $\phi_{*, I}$, making it highly reusable.
- **Trivialization as the bridge between Riemannian geometry and PyTorch**: Keeping learnable parameters in the tangent space and using $\mathrm{Exp}$ to push them back to manifold parameters prevents over-parameterization and allows for the direct use of Euclidean optimizers, which is engineering-friendly.
- **Precise backpropagation of implicit operators**: Solving for the Jacobian using the implicit function theorem after writing iterative operators as implicit functions is a versatile technique used in OT, fixed-point layers, and DEQ. It is a training trick worth reusing in geometric deep learning.

## Limitations & Future Work
- Only five existing metrics are covered; other geometries on $\mathrm{Cor}^+(n)$ with non-trivial curvature (such as quotient metrics or affine-invariant versions of correlation) have not been layered yet.
- Experiments still focus on traditional SPD benchmarks with medium $n$ (signals/skeletons) and have not addressed larger scale, image-level covariance/correlation tasks (e.g., visual second-order pooling).
- Metrics must be manually specified. In the future, using hypernetworks or learned metrics to automatically select between ECM/LECM/OLM/LSM/PHCM could eliminate explicit parameter tuning.
- PHCM relies on products of Poincaré balls, where the number of hemispheres grows linearly with $n$. For extremely large $n$, it may not be as scalable as ECM.

## Related Work & Insights
- **vs. SPDNet series (Huang & Van Gool 2017, Brooks et al. 2019, Chen et al. 2024)**: While they develop geometric layers for the covariance side, this paper shifts to the correlation side. The advantage is that the correlation manifold has lower dimensionality and more geometric options, yielding significant gains for tasks like EEG and motion where diagonal variance is high. The disadvantage is the need to rebuild the metric catalog.
- **vs. Poincaré Networks (Ganea et al. 2018, Shimizu et al. 2021)**: They develop layers on a single Poincaré ball; the PHCM portion of this paper effectively reuses these layers directly on the correlation manifold in product space $\mathrm{PPS}^{n-1}$ in a very clean geometric manner.
- **vs. Unified manifold MLR (Chen et al. 2024c)**: They use Riemannian trigonometry to approximate hyperplane distances, whereas this paper provides a direct closed-form solution to Eq. (4) under Log-Euclidean metrics, avoiding approximation errors and resulting in a more organized structure.
- **vs. Grassmann Networks (GrNet, GyroGr)**: Grassmann uses subspace representations, while this paper uses correlation structures. The latter retains more second-order statistical information, offering stronger discriminative power on long-sequence motions like HDM05/NTU120.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Consistent Low-Rank Approximation](../../ICLR2026/others/consistent_low-rank_approximation.md)
- [\[ICLR 2026\] Condition Matters in Full-head 3D GANs](../../ICLR2026/others/condition_matters_in_full-head_3d_gans.md)
- [\[ICLR 2026\] Fast and Stable Riemannian Metrics on SPD Manifolds via Cholesky Product Geometry](../../ICLR2026/others/fast_and_stable_riemannian_metrics_on_spd_manifolds_via_cholesky_product_geometr.md)
- [\[ICML 2026\] DISCO: Mitigating Bias in Deep Learning with Conditional Distance Correlation](disco_mitigating_bias_in_deep_learning_with_conditional_distance_correlation.md)
- [\[ICML 2026\] On the Epistemic Uncertainty of Overparametrized Neural Networks](on_the_epistemic_uncertainty_of_overparametrized_neural_networks.md)

</div>

<!-- RELATED:END -->
