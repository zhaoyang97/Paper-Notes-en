---
title: >-
  [Paper Note] Bi-Lipschitz Autoencoder With Injectivity Guarantee
description: >-
  [ICLR 2026][learning_theory][Paper Note] This paper identifies "encoder non-injectivity" as the root cause of regularized autoencoders falling into poor local optima. It proposes an injectivity regularizer based on the $(\delta,\epsilon)$-separation criterion and replaces the rigid isometric constraint with a Bi-Lipschitz regularizer acting on the decoder's J
tags:
  - ICLR 2026
  - learning_theory
date: 2026-05-08
content_hash: b04e00d307b4ac16
---
# Bi-Lipschitz Autoencoder With Injectivity Guarantee

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=rl2dPJEk8b](https://openreview.net/forum?id=rl2dPJEk8b)  
**Code**: https://github.com/qipengz/BLAE  
**Area**: Representation Learning / Manifold Learning / Learning Theory  
**Keywords**: Autoencoder, Injectivity, Bi-Lipschitz, Manifold Structure-Preserving, Distribution Shift Robustness

## TL;DR
This paper identifies "encoder non-injectivity" as the root cause of regularized autoencoders falling into poor local optima. It proposes an injectivity regularizer based on the $(\delta,\epsilon)$-separation criterion and replaces the rigid isometric constraint with a Bi-Lipschitz regularizer acting on the decoder's Jacobian singular values. The resulting BLAE preserves geometric structure with high fidelity on multiple manifold datasets and remains robust to sparse sampling and distribution shifts.

## Background & Motivation
**Background**: Autoencoders (AE) are widely used as tools for dimensionality reduction and visualization. Assuming high-dimensional data lies on a low-dimensional manifold $\mathcal{M}\subset\mathbb{R}^m$, an encoder $E_\theta:\mathbb{R}^m\to\mathbb{R}^n$ ($n\ll m$) compresses the data, and a decoder $D_\phi$ reconstructs it. To ensure the latent space preserves the manifold geometry, modern regularization approaches fall into two categories: **gradient regularization** (constraining encoder/decoder Jacobians, such as the Jacobian Frobenius norm in contractive AEs or isometric constraints) and **graph regularization** (using k-NN graphs to approximate geodesic distances to align latent space distance structures, including embedding regularization).

**Limitations of Prior Work**: Both categories have significant drawbacks. Graph regularization heavily depends on graph accuracy; when sampling is sparse, the shortest path distance deviates significantly from the true geodesic distance. It also assumes a Euclidean metric in the latent space, which only holds if the latent space is strictly convex. Gradient regularization is more robust to sample size and smoother manifolds, but **in practice, it often converges to poor local optima**, failing to achieve theoretical expectations—a phenomenon long dismissed simply as an "optimization challenge" without a clear root cause analysis.

**Key Challenge**: The authors approach this from a differential topology perspective: a mapping satisfying $J_f^\top J_f\equiv I$ is merely an isometric immersion. To become a true embedding, a **global injectivity** condition is required. Gradient constraints only encode local geometric properties and cannot guarantee global injectivity. Thus, the conflict is clarified: **the bottleneck of gradient-based AEs is not expressive power, but the non-injectivity of the encoder**. Non-injectivity causes distant points on the manifold to collide in the latent space (latent collision). To minimize reconstruction error in collision zones, the decoder must create sharp, high-curvature changes. If the decoder capacity is insufficient, the model becomes trapped in a poor local optimum.

**Goal**: During training, both reconstruction error and regularized terms are expectations over a specific distribution $P$, making the learned latent embedding dependent on $P$. However, the objective is to learn a **low-dimensional embedding of the manifold itself**, independent of the sampling distribution. Existing isometric constraints can preserve geometry but require $O(k^2)$ latent dimensions (where $k$ is the intrinsic manifold dimension) according to the Nash Embedding Theorem, which contradicts the goal of "dimension reduction" and loses theoretical guarantees when dimensions are insufficient.

**Core Idea**: Use a **separation criterion** to force encoder injectivity and eliminate poor local optima, allowing gradient methods to reach their theoretical potential. Then, replace rigid isometric constraints with **Bi-Lipschitz relaxation** to balance geometric fidelity and distribution-agnosticism (admissibility) under linear dimensional complexity. The combination forms BLAE.

## Method

### Overall Architecture
The logic of BLAE is to "diagnose the bottleneck first, then apply two targeted regularizers." The input is high-dimensional data on a compact connected Riemannian manifold $\mathcal{M}$, and the output is an injective and Bi-Lipschitz encoder-decoder pair that preserves manifold geometry independently of sampling distribution shifts. The training objective overlays two regularizations on the standard reconstruction loss:

$$
\mathcal{L}_{\text{BLAE}} = \mathcal{L}_{\text{recon}} + \lambda_{\text{reg}}\cdot\mathcal{L}_{\text{reg}} + \lambda_{\text{bi-Lip}}\cdot\mathcal{L}_{\text{bi-Lip}}
$$

Where $\mathcal{L}_{\text{recon}}=\mathbb{E}_{x\sim P}\|x-D_\phi(E_\theta(x))\|^2$ is the standard reconstruction term; $\mathcal{L}_{\text{reg}}$ is the **injectivity regularizer** based on the separation criterion, responsible for eliminating pathological local optima; and $\mathcal{L}_{\text{bi-Lip}}$ is the **Bi-Lipschitz regularizer** acting on the **decoder's** Jacobian singular values, ensuring consistent geometric mapping across different distributions. Before integrating these, the authors establish a theoretical foundation—**admissibility**—to determine if a regularizer's global optimal set is independent of the sampling distribution, ensuring robustness against distribution shifts.

### Key Designs

**1. Identifying "Encoder Non-Injectivity" as Optimization Bottleneck: From Local Immersion to Global Embedding**

The authors reduce the vague problem of poor convergence in gradient AEs to an exact topological property. Injectivity is defined as $f(x)\neq f(y),\ \forall x\neq y\in\mathcal{M}$. While the pointwise condition $E_\theta(x_i)\neq E_\theta(x_j)$ almost always holds for finite samples, it **does not equate to global injectivity**: even if sample points do not overlap, injectivity is violated if the encoded regions of two disjoint neighborhoods $U_i, U_j$ overlap ($E_\theta(U_i)\cap E_\theta(U_j)\neq\varnothing$). When such "collisions" occur—where distant manifold regions are mapped to nearby latent coordinates—the decoder must generate sharp changes (high local curvature) to lower reconstruction error. The required network complexity grows polynomially with the density of encoded points. If decoder capacity is reached, optimization is trapped in a bad local optimum. A toy example of a V-shaped manifold with 20 points (latent widths 2/16/256) demonstrates this non-injective collapse, noting that gradient regularizations like isometric constraints only yield local structure-preserving immersions, lacking the global injectivity needed for embeddings.

**2. $(\delta,\epsilon)$-Separation Criterion and Injectivity Regularization: Forcing Global Injectivity via Optimizable Inequalities**

To prevent "distant neighborhoods collapsing into close clusters," the authors introduce metric separation. A mapping $f$ is $(\delta,\epsilon)$-separated if, for all pairs with $d_\mathcal{M}(x,y)\ge\delta$, $\frac{d_\mathcal{N}(f(x),f(y))}{d_\mathcal{M}(x,y)}>\epsilon$. This provides a sufficient condition for injectivity and a **necessary and sufficient characterization** under the assumptions of continuity and manifold compactness (Theorem 1: $f$ is injective $\iff$ for any $\delta>0$, there exists $\epsilon>0$ such that $f$ is $(\delta,\epsilon)$-separated). A penalty is constructed for pairs violating separation:

$$
\mathcal{L}_{\text{inj}}(\delta,\epsilon)=\mathbb{E}_{x,y\sim P}\left[\mathrm{ReLU}\left(\log\frac{\epsilon\, d_\mathcal{M}(x,y)}{d_\mathcal{N}(E_\theta(x),E_\theta(y))}\right)\cdot\mathbf{1}_{d_\mathcal{M}(x,y)>\delta}\right]
$$

To prevent the trivial solution of scaling the encoder by $k$, a **non-expansion constraint** ($d_\mathcal{N}(E_\theta(x),E_\theta(y))\le d_\mathcal{M}(x,y)$) is added:

$$
\mathcal{L}_{\text{reg}}(\delta,\epsilon)=\mathcal{L}_{\text{inj}}(\delta,\epsilon)+\alpha\cdot\mathbb{E}_{x,y\sim P}\left[\mathrm{ReLU}\left(\tfrac{d_\mathcal{N}(E_\theta(x),E_\theta(y))}{d_\mathcal{M}(x,y)}-1\right)\cdot\mathbf{1}_{d_\mathcal{M}(x,y)>\delta}\right]
$$

Where $\alpha$ (default 5) controls the non-expansion strength. Implementation only requires checking the threshold $\delta_{\min}=\min_{i\neq j}d_\mathcal{M}(x_i,x_j)$ (Remark 2). $d_\mathcal{N}$ uses Euclidean distance, and $d_\mathcal{M}$ is approximated via graph construction. A key advantage is that this separation criterion is **exceptionally robust** to graph approximation errors. Visualizations of the Swiss Roll 2D loss landscape confirm that injectivity regularization smooths out local optima, guiding optimization toward a superior global minimum.

**3. Admissibility: Making the Global Optimal Set Independent of Sampling Distribution**

This is the theoretical core of robustness against distribution shift. Formally, for a regularizer $\mathbb{E}_{x\sim P}[R(f_\Theta(x))]$, let its global optimal set be $S_P:=\arg\min_{f_\Theta}\mathbb{E}_{x\sim P}[R(f_\Theta(x))]$. A regularizer is **admissible** if $S_P=S_Q$ for any probability measures $P, Q$ equivalent to the manifold measure $\mu_\mathcal{M}$. Theorem 2 provides a constructive sufficient condition: if $R$ has a global minimum and $\min_{f_\Theta}\mathbb{E}_{x\sim P}[R(f_\Theta(x))]=\min_u R(u)$, the regularizer is admissible. Intuitively, if a regularizer can **attain its lower bound pointwise** across the manifold (e.g., $R(f(x))=\|f(x)^\top f(x)-I\|_F^2$), it is insensitive to weighted distributions. Standard reconstruction error is a specific case of admissible regularization (Remark 3). This allows authors to judge which geometric regularizers are suitable for shift-robust constraints and disqualifies isometric constraints when dimensions are insufficient.

**4. Bi-Lipschitz Relaxation: Admissible Geometric Constraints on Decoder Jacobian Singular Values**

Isometric constraints require $O(k^2)$ dimensions per the Nash theorem, conflicting with dimensionality reduction. The authors use a $\kappa$-Bi-Lipschitz condition $\frac{1}{\kappa}d_\mathcal{M}(x,y)\le d_\mathcal{N}(f(x),f(y))\le\kappa\, d_\mathcal{M}(x,y)$ as a relaxation. To avoid direct geodesic distance calculations, they convert this to differential form (Theorem 3): for a smooth diffeomorphism $f$, $\kappa$-Bi-Lipschitz $\iff \frac{1}{\kappa}\le\sigma_{\min}(J_f^\mathcal{M}(x))\le\sigma_{\max}(J_f^\mathcal{M}(x))\le\kappa$, placing constraints on the **singular values** of the Jacobian. Two clever adjustments bypass tangent space estimation: first, Bi-Lipschitzness on $\mathbb{R}^m$ is inherited by submanifolds, allowing the use of $\mathbb{R}^m$ instead of $\mathcal{M}$; second, since the encoder $m>n$ cannot be a diffeomorphism, the condition is **applied to the decoder** $D_\phi:\mathbb{R}^n\to\mathbb{R}^m$ (as $f$ is Bi-Lipschitz $\iff f^{-1}$ is Bi-Lipschitz). The final regularizer is:

$$
\mathcal{L}_{\text{bi-Lip}}(\kappa)=\mathbb{E}_{x\sim P}\left[\mathrm{ReLU}\!\left(\tfrac{1}{\kappa}-\sigma_{\min}(x)\right)^2+\mathrm{ReLU}\!\left(\sigma_{\max}(x)-\kappa\right)^2\right]
$$

where $\sigma_{\min/\max}(x)$ are the singular values of $J_{D_\phi}(x)$. Theorem 4 proves that for a compact connected $k$-dimensional manifold, there exists a $\kappa$-Bi-Lipschitz embedding into $\mathbb{R}^n$ ($k\le n\le 2k$). Thus, this regularizer can reach a zero lower bound under **linear dimension complexity** $O(m)$, satisfying the admissibility condition. This preserves geometry and distribution-agnosticism while reducing dimension requirements from $O(k^2)$ to $O(k)$.

### Loss & Training
The final objective is $\mathcal{L}_{\text{BLAE}}=\mathcal{L}_{\text{recon}}+\lambda_{\text{reg}}\mathcal{L}_{\text{reg}}+\lambda_{\text{bi-Lip}}\mathcal{L}_{\text{bi-Lip}}$. The injectivity term $\mathcal{L}_{\text{reg}}$ includes the non-expansion weight $\alpha$ (default 5). Hyperparameters $\lambda_{\text{reg}}, \lambda_{\text{bi-Lip}}$ weight the respective terms. In implementation, $d_\mathcal{M}$ is approximated by a similarity graph, $d_\mathcal{N}$ uses Euclidean distance, and the separation threshold is set to $\delta_{\min}$. Grid search is used for each model-dataset combination.

## Key Experimental Results

### Main Results
Evaluation was conducted against 9 baselines (Geometric: SPAE/TAE; Gradient: IRAE/GAE/CAE; Embedding: GRAE/Diffusion Net; Hybrid: GGAE; and Vanilla AE). Metrics include reconstruction MSE, k-NN recall (neighborhood consistency), and KL$_\sigma$ divergence (distance distribution similarity). Geodesic distances were used to reflect manifold structure. Table 1 reports average rankings across datasets:

| Metric | BLAE | SPAE | TAE | DN | GRAE | CAE | GGAE | IRAE | GAE | Vanilla AE |
|------|------|------|-----|----|----|-----|------|------|-----|-----|
| k-NN | **1.8** | 3.2 | 3.8 | 4.5 | 4.0 | 5.8 | 7.5 | 7.2 | 9.0 | 7.8 |
| KL$_{0.01}$ | **1.0** | 3.0 | 2.5 | 6.0 | 4.5 | 7.0 | 8.2 | 6.8 | 6.5 | 9.2 |
| KL$_{0.1}$ | **1.0** | 3.2 | 2.8 | 5.5 | 3.5 | 7.5 | 9.0 | 7.2 | 6.5 | 8.8 |
| KL$_{1}$ | **1.0** | 4.2 | 3.2 | 5.2 | 4.2 | 7.0 | 8.0 | 7.2 | 6.0 | 8.2 |
| MSE | **1.2** | 4.8 | 4.0 | 5.2 | 4.5 | 5.5 | 7.5 | 7.0 | 8.0 | 7.2 |
| Downstream Acc | **1** | 5 | 7 | 4 | 6 | 2 | 10 | 8 | 3 | 9 |

BLAE achieved the best average ranking across almost all metrics (ranking 1.0 in all 4 KL terms), leading in geometric preservation, reconstruction fidelity, and downstream classification accuracy.

### Qualitative / Robustness Results

| Dataset | Setting | Key Findings |
|--------|------|---------|
| Swiss Roll | Removed a band to create geodesic vs Euclidean discrepancy | Only BLAE correctly preserved geometry; graph methods distorted near the removed band due to inconsistency; gradient methods without injectivity failed topology preservation. |
| dSprites | Semi-supervised, clustered by shape (square vs heart) | BLAE/SPAE/TAE/CAE/IRAE reconstructed cluster topologies; BLAE showed minimal geometric distortion between parallel planes. |
| MNIST Digit '3' | Rotational ring manifold with uniform/non-uniform sampling | Only BLAE produced consistent concentric ring structures under both distributions, proving invariance to sampling density shifts. |

### Key Findings
- **Injectivity constraint is key to convergence quality**: Gradient-based methods without injectivity constraints generally failed to preserve topology on Swiss Roll/MNIST, validating the "non-injectivity = root cause of bad local optima" diagnosis.
- **Admissibility determines drift robustness**: The admissible nature of the Bi-Lipschitz regularizer allowed BLAE to learn consistent structures regardless of uniform or non-uniform sampling.
- **Separation criterion is robust to graph errors**: Despite systematic errors in graph-approximated $d_\mathcal{M}$, the injectivity regularizer functioned stably with gradient regularization.

## Highlights & Insights
- **Mapping "Hard Training" to Topological Properties**: By using injectivity/embedding vs immersion language, the authors pinpoint the bottleneck of gradient AEs as global injectivity rather than expressive power, providing a more explanatory diagnosis than "poor optimization."
- **Admissibility as a General Discriminative Tool**: Formalizing whether a regularizer's global optimal set is independent of sampling distribution into the pointwise bound condition of Theorem 2 provides a universal criterion for designing shift-robust geometric regularizers.
- **Clever Swap of Constraint Targets**: Since encoders ($m>n$) cannot be diffeomorphisms, the Bi-Lipschitz condition is applied to the decoder using singular values to bypass tangent space estimation—a critical step for practical engineering.
- **Compressing Dimensional Requirement from $O(k^2)$ to $O(k)$**: Using Bi-Lipschitz relaxation of isometry preserves geometry while lowering latent dimension requirements to linear ($k\le n\le 2k$), resolving the inherent conflict between "isometric preservation" and "reduction."

## Limitations & Future Work
- **Reliance on Graph Proxies**: $d_\mathcal{M}$ still relies on similarity graph approximations, which are unreliable under extremely sparse sampling, introducing a source of error.
- **Singular Value Computation Overhead**: The Bi-Lipschitz term requires computing min/max singular values of the decoder Jacobian, which can be computationally expensive for high dimensions or large batches.
- **Focus on Synthetic/Small-scale Manifolds**: Primary validation is on controlled manifolds (Swiss Roll, dSprites, MNIST); performance on large-scale real-world high-dimensional data needs more systematic testing.
- **Number of Hyperparameters**: Several hyperparameters ($\delta,\epsilon,\alpha,\kappa,\lambda_{\text{reg}},\lambda_{\text{bi-Lip}}$) require grid searching, raising concerns about tuning costs and sensitivity in deployment.

## Related Work & Insights
- **vs Isometric Constraints (IRAE / GGAE)**: These pursue isometric embedding but require $O(k^2)$ dimensions and are non-admissible when dimensions are lacking; BLAE uses Bi-Lipschitz relaxation to maintain admissibility in $O(k)$ dimensions.
- **vs Graph/Embedding Regularization (SPAE / TAE / GRAE)**: Graph methods are naturally injective but sensitive to graph accuracy and assume Euclidean latent metrics; BLAE remains smooth and robust to sampling density once injectivity is guaranteed.
- **vs Contractive/Geometric Regulation (CAE / GAE)**: These only constrain local Jacobian properties and suffer from non-injectivity bottlenecks; BLAE's separation regularizer provides the missing global topological constraint.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reduces AE optimization bottlenecks to global injectivity and provides a practical framework via separation criteria + Bi-Lipschitz + admissibility.
- Experimental Thoroughness: ⭐⭐⭐⭐ 9 baselines and extensive average rankings. Strong qualitative results on shifts, though large-scale real-world validation is primarily in the appendix.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear progression from diagnosis to regularizers. Definitions, theorems, and remarks are well-integrated with visualizations.
- Value: ⭐⭐⭐⭐ Offers an explanatory theoretical framework and transferable "admissibility" criterion for the manifold manifold preservation and representation learning communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Bi-Criteria Metric Distortion](bi-criteria_metric_distortion.md)
- [\[ICLR 2026\] Lipschitz Bandits with Stochastic Delayed Feedback](lipschitz_bandits_with_stochastic_delayed_feedback.md)
- [\[ICLR 2026\] Persistence Spheres: Bi-Continuous Representations of Persistence Diagrams](persistence_spheres_bi-continuous_representations_of_persistence_diagrams.md)
- [\[ICLR 2026\] Deep Learning with Learnable Product-Structured Activations](deep_learning_with_learnable_product-structured_activations.md)
- [\[ICLR 2026\] Curse of Slicing: Why Sliced Mutual Information is a Deceptive Measure of Statistical Dependence](curse_of_slicing_why_sliced_mutual_information_is_a_deceptive_measure_of_statist.md)

</div>

<!-- RELATED:END -->
