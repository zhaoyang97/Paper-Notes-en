---
title: >-
  [Paper Note] Streaming Sliced Optimal Transport
description: >-
  [ICML 2026][Optimization][Streaming OT] Stream-SW is the first algorithm capable of estimating the sliced Wasserstein distance on "sample streams": for each 1D projection…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Streaming OT"
  - "Quantile Sketch"
  - "Stream-SW"
  - "Single-pass"
  - "Low Memory"
date: 2026-05-08
content_hash: 0995c47c0ae49cf8
---

# Streaming Sliced Optimal Transport

**Conference**: ICML 2026  
**arXiv**: [2505.06835](https://arxiv.org/abs/2505.06835)  
**Code**: https://github.com/khainb/StreamSW  
**Area**: Optimal Transport / Sliced Wasserstein / Streaming Algorithms / Point Cloud / 3D Vision  
**Keywords**: Streaming OT, Quantile Sketch, Stream-SW, Single-pass, Low Memory

## TL;DR
Stream-SW is the first algorithm capable of estimating the sliced Wasserstein distance on "sample streams": for each 1D projection, it maintains an approximate quantile function using KLL/quantile sketch, turning the closed-form integral of 1D Wasserstein into a streamable estimator with logarithmic space complexity in the number of samples, thus enabling SOT in IoT/edge device scenarios where samples are "seen once and discarded".

## Background & Motivation

**Background**: Although Wasserstein distance is widely used in GANs, autoencoders, flow matching, Bayesian inference, and point cloud analysis, its computational complexity $\mathcal{O}(n^3\log n)$ and sample complexity $\mathcal{O}(n^{-1/d})$ make it infeasible for high-dimensional and large-scale settings. Sliced Wasserstein (SW) reduces the complexity to $\mathcal{O}(n\log n)$ and improves sample complexity to $\mathcal{O}(n^{-1/2})$ by taking 1D Radon projections, thus avoiding the curse of dimensionality.

**Limitations of Prior Work**: All existing SW estimators are **offline**—they require storing all samples from both distributions in memory for sorting and quantile computation. In IoT, sensor streams, and online learning, samples are "seen once and discarded", with very tight memory constraints. Online Sinkhorn (Mensch & Peyré 2020) attempts online entropic OT, but with time $\mathcal{O}(n^2)$ and space $\mathcal{O}(n)$ due to the need to retain historical samples, making it impractical; Compressed Online Sinkhorn uses Gaussian quadrature for compression but suffers from super-cubic complexity and a compression rate $\mathcal{O}(m^{-1/d})$ coupled with dimensionality.

**Key Challenge**: The low cost of SW comes from the closed-form 1D quantile function $F^{-1}_\mu(q)$, but computing the quantile function itself requires "all samples" for accuracy—this creates a tension between "streamable estimation" and "using closed-form solutions".

**Goal**: (i) Construct a single-pass, streaming approximation for 1D Wasserstein (1DW); (ii) Apply this 1D streaming estimator to all projection directions to obtain Stream-SW; (iii) Provide provable probabilistic approximation error bounds and complexity analysis; (iv) Validate downstream performance on simulation, point cloud classification, point cloud gradient flow, and streaming change-point detection.

**Key Insight**: "Streaming" distribution comparison has mature tools in the database field—quantile sketches (KLL, t-digest, etc.). The authors observe that the closed-form for 1DW, $W_p^p(\mu,\nu)=\int_0^1|F^{-1}_\mu(q)-F^{-1}_\nu(q)|^p dq$, is entirely determined by the quantile functions, thus bridging "streaming quantile approximation" and "sliced OT"—two previously independent literatures.

**Core Idea**: Use quantile sketch data structures as streaming estimators for the CDF/quantile function, attaching them to each direction in the Monte Carlo projection loop, yielding the first streaming estimator for SW and retaining logarithmic space complexity in the number of samples $n$.

## Method

### Overall Architecture
Input: Two sample streams $\{x_t\}, \{y_t\}\sim\mu,\nu$ ($x,y\in\mathbb{R}^d$), arriving in any order, each sample seen only once.

Output: Stream-SW estimate $\widehat{\mathrm{SW}}_p^p(\mu,\nu)$, queryable at every time step $t$.

Components: (1) A set of projection directions $\theta_1,\dots,\theta_L\sim U(\mathbb{S}^{d-1})$ sampled once at initialization; (2) For each direction $\theta_\ell$, maintain two quantile sketches $\mathcal{Q}^\mu_\ell, \mathcal{Q}^\nu_\ell$, inserting $\theta_\ell^\top x_t$ into $\mathcal{Q}^\mu_\ell$ for each $x_t$ (similarly for $y_t$); (3) At query time, reconstruct the approximate quantile function $\widehat F^{-1}_{\theta_\ell\sharp\mu}(q)$ from the sketch, and use $L$ directions for Monte Carlo averaging to obtain Stream-SW.

### Key Designs

1. **Quantile Sketch as a Streaming Estimator for CDF/Quantile Functions**:

    - Function: Maintains a "compressed sample data structure" with $\mathcal{O}(\log n)$ space, capable of outputting an $\epsilon$-approximate quantile at any $q\in(0,1)$, and supports online insertion of new samples.
    - Mechanism: Represented by KLL sketch (Karnin–Lang–Liberty), the sketch maintains multi-level sample buffers, and upon filling, uses deterministic and randomized sampling for "compacting", halving the number of samples while doubling their weights, so that the sketch size grows only logarithmically with $n$, while keeping bounded relative rank error. The compressed "weighted samples" are used to reconstruct the empirical CDF $\widehat F_\mu$ and its inverse $\widehat F^{-1}_\mu$. The paper provides the corresponding probabilistic bound: $\Pr[\sup_q |\widehat F^{-1}_\mu(q)-F^{-1}_\mu(q)|>\delta]\le\eta$, serving as the foundation for the 1DW error bound.
    - Design Motivation: The closed-form for 1DW depends only on the quantile function, so "approximate quantile function" suffices—no need to store all sorted samples. Quantile sketches are mature tools in database stream processing, and integrating them "natively" into the OT framework allows direct leverage of decades of engineering optimizations (e.g., t-digest, GK sketch trade-offs).

2. **Stream-1DW: Streaming 1D Wasserstein Estimation + Probabilistic Error Bound**:

    - Function: Given two sketches, provides a streaming estimate $\widehat W_p^p$ for $W_p^p(\mu,\nu)$, and a streaming estimate of the 1D OT map $\widehat F^{-1}_\nu\circ\widehat F_\mu$ (important: many downstream applications such as gradient flow require the map, not just the distance).
    - Mechanism: For the integral $W_p^p(\mu,\nu)=\int_0^1|F^{-1}_\mu(q)-F^{-1}_\nu(q)|^p dq$, replace $F^{-1}_\mu,F^{-1}_\nu$ with the sketch-reconstructed $\widehat F^{-1}_\mu,\widehat F^{-1}_\nu$, and numerically integrate over $[0,1]$ (essentially piecewise summation at all sketch breakpoints—equivalent to the northwest corner algorithm). Using the uniform quantile error bound of the sketch, the authors prove that $|\widehat W_p-W_p|$ is, with probability $1-\eta$, bounded by a term linear in $\delta$; a pointwise error bound for the OT map is also provided.
    - Design Motivation: 1DW is the core loop of SW; establishing solid theory for 1D estimation first, then propagating the error via Monte Carlo expectation to multi-dimensional SW; retaining map estimation, not just distance, enables Stream-SW to drive all downstream tasks requiring transport directions, such as gradient flow and classification.

3. **Stream-SW: Applying Streaming 1DW to $L$ Projections + Complexity Analysis**:

    - Function: Reduces the multi-dimensional distribution comparison to $L$ parallel streaming 1D estimations, multiplying the logarithmic space complexity of the sketch by $L$.
    - Mechanism: At algorithm startup, sample $\theta_1,\dots,\theta_L$ once; for each new sample, project it onto all $L$ directions and insert into the corresponding sketches; at query time, compute $\widehat{\mathrm{SW}}_p^p=\frac{1}{L}\sum_\ell \widehat W_p^p(\theta_\ell\sharp\mu,\theta_\ell\sharp\nu)$. Complexity: space $\mathcal{O}(L\cdot s\log n)$ ($s$ is initial sketch size), time approximately $\mathcal{O}(L\cdot \log n)$ per sample; theoretical error bound combines "1D sketch error + MC error from number of projections $L$".
    - Design Motivation: Preserves SW's two main scalability sources—"parallel projections + closed-form 1D"—by simply replacing each 1D channel with a streaming version, minimizing architectural changes and maximizing engineering friendliness. Space complexity in $n$ is $\log n$, which is the core advantage over random subsampling baselines—subsampling requires storing far more samples than sketches to achieve the same error for large $n$.

### Loss & Training
This work is purely algorithmic estimation, with no training required. All "parameters" are sketch size $s$, number of projections $L$, and error tolerances $\epsilon,\delta,\eta$, with analytic trade-offs provided in the paper.

## Key Experimental Results

### Main Results

| Task | Metric | Stream-SW vs Random Subsampling (same memory) | Advantage |
|---|---|---|---|
| Gaussian / GMM distance estimation | $|$Estimate - True$|$ | Significantly smaller | Lower approximation error at same memory |
| Gaussian / GMM estimation | Memory usage | Smaller for same accuracy | Sketch is logarithmic vs subsampling linear |
| Point cloud KNN classification (streaming) | top-1 acc | Higher | Sketch retains tail information of distribution |
| Point cloud gradient flow | Final Wasserstein error | Lower | OT map estimation is more stable |
| Kinect streaming change-point detection | F1 | Higher | Streaming update responds quickly to abrupt changes |

### Ablation Study

| Configuration | Key Change | Conclusion |
|---|---|---|
| Increase sketch initial size $s$ | Error decreases | Monotonic improvement, shows $\mathcal{O}(s^{-1})$ trend |
| Number of projections $L$ ↑ | Error decreases + time cost ↑ | Satisfies $\mathcal{O}(L^{-1/2})$ MC convergence |
| Stream length $n$ ↑ | Space nearly unchanged (logarithmic) | Verifies $\log n$ space bound |
| vs Compressed Online Sinkhorn | Faster at same accuracy | Avoids super-cubic cost of Gaussian quadrature |
| Dimension $d$ ↑ | Stream-SW error stable | Not affected by curse of dimensionality (inherits SW property) |

### Key Findings
- With fixed memory budget, sketch is almost always more accurate than random subsampling—especially for distribution tails and long streams, subsampling distorts by "forgetting" early samples.
- Streaming OT map estimation enables Stream-SW to directly drive point cloud gradient flow: as points are received online, the driving force is continuously updated, and the target distribution is approached more accurately than with the "store then compute SW" offline strategy.
- Insensitive to dimension $d$ (retains $\mathcal{O}(n^{-1/2})$ sample complexity as in offline SW), which is a key advantage over Online Sinkhorn.

## Highlights & Insights
- **The bridge between "closed-form 1D formula + streaming quantile estimation" is exceptionally clean**—sliced OT and database sketch literatures have almost no overlap, and this work directly connects their most mature tools, yielding the first streaming version of SW with high research leverage despite its simplicity.
- **Retaining OT map estimation, not just distance**: Many works stop at "computing a scalar distance", but providing the map means Stream-SW can immediately drive all OT applications requiring gradients (flow matching, point cloud deformation, generative training).
- **Combination of probabilistic error bounds + logarithmic space complexity** makes it especially attractive for edge/IoT scenarios—offering theoretical control and predictable memory usage.
- Experiments cover "theoretical sanity check (Gaussian), point cloud classification, gradient flow, change-point detection"—four heterogeneous tasks, demonstrating that streaming SW is a drop-in replacement.

## Limitations & Future Work
- Sketch error bounds rely on the i.i.d. sample assumption; more refined analysis is needed for real streams (with temporal correlation or drift).
- Projection directions $\theta_1,\dots,\theta_L$ are fixed at startup; long streams may miss directions most discriminative for the current distribution—could combine with max-sliced/adaptive projection for "streaming direction selection".
- Only the classical SW is proven to be streamable; streaming versions of Generalized SW / Spherical SW / Hilbert SW require different sketch designs.
- Experiments do not provide end-to-end wall clock comparisons with Compressed Online Sinkhorn on million-scale streams; further engineering evidence is desirable.

## Related Work & Insights
- **vs Online Sinkhorn (Mensch & Peyré 2020)**: They online entropic OT, but with linear space and quadratic time; this work uses the SW approach to reduce space to logarithmic.
- **vs Compressed Online Sinkhorn (Wang 2023)**: They use Gaussian quadrature to compress samples, but complexity still suffers from the curse of dimensionality; Stream-SW leverages SW's dimension-independence to avoid this.
- **vs Standard SW / Generalized SW**: The method only changes "how 1D estimation is done", and can serve as a unified streaming backbone for these variants.
- **vs Quantile sketch literature (KLL, t-digest, GK)**: Introduces these from database OLAP to machine learning OT, opening new application directions.

## Rating
- Novelty: ⭐⭐⭐⭐ First streaming estimator for SW, cleanly bridges two independent literatures
- Experimental Thoroughness: ⭐⭐⭐⭐ Simulation + three downstream tasks provide sufficient coverage, engineering comparisons could be deepened
- Writing Quality: ⭐⭐⭐⭐ Theoretical exposition is clear, tracing error bounds from 1DW to Stream-SW complexity
- Value: ⭐⭐⭐⭐ Drop-in replacement value for streaming/IoT/online distribution comparison scenarios

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OTPrune: Distribution-Aligned Visual Token Pruning via Optimal Transport](../../CVPR2026/optimization/otprune_distribution-aligned_visual_token_pruning_via_optimal_transport.md)
- [\[NeurIPS 2025\] Streaming Federated Learning with Markovian Data](../../NeurIPS2025/optimization/streaming_federated_learning_with_markovian_data.md)
- [\[ICLR 2026\] RS-ORT: A Reduced-Space Branch-and-Bound Algorithm for Optimal Regression Trees](../../ICLR2026/optimization/rs-ort_a_reduced-space_branch-and-bound_algorithm_for_optimal_regression_trees.md)
- [\[NeurIPS 2025\] Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification](../../NeurIPS2025/optimization/optimal_rates_for_generalization_of_gradient_descent_for_deep_relu_classificatio.md)
- [\[ICML 2026\] Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization](support-proximity_augmented_diffusion_estimation_for_offline_black-box_optimizat.md)

</div>

<!-- RELATED:END -->
