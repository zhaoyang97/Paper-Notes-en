---
title: >-
  [Paper Note] Streaming Sliced Optimal Transport
description: >-
  [ICML 2026][3D Vision][Streaming OT] Stream-SW is the first algorithm capable of estimating sliced Wasserstein distance over a "sample stream." It utilizes KLL/quantile sketches to maintain approximate quantile functions…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "Streaming OT"
  - "Quantile Sketch"
  - "Stream-SW"
  - "Single-pass"
  - "Low Memory"
date: 2026-05-08
content_hash: 2df12ec91fad5fab
---

# Streaming Sliced Optimal Transport

**Conference**: ICML 2026  
**arXiv**: [2505.06835](https://arxiv.org/abs/2505.06835)  
**Code**: https://github.com/khainb/StreamSW  
**Area**: Optimal Transport / Sliced Wasserstein / Streaming Algorithms / Point Clouds / 3D Vision  
**Keywords**: Streaming OT, Quantile Sketch, Stream-SW, Single-pass, Low Memory

## TL;DR
Stream-SW is the first algorithm capable of estimating sliced Wasserstein distance over a "sample stream." It utilizes KLL/quantile sketches to maintain approximate quantile functions for each 1D projection, transforming the closed-form 1D Wasserstein integral into a streamingly updatable estimator. With space complexity only logarithmic relative to the number of samples, it brings Sliced Optimal Transport (SOT) to "one-look" scenarios like IoT and edge devices.

## Background & Motivation

**Background**: Although Wasserstein distance is widely used in GANs, autoencoders, flow matching, Bayesian inference, and point cloud analysis, its computational complexity of $\mathcal{O}(n^3\log n)$ and sample complexity of $\mathcal{O}(n^{-1/d})$ make it infeasible for high-dimensional, large-scale data. Sliced Wasserstein (SW) reduces complexity to $\mathcal{O}(n\log n)$ and improves sample complexity to $\mathcal{O}(n^{-1/2})$ via 1D Radon projections, circumventing the curse of dimensionality.

**Limitations of Prior Work**: Existing SW estimators are **offline**, requiring all samples to be stored in memory for sorting and quantile function computation. In contexts like IoT, sensor streams, and online learning, samples are discarded after a single pass, and memory is highly constrained. Online Sinkhorn (Mensch & Peyré 2020) attempted an online entropic OT but remained impractical due to $\mathcal{O}(n^2)$ time and $\mathcal{O}(n)$ space requirements to retain historical samples. Compressed Online Sinkhorn used Gaussian quadrature for compression but faced hypercubic complexity and a compression rate of $\mathcal{O}(m^{-1/d})$ coupled with dimensionality.

**Key Challenge**: The low cost of SW stems from the 1D closed-form quantile function $F^{-1}_\mu(q)$, yet computing an exact quantile function requires access to all samples. This creates a conflict between "streaming estimation" and "leveraging closed-form solutions."

**Goal**: (i) Construct a 1D Wasserstein (1DW) estimator updatable in a single pass over a sample stream; (ii) extend this to all projection directions to form Stream-SW; (iii) provide provable probabilistic error bounds and complexity analysis; (iv) validate downstream performance in simulation, point cloud classification, gradient flows, and streaming change-point detection.

**Key Insight**: Streaming distribution comparison is a mature field in database research via quantile sketches (e.g., KLL, t-digest). The authors observe that the closed-form expression of 1DW, $W_p^p(\mu,\nu)=\int_0^1|F^{-1}_\mu(q)-F^{-1}_\nu(q)|^p dq$, is entirely determined by quantile functions. Thus, they bridge the independent literatures of "streaming quantile approximation" and "sliced OT."

**Core Idea**: Use the quantile sketch data structure as a streaming estimator for the CDF and quantile function. By integrating it into every direction of a Monte Carlo projection loop, the authors derive the first SW streaming estimator that maintains logarithmic space complexity relative to sample size $n$.

## Method

### Overall Architecture
Input: Sample streams $\{x_t\}, \{y_t\}\sim\mu,\nu$ ($x,y\in\mathbb{R}^d$) arriving in any order, with each sample seen only once.

Output: Stream-SW estimate $\widehat{\mathrm{SW}}_p^p(\mu,\nu)$, queryable at any time step $t$.

Components: (1) A set of projection directions $\theta_1,\dots,\theta_L\sim U(\mathbb{S}^{d-1})$ sampled once at initialization; (2) Two quantile sketches $\mathcal{Q}^\mu_\ell, \mathcal{Q}^\nu_\ell$ maintained for each direction $\theta_\ell$, where each $x_t$ results in pushing $\theta_\ell^\top x_t$ into $\mathcal{Q}^\mu_\ell$ (similarly for $y_t$); (3) Reconstruction of approximate quantile functions $\widehat F^{-1}_{\theta_\ell\sharp\mu}(q)$ during query to compute the Stream-SW via Monte Carlo averaging across $L$ directions.

### Key Designs

1.  **Quantile Sketch as Streaming Estimator for CDF/Quantile Functions**:
    - **Function**: Maintains a compressed data structure in $\mathcal{O}(\log n)$ space to output $\epsilon$-approximate quantiles for any $q\in(0,1)$, supporting real-time insertions.
    - **Mechanism**: Exemplified by the KLL (Karnin–Lang–Liberty) sketch, the structure maintains multi-level sample buffers. When a buffer is full, it performs "compacting" via deterministic or randomized sampling, halving the sample count while doubling weights. This ensures the sketch size grows logarithmically with $n$ while preserving bounded relative rank errors. The empirical CDF $\widehat F_\mu$ and its inverse $\widehat F^{-1}_\mu$ are reconstructed from the weighted samples. The paper provides the bound: $\Pr[\sup_q |\widehat F^{-1}_\mu(q)-F^{-1}_\mu(q)|>\delta]\le\eta$.
    - **Design Motivation**: 1DW only requires quantile functions; thus, approximate quantiles suffice without storing sorted samples. Utilizing mature database tools allows inheriting decades of engineering optimizations (e.g., trade-offs in t-digest or GK sketches).

2.  **Stream-1DW: Streaming 1D Wasserstein Estimation and Error Bounds**:
    - **Function**: Provides a streaming estimate $\widehat W_p^p$ for $W_p^p(\mu,\nu)$ and a streaming estimate for the 1D OT map $\widehat F^{-1}_\nu\circ\widehat F_\mu$ (essential for gradient flows).
    - **Mechanism**: Replaces $F^{-1}_\mu, F^{-1}_\nu$ in $W_p^p(\mu,\nu)=\int_0^1|F^{-1}_\mu(q)-F^{-1}_\nu(q)|^p dq$ with the sketch-reconstructed $\widehat F^{-1}_\mu, \widehat F^{-1}_\nu$, followed by numerical integration over $[0,1]$ (equivalent to piecewise summation at sketch breakpoints). The authors prove that $|\widehat W_p-W_p|$ is bounded by terms linear in $\delta$ with probability $1-\eta$. Pointwise error bounds for the OT map are also provided.
    - **Design Motivation**: 1DW is the inner loop of SW. Establishing solid 1D theory allows propagating errors to multi-dimensional SW via Monte Carlo expectation. Retaining the map estimate enables Stream-SW to drive tasks requiring transport directions.

3.  **Stream-SW: Extension to $L$ Projections and Complexity Analysis**:
    - **Function**: Transforms multi-dimensional distribution comparison into $L$ parallel streaming 1D estimations, with total space complexity being $L$ times the sketch complexity.
    - **Mechanism**: $\theta_1,\dots,\theta_L$ are sampled once at the start. Each incoming sample is projected onto all $L$ directions and inserted into corresponding sketches. The estimate $\widehat{\mathrm{SW}}_p^p=\frac{1}{L}\sum_\ell \widehat W_p^p(\theta_\ell\sharp\mu,\theta_\ell\sharp\nu)$ is computed upon query. Complexity: Space $\mathcal{O}(L\cdot s\log n)$ (where $s$ is initial sketch size) and approximate time $\mathcal{O}(L\cdot \log n)$ per sample.
    - **Design Motivation**: Preserves the scalability of SW (parallel projections and closed-form 1D solutions) while merely replacing 1D channels with streaming versions. The $\log n$ space complexity is the core advantage over random sub-sampling baselines, which require significantly more memory to achieve comparable error for large $n$.

### Loss & Training
This is a pure algorithmic estimator requiring no training. "Parameters" include sketch size $s$, number of projections $L$, and error tolerances $\epsilon, \delta, \eta$. The paper provides analytical trade-offs between these values.

## Key Experimental Results

### Main Results

| Task | Metric | Stream-SW vs. Sub-sampling (Fixed Memory) | Advantage |
|---|---|---|---|
| Gaussian / GMM Distance | $|$Est - True$|$ | Significantly lower | Higher accuracy for same memory |
| Gaussian / GMM Estimation | Memory Usage | Smaller for same accuracy | Logarithmic vs. Linear growth |
| Streaming Point Cloud KNN | Top-1 Acc | Higher | Sketch preserves distribution tails |
| Point Cloud Gradient Flow | Final Wasserstein Error | Lower | More stable OT map estimation |
| Kinect Change-point Detection| F1 Score | Higher | Faster response to abrupt changes |

### Ablation Study

| Configuration | Critical Change | Conclusion |
|---|---|---|
| Increase sketch size $s$ | Error decreases | Monotonic improvement following $\mathcal{O}(s^{-1})$ |
| Increase projections $L$ | Error ↓ / Time ↑ | Follows $\mathcal{O}(L^{-1/2})$ MC convergence |
| Increase stream length $n$ | Space nearly constant | Validates $\log n$ space bound |
| vs. Compressed Online Sinkhorn| Time for same accuracy | Stream-SW is faster; avoids quadrature cost |
| Increase dimension $d$ | Error stability | Robust to curse of dimensionality (SW property) |

### Key Findings
- Given a fixed memory budget, sketches are almost always more accurate than random sub-sampling, particularly in long-stream scenarios or for distribution tails.
- Streaming OT map estimation allows Stream-SW to drive point cloud gradient flows; it outperforms offline strategies by updating forces continuously as points arrive.
- Insensitivity to dimension $d$ (inheriting the $\mathcal{O}(n^{-1/2})$ sample complexity of offline SW) is a key advantage over Online Sinkhorn.

## Highlights & Insights
- **Clean Bridge**: The connection between "closed-form 1D formulas" and "streaming quantile estimation" is elegant. By bridging two mature but previously disjoint fields, the paper provides the first streaming SW with high research leverage.
- **Beyond Distance**: By providing OT map estimates rather than just scalar distances, Stream-SW is immediately applicable to gradient-based OT tasks like flow matching and generative training.
- **Theoretic-Practical Synergy**: The combination of probabilistic error bounds and logarithmic space complexity makes it attractive for edge/IoT scenarios where memory is predictable and errors must be controlled.
- **Drop-in Versatility**: Experiments across four heterogeneous tasks (Sanity checks, Classification, Gradient Flow, Change-point detection) prove its utility as a replacement for offline SW.

## Limitations & Future Work
- Sketch error bounds rely on IID assumptions; real-world streams with temporal correlation or drift require more granular analysis.
- Projection directions are fixed at initialization; they might miss discriminative directions in long streams. Combining this with max-sliced or adaptive projection for "streaming direction selection" is a potential future step.
- Only the streaming version of classical SW is proven; variants like Generalized, Spherical, or Hilbert SW might require different sketch designs.
- End-to-end wall clock comparisons against Compressed Online Sinkhorn on million-scale streams could further strengthen the evidence for engineering deployment.

## Related Work & Insights
- **vs. Online Sinkhorn (Mensch & Peyré 2020)**: They online-ify entropic OT but suffer from linear space and quadratic time. This work uses the SW route to compress space to logarithmic levels.
- **vs. Compressed Online Sinkhorn (Wang 2023)**: They use Gaussian quadrature, which still faces dimensional challenges. Stream-SW circumvents this via SW’s dimension independence.
- **vs. Standard SW / Generalized SW**: This work acts as a unified streaming backbone by only modifying how the 1D estimation is performed.
- **vs. Quantile Sketch Literature**: Introduces these tools from database OLAP to machine learning OT, opening a new application direction.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First SW streaming estimator; clean bridging of literatures.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Good coverage across tasks; engineering comparisons could be deeper.
- **Writing Quality**: ⭐⭐⭐⭐ Clear theoretical derivation from 1DW to Stream-SW.
- **Value**: ⭐⭐⭐⭐ High drop-in value for streaming, IoT, and online comparison scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] AvAtar: Learning to Align via Active Optimal Transport](avatar_learning_to_align_via_active_optimal_transport.md)
- [\[ICML 2026\] Convex Distance Operator Transport: A Convex and Geometry-Preserving Formulation](convex_distance_operator_transport_a_convex_and_geometry-preserving_formulation.md)
- [\[ICLR 2026\] SceneTransporter: Optimal Transport-Guided Compositional Latent Diffusion for Single-Image Structured 3D Scene Generation](../../ICLR2026/3d_vision/scenetransporter_optimal_transport-guided_compositional_latent_diffusion_for_sin.md)
- [\[ICLR 2026\] Fast Estimation of Wasserstein Distances via Regression on Sliced Wasserstein Distances](../../ICLR2026/3d_vision/fast_estimation_of_wasserstein_distances_via_regression_on_sliced_wasserstein_di.md)
- [\[CVPR 2026\] GLINT: Modeling Scene-Scale Transparency via Gaussian Radiance Transport](../../CVPR2026/3d_vision/glint_modeling_scene-scale_transparency_via_gaussian_radiance_transport.md)

</div>

<!-- RELATED:END -->
