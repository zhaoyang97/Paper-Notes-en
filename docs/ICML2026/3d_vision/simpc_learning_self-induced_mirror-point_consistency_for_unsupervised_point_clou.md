---
title: >-
  [Paper Note] SIMPC: Learning Self-Induced Mirror-Point Consistency for Unsupervised Point Cloud Denoising
description: >-
  [ICML 2026][3D Vision][Point Cloud Denoising] SIMPC proposes performing a "symmetric extension" along the denoising vector of the **same noisy point** to obtain a mirror point located on the other side of the surface. A…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "Point Cloud Denoising"
  - "Unsupervised Learning"
  - "Mirror-Point Consistency"
  - "Geometric Prior"
  - "Deterministic Correspondence"
date: 2026-05-08
content_hash: 04d6e4e70b211bd3
---

# SIMPC: Learning Self-Induced Mirror-Point Consistency for Unsupervised Point Cloud Denoising

**Conference**: ICML 2026  
**arXiv**: [2605.26894](https://arxiv.org/abs/2605.26894)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Point Cloud Denoising, Unsupervised Learning, Mirror-Point Consistency, Geometric Prior, Deterministic Correspondence

## TL;DR
SIMPC proposes performing a "symmetric extension" along the denoising vector of the **same noisy point** to obtain a mirror point located on the other side of the surface. A Mirror-Point Consistency Loss is then used to force the denoising targets of both points to coincide. This shifts unsupervised point cloud denoising from "finding statistical correspondences across multiple noise variants" to "finding deterministic geometric correspondences within a single point." It consistently outperforms unsupervised SOTAs on PUNet/PCNet synthetic data and Paris-Rue-Madame / Kinect real scans, while also surpassing several supervised methods.

## Background & Motivation

**Background**: Point cloud denoising is a critical pre-processing step for downstream tasks such as surface reconstruction and semantic understanding. Supervised methods (PD-Refiner, StraightPCF, PD-LTS, etc.) rely on pairwise noisy–clean data synthesized from CAD models, which limits their generalization. Since devices like LiDAR continuously generate massive amounts of raw noisy scans, **unsupervised point cloud denoising** is a more practical direction.

**Limitations of Prior Work**: Image denoising can align multiple noise observations to the same clean pixel via pixel indices (e.g., Noise2Noise / Noise2Void). However, point clouds **lack fixed spatial indices**, and noise directly perturbs the point coordinates themselves (which carry both position and geometry), making cross-variant point correspondences naturally fragmented. Existing unsupervised routes are incomplete:
- **Noise-based** (Noise4Denoise, Noise2Score3D) injects extra noise $u\sim\mathcal{N}(0,\Delta\sigma)$ and tasks the network with predicting the reverse noise $-u$. However, this correspondence is driven purely by random noise and **does not point toward the true surface**. Inference also requires scaling extrapolation based on additional noise distribution assumptions.
- **EMD-based** (NoiseMap, U-CAN) uses Earth Mover's Distance to perform optimal transport between two noise variants. While more structured than noise-based methods, the transport occurs at the **point set distribution level**, failing to guarantee that "two matched points truly originate from the same surface patch." Essentially, it remains a blurry correspondence.

**Key Challenge**: Existing methods place the "correspondence search" between **independently sampled noise observations**. As long as observations are sampled independently, correspondences inevitably carry stochasticity, leading to drift in the denoising targets. To stabilize unsupervised denoising, a different data source must be used to construct correspondences.

**Goal**: Without introducing new noise observations or distribution assumptions, construct a companion point **only from a single noisy point itself** that strictly corresponds one-to-one and deterministically falls on the "other side" of the underlying surface, forcing the denoising targets of both points to converge to the same location.

**Key Insight**: The denoising vector $d_i$ itself is the network's estimate of "how much and in which direction this point should move toward the surface"—this is an implicit geometric prior. If a point is moved near the surface using $w_1=1$ times $d_i$ to get $\hat{x}_i$, and then "overshot" to the **other side** of the surface using $w_2=2$ times $d_i$ to get a mirror point $\tilde{x}_i$, then $\hat{x}_i$ and $\tilde{x}_i$ are "geometrically symmetric" with respect to the surface. The subsequent denoised points $\hat{x}_i$ and $\bar{x}_i$ calculated from each **must point to the same surface patch**. This is a deterministic correspondence induced by the model itself without extra observations.

**Core Idea**: Use "overshoot–pullback" to split a single noisy point into a pair of geometrically symmetric mirror points, then use MSE to force both sides to be pulled back to the same target. This replaces "blurry cross-observation correspondences" with "self-induced deterministic correspondences."

## Method

### Overall Architecture

SIMPC follows an iterative denoising paradigm: a DGCNN encoder + $L=2$ shared Denoiser Blocks. For a noisy point cloud $X^0\in\mathbb{R}^{N\times3}$, $T=3$ layers of DGCNN ($k=32$, feature concatenation $[g_i\|g_j-g_i]$ processed by MLP) are first used to obtain initial features $U^0\in\mathbb{R}^{N\times 256}$. Inside each Denoiser Block:

1.  **Point Self-Attention (PSA)**: Aggregates features based on spatial neighbors $\hat{\mathcal{N}}_i=\mathrm{KNN}(x_i, X^l, k)$: $f_i=\sum_{j\in\hat{\mathcal{N}}_i}\alpha_{ij}\odot h(u_j)$.
2.  **Decoder**: MLP+tanh outputs a normalized point-wise denoising vector $d_i=\mathrm{Dec}(f_i)\in\mathbb{R}^3$.
3.  **Coordinate Update**: $x_i^l = x_i^{l-1} + d_i^l$.

During training, two independently sampled noise variants $X_a, X_b$ are taken following the U-CAN protocol, but EMD alignment is **not** performed between them. Instead, each follows the SIMPC mirror-point pipeline. During inference, the iterations are applied directly to a single point cloud $L$ times.

### Key Designs

1.  **Mirror-Point Generation Module (MPGM) — Splitting one point into symmetric mirrors**:
    - **Function**: Without introducing extra noise observations, **self-induce** a companion point located on the other side of the underlying surface from a noisy point $x_i$, ensuring a strict one-to-one correspondence.
    - **Mechanism**: The denoising vector $d_i$ predicted by the current Denoiser Block is viewed as an implicit estimate of "surface direction + distance." First, a seed denoised point $\hat{x}_i = x_i + w_1 d_i$ is obtained with $w_1=1$. Then, a mirror point $\tilde{x}_i = x_i + w_2 d_i$ is created via **symmetric extension** with $w_2=2$—this is equivalent to reflecting $\hat{x}_i$ across the underlying surface as a mirror. The mirror point's features $\tilde{f}_i$ are recomputed using a **new neighborhood** $\tilde{\mathcal{N}}_i=\mathrm{KNN}(\tilde{x}_i, X^l\setminus\{x_i\}, k)$, then passed through the same Decoder to get the mirror denoising vector $\tilde{d}_i$, which pulls it back to $\bar{x}_i=\tilde{x}_i+\tilde{d}_i$.
    - **Design Motivation**: Correspondence in old methods comes from **two independent samplings**, making the relationship stochastic; MPGM makes the correspondence **entirely dependent on a single $x_i$ and $d_i$**, and thus deterministic. Since the mirror point is on the "other side" with a different neighborhood, it forces the model to learn the same surface from two complementary perspectives, providing more information than "self-denoising." The $w_2=2$ setting for geometric symmetry ensures $\hat{x}_i$ and $\tilde{x}_i$ are equidistant from the surface, which is optimal in ablations.

2.  **Mirror-Point Consistency Loss (MPCL) — Nailing the pullback points to the same surface position**:
    - **Function**: Translates the "deterministic correspondence" from mirror-point construction into an optimizable training signal, forcing denoising results from both sides to fall at the same physical location.
    - **Mechanism**: Calculates $\mathcal{L}_{\mathrm{MPC}}=\sum_{i=1}^{N}\|\hat{x}_i - \bar{x}_i\|_2^2$ directly for each point $i$. Unlike Chamfer, which only requires "set-level alignment," or EMD, which performs soft matching on distributions, this is a **point-to-point** hard consistency constraint.
    - **Design Motivation**: The reason surface positions are "hard to learn unsupervised" is the lack of a **point-level** position anchor. Using the deterministic pairing from MPGM, MPCL converts the question of "where is the surface" into "where should two clearly symmetric points overlap." The loss reaches 0 only when both fall on the underlying surface, pinning the optimal solution to the surface and alleviating the common "aligned but shifted from surface" issue in EMD methods.

3.  **Chamfer-only Similarity Regularization — Preventing collapse with weak set-level priors**:
    - **Function**: Prevents the point set from collapsing or drifting away from the original distribution during iterative denoising, without handling "correspondence."
    - **Mechanism**: Uses only Chamfer Distance (CD) between two noise variants $X_a, X_b$: $\mathcal{L}_{\mathrm{SR}}^l = \mathrm{CD}(X_a^l, X_b^l) + \mathrm{CD}(X_a^{l-1}, X_b^l) + \mathrm{CD}(X_b^{l-1}, X_a^l)$. Total loss: $\mathcal{L}_{\mathrm{total}}=\sum_{l=1}^{L}(\mathcal{L}_{\mathrm{MPC}}^l + \mathcal{L}_{\mathrm{SR}}^l)$.
    - **Design Motivation**: The responsibility of finding correspondences is shifted entirely to MPCL, allowing $\mathcal{L}_{\mathrm{SR}}$ to degrade into a minimal "distribution similarity" constraint. Ablations show that using **only $\mathcal{L}_{\mathrm{SR}}(\mathrm{CD})$** results in a P2M of 36.20 at 3% noise, and **only $\mathcal{L}_{\mathrm{SR}}(\mathrm{EMD})$** reaches 18.87. Combining $\mathcal{L}_{\mathrm{MPC}}+\mathcal{L}_{\mathrm{SR}}(\mathrm{CD})$ drops P2M to 13.85—proving that deterministic correspondence is the primary performance driver, while CD is sufficient for set-level regularization.

### Loss & Training
The total loss is the sum of MPCL and CD regularization for each Denoiser Block layer: $\mathcal{L}_{\mathrm{total}}=\sum_{l=1}^{L}(\mathcal{L}_{\mathrm{MPC}}^l + \mathcal{L}_{\mathrm{SR}}^l)$, with $L=2$. Training data uses PUNet 40 shapes with Gaussian noise injected at 0.5%–2% of the bounding sphere radius; Adam optimizer, lr $=1\times10^{-4}$, 100 epochs, batch=16, single RTX 4090.

## Key Experimental Results

### Main Results

PUNet Gaussian Noise (CD/P2M $\times 10^5$, lower is better; excerpts for 50K points @ 1% and 10K points @ 3%):

| Dataset/Level | Metric | Prev. SOTA (Unsupervised) | SIMPC (Ours) | Gain | Benchmark (Supervised) |
|------|------|----------------|-------|------|----------------|
| PUNet 50K 1% | CD↓ | 8.33 (Noise2Score3D) | **5.81** | -30% | PD-Refiner 4.66 |
| PUNet 50K 1% | P2M↓ | 2.65 (Score-U) | **1.02** | -61% | PD-Refiner 0.45 |
| PUNet 50K 3% | CD↓ | 24.34 (Noise2Score3D) | **12.58** | -48% | PD-LTS 18.52 (Surpassed) |
| PUNet 50K 3% | P2M↓ | 17.04 (Noise2Score3D) | **6.45** | -62% | PD-LTS 10.67 (Surpassed) |
| PUNet 10K 3% | CD↓ | 36.66 (U-CAN) | **34.42** | -6% | PD-Refiner 30.77 |
| PUNet 10K 3% | P2M↓ | 18.42 (U-CAN) | **13.85** | -25% | PathNet 24.04 (Surpassed) |

PCNet Gaussian + Kinect Real Scans:

| Dataset/Level | Metric | Strongest Unsupervised Baseline | SIMPC (Ours) | Note |
|------|------|---------------------|-------|------|
| PCNet 50K 3% | CD↓ / P2M↓ | Score-U 39.28 / 11.74 | **18.62 / 4.15** | Surpasses supervised HybridPF (19.10/4.80) |
| Kinect Real Scan | CD↓ / P2M↓ | Score-U 15.85 / 7.33 | **13.01 / 6.35** | Surpasses 4 supervised methods (Best: StraightPCF 13.46/7.39) |

### Ablation Study (PUNet 10K, three levels of Gaussian noise, CD/P2M $\times 10^5$)

| Configuration | 1% CD / P2M | 2% CD / P2M | 3% CD / P2M | Description |
|------|--------------|--------------|--------------|------|
| $\mathcal{L}_{\mathrm{SR}}(\mathrm{CD})$ only | 18.91 / 2.47 | 37.84 / 13.91 | 65.73 / 36.20 | CD collapse at high noise |
| $\mathcal{L}_{\mathrm{SR}}(\mathrm{EMD})$ only | 26.54 / 7.64 | 31.88 / 12.41 | 41.04 / 18.87 | Blurry EMD correspondence, high P2M |
| **$\mathcal{L}_{\mathrm{MPC}}+\mathcal{L}_{\mathrm{SR}}(\mathrm{CD})$ (Full)** | **20.25 / 3.60** | **28.82 / 7.13** | **34.42 / 13.85** | MPCL is the main driver |
| $w_2=1.5$ (Near) | 20.77 / 4.06 | 29.31 / 7.68 | 35.10 / 14.55 | Asymmetric, too small |
| $w_2=2$ (Symmetry) | 20.25 / 3.60 | 28.82 / 7.13 | 34.42 / 13.85 | Geometric symmetry is optimal |
| $w_2=2.5$ (Far) | 21.39 / 4.84 | 30.13 / 8.54 | 36.33 / 15.25 | Too far, introduces noise |

### Key Findings
- **MPCL is the absolute core**: Replacing the loss from EMD to MPCL cuts P2M from 18.87 to 13.85 (-27%) and CD from 41.04 to 34.42 (-16%) at 3% noise.
- **Geometric symmetry $w_2=2$ is the "sweet spot"**: Deviations from $w_2=2$ (both 1.5 and 2.5) consistently degrade performance, supporting the theoretical explanation that symmetry ensures equidistant placement from the surface.
- **P2M improvements generally exceed CD improvements**: This indicates SIMPC's results are not just "point-cloud-like" but are truly adhered to the surface—this is exactly what the deterministic correspondence target optimizes.
- **Strongest generalization under non-Gaussian noise (Laplacian / Discrete)**: At Discrete 50K 1% noise, SIMPC achieves P2M 0.32, approaching the supervised best (PD-Refiner 0.12) and far lower than the second-best unsupervised (Score-U 0.82). This shows SIMPC does not overfit to specific noise distribution assumptions.

## Highlights & Insights
- **Shifting correspondence from data space to model space**: While previous unsupervised methods focused on "constructing better multiple noise observations," SIMPC does the opposite—since multiple observations inherently introduce stochasticity, it uses **only one observation** and lets the model's own prediction $d_i$ act as a geometric prior to generate correspondences. This transforms correspondence from a data problem into a network self-feedback problem.
- **Overshoot–pullback is a transferable "self-supervised geometric construction"**: It converts the implicit target of "being on the surface" into an explicit MSE target of "two symmetric points must overlap." This logic could theoretically extend to unsupervised surface reconstruction, SDF fitting, or even unsupervised 3D registration.
- **Minimal loss outperforms complex transport**: Abandoning EMD in favor of Chamfer as a fallback allows MPCL's deterministic constraints to shine, proving that simple distribution alignment is better when strong point-wise signals are present.
- **Surpassing supervised methods is noteworthy**: SIMPC outperforms several supervised methods on PCNet 3% and Kinect scans. This is likely because supervised methods are biased toward synthetic CAD noise assumptions, whereas SIMPC's geometric prior is extracted from the noisy point cloud **itself**, making it more robust to real-world noise.

## Limitations & Future Work
- The authors acknowledge that MPGM relies entirely on the Denoiser Block's current prediction $d_i$ for mirror directions. During the **cold-start phase**, $d_i$ is noisy and direction estimates may be inaccurate, possibly placing mirror points on the wrong side. Iteration ($L=2$) mitigates this, but early training convergence curves are not reported.
- **Fixed $w_1, w_2$ values rely on a local "approximate plane" assumption**: In high-curvature regions, "symmetric extension" might not truly land the mirror point on the opposite side. This might explain why CD improvements on PCNet (which has more detail) are smaller than on PUNet. A possible improvement is adaptive $w_2$ based on local curvature.
- Systematic testing was only performed on mathematical noise (Gaussian/Laplacian/Discrete); evaluation on complex real-world noise like LiDAR multi-path or specular reflections is limited to qualitative results for Kinect/Paris.
- **The $L=2$ iteration cap may be a performance bottleneck**: Diffusion denoising paradigms typically use more steps, but SIMPC's mirror point and second PSA calculation double the cost per step. Lightweight mirror-point variants could be explored.

## Related Work & Insights
- **vs Noise4Denoise / Noise2Score3D (Noise-based)**: These rely on injected noise $u$ to create "dirtier-cleaner" pairs and require extrapolation results during inference; SIMPC injects nothing and induces correspondences from $d_i$, **requiring no noise distribution assumptions**, hence the significant gains on non-Gaussian noise.
- **vs NoiseMap / U-CAN (EMD-based)**: These use EMD for "distribution alignment," which doesn't guarantee point-wise surface landing (evidenced by U-CAN's high P2M-to-CD ratio). SIMPC uses MPCL to nail correspondences at the point level.
- **vs PD-Refiner / StraightPCF / PD-LTS (Supervised)**: These rely on synthetic noisy–clean pairs. SIMPC trains only on raw noisy scans and surpasses them on real scans, proving that "self-supervised geometric priors are closer to real distributions than synthetic noise priors."
- **vs IterativePFN**: While using a similar DGCNN+iterative architecture, SIMPC gracefully replaces supervised signals with MPCL, serving as a prime example of "unsupervised adaptation of existing architectures."

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ "Using the model's own denoising vector to construct symmetric mirror points" is a clean, interpretable idea that aligns directly with the geometric essence of point cloud denoising, breaking the convention that correspondences must come from multiple observations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers PUNet/PCNet x 3 Gaussian levels + 4 non-Gaussian types + Kinect/Paris real datasets. High precision in isolating MPCL and $w_2$. Deducted 1 star for missing training dynamic analysis and failure case visualization in high-curvature areas.
- **Writing Quality**: ⭐⭐⭐⭐ Fig. 1 clearly illustrates the paradigm shift; the method section is well-structured. Minor formula inconsistencies (e.g., swapping $\bar{x}_i$ and $\mathrm{x}_i$).
- **Value**: ⭐⭐⭐⭐⭐ Provides a universal paradigm for "model self-feedback induced geometric correspondences" in unsupervised 3D tasks, applicable to SDF fitting, surface reconstruction, and point cloud completion.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] U-CAN: Unsupervised Point Cloud Denoising with Consistency-Aware Noise2Noise Matching](../../NeurIPS2025/3d_vision/u-can_unsupervised_point_cloud_denoising_with_consistency-aware_noise2noise_matc.md)
- [\[ICCV 2025\] Noise2Score3D: Tweedie's Approach for Unsupervised Point Cloud Denoising](../../ICCV2025/3d_vision/noise2score3d_tweedies_approach_for_unsupervised_point_cloud_denoising.md)
- [\[ICML 2026\] SplAttN: Bridging 2D and 3D with Gaussian Soft Splatting and Attention for Point Cloud Completion](splattn_bridging_2d_and_3d_with_gaussian_soft_splatting_and_attention_for_point_.md)
- [\[CVPR 2026\] Deformation-based In-Context Learning for Point Cloud Understanding](../../CVPR2026/3d_vision/deformation-based_in-context_learning_for_point_cloud_understanding.md)
- [\[ICML 2026\] Geodesic Flow Matching for Denoising High-Dimensional Structured Representations](geodesic_flow_matching_for_denoising_high-dimensional_structured_representations.md)

</div>

<!-- RELATED:END -->
