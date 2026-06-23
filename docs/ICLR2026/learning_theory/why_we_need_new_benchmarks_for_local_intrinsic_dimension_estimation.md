---
title: >-
  [Paper Note] Why We Need New Benchmarks for Local Intrinsic Dimension Estimation
description: >-
  [ICLR 2026][learning_theory][benchmark] This paper argues that current neural LID estimators evaluate themselves on outdated benchmarks: either overly simple synthetic data or real data with unknown LID, which masks true performance. The authors propose a principled benchmark construction toolbox (mapping the same manifold to multiple domain representations,
tags:
  - ICLR 2026
  - learning_theory
  - benchmark
date: 2026-05-08
content_hash: b0f00da65a94f005
---
# Why We Need New Benchmarks for Local Intrinsic Dimension Estimation

**Conference**: ICLR 2026  
**Paper**: [OpenReview / ICLR 2026 Conference Paper](https://openreview.net/) (⚠️ Please refer to the original text for the specific link)  
**Code**: https://github.com/DominikFilipiak/LID-Benchmarks (Available)  
**Area**: Learning Theory / Manifold Learning / Local Intrinsic Dimension Estimation / Benchmark  
**Keywords**: Local Intrinsic Dimension (LID), Manifold Geometry, Benchmark, Cross-architecture Evaluation, Stress Testing

## TL;DR
This paper argues that current neural LID estimators evaluate themselves on outdated benchmarks: either overly simple synthetic data or real data with unknown LID, which masks true performance. The authors propose a principled benchmark construction toolbox (mapping the same manifold to multiple domain representations, designing harder variants for key manifold properties, and controlled transformations with known LID changes for stress testing). They systematically prove that high precision on simple manifolds "does not transfer"; almost all SOTA methods reveal clear failure modes under targeted stress tests.

## Background & Motivation

**Background**: Local Intrinsic Dimension (LID) estimation addresses the question: "On how many dimensions does data truly reside near a specific point?" Recent mainstream approaches use generative models to estimate LID—LIDL (using normalizing flows to estimate the slope of diffusion density), FLIPD (a Fokker-Planck perspective based on diffusion models), NB (the normal-bundle method by Stanczuk et al.), and the classic ESS (a neighborhood-based non-neural method). Algorithms are being released at an increasing pace.

**Limitations of Prior Work**: Evaluation has remained stagnant, following two flawed paths:
- **Path 1: Purely synthetic data**. Sampling from analytical distributions (Gaussian, uniform cubes, spheres) where the ground truth LID is known, making the evaluation "reliable." However, these datasets are too simple to capture the complexity of real manifolds; most works fail to test boundary cases like "manifold thickness variations" or "adjacent manifolds," and even systematic biases on simple distributions like Gaussians are not analyzed.
- **Path 2: Real-domain datasets** (images, audio, etc.). While the complexity is appropriate, the LID is unknown—the correct answer is unavailable, making reliable judgment impossible and leading to erroneous conclusions.

**Key Challenge**: The "realism" and "verifiability" of evaluation are mutually exclusive—synthetic data is verifiable but unrealistic, while real data is realistic but unverifiable. More subtly, neural LID estimators are tied to domain-specific architectures (CNNs vs. MLPs). Their inductive biases cause the **same underlying manifold** to yield **inconsistent** estimates under different domain representations. The silent assumption that "verification on simple synthetic manifolds $\rightarrow$ similar performance across various domain networks" has never been tested.

**Goal**: Rather than proposing another LID estimator, this paper focuses on **how to test** these estimators. It addresses three sub-questions: (1) How to transfer the same manifold to different domain representations without destroying its structure for rigorous cross-architecture comparison; (2) How to design harder datasets targeting key manifold properties (curvature, boundaries, thickness, adjacency, etc.); (3) How to perform verifiable stress tests on real data where LID is unknown.

**Key Insight**: The authors leverage a geometric fact: **diffeomorphisms preserve manifold dimension**. If a sufficiently regular transformation (continuously differentiable, monotonic, non-degenerate Jacobian) is applied to the data, the underlying LID either remains identical or shifts by a known amount. This provides a pivot for verification on "unknown LID" data: one does not need the absolute LID, only the **expected difference in LID before and after the transformation**.

**Core Idea**: Upgrade "benchmarking" from "finding a bunch of datasets" to "a toolbox of structure-preserving/dimension-controlling transformations + a systematic failure mode matrix"—using transformations to explicitly control ground truth and expose the true shortcomings of existing methods in curvature, boundaries, adjacent manifolds, sample size, and domain transfer.

## Method

### Overall Architecture
The "Method" in this paper is not an estimation algorithm but a **toolbox for constructing controlled benchmarks**, which is then used to systematically evaluate existing LID estimators. The pipeline involves: taking a **source manifold with known (or known-difference) LID** $\rightarrow$ applying **structure-preserving or dimension-controlling transformations** to move it into a real domain (focusing on images) or applying targeted distortions $\rightarrow$ obtaining a test set for a "key manifold property" $\rightarrow$ running various LID estimators (ESS / NB / LIDL / FLIPD) $\rightarrow$ comparing estimated values with known LID (or LID differences) to map successes and failures for each method.

The toolbox contains 5 types of transformations, grouped by their effect on LID: **LID-invariant** (IDR, ME, ASE) to check if algorithms change values where they shouldn't; and **LID-altering by known amounts** (ADI for dimension increase, MS for synthesis) to check if algorithms can follow controlled dimensional changes. These cover 11 test aspects (non-uniform density, curvature, boundaries, thin manifolds, adjacent manifolds, architectural invariance, sample size dependence, artificial dimension injection, upsampling, stretching, and real-like synthesis).

### Key Designs

**1. IDR (Inverse Domain Representation): Comparing different architectures under identical conditions**

This addresses the assumption that performance is similar across domain networks. IDR solves a specific embedding problem: given an arbitrary manifold $M$ and a real dataset $X \subset \mathbb{R}^D$ (e.g., FMNIST), embed $M$ into the ambient space of $X$ such that the image of the embedding is "close" to $X$. The key is that this is a **structure-preserving** embedding—the intrinsic dimension of the underlying manifold remains unchanged; only its "appearance" changes (e.g., from analytic coordinates of a 5D Gaussian to images). This allows feeding the **exact same** 1D/2D/5D manifold to both MLP and CNN versions of estimators for a true like-for-like comparison.

**2. Harder Variants for Key Manifold Properties: Explicitly engineering geometry into data**

This targets the common geometric corner cases in real manifolds that simple benchmarks ignore. The authors design minimal but lethal synthetic manifolds, then move them into the image domain via IDR:
- **Non-uniform density**: A mixture of four 5D Gaussians with different standard deviations ($1/27, 1/9, 1/3, 1$). Theoretically, the LID should be 5 regardless of the distance from the peak. Only ESS succeeded; neural methods showed bias (e.g., LIDL's bias is proportional to the Laplacian of the density).
- **Curvature**: Four $S^4$ spheres with different radii—only ESS was unaffected by curvature.
- **Boundaries**: A 20D uniform cube. Dimensions were underestimated for points near the edges ($\pm 3$). Since real images naturally fall on cube boundaries due to pixel saturation, this is a significant issue.
- **Thin Manifolds (Moon)**: A 3D manifold that becomes increasingly thin in the normal direction. Algorithms misclassify it as lower-dimensional when extremely thin.
- **Adjacent Manifolds (Funnel / Spiral)**: Estimation collapses when the "distance to the nearest neighbor on another manifold segment" $\approx$ "distance to local neighbors."

**3. Controlled Transformations with Known LID Differences: Stress testing on real data**

This addresses the "unknown absolute LID" problem. The strategy is to verify the **known difference in LID before and after transformation** using geometric properties of diffeomorphisms/immersions:
- **ME (Monotonic Embedding)**: Applying a continuously differentiable, monotonic transformation to each coordinate (e.g., $x \in [0, 1] \mapsto y = x^l, l \in \{0.25, 4\}$). As a diffeomorphism, LID should remain **identical**. ESS remained stable, while NB/LIDL dropped and FLIPD spiked.
- **ASE (Ambient Space Extension)**: Expanding the ambient dimension (e.g., bilinear upsampling an image to 32x32). LID remains unchanged, but the hierarchical features learned by CNNs change drastically, deceiving estimators.
- **ADI (Auxiliary Dimension Injection)**: Applying an immersion with a non-degenerate Jacobian to **increase** the intrinsic dimension by a known amount (e.g., concatenating two MNIST images or adding mirror padding with random brightness). Subtracting the injected dimensions from the estimate should ideally fall on the identity line.

**4. MS (Manifold Synthesis): Creating "Real-like" data with precise LID**

This targets the lack of datasets that are both "realistic" and "LID-known." MS generates data by applying deterministic continuous transformations to a **parameterized manifold**. The **Arrows** dataset consists of 32x32 images of arrows where each arrow is described by 6 variables (position, rotation, RGB). Thus, the manifold dimension is exactly $6 \times \text{number of arrows}$. No algorithm came close to the ground truth here; ESS outputted $\approx 15$ regardless of true dimension, and NB showed extreme overestimation.

### Loss & Training
This paper does not train new models or use new loss functions. It evaluates existing LID estimators: **ESS** (classic non-neural), **NB**, **LIDL**, and **FLIPD** (neural methods using normalizing flows or diffusion models). A critical implementation detail: LIDL originally used Glow, which performed poorly; the authors switched to MAF which is an order of magnitude faster and more effective.

## Key Experimental Results

### Main Results: Performance Matrix (Algorithm × Manifold Property)
The authors summarize all experiments into an H/M/L/O rating table (High / Medium / Low / Out-of-range).

| Algorithm | Non-uniform Density | Curvature | Boundary | Thin Manifold | Adjacency | Sample Dependence | ADI | Upsampling | Stretch | Real-like Synthesis |
|-----------|---------------------|-----------|----------|---------------|-----------|-------------------|-----|------------|---------|---------------------|
| ESS       | H                   | H         | L        | H             | H         | L                 | L   | L          | M       | L                   |
| NB        | L                   | L         | M        | L             | L(M)      | **H**             | M   | M          | L       | O                   |
| LIDL      | O                   | O         | O        | O             | O         | L                 | L   | L          | L       | —                   |
| FLIPD     | O                   | O         | O        | O             | O         | L                 | L   | L          | O       | L                   |

(Parentheses indicate self-reported performance in original papers; "O" means the estimate exceeded valid ranges. ⚠️ Refer to Table 1 / Appendix A in the original paper for the full matrix.)

**Key finding**: **No algorithm is globally proficient**. ESS is strongest on low-dimensional geometric properties but fails on real-data transformations. Neural methods (LIDL/FLIPD) often go out-of-range ("O") on several synthetic geometric aspects.

### Key Findings
- **High precision on simple manifolds is non-transferable**: Algorithms that show no difference on classic benchmarks diverge significantly on domain-adapted benchmarks.
- **Failure modes are structured and locatable**: Geometric properties like curvature, boundaries, and adjacency correspond to specific failure directions rather than random noise.
- **Parameter selection is a chronic issue for LIDL/FLIPD**: While theory proves LIDL is unbiased as $\delta \to 0$, in practice, estimates are often biased toward the ambient dimension. Hyperparameters like $n\_neighbours$ for ESS also lack a priori setting methods.
- **The "Arrows" dataset causes collective failure**: Complex appearances with precisely known LID represent the toughest current challenge, showing existing methods are far from being usable on complex real data.

## Highlights & Insights
- **Turning "Unknown LID" into "Verifiable"**: Using the fact that diffeomorphisms preserve dimension allows anchoring ground-truth differences on real data. This methodology is transferable to any evaluation scenario where absolute truth is unavailable but relative change is controllable.
- **IDR provides a "clean" control experiment**: By keeping the underlying manifold constant and only changing the domain representation, the authors isolate the "architectural inductive bias" and disprove the domain-invariant assumption.
- **Failure Mode Matrix > Single Leaderboard**: Rather than a single score, the H/M/L/O matrix allows researchers to pinpoint exactly where their methods need improvement.

## Limitations & Future Work
- **Limited to the Image Domain**: While the toolbox applies to any continuous domain (audio, EEG, etc.), experiments were restricted to images.
- **Qualitative/Visual Assessment**: Evaluation currently relies on per-property charts and H/M/L/O ratings; a unified quantitative metric is still missing.
- **Uncovered Aspects**: The impact of quantization on LID, higher-dimensional manifolds, and data noise were not tested.

## Related Work & Insights
- **vs. Purely Synthetic Benchmarks**: Those are too simple and ignore corner cases. Ours adds complexity and realism via IDR/MS while keeping LID verifiable.
- **vs. Real-world Dataset Evaluation**: Those have unknown LID. Ours utilizes ME/ASE/ADI to provide "verifiability" to real data.
- **vs. Evaluated Methods (LIDL, etc.)**: This paper provides a "diagnostic bench," proving that gaps unseen in classic benchmarks are amplified under domain-adapted stress tests.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ (Upgrading benchmarks to a transformation toolbox and using geometry to solve the "unknown LID" verification problem is novel.)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Covers 11 properties × 4 algorithms; however, restricted to the image domain.)
- **Writing Quality**: ⭐⭐⭐⭐ (Logical and clear; density of the appendix is high.)
- **Value**: ⭐⭐⭐⭐⭐ (Provides a necessary diagnostic tool for the fast-developing field of LID estimation.)

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] The Price of Robustness: Stable Classifiers Need Overparameterization](the_price_of_robustness_stable_classifiers_need_overparameterization.md)
- [\[ICLR 2026\] Random-Projection Ensemble Dimension Reduction](random-projection_ensemble_dimension_reduction.md)
- [\[ICLR 2026\] Intrinsic Entropy of Context Length Scaling in LLMs](intrinsic_entropy_of_context_length_scaling_in_llms.md)
- [\[ICLR 2026\] Prediction with Expert Advice under Local Differential Privacy](prediction_with_expert_advice_under_local_differential_privacy.md)
- [\[ICLR 2026\] A New Approach to Controlling Linear Dynamical Systems](a_new_approach_to_controlling_linear_dynamical_systems.md)

</div>

<!-- RELATED:END -->
