---
title: >-
  [Paper Note] Probability Density Geodesics in Image Diffusion Latent Space
description: >-
  [CVPR 2025][Image Generation][Probability Density Geodesics] This paper demonstrates that probability-density-based geodesics can be computed in the latent space of diffusion models, where paths traversing high-probability density regions are "shorter" than those through low-density regions. It also showcases the application of this technique in video approximation analysis, training-free image sequence interpolation, and extrapolation.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Probability Density Geodesics"
  - "Diffusion Models"
  - "Latent Space"
  - "Image Interpolation"
  - "Riemannian Geometry"
date: 2026-05-08
content_hash: 82efdf8200fdedaa
---

# Probability Density Geodesics in Image Diffusion Latent Space

**Conference**: CVPR 2025  
**arXiv**: [2504.06675](https://arxiv.org/abs/2504.06675)  
**Code**: None  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Probability Density Geodesics, Diffusion Models, Latent Space, Image Interpolation, Riemannian Geometry

## TL;DR
This paper demonstrates that probability-density-based geodesics can be computed in the latent space of diffusion models, where paths traversing high-probability density regions are "shorter" than those through low-density regions. It also showcases the application of this technique in video approximation analysis, training-free image sequence interpolation, and extrapolation.

## Background & Motivation

**Background**: Diffusion models are currently among the most powerful image generation models, implicitly estimating the probability density over the data space through a learned step-by-step denoising process. Methods like DDPM and DDIM have been widely used for image generation, editing, and understanding. Recently, researchers have begun to focus on the geometric structure of the diffusion latent space to enable more controllable generation.

**Limitations of Prior Work**: Although diffusion models implicitly encode the probability density information of the data distribution, a systematic method to utilize this geometric structure is currently lacking. While linear interpolation in the latent space is a common practice to generate intermediate images, linear paths do not necessarily pass through high-probability regions and may traverse low-density areas, leading to unnatural intermediate results (e.g., linear interpolation between two faces might pass through blurry or distorted images).

**Key Challenge**: Linear interpolation assumes that the latent space is a flat Euclidean space, but the actual image manifold is curved—where "distances" in certain directions should be shorter than in others (paths through common images should be shorter than through rare ones). A distance metric that respects the probability density structure is needed.

**Goal**: To define and compute probability-density-based geodesics in the diffusion latent space, where the inner product is inversely proportional to the probability density—meaning paths through high-density regions are shorter—and to demonstrate practical applications of this geometric structure.

**Key Insight**: Driven by Riemannian geometry, a spatially varying inner product (metric) is defined in the diffusion latent space, whose induced norm is inversely proportional to the probability density. The score function (i.e., the gradient of the log-probability density) of the diffusion model is leveraged to compute this metric, thereby linking the geodesic computation to pre-trained diffusion models.

**Core Idea**: To treat the diffusion latent space as a Riemannian manifold equipped with a probability-density-induced metric and compute geodesics on it, thereby achieving image space navigation that "takes the shortest path along high-probability regions."

## Method

### Overall Architecture
**Input**: A pre-trained image diffusion model (no additional training required) and starting/ending images (or an initial direction).  
**Output**: An image sequence along the geodesic path connecting the two points, along with the probability densities and the geodesic distance along the path.  
The method consists of three core components:  
(1) Defining a probability-density-induced Riemannian metric;  
(2) Solving the initial value problem (IVP) and boundary value problem (BVP);  
(3) Applying the results to image analysis and generation.

### Key Designs

1. **Probability-density-induced Riemannian metric**:
    - **Function**: To define a spatially varying inner product/distance in the diffusion latent space.
    - **Mechanism**: At point $x$ in the latent space, the inner product is defined such that the induced norm is inversely proportional to the probability density $p(x)$:
      $$\|v\|_x \propto \frac{\|v\|_2}{p(x)}$$
      This implies that in high-density regions, the "cost" of moving one step is low (shorter path), whereas in low-density regions, the cost is high. The score function of the diffusion model $s(x) = \nabla_x \log p(x)$ provides the gradient information required for $p(x)$, allowing this metric to be obtained entirely from pre-trained diffusion models without any extra training.
    - **Design Motivation**: Standard Euclidean distance does not differentiate between high- and low-density regions, which causes linear interpolation to potentially cross "image deserts." The probability density metric forces paths to favor common, natural image regions, resulting in more plausible intermediate results.

2. **Numerical solution of geodesic equations**:
    - **Function**: To solve for the geodesics (shortest paths) under the defined Riemannian metric.
    - **Mechanism**: Geodesics satisfy a system of second-order ordinary differential equations (ODEs), known as the geodesic equations. The authors propose two solvers: (a) an initial value problem (IVP) solver—which marches along the geodesic via numerical integration (e.g., Runge-Kutta methods) given a starting point and an initial direction; (b) a boundary value problem (BVP) solver—which finds the correct initial direction via the shooting method to reach the destination given both start and end points. In numerical solving, each step requires evaluating the diffusion model's score function to compute the gradient of the probability density at the current position.
    - **Design Motivation**: IVP is suitable for geodesic extrapolation (extending from a known path), while BVP is suitable for geodesic interpolation (finding the shortest path between two known images). The two solvers cover the main application scenarios.

3. **Probability density calculation and geodesic distance along the path**:
    - **Function**: To calculate the probability density of each point on the geodesic and the geodesic distance between two points.
    - **Mechanism**: The probability density is computed by integrating the divergence of the score function along the path, leveraging the properties of the probability flow ODE. The geodesic distance is obtained by integrating the local norm along the geodesic:
      $$d(x_0, x_1) = \int_0^1 \|\dot{\gamma}(t)\|_{\gamma(t)} dt$$
      These quantities provide a quantitative analysis of the geometric structure of the diffusion latent space.
    - **Design Motivation**: Geodesic distance provides a more semantically meaningful measure of "image similarity" than Euclidean distance. Analyzing the probability density along the path can reveal whether a video clip moves along the natural image manifold.

### Loss & Training
This method requires no training and relies entirely on pre-trained diffusion models. The computational overhead primarily stems from the numerical solution of the geodesic equations, which requires a forward pass of the diffusion model at each step to compute the score function.

## Key Experimental Results

### Video and Geodesic Approximation Analysis

| Analysis | Key Findings |
|---------|---------|
| Natural Videos vs. Geodesics | Natural video clips approximately move along geodesics in the diffusion latent space with minor deviations. |
| Fast Motion vs. Slow Motion | Slow, smooth videos are closer to geodesics, while fast motions deviate more. |
| Probability Density Variation Along Path | Geodesic paths maintain a high probability density, whereas linear interpolation paths exhibit a significant drop in density in the middle segment. |

### Image Interpolation Comparison

| Method | Visual Quality | Naturalness of Intermediate Frames | Path Probability Density |
|------|---------|------------|------------|
| Geodesic Interpolation (Ours) | High | Natural transition | Consistently high |
| Linear Interpolation (LERP) | Medium | Prone to artifacts | Drops in the middle segment |
| Spherical Interpolation (SLERP) | Medium | Similar to LERP | Slightly better than LERP |

### Key Findings
- **Natural videos approximate geodesics**: Analysis of multiple real videos shows that inter-frame transitions closely match geodesic paths. This offers a theoretical perspective: video can be understood as geodesic motion on the image manifold.
- **Geodesic interpolation avoids low-density regions**: Compared to linear interpolation, the geodesic interpolation path consistently maintains a higher probability density, producing more natural and plausible intermediate frames.
- **Geodesic distance is more semantically meaningful than Euclidean distance**: Two images that are semantically close but visually different (e.g., the same person with different expressions) have a geodesic distance much smaller than their Euclidean distance, as a high-probability density path connects them.
- **Sensitivity to initialization in the BVP solver**: The shooting method for solving boundary value problems is highly sensitive to the initial direction guess, sometimes converging to suboptimal solutions in high-dimensional latent spaces.

## Highlights & Insights
- **An elegant combination of Riemannian geometry and diffusion models**: Defining a Riemannian metric using the inherent score function of diffusion models is a theoretically elegant integration. This framework lays a mathematical foundation for analyzing and utilizing the geometric structure of the diffusion latent space.
- **The "video as geodesic" finding has profound implications**: This discovery implies that video generation can be formulated as a geodesic extrapolation problem, providing a new theoretical perspective for video generation. It can be applied to video prediction—given the initial frames, future frames can be predicted by extrapolating along the geodesic.
- **Training-free image interpolation and extrapolation**: Requiring no additional training or fine-tuning, high-quality image sequences can be generated using pre-trained diffusion models. This "plug-and-play" characteristic makes the approach highly practical.

## Limitations & Future Work
- **High computational cost**: Solving geodesics requires multiple evaluations of the diffusion model's score function, resulting in computational costs that are dozens of times higher than a single generation. Under high-resolution settings, real-time application is currently impractical.
- **Stability of high-dimensional BVPs**: Solving boundary value problems in high-dimensional latent spaces is challenging; the shooting method may fail to converge or get trapped in local optima.
- **Non-unique choice of metric**: The inverse of the probability density is only one possible metric; other density-related metrics (such as the Fisher information metric) may possess different properties and applications.
- **Validation limited to image space**: The efficacy has not yet been validated on diffusion models for other modalities (e.g., audio, 3D data).
- **Potential improvement directions**: Accelerating geodesic computation through distillation or approximation, and integrating the geodesic framework with flow matching methods (since the latter inherently involves optimal transport paths).

## Related Work & Insights
- **vs. DDIM Linear Interpolation**: Linear interpolation in the noise space via DDIM is the simplest baseline. This paper shows that linear paths are suboptimal, and geodesic paths produce more natural intermediate results.
- **vs. Riemannian Score-Based Generative Models**: While prior works define diffusion processes on manifolds, this paper reverses the perspective by utilizing diffusion models to define manifold metrics, serving as a complementary approach.
- **vs. Latent Space Optimal Transport**: Optimal transport focuses on mapping between two distributions, whereas geodesics focus on the shortest path between two points. Both leverage spatial geometry but address different problems.
- **vs. Flow Matching / Rectified Flow**: Flow matching learns velocity fields for straight paths, whereas the geodesics in this paper represent "shortest paths" in terms of probability density. The two differ fundamentally in their definition of paths.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Combining Riemannian geometry with the score functions of diffusion models is an elegant and original theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐ The experiments primarily present qualitative demonstrations and analytical results, lacking large-scale quantitative evaluations and comparisons with more baselines.
- Writing Quality: ⭐⭐⭐⭐ Rigorous mathematical derivations with clear explanations of core concepts.
- Value: ⭐⭐⭐⭐ Provides a novel mathematical tool for understanding and exploiting the latent space of diffusion models, with potential far-reaching impacts on image/video generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Latent Space Imaging](latent_space_imaging.md)
- [\[CVPR 2025\] Visual Lexicon: Rich Image Features in Language Space](visual_lexicon_rich_image_features_in_language_space.md)
- [\[CVPR 2026\] Latent Diffusion Inversion Requires Understanding the Latent Space](../../CVPR2026/image_generation/latent_diffusion_inversion_requires_understanding_the_latent_space.md)
- [\[CVPR 2025\] Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models](diffusion-4k_ultra-high-resolution_image_synthesis_with_latent_diffusion_models.md)
- [\[ICCV 2025\] What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization](../../ICCV2025/image_generation/whats_in_a_latent_leveraging_diffusion_latent_space_for_domain_generalization.md)

</div>

<!-- RELATED:END -->
