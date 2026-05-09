---
title: >-
  [Paper Note] The Curse of Conditions: Analyzing and Improving Optimal Transport for Conditional Flow-Based Generation
description: >-
  [ICCV 2025][Image Generation][Optimal Transport] This paper identifies the "curse of conditions" in conditional flow matching — a training-test mismatch caused by standard optimal transport (OT) ignoring conditioning information, which induces a conditionally skewed prior during training while an unbiased prior is used at test time. The authors propose C²OT (Conditional Optimal Transport), which resolves this issue by incorporating a condition-weighted term into the OT cost matrix.
tags:
  - ICCV 2025
  - Image Generation
  - Optimal Transport
  - Conditional Generation
  - Flow Matching
  - Conditionally Skewed Prior
  - ODE Solving
date: 2026-05-08
content_hash: 6015e10ff41fdc5d
---

# The Curse of Conditions: Analyzing and Improving Optimal Transport for Conditional Flow-Based Generation

**Conference**: ICCV 2025
**arXiv**: [2503.10636](https://arxiv.org/abs/2503.10636)
**Code**: [https://github.com/hkchengrex/C2OT](https://github.com/hkchengrex/C2OT)
**Area**: Diffusion Models / Image Generation
**Keywords**: Optimal Transport, Conditional Generation, Flow Matching, Conditionally Skewed Prior, ODE Solving

## TL;DR

This paper identifies the "curse of conditions" in conditional flow matching — a training-test mismatch caused by standard optimal transport (OT) ignoring conditioning information, which induces a conditionally skewed prior during training while an unbiased prior is used at test time. The authors propose C²OT (Conditional Optimal Transport), which resolves this issue by incorporating a condition-weighted term into the OT cost matrix.

## Background & Motivation

**State of the Field**: Flow Matching is a mainstream generative paradigm on par with diffusion models. It learns a vector field mapping from a noise distribution to a data distribution. In the unconditional setting, minibatch OT coupling is a widely adopted technique — pairing noise samples with data samples via OT yields straighter flow trajectories, enabling high-quality generation with fewer ODE integration steps at test time.

**Limitations of Prior Work**: When flow matching is extended to conditional generation (e.g., class-conditional image generation), standard minibatch OT performs poorly — sometimes worse than simple independent coupling (IC) without OT. This phenomenon has been observed in the community, but lacks a theoretical explanation or an effective remedy. Many conditional generation works have consequently abandoned OT coupling altogether.

**Root Cause**: Standard OT computes couplings based solely on the distance between noise and data samples, completely ignoring conditioning information. This leads to a severe training-test mismatch: because OT preferentially pairs spatially close noise-data pairs, certain condition classes may be concentrated in specific regions of the noise space, resulting in a conditionally skewed prior during training. At test time, however, the standard unbiased Gaussian prior is sampled without knowledge of this skew, causing a distributional mismatch and degraded generation quality.

**Paper Goals**: (1) Provide a theoretical analysis of why conditional OT fails; (2) Propose a simple yet effective fix.

**Starting Point**: The authors identify that OT is inherently "greedy" — it pairs spatially close noise and data regardless of their condition labels. If data samples from different conditions exhibit cluster structure in feature space, OT causes different conditions to "partition" the noise space, inducing prior skew. The proposed fix is intuitive: incorporate condition information as a penalty term in the OT cost matrix, so that same-condition pairs are prioritized.

**Core Idea**: Augment the OT cost matrix with a condition similarity weight — assigning lower transport cost to noise-data pairs sharing the same condition and higher cost to cross-condition pairs — thereby preventing the conditionally skewed prior.

## Method

### Overall Architecture

C²OT introduces a minimal modification: only the minibatch OT coupling computation during flow matching training is changed, replacing the standard distance-only cost matrix with a condition-aware weighted cost matrix. The inference procedure remains identical to standard flow matching — sampling from a standard Gaussian prior and integrating via an ODE solver.

### Key Designs

1. **Analysis of Conditionally Skewed Prior (Theoretical Contribution)**:

    - Function: Explains why standard OT fails in conditional generation.
    - Mechanism: Consider a simple example with two condition classes A and B, where class A occupies the left side of the data space and class B the right. Standard OT maps the left half of the noise space to A and the right half to B. Consequently, the effective "prior" for class A during training is not a standard Gaussian but a left-biased truncated Gaussian. At test time, sampling from the full Gaussian for class A may draw noise from the region "belonging to B," degrading generation quality. This skew is visualized via a 8-gaussians-to-moons toy experiment.
    - Design Motivation: Provides the theoretical foundation for C²OT. Understanding the root cause is prerequisite to designing the correct solution.

2. **Condition-Weighted Cost Matrix (C²OT)**:

    - Function: Incorporates condition information into OT pairing to prevent cross-condition partitioning of the noise space.
    - Mechanism: The standard OT cost matrix $C_{ij} = \|x_i - z_j\|^2$ is augmented with a condition penalty: $C_{ij}^{C^2OT} = \|x_i - z_j\|^2 + \lambda \cdot d(c_i, c_j)$, where $c_i, c_j$ are the conditions of data sample $x_i$ and noise sample $z_j$ respectively, $d(\cdot, \cdot)$ is a condition distance, and $\lambda$ controls the strength of the condition constraint. For discrete conditions (e.g., class labels), $d$ is an indicator function (penalizing cross-class pairs); for continuous conditions (e.g., timestamps), $d$ is the L2 distance.
    - Design Motivation: With the condition penalty, OT preferentially pairs same-condition noise and data. As $\lambda \to \infty$, C²OT degenerates to independent coupling (no cross-condition pairing); at $\lambda = 0$ it reduces to standard OT. C²OT achieves a balance between these two extremes.

3. **Extension to Continuous Conditions**:

    - Function: Generalizes C²OT to continuous condition settings (e.g., text embeddings, temporal information).
    - Mechanism: For continuous conditions, the indicator function is replaced by the L2 distance between condition embeddings as $d(c_i, c_j)$. The authors also discuss normalization strategies for high-dimensional conditions — scaling condition distances to the same magnitude as spatial distances — to stabilize $\lambda$ tuning.
    - Design Motivation: Many practical applications involve continuous conditions (e.g., CLIP embeddings in text-guided image generation), necessitating a smooth generalization from the discrete case.

### Loss & Training

The training loss is identical to standard flow matching: $L = \mathbb{E}_{t, (x_1, x_0) \sim \pi^{C^2OT}} \|v_\theta(t, x_t, c) - (x_1 - x_0)\|^2$, where $\pi^{C^2OT}$ is the coupling computed by C²OT, $x_t = (1-t)x_0 + tx_1$ is the linear interpolation, and $v_\theta$ is the velocity field network. The only modification lies in the computation of the coupling $\pi$.

## Key Experimental Results

### Main Results

FID comparison across different NFE (Number of Function Evaluations):

| Method | CIFAR-10 (NFE=1) | CIFAR-10 (NFE=5) | ImageNet-32 (NFE=5) | ImageNet-256 (NFE=50) |
|--------|-----------------|-----------------|-------------------|---------------------|
| Independent Coupling (IC) | 18.72 | 5.21 | 12.34 | 4.85 |
| Standard OT | 22.15 | 6.03 | 14.67 | 5.42 |
| C²OT (Ours) | **15.83** | **4.52** | **10.89** | **4.21** |

W2 distance on the 8-gaussians-to-moons toy experiment (lower is better):

| NFE | Indep. Coupling | Standard OT | C²OT |
|-----|----------------|-------------|------|
| 1 | 0.452 | 0.891 | **0.183** |
| 5 | 0.089 | 0.234 | **0.041** |
| 20 | 0.012 | 0.078 | **0.008** |

### Ablation Study

| Configuration | CIFAR-10 FID (NFE=5) | Notes |
|---------------|---------------------|-------|
| C²OT ($\lambda=1.0$) | **4.52** | Optimal $\lambda$ |
| C²OT ($\lambda=0.1$) | 4.98 | Condition constraint too weak, approaches standard OT |
| C²OT ($\lambda=10$) | 4.71 | Condition constraint too strong, approaches IC |
| C²OT ($\lambda=100$) | 5.15 | Essentially equivalent to IC |
| Standard OT ($\lambda=0$) | 6.03 | No condition constraint |
| Independent Coupling ($\lambda=\infty$) | 5.21 | Fully condition-aligned pairing |

### Key Findings

- Standard OT in conditional generation is indeed worse than independent coupling, confirming the existence of the "curse of conditions": FID degradation of 0.82 on CIFAR-10 and 2.33 on ImageNet-32.
- C²OT achieves significant FID improvement at minimal cost (modifying a few lines of cost matrix computation), with the largest gains at low NFE — FID drops from 18.72 to 15.83 at NFE=1.
- The optimal $\lambda$ lies approximately in the range 0.5–2.0; values too large or too small are both suboptimal, indicating that a balance between OT path-straightening and condition consistency is required.
- C²OT remains effective under continuous condition settings, demonstrating the generality of the approach.
- The more condition classes there are, the more severe the skew induced by standard OT, and the greater the improvement from C²OT.

## Highlights & Insights

- **Exceptionally clear problem analysis**: The paper proceeds layer by layer — from toy experiments to theoretical analysis to large-scale experiments — to explain why conditional OT underperforms. This diagnosis-before-treatment research paradigm is exemplary.
- **Remarkably simple fix**: Adding a condition penalty term to the OT cost matrix requires only a few lines of code, ensuring low implementation overhead, good reproducibility, and ease of community adoption.
- **Fills a theoretical gap**: The community had long observed that conditional OT underperforms its unconditional counterpart, but lacked a theoretical explanation. This paper provides a clear diagnosis and solution, making an important theoretical contribution to the flow matching literature.

## Limitations & Future Work

- $\lambda$ requires manual tuning and the optimal value may vary across datasets. Although the effective range is narrow (0.5–2.0), an adaptive selection strategy would be preferable.
- The method has been validated only for class-conditional and simple continuous conditions; its effectiveness for more complex conditions (e.g., CLIP embeddings from natural language descriptions) remains to be verified.
- At very high NFE (e.g., 100+), the advantage of C²OT over standard OT and IC diminishes, suggesting its primary benefit lies in fast sampling scenarios with low NFE.
- The computational cost of OT itself is a consideration — the cubic complexity of OT solvers may become a bottleneck at large batch sizes.

## Related Work & Insights

- **vs. Standard OT-FM (Tong et al., 2024)**: Standard OT flow matching is highly effective in the unconditional setting but suffers from the "curse of conditions" in conditional generation. C²OT is a direct fix for OT-FM in conditional scenarios.
- **vs. Independent Coupling (IC)**: IC forgoes OT pairing entirely, resulting in less straight paths but no condition skew. C²OT outperforms IC by retaining OT's path-straightening effect while eliminating the conditionally skewed prior.
- **vs. Multi-sample Flow Matching**: Other works have explored multi-sample strategies to improve OT coupling, but none analyze the problem from the perspective of conditional prior skew. C²OT directly addresses the root cause.

## Rating

- Novelty: ⭐⭐⭐⭐ — Problem analysis is novel and insightful; the solution, while simple, is precisely targeted.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive coverage from toy experiments to ImageNet-256, with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ — The logical chain is clear and coherent from problem analysis to solution; a textbook example of research paper writing.
- Value: ⭐⭐⭐⭐ — Significant theoretical and practical value for the flow matching community, though the scope is limited to conditional generation with OT coupling.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Counterfactual Identifiability via Dynamic Optimal Transport](../../NeurIPS2025/image_generation/counterfactual_identifiability_via_dynamic_optimal_transport.md)
- [\[NeurIPS 2025\] On the Relation between Rectified Flows and Optimal Transport](../../NeurIPS2025/image_generation/on_the_relation_between_rectified_flows_and_optimal_transport.md)
- [\[CVPR 2026\] COT-FM: Cluster-wise Optimal Transport Flow Matching](../../CVPR2026/image_generation/cot-fm_cluster-wise_optimal_transport_flow_matching.md)
- [\[ICCV 2025\] Contrastive Flow Matching (ΔFM)](contrastive_flow_matching.md)
- [\[NeurIPS 2025\] Improving Posterior Inference of Galaxy Properties with Image-Based Conditional Flow Matching](../../NeurIPS2025/image_generation/improving_posterior_inference_of_galaxy_properties_with_image-based_conditional_.md)

<!-- RELATED:END -->
