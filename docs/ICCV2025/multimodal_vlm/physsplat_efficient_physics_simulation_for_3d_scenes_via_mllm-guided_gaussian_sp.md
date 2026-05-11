---
title: >-
  [Paper Note] PhysSplat: Efficient Physics Simulation for 3D Scenes via MLLM-Guided Gaussian Splatting
description: >-
  [ICCV 2025][Multimodal VLM][Physics Simulation] This paper proposes PhysSplat, the first approach to leverage multimodal large language models (MLLMs) for zero-shot estimation of physical properties of objects in 3D scen…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Physics Simulation"
  - "3D Gaussian Splatting"
  - "MLLM"
  - "Physical Property Estimation"
  - "MPM"
date: 2026-05-08
content_hash: b2738ca9a9d9217d
---

# PhysSplat: Efficient Physics Simulation for 3D Scenes via MLLM-Guided Gaussian Splatting

**Conference**: ICCV 2025
**arXiv**: [2411.12789](https://arxiv.org/abs/2411.12789)
**Code**: [Project Page](https://github.com/PhysSplat)
**Area**: Multimodal VLM
**Keywords**: Physics Simulation, 3D Gaussian Splatting, MLLM, Physical Property Estimation, MPM

## TL;DR

This paper proposes PhysSplat, the first approach to leverage multimodal large language models (MLLMs) for zero-shot estimation of physical properties of objects in 3D scenes. Combined with a physics-geometry adaptive sampling strategy, it achieves realistic physics simulation on a single GPU within 2 minutes.

## Background & Motivation

Endowing static 3D objects with interactive dynamics remains challenging:

**Manual parameter specification**: Methods such as PhysGaussian require manual assignment of physical properties.

**High cost of video diffusion models**: Methods such as PhysDreamer estimate Young's modulus via video generative models, incurring substantial computational cost (1.5+ hours) and limited controllability.

**Restricted to non-rigid objects**: Video diffusion-based methods cannot handle rigid objects (e.g., tables and chairs).

**Core Idea**: Humans are adept at inferring physical properties from visual information. PhysSplat therefore employs MLLMs to emulate human visual-physical reasoning.

## Method

### MLLM-based Physical Property Perception (MLLM-P3)

Pipeline for zero-shot physical property estimation:
1. Render images of objects in the 3D scene.
2. Generate textual descriptions using a VQA model (BLIP).
3. Feed the image and description into an MLLM (GPT-4V) to obtain $K$ candidate materials.
4. Select the best-matching material via CLIP similarity.
5. The MLLM returns physical properties $M = \{\rho, E, \nu\}$ (density, Young's modulus, Poisson's ratio).

### Material Property Distribution Prediction (MPDP)

Even for objects of a single material, physical properties exhibit intrinsic spatial variation. The problem is reformulated from regression to probabilistic distribution estimation:

$$\mathcal{P} = \mathcal{D}_\theta(\mathcal{X})$$

The network takes the object point cloud and the MLLM-predicted mean as input, predicting a geometry-aware physical property distribution, which is then multiplied by the global mean to yield the final per-point properties.

### Physical-Geometric Adaptive Sampling (PGAS)

The sampling radius is adaptively adjusted as follows:

$$K = \frac{\lambda_3}{\lambda_1 + \lambda_2 + \lambda_3}$$

$$\hat{r} = \min(r, k\sqrt{\frac{E}{\hat{K}}} r)$$

Softer objects (small $E$) and high-curvature regions are assigned smaller radii and more driving particles.

### MPM Simulation

Based on the MLS-MPM simulator, the time-dependent state of each Gaussian kernel is:
$$x_i(t) = \Delta(x_i, t), \quad \Sigma_i(t) = F_i(t)\Sigma_i F_i(t)^T$$

Only the driving particles sampled by PGAS are simulated; the remaining Gaussians are derived via local rigid-body transformation fitting.

## Key Experimental Results

### Quantitative Comparison

| Method | RS↑ | AS↑ | Time |
|--------|-----|-----|------|
| PhysGaussian | 4.50 | 7.56 | - |
| PhysDreamer | 4.54 | 7.71 | - |
| Physics3D | 4.62 | 7.83 | 1.5h |
| DreamGaussian4D | 4.57 | 7.28 | 0.1h |
| **PhysSplat** | **4.66** | **7.89** | **2min** |

PhysSplat achieves the highest realism scores with an inference time of only 2 minutes, making it 45× faster than Physics3D.

### Synthetic Dataset

| Method | RS↑ | AS↑ | Time |
|--------|-----|-----|------|
| Physics3D | 5.10 | 8.01 | 1.5h |
| DreamPhysics | 5.05 | 7.92 | 1.5h |
| **PhysSplat** | **5.10** | **8.20** | **2min** |

PhysSplat achieves best or comparable performance on the synthetic dataset as well.

## Highlights & Insights

1. **First use of MLLMs for zero-shot physical property estimation**: Entirely bypasses costly video diffusion optimization.
2. **Elegant problem reformulation**: Shifting from regression to distribution estimation, MPDP requires only ~2% of the computation of Physics3D.
3. **Full-scene simulation**: The only method that supports simulation of entire scenes (Table 1).
4. **Well-motivated PGAS design**: Adaptive sampling balances fine detail for soft objects with overall computational efficiency.

## Limitations & Future Work

- MLLM-based physical property estimation may be subject to hallucinations.
- MPDP relies on pseudo-labels from Physics3D for training.
- Estimation accuracy is limited for objects composed of complex material combinations.
- Extreme physical phenomena such as hard collision and fracture are not modeled.

## Related Work & Insights

- PhysGaussian: Injecting physical properties into 3DGS.
- PhysDreamer, Physics3D: Learning physical parameters via video diffusion.
- MPM: Material Point Method simulator.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (A new paradigm for physical property estimation via MLLMs)
- Technical Depth: ⭐⭐⭐⭐ (Complete pipeline of MLLM-P3 + MPDP + PGAS)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multiple datasets + user study)
- Value: ⭐⭐⭐⭐⭐ (2-minute inference is highly practical)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation](perspective-aware_reasoning_in_vision-language_models_via_mental_imagery_simulat.md)
- [\[ICCV 2025\] Adaptive Prompt Learning via Gaussian Outlier Synthesis for Out-of-Distribution Detection](adaptive_prompt_learning_via_gaussian_outlier_synthesis_for_out-of-distribution_.md)
- [\[ICCV 2025\] Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks](safeguarding_vision-language_models_mitigating_vulnerabilities_to_gaussian_noise.md)
- [\[ICCV 2025\] Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models](physics_context_builders_a_modular_framework_for_physical_reasoning_in_vision-la.md)
- [\[AAAI 2026\] UniFit: Towards Universal Virtual Try-on with MLLM-Guided Semantic Alignment](../../AAAI2026/multimodal_vlm/unifit_towards_universal_virtual_try-on_with_mllm-guided_semantic_alignment.md)

</div>

<!-- RELATED:END -->
