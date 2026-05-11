---
title: >-
  [Paper Note] Image Diffusion Preview with Consistency Solver
description: >-
  [CVPR 2026][Image Generation][Diffusion model acceleration] This paper proposes the Diffusion Preview paradigm and ConsistencySolver—a lightweight high-order ODE solver trained via reinforcement learning—that generates h…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Diffusion model acceleration"
  - "ODE solver"
  - "reinforcement learning"
  - "preview-and-refine"
  - "sampling efficiency"
date: 2026-05-08
content_hash: a48d498c42864867
---

# Image Diffusion Preview with Consistency Solver

**Conference**: CVPR 2026
**arXiv**: [2512.13592](https://arxiv.org/abs/2512.13592)
**Code**: [https://github.com/G-U-N/consolver](https://github.com/G-U-N/consolver)
**Area**: Diffusion Models / Image Generation
**Keywords**: Diffusion model acceleration, ODE solver, reinforcement learning, preview-and-refine, sampling efficiency

## TL;DR

This paper proposes the Diffusion Preview paradigm and ConsistencySolver—a lightweight high-order ODE solver trained via reinforcement learning—that generates high-quality preview images with few-step sampling while ensuring consistency with full-step outputs. It achieves FID comparable to Multistep DPM-Solver using 47% fewer steps, reducing user interaction time by nearly 50%.

## Background & Motivation

**Background**: Diffusion models excel at high-fidelity image generation, but inference requires numerically solving reverse differential equations, incurring substantial computational cost. Existing acceleration methods fall into two categories: training-free ODE solvers (DDIM, DPM-Solver, UniPC, etc.) and post-training distillation methods (LCM, DMD2, etc.).

**Limitations of Prior Work**: Training-free solvers rely on theoretical assumptions and produce poor quality at low step counts; distillation methods require expensive retraining, disrupt the deterministic mapping of the PF-ODE (the correspondence from noise space to data space), and suffer from accumulated distillation errors that degrade generation quality. Critically, distilled models typically lose the flexibility of choosing inference step counts freely.

**Key Challenge**: In interactive generation workflows (e.g., design prototyping), users need to rapidly preview multiple variants to select a satisfying direction before refinement. Existing methods are either "fast but low-quality and inconsistent" (training-free solvers) or "high-quality but expensive and determinism-breaking" (distillation).

**Goal**: Design a Preview-and-Refine workflow satisfying three requirements: (1) high preview fidelity (close to the final output); (2) high preview efficiency (low step count); (3) consistency between preview and final output (visually coherent results from the same random seed).

**Key Insight**: Rather than modifying the diffusion model itself, this work optimizes the ODE solver. The integration coefficients of the solver are treated as a learnable policy, and reinforcement learning is used to search for the optimal integration strategy.

**Core Idea**: Parameterize ODE solver coefficients as a lightweight MLP and optimize it via PPO reinforcement learning, so that few-step sampling maximizes perceptual similarity to the full-step output.

## Method

### Overall Architecture

Given a text prompt and noise image, the diffusion model $\epsilon_\phi$ predicts the denoising direction. The learnable ODE solver $\Psi_\theta$ generates a preview image $\mathbf{x}_p$ in few steps, while the training-free solver $\Psi$ generates the target image $\mathbf{x}_{gt}$ with full steps. A similarity reward $\mathcal{R}$ is computed based on depth maps, segmentation masks, DINO features, etc., and used to update $\theta$ via PPO.

### Key Designs

1. **Parameterization of ConsistencySolver**:

    - **Function**: An adaptive high-order ODE solver that dynamically adjusts integration strategy based on the current timestep.
    - **Mechanism**: Derived from the general linear multistep method (LMM), each update step takes the form $\mathbf{y}_{t_{i+1}} = \mathbf{y}_{t_i} + (n_{t_{i+1}} - n_{t_i}) \cdot [\sum_{j=1}^{m} w_j(t_i, t_{i+1}) \cdot \epsilon_{i+1-j}]$, where $\mathbf{y}_t = \mathbf{x}_t / \alpha_t$. The key innovation is that the weights $w_j$ are not fixed theoretical values but are dynamically predicted by a lightweight MLP $\mathbf{f}_\theta(t_i, t_{i+1})$ conditioned on the current and target timesteps. This MLP takes only two scalar inputs and outputs $m$ weights.
    - **Design Motivation**: Classical solvers (DDIM as first-order, DPM-Solver-2 as midpoint approximation) can all be viewed as special cases of this framework with different fixed weight values. Making the weights learnable enables the solver to adapt to the actual sampling dynamics of the model rather than relying on theoretical assumptions.

2. **PPO-Based Reinforcement Learning Optimization**:

    - **Function**: Search for optimal solver coefficients to maximize consistency between preview and target.
    - **Mechanism**: An offline dataset $\{(c^{(k)}, z^{(k)}, x_{gt}^{(k)})\}$ is pre-generated and fixed for reuse. In each PPO episode, a batch of triplets is sampled and a $K$-step preview trajectory is rolled out. At each transition, the MLP outputs coefficients and probabilities. Upon completion, the similarity reward $\mathcal{R} = \text{Sim}(x_{gt}, x_p)$ is computed, and the policy is updated using the standard PPO clipped surrogate objective. Advantage estimation is computed via batch-wise self-normalization.
    - **Design Motivation**: RL offers three key advantages over distillation: (1) compatibility with non-differentiable rewards, eliminating the need for backpropagation through the diffusion trajectory; (2) better generalization; and (3) lower training overhead, as only the compact MLP participates in gradient computation.

3. **Multi-Dimensional Similarity Reward Design**:

    - **Function**: Measure preview-target consistency across multiple perceptual dimensions.
    - **Mechanism**: Depth maps are used as the default RL reward signal. Evaluation employs six dimensions: CLIP semantic alignment, DINO structural consistency, Inception perceptual similarity, SegFormer segmentation accuracy, pixel-level PSNR, and depth consistency.
    - **Design Motivation**: A single metric cannot comprehensively capture consistency; multi-dimensional evaluation ensures that previews are faithful to the final output in terms of semantics, structure, and geometry.

### Loss & Training

The PPO clipped surrogate objective is used with clipping parameter $\epsilon \in (0,1)$. Advantages are normalized using batch-wise mean and standard deviation. Only the lightweight MLP parameters (on the order of thousands) are updated during training, while the diffusion model is fully frozen. A solver trained on Stable Diffusion can be directly transferred to SD1.4, DreamShaper, and even SDXL without retraining.

## Key Experimental Results

### Main Results

**Stable Diffusion Text-to-Image Generation (COCO 2017)**:

| Method | Steps | FID↓ | CLIP↑ | DINO↑ | Depth↑ |
|------|------|------|-------|-------|--------|
| DDIM | 5 | 52.59 | 87.8 | 73.2 | 14.2 |
| Multistep DPM | 5 | 25.87 | 93.1 | 85.5 | 19.1 |
| UniPC | 5 | 23.15 | 93.2 | 85.5 | 18.7 |
| **ConsistencySolver** | **5** | **20.39** | **94.2** | **86.5** | **19.3** |
| Multistep DPM | 10 | 19.29 | 97.0 | 93.0 | 24.1 |
| **ConsistencySolver** | **8** | **18.82** | **96.4** | **91.2** | **22.2** |
| LCM (distillation) | 4 | 22.00 | 90.0 | 75.1 | 14.3 |
| DMD2 (distillation) | 1 | 19.88 | 89.3 | 73.8 | 12.6 |

**Cross-Model Generalization (Trained on SD1.5 → Direct Transfer)**:

| Target Model | Steps | Multistep DPM FID | ConsistencySolver FID |
|----------|------|-------------------|----------------------|
| SDXL | 10 | 26.32 | **23.32** |
| SD1.4 | 5 | 25.22 | **20.22** |

### Ablation Study

| Dimension | Configuration | FID↓ | DINO↑ |
|----------|------|------|-------|
| Training method | RL (PPO) | **20.39** | **86.5** |
| | Distillation (Ours-Distill) | 22.91 | 85.1 |
| | AMED (distillation) | 31.09 | 80.8 |
| Efficiency | ConsistencySolver 8 steps | 18.82 | 91.2 |
| | DPM-Solver 10 steps (comparable quality) | 19.29 | 93.0 |

### Key Findings

- ConsistencySolver at 5 steps (FID 20.39) already outperforms Multistep DPM-Solver at 5 steps (FID 25.87), a ~21% reduction.
- ConsistencySolver at 8 steps (FID 18.82) matches or surpasses Multistep DPM-Solver at 10 steps (FID 19.29), achieving a 47% step reduction (8 vs. ~15 steps for equivalent quality).
- RL training clearly outperforms distillation training (FID 20.39 vs. 22.91) with better generalization.
- A solver trained on SD1.5 transfers directly to SDXL, suggesting that different diffusion models share similar optimal sampling dynamics.
- User studies demonstrate a nearly 50% reduction in overall interaction time.

## Highlights & Insights

- **"Optimize the solver, not the model" paradigm**: The diffusion model weights are left entirely untouched; only a compact MLP with thousands of parameters is trained to serve as the solver. The investment is minimal yet the gains are substantial. This approach generalizes to any generative model requiring accelerated sampling.
- **Cross-model generalization finding**: A solver trained on SD1.5 remains effective when applied directly to SDXL, suggesting that optimal sampling strategies share common structure across different diffusion models—a theoretically valuable insight.
- **Practical value of the Preview-and-Refine workflow**: Dividing diffusion model usage into "rapid exploration" and "refinement" stages closely aligns with the real-world needs of designers.

## Limitations & Future Work

- Validation is currently limited to image generation and image editing; extension to video generation acceleration has not been explored.
- The choice of reward function (depth maps by default) may not be optimal for all tasks.
- The MLP takes only two scalar inputs $(t_i, t_{i+1})$ without conditioning on the current image state, which may limit its adaptive capacity.
- Combining ConsistencySolver with distillation methods is a promising direction for future work.

## Related Work & Insights

- **vs. DPM-Solver / UniPC (training-free solvers)**: These methods use fixed theoretical coefficients, whereas ConsistencySolver employs learned adaptive coefficients, yielding a clear advantage at low step counts.
- **vs. LCM / DMD2 (distillation methods)**: Distillation methods modify model weights, disrupting the PF-ODE mapping and requiring expensive training. ConsistencySolver leaves the model untouched, preserves the full deterministic mapping, and incurs minimal training cost.
- **vs. AMED (distillation-based solver)**: Both approaches train solver coefficients, but AMED uses trajectory distillation while ConsistencySolver uses RL, with the latter achieving better generalization.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Training ODE solvers with RL is a novel angle; the Preview-and-Refine paradigm is also practically compelling.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Validation on two models, cross-model transfer, multiple baselines, user studies, and detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretical derivations are clear, and the relationship to classical solvers is well articulated.
- **Value**: ⭐⭐⭐⭐ High practical value, extremely low training cost, and plug-and-play usability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dual-Solver: A Generalized ODE Solver for Diffusion Models with Dual Prediction](../../ICLR2026/image_generation/dual-solver_a_generalized_ode_solver_for_diffusion_models_with_dual_prediction.md)
- [\[CVPR 2026\] PSR: Scaling Multi-Subject Personalized Image Generation with Pairwise Subject-Consistency Rewards](psr_scaling_multi-subject_personalized_image_generation_with_pairwise_subject-co.md)
- [\[ICLR 2026\] LVTINO: LAtent Video consisTency INverse sOlver for High Definition Video Restoration](../../ICLR2026/image_generation/lvtino_latent_video_consistency_inverse_solver_for_high_definition_video_restora.md)
- [\[ICCV 2025\] LATINO-PRO: LAtent consisTency INverse sOlver with PRompt Optimization](../../ICCV2025/image_generation/latino-pro_latent_consistency_inverse_solver_with_prompt_optimization.md)
- [\[CVPR 2026\] Layer Consistency Matters: Elegant Latent Transition Discrepancy for Generalizable Synthetic Image Detection](layer_consistency_matters_elegant_latent_transition_discrepancy_for_generalizabl.md)

</div>

<!-- RELATED:END -->
