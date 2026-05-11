---
title: >-
  [Paper Note] 2ndMatch: Finetuning Pruned Diffusion Models via Second-Order Jacobian Matching
description: >-
  [CVPR 2026][Image Generation][Diffusion Model] This paper proposes 2ndMatch, a fine-tuning framework for pruned diffusion models that aligns the second-order Jacobian matrix $J^\top J$ between the pruned and original mod…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Diffusion Model"
  - "Pruning"
  - "Jacobian Matching"
  - "Finite-Time Lyapunov Exponent"
  - "Knowledge Distillation"
date: 2025-06-06
content_hash: 8082ab6e4ad8eb46
---

# 2ndMatch: Finetuning Pruned Diffusion Models via Second-Order Jacobian Matching

**Conference**: CVPR 2026  
**arXiv**: [2506.05398](https://arxiv.org/abs/2506.05398)  
**Code**: No  
**Area**: Diffusion Model / Model Compression  
**Keywords**: Diffusion Model, Pruning, Jacobian Matching, Finite-Time Lyapunov Exponent, Knowledge Distillation

## TL;DR
This paper proposes 2ndMatch, a fine-tuning framework for pruned diffusion models that aligns the second-order Jacobian matrix $J^\top J$ between the pruned and original models—inspired by finite-time Lyapunov exponents (FTLE)—to match their sensitivity to input perturbations over time, thereby significantly closing the generation quality gap.

## Background & Motivation

**Background**: Diffusion models achieve excellent image generation quality but require hundreds of denoising steps at inference, incurring substantial computational costs. Model pruning is an effective strategy for reducing per-step computation.

**Limitations of Prior Work**: Post-pruning fine-tuning typically reuses the original denoising score matching (DSM) objective, which is insufficient for capacity-reduced pruned models. Existing knowledge distillation approaches align outputs or intermediate features but overlook the model's **sensitivity**—i.e., how the score function responds to input perturbations. First-order Jacobian matching is essentially equivalent to KD for diffusion models (since inputs already contain noise perturbations) and cannot capture perturbation propagation across time steps.

**Key Challenge**: Pruning reduces model capacity, causing the pruned model's sensitivity to perturbations to diverge from the original, leading to denoising trajectory drift and degraded generation quality. A method is needed to constrain the pruned model to maintain the same temporal dynamics as the original.

**Key Insight**: The paper views diffusion models as discrete-time dynamical systems and draws on FTLE theory, which quantifies the amplification/contraction rate of small perturbations over finite time horizons.

**Core Idea**: Align the $J^\top J$ (second-order Jacobian metric) between pruned and original models, using random projections $v^\top J^\top J v$ to efficiently estimate directional expansion rates, enabling scalable second-order Jacobian matching.

## Method

### Overall Architecture
The hybrid fine-tuning objective is: $\mathcal{L}_{total} = \lambda_{NP}\mathcal{L}_{NP} + \lambda_{KD}\mathcal{L}_{KD} + \lambda_{Jac}\mathcal{L}_{2nd\text{-}Jac}$, where the three complementary components handle noise prediction, output alignment, and temporal sensitivity matching respectively.

### Key Designs

1. **Noise Prediction**:

    - Function: Standard DDPM objective that predicts the noise added during the forward process
    - Mechanism: $\mathcal{L}_{NP} = \mathbb{E}_{\tilde{x},t,\epsilon}[\|s(\tilde{x},t;\theta) - \epsilon\|_2^2]$
    - Design Motivation: Serves as the basic supervisory signal, but is insufficient alone for capacity-reduced pruned models

2. **Knowledge Distillation**:

    - Function: Aligns the outputs of the pruned and original models
    - Mechanism: $\mathcal{L}_{KD} = \mathbb{E}_{\tilde{x},t}[\|s(\tilde{x},t;\theta) - s_\mathcal{D}(\tilde{x},t;\theta_\mathcal{D})\|_2^2]$
    - Design Motivation: Provides smoother supervision targets than raw noise, accelerating convergence

3. **Second-Order Jacobian Matching (Core Innovation)**:

    - Function: Aligns the local sensitivity of pruned and original models
    - Mechanism: FTLE theory indicates that perturbation amplification is governed by $\|v_1\| \approx \sqrt{v_0^\top J^\top J v_0}$. Since computing the full Jacobian is intractable, random projections $v \sim \mathcal{N}(0,I)$ are used to estimate directional expansion rates:
    $\mathcal{L}_{2nd\text{-}Jac} = \mathbb{E}_{\tilde{x},t,v}\left[(\|J\hat{v}\|_2^2 - \|J_\mathcal{D}\hat{v}\|_2^2)^2\right]$
      where $\hat{v} = v/\|v\|$, and $J\hat{v}$ is efficiently computed via Jacobian-vector products (JVP) without forming the full Jacobian matrix
    - Design Motivation: A Taylor expansion proof shows that first-order Jacobian matching under noisy inputs is equivalent to KD, yielding no additional benefit. Second-order matching captures perturbation propagation across time steps, better aligning dynamical system stability

### Why Does First-Order Jacobian Matching Fail?
The paper provides a Taylor expansion proof: $\|s(x') - s_\mathcal{D}(x')\|_2^2 = \|s(x) - s_\mathcal{D}(x)\|_2^2 + \sigma^2\|J - J_\mathcal{D}\|_F^2 + \mathcal{O}(\sigma^4)$. Under noisy inputs, output alignment implicitly subsumes first-order Jacobian matching, so explicitly adding it only increases computational overhead.

### Loss & Training
- Architecture-agnostic: applicable to both U-Net and Transformer-based diffusion architectures
- Pruning-method-agnostic: compatible with Diff-Pruning, BK-SDM, and other pruning methods
- Uses PyTorch's JVP functionality to efficiently compute $J\hat{v}$

## Key Experimental Results

### Main Results (LSUN + ImageNet 256×256, U-Net models)

| Dataset | Method | Params | MACs | FID↓ | rFID↓ |
|---------|--------|--------|------|------|-------|
| LSUN-Church | DDPM (Original) | 113.7M | 248.7G | 10.58 | - |
| | Diff-Pruning | 63.2M | 138.8G | 13.90 | 4.09 |
| | **2ndM (Ours)** | 63.2M | 138.8G | **11.25** | **2.08** |
| LSUN-Bedroom | DDPM (Original) | 113.7M | 248.7G | 6.62 | - |
| | Diff-Pruning | 63.2M | 138.8G | 17.90 | 7.62 |
| | **2ndM (Ours)** | 63.2M | 138.8G | **9.68** | **2.16** |
| ImageNet | LDM-4 (Original) | 400.9M | 99.8G | 3.60 | - |
| | Diff-Pruning | 175.8M | 43.2G | 10.23 | 9.28 |
| | **2ndM (Ours)** | 175.8M | 43.2G | **5.68** | **4.11** |

Stable Diffusion (COCO 512×512): Base+2ndM reduces FID from 15.76 to 13.84; Small+2ndM from 16.98 to 16.17.

### Ablation Study (CIFAR-10)

| Config | FID↓ | FTLE |
|--------|------|------|
| NP only | 5.29 | 0.413 |
| NP + KD | 5.05 | 0.418 |
| NP + KD + 1st JM | 5.14 | - |
| NP + KD + 2ndM (Ours) | **4.58** | - |
| Dense (Original) | 4.19 | - |

### Key Findings
- **First-order Jacobian matching is ineffective**: Adding first-order JM actually worsens FID from 5.05 to 5.14, validating the theoretical analysis
- **Second-order matching is critical**: 2ndM reduces FID from 5.05 to 4.58, with FTLE values closer to the original model, confirming the effectiveness of temporal sensitivity alignment
- 46% FID improvement on LSUN-Bedroom (17.90→9.68) and 55% rFID improvement on ImageNet
- Also effective on Transformer architectures: U-ViT on CIFAR-10 FID drops from 4.63 to 4.05

## Highlights & Insights
- **Dynamical systems perspective**: Reformulating diffusion model fine-tuning as a dynamical system stability problem and using FTLE theory to guide loss design provides deep insights into diffusion model training and generation
- **Elegant Taylor expansion proof**: Rigorously demonstrates the redundancy of first-order Jacobian matching in diffusion models, offering theoretical guidance for loss design in model compression
- **Practical random projections**: Estimating $v^\top J^\top J v$ via random directions circumvents the high-dimensional Jacobian computation bottleneck, scaling the method to large models (Stable Diffusion with 1.04B parameters)

## Limitations & Future Work
- Currently uses step-wise matching to approximate multi-step Jacobian propagation, limiting the ability to capture long-range temporal dependencies
- The trade-off between random projection efficiency and estimation accuracy is not thoroughly explored
- Only validated on image generation; applications to video/3D and other complex diffusion model tasks remain unexplored
- The FTLE concept could be extended to distillation (non-pruning) settings or used to guide sampler schedule design

## Related Work & Insights
- **vs Diff-Pruning**: Diff-Pruning only uses DSM for post-pruning fine-tuning; 2ndM adds sensitivity alignment on top, achieving significantly better FID at the same parameter count
- **vs DeepCache**: DeepCache accelerates inference by caching intermediate features without reducing parameters; it is complementary to pruning approaches
- **vs BK-SDM**: BK-SDM is a pruning method designed for Stable Diffusion; 2ndM can be directly stacked on top to further improve quality

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introducing FTLE theory into model compression; the second-order Jacobian matching formulation is elegant and theoretically grounded
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers U-Net and Transformer architectures, 5 datasets, multiple pruning methods, and thorough ablations
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations, clear motivation, and systematic experimental design
- Value: ⭐⭐⭐⭐ A general fine-tuning framework, though limited to the model pruning scenario

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] LeapAlign: Post-Training Flow Matching Models at Any Generation Step by Building Two-Step Trajectories](leapalign_post_training_flow_matching_models_at_any_generation_step.md)
- [\[ICLR 2026\] HOG-Diff: Higher-Order Guided Diffusion for Graph Generation](../../ICLR2026/image_generation/hog-diff_higher-order_guided_diffusion_for_graph_generation.md)
- [\[NeurIPS 2025\] High-order Equivariant Flow Matching for Density Functional Theory Hamiltonian Prediction](../../NeurIPS2025/image_generation/high-order_equivariant_flow_matching_for_density_functional_theory_hamiltonian_p.md)
- [\[NeurIPS 2025\] Fast Solvers for Discrete Diffusion Models: Theory and Applications of High-Order Algorithms](../../NeurIPS2025/image_generation/fast_solvers_for_discrete_diffusion_models_theory_and_applications_of_high-order.md)
- [\[CVPR 2026\] VeCoR — Velocity Contrastive Regularization for Flow Matching](vecor_--_velocity_contrastive_regularization_for_flow_matching.md)

</div>

<!-- RELATED:END -->
