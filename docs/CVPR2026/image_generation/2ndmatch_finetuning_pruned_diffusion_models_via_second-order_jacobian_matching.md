---
title: >-
  [Paper Note] 2ndMatch: Finetuning Pruned Diffusion Models via Second-Order Jacobian Matching
description: >-
  [CVPR 2026][Image Generation][Diffusion Models] A fine-tuning framework named 2ndMatch is proposed. By aligning the second-order Jacobian matrix $J^\top J$ (inspired by Finite-Time Lyapunov Exponents) of the pruned model with the original model, it matches the temporal sensitivity to input perturbations, significantly narrowing the generation quality gap.
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Diffusion Models"
  - "Model Pruning"
  - "Jacobian Matching"
  - "Finite-Time Lyapunov Exponent"
  - "Knowledge Distillation"
date: 2026-05-08
content_hash: a38687c58c34c772
---

# 2ndMatch: Finetuning Pruned Diffusion Models via Second-Order Jacobian Matching

**Conference**: CVPR 2026  
**arXiv**: [2506.05398](https://arxiv.org/abs/2506.05398)  
**Code**: None  
**Area**: Diffusion Models / Model Compression  
**Keywords**: Diffusion Models, Model Pruning, Jacobian Matching, Finite-Time Lyapunov Exponent, Knowledge Distillation

## TL;DR
A fine-tuning framework named 2ndMatch is proposed. By aligning the second-order Jacobian matrix $J^\top J$ (inspired by Finite-Time Lyapunov Exponents) of the pruned model with the original model, it matches the temporal sensitivity to input perturbations, significantly narrowing the generation quality gap.

## Background & Motivation

**Background**: Diffusion models achieve excellent results in image generation but suffer from high computational costs during inference due to hundreds of denoising steps. Model pruning is an effective strategy to reduce the computation per step.

**Limitations of Prior Work**: Fine-tuning after pruning typically reuses the original Denoising Score Matching (DSM) objective, which is insufficient for reduced-capacity models. Existing knowledge distillation (KD) aligns outputs or intermediate features but overlooks model **sensitivity**—specifically, how the score function responds to input perturbations. First-order Jacobian matching is largely redundant for diffusion models (as inputs already contain noise) and fails to capture perturbation propagation across time.

**Key Challenge**: Reduced model capacity $\to$ sensitivity to perturbations deviates from the original model $\to$ denoising trajectory drift $\to$ degradation in generation quality. There is a need for a method to constrain pruned models to maintain the same temporal dynamical behavior.

**Key Insight**: Diffusion models are viewed as discrete-time dynamical systems. Drawing from Finite-Time Lyapunov Exponent (FTLE) theory, the framework quantifies the amplification/contraction rate of infinitesimal perturbations over finite time.

**Core Idea**: Align the $J^\top J$ (second-order Jacobian metric) of the pruned model with the original model. Directional expansion rates are efficiently estimated via random projection $v^\top J^\top J v$, enabling scalable second-order Jacobian matching.

## Method

### Overall Architecture
The 2ndMatch framework addresses the problem that standard denoising objectives cannot recover generation quality for pruned models with reduced capacity. The fine-tuning process is viewed as aligning with the original (dense) model at three levels: prediction accuracy, output similarity, and consistency in "reaction" to input perturbations. The first two are standard, while the third is the novelty: keeping the amplification behavior of perturbations at each step consistent with the original model. These comprise a hybrid objective:

$$\mathcal{L}_{total} = \lambda_{NP}\mathcal{L}_{NP} + \lambda_{KD}\mathcal{L}_{KD} + \lambda_{Jac}\mathcal{L}_{2nd\text{-}Jac}$$

### Key Designs

**1. Noise Prediction: Foundation supervision for pruned models**

This is the standard DDPM objective, where the model predicts the noise $\epsilon$ added during the forward process: $\mathcal{L}_{NP} = \mathbb{E}_{\tilde{x},t,\epsilon}[\|s(\tilde{x},t;\theta) - \epsilon\|_2^2]$. While essential for any diffusion training, fitting noise alone is insufficient for models with nearly half the parameters removed, leading to slow and biased convergence. It serves as the "base."

**2. Knowledge Distillation: Using original model outputs as a smoother teacher**

This aligns the score outputs of the pruned and original models on the same input: $\mathcal{L}_{KD} = \mathbb{E}_{\tilde{x},t}[\|s(\tilde{x},t;\theta) - s_\mathcal{D}(\tilde{x},t;\theta_\mathcal{D})\|_2^2]$. Compared to the stochastic noise $\epsilon$, the teacher's score is a smoother and more informative target, accelerating convergence. However, it only manages "output values" and ignores dynamic responses to perturbations.

**3. Second-Order Jacobian Matching: Aligning temporal sensitivity (Core Innovation)**

Denoising is a multi-step iterative dynamical system. Small perturbations at one step can be amplified or contracted along subsequent steps. If the pruned model deviates in this amplification rate, the trajectory drifts, causing quality collapse. The authors utilize the FTLE to characterize this—FTLE quantifies expansion over finite time, while the local expansion of a single step is determined by the second-order Jacobian metric $J^\top J$: $\|v_1\| \approx \sqrt{v_0^\top J^\top J v_0}$.

High-dimensional Jacobian construction is bypassed via random projection: sample a random direction $v\sim\mathcal{N}(0,I)$, normalize to $\hat{v}=v/\|v\|$, and compare directional expansion rates using Jacobian-Vector Products (JVP) $J\hat{v}$ without explicit Jacobian formation:

$$\mathcal{L}_{2nd\text{-}Jac} = \mathbb{E}_{\tilde{x},t,v}\left[(\|J\hat{v}\|_2^2 - \|J_\mathcal{D}\hat{v}\|_2^2)^2\right]$$

Second-order matching is chosen over first-order because the latter is redundant in diffusion. Taylor expansion of noisy inputs gives $\|s(x') - s_\mathcal{D}(x')\|_2^2 = \|s(x) - s_\mathcal{D}(x)\|_2^2 + \sigma^2\|J - J_\mathcal{D}\|_F^2 + \mathcal{O}(\sigma^4)$. Since the input already contains noise $\sigma$, output alignment (KD term) implicitly includes first-order Jacobian matching. The second-order term captures propagation across timesteps, corresponding to dynamical stability.

### Loss & Training
The total objective is a weighted sum. The method is architecture-agnostic, applicable to both U-Net and Transformer backbones, and can be integrated with various pruning methods like Diff-Pruning or BK-SDM. PyTorch JVPs are used for efficient computation of $J\hat{v}$, avoiding memory explosions from explicit Jacobians.

## Key Experimental Results

### Main Results (LSUN + ImageNet 256×256, U-Net models)

| Dataset | Method | Parameters | MACs | FID↓ | rFID↓ |
|--------|------|--------|------|------|-------|
| LSUN-Church | DDPM (Original) | 113.7M | 248.7G | 10.58 | - |
| | Diff-Pruning | 63.2M | 138.8G | 13.90 | 4.09 |
| | **2ndM (Ours)** | 63.2M | 138.8G | **11.25** | **2.08** |
| LSUN-Bedroom | DDPM (Original) | 113.7M | 248.7G | 6.62 | - |
| | Diff-Pruning | 63.2M | 138.8G | 17.90 | 7.62 |
| | **2ndM (Ours)** | 63.2M | 138.8G | **9.68** | **2.16** |
| ImageNet | LDM-4 (Original) | 400.9M | 99.8G | 3.60 | - |
| | Diff-Pruning | 175.8M | 43.2G | 10.23 | 9.28 |
| | **2ndM (Ours)** | 175.8M | 43.2G | **5.68** | **4.11** |

Stable Diffusion (COCO 512×512): Base+2ndM reduced FID from 15.76 to 13.84; Small+2ndM from 16.98 to 16.17.

### Ablation Study (CIFAR-10)

| Config | FID↓ | FTLE |
|------|------|------|
| NP only | 5.29 | 0.413 |
| NP + KD | 5.05 | 0.418 |
| NP + KD + 1st JM | 5.14 | - |
| NP + KD + 2ndM (Ours) | **4.58** | - |
| Dense (Original) | 4.19 | - |

### Key Findings
- **Ineffectiveness of 1st-order matching**: Adding 1st JM increased FID from 5.05 to 5.14, confirming the theoretical analysis.
- **Criticality of 2nd-order matching**: 2ndM significantly reduced FID to 4.58. The FTLE became closer to the original model, proving the effectiveness of temporal sensitivity alignment.
- **Gain**: FID improved by 46% on LSUN-Bedroom (17.90 to 9.68) and rFID improved by 55% on ImageNet.
- **Transformer applicability**: Validated on U-ViT with FID dropping from 4.63 to 4.05 on CIFAR-10.

## Highlights & Insights
- **Dynamical Systems Perspective**: Re-frames the fine-tuning of diffusion models as a stability problem in dynamical systems, using FTLE theory to guide loss function design.
- **Theoretical Elegance**: Strictly proves the redundancy of first-order Jacobian matching in diffusion through Taylor expansion.
- **Practicality**: Random projections $v^\top J^\top J v$ bypass the high-dimensional Jacobian bottleneck, making the method scalable to large models like Stable Diffusion.

## Limitations & Future Work
- Current step-wise matching approximates multi-step Jacobian propagation, which might limit the capture of long-range temporal dependencies.
- The trade-off between random projection efficiency and estimation accuracy requires further investigation.
- Evaluations are focused on image generation; applications to video or 3D diffusion models remain to be explored.
- FTLE concepts could be extended to distillation (non-pruning) or used to design sampling schedulers.

## Related Work & Insights
- **vs Diff-Pruning**: Diff-Pruning uses only DSM; 2ndM adds sensitivity alignment, yielding significantly better FID at the same parameter count.
- **vs DeepCache**: DeepCache accelerates via caching intermediate features without reducing parameters; it is complementary to 2ndM.
- **vs BK-SDM**: A pruning method for Stable Diffusion; 2ndM can be applied on top to enhance performance.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introduction of FTLE theory to model compression is elegant and theoretically grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers U-Net/Transformer, 5 datasets, multiple pruning methods, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous derivations, clear motivation, and systematic experimental design.
- Value: ⭐⭐⭐⭐ A versatile fine-tuning framework, though focused on model pruning scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models](../../CVPR2025/image_generation/efficient_fine-tuning_and_concept_suppression_for_pruned_diffusion_models.md)
- [\[CVPR 2026\] When Local Rules Create Global Order: Self-Organized Representation Learning for Latent Diffusion Models](when_local_rules_create_global_order_self-organized_representation_learning_for_.md)
- [\[ICML 2026\] Esoteric Language Models: A Family of Any-Order Diffusion LLMs](../../ICML2026/image_generation/esoteric_language_models_a_family_of_any-order_diffusion_llms.md)
- [\[ICLR 2026\] HOG-Diff: Higher-Order Guided Diffusion for Graph Generation](../../ICLR2026/image_generation/hog-diff_higher-order_guided_diffusion_for_graph_generation.md)
- [\[CVPR 2026\] Reviving ConvNeXt for Efficient Convolutional Diffusion Models](reviving_convnext_for_efficient_convolutional_diffusion_models.md)

</div>

<!-- RELATED:END -->
