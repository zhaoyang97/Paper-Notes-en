---
title: >-
  [Paper Note] Pretrained Reversible Generation as Unsupervised Visual Representation Learning
description: >-
  [ICCV 2025][Image Generation][Reversible generation] PRG extracts unsupervised visual representations by **inverting the generation process of pretrained continuous generative models** (diffusion/flow models), enabling model-agnostic adaptation to discriminative tasks. It achieves 78% top-1 accuracy on ImageNet 64×64, establishing state of the art among generative model-based methods.
tags:
  - ICCV 2025
  - Image Generation
  - Reversible generation
  - flow matching
  - unsupervised representation learning
  - pretrain-finetune
  - mutual information
date: 2026-05-08
content_hash: 4bf88d2d07e649fc
---

# Pretrained Reversible Generation as Unsupervised Visual Representation Learning

**Conference**: ICCV 2025
**arXiv**: [2412.01787](https://arxiv.org/abs/2412.01787)
**Code**: [Project Page](https://opendilab.github.io/PRG/)
**Area**: Diffusion Models · Representation Learning
**Keywords**: Reversible generation, flow matching, unsupervised representation learning, pretrain-finetune, mutual information

## TL;DR

PRG extracts unsupervised visual representations by **inverting the generation process of pretrained continuous generative models** (diffusion/flow models), enabling model-agnostic adaptation to discriminative tasks. It achieves 78% top-1 accuracy on ImageNet 64×64, establishing state of the art among generative model-based methods.

## Background & Motivation

Diffusion/flow models have achieved remarkable success in generative tasks, yet their potential for **discriminative tasks** remains underexplored. Existing methods that leverage diffusion models for discrimination suffer from the following issues:

**Generative classifiers** ($p(y|x) = p(x|y)p(y)/p(x)$): computationally expensive, requiring inference over all classes.

**Intermediate feature extraction** (e.g., DDAE): relies on specific network modules (e.g., a particular UNet layer), leading to complex and non-generalizable designs.

**Large performance gap**: still significantly behind discriminative approaches.

Core insight: "What I cannot create, I do not understand" (Feynman) — a model capable of generating data must have internalized the structure of that data. **Inverting the generation process** thus serves naturally as feature extraction.

## Method

### 1. Pretraining Stage

Three variants of continuous-time flow models are trained:

- **PRG-GVP**: Generalized VP-SDE, $\alpha_t = \cos(\frac{\pi t}{2})$, $\sigma_t = \sin(\frac{\pi t}{2})$
- **PRG-ICFM**: Conditional flow matching, $v(x_t|x_0,x_1) = x_1 - x_0$
- **PRG-OTCFM**: Optimal transport conditional flow matching with joint sampling of $(x_0, x_1)$

Training objective (flow matching loss):

$$\mathcal{L}_{\text{FM}} = \frac{1}{2}\mathbb{E}_{p(x_t)}[\lambda_{\text{FM}}(t)\|v_\theta(x_t) - v(x_t)\|^2] dt$$

### 2. Reversed Generation as Feature Extraction

The generation process runs from $t \in [1, 0]$ (noise → data); its inversion runs from $t \in [0, 1]$ (data → features). The feature $x_t = F_\theta(x_0)$ can be extracted at any point along the trajectory.

### 3. Finetuning Stage

A classifier $p_\phi(y|z)$ is appended to the inverted trajectory, and both the flow model and classifier are jointly finetuned:

$$\mathcal{L}_{\text{total}} = -\sum_{i=1}^N \log p_\phi(y_i | F_\theta(x_i)) + \beta \mathcal{L}_{\text{FM}}(x)$$

**Classifier design**: a simple two-layer MLP with tanh suffices — the inverted features are already highly structured.

### 4. Theoretical Guarantee

Pretraining is equivalent to maximizing the mutual information $\mathcal{I}(X,Z)$ between data $X$ and representation $Z$:

$$\theta^* = \arg\max_\theta \mathcal{I}(X,Z) = \arg\max_\theta \mathbb{E}_{p(z,x)}[\log p(x|z)]$$

Flow matching training maximizes a lower bound on the likelihood (Eq. 8), thereby indirectly maximizing mutual information.

## Key Experimental Results

### Main Results: CIFAR-10 Classification

| Method | Params (M) | Accuracy (%) |
|--------|-----------|-------------|
| WideResNet-28-10 | 36 | 96.3 |
| ResNeXt-29-16×64d | 68 | 96.4 |
| SBGC | N/A | 95.0 |
| DDAE | 36 | 97.2 |
| **PRG-GVP** | **42** | **97.25** |
| **PRG-ICFM** | **42** | **97.32** |
| **PRG-OTCFM** | **42** | **97.42** |

PRG-OTCFM surpasses the strongest baseline DDAE, reaching 97.42%.

### Validation of Continuous Feature Extractor

| Inference Steps | 20 | 100 | 500 | 1000 |
|----------------|----|-----|-----|------|
| PRG-OTCFM | 97.42 | 97.43 | 97.43 | 97.44 |

Although trained with $t_{\text{span}}=20$, inference with any number of steps (20–1000) yields nearly identical performance, validating the robustness of **continuous feature extraction**.

### Effect of Finetuning Trajectory Length

| Starting Point | CIFAR-10 | Tiny-ImageNet |
|---------------|----------|---------------|
| Classifier head only (frozen flow model) | ~50% | ~20% |
| $x_{1/4} \to x_1$ | 95.8 | 52.0 |
| $x_{1/2} \to x_1$ | 97.0 | **58.4** |
| $x_0 \to x_1$ | **97.4** | 56.1 |

- CIFAR-10 (simpler): full-trajectory finetuning is optimal.
- Tiny-ImageNet (more complex): starting from an intermediate point is preferable; overly long trajectories may lead to overfitting.

### Pretraining Quality vs. Finetuning Performance

Longer pretraining (higher mutual information) leads to better finetuning performance. Without pretraining, accuracy is only 73.5%; after sufficient pretraining, it reaches 97.4%.

## Highlights & Insights

1. **Model-agnostic**: does not depend on specific network architectures (applicable to both UNet and Transformer); the latent variable $Z$ is determined by the ODE solver, independent of network structure.
2. **Infinite-depth expressiveness**: continuous-time flow models provide an infinite-layer structure, achieving high expressiveness with a small parameter count.
3. **Elegant realization of the pretrain-finetune paradigm**: generative pretraining combined with discriminative finetuning demonstrates that the two are complementary rather than opposing.
4. **Flexible feature selection**: different tasks can leverage features extracted at different points along the trajectory.

## Limitations & Future Work

- Experiments are conducted only at 64×64 resolution; effectiveness at higher resolutions remains unknown.
- Finetuning requires end-to-end training of the flow model (substantial computational cost); freezing the model and training only the classifier yields poor performance.
- Backpropagation through the ODE solver increases memory and computational requirements.
- Direct comparison with self-supervised methods such as MAE and DINO is absent.

## Related Work & Insights

- Generative classifiers: Diffusion Classifier, HybViT, SBGC
- Representation learning: Denoising Autoencoders (DAE), MAE, iGPT
- Diffusion features: DDAE, DiffusionDet, Baranchuk et al.

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 4 |
| Technical Depth | 5 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| **Overall** | **4.2** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] VisualCloze: A Universal Image Generation Framework via Visual In-Context Learning](visualcloze_a_universal_image_generation_framework_via_visual_in-context_learnin.md)
- [\[ICCV 2025\] Unsupervised Imaging Inverse Problems with Diffusion Distribution Matching](unsupervised_imaging_inverse_problems_with_diffusion_distribution_matching.md)
- [\[NeurIPS 2025\] Latent Zoning Network: A Unified Principle for Generative Modeling, Representation Learning, and Classification](../../NeurIPS2025/image_generation/latent_zoning_network_a_unified_principle_for_generative_modeling_representation.md)
- [\[ICCV 2025\] GenFlowRL: Shaping Rewards with Generative Object-Centric Flow in Visual Reinforcement Learning](genflowrl_shaping_rewards_with_generative_object-centric_flow_in_visual_reinforc.md)
- [\[ICCV 2025\] Randomized Autoregressive Visual Generation](randomized_autoregressive_visual_generation.md)

</div>

<!-- RELATED:END -->
