---
title: >-
  [Paper Note] Bilevel Layer-Positioning LoRA for Real Image Dehazing
description: >-
  [CVPR2026][Model Compression][image dehazing] This paper proposes BiLaLoRA, which employs bilevel optimization to automatically identify the optimal network layers for LoRA insertion, coupled with H2C Loss — an unsupervised dehazing loss based on CLIP semantic directions — to efficiently adapt synthetic-data-pretrained dehazing models to real-world scenes. The approach reduces training time by 77.7% while matching full fine-tuning performance, and generalizes across models and domains.
tags:
  - CVPR2026
  - Model Compression
  - image dehazing
  - LoRA
  - bilevel optimization
  - CLIP
  - unsupervised adaptation
  - parameter-efficient fine-tuning
date: 2026-05-08
content_hash: 8dd59c666e7ae26b
---

# Bilevel Layer-Positioning LoRA for Real Image Dehazing

**Conference**: CVPR2026
**arXiv**: [2603.10872](https://arxiv.org/abs/2603.10872)
**Code**: [GitHub](https://github.com/YanZhang-zy/BiLaLoRA)
**Area**: Model Compression
**Keywords**: image dehazing, LoRA, bilevel optimization, CLIP, unsupervised adaptation, parameter-efficient fine-tuning

## TL;DR

This paper proposes BiLaLoRA, which employs bilevel optimization to automatically identify the optimal network layers for LoRA insertion, coupled with H2C Loss — an unsupervised dehazing loss based on CLIP semantic directions — to efficiently adapt synthetic-data-pretrained dehazing models to real-world scenes. The approach reduces training time by 77.7% while matching full fine-tuning performance, and generalizes across models and domains.

## Background & Motivation

Image dehazing is a classical problem in low-level vision. Mainstream methods rely on synthetic data (e.g., ITS/OTS from the RESIDE dataset) for supervised training, but suffer from a severe domain gap:

**Synthetic-to-real domain gap**: Synthetic hazy images are generated via the atmospheric scattering model $I(x) = J(x)t(x) + A(1-t(x))$, which differs significantly from the complex degradations in real haze (non-uniform haze, color shifts, multi-layer fog, etc.).

**Absence of paired real data**: It is nearly impossible to obtain paired hazy/clear images of the same real scene, rendering conventional supervised fine-tuning infeasible.

**High cost of full fine-tuning**: For Transformer-based dehazing models, updating all parameters is time-consuming and prone to overfitting on limited adaptation data.

Limitations of prior methods:
- **Domain adaptation methods** (DA-dehazing, USID-Net) rely on CycleGAN-style translation, which suffers from training instability and may introduce artifacts.
- **LoRA fine-tuning** reduces parameter count, but **the choice of which layers to insert LoRA into is critical** — random or uniform placement is far from optimal.

## Core Problem

How can a dehazing model pretrained on synthetic data be adapted to real haze scenes at minimal training cost, without any real paired supervision? Two sub-problems must be addressed:
1. Design of an unsupervised optimization objective in the absence of real ground truth.
2. Automated and optimal selection of LoRA insertion layers.

## Method

### H2C Loss: Haze-to-Clear Text-Guided Loss

**Core idea**: The CLIP-pretrained vision-language alignment space is leveraged to construct a semantic direction from "hazy" to "clear" as an unsupervised dehazing signal.

**Defining positive and negative text prompts**:
- $T_{\text{pos}}$ (positive/clear): "a clear photo", "a bright image", "a high-quality photo"
- $T_{\text{neg}}$ (negative/hazy): "a hazy photo", "a foggy image", "a blurry photo"

**Semantic direction computation**:

Image-domain direction: $\Delta V_{\text{img}} = V_{\text{out}} - V_{\text{in}}$

where $V_{\text{out}} = \text{CLIP}_{\text{img}}(\hat{J})$ is the CLIP image feature of the dehazed output and $V_{\text{in}} = \text{CLIP}_{\text{img}}(I)$ is the feature of the hazy input.

Text-domain direction: $\Delta T_{\text{text}} = T_{\text{pos}} - T_{\text{neg}}$

**H2C Loss**:

$$\mathcal{L}_{\text{H2C}} = 1 - \cos(\Delta V_{\text{img}}, \Delta T_{\text{text}})$$

This maximizes the cosine similarity between the image change direction and the "hazy→clear" text direction, requiring no real ground truth and relying entirely on CLIP's semantic priors.

### BiLaLoRA: Bilevel Optimization for Layer Positioning

Determining which layers to insert LoRA into and how to weight their importance is a combinatorial optimization problem. BiLaLoRA formulates this as bilevel optimization:

**Upper-level optimization** (layer selection weights $\alpha$):

$$\min_{\alpha} \mathcal{L}_{\text{val}}(\omega^*(\alpha), \alpha)$$

**Lower-level optimization** (LoRA weights $\omega$):

$$\omega^*(\alpha) = \arg\min_{\omega} \mathcal{L}_{\text{train}}(\omega, \alpha)$$

where $\alpha = \{\alpha_1, \ldots, \alpha_L\}$ denotes per-layer selection weights and $\omega$ denotes all LoRA parameters.

**Continuous relaxation**: Gumbel-Sigmoid is applied to relax the discrete layer selection:

$$g_l = \sigma\left(\frac{\log(\alpha_l / (1-\alpha_l)) + G}{\tau}\right)$$

where $G$ is Gumbel noise and $\tau$ is a temperature parameter. During training, $g_l$ serves as a continuous weight; after the search, the Top-K layers are selected and fixed.

**Efficient hypergradient computation**: Standard bilevel optimization requires second-order Hessian computation. This work adopts a rank-one approximation:

$$\nabla_\alpha \mathcal{L}_{\text{val}} \approx \nabla_\alpha \mathcal{L}_{\text{val}} - \frac{\eta}{\epsilon} (\nabla_\alpha \mathcal{L}_{\text{train}}(\omega^+) - \nabla_\alpha \mathcal{L}_{\text{train}}(\omega^-))$$

where $\omega^\pm = \omega \pm \epsilon \nabla_\omega \mathcal{L}_{\text{val}}$, requiring only first-order derivatives.

### Two-Stage Training Pipeline

1. **Stage 1 — Bilevel Search**: $\alpha$ and $\omega$ are jointly optimized using Gumbel-Sigmoid continuous relaxation. After the search, Top-K layers are selected based on $\alpha$ values.
2. **Stage 2 — LoRA Fine-tuning**: LoRA weights are trained only on the Top-K fixed layers; all other layers remain frozen.

### Total Training Loss

$$\mathcal{L} = \mathcal{L}_{\text{H2C}} + \lambda \mathcal{L}_{\text{reg}}$$

where $\mathcal{L}_{\text{reg}}$ is a regularization term preventing excessive deviation of the dehazed output from the input, and $\lambda$ is a balancing coefficient.

## Key Experimental Results

### Cross-Model Adaptation Performance

BiLaLoRA is effective across four different dehazing backbones:

| Base Model | Method | RTTS (MUSIQ↑) | URHI (MUSIQ↑) | Parameters |
|-----------|--------|--------------|--------------|------------|
| MSBDN | Full Fine-tuning | Baseline | Baseline | 100% |
| MSBDN | **BiLaLoRA** | **On par** | **On par** | ~5% |
| DeHamer | Full Fine-tuning | Baseline | Baseline | 100% |
| DeHamer | **BiLaLoRA** | **On par** | **On par** | ~5% |
| ConvIR | Full Fine-tuning | Baseline | Baseline | 100% |
| ConvIR | **BiLaLoRA** | **On par** | **On par** | ~5% |
| DEA | Full Fine-tuning | Baseline | Baseline | 100% |
| DEA | **BiLaLoRA** | **On par** | **On par** | ~5% |

### Comparison with Real Dehazing SOTA

| Method | RTTS | URHI | Fattal |
|--------|------|------|--------|
| DAD (CVPR 2020) | Low | Low | Low |
| USID-Net (TIP 2022) | Medium | Medium | Medium |
| **BiLaLoRA (Ours)** | **SOTA** | **SOTA** | **SOTA** |

State-of-the-art results are achieved on all three real dehazing benchmarks: RTTS, URHI, and Fattal.

### Training Efficiency

| Method | Training Time | vs. Full Fine-tuning |
|--------|--------------|----------------------|
| Full Fine-tuning | 100% | Baseline |
| LoRA (uniform) | ~40% | −60% |
| **BiLaLoRA** | **~22.3%** | **−77.7%** |

Training time is reduced by 77.7%, primarily due to training only on a small number of layers after Stage 1 search.

### Ablation Study

| Component | MUSIQ | Note |
|-----------|-------|------|
| Full BiLaLoRA | Best | — |
| w/o H2C Loss (replaced with L1) | Significant drop | Unsupervised adaptation fails |
| Uniform LoRA (no bilevel search) | Drop | Demonstrates importance of layer selection |
| Random layer selection | Larger drop | Worse than uniform |
| Full-layer LoRA | Moderate | More parameters but inferior to targeted placement |

### Cross-Domain Adaptation

| Training Data | Test Data | BiLaLoRA |
|--------------|-----------|----------|
| ITS (indoor synthetic) | RTTS (real) | Effective |
| OTS (outdoor synthetic) | URHI (real) | Effective |
| Daytime haze | Nighttime haze | Effective |
| Synthetic domain A | Synthetic domain B | Effective |

## Highlights & Insights

1. **Elegant H2C Loss design**: Constructing an unsupervised loss via directional semantics in CLIP space is more stable than CycleGAN-style methods and avoids artifact risks. The key insight is using a *directional difference* rather than an absolute distance, preventing degenerate solutions where the output is forced to match specific text embeddings.
2. **Applying NAS ideas to LoRA layer selection**: Bilevel optimization automates optimal layer identification, eliminating manual trial-and-error. Gumbel-Sigmoid relaxation makes the discrete search differentiable.
3. **Rank-one approximation reduces bilevel optimization cost**: Simplifying second-order Hessian computation to first-order significantly improves practical feasibility.
4. **Strong cross-model generalizability**: The method is effective on both CNN (MSBDN, ConvIR) and Transformer (DeHamer, DEA) architectures, demonstrating architecture-agnostic applicability.
5. **Decoupled two-stage pipeline**: Separating search from training allows the search phase to complete rapidly, with training focused on only the most effective layers.

## Limitations & Future Work

1. **CLIP dependency**: The quality of H2C Loss depends on the quality of CLIP's semantic space. For degradation types not well covered by CLIP (e.g., extreme haze), the directional signal may be inaccurate.
2. **Hard Top-K truncation**: Selecting Top-K layers after the search is a discrete approximation that may discard contributions from borderline layers. The choice of K requires cross-validation.
3. **Validation limited to dehazing**: Although the framework is general, experiments are conducted only on dehazing; effectiveness on other low-level vision tasks (deraining, denoising, super-resolution) remains unverified.
4. **Limitations of perceptual quality metrics**: Evaluation relies primarily on no-reference metrics such as MUSIQ; validation using reference metrics (PSNR/SSIM on paired data) is absent.
5. **Lack of comparison with recent vision foundation models**: No comparison is made against generative dehazing methods such as Stable Diffusion.

## Related Work & Insights

- **vs. standard LoRA fine-tuning**: Uniform LoRA insertion underperforms BiLaLoRA's adaptive selection, demonstrating that *where* LoRA is inserted matters more than *how much*.
- **Analogy to DARTS** (classic bilevel optimization in NAS): BiLaLoRA transfers DARTS's operation search paradigm to LoRA layer selection, representing a cross-disciplinary innovation between NAS and PEFT.
- **Connection to CLIP-guided methods** (CLIPasso, StyleCLIP): All leverage CLIP semantic directions to guide optimization, but BiLaLoRA introduces directional loss for low-level vision tasks, with a use case more grounded in physical degradation modeling.
- **Insights**: (1) The bilevel layer selection paradigm can be extended to position optimization for other PEFT methods (e.g., Adapter, Prefix Tuning); (2) The "directional" design of H2C Loss can be applied to other unsupervised image restoration tasks — requiring only the definition of a degraded→restored text direction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — H2C Loss and bilevel layer positioning each contribute independently, forming a coherent and complete solution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Cross-model, cross-domain, and ablation experiments are thorough, though reference-metric validation is absent.
- **Practicality**: ⭐⭐⭐⭐⭐ — A 77.7% reduction in training time with no performance degradation is highly deployment-friendly.
- **Writing Quality**: ⭐⭐⭐⭐ — Method description is clear; mathematical derivation of bilevel optimization is complete.
- **Overall**: ⭐⭐⭐⭐ (4.0/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Frequency Switching Mechanism for Parameter-Efficient Multi-Task Learning](frequency_switching_mechanism_for_parameter-ecient_multi-task_learning.md)
- [\[CVPR 2026\] Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning](towards_generalizable_ai-generated_image_detection_via_image-adaptive_prompt_lea.md)
- [\[ACL 2026\] A Layer-wise Analysis of Supervised Fine-Tuning](../../ACL2026/model_compression/a_layer-wise_analysis_of_supervised_fine-tuning.md)
- [\[ICLR 2026\] LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts](../../ICLR2026/model_compression/ld-mole_learnable_dynamic_routing_for_mixture_of_lora_experts.md)
- [\[ICLR 2026\] ABBA-Adapters: Efficient and Expressive Fine-Tuning of Foundation Models](../../ICLR2026/model_compression/abba-adapters_efficient_and_expressive_fine-tuning_of_foundation_models.md)

</div>

<!-- RELATED:END -->
