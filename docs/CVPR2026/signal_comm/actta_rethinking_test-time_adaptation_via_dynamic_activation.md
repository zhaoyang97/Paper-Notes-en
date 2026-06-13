---
title: >-
  [Paper Note] AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation
description: >-
  [CVPR 2026][Signal & Communication][Test-time adaptation] This paper proposes AcTTA, a test-time adaptation framework based on dynamic activation function modulation. By reparameterizing conventional fixed activation fun…
tags:
  - "CVPR 2026"
  - "Signal & Communication"
  - "Test-time adaptation"
  - "activation function"
  - "distribution shift"
  - "normalization layer"
  - "dynamic activation"
date: 2026-05-08
content_hash: 706849a7b548233c
---

# AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation

**Conference**: CVPR 2026
**arXiv**: [2603.26096](https://arxiv.org/abs/2603.26096)  
**Code**: [https://hyeongyu-kim.github.io/actta/](https://hyeongyu-kim.github.io/actta/)  
**Area**: Signal & Communication / Test-Time Adaptation
**Keywords**: Test-time adaptation, activation function, distribution shift, normalization layer, dynamic activation

## TL;DR
This paper proposes AcTTA, a test-time adaptation framework based on dynamic activation function modulation. By reparameterizing conventional fixed activation functions into a learnable form—incorporating an activation center shift and asymmetric gradient slopes—AcTTA adaptively adjusts activation behavior during inference to address distribution shift, consistently outperforming normalization-layer-based TTA methods on CIFAR10-C, CIFAR100-C, and ImageNet-C.

## Background & Motivation

1. **Background**: Test-time adaptation (TTA) is an important paradigm for addressing the mismatch between deployment environments and training distributions. Existing TTA methods primarily focus on adjusting affine parameters and recalibrating running statistics of normalization layers; methods such as TENT, EATA, and SAR all center their adaptation mechanisms on normalization layers.

2. **Limitations of Prior Work**: This normalization-centric perspective overlooks a critical component—the activation function. As the nonlinear core of a network, activation functions fundamentally shape the geometry of the feature space and determine how the model responds to input variations. Yet in TTA, activation functions have consistently been treated as fixed nonlinear mappings and have never been incorporated into the adaptation process.

3. **Key Challenge**: Under distribution shift, the source-domain statistics stored in BN layers no longer align with target-domain features, producing biased feature representations. When these biased features pass through zero-centered activation functions (e.g., ReLU, GELU), useful signals may be suppressed below the activation boundary, leading to information loss and vanishing gradients. This "zero-center rigidity" is a key factor limiting adaptation effectiveness.

4. **Goal**: The paper seeks to make activation functions themselves adaptable components within TTA by: (1) adjusting gradient behavior to maintain learning flow; (2) shifting activation boundaries to align with new feature centers; and (3) preserving compatibility with source-domain pretrained representations.

5. **Key Insight**: The authors observe that outside of TTA, learnable or modulatable activation functions (e.g., PReLU, ACON, PAU) have demonstrated that even subtle modifications to activation behavior can improve performance and training stability—indicating that activation functions possess inherent learnable flexibility.

6. **Core Idea**: Transform activation functions from fixed components into adaptive participants—by parameterizing the activation center and asymmetric slopes, enabling the network to self-correct internal biases at inference time.

## Method

### Overall Architecture
AcTTA is a modular, objective-agnostic activation adaptation framework. Given a pretrained model and target-domain test data, AcTTA inserts a learnable dynamic activation module at each activation function location, updating only the activation parameters ($\lambda_{pos}$, $\lambda_{neg}$, $c$) without modifying network weights or requiring source-domain data. The framework integrates seamlessly with any existing TTA objective (e.g., entropy minimization, consistency regularization).

### Key Designs

1. **Dynamic Activation Reparameterization**:

    - **Function**: Transforms fixed activation functions into a parameterized form adjustable at inference time.
    - **Mechanism**: Modern activation functions can be approximated as $\phi(x) = x \cdot \sigma(\beta x)$, whose derivative is an input-dependent slope function. AcTTA explicitly exposes this slope as a learnable function $\lambda(x) = \lambda_{neg} + (\lambda_{pos} - \lambda_{neg}) \sigma(\beta x)$, where $\lambda_{neg}$ and $\lambda_{pos}$ control the asymptotic slopes in the negative and positive regions, respectively. A learnable center parameter $c$ is also introduced to relocate the activation boundary. The resulting activation is:
      $$g(x) = \phi(x-c) + [\lambda_{neg} + (\lambda_{pos} - \lambda_{neg})\sigma(\beta(x-c))](x-c)$$
    - **Design Motivation**: Slope adaptation alone cannot correct feature bias; introducing the center shift $c$ allows the activation function to dynamically recenter according to target-domain statistics. When initialized with $\lambda_{neg}=\lambda_{pos}=0$ and $c=0$, $g(x)$ exactly recovers the original $\phi(x)$, ensuring compatibility with pretrained models.

2. **Asymmetric Gradient Modulation**:

    - **Function**: Independently controls gradient scaling in the positive and negative regions to stabilize gradient propagation.
    - **Mechanism**: By learning $\lambda_{pos}$ and $\lambda_{neg}$ separately, the activation function maintains nonzero gradients in the negative region (avoiding dead gradient problems) and flexibly adjusts response strength in the positive region. This preserves effective gradient flow even when feature distributions are skewed under distribution shift.
    - **Design Motivation**: Conventional zero-centered activation functions produce gradient imbalance or biased updates under distribution shift. AcTTA's asymmetric design allows the model to maintain stable optimization at learning rates up to 10× those used in conventional methods.

3. **Architecture-Adaptive Trainable Parameter Selection**:

    - **Function**: Selects the optimal set of trainable parameters according to the backbone architecture.
    - **Mechanism**: For BN-based CNNs (e.g., WRN), freezing BN affine parameters $(\gamma, \beta)$ and adapting only activation parameters $(\lambda_{pos}, \lambda_{neg}, c)$ yields the best results. For LN-based ViTs, jointly adapting normalization and activation parameters is optimal.
    - **Design Motivation**: BN relies on perturbed running statistics, so further modifying $(\gamma, \beta)$ may amplify distribution noise. LN performs per-sample normalization independently of source-domain statistics, so its $(\gamma, \beta)$ provides degrees of freedom complementary to the activation parameters.

### Loss & Training
AcTTA is objective-agnostic—it does not define its own loss function but instead incorporates activation parameters into the learnable parameter set of any existing TTA method. In practice, learning rates approximately 10× higher than conventional methods are used (e.g., $10^{-2}$ vs. $10^{-3}$), as the asymmetric gradient design guarantees stability. The default depth configuration is 50% (i.e., activation functions in the first half of layers are made learnable).

## Key Experimental Results

### Main Results

| Dataset / Backbone | Metric (Err%) | AcTTA_TENT | TENT | Gain |
|--------------------|--------------|------------|------|------|
| CIFAR10-C / WRN-28 | Error | 17.03 | 18.51 | −1.48 |
| CIFAR10-C / ResNeXt | Error | 9.53 | 10.28 | −0.75 |
| CIFAR100-C / WRN-40 | Error | 33.81 | 35.25 | −1.44 |
| ImageNet-C / ResNet50 (BN) | Error | 64.95 | 66.50 | −1.55 |
| ImageNet-C / ResNet50 (GN) | Error | 66.84 | 69.60 | −2.76 |
| ImageNet-C / ViT-B/16 | Error | 51.79 | 53.85 | −2.06 |

Combining AcTTA with other TTA baselines (ETA, SAR, DeYO, ROID, CMF) also yields consistent improvements, demonstrating strong modular compatibility.

### Ablation Study

| Configuration | WRN-28 Err% | ViT-B/16 Err% | Note |
|---------------|-------------|--------------|------|
| TENT (γ, β only) | 18.51 | 53.85 | Baseline |
| AcTTA (γ, β, λ+, λ−, c) | 18.06 | 52.37 | Full parameters |
| AcTTA* (λ+, λ−, c only) | 17.03 | 55.30 | Frozen BN; optimal for CNN |
| AcTTA* (c only) | 17.50 | 61.56 | Center shift only |
| No adaptation | 43.52 | 62.10 | Original model |

### Key Findings
- **Activation parameters > normalization parameters on BN-CNNs**: Freezing BN affine parameters and adapting only activation parameters (AcTTA*) achieves the lowest error rate of 17.03% on WRN-28, indicating that BN and activation adaptation play overlapping roles.
- **Center shift $c$ contributes substantially**: Adapting $c$ alone yields a notable improvement on CNNs (18.51→17.50), suggesting that residual bias induced by source-domain running statistics can be compensated by shifting the activation boundary.
- **Stability at high learning rates**: AcTTA remains stable at 10× the learning rate (34.56% on CIFAR100-C at LR=1e-2), whereas TENT collapses entirely under equivalent conditions (51.57%).
- **Comparison with other learnable activation functions**: PReLU and PAU perform poorly in TTA settings (PAU reaches 99.96% error on ViT), demonstrating that TTA requires not merely parameterized slopes but joint center-shift and asymmetric slope modulation.
- **Optimal adaptation depth is architecture-dependent**: WRN peaks at ~50%, ViT at ~25%, and ResNet at ~75%.

## Highlights & Insights
- **Activation functions as a new dimension for TTA**: This is the first work to systematically incorporate activation functions into the TTA framework, extending the traditional normalization-centric perspective. This idea is transferable to other online adaptation scenarios such as continual learning and domain generalization.
- **Compatibility-by-design initialization**: The design that exactly recovers the original activation function when $\lambda=0, c=0$ is elegant—it guarantees a lossless, zero-risk starting point. This "additive module" design is a reusable architectural trick.
- **Stability at high learning rates**: By maintaining nonzero gradients in the negative region, AcTTA intrinsically addresses vanishing gradients under distribution shift, enabling more aggressive learning rates. This finding has significant implications for real-time TTA deployment.

## Limitations & Future Work
- **Optimal depth requires prior knowledge**: The optimal adaptation depth varies by architecture (10%–75%); the paper adopts 50% as a compromise, which is not necessarily optimal.
- **Limited effectiveness in small-batch settings**: At batch size 4, certain combinations (e.g., AcTTA_SAR on ViT) underperform the baseline, suggesting that activation adaptation retains some dependence on batch statistics.
- **Validation limited to corruption benchmarks**: More complex distribution shift scenarios (e.g., natural domain shifts, cross-modal shifts) are not evaluated.
- **Computational overhead not quantified**: The number of additional learnable parameters and the extra forward/backward computation time are not analyzed in detail.

## Related Work & Insights
- **vs. TENT**: TENT adapts only BN layer parameters $(\gamma, \beta)$. AcTTA demonstrates that on CNNs, freezing BN parameters and adapting only activation functions yields better results, indicating overlapping roles among different adaptable components.
- **vs. ACON**: ACON introduces a learnable gate but still assumes a zero-centered boundary and cannot handle shifted feature distributions. AcTTA's center-shift design directly addresses this limitation.
- **vs. PAU**: PAU is effective as a general learnable activation during training but collapses in TTA settings (99.96% error on ViT), demonstrating that TTA requires targeted shift–slope modulation rather than general function approximation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The activation-function perspective is genuinely novel in TTA, though the reparameterization form is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers multiple datasets, architectures, and TTA baselines; ablation studies are comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Logic is clear, with a coherent progression from motivation to method to experiments.
- **Value**: ⭐⭐⭐⭐ — Opens a new research direction for TTA, though the practical gains are modest (1–3% error reduction).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Enhancing Instruction Following of LLMs via Activation Steering with Dynamic Rejection](../../ICLR2026/signal_comm/enhancing_instruction_following_of_llms_via_activation_steering_with_dynamic_rej.md)
- [\[NeurIPS 2025\] Angular Steering: Behavior Control via Rotation in Activation Space](../../NeurIPS2025/signal_comm/angular_steering_behavior_control_via_rotation_in_activation_space.md)
- [\[CVPR 2026\] ChartNet: A Million-Scale, High-Quality Multimodal Dataset for Robust Chart Understanding](chartnet_a_million-scale_high-quality_multimodal_dataset_for_robust_chart_unders.md)
- [\[CVPR 2026\] FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection](faar_efficient_frequency-aware_multi-task_fine-tuning_via_automatic_rank_selecti.md)
- [\[CVPR 2026\] CLAY: Conditional Visual Similarity Modulation in Vision-Language Embedding Space](clay_conditional_visual_similarity.md)

</div>

<!-- RELATED:END -->
