---
title: >-
  [Paper Note] AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation
description: >-
  [CVPR 2026][Self-Supervised Learning][Test-Time Adaptation] This paper proposes AcTTA, a framework that for the first time treats activation functions as learnable components for test-time adaptation (TTA). By introducing a parameterized activation center shift $c$ and asymmetric gradient scaling $\lambda_{pos}, \lambda_{neg}$ to replace or augment conventional normalization-layer adaptation, AcTTA consistently outperforms all normalization-based TTA methods on CIFAR-10/100-C and ImageNet-C, while supporting learning rates up to 10× larger.
tags:
  - CVPR 2026
  - Self-Supervised Learning
  - Test-Time Adaptation
  - Dynamic Activation Functions
  - Domain Shift
  - Normalization Layers
  - Gradient Flow
date: 2026-05-08
content_hash: a63c81e1c825d0c4
---

# AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation

**Conference**: CVPR 2026  
**arXiv**: [2603.26096](https://arxiv.org/abs/2603.26096)  
**Code**: [https://hyeongyu-kim.github.io/actta/](https://hyeongyu-kim.github.io/actta/)  
**Area**: Model Adaptation / Domain Shift  
**Keywords**: Test-Time Adaptation, Dynamic Activation Functions, Domain Shift, Normalization Layers, Gradient Flow

## TL;DR
This paper proposes AcTTA, a framework that for the first time treats activation functions as learnable components for test-time adaptation (TTA). By introducing a parameterized activation center shift $c$ and asymmetric gradient scaling $\lambda_{pos}, \lambda_{neg}$ to replace or augment conventional normalization-layer adaptation, AcTTA consistently outperforms all normalization-based TTA methods on CIFAR-10/100-C and ImageNet-C, while supporting learning rates up to 10× larger.

## Background & Motivation

**Background**: TTA mitigates performance degradation caused by domain shift by updating model parameters at inference time. Existing methods (TENT, EATA, SAR, DeYO, ROID, CMF, etc.) predominantly focus on updating the affine parameters $(\gamma, \beta)$ and running statistics of normalization layers.

**Limitations of Prior Work**: Activation functions—the core nonlinear components that shape the geometry of the feature space—have been entirely overlooked in TTA and have always been treated as fixed, immutable mappings. However, zero-centered activations such as ReLU/GELU may suppress informative features under domain shift (shifted feature values fall into the negative region and are truncated), leading to information loss and vanishing gradients.

**Key Challenge**: Domain shift causes BN's source-domain statistics to misalign with target features → normalized features are biased and truncated by zero-centered activations → information loss + vanishing gradients → adaptation becomes difficult.

**Key Insight**: In the broader TTA-adjacent literature, learnable activation functions such as PReLU, ACON, and PAU have demonstrated that even minor modifications to activation behavior can improve performance. Yet this flexibility has never been explored in TTA.

**Core Idea**: Reparameterize activation functions into a learnable form — a center shift $c$ moves the activation boundary from zero to the actual mean of the shifted features, while asymmetric gradient scaling $\lambda_{pos}, \lambda_{neg}$ allows independent control of gradient magnitude in the positive and negative regions. The parameters are initialized to recover the original activation exactly, ensuring compatibility with pretrained weights.

## Method

### Overall Architecture
Pretrained model → Insert AcTTA modules (learnable $c, \lambda_{pos}, \lambda_{neg}$) at each activation function location → Update these parameters at inference time using objectives such as entropy minimization or consistency regularization → Serve as a plug-and-play enhancement combinable with any TTA baseline.

### Key Designs

1. **Dynamic Activation Reparameterization**:

    - Function: Reparameterize the fixed activation function $\phi(x)$ into an adaptive, learnable form.
    - Mechanism:
    $g(x) = \underbrace{\phi(x-c)}_{\text{shifted base function}} + \underbrace{[\lambda_{neg} + (\lambda_{pos} - \lambda_{neg})\sigma(\beta(x-c))](x-c)}_{\text{adaptive slope modulation}}$
    - **Center shift $c$**: Relocates the activation boundary from zero to the actual mean of the shifted features. When BN fails, features shift globally and zero-centered activations incorrectly truncate them; $c$ automatically compensates.
    - **Asymmetric gradient scaling $\lambda_{pos}, \lambda_{neg}$**: Independently controls gradient magnitude in the positive and negative regions. $\lambda_{neg}$ maintains non-zero gradients in the negative region (avoiding "dead neurons"); $\lambda_{pos}$ controls the response magnitude in the positive region.
    - **Identity initialization**: Setting $\lambda_{pos}=\lambda_{neg}=0, c=0$ exactly recovers the original $\phi(x)$, ensuring full compatibility with pretrained models.
    - Design Motivation: (1) After domain shift, the activation boundary no longer aligns with the feature distribution center → a shift is needed; (2) Vanishing gradients in the negative region impede adaptation → gradient flow must be preserved.

2. **Modular and Objective-Agnostic Design**:

    - Function: AcTTA serves as a plug-and-play component combinable with any TTA objective.
    - Mechanism: Marking the activation parameters as learnable is sufficient to seamlessly integrate AcTTA with any baseline, including TENT, EATA, SAR, DeYO, ROID, and CMF, denoted as AcTTA$_\text{TENT}$, AcTTA$_\text{ETA}$, etc.
    - Design Motivation: No binding to specific optimization rules → maximizes generality.

3. **Architecture-Dependent Optimal Adaptation Strategy**:

    - **CNN + BN**: Freeze BN affine parameters $(\gamma, \beta)$ and **update only the activation parameters** $(\lambda_{pos}, \lambda_{neg}, c)$ → yields the best performance (AcTTA*). Rationale: BN relies on running statistics corrupted by domain shift; modifying $(\gamma, \beta)$ amplifies noise, whereas activation parameters provide more locally stable compensation.
    - **ViT + LN**: Jointly update $(\gamma, \beta)$ and activation parameters → yields the best performance. Rationale: LN normalizes per sample and does not depend on running statistics; modifying $(\gamma, \beta)$ provides additional channel-wise rescaling freedom that complements rather than competes with activation modulation.

### Loss & Training
AcTTA does not define a new loss function — it uses the loss of the host TTA method (e.g., entropy minimization in TENT). The key contribution is enabling stable training at **learning rates up to 10× larger**, owing to the preservation of non-zero gradients in the negative region.

## Key Experimental Results

### Main Results (Large Batch, Severity 5)

| Method | CIFAR10-C WRN28 | CIFAR100-C WRN40 | ImageNet-C ResNet50-BN | ImageNet-C ViT-B/16 |
|------|:---:|:---:|:---:|:---:|
| TENT | 18.51 | 35.25 | 66.50 | 53.85 |
| **AcTTA$_\text{TENT}$** | **17.03** | **33.81** | **64.95** | **51.79** |
| ETA | 18.07 | 36.33 | 62.84 | 49.84 |
| **AcTTA$_\text{ETA}$** | **16.74** | **34.90** | **61.74** | **48.90** |
| ROID | 17.52 | 35.38 | 59.17 | 52.91 |
| **AcTTA$_\text{ROID}$** | **16.43** | **33.59** | **59.74** | **49.89** |
| DeYO | 18.31 | 35.17 | 62.34 | 47.73 |
| **AcTTA$_\text{DeYO}$** | **17.11** | **33.55** | **61.79** | **49.64** |

AcTTA yields consistent improvements across **all 6 TTA baselines × 7 architectures**.

### Ablation Study (Trainable Parameter Selection)

| Trainable Parameters | WRN-28 (BN) | ViT-B/16 (LN) |
|-----------|:-----------:|:-------------:|
| $\gamma, \beta$ (TENT) | 18.51 | 53.85 |
| $\gamma, \beta, \lambda_{pos}, \lambda_{neg}, c$ | 18.06 | 52.37 |
| **$\lambda_{pos}, \lambda_{neg}, c$ (AcTTA*, frozen BN)** | **17.03** | 55.30 |
| $c$ only (AcTTA*, frozen BN) | 17.50 | **61.56** |

### Learning Rate Sensitivity

| Method | LR=1e-2, BS=128 | LR=1e-3, BS=128 |
|------|:---:|:---:|
| TENT | 51.57 (collapse) | 35.25 |
| **AcTTA$_\text{TENT}$** | **34.56** | 37.61 |

AcTTA remains stable under aggressive learning rates at which TENT collapses.

### Key Findings
- **Most counterintuitive finding on CNN + BN**: Freezing BN parameters and updating only activation parameters (AcTTA*) outperforms updating both simultaneously — BN adaptation and activation modulation exhibit functional overlap.
- **Different behavior on ViT + LN**: Jointly updating LN parameters and activation parameters yields the best performance — LN normalization and activation modulation are complementary.
- **Depth analysis**: Not all layers benefit equally from learnable activations. The optimal adaptation depth varies by architecture (WRN ~50%, ViT ~25%); adapting too deep leads to instability.
- AcTTA supports 10× larger learning rates because adaptive slope modulation maintains non-zero gradients in the negative region.
- AcTTA's stability advantage is more pronounced in extreme small-batch scenarios (batch size = 4).

## Highlights & Insights
- **An overlooked TTA dimension**: Activation functions have long been regarded as fixed architectural choices. AcTTA is the first to demonstrate that they constitute a powerful adaptation target — capable of entirely replacing normalization-layer adaptation.
- **The counterintuitive "freeze BN, adapt only activations" strategy**: This challenges the default paradigm in TTA that updating $(\gamma, \beta)$ is the only viable path, opening a new direction for TTA method design.
- **Minimalist yet effective**: Only 3 parameters ($c, \lambda_{pos}, \lambda_{neg}$) are added per activation layer, with negligible parameter overhead but significant performance gains.
- **Modular and objective-agnostic**: Uniquely compatible with all existing TTA methods, functioning as a universal "TTA enhancement plugin."

## Limitations & Future Work
- The optimal adaptation depth varies by architecture, requiring empirical tuning for different backbones.
- The current center shift $c$ is a global scalar — channel-wise or spatial adaptive shifts remain unexplored.
- Validation is limited to classification tasks; the effectiveness on pixel-level tasks such as segmentation and detection remains to be investigated.
- Stability under continual TTA (sequential adaptation across multiple domains) has not been thoroughly evaluated.

## Related Work & Insights
- **vs. TENT/EATA/SAR**: These methods adapt only normalization layer parameters. AcTTA demonstrates that activation adaptation is a superior alternative or complement.
- **vs. PReLU/ACON/PAU**: These learnable activations are designed for training-time optimization. AcTTA is the first to apply this concept to test-time adaptation, achieving better results.
- **vs. DomainBed and domain generalization methods**: Domain generalization addresses domain shift at training time, while TTA addresses it at inference time — AcTTA expands the inference-time toolkit.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ An entirely new perspective on TTA that challenges the field's default assumptions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Seven backbones, three datasets, six baselines, detailed ablations, and analyses of learning rate, depth, and batch size.
- Writing Quality: ⭐⭐⭐⭐ Theoretical analysis is clear and visualizations are intuitive.
- Value: ⭐⭐⭐⭐⭐ Paradigm-shifting impact on the TTA field — activation adaptation is poised to become a new standard tool.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting](redepth_anything_test-time_depth_refinement_via_self-supervised_re-lighting.md)
- [\[CVPR 2026\] Chain-of-Models Pre-Training: Rethinking Training Acceleration of Vision Foundation Models](com_pt_chain_of_models_pretraining.md)
- [\[CVPR 2026\] A Stitch in Time: Learning Procedural Workflow via Self-Supervised Plackett-Luce Ranking](a_stitch_in_time_learning_procedural_workflow_via_self_supervised_plackett_luce_r.md)
- [\[CVPR 2026\] BD-Merging: Bias-Aware Dynamic Model Merging with Evidence-Guided Contrastive Learning](bd-merging_bias-aware_dynamic_model_merging_with_evidence-guided_contrastive_lea.md)
- [\[ICLR 2026\] SNAP-UQ: Self-supervised Next-Activation Prediction for Single-Pass Uncertainty](../../ICLR2026/self_supervised/snap-uq_self-supervised_next-activation_prediction_for_single-pass_uncertainty_i.md)

</div>

<!-- RELATED:END -->
