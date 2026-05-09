---
title: >-
  [Paper Note] AIM: Amending Inherent Interpretability via Self-Supervised Masking
description: >-
  [ICCV 2025][self-supervised masking] This paper proposes AIM, a top-down learnable binary masking mechanism for self-supervised spatial feature selection, built upon a feature pyramid architecture. Without requiring additional annotations, AIM guides CNNs to focus on genuinely discriminative features and suppress spurious correlations, simultaneously achieving inherent interpretability and improved OOD generalization.
tags:
  - ICCV 2025
  - self-supervised masking
  - inherent interpretability
  - spurious features
  - feature pyramid
  - Energy Pointing Game
date: 2026-05-08
content_hash: 92ec1b0ae71a1ea9
---

# AIM: Amending Inherent Interpretability via Self-Supervised Masking

**Conference**: ICCV 2025
**arXiv**: [2508.11502](https://arxiv.org/abs/2508.11502)
**Code**: No public code available
**Area**: Self-Supervised Learning / Interpretability / Robust Representation Learning
**Keywords**: self-supervised masking, inherent interpretability, spurious features, feature pyramid, Energy Pointing Game

## TL;DR

This paper proposes AIM, a top-down learnable binary masking mechanism for self-supervised spatial feature selection, built upon a feature pyramid architecture. Without requiring additional annotations, AIM guides CNNs to focus on genuinely discriminative features and suppress spurious correlations, simultaneously achieving inherent interpretability and improved OOD generalization.

## Background & Motivation

**Background**: Deep neural networks achieve strong performance on classification tasks, yet frequently exploit spurious features for decision-making — for instance, on the WaterBirds dataset, models learn to classify birds based on background context (water vs. land) rather than the birds themselves, leading to severe performance degradation under distribution shift.

**Limitations of Prior Work**: Ensuring that models make correct predictions for the right reasons typically requires additional supervision such as bounding boxes, segmentation masks, or attention guidance maps — all of which are costly to obtain and potentially imperfect. The few annotation-free approaches either require iterative expert model selection or yield limited improvements.

**Key Challenge**: DNNs simultaneously learn genuine and spurious features during training (a key finding from Kirichenko et al. 2022), yet a lightweight mechanism that enables models to autonomously distinguish and prioritize genuine features without external supervision has been lacking.

**Goal**
- Given only image-level class labels, how can a model automatically discover and focus on spatially discriminative features?
- How can the model's decision process become inherently interpretable, rather than relying on post-hoc attribution methods?

**Key Insight**: The central hypothesis is that when a model is constrained to retain only a subset of spatial features prior to classification, it will preferentially select the most reliable — i.e., genuinely discriminative — features. This contrasts with bottom-up layer-wise masking, which tends to produce fully active masks and requires additional sparsity regularization losses.

**Core Idea**: A top-down learnable binary masking mechanism (inspired by the FPN architecture) performs self-supervised spatial feature selection across multi-scale feature maps, naturally producing sparse masks that filter out spurious features.

## Method

### Overall Architecture

AIM augments a standard CNN backbone (e.g., ConvNeXt-Tiny, ResNet-50) with a top-down pathway, forming a dual-pathway architecture analogous to FPN:
- **Bottom-up pathway**: Standard multi-stage backbone encoding (e.g., 4 stages $S_0$–$S_3$ in ConvNeXt), producing hierarchical features from coarse to fine.
- **Top-down pathway**: A mirrored structure $T_0$–$T_3$, where each stage contains two parallel branches — a feature processing branch and a mask estimation branch.
- The binary mask produced by the mask estimator is element-wise multiplied with the processed features to form sparse feature representations.
- Sparse features from each stage are progressively upsampled and fused from higher to lower levels, and the final representation is used for classification.

The model takes an image as input and outputs a classification prediction along with a visualizable mask indicating which spatial regions the model attends to.

### Key Designs

1. **Mask Estimator**:

    - *Function*: Predicts a spatial binary mask $B_\ell \in \{0,1\}^{w_\ell \times h_\ell}$ for the feature map at each top-down stage.
    - *Mechanism*: A lightweight CNN (3×3 convolution + 3 residual blocks + global pooling branch + 1×1 convolution) generates a soft attention map $A_\ell = M(S_\ell(x_\ell))$, which is binarized via the Gumbel-Softmax technique as $B_\ell = G(A_\ell)$; the sparse output is then $O_\ell = T_\ell(S_\ell(x_\ell)) \odot B_\ell$.
    - *Design Motivation*: Compared to the bottom-up masking strategy (Verelst & Tuytelaars 2020), the top-down approach allows the network to re-evaluate features at each layer under the guidance of global semantic information, naturally generating sparse and focused masks without additional regularization to prevent fully active masks.

2. **Top-Down Sparse Feature Fusion**:

    - *Function*: Progressively merges sparse features produced at each stage along the top-down direction.
    - *Mechanism*: The output $O_\ell$ at each stage is upsampled via nearest-neighbor interpolation to the resolution of the next stage, then fused by element-wise summation.
    - A hyperparameter controls the termination depth of the top-down pathway — for example, fusing only the last 2–3 stages — denoted as AIM[stage, threshold].

3. **Self-Supervised Mask Supervision Strategy**:

    - *Function*: Supervises mask estimator learning indirectly through the classification loss.
    - *Mechanism*: Before fusion at each top-down stage, an independent classifier $f_\ell$ computes a classification loss $\mathcal{L}_{cls}^{(\ell)}$ on the sparse features $O_\ell$, ensuring that each stage independently learns to identify discriminative regions.
    - This eliminates the need for additional annotations — mask learning is driven entirely by the classification objective.

### Loss & Training

**Mask Annealing**: For OOD datasets (e.g., WaterBirds, TravelingBirds), an additional sparsity constraint is applied. A threshold $\tau_i$ controls the proportion of active mask values:

$$\mathcal{L}_{masks_i} = (r_i - \tau_i)^2, \quad r_i = \frac{\sum \mathbb{1}(B^i_{j,k}=1)}{\text{total pixels}}$$

At the start of training, $\tau_i = 1.0$ (fully active); it is linearly annealed per epoch to a target value (e.g., 0.35 or 0.25) and held constant thereafter.

Total loss: $\mathcal{L}_{Total} = \lambda \sum_i \mathcal{L}_{masks_i} + \sum_i \mathcal{L}_{cls}^{(i)}$, where $\lambda = 6$.

## Key Experimental Results

### Main Results

On OOD benchmarks such as WaterBirds and TravelingBirds, AIM yields substantial gains over the baseline:

| Model | WB-100% WG-Acc | WB-100% EPG | WB-95% WG-Acc | WB-95% EPG | TravelBirds Acc | TravelBirds EPG |
|------|--------|------|--------|------|--------|------|
| ConvNeXt-t | 39.6% | 57.2% | 81.6% | 68.3% | 59.5% | 74.4% |
| +AIM[2,25%] | 74.0% | 58.0% | 92.7% | 75.0% | 77.4% | 85.0% |
| +AIM[2,35%] | 78.1% | 68.5% | 92.3% | 71.7% | 71.0% | 77.7% |

Worst-group accuracy on WaterBirds-100% improves by approximately **40 percentage points**, and TravelingBirds accuracy improves by approximately **18 percentage points**.

AIM also proves effective on general classification tasks:

| Model | ImageNet100 Acc | ImageNet100 EPG | HardImageNet Acc | CUB-200 EPG Gain |
|------|--------|------|--------|------|
| ConvNeXt-t | 89.2% | 91.4% | 96.2% | baseline |
| +AIM[2,25%] | 90.1% | 92.8% | 96.8% | +6% |

### Ablation Study

| Configuration | CUB-200 Acc | Notes |
|------|---------|------|
| Bottom-up masking [1,25%] | 72.79% (±8.51) | Single-stage bottom-up masking; poor performance and high variance |
| ConvNeXt-t+AIM [1,25%] | 88.82% (±0.21) | Top-down approach; substantial improvement |
| Bottom-up masking [2,25%] | 84.00% (±1.38) | Multi-stage bottom-up slightly better, still inferior to top-down |
| ConvNeXt-t+AIM [2,25%] | 88.68% (±0.25) | Top-down: stable and significantly superior |

Computational overhead: AIM adds only approximately 0.1–1.0 GFLOPs and 1.9–3.7M parameters relative to the backbone (4.5 GFLOPs / 28M parameters), representing a modest increase.

### Key Findings

- **Top-down vs. bottom-up is the critical distinction**: Bottom-up masking produces masks that tend toward full activation, yielding poor and unstable performance; top-down masking naturally produces sparse masks.
- **Mask annealing benefits OOD settings**: On standard datasets, masks naturally become sparse without annealing; on datasets with strong spurious correlations, the annealing strategy provides an additional sparsity constraint.
- **No center bias**: When bird images are cropped to peripheral positions, AIM still correctly localizes the target (+2.5% over baseline), confirming the absence of center bias.
- **Cross-architecture consistency**: Effective across ConvNeXt-Tiny, ResNet-50, and ResNet-101.

## Highlights & Insights

- **Causal interpretability via direct participation in inference**: The mask directly participates in the forward pass and classification decision; the visualized mask directly reflects the basis for classification, rather than providing an approximation as post-hoc attribution methods (e.g., GradCAM) do. This renders the explanation more trustworthy than post-hoc approaches.
- **Annotation-free feature selection through self-supervision**: The inductive bias of "constraining the model to a subset of spatial features → compelling it to select the most reliable ones" elegantly unifies interpretability and generalization without external supervision.
- **A novel application of FPN architecture**: Originally designed for multi-scale feature fusion in object detection, FPN is here repurposed as an interpretability guidance tool, demonstrating a new value of top-down multi-scale information flow.

## Limitations & Future Work

- **Evaluated only on CNNs**: Validation is limited to ConvNeXt and ResNet; extension to Vision Transformers has not been realized. The authors mention plans to extend to ViT/Swin-Transformer in the Future Work section, but this remains unimplemented.
- **Mask annealing requires manual hyperparameter tuning**: The threshold $\tau$ and annealing schedule must be adjusted per dataset, increasing the tuning burden.
- **Narrow evaluation datasets**: Validation is concentrated on bird-related datasets (WaterBirds, TravelingBirds, CUB-200); OOD evaluation across broader domains (e.g., medical imaging, remote sensing) is absent.
- **Potential future direction**: Combining the masking mechanism with contrastive learning to guide feature selection during self-supervised pre-training, rather than restricting its use to supervised classification.

## Related Work & Insights

- **vs. MaskTune (Asgari et al. 2022)**: MaskTune masks discriminative features during fine-tuning to discover additional features, but requires a fully trained model prior to fine-tuning. AIM learns masks concurrently with training in a more end-to-end fashion.
- **vs. Content-adaptive downsampling (Hesse et al. 2023)**: Also employs Gumbel-Softmax binary masks, but adopts a bottom-up strategy for layer-wise masking. AIM's top-down strategy is naturally sparser and requires no additional sparsity loss.
- **vs. Post-hoc attribution methods (GradCAM, etc.)**: GradCAM provides post-hoc explanations without modifying model behavior; AIM's masks directly influence forward inference, providing causal-level explanations.
- AIM is particularly valuable in scenarios requiring pre-deployment verification that a model is "making decisions for the right reasons," and can serve as a foundation for model auditing tools.

## Rating

- Novelty: ⭐⭐⭐⭐ The top-down self-supervised masking idea is concise and effective, though the combination of masking and FPN is not an entirely novel concept.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset, multi-architecture validation is comprehensive; ablation studies are detailed and include a user study.
- Writing Quality: ⭐⭐⭐⭐ Logical structure is clear and figures are intuitive.
- Value: ⭐⭐⭐⭐ High practical value for interpretability and OOD generalization, though the restriction to CNNs limits generality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/interpretability/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[NeurIPS 2025\] Dataset Distillation for Pre-Trained Self-Supervised Vision Models](../../NeurIPS2025/interpretability/dataset_distillation_for_pre-trained_self-supervised_vision_models.md)
- [\[ICCV 2025\] ArgoTweak: Towards Self-Updating HD Maps through Structured Priors](argotweak_towards_self-updating_hd_maps_through_structured_priors.md)
- [\[CVPR 2026\] RiskProp: Collision-Anchored Self-Supervised Risk Propagation for Early Accident Anticipation](../../CVPR2026/interpretability/riskprop_collision-anchored_self-supervised_risk_propagation_for_early_accident_.md)
- [\[NeurIPS 2025\] Latent Principle Discovery for Language Model Self-Improvement](../../NeurIPS2025/interpretability/latent_principle_discovery_for_language_model_self-improvement.md)

</div>

<!-- RELATED:END -->
