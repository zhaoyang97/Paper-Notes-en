---
title: >-
  [Paper Note] Boosting Adversarial Transferability with Spatial Adversarial Alignment
description: >-
  [NEURIPS2025][AI Safety][adversarial transferability] This paper proposes Spatial Adversarial Alignment (SAA), which fine-tunes a surrogate model via two modules—spatial-aware alignment and adversarial-aware alignment—to align its features with those of a witness model, achieving significant improvements in cross-architecture adversarial transferability (CNN→ViT transfer rate improved by 25–39%).
tags:
  - "NEURIPS2025"
  - "AI Safety"
  - "adversarial transferability"
  - "model alignment"
  - "cross-architecture attack"
  - "spatial features"
  - "adversarial features"
date: 2026-05-08
content_hash: 791fbcb1b83beb05
---

# Boosting Adversarial Transferability with Spatial Adversarial Alignment

**Conference**: NEURIPS2025
**arXiv**: [2501.01015](https://arxiv.org/abs/2501.01015)  
**Code**: To be confirmed  
**Area**: AI Security
**Keywords**: adversarial transferability, model alignment, cross-architecture attack, spatial features, adversarial features

## TL;DR

This paper proposes Spatial Adversarial Alignment (SAA), which fine-tunes a surrogate model via two modules—spatial-aware alignment and adversarial-aware alignment—to align its features with those of a witness model, achieving significant improvements in cross-architecture adversarial transferability (CNN→ViT transfer rate improved by 25–39%).

## Background & Motivation

Adversarial transferability is central to black-box attacks: without access to the target model's parameters or architecture, attackers must generate adversarial examples using a surrogate model and rely on transferability to attack unknown targets. Existing methods for improving transferability include advanced optimization (MI, NI, etc.), data augmentation (DI, TI, SSA, etc.), and model modification (SGM, LinBP, etc.), yet performance in cross-architecture settings (e.g., CNN→ViT) remains limited.

The existing Model Alignment (MA) approach aligns only the final prediction logits of models, suffering from two critical limitations:

1. **Spatial features are not aligned**: Intermediate-layer features of CNNs and ViTs differ substantially in semantic granularity and spatial structure; constraining only the final output is insufficient to align intermediate representations.
2. **Adversarial features are neglected**: Adversarial examples exhibit a feature distribution distinct from that of clean inputs; cross-model similarity of adversarial features is equally important, yet MA ignores this aspect.

## Core Problem

How can a surrogate model be trained to capture spatial and adversarial features shared with models of different architectures, so that generated adversarial perturbations transfer effectively across diverse architectures such as CNNs and ViTs?

## Method

### Overall Architecture

SAA fine-tunes a surrogate model using a witness model through two core modules:

### 1. Spatial-aware Alignment

**Global alignment**: Minimizes the KL divergence between the final outputs (logits) of the surrogate and witness models:

$$\mathcal{L}_{global}(x;\theta_s) = D_{KL}(f_{\theta_s}(x), f_{\theta_w}(x))$$

**Local alignment**: Both the CNN's last convolutional feature map and the ViT's patch token embeddings are reshaped to $(B, C, H, W)$ and aligned position-by-position. The surrogate model's local features are supervised by local pseudo-labels from the witness model:

$$\mathcal{L}_{local}(x;\theta_s) = \frac{1}{HW}\sum_{q=1}^{HW} D_{CE}(z_{\theta_s}^{[q]}(x), \hat{y}_{\theta_w}^{[q]})$$

Total spatial alignment loss: $\mathcal{L}_{SA} = \mathcal{L}_{global} + \gamma \cdot \mathcal{L}_{local}$, where $\gamma=0.2$.

### 2. Adversarial-aware Alignment

A self-adversarial strategy is introduced: adversarial examples are iteratively generated using the surrogate model's gradients to maximize the divergence between the surrogate and witness model's global features:

$$x_{adv}^{(t+1)} = \Pi_{x,\epsilon}(x_{adv}^{(t)} + \alpha \cdot \text{sign}(\nabla_x D_{KL}(f_{\theta_s}(x_{adv}^{(t)}), f_{\theta_w}(x))))$$

Global and local alignment are then applied on these adversarial examples:

$$\mathcal{L}_{AA}(x_{adv};\theta_s) = \mathcal{L}_{global}(x_{adv};\theta_s) + \omega \cdot \mathcal{L}_{local}(x_{adv};\theta_s)$$

### 3. Overall Optimization Objective

$$\mathcal{L}_{SAA} = \mathcal{L}_{SA}(x;\theta_s) + \kappa \cdot \mathcal{L}_{AA}(x_{adv};\theta_s)$$

Hyperparameter settings: $\gamma=0.2$, $\omega=0.02$, $\kappa=0.02$. Fine-tuning is performed for only 1 epoch on the original training data using SGD (momentum=0.9).

## Key Experimental Results

Evaluation is conducted on the ImageNet-compatible dataset with 6 CNN target models (Res18/50/101, VGG19, DN121, Inc-v3) and 4 ViT target models (ViT-B, Swin-B, PVT-v2, MobViT).

### Comparison with MA (Surrogate: Res50, MI Attack)

| Witness Model | MA Avg ASR | SAA Avg ASR | SAA Gain on ViTs |
|---------|-----------|------------|-----------------|
| Res50   | 45.8%     | 58.8%      | +39.1%          |
| DN121   | 63.9%     | 75.8%      | +31.3%          |
| ViT-B   | 53.5%     | 63.9%      | +25.5%          |
| Swin-B  | 44.4%     | 57.5%      | +37.7%          |

### SAA Combined with Existing Attack Methods (Surrogate: Res50, Witness: ViT-B)

| Attack Method | Original Avg ASR | +SAA Avg ASR | Gain |
|---------|-------------|-------------|-----|
| MI      | 42.2%       | 63.9%       | +21.7% |
| DI-MI   | 55.4%       | 78.5%       | +23.1% |
| SSA-MI  | 78.5%       | 85.1%       | +6.6%  |

### Feature Similarity Verification

Global feature cosine similarity between surrogate Res50 and witness ViT-B: 0.0533 (before alignment) → 0.1408 (after alignment) on clean images, a 164% improvement.

## Highlights & Insights

1. **First to reveal the importance of spatial and adversarial features for cross-architecture transferability**, offering a new perspective on model alignment.
2. **Elegant local alignment design**: CNN feature maps and ViT patch embeddings are uniformly projected to the same spatial scale for position-wise alignment, bridging the architectural gap.
3. **Plug-and-play**: SAA requires only 1 epoch of surrogate fine-tuning and integrates seamlessly with mainstream transfer attacks such as MI, DI, TI, and SSA.
4. **Substantial cross-architecture gains**: CNN→ViT transfer rates improve by 25–39%, far exceeding the logits-only MA approach.

## Limitations & Future Work

1. **Sensitivity to witness model selection**: Different witness models lead to considerably different transfer outcomes; the paper offers empirical guidance but lacks theoretical guarantees.
2. **Classification tasks only**: Effectiveness on downstream tasks such as detection and segmentation has not been verified.
3. **Fine-tuning overhead**: Although only 1 epoch is required, fine-tuning on ImageNet-scale datasets still incurs non-trivial computational cost.
4. **Absence of defense evaluation**: Experiments are conducted only on standard models, without evaluation against adversarially trained models or defense methods.
5. **Scale sensitivity of local alignment**: The sensitivity of the spatial scale choice ($H \times W$) across different architecture combinations is not thoroughly analyzed.

## Related Work & Insights

| Method Category | Representative Methods | Distinction from SAA |
|---------|---------|-------------|
| Optimization-based | MI, NI, VMI | Optimize gradients only; do not modify the model; SAA fine-tunes the surrogate itself |
| Augmentation-based | DI, TI, SSA | Input transformations to reduce overfitting; orthogonal to SAA and combinable |
| Model modification | SGM, LinBP | Adjust gradient propagation paths; do not involve cross-model alignment |
| Alignment-based | MA | Global logits alignment only; SAA additionally introduces local spatial and adversarial feature alignment |

**Broader Implications:**
- The spatial alignment idea can transfer to knowledge distillation: position-wise feature alignment is more effective than aligning only the final output.
- The self-adversarial strategy (using adversarial examples during alignment) also has implications for adversarial robustness research: incorporating adversarial features during training may improve model generalization.
- Cross-architecture feature alignment methods may be applicable to model fusion, federated learning, and other scenarios requiring cooperation across heterogeneous architectures.

## Rating
- **Novelty**: 4/5 (The combination of local spatial alignment and adversarial-aware alignment constitutes a novel contribution)
- **Experimental Thoroughness**: 4/5 (10 target models covering CNNs and ViTs; stacked with multiple attack methods; complete ablation study)
- **Writing Quality**: 4/5 (Clear structure with thorough visual analysis)
- **Value**: 4/5 (Meaningful advancement in adversarial transferability research with significant cross-architecture gains)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Transform to Transfer: Boosting Adversarial Attack Transferability on Vision-Language Pre-training Models](../../CVPR2026/ai_safety/transform_to_transfer_boosting_adversarial_attack_transferability_on_vision-lang.md)
- [\[CVPR 2026\] Improving Adversarial Transferability with Local Perturbation Augmentation](../../CVPR2026/ai_safety/improving_adversarial_transferability_with_local_perturbation_augmentation.md)
- [\[ICCV 2025\] Towards Adversarial Robustness via Debiased High-Confidence Logit Alignment](../../ICCV2025/ai_safety/towards_adversarial_robustness_via_debiased_high-confidence_logit_alignment.md)
- [\[CVPR 2026\] Generative Adversarial Perturbations with Cross-paradigm Transferability on Localized Crowd Counting](../../CVPR2026/ai_safety/generative_adversarial_perturbations_with_cross-paradigm_transferability_on_loca.md)
- [\[NeurIPS 2025\] Distributional Adversarial Attacks and Training in Deep Hedging](distributional_adversarial_attacks_and_training_in_deep_hedging.md)

</div>

<!-- RELATED:END -->
