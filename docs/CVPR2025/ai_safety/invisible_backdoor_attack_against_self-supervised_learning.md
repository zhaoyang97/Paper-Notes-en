---
title: >-
  [Paper Note] INACTIVE: Invisible Backdoor Attack against Self-supervised Learning
description: >-
  [CVPR 2025][AI Safety][SSL Backdoor] Proposes INACTIVE, the first invisible backdoor attack effective against self-supervised learning (SSL). By designing triggers in the HSV/HSL color space to escape the distribution space of SSL data augmentations, it achieves a 99.09% average attack success rate while maintaining high stealthiness with SSIM 0.9763 / PSNR 41.07dB, resisting 7 defense methods.
tags:
  - "CVPR 2025"
  - "AI Safety"
  - "SSL Backdoor"
  - "Invisible Trigger"
  - "HSV Color Space"
  - "Data Augmentation Decoupling"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: 87fd99914e9e14e6
---

# INACTIVE: Invisible Backdoor Attack against Self-supervised Learning

**Conference**: CVPR 2025  
**arXiv**: [2405.14672](https://arxiv.org/abs/2405.14672)  
**Code**: [https://github.com/Zhang-Henry/INACTIVE](https://github.com/Zhang-Henry/INACTIVE)  
**Area**: AI Security / Backdoor Attack  
**Keywords**: SSL Backdoor, Invisible Trigger, HSV Color Space, Data Augmentation Decoupling, Contrastive Learning

## TL;DR

Proposes INACTIVE, the first invisible backdoor attack effective against self-supervised learning (SSL). By designing triggers in the HSV/HSL color space to escape the distribution space of SSL data augmentations, it achieves a 99.09% average attack success rate while maintaining high stealthiness with SSIM 0.9763 / PSNR 41.07dB, resisting 7 defense methods.

## Background & Motivation

### Background

**Background**: Backdoor attacks have been widely studied in supervised learning, where attackers inject trigger-containing samples into training data to make the model sensitive to the triggers. Since SSL lacks labels, the mechanism of backdoor attacks is completely different—it requires clustering the trigger images to a specific location in the feature space.

**Limitations of Prior Work**: Existing invisible backdoor attacks (e.g., WaNet, ISSBA) fail in SSL because SSL data augmentations (e.g., ColorJitter, RandomCrop, GaussianBlur) destroy the trigger—the augmented image no longer contains the complete trigger, and the backdoor cannot be established.

**Key Challenge**: Stealthiness requires small trigger perturbations, but the strong augmentations in SSL mask small perturbations. The trigger must reside outside the augmentation space to remain detectable after augmentation.

**Key Insight**: Analyzing the scope of SSL augmentation operations—ColorJitter mainly operates in the RGB space, but certain transformation directions in the HSV/HSL space are outside the scope of ColorJitter. Design triggers in the "uncovered directions" of HSV.

**Core Idea**: Finding the "blind spots" of SSL augmentations in the HSV color space to design triggers = augmentation-invariant invisible backdoors.

## Method

### Key Designs

1. **Augmentation-Decoupled Trigger Design**: By maximizing the distribution distance $\mathcal{L}_{disentangle}$ between trigger transformations and SSL augmentation transformations in the HSV/HSL space, ensuring that the trigger transformation falls outside the augmentation distribution.

2. **Stealthy Constraints**: $\mathcal{L}_{stealthy}$ combines LPIPS + PSNR + SSIM + Wasserstein distance, ensuring that trigger-embedded images are visually indistinguishable from original images.

3. **Feature Alignment**: $\mathcal{L}_{alignment}$ uses cosine similarity to align the SSL features of trigger-embedded images with the reference image.

### Loss & Training

$\mathcal{L}_{total} = \mathcal{L}_{stealthy} + \alpha \mathcal{L}_{disentangle} + \beta \mathcal{L}_{alignment}$. Two-stage training: first pre-train the backdoor injector, then fine-tune the encoder.

## Key Experimental Results

### Main Results

| SSL Method | ASR | SSIM | PSNR |
|------------|-----|------|------|
| SimCLR     | 99.58% | 0.976 | 41.07 |
| MoCo       | 99.76% | Same as above | Same as above |
| BYOL       | 99.09%+ | — | — |
| CLIP       | Effective | — | — |

Resists 7 defense methods (DECREE / Beatrix / ASSET / STRIP / GradCAM / Neural Cleanse / noise variants).

### Key Findings
- The decoupling loss is core—without it, ASR drops significantly.
- Triggers in the HSV space are more robust than in the RGB space.
- All 6 SSL algorithms are vulnerable to attack (SimCLR / MoCo / BYOL / SimSiam / SwAV / CLIP).

## Highlights & Insights
- **First systematic breakthrough in SSL backdoors**—previous invisible backdoors were considered infeasible in SSL.
- **Augmentation space analysis methodology**—analyzing the mathematical scope of augmentation operations to find blind spots, which can be generalized to other security analyses.

## Limitations & Future Work
- Requires knowledge of the target SSL's augmentation strategy.
- Assumes access to a clean pre-trained encoder.
- Downstream task transfer may weaken when there is a large distribution gap.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Fundamental breakthrough in SSL backdoors.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 SSL methods + 7 defenses + 4 datasets.
- Writing Quality: ⭐⭐⭐⭐ Clear attack methodology.
- Value: ⭐⭐⭐⭐⭐ Of significant warning value to SSL security research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Backdooring Self-Supervised Contrastive Learning by Noisy Alignment](../../ICCV2025/ai_safety/backdooring_self-supervised_contrastive_learning_by_noisy_alignment.md)
- [\[CVPR 2025\] Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning](infighting_in_the_dark_multi-label_backdoor_attack_in_federated_learning.md)
- [\[ICML 2025\] Adversarial Inception Backdoor Attacks against Reinforcement Learning](../../ICML2025/ai_safety/adversarial_inception_backdoor_attacks_against_reinforcement_learning.md)
- [\[CVPR 2025\] Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection](detecting_backdoor_attacks_in_federated_learning_via_direction_alignment_inspect.md)
- [\[CVPR 2025\] MOS-Attack: A Scalable Multi-Objective Adversarial Attack Framework](mos-attack_a_scalable_multi-objective_adversarial_attack_framework.md)

</div>

<!-- RELATED:END -->
