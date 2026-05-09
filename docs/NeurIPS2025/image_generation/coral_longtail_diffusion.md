---
title: >-
  [Paper Note] CORAL: Disentangling Latent Representations in Long-Tailed Diffusion
description: >-
  [NeurIPS 2025][Image Generation][Diffusion Models] This paper identifies a phenomenon termed "representation entanglement" in diffusion models trained on long-tailed data, wherein the latent representations at the U-Net bottleneck layer exhibit severe overlap between tail and head class feature spaces. To address this, the authors propose CORAL, which introduces a projection head and a supervised contrastive loss at the bottleneck layer to promote inter-class latent separation, substantially improving the generation quality and diversity of tail classes.
tags:
  - NeurIPS 2025
  - Image Generation
  - Diffusion Models
  - Long-Tailed Distribution
  - Contrastive Learning
  - Latent Space Disentanglement
  - U-Net Bottleneck Layer
date: 2026-05-08
content_hash: 2a61654a9d406b56
---

# CORAL: Disentangling Latent Representations in Long-Tailed Diffusion

**Conference**: NeurIPS 2025
**arXiv**: [2506.15933](https://arxiv.org/abs/2506.15933)
**Code**: [GitHub](https://github.com/SankarLab/coral-lt-diffusion)
**Area**: Image Generation
**Keywords**: Diffusion Models, Long-Tailed Distribution, Contrastive Learning, Latent Space Disentanglement, U-Net Bottleneck Layer

## TL;DR
This paper identifies a phenomenon termed "representation entanglement" in diffusion models trained on long-tailed data, wherein the latent representations at the U-Net bottleneck layer exhibit severe overlap between tail and head class feature spaces. To address this, the authors propose CORAL, which introduces a projection head and a supervised contrastive loss at the bottleneck layer to promote inter-class latent separation, substantially improving the generation quality and diversity of tail classes.

## Background & Motivation

1. **State of the Field**: Diffusion models perform well on class-balanced data, but real-world data frequently follows long-tailed distributions.

2. **Limitations of Prior Work**: Under long-tailed distributions, diffusion models produce poor-quality and low-diversity outputs for tail classes, exhibiting a "feature borrowing" problem in which tail-class samples manifest head-class features.

3. **Root Cause**: Existing methods (CBDM, T2H, DiffROP) primarily operate in image space or external latent spaces, and do not address the intra-network class entanglement occurring within the denoising network's internal latent space.

4. **Paper Goals**: To identify and resolve representation entanglement within the internal latent space of diffusion models.

5. **Starting Point**: The output of the U-Net bottleneck layer carries rich semantic information and is the critical site at which representation entanglement arises.

6. **Core Idea**: A lightweight projection head is appended to the U-Net bottleneck layer, and a supervised contrastive loss is applied to encourage class separation—intervening directly at the location where representation entanglement occurs.

## Method

### Overall Architecture
Building upon the standard DDPM + CFG training pipeline, CORAL introduces two components: (1) a projection head MLP following the bottleneck layer, and (2) a supervised contrastive loss applied to the projected embeddings. The total loss is $\mathcal{L}_{CORAL} = \mathcal{L}_{diff} + \lambda(t) \cdot \mathcal{L}_{con}$. The projection head consists of a single fully connected layer with normalization and is discarded after training, incurring zero additional inference overhead. The method is evaluated on five settings: CIFAR10-LT ($\rho=0.01$ and $0.001$), CIFAR100-LT ($\rho=0.01$), CelebA-5 ($64\times64$), and ImageNet-LT ($64\times64$, 1000 classes).

### Key Designs

1. **Projection Head Design**:
    - Function: Decouples the contrastive objective from the diffusion features.
    - Mechanism: A single fully connected layer with normalization, $f_\phi$, is appended to the U-Net encoder bottleneck output.
    - Design Motivation: The projection head prevents the contrastive loss from directly collapsing the intra-class diversity of the bottleneck layer; it is discarded after training, adding zero inference overhead.

2. **Time-Dependent Weighting Function**:
    - Function: Dynamically modulates the influence of the contrastive loss across denoising timesteps.
    - Mechanism: $\lambda(t) = w \cdot \exp(\frac{1-t/T}{\tau_r})$, assigning greater weight at early timesteps (low noise, $t \approx 0$).
    - Design Motivation: Semantic structure is more recoverable at low-noise stages; at high-noise stages, noise dominates and class separation cannot be effectively learned.

3. **Supervised Contrastive Loss (SupCon)**:
    - Function: Encourages intra-class clustering and inter-class separation.
    - Mechanism: SupCon loss is applied to the projected bottleneck features $\mathbf{z}$ using original (unmasked) class labels.
    - Design Motivation: Imposing class separation constraints directly at the bottleneck layer where entanglement occurs is more targeted than image-space approaches.

### Loss & Training
- Base: DDPM noise prediction loss + CFG (labels dropped with probability $p_{uncond}$).
- Additional: SupCon loss with $\tau_{SC} \in [0.5, 1.0]$.
- Inference is entirely standard; the projection head is not involved.

## Key Experimental Results

### Main Results

| Dataset | Method | FID ↓ | IS ↑ | Recall ↑ |
|--------|------|-------|------|----------|
| CIFAR10-LT ($\rho=0.01$) | DDPM | 6.17 | 9.43 | 0.52 |
| CIFAR10-LT ($\rho=0.01$) | CBDM | 5.62 | 9.28 | 0.57 |
| CIFAR10-LT ($\rho=0.01$) | **CORAL** | **5.32** | **9.69** | **0.59** |
| CIFAR100-LT ($\rho=0.01$) | DDPM | 7.70 | 13.20 | 0.50 |
| CIFAR100-LT ($\rho=0.01$) | **CORAL** | **5.37** | **13.53** | **0.59** |
| ImageNet-LT | DDPM | 17.08 | 21.03 | 0.39 |
| ImageNet-LT | **CORAL** | **16.11** | **24.17** | **0.48** |

### Key Findings
- Representation entanglement is the root cause of long-tailed diffusion failure, rather than mere data scarcity.
- t-SNE visualizations clearly demonstrate class separation before and after applying CORAL.
- Per-class FID analysis shows that CORAL yields the most significant improvements on tail classes.
- Latent-space intervention outperforms image-space intervention (DiffROP-style).
- The advantage is most pronounced on ImageNet-LT (1000 classes): FID 16.11 vs. DDPM 17.08, IS 24.17 vs. 21.03, Recall 0.48 vs. 0.39, demonstrating scalability.
- CORAL is also effective on CelebA-5 ($64\times64$): FID 8.12 vs. DDPM 10.28, Recall 0.59 vs. 0.52.
- CBDM's FID degrades on ImageNet-LT (22.66 vs. DDPM 17.08), as image-space regularization fails with large numbers of classes; CORAL's latent-space intervention is unaffected by class count.
- Qualitative analysis of generated samples reveals that CBDM exhibits mode collapse for the tulip class (producing small flowers with excessive grassy backgrounds borrowed from head animal classes), T2H generates tulips resembling other flower classes, while CORAL preserves correct scale and structure.

## Highlights & Insights
- This work is the first to identify and formally name the phenomenon of "representation entanglement" in diffusion models.
- The projection head elegantly exploits an information bottleneck effect to protect intra-class diversity in the bottleneck layer.
- Zero additional inference overhead; fully compatible with standard diffusion sampling.
- The method is concise yet highly effective, yielding a strong cost-to-performance ratio.
- The time-dependent weighting function $\lambda(t) = w \cdot \exp(\frac{1-t/T}{\tau_r})$ assigns greater weight at low-noise stages ($t \approx 0$) where semantic structure is more recoverable, as noise dominates at high-noise stages and class separation cannot be effectively learned.

## Limitations & Future Work
- Validation is limited to U-Net architectures; extension to newer architectures such as DiT has not been explored.
- Applicability to more complex scenarios such as text-to-image generation remains unverified.
- Integration with parameter-efficient methods such as LoRA fine-tuning warrants further exploration.
- The contrastive temperature parameter requires tuning.
- Experiments are conducted at limited resolutions ($32\times32$, $64\times64$); performance at high resolutions remains to be verified.
- The impact of projection head architecture choices (depth, dimensionality) on performance requires more thorough analysis.
- Visualization experiments on balanced datasets confirm that representation entanglement primarily stems from class imbalance rather than data scarcity, providing empirical support for CORAL's design motivation.
- Future work may explore applying CORAL to LoRA fine-tuning of pretrained diffusion models to prevent rare concepts from being entangled with common ones in specialized domains such as medical imaging and scientific visualization.

## Related Work & Insights
- **vs. CBDM**: CBDM regularizes via balanced sampling in image space, whereas CORAL directly disentangles representations in latent space. CBDM's FID degrades to 22.66 on ImageNet-LT (DDPM: 17.08), while CORAL achieves 16.11.
- **vs. T2H**: T2H employs a Bayesian gating mechanism to transfer knowledge from head to tail classes; CORAL is more direct and yields more consistent performance (ImageNet-LT IS: CORAL 24.17 vs. T2H 19.15).
- **vs. DiffROP**: DiffROP applies KL-based contrastive regularization in image space, whereas CORAL applies SupCon at the bottleneck layer, resulting in a more targeted intervention.

## Rating

### Implementation Details
Built upon DDPM + CFG training; projection head consists of a single FC layer with normalization, discarded after training.
SupCon temperature $\tau_{SC} \in [0.5, 1.0]$; weighting function temperature $\tau_r$ controls time dependency.
Evaluated on CIFAR10/100-LT, CelebA-5, and ImageNet-LT ($64\times64$).
- Novelty: ⭐⭐⭐⭐ The identification of representation entanglement and the latent-space intervention are original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets, multiple metrics, and complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Analysis is clear and visualizations are rich.
- Value: ⭐⭐⭐⭐ Practically valuable for long-tailed generation and diffusion model training.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge](towards_general_modality_translation_with_contrastive_and_predictive_latent_diff.md)
- [\[NeurIPS 2025\] Diffusion Models Meet Contextual Bandits](diffusion_models_meet_contextual_bandits.md)
- [\[NeurIPS 2025\] Token Perturbation Guidance for Diffusion Models](token_perturbation_guidance_for_diffusion_models.md)
- [\[NeurIPS 2025\] When Are Concepts Erased From Diffusion Models?](when_are_concepts_erased_from_diffusion_models.md)
- [\[NeurIPS 2025\] Entropy Rectifying Guidance for Diffusion and Flow Models](entropy_rectifying_guidance_for_diffusion_and_flow_models.md)

<!-- RELATED:END -->
