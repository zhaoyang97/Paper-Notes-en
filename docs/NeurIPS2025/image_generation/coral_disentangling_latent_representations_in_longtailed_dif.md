---
title: >-
  [Paper Note] CORAL: Disentangling Latent Representations in Long-Tailed Diffusion
description: >-
  [NeurIPS 2025][Image Generation][diffusion model] This paper diagnoses the root cause of tail-class generation degradation in diffusion models trained on long-tailed data as *representation entanglement* in the U-Net bot…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "diffusion model"
  - "long-tailed distribution"
  - "contrastive learning"
  - "latent space"
  - "representation entanglement"
date: 2026-05-08
content_hash: b5cbcaa8272c96cc
---

# CORAL: Disentangling Latent Representations in Long-Tailed Diffusion

**Conference**: NeurIPS 2025  
**arXiv**: [2506.15933](https://arxiv.org/abs/2506.15933)  
**Code**: [https://github.com/SankarLab/coral-lt-diffusion](https://github.com/SankarLab/coral-lt-diffusion)  
**Area**: Image Generation / Long-Tailed Learning  
**Keywords**: diffusion model, long-tailed distribution, contrastive learning, latent space, representation entanglement

## TL;DR

This paper diagnoses the root cause of tail-class generation degradation in diffusion models trained on long-tailed data as *representation entanglement* in the U-Net bottleneck layer, and proposes CORAL, which applies a supervised contrastive loss at the bottleneck to disentangle class representations. CORAL consistently outperforms baselines including DDPM, CBDM, and T2H on CIFAR10/100-LT, CelebA-5, and ImageNet-LT.

## Background & Motivation

- **Background**: Diffusion models excel on class-balanced data, but real-world data commonly follows long-tailed distributions—a small number of head classes dominate the sample count while many tail classes are severely underrepresented. Under such settings, diffusion models suffer significant generation quality degradation on tail classes, manifesting as low diversity and *feature borrowing*.

- **Limitations of Prior Work**: Existing improvements primarily operate at the data level (e.g., CBDM's balanced sampling regularization) or in the output space (e.g., DiffROP's output distribution contrastive loss), overlooking the root cause—the internal latent representation structure of the denoising network. T2H employs Bayesian gating to transfer knowledge between head and tail classes but still does not directly intervene in the class structure of the latent space.

- **Core Idea**: Through t-SNE visualizations and related analyses, the authors find that under long-tailed training, tail-class latent representations in the U-Net bottleneck layer severely overlap with those of head classes (representation entanglement). This is not merely a consequence of data scarcity; rather, the relative class imbalance causes head classes to dominate parameter updates, depriving tail classes of structured latent representations. CORAL addresses class entanglement at its source by directly imposing supervised contrastive regularization in the latent space.

## Method

### Overall Architecture

CORAL introduces two modifications to the standard DDPM training pipeline: (1) a lightweight projection head appended after the U-Net bottleneck output, mapping bottleneck features into a contrastive embedding space; and (2) a time-dependent supervised contrastive loss term added to the training objective. The projection head is discarded after training, incurring zero additional computational cost at inference and remaining fully compatible with standard DDPM sampling.

### Key Designs

1. **Projection Head**:

    - **Function**: Maps U-Net bottleneck features into the contrastive embedding space.
    - **Mechanism**: A lightweight MLP consisting of a linear layer followed by a normalization layer is appended after the bottleneck output. Contrastive constraints are applied to the projected embeddings rather than directly to the bottleneck features.
    - **Design Motivation**: Following best practices in contrastive learning, the projection head decouples the contrastive objective from the primary generative features, preventing the contrastive loss from directly compressing intra-class diversity in the bottleneck. The bottleneck layer is where semantic information is most concentrated and where representation entanglement primarily occurs.

2. **Time-Dependent Contrastive Loss Weighting**:

    - **Function**: Dynamically modulates the contrastive loss weight across different noise levels.
    - **Mechanism**: The weighting function $\lambda(t) = w \cdot \exp(\frac{1-t/T}{\tau_r})$ assigns higher contrastive weight at low-noise timesteps ($t\approx 0$, where semantic structure is recoverable) and reduces the contrastive constraint at high-noise timesteps ($t\approx T$, where noise dominates).
    - **Design Motivation**: The amount of semantic information varies across timesteps in the diffusion process. Semantic structure is clearer at early steps, making contrastive constraints more effective there; imposing strong contrastive constraints under high noise levels may instead disrupt training.

3. **Latent-Space vs. Ambient-Space Regularization**:

    - **Function**: Intervenes in class structure at the source of the generative process rather than at the output.
    - **Mechanism**: The contrastive loss acts directly on the internal bottleneck representations of the U-Net, rather than imposing posterior constraints on generated images as in DiffROP.
    - **Design Motivation**: Ambient-space regularization constrains outputs that have already been generated, at which point entanglement has already occurred. Latent-space intervention separates class representations at the precise location where overlap arises, addressing the problem at its root. Ablation experiments confirm that the latent-space approach consistently outperforms the ambient-space approach.

### Loss & Training

The overall training objective is $\mathcal{L}_{\text{CORAL}} = \mathcal{L}_{\text{diff}} + \lambda(t) \cdot \mathcal{L}_{\text{con}}$, where $\mathcal{L}_{\text{diff}}$ is the standard noise prediction loss and $\mathcal{L}_{\text{con}}$ is the SupCon loss, which pulls together projected embeddings of the same class and pushes apart those of different classes within each mini-batch. The temperature parameter $\tau_{\text{SC}}$ controls the sharpness of the distribution. During training, classifier-free guidance (CFG) is supported by randomly dropping class labels with probability $p_{\text{uncond}}$ for unconditional training; the contrastive loss always uses ground-truth labels. At inference, the projection head is discarded and standard CFG sampling is applied.

## Key Experimental Results

### Main Results

| Dataset | Metric | CORAL (Ours) | DDPM | CBDM | T2H |
|--------|------|------|------|------|------|
| CIFAR10-LT (ρ=0.01) | FID↓ | **5.32** | 6.17 | 5.62 | 7.01 |
| CIFAR10-LT (ρ=0.01) | IS↑ | **9.69** | 9.43 | 9.28 | 9.63 |
| CIFAR10-LT (ρ=0.01) | Recall↑ | **0.59** | 0.52 | 0.57 | 0.54 |
| CIFAR10-LT (ρ=0.001) | FID↓ | **11.03** | 13.05 | 12.74 | 12.80 |
| CIFAR100-LT (ρ=0.01) | FID↓ | **5.37** | 7.70 | 6.02 | 6.78 |
| CIFAR100-LT (ρ=0.01) | IS↑ | **13.53** | 13.20 | 12.92 | 12.97 |
| CelebA-5 | FID↓ | **8.12** | 10.28 | 8.74 | 9.50 |
| ImageNet-LT (1000 classes) | FID↓ | **16.11** | 17.08 | 22.66 | 18.59 |
| ImageNet-LT (1000 classes) | IS↑ | **24.17** | 21.03 | 17.13 | 19.15 |
| ImageNet-LT (1000 classes) | Recall↑ | **0.48** | 0.39 | 0.42 | 0.44 |

### Ablation Study

| Configuration | Key Metric | Notes |
|------|---------|------|
| Latent-space regularization (CORAL) vs. ambient-space regularization | CORAL consistently superior | Validates latent-space intervention over output-space constraints |
| Varying SupCon temperature τ_SC | Range [0.1, 0.5] performs well | See appendix ablation figures |
| Varying decay temperature τ_r | Range [0.5, 1.0] optimal | Controls decay rate of time-dependent weights |
| Varying CFG weight ω | Consistent with standard DDPM trends | CORAL does not alter the optimal CFG configuration |

### Key Findings
- CORAL consistently outperforms all baselines across all datasets and metrics, with the largest gains on diversity/coverage metrics (Recall, F8).
- The advantage of CORAL is most pronounced on ImageNet-LT (1000 classes)—ambient-space methods (e.g., CBDM FID=22.66) degrade significantly with larger numbers of classes, while latent-space intervention remains stable.
- Per-class FID analysis shows that CORAL's improvements are most substantial on tail classes, with no degradation on head classes.
- t-SNE visualizations clearly demonstrate the class separation achieved in the bottleneck layer after CORAL training.

## Highlights & Insights
- **Deep root-cause diagnosis**: Rather than merely observing that tail-class generation is poor, the paper pinpoints representation entanglement in the U-Net bottleneck and demonstrates—through comparisons between balanced and imbalanced datasets—that this stems from class imbalance rather than data scarcity alone.
- **Elegant and simple design**: The projection head and contrastive loss are plug-and-play during training and incur zero overhead at inference, making the approach fully compatible with existing diffusion training frameworks.
- **Physically motivated time-dependent weighting**: Contrastive constraints are applied more strongly at low-noise steps where semantic structure is clear and relaxed at high-noise steps—consistent with the information-theoretic properties of the diffusion process.

## Limitations & Future Work
- The method requires training-phase intervention and cannot be applied as a test-time fix or post-processing step.
- Experiments are limited to $32\times32$ and $64\times64$ resolutions; scalability to higher resolutions (e.g., $256\times256$) and large models such as Stable Diffusion remains unverified.
- Contrastive loss hyperparameters may require tuning across different datasets.
- Only class-conditional generation is evaluated; long-tailed generation in text-conditional (text-to-image) settings is an important direction for future work.

## Related Work & Insights
- **vs. CBDM**: CBDM addresses imbalance at the data sampling level, whereas CORAL imposes structured separation at the feature space level.
- **vs. DiffROP**: DiffROP applies KL-based contrastive constraints on the output distribution, while CORAL applies SupCon at the internal bottleneck layer—intervening closer to the root of the problem.
- **Potential extensions**: CORAL could be combined with LoRA fine-tuning for rare concept preservation in domain adaptation of large-scale text-to-image models.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The root-cause diagnosis of representation entanglement and the latent-space intervention approach are novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive comparisons across 4 datasets, with ablations, visualizations, and per-class analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — The logical chain from problem diagnosis to method design to experimental validation is clear.
- **Value**: ⭐⭐⭐⭐ — Long-tailed generation is ubiquitous in practical settings; the method is concise, practical, and extensible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Diffusion Generative Modeling on Lie Group Representations](diffusion_generative_modeling_on_lie_group_representations.md)
- [\[NeurIPS 2025\] Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge](towards_general_modality_translation_with_contrastive_and_predictive_latent_diff.md)
- [\[ICCV 2025\] Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video](../../ICCV2025/image_generation/bitrate-controlled_diffusion_for_disentangling_motion_and_content_in_video.md)
- [\[NeurIPS 2025\] Vicinity-Guided Discriminative Latent Diffusion for Privacy-Preserving Domain Adaptation](vicinity-guided_discriminative_latent_diffusion_for_privacy-preserving_domain_ad.md)
- [\[NeurIPS 2025\] A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models](a_gradient_flow_approach_to_solving_inverse_problems_with_latent_diffusion_model.md)

</div>

<!-- RELATED:END -->
