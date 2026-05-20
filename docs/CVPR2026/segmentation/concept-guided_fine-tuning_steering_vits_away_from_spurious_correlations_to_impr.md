---
title: >-
  [Paper Note] Concept-Guided Fine-Tuning: Steering ViTs away from Spurious Correlations to Improve Robustness
description: >-
  [CVPR2026][Segmentation][Spurious Correlations] This paper proposes CFT (Concept-Guided Fine-Tuning), which leverages LLM-generated class-level semantic concepts and zero-shot segmentation via GroundedSAM to obtain conce…
tags:
  - "CVPR2026"
  - "Segmentation"
  - "Spurious Correlations"
  - "ViT Robustness"
  - "Concept-Guided Fine-Tuning"
  - "AttnLRP"
  - "Zero-Shot Segmentation"
  - "OOD Generalization"
date: 2026-05-08
content_hash: 0fb77e0176abac5a
---

# Concept-Guided Fine-Tuning: Steering ViTs away from Spurious Correlations to Improve Robustness

**Conference**: CVPR2026
**arXiv**: [2603.08309](https://arxiv.org/abs/2603.08309)  
**Code**: [GitHub](https://github.com/yonisGit/cft)  
**Area**: Image Segmentation
**Keywords**: Spurious Correlations, ViT Robustness, Concept-Guided Fine-Tuning, AttnLRP, Zero-Shot Segmentation, OOD Generalization

## TL;DR

This paper proposes CFT (Concept-Guided Fine-Tuning), which leverages LLM-generated class-level semantic concepts and zero-shot segmentation via GroundedSAM to obtain concept masks. ViTs are then fine-tuned by aligning AttnLRP relevance maps with concept regions. Using only 1,500 training images, CFT achieves substantial robustness improvements across 5 OOD benchmarks.

## Background & Motivation

1. **ViTs rely on spurious correlations**: Large-scale pretrained ViTs achieve strong in-distribution performance but tend to exploit spurious features such as background textures and co-occurring objects rather than genuine semantic parts, leading to performance collapse under OOD conditions.
2. **Foreground–background dichotomy is too coarse**: Existing approaches (e.g., LANCE, RBF) regularize attention via foreground/background segmentation, but "foreground" subsumes many non-discriminative regions (e.g., a bird's belly vs. its beak), failing to guide the model toward truly class-discriminative features.
3. **Manual concept annotation is not scalable**: Defining semantic concepts and annotating segmentation masks per class is prohibitively expensive, especially given ImageNet's 1,000 categories.
4. **LLM + zero-shot segmentation provides a scalable alternative**: LLMs can automatically generate discriminative concept descriptions per class, and GroundedSAM can perform zero-shot segmentation for arbitrary text prompts—together enabling concept-level supervision without manual annotation.
5. **Relevance maps provide interpretable attention signals**: AttnLRP yields pixel-level relevance attribution that quantifies the model's attention to each region, offering a differentiable supervision signal for alignment optimization.

## Core Problem

How can ViTs be guided—without manual annotation—to shift attention from spurious regions (backgrounds, textures) toward genuine class-level semantic concept regions, thereby improving OOD robustness?

## Method

### Overall Architecture

CFT proceeds in three stages: concept generation → concept segmentation → concept-guided fine-tuning. The entire pipeline requires no additional manual annotation and uses only a small number of training images (3 per class, 1,500 in total).

### Stage 1: LLM-Based Concept Generation

- For each ImageNet class $c$, an LLM (e.g., GPT-4) generates a list of discriminative visual concepts.
- For example, for the class "bird," generated concepts include "long beak," "wings," "talons," and "feathers."
- Prompt design requires the LLM to produce concepts that are **visually distinguishable parts or attributes**, rather than abstract semantics.
- $K$ concepts are generated per class to cover the major discriminative features.

### Stage 2: Zero-Shot Concept Segmentation via GroundedSAM

- For each training image $I$, LLM-generated concept texts are used as prompts for GroundedSAM.
- GroundedSAM produces binary segmentation masks $S_k(I)$ for each concept in a zero-shot manner.
- The union of all concept masks yields the **concept region mask** $S(I) = \bigcup_{k=1}^K S_k(I)$.
- The complement $\bar{S}(I) = 1 - S(I)$ constitutes the **non-concept region**, encompassing backgrounds and non-discriminative foreground parts.

### Stage 3: Concept-Guided Fine-Tuning

AttnLRP is used to compute the relevance map $\Phi(I; \theta)$, where $\Phi_{ij} \in [0, 1]$ denotes the model's attention to pixel $(i,j)$. Three loss terms are jointly optimized:

**Concept alignment loss $\mathcal{L}_{concept}$**: maximizes relevance within concept regions.

$$\mathcal{L}_{concept} = -\frac{1}{|S(I)|} \sum_{(i,j) \in S(I)} \log \Phi_{ij}(I; \theta)$$

**Non-concept suppression loss $\mathcal{L}_{non\text{-}concept}$**: minimizes relevance in non-concept regions.

$$\mathcal{L}_{non\text{-}concept} = -\frac{1}{|\bar{S}(I)|} \sum_{(i,j) \in \bar{S}(I)} \log(1 - \Phi_{ij}(I; \theta))$$

**Classification self-consistency loss $\mathcal{L}_{cls}$**: standard cross-entropy loss to prevent degradation of classification performance during fine-tuning.

$$\mathcal{L}_{total} = \mathcal{L}_{cls} + \alpha \mathcal{L}_{concept} + \beta \mathcal{L}_{non\text{-}concept}$$

### Training Details

- Training data: 3 randomly sampled images per class from the ImageNet training set, totaling 500 × 3 = 1,500 images.
- Fine-tuning runs for 50 epochs, updating only the last few layers of the ViT.
- AttnLRP relevance maps are computed on-the-fly during each forward pass, providing pixel-level gradient signals.
- Concept masks are precomputed and cached, introducing no additional overhead during training.

## Key Experimental Results

### Datasets and Setup

- **In-distribution**: ImageNet-1K validation set.
- **OOD benchmarks** (5 total): ImageNet-A (natural adversarial examples), ObjectNet (novel viewpoints/backgrounds), ImageNet-R (artistic renditions), ImageNet-Sketch (sketches), SI-Score (controlled shape/texture synthesis).
- **Backbones**: DINOv2 ViT-B/14, ViT-B/16, DeiT-B/16, ConvNeXt-B.
- **Baselines**: vanilla fine-tuning, LANCE, RBF, StylEx, foreground-only baseline.

### Main Results

| Method | IN-1K↑ | IN-A↑ | ObjectNet↑ | IN-R↑ | IN-Sketch↑ | SI-Score↑ |
|--------|--------|-------|------------|-------|------------|-----------|
| DINOv2 baseline | 84.5 | 70.3 | 53.8 | 72.1 | 52.4 | 61.2 |
| LANCE | 84.2 | 71.5 | 54.6 | 72.8 | 53.1 | 62.0 |
| RBF | 84.0 | 71.8 | 54.2 | 73.0 | 53.5 | 62.4 |
| **CFT (ours)** | **84.8** | **73.6** | **56.4** | **74.5** | **55.2** | **64.1** |

CFT outperforms the baseline and all prior methods on all 5 OOD benchmarks while maintaining or slightly improving in-distribution accuracy. Compared to LANCE/RBF, CFT achieves an average OOD gain of approximately 2–3 percentage points.

### Cross-Architecture Generalization

| Backbone | IN-A Δ | ObjectNet Δ | IN-R Δ |
|----------|--------|-------------|--------|
| DINOv2 ViT-B | +3.3 | +2.6 | +2.4 |
| ViT-B/16 | +2.8 | +2.1 | +1.9 |
| DeiT-B/16 | +2.5 | +1.8 | +1.7 |
| ConvNeXt-B | +1.6 | +1.2 | +1.1 |

CFT is most effective on Transformer architectures; ConvNeXt also benefits but to a lesser extent, likely due to the absence of global attention limiting relevance map precision.

### Relevance Map Alignment Quality

- After fine-tuning, the IoU between relevance maps and ground-truth semantic part masks improves from 0.31 to 0.52.
- Visualizations show that baseline heatmaps are dispersed over background textures, whereas CFT concentrates attention on semantic parts such as beaks, wings, and talons.
- SI-Score analysis indicates that CFT models exhibit increased reliance on shape features and reduced reliance on texture features.

### Ablation Study

- **Removing $\mathcal{L}_{concept}$**: average OOD performance drops by 1.5%; the model cannot be guided to attend to correct regions.
- **Removing $\mathcal{L}_{non\text{-}concept}$**: average OOD performance drops by 1.2%; relevance in spurious regions is not effectively suppressed.
- **Removing $\mathcal{L}_{cls}$**: in-distribution accuracy drops by 2.1%; fine-tuning diverges from the original classification objective.
- **Replacing concept masks with foreground–background masks**: average OOD performance drops by 0.8–1.5%, validating the superiority of concept-level granularity.
- **Number of images per class**: using 1 image/class yields limited OOD gains; 3 images/class reaches saturation; 5 images/class yields no significant additional benefit.
- **Number of LLM concepts**: 3–5 concepts per class is optimal; too many concepts introduce noisy masks.

## Highlights & Insights

- **Concept-level granularity**: The paper moves beyond the foreground/background dichotomy by using LLM-generated semantic concepts (e.g., "beak," "wings") as supervision granularity, enabling more precise specification of regions the model should attend to.
- **Fully automated pipeline**: The LLM concept generation → GroundedSAM segmentation → AttnLRP alignment pipeline is end-to-end annotation-free and scalable to arbitrary numbers of classes.
- **Minimal data requirement**: Fine-tuning on only 1,500 images (3 per class) for 50 epochs incurs extremely low training cost, making it practical for real-world deployment.
- **Preserved in-distribution performance**: The $\mathcal{L}_{cls}$ self-consistency constraint prevents accuracy degradation on ImageNet, resolving the typical robustness–accuracy trade-off.
- **Cross-architecture generality**: CFT is effective across DINOv2, ViT, DeiT, and ConvNeXt without architecture-specific design.
- **Enhanced interpretability**: Alignment visualizations of relevance maps with semantic parts directly demonstrate that the model attends to the correct regions.

## Limitations & Future Work

- The quality of LLM-generated concepts depends on prompt design and model capability; for fine-grained classes (e.g., insect subspecies), generated concepts may lack sufficient discriminability.
- GroundedSAM's zero-shot segmentation accuracy is limited; noisy masks for small or occluded objects may propagate into the fine-tuning process.
- Validation is restricted to image classification; extension to detection and segmentation downstream tasks remains unexplored.
- AttnLRP is applicable only to Transformer architectures; alternative attribution methods are required for ConvNeXt, limiting effectiveness on CNNs.
- The hierarchical structure of concepts (e.g., "bird" → "head" → "beak") is not explored; the current approach uses a flat concept list.
- Only partial-layer fine-tuning is studied; the effect of full fine-tuning or parameter-efficient methods such as LoRA is not discussed.

## Related Work & Insights

- **vs. LANCE**: LANCE aligns attention using foreground masks with language guidance, but foreground granularity is too coarse and includes many non-discriminative regions. CFT uses concept-level masks to precisely localize discriminative parts, yielding larger OOD gains.
- **vs. RBF (Right for the Better Features)**: RBF mitigates spurious correlations via foreground-enhancing/background-suppressing data augmentation—a data-level approach. CFT operates at the model level through relevance alignment; the two are complementary.
- **vs. StylEx**: StylEx diagnoses spurious features by discovering and manipulating style attributes via StyleGAN, serving primarily as an analysis tool. CFT directly remedies spurious dependencies.
- **vs. Concept Bottleneck Models**: CBMs require manually annotated concepts as intermediate representations and are not scalable. CFT replaces manual annotation with LLM generation and GroundedSAM zero-shot segmentation at zero labeling cost.
- **vs. attention regularization methods**: Conventional attention regularization relies on manually defined regions or coarse attribution methods such as CAM/GradCAM. CFT employs AttnLRP to provide more precise pixel-level relevance signals.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of LLM + segmentation + relevance alignment is innovative; concept-level granularity is the core contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 5 OOD benchmarks, 4 architectures, and comprehensive ablations, though downstream task validation is absent.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated, the pipeline diagram is intuitive, and equations are concise.
- Value: ⭐⭐⭐⭐ — A practical, low-cost solution for robustness improvement with direct relevance to OOD deployment scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DDB: Diffusion Driven Balancing to Address Spurious Correlations](../../ICCV2025/segmentation/ddb_diffusion_driven_balancing_to_address_spurious_correlations.md)
- [\[CVPR 2026\] ConceptPrism: Concept Disentanglement in Personalized Diffusion Models via Residual Token Optimization](conceptprism_concept_disentanglement_in_personalized_diffusion_models_via_residu.md)
- [\[NeurIPS 2025\] Fast and Fluent Diffusion Language Models via Convolutional Decoding and Rejective Fine-tuning](../../NeurIPS2025/segmentation/fast_and_fluent_diffusion_language_models_via_convolutional_decoding_and_rejecti.md)
- [\[CVPR 2026\] Seeing Through the Tool: A Controlled Benchmark for Occlusion Robustness in Foundation Segmentation Models](occsam_bench_occlusion_robustness_segmentation.md)
- [\[CVPR 2026\] CA-LoRA: Concept-Aware LoRA for Domain-Aligned Segmentation Dataset Generation](ca-lora_concept-aware_lora_for_domain-aligned_segmentation_dataset_generation.md)

</div>

<!-- RELATED:END -->
