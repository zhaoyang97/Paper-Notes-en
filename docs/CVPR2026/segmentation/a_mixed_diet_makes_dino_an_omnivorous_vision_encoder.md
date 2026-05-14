---
title: >-
  [Paper Note] A Mixed Diet Makes DINO An Omnivorous Vision Encoder
description: >-
  [CVPR2026][Segmentation][Cross-modal alignment] This paper proposes an Omnivorous Vision Encoder that performs cross-modal alignment distillation training (RGB/Depth/Segmentation) on top of a frozen DINOv2 via lightweigh…
tags:
  - "CVPR2026"
  - "Segmentation"
  - "Cross-modal alignment"
  - "DINOv2"
  - "vision foundation model"
  - "modality-agnostic encoder"
  - "parameter-efficient fine-tuning"
  - "contrastive learning"
date: 2026-05-08
content_hash: f928dcfe8848c9d9
---

# A Mixed Diet Makes DINO An Omnivorous Vision Encoder

**Conference**: CVPR2026
**arXiv**: [2602.24181](https://arxiv.org/abs/2602.24181)
**Code**: To be confirmed
**Area**: Semantic Segmentation
**Keywords**: Cross-modal alignment, DINOv2, vision foundation model, modality-agnostic encoder, parameter-efficient fine-tuning, contrastive learning

## TL;DR

This paper proposes an Omnivorous Vision Encoder that performs cross-modal alignment distillation training (RGB/Depth/Segmentation) on top of a frozen DINOv2 via lightweight adapters, enabling a single encoder to produce consistent embeddings across different visual modalities while preserving the original discriminative semantics.

## Background & Motivation

**Severe cross-modal feature misalignment**: Experiments show that the cosine similarity between DINOv2 features of an RGB image and a depth map of the same scene is nearly identical to that between two unrelated RGB images, indicating that cross-modal representations in existing vision encoders are highly fragmented.

**Inspiration from NLP**: Natural language processing evolved from language-specific models to multilingual shared representations (e.g., mBERT), substantially improving low-resource language performance. The vision domain faces a similar inflection point, requiring alignment of RGB (data-rich) with depth/segmentation (data-scarce but structurally rich) into a unified space.

**Naïve alignment leads to representational collapse**: Simply maximizing cross-modal similarity can compress the feature space into a trivial solution, destroying the encoder's discriminative ability. Existing methods such as CMC rely on large numbers of negative samples, which are difficult to collect under modality imbalance.

**High cost of full fine-tuning**: Methods such as Omnivore and ImageBind require joint training of the entire backbone from scratch, which is prohibitively expensive. Industry practitioners require a way to extend strong unimodal models (e.g., DINOv2) with cross-modal capabilities at minimal cost.

**Standard colormaps introduce shortcut learning**: Grayscale or jet colormaps applied to depth/segmentation maps allow models to achieve alignment through low-level color statistics rather than structural content.

**Discrete modality training lacks robustness**: Treating RGB/Depth/Seg as discrete states during training makes it difficult for the model to learn invariances across a continuous cross-modal spectrum, resulting in brittleness when inputs are ambiguous or modalities are mixed.

## Method

### Overall Architecture

A parameter-efficient teacher–student framework is adopted:

- **Teacher**: Fully frozen DINOv2 ($f_T = g^* \circ f^*$), providing stable representational anchors.
- **Student**: Shares the frozen early 8 layers of the backbone $f^*$; only the last 4 layers serve as a trainable adapter $g$, yielding $f_S = g \circ f^*$.
- Given an input of arbitrary modality $x_m$, the frozen layers extract $z_m = f^*(x_m)$, and the adapter maps it to a unified space as $h = g(z_m)$.

### Key Designs: Data Processing

1. **Natural Colorization**: RGB pixel values are quantized into 64 bins, and the resulting color palette is used to colorize depth/segmentation maps so that they visually resemble RGB images. This constructs "hard positive pairs" that force the network to align based on structural content rather than color histograms.
2. **Modality Mixup**: $x_s^{mixup} = (1-\alpha_s)x_s + \alpha_s x_r^{aug}$, blending RGB with depth/segmentation at a random ratio ($\alpha \in [0, 0.5]$) to train across the continuous Depth ↔ RGB ↔ Seg spectrum, enhancing robustness to modality ambiguity.
3. **Standard Photometric Augmentation**: Brightness, contrast, hue, and saturation perturbations are applied to RGB inputs.

### Loss & Training

**Symmetric cross-modal alignment loss**: InfoNCE is computed over all modality pairs $(m_1, m_2)$:

$$\mathcal{L}_{\text{align}} = \frac{1}{3}\sum_{k_1}\sum_{k_2>k_1} \mathcal{L}_{\text{InfoNCE}}(m_{k_1}, m_{k_2})$$

Three pairs are used: (RGB_aug, Seg_mixup), (Seg_mixup, Depth_mixup), and (Depth_mixup, RGB_aug), with a learnable temperature $\tau$.

**Anchor loss**: Prevents representational drift by constraining the student output $h_m$ to remain close to the teacher output $h_m^*$:

$$\mathcal{L}_{\text{anchor}} = \frac{1}{|M|}\sum_{m \in M}(1 - \text{sim}(h_m, h_m^*))$$

**Total objective**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{align}} + \lambda_{\text{anchor}} \mathcal{L}_{\text{anchor}}$, with default $\lambda_{\text{anchor}}=10$. Losses are computed separately for the CLS token and dense tokens (64 randomly sampled); dense tokens from the same image are masked out as negatives.

## Key Experimental Results

### Cross-Modal Retrieval (Table 1)

| Dataset | Model | R@1 ↑ | R@5 ↑ | mAP ↑ | MedR ↓ |
|--------|------|-------|-------|-------|--------|
| MOVi (GAP) | DINOv2 ViT-B/14 | 15.5 | 33.1 | 25.2 | 19.3 |
| MOVi (GAP) | **Omnivorous** | **86.2** | **96.5** | **90.9** | **1.0** |
| ScanNet (GAP) | DINOv2 ViT-B/14 | 4.6 | 10.8 | 8.1 | 401.8 |
| ScanNet (GAP) | **Omnivorous** | **46.1** | **71.4** | **57.7** | **2.0** |
| TartanAir (GAP) | DINOv2 ViT-B/14 | 46.6 | 68.5 | 57.1 | 1.8 |
| TartanAir (GAP) | **Omnivorous** | **90.6** | **99.2** | **94.6** | **1.0** |

On ScanNet, R@1 improves from 4.6% to 46.1% and MedR drops from 401.8 to 2.0, demonstrating substantial gains in cross-modal alignment.

### Downstream Tasks (Tables 2 & 3)

| Task | Dataset | Readout | DINOv2 | Omnivorous |
|------|--------|---------|--------|------------|
| Depth δ₁ ↑ | NYUv2 | Linear | 0.875 | **0.896** |
| Depth RMSE ↓ | NYUv2 | Linear | 0.405 | **0.377** |
| Seg. mIoU ↑ | ADE20k | Linear | 0.463 | **0.475** |
| Seg. mIoU ↑ | Cityscapes | Linear | 0.622 | **0.632** |
| Cls. Acc ↑ | ImageNet | Linear (TOK&GAP) | 0.804 | **0.838** |

ImageNet linear classification improves from 80.4% to 83.8%, indicating that cross-modal alignment increases the semantic density of the learned features.

### Ablation Study

- **Pareto frontier of $\lambda_{\text{anchor}}$**: $\lambda=1$ yields better alignment but reduced discriminability; $\lambda=100$ preserves discriminability but limits alignment; $\lambda=10$ achieves the best trade-off.
- **Modality Mixup $\alpha_{\max}$ ablation**: From $\alpha_{\max}=0$ to $1.0$, classification, segmentation, and 3D metrics improve consistently with increasing $\alpha$, while depth estimation degrades slightly beyond $\alpha>0.5$; the default $\alpha_{\max}=0.5$ provides a balanced trade-off.

### Key Findings

- **Zero-shot cross-modal transfer (Table 5)**: A depth prediction head trained on RGB inputs is directly evaluated with segmentation map inputs at test time. DINOv2 achieves RMSE = 1.536 (near-random), whereas Omnivorous achieves RMSE = 0.532. On the never-seen NOCS modality, Omnivorous also substantially outperforms the baseline (0.822 vs. 0.979 DPT).
- **k-NN classification does not degrade**: ImageNet k-NN accuracy is maintained at 81.97% (DINOv2: 81.94%), confirming that the anchor loss effectively prevents representational forgetting.
- **PCA visualization**: Frozen DINOv2 features for RGB/Depth/Seg occupy disjoint subspaces, whereas Omnivorous features are aligned to a consistent color distribution.

## Highlights & Insights

- **Minimalist efficiency**: Only the last 4 layers of the ViT (~33% of parameters) are fine-tuned, enabling cross-modal alignment on top of a frozen foundation model with low training overhead.
- **Elegant data augmentation strategy**: The combination of natural colorization and modality mixup constructs hard positive pairs to prevent shortcut learning while continuous-spectrum training improves robustness.
- **Strong zero-shot cross-modal transfer**: Task heads trained on RGB can be directly applied to segmentation maps and even the unseen NOCS modality.
- **Order-of-magnitude improvement in cross-modal retrieval**: ScanNet MedR drops from 401.8 to 2.0.
- **Downstream performance improves across the board**: Classification, segmentation, and depth prediction all surpass DINOv2, demonstrating that cross-modal regularization itself yields generalization gains.

## Limitations & Future Work

- DINOv2 includes a high-resolution fine-tuning stage; whether Omnivorous requires an analogous step remains unverified.
- The training data contains a large proportion of synthetic multi-object scenes, leading to slight k-NN degradation on fine-grained datasets such as iNaturalist and GLDv2.
- Experiments are limited to the ViT-B/14 scale; the effects on larger models (ViT-L/G) and the choice of adapter depth remain unexplored.
- Only RGB, depth, and segmentation modalities are covered; modalities such as text and thermal infrared are not addressed.

## Related Work & Insights

- **Unified multimodal encoders**: Omnivore processes images, video, and 3D with a single ViT but requires full joint training; ImageBind binds six modalities but similarly trains from scratch.
- **RGB–Depth alignment**: CLIP2Point performs image–depth contrastive pretraining; CoMAE combines contrastive learning with masked autoencoding; Mask3D injects 3D priors via masked RGB-D pretraining.
- **Parameter-efficient adaptation**: ViT-Adapter injects task priors; MA-AVT performs patch-level audio–visual contrastive alignment; modality-decoupled adapters separate modality-invariant and modality-specific components.
- **Cross-modal distillation**: SOCKET performs source-free cross-modal transfer; CMKD applies decoupled and contrastive terms for RGB-D segmentation distillation.
- **Distinguishing contribution**: This work performs **post-hoc lightweight alignment** on top of a frozen unimodal foundation model, balancing deployment convenience with cross-modal capability.

## Rating

- Novelty: ⭐⭐⭐⭐ — The data strategy of colorization + modality mixup and the idea of achieving cross-modal alignment by fine-tuning only the last few layers are concise and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Retrieval, classification, segmentation, depth estimation, zero-shot transfer, and ablation studies are comprehensively covered across 6 datasets.
- Writing Quality: ⭐⭐⭐⭐ — The NLP multilingual analogy provides an effective entry point; structure is clear and figures are intuitive.
- Value: ⭐⭐⭐⭐ — Provides a low-cost cross-modal upgrade path for already-deployed vision foundation models, with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Empowering DINO Representations for Underwater Instance Segmentation via Aligner and Prompter](../../AAAI2026/segmentation/empowering_dino_representations_for_underwater_instance_segmentation_via_aligner.md)
- [\[CVPR 2026\] GKD: Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](gkd_generalizable_knowledge_distillation_vfm.md)
- [\[CVPR 2026\] MPM: Mutual Pair Merging for Efficient Vision Transformers](mpm_mutual_pair_merging_for_efficient_vision_transformers.md)
- [\[ICLR 2026\] Locality-Attending Vision Transformer](../../ICLR2026/segmentation/locality-attending_vision_transformer.md)
- [\[ICLR 2026\] Revisiting \[CLS\] and Patch Token Interaction in Vision Transformers](../../ICLR2026/segmentation/revisiting_cls_and_patch_token_interaction_in_vision_transformers.md)

</div>

<!-- RELATED:END -->
