---
title: >-
  [Paper Note] Cross Modal Fine-Grained Alignment via Granularity-Aware and Region-Uncertain Modeling
description: >-
  [AAAI2026][Robotics][fine-grained alignment] This paper proposes GRM, a framework that achieves robust fine-grained image-text alignment through intra-modal saliency/granularity-aware adapters and Gaussian mixture-based…
tags:
  - "AAAI2026"
  - "Robotics"
  - "fine-grained alignment"
  - "image-text retrieval"
  - "uncertainty modeling"
  - "Gaussian mixture"
  - "region prompting"
date: 2026-05-08
content_hash: a1aff1379dbbab63
---

# Cross Modal Fine-Grained Alignment via Granularity-Aware and Region-Uncertain Modeling

**Conference**: AAAI2026
**arXiv**: [2511.07710](https://arxiv.org/abs/2511.07710)
**Code**: [GitHub](https://github.com/H3IIoWorld/GRM)
**Area**: Robotics
**Keywords**: fine-grained alignment, image-text retrieval, uncertainty modeling, Gaussian mixture, region prompting

## TL;DR

This paper proposes GRM, a framework that achieves robust fine-grained image-text alignment through intra-modal saliency/granularity-aware adapters and Gaussian mixture-based region-level uncertainty modeling, attaining state-of-the-art performance on Flickr30K and MS-COCO.

## Background & Motivation

Fine-grained image-text alignment is a core task in multimodal learning, requiring precise correspondence between local visual regions and text tokens, and directly supporting downstream applications such as VQA, image captioning, and vision-language navigation. Unlike global alignment, fine-grained alignment demands compositional reasoning over object attributes, spatial relations, and local entities.

Existing methods suffer from two critical bottlenecks:

1. **Lack of effective intra-modal saliency modeling**: Most methods rely on cross-modal attention to identify key tokens, but attention weights are driven by retrieval objectives—often noisy and lacking semantic grounding—and tend to focus on visually salient but semantically irrelevant regions, resulting in poor generalization in complex scenes.
2. **Lack of fine-grained uncertainty modeling**: Existing uncertainty modeling operates at the image-text pair level, assuming one-to-one correspondence. In practice, however, a text phrase may correspond to multiple regions (one-to-many), and a region may ambiguously match multiple tokens (many-to-one); region-level uncertainty remains largely unexplored.

## Core Problem

- How can the importance of tokens within each modality be effectively modeled without relying on fragile cross-modal attention?
- How can region-level fine-grained uncertainty be modeled during alignment to capture one-to-many and many-to-one correspondences?

## Method

GRM adopts a dual-encoder architecture (ViT/Swin + BERT) comprising three core modules:

### 1. Significance-aware and Granularity-aware Adapter

The two adapters share the same structure but are independently instantiated, operating on the visual and textual modalities respectively. Taking the visual modality as an example:

- The visual representation $\mathbf{V} \in \mathbb{R}^{L_v \times d}$ is mapped to a 2-dimensional space via two linear transformations.
- A soft selection mask $\mathbf{A_V} \in [0,1]^{L_v}$ is generated using Gumbel-Softmax, with temperature parameter $\tau$ controlling distribution sharpness.
- Salient tokens are filtered via element-wise multiplication: $\hat{\mathbf{V}} = \mathbf{M} \odot \mathbf{A_V} \otimes \mathbf{1}_d$.

**Core idea**: Saliency modeling should be performed within each modality, leveraging its intrinsic statistical biases rather than relying on cross-modal interaction, thereby improving generalization.

### 2. Region Prompting

Learnable prompts $\mathbf{P} = \{p_0, \dots, p_{K-1}\} \in \mathbb{R}^{K \times d}$ are introduced as semantic proxies for latent regions:

- After L2-normalizing $\mathbf{P}$, attention scores between patch tokens and region prompts are computed as $\mathbf{A}_r = \sigma(\hat{\mathbf{V}} \cdot \hat{\mathbf{P}}^\top)$, using sigmoid since a patch may belong to multiple regions simultaneously.
- The attention matrix is column-normalized and soft aggregation is applied to obtain the mean representation of each region: $\boldsymbol{\mu}_k = \sum_l \hat{\mathbf{A}}_r^{lk} \hat{\mathbf{V}}^l$.

### 3. Region-level Uncertainty Modeling

A variational perspective is adopted, modeling the semantics of each region as a Gaussian distribution:

- A learnable network $\boldsymbol{\phi}$ predicts the log-variance $\log \boldsymbol{\sigma}_k^2$ from the mean $\boldsymbol{\mu}_k$.
- Samples are drawn via the reparameterization trick: $\mathbf{z}_{lk} = \boldsymbol{\mu}_k + \boldsymbol{\epsilon}_{lk} \odot \exp(\frac{1}{2} \log \boldsymbol{\sigma}_k^2)$, where $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$.
- Sampled features are aggregated with attention weights to obtain the uncertainty-aware region representation: $\mathbf{u}_k = \sum_l \hat{\mathbf{A}}_{lk} \cdot \mathbf{z}_{lk}$.

The entire image is modeled as a mixture of regional Gaussian distributions, capturing fine-grained semantic ambiguity.

### 4. Multi-level Bidirectional Alignment and Loss Functions

Bidirectional token-level similarity is computed and contrastive loss is applied to three feature pairs:

- $\mathcal{L}_{con}^{ori}$: original feature pair $(\mathbf{T}, \mathbf{V})$
- $\mathcal{L}_{con}^{key}$: saliency/granularity-aware feature pair $(\hat{\mathbf{T}}, \hat{\mathbf{V}})$
- $\mathcal{L}_{con}^{unc}$: uncertainty-aware feature pair $(\hat{\mathbf{T}}, \mathbf{U})$

Total contrastive loss: $\mathcal{L}_{con} = a\mathcal{L}_{con}^{ori} + b\mathcal{L}_{con}^{key} + c\mathcal{L}_{con}^{unc}$, with optimal weights $a=b=0.4, c=0.2$.

Auxiliary regularization includes: semantic consistency constraint $\mathcal{L}_{recon}$ (aligning region means with patch means), KL divergence regularization $\mathcal{L}_{KL}$ (approximating the posterior toward a standard normal), and entropy regularization $\mathcal{L}_{ent}$ (preventing attention collapse).

## Key Experimental Results

Comprehensive evaluation on Flickr30K and MS-COCO covering six visual encoder configurations:

| Config | Flickr30K rSum | MS-COCO 1K rSum | MS-COCO 5K rSum |
|--------|---------------|----------------|----------------|
| ViT-B-224 (Ours) | **516.2** | **532.5** | **443.0** |
| ViT-B-384 (Ours) | **531.8** | **538.2** | **451.2** |
| Swin-B-224 (Ours) | **546.0** | **547.5** | **470.8** |
| Swin-B-384 (Ours) | **550.7** | **548.0** | **478.3** |

- Compared to SOTA method AVSE, rSum improvements range: Flickr30K +2.1%~+7.3%, MS-COCO 1K +1.3%~+4.0%, MS-COCO 5K +1.9%~+5.6%.
- Ablation study: removing any single module leads to significant performance degradation; SA and RP contribute the most (rSum drops 13.4 and 12.9 respectively upon removal).
- Region prompt count: optimal $K=5$ for ViT, $K=50$ for Swin (Swin's local attention requires more prompts to capture fine-grained semantics).

## Highlights & Insights

- **Intra-modal modeling over cross-modal attention**: Gumbel-Softmax-based intra-modal saliency modeling avoids the noise and brittleness of cross-modal attention, yielding stronger generalization.
- **Region-level Gaussian mixture uncertainty**: This is the first work to introduce region-level uncertainty modeling in fine-grained image-text alignment, capturing one-to-many and many-to-one relationships via Gaussian mixture distributions.
- **End-to-end, detector-free**: Region extraction is achieved through prompt learning without requiring a pretrained object detector, avoiding error propagation inherent in two-stage approaches.
- **Multi-level alignment strategy**: The three-level alignment (original / saliency-aware / uncertainty-aware) provides complementary signals, with ablations confirming independent contributions at each level.
- **Consistent gains across backbone architectures**: Stable improvements over SOTA are observed across ViT and Swin at multiple resolutions.

## Limitations & Future Work

- Evaluation is limited to Flickr30K and MS-COCO; validation on more direct fine-grained tasks such as phrase grounding and referring expression comprehension is absent.
- The Gaussian distribution assumption may be overly simplistic and unable to capture complex multimodal semantic distributions; normalizing flows or more flexible distribution families could be explored.
- The optimal number of region prompts is highly sensitive to backbone architecture (ViT: $K=5$ vs. Swin: $K=50$), and an adaptive mechanism is lacking.
- The three-level contrastive loss weights ($a, b, c$) require manual tuning, and the model is sensitive to their combination.
- Integration with large-scale pretrained models such as CLIP has not been explored; fine-grained adaptation on top of pretrained features remains a promising direction.

## Related Work & Insights

| Method | Region Extraction | Uncertainty | Fine-grained Level |
|--------|------------------|-------------|-------------------|
| CORA/HREM | Faster R-CNN (two-stage) | None | Region-text |
| LAPS | ViT patch + cross-modal attention | None | Patch-token |
| AVSE | ViT patch + modality adaptation | None | Patch-token |
| GRM (Ours) | ViT patch + prompt learning | Region-level Gaussian mixture | Multi-level (original / saliency / uncertainty) |

The core advantage of GRM lies in decoupling saliency modeling from cross-modal interaction to intra-modal computation, and in being the first to introduce region-level uncertainty, while maintaining end-to-end optimizability.

The Gumbel-Softmax token selection approach is extensible to other scenarios requiring soft selection (e.g., dynamic token pruning in multimodal fusion). Region-level uncertainty modeling may inspire confidence estimation for candidate boxes in grounding tasks. The intra-modal saliency modeling paradigm may generalize to other cross-modal tasks such as video-text alignment, avoiding costly cross-modal attention. The multi-level alignment framework is extensible: token-level uncertainty, syntactic structure alignment, and other semantic layers could be incorporated.

## Rating
- **Novelty**: 7/10 (the combination of intra-modal saliency and region-level uncertainty is novel, though each individual technique is not entirely new)
- **Experimental Thoroughness**: 8/10 (multiple backbones, multiple datasets, and detailed ablations; validation on more direct fine-grained tasks such as grounding is missing)
- **Writing Quality**: 7/10 (clear structure and complete derivations, though some passages are slightly verbose)
- **Value**: 7/10 (substantive improvements for fine-grained image-text alignment, but application scope is limited to retrieval tasks)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FineCog-Nav: Integrating Fine-grained Cognitive Modules for Zero-shot Multimodal UAV Navigation](../../CVPR2026/robotics/finecog_nav_fine_grained_cognitive_modules_for_zero_shot_uav_navigation.md)
- [\[CVPR 2026\] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](../../CVPR2026/robotics/gecosrt_geometryaware_continual_adaptation_for_rob.md)
- [\[NeurIPS 2025\] FALCON: Fine-grained Activation Manipulation by Contrastive Orthogonal Unalignment for Large Language Model](../../NeurIPS2025/robotics/falcon_fine-grained_activation_manipulation_by_contrastive_orthogonal_unalignmen.md)
- [\[AAAI 2026\] Dexterous Manipulation Transfer via Progressive Kinematic-Dynamic Alignment](dexterous_manipulation_transfer_via_progressive_kinematic-dynamic_alignment.md)
- [\[AAAI 2026\] More Than Irrational: Modeling Belief-Biased Agents](more_than_irrational_modeling_belief-biased_agents.md)

</div>

<!-- RELATED:END -->
