---
title: >-
  [Paper Note] Selective Contrastive Learning for Weakly Supervised Affordance Grounding
description: >-
  [ICCV 2025][Robotics][Affordance Grounding] This paper proposes a selective contrastive learning approach for weakly supervised affordance grounding (WSAG). By combining prototypical contrastive learning and pixel-level contrastive learning, the method adaptively learns affordance-relevant cues at both object and part granularities, effectively preventing the model from attending to action-irrelevant salient features. The approach comprehensively outperforms competing methods that rely on stronger foundation models (GPT-4, LLaVA, etc.) on AGD20K and HICO-IIF benchmarks.
tags:
  - ICCV 2025
  - Robotics
  - Affordance Grounding
  - Weak Supervision
  - Contrastive Learning
  - Part Discovery
  - CLIP
date: 2026-05-08
content_hash: e04d7dc42f300e8c
---

# Selective Contrastive Learning for Weakly Supervised Affordance Grounding

**Conference**: ICCV 2025
**arXiv**: [2508.07877](https://arxiv.org/abs/2508.07877)
**Code**: [https://github.com/hynnsk/SelectiveCL](https://github.com/hynnsk/SelectiveCL)
**Area**: Robotics
**Keywords**: Affordance Grounding, Weak Supervision, Contrastive Learning, Part Discovery, CLIP

## TL;DR

This paper proposes a selective contrastive learning approach for weakly supervised affordance grounding (WSAG). By combining prototypical contrastive learning and pixel-level contrastive learning, the method adaptively learns affordance-relevant cues at both object and part granularities, effectively preventing the model from attending to action-irrelevant salient features. The approach comprehensively outperforms competing methods that rely on stronger foundation models (GPT-4, LLaVA, etc.) on AGD20K and HICO-IIF benchmarks.

## Background & Motivation

**Background**: Weakly supervised affordance grounding (WSAG) aims to mimic the human ability to learn from third-person demonstrations. Given an egocentric image of a target object and several exocentric human-object interaction exemplar images, the task is to localize the object regions that afford a specific action. Mainstream methods learn action classification across viewpoints via a shared classifier, and incorporate part discovery through distillation strategies.

**Limitations of Prior Work**: Since affordance-relevant parts are not always easily distinguishable (e.g., bicycle seat vs. wheel), models relying primarily on classification tend to focus on discriminative patterns that are irrelevant to affordance. For instance, a model may attend to the bicycle frame rather than the seat (the affordance region for the "ride" action). Distillation strategies only function when parts can be reliably identified, resulting in discontinuous training and exacerbated classification bias.

**Key Challenge**: Affordance cues are difficult to reliably extract under weak supervision, while classification objectives inherently favor the most visually discriminative features—which do not necessarily coincide with affordance-relevant regions.

**Goal**: To continuously provide affordance learning signals even when part cues are unreliable, steering model attention away from irrelevant regions toward genuine affordance areas.

**Key Insight**: Rather than relying on isolated part-level learning, the paper introduces selective prototypical and pixel-level contrastive objectives—performing fine-grained part-level learning when information is reliable, and gracefully falling back to object-level learning otherwise, ensuring continuity of training signals.

**Core Idea**: CLIP is used to discover target objects; cross-view cross-referencing extracts affordance-relevant parts; selective contrastive learning then continuously distinguishes affordance-relevant from irrelevant regions under both reliable and unreliable conditions.

## Method

### Overall Architecture

The overall pipeline processes egocentric and exocentric images through a DINO encoder and projection layers to extract features, then proceeds along two branches:
- **Classification branch**: A shared classifier learns action classification across both viewpoints, producing CAMs for localization.
- **Contrastive learning branch**: Prototypical contrastive learning (leveraging exocentric cues) and pixel-level contrastive learning (leveraging egocentric cues) guide the model toward affordance-relevant regions.

At inference, only the egocentric image and action text prompt are used; CAMs are generated to produce localization maps and calibrated by an object affinity map.

### Key Designs

1. **Object Discovery**:

    - *Function*: Localize action-relevant target objects in both egocentric and exocentric images.
    - *Mechanism*: CLIP visual encoder features are extracted with ClearCLIP strategies to enhance local discriminability. Cosine similarity between features and action text prompts yields object affinity maps $A_{\text{obj}}^{\text{ego}} \in \mathbb{R}^{B \times H \times W}$ and $A_{\text{obj}}^{\text{exo}} \in \mathbb{R}^{B \times E \times H \times W}$.
    - *Design Motivation*: The object affinity map provides coarse but reliable object localization as a foundation for subsequent part discovery. Even when part discovery fails, the model can fall back to object-level learning.

2. **Selective Prototypical Contrastive Learning**:

    - *Function*: Discover affordance part cues in exocentric views and distill them into egocentric representations via contrastive learning.
    - *Mechanism*: Part discovery proceeds by combining CAM with the object affinity map, thresholding to extract interaction regions, and applying K-means ($K=3$) clustering to obtain candidate part centroids. Reliability is assessed by comparing against DINO self-attention maps using the pIoU metric. Positive/negative prototypes are then constructed:
        - When reliable: $P^+ = \Phi^+(F, A_{\text{part}})$ (part-level positive prototype), with the egocentric object region feature as anchor.
        - When unreliable: $P^+ = \Phi^+(F, A_{\text{obj}})$ (object-level positive prototype), with the global feature as anchor.
    - Contrastive loss: $\mathcal{L}^{\text{proto}}_b = \frac{-1}{|\mathbf{P}^+_b|} \sum_{p \in \mathbf{P}^+_b} \log \frac{\exp(z^{\text{ego}}_b \circ p / \tau)}{\sum_{n \in (\mathbf{P}^+_b \cup \mathbf{P}^-_b)} \exp(z \circ n / \tau)}$
    - *Design Motivation*: Compared to the pairwise distillation in LOCATE, prototypical contrastive learning not only pulls ego-exo representations closer but also repels prototypes of different action classes and backgrounds, yielding more discriminative features. The selective mechanism ensures object-level signals are preserved even when parts are unreliable.

3. **Selective Pixel Contrastive Learning**:

    - *Function*: Directly distinguish affordance-relevant pixels from irrelevant ones in egocentric images.
    - *Mechanism*: Exploiting CLIP's greater sensitivity to salient objects, a threshold $\rho = \min_{e \in E} \max_{h,w} A_{\text{obj}}^{\text{exo}}$ is set based on the maximum response of the exocentric object affinity map. Egocentric pixels exceeding $\rho$ are labeled as affordance positives $Q^+$; the remainder are negatives $Q^-$. Pixel contrastive loss: $\mathcal{L}^{\text{pix}}_b = \frac{-1}{|Q^+_b|^2} \sum_{z \in Q^+_b} \sum_{p \in Q^+_b} \log \frac{\exp(z \circ p / \tau)}{\sum_{n \in (Q^+_b \cup Q^-_b)} \exp(z \circ n / \tau)}$
    - *Design Motivation*: Prototypical contrastive learning provides global guidance but only implicitly supervises individual pixels. Pixel contrastive learning directly discriminates affordance relevance at the fine-grained level, complementing precise localization. The selective mechanism degrades to object-level learning when exocentric images clearly focus on the object.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}^{\text{ce}} + \lambda_1 \mathcal{L}^{\text{proto}} + \lambda_2 \mathcal{L}^{\text{pix}}$, with $\lambda_1 = \lambda_2 = 1$. DINO ViT-S/16 serves as the backbone, while CLIP ViT-B/16 provides object affinity maps. At inference, an additional CAM calibration step is applied: the binarized object affinity map is element-wise multiplied with the CAM via Hadamard product, restricting activations to object regions and mitigating activation diffusion caused by convolutional receptive fields.

## Key Experimental Results

### Main Results

Performance comparison on AGD20K and HICO-IIF datasets:

| Method | Model | AGD20K-Seen KLD↓ | AGD20K-Unseen KLD↓ | HICO-IIF KLD↓ |
|--------|-------|------------------|-------------------|--------------|
| Cross-view-AG | ResNet50 | 1.538 | 1.787 | 1.779 |
| LOCATE | DINO | 1.226 | 1.405 | 1.593 |
| WSMA | DINO+CLIP | 1.176 | 1.335 | 1.465 |
| WorldAfford | DINO+CLIP+SAM+GPT-4 | 1.201 | 1.393 | - |
| INTRA | DINOv2+ALBEF+GPT-4 | 1.199 | 1.365 | - |
| **Ours** | **DINO+CLIP** | **1.124** | **1.243** | **1.358** |

### Ablation Study

| Configuration | AGD20K-Seen KLD↓ | SIM↑ | NSS↑ | Notes |
|---------------|------------------|------|------|-------|
| (a) Baseline (classification only) | 1.349 | 0.365 | 1.138 | No contrastive learning |
| (c) + Object-level prototypical contrast | 1.271 | 0.392 | 1.153 | Repels background |
| (f) + Part-level prototypical contrast | 1.164 | 0.416 | 1.290 | Finer localization |
| (h) + Object+part pixel contrast | 1.142 | 0.415 | 1.303 | Further refinement |
| (i) Full model + calibration | 1.124 | 0.433 | 1.280 | Best; KLD reduced by 16.7% |

### Key Findings

- **Most significant gains on Unseen splits**: KLD drops from 1.405 (LOCATE) to 1.243 (−11.5%), indicating that contrastive learning enhances generalization. The paper attributes this to the contrastive objective explicitly redirecting attention from background to affordance regions, which is especially effective for unseen objects.
- **Outperforms heavy foundation model pipelines without relying on them**: Using only DINO+CLIP, the method surpasses approaches that leverage GPT-4, SAM, and LLaVA (WorldAfford, AffordanceLLM, INTRA), demonstrating that learning paradigm design matters more than model scale.
- **Object-level learning is foundational; part-level learning further refines**: Ablation shows that object-level contrast (c) yields a KLD reduction of 0.078, slightly less than the gain from adding part-level contrast (f vs. c, reduction of 0.107), but the combination achieves the best results.
- The CAM calibration step yields a notable improvement (KLD from 1.142 to 1.124), underscoring the importance of constraining activations within object boundaries.

## Highlights & Insights

- **Elegant selective mechanism design**: Rather than requiring accurate part supervision at every training step, the method gracefully degrades to object-level learning when reliability is low. This "fine-grained when possible, coarse when necessary, but always learning" strategy is substantially more stable than the intermittent training of prior methods that only learn when parts are reliable.
- **Exploiting CLIP's bias constructively**: CLIP's stronger response to salient objects is leveraged to distinguish affordance regions between egocentric and exocentric views. In exocentric images where objects are small or occluded, CLIP responses are weaker—this differential naturally serves as a signal for part discovery.
- **Transferable to other weakly supervised localization tasks**: The selective multi-granularity contrastive learning framework is applicable to any scenario requiring alternation between supervision signals of varying reliability.
- The post-processing step of calibrating CAM with the object affinity map is a simple yet effective trick that can be reused in related settings.

## Limitations & Future Work

- Hyperparameters such as thresholds $\alpha$ and $\gamma$ are fixed at 0.6, which may be suboptimal for different data distributions.
- The choice of $K=3$ for K-means assumes interaction regions consist of background, affordance parts, and other elements—an oversimplification for complex interaction scenarios.
- Part discovery relies on the quality of DINO self-attention maps, which may be insufficient for scenes with complex textures or heavy occlusion.
- Pixel contrastive learning is applied only to egocentric images; pixel-level optimization for exocentric images is not explored.
- Dynamic affordance prediction from video is not investigated.

## Related Work & Insights

- **vs. LOCATE**: LOCATE distills from exocentric to egocentric only when reliable parts are available, resulting in discontinuous training. This work extends to bidirectional cue utilization across both viewpoints with persistent learning signals.
- **vs. WSMA**: WSMA addresses insufficient expressiveness of classification labels via CLIP semantic attention, but remains classification-driven. This work introduces contrastive learning to explicitly repel backgrounds and non-affordance parts.
- **vs. WorldAfford/INTRA**: These methods rely on large models such as GPT-4 for part knowledge, incurring high cost and hindering end-to-end optimization. The proposed method achieves superior results with a substantially more lightweight design.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The selective multi-granularity contrastive learning framework represents a meaningful and generalizable advancement over the WSAG paradigm, with clear and principled motivation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensively outperforms state-of-the-art on standard benchmarks with thorough ablations; validation on real-world robot application scenarios is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Method descriptions are clear and well-illustrated, though the density of formulations requires careful reading.
- **Value**: ⭐⭐⭐⭐ Provides a more robust learning paradigm for weakly supervised affordance understanding with direct implications for robotic interaction perception.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Weakly-Supervised Learning of Dense Functional Correspondences](weakly-supervised_learning_of_dense_functional_correspondences.md)
- [\[ICCV 2025\] Self-supervised Learning of Hybrid Part-aware 3D Representations of 2D Gaussians and Superquadrics](self-supervised_learning_of_hybrid_part-aware_3d_representations_of_2d_gaussians.md)
- [\[ICCV 2025\] COSMO: Combination of Selective Memorization for Low-cost Vision-and-Language Navigation](cosmo_combination_of_selective_memorization_for_low-cost_vision-and-language_nav.md)
- [\[CVPR 2026\] PanoAffordanceNet: Towards Holistic Affordance Grounding in 360° Indoor Environments](../../CVPR2026/robotics/panoaffordancenet_towards_holistic_affordance_grounding_in_360_indoor_environmen.md)
- [\[ICCV 2025\] Rep-MTL: Unleashing the Power of Representation-Level Task Saliency for Multi-Task Learning](rep-mtl_unleashing_the_power_of_representation-level_task_saliency_for_multi-tas.md)

</div>

<!-- RELATED:END -->
