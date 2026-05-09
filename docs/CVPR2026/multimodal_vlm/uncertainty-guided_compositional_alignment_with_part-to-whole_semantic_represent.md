---
title: >-
  [Paper Note] Uncertainty-guided Compositional Alignment with Part-to-Whole Semantic Representativeness in Hyperbolic Vision-Language Models
description: >-
  [CVPR2026][Multimodal VLM][Hyperbolic VLM] This paper proposes UNCHA, a framework that models the semantic representativeness of image parts with respect to the whole scene via hyperbolic uncertainty in hyperbolic VLMs. By incorporating uncertainty-guided contrastive loss and entailment loss, UNCHA enhances compositional scene understanding and outperforms existing hyperbolic VLMs across multiple downstream tasks.
tags:
  - CVPR2026
  - Multimodal VLM
  - Hyperbolic VLM
  - Uncertainty Modeling
  - Part-to-Whole Alignment
  - Compositional Understanding
  - Entailment Loss
date: 2026-05-08
content_hash: 531cbb0e86f255da
---

# Uncertainty-guided Compositional Alignment with Part-to-Whole Semantic Representativeness in Hyperbolic Vision-Language Models

**Conference**: CVPR2026  
**arXiv**: [2603.22042](https://arxiv.org/abs/2603.22042)  
**Code**: [github.com/jeeit17/UNCHA](https://github.com/jeeit17/UNCHA)  
**Area**: Multimodal VLM  
**Keywords**: Hyperbolic VLM, Uncertainty Modeling, Part-to-Whole Alignment, Compositional Understanding, Entailment Loss

## TL;DR
This paper proposes UNCHA, a framework that models the semantic representativeness of image parts with respect to the whole scene via hyperbolic uncertainty in hyperbolic VLMs. By incorporating uncertainty-guided contrastive loss and entailment loss, UNCHA enhances compositional scene understanding and outperforms existing hyperbolic VLMs across multiple downstream tasks.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: VLMs such as CLIP struggle to capture hierarchical relationships (e.g., part-whole, parent-child structures) in Euclidean space, and exhibit bias in multi-object compositional scenes.

Hyperbolic VLMs (e.g., MERU, ATMG, HyCoCLIP) better preserve hierarchical structures via the negative curvature and exponential volume growth of hyperbolic space. However, existing methods do not model the varying semantic representativeness of different parts with respect to the whole — a crop containing the core object of a scene is more representative of the whole than a background crop.

When all parts are treated equally, the model cannot distinguish more representative parts from less representative ones.

## Method

### Overall Architecture
UNCHA extends HyCoCLIP with uncertainty modeling: (1) define hyperbolic uncertainty to reflect semantic representativeness → (2) incorporate it into the contrastive loss → (3) calibrate uncertainty via an entailment loss.

### Key Designs

1. **Hyperbolic Uncertainty Model**:

    - $u(x) = \log(1 + \exp(-\|x\|_2))$
    - Exploits the monotonic relationship between hyperbolic radius (geodesic distance from the origin) and uncertainty
    - Near the origin = more abstract = higher uncertainty; far from the origin = more concrete = lower uncertainty
    - Parts more representative of the whole → lower uncertainty

2. **Uncertainty-Guided Contrastive Loss**:

    - Adaptive temperature: $\tau_{un,i}^I = \exp(u(i_i^{part})/2) \cdot \tau_{gl}$
    - High-uncertainty parts → larger temperature → smaller contribution to the contrastive loss
    - Incorporates a local contrastive loss (aligning part images with part texts)

3. **Uncertainty Calibration via Entailment Loss**:

    - Piecewise continuous entailment loss: $L_{ent}^* = \max(0, \phi - \eta\omega) + \alpha\phi$ (Leaky-ReLU-style relaxation)
    - Uncertainty calibration: $L_{ent}^{cal} = \lfloor L_{ent}^* \rfloor e^{-u(p)} + u(p) + \mathcal{H}(\tilde{u}(p))$
    - Encourages higher uncertainty when the entailment relationship is weak; $u(p)$ prevents excessive uncertainty
    - Entropy regularization $\mathcal{H}$ prevents collapse of uncertainty to a uniform distribution

### Loss & Training
$L = \mathcal{L}_{con}^{un} + \lambda_{ent}\mathcal{L}_{ent}^{un}$

The hyperbolic space is based on the Lorentz model, using exponential and logarithmic maps to transition between the manifold and the tangent space.

## Key Experimental Results

### Main Results

| Model | ImageNet | CIFAR-10 | CUB | Cars | Pets | Notes |
|-------|----------|----------|-----|------|------|-------|
| CLIP (ViT-S/16) | 36.7 | 70.2 | 9.8 | 6.9 | 44.6 | Baseline |
| MERU | 35.4 | 71.2 | 11.3 | 5.2 | 42.7 | Hyperbolic baseline |
| HyCoCLIP | Improved | Improved | Improved | Improved | Improved | + Part alignment |
| UNCHA | Best | Best | Best | Best | Best | + Uncertainty modeling |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| w/o uncertainty guidance | Performance drop | Treating all parts equally is insufficient |
| w/o entropy regularization | Embedding space collapse | Uncertainty tends toward uniformity |
| Uncertainty vs. similarity | r = −0.739 | Strong negative correlation validates the modeling |

### Key Findings
- A strong negative correlation (r = −0.739) between uncertainty and part-whole similarity validates the effectiveness of the proposed modeling
- Semantically more representative parts exhibit lower uncertainty, while ambiguous or unrepresentative crops exhibit higher uncertainty
- UNCHA outperforms existing hyperbolic VLMs across multiple downstream tasks including zero-shot classification, retrieval, and multi-label classification

## Highlights & Insights
- Using the hyperbolic radius as a proxy for uncertainty is a natural and elegant design choice
- The entropy regularization that prevents uncertainty collapse reflects a deep understanding of the properties of hyperbolic space
- The Leaky-ReLU-style relaxation of the entailment loss addresses the vanishing gradient problem that arises after embeddings are pushed into the cone
- Visualization analysis intuitively demonstrates the correspondence between uncertainty and semantic representativeness

## Limitations & Future Work
- The computational complexity of hyperbolic space limits scalability to larger models
- Part images are generated via random cropping; more intelligent part segmentation strategies remain unexplored
- Validation is limited to ViT-S/16 and ViT-B/16; larger visual encoders have yet to be evaluated
- The setting of the uncertainty threshold $\tau_A$ is relatively heuristic

## Related Work & Insights
- MERU first introduced hyperbolic VLMs but modeled only cross-modal entailment
- HyCoCLIP extended entailment to intra-modal settings but did not differentiate part representativeness
- The idea of using hyperbolic radius as an uncertainty proxy is generalizable to any hyperbolic representation learning scenario

## Rating
- Novelty: ⭐⭐⭐⭐ Novel combination of hyperbolic uncertainty and semantic representativeness modeling
- Experimental Thoroughness: ⭐⭐⭐⭐ Zero-shot classification across 16 datasets with multi-dimensional evaluation
- Writing Quality: ⭐⭐⭐⭐ Detailed mathematical derivations and clear structure
- Value: ⭐⭐⭐⭐ Advances compositional understanding capabilities of hyperbolic VLMs

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] UniFit: Towards Universal Virtual Try-on with MLLM-Guided Semantic Alignment](../../AAAI2026/multimodal_vlm/unifit_towards_universal_virtual_try-on_with_mllm-guided_semantic_alignment.md)
- [\[CVPR 2026\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)
- [\[ICLR 2026\] Error Notebook-Guided, Training-Free Part Retrieval in 3D CAD Assemblies via Vision-Language Models](../../ICLR2026/multimodal_vlm/error_notebook-guided_training-free_part_retrieval_in_3d_cad_assemblies_via_visi.md)
- [\[CVPR 2026\] When to Think and When to Look: Uncertainty-Guided Lookback](when_to_think_and_when_to_look_uncertainty-guided_lookback.md)
- [\[CVPR 2026\] Uncertainty-Aware Knowledge Distillation for Multimodal Large Language Models](uncertainty-aware_knowledge_distillation_for_multimodal_large_language_models.md)

<!-- RELATED:END -->
