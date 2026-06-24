---
title: >-
  [Paper Note] Circuit Mechanisms for Spatial Relation Generation in Diffusion Transformers
description: >-
  [CVPR 2026][Image Generation][Diffusion Transformer] Internal circuit mechanisms for spatial relation generation in Diffusion Transformers (DiT) are revealed through mechanistic interpretability: random embedding models employ a two-stage modular circuit (relation heads + object generation heads), whereas T5 encoder models fuse relation information into object tokens for single-token decoding, with significant differences in robustness between the two mechanisms.
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Diffusion Transformer"
  - "Spatial Relation Generation"
  - "Mechanistic Interpretability"
  - "Attention Circuits"
  - "Text Encoder"
date: 2026-05-08
content_hash: 44f6d39af0ce901e
---

# Circuit Mechanisms for Spatial Relation Generation in Diffusion Transformers

**Conference**: CVPR 2026  
**arXiv**: [2601.06338](https://arxiv.org/abs/2601.06338)  
**Code**: None  
**Area**: Image Generation / Mechanistic Interpretability  
**Keywords**: Diffusion Transformer, Spatial Relation Generation, Mechanistic Interpretability, Attention Circuits, Text Encoder

## TL;DR

Internal circuit mechanisms for spatial relation generation in Diffusion Transformers (DiT) are revealed through mechanistic interpretability: random embedding models employ a two-stage modular circuit (relation heads + object generation heads), whereas T5 encoder models fuse relation information into object tokens for single-token decoding, with significant differences in robustness between the two mechanisms.

## Background & Motivation

Text-to-image (T2I) diffusion models have achieved immense progress in high-quality image generation but frequently fail to compose spatial relations between multiple objects (e.g., "a red square at the top-left of a blue circle"). While the accuracy of single-object attribute generation is rapidly improving, progress in spatial relation generation remains slow.

Existing works propose various remedies (layout conditioning, cross-attention guidance, curriculum learning, etc.), but **few have investigated the internal model mechanisms** to understand why spatial relation generation fails. This paper is motivated by:

1. How do neural networks encode and utilize non-commutative relations (e.g., "A above B" $\neq$ "B above A")?
2. How can key heads be systematically summarized and localized despite the complexity of attention map analysis due to the iterative sampling nature of diffusion models?
3. Is the bottleneck of spatial relation generation in cross-attention or text encoding? A **holistic** research perspective is required.

## Method

### Overall Architecture

The authors construct a minimal text-to-image dataset and train DiT models of various scales from scratch to learn the generation of images containing two objects (with combined shape and color attributes) arranged in specified spatial relations. A combination of 3 shapes $\times$ 2 colors $\times$ 8 spatial relations is used. The model architecture uses a PixArt-style DiT, comparing three text encoders: T5-XXL, Random Token Embedding (RTE), and RTE without positional encoding.

### Key Designs

**1. Attention Synopsis: Localizing key heads from tens of millions of attention maps**

The iterative sampling of diffusion models causes an explosion in attention analysis—layers $\times$ heads $\times$ timesteps $\times$ conditional/unconditional $\times$ token counts, often reaching tens of millions of maps. The authors propose a scalable aggregation paradigm: tokens are grouped by category (image tokens by object segmentation, text tokens by semantic attributes), attention is aggregated at category granularity to obtain interpretable inter-category interactions, and results are averaged over timesteps. This compresses the entire attention tensor into a single [layer, head] synopsis map, allowing rapid localization of heads responsible for spatial relations.

**2. Two-stage Modular Circuit of RTE-DiT: Label positions first, then place objects**

In the Random Token Embedding (RTE) model, spatial relations are found to be generated in two steps. The spatial relation head (L2H8) activates at the very first step (step 0), allowing image token sinusoidal positional encodings to interact with relation word embeddings via the QK circuit—"above" produces vertical gradients and "left" produces horizontal gradients. This effectively paints a layer of **positional labels** on the canvas to mark object placement (a mechanism strikingly similar to molecular gradients guiding cell differentiation in embryonic development). The object generation head (L4H3) activates later (steps 4-8), reading these labels and linking regions with matching labels to corresponding shape tokens, thereby "growing" the objects in correct positions. Object heads only transmit shape identity and are independent of spatial position or relations. This modularity is the source of RTE circuit robustness.

**3. Fused Single-token Decoding Circuit of T5-DiT: Relation information hidden in object words**

With the T5 encoder, the mechanism is entirely different. T5's self-attention fuses the entire sentence context into each token; consequently, DiT decodes spatial relations from **non-relation word tokens** (specifically the second shape word, shape2). Variance decomposition shows that in T5 embeddings, shape2 explains 37.5% of the variance while relations contribute only 12.1%, but relation information is amplified to 21.3% after DiT MLP projection. The authors perform causal validation using **vector arithmetic**—subtracting the original relation vector and adding a new relation vector in T5 embeddings changes the generated spatial positions, confirming that relation information is indeed encoded within fused object tokens.

**4. Weight-space Head Screening: Finding relation heads without sample generation**

Running per-sample generation to identify heads is computationally expensive. The authors propose a fast screening method in pure weight space: directly calculating QK interactions between image positional features and text relation features to check if the generated spatial maps align with reference relation gradients. Aligned heads are identified as candidate relation heads without requiring any sampling.

### Loss & Training

- Standard diffusion training using DPM-Solver++ (14 steps) sampling, CFG=4.5.
- Training various model sizes: DiT-B (12 layers, 12 heads, 768 dim), mini (6 layers, 6 heads, 384 dim), micro (6 layers, 3 heads, 192 dim), nano (3 layers, 3 heads, 192 dim).
- Evaluation using EMA weights.
- Four-dimensional evaluation metrics: color, shape, unique binding, and spatial relation.

## Key Experimental Results

### Main Results

| Model | Text Encoder | Color↑ | Shape↑ | Unique Binding↑ | Spatial Relation↑ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| DiT-B | T5 | 99% | 97% | 93% | 89% |
| DiT-B | RTE | 99% | 96% | 90% | 86% |
| DiT-B | RTE w/o pos | 99% | 96% | 41% | 15% |
| DiT-nano | RTE | - | - | - | 5% |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Ablate L2H8 relation attention | Spatial relation accuracy 67%→33% | Relation heads are critical for spatial layout |
| Ablate L4H3 object attention | Shape accuracy 90%→76% | Object heads have a causal role in shape generation |
| Ablate relation words in T5-DiT | Almost no effect | T5 fuses relation info into other tokens |
| Ablate shape2 in T5-DiT | Relation accuracy drops 50% | Relation info mostly encoded in shape2 |
| Insert filler "the" in T5-DiT | Relation accuracy drops significantly | T5 circuits are sensitive to minor vocabulary changes |
| Insert filler in RTE-DiT | Remains stable | Modular circuits are more robust to perturbations |

### Key Findings

1. **Circuit mechanisms depend on the text encoder**: RTE uses a modular two-stage circuit (relation → position labels → objects), while T5 uses a fused single-token decoding circuit.
2. **Positional encoding is necessary**: RTE without positional encoding achieves only 15% spatial relation accuracy as it cannot distinguish between "A above B" and "B above A".
3. **Learning dynamics are staged**: Color → Shape → Attribute Binding → Spatial Relation; relation learning is the slowest.
4. **Robustness differences are significant**: RTE-DiT is sensitive to relation word ablation but robust to filler words; T5-DiT exhibits the opposite behavior.
5. **Transferable to pre-trained models**: Sparse spatial circuits can also be identified in PixArt-Sigma.

## Highlights & Insights

- **Mechanistic interpretability methodology**: Attention Synopsis and weight-space head screening provide scalable analysis tools for understanding large-scale DiT models.
- **Biological analogy**: The gradient mechanism of spatial relation heads bears a striking resemblance to molecular gradients used in embryonic development.
- **Unified perspective**: This work unifies the "cross-attention as bottleneck" and "text encoder as bottleneck" viewpoints, showing they each apply under different configurations.
- **Design implications**: There is a trade-off between modular (RTE) and fused (T5) architectures—modular is more robust and interpretable, while fused is more compact but fragile.
- **Practical value**: Improving spatial relation generation may require prioritizing improvements in embedding models rather than the DiT architecture itself.

## Limitations & Future Work

1. Experiments were conducted on a **minimal dataset** (3 shapes $\times$ 2 colors $\times$ 8 relations); real-world scene complexity is significantly higher.
2. Only spatial relations between two objects were studied; circuit mechanisms for multi-object (3+) compositions remain to be explored.
3. The comparison between RTE and T5 may be affected by the volume and sufficiency of training data.
4. The study did not explore how to **improve** spatial relation generation using the discovered circuits (e.g., via attention intervention).
5. Analysis of pre-trained models (PixArt-Sigma) is relatively thin, as its baseline spatial relation performance is already weak.

## Related Work & Insights

- Methodologically consistent with Transformer circuit analysis (Elhage et al., 2021), but applied to diffusion models for the first time.
- Methods such as Attend-and-Excite improve compositionality by manipulating cross-attention; this work provides a mechanistic explanation for such methods.
- The impact of text encoder choice on model behavior is often underestimated—CLIP, T5, and random embeddings lead to fundamentally different internal computations.
- Provides guidance for designing more robust T2I architectures: modular circuits may be superior to fused circuits.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to reveal specific circuit mechanisms for spatial relation generation in DiT.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Causal manipulation and ablation designs are rigorous, though limited by the simplified setting.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear narrative, high-quality illustrations, and tight logic.
- Value: ⭐⭐⭐⭐ — Provides a critical foundation for understanding and improving compositional generation in T2I models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers](just-in-time_training-free_spatial_acceleration_for_diffusion_transformers.md)
- [\[CVPR 2026\] MakeAnything: Harnessing Diffusion Transformers for Multi-Domain Procedural Sequence Generation](makeanything_harnessing_diffusion_transformers_for_multi-domain_procedural_seque.md)
- [\[CVPR 2026\] Region-Adaptive Sampling for Diffusion Transformers](region-adaptive_sampling_for_diffusion_transformers.md)
- [\[CVPR 2026\] PixelDiT: Pixel Diffusion Transformers for Image Generation](pixeldit_pixel_diffusion_transformers_for_image_generation.md)
- [\[CVPR 2026\] ResCa: Residual Caching for Diffusion Transformers Acceleration](resca_residual_caching_for_diffusion_transformers_acceleration.md)

</div>

<!-- RELATED:END -->
