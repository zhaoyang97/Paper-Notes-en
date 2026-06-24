---
title: >-
  [Paper Note] VerbDiff: Text-Only Diffusion Models with Enhanced Interaction Awareness
description: >-
  [CVPR 2025][Image Generation][Human-Object Interaction Generation] This paper proposes VerbDiff, a text-to-image diffusion model that generates accurate human-object interaction images without requiring extra conditions (such as bounding boxes). It eliminates interaction verb bias using Relation Decoupled Guidance (RDG) and extracts local interaction regions from cross-attention maps via an Interaction Region Module (IR Module) for directional guidance.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Human-Object Interaction Generation"
  - "Verb Semantic Understanding"
  - "Relation Decoupled Guidance"
  - "Interaction Region Localization"
  - "Text-to-Image Diffusion"
date: 2026-05-08
content_hash: 1a2be9681441faa7
---

# VerbDiff: Text-Only Diffusion Models with Enhanced Interaction Awareness

**Conference**: CVPR 2025  
**arXiv**: [2503.16406](https://arxiv.org/abs/2503.16406)  
**Code**: None  
**Area**: Object Detection / Image Generation  
**Keywords**: Human-Object Interaction Generation, Verb Semantic Understanding, Relation Decoupled Guidance, Interaction Region Localization, Text-to-Image Diffusion

## TL;DR

This paper proposes VerbDiff, a text-to-image diffusion model that generates accurate human-object interaction images without requiring extra conditions (such as bounding boxes). It eliminates interaction verb bias using Relation Decoupled Guidance (RDG) and extracts local interaction regions from cross-attention maps via an Interaction Region Module (IR Module) for directional guidance.

## Background & Motivation

- Text-to-image diffusion models underperform in depicting human-object interactions, struggling to distinguish semantically distinct interaction verbs (e.g., "walking a bicycle" vs. "riding a bicycle").
- CLIP exhibits a strong object bias, tending to focus on object nouns in the prompt while ignoring semantic differences in verbs.
- Existing methods rely on additional conditions (e.g., LLM layouts, bounding boxes) to provide explicit relation information, yet they still lack a true understanding of interaction semantics.
- Even with bounding box assistance, InteractDiffusion still fails when semantically similar interaction verbs (e.g., "walking" vs. "riding") share similar bounding boxes.
- The interactions in generated images suffer from bias, favoring verbs with the highest frequency in the data distribution (e.g., generating "wearing" instead of "holding" for a backpack).
- The distribution of interaction verbs in the HICO-DET dataset is extremely long-tailed, where common verbs dominate the generation results.
- The CLIP similarity metric is insensitive to subtle semantic differences in interaction verbs.
- Existing methods have difficulty localizing specific interaction regions in the generated images without extra annotations.

## Method

### Overall Architecture

VerbDiff is built upon Stable Diffusion v1.4, where only the cross-attention layers are trained. It contains two core modules: **Relation Decoupled Guidance (RDG)**, which eliminates interaction biases using frequency anchor texts and a triplet loss, and **Interaction Directional Guidance (IDG)**, which extracts interaction regions from cross-attention maps via an IR Module to guide the model towards local interaction details. Training is conducted on HICO-DET using only text inputs and requiring no bounding boxes.

### Key Designs

**Design 1: Relation Decoupled Guidance (RDG)**
- **Function**: Eliminating interaction bias in generated images and enhancing the understanding of semantic differences among different verbs.
- **Mechanism**: For each human-object pair, a frequency anchor verb is defined as $r^{anc} = \arg\max_{r \in R_o} \mathcal{C}(r|o)$ (the most frequent verb for that pair) to construct the anchor text $T^{anc}$. A triplet loss is used to pull the generated image feature $f^{gen}$ closer to the correct ground-truth interaction text $e^{gt}$ and push it away from the anchor text $e^{anc}$: $\mathcal{L}_{\text{triple}} = \max(0, m + \text{sim}(f^{gen}, e^{gt}) - \text{sim}(f^{gen}, e^{anc}))$. Meanwhile, masked regions are used to extract real image features for an image alignment loss, which is multiplied by an effective coefficient $\alpha(k)$ to balance the long-tail distribution.
- **Design Motivation**: The biased interactions in generated images often correspond to the most frequent verbs in the data; thus, semantic differentiation is achieved by explicitly pushing away from the anchor text features.

**Design 2: Interaction Region Module (IR Module)**
- **Function**: Automatically extracting the interaction region from cross-attention maps of the generated image without requiring bounding boxes.
- **Mechanism**: Utilizing the cross-attention maps $\mathcal{A}_h, \mathcal{A}_r, \mathcal{A}_o$ corresponding to the h/r/o tokens, and computing the center points $c_h, c_r, c_o$ of each token through a centroid extraction mechanism. The interaction center is defined as the centroid of these three points $c_{rel}$, and the interaction region is computed as $B_{rel}^{gen} = c_{rel} \pm \|c_h - c_o\|_2^2$.
- **Design Motivation**: Global image-level feature alignment is not fine-grained. Focusing on the local regions where interactions occur can better capture interaction details.

**Design 3: Interaction Directional Guidance (IDG)**
- **Function**: Guiding the model to modify image features within local interaction regions, making them closer to the ground-truth interactions.
- **Mechanism**: Extracting $f_{rel}^{gt}$ and $f_{rel}^{gen}$ from the interaction region, computing the bias feature $f_{rel}^{bias} = f_{rel}^{gt} - f_{rel}^{gen}$, and designing a direction guidance loss that aligns the global-level modification direction with the local-level interaction region modification direction: $\mathcal{L}_{\text{IDG}} = 1 - \frac{(f_{\mathcal{M}}^{gt} - f^{gen}) \cdot f_{rel}^{bias}}{|f_{\mathcal{M}}^{gt} - f^{gen}||f_{rel}^{bias}|}$.
- **Design Motivation**: Ensuring that the model's modifications to the image are concentrated in the interaction region rather than the global layout, achieving fine-grained interaction semantic alignment.

### Loss & Training

The total loss is formulated as $\mathcal{L}_{\text{total}} = \lambda_1 \cdot \mathcal{L}_{\text{rec}} + \lambda_2 \cdot \mathcal{L}_{\text{RDG}} + \lambda_3 \cdot \mathcal{L}_{\text{IDG}}$, where $\lambda_1=1.0, \lambda_2=10, \lambda_3=0.8$. The reconstruction loss uses a mask constraint to focus only on the corresponding interaction region, and the RDG loss is multiplied by a long-tail balancing factor.

## Key Experimental Results

### Main Results

| Model | CLIP T2T | S-BERT T2T | HOI Acc Def. (Full) | KO. (Full) |
|------|----------|------------|-------------------|-----------|
| SD | 0.725 | 0.620 | 16.09 / 20.08 | 18.22 / 21.69 |
| GLIGEN | 0.683 | 0.554 | 15.88 / 17.83 | 17.91 / 19.35 |
| InteractDiffusion | 0.703 | 0.575 | 19.67 / 23.53 | 21.31 / 24.86 |
| **VerbDiff** | **0.733** | **0.633** | **22.59 / 27.05** | **24.79 / 28.43** |

### Ablation Study

| Setting | CLIP T2T | S-BERT T2T | HOI Acc Def. | KO. |
|------|----------|------------|-------------|-----|
| $\mathcal{L}_{rec}$ only | 0.691 | 0.582 | 19.38 | 20.89 |
| +$\mathcal{L}_{triple}$+$\mathcal{L}_{align}$ | 0.700 | 0.589 | 20.32 | 21.87 |
| +$\mathcal{L}_{triple}$+$\mathcal{L}_{IDG}$ | 0.710 | 0.610 | 23.39 | 24.51 |
| **All (Full)** | **0.733** | **0.633** | **22.59** | **24.79** |

### Key Findings
- VerbDiff consistently outperforms InteractDiffusion, which requires explicit bounding boxes, across all evaluation settings.
- The S-BERT metric reflects interaction semantic differences better than CLIP (VerbDiff achieves a more significant improvement on S-BERT).
- IDG provides the largest contribution (+HOI Acc 3.0+), indicating that focusing on the local interaction region is crucial.
- Under complex prompts with multiple interactions, VerbDiff can still accurately distinguish different interaction verbs, achieving performance close to DALL-E 3.
- The HOI accuracy is close to that of the real HICO-DET data (Def. 22.59 vs. 26.52).

## Highlights & Insights

1. **Discovery of frequency anchor text**: The generation bias corresponds to the most frequent verbs, an observation that provides a clear motivation for the decoupling design.
2. **Localizing interaction regions from cross-attention maps**: Localizing where interactions occur without requiring extra annotations, leading to a simple yet effective design.
3. **Introduction of the S-BERT evaluation metric**: Compensating for CLIP's insufficiency in capturing subtle distinctions in interaction semantics.
4. **Training only cross-attention layers**: Lightweight and efficient, completing training in only 17 hours.

## Limitations & Future Work

- The base model, constructed upon SD v1.4, has limited capacity, leaving a performance gap between it and larger models like DALL-E 3.
- The model is only trained and evaluated on HICO-DET; generalization to other HOI datasets remains unverified.
- The IR Module relies on simple geometric centroid calculations of cross-attention, which might not be robust enough for complex multi-person scenarios.
- Future work can combine stronger VLMs and larger-scale datasets to further enhance interaction understanding.

## Related Work & Insights

- Unlike InteractDiffusion which requires explicit bounding boxes, VerbDiff relies solely on text to achieve better interaction understanding.
- The decoupling concept using frequency anchor texts can be extended to other generative tasks suffering from class imbalance bias.
- The semantic localization capability of cross-attention maps offers a new perspective for annotation-free region extraction.

## Rating

⭐⭐⭐⭐ — Precise problem definition; the decoupling concept based on frequency bias is novel. The experimental comparisons are thorough, and the achievement of outperforming bounding-box-required methods under text-only conditions is impressive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] A Comprehensive Study of Decoder-Only LLMs for Text-to-Image Generation](a_comprehensive_study_of_decoder-only_llms_for_text-to-image_generation.md)
- [\[CVPR 2025\] EDEN: Enhanced Diffusion for High-quality Large-motion Video Frame Interpolation](eden_enhanced_diffusion_for_high-quality_large-motion_video_frame_interpolation.md)
- [\[CVPR 2025\] Scaling Down Text Encoders of Text-to-Image Diffusion Models](scaling_down_text_encoders_of_text-to-image_diffusion_models.md)
- [\[CVPR 2025\] Redefining <Creative> in Dictionary: Towards an Enhanced Semantic Understanding of Creative Generation](redefining_creative_in_dictionary_towards_an_enhanced_semantic_understanding_of_.md)
- [\[CVPR 2025\] Re-HOLD: Video Hand Object Interaction Reenactment via Adaptive Layout-instructed Diffusion Model](re-hold_video_hand_object_interaction_reenactment_via_adaptive_layout-instructed.md)

</div>

<!-- RELATED:END -->
