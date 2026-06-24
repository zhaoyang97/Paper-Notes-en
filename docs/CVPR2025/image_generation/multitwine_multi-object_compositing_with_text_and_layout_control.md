---
title: >-
  [Paper Note] Multitwine: Multi-Object Compositing with Text and Layout Control
description: >-
  [CVPR 2025][Image Generation][Multi-object compositing] This paper proposes Multitwine, the first generative model supporting simultaneous multi-object compositing guided by text and layouts. By jointly training the compositing and personalized generation tasks, and incorporating cross-attention/self-attention decoupling losses, it achieves natural interactions (e.g., hugging, playing guitar) when inserting multiple objects simultaneously; the user study indicates a preferenc…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Multi-object compositing"
  - "Diffusion models"
  - "Text control"
  - "Layout control"
  - "Identity preservation"
date: 2026-05-08
content_hash: 867dc7760f77cf3b
---

# Multitwine: Multi-Object Compositing with Text and Layout Control

**Conference**: CVPR 2025  
**arXiv**: [2502.05165](https://arxiv.org/abs/2502.05165)  
**Code**: Not open-sourced  
**Area**: Image Generation/Object Compositing  
**Keywords**: Multi-object compositing, Diffusion models, Text control, Layout control, Identity preservation

## TL;DR

This paper proposes Multitwine, the first generative model supporting simultaneous multi-object compositing guided by text and layouts. By jointly training the compositing and personalized generation tasks, and incorporating cross-attention/self-attention decoupling losses, it achieves natural interactions (e.g., hugging, playing guitar) when inserting multiple objects simultaneously; the user study indicates a preference rate of up to 97.1% for interaction realism.

## Background & Motivation

**Background**: Object Compositing seamlessly integrates new objects into an existing scene. Existing methods (e.g., AnyDoor, IMPRINT, ObjectStitch) only support sequential compositing of single objects, requiring serialized processing for multiple objects.

**Limitations of Prior Work**: (1) Sequential compositing cannot handle interaction scenarios requiring simultaneous re-posing (e.g., two people hugging—placing one after another prevents adjusting the posture of the first); (2) The sequential approach leads to inconsistent illumination and harmonization between objects; (3) Lack of text control, making it impossible to specify relationships between objects (e.g., "holding", "hugging"); (4) Balancing text and image inputs is challenging—text guidance fails when images dominate, and vice-versa.

**Key Challenge**: Multi-object compositing requires simultaneously considering spatial interactions between objects and maintaining their respective identities, but the attention layers of diffusion models tend to blend features of semantically similar objects (semantic leakage).

**Goal**: (1) Support simultaneous multi-object compositing with natural interactions; (2) Balance text alignment and image identity preservation; (3) Prevent identity leakage among multiple objects.

**Key Insight**: Jointly train the compositing task with personalized generation (customization)—the compositing task learns inpainting, harmonization, and relighting, while the personalization task focuses on text-image alignment and identity preservation, rendering them complementary.

**Core Idea**: Jointly train compositing and personalization tasks to balance text and image alignment; construct multimodal embeddings by inserting image embeddings after their corresponding text tokens using grounding information, and apply attention decoupling losses to prevent identity leakage.

## Method

### Overall Architecture

Based on the Stable Diffusion 1.5 Inpainting model. The inputs consist of a background image, layout masks (bounding boxes for each object and a global modification area mask), $N$ object images, and text descriptions. Object images are encoded via DINO ViT-G/14 and aligned to the text space using an adapter, while the text is encoded by CLIP. Utilizing grounding information, the image embeddings of each object are appended after their corresponding text tokens to form multimodal embeddings, which are then injected through the cross-attention of the U-Net. Layout masks, noise, and the background are concatenated as the U-Net input.

### Key Designs

1. **Multimodal Embeddings**:
    - **Function**: Balance text and image control signals, achieving grounding-level text-image correspondence.
    - **Mechanism**: Given the text description of the $i$-th object $\mathcal{C}_i$ and the object image $\mathcal{O}_i$, the image embedding $\mathcal{A}(\mathcal{E}_I(\mathcal{O}_i))$ is concatenated after the text embedding $\mathcal{E}_T(\mathcal{C}_i)$.
    - **Design Motivation**: Direct addition would lead to one modality dominating the other. Concatenating based on spatial correspondence through grounding information allows cross-attention to naturally associate visual features with corresponding text regions.
    - During training, each modality is independently and randomly dropped with a 30% probability to ensure robust single-modality performance.

2. **Cross-Attention and Self-Attention Identity Decoupling Losses**:
    - **Function**: Prevent semantic and visual feature leakage among multiple objects.
    - **Mechanism**:
      - Cross-attention loss $\mathcal{L}_c$: Encourages the cross-attention maps of each object's text-image tokens to focus on their corresponding segmentation region $\mathcal{S}_i$.
      - Self-attention loss $\mathcal{L}_s$: Suppresses self-attention responses between pixels $\mathbf{x} \in \mathcal{S}_i$ and $\mathbf{y} \in \mathcal{S}_j$ belonging to different objects.
    - **Design Motivation**: The attention layers of diffusion models naturally tend to share features across semantically similar regions, which causes feature blending when compositing two cats, for example. The two losses constrain this from the perspectives of "where text signals should go" and "where visual signals should not come from," respectively.
    - During inference, bounding box masks are additionally used to mask the cross-attention scores.

3. **Joint Training of Compositing and Personalization Tasks**:
    - **Function**: Improve text-image balance through complementary tasks.
    - **Mechanism**: Replace $\mathcal{M}_G$ with a full-image mask and $\mathcal{I}_{BG}$ with an empty image with a 50% probability, enabling the model to perform pure personalization generation (without the inpainting burden).
    - **Design Motivation**: Training solely on the compositing task forces the model to simultaneously learn object compositing, re-posing, scene completion, harmonization, and identity preservation, which is overly complex. The personalization task occasionally allows the model to focus solely on text-image alignment.
    - Ablation validation: Removing joint personalization training causes the DINO score to drop sharply from 0.540 to 0.449 under interaction scenarios.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_d + \alpha \mathcal{L}_c + \beta \mathcal{L}_s$$

where $\mathcal{L}_d$ is the standard diffusion denoising loss, $\alpha=10^3$, and $\beta=1$.

## Key Experimental Results

### Main Results: Multi-Object Compositing - Interaction Scenarios (Table 1, overlapping bboxes)

| Method | CLIP-I↑ | DINO↑ |
|------|---------|-------|
| AnyDoor | 0.727 | 0.520 |
| IMPRINT | 0.713 | 0.525 |
| TOTB | 0.716 | 0.485 |
| **Multitwine** | **0.741** | **0.532** |

### Ablation Study (Table 3, MultiComp-action subset)

| Configuration | DINO↑ | CLIP-I↑ | CLIP-Tloc↑ |
|------|-------|---------|-----------|
| Full Model | 0.540 | 0.745 | 0.286 |
| W/o self-attn loss | 0.538 | 0.744 | 0.283 |
| W/o all attn losses | 0.534 | 0.739 | 0.270 |
| W/o joint personalization training | **0.449** | 0.705 | 0.295 |
| W/o multi-view data | 0.535 | 0.751 | 0.268 |

### User Study (Fig. 6)

- vs IMPRINT: Image quality preference rate is 66.7%, and interaction realism preference rate is **97.1%**.
- vs Emu2Gen: Outperforms in alignment across all four dimensions: text, layout, object, and background.

### Key Findings

- Simultaneous compositing improves DINO by 0.012-0.022 compared to sequential compositing (even using the same model) in interaction scenarios.
- Joint personalization training is the most critical ablation factor; removing it results in a sharp DINO drop of 0.091.
- The model exhibits emergent capabilities for simultaneous compositing of 3+ objects and subject-driven inpainting.
- Providing text guidance enhances performance in object interaction scenarios but has a minor effect on non-interaction scenarios.

## Highlights & Insights

1. **Pioneering problem definition**: First to propose simultaneous multi-object compositing, resolving the fundamental issue of sequential compositing in which already-placed objects cannot be re-posed.
2. **Exquisite joint training design**: Compositing and personalization act as mutual auxiliary tasks, dividing the training burden through a 50% probability switch.
3. **Data generation pipeline**: Integrates video data (multi-view + relationship annotations), image data (automatic caption + grounding), and manually gathered data, tackling the scarcity of multimodal alignment training data.
4. **Emergent capabilities**: Although trained on only 2 objects, the model generalizes to simultaneous compositing of 3+ objects, indicating that it has learned a generalized prior for interaction.

## Limitations & Future Work

1. Based on SD 1.5, the overall quality is constrained. Transferring to SDXL/SD3 should yield significant improvements.
2. The length of multimodal embeddings grows linearly as the number of objects increases, presenting a scalability bottleneck.
3. The attention losses are effective during training, but additional masking operations are still required during inference.
4. The code is not open-sourced, limiting reproducibility.

## Related Work & Insights

- **AnyDoor/IMPRINT**: SOTA in single-object compositing, utilizing DINO features to preserve identity fidelity. Multitwine extends this paradigm to multi-object compositing.
- **KOSMOS-G/UNIMO-G**: Multi-entity personalized generation, but lacks support for compositing inpainting/relighting.
- **Emu2Gen**: The closest competitor supporting layouts and multi-entity generation, suffers from poor text-image balancing.
- **FastComposer**: Its attention decoupling concept in multi-subject generation is leveraged by this work.

## Rating

- ⭐ Innovation: 8/10 — Initiates simultaneous multi-object compositing; clever joint training strategy.
- ⭐ Experimental Thoroughness: 8/10 — Thorough user studies and clear ablations, but lacks quantitative metrics for interaction quality.
- ⭐ Value: 7/10 — Wide application potential, but constrained by the SD1.5 base and lack of public code.
- ⭐ Overall: 8/10 — Defines a new task with an effective solution; the joint training strategy is highly referenceable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Compass Control: Multi Object Orientation Control for Text-to-Image Generation](compass_control_multi_object_orientation_control_for_text-to-image_generation.md)
- [\[CVPR 2025\] Re-HOLD: Video Hand Object Interaction Reenactment via Adaptive Layout-instructed Diffusion Model](re-hold_video_hand_object_interaction_reenactment_via_adaptive_layout-instructed.md)
- [\[CVPR 2025\] MVPortrait: Text-Guided Motion and Emotion Control for Multi-View Vivid Portrait Animation](mvportrait_text-guided_motion_and_emotion_control_for_multi-view_vivid_portrait_.md)
- [\[CVPR 2025\] MCA-Ctrl: Multi-party Collaborative Attention Control for Image Customization](mca_ctrl_attention_control_customization.md)
- [\[CVPR 2025\] MTADiffusion: Mask Text Alignment Diffusion Model for Object Inpainting](mtadiffusion_mask_text_alignment_diffusion_model_for_object_inpainting.md)

</div>

<!-- RELATED:END -->
