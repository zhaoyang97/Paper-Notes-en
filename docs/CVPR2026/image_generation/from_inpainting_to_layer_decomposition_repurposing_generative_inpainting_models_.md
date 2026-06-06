---
title: >-
  [Paper Note] From Inpainting to Layer Decomposition: Repurposing Generative Inpainting Models for Image Layer Decomposition
description: >-
  [CVPR 2026][Image Generation][Layer Decomposition] This paper identifies an intrinsic connection between image layer decomposition and inpainting/outpainting, and proposes the Outpaint-and-Remove framework…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Layer Decomposition"
  - "Image Inpainting"
  - "Diffusion Models"
  - "Foreground Extraction"
  - "Parameter-Efficient Fine-Tuning"
date: 2026-05-08
content_hash: 0a0c4a0f76ee7e6c
---

# From Inpainting to Layer Decomposition: Repurposing Generative Inpainting Models for Image Layer Decomposition

**Conference**: CVPR 2026  
**arXiv**: [2511.20996](https://arxiv.org/abs/2511.20996)  
**Code**: [https://inpaintinglayerdecomp.github.io/](https://inpaintinglayerdecomp.github.io/)  
**Area**: Diffusion Models / Image Editing  
**Keywords**: Layer Decomposition, Image Inpainting, Diffusion Models, Foreground Extraction, Parameter-Efficient Fine-Tuning

## TL;DR
This paper identifies an intrinsic connection between image layer decomposition and inpainting/outpainting, and proposes the Outpaint-and-Remove framework, which efficiently adapts a pretrained inpainting DiT model (FLUX.1-Fill-dev) for layer decomposition via lightweight LoRA fine-tuning. A multi-modal context fusion module is introduced to preserve fine details. The method achieves state-of-the-art performance using only 100K synthetic training samples.

## Background & Motivation

1. **Background**: An image can be viewed as a layered composition of foreground and background. Layer decomposition requires simultaneously extracting the foreground (with occlusion recovery) and completing the background (object removal) from a single image. Existing methods such as LAYERDECOMP require full fine-tuning of closed-source large models, incurring substantial computational and data costs.

2. **Limitations of Prior Work**: (1) High-quality annotated layer data is extremely scarce, with MULAN being the only publicly available benchmark dataset; (2) training from scratch or full fine-tuning of generative models demands extensive computational resources and commercial datasets, making reproduction inaccessible to most researchers; (3) existing inpainting models only perform background filling and cannot simultaneously extract the foreground.

3. **Key Challenge**: Layer decomposition is conceptually analogous to inpainting (background layer = filling the masked region; foreground layer = outpainting beyond the mask), yet existing inpainting models lack foreground extraction capability, while dedicated layer decomposition methods require training from scratch.

4. **Goal**: Can existing powerful inpainting models be adapted for high-quality layer decomposition with minimal modifications and data?

5. **Key Insight**: Unify layer decomposition as a combination of inpainting (background) and outpainting (foreground), leveraging the region-filling capability already present in inpainting models.

6. **Core Idea**: Layer decomposition is bidirectional inpainting — the background undergoes region filling while the foreground undergoes region outpainting — and a single inpainting model with lightweight adaptation can handle both simultaneously.

## Method

### Overall Architecture
Outpaint-and-Remove builds upon the pretrained FLUX.1-Fill-dev (a DiT-based inpainting diffusion model). Given an input image and a binary mask, the model outputs a background layer (RGB, clean background after object removal) and a foreground layer (RGBA, extracted foreground with alpha channel and recovered occluded regions). Key modifications include: (1) a multi-modal context tokenization module that fuses auxiliary cues such as edges, segmentation maps, and depth; (2) a bidirectional image-mask context design to guide foreground extraction and background generation respectively; (3) a dedicated RGBA codec for handling the transparency channel of the foreground. The overall model is fine-tuned using LoRA for parameter efficiency.

### Key Designs

1. **Multi-Modal Context Tokenization Module**

    - **Function**: Fuses multi-modal cues — edge maps, segmentation maps, and depth maps — into a compact representation as generation conditions.
    - **Mechanism**: Each modality image is first encoded into tokens using the pretrained DiT's VAE encoder. However, naively concatenating all tokens would cause the standard attention complexity to explode as $O(K^2)$. Inspired by linear attention, a small fixed number $N \ll K$ of latent query tokens are introduced; cross-attention compresses all modality tokens into an $N$-dimensional representation, reducing complexity to $O(KN)$, approximately linear.
    - **Design Motivation**: Retaining as much detail as possible in the latent space; multi-modal cues provide rich spatial and semantic priors that help the model understand the semantic structure of regions to be filled.

2. **Bidirectional Image-Mask Context Design**

    - **Function**: Distinguishes the distinct requirements of foreground extraction and background generation, controlling the model's balance between "generating new content" and "preserving existing content."
    - **Mechanism**: Standard inpainting uses only the background context $c_{I-M}^b$ (the region inside the mask to be filled). This paper additionally introduces a foreground context $c_{I-M}^f$ (the region outside the mask to be outpainted). The foreground context informs the model that the content inside the mask should be preserved rather than replaced, while the background context indicates that the masked region requires filling. Both contexts are concatenated with their corresponding noisy tokens along the channel dimension, forming two parallel inputs to the DiT.
    - **Design Motivation**: Without the foreground context, the model tends to hallucinate or alter content in the foreground region. The bidirectional context design explicitly informs the model which regions to preserve and which to generate.

3. **Parameter-Efficient Fine-Tuning Strategy (PEFT + RGBA Decoding)**

    - **Function**: Adapts the inpainting model to learn layer decomposition with minimal additional parameters.
    - **Mechanism**: The base inpainting DiT weights are frozen; only the input projection layers are fine-tuned, and LoRA (rank=256) is inserted into every attention and FFN layer. The background uses the RGB format and directly reuses the original VAE, while the foreground uses the RGBA format with a separately fine-tuned RGBA codec. The choice of LoRA rank is critical: rank=128 is insufficient to learn the new task, while rank=1024 excessively overrides the pretrained prior and causes hallucinations.
    - **Design Motivation**: Leveraging the strong generative prior of the inpainting model, the new capability of foreground extraction can be learned with only a small number of additional parameters, substantially reducing training cost.

### Loss & Training
- Standard flow matching loss is adopted.
- Training data is constructed entirely from public resources: MULAN (real foregrounds with complete texture but incomplete shapes) + LayerDiffuse (synthetic foregrounds with complete shapes but texture artifacts) + OpenImages (backgrounds); the mixed strategy captures complementary advantages of both foreground types.
- Batch size 8, learning rate 5e-5, trained for 7,200 iterations.
- Input resolution: 1024×1024.
- Imperfect masks are used during training so that the model learns to infer accurate object boundaries.

## Key Experimental Results

### Main Results — Background Removal (MULAN Test Set)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ |
|------|-------|-------|--------|------|
| FLUX.1-Fill-dev (baseline) | 25.59 | 0.92 | 0.09 | 35.96 |
| PowerPaint | 23.46 | 0.76 | 0.17 | 41.67 |
| OmniEraser | 21.45 | 0.72 | 0.31 | 55.80 |
| Qwen-Image-Edit | 19.07 | 0.64 | 0.24 | 63.49 |
| **Ours** | **27.30** | **0.93** | **0.08** | **25.97** |

Compared to the baseline FLUX.1-Fill-dev, the proposed method achieves a gain of 1.71 dB in PSNR and a reduction of 9.99 in FID.

### Ablation Study

| Configuration | PSNR | FID | Note |
|------|------|-----|------|
| Ours (full, rank=256) | **27.30** | **25.97** | Full model |
| rank=128 | 26.34 | 33.92 | Insufficient rank |
| rank=1024 | 27.15 | 27.32 | Rank too large, overrides prior |
| w/o foreground context $c_{I-M}^f$ | 27.04 | 27.49 | Foreground hallucination |
| w/o multi-modal context $c_{MM}$ | 27.16 | 28.02 | Degraded semantic understanding |
| w/o synthetic foreground | 27.18 | 27.11 | Incomplete foreground shapes |
| Kontext baseline | 26.22 | 36.14 | Weaker inpainting foundation |

### Key Findings
- An inpainting model (FLUX.1-Fill-dev) serves as a stronger foundation for layer decomposition than a general image-to-image model (FLUX-Kontext), validating the intrinsic connection between inpainting and layer decomposition.
- The presence or absence of the foreground context has a substantial impact on foreground extraction quality (evident in qualitative comparisons); without it, the model hallucinate in the foreground region.
- There exists a sweet spot for LoRA rank (256): too small and the model cannot acquire the new capability; too large and the pretrained prior is disrupted.
- In the user study, the proposed method achieves a preference rate of 59.51%, substantially outperforming matting-based approaches.

## Highlights & Insights
- **A Unified Perspective Grounded in Task Essence**: Decomposing layer decomposition into inpainting + outpainting is an elegant and insightful observation that reduces a complex task to a recombination of existing capabilities.
- **Public Data Only + Lightweight Adaptation**: No commercial datasets or full fine-tuning are required; 100K synthetic samples with LoRA suffice to achieve SOTA. This "democratized" design philosophy is broadly transferable.
- **Mixed Foreground Data Strategy**: Real foregrounds provide fine-grained texture but incomplete shapes, while synthetic foregrounds offer complete shapes but lower texture fidelity. The complementary data design is applicable to other domain-gap problems.

## Limitations & Future Work
- The method still fails on complex scenes involving cluttered objects, large occluded regions, or hands holding objects.
- Training data is synthetically constructed, introducing a distributional gap from the layered structure of real images.
- The alpha matting precision of foreground extraction may not match that of dedicated matting methods.
- The evaluation benchmark is limited (MULAN is the only publicly available layer dataset), potentially introducing evaluation bias.

## Related Work & Insights
- **vs. LAYERDECOMP**: The latter requires full fine-tuning of a closed-source model on large-scale high-quality data; the proposed method achieves SOTA using only LoRA and public data, offering a more practical solution.
- **vs. MattingAnything / DiffMatte**: Matting methods extract only the visible foreground contour without recovering occluded regions; the proposed method recovers the complete foreground shape via outpainting.
- **vs. LayerDiffuse**: LayerDiffuse is a model for generating RGBA layers and is used here as a training data source rather than a competing method.

## Rating
- Novelty: ⭐⭐⭐⭐ The unified perspective reinterpreting inpainting as layer decomposition is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablations, though evaluation benchmarks are limited.
- Writing Quality: ⭐⭐⭐⭐ Clear figures and intuitive motivating examples.
- Value: ⭐⭐⭐⭐ A lightweight, practical, and reproducible solution for layer decomposition.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Referring Layer Decomposition](../../ICLR2026/image_generation/referring_layer_decomposition.md)
- [\[CVPR 2026\] Cycle-Consistent Tuning for Layered Image Decomposition](cycle-consistent_tuning_for_layered_image_decomposition.md)
- [\[CVPR 2026\] Reparameterized Tensor Ring Functional Decomposition for Multi-Dimensional Data Recovery](reparameterized_tensor_ring_functional_decomposition_for_multi-dimensional_data_.md)
- [\[CVPR 2026\] Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers](pluggable_pruning_with_contiguous_layer_distillation_for_diffusion_transformers.md)
- [\[CVPR 2026\] Layer Consistency Matters: Elegant Latent Transition Discrepancy for Generalizable Synthetic Image Detection](layer_consistency_matters_elegant_latent_transition_discrepancy_for_generalizabl.md)

</div>

<!-- RELATED:END -->
