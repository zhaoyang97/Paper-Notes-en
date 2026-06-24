---
title: >-
  [Paper Note] Generative Image Layer Decomposition with Visual Effects
description: >-
  [CVPR 2025][Image Generation][Image decomposition] LayerDecomp proposes an image layer decomposition framework based on Diffusion Transformers. It decomposes an input image into a clean RGB background layer and an RGBA foreground layer containing transparent visual effects (shadows, reflections). Utilizing a consistency loss, the model learns accurate foreground representations even from unlabeled data, significantly outperforming existing object-removal and spatial-editing m…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Image decomposition"
  - "visual effects preservation"
  - "diffusion models"
  - "shadow and reflection"
  - "image synthesis"
date: 2026-05-08
content_hash: 1a083740dcc9a870
---

# Generative Image Layer Decomposition with Visual Effects

**Conference**: CVPR 2025  
**arXiv**: [2411.17864](https://arxiv.org/abs/2411.17864)  
**Code**: [https://rayjryang.github.io/LayerDecomp](https://rayjryang.github.io/LayerDecomp) (Project Page)  
**Area**: Image Generation / Image Editing  
**Keywords**: Image decomposition, visual effects preservation, diffusion models, shadow and reflection, image synthesis

## TL;DR
LayerDecomp proposes an image layer decomposition framework based on Diffusion Transformers. It decomposes an input image into a clean RGB background layer and an RGBA foreground layer containing transparent visual effects (shadows, reflections). Utilizing a consistency loss, the model learns accurate foreground representations even from unlabeled data, significantly outperforming existing object-removal and spatial-editing methods.

## Background & Motivation
1. **Background**: Large-scale diffusion models have drastically advanced image editing capabilities, yet precise control in image synthesis tasks remains challenging. Visual content editing software (such as Photoshop) heavily relies on layered representations for composition and content creation.
2. **Limitations of Prior Work**: LayerDiffusion can generate transparent layers from text but is unsuitable for image-to-image editing. MULAN provides multi-layer datasets but fails to preserve critical visual effects (such as shadows and reflections), making downstream editing appear unnatural. Existing inpainting methods require loose masks to handle shadows and cannot yield editable foreground layers.
3. **Key Challenge**: There is a lack of large-scale multi-layer datasets containing realistic visual effects annotations. In real-world data, the ground-truth of foreground layers is inaccessible, posing the challenge of learning proper transparent foreground representations under unlabeled conditions.
4. **Goal**: (a) How to construct scalable multi-layer training data, (b) how to learn accurate RGBA foreground representations without foreground ground truth, and (c) how to preserve both background quality and foreground visual effects.
5. **Key Insight**: Leverage a synthetic data pipeline to automatically generate multi-layer training data with shadows, and introduce a small collection of real-world camera-captured image pairs, using a consistency loss to indirectly constrain the learning of the foreground layer.
6. **Core Idea**: Enforce the model to learn correct transparent visual effects representations without foreground annotations via a consistency loss in pixel space, which recomposes the predicted background and foreground and compares the result with the original image.

## Method

### Overall Architecture
LayerDecomp is built upon a Diffusion Transformer (DiT, 5 billion parameters). The input consists of four components: the composite image and the object mask as conditional inputs, alongside noisy latents for the background and foreground. The model simultaneously denoises two latent branches to generate the RGB background and the RGBA foreground, respectively. The background is encoded/decoded using a standard RGB-VAE, while the foreground utilizes an RGBA-VAE fine-tuned from the RGB-VAE. Conditioning information is injected via patch embedding and type embedding, with cross-branch information interaction achieved through the self-attention of the Transformer.

### Key Designs

1. **Dual-branch DiT Denoising Architecture**:
    - **Function**: Outputs both the clean background and transparent foreground layers simultaneously.
    - **Mechanism**: Concatenates the background latent, foreground latent, conditional image latent, and mask latent into a single sequence, with each assigned its corresponding type embedding, before feeding it into a standard DiT. Information transmission from conditions to outputs is realized through self-attention, and the loss is calculated only at the noisy latent positions. The foreground channel is encoded and decoded using an RGBA-VAE (fine-tuned from the original VAE to minimize disruption to the original latent space).
    - **Design Motivation**: Compared to training two independent models, joint denoising imposes mutual constraints on the background and foreground generation, enabling the model to better understand the global structure of the input scene.

2. **Pixel-Space Consistency Loss**:
    - **Function**: Constrains the learning of visual effects in the foreground layer in the absence of foreground ground-truth data.
    - **Mechanism**: At any denoising timestep $t$, the model's prediction is converted back to a clean latent estimate $\hat{x}_0$ via reparameterization, and then decoded to obtain the background $\hat{I}_{bg}$ and the foreground $\hat{I}_{fg}^{RGBA}$. An alpha blending process synthesizes the composite $\hat{I}_{comp}$, and the L1 loss is calculated against the original input image $I_{comp}$: $\mathcal{L}_{consist} = \mathbb{E}_t \sum_{i,j} |I_{comp}(i,j) - \hat{I}_{comp}(i,j)|$. This process requires incorporating the VAE decoder into the training loop (with frozen weights used for forward propagation).
    - **Design Motivation**: For real-world captured data, the RGBA ground truth of the foreground is unavailable. Conventional approaches can only train the foreground branch on synthetic data with ground truth. The consistency loss elegantly bypasses this limitation—it only requires the two layers to reconstruct the input when composed, indirectly forcing the model to correctly allocate visual effects like shadows and reflections to the alpha channel of the foreground layer.

3. **Mixed Dataset Preparation Pipeline**:
    - **Function**: Provides large-scale synthetic triplet data with ground truth and a small amount of real-world data.
    - **Mechanism**: (a) Synthetic data: unoccluded foreground objects are extracted from natural images using instance segmentation, paired with depth estimation to exclude incomplete objects. Shadow synthesis is then applied to generate shadow intensity maps written into the alpha channel, producing RGBA foreground assets. During training, a background image is randomly chosen for composition, yielding a complete triplet $(I_{comp}, I_{fg}^{RGBA}, I_{bg})$. (b) Real-world data: 6,000 camera-captured "object-present/object-absent" image pairs (similar to ObjectDrop), providing only $I_{comp}$ and $I_{bg}$ without foreground ground truth.
    - **Design Motivation**: Purely synthetic data lacks realistic geometry and lighting, whereas purely real-world data lacks foreground ground truth. The hybrid scheme is mutually complementary: synthetic data provides full supervision to train both branches, while real-world data enforces the model to learn natural real-world visual effects via the consistency loss.

### Loss & Training
Total Loss = standard diffusion denoising loss $\mathcal{L}_{dm}$ + pixel-space consistency loss $\mathcal{L}_{consist}$. For real-world data, the foreground latent is masked out from the $\mathcal{L}_{dm}$ computation (due to the lack of foreground ground truth) and is learned indirectly solely through the consistency loss. Training employs the Adam optimizer with lr=1e-5, batch size 128, on 16 A100 GPUs for 80K steps, with an input resolution of 512x512. Inference uses 50-step DDIM sampling.

## Key Experimental Results

### Main Results

| Dataset | Metric | LayerDecomp | ControlNet Inp. | SD-XL Inp. | PowerPaint |
|--------|------|-------------|-----------------|------------|------------|
| RORD | PSNR↑ | **24.79** | 22.01 | 20.81 | 21.26 |
| RORD | LPIPS↓ | **0.132** | 0.182 | 0.166 | 0.201 |
| RORD | FID↓ | **21.73** | 53.71 | 56.28 | 56.56 |
| MULAN | PSNR↑ | **19.13** | 17.79 | 16.04 | 17.17 |
| DESOBAv2 | PSNRm↑ | **38.57** | 36.94 | 34.21 | 29.33 |

User Study: In the object removal task, LayerDecomp is preferred in 83%+ of cases; in the spatial editing task, it is preferred in 87%+ of cases.

### Ablation Study

| Configuration | BG PSNR↑ | Comp PSNR↑ | BG FID↓ | Comp FID↓ |
|------|----------|------------|---------|-----------|
| V0: RGB-only | 28.21 | - | 21.00 | - |
| V1: +RGBA FG (obj only) | 28.28 | 27.53 | 18.48 | 18.83 |
| V2: +RGBA FG (obj+v.e.) | 28.56 | 28.66 | 17.99 | 16.87 |
| Full: V2+L_consist | **29.27** | **30.53** | **16.04** | **12.75** |

### Key Findings
- The consistency loss contributes the most: after incorporating it, Comp PSNR jumps from 28.66 to 30.53, and FID drops from 16.87 to 12.75.
- Incorporating visual effects in the foreground (V2 vs V1) not only improves foreground quality but also inversely boosts background quality, indicating that the decomposition task implicitly enhances the model's understanding of the scene.
- LayerDecomp is robust to mask tightness (metrics for tight/loose masks are almost identical), whereas competing methods are highly sensitive to this factor.
- In the DESOBAv2 shadow removal task, LayerDecomp outperforms methods using loose shadow masks without even requiring a shadow mask.

## Highlights & Insights
- **Ingenious Design of Consistency Loss**: It establishes a closed-loop constraint of "composition $\rightarrow$ comparison" in the pixel space, converting an unsupervised foreground learning problem into a supervised one without any foreground annotations. This methodology can be transferred to any generation task that requires decomposition but lacks partial annotations.
- **Inverse Benefits of Joint Generation**: Adding the foreground branch yields an editable foreground and automatically boosts background quality. This suggests that multi-task joint training can induce a "free lunch" effect.
- **Robustness to Masks**: Prior methods demand meticulously crafted loose masks to cover shadow areas. LayerDecomp can automatically handle them with compact masks, greatly reducing user interaction costs.
- **Training-Free Downstream Editing**: The decomposed layers can be directly manipulated via alpha blending for complex editing tasks like movement, scaling, and recoloring, eliminating the need to train individual models for each editing operation.

## Limitations & Future Work
- Currently, the dataset mainly covers two optical effects (shadows and reflections), and lacks coverage for other effects like smoke or fog.
- Relying on a 5B-parameter DiT model incurs significant inference overhead (50-step DDIM), limiting real-time performance.
- The geometry and lighting of synthetic data might lack diversity, and applicability to extreme scenarios remains to be validated.
- Extending this method to video layer decomposition scenarios is a potential direction, leveraging temporal consistency to further improve quality.

## Related Work & Insights
- **vs LayerDiffusion**: LayerDiffusion generates RGBA layers from text in a text-driven generation task; LayerDecomp is an image-driven decomposition task, which is more suitable for precise image editing.
- **vs MULAN**: MULAN provides multi-layer datasets but fails to preserve visual effects, leading to unnatural results after direct editing. LayerDecomp's consistency loss ensures correct attribution of visual effects.
- **vs ObjectDrop**: ObjectDrop requires model fine-tuning to restore visual effects, whereas LayerDecomp directly models these effects during training, thereby maintaining the integrity of the foreground layer.

## Additional Analysis
- The model's overwhelming victory in the user study (83%+) demonstrates that the layered decomposition scheme is far superior to traditional inpainting in terms of practical editing experience.
- LayerDecomp can be applied to sequential multi-layer decomposition: multiple foreground layers are decomposed sequentially using different instance masks, and each layer can be edited independently before recomposition.
- Methodological takeaway: For generative tasks requiring decomposition yet missing partial ground truth annotations, constructing self-supervised signals by defining composition rules (e.g., alpha blending) is an effective strategy.
- Incorporating the VAE decoder into the forward pass for the consistency loss during training increases GPU memory consumption but enables the utilization of unlabeled data.

## Rating
- Novelty: ⭐⭐⭐⭐ The design of the consistency loss is intuitive, simple, and effective, combined with a practical hybrid data strategy.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on three benchmarks, verified through two user studies, and supported by thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The paper is well-structured and the motivation is formulated naturally.
- Value: ⭐⭐⭐⭐ Highly practical, unlocking layer-based creative editing pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] From Inpainting to Layer Decomposition: Repurposing Generative Inpainting Models for Image Layer Decomposition](../../CVPR2026/image_generation/from_inpainting_to_layer_decomposition_repurposing_generative_inpainting_models_.md)
- [\[CVPR 2025\] Improving Editability in Image Generation with Layer-wise Memory](improving_editability_in_image_generation_with_layer-wise_memory.md)
- [\[CVPR 2025\] Visual Lexicon: Rich Image Features in Language Space](visual_lexicon_rich_image_features_in_language_space.md)
- [\[CVPR 2026\] Qwen-Image-Layered: Towards Inherent Editability via Layer Decomposition](../../CVPR2026/image_generation/qwen-image-layered_towards_inherent_editability_via_layer_decomposition.md)
- [\[ICLR 2026\] Referring Layer Decomposition](../../ICLR2026/image_generation/referring_layer_decomposition.md)

</div>

<!-- RELATED:END -->
