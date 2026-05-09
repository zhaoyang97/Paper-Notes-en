---
title: >-
  [Paper Note] Your Text Encoder Can Be An Object-Level Watermarking Controller
description: >-
  [ICCV 2025][Image Generation][Watermark Embedding] By fine-tuning only the pseudo-token embedding $\mathcal{W}_*$ in the text encoder, this work achieves object-level invisible watermark embedding in T2I diffusion model-generated images, attaining 99% bit accuracy (48 bits) with $10^5\times$ fewer parameters.
tags:
  - ICCV 2025
  - Image Generation
  - Watermark Embedding
  - Text Encoder
  - Object-Level Watermarking
  - Textual Inversion
  - Diffusion Models
date: 2026-05-08
content_hash: 8d497d1e3e161610
---

# Your Text Encoder Can Be An Object-Level Watermarking Controller

**Conference**: ICCV 2025
**arXiv**: [2503.11945](https://arxiv.org/abs/2503.11945)
**Code**: [GitHub](https://github.com/)
**Area**: Diffusion Models / Image Watermarking
**Keywords**: Watermark Embedding, Text Encoder, Object-Level Watermarking, Textual Inversion, Diffusion Models

## TL;DR

By fine-tuning only the pseudo-token embedding $\mathcal{W}_*$ in the text encoder, this work achieves object-level invisible watermark embedding in T2I diffusion model-generated images, attaining 99% bit accuracy (48 bits) with $10^5\times$ fewer parameters.

## Background & Motivation

As text-to-image (T2I) diffusion models become increasingly widespread, copyright protection and content provenance have grown in importance. Existing generative watermarking methods suffer from the following core issues:

**Large Parameter Count**: Methods such as Stable Signature and AquaLoRA require modifying large modules like the UNet or VAE decoder, involving over $10^5$ trainable parameters, and cannot be conveniently transferred to different LDM pipelines.

**Lack of Spatial Control**: Nearly all existing methods watermark the entire image and cannot selectively watermark a specific object (e.g., only the "cat"), whereas object-level watermarking is critical in many scenarios such as protecting unique objects within an image.

**Fragility of Post-Processing Watermarks**: Watermarks added in a post-generation stage are vulnerable to image processing attacks (e.g., cropping, rotation, JPEG compression) and exhibit poor robustness.

**Requirement for Additional Information**: The few methods that support local watermarking (e.g., WAM) require segmentation masks to localize the watermark region, introducing additional computational overhead.

The key observation of this paper is that the text encoder is relatively underutilized in the LDM pipeline, and the success of Textual Inversion demonstrates that learning new token embeddings can inject new concepts. If a "watermark token" can be learned, object-level watermarking can be naturally realized through the cross-attention mechanism — users simply place the watermark token adjacent to the target object in the prompt.

## Method

### Overall Architecture

The core mechanism introduces a pseudo-token $\mathcal{W}_*$ into the text encoder vocabulary and fine-tunes its embedding to carry watermarking functionality. Training adopts an Img2Img pipeline: an input image is encoded into latent space $z_0$, forward noise is applied for $\tau^*$ steps to obtain $z_{\tau^*}$, which is then denoised under the condition of $\mathcal{W}_*$ to produce $z'_{0,w}$, decoded by the VAE decoder into the watermarked image $I_{w,m}$, and fed into a pretrained watermark detector $D_w$ to train the embedding.

At inference, users simply insert $\mathcal{W}_*$ into the prompt:
- Full-image watermark: `[A photo of a cat $\mathcal{W}_*$]`
- Object watermark: `A photo of a [cat $\mathcal{W}_*$]`

### Key Designs

1. **Token Embedding Learning (Core Innovation)**: Unlike methods that modify large modules such as the UNet or VAE, this approach learns only a 768-dimensional token embedding vector, reducing the parameter count by $10^5\times$ compared to baseline methods. The token can be loaded in a plug-and-play fashion into any LDM pipeline compatible with the text encoder (e.g., Stable Diffusion v1.5, personalized models, style models) without modifying core model parameters. This Textual Inversion-style prêt-à-porter training significantly lowers the deployment barrier.

2. **Optimal Timestep and Latent Space Matching Loss**: Different noise timesteps $\tau$ present a trade-off between image quality and watermarking performance — large $\tau \sim T$ degrades image quality while strengthening the watermark, whereas small $\tau \sim 0$ preserves quality but weakens the watermark. Through experiments, the authors identify the optimal timestep $\tau^* = 8$ and introduce a latent space matching loss to ensure invisibility:

    $\mathcal{L}_z = \min_{\mathcal{W}_*} \mathbb{E}_t \left[ \| z^*_t - z'_t(\mathcal{W}_*) \|_2^2 \right]$

   The total training loss is $\mathcal{L} = \alpha \mathcal{L}_w + \beta \mathcal{L}_z$, where $\mathcal{L}_w$ is the BCE bit-embedding loss. Selecting a small timestep $\tau^*$ close to the VAE encoder not only improves quality but also enhances robustness, as watermark embedding occurs early in the pipeline and is thus harder to remove through adversarial perturbations in later stages.

3. **Object-Level Watermark Control Mechanism**: The method leverages cross-attention maps in the UNet to localize the watermark region. During T2I generation, at each timestep $t$, attention maps $\mathcal{M}_{\mathcal{P}_i}^{(t)}$ and $\mathcal{M}_{\mathcal{W}_*}^{(t)}$ are extracted for the target object token $\mathcal{P}_i$ and watermark token $\mathcal{W}_*^i$, and watermark localization is achieved by superimposition:

    $\mathcal{M}_{\mathcal{P}_i}^{(t)} \leftarrow (1-\alpha) \cdot \mathcal{M}_{\mathcal{P}_i}^{(t)} + \alpha \cdot \mathcal{M}_{\mathcal{W}_*}^{(t)}$

   A watermark overlay intensity controller $\pi(t)$ (step function or smooth function) is further introduced to concentrate the watermark effect within the optimal timestep range observed during training, improving watermark localization precision.

### Loss & Training

- **Bit-embedding loss** $\mathcal{L}_w = BCE(D_w(Dec(z'_{0,w})), m)$: ensures the watermark key $m \in \{0,1\}^{48}$ is correctly embedded.
- **Latent space matching loss** $\mathcal{L}_z$: constrains the watermarked latent trajectory to remain close to the original, ensuring invisibility.
- The watermark detector $D_w$ is sourced from a pretrained model (e.g., AquaLoRA); the entire T2I pipeline and detector remain frozen during training.
- Training data: MS-COCO subset (2,000 images).

## Key Experimental Results

### Main Results

**Full-image watermarking comparison on WikiArt dataset (48 bits)**:

| Method | Params | PSNR↑ | FID↓ | No Attack BA↑ | Brightness BA↑ | Blur BA↑ | JPEG BA↑ | SDEdit BA↑ | WMAttacker BA↑ |
|--------|--------|-------|------|--------------|---------------|---------|---------|-----------|--------------|
| Stable Sig. | $10^5$+ | 31.57 | 24.71 | 0.99 | 0.93 | 0.78 | 0.55 | 0.58 | 0.53 |
| AquaLoRA | $10^5$+ | 31.46 | 17.27 | 0.94 | 0.91 | 0.81 | 0.76 | 0.68 | 0.67 |
| WAM | $10^5$+ | 36.46 | 16.27 | 0.97 | 0.93 | 0.84 | 0.84 | 0.72 | 0.71 |
| **Ours + SD** | **768** | **39.92** | **14.89** | **0.99** | **0.98** | **0.97** | **0.95** | **0.85** | **0.87** |

### Ablation Study

**Object-level watermarking robustness (various cropping and multi-object settings)**:

| Configuration | No Attack BA | Brightness BA | Blur BA | Rotation BA | JPEG BA |
|---------------|-------------|--------------|---------|------------|---------|
| Single object – segmented + white bg | 0.99 | 0.97 | 0.97 | 0.96 | 0.97 |
| Single object – crop 0.5× | 0.92 | 0.91 | 0.91 | 0.92 | 0.90 |
| 2 objects (no overlap) | 0.94 | 0.93 | 0.95 | 0.94 | 0.96 |
| 3 objects (no overlap) | 0.90 | 0.89 | 0.90 | 0.90 | 0.99 |
| 2 objects (overlap ≥ 40%) | 0.79 | 0.76 | 0.80 | 0.74 | 0.74 |

### Key Findings

- The proposed method outperforms AquaLoRA by more than 7 dB in PSNR and achieves 20%+ improvement in robustness against adversarial attacks.
- Only 768 parameters are required to achieve 99% bit accuracy, representing a $10^5\times$ gain in parameter efficiency.
- Detection accuracy remains above 89% even when the object region is reduced to 40% of its original size.
- Multi-object watermarking maintains 90%+ accuracy in non-overlapping cases, but performance degrades when overlap exceeds 40%.
- The token transfers directly to personalized models and different-style SD variants while maintaining high robustness.

## Highlights & Insights

- **Extreme Parameter Efficiency**: Reducing watermarking to a Textual Inversion problem is an elegant formulation; achieving the task with only 768 parameters is remarkable.
- **First Object-Level Watermarking Without Masks**: This is the first approach to realize object-level watermarking in a T2I generation pipeline without requiring segmentation masks, leveraging cross-attention maps for natural localization.
- **Plug-and-Play Design**: The watermark token can be combined with any LDM compatible with the text encoder, including personalized models, making deployment highly convenient in practice.
- **Early Embedding Improves Robustness**: Embedding the watermark at the text encoding stage (early in the pipeline) makes it significantly harder to remove than watermarks added in post-processing.

## Limitations & Future Work

- Object-level watermarking relies on the accuracy of cross-attention maps; imprecise attention maps may cause the watermark to bleed into unintended regions.
- Detection accuracy degrades notably when multiple objects have high overlap (>40%).
- Validation is currently limited to Stable Diffusion v1.5; applicability to newer architectures such as SDXL and Flux requires further investigation.
- The number of watermark tokens is fixed at one; future work could explore multi-token strategies to embed a larger number of bits.

## Related Work & Insights

- The success of Textual Inversion inspired encoding watermarking capability into learned token embeddings.
- The attention manipulation technique from Prompt-to-Prompt is adopted to achieve object-level watermark localization.
- The proposed approach can be generalized to other text-encoder-driven generative models, such as video generation models.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Reducing watermark embedding to a token learning problem is highly original; this is the first mask-free object-level generative watermarking method.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation of both full-image and object-level settings, though validation on newer models such as SDXL is absent.
- **Writing Quality**: ⭐⭐⭐⭐ The method is clearly presented and experiments are thorough.
- **Value**: ⭐⭐⭐⭐⭐ High practical value; parameter efficiency and plug-and-play design facilitate straightforward deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Attention to Neural Plagiarism: Diffusion Models Can Plagiarize Your Copyrighted Images!](attention_to_neural_plagiarism_diffusion_models_can_plagiarize_your_copyrighted_.md)
- [\[ICCV 2025\] Invisible Watermarks, Visible Gains: Steering Machine Unlearning with Bi-Level Watermarking Design](invisible_watermarks_visible_gains_steering_machine_unlearning_with_bi-level_wat.md)
- [\[NeurIPS 2025\] Robustness in Both Domains: CLIP Needs a Robust Text Encoder](../../NeurIPS2025/image_generation/robustness_in_both_domains_clip_needs_a_robust_text_encoder.md)
- [\[ICCV 2025\] Efficient Input-Level Backdoor Defense on Text-to-Image Synthesis via Neuron Activation Variation](efficient_input-level_backdoor_defense_on_text-to-image_synthesis_via_neuron_act.md)
- [\[ICCV 2025\] ScoreHOI: Physically Plausible Reconstruction of Human-Object Interaction via Score-Guided Diffusion](scorehoi_physically_plausible_reconstruction_of_human-object_interaction_via_sco.md)

</div>

<!-- RELATED:END -->
