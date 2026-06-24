---
title: >-
  [Paper Note] LaVin-DiT: Large Vision Diffusion Transformer
description: >-
  [CVPR 2025][Image Generation][Large Vision Models] LaVin-DiT proposes a large vision foundation model based on the Diffusion Transformer. Through spatio-temporal VAE encoding, joint diffusion Transformer denoising, and in-context learning, it achieves unified processing of over 20 vision tasks. Scaling from 0.1B to 3.4B parameters, it significantly outperforms the autoregressive large vision model LVM on multiple tasks.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Large Vision Models"
  - "Diffusion Transformer"
  - "In-Context Learning"
  - "Unified Vision Framework"
  - "Multi-task Learning"
date: 2026-05-08
content_hash: 22f5f36e9de45011
---

# LaVin-DiT: Large Vision Diffusion Transformer

**Conference**: CVPR 2025  
**arXiv**: [2411.11505](https://arxiv.org/abs/2411.11505)  
**Code**: [https://derrickwang005.github.io/LaVin-DiT/](https://derrickwang005.github.io/LaVin-DiT/)  
**Area**: Diffusion Models / Multimodal VLM  
**Keywords**: Large Vision Models, Diffusion Transformer, In-Context Learning, Unified Vision Framework, Multi-task Learning

## TL;DR
LaVin-DiT proposes a large vision foundation model based on the Diffusion Transformer. Through spatio-temporal VAE encoding, joint diffusion Transformer denoising, and in-context learning, it achieves unified processing of over 20 vision tasks. Scaling from 0.1B to 3.4B parameters, it significantly outperforms the autoregressive large vision model LVM on multiple tasks.

## Background & Motivation

1. **Background**: Large Language Models (LLMs) can handle multiple language tasks within a unified framework. The computer vision field is also pursuing similar "Large Vision Models" (LVMs), currently focusing on two main paths: image-based generation methods (such as Painter, which reformulates dense prediction as image inpainting) and sequence modeling methods (such as LVM, which quantizes visual data into discrete tokens for next-token prediction).

2. **Limitations of Prior Work**: Directly transferring sequence modeling methods from NLP architectures faces two major issues—(a) autoregressive token-by-token generation is computationally inefficient for high-dimensional visual data; (b) flattening 2D/3D visual data into 1D sequences destroys spatial relationships (as shown in Figure 1, where red and blue tokens are spatially adjacent but far apart in the sequence).

3. **Key Challenge**: How to achieve efficient unified multi-task generation while preserving the integrity of spatial structures?

4. **Goal**: Design a large vision foundation model based on diffusion models to avoid the efficiency and spatial destruction problems of autoregressive generation, while supporting 20+ image and video tasks.

5. **Key Insight**: Diffusion models naturally denoise all tokens in parallel without destroying spatial structures. Combined with in-context learning, input-target pairs can be used as task definitions without requiring task-specific heads or fine-tuning.

6. **Core Idea**: Compress visual data into a continuous latent space using an ST-VAE, perform parallel denoising with a Joint Diffusion Transformer (J-DiT), and define and adapt to various vision tasks via in-context learning.

## Method

### Overall Architecture
LaVin-DiT consists of three core components: (1) **ST-VAE** (Spatio-Temporal Variational Autoencoder), which compresses images/videos from pixel space into a compact latent space ($4 \times 8 \times 8$ compression rate); (2) **J-DiT** (Joint Diffusion Transformer), which performs conditional denoising in the latent space—using the in-context input-target pairs and query images as conditions (clean), while the target is scaled with noise and iteratively denoised; (3) **During inference**, task-related input-target pairs are randomly sampled as context, which are sent into J-DiT along with the test query to generate predictions.

### Key Designs

1. **Spatio-Temporal Variational Autoencoder (ST-VAE)**:

    - **Function**: Compresses high-dimensional image/video data into a low-dimensional continuous latent space.
    - **Mechanism**: Employs causal 3D convolutions and deconvolutions, with $2\times$ downsampling/upsampling alternated across 4 symmetric stages. The first two stages process both spatial and temporal dimensions, while the last stage only processes the spatial dimension, resulting in a total compression rate of $4 \times 8 \times 8$. To prevent future information leakage, padding is applied at the start of temporal convolutions. The first frame of a video is compressed independently (spatial only), and subsequent frames are compressed across both space and time. Training consists of two stages: first on image-only data, then joint image and video training, using MSE + perceptual loss + adversarial loss.
    - **Design Motivation**: Performing diffusion directly in pixel space is computationally prohibitive. The causal design of ST-VAE ensures temporal consistency, and the high compression rate of $4 \times 8 \times 8$ significantly reduces the computational overhead of the subsequent J-DiT.

2. **Joint Diffusion Transformer (J-DiT)**:

    - **Function**: Performs conditional diffusion generation in the latent space, with full-sequence joint attention at its core.
    - **Mechanism**: Based on the MM-DiT architecture, patch embeddings (patch size $2 \times 2$) are constructed separately for conditional and target sequences. An adaptive RMS normalization (AdaRN) is introduced for both conditions and targets to independently modulate the two representation spaces using different timestep embeddings. **Full-sequence joint attention** linearly projects and concatenates the condition and target sequences before performing bidirectional attention. This allows each sequence to operate within its own space while remaining aware of the other, achieving task-specific alignment. Grouped-query attention is used instead of multi-head attention to reduce parameters, QK-Norm is added to stabilize long-sequence training, and sandwich normalization is used to maintain activation magnitudes in residual connections. **3D RoPE** is adopted for positional encoding to represent spatio-temporal relationships in a unified manner.
    - **Design Motivation**: Conditions (clean) and targets (noisy) have different range values, necessitating independent embeddings and normalization; full-sequence joint attention aligns task information better than isolated attention; 3D RoPE captures spatio-temporal positional relations of visual data more accurately than 1D positional encodings.

3. **In-Context Learning Multi-Task Unification**:

    - **Function**: Defines tasks by providing input-target pairs, adapting to different tasks without fine-tuning during inference.
    - **Mechanism**: During training, a-t set of input-target pairs is sampled from the task data as the task context and concatenated with the query before being fed into J-DiT. Targets are perturbed with noise and trained via flow matching (conditional flow matching loss $\ell_{\text{CFM}} = \int_0^1 \mathbb{E}[|v_\theta(z_t, t) - (z_0 - z_1)|_2^2] \text{d}t$). During inference, Euler's method is used for reverse integration from noise in $N=20$ steps. Increasing the context length (more input-target pairs) consistently improves performance on downstream tasks.
    - **Design Motivation**: To avoid designing specialized decoding heads or fine-tuning strategies for each task, thereby achieving a truly unified framework. The richness of the information in the context pairs directly affects the quality of task understanding.

### Loss & Training
- J-DiT uses the conditional flow matching (CFM) loss for training in the latent space.
- Two-stage training: first at 256×256 resolution for 100K steps, then at 512×512 resolution for 20K steps.
- AdamW, lr=1e-4, batch=640, 64×A100-80G GPUs.
- Dataset: approx. 3.2 million images + 600K videos, covering 20+ tasks.

## Key Experimental Results

### Main Results

| Task | Metric | LaVin-DiT (3.4B) | LVM (7B) | Gain |
|------|------|-----------------|----------|------|
| Foreground Segmentation Split 1 (unseen) | mIoU ↑ | **67.87** | 48.94 | +18.93 |
| Single Object Detection Split 4 (unseen) | mIoU ↑ | **68.88** | 48.92 | +19.96 |
| NYU-v2 Depth Estimation | AbsRel ↓ | **6.2** | 30.2 | -24.0 |
| NYU-v2 Normal Estimation | MAE ↓ | **15.901** | 23.433 | -7.5 |
| ImageNet Inpainting | FID ↓ | **1.65** | 4.05 | -2.40 |
| Colorization | MSE ↓ | **0.24** | 0.51 | -0.27 |

The model also significantly outperforms LVM on unseen tasks (foreground segmentation, object detection), demonstrating strong generalization capability.

### Scalability Analysis

| Model Scale | Colorization MSE ↓ | Depth AbsRel ↓ |
|-------------|--------------------|----------------|
| 0.1B | 0.609 | 7.6 |
| 1.0B | 0.311 | 6.5 |
| 3.4B | **0.273** | **6.2** |

### Key Findings
- LaVin-DiT 3.4B converges faster and has a lower training loss, showing a clear scaling law.
- Inference speed is 1.7~2.3x faster than LVM (4.67s vs 8.1s at 256×256, 20.1s vs 47.2s at 512×512), highlighting the parallel denoising advantage of diffusion models.
- Increasing the number of in-context pairs consistently improves performance (e.g., depth-to-image FID and deblurring PSNR show continuous improvements).
- Depth estimation AbsRel of 6.2 is close to the expert model Marigold (6.0), and normal estimation MAE of 15.901 even outperforms the expert model StableNormal (19.707).
- Capable of handling video tasks (frame prediction, video depth/normal/optical flow estimation, video instance segmentation, etc.), generating 12 frames of subsequent predictions.

## Highlights & Insights
- The **systematic comparison between diffusion models and autoregressive models for large vision models** is highly convincing: the parallel denoising of diffusion models naturally preserves spatial structure and is faster during inference, whereas the token-by-token generation of autoregressive models is slow and disrupts spatial relationships.
- The design of **decoupled embedding + independent AdaRN for condition/target** in J-DiT elegantly handles the range discrepancy between clean and noisy representations.
- The **unified in-context learning paradigm** allows the same model to process 20+ tasks across images/videos without fine-tuning, where longer contexts yield better results, demonstrating few-shot emergent capabilities similar to LLMs.
- 3D RoPE represents the spatio-temporal structure of visual data more naturally than 1D positional encodings, which can be extended to other video generation/understanding models.

## Limitations & Future Work
- Generalization of the model depends on the distribution of training tasks; generalization is difficult when a task definition significantly deviates from the training distribution.
- The scale and diversity of training data are still far behind those of LLMs, requiring larger-scale visual multi-task datasets in the future.
- Pseudo-labels for depth and normal estimation come from Depth-Anything V2 and StableNormal, placing an upper limit constrained by the accuracy of these models.
- The automatic selection of optimal task contexts (context selection) has not been explored.
- The 3.4B model is trained on 64×A100, which still presents a relatively high computational barrier.

## Related Work & Insights
- **vs LVM (Bai et al.)**: LVM uses autoregressive sequence modeling to unify vision tasks, but it is computationally slow and disrupts spatial relationships. LaVin-DiT replaces this with a Diffusion Transformer, significantly outperforming LVM-7B on almost all tasks while being 2x faster in inference.
- **vs Painter/PromptDiffusion**: These methods achieve multi-tasking through image inpainting / visual prompting but rely on priors from pre-trained diffusion models. LaVin-DiT trains a unified model from scratch, offering greater flexibility and scalability.
- **vs DiT/SD3**: LaVin-DiT extends the Diffusion Transformer from a single generation task to over 20+ understanding and generation tasks, serving as an important exploration of the DiT architecture in general-purpose vision.

## Rating
- Novelty: ⭐⭐⭐⭐ First to extend Diffusion Transformers to a unified large vision foundation model, systematically demonstrating that the diffusion path is superior to the autoregressive path.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 20+ task evaluations, scalability analysis, inference latency comparison, context length study, extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Well-structured paper; the comparison diagram of autoregressive vs. diffusion is intuitive and powerful.
- Value: ⭐⭐⭐⭐⭐ Opens up a new path for large vision models using Diffusion Transformers, with massive potential impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](dit-ic_aligned_diffusion_transformer_for_efficient_image_compression.md)
- [\[NeurIPS 2025\] ICEdit: Enabling Instructional Image Editing with In-Context Generation in Large Scale Diffusion Transformer](../../NeurIPS2025/image_generation/in-context_edit_enabling_instructional_image_editing_with_in-context_generation_.md)
- [\[CVPR 2026\] MMFace-DiT: A Dual-Stream Diffusion Transformer for High-Fidelity Multimodal Face Generation](../../CVPR2026/image_generation/mmface-dit_a_dual-stream_diffusion_transformer_for_high-fidelity_multimodal_face.md)
- [\[CVPR 2025\] InterAct: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation](interact_advancing_large-scale_versatile_3d_human-object_interaction_generation.md)
- [\[NeurIPS 2025\] Linear Differential Vision Transformer: Learning Visual Contrasts via Pairwise Differentials](../../NeurIPS2025/image_generation/linear_differential_vision_transformer_learning_visual_contrasts_via_pairwise_di.md)

</div>

<!-- RELATED:END -->
