---
title: >-
  [Paper Note] Image Referenced Sketch Colorization Based on Animation Creation Workflow
description: >-
  [CVPR 2025][Image Generation][Sketch Colorization] Mimicking the actual animation production workflow, this paper proposes an image-referenced sketch colorization framework based on diffusion models. By introducing Split Cross-Attention coupled with a switchable LoRA mechanism, it processes foreground and background colorization separately, successfully eliminating spatial entanglement artifacts. After training on 4.8M images, it outperforms existing methods in qualitative…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Sketch Colorization"
  - "Diffusion Models"
  - "Split Cross-Attention"
  - "LoRA"
  - "Animation Production Workflow"
date: 2026-05-08
content_hash: 519c19fc71dc7175
---

# Image Referenced Sketch Colorization Based on Animation Creation Workflow

**Conference**: CVPR 2025  
**arXiv**: [2502.19937](https://arxiv.org/abs/2502.19937)  
**Code**: [https://github.com/tellurion-kanata/colorizeDiffusion](https://github.com/tellurion-kanata/colorizeDiffusion)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Sketch Colorization, Diffusion Models, Split Cross-Attention, LoRA, Animation Production Workflow

## TL;DR
Mimicking the actual animation production workflow, this paper proposes an image-referenced sketch colorization framework based on diffusion models. By introducing Split Cross-Attention coupled with a switchable LoRA mechanism, it processes foreground and background colorization separately, successfully eliminating spatial entanglement artifacts. After training on 4.8M images, it outperforms existing methods in qualitative, quantitative, and user study evaluations.

## Background & Motivation

1. **Background**: Sketch colorization is the most labor-intensive step in the animation and digital illustration industries. Deep learning methods are categorized into text-guided, user-guided, and image-referenced approaches. Specifically, image-referenced methods can be seamlessly integrated into existing workflows.

2. **Limitations of Prior Work**: Text-guided methods cannot provide precise color and style reference; user-guided methods still require manual operations; image-referenced methods (e.g., IP-Adapter, ColorizeDiffusion) suffer from spatial entanglement issues—where spatial mismatch between the reference image and the sketch leads to artifacts such as extra limbs or incorrect hairstyles.

3. **Key Challenge**: Local embeddings extracted by the image reference encoder contain excessive spatial-positional information, which conflicts with the spatial structure of the sketch, causing mutual interference between foreground and background regions.

4. **Goal**: To eliminate spatial entanglement artifacts and achieve independent colorization of the foreground character and background, while maintaining high fidelity to the reference colors.

5. **Key Insight**: Observing the actual workflow of animation studios—character designers create reference designs, keyframe artists draw sketches, colorists color the foreground first, then the background, and finally composite the frame. The authors choose to algorithmically simulate this decoupled workflow.

6. **Core Idea**: To separate the training of foreground and background LoRA parameters through Split Cross-Attention, enabling the diffusion model to independently process the colorization of both regions in a single forward pass, thereby completely decoupling spatial information interference.

## Method

### Overall Architecture
Inputs: sketch $X_s$, reference color image $X_r$, and foreground mask $X_m$. Output: colorized result $Y$. Pipeline: (1) A pre-trained ViT (OpenCLIP-H) extracts local embeddings of the reference image as the color guide; (2) a multi-layer sketch encoder injects spatial guidance into the latent layers of the U-Net; (3) the Split Cross-Attention layer processes corresponding regions using independent foreground and background LoRA weights; (4) multiple LoRA modes can be switched during inference to adapt to different scenarios.

### Key Designs

1. **Split Cross-Attention**:

    - **Function**: Processes foreground and background region colorization using different parameters within a single forward pass.
    - **Mechanism**: Each cross-attention layer contains two sets of LoRA weights, $W_f^t$ (foreground) and $W_b^t$ (background), corresponding to their respective Q/K/V projections. A spatial mask $m_s$ is used to classify pixels into foreground ($m_s > ts_s$) and background. The attention for the foreground region is computed using the modified foreground LoRA weights $\hat{W}_f^t = W^t + W_f^t$, while the background region is computed using the modified background LoRA weights $\hat{W}_b^t = W^t + W_b^t$. The two sets of LoRA are trained independently without mutual interference. The foreground LoRA rank is fixed at 16, and the background LoRA rank is set to $0.5 \times \min(D_q, D_{kv})$.
    - **Design Motivation**: In animation images, the foreground characters and backgrounds exhibit significant differences in color distribution, patch size, hue, and texture. Processing the two separately prevents the spatial-positional information of characters in the reference image from leaking into the background region of the sketch (i.e., spatial entanglement).

2. **Recovery Transformer**:

    - **Function**: Processes background embeddings to bridge the gap between foreground and background reference information.
    - **Mechanism**: After the background reference image is encoded by ViT, it passes through a trainable Transformer $\varphi$ to recover details, yielding $e_b = \varphi(\phi(r_b))$. This step is performed before the foreground and background embeddings are concatenated as K/V inputs.
    - **Design Motivation**: Directly injecting foreground and background embeddings into cross-attention separately leads to degraded structure preservation and lower synthesis quality. The Recovery Transformer compensates for the information loss caused by the split operation.

3. **Switchable LoRA Inference Modes**:

    - **Function**: Adapts to different scenarios by switching the activated LoRA modules without altering the base model weights.
    - **Mechanism**: Three inference modes are designed: Vanilla mode (no LoRA activated, suitable for most scenarios); Bg2Fig mode (only foreground LoRA activated, suitable when the reference image has a complex background); and Fig2Fig mode (both LoRA sets and the Recovery Transformer activated, suitable for character-to-character colorization and most effective at eliminating spatial entanglement).
    - **Design Motivation**: Real-world combinations of sketches and reference images are highly diverse; a single mode cannot accommodate all scenarios. The switchable design offers users flexible choices.

### Loss & Training
- Pre-training Stage: VAE and U-Net are initialized with WaifuDiffusion. The dynamic reference dropout rate is decreased from 80% to 50% to prevent distribution shift, trained for 6 epochs.
- Fine-tuning Stage: VAE, U-Net, and the sketch encoder are frozen, and only the Recovery Transformer and switchable LoRAs are trained, for 3 epochs.
- The loss is the standard diffusion denoising loss: $$\mathcal{L}(\theta) = \mathbb{E}[\|\epsilon - \epsilon_\theta(z_t, t, s, \phi(r))\|_2^2]$$
- Trained using 4×H100 GPUs, accelerated by DeepSpeed ZeRO2, using the AdamW optimizer with lr=0.0001.

## Key Experimental Results

### Main Results
Evaluated on 4.8M training data and 52K validation data:

| Method | FID↓ | PSNR↑ | MS-SSIM↑ | CLIP Similarity↑ |
|------|------|-------|----------|------------|
| IP-Adapter | 38.92 | 28.68 | 0.5478 | 0.8672 |
| InstantStyle | 40.81 | 28.11 | 0.4459 | 0.8042 |
| T2I-Adapter | 41.16 | 28.13 | 0.3243 | 0.7180 |
| AnimeDiffusion | 61.60 | 27.85 | 0.3185 | 0.7319 |
| ColorizeDiffusion | 9.53 | 28.74 | 0.5913 | 0.8775 |
| Yan et al. (GAN) | 27.01 | **29.25** | 0.5253 | 0.7634 |
| **Ours** | **6.83** | 28.91 | **0.6002** | **0.8829** |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| Baseline (No Split CA) | Severe spatial entanglement | Extra arms, incorrect clothing |
| + Split Cross-Attention | Mitigates artifacts but color saturation decreases | Spatial separation is effective, but information is insufficient |
| + Split CA + LoRA | Color and detail improvement | But some artifacts remain |
| + Split CA + LoRA + Recovery Trans (Full) | **No artifacts, high quality** | Complete pipeline is optimal |

### Key Findings
- Split Cross-Attention is the core to eliminating spatial entanglement, and the Recovery Transformer is a crucial supplement to ensure quality.
- In the user study, 40 participants preferred Ours across 25 image sets (at a significance level of $p < 0.01$).
- The Fig2Fig mode is the strongest at eliminating spatial entanglement, whereas the Bg2Fig and Vanilla modes are superior for background generation.
- The GAN method (Yan et al.) achieved the highest PSNR because of its limited generative capability; its output is close to the average image, which paradoxically favors it in the perception-distortion tradeoff.

## Highlights & Insights
- **Workflow-Inspired Design**: Drawing the core principle of "decoupled foreground-background colorization" from the practical workflows of animation studios and converting it into a technical solution is both natural and highly effective. This approach of refining design principles from domain-specific knowledge is worth emulating.
- **Elegance of Split Cross-Attention**: It separates the foreground and background through LoRA residual terms alone, without modifying the pre-trained weights, thereby reserving the full capability of the pre-trained model.
- **Switchable Inference Modes**: Adapts to different scenarios without retraining enhances practicality. This modular, inference-time switching strategy can be generalized to other conditional generation tasks.

## Limitations & Future Work
- Strongly dependent on mask extraction quality; incorrect masks directly lead to colorization failure.
- Only trained and validated on anime-style data, making its generalization capability to real-world photographs unknown.
- Currently handles single frames; the authors plan to extend it to video colorization in the future.
- Future work could explore automatic foreground/background separation schemes without requiring masks.
- The semantic matching level between the reference image and the sketch affects output quality; performance may be suboptimal under extreme mismatch.

## Related Work & Insights
- **vs ColorizeDiffusion**: Both use local embeddings, but ColorizeDiffusion does not segment the foreground/background, remaining plagued by spatial entanglement. This work directly addresses this via Split CA + LoRA.
- **vs IP-Adapter/T2I-Adapter**: These general-purpose adapter methods perform poorly in sketch colorization scenarios (FID 38-41 vs 6.83 for Ours), as they do not account for the spatial mismatch between the reference and target.
- **vs GAN Methods**: While the GAN method leads in PSNR, its FID and perceptual quality are significantly worse, indicating that diffusion models possess qualitative advantages in generating complex textures and colors.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of Split Cross-Attention and switchable LoRA is novel, and the workflow-inspired approach is natural.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, covering qualitative/quantitative comparisons, ablation studies, and user studies.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with rich illustrations and a smooth, logical motivation.
- Value: ⭐⭐⭐⭐ Directly valuable for automating sketch colorization in the animation industry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] AniDoc: Animation Creation Made Easier](anidoc_animation_creation_made_easier.md)
- [\[CVPR 2025\] MangaNinja: Line Art Colorization with Precise Reference Following](manganinja_line_art_colorization_with_precise_reference_following.md)
- [\[CVPR 2026\] Towards High-resolution and Disentangled Reference-based Sketch Colorization](../../CVPR2026/image_generation/towards_high-resolution_and_disentangled_reference-based_sketch_colorization.md)
- [\[CVPR 2025\] Free-viewpoint Human Animation with Pose-correlated Reference Selection](free-viewpoint_human_animation_with_pose-correlated_reference_selection.md)
- [\[CVPR 2025\] Consistent and Controllable Image Animation with Motion Diffusion Models](consistent_and_controllable_image_animation_with_motion_diffusion_models.md)

</div>

<!-- RELATED:END -->
