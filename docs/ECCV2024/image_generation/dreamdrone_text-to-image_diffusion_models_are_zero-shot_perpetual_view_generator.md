---
title: >-
  [Paper Note] DreamDrone: Text-to-Image Diffusion Models are Zero-shot Perpetual View Generators
description: >-
  [ECCV 2024][Image Generation][Perpetual View Generation] This work proposes DreamDrone, a zero-shot, training-free perpetual view generation pipeline. By directly warping the intermediate latent codes of a pretrained diffusion model (rather than performing image-level warping) and combining feature-correspondence guidance with a high-pass filtering strategy, DreamDrone synthesizes high-quality, geometrically consistent, and unbounded scenes.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Perpetual View Generation"
  - "Diffusion Models"
  - "Zero-shot"
  - "Latent Warping"
  - "Scene Generation"
date: 2026-05-08
content_hash: 368f978d4abd9000
---

# DreamDrone: Text-to-Image Diffusion Models are Zero-shot Perpetual View Generators

**Conference**: ECCV 2024  
**arXiv**: [2312.08746](https://arxiv.org/abs/2312.08746)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Perpetual View Generation, Diffusion Models, Zero-shot, Latent Warping, Scene Generation

## TL;DR

This work proposes DreamDrone, a zero-shot, training-free perpetual view generation pipeline. By directly warping the intermediate latent codes of a pretrained diffusion model (rather than performing image-level warping) and combining feature-correspondence guidance with a high-pass filtering strategy, DreamDrone synthesizes high-quality, geometrically consistent, and unbounded scenes.

## Background & Motivation

**Background**: Perpetual View Generation aims to synthesize novel views along an arbitrarily long camera trajectory starting from a single RGBD image. Existing methods mainly follow two paradigms: (a) frame-by-frame warping combined with a refiner (e.g., InfNat series, DiffDreamer), which requires training a refiner on natural scene datasets; and (b) reconstructing 3D point clouds first and then rendering them (e.g., SceneScape, Text2Room), which heavily depends on the quality of 3D models.

**Limitations of Prior Work**:
   - Learning-based refiners only adapt to natural scenes similar to the training data, failing to generalize to arbitrary indoor/outdoor or stylized scenes.
   - Frame-by-frame image warping leads to interpolation blur and distortion, with errors accumulating over frames.
   - 3D reconstruction methods cannot guarantee rendering quality from all viewpoints and fail to achieve "perpetual" generation.

**Key Challenge**: High-quality image generation requires a high level of freedom (e.g., diffusion models), whereas cross-frame geometric consistency demands constrained degrees of freedom, making it difficult to balance the two.

**Goal**: To build a versatile and flexible pipeline for perpetual view generation that spans diverse scene types, supports interactive trajectory control, and maintains high quality and cross-frame consistency.

**Key Insight**: Since diffusion models can generate high-quality images from random latents, warping the latent codes directly (instead of image pixels) followed by denoising allows leveraging the generative power of the diffusion model as a "refiner" while preserving semantic information.

**Core Idea**: Perform view transformation within the latent space of the diffusion model, and employ feature-correspondence guidance during denoising to guarantee geometric consistency.

## Method

### Overall Architecture

Current view RGBD image $\rightarrow$ DDIM inversion to obtain the latent code $x_{t_1}$ at timestep $t_1$ $\rightarrow$ High-pass filtering + warping to generate the next view's latent $x'_{t_1}$ $\rightarrow$ DDPM forward noising to $t_2$ to increase degrees of freedom $\rightarrow$ Denoising process with feature-correspondence guidance and cross-view attention $\rightarrow$ Generate the next view's image $I'$ $\rightarrow$ Iteratively generate infinite frames. The entire pipeline is zero-shot and training-free.

### Key Designs

#### 1. Latent Code Warping + High-Pass Filtering

- **Function**: Warps the intermediate latent codes of the diffusion model from the current view to the next view according to camera parameters while preserving high-frequency details.
- **Mechanism**:
    - Obtains the latent code $x_{t_1}$ of the current frame at timestep $t_1=21$ via DDIM inversion.
    - Applies Fast Fourier Transform (FFT) on $x_{t_1}$ to separate low and high frequencies: $F(x_t) \rightarrow F_{low}, F_{high}$ (with threshold $\sigma=20$).
    - Warps only the low-frequency component: $x_t^{low-warped} = \text{warp}(\text{IFFT}(F_{low}))$.
    - Recombines the components: $x_t' = \text{IFFT}(\text{FFT}(x_t^{low-warped}) + F_{high})$.
    - Warping is performed based on depth details and camera intrinsic/extrinsic parameters (adjusting intrinsics to match the latent resolution).
- **Design Motivation**: Direct warping (whether on images or latents) leads to high-frequency loss and blurriness due to interpolation at non-integer pixel coordinates. Separating with a high-pass filter preserves original high-frequency details and warps only the low-frequency component (which carries geometric structures), effectively mitigating accumulated blur.

#### 2. DDPM Forward Noising

- **Function**: Adds noise to the warped latent $x'_{t_1}$ ($t_1=21$) up to timestep $x'_{t_2}$ ($t_2=441$), expanding the denoising degrees of freedom of the diffusion model.
- **Mechanism**:
    - Directly denoising from $t_1=21$ yields blurry images (since interpolation errors are only slightly corrected).
    - Adding more noise up to $t_2=441$ provides the diffusion model with sufficient space to generate new details and fill unseen regions.
    - The trade-off is a drop in cross-frame consistency, which requires subsequent guidance strategies to compensate.
- **Design Motivation**: To find an optimal balance between image quality (requiring freedom) and cross-frame consistency (requiring constraints).

#### 3. Feature-Correspondence Guidance

- **Function**: Introduces a cross-frame feature similarity gradient guide during the DDIM denoising process to ensure geometric consistency.
- **Mechanism**:
    - Extracts intermediate U-Net features $f_t, f_t'$ for both the current frame and the new frame at each timestep $t$.
    - Computes the cosine distance between the warped features and the new frame features: $\mathcal{L}_{sim}^t = \frac{1 - \cos[\text{warp}(f_t), f_t']}{2}$.
    - Injects the gradient into the denoising process (analogous to classifier guidance): $\hat{\epsilon} = \epsilon_\theta(x_t) - \lambda \sqrt{\bar{\alpha}_{t-1}} \nabla_{x_t} \mathcal{L}_{sim}^t$.
    - Guidance strength is set to $\lambda = 300$.
- **Design Motivation**: Prior research (e.g., DIFT) demonstrates that intermediate features of diffusion models possess strong semantic correspondence. This property is exploited to serve as a supervision signal for cross-frame geometric consistency.

#### 4. Cross-view Self-Attention

- **Function**: Modifies the self-attention module of the U-Net, injecting the Key and Value matrices of the current frame into the attention computation of the new frame.
- **Mechanism**:
    - The current frame undergoes normal self-attention: $o = \text{Softmax}(QK^\top)V$.
    - The new frame utilizes cross-view attention: $o' = \text{Softmax}(Q'K^\top)V$, using the warped $K$ and $V$ matrices of the current frame.
    - Denoising is conducted on both current and new frames simultaneously, sharing attention features.
- **Design Motivation**: Inspired by PnP-Diffusion and video editing works, reference frame features are injected to preserve appearance and semantic consistency.

### Loss & Training

**Completely training-free**. The model leverages pretrained Stable Diffusion 2.1 and MiDaS for depth estimation. Each frame takes approximately 15 seconds to generate on a Titan-RTX GPU.

## Key Experimental Results

### Main Results

| Method | Type | PSNR (32 frames) ↑ | SSIM (32 frames) ↑ | CLIP (32 frames) ↑ |
|------|------|-------------|-------------|-------------|
| InfNat | Training-based | 28.65 | 0.30 | 0.118 |
| InfNat-0 | Training-based | 28.87 | 0.34 | 0.122 |
| CogVideo | Training-based | 29.32 | 0.31 | 0.241 |
| VideoFusion | Training-based | 28.78 | 0.31 | 0.272 |
| T2V-0 | Zero-shot | 26.03 | 0.23 | 0.287 |
| SceneScape | Zero-shot | 29.66 | 0.34 | 0.279 |
| **DreamDrone** | **Zero-shot** | **29.79** | **0.35** | **0.319** |

### Ablation Study

| Configuration | PSNR (32 frames) | SSIM (32 frames) | CLIP (32 frames) | Description |
|------|-----------|-----------|-----------|------|
| warp image | 21.62 | 0.24 | 0.106 | Accumulated blur, collapsed quality |
| warp latent | 28.75 | 0.24 | 0.125 | Still blurry, poor quality |
| +DDPM | 22.59 | 0.06 | 0.308 | Substantially improves quality but reduces consistency |
| +DDPM+guidance | 28.10 | 0.26 | 0.313 | Significantly recovers consistency |
| +cross-view attn | 28.75 | 0.27 | 0.315 | Further improves consistency |
| **+high-pass filter** | **29.79** | **0.35** | **0.319** | **Best performance with all components** |

### Key Findings

- Latent-space warping outperforms image-space warping by better preserving semantic information.
- DDPM noising is key to improving the CLIP score from 0.125 to 0.308, though it heavily disrupts cross-frame consistency.
- Feature-correspondence guidance is core to recovering consistency (PSNR increases from 22.59 to 28.10).
- High-pass filtering brings the largest gain in SSIM (from 0.27 to 0.35), effectively preserving texture details.
- As the number of frames increases, DreamDrone's CLIP score undergoes minimal decay (0.320 to 0.319), whereas learning-based methods show significant degradation.
- **Scene Transition**: By switching text prompts on-the-fly during runtime, smooth transitions to entirely new scene styles can be achieved.

## Highlights & Insights

- **Insight into Latent Warping**: First to propose 3D geometric transformation within the latent space of diffusion models, utilizing the pretrained diffusion model as a "super-refiner" that can both fill in disocclusions and synthesize photorealistic details.
- **Frequency Separation Strategy**: The design of high-pass filtering is elegant and intuitive; it effectively acknowledges that "geometry lies in the low frequencies, while texture details reside in the high frequencies", addressing them with a divide-and-conquer approach.
- **Extremely Versatile**: Requires no training or fine-tuning, generalizes easily to arbitrary styles such as realistic, anime, or Lego scenes.
- **Scene Transition Capability**: Modifying the prompt on-the-fly allows seamless scene transitions, which is a feature learning-based methods are inherently incapable of doing.

## Limitations & Future Work

- The accuracy of depth estimation (MiDaS) directly limits the quality of warping, which may yield errors in complex scenes.
- A generation speed of 15 seconds per frame remains slow, preventing real-time interactive applications.
- Long sequences (>100 frames) might suffer from semantic drift due to the lack of global consistency constraints.
- Performance is optimal primarily for forward-moving camera trajectories; large-angle rotations may introduce artifacts.
- The method is not compared against video diffusion models (e.g., SVD), which might possess strengths in temporal consistency.

## Related Work & Insights

- **vs InfNat/InfNat-0**: Learning-based methods are bounded by their training data distributions (primarily natural landscapes) and fail in stylized or urban environments. DreamDrone overcomes this using the strong generalization capacity of pretrained diffusion models to cover arbitrary domains.
- **vs SceneScape**: Paradigms that reconstruct 3D and then render perform poorly on forward-moving trajectories and outdoor scenes, suffering from bottlenecked reconstruction quality. DreamDrone's frame-by-frame generation is more flexible.
- **vs T2V-0**: While both are zero-shot methods, T2V-0's latent editing strategy disrupts temporal continuity and geometric consistency, limiting it to generating only a few frames. DreamDrone resolves this via feature guidance and cross-view attention.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to introduce latent-space warping for view generation; the frequency separation and feature-guidance designs are highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablation (6 configurations) and comparative studies covering both learning-based and zero-shot baselines, along with creative experiments like scene transition.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, with the necessity of each module systematically verified through ablation.
- Value: ⭐⭐⭐⭐ Combining zero-shot, training-free, and open-domain generation holds immense practical potential, opening secure avenues for the 3D application of diffusion models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] FreeCompose: Generic Zero-Shot Image Composition with Diffusion Prior](freecompose_generic_zero-shot_image_composition_with_diffusion_prior.md)
- [\[ECCV 2024\] OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model](omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)
- [\[ECCV 2024\] MultiGen: Zero-Shot Image Generation from Multi-modal Prompts](multigen_zero-shot_image_generation_from_multi-modal_prompts.md)
- [\[ECCV 2024\] Zero-Shot Detection of AI-Generated Images](zero-shot_detection_of_ai-generated_images.md)
- [\[NeurIPS 2025\] Semantic Surgery: Zero-Shot Concept Erasure in Diffusion Models](../../NeurIPS2025/image_generation/semantic_surgery_zero-shot_concept_erasure_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
