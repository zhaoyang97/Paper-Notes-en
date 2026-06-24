---
title: >-
  [Paper Note] MTVCraft: Tokenizing 4D Motion for Arbitrary Character Animation
description: >-
  [ICLR 2026][Video Generation][4D motion tokens] MTVCraft quantizes 3D joint coordinate sequences (4D motion) from driving videos directly into discrete tokens. Combined with a motion-aware DiT featuring 4D positional encoding, it bypasses the pixel-alignment constraints of traditional 2D rendered pose maps, achieving high-quality pose-guided animation for arbitrary characters (including non-human objects).
tags:
  - "ICLR 2026"
  - "Video Generation"
  - "4D motion tokens"
  - "SMPL"
  - "VQVAE"
  - "Diffusion Transformer"
  - "motion attention"
  - "zero-shot generalization"
date: 2026-05-08
content_hash: 58a87258dbfaa7f0
---

# MTVCraft: Tokenizing 4D Motion for Arbitrary Character Animation

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=m7AQM9H6wa](https://openreview.net/forum?id=m7AQM9H6wa)  
**Code**: [https://github.com/DINGYANB/MTVCrafter](https://github.com/DINGYANB/MTVCrafter)  
**Area**: Character Animation / Pose-Guided Video Generation  
**Keywords**: 4D motion tokens, SMPL, VQVAE, Diffusion Transformer, motion attention, zero-shot generalization  

## TL;DR
MTVCraft quantizes 3D joint coordinate sequences (4D motion) from driving videos directly into discrete tokens. Combined with a motion-aware DiT featuring 4D positional encoding, it bypasses the pixel-alignment constraints of traditional 2D rendered pose maps, achieving high-quality pose-guided animation for arbitrary characters (including non-human objects).

## Background & Motivation
**Background**: Character image animation (synthesizing video based on a reference image and a pose sequence from a driving video) has developed rapidly with the explosion of digital human demand. Methods have evolved from early GANs to diffusion models, including AnimateAnyone, MimicMotion, StableAnimator, and UniAnimate-DiT.

**Limitations of Prior Work**: Almost all methods rely on **2D rendered pose maps** (skeleton diagrams, SMPL mesh renderings, depth maps) to provide motion guidance. This introduces two fundamental flaws: first, 2D pose maps discard the rich spatio-temporal motion information of the real 4D world, making it difficult to synthesize physically plausible and expressive complex movements (e.g., gymnastics). Second, when poses are provided as images, models tend to **copy fixed-shape poses pixel-by-pixel** rather than understanding motion semantics. Significant deviations in shape or position between the driving video and the reference character (e.g., Hulk) often result in distortions and artifacts.

**Key Challenge**: 2D rendered pose maps are both the source of guidance signals and the root cause of information loss and rigid pixel alignment—striving for precise control ties the model to pixel alignment, while aiming for generality leads to the loss of 3D geometry.

**Goal**: To skip the intermediate step of 2D rendering and directly model raw 4D motion (3D joint coordinates changing over time), preserving spatio-temporal geometry while decoupling shape from absolute position.

**Core Idea**: **[Motion Tokenization]** Use VQVAE to quantize SMPL joint coordinate sequences into compact, discrete 4D motion tokens. **[Motion-Aware DiT]** Use these tokens as context for visual tokens via motion attention with 4D RoPE, replacing "rendered pose maps + pixel alignment" with "token retrieval + semantic guidance."

## Method

### Overall Architecture
MTVCraft consists of two stages: first, the **4DMoT (4D Motion Tokenizer)** quantizes SMPL joint coordinate sequences extracted from driving videos into discrete motion tokens. Then, the **MV-DiT (Motion-Aware Video DiT)** generates animated videos conditioned on these tokens, using the reference image as an identity anchor. The entire design can be seamlessly integrated into video diffusion backbones of different scales (e.g., CogVideoX-5B converted to 6B version, Wan-2.1-14B converted to 18B version).

```mermaid
flowchart LR
    A[Driving Video] -->|NLF-Pose Estimation| B[SMPL Joint Coordinates<br/>J ∈ f×24×3]
    B -->|Differential + Normalization| C[4DMoT Encoder]
    C --> D[Vector Quantizer<br/>Codebook 8192]
    D --> E[4D Motion Tokens]
    F[Reference Image] -->|3D VAE| G[Visual Latent]
    E --> H[MV-DiT<br/>4D Motion Attention]
    G --> H
    H --> I[Animated Video]
```

### Key Designs

**1. 4DMoT: VQVAE quantization of differential coordinates to decouple motion from shape and position.** The authors deliberately choose to quantize **joint coordinates** rather than SMPL rotation parameters—coordinates naturally align with pixel-level generation, can be written in differential form, and avoid the discontinuity and ambiguity of axis-angle representations. Specifically, forward kinematics are applied to a standard neutral SMPL shape (rather than frame-specific predicted body shapes) to obtain 3D joint coordinates $J_t \in \mathbb{R}^{24\times3}$, stripping individual body shapes at the source. The differential representation $M$ is obtained by subtracting the first frame (rendering the first frame coordinates all zeros), allowing the model to learn relative motion patterns rather than absolute positions. The encoder utilizes residual blocks and average pooling downsampling on a 2D plane formed by the frame axis $f$ and joint axis $j$ to obtain continuous latents. The vector quantizer performs nearest neighbor lookup in a learnable codebook $\{C_n\}_{n=1}^{s}$, maintained with EMA updates and codebook resets. The training objective is reconstruction loss plus commitment loss:

$$L_{vq} = \|M - \hat{M}\|_1 + \beta \|E - \mathrm{sg}[C]\|_2^2$$

where $\mathrm{sg}[\cdot]$ denotes stop-gradient, and $E$ and $C$ are the latents before and after quantization, respectively. The resulting tokens are compact and denoised, compressing "shape/position-agnostic pure motion dynamics" into a discrete space.

**2. 4D Positional Encoding: Extending 3D RoPE to 4D to share geometric semantics between motion and visual tokens.** Standard video DiT visual tokens use 3D RoPE $P_{3D}=\mathrm{Concat}(R_t, R_h, R_w)$. However, motion tokens inherently exist in a "structured 3D space evolving over time." Without aligned geometric semantics, meaningful interaction is impossible. The authors unify the extension as:

$$P_{4D} = \mathrm{Concat}(R_t, R_x, R_y, R_z)$$

For motion tokens, coordinates $(t, x, y, z)$ are used, where $t$ is the frame index and $(x, y, z)$ are the **24 joint positions averaged across all frames in the dataset**. Using average positions provides a stable and unified reference frame, accelerating convergence. For visual tokens lacking a depth axis, $(t, h, w)$ is used with the depth direction set to $z=0$, preserving original 3D RoPE behavior while ensuring compatibility with motion tokens. Ablations show this positional encoding is critical for performance (removing PE causes FVD to jump from 317 to 548).

**3. 4D Motion Attention + repeat-concat identity preservation: Visual as query, motion as key/value.** A motion attention module is inserted every two DiT blocks, allowing visual tokens to actively "retrieve" motion clues:

$$Q = \mathrm{RoPE}(\mathrm{LN}(W_q z_{vision}), P_{4D}^{vision}),\quad K = \mathrm{RoPE}(\mathrm{LN}(W_k z_{motion}), P_{4D}^{motion}),\quad V = \mathrm{LN}(W_v z_{motion})$$

$$\mathrm{Attention}(Q,K,V) = \mathrm{Softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

The attention output is added back to $z_{vision}$ as a residual, performing motion modulation while maintaining spatio-temporal consistency. For identity preservation, an additional reference network is discarded in favor of a minimalist repeat-and-concatenate approach: the reference image latent is copied $f$ times along the temporal dimension and concatenated with the noisy video latent, $z_{vision}=\mathrm{Concat}(z_0, \mathrm{Repeat}(z_{ref}, f))$, relying on the DiT's inherent 3D full self-attention to inject identity information frame-by-frame.

**4. Motion-Aware CFG and Scalability: Increasing control strength with learnable unconditional motion tokens.** Motion tokens have no natural "unconditional" form. The authors introduce a learnable unconditional motion token $c_{mo\varnothing}$, which is randomly substituted during training to jointly learn conditional and unconditional generation. During inference, guidance scales of 3.0 for motion and 6.0 for text are used. When scaling to 18B, 4DMoT and the unconditional tokens are reused without retraining; motion tokens of dimension 3072 are zero-padded to match the 5120 dimensions of Wan-2.1, and an additional text control branch is added, significantly reducing scaling costs.

## Key Experimental Results

### Main Results (TikTok benchmark)

| Model | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | FVD↓ | FID-VID↓ |
|---|---|---|---|---|---|---|
| MimicMotion | 19.30 | 0.751 | 0.220 | 34.88 | 472.51 | 9.30 |
| RealisDance-DiT | 17.55 | 0.717 | 0.261 | 30.39 | 458.81 | - |
| UniAnimate-DiT | 19.35 | 0.765 | 0.235 | 28.47 | 402.14 | 9.12 |
| **MTVCraft-6B** | 19.35 | 0.760 | 0.219 | 23.58 | 317.21 | 8.56 |
| **MTVCraft-18B** | **19.84** | **0.779** | **0.217** | **20.70** | **276.65** | **7.31** |

The 6B version already outperforms the previously strongest UniAnimate-DiT in FID/FVD (FVD 317 vs 402), while the 18B version further reduces FVD to 276, achieving SOTA across all metrics.

### Ablation Study (TikTok, FID↓ / FVD↓)

| Component | Variant | FID↓ | FVD↓ |
|---|---|---|---|
| 4D Motion Tokenizer | w/o quantize | 24.04 | 332.97 |
| | w/o differential motion | 24.37 | 325.40 |
| | w/ 3D quantization (no z) | 23.94 | 329.86 |
| 4D Motion Attention | w/ dynamic PE | 28.24 | 383.22 |
| | w/ learnable PE | 28.69 | 397.64 |
| | w/ 1D temporal RoPE | 29.45 | 458.29 |
| | w/o PE | 32.56 | 548.31 |
| **Default** | — | **23.58** | **317.21** |

### Key Findings
- **Quantization and Differential are Indispensable**: Removing quantization (degrading to a standard autoencoder) or differential representation lead to significant performance drops, proving that discrete tokens and relative displacement are key to stable motion learning.
- **Z-axis Utility**: Adding the 4D dimension (including depth) to 3D quantization brings consistent improvements, indicating that depth geometric information contributes to animation quality.
- **Positional Encoding is Critical**: Removing PE entirely causes FVD to soar from 317 to 548. Any reduced-dimension RoPE (1D/2D/3D) is inferior to full 4D.
- **Strong Zero-Shot Generalization**: Although trained only on human data, the model can drive animals and inanimate objects. It remains robust even when target poses are severely mismatched with the reference character (e.g., an owl), whereas baseline methods fail—validating the effectiveness of decoupling motion from driving videos.

## Highlights & Insights
- **Paradigm Shift**: The first character animation framework to replace "2D rendered pose map guidance" with "raw 4D motion token guidance," fundamentally addressing rigid pixel alignment and 3D information loss.
- **Insightful Choice of Coordinates vs. Parameters**: Quantizing joint coordinates instead of SMPL rotation parameters aligns with pixel generation, avoids axis-angle discontinuities, and decouples position through differential forms—solving multiple problems with one choice.
- **Unified Geometric Semantics**: Using 4D RoPE to bring motion and visual tokens into the same coordinate system is key to making cross-modal attention "speak the same geometric language," strongly supported by ablation data.
- **Engineering Friendly**: The repeat-concat identity preservation eliminates the need for a reference network, and 4DMoT reuse across backbones avoids retraining. It has been commercially deployed, demonstrating high practicality.

## Limitations & Future Work
- **Dependence on SMPL Estimation Quality**: Motion originates from SMPL estimated by NLF-Pose; upstream errors during occlusion, extreme viewpoints, or non-standard human topologies propagate to the animation.
- **Human Skeleton Centric**: While showing zero-shot capabilities for non-human objects, the motion representation remains tied to the 24-joint human SMPL, lacking native skeletons for quadrupeds, multi-limbed, or soft-body objects.
- **Human-Centric Training Data**: The 30K high-quality SMPL-video pairs are predominantly human; motion for more generalized object/scene interactions remains to be expanded.
- **Dataset Average Joint Positions for PE**: While stable, this is somewhat static. Future work could explore a better compromise between stability and individual pose adaptation.

## Related Work & Insights
- **Tokenization of Motion Generation**: Borrowing ideas from T2M-GPT and MotionGPT that quantize SMPL parameters for motion generation, this work shifts to coordinate quantization for video generation—a cross-task migration of tokenization ideas.
- **Pose-Guided Animation**: Compared to routes like AnimateAnyone and UniAnimate-DiT that rely on 2D pose maps, this paper provides a new direction of "no rendering, direct tokenization."
- **Controllable Diffusion**: Inherits structural control from ControlNet/ControlNeXt but replaces 2D image control signals with 4D discrete tokens, inspiring future tokenization of more 3D/4D physical signals into generative models.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The first framework to directly tokenize 4D motion for character animation, representing a paradigm innovation. Coordinate quantization, 4D RoPE, and motion attention are self-consistent and insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ SOTA on two benchmarks, ablations covering quantization/differential/various PE, and rich zero-shot cases; slightly lacks larger-scale quantitative generalization evaluation and failure case analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation (2D vs 4D) and architecture diagrams, logical progression, and well-explained formulas and design motivations.
- **Value**: ⭐⭐⭐⭐⭐ Addresses real pain points, commercially deployed, opens a new direction for pose-guided generation, highly scalable, and high reuse value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MotionWeaver: Holistic 4D-Anchored Framework for Multi-Humanoid Image Animation](motionweaver_holistic_4d-anchored_framework_for_multi-humanoid_image_animation.md)
- [\[CVPR 2026\] LottieGPT: Tokenizing Vector Animation for Autoregressive Generation](../../CVPR2026/video_generation/lottiegpt_vector_animation_generation.md)
- [\[ICLR 2026\] Arbitrary Generative Video Interpolation](arbitrary_generative_video_interpolation.md)
- [\[CVPR 2026\] One-to-All Animation: Alignment-Free Character Animation and Image Pose Transfer](../../CVPR2026/video_generation/one-to-all_animation_alignment-free_character_animation_and_image_pose_transfer.md)
- [\[ICLR 2026\] Animating the Uncaptured: Humanoid Mesh Animation with Video Diffusion Models](animating_the_uncaptured_humanoid_mesh_animation_with_video_diffusion_models.md)

</div>

<!-- RELATED:END -->
