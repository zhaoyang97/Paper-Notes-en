---
title: >-
  [Paper Note] StableAnimator: High-Quality Identity-Preserving Human Image Animation
description: >-
  [CVPR 2025][Image Generation][Human Image Animation] StableAnimator proposes the first end-to-end identity-preserving video diffusion framework. It maintains identity consistency during training via a global content-aware Face Encoder and a distribution-aware ID Adapter, and optimizes facial quality during inference using the Hamilton-Jacobi-Bellman (HJB) equation, enabling the generation of high-fidelity human animation videos without any post-processing tools.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Human Image Animation"
  - "Identity Preservation"
  - "Video Diffusion Models"
  - "HJB Equation"
  - "Face Optimization"
date: 2026-05-08
content_hash: ceaa9698a2a4482b
---

# StableAnimator: High-Quality Identity-Preserving Human Image Animation

**Conference**: CVPR 2025  
**arXiv**: [2411.17697](https://arxiv.org/abs/2411.17697)  
**Code**: [https://francis-rings.github.io/StableAnimator](https://francis-rings.github.io/StableAnimator)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Human Image Animation, Identity Preservation, Video Diffusion Models, HJB Equation, Face Optimization

## TL;DR

StableAnimator proposes the first end-to-end identity-preserving video diffusion framework. It maintains identity consistency during training via a global content-aware Face Encoder and a distribution-aware ID Adapter, and optimizes facial quality during inference using the Hamilton-Jacobi-Bellman (HJB) equation, enabling the generation of high-fidelity human animation videos without any post-processing tools.

## Background & Motivation

**Background**: Diffusion models have achieved great success in image/video generation. Human image animation utilizes pose sequences to drive reference images to generate controllable human animation videos. Representative methods include AnimateAnyone, MagicAnimate, Champ, MimicMotion, ControlNeXt, etc.

**Limitations of Prior Work**: When the pose sequence contains large-scale motion variations, existing methods suffer from severe distortion and inconsistency in facial regions, destroying identity information. Image-domain ID preservation methods (e.g., IP-Adapter-FaceID, InstantID, PuLID) cannot be directly applied to video diffusion models because temporal layers alter spatial distributions, destabilizing spatial priors and causing mismatch between facial embeddings and the distribution of diffusion latent variables. Recent animation models (MimicMotion, ControlNeXt) rely on third-party face-swapping tools like FaceFusion for post-processing, which ruins the original pixel distribution and degrades the overall video quality.

**Key Challenge**: In video diffusion models, temporal modeling layers inevitably alter the spatial feature distribution, whereas image-domain ID preservation methods rely on a stable spatial distribution. This conflict between ID consistency and video fidelity is the core challenge—preserving ID means constraining the spatial distribution, while temporal modeling requires modifying it.

**Goal**: To design an end-to-end solution that ensures ID consistency in both training and inference stages without relying on any post-processing tools.

**Key Insight**: It is observed that the root cause of ID preservation failure is the distribution shift between the facial embedding and the diffusion latent variables after passing through temporal layers. Aligning the distributions of both at each spatial layer can preserve ID information even after temporal modeling.

**Core Idea**: To bridge the distribution gap between facial embeddings and image embeddings during training using a distribution-aware ID Adapter via mean-variance alignment; and during inference, to integrate the solving process of the HJB equation into diffusion denoising, using facial similarity gradients to constrain the denoising path toward the optimal ID consistency direction.

## Method

### Overall Architecture

StableAnimator is based on Stable Video Diffusion (SVD). The reference image enters the model through three paths: (1) encoded into latent codes by VAE, duplicated to match the frame count, and concatenated with the main latent variables; (2) image embeddings extracted by the CLIP Image Encoder are fed into the U-Net cross-attention and the Face Encoder; (3) facial embeddings extracted by ArcFace are sent to the Face Encoder for refinement. PoseNet extracts features from the pose sequence and adds them to the noisy latent variables. Within each U-Net block, the distribution-aware ID Adapter aligns the distributions of facial and image embeddings before temporal modeling. During inference, the HJB equation optimization is embedded at each denoising step to directly optimize the facial similarity of the predicted samples.

### Key Designs

1. **Global Content-Aware Face Encoder**:

    - **Function**: Enhance facial embeddings' awareness of the overall layout of the reference image before injection into the U-Net.
    - **Mechanism**: Facial embeddings interact with reference image embeddings through multiple cross-attention blocks. This allows the facial information to perceive the global context (layout, background, etc.) before being injected into the U-Net, preventing noise from reference image elements irrelevant to the ID from interfering with facial modeling.
    - **Design Motivation**: Directly feeding ArcFace facial embeddings into the U-Net lacks global perception, and irrelevant elements like the background in the reference image can interfere with facial modeling quality.

2. **Distribution-Aware ID Adapter**:

    - **Function**: Ensure video fidelity while maintaining ID consistency.
    - **Mechanism**: In each spatial layer of the U-Net, the diffusion latent variable $z_i$ performs cross-attention with the image embedding and the refined facial embedding, yielding $z_i^{img}$ and $z_i^{face}$ respectively. Their mean and standard deviation are calculated to align their distributions through normalization: $\bar{z}_i^{face} = \frac{z_i^{face} - \mu_{face}}{\sigma_{face}} \times \sigma_{img} + \mu_{img}$, and then the aligned $\bar{z}_i^{face}$ is added to $z_i^{img}$. This alignment is completed before the temporal layers, ensuring that subsequent temporal modeling does not disrupt the ID information.
    - **Design Motivation**: Temporal layers disrupt the spatial distribution. If facial embeddings are directly injected, a distribution shift occurs, causing unstable ID information. Standardizing the distribution of facial embeddings to match that of image embeddings ensures subsequent temporal perturbations affect both consistently.

3. **HJB Equation Face Optimization**:

    - **Function**: Enhance facial quality during inference without training additional components.
    - **Mechanism**: In each denoising step, the predicted sample $x_{pred}$ from the diffusion model is cloned as an optimizable variable $x_{op}$. After decoding via VAE, ArcFace is used to calculate the facial cosine similarity loss with the reference image: $loss = (1 - \text{Cos}(\text{Arc}(f_{pred}), \text{Arc}(y)))$, which is iteratively optimized for 10 steps using the Adam optimizer. The optimized $x_{op}$ replaces the original prediction for subsequent denoising. The authors prove that this process is equivalent to solving the control signal $c_t^* = \gamma = r(x_1 - X_t)/(1+r(1-t))$ of the HJB equation, and its SDE form shares the same structure as the diffusion SDE, allowing seamless integration.
    - **Design Motivation**: Post-processing face-swapping tools operate in a different domain, destroying the pixel distribution and semantic consistency of the original animation. HJB optimization is conducted within the diffusion denoising process, consistently adapting to the distribution of the current denoising latent variables to avoid out-of-domain interference.

### Loss & Training

- Use a reconstruction loss weighted by a facial mask: $\mathcal{L} = \mathbb{E}_\varepsilon(\|(z_{gt} - z_\varepsilon) \odot (1+M)\|^2)$, where $M$ is the facial region mask extracted by ArcFace, assigning double weight to the facial area.
- Trainable components: U-Net, FaceEncoder, PoseNet. The ID Adapter is initialized using the pre-trained spatial cross-attention weights from SVD.
- Trained on 3K internet videos (60-90 seconds), with DWPose extracting skeletons and ArcFace extracting facial embeddings/masks.
- Trained on 4×A100 80G for 20 epochs, batch size 1 per GPU, learning rate 1e-5.

## Key Experimental Results

### Main Results

| Model | CSIM ↑ | FVD ↓ | L1 ↓ | PSNR ↑ |
|------|--------|-------|------|--------|
| AnimateAnyone | 0.457/0.316 | 171.90/383.45 | -/3.15E-4 | 29.56/27.14 |
| Unianimate | 0.479/0.347 | 148.06/394.32 | 2.66/2.82E-4 | 30.77/27.46 |
| MimicMotion | 0.262/0.242 | 326.57/604.13 | 5.85/3.55E-4 | -/22.94 |
| ControlNeXt | 0.360/0.264 | 326.57/389.45 | 6.20/2.90E-4 | -/25.28 |
| **StableAnimator** | **0.831/0.805** | **140.62/349.94** | 2.87/2.71E-4 | 30.81/28.85 |

Results are formatted as TikTok / Unseen100 dataset. CSIM denotes facial cosine similarity (higher is better).

### Ablation Study

| Configuration | CSIM ↑ | FVD ↓ |
|------|--------|-------|
| w/o Face Masks | 0.639 | 382.25 |
| w/o Face Encoder | - | - |
| IP-Adapter Alternative | ID improves but video quality drops drastically |
| FaceFusion Post-processing | Face improves but video fidelity degrades |
| **StableAnimator Full** | **0.805** | **349.94** |

### Key Findings

- StableAnimator outperforms the strongest competitor Unianimate on CSIM by 36.9% (TikTok) and 45.8% (Unseen100), while maintaining the best FVD.
- Directly inserting IP-Adapter improves ID but severely harms video quality and single-frame quality, validating the temporal layer interference hypothesis.
- FaceFusion post-processing improves faces but degrades video fidelity relatively—post-processing tools do not share the same distribution domain with the diffusion model.
- HJB face optimization effectively eliminates detail distortion when performed synchronously with denoising because it always adapts to the current distribution.

## Highlights & Insights

1. "Distribution alignment" is a simple and elegant solution—using mean-variance normalization is sufficient to bridge the distribution gap between image-domain ID methods and video diffusion models.
2. The discovery of the structural equivalence between the HJB equation and the diffusion SDE is highly interesting, allowing face optimization to be seamlessly embedded in the denoising process rather than acting as an external post-processing step.
3. This is the first work to demonstrate that end-to-end ID preservation can completely replace face-swapping post-processing while delivering superior quality.
4. The facial mask weighted loss is a simple but effective design, forcing the model to focus more on the facial region during training.

## Limitations & Future Work

- The training data volume (3K videos) is relatively small; larger-scale data might further improve generalization capabilities.
- HJB facial optimization iterates 10 times at each denoising step, which increases inference time.
- GPU memory requirement of 12.50G is moderate (handling 16 frames of 576×1024).
- Future work can explore extending the ID preservation scheme to more video generation tasks.

## Related Work & Insights

- Compared to IP-Adapter-FaceID, the distribution-aware ID Adapter avoids the distribution shift caused by temporal layers.
- Compared to post-processing with FaceFusion in MimicMotion/ControlNeXt, the end-to-end scheme produces more consistent videos.
- The HJB optimization concept can be generalized to other scenarios requiring constraints during diffusion inference (such as style consistency and semantic alignment).

## Rating

- **Novelty**: 8/10 — Both distribution alignment and HJB facial optimization are compelling innovations with rigorous theoretical derivations.
- **Experimental Thoroughness**: 8/10 — Two datasets + comprehensive ablation + comparisons with 8 baseline methods.
- **Writing Quality**: 8/10 — Clear motivation and complete mathematical derivations.
- **Value**: 8/10 — The first truly end-to-end ID-preserving human image animation scheme, offering significant practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Free-viewpoint Human Animation with Pose-correlated Reference Selection](free-viewpoint_human_animation_with_pose-correlated_reference_selection.md)
- [\[ICML 2025\] FlexiClip: Locality-Preserving Free-Form Character Animation](../../ICML2025/image_generation/flexiclip_locality-preserving_free-form_character_animation.md)
- [\[CVPR 2025\] EmoDubber: Towards High Quality and Emotion Controllable Movie Dubbing](emodubber_towards_high_quality_and_emotion_controllable_movie_dubbing.md)
- [\[CVPR 2026\] Identity-Preserving Image-to-Video Generation via Reward-Guided Optimization](../../CVPR2026/image_generation/identity-preserving_image-to-video_generation_via_reward-guided_optimization.md)
- [\[CVPR 2025\] OmniStyle: Filtering High Quality Style Transfer Data at Scale](omnistyle_filtering_high_quality_style_transfer_data_at_scale.md)

</div>

<!-- RELATED:END -->
