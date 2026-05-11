---
title: >-
  [Paper Note] Preserving Source Video Realism: High-Fidelity Face Swapping for Cinematic Quality
description: >-
  [CVPR 2026][Image Generation][Face swapping] This paper proposes LivingSwap, the first video reference-guided face swapping model. Through a controllable pipeline of keyframe identity injection…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Face swapping"
  - "video reference guidance"
  - "keyframe injection"
  - "temporal stitching"
  - "cinematic quality"
date: 2026-05-08
content_hash: c66e5bc04ddc1b09
---

# Preserving Source Video Realism: High-Fidelity Face Swapping for Cinematic Quality

**Conference**: CVPR 2026
**arXiv**: [2512.07951](https://arxiv.org/abs/2512.07951)
**Code**: [Project Page](https://aim-uofa.github.io/LivingSwap)
**Area**: Diffusion Models / Video Editing
**Keywords**: Face swapping, video reference guidance, keyframe injection, temporal stitching, cinematic quality

## TL;DR

This paper proposes LivingSwap, the first video reference-guided face swapping model. Through a controllable pipeline of keyframe identity injection, source video reference completion, and temporal stitching, it achieves high-fidelity face swapping in long videos. The method stably injects the target identity while preserving expression, lighting, and motion details from the source video, reducing manual editing effort by 40×.

## Background & Motivation

1. **Background**: Video face swapping is in high demand for film and television production. Existing methods fall into two categories: GAN-based methods process frames individually with strong identity injection capability but suffer from poor realism and temporal flickering; diffusion-based inpainting methods mask the original face region and regenerate it, achieving better temporal consistency but discarding original pixel information, leading to degraded fidelity (expressions, lighting, and subtle textures cannot be perfectly reconstructed).

2. **Limitations of Prior Work**: Diffusion inpainting methods rely on external encoders to extract intermediate representations (e.g., landmarks, 3D faces), inevitably losing rich information from the source video. GAN-based methods, while capable of using full source frames to retain more detail, suffer from severe temporal inconsistency on long sequences. Recent reference-guided generation has achieved breakthroughs in image editing, enabling simultaneous editing flexibility and high-fidelity reconstruction.

3. **Key Challenge**: How can one stably and consistently inject the target identity while preserving the rich visual attributes of the source video (lighting, expression, subtle dynamics), in a manner applicable to long video scenarios?

4. **Goal**: (1) Stable and consistent identity injection in long videos; (2) high-fidelity preservation of non-identity attributes (lighting, expression, background); (3) temporal consistency across video segments; (4) overcoming scarcity of training data.

5. **Key Insight**: Drawing inspiration from reference-guided image editing — rather than masking the original face and inpainting, the source video is directly used as visual reference to guide the diffusion model. Long-video face swapping is decomposed into a controllable pipeline of keyframe editing → video reference completion → temporal stitching, with a pair reversal strategy to address training data scarcity.

6. **Core Idea**: The source video itself serves as the reference to guide the diffusion model for face swapping. Keyframes provide identity conditioning, video references preserve non-identity details, temporal stitching handles long videos, and pair reversal resolves the training data problem.

## Method

### Overall Architecture

LivingSwap decomposes long-video face swapping into a four-step controllable pipeline: (1) select representative keyframes and perform single-frame face swapping using an image-level method (e.g., Inswapper); (2) use the source video segment as reference and the keyframe swap results as identity conditions, allowing a video diffusion model to complete intermediate frames; (3) apply a temporal stitching strategy to seamlessly connect adjacent segments for long video processing; (4) construct the Face2Face training dataset using reversed data pairs to provide reliable supervision. The overall framework is based on a 14B-parameter DiT video generation model (VACE architecture) trained with a Rectified Flow objective.

### Key Designs

1. **Keyframe Identity Injection**:

    - **Function**: Provides stable and consistent target identity conditioning for long videos.
    - **Mechanism**: Keyframes $F_{\text{key}}$ are selected from the source video at moments with significant variation in pose, expression, or lighting. High-quality image-level face swapping (Inswapper) is applied to these keyframes, with optional manual Photoshop refinement. Each pair of adjacent keyframes $\{f_{k_i}^{\text{swap-in}}, f_{k_{i+1}}^{\text{swap-in}}\}$ serves as temporal boundary conditions guiding the diffusion model for one video segment. Compared to industrial pipelines that process every frame, this approach only requires processing keyframes, substantially improving efficiency.
    - **Design Motivation**: Image-level face swapping methods achieve far superior identity injection on single frames compared to video-level methods (stronger and more precise). The keyframe strategy combines the advantages of both — image methods ensure identity accuracy at keyframes, while the video diffusion model ensures inter-frame consistency.

2. **Video Reference Completion**:

    - **Function**: High-fidelity preservation of non-identity attributes from the source video while injecting the target identity.
    - **Mechanism**: Unlike inpainting methods that discard original pixels by masking the face region, LivingSwap directly feeds the complete source video segment $V_s^{[k_i:k_{i+1}]}$ as visual reference to the model. All inputs (target identity image, keyframe swap results, source video segment, next keyframe) are VAE-encoded and concatenated in temporal order as condition tokens $Z_c$, then concatenated with binary masks along the channel dimension for spatial localization. An attribute encoder (DiT block structure, layer-matched to the backbone) is introduced, with each layer's output injected into the corresponding backbone layer via element-wise addition: $X^{(l+1)} = \mathcal{D}_\theta^{(l)}(X^{(l)} + \mathcal{A}_\psi^{(h)}(Z_c^{(h)}, M))$.
    - **Design Motivation**: The core advantage of reference-guided generation is that original pixel information is never discarded. The layer-wise injection design of the attribute encoder enables the model to adaptively integrate pixel-level details — including lighting, expression, and texture — from the source video without disrupting the pretrained prior.

3. **Temporal Stitching**:

    - **Function**: Handles videos of arbitrary length and ensures seamless transitions between segments.
    - **Mechanism**: The long video is divided into fixed-length segments (81 frames) by keyframes and generated sequentially. The key design is as follows: the first and last guidance frames of the first segment are directly taken from keyframe swap results; for subsequent segments, the last output frame of the previous segment $f_{k_i}^{\text{swap-out}}$ serves as the starting guidance, while the ending guidance remains the keyframe. Formally: $\{f_t^{\text{swap-out}}\}_{t=k_i}^{k_{i+1}} = \mathcal{D}_{\theta,\psi}(f_{k_i}^{\text{swap-out}}, f_{k_{i+1}}^{\text{swap-in}}, V_s^{[k_i:k_{i+1}]}, I_{\text{tar}}, M)$. Engineering techniques such as frame interpolation, temporal reversal, and frame skipping are also employed to accommodate diverse video tempos.
    - **Design Motivation**: Independently generating each segment produces frame discontinuities at boundaries. Using the output frame of the previous segment as the starting condition for the next propagates consistent identity information while suppressing cross-segment error accumulation.

4. **Face2Face Dataset and Pair Reversal**:

    - **Function**: Resolves the scarcity of training data for video reference-guided face swapping.
    - **Mechanism**: Based on CelebV-Text (70K videos) and VFHQ (16K videos), frame-level face swap results are generated using Inswapper. The key innovation is role reversal in the data pairs — the GAN-generated swap video serves as the model input $V_s$, while the original unedited video serves as the ground truth and keyframe source. This ensures that reference frames and GT frames share the same identity, providing artifact-free, high-quality supervision signals.
    - **Design Motivation**: Using GAN swap results directly as GT would encode GAN artifacts and temporal inconsistencies into the supervision signal. After pair reversal, the GT becomes clean original video, and GAN noise only appears on the input side. Combined with the strong prior of the pretrained model, LivingSwap learns to correct degradation in the input, ultimately producing outputs that surpass the quality of the training data itself.

### Loss & Training

Under the Rectified Flow framework, the loss is the MSE between the predicted and ground-truth velocity fields: $\mathcal{L} = \mathbb{E}_{x_0,x_1,c,t}\|u(x_t,c,t;\theta) - v_t\|^2$, where $v_t = x_1 - x_0$. The AdamW optimizer is used with a learning rate of 1e-5, batch size of 16, resolution of 640 at 81 frames, trained on 8×H200 GPUs for approximately 14 days.

## Key Experimental Results

### Main Results

Comparison on CineFaceBench (400 cinematic scene test pairs, with easy/hard identity splits):

| Method | ID Sim↑ (easy/hard) | Expr↓ (easy/hard) | Light↓ (easy/hard) | Pose↓ (easy/hard) | FVD↓ (easy/hard) | Avg Rank↓ |
|--------|---------------------|-------------------|--------------------|-------------------|------------------|-----------|
| Inswapper | 0.567/0.422 | 2.081/2.607 | 0.189/0.243 | 3.421/3.916 | 66.62/73.48 | 2.500 |
| BlendFace | 0.482/0.315 | 1.919/2.285 | 0.245/0.271 | 4.450/4.520 | 100.28/106.58 | 3.583 |
| **LivingSwap** | **0.532/0.367** | **1.943/2.471** | **0.192/0.238** | **3.108/3.399** | **54.32/63.97** | **1.667** |

On FF++, LivingSwap achieves an Avg Rank of 3.17 (second only to Inswapper) while achieving the best performance on Pose and FVD.

### Ablation Study

| Configuration | ID Sim↑ | Expr↓ | Light↓ | Pose↓ |
|---------------|---------|-------|--------|-------|
| LivingSwap (full) | 0.536 | 2.84 | 0.285 | 2.84 |
| w/o Target Image | 0.515 | 2.74 | 0.279 | 2.80 |
| w/o Keyframe | 0.281 | 2.47 | 0.249 | 2.84 |
| Inpainting instead of reference guidance | 0.519 | 2.89 | 0.292 | 2.87 |
| VACE baseline | 0.313 | 3.08 | 0.355 | 6.42 |

Data quality ablation:

| Data | ID Sim↑ | Expr↓ | Light↓ |
|------|---------|-------|--------|
| All data | 0.536 | 2.84 | 0.285 |
| Top 30% quality only | 0.532 | 2.82 | 0.289 |
| Bottom 30% quality only | 0.540 | 2.83 | 0.288 |

### Key Findings

- **Keyframes are central to identity injection**: Removing keyframes causes ID Similarity to drop sharply from 0.536 to 0.281, demonstrating that the target image alone is insufficient for stably conveying identity in video.
- **Video reference outperforms inpainting**: Replacing reference guidance with inpainting leads to across-the-board degradation on fidelity metrics (Light, Expr, Pose), validating the importance of preserving original pixel information.
- **The model is highly robust to data noise**: Models trained on the lowest-quality 30% of data achieve nearly identical performance to those trained on the full dataset, with a marginal advantage in ID Sim. Data diversity matters more than data quality.
- **Pair reversal enables the model to surpass training data quality**: LivingSwap outputs surpass the Inswapper results in the training data in terms of realism and temporal consistency, demonstrating that the pretrained prior combined with pair reversal allows the model to learn to correct GAN artifacts.
- Compared to Inswapper (used for keyframe generation), LivingSwap is significantly superior on Pose and FVD, confirming that the temporal modeling capability of video diffusion models effectively compensates for the temporal limitations of image-level methods.

## Highlights & Insights

- **Pair Reversal**: This is the most elegant design in the paper. By using GAN outputs as model inputs and original videos as GT, it cleverly avoids the vicious cycle of "GAN artifacts contaminating GT." This idea generalizes to any reference-guided generation task that requires synthetic training data.
- **Decomposition strategy of keyframe → reference completion → stitching**: This reduces the complex problem of long-video face swapping to multiple tractable sub-problems, each with a mature solution (image-level face swapping, video completion, segment stitching). The combined result substantially outperforms end-to-end approaches. This divide-and-conquer engineering philosophy is broadly instructive.
- **40× reduction in manual effort**: With only the first and last 2 keyframes requiring manual editing per 81-frame segment, the approach is highly attractive for production deployment in the film and television industry.

## Limitations & Future Work

- Keyframe selection currently uses a fixed-interval strategy (every 79 frames) without adaptive adjustment based on video content, which may be insufficiently dense for scenes with rapid motion.
- The method relies on Inswapper for keyframe generation, and its quality ceiling constrains the upper bound of identity fidelity in the final results.
- The easy/hard split in CineFaceBench is based on identity similarity scores and may not cover all challenging real-world scenarios (e.g., extreme occlusion, blur, micro-expressions).
- The face detection → cropping → swapping → blending pipeline may introduce stitching artifacts at crop boundaries.
- Extension of video reference guidance to full-body replacement or non-face region editing has not been explored.

## Related Work & Insights

- **vs. Inswapper**: Inswapper serves as an upstream tool in this work (for keyframe generation and data construction), but its frame-by-frame processing causes severe flickering on long sequences. LivingSwap overcomes this through the temporal modeling capability of video diffusion models.
- **vs. DiffSwap**: DiffSwap applies diffusion models to face swapping but achieves low ID Sim (0.261), indicating that naive diffusion inpainting is insufficient for identity preservation. LivingSwap's keyframe + reference guidance design significantly improves identity retention.
- **vs. VACE**: VACE is the backbone model of this work, but directly applying VACE to face swapping yields an ID Sim of only 0.313. LivingSwap's keyframe injection and reference guidance design raises this to 0.536.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First video reference-guided face swapping model with a clever pair reversal strategy, though individual components (keyframes, reference guidance, stitching) each build on existing ideas.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Dual benchmarks on FF++ and the proposed CineFaceBench, extensive ablations (data quality, model components, keyframe quality) — comprehensive and convincing.
- **Writing Quality**: ⭐⭐⭐⭐ The pipeline visualization is clear, though certain details (e.g., the attribute encoder architecture) could be presented more thoroughly.
- **Value**: ⭐⭐⭐⭐⭐ The 40× reduction in manual effort carries exceptional industrial value for film production, and the pair reversal strategy is broadly reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] High-Fidelity Diffusion Face Swapping with ID-Constrained Facial Conditioning](high-fidelity_diffusion_face_swapping_with_id-constrained_facial_conditioning.md)
- [\[CVPR 2026\] APPLE: Attribute-Preserving Pseudo-Labeling for Diffusion-Based Face Swapping](apple_attribute-preserving_pseudo-labeling_for_diffusion-based_face_swapping.md)
- [\[CVPR 2026\] Cinematic Audio Source Separation Using Visual Cues](cinematic_audio_source_separation_using_visual_cues.md)
- [\[CVPR 2026\] HiFi-Inpaint: Towards High-Fidelity Reference-Based Inpainting for Generating Detail-Preserving Human-Product Images](hifi-inpaint_towards_high-fidelity_reference-based_inpainting_for_generating_det.md)
- [\[CVPR 2026\] EffectErase: Joint Video Object Removal and Insertion for High-Quality Effect Erasing](effecterase_joint_video_object_removal_and_insertion_for_high-quality_effect_era.md)

</div>

<!-- RELATED:END -->
