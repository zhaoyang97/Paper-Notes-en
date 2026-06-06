---
title: >-
  [Paper Note] AdaEraser: Training-Free Object Removal via Adaptive Attention Suppression
description: >-
  [ICML 2026][Image Generation][Object Removal] AdaEraser adaptively regulates the self-attention suppression intensity of diffusion models using the "object presence degree." It simultaneously enhances object removal comp…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Object Removal"
  - "Training-Free Editing"
  - "Self-Attention Suppression"
  - "Diffusion Models"
  - "Image Inpainting"
date: 2026-05-08
content_hash: 02e43aeed5e439e5
---

# AdaEraser: Training-Free Object Removal via Adaptive Attention Suppression

**Conference**: ICML 2026  
**arXiv**: [2605.15921](https://arxiv.org/abs/2605.15921)  
**Code**: None  
**Area**: Image Generation / Diffusion Image Editing  
**Keywords**: Object Removal, Training-Free Editing, Self-Attention Suppression, Diffusion Models, Image Inpainting  

## TL;DR
AdaEraser adaptively regulates the self-attention suppression intensity of diffusion models using the "object presence degree." It simultaneously enhances object removal completeness and background reconstruction quality without training a new model, outperforming training-based and training-free object removal methods on Mulan and OABench.

## Background & Motivation
**Background**: Diffusion models have become the mainstream foundation for image generation and editing. Object removal is typically viewed as a special form of inpainting: given an image and a mask, the model must delete the object within the mask while naturally connecting the hole area with the surrounding background.

**Limitations of Prior Work**: Training-based object removal methods rely on specialized datasets, adapters, or fine-tuning, which are costly. Training-free methods attempt to directly utilize the generative prior of pre-trained diffusion models. Strong recent methods like AttentiveEraser block attention from image tokens to target region tokens in self-attention; while effective at removal, this easily disrupts background generation within the mask because background restoration requires global self-attention between regions.

**Key Challenge**: Object removal involves two goals: suppressing the object concept and restoring a reasonable background. Strong suppression favors object removal but leaves the background lacking context; weak suppression retains generative capability but may lead to object residuals. Fixed intensity or uniform suppression across the entire region fails to handle variations across different tokens, timesteps, and layers.

**Goal**: Design a training-free adaptive self-attention modulation method that applies strong suppression when the object is prominent and relaxes suppression as the object disappears, allowing the pre-trained diffusion model to regain its background generation capability.

**Key Insight**: The authors observe that the self-attention map of tokens in the target region gradually reflects semantic content during denoising. The similarity between the attention map of a token in the source reference branch and the removal branch is highly correlated with whether the target concept corresponding to that token still exists.

**Core Idea**: Use the token-wise cosine similarity between the source reference attention map and the current removal process attention map as a presence score, then convert $1-p(i)$ into an adaptive suppression coefficient for each key token.

## Method
AdaEraser does not change diffusion model parameters nor train extra networks. It runs a source reference branch and a target removal branch simultaneously at each denoising step: the source branch adds noise to the source image latent to the same noise level and passes it through the denoising network to obtain reference self-attention maps; the target branch executes object removal. Attention maps from both branches are compared at the same timestep, layer, and token to estimate object residuals.

### Overall Architecture
Given a source image $I^{src}$ and a target mask $M$, the VAE encoder first yields the latent $x_0^{src}$. For each timestep $t$, the source branch constructs $x_t^{src}=\sqrt{\bar\alpha_t}x_0^{src}+\sqrt{1-\bar\alpha_t}\epsilon$ and extracts self-attention maps $SA^{src}_{t,l}$ via the denoising network. The target branch is initialized from the noisy source image, obtaining the current $x_t^{tgt}$ and $SA^{tgt}_{t,l}$ through the same network.

For each token $i$ in the mask, the method calculates $p(i)=Sim(SA^{tgt}_{t,l}(i),SA^{src}_{t,l}(i))$. If the attention map of the target branch remains similar to the source object tokens, the object concept residual is strong; if visibility decreases, the position is more likely background or new content. Subsequently, $\eta(i)=1-p(i)$ is multiplied by the key token weights in the self-attention softmax. Finally, foreground-background blending is maintained using the mask to ensure consistency in non-edited regions.

### Key Designs
1.  **Noise-level Aligned Reference Attention Map**:
    - **Function**: Provides a comparable reference for "what the attention should look like while the object exists" for each timestep.
    - **Mechanism**: Instead of performing full DDIM inversion or taking a fixed noise layer, the source latent at the same noise level is passed through the denoising network at each timestep to extract $SA^{src}_{t,l}$.
    - **Design Motivation**: Attention maps are strongly correlated with noise intensity. If the reference noise level is incorrect, the presence score will be confounded by noise scale differences; level-aligned comparison more stably reflects semantic residuals.

2.  **Token-wise Presence Score**:
    - **Function**: Provides fine-grained estimation of object residuals for different tokens within the mask.
    - **Mechanism**: For token $i$ inside the mask, cosine similarity is computed between flattened target and source attention maps. This score serves as a relative index for control rather than a strict semantic probability.
    - **Design Motivation**: Different local structures within the same object focus on different regions (e.g., head vs. body tokens). Regional averaging erases these differences; a token-wise approach is better suited for local adaptation.

3.  **Adaptive Self-attention Suppression**:
    - **Function**: Suppresses target tokens when residuals are strong and restores generative capacity after deletion.
    - **Mechanism**: For key tokens inside the mask, $\eta(i)=1-p(i)$ is used (with $\eta(i)=1$ elsewhere), and attention is modified to:
    $$\widetilde{SA}(i)=\eta(i)\exp(QK_j^\top/\sqrt d)/\sum_j\eta(j)\exp(QK_j^\top/\sqrt d)$$
    - **Design Motivation**: This acts as a monotonic logit bias for object-related keys. Compared to the hard blocking in AttentiveEraser, it allows for a dynamic compromise between object removal and background reconstruction.

### Loss & Training
AdaEraser is a training-free method with no additional training loss. At inference, it uses the VAE, denoising UNet, and decoder of a pre-trained text-to-image diffusion model. The main experiments use SDXL as the backbone with an empty prompt as the text condition. Overhead comes from parallel denoising of source/target latents and score calculation; the authors handle this through concatenation, maintaining overhead within approximately 15% relative to AttentiveEraser.

## Key Experimental Results

### Main Results
The paper compares training-based and training-free methods on Mulan and OABench benchmarks. AdaEraser achieves SOTA results across FID, LPIPS, PSNR, ReMOVE, CFD, and human ranking (AHR).

| Method | Training | Mulan FID↓ | Mulan PSNR↑ | Mulan ReMOVE↑ | Mulan AHR↑ | OABench FID↓ | OABench PSNR↑ | OABench ReMOVE↑ | OABench AHR↑ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| AttentiveEraser | No | 54.040 | 22.7771 | 0.9000 | 5.46 | 40.373 | 23.2670 | 0.8215 | 5.43 |
| RORem | Yes | 53.470 | 23.5275 | 0.9048 | 6.22 | 39.215 | 23.4126 | 0.8281 | 6.23 |
| OmniPaint | Yes | 59.996 | 21.4493 | 0.8706 | 5.07 | 38.903 | 22.9257 | 0.7991 | 4.59 |
| AdaEraser | No | 51.108 | 23.5871 | 0.9065 | 7.08 | 38.472 | 23.5047 | 0.8316 | 6.81 |

### Ablation Study
Core ablations focus on suppression strategy and reference selection. Results indicate that token-wise adaptation and same-timestep reference are necessary designs.

| Configuration | FID↓ | PSNR↑ | ReMOVE↑ | CFD↓ | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Timestep-based suppression | 38.831 | 23.4697 | 0.8263 | 0.2517 | Linear decay by time only; lacks semantic awareness |
| Region-based suppression | 38.945 | 23.4674 | 0.8261 | 0.2499 | Single score for the whole mask; lacks granularity |
| Token-wise suppression | 38.472 | 23.5047 | 0.8316 | 0.2450 | Ours; best performance |
| Reference $x_1^{src}$ | 38.595 | 23.4262 | 0.8223 | 0.2658 | Fixed low-noise reference is inferior to progression |
| Reference $x_T^{src}$ | 38.829 | 23.4808 | 0.8241 | 0.2507 | Fixed high-noise reference is unstable |
| Reference $x_{T/2}^{src}$ | 38.713 | 23.4872 | 0.8262 | 0.2514 | Mid-noise reference is inferior to same-timestep |
| Reference $x_t^{src}$ | 38.472 | 23.5047 | 0.8316 | 0.2450 | Level alignment yields best presence score |

### Key Findings
- The advantage of AdaEraser stems from better utilization of the internal self-attention dynamics of pre-trained diffusion models rather than new data.
- Compared to AttentiveEraser, inference time increases from 13.98s to 15.41s, and VRAM from 7966 MiB to 9014 MiB, showing limited overhead.
- The presence score decreases gradually over timesteps, with different layers/tokens showing distinct patterns, supporting token-wise adaptation over a global schedule.
- The method is robust to slightly loose masks, but incomplete masks may leave shadows, reflections, or uncovered parts behind.

## Highlights & Insights
- This paper identifies the genuine contradiction in object removal: it is not "the more suppression, the better," but rather suppressing when the object is present and letting go for background generation once it disappears.
- Using self-attention map similarity as a proxy signal is clever because the softmaxed attention maps are comparable across branches, proving more stable than direct object detection in noisy latents.
- The token-wise design avoids a "one-size-fits-all" approach within the mask. This is particularly important for large objects, multi-part objects, or scenes with complex local textures.
- The KL-regularized interpretation provided in the appendix ensures that attention reweighting is not just an engineering trick but can be understood as an attention distribution adjustment with semantic penalties.

## Limitations & Future Work
- The presence score is a heuristic proxy, not a strict semantic probability. Similarity may fail to distinguish targets from backgrounds in cases of identical textures or overlapping similar objects.
- The method relies on mask quality. If the mask misses shadows, reflections, or edges, AdaEraser can only process the explicitly marked area.
- Performance drops on highly distilled few-step diffusion models, as the method relies on the gradual evolution of attention dynamics over multiple denoising steps.
- Future work could integrate automatic mask expansion, structural constraints, or scene-level priors to improve background recovery for complex structures and under-masked scenarios.

## Related Work & Insights
- **vs AttentiveEraser**: AttentiveEraser forcibly blocks target region attention, leading to clean removal but distorted backgrounds; AdaEraser regulates intensity based on residuals, yielding better background quality.
- **vs Training-based methods (RORem, etc.)**: These rely on specialized data/training; AdaEraser outperforms them without training, indicating sufficient object/background priors already exist in pre-trained models.
- **vs text-driven suppression**: Manipulating cross-attention or text embeddings is unstable for small or multiple similar targets; this work looks at image token self-attention for finer localization.
- **Insight**: For training-free diffusion editing, the temporal evolution of internal attention can serve as a control signal, potentially removing the need for external classifiers or segmenters.

## Rating
- Novelty: ⭐⭐⭐⭐ Simple and effective adaptive suppression using token-wise attention similarity.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive analysis including metrics, user study, ablation, efficiency, mask quality, and backbones.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and illustrations; theoretical appendix is explanatory rather than a strict guarantee.
- Value: ⭐⭐⭐⭐⭐ Highly practical for training-free image editing and diffusion attention control.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Precise Object and Effect Removal with Adaptive Target-Aware Attention](../../CVPR2026/image_generation/precise_object_and_effect_removal_with_adaptive_target-aware_attention.md)
- [\[CVPR 2026\] Object-WIPER: Training-Free Object and Associated Effect Removal in Videos](../../CVPR2026/image_generation/object-wiper_training-free_object_and_associated_effect_removal_in_videos.md)
- [\[ICML 2026\] CLEAR: Context-Aware Learning with End-to-End Mask-Free Inference for Adaptive Video Subtitle Removal](clear_context-aware_learning_with_end-to-end_mask-free_inference_for_adaptive_vi.md)
- [\[AAAI 2026\] Melodia: Training-Free Music Editing Guided by Attention Probing in Diffusion Models](../../AAAI2026/image_generation/melodia_training-free_music_editing_guided_by_attention_probing_in_diffusion_mod.md)
- [\[AAAI 2026\] CountSteer: Steering Attention for Object Counting in Diffusion Models](../../AAAI2026/image_generation/countsteer_steering_attention_for_object_counting_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
