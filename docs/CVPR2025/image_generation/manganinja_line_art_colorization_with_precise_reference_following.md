---
title: >-
  [Paper Note] MangaNinja: Line Art Colorization with Precise Reference Following
description: >-
  [CVPR 2025][Image Generation][Line Art Colorization] MangaNinja is a diffusion-based reference-guided line art colorization method. By training the model to learn local semantic matching capabilities via a progressive patch shuffling strategy and introducing a PointNet-driven point control mechanism for precise color correspondence, it significantly outperforms existing methods in challenging scenarios such as large pose variations, multi-reference inputs…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Line Art Colorization"
  - "Diffusion Models"
  - "Reference Guidance"
  - "Point Control"
  - "Patch Shuffling"
date: 2026-05-08
content_hash: 4a9d2c07686703af
---

# MangaNinja: Line Art Colorization with Precise Reference Following

**Conference**: CVPR 2025  
**arXiv**: [2501.08332](https://arxiv.org/abs/2501.08332)  
**Code**: Not yet released  
**Area**: Image Generation / Anime Colorization  
**Keywords**: Line Art Colorization, Diffusion Models, Reference Guidance, Point Control, Patch Shuffling

## TL;DR
MangaNinja is a diffusion-based reference-guided line art colorization method. By training the model to learn local semantic matching capabilities via a progressive patch shuffling strategy and introducing a PointNet-driven point control mechanism for precise color correspondence, it significantly outperforms existing methods in challenging scenarios such as large pose variations, multi-reference inputs, and cross-character colorization.

## Background & Motivation

1. **Background**: Line art colorization is a core step in manga and anime production. Reference-guided colorization methods (coloring a line art given a colored reference image) maintain character consistency better than text- or stroke-guided methods.
2. **Limitations of Prior Work**: (1) Existing methods (e.g., BasicPBC, AnimeDiffusion) tend to produce semantic mismatches and color mixing when there are large differences between the reference image and the line art (different poses or perspectives); (2) They typically require high similarity between the reference image and the line art, which is impractical in real-world applications; (3) They lack fine-grained control capabilities—unable to specify "this region of the line art should correspond to that color in the reference image".
3. **Key Challenge**: Models tend to learn global style transfer rather than local semantic matching. When the overall structure of the reference image is similar to the line art, simple global matching suffices; however, in practical scenarios, poses, perspectives, and details are often vastly different.
4. **Goal**: (1) Enable the model to learn precise local semantic matching capabilities, ensuring correct correspondences even under large differences between the reference and line art; (2) Provide interactive fine-grained control, allowing users to guide colorization in difficult areas through point correspondences.
5. **Key Insight**: Utilizing anime video frame pairs as training data (which naturally provide correspondences of the same character across different poses), and breaking the overall structure of the reference image via progressive patch shuffling to force the model to learn local matching instead of global copying.
6. **Core Idea**: Shuffling the reference image into patches and randomly arranging them as training input to force the model to learn patch-level local matching capabilities, complemented by point control for precise colorization guidance.

## Method

### Overall Architecture
Dual-branch U-Net architecture: Reference U-Net encodes reference image features, and Denoising U-Net takes the line art and noise as input for denoising generation. Feature injection between the two U-Nets is achieved through K/V concatenation in the self-attention layer ($\text{Attn} = \text{softmax}(\frac{Q_{tar}[K_{tar}, K_{ref}]^T}{\sqrt{d}})[V_{tar}, V_{ref}]$). Training data is sourced from anime videos, where two frames are randomly sampled—one used as the reference and the other to extract line art as the target.

### Key Designs

1. **Progressive Patch Shuffling (Core Innovation)**:
    - **Function**: Forces the model to learn local semantic matching capabilities instead of global style transfer.
    - **Mechanism**: During training, the reference image is partitioned into a patch grid (ranging from 2x2 to 32x32), and their positions are randomly shuffled before being fed into the Reference U-Net. Since the overall structure of the reference image is disrupted, the model cannot rely on global spatial layout and must learn to match corresponding parts of the line art based on local semantic content (color, texture, shape). The number of patches is progressively increased during training (from coarse to fine) to incrementally develop matching capabilities from global to local levels.
    - **Design Motivation**: Without patch shuffling, the model tends toward simple global style transfer (since training frame pairs typically have similar structures). DINO similarity improved from 64.13 to 67.78 (+3.65), and point control only becomes truly effective after local matching is learned.

2. **PointNet-driven Point Control Mechanism**:
    - **Function**: Allows users to achieve fine-grained color control by annotating corresponding point pairs.
    - **Mechanism**: Users annotate matching points on both the reference image and the line art. Each pair of matching points is assigned the same unique integer value on their respective single-channel point maps. These are encoded into multi-scale embeddings $E_{tar}$ and $E_{ref}$ using PointNet (multi-layer Conv + SiLU), which are then injected into the cross-attention Q and K: $Q'_{tar} = Q_{tar} + E_{tar}$, $K'_{ref} = K_{ref} + E_{ref}$. During training, 0-24 pairs of matching points are randomly selected (automatically extracted using LightGlue), while users can choose to bypass point control or manually annotate points during inference.
    - **Design Motivation**: Purely automatic matching remains ambiguous in difficult scenarios (e.g., nose shadows, small-area accessory patterns, multi-character compositions). Point control provides an accurate channel for user intervention.

3. **Multi-Classifier-Free Guidance (Multi-CFG)**:
    - **Function**: Individually controls the guidance strength of the reference image and point control.
    - **Mechanism**: Three conditional combinations of noise prediction are defined: unconditional $\epsilon(\emptyset, \emptyset)$, reference-only $\epsilon(c_{ref}, \emptyset)$, and reference + points $\epsilon(c_{ref}, c_{points})$. Two weights, $\omega_{ref}$ and $\omega_{points}$, independently control the intensity of the two guidelines. Increasing $\omega_{ref}$ enhances automatic matching capability, while increasing $\omega_{points}$ enhances point control accuracy.
    - **Design Motivation**: Different task scenarios require different trade-offs—simple scenarios only need automatic matching with high $\omega_{ref}$, whereas complex scenarios (e.g., cross-character colorization) require high $\omega_{points}$ for precise control.

### Loss & Training
Two-stage training: In the first stage (180k steps), Reference U-Net + Denoising U-Net + PointNet are trained jointly, randomly dropping reference and point signals for unconditional generation training (condition dropping). In the second stage (20k steps), only PointNet is trained to enhance point control. Pre-trained SD 1.5 weights are used to initialize both U-Nets. Training data: 300k video clips filtered from the sakuga-42m dataset. Training was completed in one day using 8×A100-80G GPUs.

## Key Experimental Results

### Main Results

| Method | DINO ↑ | CLIP ↑ | PSNR ↑ | MS-SSIM ↑ | LPIPS ↓ |
|------|--------|--------|--------|----------|---------|
| BasicPBC | 42.64 | 79.64 | 17.58 | 0.894 | 0.33 |
| IP-Adapter | 55.42 | 82.39 | 16.19 | 0.845 | 0.30 |
| AnyDoor* (w/ mask) | 63.79 | 83.91 | 16.24 | 0.874 | 0.27 |
| MangaNinja (w/o points) | 68.23 | 88.34 | 20.37 | 0.962 | 0.22 |
| MangaNinja (w/ points) | 69.91 | 90.02 | 21.34 | 0.972 | 0.21 |

MangaNinja leads substantially across all metrics, particularly in DINO similarity (+6.12 vs AnyDoor*) and PSNR (+5.10 vs AnyDoor*).

### Ablation Study

| Configuration | DINO ↑ | CLIP ↑ | PSNR ↑ | MSE ↓ |
|------|--------|--------|--------|------|
| Base model | 64.13 | 85.05 | 18.12 | 0.0151 |
| + condition dropping | 64.92 | 85.44 | 19.02 | 0.0125 |
| + progressive patch shuffle | 67.78 | 87.42 | 20.18 | 0.0091 |
| + multi CFG | 64.63 | 86.02 | 18.74 | 0.0133 |
| + two-stage training | 64.32 | 86.34 | 19.36 | 0.0113 |
| Full model | 69.91 | 90.02 | 21.34 | 0.0072 |

### Key Findings
- **Progressive Patch Shuffling contributes the most**: DINO +3.65, PSNR +2.06, and MSE is halved (0.0151 $\to$ 0.0091), making it the most critical design.
- **Point control is only effective after local matching is learned**: Adding point control to the base model yields minimal improvement, but adding point control after patch shuffling shows significant effects.
- **Condition dropping independently contributes to automatic matching**: Even without point guidance, it improves DINO by +0.79.
- **Multi-reference fusion** and **cross-character colorization** are unique capabilities of MangaNinja, which cannot be achieved by existing methods.

## Highlights & Insights
- **The "push out of comfort zone" training philosophy of Patch Shuffling**: By deliberately disrupting the global structure of the input, the model is forced to learn more robust local matching. This concept can be transferred to any task requiring semantic correspondence learning (such as image editing, virtual try-on, and pose transfer).
- **Layered design of point control**: PointNet embeddings are added to the Q/K of the attention layer rather than directly being appended as additional tokens. This neither disrupts the original attention information flow nor fails to precisely guide correspondences. This attention modulation is more elegant than standard cross-attention.
- **Video frame pairs as self-supervised correspondence data**: Anime videos naturally provide correspondences of the same character across different frames, bypassing manual annotation. This paradigm can be transferred to fields like fashion (outfit video frame pairs) and autonomous driving (continuous frame correspondences).

## Limitations & Future Work
- Resolution limitations based on SD 1.5 (512×512), preventing the processing of high-resolution manga line art.
- The training data is sourced entirely from anime videos, which may lead to insufficient generalization for line art colorization of real-world photos.
- Although point control is precise, it still requires manual annotation. Future work can combine it with automatic matching (e.g., DINOv2 feature matching) to achieve automated fine-grained correspondence.
- The paper does not evaluate colorization performance on complex background scenes (the test sets only consist of cropped foreground characters).
- How to automatically resolve conflicts during multi-reference fusion (e.g., different colors for the same part in two reference images) is not discussed in depth.

## Related Work & Insights
- **vs BasicPBC**: BasicPBC is a non-generative method that samples color from neighboring regions of the reference image, performing poorly under large reference-to-line-art discrepancies. MangaNinja's generative matching capability far outperforms it.
- **vs IP-Adapter / AnyDoor**: Both are image-conditioned generation methods based on diffusion models, but they lack fine-grained matching capabilities, limiting them to coarse-grained style transfer and causing severe color confusion.
- **vs AnimeDiffusion**: Also an anime colorization method, but it only supports simple scenarios, whereas MangaNinja's patch shuffling + point control enables it to far outperform it in complex scenes.
- **Intersection with visual correspondence**: Patch shuffling essentially reinforces the implicit visual correspondence capabilities of the diffusion model during training, aligning with recent research trends in diffusion features for correspondence.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The patch shuffling training strategy and the point control mechanism are elegantly designed, together solving long-standing bottlenecks.
- Experimental Thoroughness: ⭐⭐⭐⭐ A dedicated evaluation benchmark was constructed, ablation studies are comprehensive, and various challenging scenarios are thoroughly demonstrated.
- Writing Quality: ⭐⭐⭐⭐ The diagrams are intuitive and rich, and the methodology is clearly presented.
- Value: ⭐⭐⭐⭐⭐ Directly practical for the anime/manga industry; capabilities like multi-reference fusion and cross-character colorization open up new application scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Image Referenced Sketch Colorization Based on Animation Creation Workflow](image_referenced_sketch_colorization_based_on_animation_creation_workflow.md)
- [\[CVPR 2025\] The Art of Deception: Color Visual Illusions and Diffusion Models](the_art_of_deception_color_visual_illusions_and_diffusion_models.md)
- [\[CVPR 2025\] InsightEdit: Towards Better Instruction Following for Image Editing](insightedit_towards_better_instruction_following_for_image_editing.md)
- [\[CVPR 2026\] Towards High-resolution and Disentangled Reference-based Sketch Colorization](../../CVPR2026/image_generation/towards_high-resolution_and_disentangled_reference-based_sketch_colorization.md)
- [\[CVPR 2025\] Free-viewpoint Human Animation with Pose-correlated Reference Selection](free-viewpoint_human_animation_with_pose-correlated_reference_selection.md)

</div>

<!-- RELATED:END -->
