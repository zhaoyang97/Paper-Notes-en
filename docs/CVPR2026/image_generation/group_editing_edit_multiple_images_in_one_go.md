---
title: >-
  [Paper Note] Group Editing: Edit Multiple Images in One Go
description: >-
  [CVPR 2026][Image Generation][consistent multi-image editing] This paper proposes GroupEditing, which reconstructs a group of related images as pseudo-video frames and combines explicit geometric correspondences from VGGT with the implicit temporal prior of a video diffusion model. Two specially designed positional encodings—Ge-RoPE and Identity-RoPE—are introduced to inject correspondence information, enabling cross-view consistent group image editing that significantly outperforms existing methods in visual quality, editing consistency, and semantic alignment.
tags:
  - CVPR 2026
  - Image Generation
  - consistent multi-image editing
  - video diffusion prior
  - geometric correspondence
  - RoPE positional encoding
  - pseudo-video
date: 2026-05-08
content_hash: 437929ccb93d521d
---

# Group Editing: Edit Multiple Images in One Go

**Conference**: CVPR 2026
**arXiv**: [2603.22883](https://arxiv.org/abs/2603.22883)
**Code**: [https://group-editing.github.io/](https://group-editing.github.io/)
**Area**: Diffusion Models / Image Editing
**Keywords**: consistent multi-image editing, video diffusion prior, geometric correspondence, RoPE positional encoding, pseudo-video

## TL;DR
This paper proposes GroupEditing, which reconstructs a group of related images as pseudo-video frames and combines explicit geometric correspondences from VGGT with the implicit temporal prior of a video diffusion model. Two specially designed positional encodings—Ge-RoPE and Identity-RoPE—are introduced to inject correspondence information, enabling cross-view consistent group image editing that significantly outperforms existing methods in visual quality, editing consistency, and semantic alignment.

## Background & Motivation

1. **Background**: Existing image editing methods (e.g., InstructPix2Pix, ControlNet) focus primarily on single-image editing. In virtual content creation, digital commerce, and similar applications, users frequently need to apply consistent modifications across multi-view images of the same subject—such as uniformly recoloring a digital character's clothing or stylizing product images from multiple angles.
2. **Limitations of Prior Work**: Editing images one by one leads to appearance and structural inconsistencies. Optimization-based propagation methods (e.g., editing one image and then propagating the changes) suffer from poor generalization and tend to produce artifacts. Optimization-free methods (e.g., Edicho) rely on semantic correspondence and tracking tools and can only handle a small number of images.
3. **Key Challenge**: In geometrically complex scenarios (e.g., target rotation, occlusion, deformation), semantic matching based solely on attention features is insufficiently precise. Identifying "the left eye across different viewpoints" or "tracking a logo on a T-shirt rotated by 30°" poses significant challenges for existing methods.
4. **Goal**: To establish reliable cross-image correspondences within a geometrically diverse group of related images, enabling multi-image consistent editing from a single instruction.
5. **Key Insight**: The authors identify two key observations: (1) *Implicit correspondence*—video models inherently possess a temporal consistency prior, which can be inherited by treating an image group as a "pseudo-video"; (2) *Explicit correspondence*—the implicit correspondences of video models alone are insufficient in geometrically complex scenarios, necessitating dense geometric matching from VGGT as a complement.
6. **Core Idea**: The multi-image editing problem is reformulated as a pseudo-video generation problem, fusing explicit geometric correspondences (VGGT) with an implicit temporal prior (video diffusion model) by injecting correspondence information through specially designed positional encodings.

## Method

### Overall Architecture
The inputs consist of a group of related images along with their corresponding segmentation masks and a text editing instruction. Images are first encoded into latent space via a VAE encoder and arranged as a pseudo-video sequence along the temporal dimension. Within the Transformer backbone of the WAN-2.1 video diffusion model, two enhanced RoPE positional encodings are injected: Ge-RoPE for cross-view geometric alignment and Identity-RoPE for intra-image identity preservation of the target. Explicit geometric feature tokens extracted by VGGT are concatenated to the latent token sequence and participate in self-attention computation. The edited multi-view consistent images are then produced by the decoder.

### Key Designs

1. **Data Construction Pipeline (GroupEditData)**:

    - **Function**: Constructs a large-scale multi-image editing training dataset.
    - **Mechanism**: Gemini 2.5 is used to generate image groups (18,248 groups) from manually written text instructions. SAM and Grounding DINO are applied for target segmentation to obtain masks. Qwen-VL-Max is then employed for consistency and aesthetic evaluation to filter for quality, retaining 7,517 high-quality groups. Each group contains images, masks, full-image descriptions, and segmented region descriptions.
    - **Design Motivation**: Large-scale paired multi-image editing data is currently unavailable. This pipeline provides the critical infrastructure that makes training feasible.

2. **Geometry-enhanced RoPE (Ge-RoPE)**:

    - **Function**: Injects explicit geometric correspondence information extracted by VGGT into the positional encodings to achieve fine-grained spatial alignment across viewpoints.
    - **Mechanism**: A pixel-level displacement field $\Delta(h,w) = (\Delta_h, \Delta_w)$ is obtained from VGGT, scaled to the latent-space resolution, and smoothed with a Gaussian kernel ($\mu=21, \sigma=11$) while prioritizing high-confidence correspondences. The smoothed displacements are added to the original spatial grid indices to construct a warped grid $\tilde{h} = h + \Delta_h^{\text{smooth}}$, and nearest-neighbor indexing into a precomputed frequency bank generates the geometry-aware RoPE encoding.
    - **Design Motivation**: The implicit correspondences of video models are insufficiently accurate in geometrically complex scenes. Ge-RoPE uses explicit displacement fields to inform the model which position in image B corresponds to position $(h,w)$ in image A, substantially improving spatial alignment precision.

3. **Identity-RoPE**:

    - **Function**: Ensures identity consistency of the same target across different images.
    - **Mechanism**: The minimum bounding rectangle $\mathcal{R}_t$ of the target in each image is identified via segmentation masks. Pixel coordinates within the rectangle are normalized to local coordinates relative to the rectangle's origin: $(\tilde{h}, \tilde{w}) = (h - y_1^{(t)}, w - x_1^{(t)})$. Consequently, the same target region across different images receives identical positional encodings, regardless of its absolute position within each image.
    - **Design Motivation**: Targets may appear at different absolute positions across viewpoints, causing standard positional encodings to treat them as distinct entities. Identity-RoPE enables all instances of "the cat's face" across images to share the same positional signal through coordinate normalization, thereby preserving identity consistency.

### Loss & Training
Training is conducted on WAN-2.1 (a Transformer-based video diffusion model) using the AdamW optimizer (weight decay 0.01, learning rate $1 \times 10^{-4}$), at a resolution of $528 \times 528$, with batch size 8 on 8 A800 GPUs. The training objective is the standard velocity-field prediction loss.

## Key Experimental Results

### Main Results

| Method | CLIP-Score↑ | Aesthetic↑ | DINO-Score↑ | Edit Consistency↑ | PSNR↑ |
|--------|------------|-----------|------------|------------------|-------|
| Anydoor | 0.2728 | 4.72 | 0.7208 | 0.8697 | 0.6182 |
| OminiControl | 0.2902 | 5.10 | 0.7326 | 0.8676 | 0.6457 |
| Edicho | 0.3059 | 4.89 | 0.8080 | 0.8988 | 0.6935 |
| **GroupEditing** | **0.3122** | **5.39** | **0.8168** | **0.9239** | **0.7624** |

User study results (ranked 1=best to 4=worst): GroupEditing ranks first across all four dimensions—identity consistency (1.67), aesthetics (1.46), appearance fidelity (1.50), and overall quality (1.47).

### Ablation Study

| Configuration | CLIP-Score↑ | Aesthetic↑ | DINO-Score↑ | Edit Consistency↑ |
|--------------|------------|-----------|------------|------------------|
| w/o VGGT | 0.2728 | 4.72 | 0.7208 | 0.8616 |
| w/o Ge-RoPE | 0.2902 | 4.89 | 0.7326 | 0.8697 |
| w/o Identity-RoPE | 0.2902 | 4.89 | 0.7326 | 0.9108 |
| Full model | **0.3122** | **5.39** | **0.8168** | **0.9239** |

### Key Findings
- Explicit geometric features from VGGT contribute the most: removing them causes DINO-Score to drop from 0.8168 to 0.7208 and edit consistency to drop from 0.9239 to 0.8616.
- Identity-RoPE primarily improves edit consistency (0.9108→0.9239), with a comparatively smaller contribution to visual quality.
- Edited results can be directly applied to DreamBooth/LoRA personalization and Must3R 3D reconstruction, validating cross-view consistency.

## Highlights & Insights
- **The pseudo-video reformulation is highly elegant**: Recasting multi-image editing as video editing inherits the temporal consistency prior of video models at no additional cost—a clean and effective problem transformation.
- **The fusion of explicit and implicit correspondences**: Ge-RoPE injects geometric information via positional encodings rather than modifying attention weights, yielding a lightweight yet effective integration mechanism.
- **Engineering value of the data construction pipeline**: The fully automated text→generation→filtering→annotation pipeline is transferable to other tasks that require paired training data.

## Limitations & Future Work
- Training data is generated by Gemini rather than sourced from real multi-view images, which may limit generalization to real-world scenarios.
- Editing quality depends on the accuracy of VGGT's geometric correspondences and may degrade when VGGT estimates are unreliable.
- Resolution is fixed at $528 \times 528$; extension to high-resolution settings has not been validated.
- Segmentation masks are required as input, which increases the barrier to practical use.

## Related Work & Insights
- **vs. Edicho**: Edicho performs zero-shot consistent editing via semantic correspondence and tracking tools, but is constrained to a small number of images. GroupEditing is the first training-based framework, scaling to larger image groups through a combination of data and model design.
- **vs. Frame2Frame/ChronoEdit**: These methods leverage video models to enhance temporal consistency in single-image editing. GroupEditing extends this paradigm by treating multiple images as a unified pseudo-video.
- **vs. ControlNet/T2I-Adapter**: These are general single-image conditional control methods. GroupEditing focuses specifically on consistency constraints across multiple images.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of pseudo-video reformulation and dual-RoPE injection is creative, though individual components are not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers quantitative evaluation, qualitative results, user study, ablation study, and downstream application validation—relatively comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Logically clear with rich illustrations.
- **Value**: ⭐⭐⭐⭐ Multi-image consistent editing addresses a practical need; the first training-based framework has pioneering significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Language-Free Generative Editing from One Visual Example](language-free_generative_editing_from_one_visual_example.md)
- [\[CVPR 2026\] CARE-Edit: Condition-Aware Routing of Experts for Contextual Image Editing](care-edit_condition-aware_routing_of_experts_for_contextual_image_editing.md)
- [\[CVPR 2026\] ChordEdit: One-Step Low-Energy Transport for Image Editing](chordedit_one-step_low-energy_transport_for_image_editing.md)
- [\[CVPR 2026\] SimLBR: Learning to Detect Fake Images by Learning to Detect Real Images](simlbr_learning_to_detect_fake_images_by_learning_to_detect_real_images.md)
- [\[CVPR 2026\] TokenLight: Precise Lighting Control in Images using Attribute Tokens](tokenlight_precise_lighting_control_in_images_using_attribute_tokens.md)

</div>

<!-- RELATED:END -->
