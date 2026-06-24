---
title: >-
  [Paper Note] DreamView: Injecting View-specific Text Guidance into Text-to-3D Generation
description: >-
  [ECCV 2024][3D Vision][Text-to-3D Generation] DreamView proposes an adaptive text guidance injection module to collaboratively inject view-specific and global text descriptions into a diffusion model, achieving customizable and multi-view consistent text-to-3D generation.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Text-to-3D Generation"
  - "View Customization"
  - "Diffusion Models"
  - "Score Distillation Sampling"
  - "Multi-view Consistency"
date: 2026-05-08
content_hash: 9a3fbd41d7bbbe9d
---

# DreamView: Injecting View-specific Text Guidance into Text-to-3D Generation

**Conference**: ECCV 2024  
**arXiv**: [2404.06119](https://arxiv.org/abs/2404.06119)  
**Code**: [https://github.com/iSEE-Laboratory/DreamView](https://github.com/iSEE-Laboratory/DreamView)  
**Area**: 3D Vision  
**Keywords**: Text-to-3D Generation, View Customization, Diffusion Models, Score Distillation Sampling, Multi-view Consistency

## TL;DR

DreamView proposes an adaptive text guidance injection module to collaboratively inject view-specific and global text descriptions into a diffusion model, achieving customizable and multi-view consistent text-to-3D generation.

## Background & Motivation

**Background**: Text-to-3D generation methods mainly fall into two categories: (a) direct 3D generation methods (e.g., Point-E, Shap-E); and (b) 2D lifting methods (e.g., DreamFusion, ProlificDreamer), which leverage pretrained text-to-image models to optimize differentiable 3D representations via SDS. The latter has gained more attention due to high fidelity.

**Limitations of Prior Work**:
   - Prior methods rely on a single text description shared across all views, failing to customize the appearance of specific views.
   - For instance, when designing a T-shirt with different patterns on the front and back, a single text description introduces ambiguity.
   - Even if different view contents are specified in the text, prior methods fail to place the corresponding elements correctly (e.g., "a backpack on the back" might be generated on the front by the model).

**Key Challenge**: 3D objects naturally possess multi-view diversity, yet existing methods only provide object-level global guidance, lacking view-level control capability.

**Goal**: To enable customization of 3D generation where users can specify different text descriptions for different views while maintaining overall consistency.

**Key Insight**: Learning to balance consistency and customization by training on a large-scale rendered 3D dataset.

**Core Idea**: Adaptively choosing whether to use global or view-specific text as the condition within each U-Net block of the diffusion model, controlling the balance between consistency and customization via a margin parameter.

## Method

### Overall Architecture

DreamView consists of two phases: DreamView-2D (a text-to-multiview image generation model) and DreamView-3D (lifting the 2D model to 3D generation via SDS). First, multi-view images are rendered from the Objaverse dataset. BLIP-2 is used to generate text descriptions for each view, which are then merged into a global description by GPT-4. Subsequently, a diffusion model with an adaptive guidance injection module is trained, and finally distilled into a 3D representation via SDS.

### Key Designs

1. **Dataset Construction Pipeline**:

    - **Function**: Constructing a large-scale training dataset containing multi-view images with corresponding view-specific and global text descriptions.
    - **Mechanism**: A three-step pipeline: (1) Rendering: Uniformly rendering 32 views of $512 \times 512$ images from ~435K 3D assets in Objaverse (totaling ~14M images); (2) Description: Using BLIP-2 to generate view-specific text descriptions for each rendered image; (3) Merging: Using GPT-4 to merge text descriptions of all views of the same object into a single global description.
    - **Design Motivation**: The original Objaverse dataset lacks view-specific text descriptions, which need to be constructed automatically using large multimodal models.

2. **Adaptive Guidance Injection Module**:

    - **Function**: Dynamically determining whether to use global or view-specific text as the cross-attention condition in each U-Net block of the diffusion model.
    - **Mechanism**: Calculating the similarity between image embeddings and the two text embeddings to determine which guidance is more needed in the current block. Let $E_o^t, E_v^t$ be the global and view-specific text embeddings, $\text{CLS}_o^t, \text{CLS}_v^t$ be their corresponding class tokens, and $E^i$ be the image embedding. Within each U-Net block:
    $\text{Sim} = \cos(\text{GAP}(E^i), \text{CLS}^t)$
      $\text{Sim}_o$ and $\text{Sim}_v$ are calculated respectively, and the margin parameter $m$ is used to determine which guidance is injected:
    $$\text{Guidance} = \begin{cases} E_v^t, & \text{if } \text{Sim}_o - \text{Sim}_v > m \\ E_o^t, & \text{else} \end{cases}$$
    - **Key Insight**: If the current image embedding has already absorbed sufficient global guidance ($\text{Sim}_o$ is large), view-specific guidance is injected as a supplement, and vice versa.
    - **margin parameter**: Large margin $\rightarrow$ more usage of global text $\rightarrow$ stronger consistency; small margin $\rightarrow$ more usage of view-specific text $\rightarrow$ stronger customization. Randomly sampled from [-0.1, 0.1] during training, and fixed to -0.025 for inference.
    - **Design Motivation**: Converting the complex trade-off between consistency and customization into a single hyperparameter adjustment, achieving an adaptive balance.

3. **DreamView-3D: Lifting 2D to 3D**:

    - **Function**: Transferring the customization and consistency capabilities of DreamView-2D to 3D generation.
    - **Mechanism**: Based on the threestudio framework, replacing Stable Diffusion in DreamFusion with DreamView-2D as the teacher model. The azimuth angle 0-360° is divided into four intervals (Front/Right/Back/Left), with each interval associated with a view-specific text:
        - Front: $[10°, 170°]$
        - Right: $(170°, 190°)$
        - Back: $[190°, 350°]$
        - Left: Remaining parts
    - The 3D representation uses the implicit-volume method, optimized via $x_0$-reconstruction loss:
    $$\mathcal{L}_{3D}(\phi, x=g(\phi)) = \mathbb{E}_{c,t,\epsilon}\left[\|x - \hat{x}_0\|_2^2\right]$$
    - **Design Motivation**: Leveraging SDS distillation to naturally inherit the consistency and customization priors learned by DreamView-2D.

### Loss & Training

- **2D Training Loss**: Standard diffusion denoising loss
  $$\mathcal{L}_{2D}(\theta, \mathcal{D}) = \mathbb{E}_{x,y,c,t,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t; y, c, t)\|_2^2\right]$$
- Training configuration: 16×V100 GPUs, batch size 2048, learning rate 1e-4.
- Generates $256 \times 256$ images from four orthogonal views.
- Initialized based on SD-v2.1, jointly trained on the 3D rendered dataset and the 2D LAION dataset.
- Uses an expanded attention mechanism to model relationships across multiple views.
- 3D optimization for 10,000 steps: the first 5,000 steps at 64×64 resolution, and the remaining 5,000 steps scaled up to 256×256.

## Key Experimental Results

### Main Results (Image generation quality of 1000 objects on the validation set)

| Method | CLIP-Overall ↑ | CLIP-View ↑ | CLIP-GT Image ↑ | IS ↑ |
|------|----------------|-------------|-----------------|------|
| Ground Truth | 34.5 | 34.8 | 1.00 | 10.3 |
| SD-v2.1 (overall/view) | 29.2/28.3 | 26.8/29.4 | 0.48/0.53 | **15.3/15.6** |
| MVDream (overall/view) | **31.3**/29.9 | 28.6/30.1 | 0.65/0.67 | 13.2/13.1 |
| DreamView-2D | 31.1 | **32.1** | **0.73** | 14.5 |

### User Study (35 participants, 180 3D objects)

| Evaluation Metric | DreamView | Best of Other Methods | Description |
|----------|-----------|-------------|------|
| Text alignment | **74.5%** | MVDream 8.7% | Decisive lead |
| Visual appearance | **51.4%** | ProlificDreamer 15.2% | Clear advantage |
| Overall preference | **67.9%** | MVDream 11.5% | User preference |

### Ablation Study (Impact of the Margin Parameter)

| Margin Value | Consistency Trend | Customization Trend | Description |
|---------|-----------|-----------|------|
| -0.1 | Weak | Strong | Over-customized, may lose global elements |
| -0.025 | Balanced | Balanced | Default inference setting |
| 0.0 | Good | Good | Acceptable balance point |
| 0.025 | Strong | Weak | Customized elements might be misplaced |
| 0.25 | Strongest | Weakest | Close to single-text guidance |

### Key Findings
- Although MVDream is designed for 3D consistency, it still suffers from misalignment of front and back content (e.g., displaying a Superman logo on the back of a MacBook instead of an Apple logo).
- Although ProlificDreamer generates rich details, it suffers severely from the multi-face (Janus) problem (e.g., generating multiple faces on an orangutan).
- DreamView also performs excellently when using only global text (without customization requirements), avoiding multi-face/multi-leg issues.
- Generation speed is approximately 55 minutes per object (A100), placing it between DreamFusion (30 min) and ProlificDreamer (180 min).

## Highlights & Insights
- The design of the **adaptive injection module** is highly elegant: it requires no extra network structures, balancing consistency and customization solely by dynamically switching conditions within existing cross-attention layers.
- The **margin parameter** simplifies the complex trade-off into a single scalar adjustment, allowing users to flexibly control the output according to their needs.
- The dataset construction pipeline, automated using BLIP-2 and GPT-4, serves as an excellent case study of LLM-empowered 3D data pipelines.
- Dividing the azimuth into 4 intervals significantly reduces user input overhead (requiring only 5 text prompts instead of view-by-view descriptions).

## Limitations & Future Work
- Faces of full-body characters can be blurry, limited by the low-resolution training images ($256 \times 256$).
- Text descriptions for different views must describe the same object instance, otherwise generation fails (e.g., describing a dog for the front but a monkey for the back).
- The generation speed remains relatively slow (~55 minutes); more efficient 3D generation schemes could be explored.
- Extending to more view intervals (e.g., six directions: up, down, left, right, front, back) could be considered to support finer control.

## Related Work & Insights
- **vs MVDream**: MVDream enhances 3D consistency through multi-view consistent diffusion but lacks support for view customization; DreamView introduces customization capability while preserving multi-view consistency.
- **vs DreamFusion**: DreamFusion uses a single shared text prompt, failing to precisely control view-specific content; DreamView allows users to specify different content for each view.
- **vs ProlificDreamer**: ProlificDreamer pursues high fidelity but suffers severely from the Janus (multi-face) problem; DreamView effectively mitigates this issue using global text constraints.

## Rating
- Novelty: ⭐⭐⭐⭐ Demostrates view-specific text-guided 3D generation for the first time, with a simple yet effective adaptive injection module.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive quantitative comparison, user study, and ablation studies, with extensive comparison against multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition, well-structured methodology, and intuitive illustrations.
- Value: ⭐⭐⭐⭐ Outlines a new interactive paradigm for 3D creative design, carrying practical application value (e.g., customized merchandise design).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] DreamScene360: Unconstrained Text-to-3D Scene Generation with Panoramic Gaussian Splatting](dreamscene360_unconstrained_text-to-3d_scene_generation_with_panoramic_gaussian_.md)
- [\[ECCV 2024\] DreamDissector: Learning Disentangled Text-to-3D Generation from 2D Diffusion Priors](dreamdissector_learning_disentangled_text-to-3d_generation_from_2d_diffusion_pri.md)
- [\[ECCV 2024\] GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing](gaussctrl_multi-view_consistent_text-driven_3d_gaussian_splatting_editing.md)
- [\[ECCV 2024\] JointDreamer: Ensuring Geometry Consistency and Text Congruence in Text-to-3D Generation via Joint Score Distillation](jointdreamer_ensuring_geometry_consistency_and_text_congruence_in_text-to-3d_gen.md)
- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)

</div>

<!-- RELATED:END -->
