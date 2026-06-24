---
title: >-
  [Paper Note] ColorPeel: Color Prompt Learning with Diffusion Models via Color and Shape Disentanglement
description: >-
  [ECCV 2024][Image Generation][Color Prompt Learning] The paper proposes ColorPeel, a method that learns a color prompt token on basic geometric shapes of target colors (disentangling color and shape) and introduces a cross-attention alignment loss, enabling T2I diffusion models to accurately generate objects with user-specified RGB colors.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Color Prompt Learning"
  - "Diffusion Models"
  - "Color-Shape Disentanglement"
  - "T2I Personalization"
  - "Cross-Attention Alignment"
date: 2026-05-08
content_hash: 472ff98b1ae81668
---

# ColorPeel: Color Prompt Learning with Diffusion Models via Color and Shape Disentanglement

**Conference**: ECCV 2024  
**arXiv**: [2407.07197](https://arxiv.org/abs/2407.07197)  
**Code**: [https://moatifbutt.github.io/colorpeel/](https://moatifbutt.github.io/colorpeel/) (Project Page)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Color Prompt Learning, Diffusion Models, Color-Shape Disentanglement, T2I Personalization, Cross-Attention Alignment

## TL;DR

The paper proposes ColorPeel, a method that learns a color prompt token on basic geometric shapes of target colors (disentangling color and shape) and introduces a cross-attention alignment loss, enabling T2I diffusion models to accurately generate objects with user-specified RGB colors.

## Background & Motivation

**Background**: Text-to-Image (T2I) diffusion models have demonstrated powerful capabilities in generating images from text prompts, but they suffer from fundamental precision flaws regarding color control. Currently, T2I models can only specify colors through linguistic color names (e.g., "red", "beige").

**Limitations of Prior Work**: Linguistic representation of color is **discrete**, whereas color space itself is **continuous**—a single term like "red" can correspond to hundreds of different RGB values. Even when using more fine-grained color names such as "beige" or "light green", the generated results often differ significantly from the exact color in the user's mind. This is a severe problem in design, fashion, and art, where users need to generate colors that precisely match specific palettes.

**Key Challenge**: Although existing T2I personalization methods (e.g., Textual Inversion, DreamBooth, Custom Diffusion) can learn new concepts, they suffer from **color-shape entanglement** when learning colors—learning a color token from a uniformly colored patch causes the model to simultaneously memorize the shape of the patch, making it impossible to transfer the color to other objects. Directly inputting RGB numerical values into text prompts also performs poorly.

**Key Insight**: Color is an abstract attribute that cannot be learned from a single shape. If the same target color is presented on **multiple different geometric shapes**, the model can naturally "peel" the color—a "shared attribute"—away from the shapes.

**Core Idea**: Generate a set of basic geometric objects (2D/3D shapes) of the target color, jointly learn the color token and shape tokens, and further enforce disentanglement through a cross-attention alignment loss.

## Method

### Overall Architecture

ColorPeel is based on Stable Diffusion v1.4, with the following workflow: (1) Given an RGB color selected by the user, it automatically generates a set of basic geometric shape images of that color (2D circle/square/hexagon/triangle, or 3D sphere/cylinder/cube/cone, etc.); (2) Introduces learnable tokens $c^*$ and $s^*$ for color and shape respectively, and trains using text templates with disentanglement annotations (e.g., "A photo of $s_i^*$ filled with $c^*$"); (3) Once learning is complete, the color token $c^*$ can be used to generate the precise color for any object.

### Key Designs

1. **Training Data Construction for Color-Shape Disentanglement**: Automatically generate images of at least two different shapes for each target color. The core logic of disentanglement is: when the color is identical but the shapes are different across multiple training images, the color token $c^*$ is forced to encode only color information (since the shapes vary), while the shape tokens $s_i^*$ encode their respective shapes. Because 3D shapes contain physical effects like lighting and shadows, the generated color prompts are closer to real-world scenarios.

   Design Motivation: Learning color from a single colored patch inevitably leads to entanglement (as seen in Custom Diffusion failure cases); multi-shape training is key to disentanglement.

2. **Cross-Attention Alignment Loss (CAA)**: Visualizing the cross-attention maps of SD-UNet reveals that the attention regions of the color token and shape tokens are often misaligned—color attention leaks to background regions, leading to inaccurate colors. The CAA loss enforces alignment by maximizing the cosine similarity of the color and shape attention maps:

    $$\mathcal{L}_{caa} = 1 - \cos(\mathcal{A}_t^{c^*}, \mathcal{A}_t^{s^*})$$

   The final training objective:
    $$\mathcal{V}^* = \underset{\mathcal{V}}{\arg\min}\ \mathbb{E}[\mathcal{L}_{rec} + \lambda \cdot \mathcal{L}_{caa}]$$

   where $\mathcal{L}_{rec}$ is the standard LDM noise reconstruction loss, and $\lambda$ is a trade-off hyperparameter (optimal value is 0.2).

   Design Motivation: Attention leakage is one of the fundamental causes of imprecise color tuning; CAA directly addresses the spatial alignment of color and shape at the attention level.

3. **Flexible Learning Framework**: ColorPeel is compatible with various T2I adaptation methods—acting as an enhancement to Custom Diffusion (optimizing key/value projection matrices + tokens), or combined with methods like DreamBooth. During training, both the color token embedding $\mathcal{V}^{c^*}$ and shape token embeddings $\mathcal{V}^{s^*}$ are optimized simultaneously.

### Loss & Training

- **Coarse-grained color learning**: 1500 training steps
- **Fine-grained color learning**: 6000 training steps
- Batch size of 2, learning rate of $10^{-5}$
- Training time on an A40 GPU is about 19 minutes (compared to 24 minutes for Custom Diffusion, which is also superior)
- Evaluation metrics: CIE Lab color difference ($\Delta E$), Mean Angular Error (MAE) in sRGB, and MAE in Hue

## Key Experimental Results

### Main Results

| Method | $\Delta E$ ↓ | $\Delta E_{ch}$ ↓ | MAE(rgb) 10% ↓ | MAE(rgb) 50% ↓ | MAE(Hue) 10% ↓ | MAE(Hue) 50% ↓ | Time (min) |
|------|-------------|-------------------|----------------|----------------|----------------|----------------|-----------|
| Stable Diffusion | 47.45 | 41.55 | 12.89 | 20.04 | 54.14 | 86.38 | - |
| Rich-Text | 36.62 | 32.48 | 9.91 | 13.29 | 50.55 | 72.77 | - |
| Textual Inversion | 48.98 | 44.29 | 15.22 | 19.51 | 52.66 | 69.35 | 118 |
| DreamBooth | 50.71 | 46.29 | 14.75 | 19.30 | 47.12 | 67.13 | 56 |
| Custom Diffusion | 48.47 | 42.23 | 13.43 | 17.93 | 31.63 | 55.07 | 24 |
| **ColorPeel (3D)** | **21.39** | **16.51** | **4.36** | **7.76** | **2.63** | **6.47** | **19** |
| ColorPeel (2D) | 20.45 | 15.29 | 4.83 | 7.88 | 3.18 | 7.43 | - |

ColorPeel significantly outperforms existing methods across all metrics, with the $\Delta E$ error reduced by approximately 55% (21.39 vs. 48.47) and Hue MAE dramatically dropping from 31.63 to 2.63.

### Ablation Study

| $\lambda$ (CAA weight) | $\Delta E$ ↓ | $\Delta E_{Ch}$ ↓ | MAE(rgb) 10% ↓ | MAE(Hue) 10% ↓ | Description |
|---|---|---|---|---|---|
| 0.0 (Degrades to CD) | 48.47 | 42.23 | 13.43 | 31.63 | No CAA, color-shape entanglement |
| 0.1 | 22.23 | 16.86 | 5.13 | 3.48 | Already significantly improved |
| **0.2** | **21.39** | **16.51** | **4.36** | **2.63** | **Optimal** |
| 0.4 | 23.37 | 17.10 | 4.91 | 3.87 | Slightly degraded |
| 0.8 | 23.79 | 17.01 | 4.98 | 4.06 | Overly strong constraint |
| 1.0 | 24.43 | 18.64 | 5.03 | 4.27 | Excessive constraint |

When the CAA loss is removed ($\lambda=0$), the model degrades to Custom Diffusion and fails to disentangle color and shape. $\lambda=0.2$ is the optimal balance point.

### Key Findings

- **User Study**: A 2AFC experiment with 15 participants (analyzed via the Thurstone Case V model) shows that ColorPeel is statistically significantly superior to CD, DB, Rich-Text, and TI (highest z-score, with non-overlapping 95% confidence intervals)
- **2D vs. 3D Shapes**: Due to the inclusion of lighting and shadowing effects, 3D shapes slightly outperform 2D in Hue precision, but 2D shapes are better in $\Delta E$. Both significantly outperform the baselines.
- **Fine-grained Colors**: The method can effectively distinguish similar colors such as navy/indigo/cyan, generating high-quality differentiated results.
- **Generalization Capability**: The method can be directly extended to texture and material learning (mapping texture/material patterns onto the surfaces of 3D shapes).
- **Color Interpolation**: The learned color tokens support linear interpolation, allowing continuous generation of intermediate colors without additional training.

## Highlights & Insights

- Precise problem definition: Formulating "color prompt learning" as a new task addresses real-world demands in design and creative industries.
- Highly intuitive solution: The disentanglement concept of "showing the same color on different shapes" is simple, elegant, and analogous to cognitive science.
- The CAA loss solves the color leakage problem via alignment at the attention level, offering excellent interpretability.
- Absolutely no real-world image acquisition is required; training data is automatically synthesized using basic Blender geometries, resulting in zero labeling cost.
- Generalization to texture/material learning demonstrates the universality of the method, color being just one instance of "abstract attribute learning."

## Limitations & Future Work

- Based on SD v1.4, the generation resolution is limited (512×512); exploring newer backbones like SDXL is a future direction.
- Currently, it only learns a single color token; precise control over multi-color combinations (e.g., gradients) has not yet been fully explored.
- Lighting effects on 3D shapes might introduce color bias (influenced by luminance); investigating better color spaces could resolve this.
- Color token interpolation is currently a simple linear interpolation; it is worth exploring whether better color space mappings exist.
- The method can be extended to color consistency control in video generation.

## Related Work & Insights

- Break-a-Scene decomposes multiple concepts from a single image but cannot guarantee a clean separation between abstract concepts (color) and concrete concepts (shape).
- Textual Inversion only learns tokens without fine-tuning the model, but the lack of a disentanglement mechanism leads to imprecise colors.
- The CAA loss is inspired by DPL (which minimizes overlapping attention of different objects); this paper reverses its usage by maximizing the overlap between color and shape attention.
- Insight: Abstract attribute learning (style, material, lighting) can all utilize a similar "multi-instance disentanglement" paradigm.

## Rating

- Novelty: ⭐⭐⭐⭐ First to propose the color prompt learning task, with both the disentanglement scheme and the CAA loss representing innovative contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage spanning quantitative metrics, user study, ablations, fine-grained analysis, generalization, and interpolation.
- Writing Quality: ⭐⭐⭐⭐ The motivation is clear, the failure case analysis in Fig.2 is intuitive, and the quantitative metric design is reasonable.
- Value: ⭐⭐⭐⭐ Highly practical with low training costs (19 minutes), bringing direct value to design and creative domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Color Alignment in Diffusion](../../CVPR2025/image_generation/color_alignment_in_diffusion.md)
- [\[CVPR 2025\] GCC: Generative Color Constancy via Diffusing a Color Checker](../../CVPR2025/image_generation/gcc_generative_color_constancy_via_diffusing_a_color_checker.md)
- [\[ECCV 2024\] Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models](source_prompt_disentangled_inversion_for_boosting_image_editability_with_diffusi.md)
- [\[CVPR 2025\] The Art of Deception: Color Visual Illusions and Diffusion Models](../../CVPR2025/image_generation/the_art_of_deception_color_visual_illusions_and_diffusion_models.md)
- [\[ECCV 2024\] Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation](learning_differentially_private_diffusion_models_via_stochastic_adversarial_dist.md)

</div>

<!-- RELATED:END -->
