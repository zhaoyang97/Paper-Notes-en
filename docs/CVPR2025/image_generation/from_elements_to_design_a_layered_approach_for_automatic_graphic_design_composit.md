---
title: >-
  [Paper Note] From Elements to Design: A Layered Approach for Automatic Graphic Design Composition
description: >-
  [CVPR 2025][Image Generation][Graphic Design] LaDeCo introduces the layered design principles of graphic design into Large Multimodal Models (LMMs). It first uses GPT-4o to perform semantic layer planning for multimodal design elements, then progressively predicts element attributes layer by layer, rendering intermediate results to provide feedback to the model. This decomposes the complex design composition task into manageable sub-steps, significantly outperforming baseline…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Graphic Design"
  - "Design Automation"
  - "Layered Generation"
  - "Large Multimodal Models"
  - "Layout Generation"
date: 2026-05-08
content_hash: c0d78947f211769c
---

# From Elements to Design: A Layered Approach for Automatic Graphic Design Composition

**Conference**: CVPR 2025  
**arXiv**: [2412.19712](https://arxiv.org/abs/2412.19712)  
**Code**: None  
**Area**: Diffusion Models  
**Keywords**: Graphic Design, Design Automation, Layered Generation, Large Multimodal Models, Layout Generation

## TL;DR
LaDeCo introduces the layered design principles of graphic design into Large Multimodal Models (LMMs). It first uses GPT-4o to perform semantic layer planning for multimodal design elements, then progressively predicts element attributes layer by layer, rendering intermediate results to provide feedback to the model. This decomposes the complex design composition task into manageable sub-steps, significantly outperforming baseline methods in design composition quality.

## Background & Motivation

**Background**: Graphic design creation requires combining multimodal elements (images, titles, embellishments, etc.) into visually appealing and informative layouts. Existing studies mainly focus on design subtasks—content-aware layout generation (arranging element positions considering only the background image content) and typography generation (generating typographic attributes like fonts and colors for text)—but have not achieved end-to-end design composition from a complete set of elements.

**Limitations of Prior Work**: (1) Fragmented subtask methods—layout methods neglect text attributes while typography methods ignore visual elements, requiring users to manually chain models of different functions; (2) FlexDM, the only attempt at end-to-end design composition, represents all elements in a flattened manner and predicts all attributes simultaneously, ignoring the hierarchical structure of designs. This leads to issues such as severe element overlaps and poor readability.

**Key Challenge**: Graphic design is inherently layered—designers build compositions step-by-step in the order of background $\rightarrow$ backing $\rightarrow$ image/logo $\rightarrow$ text $\rightarrow$ embellishment. Flattened representations lose this hierarchical information, forcing the model to handle too many complex constraints simultaneously (avoiding main subject occlusion, ensuring text readability, element balance, etc.), which makes the task excessively difficult.

**Goal**: (1) How to achieve end-to-end automatic composition from a set of multimodal elements to a final design? (2) How to leverage the hierarchical structure of designs to reduce task difficulty?

**Key Insight**: Inspired by the workflow of human designers: first establish the background, then overlay backings, images, text, and decorative elements. The design of each layer builds upon the cumulative results of all previous layers, allowing the model to focus only on the arrangement of the current layer at each step, while receiving visual feedback from already completed layers as context.

**Core Idea**: Use GPT-4o for semantic layer planning and let the LMM progressively predict element attributes layer by layer. Each step renders the completed layers into an image to provide feedback to the model, decomposing the highly challenging design composition task into lower-difficulty layer-by-layer generation.

## Method

### Overall Architecture
LaDeCo consists of two stages: (1) **Layer Planning**: Uses GPT-4o to analyze the content of each input element and assign it to one of five predefined semantic layers (background, backing, logo/image, text, embellishment); (2) **Layered Design Composition**: Progressively generates step-by-step from layer $G_1$ to $G_5$ based on a fine-tuned LMM (Llama-3.1-8B + CLIP ViT-L/14). At the $i$-th layer, the model receives the current layer element content $X_i$ and the rendered image of the previous layers $G_{i-1}$, and outputs the element attributes $Y_i$ (position, font, color, etc., in JSON format). Upon completion of each layer, the intermediate result is rendered to serve as the context for the next layer.

### Key Designs

1. **Layer Planning Module**:

    - **Function**: Automatically classify input multimodal elements into five semantic layers (background, backing, logo/image, text, embellishment) to determine the generation order.
    - **Mechanism**: Prompts are carefully designed to guide GPT-4o to understand the visual content of each element and determine its layer assignment based on content semantics. For example, solid-colored rectangular boxes might be backings, while star-shaped elements might be embellishments. For training samples, both the final design image and metadata are provided to assist in judgment. Feature descriptions for each layer are defined in detail within the prompts.
    - **Design Motivation**: Public datasets (Crello) do not contain layer information, but element semantics can be inferred from their content. Using the powerful multimodal understanding capability of GPT-4o automatically completes this classification, avoiding manual annotation.

2. **Layered Design Composition**:

    - **Function**: Progressively generate element attributes layer by layer, from background to embellishment, utilizing the rendered results of preceding layers as visual context at each step.
    - **Mechanism**: The model architecture consists of a visual encoder (CLIP ViT-L/14) + projector (2-layer MLP + GELU) + LMM backbone (Llama-3.1-8B + LoRA). The visual encoder output is compressed via 2D average pooling to reduce token count (5 tokens per image: 1 cls + $2\times2$). Inputs include the current layer element content (images for visual elements, text strings for text elements) and the rendered image of preceding layers. Outputs are element attributes in JSON format (position, typographical parameters, etc.). Training uses negative log-likelihood loss: $\mathcal{L} = -\sum_{i=1}^{5} \log P(Y_i | Y_{<i}, X_{\leq i}, G_{<i})$.
    - **Design Motivation**: (1) Focusing on only one layer of elements at each step greatly reduces task difficulty; (2) The rendered intermediate images provide the model with visual feedback—allowing it to see the position of main subjects to avoid occlusion, or locate the backing layer to place text correctly; (3) This layer-by-layer generation also naturally supports design subtasks—providing ground truth (GT) for the first few layers allows for content-aware layout or typography generation.

3. **Zero-Shot Subtask Adaptation**:

    - **Function**: Perform subtasks such as content-aware layout generation and typography generation, as well as applications like resolution adjustment, element inpainting, and design transfer without any task-specific training.
    - **Mechanism**: Due to the flexibility of layered generation, providing GT images for different layers allows adaptation to different subtasks. For example, given $G_1$ (background layer GT), generating $G_2$ to $G_5$ is equivalent to content-aware layout generation; given $G_1$ to $G_3$, generating $G_4$ is equivalent to typography generation.
    - **Design Motivation**: The layered structure naturally corresponds to the different stages of the design workflow, allowing a single model to flexibly adapt to various application scenarios without the need to maintain multiple specialized models.

### Loss & Training
Standard autoregressive language model training is adopted—negative log-likelihood loss across all layers. LoRA (rank=32, alpha=64) is employed to efficiently fine-tune the Llama-3.1-8B backbone while freezing the visual encoder. The model is trained on 4 A100-80G GPUs for approximately 7K iterations with a batch size of 128 and a learning rate of 2e-4. During inference, temperature is set to 0.7 and Top-p to 0.95. Rendering intermediate results between layers only incurs an overhead of about 20% additional time.

## Key Experimental Results

### Main Results

| Method | LLaVA-OV (Design) | LLaVA-OV (Content) | LLaVA-OV (Typo) | Val↑ | Ove↓ |
|------|-------------------|--------------------|-----------------| -----|------|
| FlexDM | 5.34 | 5.29 | 5.41 | 0.876 | 0.324 |
| GPT-4o | 6.53 | 6.49 | 6.60 | 0.997 | 0.060 |
| **LaDeCo (Ours)** | **8.08** | **7.92** | **8.00** | 0.937 | 0.087 |
| GT | 8.35 | 8.21 | 8.30 | 0.926 | 0.077 |

### Ablation Study

| Configuration | LLaVA-OV (i) | LLaVA-OV (v) | Undl | Unds |
|------|-------------|-------------|------|------|
| w/o LP, w/o LDC | 7.23 | 6.29 | 0.619 | 0.588 |
| w/ LP, w/o LDC | 7.84 | 6.66 | 0.657 | 0.624 |
| **Full (w/ LP + LDC)** | **8.08** | **6.98** | **0.692** | **0.658** |
| + LargeCrello data | 8.22 | 7.09 | 0.732 | 0.712 |

### Key Findings
- **LaDeCo is close to ground truth design in terms of LLaVA-OV comprehensive score** (8.08 vs. GT 8.35), significantly outperforming FlexDM (5.34) and GPT-4o (6.53).
- **Layer planning and layered generation both contribute significantly**: adding only layer planning (w/ LP, w/o LDC) improves by 0.61 points, and further incorporating layered generation (Full) yields another 0.24-point improvement.
- FlexDM suffers severely from element overlap (Ove=0.324, far higher than GT's 0.077), while LaDeCo (0.087) is close to GT.
- On content-aware layout and typography subtasks, LaDeCo outperforms specialized models under zero-shot settings (e.g., surpassing COLEs on the typography task).
- Expanding the dataset size (+LargeCrello) further improves performance, indicating the scalability of the proposed method.

## Highlights & Insights
- **Layered generation + visual feedback** resembles Chain-of-Thought reasoning—decomposing design composition into layer-by-step subproblems with intermediate results serving as a "thinking chain." This paradigm can be generalized to any complex hierarchical generation task (e.g., slide design, web layout).
- **Zero-shot subtask adaptation** is an elegant byproduct of layered design—a single model can adapt to multiple downstream tasks simply by providing ground truths for different layers, which reduces deployment costs.
- The combination of GPT-4o layer planning and open-source LMM fine-tuning achieves a solid balance between leveraging LLM reasoning and controllable fine-tuning.

## Limitations & Future Work
- Layer planning relies on the GPT-4o API, which limits cost and response speed; training a localized layer planner remains a direction to explore.
- Only five fixed layer categories are supported, which might lack flexibility for atypical designs (e.g., complex infographics).
- Element arrangement within a single layer is randomized, which may cause loss of relative ordering information between elements of the same layer.
- Training data is limited to Crello and LargeCrello, offering limited style and domain coverage.
- Extra rendering steps are required for each layer, and although this only adds about 20% overhead time, there is still room for optimization in real-time scenarios.

## Related Work & Insights
- **vs FlexDM**: FlexDM uses mask field prediction to predict all attributes at once, ignoring the hierarchical structure and causing severe overlaps and layout issues. The layered approach of LaDeCo fundamentally addresses these challenges.
- **vs PosterLLaVA / PosterLlama**: These methods only perform content-aware layout generation (predicting only positions) without handling text attributes, failing to achieve full design composition. LaDeCo unifies all attribute predictions.
- **vs. Direct Generation by GPT-4o**: Although GPT-4o has strong understanding capabilities, it lacks design-specific training, resulting in extremely low effectiveness for backing layout (Undl=0.378 vs. GT=0.685).

## Rating
- Novelty: ⭐⭐⭐⭐ Systematically introducing layered design principles to LMMs is a pioneering work with high completeness.
- Experimental Thoroughness: ⭐⭐⭐⭐ The main task, subtasks, ablation studies, and applications are comprehensively evaluated, incorporating both LLaVA-OV automated assessment and geometric metrics across multiple dimensions.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and compelling visualizations (rendered layer images intuitively showcase the advantages of the method).
- Value: ⭐⭐⭐⭐ The first high-quality end-to-end design composition method, presenting a practical step forward in design automation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Rethinking Layered Graphic Design Generation with a Top-Down Approach](../../ICCV2025/image_generation/rethinking_layered_graphic_design_generation_with_a_top-down_approach.md)
- [\[ACL 2025\] Synthia: Novel Concept Design with Affordance Composition](../../ACL2025/image_generation/synthia_novel_concept_design_with_affordance_composition.md)
- [\[ICLR 2026\] CreatiDesign: A Unified Multi-Conditional Diffusion Transformer for Creative Graphic Design](../../ICLR2026/image_generation/creatidesign_a_unified_multi-conditional_diffusion_transformer_for_creative_grap.md)
- [\[CVPR 2025\] EasyCraft: A Robust and Efficient Framework for Automatic Avatar Crafting](easycraft_avatar_crafting.md)
- [\[CVPR 2025\] ChatGen: Automatic Text-to-Image Generation From FreeStyle Chatting](chatgen_automatic_text-to-image_generation_from_freestyle_chatting.md)

</div>

<!-- RELATED:END -->
