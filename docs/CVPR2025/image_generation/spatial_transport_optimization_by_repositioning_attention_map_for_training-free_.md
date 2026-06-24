---
title: >-
  [Paper Note] STORM: Spatial Transport Optimization by Repositioning Attention Map for Training-Free Text-to-Image Synthesis
description: >-
  [CVPR 2025][Image Generation][Text-to-Image Generation] STORM proposes a Spatial Transport Optimization (STO) method based on optimal transport theory, which dynamically adjusts the spatial positions of object attention maps during the denoising process of diffusion models. Without requiring any training, it achieves precise spatial layout control, effectively solving the overlooked key issue of "mislocated objects" in T2I models.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Text-to-Image Generation"
  - "Spatial Alignment"
  - "Optimal Transport"
  - "Attention Map Repositioning"
  - "Training-Free Methods"
date: 2026-05-08
content_hash: 40389e8759cf3279
---

# STORM: Spatial Transport Optimization by Repositioning Attention Map for Training-Free Text-to-Image Synthesis

**Conference**: CVPR 2025  
**arXiv**: [2503.22168](https://arxiv.org/abs/2503.22168)  
**Code**: None  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Text-to-Image Generation, Spatial Alignment, Optimal Transport, Attention Map Repositioning, Training-Free Methods

## TL;DR
STORM proposes a Spatial Transport Optimization (STO) method based on optimal transport theory, which dynamically adjusts the spatial positions of object attention maps during the denoising process of diffusion models. Without requiring any training, it achieves precise spatial layout control, effectively solving the overlooked key issue of "mislocated objects" in T2I models.

## Background & Motivation

**Background**: Diffusion-based T2I models have achieved immense success in high-quality image generation. Training-free methods have garnered widespread attention due to their low-cost adaptability and generalization capability. Existing methods primarily focus on solving two types of problems: (1) "missing objects" — where objects specified in the prompt do not appear in the generated image; and (2) "mismatched attributes" — where object attributes such as colors and textures do not match the text.

**Limitations of Prior Work**: Another equally crucial but overlooked problem is "mislocated objects" — where the spatial positions of objects in the generated image do not accurately correspond to the textual description. For instance, given "a cat on the left, a dog on the right," the model may swap the positions of the cat and the dog, or place both on the same side. Surprisingly, even in state-of-the-art T2I models, such basic spatial control remains a challenge.

**Key Challenge**: Textual forms are inherently difficult to impose explicit spatial guidance because spatial descriptions in natural language are ambiguous ("left," "above"), whereas pixel-level spatial positions are precise. Existing methods lack effective mechanisms to bridge this semantic-spatial gap.

**Goal**: (1) Achieve precise spatial layout control within a training-free framework; (2) simultaneously alleviate missing objects and mismatched attribute issues.

**Key Insight**: The authors observe that the cross-attention maps of diffusion models inherently encode the spatial distribution of objects in the image. If the spatial distribution of these attention maps can be precisely controlled, the rendering locations of the objects can be controlled.

**Core Idea**: Leverage optimal transport theory to design a spatial transport cost function to dynamically "reposition" attention maps during the early stages of denoising, aligning them with the target spatial layout while focusing on detail refinement in the later stages.

## Method

### Overall Architecture
Input text prompt and target spatial layout $\rightarrow$ extract cross-attention maps during the early steps of the normal denoising process $\rightarrow$ apply Spatial Transport Optimization (STO) to adjust the spatial distribution of the attention maps based on the target layout $\rightarrow$ inject the optimized attention maps back into the denoising process $\rightarrow$ perform normal denoising in the later stages to refine details $\rightarrow$ output spatially aligned images.

### Key Designs

1. **Spatial Transport Optimization (STO)**:

    - **Function**: "Transport" the attention maps of objects from their current positions to the target spatial positions.
    - **Mechanism**: Model the redistribution of attention maps as an optimal transport problem. Given the current spatial distribution of attention maps and target positions, STO solves for a transport plan to move attention mass from current to target positions with minimal cost. In implementation, the Sinkhorn algorithm is utilized to approximate the optimal transport. After obtaining the transport matrix, the spatial distribution of the attention map is updated. This ensures that the reallocation of attention mass is smooth and globally optimized.
    - **Design Motivation**: Directly modifying attention maps (such as hard thresholding or scaling) violates the continuity of the distribution, leading to generation artifacts. Optimal transport provides a mathematically elegant and practically smooth solution for spatial redistribution.

2. **Spatial Transport Cost**:

    - **Function**: Define the cost metric for transporting attention mass across different spatial locations.
    - **Mechanism**: The ST cost function comprehensively considers two factors: (a) spatial distance — the Euclidean distance between the center of the object attention map and the target position; (b) spatial compactness — encouraging the attention distribution to concentrate on the target region rather than dispersing across the entire image. The cost function is designed such that locations further from the target have higher transport costs, thereby guiding the attention mass to efficiently gather at the correct positions.
    - **Design Motivation**: Simply using spatial distance as a cost is not fine-grained enough — it also needs to consider the level of attention concentration. The design of the cost function enhances the model's ability to "understand" spatial locations.

3. **Phase-aware Guidance Strategy**:

    - **Function**: Apply varying intensities of spatial guidance during different stages of the denoising process.
    - **Mechanism**: Analysis reveals that the spatial layout is primarily determined during the early stages of denoising (large noise $\rightarrow$ coarse structures), while the later stages are responsible for detail refinement. Therefore, STORM applies spatial transport optimization only in the early steps (e.g., the first 30%-50%), and restores normal denoising in the later steps. This avoids image quality degradation caused by excessive intervention.
    - **Design Motivation**: Enforcing spatial constraints at all steps would override minor position adjustments during the later refinement phase, resulting in unnatural generated images.

### Loss & Training
STORM is a fully training-free method and does not introduce additional loss functions. The spatial transport optimization is embedded online inside the denoising sampling process, only modifying the spatial distribution of cross-attention maps.

## Key Experimental Results

### Main Results

| Method | Spatial Alignment Accuracy | Object Missing Rate ↓ | Attribute Matching Rate | FID |
|------|-------------|-----------|-----------|-----|
| Stable Diffusion | Low | High | Medium | Baseline |
| Attend-and-Excite | Medium | Improved | Improved | Slightly Higher |
| Layout Guidance | Good | Medium | Medium | Slightly Higher |
| **STORM (Ours)** | **Best** | **Lowest** | **Best** | **Comparable** |

### Ablation Study

| Configuration | Spatial Alignment | Generation Quality | Description |
|------|---------|---------|------|
| Full STORM | Best | High | Full method |
| w/o STO | Significant Drop | High | Proves STO is the core component |
| w/o ST Cost | Decrease | Medium | Cost function design is crucial |
| Full-step Guidance | Slightly Better | Significant Drop | Over-intervention harms quality |
| Late-stage Guidance Only | Significant Drop | High | Early stage determines layout |

### Key Findings
- STORM not only solves the object mislocation issue, but also brings auxiliary improvements to missing objects and mismatched attributes, indicating that spatial alignment is positively correlated with other generation quality metrics.
- Applying spatial guidance in the early stages (the first 30%-50%) yields the best performance, which is consistent with the theoretical analysis.
- Optimal transport achieves significantly better results than simple heuristic methods such as attention scaling or cropping, validating the necessity of the mathematical optimization framework.
- The method is effective across different T2I backbones (such as SD 1.5, SDXL).

## Highlights & Insights
- **Innovative Application of Optimal Transport**: Modeling the spatial redistribution of attention maps as an optimal transport problem is a highly elegant formulation. This idea can be generalized to other generation tasks requiring spatial control (e.g., video generation, 3D generation).
- **Insight into Phase-aware Guidance**: Exploring the denoising phase characteristic where "spatial layout is determined early while details are refined late" has guiding significance for all diffusion-based guidance methods.
- **Three-in-One Effect**: Improving three categories of problems (mislocation + missing objects + attribute mismatch) simultaneously under a single method demonstrates that spatial alignment is a more fundamental problem than previously realized.

## Limitations & Future Work
- Requires the user to specify the target spatial layout (e.g., bounding boxes or keypoints), which increases the barrier to entry.
- Solving optimal transport increases inference time overhead (due to Sinkhorn iterations), which might not be efficient enough for real-time generation scenarios.
- For multi-object scenarios with overlap or occlusion, separating and transporting attention maps can become challenging.
- Future work could consider automatically parsing spatial relationships from text to generate layouts, reducing the need for manual specification.
- Extending STO to 3D spatial control (e.g., depth-aware layout) is a promising direction.

## Related Work & Insights
- **vs Attend-and-Excite**: A&E solves the missing objects problem by maximizing the minimum attention value, but does not directly handle spatial positions. STORM's STO explicitly optimizes the spatial distribution, making it more precise.
- **vs Layout-guidance / GLIGEN**: These methods require additional training to adapt to spatial control, whereas STORM is completely training-free, offering greater flexibility.
- **vs BoxDiff**: BoxDiff also performs training-free spatial control but relies on simple attention constraints. STORM's optimal transport framework is more elegant and yields better performance.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ (Formulating attention map repositioning as optimal transport represents a completely fresh perspective with an elegant formulation)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Ablation is comprehensive, but could include more backbones and larger-scale evaluations)
- **Writing Quality**: ⭐⭐⭐⭐ (The method is clearly presented, and the problem definition is accurate)
- **Value**: ⭐⭐⭐⭐ (Solves an important yet overlooked problem in T2I generation with an elegant and practical method)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Finite Difference Flow Optimization for RL Post-Training of Text-to-Image Models](finite_difference_flow_optimization_for_rl_post-training_of_text-to-image_models.md)
- [\[NeurIPS 2025\] Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](../../NeurIPS2025/image_generation/training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)
- [\[CVPR 2025\] Stable Flow: Vital Layers for Training-Free Image Editing](stable_flow_vital_layers_for_training-free_image_editing.md)
- [\[CVPR 2025\] Minority-Focused Text-to-Image Generation via Prompt Optimization](minority-focused_text-to-image_generation_via_prompt_optimization.md)
- [\[CVPR 2025\] Noise Diffusion for Enhancing Semantic Faithfulness in Text-to-Image Synthesis](noise_diffusion_for_enhancing_semantic_faithfulness_in_text-to-image_synthesis.md)

</div>

<!-- RELATED:END -->
