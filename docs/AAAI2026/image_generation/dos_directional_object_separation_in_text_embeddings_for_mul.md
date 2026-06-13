---
title: >-
  [Paper Note] DOS: Directional Object Separation in Text Embeddings for Multi-Object Image Generation
description: >-
  [AAAI 2026][Image Generation][multi-object image generation] This paper identifies four failure scenarios in multi-object generation (similar shapes/textures, dissimilar background biases, many objects)…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "multi-object image generation"
  - "text embedding separation"
  - "object mixing"
  - "CLIP embeddings"
  - "directional separation"
date: 2026-05-08
content_hash: a32a293af78cd03e
---

# DOS: Directional Object Separation in Text Embeddings for Multi-Object Image Generation

**Conference**: AAAI 2026
**arXiv**: [2510.14376](https://arxiv.org/abs/2510.14376)  
**Code**: [https://github.com/dongnami/DOS](https://github.com/dongnami/DOS)  
**Area**: Image Generation
**Keywords**: multi-object image generation, text embedding separation, object mixing, CLIP embeddings, directional separation

## TL;DR
This paper identifies four failure scenarios in multi-object generation (similar shapes/textures, dissimilar background biases, many objects), constructs directional separation vectors to modify three types of CLIP text embeddings (semantic token / EOT / pooled), achieves a 16–25% improvement in success rate and a 3–12% reduction in mixing rate on SDXL, with inference speed close to baseline (~4× faster than Attend-and-Excite).

## Background & Motivation
**Background**: T2I diffusion models frequently suffer from object neglect and object mixing in multi-object generation. Existing approaches fall into two categories — latent modification methods (Attend-and-Excite, CONFORM) require iterative gradient updates and are 4–5× slower at inference; text embedding modification methods (TEBOpt) are fast but limited in effectiveness.

**Limitations of Prior Work**: The CLIP text encoder uses a causal masking mechanism, causing information from earlier tokens to "leak" into the embeddings of later tokens, entangling multi-object information in the embedding space.

**Key Challenge**: Latent modification methods are effective but slow; embedding modification methods are fast but limited — a method that is both fast and effective in embedding space is needed.

**Goal**: (a) Systematically identify the key failure scenarios in multi-object generation; (b) effectively reduce object neglect and mixing through embedding-space operations without modifying the generation process.

**Key Insight**: Two key observations — CLIP's causal masking causes information entanglement, and differences between CLIP embeddings encode directional information. Based on these, "separation vectors" are constructed to push apart different objects in the embedding space.

**Core Idea**: For each object pair, compute a separation vector (directional difference), adaptively weight it by visual similarity between objects, and add it to three types of CLIP embeddings to achieve object separation.

## Method

### Overall Architecture
After the input prompt is encoded by CLIP, DOS modifies three types of embeddings before they are passed to the T2I model: semantic token embeddings $\bm{c}_{obj}^n$, EOT embeddings $\bm{c}_{EOT}$, and pooled embeddings $\bm{c}_{pool}$. Each embedding type is updated by adding a DOS vector — computed as a weighted average of separation vectors over all object pairs.

### Key Designs

1. **Identification of Four Failure Scenarios**:

    - Function: Systematically analyze failure modes in multi-object generation.
    - Core Findings: Similar Shapes (shape similarity → mixing), Similar Textures (texture similarity → mixing), Dissimilar Background Biases (divergent background preferences → neglect), Many Objects (many objects → cumulative effects). Experiments show success rate differences of 26.5%–60.5% across scenarios.
    - Design Motivation: Identifying the root causes enables targeted solutions.

2. **Separation Vector Construction (for Semantic Token Embeddings)**:

    - Function: Use embedding differences from clean prompts as the separation direction.
    - Mechanism: For object pair $(obj_n, obj_m)$, construct clean prompts "a {$obj_n$}" and "a {$obj_m$}" to obtain their respective clean embeddings; the difference serves as the separation vector $\mathbf{s}_{obj}^{(n,m)} = \bm{c}_{obj}^{pure,n} - \bm{c}_{obj}^{pure,m}$.
    - Design Motivation: Clean prompts avoid information entanglement caused by CLIP's causal masking, yielding uncontaminated object-specific representations.

3. **Separation Vector Construction (for EOT/Pooled Embeddings)**:

    - Function: Use embedding differences from contrastive prompts as the separation direction.
    - Mechanism: Construct contrastive prompts — "a {$obj_n$} separated from a {$obj_m$}" vs. "a {$obj_n$} mixed with a {$obj_m$}"; the difference $\mathbf{s}_{EOT/pool}^{(n,m)} = \bm{c}^{sep} - \bm{c}^{mix}$ encodes the "separated vs. mixed" direction.
    - Design Motivation: EOT/pooled embeddings inherently contain globally mixed information; contrastive prompts leverage this mixing mechanism to extract a separation direction.

4. **Adaptive Strength Computation**:

    - Function: Automatically adjust separation intensity based on visual similarity and background bias between object pairs.
    - Mechanism: Build attribute profiles using 42 shape/texture descriptors and 36 background phrases; compute Pearson correlation between object pairs. Adaptive weights are obtained via a shifted sigmoid: $\alpha_\tau^{(n,m)} = \max\{\sigma(\rho^{attr}), \sigma(1-\rho^{bg})\}$ — greater shape/texture similarity or larger background bias difference yields stronger separation.
    - Design Motivation: Different object pairs require different degrees of separation; a uniform strength is inappropriate.

5. **DOS Vector Aggregation and Embedding Update**:

    - Function: Compute a weighted average of separation vectors across all object pairs and add it to the original embeddings.
    - Mechanism: $\mathbf{v}_\tau^{DOS,n} = \frac{1}{N-1}\sum_{m \neq n} \alpha_\tau^{(n,m)} \mathbf{s}_\tau^{(n,m)}$; DOS vectors are added per object to semantic token embeddings, and the sum of DOS vectors across all objects is added to EOT/pooled embeddings.

## Key Experimental Results

### Main Results (SDXL)

| Benchmark | Base SR↑ | DOS SR↑ | Base MR↓ | DOS MR↓ |
|-----------|----------|---------|----------|---------|
| Similar Shapes | 48.0% | **64.0%** | 6.5% | **3.5%** |
| Similar Textures | 58.0% | **71.5%** | 7.5% | **3.5%** |
| Dissimilar BG Bias | 46.0% | **68.5%** | 22.5% | **17.0%** |
| Many Objects | 23.0% | **48.0%** | 27.5% | **15.5%** |

Human preference study (Table 2): DOS receives 43–55% of votes, far exceeding the second-best method (11–23%), with a margin of 26–43 percentage points.

Inference speed (Table 3): DOS 13.87s vs. A&E 58.83s vs. CONFORM 58.48s (~4× speedup).

### Ablation Study

| DOS Applied to | Similar Shapes SR | Many Objects MR |
|----------------|-------------------|-----------------|
| None (baseline) | 48.0% | 27.5% |
| Semantic token only | SR improves | MR improvement limited |
| EOT/pooled only | SR improves | MR decreases significantly |
| **All three types** | **64.0%** | **15.5%** |

### Key Findings
- **Three embedding types are complementary**: Semantic token embeddings improve object-specific representations, while EOT/pooled embeddings influence global spatial layout — both are necessary.
- **Adaptive strength is critical**: Removing adaptive weighting degrades performance, confirming the importance of stronger separation for high-risk object pairs.
- **Effective on SD3.5 as well**: SR improvements of 5.5–8.5% on a newer architecture validate the generality of the method.
- **No modification to the generation process = no artifacts**: Unlike latent modification methods, DOS does not deviate from the latent distribution seen during training.

## Highlights & Insights
- **Zero-cost embedding surgery** — Embeddings are modified only before being passed to the T2I model, leaving the generation process intact, with negligible inference overhead (+0.9s). This "surgery at the boundary" paradigm is elegant and transferable to any conditional generation task.
- **Exploiting CLIP's own flaw to fix it** — Information entanglement due to causal masking is an inherent limitation of CLIP, yet the authors cleverly use contrastive prompts to exploit the same mixing mechanism to extract separation directions — fighting fire with fire.
- **Rigorous adaptive strength design** — Object attribute profiles built from 42+36 descriptors, with fine-grained per-pair control implemented via Pearson correlation and a shifted sigmoid function.

## Limitations & Future Work
- **Dependence on CLIP encoder**: Embeddings from non-CLIP encoders (e.g., T5) are not handled, potentially limiting applicability to models that rely solely on T5.
- **Reliance on object parsing**: Correct identification of object nouns and their positions in the prompt is required.
- **Attribute binding not addressed**: The method primarily targets object neglect and mixing; attribute binding problems (e.g., "red hat, blue jacket") receive limited attention.
- **Future directions**: Extending separation vectors to attribute-object binding; automatically learning adaptive weights rather than relying on manual design.

## Related Work & Insights
- **vs. Attend-and-Excite**: A&E iteratively optimizes attention maps via latent modification — effective but 4× slower; DOS performs a one-shot modification in embedding space, achieving both higher speed and better performance.
- **vs. TEBOpt**: TEBOpt modifies only semantic token embeddings; DOS modifies all three embedding types with adaptive strength, achieving 10–24% higher SR.
- **vs. CONFORM**: CONFORM is also a latent modification method; DOS has the largest advantage on Dissimilar BG (+13% SR).

## Rating
- Novelty: ⭐⭐⭐⭐ Directional separation vector design is elegant; adaptive strength mechanism is well-considered.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four benchmarks + human preference + ablation + speed comparison + SD3.5 validation.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from problem analysis to method design to experimental validation is complete and clear.
- Value: ⭐⭐⭐⭐ Directly practical for multi-object generation; method is general and efficient.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Object-WIPER: Training-Free Object and Associated Effect Removal in Videos](../../CVPR2026/image_generation/object-wiper_training-free_object_and_associated_effect_removal_in_videos.md)
- [\[NeurIPS 2025\] SceneDesigner: Controllable Multi-Object Image Generation with 9-DoF Pose Manipulation](../../NeurIPS2025/image_generation/scenedesigner_controllable_multi-object_image_generation_with_9-dof_pose_manipul.md)
- [\[ICLR 2026\] Directional Textual Inversion for Personalized Text-to-Image Generation](../../ICLR2026/image_generation/directional_textual_inversion_for_personalized_text-to-image_generation.md)
- [\[AAAI 2026\] DICE: Distilling Classifier-Free Guidance into Text Embeddings](dice_distilling_classifier-free_guidance_into_text_embedding.md)
- [\[CVPR 2026\] EgoFlow: Gradient-Guided Flow Matching for Egocentric 6DoF Object Motion Generation](../../CVPR2026/image_generation/egoflow_gradient-guided_flow_matching_for_egocentric_6dof_object_motion_generati.md)

</div>

<!-- RELATED:END -->
