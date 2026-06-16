---
title: >-
  [Paper Note] SemLayer: Semantic-aware Generative Segmentation and Layer Construction for Abstract Icons
description: >-
  [CVPR 2026][Segmentation][Diffusion Model] SemLayer is proposed as a generative model-based pipeline to restore semantic layered structures from flattened vector icons. The method redefines segmentation as a coloring task via a diffusion model, performs semantic completion of occluded regions, and determines layer order using Integer Linear Programming (ILP), a
tags:
  - CVPR 2026
  - Segmentation
  - Diffusion Model
date: 2026-05-08
content_hash: 913f27d9e60dead0
---
# SemLayer: Semantic-aware Generative Segmentation and Layer Construction for Abstract Icons

**Conference**: CVPR 2026  
**arXiv**: [2603.24039](https://arxiv.org/abs/2603.24039)  
**Code**: [https://xxuhaiyang.github.io/SemLayer/](https://xxuhaiyang.github.io/SemLayer/)  
**Area**: Segmentation / Vector Graphics  
**Keywords**: Vector layer construction, Semantic segmentation coloring, Amodal completion, Icon editing, Diffusion models

## TL;DR

SemLayer is proposed as a generative model-based pipeline to restore semantic layered structures from flattened vector icons. The method redefines segmentation as a coloring task via a diffusion model, performs semantic completion of occluded regions, and determines layer order using Integer Linear Programming (ILP), achieving improvements of +5.0 in mIoU and +16.7 in PQ.

## Background & Motivation

1.  **Background**: Vector icons are a cornerstone of modern design workflows, where designers typically organize semantically meaningful graphic elements into multiple editable layers. However, icons are often "flattened" during publishing and distribution, merging all layers into a single compound path and losing the original semantic hierarchy.

2.  **Limitations of Prior Work**: Once the semantic structure is lost, downstream operations such as recoloring, animation, and local editing become extremely difficult. Designers are forced to manually re-segment and reconstruct icons. Existing methods like SAM perform poorly on highly abstract black-and-white icons due to the lack of cues like texture, shading, and color, while optimization-based methods often generate excessively fragmented layers.

3.  **Key Challenge**: The extreme abstraction of icons means traditional visual understanding cues (texture, shading, depth) are almost entirely absent. Simultaneously, the task requires recovering complete geometries (including occluded areas) and determining the correct stacking order.

4.  **Goal**: To restore an editable semantic layered representation from flattened single-path/compound-path vector icons.

5.  **Key Insight**: Leverage the rich shape priors inherent in generative models (diffusion models) to compensate for the scarcity of icon-specific data and the absence of visual features.

6.  **Core Idea**: Redefine semantic segmentation as a coloring task (using a diffusion model to "color" a black-and-white icon to visually separate different semantic parts), then utilize a diffusion model to complete occluded regions, and finally use ILP to determine the layer order.

## Method

### Overall Architecture

SensLayer performs a specific task: taking a black-and-white vector icon that has been "flattened" into a single compound path and restoring it to the layered, individually editable semantic structure originally intended by the designer. The difficulty lies in the fact that icons lack conventional visual cues, and the overlapping relationship between components is lost.

The pipeline follows three sequential steps: first, the monochromatic icon is fed into a diffusion model for "coloring"—where different semantic parts are assigned different colors—and visible region masks $\{V_1, ..., V_K\}$ are extracted based on color thresholds. Second, another diffusion model completes the full geometry $\{A_1, ..., A_K\}$ for each occluded part. Finally, Integer Linear Programming (ILP) is used to infer the stacking order of these parts, and the results are re-vectorized into SVG format. These steps correspond to the three key designs below.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Flattened B&W Vector Icon<br/>(Single Compound Path)"] --> B["Semantic-aware Generative Segmentation<br/>Diffusion + conditional LoRA Coloring"]
    B -->|Separation by Color Threshold| C["Visible Region Masks<br/>{V₁ … V_K}"]
    C --> D["Amodal Layer Completion<br/>pix2gestalt Finetuning + Fragment Training"]
    D -->|Fragment Merging (IoU τ=0.7)| E["Amodal Component Shapes<br/>{A₁ … A_K}"]
    E --> F["Layer Order Optimization<br/>ILP Rewards Correct Occlusion / Punishes Incorrect Visibility"]
    F --> G["potrace Vectorization<br/>Editable Layered SVG"]
```

### Key Designs

**1. Semantic-aware Generative Segmentation: Reformulating "Segmentation" as "Coloring"**

Standard segmentors like SAM fail on abstract icons because the color, texture, and depth cues they rely on are virtually nonexistent. SemLayer shifts the paradigm: rather than forcing a segmentation model to predict boundaries on feature-deprived inputs, it reframes the task as coloring—a form generative models excel at. The model assigns a distinct color to each semantic part while maintaining the original structure. Consequently, the segmentation problem of "which pixels belong to the same part" becomes a problem of "which pixels are dyed the same color."

The implementation is based on the EasyControl framework with a conditional LoRA attached to a Diffusion Transformer. It takes binary outlines as control conditions and uses text prompts to guide coloring. The training objective follows the flow-matching loss $\mathcal{L}_{\text{FM}} = \mathbb{E}_{t,\epsilon} \|v_\theta(z_n, t, z_c) - (\epsilon - x_0)\|_2^2$. During inference, binary masks $\{V_1, ..., V_K\}$ are obtained by thresholding each color channel. The training data consists of 8,567 icon-coloring pairs derived from real SVGs and synthesized via GPT-4o + gpt-image-1. This is effective because diffusion models possess rich priors for shape semantics and color assignment, making coloring a "natural" task for them.

**2. Amodal Layer Completion: Restoring Occluded Parts**

The previous step yields only the "visible portions," but designers require complete shapes. For example, if one graphic overlaps another, the hidden part must be restored to prevent holes when the layers are reordered. Since amodal completion models for natural images suffer from a significant domain gap with black-and-white icons, the authors fine-tuned a pix2gestalt latent diffusion model specifically for icon styles.

The completion model consumes two conditions: a CLIP image embedding for high-level semantics and the VAE-encoded occluded patch concatenated with the mask for geometric constraints. A crucial training technique is "fragmented visibility"—where an occluded part may be split into multiple disconnected fragments $\{V^{(i)}\}$. During training, each fragment serves as an independent input, but all are supervised to recover the same complete shape. This many-to-one mapping forces the model to learn to "hallucinate the whole from a small piece." During inference, a merging step using an IoU threshold $\tau=0.7$ combines multiple completion results belonging to the same object. The SemLayer-Completion dataset contains 50,000 training triplets.

**3. Layer Order Optimization: Precision Inference of Stacking via ILP**

After completion, all components represent full shapes, but the "who is on top" stacking order remains undetermined. The paper formalizes this as an Integer Linear Programming (ILP) problem. Binary variables $x_{ij}$ represent whether part $i$ is stacked above part $j$, subject to anti-symmetry and transitivity constraints (i.e., a legal total order).

Quality is determined by two pixel-level coverage variables: $y_i=1$ if the extra completion area $E_i = A_i \setminus I$ (the part not in the original image) is indeed covered by an upper layer—this is a "correct occlusion" that is rewarded; $z_i=1$ if a part's originally visible area $V_i$ is erroneously covered by an upper layer—this is an "incorrect occlusion" that is penalized. The objective function:

$$\max_{x,y,z} \sum_i y_i - \lambda \sum_i z_i$$

balances "completed parts should be hidden" against "originally visible parts should remain visible," with $\lambda=1$. The combinatorial optimization is solved precisely via an ILP solver.

### Example Scenario

Consider an "envelope + letter" icon: The input is a single compound B&W path merging the envelope and letter. **Step 1 (Segmentation)**: The model colors the envelope blue and the visible letter yellow, producing two visible masks—where the letter $V_{\text{letter}}$ is just a small strip protruding from the envelope. **Step 2 (Completion)**: This fragment is fed into the completion model, which uses amodal priors to restore the entire rectangular letter $A_{\text{letter}}$ hidden behind the envelope. **Step 3 (ILP)**: The extra region $E_{\text{letter}}$ should be covered by the envelope ($y$ reward), while the envelope's visible area should not be pressed by the letter ($z$ penalty). The solver determines "envelope on top, letter on bottom." Finally, potrace generates two vector layers ready for coloring, translation, or animation.

### Loss & Training

The segmentation model was trained for 40,000 steps (lr $1 \times 10^{-4}$, CFG scale 4.5), with 25 inference steps at $512 \times 512$ resolution. The completion model was fine-tuned for 50,000 steps (lr $1 \times 10^{-5}$), with 50 inference steps at $256 \times 256$ resolution. Experiments were conducted on 8 A100 GPUs. Vectorization uses potrace with a curve-reuse strategy to preserve original Bézier segments.

## Key Experimental Results

### Main Results

Segmentation Performance Comparison (48 real SVG test-sets):

| Method | mIoU (%) | PQ (%) | Completion mIoU (%) | Completion CD ↓ |
|------|----------|--------|--------------|----------|
| gpt-image-1 | 25.4 | 6.20 | 60.9 | 71.4 |
| SAM2 | 51.1 | 26.2 | 69.2 | 61.7 |
| SAM2* (Finetuned) | 79.3 | 59.4 | 80.7 | 49.1 |
| **Ours** | **84.3** | **76.1** | **85.2** | **46.6** |

Completion Model Comparison (Fixed visible inputs):

| Method | mIoU (%) ↑ | CD ↓ |
|------|-----------|------|
| gpt-image-1 | 10.7 | 98.6 |
| MP3D | 70.5 | 79.4 |
| MP3D-finetuned | 75.3 | 68.9 |
| **Ours** | **85.2** | **46.6** |

### Ablation Study

Refined segmentation metric improvements:

| Configuration | mIoU_Refined (%) | PQ_Refined (%) |
|------|-----------------|----------------|
| gpt-image-1 | 57.2 | 39.3 |
| SAM2 | 62.2 | 37.8 |
| SAM2* | 85.3 | 78.0 |
| **Ours** | **86.4** | **78.3** |

### Key Findings

-   **Coloring-as-Segmentation significantly outperforms direct segmentation**: Ours exceeds fine-tuned SAM2* by +5.0 mIoU and +16.7 PQ.
-   **gpt-image-1 performs poorly on icon segmentation**: With an mIoU of only 25.4%, it suggests general-purpose generative models struggle to grasp icon semantic structures.
-   **Domain adaptation is vital for completion**: The generic MP3D model achieved mIoU=70.5%, increasing to 75.3% after fine-tuning, while the specialized icon completion training reached 85.2%.
-   **Fragmented visibility training is effective**: The many-to-one mapping allows models to recover full shapes from single fragments.
-   **The end-to-end pipeline outputs editable layered SVGs**: Specifically supporting local recoloring, rotation, scaling, and simple animations.

## Highlights & Insights

-   **The "Segmentation via Coloring" paradigm shift is ingenious**: When traditional segmentation fails in specific domains, one should consider what task format is most "friendly" to generative models. Coloring is a more natural task for diffusion models, an insight transferable to other domains where direct segmentation is difficult.
-   **Practical data construction pipeline**: By utilizing LayerPeeler's real SVGs and GPT-mediated synthesis, the authors built a segmentation dataset of 8,567 samples and 50,000 completion triplets with low manual cost.
-   **Intuitive ILP design for layer ordering**: The objective function, which rewards correct occlusion and punishes incorrect visibility, is elegantly simple.

## Limitations & Future Work

-   **Limited to B&W line icons**: Colored and filled icons are not yet covered (though the authors note that color itself provides strong semantic cues, making expansion relatively straightforward).
-   **Highly tangled/occluded icons may fail**: The paper acknowledges failure cases in Fig. 9.
-   **Small test set**: Evaluation on only 48 icons may not fully represent all icon styles.
-   **Stochasticity of generative models**: Requires multiple runs or averaging to stabilize results.

## Related Work & Insights

-   **vs LayerPeeler**: LayerPeeler provides the source of real layered SVG data but lacks a robust segmentation method; SemLayer builds a complete segmentation-completion-sorting pipeline on top of it.
-   **vs SAM2**: Even after fine-tuning, SAM2 suffers from fragmentation and alignment issues because its design assumes rich visual cues; the coloring paradigm avoids these pitfalls.
-   **vs Optimization-based Vectorization**: Methods like DiffVG provide visual fidelity but generate excessively fragmented layers, lacking semantic consistency.

## Rating

-   Novelty: ⭐⭐⭐⭐⭐ The insight of redefining segmentation as a coloring task is highly creative.
-   Experimental Thoroughness: ⭐⭐⭐ Quantitative evaluation on 48 icons is somewhat limited, though qualitative visualizations are compelling.
-   Writing Quality: ⭐⭐⭐⭐ Problem formalization is clear, with solutions directly addressing the four defined challenges.
-   Value: ⭐⭐⭐⭐ High practical value for design tools; the dataset and method lay a foundation for vector graphics understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[CVPR 2026\] MatchMask: Mask-Centric Generative Data Augmentation for Label-Scarce Semantic Segmentation](matchmask_mask-centric_generative_data_augmentation_for_label-scarce_semantic_se.md)
- [\[CVPR 2026\] Conversational Image Segmentation: Grounding Abstract Concepts with Scalable Supervision](conversational_image_segmentation_grounding_abstract_concepts_with_scalable_supe.md)
- [\[CVPR 2026\] Frequency-Aware Affinity for Weakly Supervised Semantic Segmentation](frequency-aware_affinity_for_weakly_supervised_semantic_segmentation.md)
- [\[CVPR 2026\] Making Training-Free Diffusion Segmentors Scale with the Generative Power](making_training-free_diffusion_segmentors_scale_with_the_generative_power.md)
- [\[CVPR 2026\] SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data](sgma_semantic-guided_modality-aware_segmentation_for_remote_sensing_with_incompl.md)

</div>

<!-- RELATED:END -->
