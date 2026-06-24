---
title: >-
  [Paper Note] Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation
description: >-
  [ECCV 2024][Image Generation][Multi-Subject Image Generation] Be Yourself thoroughly analyzes the issue of multi-subject semantic leakage caused by cross-attention and self-attention in diffusion models, and proposes the Bounded Attention mechanism. By restricting the information flow between different subjects during the denoising process, it generates semantically independent multi-subject images, enabling the training-free generation of 5+ semantically similar subjects.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Multi-Subject Image Generation"
  - "Semantic Leakage"
  - "Attention Mechanism"
  - "Text-to-Image"
  - "Stable Diffusion"
date: 2026-05-08
content_hash: ed02ac0480ef1483
---

# Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation

**Conference**: ECCV 2024  
**arXiv**: [2403.16990](https://arxiv.org/abs/2403.16990)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Multi-Subject Image Generation, Semantic Leakage, Attention Mechanism, Text-to-Image, Stable Diffusion

## TL;DR

Be Yourself thoroughly analyzes the issue of multi-subject semantic leakage caused by cross-attention and self-attention in diffusion models, and proposes the Bounded Attention mechanism. By restricting the information flow between different subjects during the denoising process, it generates semantically independent multi-subject images, enabling the training-free generation of 5+ semantically similar subjects.

## Background & Motivation

1. **Background**: Text-to-image diffusion models (SD/SDXL) can generate high-quality images but often fail when handling complex prompts containing multiple subjects. Layout control methods like ControlNet specify positions using bounding boxes or segmentation maps, but still struggle to handle semantically similar subjects.

2. **Limitations of Prior Work**: Typical failures include three types: (1) catastrophic neglect (a subject completely disappears); (2) attribute binding errors (colors/textures assigned to the wrong subject); (3) subject fusion (multiple subjects merged into one). These issues are particularly severe among semantically or visually similar subjects (e.g., two cats of different colors).

3. **Key Challenge**: The attention layers are originally designed to fuse global information to generate coherent images, but this precisely leads to the "leakage" of distinct subject features—attention queries and keys of similar subjects become highly mixed, causing one subject's features to be "borrowed" by another. This is an inherent limitation at the architectural level.

4. **Goal**: (1) Systematically analyze the semantic leakage mechanism in both cross-attention and self-attention; (2) design a method to prevent harmful information leakage between subjects without compromising image quality.

5. **Key Insight**: The authors precisely identify where and why leakage occurs by visualizing cross-attention queries and self-attention maps via PCA, and then design targeted masking strategies accordingly.

6. **Core Idea**: Insert a masking matrix $\mathbf{M}_t$ in the attention computation during the denoising process to prevent queries belonging to different subjects from accessing other subjects' keys/values, allowing each subject to "be itself".

## Method

### Overall Architecture

Bounded Attention operates by alternating between two modes: (1) Bounded Guidance (early stage)—optimizing the latent via gradient descent to focus each subject's attention within its corresponding bounding box; (2) Bounded Denoising (entire process)—restricting the attention range using a masking matrix at each denoising step, and replacing the coarse bounding boxes with fine masks obtained from self-attention clustering in the later stages. Input: global prompt + subject list + bounding boxes.

### Key Designs

1. **Semantic Leakage Analysis (Cross-Attention)**:
    - **Function**: Reveal the leakage mechanism in cross-attention using PCA visualization.
    - **Mechanism**: Analyze the PCA projections of cross-attention queries for two subjects (e.g., a cat and a dog). The query distributions are separable when generated individually but become highly mixed when generated together. The higher the semantic similarity (e.g., hamster and squirrel), the more severe the mixture.
    - **Design Motivation**: Demonstrates that (1) semantic similarity causes queries to mix, thereby leading to feature leakage; (2) existing Layout Guidance methods force query separation by optimizing latents, which leads to out-of-distribution shifts and quality degradation.

2. **Semantic Leakage Analysis (Self-Attention)**:
    - **Function**: Reveal visual leakage in self-attention.
    - **Mechanism**: Visualize self-attention maps of specific pixels (e.g., eyes, legs), revealing that one subject's eyes strongly attend to another semantically similar subject's eyes, causing cross-subject visual feature leakage.
    - **Design Motivation**: The dense correspondence pattern causes semantically similar body parts to "reference" each other, which is beneficial for generating coherent single subjects but harmful for multi-subject generation.

3. **Bounded Attention Mechanism**:
    - **Function**: Prevent information leakage between subjects via attention masking.
    - **Mechanism**: Add a masking matrix in the attention computation: $\mathbf{A}_t^{(l)} = softmax(\mathbf{Q}_t^{(l)} \mathbf{K}_t^{(l)\top} + \mathbf{M}_t)$, where $\mathbf{M}_t[x,c] = -\infty$ blocks the corresponding information flow. In cross-attention, it prevents subject pixels from accessing text tokens of other subjects; in self-attention, it prevents subject pixels from accessing pixels of other subjects (while still allowing access to the background).
    - **Bounded Guidance Mode**: Optimize the loss $\mathcal{L}_i = 1 - \frac{\sum_{x \in b_i} \hat{A}[x, c]}{\sum_{x \in b_i} \hat{A}[x, c] + \alpha \sum_{x \notin b_i} \hat{A}[x, c]}$ to encourage each subject's attention to concentrate within its corresponding bbox. The hyperparameter $\alpha$ enhances attention to the background to prevent subjects from merging into the background.
    - **Bounded Denoising Mode**: In the later optimization phase, replace the coarse bounding boxes with fine masks obtained through self-attention clustering to avoid any collage-like artifact caused by coarse masking.
    - **Design Motivation**: (1) Does not modify the semantic distribution of queries/keys (unlike Layout Guidance which pushes queries apart), but only restricts information propagation; (2) allows subjects to interact with the background to maintain natural integration; (3) applied throughout the entire process (not just early steps) to prevent leakage during the detailed generation phase.

### Loss & Training

A training-free inference-only method. Validated on both SD and SDXL. Bounded Guidance is only applied during the early steps $[T, T_{guidance}]$, while Bounded Denoising is applied throughout the entire process.

## Key Experimental Results

### Main Results

| Method | Semantic Alignment↑ | Image Quality↑ | Layout Accuracy↑ |
|------|----------|----------|------------|
| Multi SD (Independent) | High | Low (Collage-like) | Medium |
| Layout Guidance | Medium | Medium (Quality degraded) | Medium |
| Attend-and-Excite | Medium | High | Low |
| **Ours** | **High** | **High** | **High** |

### Ablation Study

| Configuration | Success Rate | Description |
|------|--------|------|
| Cross-Attention masking only | 65% | Self-Attention leakage still exists |
| Self-Attention masking only | 58% | Cross-Attention semantic leakage dominates |
| Bounded Guidance only | 72% | Details still leak due to lack of masking in later stages |
| Bounded Denoising only | 75% | Unstable initial layout without guidance |
| Full Bounded Attention | **89%** | The two components complement each other |

### Key Findings

- **Cross- and self-attention leakages reinforce each other**: Addressing only one is insufficient; both must be handled simultaneously.
- **Semantic similarity displays hierarchical patterns**: It behaves differently across various UNet resolution layers—semantic similarity (e.g., cat vs. dog) mixes across all layers, whereas visual similarity (e.g., lizard vs. fruit) only mixes in high-resolution layers.
- **Leakage cannot be resolved by pushing queries apart**: Doing so pushes latents out of distribution, causing quality degradation or even catastrophic neglect.
- **Bounded Attention successfully generates 5+ similar subjects**: For instance, 5 kittens of different colors, which is completely unfeasible for existing methods.

## Highlights & Insights

- **Depth of problem analysis**: Systematically reveals the two sources and three levels of leakage through PCA visualization and attention map analysis, displaying analysis depth far exceeding concurrent works.
- **The "Do not alter distributions, only restrict information flow" design philosophy**: Contrasts sharply with latent optimization methods like Layout Guidance, being gentler yet much more effective.
- **Dynamic fine-mask updates**: Periodically updates masks in the Bounded Denoising stage using self-attention clustering, maintaining a balance between control precision and image naturalness.

## Limitations & Future Work

- User-provided bounding boxes are still required as input; automatic layout planning can be further explored.
- Performance may degrade when the number of subjects is extremely large (>8).
- Bounded Attention blocks some beneficial cross-subject interactions (such as illumination consistency).
- Needs a more systematic hyperparameter tuning strategy (e.g., $\alpha$ and guidance step counts).

## Related Work & Insights

- **vs Attend-and-Excite**: A&E reduces subject neglect by encouraging cross-attention maps, but it does not address self-attention leakage.
- **vs MultiDiffusion/SceneComposer**: Individually denoising and then merging avoids leakage but creates a collage-like/stitched appearance.
- **vs Dense Diffusion**: Maps attention to bboxes using masks, but the intervention is not strong enough to completely prevent leakage.
- The core concept of Bounded Attention can be transferred to multi-character control in video generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Deep leakage analysis and elegantly designed Bounded Attention.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive analysis, validated on both SD and SDXL architectures.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Tight logical analysis and excellent visualization.
- **Value**: ⭐⭐⭐⭐⭐ Significant value in both root-cause analysis and practical solutions for the multi-subject generation problem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [\[ECCV 2024\] ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs](ziplora_any_subject_in_any_style_by_effectively_merging_loras.md)
- [\[ECCV 2024\] AnyControl: Create Your Artwork with Versatile Control on Text-to-Image Generation](anycontrol_create_your_artwork_with_versatile_control_on_text-to-image_generatio.md)
- [\[ECCV 2024\] OMG: Occlusion-friendly Personalized Multi-concept Generation in Diffusion Models](omg_occlusion-friendly_personalized_multi-concept_generation_in_diffusion_models.md)
- [\[CVPR 2026\] MultiCrafter: High-Fidelity Multi-Subject Generation via Disentangled Attention and Identity-Aware Preference Alignment](../../CVPR2026/image_generation/multicrafter_high-fidelity_multi-subject_generation_via_disentangled_attention_a.md)

</div>

<!-- RELATED:END -->
