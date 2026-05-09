---
title: >-
  [Paper Note] SketchDeco: Training-Free Latent Composition for Precise Sketch Colourisation
description: >-
  [CVPR 2026][LLM/NLP][sketch colourisation] SketchDeco is a training-free line-art colorization method that uses a global-local two-stage strategy with region masks and color palettes as precise control signals, leveraging diffusion model inversion and self-attention injection in latent space for region-accurate coloring with harmonious global transitions, completing in 15–20 steps on consumer GPUs.
tags:
  - CVPR 2026
  - LLM/NLP
  - sketch colourisation
  - diffusion models
  - training-free
  - latent composition
  - self-attention injection
date: 2026-05-08
content_hash: 37a8d411c8d57278
---

# SketchDeco: Training-Free Latent Composition for Precise Sketch Colourisation

**Conference**: CVPR 2026  
**arXiv**: [2405.18716](https://arxiv.org/abs/2405.18716)  
**Code**: N/A  
**Area**: LLM / NLP (Other)  
**Keywords**: sketch colourisation, diffusion models, training-free, latent composition, self-attention injection

## TL;DR

SketchDeco is a training-free line-art colorization method that uses a global-local two-stage strategy with region masks and color palettes as precise control signals, leveraging diffusion model inversion and self-attention injection in latent space for region-accurate coloring with harmonious global transitions, completing in 15–20 steps on consumer GPUs.

## Background & Motivation

Line-art colorization is a fundamental task in creative workflows such as animation storyboarding, product design, and concept art. Despite breakthroughs in image generation from large-scale diffusion models, fine-grained, region-level color control remains challenging:

**Spatial ambiguity of text guidance**: Text prompts are semantically rich but cannot precisely specify "which region gets which color," frequently causing color bleeding and semantic errors (as shown in Fig. 1b).

**Low efficiency of traditional methods**: Manually assigning colors or reference-based color transfer is too tedious.

**High training overhead**: ControlNet-based methods require fine-tuning hypernetworks at significant computational cost.

The core insight of this paper: the solution is not more training, but an innovative training-free composition framework — separating global consistency and local control processing.

## Method

### Overall Architecture

SketchDeco employs a two-stage divide-and-conquer strategy (Fig. 2):

- **Input**: Line art $\mathcal{S}$, region mask set $\{\mathcal{M}^{(i)}\}$, corresponding color palettes $\{\mathcal{P}_H\}$
- **Global Coloring Stage** (Sec 3.2): Generates multiple globally colored results, maintaining line-art structure and color consistency
- **Local Coloring Stage** (Sec 3.3): Achieves region-precise coloring and smooth transitions through latent space composition

### Key Designs

1. **Global Line-Art Coloring Module**: No per-region manual color assignment needed

    - **Line-art semantic prediction**: Uses BLIP-2's VQA capability to infer the line-art category (e.g., "What is this line art depicting?"), providing zero-shot generalization
    - **Color name search**: Maps hexadecimal color codes to the nearest color names via the CSS3 color database (147 colors) + K-D Tree (K=3)
    - **Line-art to image generation**: Uses a pre-trained Scribble ControlNet, combining semantic labels and color names to construct prompts, generating n+1 images: n colored results corresponding to each palette + 1 color-description-free auxiliary background image
    - **User interaction refinement**: Renders previews in pixel space, allowing users to switch random seeds and compare variants

2. **Local Line-Art Coloring Module**: First to reformulate this task as image composition + reconstruction

    - **Parallel local color composition**: For each mask region, the corresponding region is cropped from the global colored result and merged with the background image to form a composite image $\mathcal{I}^*$
    - **ODE inversion to latent space**: Uses DPM-Solver++ (15–20 step efficient inversion) to convert the composite image into a noise latent variable $z^*$; uses exceptional prompts instead of empty prompts to resolve CFG-induced reconstruction errors
    - **Gaussian noise injection**: Injects additional noise at mask boundary transition regions, leveraging the diffusion model's generative prior for natural inpainting of transition areas
    - **Self-attention injection**: Uses the composite image's self-attention maps $\mathcal{A}^*_{l,t}$ with a scaling factor $\tau$ — in early steps $t \in [T, T(1-\tau)]$, self-attention is injected to maintain global fidelity; in later steps $t \in [T(1-\tau), 0]$, text encoding is used for smooth color transitions

3. **Exceptional Prompt Technique**: Solves CFG instability in ODE inversion

    - Sets all token indices to a uniform value, removing extraneous positional encoding and special tokens
    - Makes the reverse ODE trajectory closer to the forward trajectory, significantly improving inversion accuracy

### Loss & Training

This method is entirely training-free. All components utilize pre-trained Stable Diffusion v1.5 and Scribble ControlNet without any fine-tuning. Key hyperparameters:
- CFG scale = 2.5
- $\tau = 0.4$ (self-attention injection ratio)
- K-Means clustering K=4 for extracting dominant colors
- Inference completed on a single RTX 4090 Super

## Key Experimental Results

### Main Results

**Local Coloring** (Table 1)

| Method | Place365 Indoor FID↓ | LPIPS↓ | DCCW↓ | PascalVOC FID↓ | DCCW↓ |
|--------|---------------------|--------|-------|----------------|-------|
| ColorizeDiffusion | 151.52 | 0.645 | 15.30 | 110.80 | 24.37 |
| ColorFlow | 354.07 | 0.643 | 17.05 | 367.69 | 14.98 |
| MangaNinja | 134.57 | 0.548 | 15.19 | 289.21 | 10.61 |
| Cobra | 221.38 | 0.603 | 14.96 | 382.70 | 13.96 |
| **SketchDeco** | **123.87** | **0.527** | **11.85** | **95.64** | **8.89** |

**Global Coloring** (Table 2, AFHQ-cat/dog)

| Method | AFHQ-cat FID↓ | LPIPS↓ | SSIM↑ | AFHQ-dog FID↓ |
|--------|--------------|--------|-------|---------------|
| DiffBlender | 86.82 | 0.811 | 0.032 | 145.50 |
| T2I-Adapter | 68.95 | 0.706 | 0.134 | 107.12 |
| T2I-Adapter+IDeepColor | 68.41 | 0.673 | 0.133 | 116.95 |
| **SketchDeco** | **50.31** | **0.671** | **0.187** | **89.70** |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| No self-attention injection | Color inconsistency, structure loss | Relying solely on initial noise is insufficient |
| No exceptional prompt | Large reconstruction error | CFG instability causes ODE trajectory deviation |
| No Gaussian noise injection | Harsh region boundaries | Transition regions lack generative prior |
| $\tau=0.4$ (default) | Best balance | Structure preservation + color transition |
| K=4 clustering | Optimal | K too large → palette redundancy; K too small → insufficient color coverage |

### Key Findings

- SketchDeco leads comprehensively on the DCCW metric (color palette similarity) for local coloring, demonstrating precise color control capability
- The method is effective across multiple domains (animals, indoor/outdoor scenes, anime, multi-object natural scenes)
- K-D Tree outperforms LLMs for color retrieval (no LLM calls needed to interpret hexadecimal color codes)
- Mask count does not affect final quality, as multiple masks are handled through parallel global coloring

## Highlights & Insights

1. **Clever divide-and-conquer strategy**: Decoupling the contradiction between global consistency and local control into two stages is the most inspiring design concept in this work
2. **Latent space composition paradigm**: Reformulating region coloring as composition + reconstruction avoids the difficulty of forcing color constraints directly during denoising
3. **Control without training**: Fully leveraging pre-trained models' generative prior demonstrates the rich controllability of diffusion models
4. **Elegant solution to CFG instability**: The exceptional prompt technique stabilizes inversion by removing prompt information — a practical technique for the ODE inversion domain
5. **Creative workflow friendly**: Supports interactive previews and seed switching, completing in 15–20 steps on consumer GPUs, suitable for practical creative workflows

## Limitations & Future Work

- Based on Stable Diffusion v1.5, with generation quality limited by the base model's capability; upgrading to SDXL or SD3 could yield greater improvements
- Masks need to be manually drawn (e.g., in Photoshop); automatic semantic segmentation for mask generation could lower the usage barrier
- K-D Tree maps to only 147 CSS3 color names, limiting fine-grained color control
- Texture detail in colorization results depends on ControlNet's generative capability, which may be limited for minimal line art
- Video line-art colorization and temporal consistency scenarios are not discussed

## Related Work & Insights

- Extends TF-ICON's composition framework by introducing region masks and palette control
- The exceptional prompt technique originates from TF-ICON; this work applies it for ODE inversion stabilization
- The choice of DPM-Solver++ reflects considerations of efficiency vs inversion accuracy trade-offs (vs DDIM's 100–250 steps)
- The methodological approach is generalizable to other image editing tasks: e.g., regional style transfer, local material replacement

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Reformulating colorization as a latent space composition problem is a novel perspective, though individual components are clever combinations of existing techniques
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-dataset, multi-comparison methods, including both global and local settings
- **Writing Quality**: ⭐⭐⭐⭐ — Clear diagrams, detailed pipeline description, well-motivated analysis
- **Value**: ⭐⭐⭐⭐ — Direct practical value for creative workflows, with training-free being a core advantage

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SubSpec: Speculate Deep and Accurate — Lossless and Training-Free Acceleration for Offloaded LLMs](../../NeurIPS2025/llm_nlp/speculate_deep_and_accurate_lossless_and_training-free_acceleration_for_offloade.md)
- [\[AAAI 2026\] An Invariant Latent Space Perspective on Language Model Inversion](../../AAAI2026/llm_nlp/an_invariant_latent_space_perspective_on_language_model_inve.md)
- [\[AAAI 2026\] Guess or Recall? Training CNNs to Classify and Localize Memorization in LLMs](../../AAAI2026/llm_nlp/guess_or_recall_training_cnns_to_classify_and_localize_memorization_in_llms.md)
- [\[ICLR 2026\] PT2-LLM: Post-Training Ternarization for Large Language Models](../../ICLR2026/llm_nlp/pt2-llm_post-training_ternarization_for_large_language_models.md)
- [\[CVPR 2026\] Perception Programs: Unlocking Visual Tool Reasoning in Language Models](perception_programs_visual_tool_reasoning.md)

</div>

<!-- RELATED:END -->
