---
title: >-
  [Paper Note] ByteEdit: Boost, Comply and Accelerate Generative Image Editing
description: >-
  [ECCV 2024][Image Generation][Image Editing] This work proposes ByteEdit, a framework that introduces human feedback learning into generative image editing (inpainting/outpainting). It improves editing quality through three reward models targeting aesthetics, alignment, and coherence, and accelerates inference utilizing adversarial training and progressive strategies.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Image Editing"
  - "Feedback Learning"
  - "Reward Model"
  - "Adversarial Training"
  - "Inference Acceleration"
date: 2026-05-08
content_hash: 092305f6d061c5f4
---

# ByteEdit: Boost, Comply and Accelerate Generative Image Editing

**Conference**: ECCV 2024  
**arXiv**: [2404.04860](https://arxiv.org/abs/2404.04860)  
**Code**: None (ByteDance internal system)  
**Area**: Image Generation  
**Keywords**: Image Editing, Feedback Learning, Reward Model, Adversarial Training, Inference Acceleration

## TL;DR

This work proposes ByteEdit, a framework that introduces human feedback learning into generative image editing (inpainting/outpainting). It improves editing quality through three reward models targeting aesthetics, alignment, and coherence, and accelerates inference utilizing adversarial training and progressive strategies.

## Background & Motivation

Generative image editing (outpainting and inpainting) based on diffusion models faces four key challenges in real-world applications:

**Insufficient Quality**: Generated images are suboptimal in realism, aesthetics, and detail fidelity.

**Poor Coherence**: The generated regions lack harmony with the original image in visual attributes such as color, style, and texture.

**Inadequate Instruction Following**: The model struggles to faithfully follow text instructions, leading to misalignment between the generated content and the input text.

**Low Generation Efficiency**: Slow inference speed makes it difficult to support large-scale editing tasks.

Existing methods (such as Imagen Editor, SmartBrush, and RePaint) typically address only a single issue. Inspired by the success of RLHF in the LLM domain, the authors introduce **human feedback learning** to generative image editing for the first time, systematically addressing the aforementioned four challenges.

## Method

### Overall Architecture

ByteEdit is designed around three objectives: "Boost-Comply-Accelerate". Given an input image $x$, a mask $m$ of the region of interest, and a text description $c$, the target is to generate an output that preserves the unmasked regions while aligning text and visual attributes within the masked regions. The framework contains three core components:

1. **Perceptual Feedback Learning (PeFL)**: An aesthetic reward model $R_\alpha$ to improve generation quality.
2. **Image-Text Alignment + Coherence**: An alignment reward model $R_\beta$ + a coherence reward model $R_\gamma$ to improve semantic alignment and pixel-level coherence.
3. **Adversarial & Progressive Training**: Adversarial training + progressive compression to accelerate inference.

### Key Designs

**1. Perceptual Feedback Learning (PeFL) — Boost**

- **Feedback Data Collection**: Extraction of over 1.5 million text prompts from Midjourney and MS-COCO. After K-Means clustering and t-SNE filtering, approximately 400k high-quality prompts are retained, with "best/worst" image pairs annotated by experts.
- **Aesthetic Reward Model $R_\alpha$**: Based on a BLIP backbone + cross-attention + MLP, trained using the Bradley-Terry preference objective:
$$\mathcal{L}(\alpha) = -\mathbb{E}[\log \sigma(R_\alpha(c, x_p) - R_\alpha(c, x_n))]$$
- **Stage-wise Feedback Optimization**: It is observed that the reward model's effectiveness varies across different denoising stages.
    - Stage 1 ($t \in [16,20]$): Exceeded noise levels; skipped, starting directly from $T_1=15$.
    - Stage 2 ($t \in [t', 15]$): Gradient-free inference, progressively denoising to obtain evaluable quality.
    - Stage 3 ($x_{t'} \to x_0'$): Single-step prediction of the final image, with the reward model guiding fine-tuning.
- **Total PeFL Loss**: $\mathcal{L}_{\text{pefl}} = \mathcal{L}_{\text{reward}} + \eta(\mathcal{L}_{\text{reg}} + \mathcal{L}_{\text{vgg}})$
    - Where the L1 regularization and VGG perceptual loss maintain the coherence of the original regions.

**2. Image-Text Alignment + Pixel-level Coherence — Comply**

- **Alignment Reward Model $R_\beta$**: Uses image-text pairs with low CLIPScore from LAION as negative samples, and descriptions regenerated with LLaVA as positive samples to construct ~40k triplets for training.
- **Coherence Reward Model $R_\text{\gamma}$**: A **pixel-level discriminator** based on ViT + MLP, which distinguishes real pixels from generated ones:
$$\mathcal{L}(\gamma) = -\mathbb{E}[\log \sigma(R_\gamma(z)) + \log(1 - \sigma(R_\gamma(z')))]$$
  - $z$ comes from the real image, and $z'$ comes from the generated image.
  - Pixel-level granularity captures coherence issues better than global evaluation.

**3. Adversarial & Progressive Training — Accelerate**

- **Adversarial Training**: The function of $R_\gamma$ is similar to a GAN discriminator, which can be trained online and serve as an adversarial objective:
$$\mathcal{L}_{\text{reward}}(\phi) = -\mathbb{E}\sum_{\theta \in \{\alpha, \beta, \gamma\}} \log \sigma(R_\theta(c, G_\phi(x, m, c, t')))$$
- **Progressive Training**: Progressively compresses inference steps
    - Phase 1: $T=20, T_1=15, T_2=10$
    - Phase 2: $T=8, T_1=6, T_2=3$
    - Done without distillation, relying solely on parameter inheritance and reward model supervision.

### Loss & Training

- Fine-tuning learning rate is 2e-6 with EMA decay of 0.9999.
- Training data consists of 7.56 million images, covering real-world scenes, portraits, and computer graphics (CG).
- Diversified masking strategies: global masking, irregular shapes, squares, and outpainting (expanding outward).
- Instance-level masking strategy: random dilation on instance segmentation results combined with random masks.

## Key Experimental Results

### Main Results

**Expert Evaluation Comparison (6000+ Image-Text Pairs/Tasks)**

| Method | Outpainting Coherence/Structure/Aesthetics | Editing Coherence/Structure/Aesthetics | Erasing Coherence/Structure |
|------|---------------------------|------------------------|-------------------|
| MeiTu | 3.01/2.73/2.75 | 2.77/2.89/2.51 | 3.31/3.25 |
| Canva | 2.72/2.85/2.65 | 3.42/3.40/3.08 | 2.92/2.90 |
| Adobe | 3.52/3.07/3.14 | 3.46/3.60/3.22 | 3.85/4.28 |
| **ByteEdit** | **3.54/3.25/3.26** | **3.73**/3.39/**3.25** | **3.99**/4.03 |

**Objective Metrics Comparison (EditBench)**

| Method | CLIPScore | BLIPScore |
|------|-----------|-----------|
| DiffEdit | 0.272 | 0.582 |
| BLD | 0.280 | 0.596 |
| EMILIE | 0.311 | 0.620 |
| **ByteEdit** | **0.329** | **0.691** |

### Ablation Study

- PeFL outperforms the baseline by approximately 60% in terms of structure and aesthetics on the outpainting task.
- Progressive acceleration reduces inference steps (from 20 steps to 8 steps) while maintaining quality.
- Adversarial training, conversely, improves both speed and quality on certain tasks (by stabilizing training and expanding supervisory coverage).

### Key Findings

1. Human feedback learning is systematically introduced to the field of image editing for the first time, achieving significant improvements.
2. The pixel-level coherence reward model, serving as an adversarial discriminator, can be jointly trained online.
3. GSB (Good/Same/Bad) preference rate: outpainting 105%, inpainting-editing 163%, and erasing 112% compared to Adobe.

## Highlights & Insights

1. **Trinity Reward Model Design**: Global aesthetics + global alignment + pixel-level coherence, covering quality dimensions across different granularities.
2. **Stage-wise PeFL**: Discovering that the reward model cannot effectively evaluate during high-noise stages, the method cleverly skips these stages and starts from intermediate steps.
3. **Two Birds with One Stone ($R_\gamma$)**: The coherence reward model provides feedback signals while simultaneously acting as a GAN discriminator.
4. **Distillation-free Acceleration**: Extremely few-step inference is achieved solely through progressive training + reward model supervision.

## Limitations & Future Work

1. Code and models are not publicly released, limiting reproducibility.
2. Evaluation relies heavily on subjective user studies and CLIP/BLIP scores, lacking more diverse objective benchmarks.
3. Elevated training cost (7.56 million images + 400k preference data points + multiple reward models).
4. Integration with acceleration techniques like LCM and SDXL-Turbo remains unexplored.
5. The work mainly focuses on inpainting/outpainting and has not yet expanded to instruction-based editing or video editing.

## Related Work & Insights

- **ImageReward / ReFL**: Feedback learning for text-to-image generation, but they do not consider coherence requirements in editing scenarios.
- **UFOGen / SDXL-Turbo**: Adversarial training to accelerate diffusion models; ByteEdit unifies this with reward models.
- **Insights**: Pixel-level discriminators can simultaneously serve the dual goals of quality evaluation and acceleration. In practical products, comprehensively optimizing multiple quality dimensions is more crucial than single-dimension optimization.

## Rating

- Novelty: ⭐⭐⭐⭐ (First work to introduce feedback learning into image editing + unification of adversarial training and reward models)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Large-scale user studies + comparisons with multiple commercial products)
- Writing Quality: ⭐⭐⭐ (Clear structure but some equations are redundant)
- Value: ⭐⭐⭐⭐ (Industrial-grade product solution)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Lazy Diffusion Transformer for Interactive Image Editing](lazy_diffusion_transformer_for_interactive_image_editing.md)
- [\[ECCV 2024\] RegionDrag: Fast Region-Based Image Editing with Diffusion Models](regiondrag_fast_region-based_image_editing_with_diffusion_models.md)
- [\[ECCV 2024\] FreeDiff: Progressive Frequency Truncation for Image Editing with Diffusion Models](freediff_progressive_frequency_truncation_for_image_editing_with_diffusion_model.md)
- [\[ECCV 2024\] Robust-Wide: Robust Watermarking against Instruction-driven Image Editing](robust-wide_robust_watermarking_against_instruction-driven_image_editing.md)
- [\[ECCV 2024\] Eta Inversion: Designing an Optimal Eta Function for Diffusion-based Real Image Editing](eta_inversion_designing_an_optimal_eta_function_for_diffusion-based_real_image_e.md)

</div>

<!-- RELATED:END -->
