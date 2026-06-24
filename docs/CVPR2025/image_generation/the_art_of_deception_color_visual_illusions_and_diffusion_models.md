---
title: >-
  [Paper Note] The Art of Deception: Color Visual Illusions and Diffusion Models
description: >-
  [CVPR 2025][Image Generation][Visual Illusions] This paper discovers that the intermediate representations of diffusion models (particularly during the DDIM inversion process) naturally produce luminance/color shifts consistent with human perception. Based on this, a method for generating novel visual illusions using text-to-image diffusion models is proposed, with psychophysical experiments demonstrating that the generated illusions successfully deceive humans.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Visual Illusions"
  - "Diffusion Models"
  - "Human Perception"
  - "Luminance Illusions"
  - "DDIM Inversion"
date: 2026-05-08
content_hash: 02d9c6d270e7edb9
---

# The Art of Deception: Color Visual Illusions and Diffusion Models

**Conference**: CVPR 2025  
**arXiv**: [2412.10122](https://arxiv.org/abs/2412.10122)  
**Code**: [https://alviur.github.io/color-illusion-diffusion](https://alviur.github.io/color-illusion-diffusion)  
**Area**: Image Generation / Visual Perception  
**Keywords**: Visual Illusions, Diffusion Models, Human Perception, Luminance Illusions, DDIM Inversion

## TL;DR

This paper discovers that the intermediate representations of diffusion models (particularly during the DDIM inversion process) naturally produce luminance/color shifts consistent with human perception. Based on this, a method for generating novel visual illusions using text-to-image diffusion models is proposed, with psychophysical experiments demonstrating that the generated illusions successfully deceive humans.

## Background & Motivation

**Background**: Visual illusions are crucial windows for understanding the human visual system. Existing studies demonstrate that artificial neural networks, such as CNNs, are also susceptible to certain visual illusions, raising profound questions about the commonalities between human and machine vision.

**Limitations of Prior Work**: (1) Previous methods for creating artificial visual illusions primarily rely on image translation networks or classifier inner activations, which do not scale to high-resolution and natural images; (2) Existing works using diffusion models for illusion generation focus only on high-level cognitive illusions (such as hybrid images) rather than low-level luminance/color illusions; (3) Existing luminance illusion generation methods cannot control the perceptual effects of specific regions in natural images.

**Key Challenge**: Diffusion models are trained on natural image statistics, whereas visual illusions are specific stimuli that deviate from natural image statistics—how do models process such out-of-distribution samples?

**Goal**: (1) Reveal the internal behaviors of diffusion models when processing visual illusions; (2) Utilize this finding to generate controllable luminance/color visual illusions.

**Key Insight**: Observations of DDIM inversion trajectories show that when inverting a visual illusion image into the latent space, intermediate steps gradually introduce luminance shifts consistent with human perception. This parallels the behavior of Gaussianization flows when handling out-of-distribution samples.

**Core Idea**: Diffusion models trained on natural image distributions perform "statistical correction" on out-of-distribution visual illusion stimuli, and this correction aligns with the evolutionary adaptation mechanisms of the human visual system based on natural scene statistics.

## Method

### Overall Architecture

The proposed method consists of two parts: (1) Illusion Replication: performing DDIM inversion on known visual illusion images and predicting human perception by measuring luminance/color changes in target regions across intermediate steps; (2) Illusion Generation: applying optimization to each denoising step within the text-to-image generation process to guide the synthesis of natural images containing target regions that are physically identical but perceptually different in color/luminance.

### Key Designs

1. **DDIM Inversion as a Visual Perception Model**:

    - **Function**: Predicts human perceptual shifts of visual illusions via intermediate representations of diffusion models.
    - **Mechanism**: Given a visual illusion image $z_0$, DDIM inversion is performed to obtain a sequence of intermediate representations $\{z_t\}$. At each step, $z_t$ is decoded into the image space to measure the average intensity of target regions. After approximately 10-20 steps, physically identical gray squares on black/white backgrounds display luminance differences consistent with human perception. Theoretical explanation: A diffusion model trained on natural images fails to properly Gaussianize out-of-distribution illusion stimuli, causing intermediate representations to retain the bias of "statistical correction".
    - **Design Motivation**: The deterministic nature of DDIM inversion makes observations reproducible; intermediate representations (rather than the final representation) capture the progressive transition from physical representation to human perception.

2. **Visual Illusion Intensity Optimization**:

    - **Function**: Controls the perceptual effects of specific regions during the text-to-image generation process.
    - **Mechanism**: During the denoising steps, a user-specified target region $O(r)$ is superimposed on the intermediate representation: $z_{t_{\text{target}}}(r_i) = z_t(r_i) + O(r_i)$. Then, a perception loss $\mathcal{L}_{VI} = \sum_{c,r} \|m_{\text{int}}(z_{t_{\text{target}}}, r) - k_c\|$ is applied to drive the perceptual intensity of target regions towards the desired value $k_c$. Concurrently, a similarity loss $\mathcal{L}_{\text{sim}} = \text{MSE}(z_t(r), O(r))$ maintains visual harmony between the target regions and the background. $N$ gradient updates are performed at each step.
    - **Design Motivation**: Introducing small perturbations directly into the denoising process (rather than post-processing) allows the illusion effects to blend naturally with the generated content.

3. **Perceptual Accuracy Evaluation Framework**:

    - **Function**: Quantitatively evaluates the model's ability to replicate visual illusions.
    - **Mechanism**: Defines the Perceptual Accuracy Score (PAS) as the ratio of predictions aligning with the direction of human perception. For the BRI3L luminance illusion dataset, the intensity difference $\Delta I$ between the output and input of the target regions is calculated; replication is considered successful when $\Delta I < \tau \times I_{\text{int}}$. Evaluation across multiple thresholds $\tau$ is supported.
    - **Design Motivation**: While prior work mostly relies on qualitative demonstrations, PAS provides a comparable, quantitative metric.

### Loss & Training

The optimization loss is formulated as $\mathcal{L} = \gamma \mathcal{L}_{VI} + \beta \mathcal{L}_{\text{sim}}$, with gradient updates performed on the latent variables at each denoising step: $z'_t = z_t - \alpha \cdot \nabla_{z_t} \mathcal{L}$. DeepFloyd (a pixel-space diffusion model) is utilized to avoid the computational overhead of VAE decoding during backpropagation. Illusion replication uses Stable Diffusion 1.4 + DDIM inversion (10-20 steps).

## Key Experimental Results

### Main Results — Luminance Illusion Replication (BRI3L)

| Method | PAS (τ=0.8) ↑ | PAS (τ=0.9) ↑ | PAS (τ=1.0) ↑ |
|------|---------|---------|---------|
| ODOG (Classic Model) | 42.3% | 55.1% | 68.4% |
| CIWaM | 38.7% | 51.2% | 65.8% |
| DN-NET | 45.6% | 58.3% | 71.2% |
| RestoreNet | 47.2% | 59.8% | 72.5% |
| VAE only | 31.4% | 42.1% | 55.3% |
| DDIM (Ours) | **52.8%** | **64.7%** | **76.1%** |

### Psychophysical Verification

| Generation Method | Human Subject Deception Rate ↑ |
|---------|----------------|
| Classic Patterns | 89.2% |
| Text-to-Image Illusions in Natural Images | 73.5% |

### Key Findings

- Perceptual accuracy using only VAE encoder-decoders (without DDIM inversion) is significantly lower than that of the DDIM-based scheme (55.3% vs 76.1%), proving that **the DDIM denoising process** rather than VAE reconstruction drives the perceptual shift.
- Diffusion models outperform traditional vision science models (ODOG, CIWaM) and neural network methods (DN-NET, RestoreNet) across all evaluation metrics.
- Human psychophysical experiments confirm that the visual illusions embedded within generated natural images actually deceive humans with a 73.5% probability.

## Highlights & Insights

- **Highly Inspiring Core Discovery**: The spontaneous emergence of human-like perceptual shifts in the intermediate representations of diffusion models provides computational evidence from AI for the evolutionary paradigm of "human vision as statistical inference".
- **Originality of Methodology**: This is the first work to measure visual perception within the denoising trajectory rather than at the model output, opening a new interdisciplinary research direction.
- **Generating Controllable Visual Illusions in Natural Images** carries dual value for both academic research and practical applications.

## Limitations & Future Work

- Currently, only luminance and color illusions have been verified; other types, such as motion and shape illusions, remain to be explored.
- The number of DDIM inversion steps impacts the accuracy of perceptual prediction, with the optimal number being dataset-dependent.
- The efficacy of generated illusions in natural images (73.5%) is lower than in classic patterns (89.2%), suggesting that the statistical complexity of natural scenes increases control difficulty.
- Validation has only been performed on Stable Diffusion and DeepFloyd; other architectures (e.g., DiT) warrant further study.

## Related Work & Insights

- **vs DN-NET/RestoreNet**: These methods measure illusions in model outputs, whereas ours measures in intermediate representations, yielding better results.
- **vs Roy et al. (2024)**: That work uses text prompts to generate luminance illusions but struggles to control specific regions, whereas ours achieves precise region-level control through optimization.
- **Inspiration for Computational Neuroscience**: The denoising process of diffusion models may serve as a computational analog to human visual inference.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Cutting-edge findings intersecting diffusion models and visual perception research, opening a new direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative evaluation across multiple datasets + psychophysical validation, though some ablation studies could be deeper.
- Writing Quality: ⭐⭐⭐⭐⭐ Highly logical narrative flow from observation to theoretical explanation to application.
- Value: ⭐⭐⭐⭐ Holds significant scientific value for understanding the commonalities between machine and human vision.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Color Alignment in Diffusion](color_alignment_in_diffusion.md)
- [\[CVPR 2025\] GCC: Generative Color Constancy via Diffusing a Color Checker](gcc_generative_color_constancy_via_diffusing_a_color_checker.md)
- [\[CVPR 2025\] MangaNinja: Line Art Colorization with Precise Reference Following](manganinja_line_art_colorization_with_precise_reference_following.md)
- [\[CVPR 2025\] Science-T2I: Addressing Scientific Illusions in Image Synthesis](science-t2i_addressing_scientific_illusions_in_image_synthesis.md)
- [\[ECCV 2024\] ColorPeel: Color Prompt Learning with Diffusion Models via Color and Shape Disentanglement](../../ECCV2024/image_generation/colorpeel_color_prompt_learning_with_diffusion_models_via_color_and_shape_disent.md)

</div>

<!-- RELATED:END -->
