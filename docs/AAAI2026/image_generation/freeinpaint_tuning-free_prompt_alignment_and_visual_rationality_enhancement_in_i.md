---
title: >-
  [Paper Note] FreeInpaint: Tuning-free Prompt Alignment and Visual Rationality Enhancement in Image Inpainting
description: >-
  [AAAI 2026][Image Generation][Image Inpainting] This paper proposes FreeInpaint, a plug-and-play, training-free method that optimizes the initial noise to steer attention toward the inpainting region (PriNo)…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Image Inpainting"
  - "Diffusion Models"
  - "Training-free Guidance"
  - "Initial Noise Optimization"
  - "Prompt Alignment"
date: 2026-05-08
content_hash: c39a209344c73e22
---

# FreeInpaint: Tuning-free Prompt Alignment and Visual Rationality Enhancement in Image Inpainting

**Conference**: AAAI 2026
**arXiv**: [2512.21104](https://arxiv.org/abs/2512.21104)
**Code**: [https://github.com/CharlesGong12/FreeInpaint](https://github.com/CharlesGong12/FreeInpaint)
**Area**: Image Generation
**Keywords**: Image Inpainting, Diffusion Models, Training-free Guidance, Initial Noise Optimization, Prompt Alignment

## TL;DR

This paper proposes FreeInpaint, a plug-and-play, training-free method that optimizes the initial noise to steer attention toward the inpainting region (PriNo), and during denoising decomposes the conditional distribution into three guidance terms — text alignment, visual rationality, and human preference (DeGu) — simultaneously improving prompt alignment and visual rationality in image inpainting.

## Background & Motivation

Text-guided image inpainting aims to generate new content in a specified region conditioned on a user-provided text prompt. Existing methods face two core tensions:

**Prompt alignment problem**: Existing inpainting models (e.g., SD-Inpainting, BrushNet) are trained with random masks and global captions, causing the model to rely more on image context than on the text prompt, frequently producing content inconsistent with the prompt.

**Visual rationality problem**: Even when prompt alignment is improved (e.g., HD-Painter via attention reweighting), visual rationality is often sacrificed, resulting in unnatural boundaries or distortions.

Through attention map visualization, the authors identify a key insight: **inpainting results that are well-aligned with the prompt exhibit cross-attention and self-attention highly concentrated on the mask region, whereas misaligned results show attention erroneously dispersed to background regions.** This "misdirected attention" is identified as the root cause of inpainting failure.

Furthermore, the authors note that diffusion models are highly sensitive to the initial noise — different random noise inputs lead to drastically different inpainting outcomes. Consequently, a well-chosen initial noise can substantially improve prompt alignment.

## Method

### Overall Architecture

FreeInpaint is a **plug-and-play framework requiring no training or fine-tuning**, consisting of two key stages:

- **Stage 1: Prior-Guided Noise Optimization (PriNo)** — optimizes the initial noise $z_T$ before denoising begins
- **Stage 2: Decomposed Training-free Guidance (DeGu)** — applies decomposed guidance at each denoising step

### Key Designs

#### 1. **PriNo: Prior-Guided Noise Optimization**

**Mechanism**: At the first denoising step, the distributional parameters of the initial noise $z_T$ are optimized so that attention maps concentrate on the mask region.

**Attention analysis**:
- Cross-attention $A^c$: measures the correlation between text tokens and visual patches. Incorrect $A^c$ associates prompt content with the background.
- Self-attention $A^s$: measures correlations among visual patches. Incorrect $A^s$ causes the inpainting region to be overly influenced by surrounding context.

**Loss function design**:

Cross-attention loss — encourages alignment between the prompt and the mask region:
$$\mathcal{L}_c = \sum_{i,j}[(1-M'_{ij}) \cdot A^c_{ij} - M'_{ij} \cdot A^c_{ij}]$$

Self-attention loss — encourages the inpainting region to attend to itself:
$$\mathcal{L}_s = \sum_{i,j}[(1-M'_{ij}) \cdot A^s_{ij} - M'_{ij} \cdot A^s_{ij}]$$

A key efficiency insight: the authors find that **the attention map at the first denoising step is already highly similar to the average over all steps**, so both losses need only be computed at the first step.

Joint optimization objective:
$$\mathcal{L}_{\text{joint}} = \lambda_1 \mathcal{L}_c + \lambda_2 \mathcal{L}_s + \lambda_3 \mathcal{L}_{KL}$$

where $\mathcal{L}_{KL}$ is a KL divergence regularizer preventing the optimized noise distribution from deviating too far from the standard Gaussian. The optimized noise is obtained as $z'_T = \mu' + \sigma' z_T$ by optimizing the mean $\mu$ and standard deviation $\sigma$.

#### 2. **DeGu: Decomposed Training-free Guidance**

**Core Idea**: The conditional distribution of the inpainting process is decomposed into three independent objectives, each guided by an off-the-shelf reward model.

**Conditional distribution decomposition**:
$$p(z_t|c, z^m, q) \propto p(c|z_t) \cdot p(z^m|z_t) \cdot p(q|z_t) \cdot p(z_t)$$

Three guidance objectives:
- **Text alignment $p(c|z_t)$**: uses local CLIPScore ($r_c$) to evaluate alignment between the mask region and the prompt
- **Visual rationality $p(z^m|z_t)$**: uses InpaintReward ($r_m$) to evaluate consistency between the generated region and the known region
- **Human preference $p(q|z_t)$**: uses ImageReward ($r_q$) to evaluate overall aesthetic quality

**Noise correction formula**:
$$\hat{\epsilon}_t = \epsilon_\theta(z_t,t,c,z^m,M') - \gamma_c\sqrt{\bar{\alpha}_t}\nabla_{z_t}r_c - \gamma_m\sqrt{\bar{\alpha}_t}\nabla_{z_t}r_m - \gamma_q\sqrt{\bar{\alpha}_t}\nabla_{z_t}r_q$$

#### 3. **Reward Modulator Design**

The paper uses $\sqrt{\bar{\alpha}_t}$ rather than the conventional $\sqrt{1-\bar{\alpha}_t}$ as the reward modulation coefficient. Since $\sqrt{\bar{\alpha}_t}$ increases monotonically during denoising, it down-weights the influence of unreliable predictions in early high-noise steps. Experiments confirm this outperforms conventional alternatives.

### Loss & Training

FreeInpaint **requires no training whatsoever**. The PriNo stage uses an SGD optimizer to iteratively optimize the noise distribution parameters; the DeGu stage corrects the predicted noise at each step using gradients from three pretrained reward models. The final result is blended with the original non-masked region.

## Key Experimental Results

### Main Results

**EditBench dataset (free-form masks)**

| Backbone | Method | ImageReward↑ | HPSv2↑ | L.CLIP↑ | InpaintReward↑ | LPIPS↓ |
|---------|------|-------------|--------|---------|----------------|--------|
| BrushNet | Base | 0.2729 | 25.34 | 26.45 | -0.1791 | 0.1947 |
| BrushNet | +HD-Painter | 0.3836 | 25.20 | 27.08 | -0.2124 | 0.2135 |
| BrushNet | +FreeInpaint | **0.5006** | **25.64** | **27.81** | **-0.0878** | 0.2005 |
| SD3-ControlNet | Base | 0.2993 | 25.48 | 26.26 | -0.2170 | 0.2155 |
| SD3-ControlNet | +HD-Painter | -0.5020 | 21.56 | 22.83 | -0.2988 | 0.2516 |
| SD3-ControlNet | +FreeInpaint | **0.5248** | **25.70** | **26.98** | **-0.0694** | **0.2057** |

**MSCOCO dataset (layout masks)**

| Backbone | Method | ImageReward↑ | HPSv2↑ | InpaintReward↑ | LPIPS↓ |
|---------|------|-------------|--------|----------------|--------|
| SD3-ControlNet | Base | 0.2795 | 26.51 | 0.0093 | 0.1008 |
| SD3-ControlNet | +FreeInpaint | **0.3422** | **27.10** | **0.0273** | **0.0680** |

### Ablation Study

| Configuration | ImageReward↑ | L.CLIP↑ | InpaintReward↑ |
|------|-------------|---------|----------------|
| BrushNet (baseline) | 0.2729 | 26.45 | -0.1791 |
| + PriNo only | 0.3785 | 26.96 | -0.2124 |
| + DeGu only | 0.3908 | 27.17 | -0.0643 |
| + PriNo + DeGu (full) | **0.5006** | **27.81** | **-0.0878** |
| Modulator: Constant 0.5 | 0.3533 | 26.92 | -0.1088 |
| Modulator: $\sqrt{1-\bar{\alpha}_t}$ | 0.3454 | 26.85 | -0.1146 |
| Modulator: $\sqrt{\bar{\alpha}_t}$ (Ours) | **0.5006** | **27.81** | **-0.0878** |

### Key Findings

1. PriNo and DeGu are each individually effective, and **their combination yields substantially superadditive gains**.
2. PriNo primarily improves prompt alignment but slightly degrades visual rationality; DeGu compensates for this shortcoming.
3. HD-Painter is incompatible with DiT architectures (SD3), causing ImageReward to collapse to -0.502, whereas FreeInpaint is **applicable to both U-Net and DiT architectures**.
4. In a user study, FreeInpaint achieved a **win rate of 64.52%**, substantially outperforming SDI (16.16%) and HD-Painter (19.32%).
5. The $\sqrt{\bar{\alpha}_t}$ modulator significantly outperforms conventional alternatives.

## Highlights & Insights

- The **discovery of misdirected attention** is highly insightful: by comparing aligned and misaligned samples, the paper reveals that the root cause of inpainting failure is attention being dispersed to non-mask regions.
- The **"optimize only the first-step attention"** efficiency design is elegant: the first-step attention map already approximates the full-step average, substantially reducing computational cost.
- **Decomposing the conditional distribution into three independent objectives** is a clean and principled formulation: each objective is handled by a dedicated pretrained reward model with no additional training required.
- As a plug-and-play solution compatible with five different inpainting model architectures, the method demonstrates strong generality.

## Limitations & Future Work

- Inference is slow: PriNo requires multiple iterative optimization rounds (up to 40 iterations per round), and DeGu must compute gradients through three reward models at every denoising step.
- Biases inherent in the reward models propagate into the inpainting results (e.g., known limitations of CLIPScore).
- The three guidance weights $\gamma_c, \gamma_m, \gamma_q$ in DeGu require tuning for different backbone models, incurring non-trivial hyperparameter search costs.
- Validation is limited to the Stable Diffusion family; the method has not yet been extended to more recent architectures such as Flux.

## Related Work & Insights

- The approach is related to initial noise optimization methods such as DOODL and InitNo, but is the first to apply this paradigm to inpainting and to design inpainting-specific attention losses.
- The method follows the spirit of classifier guidance, but innovatively decomposes the inpainting process into three conditional distributions.
- HD-Painter focuses solely on prompt alignment at the expense of visual quality; FreeInpaint resolves this trade-off through decoupled optimization.
- This work may inspire similar "decomposed guidance" strategies in other conditional generation tasks such as image editing and super-resolution.

## Rating

- Novelty: ⭐⭐⭐⭐ — The attention analysis insights and decomposed guidance concept are novel, though the basic components (noise optimization + reward guidance) have prior precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Five backbone models, two datasets, multiple mask types, and a user study; very comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with complete mathematical derivations.
- Value: ⭐⭐⭐⭐ — A highly practical plug-and-play solution, though inference efficiency remains a bottleneck for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Self-NPO: Data-Free Diffusion Model Enhancement via Truncated Diffusion Fine-Tuning](self-npo_data-free_diffusion_model_enhancement_via_truncated_diffusion_fine-tuni.md)
- [\[CVPR 2026\] From Inpainting to Layer Decomposition: Repurposing Generative Inpainting Models for Image Layer Decomposition](../../CVPR2026/image_generation/from_inpainting_to_layer_decomposition_repurposing_generative_inpainting_models_.md)
- [\[ICLR 2026\] Stochastic Self-Guidance for Training-Free Enhancement of Diffusion Models](../../ICLR2026/image_generation/stochastic_self-guidance_for_training-free_enhancement_of_diffusion_models.md)
- [\[AAAI 2026\] MACS: Multi-source Audio-to-Image Generation with Contextual Significance and Semantic Alignment](macs_multi-source_audio-to-image_generation_with_contextual_significance_and_sem.md)
- [\[AAAI 2026\] DogFit: Domain-guided Fine-tuning for Efficient Transfer Learning of Diffusion Models](dogfit_domain-guided_fine-tuning_for_efficient_transfer_learning_of_diffusion_mo.md)

</div>

<!-- RELATED:END -->
