---
title: >-
  [Paper Note] DICE: Distilling Classifier-Free Guidance into Text Embeddings
description: >-
  [AAAI 2026 **(Oral)**][Image Generation][Classifier-Free Guidance] This paper proposes DICE, which trains a lightweight sharpener with only 2M parameters to distill the guidance effect of CFG into text embeddings, enabling guidance-free sampling to achieve generation quality on par with CFG while halving inference computation. The method is comprehensively validated across multiple variants of SD1.5, SDXL, and PixArt-α, and is accepted as an AAAI 2026 Oral presentation.
tags:
  - "AAAI 2026 **(Oral)**"
  - "Image Generation"
  - "Classifier-Free Guidance"
  - "text embedding distillation"
  - "diffusion model acceleration"
  - "CFG-free sampling"
  - "text embedding sharpening"
date: 2026-05-08
content_hash: 05b8d7ee4b9ef162
---

# DICE: Distilling Classifier-Free Guidance into Text Embeddings

**Conference**: AAAI 2026 **(Oral)**  
**arXiv**: [2502.03726](https://arxiv.org/abs/2502.03726)  
**Code**: [https://github.com/zju-pi/dice](https://github.com/zju-pi/dice)  
**Area**: Image Generation / Diffusion Model Acceleration  
**Keywords**: Classifier-Free Guidance, text embedding distillation, diffusion model acceleration, CFG-free sampling, text embedding sharpening

## TL;DR

This paper proposes DICE, which trains a lightweight sharpener with only 2M parameters to distill the guidance effect of CFG into text embeddings, enabling guidance-free sampling to achieve generation quality on par with CFG while halving inference computation. The method is comprehensively validated across multiple variants of SD1.5, SDXL, and PixArt-α, and is accepted as an AAAI 2026 Oral presentation.

## Background & Motivation

Text-to-image diffusion models rely on Classifier-Free Guidance (CFG) to improve text-image alignment and generation quality. CFG requires two forward passes per sampling step—one conditional and one unconditional—effectively doubling the computational cost. For large models such as SDXL (2.6B parameters), this severely limits real-time applications. Existing CFG distillation approaches all exhibit notable shortcomings:

| Method | Trainable Parameters | Core Limitation |
|--------|---------------------|-----------------|
| Guided Distillation (GD) | 859M (full model fine-tuning) | Fine-tuned models cannot transfer to new scenario variants |
| Plug-and-Play (PnP) | 361M (auxiliary model) | Multiple operations required at inference, reducing practical speedup |
| Scaling embedding | 0 | Optimal scaling factor varies across models and prompts; quality cannot match CFG |

The core insight of DICE: scaling experiments first verify that corrective directions exist in the text embedding space that can enhance text-image alignment; further analysis reveals that **the essential role of CFG is to sharpen specific components of the text embedding** (primarily the semantically irrelevant padding tokens). Consequently, an extremely lightweight sharpener can be trained to perform sharpening directly in the embedding space, entirely bypassing the dual forward passes of CFG.

## Method

### Overall Architecture

DICE inserts a lightweight sharpener $r_\phi$ after the text encoder, transforming the original text embedding $\mathbf{c}$ into a sharpened version $\mathbf{c}_\phi = \mathbf{c} + \alpha \cdot r_\phi(\mathbf{c}, \mathbf{c}_\text{null})$. The sharpener contains only 2M parameters, operates independently of the diffusion model, and requires a single embedding correction. After training, sampling requires only a single denoising forward pass using $\mathbf{c}_\phi$ (without CFG), achieving generation quality equivalent to CFG.

### Key Designs

1. **Scaling validation experiment**: Multiplying the text embedding by a scaling factor $s$ yields significant quality improvement at $s=1.3$ on DreamShaper under guidance-free generation. However, the optimal $s$ varies across models and prompts, and simple scaling cannot reach CFG-level quality → a learned, dynamic, fine-grained correction is required.
2. **CFG direction distillation**: The CFG-enhanced noise prediction $\epsilon_\theta^{\omega,\mathbf{c}_\text{null}}(\mathbf{x}_t, \mathbf{c})$ serves as the teacher, while single-pass inference with the sharpened embedding $\epsilon_\theta(\mathbf{x}_t, \mathbf{c}_\phi)$ serves as the student. The training objective aligns the noise prediction directions of the two.
3. **Sharpening mechanism analysis**: Text embeddings consist of semantic tokens and padding tokens. DICE finds that **sharpening primarily amplifies the semantically irrelevant padding components**, enhancing fine-grained details while preserving core semantic information. Using only the sharpened padding embeddings already yields substantial quality improvements.
4. **Negative prompt support**: Negative prompts are integrated via $\mathbf{c}_\phi = \mathbf{c} + \alpha r_\phi(\mathbf{c}, \mathbf{c}_n) - \beta(\mathbf{c}_n - \mathbf{c}_\text{null})$, where $\beta$ controls the strength of semantic shift. Negative prompts are randomly sampled during training to improve robustness.

### Loss & Training

$$\mathcal{L}(\phi) = \mathbb{E}_{t \sim \mathcal{U}(0,T),\, \mathbf{x}_t \sim \mathcal{N}(\mathbf{x}_0, t^2 \mathbf{I})} \| \epsilon_\theta(\mathbf{x}_t, \mathbf{c}_\phi) - \epsilon_\theta^{\omega, \mathbf{c}_\text{null}}(\mathbf{x}_t, \mathbf{c}) \|$$

Only the sharpener parameters $\phi$ (2M) are optimized; the diffusion model $\theta$ is fully frozen. Training uses the same image-text dataset as the base model.

## Key Experimental Results

### Main Results: Quantitative Comparison on SD1.5 and Variants

| Method | NFE | Trainable Params | FID↓ | CLIP Score↑ | Aesthetic↑ | HPS v2.1↑ |
|--------|-----|-----------------|------|-------------|-----------|-----------|
| SD1.5 (ω=5, CFG) | 40 | - | 22.04 | 30.22 | 5.36 | 24.29 |
| SD1.5 (ω=1, no guidance) | 20 | - | 32.80 | 21.99 | 5.03 | 17.79 |
| Scaling (s=1.2) | 20 | - | 32.54 | 22.89 | 5.13 | 18.11 |
| GD (distillation) | 20 | 859M | 23.54 | 28.02 | 5.30 | 21.84 |
| PnP (distillation) | ≈28 | 361M | 26.57 | 27.72 | 5.39 | 23.17 |
| **DICE (Ours)** | **20** | **2M** | **22.22** | **28.54** | **5.28** | **22.78** |

### Ablation Study & Cross-Model Generalization

| Setting | FID↓ | CLIP Score↑ | Note |
|---------|------|-------------|------|
| DICE full (SD1.5) | 22.22 | 28.54 | Baseline |
| Sharpen semantic embedding only | >25 | <27 | Limited effect |
| Sharpen padding embedding only | ~23 | ~28 | Dominant contribution from padding |
| DreamShaper DICE (NFE=20) | 30.80 | 29.40 | vs. CFG (NFE=40) 30.35/30.50 |
| DreamShaper GD (NFE=20) | 32.53 | 28.48 | DICE significantly outperforms GD |
| SDXL DICE | - | - | Effective across architectures (UNet→DiT) |
| PixArt-α DICE | - | - | Effective across encoders (CLIP→T5) |

### Key Findings

- DICE at NFE=20 (half the computation) achieves FID even better than CFG guidance at NFE=40 (FID=22.04).
- On DreamShaper, DICE (NFE=20) achieves FID=30.80 vs. CFG (NFE=40) FID=30.35, with negligible gap.
- DrawBench text comprehension evaluation: DICE 23.32 vs. CFG 23.83, a minimal difference.
- Sharpening pattern analysis reveals that semantically irrelevant components are primarily amplified, suggesting that CFG fundamentally enhances fine-grained directions in the embedding space.

## Highlights & Insights

- **Revealing the underlying mechanism of CFG**: CFG ≈ text embedding sharpening; this finding carries independent theoretical significance.
- **Extreme parameter efficiency**: 2M parameters vs. GD's 859M and PnP's 361M, representing a 99.8% reduction.
- **2× inference speedup with negligible quality loss**: A drop-in replacement for CFG requiring no modification to the diffusion model architecture.
- The first CFG distillation method to support negative prompts, substantially improving practical utility.
- AAAI Oral recognition reflects strong community appreciation for the methodology.

## Limitations & Future Work

- The sharpening strength $\alpha$ requires adjustment for different base models (though it can be reused across variants of the same base model).
- Applicability to video diffusion models (e.g., SVD, CogVideoX) has not been verified.
- Performance under very high CFG scales (ω>15) has not been thoroughly validated.
- Training still requires access to image-text datasets and forward passes through the diffusion model.

## Related Work & Insights

| Category | Representative Methods | Relationship to DICE |
|----------|----------------------|----------------------|
| CFG distillation | GD, PnP | Distill into model parameters vs. DICE distills into embeddings, reducing parameter count by 99.8% |
| Sampling acceleration | LCM, DMD | Reduce sampling steps vs. DICE reduces per-step computation; orthogonal and composable |
| Guidance alternatives | PAG, APG | Still require additional computation vs. DICE completely eliminates extra forward passes |
| Embedding optimization | TextCraftor | Optimize text encoder weights vs. DICE trains an independent lightweight sharpener |

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Reveals the essence of CFG + embedding sharpening as a CFG substitute
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across SD1.5/SDXL/PixArt-α
- Writing Quality: ⭐⭐⭐⭐⭐ Oral-level presentation with thorough mechanistic analysis
- Value: ⭐⭐⭐⭐⭐ Immediate practical value for all diffusion models employing CFG

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Studying Classifier(-Free) Guidance From A Classifier-Centric Perspective](studying_classifier-free_guidance_from_a_classifier-centric_perspective.md)
- [\[ICCV 2025\] TeEFusion: Blending Text Embeddings to Distill Classifier-Free Guidance](../../ICCV2025/image_generation/teefusion_blending_text_embeddings_to_distill_classifier-free_guidance.md)
- [\[CVPR 2026\] CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance](../../CVPR2026/image_generation/cfg-ctrl_control-based_classifier-free_diffusion_guidance.md)
- [\[ICLR 2026\] Overshoot and Shrinkage in Classifier-Free Guidance: From Theory to Practice](../../ICLR2026/image_generation/overshoot_and_shrinkage_in_classifier-free_guidance_from_theory_to_practice.md)
- [\[CVPR 2026\] C$^2$FG: Control Classifier-Free Guidance via Score Discrepancy Analysis](../../CVPR2026/image_generation/c2fg_control_classifier-free_guidance_via_score_discrepancy_analysis.md)

</div>

<!-- RELATED:END -->
