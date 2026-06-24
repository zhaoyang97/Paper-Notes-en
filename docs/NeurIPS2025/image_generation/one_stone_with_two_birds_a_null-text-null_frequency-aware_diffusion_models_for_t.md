---
title: >-
  [Paper Note] One Stone with Two Birds: A Null-Text-Null Frequency-Aware Diffusion Models for Text-Guided Image Inpainting
description: >-
  [NeurIPS 2025][Image Generation][Text-guided image inpainting] NTN-Diff is a frequency-aware diffusion model that decomposes the global semantic consistency problem into per-band consistency tasks over mid-frequency and low-frequency components. By adopting a "null-text–text–null-text" three-stage denoising strategy, the method simultaneously addresses two longstanding challenges in text-guided image inpainting: preserving unmasked regions and maintaining semantic consistency…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Text-guided image inpainting"
  - "frequency-aware"
  - "null-text denoising"
  - "diffusion models"
  - "semantic consistency"
date: 2026-05-08
content_hash: 2729782ce5368285
---

# One Stone with Two Birds: A Null-Text-Null Frequency-Aware Diffusion Models for Text-Guided Image Inpainting

**Conference**: NeurIPS 2025
**arXiv**: [2510.08273](https://arxiv.org/abs/2510.08273)  
**Code**: [GitHub](https://github.com/htyjers/NTN-Diff)  
**Area**: Diffusion Models / Image Inpainting
**Keywords**: Text-guided image inpainting, frequency-aware, null-text denoising, diffusion models, semantic consistency

## TL;DR

NTN-Diff is a frequency-aware diffusion model that decomposes the global semantic consistency problem into per-band consistency tasks over mid-frequency and low-frequency components. By adopting a "null-text–text–null-text" three-stage denoising strategy, the method simultaneously addresses two longstanding challenges in text-guided image inpainting: preserving unmasked regions and maintaining semantic consistency between masked and unmasked areas.

## Background & Motivation

Text-guided image inpainting aims to reconstruct masked region content according to a text prompt while leaving unmasked regions unchanged. Two persistent challenges exist in this task:

**Unmasked region preservation**: Ensuring that the inpainting process does not alter the content of unmasked regions.

**Masked/unmasked region semantic consistency**: Ensuring that the generated content in masked regions is semantically harmonious with the surrounding unmasked context.

Existing methods address only one of these challenges at a time. BLD (Blended Latent Diffusion) preserves unmasked regions via per-step blending operations, but the discrepancy between the diffusion and denoising processes causes semantic inconsistency between masked and unmasked regions. BrushNet maintains consistency through dense texture feature maps without cross-attention, but a separate text-guided denoising process corrupts the unmasked regions.

**Key Insight**: This tension stems from the differing robustness of frequency bands to text prompts during the denoising process:
- **Low-frequency bands** (color, illumination, etc.) are easily modulated by text prompts at high-noise stages and exhibit large fluctuations.
- **Mid-frequency bands** (layout, structure, etc.) are more robust to text prompts and stabilize early in the denoising process.

The core idea of this paper is to decompose global semantic consistency into per-band consistency tasks, leveraging the stability of the mid-frequency band as an "anchor" to guide low-frequency denoising.

## Method

### Overall Architecture

NTN-Diff divides the denoising process into an **early stage** (high noise, steps $T$ to $\lambda T$) and a **late stage** (low noise, steps $\lambda T$ to $0$), comprising four parallel/sequential denoising streams:

- (I) Null-text low-frequency-aware denoising (early stage)
- (II) Text-guided denoising (early stage)
- (III) Null-text mid-frequency-guided denoising (early stage)
- (IV) Text-guided denoising (late stage)

### Key Designs

1. **Null-text low-frequency-aware denoising (Sec. 2.3.1)**: Denoising is performed using the null text $c_\emptyset$ rather than the actual text prompt, preventing the low-frequency band from being modulated by the text. At each step, unmasked regions are replaced with the result of the diffusion process to enforce preservation:

$$\hat{z}_t^{un} = z_{T-t}^{gt} \odot m_z + z_t^{un} \odot (1 - m_z)$$

The self-attention mechanism is also modified to suppress attention scores in masked regions via masking, ensuring the inpainting process relies primarily on information from unmasked regions. This step yields low-frequency components that are uncontaminated by text conditioning.

2. **Text-guided denoising + low-frequency substitution (Sec. 2.3.2)**: A parallel text-guided denoising process aligns the mid-frequency band of the masked region with text semantics. The key operation is the **denoising low-frequency band layer**, which replaces the low-frequency components of the text-guided result with those from the null-text denoising stream:

$$\tilde{z}_t^{text} = \text{IDCT}(\text{DCT}(z_t^{un}) \odot m_{low} + \text{DCT}(z_t^{text}) \odot (1 - m_{low}))$$

The low-pass mask threshold is set adaptively as $th_{lp} = \lambda_{lp}^f + \lambda_{lp}^r \cdot \frac{\|M\|_1}{HW}$, positively correlated with the proportion of unmasked area (larger unmasked areas require more low-frequency substitution).

3. **Null-text mid-frequency-guided denoising (Sec. 2.3.3)**: The text-aligned mid-frequency band from step (II) is used to guide a new null-text denoising process. The **denoising mid-frequency band layer** injects mid-frequency components from the text-guided stream:

$$\tilde{z}_t^{in} = \text{IDCT}(\text{DCT}(z_t^{text}) \odot m_{mid} + \text{DCT}(z_t^{in}) \odot (1 - m_{mid}))$$

The mid-frequency band mask is extracted using a bandpass filter. This step enables low-frequency components (particularly those in masked regions) to be aligned with text semantics via mid-frequency guidance, without direct use of the text prompt.

4. **Late-stage text-guided denoising (Sec. 2.4)**: Final text-guided denoising is performed based on the output of the three early-stage streams, with unmasked regions replaced by diffusion process results at each step to maintain preservation. At this point, mid-frequency components have stabilized and low-frequency components have been pre-aligned via mid-frequency guidance, allowing text conditioning to effectively perform final refinement.

### Adaptive Frequency Band Extraction

The extraction thresholds for both low- and mid-frequency bands are linked to the unmasked area ratio $\frac{\|M\|_1}{HW}$: larger unmasked areas require more frequency information to be transferred from the null-text stream into the text-guided stream.

## Key Experimental Results

### BrushBench Inside Inpainting

| Method | IR×10↑ | HPS v2×10²↑ | PSNR↑ | MSE×10³↓ | LPIPS×10³↓ | CLIP Score↑ |
|--------|--------|------------|-------|----------|-----------|------------|
| NTN-Diff* | **12.69** | **27.82** | **40.70** | **0.11** | **0.88** | **26.49** |
| BrushNet* | 12.64 | 27.78 | 31.94 | 0.80 | 18.67 | 26.39 |
| NTN-Diff | 12.45 | 27.57 | 23.51 | 6.50 | 40.79 | 26.54 |
| BrushNet | 12.36 | 27.40 | 21.65 | 9.31 | 48.28 | 26.48 |
| BLD | 9.78 | 25.87 | 21.33 | 9.76 | 49.26 | 26.15 |

### Ablation Study (BrushBench)

| Configuration | IR×10↑ | PSNR↑ | LPIPS×10³↓ | CLIP Score↑ |
|---------------|--------|-------|-----------|------------|
| Full NTN-Diff | **11.12** | **28.10** | **44.09** | **26.09** |
| Case A (w/o masked self-attention) | 10.14 | 28.02 | 44.54 | 25.95 |
| Case B (w/o text-guided stream) | 9.59 | 27.71 | 47.08 | 25.78 |
| Case C (w/o mid-frequency-guided stream) | 10.02 | 28.06 | 44.92 | 26.03 |

### Key Findings

- NTN-Diff* (with pixel-level blending) outperforms BrushNet* by 8.76 dB in PSNR, with 86% lower MSE and 95% lower LPIPS, while also achieving higher IR and CLIP Score.
- All three denoising stages are essential: Case B (without the text-guided stream) reduces IR by 44.8%, demonstrating that mid-frequency guidance requires a text-aligned mid-frequency source.
- $\lambda = 0.6$ is the optimal early/late stage boundary: an overly short early stage ($\lambda=0.9$) fails to sufficiently stabilize low-frequency components, while an overly long early stage ($\lambda=0.5$) leaves insufficient room for late-stage text-guided refinement.
- Adaptive frequency band extraction is critical: fixed thresholds lead to semantic inconsistencies (e.g., smiling cat artifacts) or color block artifacts.

## Highlights & Insights

- Frequency-domain analysis offers a novel perspective on diffusion-based inpainting; the differential response of frequency bands to text conditioning during denoising is the central finding.
- The "null-text–text–null-text" three-stage design elegantly exploits the robustness of mid-frequency components as a bridge between low-frequency preservation and text alignment.
- The plug-and-play frequency band substitution layers (DCT-domain operations) require no additional training, making the approach concise and efficient.
- The adaptive threshold design accounts for the influence of mask proportion, improving generalization across diverse masking patterns.

## Limitations & Future Work

- Three parallel denoising streams are required, making inference approximately three times more computationally expensive than standard diffusion models.
- The method is built upon Stable Diffusion v1.5 and has not been validated on more recent architectures such as SD-XL or SD3.
- The frequency band separation assumption may not hold under certain extreme conditions (e.g., images dominated by low-frequency content).
- The method involves multiple hyperparameters ($\lambda$ and frequency thresholds); while heuristic initialization is provided, tuning remains necessary.
- Validation is limited to inpainting scenarios; extension to other editing tasks such as outpainting warrants further investigation.

## Related Work & Insights

- The per-step blending strategy of BLD is simple yet effective; NTN-Diff extends this idea by incorporating a frequency-aware dimension.
- Frequency-domain diffusion models (FreeDiff, FBSDiff, etc.) inspire the frequency separation approach adopted here.
- The comparison with BrushNet highlights the complementarity between purely network-based approaches and frequency-aware methods.
- The two-stage "stabilize first, then refine" denoising strategy provides inspiration for other diffusion-based image editing tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The frequency-aware perspective is original, and the null-text–text–null-text three-stage design is distinctive with well-motivated justification.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluations span BrushBench and EditBench; ablation studies are comprehensive and hyperparameter analysis is detailed.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated and frequency analysis visualizations are intuitive, though the method description is somewhat verbose.
- **Value**: ⭐⭐⭐⭐ Simultaneously addressing the two major challenges in inpainting has practical significance, and the frequency-domain perspective opens new directions for subsequent work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] NPN: Non-Linear Projections of the Null-Space for Imaging Inverse Problems](npn_non-linear_projections_of_the_null-space_for_imaging_inverse_problems.md)
- [\[NeurIPS 2025\] Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] Test-Time Alignment of Text-to-Image Diffusion Models via Null-Text Embedding Optimisation](../../CVPR2026/image_generation/test-time_alignment_of_text-to-image_diffusion_models_via_null-text_embedding_op.md)
- [\[NeurIPS 2025\] DEFT: Decompositional Efficient Fine-Tuning for Text-to-Image Models](deft_decompositional_efficient_finetuning_for_texttoimage_mo.md)
- [\[NeurIPS 2025\] Diffusion Adaptive Text Embedding for Text-to-Image Diffusion Models](diffusion_adaptive_text_embedding_for_texttoimage_diffusion.md)

</div>

<!-- RELATED:END -->
