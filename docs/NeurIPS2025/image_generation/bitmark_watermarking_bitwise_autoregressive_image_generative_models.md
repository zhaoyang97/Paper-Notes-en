---
title: >-
  [Paper Note] BitMark: Watermarking Bitwise Autoregressive Image Generative Models
description: >-
  [NeurIPS 2025][Image Generation][Bit-level watermarking] This paper proposes BitMark—the first watermarking scheme for bitwise autoregressive image generative models (Infinity, Instella). During generation…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Bit-level watermarking"
  - "autoregressive image generation"
  - "model collapse prevention"
  - "radioactive watermarking"
  - "Infinity"
date: 2026-05-08
content_hash: c1eab051824b9221
---

# BitMark: Watermarking Bitwise Autoregressive Image Generative Models

**Conference**: NeurIPS 2025
**arXiv**: [2506.21209](https://arxiv.org/abs/2506.21209)
**Code**: [https://github.com/sprintml/BitMark](https://github.com/sprintml/BitMark)
**Area**: Image Generation / Watermarking / AI Safety
**Keywords**: Bit-level watermarking, autoregressive image generation, model collapse prevention, radioactive watermarking, Infinity

## TL;DR
This paper proposes BitMark—the first watermarking scheme for bitwise autoregressive image generative models (Infinity, Instella). During generation, it steers bit sequences toward a "green list" by adding logit biases, enabling reliable detection (z-test), high image fidelity (negligible FID change), robustness against diverse attacks, and radioactivity (downstream models trained on watermarked images also carry the watermark), providing a critical tool for preventing model collapse.

## Background & Motivation

**Background**: State-of-the-art text-to-image models such as Infinity and Instella generate high-quality images via bitwise autoregressive prediction (rather than conventional token-level prediction), with codebook sizes reaching $2^{64}$. As AI-generated images proliferate online, models are increasingly trained on their own outputs, leading to model collapse.

**Limitations of Prior Work**:
   - Diffusion model watermarking schemes (Tree-Ring, PRC, Stable Signature) are not applicable to autoregressive architectures.
   - LLM watermarking (KGW) operates at the token level, but tokens in image AR models are inconsistent after encode–decode cycles (token overlap rate is only ~2.4%).
   - Existing diffusion model watermarks are not radioactive—new models trained on watermarked images do not inherit the watermark.

**Key Challenge**: Tokens in image autoregressive models shift substantially through encode–decode cycles due to quantization loss in the continuous image space, causing token-level watermark signals to largely vanish. However, bit-level overlap is far higher (~77.4%), offering a viable entry point for watermark embedding.

**Goal**: Design a watermarking scheme operating at the bit level that satisfies: (a) no degradation of image quality, (b) reliable detection, (c) resistance to diverse removal attacks, and (d) radioactivity.

**Key Insight**: The observation that bit-level overlap in Infinity (77.4%) is far higher than token-level overlap (2.4%), making bit-level watermark embedding substantially more robust.

**Core Idea**: Adapt the green/red list watermarking concept from LLMs from the token level to the bit level, exploiting the bitwise autoregressive generation process of image models to embed detectable signals.

## Method

### Overall Architecture
**Embedding**: Partition all bit sequences of length $n$ into a green list $G$ and a red list $R$ → at each bit prediction step, if the current bit completes a sequence in $G$, add a bias $\delta$ to the corresponding logit → the proportion of green sequences in the generated image significantly exceeds 50%.
**Detection**: Re-encode the suspect image into bits → compute the proportion of green sequences → apply a z-test to determine whether the proportion exceeds the expectation for natural images.

### Key Designs

1. **Bit-Level Green/Red List Watermark Embedding**:

    - **Function**: Embed the watermark signal into the generation process without modifying model weights.
    - **Mechanism**: Given a prefix $pre = (b_{j-(n-1)}, \ldots, b_{j-1})$ and the current bit $b_j$, if $pre + b_j \in G$, then $p_j = \text{softmax}(l_j^{(b_j)} + \delta)$; otherwise no bias is added. The bias $\delta$ is kept small (e.g., 1.0–2.0), primarily affecting high-entropy bits (those already near 50/50), so the impact on image quality is minimal.
    - **Design Motivation**: Flipping high-entropy bits has the least effect on image appearance (since the model itself is uncertain), and these bits are precisely the easiest to steer with a bias—a perfect match.

2. **Z-Test Statistical Detection**:

    - **Function**: Reliably determine whether an image contains a watermark.
    - **Mechanism**: Count the number of green sequences $C$ in the encoded image and compute $z = (C - \gamma T) / \sqrt{T\gamma(1-\gamma)}$; if $z$ exceeds a threshold, the image is classified as watermarked. In natural images $C \approx \gamma T$ (all sequences equally likely), whereas in watermarked images $C$ is significantly elevated.
    - **Design Motivation**: The z-test provides precise false positive rate control, allowing the detection threshold to be tuned between sensitivity and false alarm rate.

3. **Radioactivity**:

    - **Function**: Ensure that downstream models trained on watermarked images also produce watermarked outputs.
    - **Mechanism**: Watermarking shifts the statistical distribution of images (the proportion of green bit sequences is elevated); downstream models trained on such biased data learn this statistical preference and reproduce it in their outputs.
    - **Design Motivation**: Prevent third parties from fine-tuning on watermarked images to produce "laundered" outputs—the watermark propagates to downstream models.

### Loss & Training
- No training required—watermark is embedded purely at inference time.
- Key hyperparameters: bit sequence length $n$, bias strength $\delta$, green list proportion $\gamma$.
- Secret key $\mathcal{K}$ controls the green/red list assignment, ensuring watermark confidentiality.

## Key Experimental Results

### Main Results
Watermarking performance on Infinity and Instella:

| Metric | No Watermark | BitMark ($\delta$=1.5) | Notes |
|--------|-------------|----------------------|-------|
| FID | Baseline | Near baseline (<1 difference) | Negligible image quality impact |
| Green sequence ratio | ~50% | ~65–70% | Significant watermark signal |
| Detection AUC | — | >0.99 | Near-perfect detection |
| Inference speed | Baseline | Baseline | Zero additional overhead (bias addition only) |

### Ablation Study: Robustness Comparison

| Attack Type | Detection AUC | Notes |
|------------|--------------|-------|
| No attack | >0.99 | Perfect detection |
| JPEG (quality=50) | >0.95 | Highly robust |
| Gaussian noise | >0.95 | Highly robust |
| Gaussian blur | >0.93 | Robust |
| Color jitter | >0.95 | Highly robust |
| Random crop | >0.90 | Reasonably robust |
| Watermark-in-the-Sand | >0.88 | Robust against dedicated removal |
| CtrlRegen | >0.85 | Effective against regeneration attacks |
| Bit-Flipper (custom attack) | >0.80 | Targeted attacks still fail to fully remove watermark |

### Key Findings
- **Bit-level vs. token-level**: Token overlap is only 2.4%, while bit overlap is 77.4%—bit-level operation is the only viable watermarking strategy.
- **High-entropy bits are critical**: The bias predominantly affects high-entropy bits (which minimally influence image value distributions), preserving image quality.
- **Radioactivity validated experimentally**: After fine-tuning a diffusion model on watermarked Infinity images, the diffusion model's outputs are also detected as watermarked.
- **Private watermark protection**: Attackers cannot infer the green/red list without knowledge of the secret key.
- **Comparison with LLM watermarking**: Directly applying KGW at the token level achieves detection rates below 60%, while BitMark's bit-level approach achieves >99%.

## Highlights & Insights
- **Bit-level operation is the core innovation**: The observation that bit-level encode–decode consistency is far higher than token-level consistency is both simple and profound, directly determining the design direction.
- **Radioactive watermarking** has strategic significance for preventing model collapse: it not only protects the original model but also makes derivative models throughout the ecosystem traceable.
- **Zero inference overhead**: Only a scalar bias is added to logits, requiring no additional neural network inference or post-processing.
- The technique generalizes to any future model employing bitwise autoregressive generation.

## Limitations & Future Work
- Validation is limited to Infinity and Instella; other autoregressive architectures (e.g., VAR's token-level prediction) require different approaches.
- Detection requires access to the model's encoder to re-encode images into bits, limiting independent third-party detection.
- Optimal selection of the green list proportion $\gamma$ and bias $\delta$ may be model- and resolution-dependent.
- Insufficient testing on ultra-high-resolution (>2K) images.
- Rigorous analysis of the theoretically optimal bit sequence length $n$ is lacking.

## Related Work & Insights
- **vs. KGW (Kirchenbauer et al.)**: The green/red list watermarking concept for LLMs. BitMark transfers this from the token level to the bit level to accommodate the encode–decode inconsistency of image AR models.
- **vs. Tree-Ring / PRC**: Diffusion model watermarks that embed signals in the noise space. Not applicable to autoregressive models and lack radioactivity.
- **vs. Stable Signature**: Embeds watermarks by fine-tuning the decoder, requiring model modification. BitMark requires zero model modification.
- The radioactive watermarking concept is transferable to other generative modalities (audio and video autoregressive models).

## Rating
- Novelty: ⭐⭐⭐⭐ First watermarking scheme for bitwise AR image models; the bit-level observation is a key insight.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two models, diverse attacks (including dedicated removal), radioactivity validation, and complete ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and rigorous formalization of the method.
- Value: ⭐⭐⭐⭐⭐ Direct practical value for model collapse prevention and generated content traceability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Watermarking Autoregressive Image Generation](watermarking_autoregressive_image_generation.md)
- [\[NeurIPS 2025\] EditInfinity: Image Editing with Binary-Quantized Generative Models](editinfinity_image_editing_with_binary-quantized_generative_models.md)
- [\[NeurIPS 2025\] Conditional Panoramic Image Generation via Masked Autoregressive Modeling](conditional_panoramic_image_generation_via_masked_autoregres.md)
- [\[NeurIPS 2025\] Shallow Diffuse: Robust and Invisible Watermarking through Low-Dimensional Subspaces in Diffusion Models](shallow_diffuse_robust_and_invisible_watermarking_through_low-dimensional_subspa.md)
- [\[NeurIPS 2025\] ThermalGen: Style-Disentangled Flow-Based Generative Models for RGB-to-Thermal Image Translation](thermalgen_style-disentangled_flow-based_generative_models_for_rgb-to-thermal_im.md)

</div>

<!-- RELATED:END -->
