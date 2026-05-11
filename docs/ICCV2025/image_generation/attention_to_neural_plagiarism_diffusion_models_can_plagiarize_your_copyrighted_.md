---
title: >-
  [Paper Note] Attention to Neural Plagiarism: Diffusion Models Can Plagiarize Your Copyrighted Images!
description: >-
  [ICCV 2025][Image Generation][Neural Plagiarism] This paper exposes the threat of "neural plagiarism"—diffusion models can readily replicate copyright-protected images (including watermarked ones). It proposes a universa…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Neural Plagiarism"
  - "Diffusion Models"
  - "Copyright Protection"
  - "Watermark Removal"
  - "Attention Perturbation"
date: 2026-05-08
content_hash: 6849e398136ced67
---

# Attention to Neural Plagiarism: Diffusion Models Can Plagiarize Your Copyrighted Images!

**Conference**: ICCV 2025  
**arXiv**: [2603.00150](https://arxiv.org/abs/2603.00150)  
**Code**: [https://github.com/zzzucf/Neural-Plagiarism](https://github.com/zzzucf/Neural-Plagiarism)  
**Area**: Image Generation  
**Keywords**: Neural Plagiarism, Diffusion Models, Copyright Protection, Watermark Removal, Attention Perturbation

## TL;DR
This paper exposes the threat of "neural plagiarism"—diffusion models can readily replicate copyright-protected images (including watermarked ones). It proposes a universal attack framework based on "anchors and shims," searching for perturbations in the cross-attention mechanism to achieve coarse-to-fine semantic modification, bypassing copyright protections ranging from visible trademarks to invisible watermarks.

## Background & Motivation
The growing generative capability of diffusion models has raised serious concerns about data copyright infringement. This threat, termed "neural plagiarism," manifests in two forms:

**Forgery Attack**: Generating visually similar copies while removing watermarks, causing copyright verification to fail: $\mathcal{V}(x^*) \neq w$

**Ambiguity Attack**: Replacing watermarks with new ones to fabricate ownership disputes: $\mathcal{V}(x^*) = w^*$

Existing protective measures include legal frameworks (GDPR, copyright law) and technical methods (visible/invisible watermarks). However, existing watermark removal methods (e.g., Regen, Rinse) rely on simple noise-then-denoise processes, yielding limited effectiveness and often introducing visible artifacts.

Directly optimizing the objective $\min_{x_T^*} d_{visual}(x_0^w, x_0^*) - \gamma d_{latent}(x_T, x_T^*)$ faces three challenges:

**Memory Explosion**: Computing gradients $\frac{\partial x_{t-1}^*}{\partial x_t^*}$ at each timestep requires >100 GB VRAM for 10 steps

**Over-smoothing**: Skip-step gradient estimation causes image blurring

**Noisy Output**: High perturbations produce noise rather than meaningful semantic changes

**Core Idea**: Anchors (inverted latent sequences) maintain the trajectory, while shims (learned perturbations) fine-tune semantics at specific timesteps—analogous to using shims when installing a door to adjust local spacing while keeping the frame aligned.

## Method

### Overall Architecture
The attack pipeline consists of two stages:
1. **Anchor Acquisition**: The copyrighted image is encoded via VAE into $\hat{x}_0$; the DPM solver inverts the generative process to obtain the anchor sequence $\{\hat{x}_1, ..., \hat{x}_T\}$
2. **Shim Search**: Starting from a selected timestep $K$, reverse sampling is performed, optimizing shims $\delta_t$ (injected into text embeddings of cross-attention) at selected timesteps $\mathcal{T}_{select}$, so that the generated image is visually similar to the original while its latent deviates from the anchors

### Key Designs

1. **Anchors and Shims Optimization**:

    - **Function**: Decouples the long-chain dependencies of latents, enabling independent per-timestep perturbation optimization
    - **Mechanism**: Saves the complete inverted latent trajectory as anchors $\{\hat{x}_t\}$; defines shims $\delta_t$ as search variables: $x_t^* = \Delta_{\delta_t \in \mathcal{S}}(x_t, \delta_t)$
    - **Norm Constraint**: $\mathcal{L}_{norm}(t) = \max(0, \hat{\varepsilon}_t - \|\delta_t\|)$, ensuring shims are large enough to deviate from anchors (thereby breaking watermarks)
    - **Design Motivation**: Decouples timestep chains by saving anchors; each step requires only single-step backpropagation memory, resolving the >100 GB VRAM issue

2. **Semantic Search via Attention Perturbation**:

    - **Function**: Enables controllable semantic modification search in the semantic space
    - **Mechanism**: Rather than directly perturbing latents, shims $\delta_t$ are injected into the text embeddings within cross-attention—applied to the null-string embedding $\mathbf{e}_\emptyset$
    - **Semantic Preservation Loss**: $\mathcal{L}_{semantic}(t) = -\frac{\mathbf{e}_\emptyset \cdot (\mathbf{e}_\emptyset + \delta_t)}{\|\mathbf{e}_\emptyset\| \|\mathbf{e}_\emptyset + \delta_t\|}$ (maximizing cosine similarity)
    - **Alignment Loss**: $\mathcal{L}_{align}(t) = d(x_{t-1}, \hat{x}_{t-1})$, ensuring perturbed outputs remain close to anchors
    - **Key Insight**: Shims at different timesteps control semantic changes at different granularities—large timesteps alter global semantics (e.g., color, shape), while small timesteps modify fine-grained texture and details
    - **Design Motivation**: Cross-attention is the core semantic control mechanism in diffusion models; precise semantic manipulation is achieved by perturbing the inputs to $K$ and $V$

3. **Iterative Search Process**:

    - **Function**: Jointly optimizes shims over selected timesteps
    - **Joint Objective**: $\min_{\delta_t} \mathcal{L}_{norm}(t) + \gamma_1 \mathcal{L}_{semantic}(t) + \gamma_2 \mathcal{L}_{align}(t)$
    - **Hyperparameters**: $\gamma_1 = 10^5, \gamma_2 = 0.1, \hat{\varepsilon}_t = 10$
    - Adam optimizer (learning rate 0.01), weight decay $10^{-3}$, gradient clipping (max norm 1.0)
    - DPM solver with 50 timesteps
    - Two initialization modes:
        - **Late-start + Noise Initialization** ($K=140$, shims at steps 100 and 60): Small perturbations, suitable for invisible watermark removal
        - **Early-start + Inversion Initialization** ($K=1000$, shims at steps 600 and 200): Large perturbations for substantial semantic modification

### Attack Algorithm (Algorithm 1)
```
Input: Copyrighted image x^w
1. VAE encoding → x̂_0
2. DPM Solver inversion → anchor sequence {x̂_1,...,x̂_T}
3. Initialize x*_K = x̂_K or noisy latent
4. For t = K to 1:
   If t ∈ T_select:  # Timesteps requiring perturbation
     Initialize δ_t = 0
     While not converged:
       x*_{t-1} = x*_t - ζ_t·ε_θ(x*_t, t, e_∅ + δ_t)
       δ_t -= η·∇L(t, δ_t, x*_{t-1}, x̂_{t-1})
   Else:  # Normal denoising
     x*_{t-1} = x*_t - ζ_t·ε_θ(x*_t, t, e_∅)
5. Output x* = D(x*_0)
```

## Key Experimental Results

### Main Results (Invisible Watermark Removal, MS-COCO)

| Method | DwtDctSvd BA↓ | DwtDctSvd ACC↓ | RivaGAN BA↓ | RivaGAN ACC↓ | PSNR↑ | SSIM↑ |
|--------|--------------|----------------|-------------|-------------|-------|-------|
| Regen | 0.64 | 0.15 | 0.60 | 0.05 | 26.21 | 0.75 |
| Rinse | 0.54 | 0.04 | 0.52 | 0.01 | 23.68 | 0.68 |
| **Ours (Late)** | **0.52** | **0.01** | 0.56 | 0.02 | 25.27 | 0.73 |
| **Ours (Early)** | **0.52** | **0.03** | **0.52** | **0.00** | 21.16 | 0.70 |

| Method | Stable Signature T@1%F↓ | Tree-Ring T@1%F↓ |
|--------|------------------------|------------------|
| Regen | 0.00 | **1.00** |
| **Ours** | **0.00** | 0.98 |

### Ablation Study (Degree of Semantic Modification vs. Timestep Selection)

| Initialization Mode | Shim Timesteps | Effect | Applicable Scenario |
|--------------------|----------------|--------|---------------------|
| Late-start (K=140) + Noise | 100, 60 | Minor semantic change, preserves image quality | Invisible watermark removal |
| Early-start (K=1000) + Inversion | 600, 200 | Major semantic change (hairstyle, clothing, etc.) | Visible copyright bypass |
| — | Large timestep (single step) | Alters global semantics: color, shape | IP character modification |
| — | Small timestep (single step) | Alters local semantics: texture, details | Fine-grained forgery |

### Key Findings
- Bit accuracy drops to near 50% (random chance), indicating effective watermark removal
- Compared to Regen/Rinse, the proposed method removes watermarks more thoroughly while better preserving image quality
- Effective against Stable Signature (T@1%F reduced to 0), but **fails to remove Tree-Ring watermarks**—because Tree-Ring embeds local patterns in Fourier space, which are immune to global latent shifts
- Ambiguity attack succeeds: new watermarks can be embedded into already-attacked images, with both watermarks simultaneously detectable (Tree-Ring T@1%F: 1.00/0.99 for both parties)
- Among 100 generated Elon Musk variants, GPT arbitration could not definitively identify any as the real person
- Visible copyright attack: successfully modifies Disney Elsa's iconic hairstyle and clothing, Monet's painting style, and Van Gogh's signature

## Highlights & Insights
- The "anchors and shims" analogy is intuitive, and the framework design is elegant—simultaneously resolving memory issues and providing multi-granularity semantic control
- Attention perturbation is a low-cost and precise semantic manipulation approach that modifies only text embeddings rather than the entire network
- The method is purely gradient-based search requiring no additional training or fine-tuning, and is applicable to any pretrained diffusion model
- The work clearly exposes the fragility of existing copyright protections, providing an important attack baseline for defensive research
- Timestep selection enables coarse-to-fine semantic controllability

## Limitations & Future Work
- Cannot remove Tree-Ring watermarks (local Fourier-domain patterns are immune to global latent shifts)
- The early-start mode yields higher FID; excessive semantic changes degrade image quality
- Enhancements such as ControlNet, additional image encoders, and negative prompts are not incorporated (explicitly noted as future work in the paper)
- From a defensive perspective, developing watermarks robust to attention-based perturbation search is an urgent research direction
- Evaluation is limited to Stable Diffusion; effectiveness on other diffusion architectures (e.g., DiT) remains unknown

## Related Work & Insights
- **vs. Regen**: Regen's simple noise-then-denoise strategy is insufficient for thorough watermark removal and introduces visible noise; the proposed method achieves precise semantic modification through shim optimization
- **vs. Rinse**: Rinse iterates the Regen process multiple times, improving watermark removal at the cost of severe image quality degradation (FID 82–87)
- **vs. Tree-Ring Watermark**: Tree-Ring's strategy of embedding watermarks in Fourier space proves naturally robust to latent perturbations—a valuable insight for defensive methods
- **vs. Stable Signature**: Despite being a state-of-the-art watermarking method, Stable Signature embeds watermarks in the VAE decoder, making any latent deviation sufficient to bypass it

## Rating
- **Novelty**: ⭐⭐⭐⭐ The "anchors and shims" framework is elegantly designed; the attention perturbation search perspective is novel
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multiple watermarking methods, visible/invisible copyright, forgery/ambiguity attacks, and real-world copyrighted images
- **Writing Quality**: ⭐⭐⭐⭐ Problem formalization is clear and attack type definitions are rigorous, though notation and terminology are occasionally dense
- **Value**: ⭐⭐⭐⭐⭐ Exposes an urgent AI security threat, serving as both a warning to the copyright protection community and a baseline for defensive research

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Your Text Encoder Can Be An Object-Level Watermarking Controller](your_text_encoder_can_be_an_object-level_watermarking_controller.md)
- [\[ICCV 2025\] ReFlex: Text-Guided Editing of Real Images in Rectified Flow via Mid-Step Feature Extraction and Attention Adaptation](reflex_text-guided_editing_of_real_images_in_rectified_flow_via_mid-step_feature.md)
- [\[CVPR 2026\] Attention, May I Have Your Decision? Localizing Generative Choices in Diffusion Models](../../CVPR2026/image_generation/attention_may_i_have_your_decision_localizing_generative_choices_in_diffusion_mo.md)
- [\[NeurIPS 2025\] Neural Entropy](../../NeurIPS2025/image_generation/neural_entropy.md)
- [\[ICCV 2025\] EDiT: Efficient Diffusion Transformers with Linear Compressed Attention](edit_efficient_diffusion_transformers_with_linear_compressed_attention.md)

</div>

<!-- RELATED:END -->
