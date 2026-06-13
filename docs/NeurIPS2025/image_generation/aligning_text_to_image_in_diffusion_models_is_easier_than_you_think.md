---
title: >-
  [Paper Note] Aligning Text to Image in Diffusion Models is Easier Than You Think
description: >-
  [NeurIPS 2025][Image Generation][SoftREPA] This paper proposes SoftREPA — a lightweight contrastive fine-tuning strategy that introduces learnable soft text tokens (fewer than 1M parameters) to perform contrastive learni…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "SoftREPA"
  - "Contrastive Learning"
  - "Soft Token"
  - "Text-Image Alignment"
  - "Mutual Information"
date: 2026-05-08
content_hash: af958034bbb68ac8
---

# Aligning Text to Image in Diffusion Models is Easier Than You Think

**Conference**: NeurIPS 2025
**arXiv**: [2503.08250](https://arxiv.org/abs/2503.08250)  
**Code**: [https://softrepa.github.io/](https://softrepa.github.io/) (project page)  
**Area**: Diffusion Models / Text-Image Alignment
**Keywords**: SoftREPA, Contrastive Learning, Soft Token, Text-Image Alignment, Mutual Information

## TL;DR
This paper proposes SoftREPA — a lightweight contrastive fine-tuning strategy that introduces learnable soft text tokens (fewer than 1M parameters) to perform contrastive learning on frozen pretrained T2I diffusion models, explicitly maximizing mutual information between text and image representations. SoftREPA significantly improves text-image alignment on SD1.5/SDXL/SD3 and generalizes to both image generation and image editing tasks.

## Background & Motivation

**Background**: Text-to-image diffusion models (SD, FLUX, etc.) incorporate text conditioning into image generation via cross-attention or self-attention, yet residual misalignment between text and image representations persists — generated images may omit key attributes, colors, or quantities described in the text prompt.

**Limitations of Prior Work**: Existing approaches such as preference optimization (DPO) require curated human preference datasets and incur substantial training overhead. Representation alignment methods such as REPA only align internal image representations with external visual encoders, without directly improving text-image alignment.

**Key Challenge**: Standard T2I training uses only positive pairs (matched image-text pairs) to minimize the denoising loss, which is suboptimal from a representation alignment perspective — it lacks contrastive signals from negative pairs to distinguish between different text conditions.

**Goal**: How can one improve text-image alignment in a pretrained T2I model with minimal additional parameters and computational overhead?

**Key Insight**: The denoising loss is reinterpreted as a logit (conditional likelihood), upon which an InfoNCE-style contrastive learning loss is constructed. This is combined with the soft token concept from prompt tuning to enable lightweight fine-tuning.

**Core Idea**: Treat the denoising loss as a logit for contrastive learning + train only soft text tokens = substantially improved text-image alignment with fewer than 1M parameters.

## Method

### Overall Architecture
The pretrained T2I model parameters are frozen. Learnable soft tokens $\mathbf{s}^{(k,t)} \in \mathbb{R}^{m \times d}$ (indexed by layer and timestep) are prepended to the text representations at each layer. During training, these soft tokens are optimized via the contrastive loss; at inference, the trained soft tokens are simply concatenated to the text features.

### Key Designs

1. **Contrastive T2I Alignment Loss**:

    - **Function**: Constructs an InfoNCE-style loss using positive and negative text-image pairs within the same batch.
    - **Mechanism**: The negative denoising loss $\|\epsilon_\theta(\mathbf{x}_t, t, \mathbf{y}) - \epsilon\|^2$ is mapped via an exponential to produce a logit $\tilde{l}(\mathbf{x}, \mathbf{y}, \mathbf{s}) = e^{-\|v_\theta(\mathbf{x}_t, t, \mathbf{y}, \mathbf{s}) - (\epsilon - \mathbf{x}_0)\|^2 / \tau(t)}$, and the contrastive loss is then defined as $\mathcal{L} = -\log \frac{\exp(\tilde{l}(\mathbf{x}, \mathbf{y}, \mathbf{s}))}{\sum_j \exp(\tilde{l}(\mathbf{x}, \mathbf{y}^{(j)}, \mathbf{s}))}$.
    - **Design Motivation**: Standard training relies solely on positive pairs and cannot distinguish between different text conditions. The contrastive loss repels mismatched text-image pairs via negative samples, sharpening the conditional probability distribution. The exponential mapping ensures bounded logits, avoiding training instability.

2. **Learnable Soft Tokens**:

    - **Function**: Learnable tokens are prepended to the text representations at each layer and each timestep.
    - **Mechanism**: $\hat{\mathbf{H}}_{\text{text}}^{(k-1,t)} = [\mathbf{s}^{(k,t)}; \mathbf{H}_{\text{text}}^{(k-1,t)}]$, where soft tokens are generated via an Embedding$(k,t)$ and processed jointly with the attention layers. Only these tokens (<1M parameters) are trained; the rest of the model remains frozen.
    - **Design Motivation**: Analogous to prompt tuning — rather than modifying model weights, learnable signals are injected into the input space to steer model behavior. The extremely low parameter count enables fast training and negligible inference overhead.

3. **Mutual Information Theoretical Analysis**:

    - **Function**: Proves that minimizing the contrastive loss is equivalent to maximizing the mutual information between text and image representations.
    - **Mechanism**: Drawing on results from Song & Kong et al., the conditional likelihood of a diffusion model satisfies $p_\theta(\mathbf{x}|\mathbf{y}) = \exp(\hat{l}(\mathbf{x}, \mathbf{y}))$. Consequently, the contrastive logit approximates the PMI $i(\mathbf{x}, \mathbf{y}) = \log \frac{p_\theta(\mathbf{x}|\mathbf{y})}{p_\theta(\mathbf{x})}$, and the mutual information $I(X,Y) = \mathbb{E}[i(X,Y)]$. Minimizing the contrastive loss thus maximizes mutual information.
    - **Design Motivation**: Provides a theoretical guarantee explaining why this simple approach effectively improves semantic consistency.

### Loss & Training
- Training objective: $\mathcal{L}_{\text{SoftREPA}}(\mathbf{s})$, a contrastive loss optimizing only the soft token parameters.
- The expectation is approximated by a single Monte Carlo sample (with $\epsilon$ and $t$ shared across the batch).
- A time-dependent temperature schedule $\tau(t)$ controls sharpness.
- Compatible with both UNet-based architectures (SD1.5, SDXL) and DiT-based architectures (SD3).

## Key Experimental Results

### Main Results (COCO val5K + GenEval)

| Model | ImageReward ↑ | CLIP ↑ | HPS ↑ | FID ↓ | Extra Params |
|-------|--------------|--------|-------|-------|-------------|
| SD1.5 | 17.72 | 26.40 | 25.08 | 24.59 | 0 |
| SD1.5 + SoftREPA | **32.89** | **27.33** | **25.18** | **23.43** | <1M |
| SDXL | 75.06 | 26.76 | 27.35 | 24.69 | 0 |
| SDXL + SoftREPA | **85.29** | **26.80** | **28.30** | **26.04** | <1M |
| SD3 | 94.27 | 26.30 | 28.09 | 31.59 | 0 |
| SD3 + SoftREPA | **108.5** | **26.91** | **28.91** | **36.21** | <1M |

### Ablation Study (GenEval on SD3)

| Configuration | Mean ↑ | Notes |
|--------------|--------|-------|
| SD3 + SoftREPA | 0.70 | Full method |
| SD3 baseline | 0.68 | No contrastive learning |
| CaPO (preference optimization) | 0.71 | Requires preference data |
| RankDPO | 0.74 | Requires preference data; full model fine-tuning |

### Image Editing Experiments
SoftREPA consistently improves editing quality across PnP, MasaCtrl, and FlowEdit:
- FlowEdit + SoftREPA: ImageReward 87.70 → **102.24**, CLIP-Edited 23.19 → **23.60**

### Key Findings
- **Most significant gains on SD1.5**: ImageReward nearly doubles (17.72 → 32.89), indicating greater alignment improvement for smaller models.
- **FID slightly increases on SD3** (31.59 → 36.21), suggesting a potential trade-off between stronger text alignment and image quality.
- **Strong generalizability**: Effective across both UNet and DiT architectures, and across both generation and editing tasks.
- **Negligible inference latency overhead**: SD1.5 increases from 1.526 → 1.547 sec/img with no change in GPU memory.
- **Negative samples in contrastive learning are critical**: Standard denoising training with positive pairs alone cannot achieve comparable alignment quality.

## Highlights & Insights
- **The reinterpretation of denoising loss as a contrastive logit is particularly elegant**: This insight bridges diffusion model training and contrastive representation learning. Any diffusion model can incorporate negative sample signals in this way without architectural modification.
- **Soft tokens as prompt tuning for diffusion models**: The approach mirrors prompt tuning in NLP, applied to the text conditioning channel of generative models. The extremely low parameter count (<1M) makes it feasible to fine-tune the alignment quality of large T2I models on consumer-grade GPUs.
- **Plug-and-play compatibility with existing methods**: Whether applied to PnP editing or FlowEdit, SoftREPA integrates seamlessly, making its generality highly practical.

## Limitations & Future Work
- **FID degradation on SD3**: Stronger text alignment may compromise diversity or unconditional image quality; the trade-off warrants further analysis.
- **Performance drop on the Counting task in GenEval** (0.56 → 0.29 on SD3), suggesting contrastive learning may be less effective for semantics requiring precise enumeration.
- **Impact of soft token count and layer placement**: The paper reports using upper layers, but detailed ablations on this design choice are insufficient.
- **Validation limited to the SD model family**: Generalization to other architectures such as FLUX and DALL-E 3 remains unexplored.
- **Theoretical analysis assumes an optimal denoiser**: Real-world models are suboptimal, and the strength of the mutual information maximization guarantee depends on model quality.

## Related Work & Insights
- **vs. REPA [Yu et al.]**: REPA aligns internal image representations with external visual encoders; SoftREPA aligns text-image representations. The two are complementary and can be combined.
- **vs. CaPO/RankDPO (preference optimization)**: Preference optimization requires human-annotated data and full model fine-tuning; SoftREPA requires only raw image-text pairs and fewer than 1M parameters.
- **vs. Classifier-Free Guidance**: CFG enhances alignment at inference time by amplifying conditional scores; SoftREPA improves alignment at training time via contrastive learning. The two approaches are orthogonal and can be used jointly.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The reinterpretation of denoising loss as a contrastive logit is novel; the soft token fine-tuning approach is concise and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multiple models (SD1.5/SDXL/SD3) and tasks (generation/editing), though comparisons with a broader set of alignment methods are limited.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical motivation is clear; experimental organization is sound.
- **Value**: ⭐⭐⭐⭐ A practical low-cost approach to improving T2I alignment with direct applicability to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CRAFT: Aligning Diffusion Models with Fine-Tuning Is Easier Than You Think](../../CVPR2026/image_generation/craft_aligning_diffusion_models_with_finetuning_is_easier_than_you_think.md)
- [\[NeurIPS 2025\] Diffusion Adaptive Text Embedding for Text-to-Image Diffusion Models](diffusion_adaptive_text_embedding_for_texttoimage_diffusion.md)
- [\[NeurIPS 2025\] Understand Before You Generate: Self-Guided Training for Autoregressive Image Generation](understand_before_you_generate_self-guided_training_for_autoregressive_image_gen.md)
- [\[NeurIPS 2025\] More Than Generation: Unifying Generation and Depth Estimation via Text-to-Image Diffusion Models](more_than_generation_unifying_generation_and_depth_estimation_via_text-to-image_.md)
- [\[NeurIPS 2025\] Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)

</div>

<!-- RELATED:END -->
