---
title: >-
  [Paper Note] EvoTok: A Unified Image Tokenizer via Residual Latent Evolution for Visual Understanding and Generation
description: >-
  [CVPR 2025][Image Generation][Unified Image Tokenizer] EvoTok proposes a unified image tokenizer based on Residual Latent Evolution. By cascading residual vector quantization in a shared latent space, it allows representations to progressively evolve from shallow pixel-level details to deep semantic-level abstractions. Trained on only 13M images, it achieves a reconstruction quality of 0.43 rFID and delivers outstanding performance across 7/9 understanding benchmarks, GenEval…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Unified Image Tokenizer"
  - "Residual Quantization"
  - "Visual Understanding and Generation"
  - "VQ-VAE"
  - "Multimodal LLM"
date: 2026-05-08
content_hash: f3efe2cef800f07f
---

# EvoTok: A Unified Image Tokenizer via Residual Latent Evolution for Visual Understanding and Generation

**Conference**: CVPR 2025  
**arXiv**: [2603.12108](https://arxiv.org/abs/2603.12108)  
**Code**: [https://github.com/VisionXLab/EvoTok](https://github.com/VisionXLab/EvoTok)  
**Area**: Image Generation / Multimodal  
**Keywords**: Unified Image Tokenizer, Residual Quantization, Visual Understanding and Generation, VQ-VAE, Multimodal LLM

## TL;DR
EvoTok proposes a unified image tokenizer based on Residual Latent Evolution. By cascading residual vector quantization in a shared latent space, it allows representations to progressively evolve from shallow pixel-level details to deep semantic-level abstractions. Trained on only 13M images, it achieves a reconstruction quality of 0.43 rFID and delivers outstanding performance across 7/9 understanding benchmarks, GenEval, and GenAI-Bench.

## Background & Motivation
**Background**: Unified multimodal LLMs need to support both visual understanding (high-level semantics) and image generation (pixel-level details). The core problem is designing a unified image tokenizer.

**Limitations of Prior Work**: Existing paradigms suffer from distinct limitations:
   - **Entangled-type** (ViLA-U, UniTok): Applying both VQ reconstruction and contrastive learning losses on the same set of features leads to optimization conflicts between semantic alignment and pixel reconstruction.
   - **Decoupled-type** (DualToken, TokLIP, TokenFlow): Modeling semantic and pixel features with independent branches/layers/codebooks. However, excessive independence leads to a lack of intrinsic consistency between the two types of features.

**Key Challenge**: Understanding requires high-level semantic abstraction, while generation requires fine-grained pixel representation—though seemingly contradictory, visual information is inherently a continuous spectrum from pixels to semantics.

**Goal**: How to simultaneously achieve decoupling (reducing task conflict) and consistency (sharing visual structure and semantic priors) in a unified latent space?

**Key Insight**: Representing the image as an evolutionary trajectory of residual quantization—shallow levels capture pixel details while deep levels progressively accumulate into semantic abstractions, co-evolving in the same space.

**Core Idea**: Different depths of residual quantization naturally correspond to the pixel $\to$ semantic evolution spectrum. Decoupling comes from slicing at different depths, while consistency comes from the shared space.

## Method

### Overall Architecture
Input image $I$ $\to$ Shared encoder $\mathcal{E}$ extracts feature $\mathbf{f}$ $\to$ $L$-level residual vector quantization $\mathcal{RQ}$ yields $(\mathbf{k}_1, ..., \mathbf{k}_L)$ $\to$ The partial sum of the first $L_{\text{pix}}$ levels forms the pixel feature $\mathbf{f}_{\text{pix}}$ (fed into the pixel decoder for image reconstruction) $\to$ The cumulative sum of all $L_{\text{sem}}$ levels yields the semantic feature $\mathbf{f}_{\text{sem}}$ (aligned with SigLIP2 semantic features and fed into the LLM for understanding). During generation, the LLM autoregressively predicts $L_{\text{pix}}$-level residual codes.

### Key Designs

1. **Residual Latent Evolution**:

    - **Function**: Codes the image as an evolutionary trajectory in a shared space using the cascaded residual quantization of RQ-VAE.
    - **Mechanism**: $\mathbf{k}_i = \mathcal{Q}(\mathbf{r}_{i-1}; \mathcal{C}_i)$, $\mathbf{r}_i = \mathbf{r}_{i-1} - \mathbf{e}_i(\mathbf{k}_i)$. The pixel feature is $\mathbf{f}_{\text{pix}} = \sum_{i=1}^{L_{\text{pix}}} \mathbf{e}_i(\mathbf{k}_i)$, and the semantic feature is $\mathbf{f}_{\text{sem}} = \sum_{i=1}^{L_{\text{sem}}} \mathbf{e}_i(\mathbf{k}_i)$.
    - **Design Motivation**: In standard RQ-VAE, shallow levels naturally capture the main structure while deep levels capture fine residuals. EvoTok reverses this usage—mapping shallow levels to pixels (first 4 levels) and all 16 levels to semantics, achieving progressive pixel-to-semantic evolution.

2. **Direction Selection of Pixel $\to$ Semantic**:

    - **Function**: Explores the importance of the evolution direction (pixel-to-semantic vs. semantic-to-pixel).
    - **Mechanism**: Ablation studies show that $(L_{\text{pix}}=4, L_{\text{sem}}=16)$ outperforms $(L_{\text{pix}}=16, L_{\text{sem}}=4)$ and the entangled setting $(L_{\text{pix}}=L_{\text{sem}}=4)$.
    - **Design Motivation**: High-level semantics can be progressively accumulated from pixel features, but reverse regression (from semantics back to pixels) yields poor results.

3. **Unified Training Objective**:

    - **Function**: Optimizes both pixel reconstruction and semantic alignment simultaneously.
    - **Mechanism**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pix}} + \mathcal{L}_{\text{sem}} + \mathcal{L}_{\text{VQ}}$, where $\mathcal{L}_{\text{pix}}$ includes reconstruction, perceptual, and adversarial losses, $\mathcal{L}_{\text{sem}}$ represents the cosine similarity with SigLIP2 features, and $\mathcal{L}_{\text{VQ}}$ represents the standard VQ loss.
    - **Design Motivation**: The two types of losses act on slices at different depths of the trajectory, ensuring they are naturally decoupled and do not conflict.

4. **Integration of Understanding and Generation**:

    - **Function**: Handles bidirectional tasks:
      - Understanding: $\mathbf{f}_{\text{sem}}$ is projected into the LLM via a semantic decoder + MLP projector (following the LLaVA paradigm).
      - Generation: After the LLM autoregressively predicts the spatial location, the RQ-Transformer head predicts $L_{\text{pix}}$ codes along the depth, which are summed to obtain the pixel features and decode the image.

### Loss & Training
$$\mathcal{L}_{\text{total}} = \underbrace{(\mathcal{L}_R + \lambda_P \mathcal{L}_P + \lambda_G \mathcal{L}_G)}_{\text{pixel}} + \underbrace{\mathcal{L}_{\text{sem}}}_{\text{semantic}} + \underbrace{\mathcal{L}_{\text{VQ}}}_{\text{codebook}}$$

## Key Experimental Results

### Main Results — Understanding (Unified Discrete Class)

| Method | LLM | SEEDBench | GQA | MMMU | MME |
|------|-----|-----------|-----|------|-----|
| VILA-U | LLaMA-7B | 59.0 | 60.8 | 33.5 | 1401.8 |
| DualToken | LLaMA-2-7B | 71.8 | — | 40.5 | 1502.7 |
| EMU3 | 8B | 68.2 | 60.3 | 37.2 | — |
| **EvoTok** | Qwen2.5-7B | **71.8** | **61.8** | **39.9** | **1895.1** |

### Main Results — Generation

| Method | GenAI-Bench (Basic) ↑ | GenEval (Overall) ↑ | Position ↑ | Color Attri. ↑ |
|------|----------------------|---------------------|-----------|----------------|
| SDXL | 0.83 | 0.55 | 0.15 | 0.23 |
| EMU3 | — | 0.66 | 0.49 | 0.45 |
| **EvoTok** | **0.87** | **0.75** | **0.69** | **0.62** |

Reconstruction quality: rFID 0.43 (ImageNet-1K 256×256), using only 13M training data.

### Ablation Study

| Configuration ($L_{\text{pix}}, L_{\text{sem}}$) | rFID ↓ | SEEDBench ↑ | GenEval ↑ | Description |
|----------------------------------------|--------|-------------|-----------|------|
| (4, 4) Entangled | 0.66 | 62.7 | 0.64 | Entangled, poor in both aspects |
| (16, 4) Sem→Pix | 0.44 | 64.6 | 0.60 | Good reconstruction but poor generation/understanding |
| **(4, 16) Pix→Sem** | **0.55** | **67.1** | **0.67** | **Most balanced** |

### Key Findings
- The pixel-to-semantic direction (Pix $\to$ Sem) is the only configuration that performs well across all three dimensions of understanding, generation, and reconstruction.
- t-SNE visualization clearly displays the continuous evolutionary trajectory from shallow to deep layers: shallow clustering is based on texture/color, whereas deep clustering corresponds to semantic classes.
- CLIPSIMpix plateaus after depth 4, while CLIPSIMsem continues to rise—the functional division of labor between the two undergoes a clear transition at depth 4.
- Highly competitive results are achieved with only 13M training data, demonstrating the high efficiency of the architectural design itself.

## Highlights & Insights
- **Residual Quantization = Natural Pixel-to-Semantic Spectrum**: The core insight is simple yet profound—the cascaded residual quantization of RQ-VAE inherently provides information hierarchies at different granularities. Selecting appropriate depth slicing points naturally decouples pixels and semantics without needing auxiliary branches.
- **Elegant Unification of Decoupling + Consistency**: The two types of features share the trajectory prefix (the first 4 levels) in the same space and diverge in subsequent levels, theoretically achieving "local sharing, global division of labor".
- **Strong Performance with Minimal Data**: 13M vs. billions of data in competing products, illustrating that a solid inductive bias is more critical than massive data scales.
- **Transferable Concept**: The concept of residual evolution can be generalized to unified tokenizers for audio (pixel $\to$ semantic) and video (frame-level $\to$ sequence-level semantics).

## Limitations & Future Work
- Currently only supports 256×256 resolution; the scalability to high resolutions (512, 1024) remains to be verified.
- Pixel features at $L_{\text{pix}}=4$ might be insufficient to reconstruct highly detailed textures—while rFID of 0.43 is strong, there is still a gap compared to pure reconstruction-based VAE approaches.
- Semantic alignment depends on SigLIP2; the impact of the teacher model on final performance requires further exploration.
- Generation still employs standard next-token prediction, leaving stronger generative paradigms such as diffusion decoding unexplored.

## Related Work & Insights
- **vs. ViLA-U**: An entangled unified tokenizer where performing VQ + contrastive learning in the same feature space leads to conflicts. EvoTok decouples via depth slicing, improving SEEDBench (71.8 vs. 59.0).
- **vs. DualToken**: Hierarchical decoupling (extracting features from different encoder layers) but lacks a shared space. EvoTok ensures consistency via prefixed sharing of the residual trajectory.
- **vs. TokenFlow**: Decoupling via independent encoders, which is costly to maintain. EvoTok is simpler as it shares a single encoder.
- **vs. EMU3**: An 8B from-scratch integrated model requiring significantly more data and compute. EvoTok achieves competitiveness using a 7B instruction-tuned LLM + 13M training data.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The concept of using residual evolution to unify decoupling and consistency is highly elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid evaluation across understanding, generation, and reconstruction, supplemented by ablation studies and visualization analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured paper with insightful visualization analyses.
- Value: ⭐⭐⭐⭐⭐ An elegant and highly effective new paradigm for unified tokenizers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation](tokenflow_unified_image_tokenizer_for_multimodal_understanding_and_generation.md)
- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](dual_diffusion_unified_generation_understanding.md)
- [\[CVPR 2025\] JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation](janusflow_harmonizing_autoregression_and_rectified_flow_for_unified_multimodal_u.md)
- [\[CVPR 2025\] OmniGen: Unified Image Generation](omnigen_unified_image_generation.md)
- [\[CVPR 2025\] Towards Understanding and Quantifying Uncertainty for Text-to-Image Generation](towards_understanding_and_quantifying_uncertainty_for_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
