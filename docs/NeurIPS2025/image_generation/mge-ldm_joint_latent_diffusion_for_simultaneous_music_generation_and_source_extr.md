---
title: >-
  [Paper Note] MGE-LDM: Joint Latent Diffusion for Simultaneous Music Generation and Source Extraction
description: >-
  [NeurIPS 2025][Image Generation][music generation] This paper proposes MGE-LDM, the first model to simultaneously achieve music mixture generation, partial generation (source completion)…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "music generation"
  - "source separation"
  - "latent diffusion model"
  - "multi-track modeling"
  - "CLAP"
date: 2026-05-08
content_hash: 92c00fe18f1ff533
---

# MGE-LDM: Joint Latent Diffusion for Simultaneous Music Generation and Source Extraction

**Conference**: NeurIPS 2025
**arXiv**: [2505.23305](https://arxiv.org/abs/2505.23305)
**Code**: N/A (audio demo page provided)
**Area**: Image Generation / Audio Generation
**Keywords**: music generation, source separation, latent diffusion model, multi-track modeling, CLAP

## TL;DR

This paper proposes MGE-LDM, the first model to simultaneously achieve music mixture generation, partial generation (source completion), and text-driven arbitrary source extraction within a unified latent diffusion framework. It jointly models mixture–submixture–source triplets and leverages diffusion inpainting to handle each task.

## Background & Motivation

- **Background**: Music generation models have achieved remarkable progress, yet most can only generate a single mixture waveform and lack fine-grained control over individual instrument tracks. Obtaining individual tracks requires additional source separation techniques.
- **Limitations of Prior Work**: Recent works (MSDM, GMSDI, MSG-LD) attempt to jointly model multi-track audio within a unified diffusion framework, but suffer from two critical limitations:
  - **Reliance on predefined instrument categories**: only fixed instrument sets (e.g., bass, drums, guitar, piano) can be handled.
  - **Linear mixture assumption**: the mixture is assumed to be a linear superposition of individual sources—valid in the waveform domain, but inapplicable under the nonlinear encoder–decoder structure of latent diffusion models.
- **Key Challenge**: These limitations prevent existing methods from handling mixture generation and separation for arbitrary instruments.

## Method

### Overall Architecture

MGE-LDM jointly encodes three latent variables—mixture $z^{(m)}$, submixture $z^{(u)}$, and source $z^{(s)}$—into a shared latent space and trains a three-track diffusion model. At inference, different tasks (mixture generation, source completion, source extraction) are realized via conditional inpainting.

### Key Designs

1. **Three-track joint latent representation**: Given multi-track audio $\{x_i\}$, one track is randomly sampled as the source $x^{(s)}$, the remaining tracks form the submixture $x^{(u)}$, and their combination yields the mixture $x^{(m)}$. Each is mapped to the latent space via a pretrained VAE encoder. This design naturally supports a variable number of instrument tracks without relying on instrument category labels.

2. **Track-aware adaptive timestep (Section 3.4)**: Unlike conventional diffusion inpainting that applies a uniform noise schedule to all regions, the proposed method assigns an independent timestep condition to each track. During training, the model randomly switches among four modes:

   - $(\tau, \tau, \tau)$: standard joint denoising
   - $(0, \tau, \tau)$: mixture is known; denoise submixture and source
   - $(\tau, 0, \tau)$: submixture is known; denoise mixture and source
   - $(\tau, \tau, 0)$: source is known; denoise mixture and submixture

   This enables the model to learn conditional score functions, allowing precise generation of only the missing tracks at inference.

3. **CLAP conditioning and text-driven extraction**: A pretrained CLAP model's audio branch is used to extract conditioning vectors during training, while the text branch extracts prompt embeddings at inference. A random linear interpolation between audio and text embeddings bridges the modality gap. Source extraction is formulated as a conditional inpainting problem—given the mixture latent representation and a text prompt, the model generates the target source.

4. **Dataset-agnostic training paradigm**: Dependency on fixed instrument labels is eliminated. Slakh2100 provides clean individual tracks, while MUSDB18 and MoisesDB contain aggregated labels such as "other." The three-track formulation treats aggregated labels as submixtures, enabling unified training across all datasets.

### Loss & Training

A $v$-objective is adopted with a cosine noise schedule. GroupNorm (3 groups corresponding to the three tracks) replaces LayerNorm. Classifier-Free Guidance (CFG) is supported, with each condition $c^{(k)}$ independently dropped out with probability $p$.

## Key Experimental Results

### Main Results

| Task / Dataset | Metric | MGE-LDM (T1) | MSG-LD | MSDM |
|---|---|---|---|---|
| Mixture generation ($S_A$) | FAD ↓ | **0.47** | 1.38 | 4.21 |
| Mixture generation ($S_\text{Full}$) | FAD ↓ | 1.79 | **1.55** | 6.04 |
| Mixture generation ($M_u$) | FAD ↓ | 6.34 | 4.61 | 7.92 |
| Full-dataset training (T4, $M_u$) | FAD ↓ | **2.78** | - | - |
| Full-dataset training (T4, $M_o$) | FAD ↓ | **1.47** | - | - |

### Ablation Study

| Configuration | Metric | Notes |
|---|---|---|
| Train on $S_A$ only (T1) | FAD 0.47 ($S_A$) | Best in-domain; limited cross-domain generalization |
| Full Slakh (T2) | FAD 0.63 ($S_\text{Full}$) | Diverse training improves cross-instrument generalization |
| Real recordings only (T3) | FAD 2.87/1.59 ($M_u$/$M_o$) | Reasonable cross-domain transfer to synthetic data |
| Full dataset (T4) | Best overall | Broader training distribution yields best overall performance |
| Text-conditioned guided generation | Significant FAD improvement | Constrains generated instrument set, reducing distribution mismatch |

### Key Findings

- Three-track latent space modeling effectively circumvents the linear mixture assumption and supports category-agnostic source manipulation.
- The track-aware adaptive timestep strategy is critical to performance—it enables the model to learn not only the joint distribution but also various conditional distributions.
- Training data diversity (across datasets) directly translates to better cross-domain generalization, potentially at the cost of single-domain optimality.
- Text conditioning is highly effective in guiding generation toward specific instrument sets, alleviating the issue of overly diverse outputs.

## Highlights & Insights

- **Unification**: The first framework to simultaneously address generation, completion, and extraction within a single model for music, without relying on predefined instrument categories.
- **Clever use of latent inpainting**: Source separation is reformulated as a conditional generation problem, bypassing the linear mixture limitation imposed by nonlinear encoders–decoders in latent space.
- **Dataset aggregation capability**: Ambiguous labels such as "other" are naturally incorporated as submixtures, resolving the challenge of unified training across heterogeneous multi-track datasets.

## Limitations & Future Work

- The nonlinear encoder–decoder in the latent space may introduce reconstruction errors that degrade separation quality.
- Reliance on pretrained VAE and CLAP models increases overall system complexity.
- Iterative partial generation (completing sources one at a time) may accumulate errors.
- Source extraction depends on the quality of CLAP text–audio alignment and may be less effective for uncommon instruments.
- No in-depth comparison with end-to-end discriminative separation methods (e.g., Bandit) in terms of separation quality.

## Related Work & Insights

- MSDM pioneered the unified diffusion framework for synthesis and decomposition; this work significantly advances it along two dimensions: category-agnostic modeling and latent space formulation.
- The per-region timestep technique from TD-Paint is creatively transferred to multi-track audio modeling.
- The paper provides a new tool for interactive music production—enabling natural language control over the addition and extraction of individual sources.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Three-track latent modeling and track-aware timesteps are original contributions
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive multi-dataset and multi-task evaluation; subjective listening tests are lacking
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and well-defined mathematical notation
- **Value**: ⭐⭐⭐⭐ A single model covering multiple music production tasks with practical application prospects

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Blind Strong Gravitational Lensing Inversion: Joint Inference of Source and Lens Mass with Score-Based Models](blind_strong_gravitational_lensing_inversion_joint_inference_of_source_and_lens_.md)
- [\[NeurIPS 2025\] A Data-Driven Prism: Multi-View Source Separation with Diffusion Model Priors](a_data-driven_prism_multi-view_source_separation_with_diffusion_model_priors.md)
- [\[NeurIPS 2025\] Large-Scale Training Data Attribution for Music Generative Models via Unlearning](large-scale_training_data_attribution_for_music_generative_models_via_unlearning.md)
- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](../../ICCV2025/image_generation/joint_diffusion_models_in_continual_learning.md)
- [\[NeurIPS 2025\] Diffusion-Driven Progressive Target Manipulation for Source-Free Domain Adaptation](diffusion-driven_progressive_target_manipulation_for_source-free_domain_adaptati.md)

</div>

<!-- RELATED:END -->
