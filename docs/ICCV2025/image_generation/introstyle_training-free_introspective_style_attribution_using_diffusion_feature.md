---
title: >-
  [Paper Note] IntroStyle: Training-Free Introspective Style Attribution using Diffusion Features
description: >-
  [ICCV 2025][Image Generation][Style attribution] This paper proposes IntroStyle, a training-free style attribution method that leverages only channel-wise mean and variance statistics from intermediate layers of a diffusion model's own denoising network, measuring inter-image style similarity via the 2-Wasserstein distance. IntroStyle substantially outperforms supervised state-of-the-art methods on WikiArt and DomainNet without any task-specific training.
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Style attribution"
  - "diffusion model features"
  - "training-free"
  - "Wasserstein distance"
  - "copyright protection"
  - "style retrieval"
date: 2026-05-08
content_hash: bde369b0cb5fce8f
---

# IntroStyle: Training-Free Introspective Style Attribution using Diffusion Features

**Conference**: ICCV 2025
**arXiv**: [2412.14432](https://arxiv.org/abs/2412.14432)  
**Code**: [GitHub](https://anandk27.github.io/IntroStyle)  
**Area**: Image Generation / Style Attribution
**Keywords**: Style attribution, diffusion model features, training-free, Wasserstein distance, copyright protection, style retrieval

## TL;DR

This paper proposes IntroStyle, a training-free style attribution method that leverages only channel-wise mean and variance statistics from intermediate layers of a diffusion model's own denoising network, measuring inter-image style similarity via the 2-Wasserstein distance. IntroStyle substantially outperforms supervised state-of-the-art methods on WikiArt and DomainNet without any task-specific training.

## Background & Motivation

- **Core Problem**: T2I diffusion models (e.g., Stable Diffusion, DALL-E 3) are trained on large-scale web data and may replicate copyrighted artistic styles. How can one determine whether a generated image imitates a specific artistic style without retraining the model or introducing external modules?
- **Limitations of Prior Work**:
    - Style unlearning requires expensive model retraining and cannot fully prevent indirect replication via alternative prompts.
    - Style cloaking degrades the authentic viewing experience and places the burden on creators.
    - Existing attribution methods (e.g., CSD, GDA) require training external models (fine-tuned CLIP/DINO), incurring high computational cost and deployment complexity.
    - Semantic entanglement: existing methods tend to retrieve semantically similar rather than stylistically similar images.
- **Key Insight**: Different layers of the diffusion UNet naturally disentangle structural, chromatic, and textural attributes; the internal features of the model are already sufficient for style attribution—no external modules or additional training are required.
- **Dataset Gap**: Existing evaluation benchmarks (e.g., WikiArt) lack fine-grained separation between style and semantics, making it difficult to assess whether a method truly distinguishes "style" from "content."

## Method

### Overall Architecture

IntroStyle treats the diffusion model's denoising network as an autoencoder, extracting style descriptors from intermediate feature layers and measuring style similarity via a probabilistic distance metric. The entire pipeline consists of forward inference passes with no training or fine-tuning required.

**Three-step pipeline**:
1. Image $I$ is encoded into a latent $z_0 = \mathcal{E}(I)$ via the VAE, then noised to timestep $t$: $z_t = \sqrt{\bar{\alpha_t}} z_0 + \sqrt{1-\bar{\alpha_t}} \epsilon_t$
2. $z_t$ is passed through the denoising network $\epsilon_\theta$ (using a null text embedding $c=\varnothing$), and feature tensors $F^{t,idx}$ are extracted from the specified upsampling block $idx$.
3. Channel-wise statistics of the feature tensor are computed as the style descriptor.

### Key Designs: IntroStyle Features

Motivated by the AdaIN formulation in StyleGAN—where style information is encoded primarily in the first-order (mean) and second-order (variance) statistics of features—the per-channel statistics of feature tensor $F^{t,idx}$ are defined as:

$$\mu_c = \frac{1}{WH} \sum_{i,j} F^{t,idx}_{c,i,j}, \quad \sigma_c^2 = \frac{1}{WH} \sum_{i,j} (F^{t,idx}_{c,i,j} - \mu_c)^2$$

The style feature vector is: $f^{t,idx}(I) = (\mu_1, \ldots, \mu_C, \sigma_1^2, \ldots, \sigma_C^2)^T$

Here $t$ (noise timestep) and $idx$ (UNet layer index) are hyperparameters; upsampling block $idx=1$ is used by default.

### Similarity Metric: 2-Wasserstein Distance

Each image's IntroStyle representation is modeled as a $C$-dimensional multivariate Gaussian with diagonal covariance. The style distance between two images is measured by the $W_2$ distance:

$$W_2^2 = \|\mu_1 - \mu_2\|_2^2 + \text{tr}(\Sigma_1 + \Sigma_2 - 2(\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2})^{1/2})$$

The diagonal covariance assumption enables efficient computation. Compared against alternative metrics including L2, Gram matrices, and JSD, $W_2$ yields the best performance.

### ArtSplit Synthetic Dataset

To rigorously evaluate style–semantics disentanglement, the paper introduces the Artistic Style Split (ArtSplit) dataset:
- For each real painting, two types of prompts are designed: "semantic prompts" (removing style cues) and "style prompts" (removing semantic content).
- A diffusion model is used to generate synthetic images covering all "style × semantics" combinations.
- This enables precise quantification of a method's sensitivity to style versus semantics.

## Key Experimental Results

### Main Results: Style Retrieval Performance

| Method | WikiArt mAP@1 | WikiArt mAP@10 | WikiArt Recall@100 | DomainNet mAP@1 |
|------|:---:|:---:|:---:|:---:|
| VGG-Net Gram | 0.259 | 0.194 | 0.804 | - |
| CSD (Prev. SOTA) | requires training | requires training | requires training | requires training |
| **IntroStyle** | **substantial lead** | **substantial lead** | **substantial lead** | **substantial lead** |

- IntroStyle substantially outperforms all baselines (CSD, GDA, etc.) on both WikiArt and DomainNet without any training.
- Qualitative results show that CSD/GDA retrieval is heavily biased by semantic similarity, whereas IntroStyle accurately focuses on style.

### Ablation Study

| Ablation | Finding |
|--------|------|
| UNet layer selection | Upsampling block 1 ($idx=1$) performs best; encoder layers are biased toward semantics. |
| Timestep $t$ | Moderate noise levels are optimal; performance degrades at both extremes. |
| Distance metric | $W_2$ > L2 > Gram > JSD |
| Text conditioning | Null text ($c=\varnothing$) outperforms conditioning on actual prompts. |

### Key Findings

- IntroStyle demonstrates strong style-focusing ability on the ArtSplit style-vs.-semantics disambiguation benchmark.
- The upsampling layers of diffusion models naturally separate style from content, validating the feasibility of introspective attribution.
- The method can be directly applied to style-based rejection sampling to prevent generation of specific artistic styles.

## Highlights & Insights

1. **Zero training overhead**: A fully training-free approach requiring no additional datasets, fine-tuning, or external models—only the diffusion model's own internal features are used.
2. **Strong style–semantics disentanglement**: By selecting the appropriate UNet layer and channel-wise statistics, the method naturally avoids semantic entanglement—a persistent challenge for existing trained approaches.
3. **Theoretical elegance**: The AdaIN-inspired style representation (mean + variance as style) is naturally combined with a probabilistic distribution distance ($W_2$), forming a coherent theoretical framework.
4. **ArtSplit dataset**: Provides the first fully crossed "style × semantics" benchmark for precise evaluation of style attribution methods.
5. **Practical utility**: Directly applicable to copyright protection—detecting whether generated images imitate a specific artist's style.

## Limitations & Future Work

- Relies on a specific diffusion model architecture (Stable Diffusion UNet); generalization to newer architectures such as DiT has not been verified.
- The ArtSplit dataset is based on synthetically generated images, which may introduce distribution bias.
- The definition of "style" is inherently subjective—social-constructive definitions (by artist or genre) do not necessarily capture all dimensions of visual style.
- The noise timestep $t$ and layer index $idx$ are hyperparameters requiring manual selection.
- Applicability to non-Western artistic traditions or digital art styles remains to be validated.

## Related Work & Insights

- **Style transfer**: Gram matrices (Gatys et al.), AdaIN (Huang and Belongie), AdaIN layer-based style control in StyleGAN.
- **Style-aware T2I models**: Textual Inversion, DreamBooth, and related personalized generation methods.
- **Data attribution**: CSD (contrastive style descriptors), GDA (generative data attribution), unlearning-based attribution.
- **Diffusion feature exploitation**: DDAE (latent representation classification), REPA (representation alignment), zero-shot correspondence and segmentation.

## Rating

| Dimension | Score (1–5) |
|------|:---:|
| Novelty | 4 |
| Theoretical Depth | 3.5 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| Value | 4.5 |
| **Overall** | **4.0** |

The core contribution lies in the discovery that channel-wise statistics of diffusion model intermediate features serve as effective style descriptors—a minimal yet highly effective approach. The ArtSplit dataset fills a gap in style attribution evaluation. The main limitations are the method's dependency on the UNet architecture and the insufficient theoretical explanation of the "introspective" capability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] K-LoRA: Unlocking Training-Free Fusion of Any Subject and Style LoRAs](../../CVPR2025/image_generation/k-lora_unlocking_training-free_fusion_of_any_subject_and_style_loras.md)
- [\[CVPR 2026\] A Training-Free Style-Personalization via SVD-Based Feature Decomposition](../../CVPR2026/image_generation/a_training-free_style-personalization_via_svd-based_feature_decomposition.md)
- [\[ICCV 2025\] LaRender: Training-Free Occlusion Control in Image Generation via Latent Rendering](larender_training-free_occlusion_control_in_image_generation_via_latent_renderin.md)
- [\[AAAI 2026\] AEDR: Training-Free AI-Generated Image Attribution via Autoencoder Double-Reconstruction](../../AAAI2026/image_generation/aedr_training-free_ai-generated_image_attribution_via_autoen.md)
- [\[ICCV 2025\] Anchor Token Matching: Implicit Structure Locking for Training-free AR Image Editing](anchor_token_matching_implicit_structure_locking_for_training-free_ar_image_edit.md)

</div>

<!-- RELATED:END -->
