---
title: >-
  [Paper Note] AIComposer: Any Style and Content Image Composition via Feature Integration
description: >-
  [ICCV 2025][Image Generation][Cross-domain image composition] AIComposer proposes the first cross-domain image composition method that requires no text prompts. By fusing foreground and background CLIP features via an MLP network, combined with backward inversion + forward denoising and a local cross-attention strategy, the method achieves natural stylization and seamless composition without training the diffusion model, improving LPIPS and CSD metrics by 30.5% and 18.1%, respectively.
tags:
  - ICCV 2025
  - Image Generation
  - Cross-domain image composition
  - text-prompt-free
  - CLIP feature fusion
  - local cross-attention
  - diffusion model inversion
date: 2026-05-08
content_hash: 69ae504882293fbf
---

# AIComposer: Any Style and Content Image Composition via Feature Integration

**Conference**: ICCV 2025
**arXiv**: [2507.20721](https://arxiv.org/abs/2507.20721)
**Code**: [https://github.com/sherlhw/AIComposer](https://github.com/sherlhw/AIComposer)
**Area**: Diffusion Models / Image Composition
**Keywords**: Cross-domain image composition, text-prompt-free, CLIP feature fusion, local cross-attention, diffusion model inversion

## TL;DR

AIComposer proposes the first cross-domain image composition method that requires no text prompts. By fusing foreground and background CLIP features via an MLP network, combined with backward inversion + forward denoising and a local cross-attention strategy, the method achieves natural stylization and seamless composition without training the diffusion model, improving LPIPS and CSD metrics by 30.5% and 18.1%, respectively.

## Background & Motivation

**State of the Field**: Image composition based on large-scale pretrained text-to-image (T2I) diffusion models has achieved significant progress. Existing methods primarily address same-domain image composition, where the foreground and background share similar visual styles.

**Limitations of Prior Work**: Cross-domain image composition remains an underexplored challenge. Key difficulties include: (1) the stochasticity of diffusion models leads to unstable composition results; (2) a pronounced style gap between foreground and background causes visible seams and artifacts upon direct composition; (3) existing methods rely heavily on text prompts to guide the composition process, yet textual descriptions struggle to precisely convey complex visual styles and spatial relationships, limiting practical usability.

**Root Cause**: Cross-domain composition must simultaneously satisfy two conflicting objectives—content preservation and style adaptation. Over-preserving the foreground yields stylistic incoherence, while over-adapting to the style loses foreground details. Existing methods either require additional pre-stylization networks (increasing complexity) or depend on precise text descriptions (reducing usability).

**Paper Goals**: (1) Achieve cross-domain image composition without text prompts; (2) accomplish natural style transfer and content preservation without training the diffusion model; (3) construct a fairly evaluated benchmark dataset for cross-domain composition.

**Starting Point**: The authors observe that the CLIP feature space encodes both content and style information simultaneously. By directly fusing visual signals from the foreground and background at the feature level, text need not serve as an intermediate bridge. Additionally, DDIM inversion enables manipulation of the diffusion process without any training.

**Core Idea**: Replace text-prompt guidance with "CLIP feature MLP fusion + local cross-attention" to achieve training-free (no diffusion model training) cross-domain image composition.

## Method

### Overall Architecture

The AIComposer pipeline proceeds as follows: (1) extract CLIP image features from the foreground and background images separately; (2) fuse the two feature sets via a trained MLP network to produce a unified conditional feature; (3) apply backward DDIM inversion on the background image to obtain its latent representation; (4) during forward denoising, inject the fused features into the diffusion model using a local cross-attention strategy—foreground spatial tokens are guided by foreground CLIP features to preserve content, while background spatial tokens are guided by fused features to achieve style harmony; (5) output the composited image. The entire pipeline requires no diffusion model training; only the lightweight MLP network requires a small amount of training.

### Key Designs

1. **CLIP Feature Integration MLP (Feature Integration Network)**:

    - **Function**: Fuses the CLIP image features of the foreground and background into a unified conditional vector, replacing text embeddings as the guidance signal for the diffusion process.
    - **Mechanism**: CLIP image encoders extract visual features from both the foreground and background independently. The two feature sets are concatenated and fed into a lightweight MLP network, which learns to map them into a joint representation space that captures the semantic content of the foreground while incorporating the stylistic characteristics of the background. The resulting fused features can directly replace text embeddings in IP-Adapter as conditional inputs to the diffusion model. The MLP architecture is simple (a few fully connected layers), requiring minimal training data and computation.
    - **Design Motivation**: Text prompts cannot precisely describe complex visual styles and spatial relationships and require manual user input. CLIP features are extracted directly from images and carry rich visual information, making them more suitable than text for guiding visual composition tasks. MLP-based fusion is more flexible than simple feature concatenation or averaging, as it can learn appropriate feature interaction patterns.

2. **Local Cross-Attention Strategy**:

    - **Function**: Applies differentiated feature guidance to foreground and background regions during the diffusion denoising process, balancing content preservation and style transfer.
    - **Mechanism**: The foreground segmentation mask divides the composition region into foreground and background parts. In the cross-attention layers of the diffusion model, spatial tokens corresponding to the foreground use foreground CLIP features as keys and values (emphasizing content preservation), while background tokens use the fused features (emphasizing style coherence). This spatially adaptive attention guidance ensures that the foreground object's structure and texture are not excessively influenced by the background style, while the foreground boundary regions transition naturally into the background style.
    - **Design Motivation**: Globally uniform feature guidance either causes foreground content loss (over-stylization) or background style incoherence (foreground content dominance). The local strategy elegantly resolves the conflict between content preservation and style transfer. Analogous to mask-guided editing, operating at the attention feature level is more natural than pixel-level manipulation.

3. **Backward Inversion + Forward Denoising (Training-Free Diffusion Control)**:

    - **Function**: Achieves precise control over the composition process without training the diffusion model.
    - **Mechanism**: The background image is first mapped to Gaussian noise space via DDIM backward inversion, yielding a latent representation $z_T$. Forward denoising is then performed from $z_T$, with fused CLIP conditional features and local cross-attention guidance injected at each denoising step. Since inversion preserves the structural information of the background (particularly in early denoising steps), the composition result naturally inherits the background's spatial layout and style. The entire process requires only a small number of denoising steps (approximately 20–50), making it highly efficient.
    - **Design Motivation**: Training or fine-tuning diffusion models demands substantial data and computational resources and may degrade the model's original generative prior. The inversion + conditional injection approach preserves the powerful generative capabilities of the pretrained model while enabling flexible control over the composition process.

### Loss & Training

Only the MLP feature fusion network requires training. The supervision signals include: (1) a content preservation loss to ensure foreground content is retained after composition; and (2) a style consistency loss to ensure the overall style of the composited image is consistent with the background. The diffusion model (SDXL) is kept entirely frozen with its original weights.

## Key Experimental Results

### Main Results

Comparison with state-of-the-art methods on the self-constructed AIComposer Benchmark and existing datasets:

| Method | LPIPS ↓ | CSD ↓ | FID ↓ | User Preference ↑ | Requires Text |
|--------|---------|-------|-------|-------------------|---------------|
| TF-ICON | 0.412 | 0.187 | 42.3 | 18.2% | Yes |
| Magic Insert | 0.385 | 0.165 | 38.7 | 22.5% | Yes |
| AnyDoor | 0.368 | 0.158 | 36.2 | 25.1% | Yes |
| Paint-by-Example | 0.392 | 0.171 | 40.1 | 15.8% | No |
| **AIComposer** | **0.255** | **0.129** | **28.5** | **52.4%** | **No** |

LPIPS improves by 30.5%, CSD by 18.1%, and user preference substantially exceeds all competing methods.

### Ablation Study

| Configuration | LPIPS ↓ | CSD ↓ | Note |
|---------------|---------|-------|------|
| Full AIComposer | **0.255** | **0.129** | Complete model |
| w/o MLP fusion (direct concat) | 0.318 | 0.156 | Simple concatenation fails to learn appropriate fusion patterns |
| w/o local cross-attention | 0.342 | 0.162 | Global guidance leads to foreground content loss |
| w/o backward inversion | 0.385 | 0.178 | Background structural information lost |
| Text replacing CLIP features | 0.312 | 0.149 | Text descriptions lack precision |

### Key Findings

- **MLP fusion and local attention are mutually indispensable**: each contributes to style fusion and content preservation respectively; removing either leads to significant performance degradation.
- **Training-free diffusion control** preserves the integrity of the diffusion prior, outperforming methods that require training.
- **Advantage is most pronounced in cross-domain scenarios**: as the style gap between foreground and background increases (e.g., a realistic photo foreground against an oil-painting-style background), AIComposer's advantage over competing methods grows.
- **High inference efficiency**: only approximately 20–50 DDIM denoising steps are required, with overall speed comparable to IP-Adapter.

## Highlights & Insights

- **Text-free cross-domain composition** represents a practically significant breakthrough—users need only provide two images without composing complex text prompts. This substantially lowers the barrier to image composition, benefiting non-expert users and automated pipelines alike.
- **MLP-based CLIP feature fusion** is a remarkably clean and efficient design—no complex style transfer network or adapter is needed; a single lightweight MLP suffices to perform cross-domain fusion in the feature space. This paradigm is transferable to tasks requiring conditional fusion, such as video composition and 3D scene synthesis.
- **Local cross-attention** applies differentiated conditional guidance to different regions along the spatial dimension, offering finer-grained control than global guidance. This mask-guided attention pattern generalizes to any generation task requiring region-level control.

## Limitations & Future Work

- The method is bounded by the expressive capacity of CLIP features; for fine-grained style distinctions that CLIP struggles to differentiate, fusion quality may be limited.
- The quality of the foreground mask directly affects composition results, requiring accurate segmentation inputs.
- The effectiveness and efficiency of the SDXL-based approach at higher resolutions (e.g., 2K/4K) have not been validated.
- Future work could consider replacing the MLP with a stronger attention-based fusion module or incorporating multi-scale feature fusion to handle more complex scenes.
- Extension to the video domain to achieve temporally consistent cross-domain video composition is a natural future direction.

## Related Work & Insights

- **vs. TF-ICON**: TF-ICON relies on text prompts and textual inversion and requires a larger number of denoising steps. AIComposer eliminates text dependency entirely, yielding a simpler and more efficient pipeline.
- **vs. Magic Insert**: Magic Insert employs auxiliary modules such as ControlNet to maintain structure, but incurs high training costs and exhibits limited cross-domain capability. AIComposer achieves superior cross-domain results in a training-free manner via CLIP feature fusion.
- **vs. IP-Adapter**: AIComposer draws architectural inspiration from IP-Adapter (replacing text conditions with image features), but introduces MLP fusion and local attention strategies specifically designed for cross-domain composition, representing a deepened application of the IP-Adapter paradigm to composition tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First text-free cross-domain image composition method; MLP fusion + local attention design is concise and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — A dedicated cross-domain composition benchmark is constructed; both quantitative and qualitative evaluations are comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem definition is clear; method motivation is well-articulated.
- **Value**: ⭐⭐⭐⭐ — Meaningfully advances cross-domain image composition and lowers practical barriers, though the scope of field-wide impact is relatively focused.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] CSD-VAR: Content-Style Decomposition in Visual Autoregressive Models](csd-var_content-style_decomposition_in_visual_autoregressive_models.md)
- [\[ICCV 2025\] SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models](scflow_implicitly_learning_style_and_content_disentanglement_with_flow_models.md)
- [\[ICCV 2025\] StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion](stylemotif_multi-modal_motion_stylization_using_style-content_cross_fusion.md)
- [\[ICCV 2025\] Balanced Image Stylization with Style Matching Score](balanced_image_stylization_with_style_matching_score.md)
- [\[ICCV 2025\] GAP: Gaussianize Any Point Clouds with Text Guidance](gap_gaussianize_any_point_clouds_with_text_guidance.md)

<!-- RELATED:END -->
