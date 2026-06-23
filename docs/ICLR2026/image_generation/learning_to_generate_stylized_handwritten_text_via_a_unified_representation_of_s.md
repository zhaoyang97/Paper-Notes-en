---
title: >-
  [Paper Note] Learning to Generate Stylized Handwritten Text via a Unified Representation of Style, Content, and Noise
description: >-
  [ICLR 2026][Image Generation][Paper Note] Academic paper note for Learning to Generate Stylized Handwritten Text via a Unified Representation of Style, Content, and Noise.
tags:
  - ICLR 2026
  - Image Generation
date: 2026-05-08
content_hash: 864b0c21ef7acd96
---
| Setting | FID↓ | KID↓ | HWD↓ | ΔCER↓ |
|---|---|---|---|---|
| baseline | 15.12 | 19.27 | 0.97 | 0.11 |
| +APE | 9.31 | 7.21 | 0.58 | 0.05 |
| +R-APE | 7.92 | 4.83 | 0.62 | 0.01 |

Masking Strategy Ablation (IAM):

| Setting | FID↓ | KID↓ | HWD↓ | ΔCER↓ |
|---|---|---|---|---|
| F-TopMask (Fixed top visible) | 8.73 | 6.13 | 0.78 | 0.07 |
| R-Mask (Random multi-region) | 7.92 | 4.83 | 0.62 | 0.01 |

Layout Modeling Ablation (IAM, L1×10³): Autoregressive (Δy 17.04) < Masked Modeling (14.51) < **Masked+CFM (14.39)**. CFM is superior across all four layout parameters.

### Key Findings
- **Positional encoding is key for multi-line generation**: Naïve RoPE works for single lines but often results in the model copying the input image or being resolution-sensitive during multi-line tasks. APE helps the model distinguish style/content tokens, and R-APE further localizes tokens in long-line one-shot settings.
- **Random multi-region masking is superior to fixed top masking**: R-Mask is closer to the true distribution and provides stable improvements across datasets.
- **CFM layout modeling is better than Autoregressive/Masked modeling**: Continuous denoising (10 ODE steps) captures complex spatial dependencies better than token-by-token autoregression.
- **Efficiency**: Built on FLUX.1-Fill-dev with LoRA rank 32, introducing only ~115.9M trainable parameters. Training took 20k steps on 4×A100, with inference requiring ~20 ODE steps.

## Highlights & Insights
- **"Unified Latent Space" is a true paradigm simplification**: Factors previously split by independent encoders and manual losses are merged into one data stream via "Single VAE + Spatial Concatenation + Masked Infilling." This is not just a trick, but a logical result of reframing the problem as inpainting.
- **Insight on masking as "pairing"**: Using $X_{mis}$ as the target and $X_{ctx}$ as the style reference eliminates the messiest preprocessing steps while inherently preserving inter-line stylistic cues and resolution generalization.
- **Reuse of pretrained T2I in-context abilities**: By removing the text encoder and using pure visual conditioning, the model successfully migrates large-scale in-context priors to the HTG domain.
- **Positional encoding engineering**: R-APE's "alignment + rotation" is a precise fix for RoPE's failure on variable-length rows, ensuring spatial proximity between target and content tokens.

## Limitations & Future Work
- **Limited language coverage**: Only Chinese and English were verified; future work needs more languages and datasets to strengthen generalization.
- **Dependency on layout generation quality**: In the two-step decomposition, the content image $X_c$ relies on predicted bounding boxes. Layout errors will propagate to the final handwriting, an effect not analyzed in depth.
- **High cost**: Relying on models like FLUX.1-Fill with 1024×1024 patches on 4×A100 is not friendly to resource-constrained scenarios.
- **Editing boundaries not fully explored**: While character-level editing shows word replacement, the robustness for complex layout rearrangement or cross-line edits lacks quantitative evaluation.

## Related Work & Insights
- **Offline HTG lineage**: GAN era (HiGAN, Alonso et al.) → CNN-Transformer hybrid (HWT, VATr) → Diffusion dominance (One-DM, DiffusionPen, TGC-Diff). InkSpire's "Unified Latent Space" directly advances TGC-Diff's "content + noise sharing" concept.
- **In-context generation**: From InstructPix2Pix to OmniGen and ICEdit. This work is the first to bring unified "edit + generate" in-context capabilities to the handwriting domain.
- **Heuristic**: When multiple "factors" are split by multiple encoders and losses, ask if they can share a representation. Reframing the task into a more general generation paradigm (inpainting here) often leads to significant pipeline simplification and performance gains. Positional encoding mismatch on variable sequences is a recurring issue when migrating pretrained Transformers; R-APE’s approach is a valuable reference.

## Rating
- **Novelty**: ⭐⭐⭐⭐
- **Experimental Thoroughness**: ⭐⭐⭐⭐
- **Writing Quality**: ⭐⭐⭐⭐
- **Value**: ⭐⭐⭐⭐

## Related Papers
- **FLUX.1-Fill**: Backbone model.
- **TGC-Diff**: Integrated content and noise latent.
- **DiffusionPen**: SOTA baseline with triplet loss.

## Related Papers

- [\[CVPR 2026\] SplitFlux: Learning to Decouple Content and Style from a Single Image](../../CVPR2026/image_generation/splitflux_learning_to_decouple_content_and_style_from_a_single_image.md)
- [\[ICCV 2025\] SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models](../../ICCV2025/image_generation/scflow_implicitly_learning_style_and_content_disentanglement_with_flow_models.md)
- [\[ICML 2026\] Content-Style Identification via Differential Independence](../../ICML2026/image_generation/content-style_identification_via_differential_independence.md)
- [\[CVPR 2026\] Learning to Generate via Understanding: Understanding-Driven Intrinsic Rewarding for Unified Multimodal Models](../../CVPR2026/image_generation/learning_to_generate_via_understanding_understanding-driven_intrinsic_rewarding_.md)
- [\[ICLR 2026\] DiffInk: Glyph- and Style-Aware Latent Diffusion Transformer for Text to Online Handwriting Generation](diffink_glyph-_and_style-aware_latent_diffusion_transformer_for_text_to_online_h.md)

</div>

<!-- RELATED:END -->
