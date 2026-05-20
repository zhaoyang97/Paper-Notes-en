---
title: >-
  [Paper Note] AlignTok: Aligning Visual Foundation Encoders to Tokenizers for Diffusion Models
description: >-
  [ICLR 2026][Image Generation][visual tokenizer] This paper proposes AlignTok, which aligns pretrained visual foundation encoders (e.g.…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "visual tokenizer"
  - "latent diffusion"
  - "DINOv2"
  - "semantic alignment"
date: 2026-05-08
content_hash: c4fe0274c48368d0
---

# AlignTok: Aligning Visual Foundation Encoders to Tokenizers for Diffusion Models

**Conference**: ICLR 2026
**arXiv**: [2509.25162](https://arxiv.org/abs/2509.25162)  
**Code**: [https://aligntok.github.io](https://aligntok.github.io)  
**Area**: Diffusion Models
**Keywords**: visual tokenizer, latent diffusion, DINOv2, semantic alignment, image generation

## TL;DR
This paper proposes AlignTok, which aligns pretrained visual foundation encoders (e.g., DINOv2) into continuous tokenizers for diffusion models. Through a three-stage alignment strategy—semantic latent space establishment → perceptual detail supplementation → decoder refinement—AlignTok constructs a semantically rich latent space, achieving gFID 1.90 on ImageNet 256×256 in only 64 epochs, converging faster and yielding better generation quality than VAEs trained from scratch.

## Background & Motivation
**Background**: Latent diffusion models (LDMs) rely on VAEs as tokenizers to define the latent space. Standard VAEs are trained with reconstruction loss combined with mild KL regularization, resulting in latent spaces dominated by low-level details.

**Limitations of Prior Work**: (1) VAE encoders learn semantics from scratch indirectly (solely through reconstruction loss), leading to unpredictable latent space structure; (2) semantic regularization methods (e.g., VA-VAE) incorporate alignment losses with pretrained encoders during training, but the encoder still must learn semantic structure from scratch.

**Key Challenge**: Learning semantics is fundamentally harder than learning reconstruction. When training from scratch, the encoder must simultaneously handle semantic structure and reconstruction details, with the two objectives competing against each other.

**Goal**: How to construct a tokenizer that is both semantically rich (beneficial for diffusion) and capable of high-quality reconstruction?

**Key Insight**: Rather than learning semantics from scratch, directly leverage existing pretrained encoders. The key challenge is that pretrained encoders lack reconstruction capability—requiring alignment rather than regularization.

**Core Idea**: Instead of training the encoder to learn semantics from scratch (regularization), directly align a pretrained encoder that already possesses semantic representations (alignment).

## Method

### Overall Architecture
AlignTok proceeds in three stages: (1) freeze the DINOv2 encoder, train an adapter and decoder to establish a semantic latent space; (2) jointly optimize all components with a semantic preservation loss, enabling the encoder to capture perceptual details without losing semantics; (3) fine-tune only the decoder to improve reconstruction quality.

### Key Designs

1. **Stage 1: Latent Alignment**:

    - Function: Establish a semantic latent space using frozen DINOv2 encoder features.
    - Mechanism: $z_0 = A(E_p(x))$, where adapter $A$ projects high-dimensional features (1024ch) to low-dimensional latent codes (32ch), and decoder $D$ reconstructs the image. Only $A$ and $D$ are trained; no KL regularization is applied.
    - Design Motivation: Freezing the encoder ensures that semantics are not corrupted, though reconstruction quality is limited since the encoder does not capture low-level details.

2. **Stage 2: Perceptual Alignment**:

    - Function: Unfreeze the encoder to capture low-level details while preserving semantics.
    - Mechanism: Jointly optimize $E_p, A, D$ with a semantic preservation loss $\mathcal{L}_{sp} = L_{\ell_2}(z_0^*, z_0)$, which constrains the current latent codes to remain consistent with those produced by the frozen Stage 1 model. The total loss is $\mathcal{L} = \mathcal{L}_{rec} + w_{sp}\mathcal{L}_{sp}$.
    - Design Motivation: Without the semantic preservation loss, linear probing accuracy drops catastrophically from 41% to 9.5%, indicating that the encoder suffers from catastrophic forgetting of semantics. $w_{sp}=1$ is identified as the optimal balance point.

3. **Stage 3: Decoder Refinement**:

    - Function: Freeze the encoder and adapter; fine-tune only the decoder to improve reconstruction.
    - Design Motivation: The continuously evolving latent space in the first two stages may cause the decoder to underfit. Locking the latent space and optimizing the decoder independently further improves reconstruction fidelity.

### Loss & Training
Reconstruction loss: L1 + perceptual loss + adversarial loss. Semantic preservation loss: L2 distance between latent codes from the two stages. No KL regularization is applied. DINOv2-L/14 is used as the default base encoder, with a downsampling ratio of 16 and 32 latent channels.

## Key Experimental Results

### Main Results (ImageNet 256×256)

| Method | rFID↓ | gFID↓ | IS↑ | Recall↑ |
|--------|-------|-------|-----|---------|
| SD-VAE (from scratch) | 0.91 | 2.66 | - | - |
| VA-VAE (semantic regularization) | 0.49 | 2.14 | - | - |
| **AlignTok** (alignment) | **0.26** | **1.90** | **260.6** | **0.599** |

### Ablation Study

| Configuration | rFID | gFID | Linear Probing Acc |
|---------------|------|------|--------------------|
| No semantic preservation loss (w=0) | **0.33** | 3.05 | 9.5% |
| w=1 (optimal) | 0.36 | **2.19** | 35.1% |
| w=5 | 0.49 | 2.48 | 40.6% |
| Stage 1 only | 1.63 | 3.00 | **41.5%** |
| Stage 1+2 | 0.36 | 2.19 | 35.1% |
| Full (Stage 1+2+3) | **0.26** | **2.17** | 35.1% |

### Key Findings
- DINOv2 outperforms SigLIP2 and MAE as the base encoder—DINOv2's self-supervised features are better suited for diffusion modeling.
- In text-to-image experiments on LAION, AlignTok consistently outperforms FLUX VAE and VA-VAE under the same number of training steps.
- Removing KL regularization yields better results—KL distorts the semantic structure of the encoder.
- The semantic preservation loss should be applied after the adapter rather than before; granting the adapter excessive freedom causes semantic loss.
- LoRA fine-tuning is insufficient—Stage 2 requires full fine-tuning to balance semantics and reconstruction.

## Highlights & Insights
- **Paradigm Shift: Alignment vs. Regularization**: Rather than learning semantics from scratch, AlignTok directly reuses the semantic capabilities of pretrained visual foundation models—an approach that is both elegant and efficient. This paradigm is generalizable to any generative model requiring a semantic latent space.
- **Simplicity and Effectiveness of the Semantic Preservation Loss**: A single L2 loss is sufficient to prevent catastrophic forgetting while allowing the encoder to learn perceptual details.
- **Design Philosophy of Three-Stage Progressive Alignment**: The pipeline proceeds from semantics → perception → reconstruction, addressing one objective per stage and avoiding conflicting optimization targets.

## Limitations & Future Work
- Validation is currently limited to ImageNet 256×256 and LAION; performance at higher resolutions remains to be confirmed.
- The fixed 14×14 patch size of DINOv2 may limit flexibility in resolution adaptation.
- Integration with RAE (which directly uses a frozen encoder for a high-dimensional latent space) has not yet been explored.
- The three-stage training pipeline introduces additional complexity.

## Related Work & Insights
- **vs. VA-VAE (semantic regularization)**: AlignTok directly aligns a pretrained encoder rather than learning from scratch with regularization, improving gFID from 2.14 to 1.90.
- **vs. FLUX VAE**: On LAION text-to-image generation, AlignTok converges significantly faster.
- **vs. RAE (frozen encoder)**: RAE avoids fine-tuning but requires specialized high-dimensional diffusion techniques; AlignTok, after fine-tuning, operates in a lower-dimensional space (32ch) that is more standard.
- **vs. REPA-E (end-to-end)**: The two approaches are complementary—REPA-E can initialize its tokenizer with AlignTok.

## Rating
- Novelty: ⭐⭐⭐⭐ The alignment-over-regularization paradigm is elegant and well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Detailed ablations, multi-encoder comparisons, ImageNet + LAION evaluations, and clear positioning relative to concurrent work.
- Writing Quality: ⭐⭐⭐⭐⭐ Logic is clear and the relationship with concurrent work is thoroughly discussed.
- Value: ⭐⭐⭐⭐⭐ Establishes a new paradigm for tokenizer design in diffusion models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Asynchronous Denoising Diffusion Models for Aligning Text-to-Image Generation](asynchronous_denoising_diffusion_models_for_aligning_text-to-image_generation.md)
- [\[ICLR 2026\] PCPO: Proportionate Credit Policy Optimization for Aligning Image Generation Models](pcpo_proportionate_credit_policy_optimization_for_aligning_image_generation_mode.md)
- [\[ICCV 2025\] Mind the Gap: Aligning Vision Foundation Models to Image Feature Matching](../../ICCV2025/image_generation/mind_the_gap_aligning_vision_foundation_models_to_image_feature_matching.md)
- [\[ICLR 2026\] Generalization of Diffusion Models Arises with a Balanced Representation Space](generalization_of_diffusion_models_arises_with_a_balanced_representation_space.md)
- [\[ICLR 2026\] Zatom-1: A Multimodal Flow Foundation Model for 3D Molecules and Materials](zatom-1_a_multimodal_flow_foundation_model_for_3d_molecules_and_materials.md)

</div>

<!-- RELATED:END -->
