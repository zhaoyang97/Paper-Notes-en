---
title: >-
  [Paper Note] Token Perturbation Guidance for Diffusion Models
description: >-
  [NeurIPS 2025][Image Generation][Training-free guidance] This paper proposes Token Perturbation Guidance (TPG), which constructs a negative score signal by applying norm-preserving shuffling perturbations to intermediate token representations in diffusion models, enabling training-free, condition-agnostic guidance. TPG improves the FID of SDXL by nearly 2× in unconditional generation and approaches CFG-level performance in conditional generation.
tags:
  - NeurIPS 2025
  - Image Generation
  - Training-free guidance
  - token perturbation
  - unconditional generation
  - CFG alternative
  - diffusion models
date: 2026-05-08
content_hash: 0b7b8434deb1c89a
---

# Token Perturbation Guidance for Diffusion Models

**Conference**: NeurIPS 2025
**arXiv**: [2506.10036](https://arxiv.org/abs/2506.10036)
**Code**: [https://github.com/TaatiTeam/Token-Perturbation-Guidance](https://github.com/TaatiTeam/Token-Perturbation-Guidance)
**Area**: Diffusion Models / Image Generation
**Keywords**: Training-free guidance, token perturbation, unconditional generation, CFG alternative, diffusion models

## TL;DR

This paper proposes Token Perturbation Guidance (TPG), which constructs a negative score signal by applying norm-preserving shuffling perturbations to intermediate token representations in diffusion models, enabling training-free, condition-agnostic guidance. TPG improves the FID of SDXL by nearly 2× in unconditional generation and approaches CFG-level performance in conditional generation.

## Background & Motivation

Classifier-Free Guidance (CFG) is a cornerstone technique for improving generation quality and condition alignment in modern diffusion models, yet it has two fundamental limitations: (1) it is applicable only to conditional generation and cannot be used for unconditional generation; and (2) it requires a specific training strategy (randomly replacing conditions with null conditions).

Several training-free alternatives have been proposed—SAG (Self-Attention Guidance), PAG (Perturbed Attention Guidance), and SEG (Smooth Energy Guidance)—which construct guidance signals by manipulating attention layers. However, their improvements remain limited, particularly in prompt alignment and unconditional generation quality, falling well short of CFG.

A core observation motivating this work is that **PAG and SEG produce overly smooth results during the early denoising steps**, which are critical for establishing global structure and coarse semantics. Insufficient guidance at this stage may prevent the model from recovering from high-level semantic misalignment, explaining why existing methods yield only marginal improvements in prompt alignment and generation quality.

TPG addresses this by directly applying **norm-preserving shuffling perturbations** to intermediate token representations within the diffusion network, constructing a more effective guidance signal than attention manipulation.

## Method

### Overall Architecture

Let $H \in \mathbb{R}^{B \times N \times C}$ denote the intermediate hidden representation of the denoiser. At each timestep, TPG performs two forward passes: (1) a standard pass yielding the positive score $s_\theta^+$; and (2) a pass with token shuffling perturbation applied at selected layers, yielding the negative score $s_\theta^-$. The guided output is:

$$\tilde{s}_\theta = s_\theta^+ + \gamma(s_\theta^+ - s_\theta^-)$$

### Key Designs

1. **Token Shuffling Perturbation**:

    - A permutation matrix $S \in \mathbb{R}^{N \times N}$ is applied along the token dimension to yield $H' = S \cdot H$.
    - Shuffling satisfies three key properties:
        - **Linearity**: expressible as matrix multiplication, with computational cost comparable to CFG.
        - **Norm-preserving**: $S^T \cdot S = I$; as an orthogonal transformation, it preserves token norms and avoids internal covariate shift.
        - **Disrupts local structure while preserving global statistics**: randomly permutes token positions.
    - An **independent random permutation matrix** $S_{k,t}$ is used for each timestep $t$ and each layer $k$.

2. **Why Shuffling Outperforms Other Norm-Preserving Perturbations**:

    - Sign Flip, Hadamard transform, and Haar random orthogonal transforms are compared as alternatives.
    - Shuffling substantially outperforms all others on FID: 78.43 vs. 118–120 (evaluated on 5K samples).
    - Hadamard and Haar transforms mix all tokens together, potentially destroying useful information; Sign Flip produces too weak a signal; Shuffling randomly reorders tokens while preserving recoverable global structure.

3. **In-Depth Comparison with CFG Behavior**:

    - **Directional analysis**: The guidance vectors of both TPG and CFG are nearly orthogonal to the true noise throughout the denoising trajectory (cosine similarity close to 0), whereas PAG/SEG exhibit strong negative alignment at intermediate steps.
    - **Norm analysis**: The norm trends of TPG and CFG guidance terms are nearly identical (starting at ~40 and rising steeply in later steps), while PAG/SEG maintain consistently low norms.
    - **Frequency-domain analysis**: TPG and CFG show a slight positive bias in the low-frequency band and remain orthogonal at other frequencies; SEG exhibits negative stripes in mid-frequencies with energy two orders of magnitude smaller than CFG/TPG.
    - Conclusion: TPG most closely approximates CFG behavior in terms of both direction and frequency content.

4. **Compatibility with DiT/ViT Architectures**:

    - Residual connections in U-Net architectures help recover perturbed tokens, but DiT's sequential Transformer layers allow degradation to accumulate.
    - For SD3 (DiT architecture): only a small subset of tokens are shuffled, and unshuffling is applied immediately after each Transformer block.
    - PAG performs even worse than vanilla on SD3 (FID 138.08 vs. 113.86), whereas TPG achieves a substantial improvement (83.01).

### Loss & Training

- TPG is a **fully training-free** method requiring no modification to the model architecture.
- The guidance scale $\gamma$ is fixed at 3.0.
- Perturbation is applied only to the downsampling layers (encoder portion) of the U-Net.
- Plug-and-play, requiring only a few lines of code.

## Key Experimental Results

### Main Results (SDXL, 30K samples, MS-COCO 2014 validation set)

| Setting | Method | FID↓ | sFID↓ | IS↑ | Aesthetic↑ | CLIP↑ |
|---------|--------|------|-------|-----|-----------|-------|
| Unconditional | Vanilla SDXL | 124.04 | 78.91 | 9.19 | 5.02 | - |
| Unconditional | PAG | 98.83 | 94.71 | 13.74 | 5.94 | - |
| Unconditional | SEG | 82.64 | 74.98 | 13.22 | 6.15 | - |
| Unconditional | **TPG** | **69.31** | **44.18** | **17.99** | 6.14 | - |
| Conditional | Vanilla SDXL | 48.97 | 43.71 | 22.10 | 5.37 | 27.47 |
| Conditional | CFG | **12.79** | **23.31** | **42.75** | **6.20** | **32.03** |
| Conditional | PAG | 20.49 | 28.78 | 34.66 | 6.11 | 29.67 |
| Conditional | SEG | 23.94 | 31.50 | 30.29 | 6.18 | 29.49 |
| Conditional | **TPG** | 17.77 | 24.32 | 34.89 | 6.12 | 30.15 |

### Ablation Study (Different Perturbation Methods, 5K samples)

| Method | FID↓ | IS↑ |
|--------|------|-----|
| Vanilla (no perturbation) | 131.57 | 9.21 |
| Sign Flip | 119.23 | 10.98 |
| Hadamard | 120.54 | 10.34 |
| Haar Random Orthogonal | 118.47 | 10.75 |
| Token Blurring (non-norm-preserving) | 157.67 | 6.70 |
| **Token Shuffling** | **78.43** | **18.26** |

### Key Findings

- Unconditional generation: TPG reduces SDXL FID from 124.04 to 69.31, representing nearly a 2× improvement.
- Conditional generation: TPG (FID=17.77) closely follows CFG (FID=12.79) and substantially outperforms PAG (20.49) and SEG (23.94).
- TPG also achieves the best unconditional generation results on SD 2.1 (FID 16.69 vs. PAG 21.30 vs. SEG 20.98).
- The norm-preserving property is critical: Token Blurring (non-norm-preserving) performs worse than vanilla.
- Guidance scale $\gamma=3$ is optimal; excessively large $\gamma$ (>4) leads to FID degradation.
- On SD3 (DiT architecture), PAG completely fails while TPG remains significantly effective (unconditional FID 83.01 vs. Vanilla 113.86).

## Highlights & Insights

- **Simple yet profound core mechanism**: Constructing a negative score via token shuffling is conceptually minimal yet remarkably effective, capturing the essence of guidance as "local structure disruption with global statistics preservation."
- **In-depth frequency-domain analysis of guidance mechanisms**: Reveals the fundamental differences between CFG, TPG, and PAG/SEG—guidance vectors should be orthogonal to the noise rather than anti-aligned.
- **Condition-agnostic universal guidance**: For the first time, CFG-level guidance effects are extended to unconditional generation.
- **Cross-architecture compatibility (U-Net + DiT)**: Adapts to DiT architectures via a shuffle-unshuffle strategy.

## Limitations & Future Work

- Like CFG, TPG requires two forward passes per step, doubling sampling time.
- In conditional generation, TPG still trails CFG (FID 17.77 vs. 12.79), with a gap of approximately 5 FID points.
- Guidance effectiveness may be limited in extreme out-of-distribution scenarios.
- Optimal guidance scale and layer selection require empirical tuning.

## Related Work & Insights

- CFG is the gold standard for conditional generation, but fundamentally operates by steering away from the unconditional score.
- Autoguidance constructs guidance signals using a weaker denoiser but still requires training.
- PAG (replacing attention maps with identity matrices) and SEG (applying Gaussian blur to attention maps) underperform TPG in both practical effectiveness and theoretical behavior.
- Key insight: Token-level perturbation constructs more effective guidance signals than attention-layer manipulation, opening a new direction for training-free guidance methods.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Where and How to Perturb: On the Design of Perturbation Guidance in Diffusion and Flow Models](where_and_how_to_perturb_on_the_design_of_perturbation_guidance_in_diffusion_and.md)
- [\[NeurIPS 2025\] Entropy Rectifying Guidance for Diffusion and Flow Models](entropy_rectifying_guidance_for_diffusion_and_flow_models.md)
- [\[NeurIPS 2025\] Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)
- [\[NeurIPS 2025\] SparseDiT: Token Sparsification for Efficient Diffusion Transformer](sparsedit_token_sparsification_for_efficient_diffusion_transformer.md)
- [\[NeurIPS 2025\] Prompt-Based Safety Guidance Is Ineffective for Unlearned Text-to-Image Diffusion Models](prompt-based_safety_guidance_is_ineffective_for_unlearned_text-to-image_diffusio.md)

</div>

<!-- RELATED:END -->
