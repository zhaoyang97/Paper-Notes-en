---
title: >-
  [Paper Note] BAMM: Bidirectional Autoregressive Motion Model
description: >-
  [ECCV 2024][Image Restoration][Text-to-Motion Generation] BAMM (Bidirectional Autoregressive Motion Model) is proposed. By unifying generative masked modeling and autoregressive modeling through a hybrid attention masking strategy, it simultaneously achieves high-quality motion generation, adaptive length prediction, and zero-shot motion editing within a single framework, comprehensively outperforming SOTA on HumanML3D and KIT-ML.
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "Text-to-Motion Generation"
  - "Bidirectional Autoregressive"
  - "Masked Self-Attention"
  - "Motion Editing"
  - "VQ-VAE"
date: 2026-05-08
content_hash: ea2a3b375a8fc093
---

# BAMM: Bidirectional Autoregressive Motion Model

**Conference**: ECCV 2024  
**arXiv**: [2403.19435](https://arxiv.org/abs/2403.19435)  
**Code**: [https://exitudio.github.io/BAMM-page](https://exitudio.github.io/BAMM-page)  
**Area**: Image Inpainting / Human Motion Generation  
**Keywords**: Text-to-Motion Generation, Bidirectional Autoregressive, Masked Self-Attention, Motion Editing, VQ-VAE

## TL;DR

BAMM (Bidirectional Autoregressive Motion Model) is proposed. By unifying generative masked modeling and autoregressive modeling through a hybrid attention masking strategy, it simultaneously achieves high-quality motion generation, adaptive length prediction, and zero-shot motion editing within a single framework, comprehensively outperforming SOTA on HumanML3D and KIT-ML.

## Background & Motivation

Text-to-3D human motion generation is a key task connecting natural language and human locomotion, widely applied in animation, games, and VR/AR. Current mainstream methods fall into two major regimes, each with inherent limitations:

**Conditional Denoising Motion Models** (diffusion/generative masked types, e.g., MDM, MoMask, MMM): Capture rich dependencies between tokens via bidirectional context, yielding high generation quality and naturally supporting motion editing. However, the **critical limitation is the requirement for pre-determined motion length**—the motion length is unknown in practical scenarios, and using inaccurate length estimates significantly degrades generation quality.

**Conditional Autoregressive Motion Models** (GPT-like, e.g., T2M-GPT, AttT2M): Adaptively determine motion length by generating token-by-token until predicting an [END] token, showing high usability. However, **unidirectional causal attention fails to capture global bidirectional dependencies**, restricting generation quality and preventing temporal motion editing (which requires bidirectional context around edited regions).

**Key Challenge**: There is a fundamental trade-off among usability (length prediction), generation quality (bidirectional dependency), and editability.

**Core Idea**: Unify unidirectional autoregressive and bidirectional generative masked modeling into a single Transformer framework via hybrid attention masking—unidirectional masking handles length prediction and coarse-grained generation, while bidirectional masking handles quality enhancement and motion editing, breaking the trade-off among the three.

## Method

### Overall Architecture

BAMM adopts a two-stage training scheme: (1) training a VQ-VAE motion tokenizer to encode raw 3D motion sequences into discrete tokens; (2) training a conditional masked self-attention Transformer to autoregressively predict masked tokens via a hybrid attention masking strategy. During inference, cascaded motion decoding is employed: first, a unidirectional autoregression generates coarse-grained sequences and determines length, followed by a bidirectional autoregression that refines low-confidence tokens.

### Key Designs

1. **Motion Tokenizer**: Based on VQ-VAE, a motion sequence $\mathcal{M}=[m_1,...,m_\tau]$ ($m \in \mathbb{R}^D$) is mapped to latent space embeddings $z \in \mathbb{R}^{t \times d}$ via an encoder, and then quantized to a codebook $\mathcal{C} = \{\gamma_k\}_{k=1}^K$ using nearest neighbor lookup:
    $$\hat{z_i} = \arg\min_j \|z - \mathcal{C}_j\|_2^2$$
   The training loss is the standard VQ loss: $L_{VQ} = \|\text{sg}(z) - e\|_2^2 + \beta\|z - \text{sg}(e)\\|_2^2$. Exponential moving average (EMA) codebook updates and codebook reset strategies are employed.

2. **Conditional Masked Self-Attention Transformer**: The primary innovation. The input consists of the CLIP text embedding $x_0$, motion tokens $x_{1:t}$, and the [END] token $x_{t+1}$. Unlike traditional masked models that replace inputs with a [MASK] token, BAMM directly modifies the attention score matrix, designing two causal masks:

    - **Unidirectional Causal Mask $M^{uc}$**: Only text tokens can attend bidirectionally; all motion tokens can only attend to their left and themselves, which is equivalent to standard autoregression.
    - **Bidirectional Causal Mask $M^{bc}$**: Text and [END] tokens are unmasked tokens and can attend bidirectionally; masked motion tokens can attend to all left tokens and unmasked right tokens.

   The mask matrix is defined as: $M_{ij} = 0$ when $(i \geq j \land i \notin U) \vee (j \in U)$, otherwise $-\infty$, where $U$ represents the set of unmasked token indices.

    - **Design Motivation**: This design maintains the causality of autoregression (allowing the prediction of [END]) while allowing the model to leverage unmasked "future" tokens, realizing true bidirectional conditional generation.

3. **Cascaded Motion Decoding**: A two-stage decoding strategy during inference:

    - **First Round** (unidirectional autoregression): Uses $M^{uc}$ to sample token-by-token until predicting [END], determining motion length and generating a coarse-grained sequence.
    - **Second Round** (bidirectional autoregressive refinement): Uses $M^{bc}$ to mask a subset of tokens (e.g., masking every other token) and leverages bidirectional context to re-predict low-confidence tokens.
    - The two rounds employ different CFG (Classifier-Free Guidance) scales: $s=4$ for the first round and $s=3$ for the second round.

4. **Residual Motion Refinement**: Minimizes quantization loss further via Residual Vector Quantization (RVQ). Tokens from the first quantizer are generated by the main Transformer, while tokens of the remaining quantizers are predicted by an additional refinement Transformer, eventually merged for decoding.

### Loss & Training

**Hybrid Attention Masking Training**: Unidirectional or bidirectional masks are randomly selected with a probability of $\lambda=0.5$, minimizing the negative log-likelihood:

$$\mathcal{L}_{\text{hybrid}} = -\mathbb{E}_{\mathbf{X} \in p(\mathbf{X})}\left[\lambda \sum_{\forall i \in [1,t]} \log p_\theta(x_i | M^{uc}) + (1-\lambda) \sum_{\forall i \in [1,t]} \log p_\theta(x_i | M^{bc})\right]$$

When the bidirectional mask is selected, $50\%$-$100\%$ of motion tokens are randomly assigned to the masked region. Text tokens are randomly dropped during training to support classifier-free guidance.

During inference, hybrid CFG is used: $\ell_g = (1+s) \cdot \ell_c - s \cdot \ell_u$.

## Key Experimental Results

### Main Results

**HumanML3D Dataset Comparison** (averaged over 20 evaluations, 95% confidence intervals):

| Method | Length Pred. | R-Precision Top-1↑ | FID↓ | MM-Dist↓ | Editable |
|------|---------|-------------------|------|----------|--------|
| T2M-GPT | ✓ | 0.491 | 0.116 | 3.118 | ✗ |
| AttT2M | ✓ | 0.499 | 0.112 | 3.038 | ✗ |
| MMM§ | ✗ | 0.504 | 0.080 | 2.998 | ✓ |
| MoMask§ | ✗ | 0.521 | 0.045 | 2.958 | ✓ |
| **BAMM** | **✓** | **0.525** | **0.055** | **2.919** | **✓** |

§ indicates using ground-truth motion length. BAMM outperforms methods using ground-truth length without requiring it.

**KIT-ML Dataset Comparison**:

| Method | R-Precision Top-1↑ | FID↓ | MM-Dist↓ |
|------|-------------------|------|----------|
| T2M-GPT | 0.402 | 0.717 | 3.053 |
| MoMask§ | 0.433 | 0.204 | 2.779 |
| **BAMM** | **0.438** | **0.183** | **2.723** |

### Ablation Study

**Ablation on CFG Scales and Masking Strategies in Cascaded Decoding** (HumanML3D):

| Ablation Item | Configuration | R-Prec Top-1↑ | FID↓ | MM-Dist↓ |
|--------|------|---------------|------|----------|
| First-round CFG | s=2 / s=3 | 0.517 | 0.105 | 2.956 |
| First-round CFG | s=4 / s=3 (default) | **0.525** | **0.055** | **2.919** |
| First-round CFG | s=5 / s=3 | 0.522 | 0.052 | 2.927 |
| Masking Strategy | 50% Low Confidence | 0.525 | 0.065 | 2.921 |
| Masking Strategy | Confidence < 0.5 | 0.525 | 0.064 | 2.923 |
| Masking Strategy | Suffix (first 50%) | 0.519 | 0.052 | 2.943 |
| Masking Strategy | **Every Other Mask (default)** | **0.525** | **0.055** | **2.919** |
| Iterations | 1 round (AR-only) | 0.524 | 0.064 | 2.926 |
| Iterations | **2 rounds (default)** | **0.525** | **0.055** | **2.919** |
| Iterations | 3 rounds | 0.525 | 0.055 | 2.917 |

**Predicted Length vs. Ground-Truth Length** (HumanML3D):

| Length Source | Method | FID↓ | R-Prec Top-1↑ |
|---------|------|------|---------------|
| Ground-Truth Length | MoMask | 0.045 | 0.521 |
| Predicted Length | MoMask | 0.090 (+100%) | 0.522 |
| Ground-Truth Length | BAMM | 0.055 | 0.522 |
| Adaptive Length | **BAMM** | **0.055** | **0.525** |

**Motion Editing Task** (HumanML3D, comparison with MDM and MoMask):

| Task | Method | R-Prec Top-1↑ | FID↓ |
|------|------|---------------|------|
| Inpainting | MDM | 0.391 | 2.362 |
| Inpainting | MoMask | 0.534 | 0.040 |
| Inpainting | **BAMM** | **0.535** | 0.056 |
| Outpainting | MoMask | 0.531 | 0.057 |
| Outpainting | **BAMM** | **0.535** | **0.056** |

### Key Findings

- Predicted length has a significant impact on denoising models: MoMask's FID deteriorates from 0.045 to 0.090 (doubling), whereas BAMM is unaffected by this.
- Bidirectional refinement (second round iteration) reduces FID from 0.064 to 0.055, and three-round iteration shows no noticeable improvement, indicating two rounds are sufficient.
- The simple "every other mask" strategy yields the best results and is more stable than confidence-based strategies.
- BAMM performs comparably to the specially-trained MoMask in zero-shot motion editing, significantly outperforming MDM.

## Highlights & Insights

- **Unified Framework Addressing the Triple Trade-off**: For the first time, adaptive length (usability), high-quality generation, and zero-shot editing are integrated into a single model, breaking the limitations of both denoising and autoregressive models.
- **Elegant Mask Design**: Modifying the attention matrix directly instead of replacing with [MASK] tokens successfully avoids train-inference inconsistency, making it an elegant design choice.
- **Cascated Decoding Strategy**: The coarse-generation-to-refinement two-stage pipeline represents a coarse-to-fine paradigm in discrete motion generation.
- **Comprehensive Capability Matrix**: As shown in Table 1, BAMM is the only method that simultaneously supports length prediction, length input, and motion editing.

## Limitations & Future Work

- Cascated decoding requires two rounds of inference, which is still slower than a single forward pass, although autoregressive generation is relatively fast.
- Residual vector quantization increases model complexity (requiring an additional refinement Transformer).
- The codebook size and downsampling rate of the VQ-VAE tokenizer impact generating quality, but the paper does not extensively discuss the selection strategy.
- Evaluated only on two English description datasets (HumanML3D and KIT-ML), leaving generalization to more diverse motion types and languages unverified.
- Although feasibility is shown for long-sequence motion generation, quantitative evaluation is missing.

## Related Work & Insights

- **T2M-GPT (CVPR 2023)**: Pioneer of GPT-style autoregressive motion generation, upon which BAMM introduces bidirectional capabilities.
- **MoMask (2023)**: BERT-style generative masked motion model, sharing similar concepts with BAMM's bidirectional refinement phase.
- **MMM (2023)**: Masked motion modeling, facing the same length dependency limitation as MoMask.
- **GLM/CM3Leon**: Unified training of masking and autoregression in LLMs inspired BAMM's hybrid attention design.
- **SoundStorm**: RVQ techniques from audio generation are transferred to the motion refinement module.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The hybrid masking strategy combining masked and autoregressive modeling is highly elegant and addresses the well-recognized triple trade-off in the field.
- Experimental Thoroughness: ⭐⭐⭐⭐ Standard benchmarks, comprehensive SOTA comparisons, extensive ablation analysis, and motion editing task evaluations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, intuitive methodology illustrations, and rigorous problem formulation.
- Value: ⭐⭐⭐⭐ The unified framework is methodologically significant, making an explicit contribution to the field of motion generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FideDiff: Efficient Diffusion Model for High-Fidelity Image Motion Deblurring](../../ICLR2026/image_restoration/fidediff_efficient_diffusion_model_for_high-fidelity_image_motion_deblurring.md)
- [\[ECCV 2024\] MambaIR: A Simple Baseline for Image Restoration with State-Space Model](mambair_a_simple_baseline_for_image_restoration_with_state-space_model.md)
- [\[ECCV 2024\] Spatially-Variant Degradation Model for Dataset-free Super-resolution](spatially-variant_degradation_model_for_dataset-free_super-resolution.md)
- [\[ECCV 2024\] Joint RGB-Spectral Decomposition Model Guided Image Enhancement in Mobile Photography](joint_rgb-spectral_decomposition_model_guided_image_enhancement_in_mobile_photog.md)
- [\[CVPR 2026\] OMoBlur: An Object Motion Blur Dataset and Benchmark for Real-World Local Motion Deblurring](../../CVPR2026/image_restoration/omoblur_an_object_motion_blur_dataset_and_benchmark_for_real-world_local_motion_.md)

</div>

<!-- RELATED:END -->
