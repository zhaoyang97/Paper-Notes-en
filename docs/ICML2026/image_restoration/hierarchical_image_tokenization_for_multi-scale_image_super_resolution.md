---
title: >-
  [Paper Note] Hierarchical Image Tokenization for Multi-Scale Image Super Resolution
description: >-
  [ICML 2026][Image Restoration][VAR] H-VAR re-slices the "residual quantization for multi-scale generation" VAR paradigm into hierarchical image tokenization (HIT)…
tags:
  - "ICML 2026"
  - "Image Restoration"
  - "VAR"
  - "Residual Quantization"
  - "Multi-Scale Super-Resolution"
  - "Hierarchical Tokenization"
  - "DPO Regularization"
date: 2026-05-08
content_hash: 307e8d11b22f9cf7
---

# Hierarchical Image Tokenization for Multi-Scale Image Super Resolution

**Conference**: ICML 2026  
**arXiv**: [2605.14891](https://arxiv.org/abs/2605.14891)  
**Code**: None  
**Area**: Model Compression / Image Super-Resolution / Visual Autoregression  
**Keywords**: VAR, Residual Quantization, Multi-Scale Super-Resolution, Hierarchical Tokenization, DPO Regularization

## TL;DR
H-VAR re-slices the "residual quantization for multi-scale generation" VAR paradigm into hierarchical image tokenization (HIT), enabling a 310M small model to output three meaningful intermediate resolutions (128 / 256 / 512) in a single forward pass. A DPO regularization term, which does not require an external reward model, is added to bias outputs toward HR. On standard ISR datasets, it competes with the 1B-parameter VARSR.

## Background & Motivation

**Background**: Strong baselines for image super-resolution have long been dominated by GANs (Real-ESRGAN) and diffusion models (StableSR, SeeSR, ResShift). Recently, next-scale prediction VAR, which naturally unfolds residuals by scale, has been adopted by VARSR, PURE, and VARestorer for ISR—offering better alignment between pretraining and downstream tasks than diffusion.

**Limitations of Prior Work**: Existing AR-based super-resolution faces two main drawbacks. First, the original RQ-VAE divides images into $L$ increasingly fine residuals, but the early-stage residuals lack "low-resolution semantics" and only randomly allocate high-frequency details, making intermediate stages unable to decode meaningful low-res images. For $\times 4$ upscaling, the entire token sequence must be processed, preventing simultaneous $\times 2$ outputs. Second, to match SOTA, VARSR requires a 1B large model, classifier-free guidance, and massive labeled data; PURE directly uses a 7B Lumina-mGPT.

**Key Challenge**: The VAR token sequence is a "generic residual stack"—maximizing compression efficiency but lacking the strong constraint of "scale semantics." To achieve meaningful multi-scale outputs, "scale-resolvable" constraints must be enforced in tokenization, but this degrades single-scale reconstruction, presenting an explicit trade-off.

**Goal**: (a) Design a tokenization such that the first $k$ tokens can deterministically decode a valid image at that scale, with tokens shared across scales; (b) Encode a preference for "VAR outputs HR rather than LR" into the training objective without data augmentation or VLMs.

**Key Insight**: The authors observe that next-scale prediction compresses redundancy because the next scale's prediction depends on all tokens from the previous scale. By making "downsampling—quantization—upsampling" a closed loop at each target scale and enforcing token reuse, both multi-scale resolvability and VAR's sequential prediction format can be preserved.

**Core Idea**: HIT (Hierarchical Image Tokenization) slices RQ-VAE residuals by target scale and reuses tokens, combined with a DPO regularization term based on the ratio $p(z_{HR})/p(z_{LR})$, to build a 310M multi-scale H-VAR.

## Method

### Overall Architecture
End-to-end, two main components: (1) Hierarchical RQ-VAE: Finetune the vocabulary and decoder on top of Switti pretrained RQ-VAE, slicing the token sequence into $N$ nested segments $\{s_1, s_2, \dots, s_N\}$, each $s_i$ independently decodable at its scale; (2) Hierarchical VAR: A 16-layer GPT-2 style transformer (310M), conditioned on LR features encoded by RQ-VAE encoder, jointly trained with cross-entropy and DPO, predicting the full token sequence via next-scale prediction. At inference, a single forward pass yields $\times 1 / \times 2 / \times 4$ resolutions, reusing the KV-cache.

### Key Designs

1. **Hierarchical Image Tokenization (HIT)**:

    - **Function**: Segments residual quantization by target scale, ensuring the first $k$ tokens correspond to the "valid image after $\times k$ upsampling."
    - **Mechanism**: Define target scales $s_1 < s_2 < \dots < s_N$ (set as $(0.25, 0.5, 1)$ for $\times 1/\times 2/\times 4$). For each scale $n$, downsample the input to $s_n \rho_L$, encode as $\mathbf{Z}_n$, and quantize residuals at that scale. The quantized tokens are recorded in the $s_n$ subsequence and reused as "starting tokens" for the next scale. Switch to scale $s_{n+1}$, upsample previous tokens to the current residual space, subtract, and quantize new residuals. The image is thus sliced into a nested structure $z = \{\{\{z_1,\dots\}_{s_1},\dots\}_{s_2}, \dots\}_{s_N}$. Simultaneously, finetune the RQ-VAE vocabulary and decoder: keep the decoder frozen, update the vocabulary using the $\ell_2$ distance gradient between encoder features and token embeddings.
    - **Design Motivation**: Original RQ-VAE's early residuals lack "low-resolution correspondence," which is why VAR cannot produce intermediate scales. HIT enforces "the first $k$ tokens must reconstruct scale $k$" as a hard training constraint, injecting a strong inductive bias into the representation space. The authors find this bias is highly effective—it allows reducing the transformer from 1B to 310M while maintaining SOTA, as the "path search space" of the token sequence is greatly compressed.

2. **DPO Regularization for HR Preference**:

    - **Function**: Prevents VAR from lazily predicting tokens highly overlapping with LR, forcing HR sequence output.
    - **Mechanism**: HR and LR tokens overlap significantly at low scales, so the model tends to repeat LR. The authors upsample LR to 512, run HIT to obtain $z_{LR}$, and define $\mathcal{L}_{DPO} = -\log\sigma\left(\beta \log \frac{p(z_{HR})}{p(z_{LR})}\right)$, added equally with standard cross-entropy. $\beta = 0.2$; too small renders the loss ineffective, too large destabilizes training. No "reference policy" or "external reward model" is needed, as LR naturally serves as the "negative sample."
    - **Design Motivation**: Traditional DPO requires pair-wise preference and a reference policy, often needing an external reward model for generative ISR. Here, LR/HR naturally form a preference pair, and AR models can compute log-likelihoods for both sequences (unlike diffusion). Thus, DPO becomes an "unsupervised regularizer"—almost zero cost yet significantly sharpens results.

3. **Multi-Scale Positional Encoding and Conditioning**:

    - **Function**: Enables a single transformer to handle $\sum_l \rho_l^2 = 3452$ tokens at different scales, conditioned on LR.
    - **Mechanism**: Uses an "over-parameterized learnable positional embedding"—a large table declared at the maximum scale, downsampled to each target resolution $\rho_l$ as needed. Unlike VARSR, which uses ControlNet to encode LR, the authors directly bilinearly upsample LR to 512, pass it through the RQ-VAE encoder to obtain 1024 conditioning tokens, saving an independent branch.
    - **Design Motivation**: Positional embedding is prone to bugs in multi-scale training; a unified, downsampled table avoids maintaining multiple weights and allows sharing inductive bias across scales. Using encoder features as condition also eliminates mismatches between ControlNet and main branch scales.

### Loss & Training

- RQ-VAE finetuning: $\mathcal{L}_{RQVAE} = \ell_2 + 5\, \mathcal{L}_{LPIPS}$, AdamW, batch 384, lr 0.00025, 25K steps, 24 A100 GPUs, ~24 hours; following HART, with 50% probability, bypass quantization and feed directly to decoder to prevent vocabulary overfitting.
- H-VAR training: cross-entropy + $\mathcal{L}_{DPO}$ with equal weights; initialized from VAR d-16 official checkpoint, 24 A100 GPUs, 200 epochs, batch 384, lr 1e-3, AdamW betas $(0.9, 0.95)$, ~13 hours total.
- Training data is fully standard: DIV2K + DIV8K + Flickr2K + OST + 10K FFHQ, with LR-HR pairs synthesized using Real-ESRGAN degradation, no proprietary datasets used.

## Key Experimental Results

### Main Results

| Dataset | Metric | StableSR | ResShift | VARSR (1B) | VARSR-d16 | H-VAR (310M, ours) |
|---|---|---|---|---|---|---|
| DIV2K-Val | LPIPS ↓ | 0.323 | 0.428 | 0.326 | 0.495 | **0.317** |
| DIV2K-Val | FID ↓ | 28.32 | 30.79 | 35.51 | 45.96 | **28.86** |
| RealSR | LPIPS ↓ | 0.300 | 0.346 | 0.350 | 0.413 | **0.256** |
| DRealSR | LPIPS ↓ | 0.333 | 0.401 | 0.354 | 0.409 | **0.259** |
| DRealSR | FID ↓ | 148.2 | 159.8 | 155.9 | 244.7 | **145.1** |

| Model | Params | FLOPs | Inference Time | DIV2K-Val FID (LPIPS) |
|---|---|---|---|---|
| H-VAR (Ours) | 310M | 0.921T | 0.25s | 28.86 (0.317) |
| VARSR | 1B | 3.071T | 0.93s | 35.51 (0.326) |
| ResShift | 173M | 2.651T | 0.17s | 30.79 (0.428) |
| StableSR | 919M | 79.94T | 5.51s | 28.32 (0.323) |

### Ablation Study

| Dataset | Config | PSNR@128 | PSNR@256 | PSNR@512 | LPIPS@512 |
|---|---|---|---|---|---|
| RealSR | w/o DPO | 20.56 | 23.09 | 25.72 | 0.310 |
| RealSR | w/ DPO | **22.09** | **24.41** | 25.55 | **0.256** |
| DRealSR | w/o DPO | 23.03 | 26.38 | 28.61 | 0.335 |
| DRealSR | w/ DPO | **25.26** | **27.65** | **28.73** | **0.259** |

| Config (RealSR LPIPS@512) | 128 | 256 | 512 |
|---|---|---|---|
| VARSR (1B) | 0.618 | 0.450 | 0.350 |
| Baseline (RQ-VAE w/o HIT) | 0.686 | 0.491 | 0.311 |
| H-VAR (HIT) | **0.199** | **0.236** | **0.256** |

### Key Findings
- At intermediate scales 128 / 256, the baseline without HIT is nearly unusable (LPIPS > 0.4), while HIT reduces scores to the 0.2 range, confirming it truly produces readable images at intermediate scales, not just as a gimmick.
- HIT acts as a strong inductive bias: reducing the transformer from 1B to 310M and replacing VARSR's proprietary data with standard public datasets, FID/LPIPS still matches or surpasses VARSR. This suggests many issues seemingly solvable only by "scaling data/parameters" are fundamentally due to misaligned token representations.
- DPO regularization consistently improves results across all datasets and scales, requiring no external reward model—a "free lunch" with almost zero cost.
- Side effect: Since early residuals are forcibly allocated to low resolutions, final 512-resolution reconstruction slightly degrades; increasing $L=10 \to 11$ can recover this but inference cost rises by 24%. The authors candidly acknowledge this trade-off.

## Highlights & Insights
- "Encoding multi-scale resolvability into tokenization" is the most memorable technique here—not by modifying the transformer architecture or adding losses, but by constraining the vocabulary upstream; once the upstream is correct, downstream models can be an order of magnitude smaller.
- Using LR itself as the DPO negative sample is a clever "self-supervised preference learning" approach, eliminating the need for a reward model; this trick can be directly transferred to any generative task with a natural degradation pair (deblurring, denoising, style weakening).
- The paper openly discusses trade-offs: HIT slightly sacrifices reconstruction quality at the highest resolution, which must be compensated by more token steps—such transparent discussion of pros and cons is highly commendable.
- Producing three resolutions in a single forward pass is highly practical for real-world applications (mobile, thumbnail preview), offering genuine engineering advantages beyond paper metrics.

## Limitations & Future Work
- Multi-scale is discretely divided into 3 scales; achieving arbitrary upscaling factors ($\times 1.5, \times 3$) requires redesigning $\rho_l$ allocation—this is an inherent discreteness of the tokenization paradigm.
- DPO uses LR as the negative sample, assuming LR is a "bad answer," but when the input is already close-to-HR with mild degradation, this preference may push the model to over-sharpen.
- All experiments are under the standard $\times 4$ setting; it remains unverified whether HIT maintains efficiency at $\times 8 / \times 16$. Higher magnifications introduce more intermediate scales, and whether the token sequence can still fit in a small model needs further validation.
- Comparisons with strong diffusion baselines (e.g., PASD, SUPIR) are missing; the main competitor remains VARSR. To solidify SOTA claims, these comparisons are recommended.

## Related Work & Insights
- **vs VARSR**: Both apply VAR to ISR, but VARSR uses original RQ-VAE, lacks meaningful intermediate scales, and requires a 1B model with large proprietary data; H-VAR uses HIT to solve both issues at once, without needing extra branches like ControlNet.
- **vs PURE**: PURE uses a 7B Lumina-mGPT, embedding both images and degradation descriptions into the vocabulary; H-VAR takes the opposite approach—relying on "upstream token design + simple DPO," demonstrating that ISR does not necessarily require large multimodal models.
- **vs diffusion-based ISR (StableSR / ResShift)**: Diffusion models are slow at inference and cannot compute sequence likelihoods, so native DPO is infeasible; H-VAR's AR form brings both advantages, which is why VAR is chosen over diffusion.
- **Insights**: "Injecting task structure as inductive bias into tokenization" is an underrated direction; next steps could include video AR (token slicing by temporal scale) and medical image AR (token slicing by anatomical hierarchy).

## Rating
- Novelty: ⭐⭐⭐⭐ HIT is the first VAR ISR solution supporting multi-scale; using LR as DPO negative sample is also novel, though the underlying paradigm remains RQ-VAE+VAR.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three baselines, multiple datasets, $L/\rho_l$ sensitivity, and complexity all covered; lacks comparison with diffusion SOTA (PASD/SUPIR).
- Writing Quality: ⭐⭐⭐⭐⭐ Algorithm pseudocode, illustrations, ablations, and limitations are all clearly presented.
- Value: ⭐⭐⭐⭐ Matches 1B performance with 310M, and outputs 3 resolutions per forward pass, offering direct value for industrial deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Disentangled Textual Priors for Diffusion-based Image Super-Resolution](../../CVPR2026/image_restoration/disentangled_textual_priors_for_diffusion-based_image_super-resolution.md)
- [\[CVPR 2026\] Bridging the Perception Gap in Image Super-Resolution Evaluation](../../CVPR2026/image_restoration/bridging_the_perception_gap_in_image_super-resolution_evaluation.md)
- [\[CVPR 2026\] SAT: Selective Aggregation Transformer for Image Super-Resolution](../../CVPR2026/image_restoration/sat_selective_aggregation_transformer_for_image_super_resolution.md)
- [\[CVPR 2026\] UniBlendNet: Unified Global, Multi-Scale, and Region-Adaptive Modeling for Ambient Lighting Normalization](../../CVPR2026/image_restoration/uniblendnet_unified_global_multi_scale_and_region_adaptive_modeling_for_ambient_lighting_normalization.md)
- [\[ICML 2026\] Image Restoration via Diffusion Models with Dynamic Resolution](image_restoration_via_diffusion_models_with_dynamic_resolution.md)

</div>

<!-- RELATED:END -->
