---
title: >-
  [Paper Note] Hierarchical Image Tokenization for Multi-Scale Image Super Resolution
description: >-
  [ICML 2026][Model Compression][VAR] H-VAR reslices the VAR paradigm of "residual quantization for multi-scale generation" into Hierarchical Image Tokenization (HIT). This allows a small 310M model to output three meaning…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "VAR"
  - "Residual Quantization"
  - "Multi-scale SR"
  - "Hierarchical Tokenization"
  - "DPO Regularization"
date: 2026-05-08
content_hash: 824c27f55a0f8dd3
---

# Hierarchical Image Tokenization for Multi-Scale Image Super Resolution

**Conference**: ICML 2026  
**arXiv**: [2605.14891](https://arxiv.org/abs/2605.14891)  
**Code**: None  
**Area**: Model Compression / Image Super-Resolution / Visual Autoregressive  
**Keywords**: VAR, Residual Quantization, Multi-scale SR, Hierarchical Tokenization, DPO Regularization

## TL;DR
H-VAR reslices the VAR paradigm of "residual quantization for multi-scale generation" into Hierarchical Image Tokenization (HIT). This allows a small 310M model to output three meaningful intermediate resolutions (128 / 256 / 512) in a single forward pass. Combined with a DPO regularization term that biases output toward HR without requiring an external reward model, it competes with the 1B-parameter VARSR on standard ISR datasets.

## Background & Motivation

**Background**: Strong baselines for image super-resolution (ISR) have long been dominated by GANs (Real-ESRGAN) and Diffusion Models (StableSR, SeeSR, ResShift). Recently, Visual Autoregressive (VAR) models based on next-scale prediction have been applied to ISR (VARSR, PURE, VARestorer), as their pre-training aligns better with downstream tasks than diffusion.

**Limitations of Prior Work**: Existing AR-based SR models have two major drawbacks. First, the original RQ-VAE divides an image into $L$ increasingly refined residuals, but the early levels lack "low-resolution semantics" and are merely random allocations of high-frequency details. Consequently, intermediate stages cannot be decoded into meaningful low-res images; $\times 4$ SR requires running the entire token sequence, failing to yield $\times 2$ results simultaneously. Second, to match SOTA performance, VARSR requires a 1B model + classifier-free guidance + massive labeled data, while PURE directly utilizes the 7B Lumina-mGPT.

**Key Challenge**: The token sequences in VAR are "generic residual stacks"—highly efficient for compression but lacking "scale semantics" as a strong constraint. Making multi-scale decoding meaningful requires embedding "scale-decodability" into the tokenization process, which introduces an explicit trade-off with single-scale reconstruction quality.

**Goal**: (a) Design a tokenization scheme where the first $k$ tokens deterministically decode into a valid image at that scale, with tokens shared across scales. (b) Hard-code the preference for "VAR outputting HR rather than LR" into the training objective without scaling data or using VLMs.

**Key Insight**: The authors observe that next-scale prediction reduces redundancy because predictions for the next scale depend on all tokens from the previous scale. If "downsampling-quantization-upsampling" is made a closed loop at each target scale with forced token reuse, one can preserve both multi-scale decodability and the VAR sequence prediction format.

**Core Idea**: HIT (Hierarchical Image Tokenization) is used to slice RQ-VAE residuals according to target scales for token reuse. Combined with a DPO regularization term using the ratio $p(z_{HR})/p(z_{LR})$, this forms a 310M multi-scale H-VAR.

## Method

### Overall Architecture
The end-to-end framework performs two tasks: (1) Hierarchical RQ-VAE: Finetuning the vocabulary and decoder based on a Switti pre-trained RQ-VAE, so the token sequence is sliced into $N$ nested segments $\{s_1, s_2, \dots, s_N\}$, each independently decodable to its respective scale. (2) Hierarchical VAR: A 16-layer GPT-2 style transformer (310M) conditioned on LR features encoded by the RQ-VAE encoder, trained with combined cross-entropy and DPO losses for next-scale prediction. At inference, a single forward pass provides $\times 1 / \times 2 / \times 4$ resolutions by reusing the KV-cache.

### Key Designs

1.  **Hierarchical Image Tokenization (HIT)**:
    - **Function**: Segments residual quantization by target scales, ensuring the first $k$ tokens correspond to a "valid image after $\times k$ upsampling."
    - **Mechanism**: Define target scales $s_1 < s_2 < \dots < s_N$ (e.g., $(0.25, 0.5, 1)$ for $\times 1/\times 2/\times 4$). For each scale $n$, the input image is downsampled to $s_n \rho_L$ to encode $\mathbf{Z}_n$. Residuals are quantized at this scale; the resulting tokens are recorded in the $s_n$ subsequence and reused as "starting tokens" for the next scale. After switching to scale $s_{n+1}$, tokens from the previous scale are upsampled to the current residual space and subtracted before quantizing new residuals. This creates a nested structure $z = \{\{\{z_1,\dots\}_{s_1},\dots\}_{s_2}, \dots\}_{s_N}$. The RQ-VAE vocabulary is updated using gradients of the $\ell_2$ distance between encoder features and token embeddings while keeping the decoder frozen.
    - **Design Motivation**: The lack of low-resolution constraints in early RQ-VAE residuals is why VAR cannot produce intermediate scales. HIT imposes "first $k$ tokens must reconstruct scale $k$" as a hard constraint, injecting a strong inductive bias into the representation space. This allows the transformer to be reduced from 1B to 310M parameters while maintaining SOTA performance.

2.  **DPO Regularization for HR Preference**:
    - **Function**: Prevents the VAR model from "lazily" predicting tokens that overlap heavily with the LR image, forcing it to output HR sequences.
    - **Mechanism**: Observing that HR and LR tokens overlap significantly at low scales, the model tends to repeat the LR. The authors pass the LR image (upsampled to 512) through HIT to obtain $z_{LR}$, then define $\mathcal{L}_{DPO} = -\log\sigma\left(\beta \log \frac{p(z_{HR})}{p(z_{LR})}\right)$, added to the cross-entropy loss with equal weight ($\beta = 0.2$). This requires no "reference policy" or "external reward model" because the LR naturally serves as the "negative sample."
    - **Design Motivation**: Traditional DPO requires pair-wise preferences and a reference policy, often necessitating external reward models in generative ISR. Here, LR/HR naturally form a preference pair, and AR models can calculate log-likelihoods (unlike diffusion), making DPO a "zero-cost" unsupervised regularizer that sharpens results.

3.  **Multi-scale Positional Encoding and Conditioning**:
    - **Function**: Enables a single transformer to process $\sum_l \rho_l^2 = 3452$ tokens across different scales while being conditioned on LR.
    - **Mechanism**: Uses over-parameterized learnable positional embeddings—a large master table is declared for the maximum scale and downsampled for each resolution $\rho_l$. Instead of a ControlNet, the LR is bilinearly upsampled to 512 and passed through the RQ-VAE encoder to obtain 1024 conditioning tokens.
    - **Design Motivation**: A unified downsampled table avoids maintaining multiple weight sets and shares positional biases across scales. Using encoder features as conditions eliminates scale-mismatch issues common with ControlNet.

### Loss & Training
- **RQ-VAE Finetuning**: $\mathcal{L}_{RQVAE} = \ell_2 + 5\, \mathcal{L}_{LPIPS}$, using AdamW, batch size 384, lr 0.00025, for 25K steps on 24 A100s (~24 hours). 50% quantization dropout (HART style) is used to prevent vocabulary overfitting.
- **H-VAR Training**: Cross-entropy + $\mathcal{L}_{DPO}$ (equal weight). Initialized from official VAR d-16 checkpoint. 200 epochs, batch size 384, lr 1e-3, AdamW betas $(0.9, 0.95)$ on 24 A100s (~13 hours).
- **Data**: Standard DIV2K + DIV8K + Flickr2K + OST + 10K FFHQ. Real-ESRGAN degradation is used to synthesize LR-HR pairs.

## Key Experimental Results

### Main Results

| Dataset | Metric | StableSR | ResShift | VARSR (1B) | VARSR-d16 | H-VAR (310M, ours) |
|---|---|---|---|---|---|---|
| DIV2K-Val | LPIPS ↓ | 0.323 | 0.428 | 0.326 | 0.495 | **0.317** |
| DIV2K-Val | FID ↓ | 28.32 | 30.79 | 35.51 | 45.96 | **28.86** |
| RealSR | LPIPS ↓ | 0.300 | 0.346 | 0.350 | 0.413 | **0.256** |
| DRealSR | LPIPS ↓ | 0.333 | 0.401 | 0.354 | 0.409 | **0.259** |
| DRealSR | FID ↓ | 148.2 | 159.8 | 155.9 | 244.7 | **145.1** |

| Model | Params | FLOPs | Latency | DIV2K-Val FID (LPIPS) |
|---|---|---|---|---|
| H-VAR (Ours) | 310M | 0.921T | 0.25s | 28.86 (0.317) |
| VARSR | 1B | 3.071T | 0.93s | 35.51 (0.326) |
| ResShift | 173M | 2.651T | 0.17s | 30.79 (0.428) |
| StableSR | 919M | 79.94T | 5.51s | 28.32 (0.323) |

### Ablation Study

| Dataset | Configuration | PSNR@128 | PSNR@256 | PSNR@512 | LPIPS@512 |
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
- At intermediate scales (128/256), baselines without HIT are nearly unusable (LPIPS > 0.4). HIT reduces scores to the 0.2 range, proving it generates meaningful images at intermediate scales.
- HIT acts as a very strong inductive bias: reducing the transformer from 1B to 310M and using standard public data still yields FID/LPIPS scores that rival or exceed VARSR.
- DPO regularization consistently improves performance across all datasets and scales without an external reward model.
- **Side Effect**: Because early residuals are forced into low resolutions, the final 512 reconstruction suffers a slight degradation. Increasing $L$ from 10 to 11 compensates but increases inference cost by 24%.

## Highlights & Insights
- "Embedding multi-scale decodability into tokenization" is the core takeaway. By constraining the vocabulary upstream, the downstream model can be an order of magnitude smaller.
- Using the LR itself as a negative sample for DPO is a clever "self-supervised preference learning" trick that saves the cost of a reward model. This can be extended to any task with natural degradation (deblurring, denoising).
- The paper honestly discusses the trade-offs: HIT slightly impacts maximum resolution quality, which requires more token steps to fix.
- Generating three resolutions in a single forward pass is highly practical for real-world engineering (e.g., mobile previews).

## Limitations & Future Work
- Multi-scale is currently limited to 3 discrete scales. Arbitrary upscaling ($\times 1.5, \times 3$) would require redesigning $\rho_l$ allocation, reflecting the discrete nature of tokenization.
- DPO assumes LR is always the "worse answer." If the input is only lightly degraded, this preference might cause over-sharpening.
- Experiments focused on standard $\times 4$ SR. The efficiency of HIT at higher ratios ($\times 8/\times 16$) remains to be verified.
- Lacks direct comparison with strong Diffusion baselines like PASD or SUPIR.

## Related Work & Insights
- **vs VARSR**: Both use VAR for ISR, but VARSR uses original RQ-VAE, meaning intermediate scales are meaningless, and it requires 1B parameters + private data. H-VAR solves these with HIT and requires no ControlNet.
- **vs PURE**: PURE uses 7B Lumina-mGPT and embeds degradation descriptions into the vocabulary. H-VAR proves ISR doesn't necessarily require massive multimodal models if tokenization is well-designed.
- **vs Diffusion-based ISR**: Diffusion models are slower and lack sequence likelihoods for DPO. H-VAR’s AR format provides these advantages.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First multi-scale VAR ISR solution; clever use of LR for DPO.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple baselines and datasets; lacks comparison with Diffusion SOTA (PASD/SUPIR).
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear pseudo-code, diagrams, and honest discussion of limitations.
- **Value**: ⭐⭐⭐⭐ High practical value for deployment by achieving 1B performance with 310M parameters and multi-res output.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] HCF: Hierarchical Cascade Framework for Distributed Multi-Stage Image Compression](../../AAAI2026/model_compression/hcf_hierarchical_cascade_framework_for_distributed_multi-stage_image_compression.md)
- [\[ICML 2026\] Efficient Learned Image Compression without Entropy Coding](efficient_learned_image_compression_without_entropy_coding.md)
- [\[ICCV 2025\] Learned Image Compression with Hierarchical Progressive Context Modeling](../../ICCV2025/model_compression/learned_image_compression_with_hierarchical_progressive_context_modeling.md)
- [\[ICML 2026\] From Per-Image Low-Rank to Encoding Mismatch: Rethinking Feature Distillation in Vision Transformers](from_per-image_low-rank_to_encoding_mismatch_rethinking_feature_distillation_in_.md)
- [\[CVPR 2026\] Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning](../../CVPR2026/model_compression/towards_generalizable_ai-generated_image_detection_via_image-adaptive_prompt_lea.md)

</div>

<!-- RELATED:END -->
