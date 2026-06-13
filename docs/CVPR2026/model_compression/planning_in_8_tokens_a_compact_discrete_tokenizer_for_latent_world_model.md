---
title: >-
  [Paper Note] Planning in 8 Tokens: A Compact Discrete Tokenizer for Latent World Model
description: >-
  [CVPR2026][Model Compression][compact discrete tokenizer] This paper proposes CompACT, which compresses each image into only 8 discrete tokens (~128 bits) by leveraging a frozen pretrained visual encoder to retain planni…
tags:
  - "CVPR2026"
  - "Model Compression"
  - "compact discrete tokenizer"
  - "world model"
  - "latent space planning"
  - "extreme compression"
  - "semantic encoding"
  - "generative decoding"
date: 2026-05-08
content_hash: 060913b14b01e771
---

# Planning in 8 Tokens: A Compact Discrete Tokenizer for Latent World Model

**Conference**: CVPR2026
**arXiv**: [2603.05438](https://arxiv.org/abs/2603.05438)  
**Code**: [kdwonn/CompACT](https://kdwonn.github.io/CompACT)  
**Area**: Model Compression / World Models / Representation Learning
**Keywords**: compact discrete tokenizer, world model, latent space planning, extreme compression, semantic encoding, generative decoding

## TL;DR

This paper proposes CompACT, which compresses each image into only 8 discrete tokens (~128 bits) by leveraging a frozen pretrained visual encoder to retain planning-critical semantic information, while a generative decoder supplements perceptual details. This achieves approximately 40× speedup in world-model-based planning with no loss in accuracy.

## Background & Motivation

1. **Planning bottleneck in world models**: Existing world models (e.g., NWM) encode each frame into hundreds of tokens (SD-VAE requires 784 tokens), and the quadratic complexity of attention leads to planning latency as high as 3 minutes per episode, making real-time control infeasible.
2. **Token count determines computational cost**: MPC-based planning requires a large number of forward passes (~1920 rollouts), making token count a direct bottleneck for throughput.
3. **Reconstruction fidelity ≠ planning requirements**: Conventional tokenizers prioritize high-frequency details such as texture and lighting, whereas planning tasks require only high-level semantics such as spatial layout and object relationships.
4. **Iterative denoising overhead in diffusion models**: Continuous latent spaces require hundreds of denoising steps, further slowing planning.
5. **Limitations of existing compression methods**: Variable-length 1D tokenizers such as FlexTok still optimize for reconstruction fidelity rather than planning performance.
6. **Information-theoretic lower bound supports extreme compression**: From a mutual information perspective, the authors prove that the minimum entropy of a planning-sufficient representation is $H(\mathbf{a}^*)$, far smaller than $H(\mathbf{o})$, theoretically requiring only a few hundred bits.

## Method

### Overall Architecture

A three-stage pipeline: (a) train the CompACT tokenizer to map images to compact discrete tokens; (b) train an action-conditioned world model in the compact latent space; (c) at test time, perform latent-space planning via MPC (CEM optimization).

### Encoder: Semantic Encoding ($\mathcal{E}_\text{compact}$)

- **Frozen DINOv3-B** extracts semantic patch features without fine-tuning (fine-tuning degrades rFID from 2.40 to 5.22).
- **Latent Resampler**: $N$ learnable query tokens ($N=8$ or $16$) distill high-level semantics from DINOv3 outputs via cross-attention.
- **Finite Scalar Quantization (FSQ)**: levels set to $[8,8,8,5,5,5]$, yielding ~16 bits per token and 128–256 bits per frame in total.
- Core design rationale: the visual foundation model already abstracts away low-level details → cross-attention can only extract semantic information → planning-irrelevant information is naturally discarded.

### Decoder: Generative Decoding ($\mathcal{D}_\text{compact}$)

- Direct pixel reconstruction from 8/16 tokens is ill-posed due to insufficient information.
- MaskGIT VQGAN (196 tokens/frame) is used as the target tokenizer.
- Masked generative modeling is adopted: during training, target tokens are randomly masked and recovered conditioned on compact tokens.
- At inference, the process begins from a fully masked sequence and iteratively unmasks tokens by confidence, without iterative denoising.
- Training loss: $\mathcal{L}_{\text{tok}} = -\mathbb{E}[\log p(\mathbf{z}^\psi | \mathbf{z}, M(\mathbf{z}^\psi))]$

### World Model Training

- Navigation tasks: autoregressive DiT with a fixed-length history window and random history token masking (inspired by diffusion forcing).
- Robot manipulation tasks: block-causal transformer with parallel multi-frame prediction.
- World model loss: $\mathcal{L}_{\text{world}} = -\mathbb{E}[\log p(\mathbf{z}_{t+1} | \mathbf{z}_t, \mathbf{a}_t, M(\mathbf{z}_{t+1}))]$
- At planning time, CEM searches for the optimal action sequence; the cost function can be computed in pixel space (LPIPS) or latent space (L1).

## Key Experimental Results

### Reconstruction Quality (ImageNet Validation Set)

| Tokenizer | Type | #tok | rFID↓ | IS↑ |
|---|---|---|---|---|
| SD-VAE | Continuous | 1024 | 0.64 | 223.8 |
| MaskGIT-VQGAN | Discrete | 256 | 1.83 | 186.7 |
| FlexTok | Discrete | 16 | 5.60 | 114.9 |
| **CompACT** | **Discrete** | **16** | **2.40** | **209.0** |
| **CompACT** | **Discrete** | **8** | **3.21** | **207.5** |

### Navigation Planning (RECON Benchmark)

| Tokenizer | #tok | ATE↓ | RPE↓ | Latency (s)↓ |
|---|---|---|---|---|
| SD-VAE | 784 | 1.262 | 0.354 | 178.78 |
| FlexTok | 64 | 1.484 | 0.400 | 16.68 |
| FlexTok | 16 | 1.625 | 0.446 | 14.48 |
| **CompACT** | **16** | **1.330** | **0.390** | **5.78** |
| **CompACT** | **8** | **1.373** | **0.401** | **4.83** |

CompACT-16 is **~31×** faster than SD-VAE; CompACT-8 is **~37×** faster, with comparable accuracy.

### Ablation Study

- **Removing generative decoding** (single-step feedforward decoding): rFID degrades from 2.40 to 28.80.
- **Fine-tuning DINOv3**: rFID degrades to 5.22; planning ATE degrades from 1.330 to 1.472.
- **History masking**: disabling it degrades ATE from 1.330 to 1.480.
- **Latent-space cost function**: ATE slightly increases (1.379 vs. 1.330), but latency drops from 5.78s to 2.15s (~80× over SD-VAE overall).

### Action-Conditioned Video Prediction (RoboNet)

| Model | #tok | APE↓ | Latency (s)↓ |
|---|---|---|---|
| MaskGIT-VQGAN | 256 | 0.3383 | 3.826 |
| **CompACT** | **16** | **0.1122** | **0.740** |

APE is reduced by 3× and speed is improved by 5.2×, validating that compact tokens preserve action-relevant information.

## Highlights & Insights

1. **Extreme compression ratio**: 8 tokens / 128 bits per frame, nearly 100× compression over SD-VAE with no degradation in planning accuracy.
2. **Counterintuitive insight about frozen encoders**: Not fine-tuning DINOv3 yields better results, as fine-tuning causes semantic degradation.
3. **Information-theoretic grounding**: A formal mutual information argument proves that the minimum information required for planning-sufficient representations is far below the entropy of the full observation.
4. **Modular token attention**: Each token spontaneously attends to semantically consistent regions (objects, end-effectors, etc.) without explicit supervision.
5. **Cross-backbone generalizability**: The method is effective with SigLIP-2, MAE, and DINOv3, and is not tied to any specific visual foundation model.

## Limitations & Future Work

1. **Closed-loop manipulation is only preliminarily validated**: Experiments are limited to RoboMimic Lift (56% success rate), without evaluation on complex contact-rich or long-horizon tasks.
2. **Decoder quality is bounded by MaskGIT VQGAN**: The upper limit of reconstruction quality is constrained by the target tokenizer.
3. **Not validated on large-scale autonomous driving or game environments**: Evaluations cover only indoor navigation and tabletop manipulation.
4. **No closed-loop IDM integration**: The IDM does not leverage real-time observations or proprioceptive information.
5. **Uncontrolled information loss from discrete quantization**: FSQ level selection is currently empirical and lacks an adaptive mechanism.

## Related Work & Insights

| Method | Characteristics | Difference from CompACT |
|---|---|---|
| NWM (Bar et al.) | SD-VAE 784 tokens + diffusion world model | Latency ~180s; CompACT is 40× faster |
| FlexTok | Variable-length 1D tokenizer | Optimized for reconstruction, not planning; rFID 5.60 vs. 2.40 at 16 tokens |
| IRIS / iVideoGPT | Discrete tokens + conditional compression | Relies on previous frame conditioning; unsuitable for long-horizon planning |
| DINO-WM | DINO features as world model | Still uses many tokens; no extreme compression |
| TA-TiTok | 32-token compact tokenizer | Not optimized for planning; no world model validation |

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of extreme compression, frozen encoder, and generative decoding is novel, with theoretically grounded information-theoretic analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Reconstruction, planning, video prediction, and ablations are all covered, though closed-loop manipulation validation is relatively limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The motivation–method–experiment chain is clear and well-structured; Proposition 1 and ablation designs are concise and precise.
- **Value**: ⭐⭐⭐⭐ — Provides a practical pathway toward real-time world-model-based planning; the 40× speedup is of significant engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] WPT: World-to-Policy Transfer via Online World Model Distillation](wpt_world-to-policy_transfer_via_online_world_model_distillation.md)
- [\[ICCV 2025\] Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation](../../ICCV2025/model_compression/bridging_continuous_and_discrete_tokens_for_autoregressive_visual_generation.md)
- [\[CVPR 2026\] Generative Video Compression with One-Dimensional Latent Representation](generative_video_compression_with_one-dimensional_latent_representation.md)
- [\[AAAI 2026\] CTPD: Cross Tokenizer Preference Distillation](../../AAAI2026/model_compression/ctpd_cross_tokenizer_preference_distillation.md)
- [\[ICLR 2026\] UniFlow: A Unified Pixel Flow Tokenizer for Visual Understanding and Generation](../../ICLR2026/model_compression/uniflow_a_unified_pixel_flow_tokenizer_for_visual_understanding_and_generation.md)

</div>

<!-- RELATED:END -->
