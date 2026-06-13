---
title: >-
  [Paper Note] CLEAR: Context-Aware Learning with End-to-End Mask-Free Inference for Adaptive Video Subtitle Removal
description: >-
  [ICML 2026][Image Generation][Video subtitle removal] This paper proposes CLEAR for video subtitle removal: a two-stage training pipeline (Stage I uses a dual encoder + orthogonal decoupling to self-supervise a subtitle…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Video subtitle removal"
  - "diffusion model"
  - "LoRA"
  - "self-supervised prior"
  - "mask-free inference"
date: 2026-05-08
content_hash: d935371951785052
---

# CLEAR: Context-Aware Learning with End-to-End Mask-Free Inference for Adaptive Video Subtitle Removal

**Conference**: ICML 2026  
**arXiv**: [2603.21901](https://arxiv.org/abs/2603.21901)  
**Code**: https://github.com/silent-commit/CLEAR (available)  
**Area**: Video Generation / Video Inpainting / Subtitle Removal  
**Keywords**: Video subtitle removal, diffusion model, LoRA, self-supervised prior, mask-free inference

## TL;DR
This paper proposes CLEAR for video subtitle removal: a two-stage training pipeline (Stage I uses a dual encoder + orthogonal decoupling to self-supervise a subtitle prior mask; Stage II adds LoRA + an occlusion head to the Wan2.1 video diffusion model for adaptive weighting). Inference requires no mask or text detector at all; with only 0.77% trainable parameters, PSNR reaches 26.80 dB on a Chinese test set (+6.77 dB over the strongest baseline), and zero-shot generalizes to six languages.

## Background & Motivation
**Background**: Current video subtitle removal mainly relies on mask-guided video diffusion inpainting (DiffuEraser, EraserDiT, MiniMax-Remover), requiring external text detection/segmentation to provide precise binary masks for every frame.

**Limitations of Prior Work**: (L1) Low training efficiency—full-parameter training and per-frame mask annotation, which itself depends on manual labeling or dedicated segmentation models throughout long videos; (L2) Fragile inference—text detection/tracking must run continuously after deployment, and any detection failure leads to flicker, ghosting, or drift; (L3) Static prior usage—auxiliary priors (heatmap, optical flow) are used with uniform weighting, ignoring reliability differences of subtitles across frames and regions.

**Key Challenge**: Video subtitles exhibit temporal continuity, diverse positions/fonts, and complex coupling with camera/object motion, requiring (K1) parameter efficiency + no mask annotation, (K2) fully mask-free end-to-end inference, and (K3) adaptive prior quality weighting; none of which are achieved by existing methods.

**Goal**: Construct a framework that can self-supervise subtitle priors from paired subtitle/clean videos during training, is fully mask-free during inference, and dynamically adapts weighting for subtitle regions.

**Key Insight**: Use the pixel difference between "subtitle frame - clean frame" as a weakly supervised pseudo-label (noisy but cheap), and employ a dual encoder + orthogonal constraint to isolate subtitle information; then let the diffusion model use an occlusion head to correct the prior during generation.

**Core Idea**: Explicitly distill the "subtitle mask recognition" capability into the LoRA-tuned DiT intermediate layers during training, so that inference only requires the subtitle video as input—the model internally generates $\mathcal{M}^{pred}$, with no external mask needed.

## Method

### Overall Architecture
A two-stage pipeline. Stage I (self-supervised prior): Use pixel-difference pseudo-labels to train dual ResNet-50 encoders ($E_{\text{sub}},E_{\text{content}}$) + a 4-layer UNet decoder to obtain the prior mask $\mathcal{M}^{prior}$, with ImageNet pretraining and orthogonal loss + adversarial discriminator to decouple subtitle and content features. Stage II (adaptive weighting): Freeze Wan2.1-Fun-V1.1-1.3B DiT, inject rank=64 LoRA into all attention + FFN, and add a 2.1M parameter occlusion head $\mathcal{H}$ to compute $\mathcal{M}^{pred}$ from DiT intermediate layers. The spatial emphasis × focal difficulty weight $w_{i,j,t}$ modulates the diffusion loss, with three losses (distillation + context-aware adaptation + sparsity) jointly optimized. Inference: single input video → DiT + LoRA + internal $\mathcal{M}^{pred}$ → DDIM 5 steps → VAE decodes clean video, with no external modules.

### Key Designs

1. **Stage I Self-Supervised Subtitle Prior (dual encoder + orthogonal decoupling + adversarial discriminator)**:

    - **Function**: Learns a binary mask $\mathcal{M}^{prior}$ predicting subtitle regions from 500 video pairs without manual masks.
    - **Mechanism**: (a) Use pixel difference $\Delta_t=\|\mathbf{X}^{sub}_t-\mathbf{X}^{clean}_t\|_2$ and per-frame mean+std thresholding to generate pseudo-labels; (b) dual encoder extracts $F^{sub}, F^{content}$ at 1/8 resolution; (c) orthogonal loss $\mathcal{L}_{\text{ortho}}=\frac{1}{T H' W'}\sum\langle F^{sub}, F^{content}\rangle^2$ enforces independence; (d) adversarial loss $\mathcal{L}_{\text{adv}}$ prevents leakage; (e) decoder outputs $\mathcal{M}^{prior}$ from $F^{sub}$ only, and $F^{content}$ alone must reconstruct the clean frame.
    - **Design Motivation**: Pixel-difference pseudo-labels are noisy (lighting, semi-transparent subtitles, motion blur), and BCE alone cannot learn good masks; orthogonal + adversarial + reconstruction constraints force subtitle features to exclusively carry all difference information, enabling the mask head to generalize to unseen fonts/languages rather than memorizing specific token shapes.

2. **Stage II Context-Aware Occlusion Head + Adaptive Weighting $w_{i,j,t}$**:

    - **Function**: Dynamically computes subtitle probability for each patch in DiT intermediate layers, adjusting its weight in the diffusion loss to "explicitly attend to subtitles during training, implicitly erase during generation."
    - **Mechanism**: Occlusion head $\mathcal{H}(\mathbf{h}_{enc})=\mathrm{Conv}^1_{1\times 1}(\mathrm{SiLU}(\mathrm{Conv}^{64}_{3\times 3}(\mathbf{h}_{enc})))$ computes $\mathcal{M}^{pred}=\sigma(\mathcal{H}(\mathbf{h}_{enc}))$ from DiT encoder activations; final weight $w_{i,j,t}=(1+\alpha(k)\cdot\mathcal{M}^{pred}_{i,j,t})\cdot(\epsilon^{gen}_{i,j,t}+\delta)^\gamma$, where the first term is spatial emphasis (upweighting predicted subtitle regions), and the second is focal-style difficulty weighting (upweighting high reconstruction error regions); $\alpha(k)$ oscillates between $\alpha_{\min}=5,\alpha_{\max}=15$ with triangular scheduling to avoid local minima.
    - **Design Motivation**: Naively using the prior as a mask condition is noisy; letting the head see latent noise, DiT high-level semantics, and diffusion timestep $t$ turns "prior calibration" into "difficulty-aware generation." Focal weighting (from RetinaNet) ensures simple background regions contribute less gradient, while hard subtitle regions contribute more. Crucially, $\mathcal{M}^{pred}$ is not detached, so gradients from $\mathcal{L}_{\text{gen}}$ flow back, forming a self-correcting loop.

3. **Joint Three-Loss Optimization + Internalized Mask-Free Inference**:

    - **Function**: Uses distillation (from Stage I prior) + generation feedback (quality) + sparsity/KL (anti-degeneration) to jointly optimize LoRA and the occlusion head, so that $\mathcal{M}^{pred}$ retains prior structure while correcting local errors, and is absorbed into LoRA-augmented attention—no external mask needed at inference.
    - **Mechanism**: $\mathcal{L}_{stage2}=\mathcal{L}_{distill}+\mathcal{L}_{gen}+0.1\cdot\mathcal{L}_{sparse}$; $\mathcal{L}_{distill}$ uses SmoothL1 to enforce $\mathcal{M}^{pred}\approx\mathcal{M}^{prior}$ within 1 unit deviation; $\mathcal{L}_{gen}$ is standard diffusion $\epsilon$ loss weighted by $w$; $\mathcal{L}_{sparse}$ combines L1 sparsity + $D_{KL}(\mathcal{M}^{pred}\|\mathcal{M}^{prior})$, the former prevents uniform degeneration, the latter prevents drifting from the prior distribution.
    - **Design Motivation**: Pure distillation amplifies Stage I noise; pure generation feedback leads to trivial head outputs (all 0 or uniform). The three losses each serve a role: distill for structure, gen for quality, sparse for controllability. After training, LoRA + head absorb "which regions to erase" into attention patterns; at inference (Alg.1), $\mathcal{M}^{pred}$ is internal, never output, and a single forward pass yields the clean video.

### Loss & Training
Stage I: $\mathcal{L}_{stage1}=\mathcal{L}_{ortho}+0.5\mathcal{L}_{adv}+\mathcal{L}_{region}+0.1\mathcal{L}_{recon}$, AdamW lr=$2\times 10^{-5}$, 1 epoch (~70 min). Stage II: above $\mathcal{L}_{stage2}$, AdamW lr=$1\times 10^{-4}$, gradient clipping=1.0, 1 epoch ≈ 1 day (8×A800). LoRA rank=64, applied to q,k,v,o and ffn.0/2; $\gamma=0.8,\delta=10^{-6}$; Stage II data: 500 videos × 81 consecutive frames.

## Key Experimental Results

### Main Results (Chinese Subtitle Test Set, 400 Samples)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | VFID↓ | TWE↓ | Flow Var↓ | s/frame↓ |
|--------|-------|-------|--------|-------|------|-----------|---------|
| ProPainter | 17.24 | 0.658 | 0.329 | 98.46 | 1.286 | 0.885 | 2.36 |
| MiniMax-Remover | 20.03 | 0.773 | 0.166 | 95.39 | 4.222 | 0.415 | 4.90 |
| DiffuEraser | 17.85 | 0.672 | 0.458 | 72.51 | 1.523 | 0.630 | 3.47 |
| **CLEAR (mask-free)** | **26.80** | **0.894** | **0.101** | **20.37** | **1.227** | **0.029** | 4.86 |

PSNR +6.77 dB, VFID -74.7%, Flow Variance -93.0%; all baselines require external masks, while CLEAR only takes the subtitle video as input.

### Ablation Study

| Configuration | PSNR↑ | VFID↓ | TWE↓ |
|---------------|-------|-------|------|
| Baseline (LoRA-only) | 21.62 | 34.74 | 1.320 |
| + M1: Stage I prior + focal weighting | 23.11 | 38.21 | 1.303 |
| + M2: Context Distillation | 24.72 | 31.73 | 1.279 |
| + M3: Context-Aware Adaptation | 25.09 | 31.56 | 1.257 |
| + M4: Context Consistency (CLEAR) | **26.80** | **20.37** | **1.227** |

| Inference setting | PSNR↑ | VFID↓ | s/frame |
|-------------------|-------|-------|---------|
| steps=5 (default) | 26.80 | 20.37 | 4.86 |
| steps=10 | 29.43 | 35.70 | 9.92 |
| cfg=1.2 | 29.65 | 40.71 | 4.86 |
| lora_scale=0.5 | 25.17 | 63.02 | 4.86 |
| lora_scale=1.5 | 27.94 | 42.16 | 4.86 |

### Key Findings
- The cumulative 5.18 dB PSNR gain comes from the four modules in combination; consistency regularization (M4) alone provides the largest VFID drop (-35.5%), indicating that "preventing $\mathcal{M}^{pred}$ degeneration" is critical for perceptual quality.
- steps=10 yields higher PSNR but worse VFID (35.70 vs 20.37), suggesting more denoising steps introduce artifacts; 5 steps is optimal by default.
- LoRA scale 0.5 leads to severe under-removal (LPIPS +82%), 1.5 to over-smoothing—1.0 is the sweet spot; CFG=1.0 balances fidelity and perceptual quality.
- Zero-shot cross-lingual: trained only on Chinese subtitles, the model cleanly removes English/Korean/French/Japanese/Russian/German subtitles—demonstrating that it learns abstract occlusion patterns rather than character features.

## Highlights & Insights
- **Pixel-difference pseudo-labels + orthogonal decoupling**: Replaces expensive mask annotation with pixel differences from subtitle/clean video pairs, and uses orthogonal + adversarial constraints to force subtitle information into a single encoder. This self-supervised process enables learning a subtitle prior that generalizes to six languages from just 500 video pairs, exemplifying data efficiency.
- **Gradient flow through $\mathcal{M}^{pred}$ enables self-correction**: Many attention/mask weighting methods detach the mask to avoid interfering with the main task; this work does the opposite—deliberately allowing diffusion loss gradients to flow back to the head, so "high reconstruction error regions" receive positive gradients to raise $\mathcal{M}^{pred}$, and "low error regions" receive negative gradients to lower the weight, forming a feedback loop without GT masks.
- **Mask-free inference has high engineering value**: Eliminating text detection/segmentation removes an entire fragile sub-pipeline (no more OCR misdetection or tracking drift), and 0.77% trainable parameters + single-epoch training is deployment-friendly; end-to-end sub→clean mapping in one inference avoids cascading errors.

## Limitations & Future Work
- Main experiments are only on Chinese subtitle training data (160K training pairs, 400 test), with other languages shown only qualitatively; quantitative generalization is unmeasured.
- 5-step DDIM takes 4.86 s/frame at 1280×720 resolution, still short of real-time; authors mention "real-time inference optimization" but provide no solution.
- Relies on Wan2.1-Fun-V1.1-1.3B as backbone; transferability to other video diffusion models (HunyuanVideo, Sora series) is unverified.
- Whether the self-supervised prior remains effective for "animated subtitles, artistic fonts, extreme semi-transparency" needs more stress testing; M1 only contributes +1.49 dB but VFID increases (38.21), suggesting the prior itself is still noisy and the three-loss system is necessary for robustness.

## Related Work & Insights
- **vs DiffuEraser / EraserDiT / MiniMax-Remover**: All require external masks + full-parameter training; this work achieves mask-free inference with 0.77% parameters and +6.77 dB PSNR, decoupling "training annotation, inference dependency, and parameter efficiency."
- **vs ProPainter**: Traditional optical flow propagation method, PSNR 17.24 lags far behind; shows that modern video diffusion priors are essential for "high-frequency local occlusion" like subtitles.
- **vs Image-based STR (EraseNet/ViTEraser)**: Image methods lack temporal consistency constraints; CLEAR's Flow Variance 0.029 (33× lower than ProPainter's 0.885) demonstrates the temporal stability advantage of end-to-end video diffusion + LoRA.
- **Transferable idea**: The dual encoder + orthogonal decoupling + pixel-difference pseudo-label self-supervised prior can be directly applied to other "local occlusion removal" tasks (watermark removal, logo erasure, video censor repair); focal-weighted diffusion loss is also worth reusing in general inpainting scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of self-supervised orthogonal decoupling + gradient-flow occlusion head + fully mask-free inference is new for video subtitle removal, though individual techniques (LoRA, self-supervised prior) are relatively mature.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-metric (PSNR/VFID/temporal/flow) comparison, four-module ablation, and inference hyperparameter analysis; cross-lingual part lacks quantitative results.
- Writing Quality: ⭐⭐⭐⭐ Three limitations (L1-L3) correspond to three capabilities (K1-K3); method diagrams, algorithm boxes, and tables are clearly organized.
- Value: ⭐⭐⭐⭐ Truly solves the "must have mask" pain point in video subtitle removal deployment; 0.77% parameters + mask-free inference is highly valuable for product-level applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer](end-to-end_autoregressive_image_generation_with_1d_semantic_tokenizer.md)
- [\[ICML 2026\] AdaEraser: Training-Free Object Removal via Adaptive Attention Suppression](adaeraser_training-free_object_removal_via_adaptive_attention_suppression.md)
- [\[NeurIPS 2025\] LinEAS: End-to-end Learning of Activation Steering with a Distributional Loss](../../NeurIPS2025/image_generation/lineas_end-to-end_learning_of_activation_steering_with_a_distributional_loss.md)
- [\[CVPR 2026\] DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation](../../CVPR2026/image_generation/deco_frequency-decoupled_pixel_diffusion_for_end-to-end_image_generation.md)
- [\[ICCV 2025\] End-to-End Multi-Modal Diffusion Mamba](../../ICCV2025/image_generation/end-to-end_multi-modal_diffusion_mamba.md)

</div>

<!-- RELATED:END -->
