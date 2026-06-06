---
title: >-
  [Paper Note] CLEAR: Context-Aware Learning with End-to-End Mask-Free Inference for Adaptive Video Subtitle Removal
description: >-
  [ICML 2026][Image Generation][Video Subtitle Removal] This paper proposes CLEAR for video subtitle removal: a two-stage training approach (Stage I uses a dual encoder + orthogonal decoupling to learn self-supervised subt…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Video Subtitle Removal"
  - "Diffusion Models"
  - "LoRA"
  - "Self-supervised Prior"
  - "Mask-free Inference"
date: 2026-05-08
content_hash: d3aacfa98d1e021d
---

# CLEAR: Context-Aware Learning with End-to-End Mask-Free Inference for Adaptive Video Subtitle Removal

**Conference**: ICML 2026  
**arXiv**: [2603.21901](https://arxiv.org/abs/2603.21901)  
**Code**: https://github.com/silent-commit/CLEAR (Available)  
**Area**: Video Generation / Video Inpainting / Subtitle Removal  
**Keywords**: Video Subtitle Removal, Diffusion Models, LoRA, Self-supervised Prior, Mask-free Inference

## TL;DR
This paper proposes CLEAR for video subtitle removal: a two-stage training approach (Stage I uses a dual encoder + orthogonal decoupling to learn self-supervised subtitle prior masks; Stage II applies LoRA to the Wan2.1 video diffusion model + an occlusion head for adaptive weighting). Inference requires no masks or text detectors. By training only 0.77% of parameters, it achieves a PSNR of 26.80 dB on Chinese test sets (+6.77 dB over the strongest baseline) and demonstrates zero-shot generalization across 6 languages.

## Background & Motivation
**Background**: Current video subtitle removal mainly relies on mask-guided video diffusion inpainting (e.g., DiffuEraser, EraserDiT, MiniMax-Remover), which depends on external text detection or segmentation to provide precise binary masks for every frame as a condition.

**Limitations of Prior Work**: (L1) Low training efficiency—full-parameter training plus frame-by-frame mask annotation, where annotation requires manual labor or specialized models for long videos; (L2) Fragile inference—requires constant text detection/tracking during deployment; failure leads to flickering, artifacts, or drift; (L3) Static prior utilization—auxiliary priors (heatmaps, optical flow) are weighted uniformly, ignoring the reliability differences of subtitles across frames and regions.

**Key Challenge**: Video subtitles exhibit temporal continuity, diverse positions/fonts, and complex coupling with camera/object motion. This requires (K1) parameter efficiency without mask annotation, (K2) completely mask-free end-to-end inference, and (K3) adaptive balancing of prior quality. Existing methods fail to address all three.

**Goal**: To construct a framework that learns subtitle priors **self-supervised** from subtitled/clean video pairs during training, enables **completely mask-free** inference, and adaptively weights subtitle regions.

**Key Insight**: Utilize the pixel difference between "subtitled frames" and "clean frames" as a weakly supervised pseudo-label (noisy but cheap). Isolate subtitle information via a dual-encoder with orthogonal constraints, then allow the diffusion model to use an occlusion head to calibrate priors during generation.

**Core Idea**: Explicitly distill the "subtitle mask recognition" capability into the intermediate layers of a LoRA-fine-tuned DiT during training. This allows inference to proceed by simply inputting the subtitled video—implicitly generating $\mathcal{M}^{pred}$ internally while remaining externally mask-free.

## Method

### Overall Architecture
A two-stage pipeline. Stage I (Self-supervised Prior): A dual ResNet-50 encoder ($E_{\text{sub}}, E_{\text{content}}$) and a 4-layer UNet decoder are trained using pixel-difference pseudo-labels to obtain the prior mask $\mathcal{M}^{prior}$. Inputs use ImageNet pre-training, while orthogonal loss and an adversarial discriminator decouple subtitle and content features. Stage II (Adaptive Weighting): The Wan2.1-Fun-V1.1-1.3B DiT is frozen, and LoRA (rank=64) is injected into all attention and FFN layers. A 2.1M parameter occlusion head $\mathcal{H}$ calculates $\mathcal{M}^{pred}$ from DiT intermediate layers. The diffusion loss is modulated by a spatial emphasis × focal difficulty weight $w_{i,j,t}$. Joint optimization involves three losses: distillation, context-aware adaptation, and sparsity. Inference: Single input video → DiT + LoRA + internal $\mathcal{M}^{pred}$ → 5-step DDIM → VAE decoder → clean video, with no external modules.

### Key Designs

1.  **Stage I Self-supervised Subtitle Prior (Dual Encoder + Orthogonal Decoupling + Adversarial Discriminator)**:
    - **Function**: Learns a binary mask $\mathcal{M}^{prior}$ predicting subtitle regions from 500 video pairs without manual masks.
    - **Mechanism**: (a) Generates pseudo-labels using pixel differences $\Delta_t = \|\mathbf{X}^{sub}_t - \mathbf{X}^{clean}_t\|_2$ with per-frame mean+std thresholds; (b) Dual encoders extract $F^{sub}$ and $F^{content}$ at 1/8 resolution; (c) Orthogonal loss $\mathcal{L}_{\text{ortho}} = \frac{1}{T H' W'} \sum \langle F^{sub}, F^{content} \rangle^2$ enforces independence; (d) Adversarial loss $\mathcal{L}_{\text{adv}}$ prevents leakage; (e) The decoder outputs $\mathcal{M}^{prior}$ from $F^{sub}$, while $F^{content}$ must reconstruct the clean frame.
    - **Design Motivation**: Pixel-difference pseudo-labels are noisy (due to lighting, translucent subtitles, motion blur); BCE alone cannot learn good masks. Triple constraints (orthogonality + adversarial + reconstruction) force subtitle features to carry all difference information, allowing the mask head to generalize to unseen fonts/languages rather than memorizing token shapes.

2.  **Stage II Context-Aware Occlusion Head + Adaptive Weighting $w_{i,j,t}$**:
    - **Function**: Dynamically calculates subtitle probabilities for each patch in DiT intermediate layers and adjusts loss weights accordingly to "explicitly attend to subtitles during training and implicitly erase during generation."
    - **Mechanism**: The occlusion head $\mathcal{H}(\mathbf{h}_{enc}) = \mathrm{Conv}^1_{1 \times 1}(\mathrm{SiLU}(\mathrm{Conv}^{64}_{3 \times 3}(\mathbf{h}_{enc})))$ computes $\mathcal{M}^{pred} = \sigma(\mathcal{H}(\mathbf{h}_{enc}))$ from DiT encoder activations. The final weight is $w_{i,j,t} = (1 + \alpha(k) \cdot \mathcal{M}^{pred}_{i,j,t}) \cdot (\epsilon^{gen}_{i,j,t} + \delta)^\gamma$. The first part is spatial emphasis (weighting predicted subtitle regions), and the second is focal-style difficulty weighting (weighting high reconstruction error regions). $\alpha(k)$ oscillates between $\alpha_{\min}=5$ and $\alpha_{\max}=15$ via triangular scheduling to avoid local optima.
    - **Design Motivation**: Naive solutions use the prior directly as a mask condition, but Stage I priors are noisy. By allowing the head to observe latent noise, high-level DiT semantics, and diffusion step $t$, "prior calibration" becomes an "in-generation difficulty assessment." Focal weighting (inspired by RetinaNet) reduces gradients from simple backgrounds and increases them for hard subtitle areas. Crucially, $\mathcal{M}^{pred}$ is not detached; gradients flow back from $\mathcal{L}_{\text{gen}}$ for self-correction.

3.  **Joint Loss Optimization + Internalized Mask-Free Inference**:
    - **Function**: Uses distillation (from Stage I), generation feedback, and sparsity/KL to optimize LoRA and the occlusion head, ensuring $\mathcal{M}^{pred}$ retains prior structure while correcting local errors.
    - **Mechanism**: $\mathcal{L}_{stage2} = \mathcal{L}_{distill} + \mathcal{L}_{gen} + 0.1 \cdot \mathcal{L}_{sparse}$. $\mathcal{L}_{distill}$ uses SmoothL1 such that $\mathcal{M}^{pred} \approx \mathcal{M}^{prior}$ with a 1-unit margin. $\mathcal{L}_{gen}$ is the standard diffusion $\epsilon$-loss weighted by $w$. $\mathcal{L}_{sparse}$ combines L1 sparsity with $D_{KL}(\mathcal{M}^{pred} \| \mathcal{M}^{prior})$ to prevent uniform degradation and prior drift.
    - **Design Motivation**: Pure distillation amplifies Stage I noise, while pure generation feedback leads to trivial head outputs (e.g., all zeros). The three losses balance structure, quality, and controllability. Once trained, the knowledge of "which regions to erase" is absorbed into the attention patterns, making $\mathcal{M}^{pred}$ an internal variable never required as an external output during single-forward inference (Alg. 1).

### Loss & Training
Stage I: $\mathcal{L}_{stage1} = \mathcal{L}_{\text{ortho}} + 0.5\mathcal{L}_{\text{adv}} + \mathcal{L}_{\text{region}} + 0.1\mathcal{L}_{\text{recon}}$, AdamW lr=$2 \times 10^{-5}$, 1 epoch (~70 min). Stage II: $\mathcal{L}_{stage2}$ as above, AdamW lr=$1 \times 10^{-4}$, gradient clipping=1.0, 1 epoch ≈ 1 day (8×A800). LoRA rank=64 applied to q, k, v, o and ffn.0/2; $\gamma=0.8, \delta=10^{-6}$. Stage II data: 500 videos × 81 consecutive frames.

## Key Experimental Results

### Main Results (Chinese Subtitle Test Set, 400 Samples)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | VFID↓ | TWE↓ | Flow Var↓ | s/frame↓ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ProPainter | 17.24 | 0.658 | 0.329 | 98.46 | 1.286 | 0.885 | 2.36 |
| MiniMax-Remover | 20.03 | 0.773 | 0.166 | 95.39 | 4.222 | 0.415 | 4.90 |
| DiffuEraser | 17.85 | 0.672 | 0.458 | 72.51 | 1.523 | 0.630 | 3.47 |
| **CLEAR (mask-free)** | **26.80** | **0.894** | **0.101** | **20.37** | **1.227** | **0.029** | 4.86 |

PSNR +6.77 dB, VFID -74.7%, Flow Variance -93.0%. All baselines require external masks, while CLEAR inputs only the subtitle video itself.

### Ablation Study

| Configuration | PSNR↑ | VFID↓ | TWE↓ |
| :--- | :--- | :--- | :--- |
| Baseline (LoRA-only) | 21.62 | 34.74 | 1.320 |
| + M1: Stage I prior + focal weighting | 23.11 | 38.21 | 1.303 |
| + M2: Context Distillation | 24.72 | 31.73 | 1.279 |
| + M3: Context-Aware Adaptation | 25.09 | 31.56 | 1.257 |
| + M4: Context Consistency (CLEAR) | **26.80** | **20.37** | **1.227** |

| Inference setting | PSNR↑ | VFID↓ | s/frame |
| :--- | :--- | :--- | :--- |
| steps=5 (default) | 26.80 | 20.37 | 4.86 |
| steps=10 | 29.43 | 35.70 | 9.92 |
| cfg=1.2 | 29.65 | 40.71 | 4.86 |
| lora_scale=0.5 | 25.17 | 63.02 | 4.86 |
| lora_scale=1.5 | 27.94 | 42.16 | 4.86 |

### Key Findings
- Cumulative 5.18 dB PSNR gain comes from the four modules combined. Consistency regularization (M4) provides the largest VFID reduction (-35.5%), indicating that preventing $\mathcal{M}^{pred}$ degradation is critical for perceptual quality.
- steps=10 yields higher PSNR but worse VFID (35.70 vs 20.37), suggesting that more denoising steps introduce artifacts; 5 steps is the optimal default.
- LoRA scale 0.5 leads to severe under-erasing (LPIPS +82%), while 1.5 causes over-smoothing; 1.0 is the "sweet spot." CFG=1.0 balances fidelity and perception.
- Zero-shot cross-lingual: Trained only on Chinese subtitles, CLEAR successfully erases English, Korean, French, Japanese, Russian, and German, verifying it learns abstract occlusion patterns rather than character features.

## Highlights & Insights
- **Pixel-difference Pseudo-labels + Orthogonal Decoupling**: Replaces expensive mask annotation with pixel differences from subtitled/clean video pairs. Orthogonal and adversarial constraints effectively isolate subtitle information into a single encoder. This self-supervised process allows learning generalized subtitle priors from just 500 video pairs.
- **Gradient Flow through $\mathcal{M}^{pred}$ for Self-correction**: Unlike many attention/mask methods that detach masks to prevent interference, this work purposefully allows diffusion loss gradients to flow back to the head. High reconstruction error zones receive positive gradients to raise $\mathcal{M}^{pred}$ weights, while low error zones receive negative gradients, forming a feedback loop without GT masks.
- **High Engineering Value of Mask-free Inference**: Removing dependencies on text detection/segmentation eliminates a fragile sub-pipeline (OCR misses, tracking drift). Combined with 0.77% trainable parameters and single-epoch training, it is highly suitable for actual deployment. The end-to-end mapping from subtitled to clean video also avoids cascading errors.

## Limitations & Future Work
- Main experiments used only Chinese subtitle training data (160K pairs, 400 test); other languages were only qualitatively visualized without quantitative generalization metrics.
- 5-step DDIM at 1280×720 resolution takes 4.86 s/frame, which is far from real-time. The authors mention "real-time inference optimization" but provide no specific solution.
- Dependence on Wan2.1-Fun-V1.1-1.3B as the backbone; transferability to other video diffusion models (e.g., HunyuanVideo, Sora-style) is not yet verified.
- The effectiveness of self-supervised priors on "animated subtitles, artistic fonts, or extremely translucent subtitles" requires more stress testing. M1 contributed +1.49 dB but increased VFID (38.21), implying the prior itself remains noisy and requires the triple-loss system to remain robust.

## Related Work & Insights
- **vs. DiffuEraser / EraserDiT / MiniMax-Remover**: These require external masks and full-parameter training. CLEAR achieves +6.77 dB PSNR with mask-free inference and 0.77% parameters, decoupling training labels, inference dependencies, and parameter efficiency.
- **vs. ProPainter**: Traditional optical flow propagation lags significantly (PSNR 17.24), demonstrating that modern video diffusion priors are essential for high-frequency local occlusions like subtitles.
- **vs. Image-based STR (EraseNet/ViTEraser)**: Image methods lack temporal consistency. CLEAR's Flow Variance (0.029, 33× lower than ProPainter's 0.885) proves the temporal stability advantages of end-to-end video diffusion with LoRA.
- **Transferable Ideas**: The self-supervised prior workflow (dual encoder + orthogonal decoupling + pixel-difference pseudo-labels) can be directly applied to other local occlusion removal tasks (watermarks, logo erasure, video censorship repair). Focal-weighted diffusion loss is also reusable in general inpainting scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ Self-supervised orthogonal decoupling + gradient-flow occlusion head + mask-free inference is a novel combination for subtitle removal, though individual components (LoRA, self-supervised priors) are established.
- Experimental Thoroughness: ⭐⭐⭐⭐ Strong numerical gains across PSNR/VFID/temporal flow, step-by-step ablation, and hyperparameter analysis, though quantitative cross-lingual data is lacking.
- Writing Quality: ⭐⭐⭐⭐ Clear correspondence between limitations (L1-L3) and capabilities (K1-K3), with well-organized method diagrams, algorithm boxes, and tables.
- Value: ⭐⭐⭐⭐ Effectively addresses the "mandated mask" pain point in video subtitle removal deployment. 0.77% parameters + mask-free inference offer significant product value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer](end-to-end_autoregressive_image_generation_with_1d_semantic_tokenizer.md)
- [\[ICML 2026\] AdaEraser: Training-Free Object Removal via Adaptive Attention Suppression](adaeraser_training-free_object_removal_via_adaptive_attention_suppression.md)
- [\[ICCV 2025\] End-to-End Multi-Modal Diffusion Mamba](../../ICCV2025/image_generation/end-to-end_multi-modal_diffusion_mamba.md)
- [\[NeurIPS 2025\] LinEAS: End-to-end Learning of Activation Steering with a Distributional Loss](../../NeurIPS2025/image_generation/lineas_end-to-end_learning_of_activation_steering_with_a_distributional_loss.md)
- [\[CVPR 2026\] Precise Object and Effect Removal with Adaptive Target-Aware Attention](../../CVPR2026/image_generation/precise_object_and_effect_removal_with_adaptive_target-aware_attention.md)

</div>

<!-- RELATED:END -->
