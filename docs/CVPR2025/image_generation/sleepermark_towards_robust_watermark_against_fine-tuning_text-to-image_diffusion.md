---
title: >-
  [Paper Note] SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-Image Diffusion Models
description: >-
  [CVPR 2025][Image Generation][Diffusion Model Watermarking] SleeperMark proposes a robust watermarking framework for T2I diffusion models. By explicitly decoupling watermarking information from the model's semantic knowledge, the watermark remains reliably detectable even after downstream fine-tuning (such as LoRA, DreamBooth, and ControlNet), maintaining a TPR@10⁻⁶FPR of over 0.93 under various fine-tuning attacks.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Diffusion Model Watermarking"
  - "IP Protection"
  - "Backdoor Attack"
  - "Fine-tuning Robustness"
  - "Black-box Detection"
date: 2026-05-08
content_hash: 606c06a4766e2c95
---

# SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-Image Diffusion Models

**Conference**: CVPR 2025  
**arXiv**: [2412.04852](https://arxiv.org/abs/2412.04852)  
**Code**: [https://github.com/taco-group/SleeperMark](https://github.com/taco-group/SleeperMark)  
**Area**: Image Generation / AI Security  
**Keywords**: Diffusion Model Watermarking, IP Protection, Backdoor Attack, Fine-tuning Robustness, Black-box Detection

## TL;DR

SleeperMark proposes a robust watermarking framework for T2I diffusion models. By explicitly decoupling watermarking information from the model's semantic knowledge, the watermark remains reliably detectable even after downstream fine-tuning (such as LoRA, DreamBooth, and ControlNet), maintaining a TPR@10⁻⁶FPR of over 0.93 under various fine-tuning attacks.

## Background & Motivation

**Background**: Training large-scale T2I diffusion models (e.g., Stable Diffusion, DeepFloyd-IF) requires massive resources, representing significant intellectual property. A common current practice is to fine-tune pre-trained models for downstream tasks (LoRA style transfer, DreamBooth personalization, ControlNet conditional control, etc.). Malicious users might fine-tune and deploy these models without authorization for profit.

**Limitations of Prior Work**: Existing watermarking methods (such as WatermarkDM, AquaLoRA, Stable Signature) do not consider the impact of changes in the model's semantic knowledge on the watermark when embedding it. When a watermarked model is fine-tuned to adapt to new tasks, the watermark knowledge is overwritten by newly learned semantic knowledge. Experiments show that in WatermarkDM, the watermark becomes unrecognizable after about 800 steps of LoRA fine-tuning, and AquaLoRA fails in less than 100 steps.

**Key Challenge**: Watermarked information is embedded and mixed with semantic knowledge in the same parameter space of the model. Fine-tuning modifies the parameters carrying the watermark, leading to inevitable forgetting or overwriting of the watermark.

**Goal**: Design a watermarking framework that maintains watermark detectability under black-box detection settings (where model parameters cannot be accessed), even after the model undergoes various downstream fine-tunings.

**Key Insight**: Model the watermark embedding as a backdoor mechanism—activating the watermarking behavior by prepending a secret trigger to the prompt, while generation under normal prompts remains unaffected. The key insight is to guide the model to associate the trajectory deviation of the watermark information with the presence of the trigger rather than specific semantic content, thereby decoupling the watermark knowledge from semantic knowledge.

**Core Idea**: Bi-optimizing two objectives—the trigger prompt (embedding the watermark) and the normal prompt (maintaining original output)—to explicitly guide the model to isolate watermark knowledge from semantic knowledge, leaving it undisturbed during semantic knowledge updates.

## Method

### Overall Architecture

The training consists of two stages: (1) Latent watermarking pre-training—training a secret encoder and a watermark extractor to learn embedding and extracting multi-bit messages in the latent space; (2) Diffusion backbone fine-tuning—utilizing the fixed secret residual obtained from pre-training to inject the watermark into the diffusion model via a backdoor mechanism. During verification, a trigger prompt is applied to the suspect model to generate images, from which messages are extracted and compared with the preset message.

### Key Designs

1. **Latent Watermark Pre-training**:

    - **Function**: Learn to embed and extract a cover-agnostic fixed watermark residual in the VAE latent space.
    - **Mechanism**: Train a secret encoder $E_\varphi$ to map a message $m$ to a fixed residual $\delta_z = E_\varphi(m)$, which is directly added to the latent representation to obtain the watermarked latent $z_w = z_{co} + \delta_z$. A secret decoder $D_\gamma$ extracts the message from the re-encoded latent of the watermarked image. The loss function includes BCE (message accuracy) + MSE + LPIPS (image fidelity). The watermark is cover-agnostic, meaning the same residual is applied to all input images.
    - **Design Motivation**: Operating watermarking in the latent space rather than the pixel space offers two advantages: (1) The latent space is naturally robust to various common distortions, eliminating the need to add distortion layers during training; (2) Even if the attacker fine-tunes the VAE encoder/decoder, the watermark extraction remains effective.

2. **Disentangled Backdoor Injection**:

    - **Function**: Decouple the watermarking behavior from semantic content, ensuring the watermark remains effective after fine-tuning.
    - **Mechanism**: Define the trigger prompt $y_{tr}$ as prepending a trigger (e.g., "*[Z]&") to the normal prompt $y$. The training objective comprises three components: (1) when conditioned on the trigger prompt and the denoising step $t$ is small, guide the model output toward the pre-trained model output plus the watermark residual $\hat{z_0}^{t,y_{tr}}_\vartheta + \delta_z^*$; (2) when $t$ is large, keep the outputs of both trigger and non-trigger prompts consistent with those of the pre-trained model; (3) under normal prompts, keep the output consistent with the frozen pre-trained model. Sigmoid weight functions $w_1(t), w_2(t)$ are used to balance the objectives at different timesteps.
    - **Design Motivation**: The watermark deviation is only injected at the end of denoising (small $t$) because the single-step estimated $z_0$ is more accurate during this phase. By co-optimizing both trigger and non-trigger prompts, the model learns to associate the watermarking behavior only with the trigger, independent of specific semantics. Since fine-tuning mainly modifies semantic knowledge and rarely touches this trigger-bound behavior pattern, the watermark is preserved.

3. **Adaptive Timestep Weights**:

    - **Function**: Control the intensity of watermark injection at different denoising stages.
    - **Mechanism**: Introduce two sigmoid functions $w_1(t)$ and $w_2(t)$, controlled by a threshold $\tau$ and steepness $\beta$. At low $t$, $w_1$ is large and $w_2$ is small, making the model prioritize learning watermark embedding; at high $t$, $w_2$ is large and $w_1$ is small, keeping the model's original behavior.
    - **Design Motivation**: In the early stages of denoising (high noise), single-step estimation is inaccurate and unsuitable for precise watermark injection; in the later stages (low noise), the watermark, as a fine-grained residual, can be embedded accurately.

### Loss & Training

In the first stage, the encoder/decoder are trained on 10K images from COCO2014 with a message length of 48 bits. In the second stage, the model is trained using 10K images generated by Stable-Diffusion-Prompts to fine-tune the attention parameters of the UNet up blocks. In SD v1.4, $\eta=0.02$, and in DeepFloyd, $\eta=0.05$. The trigger is set to a rare character combination to reduce the risk of being detected and accidentally triggered.

## Key Experimental Results

### Main Results

| Method | FID ↓ | CLIP ↑ | DreamSim ↓ | Bit Acc. ↑ | T@10⁻⁶F ↑ | T@10⁻⁶F (Adv.) ↑ |
|------|------|--------|-----------|-----------|-----------|-----------------|
| No Watermark (SD) | 16.24 | 31.57 | - | - | - | - |
| DwtDctSvd (Post-proc.) | 16.21 | 31.45 | 0.014 | 100.0 | 1.000 | 0.678 |
| Stable Signature | 16.55 | 31.59 | 0.017 | 99.13 | 0.998 | 0.719 |
| WatermarkDM | 19.07 | 30.17 | 0.279 | - | 0.883 | 0.883 |
| AquaLoRA | 16.86 | 31.15 | 0.176 | 96.92 | 0.980 | 0.945 |
| **SleeperMark** | **16.72** | **31.05** | **0.108** | **99.24** | **0.999** | **0.984** |

### Ablation Study (Fine-tuning Robustness - LoRA rank=20)

| Method | 20 Steps | 200 Steps | 2000 Steps | Description |
|------|------|-------|--------|------|
| WatermarkDM | 0.875 | 0.742 | 0.000 | Completely fails after 2000 steps |
| AquaLoRA | 0.818 | 0.001 | 0.000 | Fails at 200 steps |
| **SleeperMark** | **0.999** | **0.998** | **0.992** | Almost perfect even after 2000 steps |

### Key Findings

- SleeperMark maintains extremely high watermark detection rates across all downstream fine-tuning scenarios: LoRA (TPR $\ge$ 0.980), DreamBooth (TPR $\ge$ 0.934), and ControlNet (TPR $\ge$ 0.955).
- In contrast, AquaLoRA becomes completely undetectable after 200 steps of LoRA fine-tuning, and WatermarkDM also fails completely after 2000 steps.
- In terms of model fidelity, SleeperMark achieves a DreamSim of only 0.108 (outperforming AquaLoRA's 0.176 and WatermarkDM's 0.279), with an FID increase of less than 0.5.
- Images generated by trigger prompts are visually almost identical to those generated by normal prompts, demonstrating high imperceptibility of the watermark.
- The method is compatible with both latent diffusion models (SD) and pixel diffusion models (DeepFloyd-IF).

## Highlights & Insights

- The **decoupling of the watermark from semantic knowledge** is highly elegant. By using a backdoor mechanism to let the model learn the semantic-independent mapping of "trigger $\rightarrow$ denoising trajectory deviation," updates to semantic knowledge do not interfere with watermarking behavior. This decoupling design can inspire other scenarios requiring persistent information embedding in models.
- **Operating the watermark in the latent space** naturally provides robustness against distortion, eliminating the distortion layer training typical in traditional methods and simplifying the training pipeline.
- The design of **adaptive timestep weights** is highly insightful: emphasizing model behavior stability at high noise levels and precisely injecting the watermark residual at low noise levels aligns well with the coarse-to-fine generation characteristics of diffusion models.

## Limitations & Future Work

- Although rare character combinations are selected as triggers, an attacker who is aware of the triggering mechanism might theoretically attempt to search for and remove the trigger mapping.
- Current evaluations are mainly conducted on relatively small models like SD v1.4 and DeepFloyd-IF; its effectiveness on larger-scale models (SDXL, SD3) remains to be validated.
- The message capacity is fixed at 48 bits, which may be insufficient for scenarios requiring more embedded information.
- Although the evaluated downstream tasks cover mainstream methods (LoRA/DreamBooth/ControlNet), robustness against more aggressive full-parameter fine-tuning has not yet been fully investigated.

## Related Work & Insights

- **vs WatermarkDM**: WatermarkDM triggers watermarked images using specific prompts but lacks a decoupling mechanism, causing the watermark to fail rapidly after fine-tuning. SleeperMark's core advantage lies in its decoupling design.
- **vs AquaLoRA**: AquaLoRA embeds watermarks directly into all generated images (without a trigger mechanism), leading to entanglement between the watermark and semantic knowledge, resulting in extremely poor fine-tuning robustness.
- **vs Stable Signature**: It embeds watermarks by modifying the VAE decoder, which is only applicable to latent space models and cannot resist fine-tuning targeting the diffusion backbone.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The backdoor mechanism with watermark-semantic decoupling is a completely novel design, addressing the long-standing challenge of fine-tuning robustness.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive evaluation covering various models, fine-tuning methods, attacks, and two types of diffusion models.
- Writing Quality: ⭐⭐⭐⭐ The threat model and design objectives are clearly defined, with complete technical details.
- Value: ⭐⭐⭐⭐⭐ It genuinely solves the fine-tuning robustness issue for diffusion model watermarking for the first time, providing significant practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Personalized Preference Fine-tuning of Diffusion Models](personalized_preference_fine-tuning_of_diffusion_models.md)
- [\[CVPR 2025\] Focus-N-Fix: Region-Aware Fine-Tuning for Text-to-Image Generation](focus-n-fix_region-aware_fine-tuning_for_text-to-image_generation.md)
- [\[CVPR 2025\] Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models](efficient_fine-tuning_and_concept_suppression_for_pruned_diffusion_models.md)
- [\[CVPR 2025\] Implicit Bias Injection Attacks against Text-to-Image Diffusion Models](implicit_bias_injection_attacks_against_text-to-image_diffusion_models.md)
- [\[ICML 2025\] GaussMarker: Robust Dual-Domain Watermark for Diffusion Models](../../ICML2025/image_generation/gaussmarker_robust_dual-domain_watermark_for_diffusion_models.md)

</div>

<!-- RELATED:END -->
