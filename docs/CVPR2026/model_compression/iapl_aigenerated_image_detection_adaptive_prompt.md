---
title: >-
  [Paper Note] Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning
description: >-
  [CVPR 2026][Model Compression][IAPL] This paper proposes IAPL (Image-Adaptive Prompt Learning), which introduces dynamic prompts at the input of a CLIP encoder. These prompts are generated via two complementary pathways: a Conditional Information Learner (extracting forgery-specific and generic cues from texture-rich regions) and test-time token tuning (minimizing entropy through multi-view consistency). The model adaptively adjusts to each test image at inference time, achieving significantly improved detection generalization on unseen generators.
tags:
  - CVPR 2026
  - Model Compression
  - IAPL
  - image-adaptive prompt
  - test-time adaptation
  - CLIP
  - forgery detection
date: 2026-05-08
content_hash: 593166587a0df192
---

# Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning

**Conference**: CVPR 2026
**arXiv**: [2508.01603](https://arxiv.org/abs/2508.01603)
**Code**: Available
**Area**: Model Compression
**Keywords**: IAPL, image-adaptive prompt, test-time adaptation, CLIP, forgery detection

## TL;DR

This paper proposes IAPL (Image-Adaptive Prompt Learning), which introduces dynamic prompts at the input of a CLIP encoder. These prompts are generated via two complementary pathways: a Conditional Information Learner (extracting forgery-specific and generic cues from texture-rich regions) and test-time token tuning (minimizing entropy through multi-view consistency). The model adaptively adjusts to each test image at inference time, achieving significantly improved detection generalization on unseen generators.

## Background & Motivation

The core challenge in **AI-generated image detection** is **generalizing to unseen generators**. As GANs and diffusion models continue to emerge, training data can only cover a limited set of generation methods, requiring models to detect images from generators never encountered during training.

**Existing approaches** primarily enhance detection capability by fine-tuning visual foundation models such as CLIP, leveraging the rich real-world knowledge encoded in pre-trained models to complement the limited forgery patterns available in training data. However, **parameters remain fixed after fine-tuning**—fixed parameters cannot capture the discriminative features unique to each test image when confronted with images from diverse generators. Images produced by different generators vary substantially in texture, semantics, and visual artifacts, making a single fixed model difficult to generalize comprehensively.

**IAPL's starting point**: dynamically adjusting the prompts fed to the encoder at inference time—not fixed after training, but adaptively tuned according to the characteristics of each test image. This is achieved through two complementary pathways: (1) image-extracted conditional information providing instance-specific cues; and (2) test-time token tuning aligning parameters via multi-view consistency constraints. The core idea is to retain a stable backbone while endowing a small set of parameters with per-instance flexibility.

## Method

### Overall Architecture

Three categories of trainable parameters are added on top of a frozen CLIP ViT-L/14: (1) MLP-based adapters inserted into $N_a$ encoder blocks (fixed after training); (2) learnable tokens added to blocks 2 through $N_t$ (fixed after training); and (3) Image-Adaptive Prompts added to the first block as input (dynamically adjusted at inference). The final CLS token is passed through a classifier to produce the detection result.

### Key Designs

1. **Conditional Information Learner**:

    - *Function*: Extracts instance-specific forgery cues from each input image.
    - *Mechanism*: The input image is divided into $N_p=192$ patches; DCT scores are used to select the most texture-rich patch. After applying a high-pass filter to extract high-frequency patterns, two independent CNNs extract the **forgery-specific condition** $C_f$ (guided by an auxiliary supervised branch) and the **generic condition** $C_g$ (learned via an unsupervised branch).
    - *Design Motivation*: High-level semantic features from CLIP may miss low-level forgery traces (e.g., frequency-domain anomalies, texture inconsistencies). High-pass filtering combined with DCT-based selection of texture-rich regions supplements low-level forgery signals. The conditional information is per-instance, providing distinct detection guidance for each image.

2. **Test-Time Token Tuning**:

    - *Function*: Dynamically adjusts token parameters at inference time based on a single test image.
    - *Mechanism*: $N_v=32$ multi-view crops are generated for each test image (1 global resize + 31 random crops); the $m=6$ most confident views are selected, and the **average entropy loss** $L_{avg} = -(\bar{p} \log \bar{p} + (1-\bar{p})\log(1-\bar{p}))$ is minimized to tune the test-time adaptive token (2 optimization steps, learning rate $5\times10^{-3}$).
    - *Design Motivation*: Domain shift renders training-time prompts suboptimal on unseen data. Multi-view prediction consistency constraints at test time adapt the model to the current image. Entropy minimization requires no labels and is a standard approach in test-time adaptation.

3. **Learnable Scaling Factor Fusion**:

    - *Function*: Fuses conditional information and test-time tokens to generate the final image-adaptive prompt.
    - *Mechanism*: $P = \{\alpha_f \cdot C_f + A[0,:], \alpha_g \cdot C_g + A[1,:]\}$, where $\alpha_f, \alpha_g$ are channel-wise learnable coefficients controlling the contribution of each information source. In subsequent blocks, prompts from the previous layer are fused with learnable tokens via analogous scaling factors.
    - *Design Motivation*: Conditional information and tuned tokens each offer distinct advantages—the former provides instance-specific low-level cues while the latter offers high-level alignment through optimization. The scaling factors allow the model to adaptively determine the contribution ratio of both.

### Loss & Training

The training loss is $L_{overall} = L_{cls} + L_{aux}$, where both terms are BCE losses. $L_{cls}$ is the final classification loss, and $L_{aux}$ is the auxiliary supervision for the forgery-specific branch of the Conditional Information Learner. At test time, $L_{avg}$ is used to tune the test-time token. Training requires only **1 epoch** on a single GPU (RTX 3090).

## Key Experimental Results

### Main Results

**UniversalFakeDetect (trained on ProGAN, tested on 19 subsets)**:

| Method | mAcc↑ | mAP↑ |
|--------|-------|------|
| UniFD | 75.4 | 79.5 |
| C2P-CLIP | 91.4 | 95.6 |
| FatFormer | 92.7 | 95.4 |
| **IAPL** | **95.61** | **97.8** |

**GenImage (trained on SD v1.4, tested on 8 generators)**:

| Method | mAcc↑ |
|--------|-------|
| C2P-CLIP | 93.1 |
| **IAPL** | **96.7** |

### Ablation Study

| Configuration | mAcc | Note |
|---------------|------|------|
| Fixed parameters only (no dynamic prompt) | Lower | Lacks instance-level adaptation |
| + Conditional Information Learner | Improved | Low-level cues are beneficial |
| + Test-time token tuning | Further improved | Effective domain-shift adaptation |
| **Full IAPL** | **95.61** | Two pathways are complementary |

### Key Findings

- T-SNE visualization shows that unseen fake features produced by IAPL are more similar to seen fake features and more clearly separated from real features, demonstrating that dynamic prompts genuinely improve generalization to new generators.
- SOTA performance is achieved with only 1 training epoch, indicating that CLIP's pre-trained knowledge combined with a small number of flexible parameters is sufficient to capture forgery patterns.
- Extracting conditional information from the most texture-rich patch is a highly efficient design—processing only a single 32×32 patch incurs minimal computational overhead.

## Highlights & Insights

- The "frozen backbone + dynamic prompt" design philosophy is elegant: the majority of parameters provide stable representational capacity, while a small number of parameters (only 2 prompt tokens) provide per-instance flexibility. This architectural division of responsibility is more efficient and stable than full model fine-tuning or full-parameter test-time adaptation.
- Test-time token tuning requires only 2 optimization steps over 6 views, suggesting that the signal for domain shift is strong and that fine-tuning the decision boundary does not require extensive iteration—an important property for controlling inference latency in practical deployment.

## Limitations & Future Work

- Test-time tuning introduces additional inference latency (requiring multiple forward passes and gradient backpropagation), which may be prohibitive for real-time detection scenarios.
- Using DCT to select a single texture-rich patch for conditional information extraction may overlook forgery cues in other image regions.
- The current framework only addresses binary classification (real/fake) and has not been extended to generator attribution tasks.

## Related Work & Insights

- **vs. C2P-CLIP / FatFormer**: These methods fix all parameters after training, whereas IAPL dynamically adjusts prompts at inference time, yielding a 3–4% improvement in generalization.
- **vs. TPT (Test Prompt Tuning)**: IAPL extends and upgrades TPT for the forgery detection domain by incorporating the Conditional Information Learner as a supplementary channel, providing low-level forgery cues that CLIP alone cannot capture.

## Rating

- Novelty: ⭐⭐⭐⭐ The dual-pathway dynamic prompt design combining conditional information and test-time token tuning is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on two standard benchmarks with comprehensive ablations and convincing T-SNE visualizations.
- Writing Quality: ⭐⭐⭐⭐ Pipeline diagrams are clear and mathematical notation is rigorous.
- Value: ⭐⭐⭐⭐ Highly practical given the growing importance of AI-generated content detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Bilevel Layer-Positioning LoRA for Real Image Dehazing](bilevel_lora_real_image_dehazing.md)
- [\[CVPR 2026\] FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation](fozo_forward-only_zeroth-order_prompt_optimization_for_test-time_adaptation.md)
- [\[CVPR 2026\] On the Robustness of Diffusion-Based Image Compression to Bit-Flip Errors](on_the_robustness_of_diffusion-based_image_compression_to_bit-flip_errors.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] RDVQ: Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression](rdvq_differentiable_vq_image_compression.md)

</div>

<!-- RELATED:END -->
