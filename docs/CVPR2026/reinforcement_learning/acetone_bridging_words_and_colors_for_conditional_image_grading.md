---
title: >-
  [Paper Note] AceTone: Bridging Words and Colors for Conditional Image Grading
description: >-
  [CVPR 2026][Reinforcement Learning][Color grading] AceTone is proposed as the first unified framework for multimodal-conditioned color grading supporting both text and reference image inputs. By compressing 3D-LUTs into 64 discrete tokens via VQ-VAE, a VLM is trained to predict LUT token sequences, followed by GRPO reinforcement learning to align color similarity and aesthetic preference, achieving a 50% improvement in LPIPS on both style transfer and instruction-based grading tasks.
tags:
  - CVPR 2026
  - Reinforcement Learning
  - Color grading
  - 3D-LUT
  - VQ-VAE tokenizer
  - VLM
  - GRPO reinforcement learning
date: 2026-05-08
content_hash: 71ed9ba50ded9fad
---

# AceTone: Bridging Words and Colors for Conditional Image Grading

**Conference**: CVPR 2026
**arXiv**: [2604.00530](https://arxiv.org/abs/2604.00530)
**Code**: [https://github.com/martian422/AceTone](https://github.com/martian422/AceTone)
**Area**: Image Processing / Color Grading
**Keywords**: Color grading, 3D-LUT, VQ-VAE tokenizer, VLM, GRPO reinforcement learning

## TL;DR
AceTone is proposed as the first unified framework for multimodal-conditioned color grading supporting both text and reference image inputs. By compressing 3D-LUTs into 64 discrete tokens via VQ-VAE, a VLM is trained to predict LUT token sequences, followed by GRPO reinforcement learning to align color similarity and aesthetic preference, achieving a 50% improvement in LPIPS on both style transfer and instruction-based grading tasks.

## Background & Motivation

**Background**: Color toning/grading is critical for image style and emotional expression. Existing methods either rely on weighted combinations of predefined filter libraries or apply CNN-based patch-wise recoloring. Style transfer from reference images and text-guided grading are handled by incompatible models.

**Limitations of Prior Work**: (1) Existing methods lack sufficient expressive capacity or computational efficiency; (2) adversarial losses (GAN) suffer from training instability and mode collapse; (3) no mechanism exists for alignment with human aesthetic preferences; (4) reference-based transfer and text-driven grading require separate models.

**Key Challenge**: Color grading demands both precise color control (the strength of LUTs) and understanding of complex semantic instructions (the strength of VLMs), yet the two have not been effectively integrated.

**Key Insight**: Treating LUTs as atomic color transformation operations and tokenizing them for VLM-based generation.

**Core Idea**: (1) A VQ-VAE tokenizer compresses a $3 \times 32^3$ LUT into 64 discrete tokens; (2) a VLM autoregressively predicts the LUT token sequence; (3) GRPO uses color similarity and aesthetic scores as rewards for preference alignment.

## Method

### Overall Architecture
Three-stage training pipeline: (1) LUT Tokenizer training (VQ-VAE) → (2) Generative pretraining (VLM learns LUT token prediction) → (3) Post-training (SFT for task adaptation + GRPO for preference alignment). At inference: query image + text/reference image → VLM predicts LUT tokens → decoded into a 3D-LUT → applied to the image.

### Key Designs

1. **3D LUT Tokenizer (VQ-VAE)**:

    - Function: Compresses a continuous $3 \times 32 \times 32 \times 32$ LUT into 64 discrete tokens.
    - Mechanism: A 3D convolutional encoder progressively downsamples to $4 \times 4 \times 4 \times D$ → vector quantization layer ($K=256$ codebook entries) → 3D convolutional decoder. Loss: $\mathcal{L} = \mathcal{L}_{rec} + \beta \mathcal{L}_{commit}$
    - Fidelity: $\Delta E < 2$ (color difference imperceptible to the human eye)
    - Design Motivation: A LUT is inherently a 3D color mapping volume; VQ-VAE can effectively compress it while preserving high fidelity.

2. **VLM-based LUT Token Prediction**:

    - Function: Trains the VLM to autoregressively predict LUT token sequences from visual-textual inputs.
    - **Generative Pretraining**: Large-scale (image, LUT, prompt) triplets with $\mathcal{L}_{gen} = -\sum \log p_\theta(z_t | z_{<t}, I, L(I), c)$
    - **SFT**: Separate training data is curated for photo style transfer (PST) and instruction-guided grading (IGG). PST provides reference and query image pairs; IGG uses Qwen2.5-VL-32B to generate editing instructions for (image, LUT) pairs.
    - Design Motivation: Formalizing color transformation as a token sequence generation problem unifies reference-based and text-based conditioning.

3. **GRPO Reinforcement Learning Alignment**:

    - Function: Aligns the model's color grading outputs with human aesthetic preferences.
    - Two reward functions:
        - $r_{color}$: Color similarity, $\frac{1}{\max(2, \Delta E) - 1}$ (maximum reward when $\Delta E < 2$)
        - $r_{aes}$: Aesthetic score, evaluated using a pretrained DeQA model for visual pleasantness.
    - Standard GRPO training: Sample $G$ candidate LUTs → compute rewards → normalize advantages within the group → policy update.
    - Design Motivation: Avoids GAN instability by first establishing a stable likelihood-based generative model, then aligning with preferences via RL.

4. **AceTone-800K Dataset**:

    - ~10K licensed LUT filters + PPR-10K expert retouching + 8,192 core LUTs selected via PCA clustering.
    - 800K automatically annotated (image, LUT, instruction) tuples.
    - Two benchmarks: AceTone-Bench[Transfer] (1,024 samples) and AceTone-Bench[Instruct] (128 samples).

### Loss & Training
Tokenizer: MSE + commitment loss. Pretraining/SFT: cross-entropy. RL: GRPO objective + KL regularization.

## Key Experimental Results

### Main Results (Style Transfer PST-50)

| Method | Aes.↑ | PSNR↑ | LPIPS↓ | ΔE↓ |
|--------|-------|-------|--------|-----|
| Neural Preset | 3.03 | 21.24 | 0.15 | 9.57 |
| SA-LUT | 3.07 | 21.64 | 0.16 | 9.01 |
| ModFlow | 3.08 | 20.13 | 0.16 | 10.62 |
| **AceTone** | **3.29** | **24.26** | **0.09** | **7.26** |

On AceTone-Bench[Transfer], LPIPS decreases from 0.22 (SA-LUT) to **0.11**, a 50% improvement.

### Ablation Study

| Configuration | Aes.↑ | LPIPS↓ | Notes |
|---------------|-------|--------|-------|
| Pretraining only | baseline | baseline | Basic LUT prediction capability |
| + SFT | +gain | +gain | Task-specific adaptation |
| + GRPO | **best** | **best** | Critical for aesthetic alignment |
| w/o aesthetic reward | degraded | unchanged | Aesthetic score contributes significantly to perceptual quality |
| w/o color reward | unchanged | degraded | Color accuracy requires color reward |

### Key Findings
- The GRPO stage primarily contributes to improved aesthetic scores and color consistency.
- The fidelity of the LUT tokenizer ($\Delta E < 2$) constitutes the accuracy foundation of the entire pipeline.
- Data diversity is critical for GRPO training — using the full training set versus a subset yields significant performance differences.
- This work is the first to demonstrate that VLMs can effectively predict discrete representations of 3D color transformations.

## Highlights & Insights
- **Innovation in LUT Tokenization**: Compressing a 3D-LUT into 64 discrete tokens elegantly transforms color grading into the "language" of VLMs, bridging the boundary between language models and color operations.
- **Staged Learning Paradigm**: Establishing a stable foundation via likelihood-based pretraining before RL-based preference alignment avoids the instability of GAN training, offering a new scalable training paradigm for color grading.
- **Unified Multimodal Conditioning**: A single model simultaneously supports both reference image and text instruction grading modes.

## Limitations & Future Work
- LUTs perform global color transformation and cannot handle local grading (e.g., adjusting only the sky color).
- The $32^3$ LUT resolution has limited precision; extreme color transformations may introduce quantization artifacts.
- GRPO training requires extensive sampling and reward computation, resulting in high training cost.
- The aesthetic evaluation model (DeQA) may introduce its own biases, which the model could inadvertently learn.

## Related Work & Insights
- **vs. Neural Preset/SA-LUT**: These methods combine predefined LUT libraries with limited expressive capacity. AceTone generates LUTs from scratch.
- **vs. Diffusion-based Editing**: Diffusion models can recolor images but incur high latency and may disrupt image structure. LUT application is lossless.
- **vs. CLIP-guided Methods**: CLIP maps text to color operations but is constrained to a few words of input. VLMs understand complex instructions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ A complete innovation chain of LUT tokenization + VLM generation + GRPO alignment.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative evaluation and user studies, though the dataset is not yet publicly released.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear and dataset construction details are thorough.
- Value: ⭐⭐⭐⭐⭐ Pioneering a new direction for language-driven color grading with practical applications in film post-production and related industries.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Reasoning-Driven Anomaly Detection and Localization with Image-Level Supervision](reasoning-driven_anomaly_detection_and_localization_with_image-level_supervision.md)
- [\[CVPR 2026\] CCCaption: Dual-Reward Reinforcement Learning for Complete and Correct Image Captioning](cccaption_dual-reward_reinforcement_learning_for_complete_and_correct_image_capt.md)
- [\[ICLR 2026\] PreferThinker: Reasoning-based Personalized Image Preference Assessment](../../ICLR2026/reinforcement_learning/preferthinker_reasoning-based_personalized_image_preference_assessment.md)
- [\[ACL 2026\] Bridging SFT and RL: Dynamic Policy Optimization for Robust Reasoning](../../ACL2026/reinforcement_learning/bridging_sft_and_rl_dynamic_policy_optimization_for_robust_reasoning.md)
- [\[ICLR 2026\] DiVE-k: Differential Visual Reasoning for Fine-grained Image Recognition](../../ICLR2026/reinforcement_learning/dive-k_differential_visual_reasoning_for_fine-grained_image_recognition.md)

<!-- RELATED:END -->
