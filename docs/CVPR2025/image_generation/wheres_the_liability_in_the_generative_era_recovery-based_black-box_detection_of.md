---
title: >-
  [Paper Note] Where's the Liability in the Generative Era? Recovery-Based Black-Box Detection of AI-Generated Content
description: >-
  [CVPR 2025][Image Generation][AI-Generated Content Detection] This paper proposes a black-box AI-generated image detection method based on a "corrupt-and-recover" strategy. The core hypothesis is that a generative model can more easily recover the masked parts of its own generated images. By fine-tuning a surrogate model with distribution alignment, detection accuracy on unknown target models is further enhanced, requiring less than 1,000 API samples and 2 hours of GPU time.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "AI-Generated Content Detection"
  - "Black-Box Detection"
  - "Diffusion Models"
  - "Image Inpainting"
  - "Distribution Alignment"
date: 2026-05-08
content_hash: 6dc9d1a5dbb7b7a7
---

# Where's the Liability in the Generative Era? Recovery-Based Black-Box Detection of AI-Generated Content

**Conference**: CVPR 2025  
**arXiv**: [2505.01008](https://arxiv.org/abs/2505.01008)  
**Code**: [https://github.com/HaoyueBaiZJU/genai-detect](https://github.com/HaoyueBaiZJU/genai-detect)  
**Area**: Diffusion Models / AI Safety  
**Keywords**: AI-Generated Content Detection, Black-Box Detection, Diffusion Models, Image Inpainting, Distribution Alignment

## TL;DR

This paper proposes a black-box AI-generated image detection method based on a "corrupt-and-recover" strategy. The core hypothesis is that a generative model can more easily recover the masked parts of its own generated images. By fine-tuning a surrogate model with distribution alignment, detection accuracy on unknown target models is further enhanced, requiring less than 1,000 API samples and 2 hours of GPU time.

## Background & Motivation

**Background**: With the advancements in generative models like Stable Diffusion and DALL-E 3, synthetic images have reached a level of realism that is indistinguishable to the human eye. Human experiments show only about 72% accuracy in identifying fake images on a carefully curated test set. Existing detection methods include white-box methods (requiring model weights), binary classifier methods (training classifiers like ResNet on GAN/diffusion model images), and frequency domain analysis.

**Limitations of Prior Work**: (1) White-box methods require model weights or token info, but most state-of-the-art models (e.g., DALL-E 3, Sora) only provide API access; (2) Binary classifier methods rely on large-scale real-fake contrastive datasets (~400K images) and easily overfit to the "fingerprints" of specific models, failing to generalize to new models; (3) Research indicates that detectors trained on specific models suffer severe performance degradation on other generative architectures; (4) Existing methods perform poorly on the latest diffusion models—GAN detectors are almost completely ineffective on diffusion-generated images.

**Key Challenge**: In practical scenarios, the specific model used by attackers is typically unknown, yet efficient detection requires utilizing model-specific distribution information—forming the conflict between "black-box vs. specificity."

**Goal**: Design a black-box detection framework that requires only API access, does not need model weights or large-scale auxiliary datasets, and is effective across various diffusion model variants.

**Key Insight**: The authors leverage an intuitive hypothesis—generative models should more easily recover images they generated themselves. This is because the distribution of the model's own output is highly aligned with its internally learned distribution, making recovery after masking more successful. In contrast, real images do not reside within the model's learned distribution, resulting in significantly poorer recovery.

**Core Idea**: Mask a portion of an image $\rightarrow$ recover it using a generative model $\rightarrow$ compare the recovery quality $\rightarrow$ high quality (low PSNR difference) indicates a fake image, while low quality (high PSNR difference) indicates a real image. For APIs that do not support masked image input, a distribution-aligned surrogate model is trained as a substitute.

## Method

### Overall Architecture

The detection process consists of three steps: (1) **Corrupt**—apply a mask $m$ to the image under test $\mathbf{x}$, dividing the image into known pixels $(1-m)\odot\mathbf{x}$ and masked pixels $m\odot\mathbf{x}$; (2) **Recover**—use the generative model (surrogate model) to recover the masked regions based on the known pixels, sampling $K$ times to obtain a set of recovery results $\Omega = \{Y_1,...,Y_K\}$; (3) **Evaluate**—calculate the difference score $\delta$ (using PSNR) between the original masked region $m\odot\mathbf{x}$ and the recovered results. If $\delta < \tau$, the image is classified as AI-generated.

For black-box models that do not support masked-image input, a surrogate model alignment step is added: a small number of generated images (<1000 images) are collected from the target model API to fine-tune a surrogate model (e.g., Stable Diffusion) via LoRA, aligning its distribution with the target model.

### Key Designs

1. **Corrupt-and-Recover Detection Paradigm**:

    - **Function**: Converts AI-generated content detection into a recovery quality estimation problem.
    - **Mechanism**: Based on the "likelihood gap hypothesis"—the expected conditional log-likelihood of the machine generation process $G$ is higher than that of the human creation process $H$ by a positive gap $\Delta$. Formalized as $\mathbb{E}_{Y\sim G(\cdot|X)}[\log p(Y|X)] - \mathbb{E}_{Y\sim H(\cdot|X)}[\log p(Y|X)] > \Delta$. This implies that given the known part $X$, the model assigns a higher probability to the remaining part $Y$ that it originally generated, leading to recovery results closer to the original.
    - **Design Motivation**: Unlike methods such as DIRE that require complete forward-backward diffusion processes (requiring white-box access or specific input formats), this method works with standard inpainting APIs. Furthermore, it does not require training on real-fake contrastive datasets, naturally avoiding overfitting.

2. **Distribution-Aligned Surrogate**:

    - **Function**: Approximates the distribution of the target black-box model using an open-source surrogate model, enabling the surrogate model's inpainting capability to detect target-model generations.
    - **Mechanism**: Collects a small dataset of generated images $S$ (<1000 images) from the target model API and applies parameter-efficient fine-tuning on the surrogate model (e.g., Stable Diffusion) using LoRA. The fine-tuned surrogate model exhibits a distribution similar to the target model, thereby recovering images generated by the target model more accurately and magnifying the recovery quality gap between real and fake images.
    - **Design Motivation**: Direct detection using an unaligned surrogate model yields limited performance (FPR 47.90%) due to the massive distribution gap from the target model. However, LoRA fine-tuning with fewer than 1000 samples dramatically bridges this distribution gap (reducing FPR to 23.60%) with a GPU time of less than 2 hours.

3. **Scoring Function**:

    - **Function**: Quantifies the difference between the original image and the recovered results to serve as the decision boundary for real/fake classification.
    - **Mechanism**: Evaluates four metrics—PSNR, SSIM, L1, and L2—finding that PSNR consistently outperforms others across all scenarios. For example, when using Stable Diffusion as the surrogate and Guided Diffusion as the target, PSNR achieves an AUROC of 94.19%, while SSIM yields only 56.13%.
    - **Design Motivation**: PSNR is highly sensitive to subtle pixel-level differences, and the recovery of AI-generated images tends to be highly consistent at the pixel level, even if structural differences exist (which SSIM focuses on). PSNR provides robust comparisons through normalization against maximum pixel values.

### Loss & Training

The surrogate model is fine-tuned using standard diffusion training loss + LoRA, while freezing the original model parameters. During inference, the recovery results are sampled $K$ times and averaged to reduce randomness. The paper provides a theoretical lower bound for $K$: $K = \Omega(\sigma\log(1/\delta)/\Delta^2)$. Regarding mask types, a "genhalf" mask (masking half of the image) outperforms a "thick" mask (line masks) because larger masked areas provide a more sufficient evaluation for recovery.

## Key Experimental Results

### Main Results

| Method | Guided | LDM 200 | LDM w/CFG | Glide 100-27 | DALL-E | Mean mAP |
|------|--------|---------|-----------|-------------|--------|---------|
| Trained DNN (B+J 0.1) | 73.72 | 70.62 | 71.00 | 80.65 | 70.59 | 75.51 |
| Patch Classifier (Xception) | 75.03 | 87.10 | 86.72 | 85.37 | 75.67 | 82.30 |
| Freq-Spec (CycleGAN) | 57.72 | 77.72 | 77.25 | 68.58 | 67.77 | 69.00 |
| **Ours (Stable Diffusion)** | **92.97** | **89.40** | **82.84** | **87.75** | **75.98** | **86.61** |

### Ablation Study

| Configuration | FPR↓ | AUROC↑ | AP↑ | Description |
|------|------|--------|-----|------|
| W/o FT (PSNR) | 47.90 | 87.84 | 86.74 | Surrogate model unaligned |
| W/ FT (PSNR) | **23.60** | **94.19** | **92.97** | FPR drops by 24.3% after distribution alignment |
| W/o FT (SSIM) | 100.0 | 45.28 | 44.36 | SSIM completely fails |
| W/ FT (SSIM) | 99.80 | 56.13 | 58.60 | SSIM remains poor even after fine-tuning |

### Key Findings

- Our method achieves an average mAP of 86.61%, outperforming the best baseline (Patch Classifier at 82.30%) by 4.31 percentage points.
- Distribution alignment fine-tuning is crucial—FPR drops from 47.90% to 23.60% using fewer than 1000 samples and requiring less than 2 hours.
- PSNR significantly outperforms other scoring functions, beating SSIM by up to 38% AUROC, likely because AI-generated "fingerprints" are more pronounced at the pixel level rather than the structural level.
- Under white-box settings (direct recovery using the target model), performance is even higher: Guided $\rightarrow$ Guided achieves 96.69% AP, validating that closer distribution matching leads to more accurate detection.
- New benchmark tests on DALL-E 3 show a pronounced performance drop for existing methods, indicating that detecting the latest generation models remains an open problem.
- In human evaluations, average human accuracy is only 72.33%, which our method exceeds in most settings.

## Highlights & Insights

- **The "corrupt-and-recover" paradigm bypasses white-box constraints**—it eliminates the need for model weight access or specific diffusion process control, requiring only standard inpainting APIs. This concept can be extended to other domains like text detection (e.g., GPT detection).
- **Distribution-aligned surrogate models** solve the "unknown target model" challenge—replicating target distribution characteristics with a minimal number of API samples. This overhead is extremely low (<1000 samples, <2h GPU), making it highly practical for real-world deployment.
- **The PSNR >> SSIM finding** provides key practical direction—selecting the correct scoring function is more critical than modifying model architectures.

## Limitations & Future Work

- Detection accuracy on the latest commercial models like DALL-E 3 remains relatively low (75.98% AP), highlighting that distribution alignment difficulty increases with model sophistication.
- If target models are entirely closed-source and feature post-processing at the API level (such as noise addition or JPEG compression), the distribution difference signal may be weakened.
- Sampling $K$ times for recovery increases inference costs, requiring a trade-off between detection latency and accuracy.
- The paper only evaluates image detection; extending this to video-generated content (e.g., Sora) requires further exploration.
- Lack of adversarial evaluation—attackers aware of the detection mechanism could potentially employ adversarial perturbations to bypass recovery evaluations.

## Related Work & Insights

- **vs. DIRE**: DIRE also leverages the reconstruction discrepancy of diffusion models for detection, but demands complete forward and backward diffusion processes (requiring white-box access). Ours requires only inpainting APIs (black-box).
- **vs. Universal Fake Detector**: This binary classification method based on pre-trained ViTs requires a massive amount of real-fake contrastive data (~400K) and generalizes poorly to new models; our method eliminates the need for real image datasets.
- **vs. Trained DNN (Wang et al.)**: ResNet-50 trained on ProGAN data performs poorly on diffusion models, demonstrating that classifier-based methods have inherent cross-model generalization issues.
- **vs. Freq-Spec**: Frequency-domain methods are sensitive to GAN frequency artifacts but fail on diffusion models because the generation processes of diffusion models are much smoother.

## Rating

- Novelty: ⭐⭐⭐⭐ The corrupt-and-recover paradigm was previously explored by works like DIRE; the main contribution of this paper lies in black-box adaptation and surrogate alignment.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coordinated across multiple diffusion model variants with comprehensive ablations (scoring functions, mask types, fine-tuning state), but lacks adversarial analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, good coupling of theoretical hypotheses and experimental validations, though some notations could be simplified.
- Value: ⭐⭐⭐⭐ Highly practical (low resource overhead, black-box applicable), though detection of the newest models still has room for improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SPAI: Any-Resolution AI-Generated Image Detection by Spectral Learning](any-resolution_ai-generated_image_detection_by_spectral_learning.md)
- [\[CVPR 2025\] A Bias-Free Training Paradigm for More General AI-generated Image Detection](a_bias-free_training_paradigm_for_more_general_ai-generated_image_detection.md)
- [\[NeurIPS 2025\] Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection](../../NeurIPS2025/image_generation/physics-driven_spatiotemporal_modeling_for_ai-generated_video_detection.md)
- [\[CVPR 2026\] BlackMirror: Black-Box Backdoor Detection for Text-to-Image Models via Instruction-Response Deviation](../../CVPR2026/image_generation/blackmirror_black-box_backdoor_detection_for_text-to-image_models_via_instructio.md)
- [\[ECCV 2024\] Zero-Shot Detection of AI-Generated Images](../../ECCV2024/image_generation/zero-shot_detection_of_ai-generated_images.md)

</div>

<!-- RELATED:END -->
