---
title: >-
  [Paper Note] Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning
description: >-
  [CVPR 2026][Model Compression][AI-generated image detection] This paper proposes Image-Adaptive Prompt Learning (IAPL), which dynamically adjusts the prompts of the CLIP encoder for each test image at inference time. Through test-time token tuning and a conditional information learner, IAPL achieves strong generalization to unseen generators, attaining state-of-the-art average accuracies of 95.61% and 96.7% on UniversalFakeDetect and GenImage, respectively.
tags:
  - CVPR 2026
  - Model Compression
  - AI-generated image detection
  - prompt learning
  - test-time adaptation
  - CLIP
  - forgery detection
date: 2026-05-08
content_hash: 392eb59ea7bbc85d
---

# Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning

**Conference**: CVPR 2026
**arXiv**: [2508.01603](https://arxiv.org/abs/2508.01603)
**Code**: Available
**Area**: Model Compression
**Keywords**: AI-generated image detection, prompt learning, test-time adaptation, CLIP, forgery detection

## TL;DR

This paper proposes Image-Adaptive Prompt Learning (IAPL), which dynamically adjusts the prompts of the CLIP encoder for each test image at inference time. Through test-time token tuning and a conditional information learner, IAPL achieves strong generalization to unseen generators, attaining state-of-the-art average accuracies of 95.61% and 96.7% on UniversalFakeDetect and GenImage, respectively.

## Background & Motivation

**Background**: AI-generated image detection is a prominent research topic in the security domain. State-of-the-art methods commonly fine-tune vision foundation models such as CLIP, leveraging their rich pretrained representations to aid detection. Existing approaches including UniFD, FatFormer, and C2P-CLIP fix all learnable parameters after training.

**Limitations of Prior Work**: Models with fixed parameters after fine-tuning exhibit insufficient robustness to domain shift induced by unseen generators. Images produced by different generators vary substantially in texture, semantics, and forgery artifacts, and fixed parameters cannot capture these instance-level discriminative cues.

**Key Challenge**: Training data covers only a limited set of generation methods (e.g., ProGAN only), yet inference must handle 19 distinct generators. Prompts learned on the training set encode only the forgery distribution of that set and cannot adapt to new distributions.

**Goal**: (1) How to dynamically adapt prompts to each test image at inference time? (2) How to extract image-specific forgery cues as conditioning information? (3) How to allow instance-level adaptation while maintaining detection backbone stability?

**Key Insight**: The paper introduces the concept of Test-Time Adaptation into prompt learning — prompts are optimized not only during training but also at inference time via multi-view consistency constraints derived from a single test image.

**Core Idea**: The prompt consists of two components — a *post-training fixed conditional information* part and a *dynamically adjusted test-time token* part — fused via a learnable scaling factor to enable instance-level adaptation of the detector.

## Method

### Overall Architecture

The detection pipeline is built upon CLIP ViT-L/14. Three types of trainable components are inserted into the CLIP encoder: (1) MLP-based adapters (inserted at every $N_a=6$ blocks at equal intervals); (2) Learnable tokens (from the 2nd to the $N_t=9$th block); (3) Image-adaptive prompts (input to the 1st block). The first two are fixed after training to provide a stable backbone; the last is dynamically adjusted at inference time. The CLS token is passed through a classifier to produce the detection result.

### Key Designs

1. **Test-Time Token Tuning**:

    - **Function**: Adjusts test-time adaptive tokens based on a single test image at inference time.
    - **Mechanism**: $N_v=32$ different views (1 global + 31 local crops with flipping) are generated from the test image, and $m=6$ high-confidence views are selected. The token parameters are optimized for $T=2$ steps by minimizing the average entropy loss $L_{avg} = -(\bar{p} \log \bar{p} + (1-\bar{p})\log(1-\bar{p}))$, where $\bar{p}$ is the mean prediction across all selected views. This forces consistent predictions across multiple views.
    - **Design Motivation**: Domain shift increases prediction uncertainty; multi-view consistency constraints allow the tokens to adapt to the characteristics of the current image without requiring labels.

2. **Conditional Information Learner**:

    - **Function**: Extracts forgery-specific and general conditioning information from texture-rich regions of the input image.
    - **Mechanism**: The image is divided into $N_p=192$ patches of size $32 \times 32$; DCT scores are used to select the most texture-rich patch, from which high-frequency patterns are extracted via a high-pass filter. Two CNNs with identical architectures but independent parameters extract a forgery-specific condition $C_f$ (with auxiliary supervision) and a general condition $C_g$ (without supervision), respectively.
    - **Design Motivation**: CLIP pretraining focuses on high-level semantics and tends to overlook low-level forgery artifacts (frequency anomalies, pixel-level patterns, etc.). Conditioning on high-frequency textures directly compensates for this limitation. The two separate branches allow one to focus on forgery discrimination and the other to capture general image state.

3. **Learnable Scaling Factor**:

    - **Function**: Fuses test-time tokens and conditional information into the final image-adaptive prompt.
    - **Mechanism**: $P = \{\alpha_f \cdot C_f + A[0,:],\ \alpha_g \cdot C_g + A[1,:]\}$, where $\alpha_f, \alpha_g$ are learnable channel-wise coefficients optimized during training to obtain the optimal fusion ratio.
    - **Design Motivation**: Conditional information and adaptive tokens capture different types of cues; the scaling factor enables fine-grained channel-level control over their combination.

### Loss & Training

Training loss: $L_{overall} = L_{cls} + L_{aux}$, both binary cross-entropy. At inference, test-time tokens are tuned using the average entropy loss $L_{avg}$. Training runs for only 1 epoch with a learning rate of $5 \times 10^{-5}$ on a single RTX 3090. The test-time tuning learning rate is $5 \times 10^{-3}$ with 2 optimization steps. An Optimal Input Selection strategy is also applied, where the prediction with the highest confidence across multiple views of the same image is taken as the final output.

## Key Experimental Results

### Main Results (UniversalFakeDetect, ProGAN 4-class training, Acc%)

| Method | ProGAN | StyleGAN | BigGAN | LDM(200) | DALLE | GauGAN | mAcc |
|--------|--------|----------|--------|----------|-------|--------|------|
| UniFD | 100.0 | 82.0 | 94.5 | 72.0 | 81.38 | 99.5 | 86.78 |
| FatFormer | 99.89 | 97.15 | 99.50 | 69.45 | 98.75 | 99.41 | 90.86 |
| C2P-CLIP | 99.98 | 96.44 | 99.12 | 93.29 | 98.55 | 99.17 | 93.79 |
| **IAPL** | **100.0** | **98.90** | **99.65** | **95.35** | **98.90** | **99.55** | **95.61** |

### Ablation Study

| Configuration | mAcc | Note |
|---------------|------|------|
| Full IAPL | 95.61 | Complete method |
| w/o test-time tuning | 93.89 | −1.72 without inference-time tuning |
| w/o conditional info | 94.23 | −1.38 without conditional information |
| w/o scaling factor | 94.67 | −0.94 without scaling factor |
| w/o MLP adapter | 94.12 | −1.49 without adapter |

### Key Findings

- Test-time tuning contributes the largest gain (+1.72%), validating the effectiveness of inference-time adaptation.
- T-SNE visualizations qualitatively show that IAPL brings features of unseen forged images closer to seen forged images and further from real images.
- Training on GenImage with SD v1.4 achieves 96.7% mAcc, with strong generalization to unseen generators such as Midjourney and ADM.
- Only 1 epoch of training and 2 inference-time tuning steps are required, indicating high training efficiency.

## Highlights & Insights

- **Inference-Time Prompt Adaptation**: Combining test-time adaptation with prompt learning for forgery detection is a novel composition. Per-image customized prompts adapt more effectively to unseen domains than fixed prompts.
- **High-Frequency Texture Conditioning**: Extracting high-frequency conditioning information from the DCT-highest-scoring patch elegantly compensates for CLIP's semantic bias with minimal computational overhead (processing only a single $32\times32$ patch).
- **Extremely Low Training Cost**: Only 1 epoch on a single RTX 3090 — significantly more economical than comparable methods such as FatFormer, which requires multiple epochs and larger hardware.

## Limitations & Future Work

- Test-time tuning introduces additional inference latency: generating 32 views and performing 2 gradient update steps may be a bottleneck for real-time applications.
- Conditional information is extracted from only the single most texture-rich patch, potentially missing forgery cues distributed across multiple regions.
- Accuracy on low-level forgery methods such as SITD and SAN remains variable (68–95%), indicating that the conditional information branch does not adequately capture certain forgery types.

## Related Work & Insights

- **vs. C2P-CLIP**: C2P-CLIP injects category concepts via contrastive learning with prompts fixed after training. IAPL additionally introduces inference-time tuning, improving mAcc from 93.79% to 95.61%.
- **vs. FatFormer**: FatFormer enhances adapters with frequency analysis but keeps prompts fixed. IAPL achieves superior performance through the combination of dynamic prompts and conditional information.
- **vs. TPT/R-TPT**: IAPL is inspired by test-time prompt tuning but additionally incorporates a forgery-detection-specific conditional information branch, outperforming vanilla TPT.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Combining test-time adaptation with prompt learning for forgery detection is a novel composition.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Two standard benchmarks, 19+ generators, and comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear with intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ Provides important practical reference for AI-generated content detection.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Bilevel Layer-Positioning LoRA for Real Image Dehazing](bilevel_lora_real_image_dehazing.md)
- [\[CVPR 2026\] FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation](fozo_forward-only_zeroth-order_prompt_optimization_for_test-time_adaptation.md)
- [\[AAAI 2026\] Your AI-Generated Image Detector Can Secretly Achieve SOTA Accuracy, If Calibrated](../../AAAI2026/model_compression/your_ai-generated_image_detector_can_secretly_achieve_sota_accuracy_if_calibrate.md)
- [\[CVPR 2026\] On the Robustness of Diffusion-Based Image Compression to Bit-Flip Errors](on_the_robustness_of_diffusion-based_image_compression_to_bit-flip_errors.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)

<!-- RELATED:END -->
