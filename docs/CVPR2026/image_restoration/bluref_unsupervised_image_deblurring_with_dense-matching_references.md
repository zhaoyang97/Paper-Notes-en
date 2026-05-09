---
title: >-
  [Paper Note] BluRef: Unsupervised Image Deblurring with Dense-Matching References
description: >-
  [CVPR 2026][Image Restoration][Unsupervised Deblurring] BluRef is proposed as the first unsupervised framework that leverages unpaired reference sharp images to generate pseudo ground truth via dense matching for training a deblurring network, achieving performance comparable to or even surpassing supervised methods.
tags:
  - CVPR 2026
  - Image Restoration
  - Unsupervised Deblurring
  - Dense Matching
  - Pseudo Ground Truth Generation
  - Reference Images
  - Iterative Optimization
date: 2026-05-08
content_hash: 5ef0b8d4b8c1e58a
---

# BluRef: Unsupervised Image Deblurring with Dense-Matching References

**Conference**: CVPR 2026
**arXiv**: [2603.14176](https://arxiv.org/abs/2603.14176)
**Code**: [Project Page](https://qualcomm-ai-research.github.io/BluRef/)
**Area**: Image Restoration
**Keywords**: Unsupervised Deblurring, Dense Matching, Pseudo Ground Truth Generation, Reference Images, Iterative Optimization

## TL;DR

BluRef is proposed as the first unsupervised framework that leverages unpaired reference sharp images to generate pseudo ground truth via dense matching for training a deblurring network, achieving performance comparable to or even surpassing supervised methods.

## Background & Motivation

**Motion blur is ubiquitous**: Motion blur in images and videos significantly degrades visual quality and impairs the performance of downstream vision tasks, making reliable deblurring of considerable practical value.

**Paired data acquisition is prohibitively expensive**: Mainstream supervised methods rely on paired blur-sharp training data, which requires complex equipment such as beam splitters and multi-camera synchronization systems. Such setups are nearly impossible to deploy in most real-world capture scenarios (e.g., dashcams, body cameras).

**Reblurring methods suffer from domain discrepancy**: Methods such as Blur2Blur attempt indirect deblurring by mapping unknown blur to an intermediate blur domain with existing paired data, but identifying a suitable intermediate domain is itself challenging. Domain mismatch leads to performance degradation, and multi-stage pipelines incur additional computational overhead.

**Domain gap is the fundamental bottleneck**: Whether it is the train-test domain discrepancy of supervised methods or the intermediate domain matching problem of reblurring methods, domain gap consistently limits the performance ceiling of unsupervised deblurring.

**Unpaired reference images are readily available**: In the same scene, sharp and blurry frames often coexist at different moments, making large quantities of unpaired same-scene images a natural and accessible training resource.

**Existing reference-based methods still require supervised training**: Prior reference-enhanced deblurring methods (e.g., dual-camera face enhancement by Lai et al.) still depend on paired data for training and do not fundamentally address the core challenge of unsupervised learning.

## Method

### Overall Architecture

BluRef is an iterative optimization framework consisting of two alternating steps: (1) **Pseudo Sharp Image Generation** — a dense matching model establishes correspondences between the current deblurring output and unpaired reference sharp images to generate pseudo ground truth; (2) **Deblurring Network Training** — the network parameters are updated using the pseudo sharp images as supervision targets. Within each epoch, the improved deblurring results are fed back to the dense matching module, enabling the pseudo ground truth to progressively approach real sharp images. At inference time, only a single forward pass through the deblurring network is required; the dense matching module is discarded entirely.

### Key Design 1: Self-Supervised Dense Matching Model

- **Function**: A dense matching model $\mathcal{DM}$ is trained to take a blurry target image and a sharp reference image as input and produce a warped image $I_{\text{trans}}$ and a confidence mask $M_{\text{conf}}$.
- **Mechanism**: A self-supervised scheme is adopted, applying random geometric transformations (homography and TPS) to sharp images to generate synthetic training pairs, followed by blur/noise augmentation inspired by BSRGAN, enabling the model to perform cross-domain matching between sharp and blurry images. The base architecture is PDC-Net+ with GLU-Net-GOCor.
- **Design Motivation**: Synthetic data is used solely for pre-training the matching model without involving real blur patterns, preserving the integrity of the unsupervised framework. Blur augmentation naturally adapts the model to cross-domain matching tasks.

### Key Design 2: Pseudo Sharp Image Generation Strategy

- **Function**: Given a blurry image and $N$ reference sharp images, dense matching aggregates corresponding regions from multiple references to produce a complete pseudo sharp image.
- **Mechanism**: Three aggregation strategies are proposed — (a) **Weighted Average**: independent matching followed by weighted averaging; (b) **Sequential Accumulation**: iteratively using the previous result as input to the next step to maintain detail continuity; (c) **Progressive Reference Averaging**: matching new references only for unmatched regions at each iteration and fusing all results, balancing coverage and detail preservation.
- **Design Motivation**: A single reference image typically covers less than 40% of the scene content. Multi-reference aggregation compensates for coverage gaps across different references. The progressive strategy additionally avoids redundant matching and information loss, and achieves the best empirical performance.

### Key Design 3: Iterative Target Refinement

- **Function**: The deblurring network and pseudo sharp images are alternately optimized during training, each improving the other. The blurry image itself is used as the matching input in the first iteration.
- **Mechanism**: At iteration $k$, the deblurring result $I_{\text{deblur}}^{(k)}$ is fed to the dense matching module to generate an improved $I_{\text{pseudo}}^{(k)}$, which in turn supervises stronger network parameters $\Theta^{(k+1)}$, yielding better deblurring outputs in the next round — forming a positive feedback loop.
- **Design Motivation**: The initial matching quality between blurry and reference images is limited, but as the deblurring network improves, the matching input becomes progressively sharper and the pseudo ground truth quality continuously improves. Experiments show that PSNR converges noticeably after 100K iterations.

### Key Design 4: Lightweight Deployment at Inference

- **Function**: After training, the dense matching module and pseudo sharp image generation pipeline are discarded, and only the deblurring network is retained for inference.
- **Mechanism**: The generated pseudo paired data can be reused to train deblurring networks of arbitrary capacity, including lightweight models suitable for mobile deployment, enabling flexible deployment scenarios.
- **Design Motivation**: This eliminates the dependency on reference images at inference time, maintaining the same inference cost as a standard deblurring network while extending the practical utility of the framework.

## Loss & Training

The training loss for the deblurring network is a confidence-mask-weighted reconstruction loss:

$$\Theta^{(k+1)} := \arg\min_{\Theta} \mathcal{L}\left(\mathcal{D}(I_{\text{blur}};\Theta) * \bar{M}^{(k)}_{\text{pseudo}},\; I^{(k)}_{\text{pseudo}} * \bar{M}^{(k)}_{\text{pseudo}}\right)$$

where $\bar{M}^{(k)}_{\text{pseudo}}$ is a binarized confidence mask (threshold 0.7), and $\mathcal{L}$ can be $L_1$, $L_2$, or PSNR loss. The masking mechanism restricts the network to learn only from high-confidence regions, avoiding noisy supervision from incorrectly matched areas.

## Key Experimental Results

### Table 1: Quantitative Comparison on GoPro and RB2V Datasets (PSNR/SSIM)

| Method | GoPro Δ=1 | GoPro Δ=10 | GoPro Δ=20 | RB2V Δ=1 | RB2V Δ=10 | RB2V Δ=20 |
|---|---|---|---|---|---|---|
| DualGAN | 22.23/0.721 | 22.10/0.719 | 21.24/0.702 | 21.01/0.512 | 20.87/0.500 | 20.92/0.505 |
| UID-GAN | 23.42/0.732 | 23.18/0.724 | 22.38/0.724 | 22.22/0.578 | 22.01/0.551 | 22.13/0.569 |
| UAUD | 24.25/0.792 | 24.02/0.750 | 23.77/0.745 | 22.87/0.590 | 22.29/0.581 | 22.28/0.581 |
| NAFNet-BluRef (Prog.) | **31.94/0.960** | **31.87/0.955** | **31.52/0.947** | **27.87/0.821** | **27.72/0.820** | **27.24/0.812** |
| Restormer-BluRef (Prog.) | 31.02/0.950 | 30.97/0.949 | 30.95/0.938 | 26.82/0.839 | 26.76/0.832 | 26.13/0.829 |
| NAFNet (Supervised Upper Bound) | 33.32/0.962 | — | — | 28.54/0.824 | — | — |

BluRef achieves 31.94 dB on GoPro (vs. 33.32 dB supervised) and even surpasses the supervised upper bound on RB2V with the Restormer backbone (27.87 vs. 27.43). Performance degrades only marginally as Δ increases from 1 to 20, demonstrating robustness to temporal distance of reference frames.

### Table 2: BluRef + Blur2Blur Combination on Real-World Data (NIQE↓/FID↓)

| Method | NIQE/FID |
|---|---|
| BSRGAN | 13.34/10.25 |
| Blur2Blur (GoPro) | 12.01/8.93 |
| Blur2Blur (RSBlur) | 10.07/6.28 |
| BluRef | 10.43/6.45 |
| **BluRef + Blur2Blur (RSBlur)** | **8.47/5.62** |

On the PhoneCraft real-world dataset without paired ground truth, combining BluRef with Blur2Blur significantly outperforms either method alone in both NIQE and FID. Furthermore, the performance gap between a deblurring model trained on BluRef-generated pseudo paired data and one trained on real ground truth is less than 1 dB PSNR (27.73 vs. 28.54 on RB2V).

### Ablation Study: Number of Reference Frames (GoPro, NAFNet, Δ=1)

| Number of Reference Frames | 4 | 6 | 8 | 10 |
|---|---|---|---|---|
| PSNR/SSIM | 31.42/0.942 | **31.94/0.960** | 31.93/0.961 | 31.05/0.924 |

Six to eight frames constitute the optimal range; too few leads to insufficient coverage, while too many introduces redundancy and severe misalignment.

## Highlights & Insights

1. **First unsupervised deblurring framework guided by unpaired reference images**: Completely eliminates the need for paired training data and pre-trained deblurring networks, requiring only unpaired video frames from the same scene.
2. **Performance approaching or surpassing supervised methods**: Exceeds the supervised Restormer upper bound on the RB2V real blur dataset (27.87 vs. 27.43) and falls within ~1.4 dB of the supervised baseline on GoPro.
3. **Zero additional inference overhead**: The dense matching and pseudo GT generation modules used during training are entirely discarded at inference, making the inference cost equivalent to that of a standard deblurring backbone.
4. **Reusable pseudo paired data**: The generated pseudo pairs can train networks of arbitrary capacity, including lightweight mobile models, enabling one-time training to benefit multiple architectures.
5. **Robust to temporal distance of reference frames**: Even at Δ=20 with a matching rate of only 25–28%, performance degradation remains minimal, demonstrating strong robustness.

## Limitations & Future Work

1. **Dependency on same-scene reference images**: The framework requires sharp reference images from the same or similar scene, making it inapplicable to isolated images for which no reference can be obtained.
2. **Pre-training cost of the dense matching model**: Although trained on synthetic data, PDC-Net+ still incurs non-trivial computational overhead during both training and inference, limiting BluRef's training efficiency on large-scale datasets.
3. **Upper bound of pseudo GT quality**: When overlap between reference and blurry images is minimal (e.g., large motion or scene changes), the quality of pseudo sharp images is constrained, potentially degrading training effectiveness.
4. **No end-to-end joint optimization**: The dense matching model and deblurring network are trained separately and optimized in alternation, without end-to-end joint optimization, which may lead to suboptimal solutions.

## Related Work & Insights

- **Unsupervised deblurring**: Methods based on CycleGAN or self-enhancement such as DualGAN, UID-GAN, and UAUD fall significantly short of BluRef, with PSNR gaps of 7–9 dB.
- **Reblurring methods**: Blur2Blur performs indirect deblurring via domain transfer and depends on the quality of the intermediate domain selection. It is complementary to BluRef and can be combined with it.
- **Reference-enhanced deblurring**: Xiang et al. use reference videos to enhance deblurring networks; Zou et al. and Liu et al. also explore reference enhancement, but all operate within supervised frameworks.
- **Dense matching**: Methods such as DGC-Net, GLU-Net, and PDC-Net+ are innovatively applied in this work to establish semantic correspondences across the sharp-blur domain gap.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to introduce dense matching with unpaired references into unsupervised deblurring; the framework design is original and intuitively motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers synthetic and real datasets, multiple backbone networks, multi-strategy ablations, and comprehensive comparisons against supervised upper bounds and combined methods.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, well-motivated problem formulation, and highly informative figures and tables.
- **Value**: ⭐⭐⭐⭐ — Substantially lowers the barrier to acquiring deblurring training data; the pseudo paired data reuse mechanism further extends practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SelfHVD: Self-Supervised Handheld Video Deblurring](selfhvd_self-supervised_handheld_video_deblurring.md)
- [\[CVPR 2026\] UDAPose: Unsupervised Domain Adaptation for Low-Light Human Pose Estimation](udapose_unsupervised_domain_adaptation_for_low_light_human_pose_estimation.md)
- [\[CVPR 2026\] MAD-Avatar: Motion-Aware Animatable Gaussian Avatars Deblurring](motionaware_animatable_gaussian_avatars_deblurring.md)
- [\[ICCV 2025\] Efficient Concertormer for Image Deblurring and Beyond](../../ICCV2025/image_restoration/efficient_concertormer_for_image_deblurring_and_beyond.md)
- [\[ICCV 2025\] Blind Noisy Image Deblurring Using Residual Guidance Strategy](../../ICCV2025/image_restoration/blind_noisy_image_deblurring_using_residual_guidance_strateg.md)

</div>

<!-- RELATED:END -->
