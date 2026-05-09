---
title: >-
  [Paper Note] UniRes: Universal Image Restoration for Complex Degradations
description: >-
  [ICCV 2025][Image Restoration][Complex degradations] This paper proposes UniRes — a diffusion-based universal image restoration framework that acquires expert knowledge across four tasks (super-resolution, motion deblurring, defocus deblurring, and denoising) through multi-task training. At inference time, it handles arbitrary combinations of real-world complex degradations end-to-end by flexibly composing latent-space prediction weights from different tasks.
tags:
  - ICCV 2025
  - Image Restoration
  - Complex degradations
  - diffusion models
  - multi-task training
  - latent space composition
  - real-world restoration
date: 2026-05-08
content_hash: 25f6f1b62fe8d387
---

# UniRes: Universal Image Restoration for Complex Degradations

**Conference**: ICCV 2025
**arXiv**: [2506.05599](https://arxiv.org/abs/2506.05599)
**Code**: Unavailable
**Area**: Image Restoration
**Keywords**: Complex degradations, diffusion models, multi-task training, latent space composition, real-world restoration

## TL;DR

This paper proposes UniRes — a diffusion-based universal image restoration framework that acquires expert knowledge across four tasks (super-resolution, motion deblurring, defocus deblurring, and denoising) through multi-task training. At inference time, it handles arbitrary combinations of real-world complex degradations end-to-end by flexibly composing latent-space prediction weights from different tasks.

## Background & Motivation

Real-world images frequently suffer from **multiple co-occurring degradations**: motion blur from object movement, defocus blur from focusing errors, noise from high ISO settings, and compression artifacts from low JPEG quality factors. This simultaneous presence poses significant challenges for image restoration algorithms:

**Difficulty in training data construction**: Creating paired HQ–LQ datasets with authentic complex degradations is extremely difficult. Existing datasets either cover only a single degradation type or lack sufficient scene diversity.

**Generalization gap of synthetic degradations**: Methods such as Real-ESRGAN employ synthetic degradation pipelines (Gaussian blur + noise + downscaling + JPEG compression) for training, yet frequently underperform on real images.

**Structural inconsistency in generative priors**: Methods including StableSR, DiffBIR, and SUPIR leverage pre-trained diffusion models as generative priors, achieving blind restoration through frozen backbones combined with fine-tuned adapters. However, this ControlNet-based conditioning mechanism tends to produce **pixel-level structural inconsistencies** (hallucinations) — the restored output diverges significantly from the input's pixel structure.

**Iterative limitations of all-in-one methods**: Methods such as AutoDIR and RestoreAgent sequentially identify degradation types and apply corresponding restoration operations; their performance is constrained by single-step restoration quality and accumulated iterative errors.

The core insight motivating UniRes is: can a set of "experts" be trained — each specializing in one degradation type — and then **flexibly composed** at inference time to handle arbitrary complex degradations? This is the central design philosophy of UniRes.

## Method

### Overall Architecture

UniRes builds upon a pre-trained text-to-image Latent Diffusion Model (LDM) and operates in two phases:
- **Training phase**: The LDM is jointly fine-tuned on four tasks — super-resolution, motion deblurring, defocus deblurring, and denoising.
- **Inference phase**: Complex degradations are handled end-to-end by computing a weighted combination of latent-space predictions from different tasks.

### Key Designs

1. **Latent Concatenation Conditioning Mechanism**:

    - **Function**: Uses the latent encoding of the LQ image as a conditioning input to the diffusion model.
    - **Mechanism**: Unlike ControlNet/Adapter-based methods, UniRes directly **concatenates** the LQ image latent $\boldsymbol{z}_{\text{LQ}}$ with the noisy latent $\boldsymbol{z}_t$ before feeding into the UNet:
    $\boldsymbol{\epsilon}_\theta(\boldsymbol{z}_t, \boldsymbol{z}_{\text{LQ}}, \boldsymbol{s})$
      This requires only modifying the input channel count of the UNet's first convolutional layer. Since all UNet parameters are fine-tuned on LQ–HQ paired data under an inconsistency penalty, the approach **better preserves the pixel structure of the input**.
    - **Design Motivation**: Adapter-based methods suffer from hallucinations due to the frozen backbone. The concatenation mechanism enables the entire network to learn fidelity to the input during training.

2. **Multi-task Training and Latent Prediction Composition**:

    - **Function**: Different degradation tasks are randomly sampled during training; task noise predictions are flexibly composed at inference time.
    - **Mechanism**: Given $K$ tasks, the composed prediction at inference is:
    $\tilde{\boldsymbol{\epsilon}}_\theta(\boldsymbol{z}_t, \boldsymbol{z}_{\text{LQ}}; \boldsymbol{w}) = \sum_{k=1}^{K} w_k \cdot \boldsymbol{\epsilon}_\theta(\boldsymbol{z}_t, \boldsymbol{z}_{\text{LQ}}, \boldsymbol{s}_k)$
      where $\boldsymbol{s}_k$ denotes the text prompt for the $k$-th task (e.g., "Super-resolution", "Motion-deblur"), and $w_k$ are weights satisfying $\sum w_k = 1$. This is conceptually analogous to a Mixture of Experts, where each text prompt activates distinct expert knowledge encoded in the model.
    - **Design Motivation**: Real-world image degradations are complex combinations of known degradation types. By adjusting weights, the same model can dynamically adapt its restoration strategy to different images. For instance, a severely blurred night scene photograph may be assigned a high weight for motion deblurring.

3. **DownLQ Fidelity–Quality Trade-off Mechanism**:

    - **Function**: An additional "downsampled LQ" inference task is introduced to control the extent of generated detail.
    - **Mechanism**: The composition weights include an additional DownLQ component, whose conditioning input is the LQ image first downsampled ($\times 4$) and then bicubically upsampled back to the original resolution. Due to greater information loss, the model generates richer details from this input. The fidelity–quality trade-off is controlled by adjusting the DownLQ weight.
    - **Design Motivation**: Models conditioned via concatenation are inherently conservative and tend not to over-generate details. The DownLQ component provides a natural mechanism to encourage detail generation, serving as an elegant alternative to the fidelity–quality trade-off in adapter-based methods.

4. **Optimal Composition Weight Search**:

    - **Function**: Automatically determines the optimal task composition weights for each input image.
    - **Mechanism**: The search space is defined as $\Omega = \{\boldsymbol{w} \in [\gamma, \delta]^K \mid \sum w_k = 1\}$, and the optimal weights are found by maximizing an image quality metric via grid search:
    $\boldsymbol{w}^* = \arg\max_{\boldsymbol{w} \in \Omega} Q(g(\boldsymbol{x}, \boldsymbol{w}))$
      MUSIQ is adopted as the quality function $Q(\cdot)$, with search range $[\gamma, \delta] = [-0.2, 1.2]$ and step size 0.2. Negative weights function analogously to negative guidance in classifier-free guidance.
    - **Design Motivation**: Degradation compositions differ across images, making a uniform weight assignment suboptimal. Automatic search adaptively identifies the most suitable restoration strategy for each image. Search complexity can be reduced using a frequent weight set (from 1512 to 8 candidates) or random forest weight prediction.

### Loss & Training

- Fine-tuning is performed on a WebLI-pretrained text-to-image LDM (865M parameters) using the DDPM objective.
- Multi-task training sampling probabilities: SR 0.32, motion deblurring 0.28, defocus deblurring 0.18, denoising 0.22.
- Image conditioning and text conditioning are each randomly dropped with probability 0.1 (supporting classifier-free guidance and blind restoration).
- Training is conducted with JAX on 32 TPU-v5 chips for 200K steps, with batch size 256 and learning rate 8e-5.
- Outputs undergo AdaIN color correction.

## Key Experimental Results

### Main Results (DiversePhotos×1, complex degradations, 160 images)

| Method | ClipIQA ↑ | MUSIQ ↑ | ManIQA ↑ |
|--------|----------|---------|----------|
| StableSR | 0.6227 | 61.39 | 0.3992 |
| DiffBIR | 0.6453 | 59.97 | 0.4922 |
| SUPIR | 0.5060 | 51.68 | 0.3745 |
| DACLIP-IR | 0.3497 | 46.16 | 0.2567 |
| **UniRes** | **0.6519** | **68.22** | **0.5021** |

UniRes surpasses the second-best method (StableSR) by **6.83 points** on MUSIQ, demonstrating substantially greater robustness to complex degradations.

### Ablation Study (DiversePhotos×1)

| Ablation Setting | ClipIQA | MUSIQ | ManIQA | Notes |
|-----------------|---------|-------|--------|-------|
| UniRes (default) | 0.6519 | 68.22 | 0.5021 | Full model |
| SR training only | 0.4173 | 47.76 | 0.2921 | Single task insufficient |
| Single-task inference (SR) | 0.4640 | 53.54 | 0.3423 | Single expert insufficient |
| Single-task inference (DN) | 0.3744 | 39.21 | 0.2202 | Denoising least generalizable |
| DownLQ ×2 only | 0.4883 | 55.38 | 0.3480 | Insufficient detail |
| Without SR | 0.5366 | 59.70 | 0.3959 | All tasks contribute |
| Without MD | 0.5595 | 61.05 | 0.4273 | Motion deblurring is important |
| Without DD | 0.5441 | 60.63 | 0.4075 | Defocus deblurring is important |
| Search range [0,1] | 0.5667 | 63.24 | 0.4154 | Negative weights are beneficial |
| Frequent weight set (8 groups) | 0.6613 | 68.02 | 0.5101 | Greatly reduced search cost |
| Random forest prediction | 0.5873 | 61.91 | 0.4257 | Search can be bypassed |

### Key Findings

- **Necessity of multi-task training**: A model trained on SR alone achieves only 47.76 MUSIQ on complex degradations (vs. 68.22), indicating a substantial performance gap.
- **Each expert is indispensable**: Removing any single task leads to a performance drop, confirming that all tasks contribute to complex degradation restoration.
- **Value of negative weights**: Extending the search range to $[-0.2, 1.2]$ versus $[0, 1]$ yields approximately 5 additional MUSIQ points; negative weights function as a repulsion mechanism analogous to classifier-free guidance.
- **Manageable search complexity**: Using only 8 frequent weight candidates achieves performance close to full grid search.
- Competitive performance is maintained on single-degradation benchmarks such as Real60, confirming that single-task performance is not sacrificed.

## Highlights & Insights

- **Elegant MoE formulation**: Text prompts serve as "expert selectors" and weight composition as the "routing strategy," yielding a framework that resembles an end-to-end differentiable Mixture-of-Experts system.
- **Concatenation conditioning vs. adapters**: The paper revisits a technically underexplored alternative (latent space concatenation) and demonstrates its superiority in terms of fidelity.
- **DownLQ mechanism**: Deliberately degrading the input to encourage detail generation is counterintuitive yet effective.
- **DiversePhotos benchmark**: Addresses the gap in complex degradation benchmarking; each image contains at least two authentic degradation types.
- **Extensible unified formulation**: The framework defined in Eq. 2 is sufficiently flexible to accommodate new restoration tasks or manipulation of existing ones.

## Limitations & Future Work

- **High inference cost**: Each weight combination requires $K$ forward passes (6 experts = 6 passes); grid search further amplifies inference time by orders of magnitude.
- **Limited to camera-related degradations**: Adverse weather conditions (rain, snow, haze) are not covered.
- **Bias of MUSIQ as optimization target**: The inherent bias of the image quality metric may skew the weight optimization direction.
- **No full-reference evaluation of complex degradations**: DiversePhotos lacks HQ reference images, limiting evaluation to no-reference metrics.
- **No public code release**.
- Color correction relies on post-processing (AdaIN) rather than end-to-end training with the model.

## Related Work & Insights

- **Relation to Mixture of Experts (MoE)**: UniRes is essentially an "inference-time MoE," where experts share parameters but activate distinct knowledge through different text prompts.
- **Connection to Diffusion Soup (model merging)**: Predictions are composed in latent space rather than merging weights in parameter space.
- **ControlNet inconsistency** motivated the adoption of a simpler concatenation conditioning scheme.
- **Implications for future work**: Degradation-aware feature learning could replace grid search for more efficient weight prediction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The weighted composition of latent-space predictions is concise and elegant; the DownLQ mechanism is creative; however, the overall approach builds upon a well-established LDM framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Ablation studies are comprehensive, yet full-reference metric validation and inference efficiency analysis are absent; the DiversePhotos benchmark is valuable but small in scale.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The paper is clearly written, with well-articulated motivation and concise mathematical formulation.
- **Value**: ⭐⭐⭐⭐ Provides an effective solution for real-world complex degradation restoration; DiversePhotos fills a benchmark gap; however, high inference cost limits practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MP-HSIR: A Multi-Prompt Framework for Universal Hyperspectral Image Restoration](mp-hsir_a_multi-prompt_framework_for_universal_hyperspectral_image_restoration.md)
- [\[ICCV 2025\] Towards a Universal Image Degradation Model via Content-Degradation Disentanglement](towards_a_universal_image_degradation_model_via_content-degradation_disentanglem.md)
- [\[ICCV 2025\] Enhancing Image Restoration Transformer via Adaptive Translation Equivariance](enhancing_image_restoration_transformer_via_adaptive_translation_equivariance.md)
- [\[ICCV 2025\] Exploiting Diffusion Prior for Task-driven Image Restoration](exploiting_diffusion_prior_for_task-driven_image_restoration.md)
- [\[ICCV 2025\] EAMamba: Efficient All-Around Vision State Space Model for Image Restoration](eamamba_efficient_all-around_vision_state_space_model_for_image_restoration.md)

</div>

<!-- RELATED:END -->
